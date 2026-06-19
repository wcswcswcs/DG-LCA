#!/usr/bin/env python3
"""Summarize runtime P3 controller no-go evidence from real v22.18 runs."""

from __future__ import annotations

from pathlib import Path
import shlex
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_18_common import OUT_ROOT, append_exec, ensure_out, finite_float, read_rows, write_rows  # noqa: E402


SCOPES = [
    {
        "scope": "D1_MNIST_FMNIST_KMNIST_h440",
        "summary": "v22_18_mlp_fu_noharm_improvement_summary_v22_18_runtime_p3_repair2_d1_h440.csv",
        "task": "v22_18_mlp_fu_task_matrix_v22_18_runtime_p3_repair2_d1_h440.csv",
        "source": "v22_17_task_useful_source_matrix_v22_18_runtime_p3_repair2_d1_h440.csv",
    },
    {
        "scope": "Tabular_Spam_Wine_s012_h200_clearproxy",
        "summary": "v22_18_mlp_fu_noharm_improvement_summary_v22_18_runtime_p3_repair2_tabular_spam_wine_s012_h200_clearproxy.csv",
        "task": "v22_18_mlp_fu_task_matrix_v22_18_runtime_p3_repair2_tabular_spam_wine_s012_h200_clearproxy.csv",
        "source": "v22_17_task_useful_source_matrix_v22_18_runtime_p3_repair2_tabular_spam_wine_s012_h200_clearproxy.csv",
    },
    {
        "scope": "HighSignal_Spam_Card_Wine_s012_h200_interval1",
        "summary": "v22_18_mlp_fu_noharm_improvement_summary_v22_18_runtime_p3_tabular_highsignal_s012_h200.csv",
        "task": "v22_18_mlp_fu_task_matrix_v22_18_runtime_p3_tabular_highsignal_s012_h200.csv",
        "source": "v22_17_task_useful_source_matrix_v22_18_runtime_p3_tabular_highsignal_s012_h200.csv",
    },
    {
        "scope": "HighSignal_Spam_Card_Wine_s012_h200_interval10",
        "summary": "v22_18_mlp_fu_noharm_improvement_summary_v22_18_runtime_p3_tabular_highsignal_s012_h200_i10.csv",
        "task": "v22_18_mlp_fu_task_matrix_v22_18_runtime_p3_tabular_highsignal_s012_h200_i10.csv",
        "source": "v22_17_task_useful_source_matrix_v22_18_runtime_p3_tabular_highsignal_s012_h200_i10.csv",
    },
    {
        "scope": "HighSignal_Spam_Card_Wine_s012_h200_interval50_pilot",
        "summary": "v22_18_mlp_fu_noharm_improvement_summary_v22_18_runtime_p3_tabular_highsignal_s012_h200_i50_pilot.csv",
        "task": "v22_18_mlp_fu_task_matrix_v22_18_runtime_p3_tabular_highsignal_s012_h200_i50_pilot.csv",
        "source": "v22_17_task_useful_source_matrix_v22_18_runtime_p3_tabular_highsignal_s012_h200_i50_pilot.csv",
    },
    {
        "scope": "HighSignal_Spam_Card_Wine_s012_h200_interval50_scale05_pilot",
        "summary": "v22_18_mlp_fu_noharm_improvement_summary_v22_18_runtime_p3_tabular_highsignal_s012_h200_i50_strongscale05_pilot.csv",
        "task": "v22_18_mlp_fu_task_matrix_v22_18_runtime_p3_tabular_highsignal_s012_h200_i50_strongscale05_pilot.csv",
        "source": "v22_17_task_useful_source_matrix_v22_18_runtime_p3_tabular_highsignal_s012_h200_i50_strongscale05_pilot.csv",
    },
    {
        "scope": "HighSignal_Spam_Card_Wine_s012_h200_export_policy_interval50_controls",
        "summary": "v22_18_mlp_fu_noharm_improvement_summary_v22_18_runtime_export_policy_highsignal_s012_h200_i50_controls.csv",
        "task": "v22_18_mlp_fu_task_matrix_v22_18_runtime_export_policy_highsignal_s012_h200_i50_controls.csv",
        "source": "v22_17_task_useful_source_matrix_v22_18_runtime_export_policy_highsignal_s012_h200_i50_controls.csv",
    },
    {
        "scope": "HighSignal_Spam_Card_Wine_s012_h200_export_policy_interval100_controls",
        "summary": "v22_18_mlp_fu_noharm_improvement_summary_v22_18_runtime_export_policy_highsignal_s012_h200_i100_controls.csv",
        "task": "v22_18_mlp_fu_task_matrix_v22_18_runtime_export_policy_highsignal_s012_h200_i100_controls.csv",
        "source": "v22_17_task_useful_source_matrix_v22_18_runtime_export_policy_highsignal_s012_h200_i100_controls.csv",
    },
    {
        "scope": "HighSignal_Spam_Card_Wine_s012_h200_export_policy_split_controls",
        "summary": "v22_18_mlp_fu_noharm_improvement_summary_v22_18_runtime_export_policy_split_highsignal_s012_h200_toler10_controls.csv",
        "task": "v22_18_mlp_fu_task_matrix_v22_18_runtime_export_policy_split_highsignal_s012_h200_toler10_controls.csv",
        "source": "v22_17_task_useful_source_matrix_v22_18_runtime_export_policy_split_highsignal_s012_h200_toler10_controls.csv",
    },
    {
        "scope": "HighSignal_Spam_Card_Wine_s012_h200_export_policy_split_thr055_controls",
        "summary": "v22_18_mlp_fu_noharm_improvement_summary_v22_18_runtime_export_policy_split_highsignal_s012_h200_toler10_thr055_controls.csv",
        "task": "v22_18_mlp_fu_task_matrix_v22_18_runtime_export_policy_split_highsignal_s012_h200_toler10_thr055_controls.csv",
        "source": "v22_17_task_useful_source_matrix_v22_18_runtime_export_policy_split_highsignal_s012_h200_toler10_thr055_controls.csv",
    },
    {
        "scope": "HighSignal_Spam_Card_Wine_s0_h200_export_policy_split_thr055_cap010_pilot",
        "summary": "v22_18_mlp_fu_noharm_improvement_summary_v22_18_runtime_export_policy_split_highsignal_s0_h200_thr055_cap010_pilot.csv",
        "task": "v22_18_mlp_fu_task_matrix_v22_18_runtime_export_policy_split_highsignal_s0_h200_thr055_cap010_pilot.csv",
        "source": "v22_17_task_useful_source_matrix_v22_18_runtime_export_policy_split_highsignal_s0_h200_thr055_cap010_pilot.csv",
    },
    {
        "scope": "HighSignal_Spam_Card_Wine_s0_h200_export_policy_split_thr055_cap020_pilot",
        "summary": "v22_18_mlp_fu_noharm_improvement_summary_v22_18_runtime_export_policy_split_highsignal_s0_h200_thr055_cap020_pilot.csv",
        "task": "v22_18_mlp_fu_task_matrix_v22_18_runtime_export_policy_split_highsignal_s0_h200_thr055_cap020_pilot.csv",
        "source": "v22_17_task_useful_source_matrix_v22_18_runtime_export_policy_split_highsignal_s0_h200_thr055_cap020_pilot.csv",
    },
]


def _mean(rows: list[dict[str, str]], field: str) -> float:
    vals = [finite_float(r.get(field)) for r in rows if r.get(field) not in {"", None}]
    return sum(vals) / max(1, len(vals)) if vals else 0.0


def _rows_for_model(rows: list[dict[str, str]], model: str) -> list[dict[str, str]]:
    return [r for r in rows if r.get("model_name") == model]


def _best(rows: list[dict[str, str]], *, control: bool) -> dict[str, str]:
    candidates = [r for r in rows if int(float(r.get("is_control_variant") or 0)) == int(control)]
    return max(
        candidates,
        key=lambda r: (
            int(float(r.get("NLL_improvement_rows") or 0)),
            int(float(r.get("AUC_improvement_rows") or 0)),
            -finite_float(r.get("max_controller_overhead_ratio"), 999.0),
        ),
        default={},
    )


def _reason(real: dict[str, str], control: dict[str, str], rows_required: int) -> str:
    reasons: list[str] = []
    if int(float(real.get("rows") or 0)) < rows_required:
        reasons.append("row_count_below_official_D_scope")
    if int(float(real.get("NLL_improvement_rows") or 0)) < 5:
        reasons.append("NLL_improvement_rows_lt_5")
    if int(float(real.get("AUC_improvement_rows") or 0)) < 5:
        reasons.append("AUC_improvement_rows_lt_5")
    if int(float(real.get("accuracy_improvement_rows") or 0)) < 4:
        reasons.append("accuracy_improvement_rows_lt_4")
    if finite_float(real.get("max_controller_overhead_ratio"), 999.0) > 0.25:
        reasons.append("controller_overhead_gt_0.25")
    if not control:
        reasons.append("no_control_rows_for_official")
    else:
        if int(float(control.get("NLL_improvement_rows") or 0)) >= int(float(real.get("NLL_improvement_rows") or 0)):
            reasons.append("best_control_NLL_improvement_not_lower")
        if int(float(control.get("AUC_improvement_rows") or 0)) >= int(float(real.get("AUC_improvement_rows") or 0)):
            reasons.append("best_control_AUC_improvement_not_lower")
    if int(float(real.get("D_noharm_pass") or 0)) == 0:
        reasons.append("D_noharm_failed")
    return ";".join(reasons)


def main() -> None:
    ensure_out()
    out_rows: list[dict[str, Any]] = []
    for spec in SCOPES:
        summary = read_rows(OUT_ROOT / spec["summary"])
        task_rows = read_rows(OUT_ROOT / spec["task"])
        source_rows = read_rows(ROOT / "results/v22_17" / spec["source"])
        best_real = _best(summary, control=False)
        best_control = _best(summary, control=True)
        for row in summary:
            model = row.get("model_name", "")
            model_task = _rows_for_model(task_rows, model)
            model_source = _rows_for_model(source_rows, model)
            out_rows.append(
                {
                    "scope": spec["scope"],
                    "model_name": model,
                    "v22_18_variant": row.get("v22_18_variant", ""),
                    "is_control_variant": row.get("is_control_variant", ""),
                    "rows": row.get("rows", ""),
                    "NLL_improvement_rows": row.get("NLL_improvement_rows", ""),
                    "AUC_improvement_rows": row.get("AUC_improvement_rows", ""),
                    "accuracy_improvement_rows": row.get("accuracy_improvement_rows", ""),
                    "mean_NLL_delta": row.get("mean_NLL_delta", ""),
                    "max_tail_q99_delta": row.get("max_tail_q99_delta", ""),
                    "max_controller_overhead_ratio": row.get("max_controller_overhead_ratio", ""),
                    "D_noharm_pass": row.get("D_noharm_pass", ""),
                    "D_improvement_pass": row.get("D_improvement_pass", ""),
                    "mean_gate_accept_count": _mean(model_task, "gate_accept_count"),
                    "mean_intervention_count": _mean(model_task, "intervention_count"),
                    "mean_benefit_p3_score": _mean(model_task, "mean_benefit_p3_score"),
                    "mean_benefit_policy_export_score": _mean(model_task, "mean_benefit_policy_export_score"),
                    "mean_benefit_policy_export_accept_score": _mean(model_task, "mean_benefit_policy_export_accept_score"),
                    "benefit_policy_export_runtime_integrated_rows": sum(int(float(r.get("benefit_policy_export_runtime_integrated") or 0)) for r in model_task),
                    "logged_source_rows": len(model_source),
                    "best_real_variant": best_real.get("v22_18_variant", ""),
                    "best_real_NLL_improvement_rows": best_real.get("NLL_improvement_rows", ""),
                    "best_control_variant": best_control.get("v22_18_variant", ""),
                    "has_control_rows": int(bool(best_control)),
                    "best_control_NLL_improvement_rows": best_control.get("NLL_improvement_rows", ""),
                    "no_go_reason_if_best_real": _reason(best_real, best_control, 9 if spec["scope"].startswith("D1") else 9) if row is best_real else "",
                }
            )
    out = OUT_ROOT / "v22_18_runtime_p3_no_go_audit.csv"
    write_rows(out, out_rows)
    cmd = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
    append_exec(
        cmd,
        task_id="D-runtime-P3-no-go-audit",
        status="pass" if out_rows else "fail",
        exit_code=0 if out_rows else 1,
        files=str(out.relative_to(ROOT)),
        note=f"rows={len(out_rows)}; summarizes runtime P3 vs controls no-go evidence from D1, tabular clear-proxy, and high-signal cadence runs",
    )


if __name__ == "__main__":
    main()
