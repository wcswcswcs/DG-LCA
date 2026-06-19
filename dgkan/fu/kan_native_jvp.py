"""KAN native JVP/VJP summary helpers for v22.18."""

from __future__ import annotations

from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in {float("inf"), float("-inf")}:
        return default
    return out


def present_nonnegative(row: dict[str, Any], field: str) -> bool:
    value = row.get(field)
    if value in {"", None}:
        return False
    return safe_float(value) >= 0.0


def present(row: dict[str, Any], field: str) -> bool:
    return row.get(field) not in {"", None}


def summarize_native_ladder(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    models = sorted({str(r.get("model_name", "")) for r in rows if str(r.get("model_name", "")).endswith("_FU_NATIVE_JVP_REFRESH")})
    for model_name in models:
        vals = [r for r in rows if r.get("model_name") == model_name]
        n = len(vals)
        official_ratio_vals = [
            safe_float(r.get("official_full_loop_ratio_vs_mlp"), safe_float(r.get("full_loop_ratio_vs_mlp"), 999.0)) for r in vals
        ]
        official_overhead_vals = [
            safe_float(r.get("official_controller_overhead_ratio"), safe_float(r.get("controller_overhead_ratio"), 999.0)) for r in vals
        ]
        out.append(
            {
                "model_name": model_name,
                "rows": n,
                "basis_energy_ge_050": sum(safe_float(r.get("basis_channel_energy_fraction")) >= 0.50 for r in vals),
                "source_loss_horizon_ge_0": sum(present_nonnegative(r, "KAN_source_loss_horizon") for r in vals),
                "source_loss_h3200_ge_0": sum(present_nonnegative(r, "KAN_source_loss_h3200") for r in vals),
                "source_loss_h4800_ge_0": sum(present_nonnegative(r, "KAN_source_loss_h4800") for r in vals),
                "nll_improve_vs_kan": sum(safe_float(r.get("NLL_delta_vs_KAN_AdamW_native")) < -0.01 for r in vals),
                "auc_improve_vs_kan": sum(safe_float(r.get("AUC_loss_time_delta_vs_KAN_AdamW_native")) < 0.0 for r in vals),
                "controller_overhead_le_025": sum(safe_float(r.get("controller_overhead_ratio")) <= 0.25 for r in vals),
                "full_loop_ratio_le_3": sum(safe_float(r.get("full_loop_ratio_vs_mlp"), 999.0) <= 3.0 for r in vals),
                "official_controller_overhead_le_025": sum(v <= 0.25 for v in official_overhead_vals),
                "official_full_loop_ratio_le_3": sum(v <= 3.0 for v in official_ratio_vals),
                "diagnostic_separated_timing_rows": sum(present(r, "official_full_loop_ratio_vs_mlp") and present(r, "official_controller_overhead_ratio") for r in vals),
                "uses_dense_output_jacobian_any": sum(int(float(r.get("uses_dense_output_jacobian_official") or 0)) for r in vals),
                "max_full_loop_ratio": max((safe_float(r.get("full_loop_ratio_vs_mlp")) for r in vals), default=0.0),
                "max_official_full_loop_ratio": max(official_ratio_vals, default=0.0),
                "max_controller_overhead": max((safe_float(r.get("controller_overhead_ratio")) for r in vals), default=0.0),
                "max_official_controller_overhead": max(official_overhead_vals, default=0.0),
                "mean_diagnostic_source_eval_step_ms": sum(safe_float(r.get("diagnostic_source_eval_step_ms")) for r in vals) / max(1, n),
            }
        )
    return out
