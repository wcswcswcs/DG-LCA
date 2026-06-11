"""v22.01 efficiency gates for official and active-repair carrier paths."""

from __future__ import annotations

from typing import Any

from dgkan.fu.source_chain import finite_float


def int_flag(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def v22_01_efficiency_gate(row: dict[str, Any]) -> dict[str, int | str]:
    fwd = finite_float(row.get("forward_ratio_vs_mlp"), 999.0)
    step = finite_float(row.get("step_ratio_vs_mlp"), 999.0)
    mem = finite_float(row.get("memory_ratio_vs_mlp"), 999.0)
    grad = int_flag(row.get("gradcheck_pass"))
    fallback = int_flag(row.get("fallback_kernel_used"))
    official = int_flag(row.get("official_fused_kernel_complete"))
    no_mat = int_flag(row.get("no_materialize_complete"))
    safety = int_flag(row.get("denominator_safety_pass", row.get("width_safety_pass", 1)))
    telemetry_free = int_flag(row.get("training_path_telemetry_free", 1))
    basis_reduced = finite_float(row.get("basis_materialized_reduction_fraction"), 1.0)
    active_frac = finite_float(row.get("active_center_fraction"), 0.2)
    e1 = int(fwd <= 1.25 and step <= 1.25 and mem <= 1.10 and grad and not fallback)
    s1 = int(e1 and official and no_mat)
    near_rat = int(fwd <= 3.0 and step <= 2.0 and mem <= 1.20 and grad and safety and telemetry_free)
    near_rbf = int(fwd <= 3.0 and step <= 2.0 and mem <= 1.20 and grad and basis_reduced >= 0.70 and 0.05 <= active_frac <= 0.60)
    blockers: list[str] = []
    if fwd > 1.25:
        blockers.append("forward_ratio")
    if step > 1.25:
        blockers.append("step_ratio")
    if mem > 1.10:
        blockers.append("memory_ratio")
    if not grad:
        blockers.append("gradcheck")
    if fallback:
        blockers.append("fallback_kernel")
    if not official:
        blockers.append("official_fused_missing")
    if not no_mat:
        blockers.append("materialization")
    return {
        "v22_01_E1_exploration_gate": e1,
        "v22_01_S1_official_like_gate": s1,
        "v22_01_NearE1_RAT_gate": near_rat,
        "v22_01_NearE1_RBF_gate": near_rbf,
        "v22_01_efficiency_blocker": "pass" if s1 else ";".join(blockers),
    }


__all__ = ["v22_01_efficiency_gate"]
