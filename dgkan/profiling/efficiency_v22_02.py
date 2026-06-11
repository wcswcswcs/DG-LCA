"""v22.02 efficiency gates for full-loop officialization and active repair."""

from __future__ import annotations

from typing import Any

from dgkan.fu.source_chain import finite_float


def int_flag(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def v22_02_efficiency_gate(row: dict[str, Any]) -> dict[str, Any]:
    fwd = finite_float(row.get("forward_ratio_vs_mlp"), 999.0)
    step = finite_float(row.get("step_ratio_vs_mlp"), 999.0)
    mem = finite_float(row.get("memory_ratio_vs_mlp"), 999.0)
    grad = int_flag(row.get("gradcheck_pass"))
    full_loop = int_flag(row.get("full_loop_timing_pass"))
    official = int_flag(row.get("official_fused_kernel_complete"))
    no_mat = int_flag(row.get("no_materialize_complete"))
    same_kernel = int_flag(row.get("functional_runner_uses_same_kernel"))
    fallback = int_flag(row.get("fallback_kernel_used"))
    safety = int_flag(row.get("safety_pass", row.get("denominator_safety_pass", row.get("width_safety_pass", 1))))
    materialized_bytes = finite_float(row.get("basis_materialized_bytes"), 0.0)
    dense_reference_bytes = finite_float(row.get("dense_reference_bytes"), 0.0)
    basis_materialization_ok = int(dense_reference_bytes <= 0.0 or materialized_bytes <= 0.25 * dense_reference_bytes)

    e1 = int(fwd <= 1.75 and step <= 1.75 and mem <= 1.20 and grad)
    s1 = int(fwd <= 1.25 and step <= 1.25 and mem <= 1.05 and grad and full_loop and official and no_mat and same_kernel and not fallback)
    near_e1 = int(fwd <= 3.0 and step <= 2.0 and mem <= 1.20 and grad and safety)
    near_e1_rbf = int(near_e1 and basis_materialization_ok)

    blockers: list[str] = []
    if fwd > 1.25:
        blockers.append("forward_ratio")
    if step > 1.25:
        blockers.append("step_ratio")
    if mem > 1.05:
        blockers.append("memory_ratio")
    if not grad:
        blockers.append("gradcheck")
    if not full_loop:
        blockers.append("full_loop_timing")
    if not official:
        blockers.append("official_fused_missing")
    if not no_mat:
        blockers.append("materialization")
    if not same_kernel:
        blockers.append("functional_runner_kernel_mismatch")
    if fallback:
        blockers.append("fallback_kernel")
    return {
        "v22_02_E1_exploration_gate": e1,
        "v22_02_S1_official_like_gate": s1,
        "v22_02_NearE1_gate": near_e1,
        "v22_02_NearE1_RBF_gate": near_e1_rbf,
        "v22_02_efficiency_blocker": "pass" if s1 else ";".join(blockers),
    }


def efficiency_v22_02_unit_tests() -> list[dict[str, Any]]:
    good = v22_02_efficiency_gate(
        {
            "forward_ratio_vs_mlp": 1.0,
            "step_ratio_vs_mlp": 1.0,
            "memory_ratio_vs_mlp": 1.0,
            "gradcheck_pass": 1,
            "full_loop_timing_pass": 1,
            "official_fused_kernel_complete": 1,
            "no_materialize_complete": 1,
            "functional_runner_uses_same_kernel": 1,
            "fallback_kernel_used": 0,
        }
    )
    blocked = v22_02_efficiency_gate({"forward_ratio_vs_mlp": 4.0, "gradcheck_pass": 1})
    return [
        {"test": "s1_pass", "expected": 1, "actual": good["v22_02_S1_official_like_gate"], "pass": int(good["v22_02_S1_official_like_gate"] == 1)},
        {"test": "near_e1_blocked_forward", "expected": 0, "actual": blocked["v22_02_NearE1_gate"], "pass": int(blocked["v22_02_NearE1_gate"] == 0)},
    ]


__all__ = ["efficiency_v22_02_unit_tests", "int_flag", "v22_02_efficiency_gate"]
