"""v22.06 efficiency classifiers.

The thresholds mirror the v22.06 plan and wrap the measured profiler rows.  No
numbers are synthesized here; callers must provide measured timing rows.
"""

from __future__ import annotations

from math import isfinite
from typing import Any


def _f(value: Any, default: float = float("nan")) -> float:
    try:
        if value in {"", None}:
            return default
        out = float(value)
    except Exception:
        return default
    return out if isfinite(out) else default


def classify_carrier_efficiency_v22_06(row: dict[str, Any]) -> dict[str, Any]:
    forward = _f(row.get("forward_ratio_vs_mlp", row.get("best_forward_ratio")))
    step = _f(row.get("step_ratio_vs_mlp", row.get("best_step_ratio")))
    memory = _f(row.get("memory_ratio_vs_mlp", row.get("best_memory_ratio")), 99.0)
    same_kernel = int(_f(row.get("same_kernel_functional_runner_proof", row.get("same_kernel_runner_proof")), 0.0))
    fallback = int(_f(row.get("fallback_kernel_used"), 0.0))
    official = int(_f(row.get("official_fused_kernel_complete", row.get("full_loop_official_closure")), 0.0))
    pass_s1 = int(official and same_kernel and not fallback and forward <= 1.25 and step <= 1.25 and memory <= 1.05)
    blockers = []
    if not official:
        blockers.append("official_fused_kernel_incomplete")
    if not same_kernel:
        blockers.append("same_kernel_runner_proof_missing")
    if fallback:
        blockers.append("fallback_kernel_used")
    if forward > 1.25:
        blockers.append("forward_ratio")
    if step > 1.25:
        blockers.append("step_ratio")
    if memory > 1.05:
        blockers.append("memory_ratio")
    return {
        "v22_06_S1_pass": pass_s1,
        "v22_06_decision": "OfficialEfficientCarrier" if pass_s1 else "EfficiencyBlocked",
        "v22_06_blocker": ";".join(blockers),
    }


def classify_drat_drbf_v22_06(row: dict[str, Any]) -> dict[str, Any]:
    forward = _f(row.get("forward_ratio_vs_mlp"))
    step = _f(row.get("step_ratio_vs_mlp"))
    memory = _f(row.get("memory_ratio_vs_mlp"), 99.0)
    grad = int(_f(row.get("gradcheck_pass"), 0.0))
    official = int(_f(row.get("official_fused_kernel_complete"), 0.0))
    den_safe = int(_f(row.get("den_safety_pass", 1.0), 1.0))
    near = int(forward <= 3.0 and step <= 2.0 and memory <= 1.2 and grad)
    official_pass = int(forward <= 1.5 and step <= 1.5 and memory <= 1.1 and grad and official and den_safe)
    return {
        "near_E1": near,
        "production_fused_official_pass": official_pass,
        "v22_06_official_decision": "ProductionFusedPass" if official_pass else ("OfficialFusedBlocked" if near else "RejectedForThisVersion"),
        "v22_06_official_blocker": "" if official_pass else ("official_fused_missing_or_ratio" if near else "near_E1_missing"),
    }


def efficiency_v22_06_unit_tests() -> list[dict[str, Any]]:
    ok = classify_carrier_efficiency_v22_06(
        {
            "forward_ratio_vs_mlp": 1.1,
            "step_ratio_vs_mlp": 1.2,
            "memory_ratio_vs_mlp": 1.03,
            "same_kernel_runner_proof": 1,
            "official_fused_kernel_complete": 1,
        }
    )
    bad = classify_drat_drbf_v22_06(
        {
            "forward_ratio_vs_mlp": 2.0,
            "step_ratio_vs_mlp": 1.2,
            "memory_ratio_vs_mlp": 1.0,
            "gradcheck_pass": 1,
            "official_fused_kernel_complete": 0,
        }
    )
    return [
        {"case": "carrier_pass", "actual": ok["v22_06_S1_pass"], "expected": 1, "pass": int(ok["v22_06_S1_pass"] == 1)},
        {"case": "near_but_not_official", "actual": bad["near_E1"], "expected": 1, "pass": int(bad["near_E1"] == 1 and bad["production_fused_official_pass"] == 0)},
    ]


__all__ = ["classify_carrier_efficiency_v22_06", "classify_drat_drbf_v22_06", "efficiency_v22_06_unit_tests"]
