"""Conservative v22.05 efficiency and metric-overhead helpers."""

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


def classify_efficiency_v22_05(row: dict[str, Any]) -> dict[str, Any]:
    carrier = str(row.get("carrier", ""))
    forward = _f(row.get("forward_ratio_vs_mlp", row.get("best_forward_ratio")))
    step = _f(row.get("step_ratio_vs_mlp", row.get("best_step_ratio")))
    memory = _f(row.get("memory_ratio_vs_mlp", row.get("best_memory_ratio")), 1.0)
    same_kernel = int(_f(row.get("same_kernel_functional_runner_proof", row.get("v22_04_same_kernel_runner_proof")), 0.0))
    fallback = int(_f(row.get("fallback_kernel_used"), 0.0))
    official = int(_f(row.get("official_fused_kernel_complete", row.get("full_loop_official_closure")), 0.0))
    functional_overhead = _f(row.get("functional_overhead_ratio"), 1.0)
    pass_s1 = int(
        official
        and same_kernel
        and not fallback
        and (not isfinite(forward) or forward <= 1.25)
        and (not isfinite(step) or step <= 1.25)
        and (not isfinite(memory) or memory <= 1.05)
        and (not isfinite(functional_overhead) or functional_overhead <= 1.10)
    )
    blocker = []
    if not official:
        blocker.append("official_fused_kernel_incomplete")
    if not same_kernel:
        blocker.append("same_kernel_runner_proof_missing")
    if fallback:
        blocker.append("fallback_kernel_used")
    if isfinite(forward) and forward > 1.25:
        blocker.append("forward_ratio")
    if isfinite(step) and step > 1.25:
        blocker.append("step_ratio")
    if isfinite(memory) and memory > 1.05:
        blocker.append("memory_ratio")
    if isfinite(functional_overhead) and functional_overhead > 1.10:
        blocker.append("functional_overhead")
    return {
        "carrier": carrier,
        "v22_05_S1_pass": pass_s1,
        "v22_05_decision": "OfficialEfficientCarrier" if pass_s1 else "EfficiencyBlocked",
        "v22_05_blocker": ";".join(blocker),
    }


def classify_drat_drbf_v22_05(row: dict[str, Any]) -> dict[str, Any]:
    forward = _f(row.get("forward_ratio_vs_mlp"))
    step = _f(row.get("step_ratio_vs_mlp"))
    memory = _f(row.get("memory_ratio_vs_mlp"), 99.0)
    grad = int(_f(row.get("gradcheck_pass"), 0.0))
    official = int(_f(row.get("official_fused_kernel_complete"), 0.0))
    near = int(forward <= 3.0 and step <= 2.0 and memory <= 1.2 and grad)
    e1 = int(forward <= 1.75 and step <= 1.50 and memory <= 1.10 and official)
    return {
        "micro_near_E1": near,
        "E1_official": e1,
        "drat_drbf_blocker": "" if e1 else ("official_fused_missing" if near and not official else "near_E1_missing"),
    }


def efficiency_v22_05_unit_tests() -> list[dict[str, Any]]:
    rows = []
    pass_row = classify_efficiency_v22_05(
        {
            "carrier": "D-CHE",
            "forward_ratio_vs_mlp": 1.1,
            "step_ratio_vs_mlp": 1.2,
            "memory_ratio_vs_mlp": 1.01,
            "same_kernel_functional_runner_proof": 1,
            "official_fused_kernel_complete": 1,
            "functional_overhead_ratio": 1.02,
        }
    )
    rows.append({"case": "dche_pass", "expected": 1, "actual": pass_row["v22_05_S1_pass"], "pass": int(pass_row["v22_05_S1_pass"] == 1)})
    fail_row = classify_efficiency_v22_05(
        {
            "carrier": "D-CHE",
            "forward_ratio_vs_mlp": 1.3,
            "step_ratio_vs_mlp": 1.0,
            "memory_ratio_vs_mlp": 1.0,
            "same_kernel_functional_runner_proof": 1,
            "official_fused_kernel_complete": 1,
        }
    )
    rows.append({"case": "ratio_block", "expected": 0, "actual": fail_row["v22_05_S1_pass"], "pass": int(fail_row["v22_05_S1_pass"] == 0)})
    near = classify_drat_drbf_v22_05({"forward_ratio_vs_mlp": 2.0, "step_ratio_vs_mlp": 1.9, "memory_ratio_vs_mlp": 1.1, "gradcheck_pass": 1})
    rows.append({"case": "drat_near", "expected": 1, "actual": near["micro_near_E1"], "pass": int(near["micro_near_E1"] == 1)})
    return rows


__all__ = ["classify_drat_drbf_v22_05", "classify_efficiency_v22_05", "efficiency_v22_05_unit_tests"]
