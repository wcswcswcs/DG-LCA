"""Conservative v22.04 efficiency route helpers."""

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


def classify_efficiency_v22_04(row: dict[str, Any]) -> dict[str, Any]:
    carrier = str(row.get("carrier", ""))
    closure = int(_f(row.get("full_loop_official_closure"), 0.0))
    same_kernel = int(_f(row.get("v22_04_same_kernel_runner_proof", row.get("v22_03_same_kernel_functional_runner_proof")), 0.0))
    forward = _f(row.get("best_forward_ratio"))
    step = _f(row.get("best_step_ratio"))
    memory = _f(row.get("best_memory_ratio"), 1.0)
    pass_s1 = int(
        closure
        and same_kernel
        and (not isfinite(forward) or forward <= 1.25)
        and (not isfinite(step) or step <= 1.25)
        and (not isfinite(memory) or memory <= 1.05)
    )
    blocker = []
    if not closure:
        blocker.append("official_closure_missing")
    if not same_kernel:
        blocker.append("same_kernel_runner_proof_missing")
    if isfinite(forward) and forward > 1.25:
        blocker.append("forward_ratio")
    if isfinite(step) and step > 1.25:
        blocker.append("step_ratio")
    if isfinite(memory) and memory > 1.05:
        blocker.append("memory_ratio")
    return {
        "carrier": carrier,
        "v22_04_S1_pass": pass_s1,
        "v22_04_decision": "OfficialEfficientCarrier" if pass_s1 else "EfficiencyBlocked",
        "v22_04_blocker": ";".join(blocker),
    }


def efficiency_v22_04_unit_tests() -> list[dict[str, Any]]:
    rows = [
        ("pass", {"carrier": "D-CHE", "full_loop_official_closure": 1, "v22_04_same_kernel_runner_proof": 1}, 1),
        ("closure_missing", {"carrier": "D-RAT", "full_loop_official_closure": 0, "v22_04_same_kernel_runner_proof": 1}, 0),
        ("ratio_block", {"carrier": "D-X", "full_loop_official_closure": 1, "v22_04_same_kernel_runner_proof": 1, "best_forward_ratio": 1.4}, 0),
    ]
    out = []
    for name, row, expected in rows:
        got = classify_efficiency_v22_04(row)
        out.append({"case": name, "expected_pass": expected, "actual_pass": got["v22_04_S1_pass"], "pass": int(got["v22_04_S1_pass"] == expected)})
    return out


__all__ = ["classify_efficiency_v22_04", "efficiency_v22_04_unit_tests"]
