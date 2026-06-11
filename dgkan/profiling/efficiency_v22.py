"""v22 full-loop efficiency officialization gates."""

from __future__ import annotations

from typing import Any

from dgkan.profiling.efficiency_v20 import efficiency_waterfall_rows, profile_isolated
from dgkan.profiling.efficiency_v21 import finite_float


def v22_efficiency_gate(row: dict[str, Any]) -> dict[str, Any]:
    forward = finite_float(row.get("forward_ratio_vs_mlp", row.get("forward_ratio")))
    backward = finite_float(row.get("backward_ratio_vs_mlp", row.get("backward_ratio")))
    step = finite_float(row.get("step_ratio_vs_mlp", row.get("training_step_ratio")))
    memory = finite_float(row.get("memory_ratio_vs_mlp", row.get("memory_ratio")))
    if row.get("full_loop_matches_profiler_ratio") not in {"", None}:
        full_loop = finite_float(row.get("full_loop_matches_profiler_ratio"), 999.0)
    elif int(row.get("full_loop_timing_pass", row.get("phase_level_timing_present", 0)) or 0):
        full_loop = 1.0
    else:
        full_loop = 999.0
    grad_relerr = finite_float(row.get("manual_grad_relerr_max"), 999.0)
    no_materialize = int(row.get("no_materialize_complete", 0) or 0)
    official_fused = int(row.get("official_fused_kernel_complete", 0) or 0)
    e1 = int(
        forward <= 1.25
        and backward <= 1.30
        and step <= 1.25
        and (memory <= 1.10 or memory == 0.0)
        and full_loop <= 1.10
        and grad_relerr <= 1.0e-4
        and no_materialize
    )
    s1 = int(
        e1
        and forward <= 1.15
        and backward <= 1.20
        and step <= 1.15
        and (memory <= 1.05 or memory == 0.0)
        and official_fused
    )
    e2 = int(s1 and forward <= 1.10 and backward <= 1.15 and step <= 1.10)
    blockers: list[str] = []
    if forward > 1.25:
        blockers.append("ForwardFullLoopBlocked")
    if backward > 1.30:
        blockers.append("BackwardFullLoopBlocked")
    if step > 1.25:
        blockers.append("StepFullLoopBlocked")
    if memory > 1.10 and memory != 0.0:
        blockers.append("MemoryFullLoopBlocked")
    if full_loop > 1.10:
        blockers.append("ProfilerMismatchBlocked")
    if grad_relerr > 1.0e-4:
        blockers.append("GradcheckBlocked")
    if not no_materialize:
        blockers.append("DenseMaterializationBlocked")
    if not official_fused:
        blockers.append("OfficialFusedKernelMissing")
    return {
        "v22_E1_exploration_gate": e1,
        "v22_S1_official_like_gate": s1,
        "v22_E2_margin_gate": e2,
        "v22_efficiency_blocker": "pass" if e1 else ";".join(blockers),
    }


__all__ = ["profile_isolated", "efficiency_waterfall_rows", "v22_efficiency_gate"]
