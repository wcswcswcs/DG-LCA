"""v20 official-ish efficiency gate helpers."""

from __future__ import annotations

from typing import Any

from dgkan.profiling.efficiency_v17 import profile_isolated


def v20_officialish_gate(row: dict[str, Any]) -> dict[str, Any]:
    forward = float(row.get("forward_ratio", row.get("forward_ratio_vs_mlp", 999.0)) or 999.0)
    step = float(row.get("training_step_ratio", row.get("step_ratio_vs_mlp", 999.0)) or 999.0)
    memory = float(row.get("memory_ratio", row.get("memory_ratio_vs_mlp", 999.0)) or 999.0)
    audit_raw = float(row.get("audit_overhead_ratio", 999.0) or 999.0)
    audit_separated = int(row.get("audit_cost_separated", 0) or 0)
    # v20 S1 requires LineC/horizon audit cost to be separated from the
    # training loop. The raw audit readback is still reported, but it should
    # not fail full-loop timing once it is explicitly out of loop.
    audit = 0.0 if audit_separated else audit_raw
    gradcheck = int(row.get("gradcheck_pass", row.get("manual_correctness_pass", 0)) or 0)
    no_materialize = int(row.get("no_materialize_complete", row.get("no_materialize_kernel_complete", 0)) or 0)
    full_loop = int(row.get("full_loop_timing_pass", row.get("phase_level_timing_present", 0)) or 0)
    official_fused = int(row.get("official_fused_kernel_complete", 0) or 0)
    passed = int(
        forward <= 1.25
        and step <= 1.25
        and (memory <= 1.05 or memory == 0.0)
        and audit <= 0.15
        and gradcheck
        and no_materialize
        and full_loop
        and official_fused
    )
    reasons = []
    if forward > 1.25:
        reasons.append("ForwardOfficialBlocked")
    if step > 1.25:
        reasons.append("StepOfficialBlocked")
    if memory > 1.05 and memory != 0.0:
        reasons.append("MemoryOfficialBlocked")
    if audit > 0.15:
        reasons.append("AuditOverheadBlocked")
    if not gradcheck:
        reasons.append("GradcheckBlocked")
    if not no_materialize:
        reasons.append("DenseMaterializationBlocked")
    if not full_loop:
        reasons.append("FullLoopTimingMissing")
    if not official_fused:
        reasons.append("OfficialFusedKernelMissing")
    return {
        "audit_overhead_ratio_raw": audit_raw,
        "audit_overhead_ratio_training_loop": audit,
        "v20_officialish_gate": passed,
        "v20_officialish_blocker": "pass" if passed else ";".join(reasons),
    }


def efficiency_waterfall_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    phases = [
        "forward_only_ms",
        "basis_eval_ms",
        "readout_contraction_ms",
        "backward_input_ms",
        "backward_param_ms",
        "optimizer_update_ms_SGD",
        "optimizer_update_ms_AdamW",
        "optimizer_update_ms_manualFU",
        "functional_direction_ms",
        "functional_commit_ms",
        "linec_audit_ms",
        "step_training_only_ms",
    ]
    return [
        {
            "carrier": row.get("carrier", ""),
            "kernel_variant": row.get("kernel_variant", row.get("repair_variant", "")),
            "batch_size": row.get("batch_size", ""),
            "phase": phase,
            "ms": row.get(phase, ""),
            "measurement_status": "measured" if row.get(phase, "") != "" else "not_isolated",
        }
        for phase in phases
    ]


__all__ = ["profile_isolated", "v20_officialish_gate", "efficiency_waterfall_rows"]
