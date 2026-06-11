"""Function-space actuation readback helpers for v20.

The routines here summarize diagnostics emitted by FU mechanisms such as
M20-M28.  They never inspect validation/test labels and never create update
directions.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from dgkan.fu.source_channel import finite_float, spearman


def actuation_rows_from_traces(traces: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for tr in traces:
        r2 = finite_float(tr.get("ActuationR2"))
        if not math.isfinite(r2):
            continue
        rows.append(
            {
                "run_label": tr.get("run_label", ""),
                "job_index": tr.get("job_index", ""),
                "carrier": tr.get("carrier", ""),
                "basis_repair_variant": tr.get("basis_repair_variant", ""),
                "continuation_id": tr.get("continuation_id", ""),
                "mechanism": tr.get("mechanism", ""),
                "step": tr.get("step", ""),
                "ActuationR2": r2,
                "ActuationCosine": tr.get("ActuationCosine", ""),
                "ActuationResidualNorm": tr.get("projection_residual_norm", ""),
                "B1_gain": tr.get("B1_gain", ""),
                "B2_transfer_gain": tr.get("B2_transfer_gain", ""),
                "B3_safety_gain": tr.get("B3_safety_gain", ""),
                "operator_gate_accept": tr.get("operator_gate_accept", ""),
                "source_state_gate_accept": tr.get("source_state_gate_accept", ""),
                "generalization_gate_accept": tr.get("generalization_gate_accept", ""),
            }
        )
    return rows


def actuation_summary(actuation_rows: Iterable[dict[str, Any]], source_rows: Iterable[dict[str, Any]] = ()) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in actuation_rows:
        key = (str(row.get("carrier")), str(row.get("basis_repair_variant")), str(row.get("continuation_id")))
        grouped.setdefault(key, []).append(row)

    source_by_key = {
        (str(r.get("carrier")), str(r.get("basis_repair_variant")), str(r.get("continuation_id"))): r
        for r in source_rows
    }
    out = []
    for (carrier, variant, cid), group in sorted(grouped.items()):
        vals = [finite_float(r.get("ActuationR2")) for r in group]
        vals = [v for v in vals if math.isfinite(v)]
        residuals = [finite_float(r.get("ActuationResidualNorm")) for r in group]
        residuals = [v for v in residuals if math.isfinite(v)]
        src = source_by_key.get((carrier, variant, cid), {})
        item = {
            "carrier": carrier,
            "basis_repair_variant": variant,
            "continuation_id": cid,
            "mechanism": group[0].get("mechanism", ""),
            "actuation_trace_rows": len(group),
            "ActuationR2_min": min(vals) if vals else "",
            "ActuationR2_max": max(vals) if vals else "",
            "ActuationR2_mean": sum(vals) / len(vals) if vals else "",
            "ActuationResidualNorm_mean": sum(residuals) / len(residuals) if residuals else "",
            "source_h800_mean": src.get("source_h800_mean", ""),
            "source_h1600_mean": src.get("source_h1600_mean", ""),
            "source_h3200_mean": src.get("source_h3200_mean", ""),
            "source_h4800_mean": src.get("source_h4800_mean", ""),
            "high_actuation": int(vals and max(vals) >= 0.70),
            "source_success": int(finite_float(src.get("source_h1600_mean"), -999.0) >= 0.005),
        }
        item["actuation_r2_source_spearman_hint"] = spearman(
            [r.get("ActuationR2") for r in group],
            [src.get("source_h1600_mean") for _ in group],
        )
        out.append(item)
    return out
