"""v22.09 efficiency telemetry helpers."""

from __future__ import annotations

from typing import Any

from dgkan.profiling.efficiency_v22_06 import efficiency_v22_06_unit_tests


DRAT_COMPONENT_KEYS = [
    "numerator_eval_ms",
    "denominator_eval_ms",
    "reciprocal_ms",
    "numden_fused_ms",
    "safety_clamp_ms",
    "derivative_telemetry_ms",
    "train_path_without_telemetry_ms",
    "component_sum_vs_total_error",
]

DRBF_COMPONENT_KEYS = [
    "active_center_fraction",
    "mean_local_K",
    "local_gather_ms",
    "exp_eval_ms",
    "local_backward_ms",
    "dense_materialization_bytes",
    "no_dense_materialization_proof",
    "component_sum_vs_total_error",
]


def component_telemetry_complete(row: dict[str, Any]) -> int:
    carrier = str(row.get("carrier", ""))
    keys = DRAT_COMPONENT_KEYS if carrier == "D-RAT" else DRBF_COMPONENT_KEYS if carrier == "D-RBF" else []
    return int(bool(keys) and all(str(row.get(k, "")).strip() for k in keys))


def efficiency_v22_09_unit_tests() -> list[dict[str, Any]]:
    base = efficiency_v22_06_unit_tests()
    drat = {"carrier": "D-RAT", **{k: 1 for k in DRAT_COMPONENT_KEYS}}
    drbf = {"carrier": "D-RBF", **{k: 1 for k in DRBF_COMPONENT_KEYS}}
    return base + [
        {"test": "drat_component_key_count", "pass": int(len(DRAT_COMPONENT_KEYS) == 8)},
        {"test": "drbf_component_key_count", "pass": int(len(DRBF_COMPONENT_KEYS) == 8)},
        {"test": "component_complete", "pass": int(component_telemetry_complete(drat) and component_telemetry_complete(drbf))},
    ]
