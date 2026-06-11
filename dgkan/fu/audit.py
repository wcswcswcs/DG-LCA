"""FU audit row helpers."""

from __future__ import annotations

from typing import Any


FORBIDDEN_DIRECTION_SOURCES = {
    "validation",
    "test",
    "future",
    "query",
    "LineC",
    "CEp99",
    "NLL",
    "ECE",
    "AUCtime",
    "BrierAudit",
}


def direction_provenance_row(
    carrier: str,
    mechanism: str,
    source: str,
    *,
    uses_validation_test_future_query: int = 0,
    audit_metrics_used_for_direction: int = 0,
) -> dict[str, Any]:
    return {
        "carrier": carrier,
        "mechanism": mechanism,
        "direction_source": source,
        "uses_validation_test_future_query": int(uses_validation_test_future_query),
        "audit_metrics_used_for_direction": int(audit_metrics_used_for_direction),
        "forbidden_source_violation": int(
            uses_validation_test_future_query or audit_metrics_used_for_direction or any(token in source for token in FORBIDDEN_DIRECTION_SOURCES)
        ),
    }
