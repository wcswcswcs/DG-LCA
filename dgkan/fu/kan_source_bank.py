"""Audit helpers for KAN source-bank energy and retention."""

from __future__ import annotations

from typing import Any

from dgkan.fu.source_chain import finite_float


def source_bank_retention(current_energy: Any, previous_energy: Any) -> float | str:
    prev = finite_float(previous_energy)
    cur = finite_float(current_energy)
    if prev <= 0.0 or cur != cur:
        return ""
    return max(0.0, cur) / prev


def kan_source_bank_row(row: dict[str, Any]) -> dict[str, Any]:
    low_degree = finite_float(row.get("low_degree_source_energy_h3200"), 0.0)
    high_degree = finite_float(row.get("high_degree_reservoir_energy_h3200"), 0.0)
    low_freq = finite_float(row.get("low_frequency_source_energy_h3200"), 0.0)
    high_freq = finite_float(row.get("high_frequency_reservoir_energy_h3200"), 0.0)
    source_energy = max(low_degree, low_freq)
    reservoir_energy = max(high_degree, high_freq)
    total = source_energy + reservoir_energy
    return {
        "low_degree_source_energy": low_degree,
        "high_degree_reservoir_energy": high_degree,
        "low_frequency_source_energy": low_freq,
        "high_frequency_reservoir_energy": high_freq,
        "source_bank_energy": source_energy,
        "reservoir_bank_energy": reservoir_energy,
        "source_bank_fraction": source_energy / total if total > 0.0 else "",
        "source_bank_retention_h4800_over_h3200": source_bank_retention(
            max(finite_float(row.get("low_degree_source_energy_h4800"), 0.0), finite_float(row.get("low_frequency_source_energy_h4800"), 0.0)),
            source_energy,
        ),
    }


def kan_source_bank_unit_tests() -> list[dict[str, Any]]:
    row = kan_source_bank_row(
        {
            "low_degree_source_energy_h3200": 2.0,
            "high_degree_reservoir_energy_h3200": 1.0,
            "low_frequency_source_energy_h3200": 0.5,
            "high_frequency_reservoir_energy_h3200": 0.25,
            "low_degree_source_energy_h4800": 1.0,
            "low_frequency_source_energy_h4800": 0.25,
        }
    )
    return [
        {"test": "source_fraction", "expected": 2.0 / 3.0, "actual": row["source_bank_fraction"], "pass": int(abs(float(row["source_bank_fraction"]) - 2.0 / 3.0) < 1.0e-12)},
        {"test": "retention", "expected": 0.5, "actual": row["source_bank_retention_h4800_over_h3200"], "pass": int(abs(float(row["source_bank_retention_h4800_over_h3200"]) - 0.5) < 1.0e-12)},
    ]


__all__ = ["kan_source_bank_row", "kan_source_bank_unit_tests", "source_bank_retention"]
