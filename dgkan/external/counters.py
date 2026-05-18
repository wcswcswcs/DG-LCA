"""Efficiency counter schema helpers."""

from __future__ import annotations


def unavailable_counter_row(stage: str) -> dict:
    return {"stage": stage, "status": "not_run", "reason": "v90_counter_not_implemented_yet"}
