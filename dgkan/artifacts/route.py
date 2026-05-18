"""Route helpers for v9 decision files."""

from __future__ import annotations

from typing import Any, Dict


def route_not_complete_decision(*, primary_blocker: str, next_required_implementation: str) -> Dict[str, Any]:
    return {
        "route": "R0-RefactorAuditInProgress",
        "legacy_parity_pass": 0,
        "clean_ce_pass": 0,
        "full_edge_contract_pass": 0,
        "full_edge_task_pass": 0,
        "functional_causality_pass": 0,
        "external_fair_pass": 0,
        "training_compute_fair_pass": 0,
        "robustness_pass": 0,
        "continual_balance_pass": 0,
        "best_candidate": "metric_unavailable",
        "best_model_level": "metric_unavailable",
        "success_v90_core_skeleton": 1,
        "success_v90_contract_hardening": 1,
        "success_v90_full_edge": 0,
        "success_v90_external_fair": 0,
        "success_v90_strong": 0,
        "primary_blocker": primary_blocker,
        "next_required_implementation": next_required_implementation,
    }
