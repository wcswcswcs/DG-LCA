"""Shared constants for v9 clean DG-KAN experiments."""

from __future__ import annotations

OFFICIAL_LOSS_TYPE = "CE"
OFFICIAL_LABEL_SMOOTHING = 0.0

FAIR_PARAM_RATIO_MAX = 1.05
FAIR_FLOPS_RATIO_MAX = 1.05
FAIR_STEP_RATIO_MAX = 1.50
FAIR_MEMORY_RATIO_MAX = 1.05
UNKNOWN_TIME_FRACTION_MAX = 0.10

V9_REQUIRED_ARTIFACTS = (
    "run_manifest.json",
    "code_audit_legacy_current.csv",
    "candidate_registry_clean.csv",
    "contract_audit_clean.csv",
    "legacy_refactor_parity.csv",
    "label_smoothing_removal_audit.csv",
    "full_edge_implementation_audit.csv",
    "gradcheck_full_edge.csv",
    "clean_transitional_revalidation.csv",
    "full_edge_external_validation.csv",
    "functional_causality_controls.csv",
    "robust_timing_protocols.csv",
    "training_compute_counter.csv",
    "kanbefair_baseline_reproduction.csv",
    "external_fair_envelope.csv",
    "robustness_perturbation.csv",
    "continual_balance_multisplit.csv",
    "boundary_audit.csv",
    "route_decision.json",
    "aggregate_decision.json",
    "failure_table.csv",
)
