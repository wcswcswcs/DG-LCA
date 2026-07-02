"""Functional population trust-flow optimizer aliases for DG-KAN v23.01."""

from __future__ import annotations

from dgkan.optim.edge_sobolev_population_flow import (
    EdgeSobolevAdamW,
    EdgeSobolevPopulationFlow,
    EdgeSobolevSNRFU,
    EdgeSobolevSNRFUWithDebtVeto,
    mark_kan_edge_params,
)


class FunctionalPopulationTrustFlow(EdgeSobolevSNRFU):
    """Population-gated functional edge optimizer.

    v23.01 uses the v23.00R optimizer implementation with
    ``edge_metric_type='functional_gram'`` and finite-step trust handled by the
    runner. This subclass is an explicit import target for identity checks.
    """


FunctionalEdgeAdamW = EdgeSobolevAdamW
FunctionalPopulationTrustFlowWithDebtVeto = EdgeSobolevSNRFUWithDebtVeto


__all__ = [
    "FunctionalEdgeAdamW",
    "FunctionalPopulationTrustFlow",
    "FunctionalPopulationTrustFlowWithDebtVeto",
    "EdgeSobolevPopulationFlow",
    "mark_kan_edge_params",
]
