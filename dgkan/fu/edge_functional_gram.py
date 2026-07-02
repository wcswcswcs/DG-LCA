"""Functional edge Gram helpers for DG-KAN v23.01.

This module intentionally reuses the v23.00R dense functional Gram
implementation. The separate file exists so v23.01 identity checks can import a
functional-geometry module by name without pretending the implementation is new.
"""

from __future__ import annotations

from dgkan.fu.edge_sobolev_metrics import (
    basis_name_for_key,
    functional_edge_gram,
    functional_gram_condition,
    functional_gram_cost,
    functional_gram_inverse_retention,
    functional_gram_whitened_vector,
)

__all__ = [
    "basis_name_for_key",
    "functional_edge_gram",
    "functional_gram_condition",
    "functional_gram_cost",
    "functional_gram_inverse_retention",
    "functional_gram_whitened_vector",
]
