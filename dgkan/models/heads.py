"""Head descriptors for clean v9 candidates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeadDescriptor:
    head_type: str
    edge_basis: str
    full_edge_compatible: int


POLY2_SILU_HEAD = HeadDescriptor("poly2_silu_head", "poly2_silu", 1)
