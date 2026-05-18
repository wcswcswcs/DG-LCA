"""Transitional packed stack markers.

These classes are audit metadata wrappers first.  They keep v8.x style routes
explicitly tagged as transitional so they cannot be counted as full-edge
PureKAN candidates.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransitionalStackDescriptor:
    stack_type: str
    model_level: str = "transitional"
    full_edge_purekan_pass: int = 0


PackedLinearSiluStack = TransitionalStackDescriptor("linear_silu_packed")
PackedPrefixNoInputGradStack = TransitionalStackDescriptor("linear_silu_prefix_no_input_grad")
CachedAtenLowerOnlySiluBackwardStack = TransitionalStackDescriptor("linear_silu_cached_aten_lower_only")
