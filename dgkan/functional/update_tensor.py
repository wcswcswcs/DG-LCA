"""Compatibility shim for v17.1 UpdateTensor imports."""

from __future__ import annotations

from dgkan.fu.core import UpdateTensor, apply_update, small_step_sanity
from dgkan.fu.update_semantics import update_type_manifest_row, validate_update_semantics

__all__ = [
    "UpdateTensor",
    "apply_update",
    "small_step_sanity",
    "update_type_manifest_row",
    "validate_update_semantics",
]
