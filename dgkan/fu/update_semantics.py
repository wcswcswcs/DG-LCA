"""Update-semantics validation helpers for v17.0.1."""

from __future__ import annotations

from dataclasses import dataclass

from dgkan.fu.core import UpdateTensor
from dgkan.fu.types import canonical_update_row


@dataclass
class UpdateValidation:
    valid: int
    reason: str


def validate_update_semantics(update: UpdateTensor) -> UpdateValidation:
    row = canonical_update_row(update)
    if row["direction_type"] == "unknown":
        return UpdateValidation(0, "unknown_direction_type")
    if row["sign_convention"] == "unknown":
        return UpdateValidation(0, "unknown_sign_convention")
    if update.kind == "cotangent" and update.space != "function":
        return UpdateValidation(0, "cotangent_space_mismatch")
    return UpdateValidation(1, "pass")


def update_type_manifest_row(update: UpdateTensor) -> dict[str, object]:
    row = canonical_update_row(update)
    validation = validate_update_semantics(update)
    row["semantics_valid"] = validation.valid
    row["validation_reason"] = validation.reason
    return row
