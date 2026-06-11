"""Carrier descriptors for v17.0.1 FU audits."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class CarrierDescriptor:
    carrier_id: str
    family: str
    candidate_id: str
    basis_name: str
    promotion_allowed: int = 0

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def named_roles(family: str) -> list[str]:
    if family == "MLP":
        return ["hidden_weight", "hidden_bias", "readout"]
    if family == "LQ":
        return ["legendre_frame", "projection", "readout"]
    if family == "D-RAT":
        return ["numerator", "denominator", "readout"]
    if family == "D-FOU":
        return ["frequency_band", "band_readout", "mixing"]
    if family == "D-RBF":
        return ["center", "width", "local_readout"]
    if family == "D-WAV":
        return ["support", "scale", "local_readout"]
    return ["basis", "degree", "readout"]
