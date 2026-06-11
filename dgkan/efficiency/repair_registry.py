"""Family-specific repair registry for v17.0.1 basis efficiency audits."""

from __future__ import annotations


REPAIR_REGISTRY = {
    "D-CHE": ["CHE-R1", "CHE-R2", "CHE-R3", "CHE-R4", "CHE-R5", "CHE-R6"],
    "D-FOU": ["FOU-R1", "FOU-R2", "FOU-R3", "FOU-R4", "FOU-R5"],
    "LQ": ["LQ-R1", "LQ-R2", "LQ-R3", "LQ-R4", "LQ-R5"],
    "D-RAT": ["RAT-R1", "RAT-R2", "RAT-R3", "RAT-R4", "RAT-R5"],
    "D-RBF": ["RBF-R1", "RBF-R2", "RBF-R3", "RBF-R4", "RBF-R5"],
    "D-WAV": ["WAV-R1", "WAV-R2", "WAV-R3", "WAV-R4"],
}


def repair_attempts(family: str) -> list[str]:
    return list(REPAIR_REGISTRY.get(family, []))
