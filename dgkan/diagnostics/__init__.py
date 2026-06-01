"""Reusable diagnostics helpers for DG-KAN experiment runners."""

from .classic_basis import (
    CLASSIC_FAMILY_SPECS,
    fnum,
    incremental_peak_bytes,
    memory_ratio,
    parse_csv,
    parse_ints,
    train_mlp_reference,
)

__all__ = [
    "CLASSIC_FAMILY_SPECS",
    "fnum",
    "incremental_peak_bytes",
    "memory_ratio",
    "parse_csv",
    "parse_ints",
    "train_mlp_reference",
]
