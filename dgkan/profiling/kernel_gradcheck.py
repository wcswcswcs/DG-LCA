"""Compatibility shim for v17.1 kernel gradcheck imports."""

from __future__ import annotations

from dgkan.kernels.v17_basis import kernel_correctness_row

__all__ = ["kernel_correctness_row"]
