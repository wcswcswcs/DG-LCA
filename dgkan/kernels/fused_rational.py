"""Compatibility alias for v22.02 rational active-repair audits.

The current repository exposes the rational path through
``dgkan.kernels.rational_fused``. This module preserves the plan's file name
without claiming an official fused train path.
"""

from __future__ import annotations

from dgkan.kernels.rational_fused import eval_rational, kernel_status


__all__ = ["eval_rational", "kernel_status"]
