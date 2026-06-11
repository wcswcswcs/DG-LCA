"""Compatibility alias for v22.02 local RBF active-repair audits."""

from __future__ import annotations

from dgkan.kernels.rbf_sparse import forward_fused, kernel_status


__all__ = ["forward_fused", "kernel_status"]
