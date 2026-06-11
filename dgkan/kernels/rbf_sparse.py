"""v20 compact RBF sparse-kernel status shim."""

from __future__ import annotations

from dgkan.kernels import fused_rbf


def kernel_status() -> dict[str, object]:
    return {
        "family": "D-RBF",
        "source_module": "dgkan.kernels.fused_rbf",
        "official_fused_kernel_available": 0,
        "no_materialize_available": 1,
        "supported_variants": "sparse_local_smoke_only",
    }


forward_fused = getattr(fused_rbf, "forward_fused", None)
