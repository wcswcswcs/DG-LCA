"""v20 Legendre/LQ kernel status shim.

No production fused CUDA path is present yet; v20 runners must record this as a
blocker instead of counting LQ as officialized.
"""

from __future__ import annotations

from dgkan.kernels.v17_basis import repaired_eval


def kernel_status() -> dict[str, object]:
    return {
        "family": "LQ",
        "source_module": "dgkan.kernels.v17_basis.repaired_eval",
        "official_fused_kernel_available": 0,
        "no_materialize_available": 0,
        "supported_variants": "torch_recurrence_smoke_only",
    }


def eval_legendre(z):
    return repaired_eval(z, "LQ")
