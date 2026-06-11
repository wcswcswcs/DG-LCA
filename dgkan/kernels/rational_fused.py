"""v20 rational kernel status shim."""

from __future__ import annotations

from dgkan.kernels.v17_basis import repaired_eval


def kernel_status() -> dict[str, object]:
    return {
        "family": "D-RAT",
        "source_module": "dgkan.kernels.v17_basis.repaired_eval",
        "official_fused_kernel_available": 0,
        "no_materialize_available": 0,
        "supported_variants": "branchless_horner_smoke_only",
    }


def eval_rational(z):
    return repaired_eval(z, "D-RAT")
