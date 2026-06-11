"""v20 Chebyshev fused-kernel status shim."""

from __future__ import annotations

from dgkan.kernels import fused_chebyshev_k3


SUPPORTED_VARIANTS = {
    "CHE20-R2-low-degree-k3-triton-officialize": "cheby_k3_triton_l3_matmul",
    "CHE20-R3-k4-triton-no-materialize-officialize": "cheby_k4_triton_l3_matmul",
    "CHE20-R4-k3-gradbuf-triton-officialize": "cheby_k3_triton_l3_gradbuf",
}


def kernel_status() -> dict[str, object]:
    return {
        "family": "D-CHE",
        "source_module": "dgkan.kernels.fused_chebyshev_k3",
        "official_fused_kernel_available": 1,
        "no_materialize_available": 1,
        "supported_variants": ";".join(sorted(SUPPORTED_VARIANTS)),
    }


forward_matmul = fused_chebyshev_k3.forward_matmul
backward = fused_chebyshev_k3.backward
backward_from_grad_logits = fused_chebyshev_k3.backward_from_grad_logits
