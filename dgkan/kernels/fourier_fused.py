"""v20 Fourier fused-kernel status shim."""

from __future__ import annotations

from dgkan.kernels import fused_fourier_k2


SUPPORTED_VARIANTS = {
    "FOU20-R2-officialize-lowfreq-k2-stream": "fourier_k2_triton_l3_matmul",
    "FOU20-R4-k4-triton-no-materialize-officialize": "fourier_k4_triton_l3_matmul",
}


def kernel_status() -> dict[str, object]:
    return {
        "family": "D-FOU",
        "source_module": "dgkan.kernels.fused_fourier_k2",
        "official_fused_kernel_available": 1,
        "no_materialize_available": 1,
        "supported_variants": ";".join(sorted(SUPPORTED_VARIANTS)),
    }


forward_matmul = fused_fourier_k2.forward_matmul
backward = fused_fourier_k2.backward
backward_from_grad_logits = fused_fourier_k2.backward_from_grad_logits
