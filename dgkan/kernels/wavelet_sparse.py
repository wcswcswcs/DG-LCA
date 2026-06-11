"""v20 wavelet sparse-kernel status shim."""

from __future__ import annotations

from dgkan.kernels import fused_hat_wavelet


def kernel_status() -> dict[str, object]:
    return {
        "family": "D-WAV",
        "source_module": "dgkan.kernels.fused_hat_wavelet",
        "official_fused_kernel_available": 0,
        "no_materialize_available": 1,
        "supported_variants": "sparse_support_smoke_only",
    }


forward_fused = getattr(fused_hat_wavelet, "forward_fused", None)
