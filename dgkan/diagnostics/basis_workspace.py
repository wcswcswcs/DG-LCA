"""Basis-kernel workspace diagnostics for v12.31 and later.

The code in this module is deliberately implementation-level, not experiment
orchestration.  Runners pass in concrete models/batches and this module returns
measured CUDA peak phases plus static component estimates.  The static estimates
are audit fields only; route gates should use measured peak ratios.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BasisWorkspaceCandidate:
    candidate_id: str
    family: str
    method_id: str
    repair_note: str
    exact_kernel_implemented: int = 0
    actual_file_path: str = ""
    actual_symbol: str = ""
    uses_triton: int = 0
    uses_cuda_extension: int = 0
    uses_torch_autograd_graph: int = 0
    uses_loss_backward: int = 0
    materializes_basis_tensor: int = 1
    materializes_derivative_tensor: int = 1
    materializes_readout_grad_tensor: int = 1
    basis_tensor_shape_if_any: str = ""
    derivative_tensor_shape_if_any: str = ""


V1231_BASIS_CANDIDATES: dict[str, BasisWorkspaceCandidate] = {
    "D-RAT7-FusedGroupRationalNoMaterialize": BasisWorkspaceCandidate(
        "D-RAT7-FusedGroupRationalNoMaterialize",
        "D-RAT",
        "B7ex-RationalKAT-flashgroup-G16-h32-linearresGain19200-freezeBackbone-hiddenBias-L3",
        "existing no-pair rational path used as no-materialize audit proxy; no new fused kernel claimed",
    ),
    "D-RAT8-RecomputeDenominatorBackward": BasisWorkspaceCandidate(
        "D-RAT8-RecomputeDenominatorBackward",
        "D-RAT",
        "B7eq-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-logitBias-L3",
        "existing checkpoint-like denominator path; recompute kernel not newly introduced",
    ),
    "D-RAT9-ChunkedReadoutGradNoPersistentBasis": BasisWorkspaceCandidate(
        "D-RAT9-ChunkedReadoutGradNoPersistentBasis",
        "D-RAT",
        "B7cv-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-readGradR64-crossWarm-L3",
        "existing readGradR64 chunked readout-grad path",
    ),
    "D-RAT10-FusedDenNumReadoutGrad": BasisWorkspaceCandidate(
        "D-RAT10-FusedDenNumReadoutGrad",
        "D-RAT",
        "B7ey-RationalKAT-flashgroup-G16-h32-linearresGain19200-freezeBackbone-hiddenBias-manualAdamW-L3",
        "existing no-pair manual-AdamW path; den/num fused gradient not newly claimed",
    ),
    "D-RAT11-LowMemGroupSharedDenomPlusLineC": BasisWorkspaceCandidate(
        "D-RAT11-LowMemGroupSharedDenomPlusLineC",
        "D-RAT",
        "B7lp-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3",
        "existing pairNorm/shared-denominator LineC-oriented path",
    ),
    "D-RAT12-RationalWorkspaceMinStrongDiag": BasisWorkspaceCandidate(
        "D-RAT12-RationalWorkspaceMinStrongDiag",
        "D-RAT",
        "B7fa-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairReadBucketR136G32-crossZero-freezeBackbone-hiddenBias-manualAdamW-L3",
        "existing pairReadBucket low-memory diagnostic path",
    ),
    "D-RAT13-RationalLogitBatchRMSNormSG": BasisWorkspaceCandidate(
        "D-RAT13-RationalLogitBatchRMSNormSG",
        "D-RAT",
        "B7md-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-logitBatchRMSNormSG-freezeBackbone-hiddenBias-manualAdamW-L3",
        "existing B7lp anchor with label-free stop-gradient batch logit RMS normalization; output-geometry/tail audit, not CE-directed",
    ),
    "D-RAT14-RationalLogitBatchRMSMixSG025": BasisWorkspaceCandidate(
        "D-RAT14-RationalLogitBatchRMSMixSG025",
        "D-RAT",
        "B7mg-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-logitBatchRMSMixSG025-freezeBackbone-hiddenBias-manualAdamW-L3",
        "existing B7md residual output-geometry repair with 25 percent stop-gradient logit RMS mix; not CE-directed",
    ),
    "D-RAT15-RationalLogitBatchRMSMixSG050": BasisWorkspaceCandidate(
        "D-RAT15-RationalLogitBatchRMSMixSG050",
        "D-RAT",
        "B7mh-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-logitBatchRMSMixSG050-freezeBackbone-hiddenBias-manualAdamW-L3",
        "existing B7md residual output-geometry repair with 50 percent stop-gradient logit RMS mix; not CE-directed",
    ),
    "D-WAV6-HatWaveletNoMaterializeLocal2": BasisWorkspaceCandidate(
        "D-WAV6-HatWaveletNoMaterializeLocal2",
        "D-WAV",
        "B5h-HatWaveletKAN-local-K4",
        "existing local hat-wavelet path",
    ),
    "D-WAV7-SharedScaleLocalWavelet": BasisWorkspaceCandidate(
        "D-WAV7-SharedScaleLocalWavelet",
        "D-WAV",
        "B5i-HatWaveletKAN-K4-inputcrossL4P128-tritonL3",
        "existing input-cross local wavelet path",
    ),
    "D-WAV8-ScaleEntropyCappedWavelet": BasisWorkspaceCandidate(
        "D-WAV8-ScaleEntropyCappedWavelet",
        "D-WAV",
        "B5n-HatWaveletKAN-K4-inputcrossL4P128-linearraw010-tritonL3",
        "existing low-raw residual wavelet path",
    ),
    "D-WAV9-TailCoverageNoExtraWorkspace": BasisWorkspaceCandidate(
        "D-WAV9-TailCoverageNoExtraWorkspace",
        "D-WAV",
        "B5k-HatWaveletKAN-K4-inputcrossL4P128-linearres-tritonL3",
        "existing local-tail coverage path",
    ),
    "D-CHE6-K3DegreeEnergyCap": BasisWorkspaceCandidate(
        "D-CHE6-K3DegreeEnergyCap",
        "D-CHE",
        "B3m-ChebyKAN-K3-h112-tritonL3-gradbuf",
        "existing K3 Chebyshev grad-buffer path",
    ),
    "D-CHE7-LateEnableK4": BasisWorkspaceCandidate(
        "D-CHE7-LateEnableK4",
        "D-CHE",
        "B3f-ChebyKAN-K4-tritonL3-matmulTile",
        "existing K4 Chebyshev tiled path",
    ),
    "D-CHE8-OrthogonalInputNormNoExtraMem": BasisWorkspaceCandidate(
        "D-CHE8-OrthogonalInputNormNoExtraMem",
        "D-CHE",
        "B3z-ChebyKAN-K3-h88-paircrossR32-tritonL3-gradbuf",
        "existing compact paircross Chebyshev path used for input-norm audit",
    ),
    "D-CHE9-RoleWiseLowDegreeResidual": BasisWorkspaceCandidate(
        "D-CHE9-RoleWiseLowDegreeResidual",
        "D-CHE",
        "B3v-ChebyKAN-K3-h112-inputcrossL4P128-linearres-tritonL3-gradbuf",
        "existing low-degree residual Chebyshev path",
    ),
    "D-RBF7-CompactK4PlusLinearResidual": BasisWorkspaceCandidate(
        "D-RBF7-CompactK4PlusLinearResidual",
        "D-RBF",
        "B2v-GaussianRBF-K4-inputcrossL4P128-linearres-tritonL3",
        "existing compact K4 RBF residual path",
    ),
    "D-RBF8-QuantileCentersFixedWidth": BasisWorkspaceCandidate(
        "D-RBF8-QuantileCentersFixedWidth",
        "D-RBF",
        "B2z-GaussianRBF-K4-orthoInputrot2L4P128-tritonL3",
        "existing orthogonal-input RBF path",
    ),
    "D-RBF9-TriangularBumpNoExp": BasisWorkspaceCandidate(
        "D-RBF9-TriangularBumpNoExp",
        "D-RBF",
        "B2ab-GaussianRBF-K4-orthoInputsqL4P128-tritonL3",
        "existing no-new-exp diagnostic proxy; no triangular kernel claimed",
    ),
    "D-RBF10-LocalRBFExpressionPatch": BasisWorkspaceCandidate(
        "D-RBF10-LocalRBFExpressionPatch",
        "D-RBF",
        "B2w-GaussianRBF-K4-inputrot2L4P256-tritonL3",
        "existing wider local RBF expression path",
    ),
    "D-FOU6-LowFreqPlusIdentityResidual": BasisWorkspaceCandidate(
        "D-FOU6-LowFreqPlusIdentityResidual",
        "D-FOU",
        "B4p-FourierKAN-lowfreq-K4-h64-linearres-gemmDirectL3",
        "existing low-frequency Fourier residual path",
    ),
    "D-FOU7-LateEnableK4WithEnergyCap": BasisWorkspaceCandidate(
        "D-FOU7-LateEnableK4WithEnergyCap",
        "D-FOU",
        "B4n-FourierKAN-lowfreq-K4-h64-linearres-tritonL3-matmulTile",
        "existing low-frequency K4 tiled Fourier path",
    ),
    "D-FOU8-PhaseAmplitudeSharedNoExtraMem": BasisWorkspaceCandidate(
        "D-FOU8-PhaseAmplitudeSharedNoExtraMem",
        "D-FOU",
        "B4w-FourierKAN-lowfreq-K4-h8-linearres050-tritonL3-matmulTile",
        "existing compact low-hidden Fourier path",
    ),
    "D-FOU9-FourierNoiseLeakGuard": BasisWorkspaceCandidate(
        "D-FOU9-FourierNoiseLeakGuard",
        "D-FOU",
        "B4g-FourierKAN-lowfreq-K2-tritonL3-matmulTile",
        "existing low-frequency K2 Fourier path",
    ),
}


V1232_BASIS_CANDIDATES: dict[str, BasisWorkspaceCandidate] = {
    "D-RAT16-DenP01FloorNoCE": BasisWorkspaceCandidate(
        "D-RAT16-DenP01FloorNoCE",
        "D-RAT",
        "B7il-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatCenterSG-hiddenRatScale050-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3",
        "existing stop-gradient centered hidden rational tail proxy; no explicit denominator-p01 floor kernel is claimed",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="GroupedRationalKATKAN",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-RAT17-RPrimeCapNoCE": BasisWorkspaceCandidate(
        "D-RAT17-RPrimeCapNoCE",
        "D-RAT",
        "B7kc-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3",
        "existing low-amplitude bounded hidden rational residual proxy for derivative cap; no CE-directed rule",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="GroupedRationalKATKAN",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-RAT18-TangentNormTrustNoCE": BasisWorkspaceCandidate(
        "D-RAT18-TangentNormTrustNoCE",
        "D-RAT",
        "B7lq-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3",
        "existing stop-gradient batch cap and pairNorm path used as tangent-norm trust proxy; no label/CE source",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="GroupedRationalKATKAN",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-RAT19-DenSlopeJointStabilityNoCE": BasisWorkspaceCandidate(
        "D-RAT19-DenSlopeJointStabilityNoCE",
        "D-RAT",
        "B7kf-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3",
        "existing small smooth hidden tanh residual plus stop-gradient cap proxy for den/slope stability; no CE-tail direction",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="GroupedRationalKATKAN",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-RAT20-LogitSpectrumClipStopGrad": BasisWorkspaceCandidate(
        "D-RAT20-LogitSpectrumClipStopGrad",
        "D-RAT",
        "B7ma-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossPCAR128-crossZero-b32-logitRMSNormSG-freezeBackbone-hiddenBias-manualAdamW-L3",
        "existing samplewise stop-gradient logit RMS geometry proxy for spectrum clipping; CE metrics remain audit only",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="GroupedRationalKATKAN",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-RAT21-EntropyFloorUnlabeled": BasisWorkspaceCandidate(
        "D-RAT21-EntropyFloorUnlabeled",
        "D-RAT",
        "B7mg-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-logitBatchRMSMixSG025-freezeBackbone-hiddenBias-manualAdamW-L3",
        "existing 25 percent stop-gradient batch RMS mix used as unlabeled entropy/top-tail proxy; no explicit entropy floor kernel is claimed",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="GroupedRationalKATKAN",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-RAT22-TopGapFloorNoLabel": BasisWorkspaceCandidate(
        "D-RAT22-TopGapFloorNoLabel",
        "D-RAT",
        "B7ki-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-crossBatchSGCap100-freezeBackbone-hiddenBias-manualAdamW-L3",
        "existing no-hidden stop-gradient batch cap proxy for top-gap floor; no label/CE target",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="GroupedRationalKATKAN",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-RAT23-LogitNormEMAAnchorNoCE": BasisWorkspaceCandidate(
        "D-RAT23-LogitNormEMAAnchorNoCE",
        "D-RAT",
        "B7mb-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-logitRMSNormSG-freezeBackbone-hiddenBias-manualAdamW-L3",
        "existing stop-gradient logit RMS path used as logit-norm anchor proxy; no EMA state or CE-derived direction is claimed",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="GroupedRationalKATKAN",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-CHE10-FusedRecurrenceNoMaterialize-K3": BasisWorkspaceCandidate(
        "D-CHE10-FusedRecurrenceNoMaterialize-K3",
        "D-CHE",
        "B3e-ChebyKAN-K3-tritonL3-matmulTile",
        "existing Triton Chebyshev K3 recurrence matmul/backward audited as true no-materialize vertical slice",
        exact_kernel_implemented=1,
        actual_file_path="dgkan/kernels/fused_chebyshev_k3.py",
        actual_symbol="forward_matmul/backward",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-CHE11-FusedRecurrenceNoMaterialize-K4": BasisWorkspaceCandidate(
        "D-CHE11-FusedRecurrenceNoMaterialize-K4",
        "D-CHE",
        "B3f-ChebyKAN-K4-tritonL3-matmulTile",
        "existing Triton Chebyshev K4 recurrence matmul/backward audited as true no-materialize vertical slice",
        exact_kernel_implemented=1,
        actual_file_path="dgkan/kernels/fused_chebyshev_k3.py",
        actual_symbol="forward_matmul_k4/backward_k4",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-FOU10-FusedSincosLowFreqK2": BasisWorkspaceCandidate(
        "D-FOU10-FusedSincosLowFreqK2",
        "D-FOU",
        "B4g-FourierKAN-lowfreq-K2-tritonL3-matmulTile",
        "existing Triton Fourier K2 sin/cos matmul/backward audited as true no-materialize vertical slice",
        exact_kernel_implemented=1,
        actual_file_path="dgkan/kernels/fused_fourier_k2.py",
        actual_symbol="forward_matmul/backward",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-FOU11-FusedSincosLowFreqK4SharedAmp": BasisWorkspaceCandidate(
        "D-FOU11-FusedSincosLowFreqK4SharedAmp",
        "D-FOU",
        "B4n-FourierKAN-lowfreq-K4-h64-linearres-tritonL3-matmulTile",
        "existing Triton Fourier K4 low-frequency linear-residual matmul/backward audited as true no-materialize vertical slice",
        exact_kernel_implemented=1,
        actual_file_path="dgkan/kernels/fused_fourier_k2.py",
        actual_symbol="forward_matmul_k4_linearres/backward_k4_linearres",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
}


V1233_BASIS_CANDIDATES: dict[str, BasisWorkspaceCandidate] = {
    "D-RAT24-DenDerivativeTelemetryKernel": BasisWorkspaceCandidate(
        "D-RAT24-DenDerivativeTelemetryKernel",
        "D-RAT",
        "B7il-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatCenterSG-hiddenRatScale050-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3",
        "existing grouped rational path with internal basis_diagnostics telemetry; exact fused telemetry kernel is not newly claimed",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="GroupedRationalKATKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-RAT25-DenDerivativeTelemetryRecomputeBackward": BasisWorkspaceCandidate(
        "D-RAT25-DenDerivativeTelemetryRecomputeBackward",
        "D-RAT",
        "B7ey-RationalKAT-flashgroup-G16-h32-linearresGain19200-freezeBackbone-hiddenBias-manualAdamW-L3",
        "existing low-memory manual-AdamW rational path with post-hoc denominator/derivative telemetry; recompute telemetry kernel not claimed",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="GroupedRationalKATKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-RAT26-TangentTrustRegionNoCE": BasisWorkspaceCandidate(
        "D-RAT26-TangentTrustRegionNoCE",
        "D-RAT",
        "B7lq-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3",
        "existing stop-gradient batch cap and pairNorm path used as tangent trust-region proxy; no label/CE source",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="GroupedRationalKATKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-RAT27-DenSlopeGuardNoCE": BasisWorkspaceCandidate(
        "D-RAT27-DenSlopeGuardNoCE",
        "D-RAT",
        "B7kf-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3",
        "existing smooth hidden residual and stop-gradient cap proxy for denominator/slope guard; no CE-tail direction",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="GroupedRationalKATKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-RAT28-GroupDiversityPreservingRational": BasisWorkspaceCandidate(
        "D-RAT28-GroupDiversityPreservingRational",
        "D-RAT",
        "B7lp-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3",
        "existing pairNorm/shared-denominator rational path audited for group diversity telemetry",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="GroupedRationalKATKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-RAT29-LineCStableTangentMix": BasisWorkspaceCandidate(
        "D-RAT29-LineCStableTangentMix",
        "D-RAT",
        "B7ki-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-crossBatchSGCap100-freezeBackbone-hiddenBias-manualAdamW-L3",
        "existing stop-gradient batch-cap rational path used as LineC-stable tangent mix proxy",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="GroupedRationalKATKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-RAT30-LowMemoryTelemetryStrong": BasisWorkspaceCandidate(
        "D-RAT30-LowMemoryTelemetryStrong",
        "D-RAT",
        "B7fa-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairReadBucketR136G32-crossZero-freezeBackbone-hiddenBias-manualAdamW-L3",
        "existing pairReadBucket low-memory rational path with telemetry audit; strong fused telemetry kernel not claimed",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="GroupedRationalKATKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-RAT31-TelemetryAblationControl": BasisWorkspaceCandidate(
        "D-RAT31-TelemetryAblationControl",
        "D-RAT",
        "B7ex-RationalKAT-flashgroup-G16-h32-linearresGain19200-freezeBackbone-hiddenBias-L3",
        "v12.33 telemetry ablation control: existing no-pair rational path, not a new tail-stability kernel",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="GroupedRationalKATKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-RAT32-LogitSpectrumTelemetryRepairNoCE": BasisWorkspaceCandidate(
        "D-RAT32-LogitSpectrumTelemetryRepairNoCE",
        "D-RAT",
        "B7ma-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossPCAR128-crossZero-b32-logitRMSNormSG-freezeBackbone-hiddenBias-manualAdamW-L3",
        "fallback repair after telemetry tail blocker: label-free stop-gradient logit RMS geometry proxy; no CE-tail direction and no exact fused rational telemetry kernel claimed",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="GroupedRationalKATKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-RAT33-LogitNormAnchorTelemetryRepairNoCE": BasisWorkspaceCandidate(
        "D-RAT33-LogitNormAnchorTelemetryRepairNoCE",
        "D-RAT",
        "B7mb-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-logitRMSNormSG-freezeBackbone-hiddenBias-manualAdamW-L3",
        "fallback repair after telemetry tail blocker: label-free stop-gradient logit-norm anchor proxy; no CE-tail direction and no exact fused rational telemetry kernel claimed",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="GroupedRationalKATKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-CHE12-LifetimeRecomputeBackward-K3": BasisWorkspaceCandidate(
        "D-CHE12-LifetimeRecomputeBackward-K3",
        "D-CHE",
        "B3e-ChebyKAN-K3-tritonL3-matmulTile",
        "existing exact Chebyshev K3 no-materialize kernel audited for lifetime overlap; no new kernel claimed",
        exact_kernel_implemented=1,
        actual_file_path="dgkan/kernels/fused_chebyshev_k3.py",
        actual_symbol="forward_matmul/backward",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-CHE13-FusedReadoutGradNoMaterialize-K3": BasisWorkspaceCandidate(
        "D-CHE13-FusedReadoutGradNoMaterialize-K3",
        "D-CHE",
        "B3e-ChebyKAN-K3-tritonL3-matmulTile",
        "existing Chebyshev K3 exact path used for fused readout-grad lifetime audit; no wrapper success counted without memory gate",
        exact_kernel_implemented=1,
        actual_file_path="dgkan/kernels/fused_chebyshev_k3.py",
        actual_symbol="forward_matmul/backward",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-CHE14-OptimizerStateLifetimeReuse-K3": BasisWorkspaceCandidate(
        "D-CHE14-OptimizerStateLifetimeReuse-K3",
        "D-CHE",
        "B3m-ChebyKAN-K3-h112-tritonL3-gradbuf",
        "existing Chebyshev K3 grad-buffer path audited for optimizer-state lifetime; exact fused kernel not newly extended",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="ChebyKAN",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=1,
    ),
    "D-CHE15-FullStepNoMaterialize-K3": BasisWorkspaceCandidate(
        "D-CHE15-FullStepNoMaterialize-K3",
        "D-CHE",
        "B3e-ChebyKAN-K3-tritonL3-matmulTile",
        "existing Chebyshev K3 exact path audited as full-step no-materialize candidate",
        exact_kernel_implemented=1,
        actual_file_path="dgkan/kernels/fused_chebyshev_k3.py",
        actual_symbol="forward_matmul/backward",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-FOU12-LifetimeRecomputeBackward-K2": BasisWorkspaceCandidate(
        "D-FOU12-LifetimeRecomputeBackward-K2",
        "D-FOU",
        "B4g-FourierKAN-lowfreq-K2-tritonL3-matmulTile",
        "existing exact Fourier K2 no-materialize kernel audited for lifetime overlap; no new kernel claimed",
        exact_kernel_implemented=1,
        actual_file_path="dgkan/kernels/fused_fourier_k2.py",
        actual_symbol="forward_matmul/backward",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-FOU13-FusedReadoutGradNoMaterialize-K2": BasisWorkspaceCandidate(
        "D-FOU13-FusedReadoutGradNoMaterialize-K2",
        "D-FOU",
        "B4g-FourierKAN-lowfreq-K2-tritonL3-matmulTile",
        "existing Fourier K2 exact path used for fused readout-grad lifetime audit; no wrapper success counted without memory gate",
        exact_kernel_implemented=1,
        actual_file_path="dgkan/kernels/fused_fourier_k2.py",
        actual_symbol="forward_matmul/backward",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-FOU14-SincosSharedWorkspace-K2": BasisWorkspaceCandidate(
        "D-FOU14-SincosSharedWorkspace-K2",
        "D-FOU",
        "B4g-FourierKAN-lowfreq-K2-tritonL3-matmulTile",
        "existing Fourier K2 exact path audited as shared sin/cos workspace candidate",
        exact_kernel_implemented=1,
        actual_file_path="dgkan/kernels/fused_fourier_k2.py",
        actual_symbol="forward_matmul/backward",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-FOU15-FullStepNoMaterialize-K2": BasisWorkspaceCandidate(
        "D-FOU15-FullStepNoMaterialize-K2",
        "D-FOU",
        "B4g-FourierKAN-lowfreq-K2-tritonL3-matmulTile",
        "existing Fourier K2 exact path audited as full-step no-materialize candidate",
        exact_kernel_implemented=1,
        actual_file_path="dgkan/kernels/fused_fourier_k2.py",
        actual_symbol="forward_matmul/backward",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-RBF11-CompactExpressionRepair-Monitor": BasisWorkspaceCandidate(
        "D-RBF11-CompactExpressionRepair-Monitor",
        "D-RBF",
        "B2v-GaussianRBF-K4-inputcrossL4P128-linearres-tritonL3",
        "RBF compact expression monitor only; no exact fused lifetime kernel claimed",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="GaussianRBFKAN",
        uses_triton=1,
        uses_loss_backward=1,
    ),
    "D-WAV10-HatWaveletLifetimeRepair-Monitor": BasisWorkspaceCandidate(
        "D-WAV10-HatWaveletLifetimeRepair-Monitor",
        "D-WAV",
        "B5h-HatWaveletKAN-local-K4",
        "Wavelet lifetime repair monitor only; no exact fused lifetime kernel claimed",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="HatWaveletKAN",
        uses_triton=1,
        uses_loss_backward=1,
    ),
}


V12342_BASIS_CANDIDATES: dict[str, BasisWorkspaceCandidate] = {
    **V1233_BASIS_CANDIDATES,
    "D-RAT34-TrainingDenDerivativeStabilityNoCE": BasisWorkspaceCandidate(
        "D-RAT34-TrainingDenDerivativeStabilityNoCE",
        "D-RAT",
        "B7lq-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3",
        "v12.34.2 rational denominator/derivative stability substrate proxy using existing stop-gradient batch-cap path; no CE-tail direction and no new fused kernel claimed",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="GroupedRationalKATKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-RAT35-ReadoutRationalDecoupleNoCE": BasisWorkspaceCandidate(
        "D-RAT35-ReadoutRationalDecoupleNoCE",
        "D-RAT",
        "B7fa-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairReadBucketR136G32-crossZero-freezeBackbone-hiddenBias-manualAdamW-L3",
        "existing pair-read bucket rational path audited as readout/rational decoupling substrate; no label/CE source",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="GroupedRationalKATKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-RAT36-TangentConditionStabilizerNoCE": BasisWorkspaceCandidate(
        "D-RAT36-TangentConditionStabilizerNoCE",
        "D-RAT",
        "B7kc-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3",
        "existing bounded hidden-rational residual path used for tangent-condition substrate audit; no CE/vector direction",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="GroupedRationalKATKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-RAT37-LineCStableTangentMixV2": BasisWorkspaceCandidate(
        "D-RAT37-LineCStableTangentMixV2",
        "D-RAT",
        "B7ki-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-crossBatchSGCap100-freezeBackbone-hiddenBias-manualAdamW-L3",
        "existing LineC-stable stop-gradient batch cap rational path; LineC remains audit only",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="GroupedRationalKATKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-RAT38-GroupDiversityTransportNoCE": BasisWorkspaceCandidate(
        "D-RAT38-GroupDiversityTransportNoCE",
        "D-RAT",
        "B7lp-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3",
        "existing pairNorm rational path audited for group-function diversity transport; no label/CE direction",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="GroupedRationalKATKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-RAT39-DenDerivativeSubstrateRepairNoCE": BasisWorkspaceCandidate(
        "D-RAT39-DenDerivativeSubstrateRepairNoCE",
        "D-RAT",
        "B7ey-RationalKAT-flashgroup-G16-h32-linearresGain19200-freezeBackbone-hiddenBias-manualAdamW-L3",
        "existing manual-AdamW rational path reused for denominator/derivative substrate repair audit; no new training-time stability kernel claimed",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="GroupedRationalKATKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-CHE16-DegreeEnergyDampingSubstrate": BasisWorkspaceCandidate(
        "D-CHE16-DegreeEnergyDampingSubstrate",
        "D-CHE",
        "B3m-ChebyKAN-K3-h112-tritonL3-gradbuf",
        "Chebyshev degree-energy damping substrate using existing grad-buffer K3 path; lifetime gate still measured, not assumed",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="BasisKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=1,
    ),
    "D-CHE17-HighDegreeLateEnableSubstrate": BasisWorkspaceCandidate(
        "D-CHE17-HighDegreeLateEnableSubstrate",
        "D-CHE",
        "B3f-ChebyKAN-K4-tritonL3-matmulTile",
        "Chebyshev high-degree late-enable substrate using existing K4 Triton path; no new scheduler claimed",
        exact_kernel_implemented=1,
        actual_file_path="dgkan/kernels/fused_chebyshev_k3.py",
        actual_symbol="forward_matmul_k4/backward_k4",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-CHE18-RoleDegreeEnergyCapSubstrate": BasisWorkspaceCandidate(
        "D-CHE18-RoleDegreeEnergyCapSubstrate",
        "D-CHE",
        "B3v-ChebyKAN-K3-h112-inputcrossL4P128-linearres-tritonL3-gradbuf",
        "Chebyshev role-degree energy cap substrate via existing input-cross residual path; functional cap not assumed",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="BasisKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=1,
    ),
    "D-CHE19-ChebyTangentTrustSubstrate": BasisWorkspaceCandidate(
        "D-CHE19-ChebyTangentTrustSubstrate",
        "D-CHE",
        "B3z-ChebyKAN-K3-h88-paircrossR32-tritonL3-gradbuf",
        "compact Chebyshev paircross path audited as tangent-trust substrate; no LineC target used for direction",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="BasisKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=1,
    ),
    "D-FOU16-FrequencyBandDampingSubstrate": BasisWorkspaceCandidate(
        "D-FOU16-FrequencyBandDampingSubstrate",
        "D-FOU",
        "B4g-FourierKAN-lowfreq-K2-tritonL3-matmulTile",
        "Fourier low-frequency band damping substrate using existing exact K2 path",
        exact_kernel_implemented=1,
        actual_file_path="dgkan/kernels/fused_fourier_k2.py",
        actual_symbol="forward_matmul/backward",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-FOU17-PhaseStabilityCorrectionSubstrate": BasisWorkspaceCandidate(
        "D-FOU17-PhaseStabilityCorrectionSubstrate",
        "D-FOU",
        "B4n-FourierKAN-lowfreq-K4-h64-linearres-tritonL3-matmulTile",
        "Fourier phase-stability substrate via existing K4 low-frequency linear-residual path",
        exact_kernel_implemented=1,
        actual_file_path="dgkan/kernels/fused_fourier_k2.py",
        actual_symbol="forward_matmul_k4_linearres/backward_k4_linearres",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-FOU18-LowFreqSignalTransportSubstrate": BasisWorkspaceCandidate(
        "D-FOU18-LowFreqSignalTransportSubstrate",
        "D-FOU",
        "B4p-FourierKAN-lowfreq-K4-h64-linearres-gemmDirectL3",
        "Fourier low-frequency signal transport substrate via existing gemmDirect residual path",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="BasisKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=1,
    ),
    "D-FOU19-HighFreqNoiseLeakVetoSubstrate": BasisWorkspaceCandidate(
        "D-FOU19-HighFreqNoiseLeakVetoSubstrate",
        "D-FOU",
        "B4w-FourierKAN-lowfreq-K4-h8-linearres050-tritonL3-matmulTile",
        "compact Fourier high-frequency noise-leak veto substrate proxy; no label/noise target direction",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="BasisKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=1,
    ),
    "D-RBF12-CenterOccupancyRebalanceSubstrate": BasisWorkspaceCandidate(
        "D-RBF12-CenterOccupancyRebalanceSubstrate",
        "D-RBF",
        "B2v-GaussianRBF-K4-inputcrossL4P128-linearres-tritonL3",
        "RBF center-occupancy rebalance substrate via existing compact local residual path; no exact RBF fused lifetime kernel claimed",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="BasisKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
    ),
    "D-RBF13-WidthConditionGuardSubstrate": BasisWorkspaceCandidate(
        "D-RBF13-WidthConditionGuardSubstrate",
        "D-RBF",
        "B2z-GaussianRBF-K4-orthoInputrot2L4P128-tritonL3",
        "RBF width-condition guard substrate via existing orthogonal input path; width telemetry is audit only",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="BasisKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
    ),
    "D-RBF14-OOGBoundaryRepairSubstrate": BasisWorkspaceCandidate(
        "D-RBF14-OOGBoundaryRepairSubstrate",
        "D-RBF",
        "B2w-GaussianRBF-K4-inputrot2L4P256-tritonL3",
        "RBF out-of-grid boundary repair substrate via existing wider local rotation path",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="BasisKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
    ),
    "D-RBF15-LocalCurvatureSmoothSubstrate": BasisWorkspaceCandidate(
        "D-RBF15-LocalCurvatureSmoothSubstrate",
        "D-RBF",
        "B2ab-GaussianRBF-K4-orthoInputsqL4P128-tritonL3",
        "RBF local-curvature smooth substrate via existing orthogonal projected-square path",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="BasisKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
    ),
    "D-RBF16-ActiveCenterDiversityTransportSubstrate": BasisWorkspaceCandidate(
        "D-RBF16-ActiveCenterDiversityTransportSubstrate",
        "D-RBF",
        "B2v-GaussianRBF-K4-inputcrossL4P128-linearres-tritonL3",
        "RBF active-center diversity transport substrate using existing compact local residual path",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="BasisKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
    ),
    "D-WAV11-ScaleEnergyBalanceSubstrate": BasisWorkspaceCandidate(
        "D-WAV11-ScaleEnergyBalanceSubstrate",
        "D-WAV",
        "B5h-HatWaveletKAN-local-K4",
        "Wavelet scale-energy balance substrate via existing local support path; no exact wavelet fused lifetime kernel claimed",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="BasisKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
    ),
    "D-WAV12-LocalSupportOccupancyRepairSubstrate": BasisWorkspaceCandidate(
        "D-WAV12-LocalSupportOccupancyRepairSubstrate",
        "D-WAV",
        "B5i-HatWaveletKAN-K4-inputcrossL4P128-tritonL3",
        "Wavelet local-support occupancy repair substrate via existing input-cross local support path",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="BasisKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
    ),
    "D-WAV13-LocalTailCoverageGuardSubstrate": BasisWorkspaceCandidate(
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "D-WAV",
        "B5k-HatWaveletKAN-K4-inputcrossL4P128-linearres-tritonL3",
        "Wavelet local-tail coverage guard substrate via existing linear residual local support path",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="BasisKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
    ),
    "D-WAV14-SupportOverlapEntropyGuardSubstrate": BasisWorkspaceCandidate(
        "D-WAV14-SupportOverlapEntropyGuardSubstrate",
        "D-WAV",
        "B5n-HatWaveletKAN-K4-inputcrossL4P128-linearraw010-tritonL3",
        "Wavelet support-overlap entropy guard substrate via existing low raw residual path",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="BasisKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
    ),
    "D-WAV15-ScaleDiversityTransportSubstrate": BasisWorkspaceCandidate(
        "D-WAV15-ScaleDiversityTransportSubstrate",
        "D-WAV",
        "B5k-HatWaveletKAN-K4-inputcrossL4P128-linearres-tritonL3",
        "Wavelet scale-diversity transport substrate via existing local support linear residual path",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="BasisKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
    ),
}


V1235_BASIS_CANDIDATES: dict[str, BasisWorkspaceCandidate] = {
    **V12342_BASIS_CANDIDATES,
    "D-RAT40-ResponseReadyReadoutCouplingSubstrate": BasisWorkspaceCandidate(
        "D-RAT40-ResponseReadyReadoutCouplingSubstrate",
        "D-RAT",
        "B7fa-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairReadBucketR136G32-crossZero-freezeBackbone-hiddenBias-manualAdamW-L3",
        "v12.35 response-ready rational readout-coupling substrate mapped to existing pair-read bucket path; LineC and CE remain audit-only",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="GroupedRationalKATKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-CHE20-DegreeNormalizedReadoutHealthSubstrate": BasisWorkspaceCandidate(
        "D-CHE20-DegreeNormalizedReadoutHealthSubstrate",
        "D-CHE",
        "B3v-ChebyKAN-K3-h112-inputcrossL4P128-linearres-tritonL3-gradbuf",
        "v12.35 Chebyshev degree-normalized readout health candidate using existing input-cross residual path; no new exact kernel claimed",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="BasisKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=1,
    ),
    "D-FOU20-LowFreqIdentityResidualHealthSubstrate": BasisWorkspaceCandidate(
        "D-FOU20-LowFreqIdentityResidualHealthSubstrate",
        "D-FOU",
        "B4n-FourierKAN-lowfreq-K4-h64-linearres-tritonL3-matmulTile",
        "v12.35 Fourier low-frequency identity-residual task-health candidate mapped to existing K4 linear-residual path",
        exact_kernel_implemented=1,
        actual_file_path="dgkan/kernels/fused_fourier_k2.py",
        actual_symbol="forward_matmul_k4_linearres/backward_k4_linearres",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-RBF17-CompactCapacityK4HealthSubstrate": BasisWorkspaceCandidate(
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "D-RBF",
        "B2v-GaussianRBF-K4-inputcrossL4P128-linearres-tritonL3",
        "v12.35 compact-capacity RBF task-health candidate using existing K4 input-cross residual path; no exact fused RBF kernel claimed",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="BasisKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
    ),
    "D-WAV16-SupportStableHatHealthSubstrate": BasisWorkspaceCandidate(
        "D-WAV16-SupportStableHatHealthSubstrate",
        "D-WAV",
        "B5k-HatWaveletKAN-K4-inputcrossL4P128-linearres-tritonL3",
        "v12.35 support-stable hat-wavelet health candidate using existing local support linear-residual path; no Morlet/heavy path introduced",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="BasisKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
    ),
    "D-WAV17-Raw005SupportHealthSubstrate": BasisWorkspaceCandidate(
        "D-WAV17-Raw005SupportHealthSubstrate",
        "D-WAV",
        "B5o-HatWaveletKAN-K4-inputcrossL4P128-linearraw005-tritonL3",
        "v14.3 support/reservoir diagnostic using existing lower raw-linear-residual hat-wavelet path; no LineC/tail direction and no new kernel claimed",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="BasisKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-WAV18-Raw002SupportHealthSubstrate": BasisWorkspaceCandidate(
        "D-WAV18-Raw002SupportHealthSubstrate",
        "D-WAV",
        "B5p-HatWaveletKAN-K4-inputcrossL4P128-linearraw002-tritonL3",
        "v14.3 support/reservoir diagnostic using existing very-low raw-linear-residual hat-wavelet path; no LineC/tail direction and no new kernel claimed",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="BasisKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
    "D-WAV19-Raw001SupportHealthSubstrate": BasisWorkspaceCandidate(
        "D-WAV19-Raw001SupportHealthSubstrate",
        "D-WAV",
        "B5q-HatWaveletKAN-K4-inputcrossL4P128-linearraw001-tritonL3",
        "v14.3 support/reservoir diagnostic using existing near-zero raw-linear-residual hat-wavelet path; no LineC/tail direction and no new kernel claimed",
        exact_kernel_implemented=0,
        actual_file_path="dgkan/models/fc_purekan_primitives.py",
        actual_symbol="BasisKAN.basis_diagnostics",
        uses_triton=1,
        uses_loss_backward=1,
        materializes_basis_tensor=0,
        materializes_derivative_tensor=0,
        materializes_readout_grad_tensor=0,
    ),
}


V1232_WORKSPACE_FIELDS = [
    *[
        "run_id",
        "family",
        "candidate_id",
        "mapped_method_id",
        "dataset",
        "seed",
        "batch_size",
        "hidden_dim",
        "basis_K",
        "group_count",
        "forward_peak_bytes",
        "backward_peak_bytes",
        "update_peak_bytes",
        "raw_peak_bytes",
        "incremental_peak_bytes",
        "mlp_raw_peak_bytes",
        "mlp_incremental_peak_bytes",
        "raw_memory_ratio_vs_mlp",
        "incremental_memory_ratio_vs_mlp",
        "step_time_ms",
        "profile_steps",
        "mlp_step_time_ms",
        "step_ratio_vs_mlp",
        "basis_activation_bytes",
        "basis_derivative_bytes",
        "readout_grad_bytes",
        "coeff_grad_bytes",
        "optimizer_state_bytes",
        "workspace_temp_bytes",
        "saved_tensor_bytes",
        "num_custom_kernels",
        "num_torch_ops",
        "num_gemm_calls",
        "num_exp_calls",
        "num_sin_cos_calls",
        "num_gather_scatter_calls",
        "uses_dense_basis_materialization",
        "uses_recompute_backward",
        "uses_chunked_grad",
        "uses_fused_update",
        "workspace_gate_pass",
        "workspace_strong_gate_pass",
        "top_peak_source",
        "component_byte_accounting_source",
        "exact_kernel_implemented",
        "actual_file_path",
        "actual_symbol",
        "uses_triton",
        "uses_cuda_extension",
        "uses_torch_autograd_graph",
        "uses_loss_backward",
        "materializes_basis_tensor",
        "materializes_derivative_tensor",
        "materializes_readout_grad_tensor",
        "basis_tensor_shape_if_any",
        "derivative_tensor_shape_if_any",
        "uses_label_or_ce_for_direction",
        "uses_y_for_stats",
        "forbidden_token_present",
        "repair_note",
        "promotion_allowed",
        "no_fake",
    ]
]


V1233_WORKSPACE_FIELDS = [
    *V1232_WORKSPACE_FIELDS,
    "telemetry_required",
    "lifetime_repair_target",
]


V12342_WORKSPACE_FIELDS = [
    *V1233_WORKSPACE_FIELDS,
    "substrate_gate_candidate",
    "all_basis_telemetry_required",
]


V1235_WORKSPACE_FIELDS = [
    *V12342_WORKSPACE_FIELDS,
    "substrate_health_gate_candidate",
    "response_dictionary_required",
]


WORKSPACE_FIELDS = [
    "run_id",
    "family",
    "candidate_id",
    "mapped_method_id",
    "dataset",
    "seed",
    "batch_size",
    "hidden_dim",
    "basis_K",
    "group_count",
    "forward_peak_bytes",
    "backward_peak_bytes",
    "update_peak_bytes",
    "raw_peak_bytes",
    "incremental_peak_bytes",
    "mlp_raw_peak_bytes",
    "mlp_incremental_peak_bytes",
    "raw_memory_ratio_vs_mlp",
    "incremental_memory_ratio_vs_mlp",
    "step_time_ms",
    "profile_steps",
    "mlp_step_time_ms",
    "step_ratio_vs_mlp",
    "basis_activation_bytes",
    "basis_derivative_bytes",
    "readout_grad_bytes",
    "coeff_grad_bytes",
    "optimizer_state_bytes",
    "workspace_temp_bytes",
    "saved_tensor_bytes",
    "num_custom_kernels",
    "num_torch_ops",
    "num_gemm_calls",
    "num_exp_calls",
    "num_sin_cos_calls",
    "num_gather_scatter_calls",
    "uses_dense_basis_materialization",
    "uses_recompute_backward",
    "uses_chunked_grad",
    "uses_fused_update",
    "workspace_gate_pass",
    "workspace_strong_gate_pass",
    "top_peak_source",
    "component_byte_accounting_source",
    "exact_kernel_implemented",
    "repair_note",
    "promotion_allowed",
    "no_fake",
]


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(float(x.strip())) for x in str(text).split(",") if x.strip()]


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        if value == "" or value is None:
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def tensor_bytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def parameter_bytes(model: torch.nn.Module, trainable_only: bool = False) -> int:
    total = 0
    seen: set[int] = set()
    for p in model.parameters():
        if trainable_only and not p.requires_grad:
            continue
        ptr = int(p.data_ptr())
        if ptr in seen:
            continue
        seen.add(ptr)
        total += tensor_bytes(p)
    return total


def buffer_bytes(model: torch.nn.Module) -> int:
    total = 0
    seen: set[int] = set()
    for b in model.buffers():
        ptr = int(b.data_ptr())
        if ptr in seen:
            continue
        seen.add(ptr)
        total += tensor_bytes(b)
    return total


def optimizer_state_bytes(opt: torch.optim.Optimizer) -> int:
    total = 0
    seen: set[int] = set()
    for state in opt.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                ptr = int(value.data_ptr())
                if ptr in seen:
                    continue
                seen.add(ptr)
                total += tensor_bytes(value)
    return total


def spec_int(spec: Any, name: str, default: int = 0) -> int:
    try:
        value = getattr(spec, name, default)
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def spec_bool(spec: Any, name: str, default: bool = False) -> bool:
    return bool(getattr(spec, name, default))


def static_workspace_accounting(
    spec: Any,
    model: torch.nn.Module,
    batch_size: int,
    input_dim: int,
    output_dim: int,
    opt: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    hidden_dim = spec_int(spec, "hidden_dim", 0)
    basis_k = spec_int(spec, "num_basis", spec_int(spec, "degree", 0))
    group_count = max(1, spec_int(spec, "group_count", spec_int(spec, "groups", 1)))
    dtype_bytes = 4
    dense_basis = int(spec_bool(spec, "uses_dense_basis_tensor", False) or "dense" in str(getattr(spec, "workspace_strategy", "")).lower())
    active_basis_bytes = int(batch_size) * max(1, hidden_dim) * max(1, basis_k) * dtype_bytes
    basis_activation_bytes = active_basis_bytes if dense_basis else 0
    basis_derivative_bytes = active_basis_bytes if dense_basis and not spec_bool(spec, "recompute_backward", False) else 0
    readout_grad_bytes = max(1, hidden_dim) * max(1, output_dim) * dtype_bytes
    coeff_grad_bytes = parameter_bytes(model, trainable_only=True)
    opt_bytes = optimizer_state_bytes(opt) if opt is not None else 0
    workspace_temp = 0 if spec_bool(spec, "recompute_backward", False) else max(1, batch_size) * max(1, hidden_dim) * max(1, min(basis_k, 8)) * dtype_bytes
    saved_tensor = int(batch_size) * (max(1, input_dim) + max(1, hidden_dim) + max(1, output_dim)) * dtype_bytes
    cid = str(getattr(spec, "candidate_id", ""))
    family = str(getattr(spec, "family", "")).lower()
    num_exp = int("rbf" in family or "rational" in family or "B2" in cid)
    num_sin = int("fourier" in family or "B4" in cid)
    num_gather = int("wavelet" in family or "hat" in family or "B5" in cid or "pair" in cid.lower())
    num_gemm = 2 + int("readblock" in cid.lower() or "matmul" in cid.lower() or "gemm" in cid.lower())
    return {
        "hidden_dim": hidden_dim,
        "basis_K": basis_k,
        "group_count": group_count,
        "basis_activation_bytes": basis_activation_bytes,
        "basis_derivative_bytes": basis_derivative_bytes,
        "readout_grad_bytes": readout_grad_bytes,
        "coeff_grad_bytes": coeff_grad_bytes,
        "optimizer_state_bytes": opt_bytes,
        "workspace_temp_bytes": workspace_temp,
        "saved_tensor_bytes": saved_tensor,
        "num_custom_kernels": int("triton" in cid.lower() or "flash" in cid.lower()),
        "num_torch_ops": 1 + num_exp + num_sin + num_gather,
        "num_gemm_calls": num_gemm,
        "num_exp_calls": num_exp,
        "num_sin_cos_calls": num_sin,
        "num_gather_scatter_calls": num_gather,
        "uses_dense_basis_materialization": dense_basis,
        "uses_recompute_backward": int(spec_bool(spec, "recompute_backward", False) or "recompute" in cid.lower()),
        "uses_chunked_grad": int("gradr" in cid.lower() or "chunk" in cid.lower() or "bucket" in cid.lower()),
        "uses_fused_update": int("manualadamw" in cid.lower() or "fused" in cid.lower()),
        "component_byte_accounting_source": "static_spec_shape_estimate_not_gate_source",
    }


def cuda_peak() -> float:
    return float(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else float("nan")


def measured_phase_step(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    xb: torch.Tensor,
    yb: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        baseline = float(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline = float("nan")
    opt.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    logits = model(xb)
    loss = F.cross_entropy(logits, yb)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_peak = cuda_peak()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    backward_peak = cuda_peak()
    opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    update_peak = cuda_peak()
    step_ms = (time.perf_counter() - t0) * 1000.0
    raw_peak = max(forward_peak, backward_peak, update_peak)
    incremental = max(0.0, raw_peak - baseline) if math.isfinite(raw_peak) and math.isfinite(baseline) else float("nan")
    return {
        "memory_baseline_bytes": baseline,
        "forward_peak_bytes": forward_peak,
        "backward_peak_bytes": backward_peak,
        "update_peak_bytes": update_peak,
        "raw_peak_bytes": raw_peak,
        "incremental_peak_bytes": incremental,
        "step_time_ms": step_ms,
    }


def measured_phase_window(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    xb: torch.Tensor,
    yb: torch.Tensor,
    device: torch.device,
    steps: int,
) -> dict[str, float]:
    """Measure several post-warmup train steps and aggregate conservatively."""
    steps = max(1, int(steps))
    samples = [measured_phase_step(model, opt, xb, yb, device) for _ in range(steps)]

    def max_field(name: str) -> float:
        vals = [fnum(s.get(name)) for s in samples if math.isfinite(fnum(s.get(name)))]
        return max(vals) if vals else float("nan")

    step_times = [fnum(s.get("step_time_ms")) for s in samples if math.isfinite(fnum(s.get("step_time_ms")))]
    step_q90 = float(torch.tensor(step_times).quantile(0.90).item()) if step_times else float("nan")
    baseline_vals = [fnum(s.get("memory_baseline_bytes")) for s in samples if math.isfinite(fnum(s.get("memory_baseline_bytes")))]
    return {
        "memory_baseline_bytes": min(baseline_vals) if baseline_vals else float("nan"),
        "forward_peak_bytes": max_field("forward_peak_bytes"),
        "backward_peak_bytes": max_field("backward_peak_bytes"),
        "update_peak_bytes": max_field("update_peak_bytes"),
        "raw_peak_bytes": max_field("raw_peak_bytes"),
        "incremental_peak_bytes": max_field("incremental_peak_bytes"),
        "step_time_ms": step_q90,
        "profile_steps": steps,
    }


def train_epoch_timed(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, float]:
    """Run one supervised hardening epoch and return timing/loss audit stats."""
    model.train()
    times: list[float] = []
    losses: list[float] = []
    perm = torch.randperm(int(x_train.shape[0]), device=device, generator=generator)
    for off in range(0, int(x_train.shape[0]), int(batch_size)):
        idx = perm[off:off + int(batch_size)]
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        loss = F.cross_entropy(model(x_train[idx]), y_train[idx])
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1000.0)
        losses.append(float(loss.detach().item()))
    step_q90 = float(torch.tensor(times, device=device).quantile(0.90).item()) if times else float("nan")
    loss_mean = float(sum(losses) / len(losses)) if losses else float("nan")
    return {
        "train_loss_mean": loss_mean,
        "step_time_q90_ms": step_q90,
        "epoch_steps": len(times),
    }


def finite_mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def trajectory_auc(values: list[float]) -> float:
    """Small-budget trajectory AUC audit: mean over epoch samples."""
    return finite_mean(values)


def trajectory_time_auc(values: list[float], step_times_ms: list[float]) -> float:
    pairs = [
        (float(v), float(t))
        for v, t in zip(values, step_times_ms)
        if math.isfinite(float(v)) and math.isfinite(float(t)) and float(t) > 0.0
    ]
    return float(sum(v * t for v, t in pairs) / len(pairs)) if pairs else float("nan")


def memory_ratio(numer: float, denom: float) -> float:
    d = fnum(denom)
    if d > 0:
        return fnum(numer) / d
    return float("nan")


def workspace_gate(raw_ratio: float, incremental_ratio: float, step_ratio: float) -> int:
    return int(fnum(raw_ratio, 999.0) <= 1.50 and fnum(incremental_ratio, 999.0) <= 3.00 and fnum(step_ratio, 999.0) <= 1.75)


def workspace_strong_gate(raw_ratio: float, incremental_ratio: float, step_ratio: float) -> int:
    return int(fnum(raw_ratio, 999.0) <= 1.20 and fnum(incremental_ratio, 999.0) <= 2.00 and fnum(step_ratio, 999.0) <= 1.30)


def workspace_gate_v1232(raw_ratio: float, incremental_ratio: float, step_ratio: float) -> int:
    """v12.32 stricter P0 workspace gate."""
    return int(fnum(raw_ratio, 999.0) <= 1.05 and fnum(incremental_ratio, 999.0) <= 1.75 and fnum(step_ratio, 999.0) <= 1.75)


def workspace_strong_gate_v1232(raw_ratio: float, incremental_ratio: float, step_ratio: float) -> int:
    """v12.32 strong gate.  Raw ratio is kept at the P0 bound."""
    return int(fnum(raw_ratio, 999.0) <= 1.05 and fnum(incremental_ratio, 999.0) <= 1.35 and fnum(step_ratio, 999.0) <= 1.40)


def workspace_gate_v1233(raw_ratio: float, incremental_ratio: float, step_ratio: float) -> int:
    """v12.33 strict workspace gate; same numerical P0 bound as v12.32."""
    return workspace_gate_v1232(raw_ratio, incremental_ratio, step_ratio)


def workspace_strong_gate_v1233(raw_ratio: float, incremental_ratio: float, step_ratio: float) -> int:
    """v12.33 strong workspace gate; same numerical strong bound as v12.32."""
    return workspace_strong_gate_v1232(raw_ratio, incremental_ratio, step_ratio)


def workspace_gate_v12342(raw_ratio: float, incremental_ratio: float, step_ratio: float) -> int:
    """v12.34.2 substrate efficiency gate S memory/time portion."""
    return int(fnum(raw_ratio, 999.0) <= 1.50 and fnum(incremental_ratio, 999.0) <= 2.50 and fnum(step_ratio, 999.0) <= 1.50)


def workspace_strong_gate_v12342(raw_ratio: float, incremental_ratio: float, step_ratio: float) -> int:
    """v12.34.2 healthy-base memory/time portion."""
    return int(fnum(raw_ratio, 999.0) <= 1.25 and fnum(incremental_ratio, 999.0) <= 1.25 and fnum(step_ratio, 999.0) <= 1.25)


def workspace_gate_v1235(raw_ratio: float, incremental_ratio: float, step_ratio: float) -> int:
    """v12.35 substrate-health map memory/time portion."""
    return int(fnum(raw_ratio, 999.0) <= 1.05 and fnum(incremental_ratio, 999.0) <= 1.75 and fnum(step_ratio, 999.0) <= 1.75)


def workspace_strong_gate_v1235(raw_ratio: float, incremental_ratio: float, step_ratio: float) -> int:
    """v12.35 healthy-base memory/time portion."""
    return int(fnum(raw_ratio, 999.0) <= 1.05 and fnum(incremental_ratio, 999.0) <= 1.25 and fnum(step_ratio, 999.0) <= 1.25)


def top_peak_source(row: dict[str, Any]) -> str:
    measured = {
        "forward_phase": fnum(row.get("forward_peak_bytes"), -1.0),
        "backward_phase": fnum(row.get("backward_peak_bytes"), -1.0),
        "update_phase": fnum(row.get("update_peak_bytes"), -1.0),
    }
    phase = max(measured, key=measured.get)
    components = {
        "basis_activation": fnum(row.get("basis_activation_bytes"), -1.0),
        "basis_derivative": fnum(row.get("basis_derivative_bytes"), -1.0),
        "readout_grad": fnum(row.get("readout_grad_bytes"), -1.0),
        "coeff_grad": fnum(row.get("coeff_grad_bytes"), -1.0),
        "optimizer_state": fnum(row.get("optimizer_state_bytes"), -1.0),
        "temp_workspace": fnum(row.get("workspace_temp_bytes"), -1.0),
    }
    component = max(components, key=components.get)
    return f"{phase}:{component}"


def component_peak_rows(workspace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in workspace_rows:
        for component in [
            "basis_activation_bytes",
            "basis_derivative_bytes",
            "readout_grad_bytes",
            "coeff_grad_bytes",
            "optimizer_state_bytes",
            "workspace_temp_bytes",
            "saved_tensor_bytes",
        ]:
            rows.append({
                "stage": "V1231_BASIS_COMPONENT_PEAK",
                "run_id": row.get("run_id", ""),
                "family": row.get("family", ""),
                "candidate_id": row.get("candidate_id", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "component": component.replace("_bytes", ""),
                "estimated_bytes": row.get(component, ""),
                "component_byte_accounting_source": row.get("component_byte_accounting_source", ""),
                "measured_top_peak_source": row.get("top_peak_source", ""),
                "promotion_allowed": 0,
                "no_fake": 1,
            })
    return rows


def microkernel_correctness_rows(candidates: list[BasisWorkspaceCandidate]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cand in candidates:
        rows.append({
            "stage": "V1231_BASIS_MICROKERNEL_CORRECTNESS",
            "family": cand.family,
            "candidate_id": cand.candidate_id,
            "mapped_method_id": cand.method_id,
            "exact_kernel_implemented": cand.exact_kernel_implemented,
            "gradient_correctness_executed": 0,
            "gradient_correctness_pass": 0,
            "reason": "no new custom microkernel introduced in v12.31; existing dgkan primitive path measured by workspace truth",
            "used_for_promotion": 0,
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    return rows


def kernel_implementation_manifest_rows(candidates: list[BasisWorkspaceCandidate]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cand in candidates:
        rows.append({
            "stage": "V1232_KERNEL_IMPLEMENTATION_MANIFEST",
            "candidate_id": cand.candidate_id,
            "family": cand.family,
            "mapped_method_id": cand.method_id,
            "actual_file_path": cand.actual_file_path,
            "actual_symbol": cand.actual_symbol,
            "exact_kernel_implemented": cand.exact_kernel_implemented,
            "uses_triton": cand.uses_triton,
            "uses_cuda_extension": cand.uses_cuda_extension,
            "uses_torch_autograd_graph": cand.uses_torch_autograd_graph,
            "uses_loss_backward": cand.uses_loss_backward,
            "materializes_basis_tensor": cand.materializes_basis_tensor,
            "materializes_derivative_tensor": cand.materializes_derivative_tensor,
            "materializes_readout_grad_tensor": cand.materializes_readout_grad_tensor,
            "basis_tensor_shape_if_any": cand.basis_tensor_shape_if_any,
            "derivative_tensor_shape_if_any": cand.derivative_tensor_shape_if_any,
            "uses_label_or_ce_for_direction": 0,
            "uses_y_for_stats": 0,
            "forbidden_token_present": 0,
            "workspace_temp_bytes_static": "",
            "repair_note": cand.repair_note,
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    return rows


def exact_kernel_audit_rows(
    candidates: list[BasisWorkspaceCandidate],
    audit_by_candidate: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    audit_by_candidate = audit_by_candidate or {}
    rows: list[dict[str, Any]] = []
    for cand in candidates:
        audit = audit_by_candidate.get(cand.candidate_id, {})
        manual_forward = int(fnum(audit.get("manual_forward_available"), 0.0))
        manual_backward = int(fnum(audit.get("manual_backward_available"), 0.0))
        forward_err = fnum(audit.get("output_max_abs_error"))
        grad_err = fnum(audit.get("grad_relerr_max"))
        grad_cos = fnum(audit.get("grad_cos_min"))
        gradcheck = int(
            cand.exact_kernel_implemented
            and manual_forward
            and manual_backward
            and (not math.isfinite(forward_err) or forward_err <= 5.0e-4)
            and (not math.isfinite(grad_err) or grad_err <= 5.0e-3)
            and (not math.isfinite(grad_cos) or grad_cos >= 0.999)
        )
        no_materialize_hard_gate = int(
            cand.exact_kernel_implemented
            and cand.materializes_basis_tensor == 0
            and cand.materializes_derivative_tensor == 0
        )
        rows.append({
            "stage": "V1232_EXACT_KERNEL_AUDIT",
            "candidate_id": cand.candidate_id,
            "family": cand.family,
            "mapped_method_id": cand.method_id,
            "actual_file_path": cand.actual_file_path,
            "actual_symbol": cand.actual_symbol,
            "exact_kernel_implemented": cand.exact_kernel_implemented,
            "materializes_basis_tensor": cand.materializes_basis_tensor,
            "materializes_derivative_tensor": cand.materializes_derivative_tensor,
            "materializes_readout_grad_tensor": cand.materializes_readout_grad_tensor,
            "manual_gradcheck_executed": int(bool(audit)),
            "manual_forward_available": manual_forward,
            "manual_backward_available": manual_backward,
            "manual_gradcheck_pass": gradcheck,
            "max_forward_abs_error": audit.get("output_max_abs_error", ""),
            "max_grad_rel_error": audit.get("grad_relerr_max", ""),
            "grad_cos_min": audit.get("grad_cos_min", ""),
            "A4_expression_smoke_pass": gradcheck,
            "no_materialize_hard_gate_pass": no_materialize_hard_gate,
            "reason": audit.get("error", "") or ("exact kernel audited" if cand.exact_kernel_implemented else "workspace/proxy alias only; no exact fused kernel claimed"),
            "used_for_promotion": 0,
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    return rows


def rational_tail_metrics(model: torch.nn.Module, x: torch.Tensor) -> dict[str, float | int | str]:
    """Unlabeled rational/output-tail audit metrics.

    The function intentionally avoids labels and CE.  Denominator derivative
    internals are not exposed uniformly across existing grouped rational paths,
    so denominator fields are marked as proxy/instrumentation unavailable unless
    future kernels expose them directly.
    """
    model.eval()
    with torch.no_grad():
        logits = model(x).float()
        probs = logits.softmax(dim=1)
        sorted_probs = probs.sort(dim=1, descending=True).values
        top_gap = sorted_probs[:, 0] - sorted_probs[:, 1] if sorted_probs.shape[1] > 1 else sorted_probs[:, 0]
        entropy = -(probs.clamp_min(1.0e-8).log() * probs).sum(dim=1)
        logit_norm = logits.norm(dim=1)
        abs_logits = logits.abs().flatten()
    q = lambda t, p: float(torch.quantile(t.float(), float(p)).item()) if t.numel() else float("nan")
    return {
        "den_min_batch": float("nan"),
        "den_p01_batch": float("nan"),
        "den_condition_batch": float("nan"),
        "r_prime_p95": float("nan"),
        "r_prime_p99": float("nan"),
        "r_double_prime_p95": float("nan"),
        "r_double_prime_p99": float("nan"),
        "rational_output_p99": q(abs_logits, 0.99),
        "logit_norm_p99": q(logit_norm, 0.99),
        "unlabeled_entropy_mean": float(entropy.mean().item()) if entropy.numel() else float("nan"),
        "unlabeled_entropy_p05": q(entropy, 0.05),
        "unlabeled_top1_top2_gap_p01": q(top_gap, 0.01),
        "denominator_instrumentation_available": 0,
        "denominator_audit_note": "existing grouped rational paths do not expose per-sample denominator/derivative tensors; logits/probability tail metrics are unlabeled proxies",
    }


def _safe_quantile(t: torch.Tensor, p: float) -> float:
    return float(torch.quantile(t.float().flatten(), float(p)).item()) if t.numel() else float("nan")


def rational_telemetry_metrics(model: torch.nn.Module, x: torch.Tensor, seed: int = 123333) -> dict[str, float | int | str]:
    """v12.33 unlabeled rational denominator/derivative telemetry.

    This is an audit helper, not a training direction.  It uses model state and
    unlabeled inputs only.  When the grouped rational module exposes numerator,
    denominator and normalization helpers, denominator/derivative tensors are
    reconstructed under ``no_grad`` for telemetry; no CE/NLL/ECE/labels are read.
    """
    out: dict[str, float | int | str] = dict(rational_tail_metrics(model, x))
    telemetry_available = 0
    telemetry_bytes = 0
    try:
        if hasattr(model, "basis_diagnostics"):
            diag = model.basis_diagnostics(x)
            out.update({
                "den_min_batch": fnum(diag.get("den_min")),
                "den_p01_batch": fnum(diag.get("den_p01")),
                "den_condition_batch": fnum(diag.get("den_condition")),
                "r_prime_p95": fnum(diag.get("r_prime_p95")),
                "r_double_prime_p95": fnum(diag.get("r_double_prime_p95")),
                "group_function_diversity": fnum(diag.get("group_function_diversity")),
                "group_dead_fraction": fnum(diag.get("group_dead_fraction")),
            })
            telemetry_available = 1
    except Exception as exc:  # noqa: BLE001 - telemetry error is an audit field.
        out["basis_diagnostics_error"] = f"{type(exc).__name__}: {exc}"

    try:
        if hasattr(model, "numerator") and hasattr(model, "denominator") and hasattr(model, "_norm_input"):
            with torch.no_grad():
                z = model._norm_input(x)
                groups = int(model.numerator.shape[0])
                bsz, dim = int(z.shape[0]), int(z.shape[1])
                group_size = max(1, dim // max(1, groups))
                if groups * group_size == dim:
                    zg = z.reshape(bsz, groups, group_size).float()
                    a = model.numerator.detach().float()[:, None, :]
                    b_abs = model.denominator.detach().float().abs()[:, None, :]
                    x2 = zg * zg
                    x3 = x2 * zg
                    x4 = x3 * zg
                    x5 = x4 * zg
                    ax = zg.abs()
                    ax2 = ax * ax
                    ax3 = ax2 * ax
                    ax4 = ax3 * ax
                    p = a[..., 0] + a[..., 1] * zg + a[..., 2] * x2 + a[..., 3] * x3 + a[..., 4] * x4 + a[..., 5] * x5
                    q = 1.0 + b_abs[..., 0] * ax + b_abs[..., 1] * ax2 + b_abs[..., 2] * ax3 + b_abs[..., 3] * ax4
                    p1 = a[..., 1] + 2.0 * a[..., 2] * zg + 3.0 * a[..., 3] * x2 + 4.0 * a[..., 4] * x3 + 5.0 * a[..., 5] * x4
                    p2 = 2.0 * a[..., 2] + 6.0 * a[..., 3] * zg + 12.0 * a[..., 4] * x2 + 20.0 * a[..., 5] * x3
                    sign_x = torch.where(zg < 0.0, torch.full_like(zg, -1.0), torch.ones_like(zg))
                    q1 = sign_x * (b_abs[..., 0] + 2.0 * b_abs[..., 1] * ax + 3.0 * b_abs[..., 2] * ax2 + 4.0 * b_abs[..., 3] * ax3)
                    q2 = 2.0 * b_abs[..., 1] + 6.0 * b_abs[..., 2] * ax + 12.0 * b_abs[..., 3] * ax2
                    q_safe = q.clamp_min(1.0e-8)
                    r = p / q_safe
                    r_prime = p1 / q_safe - p * q1 / q_safe.square()
                    r_double_prime = p2 / q_safe - 2.0 * p1 * q1 / q_safe.square() - p * q2 / q_safe.square() + 2.0 * p * q1.square() / q_safe.pow(3)
                    group_energy = r.square().mean(dim=(0, 2))
                    out.update({
                        "den_min_batch": float(q.min().item()),
                        "den_p01_batch": _safe_quantile(q, 0.01),
                        "den_p99_batch": _safe_quantile(q, 0.99),
                        "den_condition_batch": float((q.max() / q.min().clamp_min(1.0e-8)).item()),
                        "r_prime_p95": _safe_quantile(r_prime.abs(), 0.95),
                        "r_prime_p99": _safe_quantile(r_prime.abs(), 0.99),
                        "r_double_prime_p95": _safe_quantile(r_double_prime.abs(), 0.95),
                        "r_double_prime_p99": _safe_quantile(r_double_prime.abs(), 0.99),
                        "rational_output_p99": _safe_quantile(r.abs(), 0.99),
                        "group_function_diversity": float(group_energy.std(unbiased=False).item() / group_energy.mean().clamp_min(1.0e-8).item()),
                        "group_dead_fraction": float((group_energy < 1.0e-8).float().mean().item()),
                    })
                    telemetry_available = 1
                    telemetry_bytes += tensor_bytes(q) + tensor_bytes(r_prime) + tensor_bytes(r_double_prime)
                else:
                    out["telemetry_shape_note"] = f"input_dim_{dim}_not_divisible_by_groups_{groups}"
    except Exception as exc:  # noqa: BLE001
        out["telemetry_reconstruction_error"] = f"{type(exc).__name__}: {exc}"

    try:
        with torch.no_grad():
            xb = x[: min(64, int(x.shape[0]))]
            base = model(xb).float()
            gen = torch.Generator(device=xb.device).manual_seed(int(seed))
            noise = torch.randn(xb.shape, device=xb.device, generator=gen, dtype=xb.dtype) * 0.01
            pert = model((xb + noise).clamp(0.0, 1.0)).float()
            response = (pert - base) / 0.01
            centered = response - response.mean(dim=0, keepdim=True)
            s = torch.linalg.svdvals(centered.float())
            sq = s.square()
            denom = sq.sum().clamp_min(1.0e-8)
            rank = float((denom.square() / sq.square().sum().clamp_min(1.0e-8)).item()) if s.numel() else float("nan")
            top_share = float((sq.max() / denom).item()) if s.numel() else float("nan")
            out.update({
                "tangent_effective_rank": rank,
                "tangent_top_eigen_share": top_share,
            })
            telemetry_bytes += tensor_bytes(response)
    except Exception as exc:  # noqa: BLE001
        out["tangent_telemetry_error"] = f"{type(exc).__name__}: {exc}"

    out.update({
        "denominator_instrumentation_available": int(telemetry_available),
        "telemetry_available": int(telemetry_available),
        "telemetry_materialized_bytes": int(telemetry_bytes),
        "exact_fused_rational_kernel": 0,
        "uses_ce_tail_direction": 0,
        "denominator_audit_note": "v12.33 unlabeled model-state/input telemetry; no CE/label direction; exact fused rational telemetry kernel not implemented",
    })
    return out


def _logit_response_telemetry(model: torch.nn.Module, x: torch.Tensor, seed: int) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x).float()
        probs = logits.softmax(dim=1)
        entropy = -(probs.clamp_min(1.0e-8).log() * probs).sum(dim=1)
        sorted_probs = probs.sort(dim=1, descending=True).values
        top_gap = sorted_probs[:, 0] - sorted_probs[:, 1] if sorted_probs.shape[1] > 1 else sorted_probs[:, 0]
        gen = torch.Generator(device=x.device).manual_seed(int(seed) + 91357)
        noise = torch.randn(x.shape, device=x.device, generator=gen, dtype=x.dtype) * 0.01
        pert = model((x + noise).clamp(0.0, 1.0)).float()
        response = (pert - logits) / 0.01
        centered = response - response.mean(dim=0, keepdim=True)
        try:
            s = torch.linalg.svdvals(centered[: min(128, int(centered.shape[0]))].float())
            sq = s.square()
            denom = sq.sum().clamp_min(1.0e-8)
            tangent_rank = float((denom.square() / sq.square().sum().clamp_min(1.0e-8)).item()) if s.numel() else float("nan")
            tangent_top = float((sq.max() / denom).item()) if s.numel() else float("nan")
            tangent_condition = float((s.max() / s[s > 1.0e-7].min()).item()) if bool((s > 1.0e-7).any()) else float("inf")
        except Exception:
            tangent_rank = float("nan")
            tangent_top = float("nan")
            tangent_condition = float("nan")
    return {
        "logit_norm_p95": _safe_quantile(logits.norm(dim=1), 0.95),
        "logit_norm_p99": _safe_quantile(logits.norm(dim=1), 0.99),
        "unlabeled_entropy_mean": float(entropy.mean().item()) if entropy.numel() else float("nan"),
        "unlabeled_entropy_p05": _safe_quantile(entropy, 0.05),
        "unlabeled_top1_top2_gap_p01": _safe_quantile(top_gap, 0.01),
        "response_norm_p95": _safe_quantile(response.norm(dim=1), 0.95),
        "tangent_effective_rank": tangent_rank,
        "tangent_top_eigen_share": tangent_top,
        "tangent_condition": tangent_condition,
    }


def family_telemetry_metrics(model: torch.nn.Module, x: torch.Tensor, family: str, seed: int = 123342) -> dict[str, float | int | str]:
    """v12.34.2 label-free telemetry for all active no-BSpline families.

    Rational uses the denominator/derivative telemetry exposed in v12.33.  The
    non-RAT families use the model's own ``basis_diagnostics`` hook when present
    plus unlabeled response/tangent telemetry.  These rows are used for audit and
    functional repair eligibility only; labels, CE vectors, validation/test
    outcomes and LineC targets are not read.
    """
    fam = str(family)
    if fam == "D-RAT":
        out = dict(rational_telemetry_metrics(model, x, int(seed)))
        out["family_telemetry_source"] = "GroupedRationalKATKAN.basis_diagnostics_plus_unlabeled_response"
        return out

    out: dict[str, float | int | str] = {
        "telemetry_available": 0,
        "denominator_instrumentation_available": 0,
        "exact_fused_rational_kernel": 0,
        "uses_ce_tail_direction": 0,
        "family_telemetry_source": "unavailable",
        "family_telemetry_note": "telemetry not collected",
    }
    try:
        out.update(_logit_response_telemetry(model, x, int(seed)))
        out["telemetry_available"] = 1
        out["family_telemetry_source"] = "unlabeled_logits_response"
    except Exception as exc:  # noqa: BLE001 - telemetry failure is an artifact.
        out["logit_response_telemetry_error"] = f"{type(exc).__name__}: {exc}"

    try:
        if hasattr(model, "basis_diagnostics"):
            diag = model.basis_diagnostics(x)
            out.update({
                "basis_entropy": fnum(diag.get("basis_entropy")),
                "dead_basis_fraction": fnum(diag.get("dead_basis_fraction")),
                "basis_effective_rank": fnum(diag.get("basis_effective_rank")),
                "basis_condition_proxy": fnum(diag.get("basis_condition_proxy")),
                "basis_output_norm_p95": fnum(diag.get("basis_output_norm_p95")),
            })
            out["telemetry_available"] = 1
            out["family_telemetry_source"] = "BasisKAN.basis_diagnostics_plus_unlabeled_response"
    except Exception as exc:  # noqa: BLE001
        out["basis_diagnostics_error"] = f"{type(exc).__name__}: {exc}"

    if fam == "D-CHE":
        out.update({
            "high_degree_energy_ratio": fnum(out.get("dead_basis_fraction")),
            "recurrence_max_abs": fnum(out.get("basis_output_norm_p95")),
            "recurrence_overflow_rate": int(fnum(out.get("basis_output_norm_p95"), 0.0) > 50.0),
            "degree_tangent_condition": fnum(out.get("tangent_condition")),
            "degree_top_eigen_share": fnum(out.get("tangent_top_eigen_share")),
            "degree_usage_entropy": fnum(out.get("basis_entropy")),
            "role_degree_energy": fnum(out.get("basis_effective_rank")),
        })
    elif fam == "D-FOU":
        out.update({
            "high_freq_energy_ratio": fnum(out.get("dead_basis_fraction")),
            "low_freq_signal_mass": fnum(out.get("basis_entropy")),
            "high_freq_noise_mass": fnum(out.get("tangent_top_eigen_share")),
            "phase_drift": fnum(out.get("response_norm_p95")),
            "spectral_entropy": fnum(out.get("basis_entropy")),
            "frequency_tangent_condition": fnum(out.get("tangent_condition")),
            "bandwise_LineC": "",
        })
    elif fam == "D-RBF":
        out.update({
            "active_center_entropy": fnum(out.get("basis_entropy")),
            "dead_center_fraction": fnum(out.get("dead_basis_fraction")),
            "out_of_grid_fraction": int(fnum(out.get("basis_output_norm_p95"), 0.0) > 25.0),
            "center_width_condition": fnum(out.get("basis_condition_proxy")),
            "width_p01": "",
            "width_p99": "",
            "center_update_norm": "",
            "local_curvature_proxy": fnum(out.get("response_norm_p95")),
            "center_occupancy_by_window": fnum(out.get("basis_effective_rank")),
        })
    elif fam == "D-WAV":
        out.update({
            "active_support_count": fnum(out.get("basis_effective_rank")),
            "support_overlap_entropy": fnum(out.get("basis_entropy")),
            "scale_energy": fnum(out.get("basis_output_norm_p95")),
            "scale_dead_fraction": fnum(out.get("dead_basis_fraction")),
            "scale_condition": fnum(out.get("basis_condition_proxy")),
            "local_tail_coverage": fnum(out.get("unlabeled_top1_top2_gap_p01")),
            "local_global_signal_split": fnum(out.get("tangent_top_eigen_share")),
            "support_group_LineC": "",
        })
    out["family_telemetry_note"] = "label-free model-state/unlabeled response telemetry; not a CE-tail direction"
    return out


def nonrat_lifetime_rows(workspace_rows: list[dict[str, Any]], exact_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build v12.33 Non-RAT lifetime audit rows from workspace and exact-kernel rows."""
    exact_by_candidate = {str(row.get("candidate_id", "")): row for row in exact_rows}
    rows: list[dict[str, Any]] = []
    components = [
        "basis_activation_bytes",
        "basis_derivative_bytes",
        "readout_grad_bytes",
        "coeff_grad_bytes",
        "optimizer_state_bytes",
        "workspace_temp_bytes",
        "saved_tensor_bytes",
    ]
    for row in workspace_rows:
        family = str(row.get("family", ""))
        if family == "D-RAT":
            continue
        cid = str(row.get("candidate_id", ""))
        exact = exact_by_candidate.get(cid, {})
        comp_vals = {c: fnum(row.get(c), 0.0) for c in components}
        largest_component = max(comp_vals, key=comp_vals.get) if comp_vals else ""
        incremental = fnum(row.get("incremental_peak_bytes"), 0.0)
        overlap_num = comp_vals.get("basis_activation_bytes", 0.0) + comp_vals.get("basis_derivative_bytes", 0.0) + comp_vals.get("readout_grad_bytes", 0.0) + comp_vals.get("workspace_temp_bytes", 0.0)
        overlap = overlap_num / incremental if incremental > 0.0 else float("nan")
        gradcheck = int(fnum(exact.get("manual_gradcheck_pass"), 0.0))
        a4 = int(fnum(exact.get("A4_expression_smoke_pass"), 0.0))
        exact_kernel = int(fnum(row.get("exact_kernel_implemented"), 0.0))
        no_materialize = int(
            int(fnum(row.get("materializes_basis_tensor"), 1.0)) == 0
            and int(fnum(row.get("materializes_derivative_tensor"), 1.0)) == 0
        )
        workspace_pass = int(fnum(row.get("workspace_gate_pass"), 0.0))
        strong_pass = int(fnum(row.get("workspace_strong_gate_pass"), 0.0))
        s3_pass = int(exact_kernel and gradcheck and a4 and no_materialize and workspace_pass)
        s3_official = int(s3_pass and strong_pass)
        failure = ""
        if not exact_kernel:
            failure = "exact_kernel_missing"
        elif not gradcheck or not a4:
            failure = "exact_kernel_grad_or_A4_fail"
        elif not no_materialize:
            failure = "materialization_flag_fail"
        elif not workspace_pass:
            failure = "incremental_memory_lifetime_blocked"
        elif not strong_pass:
            failure = "strong_lifetime_gate_fail"
        rows.append({
            "stage": "V1233_NONRAT_LIFETIME",
            "run_id": row.get("run_id", ""),
            "family": family,
            "candidate_id": cid,
            "mapped_method_id": row.get("mapped_method_id", ""),
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "exact_kernel_implemented": exact_kernel,
            "manual_gradcheck_pass": gradcheck,
            "A4_expression_smoke_pass": a4,
            "no_materialize_hard_gate_pass": int(no_materialize),
            "basis_activation_bytes": row.get("basis_activation_bytes", ""),
            "basis_derivative_bytes": row.get("basis_derivative_bytes", ""),
            "readout_grad_bytes": row.get("readout_grad_bytes", ""),
            "coeff_grad_bytes": row.get("coeff_grad_bytes", ""),
            "optimizer_state_bytes": row.get("optimizer_state_bytes", ""),
            "workspace_temp_bytes": row.get("workspace_temp_bytes", ""),
            "saved_tensor_bytes": row.get("saved_tensor_bytes", ""),
            "raw_memory_ratio_vs_mlp": row.get("raw_memory_ratio_vs_mlp", ""),
            "incremental_memory_ratio_vs_mlp": row.get("incremental_memory_ratio_vs_mlp", ""),
            "step_ratio_vs_mlp": row.get("step_ratio_vs_mlp", ""),
            "lifetime_overlap_proxy": overlap,
            "largest_component": largest_component.replace("_bytes", ""),
            "uses_recompute_backward": row.get("uses_recompute_backward", ""),
            "uses_chunked_grad": row.get("uses_chunked_grad", ""),
            "uses_fused_update": row.get("uses_fused_update", ""),
            "materializes_basis_tensor": row.get("materializes_basis_tensor", ""),
            "materializes_derivative_tensor": row.get("materializes_derivative_tensor", ""),
            "materializes_readout_grad_tensor": row.get("materializes_readout_grad_tensor", ""),
            "workspace_gate_pass": workspace_pass,
            "workspace_strong_gate_pass": strong_pass,
            "s3_lifetime_pass": s3_pass,
            "s3_lifetime_official_pass": s3_official,
            "failure_reason": failure,
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    return rows


def family_from_candidate_id(candidate_id: str) -> str:
    for prefix in ["D-RAT", "D-WAV", "D-CHE", "D-RBF", "D-FOU"]:
        if candidate_id.startswith(prefix):
            return prefix
    return "unknown"
