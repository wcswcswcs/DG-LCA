"""Classic no-BSpline basis diagnostics shared by experiment runners.

This module intentionally contains reusable mechanics that should not live in a
versioned runner: candidate-family aliases, numeric parsing, memory accounting,
and the MLP reference used for basis-family comparisons. Runners should only
orchestrate datasets and artifacts.
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.nn.functional as F

from dgkan.models.fc_purekan_primitives import MLPBaseline
from dgkan.training.eval import classification_basic

CLASSIC_FAMILY_SPECS = {
    "Rational": "B7b-RationalKAT-flashgroup-G16-h112-linearres-tritonL3",
    "RationalB7me": "B7me-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-logitBatchRMSNormSG150-freezeBackbone-hiddenBias-manualAdamW-L3",
    "RationalB7lp": "B7lp-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3",
    "RationalB7lz": "B7lz-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossPCAR128-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3",
    "RationalB7ma": "B7ma-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossPCAR128-crossZero-b32-logitRMSNormSG-freezeBackbone-hiddenBias-manualAdamW-L3",
    "RationalB7dq": "B7dq-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossZero-b32-readscale010-L3",
    "RationalB7dr": "B7dr-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-readscale010-L3",
    "RationalB7ds": "B7ds-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-readscale025-crossWarm-L3",
    "RationalB7dt": "B7dt-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-readscale010-crossWarm-L3",
    "RationalB7em": "B7em-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairStd-crossZero-b32-freezeBackbone-hiddenBias-L3",
    "RationalB7en": "B7en-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-freezeBackbone-hiddenBias-L3",
    "RationalB7eq": "B7eq-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-logitBias-L3",
    "RationalB7er": "B7er-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-readscale050-freezeBackbone-hiddenBias-logitBias-L3",
    "D25-RationalMemoryCut-readscaleShared": "B7bf-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-readscale010-L3",
    "D26-RationalCouplingLift-noWhiten": "B7eq-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-logitBias-L3",
    "D27-RationalPairNormStopGrad-light": "B7em-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairStd-crossZero-b32-freezeBackbone-hiddenBias-L3",
    "D28-RationalTaskTrajectoryWarmNoExtraMem": "B7ht-RationalKAT-flashgroup-G16-h32-linearresGain19200-linearResRamp09600-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3",
    "D29-RationalB7lpLowMemCouplingMix": "B7df-RationalKAT-flashgroup-G16-h32-linearres-pairReadBucketR136G32-crossZero-L3",
    "D30-RationalCheaperR120Pair": "B7et-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR120-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3",
    "D31-RationalNoPairReadout": "B7ex-RationalKAT-flashgroup-G16-h32-linearresGain19200-freezeBackbone-hiddenBias-L3",
    "D32-RationalBucketG32Manual": "B7fa-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairReadBucketR136G32-crossZero-freezeBackbone-hiddenBias-manualAdamW-L3",
    "D33-RationalNoPairManual": "B7ey-RationalKAT-flashgroup-G16-h32-linearresGain19200-freezeBackbone-hiddenBias-manualAdamW-L3",
    "D34-RationalActivationCheckpointNoReadoutCache": "B7ex-RationalKAT-flashgroup-G16-h32-linearresGain19200-freezeBackbone-hiddenBias-L3",
    "D35-RationalGroupedWorkspaceReuse": "B7fa-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairReadBucketR136G32-crossZero-freezeBackbone-hiddenBias-manualAdamW-L3",
    "D36-RationalReadoutGradRecompute": "B7bf-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-readscale010-L3",
    "D37-RationalDenominatorStateFP16Audit": "B7eq-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-logitBias-L3",
    "D38-RationalLowMemNoPairLineCRepeat": "B7ey-RationalKAT-flashgroup-G16-h32-linearresGain19200-freezeBackbone-hiddenBias-manualAdamW-L3",
    "D39-RationalGroupWorkspaceRecomputeV2": "B7fa-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairReadBucketR136G32-crossZero-freezeBackbone-hiddenBias-manualAdamW-L3",
    "D40-RationalReadoutGradChunked": "B7cv-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-readGradR64-crossWarm-L3",
    "D41-RationalHiddenResLowMemV2": "B7jy-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-hiddenSignSqResidual010-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3",
    "D42-RationalDenStateFP16PlusCheckpoint": "B7eq-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-logitBias-L3",
    "D43-RationalPairNormLineCNoExtraMem": "B7lp-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3",
    "D-RAT1-GroupWorkspaceRecomputeV3": "B7fa-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairReadBucketR136G32-crossZero-freezeBackbone-hiddenBias-manualAdamW-L3",
    "D-RAT2-ReadoutGradChunkedV2": "B7cv-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-readGradR64-crossWarm-L3",
    "D-RAT3-DenStateFP16Checkpoint": "B7eq-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-logitBias-L3",
    "D-RAT4-GroupRationalSharedDenom": "B7lp-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3",
    "D-RAT5-RationalLineCResidualMix": "B7jy-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-hiddenSignSqResidual010-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3",
    "D-RAT6-RationalTaskTrajectoryWarmNoExtraMemV2": "B7ht-RationalKAT-flashgroup-G16-h32-linearresGain19200-linearResRamp09600-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3",
    "D-CHE1-K3-degreeEnergyCap": "B3m-ChebyKAN-K3-h112-tritonL3-gradbuf",
    "D-CHE2-K4-lateEnableHighDegree": "B3f-ChebyKAN-K4-tritonL3-matmulTile",
    "D-CHE3-K4-roleWiseEnergyCap": "B3k-ChebyKAN-K3-h120-tritonL3-gradbuf",
    "D-CHE4-K3K4-lowDegreeResidual": "B3v-ChebyKAN-K3-h112-inputcrossL4P128-linearres-tritonL3-gradbuf",
    "D-CHE5-LineCEnergyMonitor": "B3w-ChebyKAN-K3-h112-inputcrossL4P128-linearres050-tritonL3-gradbuf",
    "D-WAV1-HatLocalK4ScaleStable": "B5h-HatWaveletKAN-local-K4",
    "D-WAV2-TriangleK4OccupancyBalanced": "B5i-HatWaveletKAN-K4-inputcrossL4P128-tritonL3",
    "D-WAV3-HaarLiteLocalSupportDiagnostic": "B5j-HatWaveletKAN-K4-inputrot2L4P128-tritonL3",
    "D-WAV4-ScaleDiversitySmallResidual": "B5k-HatWaveletKAN-K4-inputcrossL4P128-linearres-tritonL3",
    "D-WAV5-LocalTailCoverageMonitor": "B5n-HatWaveletKAN-K4-inputcrossL4P128-linearraw010-tritonL3",
    "D-RBF1-FastKANFixedCenterK4": "B2s-GaussianRBF-stream-K4-recompute",
    "D-RBF2-CompactLocalRBFKactive4": "B2t-GaussianRBF-K4-inputcrossL4P128-tritonL3",
    "D-RBF3-CompactLocalRBFKactive8": "B2w-GaussianRBF-K4-inputrot2L4P256-tritonL3",
    "D-RBF4-TrainStreamQuantileCenters": "B2z-GaussianRBF-K4-orthoInputrot2L4P128-tritonL3",
    "D-RBF5-TriangularBumpApprox": "B2ab-GaussianRBF-K4-orthoInputsqL4P128-tritonL3",
    "D-RBF6-IdentityCompactRBFResidual": "B2v-GaussianRBF-K4-inputcrossL4P128-linearres-tritonL3",
    "D-FOU1-K2LowfreqIdentity": "B4g-FourierKAN-lowfreq-K2-tritonL3-matmulTile",
    "D-FOU2-K4LowfreqLinearResidual": "B4p-FourierKAN-lowfreq-K4-h64-linearres-gemmDirectL3",
    "D-FOU3-LearnedAmplitudeResidual": "B4q-FourierKAN-lowfreq-K4-h48-linearres-gemmDirectL3",
    "D-FOU4-LateEnableK4FromK2": "B4n-FourierKAN-lowfreq-K4-h64-linearres-tritonL3-matmulTile",
    "D-FOU5-MultiScaleLowK": "B4w-FourierKAN-lowfreq-K4-h8-linearres050-tritonL3-matmulTile",
    "Fourier": "B4g-FourierKAN-lowfreq-K2-tritonL3-matmulTile",
    "FourierB4p": "B4p-FourierKAN-lowfreq-K4-h64-linearres-gemmDirectL3",
    "FourierB4q": "B4q-FourierKAN-lowfreq-K4-h48-linearres-gemmDirectL3",
    "FourierB4v": "B4v-FourierKAN-lowfreq-K4-h8-linearres050-gemmDirectL3",
    "FourierB4w": "B4w-FourierKAN-lowfreq-K4-h8-linearres050-tritonL3-matmulTile",
}


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


def incremental_peak_bytes(peak_bytes: float, baseline_bytes: float) -> float:
    if math.isfinite(peak_bytes) and math.isfinite(baseline_bytes):
        return max(0.0, float(peak_bytes) - float(baseline_bytes))
    return float("nan")


def memory_ratio(family_peak: float, mlp_peak: float, fallback: float = float("nan")) -> float:
    denom = fnum(mlp_peak)
    if denom > 0:
        return fnum(family_peak) / denom
    return fallback


def train_mlp_reference(args: Any, dataset: str, seed: int, input_dim: int, output_dim: int, x_train, y_train, x_val, y_val, device) -> dict[str, Any]:
    hidden = int(args.mlp_hidden)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        baseline_alloc = float(torch.cuda.memory_allocated(device))
    else:
        baseline_alloc = float("nan")
    model = MLPBaseline(input_dim, output_dim, hidden, seed + 1224000, device).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    gen = torch.Generator(device=device).manual_seed(seed + 1224100)
    times: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _epoch in range(int(args.epochs)):
        perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen)
        for off in range(0, int(x_train.shape[0]), int(args.batch_size)):
            idx = perm[off:off + int(args.batch_size)]
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
    peak = float(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else float("nan")
    incr_peak = incremental_peak_bytes(peak, baseline_alloc)
    ev = classification_basic(model, x_val, y_val)
    q90 = float(torch.tensor(times).quantile(0.90).item()) if times else float("nan")
    return {
        "mlp_reference_id": f"MLP-h{hidden}-AdamW",
        "mlp_val_acc": ev.get("acc", ""),
        "mlp_NLL": ev.get("NLL", ""),
        "mlp_ECE": ev.get("ECE", ""),
        "mlp_CEp99": ev.get("CEp99", ""),
        "mlp_step_time_q90_ms": q90,
        "mlp_peak_memory_bytes": peak,
        "mlp_memory_baseline_bytes": baseline_alloc,
        "mlp_incremental_peak_memory_bytes": incr_peak,
        "dataset": dataset,
        "seed": seed,
    }
