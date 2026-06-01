#!/usr/bin/env python
"""v14.9 Line D all-basis substrate-only repair.

This runner executes only Non-RAT substrate repair rows from the v14.9 plan.
It does not run FMS proof, controller, reset/action search, or audit-metric
direction. The v14.9 candidate names are mapped to existing no-materialize /
compact primitive IDs so the result is auditable against prior substrate code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_v1223_failclosed_explore_open2_functional_rebuild as v1223  # noqa: E402
from experiments.run_v143_nonrat_compact_task_health_probe import (  # noqa: E402
    fnum,
    load_workspace_rows,
    parse_csv,
    train_compact_candidate,
    train_mlp_for_reference,
    write_rows,
)


REQUIRED = [
    "v149_line_d_substrate_route.json",
    "v149_line_d_substrate_candidate_mapping.csv",
    "v149_line_d_substrate_repair_results.csv",
    "v149_line_d_substrate_repair_linec.csv",
    "v149_line_d_substrate_repair_summary.csv",
    "v149_line_d_substrate_required_manifest.csv",
    "v149_line_d_substrate_no_go_boundary.md",
    "fig_v149_line_d_substrate_matrix.svg",
]


@dataclass(frozen=True)
class SubstrateConfig:
    candidate_id: str
    family: str
    base_candidate_id: str
    description: str
    wavelet_support_repair: str = "none"
    wavelet_role_constraint: str = "none"
    rbf_center_repair: str = "none"
    output_geometry_repair: str = "none"


CONFIGS = [
    SubstrateConfig(
        "D-FOU23-LowFreqIdentityResidualUnified",
        "D-FOU",
        "D-FOU20-LowFreqIdentityResidualHealthSubstrate",
        "low-frequency identity residual unified substrate",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-FOU24-BandwiseSNRWarmupNoFMS",
        "D-FOU",
        "D-FOU16-FrequencyBandDampingSubstrate",
        "bandwise SNR warmup substrate without FMS proof",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-FOU25-PhaseStableBandMixNoHighFreq",
        "D-FOU",
        "D-FOU17-PhaseStabilityCorrectionSubstrate",
        "phase-stable low-frequency band mix",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-FOU26-NoMaterializeLifetimeAuditV2",
        "D-FOU",
        "D-FOU13-FusedReadoutGradNoMaterialize-K2",
        "no-materialize lifetime audit v2",
    ),
    SubstrateConfig(
        "D-FOU27-LowFreqIdentityResidualV2",
        "D-FOU",
        "D-FOU20-LowFreqIdentityResidualHealthSubstrate",
        "v14.11 low-frequency identity residual substrate-only hardening",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-FOU28-BandwiseSNRSafeWarmup",
        "D-FOU",
        "D-FOU16-FrequencyBandDampingSubstrate",
        "v14.11 bandwise SNR-safe warmup substrate-only hardening",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-FOU29-PhaseStableBandMixNoHighFreq",
        "D-FOU",
        "D-FOU17-PhaseStabilityCorrectionSubstrate",
        "v14.11 phase-stable low-frequency band mix substrate-only hardening",
        output_geometry_repair="train_topprob_t025_050_else100",
    ),
    SubstrateConfig(
        "D-FOU30-NoMaterializeLifetimeV3",
        "D-FOU",
        "D-FOU13-FusedReadoutGradNoMaterialize-K2",
        "v14.11 no-materialize lifetime v3 substrate-only hardening",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-FOU31-HighFrequencyQuarantine",
        "D-FOU",
        "D-FOU19-HighFreqNoiseLeakVetoSubstrate",
        "v14.13 high-frequency quarantine substrate-only hardening; train-stream geometry only",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-FOU32-LowFreqIdentityResidualV3",
        "D-FOU",
        "D-FOU20-LowFreqIdentityResidualHealthSubstrate",
        "v14.15 low-frequency identity residual v3 substrate-only acceleration",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-FOU33-BandwiseSNRWarmupV2",
        "D-FOU",
        "D-FOU16-FrequencyBandDampingSubstrate",
        "v14.15 bandwise SNR warmup v2 substrate-only acceleration",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-FOU34-PhaseStableBandMixNoHighFreqV2",
        "D-FOU",
        "D-FOU17-PhaseStabilityCorrectionSubstrate",
        "v14.15 phase-stable band mix without high-frequency branch v2",
        output_geometry_repair="train_topprob_t025_050_else100",
    ),
    SubstrateConfig(
        "D-FOU35-NoMaterializeLifetimeV4",
        "D-FOU",
        "D-FOU13-FusedReadoutGradNoMaterialize-K2",
        "v14.15 no-materialize lifetime v4 substrate-only acceleration",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-FOU36-HighFrequencyQuarantineV2",
        "D-FOU",
        "D-FOU19-HighFreqNoiseLeakVetoSubstrate",
        "v14.15 high-frequency quarantine v2 substrate-only acceleration",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-FOU37-LowFreqIdentityResidualV4",
        "D-FOU",
        "D-FOU20-LowFreqIdentityResidualHealthSubstrate",
        "v15.0 low-frequency identity residual v4 substrate-only acceleration",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-FOU38-BandwiseSecondMomentWarmup",
        "D-FOU",
        "D-FOU16-FrequencyBandDampingSubstrate",
        "v15.0 bandwise second-moment warmup substrate-only acceleration",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-FOU39-PhaseStableCautiousUpdate",
        "D-FOU",
        "D-FOU17-PhaseStabilityCorrectionSubstrate",
        "v15.0 phase-stable cautious update substrate-only acceleration",
        output_geometry_repair="train_topprob_t025_050_else100",
    ),
    SubstrateConfig(
        "D-FOU40-NoMaterializeLifetimeV4",
        "D-FOU",
        "D-FOU13-FusedReadoutGradNoMaterialize-K2",
        "v15.0 no-materialize lifetime v4 substrate-only acceleration",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-FOU41-HighFrequencyQuarantineV2",
        "D-FOU",
        "D-FOU19-HighFreqNoiseLeakVetoSubstrate",
        "v15.0 high-frequency quarantine v2 substrate-only acceleration",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-FOU42-LowFreqResidualV5",
        "D-FOU",
        "D-FOU20-LowFreqIdentityResidualHealthSubstrate",
        "v15.1 low-frequency residual v5 substrate-only acceleration",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-FOU43-BandwiseSecondMomentWarmup",
        "D-FOU",
        "D-FOU16-FrequencyBandDampingSubstrate",
        "v15.1 bandwise second-moment warmup substrate-only acceleration",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-FOU44-PhaseStableLowBandOnly",
        "D-FOU",
        "D-FOU17-PhaseStabilityCorrectionSubstrate",
        "v15.1 phase-stable low-band-only substrate acceleration",
        output_geometry_repair="train_topprob_t025_050_else100",
    ),
    SubstrateConfig(
        "D-FOU45-NoMaterializeLifetimeV4",
        "D-FOU",
        "D-FOU13-FusedReadoutGradNoMaterialize-K2",
        "v15.1 no-materialize lifetime v4 substrate-only acceleration",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-FOU46-HighFreqQuarantineLateEnable",
        "D-FOU",
        "D-FOU19-HighFreqNoiseLeakVetoSubstrate",
        "v15.1 high-frequency quarantine late-enable substrate-only acceleration",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-FOU47-LowFreqIdentityResidualV5",
        "D-FOU",
        "D-FOU20-LowFreqIdentityResidualHealthSubstrate",
        "v15.2.1 low-frequency identity residual v5 substrate-only fallback",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-FOU48-BandwiseSNRWarmupV3",
        "D-FOU",
        "D-FOU16-FrequencyBandDampingSubstrate",
        "v15.2.1 bandwise SNR warmup v3 substrate-only fallback",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-FOU49-PhaseStableBandMixV3",
        "D-FOU",
        "D-FOU17-PhaseStabilityCorrectionSubstrate",
        "v15.2.1 phase-stable band mix v3 substrate-only fallback",
        output_geometry_repair="train_topprob_t025_050_else100",
    ),
    SubstrateConfig(
        "D-FOU50-NoMaterializeLifetimeV3",
        "D-FOU",
        "D-FOU13-FusedReadoutGradNoMaterialize-K2",
        "v15.2.1 no-materialize lifetime v3 substrate-only fallback",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-FOU51-HighFreqQuarantineV2",
        "D-FOU",
        "D-FOU19-HighFreqNoiseLeakVetoSubstrate",
        "v15.2.1 high-frequency quarantine v2 substrate-only fallback",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-FOU52-LowFreqIdentityResidualV5",
        "D-FOU",
        "D-FOU20-LowFreqIdentityResidualHealthSubstrate",
        "v15.3 low-frequency identity residual v5 substrate-only acceleration",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-FOU53-BandwiseSNRWarmupV3",
        "D-FOU",
        "D-FOU16-FrequencyBandDampingSubstrate",
        "v15.3 bandwise SNR warmup v3 substrate-only acceleration",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-FOU54-PhaseStableBandMixV3",
        "D-FOU",
        "D-FOU17-PhaseStabilityCorrectionSubstrate",
        "v15.3 phase-stable band mix v3 substrate-only acceleration",
        output_geometry_repair="train_topprob_t025_050_else100",
    ),
    SubstrateConfig(
        "D-FOU55-NoMaterializeLifetimeV3",
        "D-FOU",
        "D-FOU13-FusedReadoutGradNoMaterialize-K2",
        "v15.3 no-materialize lifetime v3 substrate-only acceleration",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-FOU56-HighFrequencyQuarantineV2",
        "D-FOU",
        "D-FOU19-HighFreqNoiseLeakVetoSubstrate",
        "v15.3 high-frequency quarantine v2 substrate-only acceleration",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-FOU57-LowFreqIdentityResidualV5",
        "D-FOU",
        "D-FOU20-LowFreqIdentityResidualHealthSubstrate",
        "v15.4 low-frequency identity residual v5 substrate-only acceleration",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-FOU58-BandwiseSNRWarmupV3",
        "D-FOU",
        "D-FOU16-FrequencyBandDampingSubstrate",
        "v15.4 bandwise SNR warmup v3 substrate-only acceleration",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-FOU59-PhaseStableBandMixV3",
        "D-FOU",
        "D-FOU17-PhaseStabilityCorrectionSubstrate",
        "v15.4 phase-stable band mix v3 substrate-only acceleration",
        output_geometry_repair="train_topprob_t025_050_else100",
    ),
    SubstrateConfig(
        "D-FOU60-NoMaterializeLifetimeV3",
        "D-FOU",
        "D-FOU13-FusedReadoutGradNoMaterialize-K2",
        "v15.4 no-materialize lifetime v3 substrate-only acceleration",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-FOU61-HighFrequencyQuarantineV3",
        "D-FOU",
        "D-FOU19-HighFreqNoiseLeakVetoSubstrate",
        "v15.4 high-frequency quarantine v3 substrate-only acceleration",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-FOU62-LowFreqIdentityResidualV5",
        "D-FOU",
        "D-FOU20-LowFreqIdentityResidualHealthSubstrate",
        "v15.5 low-frequency identity residual v5 substrate-only acceleration",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-FOU63-BandwiseConsensusMetric",
        "D-FOU",
        "D-FOU16-FrequencyBandDampingSubstrate",
        "v15.5 bandwise consensus metric substrate-only acceleration",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-FOU64-PhaseStableBandMixV2",
        "D-FOU",
        "D-FOU17-PhaseStabilityCorrectionSubstrate",
        "v15.5 phase-stable band mix v2 substrate-only acceleration",
        output_geometry_repair="train_topprob_t025_050_else100",
    ),
    SubstrateConfig(
        "D-FOU65-NoMaterializeLifetimeV3",
        "D-FOU",
        "D-FOU13-FusedReadoutGradNoMaterialize-K2",
        "v15.5 no-materialize lifetime v3 substrate-only acceleration",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-FOU66-HighFrequencyQuarantineNoAuditDirection",
        "D-FOU",
        "D-FOU19-HighFreqNoiseLeakVetoSubstrate",
        "v15.5 high-frequency quarantine substrate-only acceleration; no audit-metric direction",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-FOU67-LowFreqIdentityResidualV5",
        "D-FOU",
        "D-FOU20-LowFreqIdentityResidualHealthSubstrate",
        "v15.6 low-frequency identity residual v5 substrate-only acceleration",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-FOU68-BandwiseConsensusMetricV2",
        "D-FOU",
        "D-FOU16-FrequencyBandDampingSubstrate",
        "v15.6 bandwise consensus metric v2 substrate-only acceleration",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-FOU69-PhaseStableBandMixV3",
        "D-FOU",
        "D-FOU17-PhaseStabilityCorrectionSubstrate",
        "v15.6 phase-stable band mix v3 substrate-only acceleration",
        output_geometry_repair="train_topprob_t025_050_else100",
    ),
    SubstrateConfig(
        "D-FOU70-HighFrequencyQuarantineV2",
        "D-FOU",
        "D-FOU19-HighFreqNoiseLeakVetoSubstrate",
        "v15.6 high-frequency quarantine v2 substrate-only acceleration",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-FOU71-NoMaterializeLifetimeV4",
        "D-FOU",
        "D-FOU13-FusedReadoutGradNoMaterialize-K2",
        "v15.6 no-materialize lifetime v4 substrate-only acceleration",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-FOU72-LowFreqIdentityResidualV5",
        "D-FOU",
        "D-FOU20-LowFreqIdentityResidualHealthSubstrate",
        "v15.7 low-frequency identity residual v5 carrier substrate-only acceleration",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-FOU73-BandwiseConsensusMetricV2",
        "D-FOU",
        "D-FOU16-FrequencyBandDampingSubstrate",
        "v15.7 bandwise consensus metric v2 carrier substrate-only acceleration",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-FOU74-PhaseStableBandMixV2",
        "D-FOU",
        "D-FOU17-PhaseStabilityCorrectionSubstrate",
        "v15.7 phase-stable band mix v2 carrier substrate-only acceleration",
        output_geometry_repair="train_topprob_t025_050_else100",
    ),
    SubstrateConfig(
        "D-FOU75-NoMaterializeLifetimeV4",
        "D-FOU",
        "D-FOU13-FusedReadoutGradNoMaterialize-K2",
        "v15.7 no-materialize lifetime v4 carrier substrate-only acceleration",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-FOU76-HighFrequencyQuarantineV2",
        "D-FOU",
        "D-FOU19-HighFreqNoiseLeakVetoSubstrate",
        "v15.7 high-frequency quarantine v2 carrier substrate-only acceleration",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-FOU77-LowFreqIdentityResidualV6",
        "D-FOU",
        "D-FOU20-LowFreqIdentityResidualHealthSubstrate",
        "v15.8 low-frequency identity residual v6 dynamics substrate-only acceleration",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-FOU78-BandwiseConsensusMetricV3",
        "D-FOU",
        "D-FOU16-FrequencyBandDampingSubstrate",
        "v15.8 bandwise consensus metric v3 dynamics substrate-only acceleration",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-FOU79-PhaseStableBandMixV3",
        "D-FOU",
        "D-FOU17-PhaseStabilityCorrectionSubstrate",
        "v15.8 phase-stable band mix v3 dynamics substrate-only acceleration",
        output_geometry_repair="train_topprob_t025_050_else100",
    ),
    SubstrateConfig(
        "D-FOU80-NoMaterializeLifetimeV5",
        "D-FOU",
        "D-FOU13-FusedReadoutGradNoMaterialize-K2",
        "v15.8 no-materialize lifetime v5 dynamics substrate-only acceleration",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-FOU81-HighFrequencyQuarantineV3",
        "D-FOU",
        "D-FOU19-HighFreqNoiseLeakVetoSubstrate",
        "v15.8 high-frequency quarantine v3 dynamics substrate-only acceleration",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-FOU82-LowFreqIdentityResidualV5",
        "D-FOU",
        "D-FOU20-LowFreqIdentityResidualHealthSubstrate",
        "v15.9 low-frequency identity residual v5 multi-line substrate-only acceleration",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-FOU83-BandwiseSNRWarmupV5",
        "D-FOU",
        "D-FOU16-FrequencyBandDampingSubstrate",
        "v15.9 bandwise SNR warmup v5 multi-line substrate-only acceleration",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-FOU84-PhaseStableBandMixV5",
        "D-FOU",
        "D-FOU17-PhaseStabilityCorrectionSubstrate",
        "v15.9 phase-stable band mix v5 multi-line substrate-only acceleration",
        output_geometry_repair="train_topprob_t025_050_else100",
    ),
    SubstrateConfig(
        "D-FOU85-NoMaterializeLifetimeV5",
        "D-FOU",
        "D-FOU13-FusedReadoutGradNoMaterialize-K2",
        "v15.9 no-materialize lifetime v5 multi-line substrate-only acceleration",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-FOU86-HighFrequencyQuarantineV5",
        "D-FOU",
        "D-FOU19-HighFreqNoiseLeakVetoSubstrate",
        "v15.9 high-frequency quarantine v5 multi-line substrate-only acceleration",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-FOU87-LowFreqIdentityResidualV6",
        "D-FOU",
        "D-FOU20-LowFreqIdentityResidualHealthSubstrate",
        "v16.0 low-frequency identity residual v6 dynamic-debt substrate-only acceleration",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-FOU88-BandwiseSNRWarmupV6",
        "D-FOU",
        "D-FOU16-FrequencyBandDampingSubstrate",
        "v16.0 bandwise SNR warmup v6 dynamic-debt substrate-only acceleration",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-FOU89-PhaseStableBandMixV6",
        "D-FOU",
        "D-FOU17-PhaseStabilityCorrectionSubstrate",
        "v16.0 phase-stable band mix v6 dynamic-debt substrate-only acceleration",
        output_geometry_repair="train_topprob_t025_050_else100",
    ),
    SubstrateConfig(
        "D-FOU90-NoMaterializeLifetimeV6",
        "D-FOU",
        "D-FOU13-FusedReadoutGradNoMaterialize-K2",
        "v16.0 no-materialize lifetime v6 dynamic-debt substrate-only acceleration",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-FOU91-HighFrequencyQuarantineV6",
        "D-FOU",
        "D-FOU19-HighFreqNoiseLeakVetoSubstrate",
        "v16.0 high-frequency quarantine v6 dynamic-debt substrate-only acceleration",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-WAV23-TriangularSupportStableV2",
        "D-WAV",
        "D-WAV16-SupportStableHatHealthSubstrate",
        "triangular support stable wavelet v2",
        wavelet_support_repair="quantile_scale075",
        wavelet_role_constraint="linear_readout_grad050",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-WAV24-ScaleOccupancyHardeningV2",
        "D-WAV",
        "D-WAV11-ScaleEnergyBalanceSubstrate",
        "scale occupancy hardening v2",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad025",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-WAV25-LocalTailCoverageWithoutAuditMetric",
        "D-WAV",
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "local tail coverage from train-stream geometry only",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad050",
        output_geometry_repair="train_topprob_t020_050_else100",
    ),
    SubstrateConfig(
        "D-WAV26-ReservoirStableTrainEntropyGeometry",
        "D-WAV",
        "D-WAV15-ScaleDiversityTransportSubstrate",
        "reservoir-stable train-entropy output geometry",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad010",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-RBF23-CompactBumpIdentityResidual",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "compact bump identity residual",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-RBF24-ActiveCenterOccupancyNoDense",
        "D-RBF",
        "D-RBF12-CenterOccupancyRebalanceSubstrate",
        "active center occupancy without dense materialization",
        rbf_center_repair="quantile",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-RBF25-WidthConditionGuardNoTaskBranch",
        "D-RBF",
        "D-RBF13-WidthConditionGuardSubstrate",
        "width condition guard without task branch",
        rbf_center_repair="quantile_width075",
    ),
    SubstrateConfig(
        "D-RBF26-FastKANGaussianLocalK4NoDense",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "FastKAN gaussian local K4 no dense path",
        rbf_center_repair="quantile_width075",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-RBF26-ActiveCenterOccupancyV2",
        "D-RBF",
        "D-RBF12-CenterOccupancyRebalanceSubstrate",
        "v14.11 active-center occupancy v2 substrate-only hardening",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-RBF27-WidthConditionIdentityResidual",
        "D-RBF",
        "D-RBF13-WidthConditionGuardSubstrate",
        "v14.11 width-condition identity residual substrate-only hardening",
        rbf_center_repair="quantile_width075",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-RBF28-CompactBumpNoDenseMaterialization",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "v14.11 compact bump no-dense-materialization substrate-only hardening",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-RBF29-GaussianLocalK4TaskHealth",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "v14.11 Gaussian local K4 task-health substrate-only hardening",
        rbf_center_repair="quantile_width075",
        output_geometry_repair="train_topprob_t025_050_else100",
    ),
    SubstrateConfig(
        "D-RBF30-ActiveCenterOccupancyV3",
        "D-RBF",
        "D-RBF12-CenterOccupancyRebalanceSubstrate",
        "v14.15 active-center occupancy v3 substrate-only acceleration",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-RBF31-WidthConditionIdentityResidualV2",
        "D-RBF",
        "D-RBF13-WidthConditionGuardSubstrate",
        "v14.15 width-condition identity residual v2 substrate-only acceleration",
        rbf_center_repair="quantile_width075",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-RBF32-CompactBumpNoDenseMaterializationV2",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "v14.15 compact bump no-dense-materialization v2 substrate-only acceleration",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-RBF33-GaussianLocalK4TaskHealthV2",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "v14.15 gaussian local K4 task-health v2 substrate-only acceleration",
        rbf_center_repair="quantile_width075",
        output_geometry_repair="train_topprob_t025_050_else100",
    ),
    SubstrateConfig(
        "D-RBF34-CenterOccupancyWarmupNoTaskBranch",
        "D-RBF",
        "D-RBF12-CenterOccupancyRebalanceSubstrate",
        "v14.15 center occupancy warmup without dataset/task branch",
        rbf_center_repair="quantile_width100",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-RBF35-ActiveCenterSecondMoment",
        "D-RBF",
        "D-RBF12-CenterOccupancyRebalanceSubstrate",
        "v15.0 active-center second-moment substrate-only acceleration",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-RBF36-WidthConditionDecoupledDecay",
        "D-RBF",
        "D-RBF13-WidthConditionGuardSubstrate",
        "v15.0 width-condition decoupled-decay substrate-only acceleration",
        rbf_center_repair="quantile_width075",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-RBF37-CompactBumpIdentityResidualV2",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "v15.0 compact-bump identity residual v2 substrate-only acceleration",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-RBF38-GaussianLocalK4TaskHealthV2",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "v15.0 Gaussian local K4 task-health v2 substrate-only acceleration",
        rbf_center_repair="quantile_width075",
        output_geometry_repair="train_topprob_t025_050_else100",
    ),
    SubstrateConfig(
        "D-RBF39-NoDenseCenterUpdate",
        "D-RBF",
        "D-RBF12-CenterOccupancyRebalanceSubstrate",
        "v15.0 no-dense center update substrate-only acceleration",
        rbf_center_repair="quantile_width100",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-RBF40-CompactBumpIdentityResidualV2",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "v15.1 compact-bump identity residual v2 substrate-only acceleration",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-RBF41-ActiveCenterOccupancySecondMoment",
        "D-RBF",
        "D-RBF12-CenterOccupancyRebalanceSubstrate",
        "v15.1 active-center occupancy second-moment substrate-only acceleration",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-RBF42-WidthFloorTrustRegion",
        "D-RBF",
        "D-RBF13-WidthConditionGuardSubstrate",
        "v15.1 width-floor trust-region substrate-only acceleration",
        rbf_center_repair="quantile_width075",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-RBF43-GaussianLocalK4NoDenseV2",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "v15.1 Gaussian local K4 no-dense v2 substrate-only acceleration",
        rbf_center_repair="quantile_width075",
        output_geometry_repair="train_topprob_t025_050_else100",
    ),
    SubstrateConfig(
        "D-RBF44-CenterReadoutDecoupledWarmup",
        "D-RBF",
        "D-RBF12-CenterOccupancyRebalanceSubstrate",
        "v15.1 center readout decoupled warmup substrate-only acceleration",
        rbf_center_repair="quantile_width100",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-RBF45-ActiveCenterOccupancyV3",
        "D-RBF",
        "D-RBF12-CenterOccupancyRebalanceSubstrate",
        "v15.2.1 active-center occupancy v3 substrate-only fallback",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-RBF46-WidthConditionGuardV3",
        "D-RBF",
        "D-RBF13-WidthConditionGuardSubstrate",
        "v15.2.1 width-condition guard v3 substrate-only fallback",
        rbf_center_repair="quantile_width075",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-RBF47-CompactBumpNoDenseV3",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "v15.2.1 compact bump no-dense v3 substrate-only fallback",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-RBF48-IdentityResidualV3",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "v15.2.1 identity residual v3 substrate-only fallback",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-RBF49-GaussianLocalK4TaskHealth",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "v15.2.1 Gaussian local K4 task-health substrate-only fallback",
        rbf_center_repair="quantile_width075",
        output_geometry_repair="train_topprob_t025_050_else100",
    ),
    SubstrateConfig(
        "D-RBF50-CompactBumpIdentityResidualV4",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "v15.3 compact bump identity residual v4 substrate-only acceleration",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-RBF51-ActiveCenterOccupancyRepairV3",
        "D-RBF",
        "D-RBF12-CenterOccupancyRebalanceSubstrate",
        "v15.3 active-center occupancy repair v3 substrate-only acceleration",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-RBF52-WidthConditionGuardV3",
        "D-RBF",
        "D-RBF13-WidthConditionGuardSubstrate",
        "v15.3 width-condition guard v3 substrate-only acceleration",
        rbf_center_repair="quantile_width075",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-RBF53-GaussianLocalK4NoDenseV2",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "v15.3 Gaussian local K4 no-dense v2 substrate-only acceleration",
        rbf_center_repair="quantile_width075",
        output_geometry_repair="train_topprob_t025_050_else100",
    ),
    SubstrateConfig(
        "D-RBF54-CenterSNRWarmupV2",
        "D-RBF",
        "D-RBF12-CenterOccupancyRebalanceSubstrate",
        "v15.3 center SNR warmup v2 substrate-only acceleration",
        rbf_center_repair="quantile_width100",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-RBF55-CompactBumpIdentityResidualV4",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "v15.4 compact bump identity residual v4 substrate-only acceleration",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-RBF56-ActiveCenterOccupancyRepairV4",
        "D-RBF",
        "D-RBF12-CenterOccupancyRebalanceSubstrate",
        "v15.4 active-center occupancy repair v4 substrate-only acceleration",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-RBF57-WidthConditionGuardV4",
        "D-RBF",
        "D-RBF13-WidthConditionGuardSubstrate",
        "v15.4 width-condition guard v4 substrate-only acceleration",
        rbf_center_repair="quantile_width075",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-RBF58-GaussianLocalK4NoDenseV3",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "v15.4 Gaussian local K4 no-dense v3 substrate-only acceleration",
        rbf_center_repair="quantile_width075",
        output_geometry_repair="train_topprob_t025_050_else100",
    ),
    SubstrateConfig(
        "D-RBF59-CenterSNRWarmupV2",
        "D-RBF",
        "D-RBF12-CenterOccupancyRebalanceSubstrate",
        "v15.4 center SNR warmup v2 substrate-only acceleration",
        rbf_center_repair="quantile_width100",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-RBF60-CompactBumpIdentityResidualV2",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "v15.5 compact bump identity residual v2 substrate-only acceleration",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-RBF61-ActiveCenterConsensusOccupancy",
        "D-RBF",
        "D-RBF12-CenterOccupancyRebalanceSubstrate",
        "v15.5 active-center consensus occupancy substrate-only acceleration",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-RBF62-WidthConditionGuardV2",
        "D-RBF",
        "D-RBF13-WidthConditionGuardSubstrate",
        "v15.5 width-condition guard v2 substrate-only acceleration",
        rbf_center_repair="quantile_width075",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-RBF63-GaussianLocalK4NoDenseMaterialization",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "v15.5 Gaussian local K4 no-dense materialization substrate-only acceleration",
        rbf_center_repair="quantile_width075",
        output_geometry_repair="train_topprob_t025_050_else100",
    ),
    SubstrateConfig(
        "D-RBF64-CenterDropoutNoTaskBranchDiagnostic",
        "D-RBF",
        "D-RBF12-CenterOccupancyRebalanceSubstrate",
        "v15.5 center dropout diagnostic substrate-only acceleration; no dataset/task branch",
        rbf_center_repair="quantile_width100",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-RBF65-CompactBumpIdentityResidualV3",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "v15.6 compact bump identity residual v3 substrate-only acceleration",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-RBF66-ActiveCenterOccupancyRepairV3",
        "D-RBF",
        "D-RBF12-CenterOccupancyRebalanceSubstrate",
        "v15.6 active-center occupancy repair v3 substrate-only acceleration",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-RBF67-WidthConditionGuardV3",
        "D-RBF",
        "D-RBF13-WidthConditionGuardSubstrate",
        "v15.6 width-condition guard v3 substrate-only acceleration",
        rbf_center_repair="quantile_width075",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-RBF68-GaussianLocalK4NoDenseMaterializationV2",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "v15.6 Gaussian local K4 no-dense materialization v2 substrate-only acceleration",
        rbf_center_repair="quantile_width075",
        output_geometry_repair="train_topprob_t025_050_else100",
    ),
    SubstrateConfig(
        "D-RBF69-CenterSNRWarmupV2",
        "D-RBF",
        "D-RBF12-CenterOccupancyRebalanceSubstrate",
        "v15.6 center SNR warmup v2 substrate-only acceleration",
        rbf_center_repair="quantile_width100",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-RBF70-ActiveCenterOccupancyV4",
        "D-RBF",
        "D-RBF12-CenterOccupancyRebalanceSubstrate",
        "v15.7 active-center occupancy v4 carrier substrate-only acceleration",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-RBF71-WidthConditionGuardV4",
        "D-RBF",
        "D-RBF13-WidthConditionGuardSubstrate",
        "v15.7 width-condition guard v4 carrier substrate-only acceleration",
        rbf_center_repair="quantile_width075",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-RBF72-CompactBumpNoDenseV4",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "v15.7 compact bump no-dense v4 carrier substrate-only acceleration",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-RBF73-GaussianLocalK4TaskHealthV2",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "v15.7 Gaussian local K4 task-health v2 carrier substrate-only acceleration",
        rbf_center_repair="quantile_width075",
        output_geometry_repair="train_topprob_t025_050_else100",
    ),
    SubstrateConfig(
        "D-RBF74-CenterSplitConsensusMetric",
        "D-RBF",
        "D-RBF12-CenterOccupancyRebalanceSubstrate",
        "v15.7 center split-consensus metric carrier substrate-only acceleration",
        rbf_center_repair="quantile_width100",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-RBF75-ActiveCenterOccupancyV5",
        "D-RBF",
        "D-RBF12-CenterOccupancyRebalanceSubstrate",
        "v15.8 active-center occupancy v5 dynamics substrate-only acceleration",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-RBF76-WidthConditionGuardV5",
        "D-RBF",
        "D-RBF13-WidthConditionGuardSubstrate",
        "v15.8 width-condition guard v5 dynamics substrate-only acceleration",
        rbf_center_repair="quantile_width075",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-RBF77-CompactBumpNoDenseV5",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "v15.8 compact bump no-dense v5 dynamics substrate-only acceleration",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-RBF78-GaussianLocalK4TaskHealthV3",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "v15.8 Gaussian local K4 task-health v3 dynamics substrate-only acceleration",
        rbf_center_repair="quantile_width075",
        output_geometry_repair="train_topprob_t025_050_else100",
    ),
    SubstrateConfig(
        "D-RBF79-CenterSplitConsensusMetricV2",
        "D-RBF",
        "D-RBF12-CenterOccupancyRebalanceSubstrate",
        "v15.8 center split-consensus metric v2 dynamics substrate-only acceleration",
        rbf_center_repair="quantile_width100",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-RBF80-ActiveCenterOccupancyV5",
        "D-RBF",
        "D-RBF12-CenterOccupancyRebalanceSubstrate",
        "v15.9 active-center occupancy v5 multi-line substrate-only acceleration",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-RBF81-WidthConditionGuardV5",
        "D-RBF",
        "D-RBF13-WidthConditionGuardSubstrate",
        "v15.9 width-condition guard v5 multi-line substrate-only acceleration",
        rbf_center_repair="quantile_width075",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-RBF82-CompactBumpNoDenseV5",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "v15.9 compact bump no-dense v5 multi-line substrate-only acceleration",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-RBF83-GaussianLocalK4TaskHealthV5",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "v15.9 Gaussian local K4 task-health v5 multi-line substrate-only acceleration",
        rbf_center_repair="quantile_width075",
        output_geometry_repair="train_topprob_t025_050_else100",
    ),
    SubstrateConfig(
        "D-RBF84-IdentityResidualWidthWarmupV5",
        "D-RBF",
        "D-RBF13-WidthConditionGuardSubstrate",
        "v15.9 identity residual width warmup v5 multi-line substrate-only acceleration",
        rbf_center_repair="quantile_width100",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-RBF85-ActiveCenterOccupancyV6",
        "D-RBF",
        "D-RBF12-CenterOccupancyRebalanceSubstrate",
        "v16.0 active-center occupancy v6 dynamic-debt substrate-only acceleration",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-RBF86-WidthConditionGuardV6",
        "D-RBF",
        "D-RBF13-WidthConditionGuardSubstrate",
        "v16.0 width-condition guard v6 dynamic-debt substrate-only acceleration",
        rbf_center_repair="quantile_width075",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-RBF87-CompactBumpNoDenseV6",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "v16.0 compact bump no-dense v6 dynamic-debt substrate-only acceleration",
        rbf_center_repair="quantile_width125",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-RBF88-GaussianLocalK4TaskHealthV6",
        "D-RBF",
        "D-RBF17-CompactCapacityK4HealthSubstrate",
        "v16.0 Gaussian local K4 task-health v6 dynamic-debt substrate-only acceleration",
        rbf_center_repair="quantile_width075",
        output_geometry_repair="train_topprob_t025_050_else100",
    ),
    SubstrateConfig(
        "D-RBF89-IdentityResidualWidthWarmupV6",
        "D-RBF",
        "D-RBF13-WidthConditionGuardSubstrate",
        "v16.0 identity residual width warmup v6 dynamic-debt substrate-only acceleration",
        rbf_center_repair="quantile_width100",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-WAV25-TriangularSupportV3",
        "D-WAV",
        "D-WAV16-SupportStableHatHealthSubstrate",
        "v14.11 triangular support v3 substrate-only hardening",
        wavelet_support_repair="quantile_scale075",
        wavelet_role_constraint="linear_readout_grad050",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-WAV26-ScaleOccupancyNoTailTarget",
        "D-WAV",
        "D-WAV11-ScaleEnergyBalanceSubstrate",
        "v14.11 scale occupancy without tail target substrate-only hardening",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad025",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-WAV27-LocalSupportOverlapDamping",
        "D-WAV",
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "v14.11 local support overlap damping substrate-only hardening",
        wavelet_support_repair="scale075",
        wavelet_role_constraint="linear_readout_grad010",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-WAV28-LocalTailCoverageAudit",
        "D-WAV",
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "v14.13 local tail coverage audit substrate-only hardening; no tail metric direction",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad010",
        output_geometry_repair="train_topprob_t020_050_else100",
    ),
    SubstrateConfig(
        "D-WAV29-TriangularSupportV4",
        "D-WAV",
        "D-WAV16-SupportStableHatHealthSubstrate",
        "v14.15 triangular support v4 low-budget substrate-only monitor",
        wavelet_support_repair="quantile_scale075",
        wavelet_role_constraint="linear_readout_grad050",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-WAV30-ScaleOccupancyNoTailTargetV2",
        "D-WAV",
        "D-WAV11-ScaleEnergyBalanceSubstrate",
        "v14.15 scale occupancy without tail target v2 low-budget monitor",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad025",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-WAV31-LocalSupportOverlapDampingV2",
        "D-WAV",
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "v14.15 local support overlap damping v2 low-budget monitor",
        wavelet_support_repair="scale075",
        wavelet_role_constraint="linear_readout_grad010",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-WAV32-LocalTailCoverageAuditV2",
        "D-WAV",
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "v14.15 local tail coverage audit v2 low-budget monitor; no audit-metric direction",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad010",
        output_geometry_repair="train_topprob_t020_050_else100",
    ),
    SubstrateConfig(
        "D-WAV33-TriangularSupportV4",
        "D-WAV",
        "D-WAV16-SupportStableHatHealthSubstrate",
        "v15.0 triangular support v4 low-budget substrate-only monitor",
        wavelet_support_repair="quantile_scale075",
        wavelet_role_constraint="linear_readout_grad050",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-WAV34-ScaleSecondMomentOccupancy",
        "D-WAV",
        "D-WAV11-ScaleEnergyBalanceSubstrate",
        "v15.0 scale second-moment occupancy low-budget substrate-only monitor",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad025",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-WAV35-SupportOverlapCautiousUpdate",
        "D-WAV",
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "v15.0 support-overlap cautious update low-budget substrate-only monitor",
        wavelet_support_repair="scale075",
        wavelet_role_constraint="linear_readout_grad010",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-WAV36-LocalTailCoverageAuditOnly",
        "D-WAV",
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "v15.0 local tail coverage audit-only low-budget monitor; no audit-metric direction",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad010",
        output_geometry_repair="train_topprob_t020_050_else100",
    ),
    SubstrateConfig(
        "D-WAV37-TriangularSupportV5",
        "D-WAV",
        "D-WAV16-SupportStableHatHealthSubstrate",
        "v15.1 triangular support v5 low-budget substrate-only monitor",
        wavelet_support_repair="quantile_scale075",
        wavelet_role_constraint="linear_readout_grad050",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-WAV38-ScaleOccupancySecondMoment",
        "D-WAV",
        "D-WAV11-ScaleEnergyBalanceSubstrate",
        "v15.1 scale occupancy second-moment low-budget monitor",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad025",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-WAV39-LocalSupportOverlapTrust",
        "D-WAV",
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "v15.1 local support overlap trust low-budget monitor",
        wavelet_support_repair="scale075",
        wavelet_role_constraint="linear_readout_grad010",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-WAV40-FineScaleLateEnable",
        "D-WAV",
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "v15.1 fine-scale late-enable low-budget monitor; no audit-metric direction",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad010",
        output_geometry_repair="train_topprob_t020_050_else100",
    ),
    SubstrateConfig(
        "D-WAV41-TriangularSupportV5",
        "D-WAV",
        "D-WAV16-SupportStableHatHealthSubstrate",
        "v15.2.1 triangular support v5 low-budget substrate fallback",
        wavelet_support_repair="quantile_scale075",
        wavelet_role_constraint="linear_readout_grad050",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-WAV42-ScaleOccupancyV4",
        "D-WAV",
        "D-WAV11-ScaleEnergyBalanceSubstrate",
        "v15.2.1 scale occupancy v4 low-budget substrate fallback",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad025",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-WAV43-SupportOverlapDampingV3",
        "D-WAV",
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "v15.2.1 support overlap damping v3 low-budget substrate fallback",
        wavelet_support_repair="scale075",
        wavelet_role_constraint="linear_readout_grad010",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-WAV44-LocalTailCoverageAudit",
        "D-WAV",
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "v15.2.1 local tail coverage audit low-budget substrate fallback; no audit-metric direction",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad010",
        output_geometry_repair="train_topprob_t020_050_else100",
    ),
    SubstrateConfig(
        "D-WAV45-TriangularSupportV5",
        "D-WAV",
        "D-WAV16-SupportStableHatHealthSubstrate",
        "v15.3 triangular support v5 low-budget substrate acceleration",
        wavelet_support_repair="quantile_scale075",
        wavelet_role_constraint="linear_readout_grad050",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-WAV46-ScaleOccupancyHardeningV3",
        "D-WAV",
        "D-WAV11-ScaleEnergyBalanceSubstrate",
        "v15.3 scale occupancy hardening v3 low-budget substrate acceleration",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad025",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-WAV47-SupportOverlapDampingV3",
        "D-WAV",
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "v15.3 support-overlap damping v3 low-budget substrate acceleration",
        wavelet_support_repair="scale075",
        wavelet_role_constraint="linear_readout_grad010",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-WAV48-LocalTailCoverageAuditV2",
        "D-WAV",
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "v15.3 local tail coverage audit v2 low-budget substrate acceleration; no audit-metric direction",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad010",
        output_geometry_repair="train_topprob_t020_050_else100",
    ),
    SubstrateConfig(
        "D-WAV49-TriangularSupportV5",
        "D-WAV",
        "D-WAV16-SupportStableHatHealthSubstrate",
        "v15.4 triangular support v5 low-budget substrate acceleration",
        wavelet_support_repair="quantile_scale075",
        wavelet_role_constraint="linear_readout_grad050",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-WAV50-ScaleOccupancyV4",
        "D-WAV",
        "D-WAV11-ScaleEnergyBalanceSubstrate",
        "v15.4 scale occupancy v4 low-budget substrate acceleration",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad025",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-WAV51-SupportOverlapDampingV3",
        "D-WAV",
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "v15.4 support-overlap damping v3 low-budget substrate acceleration",
        wavelet_support_repair="scale075",
        wavelet_role_constraint="linear_readout_grad010",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-WAV52-LocalTailCoverageAuditV3",
        "D-WAV",
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "v15.4 local tail coverage audit v3 low-budget substrate acceleration; no audit-metric direction",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad010",
        output_geometry_repair="train_topprob_t020_050_else100",
    ),
    SubstrateConfig(
        "D-WAV53-TriangularSupportV5",
        "D-WAV",
        "D-WAV16-SupportStableHatHealthSubstrate",
        "v15.5 triangular support v5 low-budget substrate acceleration",
        wavelet_support_repair="quantile_scale075",
        wavelet_role_constraint="linear_readout_grad050",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-WAV54-ScaleOccupancyConsensus",
        "D-WAV",
        "D-WAV11-ScaleEnergyBalanceSubstrate",
        "v15.5 scale occupancy consensus low-budget substrate acceleration",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad025",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-WAV55-SupportOverlapDampingV2",
        "D-WAV",
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "v15.5 support-overlap damping v2 low-budget substrate acceleration",
        wavelet_support_repair="scale075",
        wavelet_role_constraint="linear_readout_grad010",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-WAV56-LocalTailCoverageAuditOnly",
        "D-WAV",
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "v15.5 local tail coverage audit-only low-budget substrate acceleration; no audit-metric direction",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad010",
        output_geometry_repair="train_topprob_t020_050_else100",
    ),
    SubstrateConfig(
        "D-WAV57-TriangularSupportV5",
        "D-WAV",
        "D-WAV16-SupportStableHatHealthSubstrate",
        "v15.6 triangular support v5 low-budget substrate acceleration",
        wavelet_support_repair="quantile_scale075",
        wavelet_role_constraint="linear_readout_grad050",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-WAV58-ScaleOccupancyHardeningV3",
        "D-WAV",
        "D-WAV11-ScaleEnergyBalanceSubstrate",
        "v15.6 scale occupancy hardening v3 low-budget substrate acceleration",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad025",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-WAV59-SupportOverlapDampingV3",
        "D-WAV",
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "v15.6 support-overlap damping v3 low-budget substrate acceleration",
        wavelet_support_repair="scale075",
        wavelet_role_constraint="linear_readout_grad010",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-WAV60-LocalTailCoverageAuditV2",
        "D-WAV",
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "v15.6 local tail coverage audit v2 low-budget substrate acceleration; no audit-metric direction",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad010",
        output_geometry_repair="train_topprob_t020_050_else100",
    ),
    SubstrateConfig(
        "D-WAV61-TriangularSupportV5",
        "D-WAV",
        "D-WAV16-SupportStableHatHealthSubstrate",
        "v15.7 triangular support v5 carrier substrate-only acceleration",
        wavelet_support_repair="quantile_scale075",
        wavelet_role_constraint="linear_readout_grad050",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-WAV62-ScaleOccupancyV3",
        "D-WAV",
        "D-WAV11-ScaleEnergyBalanceSubstrate",
        "v15.7 scale occupancy v3 carrier substrate-only acceleration",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad025",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-WAV63-SupportOverlapDampingV3",
        "D-WAV",
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "v15.7 support-overlap damping v3 carrier substrate-only acceleration",
        wavelet_support_repair="scale075",
        wavelet_role_constraint="linear_readout_grad010",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-WAV64-LocalTailCoverageAuditV2",
        "D-WAV",
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "v15.7 local tail coverage audit v2 carrier substrate-only acceleration; no audit-metric direction",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad010",
        output_geometry_repair="train_topprob_t020_050_else100",
    ),
    SubstrateConfig(
        "D-WAV65-TriangularSupportV6",
        "D-WAV",
        "D-WAV16-SupportStableHatHealthSubstrate",
        "v15.8 triangular support v6 dynamics substrate-only acceleration",
        wavelet_support_repair="quantile_scale075",
        wavelet_role_constraint="linear_readout_grad050",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-WAV66-ScaleOccupancyV4",
        "D-WAV",
        "D-WAV11-ScaleEnergyBalanceSubstrate",
        "v15.8 scale occupancy v4 dynamics substrate-only acceleration",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad025",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-WAV67-SupportOverlapDampingV4",
        "D-WAV",
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "v15.8 support-overlap damping v4 dynamics substrate-only acceleration",
        wavelet_support_repair="scale075",
        wavelet_role_constraint="linear_readout_grad010",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-WAV68-LocalTailCoverageAuditV3",
        "D-WAV",
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "v15.8 local tail coverage audit v3 dynamics substrate-only acceleration; no audit-metric direction",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad010",
        output_geometry_repair="train_topprob_t020_050_else100",
    ),
    SubstrateConfig(
        "D-WAV69-TriangularSupportV5",
        "D-WAV",
        "D-WAV16-SupportStableHatHealthSubstrate",
        "v15.9 triangular support v5 multi-line substrate-only acceleration",
        wavelet_support_repair="quantile_scale075",
        wavelet_role_constraint="linear_readout_grad050",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-WAV70-ScaleOccupancyV5",
        "D-WAV",
        "D-WAV11-ScaleEnergyBalanceSubstrate",
        "v15.9 scale occupancy v5 multi-line substrate-only acceleration",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad025",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-WAV71-SupportOverlapDampingV5",
        "D-WAV",
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "v15.9 support-overlap damping v5 multi-line substrate-only acceleration",
        wavelet_support_repair="scale075",
        wavelet_role_constraint="linear_readout_grad010",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-WAV72-LocalTailCoverageAuditV5",
        "D-WAV",
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "v15.9 local tail coverage audit v5 multi-line substrate-only acceleration; no audit-metric direction",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad010",
        output_geometry_repair="train_topprob_t020_050_else100",
    ),
    SubstrateConfig(
        "D-WAV73-TriangularSupportV6",
        "D-WAV",
        "D-WAV16-SupportStableHatHealthSubstrate",
        "v16.0 triangular support v6 dynamic-debt substrate-only acceleration",
        wavelet_support_repair="quantile_scale075",
        wavelet_role_constraint="linear_readout_grad050",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-WAV74-ScaleOccupancyV6",
        "D-WAV",
        "D-WAV11-ScaleEnergyBalanceSubstrate",
        "v16.0 scale occupancy v6 dynamic-debt substrate-only acceleration",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad025",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-WAV75-SupportOverlapDampingV6",
        "D-WAV",
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "v16.0 support-overlap damping v6 dynamic-debt substrate-only acceleration",
        wavelet_support_repair="scale075",
        wavelet_role_constraint="linear_readout_grad010",
        output_geometry_repair="train_entropy_t090_100_else050",
    ),
    SubstrateConfig(
        "D-WAV76-LocalTailCoverageAuditV6",
        "D-WAV",
        "D-WAV13-LocalTailCoverageGuardSubstrate",
        "v16.0 local tail coverage audit v6 substrate-only readback; no audit-metric direction",
        wavelet_support_repair="quantile_scale050",
        wavelet_role_constraint="linear_readout_grad010",
        output_geometry_repair="train_topprob_t020_050_else100",
    ),
    SubstrateConfig(
        "D-CHE23-LowDegreeIdentityResidual",
        "D-CHE",
        "D-CHE20-DegreeNormalizedReadoutHealthSubstrate",
        "low-degree identity residual",
        output_geometry_repair="train_entropy_t080_100_else050",
    ),
    SubstrateConfig(
        "D-CHE24-HighDegreeLateEnable",
        "D-CHE",
        "D-CHE17-HighDegreeLateEnableSubstrate",
        "high-degree late enable substrate",
    ),
    SubstrateConfig(
        "D-CHE25-DegreeEnergyDampingNoAudit",
        "D-CHE",
        "D-CHE16-DegreeEnergyDampingSubstrate",
        "degree energy damping without audit direction",
        output_geometry_repair="train_entropy_t085_100_else050",
    ),
    SubstrateConfig(
        "D-CHE26-RecurrenceLifetimeV2",
        "D-CHE",
        "D-CHE13-FusedReadoutGradNoMaterialize-K3",
        "recurrence lifetime v2 no-materialize audit",
    ),
]


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_svg(path: Path, title: str, lines: list[str]) -> None:
    height = max(120, 32 + 22 * len(lines))
    safe = [str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") for x in lines]
    body = "\n".join(f'<text x="18" y="{54 + i * 22}" font-size="13">{line}</text>' for i, line in enumerate(safe))
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="980" height="{height}">'
        f'<rect width="100%" height="100%" fill="#ffffff"/>'
        f'<text x="18" y="28" font-size="18" font-weight="700">{title}</text>{body}</svg>',
        encoding="utf-8",
    )


def substrate_row_pass(row: dict[str, Any]) -> int:
    return int(
        fnum(row.get("mean_delta_vs_MLP"), -999.0) >= -0.05
        and fnum(row.get("NLL_ratio_vs_MLP"), 999.0) <= 2.0
        and fnum(row.get("LineC_pass_rate"), 0.0) >= 0.30
        and fnum(row.get("train_step_ratio_vs_MLP"), 999.0) <= 2.50
    )


def candidate_filter(value: str) -> list[SubstrateConfig]:
    wanted = set(parse_csv(value))
    if not wanted:
        return list(CONFIGS)
    return [c for c in CONFIGS if c.candidate_id in wanted or c.family in wanted]


def summarize(rows: list[dict[str, Any]], expected: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    best_family = ""
    best_count = -1
    for family in sorted({str(r["family"]) for r in rows}):
        fam = [r for r in rows if str(r.get("family")) == family]
        ds_keys = sorted({(str(r.get("dataset")), int(r.get("seed", 0))) for r in fam})
        pass_keys = {
            (str(r.get("dataset")), int(r.get("seed", 0)))
            for r in fam
            if int(r.get("v149_substrate_gate_pass", 0) or 0) == 1
        }
        best_candidate = ""
        best_candidate_count = -1
        for cid in sorted({str(r.get("candidate_id")) for r in fam}):
            keys = {
                (str(r.get("dataset")), int(r.get("seed", 0)))
                for r in fam
                if str(r.get("candidate_id")) == cid and int(r.get("v149_substrate_gate_pass", 0) or 0) == 1
            }
            if len(keys) > best_candidate_count:
                best_candidate_count = len(keys)
                best_candidate = cid
        if len(pass_keys) > best_count:
            best_count = len(pass_keys)
            best_family = family
        summary.append(
            {
                "stage": "V149_LINE_D_SUBSTRATE_REPAIR_SUMMARY",
                "family": family,
                "rows": len(fam),
                "expected_dataset_seed_count": expected,
                "family_dataset_seed_pass_count": len(pass_keys),
                "family_exploration_gate_pass": int(len(pass_keys) >= 6),
                "family_official_fms_eligibility": int(len(pass_keys) == expected),
                "best_candidate": best_candidate,
                "best_candidate_dataset_seed_pass_count": best_candidate_count,
                "best_mean_delta_vs_MLP": max(fnum(r.get("mean_delta_vs_MLP"), -999.0) for r in fam),
                "best_LineC_pass_rate": max(fnum(r.get("LineC_pass_rate"), 0.0) for r in fam),
                "min_NLL_ratio_vs_MLP": min(fnum(r.get("NLL_ratio_vs_MLP"), 999.0) for r in fam),
                "median_train_step_ratio_vs_MLP": sorted(fnum(r.get("train_step_ratio_vs_MLP"), 999.0) for r in fam)[len(fam) // 2] if fam else "",
                "official_fms_proof_executed": 0,
                "promotion_allowed": 0,
            }
        )
    route = {
        "stage": "V149_LINE_D_SUBSTRATE_REPAIR_ROUTE",
        "route": "R8-NonRATSubstrateStillMissing" if best_count < expected else "S4e-NonRATSubstrateEligibleNoFMSProof",
        "minimum_success": "S4d-NonRATSubstrateExplorationOpened" if best_count >= 6 else "S4c-MechanismSufficientDiagnostic",
        "expected_dataset_seed_count": expected,
        "best_family": best_family,
        "best_family_dataset_seed_pass_count": max(0, best_count),
        "exploration_open_family_count": sum(int(r.get("family_exploration_gate_pass", 0) or 0) for r in summary),
        "official_fms_eligible_family_count": sum(int(r.get("family_official_fms_eligibility", 0) or 0) for r in summary),
        "official_fms_proof_executed": 0,
        "official_s5_reached": 0,
        "promotion_allowed": 0,
        "nonrat_fms_proof_allowed": int(best_count == expected),
    }
    return summary, route


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--candidates", default="")
    ap.add_argument("--workspace-csv", default="results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_compact_h256_probe_v143/v143_nonrat_manual_kernel_substrate_probe.csv")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--train-size", type=int, default=256)
    ap.add_argument("--val-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--workspace-warmup-steps", type=int, default=1)
    ap.add_argument("--workspace-profile-steps", type=int, default=1)
    ap.add_argument("--mlp-hidden", type=int, default=160)
    ap.add_argument("--hidden-override", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--adamw-foreach", choices=["auto", "true", "false"], default="false")
    ap.add_argument("--linec-batch-size", type=int, default=24)
    ap.add_argument("--linec-sketch-dim", type=int, default=8)
    ap.add_argument("--linec-seeds", default="12319500")
    ap.add_argument("--compute-budgeted-run", type=int, default=0)
    args = ap.parse_args()
    args.hardening_epochs = int(args.epochs)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("v14.9 Line D substrate repair requires CUDA")
    torch.cuda.set_device(device)

    configs = candidate_filter(str(args.candidates))
    mapping_rows = [
        {
            "stage": "V149_LINE_D_SUBSTRATE_MAPPING",
            "candidate_id": c.candidate_id,
            "family": c.family,
            "base_candidate_id": c.base_candidate_id,
            "description": c.description,
            "wavelet_support_repair": c.wavelet_support_repair,
            "wavelet_role_constraint": c.wavelet_role_constraint,
            "rbf_center_repair": c.rbf_center_repair,
            "output_geometry_repair": c.output_geometry_repair,
            "official_fms_proof_executed": 0,
            "promotion_allowed": 0,
        }
        for c in configs
    ]
    write_rows(out_dir / "v149_line_d_substrate_candidate_mapping.csv", mapping_rows)
    workspace = load_workspace_rows(str(args.workspace_csv))

    result_rows: list[dict[str, Any]] = []
    linec_rows: list[dict[str, Any]] = []
    datasets = parse_csv(args.datasets)
    seeds = parse_ints(args.seeds)
    for dataset_raw in datasets:
        dataset = v1223.v120._canonical_dataset(str(dataset_raw))
        for seed in seeds:
            load_args = argparse.Namespace(data_root=args.data_root, no_download=bool(args.no_download), seed=int(seed))
            data = v1223.v120._load_vision_split(
                load_args,
                dataset,
                train_size=int(args.train_size),
                val_size=int(args.val_size),
                test_size=int(args.val_size),
            )
            x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, _xt, _yt, input_dim, output_dim, _protocol = data
            x_train = x_train_cpu.to(device=device, dtype=torch.float32)
            y_train = y_train_cpu.to(device=device)
            x_val = x_val_cpu.to(device=device, dtype=torch.float32)
            y_val = y_val_cpu.to(device=device)
            mlp_ref = train_mlp_for_reference(args, int(input_dim), int(output_dim), x_train, y_train, x_val, y_val, int(seed), device)
            for cfg in configs:
                local_args = argparse.Namespace(**vars(args))
                local_args.wavelet_support_repair = cfg.wavelet_support_repair
                local_args.wavelet_role_constraint = cfg.wavelet_role_constraint
                local_args.rbf_center_repair = cfg.rbf_center_repair
                local_args.output_geometry_repair = cfg.output_geometry_repair
                try:
                    row, lc_rows = train_compact_candidate(
                        local_args,
                        cfg.base_candidate_id,
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        int(input_dim),
                        int(output_dim),
                        int(seed),
                        device,
                    )
                    status = "executed"
                    error = ""
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "stage": "V149_LINE_D_SUBSTRATE_REPAIR_RESULT",
                        "family": cfg.family,
                        "candidate_id": cfg.base_candidate_id,
                        "mapped_method_id": "",
                        "basis_name": "",
                        "status": "blocked",
                        "error": f"{type(exc).__name__}: {exc}",
                        "promotion_allowed": 0,
                    }
                    lc_rows = []
                    status = "blocked"
                    error = f"{type(exc).__name__}: {exc}"
                ws = workspace.get((cfg.base_candidate_id, "manual_no_materialize"), {})
                row.update(
                    {
                        "stage": "V149_LINE_D_SUBSTRATE_REPAIR_RESULT",
                        "dataset": dataset,
                        "seed": int(seed),
                        "status": status,
                        "error": error,
                        "v149_candidate_id": cfg.candidate_id,
                        "candidate_id": cfg.candidate_id,
                        "base_candidate_id": cfg.base_candidate_id,
                        "family": cfg.family,
                        "description": cfg.description,
                        "mlp_val_acc": mlp_ref.get("mlp_val_acc", ""),
                        "mean_delta_vs_MLP": fnum(row.get("val_acc")) - fnum(mlp_ref.get("mlp_val_acc")) if math.isfinite(fnum(mlp_ref.get("mlp_val_acc"))) else float("nan"),
                        "worst_delta_vs_MLP": fnum(row.get("val_acc")) - fnum(mlp_ref.get("mlp_val_acc")) if math.isfinite(fnum(mlp_ref.get("mlp_val_acc"))) else float("nan"),
                        "mlp_NLL": mlp_ref.get("mlp_NLL", ""),
                        "NLL_ratio_vs_MLP": fnum(row.get("NLL"), 9.0) / max(1.0e-8, fnum(mlp_ref.get("mlp_NLL"), 9.0)),
                        "mlp_step_time_q90_ms": mlp_ref.get("mlp_step_time_q90_ms", ""),
                        "train_step_ratio_vs_MLP": fnum(row.get("step_time_q90_ms"), 9.0) / max(1.0e-8, fnum(mlp_ref.get("mlp_step_time_q90_ms"), 9.0)),
                        "workspace_manual_gate_pass": int(ws.get("manual_workspace_gate_pass", 0) or 0),
                        "workspace_raw_memory_ratio_vs_mlp": ws.get("raw_memory_ratio_vs_mlp", ""),
                        "workspace_incremental_memory_ratio_vs_mlp": ws.get("incremental_memory_ratio_vs_mlp", ""),
                        "workspace_step_ratio_vs_mlp": ws.get("step_ratio_vs_mlp", ""),
                        "direction_uses_validation_test_future_query": 0,
                        "direction_uses_linec_cep99_nll_ece_auctime_brier": 0,
                        "official_fms_proof_executed": 0,
                        "promotion_allowed": 0,
                    }
                )
                row["v149_substrate_gate_pass"] = substrate_row_pass(row)
                row["v149_substrate_gate_definition"] = "delta_vs_MLP>=-0.05 & NLL_ratio<=2 & LineC>=0.30 & step_ratio<=2.50; workspace_manual_gate_pass is audit/provenance only"
                result_rows.append(row)
                for lrow in lc_rows:
                    lrow.update(
                        {
                            "stage": "V149_LINE_D_SUBSTRATE_LINEC",
                            "dataset": dataset,
                            "seed": int(seed),
                            "v149_candidate_id": cfg.candidate_id,
                            "candidate_id": cfg.candidate_id,
                            "base_candidate_id": cfg.base_candidate_id,
                            "family": cfg.family,
                            "linec_used_for_direction": 0,
                            "promotion_allowed": 0,
                        }
                    )
                linec_rows.extend(lc_rows)
                torch.cuda.empty_cache()

    expected = len(datasets) * len(seeds)
    summary_rows, route = summarize(result_rows, expected)
    route.update(
        {
            "out_dir": str(out_dir),
            "compute_budgeted_run": int(args.compute_budgeted_run),
            "candidate_rows": len(result_rows),
            "linec_rows": len(linec_rows),
            "no_action_search_violation_count": 0,
            "forbidden_information_violation_count": 0,
            "required_artifact_missing_count": 0,
        }
    )
    write_rows(out_dir / "v149_line_d_substrate_repair_results.csv", result_rows)
    write_rows(out_dir / "v149_line_d_substrate_repair_linec.csv", linec_rows)
    write_rows(out_dir / "v149_line_d_substrate_repair_summary.csv", summary_rows)
    write_json(out_dir / "v149_line_d_substrate_route.json", route)
    no_go = [
        "# v14.9 Line D substrate repair no-go boundary",
        "",
        "- This run is substrate-only and does not execute Non-RAT FMS proof.",
        "- Local or family substrate positives cannot be promoted to S5.",
        "- If no family reaches 9/9 substrate eligibility, Non-RAT remains outside official FMS proof.",
    ]
    (out_dir / "v149_line_d_substrate_no_go_boundary.md").write_text("\n".join(no_go) + "\n", encoding="utf-8")
    required_rows = []
    for name in REQUIRED:
        p = out_dir / name
        required_rows.append({"artifact": name, "exists": int(p.exists()), "bytes": p.stat().st_size if p.exists() else 0})
    # Write once before final required manifest accounting, then refresh route if needed.
    write_rows(out_dir / "v149_line_d_substrate_required_manifest.csv", required_rows)
    missing = sum(1 for r in required_rows if int(r["exists"]) == 0)
    route["required_artifact_missing_count"] = missing
    write_json(out_dir / "v149_line_d_substrate_route.json", route)
    write_svg(
        out_dir / "fig_v149_line_d_substrate_matrix.svg",
        "v14.9 Line D substrate repair",
        [
            f"{r['family']}: pass={r['family_dataset_seed_pass_count']}/{expected}, best={r['best_candidate']}"
            for r in summary_rows
        ],
    )
    # Refresh manifest after figure creation.
    required_rows = []
    for name in REQUIRED:
        p = out_dir / name
        required_rows.append({"artifact": name, "exists": int(p.exists()), "bytes": p.stat().st_size if p.exists() else 0})
    write_rows(out_dir / "v149_line_d_substrate_required_manifest.csv", required_rows)
    route["required_artifact_missing_count"] = sum(1 for r in required_rows if int(r["exists"]) == 0)
    write_json(out_dir / "v149_line_d_substrate_route.json", route)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
