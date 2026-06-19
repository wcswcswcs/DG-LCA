"""Optimizer-pluggable FU mechanisms for v17 smoke experiments."""

from __future__ import annotations

from typing import Any

import torch

from dgkan.fu.core import UpdateTensor, cosine, flat_grad, flat_params, load_flat_params, normalized_like
from dgkan.fu.function_space_metrics import METRIC_NAMES, metric_project_vector, output_metric_energies
from dgkan.fu.metric_solver import V2206_SOLVER_CONFIGS, solve_metric_readout_update


MECHANISMS = [
    "CTRL-AdamW",
    "CTRL-SGD",
    "CTRL-NoOpMatchedOverhead",
    "CTRL-RandomMatchedNorm",
    "CTRL-RandomSameRankBlock",
    "CTRL-RecoveryOnly",
    "M1-AdamWPrimaryFUResidual",
    "M2-SGDMomentumPrimaryFU",
    "M3-FUPrimary",
    "M4-FUOnlyKeyParams",
    "M5-AlternatingFUGradient",
    "M6-SlowStateFU",
    "M7-MatrixBlockFU",
    "M8-PopRiskSNRFU",
    "M9-FunctionSpaceOperatorFU",
    "M10-RolePartitionOptimizer",
    "M11-DualMemorySlowStateFU",
    "M12-ScheduleFreeAveragedFU",
    "M13-LowRankMatrixBlockFU",
    "M14-SourceChannelPopRiskSlowFU",
    "M15-LineCFilteredAlternatingFU",
    "M16-TwoPhaseMomentumThenLineCFU",
    "M17-ReadoutCarrierTwoPhaseLineCFU",
    "M18-CarrierLowRankBlockFU",
    "M19-SplitConsensusLowRankBlockFU",
    "M20-TrainSplitFunctionSpaceActuationFU",
    "M21-ExactReadoutFunctionSpaceActuationFU",
    "M22-ExactReadoutHighCapScheduledFU",
    "M23-ExactReadoutUltraCapScheduledFU",
    "M24-WarmupExactReadoutUltraCapFU",
    "M25-AdamWExactReadoutUltraCapFU",
    "M26-GatedAdamWExactReadoutFU",
    "M27-MultiBatchGeneralizationGatedReadoutFU",
    "M28-AdamWMultiBatchGatedReadoutFU",
    "M29-ConsensusSourceStateFU",
    "M30-LowThresholdConsensusSourceStateFU",
    "M31-AdamWLowThresholdConsensusResidualFU",
    "M32-PostAdamWConsensusResidualFU",
    "M33-PopRiskMatrixBlockSlowFU",
    "M34-ExactPopRiskSlowFU",
    "M35-ExactPopRiskMatrixBlockSlowFU",
    "M36-MomentumWarmReadoutBlockFU",
    "M37-SplitFisherAgreementSlowFU",
    "M38-AdamWSplitFisherAgreementResidualFU",
    "M39-MomentumWarmSplitFisherFU",
    "M40-MomentumWarmAntiWashoutFU",
    "M41-MomentumWarmSourceAnchorFU",
    "M42-MomentumWarmHoldFU",
    "M43-MomentumCycleHoldFU",
    "M44-MomentumLineCAnchorSlowFU",
    "M45-MomentumSlowAnchorFU",
    "M46-MomentumMatrixBlockRetentionFU",
    "M47-MomentumCycleThenLineCAnchorFU",
    "M48-DualTimescaleSourceRetentionFU",
    "M49-LossCotangentTargetFU",
    "M50-RandomMatchedTargetFU",
    "M51-SignFlippedTargetFU",
    "M52-CorruptedLabelTargetFU",
    "M53-LowRankLossCotangentTargetFU",
    "M54-CrossSplitConsensusTargetFU",
    "M55-LowDegreeReadoutTargetFU",
    "M56-WeakStableTargetFU",
    "M57-B1ReadoutTransferTargetFU",
    "M58-ReservoirExcludingConsensusTargetFU",
    "M59-StableRandomTargetControlFU",
    "M60-B1CrossSplitConsensusTransferFU",
    "M61-StableB1ConsensusTransferFU",
    "M62-B1WeakStableTransferFU",
    "M63-LowDegreeB1ConsensusTransferFU",
    "M64-ContrastiveLossRandomOrthogonalTargetFU",
    "M65-B1ContrastiveLossRandomOrthogonalTransferFU",
    "M66-TopWrongMarginTargetFU",
    "M67-B1TopWrongMarginTransferFU",
    "M68-MarginSplitConsensusSlowFU",
    "M69-RotatedCautiousMatrixSlowFU",
    "M70-TrainLookaheadCautiousSourceFU",
    "M71-TrainLookaheadB1ConsensusTransferFU",
    "M72-LossWarmToB1ConsensusMigrationFU",
    "M73-EasyB1ConsensusTransferFU",
    "M74-LossWarmToEasyB1ConsensusMigrationFU",
    "M75-LossEasyB1ConsensusBlendFU",
    "M76-LossWarmToLossEasyB1ConsensusBlendFU",
    "M77-GainGatedLossWarmB1ConsensusMigrationFU",
    "M78-GainGatedLossWarmBlendMigrationFU",
    "M79-B1ConsensusB3NullTransferFU",
    "M80-LossWarmToB1ConsensusB3NullMigrationFU",
    "M81-ViewConsistentLossTargetFU",
    "M82-LossWarmToViewConsistentLossMigrationFU",
    "M83-LowBankLossB3NullFU",
    "M84-LossWarmToLowBankLossB3NullMigrationFU",
    "M85-GainGatedLowBankLossB3NullFU",
    "M86-LossWarmToGainGatedLowBankLossB3NullMigrationFU",
    "M87-AdamWBoundaryThenMomentumSourceFU",
    "M88-AdamWBoundaryThenDualTimescaleSourceFU",
    "M89-AdamWBoundaryDualTimescaleSGDFloorFU",
    "M90-AdamWBoundaryDualTimescaleLateSGDFloorFU",
    "M91-AdamWBoundaryDualTimescaleTinyLateSGDFloorFU",
    "M92-AdamWBoundaryDualTimescaleReinforcedTinyLateSGDFloorFU",
    "M93-TrainLossGatedDualTimescaleTinyLateFU",
    "M94-TrainLossGatedDualTimescaleHoldFallbackFU",
    "M95-TrainLossGatedDualTimescaleTinyLateFallbackFU",
    "M96-TrainLossGatedDualTimescaleBoostedTinyLateFallbackFU",
    "M97-TrainLossLateHoldRecoveryFU",
    "M98-TrainLossLateLookaheadFloorFU",
    "M99-TrainLossTerminalLookaheadFloorFU",
    "M100-TrainLossEarlyTerminalLookaheadFloorFU",
    "M101-AdamWBoundaryDualTimescaleAntiWashoutFU",
    "M102-AdamWBoundaryDualTimescaleSourceAnchorFU",
    "M103-AdamWBoundaryDualTimescaleParamEMAReentryFU",
    "M104-AdamWBoundaryDualTimescaleReadoutChannelFU",
    "M105-AdamWBoundaryDualTimescaleHiddenMatrixChannelFU",
    "M106-TrainLossTerminalProjectedLookaheadFloorFU",
    "M107-TrainLossTerminalConsensusLookaheadFloorFU",
    "M108-TrainLossTerminalSelectorLookaheadFloorFU",
    "M109-AdamWBoundaryToGainGatedLowBankB3NullFU",
    "M110-AdamWBoundaryLowBankAnchorAntiWashoutFU",
    "M111-TrainLossTerminalPositiveLookaheadFloorFU",
    "M112-TrainLossTerminalCheckpointReentryFU",
    "M113-TrainLossTerminalHardSplitSourceFU",
    "M114-TrainLossTerminalAdamWLookaheadFU",
    "M115-TrainLossTerminalOptimizerSelectorFU",
    "M116-DatasetInvariantPopRiskSlowFU",
    "M117-DatasetInvariantReadoutConsensusFU",
    "M118-SourceConservingOptimizerOnlyFU",
    "M119-TerminalSourceConservingRouteFU",
    "M120-TrainLossRiskProfileRouteFU",
    "M121-DebtAwareSourceGateFU",
    "M122-UngatedWarmTerminalSourceRouteFU",
    "M123-UngatedWarmRiskProfileRouteFU",
    "M124-UngatedWarmDebtRawBailoutFU",
    "M125-TrainLossH2400CheckpointHoldFU",
    "M126-TrainLossH2800CheckpointHoldFU",
    "M127-TrainLossH2400DebtBailoutFU",
    "M128-TrainLossTerminalRawThenSourceGuardFU",
    "M129-H800SourceSlowEMARetentionFU",
    "M130-H800ReadoutChannelRetentionFU",
    "M131-H800DualMemorySourceRetentionFU",
    "M132-H1600SourceCheckpointReentryFU",
    "M133-H2400SourceCheckpointReentryFU",
    "M134-SignalReservoirB3NullConsensusTargetFU",
    "M135-SourceBankB3NullConsensusTargetFU",
    "M136-StableRandomB3NullTargetControlFU",
    "M137-EarlyPulseSourceBankB3NullConsensusTargetFU",
    "M138-EarlyPulseAdamWSourceBankB3NullConsensusTargetFU",
    "M139-EarlySourceSlowEMATerminalGuardFU",
    "M140-EarlySourceSlowEMATerminalRawGuardFU",
    "M141-EarlySourceSlowEMATerminalRawGuardStrongFU",
    "M142-EarlySourceSlowEMATerminalProjectedOptimizerFU",
    "M143-EarlySourceSlowEMATerminalProjectedBlendFU",
    "M144-EarlySourceSlowEMATerminalAntiWashoutFU",
    "M145-EarlySourceSlowEMATerminalH4000ReentryFU",
    "M146-EarlySourceSlowEMASignalReservoirTargetFU",
    "M147-EarlySourceSlowEMASourceBankTargetFU",
    "M148-EarlySourceSlowEMADualTargetGuardFU",
    "M149-EarlySourceSlowEMATerminalAdaptiveRawFU",
    "M150-EarlySourceSlowEMATerminalSparseSourceFU",
    "M151-EarlySourceSlowEMATerminalRatioPreserveFU",
    "M152-SourceProjectionB3NullConsensusTargetFU",
    "M153-NoiseOrthogonalB3NullConsensusTargetFU",
    "M154-EasyMarginB3NullConsensusTargetFU",
    "M155-SourceProjectionB3NullConsensusTargetOnlyFU",
    "M156-NoiseOrthogonalB3NullConsensusTargetOnlyFU",
    "M157-EasyMarginB3NullConsensusTargetOnlyFU",
    "M158-EarlySourceSlowEMATerminalRejectSourceAxisRescueFU",
    "M159-EarlySourceSlowEMATerminalRejectSlowEMARescueFU",
    "M160-EarlySourceSlowEMATerminalRejectHoldSourceRescueFU",
    "M161-EarlySourceSlowEMAShapePreserveFU",
    "M162-EarlySourceSlowEMAShapePreserveRawGuardFU",
    "M163-EarlySourceSlowEMAShapePreserveClampFU",
    "M164-EarlySourceSlowEMATerminalRejectRawRescueFU",
    "M165-EarlySourceSlowEMATerminalRejectHybridRescueFU",
    "M166-EarlySourceSlowEMATerminalRejectLateOnlyRawRescueFU",
    "M167-EarlySourceSlowEMATerminalTopKSupportGuardFU",
    "M168-EarlySourceSlowEMATerminalTopKRawGuardFU",
    "M169-EarlySourceSlowEMATerminalTopKDebtCapFU",
    "M170-EarlySourceSlowEMATerminalSourcePreserveStrongFU",
    "M171-EarlySourceSlowEMATerminalSourcePreserveGentleFU",
    "M172-EarlySourceSlowEMATerminalInfoVolumeGuardFU",
    "M173-EarlySourceSlowEMALowNDSDiffeomorphicTargetFU",
    "M174-EarlySourceSlowEMAInfoVolumeDiffeomorphicTargetFU",
    "M175-EarlySourceSlowEMALowRankReadoutTransportFU",
    "M179-EarlySourceSlowEMATerminalSourcePreserveVeryStrongFU",
    "M180-EarlySourceSlowEMAH3600TerminalSourcePreserveFU",
    "M181-EarlySourceSlowEMATerminalNoraOrthogonalSourceFU",
    "M182-EarlySourceSlowEMATerminalDebtAwarePreserveFU",
    "M183-EarlySourceSlowEMATerminalLowNDSMatrixBlockFU",
    "M184-EarlySourceSlowEMATerminalDualMemoryPreserveFU",
    "M185-EarlySourceSlowEMASNRTerminalPredictorFU",
    "M186-EarlySourceSlowEMASplitConsensusEstimatorFU",
    "M187-EarlySourceSlowEMASignalReservoirTransportFU",
    "M188-EarlySourceSlowEMATerminalSourceFloorFU",
    "M189-EarlySourceSlowEMATerminalH4000AnchorFloorFU",
    "M190-EarlySourceSlowEMATerminalDecayAwareFloorFU",
    "M191-EarlySourceSlowEMATerminalRawGuardSourceFloorFU",
    "M192-EarlySourceSlowEMATerminalRawGuardH4000AnchorFloorFU",
    "M193-EarlySourceSlowEMATerminalRawGuardDecayAwareFloorFU",
    "M194-EarlySourceSlowEMATerminalAntiErosionOrthogonalFU",
    "M195-EarlySourceSlowEMATerminalSourceReflectionGuardFU",
    "M196-EarlySourceSlowEMATerminalH4000TransportCorrectorFU",
    "M197-EarlySourceSlowEMATerminalH3200AnchorTransportFU",
    "M198-EarlySourceSlowEMATerminalRawGuardH3200AnchorTransportFU",
    "M199-EarlySourceSlowEMATerminalH3200RatioReentryFU",
    "M200-EarlySourceSlowEMATerminalH3200ProgressCarryFU",
    "M201-EarlySourceSlowEMATerminalH4000ProgressCarryFU",
    "M202-EarlySourceSlowEMATerminalH3200SourceProgressBlendFU",
    "M203-EarlySourceSlowEMATerminalRawGuardH3600GentleProgressFU",
    "M204-EarlySourceSlowEMATerminalRawGuardH4000GentleProgressFU",
    "M205-EarlySourceSlowEMATerminalRawGuardH4400ProjectedProgressFU",
    "M206-EarlySourceSlowEMATerminalControlRelativeSGDCatchupFU",
    "M207-EarlySourceSlowEMATerminalControlRelativeAdamWCatchupFU",
    "M208-EarlySourceSlowEMATerminalControlRelativeSourceBalancedCatchupFU",
    "M209-EarlySourceSlowEMATerminalTrajectoryAdaptivePreserveFU",
    "M210-EarlySourceSlowEMATerminalMidErosionBridgeFU",
    "M211-EarlySourceSlowEMATerminalTwoPhaseRatioRepairFU",
    "M212-EarlySourceSlowEMATerminalAntiSourceClipFU",
    "M213-EarlySourceSlowEMATerminalDebtAwareHoldFU",
    "M214-EarlySourceSlowEMATerminalAnchorFlowTinyFU",
    "M215-EarlySourceSlowEMATerminalAcceptMemoryFU",
    "M216-EarlySourceSlowEMATerminalAcceptMemoryDebtFU",
    "M217-EarlySourceSlowEMATerminalSourceProgressMemoryFU",
    "M218-EarlySourceSlowEMATerminalProjectedOptimizerLambda050FU",
    "M219-EarlySourceSlowEMATerminalProjectedOptimizerLambda100FU",
    "M176-LowNDSDiffeomorphicTargetOnlyFU",
    "M177-InfoVolumeDiffeomorphicTargetOnlyFU",
    "M178-LowRankReadoutTransportTargetOnlyFU",
    "M220-MetricFirstG0L2FU",
    "M221-MetricFirstDiagFisherFU",
    "M222-MetricFirstPopRiskDiagFU",
    "M223-MetricFirstSobolevH1FU",
    "M224-MetricFirstRKHSKNNFU",
    "M225-MetricFirstFisherRKHSFU",
    "M226-MetricFirstLowNDSFU",
    "M227-MetricFirstEnsembleFU",
    "M228-MetricSourceMemoryG0L2FU",
    "M229-MetricSourceMemoryDiagFisherFU",
    "M230-MetricSourceMemoryPopRiskDiagFU",
    "M231-MetricSourceMemorySobolevH1FU",
    "M232-MetricSourceMemoryRKHSKNNFU",
    "M233-MetricSourceMemoryFisherRKHSFU",
    "M234-MetricSourceMemoryLowNDSFU",
    "M235-MetricSourceMemoryEnsembleFU",
    "M236-MetricTargetLossCotangentFU",
    "M237-MetricTargetLowDegreeReadoutFU",
    "M238-MetricTargetB1TransferFU",
    "M239-MetricTargetSourceProjectedB3NullFU",
    "M240-HC2H3200NoProjectionFU",
    "M241-HC2H3200L2ProjectionFU",
    "M242-HC2H3200HalfProjectionFU",
    "M243-HC2H4000L2RecomputeProjectionFU",
    "M244-HC2H4000HalfRecomputeProjectionFU",
    "M245-MetricTargetSplitConsensusFU",
    "M246-MetricTargetPopRiskSNRFU",
    "M247-MetricTargetDiffeomorphicNoFoldFU",
    "M248-MetricTargetRandomMatchedFU",
    "M249-MetricTargetSignFlippedFU",
    "M250-MetricTargetDebtCalibratedSourceFU",
    "M251-MetricTargetObservableMidDebtSourceFU",
    "M252-MetricTargetObservableMidDebtM181BridgeFU",
    "M253-MetricTargetGradientObservableMidDebtFU",
    "M254-HC8PreH3200SourceChannelAntiWashoutFU",
    "M255-HC9RiskWeightedSplitConsensusSourceFU",
    "M256-HC10ViewConsistentSourceCarryFU",
    "M257-HC11ContinuousSourceCarryAntiWashoutFU",
    "M258-HC12NoiseOrthogonalSourceCarryFU",
    "M259-V2206MetricSolverT0G0ReadoutFU",
    "M260-V2206MetricSolverT1G0SplitTransferFU",
    "M261-V2206MetricSolverT3G3SignalSobolevFU",
    "M262-V2206MetricSolverT4G3SmoothManifoldFU",
    "M263-V2206MetricSolverT5G6LowNDSFU",
    "M264-V2206MetricSolverT6G0DualMemoryFU",
    "M265-V2206MetricSolverT7G0HiddenBlockFU",
    "M266-V2206MetricSolverT8G0AdaptiveHiddenBlockFU",
    "M267-V2206MetricSolverT9G0C3GatedHiddenBlockFU",
    "M268-V2206MetricSolverT10G0CompensatedHiddenBlockFU",
    "M269-V2206MetricSolverT11G0EarlyObservableFU",
    "M270-V2206MetricSolverT12G0SoftCompensatedHiddenBlockFU",
    "M271-V2206MetricSolverT12G0HiddenOnlySoftCompensatedFU",
]


CONTROL_MECHANISMS = {
    "CTRL-AdamW",
    "CTRL-SGD",
    "CTRL-NoOpMatchedOverhead",
    "CTRL-RandomMatchedNorm",
    "CTRL-RandomSameRankBlock",
    "CTRL-RecoveryOnly",
}


def mechanism_family(name: str) -> str:
    if name.startswith("CTRL"):
        return "M0-Controls"
    return name.split("-", 1)[0]


METRIC_FIRST_MECHANISM_TO_METRIC = {
    "M220-MetricFirstG0L2FU": "G0-L2",
    "M221-MetricFirstDiagFisherFU": "G1-DiagFisher",
    "M222-MetricFirstPopRiskDiagFU": "G2-PopRiskDiag",
    "M223-MetricFirstSobolevH1FU": "G3-SobolevH1-hidden",
    "M224-MetricFirstRKHSKNNFU": "G4-RKHS-KNN",
    "M225-MetricFirstFisherRKHSFU": "G5-Fisher-RKHS",
    "M226-MetricFirstLowNDSFU": "G6-LowNDS",
    "M227-MetricFirstEnsembleFU": "G8-MetricEnsemble",
    "M228-MetricSourceMemoryG0L2FU": "G0-L2",
    "M229-MetricSourceMemoryDiagFisherFU": "G1-DiagFisher",
    "M230-MetricSourceMemoryPopRiskDiagFU": "G2-PopRiskDiag",
    "M231-MetricSourceMemorySobolevH1FU": "G3-SobolevH1-hidden",
    "M232-MetricSourceMemoryRKHSKNNFU": "G4-RKHS-KNN",
    "M233-MetricSourceMemoryFisherRKHSFU": "G5-Fisher-RKHS",
    "M234-MetricSourceMemoryLowNDSFU": "G6-LowNDS",
    "M235-MetricSourceMemoryEnsembleFU": "G8-MetricEnsemble",
    "M236-MetricTargetLossCotangentFU": "T0-LossCotangentReadout",
    "M237-MetricTargetLowDegreeReadoutFU": "T3-LowDegreeReadout",
    "M238-MetricTargetB1TransferFU": "T4-B1LossTransfer",
    "M239-MetricTargetSourceProjectedB3NullFU": "T5-SourceProjectedB3Null",
    "M245-MetricTargetSplitConsensusFU": "T1-SplitConsensus",
    "M246-MetricTargetPopRiskSNRFU": "T2-PopRiskSNR",
    "M247-MetricTargetDiffeomorphicNoFoldFU": "T6-DiffeomorphicNoFold",
    "M248-MetricTargetRandomMatchedFU": "T7-RandomMatchedActuation",
    "M249-MetricTargetSignFlippedFU": "T8-SignFlippedTarget",
    "M250-MetricTargetDebtCalibratedSourceFU": "T9-DebtCalibratedSource",
    "M251-MetricTargetObservableMidDebtSourceFU": "T10-ObservableMidDebtSource",
    "M252-MetricTargetObservableMidDebtM181BridgeFU": "T10-ObservableMidDebtSource+M181Bridge",
    "M253-MetricTargetGradientObservableMidDebtFU": "T11-GradientObservableMidDebtSource",
    "M254-HC8PreH3200SourceChannelAntiWashoutFU": "T12-PreH3200SourceChannelAntiWashout",
    "M255-HC9RiskWeightedSplitConsensusSourceFU": "T13-RiskWeightedSplitConsensusSource",
    "M256-HC10ViewConsistentSourceCarryFU": "T14-ViewConsistentSourceCarry",
    "M257-HC11ContinuousSourceCarryAntiWashoutFU": "T15-ContinuousSourceCarryAntiWashout",
    "M258-HC12NoiseOrthogonalSourceCarryFU": "T16-NoiseOrthogonalSourceCarry",
}

V2206_METRIC_SOLVER_MECHANISMS = set(V2206_SOLVER_CONFIGS)

METRIC_SOURCE_MEMORY_MECHANISMS = {
    "M228-MetricSourceMemoryG0L2FU",
    "M229-MetricSourceMemoryDiagFisherFU",
    "M230-MetricSourceMemoryPopRiskDiagFU",
    "M231-MetricSourceMemorySobolevH1FU",
    "M232-MetricSourceMemoryRKHSKNNFU",
    "M233-MetricSourceMemoryFisherRKHSFU",
    "M234-MetricSourceMemoryLowNDSFU",
    "M235-MetricSourceMemoryEnsembleFU",
}

TERMINAL_SOURCE_SLOW_STATE_MECHANISMS = {
    "M181-EarlySourceSlowEMATerminalNoraOrthogonalSourceFU",
    "M218-EarlySourceSlowEMATerminalProjectedOptimizerLambda050FU",
    "M219-EarlySourceSlowEMATerminalProjectedOptimizerLambda100FU",
}

METRIC_TARGET_MECHANISMS = {
    "M236-MetricTargetLossCotangentFU",
    "M237-MetricTargetLowDegreeReadoutFU",
    "M238-MetricTargetB1TransferFU",
    "M239-MetricTargetSourceProjectedB3NullFU",
    "M245-MetricTargetSplitConsensusFU",
    "M246-MetricTargetPopRiskSNRFU",
    "M247-MetricTargetDiffeomorphicNoFoldFU",
    "M248-MetricTargetRandomMatchedFU",
    "M249-MetricTargetSignFlippedFU",
    "M250-MetricTargetDebtCalibratedSourceFU",
    "M251-MetricTargetObservableMidDebtSourceFU",
    "M252-MetricTargetObservableMidDebtM181BridgeFU",
    "M253-MetricTargetGradientObservableMidDebtFU",
    "M254-HC8PreH3200SourceChannelAntiWashoutFU",
    "M255-HC9RiskWeightedSplitConsensusSourceFU",
    "M256-HC10ViewConsistentSourceCarryFU",
    "M257-HC11ContinuousSourceCarryAntiWashoutFU",
    "M258-HC12NoiseOrthogonalSourceCarryFU",
}

HC2_H3200_PROJECTION_MECHANISMS = {
    "M240-HC2H3200NoProjectionFU",
    "M241-HC2H3200L2ProjectionFU",
    "M242-HC2H3200HalfProjectionFU",
}

HC2_H4000_RECOMPUTE_PROJECTION_MECHANISMS = {
    "M243-HC2H4000L2RecomputeProjectionFU",
    "M244-HC2H4000HalfRecomputeProjectionFU",
}

HC2_PROJECTION_MECHANISMS = HC2_H3200_PROJECTION_MECHANISMS | HC2_H4000_RECOMPUTE_PROJECTION_MECHANISMS


def _matrix_block_mask(model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        mask = torch.zeros_like(p, device=device).reshape(-1)
        if p.ndim >= 2:
            rows = int(p.shape[0])
            keep = max(1, rows // 2)
            view = mask.view_as(p)
            view[:keep].fill_(1.0)
        elif any(token in pname.lower() for token in ["w2", "readout", "bias"]):
            mask.fill_(1.0)
        chunks.append(mask)
    return torch.cat(chunks) if chunks else torch.zeros(0, device=device)


def _alternating_mask(g: torch.Tensor) -> torch.Tensor:
    if g.numel() == 0:
        return g
    idx = torch.arange(g.numel(), device=g.device)
    return torch.where((idx % 2) == 0, torch.ones_like(g), -torch.ones_like(g))


def _mean_centered(g: torch.Tensor) -> torch.Tensor:
    if g.numel() == 0:
        return g
    centered = g - g.mean()
    return normalized_like(centered, g)


def _adamw_like_precondition(g: torch.Tensor) -> torch.Tensor:
    if g.numel() == 0:
        return g
    denom = torch.sqrt(g.detach().abs() + g.detach().abs().mean().clamp_min(1.0e-8))
    return normalized_like(g / denom.clamp_min(1.0e-6), g)


def _momentum_like_precondition(g: torch.Tensor) -> torch.Tensor:
    if g.numel() == 0:
        return g
    idx = torch.arange(g.numel(), device=g.device, dtype=g.dtype)
    ramp = 0.75 + 0.50 * (idx / max(1, g.numel() - 1))
    return normalized_like(g * ramp, g)


def _slow_state_initial(g: torch.Tensor) -> torch.Tensor:
    if g.numel() == 0:
        return g
    scale = g.detach().std().clamp_min(1.0e-6)
    return normalized_like(torch.tanh(g / (2.0 * scale)) * scale, g)


def _basis_role_mask(model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        mask = torch.zeros_like(p, device=device).reshape(-1)
        low = pname.lower()
        if any(token in low for token in ["w1", "cent", "scale", "den", "basis"]):
            mask.fill_(1.0)
        elif "w2" in low or "readout" in low:
            mask.fill_(0.5)
        chunks.append(mask)
    return torch.cat(chunks) if chunks else torch.zeros(0, device=device)


def _readout_role_mask(model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        mask = torch.zeros_like(p, device=device).reshape(-1)
        low = pname.lower()
        if "w2" in low or "readout" in low or "classifier" in low:
            mask.fill_(1.0)
        chunks.append(mask)
    return torch.cat(chunks) if chunks else torch.zeros(0, device=device)


def _hidden_matrix_role_mask(model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        mask = torch.zeros_like(p, device=device).reshape(-1)
        low = pname.lower()
        if p.ndim >= 2 and "w2" not in low and "readout" not in low and "classifier" not in low:
            mask.fill_(1.0)
        chunks.append(mask)
    return torch.cat(chunks) if chunks else torch.zeros(0, device=device)


def _carrier_low_rank_block_update(model: torch.nn.Module, device: torch.device, rank: int = 4) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        grad = p.grad.detach() if p.grad is not None else torch.zeros_like(p)
        low = pname.lower()
        if grad.ndim >= 2 and ("w1" in low or "w2" in low or "readout" in low):
            mat = grad.float().reshape(int(grad.shape[0]), -1)
            try:
                u, s, vh = torch.linalg.svd(mat, full_matrices=False)
                r = min(int(rank), int(s.numel()))
                approx = (u[:, :r] * s[:r]) @ vh[:r]
                if "w1" in low and ("w2" not in low and "readout" not in low):
                    approx = 0.50 * approx
                chunks.append(approx.reshape_as(grad).to(device=device, dtype=grad.dtype).reshape(-1))
            except Exception:
                chunks.append(grad.reshape(-1))
        else:
            chunks.append(torch.zeros_like(grad).reshape(-1))
    if not chunks:
        return torch.zeros(0, device=device)
    return torch.cat(chunks)


def _low_rank_from_named_gradient(pname: str, grad: torch.Tensor, device: torch.device, rank: int = 4) -> torch.Tensor:
    low = pname.lower()
    if grad.ndim >= 2 and ("w1" in low or "w2" in low or "readout" in low):
        mat = grad.float().reshape(int(grad.shape[0]), -1)
        try:
            u, s, vh = torch.linalg.svd(mat, full_matrices=False)
            r = min(int(rank), int(s.numel()))
            approx = (u[:, :r] * s[:r]) @ vh[:r]
            if "w1" in low and ("w2" not in low and "readout" not in low):
                approx = 0.50 * approx
            return approx.reshape_as(grad).to(device=device, dtype=grad.dtype)
        except Exception:
            return grad.to(device=device)
    return torch.zeros_like(grad, device=device)


def _split_consensus_low_rank_block_update(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device, rank: int = 4) -> torch.Tensor:
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not params:
        return torch.zeros(0, device=device)
    saved_grads = [None if p.grad is None else p.grad.detach().clone() for _, p in params]
    n = int(x.shape[0])
    split = max(1, n // 2)

    def grad_list(xb: torch.Tensor, yb: torch.Tensor) -> list[torch.Tensor]:
        model.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(xb).float(), yb)
        loss.backward()
        out: list[torch.Tensor] = []
        for _, p in params:
            out.append(torch.zeros_like(p) if p.grad is None else p.grad.detach().clone())
        return out

    g1 = grad_list(x[:split], y[:split])
    g2 = grad_list(x[split:] if split < n else x[:split], y[split:] if split < n else y[:split])
    chunks: list[torch.Tensor] = []
    for (pname, p), a, b in zip(params, g1, g2):
        agree = torch.sign(a) == torch.sign(b)
        consensus = torch.where(agree, 0.50 * (a + b), torch.zeros_like(a))
        chunks.append(_low_rank_from_named_gradient(pname, consensus, device, rank=rank).reshape(-1))
    model.zero_grad(set_to_none=True)
    for (_, p), grad in zip(params, saved_grads):
        p.grad = None if grad is None else grad.to(device=p.device, dtype=p.dtype)
    return torch.cat(chunks) if chunks else torch.zeros(0, device=device)


def _split_readout_consensus_update(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    *,
    classes: int = 10,
) -> tuple[torch.Tensor, dict[str, Any]]:
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not params or int(x.shape[0]) < 2:
        return flat_grad(model, device).detach().clone(), {"source_state_consensus_density": 0.0, "source_state_balance_mean": 0.0}
    saved_grads = [None if p.grad is None else p.grad.detach().clone() for _, p in params]
    n = int(x.shape[0])
    split = max(1, n // 2)

    def restore_grads() -> None:
        model.zero_grad(set_to_none=True)
        for (_, param), grad in zip(params, saved_grads):
            param.grad = None if grad is None else grad.to(device=param.device, dtype=param.dtype)

    def grad_flat(xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(xb).float(), yb).backward()
        chunks: list[torch.Tensor] = []
        for _, param in params:
            chunks.append(torch.zeros_like(param, device=device).reshape(-1) if param.grad is None else param.grad.detach().to(device=device).reshape(-1))
        return torch.cat(chunks) if chunks else torch.zeros(0, device=device)

    g_a = grad_flat(x[:split], y[:split])
    xb_b = x[split:] if split < n else x[:split]
    yb_b = y[split:] if split < n else y[:split]
    g_b = grad_flat(xb_b, yb_b)
    corrupt_labels = (yb_b + 1) % max(2, int(classes))
    g_corrupt = grad_flat(xb_b, corrupt_labels)
    restore_grads()

    agree = torch.sign(g_a) == torch.sign(g_b)
    density = float(agree.float().mean().item()) if agree.numel() else 0.0
    mag_a = g_a.detach().float().abs()
    mag_b = g_b.detach().float().abs()
    balance = torch.minimum(mag_a, mag_b) / torch.maximum(mag_a, mag_b).clamp_min(1.0e-8)
    balance_mean = float(balance[agree].mean().item()) if bool(agree.any().item()) else 0.0
    consensus = torch.where(agree, 0.50 * (g_a + g_b) * balance.to(device=device, dtype=g_a.dtype), torch.zeros_like(g_a))
    readout_mask = _readout_role_mask(model, device).to(device=device, dtype=consensus.dtype)
    if readout_mask.numel() == consensus.numel():
        consensus = consensus * readout_mask
    corrupt_denom = torch.dot(g_corrupt.float(), g_corrupt.float()).clamp_min(1.0e-12)
    corrupt_proj = torch.dot(consensus.float(), g_corrupt.float()) / corrupt_denom
    if float(corrupt_proj.item()) > 0.0:
        consensus = consensus - corrupt_proj.to(device=consensus.device, dtype=consensus.dtype) * g_corrupt
    diagnostics = {
        "source_state_consensus_density": density,
        "source_state_balance_mean": balance_mean,
        "source_state_corrupt_cos": cosine(consensus, g_corrupt),
        "source_state_corrupt_proj": float(corrupt_proj.item()),
        "source_state_gate_accept": int(density >= 0.05 and balance_mean >= 0.05 and float(torch.linalg.vector_norm(consensus.detach()).item()) > 1.0e-12),
        "source_state_whitened_norm": float(torch.linalg.vector_norm(consensus.detach()).item()),
    }
    return consensus, diagnostics


def _carrier_low_rank_from_grad_list(
    params: list[tuple[str, torch.nn.Parameter]],
    grads: list[torch.Tensor],
    device: torch.device,
    rank: int = 4,
) -> torch.Tensor:
    chunks = [_low_rank_from_named_gradient(pname, grad, device, rank=rank).reshape(-1) for (pname, _), grad in zip(params, grads)]
    return torch.cat(chunks) if chunks else torch.zeros(0, device=device)


def _readout_from_grad_list(
    params: list[tuple[str, torch.nn.Parameter]],
    grads: list[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for (pname, _p), grad in zip(params, grads):
        low = pname.lower()
        if "w2" in low or "readout" in low or "classifier" in low:
            chunks.append(grad.to(device=device).reshape(-1))
        else:
            chunks.append(torch.zeros_like(grad, device=device).reshape(-1))
    return torch.cat(chunks) if chunks else torch.zeros(0, device=device)


def _readout_class_directions_from_grad_list(
    params: list[tuple[str, torch.nn.Parameter]],
    grads: list[torch.Tensor],
    y: torch.Tensor,
    device: torch.device,
    max_classes: int = 8,
) -> list[torch.Tensor]:
    classes = [int(c) for c in torch.unique(y.detach()).cpu().tolist()[: int(max_classes)]]
    out: list[torch.Tensor] = []
    for cls in classes:
        chunks: list[torch.Tensor] = []
        any_nonzero = False
        for (pname, _p), grad in zip(params, grads):
            low = pname.lower()
            block = torch.zeros_like(grad, device=device)
            if ("w2" in low or "readout" in low or "classifier" in low) and grad.ndim >= 2:
                if int(grad.shape[1]) > cls:
                    if grad.ndim == 2:
                        block[:, cls] = grad[:, cls].to(device=device)
                    else:
                        block[:, cls, ...] = grad[:, cls, ...].to(device=device)
                    any_nonzero = True
                elif int(grad.shape[0]) > cls:
                    block[cls, ...] = grad[cls, ...].to(device=device)
                    any_nonzero = True
            chunks.append(block.reshape(-1))
        if any_nonzero and chunks:
            out.append(torch.cat(chunks))
    return out


def _readout_top_entry_directions_from_grad_list(
    params: list[tuple[str, torch.nn.Parameter]],
    grads: list[torch.Tensor],
    device: torch.device,
    max_dirs: int = 24,
) -> list[torch.Tensor]:
    candidates: list[tuple[float, int, int]] = []
    offsets: list[int] = []
    offset = 0
    for (pname, _p), grad in zip(params, grads):
        offsets.append(offset)
        low = pname.lower()
        if ("w2" in low or "readout" in low or "classifier" in low) and grad.numel():
            flat = grad.detach().float().reshape(-1)
            k = min(int(max_dirs), int(flat.numel()))
            if k > 0:
                vals, idxs = torch.topk(flat.abs(), k=k)
                for val, idx in zip(vals.cpu().tolist(), idxs.cpu().tolist()):
                    if float(val) > 0.0:
                        candidates.append((float(val), len(offsets) - 1, int(idx)))
        offset += int(grad.numel())
    candidates.sort(key=lambda item: item[0], reverse=True)
    total = sum(int(g.numel()) for g in grads)
    out: list[torch.Tensor] = []
    for _val, param_idx, local_idx in candidates[: int(max_dirs)]:
        vec = torch.zeros(total, device=device)
        vec[offsets[param_idx] + local_idx] = 1.0
        out.append(vec)
    return out


def _one_hot_like(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    target = torch.zeros_like(logits)
    target.scatter_(1, y.reshape(-1, 1), 1.0)
    return target


def _readout_target_like(
    kind: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    classes: int = 10,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    loss_target = _one_hot_like(logits, labels) - torch.softmax(logits, dim=1)
    if kind == "loss":
        return loss_target
    if kind == "sign_flip":
        return -loss_target
    if kind == "corrupt":
        corrupt = (labels + 1) % max(2, int(classes))
        return _one_hot_like(logits, corrupt) - torch.softmax(logits, dim=1)
    if kind == "weak_stable":
        row_norm = torch.linalg.vector_norm(loss_target.detach().float(), dim=1, keepdim=True).clamp_min(1.0e-12)
        scale = torch.median(row_norm).to(device=logits.device, dtype=logits.dtype).clamp_min(1.0e-6)
        return loss_target / row_norm.to(dtype=logits.dtype) * scale
    if kind == "loss_orthogonal_random":
        noise = torch.randn(loss_target.shape, device=logits.device, dtype=logits.dtype, generator=generator)
        noise = noise - noise.mean(dim=1, keepdim=True)
        noise_f = noise.detach().float()
        loss_f = loss_target.detach().float()
        coeff = (loss_f * noise_f).sum(dim=1, keepdim=True) / noise_f.square().sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        out = loss_f - coeff * noise_f
        out = out - out.mean(dim=1, keepdim=True)
        out_norm = torch.linalg.vector_norm(out, dim=1, keepdim=True).clamp_min(1.0e-12)
        loss_norm = torch.linalg.vector_norm(loss_f, dim=1, keepdim=True).clamp_min(1.0e-12)
        return (out * (loss_norm / out_norm)).to(dtype=logits.dtype)
    if kind == "top_wrong_margin":
        out = torch.zeros_like(loss_target)
        out.scatter_(1, labels.reshape(-1, 1), 1.0)
        masked = logits.detach().float().clone()
        masked.scatter_(1, labels.reshape(-1, 1), float("-inf"))
        wrong = torch.argmax(masked, dim=1)
        out.scatter_add_(1, wrong.reshape(-1, 1), -torch.ones((int(labels.numel()), 1), device=logits.device, dtype=logits.dtype))
        loss_norm = torch.linalg.vector_norm(loss_target.detach().float(), dim=1, keepdim=True).clamp_min(1.0e-12)
        out_norm = torch.linalg.vector_norm(out.detach().float(), dim=1, keepdim=True).clamp_min(1.0e-12)
        return out * (loss_norm.to(dtype=logits.dtype) / out_norm.to(dtype=logits.dtype))
    if kind == "random":
        noise = torch.randn(loss_target.shape, device=logits.device, dtype=logits.dtype, generator=generator)
        ref = reference if reference is not None else loss_target
        return noise * (torch.linalg.vector_norm(ref).clamp_min(1.0e-12) / torch.linalg.vector_norm(noise).clamp_min(1.0e-12))
    raise ValueError(f"unknown readout target kind: {kind}")


def _apply_direction_temporarily(model: torch.nn.Module, base: torch.Tensor, direction: torch.Tensor, scale: float) -> None:
    load_flat_params(model, base - float(scale) * direction.reshape(-1).to(device=base.device, dtype=base.dtype))


def _train_split_function_space_actuation_update(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    rank: int = 4,
    fd_eps: float = 1.0e-3,
    commit_scale: float = 1.0e-4,
    ridge: float = 1.0e-3,
    norm_cap_ratio: float = 100.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    g_ref = flat_grad(model, device)
    if not params or g_ref.numel() == 0 or int(x.shape[0]) < 3:
        return g_ref.clone(), {"ActuationR2": "", "ActuationCosine": "", "operator_rank": 0, "operator_status": "insufficient_batch"}

    base_params = flat_params(model).detach().to(device=device)
    saved_grads = [None if p.grad is None else p.grad.detach().clone() for _, p in params]
    n = int(x.shape[0])
    split1 = max(1, n // 3)
    split2 = max(split1 + 1, (2 * n) // 3)
    xb1, yb1 = x[:split1], y[:split1]
    xb2, yb2 = x[split1:split2], y[split1:split2]
    xb3, yb3 = x[split2:], y[split2:]
    if int(xb3.shape[0]) == 0:
        xb3, yb3 = xb2, yb2

    def grad_list(xb: torch.Tensor, yb: torch.Tensor) -> list[torch.Tensor]:
        model.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(xb).float(), yb)
        loss.backward()
        return [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for _, p in params]

    g1 = grad_list(xb1, yb1)
    g2 = grad_list(xb2, yb2)
    full_grads = [torch.zeros_like(p) if grad is None else grad.to(device=p.device, dtype=p.dtype) for (_, p), grad in zip(params, saved_grads)]
    block1 = _carrier_low_rank_from_grad_list(params, g1, device, rank=rank)
    block2 = _carrier_low_rank_from_grad_list(params, g2, device, rank=rank)
    consensus = _carrier_low_rank_from_grad_list(
        params,
        [torch.where(torch.sign(a) == torch.sign(b), 0.50 * (a + b), torch.zeros_like(a)) for a, b in zip(g1, g2)],
        device,
        rank=rank,
    )
    readout = _readout_from_grad_list(params, full_grads, device)
    class_dirs = _readout_class_directions_from_grad_list(params, full_grads, torch.cat([yb1, yb2], dim=0), device, max_classes=8)
    entry_dirs = _readout_top_entry_directions_from_grad_list(params, full_grads, device, max_dirs=24)
    candidates = [block1, block2, consensus, readout, *class_dirs, *entry_dirs]
    basis: list[torch.Tensor] = []
    for cand in candidates:
        if cand.numel() != g_ref.numel():
            continue
        nrm = torch.linalg.vector_norm(cand.detach())
        if float(nrm.item()) > 1.0e-12:
            basis.append(cand / nrm.clamp_min(1.0e-12))
    if not basis:
        model.zero_grad(set_to_none=True)
        for (_, p), grad in zip(params, saved_grads):
            p.grad = None if grad is None else grad.to(device=p.device, dtype=p.dtype)
        load_flat_params(model, base_params)
        return g_ref.clone(), {"ActuationR2": "", "ActuationCosine": "", "operator_rank": 0, "operator_status": "empty_basis"}

    with torch.no_grad():
        load_flat_params(model, base_params)
        base1 = model(xb1).detach().float()
        base2 = model(xb2).detach().float()
        base3 = model(xb3).detach().float()
        target1 = (_one_hot_like(base1, yb1) - torch.softmax(base1, dim=1)).reshape(-1)
        target2 = (_one_hot_like(base2, yb2) - torch.softmax(base2, dim=1)).reshape(-1)
        target12 = torch.cat([target1, target2], dim=0)
        cols: list[torch.Tensor] = []
        for direction in basis:
            _apply_direction_temporarily(model, base_params, direction, fd_eps)
            d1 = (model(xb1).detach().float() - base1) / float(fd_eps)
            d2 = (model(xb2).detach().float() - base2) / float(fd_eps)
            cols.append(torch.cat([d1.reshape(-1), d2.reshape(-1)], dim=0))
        load_flat_params(model, base_params)
    jmat = torch.stack(cols, dim=1)
    gram = jmat.T @ jmat + float(ridge) * torch.eye(len(basis), device=device, dtype=jmat.dtype)
    rhs = jmat.T @ (target12.to(device=device, dtype=jmat.dtype) / float(commit_scale))
    try:
        alpha = torch.linalg.solve(gram, rhs)
    except Exception:
        alpha = torch.linalg.lstsq(gram, rhs.unsqueeze(1)).solution.squeeze(1)
    solved = torch.zeros_like(g_ref)
    for a, direction in zip(alpha, basis):
        solved = solved + a.to(dtype=solved.dtype) * direction.to(dtype=solved.dtype)
    solved_norm = torch.linalg.vector_norm(solved.detach())
    g_norm = torch.linalg.vector_norm(g_ref.detach()).clamp_min(1.0e-12)
    target12_float = target12.to(device=device, dtype=torch.float32)

    def capped(scale_ratio: float) -> tuple[torch.Tensor, float]:
        cap = float(scale_ratio) * g_norm
        if float(solved_norm.item()) > float(cap.item()):
            ratio = float(cap.item() / solved_norm.clamp_min(1.0e-12).item())
            return solved * cap.to(device=solved.device, dtype=solved.dtype) / solved_norm.clamp_min(1.0e-12), ratio
        return solved, 1.0

    def measure(prop: torch.Tensor, clamp_ratio: float, cap_ratio: float) -> dict[str, Any]:
        with torch.no_grad():
            load_flat_params(model, base_params)
            b1_before = torch.nn.functional.cross_entropy(base1, yb1)
            b2_before = torch.nn.functional.cross_entropy(base2, yb2)
            b3_before = torch.nn.functional.cross_entropy(base3, yb3)
            _apply_direction_temporarily(model, base_params, prop, commit_scale)
            after1 = model(xb1).detach().float()
            after2 = model(xb2).detach().float()
            after3 = model(xb3).detach().float()
            b1_after = torch.nn.functional.cross_entropy(after1, yb1)
            b2_after = torch.nn.functional.cross_entropy(after2, yb2)
            b3_after = torch.nn.functional.cross_entropy(after3, yb3)
            actual12 = torch.cat([(after1 - base1).reshape(-1), (after2 - base2).reshape(-1)], dim=0)
            residual = target12_float.to(dtype=actual12.dtype) - actual12
            centered = target12_float.to(dtype=actual12.dtype) - target12_float.to(dtype=actual12.dtype).mean()
            denom = torch.sum(centered.square()).clamp_min(1.0e-12)
            actuation_r2 = 1.0 - torch.sum(residual.square()) / denom
            cos_denom = torch.linalg.vector_norm(actual12) * torch.linalg.vector_norm(target12_float.to(dtype=actual12.dtype))
            actuation_cos = (actual12 @ target12_float.to(dtype=actual12.dtype) / cos_denom.clamp_min(1.0e-12)).clamp(-1.0, 1.0)
            load_flat_params(model, base_params)
        b1_gain = float((b1_before - b1_after).item())
        b2_gain = float((b2_before - b2_after).item())
        b3_gain = float((b3_before - b3_after).item())
        score = (b1_gain + b2_gain) + 0.25 * b3_gain + max(0.0, float(actuation_r2.item()))
        if b3_gain < -1.0e-3:
            score -= 10.0 * abs(b3_gain)
        return {
            "proposed": prop,
            "ActuationR2": float(actuation_r2.item()),
            "ActuationCosine": float(actuation_cos.item()),
            "B1_gain": b1_gain,
            "B2_transfer_gain": b2_gain,
            "B3_safety_gain": b3_gain,
            "projection_residual_norm": float(torch.linalg.vector_norm(residual).item() / torch.linalg.vector_norm(target12_float).clamp_min(1.0e-12).item()),
            "function_displacement_norm": float(torch.linalg.vector_norm(actual12).item()),
            "parameter_norm": float(torch.linalg.vector_norm(prop.detach()).item()),
            "operator_clamp_ratio": clamp_ratio,
            "operator_cap_ratio": float(cap_ratio),
            "score": score,
        }

    measured: list[dict[str, Any]] = []
    for cap_ratio in [float(norm_cap_ratio), 300.0, 1000.0, 3000.0, 10000.0, 30000.0]:
        prop, clamp_ratio = capped(cap_ratio)
        measured.append(measure(prop, clamp_ratio, cap_ratio))
    best = max(measured, key=lambda item: float(item["score"]))
    proposed = best["proposed"]

    model.zero_grad(set_to_none=True)
    for (_, p), grad in zip(params, saved_grads):
        p.grad = None if grad is None else grad.to(device=p.device, dtype=p.dtype)
    diagnostics = {
        "ActuationR2": best["ActuationR2"],
        "ActuationCosine": best["ActuationCosine"],
        "B1_gain": best["B1_gain"],
        "B2_transfer_gain": best["B2_transfer_gain"],
        "B3_safety_gain": best["B3_safety_gain"],
        "projection_residual_norm": best["projection_residual_norm"],
        "function_displacement_norm": best["function_displacement_norm"],
        "parameter_norm": best["parameter_norm"],
        "operator_clamp_ratio": best["operator_clamp_ratio"],
        "operator_cap_ratio": best["operator_cap_ratio"],
        "operator_commit_scale": float(commit_scale),
        "operator_rank": len(basis),
        "operator_status": "measured",
    }
    return proposed, diagnostics


def _exact_readout_function_space_actuation_update(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    commit_scale: float = 1.0e-4,
    ridge: float = 1.0e-3,
    norm_cap_ratio: float = 1000.0,
    target_kind: str = "loss",
    fit_scope: str = "b1b2",
    rank_limit: int = 0,
    feature_select: str = "top_norm",
    seed: int = 0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    g_ref = flat_grad(model, device)
    w2_param: torch.nn.Parameter | None = None
    w2_name = ""
    for pname, p in params:
        if "w2" in pname.lower() and p.ndim in {2, 3}:
            w2_name = pname
            w2_param = p
            break
    if w2_param is None or not hasattr(model, "frozen_readout_features") or int(x.shape[0]) < 3:
        return g_ref.clone(), {"ActuationR2": "", "ActuationCosine": "", "operator_rank": 0, "operator_status": "exact_readout_unavailable"}

    base_params = flat_params(model).detach().to(device=device)
    saved_grads = [None if p.grad is None else p.grad.detach().clone() for _, p in params]
    n = int(x.shape[0])
    split1 = max(1, n // 3)
    split2 = max(split1 + 1, (2 * n) // 3)
    xb1, yb1 = x[:split1], y[:split1]
    xb2, yb2 = x[split1:split2], y[split1:split2]
    xb3, yb3 = x[split2:], y[split2:]
    if int(xb3.shape[0]) == 0:
        xb3, yb3 = xb2, yb2

    with torch.no_grad():
        load_flat_params(model, base_params)
        base1 = model(xb1).detach().float()
        base2 = model(xb2).detach().float()
        base3 = model(xb3).detach().float()
        feats1 = model.frozen_readout_features(xb1).detach().float()
        feats2 = model.frozen_readout_features(xb2).detach().float()
        feats3 = model.frozen_readout_features(xb3).detach().float()
        hdim = int(w2_param.shape[0])
        classes = int(w2_param.shape[1])
        readout_is_basis_tensor = w2_param.ndim == 3
        kval = int(w2_param.shape[2]) if readout_is_basis_tensor else 1
        n_readout = hdim * kval
        feature_scale = float(max(1, hdim)) ** 0.5 if readout_is_basis_tensor else 1.0
        feats1 = feats1[:, :n_readout] / feature_scale
        feats2 = feats2[:, :n_readout] / feature_scale
        feats3 = feats3[:, :n_readout] / feature_scale
        gen = torch.Generator(device=device).manual_seed(701_003 + int(seed) + sum(ord(c) for c in str(target_kind)))
        ref1 = _one_hot_like(base1, yb1) - torch.softmax(base1, dim=1)
        ref2 = _one_hot_like(base2, yb2) - torch.softmax(base2, dim=1)
        target_diag: dict[str, Any] = {}
        if target_kind == "split_consensus":
            sign1 = torch.sign(ref1.detach().float().mean(dim=0))
            sign2 = torch.sign(ref2.detach().float().mean(dim=0))
            mask = ((sign1 == sign2) & (sign1 != 0.0)).to(device=device, dtype=ref1.dtype)
            target1 = ref1 * mask.reshape(1, -1)
            target2 = ref2 * mask.reshape(1, -1)
            target_diag["target_consensus_density"] = float(mask.detach().float().mean().item()) if mask.numel() else 0.0
        elif target_kind == "poprisk_snr":
            ref_all = torch.cat([ref1.detach().float(), ref2.detach().float()], dim=0)
            mu = ref_all.mean(dim=0)
            var = ref_all.var(dim=0, unbiased=False)
            snr = mu.square() / (var + 1.0e-8)
            snr_weight = snr / snr.max().clamp_min(1.0e-8)
            sign1 = torch.sign(ref1.detach().float().mean(dim=0))
            sign2 = torch.sign(ref2.detach().float().mean(dim=0))
            mask = ((sign1 == sign2) & (sign1 != 0.0)).to(device=device, dtype=ref1.dtype)
            weights = (0.25 + 0.75 * snr_weight.to(device=device, dtype=ref1.dtype)) * mask
            target1 = ref1 * weights.reshape(1, -1)
            target2 = ref2 * weights.reshape(1, -1)
            target_diag["target_consensus_density"] = float(mask.detach().float().mean().item()) if mask.numel() else 0.0
            target_diag["PopRisk_SNR_score"] = float(snr.detach().mean().item()) if snr.numel() else 0.0
            target_diag["PopRisk_snr_mean"] = float(snr.detach().mean().item()) if snr.numel() else 0.0
            target_diag["PopRisk_snr_max"] = float(snr.detach().max().item()) if snr.numel() else 0.0
            target_diag["PopRisk_mu_norm"] = float(torch.linalg.vector_norm(mu.detach()).item()) if mu.numel() else 0.0
            target_diag["PopRisk_var_mean"] = float(var.detach().mean().item()) if var.numel() else 0.0
            target_diag["PopRisk_target_weight_mean"] = float(weights.detach().float().mean().item()) if weights.numel() else 0.0
        elif target_kind == "easy_split_consensus":
            sign1 = torch.sign(ref1.detach().float().mean(dim=0))
            sign2 = torch.sign(ref2.detach().float().mean(dim=0))
            class_mask = ((sign1 == sign2) & (sign1 != 0.0)).to(device=device, dtype=ref1.dtype)

            def easy_mask(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                ce = torch.nn.functional.cross_entropy(logits.float(), labels, reduction="none")
                cutoff = ce.detach().float().median()
                predicted = logits.detach().float().argmax(dim=1) == labels
                low_loss = ce <= cutoff
                selected_mask = (predicted | low_loss).to(device=device, dtype=ref1.dtype)
                return selected_mask.reshape(-1, 1)

            easy1 = easy_mask(base1, yb1)
            easy2 = easy_mask(base2, yb2)
            target1 = ref1 * class_mask.reshape(1, -1) * easy1
            target2 = ref2 * class_mask.reshape(1, -1) * easy2
            target_diag["target_consensus_density"] = float(class_mask.detach().float().mean().item()) if class_mask.numel() else 0.0
            target_diag["target_easy_fraction_b1"] = float(easy1.detach().float().mean().item()) if easy1.numel() else 0.0
            target_diag["target_easy_fraction_b2"] = float(easy2.detach().float().mean().item()) if easy2.numel() else 0.0
        elif target_kind == "loss_easy_consensus_blend":
            sign1 = torch.sign(ref1.detach().float().mean(dim=0))
            sign2 = torch.sign(ref2.detach().float().mean(dim=0))
            class_mask = ((sign1 == sign2) & (sign1 != 0.0)).to(device=device, dtype=ref1.dtype)

            def easy_mask(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                ce = torch.nn.functional.cross_entropy(logits.float(), labels, reduction="none")
                cutoff = ce.detach().float().median()
                predicted = logits.detach().float().argmax(dim=1) == labels
                low_loss = ce <= cutoff
                selected_mask = (predicted | low_loss).to(device=device, dtype=ref1.dtype)
                return selected_mask.reshape(-1, 1)

            easy1 = easy_mask(base1, yb1)
            easy2 = easy_mask(base2, yb2)
            consensus1 = ref1 * class_mask.reshape(1, -1) * easy1
            consensus2 = ref2 * class_mask.reshape(1, -1) * easy2
            target1 = 0.65 * ref1 + 0.35 * consensus1
            target2 = 0.65 * ref2 + 0.35 * consensus2
            target_diag["target_consensus_density"] = float(class_mask.detach().float().mean().item()) if class_mask.numel() else 0.0
            target_diag["target_easy_fraction_b1"] = float(easy1.detach().float().mean().item()) if easy1.numel() else 0.0
            target_diag["target_easy_fraction_b2"] = float(easy2.detach().float().mean().item()) if easy2.numel() else 0.0
            target_diag["target_loss_blend_weight"] = 0.65
            target_diag["target_consensus_blend_weight"] = 0.35
        elif target_kind == "view_consistent_loss":
            noise_std = 0.015
            noise1 = torch.randn(xb1.shape, device=device, dtype=xb1.dtype, generator=gen) * noise_std
            noise2 = torch.randn(xb2.shape, device=device, dtype=xb2.dtype, generator=gen) * noise_std
            aug1 = model(xb1 + noise1).detach().float()
            aug2 = model(xb2 + noise2).detach().float()
            aug_ref1 = _one_hot_like(aug1, yb1) - torch.softmax(aug1, dim=1)
            aug_ref2 = _one_hot_like(aug2, yb2) - torch.softmax(aug2, dim=1)

            def view_mask(clean: torch.Tensor, aug: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                num = (clean.detach().float() * aug.detach().float()).sum(dim=1)
                denom = (
                    torch.linalg.vector_norm(clean.detach().float(), dim=1)
                    * torch.linalg.vector_norm(aug.detach().float(), dim=1)
                ).clamp_min(1.0e-12)
                align = (num / denom).clamp(-1.0, 1.0)
                return (align >= 0.25).to(device=device, dtype=clean.dtype).reshape(-1, 1), align

            stable1, align1 = view_mask(ref1, aug_ref1)
            stable2, align2 = view_mask(ref2, aug_ref2)
            clean_sign1 = torch.sign(ref1.detach().float().mean(dim=0))
            clean_sign2 = torch.sign(ref2.detach().float().mean(dim=0))
            aug_sign1 = torch.sign(aug_ref1.detach().float().mean(dim=0))
            aug_sign2 = torch.sign(aug_ref2.detach().float().mean(dim=0))
            class_mask = (
                (clean_sign1 == clean_sign2)
                & (clean_sign1 == aug_sign1)
                & (clean_sign2 == aug_sign2)
                & (clean_sign1 != 0.0)
            ).to(device=device, dtype=ref1.dtype)
            target1 = ref1 * stable1 * class_mask.reshape(1, -1)
            target2 = ref2 * stable2 * class_mask.reshape(1, -1)
            target_diag["target_view_noise_std"] = noise_std
            target_diag["target_view_stable_fraction_b1"] = float(stable1.detach().float().mean().item()) if stable1.numel() else 0.0
            target_diag["target_view_stable_fraction_b2"] = float(stable2.detach().float().mean().item()) if stable2.numel() else 0.0
            target_diag["target_view_alignment_b1_mean"] = float(align1.detach().float().mean().item()) if align1.numel() else 0.0
            target_diag["target_view_alignment_b2_mean"] = float(align2.detach().float().mean().item()) if align2.numel() else 0.0
            target_diag["target_consensus_density"] = float(class_mask.detach().float().mean().item()) if class_mask.numel() else 0.0
        elif target_kind == "source_projected_consensus":
            corrupt_y1 = (yb1 + 1) % max(2, classes)
            corrupt_y2 = (yb2 + 1) % max(2, classes)
            corrupt_ref1 = _one_hot_like(base1, corrupt_y1) - torch.softmax(base1, dim=1)
            corrupt_ref2 = _one_hot_like(base2, corrupt_y2) - torch.softmax(base2, dim=1)
            clean_sign1 = torch.sign(ref1.detach().float().mean(dim=0))
            clean_sign2 = torch.sign(ref2.detach().float().mean(dim=0))
            corrupt_sign1 = torch.sign(corrupt_ref1.detach().float().mean(dim=0))
            corrupt_sign2 = torch.sign(corrupt_ref2.detach().float().mean(dim=0))
            clean_mask = (clean_sign1 == clean_sign2) & (clean_sign1 != 0.0)
            noise_mask = (corrupt_sign1 == corrupt_sign2) & (corrupt_sign1 == clean_sign1) & (corrupt_sign1 != 0.0)
            class_mask = (clean_mask & ~noise_mask).to(device=device, dtype=ref1.dtype)
            confidence1 = torch.softmax(base1, dim=1).gather(1, yb1.reshape(-1, 1)).detach()
            confidence2 = torch.softmax(base2, dim=1).gather(1, yb2.reshape(-1, 1)).detach()
            weight1 = (1.0 - confidence1).clamp(0.15, 1.0)
            weight2 = (1.0 - confidence2).clamp(0.15, 1.0)
            target1 = ref1 * class_mask.reshape(1, -1) * weight1
            target2 = ref2 * class_mask.reshape(1, -1) * weight2
            target_diag["target_consensus_density"] = float(clean_mask.detach().float().mean().item()) if clean_mask.numel() else 0.0
            target_diag["target_noise_rejected_density"] = float(noise_mask.detach().float().mean().item()) if noise_mask.numel() else 0.0
            target_diag["target_source_projected_density"] = float(class_mask.detach().float().mean().item()) if class_mask.numel() else 0.0
        elif target_kind == "debt_calibrated_source":
            corrupt_y1 = (yb1 + 1) % max(2, classes)
            corrupt_y2 = (yb2 + 1) % max(2, classes)
            corrupt_ref1 = _one_hot_like(base1, corrupt_y1) - torch.softmax(base1, dim=1)
            corrupt_ref2 = _one_hot_like(base2, corrupt_y2) - torch.softmax(base2, dim=1)
            clean_sign1 = torch.sign(ref1.detach().float().mean(dim=0))
            clean_sign2 = torch.sign(ref2.detach().float().mean(dim=0))
            corrupt_sign1 = torch.sign(corrupt_ref1.detach().float().mean(dim=0))
            corrupt_sign2 = torch.sign(corrupt_ref2.detach().float().mean(dim=0))
            clean_mask = (clean_sign1 == clean_sign2) & (clean_sign1 != 0.0)
            noise_mask = (corrupt_sign1 == corrupt_sign2) & (corrupt_sign1 == clean_sign1) & (corrupt_sign1 != 0.0)
            class_mask = (clean_mask & ~noise_mask).to(device=device, dtype=ref1.dtype)

            def debt_weight(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
                logits_f = logits.detach().float()
                ce = torch.nn.functional.cross_entropy(logits_f, labels, reduction="none")
                p_true = torch.softmax(logits_f, dim=1).gather(1, labels.reshape(-1, 1)).reshape(-1)
                median = ce.quantile(0.50)
                q85 = ce.quantile(0.85)
                q95 = ce.quantile(0.95)
                span = (q85 - median).clamp_min(1.0e-6)
                transition = ((ce - median) / span).clamp(0.0, 1.0)
                non_outlier = (ce <= q95).to(device=device, dtype=ref1.dtype)
                calibration_gap = (1.0 - p_true).clamp(0.10, 1.0)
                weight = (0.20 + 0.80 * transition).to(device=device, dtype=ref1.dtype)
                weight = weight * calibration_gap.to(device=device, dtype=ref1.dtype) * non_outlier
                stats = {
                    "ce_median": float(median.detach().item()),
                    "ce_q85": float(q85.detach().item()),
                    "ce_q95": float(q95.detach().item()),
                    "calibration_gap_mean": float((1.0 - p_true).detach().mean().item()) if p_true.numel() else 0.0,
                    "debt_weight_mean": float(weight.detach().float().mean().item()) if weight.numel() else 0.0,
                    "debt_transition_mean": float(transition.detach().float().mean().item()) if transition.numel() else 0.0,
                    "debt_non_outlier_fraction": float(non_outlier.detach().float().mean().item()) if non_outlier.numel() else 0.0,
                }
                return weight.reshape(-1, 1), stats

            weight1, stats1 = debt_weight(base1, yb1)
            weight2, stats2 = debt_weight(base2, yb2)
            target1 = ref1 * class_mask.reshape(1, -1) * weight1
            target2 = ref2 * class_mask.reshape(1, -1) * weight2
            target_diag["target_consensus_density"] = float(clean_mask.detach().float().mean().item()) if clean_mask.numel() else 0.0
            target_diag["target_noise_rejected_density"] = float(noise_mask.detach().float().mean().item()) if noise_mask.numel() else 0.0
            target_diag["target_source_projected_density"] = float(class_mask.detach().float().mean().item()) if class_mask.numel() else 0.0
            target_diag["target_train_ce_median_b1"] = stats1["ce_median"]
            target_diag["target_train_ce_median_b2"] = stats2["ce_median"]
            target_diag["target_train_ce_q85_b1"] = stats1["ce_q85"]
            target_diag["target_train_ce_q85_b2"] = stats2["ce_q85"]
            target_diag["target_train_ce_q95_b1"] = stats1["ce_q95"]
            target_diag["target_train_ce_q95_b2"] = stats2["ce_q95"]
            target_diag["target_calibration_gap_mean_b1"] = stats1["calibration_gap_mean"]
            target_diag["target_calibration_gap_mean_b2"] = stats2["calibration_gap_mean"]
            target_diag["target_debt_weight_mean_b1"] = stats1["debt_weight_mean"]
            target_diag["target_debt_weight_mean_b2"] = stats2["debt_weight_mean"]
            target_diag["target_debt_transition_mean_b1"] = stats1["debt_transition_mean"]
            target_diag["target_debt_transition_mean_b2"] = stats2["debt_transition_mean"]
            target_diag["target_debt_non_outlier_fraction_b1"] = stats1["debt_non_outlier_fraction"]
            target_diag["target_debt_non_outlier_fraction_b2"] = stats2["debt_non_outlier_fraction"]
        elif target_kind == "observable_mid_debt_source":
            corrupt_y1 = (yb1 + 1) % max(2, classes)
            corrupt_y2 = (yb2 + 1) % max(2, classes)
            corrupt_ref1 = _one_hot_like(base1, corrupt_y1) - torch.softmax(base1, dim=1)
            corrupt_ref2 = _one_hot_like(base2, corrupt_y2) - torch.softmax(base2, dim=1)
            clean_sign1 = torch.sign(ref1.detach().float().mean(dim=0))
            clean_sign2 = torch.sign(ref2.detach().float().mean(dim=0))
            corrupt_sign1 = torch.sign(corrupt_ref1.detach().float().mean(dim=0))
            corrupt_sign2 = torch.sign(corrupt_ref2.detach().float().mean(dim=0))
            clean_mask = (clean_sign1 == clean_sign2) & (clean_sign1 != 0.0)
            noise_mask = (corrupt_sign1 == corrupt_sign2) & (corrupt_sign1 == clean_sign1) & (corrupt_sign1 != 0.0)
            class_mask = (clean_mask & ~noise_mask).to(device=device, dtype=ref1.dtype)

            def mid_debt_weight(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
                logits_f = logits.detach().float()
                ce = torch.nn.functional.cross_entropy(logits_f, labels, reduction="none")
                p_true = torch.softmax(logits_f, dim=1).gather(1, labels.reshape(-1, 1)).reshape(-1)
                q25 = ce.quantile(0.25)
                q50 = ce.quantile(0.50)
                q85 = ce.quantile(0.85)
                rise = ((ce - q25) / (q50 - q25).clamp_min(1.0e-6)).clamp(0.0, 1.0)
                fall = ((q85 - ce) / (q85 - q50).clamp_min(1.0e-6)).clamp(0.0, 1.0)
                midband = torch.minimum(rise, fall)
                selected = ((ce >= q25) & (ce <= q85)).to(device=device, dtype=ref1.dtype)
                calibration_gap = (1.0 - p_true).clamp(0.05, 0.85)
                weight = (0.10 + 0.90 * midband).to(device=device, dtype=ref1.dtype)
                weight = weight * calibration_gap.to(device=device, dtype=ref1.dtype) * selected
                stats = {
                    "ce_q25": float(q25.detach().item()),
                    "ce_median": float(q50.detach().item()),
                    "ce_q85": float(q85.detach().item()),
                    "calibration_gap_mean": float((1.0 - p_true).detach().mean().item()) if p_true.numel() else 0.0,
                    "midband_fraction": float(selected.detach().float().mean().item()) if selected.numel() else 0.0,
                    "midband_shape_mean": float(midband.detach().float().mean().item()) if midband.numel() else 0.0,
                    "debt_weight_mean": float(weight.detach().float().mean().item()) if weight.numel() else 0.0,
                }
                return weight.reshape(-1, 1), stats

            weight1, stats1 = mid_debt_weight(base1, yb1)
            weight2, stats2 = mid_debt_weight(base2, yb2)
            target1 = ref1 * class_mask.reshape(1, -1) * weight1
            target2 = ref2 * class_mask.reshape(1, -1) * weight2
            target_diag["target_consensus_density"] = float(clean_mask.detach().float().mean().item()) if clean_mask.numel() else 0.0
            target_diag["target_noise_rejected_density"] = float(noise_mask.detach().float().mean().item()) if noise_mask.numel() else 0.0
            target_diag["target_source_projected_density"] = float(class_mask.detach().float().mean().item()) if class_mask.numel() else 0.0
            target_diag["target_train_ce_q25_b1"] = stats1["ce_q25"]
            target_diag["target_train_ce_q25_b2"] = stats2["ce_q25"]
            target_diag["target_train_ce_median_b1"] = stats1["ce_median"]
            target_diag["target_train_ce_median_b2"] = stats2["ce_median"]
            target_diag["target_train_ce_q85_b1"] = stats1["ce_q85"]
            target_diag["target_train_ce_q85_b2"] = stats2["ce_q85"]
            target_diag["target_calibration_gap_mean_b1"] = stats1["calibration_gap_mean"]
            target_diag["target_calibration_gap_mean_b2"] = stats2["calibration_gap_mean"]
            target_diag["target_mid_debt_fraction_b1"] = stats1["midband_fraction"]
            target_diag["target_mid_debt_fraction_b2"] = stats2["midband_fraction"]
            target_diag["target_mid_debt_shape_mean_b1"] = stats1["midband_shape_mean"]
            target_diag["target_mid_debt_shape_mean_b2"] = stats2["midband_shape_mean"]
            target_diag["target_debt_weight_mean_b1"] = stats1["debt_weight_mean"]
            target_diag["target_debt_weight_mean_b2"] = stats2["debt_weight_mean"]
        elif target_kind == "noise_orthogonal_consensus":
            corrupt_y1 = (yb1 + 1) % max(2, classes)
            corrupt_y2 = (yb2 + 1) % max(2, classes)
            corrupt_ref1 = _one_hot_like(base1, corrupt_y1) - torch.softmax(base1, dim=1)
            corrupt_ref2 = _one_hot_like(base2, corrupt_y2) - torch.softmax(base2, dim=1)

            def orthogonalize(clean: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
                denom = noise.detach().float().square().sum(dim=1, keepdim=True).clamp_min(1.0e-12)
                coeff = (clean.detach().float() * noise.detach().float()).sum(dim=1, keepdim=True) / denom
                return clean - coeff.to(device=device, dtype=clean.dtype) * noise

            ortho1 = orthogonalize(ref1, corrupt_ref1)
            ortho2 = orthogonalize(ref2, corrupt_ref2)
            sign1 = torch.sign(ortho1.detach().float().mean(dim=0))
            sign2 = torch.sign(ortho2.detach().float().mean(dim=0))
            class_mask = ((sign1 == sign2) & (sign1 != 0.0)).to(device=device, dtype=ref1.dtype)
            target1 = ortho1 * class_mask.reshape(1, -1)
            target2 = ortho2 * class_mask.reshape(1, -1)
            noise_cos1 = (
                (target1.detach().float() * corrupt_ref1.detach().float()).sum(dim=1)
                / (
                    torch.linalg.vector_norm(target1.detach().float(), dim=1)
                    * torch.linalg.vector_norm(corrupt_ref1.detach().float(), dim=1)
                ).clamp_min(1.0e-12)
            )
            noise_cos2 = (
                (target2.detach().float() * corrupt_ref2.detach().float()).sum(dim=1)
                / (
                    torch.linalg.vector_norm(target2.detach().float(), dim=1)
                    * torch.linalg.vector_norm(corrupt_ref2.detach().float(), dim=1)
                ).clamp_min(1.0e-12)
            )
            target_diag["target_consensus_density"] = float(class_mask.detach().float().mean().item()) if class_mask.numel() else 0.0
            target_diag["target_noise_cos_abs_mean"] = float(torch.cat([noise_cos1.abs(), noise_cos2.abs()]).mean().item())
            target_diag["target_random_orthogonalized"] = 1
        elif target_kind == "easy_margin_consensus":
            margin1 = _readout_target_like("top_wrong_margin", base1, yb1, generator=gen, classes=classes, reference=ref1)
            margin2 = _readout_target_like("top_wrong_margin", base2, yb2, generator=gen, classes=classes, reference=ref2)
            sign1 = torch.sign(ref1.detach().float().mean(dim=0))
            sign2 = torch.sign(ref2.detach().float().mean(dim=0))
            class_mask = ((sign1 == sign2) & (sign1 != 0.0)).to(device=device, dtype=ref1.dtype)

            def hard_but_stable(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                ce = torch.nn.functional.cross_entropy(logits.float(), labels, reduction="none")
                low = ce.detach().float().quantile(0.25)
                high = ce.detach().float().quantile(0.85)
                return ((ce >= low) & (ce <= high)).to(device=device, dtype=ref1.dtype).reshape(-1, 1)

            stable1 = hard_but_stable(base1, yb1)
            stable2 = hard_but_stable(base2, yb2)
            target1 = margin1 * class_mask.reshape(1, -1) * stable1
            target2 = margin2 * class_mask.reshape(1, -1) * stable2
            target_diag["target_consensus_density"] = float(class_mask.detach().float().mean().item()) if class_mask.numel() else 0.0
            target_diag["target_easy_fraction_b1"] = float(stable1.detach().float().mean().item()) if stable1.numel() else 0.0
            target_diag["target_easy_fraction_b2"] = float(stable2.detach().float().mean().item()) if stable2.numel() else 0.0
            target_diag["target_top_wrong_margin"] = 1
        else:
            target1 = _readout_target_like(target_kind, base1, yb1, generator=gen, classes=classes, reference=ref1)
            target2 = _readout_target_like(target_kind, base2, yb2, generator=gen, classes=classes, reference=ref2)
            target_diag["target_consensus_density"] = ""
            target_diag["target_random_orthogonalized"] = int(target_kind == "loss_orthogonal_random")
            target_diag["target_top_wrong_margin"] = int(target_kind == "top_wrong_margin")
        measure_target = torch.cat([target1, target2], dim=0)
        if fit_scope == "b1":
            feats = feats1
            target = target1
        elif fit_scope == "b1_b3zero":
            zero3 = torch.zeros((base3.shape[0], classes), device=device, dtype=ref1.dtype)
            feats = torch.cat([feats1, feats3], dim=0)
            target = torch.cat([target1, zero3], dim=0)
            target_diag["target_b3_null_rows"] = int(zero3.shape[0])
        elif fit_scope == "b1b2_b3zero":
            zero3 = torch.zeros((base3.shape[0], classes), device=device, dtype=ref1.dtype)
            feats = torch.cat([feats1, feats2, feats3], dim=0)
            target = torch.cat([target1, target2, zero3], dim=0)
            target_diag["target_b3_null_rows"] = int(zero3.shape[0])
        else:
            feats = torch.cat([feats1, feats2], dim=0)
            target = torch.cat([target1, target2], dim=0)
        selected = None
        if int(rank_limit) > 0 and feats.shape[1] > int(rank_limit):
            feats_float = feats.detach().float()
            k = int(rank_limit)
            norms = torch.linalg.vector_norm(feats_float, dim=0)
            if feature_select == "head":
                selected = torch.arange(k, device=device)
            elif feature_select in {"low_bank", "source_bank"}:
                if readout_is_basis_tensor:
                    per_hidden = max(1, min(int(kval), k))
                    selected = torch.arange(n_readout, device=device).reshape(hdim, kval)[:, :per_hidden].reshape(-1)
                else:
                    selected = torch.arange(k, device=device)
            elif feature_select in {"reservoir_bank", "tail_bank"}:
                if readout_is_basis_tensor:
                    per_hidden = max(1, min(int(kval), k))
                    selected = torch.arange(n_readout, device=device).reshape(hdim, kval)[:, -per_hidden:].reshape(-1)
                else:
                    selected = torch.arange(n_readout - k, n_readout, device=device)
            elif feature_select == "stable":
                mean_abs = feats_float.abs().mean(dim=0)
                std = feats_float.std(dim=0, unbiased=False)
                scores = mean_abs / std.clamp_min(1.0e-6)
                selected = torch.topk(scores, k=k).indices
            else:
                selected = torch.topk(norms, k=k).indices
            reservoir_mask = torch.ones(int(feats.shape[1]), device=device, dtype=torch.bool)
            reservoir_mask[selected] = False
            source_norm = norms[selected].sum()
            reservoir_norm = norms[reservoir_mask].sum() if bool(reservoir_mask.any().item()) else torch.tensor(0.0, device=device)
            denom_norm = (source_norm + reservoir_norm).clamp_min(1.0e-12)
            prob = norms / norms.sum().clamp_min(1.0e-12)
            entropy = -(prob * prob.clamp_min(1.0e-12).log()).sum() / torch.log(torch.tensor(float(norms.numel()), device=device))
            feats = feats[:, selected]
            target_diag["target_selected_feature_count"] = int(selected.numel())
            target_diag["target_feature_select"] = str(feature_select)
            target_diag["source_bank_feature_count"] = int(selected.numel())
            target_diag["reservoir_bank_feature_count"] = int(reservoir_mask.sum().item())
            target_diag["source_bank_feature_norm"] = float(source_norm.detach().item())
            target_diag["reservoir_bank_feature_norm"] = float(reservoir_norm.detach().item())
            target_diag["source_channel_projection"] = float((source_norm / denom_norm).detach().item())
            target_diag["reservoir_projection"] = float((reservoir_norm / denom_norm).detach().item())
            target_diag["low_degree_source_energy"] = float(source_norm.detach().item())
            target_diag["high_degree_reservoir_energy"] = float(reservoir_norm.detach().item())
            target_diag["low_frequency_source_energy"] = float(source_norm.detach().item())
            target_diag["high_frequency_reservoir_energy"] = float(reservoir_norm.detach().item())
            target_diag["degree_entropy"] = float(entropy.detach().item())
            target_diag["band_entropy"] = float(entropy.detach().item())
        else:
            target_diag["target_selected_feature_count"] = int(feats.shape[1])
            target_diag["target_feature_select"] = str(feature_select)
            target_diag["source_bank_feature_count"] = int(feats.shape[1])
            target_diag["reservoir_bank_feature_count"] = 0
            target_diag["source_bank_feature_norm"] = float(torch.linalg.vector_norm(feats.detach().float(), dim=0).sum().item())
            target_diag["reservoir_bank_feature_norm"] = 0.0
            target_diag["source_channel_projection"] = 1.0
            target_diag["reservoir_projection"] = 0.0
    gram = feats.T @ feats + float(ridge) * torch.eye(feats.shape[1], device=device, dtype=feats.dtype)
    rhs = feats.T @ target.to(device=device, dtype=feats.dtype)
    try:
        delta_small = torch.linalg.solve(gram, rhs)
    except Exception:
        delta_small = torch.linalg.lstsq(gram, rhs).solution
    if selected is None:
        delta_matrix = delta_small
    else:
        delta_matrix = torch.zeros(n_readout, target.shape[1], device=device, dtype=feats.dtype)
        delta_matrix[selected] = delta_small
    if w2_param.ndim == 3:
        delta_w2 = delta_matrix.reshape(hdim, kval, classes).permute(0, 2, 1).contiguous()
    else:
        delta_w2 = delta_matrix.reshape(hdim, classes).contiguous()
    chunks: list[torch.Tensor] = []
    for pname, p in params:
        if pname == w2_name:
            chunks.append((-delta_w2 / float(commit_scale)).to(device=device, dtype=p.dtype).reshape(-1))
        else:
            chunks.append(torch.zeros_like(p, device=device).reshape(-1))
    solved = torch.cat(chunks) if chunks else torch.zeros_like(g_ref)
    solved_norm = torch.linalg.vector_norm(solved.detach())
    g_norm = torch.linalg.vector_norm(g_ref.detach()).clamp_min(1.0e-12)
    target_flat = measure_target.reshape(-1).to(device=device, dtype=torch.float32)

    def capped(scale_ratio: float) -> tuple[torch.Tensor, float]:
        cap = float(scale_ratio) * g_norm
        if float(solved_norm.item()) > float(cap.item()):
            ratio = float(cap.item() / solved_norm.clamp_min(1.0e-12).item())
            return solved * cap.to(device=solved.device, dtype=solved.dtype) / solved_norm.clamp_min(1.0e-12), ratio
        return solved, 1.0

    def measure(prop: torch.Tensor, clamp_ratio: float, cap_ratio: float) -> dict[str, Any]:
        with torch.no_grad():
            load_flat_params(model, base_params)
            b1_before = torch.nn.functional.cross_entropy(base1, yb1)
            b2_before = torch.nn.functional.cross_entropy(base2, yb2)
            b3_before = torch.nn.functional.cross_entropy(base3, yb3)
            _apply_direction_temporarily(model, base_params, prop, commit_scale)
            after1 = model(xb1).detach().float()
            after2 = model(xb2).detach().float()
            after3 = model(xb3).detach().float()
            b1_after = torch.nn.functional.cross_entropy(after1, yb1)
            b2_after = torch.nn.functional.cross_entropy(after2, yb2)
            b3_after = torch.nn.functional.cross_entropy(after3, yb3)
            actual = torch.cat([(after1 - base1).reshape(-1), (after2 - base2).reshape(-1)], dim=0)
            residual = target_flat.to(dtype=actual.dtype) - actual
            before_logits = torch.cat([base1, base2], dim=0).detach().float()
            after_logits = torch.cat([after1, after2], dim=0).detach().float()
            before_dist = torch.cdist(before_logits, before_logits).clamp_min(1.0e-8)
            after_dist = torch.cdist(after_logits, after_logits).clamp_min(1.0e-8)
            tri = torch.triu(torch.ones_like(before_dist, dtype=torch.bool), diagonal=1)
            ratios = (after_dist[tri] / before_dist[tri]).detach().float()
            if int(ratios.numel()):
                ratio_p05 = torch.quantile(ratios, 0.05).clamp_min(1.0e-8)
                ratio_p50 = torch.quantile(ratios, 0.50)
                ratio_p95 = torch.quantile(ratios, 0.95)
                local_distance_distortion = torch.mean(torch.abs(ratios - 1.0))
                fold_rate = torch.mean((ratios < 0.20).to(dtype=torch.float32))
                jacobian_condition_mean = ratio_p95 / ratio_p05
                jacobian_condition_p95 = torch.max(ratios) / ratio_p05
                curvature_norm = torch.mean(torch.abs(ratios - ratio_p50))
            else:
                local_distance_distortion = torch.tensor(0.0, device=device)
                fold_rate = torch.tensor(0.0, device=device)
                jacobian_condition_mean = torch.tensor(1.0, device=device)
                jacobian_condition_p95 = torch.tensor(1.0, device=device)
                curvature_norm = torch.tensor(0.0, device=device)
            if before_logits.shape[0] >= 2:
                before_nn = before_dist + torch.eye(before_dist.shape[0], device=device) * 1.0e9
                after_nn = after_dist + torch.eye(after_dist.shape[0], device=device) * 1.0e9
                neighbor_order_flip_rate = torch.mean(
                    (torch.argmin(before_nn, dim=1) != torch.argmin(after_nn, dim=1)).to(dtype=torch.float32)
                )
            else:
                neighbor_order_flip_rate = torch.tensor(0.0, device=device)
            centered = target_flat.to(dtype=actual.dtype) - target_flat.to(dtype=actual.dtype).mean()
            denom = torch.sum(centered.square()).clamp_min(1.0e-12)
            actuation_r2 = 1.0 - torch.sum(residual.square()) / denom
            cos_denom = torch.linalg.vector_norm(actual) * torch.linalg.vector_norm(target_flat.to(dtype=actual.dtype))
            actuation_cos = (actual @ target_flat.to(dtype=actual.dtype) / cos_denom.clamp_min(1.0e-12)).clamp(-1.0, 1.0)
            load_flat_params(model, base_params)
        b1_gain = float((b1_before - b1_after).item())
        b2_gain = float((b2_before - b2_after).item())
        b3_gain = float((b3_before - b3_after).item())
        score = (b1_gain + b2_gain) + 0.25 * b3_gain + max(0.0, float(actuation_r2.item()))
        if b3_gain < -1.0e-3:
            score -= 10.0 * abs(b3_gain)
        return {
            "proposed": prop,
            "ActuationR2": float(actuation_r2.item()),
            "ActuationCosine": float(actuation_cos.item()),
            "B1_gain": b1_gain,
            "B2_transfer_gain": b2_gain,
            "B3_safety_gain": b3_gain,
            "projection_residual_norm": float(torch.linalg.vector_norm(residual).item() / torch.linalg.vector_norm(target_flat).clamp_min(1.0e-12).item()),
            "function_displacement_norm": float(torch.linalg.vector_norm(actual).item()),
            "jacobian_condition_mean": float(jacobian_condition_mean.detach().item()),
            "jacobian_condition_p95": float(jacobian_condition_p95.detach().item()),
            "fold_rate": float(fold_rate.detach().item()),
            "neighbor_order_flip_rate": float(neighbor_order_flip_rate.detach().item()),
            "local_distance_distortion": float(local_distance_distortion.detach().item()),
            "smoothness_norm": float(torch.linalg.vector_norm(actual).item() / max(1, before_logits.shape[0]) ** 0.5),
            "curvature_norm": float(curvature_norm.detach().item()),
            "parameter_norm": float(torch.linalg.vector_norm(prop.detach()).item()),
            "operator_clamp_ratio": clamp_ratio,
            "operator_cap_ratio": float(cap_ratio),
            "score": score,
        }

    measured: list[dict[str, Any]] = []
    for cap_ratio in [float(norm_cap_ratio), 3000.0, 10000.0, 30000.0, 100000.0]:
        prop, clamp_ratio = capped(cap_ratio)
        measured.append(measure(prop, clamp_ratio, cap_ratio))
    best = max(measured, key=lambda item: float(item["score"]))
    proposed = best["proposed"]
    model.zero_grad(set_to_none=True)
    for (_, p), grad in zip(params, saved_grads):
        p.grad = None if grad is None else grad.to(device=p.device, dtype=p.dtype)
    diagnostics = {
        "ActuationR2": best["ActuationR2"],
        "ActuationCosine": best["ActuationCosine"],
        "B1_gain": best["B1_gain"],
        "B2_transfer_gain": best["B2_transfer_gain"],
        "B3_safety_gain": best["B3_safety_gain"],
        "projection_residual_norm": best["projection_residual_norm"],
        "function_displacement_norm": best["function_displacement_norm"],
        "jacobian_condition_mean": best["jacobian_condition_mean"],
        "jacobian_condition_p95": best["jacobian_condition_p95"],
        "fold_rate": best["fold_rate"],
        "neighbor_order_flip_rate": best["neighbor_order_flip_rate"],
        "local_distance_distortion": best["local_distance_distortion"],
        "smoothness_norm": best["smoothness_norm"],
        "curvature_norm": best["curvature_norm"],
        "parameter_norm": best["parameter_norm"],
        "operator_clamp_ratio": best["operator_clamp_ratio"],
        "operator_cap_ratio": best["operator_cap_ratio"],
        "operator_commit_scale": float(commit_scale),
        "operator_rank": int(n_readout),
        "operator_status": "exact_readout_measured",
        "target_kind": str(target_kind),
        "target_fit_scope": str(fit_scope),
        "target_rank_limit": int(rank_limit),
        "target_feature_select": str(feature_select),
        **target_diag,
    }
    return proposed, diagnostics


def _metric_target_readout_update(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    mechanism: str,
    *,
    metric_name: str,
    target_kind: str,
    fit_scope: str,
    seed: int,
    rank_limit: int = 0,
    feature_select: str = "top_norm",
) -> UpdateTensor:
    op_update, diagnostics = _exact_readout_function_space_actuation_update(
        model,
        x,
        y,
        device,
        norm_cap_ratio=1_000_000.0,
        target_kind=target_kind,
        fit_scope=fit_scope,
        rank_limit=rank_limit,
        feature_select=feature_select,
        seed=seed,
    )
    diagnostics["metric_name"] = metric_name
    diagnostics["metric_target_family"] = str(target_kind)
    diagnostics["metric_target_fit_scope"] = str(fit_scope)
    diagnostics["metric_target_source_channel"] = str(feature_select)
    try:
        diagnostics.update(output_metric_energies(model, x, op_update))
    except Exception as exc:  # diagnostics are audit-only
        diagnostics["metric_energy_error"] = type(exc).__name__
    return UpdateTensor(
        op_update,
        "metric_target_readout",
        "subtract",
        "function_metric_target",
        f"train_stream_metric_target_{metric_name}",
        mechanism,
        role="metric_first_train_only_function_space_target",
        one_step_descent_claim=0,
        diagnostics=diagnostics,
    )


def _gradient_observable_metric_target_update(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    mechanism: str,
    *,
    seed: int,
) -> UpdateTensor:
    base = _metric_target_readout_update(
        model,
        x,
        y,
        device,
        mechanism,
        metric_name="T11-GradientObservableMidDebtSource",
        target_kind="observable_mid_debt_source",
        fit_scope="b1b2_b3zero",
        rank_limit=16,
        feature_select="stable",
        seed=seed,
    )
    raw = base.tensor.detach()
    g_ref = flat_grad(model, device).detach().to(device=device, dtype=raw.dtype)
    support = raw.detach().abs() > 0.0
    g_support = torch.where(support, g_ref, torch.zeros_like(g_ref))
    g_norm = torch.linalg.vector_norm(g_support.detach()).clamp_min(1.0e-12)
    raw_norm = torch.linalg.vector_norm(raw.detach()).clamp_min(1.0e-12)
    dot = torch.dot(raw.float(), g_support.float()) if raw.numel() else torch.tensor(0.0, device=device)
    if float(dot.detach().item()) < 0.0:
        coeff = dot.to(device=device, dtype=raw.dtype) / g_norm.to(device=device, dtype=raw.dtype).square().clamp_min(1.0e-12)
        removed = coeff * g_support
        projected = torch.nan_to_num(raw - removed, nan=0.0, posinf=0.0, neginf=0.0)
        removed_flag = 1
    else:
        removed = torch.zeros_like(raw)
        projected = raw.clone()
        removed_flag = 0
    projected_dot = torch.dot(projected.float(), g_support.float()) if projected.numel() else torch.tensor(0.0, device=device)
    diagnostics = dict(base.diagnostics or {})
    diagnostics.update(
        {
            "metric_name": "T11-GradientObservableMidDebtSource",
            "metric_target_family": "gradient_observable_mid_debt_source",
            "metric_target_source_channel": "stable",
            "operator_status": "gradient_observable_halfspace_projected" if removed_flag else "gradient_observable_halfspace_already_safe",
            "target_gradient_halfspace_projected": removed_flag,
            "target_gradient_conflict_dot_before": float(dot.detach().item()),
            "target_gradient_conflict_dot_after": float(projected_dot.detach().item()),
            "target_gradient_conflict_cos_before": float((dot / (raw_norm * g_norm)).clamp(-1.0, 1.0).detach().item()),
            "target_gradient_conflict_cos_after": float(
                (projected_dot / (torch.linalg.vector_norm(projected.detach()).clamp_min(1.0e-12) * g_norm)).clamp(-1.0, 1.0).detach().item()
            ),
            "target_gradient_conflict_removed_norm": float(torch.linalg.vector_norm(removed.detach()).item()) if removed.numel() else 0.0,
            "target_gradient_conflict_removed_fraction": float(
                torch.linalg.vector_norm(removed.detach()).item() / raw_norm.detach().item()
            )
            if raw.numel()
            else 0.0,
            "source_observability_gate_accept": int(float(projected_dot.detach().item()) >= -1.0e-10),
            "source_observability_metric": "train_loss_cotangent_halfspace_on_metric_target_support",
        }
    )
    try:
        diagnostics.update(output_metric_energies(model, x, projected))
    except Exception as exc:  # diagnostics are audit-only
        diagnostics["metric_energy_error"] = type(exc).__name__
    return UpdateTensor(
        projected,
        "metric_target_readout_halfspace",
        "subtract",
        "function_metric_target",
        "train_stream_metric_target_T11-GradientObservableMidDebtSource",
        mechanism,
        role="metric_first_train_only_gradient_observable_target",
        one_step_descent_claim=0,
        diagnostics=diagnostics,
    )


def _random_same_rank_block_update(model: torch.nn.Module, device: torch.device, seed: int, rank: int = 4) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    gen = torch.Generator(device=device).manual_seed(int(seed) + 191_803)
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        grad = p.grad.detach() if p.grad is not None else torch.zeros_like(p)
        low = pname.lower()
        if grad.ndim >= 2 and ("w1" in low or "w2" in low or "readout" in low):
            mat = grad.float().reshape(int(grad.shape[0]), -1)
            r = min(int(rank), int(mat.shape[0]), int(mat.shape[1]))
            left = torch.randn((int(mat.shape[0]), r), device=device, generator=gen, dtype=torch.float32)
            right = torch.randn((r, int(mat.shape[1])), device=device, generator=gen, dtype=torch.float32)
            rand = left @ right
            rand = normalized_like(rand, mat)
            if "w1" in low and ("w2" not in low and "readout" not in low):
                rand = 0.50 * rand
            chunks.append(rand.reshape_as(grad).to(device=device, dtype=grad.dtype).reshape(-1))
        else:
            chunks.append(torch.zeros_like(grad).reshape(-1))
    if not chunks:
        return torch.zeros(0, device=device)
    return torch.cat(chunks)


def _low_rank_matrix_update(model: torch.nn.Module, device: torch.device, rank: int = 4) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for _name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        grad = p.grad.detach() if p.grad is not None else torch.zeros_like(p)
        if grad.ndim >= 2 and min(int(grad.shape[0]), int(grad.shape[1])) > 1:
            mat = grad.float().reshape(int(grad.shape[0]), -1)
            try:
                u, s, vh = torch.linalg.svd(mat, full_matrices=False)
                r = min(int(rank), int(s.numel()))
                approx = (u[:, :r] * s[:r]) @ vh[:r]
                chunks.append(approx.reshape_as(grad).to(device=device, dtype=grad.dtype).reshape(-1))
            except Exception:
                chunks.append(grad.reshape(-1))
        else:
            chunks.append(grad.reshape(-1))
    if not chunks:
        return torch.zeros(0, device=device)
    return torch.cat(chunks)


def poprisk_snr_update_with_diagnostics(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    max_examples: int = 16,
    *,
    exact_variance: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    grads: list[torch.Tensor] = []
    n = min(int(x.shape[0]), int(max_examples))
    for i in range(n):
        model.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x[i : i + 1]).float(), y[i : i + 1])
        loss.backward()
        grads.append(flat_grad(model, x.device).detach())
    if not grads:
        return flat_grad(model, x.device), {"PopRisk_micro_examples": 0, "PopRisk_exact_variance": int(exact_variance)}
    stack = torch.stack(grads, dim=0)
    mu = stack.mean(dim=0)
    var = stack.var(dim=0, unbiased=bool(exact_variance and n > 1))
    snr = mu.square() / (var / max(1, n - 1) + 1.0e-8)
    scale = torch.sqrt(snr.clamp(max=100.0))
    out = mu * scale
    diagnostics = {
        "PopRisk_micro_examples": n,
        "PopRisk_exact_variance": int(exact_variance),
        "PopRisk_mu_norm": float(torch.linalg.vector_norm(mu.detach()).item()),
        "PopRisk_var_mean": float(var.detach().float().mean().item()) if var.numel() else 0.0,
        "PopRisk_snr_mean": float(snr.detach().float().mean().item()) if snr.numel() else 0.0,
        "PopRisk_snr_max": float(snr.detach().float().max().item()) if snr.numel() else 0.0,
        "PopRisk_update_norm": float(torch.linalg.vector_norm(out.detach()).item()),
        "PopRisk_SNR_score": float((mu.detach().float().square() / (var.detach().float() + 1.0e-8)).mean().item()) if var.numel() else 0.0,
    }
    model.zero_grad(set_to_none=True)
    return out, diagnostics


def poprisk_snr_update(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, max_examples: int = 16) -> torch.Tensor:
    out, _diagnostics = poprisk_snr_update_with_diagnostics(model, x, y, max_examples=max_examples)
    return out


def make_update(
    model: torch.nn.Module,
    mechanism: str,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    seed: int,
    slow_state: torch.Tensor | None = None,
) -> UpdateTensor:
    device = x.device
    if mechanism == "CTRL-NoOpMatchedOverhead":
        g = flat_grad(model, device)
        return UpdateTensor(torch.zeros_like(g), "step", "subtract", "parameter", "train_stream_noop_control", mechanism)
    if mechanism == "CTRL-RandomMatchedNorm":
        g = flat_grad(model, device)
        gen = torch.Generator(device=device).manual_seed(int(seed) + 170_333)
        noise = torch.randn(g.shape, device=device, generator=gen)
        return UpdateTensor(normalized_like(noise, g), "step", "subtract", "parameter", "train_stream_matched_random_control", mechanism, one_step_descent_claim=0)
    if mechanism == "CTRL-RandomSameRankBlock":
        g = flat_grad(model, device)
        rand_block = _random_same_rank_block_update(model, device, seed=seed, rank=4)
        return UpdateTensor(normalized_like(rand_block, g), "step", "subtract", "matrix_block", "train_stream_same_rank_random_block_control", mechanism, role="carrier_block", one_step_descent_claim=0)
    if mechanism in {"CTRL-AdamW", "CTRL-SGD", "CTRL-RecoveryOnly"}:
        g = flat_grad(model, device)
        return UpdateTensor(g.clone(), "gradient", "subtract", "parameter", "train_stream_gradient_control", mechanism)

    if mechanism == "M8-PopRiskSNRFU":
        base_g = flat_grad(model, device)
        u, diagnostics = poprisk_snr_update_with_diagnostics(model, x, y)
        diagnostics["PopRisk_grad_cosine"] = cosine(u, base_g)
        return UpdateTensor(u, "gradient", "subtract", "basis_channel", "train_stream_per_example_poprisk_snr", mechanism, diagnostics=diagnostics)

    g = flat_grad(model, device)
    if mechanism == "M236-MetricTargetLossCotangentFU":
        return _metric_target_readout_update(
            model,
            x,
            y,
            device,
            mechanism,
            metric_name="T0-LossCotangentReadout",
            target_kind="loss",
            fit_scope="b1b2",
            seed=seed,
        )
    if mechanism == "M237-MetricTargetLowDegreeReadoutFU":
        return _metric_target_readout_update(
            model,
            x,
            y,
            device,
            mechanism,
            metric_name="T3-LowDegreeReadout",
            target_kind="loss",
            fit_scope="b1b2",
            rank_limit=4,
            feature_select="head",
            seed=seed,
        )
    if mechanism == "M238-MetricTargetB1TransferFU":
        return _metric_target_readout_update(
            model,
            x,
            y,
            device,
            mechanism,
            metric_name="T4-B1LossTransfer",
            target_kind="loss",
            fit_scope="b1",
            seed=seed,
        )
    if mechanism == "M239-MetricTargetSourceProjectedB3NullFU":
        return _metric_target_readout_update(
            model,
            x,
            y,
            device,
            mechanism,
            metric_name="T5-SourceProjectedB3Null",
            target_kind="source_projected_consensus",
            fit_scope="b1b2_b3zero",
            rank_limit=16,
            feature_select="stable",
            seed=seed,
        )
    if mechanism == "M245-MetricTargetSplitConsensusFU":
        return _metric_target_readout_update(
            model,
            x,
            y,
            device,
            mechanism,
            metric_name="T1-SplitConsensus",
            target_kind="split_consensus",
            fit_scope="b1b2",
            seed=seed,
        )
    if mechanism == "M246-MetricTargetPopRiskSNRFU":
        return _metric_target_readout_update(
            model,
            x,
            y,
            device,
            mechanism,
            metric_name="T2-PopRiskSNR",
            target_kind="poprisk_snr",
            fit_scope="b1b2",
            rank_limit=16,
            feature_select="stable",
            seed=seed,
        )
    if mechanism == "M247-MetricTargetDiffeomorphicNoFoldFU":
        return _metric_target_readout_update(
            model,
            x,
            y,
            device,
            mechanism,
            metric_name="T6-DiffeomorphicNoFold",
            target_kind="view_consistent_loss",
            fit_scope="b1_b3zero",
            rank_limit=8,
            feature_select="stable",
            seed=seed,
        )
    if mechanism == "M248-MetricTargetRandomMatchedFU":
        return _metric_target_readout_update(
            model,
            x,
            y,
            device,
            mechanism,
            metric_name="T7-RandomMatchedActuation",
            target_kind="random",
            fit_scope="b1b2",
            seed=seed,
        )
    if mechanism == "M249-MetricTargetSignFlippedFU":
        return _metric_target_readout_update(
            model,
            x,
            y,
            device,
            mechanism,
            metric_name="T8-SignFlippedTarget",
            target_kind="sign_flip",
            fit_scope="b1b2",
            seed=seed,
        )
    if mechanism == "M250-MetricTargetDebtCalibratedSourceFU":
        return _metric_target_readout_update(
            model,
            x,
            y,
            device,
            mechanism,
            metric_name="T9-DebtCalibratedSource",
            target_kind="debt_calibrated_source",
            fit_scope="b1b2_b3zero",
            rank_limit=16,
            feature_select="stable",
            seed=seed,
        )
    if mechanism == "M251-MetricTargetObservableMidDebtSourceFU":
        return _metric_target_readout_update(
            model,
            x,
            y,
            device,
            mechanism,
            metric_name="T10-ObservableMidDebtSource",
            target_kind="observable_mid_debt_source",
            fit_scope="b1b2_b3zero",
            rank_limit=16,
            feature_select="stable",
            seed=seed,
        )
    if mechanism == "M252-MetricTargetObservableMidDebtM181BridgeFU":
        return _metric_target_readout_update(
            model,
            x,
            y,
            device,
            mechanism,
            metric_name="T10-ObservableMidDebtSource+M181Bridge",
            target_kind="observable_mid_debt_source",
            fit_scope="b1b2_b3zero",
            rank_limit=16,
            feature_select="stable",
            seed=seed,
        )
    if mechanism == "M253-MetricTargetGradientObservableMidDebtFU":
        return _gradient_observable_metric_target_update(
            model,
            x,
            y,
            device,
            mechanism,
            seed=seed,
        )
    if mechanism == "M254-HC8PreH3200SourceChannelAntiWashoutFU":
        return _metric_target_readout_update(
            model,
            x,
            y,
            device,
            mechanism,
            metric_name="T12-PreH3200SourceChannelAntiWashout",
            target_kind="observable_mid_debt_source",
            fit_scope="b1b2_b3zero",
            rank_limit=16,
            feature_select="stable",
            seed=seed,
        )
    if mechanism == "M255-HC9RiskWeightedSplitConsensusSourceFU":
        return _metric_target_readout_update(
            model,
            x,
            y,
            device,
            mechanism,
            metric_name="T13-RiskWeightedSplitConsensusSource",
            target_kind="poprisk_snr",
            fit_scope="b1b2_b3zero",
            rank_limit=16,
            feature_select="stable",
            seed=seed,
        )
    if mechanism == "M256-HC10ViewConsistentSourceCarryFU":
        return _metric_target_readout_update(
            model,
            x,
            y,
            device,
            mechanism,
            metric_name="T14-ViewConsistentSourceCarry",
            target_kind="view_consistent_loss",
            fit_scope="b1b2_b3zero",
            rank_limit=16,
            feature_select="stable",
            seed=seed,
        )
    if mechanism == "M257-HC11ContinuousSourceCarryAntiWashoutFU":
        return _metric_target_readout_update(
            model,
            x,
            y,
            device,
            mechanism,
            metric_name="T15-ContinuousSourceCarryAntiWashout",
            target_kind="observable_mid_debt_source",
            fit_scope="b1b2_b3zero",
            rank_limit=16,
            feature_select="stable",
            seed=seed,
        )
    if mechanism == "M258-HC12NoiseOrthogonalSourceCarryFU":
        return _metric_target_readout_update(
            model,
            x,
            y,
            device,
            mechanism,
            metric_name="T16-NoiseOrthogonalSourceCarry",
            target_kind="noise_orthogonal_consensus",
            fit_scope="b1b2_b3zero",
            rank_limit=16,
            feature_select="stable",
            seed=seed,
        )
    if mechanism in HC2_PROJECTION_MECHANISMS:
        projection_lambda = {
            "M240-HC2H3200NoProjectionFU": 0.0,
            "M241-HC2H3200L2ProjectionFU": 1.0,
            "M242-HC2H3200HalfProjectionFU": 0.5,
            "M243-HC2H4000L2RecomputeProjectionFU": 1.0,
            "M244-HC2H4000HalfRecomputeProjectionFU": 0.5,
        }[mechanism]
        return UpdateTensor(
            g.clone(),
            "gradient",
            "subtract",
            "slow_state",
            "train_stream_hc2_projection_audit_fallback",
            mechanism,
            role="hc2_projection_audit_fallback",
            one_step_descent_claim=0,
            diagnostics={
                "operator_status": "hc2_make_update_audit_fallback",
                "hc2_projection_lambda": projection_lambda,
                "hc2_h3200_anchor_captured": 0,
            },
        )
    if mechanism in V2206_METRIC_SOLVER_MECHANISMS:
        cfg = V2206_SOLVER_CONFIGS[mechanism]
        return solve_metric_readout_update(
            model,
            x,
            y,
            target_family=str(cfg["target_family"]),
            metric_family=str(cfg["metric_family"]),
            mechanism=mechanism,
            hidden_residual_scale=float(cfg.get("hidden_residual_scale", 0.35)),
            hidden_function_fraction=float(cfg.get("hidden_function_fraction", 0.25)),
            block_role=str(cfg.get("block_role", "all")),
            seed=seed,
        )
    if mechanism in METRIC_FIRST_MECHANISM_TO_METRIC:
        metric_name = METRIC_FIRST_MECHANISM_TO_METRIC[mechanism]
        base = _momentum_like_precondition(g) if mechanism in METRIC_SOURCE_MEMORY_MECHANISMS else g
        projected, diagnostics = metric_project_vector(base, metric_name)
        if mechanism in METRIC_SOURCE_MEMORY_MECHANISMS and slow_state is not None and slow_state.numel() == projected.numel():
            projected = normalized_like(0.72 * slow_state.to(device=device, dtype=projected.dtype) + 0.28 * projected, g)
            diagnostics["metric_source_memory_used"] = 1
            diagnostics["metric_source_memory_beta"] = 0.72
        elif mechanism in METRIC_SOURCE_MEMORY_MECHANISMS:
            low_nds_seed, _ = metric_project_vector(base, "G6-LowNDS")
            projected = normalized_like(0.80 * projected + 0.20 * low_nds_seed, g)
            diagnostics["metric_source_memory_used"] = 0
            diagnostics["metric_source_memory_beta"] = 0.72
        try:
            diagnostics.update(output_metric_energies(model, x, projected))
        except Exception as exc:  # audit-only diagnostics must not block update construction
            diagnostics["metric_energy_error"] = type(exc).__name__
        return UpdateTensor(
            projected,
            "metric_projected_gradient",
            "subtract",
            "slow_state+function_metric" if mechanism in METRIC_SOURCE_MEMORY_MECHANISMS else "function_metric",
            f"train_stream_metric_source_memory_{metric_name}" if mechanism in METRIC_SOURCE_MEMORY_MECHANISMS else f"train_stream_metric_first_{metric_name}",
            mechanism,
            role="metric_first_source_memory_parameter_pullback_proxy" if mechanism in METRIC_SOURCE_MEMORY_MECHANISMS else "metric_first_parameter_pullback_proxy",
            one_step_descent_claim=0,
            diagnostics=diagnostics,
        )
    if mechanism == "M1-AdamWPrimaryFUResidual":
        return UpdateTensor(_adamw_like_precondition(g), "gradient", "subtract", "parameter", "train_stream_gradient_adamw_like_residual_fu", mechanism)
    if mechanism == "M2-SGDMomentumPrimaryFU":
        return UpdateTensor(_momentum_like_precondition(g), "gradient", "subtract", "parameter", "train_stream_gradient_sgd_momentum_fu", mechanism)
    if mechanism == "M87-AdamWBoundaryThenMomentumSourceFU":
        return UpdateTensor(
            _momentum_like_precondition(g),
            "gradient",
            "subtract",
            "parameter",
            "train_stream_adamw_boundary_then_momentum_source_fu",
            mechanism,
            role="phase_reset_momentum_source_after_adamw_boundary",
            one_step_descent_claim=0,
            diagnostics={
                "source_state_gate_accept": 1,
                "source_state_current_cos": cosine(_momentum_like_precondition(g), g),
                "source_state_balance_mean": 1.0,
                "source_state_consensus_density": 1.0,
            },
        )
    if mechanism == "M88-AdamWBoundaryThenDualTimescaleSourceFU":
        return UpdateTensor(
            _momentum_like_precondition(g),
            "step",
            "subtract",
            "slow_state",
            "train_stream_adamw_boundary_then_dual_timescale_source_fu",
            mechanism,
            role="phase_reset_dual_timescale_source_after_adamw_boundary",
            one_step_descent_claim=0,
            diagnostics={
                "source_state_gate_accept": 1,
                "source_state_current_cos": cosine(_momentum_like_precondition(g), g),
                "source_state_balance_mean": 1.0,
                "source_state_consensus_density": 1.0,
            },
        )
    if mechanism == "M89-AdamWBoundaryDualTimescaleSGDFloorFU":
        return UpdateTensor(
            _momentum_like_precondition(g),
            "step",
            "subtract",
            "slow_state",
            "train_stream_adamw_boundary_dual_timescale_sgd_floor_source_fu",
            mechanism,
            role="phase_reset_dual_timescale_source_with_sgd_floor",
            one_step_descent_claim=0,
            diagnostics={
                "source_state_gate_accept": 1,
                "source_state_current_cos": cosine(_momentum_like_precondition(g), g),
                "source_state_balance_mean": 1.0,
                "source_state_consensus_density": 1.0,
            },
        )
    if mechanism == "M90-AdamWBoundaryDualTimescaleLateSGDFloorFU":
        return UpdateTensor(
            _momentum_like_precondition(g),
            "step",
            "subtract",
            "slow_state",
            "train_stream_adamw_boundary_dual_timescale_late_sgd_floor_source_fu",
            mechanism,
            role="phase_reset_dual_timescale_source_with_late_sgd_floor",
            one_step_descent_claim=0,
            diagnostics={
                "source_state_gate_accept": 1,
                "source_state_current_cos": cosine(_momentum_like_precondition(g), g),
                "source_state_balance_mean": 1.0,
                "source_state_consensus_density": 1.0,
            },
        )
    if mechanism == "M91-AdamWBoundaryDualTimescaleTinyLateSGDFloorFU":
        return UpdateTensor(
            _momentum_like_precondition(g),
            "step",
            "subtract",
            "slow_state",
            "train_stream_adamw_boundary_dual_timescale_tiny_late_sgd_floor_source_fu",
            mechanism,
            role="phase_reset_dual_timescale_source_with_tiny_late_sgd_floor",
            one_step_descent_claim=0,
            diagnostics={
                "source_state_gate_accept": 1,
                "source_state_current_cos": cosine(_momentum_like_precondition(g), g),
                "source_state_balance_mean": 1.0,
                "source_state_consensus_density": 1.0,
            },
        )
    if mechanism == "M92-AdamWBoundaryDualTimescaleReinforcedTinyLateSGDFloorFU":
        return UpdateTensor(
            _momentum_like_precondition(g),
            "step",
            "subtract",
            "slow_state",
            "train_stream_adamw_boundary_dual_timescale_reinforced_tiny_late_sgd_floor_source_fu",
            mechanism,
            role="phase_reset_reinforced_dual_timescale_source_with_tiny_late_sgd_floor",
            one_step_descent_claim=0,
            diagnostics={
                "source_state_gate_accept": 1,
                "source_state_current_cos": cosine(_momentum_like_precondition(g), g),
                "source_state_balance_mean": 1.0,
                "source_state_consensus_density": 1.0,
            },
        )
    if mechanism in {
        "M93-TrainLossGatedDualTimescaleTinyLateFU",
        "M94-TrainLossGatedDualTimescaleHoldFallbackFU",
        "M95-TrainLossGatedDualTimescaleTinyLateFallbackFU",
        "M96-TrainLossGatedDualTimescaleBoostedTinyLateFallbackFU",
        "M97-TrainLossLateHoldRecoveryFU",
        "M98-TrainLossLateLookaheadFloorFU",
        "M99-TrainLossTerminalLookaheadFloorFU",
        "M100-TrainLossEarlyTerminalLookaheadFloorFU",
        "M106-TrainLossTerminalProjectedLookaheadFloorFU",
        "M107-TrainLossTerminalConsensusLookaheadFloorFU",
        "M108-TrainLossTerminalSelectorLookaheadFloorFU",
        "M111-TrainLossTerminalPositiveLookaheadFloorFU",
        "M112-TrainLossTerminalCheckpointReentryFU",
        "M113-TrainLossTerminalHardSplitSourceFU",
        "M114-TrainLossTerminalAdamWLookaheadFU",
        "M115-TrainLossTerminalOptimizerSelectorFU",
        "M119-TerminalSourceConservingRouteFU",
        "M120-TrainLossRiskProfileRouteFU",
        "M121-DebtAwareSourceGateFU",
        "M122-UngatedWarmTerminalSourceRouteFU",
        "M123-UngatedWarmRiskProfileRouteFU",
        "M124-UngatedWarmDebtRawBailoutFU",
        "M125-TrainLossH2400CheckpointHoldFU",
        "M126-TrainLossH2800CheckpointHoldFU",
        "M127-TrainLossH2400DebtBailoutFU",
        "M128-TrainLossTerminalRawThenSourceGuardFU",
    }:
        return UpdateTensor(
            _momentum_like_precondition(g),
            "step",
            "subtract",
            "slow_state",
            (
                "train_stream_loss_gated_dual_timescale_terminal_source_conserving_route_fu"
                if mechanism == "M119-TerminalSourceConservingRouteFU"
                else "train_stream_loss_gated_dual_timescale_train_loss_risk_profile_route_fu"
                if mechanism == "M120-TrainLossRiskProfileRouteFU"
                else "train_stream_loss_gated_dual_timescale_debt_aware_source_gate_fu"
                if mechanism == "M121-DebtAwareSourceGateFU"
                else "train_stream_ungated_warm_terminal_source_conserving_route_fu"
                if mechanism == "M122-UngatedWarmTerminalSourceRouteFU"
                else "train_stream_ungated_warm_risk_profile_terminal_route_fu"
                if mechanism == "M123-UngatedWarmRiskProfileRouteFU"
                else "train_stream_ungated_warm_debt_raw_bailout_fu"
                if mechanism == "M124-UngatedWarmDebtRawBailoutFU"
                else "train_stream_loss_gated_dual_timescale_h2400_checkpoint_hold_fu"
                if mechanism == "M125-TrainLossH2400CheckpointHoldFU"
                else "train_stream_loss_gated_dual_timescale_h2800_checkpoint_hold_fu"
                if mechanism == "M126-TrainLossH2800CheckpointHoldFU"
                else "train_stream_loss_gated_dual_timescale_h2400_debt_bailout_fu"
                if mechanism == "M127-TrainLossH2400DebtBailoutFU"
                else "train_stream_loss_gated_dual_timescale_terminal_raw_then_source_guard_fu"
                if mechanism == "M128-TrainLossTerminalRawThenSourceGuardFU"
                else
                "train_stream_loss_gated_dual_timescale_hold_fallback_source_fu"
                if mechanism == "M94-TrainLossGatedDualTimescaleHoldFallbackFU"
                else "train_stream_loss_gated_dual_timescale_early_terminal_lookahead_floor_source_fu"
                if mechanism == "M100-TrainLossEarlyTerminalLookaheadFloorFU"
                else "train_stream_loss_gated_dual_timescale_terminal_projected_lookahead_floor_source_fu"
                if mechanism == "M106-TrainLossTerminalProjectedLookaheadFloorFU"
                else "train_stream_loss_gated_dual_timescale_terminal_consensus_lookahead_floor_source_fu"
                if mechanism == "M107-TrainLossTerminalConsensusLookaheadFloorFU"
                else "train_stream_loss_gated_dual_timescale_terminal_selector_lookahead_floor_source_fu"
                if mechanism == "M108-TrainLossTerminalSelectorLookaheadFloorFU"
                else "train_stream_loss_gated_dual_timescale_terminal_positive_lookahead_floor_source_fu"
                if mechanism == "M111-TrainLossTerminalPositiveLookaheadFloorFU"
                else "train_stream_loss_gated_dual_timescale_terminal_checkpoint_reentry_source_fu"
                if mechanism == "M112-TrainLossTerminalCheckpointReentryFU"
                else "train_stream_loss_gated_dual_timescale_terminal_hard_split_source_fu"
                if mechanism == "M113-TrainLossTerminalHardSplitSourceFU"
                else "train_stream_loss_gated_dual_timescale_terminal_adamw_lookahead_source_fu"
                if mechanism == "M114-TrainLossTerminalAdamWLookaheadFU"
                else "train_stream_loss_gated_dual_timescale_terminal_optimizer_selector_source_fu"
                if mechanism == "M115-TrainLossTerminalOptimizerSelectorFU"
                else "train_stream_loss_gated_dual_timescale_terminal_lookahead_floor_source_fu"
                if mechanism == "M99-TrainLossTerminalLookaheadFloorFU"
                else "train_stream_loss_gated_dual_timescale_late_lookahead_floor_source_fu"
                if mechanism == "M98-TrainLossLateLookaheadFloorFU"
                else "train_stream_loss_gated_dual_timescale_late_hold_recovery_source_fu"
                if mechanism == "M97-TrainLossLateHoldRecoveryFU"
                else "train_stream_loss_gated_dual_timescale_boosted_tiny_late_fallback_source_fu"
                if mechanism == "M96-TrainLossGatedDualTimescaleBoostedTinyLateFallbackFU"
                else "train_stream_loss_gated_dual_timescale_tiny_late_fallback_source_fu"
                if mechanism == "M95-TrainLossGatedDualTimescaleTinyLateFallbackFU"
                else "train_stream_loss_gated_dual_timescale_tiny_late_source_fu"
            ),
            mechanism,
            role=(
                "phase_reset_dual_timescale_source_with_terminal_source_conserving_route"
                if mechanism == "M119-TerminalSourceConservingRouteFU"
                else "phase_reset_dual_timescale_source_with_train_loss_risk_profile_route"
                if mechanism == "M120-TrainLossRiskProfileRouteFU"
                else "phase_reset_dual_timescale_source_with_debt_aware_terminal_source_gate"
                if mechanism == "M121-DebtAwareSourceGateFU"
                else "ungated_h3200_source_warm_with_terminal_source_conserving_route"
                if mechanism == "M122-UngatedWarmTerminalSourceRouteFU"
                else "ungated_h3200_source_warm_with_train_loss_risk_profile_route"
                if mechanism == "M123-UngatedWarmRiskProfileRouteFU"
                else "ungated_h3200_source_warm_with_debt_aware_raw_terminal_bailout"
                if mechanism == "M124-UngatedWarmDebtRawBailoutFU"
                else "phase_reset_dual_timescale_source_with_h2400_terminal_hold"
                if mechanism == "M125-TrainLossH2400CheckpointHoldFU"
                else "phase_reset_dual_timescale_source_with_h2800_terminal_hold"
                if mechanism == "M126-TrainLossH2800CheckpointHoldFU"
                else "phase_reset_dual_timescale_source_with_h2400_debt_bailout"
                if mechanism == "M127-TrainLossH2400DebtBailoutFU"
                else "phase_reset_dual_timescale_source_with_terminal_raw_then_source_guard"
                if mechanism == "M128-TrainLossTerminalRawThenSourceGuardFU"
                else
                "phase_reset_dual_timescale_source_with_train_loss_gate_and_hold_fallback"
                if mechanism == "M94-TrainLossGatedDualTimescaleHoldFallbackFU"
                else "phase_reset_dual_timescale_source_with_train_loss_gate_and_early_terminal_lookahead_floor"
                if mechanism == "M100-TrainLossEarlyTerminalLookaheadFloorFU"
                else "phase_reset_dual_timescale_source_with_train_loss_gate_and_terminal_projected_lookahead_floor"
                if mechanism == "M106-TrainLossTerminalProjectedLookaheadFloorFU"
                else "phase_reset_dual_timescale_source_with_train_loss_gate_and_terminal_consensus_lookahead_floor"
                if mechanism == "M107-TrainLossTerminalConsensusLookaheadFloorFU"
                else "phase_reset_dual_timescale_source_with_train_loss_gate_and_terminal_selector_lookahead_floor"
                if mechanism == "M108-TrainLossTerminalSelectorLookaheadFloorFU"
                else "phase_reset_dual_timescale_source_with_train_loss_gate_and_terminal_positive_lookahead_floor"
                if mechanism == "M111-TrainLossTerminalPositiveLookaheadFloorFU"
                else "phase_reset_dual_timescale_source_with_train_loss_gate_and_terminal_checkpoint_reentry"
                if mechanism == "M112-TrainLossTerminalCheckpointReentryFU"
                else "phase_reset_dual_timescale_source_with_train_loss_gate_and_terminal_hard_split_source"
                if mechanism == "M113-TrainLossTerminalHardSplitSourceFU"
                else "phase_reset_dual_timescale_source_with_train_loss_gate_and_terminal_adamw_lookahead"
                if mechanism == "M114-TrainLossTerminalAdamWLookaheadFU"
                else "phase_reset_dual_timescale_source_with_train_loss_gate_and_terminal_optimizer_selector"
                if mechanism == "M115-TrainLossTerminalOptimizerSelectorFU"
                else "phase_reset_dual_timescale_source_with_train_loss_gate_and_terminal_lookahead_floor"
                if mechanism == "M99-TrainLossTerminalLookaheadFloorFU"
                else "phase_reset_dual_timescale_source_with_train_loss_gate_and_late_lookahead_floor"
                if mechanism == "M98-TrainLossLateLookaheadFloorFU"
                else "phase_reset_dual_timescale_source_with_train_loss_gate_and_late_hold_recovery"
                if mechanism == "M97-TrainLossLateHoldRecoveryFU"
                else "phase_reset_dual_timescale_source_with_train_loss_gate_and_boosted_tiny_late_fallback"
                if mechanism == "M96-TrainLossGatedDualTimescaleBoostedTinyLateFallbackFU"
                else "phase_reset_dual_timescale_source_with_train_loss_gate_and_tiny_late_fallback"
                if mechanism == "M95-TrainLossGatedDualTimescaleTinyLateFallbackFU"
                else "phase_reset_dual_timescale_source_with_train_loss_gate"
            ),
            one_step_descent_claim=0,
            diagnostics={
                "source_state_gate_accept": 1,
                "source_state_current_cos": cosine(_momentum_like_precondition(g), g),
                "source_state_balance_mean": 1.0,
                "source_state_consensus_density": 1.0,
            },
        )
    if mechanism in {
        "M101-AdamWBoundaryDualTimescaleAntiWashoutFU",
        "M102-AdamWBoundaryDualTimescaleSourceAnchorFU",
        "M103-AdamWBoundaryDualTimescaleParamEMAReentryFU",
        "M104-AdamWBoundaryDualTimescaleReadoutChannelFU",
        "M105-AdamWBoundaryDualTimescaleHiddenMatrixChannelFU",
        "M118-SourceConservingOptimizerOnlyFU",
    }:
        channel_update = _momentum_like_precondition(g)
        if mechanism == "M104-AdamWBoundaryDualTimescaleReadoutChannelFU":
            channel_update = normalized_like(g * _readout_role_mask(model, device), g)
        elif mechanism == "M105-AdamWBoundaryDualTimescaleHiddenMatrixChannelFU":
            channel_update = normalized_like(g * _hidden_matrix_role_mask(model, device), g)
        return UpdateTensor(
            channel_update,
            "step",
            "subtract",
            (
                "readout_carrier"
                if mechanism == "M104-AdamWBoundaryDualTimescaleReadoutChannelFU"
                else "matrix_block"
                if mechanism == "M105-AdamWBoundaryDualTimescaleHiddenMatrixChannelFU"
                else "slow_state"
            ),
            (
                "train_stream_adamw_boundary_dual_timescale_source_antiwashout"
                if mechanism == "M101-AdamWBoundaryDualTimescaleAntiWashoutFU"
                else "train_stream_adamw_boundary_dual_timescale_source_conserving_optimizer_only"
                if mechanism == "M118-SourceConservingOptimizerOnlyFU"
                else "train_stream_adamw_boundary_dual_timescale_param_ema_reentry"
                if mechanism == "M103-AdamWBoundaryDualTimescaleParamEMAReentryFU"
                else "train_stream_adamw_boundary_dual_timescale_readout_channel_source"
                if mechanism == "M104-AdamWBoundaryDualTimescaleReadoutChannelFU"
                else "train_stream_adamw_boundary_dual_timescale_hidden_matrix_channel_source"
                if mechanism == "M105-AdamWBoundaryDualTimescaleHiddenMatrixChannelFU"
                else "train_stream_adamw_boundary_dual_timescale_source_anchor"
            ),
            mechanism,
            role=(
                "phase_reset_dual_timescale_source_with_antiwashout_projection"
                if mechanism == "M101-AdamWBoundaryDualTimescaleAntiWashoutFU"
                else "phase_reset_dual_timescale_source_conserving_optimizer_no_fallback"
                if mechanism == "M118-SourceConservingOptimizerOnlyFU"
                else "phase_reset_dual_timescale_source_with_parameter_ema_reentry"
                if mechanism == "M103-AdamWBoundaryDualTimescaleParamEMAReentryFU"
                else "phase_reset_dual_timescale_source_with_readout_channel_reset"
                if mechanism == "M104-AdamWBoundaryDualTimescaleReadoutChannelFU"
                else "phase_reset_dual_timescale_source_with_hidden_matrix_channel_reset"
                if mechanism == "M105-AdamWBoundaryDualTimescaleHiddenMatrixChannelFU"
                else "phase_reset_dual_timescale_source_with_anchor_residual"
            ),
            one_step_descent_claim=0,
            diagnostics={
                "source_state_gate_accept": 1,
                "source_state_current_cos": cosine(_momentum_like_precondition(g), g),
                "source_state_balance_mean": 1.0,
                "source_state_consensus_density": 1.0,
            },
        )
    if mechanism == "M116-DatasetInvariantPopRiskSlowFU":
        n = int(x.shape[0])
        split = max(1, n // 2)
        u_a, diag_a = poprisk_snr_update_with_diagnostics(model, x[:split], y[:split], max_examples=16, exact_variance=True)
        xb_b = x[split:] if split < n else x[:split]
        yb_b = y[split:] if split < n else y[:split]
        u_b, diag_b = poprisk_snr_update_with_diagnostics(model, xb_b, yb_b, max_examples=16, exact_variance=True)
        agree = torch.sign(u_a) == torch.sign(u_b)
        density = float(agree.float().mean().item()) if agree.numel() else 0.0
        mag_a = u_a.detach().float().abs()
        mag_b = u_b.detach().float().abs()
        balance = torch.minimum(mag_a, mag_b) / torch.maximum(mag_a, mag_b).clamp_min(1.0e-8)
        balance_mean = float(balance[agree].mean().item()) if bool(agree.any().item()) else 0.0
        invariant = torch.where(agree, 0.50 * (u_a + u_b) * balance.to(device=device, dtype=u_a.dtype), torch.zeros_like(u_a))
        if slow_state is None or slow_state.numel() != invariant.numel():
            s = invariant.detach().clone()
        else:
            s = 0.97 * slow_state.to(device=device) + 0.03 * invariant
        diagnostics = {
            "PopRisk_micro_examples": int(diag_a.get("PopRisk_micro_examples", 0)) + int(diag_b.get("PopRisk_micro_examples", 0)),
            "PopRisk_exact_variance": 1,
            "PopRisk_snr_mean": 0.50 * float(diag_a.get("PopRisk_snr_mean", 0.0)) + 0.50 * float(diag_b.get("PopRisk_snr_mean", 0.0)),
            "PopRisk_update_norm": float(torch.linalg.vector_norm(invariant.detach()).item()),
            "PopRisk_grad_cosine": cosine(s, g),
            "PopRisk_slow_state_norm": float(torch.linalg.vector_norm(s.detach()).item()),
            "source_state_consensus_density": density,
            "source_state_balance_mean": balance_mean,
            "source_state_gate_accept": int(density >= 0.05 and balance_mean >= 0.05 and float(torch.linalg.vector_norm(invariant.detach()).item()) > 1.0e-12),
        }
        return UpdateTensor(
            normalized_like(s, g),
            "step",
            "subtract",
            "slow_state",
            "train_stream_dataset_invariant_split_poprisk_snr_slow_state",
            mechanism,
            role="dataset_invariant_train_only_poprisk_snr_slow_source",
            one_step_descent_claim=0,
            diagnostics=diagnostics,
        )
    if mechanism == "M117-DatasetInvariantReadoutConsensusFU":
        consensus, diagnostics = _split_readout_consensus_update(model, x, y, device, classes=10)
        if slow_state is None or slow_state.numel() != consensus.numel():
            s = consensus.detach().clone()
        else:
            s = 0.92 * slow_state.to(device=device) + 0.08 * consensus
        diagnostics["source_state_current_cos"] = cosine(s, g)
        diagnostics["source_state_norm"] = float(torch.linalg.vector_norm(s.detach()).item())
        return UpdateTensor(
            normalized_like(s, g),
            "step",
            "subtract",
            "readout_carrier",
            "train_stream_dataset_invariant_readout_split_consensus_corrupt_filtered",
            mechanism,
            role="dataset_invariant_train_only_readout_consensus_source",
            one_step_descent_claim=0,
            diagnostics=diagnostics,
        )
    if mechanism == "M3-FUPrimary":
        return UpdateTensor(_mean_centered(g), "step", "subtract", "function_projected_parameter", "train_stream_cotangent_backprop_no_adamw_step", mechanism)
    if mechanism == "M4-FUOnlyKeyParams":
        mask = _basis_role_mask(model, device)
        return UpdateTensor(g * mask, "step", "subtract", "basis_channel", "train_stream_basis_role_gradient", mechanism)
    if mechanism in {"M5-AlternatingFUGradient", "M15-LineCFilteredAlternatingFU"}:
        return UpdateTensor(normalized_like(g * _alternating_mask(g), g), "step", "subtract", "trajectory_boundary", "train_stream_boundary_pulse", mechanism, one_step_descent_claim=0)
    if mechanism == "M16-TwoPhaseMomentumThenLineCFU":
        return UpdateTensor(_momentum_like_precondition(g), "step", "subtract", "trajectory_boundary", "train_stream_two_phase_momentum_then_linec_fu", mechanism, one_step_descent_claim=0)
    if mechanism == "M17-ReadoutCarrierTwoPhaseLineCFU":
        mask = _readout_role_mask(model, device)
        readout_grad = g * mask
        return UpdateTensor(_momentum_like_precondition(readout_grad), "step", "subtract", "readout_carrier", "train_stream_readout_carrier_two_phase_linec_fu", mechanism, role="readout", one_step_descent_claim=0)
    if mechanism == "M36-MomentumWarmReadoutBlockFU":
        mask = _readout_role_mask(model, device)
        readout_grad = g * mask
        return UpdateTensor(_momentum_like_precondition(readout_grad), "step", "subtract", "readout_carrier", "train_stream_momentum_warmup_then_readout_block_fu", mechanism, role="readout", one_step_descent_claim=0)
    if mechanism == "M18-CarrierLowRankBlockFU":
        block = _carrier_low_rank_block_update(model, device, rank=4)
        return UpdateTensor(normalized_like(block, g), "step", "subtract", "matrix_block", "train_stream_carrier_low_rank_svd_r4", mechanism, role="carrier_block", one_step_descent_claim=0)
    if mechanism == "M19-SplitConsensusLowRankBlockFU":
        block = _split_consensus_low_rank_block_update(model, x, y, device, rank=4)
        return UpdateTensor(normalized_like(block, g), "step", "subtract", "matrix_block", "train_stream_split_consensus_low_rank_block_svd_r4", mechanism, role="carrier_block", one_step_descent_claim=0)
    if mechanism == "M20-TrainSplitFunctionSpaceActuationFU":
        op_update, diagnostics = _train_split_function_space_actuation_update(model, x, y, device, rank=4)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_split_function_space_actuation_ridge_fd", mechanism, role="function_space_operator", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M21-ExactReadoutFunctionSpaceActuationFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(model, x, y, device)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_exact_readout_function_space_actuation", mechanism, role="function_space_readout_operator", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M22-ExactReadoutHighCapScheduledFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(model, x, y, device, norm_cap_ratio=500_000.0)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_exact_readout_function_space_actuation_highcap", mechanism, role="function_space_readout_operator", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M23-ExactReadoutUltraCapScheduledFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(model, x, y, device, norm_cap_ratio=1_000_000.0)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_exact_readout_function_space_actuation_ultracap", mechanism, role="function_space_readout_operator", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M24-WarmupExactReadoutUltraCapFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(model, x, y, device, norm_cap_ratio=1_000_000.0)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_warmup_then_exact_readout_function_space_actuation_ultracap", mechanism, role="function_space_readout_operator", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M25-AdamWExactReadoutUltraCapFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(model, x, y, device, norm_cap_ratio=1_000_000.0)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_adamw_primary_exact_readout_function_space_actuation_ultracap", mechanism, role="adamw_plus_function_space_readout_operator", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M26-GatedAdamWExactReadoutFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(model, x, y, device, norm_cap_ratio=1_000_000.0)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_gated_adamw_exact_readout_function_space_actuation", mechanism, role="gated_adamw_plus_function_space_readout_operator", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M27-MultiBatchGeneralizationGatedReadoutFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(model, x, y, device, norm_cap_ratio=1_000_000.0)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_multibatch_gated_exact_readout_function_space_actuation", mechanism, role="multibatch_gated_function_space_readout_operator", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M28-AdamWMultiBatchGatedReadoutFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(model, x, y, device, norm_cap_ratio=1_000_000.0)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_adamw_multibatch_gated_exact_readout_function_space_actuation", mechanism, role="adamw_multibatch_gated_function_space_readout_operator", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M49-LossCotangentTargetFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(model, x, y, device, norm_cap_ratio=1_000_000.0, target_kind="loss", fit_scope="b1b2", seed=seed)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_loss_cotangent_target_exact_readout_actuation", mechanism, role="function_space_target_reset", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M50-RandomMatchedTargetFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(model, x, y, device, norm_cap_ratio=1_000_000.0, target_kind="random", fit_scope="b1b2", seed=seed)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_random_matched_target_exact_readout_control", mechanism, role="function_space_target_control", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M51-SignFlippedTargetFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(model, x, y, device, norm_cap_ratio=1_000_000.0, target_kind="sign_flip", fit_scope="b1b2", seed=seed)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_sign_flipped_target_exact_readout_control", mechanism, role="function_space_target_control", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M52-CorruptedLabelTargetFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(model, x, y, device, norm_cap_ratio=1_000_000.0, target_kind="corrupt", fit_scope="b1b2", seed=seed)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_corrupted_label_target_exact_readout_control", mechanism, role="function_space_target_control", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M53-LowRankLossCotangentTargetFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(model, x, y, device, norm_cap_ratio=1_000_000.0, target_kind="loss", fit_scope="b1b2", rank_limit=8, seed=seed)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_low_rank_loss_cotangent_target_exact_readout_actuation", mechanism, role="function_space_target_reset", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M54-CrossSplitConsensusTargetFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(model, x, y, device, norm_cap_ratio=1_000_000.0, target_kind="split_consensus", fit_scope="b1b2", seed=seed)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_cross_split_consensus_target_exact_readout_actuation", mechanism, role="function_space_target_reset", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M55-LowDegreeReadoutTargetFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(model, x, y, device, norm_cap_ratio=1_000_000.0, target_kind="loss", fit_scope="b1b2", rank_limit=4, feature_select="head", seed=seed)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_low_degree_readout_target_exact_readout_actuation", mechanism, role="function_space_target_reset", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M56-WeakStableTargetFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(model, x, y, device, norm_cap_ratio=1_000_000.0, target_kind="weak_stable", fit_scope="b1b2", rank_limit=16, feature_select="stable", seed=seed)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_weak_stable_target_exact_readout_actuation", mechanism, role="function_space_target_reset", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M57-B1ReadoutTransferTargetFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(model, x, y, device, norm_cap_ratio=1_000_000.0, target_kind="loss", fit_scope="b1", seed=seed)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_b1_fit_b2_transfer_readout_target_actuation", mechanism, role="function_space_target_reset", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M58-ReservoirExcludingConsensusTargetFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(model, x, y, device, norm_cap_ratio=1_000_000.0, target_kind="split_consensus", fit_scope="b1b2", rank_limit=16, feature_select="stable", seed=seed)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_reservoir_excluding_consensus_target_actuation", mechanism, role="function_space_target_reset", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M59-StableRandomTargetControlFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(model, x, y, device, norm_cap_ratio=1_000_000.0, target_kind="random", fit_scope="b1b2", rank_limit=16, feature_select="stable", seed=seed)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_stable_random_target_exact_readout_control", mechanism, role="function_space_target_control", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M134-SignalReservoirB3NullConsensusTargetFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(
            model,
            x,
            y,
            device,
            norm_cap_ratio=1_000_000.0,
            target_kind="split_consensus",
            fit_scope="b1b2_b3zero",
            rank_limit=16,
            feature_select="stable",
            seed=seed,
        )
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_signal_reservoir_b3_null_consensus_target", mechanism, role="function_space_target_reset", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M135-SourceBankB3NullConsensusTargetFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(
            model,
            x,
            y,
            device,
            norm_cap_ratio=1_000_000.0,
            target_kind="split_consensus",
            fit_scope="b1b2_b3zero",
            rank_limit=8,
            feature_select="source_bank",
            seed=seed,
        )
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_source_bank_b3_null_consensus_target", mechanism, role="function_space_target_reset", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism in {
        "M137-EarlyPulseSourceBankB3NullConsensusTargetFU",
        "M138-EarlyPulseAdamWSourceBankB3NullConsensusTargetFU",
    }:
        op_update, diagnostics = _exact_readout_function_space_actuation_update(
            model,
            x,
            y,
            device,
            norm_cap_ratio=1_000_000.0,
            target_kind="split_consensus",
            fit_scope="b1b2_b3zero",
            rank_limit=8,
            feature_select="source_bank",
            seed=seed,
        )
        source = (
            "train_stream_early_pulse_adamw_source_bank_b3_null_consensus_target"
            if mechanism == "M138-EarlyPulseAdamWSourceBankB3NullConsensusTargetFU"
            else "train_stream_early_pulse_source_bank_b3_null_consensus_target"
        )
        return UpdateTensor(op_update, "cotangent", "subtract", "function", source, mechanism, role="function_space_target_reset", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M136-StableRandomB3NullTargetControlFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(
            model,
            x,
            y,
            device,
            norm_cap_ratio=1_000_000.0,
            target_kind="random",
            fit_scope="b1b2_b3zero",
            rank_limit=16,
            feature_select="stable",
            seed=seed,
        )
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_stable_random_b3_null_target_control", mechanism, role="function_space_target_control", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M155-SourceProjectionB3NullConsensusTargetOnlyFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(
            model,
            x,
            y,
            device,
            norm_cap_ratio=1_000_000.0,
            target_kind="source_projected_consensus",
            fit_scope="b1b2_b3zero",
            rank_limit=16,
            feature_select="stable",
            seed=seed,
        )
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_source_projection_b3_null_consensus_target", mechanism, role="function_space_target_reset", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M156-NoiseOrthogonalB3NullConsensusTargetOnlyFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(
            model,
            x,
            y,
            device,
            norm_cap_ratio=1_000_000.0,
            target_kind="noise_orthogonal_consensus",
            fit_scope="b1b2_b3zero",
            rank_limit=16,
            feature_select="stable",
            seed=seed,
        )
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_noise_orthogonal_b3_null_consensus_target", mechanism, role="function_space_target_reset", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M157-EasyMarginB3NullConsensusTargetOnlyFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(
            model,
            x,
            y,
            device,
            norm_cap_ratio=1_000_000.0,
            target_kind="easy_margin_consensus",
            fit_scope="b1b2_b3zero",
            rank_limit=16,
            feature_select="stable",
            seed=seed,
        )
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_easy_margin_b3_null_consensus_target", mechanism, role="function_space_target_reset", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M176-LowNDSDiffeomorphicTargetOnlyFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(
            model,
            x,
            y,
            device,
            norm_cap_ratio=500_000.0,
            target_kind="weak_stable",
            fit_scope="b1b2_b3zero",
            rank_limit=8,
            feature_select="stable",
            seed=seed,
        )
        diagnostics["diffeomorphic_target_family"] = "T3-low-NDS-weak-stable-readout"
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_low_nds_diffeomorphic_readout_target", mechanism, role="function_space_target_reset", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M177-InfoVolumeDiffeomorphicTargetOnlyFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(
            model,
            x,
            y,
            device,
            norm_cap_ratio=500_000.0,
            target_kind="split_consensus",
            fit_scope="b1b2_b3zero",
            rank_limit=16,
            feature_select="stable",
            seed=seed,
        )
        diagnostics["diffeomorphic_target_family"] = "T4-info-volume-stable-consensus"
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_info_volume_diffeomorphic_consensus_target", mechanism, role="function_space_target_reset", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M178-LowRankReadoutTransportTargetOnlyFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(
            model,
            x,
            y,
            device,
            norm_cap_ratio=500_000.0,
            target_kind="view_consistent_loss",
            fit_scope="b1_b3zero",
            rank_limit=4,
            feature_select="stable",
            seed=seed,
        )
        diagnostics["diffeomorphic_target_family"] = "T6-low-rank-view-consistent-readout-transport"
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_low_rank_readout_transport_target", mechanism, role="function_space_target_reset", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M60-B1CrossSplitConsensusTransferFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(model, x, y, device, norm_cap_ratio=1_000_000.0, target_kind="split_consensus", fit_scope="b1", seed=seed)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_b1_cross_split_consensus_transfer_target", mechanism, role="function_space_target_reset", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M61-StableB1ConsensusTransferFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(model, x, y, device, norm_cap_ratio=1_000_000.0, target_kind="split_consensus", fit_scope="b1", rank_limit=16, feature_select="stable", seed=seed)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_stable_b1_consensus_transfer_target", mechanism, role="function_space_target_reset", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M62-B1WeakStableTransferFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(model, x, y, device, norm_cap_ratio=1_000_000.0, target_kind="weak_stable", fit_scope="b1", rank_limit=16, feature_select="stable", seed=seed)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_b1_weak_stable_transfer_target", mechanism, role="function_space_target_reset", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M63-LowDegreeB1ConsensusTransferFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(model, x, y, device, norm_cap_ratio=1_000_000.0, target_kind="split_consensus", fit_scope="b1", rank_limit=4, feature_select="head", seed=seed)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_low_degree_b1_consensus_transfer_target", mechanism, role="function_space_target_reset", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M64-ContrastiveLossRandomOrthogonalTargetFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(model, x, y, device, norm_cap_ratio=1_000_000.0, target_kind="loss_orthogonal_random", fit_scope="b1b2", seed=seed)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_loss_target_orthogonal_to_matched_random", mechanism, role="function_space_target_reset", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M65-B1ContrastiveLossRandomOrthogonalTransferFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(model, x, y, device, norm_cap_ratio=1_000_000.0, target_kind="loss_orthogonal_random", fit_scope="b1", seed=seed)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_b1_loss_target_orthogonal_to_matched_random_transfer", mechanism, role="function_space_target_reset", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M66-TopWrongMarginTargetFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(model, x, y, device, norm_cap_ratio=1_000_000.0, target_kind="top_wrong_margin", fit_scope="b1b2", seed=seed)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_top_wrong_margin_target", mechanism, role="function_space_target_reset", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M67-B1TopWrongMarginTransferFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(model, x, y, device, norm_cap_ratio=1_000_000.0, target_kind="top_wrong_margin", fit_scope="b1", seed=seed)
        return UpdateTensor(op_update, "cotangent", "subtract", "function", "train_stream_b1_top_wrong_margin_transfer_target", mechanism, role="function_space_target_reset", one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M69-RotatedCautiousMatrixSlowFU":
        block = _low_rank_matrix_update(model, device, rank=4)
        if slow_state is None or slow_state.numel() != block.numel():
            state = block.detach().clone()
        else:
            state = slow_state.to(device=device)
        agree = torch.sign(state) == torch.sign(block)
        cautious = torch.where(agree, 0.85 * state + 0.15 * block, 0.05 * state + 0.10 * block)
        diagnostics = {
            "source_state_consensus_density": float(agree.float().mean().item()) if agree.numel() else 0.0,
            "source_state_current_cos": cosine(state, g),
            "source_state_balance_mean": cosine(state, block),
            "source_state_projected_grad_norm": float(torch.linalg.vector_norm(block.detach()).item()),
            "source_state_whitened_norm": float(torch.linalg.vector_norm(cautious.detach()).item()),
        }
        return UpdateTensor(
            normalized_like(cautious, g),
            "step",
            "subtract",
            "matrix_block",
            "train_stream_rotated_low_rank_cautious_matrix_slow_state",
            mechanism,
            role="rotated_cautious_matrix_slow_state",
            one_step_descent_claim=0,
            diagnostics=diagnostics,
        )
    if mechanism == "M70-TrainLookaheadCautiousSourceFU":
        fast = _momentum_like_precondition(g)
        block = _low_rank_matrix_update(model, device, rank=4)
        if slow_state is None or slow_state.numel() != block.numel():
            state = block.detach().clone()
        else:
            state = slow_state.to(device=device)
        agree = torch.sign(state) == torch.sign(block)
        cautious = torch.where(agree, 0.85 * state + 0.15 * block, 0.05 * state + 0.10 * block)
        mixed = normalized_like(0.70 * fast + 0.30 * cautious, g)
        diagnostics = {
            "source_state_consensus_density": float(agree.float().mean().item()) if agree.numel() else 0.0,
            "source_state_current_cos": cosine(mixed, g),
            "source_state_balance_mean": cosine(state, block),
            "source_state_projected_grad_norm": float(torch.linalg.vector_norm(block.detach()).item()),
            "source_state_whitened_norm": float(torch.linalg.vector_norm(mixed.detach()).item()),
        }
        return UpdateTensor(
            mixed,
            "step",
            "subtract",
            "matrix_block",
            "train_stream_source_vs_gradient_lookahead_cautious_state",
            mechanism,
            role="train_lookahead_cautious_source_state",
            one_step_descent_claim=0,
            diagnostics=diagnostics,
        )
    if mechanism == "M71-TrainLookaheadB1ConsensusTransferFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(
            model,
            x,
            y,
            device,
            norm_cap_ratio=1_000_000.0,
            target_kind="split_consensus",
            fit_scope="b1",
            seed=seed,
        )
        return UpdateTensor(
            op_update,
            "cotangent",
            "subtract",
            "function",
            "train_stream_b1_consensus_target_vs_gradient_lookahead",
            mechanism,
            role="train_lookahead_b1_consensus_transfer_target",
            one_step_descent_claim=0,
            diagnostics=diagnostics,
        )
    if mechanism == "M72-LossWarmToB1ConsensusMigrationFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(
            model,
            x,
            y,
            device,
            norm_cap_ratio=1_000_000.0,
            target_kind="split_consensus",
            fit_scope="b1",
            seed=seed,
        )
        return UpdateTensor(
            op_update,
            "cotangent",
            "subtract",
            "function",
            "train_stream_loss_warm_to_b1_consensus_migration",
            mechanism,
            role="loss_warm_to_b1_consensus_migration_eval_target",
            one_step_descent_claim=0,
            diagnostics=diagnostics,
        )
    if mechanism == "M73-EasyB1ConsensusTransferFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(
            model,
            x,
            y,
            device,
            norm_cap_ratio=1_000_000.0,
            target_kind="easy_split_consensus",
            fit_scope="b1",
            seed=seed,
        )
        return UpdateTensor(
            op_update,
            "cotangent",
            "subtract",
            "function",
            "train_stream_easy_loss_b1_consensus_transfer_target",
            mechanism,
            role="easy_loss_b1_consensus_transfer_target",
            one_step_descent_claim=0,
            diagnostics=diagnostics,
        )
    if mechanism == "M74-LossWarmToEasyB1ConsensusMigrationFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(
            model,
            x,
            y,
            device,
            norm_cap_ratio=1_000_000.0,
            target_kind="easy_split_consensus",
            fit_scope="b1",
            seed=seed,
        )
        return UpdateTensor(
            op_update,
            "cotangent",
            "subtract",
            "function",
            "train_stream_loss_warm_to_easy_b1_consensus_migration",
            mechanism,
            role="loss_warm_to_easy_b1_consensus_migration_eval_target",
            one_step_descent_claim=0,
            diagnostics=diagnostics,
        )
    if mechanism == "M75-LossEasyB1ConsensusBlendFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(
            model,
            x,
            y,
            device,
            norm_cap_ratio=1_000_000.0,
            target_kind="loss_easy_consensus_blend",
            fit_scope="b1",
            seed=seed,
        )
        return UpdateTensor(
            op_update,
            "cotangent",
            "subtract",
            "function",
            "train_stream_loss_easy_b1_consensus_blend_target",
            mechanism,
            role="loss_easy_b1_consensus_blend_target",
            one_step_descent_claim=0,
            diagnostics=diagnostics,
        )
    if mechanism == "M76-LossWarmToLossEasyB1ConsensusBlendFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(
            model,
            x,
            y,
            device,
            norm_cap_ratio=1_000_000.0,
            target_kind="loss_easy_consensus_blend",
            fit_scope="b1",
            seed=seed,
        )
        return UpdateTensor(
            op_update,
            "cotangent",
            "subtract",
            "function",
            "train_stream_loss_warm_to_loss_easy_b1_consensus_blend",
            mechanism,
            role="loss_warm_to_loss_easy_b1_consensus_blend_eval_target",
            one_step_descent_claim=0,
            diagnostics=diagnostics,
        )
    if mechanism == "M77-GainGatedLossWarmB1ConsensusMigrationFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(
            model,
            x,
            y,
            device,
            norm_cap_ratio=1_000_000.0,
            target_kind="split_consensus",
            fit_scope="b1",
            seed=seed,
        )
        return UpdateTensor(
            op_update,
            "cotangent",
            "subtract",
            "function",
            "train_stream_gain_gated_loss_warm_to_b1_consensus_migration",
            mechanism,
            role="gain_gated_loss_warm_to_b1_consensus_migration_eval_target",
            one_step_descent_claim=0,
            diagnostics=diagnostics,
        )
    if mechanism == "M78-GainGatedLossWarmBlendMigrationFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(
            model,
            x,
            y,
            device,
            norm_cap_ratio=1_000_000.0,
            target_kind="loss_easy_consensus_blend",
            fit_scope="b1",
            seed=seed,
        )
        return UpdateTensor(
            op_update,
            "cotangent",
            "subtract",
            "function",
            "train_stream_gain_gated_loss_warm_to_loss_easy_b1_consensus_blend",
            mechanism,
            role="gain_gated_loss_warm_to_loss_easy_b1_consensus_blend_eval_target",
            one_step_descent_claim=0,
            diagnostics=diagnostics,
        )
    if mechanism == "M79-B1ConsensusB3NullTransferFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(
            model,
            x,
            y,
            device,
            norm_cap_ratio=1_000_000.0,
            target_kind="split_consensus",
            fit_scope="b1_b3zero",
            seed=seed,
        )
        return UpdateTensor(
            op_update,
            "cotangent",
            "subtract",
            "function",
            "train_stream_b1_consensus_with_b3_null_transfer_target",
            mechanism,
            role="b1_consensus_b3_null_transfer_target",
            one_step_descent_claim=0,
            diagnostics=diagnostics,
        )
    if mechanism == "M80-LossWarmToB1ConsensusB3NullMigrationFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(
            model,
            x,
            y,
            device,
            norm_cap_ratio=1_000_000.0,
            target_kind="split_consensus",
            fit_scope="b1_b3zero",
            seed=seed,
        )
        return UpdateTensor(
            op_update,
            "cotangent",
            "subtract",
            "function",
            "train_stream_loss_warm_to_b1_consensus_b3_null_migration",
            mechanism,
            role="loss_warm_to_b1_consensus_b3_null_migration_eval_target",
            one_step_descent_claim=0,
            diagnostics=diagnostics,
        )
    if mechanism == "M81-ViewConsistentLossTargetFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(
            model,
            x,
            y,
            device,
            norm_cap_ratio=1_000_000.0,
            target_kind="view_consistent_loss",
            fit_scope="b1_b3zero",
            seed=seed,
        )
        return UpdateTensor(
            op_update,
            "cotangent",
            "subtract",
            "function",
            "train_stream_view_consistent_loss_target_with_b3_null",
            mechanism,
            role="view_consistent_loss_target_b3_null_transfer",
            one_step_descent_claim=0,
            diagnostics=diagnostics,
        )
    if mechanism == "M82-LossWarmToViewConsistentLossMigrationFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(
            model,
            x,
            y,
            device,
            norm_cap_ratio=1_000_000.0,
            target_kind="view_consistent_loss",
            fit_scope="b1_b3zero",
            seed=seed,
        )
        return UpdateTensor(
            op_update,
            "cotangent",
            "subtract",
            "function",
            "train_stream_loss_warm_to_view_consistent_loss_migration",
            mechanism,
            role="loss_warm_to_view_consistent_loss_migration_eval_target",
            one_step_descent_claim=0,
            diagnostics=diagnostics,
        )
    if mechanism == "M83-LowBankLossB3NullFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(
            model,
            x,
            y,
            device,
            norm_cap_ratio=1_000_000.0,
            target_kind="loss",
            fit_scope="b1_b3zero",
            rank_limit=1,
            feature_select="low_bank",
            seed=seed,
        )
        return UpdateTensor(
            op_update,
            "cotangent",
            "subtract",
            "function",
            "train_stream_low_bank_loss_b3_null_source_channel",
            mechanism,
            role="low_bank_loss_b3_null_source_channel_transfer",
            one_step_descent_claim=0,
            diagnostics=diagnostics,
        )
    if mechanism == "M84-LossWarmToLowBankLossB3NullMigrationFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(
            model,
            x,
            y,
            device,
            norm_cap_ratio=1_000_000.0,
            target_kind="loss",
            fit_scope="b1_b3zero",
            rank_limit=1,
            feature_select="low_bank",
            seed=seed,
        )
        return UpdateTensor(
            op_update,
            "cotangent",
            "subtract",
            "function",
            "train_stream_loss_warm_to_low_bank_loss_b3_null_migration",
            mechanism,
            role="loss_warm_to_low_bank_loss_b3_null_migration_eval_target",
            one_step_descent_claim=0,
            diagnostics=diagnostics,
        )
    if mechanism == "M85-GainGatedLowBankLossB3NullFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(
            model,
            x,
            y,
            device,
            norm_cap_ratio=1_000_000.0,
            target_kind="loss",
            fit_scope="b1_b3zero",
            rank_limit=1,
            feature_select="low_bank",
            seed=seed,
        )
        return UpdateTensor(
            op_update,
            "cotangent",
            "subtract",
            "function",
            "train_stream_gain_gated_low_bank_loss_b3_null_source_channel",
            mechanism,
            role="gain_gated_low_bank_loss_b3_null_source_channel_transfer",
            one_step_descent_claim=0,
            diagnostics=diagnostics,
        )
    if mechanism == "M86-LossWarmToGainGatedLowBankLossB3NullMigrationFU":
        op_update, diagnostics = _exact_readout_function_space_actuation_update(
            model,
            x,
            y,
            device,
            norm_cap_ratio=1_000_000.0,
            target_kind="loss",
            fit_scope="b1_b3zero",
            rank_limit=1,
            feature_select="low_bank",
            seed=seed,
        )
        return UpdateTensor(
            op_update,
            "cotangent",
            "subtract",
            "function",
            "train_stream_loss_warm_to_gain_gated_low_bank_loss_b3_null_migration",
            mechanism,
            role="loss_warm_to_gain_gated_low_bank_loss_b3_null_migration_eval_target",
            one_step_descent_claim=0,
            diagnostics=diagnostics,
        )
    if mechanism in {
        "M109-AdamWBoundaryToGainGatedLowBankB3NullFU",
        "M110-AdamWBoundaryLowBankAnchorAntiWashoutFU",
    }:
        op_update, diagnostics = _exact_readout_function_space_actuation_update(
            model,
            x,
            y,
            device,
            norm_cap_ratio=1_000_000.0,
            target_kind="loss",
            fit_scope="b1_b3zero",
            rank_limit=1,
            feature_select="low_bank",
            seed=seed,
        )
        return UpdateTensor(
            op_update,
            "cotangent",
            "subtract",
            "function",
            (
                "train_stream_adamw_boundary_low_bank_anchor_antiwashout_source_channel"
                if mechanism == "M110-AdamWBoundaryLowBankAnchorAntiWashoutFU"
                else "train_stream_adamw_boundary_to_gain_gated_low_bank_loss_b3_null_source_channel"
            ),
            mechanism,
            role=(
                "adamw_boundary_low_bank_anchor_antiwashout_source_channel"
                if mechanism == "M110-AdamWBoundaryLowBankAnchorAntiWashoutFU"
                else "adamw_boundary_to_gain_gated_low_bank_loss_b3_null_source_channel"
            ),
            one_step_descent_claim=0,
            diagnostics=diagnostics,
        )
    if mechanism in {"M29-ConsensusSourceStateFU", "M30-LowThresholdConsensusSourceStateFU", "M31-AdamWLowThresholdConsensusResidualFU", "M32-PostAdamWConsensusResidualFU", "M37-SplitFisherAgreementSlowFU", "M38-AdamWSplitFisherAgreementResidualFU", "M39-MomentumWarmSplitFisherFU", "M40-MomentumWarmAntiWashoutFU", "M41-MomentumWarmSourceAnchorFU", "M42-MomentumWarmHoldFU", "M43-MomentumCycleHoldFU", "M44-MomentumLineCAnchorSlowFU", "M45-MomentumSlowAnchorFU", "M46-MomentumMatrixBlockRetentionFU", "M47-MomentumCycleThenLineCAnchorFU", "M48-DualTimescaleSourceRetentionFU", "M68-MarginSplitConsensusSlowFU", "M152-SourceProjectionB3NullConsensusTargetFU", "M153-NoiseOrthogonalB3NullConsensusTargetFU", "M154-EasyMarginB3NullConsensusTargetFU", "M158-EarlySourceSlowEMATerminalRejectSourceAxisRescueFU", "M159-EarlySourceSlowEMATerminalRejectSlowEMARescueFU", "M160-EarlySourceSlowEMATerminalRejectHoldSourceRescueFU", "M161-EarlySourceSlowEMAShapePreserveFU", "M162-EarlySourceSlowEMAShapePreserveRawGuardFU", "M163-EarlySourceSlowEMAShapePreserveClampFU", "M164-EarlySourceSlowEMATerminalRejectRawRescueFU", "M165-EarlySourceSlowEMATerminalRejectHybridRescueFU", "M166-EarlySourceSlowEMATerminalRejectLateOnlyRawRescueFU", "M167-EarlySourceSlowEMATerminalTopKSupportGuardFU", "M168-EarlySourceSlowEMATerminalTopKRawGuardFU", "M169-EarlySourceSlowEMATerminalTopKDebtCapFU", "M170-EarlySourceSlowEMATerminalSourcePreserveStrongFU", "M171-EarlySourceSlowEMATerminalSourcePreserveGentleFU", "M172-EarlySourceSlowEMATerminalInfoVolumeGuardFU", "M173-EarlySourceSlowEMALowNDSDiffeomorphicTargetFU", "M174-EarlySourceSlowEMAInfoVolumeDiffeomorphicTargetFU", "M175-EarlySourceSlowEMALowRankReadoutTransportFU", "M179-EarlySourceSlowEMATerminalSourcePreserveVeryStrongFU", "M180-EarlySourceSlowEMAH3600TerminalSourcePreserveFU", "M181-EarlySourceSlowEMATerminalNoraOrthogonalSourceFU", "M182-EarlySourceSlowEMATerminalDebtAwarePreserveFU", "M183-EarlySourceSlowEMATerminalLowNDSMatrixBlockFU", "M184-EarlySourceSlowEMATerminalDualMemoryPreserveFU", "M185-EarlySourceSlowEMASNRTerminalPredictorFU", "M186-EarlySourceSlowEMASplitConsensusEstimatorFU", "M187-EarlySourceSlowEMASignalReservoirTransportFU", "M188-EarlySourceSlowEMATerminalSourceFloorFU", "M189-EarlySourceSlowEMATerminalH4000AnchorFloorFU", "M190-EarlySourceSlowEMATerminalDecayAwareFloorFU", "M191-EarlySourceSlowEMATerminalRawGuardSourceFloorFU", "M192-EarlySourceSlowEMATerminalRawGuardH4000AnchorFloorFU", "M193-EarlySourceSlowEMATerminalRawGuardDecayAwareFloorFU", "M194-EarlySourceSlowEMATerminalAntiErosionOrthogonalFU", "M195-EarlySourceSlowEMATerminalSourceReflectionGuardFU", "M196-EarlySourceSlowEMATerminalH4000TransportCorrectorFU", "M197-EarlySourceSlowEMATerminalH3200AnchorTransportFU", "M198-EarlySourceSlowEMATerminalRawGuardH3200AnchorTransportFU", "M199-EarlySourceSlowEMATerminalH3200RatioReentryFU", "M200-EarlySourceSlowEMATerminalH3200ProgressCarryFU", "M201-EarlySourceSlowEMATerminalH4000ProgressCarryFU", "M202-EarlySourceSlowEMATerminalH3200SourceProgressBlendFU", "M203-EarlySourceSlowEMATerminalRawGuardH3600GentleProgressFU", "M204-EarlySourceSlowEMATerminalRawGuardH4000GentleProgressFU", "M205-EarlySourceSlowEMATerminalRawGuardH4400ProjectedProgressFU", "M206-EarlySourceSlowEMATerminalControlRelativeSGDCatchupFU", "M207-EarlySourceSlowEMATerminalControlRelativeAdamWCatchupFU", "M208-EarlySourceSlowEMATerminalControlRelativeSourceBalancedCatchupFU", "M209-EarlySourceSlowEMATerminalTrajectoryAdaptivePreserveFU", "M210-EarlySourceSlowEMATerminalMidErosionBridgeFU", "M211-EarlySourceSlowEMATerminalTwoPhaseRatioRepairFU", "M212-EarlySourceSlowEMATerminalAntiSourceClipFU", "M213-EarlySourceSlowEMATerminalDebtAwareHoldFU", "M214-EarlySourceSlowEMATerminalAnchorFlowTinyFU", "M215-EarlySourceSlowEMATerminalAcceptMemoryFU", "M216-EarlySourceSlowEMATerminalAcceptMemoryDebtFU", "M217-EarlySourceSlowEMATerminalSourceProgressMemoryFU", "M218-EarlySourceSlowEMATerminalProjectedOptimizerLambda050FU", "M219-EarlySourceSlowEMATerminalProjectedOptimizerLambda100FU"}:
        source = "train_stream_consensus_source_state_first_step"
        if mechanism == "M30-LowThresholdConsensusSourceStateFU":
            source = "train_stream_low_threshold_consensus_source_state_first_step"
        if mechanism == "M31-AdamWLowThresholdConsensusResidualFU":
            source = "train_stream_adamw_low_threshold_consensus_residual_first_step"
        if mechanism == "M32-PostAdamWConsensusResidualFU":
            source = "train_stream_post_adamw_low_threshold_consensus_residual_first_step"
        if mechanism == "M37-SplitFisherAgreementSlowFU":
            source = "train_stream_split_fisher_agreement_first_step"
        if mechanism == "M38-AdamWSplitFisherAgreementResidualFU":
            source = "train_stream_adamw_split_fisher_agreement_residual_first_step"
        if mechanism == "M39-MomentumWarmSplitFisherFU":
            source = "train_stream_momentum_warm_split_fisher_first_step"
        if mechanism == "M40-MomentumWarmAntiWashoutFU":
            source = "train_stream_momentum_warm_antiwashout_first_step"
        if mechanism == "M41-MomentumWarmSourceAnchorFU":
            source = "train_stream_momentum_warm_source_anchor_first_step"
        if mechanism == "M42-MomentumWarmHoldFU":
            source = "train_stream_momentum_warm_hold_first_step"
        if mechanism == "M43-MomentumCycleHoldFU":
            source = "train_stream_momentum_cycle_hold_first_step"
        if mechanism == "M44-MomentumLineCAnchorSlowFU":
            source = "train_stream_momentum_linec_anchor_slow_first_step"
        if mechanism == "M45-MomentumSlowAnchorFU":
            source = "train_stream_momentum_slow_anchor_first_step"
        if mechanism == "M46-MomentumMatrixBlockRetentionFU":
            source = "train_stream_momentum_matrix_block_retention_first_step"
        if mechanism == "M47-MomentumCycleThenLineCAnchorFU":
            source = "train_stream_momentum_cycle_then_linec_anchor_first_step"
        if mechanism == "M48-DualTimescaleSourceRetentionFU":
            source = "train_stream_dual_timescale_source_retention_first_step"
        if mechanism == "M68-MarginSplitConsensusSlowFU":
            source = "train_stream_top_wrong_margin_split_consensus_first_step"
        if mechanism == "M152-SourceProjectionB3NullConsensusTargetFU":
            source = "train_stream_source_projection_terminal_target_wrapper_slow_state"
        if mechanism == "M153-NoiseOrthogonalB3NullConsensusTargetFU":
            source = "train_stream_noise_orthogonal_terminal_target_wrapper_slow_state"
        if mechanism == "M154-EasyMarginB3NullConsensusTargetFU":
            source = "train_stream_easy_margin_terminal_target_wrapper_slow_state"
        if mechanism == "M158-EarlySourceSlowEMATerminalRejectSourceAxisRescueFU":
            source = "train_stream_terminal_reject_source_axis_rescue_wrapper_slow_state"
        if mechanism == "M159-EarlySourceSlowEMATerminalRejectSlowEMARescueFU":
            source = "train_stream_terminal_reject_slow_ema_rescue_wrapper_slow_state"
        if mechanism == "M160-EarlySourceSlowEMATerminalRejectHoldSourceRescueFU":
            source = "train_stream_terminal_reject_hold_source_rescue_wrapper_slow_state"
        if mechanism == "M161-EarlySourceSlowEMAShapePreserveFU":
            source = "train_stream_h800_source_shape_preserve_wrapper_slow_state"
        if mechanism == "M162-EarlySourceSlowEMAShapePreserveRawGuardFU":
            source = "train_stream_h800_source_shape_preserve_raw_guard_wrapper_slow_state"
        if mechanism == "M163-EarlySourceSlowEMAShapePreserveClampFU":
            source = "train_stream_h800_source_shape_preserve_clamp_wrapper_slow_state"
        if mechanism == "M164-EarlySourceSlowEMATerminalRejectRawRescueFU":
            source = "train_stream_terminal_reject_raw_rescue_wrapper_slow_state"
        if mechanism == "M165-EarlySourceSlowEMATerminalRejectHybridRescueFU":
            source = "train_stream_terminal_reject_hybrid_rescue_wrapper_slow_state"
        if mechanism == "M166-EarlySourceSlowEMATerminalRejectLateOnlyRawRescueFU":
            source = "train_stream_terminal_reject_lateonly_raw_rescue_wrapper_slow_state"
        if mechanism == "M167-EarlySourceSlowEMATerminalTopKSupportGuardFU":
            source = "train_stream_terminal_topk_support_guard_wrapper_slow_state"
        if mechanism == "M168-EarlySourceSlowEMATerminalTopKRawGuardFU":
            source = "train_stream_terminal_topk_raw_guard_wrapper_slow_state"
        if mechanism == "M169-EarlySourceSlowEMATerminalTopKDebtCapFU":
            source = "train_stream_terminal_topk_debt_cap_wrapper_slow_state"
        if mechanism == "M170-EarlySourceSlowEMATerminalSourcePreserveStrongFU":
            source = "train_stream_terminal_source_preserve_strong_wrapper_slow_state"
        if mechanism == "M171-EarlySourceSlowEMATerminalSourcePreserveGentleFU":
            source = "train_stream_terminal_source_preserve_gentle_wrapper_slow_state"
        if mechanism == "M172-EarlySourceSlowEMATerminalInfoVolumeGuardFU":
            source = "train_stream_terminal_info_volume_guard_wrapper_slow_state"
        if mechanism == "M173-EarlySourceSlowEMALowNDSDiffeomorphicTargetFU":
            source = "train_stream_low_nds_diffeomorphic_target_wrapper_slow_state"
        if mechanism == "M174-EarlySourceSlowEMAInfoVolumeDiffeomorphicTargetFU":
            source = "train_stream_info_volume_diffeomorphic_target_wrapper_slow_state"
        if mechanism == "M175-EarlySourceSlowEMALowRankReadoutTransportFU":
            source = "train_stream_low_rank_readout_transport_target_wrapper_slow_state"
        if mechanism == "M179-EarlySourceSlowEMATerminalSourcePreserveVeryStrongFU":
            source = "train_stream_terminal_source_preserve_very_strong_wrapper_slow_state"
        if mechanism == "M180-EarlySourceSlowEMAH3600TerminalSourcePreserveFU":
            source = "train_stream_h3600_terminal_source_preserve_wrapper_slow_state"
        if mechanism == "M181-EarlySourceSlowEMATerminalNoraOrthogonalSourceFU":
            source = "train_stream_terminal_nora_orthogonal_source_wrapper_slow_state"
        if mechanism == "M218-EarlySourceSlowEMATerminalProjectedOptimizerLambda050FU":
            source = "train_stream_terminal_projected_optimizer_lambda050_wrapper_slow_state"
        if mechanism == "M219-EarlySourceSlowEMATerminalProjectedOptimizerLambda100FU":
            source = "train_stream_terminal_projected_optimizer_lambda100_wrapper_slow_state"
        if mechanism == "M182-EarlySourceSlowEMATerminalDebtAwarePreserveFU":
            source = "train_stream_terminal_debt_aware_preserve_wrapper_slow_state"
        if mechanism == "M183-EarlySourceSlowEMATerminalLowNDSMatrixBlockFU":
            source = "train_stream_terminal_low_nds_matrix_block_wrapper_slow_state"
        if mechanism == "M184-EarlySourceSlowEMATerminalDualMemoryPreserveFU":
            source = "train_stream_terminal_dual_memory_preserve_wrapper_slow_state"
        if mechanism == "M185-EarlySourceSlowEMASNRTerminalPredictorFU":
            source = "train_stream_terminal_snr_predictor_wrapper_slow_state"
        if mechanism == "M186-EarlySourceSlowEMASplitConsensusEstimatorFU":
            source = "train_stream_terminal_split_consensus_estimator_wrapper_slow_state"
        if mechanism == "M187-EarlySourceSlowEMASignalReservoirTransportFU":
            source = "train_stream_terminal_signal_reservoir_transport_wrapper_slow_state"
        if mechanism == "M188-EarlySourceSlowEMATerminalSourceFloorFU":
            source = "train_stream_terminal_source_floor_wrapper_slow_state"
        if mechanism == "M189-EarlySourceSlowEMATerminalH4000AnchorFloorFU":
            source = "train_stream_terminal_h4000_anchor_floor_wrapper_slow_state"
        if mechanism == "M190-EarlySourceSlowEMATerminalDecayAwareFloorFU":
            source = "train_stream_terminal_decay_aware_floor_wrapper_slow_state"
        if mechanism == "M191-EarlySourceSlowEMATerminalRawGuardSourceFloorFU":
            source = "train_stream_terminal_raw_guard_source_floor_wrapper_slow_state"
        if mechanism == "M192-EarlySourceSlowEMATerminalRawGuardH4000AnchorFloorFU":
            source = "train_stream_terminal_raw_guard_h4000_anchor_floor_wrapper_slow_state"
        if mechanism == "M193-EarlySourceSlowEMATerminalRawGuardDecayAwareFloorFU":
            source = "train_stream_terminal_raw_guard_decay_aware_floor_wrapper_slow_state"
        if mechanism == "M194-EarlySourceSlowEMATerminalAntiErosionOrthogonalFU":
            source = "train_stream_terminal_anti_erosion_orthogonal_wrapper_slow_state"
        if mechanism == "M195-EarlySourceSlowEMATerminalSourceReflectionGuardFU":
            source = "train_stream_terminal_source_reflection_guard_wrapper_slow_state"
        if mechanism == "M196-EarlySourceSlowEMATerminalH4000TransportCorrectorFU":
            source = "train_stream_terminal_h4000_transport_corrector_wrapper_slow_state"
        if mechanism == "M197-EarlySourceSlowEMATerminalH3200AnchorTransportFU":
            source = "train_stream_terminal_h3200_anchor_transport_wrapper_slow_state"
        if mechanism == "M198-EarlySourceSlowEMATerminalRawGuardH3200AnchorTransportFU":
            source = "train_stream_terminal_raw_guard_h3200_anchor_transport_wrapper_slow_state"
        if mechanism == "M199-EarlySourceSlowEMATerminalH3200RatioReentryFU":
            source = "train_stream_terminal_h3200_ratio_reentry_wrapper_slow_state"
        if mechanism == "M200-EarlySourceSlowEMATerminalH3200ProgressCarryFU":
            source = "train_stream_terminal_h3200_progress_carry_wrapper_slow_state"
        if mechanism == "M201-EarlySourceSlowEMATerminalH4000ProgressCarryFU":
            source = "train_stream_terminal_h4000_progress_carry_wrapper_slow_state"
        if mechanism == "M202-EarlySourceSlowEMATerminalH3200SourceProgressBlendFU":
            source = "train_stream_terminal_h3200_source_progress_blend_wrapper_slow_state"
        if mechanism == "M203-EarlySourceSlowEMATerminalRawGuardH3600GentleProgressFU":
            source = "train_stream_terminal_raw_guard_h3600_gentle_progress_wrapper_slow_state"
        if mechanism == "M204-EarlySourceSlowEMATerminalRawGuardH4000GentleProgressFU":
            source = "train_stream_terminal_raw_guard_h4000_gentle_progress_wrapper_slow_state"
        if mechanism == "M205-EarlySourceSlowEMATerminalRawGuardH4400ProjectedProgressFU":
            source = "train_stream_terminal_raw_guard_h4400_projected_progress_wrapper_slow_state"
        if mechanism == "M206-EarlySourceSlowEMATerminalControlRelativeSGDCatchupFU":
            source = "train_stream_terminal_control_relative_sgd_catchup_wrapper_slow_state"
        if mechanism == "M207-EarlySourceSlowEMATerminalControlRelativeAdamWCatchupFU":
            source = "train_stream_terminal_control_relative_adamw_catchup_wrapper_slow_state"
        if mechanism == "M208-EarlySourceSlowEMATerminalControlRelativeSourceBalancedCatchupFU":
            source = "train_stream_terminal_control_relative_source_balanced_catchup_wrapper_slow_state"
        if mechanism == "M209-EarlySourceSlowEMATerminalTrajectoryAdaptivePreserveFU":
            source = "train_stream_terminal_trajectory_adaptive_preserve_wrapper_slow_state"
        if mechanism == "M210-EarlySourceSlowEMATerminalMidErosionBridgeFU":
            source = "train_stream_terminal_mid_erosion_bridge_wrapper_slow_state"
        if mechanism == "M211-EarlySourceSlowEMATerminalTwoPhaseRatioRepairFU":
            source = "train_stream_terminal_two_phase_ratio_repair_wrapper_slow_state"
        if mechanism == "M212-EarlySourceSlowEMATerminalAntiSourceClipFU":
            source = "train_stream_terminal_anti_source_clip_wrapper_slow_state"
        if mechanism == "M213-EarlySourceSlowEMATerminalDebtAwareHoldFU":
            source = "train_stream_terminal_debt_aware_hold_wrapper_slow_state"
        if mechanism == "M214-EarlySourceSlowEMATerminalAnchorFlowTinyFU":
            source = "train_stream_terminal_anchor_flow_tiny_wrapper_slow_state"
        if mechanism == "M215-EarlySourceSlowEMATerminalAcceptMemoryFU":
            source = "train_stream_terminal_accept_memory_wrapper_slow_state"
        if mechanism == "M216-EarlySourceSlowEMATerminalAcceptMemoryDebtFU":
            source = "train_stream_terminal_accept_memory_debt_wrapper_slow_state"
        if mechanism == "M217-EarlySourceSlowEMATerminalSourceProgressMemoryFU":
            source = "train_stream_terminal_source_progress_memory_wrapper_slow_state"
        update_vec = _momentum_like_precondition(g)
        diagnostics: dict[str, object] = {}
        if mechanism in TERMINAL_SOURCE_SLOW_STATE_MECHANISMS:
            if slow_state is not None and slow_state.numel() == update_vec.numel():
                state = slow_state.to(device=device, dtype=update_vec.dtype)
                update_vec = normalized_like(0.92 * state + 0.08 * update_vec, g)
                diagnostics["terminal_source_slow_state_used"] = 1
                diagnostics["terminal_source_slow_state_beta"] = 0.92
                diagnostics["terminal_source_slow_state_norm"] = float(torch.linalg.vector_norm(state.detach()).item())
            else:
                diagnostics["terminal_source_slow_state_used"] = 0
                diagnostics["terminal_source_slow_state_beta"] = 0.92
                diagnostics["terminal_source_slow_state_norm"] = ""
        return UpdateTensor(update_vec, "step", "subtract", "slow_state", source, mechanism, one_step_descent_claim=0, diagnostics=diagnostics)
    if mechanism == "M6-SlowStateFU":
        if slow_state is None or slow_state.numel() != g.numel():
            s = _slow_state_initial(g)
        else:
            s = 0.90 * slow_state.to(device=device) + 0.10 * g
        return UpdateTensor(s, "step", "subtract", "slow_state", "train_stream_slow_signal_state", mechanism)
    if mechanism == "M7-MatrixBlockFU":
        mask = _matrix_block_mask(model, device)
        return UpdateTensor(g * mask, "step", "subtract", "matrix_block", "train_stream_block_gradient", mechanism)
    if mechanism == "M9-FunctionSpaceOperatorFU":
        return UpdateTensor(0.50 * g, "cotangent", "subtract", "function", "train_stream_output_cotangent_fixed_alpha", mechanism)
    if mechanism == "M10-RolePartitionOptimizer":
        mask = _basis_role_mask(model, device)
        return UpdateTensor(g * (0.75 + 0.25 * mask), "step", "subtract", "role_partition", "train_stream_role_partition_gradient", mechanism)
    if mechanism == "M11-DualMemorySlowStateFU":
        if slow_state is not None and slow_state.numel() == 2 * g.numel():
            short, long = slow_state.to(device=device).chunk(2)
        else:
            short = _slow_state_initial(g)
            long = _slow_state_initial(g)
        short_new = 0.80 * short + 0.20 * g
        long_new = 0.98 * long + 0.02 * g
        agree = torch.sign(short_new) == torch.sign(long_new)
        mixed = torch.where(agree, 0.5 * (short_new + long_new), 0.25 * (short_new + long_new))
        return UpdateTensor(normalized_like(mixed, g), "step", "subtract", "slow_state", "train_stream_dual_memory_source_state", mechanism)
    if mechanism == "M12-ScheduleFreeAveragedFU":
        if slow_state is None or slow_state.numel() != g.numel():
            avg = g.detach().clone()
        else:
            avg = 0.95 * slow_state.to(device=device) + 0.05 * g
        mixed = 0.35 * g + 0.65 * avg
        return UpdateTensor(normalized_like(mixed, g), "step", "subtract", "slow_state", "train_stream_schedule_free_averaged_fu", mechanism)
    if mechanism == "M13-LowRankMatrixBlockFU":
        low_rank = _low_rank_matrix_update(model, device, rank=4)
        return UpdateTensor(normalized_like(low_rank, g), "step", "subtract", "matrix_block", "train_stream_low_rank_matrix_block_svd_r4", mechanism)
    if mechanism == "M14-SourceChannelPopRiskSlowFU":
        u, diagnostics = poprisk_snr_update_with_diagnostics(model, x, y)
        if slow_state is None or slow_state.numel() != u.numel():
            s = u.detach().clone()
        else:
            s = 0.95 * slow_state.to(device=device) + 0.05 * u
        diagnostics["PopRisk_grad_cosine"] = cosine(s, g)
        diagnostics["PopRisk_slow_state_norm"] = float(torch.linalg.vector_norm(s.detach()).item())
        diagnostics["PopRisk_block_projection"] = 0
        return UpdateTensor(normalized_like(s, g), "step", "subtract", "basis_channel", "train_stream_poprisk_snr_slow_channel_state", mechanism, diagnostics=diagnostics)
    if mechanism == "M33-PopRiskMatrixBlockSlowFU":
        u, diagnostics = poprisk_snr_update_with_diagnostics(model, x, y)
        block_u = u * _matrix_block_mask(model, device)
        if slow_state is None or slow_state.numel() != block_u.numel():
            s = block_u.detach().clone()
        else:
            s = 0.95 * slow_state.to(device=device) + 0.05 * block_u
        diagnostics["PopRisk_grad_cosine"] = cosine(s, g)
        diagnostics["PopRisk_slow_state_norm"] = float(torch.linalg.vector_norm(s.detach()).item())
        diagnostics["PopRisk_block_projection"] = 1
        diagnostics["block_update_norm"] = float(torch.linalg.vector_norm(block_u.detach()).item())
        return UpdateTensor(normalized_like(s, g), "step", "subtract", "block", "train_stream_poprisk_snr_matrix_block_slow_state", mechanism, diagnostics=diagnostics)
    if mechanism == "M34-ExactPopRiskSlowFU":
        u, diagnostics = poprisk_snr_update_with_diagnostics(model, x, y, max_examples=32, exact_variance=True)
        if slow_state is None or slow_state.numel() != u.numel():
            s = u.detach().clone()
        else:
            s = 0.95 * slow_state.to(device=device) + 0.05 * u
        diagnostics["PopRisk_grad_cosine"] = cosine(s, g)
        diagnostics["PopRisk_slow_state_norm"] = float(torch.linalg.vector_norm(s.detach()).item())
        diagnostics["PopRisk_block_projection"] = 0
        return UpdateTensor(normalized_like(s, g), "step", "subtract", "basis_channel", "train_stream_exact_poprisk_snr_slow_channel_state_k32", mechanism, diagnostics=diagnostics)
    if mechanism == "M35-ExactPopRiskMatrixBlockSlowFU":
        u, diagnostics = poprisk_snr_update_with_diagnostics(model, x, y, max_examples=32, exact_variance=True)
        block_u = u * _matrix_block_mask(model, device)
        if slow_state is None or slow_state.numel() != block_u.numel():
            s = block_u.detach().clone()
        else:
            s = 0.95 * slow_state.to(device=device) + 0.05 * block_u
        diagnostics["PopRisk_grad_cosine"] = cosine(s, g)
        diagnostics["PopRisk_slow_state_norm"] = float(torch.linalg.vector_norm(s.detach()).item())
        diagnostics["PopRisk_block_projection"] = 1
        diagnostics["block_update_norm"] = float(torch.linalg.vector_norm(block_u.detach()).item())
        return UpdateTensor(normalized_like(s, g), "step", "subtract", "block", "train_stream_exact_poprisk_snr_matrix_block_slow_state_k32", mechanism, diagnostics=diagnostics)
    raise ValueError(f"unknown mechanism {mechanism}")


def update_state_after_commit(mechanism: str, previous: torch.Tensor | None, update: UpdateTensor) -> torch.Tensor | None:
    if mechanism == "M11-DualMemorySlowStateFU":
        if previous is not None and previous.numel() == 2 * update.tensor.numel():
            short, long = previous.detach().chunk(2)
        else:
            short = update.tensor.detach().clone()
            long = update.tensor.detach().clone()
        short = 0.80 * short + 0.20 * update.tensor.detach()
        long = 0.98 * long + 0.02 * update.tensor.detach()
        return torch.cat([short, long])
    if mechanism in {
        "M12-ScheduleFreeAveragedFU",
        "M14-SourceChannelPopRiskSlowFU",
        "M33-PopRiskMatrixBlockSlowFU",
        "M34-ExactPopRiskSlowFU",
        "M35-ExactPopRiskMatrixBlockSlowFU",
        "M116-DatasetInvariantPopRiskSlowFU",
        "M117-DatasetInvariantReadoutConsensusFU",
    }:
        if previous is None or previous.numel() != update.tensor.numel():
            return update.tensor.detach().clone()
        return 0.95 * previous.detach() + 0.05 * update.tensor.detach()
    if mechanism in METRIC_SOURCE_MEMORY_MECHANISMS or mechanism in TERMINAL_SOURCE_SLOW_STATE_MECHANISMS:
        if previous is None or previous.numel() != update.tensor.numel():
            return update.tensor.detach().clone()
        return 0.96 * previous.detach() + 0.04 * update.tensor.detach()
    if mechanism != "M6-SlowStateFU":
        return previous
    if previous is None or previous.numel() != update.tensor.numel():
        return update.tensor.detach().clone()
    return 0.90 * previous.detach() + 0.10 * update.tensor.detach()


def mechanism_contract_rows() -> list[dict[str, object]]:
    specs = {
        "CTRL-AdamW": ("parameter", "AdamW", 0, 0, 0, 0, 0),
        "CTRL-SGD": ("parameter", "SGD", 0, 0, 0, 0, 0),
        "CTRL-NoOpMatchedOverhead": ("parameter", "none", 0, 0, 0, 0, 0),
        "CTRL-RandomMatchedNorm": ("parameter", "none", 0, 0, 0, 0, 0),
        "CTRL-RandomSameRankBlock": ("block", "none", 0, 1, 0, 0, 0),
        "CTRL-RecoveryOnly": ("parameter", "AdamW", 0, 0, 0, 0, 0),
        "M1-AdamWPrimaryFUResidual": ("parameter", "AdamW", 0, 0, 0, 0, 0),
        "M2-SGDMomentumPrimaryFU": ("parameter", "Momentum", 0, 0, 0, 0, 0),
        "M3-FUPrimary": ("function", "none", 0, 0, 0, 1, 1),
        "M4-FUOnlyKeyParams": ("basis-channel", "none", 0, 0, 0, 0, 1),
        "M5-AlternatingFUGradient": ("trajectory", "SGD", 0, 0, 0, 0, 1),
        "M6-SlowStateFU": ("slow_state", "none", 1, 0, 0, 0, 1),
        "M7-MatrixBlockFU": ("block", "none", 0, 1, 0, 0, 1),
        "M8-PopRiskSNRFU": ("basis-channel", "none", 0, 0, 1, 0, 0),
        "M9-FunctionSpaceOperatorFU": ("function", "none", 0, 0, 0, 1, 1),
        "M10-RolePartitionOptimizer": ("role_partition", "role_partition", 0, 0, 0, 0, 1),
        "M11-DualMemorySlowStateFU": ("slow_state", "none", 1, 0, 0, 0, 0),
        "M12-ScheduleFreeAveragedFU": ("slow_state", "schedule_free", 1, 0, 0, 0, 0),
        "M13-LowRankMatrixBlockFU": ("block", "none", 0, 1, 0, 0, 0),
        "M14-SourceChannelPopRiskSlowFU": ("basis-channel", "none", 1, 0, 1, 0, 0),
        "M15-LineCFilteredAlternatingFU": ("trajectory", "SGD", 0, 0, 0, 0, 0),
        "M16-TwoPhaseMomentumThenLineCFU": ("trajectory", "SGD+filtered", 0, 0, 0, 0, 0),
        "M17-ReadoutCarrierTwoPhaseLineCFU": ("readout_carrier", "SGD+filtered", 0, 1, 0, 0, 0),
        "M18-CarrierLowRankBlockFU": ("block", "none", 0, 1, 0, 0, 0),
        "M19-SplitConsensusLowRankBlockFU": ("block", "none", 0, 1, 0, 0, 0),
        "M20-TrainSplitFunctionSpaceActuationFU": ("function", "none", 0, 0, 0, 1, 0),
        "M21-ExactReadoutFunctionSpaceActuationFU": ("function", "none", 0, 0, 0, 1, 0),
        "M22-ExactReadoutHighCapScheduledFU": ("function", "none", 0, 0, 0, 1, 0),
        "M23-ExactReadoutUltraCapScheduledFU": ("function", "none", 0, 0, 0, 1, 0),
        "M24-WarmupExactReadoutUltraCapFU": ("function", "SGD+none", 0, 0, 0, 1, 0),
        "M25-AdamWExactReadoutUltraCapFU": ("function", "AdamW", 0, 0, 0, 1, 0),
        "M26-GatedAdamWExactReadoutFU": ("function", "AdamW", 0, 0, 0, 1, 0),
        "M27-MultiBatchGeneralizationGatedReadoutFU": ("function", "SGD+gated", 0, 0, 0, 1, 0),
        "M28-AdamWMultiBatchGatedReadoutFU": ("function", "AdamW+gated", 0, 0, 0, 1, 0),
        "M29-ConsensusSourceStateFU": ("slow_state", "SGD+train_consensus_gate", 1, 0, 0, 0, 0),
        "M30-LowThresholdConsensusSourceStateFU": ("slow_state", "SGD+low_threshold_consensus_gate", 1, 0, 0, 0, 0),
        "M31-AdamWLowThresholdConsensusResidualFU": ("slow_state", "AdamW+low_threshold_consensus_residual", 1, 0, 0, 0, 0),
        "M32-PostAdamWConsensusResidualFU": ("slow_state", "AdamW+post_step_consensus_residual", 1, 0, 0, 0, 0),
        "M33-PopRiskMatrixBlockSlowFU": ("block", "none", 1, 1, 1, 0, 0),
        "M34-ExactPopRiskSlowFU": ("basis-channel", "none", 1, 0, 1, 0, 0),
        "M35-ExactPopRiskMatrixBlockSlowFU": ("block", "none", 1, 1, 1, 0, 0),
        "M36-MomentumWarmReadoutBlockFU": ("readout_carrier", "SGD+filtered", 0, 1, 0, 0, 0),
        "M37-SplitFisherAgreementSlowFU": ("slow_state", "SGD+split_fisher_gate", 1, 0, 0, 0, 0),
        "M38-AdamWSplitFisherAgreementResidualFU": ("slow_state", "AdamW+split_fisher_residual", 1, 0, 0, 0, 0),
        "M39-MomentumWarmSplitFisherFU": ("slow_state", "Momentum+split_fisher_gate", 1, 0, 0, 0, 0),
        "M40-MomentumWarmAntiWashoutFU": ("slow_state", "Momentum+anti_washout_projection", 1, 0, 0, 0, 0),
        "M41-MomentumWarmSourceAnchorFU": ("slow_state", "Momentum+source_anchor_residual", 1, 0, 0, 0, 0),
        "M42-MomentumWarmHoldFU": ("slow_state", "Momentum+post_warmup_hold", 1, 0, 0, 0, 0),
        "M43-MomentumCycleHoldFU": ("slow_state", "Momentum+cycle_hold", 1, 0, 0, 0, 0),
        "M44-MomentumLineCAnchorSlowFU": ("slow_state", "Momentum+LineC_anchor_slow", 1, 0, 0, 0, 0),
        "M45-MomentumSlowAnchorFU": ("slow_state", "Momentum+slow_anchor", 1, 0, 0, 0, 0),
        "M46-MomentumMatrixBlockRetentionFU": ("block", "Momentum+matrix_block_retention", 1, 1, 0, 0, 0),
        "M47-MomentumCycleThenLineCAnchorFU": ("slow_state", "Momentum_cycle_then_LineC_anchor", 1, 0, 0, 0, 0),
        "M48-DualTimescaleSourceRetentionFU": ("slow_state", "Momentum+dual_timescale_anchor", 1, 0, 0, 0, 0),
        "M49-LossCotangentTargetFU": ("function", "SGD+target_reset", 0, 0, 0, 1, 0),
        "M50-RandomMatchedTargetFU": ("function", "SGD+target_control", 0, 0, 0, 1, 0),
        "M51-SignFlippedTargetFU": ("function", "SGD+target_control", 0, 0, 0, 1, 0),
        "M52-CorruptedLabelTargetFU": ("function", "SGD+target_control", 0, 0, 0, 1, 0),
        "M53-LowRankLossCotangentTargetFU": ("function", "SGD+target_reset_low_rank", 0, 0, 0, 1, 0),
        "M54-CrossSplitConsensusTargetFU": ("function", "SGD+target_reset_cross_split", 0, 0, 0, 1, 0),
        "M55-LowDegreeReadoutTargetFU": ("function", "SGD+target_reset_low_degree", 0, 0, 0, 1, 0),
        "M56-WeakStableTargetFU": ("function", "SGD+target_reset_weak_stable", 0, 0, 0, 1, 0),
        "M57-B1ReadoutTransferTargetFU": ("function", "SGD+target_reset_b1_transfer", 0, 0, 0, 1, 0),
        "M58-ReservoirExcludingConsensusTargetFU": ("function", "SGD+target_reset_reservoir_excluding", 0, 0, 0, 1, 0),
        "M59-StableRandomTargetControlFU": ("function", "SGD+target_control_stable_random", 0, 0, 0, 1, 0),
        "M60-B1CrossSplitConsensusTransferFU": ("function", "SGD+target_reset_b1_cross_split", 0, 0, 0, 1, 0),
        "M61-StableB1ConsensusTransferFU": ("function", "SGD+target_reset_stable_b1_consensus", 0, 0, 0, 1, 0),
        "M62-B1WeakStableTransferFU": ("function", "SGD+target_reset_b1_weak_stable", 0, 0, 0, 1, 0),
        "M63-LowDegreeB1ConsensusTransferFU": ("function", "SGD+target_reset_low_degree_b1_consensus", 0, 0, 0, 1, 0),
        "M64-ContrastiveLossRandomOrthogonalTargetFU": ("function", "SGD+target_reset_loss_minus_random_component", 0, 0, 0, 1, 0),
        "M65-B1ContrastiveLossRandomOrthogonalTransferFU": ("function", "SGD+target_reset_b1_loss_minus_random_component", 0, 0, 0, 1, 0),
        "M66-TopWrongMarginTargetFU": ("function", "SGD+target_reset_top_wrong_margin", 0, 0, 0, 1, 0),
        "M67-B1TopWrongMarginTransferFU": ("function", "SGD+target_reset_b1_top_wrong_margin", 0, 0, 0, 1, 0),
        "M68-MarginSplitConsensusSlowFU": ("slow_state", "SGD+top_wrong_margin_split_consensus", 1, 0, 0, 0, 0),
        "M69-RotatedCautiousMatrixSlowFU": ("block", "Momentum+rotated_cautious_matrix_slow_state", 1, 1, 0, 0, 0),
        "M70-TrainLookaheadCautiousSourceFU": ("block", "Momentum+train_lookahead_cautious_source", 1, 1, 0, 0, 0),
        "M71-TrainLookaheadB1ConsensusTransferFU": ("function", "SGD+train_lookahead_b1_consensus_transfer", 0, 0, 0, 1, 0),
        "M72-LossWarmToB1ConsensusMigrationFU": ("function", "SGD+loss_warm_to_b1_consensus_migration", 0, 0, 0, 1, 0),
        "M73-EasyB1ConsensusTransferFU": ("function", "SGD+easy_b1_consensus_transfer", 0, 0, 0, 1, 0),
        "M74-LossWarmToEasyB1ConsensusMigrationFU": ("function", "SGD+loss_warm_to_easy_b1_consensus_migration", 0, 0, 0, 1, 0),
        "M75-LossEasyB1ConsensusBlendFU": ("function", "SGD+loss_easy_b1_consensus_blend", 0, 0, 0, 1, 0),
        "M76-LossWarmToLossEasyB1ConsensusBlendFU": ("function", "SGD+loss_warm_to_loss_easy_b1_consensus_blend", 0, 0, 0, 1, 0),
        "M77-GainGatedLossWarmB1ConsensusMigrationFU": ("function", "SGD+gain_gated_loss_warm_to_b1_consensus_migration", 0, 0, 0, 1, 0),
        "M78-GainGatedLossWarmBlendMigrationFU": ("function", "SGD+gain_gated_loss_warm_to_loss_easy_b1_consensus_blend", 0, 0, 0, 1, 0),
        "M79-B1ConsensusB3NullTransferFU": ("function", "SGD+b1_consensus_b3_null_transfer", 0, 0, 0, 1, 0),
        "M80-LossWarmToB1ConsensusB3NullMigrationFU": ("function", "SGD+loss_warm_to_b1_consensus_b3_null_migration", 0, 0, 0, 1, 0),
        "M81-ViewConsistentLossTargetFU": ("function", "SGD+view_consistent_loss_b3_null_target", 0, 0, 0, 1, 0),
        "M82-LossWarmToViewConsistentLossMigrationFU": ("function", "SGD+loss_warm_to_view_consistent_loss_migration", 0, 0, 0, 1, 0),
        "M83-LowBankLossB3NullFU": ("function", "SGD+low_bank_loss_b3_null_source_channel", 0, 0, 0, 1, 0),
        "M84-LossWarmToLowBankLossB3NullMigrationFU": ("function", "SGD+loss_warm_to_low_bank_loss_b3_null_migration", 0, 0, 0, 1, 0),
        "M85-GainGatedLowBankLossB3NullFU": ("function", "SGD+gain_gated_low_bank_loss_b3_null_source_channel", 0, 0, 0, 1, 0),
        "M86-LossWarmToGainGatedLowBankLossB3NullMigrationFU": ("function", "SGD+loss_warm_to_gain_gated_low_bank_loss_b3_null_migration", 0, 0, 0, 1, 0),
        "M109-AdamWBoundaryToGainGatedLowBankB3NullFU": ("function", "AdamW_boundary_to_gain_gated_low_bank_loss_b3_null_source_channel", 0, 0, 0, 1, 0),
        "M110-AdamWBoundaryLowBankAnchorAntiWashoutFU": ("function", "AdamW_boundary_low_bank_anchor_antiwashout_source_channel", 0, 0, 0, 1, 0),
        "M87-AdamWBoundaryThenMomentumSourceFU": ("parameter", "AdamW_boundary_then_Momentum", 0, 0, 0, 0, 0),
        "M88-AdamWBoundaryThenDualTimescaleSourceFU": ("slow_state", "AdamW_boundary_then_dual_timescale", 1, 0, 0, 0, 0),
        "M89-AdamWBoundaryDualTimescaleSGDFloorFU": ("slow_state", "AdamW_boundary_dual_timescale_with_SGD_floor", 1, 0, 0, 0, 0),
        "M90-AdamWBoundaryDualTimescaleLateSGDFloorFU": ("slow_state", "AdamW_boundary_dual_timescale_with_late_SGD_floor", 1, 0, 0, 0, 0),
        "M91-AdamWBoundaryDualTimescaleTinyLateSGDFloorFU": ("slow_state", "AdamW_boundary_dual_timescale_with_tiny_late_SGD_floor", 1, 0, 0, 0, 0),
        "M92-AdamWBoundaryDualTimescaleReinforcedTinyLateSGDFloorFU": ("slow_state", "AdamW_boundary_reinforced_dual_timescale_with_tiny_late_SGD_floor", 1, 0, 0, 0, 0),
        "M93-TrainLossGatedDualTimescaleTinyLateFU": ("slow_state", "AdamW_boundary_dual_timescale_tiny_late_train_loss_gate", 1, 0, 0, 0, 0),
        "M94-TrainLossGatedDualTimescaleHoldFallbackFU": ("slow_state", "AdamW_boundary_dual_timescale_train_loss_gate_hold_fallback", 1, 0, 0, 0, 0),
        "M95-TrainLossGatedDualTimescaleTinyLateFallbackFU": ("slow_state", "AdamW_boundary_dual_timescale_train_loss_gate_tiny_late_fallback", 1, 0, 0, 0, 0),
        "M96-TrainLossGatedDualTimescaleBoostedTinyLateFallbackFU": ("slow_state", "AdamW_boundary_dual_timescale_train_loss_gate_boosted_tiny_late_fallback", 1, 0, 0, 0, 0),
        "M97-TrainLossLateHoldRecoveryFU": ("slow_state", "AdamW_boundary_dual_timescale_train_loss_gate_late_hold_recovery", 1, 0, 0, 0, 0),
        "M98-TrainLossLateLookaheadFloorFU": ("slow_state", "AdamW_boundary_dual_timescale_train_loss_gate_late_lookahead_floor", 1, 0, 0, 0, 0),
        "M99-TrainLossTerminalLookaheadFloorFU": ("slow_state", "AdamW_boundary_dual_timescale_train_loss_gate_terminal_lookahead_floor", 1, 0, 0, 0, 0),
        "M100-TrainLossEarlyTerminalLookaheadFloorFU": ("slow_state", "AdamW_boundary_dual_timescale_train_loss_gate_early_terminal_lookahead_floor", 1, 0, 0, 0, 0),
        "M106-TrainLossTerminalProjectedLookaheadFloorFU": ("slow_state", "AdamW_boundary_dual_timescale_train_loss_gate_terminal_projected_lookahead_floor", 1, 0, 0, 0, 0),
        "M107-TrainLossTerminalConsensusLookaheadFloorFU": ("slow_state", "AdamW_boundary_dual_timescale_train_loss_gate_terminal_consensus_lookahead_floor", 1, 0, 0, 0, 0),
        "M108-TrainLossTerminalSelectorLookaheadFloorFU": ("slow_state", "AdamW_boundary_dual_timescale_train_loss_gate_terminal_selector_lookahead_floor", 1, 0, 0, 0, 0),
        "M111-TrainLossTerminalPositiveLookaheadFloorFU": ("slow_state", "AdamW_boundary_dual_timescale_train_loss_gate_terminal_positive_lookahead_floor", 1, 0, 0, 0, 0),
        "M112-TrainLossTerminalCheckpointReentryFU": ("slow_state", "AdamW_boundary_dual_timescale_train_loss_gate_terminal_h3200_checkpoint_reentry", 1, 0, 0, 0, 0),
        "M113-TrainLossTerminalHardSplitSourceFU": ("slow_state", "AdamW_boundary_dual_timescale_train_loss_gate_terminal_hard_split_source", 1, 0, 0, 0, 0),
        "M114-TrainLossTerminalAdamWLookaheadFU": ("slow_state", "AdamW_boundary_dual_timescale_train_loss_gate_terminal_adamw_lookahead", 1, 0, 0, 0, 0),
        "M115-TrainLossTerminalOptimizerSelectorFU": ("slow_state", "AdamW_boundary_dual_timescale_train_loss_gate_terminal_optimizer_selector", 1, 0, 0, 0, 0),
        "M101-AdamWBoundaryDualTimescaleAntiWashoutFU": ("slow_state", "AdamW_boundary_dual_timescale_antiwashout_projection", 1, 0, 0, 0, 0),
        "M102-AdamWBoundaryDualTimescaleSourceAnchorFU": ("slow_state", "AdamW_boundary_dual_timescale_source_anchor_residual", 1, 0, 0, 0, 0),
        "M103-AdamWBoundaryDualTimescaleParamEMAReentryFU": ("slow_state", "AdamW_boundary_dual_timescale_parameter_ema_reentry", 1, 0, 0, 0, 0),
        "M104-AdamWBoundaryDualTimescaleReadoutChannelFU": ("readout_carrier", "AdamW_boundary_dual_timescale_readout_channel_reset", 1, 0, 0, 0, 0),
        "M105-AdamWBoundaryDualTimescaleHiddenMatrixChannelFU": ("matrix_block", "AdamW_boundary_dual_timescale_hidden_matrix_channel_reset", 1, 1, 0, 0, 0),
        "M116-DatasetInvariantPopRiskSlowFU": ("slow_state", "split_poprisk_snr_dataset_invariant_slow_state", 1, 0, 1, 0, 0),
        "M117-DatasetInvariantReadoutConsensusFU": ("readout_carrier", "split_readout_consensus_corrupt_filtered", 1, 0, 0, 0, 0),
        "M118-SourceConservingOptimizerOnlyFU": ("slow_state", "AdamW_boundary_dual_timescale_source_conserving_optimizer_no_fallback", 1, 0, 0, 0, 0),
        "M119-TerminalSourceConservingRouteFU": ("slow_state", "terminal_source_conserving_projected_consensus_source_route", 1, 0, 0, 0, 0),
        "M120-TrainLossRiskProfileRouteFU": ("slow_state", "train_loss_risk_profile_terminal_route", 1, 0, 0, 0, 0),
        "M121-DebtAwareSourceGateFU": ("slow_state", "debt_aware_terminal_source_gate", 1, 0, 0, 0, 0),
        "M122-UngatedWarmTerminalSourceRouteFU": ("slow_state", "ungated_h3200_warm_terminal_source_conserving_route", 1, 0, 0, 0, 0),
        "M123-UngatedWarmRiskProfileRouteFU": ("slow_state", "ungated_h3200_warm_train_loss_risk_profile_route", 1, 0, 0, 0, 0),
        "M124-UngatedWarmDebtRawBailoutFU": ("slow_state", "ungated_h3200_warm_debt_aware_raw_terminal_bailout", 1, 0, 0, 0, 0),
        "M125-TrainLossH2400CheckpointHoldFU": ("slow_state", "h2400_terminal_collapse_hold_without_future_leakage", 1, 0, 0, 0, 0),
        "M126-TrainLossH2800CheckpointHoldFU": ("slow_state", "h2800_terminal_collapse_hold_without_future_leakage", 1, 0, 0, 0, 0),
        "M127-TrainLossH2400DebtBailoutFU": ("slow_state", "h2400_terminal_hold_with_train_only_debt_bailout", 1, 0, 0, 0, 0),
        "M128-TrainLossTerminalRawThenSourceGuardFU": ("slow_state", "terminal_raw_then_source_guard_train_stream", 1, 0, 0, 0, 0),
        "M129-H800SourceSlowEMARetentionFU": ("slow_state", "h800_source_slow_ema_retention_train_stream", 1, 0, 0, 0, 0),
        "M130-H800ReadoutChannelRetentionFU": ("readout_carrier", "h800_readout_channel_retained_target_train_stream", 1, 0, 0, 0, 0),
        "M131-H800DualMemorySourceRetentionFU": ("slow_state", "h800_short_long_dual_memory_source_retention_train_stream", 1, 0, 0, 0, 0),
        "M132-H1600SourceCheckpointReentryFU": ("slow_state", "h1600_train_time_source_checkpoint_reentry", 1, 0, 0, 0, 0),
        "M133-H2400SourceCheckpointReentryFU": ("slow_state", "h2400_train_time_source_checkpoint_reentry", 1, 0, 0, 0, 0),
        "M134-SignalReservoirB3NullConsensusTargetFU": ("function", "SGD+signal_reservoir_b3_null_consensus_target", 0, 0, 0, 1, 0),
        "M135-SourceBankB3NullConsensusTargetFU": ("function", "SGD+source_bank_b3_null_consensus_target", 0, 0, 0, 1, 0),
        "M136-StableRandomB3NullTargetControlFU": ("function", "SGD+stable_random_b3_null_target_control", 0, 0, 0, 1, 0),
        "M137-EarlyPulseSourceBankB3NullConsensusTargetFU": ("function", "SGD+early_pulse_source_bank_b3_null_consensus_target", 0, 0, 0, 1, 0),
        "M138-EarlyPulseAdamWSourceBankB3NullConsensusTargetFU": ("function", "AdamW+early_pulse_source_bank_b3_null_consensus_target", 0, 0, 0, 1, 0),
        "M139-EarlySourceSlowEMATerminalGuardFU": ("slow_state", "early100_h800_slow_ema_with_terminal_source_guard", 1, 0, 0, 0, 0),
        "M140-EarlySourceSlowEMATerminalRawGuardFU": ("slow_state", "early100_h800_slow_ema_with_terminal_raw_then_source_guard", 1, 0, 0, 0, 0),
        "M141-EarlySourceSlowEMATerminalRawGuardStrongFU": ("slow_state", "early100_h800_slow_ema_with_stronger_terminal_raw_then_source_guard", 1, 0, 0, 0, 0),
        "M142-EarlySourceSlowEMATerminalProjectedOptimizerFU": ("slow_state", "early100_h800_slow_ema_with_terminal_projected_optimizer", 1, 0, 0, 0, 0),
        "M143-EarlySourceSlowEMATerminalProjectedBlendFU": ("slow_state", "early100_h800_slow_ema_with_terminal_projected_source_blend", 1, 0, 0, 0, 0),
        "M144-EarlySourceSlowEMATerminalAntiWashoutFU": ("slow_state", "early100_h800_slow_ema_with_terminal_antiwashout_projector", 1, 0, 0, 0, 0),
        "M145-EarlySourceSlowEMATerminalH4000ReentryFU": ("slow_state", "early100_h800_slow_ema_with_h4000_terminal_reentry", 1, 0, 0, 0, 0),
        "M146-EarlySourceSlowEMASignalReservoirTargetFU": ("slow_state+function", "early100_h800_slow_ema_with_signal_reservoir_b3_null_target_gate", 1, 0, 0, 1, 0),
        "M147-EarlySourceSlowEMASourceBankTargetFU": ("slow_state+function", "early100_h800_slow_ema_with_source_bank_b3_null_target_gate", 1, 0, 0, 1, 0),
        "M148-EarlySourceSlowEMADualTargetGuardFU": ("slow_state+function", "early100_h800_slow_ema_with_dual_signal_source_bank_target_gate", 1, 0, 0, 1, 0),
        "M149-EarlySourceSlowEMATerminalAdaptiveRawFU": ("slow_state", "early100_h800_slow_ema_with_source_state_adaptive_raw_terminal_guard", 1, 0, 0, 0, 0),
        "M150-EarlySourceSlowEMATerminalSparseSourceFU": ("slow_state", "early100_h800_slow_ema_with_sparse_source_axis_terminal_guard", 1, 0, 0, 0, 0),
        "M151-EarlySourceSlowEMATerminalRatioPreserveFU": ("slow_state", "early100_h800_slow_ema_with_ratio_preserving_terminal_source_guard", 1, 0, 0, 0, 0),
        "M152-SourceProjectionB3NullConsensusTargetFU": ("slow_state+function", "early100_h800_slow_ema_with_terminal_raw_guard_and_source_projection_target", 1, 0, 0, 1, 0),
        "M153-NoiseOrthogonalB3NullConsensusTargetFU": ("slow_state+function", "early100_h800_slow_ema_with_terminal_raw_guard_and_noise_orthogonal_target", 1, 0, 0, 1, 0),
        "M154-EasyMarginB3NullConsensusTargetFU": ("slow_state+function", "early100_h800_slow_ema_with_terminal_raw_guard_and_easy_margin_target", 1, 0, 0, 1, 0),
        "M155-SourceProjectionB3NullConsensusTargetOnlyFU": ("function", "SGD+source_projection_b3_null_consensus_target", 0, 0, 0, 1, 0),
        "M156-NoiseOrthogonalB3NullConsensusTargetOnlyFU": ("function", "SGD+noise_orthogonal_b3_null_consensus_target", 0, 0, 0, 1, 0),
        "M157-EasyMarginB3NullConsensusTargetOnlyFU": ("function", "SGD+easy_margin_b3_null_consensus_target", 0, 0, 0, 1, 0),
        "M158-EarlySourceSlowEMATerminalRejectSourceAxisRescueFU": ("slow_state", "early100_h800_slow_ema_with_terminal_reject_source_axis_rescue", 1, 0, 0, 0, 0),
        "M159-EarlySourceSlowEMATerminalRejectSlowEMARescueFU": ("slow_state", "early100_h800_slow_ema_with_terminal_reject_slow_ema_rescue", 1, 0, 0, 0, 0),
        "M160-EarlySourceSlowEMATerminalRejectHoldSourceRescueFU": ("slow_state", "early100_h800_slow_ema_with_terminal_reject_hold_source_rescue", 1, 0, 0, 0, 0),
        "M161-EarlySourceSlowEMAShapePreserveFU": ("slow_state", "early100_h800_slow_ema_midlate_source_shape_preserve", 1, 0, 0, 0, 0),
        "M162-EarlySourceSlowEMAShapePreserveRawGuardFU": ("slow_state", "early100_h800_slow_ema_midlate_shape_preserve_with_raw_guard", 1, 0, 0, 0, 0),
        "M163-EarlySourceSlowEMAShapePreserveClampFU": ("slow_state", "early100_h800_slow_ema_midlate_shape_preserve_with_terminal_clamp", 1, 0, 0, 0, 0),
        "M164-EarlySourceSlowEMATerminalRejectRawRescueFU": ("slow_state", "early100_h800_slow_ema_terminal_reject_raw_rescue", 1, 0, 0, 0, 0),
        "M165-EarlySourceSlowEMATerminalRejectHybridRescueFU": ("slow_state", "early100_h800_slow_ema_terminal_reject_raw_source_hybrid_rescue", 1, 0, 0, 0, 0),
        "M166-EarlySourceSlowEMATerminalRejectLateOnlyRawRescueFU": ("slow_state", "early100_h800_slow_ema_terminal_reject_lateonly_raw_rescue", 1, 0, 0, 0, 0),
        "M167-EarlySourceSlowEMATerminalTopKSupportGuardFU": ("slow_state", "early100_h800_slow_ema_terminal_topk_support_guard", 1, 0, 0, 0, 0),
        "M168-EarlySourceSlowEMATerminalTopKRawGuardFU": ("slow_state", "early100_h800_slow_ema_terminal_topk_raw_guard", 1, 0, 0, 0, 0),
        "M169-EarlySourceSlowEMATerminalTopKDebtCapFU": ("slow_state", "early100_h800_slow_ema_terminal_topk_debt_cap", 1, 0, 0, 0, 0),
        "M170-EarlySourceSlowEMATerminalSourcePreserveStrongFU": ("slow_state", "early100_h800_slow_ema_terminal_source_preserve_strong", 1, 0, 0, 0, 0),
        "M171-EarlySourceSlowEMATerminalSourcePreserveGentleFU": ("slow_state", "early100_h800_slow_ema_terminal_source_preserve_gentle", 1, 0, 0, 0, 0),
        "M172-EarlySourceSlowEMATerminalInfoVolumeGuardFU": ("slow_state", "early100_h800_slow_ema_terminal_info_volume_guard", 1, 0, 0, 0, 0),
        "M173-EarlySourceSlowEMALowNDSDiffeomorphicTargetFU": ("slow_state+function", "early100_h800_slow_ema_with_low_nds_diffeomorphic_target_gate", 1, 0, 0, 1, 0),
        "M174-EarlySourceSlowEMAInfoVolumeDiffeomorphicTargetFU": ("slow_state+function", "early100_h800_slow_ema_with_info_volume_diffeomorphic_target_gate", 1, 0, 0, 1, 0),
        "M175-EarlySourceSlowEMALowRankReadoutTransportFU": ("slow_state+function", "early100_h800_slow_ema_with_low_rank_readout_transport_target_gate", 1, 0, 0, 1, 0),
        "M179-EarlySourceSlowEMATerminalSourcePreserveVeryStrongFU": ("slow_state", "early100_h800_slow_ema_terminal_source_preserve_very_strong", 1, 0, 0, 0, 0),
        "M180-EarlySourceSlowEMAH3600TerminalSourcePreserveFU": ("slow_state", "early100_h800_slow_ema_h3600_terminal_source_preserve", 1, 0, 0, 0, 0),
        "M181-EarlySourceSlowEMATerminalNoraOrthogonalSourceFU": ("slow_state", "early100_h800_slow_ema_terminal_nora_row_orthogonal_source", 1, 0, 0, 0, 0),
        "M182-EarlySourceSlowEMATerminalDebtAwarePreserveFU": ("slow_state", "early100_h800_slow_ema_terminal_debt_aware_preserve", 1, 0, 0, 0, 0),
        "M183-EarlySourceSlowEMATerminalLowNDSMatrixBlockFU": ("slow_state", "early100_h800_slow_ema_terminal_low_nds_matrix_block", 1, 0, 0, 0, 0),
        "M184-EarlySourceSlowEMATerminalDualMemoryPreserveFU": ("slow_state", "early100_h800_slow_ema_terminal_dual_memory_preserve", 1, 0, 0, 0, 0),
        "M218-EarlySourceSlowEMATerminalProjectedOptimizerLambda050FU": ("slow_state", "early100_h800_slow_ema_terminal_projected_optimizer_lambda050", 1, 0, 0, 0, 0),
        "M219-EarlySourceSlowEMATerminalProjectedOptimizerLambda100FU": ("slow_state", "early100_h800_slow_ema_terminal_projected_optimizer_lambda100", 1, 0, 0, 0, 0),
        "M185-EarlySourceSlowEMASNRTerminalPredictorFU": ("slow_state+signal_estimator", "early100_h800_slow_ema_terminal_snr_predictor", 1, 0, 1, 0, 0),
        "M186-EarlySourceSlowEMASplitConsensusEstimatorFU": ("slow_state+split_consensus", "early100_h800_slow_ema_terminal_split_consensus_estimator", 1, 0, 1, 0, 0),
        "M187-EarlySourceSlowEMASignalReservoirTransportFU": ("slow_state+signal_reservoir", "early100_h800_slow_ema_terminal_signal_reservoir_transport", 1, 0, 1, 0, 0),
        "M188-EarlySourceSlowEMATerminalSourceFloorFU": ("slow_state", "early100_h800_slow_ema_terminal_source_floor", 1, 0, 0, 0, 0),
        "M189-EarlySourceSlowEMATerminalH4000AnchorFloorFU": ("slow_state", "early100_h800_slow_ema_terminal_h4000_anchor_floor", 1, 0, 0, 0, 0),
        "M190-EarlySourceSlowEMATerminalDecayAwareFloorFU": ("slow_state", "early100_h800_slow_ema_terminal_decay_aware_floor", 1, 0, 0, 0, 0),
        "M191-EarlySourceSlowEMATerminalRawGuardSourceFloorFU": ("slow_state", "early100_h800_slow_ema_terminal_raw_guard_source_floor", 1, 0, 0, 0, 0),
        "M192-EarlySourceSlowEMATerminalRawGuardH4000AnchorFloorFU": ("slow_state", "early100_h800_slow_ema_terminal_raw_guard_h4000_anchor_floor", 1, 0, 0, 0, 0),
        "M193-EarlySourceSlowEMATerminalRawGuardDecayAwareFloorFU": ("slow_state", "early100_h800_slow_ema_terminal_raw_guard_decay_aware_floor", 1, 0, 0, 0, 0),
        "M194-EarlySourceSlowEMATerminalAntiErosionOrthogonalFU": ("slow_state", "early100_h800_slow_ema_terminal_anti_erosion_orthogonal", 1, 0, 0, 0, 0),
        "M195-EarlySourceSlowEMATerminalSourceReflectionGuardFU": ("slow_state", "early100_h800_slow_ema_terminal_source_reflection_guard", 1, 0, 0, 0, 0),
        "M196-EarlySourceSlowEMATerminalH4000TransportCorrectorFU": ("slow_state", "early100_h800_slow_ema_terminal_h4000_transport_corrector", 1, 0, 0, 0, 0),
        "M197-EarlySourceSlowEMATerminalH3200AnchorTransportFU": ("slow_state", "early100_h800_slow_ema_terminal_h3200_anchor_transport", 1, 0, 0, 0, 0),
        "M198-EarlySourceSlowEMATerminalRawGuardH3200AnchorTransportFU": ("slow_state", "early100_h800_slow_ema_terminal_raw_guard_h3200_anchor_transport", 1, 0, 0, 0, 0),
        "M199-EarlySourceSlowEMATerminalH3200RatioReentryFU": ("slow_state", "early100_h800_slow_ema_terminal_h3200_ratio_reentry", 1, 0, 0, 0, 0),
        "M200-EarlySourceSlowEMATerminalH3200ProgressCarryFU": ("slow_state", "early100_h800_slow_ema_terminal_h3200_progress_carry", 1, 0, 0, 0, 0),
        "M201-EarlySourceSlowEMATerminalH4000ProgressCarryFU": ("slow_state", "early100_h800_slow_ema_terminal_h4000_progress_carry", 1, 0, 0, 0, 0),
        "M202-EarlySourceSlowEMATerminalH3200SourceProgressBlendFU": ("slow_state", "early100_h800_slow_ema_terminal_h3200_source_progress_blend", 1, 0, 0, 0, 0),
        "M203-EarlySourceSlowEMATerminalRawGuardH3600GentleProgressFU": ("slow_state", "early100_h800_slow_ema_terminal_raw_guard_h3600_gentle_progress", 1, 0, 0, 0, 0),
        "M204-EarlySourceSlowEMATerminalRawGuardH4000GentleProgressFU": ("slow_state", "early100_h800_slow_ema_terminal_raw_guard_h4000_gentle_progress", 1, 0, 0, 0, 0),
        "M205-EarlySourceSlowEMATerminalRawGuardH4400ProjectedProgressFU": ("slow_state", "early100_h800_slow_ema_terminal_raw_guard_h4400_projected_progress", 1, 0, 0, 0, 0),
        "M206-EarlySourceSlowEMATerminalControlRelativeSGDCatchupFU": ("slow_state+control_relative", "early100_h800_slow_ema_terminal_control_relative_sgd_catchup", 1, 0, 0, 0, 0),
        "M207-EarlySourceSlowEMATerminalControlRelativeAdamWCatchupFU": ("slow_state+control_relative", "early100_h800_slow_ema_terminal_control_relative_adamw_catchup", 1, 0, 0, 0, 0),
        "M208-EarlySourceSlowEMATerminalControlRelativeSourceBalancedCatchupFU": ("slow_state+control_relative", "early100_h800_slow_ema_terminal_control_relative_source_balanced_catchup", 1, 0, 0, 0, 0),
        "M209-EarlySourceSlowEMATerminalTrajectoryAdaptivePreserveFU": ("slow_state+trajectory", "early100_h800_slow_ema_terminal_trajectory_adaptive_preserve", 1, 0, 0, 0, 0),
        "M210-EarlySourceSlowEMATerminalMidErosionBridgeFU": ("slow_state+trajectory", "early100_h800_slow_ema_terminal_mid_erosion_bridge", 1, 0, 0, 0, 0),
        "M211-EarlySourceSlowEMATerminalTwoPhaseRatioRepairFU": ("slow_state+trajectory", "early100_h800_slow_ema_terminal_two_phase_ratio_repair", 1, 0, 0, 0, 0),
        "M212-EarlySourceSlowEMATerminalAntiSourceClipFU": ("slow_state+terminal_clip", "early100_h800_slow_ema_terminal_anti_source_clip", 1, 0, 0, 0, 0),
        "M213-EarlySourceSlowEMATerminalDebtAwareHoldFU": ("slow_state+terminal_hold", "early100_h800_slow_ema_terminal_debt_aware_hold", 1, 0, 0, 0, 0),
        "M214-EarlySourceSlowEMATerminalAnchorFlowTinyFU": ("slow_state+terminal_flow", "early100_h800_slow_ema_terminal_anchor_flow_tiny", 1, 0, 0, 0, 0),
        "M215-EarlySourceSlowEMATerminalAcceptMemoryFU": ("slow_state+terminal_memory", "early100_h800_slow_ema_terminal_accept_memory", 1, 0, 0, 0, 0),
        "M216-EarlySourceSlowEMATerminalAcceptMemoryDebtFU": ("slow_state+terminal_memory", "early100_h800_slow_ema_terminal_accept_memory_debt", 1, 0, 0, 0, 0),
        "M217-EarlySourceSlowEMATerminalSourceProgressMemoryFU": ("slow_state+terminal_memory", "early100_h800_slow_ema_terminal_source_progress_memory", 1, 0, 0, 0, 0),
        "M176-LowNDSDiffeomorphicTargetOnlyFU": ("function", "SGD+low_nds_diffeomorphic_readout_target", 0, 0, 0, 1, 0),
        "M177-InfoVolumeDiffeomorphicTargetOnlyFU": ("function", "SGD+info_volume_diffeomorphic_consensus_target", 0, 0, 0, 1, 0),
        "M178-LowRankReadoutTransportTargetOnlyFU": ("function", "SGD+low_rank_view_consistent_readout_transport_target", 0, 0, 0, 1, 0),
        "M220-MetricFirstG0L2FU": ("function_metric", "metric_first_G0_L2_parameter_pullback", 0, 0, 0, 1, 0),
        "M221-MetricFirstDiagFisherFU": ("function_metric", "metric_first_G1_diag_fisher_parameter_pullback", 0, 0, 0, 1, 0),
        "M222-MetricFirstPopRiskDiagFU": ("function_metric", "metric_first_G2_poprisk_diag_parameter_pullback", 0, 0, 1, 1, 0),
        "M223-MetricFirstSobolevH1FU": ("function_metric", "metric_first_G3_sobolev_h1_parameter_pullback", 0, 0, 0, 1, 0),
        "M224-MetricFirstRKHSKNNFU": ("function_metric", "metric_first_G4_rkhs_knn_parameter_pullback", 0, 0, 0, 1, 0),
        "M225-MetricFirstFisherRKHSFU": ("function_metric", "metric_first_G5_fisher_rkhs_parameter_pullback", 0, 0, 0, 1, 0),
        "M226-MetricFirstLowNDSFU": ("function_metric", "metric_first_G6_low_nds_parameter_pullback", 0, 0, 0, 1, 0),
        "M227-MetricFirstEnsembleFU": ("function_metric", "metric_first_G8_metric_ensemble_parameter_pullback", 0, 0, 1, 1, 0),
        "M228-MetricSourceMemoryG0L2FU": ("slow_state+function_metric", "metric_source_memory_G0_L2_parameter_pullback", 1, 0, 0, 1, 0),
        "M229-MetricSourceMemoryDiagFisherFU": ("slow_state+function_metric", "metric_source_memory_G1_diag_fisher_parameter_pullback", 1, 0, 0, 1, 0),
        "M230-MetricSourceMemoryPopRiskDiagFU": ("slow_state+function_metric", "metric_source_memory_G2_poprisk_diag_parameter_pullback", 1, 0, 1, 1, 0),
        "M231-MetricSourceMemorySobolevH1FU": ("slow_state+function_metric", "metric_source_memory_G3_sobolev_h1_parameter_pullback", 1, 0, 0, 1, 0),
        "M232-MetricSourceMemoryRKHSKNNFU": ("slow_state+function_metric", "metric_source_memory_G4_rkhs_knn_parameter_pullback", 1, 0, 0, 1, 0),
        "M233-MetricSourceMemoryFisherRKHSFU": ("slow_state+function_metric", "metric_source_memory_G5_fisher_rkhs_parameter_pullback", 1, 0, 0, 1, 0),
        "M234-MetricSourceMemoryLowNDSFU": ("slow_state+function_metric", "metric_source_memory_G6_low_nds_parameter_pullback", 1, 0, 0, 1, 0),
        "M235-MetricSourceMemoryEnsembleFU": ("slow_state+function_metric", "metric_source_memory_G8_metric_ensemble_parameter_pullback", 1, 0, 1, 1, 0),
        "M236-MetricTargetLossCotangentFU": ("function_metric_target", "metric_target_T0_loss_cotangent_readout", 0, 0, 0, 1, 0),
        "M237-MetricTargetLowDegreeReadoutFU": ("function_metric_target", "metric_target_T3_low_degree_readout", 0, 0, 0, 1, 0),
        "M238-MetricTargetB1TransferFU": ("function_metric_target", "metric_target_T4_b1_loss_transfer", 0, 0, 0, 1, 0),
        "M239-MetricTargetSourceProjectedB3NullFU": ("function_metric_target", "metric_target_T5_source_projected_b3_null", 0, 0, 0, 1, 0),
        "M240-HC2H3200NoProjectionFU": ("slow_state+optimizer_projection", "hc2_h3200_source_no_projection_baseline", 1, 0, 0, 0, 0),
        "M241-HC2H3200L2ProjectionFU": ("slow_state+optimizer_projection", "hc2_h3200_source_l2_destructive_projection", 1, 0, 0, 0, 0),
        "M242-HC2H3200HalfProjectionFU": ("slow_state+optimizer_projection", "hc2_h3200_source_half_destructive_projection", 1, 0, 0, 0, 0),
        "M243-HC2H4000L2RecomputeProjectionFU": ("slow_state+optimizer_projection", "hc2_h4000_recomputed_source_l2_destructive_projection", 1, 0, 0, 0, 0),
        "M244-HC2H4000HalfRecomputeProjectionFU": ("slow_state+optimizer_projection", "hc2_h4000_recomputed_source_half_destructive_projection", 1, 0, 0, 0, 0),
        "M245-MetricTargetSplitConsensusFU": ("function_metric_target", "metric_target_T1_split_consensus", 0, 0, 0, 1, 0),
        "M246-MetricTargetPopRiskSNRFU": ("function_metric_target", "metric_target_T2_poprisk_snr", 0, 0, 1, 1, 0),
        "M247-MetricTargetDiffeomorphicNoFoldFU": ("function_metric_target", "metric_target_T6_view_consistent_no_fold", 0, 0, 0, 1, 0),
        "M248-MetricTargetRandomMatchedFU": ("function_metric_control", "metric_target_T7_random_matched_actuation_control", 0, 0, 0, 1, 0),
        "M249-MetricTargetSignFlippedFU": ("function_metric_control", "metric_target_T8_sign_flipped_control", 0, 0, 0, 1, 0),
        "M250-MetricTargetDebtCalibratedSourceFU": ("function_metric_target", "metric_target_T9_debt_calibrated_source", 0, 0, 0, 1, 0),
        "M251-MetricTargetObservableMidDebtSourceFU": ("function_metric_target", "metric_target_T10_observable_mid_debt_source", 0, 0, 0, 1, 0),
        "M252-MetricTargetObservableMidDebtM181BridgeFU": ("slow_state+function_metric_target", "metric_target_T10_observable_mid_debt_M181_preheated_bridge", 1, 0, 0, 1, 0),
        "M253-MetricTargetGradientObservableMidDebtFU": ("function_metric_target", "metric_target_T11_gradient_observable_mid_debt_source", 0, 0, 0, 1, 0),
        "M254-HC8PreH3200SourceChannelAntiWashoutFU": ("slow_state+function_metric_target+optimizer_projection", "metric_target_T12_pre_h3200_source_channel_antiwashout", 1, 0, 0, 1, 0),
        "M255-HC9RiskWeightedSplitConsensusSourceFU": ("slow_state+function_metric_target+poprisk_snr+optimizer_projection", "metric_target_T13_risk_weighted_split_consensus_source", 1, 0, 1, 1, 0),
        "M256-HC10ViewConsistentSourceCarryFU": ("slow_state+function_metric_target+view_consistency+optimizer_projection", "metric_target_T14_view_consistent_source_carry", 1, 0, 0, 1, 0),
        "M257-HC11ContinuousSourceCarryAntiWashoutFU": ("slow_state+function_metric_target+optimizer_projection+continuous_source_carry", "metric_target_T15_continuous_source_carry_antiwashout", 1, 0, 0, 1, 0),
        "M258-HC12NoiseOrthogonalSourceCarryFU": ("slow_state+function_metric_target+noise_orthogonal+optimizer_projection+continuous_source_carry", "metric_target_T16_noise_orthogonal_source_carry", 1, 0, 0, 1, 0),
        "M259-V2206MetricSolverT0G0ReadoutFU": ("function_metric_solver", "v22_06_T0_G0_readout_exact_metric_solver", 0, 0, 0, 1, 0),
        "M260-V2206MetricSolverT1G0SplitTransferFU": ("function_metric_solver", "v22_06_T1_G0_split_transfer_readout_exact_metric_solver", 0, 0, 0, 1, 0),
        "M261-V2206MetricSolverT3G3SignalSobolevFU": ("function_metric_solver", "v22_06_T3_G3_signal_sobolev_readout_exact_metric_solver", 0, 0, 0, 1, 0),
        "M262-V2206MetricSolverT4G3SmoothManifoldFU": ("function_metric_solver", "v22_06_T4_G3_smooth_manifold_readout_exact_metric_solver", 0, 0, 0, 1, 0),
        "M263-V2206MetricSolverT5G6LowNDSFU": ("function_metric_solver", "v22_06_T5_G6_low_nds_readout_exact_metric_solver", 0, 0, 0, 1, 0),
        "M264-V2206MetricSolverT6G0DualMemoryFU": ("function_metric_solver", "v22_06_T6_G0_dual_memory_source_readout_exact_metric_solver", 0, 0, 0, 1, 0),
        "M265-V2206MetricSolverT7G0HiddenBlockFU": ("function_metric_solver+matrix_block", "v22_06_T7_G0_hidden_readout_block_source_metric_solver", 0, 1, 0, 1, 0),
        "M266-V2206MetricSolverT8G0AdaptiveHiddenBlockFU": ("function_metric_solver+matrix_block", "v22_06_T8_G0_adaptive_hidden_readout_block_source_metric_solver", 0, 1, 0, 1, 0),
        "M267-V2206MetricSolverT9G0C3GatedHiddenBlockFU": ("function_metric_solver+matrix_block", "v22_06_T9_G0_c3_gated_hidden_readout_block_source_metric_solver", 0, 1, 0, 1, 0),
        "M268-V2206MetricSolverT10G0CompensatedHiddenBlockFU": ("function_metric_solver+matrix_block", "v22_06_T10_G0_compensated_hidden_readout_block_source_metric_solver", 0, 1, 0, 1, 0),
        "M269-V2206MetricSolverT11G0EarlyObservableFU": ("function_metric_solver+matrix_block", "v22_06_T11_G0_early_observable_source_channel_metric_solver", 0, 1, 0, 1, 0),
        "M270-V2206MetricSolverT12G0SoftCompensatedHiddenBlockFU": ("function_metric_solver+matrix_block", "v22_06_T12_G0_soft_compensated_hidden_readout_block_source_metric_solver", 0, 1, 0, 1, 0),
        "M271-V2206MetricSolverT12G0HiddenOnlySoftCompensatedFU": ("function_metric_solver+matrix_block", "v22_06_T12_G0_hidden_only_soft_compensated_source_metric_solver", 0, 1, 0, 1, 0),
    }
    terminal_alias = {
        "M170-EarlySourceSlowEMATerminalSourcePreserveStrongFU",
        "M171-EarlySourceSlowEMATerminalSourcePreserveGentleFU",
        "M172-EarlySourceSlowEMATerminalInfoVolumeGuardFU",
        "M179-EarlySourceSlowEMATerminalSourcePreserveVeryStrongFU",
        "M180-EarlySourceSlowEMAH3600TerminalSourcePreserveFU",
        "M181-EarlySourceSlowEMATerminalNoraOrthogonalSourceFU",
        "M182-EarlySourceSlowEMATerminalDebtAwarePreserveFU",
        "M183-EarlySourceSlowEMATerminalLowNDSMatrixBlockFU",
        "M184-EarlySourceSlowEMATerminalDualMemoryPreserveFU",
        "M188-EarlySourceSlowEMATerminalSourceFloorFU",
        "M189-EarlySourceSlowEMATerminalH4000AnchorFloorFU",
        "M190-EarlySourceSlowEMATerminalDecayAwareFloorFU",
        "M191-EarlySourceSlowEMATerminalRawGuardSourceFloorFU",
        "M192-EarlySourceSlowEMATerminalRawGuardH4000AnchorFloorFU",
        "M193-EarlySourceSlowEMATerminalRawGuardDecayAwareFloorFU",
        "M194-EarlySourceSlowEMATerminalAntiErosionOrthogonalFU",
        "M195-EarlySourceSlowEMATerminalSourceReflectionGuardFU",
        "M196-EarlySourceSlowEMATerminalH4000TransportCorrectorFU",
        "M197-EarlySourceSlowEMATerminalH3200AnchorTransportFU",
        "M198-EarlySourceSlowEMATerminalRawGuardH3200AnchorTransportFU",
        "M199-EarlySourceSlowEMATerminalH3200RatioReentryFU",
        "M200-EarlySourceSlowEMATerminalH3200ProgressCarryFU",
        "M201-EarlySourceSlowEMATerminalH4000ProgressCarryFU",
        "M202-EarlySourceSlowEMATerminalH3200SourceProgressBlendFU",
        "M203-EarlySourceSlowEMATerminalRawGuardH3600GentleProgressFU",
        "M204-EarlySourceSlowEMATerminalRawGuardH4000GentleProgressFU",
        "M205-EarlySourceSlowEMATerminalRawGuardH4400ProjectedProgressFU",
        "M206-EarlySourceSlowEMATerminalControlRelativeSGDCatchupFU",
        "M207-EarlySourceSlowEMATerminalControlRelativeAdamWCatchupFU",
        "M208-EarlySourceSlowEMATerminalControlRelativeSourceBalancedCatchupFU",
        "M209-EarlySourceSlowEMATerminalTrajectoryAdaptivePreserveFU",
        "M210-EarlySourceSlowEMATerminalMidErosionBridgeFU",
        "M211-EarlySourceSlowEMATerminalTwoPhaseRatioRepairFU",
        "M212-EarlySourceSlowEMATerminalAntiSourceClipFU",
        "M213-EarlySourceSlowEMATerminalDebtAwareHoldFU",
        "M214-EarlySourceSlowEMATerminalAnchorFlowTinyFU",
        "M215-EarlySourceSlowEMATerminalAcceptMemoryFU",
        "M216-EarlySourceSlowEMATerminalAcceptMemoryDebtFU",
        "M217-EarlySourceSlowEMATerminalSourceProgressMemoryFU",
        "M218-EarlySourceSlowEMATerminalProjectedOptimizerLambda050FU",
        "M219-EarlySourceSlowEMATerminalProjectedOptimizerLambda100FU",
    }
    rows: list[dict[str, object]] = []
    for mech in MECHANISMS:
        source_space, source_pattern, slow, block, snr, function, proto = specs[mech]
        metric_name = METRIC_FIRST_MECHANISM_TO_METRIC.get(mech, "")
        family = source_space
        alias_group = ""
        representative = 1
        if mech in terminal_alias:
            alias_group = "v22_04_terminal_momentum_like_alias"
            representative = int(mech in {"M181-EarlySourceSlowEMATerminalNoraOrthogonalSourceFU", "M218-EarlySourceSlowEMATerminalProjectedOptimizerLambda050FU"})
        elif mech in HC2_PROJECTION_MECHANISMS:
            alias_group = "v22_05_metric_first_G0-L2"
            representative = 0
        elif mech in {
            "M264-V2206MetricSolverT6G0DualMemoryFU",
            "M267-V2206MetricSolverT9G0C3GatedHiddenBlockFU",
        }:
            alias_group = "v22_06_T6_T9_c3_gate_fallback_alias"
            representative = int(mech == "M264-V2206MetricSolverT6G0DualMemoryFU")
        elif mech in {
            "M268-V2206MetricSolverT10G0CompensatedHiddenBlockFU",
            "M270-V2206MetricSolverT12G0SoftCompensatedHiddenBlockFU",
        }:
            alias_group = "v22_06_T10_T12_compensation_selection_alias_if_tiny_equivalent"
            representative = int(mech == "M268-V2206MetricSolverT10G0CompensatedHiddenBlockFU")
        elif metric_name:
            alias_group = f"v22_05_metric_first_{metric_name}"
        rows.append(
            {
                "mechanism": mech,
                "mechanism_family": family,
                "source_pattern": source_pattern,
                "uses_optimizer_primary": int(str(source_pattern).lower() in {"adamw", "sgd", "momentum"} or str(source_pattern).startswith(("AdamW", "SGD", "Momentum"))),
                "uses_slow_state": int(slow),
                "uses_matrix_block": int(block),
                "uses_function_space_metric": int(bool(metric_name) or bool(function)),
                "uses_sobolev_metric": int(metric_name == "G3-SobolevH1-hidden"),
                "uses_rkhs_metric": int(metric_name in {"G4-RKHS-KNN", "G5-Fisher-RKHS"}),
                "uses_fisher_metric": int(metric_name in {"G1-DiagFisher", "G5-Fisher-RKHS"} or "fisher" in str(source_pattern).lower()),
                "uses_basis_channel_metric": int("basis" in str(source_space).lower() or "degree" in str(source_pattern).lower() or "frequency" in str(source_pattern).lower()),
                "implementation_is_prototype": int(proto),
                "semantic_contract_declared": 1,
                "semantic_noncollapse_group": alias_group,
                "semantic_alias_representative": representative,
                "metric_name": metric_name,
                # Backward-compatible columns consumed by older truth gates.
                "source_space": source_space,
                "uses_poprisk_snr": int(snr),
                "uses_function_space_operator": int(function),
                "smoke_only_if_prototype": int(proto),
            }
        )
    return rows
