# DG-KAN v16.3 MultiSchemeFunctionalDynamics EfficiencyCensus 4GPU 执行日志

生成时间：2026-06-02 01:41:17（Asia/Singapore）

## 1. 文件 / 输出目录

- plan: /home/chengshun.wang/DG-LCA/docs/DG-KAN_v16.3_MultiSchemeFunctionalDynamics_EfficiencyCensus_4GPU_完整计划.md
- runner: /home/chengshun.wang/DG-LCA/experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py
- v16.2 shared runner: /home/chengshun.wang/DG-LCA/experiments/run_v162_multischeme_functional_dynamics_4gpu.py
- v16.0 shared kernels: /home/chengshun.wang/DG-LCA/experiments/run_v160_dynamic_geometry_debt_recovery_multiline_fu.py
- output dir: results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163
- line F out dir: results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/line_f_v163_allbasis_substrate

## 2. Repro commands

Compile:
```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py experiments/run_v162_multischeme_functional_dynamics_4gpu.py experiments/run_v149_line_d_all_basis_substrate_repair.py
```

Line P:
```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py --out-dir results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163 --line-f-out results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/line_f_v163_allbasis_substrate --run-lines P --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --batch-size 8 --train-steps 120 --horizon-steps 800 --h1600-steps 1600 --rational-steps 80 --lq-steps 80 --split-count 4 --hidden 16 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --efficiency-repeats 3 --efficiency-warmup 1
```

Line P timing repair rerun used for final artifact:
```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py --out-dir results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163 --line-f-out results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/line_f_v163_allbasis_substrate --run-lines P --device cuda:3 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --batch-size 8 --train-steps 120 --horizon-steps 800 --h1600-steps 1600 --rational-steps 80 --lq-steps 80 --split-count 4 --hidden 16 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --efficiency-repeats 3 --efficiency-warmup 1
```

A/B/C/D four-GPU shards:
```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py --out-dir results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163 --line-f-out results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/line_f_v163_allbasis_substrate --run-lines A --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --batch-size 8 --train-steps 120 --horizon-steps 800 --h1600-steps 1600 --rational-steps 80 --lq-steps 80 --split-count 4 --hidden 16 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --efficiency-repeats 3 --efficiency-warmup 1 --artifact-suffix a0 --line-a-methods A-M1a-D-CHE-PulseOnce-AdamWRecovery,A-M1b-D-CHE-PulseOnce-LRCooldownRecovery,A-M1c-D-CHE-PulseOnce-MomentumDampingRecovery,A-M1d-D-CHE-PulseOnce-DecoupledGlobalDecayRecovery,A-M1e-D-CHE-PulseOnce-DegreeWiseDecayRecovery,A-M1f-D-CHE-PulseOnce-EMAConsolidation,A-M1g-D-CHE-PulseOnce-LookaheadConsolidation,A-M1h-D-CHE-PulseOnce-SWAConsolidation,A-M1i-D-CHE-PulseEvery50-AdamWRecovery,A-M1j-D-CHE-EarlyOnlyPulse-AdamWRecovery,A-M1k-D-CHE-MidOnlyPulse-AdamWRecovery,A-M1l-D-CHE-LateOnlyPulse-AdamWRecovery,A-M2a-D-CHE-SplitConsensusDiagMetric,A-M2b-D-CHE-SplitConsensusRoleBlockMetric,A-M2c-D-CHE-SplitConsensusLowRank-r4,A-M2d-D-CHE-SplitConsensusLowRank-r8
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py --out-dir results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163 --line-f-out results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/line_f_v163_allbasis_substrate --run-lines A --device cuda:1 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --batch-size 8 --train-steps 120 --horizon-steps 800 --h1600-steps 1600 --rational-steps 80 --lq-steps 80 --split-count 4 --hidden 16 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --efficiency-repeats 3 --efficiency-warmup 1 --artifact-suffix a1 --line-a-methods A-M2e-D-CHE-SplitConsensusMetricOnlyNoProjection,A-M2f-D-CHE-SplitConsensusProjectionPlusAdamV,A-M2g-D-CHE-NegativeEigenQuarantineDiagnostic,A-M3a-D-CHE-ParameterSNRPreconditioner,A-M3b-D-CHE-DegreeRoleSNRPreconditioner,A-M3c-D-CHE-BasisChannelSNRCoverBoundary,A-M3d-D-CHE-SNRPulseScheduler,A-M3e-D-CHE-SNRRecoveryScheduler,A-M3f-D-CHE-SNRReservoirGuard,A-M4a-D-CHE-AdamSubspaceProximal-alpha000,A-M4b-D-CHE-PerExampleLowRankProximal-alpha025,A-M4c-D-CHE-OutputJacobianSketchProximal-alpha050,A-M4d-D-CHE-SplitB1B2TransferProximal-alpha100,A-M4e-D-CHE-RecoveryAwareProximal-alpha200,A-M4f-D-CHE-ControlResidualizedProximal-alpha100,A-M5a-D-CHE-CautiousFU
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py --out-dir results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163 --line-f-out results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/line_f_v163_allbasis_substrate --run-lines A --device cuda:2 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --batch-size 8 --train-steps 120 --horizon-steps 800 --h1600-steps 1600 --rational-steps 80 --lq-steps 80 --split-count 4 --hidden 16 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --efficiency-repeats 3 --efficiency-warmup 1 --artifact-suffix a2 --line-a-methods A-M5b-D-CHE-SoftCautiousFU,A-M5c-D-CHE-MGUPStyleReweightFU,A-M5d-D-CHE-AdamSecondMomentScaledFU,A-M5e-D-CHE-SophiaDiagLiteClippedFU,A-M5f-D-CHE-BlockSecondMomentFU,A-M5g-D-CHE-DecoupledDecayPlusFU,A-M6a-D-CHE-LowDegreeSignalHighDegreeReservoir,A-M6b-D-CHE-ReadoutBasisDecoupledCarrier,A-M6c-D-CHE-OrthogonalDegreeBankCarrier,A-M6d-D-CHE-OutputJacobianCarrier,A-M6e-D-CHE-DualBankSignalReservoirCarrier,A-M6f-D-CHE-HighDegreeQuarantineCarrier,A-M7a-D-CHE-LateAttachEarlyCheckpoint,A-M7b-D-CHE-LateAttachMidCheckpoint,A-M7c-D-CHE-LateAttachLateCheckpoint,A-M7d-D-CHE-LateAttachAfterLossPlateau
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py --out-dir results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163 --line-f-out results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/line_f_v163_allbasis_substrate --run-lines A --device cuda:3 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --batch-size 8 --train-steps 120 --horizon-steps 800 --h1600-steps 1600 --rational-steps 80 --lq-steps 80 --split-count 4 --hidden 16 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --efficiency-repeats 3 --efficiency-warmup 1 --artifact-suffix a3 --line-a-methods A-M7e-D-CHE-LateAttachHighSourceLowDebtWindow,A-M8a-D-CHE-AdamWRecoveryOnly,A-M8b-D-CHE-LRCooldownOnly,A-M8c-D-CHE-DecayRecoveryOnly,A-M8d-D-CHE-EMASWARecoveryOnly,A-M8e-D-CHE-LookaheadRecoveryOnly,A-M8f-D-CHE-MomentumDampingOnly,ACTRL0-D-CHE-AdamW,ACTRL1-D-CHE-NoOpMatchedOverhead,ACTRL2-D-CHE-RandomMatchedPulse,ACTRL3-D-CHE-AdamWExtraStepsMatchedTime,ACTRL4-D-CHE-RandomSubspaceSameRank,ACTRL5-D-CHE-SameActiveFractionRandomMask,ACTRL6-D-CHE-RecoveryOnlyNoPulse,ACTRL7-D-CHE-DecayOnlyRecovery
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py --out-dir results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163 --line-f-out results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/line_f_v163_allbasis_substrate --run-lines B --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --batch-size 8 --train-steps 120 --horizon-steps 800 --h1600-steps 1600 --rational-steps 80 --lq-steps 80 --split-count 4 --hidden 16 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --efficiency-repeats 3 --efficiency-warmup 1 --artifact-suffix b0 --line-b-methods B-M1a-MLP-PulseOnce-AdamWRecovery,B-M1b-MLP-PulseOnce-LRCooldownRecovery,B-M1c-MLP-PulseOnce-MomentumDampingRecovery,B-M1d-MLP-PulseOnce-DecoupledDecayRecovery,B-M1e-MLP-PulseOnce-FeatureRecenterRecovery,B-M1f-MLP-PulseOnce-EMAConsolidation,B-M1g-MLP-PulseOnce-LookaheadConsolidation,B-M1h-MLP-PulseOnce-SWAConsolidation,B-M1i-MLP-PulseEvery50-AdamWRecovery,B-M1j-MLP-EarlyOnlyPulse-AdamWRecovery,B-M1k-MLP-MidOnlyPulse-AdamWRecovery,B-M1l-MLP-LateOnlyPulse-AdamWRecovery,B-M2a-MLP-HiddenSplitConsensusDiagMetric,B-M2b-MLP-HiddenSplitConsensusRoleBlockMetric,B-M2c-MLP-HiddenSplitConsensusLowRank-r4,B-M2d-MLP-HiddenSplitConsensusLowRank-r8
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py --out-dir results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163 --line-f-out results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/line_f_v163_allbasis_substrate --run-lines B --device cuda:1 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --batch-size 8 --train-steps 120 --horizon-steps 800 --h1600-steps 1600 --rational-steps 80 --lq-steps 80 --split-count 4 --hidden 16 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --efficiency-repeats 3 --efficiency-warmup 1 --artifact-suffix b1 --line-b-methods B-M2e-MLP-HiddenMetricOnlyNoProjection,B-M2f-MLP-HiddenProjectionPlusAdamV,B-M2g-MLP-HiddenNegativeEigenQuarantineDiagnostic,B-M3a-MLP-ParameterSNRPreconditioner,B-M3b-MLP-HiddenRoleSNRPreconditioner,B-M3c-MLP-FeatureChannelSNRCoverBoundary,B-M3d-MLP-SNRPulseScheduler,B-M3e-MLP-SNRRecoveryScheduler,B-M3f-MLP-SNRReservoirGuard,B-M4a-MLP-AdamSubspaceProximal-alpha000,B-M4b-MLP-PerExampleLowRankProximal-alpha025,B-M4c-MLP-OutputJacobianSketchProximal-alpha050,B-M4d-MLP-SplitB1B2TransferProximal-alpha100,B-M4e-MLP-RecoveryAwareProximal-alpha200,B-M4f-MLP-ControlResidualizedProximal-alpha100,B-M5a-MLP-CautiousAdamW
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py --out-dir results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163 --line-f-out results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/line_f_v163_allbasis_substrate --run-lines B --device cuda:2 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --batch-size 8 --train-steps 120 --horizon-steps 800 --h1600-steps 1600 --rational-steps 80 --lq-steps 80 --split-count 4 --hidden 16 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --efficiency-repeats 3 --efficiency-warmup 1 --artifact-suffix b2 --line-b-methods B-M5b-MLP-SoftCautiousFU,B-M5c-MLP-MGUPStyleReweight,B-M5d-MLP-AdamSecondMomentScaledFU,B-M5e-MLP-SophiaDiagLiteClippedFU,B-M5f-MLP-BlockSecondMomentFU,B-M5g-MLP-DecoupledDecayPlusFU,B-M6a-MLP-HiddenSubspaceCarrier,B-M6b-MLP-HiddenReadoutDecoupledCarrier,B-M6c-MLP-HiddenOrthogonalBankCarrier,B-M6d-MLP-OutputJacobianCarrier,B-M6e-MLP-DualBankSignalReservoirCarrier,B-M6f-MLP-HiddenCarrierDecay,B-M7a-MLP-LateAttachEarlyCheckpoint,B-M7b-MLP-LateAttachMidCheckpoint,B-M7c-MLP-LateAttachLateCheckpoint,B-M7d-MLP-LateAttachAfterLossPlateau
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py --out-dir results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163 --line-f-out results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/line_f_v163_allbasis_substrate --run-lines B,C,D --device cuda:3 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --batch-size 8 --train-steps 120 --horizon-steps 800 --h1600-steps 1600 --rational-steps 80 --lq-steps 80 --split-count 4 --hidden 16 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --efficiency-repeats 3 --efficiency-warmup 1 --artifact-suffix b3 --line-b-methods B-M7e-MLP-LateAttachHighSourceLowDebtWindow,B-M8a-MLP-AdamWRecoveryOnly,B-M8b-MLP-LRCooldownOnly,B-M8c-MLP-DecayRecoveryOnly,B-M8d-MLP-EMASWARecoveryOnly,B-M8e-MLP-LookaheadRecoveryOnly,B-M8f-MLP-MomentumDampingOnly,BCTRL0-MLP-AdamW,BCTRL1-MLP-NoOpMatchedOverhead,BCTRL2-MLP-RandomMatchedPulse,BCTRL3-MLP-AdamWExtraStepsMatchedTime,BCTRL4-MLP-RandomSubspaceSameRank,BCTRL5-MLP-SameActiveFractionRandomMask,BCTRL6-MLP-RecoveryOnlyNoPulse,BCTRL7-MLP-DecayOnly
```

Line F raw substrate command:
```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v149_line_d_all_basis_substrate_repair.py --out-dir results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/line_f_v163_allbasis_substrate --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --candidates D-FOU97-LowFreqIdentityResidualV8,D-FOU98-BandwiseSNRWarmupV8,D-FOU99-PhaseStableBandMixV8,D-FOU100-NoMaterializeLifetimeV8,D-FOU101-HighFrequencyQuarantineV8,D-FOU102-SplitConsensusLowFreqMetricSmoke,D-RBF95-ActiveCenterOccupancyV8,D-RBF96-WidthConditionGuardV8,D-RBF97-CompactBumpNoDenseV8,D-RBF98-GaussianLocalK4TaskHealthV8,D-RBF99-ActiveCenterSecondMomentV8,D-RBF100-SplitConsensusCenterMetricSmoke,D-WAV81-TriangularSupportV8,D-WAV82-ScaleOccupancyV8,D-WAV83-SupportOverlapDampingV8,D-WAV84-LocalTailCoverageAuditV8,D-WAV85-SplitConsensusScaleMetricSmoke --device cuda:3 --data-root data --no-download --train-size 256 --val-size 128 --batch-size 32 --epochs 1
```

Merge / h1600 / finalize:
```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py --out-dir results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163 --line-f-out results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/line_f_v163_allbasis_substrate --run-lines MERGE_A,MERGE_B,A1600 --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --batch-size 8 --train-steps 120 --horizon-steps 800 --h1600-steps 1600 --rational-steps 80 --lq-steps 80 --split-count 4 --hidden 16 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --efficiency-repeats 3 --efficiency-warmup 1
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py --out-dir results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163 --line-f-out results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/line_f_v163_allbasis_substrate --run-lines B1600 --device cuda:1 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --batch-size 8 --train-steps 120 --horizon-steps 800 --h1600-steps 1600 --rational-steps 80 --lq-steps 80 --split-count 4 --hidden 16 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --efficiency-repeats 3 --efficiency-warmup 1
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py --out-dir results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163 --line-f-out results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/line_f_v163_allbasis_substrate --run-lines F --device cuda:3 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --batch-size 8 --train-steps 120 --horizon-steps 800 --h1600-steps 1600 --rational-steps 80 --lq-steps 80 --split-count 4 --hidden 16 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --efficiency-repeats 3 --efficiency-warmup 1
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py --out-dir results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163 --line-f-out results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/line_f_v163_allbasis_substrate --run-lines FINALIZE --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --batch-size 8 --train-steps 120 --horizon-steps 800 --h1600-steps 1600 --rational-steps 80 --lq-steps 80 --split-count 4 --hidden 16 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --efficiency-repeats 3 --efficiency-warmup 1
```

## 3. GPU assignment manifest

| round | gpu | run lines | suffix | rows | exists | role |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | cuda:0 | P |  | 27 | 1 | MLP + D-CHE efficiency census bootstrap |
| 0 | cuda:1 | P |  | 27 | 1 | MLP functional path efficiency census |
| 0 | cuda:2 | P,C,D |  | 27 | 1 | LQ/Rational monitor and efficiency census |
| 0 | cuda:3 | P,F |  | 27 | 1 | all-basis substrate and efficiency census |
| 1 | cuda:0 | A | a0 | 144 | 1 | D-CHE M1a..M8f/control shard |
| 1 | cuda:1 | A | a1 | 144 | 1 | D-CHE M1a..M8f/control shard |
| 1 | cuda:2 | A | a2 | 144 | 1 | D-CHE M1a..M8f/control shard |
| 1 | cuda:3 | A | a3 | 135 | 1 | D-CHE M1a..M8f/control shard |
| 2 | cuda:0 | B | b0 | 144 | 1 | MLP M1a..M8f/control shard |
| 2 | cuda:1 | B | b1 | 144 | 1 | MLP M1a..M8f/control shard |
| 2 | cuda:2 | B | b2 | 144 | 1 | MLP M1a..M8f/control shard |
| 2 | cuda:3 | B,C,D | b3 | 135 | 1 | MLP M1a..M8f/control shard plus LQ/Rational |
| 3 | cuda:3 | F |  | 153 | 1 | all-basis substrate |
| 4 | cuda:0 | MERGE_A,MERGE_B,A1600 |  | 18 | 1 | merge and D-CHE long horizon |
| 4 | cuda:1 | B1600 |  | 18 | 1 | MLP long horizon |
| 5 | cuda:0 | FINALIZE |  |  | 1 | route/docs/audits |

## 4. Artifact inventory

| artifact | exists | rows | bytes |
| --- | --- | --- | --- |
| v163_route_decision.json | 1 |  | 1391 |
| v163_gpu_assignment_manifest.csv | 1 | 16 | 2317 |
| v163_gpu_utilization_dashboard.csv | 1 | 4 | 194 |
| v163_efficiency_census.csv | 1 | 27 | 21824 |
| v163_efficiency_census_component_breakdown.csv | 1 | 567 | 66711 |
| v163_method_surface_manifest.csv | 1 | 126 | 34782 |
| v163_line_a_dche_results.csv | 1 | 567 | 1306291 |
| v163_line_b_mlp_results.csv | 1 | 567 | 1297586 |
| v163_line_c_lq_reanchor.csv | 1 | 63 | 19642 |
| v163_line_d_rational_monitor.csv | 1 | 45 | 46171 |
| v163_line_f_allbasis_substrate.csv | 1 | 153 | 190494 |
| v163_line_m_attribution.csv | 1 | 55 | 14175 |
| v163_failure_taxonomy.csv | 1 | 1194 | 147299 |
| v163_h800_summary.csv | 1 | 16 | 3244 |
| v163_h1600_summary.csv | 1 | 36 | 82617 |
| v163_required_artifact_manifest.csv | 1 | 40 | 1847 |
| v163_forbidden_information_audit.csv | 1 | 9 | 346 |
| v163_no_action_search_audit.csv | 1 | 5 | 138 |
| v163_direction_provenance.csv | 1 | 5994 | 10286708 |
| v163_budget_exhaustion_certificate.csv | 1 | 6 | 913 |
| v163_deferred_items.csv | 1 | 5 | 542 |
| v163_code_review_packet.zip | 1 |  | 3066398 |
| v163_implementation_readback.md | 1 |  | 2092 |
| basis_vs_mlp_forward_ratio_bar.svg | 1 |  | 5353 |
| basis_vs_mlp_backward_ratio_bar.svg | 1 |  | 5351 |
| basis_vs_mlp_update_ratio_bar.svg | 1 |  | 5348 |
| basis_vs_mlp_backward_memory_ratio_bar.svg | 1 |  | 5360 |
| phase_time_stacked_bar_by_basis.svg | 1 |  | 5345 |
| memory_component_stacked_bar_by_basis.svg | 1 |  | 5383 |
| source_retention_curve_horizon.svg | 1 |  | 3470 |
| tail_debt_curve_horizon.svg | 1 |  | 5952 |
| LineC_debt_curve_horizon.svg | 1 |  | 5946 |
| calibration_debt_curve_horizon.svg | 1 |  | 5936 |
| AUC_debt_curve_horizon.svg | 1 |  | 3462 |
| D-CHE_vs_MLP_mechanism_matrix_heatmap.svg | 1 |  | 3380 |
| KAN_specific_attribution_bar.svg | 1 |  | 6067 |
| carrier_mechanism_outcome_matrix.svg | 1 |  | 711 |
| allbasis_substrate_progress_heatmap.svg | 1 |  | 4856 |
| gpu_utilization_timeline.svg | 1 |  | 911 |
| budget_deferred_waterfall.svg | 1 |  | 1309 |

## 5. Final route snapshot

```text
route = R6-FunctionalDynamicsAllMechanismsNoGo
line_p_efficiency_census_complete = 1
S2_weak_productive_dynamics_reached = 0
S3_productive_debt_recovery_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
```

## 6. 审计备注

```text
1. Line P timing 使用实际 CUDA forward/backward/update/readback phase timer，不写 proxy timing；optimizer-update phase 使用持久 AdamW optimizer 和 live gradients。
2. P-FB1..P-FB6 fallback columns 对每个 Line P row 写入，efficiency fail 不直接 hard stop。
3. A/B 方向仍来自 train stream split-gradient / optimizer state；LineC/tail/AUC/calibration 只读回。
4. Rational 无 reset/controller/action route。
5. partial artifacts 只作为 checkpoint/resume，不作为 final coverage；required manifest 决定闭合。
6. Line M mechanism_family 从 method surface 回填，修复 v16.2 helper 输出 UNKNOWN 的元数据问题；只改审计可读性，不改 source/delta/gate 数值。
```
