# DG-KAN v16.2 MultiSchemeFunctionalDynamics 4GPU 执行日志

生成时间：2026-06-01（Asia/Singapore）

## 1. 关键文件

- plan: /home/chengshun.wang/DG-LCA/docs/DG-KAN_v16.2_MultiSchemeFunctionalDynamics_4GPU_完整计划.md
- runner: /home/chengshun.wang/DG-LCA/experiments/run_v162_multischeme_functional_dynamics_4gpu.py
- shared kernel source: /home/chengshun.wang/DG-LCA/experiments/run_v160_dynamic_geometry_debt_recovery_multiline_fu.py
- all-basis registry: /home/chengshun.wang/DG-LCA/experiments/run_v149_line_d_all_basis_substrate_repair.py
- result dir: results/v16_2_multischeme_functional_dynamics_4gpu/official_v162
- all-basis line F dir: results/v16_2_multischeme_functional_dynamics_4gpu/line_f_v162_allbasis_substrate
- fast horizon readback: 1
- executed method surface: M1a..M8f multi-scheme mechanisms + matched controls; exact per-step full trace was deferred and is not claimed as covered
- recap: /home/chengshun.wang/DG-LCA/docs/DG-KAN_v16.2_MultiSchemeFunctionalDynamics_4GPU_实验结果复盘.md
- execution log: /home/chengshun.wang/DG-LCA/docs/DG-KAN_v16.2_MultiSchemeFunctionalDynamics_4GPU_执行日志.md

## 2. 编译检查

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v162_multischeme_functional_dynamics_4gpu.py experiments/run_v149_line_d_all_basis_substrate_repair.py
```

## 3. 四卡并行分片执行指令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v162_multischeme_functional_dynamics_4gpu.py --out-dir results/v16_2_multischeme_functional_dynamics_4gpu/official_v162 --line-f-out results/v16_2_multischeme_functional_dynamics_4gpu/line_f_v162_allbasis_substrate --run-lines A --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 32 --val-size 32 --test-size 32 --batch-size 8 --train-steps 20 --horizon-steps 800 --h1600-steps 1600 --rational-steps 40 --lq-steps 40 --split-count 2 --hidden 16 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --artifact-suffix a0 --line-a-methods A-M1a-D-CHE-PulseOnce-AdamWRecovery,A-M1b-D-CHE-PulseOnce-LRCooldownRecovery,A-M1c-D-CHE-PulseOnce-MomentumDampingRecovery,A-M1d-D-CHE-PulseOnce-DecoupledGlobalDecayRecovery,A-M1e-D-CHE-PulseOnce-DegreeWiseDecayRecovery,A-M1f-D-CHE-PulseOnce-EMAConsolidation,A-M1g-D-CHE-PulseOnce-LookaheadConsolidation,A-M1h-D-CHE-PulseOnce-SWAConsolidation,A-M1i-D-CHE-PulseEvery50-AdamWRecovery,A-M1j-D-CHE-EarlyOnlyPulse-AdamWRecovery,A-M1k-D-CHE-MidOnlyPulse-AdamWRecovery,A-M1l-D-CHE-LateOnlyPulse-AdamWRecovery,A-M2a-D-CHE-SplitConsensusDiagMetric,A-M2b-D-CHE-SplitConsensusRoleBlockMetric,A-M2c-D-CHE-SplitConsensusLowRank-r4,A-M2d-D-CHE-SplitConsensusLowRank-r8
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v162_multischeme_functional_dynamics_4gpu.py --out-dir results/v16_2_multischeme_functional_dynamics_4gpu/official_v162 --line-f-out results/v16_2_multischeme_functional_dynamics_4gpu/line_f_v162_allbasis_substrate --run-lines A --device cuda:1 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 32 --val-size 32 --test-size 32 --batch-size 8 --train-steps 20 --horizon-steps 800 --h1600-steps 1600 --rational-steps 40 --lq-steps 40 --split-count 2 --hidden 16 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --artifact-suffix a1 --line-a-methods A-M2e-D-CHE-SplitConsensusMetricOnlyNoProjection,A-M2f-D-CHE-SplitConsensusProjectionPlusAdamV,A-M2g-D-CHE-NegativeEigenQuarantineDiagnostic,A-M3a-D-CHE-ParameterSNRPreconditioner,A-M3b-D-CHE-DegreeRoleSNRPreconditioner,A-M3c-D-CHE-BasisChannelSNRCoverBoundary,A-M3d-D-CHE-SNRPulseScheduler,A-M3e-D-CHE-SNRRecoveryScheduler,A-M3f-D-CHE-SNRReservoirGuard,A-M4a-D-CHE-AdamSubspaceProximal-alpha000,A-M4b-D-CHE-PerExampleLowRankProximal-alpha025,A-M4c-D-CHE-OutputJacobianSketchProximal-alpha050,A-M4d-D-CHE-SplitB1B2TransferProximal-alpha100,A-M4e-D-CHE-RecoveryAwareProximal-alpha200,A-M4f-D-CHE-ControlResidualizedProximal-alpha100,A-M5a-D-CHE-CautiousFU
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v162_multischeme_functional_dynamics_4gpu.py --out-dir results/v16_2_multischeme_functional_dynamics_4gpu/official_v162 --line-f-out results/v16_2_multischeme_functional_dynamics_4gpu/line_f_v162_allbasis_substrate --run-lines A --device cuda:2 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 32 --val-size 32 --test-size 32 --batch-size 8 --train-steps 20 --horizon-steps 800 --h1600-steps 1600 --rational-steps 40 --lq-steps 40 --split-count 2 --hidden 16 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --artifact-suffix a2 --line-a-methods A-M5b-D-CHE-SoftCautiousFU,A-M5c-D-CHE-MGUPStyleReweightFU,A-M5d-D-CHE-AdamSecondMomentScaledFU,A-M5e-D-CHE-SophiaDiagLiteClippedFU,A-M5f-D-CHE-BlockSecondMomentFU,A-M5g-D-CHE-DecoupledDecayPlusFU,A-M6a-D-CHE-LowDegreeSignalHighDegreeReservoir,A-M6b-D-CHE-ReadoutBasisDecoupledCarrier,A-M6c-D-CHE-OrthogonalDegreeBankCarrier,A-M6d-D-CHE-OutputJacobianCarrier,A-M6e-D-CHE-DualBankSignalReservoirCarrier,A-M6f-D-CHE-HighDegreeQuarantineCarrier,A-M7a-D-CHE-LateAttachEarlyCheckpoint,A-M7b-D-CHE-LateAttachMidCheckpoint,A-M7c-D-CHE-LateAttachLateCheckpoint,A-M7d-D-CHE-LateAttachAfterLossPlateau
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v162_multischeme_functional_dynamics_4gpu.py --out-dir results/v16_2_multischeme_functional_dynamics_4gpu/official_v162 --line-f-out results/v16_2_multischeme_functional_dynamics_4gpu/line_f_v162_allbasis_substrate --run-lines A --device cuda:3 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 32 --val-size 32 --test-size 32 --batch-size 8 --train-steps 20 --horizon-steps 800 --h1600-steps 1600 --rational-steps 40 --lq-steps 40 --split-count 2 --hidden 16 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --artifact-suffix a3 --line-a-methods A-M7e-D-CHE-LateAttachHighSourceLowDebtWindow,A-M8a-D-CHE-AdamWRecoveryOnly,A-M8b-D-CHE-LRCooldownOnly,A-M8c-D-CHE-DecayRecoveryOnly,A-M8d-D-CHE-EMASWARecoveryOnly,A-M8e-D-CHE-LookaheadRecoveryOnly,A-M8f-D-CHE-MomentumDampingOnly,ACTRL0-D-CHE-AdamW,ACTRL1-D-CHE-NoOpMatchedOverhead,ACTRL2-D-CHE-RandomMatchedPulse,ACTRL3-D-CHE-AdamWExtraStepsMatchedTime,ACTRL4-D-CHE-RandomSubspaceSameRank,ACTRL5-D-CHE-SameActiveFractionRandomMask,ACTRL6-D-CHE-RecoveryOnlyNoPulse,ACTRL7-D-CHE-DecayOnlyRecovery
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v162_multischeme_functional_dynamics_4gpu.py --out-dir results/v16_2_multischeme_functional_dynamics_4gpu/official_v162 --line-f-out results/v16_2_multischeme_functional_dynamics_4gpu/line_f_v162_allbasis_substrate --run-lines B --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 32 --val-size 32 --test-size 32 --batch-size 8 --train-steps 20 --horizon-steps 800 --h1600-steps 1600 --rational-steps 40 --lq-steps 40 --split-count 2 --hidden 16 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --artifact-suffix b0 --line-b-methods B-M1a-MLP-PulseOnce-AdamWRecovery,B-M1b-MLP-PulseOnce-LRCooldownRecovery,B-M1c-MLP-PulseOnce-MomentumDampingRecovery,B-M1d-MLP-PulseOnce-DecoupledDecayRecovery,B-M1e-MLP-PulseOnce-FeatureRecenterRecovery,B-M1f-MLP-PulseOnce-EMAConsolidation,B-M1g-MLP-PulseOnce-LookaheadConsolidation,B-M1h-MLP-PulseOnce-SWAConsolidation,B-M1i-MLP-PulseEvery50-AdamWRecovery,B-M1j-MLP-EarlyOnlyPulse-AdamWRecovery,B-M1k-MLP-MidOnlyPulse-AdamWRecovery,B-M1l-MLP-LateOnlyPulse-AdamWRecovery,B-M2a-MLP-HiddenSplitConsensusDiagMetric,B-M2b-MLP-HiddenSplitConsensusRoleBlockMetric,B-M2c-MLP-HiddenSplitConsensusLowRank-r4,B-M2d-MLP-HiddenSplitConsensusLowRank-r8
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v162_multischeme_functional_dynamics_4gpu.py --out-dir results/v16_2_multischeme_functional_dynamics_4gpu/official_v162 --line-f-out results/v16_2_multischeme_functional_dynamics_4gpu/line_f_v162_allbasis_substrate --run-lines B --device cuda:1 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 32 --val-size 32 --test-size 32 --batch-size 8 --train-steps 20 --horizon-steps 800 --h1600-steps 1600 --rational-steps 40 --lq-steps 40 --split-count 2 --hidden 16 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --artifact-suffix b1 --line-b-methods B-M2e-MLP-HiddenMetricOnlyNoProjection,B-M2f-MLP-HiddenProjectionPlusAdamV,B-M2g-MLP-HiddenNegativeEigenQuarantineDiagnostic,B-M3a-MLP-ParameterSNRPreconditioner,B-M3b-MLP-HiddenRoleSNRPreconditioner,B-M3c-MLP-FeatureChannelSNRCoverBoundary,B-M3d-MLP-SNRPulseScheduler,B-M3e-MLP-SNRRecoveryScheduler,B-M3f-MLP-SNRReservoirGuard,B-M4a-MLP-AdamSubspaceProximal-alpha000,B-M4b-MLP-PerExampleLowRankProximal-alpha025,B-M4c-MLP-OutputJacobianSketchProximal-alpha050,B-M4d-MLP-SplitB1B2TransferProximal-alpha100,B-M4e-MLP-RecoveryAwareProximal-alpha200,B-M4f-MLP-ControlResidualizedProximal-alpha100,B-M5a-MLP-CautiousAdamW
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v162_multischeme_functional_dynamics_4gpu.py --out-dir results/v16_2_multischeme_functional_dynamics_4gpu/official_v162 --line-f-out results/v16_2_multischeme_functional_dynamics_4gpu/line_f_v162_allbasis_substrate --run-lines B --device cuda:2 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 32 --val-size 32 --test-size 32 --batch-size 8 --train-steps 20 --horizon-steps 800 --h1600-steps 1600 --rational-steps 40 --lq-steps 40 --split-count 2 --hidden 16 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --artifact-suffix b2 --line-b-methods B-M5b-MLP-SoftCautiousFU,B-M5c-MLP-MGUPStyleReweight,B-M5d-MLP-AdamSecondMomentScaledFU,B-M5e-MLP-SophiaDiagLiteClippedFU,B-M5f-MLP-BlockSecondMomentFU,B-M5g-MLP-DecoupledDecayPlusFU,B-M6a-MLP-HiddenSubspaceCarrier,B-M6b-MLP-HiddenReadoutDecoupledCarrier,B-M6c-MLP-HiddenOrthogonalBankCarrier,B-M6d-MLP-OutputJacobianCarrier,B-M6e-MLP-DualBankSignalReservoirCarrier,B-M6f-MLP-HiddenCarrierDecay,B-M7a-MLP-LateAttachEarlyCheckpoint,B-M7b-MLP-LateAttachMidCheckpoint,B-M7c-MLP-LateAttachLateCheckpoint,B-M7d-MLP-LateAttachAfterLossPlateau
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v162_multischeme_functional_dynamics_4gpu.py --out-dir results/v16_2_multischeme_functional_dynamics_4gpu/official_v162 --line-f-out results/v16_2_multischeme_functional_dynamics_4gpu/line_f_v162_allbasis_substrate --run-lines B,C,D --device cuda:3 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 32 --val-size 32 --test-size 32 --batch-size 8 --train-steps 20 --horizon-steps 800 --h1600-steps 1600 --rational-steps 40 --lq-steps 40 --split-count 2 --hidden 16 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --artifact-suffix b3 --line-b-methods B-M7e-MLP-LateAttachHighSourceLowDebtWindow,B-M8a-MLP-AdamWRecoveryOnly,B-M8b-MLP-LRCooldownOnly,B-M8c-MLP-DecayRecoveryOnly,B-M8d-MLP-EMASWARecoveryOnly,B-M8e-MLP-LookaheadRecoveryOnly,B-M8f-MLP-MomentumDampingOnly,BCTRL0-MLP-AdamW,BCTRL1-MLP-NoOpMatchedOverhead,BCTRL2-MLP-RandomMatchedPulse,BCTRL3-MLP-AdamWExtraStepsMatchedTime,BCTRL4-MLP-RandomSubspaceSameRank,BCTRL5-MLP-SameActiveFractionRandomMask,BCTRL6-MLP-RecoveryOnlyNoPulse,BCTRL7-MLP-DecayOnly
```

Line F all-basis substrate command:

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v149_line_d_all_basis_substrate_repair.py --out-dir results/v16_2_multischeme_functional_dynamics_4gpu/line_f_v162_allbasis_substrate --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --candidates D-FOU97-LowFreqIdentityResidualV8,D-FOU98-BandwiseSNRWarmupV8,D-FOU99-PhaseStableBandMixV8,D-FOU100-NoMaterializeLifetimeV8,D-FOU101-HighFrequencyQuarantineV8,D-FOU102-SplitConsensusLowFreqMetricSmoke,D-RBF95-ActiveCenterOccupancyV8,D-RBF96-WidthConditionGuardV8,D-RBF97-CompactBumpNoDenseV8,D-RBF98-GaussianLocalK4TaskHealthV8,D-RBF99-ActiveCenterSecondMomentV8,D-RBF100-SplitConsensusCenterMetricSmoke,D-WAV81-TriangularSupportV8,D-WAV82-ScaleOccupancyV8,D-WAV83-SupportOverlapDampingV8,D-WAV84-LocalTailCoverageAuditV8,D-WAV85-SplitConsensusScaleMetricSmoke --device cuda:3 --data-root data --no-download --train-size 32 --val-size 32 --batch-size 32 --epochs 1
```

Merge / h1600 / finalize:

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v162_multischeme_functional_dynamics_4gpu.py --out-dir results/v16_2_multischeme_functional_dynamics_4gpu/official_v162 --line-f-out results/v16_2_multischeme_functional_dynamics_4gpu/line_f_v162_allbasis_substrate --run-lines MERGE_A,MERGE_B,A1600 --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 32 --val-size 32 --test-size 32 --batch-size 8 --train-steps 20 --horizon-steps 800 --h1600-steps 1600 --rational-steps 40 --lq-steps 40 --split-count 2 --hidden 16 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v162_multischeme_functional_dynamics_4gpu.py --out-dir results/v16_2_multischeme_functional_dynamics_4gpu/official_v162 --line-f-out results/v16_2_multischeme_functional_dynamics_4gpu/line_f_v162_allbasis_substrate --run-lines B1600 --device cuda:1 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 32 --val-size 32 --test-size 32 --batch-size 8 --train-steps 20 --horizon-steps 800 --h1600-steps 1600 --rational-steps 40 --lq-steps 40 --split-count 2 --hidden 16 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v162_multischeme_functional_dynamics_4gpu.py --out-dir results/v16_2_multischeme_functional_dynamics_4gpu/official_v162 --line-f-out results/v16_2_multischeme_functional_dynamics_4gpu/line_f_v162_allbasis_substrate --run-lines FINALIZE --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 32 --val-size 32 --test-size 32 --batch-size 8 --train-steps 20 --horizon-steps 800 --h1600-steps 1600 --rational-steps 40 --lq-steps 40 --split-count 2 --hidden 16 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --fast-horizon-readback 1 --reuse-if-present 1
```

## 4. GPU assignment manifest

| round | gpu | run lines | suffix | rows | artifact exists | role |
| --- | --- | --- | --- | --- | --- | --- |
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

## 5. Artifact inventory

| artifact | exists | rows | bytes |
| --- | --- | --- | --- |
| v162_route_decision.json | 1 |  | 1203 |
| v162_progress_table.csv | 1 | 12 | 797 |
| v162_method_surface_manifest.csv | 1 | 126 | 34782 |
| v162_carrier_registry_manifest.csv | 1 | 5 | 384 |
| v162_line_r_audit.csv | 1 | 9 | 346 |
| v162_forbidden_information_audit.csv | 1 | 9 | 346 |
| v162_no_action_search_audit.csv | 1 | 5 | 138 |
| v162_gpu_assignment_manifest.csv | 1 | 12 | 1908 |
| v162_gpu_utilization_summary.csv | 1 | 4 | 131 |
| v162_queue_status.csv | 1 | 12 | 332 |
| v162_line_g_dynamic_geometry.csv | 1 | 7938 | 6368461 |
| v162_dynamic_geometry_debt_accounting.csv | 1 | 7938 | 6368461 |
| v162_line_a_dche_matrix.csv | 1 | 567 | 1291705 |
| v162_line_a_horizon_recovery.csv | 1 | 3969 | 2961927 |
| v162_line_a_h1600_long.csv | 1 | 18 | 43661 |
| v162_line_b_mlp_matrix.csv | 1 | 567 | 1281841 |
| v162_line_b_horizon_recovery.csv | 1 | 3969 | 2777613 |
| v162_line_b_h1600_long.csv | 1 | 18 | 42976 |
| v162_line_c_lq_reanchor.csv | 1 | 63 | 19403 |
| v162_line_c_lq_functional.csv | 1 | 5 | 529 |
| v162_line_d_rational_monitor.csv | 1 | 45 | 45745 |
| v162_line_e_recovery_matrix.csv | 1 | 1134 | 260156 |
| v162_line_f_allbasis_results.csv | 1 | 153 | 188889 |
| v162_line_f_allbasis_substrate.csv | 1 | 153 | 188889 |
| v162_line_f_allbasis_family_summary.csv | 1 | 3 | 619 |
| v162_carrier_mechanism_summary.csv | 1 | 16 | 3036 |
| v162_carrier_mechanism_matrix.csv | 1 | 16 | 3036 |
| v162_line_m_crossline_attribution.csv | 1 | 55 | 13990 |
| v162_line_m_controls_attribution.csv | 1 | 55 | 13990 |
| v162_failure_taxonomy.csv | 1 | 1173 | 148212 |
| v162_budget_exhaustion_certificate.csv | 1 | 5 | 1028 |
| v162_deferred_items.csv | 1 | 5 | 470 |
| v162_no_go_boundary.csv | 1 | 7 | 805 |
| v162_no_go_boundary.md | 1 |  | 873 |
| v162_next_hypothesis_queue.csv | 1 | 3 | 414 |
| v162_next_hypothesis_queue.md | 1 |  | 465 |
| v162_code_review_packet.csv | 1 | 9 | 3090 |
| v162_code_review_packet.zip | 1 |  | 7487 |
| v162_execution_contract_coverage_audit.csv | 1 | 13 | 848 |
| v162_deep_coverage_audit.csv | 1 | 9 | 572 |
| v162_required_artifact_manifest.csv | 1 | 54 | 2662 |
| figures/fig_v162_carrier_mechanism_heatmap_source_h800.svg | 1 |  | 3396 |
| figures/fig_v162_carrier_mechanism_heatmap_s2_s3.svg | 1 |  | 3347 |
| figures/fig_v162_source_retention_vs_tail_recovery.svg | 1 |  | 6038 |
| figures/fig_v162_source_retention_vs_LineC_recovery.svg | 1 |  | 6037 |
| figures/fig_v162_horizon_curves_DCHE_top5.svg | 1 |  | 6119 |
| figures/fig_v162_horizon_curves_MLP_top5.svg | 1 |  | 6013 |
| figures/fig_v162_MLP_vs_DCHE_attribution_matrix.svg | 1 |  | 5637 |
| figures/fig_v162_LQ_reanchor_dashboard.svg | 1 |  | 5681 |
| figures/fig_v162_allbasis_substrate_heatmap.svg | 1 |  | 5786 |
| figures/fig_v162_recovery_mechanism_comparison.svg | 1 |  | 3290 |
| figures/fig_v162_controls_explainability_waterfall.svg | 1 |  | 6017 |
| figures/fig_v162_failure_taxonomy_heatmap.svg | 1 |  | 5805 |
| figures/fig_v162_gpu_utilization_dashboard.svg | 1 |  | 2319 |
| figures/fig_v162_deferred_items_dashboard.svg | 1 |  | 1187 |

## 6. Final route snapshot

```text
route = R6-FunctionalDynamicsAllMechanismsNoGo
S2_weak_productive_dynamics_reached = 0
S3_productive_debt_recovery_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
```

## 7. 修复与审计记录

```text
1. 新增 v16.2 runner，使用 v16.0 validated kernels，但单独生成 v162 artifacts/docs/route。
2. 新增 v16.2 substrate-only candidate registry；Line F rows 仍不能 promotion。
3. method_surface_manifest 记录每个方法的内部映射与参数 override，便于后续审计。
4. gpu_assignment_manifest 记录四卡分片；cpu_offload_used=0。
5. runtime blocker 修复：exact per-step full trace 过慢；正式结果只声明 horizon-target readback，不把未执行 trace density 写成 coverage。
6. runtime blocker 修复：初始 batch=32/hidden=32 shard 超过 75 分钟无 artifact，停止后用 batch=8/hidden=16 重跑；未落盘尝试不进入结果指标。
7. runtime blocker 修复：batch=8/hidden=16 A shard 约 45 分钟仍无中途 artifact；新增 row-level checkpoint/resume 后重跑，partial shard 不作为 final coverage。
8. runtime blocker 修复：B1600 首次与 MERGE_B 并行启动触发 pre-merge race；空文件已备份，B1600 已串行重跑。
9. runtime insight：MLP shard 比 D-CHE shard 快，主要来自 D-CHE 前后向和 LineC horizon readback 更重；不作为 scientific gate。
```

## 8. 追问后的闭环复核记录

本节不新增训练，不改变任何训练指标；只记录对 plan hard stop、route artifact、manifest/coverage 与 h1600 extension 的读回复核。

复核使用的主要文件：

```text
docs/DG-KAN_v16.2_MultiSchemeFunctionalDynamics_4GPU_完整计划.md
results/v16_2_multischeme_functional_dynamics_4gpu/official_v162/v162_route_decision.json
results/v16_2_multischeme_functional_dynamics_4gpu/official_v162/v162_required_artifact_manifest.csv
results/v16_2_multischeme_functional_dynamics_4gpu/official_v162/v162_execution_contract_coverage_audit.csv
results/v16_2_multischeme_functional_dynamics_4gpu/official_v162/v162_deep_coverage_audit.csv
results/v16_2_multischeme_functional_dynamics_4gpu/official_v162/v162_budget_exhaustion_certificate.csv
results/v16_2_multischeme_functional_dynamics_4gpu/official_v162/v162_deferred_items.csv
results/v16_2_multischeme_functional_dynamics_4gpu/official_v162/v162_no_go_boundary.csv
results/v16_2_multischeme_functional_dynamics_4gpu/official_v162/v162_line_m_controls_attribution.csv
results/v16_2_multischeme_functional_dynamics_4gpu/official_v162/v162_line_a_h1600_long.csv
results/v16_2_multischeme_functional_dynamics_4gpu/official_v162/v162_line_b_h1600_long.csv
```

复核命令：

```bash
rg -n "R6-FunctionalDynamicsAllMechanismsNoGo|No S5|hard stop|action token|controller|reset|current train-stream functional|runnable job|S2|S3|No exceptions|deferred|fallback" \
  docs/DG-KAN_v16.2_MultiSchemeFunctionalDynamics_4GPU_完整计划.md

python - <<'PY'
from pathlib import Path
import csv, json
root = Path('results/v16_2_multischeme_functional_dynamics_4gpu/official_v162')
print((root / 'v162_route_decision.json').read_text())
for name in [
    'v162_required_artifact_manifest.csv',
    'v162_execution_contract_coverage_audit.csv',
    'v162_deep_coverage_audit.csv',
    'v162_budget_exhaustion_certificate.csv',
    'v162_deferred_items.csv',
    'v162_no_go_boundary.csv',
    'v162_line_m_controls_attribution.csv',
]:
    rows = list(csv.DictReader((root / name).open(newline='')))
    print(name, len(rows))
for name in ['v162_line_a_h1600_long.csv', 'v162_line_b_h1600_long.csv']:
    rows = list(csv.DictReader((root / name).open(newline='')))
    groups = {}
    for row in rows:
        groups.setdefault(row['method'], []).append(row)
    for method, group in groups.items():
        def mean(col):
            vals = [float(r[col]) for r in group if r.get(col) not in ('', None)]
            return sum(vals) / len(vals) if vals else None
        print(name, method, len(group), mean('source_vs_best_control'), mean('bad_event'), mean('tail_debt_recovery_rate'), mean('LineC_debt_recovery_rate'))
PY
```

复核输出摘要：

```text
route = R6-FunctionalDynamicsAllMechanismsNoGo
S2_weak_productive_dynamics_reached = 0
S3_productive_debt_recovery_reached = 0
official_s5_reached = 0
promotion_allowed = 0
required_artifact_manifest_rows = 54
execution_contract_coverage_rows = 13
deep_coverage_rows = 9
budget_certificate_rows = 5
deferred_items_rows = 5
no_go_boundary_rows = 7
line_m_controls_attribution_rows = 55
KAN_specific_advantage_rows = 0
```

h1600 读回摘要：

```text
A-M4a-D-CHE-AdamSubspaceProximal-alpha000: rows=9, mean source=0.0, bad=1.0, tail=0.017574313511256316, LineC=0.4444444444444444
A-M4b-D-CHE-PerExampleLowRankProximal-alpha025: rows=9, mean source=-0.02831966347164578, bad=1.0, tail=0.03057429645997782, LineC=0.4444444444444444
B-M4a-MLP-AdamSubspaceProximal-alpha000: rows=9, mean source=-0.002453327178955078, bad=1.0, tail=0.0, LineC=0.0
B-M4b-MLP-PerExampleLowRankProximal-alpha025: rows=9, mean source=-0.12256171968248156, bad=1.0, tail=0.0, LineC=0.0
```

闭环判断：

```text
1. v16.2 coverage/contract 目标已达成，promotion/scientific success 未达成。
2. 最强 h800 row 有 source 和 retention，但 tail=0.0、LineC=0.2222222222222222，因此 S2/S3 均不能打开。
3. h1600 top-2 没有巩固 source；route 保持 R6。
4. 计划第 13/15 节要求 all lines complete 且 no S2 时写 R6，并禁止继续 action token/controller/action bank/reset route。
5. 本轮没有剩余计划内可执行补救分支；下一步必须是新的预注册 theory/substrate/base-level 计划。
```
