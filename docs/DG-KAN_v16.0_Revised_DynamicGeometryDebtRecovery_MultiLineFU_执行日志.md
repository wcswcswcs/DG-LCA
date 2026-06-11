# DG-KAN v16.0 Revised DynamicGeometryDebtRecovery MultiLineFU 执行日志

生成时间：2026-06-01（Asia/Singapore）

## 1. 关键文件

- plan: /home/chengshun.wang/DG-LCA/docs/DG-KAN_v16.0_Revised_DynamicGeometryDebtRecovery_MultiLineFU_完整计划.md
- runner: /home/chengshun.wang/DG-LCA/experiments/run_v160_dynamic_geometry_debt_recovery_multiline_fu.py
- result dir: results/v16_0_revised_dynamic_geometry_debt_recovery_multiline_fu/official_v160
- recap: /home/chengshun.wang/DG-LCA/docs/DG-KAN_v16.0_Revised_DynamicGeometryDebtRecovery_MultiLineFU_实验结果复盘.md
- execution log: /home/chengshun.wang/DG-LCA/docs/DG-KAN_v16.0_Revised_DynamicGeometryDebtRecovery_MultiLineFU_执行日志.md
- all-basis line F dir: results/v16_0_revised_dynamic_geometry_debt_recovery_multiline_fu/line_f_v160_allbasis_substrate

## 2. 编译 / 修复检查

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v160_dynamic_geometry_debt_recovery_multiline_fu.py experiments/run_v149_line_d_all_basis_substrate_repair.py
```

## 3. 四卡并行分片执行指令

Line A/B 的 shard 使用 `--artifact-suffix` 写入分片 artifact，随后 `MERGE_A/MERGE_B` 合并；C/D/F 同步跑，finalize 只重算 route/docs，不改训练指标。

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v160_dynamic_geometry_debt_recovery_multiline_fu.py --out-dir results/v16_0_revised_dynamic_geometry_debt_recovery_multiline_fu/official_v160 --line-f-out results/v16_0_revised_dynamic_geometry_debt_recovery_multiline_fu/line_f_v160_allbasis_substrate --run-lines S,A --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --batch-size 256 --train-steps 120 --horizon-steps 800 --h1600-steps 1600 --rational-steps 80 --lq-steps 80 --split-count 4 --hidden 64 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --artifact-suffix a0 --line-a-methods A0-D-CHE-AdamW,A1-D-CHE-G7R-PulseOnce-then-AdamWRecovery,A2-D-CHE-G7R-PulseEvery50-then-AdamWRecovery,A3-D-CHE-G7R-PulseEarlyOnly-then-AdamWRecovery,ACTRL0-D-CHE-NoOpMatchedOverhead,ACTRL1-D-CHE-RandomMatchedPulse-then-AdamWRecovery
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v160_dynamic_geometry_debt_recovery_multiline_fu.py --out-dir results/v16_0_revised_dynamic_geometry_debt_recovery_multiline_fu/official_v160 --line-f-out results/v16_0_revised_dynamic_geometry_debt_recovery_multiline_fu/line_f_v160_allbasis_substrate --run-lines A --device cuda:1 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --batch-size 256 --train-steps 120 --horizon-steps 800 --h1600-steps 1600 --rational-steps 80 --lq-steps 80 --split-count 4 --hidden 64 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --artifact-suffix a1 --line-a-methods A4-D-CHE-G7R-PulseMidOnly-then-AdamWRecovery,A5-D-CHE-G7R-PulseLateOnly-then-AdamWRecovery,A6-D-CHE-G7R-PulseOnce-then-MomentumDampedRecovery,A7-D-CHE-G7R-PulseOnce-then-SecondMomentAdaptRecovery,ACTRL2-D-CHE-RandomMatchedPulse-then-SameRecovery,ACTRL3-D-CHE-AdamWExtraStepsMatchedTime
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v160_dynamic_geometry_debt_recovery_multiline_fu.py --out-dir results/v16_0_revised_dynamic_geometry_debt_recovery_multiline_fu/official_v160 --line-f-out results/v16_0_revised_dynamic_geometry_debt_recovery_multiline_fu/line_f_v160_allbasis_substrate --run-lines A --device cuda:2 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --batch-size 256 --train-steps 120 --horizon-steps 800 --h1600-steps 1600 --rational-steps 80 --lq-steps 80 --split-count 4 --hidden 64 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --artifact-suffix a2 --line-a-methods A8-D-CHE-G7R-PulseOnce-then-LRCooldownRecovery,A9-D-CHE-G7R-PulseOnce-then-EMALookaheadRecovery,A10-D-CHE-G7R-PulseOnce-then-DecoupledDecayRecovery,A11-D-CHE-G7R-PulseOnce-then-RoleWiseDecayRecovery,ACTRL4-D-CHE-DecayOnlyRecovery
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v160_dynamic_geometry_debt_recovery_multiline_fu.py --out-dir results/v16_0_revised_dynamic_geometry_debt_recovery_multiline_fu/official_v160 --line-f-out results/v16_0_revised_dynamic_geometry_debt_recovery_multiline_fu/line_f_v160_allbasis_substrate --run-lines A --device cuda:3 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --batch-size 256 --train-steps 120 --horizon-steps 800 --h1600-steps 1600 --rational-steps 80 --lq-steps 80 --split-count 4 --hidden 64 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --artifact-suffix a3 --line-a-methods A12-D-CHE-G7R-PulseOnce-then-HighDegreeDecayRecovery,A13-D-CHE-G7R-PulseOnce-then-SWAConsolidation,ACTRL5-D-CHE-RecoveryOnlyNoPulse,ACTRL6-D-CHE-SamePulseNormRandomDirection
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v160_dynamic_geometry_debt_recovery_multiline_fu.py --out-dir results/v16_0_revised_dynamic_geometry_debt_recovery_multiline_fu/official_v160 --line-f-out results/v16_0_revised_dynamic_geometry_debt_recovery_multiline_fu/line_f_v160_allbasis_substrate --run-lines B --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --batch-size 256 --train-steps 120 --horizon-steps 800 --h1600-steps 1600 --rational-steps 80 --lq-steps 80 --split-count 4 --hidden 64 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --artifact-suffix b0 --line-b-methods B0-MLP-AdamW,B1-MLP-SplitConsensusHiddenMetricPulseOnce-then-AdamWRecovery,B2-MLP-SplitConsensusHiddenMetricPulseEvery50-then-AdamWRecovery,B3-MLP-FMS-AmortizedPulseOnce-then-AdamWRecovery,BCTRL0-MLP-NoOpMatchedOverhead
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v160_dynamic_geometry_debt_recovery_multiline_fu.py --out-dir results/v16_0_revised_dynamic_geometry_debt_recovery_multiline_fu/official_v160 --line-f-out results/v16_0_revised_dynamic_geometry_debt_recovery_multiline_fu/line_f_v160_allbasis_substrate --run-lines B --device cuda:1 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --batch-size 256 --train-steps 120 --horizon-steps 800 --h1600-steps 1600 --rational-steps 80 --lq-steps 80 --split-count 4 --hidden 64 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --artifact-suffix b1 --line-b-methods B4-MLP-PopRiskSNRPulseOnce-then-AdamWRecovery,B5-MLP-PulseOnce-then-MomentumDampedRecovery,B6-MLP-PulseOnce-then-SecondMomentAdaptRecovery,BCTRL1-MLP-RandomMatchedPulse-then-SameRecovery
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v160_dynamic_geometry_debt_recovery_multiline_fu.py --out-dir results/v16_0_revised_dynamic_geometry_debt_recovery_multiline_fu/official_v160 --line-f-out results/v16_0_revised_dynamic_geometry_debt_recovery_multiline_fu/line_f_v160_allbasis_substrate --run-lines B --device cuda:2 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --batch-size 256 --train-steps 120 --horizon-steps 800 --h1600-steps 1600 --rational-steps 80 --lq-steps 80 --split-count 4 --hidden 64 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --artifact-suffix b2 --line-b-methods B7-MLP-PulseOnce-then-LRCooldownRecovery,B8-MLP-PulseOnce-then-EMALookaheadRecovery,B9-MLP-PulseOnce-then-DecoupledDecayRecovery,BCTRL2-MLP-AdamWExtraStepsMatchedTime,BCTRL3-MLP-RecoveryOnlyNoPulse
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v160_dynamic_geometry_debt_recovery_multiline_fu.py --out-dir results/v16_0_revised_dynamic_geometry_debt_recovery_multiline_fu/official_v160 --line-f-out results/v16_0_revised_dynamic_geometry_debt_recovery_multiline_fu/line_f_v160_allbasis_substrate --run-lines B,C,D --device cuda:3 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --batch-size 256 --train-steps 120 --horizon-steps 800 --h1600-steps 1600 --rational-steps 80 --lq-steps 80 --split-count 4 --hidden 64 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --artifact-suffix b3 --line-b-methods B10-MLP-PulseOnce-then-SWAConsolidation,BCTRL4-MLP-DecayOnly,BCTRL5-MLP-SameActiveFractionRandomPulse
```

Line F all-basis substrate command:

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v149_line_d_all_basis_substrate_repair.py --out-dir results/v16_0_revised_dynamic_geometry_debt_recovery_multiline_fu/line_f_v160_allbasis_substrate --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --candidates D-FOU87-LowFreqIdentityResidualV6,D-FOU88-BandwiseSNRWarmupV6,D-FOU89-PhaseStableBandMixV6,D-FOU90-NoMaterializeLifetimeV6,D-FOU91-HighFrequencyQuarantineV6,D-RBF85-ActiveCenterOccupancyV6,D-RBF86-WidthConditionGuardV6,D-RBF87-CompactBumpNoDenseV6,D-RBF88-GaussianLocalK4TaskHealthV6,D-RBF89-IdentityResidualWidthWarmupV6,D-WAV73-TriangularSupportV6,D-WAV74-ScaleOccupancyV6,D-WAV75-SupportOverlapDampingV6,D-WAV76-LocalTailCoverageAuditV6 --device cuda:3 --data-root data --no-download --train-size 256 --val-size 128 --batch-size 32 --epochs 1
```

Merge / h1600 / finalize:

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v160_dynamic_geometry_debt_recovery_multiline_fu.py --out-dir results/v16_0_revised_dynamic_geometry_debt_recovery_multiline_fu/official_v160 --line-f-out results/v16_0_revised_dynamic_geometry_debt_recovery_multiline_fu/line_f_v160_allbasis_substrate --run-lines MERGE_A,MERGE_B --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --batch-size 256 --train-steps 120 --horizon-steps 800 --h1600-steps 1600 --rational-steps 80 --lq-steps 80 --split-count 4 --hidden 64 --lr 0.003 --weight-decay 0.001 --data-root data --no-download
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v160_dynamic_geometry_debt_recovery_multiline_fu.py --out-dir results/v16_0_revised_dynamic_geometry_debt_recovery_multiline_fu/official_v160 --line-f-out results/v16_0_revised_dynamic_geometry_debt_recovery_multiline_fu/line_f_v160_allbasis_substrate --run-lines A1600 --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --batch-size 256 --train-steps 120 --horizon-steps 800 --h1600-steps 1600 --rational-steps 80 --lq-steps 80 --split-count 4 --hidden 64 --lr 0.003 --weight-decay 0.001 --data-root data --no-download
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v160_dynamic_geometry_debt_recovery_multiline_fu.py --out-dir results/v16_0_revised_dynamic_geometry_debt_recovery_multiline_fu/official_v160 --line-f-out results/v16_0_revised_dynamic_geometry_debt_recovery_multiline_fu/line_f_v160_allbasis_substrate --run-lines B1600 --device cuda:1 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --batch-size 256 --train-steps 120 --horizon-steps 800 --h1600-steps 1600 --rational-steps 80 --lq-steps 80 --split-count 4 --hidden 64 --lr 0.003 --weight-decay 0.001 --data-root data --no-download
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v160_dynamic_geometry_debt_recovery_multiline_fu.py --out-dir results/v16_0_revised_dynamic_geometry_debt_recovery_multiline_fu/official_v160 --line-f-out results/v16_0_revised_dynamic_geometry_debt_recovery_multiline_fu/line_f_v160_allbasis_substrate --run-lines FINALIZE --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --batch-size 256 --train-steps 120 --horizon-steps 800 --h1600-steps 1600 --rational-steps 80 --lq-steps 80 --split-count 4 --hidden 64 --lr 0.003 --weight-decay 0.001 --data-root data --no-download --reuse-if-present 1
```

## 4. Artifact inventory

| artifact | exists | rows | bytes |
| --- | --- | --- | --- |
| v160_route_decision.json | 1 |  | 566 |
| v160_line_r_audit.csv | 1 | 8 | 313 |
| v160_line_s_split_consensus_subspace.csv | 1 | 108 | 32824 |
| v160_line_g_dynamic_geometry.csv | 1 | 2646 | 1800519 |
| v160_line_a_dche_dynamics.csv | 1 | 189 | 414732 |
| v160_line_a_horizon_recovery.csv | 1 | 1449 | 923215 |
| v160_line_a_h1600_long.csv | 1 | 18 | 42500 |
| v160_line_b_mlp_dynamics.csv | 1 | 153 | 331214 |
| v160_line_b_horizon_recovery.csv | 1 | 1197 | 683925 |
| v160_line_b_h1600_long.csv | 1 | 18 | 41759 |
| v160_line_c_lq_reanchor.csv | 1 | 63 | 19088 |
| v160_line_c_lq_functional.csv | 1 | 5 | 472 |
| v160_line_d_rational_monitor.csv | 1 | 45 | 45979 |
| v160_line_e_recovery_matrix.csv | 1 | 342 | 73899 |
| v160_line_f_allbasis_results.csv | 1 | 126 | 155600 |
| v160_line_m_crossline_attribution.csv | 1 | 13 | 2429 |
| v160_failure_taxonomy.csv | 1 | 381 | 38071 |
| v160_budget_exhaustion_certificate.csv | 1 | 6 | 933 |
| v160_no_go_boundary.csv | 1 | 7 | 760 |
| v160_next_hypothesis_queue.csv | 1 | 3 | 398 |
| v160_execution_contract_coverage_audit.csv | 1 | 12 | 822 |
| v160_deep_coverage_audit.csv | 1 | 8 | 514 |
| v160_required_artifact_manifest.csv | 1 | 45 | 2006 |
| source_retention_curve_horizon.svg | 1 |  | 5022 |
| tail_debt_curve_horizon.svg | 1 |  | 5891 |
| LineC_debt_curve_horizon.svg | 1 |  | 5887 |
| calibration_debt_curve_horizon.svg | 1 |  | 5897 |
| AUC_debt_curve_horizon.svg | 1 |  | 5875 |
| recovery_rate_by_mechanism.svg | 1 |  | 5455 |
| D-CHE_vs_MLP_source_retention.svg | 1 |  | 565 |
| D-CHE_vs_MLP_tail_recovery.svg | 1 |  | 561 |
| D-CHE_vs_MLP_LineC_recovery.svg | 1 |  | 560 |
| D-CHE_vs_MLP_DGS_horizon.svg | 1 |  | 542 |
| source_retention_vs_tail_recovery.svg | 1 |  | 6089 |
| source_retention_vs_LineC_recovery.svg | 1 |  | 6082 |
| DGS_vs_step_time.svg | 1 |  | 5930 |
| recovery_norm_vs_source_retention.svg | 1 |  | 6086 |
| LQ_historical_current_row_flip.svg | 1 |  | 5675 |
| LQ_macro_delta_vs_nearpass.svg | 1 |  | 5684 |
| LQ_protocol_drift_heatmap.svg | 1 |  | 5695 |
| basis_family_pass_count_heatmap.svg | 1 |  | 4998 |
| basis_step_memory_pareto.svg | 1 |  | 5734 |
| basis_LineC_task_health.svg | 1 |  | 5727 |
| line_gate_status.svg | 1 |  | 1045 |
| failure_taxonomy_heatmap.svg | 1 |  | 5916 |
| exhaustion_certificate_dashboard.svg | 1 |  | 1302 |

## 5. Final route snapshot

```text
route = R16-CurrentFunctionalDynamicsFamilyNoGo
promotion_allowed = 0
line_a_gate_pass = 0
line_b_gate_pass = 0
line_c_reanchor_gate_pass = 0
line_d_rational_gate_pass = 0
line_f_allbasis_gate_pass = 0
required_artifact_missing_count = 0
```

## 11. Finalizer coverage repair

首次 finalizer 后 route=R0，因为 manifest 缺 v160_execution_contract_coverage_audit.csv 与 v160_deep_coverage_audit.csv。已从实际 artifact 生成两份 coverage audit，并补入 runner 的 build_audits 写入逻辑；重跑 finalizer 后 required_artifact_missing_count=0，route=R16-CurrentFunctionalDynamicsFamilyNoGo。该修复只影响覆盖审计/manifest/route/docs，不改变训练指标。

## 12. 用户再次追问后的计划闭环复核

计划复核结论：v16.0 已经按 promotion fail-closed / exploration dynamics-open 执行到 h800/h1600；A/B/C/D/E/F gates 全部为 0，promotion_allowed=0。完整计划明确要求在这种情况下写 R16-CurrentFunctionalDynamicsFamilyNoGo，并停止当前 train-stream functional-update family，不继续新增 token/action/controller/reset。本节只补闭环复核，不新增训练、不改变指标。

## 13. 复盘 insight 补写

用户指出实验结果复盘仍然过薄，只有 artifact 摘要、缺少人工观察与科学判断。已重新读取 v16.0 已落盘 artifact，并扩展 `docs/DG-KAN_v16.0_Revised_DynamicGeometryDebtRecovery_MultiLineFU_实验结果复盘.md` 的 `2.1 人工复核分析 / Insight`。本次只补写分析文本，不新增训练、不重算指标、不改变 route。补写依据包括 `v160_line_a_summary.csv`、`v160_line_b_summary.csv`、`v160_line_a_h1600_long.csv`、`v160_line_b_h1600_long.csv`、`v160_line_m_crossline_attribution.csv`、`v160_failure_taxonomy.csv`、`v160_line_c_lq_reanchor.csv`、`v160_line_f_allbasis_family_summary.csv` 与 `v160_no_go_boundary.csv`。
