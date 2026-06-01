# DG-KAN v14.3 FunctionalValueConstraint AllBasisSubstrate 执行日志

生成时间：2026-05-29（Asia/Singapore）

本日志只记录本轮实际执行的命令、代码入口、artifact 目录与关键输出，方便后续复现；不把 smoke / compute-budgeted repair / local positive row 写成 promotion。

## 1. 代码入口

新增 runner：

```text
experiments/run_v143_functional_value_constraint_all_basis_substrate.py
```

计划文件：

```text
docs/DG-KAN_v14.3_FunctionalValueConstraint_AllBasisSubstrate_完整计划.md
```

主要输出目录：

```text
results/v14_3_functional_value_constraint_all_basis_substrate/
```

## 2. 语法检查

执行：

```bash
conda run -n kan python -m py_compile experiments/run_v143_functional_value_constraint_all_basis_substrate.py
```

结果：

```text
py_compile pass
```

## 3. Smoke

第一次 smoke 发现 required manifest / packet 写出顺序问题，随后修复 runner 的 manifest / packet finalization 顺序，并重跑。

重跑命令：

```bash
conda run -n kan python experiments/run_v143_functional_value_constraint_all_basis_substrate.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/smoke_v143 \
  --synthetic-tasks X1 --synthetic-seeds 0 --loss-interfaces CE \
  --generic-methods G0-AdamW,G5-AmortizedParameterFMS,GCTRL-RandomMatchedNorm \
  --rational-methods K0-RAT-AdamW,K1-RAT-GenericFMS-NoProjection,K3-RAT-GenericFMS-DenSlopeTrustRegion,KCTRL-RandomMatchedProjection \
  --train-steps 4 --batch-size 8 --synthetic-train-size 32 --synthetic-val-size 24 \
  --synthetic-dim 16 --mlp-hidden 16 --trace-interval 2 \
  --fms-update-interval 2 --generic-fms-update-interval 2 --compute-budgeted-run 1
```

结果：

```text
route = R1-NoGenericFMSValue
minimum_success = S0-ValueConstraintSurfaceExecuted
generic_fms_task_pass_count = 0
rational_projection_pass_method_count = 2
rational_projection_pass_methods = K1,K3
rational_fms_task_pass_count = 1
kan_specific_positive_rows = 1
nonrat_strict_substrate_pass_count = 0
required_artifact_missing_count = 0
promotion_allowed = 0
real_short_run_open_allowed = 0
```

artifact：

```text
results/v14_3_functional_value_constraint_all_basis_substrate/smoke_v143/
```

## 4. Official compute-budgeted seed0

执行：

```bash
conda run -n kan python experiments/run_v143_functional_value_constraint_all_basis_substrate.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/official_v143 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0 --loss-interfaces CE,Brier \
  --generic-methods G0-AdamW,G5-AmortizedParameterFMS,G6-AmortizedLayerFMS,G7-PhaseScheduleGenericFMS,GCTRL-RandomMatchedNorm \
  --rational-methods K0-RAT-AdamW,K1-RAT-GenericFMS-NoProjection,K2-RAT-GenericFMS-IdentityProjectionAudit,K3-RAT-GenericFMS-DenSlopeTrustRegion,K4-RAT-GenericFMS-ReadoutBasisTrustRegion,K5-RAT-GenericFMS-RoleWisePlasticityTrustRegion,K6-RAT-GenericFMS-DelayedBasisConstraint,K7-RAT-GenericFMS-PhaseScheduleConstraint,K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint,KCTRL-RandomMatchedProjection \
  --train-steps 80 --batch-size 8 --synthetic-train-size 96 --synthetic-val-size 64 \
  --synthetic-dim 16 --mlp-hidden 24 --trace-interval 40 \
  --fms-update-interval 40 --generic-fms-update-interval 40 --compute-budgeted-run 1
```

blocker：

```text
v143_code_review_packet.zip 被 packet() 自身收录，导致 zip 递归膨胀。
```

修复：

```text
experiments/run_v143_functional_value_constraint_all_basis_substrate.py
  1. packet() 跳过 v143_code_review_packet.zip 自身。
  2. 新增 --finalize-existing，用于不重跑训练地重建 route / manifest / packet。
```

清理与 finalize：

```bash
rm -f results/v14_3_functional_value_constraint_all_basis_substrate/official_v143/v143_code_review_packet.zip

conda run -n kan python experiments/run_v143_functional_value_constraint_all_basis_substrate.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/official_v143 \
  --compute-budgeted-run 1 --finalize-existing
```

结果：

```text
route = R1-NoGenericFMSValue
minimum_success = S0-ValueConstraintSurfaceExecuted
generic_fms_task_pass_count = 4
rational_projection_pass_method_count = 4
rational_projection_pass_methods = K1,K2,K3,K8
rational_fms_task_pass_count = 2
kan_specific_positive_rows = 4
rational_source_positive_rows = 39
rational_linec_tail_rejected_source_positive_rows = 35
nonrat_strict_substrate_pass_count = 0
required_artifact_missing_count = 0
promotion_allowed = 0
real_short_run_open_allowed = 0
```

artifact：

```text
results/v14_3_functional_value_constraint_all_basis_substrate/official_v143/
```

## 5. Generic FMS 200-step seed0

执行：

```bash
conda run -n kan python experiments/run_v143_functional_value_constraint_all_basis_substrate.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/generic_confirm200_seed0_v143 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0 --loss-interfaces CE,Brier \
  --generic-methods G0-AdamW,G1-PriorSNRReference,G2-ParameterFMS,G3-LayerFMS,G4-RoleFMS,G5-AmortizedParameterFMS,G6-AmortizedLayerFMS,G7-PhaseScheduleGenericFMS,GCTRL-RandomMatchedNorm \
  --rational-methods '' \
  --train-steps 200 --batch-size 8 --synthetic-train-size 96 --synthetic-val-size 64 \
  --synthetic-dim 16 --mlp-hidden 24 --trace-interval 100 \
  --fms-update-interval 40 --generic-fms-update-interval 40 --compute-budgeted-run 1
```

结果：

```text
route = R2-GenericValueKilledByBasisProjection
minimum_success = S1-GenericFMSPositive
generic_fms_task_pass_count = 5
required_artifact_missing_count = 0
promotion_allowed = 0
```

说明：

```text
这是 G-only confirmation run；没有 Rational K projection rows。
route=R2 只表示本 run 缺少 K projection，不是最终 v14.3 route。
```

## 6. Rational projection 200-step seed0 selected-K

执行：

```bash
conda run -n kan python experiments/run_v143_functional_value_constraint_all_basis_substrate.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_seed0_v143 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0 --loss-interfaces CE,Brier \
  --generic-methods G0-AdamW,G5-AmortizedParameterFMS,G6-AmortizedLayerFMS,G7-PhaseScheduleGenericFMS,GCTRL-RandomMatchedNorm \
  --rational-methods K0-RAT-AdamW,K1-RAT-GenericFMS-NoProjection,K3-RAT-GenericFMS-DenSlopeTrustRegion,K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint,KCTRL-RandomMatchedProjection \
  --train-steps 200 --batch-size 8 --synthetic-train-size 96 --synthetic-val-size 64 \
  --synthetic-dim 16 --mlp-hidden 24 --trace-interval 100 \
  --fms-update-interval 40 --generic-fms-update-interval 40 --compute-budgeted-run 1
```

结果：

```text
route = R3-RationalLineCTailUnsafe
minimum_success = S2-RationalValuePreserved
generic_fms_task_pass_count = 5
rational_projection_pass_method_count = 3
rational_projection_pass_methods = K1,K3,K8
rational_fms_task_pass_count = 3
kan_specific_positive_rows = 3
rational_source_positive_rows = 20
rational_linec_tail_rejected_source_positive_rows = 13
nonrat_strict_substrate_pass_count = 0
required_artifact_missing_count = 0
promotion_allowed = 0
```

## 7. Rational projection 200-step seed0 all-K

执行：

```bash
conda run -n kan python experiments/run_v143_functional_value_constraint_all_basis_substrate.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed0_v143 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0 --loss-interfaces CE,Brier \
  --generic-methods G0-AdamW,G5-AmortizedParameterFMS,G6-AmortizedLayerFMS,G7-PhaseScheduleGenericFMS,GCTRL-RandomMatchedNorm \
  --rational-methods K0-RAT-AdamW,K1-RAT-GenericFMS-NoProjection,K2-RAT-GenericFMS-IdentityProjectionAudit,K3-RAT-GenericFMS-DenSlopeTrustRegion,K4-RAT-GenericFMS-ReadoutBasisTrustRegion,K5-RAT-GenericFMS-RoleWisePlasticityTrustRegion,K6-RAT-GenericFMS-DelayedBasisConstraint,K7-RAT-GenericFMS-PhaseScheduleConstraint,K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint,KCTRL-RandomMatchedProjection \
  --train-steps 200 --batch-size 8 --synthetic-train-size 96 --synthetic-val-size 64 \
  --synthetic-dim 16 --mlp-hidden 24 --trace-interval 100 \
  --fms-update-interval 40 --generic-fms-update-interval 40 --compute-budgeted-run 1
```

结果：

```text
route = R3-RationalLineCTailUnsafe
minimum_success = S2-RationalValuePreserved
generic_fms_task_pass_count = 5
rational_projection_pass_method_count = 4
rational_projection_pass_methods = K1,K2,K3,K8
rational_fms_task_pass_count = 4
kan_specific_positive_rows = 6
rational_source_positive_rows = 48
rational_linec_tail_rejected_source_positive_rows = 35
nonrat_strict_substrate_pass_count = 0
required_artifact_missing_count = 0
promotion_allowed = 0
```

## 8. Generic FMS 200-step seed012

执行：

```bash
conda run -n kan python experiments/run_v143_functional_value_constraint_all_basis_substrate.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/generic_confirm200_seed012_v143 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0,1,2 --loss-interfaces CE,Brier \
  --generic-methods G0-AdamW,G1-PriorSNRReference,G2-ParameterFMS,G3-LayerFMS,G4-RoleFMS,G5-AmortizedParameterFMS,G6-AmortizedLayerFMS,G7-PhaseScheduleGenericFMS,GCTRL-RandomMatchedNorm \
  --rational-methods '' \
  --train-steps 200 --batch-size 8 --synthetic-train-size 96 --synthetic-val-size 64 \
  --synthetic-dim 16 --mlp-hidden 24 --trace-interval 100 \
  --fms-update-interval 40 --generic-fms-update-interval 40 --compute-budgeted-run 1
```

结果：

```text
route = R2-GenericValueKilledByBasisProjection
minimum_success = S1-GenericFMSPositive
generic_fms_task_pass_count = 7
required_artifact_missing_count = 0
promotion_allowed = 0
```

说明：

```text
这是 3-seed generic confirmation；证明 generic FMS signal 存在，
但仍不能写成 KAN-specific success。
```

## 9. Rational projection 200-step seed012 all-K

执行：

```bash
conda run -n kan python experiments/run_v143_functional_value_constraint_all_basis_substrate.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0,1,2 --loss-interfaces CE,Brier \
  --generic-methods G0-AdamW,G5-AmortizedParameterFMS,G6-AmortizedLayerFMS,G7-PhaseScheduleGenericFMS,GCTRL-RandomMatchedNorm \
  --rational-methods K0-RAT-AdamW,K1-RAT-GenericFMS-NoProjection,K2-RAT-GenericFMS-IdentityProjectionAudit,K3-RAT-GenericFMS-DenSlopeTrustRegion,K4-RAT-GenericFMS-ReadoutBasisTrustRegion,K5-RAT-GenericFMS-RoleWisePlasticityTrustRegion,K6-RAT-GenericFMS-DelayedBasisConstraint,K7-RAT-GenericFMS-PhaseScheduleConstraint,K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint,KCTRL-RandomMatchedProjection \
  --train-steps 200 --batch-size 8 --synthetic-train-size 96 --synthetic-val-size 64 \
  --synthetic-dim 16 --mlp-hidden 24 --trace-interval 100 \
  --fms-update-interval 40 --generic-fms-update-interval 40 --compute-budgeted-run 1
```

最终结果：

```text
route = R4-NonRATSubstrateMissing
minimum_success = S3-KANSpecificSyntheticPass
generic_fms_task_pass_count = 7
rational_projection_pass_method_count = 4
rational_projection_pass_methods = K1,K2,K3,K8
rational_fms_task_pass_count = 5
kan_specific_positive_rows = 12
rational_source_positive_rows = 100
rational_linec_tail_rejected_source_positive_rows = 73
nonrat_strict_substrate_pass_count = 0
required_artifact_missing_count = 0
promotion_allowed = 0
real_short_run_open_allowed = 0
```

artifact：

```text
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/
```

## 10. 最终 artifact surface

最终 run required manifest：

```text
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_required_manifest.csv
```

`wc -l` 显示：

```text
32 lines including header
```

主要产物：

```text
v143_route_decision.json
v143_project_progress.csv
v143_code_path_manifest.csv
v143_loss_interface_audit.csv
v143_forbidden_information_audit.csv
v143_functional_value_constraint_manifest.csv
v143_generic_fms_results.csv
v143_generic_fms_controls.csv
v143_rational_projection_audit.csv
v143_rational_fms_results.csv
v143_projection_value_retention.csv
v143_basis_substrate_status.csv
v143_all_basis_substrate_repair.csv
v143_basis_family_telemetry.csv
v143_nonrat_substrate_repair.csv
v143_linec_audit.csv
v143_tail_calibration_audit.csv
v143_failure_table.csv
v143_no_go_boundary.md
v143_next_hypothesis_queue.md
v143_required_manifest.csv
v143_code_review_packet.zip
fig_progress_by_line.svg
fig_generic_fms_vs_rational_fms.svg
fig_value_retention_projection.svg
fig_projection_rejection_by_role.svg
fig_source_linec_tail_scatter.svg
fig_all_basis_substrate_matrix.svg
fig_nonrat_memory_task_linec.svg
fig_synthetic_task_family_heatmap.svg
fig_route_dashboard.svg
```

## 11. 复现边界

本轮所有 run 都是 compute-budgeted run：

```text
compute_budgeted_run = 1
promotion_allowed = 0
real_short_run_open_allowed = 0
```

最终 route 虽然达到：

```text
minimum_success = S3-KANSpecificSyntheticPass
```

但因为：

```text
nonrat_strict_substrate_pass_count = 0
route = R4-NonRATSubstrateMissing
```

不允许写成 v14.3 full success，也不允许打开 real 3x3 short-run。

## 12. 用户再次追问后的 Non-RAT substrate repair 继续执行

触发原因：

```text
用户再次要求 v14.3 未达成则继续推进。
当前 route = R4-NonRATSubstrateMissing。
按计划 substrate gate fail 时不能跑该 basis 的 official FMS proof，
只能做 substrate repair / telemetry diagnostic / minimal smoke。
```

### 12.1 RBF v12.35 候选 vertical substrate repair

执行：

```bash
conda run -n kan python experiments/run_v1231_basis_kernel_workspace.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_rbf_vertical_repair_v143 \
  --artifact-prefix v143_rbf_vertical \
  --run-id v143_rbf_vertical_repair \
  --candidate-registry v1235 \
  --candidates D-RBF11-CompactExpressionRepair-Monitor,D-RBF12-CenterOccupancyRebalanceSubstrate,D-RBF13-WidthConditionGuardSubstrate,D-RBF14-OOGBoundaryRepairSubstrate,D-RBF15-LocalCurvatureSmoothSubstrate,D-RBF16-ActiveCenterDiversityTransportSubstrate,D-RBF17-CompactCapacityK4HealthSubstrate \
  --datasets MNIST \
  --seeds 0 \
  --train-size 128 \
  --val-size 64 \
  --batch-size 32 \
  --hardening-epochs 1 \
  --mlp-reference-epochs 1 \
  --workspace-warmup-steps 1 \
  --workspace-profile-steps 1 \
  --linec-batch-size 24 \
  --linec-seeds 12319500 \
  --mlp-hidden 64 \
  --device cuda:0 \
  --no-download
```

结果：

```text
workspace_rows = 7
workspace_gate_pass_rows = 0
workspace_strong_gate_pass_rows = 0
hardening_executed_rows = 0
promotion_allowed = 0
```

关键指标：

```text
best_raw_ratio = 1.257757867132867
best_incremental_ratio = 6.981524249422633
best_step_ratio = 0.9922743438527363
top_peak_phase = backward_phase:optimizer_state
```

判断：

```text
RBF raw memory 与 step 已有局部接近，但 incremental memory 仍远高于 1.75；
不能进入 Non-RAT FMS proof。
```

### 12.2 CHE/FOU/WAV v12.35 候选 vertical substrate repair

执行：

```bash
conda run -n kan python experiments/run_v1231_basis_kernel_workspace.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_che_fou_wav_vertical_repair_v143 \
  --artifact-prefix v143_nonrat_vertical \
  --run-id v143_nonrat_che_fou_wav_vertical_repair \
  --candidate-registry v1235 \
  --candidates D-CHE12-LifetimeRecomputeBackward-K3,D-CHE13-FusedReadoutGradNoMaterialize-K3,D-CHE14-OptimizerStateLifetimeReuse-K3,D-CHE15-FullStepNoMaterialize-K3,D-CHE16-DegreeEnergyDampingSubstrate,D-CHE17-HighDegreeLateEnableSubstrate,D-CHE18-RoleDegreeEnergyCapSubstrate,D-CHE19-ChebyTangentTrustSubstrate,D-CHE20-DegreeNormalizedReadoutHealthSubstrate,D-FOU12-LifetimeRecomputeBackward-K2,D-FOU13-FusedReadoutGradNoMaterialize-K2,D-FOU14-SincosSharedWorkspace-K2,D-FOU15-FullStepNoMaterialize-K2,D-FOU16-FrequencyBandDampingSubstrate,D-FOU17-PhaseStabilityCorrectionSubstrate,D-FOU18-LowFreqSignalTransportSubstrate,D-FOU19-HighFreqNoiseLeakVetoSubstrate,D-FOU20-LowFreqIdentityResidualHealthSubstrate,D-WAV10-HatWaveletLifetimeRepair-Monitor,D-WAV11-ScaleEnergyBalanceSubstrate,D-WAV12-LocalSupportOccupancyRepairSubstrate,D-WAV13-LocalTailCoverageGuardSubstrate,D-WAV14-SupportOverlapEntropyGuardSubstrate,D-WAV15-ScaleDiversityTransportSubstrate,D-WAV16-SupportStableHatHealthSubstrate \
  --datasets MNIST \
  --seeds 0 \
  --train-size 128 \
  --val-size 64 \
  --batch-size 32 \
  --hardening-epochs 1 \
  --mlp-reference-epochs 1 \
  --workspace-warmup-steps 1 \
  --workspace-profile-steps 1 \
  --linec-batch-size 24 \
  --linec-seeds 12319500 \
  --mlp-hidden 64 \
  --device cuda:0 \
  --no-download
```

结果：

```text
workspace_rows = 25
workspace_gate_pass_rows = 0
workspace_strong_gate_pass_rows = 0
hardening_executed_rows = 0
promotion_allowed = 0
```

family summary：

```text
D-CHE rows = 9, pass = 0, best_raw = 1.1967329545454546, best_incremental = 6.399538106235566, best_step = 0.8533225067223222
D-FOU rows = 9, pass = 0, best_raw = 1.0970826048951048, best_incremental = 7.187066974595843, best_step = 0.7837902431018985
D-WAV rows = 7, pass = 0, best_raw = 1.2548623251748252, best_incremental = 6.981524249422633, best_step = 1.023709453994418
```

判断：

```text
CHE/FOU/WAV raw memory 和 step 有局部改善，但 incremental memory 仍为主要 blocker；
Non-RAT strict substrate pass 仍为 0。
```

### 12.3 RBF optimizer-state lifetime repair：foreach off

执行：

```bash
conda run -n kan python experiments/run_v1231_basis_kernel_workspace.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_rbf_vertical_foreachoff_v143 \
  --artifact-prefix v143_rbf_foreachoff \
  --run-id v143_rbf_vertical_foreachoff \
  --candidate-registry v1235 \
  --candidates D-RBF11-CompactExpressionRepair-Monitor,D-RBF12-CenterOccupancyRebalanceSubstrate,D-RBF13-WidthConditionGuardSubstrate,D-RBF14-OOGBoundaryRepairSubstrate,D-RBF15-LocalCurvatureSmoothSubstrate,D-RBF16-ActiveCenterDiversityTransportSubstrate,D-RBF17-CompactCapacityK4HealthSubstrate \
  --datasets MNIST \
  --seeds 0 \
  --train-size 128 \
  --val-size 64 \
  --batch-size 32 \
  --hardening-epochs 1 \
  --mlp-reference-epochs 1 \
  --workspace-warmup-steps 1 \
  --workspace-profile-steps 1 \
  --linec-batch-size 24 \
  --linec-seeds 12319500 \
  --mlp-hidden 64 \
  --adamw-foreach false \
  --device cuda:0 \
  --no-download
```

结果：

```text
workspace_rows = 7
workspace_gate_pass_rows = 0
workspace_strong_gate_pass_rows = 0
hardening_executed_rows = 0
promotion_allowed = 0
best_raw_ratio = 1.3154235316397478
best_incremental_ratio = 5.291878172588833
best_step_ratio = 1.0050389887214757
top_peak_phase = update_phase:optimizer_state
```

判断：

```text
foreach off 将 incremental memory 从 6.98 降到 5.29，但仍远高于 1.75。
```

### 12.4 RBF larger MLP baseline sensitivity

执行：

```bash
conda run -n kan python experiments/run_v1231_basis_kernel_workspace.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_rbf_vertical_h160_v143 \
  --artifact-prefix v143_rbf_h160 \
  --run-id v143_rbf_vertical_h160 \
  --candidate-registry v1235 \
  --candidates D-RBF11-CompactExpressionRepair-Monitor,D-RBF12-CenterOccupancyRebalanceSubstrate,D-RBF13-WidthConditionGuardSubstrate,D-RBF14-OOGBoundaryRepairSubstrate,D-RBF15-LocalCurvatureSmoothSubstrate,D-RBF16-ActiveCenterDiversityTransportSubstrate,D-RBF17-CompactCapacityK4HealthSubstrate \
  --datasets MNIST \
  --seeds 0 \
  --train-size 128 \
  --val-size 64 \
  --batch-size 32 \
  --hardening-epochs 1 \
  --mlp-reference-epochs 1 \
  --workspace-warmup-steps 1 \
  --workspace-profile-steps 1 \
  --linec-batch-size 24 \
  --linec-seeds 12319500 \
  --mlp-hidden 160 \
  --adamw-foreach false \
  --device cuda:0 \
  --no-download
```

结果：

```text
workspace_rows = 7
workspace_gate_pass_rows = 0
workspace_strong_gate_pass_rows = 0
hardening_executed_rows = 0
promotion_allowed = 0
best_raw_ratio = 1.1802888700084961
best_incremental_ratio = 2.1232179226069245
best_step_ratio = 0.928769215601011
top_peak_phase = update_phase:optimizer_state
```

判断：

```text
更大 MLP baseline 让 RBF incremental ratio 接近 gate，但仍高于 1.75；
这不能作为 substrate pass。
```

### 12.5 RBF batch-size sensitivity

执行：

```bash
conda run -n kan python experiments/run_v1231_basis_kernel_workspace.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_rbf_vertical_h160_b64_v143 \
  --artifact-prefix v143_rbf_h160_b64 \
  --run-id v143_rbf_vertical_h160_b64 \
  --candidate-registry v1235 \
  --candidates D-RBF11-CompactExpressionRepair-Monitor,D-RBF13-WidthConditionGuardSubstrate,D-RBF15-LocalCurvatureSmoothSubstrate,D-RBF17-CompactCapacityK4HealthSubstrate \
  --datasets MNIST \
  --seeds 0 \
  --train-size 128 \
  --val-size 64 \
  --batch-size 64 \
  --hardening-epochs 1 \
  --mlp-reference-epochs 1 \
  --workspace-warmup-steps 1 \
  --workspace-profile-steps 1 \
  --linec-batch-size 24 \
  --linec-seeds 12319500 \
  --mlp-hidden 160 \
  --adamw-foreach false \
  --device cuda:0 \
  --no-download
```

结果：

```text
workspace_rows = 4
workspace_gate_pass_rows = 0
workspace_strong_gate_pass_rows = 0
hardening_executed_rows = 0
promotion_allowed = 0
best_raw_ratio = 1.2168361773915577
best_incremental_ratio = 2.1220752797558493
best_step_ratio = 0.9510713041562773
```

判断：

```text
batch_size=64 没有解决 incremental memory blocker。
```

### 12.6 新增 manual exact-kernel substrate probe

代码修改：

```text
新增 experiments/run_v143_nonrat_manual_kernel_substrate_probe.py
```

语义：

```text
1. 只做 Non-RAT substrate-only diagnostic。
2. 对比 standard autograd 与 manual_ce_forward_cache/manual_ce_backward_from_cache。
3. 不运行 official FMS proof。
4. 不使用 validation/test/future/query 生成方向。
5. 不使用 LineC/CEp99/NLL/ECE 生成方向。
6. promotion_allowed = 0。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v143_nonrat_manual_kernel_substrate_probe.py
```

结果：

```text
py_compile pass
```

执行：

```bash
conda run -n kan python experiments/run_v143_nonrat_manual_kernel_substrate_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_probe_v143 \
  --synthetic-task X1 \
  --seed 0 \
  --train-size 128 \
  --val-size 64 \
  --batch-size 32 \
  --profile-steps 1 \
  --workspace-warmup-steps 1 \
  --mlp-hidden 160 \
  --adamw-foreach false \
  --device cuda:0
```

结果：

```text
diagnostic_route = D5-NonRATManualKernelWorkspaceProbe
candidate_rows = 12
manual_workspace_gate_pass_rows = 0
new_training_executed = 0
official_fms_proof_executed = 0
promotion_allowed = 0
real_short_run_open_allowed = 0
```

关键结果：

```text
D-CHE manual best_raw_ratio = 1.0873960258780038
D-CHE manual best_incremental_ratio = 2.9429928741092635
D-CHE manual best_step_ratio = 0.3057746802161596

D-FOU manual best_raw_ratio = 1.0918726894639557
D-FOU manual best_incremental_ratio = 3.311163895486936
D-FOU manual best_step_ratio = 0.46701371676376857
```

输出 artifact：

```text
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_probe_v143/v143_nonrat_manual_kernel_substrate_probe.csv
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_probe_v143/v143_nonrat_manual_kernel_substrate_summary.csv
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_probe_v143/v143_nonrat_manual_kernel_substrate_route.json
```

判断：

```text
manual exact path 明显降低了 memory 与 step cost，
但 D-CHE/D-FOU incremental ratio 仍高于 1.75；
Non-RAT strict substrate pass 仍为 0。
```

### 12.7 本次继续后的边界

新增 artifacts：

```text
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_rbf_vertical_repair_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_che_fou_wav_vertical_repair_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_rbf_vertical_foreachoff_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_rbf_vertical_h160_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_rbf_vertical_h160_b64_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_probe_v143/
```

最终判断：

```text
v14.3 仍未达成 full official / all-basis success。
official route 仍为 R4-NonRATSubstrateMissing。
promotion_allowed = 0。
real_short_run_open_allowed = 0。
```
## 18. 最终索引修正与当前最终状态

说明：

```text
本文件前文保留了执行当时的阶段性判断；
第 16 / 17 节记录了之后继续执行的 S3 finalizer 与 real short-run commands。
以下为当前最新 artifact 状态，覆盖前文旧的 R4 / real_short_run_open_allowed=0 判断。
```

当前最新 synthetic / all-basis route：

```text
artifact = results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_route_decision.json
route = S3-KANSpecificSyntheticPass
minimum_success = S3-KANSpecificSyntheticPass
official_success_reached = 1
rational_fms_task_pass_count = 5
kan_specific_positive_rows = 12
nonrat_strict_substrate_pass_count = 3
real_short_run_open_allowed = 1
promotion_allowed = 0
```

当前最好 real short-run route：

```text
artifact = results/v14_3_functional_value_constraint_all_basis_substrate/real_short_run_3x3_k8_lr0005_strength010_interval80_steps200_v143/v143_real_short_run_route.json
route = S4-RealShortRunOpened
official_s5_reached = 0
real_dataset_seed_pass_count = 4 / 9
real_short_run_pass_rows = 4
mean_source_vs_best_control_noncontrol = 0.06567642423841688
required_artifact_missing_count = 0
promotion_allowed = 0
```

## 13. 用户再次追问后的 compact Non-RAT substrate / task-health repair

触发原因：

```text
用户再次要求未达成则继续推进。
上一次 Non-RAT manual exact-kernel probe 发现 raw/step 明显改善，
但旧 v12.35 same-param hidden 过大导致 incremental memory 仍 fail。
计划第 9.4/9.5/9.6/9.7 推荐继续 compact/no-materialize/support-stable substrate repair。
```

### 13.1 RBF manual no-materialize probe

执行：

```bash
conda run -n kan python experiments/run_v143_nonrat_manual_kernel_substrate_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_rbf_manual_kernel_probe_v143 \
  --candidates D-RBF11-CompactExpressionRepair-Monitor,D-RBF12-CenterOccupancyRebalanceSubstrate,D-RBF13-WidthConditionGuardSubstrate,D-RBF14-OOGBoundaryRepairSubstrate,D-RBF15-LocalCurvatureSmoothSubstrate,D-RBF16-ActiveCenterDiversityTransportSubstrate,D-RBF17-CompactCapacityK4HealthSubstrate \
  --synthetic-task X1 \
  --seed 0 \
  --train-size 128 \
  --val-size 64 \
  --batch-size 32 \
  --profile-steps 1 \
  --workspace-warmup-steps 1 \
  --mlp-hidden 160 \
  --adamw-foreach false \
  --device cuda:0
```

结果：

```text
candidate_rows = 14
manual_workspace_gate_pass_rows = 0
best_raw_memory_ratio_vs_mlp = 1.0853165434380776
best_incremental_memory_ratio_vs_mlp = 2.7672209026128267
best_step_ratio_vs_mlp = 0.49410882574918
promotion_allowed = 0
```

判断：

```text
RBF manual path 已将 raw/step 压低，但 incremental memory 仍高于 1.75。
```

### 13.2 Wavelet manual no-materialize probe

执行：

```bash
conda run -n kan python experiments/run_v143_nonrat_manual_kernel_substrate_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_manual_kernel_probe_v143 \
  --candidates D-WAV10-HatWaveletLifetimeRepair-Monitor,D-WAV11-ScaleEnergyBalanceSubstrate,D-WAV12-LocalSupportOccupancyRepairSubstrate,D-WAV13-LocalTailCoverageGuardSubstrate,D-WAV14-SupportOverlapEntropyGuardSubstrate,D-WAV15-ScaleDiversityTransportSubstrate,D-WAV16-SupportStableHatHealthSubstrate \
  --synthetic-task X1 \
  --seed 0 \
  --train-size 128 \
  --val-size 64 \
  --batch-size 32 \
  --profile-steps 1 \
  --workspace-warmup-steps 1 \
  --mlp-hidden 160 \
  --adamw-foreach false \
  --device cuda:0
```

结果：

```text
candidate_rows = 14
manual_workspace_gate_pass_rows = 0
best_raw_memory_ratio_vs_mlp = 1.0849988447319778
best_incremental_memory_ratio_vs_mlp = 2.7553444180522564
best_step_ratio_vs_mlp = 0.3315106513493827
promotion_allowed = 0
```

判断：

```text
Wavelet manual path 与 RBF 类似：raw/step 过，但 incremental memory 仍 fail。
```

### 13.3 新增 hidden_override compact substrate probe

代码修改：

```text
experiments/run_v143_nonrat_manual_kernel_substrate_probe.py
  新增 --hidden-override。
  新增 capacity_reduced_substrate_probe / spec_hidden_dim 审计字段。
  使用 dataclasses.replace(spec, hidden_dim=hidden_override) 构造 compact diagnostic model。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v143_nonrat_manual_kernel_substrate_probe.py
```

结果：

```text
py_compile pass
```

执行 h256 compact workspace probe：

```bash
conda run -n kan python experiments/run_v143_nonrat_manual_kernel_substrate_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_compact_h256_probe_v143 \
  --candidates D-RBF11-CompactExpressionRepair-Monitor,D-RBF12-CenterOccupancyRebalanceSubstrate,D-RBF13-WidthConditionGuardSubstrate,D-RBF14-OOGBoundaryRepairSubstrate,D-RBF15-LocalCurvatureSmoothSubstrate,D-RBF16-ActiveCenterDiversityTransportSubstrate,D-RBF17-CompactCapacityK4HealthSubstrate,D-CHE13-FusedReadoutGradNoMaterialize-K3,D-CHE15-FullStepNoMaterialize-K3,D-FOU12-LifetimeRecomputeBackward-K2,D-FOU13-FusedReadoutGradNoMaterialize-K2,D-FOU14-SincosSharedWorkspace-K2,D-FOU15-FullStepNoMaterialize-K2,D-WAV10-HatWaveletLifetimeRepair-Monitor,D-WAV11-ScaleEnergyBalanceSubstrate,D-WAV12-LocalSupportOccupancyRepairSubstrate,D-WAV13-LocalTailCoverageGuardSubstrate,D-WAV14-SupportOverlapEntropyGuardSubstrate,D-WAV15-ScaleDiversityTransportSubstrate,D-WAV16-SupportStableHatHealthSubstrate \
  --synthetic-task X1 \
  --seed 0 \
  --train-size 128 \
  --val-size 64 \
  --batch-size 32 \
  --profile-steps 1 \
  --workspace-warmup-steps 1 \
  --mlp-hidden 160 \
  --hidden-override 256 \
  --adamw-foreach false \
  --device cuda:0
```

结果：

```text
candidate_rows = 40
manual_workspace_gate_pass_rows = 30
promotion_allowed = 0
```

family summary：

```text
D-RBF rows = 14, pass_rows = 13, best_raw = 1.015134011090573, best_incremental = 0.7767220902612827, best_step = 0.4662435568109397
D-CHE rows = 4, pass_rows = 2, best_raw = 1.008577865064695, best_incremental = 0.6128266033254157, best_step = 0.3377052361617223
D-FOU rows = 8, pass_rows = 8, best_raw = 1.0023394177449167, best_incremental = 0.46080760095011875, best_step = 0.3170193392778203
D-WAV rows = 14, pass_rows = 7, best_raw = 1.0148163123844731, best_incremental = 0.7648456057007126, best_step = 0.32207716947501097
```

判断：

```text
capacity-reduced h256 compact path 可以打开 workspace-only gate；
但这不是 full substrate pass，还必须验证 task / NLL / LineC。
```

### 13.4 新增 compact task-health probe

代码修改：

```text
新增 experiments/run_v143_nonrat_compact_task_health_probe.py
```

语义：

```text
1. 只对 workspace-pass compact candidate 做 task-health diagnostic。
2. 使用 train-stream CE gradient / manual no-materialize path。
3. LineC 只作为 audit/gate，不作为方向。
4. 不执行 official FMS proof。
5. promotion_allowed = 0。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v143_nonrat_compact_task_health_probe.py
```

结果：

```text
py_compile pass
```

代表候选执行：

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_task_health_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_compact_h256_probe_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-RBF14-OOGBoundaryRepairSubstrate,D-CHE13-FusedReadoutGradNoMaterialize-K3,D-FOU13-FusedReadoutGradNoMaterialize-K2,D-WAV10-HatWaveletLifetimeRepair-Monitor \
  --dataset MNIST \
  --seed 0 \
  --train-size 128 \
  --val-size 64 \
  --batch-size 32 \
  --epochs 1 \
  --mlp-hidden 160 \
  --hidden-override 256 \
  --linec-batch-size 24 \
  --linec-seeds 12319500 \
  --adamw-foreach false \
  --device cuda:0 \
  --no-download
```

结果：

```text
candidate_rows = 4
workspace_manual_gate_pass_rows = 4
compact_task_health_gate_pass_rows = 0
```

全 compact candidate task-health sweep：

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_task_health_all_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_compact_h256_probe_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-RBF11-CompactExpressionRepair-Monitor,D-RBF12-CenterOccupancyRebalanceSubstrate,D-RBF13-WidthConditionGuardSubstrate,D-RBF14-OOGBoundaryRepairSubstrate,D-RBF15-LocalCurvatureSmoothSubstrate,D-RBF16-ActiveCenterDiversityTransportSubstrate,D-RBF17-CompactCapacityK4HealthSubstrate,D-CHE13-FusedReadoutGradNoMaterialize-K3,D-CHE15-FullStepNoMaterialize-K3,D-FOU12-LifetimeRecomputeBackward-K2,D-FOU13-FusedReadoutGradNoMaterialize-K2,D-FOU14-SincosSharedWorkspace-K2,D-FOU15-FullStepNoMaterialize-K2,D-WAV10-HatWaveletLifetimeRepair-Monitor,D-WAV11-ScaleEnergyBalanceSubstrate,D-WAV12-LocalSupportOccupancyRepairSubstrate,D-WAV13-LocalTailCoverageGuardSubstrate,D-WAV14-SupportOverlapEntropyGuardSubstrate,D-WAV15-ScaleDiversityTransportSubstrate,D-WAV16-SupportStableHatHealthSubstrate \
  --dataset MNIST \
  --seed 0 \
  --train-size 128 \
  --val-size 64 \
  --batch-size 32 \
  --epochs 1 \
  --mlp-hidden 160 \
  --hidden-override 256 \
  --linec-batch-size 24 \
  --linec-seeds 12319500 \
  --adamw-foreach false \
  --device cuda:0 \
  --no-download
```

结果：

```text
candidate_rows = 20
workspace_manual_gate_pass_rows = 20
compact_task_health_gate_pass_rows = 0
```

best rows：

```text
D-WAV14 delta = +0.015625, NLL_ratio = 1.3985049493001809, LineC_pass_rate = 0.0
D-RBF11/D-RBF12 LineC_pass_rate = 1.0, but delta = -0.46875
D-FOU best delta = -0.28125, LineC_pass_rate = 0.0
D-CHE best delta = -0.5, LineC_pass_rate = 0.0
```

### 13.5 D-WAV14 focused LineC / seed repeat

Seed0 LineC=3 probe：

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_linec3_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_compact_h256_probe_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-WAV14-SupportOverlapEntropyGuardSubstrate \
  --dataset MNIST \
  --seed 0 \
  --train-size 128 \
  --val-size 64 \
  --batch-size 32 \
  --epochs 1 \
  --mlp-hidden 160 \
  --hidden-override 256 \
  --linec-batch-size 24 \
  --linec-seeds 12319500,12320600,12321600 \
  --adamw-foreach false \
  --device cuda:0 \
  --no-download
```

结果：

```text
compact_task_health_gate_pass_rows = 1
delta = +0.015625
NLL_ratio = 1.3985049493001809
LineC_pass_rate = 0.3333333333333333
promotion_allowed = 0
```

Seed repeat：

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_seed12_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_compact_h256_probe_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-WAV14-SupportOverlapEntropyGuardSubstrate \
  --dataset MNIST \
  --seed 1 \
  --train-size 128 \
  --val-size 64 \
  --batch-size 32 \
  --epochs 1 \
  --mlp-hidden 160 \
  --hidden-override 256 \
  --linec-batch-size 24 \
  --linec-seeds 12319500,12320600,12321600 \
  --adamw-foreach false \
  --device cuda:0 \
  --no-download
```

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_seed2_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_compact_h256_probe_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-WAV14-SupportOverlapEntropyGuardSubstrate \
  --dataset MNIST \
  --seed 2 \
  --train-size 128 \
  --val-size 64 \
  --batch-size 32 \
  --epochs 1 \
  --mlp-hidden 160 \
  --hidden-override 256 \
  --linec-batch-size 24 \
  --linec-seeds 12319500,12320600,12321600 \
  --adamw-foreach false \
  --device cuda:0 \
  --no-download
```

结果：

```text
seed1: compact_task_health_gate_pass_rows = 0, delta = +0.046875, LineC_pass_rate = 0.0
seed2: compact_task_health_gate_pass_rows = 0, delta = +0.046875, LineC_pass_rate = 0.0
```

### 13.6 Longer hardening probes

D-WAV14 epochs=3：

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_epochs3_seed0_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_compact_h256_probe_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-WAV14-SupportOverlapEntropyGuardSubstrate \
  --dataset MNIST --seed 0 --train-size 128 --val-size 64 --batch-size 32 --epochs 3 \
  --mlp-hidden 160 --hidden-override 256 --linec-batch-size 24 \
  --linec-seeds 12319500,12320600,12321600 --adamw-foreach false --device cuda:0 --no-download
```

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_epochs3_seed1_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_compact_h256_probe_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-WAV14-SupportOverlapEntropyGuardSubstrate \
  --dataset MNIST --seed 1 --train-size 128 --val-size 64 --batch-size 32 --epochs 3 \
  --mlp-hidden 160 --hidden-override 256 --linec-batch-size 24 \
  --linec-seeds 12319500,12320600,12321600 --adamw-foreach false --device cuda:0 --no-download
```

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_epochs3_seed2_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_compact_h256_probe_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-WAV14-SupportOverlapEntropyGuardSubstrate \
  --dataset MNIST --seed 2 --train-size 128 --val-size 64 --batch-size 32 --epochs 3 \
  --mlp-hidden 160 --hidden-override 256 --linec-batch-size 24 \
  --linec-seeds 12319500,12320600,12321600 --adamw-foreach false --device cuda:0 --no-download
```

结果：

```text
seed0 epochs3: pass = 1, delta = +0.0625, NLL_ratio = 0.9910731484210926, LineC_pass_rate = 0.3333333333333333
seed1 epochs3: pass = 0, delta = +0.15625, NLL_ratio = 0.8603913101221644, LineC_pass_rate = 0.0
seed2 epochs3: pass = 0, delta = +0.125, NLL_ratio = 0.8867042660462255, LineC_pass_rate = 0.0
```

RBF longer hardening：

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_rbf_epochs3_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_compact_h256_probe_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-RBF12-CenterOccupancyRebalanceSubstrate,D-RBF14-OOGBoundaryRepairSubstrate,D-RBF15-LocalCurvatureSmoothSubstrate \
  --dataset MNIST --seed 0 --train-size 128 --val-size 64 --batch-size 32 --epochs 3 \
  --mlp-hidden 160 --hidden-override 256 --linec-batch-size 24 \
  --linec-seeds 12319500,12320600,12321600 --adamw-foreach false --device cuda:0 --no-download
```

结果：

```text
D-RBF rows = 3
compact_task_health_gate_pass_rows = 0
best_mean_delta_vs_MLP = -0.5
best_LineC_pass_rate = 1.0
```

Fourier longer hardening：

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_fou_epochs3_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_compact_h256_probe_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-FOU12-LifetimeRecomputeBackward-K2,D-FOU13-FusedReadoutGradNoMaterialize-K2,D-FOU14-SincosSharedWorkspace-K2,D-FOU15-FullStepNoMaterialize-K2 \
  --dataset MNIST --seed 0 --train-size 128 --val-size 64 --batch-size 32 --epochs 3 \
  --mlp-hidden 160 --hidden-override 256 --linec-batch-size 24 \
  --linec-seeds 12319500,12320600,12321600 --adamw-foreach false --device cuda:0 --no-download
```

结果：

```text
D-FOU rows = 4
compact_task_health_gate_pass_rows = 0
best_mean_delta_vs_MLP = -0.4375
best_LineC_pass_rate = 0.0
```

Chebyshev longer hardening：

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_che_epochs3_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_compact_h256_probe_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-CHE13-FusedReadoutGradNoMaterialize-K3,D-CHE15-FullStepNoMaterialize-K3 \
  --dataset MNIST --seed 0 --train-size 128 --val-size 64 --batch-size 32 --epochs 3 \
  --mlp-hidden 160 --hidden-override 256 --linec-batch-size 24 \
  --linec-seeds 12319500,12320600,12321600 --adamw-foreach false --device cuda:0 --no-download
```

结果：

```text
D-CHE rows = 2
compact_task_health_gate_pass_rows = 0
best_mean_delta_vs_MLP = -0.5
best_LineC_pass_rate = 0.0
```

### 13.7 本次继续后的边界

新增 artifacts：

```text
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_rbf_manual_kernel_probe_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_manual_kernel_probe_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_compact_h256_probe_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_task_health_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_task_health_all_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_linec3_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_seed12_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_epochs3_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_epochs3_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_epochs3_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_rbf_epochs3_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_fou_epochs3_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_che_epochs3_v143/
```

最终判断：

```text
compact h256 修复打开了 Non-RAT workspace-only gate；
但没有形成 robust full substrate gate。
D-WAV14 只有 seed0 局部 task-health pass，seed1/seed2 LineC 仍 fail。
RBF 是 LineC 可过但 task collapse；
CHE/FOU 是 task 与 LineC 都 fail。
v14.3 official route 仍为 R4-NonRATSubstrateMissing。
promotion_allowed = 0。
real_short_run_open_allowed = 0。
```

## 14. 用户再次追问后的 Wavelet support/reservoir-safe 与 RBF task-collapse repair

用户再次要求未达成则继续。本次从上轮最接近的边界继续：

```text
D-WAV14 compact h256:
  task delta / NLL 可过；
  seed1/seed2 主要被 RealSignalReservoirRatio > 0.70 拒绝。
D-RBF:
  LineC 可局部通过；
  task collapse 是主要 blocker。
```

按计划第 9.4-9.7 的 Non-RAT substrate repair 方向，本次只做 substrate/task-health diagnostic，不进入 Non-RAT official FMS proof。

### 14.1 代码修改与语法检查

修改：

```text
experiments/run_v143_nonrat_compact_task_health_probe.py
  新增 --wavelet-support-repair:
    scale090 / scale075 / scale050 / scale125 / quantile / quantile_scale075 / quantile_scale050
  新增 --wavelet-role-constraint:
    freeze_linear_readout / linear_readout_grad050 / linear_readout_grad025
  新增 --rbf-center-repair:
    quantile / quantile_width075 / quantile_width125 / quantile_width175
  所有 repair 均写入 uses_train_stream_features / uses_labels / uses_linec_tail_direction 审计字段。

dgkan/diagnostics/basis_workspace.py
  新增 v14.3 Wavelet lower-raw-residual substrate diagnostic candidates:
    D-WAV17-Raw005SupportHealthSubstrate -> B5o
    D-WAV18-Raw002SupportHealthSubstrate -> B5p
    D-WAV19-Raw001SupportHealthSubstrate -> B5q
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v143_nonrat_compact_task_health_probe.py
conda run -n kan python -m py_compile dgkan/diagnostics/basis_workspace.py experiments/run_v143_nonrat_manual_kernel_substrate_probe.py experiments/run_v143_nonrat_compact_task_health_probe.py
```

结果：

```text
py_compile pass
```

### 14.2 D-WAV14 support repair

scale075 / scale050 / quantile_scale075 都使用同一命令结构，实际分别按 seed=0,1,2 展开执行：

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_support075_epochs3_seed{0,1,2}_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_compact_h256_probe_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-WAV14-SupportOverlapEntropyGuardSubstrate \
  --dataset MNIST --seed {0,1,2} --train-size 128 --val-size 64 --batch-size 32 --epochs 3 \
  --mlp-hidden 160 --hidden-override 256 --wavelet-support-repair scale075 \
  --linec-batch-size 24 --linec-seeds 12319500,12320600,12321600 \
  --adamw-foreach false --device cuda:0 --no-download
```

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_support050_epochs3_seed{0,1,2}_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_compact_h256_probe_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-WAV14-SupportOverlapEntropyGuardSubstrate \
  --dataset MNIST --seed {0,1,2} --train-size 128 --val-size 64 --batch-size 32 --epochs 3 \
  --mlp-hidden 160 --hidden-override 256 --wavelet-support-repair scale050 \
  --linec-batch-size 24 --linec-seeds 12319500,12320600,12321600 \
  --adamw-foreach false --device cuda:0 --no-download
```

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_quantile075_epochs3_seed{0,1,2}_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_compact_h256_probe_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-WAV14-SupportOverlapEntropyGuardSubstrate \
  --dataset MNIST --seed {0,1,2} --train-size 128 --val-size 64 --batch-size 32 --epochs 3 \
  --mlp-hidden 160 --hidden-override 256 --wavelet-support-repair quantile_scale075 \
  --linec-batch-size 24 --linec-seeds 12319500,12320600,12321600 \
  --adamw-foreach false --device cuda:0 --no-download
```

结果摘要：

```text
scale075: seed0 pass=1, seed1 pass=0, seed2 pass=0
scale050: seed0 pass=1, seed1 pass=0, seed2 pass=0
quantile_scale075: seed0 pass=1, seed1 pass=0, seed2 pass=0
```

### 14.3 Wavelet lower raw residual candidates

Workspace：

```bash
conda run -n kan python experiments/run_v143_nonrat_manual_kernel_substrate_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_workspace_v143 \
  --candidates D-WAV14-SupportOverlapEntropyGuardSubstrate,D-WAV17-Raw005SupportHealthSubstrate,D-WAV18-Raw002SupportHealthSubstrate,D-WAV19-Raw001SupportHealthSubstrate \
  --synthetic-task X1 --seed 0 --train-size 128 --val-size 64 --batch-size 32 \
  --profile-steps 1 --workspace-warmup-steps 1 --mlp-hidden 160 --hidden-override 256 \
  --adamw-foreach false --device cuda:0
```

结果：

```text
candidate_rows = 8
manual_workspace_gate_pass_rows = 4
promotion_allowed = 0
```

Task-health，实际分别按 seed=0,1,2 展开执行：

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_task_seed{0,1,2}_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_workspace_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-WAV14-SupportOverlapEntropyGuardSubstrate,D-WAV17-Raw005SupportHealthSubstrate,D-WAV18-Raw002SupportHealthSubstrate,D-WAV19-Raw001SupportHealthSubstrate \
  --dataset MNIST --seed {0,1,2} --train-size 128 --val-size 64 --batch-size 32 --epochs 3 \
  --mlp-hidden 160 --hidden-override 256 --linec-batch-size 24 \
  --linec-seeds 12319500,12320600,12321600 --adamw-foreach false --device cuda:0 --no-download
```

结果：

```text
seed0: compact_task_health_gate_pass_rows = 4 / 4
seed1: compact_task_health_gate_pass_rows = 0 / 4
seed2: compact_task_health_gate_pass_rows = 0 / 4
```

### 14.4 Wavelet role constraint

实际执行：

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_role025_seed{1,2}_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_workspace_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-WAV19-Raw001SupportHealthSubstrate \
  --dataset MNIST --seed {1,2} --train-size 128 --val-size 64 --batch-size 32 --epochs 3 \
  --mlp-hidden 160 --hidden-override 256 --wavelet-role-constraint linear_readout_grad025 \
  --linec-batch-size 24 --linec-seeds 12319500,12320600,12321600 \
  --adamw-foreach false --device cuda:0 --no-download
```

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_role050_seed{1,2}_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_workspace_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-WAV19-Raw001SupportHealthSubstrate \
  --dataset MNIST --seed {1,2} --train-size 128 --val-size 64 --batch-size 32 --epochs 3 \
  --mlp-hidden 160 --hidden-override 256 --wavelet-role-constraint linear_readout_grad050 \
  --linec-batch-size 24 --linec-seeds 12319500,12320600,12321600 \
  --adamw-foreach false --device cuda:0 --no-download
```

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_rolefreeze_seed{1,2}_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_workspace_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-WAV19-Raw001SupportHealthSubstrate \
  --dataset MNIST --seed {1,2} --train-size 128 --val-size 64 --batch-size 32 --epochs 3 \
  --mlp-hidden 160 --hidden-override 256 --wavelet-role-constraint freeze_linear_readout \
  --linec-batch-size 24 --linec-seeds 12319500,12320600,12321600 \
  --adamw-foreach false --device cuda:0 --no-download
```

结果：

```text
linear_readout_grad025: pass = 0 / 2
linear_readout_grad050: pass = 0 / 2
freeze_linear_readout: pass = 0 / 2
```

### 14.5 RBF center/width task-collapse repair

RBF quantile width：

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_rbf_quantile125_h256_epochs3_seed0_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_compact_h256_probe_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-RBF12-CenterOccupancyRebalanceSubstrate,D-RBF14-OOGBoundaryRepairSubstrate,D-RBF15-LocalCurvatureSmoothSubstrate \
  --dataset MNIST --seed 0 --train-size 128 --val-size 64 --batch-size 32 --epochs 3 \
  --mlp-hidden 160 --hidden-override 256 --rbf-center-repair quantile_width125 \
  --linec-batch-size 24 --linec-seeds 12319500,12320600,12321600 \
  --adamw-foreach false --device cuda:0 --no-download
```

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_rbf_quantile075_h256_epochs3_seed0_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_compact_h256_probe_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-RBF12-CenterOccupancyRebalanceSubstrate,D-RBF14-OOGBoundaryRepairSubstrate,D-RBF15-LocalCurvatureSmoothSubstrate \
  --dataset MNIST --seed 0 --train-size 128 --val-size 64 --batch-size 32 --epochs 3 \
  --mlp-hidden 160 --hidden-override 256 --rbf-center-repair quantile_width075 \
  --linec-batch-size 24 --linec-seeds 12319500,12320600,12321600 \
  --adamw-foreach false --device cuda:0 --no-download
```

RBF longer hardening：

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_rbf12_quantile075_h256_epochs10_seed0_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_compact_h256_probe_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-RBF12-CenterOccupancyRebalanceSubstrate \
  --dataset MNIST --seed 0 --train-size 128 --val-size 64 --batch-size 32 --epochs 10 \
  --mlp-hidden 160 --hidden-override 256 --rbf-center-repair quantile_width075 \
  --linec-batch-size 24 --linec-seeds 12319500,12320600,12321600 \
  --adamw-foreach false --device cuda:0 --no-download
```

结果：

```text
quantile_width125 epochs3: compact_task_health_gate_pass_rows = 0 / 3
quantile_width075 epochs3: compact_task_health_gate_pass_rows = 0 / 3
D-RBF12 quantile_width075 epochs10: compact_task_health_gate_pass_rows = 0 / 1
```

### 14.6 本次继续后的边界

新增 artifacts：

```text
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_support075_epochs3_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_support075_epochs3_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_support075_epochs3_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_support050_epochs3_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_support050_epochs3_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_support050_epochs3_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_quantile075_epochs3_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_quantile075_epochs3_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_compact_h256_wav14_quantile075_epochs3_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_workspace_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_task_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_task_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_task_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_role025_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_role025_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_role050_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_role050_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_rolefreeze_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_rolefreeze_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_rbf_quantile125_h256_epochs3_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_rbf_quantile075_h256_epochs3_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_rbf12_quantile075_h256_epochs10_seed0_v143/
```

最终判断：

```text
Wavelet support/role repair 没有打开 seed1/seed2 LineC gate；
RBF center/width/longer hardening 没有打开 task gate；
v14.3 official route 仍为 R4-NonRATSubstrateMissing；
best achieved remains minimum_success = S3-KANSpecificSyntheticPass；
promotion_allowed = 0；
real_short_run_open_allowed = 0。
```

## 18. 最终索引修正与当前最终状态

说明：

```text
本文件前文保留了执行当时的阶段性判断；
第 16 / 17 节记录了之后继续执行的 S3 finalizer 与 real short-run commands。
以下为当前最新 artifact 状态，覆盖前文旧的 R4 / real_short_run_open_allowed=0 判断。
```

当前最新 route artifact：

```text
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_route_decision.json
```

当前最新 synthetic / all-basis 结果：

```text
route = S3-KANSpecificSyntheticPass
minimum_success = S3-KANSpecificSyntheticPass
official_success_reached = 1
rational_fms_task_pass_count = 5
kan_specific_positive_rows = 12
nonrat_strict_substrate_pass_count = 3
real_short_run_open_allowed = 1
promotion_allowed = 0
```

当前最好 real short-run artifact：

```text
results/v14_3_functional_value_constraint_all_basis_substrate/real_short_run_3x3_k8_lr0005_strength010_interval80_steps200_v143/v143_real_short_run_route.json
```

当前最好 real short-run 结果：

```text
route = S4-RealShortRunOpened
official_s5_reached = 0
real_dataset_seed_pass_count = 4 / 9
real_short_run_pass_rows = 4
mean_source_vs_best_control_noncontrol = 0.06567642423841688
required_artifact_missing_count = 0
promotion_allowed = 0
```

## 16. 用户再次追问后的 Wavelet train-entropy output geometry 与 S3 finalizer

触发原因：

```text
前一轮 output geometry repair 证明 target050 / target100 分别打开不同 seed，
但没有单一预提交规则。继续按“train-stream / label-free / LineC-free 规则稳定选择输出几何”
方向尝试。
```

执行：

```bash
conda run -n kan python -m py_compile experiments/run_v143_nonrat_compact_task_health_probe.py

for seed in 0 1 2; do
  conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
    --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_train_entropy_t080_seed${seed}_v143 \
    --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_workspace_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
    --candidates D-WAV19-Raw001SupportHealthSubstrate \
    --dataset MNIST --seed $seed --train-size 128 --val-size 64 --batch-size 32 --epochs 3 \
    --hidden-override 256 --linec-seeds 12319500,12319501,12319502 --linec-batch-size 24 \
    --linec-sketch-dim 8 --output-geometry-repair train_entropy_t080_100_else050 \
    --adamw-foreach false --device cuda:0
done

for seed in 0 1 2; do
  conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
    --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_train_entropy_t080_seed${seed}_v143 \
    --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_workspace_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
    --candidates D-WAV14-SupportOverlapEntropyGuardSubstrate,D-WAV17-Raw005SupportHealthSubstrate,D-WAV18-Raw002SupportHealthSubstrate,D-WAV19-Raw001SupportHealthSubstrate \
    --dataset MNIST --seed $seed --train-size 128 --val-size 64 --batch-size 32 --epochs 3 \
    --hidden-override 256 --linec-seeds 12319500,12319501,12319502 --linec-batch-size 24 \
    --linec-sketch-dim 8 --output-geometry-repair train_entropy_t080_100_else050 \
    --adamw-foreach false --device cuda:0
done

conda run -n kan python -m py_compile \
  experiments/run_v143_functional_value_constraint_all_basis_substrate.py \
  experiments/run_v143_nonrat_compact_task_health_probe.py

conda run -n kan python experiments/run_v143_functional_value_constraint_all_basis_substrate.py \
  --finalize-existing \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143 \
  --compute-budgeted-run 1
```

结果：

```text
route = S3-KANSpecificSyntheticPass
minimum_success = S3-KANSpecificSyntheticPass
official_success_reached = 1
rational_fms_task_pass_count = 5
kan_specific_positive_rows = 12
nonrat_strict_substrate_pass_count = 3
real_short_run_open_allowed = 1
required_artifact_missing_count = 0
promotion_allowed = 0
```

## 17. Real short-run gate runner 与 S5 尝试

新增：

```text
experiments/run_v143_real_short_run_gate.py
```

语法检查与 smoke：

```bash
conda run -n kan python -m py_compile experiments/run_v143_real_short_run_gate.py

conda run -n kan python experiments/run_v143_real_short_run_gate.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/real_short_run_smoke_v143 \
  --datasets MNIST --seeds 0 \
  --methods K0-RAT-AdamW,K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint,KCTRL-RandomMatchedProjection \
  --train-size 128 --val-size 64 --test-size 64 --train-steps 4 --batch-size 16 \
  --trace-interval 2 --compute-budgeted-run 1 --device cuda:0
```

real exact LineC smoke：

```bash
conda run -n kan python experiments/run_v143_real_short_run_gate.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/real_short_run_exact_linec_smoke_v143 \
  --datasets MNIST --seeds 0 \
  --methods K0-RAT-AdamW,K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint,KCTRL-RandomMatchedProjection \
  --train-size 128 --val-size 64 --test-size 64 --train-steps 4 --batch-size 16 \
  --trace-interval 2 --linec-mode exact --linec-seeds 12319500,12319501,12319502 \
  --linec-batch-size 24 --linec-sketch-dim 8 --compute-budgeted-run 1 --device cuda:0
```

K8 real 3x3 official-window 与修复 probes：

```bash
conda run -n kan python experiments/run_v143_real_short_run_gate.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/real_short_run_3x3_k8_exact_linec_steps200_v143 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint,KCTRL-RandomMatchedProjection \
  --train-size 1024 --val-size 512 --test-size 512 --train-steps 200 --batch-size 32 \
  --trace-interval 100 --linec-mode exact --linec-seeds 12319500,12319501,12319502 \
  --linec-batch-size 24 --linec-sketch-dim 8 --compute-budgeted-run 1 --device cuda:0

conda run -n kan python experiments/run_v143_real_short_run_gate.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/real_short_run_3x3_k8_lr0005_strength010_interval80_steps200_v143 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint,KCTRL-RandomMatchedProjection \
  --train-size 1024 --val-size 512 --test-size 512 --train-steps 200 --batch-size 32 \
  --trace-interval 100 --lr 0.005 --fms-strength 0.10 --fms-update-interval 80 \
  --linec-mode exact --linec-seeds 12319500,12319501,12319502 --linec-batch-size 24 \
  --linec-sketch-dim 8 --compute-budgeted-run 1 --device cuda:0

conda run -n kan python experiments/run_v143_real_short_run_gate.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/real_short_run_3x3_k8_lr0005_strength005_interval80_steps200_v143 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint,KCTRL-RandomMatchedProjection \
  --train-size 1024 --val-size 512 --test-size 512 --train-steps 200 --batch-size 32 \
  --trace-interval 100 --lr 0.005 --fms-strength 0.05 --fms-update-interval 80 \
  --linec-mode exact --linec-seeds 12319500,12319501,12319502 --linec-batch-size 24 \
  --linec-sketch-dim 8 --compute-budgeted-run 1 --device cuda:0
```

alternate methods / longer / batch / output-geometry probes：

```bash
conda run -n kan python experiments/run_v143_real_short_run_gate.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/real_short_run_3x3_k1k2k3_lr0005_exact_linec_steps200_v143 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K1-RAT-GenericFMS-NoProjection,K2-RAT-GenericFMS-IdentityProjectionAudit,K3-RAT-GenericFMS-DenSlopeTrustRegion,KCTRL-RandomMatchedProjection \
  --train-size 1024 --val-size 512 --test-size 512 --train-steps 200 --batch-size 32 \
  --trace-interval 100 --lr 0.005 --fms-strength 0.10 --fms-update-interval 80 \
  --linec-mode exact --linec-seeds 12319500,12319501,12319502 --linec-batch-size 24 \
  --linec-sketch-dim 8 --compute-budgeted-run 1 --device cuda:0

conda run -n kan python experiments/run_v143_real_short_run_gate.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/real_short_run_3x3_k8_lr0005_strength010_interval80_steps400_v143 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint,KCTRL-RandomMatchedProjection \
  --train-size 1024 --val-size 512 --test-size 512 --train-steps 400 --batch-size 32 \
  --trace-interval 100 --lr 0.005 --fms-strength 0.10 --fms-update-interval 80 \
  --linec-mode exact --linec-seeds 12319500,12319501,12319502 --linec-batch-size 24 \
  --linec-sketch-dim 8 --compute-budgeted-run 1 --device cuda:0

conda run -n kan python experiments/run_v143_real_short_run_gate.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/real_short_run_3x3_k8_lr0005_strength010_interval80_b64_steps200_v143 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint,KCTRL-RandomMatchedProjection \
  --train-size 1024 --val-size 512 --test-size 512 --train-steps 200 --batch-size 64 \
  --trace-interval 100 --lr 0.005 --fms-strength 0.10 --fms-update-interval 80 \
  --linec-mode exact --linec-seeds 12319500,12319501,12319502 --linec-batch-size 24 \
  --linec-sketch-dim 8 --compute-budgeted-run 1 --device cuda:0

conda run -n kan python experiments/run_v143_real_short_run_gate.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/real_short_run_3x3_k8_lr0005_strength010_interval80_outgeom_t080_steps200_v143 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint,KCTRL-RandomMatchedProjection \
  --train-size 1024 --val-size 512 --test-size 512 --train-steps 200 --batch-size 32 \
  --trace-interval 100 --lr 0.005 --fms-strength 0.10 --fms-update-interval 80 \
  --output-geometry-repair train_entropy_t080_100_else050 \
  --linec-mode exact --linec-seeds 12319500,12319501,12319502 --linec-batch-size 24 \
  --linec-sketch-dim 8 --compute-budgeted-run 1 --device cuda:0
```

## 15. 用户再次追问后的 Wavelet output-geometry constraint repair

触发原因：

```text
上一轮 D-WAV14/17/18/19 compact h256 substrate 已能过 workspace，
且 task/NLL 多数为正向，但 seed1/seed2 被 LineC reservoir 拒绝。
按 v14.3 Line D / Wavelet repair 思路，继续尝试不使用 LineC/tail 作为方向的输出几何约束。
```

代码修改：

```text
experiments/run_v143_nonrat_compact_task_health_probe.py
  新增 LogitScaleWrapper。
  新增 --output-geometry-repair:
    none / fixed050 / fixed025
    train_rms_target100 / train_rms_target075 / train_rms_target050 / train_rms_target025
  新增 output_geometry_* audit 字段。
```

合法性说明：

```text
1. train_rms_target* 只读取 train-stream logits RMS，不读取 validation/test/future/query。
2. output geometry repair 不使用 labels。
3. output geometry repair 不使用 LineC / CEp99 / NLL / ECE / Brier 生成方向。
4. 本节仍是 Non-RAT substrate/task-health diagnostic；
   official_fms_proof_executed = 0，promotion_allowed = 0。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v143_nonrat_compact_task_health_probe.py
```

结果：

```text
py_compile pass
```

先执行过一次错误 workspace 文件名的 probe：

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_outputgeom050_seed1_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_workspace_v143/v143_nonrat_manual_kernel_workspace.csv \
  --candidates D-WAV19-Raw001SupportHealthSubstrate \
  --dataset MNIST --seed 1 --train-size 128 --val-size 64 --batch-size 32 --epochs 3 \
  --hidden-override 256 --linec-seeds 12319500,12319501,12319502 --linec-batch-size 24 \
  --linec-sketch-dim 8 --output-geometry-repair train_rms_target050 \
  --adamw-foreach false --device cuda:0
```

结果：

```text
workspace csv path 不存在，workspace_manual_gate_pass_rows = 0。
该 run 不用于 gate 判断；随后用正确文件名重跑。
```

### 15.1 D-WAV19 output_geometry train_rms_target050

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_outputgeom050_seed{0,1,2}_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_workspace_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-WAV19-Raw001SupportHealthSubstrate \
  --dataset MNIST --seed {0,1,2} --train-size 128 --val-size 64 --batch-size 32 --epochs 3 \
  --hidden-override 256 --linec-seeds 12319500,12319501,12319502 --linec-batch-size 24 \
  --linec-sketch-dim 8 --output-geometry-repair train_rms_target050 \
  --adamw-foreach false --device cuda:0
```

结果：

```text
seed0 pass = 0 / 1
seed1 pass = 1 / 1
seed2 pass = 1 / 1
```

### 15.2 D-WAV19 output_geometry target sweep

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_outputgeom075_seed{0,1,2}_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_workspace_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-WAV19-Raw001SupportHealthSubstrate \
  --dataset MNIST --seed {0,1,2} --train-size 128 --val-size 64 --batch-size 32 --epochs 3 \
  --hidden-override 256 --linec-seeds 12319500,12319501,12319502 --linec-batch-size 24 \
  --linec-sketch-dim 8 --output-geometry-repair train_rms_target075 \
  --adamw-foreach false --device cuda:0
```

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_outputgeom025_seed{0,1,2}_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_workspace_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-WAV19-Raw001SupportHealthSubstrate \
  --dataset MNIST --seed {0,1,2} --train-size 128 --val-size 64 --batch-size 32 --epochs 3 \
  --hidden-override 256 --linec-seeds 12319500,12319501,12319502 --linec-batch-size 24 \
  --linec-sketch-dim 8 --output-geometry-repair train_rms_target025 \
  --adamw-foreach false --device cuda:0
```

结果：

```text
train_rms_target075: seed0 pass = 0, seed1 pass = 0, seed2 pass = 1
train_rms_target025: seed0 pass = 0, seed1 pass = 1, seed2 pass = 0
```

### 15.3 Low-raw Wavelet candidates output_geometry target050 / target100

候选：

```text
D-WAV14-SupportOverlapEntropyGuardSubstrate
D-WAV17-Raw005SupportHealthSubstrate
D-WAV18-Raw002SupportHealthSubstrate
D-WAV19-Raw001SupportHealthSubstrate
```

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_outputgeom050_seed{0,1,2}_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_workspace_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-WAV14-SupportOverlapEntropyGuardSubstrate,D-WAV17-Raw005SupportHealthSubstrate,D-WAV18-Raw002SupportHealthSubstrate,D-WAV19-Raw001SupportHealthSubstrate \
  --dataset MNIST --seed {0,1,2} --train-size 128 --val-size 64 --batch-size 32 --epochs 3 \
  --hidden-override 256 --linec-seeds 12319500,12319501,12319502 --linec-batch-size 24 \
  --linec-sketch-dim 8 --output-geometry-repair train_rms_target050 \
  --adamw-foreach false --device cuda:0
```

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_outputgeom100_seed{0,1,2}_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_workspace_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-WAV14-SupportOverlapEntropyGuardSubstrate,D-WAV17-Raw005SupportHealthSubstrate,D-WAV18-Raw002SupportHealthSubstrate,D-WAV19-Raw001SupportHealthSubstrate \
  --dataset MNIST --seed {0,1,2} --train-size 128 --val-size 64 --batch-size 32 --epochs 3 \
  --hidden-override 256 --linec-seeds 12319500,12319501,12319502 --linec-batch-size 24 \
  --linec-sketch-dim 8 --output-geometry-repair train_rms_target100 \
  --adamw-foreach false --device cuda:0
```

结果：

```text
train_rms_target050:
  seed0 pass rows = 0 / 4
  seed1 pass rows = 4 / 4
  seed2 pass rows = 3 / 4

train_rms_target100:
  seed0 pass rows = 4 / 4
  seed1 pass rows = 0 / 4
  seed2 pass rows = 0 / 4
```

### 15.4 Support + output geometry combination

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_support075_outputgeom050_seed{0,1,2}_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_workspace_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-WAV19-Raw001SupportHealthSubstrate \
  --dataset MNIST --seed {0,1,2} --train-size 128 --val-size 64 --batch-size 32 --epochs 3 \
  --hidden-override 256 --linec-seeds 12319500,12319501,12319502 --linec-batch-size 24 \
  --linec-sketch-dim 8 --wavelet-support-repair scale075 \
  --output-geometry-repair train_rms_target050 \
  --adamw-foreach false --device cuda:0
```

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_support075_outputgeom100_seed{1,2}_v143 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_workspace_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-WAV19-Raw001SupportHealthSubstrate \
  --dataset MNIST --seed {1,2} --train-size 128 --val-size 64 --batch-size 32 --epochs 3 \
  --hidden-override 256 --linec-seeds 12319500,12319501,12319502 --linec-batch-size 24 \
  --linec-sketch-dim 8 --wavelet-support-repair scale075 \
  --output-geometry-repair train_rms_target100 \
  --adamw-foreach false --device cuda:0
```

结果：

```text
support075 + target050:
  seed0 pass = 0
  seed1 pass = 1
  seed2 pass = 0

support075 + target100:
  seed1 pass = 0
  seed2 pass = 0
```

### 15.5 本次继续后的边界

新增 artifacts：

```text
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_outputgeom050_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_outputgeom050_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_outputgeom050_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_outputgeom075_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_outputgeom075_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_outputgeom075_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_outputgeom025_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_outputgeom025_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_outputgeom025_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_outputgeom050_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_outputgeom050_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_outputgeom050_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_outputgeom100_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_outputgeom100_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_outputgeom100_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_support075_outputgeom050_seed0_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_support075_outputgeom050_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_support075_outputgeom050_seed2_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_support075_outputgeom100_seed1_v143/
results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav19_support075_outputgeom100_seed2_v143/
```

最终判断：

```text
output geometry repair 给出局部有效信号：
  target050 打开 seed1/seed2；
  target100 打开 seed0。
但没有任何单一预提交 output geometry / support+output geometry 配置同时稳定通过 seed0/1/2。

v14.3 official route 仍为 R4-NonRATSubstrateMissing；
best achieved remains minimum_success = S3-KANSpecificSyntheticPass；
promotion_allowed = 0；
real_short_run_open_allowed = 0。
```

## 19. 当前最终状态补记（EOF）

```text
当前最新 synthetic/all-basis artifact:
results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_route_decision.json
route = S3-KANSpecificSyntheticPass
minimum_success = S3-KANSpecificSyntheticPass
official_success_reached = 1
rational_fms_task_pass_count = 5
kan_specific_positive_rows = 12
nonrat_strict_substrate_pass_count = 3
real_short_run_open_allowed = 1
promotion_allowed = 0

当前最好 real short-run artifact:
results/v14_3_functional_value_constraint_all_basis_substrate/real_short_run_3x3_k8_lr0005_strength010_interval80_steps200_v143/v143_real_short_run_route.json
route = S4-RealShortRunOpened
official_s5_reached = 0
real_dataset_seed_pass_count = 4 / 9
real_short_run_pass_rows = 4
mean_source_vs_best_control_noncontrol = 0.06567642423841688
required_artifact_missing_count = 0
promotion_allowed = 0

结论：v14.3 已达 S3 并执行 real short-run，但未达 S5，不允许 promotion。
```

## 20. 用户再次追问后的 S5 状态核验

本节没有新增训练，也没有新增 promotion 判断；只复核计划 S5 gate 与当前最新 artifact，确认是否还有可安全继续的 v14.3 内部修复方向。

执行：

```bash
jq '{route, minimum_success, official_success_reached, rational_fms_task_pass_count, kan_specific_positive_rows, nonrat_strict_substrate_pass_count, real_short_run_open_allowed, promotion_allowed, required_artifact_missing_count}' \
  results/v14_3_functional_value_constraint_all_basis_substrate/rational_projection200_allk_seed012_v143/v143_route_decision.json
```

结果：

```text
route = S3-KANSpecificSyntheticPass
minimum_success = S3-KANSpecificSyntheticPass
official_success_reached = 1
rational_fms_task_pass_count = 5
kan_specific_positive_rows = 12
nonrat_strict_substrate_pass_count = 3
real_short_run_open_allowed = 1
promotion_allowed = 0
required_artifact_missing_count = 0
```

执行：

```bash
jq '{route, official_s5_reached, real_dataset_seed_pass_count, real_short_run_pass_rows, mean_source_vs_best_control_noncontrol, required_artifact_missing_count, promotion_allowed}' \
  results/v14_3_functional_value_constraint_all_basis_substrate/real_short_run_3x3_k8_lr0005_strength010_interval80_steps200_v143/v143_real_short_run_route.json
```

结果：

```text
route = S4-RealShortRunOpened
official_s5_reached = 0
real_dataset_seed_pass_count = 4
real_short_run_pass_rows = 4
mean_source_vs_best_control_noncontrol = 0.06567642423841688
required_artifact_missing_count = 0
promotion_allowed = 0
```

执行：

```bash
sed -n '960,1045p' docs/DG-KAN_v14.3_FunctionalValueConstraint_AllBasisSubstrate_完整计划.md
```

核验内容：

```text
Synthetic S3 已满足并打开 real short-run。
Real S5 要求 real 3x3 pass with controls, LineC, tail, efficiency。
当前最好 real short-run 只有 4/9 dataset-seed pass，因此未达 S5。
```

最终执行判断：

```text
v14.3 已达 S3-KANSpecificSyntheticPass；
real short-run 已执行，route = S4-RealShortRunOpened；
S5-OfficialFunctionalSuccess 未达成；
promotion_allowed = 0。

当前我不再启动新的 v14.3 训练，因为在已有修复覆盖后，
我不确定如何继续安全推进到 S5，
同时不把 validation/test/LineC/tail audit metric 用作 direction、
不做 dataset-name branch、
不把 4/9 real positive 写成 official success。
```
