# DG-KAN v12.33 RationalTelemetryKernel NonRATLifetime MLPFunctionalClosure 实验结果复盘

生成时间：2026-05-27（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 diagnostic/near-pass 写成 promotion。

## 1. 计划理解

v12.33 的目标不是继续扩大 v12.32 的 D-RAT proxy tail / output-geometry / M-H observable family 网格，而是执行三条更严格的线：

```text
1. Rational internal telemetry kernel / denominator-derivative / task / AUC / LineC / tail 同位验证。
2. Non-RAT exact fused kernel 之后的 incremental memory lifetime repair。
3. MLP functional M-I1..M-I4 closure/no-go。
```

硬约束：

```text
1. strict FC-PureKAN / classic no-BSpline active basis。
2. no teacher / distillation / loss modification / sampler / class weight / dataset-name branch。
3. 不使用 label-informed initialization。
4. functional direction 不使用 label、CE vector、permuted-label CE、validation/test、future outcome、query batch 或 LineC hard target。
5. CE/NLL/ECE/CEp99 只能作为审计和坏化约束。
6. blocker 后必须执行计划 fallback；smoke/diagnostic/skipped row 不能写成 promotion。
```

## 2. 当前状态

实验已完成，最终 route 为：

```text
R2-RationalTailTaskLineCNotColocated
minimum_success = S1-RationalWorkspaceOpened
```

## 3. 本轮代码修改

### 3.1 Basis workspace / telemetry / lifetime core

修改文件：

```text
dgkan/diagnostics/basis_workspace.py
```

新增内容：

```text
1. V1233_BASIS_CANDIDATES。
2. workspace_gate_v1233 / workspace_strong_gate_v1233。
3. rational_telemetry_metrics：从 GroupedRationalKATKAN.basis_diagnostics 与 unlabeled input 重建 denominator / r' / r'' telemetry。
4. nonrat_lifetime_rows：从 workspace/exact audit 生成 Non-RAT lifetime rows。
5. output-geometry repair candidates D-RAT32/D-RAT33。
```

审计说明：

```text
1. D-RAT24..D-RAT33 均未声明 exact fused rational telemetry kernel；exact_fused_rational_kernel=0。
2. Rational telemetry 使用模型参数、unlabeled train-stream input/logits，不读取 label/CE。
3. D-RAT32/D-RAT33 是 blocker 后的 label-free output geometry fallback，不是 CE-tail direction，也不是 loss modification。
```

### 3.2 MLP functional core

修改文件：

```text
dgkan/functional/mlp_functional.py
```

新增候选：

```text
M-I1-CloneProbeCovarianceUpperBound
M-I2-PrecommitUnlabeledResponseStability
M-I3-ControlResidualizedMicroProbe
M-I4-ArchitectureNeutralSNRTransport
```

合法性说明：

```text
M-I1..M-I4 只读取模型状态、unlabeled train-stream inputs、activations/logits 与随机 probe/control response；label/CE 只存在于 supervised AdamW 和 C3 control，不进入 functional direction。
```

### 3.3 Runners / finalizer

修改/新增文件：

```text
experiments/run_v1231_basis_kernel_workspace.py
experiments/run_v1231_rational_auc_hardening.py
experiments/run_v1233_finalize_rational_telemetry_lifetime.py
```

用途：

```text
1. workspace/AUC runner 支持 --candidate-registry v1233。
2. AUC runner 写出 v1233 telemetry summary / near / official gate。
3. finalizer 生成 route、required manifest、fallback manifest、provenance、forbidden audit、Non-RAT lifetime、MLP closure、figures 与 code review packet。
```

## 4. Provenance / smoke

语法与 import 审计：

```text
py_compile pass
basis_candidates = 18
basis_exact_kernel_sum = 7
rational_candidates = 8
nonrat_candidates = 10
mi_candidates = 4
controls = 5
adyn = 5
adyn_uses_y_for_stats = 0
adyn_forbidden = 0
```

Smoke 结果：

```text
basis workspace smoke rows = 2, workspace_gate_pass_rows = 1, hardening_executed_rows = 1
rational telemetry smoke candidate_rows = 1, summary_rows = 1, linec_rows = 1
MLP M-I smoke candidate_rows = 1, control_rows = 6, linec_rows = 4
A-DYN smoke rows = 4, summary_rows = 4
```

解释：smoke 只证明入口可运行，不作为 official evidence。

## 5. Line D workspace / exact / lifetime

执行规模：

```text
candidates = D-RAT24..D-RAT31,D-CHE12..D-CHE15,D-FOU12..D-FOU15,D-RBF11,D-WAV10
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 512
val_size = 256
workspace_warmup_steps = 5
workspace_profile_steps = 20
workspace_rows = 162
```

初次 official workspace 结果：

```text
workspace_gate_pass_rows = 72
workspace_strong_gate_pass_rows = 27
hardening_executed_rows = 72
family_near_pass_rows = 0
```

追加 output-geometry repair 后，finalizer 合并 workspace：

```text
workspace_rows = 180
workspace_gate_pass_rows = 90
workspace_strong_gate_pass_rows = 27
rational_workspace_pass_count = 90
rational_workspace_strong_pass_count = 27
```

Family status：

| family | rows | workspace pass | strong pass | min raw ratio | min incremental ratio | min step ratio | status |
|---|---:|---:|---:|---:|---:|---:|---|
| D-RAT | 90 | 90 | 27 | 0.6935525746430117 | 1.3255813953488371 | 0.11092792306709183 | RationalTailTaskLineCNotColocated |
| D-CHE | 36 | 0 | 0 | 0.9524707918649935 | 6.915282392026578 | 1.1001269323572664 | NonRATLifetimeBlocked |
| D-FOU | 36 | 0 | 0 | 0.8922371267849416 | 3.37375415282392 | 0.9720037013744044 | NonRATLifetimeBlocked |
| D-RBF | 9 | 0 | 0 | 0.9166421462570316 | 3.5456810631229234 | 1.3284062034487405 | NonRATLifetimeBlocked |
| D-WAV | 9 | 0 | 0 | 0.8925832972739074 | 3.5456810631229234 | 1.4440277767300609 | NonRATLifetimeBlocked |

Exact kernel audit：

| candidate | gradcheck | max forward abs err | max grad rel err | A4 smoke |
|---|---:|---:|---:|---:|
| D-CHE12-LifetimeRecomputeBackward-K3 | 1 | 4.470348358154297e-08 | 1.9474374823857943e-07 | 1 |
| D-CHE13-FusedReadoutGradNoMaterialize-K3 | 1 | 4.470348358154297e-08 | 1.9474374823857943e-07 | 1 |
| D-CHE15-FullStepNoMaterialize-K3 | 1 | 4.470348358154297e-08 | 1.9474374823857943e-07 | 1 |
| D-FOU12-LifetimeRecomputeBackward-K2 | 1 | 1.816079020500183e-08 | 3.6563899357133778e-06 | 1 |
| D-FOU13-FusedReadoutGradNoMaterialize-K2 | 1 | 1.816079020500183e-08 | 3.6563899357133778e-06 | 1 |
| D-FOU14-SincosSharedWorkspace-K2 | 1 | 1.816079020500183e-08 | 3.6563899357133778e-06 | 1 |
| D-FOU15-FullStepNoMaterialize-K2 | 1 | 1.816079020500183e-08 | 3.6563899357133778e-06 | 1 |

Non-RAT lifetime 关键结果：

| candidate | rows | S3 rows | max incremental ratio | max step ratio | failure |
|---|---:|---:|---:|---:|---|
| D-CHE12-LifetimeRecomputeBackward-K3 | 9 | 0 | 8.206810631229235 | 1.338681237192616 | incremental_memory_lifetime_blocked |
| D-CHE13-FusedReadoutGradNoMaterialize-K3 | 9 | 0 | 8.324750830564785 | 1.3229931677855158 | incremental_memory_lifetime_blocked |
| D-CHE15-FullStepNoMaterialize-K3 | 9 | 0 | 8.404485049833887 | 1.3145931873168601 | incremental_memory_lifetime_blocked |
| D-FOU12-LifetimeRecomputeBackward-K2 | 9 | 0 | 4.2583056478405314 | 1.1910953102309672 | incremental_memory_lifetime_blocked |
| D-FOU15-FullStepNoMaterialize-K2 | 9 | 0 | 4.2583056478405314 | 1.19634775782452 | incremental_memory_lifetime_blocked |
| D-RBF11-CompactExpressionRepair-Monitor | 9 | 0 | 3.5456810631229234 | 1.5291321242879559 | exact_kernel_missing |
| D-WAV10-HatWaveletLifetimeRepair-Monitor | 9 | 0 | 4.420265780730897 | 1.647810048018132 | exact_kernel_missing |

解释：

```text
1. Chebyshev/Fourier exact kernel correctness/A4 已通过，但 incremental memory lifetime gate 没打开，因此 Non-RAT S3=0。
2. RBF/WAV 只是 monitor，没有 exact kernel，不能进入 S3。
```

## 6. Rational telemetry / AUC / tail

执行：

```text
lr = 0.0015, 0.002
candidates = D-RAT24..D-RAT31
candidate_rows_per_lr = 72
summary_rows_per_lr = 8
linec_rows_per_lr = 216
```

最接近但不能 promotion 的 rows：

| run | candidate | mean_delta_vs_MLP | worst_delta_vs_MLP | max_AUC_time_ratio_vs_MLP | max_CEp99_delta_vs_MLP | LineC pass | near |
|---|---|---:|---:|---:|---:|---:|---:|
| lr=0.0015 | D-RAT24-DenDerivativeTelemetryKernel | 0.016493055555555556 | -0.01953125 | 1.736767534554294 | 2.5882649421691895 | 27/27 | 0 |
| lr=0.0015 | D-RAT31-TelemetryAblationControl | 0.016059027777777776 | -0.01953125 | 1.0882319433994665 | 2.6084342002868652 | 20/27 | 0 |
| lr=0.0015 | D-RAT25-DenDerivativeTelemetryRecomputeBackward | 0.016059027777777776 | -0.01953125 | 1.1097224272209334 | 2.6084342002868652 | 20/27 | 0 |
| lr=0.002 | D-RAT24-DenDerivativeTelemetryKernel | 0.013888888888888888 | -0.015625 | 2.207484044165786 | 3.3433451652526855 | 27/27 | 0 |

Telemetry 关键事实：

```text
rational_telemetry_available_rows = 162
典型 min_den_p01_batch ≈ 1.0000253915786743
典型 max_r_prime_p99 ≈ 0.998537540435791
```

解释：telemetry 已经可用，但当前 denominator/rprime telemetry 没有解释 CEp99/AUC tail failure；task/LineC positive rows 仍伴随 CEp99 或 AUC_time 坏化。

## 7. Rational output-geometry fallback repair

触发原因：

```text
Rational telemetry 可用，但 tail/AUC/task/LineC 未同位；
按计划 fallback 尝试 label-free output entropy/logit geometry proxy。
```

新增候选：

```text
D-RAT32-LogitSpectrumTelemetryRepairNoCE
D-RAT33-LogitNormAnchorTelemetryRepairNoCE
```

Workspace：

```text
rows = 18
workspace_gate_pass_rows = 18
workspace_strong_gate_pass_rows = 0
```

AUC / tail：

| candidate | mean_delta_vs_MLP | worst_delta_vs_MLP | max_AUC_time_ratio_vs_MLP | max_CEp99_delta_vs_MLP | LineC pass | near |
|---|---:|---:|---:|---:|---:|---:|
| D-RAT32-LogitSpectrumTelemetryRepairNoCE | -0.06076388888888889 | -0.09375 | 2.8862288784275205 | -0.3213939666748047 | 8/27 | 0 |
| D-RAT33-LogitNormAnchorTelemetryRepairNoCE | -0.06032986111111111 | -0.09375 | 2.823751640614379 | -0.317828893661499 | 8/27 | 0 |

解释：

```text
output-geometry repair 确实降低了 CEp99 tail，但代价是 task / AUC / LineC 全面坏化。
因此它不是 Rational S2，也不能 promotion。
```

## 8. MLP M-I functional closure

执行规模：

```text
windows = 3,5,10
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_seed_bases = 12330400,12331400,12332400
candidates = M-I1..M-I4
candidate_rows = 324
control_rows = 1944
linec_rows = 3240
```

v12.33 closure gate：

| candidate | window | mean_source_vs_noop | mean_source_vs_control | mean_CouplingR2_delta | max LineC | aggregate pass |
|---|---:|---:|---:|---:|---:|---:|
| M-I1-CloneProbeCovarianceUpperBound | 3 | -0.00028935185185185184 | -0.002025462962962963 | -0.00015734842230025396 | 5 | 0 |
| M-I1-CloneProbeCovarianceUpperBound | 5 | -0.00043402777777777775 | -0.002459490740740741 | 0.0007975245052416597 | 5 | 0 |
| M-I1-CloneProbeCovarianceUpperBound | 10 | 0.00014467592592592592 | -0.0013020833333333333 | 0.00012129194532837067 | 5 | 0 |
| M-I2-PrecommitUnlabeledResponseStability | 3 | -0.00028935185185185184 | -0.002025462962962963 | 0.0006468712606268989 | 5 | 0 |
| M-I2-PrecommitUnlabeledResponseStability | 5 | -0.00043402777777777775 | -0.002459490740740741 | 0.0004327071461665954 | 5 | 0 |
| M-I2-PrecommitUnlabeledResponseStability | 10 | -0.00043402777777777775 | -0.001880787037037037 | -0.0003187701523984519 | 5 | 0 |
| M-I3-ControlResidualizedMicroProbe | 3 | 0.00014467592592592592 | -0.001591435185185185 | -0.0006465871922026611 | 5 | 0 |
| M-I3-ControlResidualizedMicroProbe | 5 | 0.0 | -0.002025462962962963 | 0.0009950871179977008 | 5 | 0 |
| M-I3-ControlResidualizedMicroProbe | 10 | -0.0010127314814814814 | -0.002459490740740741 | -0.0023612274926794424 | 5 | 0 |
| M-I4-ArchitectureNeutralSNRTransport | 3 | 0.0007233796296296296 | -0.0010127314814814814 | 0.0003406769792833802 | 5 | 0 |
| M-I4-ArchitectureNeutralSNRTransport | 5 | 0.00014467592592592592 | -0.001880787037037037 | 0.0008208783416140718 | 5 | 0 |
| M-I4-ArchitectureNeutralSNRTransport | 10 | -0.00028935185185185184 | -0.001736111111111111 | -0.00021972044262748324 | 5 | 0 |

结论：M-I1..M-I4 没有 aggregate/official pass；source_vs_control 均值全为负，当前 MLP loss-agnostic observable/response family no-go。

## 9. Line A A-DYN monitor

执行规模：

```text
candidates = A-DYN1..A-DYN5
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 512
val/test = 256
epochs = 3
rows = 63
summary_rows = 7
```

关键结果：

| candidate | mean_delta_vs_mlp | worst_delta_vs_mlp | max_AUC_time_ratio_vs_mlp | LineC nontearing pass rate |
|---|---:|---:|---:|---:|
| A-DYN1-LearnableSignalFrameWarmup | -0.012152777777777778 | -0.06640625 | 1.1780500465248718 | 0.0 |
| A-DYN4-OvercompleteRankGuardFrame | -0.020833333333333332 | -0.08984375 | 1.7658816221695062 | 0.1111111111111111 |
| A-DYN2-EarlySelfPredictiveFrame | -0.06684027777777778 | -0.19921875 | 1.8378763172780725 | 0.1111111111111111 |
| A-DYN3-OptimizerObservableFrameRefresh | -0.06770833333333333 | -0.15234375 | 2.9922357653197165 | 0.0 |
| A-DYN5-RoleEnergyBalancedFHQMonitor | -0.2404513888888889 | -0.359375 | 4.476867588411187 | 0.0 |

结论：A-DYN monitor 没有恢复 label-free near-anchor，Line F official re-entry 不触发。

## 10. Final route

最终 route：

```text
route = R2-RationalTailTaskLineCNotColocated
minimum_success = S1-RationalWorkspaceOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_rows = 180
basis_workspace_pass_count = 90
basis_workspace_strong_pass_count = 27
rational_workspace_pass_count = 90
rational_workspace_strong_pass_count = 27
rational_telemetry_available_rows = 162
rational_auc_near_pass_count = 0
rational_auc_official_pass_count = 0
nonrat_s3_lifetime_count = 0
mlp_functional_candidate_rows = 324
mlp_functional_control_rows = 1944
mlp_functional_linec_rows = 3240
mlp_functional_aggregate_pass_rows = 0
mlp_functional_no_go_current_family = 1
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
code_review_packet_sha256 = 302eadebcc7cab674a659ae64142f0b48eb25f3eb33eb4acc90d78d2894b4ed3
```

## 11. Required artifacts

主要产物：

```text
results/v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure/official_v1233/v1233_route_decision.json
results/v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure/official_v1233/v1233_required_artifact_manifest.csv
results/v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure/official_v1233/v1233_fallback_manifest.csv
results/v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure/official_v1233/v1233_rational_telemetry.csv
results/v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure/official_v1233/v1233_nonrat_lifetime.csv
results/v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure/official_v1233/v1233_mlp_functional_closure.csv
results/v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure/official_v1233/v1233_kernel_implementation_manifest.csv
results/v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure/official_v1233/v1233_exact_kernel_audit.csv
results/v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure/official_v1233/v1233_code_review_packet.zip
```

## 12. 最终科学结论

v12.33 没有达成 S5，也没有达成 Rational S2、Non-RAT S3 或 MLP S4。

已闭合事实：

```text
1. Rational workspace 已打开：最终 90/90 Rational workspace pass，27/90 strong pass。
2. Rational internal telemetry 已可用：telemetry_available_rows=162；但 task/AUC/LineC/tail 没有同位。
3. D-RAT24/25/31 等能取得 positive mean task 与较高 LineC，但 CEp99 tail 和/或 AUC_time 明显坏化。
4. D-RAT32/D-RAT33 output-geometry repair 能降低 CEp99 tail，但 task/worst/AUC/LineC 全面坏化。
5. Non-RAT exact Chebyshev/Fourier kernel gradcheck/A4 pass，但 incremental memory lifetime gate 未开，因此 S3=0。
6. M-I1..M-I4 没有 MLP functional aggregate/official pass，当前 observable/response family no-go。
7. A-DYN monitor 没有 label-free near-anchor。
8. Provenance audit 通过，required artifacts 缺失为 0，fallback 已执行。
```

新增 no-go boundary：

```text
1. Rational blocker 已从 workspace/telemetry 可见性，转为 task/AUC/LineC/tail 同位 blocker。
2. 当前 denominator/derivative telemetry 没有提供可用的合法 tail repair signal；按硬约束不能用 CE-tail 设计方向或改 loss。
3. label-free output geometry 可以压 CEp99，但会牺牲 task/AUC/LineC。
4. Non-RAT 的 blocker 是 incremental memory lifetime overlap，不是 exact kernel correctness。
5. MLP M-I closure family 不能击败 matched controls，不应继续扩同类 objective。
```

最终合法状态：

```text
R2-RationalTailTaskLineCNotColocated
minimum_success = S1-RationalWorkspaceOpened
```

我现在不确定继续在同一 Rational output-geometry / M-I observable family 上排列局部变体能形成有效机制。下一步若继续，应进入真正的 rational training-time denominator/derivative stability mechanism，或 Non-RAT optimizer/readout-gradient lifetime overlap repair，而不是继续 alias/token 网格搜索。

## 13. 用户再次追问后的 stop-contract 复核

用户再次要求确认 v12.33 是否达成目标，若未达成则继续。再次读取最终 `v1233_route_decision.json` 后，结论没有变化：

```text
route = R2-RationalTailTaskLineCNotColocated
minimum_success = S1-RationalWorkspaceOpened
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_pass_count = 90
basis_workspace_strong_pass_count = 27
rational_workspace_pass_count = 90
rational_workspace_strong_pass_count = 27
rational_telemetry_available_rows = 162
rational_auc_near_pass_count = 0
nonrat_s3_lifetime_count = 0
mlp_functional_aggregate_pass_rows = 0
mlp_functional_no_go_current_family = 1
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
```

因此最终判断仍是：

```text
v12.33 没有达成 S5；
没有达成 Rational S2；
没有达成 Non-RAT S3；
没有达成 MLP functional S4；
不允许 promotion；
合法 route 仍是 R2-RationalTailTaskLineCNotColocated；
允许 final stop。
```

本次没有新增训练实验。原因是 v12.33 计划中的不允许 final stop 条件已经闭合：Rational telemetry artifact 已生成，denominator / derivative / tangent autopsy 已执行，Non-RAT lifetime waterfall 已生成，MLP matched-control closure 已完成；同时 `hard_compute_budget_exhausted=1`、`fallback_all_executed=1`、`final_stop_allowed=1` 均成立。当前我已经不确定继续在同一 Rational output-geometry / M-I observable family 上排列局部变体能形成有效机制；继续执行会变成低价值 alias/token 网格搜索。

## 14. 用户再次追问后的 stop-contract 复核 2

用户再次要求确认 v12.33 是否达成目标，若未达成则继续。用户给出的路径：

```text
docs/DG-docs/DG-KAN_v12.33_RationalTelemetryKernel_NonRATLifetime_MLPFunctionalClosure_完整计划.md
```

本地不存在；实际计划文件仍为：

```text
docs/DG-KAN_v12.33_RationalTelemetryKernel_NonRATLifetime_MLPFunctionalClosure_完整计划.md
```

再次读取最终 `v1233_route_decision.json` 后，结论仍未变化：

```text
route = R2-RationalTailTaskLineCNotColocated
minimum_success = S1-RationalWorkspaceOpened
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_pass_count = 90
basis_workspace_strong_pass_count = 27
rational_workspace_pass_count = 90
rational_workspace_strong_pass_count = 27
rational_telemetry_available_rows = 162
rational_auc_near_pass_count = 0
nonrat_s3_lifetime_count = 0
mlp_functional_aggregate_pass_rows = 0
mlp_functional_no_go_current_family = 1
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
```

最终判断仍是：

```text
v12.33 没有达成 S5；
没有达成 Rational S2；
没有达成 Non-RAT S3；
没有达成 MLP functional S4；
不允许 promotion；
合法 route 仍是 R2-RationalTailTaskLineCNotColocated；
允许 final stop。
```

本次没有新增训练实验。原因不变：计划中的 Rational telemetry、denominator / derivative / tangent autopsy、Non-RAT lifetime waterfall、MLP matched-control closure 与 fallback 均已执行；当前 `hard_compute_budget_exhausted=1`、`fallback_all_executed=1`、`final_stop_allowed=1` 均成立。我已经不确定继续在同一 Rational output-geometry / M-I observable family 上排列局部变体能形成有效机制；继续执行会变成低价值 alias/token 网格搜索。

## 15. 用户再次追问后的 stop-contract 复核 3

再次复核用户给出的路径与最终 route：

```text
given_plan_exists = False
actual_plan_exists = True
route = R2-RationalTailTaskLineCNotColocated
minimum_success = S1-RationalWorkspaceOpened
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
rational_telemetry_available_rows = 162
rational_auc_near_pass_count = 0
nonrat_s3_lifetime_count = 0
mlp_functional_aggregate_pass_rows = 0
mlp_functional_no_go_current_family = 1
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
```

最终判断仍然不变：

```text
v12.33 没有达成 S5；
没有达成 Rational S2；
没有达成 Non-RAT S3；
没有达成 MLP functional S4；
不允许 promotion；
合法 route 仍是 R2-RationalTailTaskLineCNotColocated；
允许 final stop。
```

本次没有新增训练实验、没有新增 CSV/JSON 指标、没有改动 gate。原因仍是：计划内 fallback 与追加的 output-geometry repair 已执行，Rational telemetry / den-derivative-tangent autopsy、Non-RAT lifetime waterfall、MLP matched-control closure 均已生成 artifact；继续在同一 token family 上做局部组合，我已经不确定能形成有效机制。

## 16. 用户再次追问后的 stop-contract 复核 4

再次复核最终 route，结果仍无变化：

```text
given_plan_exists = False
actual_plan_exists = True
route = R2-RationalTailTaskLineCNotColocated
minimum_success = S1-RationalWorkspaceOpened
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_pass_count = 90
basis_workspace_strong_pass_count = 27
rational_telemetry_available_rows = 162
rational_auc_near_pass_count = 0
nonrat_s3_lifetime_count = 0
mlp_functional_aggregate_pass_rows = 0
mlp_functional_no_go_current_family = 1
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
```

最终判断仍然不变：

```text
v12.33 没有达成 S5；
没有达成 Rational S2；
没有达成 Non-RAT S3；
没有达成 MLP functional S4；
不允许 promotion；
合法 route 仍是 R2-RationalTailTaskLineCNotColocated；
允许 final stop。
```

本次没有新增训练实验、没有新增 artifact 指标、没有修改代码或 gate。继续推进需要新的机制级计划，例如真正训练期 denominator/derivative stability mechanism 或 Non-RAT optimizer/readout-gradient lifetime repair；在当前 v12.33 同一 Rational output-geometry / M-I observable family 内继续排列局部变体，我已经不确定能形成有效机制。
