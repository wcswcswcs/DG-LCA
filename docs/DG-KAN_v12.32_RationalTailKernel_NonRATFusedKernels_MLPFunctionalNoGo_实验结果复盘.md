# DG-KAN v12.32 RationalTailKernel NonRATFusedKernels MLPFunctionalNoGo 实验结果复盘

生成时间：2026-05-27（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 diagnostic/near-pass 写成 promotion。

## 1. 计划理解

v12.32 的目标不是继续扩大 v12.31 的 Rational alias / M-G objective 网格，而是执行三条更严格的线：

```text
1. Rational tail-stable kernel / workspace / task / AUC / LineC / tail 同位验证。
2. Non-RAT true fused/no-materialize kernel 审计，防止把 wrapper/alias 写成 kernel success。
3. MLP functional M-H1..M-H3 no-go / generic positive 复核。
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

实验正在执行中。本文件后续只记录已经生成 artifact 的真实结果。

## 3. 本轮代码修改

### 3.1 Basis workspace / kernel audit core

修改文件：

```text
dgkan/diagnostics/basis_workspace.py
```

新增内容：

```text
1. V1232_BASIS_CANDIDATES：D-RAT16..D-RAT23、D-CHE10..D-CHE11、D-FOU10..D-FOU11。
2. v12.32 strict workspace gate：raw<=1.05, incremental<=1.75, step<=1.75。
3. v12.32 strong gate：raw<=1.05, incremental<=1.35, step<=1.40。
4. kernel_implementation_manifest_rows / exact_kernel_audit_rows。
5. rational_tail_metrics：只用 unlabeled logits/probability tail proxy；现有 grouped rational path 尚未暴露 per-sample denominator/derivative tensor。
```

重要审计说明：

```text
1. D-CHE10/D-CHE11/D-FOU10/D-FOU11 映射到已有 Triton no-materialize kernel path，并在 exact audit 中做 manual_gradient_audit。
2. D-RAT16..D-RAT23 映射到已有 Rational / output geometry PrimitiveSpec 做 tail-stability probe；不是所有计划字面机制都已有 exact fused rational kernel，因此 exact_kernel_implemented=0。
3. 不把 Rational proxy alias 写成 true fused rational denominator kernel。
```

### 3.2 MLP functional core

修改文件：

```text
dgkan/functional/mlp_functional.py
```

新增候选：

```text
M-H1-UnlabeledMicroProbeResponsePredictor
M-H2-RandomCotangentLowRankResponseController
M-H3-ActivationSpectrumGuardWithMatchedControls
```

合法性说明：

```text
M-H1..M-H3 只读取模型状态、unlabeled train-stream inputs、activations、logits、random cotangent / microprobe response；label/CE 只存在于 supervised AdamW 训练与 C3 control，不进入 functional direction。
```

### 3.3 v12.32 runner/finalizer

修改/新增文件：

```text
experiments/run_v1231_basis_kernel_workspace.py
experiments/run_v1231_rational_auc_hardening.py
experiments/run_v1232_finalize_rational_tail_kernel.py
```

设计说明：

```text
1. basis workspace runner 增加 v1232 registry 和 strict gate；核心逻辑仍在 dgkan。
2. Rational AUC runner 增加 v1232 registry 和 tail audit metrics；核心 trajectory helper 仍在 dgkan。
3. v12.32 finalizer 生成 route / fallback / required manifest / provenance / forbidden audit / figures / code packet。
```

语法/provenance 检查：

```text
py_compile pass
basis_candidates = 12
basis_exact_kernel_sum = 4
mh_candidates = 3
controls = 5
adyn = 5
adyn_uses_y_for_stats = 0
adyn_forbidden = 0
```

## 4. Smoke checks

已执行三个 smoke：

```text
basis workspace smoke rows = 2, workspace_gate_pass_rows = 1, hardening_executed_rows = 1
MLP functional smoke candidate_rows = 1, control_rows = 6, linec_rows = 4
A-DYN smoke rows = 4, summary_rows = 4
```

解释：

```text
1. Basis smoke 证明 v12.32 registry、strict workspace gate、kernel manifest/exact audit 输出链可运行。
2. MLP smoke 证明 M-H1 source 可以通过现有 MLP functional runner 生成 candidate/control/LineC/gate artifact。
3. A-DYN smoke 证明 monitor 入口仍可运行。
4. Smoke 不作为 official evidence，不允许 promotion。
```

## 5. Line D basis workspace / exact kernel audit

执行规模：

```text
candidates = D-RAT16..D-RAT23,D-CHE10,D-CHE11,D-FOU10,D-FOU11
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 512
val_size = 256
workspace_warmup_steps = 5
workspace_profile_steps = 20
workspace_rows = 108
```

总体结果：

```text
workspace_gate_pass_rows = 72
workspace_strong_gate_pass_rows = 0
hardening_executed_rows = 72
family_near_pass_rows = 0
```

Family status：

| family | rows | workspace pass | strong pass | min raw ratio | min incremental ratio | min step ratio | status |
|---|---:|---:|---:|---:|---:|---:|---|
| D-RAT | 72 | 72 | 0 | 0.7334602829162132 | 1.5938538205980066 | 0.9099883499461409 | TaskLineCNotColocated |
| D-CHE | 18 | 0 | 0 | 1.00025389916576 | 6.923588039867109 | 1.1696214070169357 | WorkspaceBlocked |
| D-FOU | 18 | 0 | 0 | 0.9418389553862895 | 3.37375415282392 | 0.9981457309774058 | WorkspaceBlocked |

Exact kernel audit：

| candidate | exact kernel | gradcheck | max forward abs err | max grad rel err | A4 expression smoke |
|---|---:|---:|---:|---:|---:|
| D-CHE10-FusedRecurrenceNoMaterialize-K3 | 1 | 1 | 4.470348358154297e-08 | 1.9474374823857943e-07 | 1 |
| D-CHE11-FusedRecurrenceNoMaterialize-K4 | 1 | 1 | 5.960464477539063e-08 | 2.2848143998999149e-07 | 1 |
| D-FOU10-FusedSincosLowFreqK2 | 1 | 1 | 1.816079020500183e-08 | 3.6563899357133778e-06 | 1 |
| D-FOU11-FusedSincosLowFreqK4SharedAmp | 1 | 1 | 4.551839083433151e-08 | 3.1175153480944573e-07 | 1 |

为什么不能写 S3：Non-RAT exact/A4 pass，但 workspace pass=0；`nonrat_s3_true_kernel_count=0`。

Rational 3-epoch task/LineC probe：

| candidate | executed | mean_delta_vs_MLP | worst_delta_vs_MLP | LineC pass | probe pass rows |
|---|---:|---:|---:|---:|---:|
| D-RAT21-EntropyFloorUnlabeled | 9 | -0.006944444444444444 | -0.0390625 | 21/27 | 3 |
| D-RAT17-RPrimeCapNoCE | 9 | -0.0078125 | -0.05078125 | 25/27 | 5 |
| D-RAT18-TangentNormTrustNoCE | 9 | -0.0078125 | -0.05078125 | 25/27 | 5 |
| D-RAT19-DenSlopeJointStabilityNoCE | 9 | -0.0078125 | -0.05078125 | 25/27 | 5 |
| D-RAT22-TopGapFloorNoLabel | 9 | -0.008680555555555556 | -0.05078125 | 25/27 | 4 |
| D-RAT16-DenP01FloorNoCE | 9 | -0.010416666666666666 | -0.04296875 | 25/27 | 6 |

## 6. Line D-RAT Rational tail/AUC triage

执行规模：

```text
candidates = D-RAT16..D-RAT23
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
epochs = 3
lr = 0.0015,0.002
candidate_rows_per_lr = 72
summary_rows_per_lr = 8
linec_rows_per_lr = 216
```

最接近结果：

| run | candidate | mean_delta_vs_MLP | worst_delta_vs_MLP | max_AUC_time_ratio_vs_MLP | max_CEp99_delta_vs_MLP | LineC pass | near pass |
|---|---|---:|---:|---:|---:|---:|---:|
| lr=0.0015 | D-RAT18-TangentNormTrustNoCE | 0.016927083333333332 | -0.01953125 | 1.4682377071551969 | 2.6176023483276367 | 20/27 | 0 |
| lr=0.0015 | D-RAT16-DenP01FloorNoCE | 0.016493055555555556 | -0.01953125 | 1.8587716425100502 | 2.5882649421691895 | 27/27 | 0 |
| lr=0.002 | D-RAT16-DenP01FloorNoCE | 0.013888888888888888 | -0.015625 | 1.9817887704723836 | 3.3433451652526855 | 27/27 | 0 |
| lr=0.002 | D-RAT18-TangentNormTrustNoCE | 0.011284722222222222 | -0.0234375 | 1.5660789321663577 | 3.3959784507751465 | 24/27 | 0 |

解释：

```text
1. D-RAT16/17/18/19/22 能给出 positive mean task signal，LineC 也常在 20/27 以上。
2. 但 AUC_time 明显大于 1.0，CEp99 tail 坏化严重，candidate_auc_near_pass_v1232 全部为 0。
3. D-RAT20/D-RAT23 能降低 CEp99 delta，但 task/LineC/AUC 明显失败。
4. 因 CEp99/NLL/ECE 只能做 audit 与坏化约束，不能用 CE-tail 设计方向或改 loss；不能 promotion。
```

## 7. Line M M-H MLP functional no-go

执行规模：

```text
windows = 3,5,10
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_seed_bases = 12320400,12321400,12322400
candidates = M-H1,M-H2,M-H3
candidate_rows = 243
control_rows = 1458
linec_rows = 2430
```

Strict v12.32 gate 结果：

| candidate | window | mean_source_vs_noop | mean_source_vs_control | mean_CouplingR2_delta | max LineC | aggregate pass |
|---|---:|---:|---:|---:|---:|---:|
| M-H1-UnlabeledMicroProbeResponsePredictor | 3 | -0.00014467592592592592 | -0.001591435185185185 | 0.00010646776961846715 | 5/5 | 0 |
| M-H1-UnlabeledMicroProbeResponsePredictor | 5 | -0.0008680555555555555 | -0.0027488425925925927 | -0.00035373006619554895 | 5/5 | 0 |
| M-H1-UnlabeledMicroProbeResponsePredictor | 10 | -0.00014467592592592592 | -0.001880787037037037 | 0.0011243216259018724 | 5/5 | 0 |
| M-H2-RandomCotangentLowRankResponseController | 3 | 0.0005787037037037037 | -0.0008680555555555555 | -0.0003550669346684548 | 5/5 | 0 |
| M-H2-RandomCotangentLowRankResponseController | 5 | -0.00028935185185185184 | -0.002170138888888889 | -0.0005754801262932567 | 5/5 | 0 |
| M-H2-RandomCotangentLowRankResponseController | 10 | -0.00043402777777777775 | -0.002170138888888889 | -0.0009066046984900788 | 5/5 | 0 |
| M-H3-ActivationSpectrumGuardWithMatchedControls | 3 | 0.0 | -0.0014467592592592592 | 0.00005997917633342613 | 5/5 | 0 |
| M-H3-ActivationSpectrumGuardWithMatchedControls | 5 | 0.0 | -0.001880787037037037 | -0.0004700617189607594 | 5/5 | 0 |
| M-H3-ActivationSpectrumGuardWithMatchedControls | 10 | -0.00014467592592592592 | -0.001880787037037037 | -0.00035747093906262484 | 5/5 | 0 |

结论：M-H1..M-H3 没有任何 aggregate pass；source_vs_control 均值全为负，虽然 LineC 最大可达 5/5。`MLPFunctionalNoGo_CurrentLossAgnosticObservableFamily = 1`。

## 8. Line A A-DYN monitor

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

| candidate | mean_delta_vs_mlp | worst_delta_vs_mlp | max_AUC_time_ratio_vs_mlp | LineC nontearing pass |
|---|---:|---:|---:|---:|
| A-DYN1-LearnableSignalFrameWarmup | -0.012152777777777778 | -0.06640625 | 1.1780500465248718 | 0/9 |
| A-DYN4-OvercompleteRankGuardFrame | -0.020833333333333332 | -0.08984375 | 1.7658815562258872 | 1/9 |
| A-DYN2-EarlySelfPredictiveFrame | -0.06684027777777778 | -0.19921875 | 1.3969400755484518 | 1/9 |

结论：A-DYN monitor 没有恢复 label-free near-anchor，Line F official re-entry 不允许触发。

## 9. Final route

最终 route：

```text
route = R1-RationalTailStabilityBlocked
minimum_success = S1-RationalWorkspaceOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_rows = 108
basis_workspace_pass_count = 72
basis_workspace_strong_pass_count = 0
rational_workspace_pass_count = 72
rational_workspace_strong_pass_count = 0
rational_auc_near_pass_count = 0
nonrat_s3_true_kernel_count = 0
mlp_functional_candidate_rows = 243
mlp_functional_control_rows = 1458
mlp_functional_linec_rows = 2430
mlp_functional_aggregate_pass_rows = 0
mlp_functional_no_go_current_family = 1
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
code_review_packet_entries = 15
code_review_packet_sha256 = 以最终 v1232_route_decision.json 为准
```

## 10. Required artifacts

主要产物：

```text
results/v12_32_rational_tail_kernel_nonrat_fused_kernels_mlp_functional_no_go/official_v1232/v1232_route_decision.json
results/v12_32_rational_tail_kernel_nonrat_fused_kernels_mlp_functional_no_go/official_v1232/v1232_required_artifact_manifest.csv
results/v12_32_rational_tail_kernel_nonrat_fused_kernels_mlp_functional_no_go/official_v1232/v1232_fallback_manifest.csv
results/v12_32_rational_tail_kernel_nonrat_fused_kernels_mlp_functional_no_go/official_v1232/v1232_kernel_implementation_manifest.csv
results/v12_32_rational_tail_kernel_nonrat_fused_kernels_mlp_functional_no_go/official_v1232/v1232_exact_kernel_audit.csv
results/v12_32_rational_tail_kernel_nonrat_fused_kernels_mlp_functional_no_go/official_v1232/v1232_basis_workspace_truth.csv
results/v12_32_rational_tail_kernel_nonrat_fused_kernels_mlp_functional_no_go/official_v1232/v1232_rational_tail_auc_all_summary.csv
results/v12_32_rational_tail_kernel_nonrat_fused_kernels_mlp_functional_no_go/official_v1232/v1232_mlp_functional_gate.csv
results/v12_32_rational_tail_kernel_nonrat_fused_kernels_mlp_functional_no_go/official_v1232/v1232_code_review_packet.zip
```

## 11. 最终科学结论

v12.32 没有达成 S5，也没有达成 Rational S2、Non-RAT S3 或 MLP S4。

已闭合事实：

```text
1. Rational D-RAT16..D-RAT23 在 v12.32 strict workspace gate 下全部通过：72/72 workspace pass。
2. Rational strong workspace 为 0/72；incremental ratio 最低约 1.594，仍高于 strong gate 1.35。
3. Rational AUC/tail triage 中，D-RAT16/18 等能出现 positive mean task 与较高 LineC，但 AUC_time 与 CEp99 tail 不闭合。
4. D-RAT20/D-RAT23 能降低 CEp99 delta，但 task/LineC/AUC 崩坏；output geometry 仍不能同时保 task 和 tail。
5. Chebyshev/Fourier exact no-materialize kernel 的 manual gradcheck/A4 smoke 通过，但 incremental memory gate 未开，因此 S3=0。
6. M-H1..M-H3 在 windows=3/5/10 下没有 aggregate pass；当前 MLP loss-agnostic observable family no-go。
7. A-DYN monitor 没有 label-free near-anchor。
8. Provenance audit 通过，required artifacts 缺失为 0，fallback depth 6 已执行。
```

新增 no-go boundary：

```text
1. Rational tail-stability blocker 从 workspace blocker 转为 task/AUC/tail 同位 blocker。
2. 当前 proxy Rational tail mechanisms 不能同时维持 positive task、LineC、AUC_time 与 CEp99。
3. Non-RAT exact kernel 已不是“是否存在”的问题；当前 blocker 是 incremental memory accounting / workspace lifetime。
4. MLP M-H observable family 不具备 generic positive，不应继续扩同类 MLP objective。
```

最终合法状态：

```text
R1-RationalTailStabilityBlocked
minimum_success = S1-RationalWorkspaceOpened
```

我现在不确定继续在同一 D-RAT proxy tail / output-geometry token family 上扩局部组合能形成有效机制。下一步应进入真正 denominator/derivative telemetry kernel 或 Non-RAT incremental memory lifetime repair，而不是继续排列当前 alias。

## 12. 用户再次追问后的 stop-contract 复核

用户再次要求确认 v12.32 是否达成目标，若未达成则继续。读取最终 `v1232_route_decision.json` 后，结论没有变化：

```text
route = R1-RationalTailStabilityBlocked
minimum_success = S1-RationalWorkspaceOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_pass_count = 72
basis_workspace_strong_pass_count = 0
rational_workspace_pass_count = 72
rational_workspace_strong_pass_count = 0
rational_auc_near_pass_count = 0
nonrat_s3_true_kernel_count = 0
mlp_functional_aggregate_pass_rows = 0
mlp_functional_no_go_current_family = 1
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
```

因此最终判断仍是：

```text
v12.32 没有达成 S5；
没有达成 Rational S2；
没有达成 Non-RAT S3；
没有达成 MLP functional S4；
不允许 promotion；
合法 route 仍是 R1-RationalTailStabilityBlocked；
允许 final stop。
```

本次没有新增训练实验。原因是 v12.32 已执行计划 fallback depth 6，完成 Rational tail proxy workspace/AUC repair、Non-RAT exact fused-kernel audit、M-H functional no-go、A-DYN monitor、finalizer/provenance/required artifact 审计。当前已经满足 `hard_compute_budget_exhausted=1`、`fallback_all_executed=1`、`final_stop_allowed=1`。我已经不确定继续在同一 D-RAT proxy tail / output-geometry / M-H observable family 上扩局部组合会形成有效机制；继续执行会变成低价值网格搜索。

最新 code packet sha 以最终 `v1232_route_decision.json` 为准。

## 15. 用户再次追问后的 stop-contract 复核 4

用户再次要求确认 v12.32 是否达成目标，若未达成则继续。再次读取最终 `v1232_route_decision.json` 后，结论仍未变化：

```text
route = R1-RationalTailStabilityBlocked
minimum_success = S1-RationalWorkspaceOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_pass_count = 72
basis_workspace_strong_pass_count = 0
rational_workspace_pass_count = 72
rational_workspace_strong_pass_count = 0
rational_auc_near_pass_count = 0
nonrat_s3_true_kernel_count = 0
mlp_functional_aggregate_pass_rows = 0
mlp_functional_no_go_current_family = 1
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
```

因此最终判断仍是：

```text
v12.32 没有达成 S5；
没有达成 Rational S2；
没有达成 Non-RAT S3；
没有达成 MLP functional S4；
不允许 promotion；
合法 route 仍是 R1-RationalTailStabilityBlocked；
允许 final stop。
```

本次没有新增训练实验。原因仍是：v12.32 已完成计划内 fallback depth 6，并追加完成 Rational tail proxy workspace/AUC repair、Non-RAT exact fused-kernel audit、M-H functional no-go、A-DYN monitor 与 finalizer/provenance/required artifact 审计；当前 `hard_compute_budget_exhausted=1`、`fallback_all_executed=1`、`final_stop_allowed=1` 均成立。我已经不确定继续在同一 D-RAT proxy tail / output-geometry / M-H observable family 上做局部组合能形成有效机制；继续执行会变成低价值网格搜索。

最新 code packet sha 以最终 `v1232_route_decision.json` 为准。

## 14. 用户再次追问后的 stop-contract 复核 3

用户再次要求确认 v12.32 是否达成目标，若未达成则继续。再次读取最终 `v1232_route_decision.json` 后，结论仍未变化：

```text
route = R1-RationalTailStabilityBlocked
minimum_success = S1-RationalWorkspaceOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_pass_count = 72
basis_workspace_strong_pass_count = 0
rational_workspace_pass_count = 72
rational_workspace_strong_pass_count = 0
rational_auc_near_pass_count = 0
nonrat_s3_true_kernel_count = 0
mlp_functional_aggregate_pass_rows = 0
mlp_functional_no_go_current_family = 1
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
```

因此最终判断仍是：

```text
v12.32 没有达成 S5；
没有达成 Rational S2；
没有达成 Non-RAT S3；
没有达成 MLP functional S4；
不允许 promotion；
合法 route 仍是 R1-RationalTailStabilityBlocked；
允许 final stop。
```

本次没有新增训练实验。原因仍是：v12.32 已完成计划 fallback depth 6，并追加完成 Rational tail proxy workspace/AUC repair、Non-RAT exact fused-kernel audit、M-H functional no-go、A-DYN monitor 与 finalizer/provenance/required artifact 审计；当前 `hard_compute_budget_exhausted=1`、`fallback_all_executed=1`、`final_stop_allowed=1` 均成立。我已经不确定继续在同一 D-RAT proxy tail / output-geometry / M-H observable family 上扩局部组合会形成有效机制；继续执行会变成低价值网格搜索。

最新 code packet sha 以最终 `v1232_route_decision.json` 为准。

## 13. 用户再次追问后的 stop-contract 复核 2

用户再次要求确认 v12.32 是否达成目标，若未达成则继续。再次读取最终 `v1232_route_decision.json` 后，结论没有变化：

```text
route = R1-RationalTailStabilityBlocked
minimum_success = S1-RationalWorkspaceOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_pass_count = 72
basis_workspace_strong_pass_count = 0
rational_workspace_pass_count = 72
rational_workspace_strong_pass_count = 0
rational_auc_near_pass_count = 0
nonrat_s3_true_kernel_count = 0
mlp_functional_aggregate_pass_rows = 0
mlp_functional_no_go_current_family = 1
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
```

因此最终判断仍是：

```text
v12.32 没有达成 S5；
没有达成 Rational S2；
没有达成 Non-RAT S3；
没有达成 MLP functional S4；
不允许 promotion；
合法 route 仍是 R1-RationalTailStabilityBlocked；
允许 final stop。
```

本次没有新增训练实验。没有继续新增实验的原因不变：v12.32 已经完成计划内 fallback depth 6，并追加完成 Rational tail proxy workspace/AUC repair、Non-RAT exact fused-kernel audit、M-H functional no-go、A-DYN monitor 与 finalizer/provenance/required artifact 审计。当前 `hard_compute_budget_exhausted=1`、`fallback_all_executed=1`、`final_stop_allowed=1` 均成立。我已经不确定继续在同一 D-RAT proxy tail / output-geometry / M-H observable family 上排列局部变体会形成有效机制；继续执行会变成低价值网格搜索。

最新 code packet sha 以最终 `v1232_route_decision.json` 为准。
