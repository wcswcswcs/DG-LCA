# DG-KAN v12.34.2 AllBasisSubstrateFunctionalRepair 实验结果复盘

生成时间：2026-05-27（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 diagnostic/near-pass 写成 promotion。

## 1. 计划理解

v12.34.2 的目标不是继续只扩 Rational alias，而是把所有 active no-BSpline basis family 都先纳入 efficient substrate map，再只在 SubstratePass/SubstrateNearPass 上执行 basis-specific、loss-agnostic functional repair。

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

修改/新增文件：

```text
dgkan/diagnostics/basis_workspace.py
dgkan/functional/mlp_functional.py
experiments/run_v1231_basis_kernel_workspace.py
experiments/run_v1231_rational_auc_hardening.py
experiments/run_v12342_basis_functional_repair.py
experiments/run_v12342_finalize_all_basis_substrate_functional.py
```

核心内容：

```text
1. 新增 V12342_BASIS_CANDIDATES，覆盖 Rational/Chebyshev/Fourier/RBF/Wavelet 共 44 个 active no-BSpline substrate candidates。
2. 新增 workspace_gate_v12342 / workspace_strong_gate_v12342，对应计划 Gate S 的 memory/time 部分。
3. 新增 family_telemetry_metrics：Rational 使用 denominator/derivative telemetry；Non-RAT 使用 BasisKAN.basis_diagnostics + unlabeled logits/response telemetry。
4. workspace/AUC runner 支持 --candidate-registry v12342。
5. 新增 basis-specific functional P3 runner，比较 NoOp/Random/AdamWParallel/SNR controls，P4 未触发时写 explicit skip row。
6. 新增 M-J1..M-J3 MLP closure candidates。
7. 新增 finalizer 输出 v12342 required artifacts、figures、route decision 与 code review packet。
```

审计说明：

```text
1. D-RAT34..D-RAT39、D-CHE16..D-CHE19、D-FOU16..D-FOU19、D-RBF12..D-RBF16、D-WAV11..D-WAV15 映射到现有 PrimitiveSpec/内核路径；没有把 proxy alias 写成新 exact fused kernel。
2. Non-RAT telemetry 使用模型自身 basis_diagnostics 与 unlabeled train-stream response；不读取 label/CE/validation/test/LineC hard target。
3. C3 AdamWParallelDirection 只作为 non-promotable control，标记 uses_label/CE direction，不进入 promotion。
```

## 4. Provenance / smoke

语法与 registry smoke：

```text
py_compile pass
basis_candidates = 44
families = D-CHE,D-FOU,D-RAT,D-RBF,D-WAV
M-J candidates = 3
```

Smoke checks：

```text
basis workspace smoke rows = 2, workspace_gate_pass_rows = 1, hardening_executed_rows = 1
AUC/telemetry smoke candidate_rows = 1, summary_rows = 1, linec_rows = 1
basis functional P3 smoke p3_rows = 1, p3_pass_rows = 0
MLP M-J smoke candidate_rows = 1, control_rows = 6, linec_rows = 2
```

解释：smoke 只证明 v12342 registry、all-family runner 路径、basis functional P3 runner 与 M-J closure runner 可运行，不作为 official evidence，不允许 promotion。

## 5. Line D-S all-basis substrate / workspace

执行规模：

```text
candidates = 44
families = D-RAT,D-CHE,D-FOU,D-RBF,D-WAV
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 512
val_size = 256
workspace_warmup_steps = 5
workspace_profile_steps = 20
workspace_rows = 396
```

总体结果：

```text
workspace_gate_pass_rows = 137
workspace_strong_gate_pass_rows = 0
hardening_executed_rows = 137
family_near_pass_rows = 0
```

按 family 的 workspace 结果：

| family | rows | workspace pass | strong pass |
|---|---:|---:|---:|
| D-RAT | 144 | 137 | 0 |
| D-CHE | 72 | 0 | 0 |
| D-FOU | 72 | 0 | 0 |
| D-RBF | 54 | 0 | 0 |
| D-WAV | 54 | 0 | 0 |

解释：

```text
1. v12.34.2 all-basis substrate map 仍只有 Rational 打开 workspace gate。
2. Chebyshev/Fourier/RBF/Wavelet 在当前实现/测量口径下仍不能进入 substrate pass。
3. strong workspace pass 为 0，说明本轮没有健康 base，只能继续按计划对已打开/near 的候选做 functional repair 审计，不能 promotion。
```

## 6. Line D-T all-basis AUC / telemetry

### 6.1 运行 blocker 与修复

lr=0.0015 初次运行时，AUC summary 在 all-skipped Non-RAT candidate 上触发 `IndexError`。

修复内容：

```text
experiments/run_v1231_rational_auc_hardening.py
summary family 字段改为 executed[0] when present else rows[0]。
```

合理性：

```text
这是 artifact 完整性修复，不改变任何 gate。
修复后 all-skipped candidate 会写 explicit skipped summary row，不会被写成 success。
```

### 6.2 AUC / telemetry result

执行两组：

```text
lr = 0.0015, 0.002
candidates = 44
candidate_rows_per_lr = 396
summary_rows_per_lr = 44
linec_rows_per_lr = 411
```

总体 gate：

```text
lr=0.0015: candidate_auc_near_pass_rows = 0, candidate_auc_official_pass_v1233_rows = 0
lr=0.0020: candidate_auc_near_pass_rows = 0, candidate_auc_official_pass_v1233_rows = 0
```

当前结论：

```text
1. AUC/telemetry 阶段没有产生 near/official pass。
2. executed summary 仍只来自 D-RAT；Non-RAT candidates 因 workspace gate fail 被 explicit skip。
3. 下一步按计划进入 basis-specific functional P3 repair，而不是把 AUC diagnostic 写成 promotion。
```

## 7. Line B-F Rational basis-specific functional P3

执行对象：

```text
base_candidates =
  D-RAT34-TrainingDenDerivativeStabilityNoCE
  D-RAT28-GroupDiversityPreservingRational
  D-RAT26-TangentTrustRegionNoCE
functional_candidates = B-RAT1..B-RAT5
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
```

选择理由：

```text
这三个候选是 lr=0.0015 AUC summary 中 executed_rows=9 且 mean_delta 最靠前的完整执行 Rational substrate。
D-RAT38 虽然 mean 更高，但存在 skipped row，本轮不把 skipped candidate 作为第一 P3 base。
```

结果：

```text
p3_rows = 135
p3_executed_rows = 135
p3_pass_rows = 0
p4_rows = 135
p4_pass_rows = 0
```

主要失败原因：

| count | fail reason |
|---:|---|
| 61 | CouplingR2_no_gain;NoiseSignalLeak_no_drop;RealSignalReservoirRatio_no_drop;LineC_not_preserved |
| 51 | CouplingR2_no_gain;NoiseSignalLeak_no_drop;RealSignalReservoirRatio_no_drop |
| 9 | auc_time_proxy_worse;CouplingR2_no_gain;NoiseSignalLeak_no_drop;RealSignalReservoirRatio_no_drop |
| 9 | auc_time_proxy_worse;CouplingR2_no_gain;NoiseSignalLeak_no_drop;RealSignalReservoirRatio_no_drop;LineC_not_preserved |

最接近但不能 pass 的 rows：

| base | functional | dataset | seed | source_vs_best | AUC proxy delta | CEp99 delta | CouplingR2 delta | LineC |
|---|---|---|---:|---:|---:|---:|---:|---:|
| D-RAT34 | B-RAT3 | KMNIST | 0 | 0.0 | -0.18716283730833538 | -0.0022988319396972656 | 0.0015274248668856183 | 3/3 |
| D-RAT26 | B-RAT3 | KMNIST | 0 | 0.0 | -0.583133293243975 | -0.0022988319396972656 | 0.0015274248668856183 | 3/3 |
| D-RAT28 | B-RAT2 | KMNIST | 0 | 0.0 | -0.1962762584001041 | -0.0025086402893066406 | 0.0012133150420687855 | 3/3 |

解释：

```text
Rational basis-specific functional repair 没有产生 P3 pass。
最接近 rows 对 task/AUC/tail 不坏化，但 CouplingR2 gain 远低于 gate，NoiseSignalLeak / RealSignalReservoirRatio 没有按要求下降。
因此 P4 不允许触发。
```

## 8. Line M M-J MLP functional closure

执行规模：

```text
candidates = M-J1,M-J2,M-J3
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_seed_bases = 12340400,12341400,12342400
candidate_rows = 81
control_rows = 486
linec_rows = 810
gate_rows = 3
```

Aggregate：

| candidate | rows | mean source_vs_noop | mean source_vs_control | mean CouplingR2 delta | pass rows |
|---|---:|---:|---:|---:|---:|
| M-J1-CloneProbeCovarianceTransportV2 | 27 | 0.0 | -0.0014467592592592592 | -0.0003088255182890625 | 0 |
| M-J2-UnlabeledOptimizerObservableTransportV2 | 27 | 0.00043402777777777775 | -0.0010127314814814814 | -0.00009409804848446469 | 0 |
| M-J3-ArchitectureNeutralSNRTransportV2 | 27 | 0.00014467592592592592 | -0.0013020833333333333 | 0.00041898112804797394 | 0 |

结论：

```text
M-J closure 没有 exploration/official pass。
MLPFunctionalNoGo_CurrentLossAgnosticObservableFamily_v2 = 1。
```

## 9. Blocker-driven fallback: Non-RAT AdamW foreach-off lifetime repair

触发原因：

```text
Non-RAT lifetime waterfall 显示 blocker 主要来自 incremental memory / optimizer_state lifetime。
计划 8.4 明确要求在 optimizer state too large 时尝试 foreach disabled audit / optimizer-state lifetime repair。
```

本轮新增代码修改：

```text
experiments/run_v1231_basis_kernel_workspace.py
  新增 --adamw-foreach auto|true|false 与 make_adamw()，workspace profile / MLP reference / basis hardening 使用同一 AdamW foreach 配置。

experiments/run_v1231_rational_auc_hardening.py
  新增 --adamw-foreach auto|true|false，AUC/telemetry 中 MLP reference 与 basis candidate 使用同一 AdamW foreach 配置。

experiments/run_v12342_basis_functional_repair.py
  新增 --adamw-foreach auto|true|false，用于 focused Non-RAT P3 repair。

experiments/run_v12342_finalize_all_basis_substrate_functional.py
  合并 foreach-off lifetime / AUC / P3 fallback artifacts，保证最终 route 反映 blocker 修复结果。
```

### 9.1 foreach-off workspace result

执行对象：

```text
candidates = D-CHE12,D-CHE13,D-CHE15,D-FOU12,D-FOU13,D-FOU14,D-FOU15,D-FOU16,D-RBF11,D-RBF12,D-WAV10,D-WAV11
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
adamw_foreach = false
rows = 108
```

结果：

```text
workspace_gate_pass_rows = 35
workspace_strong_gate_pass_rows = 0
hardening_executed_rows = 35
```

按 candidate：

| candidate | rows | workspace pass | max incremental ratio | max step ratio |
|---|---:|---:|---:|---:|
| D-RBF11-CompactExpressionRepair-Monitor | 9 | 9 | 2.165905631659056 | 1.4680883354034604 |
| D-RBF12-CenterOccupancyRebalanceSubstrate | 9 | 9 | 2.3576864535768647 | 1.4871814162688475 |
| D-WAV11-ScaleEnergyBalanceSubstrate | 9 | 5 | 2.596144089294774 | 1.5815764191700827 |
| D-FOU12-LifetimeRecomputeBackward-K2 | 9 | 4 | 2.6012176560121767 | 1.0888114136297715 |
| D-FOU14-SincosSharedWorkspace-K2 | 9 | 4 | 2.6012176560121767 | 1.3571865007866333 |
| D-FOU16-FrequencyBandDampingSubstrate | 9 | 4 | 2.6012176560121767 | 1.1666995071622295 |
| D-CHE12/D-CHE13/D-CHE15 | 27 | 0 | >=5.08523592085236 | >=1.2113228305170776 |

解释：

```text
foreach-off 是真实进展：RBF 全部 workspace pass，部分 Fourier/Wavelet row pass。
但 strong pass 仍为 0；Chebyshev 仍被 incremental lifetime 卡住。
```

### 9.2 foreach-off AUC / task / LineC result

结果：

| candidate | executed | skipped | mean_delta_vs_MLP | worst_delta_vs_MLP | max_AUC_time_ratio_vs_MLP | max_CEp99_delta_vs_MLP | LineC pass | near |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D-FOU12-LifetimeRecomputeBackward-K2 | 4 | 5 | -0.314453125 | -0.50390625 | 7.072692139920947 | -1.9659578800201416 | 0/12 | 0 |
| D-FOU14-SincosSharedWorkspace-K2 | 4 | 5 | -0.314453125 | -0.50390625 | 4.390793909156681 | -1.9659578800201416 | 0/12 | 0 |
| D-FOU16-FrequencyBandDampingSubstrate | 4 | 5 | -0.314453125 | -0.50390625 | 4.303501406591466 | -1.9659578800201416 | 0/12 | 0 |
| D-RBF11-CompactExpressionRepair-Monitor | 9 | 0 | -0.4535590277777778 | -0.6796875 | 9.024232940264508 | -0.9338722229003906 | 14/27 | 0 |
| D-RBF12-CenterOccupancyRebalanceSubstrate | 9 | 0 | -0.4535590277777778 | -0.6796875 | 7.525988321545862 | -0.9338722229003906 | 14/27 | 0 |
| D-WAV11-ScaleEnergyBalanceSubstrate | 5 | 4 | -0.546875 | -0.671875 | 10.238309934885777 | -2.0249366760253906 | 0/15 | 0 |

解释：

```text
foreach-off 打开了部分 Non-RAT lifetime，但 task/AUC/LineC 全面坏化。
CEp99 tail 反而下降，说明 blocker 不是单纯 tail，而是 task/AUC/LineC substrate health。
```

### 9.3 focused Fourier P3 after lifetime repair

执行对象：

```text
base_candidates = D-FOU12,D-FOU14,D-FOU16
functional_candidates = B-FOU1..B-FOU5
workspace source = v12342_nonrat_lifetime_foreachoff_workspace_truth.csv
adamw_foreach = false
```

结果：

```text
p3_rows = 135
p3_executed_rows = 60
p3_pass_rows = 0
p4_rows = 60
p4_pass_rows = 0
```

最接近 rows：

| base | functional | dataset | seed | source_vs_noop | source_vs_best | AUC proxy delta | CEp99 delta | CouplingR2 delta | LineC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| D-FOU12 | B-FOU4 | MNIST | 1 | 0.00390625 | 0.0 | 0.6541709063847634 | -0.0000095367431640625 | -0.0000013030337882025123 | 0/3 |
| D-FOU14 | B-FOU4 | MNIST | 1 | 0.00390625 | 0.0 | 0.021251624072647246 | -0.0000095367431640625 | -0.0000013030337882025123 | 0/3 |
| D-FOU16 | B-FOU4 | MNIST | 1 | 0.00390625 | 0.0 | -0.3121208564679705 | -0.0000095367431640625 | -0.0000013030337882025123 | 0/3 |

解释：

```text
Focused Fourier P3 仍没有 pass。
部分 row 有 source_vs_noop 正值和 CEp99 non-harm，但 LineC=0/3 且 CouplingR2 没有 gain。
因此 P4 仍不允许触发。
```

## 10. Final route

最终 route：

```text
route = R2-SubstrateButNoFunctionalRepair
route_detail = R5-MLPFunctionalNoGo_BasisSpecificOnly also applies
minimum_success = S1-AnyFamilySubstratePass
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 7
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_rows = 504
basis_workspace_pass_count = 172
basis_workspace_strong_pass_count = 0
substrate_gate_pass_count = 11
substrate_near_pass_count = 15
healthy_base_gate_pass_count = 0
nonrat_s3_lifetime_count = 3
family_telemetry_rows = 900
basis_functional_p3_rows = 270
basis_functional_p3_pass_count = 0
basis_functional_p4_rows = 195
basis_functional_p4_pass_count = 0
mlp_functional_candidate_rows = 81
mlp_functional_control_rows = 486
mlp_functional_linec_rows = 810
mlp_functional_mj_pass_rows = 0
mlp_functional_no_go_current_family_v2 = 1
linec_audit_rows = 3755
provenance_violation_count = 0
code_review_packet_entries = 30
code_review_packet_sha256 = 6d159cd0115d5405b3131ade41816478da627a13e7e3604935f77d78de806240
```

补充 artifact 修复：

```text
计划 6.2 要求 fig_family_telemetry_vs_auc.svg 与 fig_family_telemetry_correlation_heatmap.svg。
finalizer 已补齐这两个 figure，并重新生成 route / required manifest / code packet。
```

最终 code packet：

```text
code_review_packet_entries = 30
code_review_packet_sha256 = 6d159cd0115d5405b3131ade41816478da627a13e7e3604935f77d78de806240
```

## 11. Required artifacts

主要产物：

```text
results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_route_decision.json
results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_required_artifact_manifest.csv
results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_family_substrate_summary.csv
results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_family_telemetry.csv
results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_family_telemetry_correlations.csv
results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_basis_functional_p3.csv
results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_basis_functional_p4.csv
results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_basis_functional_nonrat_foreachoff_p3.csv
results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_kernel_lifetime_waterfall.csv
results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_mlp_functional_closure.csv
results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_linec_audit.csv
results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_failure_table.csv
results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_next_hypothesis_queue.md
results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_code_review_packet.zip
```

Required manifest:

```text
missing = 0 / 28
```

## 12. 最终科学结论

v12.34.2 没有达成 S5，也没有达成 basis-specific functional P3/P4。

已闭合事实：

```text
1. all-basis substrate map 已执行 44 candidates x 3 datasets x 3 seeds。
2. Rational 仍是唯一稳定 substrate family：11 个 candidate 达到 substrate gate，15 个 candidate 达到 near substrate。
3. Rational functional P3 在 top 3 完整执行 substrate 上为 0 pass；主要 blocker 是 CouplingR2 / NoiseSignalLeak / RealSignalReservoirRatio 没有按 gate 改善。
4. AdamW foreach-off 是真实 Non-RAT lifetime 进展：RBF 18/18 workspace pass，Fourier 12/45 workspace pass，Wavelet 5/18 workspace pass；但 task/AUC/LineC 全面崩坏。
5. Fourier focused P3 在 foreach-off lifetime-open rows 上仍为 0 pass，LineC 与 CouplingR2 没闭合。
6. Chebyshev 仍被 incremental memory lifetime 卡住，没有 workspace pass。
7. M-J1..M-J3 MLP closure 仍 0 pass，当前 MLP loss-agnostic observable family v2 no-go。
8. Provenance audit 通过，required artifacts 缺失为 0。
```

新增 no-go boundary：

```text
1. v12.34.2 纠正了只看 Rational 的问题，但真正可用 substrate 仍主要来自 Rational。
2. Non-RAT 的 blocker 被进一步分解：foreach-off 能打开部分 lifetime，但打开 lifetime 后 task/AUC/LineC health 仍坏，不能直接进入 promotion。
3. 当前 basis-specific functional P3 的弱点不是 CEp99 tail，而是 LineC/CouplingR2 与 noise/reservoir co-location 没有实质增益。
4. 继续扩同一 B-RAT/B-FOU/M-J objective 网格会变成低价值排列；下一步应是机制级修复：Non-RAT task-health substrate 或更强的 basis-specific channel co-location functional，而不是继续扩大 token。
```

最终合法状态：

```text
R2-SubstrateButNoFunctionalRepair
minimum_success = S1-AnyFamilySubstratePass
```

本轮已经按计划执行了 all-basis substrate map、family telemetry、Rational P3、Non-RAT lifetime fallback、Non-RAT focused P3、MLP M-J closure、finalizer/provenance/required artifact 审计。当前 `hard_compute_budget_exhausted=1`、`fallback_all_executed=1`、`final_stop_allowed=1` 均成立。我现在不确定继续在同一 basis-functional token family 上扩局部变体能形成有效机制；继续执行会变成低价值网格搜索。

## 13. 用户再次追问后的 stop-contract 复核

用户再次要求确认 v12.34.2 是否达成目标，若未达成则继续。再次读取最终 `v12342_route_decision.json` 后，结论没有变化：

```text
route = R2-SubstrateButNoFunctionalRepair
route_detail = R5-MLPFunctionalNoGo_BasisSpecificOnly also applies
minimum_success = S1-AnyFamilySubstratePass
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_rows = 504
basis_workspace_pass_count = 172
basis_workspace_strong_pass_count = 0
substrate_gate_pass_count = 11
substrate_near_pass_count = 15
healthy_base_gate_pass_count = 0
nonrat_s3_lifetime_count = 3
basis_functional_p3_pass_count = 0
basis_functional_p4_pass_count = 0
mlp_functional_mj_pass_rows = 0
provenance_violation_count = 0
```

因此最终判断仍是：

```text
v12.34.2 没有达成 S5；
没有达成 basis-specific functional P3/P4；
不允许 promotion；
合法 route 仍是 R2-SubstrateButNoFunctionalRepair；
允许 final stop。
```

本次没有新增训练实验、没有新增 CSV/JSON 指标、没有修改 gate。原因是计划内 all-basis substrate、family telemetry、basis-specific P3、Non-RAT lifetime fallback、focused Non-RAT P3、MLP M-J closure 与 required artifact 审计均已执行；当前我已经不确定继续在同一 B-RAT/B-FOU/M-J token family 上排列局部变体能形成有效机制。

## 14. 用户再次追问后的 stop-contract 复核 2

用户再次要求确认 v12.34.2 是否达成目标，若未达成则继续。再次读取最终 `v12342_route_decision.json` 后，结论仍未变化：

```text
route = R2-SubstrateButNoFunctionalRepair
route_detail = R5-MLPFunctionalNoGo_BasisSpecificOnly also applies
minimum_success = S1-AnyFamilySubstratePass
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 7
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_rows = 504
basis_workspace_pass_count = 172
basis_workspace_strong_pass_count = 0
substrate_gate_pass_count = 11
substrate_near_pass_count = 15
healthy_base_gate_pass_count = 0
nonrat_s3_lifetime_count = 3
basis_functional_p3_pass_count = 0
basis_functional_p4_pass_count = 0
mlp_functional_mj_pass_rows = 0
mlp_functional_no_go_current_family_v2 = 1
provenance_violation_count = 0
```

因此最终判断仍是：

```text
v12.34.2 没有达成 S5；
没有达成 basis-specific functional P3/P4；
MLP M-J closure 仍为 no-go；
不允许 promotion；
合法 route 仍是 R2-SubstrateButNoFunctionalRepair；
允许 final stop。
```

本次没有新增训练实验、没有新增 artifact 指标、没有修改代码或 gate。原因仍是：计划内 all-basis substrate map、family telemetry、Rational P3、Non-RAT foreach-off lifetime fallback、focused Fourier P3、MLP M-J closure 与 finalizer/provenance/required artifact 审计均已执行；当前 `hard_compute_budget_exhausted=1`、`fallback_all_executed=1`、`final_stop_allowed=1` 均成立。我现在不确定继续在同一 B-RAT/B-FOU/M-J token family 上排列局部变体能形成有效机制；继续执行会变成低价值网格搜索。

## 15. 用户再次追问后的计划文件 stop-contract 复核 3

用户再次要求“按计划文件里的推荐思路修改，除非不确定怎么做否则继续推进”。本次直接复核计划文件 `docs/DG-KAN_v12.34.2_AllBasisSubstrateFunctionalRepair_完整计划.md` 的 Route 定义与下一步优先级：

```text
S1 = 至少一个 active basis 通过 Substrate Gate。
S2 = 至少一个 basis-specific functional repair 通过 P3。
S3 = 至少一个 basis-specific functional repair 通过 P4。
S5 = label-free strict FC-PureKAN base/substrate + functional repair 通过 official controls。
R2 = 有 substrate，但 no functional P3/P4。
R5 = MLP functional closure 失败，后续 functional 预算转向 basis-specific。
下一步优先级 = all-basis substrate map -> family telemetry -> basis-specific functional repair。
```

对照最终 `v12342_route_decision.json`：

```text
route = R2-SubstrateButNoFunctionalRepair
route_detail = R5-MLPFunctionalNoGo_BasisSpecificOnly also applies
minimum_success = S1-AnyFamilySubstratePass
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 7
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_rows = 504
basis_workspace_pass_count = 172
basis_workspace_strong_pass_count = 0
substrate_gate_pass_count = 11
substrate_near_pass_count = 15
healthy_base_gate_pass_count = 0
nonrat_s3_lifetime_count = 3
basis_functional_p3_pass_count = 0
basis_functional_p4_pass_count = 0
mlp_functional_mj_pass_rows = 0
mlp_functional_no_go_current_family_v2 = 1
provenance_violation_count = 0
```

判断：

```text
1. v12.34.2 只达到 S1，没有达到 S2/S3/S5。
2. 计划要求的三个优先级已经执行：all-basis substrate map、family telemetry、basis-specific functional repair。
3. blocker 后已执行计划修复方向：Non-RAT AdamW foreach-off lifetime fallback 与 focused Fourier P3。
4. 仍没有 P3/P4 pass，M-J closure 也仍为 no-go。
5. 因 final_stop_allowed=1 且我已经不确定继续排列同一 B-RAT/B-FOU/M-J token family 能形成有效机制，本次不启动新增训练。
```

本次没有新增训练实验、没有新增 CSV/JSON 指标、没有修改代码或 gate；只追加计划文件对照复核。继续推进需要新的机制级计划，例如真正改善 Non-RAT task-health substrate 或更强 basis-specific channel co-location functional，而不是继续扩同类 token。

## 16. 用户再次追问后的 stop-contract 复核 4

用户再次要求确认 v12.34.2 是否达成目标，若未达成则继续。再次读取最终 `v12342_route_decision.json` 后，结论仍未变化：

```text
route = R2-SubstrateButNoFunctionalRepair
route_detail = R5-MLPFunctionalNoGo_BasisSpecificOnly also applies
minimum_success = S1-AnyFamilySubstratePass
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 7
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_rows = 504
basis_workspace_pass_count = 172
basis_workspace_strong_pass_count = 0
substrate_gate_pass_count = 11
substrate_near_pass_count = 15
healthy_base_gate_pass_count = 0
nonrat_s3_lifetime_count = 3
basis_functional_p3_pass_count = 0
basis_functional_p4_pass_count = 0
mlp_functional_mj_pass_rows = 0
mlp_functional_no_go_current_family_v2 = 1
provenance_violation_count = 0
```

最终判断仍是：

```text
v12.34.2 没有达成 S5；
只达到 S1-AnyFamilySubstratePass；
没有达成 S2/S3，即 basis-specific P3/P4 仍为 0；
MLP M-J closure 仍为 no-go；
不允许 promotion；
合法 route 仍是 R2-SubstrateButNoFunctionalRepair；
允许 final stop。
```

本次没有新增训练实验、没有新增 artifact 指标、没有修改代码或 gate。原因仍是：计划内 all-basis substrate map、family telemetry、basis-specific functional repair、Non-RAT lifetime fallback、focused Fourier P3、MLP M-J closure 与 finalizer/provenance/required artifact 审计均已执行；当前 `hard_compute_budget_exhausted=1`、`fallback_all_executed=1`、`final_stop_allowed=1` 均成立。我已经不确定继续在同一 B-RAT/B-FOU/M-J token family 上扩局部变体能形成有效机制；继续执行会变成低价值网格搜索。

## 17. 用户再次追问后的 stop-contract 复核 5

用户再次要求确认 v12.34.2 是否达成目标，若未达成则继续。本次重新扫描计划文件推荐优先级，并检查最终 artifact 与 required manifest：

```text
计划推荐优先级：
1. All-basis substrate map。
2. Family telemetry。
3. Basis-specific functional repair。
4. Kernel lifetime repair。
5. MLP functional closure。
```

本地 artifact 目录已包含对应产物：`v12342_basis_workspace_truth.csv`、`v12342_family_telemetry.csv`、`v12342_basis_functional_p3.csv`、`v12342_kernel_lifetime_waterfall.csv`、`v12342_mlp_functional_closure.csv`、`v12342_route_decision.json` 等。

再次读取最终 route：

```text
route = R2-SubstrateButNoFunctionalRepair
route_detail = R5-MLPFunctionalNoGo_BasisSpecificOnly also applies
minimum_success = S1-AnyFamilySubstratePass
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 7
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_rows = 504
basis_workspace_pass_count = 172
substrate_gate_pass_count = 11
substrate_near_pass_count = 15
nonrat_s3_lifetime_count = 3
basis_functional_p3_pass_count = 0
basis_functional_p4_pass_count = 0
mlp_functional_mj_pass_rows = 0
mlp_functional_no_go_current_family_v2 = 1
provenance_violation_count = 0
```

Required manifest 复核：

```text
manifest_rows = 28
missing_rows = 0
```

最终判断仍是：

```text
v12.34.2 没有达成 S5；
只达到 S1-AnyFamilySubstratePass；
basis-specific functional P3/P4 仍为 0；
MLP M-J closure 仍为 no-go；
required artifacts 无缺失；
不允许 promotion；
合法 route 仍是 R2-SubstrateButNoFunctionalRepair；
允许 final stop。
```

本次没有新增训练实验、没有新增 CSV/JSON 指标、没有修改代码或 gate。原因仍是：计划文件推荐路径与 blocker 后 fallback 已执行完，且 `final_stop_allowed=1`。我已经不确定继续在同一 B-RAT/B-FOU/M-J token family 上扩局部变体能形成有效机制；继续执行会变成低价值网格搜索。

## 18. 用户再次追问后的 stop-contract 复核 6

用户再次要求确认 v12.34.2 是否达成目标，若未达成则继续。再次读取最终 `v12342_route_decision.json` 后，结论仍未变化：

```text
route = R2-SubstrateButNoFunctionalRepair
route_detail = R5-MLPFunctionalNoGo_BasisSpecificOnly also applies
minimum_success = S1-AnyFamilySubstratePass
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 7
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_rows = 504
basis_workspace_pass_count = 172
substrate_gate_pass_count = 11
substrate_near_pass_count = 15
nonrat_s3_lifetime_count = 3
basis_functional_p3_pass_count = 0
basis_functional_p4_pass_count = 0
mlp_functional_mj_pass_rows = 0
mlp_functional_no_go_current_family_v2 = 1
provenance_violation_count = 0
```

最终判断仍是：

```text
v12.34.2 没有达成 S5；
只达到 S1-AnyFamilySubstratePass；
没有达成 S2/S3，即 basis-specific P3/P4 仍为 0；
MLP M-J closure 仍为 no-go；
不允许 promotion；
合法 route 仍是 R2-SubstrateButNoFunctionalRepair；
允许 final stop。
```

本次没有新增训练实验、没有新增 artifact 指标、没有修改代码或 gate。当前所有计划内推荐路径与 fallback 均已执行；我已经不确定继续在同一 B-RAT/B-FOU/M-J token family 上扩局部变体能形成有效机制，因此不继续做低价值网格搜索。
