# DG-KAN v12.28 LabelFreeBaseRecovery FunctionalReentry ClassicNoBSpline 实验结果复盘

生成时间：2026-05-26（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 diagnostic/near-pass 写成 promotion。

## 1. 计划理解

v12.28 的目标不是继续排列 v12.27 pseudo/affinity token，而是测试“早期训练中形成 label-free signal frame”的 base recovery 路径，并在 base 近似可用时尝试 functional re-entry。

硬约束：

```text
1. strict FC-PureKAN。
2. no teacher / distillation / loss modification / sampler / class weight / dataset-name branch。
3. 不使用 label-informed initialization。
4. functional direction 不使用 label、CE vector、permuted-label CE、validation/test、future outcome、query batch 或 LineC hard target。
5. CE/NLL/ECE/CEp99 只能作为审计和坏化约束。
6. blocker 后必须执行 Depth 1-6 fallback。
```

## 2. 本轮代码修改

### 2.1 label-free frame tokens

修改文件：

```text
dgkan/models/fc_purekan_primitives.py
```

新增/使用机制：

```text
srhtp
lowcoherencep
blockframep
```

用途：只基于输入维度、随机低相干 frame、block-local frame 或 train-stream x geometry 初始化 `quad_proj`，不读取 label/CE。

### 2.2 A-F / A-R registry

修改文件：

```text
experiments/run_v1218_b320_label_free_ablation.py
```

新增候选：

```text
A-F1a..A-F1d
A-F2a..A-F2d
A-F3a..A-F3d
A-F4a..A-F4d
A-R1..A-R5
```

smoke blocker 修复：

```text
A-F4c-OvercompleteSparseP-h192-LF
```

原始 `sparsek8p_pupdateevery4...` 会触发 dense learnable-P workspace，但 sparseP 的 `quad_proj` 是 fixed sparse frame buffer。已改为：

```text
sparsek8p_fixedp_directreadinit090_quadramp050
```

合理性：这是实现路径修复，不降低任何 gate；避免 sparse frame 误走 dense projector update。

### 2.3 F28 functional wrapper

新增文件：

```text
experiments/run_v1228_label_free_base_reentry.py
```

新增 functional shadow candidates：

```text
F28-S1-geometryGuardThenTask
F28-S2-taskThenGeometryGuard
F28-S3-controlResidualizedDirectQuad
F28-S4-quadReservoirReleaseShadow
F28-S5-lowRankLogitSubspaceShadow
F28-S6-unlabeledCovTransportShadow
```

### 2.4 D39-D43 Rational mapping

修改文件：

```text
experiments/run_v1224_classic_hardening.py
```

新增：

```text
D39-RationalGroupWorkspaceRecomputeV2
D40-RationalReadoutGradChunked
D41-RationalHiddenResLowMemV2
D42-RationalDenStateFP16PlusCheckpoint
D43-RationalPairNormLineCNoExtraMem
```

### 2.5 v12.28 finalizer

新增/修改文件：

```text
experiments/run_v1228_finalize_label_free_base_recovery.py
```

修复内容：

```text
1. required manifest 在 route/zip 写出后重算，避免假缺失。
2. Line A gate 只统计 A-F/A-R，不把 MLP reference summary 算作 base pass。
3. functional candidate rows 只统计 V1226_COMPOSITE_FUNCTIONAL_BRIDGE_SUMMARY，不把 variant/multisketch rows 混入 candidate rows。
4. finalizer 输入按 candidate prefix 和 row identity 去重，保证重复运行幂等。
```

## 3. Provenance audit

finalizer / registry 审计结果：

```text
forbidden_token_audit_rows = 27
forbidden_token_present_count = 0
uses_y_for_stats_count = 0
```

解释：

```text
1. A-F/A-R base candidate 全部 uses_y_for_stats=0。
2. F28 functional wrapper 没有 trainprobe / signalBroad / signalBlock / trainprobeDirect forbidden token。
3. 本轮 smoke 失败来自 sparseP 与 dense learnable-P workspace 的实现冲突，不是 label leakage。
```

## 4. Line A scout

执行规模：

```text
candidates = A-F1a..A-F4d
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 512
val/test = 256
epochs = 3
rows = 162
summary_rows = 18
```

Scout 关键结果：

| candidate | mean_delta_vs_mlp | worst_delta_vs_mlp | max_AUC_time_ratio_vs_mlp | LineC pass rate |
|---|---:|---:|---:|---:|
| A-F4d-OvercompleteThenPruneP-LF | 0.00390625 | -0.0234375 | 1.1346665308667738 | 0.1111111111111111 |
| A-F2c-BranchGainFrozenEarly-LF | 0.0 | -0.02734375 | 1.0590841308362287 | 0.0 |
| A-F1d-RandomLowCoherenceP-LearnableFrameWarmup | -0.010850694444444444 | -0.046875 | 1.2661997836455803 | 0.1111111111111111 |
| A-F1b-SRHTP-LearnableFrameWarmup | -0.012152777777777778 | -0.06640625 | 1.1780500465248718 | 0.0 |

结论：Scout 出现 task 近似/正信号，但 worst/AUC/LineC 没有闭合。

## 5. Line A hardening

执行规模：

```text
candidates = top-8
train_size = 1024
val/test = 512
epochs = 8
rows = 90
summary_rows = 10
```

Hardening 结果：

| candidate | mean_delta_vs_mlp | worst_delta_vs_mlp | max_AUC_time_ratio_vs_mlp | LineC pass rate | near-anchor |
|---|---:|---:|---:|---:|---:|
| A-F2c-BranchGainFrozenEarly-LF | 0.008897569444444444 | -0.017578125 | 1.2502325072122913 | 0.1111111111111111 | 0 |
| A-F4d-OvercompleteThenPruneP-LF | 0.006076388888888889 | -0.017578125 | 1.0685439181756535 | 0.0 | 0 |
| A-F1b-SRHTP-LearnableFrameWarmup | 0.004774305555555556 | -0.044921875 | 1.0473683071863413 | 0.1111111111111111 | 0 |
| A-F1d-RandomLowCoherenceP-LearnableFrameWarmup | 0.0030381944444444445 | -0.0234375 | 1.1639792646511955 | 0.0 | 0 |

结论：A-F2c/A-F4d 保住 positive mean delta，但 worst delta、AUC-time 和 LineC 仍失败；A-F1b 的 AUC-time 接近阈值，但 worst/LineC 失败。

当前 Line A 状态：

```text
line_a_near_anchor_pass_count = 0
line_a_official_pass_count = 0
```

## 6. Failure-specific repair

执行规模：

```text
candidates = A-R1..A-R5
rows = 63
summary_rows = 7
```

Repair 结果：

| candidate | mean_delta_vs_mlp | worst_delta_vs_mlp | max_AUC_time_ratio_vs_mlp | LineC pass rate | near-anchor |
|---|---:|---:|---:|---:|---:|
| A-R2-LineCGoodTaskDirectWarm | -0.00043402777777777775 | -0.033203125 | 1.099967319213341 | 0.0 | 0 |
| A-R3-AUCTrajectoryRelease | -0.001736111111111111 | -0.033203125 | 1.0673725930919513 | 0.0 | 0 |
| A-R5-OvercompleteRoleBalance | -0.023003472222222224 | -0.052734375 | 1.4377452848196106 | 0.1111111111111111 | 0 |
| A-R4-UpdateSpectrumWeakRefresh | -0.043619791666666664 | -0.0625 | 1.7973252510765885 | 0.3333333333333333 | 0 |

结论：A-R2/A-R3 把 mean task 拉回 near-zero，但 worst/AUC/LineC 仍不闭合；A-R4 最高 LineC 为 3/9，但 task 和 AUC 明显失败。

## 7. Functional shadow

因为 Line A near-anchor 为 0，F28 只能作为 shadow diagnostic。

执行规模：

```text
base candidates = A-F2c,A-R2,A-F1b
functional candidates = F28-S1..F28-S6
candidate_rows = 54
aggregate_rows = 18
```

总结果：

```text
any_exploration_pass = 0
any_strict_majority_pass = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
promotion_allowed = 0
```

最接近 task/control 行：

```text
base = A-R2-LineCGoodTaskDirectWarm
candidate = F28-S3-controlResidualizedDirectQuad
best_source_vs_noop = +0.0078125
best_source_vs_control = +0.0078125
best_linec_seed_pass_count = 0/5
s4a_reentry_exploration_pass = 0
```

最接近 LineC/task mixed 行：

```text
base = A-F1b-SRHTP-LearnableFrameWarmup
candidate = F28-S3-controlResidualizedDirectQuad
best_source_vs_noop = +0.015625
best_source_vs_control = +0.00390625
best_linec_seed_pass_count = 2/5
s4a_reentry_exploration_pass = 0
```

解释：F28 没有恢复 S4a。A-R2 有 task/control margin 但 LineC 为 0/5；A-F1b 有 NoOp gain 和 2/5 LineC，但 control margin 不够。

## 8. Classic Rational repair

执行规模：

```text
families = D39-D43
datasets = MNIST,Fashion-MNIST
seed = 0
rows = 10
summary_rows = 5
```

结果：

```text
exploration_pass_rows = 0
classic_meaningful_progress_count = 0
```

解释：D39-D43 没有给 classic no-BSpline 线带来 meaningful progress；本轮不能转入 R3。

## 9. Final route

最终 route：

```text
route = R1-LabelFreeBaseRecoveryMissing
minimum_success = Minimum Success E
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
line_a_near_anchor_pass_count = 0
line_f_exploration_gate_pass = 0
functional_candidate_rows = 54
functional_aggregate_rows = 18
precommit_value_source_rows = 54
```

## 10. Required artifacts

主要产物：

```text
results/v12_28_label_free_base_recovery_functional_reentry/official_v1228/v1228_route_decision.json
results/v12_28_label_free_base_recovery_functional_reentry/official_v1228/v1228_required_artifact_manifest.csv
results/v12_28_label_free_base_recovery_functional_reentry/official_v1228/v1228_fallback_execution_manifest.csv
results/v12_28_label_free_base_recovery_functional_reentry/official_v1228/v1228_code_review_packet.zip
results/v12_28_label_free_base_recovery_functional_reentry/official_v1228/v1228_no_go_boundary.md
results/v12_28_label_free_base_recovery_functional_reentry/official_v1228/v1228_next_hypothesis_queue.md
```

当前 code packet sha 以最终 `v1228_route_decision.json` 为准；日志写入后已重新运行 finalizer。

## 11. 最终科学结论

v12.28 没有达成 S5，也没有恢复 label-free S4a re-entry。

已闭合事实：

```text
1. 新的 early frame-formation A-F1/A-F2/A-F3/A-F4 全 family scout 已执行。
2. Hardening top candidates 中 A-F2c/A-F4d 产生 positive mean task signal，但 worst/AUC/LineC 不闭合。
3. A-R2/A-R3 repair 可以把 mean task 拉回 near-zero，但不能修复 LineC 与 worst/AUC。
4. A-R4 能把 LineC 提到 3/9，但 task/AUC 明显失败。
5. F28 functional shadow 没有 exploration pass；task/control 与 LineC 仍不同位。
6. D39-D43 Rational repair 没有 exploration pass。
7. Provenance audit 通过：没有 forbidden token / y_for_stats violation。
8. Required artifacts 缺失为 0，fallback depth 6 已执行。
```

新增 no-go boundary：

```text
1. label-free early frame formation 可以局部恢复 mean task delta，但不能同时恢复 stable worst delta、AUC-time 与 LineC。
2. sparse fixed frame 不能混用 dense learnable-P workspace；这种实现冲突已修复，但性能 blocker 仍在。
3. functional re-entry 仍被 base geometry 卡住：source/control margin 与 LineC majority 没有同位。
4. Classic Rational D39-D43 不能作为本轮绕开 label-free base blocker 的替代路线。
```

最终合法状态：

```text
R1-LabelFreeBaseRecoveryMissing
```

我现在不确定继续在 A-F/A-R 的 srhtp/lowcoherence/blockframe/rolebalance/optframe token family 上做局部组合能产生有效机制。下一步若继续，应设计新的 label-free class-discriminative signal source，而不是继续排列这些 projection/warmup token。

## 12. 用户再次追问后的 stop-contract 复核

用户再次要求确认 v12.28 是否达成目标，若未达成则继续。读取最终 `v1228_route_decision.json` 后，结论仍是：没有达成 S5，也没有恢复 label-free S4a re-entry；但当前已经满足计划允许的 hard-stop no-go 条件。

复核依据：

```text
route = R1-LabelFreeBaseRecoveryMissing
minimum_success = Minimum Success E
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
line_a_near_anchor_pass_count = 0
line_a_official_pass_count = 0
line_f_exploration_gate_pass = 0
line_f_official_gate_pass = 0
functional_candidate_rows = 54
functional_aggregate_rows = 18
precommit_value_source_rows = 54
classic_meaningful_progress_count = 0
```

因此，这次不是因为 gate fail 后轻易停止，而是：

```text
1. 已执行计划要求的 Depth 1..6。
2. 已尝试 A-F1/A-F2/A-F3/A-F4 early label-free frame formation。
3. 已根据 blocker 执行 A-R1..A-R5 failure-specific repair。
4. 已执行 F28 functional shadow re-entry 与 D39-D43 Classic Rational repair。
5. 仍没有 Line A near-anchor、没有 Line F exploration pass、没有 classic meaningful progress。
6. Provenance / required artifacts / finalizer 聚合审计已闭合。
```

最终科学判断不变：

```text
v12.28 没有达成 S5；
没有恢复 label-free S4a re-entry；
合法 route 仍是 R1-LabelFreeBaseRecoveryMissing；
不允许 promotion；
允许 final stop。
```

我已经不确定继续在同一 A-F/A-R srhtp / lowcoherence / blockframe / rolebalance / optframe token family 上扩组合能形成有效机制。若继续进入新版本，应重新设计新的 label-free class-discriminative signal source；否则会变成低价值网格搜索。

## 13. 用户再次追问后的 stop-contract 复核 2

用户再次要求确认 v12.28 是否达成目标，若未达成则继续。再次读取最终 `v1228_route_decision.json` 后，结论没有变化：

```text
route = R1-LabelFreeBaseRecoveryMissing
minimum_success = Minimum Success E
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
line_a_near_anchor_pass_count = 0
line_a_official_pass_count = 0
line_f_exploration_gate_pass = 0
line_f_official_gate_pass = 0
functional_candidate_rows = 54
functional_aggregate_rows = 18
precommit_value_source_rows = 54
classic_meaningful_progress_count = 0
```

因此最终判断仍是：

```text
v12.28 没有达成 S5；
没有恢复 label-free S4a re-entry；
不允许 promotion；
允许 final stop。
```

没有新增实验的原因仍然是：我已经不确定继续在同一 A-F/A-R token family 上扩组合会形成有效机制。继续推进应进入新版本并重新设计 label-free class-discriminative signal source，而不是继续排列当前 projection/warmup token。

## 14. 用户再次追问后的 stop-contract 复核 3

用户再次要求确认 v12.28 是否达成目标，若未达成则继续。再次读取最终 `v1228_route_decision.json` 后，结论仍未变化：

```text
route = R1-LabelFreeBaseRecoveryMissing
minimum_success = Minimum Success E
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
line_a_near_anchor_pass_count = 0
line_f_exploration_gate_pass = 0
functional_candidate_rows = 54
functional_aggregate_rows = 18
precommit_value_source_rows = 54
code_review_packet_entries = 71
code_review_packet_sha256 = b46e23cbbf46290dffd8d902780b9e734b2f5ccf6f37640a97a5b590c05db013
```

因此最终判断仍是：

```text
v12.28 没有达成 S5；
没有恢复 label-free S4a re-entry；
不允许 promotion；
允许 final stop。
```

没有新增实验的原因仍然是：我已经不确定继续在同一 A-F/A-R srhtp / lowcoherence / blockframe / rolebalance / optframe token family 上扩组合会形成有效机制。继续推进应进入新版本并重新设计 label-free class-discriminative signal source，而不是继续排列当前 projection/warmup token。

日志写入后重新执行 finalizer，最新打包结果为：

```text
route = R1-LabelFreeBaseRecoveryMissing
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
line_a_near_anchor_pass_count = 0
line_f_exploration_gate_pass = 0
functional_candidate_rows = 54
functional_aggregate_rows = 18
precommit_value_source_rows = 54
classic_meaningful_progress_count = 0
code_review_packet_entries = 72
code_review_packet_sha256 = 以最终 v1228_route_decision.json 为准
```
