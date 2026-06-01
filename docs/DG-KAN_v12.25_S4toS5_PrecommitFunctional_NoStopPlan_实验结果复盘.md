# DG-KAN v12.25 S4toS5 PrecommitFunctional NoStopPlan 实验结果复盘

生成时间：2026-05-26（Asia/Singapore）

本复盘只写入实际执行产物中的结果；不编造未执行数据，不把 diagnostic/near-pass 写成 promotion。

## 1. 计划理解

v12.25 的目标不是继续扩大 I30/I32 小网格，而是把 v12.24 的 partial signals 组合成一个可部署的 precommit functional bridge：

```text
direct/gain 提供 task signal；
quad/reservoir 维持 LineC；
control-residualization 避免 matched controls 追平；
loss-agnostic tail proxy 防 CEp99 爆；
policy-aware P3/P4 alignment 约束 promotion。
```

计划要求不能在第一层或第二层 gate fail 后停止，必须执行 depth-3 fallback，然后由 finalizer 写 route、required artifact manifest、fallback manifest、code packet 和 no-go boundary。

## 2. 本轮新增代码修改

### 2.1 rank-aware residual frame

修改文件：

```text
dgkan/models/fc_purekan_primitives.py
```

新增逻辑：

```text
reslowrankrNNN
```

用于限制低秩 residual frame 只影响前 N 个 projector columns。这样 A70/A71/A72/A73 的 rank8/rank16 候选不是只改名字，而是有实际机制差异。

合理性：只扩展 label-free residual projector 初始化解析；不读取 label/CE，不改变 gate 阈值。

### 2.2 A70-A76

修改文件：

```text
experiments/run_v1218_b320_label_free_ablation.py
```

新增候选：

```text
A70-A51ResidualLineCFrame-rank8
A71-A51ResidualLineCFrame-rank16
A72-A51ControlResidualFrame-rank8
A73-A51ControlResidualFrame-rank16
A74-A51DirectQuadBalance-noY
A75-A51RoleEnergyTailClamp-noY
A76-A51LineCEMAAdapt-noY
```

合理性：这些候选均为 label-free A51 repair，不使用 label/CE direction，也没有降低 Line A gate。

### 2.3 D25-D29

修改文件：

```text
experiments/run_v1224_classic_hardening.py
```

新增 v12.25 Rational focused aliases：

```text
D25-RationalMemoryCut-readscaleShared
D26-RationalCouplingLift-noWhiten
D27-RationalPairNormStopGrad-light
D28-RationalTaskTrajectoryWarmNoExtraMem
D29-RationalB7lpLowMemCouplingMix
```

合理性：复用已有 primitive specs 进行计划要求的 Rational-focused hardening；不扩 Fourier 大网格，不降低 Line D gate。

### 2.4 v12.25 runner/finalizer

新增文件：

```text
experiments/run_v1225_composite_functional_bridge.py
experiments/run_v1225_finalize_precommit_functional.py
```

用途：

- Line F：执行 F25-C1/C2/C3/C4 和 depth-3 composite fallback。
- Line T：从实际 composite rows 生成 precommit value-source artifact。
- Line P：从实际 composite rows 生成 policy-aware P3/P4 alignment artifact。
- Line R/Z：生成 code/provenance manifests、required artifact manifest、fallback manifest、SVG figures、route、code review packet。

语法检查已通过。

## 3. Line A：A70-A76 residual LineC repair

执行规模：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
train_size = 1024
val/test = 512
epochs = 8
raw rows = 108
summary rows = 12
```

route JSON 初始输出：

```text
best_label_free_candidate = A51-StagedUnlabeledAdapt-warm1-r002
best_label_free_mean_delta_vs_A0 = -0.00390625
```

解释：A70-A76 已实际执行。最终 Line A gate 由 finalizer 基于 `v1225_label_free_residual_linec_summary.csv` 统一判定。

## 4. Line F：composite functional bridge

### 4.1 第一轮 C1-C4

第一轮执行后：

```text
candidate_rows = 15
aggregate_rows = 5
any_exploration_pass = 0
any_strict_majority_pass = 0
any_strict_all_pass = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
combined_bridge_any_train_shuffle_robust_all_pass = 0
```

关键边界：

```text
F25-C2-controlResidual-I27:
  best_source_vs_control = +0.00390625
  best_source_vs_noop = +0.01171875
  best_linec_seed_pass_count = 0/5

F25-C3-tailBudgetedQuadDirect-I26:
  best_linec_seed_pass_count = 3/5
  best_source_vs_control = -0.0078125
```

结论：C1-C4 没有打开 S4a/S5。按计划继续 depth-3，不停止。

### 4.2 Depth-3 rerun

新增 depth-3 candidates：

```text
F25-D3-controlResidualQuadDirect-I27
F25-D3-tailBudgetedDirectBranch-I26
F25-D3-lowBudgetDirectOnly-I26
```

重跑后：

```text
candidate_rows = 24
aggregate_rows = 8
any_exploration_pass = 0
any_strict_majority_pass = 0
any_strict_all_pass = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
combined_bridge_any_train_shuffle_robust_all_pass = 0
```

最接近 task/control 的行：

```text
F25-D3-tailBudgetedDirectBranch-I26:
  best_source_vs_control = +0.0078125
  best_source_vs_noop = +0.01171875
  best_linec_seed_pass_count = 0/5
```

最接近 LineC 的行：

```text
F25-D3-controlResidualQuadDirect-I27:
  best_linec_seed_pass_count = 4/5
  best_source_vs_control = +0.00390625
  best_source_vs_noop = +0.0078125
```

解释：depth-3 证明两类 partial signals 仍然不同位：direct-branch 可以达到 control margin 但 LineC 失败；controlResidualQuadDirect 可以达到 LineC 4/5，但 task margin 不足。

## 5. Line D：D25-D29 Rational focused hardening

执行规模：

```text
families = D25,D26,D27,D28,D29
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
rows = 45
summary rows = 5
exploration_pass_rows = 0
```

解释：Rational-focused depth-3 已执行，但没有打开 S4c。具体最接近行由 finalizer/只读复核补充。

## 6. Finalizer

finalizer 将生成并汇总：

```text
v1225_precommit_value_source.csv
v1225_policy_aware_p3.csv
v1225_core_code_review_manifest.csv
v1225_required_artifact_manifest.csv
v1225_fallback_execution_manifest.csv
v1225_route_decision.json
v1225_code_review_packet.zip
```

首次 finalizer 已生成 route，但 required manifest 中 `v1225_route_decision.json` 与 `v1225_required_artifact_manifest.csv` 在 manifest 计算时还未落盘，因此首次为：

```text
required_artifact_missing_count = 2
final_stop_allowed = 0
```

这不是实验 artifact 缺失，而是 finalizer 自引用写入顺序导致的闭合问题。复跑 finalizer 后，自引用 artifact 已闭合：

```text
required_artifact_missing_count = 0
final_stop_allowed = 1
```

## 7. 最终 route

最终 route：

```text
route = R4-S4toS5NoGoAfterDepth3Fallbacks
minimum_success = Minimum Success F
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 3
fallback_rows = 7
fallback_all_executed = 1
required_artifact_rows = 29
required_artifact_missing_count = 0
code_semantics_review_pass = 1
transitive_code_packet_missing_count = 0
```

结论：v12.25 没有达成 S5，也没有打开 S1/S2/S4a/S4c。当前停止是 Depth-3 fallback 全部执行、required artifact 无缺失后的 fail-closed no-go，不是 promotion success。

## 8. Line A 最终结果

最终 Line A：

```text
raw rows = 108
summary rows = 12
best candidate = A51-StagedUnlabeledAdapt-warm1-r002
best mean_delta_vs_A0 = -0.00390625
best worst_delta_vs_A0 = -0.0234375
best LineC pass rate = 0.0
exploration pass count = 0
official pass count = 0
```

A70-A76 关键结果：

| candidate | mean_delta_vs_A0 | worst_delta_vs_A0 | max_AUC_time_ratio_vs_mlp | LineC pass rate | exploration |
|---|---:|---:|---:|---:|---:|
| A51-StagedUnlabeledAdapt-warm1-r002 | -0.00390625 | -0.0234375 | 1.20206577701635 | 0.0 | 0 |
| A70-A51ResidualLineCFrame-rank8 | -0.015625 | -0.0390625 | 1.3096323430450634 | 0.0 | 0 |
| A71-A51ResidualLineCFrame-rank16 | -0.015625 | -0.0390625 | 1.3096322559962488 | 0.0 | 0 |
| A72-A51ControlResidualFrame-rank8 | -0.027994791666666668 | -0.07421875 | 1.5078421745025996 | 0.0 | 0 |
| A73-A51ControlResidualFrame-rank16 | -0.027994791666666668 | -0.07421875 | 1.5078424066327716 | 0.0 | 0 |
| A74-A51DirectQuadBalance-noY | -0.004123263888888889 | -0.0234375 | 1.201977507787727 | 0.0 | 0 |
| A75-A51RoleEnergyTailClamp-noY | -0.10416666666666667 | -0.13671875 | 3.329845428175645 | 0.0 | 0 |
| A76-A51LineCEMAAdapt-noY | -0.015842013888888888 | -0.0390625 | 1.3096832279131891 | 0.0 | 0 |

解读：

- A74 几乎追平 A51 的 task/AUC，但 LineC 仍为 0。
- rank-aware residual repair A70/A71 没有改善 LineC，且 task/AUC 变差。
- A72/A73/A75 明显破坏 task 或效率。
- 因此 Line A blocker 仍是 label-free task 近似恢复与 LineC non-tearing 无法同位。

## 9. Line F/P：precommit functional bridge

Depth-3 后：

```text
candidate_rows = 24
aggregate_rows = 8
any_exploration_pass = 0
any_strict_majority_pass = 0
any_strict_all_pass = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
combined_bridge_any_train_shuffle_robust_all_pass = 0
line_f_exploration_gate_pass = 0
line_f_official_gate_pass = 0
```

最接近的两类 partial signal：

```text
F25-D3-tailBudgetedDirectBranch-I26:
  best_source_vs_control = +0.0078125
  best_source_vs_noop = +0.01171875
  best_linec_seed_pass_count = 0/5

F25-D3-controlResidualQuadDirect-I27:
  best_linec_seed_pass_count = 4/5
  best_source_vs_control = +0.00390625
  best_source_vs_noop = +0.0078125
```

Line P policy alignment：

```text
policy_aware_p3_rows = 24
P3_to_P4_rank_alignment = 0.014782608695652174
policy_alignment_exploration_pass = 0
policy_alignment_official_pass = 0
```

解读：

- F25-D3-tailBudgetedDirectBranch-I26 达到 official source-control margin，但 source-noop 不足且 LineC 全失败。
- F25-D3-controlResidualQuadDirect-I27 达到 LineC majority 4/5，但 task/control margin 不足。
- 这说明 v12.25 的 precommit-safe bridge 仍然把 task/control 信号和 LineC 几何信号分散到不同候选上，不能形成 S4a/S5。

## 10. Line T：precommit value source

finalizer 基于 24 行 bridge 结果生成 value-source audit：

```text
precommit_value_source_rows = 24
AUC_source_control = 0.9625
AUC_tail_safe = 1.0
AUC_linec_majority = 1.0
precision_at_k_s5_proxy = 0.0
recall_at_k_s5_proxy = ""
exploration_visibility_gate_pass = 0
```

解读：

- source/control、tail-safe、LineC majority 这些单目标可被区分。
- 但没有 candidate 同时成为 S5 proxy 正例，因此 `precision_at_k_s5_proxy=0.0`，recall 不定义。
- 这不能作为可部署 value source，也不能打开 S2。

## 11. Line D：Rational-focused hardening

执行规模：

```text
families = D25,D26,D27,D28,D29
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
rows = 45
linec_pass_rows = 13
exploration_pass_rows = 0
```

family summary：

| family | rows | mean_delta_vs_MLP | mean CouplingR2 | mean NoiseLeak | mean Reservoir | exploration |
|---|---:|---:|---:|---:|---:|---:|
| D25-RationalMemoryCut-readscaleShared | 9 | -0.18836805555555555 | 0.11870701292145282 | 0.08543338129917781 | 0.550383425421185 | 0 |
| D26-RationalCouplingLift-noWhiten | 9 | -0.046875 | 0.13866132674462833 | 0.03826425229716632 | 0.24464679757754007 | 0 |
| D27-RationalPairNormStopGrad-light | 9 | -0.047526041666666664 | 0.13561804238057626 | 0.049300960989462 | 0.212693452835083 | 0 |
| D28-RationalTaskTrajectoryWarmNoExtraMem | 9 | -0.046875 | 0.13847373825165799 | 0.04843843438559108 | 0.21427000479565728 | 0 |
| D29-RationalB7lpLowMemCouplingMix | 9 | -0.1953125 | 0.1201312396722751 | 0.08030040313800176 | 0.5294496119022369 | 0 |

最接近 exploration 的强行：

```text
D26 / Fashion-MNIST / seed 2:
  val_acc = 0.806640625
  MLP val_acc = 0.80859375
  mean_delta_vs_MLP = -0.001953125
  step_ratio_vs_mlp = 1.1915416333609372
  memory_ratio_vs_mlp = 2.1016346983201215
  CouplingR2 = 0.20949461354821908
  NoiseSignalLeak = 0.017755774781107903
  RealSignalReservoirRatio = 0.4077603816986084
  linec_pass = 1
  exploration_gate_pass = 0
```

```text
D28 / Fashion-MNIST / seed 2:
  val_acc = 0.806640625
  MLP val_acc = 0.80859375
  mean_delta_vs_MLP = -0.001953125
  step_ratio_vs_mlp = 1.201647180307022
  memory_ratio_vs_mlp = 2.1016346983201215
  CouplingR2 = 0.20969056041624123
  NoiseSignalLeak = 0.045623552054166794
  RealSignalReservoirRatio = 0.40404945611953735
  linec_pass = 1
  exploration_gate_pass = 0
```

解读：

- D26/D28 在 Fashion-MNIST seed2 上几乎追平 MLP，并且 LineC pass。
- 仍未打开 S4c 的主 blocker 是 memory ratio 约 `2.1016`，超过计划 gate；跨 dataset 汇总也不稳定。
- D25/D29 的 memory-cut/mix 方向没有真正降低 memory ratio，反而拉大 task gap。

## 12. 代码审计与 artifact 完整性

最终审计：

```text
code_semantics_review_pass = 1
transitive_code_packet_missing_count = 0
uses_label_or_ce_for_direction = 0
uses_query_batch_for_promotion = 0
uses_validation_or_test_for_commit = 0
matched_control_scope_pass = 1
dataset_name_branch_count = 0
required_artifact_missing_count = 0
```

Code review packet：

```text
path = results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/v1225_code_review_packet.zip
entries = 53
sha256 = see final v1225_route_decision.json
```

本复盘写入后会再次执行 finalizer，因此最终 zip sha256 以最后一次 `v1225_route_decision.json` 为准，避免日志内容与 zip sha 互相递归。

最终打包命令记录在执行日志第 9 节；最终 route、manifest 和 zip sha256 以该命令完成后的 `v1225_route_decision.json` 为准。

## 13. 最终科学结论

v12.25 没有达成目标，不能写 S5。

已经闭合的事实：

1. Line A A70-A76 已执行，但没有任何 LineC non-tearing pass。
2. Line F C1-C4 与 Depth-3 D3 candidates 已执行；task/control 与 LineC majority 信号仍不同位。
3. Line T 可区分单目标 proxy，但没有 S5 proxy 正例，因此 value source 不可 promotion。
4. Line P alignment 很低，不能说明 P3/P4 排名一致。
5. Line D D25-D29 已执行；D26/D28 出现接近 MLP 的 Fashion-MNIST near-pass，但 memory ratio 仍卡住，S4c 未打开。
6. R/A/F/T/P/D/Z fallback manifest 全部 executed，Depth-3 与 hard budget 均已记录。

最终状态：

```text
R4-S4toS5NoGoAfterDepth3Fallbacks
```

下一步若进入新版本，最有价值的方向不是继续扩大同一 bridge 网格，而是：

- Line F：把 `F25-D3-tailBudgetedDirectBranch` 的 task/control margin 与 `F25-D3-controlResidualQuadDirect` 的 LineC majority 做结构性合并，而不是只调 scale。
- Line D：优先解决 D26/D28 的 memory ratio；它们已经在 Fashion-MNIST seed2 上接近 task+LineC gate。
- Line T：重新定义能预测“联合正例”的 precommit observable，单目标 AUC 高但对 S5 proxy precision 为 0。

## 14. 用户追问后继续推进：NG19-NG30 与 D30-D33

### 14.1 为什么继续

上一轮已经达到 v12.25 Depth-3 fail-closed stop contract，但用户明确要求“未达成 S5 就继续”。因此本轮继续沿第 13 节推荐方向推进两条最明确路径：

```text
1. Line F：把 task/control-positive 的 I26 direct-branch 信号和 LineC-positive 的 I27 control-residual 信号做真实结构合并。
2. Line D：针对 D26/D28 的 memory blocker，测试 cheaper/no-pair/read-bucket/manual Rational 变体。
```

### 14.2 本轮代码修改

修改文件：

```text
experiments/run_v1225_composite_functional_bridge.py
experiments/run_v1224_classic_hardening.py
experiments/run_v1225_finalize_precommit_functional.py
```

修改内容：

- `run_v1225_composite_functional_bridge.py` 支持 `I26@scale+I27@scale` 语法，对同一个 source model 顺序应用两个 actuator。这样 NG19-NG30 是真实结构合并，不是只改审计字段。
- 新增 NG19-NG30，包括 task-first、linec-first、direct-branch linec-dominant 和 budget sweep。
- `run_v1224_classic_hardening.py` 新增 D30-D33：
  - `D30-RationalCheaperR120Pair`
  - `D31-RationalNoPairReadout`
  - `D32-RationalBucketG32Manual`
  - `D33-RationalNoPairManual`
- `run_v1225_finalize_precommit_functional.py` 把 NG19-NG22 structural merge 与 D30-D33 memory repair 纳入 code review/fallback manifest。

合理性判断：

- 没有降低 official/exploration gate。
- 没有使用 query batch、label、CE 或 validation/test 构造 promotion direction。
- AdamW/CE 仍只作为 audit/control，不作为 source。

### 14.3 NG19-NG22 结果

第一次 structural merge 后：

```text
candidate_rows = 36
aggregate_rows = 12
any_exploration_pass = 0
any_strict_majority_pass = 0
any_strict_all_pass = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
```

关键新增候选：

```text
F25-NG20-linecFirstMerge-I27I26:
  best_source_vs_noop = +0.01953125
  best_source_vs_control = +0.00390625
  best_linec_seed_pass_count = 2/5
```

```text
F25-NG22-balancedMerge-directOnlyLineCGuard:
  best_source_vs_noop = +0.01171875
  best_source_vs_control = -0.00390625
  best_linec_seed_pass_count = 0/5
```

结论：NG20 是新增近似点，task/noop 与 control margin 接近，但 LineC 未到 majority。

### 14.4 D30-D33 memory repair 结果

执行规模：

```text
families = D25-D33
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
rows = 81
linec_pass_rows = 25
exploration_pass_rows = 0
```

D30-D33 最接近行：

```text
D32 / Fashion-MNIST / seed 2:
  val_acc = 0.806640625
  MLP val_acc = 0.80859375
  mean_delta_vs_MLP = -0.001953125
  step_ratio_vs_mlp = 1.1230958059082221
  memory_ratio_vs_mlp = 2.1016346983201215
  CouplingR2 = 0.21031996243230244
  linec_pass = 1
  exploration_gate_pass = 0
```

```text
D31 / Fashion-MNIST / seed 2:
  val_acc = 0.802734375
  MLP val_acc = 0.80859375
  mean_delta_vs_MLP = -0.005859375
  step_ratio_vs_mlp = 1.041847792376353
  memory_ratio_vs_mlp = 2.1016346983201215
  CouplingR2 = 0.21204527028758458
  linec_pass = 1
  exploration_gate_pass = 0
```

结论：

- D30-D33 没有降低 memory ratio；所有最接近行仍为约 `2.1016346983201215`。
- D31/D32 在 Fashion-MNIST seed2 上接近 task+LineC，但 memory gate 仍阻断 S4c。
- 因此 D30-D33 没有解决 Line D 的核心 blocker。

### 14.5 NG23-NG30 结果

针对 NG20 的 LineC 不足，继续测试 linec-dominant 和 direct-branch budget repair。最终 Line F：

```text
candidate_rows = 60
aggregate_rows = 20
any_exploration_pass = 0
any_strict_majority_pass = 0
any_strict_all_pass = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
combined_bridge_any_train_shuffle_robust_all_pass = 0
```

最接近的新增行：

```text
F25-NG27-directBranchLinecMerge-I27I26-8515:
  best_source_vs_control = +0.0078125
  best_source_vs_noop = +0.015625
  best_linec_seed_pass_count = 3/5
  train_shuffle_exploration_pass_count = 0
```

这看起来接近 gate，但细查三条 train-shuffle seed 后发现信号不同位：

```text
train_seed_base = 12240400:
  source_vs_noop = 0.0
  source_vs_control = -0.01171875
  LineC = 3/5
  CEp99_delta = +0.007875204086303711

train_seed_base = 12241400:
  source_vs_noop = +0.015625
  source_vs_control = +0.0078125
  LineC = 0/5
  CEp99_delta = +0.0004336833953857422

train_seed_base = 12242400:
  source_vs_noop = -0.00390625
  source_vs_control = -0.01171875
  LineC = 0/5
  CEp99_delta = -0.01957106590270996
```

结论：

- NG27 的 aggregate best 把 task-positive 和 LineC-positive 统计到了不同 train-shuffle seed 上。
- 单行没有同时满足 task/control、LineC majority 与 robustness。
- 因此它不是 exploration pass，更不能写 S5。

NG29 high-budget 也没有修复：

```text
best_source_vs_control = +0.01171875
best_source_vs_noop = +0.0078125
best_linec_seed_pass_count = 0/5
```

这说明提高 budget 可以增强 control margin，但会失去 LineC。

### 14.6 finalizer 聚合

本轮最终 finalizer：

```text
route = R4-S4toS5NoGoAfterDepth3Fallbacks
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
fallback_rows = 9
fallback_all_executed = 1
required_artifact_missing_count = 0
precommit_value_source_rows = 60
policy_aware_p3_rows = 60
precision_at_k_s5_proxy = 0.08333333333333333
recall_at_k_s5_proxy = 1.0
line_f_exploration_gate_pass = 0
line_d_exploration_pass_rows = 0
```

解读：

- value source 现在能在 top-k 命中少量 S5 proxy，因此 recall 变成 `1.0`，但 precision 只有 `0.08333333333333333`，低于 exploration 可用性要求。
- policy alignment 仍失败；Line F 和 Line D 都没有打开 exploration。

### 14.7 本轮结论

继续推进后仍未达成目标。

新增边界更明确：

```text
1. 真实 I26/I27 结构合并可以制造接近 gate 的 aggregate near-pass。
2. 但 task-positive 和 LineC-positive 不在同一 train-shuffle seed 上，robust gate 仍失败。
3. D30-D33 低/无 pair-readout 方向没有降低 memory ratio，因此 Line D memory blocker 未解除。
4. 当前 precommit value source 对联合成功的 precision 太低，不能作为 deployable selector。
```

因此当前仍不能 promotion，合法状态仍是：

```text
R4-S4toS5NoGoAfterDepth3Fallbacks
```

进一步继续若不进入新机制，边际价值已经很低：单纯调 I26/I27 scale 只是在 task 和 LineC seed 间移动，不会自然产生 robust co-location。下一轮需要新的机制：例如显式训练/构造一个 precommit observable 来预测“同一 train seed 上 task+LineC 同位”，或重新设计 actuator 使 LineC-positive update 本身带 task gain，而不是后加 I26 task component。

### 14.8 日志纳入后的最终打包复核

为确保本轮新增执行日志和复盘日志也进入 code review packet，写入日志后再次执行 finalizer。结果仍不是 S5：

```text
route = R4-S4toS5NoGoAfterDepth3Fallbacks
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
fallback_rows = 9
fallback_all_executed = 1
required_artifact_missing_count = 0
precommit_value_source_rows = 60
policy_aware_p3_rows = 60
precision_at_k_s5_proxy = 0.08333333333333333
recall_at_k_s5_proxy = 1.0
line_f_exploration_gate_pass = 0
line_d_exploration_pass_rows = 0
code_review_packet_entries = 53
code_review_packet_sha256 = 以最终 v1225_route_decision.json 为准
```

该复核只改变打包内容和 zip 哈希，不改变任何实验指标或 gate。最终结论仍是 no-go：NG19-NG30 与 D30-D33 已补充执行，但没有出现同一 train-shuffle seed 内同时满足 task/control、LineC majority/all、tail safety 与 robustness 的 precommit-safe candidate。

## 15. 用户再次追问后继续推进：NG31-NG43 response residualization 与 full-batch 反事实

### 15.1 为什么继续

上一节仍是：

```text
route = R4-S4toS5NoGoAfterDepth3Fallbacks
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
```

但 NG27/NG40 暴露的 blocker 很具体：task-positive 与 LineC-positive 不在同一 train-shuffle seed 上。因此继续做两类机制级反事实：

```text
1. 实现真正的 train-stream response residualization，而不是只记录 control_residualized=1。
2. 用 full-batch P4 训练降低 train-shuffle 随机性，检验不同位是否只是 minibatch order 噪声。
```

### 15.2 本轮代码修改

修改文件：

```text
experiments/run_v1225_composite_functional_bridge.py
experiments/run_v1225_finalize_precommit_functional.py
```

修改内容：

```text
1. 新增 apply_response_residualization()：
   用 train-stream refs 上的 source/control logit response 计算投影系数 alpha。
   从 source 参数 delta 中扣除 alpha * matched_control_delta。
   不读取 label、CE、validation/test、LineC hard target 或 query batch。

2. 新增 NG31-NG36 true response residualization：
   全量或半量扣除 matched-control response component。

3. 新增 NG37-NG42 partial response residualization：
   alpha clip = 0.25 / 0.10，并测试 tail-budget on/off。

4. 新增 NG43 full-batch colocation counterfactual：
   source/noop/control 公平使用 batch_size=512，检验 train-shuffle 随机性是否是同位失败主因。

5. finalizer 改为聚合多个 v1225_composite_functional_bridge*.csv，
   并把 NG31-NG43 纳入 fallback manifest、required artifacts 和 code review packet。
```

合理性判断：

```text
1. response residualization 是 v12.25 H3 指定方向，不是临时降低 gate。
2. full-batch 是公平训练协议反事实，source/noop/control 共用设置，不改变 loss。
3. 所有新增结果 promotion_allowed 仍为 0，由 finalizer 统一判定。
```

### 15.3 NG31-NG36 true response residualization 结果

总览：

```text
candidate_rows = 18
aggregate_rows = 6
any_exploration_pass = 0
any_strict_majority_pass = 0
any_strict_all_pass = 0
```

最佳行：

```text
F25-NG32-trueResponseResidual-I26-directGain:
  best_source_vs_control = +0.00390625
  best_source_vs_noop = +0.00390625
  best_linec_seed_pass_count = 0/5
```

结论：

```text
全量 response residualization 没有解决 matched-control 问题，反而把 LineC 信号打掉。
这说明简单扣除 control response component 不是缺失的 S5 拼图。
```

### 15.4 NG37-NG42 partial response residualization 结果

总览：

```text
candidate_rows = 18
aggregate_rows = 6
any_exploration_pass = 0
any_strict_majority_pass = 0
any_strict_all_pass = 0
```

最接近行：

```text
F25-NG40-quarterResponseResidual-quadDirect-I27I26:
  best_source_vs_control = +0.0078125
  best_source_vs_noop = +0.015625
  best_linec_seed_pass_count = 3/5
  train_shuffle_exploration_pass_count = 0
```

细查 NG40 三个 train-shuffle seed：

```text
train_seed_base = 12240400:
  source_vs_noop = -0.00390625
  source_vs_control = -0.01171875
  LineC = 3/5
  source_CEp99 = 7.553499221801758
  noop_CEp99 = 7.255672454833984
  response_residual_alpha = 0.5235296199145802
  alpha_clipped = 0.25
  response_residual_pre_cosine = 0.6838027903276747
  response_residual_post_cosine = 0.44747347847520713

train_seed_base = 12241400:
  source_vs_noop = -0.0390625
  source_vs_control = -0.01953125
  LineC = 0/5

train_seed_base = 12242400:
  source_vs_noop = +0.015625
  source_vs_control = +0.0078125
  LineC = 0/5
  source_CEp99 = 6.872147560119629
  noop_CEp99 = 9.146791458129883
```

结论：

```text
partial residualization 恢复了 aggregate near-pass，但仍是不同位：
LineC-positive seed 输 NoOp/control；task-positive seed LineC=0/5。
因此不能打开 exploration，更不能 S5。
```

### 15.5 NG43 full-batch colocation counterfactual

目标：检验 NG27/NG40 的不同位是否来自 minibatch order 随机性。

设置：

```text
batch_size = 512
train_size = 512
candidates = NG27, NG29, NG32, NG40
train_seed_bases = 12240400,12241400,12242400
```

结果：

```text
candidate_rows = 12
aggregate_rows = 4
any_exploration_pass = 0
any_strict_majority_pass = 0
any_strict_all_pass = 0
```

最佳行：

```text
F25-NG32-trueResponseResidual-I26-directGain:
  best_source_vs_control = +0.0078125
  best_source_vs_noop = +0.0078125
  best_linec_seed_pass_count = 0/5
```

结论：

```text
full-batch 没有修复同位问题，反而让测试邻域 LineC majority 消失。
因此 train-shuffle order 不是唯一 blocker；当前 update primitive 本身不能稳定同位 task/control gain 与 LineC。
```

### 15.6 finalizer 聚合

NG31-NG43 纳入 finalizer 后：

```text
route = R4-S4toS5NoGoAfterDepth3Fallbacks
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
fallback_rows = 12
fallback_all_executed = 1
required_artifact_missing_count = 0
precommit_value_source_rows = 108
policy_aware_p3_rows = 108
precision_at_k_s5_proxy = 0.045454545454545456
recall_at_k_s5_proxy = 1.0
line_f_exploration_gate_pass = 0
line_d_exploration_pass_rows = 0
```

解释：

```text
1. 聚合后 value-source precision 进一步下降到 0.04545，说明新增 residualization 候选扩大了 near-signal 空间，但没有提高可部署选择精度。
2. P3/P4 rank alignment 仍不支持 promotion。
3. required artifacts 仍无缺失，code/provenance gate 通过。
```

### 15.7 本轮最终结论

v12.25 继续推进后仍未达成目标。

新增 no-go 边界：

```text
1. control_residualized 字段已从记录变成真实 train-stream response residualization。
2. full response residualization 过强，会打掉 LineC 或 task signal。
3. partial response residualization 可恢复 aggregate near-pass，但 task 和 LineC 仍不同位。
4. full-batch 训练不能解决不同位，说明问题不是单纯 minibatch shuffle 噪声。
```

因此当前最强结论是：

```text
当前 I26/I27 train-stream direct-logit compensation + direct/quad role-policy family，
即使加入 true/partial response residualization 与 full-batch 反事实，
仍无法形成同一 train-shuffle seed 内 task/control、LineC、tail safety、robustness 同时通过的 S5 candidate。
```

这不是 S5，不能写 promotion。若下一步还要继续，应更换 functional update primitive，而不是继续在 I26/I27 response residualization、scale、batch-size 上做局部搜索。

## 16. 继续推进复盘：train-feature primitive 到 post-calibration，出现单点 exploration 但 S5 仍未闭合

### 16.1 是否达成目标

没有达成 S5 official functional success。

本轮继续后最终聚合仍为：

```text
route = R4-S4toS5NoGoAfterDepth3Fallbacks
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
```

注意：本轮出现了一个新的 `exploration_gate_pass=1` 单行，但不是 train-shuffle robust，也没有 strict majority/all pass，因此不能写 S5，也不能写 official P4 success。

### 16.2 本轮代码修改

修改文件：

```text
experiments/run_v1225_composite_functional_bridge.py
experiments/run_v1225_finalize_precommit_functional.py
```

新增内容：

```text
1. apply_train_feature_functional_actuator():
   用 train-stream unlabeled features/logits 构造 direct_readout delta，不使用 query batch、label、CE、LineC target。

2. NG44-NG49:
   train-feature functional primitive，包括 covariance / entropy-damped / confidence-damping / margin-guarded / centered-logit-denoise / low-tail direct。

3. NG50-NG57:
   centered-denoise tail repair 与 weight_decay 反事实。

4. NG58-NG63:
   train-stream tail-clipped feature primitive，记录 requested/applied scale 与 tail target。

5. NG64-NG69:
   source/noop/control 公平共享的 label smoothing tail repair。

6. NG70-NG81:
   post-train unlabeled logit calibration。校准只用 train-stream logit tail，不使用 label、CE、validation/test 或 LineC target。

7. finalizer:
   增加 R18-R23 code audit、fallback manifest、required artifact manifest 和 zip 打包条目。
```

合理性判断：

```text
1. 这些修改没有降低 official/strict gate。
2. 新增 direction source 均是 train-stream/precommit observable，不读取 query batch。
3. label smoothing 和 post-calibration 是 post-P3 训练/校准策略；source/noop/control 使用公平设置。
4. 所有新增结果仍由 finalizer 统一判定，未手工改 route。
```

### 16.3 NG44-NG49：train-feature primitive

总览：

```text
candidate_rows = 18
aggregate_rows = 6
any_exploration_pass = 0
any_strict_majority_pass = 0
any_strict_all_pass = 0
```

最强行：

```text
candidate = F25-NG48-featureCenteredLogitDenoise
train_seed_base = 12240400
source_acc = 0.78125
noop_acc = 0.76171875
best_control_acc = 0.765625
source_vs_noop = +0.01953125
source_vs_control = +0.015625
LineC pass = 4/5
source_NLL = 0.8881343603134155
noop_NLL = 0.8951510190963745
source_ECE = 0.22354596853256226
noop_ECE = 0.2282629907131195
source_CEp99 = 9.00432014465332
noop_CEp99 = 7.980913162231445
```

结论：NG48 是本轮第一个真正把 task/control 与 LineC majority 放到同一行的 precommit-safe primitive，但 CEp99 比 NoOp 高约 `1.0234`，超过 exploration 容忍 `+0.50`，因此不能 promotion。

### 16.4 NG50-NG57：tail repair 与 weight decay 失败

NG50-NG55：

```text
best = F25-NG54-centeredDenoiseOppositeSign
best_source_vs_control = +0.015625
best_source_vs_noop = +0.01953125
best_linec_seed_pass_count = 4
any_exploration_pass = 0
```

最强行仍被 CEp99 卡住：

```text
source_CEp99 = 9.004781723022461
noop_CEp99 = 7.980913162231445
```

NG56/NG57 weight decay 反事实：

```text
weight_decay = 0.003:
  NG48 source_CEp99 = 9.002930641174316
  noop_CEp99 = 7.979053497314453

weight_decay = 0.005:
  NG48 source_CEp99 = 9.001513481140137
  noop_CEp99 = 7.97714900970459
```

结论：weight decay 只极小降低 CEp99，不能修复 tail blocker。

### 16.5 NG58-NG63：tail-clipped primitive

总览：

```text
candidate_rows = 18
aggregate_rows = 6
any_exploration_pass = 0
best = F25-NG59-centeredDenoiseTailClippedHigh
best_source_vs_control = +0.015625
best_source_vs_noop = +0.02734375
best_linec_seed_pass_count = 1
```

最强 task 行：

```text
train_seed_base = 12242400
source_acc = 0.77734375
noop_acc = 0.75
best_control_acc = 0.76171875
source_vs_noop = +0.02734375
source_vs_control = +0.015625
source_CEp99 = 7.354544639587402
noop_CEp99 = 8.30613899230957
LineC pass = 0/5
```

结论：tail-clipped primitive 可以让 task/control 与 CEp99 同时成立，但会丢掉 LineC majority。它把 blocker 从 CEp99 转成 LineC，不是 S5 修复。

### 16.6 NG64-NG69：fair label smoothing

总览：

```text
candidate_rows = 18
aggregate_rows = 6
any_exploration_pass = 0
best = F25-NG67-centeredDenoiseOppositeSmooth001
best_source_vs_control = +0.01171875
best_source_vs_noop = +0.01953125
best_linec_seed_pass_count = 3
```

最强 task 行：

```text
train_seed_base = 12241400
source_acc = 0.7734375
noop_acc = 0.75390625
best_control_acc = 0.76171875
source_vs_noop = +0.01953125
source_vs_control = +0.01171875
source_CEp99 = 7.443153381347656
noop_CEp99 = 7.644662380218506
LineC pass = 0/5
```

另一条 LineC-positive 行为：

```text
train_seed_base = 12242400
LineC pass = 3/5
source_vs_noop = -0.01953125
source_vs_control = -0.01171875
```

结论：label smoothing 能降低 tail risk，但 task-positive 与 LineC-positive 仍不同位。

### 16.7 NG70-NG75：post-train unlabeled logit calibration

总览：

```text
candidate_rows = 18
aggregate_rows = 6
best = F25-NG70-centeredDenoisePostCalBaseTail
best_source_vs_control = +0.015625
best_source_vs_noop = +0.02734375
best_linec_seed_pass_count = 3
train_shuffle_exploration_pass_count = 1
any_strict_majority_pass = 0
```

关键单点：

```text
candidate = F25-NG70-centeredDenoisePostCalBaseTail
train_seed_base = 12242400
source_acc = 0.77734375
noop_acc = 0.75
best_control_acc = 0.76171875
source_vs_noop = +0.02734375
source_vs_control = +0.015625
LineC pass = 3/5
exploration_gate_pass = 1
source_CEp99 = 2.423889636993408
noop_CEp99 = 2.4459640979766846
source_NLL = 1.8262879848480225
noop_NLL = 1.7623122930526733
source_ECE = 0.66399085521698
noop_ECE = 0.6329833269119263
post_logit_scale = 0.0628002915598666
```

解释：

```text
1. post-train unlabeled logit calibration 成功把 CEp99 blocker 压下去。
2. 该单点满足 exploration task/control + CEp99 + LineC majority。
3. 但它不是 strict success：NLL/ECE 明显劣于 NoOp，且只有 1/3 train-shuffle positive。
4. 因此它只能记录为 audit/near-pass，不能写 S5。
```

### 16.8 NG76-NG81：post-calibration target scan

总览：

```text
candidate_rows = 18
aggregate_rows = 6
any_exploration_pass = 0
best = F25-NG80-centeredDenoisePostCalTail120NoBudget
best_source_vs_control = +0.01171875
best_source_vs_noop = +0.01953125
best_linec_seed_pass_count = 3
```

最强行：

```text
train_seed_base = 12240400
source_acc = 0.76953125
noop_acc = 0.7578125
best_control_acc = 0.7578125
source_vs_noop = +0.01171875
source_vs_control = +0.01171875
LineC pass = 3/5
source_CEp99 = 2.4736194610595703
noop_CEp99 = 2.4739267826080322
source_NLL = 1.6972846984863281
noop_NLL = 1.7069443464279175
source_ECE = 0.6339539885520935
noop_ECE = 0.6276299953460693
```

失败原因：

```text
source_vs_noop = +0.01171875 < exploration gate +0.015625
source_ECE > noop_ECE + 0.02? no, ECE within tolerance
strict margin still insufficient
```

结论：更宽 calibration target 或 no-tail-budget 能改善 NLL，但没有复现 NG70 的 exploration-positive row，也没有 train-shuffle robustness。

### 16.9 最终聚合

finalizer 聚合：

```text
route = R4-S4toS5NoGoAfterDepth3Fallbacks
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_rows = 18
fallback_all_executed = 1
required_artifact_missing_count = 0
precommit_value_source_rows = 234
policy_aware_p3_rows = 234
exploration_visibility_gate_pass = 1
precision_at_k_s5_proxy = 0.2765957446808511
recall_at_k_s5_proxy = 1.0
line_f_exploration_gate_pass = 0
line_f_official_gate_pass = 0
line_d_exploration_pass_rows = 0
```

新增边界：

```text
1. v12.25 不再是“没有任何 precommit co-location signal”：
   NG48 给出 task/control + LineC majority + NLL/ECE good，但 CEp99 fail。

2. tail clipping / label smoothing 能修 CEp99，但通常破坏 LineC 或 task/control co-location。

3. post-train unlabeled logit calibration 第一次给出单点 exploration row：
   task/control + CEp99 + LineC majority 同时成立。

4. 该单点仍不 robust，且 NLL/ECE strict gate 不成立。
```

因此 v12.25 仍未达成 S5。当前最精确 blocker 已推进为：

```text
precommit-safe train-feature primitive 可以产生 S4a-like 单点；
但要 official S5，还必须让 post-calibration 不显著恶化 NLL/ECE，
并让同一机制跨 train-shuffle seed 稳定。
```

## 17. 再次继续后的复盘：NG82-NG106 从单点 exploration 推进到单点 strict，但 S5 仍未闭合

### 17.1 是否达成目标

仍未达成 S5 official functional success。

本轮继续前的最新边界是：

```text
NG70 post-train unlabeled logit calibration 出现 1 个 exploration_gate_pass=1 单点；
但 strict_majority_pass=0，train-shuffle robust=0。
```

本轮继续后，最强信号进一步推进为：

```text
NG106 bootstrap_mom reference + weight anchor 出现 1 个 strict_majority_pass=1 单点；
但 strict_all_pass=0，train_shuffle_robust_majority_pass=0，train_shuffle_robust_all_pass=0。
```

因此这仍不是 S5，也不能写 promotion。

### 17.2 本轮代码修改是否合理

修改文件：

```text
experiments/run_v1225_composite_functional_bridge.py
experiments/run_v1225_finalize_precommit_functional.py
```

修改内容：

```text
1. 新增 train_seed_salt：修复候选名长度进入训练 seed 的混杂，允许 NG82+ 与 NG70 使用同一训练随机 salt。
2. 新增 NG82-NG87：exact NG70 seed-salt，高 post-calibration target 扫描。
3. 新增 NG88-NG93：exact NG70 seed-salt，低 post-calibration target 扫描。
4. 新增 interpolate_model_state：对 source/noop/control 全部公平做 post-train weight anchor。
5. 新增 NG94-NG105：weight-anchor + post-calibration scan/fine scan。
6. 新增 bootstrap_mom reference mode：只用无标签 train x 构造 bootstrap micro-reference chunks；不读 label、CE、LineC target、validation/test 或 query batch。
7. finalizer 增加 R24-R28 审计，并将 NG82-NG106 artifact 纳入 fallback manifest、required artifact manifest 与 code review packet。
```

审计判断：

```text
1. 本轮所有新增 direction/reference/calibration 均为 train-stream/precommit observable。
2. CE/NLL/ECE/CEp99 仍只作为 audit gate，不用于选择方向或调 gate。
3. post-weight anchor 对 source/noop/control 同时应用，不是 source-only trick。
4. 没有降低 strict gate、exploration gate 或 robustness gate。
```

### 17.3 NG82-NG87：exact seed-salt 高 target scan

目的：验证 NG76-NG81 没有复现 NG70 是否被候选名长度改变训练 seed 污染。

结果：

```text
candidate_rows = 18
aggregate_rows = 6
any_strict_majority_pass = 0
any_strict_all_pass = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
best_source_vs_control = 0.015625
best_source_vs_noop = 0.02734375
best_linec_seed_pass_count = 3
train_shuffle_exploration_pass_count = 1/3 for each NG82-NG87
```

关键行均为 `train_seed_base=12242400`。例如 NG82：

```text
source_acc = 0.77734375
noop_acc = 0.75
best_control_acc = 0.76171875
source_vs_noop = 0.02734375
source_vs_control = 0.015625
LineC = 3/5
source_NLL = 1.8051130771636963
noop_NLL = 1.7390187978744507
source_ECE = 0.660231351852417
noop_ECE = 0.6285552978515625
source_CEp99 = 2.430856704711914
noop_CEp99 = 2.4543933868408203
```

结论：seed-salt 修复后，NG70 单点 exploration 可以稳定复现；但 strict 仍被 NLL/ECE 阻断，且仍只有 1/3 train-shuffle。

### 17.4 NG88-NG93：low-tail calibration scan

目的：既然高 target 没能降低 strict badness，反向测试更低 unlabeled tail target。

结果：

```text
candidate_rows = 18
aggregate_rows = 6
any_strict_majority_pass = 0
best_source_vs_control = 0.015625
best_source_vs_noop = 0.02734375
best_linec_seed_pass_count = 3
train_shuffle_exploration_pass_count = 1/3 for each NG88-NG93
```

最强行仍是 `train_seed_base=12242400`。低 tail target 没有改善 robust 性，也没有让 strict gate 通过；更低 target 往往让 ECE 差距仍不达 strict。

### 17.5 NG94-NG99：weight-anchor post-calibration

目的：测试训练后公平权重 anchoring 是否能降低 NLL/ECE badness 或提高 train-shuffle 稳定性。

结果：

```text
candidate_rows = 18
aggregate_rows = 6
any_strict_majority_pass = 0
best_source_vs_control = 0.015625
best_source_vs_noop = 0.02734375
best_linec_seed_pass_count = 4
train_shuffle_exploration_pass_count = 1/3 at best
```

最接近 strict 的 NG94 行：

```text
candidate = F25-NG94-ng70SeedWeightAnchor050PostCal100
train_seed_base = 12242400
source_acc = 0.7734375
noop_acc = 0.7578125
best_control_acc = 0.765625
source_vs_noop = 0.015625
source_vs_control = 0.0078125
LineC = 4/5
source_NLL = 1.7464286088943481
noop_NLL = 1.7446753978729248
source_ECE = 0.6479462385177612
noop_ECE = 0.6363601088523865
source_CEp99 = 2.421077013015747
noop_CEp99 = 2.42838978767395
strict_majority_pass = 0
```

解读：weight anchor 把 NG70 从 LineC 3/5 推到 4/5，并让 ECE/CEp99 进入 strict 容忍，但 NLL 仍略高于 NoOp，未过 strict。

### 17.6 NG100-NG105：weight-anchor fine scan

目的：围绕 NG94 的近 strict 点，对 anchor alpha 和 calibration target 做局部修复。

结果：

```text
candidate_rows = 18
aggregate_rows = 6
any_strict_majority_pass = 0
best_source_vs_control = 0.015625
best_source_vs_noop = 0.0234375
best_linec_seed_pass_count = 4
train_shuffle_exploration_pass_count = 1/3 at best
```

关键行：

```text
candidate = F25-NG103-ng70SeedWeightAnchor055PostCal100
train_seed_base = 12242400
source_acc = 0.78515625
noop_acc = 0.76171875
best_control_acc = 0.76953125
source_vs_noop = 0.0234375
source_vs_control = 0.015625
LineC = 4/5
source_NLL = 1.7555862665176392
noop_NLL = 1.7439911365509033
source_ECE = 0.6578199863433838
noop_ECE = 0.6389930248260498
source_CEp99 = 2.3994922637939453
noop_CEp99 = 2.409813404083252
strict_majority_pass = 0
```

结论：fine scan 没有把固定 reference 下的 near-strict 行变成 strict；NLL/ECE 仍略差或其他 train seeds 不闭合。

### 17.7 NG106：bootstrap_mom reference scan

目的：计划中对 LineC majority 但 all/robust fail 的 fallback 是 train-stream reference bootstrap / median-of-means。这里新增 `bootstrap_mom` ref mode 并复跑 NG100-NG105。

总览：

```text
candidate_rows = 18
aggregate_rows = 6
any_strict_majority_pass = 1
any_strict_all_pass = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
combined_bridge_any_train_shuffle_robust_all_pass = 0
best_source_vs_control = 0.015625
best_source_vs_noop = 0.0234375
best_linec_seed_pass_count = 4
train_shuffle_strict_majority_pass_count = 1/3 at best
```

最强 strict 单点：

```text
candidate = F25-NG103-ng70SeedWeightAnchor055PostCal100
train_seed_base = 12242400
ref_mode = bootstrap_mom
source_acc = 0.78515625
noop_acc = 0.76171875
best_control_acc = 0.76953125
source_vs_noop = 0.0234375
source_vs_control = 0.015625
source_NLL = 1.7629508972167969
noop_NLL = 1.7821247577667236
source_ECE = 0.6592267751693726
noop_ECE = 0.6460790038108826
source_CEp99 = 2.3973495960235596
noop_CEp99 = 2.3981130123138428
LineC = 4/5
strict_majority_pass = 1
strict_all_pass = 0
exploration_gate_pass = 1
```

其他 train-shuffle seeds 的失败边界：

```text
train_seed_base = 12240400:
  NG105 source_vs_control = 0.015625, but source_vs_noop = 0.00390625 and LineC = 0/5.

train_seed_base = 12241400:
  best task/control rows have LineC = 0/5 or source_vs_control <= 0.
```

结论：

```text
bootstrap_mom reference 修复了单点 strict badness；
但没有修复 train-shuffle robustness，也没有达到 LineC all-pass。
```

### 17.8 本轮最终科学结论

v12.25 仍未达成目标，但边界进一步推进：

```text
1. NG70/NG82-NG93：post-calibration 能产生稳定复现的单 seed exploration，但 strict NLL/ECE 或 robustness 失败。
2. NG94-NG105：weight anchor 能把 LineC 从 3/5 推到 4/5，并接近 strict，但固定 refs 下仍未过。
3. NG106：bootstrap_mom refs 第一次产生 single-seed strict_majority_pass=1。
4. official S5 仍不成立，因为 strict pass 只有 1/3 train-shuffle，strict_all_pass=0，train_shuffle_robust_majority/all 仍为 0。
```

最精确 blocker 更新为：

```text
precommit-safe train-feature + post-calibration + weight-anchor + bootstrap_mom reference
已经能在单个 train-shuffle seed 上通过 strict majority gate；
但该机制不能跨 train-shuffle seed 稳定同位 task/control margin 与 LineC。
```

因此本轮不能写 S5。若继续下一版本，应更换或扩展 functional update primitive，使 LineC-positive 的 seeds 同时获得 source-vs-NoOp/control margin，而不是继续只在 NG70/NG103 的单 seed 上调 calibration/anchor。

### 17.9 finalizer

本节写入后执行：

```bash
conda run -n kan python experiments/run_v1225_finalize_precommit_functional.py
```

最终 route 与 zip sha 以 `v1225_route_decision.json` 为准。

## 18. 继续推进 NG107-NG130：S4a 打开但 S5 official 仍未达成

### 18.1 是否达成目标

没有达成 S5 official functional success。

最终聚合 route：

```text
route = S4a-PrecommitBridgeExplorationOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 0
hard_compute_budget_exhausted = 0
fallback_rows = 31
fallback_all_executed = 1
required_artifact_missing_count = 0
line_f_exploration_gate_pass = 1
line_f_official_gate_pass = 0
```

解释：本轮不是 no-go final stop，而是从 S4b/S4 single-seed strict 继续推进到 S4a exploration。它仍不是目标 S5，因为 official all-pass / robust all-pass 仍为 0。

### 18.2 本轮代码修改

修改文件：

```text
experiments/run_v1225_composite_functional_bridge.py
experiments/run_v1225_finalize_precommit_functional.py
```

修改内容：

```text
1. composite runner 增加 --ref-seed-base，固定 bootstrap_mom reference seed。
2. composite runner 增加 --train-seed-salt-override，用于公平共享的 train seed salt 反事实扫描。
3. 新增 NG110-NG115 feature-mix primitive：
   centered denoise + low-tail / margin / confidence / entropy train-feature event。
4. 新增 NG118-NG123 post-calibration target scan。
5. 新增 NG124-NG129 weight-anchor alpha scan。
6. finalizer 增加 NG107-NG130 fallback manifest / required artifact / code review packet 覆盖。
7. finalizer 新增 v1225_multisketch_response_variance_decomposition.csv/json。
```

合理性判断：

```text
1. 所有新增 functional direction 仍只使用 train-stream logits/features，不使用 query batch、label、CE、validation/test 或 LineC target。
2. train_seed_salt_override 是 source/noop/control 公平共享的训练顺序反事实，不是按结果或数据集分支。
3. post-calibration 与 weight-anchor 对 source/noop/control 公平执行，不降低任何 official gate。
4. multi-sketch decomposition 只是诊断 artifact，promotion_allowed=0，不改变 route gate。
```

### 18.3 NG107：固定 bootstrap reference seed

结果：

```text
candidate_rows = 18
any_strict_majority_pass = 1
any_strict_all_pass = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
combined_bridge_any_train_shuffle_robust_all_pass = 0
best = F25-NG103-ng70SeedWeightAnchor055PostCal100
best_source_vs_control = 0.015625
best_source_vs_noop = 0.0234375
best_linec_seed_pass_count = 4
train_shuffle_strict_majority_pass_count = 1/3
```

结论：固定 reference seed 没有把 NG106 的 single-seed strict majority 扩成 robust。失败不是 bootstrap reference seed 随 train-shuffle 抖动导致。

### 18.4 NG108 / NG109：seed-salt 与 ensemble width 反事实

NG108 扫描：

```text
train_seed_salt = 32..44
best_salt = 39
best_train_shuffle_strict_majority_pass_count = 1/3
best_train_shuffle_robust_all_pass = 0
```

NG109 扫描：

```text
ensemble_count = 32,64
train_shuffle_strict_majority_pass_count = 1/3
train_shuffle_robust_majority_pass = 0
train_shuffle_robust_all_pass = 0
```

结论：简单训练 seed 相位调整和更宽 bootstrap_mom reference 都不能解决 robustness。

### 18.5 NG110-NG115：feature mix primitive

结果：

```text
candidate_rows = 18
any_strict_majority_pass = 1
any_strict_all_pass = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
combined_bridge_any_train_shuffle_robust_all_pass = 0
all best_linec_seed_pass_count = 4
all train_shuffle_strict_majority_pass_count = 1/3
```

结论：把 centered denoise 与 low-tail/margin/confidence/entropy feature event 混合，没有把 LineC/task/control 同位扩展到其他 train-shuffle seed。

### 18.6 NG116-NG117：训练动力学修复

NG116：

```text
epochs = 12
lr = 0.0015
train_shuffle_exploration_pass_count = 2/3
exploration_train_shuffle_robust_pass = 1
train_shuffle_strict_majority_pass_count = 1/3
train_shuffle_robust_all_pass = 0
```

关键近似点：

```text
candidate = F25-NG103-ng70SeedWeightAnchor055PostCal100
train_seed_base = 12240400
source_acc = 0.77734375
noop_acc = 0.76171875
best_control_acc = 0.76171875
source_vs_noop = 0.015625
source_vs_control = 0.015625
LineC = 3/5
source_NLL = 1.8117599487304688
noop_NLL = 1.803513526916504
source_CEp99 = 2.4505879878997803
noop_CEp99 = 2.4103035926818848
```

NG117：

```text
epochs = 12
lr = 0.0010
exploration_train_shuffle_robust_pass = 0
```

结论：更长训练把 Line F 推到 S4a exploration robust，但 strict 仍被 NLL/LineC all-pass 卡住；降低 lr 反而丢掉 signal。

### 18.7 NG118-NG123 / NG124-NG129：calibration target 与 weight-anchor alpha

NG118-NG123：

```text
any_exploration_pass = 1
best candidates = NG121/NG122/NG123
train_shuffle_exploration_pass_count = 2/3
train_shuffle_strict_majority_pass_count = 1/3
```

NG124-NG129：

```text
any_exploration_pass = 1
best = F25-NG127-ng70SeedWeightAnchor060PostCal125
best_source_vs_control = 0.01953125
best_linec_seed_pass_count = 4
train_shuffle_exploration_pass_count = 2/3
train_shuffle_strict_majority_pass_count = 1/3
train_shuffle_robust_all_pass = 0
```

最强 task 单点：

```text
candidate = F25-NG129-ng70SeedWeightAnchor070PostCal125
train_seed_base = 12242400
source_acc = 0.79296875
noop_acc = 0.765625
best_control_acc = 0.765625
source_vs_noop = 0.02734375
source_vs_control = 0.02734375
LineC = 4/5
strict_majority_pass = 1
strict_all_pass = 0
```

结论：calibration target 与 weight-anchor alpha 可以增强 task/control margin，并让 exploration 达到 2/3 robust；但仍不能产生 LineC all-pass，也没有把 strict majority 扩到 2/3。

### 18.8 NG130：large LineC recheck

设置：

```text
linec_batch = 64
linec_sketch_dim = 24
candidates = NG121, NG127, NG129
```

结果：

```text
candidate_rows = 9
any_exploration_pass = 0
any_strict_majority_pass = 0
best_linec_seed_pass_count = 1
```

结论：更大的 LineC batch/sketch 没有修复 all-pass，反而把 best LineC 从 4/5 降到 1/5。这说明 LineC all-pass 缺口不是小 sketch 噪声造成的简单误判。

### 18.9 Multi-sketch decomposition

新增 artifact：

```text
v1225_multisketch_response_variance_decomposition.csv
v1225_multisketch_response_variance_decomposition_summary.json
```

关键结果：

```text
multisketch_group_rows = 507
multisketch_any_majority_pass = 1
multisketch_any_all_pass = 0
multisketch_max_pass_count = 4
```

典型边界：

```text
NG127 / train_seed_base=12242400:
  linec_pass_count = 4/5
  coupling_delta_min = 0.0007820149496839957
  noise_delta range = [-0.3669150844216347, 0.022642649710178375]
  reservoir_delta range = [-0.14064115285873413, 0.019578218460083008]

NG127 / train_seed_base=12241400:
  linec_pass_count = 0/5
  coupling_delta_min = -0.01199987256000068
```

解读：source 在部分 seeds 能改善 coupling/noise/reservoir，但 12241400 出现稳定 CouplingR2 负 delta，导致 LineC 0/5。这是机制不稳，不是单一 metric seed 噪声。

### 18.10 最终状态

最终 route：

```text
route = S4a-PrecommitBridgeExplorationOpened
minimum_success = Minimum Success D
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 0
required_artifact_missing_count = 0
code_review_packet_entries = 132
code_review_packet_sha256 = d4523320dd6bcc0a12075bc45e9464556d79cba222bccc5b03dda5d17e6b6251
```

本轮科学结论：

```text
1. v12.25 已经从 single-seed strict majority 推进到 S4a exploration robust。
2. 最强方向是 epochs=12 + bootstrap_mom refs + weight-anchor/post-calibration。
3. 但 S5 official 仍不成立，因为没有任何 row 达到 LineC 5/5；robust strict_all 仍为 0。
4. 更重 LineC 复核显示 best LineC 降到 1/5，强化了 all-pass blocker。
5. 当前 blocker 收敛为：precommit-safe task/control margin 可以打开，但 LineC all-pass 和 12241400 train-shuffle geometry 不能同位。
```

不能写成 S5，也不能写 promotion success。

### 18.11 日志写入后的最终复核

写入本复盘和执行日志后，重新执行 finalizer 打包：

```text
conda run -n kan python experiments/run_v1225_finalize_precommit_functional.py
```

最终复核结果：

```text
route = S4a-PrecommitBridgeExplorationOpened
minimum_success = Minimum Success D
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 0
hard_compute_budget_exhausted = 0
fallback_depth = 3
fallback_rows = 31
fallback_all_executed = 1
required_artifact_rows = 70
required_artifact_missing_count = 0
line_f_exploration_gate_pass = 1
line_f_official_gate_pass = 0
multisketch_any_majority_pass = 1
multisketch_any_all_pass = 0
multisketch_max_pass_count = 4
precommit_value_source_rows = 507
policy_aware_p3_rows = 507
exploration_visibility_gate_pass = 1
AUC_source_control = 0.9218314644920149
AUC_tail_safe = 1.0
AUC_linec_majority = 1.0
precision_at_k_s5_proxy = 0.9313725490196079
recall_at_k_s5_proxy = 0.9595959595959596
code_review_packet_entries = 132
code_review_packet_sha256 = d97f91225fcd9edc3a9062937eddde0cb9e14cda921ad30cb86c27d705fc9ff8
```

最终判定没有变化：

```text
v12.25 没有达成 S5；
已经打开 S4a precommit bridge exploration；
不允许 promotion；
不允许写 official success；
也不能写 fail-closed final stop，因为 final_stop_allowed = 0 且 hard_compute_budget_exhausted = 0。
```

这意味着本轮推进产生了有效的新边界，但 v12.25 目标仍未完成。下一步不能继续只做 weight-anchor/calibration target 小网格；应优先针对 `12241400` train-shuffle geometry 的 CouplingR2 负 delta 设计新的 precommit-safe geometry-preserving direction，或替换当前 LineC 响应不稳的 functional primitive。

## 19. NG131-NG137 继续复盘：geometry-preserving policy 与 train-size 复核仍未闭合 S5

### 19.1 为什么继续

上一节 finalizer 显示：

```text
route = S4a-PrecommitBridgeExplorationOpened
official_success_reached = 0
p4_pass = 0
final_stop_allowed = 0
```

因此不能把 v12.25 写成完成。根据 NG127/NG129 的失败边界，当前最明确 blocker 是：

```text
precommit-safe source 已能产生 task/control margin；
但 LineC all-pass 不成立，尤其 12241400 train-shuffle 出现稳定几何失败。
```

本轮继续尝试两个不读 label/CE/query 的修复方向：

```text
1. NG131-NG136：通过 freeze_direct / freeze_quad / quad_only 约束训练自由度，检查是否能保护 LineC geometry。
2. NG137：把 train_size 从 512 提到 1024，检查 12241400 不稳是否来自小训练子集。
```

### 19.2 本轮代码修改

修改文件：

```text
experiments/run_v1225_composite_functional_bridge.py
experiments/run_v1225_finalize_precommit_functional.py
```

修改内容：

```text
1. 新增 NG131-NG136 candidate：
   - F25-NG131-ng70SeedWeightAnchor060PostCal125FreezeDirect
   - F25-NG132-ng70SeedWeightAnchor070PostCal125FreezeDirect
   - F25-NG133-ng70SeedWeightAnchor060PostCal125FreezeQuad
   - F25-NG134-ng70SeedWeightAnchor070PostCal125FreezeQuad
   - F25-NG135-ng70SeedWeightAnchor060PostCal125QuadOnly
   - F25-NG136-ng70SeedWeightAnchor070PostCal125QuadOnly
2. finalizer 新增 R35 审计项，确认 NG131-NG136 已注册。
3. fallback manifest 增加：
   - NG131_NG136_geometry_preserving_policy_scan
   - NG137_train_size1024_stability_recheck
```

合理性判断：

```text
1. NG131-NG136 没有降低 gate，也没有使用 query/label/CE/LineC target 构造方向。
2. freeze_direct / freeze_quad / quad_only 是对 role-policy 的训练自由度约束，用于验证 geometry 破坏是否来自 direct 或 quad 的 post-P3 训练漂移。
3. NG137 只是增加 train-stream 训练样本数，检查稳定性来源；不是改评估阈值。
```

语法检查：

```text
py_compile pass
```

### 19.3 NG131-NG136 结果

运行设置：

```text
epochs = 12
lr = 0.0015
train_size = 512
ref_mode = bootstrap_mom
ref_seed_base = 12242525
train_seed_salt_override = 39
linec_seeds = 5
```

总览：

```text
candidate_rows = 18
aggregate_rows = 6
any_exploration_pass = 0
any_strict_majority_pass = 0
any_strict_all_pass = 0
```

最强行：

```text
candidate = F25-NG131-ng70SeedWeightAnchor060PostCal125FreezeDirect
best_source_vs_control = 0.00390625
best_source_vs_noop = 0.0234375
best_linec_seed_pass_count = 4
train_shuffle_exploration_pass_count = 1/3
train_shuffle_strict_majority_pass_count = 0/3
```

关键细项：

```text
NG131 / train_seed_base=12242400:
source_acc = 0.79296875
noop_acc = 0.76953125
best_control_acc = 0.7890625
source_vs_noop = 0.0234375
source_vs_control = 0.00390625
LineC = 4/5
exploration_gate_pass = 1

NG131 / train_seed_base=12241400:
source_acc = 0.78125
noop_acc = 0.77734375
best_control_acc = 0.7890625
source_vs_control = -0.0078125
LineC = 0/5
```

结论：

```text
freeze_direct 能在一个 train seed 上保留 LineC 4/5，但 matched control margin 不足；
12241400 仍然 LineC 0/5 且输 control；
freeze_quad 与 quad_only 进一步伤 task，未修复 geometry。
```

因此 geometry-preserving role policy 不足以闭合 S5。

### 19.4 NG137 train_size=1024 稳定性复核

运行设置：

```text
train_size = 1024
候选 = NG121, NG127, NG129
epochs = 12
lr = 0.0015
其余 ref/seed/LineC 设置与 NG131-NG136 相同
```

结果：

```text
candidate_rows = 9
aggregate_rows = 3
any_exploration_pass = 0
any_strict_majority_pass = 0
any_strict_all_pass = 0
best = F25-NG127-ng70SeedWeightAnchor060PostCal125
best_source_vs_control = 0.00390625
best_source_vs_noop = 0.03125
best_linec_seed_pass_count = 3
train_shuffle_exploration_pass_count = 1/3
```

解读：

```text
把 train_size 从 512 提到 1024 没有修复 train-shuffle / LineC 稳定性；
best LineC 从此前最高 4/5 降为 3/5；
source-control margin 也降到 exploration 边界 0.00390625。
```

这排除了“只是小训练子集导致 12241400 不稳”的简单解释。

### 19.5 NG137 后 finalizer

聚合结果：

```text
route = S4a-PrecommitBridgeExplorationOpened
minimum_success = Minimum Success D
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 0
hard_compute_budget_exhausted = 0
fallback_rows = 33
fallback_all_executed = 1
required_artifact_rows = 72
required_artifact_missing_count = 0
line_f_exploration_gate_pass = 1
line_f_official_gate_pass = 0
multisketch_any_majority_pass = 1
multisketch_any_all_pass = 0
multisketch_max_pass_count = 4
precommit_value_source_rows = 534
policy_aware_p3_rows = 534
exploration_visibility_gate_pass = 1
```

### 19.6 本轮结论

v12.25 仍未达成 S5：

```text
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
```

新增边界：

```text
1. freeze_direct 可以保留局部 LineC 4/5，但 source-control margin 不够，且 12241400 仍失败。
2. freeze_quad / quad_only 明显损伤 task，不能作为 geometry 修复。
3. train_size=1024 不提高 robust LineC，反而弱化 best LineC 与 control margin。
4. 当前 S4a 仍成立，但 S5 blocker 更集中：不是训练子集太小，也不是简单 direct/quad 冻结能修复，而是当前 centered-denoise / weight-anchor primitive 在 12241400 上产生稳定 geometry mismatch。
```

因此下一步如果继续，不应再沿着 `alpha / post-cal / freeze policy / train_size` 小网格推进；需要换一个真正新的 precommit functional primitive，或者设计能直接约束 train-stream geometry proxy 的 update。当前证据不足以写 S5。

## 20. NG144-NG167 继续推进复盘：低秩与 cross-ref coupling 未闭合 S5

### 20.1 是否达成目标

没有。

NG144-NG167 后，最终仍是：

```text
route = S4a-PrecommitBridgeExplorationOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
line_f_official_gate_pass = 0
```

本轮没有把 S4a exploration 写成 S5，也没有降低 gate。所有结果来自实际 `v1225_composite_functional_bridge_*.csv/json` 和 finalizer 输出。

### 20.2 本轮代码修改

修改文件：

```text
experiments/run_v1225_composite_functional_bridge.py
experiments/run_v1225_finalize_precommit_functional.py
```

新增 precommit-safe actuator：

```text
I39-TrainFeatureLowRankCouplingLift
I40-TrainFeatureEntropyWeightedLowRankCoupling
I41-TrainFeatureTailClippedLowRankCoupling
I42-TrainFeatureCrossRefCouplingLift
I43-TrainFeatureEntropyWeightedCrossRefCoupling
I44-TrainFeatureTailClippedCrossRefCoupling
```

新增候选：

```text
F25-NG144 ... F25-NG167
```

合理性审计：

```text
1. I39-I44 只使用 train-stream ref micro-batch 的 features/logits/probabilities/entropy/tail observable。
2. 不使用 query batch、真实标签、CE/NLL、validation/test 或未来 outcome 构造 direction。
3. NG150-NG155 只改变 role policy 与 post-weight interpolation alpha，不改变 official gate。
4. NG162-NG167 只复核 lower lr / longer epochs 的训练动力学，不改变 promotion 阈值。
5. finalizer 只新增 R37-R39 review 与 fallback manifest 条目，不改变 S5 判定逻辑。
```

### 20.3 NG144-NG149 low-rank coupling

目标：针对 NG138-NG143 暴露的 12241400 CouplingR2 稳定负增量，构造低秩 feature/logit coupling direction。

结果：

```text
candidate_rows = 18
aggregate_rows = 6
any_exploration_pass = 1
any_strict_majority_pass = 1
any_strict_all_pass = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
combined_bridge_any_train_shuffle_robust_all_pass = 0
```

最好聚合：

| candidate | best_source_vs_control | best_source_vs_noop | best LineC | exploration train-shuffle | strict majority |
|---|---:|---:|---:|---:|---:|
| NG144 lowRank | 0.015625 | 0.01953125 | 4/5 | 2/3 | 1/3 |
| NG146 entropy lowRank | 0.015625 | 0.01953125 | 4/5 | 2/3 | 1/3 |
| NG147 tail-clipped lowRank | 0.015625 | 0.01953125 | 4/5 | 2/3 | 1/3 |
| NG148 stable+lowRank mix | 0.015625 | 0.01953125 | 4/5 | 2/3 | 1/3 |

关键失败行：

```text
NG144 / train_seed_base=12241400:
source_acc = 0.7734375
noop_acc = 0.765625
best_control_acc = 0.78125
source_vs_noop = 0.0078125
source_vs_control = -0.0078125
NLL/CEp99/ECE gate = pass
LineC = 0/5
CouplingR2 = 0.27418483523036397 < noop 0.28765623772957816
```

解释：low-rank direction 保留了 12242400 上的 strict majority 近似点，但 12241400 仍然输 matched control 且 LineC 0/5。

### 20.4 NG150-NG155 role/alpha repair

目标：NG149 direct_branch 能把 12241400 的 CouplingR2 提高到 `0.31543071927677224 > noop 0.3094393421142916`，但 task 崩太多；因此尝试更高 post-weight alpha 与 direct_gain/all/quad_direct 角色。

结果：

```text
candidate_rows = 18
aggregate_rows = 6
any_exploration_pass = 0
any_strict_majority_pass = 0
any_strict_all_pass = 0
best = F25-NG154-lowRankQuadDirectAlpha085PostCal125
best_source_vs_control = 0.01171875
best_source_vs_noop = 0.0234375
best_linec_seed_pass_count = 3
```

关键边界：

```text
NG150/NG151 direct_branch alpha 0.85/1.00:
best LineC <= 2/5
source_vs_control <= -0.00390625

NG153 all alpha 0.85:
best LineC = 4/5
best_source_vs_control = -0.01171875

NG154/NG155 quad_direct alpha 0.85:
best_source_vs_control = 0.01171875
best LineC = 3/5
```

结论：提高 alpha 或扩大 trainable role 没有把 direct-branch 的几何收益转成 task/control-safe success。

### 20.5 NG156-NG161 cross-ref coupling

目标：用多个 train-stream ref micro-batch 的特征均值与 centered-logit 均值协方差构造 direction，尝试比全局 low-rank PC 更直接地修复跨 batch coupling。

结果：

```text
candidate_rows = 18
aggregate_rows = 6
any_exploration_pass = 1
any_strict_majority_pass = 1
any_strict_all_pass = 0
best_source_vs_control = 0.015625
best_source_vs_noop = 0.01953125
best_linec_seed_pass_count = 4
train_shuffle_exploration_pass_count = 2/3
train_shuffle_strict_majority_pass_count = 1/3
```

关键失败仍是 12241400：

```text
NG156 / train_seed_base=12241400:
source_vs_noop = 0.0078125
source_vs_control = -0.0078125
NLL/CEp99/ECE gate = pass
LineC = 0/5
```

解释：cross-ref coupling 和 low-rank coupling 的最好边界几乎相同，没有解决 train-shuffle robust all-pass。

### 20.6 NG162-NG167 lower-lr train dynamics repair

目标：在同一 low-rank/cross-ref primitive 上改用 `epochs=16, lr=0.0010`，检查失败是否只是较高 lr 下的 shuffle 不稳定。

结果：

```text
candidate_rows = 18
aggregate_rows = 6
any_exploration_pass = 0
any_strict_majority_pass = 0
any_strict_all_pass = 0
best_source_vs_control = 0.0078125
best_source_vs_noop = 0.01171875
best_linec_seed_pass_count = 3
```

结论：lower lr / longer epochs 没有提高 robustness，反而丢掉了 NG144/NG156 的 S4a exploration signal。

### 20.7 finalizer 聚合

NG144-NG167 纳入 finalizer 后：

```text
route = S4a-PrecommitBridgeExplorationOpened
minimum_success = Minimum Success D
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 0
fallback_rows = 38
fallback_all_executed = 1
required_artifact_rows = 77
required_artifact_missing_count = 0
line_f_exploration_gate_pass = 1
line_f_official_gate_pass = 0
multisketch_max_pass_count = 4
precommit_value_source_rows = 624
policy_aware_p3_rows = 624
code_semantics_review_pass = 1
```

### 20.8 本轮结论

v12.25 仍未达成 S5。

新增边界：

```text
1. low-rank coupling 与 cross-ref coupling 都只能复现 S4a 近似信号，不能过 train-shuffle robust all/majority。
2. 12241400 的 blocker 仍集中在 source-control margin 与 LineC 0/5；NLL/CEp99/ECE 在该 seed 上反而已经过 gate。
3. direct_branch 可以局部提高 CouplingR2，但 task/control 代价过高，alpha/role 修复不能合并这两个优势。
4. lower lr / longer epochs 没有稳定该方向，说明当前 blocker 不是简单训练动力学噪声。
```

当前我不再确定仅靠继续扩展 `low-rank/cross-ref/alpha/lr` 网格可以产生有效机制；下一步需要重新设计更强的 precommit geometry observable，而不是沿当前 family 做小步扫描。


