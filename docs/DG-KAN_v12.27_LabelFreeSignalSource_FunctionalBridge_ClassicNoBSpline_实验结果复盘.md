# DG-KAN v12.27 LabelFreeSignalSource FunctionalBridge ClassicNoBSpline 实验结果复盘

生成时间：2026-05-26（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 diagnostic/near-pass 写成 promotion。

## 1. 计划理解

v12.27 的目标是重新设计 label-free signal source，而不是继续排列 v12.26.1 的 A-LF token family。核心问题：

```text
是否存在一个不使用标签、不使用 CE 的 label-free signal source，
可以替代 B320-current 的 trainprobe-informed signal frame，
让 FHQ base 同时恢复 task trajectory 与 LineC geometry？
```

硬约束：

```text
1. official / exploration candidate 不允许 trainprobe / signalBroad / signalBlock / trainprobeDirect token。
2. base init 必须 uses_y_for_stats=0，不能使用 label-informed init。
3. functional direction 不允许 label、CE、query batch、validation/test、future outcome 或 LineC hard target。
4. CE/NLL/ECE/CEp99 只作为审计和坏化约束。
5. scout/hardening fail 后必须执行至少一个 signal-source repair depth，并完成 Line T / Line D / finalizer。
```

## 2. 本轮代码修改

### 2.1 label-free signal-source primitives

修改文件：

```text
dgkan/models/fc_purekan_primitives.py
```

新增 label-free init token：

```text
multiviewstablep / multiviewresidualp / multiviewblockp
temporaldriftstablep / temporaldriftresidualp / temporaldriftlowrankp
blocklocalcovp / blocklocaledgep / blocklocalstableaugp
crossrandprojp / crossprojresidualp
```

合理性：这些 frame 只从 train-stream `x`、augmentation/random projection/cotangent response 形成，不读取标签、CE、query、validation/test 或 LineC hard target。

### 2.2 A-S1..A-S5 registry

修改文件：

```text
experiments/run_v1218_b320_label_free_ablation.py
```

新增候选：

```text
A-S1a..A-S1d  MultiView signal frame
A-S2a..A-S2c  Temporal drift signal frame
A-S3a..A-S3d  Block-local / stroke geometry frame
A-S4a..A-S4c  Cross-random-projection self-predictive frame
A-S5a..A-S5d  Required repair depth candidates
```

合理性：全部 candidate_id 使用 `V1227LabelFreeSignalSource::` 前缀，`uses_y_for_stats=0`，不复用 B320-current official id。

### 2.3 D34-D38 Rational memory repair aliases

修改文件：

```text
experiments/run_v1224_classic_hardening.py
```

新增 family aliases：

```text
D34-RationalActivationCheckpointNoReadoutCache
D35-RationalGroupedWorkspaceReuse
D36-RationalReadoutGradRecompute
D37-RationalDenominatorStateFP16Audit
D38-RationalLowMemNoPairLineCRepeat
```

说明：本轮 D34-D38 先注册为现有 no-pair / read-bucket / readscale / logitBias Rational 变体的 v12.27 审计入口；是否真正降低 memory 由实际 artifact 判定，不预设成功。

### 2.4 v12.27 runner / finalizer

新增文件：

```text
experiments/run_v1227_label_free_signal_source.py
experiments/run_v1227_finalize_label_free_signal_source.py
```

用途：

```text
run_v1227_label_free_signal_source.py:
  注册 F27-G/R/C functional candidates，并复用 v12.26 label-free-only runner 强制 y_stats=None。

run_v1227_finalize_label_free_signal_source.py:
  聚合 Line A/F/T/D/R artifacts，写 route、required manifest、fallback manifest、code audit、figures、no-go boundary、next queue 和 code review packet。
```

## 3. 初始待执行实验清单

本节后续只写入实际运行后的 artifact 数字：

```text
Line A scout
Line A hardening top-4
Line A repair depth
Line F shadow / official bridge
Line D Rational memory repair
Finalizer route
```

## 4. Line A scout 结果

执行规模：

```text
candidates = A-S1a..A-S4c
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 512
val/test = 256
epochs = 3
raw rows = 144
summary rows = 16
```

Scout top rows：

| candidate | mean_delta_vs_mlp | worst_delta_vs_mlp | max_AUC_time_ratio_vs_mlp | LineC pass rate |
|---|---:|---:|---:|---:|
| A-S2a-TemporalDriftStableFrame | -0.018229166666666668 | -0.06640625 | 1.537457462054296 | 0.0 |
| A-S4c-CrossProjectionLowQuadDirect | -0.018663194444444444 | -0.05078125 | 1.3113856478388872 | 0.0 |
| A-S3b-BlockLocalEdgeEnergyFrame | -0.020833333333333332 | -0.0625 | 1.1905803761724558 | 0.0 |
| A-S4a-CrossRandomProjectionFrame | -0.022569444444444444 | -0.078125 | 1.676644574253729 | 0.1111111111111111 |
| A-S1c-MultiViewBlockLocalStable | -0.024305555555555556 | -0.0703125 | 1.4685627496129294 | 0.1111111111111111 |

结论：scout 没有 near-anchor。最好的 task rows 仍远低于 `mean >= -0.005` 和 `worst >= -0.020`，且 LineC pass rate 最高只有 `1/9`。

## 5. Line A hardening top-4

选择：

```text
A-S2a-TemporalDriftStableFrame
A-S4c-CrossProjectionLowQuadDirect
A-S4a-CrossRandomProjectionFrame
A-S3b-BlockLocalEdgeEnergyFrame
```

执行规模：

```text
train_size = 1024
val/test = 512
epochs = 8
raw rows = 54
summary rows = 6
```

Hardening 结果：

| candidate | mean_delta_vs_mlp | worst_delta_vs_mlp | max_AUC_time_ratio_vs_mlp | LineC pass rate | all-pass |
|---|---:|---:|---:|---:|---:|
| A-S3b-BlockLocalEdgeEnergyFrame | 0.0032552083333333335 | -0.017578125 | 1.5253576687811268 | 0.1111111111111111 | 0 |
| A-S2a-TemporalDriftStableFrame | -0.0026041666666666665 | -0.041015625 | 1.6641730674926594 | 0.1111111111111111 | 0 |
| A-S4c-CrossProjectionLowQuadDirect | -0.005859375 | -0.03125 | 1.602870297284908 | 0.1111111111111111 | 0 |
| A-S4a-CrossRandomProjectionFrame | -0.015625 | -0.05859375 | 1.7071587675822606 | 0.1111111111111111 | 0 |

结论：A-S3b 是本轮最强 base signal，它达到正 mean delta 和 exploration worst delta，但 `AUC_time=1.525` 且 LineC 只有 `1/9`，因此没有通过 near-anchor。

## 6. Signal-source repair depth

Hardening blocker 是 task 接近但 AUC/LineC 失败，因此按计划执行 repair：

```text
A-S5a-ViewStableLineCResidual
A-S5b-BlockStableReducedDirect
A-S5c-MultiViewBlockLowRank
A-S5d-CrossProjLineCResidual
```

执行规模：

```text
train_size = 1024
val/test = 512
epochs = 8
raw rows = 54
summary rows = 6
```

Repair 结果：

| candidate | mean_delta_vs_mlp | worst_delta_vs_mlp | max_AUC_time_ratio_vs_mlp | LineC pass rate | all-pass |
|---|---:|---:|---:|---:|---:|
| A-S5a-ViewStableLineCResidual | -0.019748263888888888 | -0.064453125 | 1.5410900159500351 | 0.3333333333333333 | 0 |
| A-S5d-CrossProjLineCResidual | -0.021267361111111112 | -0.05859375 | 1.6101212216187808 | 0.2222222222222222 | 0 |
| A-S5c-MultiViewBlockLowRank | -0.021701388888888888 | -0.05859375 | 1.5608580653120485 | 0.1111111111111111 | 0 |
| A-S5b-BlockStableReducedDirect | -0.025824652777777776 | -0.06640625 | 1.9110292734040288 | 0.1111111111111111 | 0 |

结论：A-S5a 将 LineC pass rate 提到 `3/9`，但 task、worst delta 和 AUC-time 明显失败。Repair 证明 view-stability + low-rank residual 可以移动 LineC，但没有保住 hardening 中 A-S3b 的 task trajectory。

当前 Line A 状态：

```text
line_a_near_anchor_pass_count = 0
line_a_official_pass_count = 0
```

## 7. Line F shadow functional bridge

因为 Line A 没有 near-anchor，Line F 只能 shadow diagnostic，不能 promotion。

执行设置：

```text
base candidates = A-S3b-BlockLocalEdgeEnergyFrame,A-S5a-ViewStableLineCResidual
dataset = KMNIST
seed = 0
train_seed_bases = 12270400,12271400,12272400
linec_seeds = 5
functional candidates = F27-G/R/C, 9 candidates
candidate_rows = 54
aggregate_rows = 18
```

总体结果：

```text
any_exploration_pass = 0
any_strict_majority_pass = 0
any_strict_all_pass = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
combined_bridge_any_train_shuffle_robust_all_pass = 0
```

最强 task/control rows：

| base | functional | best_source_vs_control | best_source_vs_noop | best LineC | robust |
|---|---|---:|---:|---:|---:|
| A-S3b | F27-C2-LineCEventCovTransportComp | 0.01953125 | 0.0234375 | 1/5 | 0 |
| A-S3b | F27-G1-GeometryGuardThenDirect | 0.01953125 | 0.0234375 | 1/5 | 0 |
| A-S3b | F27-G2-GeometryGuardThenQuadDirect | 0.01953125 | 0.0234375 | 1/5 | 0 |
| A-S3b | F27-G3-GeometryGuardThenLowRank | 0.01953125 | 0.0234375 | 1/5 | 0 |

最强 LineC-ish row：

```text
base = A-S5a-ViewStableLineCResidual
functional = F27-C3-LineCEventViewStableComp
best_source_vs_control = 0.0078125
best_source_vs_noop = 0.015625
best_linec_seed_pass_count = 2/5
```

结论：functional shadow 仍是 task/control 与 LineC 不同位。A-S3b 能给 task/control margin，但 LineC 只有 `1/5`；A-S5a 能稍保 LineC，但达不到 majority 且 margin 较弱。

## 8. Line D Rational D34-D38

执行规模：

```text
families = D34,D35,D36,D37,D38
datasets = MNIST,Fashion-MNIST
seeds = 0
rows = 10
summary rows = 5
exploration_pass_rows = 0
```

Family summary：

| family | rows | mean_delta_vs_MLP | mean CouplingR2 | mean NoiseLeak | mean Reservoir |
|---|---:|---:|---:|---:|---:|
| D34-RationalActivationCheckpointNoReadoutCache | 2 | -0.0244140625 | 0.14313496340760118 | 0.049313612282276154 | 0.2145945206284523 |
| D35-RationalGroupedWorkspaceReuse | 2 | -0.0234375 | 0.14177492389843005 | 0.04950017295777798 | 0.21383054554462433 |
| D36-RationalReadoutGradRecompute | 2 | -0.1962890625 | 0.09871992453564221 | 0.1021287627518177 | 0.5379572212696075 |
| D37-RationalDenominatorStateFP16Audit | 2 | -0.0234375 | 0.14187041569903402 | 0.06615076959133148 | 0.2178812026977539 |
| D38-RationalLowMemNoPairLineCRepeat | 2 | -0.0244140625 | 0.14313496340760118 | 0.049313612282276154 | 0.2145945206284523 |

最接近行：

```text
D35 / Fashion-MNIST / seed 0
val_acc = 0.779296875
MLP val_acc = 0.8046875
mean_delta_vs_MLP = -0.025390625
memory_ratio_vs_mlp = 2.1016346983201215
step_ratio_vs_mlp = 1.2103708873507812
linec_pass = 1
```

结论：D34-D38 没有解决 Rational memory blocker；memory ratio 仍约 `2.10`，因此 Line D 不能打开 R3 meaningful progress。

## 9. Finalizer 聚合结果

最终 route：

```text
route = R1-LabelFreeSignalSourceMissing
minimum_success = Minimum Success E
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 5
fallback_rows = 8
fallback_all_executed = 1
required_artifact_rows = 33
required_artifact_missing_count = 0
```

审计结果：

```text
contract_violation_count = 0
code_semantics_review_pass = 1
line_a_near_anchor_pass_count = 0
line_f_exploration_gate_pass = 0
precommit_value_source_rows = 54
classic_rows = 10
rational_memory_blocked = 1
code_review_packet_entries = 56
code_review_packet_sha256 = 以最终 v1227_route_decision.json 为准
```

## 10. 用户继续要求后的 Depth 6/7：pseudo-partition / affinity-anchor label-free signal source

### 11.1 为什么继续

上一轮最终状态仍是：

```text
route = R1-LabelFreeSignalSourceMissing
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
line_a_near_anchor_pass_count = 0
line_f_exploration_gate_pass = 0
```

用户要求未达成目标则继续。根据前一轮 no-go 边界，继续扩已有 A-S1..A-S5 小网格价值较低，因此本轮尝试新的 label-free signal-source primitive：用 unlabeled train-stream feature geometry 形成 pseudo-partition / affinity-anchor / rank-consensus frame，目标是替代缺失的 trainprobe-informed signal frame。

### 11.2 本轮代码修改

修改文件：

```text
dgkan/models/fc_purekan_primitives.py
experiments/run_v1218_b320_label_free_ablation.py
experiments/run_v1227_finalize_label_free_signal_source.py
```

新增 primitive token：

```text
pseudopartitionp
affinityanchorp
rankconsensusp
pseudoviewmixp
pseudoblockguardp
```

用途：

```text
1. pseudopartitionp：只用 unlabeled z_stats/x 几何做 pseudo cluster partition。
2. affinityanchorp：用 train-stream landmarks 构造 affinity anchor frame。
3. rankconsensusp：用随机投影 rank consensus 构造稳定方向。
4. pseudoviewmixp / pseudoblockguardp：作为 Depth 7 geometry repair，尝试降低 pseudo frame 的过度扰动。
```

新增候选：

```text
A-S6a-PseudoPartitionSignalFrame
A-S6b-PseudoPartitionLowQuadDirect
A-S6c-AffinityAnchorSignalFrame
A-S6d-RankConsensusSignalFrame
A-S6e-AffinityPseudoResidual
A-S7a-PseudoViewMixReducedDirect
A-S7b-PseudoBlockGuardResidual
A-S7c-AffinityPseudoBlockReducedDirect
A-S7d-AffinityRankViewMix
```

合理性审计：

```text
1. 所有新增 candidate 均 uses_y_for_stats=0。
2. direction/init 不使用 label、CE、query batch、validation/test 或 LineC hard target。
3. 没有重新启用 B320-current / trainprobe / signalBroad / signalBlock / trainprobeDirect forbidden token。
4. finalizer 已纳入 Depth 6/7 artifacts、fallback depth、required manifest 与 code packet。
```

### 11.3 Depth 6 scout

执行规模：

```text
candidates = A-S6a..A-S6e
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 512
val/test = 256
epochs = 3
rows = 63
summary_rows = 7
```

Scout 关键结果：

| candidate | mean_delta_vs_mlp | worst_delta_vs_mlp | max_AUC_time_ratio_vs_mlp | LineC pass rate |
|---|---:|---:|---:|---:|
| A-S6a-PseudoPartitionSignalFrame | -0.011284722222222222 | -0.03125 | 1.3248651661424997 | 0.1111111111111111 |
| A-S6c-AffinityAnchorSignalFrame | -0.0234375 | -0.0859375 | 1.5469818446336023 | 0.1111111111111111 |
| A-S6d-RankConsensusSignalFrame | -0.024305555555555556 | -0.09375 | 3.121672605752698 | 0.0 |
| A-S6b-PseudoPartitionLowQuadDirect | -0.029513888888888888 | -0.1015625 | 1.6437443459528869 | 0.1111111111111111 |
| A-S6e-AffinityPseudoResidual | -0.07942708333333333 | -0.16015625 | 3.283128236038854 | 0.0 |

结论：pseudo-partition scout 没有达到 near-anchor。A-S6a task 最接近，但 mean/worst/AUC/LineC 均未闭合。

### 11.4 Depth 6 hardening

执行规模：

```text
candidates = A-S6a,A-S6c,A-S6b
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 1024
val/test = 512
epochs = 8
rows = 45
summary_rows = 5
```

Hardening 结果：

| candidate | mean_delta_vs_mlp | worst_delta_vs_mlp | max_AUC_time_ratio_vs_mlp | LineC pass rate |
|---|---:|---:|---:|---:|
| A-S6b-PseudoPartitionLowQuadDirect | 0.006727430555555556 | -0.029296875 | 1.284561338501438 | 0.1111111111111111 |
| A-S6c-AffinityAnchorSignalFrame | 0.0013020833333333333 | -0.021484375 | 1.303095196829279 | 0.1111111111111111 |
| A-S6a-PseudoPartitionSignalFrame | -0.0008680555555555555 | -0.0390625 | 1.3216076406387973 | 0.1111111111111111 |

结论：A-S6b/A-S6c 在 hardening 后能恢复正 mean task delta，但 worst delta、AUC-time ratio 和 LineC pass rate 仍不满足 near-anchor。因此 Depth 6 没有打开 S1/S4a/S5。

### 11.5 Depth 7 repair scout

执行规模：

```text
candidates = A-S7a..A-S7d
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 512
val/test = 256
epochs = 3
rows = 54
summary_rows = 6
```

Scout 结果：

| candidate | mean_delta_vs_mlp | worst_delta_vs_mlp | max_AUC_time_ratio_vs_mlp | LineC pass rate |
|---|---:|---:|---:|---:|
| A-S7b-PseudoBlockGuardResidual | -0.06336805555555555 | -0.140625 | 2.619102515943048 | 0.0 |
| A-S7a-PseudoViewMixReducedDirect | -0.0802951388888889 | -0.16796875 | 3.5199304499696797 | 0.1111111111111111 |
| A-S7c-AffinityPseudoBlockReducedDirect | -0.09288194444444445 | -0.16796875 | 3.727095821614443 | 0.0 |
| A-S7d-AffinityRankViewMix | -0.09331597222222222 | -0.15234375 | 3.172305573612324 | 0.0 |

结论：Depth 7 guard/view-mix repair 明显恶化 task 和效率，没有进入 hardening。该结果说明 pseudo-partition geometry guard 不是缺失修复。

### 11.6 A-S6 functional shadow recheck

执行设置：

```text
base candidates = A-S6b-PseudoPartitionLowQuadDirect,A-S6c-AffinityAnchorSignalFrame
dataset = KMNIST
seed = 0
train_seed_bases = 12270400,12271400,12272400
linec_seeds = 5
rows = 54
aggregate_rows = 18
```

总结果：

```text
any_exploration_pass = 0
any_strict_majority_pass = 0
any_strict_all_pass = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
combined_bridge_any_train_shuffle_robust_all_pass = 0
```

最接近行：

```text
base = A-S6b-PseudoPartitionLowQuadDirect
candidate = F27-C3-LineCEventViewStableComp
best_source_vs_control = +0.00390625
best_source_vs_noop = +0.015625
best_linec_seed_pass_count = 1/5
```

另一类边界：

```text
base = A-S6b-PseudoPartitionLowQuadDirect
candidate = F27-R1-ReservoirSafeEventTransport
best_linec_seed_pass_count = 2/5
best_source_vs_control = 0.0
```

结论：A-S6 functional shadow 没有产生 S4a support row。task/control margin 与 LineC majority 仍不能同位。

### 11.7 Depth 6/7 finalizer 聚合

最终聚合结果：

```text
route = R1-LabelFreeSignalSourceMissing
minimum_success = Minimum Success E
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 7
fallback_all_executed = 1
required_artifact_missing_count = 0
line_a_near_anchor_pass_count = 0
line_f_exploration_gate_pass = 0
functional_candidate_rows = 108
functional_aggregate_rows = 36
scout_rows = 261
hardening_rows = 153
linec_multisketch_rows = 414
code_review_packet_entries = 70
code_review_packet_sha256 = 以最终 v1227_route_decision.json 为准
```

## 13. 用户再次追问后的 stop-contract 复核

用户再次询问 v12.27 是否达成目标，若未达成则继续。只读复核最终 route 后，结论仍是：没有达成 S5，也没有打开 S4a/S1；但当前已经满足计划允许的 hard-stop no-go 条件。

复核依据：

```text
route = R1-LabelFreeSignalSourceMissing
minimum_success = Minimum Success E
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 7
fallback_all_executed = 1
required_artifact_missing_count = 0
line_a_near_anchor_pass_count = 0
line_f_exploration_gate_pass = 0
functional_candidate_rows = 108
functional_aggregate_rows = 36
precommit_value_source_rows = 108
code_review_packet_entries = 70
```

这次不继续新实验的原因不是 early stop，而是：

```text
1. v12.27 计划要求的 Depth 1-5 已执行，并且用户继续要求后又推进到 Depth 7。
2. Depth 6 测试了 pseudo-partition / affinity-anchor / rank-consensus 新 signal-source primitive。
3. Depth 7 测试了 pseudo-viewmix / blockguard repair。
4. 新增 family 仍没有 near-anchor，也没有 S4a support row。
5. 我已经不确定继续在当前 pseudo-partition / affinity-anchor / rank-consensus / viewmix / blockguard family 上扩组合能产生有效机制。
```

因此最终科学判断不变：

```text
v12.27 没有达成 S5；
没有打开 label-free S4a/S1；
合法 route 仍是 R1-LabelFreeSignalSourceMissing；
不允许 promotion；
允许 final stop。
```

若进入下一轮，应先提出新的 label-free signal source 设计，而不是继续排列当前 projection/pseudo-partition/affinity token。

本节写入后重新执行 finalizer 打包，最终 zip sha 以 `v1227_route_decision.json` 为准。

审计修复：

```text
finalizer 在日志写入后重跑时，曾把已合并的 v1227_functional_bridge_shadow_or_official.csv 与 Depth 6 原始 functional CSV 再次合并，导致 functional_candidate_rows/aggregate_rows 从真实的 108/36 膨胀为 162/54。
已在 experiments/run_v1227_finalize_label_free_signal_source.py 新增 unique_rows()，对 finalizer 输入做幂等去重。
修复后重新运行 finalizer，functional_candidate_rows=108，functional_aggregate_rows=36。
该修复只影响聚合/打包计数，不改变任何实验 row 指标或 gate。
```

### 11.8 本轮结论

v12.27 继续推进后仍未达成 S5，也没有打开 S4a/S1：

```text
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
line_a_near_anchor_pass_count = 0
line_f_exploration_gate_pass = 0
```

新增 no-go 边界：

```text
1. pseudo-partition / affinity-anchor 可以恢复部分 positive mean task delta，但不能同时满足 worst/AUC/LineC near-anchor。
2. rank-consensus / affinity-residual 方向效率和 task 明显变差。
3. pseudo-viewmix / block-guard repair 没有提高 LineC，反而明显伤 task。
4. A-S6b/A-S6c 上的 F27 functional shadow 仍无 S4a support row。
```

因此当前合法状态仍是：

```text
R1-LabelFreeSignalSourceMissing
```

我已经不确定继续在当前 pseudo-partition / affinity-anchor / rank-consensus / viewmix / blockguard family 上扩组合能产生有效机制。若进入下一轮，应重新设计新的 label-free signal source，而不是继续对这些 token 做低价值网格搜索。

本节写入后重新执行 finalizer 打包，最终 route 与 zip sha 以 `v1227_route_decision.json` 为准。

说明：finalizer 初版曾误把 MLP reference summary 计入 A-S gate，并误把 functional variant/multisketch rows 计入 candidate summary；已修复过滤逻辑和 zip 自引用打包问题。修复只改变聚合/打包，不改变实验指标。

## 11. 最终科学结论

v12.27 没有达成 S5，也没有打开 S4a/S1。

已闭合事实：

```text
1. S1/S2/S3/S4 label-free signal-source scout 已执行，没有 near-anchor。
2. Hardening 后 A-S3b 取得正 mean delta，但 AUC-time 与 LineC 失败。
3. Repair depth A-S5a 提高 LineC 到 3/9，但 task/worst/AUC 失败。
4. F27 shadow bridge 在 A-S3b/A-S5a 上没有 S4a support row。
5. D34-D38 Rational memory repair 没有降低 memory ratio，Line D 无 meaningful progress。
6. Depth 6 pseudo-partition / affinity-anchor 只恢复部分 positive mean task delta，near-anchor 仍为 0。
7. Depth 7 pseudo-viewmix / blockguard repair 明显伤 task，没有进入 hardening。
8. A-S6b/A-S6c functional shadow 仍无 S4a support row。
9. Provenance / forbidden-token audit 通过，没有 label/CE/query/forbidden-token promotion violation。
10. Required artifacts 缺失为 0，Depth 1-7 fallback 已执行。
```

最终合法状态：

```text
R1-LabelFreeSignalSourceMissing
```

这不是 early stop，而是计划要求的 Depth 1-5 执行后继续推进到 Depth 7 的 fail-closed no-go。当前 blocker 仍是：新的 label-free signal-source family 能产生局部 task 或局部 LineC，但不能同时恢复 near-anchor 的 task trajectory、AUC-time 和 LineC geometry。

## 12. 日志写入后的最终复核

写入执行日志与复盘日志后，重新执行 finalizer 打包。最终 route 未改变：

```text
route = R1-LabelFreeSignalSourceMissing
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 7
fallback_all_executed = 1
required_artifact_missing_count = 0
line_a_near_anchor_pass_count = 0
line_f_exploration_gate_pass = 0
functional_candidate_rows = 108
functional_aggregate_rows = 36
scout_rows = 261
hardening_rows = 153
classic_rows = 10
code_review_packet_entries = 70
code_review_packet_sha256 = 以最终 v1227_route_decision.json 为准
```
