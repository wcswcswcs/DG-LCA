# DG-KAN v12.30 BasisFirst FunctionalAlsoMLP 实验结果复盘

生成时间：2026-05-26（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 diagnostic/near-pass 写成 promotion。

## 1. 计划理解

v12.30 的目标是把 classic no-BSpline basis 做成主并行线，并同时在 MLP 上验证 loss-agnostic functional update。它不是继续扩大 v12.28 的 label-free FHQ token grid。

硬约束：

```text
1. strict FC-PureKAN / classic no-BSpline active basis。
2. no teacher / distillation / loss modification / sampler / class weight / dataset-name branch。
3. 不使用 label-informed initialization。
4. functional direction 不使用 label、CE vector、permuted-label CE、validation/test、future outcome、query batch 或 LineC hard target。
5. CE/NLL/ECE/CEp99 只能作为审计和坏化约束。
6. blocker 后必须执行计划要求的 fallback。
```

## 2. 当前状态

实验尚在执行中。本文件后续只记录已经生成 artifact 的真实结果。

## 3. 本轮代码修改

### 3.1 Classic basis alias registry

修改文件：

```text
experiments/run_v1224_classic_hardening.py
```

新增 v12.30 classic no-BSpline alias：

```text
D-RAT1..D-RAT6
D-CHE1..D-CHE5
D-WAV1..D-WAV5
D-RBF1..D-RBF6
D-FOU1..D-FOU5
```

用途：把 v12.30 plan 中的 basis-first portfolio 映射到已有 strict no-BSpline PrimitiveSpec ID，供 `run_v1224_classic_hardening.py` 执行 scout / hardening / fallback。

### 3.2 A-DYN low-budget FHQ monitor

修改文件：

```text
experiments/run_v1218_b320_label_free_ablation.py
```

新增候选：

```text
A-DYN1-LearnableSignalFrameWarmup
A-DYN2-EarlySelfPredictiveFrame
A-DYN3-OptimizerObservableFrameRefresh
A-DYN4-OvercompleteRankGuardFrame
A-DYN5-RoleEnergyBalancedFHQMonitor
```

审计结果：

```text
uses_y_for_stats=0
forbidden_token_present=0
```

### 3.3 MLP functional diagnostic

新增文件：

```text
experiments/run_v1230_mlp_functional.py
```

新增 source candidates：

```text
M-F1-ActivationCovTransport
M-F2-LogitCovStabilization
M-F3-RandomCotangentJacobianSketch
M-F4-HiddenSpectralBalance
M-F5-GeometrySafeWeightAnchor
```

新增 controls：

```text
C0-TaskOnlyAdamW
C1-NoOpMatchedOverhead
C2-RandomMatchedNorm
C3-AdamWParallelDirection
C4-SNROnlyAudit
```

合理性：

```text
1. M-F1..M-F5 functional source 只读取 unlabeled activations/logits/weights。
2. C3 使用 label/CE，但只作为 AdamWParallel control，promotion_allowed=0。
3. CE/NLL/ECE/CEp99 和 LineC 只作为 audit/gate，不作为 functional direction source。
```

### 3.4 v12.30 finalizer

新增文件：

```text
experiments/run_v1230_finalize_basis_first.py
```

用途：

```text
生成 code/provenance audit、classic family status、MLP functional gate、unified LineC audit、fallback manifest、route decision、figures、no-go boundary、next hypothesis queue 和 code review packet。
```

语法/provenance 检查：

```text
py_compile pass
A-DYN count = 5, forbidden_token_present=0
classic v12.30 alias count = 27
MLP source count = 5
```

## 4. Smoke checks

已执行三个 smoke：

```text
classic basis smoke rows = 2, summary_rows = 2
MLP functional smoke candidate_rows = 1, control_rows = 6, linec_rows = 2
FHQ A-DYN smoke rows = 4, summary_rows = 4
```

结论：classic alias、MLP functional wrapper、A-DYN monitor 三个入口均能生成 artifact。Smoke 不作为 official evidence。

## 5. Line D classic basis scout

执行规模：

```text
candidates = 27
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 512
val = 256
epochs = 3
rows = 243
summary_rows = 27
exploration_pass_rows = 0
```

Scout 关键结果：

| candidate | mean_delta_vs_MLP | worst_delta_vs_MLP | max_step_ratio_vs_mlp | max_memory_ratio_vs_mlp | mean_LineC_R2 | LineC pass |
|---|---:|---:|---:|---:|---:|---:|
| D-RAT4-GroupRationalSharedDenom | -0.004774305555555556 | -0.0625 | 2.0287684040287197 | 1.1528542187855355 | 0.3186768107505501 | 9/9 |
| D-RAT3-DenStateFP16Checkpoint | -0.005208333333333333 | -0.05859375 | 1.8844231168963972 | 1.1249260859677053 | 0.3057638368168591 | 8/9 |
| D-RAT5-RationalLineCResidualMix | -0.005208333333333333 | -0.05859375 | 2.3147547106051847 | 1.124289288151012 | 0.3154504958063691 | 9/9 |
| D-RAT6-RationalTaskTrajectoryWarmNoExtraMemV2 | -0.005208333333333333 | -0.05859375 | 1.8047733951680358 | 1.1248351148510347 | 0.31492244878026976 | 9/9 |
| D-WAV5-LocalTailCoverageMonitor | -0.05642361111111111 | -0.140625 | 3.143915004292075 | 1.540504889697521 | 0.29067773089892185 | 2/9 |

结论：Scout 中 Rational 明显最接近，有 task near-zero 和 LineC 8-9/9，但 worst delta 与 step ratio 不闭合；其它 family 主要是 task collapse 或效率/内存劣化。

## 6. Line D hardening

执行规模：

```text
candidates = D-RAT4,D-RAT6,D-RAT3,D-RAT5,D-RAT1,D-WAV5,D-FOU5,D-CHE5,D-RBF6
train_size = 1024
val = 512
epochs = 8
rows = 81
summary_rows = 9
exploration_pass_rows = 0
```

Hardening 结果：

| candidate | mean_delta_vs_MLP | worst_delta_vs_MLP | max_step_ratio_vs_mlp | max_memory_ratio_vs_mlp | mean_LineC_R2 | LineC pass |
|---|---:|---:|---:|---:|---:|---:|
| D-WAV5-LocalTailCoverageMonitor | -0.03211805555555555 | -0.064453125 | 2.186109699055222 | 2.101593625498008 | 0.16343453772885783 | 5/9 |
| D-RAT6-RationalTaskTrajectoryWarmNoExtraMemV2 | -0.046875 | -0.095703125 | 1.2434165694326693 | 2.1016346983201215 | 0.13847373825165799 | 3/9 |
| D-RAT3-DenStateFP16Checkpoint | -0.046875 | -0.095703125 | 1.250204270885661 | 2.1016346983201215 | 0.13866132674462836 | 3/9 |
| D-RAT1-GroupWorkspaceRecomputeV3 | -0.046875 | -0.095703125 | 1.2434453100019722 | 2.1016346983201215 | 0.13893464978793363 | 3/9 |
| D-RAT5-RationalLineCResidualMix | -0.04709201388888889 | -0.09765625 | 1.4872401706537777 | 2.1016346983201215 | 0.1385662079861975 | 3/9 |

结论：hardening 后 Rational scout 的 LineC 优势退化到 3/9，并且 memory ratio 约 2.10；WAV5 task 相对最好但 step/memory 和 worst 仍失败。没有 FamilyNearPass。

## 7. Family-specific fallback

执行原因：

```text
hardening 没有 FamilyNearPass；
Rational 需要 memory/task/LineC targeted repair；
Chebyshev/Wavelet/RBF/Fourier 需要 task trajectory 或 compact capacity repair。
```

执行规模：

```text
candidates = D-RAT2,D-CHE3,D-CHE4,D-WAV4,D-RBF5,D-FOU3,D-FOU4
rows = 63
summary_rows = 7
exploration_pass_rows = 0
```

Fallback 结果：

| candidate | mean_delta_vs_MLP | worst_delta_vs_MLP | max_step_ratio_vs_mlp | max_memory_ratio_vs_mlp | mean_LineC_R2 | LineC pass |
|---|---:|---:|---:|---:|---:|---:|
| D-FOU4-LateEnableK4FromK2 | -0.13346354166666666 | -0.185546875 | 2.110223122961363 | 2.101593625498008 | 0.15257988475456274 | 3/9 |
| D-FOU3-LearnedAmplitudeResidual | -0.13368055555555555 | -0.18359375 | 2.0973019012258183 | 2.101593625498008 | 0.1649978830798082 | 0/9 |
| D-CHE4-K3K4-lowDegreeResidual | -0.1703559027777778 | -0.267578125 | 1.7355087150186304 | 2.101593625498008 | 0.14275564624274892 | 1/9 |
| D-WAV4-ScaleDiversitySmallResidual | -0.1911892361111111 | -0.2734375 | 2.157885394496707 | 2.101593625498008 | 0.12370946746371193 | 1/9 |
| D-RAT2-ReadoutGradChunkedV2 | -0.1916232638888889 | -0.24609375 | 1.4077076578498824 | 2.1016346983201215 | 0.12087286435599855 | 2/9 |

结论：fallback 没有修复 blocker。D-FOU3/D-FOU4 比 scout 的 Fourier task collapse 有改善，但仍远离 near-pass；D-RAT2 chunking 没有解决 hardening memory/task/LineC 同位问题。

## 8. Line M MLP functional diagnostic

### 8.1 聚合 bug 与修复

初次三窗口运行后发现：

```text
LineC_pass_count 出现 15/5，不可能成立。
```

原因：

```text
experiments/run_v1230_mlp_functional.py 在构造 candidate row 时筛选 linec_rows 漏掉 dataset 条件。
```

修复：

```text
1. source_linec / noop_linec 过滤条件增加 dataset。
2. 所有 rows 增加 window_epochs，便于 windows=3,5,10 合并审计。
3. 旧 w3/w5/w10 artifact 不进入最终 merged artifact。
```

修复后重新执行 fixed windows。

### 8.2 Fixed windows 总结果

执行规模：

```text
windows = 3,5,10
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_shuffle_seeds = 3
source candidates = M-F1..M-F5
controls = C0,C1,C2,C3,C4
candidate_rows = 405
control_rows = 2430
linec_rows = 4050
gate_rows = 15
max_linec_pass_count = 5/5
any_exploration_pass = 0
any_official_pass = 0
```

Aggregate by source/window：

| candidate | window | mean_vs_noop | mean_vs_control | mean_CouplingR2_delta | max_LineC | pass |
|---|---:|---:|---:|---:|---:|---:|
| M-F1-ActivationCovTransport | 3 | -0.00028935185185185184 | -0.001591435185185185 | -0.00012327161056974054 | 5/5 | 0 |
| M-F1-ActivationCovTransport | 5 | -0.00014467592592592592 | -0.0014467592592592592 | -0.00020124471528600984 | 5/5 | 0 |
| M-F1-ActivationCovTransport | 10 | 0.00014467592592592592 | -0.001591435185185185 | -0.00007383800094068184 | 5/5 | 0 |
| M-F3-RandomCotangentJacobianSketch | 5 | 0.0007233796296296296 | -0.0005787037037037037 | -0.00015934788263643167 | 5/5 | 0 |
| M-F4-HiddenSpectralBalance | 10 | -0.0005787037037037037 | -0.0023148148148148147 | 0.00009746978871558893 | 5/5 | 0 |
| M-F5-GeometrySafeWeightAnchor | 5 | 0.0 | -0.0013020833333333333 | -0.00019802423310325188 | 5/5 | 0 |

最接近 row：

```text
candidate = M-F3-RandomCotangentJacobianSketch
window = 5
dataset = Fashion-MNIST
seed = 1
train_seed_base = 12302400
source_vs_noop = +0.0078125
source_vs_control = +0.00390625
LineC_pass_count = 5/5
CouplingR2_delta = -0.001733436329803384
CEp99_delta_vs_noop = +0.016774654388427734
exploration_gate_pass = 0
```

解释：局部 task/control row 存在，但 average 没有 matched-control positive；最接近 row 的 CouplingR2_delta 为负，且 gate 没过。因此不能写 MLP functional generic positive，也不能触发 P4 transfer。

## 9. Line A FHQ dynamic monitor

执行规模：

```text
candidates = A-DYN1..A-DYN5
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 512
epochs = 3
rows = 63
summary_rows = 7
```

相对 MLP-same-param 关键结果：

| candidate | mean_delta_vs_mlp | worst_delta_vs_mlp |
|---|---:|---:|
| A-DYN1-LearnableSignalFrameWarmup | -0.012152777777777778 | -0.06640625 |
| A-DYN4-OvercompleteRankGuardFrame | -0.020833333333333332 | -0.08984375 |
| A-DYN2-EarlySelfPredictiveFrame | -0.06684027777777778 | -0.19921875 |
| A-DYN3-OptimizerObservableFrameRefresh | -0.06770833333333333 | -0.15234375 |
| A-DYN5-RoleEnergyBalancedFHQMonitor | -0.2404513888888889 | -0.359375 |

结论：A-DYN 没有意外恢复 FHQ label-free near-anchor；本线仍只是 monitor，不能 promotion。

## 10. Final route

Finalizer 修复：

```text
1. v1230_fhq_dynamic_monitor_ablation.csv 写出规范副本 v1230_fhq_dynamic_monitor.csv。
2. required manifest 在 manifest/zip 生成后重算，避免递归 artifact 假缺失。
```

最终 route：

```text
route = R1-BasisPortfolioNoProgress
minimum_success = Minimum Success B
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_family_near_pass_count = 0
basis_family_progress_count = 0
mlp_functional_candidate_rows = 405
mlp_functional_control_rows = 2430
mlp_functional_exploration_pass_rows = 0
mlp_functional_official_pass_rows = 0
line_a_near_anchor_pass_count = 0
factorial_transfer_rows = 1
```

Family status：

| family | status | rows | pass_rows | mean_delta_vs_MLP | max_LineC_R2 | max_memory_ratio | min_step_ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| D-RAT | MemoryBlocked | 54 | 0 | -0.07110821759259259 | 0.2103222406984544 | 2.1016346983201215 | 0.9901115955185561 |
| D-CHE | MemoryBlocked | 27 | 0 | -0.2160011574074074 | 0.2245122690314364 | 2.101593625498008 | 1.4152745979669568 |
| D-WAV | MemoryBlocked | 18 | 0 | -0.11165364583333333 | 0.23690236055431757 | 2.101593625498008 | 1.9237521333300598 |
| D-RBF | MemoryBlocked | 18 | 0 | -0.23350694444444445 | 0.22841371161637014 | 2.101593625498008 | 1.6484955628021116 |
| D-FOU | MemoryBlocked | 27 | 0 | -0.14554398148148148 | 0.2191091279533659 | 2.101593625498008 | 1.8599268229238461 |

## 11. Required artifacts

主要产物：

```text
results/v12_30_basis_first_functional_also_mlp/official_v1230/v1230_route_decision.json
results/v12_30_basis_first_functional_also_mlp/official_v1230/v1230_required_artifact_manifest.csv
results/v12_30_basis_first_functional_also_mlp/official_v1230/v1230_fallback_execution_manifest.csv
results/v12_30_basis_first_functional_also_mlp/official_v1230/v1230_classic_family_status.csv
results/v12_30_basis_first_functional_also_mlp/official_v1230/v1230_mlp_functional_candidates.csv
results/v12_30_basis_first_functional_also_mlp/official_v1230/v1230_mlp_functional_controls.csv
results/v12_30_basis_first_functional_also_mlp/official_v1230/v1230_mlp_functional_gate.csv
results/v12_30_basis_first_functional_also_mlp/official_v1230/v1230_code_review_packet.zip
```

当前 packet：

```text
code_review_packet_entries = 92
code_review_packet_sha256 = 以最终 v1230_route_decision.json 为准
```

## 12. 最终科学结论

v12.30 没有达成 S5，也没有产生 MLP functional generic positive 或 classic FamilyNearPass。

已闭合事实：

```text
1. Classic basis scout 覆盖 27 个 D-RAT/CHE/WAV/RBF/FOU candidates。
2. Rational scout 有 task near-zero 与 LineC 8-9/9，但 hardening 后退化，未形成 FamilyNearPass。
3. Hardening 与 fallback 均显示 max memory ratio 约 2.10，并且 task/worst/LineC 没有同位。
4. MLP functional M-F1..M-F5 在 windows=3,5,10、三数据集、三 seed、三 train-shuffle seed 下没有 exploration/official pass。
5. MLP functional 初次聚合 bug 已修复并重跑 fixed artifact；最终 max_LineC_pass_count = 5/5，不存在不可能计数。
6. A-DYN low-budget FHQ monitor 没有恢复 label-free near-anchor。
7. 因没有 FamilyNearPass 或 MLP functional positive，P4 transfer 未触发，只写 not_triggered matrix。
8. Provenance audit 通过，required artifacts 缺失为 0。
```

最终合法状态：

```text
R1-BasisPortfolioNoProgress
```

下一步不应继续在同一 D-RAT/CHE/WAV/RBF/FOU alias 或 M-F1..M-F5 value source 上小网格扩展。更合理的方向是先解决 classic runner 的 memory accounting / basis workspace bottleneck，或者重新设计 MLP functional value source，使它在 matched controls 下先出现稳定非零通用信号。

## 13. 用户继续要求后的 memory accounting re-audit

用户再次要求确认 v12.30 是否达成目标，若未达成则继续。读取 route 后仍为：

```text
route = R1-BasisPortfolioNoProgress
basis_family_near_pass_count = 0
mlp_functional_exploration_pass_rows = 0
promotion_allowed = 0
```

因此本轮没有继续扩展 D-RAT/CHE/WAV/RBF/FOU alias，而是按上一节提出的 memory accounting / basis workspace blocker 做修复审计。

### 13.1 本轮代码修改

修改文件：

```text
experiments/run_v1224_classic_hardening.py
```

修改内容：

```text
1. 在 MLP reference 和 family run 各自 reset_peak 前记录 memory_allocated baseline。
2. 新增 mlp_incremental_peak_memory_bytes / family_incremental_peak_memory_bytes。
3. memory_ratio_vs_mlp 使用 incremental peak ratio。
4. 原始 peak ratio 保留为 memory_ratio_raw_vs_mlp 方便审计。
5. 不降低 FamilyNearPass gate，不改 task/LineC/step threshold。
```

合理性：上一轮 `memory_ratio_vs_mlp ~= 2.10` 可能混入数据张量和 baseline allocation；incremental peak 可以更直接衡量 family 训练本身相对 MLP 的额外峰值。该修复只改变审计口径，不放宽 promotion。

### 13.2 重跑结果

Hardening memory-fix：

```text
rows = 81
summary_rows = 9
exploration_pass_rows = 0
```

关键 hardening 行：

| candidate | mean_delta_vs_MLP | worst_delta_vs_MLP | max_step_time_ratio_vs_mlp | max_memory_ratio_vs_mlp | raw_memory_ratio_vs_mlp | LineC pass |
|---|---:|---:|---:|---:|---:|---:|
| D-WAV5-LocalTailCoverageMonitor | -0.03211805555555555 | -0.064453125 | 3.3662091271548236 | 9.9670678702775 | 2.101593625498008 | 5/9 |
| D-RAT6-RationalTaskTrajectoryWarmNoExtraMemV2 | -0.046875 | -0.095703125 | 1.5808496229085398 | 9.967402206619859 | 2.1016346983201215 | 3/9 |
| D-RAT3-DenStateFP16Checkpoint | -0.046875 | -0.095703125 | 1.6292729583069736 | 9.967402206619859 | 2.1016346983201215 | 3/9 |
| D-RAT1-GroupWorkspaceRecomputeV3 | -0.046875 | -0.095703125 | 1.5750713411922186 | 9.967402206619859 | 2.1016346983201215 | 3/9 |
| D-RAT5-RationalLineCResidualMix | -0.04709201388888889 | -0.09765625 | 2.0531122127486463 | 9.967402206619859 | 2.1016346983201215 | 3/9 |

Fallback memory-fix：

```text
rows = 63
summary_rows = 7
exploration_pass_rows = 0
```

关键 fallback 行：

| candidate | mean_delta_vs_MLP | worst_delta_vs_MLP | max_step_time_ratio_vs_mlp | max_memory_ratio_vs_mlp | raw_memory_ratio_vs_mlp | LineC pass |
|---|---:|---:|---:|---:|---:|---:|
| D-FOU4-LateEnableK4FromK2 | -0.13346354166666666 | -0.185546875 | 2.2407666708594562 | 9.9670678702775 | 2.101593625498008 | 3/9 |
| D-FOU3-LearnedAmplitudeResidual | -0.13368055555555555 | -0.18359375 | 2.229848800667519 | 9.9670678702775 | 2.101593625498008 | 0/9 |
| D-CHE4-K3K4-lowDegreeResidual | -0.1703559027777778 | -0.267578125 | 1.8404135135686868 | 9.9670678702775 | 2.101593625498008 | 1/9 |
| D-WAV4-ScaleDiversitySmallResidual | -0.1911892361111111 | -0.2734375 | 2.2956933668861277 | 9.9670678702775 | 2.101593625498008 | 1/9 |
| D-RAT2-ReadoutGradChunkedV2 | -0.1916232638888889 | -0.24609375 | 1.3990575066714837 | 9.967402206619859 | 2.1016346983201215 | 2/9 |

### 13.3 解释

memory accounting 修复后，classic family 的 memory blocker 没有消失，反而更明确：raw peak ratio 约 `2.10`，incremental peak ratio 约 `9.97`。这说明 v12.30 classic basis family 的问题不是旧统计口径误伤，而是相对 MLP 的 basis/workspace 额外峰值确实过高。

同时，task/worst/LineC 仍没有同位：

```text
basis_family_near_pass_count = 0
exploration_pass_rows = 0
```

### 13.4 Finalizer 复核

finalizer 复核结果：

```text
route = R1-BasisPortfolioNoProgress
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_family_near_pass_count = 0
mlp_functional_exploration_pass_rows = 0
```

结论不变：

```text
v12.30 没有达成 S5；
没有产生 classic FamilyNearPass；
没有产生 MLP functional generic positive；
不允许 promotion；
合法 route 仍是 R1-BasisPortfolioNoProgress。
```

我现在不确定继续在同一 classic basis alias 或 M-F1..M-F5 functional value source 上扩组合能形成有效机制。若继续推进，需要新的实质机制：要么做真正的 basis kernel/workspace memory reduction，同时保持 task/LineC；要么重新设计能在 matched controls 下稳定为正的 MLP functional value source。继续排列当前 alias/token 会变成低价值网格搜索。

## 14. 用户要求后的 core-to-dgkan refactor

用户指出核心代码不应写在 runner 里，应沉淀到 `dgkan`。本轮执行的是架构审计修复，不把 smoke 写成性能进展，不改变既有 route / gate。

实际修改：

```text
1. dgkan/diagnostics/classic_basis.py
   - CLASSIC_FAMILY_SPECS
   - parse_csv / parse_ints / fnum
   - incremental_peak_bytes / memory_ratio
   - train_mlp_reference

2. dgkan/functional/mlp_functional.py
   - SOURCE_CANDIDATES / CONTROL_IDS
   - hidden_forward / functional_objective
   - grad_delta_from_objective / scale_delta / random_delta_like / apply_delta / param_norm

3. dgkan/training/eval.py
   - classification_basic

4. experiments/run_v1224_classic_hardening.py
   - 改为从 dgkan.diagnostics.classic_basis 导入核心逻辑。

5. experiments/run_v1230_mlp_functional.py
   - 改为从 dgkan.functional.mlp_functional 与 dgkan.training.eval 导入核心逻辑。

6. experiments/run_v1230_finalize_basis_first.py
   - code review manifest、implementation readback、code packet 纳入新的 dgkan 核心模块。
```

验证结果：

```text
py_compile pass
classic_specs = 64
runner_specs_same = True
mlp_sources = 5
controls = 5
runner_sources_same = True
classic_refactor_smoke_rows = 1
classic_refactor_smoke_exploration_pass_rows = 0
mlp_refactor_smoke_candidate_rows = 1
mlp_refactor_smoke_control_rows = 6
mlp_refactor_smoke_linec_rows = 2
mlp_refactor_smoke_any_exploration_pass = 0
```

解释：这次修复让 runner 回到实验编排角色，核心 registry、memory accounting、MLP functional source 和 shared eval metric 进入 `dgkan`。smoke 只证明调用链可用；它不构成 S5 / FamilyNearPass / MLP functional positive。

当前科学结论仍不变：v12.30 没有达成 S5，没有 classic FamilyNearPass，也没有 MLP functional generic positive。此次改变是代码结构与可审计性修复。
Core refactor 后重新运行 finalizer，审计与 packet 已纳入新 dgkan 核心模块：

```text
route = R1-BasisPortfolioNoProgress
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
code_review_packet_entries = 103
code_review_packet_sha256 = 以最终 v1230_route_decision.json 为准
```

## 15. 用户继续要求后的 M-F6 input-jitter functional value source repair

用户再次要求 v12.30 未达成则继续。当前状态仍为：

```text
route = R1-BasisPortfolioNoProgress
basis_family_near_pass_count = 0
mlp_functional_exploration_pass_rows = 0
promotion_allowed = 0
```

上一轮已经修复 memory accounting 并把核心代码沉到 `dgkan`。因此本轮不继续排列 M-F1..M-F5，而是测试一个新的 loss-agnostic MLP value source。

### 15.1 本轮代码修改

修改文件：

```text
dgkan/functional/mlp_functional.py
experiments/run_v1230_mlp_functional.py
```

新增候选：

```text
M-F6-InputJitterConsistency
```

机制：对 train-stream `x` 做确定性小噪声扰动，最小化扰动前后 logits 与 hidden representation 的一致性误差。

合法性审计：

```text
uses_label_for_direction = 0
uses_ce_vector_for_direction = 0
uses_validation_for_commit = 0
uses_query_batch_for_commit = 0
uses_linec_hard_target_for_direction = 0
```

解释：M-F6 只使用当前模型、train-stream input 和 deterministic noise；label 只在 supervised task optimizer / LineC audit 中存在，不进入 functional direction。

### 15.2 三窗口执行结果

执行规模：

```text
windows = 3,5,10
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_seed_bases = 12300400,12301400,12302400
linec_seeds = 5
```

M-F6 结果：

| window | rows | mean_source_vs_noop | mean_source_vs_control | mean_CouplingR2_delta | max_LineC_pass_count | exploration_pass_rows |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 27 | 0.0 | -0.0013020833333333333 | -0.0007315950322315098 | 5 | 0 |
| 5 | 27 | -0.00028935185185185184 | -0.001591435185185185 | 0.0012489990826862628 | 5 | 0 |
| 10 | 27 | -0.00014467592592592592 | -0.001880787037037037 | 0.0013354520241659618 | 5 | 0 |

解释：M-F6 的 LineC audit 局部可达到 5/5，但 source-vs-control 为负，CouplingR2 增益非常小，无法满足 exploration gate。它不是 generic positive，也不能触发 KAN transfer。

合并后 MLP functional canonical artifact：

```text
candidate_rows = 486
control_rows = 2916
linec_rows = 4860
gate_rows = 18
mlp_functional_exploration_pass_rows = 0
mlp_functional_official_pass_rows = 0
```

### 15.3 Final route

finalizer 复核：

```text
route = R1-BasisPortfolioNoProgress
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
basis_family_near_pass_count = 0
basis_family_progress_count = 0
mlp_functional_candidate_rows = 486
mlp_functional_control_rows = 2916
mlp_functional_exploration_pass_rows = 0
mlp_functional_official_pass_rows = 0
```

结论：v12.30 仍没有达成 S5，没有 classic FamilyNearPass，也没有 MLP functional generic positive。M-F6 说明输入扰动一致性这种无标签 value source 可以保持 LineC 不撕裂，但不能击败 matched controls；因此不能 promotion。

我现在更加不确定继续在 MLP loss-agnostic value source 上做轻量变体能形成有效机制。下一步若继续，应该进入新的机制设计，而不是继续给 M-F family 加单项正则式 objective。

## 16. 用户再次追问后的 stop-contract 复核

用户再次要求确认 v12.30 是否达成目标，若未达成则继续。读取最终 `v1230_route_decision.json` 后，结论仍未变化：

```text
route = R1-BasisPortfolioNoProgress
minimum_success = Minimum Success B
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_family_near_pass_count = 0
basis_family_progress_count = 0
mlp_functional_candidate_rows = 486
mlp_functional_control_rows = 2916
mlp_functional_exploration_pass_rows = 0
mlp_functional_official_pass_rows = 0
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
```

因此最终判断仍是：

```text
v12.30 没有达成 S5；
没有 classic FamilyNearPass；
没有 MLP functional generic positive；
不允许 promotion；
合法 route 仍是 R1-BasisPortfolioNoProgress；
允许 final stop。
```

这次不是因为 gate fail 后轻易停止，而是已经执行计划 fallback depth 6，完成 classic basis scout/hardening/fallback、MLP functional M-F1..M-F6 matched controls、Line A monitor、core-to-dgkan refactor、memory accounting re-audit，仍没有达成 Minimum Success A/B/C。M-F6 后我已经不确定继续给 M-F family 增加同类单项 objective 是否有有效机制；继续推进应进入新机制设计，而不是继续排列轻量 value-source token。



## 17. 用户再次追问后的 stop-contract 复核 2

用户再次要求确认 v12.30 是否达成目标，若未达成则继续。再次读取最终 `v1230_route_decision.json` 后，结论没有变化：

```text
route = R1-BasisPortfolioNoProgress
minimum_success = Minimum Success B
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_family_near_pass_count = 0
basis_family_progress_count = 0
mlp_functional_candidate_rows = 486
mlp_functional_control_rows = 2916
mlp_functional_exploration_pass_rows = 0
mlp_functional_official_pass_rows = 0
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
code_review_packet_entries = 123
code_review_packet_sha256 = 以最终 v1230_route_decision.json 为准
```

因此最终判断仍是：

```text
v12.30 没有达成 S5；
没有 classic FamilyNearPass；
没有 MLP functional generic positive；
不允许 promotion；
合法 route 仍是 R1-BasisPortfolioNoProgress；
允许 final stop。
```

没有新增实验的原因：v12.30 已完成 classic basis scout/hardening/fallback、MLP functional M-F1..M-F6 matched controls、Line A monitor、memory accounting re-audit、core-to-dgkan refactor 和 finalizer/provenance 审计。当前已经满足 `hard_compute_budget_exhausted=1`、`fallback_all_executed=1`、`final_stop_allowed=1`。我已经不确定继续在同一 classic basis alias / MLP single-objective functional value-source family 上扩局部变体会形成有效机制；继续执行会变成低价值网格搜索。
