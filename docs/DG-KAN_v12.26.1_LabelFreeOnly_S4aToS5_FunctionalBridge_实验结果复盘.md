# DG-KAN v12.26.1 LabelFreeOnly S4aToS5 FunctionalBridge 实验结果复盘

生成时间：2026-05-26（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 diagnostic/near-pass 写成 promotion。

## 1. 计划理解

v12.26.1 的核心问题是：移除 B320-current 的 label-informed initialization 后，项目是否还能保留 label-free base 与 precommit functional bridge 的 S4a/S5 路径。

硬约束：

```text
1. 不运行、不实例化 B320-current labelInit official path。
2. official/exploration candidate 不允许 trainprobe token。
3. official/exploration candidate 必须 y_for_stats=None / uses_y_for_stats=0。
4. functional direction 不允许 label、CE、query batch、validation/test、LineC hard target。
5. blocker 出现后必须执行计划要求的 Depth 1-4 fallback，不能早停。
```

## 2. 本轮代码修改

### 2.1 A-LF0..A-LF7 label-free-only base registry

修改文件：

```text
experiments/run_v1218_b320_label_free_ablation.py
```

新增候选：

```text
A-LF0-BaseNoProbe
A-LF1-OrthoBank
A-LF2-CovFrame
A-LF3-AugStable
A-LF4-ResidualLowRank
A-LF5-RoleEnergyBalance
A-LF6-CouplingAware
A-LF7-EMACovAdapt
```

合理性：

```text
1. candidate_id 不使用 B320_ID 前缀，避免携带 trainprobe token。
2. init_variant 使用 stripped B320 primitive，去除 trainprobe/signalBroad/signalBlock/trainprobeDirect token。
3. uses_y_for_stats=0。
4. 不改变 Line A gate，也不引入 label/CE direction。
```

### 2.2 v12.26.1 label-free-only functional runner

新增文件：

```text
experiments/run_v1226_label_free_only_bridge.py
```

用途：

```text
1. 根据 base_candidate_id 构造 A-LF label-free base。
2. 强制 y_stats=None。
3. 运行 train-stream/precommit functional primitives。
4. 写入 base_is_label_free、uses_label_for_init=0、uses_y_for_stats=0、forbidden_token_present 等审计字段。
```

合理性：

```text
1. 不再调用 v12.25 中硬编码 A1-noYForStats 的 build_context。
2. functional direction 复用 train-stream refs 和 unlabeled logits/features。
3. AdamWParallelDirection 只作为 control audit，promotion_allowed=0。
```

### 2.3 v12.26.1 finalizer

新增文件：

```text
experiments/run_v1226_finalize_label_free_only.py
```

用途：

```text
生成 v1226 required artifacts、code/provenance audit、fallback manifest、route decision、figures、no-go boundary、next hypothesis queue 和 code review packet。
```

语法检查：

```text
py_compile pass
```

## 3. Depth 1 / Line A label-free base

### 3.1 Scout

执行设置：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 512
val/test = 256
epochs = 3
LineC batch = 32
sketch_dim = 8
rows = 90
summary_rows = 10
```

Scout top-3：

```text
A-LF0-BaseNoProbe
A-LF1-OrthoBank
A-LF4-ResidualLowRank
```

关键 scout 结果：

| candidate | mean_delta_vs_mlp | worst_delta_vs_mlp | max_AUC_time_ratio_vs_mlp | LineC pass rate |
|---|---:|---:|---:|---:|
| A-LF0-BaseNoProbe | 0.006510416666666667 | -0.04296875 | 1.1138057945251563 | 0.1111111111111111 |
| A-LF1-OrthoBank | -0.006510416666666667 | -0.0234375 | 1.1006910594531514 | 0.0 |
| A-LF4-ResidualLowRank | -0.025173611111111112 | -0.06640625 | 1.8977774329659207 | 0.0 |

### 3.2 Hardening top-3

执行设置：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 1024
val/test = 512
epochs = 8
LineC batch = 64
sketch_dim = 24
rows = 45
summary_rows = 5
```

关键 hardening 结果：

| candidate | mean_delta_vs_mlp | worst_delta_vs_mlp | max_AUC_time_ratio_vs_mlp | LineC pass rate | all-pass |
|---|---:|---:|---:|---:|---:|
| A-LF0-BaseNoProbe | 0.009114583333333334 | -0.013671875 | 1.3301654487826118 | 0.1111111111111111 | 0 |
| A-LF1-OrthoBank | 0.006076388888888889 | -0.025390625 | 1.3265644008711077 | 0.0 | 0 |
| A-LF4-ResidualLowRank | -0.008246527777777778 | -0.044921875 | 1.230330495816023 | 0.2222222222222222 | 0 |

结论：A-LF0/A-LF1 的 task delta 为正，但 AUC 比例和 LineC 不满足 near-anchor；A-LF4 有更高 LineC rate 但 task、worst delta 和 AUC 失败。

### 3.3 LineC-oriented repair hardening

因为 top task candidates 的 blocker 是 `task 接近但 LineC fail`，按计划 4.7 补跑 residual/role-energy/coupling/EMA repair。

执行设置：

```text
candidates = A-LF4,A-LF5,A-LF6,A-LF7
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 1024
val/test = 512
epochs = 8
rows = 54
summary_rows = 6
```

结果：

| candidate | mean_delta_vs_mlp | worst_delta_vs_mlp | max_AUC_time_ratio_vs_mlp | LineC pass rate | all-pass |
|---|---:|---:|---:|---:|---:|
| A-LF4-ResidualLowRank | -0.008246527777777778 | -0.044921875 | 1.2303306215531995 | 0.2222222222222222 | 0 |
| A-LF5-RoleEnergyBalance | -0.4229600694444444 | -0.544921875 | 67843.29987176774 | 0.0 | 0 |
| A-LF6-CouplingAware | -0.09765625 | -0.130859375 | 3.3643796057293582 | 0.0 | 0 |
| A-LF7-EMACovAdapt | -0.003689236111111111 | -0.025390625 | 1.2921889696288138 | 0.1111111111111111 | 0 |

结论：LineC-oriented repair 没有打开 label-free near-anchor。A-LF7 的 mean delta 接近 near-anchor，但 worst delta、AUC 和 LineC 仍失败；A-LF5/A-LF6 明显破坏 task/efficiency。

当前 Line A 状态：

```text
official label-free base pass = 0
near-anchor pass = 0
```

按计划仍不能停止，继续执行 Depth 2-4 functional fallback，但这些 functional 只能作为 label-free base fail 后的 shadow diagnostic，不能 promotion。

## 4. Depth 2：S4a transfer diagnostic

执行设置：

```text
base candidates = A-LF0-BaseNoProbe,A-LF1-OrthoBank
dataset = KMNIST
seed = 0
train_seed_bases = 12260400,12261400,12262400
linec_seeds = 5
rows = 36
aggregate_rows = 12
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
base = A-LF0-BaseNoProbe
candidate = F26-D2-controlResidualComposite
train_seed_base = 12262400
source_vs_noop = +0.01953125
source_vs_best_control = +0.0078125
LineC_seed_pass_count = 3/5
CEp99_delta_vs_noop = +0.6398248672485352
exploration_gate_pass = 0
strict_majority_pass = 0
```

解释：Depth 2 已经出现 task/control 与 LineC majority 同位的近似行，但 CEp99 tail 超出 exploration 容忍，且 train-shuffle robust 为 0。因此不能打开 S4a，更不能写 S5。

## 5. Depth 3：risk/policy repair

执行设置：

```text
candidates = F26-D3-riskAwareTailMix,F26-D3-roleSwitchFreezeDirect,F26-D3-sequentialGeometryTask,F26-D3-downscaleQuadPolicy,F26-D3-partialResponseRiskVeto
rows = 30
aggregate_rows = 10
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
base = A-LF0-BaseNoProbe
candidate = F26-D3-downscaleQuadPolicy
train_seed_base = 12262400
source_vs_noop = +0.02734375
source_vs_best_control = +0.00390625
LineC_seed_pass_count = 2/5
CEp99_delta_vs_noop = -0.012908697128295898
exploration_gate_pass = 0
strict_majority_pass = 0
```

解释：Depth 3 的 risk/policy repair 修复了 tail，但 LineC 从 Depth 2 的 3/5 退到 2/5，source-control margin 也只有 `+0.00390625`。这说明 tail-safe policy 没有把 label-free base 的 task/control 与 LineC majority 稳定合并。

## 6. Depth 4：primitive replacement

执行设置：

```text
candidates = F26-D4-roleGainTransport,F26-D4-quadReservoirGuard,F26-D4-lowRankLogitSubspace,F26-D4-unlabeledCovarianceTransport
rows = 24
aggregate_rows = 8
```

总结果：

```text
any_exploration_pass = 0
any_strict_majority_pass = 0
any_strict_all_pass = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
combined_bridge_any_train_shuffle_robust_all_pass = 0
```

关键行：

```text
base = A-LF1-OrthoBank
candidate = F26-D4-lowRankLogitSubspace
train_seed_base = 12260400
source_vs_noop = +0.02734375
source_vs_best_control = -0.00390625
LineC_seed_pass_count = 0/5
```

```text
base = A-LF1-OrthoBank
candidate = F26-D4-unlabeledCovarianceTransport
train_seed_base = 12262400
source_vs_noop = -0.00390625
source_vs_best_control = -0.0078125
LineC_seed_pass_count = 4/5
CEp99_delta_vs_noop = -0.015999794006347656
```

解释：primitive replacement 把信号拆得更清楚：low-rank/logit subspace 可以有 source-vs-NoOp task gain，但输 matched control 且 LineC 失败；unlabeled covariance transport 可以得到 LineC 4/5 和 tail-safe，但 task/control 为负。Depth 4 没有产生 S4a single positive row。

## 7. Finalizer route

最终 route：

```text
route = R1-LabelFreeBaseMissing
minimum_success = Minimum Success E
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 4
fallback_rows = 4
fallback_all_executed = 1
required_artifact_rows = 31
required_artifact_missing_count = 0
```

含义：v12.26.1 没有达成 S5，也没有打开 S4a。它是 label-free base 缺失后的 fail-closed no-go，且 depth-4 fallback 已执行完成。

## 8. Provenance / code audit

finalizer 审计结果：

```text
forbidden_token_official_count = 0
uses_y_for_stats_official_count = 0
uses_label_for_init_count = 0
uses_ce_for_direction_count = 0
uses_query_batch_for_direction_count = 0
code_semantics_review_pass = 1
missing_core_code_refs = 0
```

解释：

```text
1. A-LF candidates 不再使用 B320_ID 作为 candidate_id。
2. base init 去除了 trainprobe/signalBroad/signalBlock/trainprobeDirect token。
3. functional runner 强制 y_stats=None，并写入 label-free provenance fields。
4. 所有 functional direction 均来自 train-stream/unlabeled observable；CE/NLL/LineC 只作为 audit gate。
```

## 9. Value source / visibility

finalizer 输出：

```text
precommit_value_source_rows = 90
support_count = 0
support_datasets = 0
support_seeds = 0
support_train_shuffle_seeds = 0
AUC_joint = ""
AUC_task = 0.20114942528735633
AUC_linec = 0.13647058823529412
AUC_tail_safe = 0.8539325842696629
exploration_visibility_gate_pass = 0
```

解释：本轮没有 S4a/S5 support rows，因此 joint AUC 不定义。单目标 proxy 也不能作为 deployable selector；尤其 task/LineC AUC 均很低。

## 10. Failure atlas / LineC multisketch

finalizer 输出：

```text
failure_atlas_rows = 90
task_positive_rows = 3
linec_majority_rows = 5
s4a_single_positive_rows = 0
linec_multisketch_rows = 1110
linec_multisketch_pass_rows = 79
```

解释：存在少量 task-positive 或 LineC-majority 局部行，但没有任何 row 同时满足 S4a single-positive 条件。LineC multisketch 中 pass row 只占局部，不能构成 all-pass 或 robust evidence。

## 11. Required artifacts / code review packet

产物：

```text
results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/v1226_route_decision.json
results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/v1226_required_artifact_manifest.csv
results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/v1226_fallback_manifest.csv
results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/v1226_code_review_packet.zip
```

当前 code packet：

```text
entries = 57
sha256 = 以最终 v1226_route_decision.json 为准
```

本复盘写入后会重新运行 finalizer，让执行日志和复盘日志进入 code review packet；因此 zip sha 不在这里手写固定值，避免日志内容与 zip hash 递归变化。

## 12. 最终科学结论

v12.26.1 没有达成目标，不能写 S5。

已闭合事实：

```text
1. Label-free-only base scout/hardening/LineC repair 已执行。
2. A-LF0/A-LF1 有正 task delta，但 LineC/efficiency/near-anchor gate 不闭合。
3. A-LF4/A-LF7 可以略提升或保持部分 LineC rate，但 task/worst/AUC 仍不够。
4. Depth 2 出现 task/control + LineC majority 近似点，但 CEp99 tail fail。
5. Depth 3 修复 tail 后丢掉 LineC majority。
6. Depth 4 primitive replacement 将 task-positive 与 LineC-positive 分散到不同候选，没有 S4a row。
7. Provenance audit 通过：没有 label/CE/query/forbidden-token promotion violation。
8. Required artifacts 缺失为 0，fallback depth 4 已执行。
```

最终合法状态：

```text
R1-LabelFreeBaseMissing
```

这比 v12.25 的 S4a 更严格：当完全去掉 B320/trainprobe-informed base 后，当前 label-free-only family 没有保住 S4a。下一步若继续，不应沿 A-LF4/A-LF7 小网格继续微调；需要重新设计 label-free base 的几何构造，让 base 本身先达到 near-anchor 或 LineC all-pass，否则后续 functional bridge 只能产生 shadow diagnostic，不能 promotion。

## 13. 用户继续要求后的 Depth 5：label-free base geometry redesign

### 13.1 为什么继续

上一轮最终 route 是：

```text
R1-LabelFreeBaseMissing
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
```

用户要求未达成目标继续推进。因此本轮不继续盲目扩大 functional grid，而是按 no-go 边界回到 base 侧，测试新的 label-free base geometry family。

### 13.2 本轮代码修改

修改文件：

```text
experiments/run_v1218_b320_label_free_ablation.py
experiments/run_v1226_finalize_label_free_only.py
```

新增 A-LF8..A-LF16：

```text
A-LF8-LowQuadOrtho
A-LF9-LowQuadBoundQ
A-LF10-LowQuadDirectRead
A-LF11-MultiFrameBank
A-LF12-MultiFrameDirect
A-LF13-ConvexFrameMix
A-LF14-SelfCondStopGrad
A-LF15-RoleCondDirect
A-LF16-FrozenBranchGain
```

合理性：

```text
1. 全部 candidate_id 使用 V1226LabelFreeOnly 前缀，不复用 B320-current official id。
2. init_variant 全部从 stripped base 出发，不含 trainprobe/signalBroad/signalBlock/trainprobeDirect。
3. uses_y_for_stats=0，forbidden_token_present=0。
4. finalizer 纳入 depth-5 scout/hardening summary、raw rows 和 fallback depth。
```

### 13.3 Depth 5 scout

执行规模：

```text
candidates = A-LF8..A-LF16
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 512
val/test = 256
epochs = 3
rows = 99
summary_rows = 11
```

Scout 关键结果：

| candidate | mean_delta_vs_mlp | worst_delta_vs_mlp | max_AUC_time_ratio_vs_mlp | LineC pass rate |
|---|---:|---:|---:|---:|
| A-LF13-ConvexFrameMix | -0.016493055555555556 | -0.0546875 | 1.1892753124141364 | 0.0 |
| A-LF15-RoleCondDirect | -0.024739583333333332 | -0.078125 | 1.3878948001641958 | 0.1111111111111111 |
| A-LF10-LowQuadDirectRead | -0.027777777777777776 | -0.06640625 | 1.3542792343300096 | 0.0 |
| A-LF14-SelfCondStopGrad | -0.04123263888888889 | -0.1171875 | 1.514113665721259 | 0.1111111111111111 |

解释：Depth 5 scout 没有出现接近 near-anchor 的 row。A-LF13 task 最好但 LineC 0；A-LF15/A-LF14 有少量 LineC，但 task/worst/AUC 明显失败。

### 13.4 Depth 5 hardening

执行规模：

```text
candidates = A-LF13,A-LF15,A-LF10
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 1024
val/test = 512
epochs = 8
rows = 45
summary_rows = 5
```

Hardening 结果：

| candidate | mean_delta_vs_mlp | worst_delta_vs_mlp | max_AUC_time_ratio_vs_mlp | LineC pass rate | all-pass |
|---|---:|---:|---:|---:|---:|
| A-LF10-LowQuadDirectRead | 0.007161458333333333 | -0.029296875 | 1.218286569691902 | 0.1111111111111111 | 0 |
| A-LF13-ConvexFrameMix | 0.00021701388888888888 | -0.037109375 | 1.3696914861074767 | 0.0 | 0 |
| A-LF15-RoleCondDirect | -0.00390625 | -0.025390625 | 1.4547377770054764 | 0.0 | 0 |

结论：A-LF10 是本轮最有价值的 base 改进，mean task delta 为正，但 worst delta、AUC ratio 和 LineC pass rate 仍不满足 near-anchor；Depth 5 没有打开 S1/S2/S4a/S5。

## 14. Depth 6：A-LF10 task-anchor LineC repair

### 14.1 为什么继续

Depth 5 暴露出新的具体 blocker：

```text
A-LF10 保住 task mean delta；
但 worst delta、AUC 和 LineC 仍失败。
```

因此继续做 A-LF10 task-anchor 的 LineC repair，而不是继续任意扩展 base grid。

### 14.2 本轮代码修改

修改文件：

```text
experiments/run_v1218_b320_label_free_ablation.py
experiments/run_v1226_finalize_label_free_only.py
```

新增 A-LF17..A-LF22：

```text
A-LF17-LowQuadDirectResidual
A-LF18-LowQuadDirectBoundQ
A-LF19-LowQuadDirectCovAdapt
A-LF20-LowQuadRoleCondDirect
A-LF21-LowQuadConvexDirect
A-LF22-LowQuadSelfCondDirect
```

合理性：

```text
1. 这些候选以 A-LF10 的 low-quad + direct-read task anchor 为基础。
2. 分别叠加 residual、boundQ、covadapt、role-conditioned、convex-frame、self-conditioned stop-grad geometry stabilizer。
3. 全部 uses_y_for_stats=0，forbidden_token_present=0。
4. 没有降低 near-anchor、LineC 或 S5 gate。
```

### 14.3 Depth 6 scout

执行规模：

```text
candidates = A-LF17..A-LF22
rows = 72
summary_rows = 8
```

Scout 结果：

| candidate | mean_delta_vs_mlp | worst_delta_vs_mlp | max_AUC_time_ratio_vs_mlp | LineC pass rate |
|---|---:|---:|---:|---:|
| A-LF21-LowQuadConvexDirect | -0.04123263888888889 | -0.10546875 | 1.8804944017329455 | 0.0 |
| A-LF22-LowQuadSelfCondDirect | -0.05164930555555555 | -0.125 | 1.8530847386385656 | 0.1111111111111111 |
| A-LF19-LowQuadDirectCovAdapt | -0.05555555555555555 | -0.10546875 | 2.960918217416046 | 0.1111111111111111 |
| A-LF17-LowQuadDirectResidual | -0.06727430555555555 | -0.11328125 | 2.9295658512815104 | 0.1111111111111111 |

解释：A-LF10 task-anchor repair 在 scout 档整体恶化，提示叠加几何稳定器会过度扰动 base。但为避免 scout 噪声，继续 hardening top-3。

### 14.4 Depth 6 hardening

执行规模：

```text
candidates = A-LF21,A-LF22,A-LF19
rows = 45
summary_rows = 5
```

Hardening 结果：

| candidate | mean_delta_vs_mlp | worst_delta_vs_mlp | max_AUC_time_ratio_vs_mlp | LineC pass rate | all-pass |
|---|---:|---:|---:|---:|---:|
| A-LF21-LowQuadConvexDirect | 0.004774305555555556 | -0.025390625 | 1.2441453676311431 | 0.1111111111111111 | 0 |
| A-LF22-LowQuadSelfCondDirect | -0.007595486111111111 | -0.041015625 | 1.3823386081984899 | 0.1111111111111111 | 0 |
| A-LF19-LowQuadDirectCovAdapt | -0.027560763888888888 | -0.056640625 | 1.7001876530638549 | 0.1111111111111111 | 0 |

结论：Depth 6 没有修复 base。A-LF21 保住正 mean delta，但 worst/AUC/LineC 仍失败；A-LF22/A-LF19 也没有改善 LineC 到 near-anchor。

### 14.5 新 base functional 复验

为确认新 base 是否至少恢复 S4a shadow signal，对 A-LF10/A-LF21 跑了四个最有信息量的 functional event：

```text
F26-D2-controlResidualComposite
F26-D3-downscaleQuadPolicy
F26-D4-lowRankLogitSubspace
F26-D4-unlabeledCovarianceTransport
```

总结果：

```text
candidate_rows = 24
aggregate_rows = 8
any_exploration_pass = 0
any_strict_majority_pass = 0
any_strict_all_pass = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
combined_bridge_any_train_shuffle_robust_all_pass = 0
```

最接近行：

```text
base = A-LF10-LowQuadDirectRead
candidate = F26-D3-downscaleQuadPolicy
train_seed_base = 12260400
source_vs_noop = +0.02734375
source_vs_best_control = +0.01171875
LineC_seed_pass_count = 2/5
CEp99_delta_vs_noop = +0.009539604187011719
NLL_delta_vs_noop = -0.0029888153076171875
ECE_delta_vs_noop = +0.020884394645690918
exploration_gate_pass = 0
strict_majority_pass = 0
```

解释：A-LF10 + downscale policy 得到 task/control 和 tail-safe 的同位，但 LineC 仍只有 2/5，且 ECE 略超 strict 容忍。因此它不是 S4a，更不是 S5。

## 15. Depth 6 后结论

v12.26.1 继续推进后仍未达成目标：

```text
S5 official success = no
S4a label-free functional exploration = no
label-free near-anchor = no
```

新增 no-go 边界：

```text
1. Depth 5 证明 low-quad/direct-read 可以恢复一部分 task mean delta，但不能同时满足 worst/AUC/LineC。
2. Multi-frame/convex/self-conditioned/role-conditioned base 几何并未恢复 B320-current 的 label-informed signal frame。
3. Depth 6 证明在 A-LF10 上叠加 residual/boundQ/covadapt/self-conditioned stabilizer会伤 task 或仍无法提高 LineC。
4. A-LF10 + downscale functional event 能做到 task/control + tail-safe，但 LineC 只有 2/5；所以 functional bridge 仍缺 geometry co-location。
```

最终判断：当前 blocker 不是单个 token 或 scale 可以修复，而是 label-free base 缺少一个能替代 trainprobe signal frame 的有效几何构造。继续沿 low-quad/direct-read/residual/covadapt 小组合搜索边际价值已经很低；下一步需要全新的 label-free signal-frame estimator，而不是继续在现有 A-LF family 上做局部组合。

本节写入后执行 finalizer：

```bash
conda run -n kan python experiments/run_v1226_finalize_label_free_only.py
```

最终 route 与 code packet sha 以 `v1226_route_decision.json` 为准。

## 16. 用户继续要求后的 Depth 7：label-free signal-frame estimator

### 16.1 为什么继续

Depth 6 后仍未达成 S5，也没有打开 label-free S4a：

```text
route = R1-LabelFreeBaseMissing
line_a_near_anchor_pass_count = 0
line_f_exploration_gate_pass = 0
```

上一节结论指出：继续 low-quad/direct-read/residual/covadapt 小组合边际价值很低，需要新的 label-free signal-frame estimator。因此本轮测试只基于初始化阶段 train-stream x 几何的 signal-frame family。

### 16.2 本轮代码修改

修改文件：

```text
experiments/run_v1218_b320_label_free_ablation.py
experiments/run_v1226_finalize_label_free_only.py
```

新增 A-LF23..A-LF30：

```text
A-LF23-PCAOrthoMix
A-LF24-RandomCotangentStable
A-LF25-BlockLocalAugStable
A-LF26-AugTangentFrame
A-LF27-DriftCotangentBank
A-LF28-LowQuadPCAOrthoDirect
A-LF29-LowQuadAugTangentDirect
A-LF30-LowQuadCotangentDirect
```

合理性审计：

```text
1. 全部 uses_y_for_stats=0，forbidden_token_present=0。
2. 不使用 optframep，因为它从 supervised optimizer update 形成 projector adaptation，存在 CE-derived direction 风险。
3. A-LF23..A-LF30 只使用 PCA/augmentation/cotangent/block-local 等无标签输入几何初始化 frame。
4. finalizer 纳入 depth-7 scout/hardening rows、fallback depth 和 code packet。
```

### 16.3 Depth 7 scout

执行规模：

```text
candidates = A-LF23..A-LF30
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 512
val/test = 256
epochs = 3
rows = 90
summary_rows = 10
```

Scout 关键结果：

| candidate | mean_delta_vs_mlp | worst_delta_vs_mlp | max_AUC_time_ratio_vs_mlp | LineC pass rate |
|---|---:|---:|---:|---:|
| A-LF25-BlockLocalAugStable | -0.006944444444444444 | -0.0703125 | 1.3054473942413496 | 0.0 |
| A-LF24-RandomCotangentStable | -0.017361111111111112 | -0.046875 | 1.8325081013409241 | 0.1111111111111111 |
| A-LF23-PCAOrthoMix | -0.048177083333333336 | -0.1171875 | 3.880308858763194 | 0.2222222222222222 |
| A-LF26-AugTangentFrame | -0.023871527777777776 | -0.1171875 | 1.4855420428653967 | 0.1111111111111111 |

解释：A-LF23 是 LineC scout 最强，但 task/AUC 很差；A-LF25 task 最接近 near-anchor，但 LineC 为 0。

### 16.4 Depth 7 hardening

执行规模：

```text
candidates = A-LF25,A-LF24,A-LF23
train_size = 1024
val/test = 512
epochs = 8
rows = 45
summary_rows = 5
```

Hardening 结果：

| candidate | mean_delta_vs_mlp | worst_delta_vs_mlp | max_AUC_time_ratio_vs_mlp | LineC pass rate | all-pass |
|---|---:|---:|---:|---:|---:|
| A-LF25-BlockLocalAugStable | 0.002170138888888889 | -0.029296875 | 1.397626945696755 | 0.1111111111111111 | 0 |
| A-LF24-RandomCotangentStable | -0.006076388888888889 | -0.044921875 | 1.5280893491609442 | 0.1111111111111111 | 0 |
| A-LF23-PCAOrthoMix | -0.012152777777777778 | -0.044921875 | 1.6739935241659412 | 0.1111111111111111 | 0 |

结论：新的 signal-frame estimator 仍未达到 near-anchor。A-LF25 能保正 mean delta，但 worst/AUC/LineC 不足；A-LF23 的 LineC scout 优势在 hardening 后退到 1/9。

### 16.5 Depth 7 functional 复验

对 A-LF25/A-LF24 跑最有信息量的四个 functional event：

```text
F26-D2-controlResidualComposite
F26-D3-downscaleQuadPolicy
F26-D4-lowRankLogitSubspace
F26-D4-unlabeledCovarianceTransport
```

总结果：

```text
candidate_rows = 24
aggregate_rows = 8
any_exploration_pass = 0
any_strict_majority_pass = 0
any_strict_all_pass = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
combined_bridge_any_train_shuffle_robust_all_pass = 0
```

最强 task/control 行：

```text
base = A-LF25-BlockLocalAugStable
candidate = F26-D2-controlResidualComposite
train_seed_base = 12262400
source_vs_noop = +0.01171875
source_vs_best_control = +0.015625
LineC_seed_pass_count = 0/5
CEp99_delta_vs_noop = +0.9030361175537109
strict_majority_pass = 0
exploration_gate_pass = 0
```

最强 NoOp gain 行：

```text
base = A-LF24-RandomCotangentStable
candidate = F26-D3-downscaleQuadPolicy
best_source_vs_noop = +0.03515625
best_source_vs_control = +0.0078125
best_linec_seed_pass_count = 0/5
```

最强 LineC 行：

```text
base = A-LF24-RandomCotangentStable
candidate = F26-D2-controlResidualComposite
best_linec_seed_pass_count = 3/5
best_source_vs_control = -0.00390625
best_source_vs_noop = +0.03125
```

解释：Depth 7 仍是 task/control 与 LineC 不同位，并且 task/control 强行伴随 tail 或 LineC failure；没有 S4a support row。

## 17. Depth 7 后结论

v12.26.1 继续推进后仍未达成目标：

```text
S5 official success = no
S4a label-free functional exploration = no
label-free near-anchor = no
```

新增 no-go 边界：

```text
1. 纯输入几何 signal-frame estimator 不能替代 B320-current 的 trainprobe-informed signal frame。
2. PCA/cotangent/augmentation/block-local frames 可产生局部 LineC 或 task 迹象，但不能同时满足 task、worst/AUC 与 LineC。
3. Functional 复验显示 A-LF25/A-LF24 上仍是 task/control 与 LineC 分离；没有 support row。
4. 当前合法 route 仍应保持 R1，而不是 S4a/S5。
```

本轮之后，我已经不确定继续在现有 A-LF token family 上扩组合会产生有效机制；更合理的下一步需要重新设计新的 label-free signal source，而不是继续排列已有 projection tokens。

## 18. 用户再次追问后的 stop-contract 复核

用户再次要求确认 v12.26.1 是否达成目标，若未达成则继续。复核计划与最终 route 后，结论仍是：没有达成 S5，也没有打开 S4a；但当前已经满足计划允许的 hard-stop no-go 条件。

复核依据：

```text
计划 mandatory fallback depths = Depth 1..4
当前实际 fallback_depth = 7
route = R1-LabelFreeBaseMissing
minimum_success = Minimum Success E
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_all_executed = 1
required_artifact_missing_count = 0
```

关键最终指标仍为：

```text
line_a_scout_rows = 351
line_a_hardening_rows = 234
line_a_depth7_scout_rows = 90
line_a_depth7_hardening_rows = 45
line_a_near_anchor_pass_count = 0
line_a_official_pass_count = 0
functional_candidate_rows = 138
functional_aggregate_rows = 46
line_f_exploration_gate_pass = 0
line_f_official_gate_pass = 0
s4a_single_positive_rows = 0
support_count = 0
task_positive_rows = 8
linec_majority_rows = 7
linec_multisketch_rows = 1702
linec_multisketch_pass_rows = 109
```

因此，这次不是因为 gate fail 后轻易停止，而是：

```text
1. 已执行计划要求的 Depth 1..4，并继续推进到 Depth 7。
2. Depth 5/6/7 分别尝试 low-quad/direct-read、A-LF10 task-anchor LineC repair、label-free signal-frame estimator。
3. 仍没有 label-free near-anchor、没有 S4a support row、没有 functional exploration pass。
4. forbidden/provenance/code audit 仍闭合，不存在 artifact 缺失。
```

最终科学判断不变：

```text
v12.26.1 没有达成 S5；
没有打开 label-free S4a；
合法 route 仍是 R1-LabelFreeBaseMissing；
不允许 promotion；
允许 final stop。
```

我已经不确定继续在现有 A-LF projection/PCA/cotangent/augmentation token family 上扩组合能形成有效机制。若继续进入新版本，应重新设计新的 label-free signal source，而不是继续排列已有 projection tokens；否则会变成低价值网格搜索。

本节写入后重新执行 finalizer 打包，最终 zip sha 以 `v1226_route_decision.json` 为准。
