# DG-KAN v12.24 S4toS5 FunctionalBridge LabelFreeClassic 实验结果复盘

生成时间：2026-05-25（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功、不补填未执行数据。

## 1. 初始理解

v12.23 的最终状态不是 S5，而是 S4：

```text
route = S4-FunctionalP3Opened
official_success_reached = 0
p4_pass = 0
```

已知最强信号来自 I24 query-reference direct-logit compensation + quad_only post-P3 update，但它仍存在三类 blocker：

```text
1. provenance blocker：query-reference compensation 不能 official promotion。
2. robustness blocker：train-shuffle 与 multi-sketch all-pass 不稳定。
3. geometry/task blocker：P3 pass 与 P4 task gain 在当前 family 中不总是重合。
```

v12.24 因此需要尝试 precommit/train-stream replacement、Line A/C 修复、Line D formal hardening，并用 R0-R12 代码审计和 required artifact manifest 做 fail-closed 收口。

## 2. 本轮新增代码修改

### 2.1 A66-A69

修改文件：

```text
experiments/run_v1218_b320_label_free_ablation.py
```

新增候选：

```text
A66-LineCAwareUnlabeledMultiSketchFrame
A67-ReservoirStabilizedResidualFrame
A68-A51TrainProbeCouplingPreservingFrame
A69-A51MultiSketchRMSQBoundQNoReadoutRepair
```

合理性：

- 四个候选均沿着 v12.24 计划要求的 label-free LineC repair 方向推进。
- 均 `uses_y_for_stats=0`，不使用标签统计构造 frame。
- 没有降低 Line A gate；如果 task/LineC 不达标，仍必须 fail-closed。

### 2.2 Train-stream bridge

新增文件：

```text
experiments/run_v1224_train_stream_functional_bridge.py
```

实现内容：

- `I28-TrainStreamEMACompensation`
- `I29-TrainProbeMedianCompensation`
- `I29-TrainProbeTrimmedMeanCompensation`
- `I31-NullLogitCompensatedShadowRelease`

这四个 v12.24 bridge 候选均使用 train-stream micro-batch 构造 direct-readout compensation delta：

```text
uses_query_batch = 0
uses_train_batch = 1
precommit_available = 1
```

对照包含：

```text
NoOpMatchedOverhead
TrainDirectLogitCompensatedRandomControl
AdamWParallelDirection
SNR-only
RandomMatchedNorm
```

注意：`AdamWParallelDirection` 对照使用 label/CE-derived optimizer delta，因此只作为 control audit，不作为 loss-agnostic promotion source。

### 2.3 Line D formal hardening

新增文件：

```text
experiments/run_v1224_classic_hardening.py
```

实现内容：

- Rational/Fourier x MNIST/Fashion-MNIST x seeds 0,1,2。
- 同步训练 `MLP-h160-AdamW` reference。
- 写入 `mean_delta_vs_MLP`、`step_ratio_vs_mlp`、`memory_ratio_vs_mlp`、LineC 指标与 exploration gate。

### 2.4 Finalizer / 审计打包

新增文件：

```text
experiments/run_v1224_finalize_functional_bridge.py
```

实现内容：

- 聚合 Line A、Line I/B、Line D、T1B microprobe。
- 写入 `v1224_core_code_review_manifest.csv`，覆盖计划要求的 R0-R12。
- 写入 fallback manifest、required artifact manifest、route decision、13 个 SVG figure。
- 打包 `v1224_code_review_packet.zip`。

审计修复：

- 首次 finalizer 中 `t1b_microprobe_auc=NaN` 会进入 JSON。已新增 `json_safe()`，把 NaN/inf 写为空字符串，避免标准 JSON 审计歧义。
- 该修复只改变序列化，不改变任何实验指标或 gate。

## 3. 最终 route

最终 route：

```text
route = R4-S4toS5NoGoAfterDepth2Fallbacks
minimum_success = Minimum Success F
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
fallback_all_executed = 1
```

解释：v12.24 不是 S5，也没有打开 S4a/S4c/S1 exploratory route。它是在计划要求的 depth-2 fallback 都执行后形成的 fail-closed no-go。

## 4. Line A：A66-A69 label-free repair

执行规模：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
train_size = 1024
val/test = 512
epochs = 8
raw rows = 99
summary rows = 11
```

关键候选：

| candidate | mean_delta_vs_A0 | worst_delta_vs_A0 | max_AUC_time_ratio_vs_mlp | LineC pass rate |
|---|---:|---:|---:|---:|
| A1-noYForStats | 0.001736 | -0.015625 | 1.148017 | 0.0 |
| A51-StagedUnlabeledAdapt-warm1-r002 | 0.003689236111111111 | -0.013671875 | 1.072721 | 0.0 |
| A66-LineCAwareUnlabeledMultiSketchFrame | -0.013021 | -0.044922 | 1.279185 | 0.0 |
| A67-ReservoirStabilizedResidualFrame | -0.100043 | -0.150391 | 3.270011 | 0.0 |
| A68-A51TrainProbeCouplingPreservingFrame | -0.013021 | -0.044922 | 1.279185 | 0.0 |
| A69-A51MultiSketchRMSQBoundQNoReadoutRepair | -0.103516 | -0.152344 | 3.332316 | 0.0 |

结论：

- A51 仍是最强 label-free task candidate，mean delta 为正，worst delta 也在较小范围。
- A66/A68 没有修复 LineC，而且 task/AUC 比 A51 明显差。
- A67/A69 明显破坏 task 和效率。
- LineC pass rate 全部为 0，因此 Line A 没有打开 S1 或 exploration gate。

## 5. Line I/B：train-stream/precommit compensation bridge

执行设置：

```text
source = v12.23 main targeted P3 source
source_actuator = I24-DirectLogitCompensatedQuadRelease
source_dataset = KMNIST
source_seed = 0
source_norm_budget = 0.009
source_signed_direction = 1.0
train_seed_bases = 12240400,12241400,12242400
epochs = 8
role_policy = quad_only
```

候选聚合：

| candidate | best source-noop delta | best source-control delta | strict majority pass count | strict all pass count |
|---|---:|---:|---:|---:|
| I28-TrainStreamEMACompensation | -0.019531 | -0.019531 | 0/3 | 0/3 |
| I29-TrainProbeMedianCompensation | 0.011719 | 0.0 | 0/3 | 0/3 |
| I29-TrainProbeTrimmedMeanCompensation | 0.0 | -0.019531 | 0/3 | 0/3 |
| I31-NullLogitCompensatedShadowRelease | 0.011719 | -0.007812 | 0/3 | 0/3 |

重要细节：

- I31 在 `train_seed_base=12242400` 有 `linec_seed_pass_count=5/5`，但 task gate 仍为 0：source acc `0.738281`，NoOp `0.726562`，best control `0.746094`，source 输 control。
- I29-median 最好 task 只达到 source 与 best control 持平：source `0.746094`，best control `0.746094`，不满足 `+0.005` gate。
- 所有候选 `uses_query_batch=0`，但没有一个同时通过 task gate 和 multi-sketch gate。

结论：

```text
any_strict_majority_pass = 0
any_strict_all_pass = 0
any_train_shuffle_robust_majority_pass = 0
any_train_shuffle_robust_all_pass = 0
```

这说明 v12.23 query-reference I24 的 audit gain 仍不能被 train-stream/precommit compensation 复制。

## 6. Line T/C：T1B online microprobe 与 response-distillation fallback

产物：

```text
v1224_t1b_online_microprobe.csv
v1224_response_distillation_fallback.csv
```

T1B microprobe：

```text
rows = 12
positive_rows = 0
class_count = 1
auc = ""
exploration_gate_pass = 0
```

解释：因为 train-stream bridge 没有任何 `strict_majority_pass=1`，microprobe target 只有单类，AUC 不定义，不能打开 S2。

Response-distillation fallback：

- 已检查 v12.23 completed-response artifacts。
- 全部标记为 `diagnostic_only=1`、`precommit_available=0`、`promotion_allowed=0`。

结论：当前仍没有 precommit-safe、loss-agnostic visibility source。

## 7. Line D：Rational/Fourier formal hardening

执行规模：

```text
families = Rational, Fourier
datasets = MNIST, Fashion-MNIST
seeds = 0,1,2
rows = 12
MLP reference = MLP-h160-AdamW
```

汇总：

| family | rows | mean_delta_vs_MLP | mean LineC CouplingR2 | mean NoiseSignalLeak | mean ReservoirRatio | exploration pass |
|---|---:|---:|---:|---:|---:|---:|
| Rational | 6 | -0.189453 | 0.123338 | 0.068985 | 0.502745 | 0 |
| Fourier | 6 | -0.231445 | 0.126281 | 0.060099 | 0.648645 | 0 |

最佳 task 行：

```text
Rational / MNIST / seed 0
val_acc = 0.701172
MLP val_acc = 0.890625
delta_vs_MLP = -0.189453
LineC CouplingR2 = 0.132385
NoiseSignalLeak = 0.126911
ReservoirRatio = 0.600957
exploration_gate_pass = 0
```

LineC 有两行达到 `linec_pass=1`，但 task delta、step/memory ratio 没有同时过 gate。因此 Line D 没有打开 S4c。

## 8. 代码审计与 artifact 完整性

R0-R12 审计：

```text
v1224_core_code_review_manifest.csv
rows = 13
code_semantics_review_pass = 1
```

Fallback manifest：

```text
v1224_fallback_execution_manifest.csv
fallback_rows = 6
fallback_all_executed = 1
```

Required artifacts：

```text
v1224_required_artifact_manifest.csv
required_artifact_rows = 28
required_artifact_missing_count = 0
```

Code review packet：

```text
path = results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_code_review_packet.zip
entries/sha256 = 以最终 v1224_route_decision.json 为准
```

zip 包含新增 v12.24 代码、v12.23 关键 P4 审计脚本、核心 primitive、计划文档、执行日志、复盘日志、结果 CSV/JSON、figures 和 logs。

## 9. 最终科学结论

本轮没有达成 v12.24 的 S5 目标。

更具体地说：

1. Line A：A51 仍能维持 task 表现，但 A66-A69 没有打开 LineC non-tearing，LineC pass rate 仍为 0。
2. Line I/B：train-stream EMA/median/trimmed/null compensation 均未复制 I24 query-reference 的 S5 级 task+LineC+robustness。
3. Line T/C：microprobe 没有正例，response-distillation 仍是 completed-response diagnostic，不能 promotion。
4. Line D：Rational/Fourier formal hardening 已执行，但相对 MLP task gap 太大，未过 exploration gate。
5. R0-R12 代码审计与 required artifacts 均闭合，没有 artifact 缺失。

因此最终状态是：

```text
R4-S4toS5NoGoAfterDepth2Fallbacks
```

这不是未执行停止，而是 depth-2 fallback 执行后的 no-go。

## 10. 下一步判断

不建议继续简单扩大当前 I24/I26/I28-I31 的 compensation grid：

- query-reference 有 audit-only gain，但 provenance 不合格；
- train-stream compensation 已经测试 EMA/median/trimmed/null 以及 3 个 train-shuffle seed，仍无 strict pass；
- LineC 可以单独过，但 task/control gate 不闭合。

更合理的下一轮方向：

```text
1. 重新设计 precommit observable，而不是只改变 direct-readout compensation 的聚合方式。
2. 对 I31 这类 LineC 全 pass 但输 control 的方向，构造 control-resistant update，而不是继续调同一 compensation delta。
3. 将 Line D 的 Rational/Fourier 从当前 MLP-gap fail 状态推进，需要先解决 task gap，而不是只优化 LineC。
```

## 11. 用户要求继续后的补充复盘：I30/I32 与 Line D candidate fallback

### 11.1 为什么继续

上一轮 v12.24 仍是：

```text
route = R4-S4toS5NoGoAfterDepth2Fallbacks
official_success_reached = 0
p4_pass = 0
```

重新核对计划后，确认还有三项 fallback 需要补齐，不能只停在 I28/I29 和 Rational/Fourier baseline：

```text
1. I28/I29 fail -> I30/T1B-guided compensation。
2. P3/P4 decouple -> I32 policy-aware P3。
3. Line D 计划列出的 Rational/Fourier candidate hardening。
```

因此本轮继续执行，并把新增结果纳入 finalizer、fallback manifest、required manifest、T1B microprobe 和 code review zip。

### 11.2 本轮代码修改是否合理

新增文件：

```text
experiments/run_v1224_i30_i32_policy_bridge.py
```

用途：

- `I30-T1BGuidedDirectCompensation`：只用 train-stream probe 的 logit drift、tail、entropy 等无标签特征选择 I26/I27、sign、scale、comp mode。
- `I32-PolicyAwareP3QuadProbe`：用无标签 train-probe policy probe 在 `quad_only/quad_direct/direct_only` 中选择 post-P3 policy，再做 P4/LineC 审计。
- 所有输出均写 `promotion_allowed=0`；不把 fallback 诊断直接写成 S5。

修改文件：

```text
experiments/run_v1224_classic_hardening.py
experiments/run_v1224_finalize_functional_bridge.py
```

修改内容：

- `run_v1224_classic_hardening.py` 增加计划中的候选：
  `RationalB7me/B7lp/B7lz/B7ma` 与 `FourierB4p/B4q/B4v/B4w`。
- `run_v1224_finalize_functional_bridge.py` 将 I30/I32 与 classic candidate fallback 纳入 route、fallback manifest、required manifest、T1B microprobe 和 zip。

合理性判断：

- 这些修改没有降低任何 gate 阈值。
- I30/I32 是计划指定 fallback，不是临时编造成功路径。
- I30/I32 使用 train-stream/probe 信息，显式记录 `uses_query_batch=0`，但最终仍需要 task/control/LineC/robustness gate。
- classic candidate fallback 只是把计划列出的 Rational/Fourier 候选跑完；未把 Line D 结果直接 promotion。

### 11.3 I30/I32 结果

总览：

```text
candidate_rows = 6
aggregate_rows = 2
any_strict_majority_pass = 0
any_strict_all_pass = 0
any_train_shuffle_robust_majority_pass = 0
any_train_shuffle_robust_all_pass = 0
promotion_allowed = 0
```

I30：

| train_seed_base | source acc | noop acc | best control acc | source-noop | source-control | LineC pass | strict majority |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12240400 | 0.7265625 | 0.71875 | 0.75390625 | 0.0078125 | -0.02734375 | 0/5 | 0 |
| 12241400 | 0.75 | 0.72265625 | 0.7578125 | 0.02734375 | -0.0078125 | 3/5 | 0 |
| 12242400 | 0.7265625 | 0.734375 | 0.76171875 | -0.0078125 | -0.03515625 | 3/5 | 0 |

I32：

| train_seed_base | source acc | noop acc | best control acc | source-noop | source-control | LineC pass | strict majority |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12240400 | 0.71875 | 0.71875 | 0.76171875 | 0.0 | -0.04296875 | 0/5 | 0 |
| 12241400 | 0.74609375 | 0.75390625 | 0.75 | -0.0078125 | -0.00390625 | 2/5 | 0 |
| 12242400 | 0.734375 | 0.73828125 | 0.76171875 | -0.00390625 | -0.02734375 | 3/5 | 0 |

聚合 best：

```text
I30 best_source_vs_noop_acc_delta = 0.02734375
I30 best_source_vs_control_acc_delta = -0.0078125
I32 best_source_vs_noop_acc_delta = 0.0
I32 best_source_vs_control_acc_delta = -0.00390625
```

结论：

- I30 能在一个 train-shuffle seed 上赢 NoOp，但仍输 matched control。
- I32 的 policy-aware probe 没有改善 P4/control 闭合。
- 两者都没有 strict majority/all pass，也没有 train-shuffle robust pass。
- 因此 I30/I32 没有打开 S4b，更不能写 S5。

### 11.4 T1B microprobe 更新

finalizer 修复后，T1B online microprobe 不再只读 I28/I29，也纳入 I30/I32 summary rows。

结果：

```text
rows = 18
positive_rows = 0
class_count = 1
auc = ""
exploration_gate_pass = 0
```

解释：I28/I29/I30/I32 均没有 strict-majority 正例，因此 target 仍为单类，AUC 不定义。不能把它作为 value source。

### 11.5 Line D candidate fallback 结果

执行矩阵：

```text
families = RationalB7me,RationalB7lp,RationalB7lz,RationalB7ma,FourierB4p,FourierB4q,FourierB4v,FourierB4w
datasets = MNIST,Fashion-MNIST
seeds = 0
rows = 16
exploration_pass_rows = 0
```

Family summary：

| family | rows | mean_delta_vs_MLP | mean CouplingR2 | mean NoiseLeak | mean Reservoir | exploration |
|---|---:|---:|---:|---:|---:|---:|
| RationalB7lp | 2 | -0.0224609375 | 0.1461290359103486 | 0.04982793517410755 | 0.21480131894350052 | 0 |
| RationalB7lz | 2 | -0.021484375 | 0.14295175817010397 | 0.048839105293154716 | 0.21461185812950134 | 0 |
| RationalB7me | 2 | -0.0869140625 | 0.1738808603382474 | 0.05942312814295292 | 0.5892070382833481 | 0 |
| RationalB7ma | 2 | -0.095703125 | 0.15257135411163358 | 0.12274881452322006 | 0.5930072963237762 | 0 |
| FourierB4q | 2 | -0.13671875 | 0.121605681057048 | 0.15859489142894745 | 0.5885363221168518 | 0 |
| FourierB4p | 2 | -0.1376953125 | 0.11050439296792752 | 0.18509235978126526 | 0.5565193593502045 | 0 |
| FourierB4v | 2 | -0.1728515625 | 0.13308169502494505 | 0.16328958794474602 | 0.7069712579250336 | 0 |
| FourierB4w | 2 | -0.1728515625 | 0.13308172853965616 | 0.16328931227326393 | 0.706971287727356 | 0 |

最接近 task gate 的行：

```text
RationalB7lp / MNIST
val_acc = 0.869140625
MLP val_acc = 0.890625
mean_delta_vs_MLP = -0.021484375
step_ratio_vs_mlp = 1.2519299847590781
memory_ratio_vs_mlp = 2.1016346983201215
CouplingR2 = 0.11629169132102457
NoiseSignalLeak = 0.058067578822374344
RealSignalReservoirRatio = 0.16125328838825226
linec_pass = 0
exploration_gate_pass = 0
```

最接近 LineC 的强 task 行：

```text
RationalB7lp / Fashion-MNIST
val_acc = 0.78125
MLP val_acc = 0.8046875
mean_delta_vs_MLP = -0.0234375
step_ratio_vs_mlp = 1.3606708948483868
memory_ratio_vs_mlp = 2.1016346983201215
CouplingR2 = 0.17596638049967261
NoiseSignalLeak = 0.04158829152584076
RealSignalReservoirRatio = 0.2683493494987488
linec_pass = 1
exploration_gate_pass = 0
```

结论：

- RationalB7lp/B7lz 相比 baseline Line D 明显缩小了 task gap，但仍没有同时满足 LineC、step/memory 和 exploration gate。
- RationalB7me/B7ma 有更多 LineC pass 行，但 task gap、memory ratio 或 reservoir 不闭合。
- FourierB4p/q/v/w 未接近 exploration。
- Line D candidate fallback 仍没有打开 S4c。

### 11.6 最终 route 更新

补充 finalizer 后：

```text
route = R4-S4toS5NoGoAfterDepth2Fallbacks
minimum_success = Minimum Success F
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
fallback_rows = 9
fallback_all_executed = 1
required_artifact_rows = 33
required_artifact_missing_count = 0
code_semantics_review_pass = 1
```

Zip 审计：

```text
path = results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_code_review_packet.zip
entries = 57
contains experiments/run_v1224_i30_i32_policy_bridge.py = True
contains v1224_i30_i32_policy_bridge.csv = True
contains v1224_classic_candidate_fallback_hardening.csv = True
```

注意：最终 zip sha256 由 finalizer 写入 `v1224_route_decision.json`。本复盘不手写该值，避免日志写入后重新打包导致 sha 循环变化。

### 11.7 本轮结论

v12.24 仍未达成 S5，也没有打开新的 exploratory route。

当前 no-go 比上一节更强，因为新增 fallback 后仍全部失败：

```text
Line A: A66-A69 未打开 LineC。
I28/I29: train-stream bridge fail。
I30: T1B-guided direct compensation 可以局部赢 NoOp，但输 control。
I32: policy-aware P3 probe 没有产生 strict pass。
T1B: microprobe 无正例，AUC 不定义。
Line D: Rational/Fourier candidate fallback 无 exploration pass。
```

下一步不应继续简单扩大 I30 的 sign/scale 或 Fourier/Rational 小网格。更合理的方向是：

```text
1. I30 需要设计能赢 matched control 的 train-stream source，而不是只赢 NoOp。
2. I32 需要新的 policy objective；当前无标签 probe 不能预测 P4/control gain。
3. Line D 若继续，应优先 RationalB7lp/B7lz 的 memory/efficiency 与 CouplingR2 gate，而不是继续扩 Fourier。
```

## 12. 用户继续要求后的补充复盘：transitive code packet 与 T1B response-distilled fallback

### 12.1 是否达成目标

仍未达成 S5 official functional success，也没有打开新的 exploratory route。

本轮继续后的 route：

```text
route = R4-S4toS5NoGoAfterDepth2Fallbacks
minimum_success = Minimum Success F
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
fallback_rows = 10
fallback_all_executed = 1
required_artifact_rows = 54
required_artifact_missing_count = 0
transitive_code_packet_missing_count = 0
```

解释：本轮不是把 no-go 改成 success，而是补齐了 v12.24 计划中两个仍应继续的缺口：

```text
1. final code packet 必须包含 v12.23/v12.24 transitive code packet。
2. T2/T3 response-level signal 存在但 T1B fail 时，必须尝试 response-distilled T1B feature generation。
```

### 12.2 本轮代码修改

修改文件：

```text
experiments/run_v1224_finalize_functional_bridge.py
```

修改内容：

```text
1. 新增 TRANSITIVE_CODE_PACKET_MEMBERS，把 v12.23 P4 compensation modes、official row scan、blend、trainable-role scan、trajectory verifier、reservoir-veto checkpoint、Line D hardening 与 summary 脚本纳入 v12.24 code review packet。
2. 新增 v1224_t1b_response_distilled_features.csv 与 v1224_t1b_response_distilled_features_summary.json。
3. 新增 response-distilled T1B summary 字段写入 route decision。
4. fallback manifest 增加 T2_T3_pass_but_T1B_fail -> response_distilled_T1B_unlabeled_feature_generation。
5. required artifact manifest 增加 response-distilled artifact 与 transitive code packet 条目。
6. 修复 response-distilled summary JSON 初版 NaN 问题，改为 JSON-safe 空字符串。
```

合理性判断：

```text
1. 这是计划 1.8 / 4.2 / 11 节 code packet 审计要求的补齐，不改变任何实验 gate。
2. response-distilled features 只使用 train-stream/unlabeled probe 字段；T2/T3 completed response 只作为 teacher boundary 诊断，不能 promotion。
3. 新增 artifact 明确 promotion_allowed=0，避免把 response-level upper-bound 冒充为 deployable source。
```

### 12.3 T1B response-distilled feature 结果

新增产物：

```text
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_t1b_response_distilled_features.csv
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_t1b_response_distilled_features_summary.json
```

关键结果：

```text
rows = 18
positive_rows = 0
class_count = 1
auc_joint = blank
precision_at_k_joint = blank
recall_at_k_joint = blank
exploration_gate_pass = 0
promotion_allowed = 0
response_teacher_auc_joint = 0.7383255633255633
response_teacher_precision_at_k_joint = 0.3181818181818182
response_teacher_recall_at_k_joint = 0.3181818181818182
```

解释：

```text
1. T2/T3 response-level teacher boundary 仍说明 completed-response 层存在信号。
2. 但可部署的 train-stream/unlabeled response-distilled feature rows 没有正例，target 仍是单类。
3. 因此 AUC/precision/recall 不定义，不能打开 T1B exploration，更不能作为 S5 source。
```

### 12.4 transitive code packet 修复结果

旧 zip 中缺少 v12.23 continuation 的若干关键脚本。本轮 finalizer 补齐后，code review packet 已确认包含：

```text
experiments/run_v1223_p4_compensation_modes.py
experiments/run_v1223_p4_official_row_scan.py
experiments/run_v1223_shadowp4_coupling_preserving_blend.py
experiments/run_v1223_p4_trainable_role_scan.py
experiments/run_v1223_p4_trajectory_gate_verifier.py
experiments/run_v1223_p4_reservoir_veto_checkpoint_scan.py
experiments/run_v1223_line_d_hardening.py
experiments/summarize_v1223_line_d_hardening.py
```

当前审计字段：

```text
transitive_code_packet_required_count = 19
transitive_code_packet_missing_count = 0
zip_entries = 69
```

注意：最终 zip 会在日志写入后再次生成，因此最终 sha256 以最终只读复核和 `v1224_route_decision.json` 为准。本复盘不手写会随日志自引用变化的中间 sha。

### 12.5 本轮最终科学结论

v12.24 仍未达成目标：

```text
S5 official functional success = no
P4 pass = no
T1B response-distilled exploration = no
Line D candidate exploration = no
```

但 no-go 现在更完整：

```text
1. I28/I29/I30/I31/I32 没有形成 robust train-stream bridge。
2. T1B microprobe 和 response-distilled features 都没有正例，AUC 不定义。
3. T2/T3 response-level upper-bound 信号不能转成 precommit/train-stream deployable source。
4. Rational/Fourier candidate fallback 没有打开 S4c。
5. final code packet 已补齐 v12.23/v12.24 transitive scripts，code packet 缺口不再是 blocker。
```

因此当前合法状态仍是：

```text
R4-S4toS5NoGoAfterDepth2Fallbacks
```

下一步如果继续，不应把 T2/T3 response teacher 当成 deployable value source；应重新设计能在 train-stream/precommit 阶段产生正例的 signal/reservoir observable，或把 RationalB7lp/B7lz 的 memory/efficiency 与 CouplingR2 gate 作为独立 classic-family hardening 主线推进。

## 13. 再次继续后的复盘：Line D Depth-2 repair 与 stop-contract 闭合

### 13.1 是否达成目标

没有达成 S5，也没有打开 S1/S2/S4a/S4c 任一 exploration route。

本轮继续后的最终 route：

```text
route = R4-S4toS5NoGoAfterDepth2Fallbacks
minimum_success = Minimum Success F
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_budget_exhausted = 1
fallback_depth = 2
fallback_rows = 12
fallback_all_executed = 1
required_artifact_missing_count = 0
```

解释：这仍不是成功。它的含义是 v12.24 的 mandatory fallback 深度已经推进到 Depth-2，且 required artifacts 不缺失。

### 13.2 本轮为什么继续

上一节 no-go 仍有两个计划口径不够严：

```text
1. route 没有显式记录 hard_budget_exhausted=1 和 fallback_depth>=2。
2. 计划要求失败时输出 executable next-generation fallback，而不只是文字建议。
```

同时，Line D candidate fallback 中最接近的 RationalB7lp/B7lz 暴露出明确 blocker：

```text
task gap 接近 exploration gate；
Fashion-MNIST 的 CouplingR2 可过 LineC；
但 memory ratio 约 2.10，MNIST CouplingR2 不够，组合 gate 不闭合。
```

因此本轮继续执行了 Rational readscale/crossWarm/pairStd/pairNorm 的 Depth-2 repair。

### 13.3 代码修改

修改文件：

```text
experiments/run_v1224_classic_hardening.py
experiments/run_v1224_finalize_functional_bridge.py
```

修改内容：

```text
1. 新增 RationalB7dq/B7dr/B7ds/B7dt/B7em/B7en/B7eq/B7er 映射。
2. finalizer 纳入 v1224_classic_depth2_efficiency_hardening.csv/json。
3. route 增加 hard_budget_exhausted、fallback_depth、classic_depth2_rows、classic_depth2_exploration_pass_rows。
4. fallback manifest 增加 LineD_candidate_fallback_fail -> Rational_readscale_crosswarm_pairnorm_depth2_hardening。
5. 新增 v1224_next_generation_fallback_plan.csv/json/md。
6. v1224_core_code_review_manifest.csv 增加 artifact_fields、gate_relation、review_status，并修复函数 line range 查找。
```

合理性判断：

```text
1. 这些修改没有降低 LineD gate。
2. RationalB7dq/B7dr/B7ds/B7dt 是 readscale/crossWarm efficiency repair。
3. RationalB7em/B7en/B7eq/B7er 是 pairStd/pairNorm/logitBias/readscale 的 reservoir/geometry repair。
4. next-generation fallback plan 只记录可执行命令，promotion_allowed=0。
```

### 13.4 Line D Depth-2 结果

执行矩阵：

```text
families = RationalB7dq,RationalB7dr,RationalB7ds,RationalB7dt,RationalB7em,RationalB7en,RationalB7eq,RationalB7er
datasets = MNIST,Fashion-MNIST
seeds = 0
train_size = 1024
val_size = 512
epochs = 8
rows = 16
```

总结果：

```text
exploration_pass_rows = 0
promotion_allowed = 0
```

最接近 task gate 的行：

| family | dataset | delta_vs_MLP | step_ratio | memory_ratio | CouplingR2 | LineC pass | exploration |
|---|---|---:|---:|---:|---:|---:|---:|
| RationalB7en | MNIST | -0.021484375 | 1.3622519040172953 | 2.1016346983201215 | 0.11629072993049316 | 0 | 0 |
| RationalB7em | MNIST | -0.021484375 | 1.345775430701884 | 2.1016346983201215 | 0.11583633984652064 | 0 | 0 |
| RationalB7eq | MNIST | -0.021484375 | 1.3481769451810812 | 2.1016346983201215 | 0.11543581587342444 | 0 | 0 |
| RationalB7er | MNIST | -0.021484375 | 1.3394975935980358 | 2.1016346983201215 | 0.11543569641139528 | 0 | 0 |

最接近 LineC 的强 task 行：

| family | dataset | delta_vs_MLP | step_ratio | memory_ratio | CouplingR2 | NoiseLeak | Reservoir | LineC pass | exploration |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RationalB7en | Fashion-MNIST | -0.0234375 | 1.2194536746032019 | 2.1016346983201215 | 0.17596690876225063 | 0.04158257693052292 | 0.26844078302383423 | 1 | 0 |
| RationalB7em | Fashion-MNIST | -0.0234375 | 1.223651174537795 | 2.1016346983201215 | 0.16910627443233062 | 0.03842644393444061 | 0.2661678194999695 | 1 | 0 |

失败边界：

```text
1. B7em/B7en/B7eq/B7er 保留了接近 task gate 的表现，但 memory_ratio 仍约 2.10，MNIST CouplingR2 不足。
2. B7dq/B7dr/B7ds/B7dt 的 readscale/crossWarm efficiency 方向没有修复 task，MNIST/Fashion-MNIST delta 明显恶化。
3. 因此 Line D Depth-2 没有打开 S4c。
```

### 13.5 core code review manifest 更新

`v1224_core_code_review_manifest.csv` 当前：

```text
rows = 14
manual_review_required = 0
code_semantics_review_pass = 1
```

本轮修复后，manifest 不再只是粗略字段；新增了：

```text
artifact_fields
gate_relation
review_status
```

关键审计结论：

```text
R2/R6/R7/R8 仍明确标记 query/validation/future 或 CE 风险，因此只能 audit-only。
R3/R4/R13 是 v12.24 train-stream/precommit 路径，但没有通过 robust gate。
R9/R10 记录 metric_seed_shared 和 matched_control_scope，避免复现 v12.23 早期 seed/scope 污染。
R11 记录 LineD hardening gate，Depth-2 仍未通过。
```

### 13.6 executable next-generation fallback

新增产物：

```text
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_next_generation_fallback_plan.csv
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_next_generation_fallback_plan.json
results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5/v1224_next_generation_fallback_plan.md
```

内容包括 3 条可执行 fallback command：

```text
NG1-rational-readscale-crosswarm-depth2
NG2-train-stream-bridge-replay
NG3-policy-aware-probe-reset
```

注意：这些是下一代 fallback 复现入口，不是成功数据。它们的 `promotion_allowed=0`，不能被当成 S5 证据。

### 13.7 本轮最终结论

v12.24 仍未达成目标：

```text
S5 official functional success = no
S4c classic exploration = no
S4a precommit compensation = no
S2 T1B visibility = no
S1 label-free LineC exploration = no
```

但本轮把 v12.24 从“结果 no-go”推进为更严格的“Depth-2 no-go”：

```text
1. Line D Depth-2 readscale/crossWarm/pairStd/pairNorm repair 已实际执行。
2. route 现在显式记录 hard_budget_exhausted=1 和 fallback_depth=2。
3. code review manifest 更细，包含 artifact_fields/gate_relation。
4. executable next-generation fallback plan 已产出。
5. required artifacts 缺失仍为 0。
```

因此当前科学结论不变但边界更清楚：

```text
Rational 系列可接近 task gate，但效率/memory 与跨数据集 CouplingR2 仍卡住；
train-stream/precommit functional bridge 没有复制 v12.23 query-reference audit-only task gain；
T1B/response-distilled visibility 没有正例；
v12.24 不能 promotion。
```

## 14. 用户再次要求继续后的 NG2-NG7 复盘：出现近似点，但 S5 仍未闭合

### 14.1 是否达成目标

没有。

继续执行 NG2-NG7 后，最终仍是：

```text
route = R4-S4toS5NoGoAfterDepth2Fallbacks
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
```

本轮没有编造 success，也没有把 diagnostic / near-pass 写成 promotion。所有数字来自实际 CSV/JSON。

### 14.2 本轮代码修改

本轮修改范围：

```text
experiments/run_v1224_finalize_functional_bridge.py
experiments/run_v1224_train_stream_functional_bridge.py
experiments/run_v1224_i30_i32_policy_bridge.py
```

修改内容：

```text
1. finalizer 纳入 NG2-NG7 artifact、fallback manifest、required manifest、route 字段和 zip。
2. 修正 next-generation fallback command：补 source-out-dir，把 candidate-names 改成 candidates。
3. train_stream / i30_i32 bridge 增加 --label-smoothing，用于 source/noop/control 共享的公平 tail-risk repair。
```

这些修改没有降低 gate，也没有改 promotion 阈值。

### 14.3 NG2/NG3 结果

NG2 train-stream bridge replay：

```text
candidate_rows = 9
any_strict_majority_pass = 0
any_train_shuffle_robust_majority_pass = 0
best I29 source_vs_noop = +0.01171875
best I29 source_vs_control = 0.0
best I31 source_vs_noop = +0.01171875
best I31 source_vs_control = -0.0078125
```

NG3 policy-aware probe reset：

```text
candidate_rows = 6
any_strict_majority_pass = 0
any_train_shuffle_robust_majority_pass = 0
best I30 source_vs_noop = +0.02734375
best I30 source_vs_control = -0.0078125
best I32 source_vs_noop = 0.0
best I32 source_vs_control = -0.00390625
```

结论：train-stream / precommit 路径仍不能 robust beat matched control。

### 14.4 NG4 近似成功点

NG4 使用更长 epoch、更低 lr、更宽 scale：

```text
epochs = 12
lr = 0.0015
i30_scales = 0.20,0.35,0.50,0.75,1.00,1.25
```

总结果：

```text
candidate_rows = 6
any_strict_majority_pass = 0
any_train_shuffle_robust_majority_pass = 0
```

最接近的 I30 行：

```text
train_seed_base = 12240400
source_acc = 0.73828125
noop_acc = 0.71875
control_acc = 0.73046875
source_vs_noop = +0.01953125
source_vs_control = +0.0078125
LineC pass = 3/5
task_gate_pass = 0
```

失败原因：

```text
source_NLL = 1.106278896331787 <= noop_NLL = 1.138135313987732
source_CEp99 = 10.711200714111328
noop_CEp99 = 9.32567024230957
strict CEp99 requires source_CEp99 <= noop_CEp99 + 0.05
```

解读：这是本轮最重要的新边界。precommit train-stream I30 确实找到一个 acc 同时赢 NoOp/control 且 LineC majority 的近似点，但 CEp99 tail-risk 失败，不能 S5。

### 14.5 NG5-NG7 tail repair

NG5 提高 weight decay：

```text
weight_decay = 0.005
best_source_vs_noop = +0.01953125
best_source_vs_control = +0.0078125
best source_CEp99 = 10.704131126403809
best noop_CEp99 = 9.319934844970703
```

结论：weight decay 没有修复 tail blocker。

NG6 label smoothing 0.05：

```text
best_source_vs_noop = +0.0078125
best_source_vs_control = -0.00390625
```

它把 CEp99 降到约 5-6 区间，但 source 不再赢 matched control，LineC 也没有 robust pass。

NG7 label smoothing 0.02：

```text
best_source_vs_noop = +0.00390625
best_source_vs_control = -0.00390625
```

结论：轻度 smoothing 也无法保留 source-control advantage。当前存在一个明确 tradeoff：

```text
no smoothing: 有 source-control/task 近似点，但 CEp99 tail fail。
smoothing: CEp99 显著改善，但 task/control/LineC advantage 消失。
```

### 14.6 最终 route 与 artifact

最终聚合：

```text
fallback_rows = 18
fallback_all_executed = 1
required_artifact_rows = 71
required_artifact_missing_count = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
combined_bridge_any_train_shuffle_robust_all_pass = 0
t1b_response_distilled_rows = 48
t1b_response_distilled_positive_rows = 0
code_review_packet_entries = 101
code_review_packet_sha256 = 984ac0d57241c33795fc6939463d9f22374f11599c34500a2efb2237c16126bd
```

注意：本复盘写入后还会再运行 finalizer，因此最终 zip sha256 以后续只读复核或最终回复为准。

### 14.7 结论

v12.24 继续推进后仍没有达成 S5。

新增科学边界：

```text
1. NG4 证明 precommit train-stream I30 可以出现 acc 同时赢 NoOp/control 的近似点。
2. 该点不是 official success，因为 strict CEp99 tail-risk 失败，且 train-shuffle robust pass 为 0。
3. tail repair 出现 tradeoff：label smoothing 能降 CEp99，但会丢掉 source-control advantage。
4. T1B response-distilled feature 扩到 48 行后仍然 positive_rows=0，不能打开 visibility source。
```

下一步如果继续，不能再只做普通 smoothing/weight-decay 微调；需要新的 tail-safe precommit observable 或训练策略，让 NG4 的 acc/control-positive 点同时满足 CEp99 与 multi-shuffle robustness。


## 15. 用户追问后继续推进：NG8-NG10 tail-safe / contrast precommit 修复

### 15.1 是否达成目标

没有达成 S5 official functional success。

NG8-NG10 继续推进后，三个新增尝试全部保持：

```text
any_strict_majority_pass = 0
any_strict_all_pass = 0
any_train_shuffle_robust_majority_pass = 0
any_train_shuffle_robust_all_pass = 0
promotion_allowed = 0
```

本轮没有降低 gate，没有把 near-pass 或 audit-only 写成 success，也没有编造数据。

### 15.2 权限提示说明

本轮出现权限确认，是因为当前沙箱中普通只读命令和带日志重定向的 CUDA 实验命令都触发：

```text
bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted
```

这只是执行环境限制；按系统规则，重要命令在沙箱失败后必须用提权方式重跑。该权限不代表允许修改阈值或允许 promotion。

### 15.3 本轮代码修改

修改文件：

```text
experiments/run_v1224_i30_i32_policy_bridge.py
experiments/run_v1224_finalize_functional_bridge.py
```

新增内容：

```text
1. tail_safe precommit probe score：只使用 train-stream unlabeled logits 的 drift/tail/confidence/logit-abs/entropy observable。
2. contrast_tail_safe precommit selector：对 source 与 matched train-stream control 的 unlabeled probe score 做 contrast。
3. --probe-control-contrast-weight 参数。
4. NG8/NG9/NG10 artifacts 被纳入 finalizer route、fallback manifest、required artifact manifest 和 code review zip。
```

审计判断：

```text
1. tail_safe / contrast_tail_safe 不使用 label、CE、loss、LineC 或 audit target。
2. NG9 的 quad_direct role-policy 对 source/noop/control 公平共享，不是 source-only 改 loss。
3. 所有新增结果 promotion_allowed 仍为 0，必须由 finalizer gate 决定 route。
```

### 15.4 NG8：tail-safe precommit probe 没有改变近似点

NG8 目标是用 unlabeled tail-risk observable 替代 NG4 的默认 probe score，尝试避开 CEp99 tail blocker。

结果：

```text
candidate_rows = 3
selected_actuator = I26-TrainDirectLogitCompensatedQuadRelease
selected_budget = 0.0045
selected_sign = -1.0
selected_comp_mode = median
best_source_vs_noop = 0.01953125
best_source_vs_best_control = 0.0078125
best_linec_seed_pass_count = 3/5
any_strict_majority_pass = 0
any_train_shuffle_robust_majority_pass = 0
```

最接近行仍是 train seed `12240400`：

```text
source_acc = 0.73828125
noop_acc = 0.71875
best_control_acc = 0.73046875
source_NLL = 1.106278896331787 <= noop_NLL = 1.138135313987732
source_CEp99 = 10.711200714111328
noop_CEp99 = 9.32567024230957
```

结论：tail-safe probe 没有改变 selection，仍复现 NG4 的近似点；acc/control 和 LineC majority 接近，但 CEp99 tail-risk 失败，不能 S5。

### 15.5 NG9：quad_direct 改善 task，但仍低于 strict gate

NG9 在 NG8 的 tail-safe selection 上把 post-P3 role policy 改为 `quad_direct`。

结果：

```text
candidate_rows = 3
selected_actuator = I26-TrainDirectLogitCompensatedQuadRelease
selected_budget = 0.0045
selected_sign = -1.0
selected_comp_mode = median
best_source_vs_noop = 0.02734375
best_source_vs_best_control = 0.00390625
best_linec_seed_pass_count = 3/5
any_strict_majority_pass = 0
any_train_shuffle_robust_majority_pass = 0
```

最接近行：

```text
train_seed_base = 12240400
source_acc = 0.7578125
noop_acc = 0.73046875
best_control_acc = 0.75390625
source_vs_noop = +0.02734375
source_vs_best_control = +0.00390625
source_NLL = 0.9908744096755981 <= noop_NLL = 1.0013419389724731
source_CEp99 = 9.620960235595703
noop_CEp99 = 8.907975196838379
```

失败原因：source-control margin 只有 `0.00390625`，低于 strict task margin；CEp99 仍高于 NoOp+0.05。quad_direct 能提高 source-vs-NoOp，但没有闭合 matched-control 与 tail-risk gate。

### 15.6 NG10：contrast-tail-safe 改变 selection，但 matched control 跟上

NG10 用 source/control probe contrast 选择 candidate，目标是避免选到 matched control 同样受益的点。

结果：

```text
candidate_rows = 3
selected_actuator = I27-TrainDirectLogitCompensatedShadowRelease
selected_budget = 0.009
selected_sign = -1.0
selected_comp_mode = median
probe_source_raw_score = -0.9014347008615732
probe_control_raw_score = -1.1697777668386697
probe_source_control_score_delta = 0.26834306597709656
best_source_vs_noop = 0.0234375
best_source_vs_best_control = 0.0
best_linec_seed_pass_count = 3/5
any_strict_majority_pass = 0
any_train_shuffle_robust_majority_pass = 0
```

最接近行：

```text
train_seed_base = 12240400
source_acc = 0.75390625
noop_acc = 0.73046875
best_control_acc = 0.75390625
source_vs_noop = +0.0234375
source_vs_best_control = 0.0
source_NLL = 0.9914307594299316 <= noop_NLL = 1.0013419389724731
source_CEp99 = 10.97416877746582
noop_CEp99 = 8.907975196838379
```

结论：contrast-tail-safe 确实把 precommit selection 从 I26 改到 I27，但 matched control 在 accuracy 上追平 source，CEp99 tail 反而更差。因此它不是 S5 修复。

### 15.7 本轮新增边界

NG8-NG10 关闭了三个假设：

```text
1. 单纯用 unlabeled tail-risk score 重新排序，不能避开 NG4 的 CEp99 blocker。
2. 公平 quad_direct role-policy 能改善 source-vs-NoOp，但 matched-control margin 仍不足，CEp99 仍失败。
3. source-control contrast precommit selector 可以改变 actuator selection，但 matched control 仍吃掉 task advantage。
```

因此 v12.24 仍不能 promotion。当前最清晰 blocker 是：

```text
train-stream/precommit source 可以出现 NoOp-positive 或 near control-positive 信号，但无法同时满足：
- source-control strict task margin；
- CEp99 tail-risk 不劣于 NoOp；
- multi-sketch majority/all gate；
- train-shuffle robustness。
```

### 15.8 finalizer

本节写入后重新执行：

```bash
conda run -n kan python experiments/run_v1224_finalize_functional_bridge.py
```

最终 route 与 artifact manifest 以后续只读复核为准；若仍为 no-go，则不能写 S5。


### 15.9 finalizer 聚合结果

NG8-NG10 与本轮日志写入后，重新执行 finalizer。结果仍然不是 S5：

```text
route = R4-S4toS5NoGoAfterDepth2Fallbacks
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
fallback_rows = 18
fallback_all_executed = 1
required_artifact_rows = 77
required_artifact_missing_count = 0
ng8_best_source_vs_control_acc_delta = 0.0078125
ng9_best_source_vs_control_acc_delta = 0.00390625
ng10_best_source_vs_control_acc_delta = 0.0
ng8_any_train_shuffle_robust_majority_pass = 0
ng9_any_train_shuffle_robust_majority_pass = 0
ng10_any_train_shuffle_robust_majority_pass = 0
code_review_packet_entries = 111
```

最终判定：v12.24 仍未达成目标。NG8-NG10 证明新的 precommit tail-safe / contrast selector 可以改变或改善局部 near-pass，但不能同时闭合 strict task margin、CEp99 tail-risk、multi-sketch gate 与 train-shuffle robustness，因此不能写 S5。


## 16. NG11-NG18 继续推进复盘：precommit-safe bridge 仍未闭合 S5

### 16.1 本轮为什么继续

NG8-NG10 后仍未达成 v12.24 目标：`official_success_reached=0`，`p4_pass=0`。继续推进时遵守两个边界：不编造数据，不把 audit-only 或未通过 gate 的结果写成 success。本轮只测试 train-stream/precommit-safe 路径，不使用 query batch，也不降低 official gate。

### 16.2 finalizer 覆盖面修复

修改 `experiments/run_v1224_finalize_functional_bridge.py`，把 NG11-NG18 纳入：

```text
route fields
fallback_execution_manifest
required_artifact_manifest
code_review_packet.zip
```

该修改不改变 promotion 条件；只是防止新增结果游离于正式审计之外。语法检查通过：`py_compile pass`。

### 16.3 NG11/NG12：weight decay 不能修复 quad_direct tail blocker

NG11/NG12 在 NG9 的 `quad_direct` 基础上分别使用 `weight_decay=0.003` 与 `0.005`。

共同结果：

```text
selected_actuator = I26-TrainDirectLogitCompensatedQuadRelease
selected_budget = 0.0045
selected_sign = -1.0
selected_comp_mode = median
best_source_vs_noop = 0.02734375
best_source_vs_best_control = 0.00390625
best_linec_seed_pass_count = 3/5
any_strict_majority_pass = 0
any_train_shuffle_robust_majority_pass = 0
```

细节上，weight decay 只让 NLL/CEp99 有极小改善，但没有让 source-control margin 达到 strict gate，也没有解决 train-shuffle robustness。因此 NG9 的 blocker 不是简单 L2/AdamW decay 不足。

### 16.4 NG13/NG14：更强 source-control contrast 仍被 matched control 吃掉

NG13/NG14 将 `probe-control-contrast-weight` 提到 2.0 与 3.0。

结果：

```text
selected_actuator = I27-TrainDirectLogitCompensatedShadowRelease
selected_budget = 0.009
selected_sign = -1.0
selected_comp_mode = median
best_source_vs_noop = 0.0234375
best_source_vs_best_control = 0.0
best_linec_seed_pass_count = 3/5 or lower across train seeds
any_strict_majority_pass = 0
```

解释：更强 contrast 确实维持了 I27 selection，但 matched control 在 accuracy 上追平 source。这个结果说明问题不是 contrast 权重太低，而是当前 train-stream compensated source 与 matched control 在 downstream task 上不可充分分离。

### 16.5 NG15-NG18：expanded I32 policy probe 的新边界

NG15-NG18 扩展 I32 policy space：`direct_gain,direct_branch_gain,quad_direct,freeze_quad`，并分别测试最终 role policy：`quad_direct/direct_branch_gain/direct_gain/direct_only`。

结果汇总：

| run | role_policy | best_source_vs_noop | best_source_vs_best_control | linec_pass | strict |
|---|---|---:|---:|---:|---:|
| NG15 | quad_direct | -0.015625 | -0.02734375 | 0/5 or 2/5 | 0 |
| NG16 | direct_branch_gain | 0.015625 | 0.0078125 | 0/5 | 0 |
| NG17 | direct_gain | 0.01953125 | 0.01171875 | 0/5 | 0 |
| NG18 | direct_only | 0.01953125 | 0.01171875 | 0/5 | 0 |

NG17/NG18 是本轮最有信息量的失败：它们在三个 train-shuffle seed 上都能出现 source-vs-NoOp 和 source-vs-control 的正向 acc margin，但 task gate 仍失败，且 LineC 多 sketch 全 0。

关键原因来自细项：

```text
NG17/NG18 train_seed_base=12240400:
source_acc = 0.62109375
noop_acc = 0.60546875
best_control_acc = 0.609375 或 0.61328125
source_NLL = 1.4355359077453613 > noop_NLL = 1.4349497556686401
source_CEp99 = 3.5707502365112305 > noop_CEp99 = 3.561063289642334
source_CouplingR2 ≈ 0.2955 < noop_CouplingR2 ≈ 0.3114
linec_seed_pass_count = 0/5
```

这说明 direct-only/direct-gain 可以给 train-stream/precommit path 带来真实 task accuracy signal，但会压低 CouplingR2 并略微恶化 NLL/CEp99，不能形成 official S5。

### 16.6 本轮新增边界

本轮继续后，v12.24 的失败边界比 NG8-NG10 更明确：

```text
1. quad_direct + weight_decay 不能修复 CEp99/control margin。
2. stronger control-contrast selection 不能阻止 matched control 追平 source。
3. expanded I32 direct-only/direct-gain 能产生 acc gain，但 LineC/CouplingR2 与 tail-risk gate 同时失败。
4. 这些结果均 uses_train_batch=1、uses_query_batch=0，但 promotion_allowed 仍为 0，不能写 S5。
```

因此本轮没有达成目标。更精确的 no-go 是：

```text
train-stream/precommit source 的 task-accuracy signal 可以出现，但当前 observable 与 role policy 无法把 task gain、matched-control separation、NLL/CEp99 tail safety、LineC CouplingR2 同时放到同一个 candidate 上。
```

### 16.7 finalizer

本节写入后执行：

```bash
conda run -n kan python experiments/run_v1224_finalize_functional_bridge.py
```

最终 route 与 zip sha 以后续 finalizer 输出为准。若仍未通过 robust/all gate，则必须保持 no-go，不能写 S5。


### 16.8 finalizer 聚合结果

NG11-NG18 纳入 finalizer 后，最终仍未达成 v12.24 目标：

```text
route = R4-S4toS5NoGoAfterDepth2Fallbacks
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
fallback_rows = 24
fallback_all_executed = 1
required_artifact_rows = 93
required_artifact_missing_count = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
combined_bridge_any_train_shuffle_robust_all_pass = 0
transitive_code_packet_missing_count = 0
code_review_packet_entries = 127
code_review_packet_sha256 = 以 v1224_route_decision.json 当前值为准
```

最终判定仍是 no-go，不是 S5。新增 NG11-NG18 关闭了若干可疑空档，但没有产生满足 strict task + LineC majority/all + train-shuffle robustness 的 precommit-safe candidate。

最终可写结论：v12.24 已从 NG8-NG10 的 near-pass 继续推进到 NG11-NG18，确认当前 train-stream bridge 的最强新信号是 `direct_gain/direct_only` 下的 accuracy gain；但该信号伴随 NLL/CEp99 轻微劣化和 LineC 0/5，因此不能 official promotion。
