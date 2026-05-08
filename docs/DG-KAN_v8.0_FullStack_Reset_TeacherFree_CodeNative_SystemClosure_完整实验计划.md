# DG-KAN v8.0 Full-Stack Reset：Teacher-Free Autonomous Advantage、Code-Native Candidate Factory 与 Kernel-Native System Closure 全面实验方案

> 本方案是对 v7.9 之后路线的全面重构，不是继续在已有计划上打补丁。  
> 核心判断是：过去几轮已经证明了三个事实：第一，KAN-family 存在超过 MLP 的表达力信号；第二，teacher-free M13 已经稳定接近 official gate；第三，当前实验系统还没有真正把 self-bootstrap、core primitive repair、phase-mapped profiler、fused/streaming package 做成 code-native measured path。  
> 因此 v8.0 不应继续“一个 probe 一个 probe 地试”，而应建立一个完整研究程序：先把实验系统变成可信的 candidate factory，再并行推进 autonomous margin closure、representation repair、kernel-native efficiency 和 generalization validation。

---

## 0. 结论先行

v7.9 没有达到 teacher-free official success。官方候选 `M13` 是 teacher-free、strict、GradPass，并且具有稳定正向信号，但它的 macro validation gap 是：

$$
\Delta Acc_{\text{macro,val,M13}}=+0.017838541666666666,
$$

而 official gate 是：

$$
\Delta Acc_{\text{macro,val}}\geq+0.0200.
$$

因此差距是：

$$
0.0200-0.017838541666666666=0.002161458333333334.
$$

这不是大失败，但也不能四舍五入成功。`M12+C3` 可以达到约 `+0.02057`，但它使用 external C3 teacher，只能作为 diagnostic assisted result，不能作为 PureKAN-NG teacher-free official success。v7.9 还显示，A2S/M9 structural screen 最高到 `+0.019921875`，几乎贴到 threshold，但仍未过线，而且 S2 不稳。这说明当前不是“KAN 没表达力”，而是“teacher-free autonomous margin 没有厚到足以稳健过 official gate”。

更深层的问题在代码侧：`run_gafu_v79_real.py` 已经把 candidate metadata、teacher leak audit、contract validator 显式落盘，这修正了 v7.8 的 route 审计风险；但真实训练/评估主体仍主要是复用旧 runner 后 postprocess。也就是说，v7.9 建立了可信审计，但没有真正实现 candidate factory。self-bootstrap、core primitive representation repair、phase-mapped profiler、dense-preserving fused/streaming package 仍然没有形成可测的主路径。

所以 v8.0 的目标不是“再找一个小超参让 M13 多 0.2%”，而是重建系统：

$$
\boxed{
\text{Code-native Candidate Factory}
+
\text{Teacher-free Autonomous Margin Closure}
+
\text{Kernel-native System Closure}
+
\text{Robustness / Scaling Validation}
}
$$

---

## 1. v8.0 总体目标

v8.0 的总目标是证明或证伪下面这句话：

$$
\boxed{
\text{PureKAN-NG 在无外部 teacher 条件下，能够稳定、严格、可复现、系统性超过 MLP-AdamW。}
}
$$

这里“稳定、严格、可复现、系统性”分别有明确含义。

稳定：不是单个 seed、单个 split 或单个 diagnostic run。必须在 10 seeds、3 个 primary datasets、预注册 validation protocol 下成立。

严格：candidate 必须是 PureKAN candidate。不能有 external teacher、不能有 non-KAN trainable parameter，必须使用 manual forward / manual backward / manual update，不能依赖 PyTorch loss.backward graph 来训练 KAN path。

可复现：所有 pass 必须来自落盘 CSV/JSON/log/manifest，no fake、no proxy、no placeholder、no hand-filled conclusion。

系统性：不能只赢 final accuracy。至少要通过 FullGridS2；正式成功还要进入 S1 / TimeAUC / ProfilerPass；强成功还要在 sample efficiency、noise robustness、scaling 或 geometry Pareto 中至少一个维度继续占优。

v8.0 将目标分为三层：

### 1.1 v8.0 Minimum Success

$$
\boxed{
\text{CodeNativePass}
\land
\text{TeacherFreeMacroSignificantPass}
\land
\text{StrictPass}
\land
\text{GradPass}
\land
\text{FullGridS2Pass}
}
$$

其中：

$$
\Delta Acc_{\text{macro,val}}\geq0.0200,
$$

$$
CI_{95\%,macro}^{low}>0,
$$

$$
p_{\text{Holm}}<0.05,
$$

$$
\Delta Acc_{\text{macro,test}}\geq0.015.
$$

并且：

```text
external_teacher_used = 0
external_teacher_logits_used = 0
external_teacher_forward_used = 0
nonKAN_param_count = 0
manual_forward = 1
manual_backward = 1
manual_update = 1
uses_loss_backward = 0
```

### 1.2 v8.0 Formal System Success

$$
\boxed{
\text{MinimumSuccess}
\land
\text{FullGridS1Pass}
\land
\text{TimeAUCPass}
\land
\text{ProfilerPass}
}
$$

其中：

$$
r_{\text{mem,max}}<1.00,
$$

$$
r_{\text{step,max}}\leq1.35,
$$

$$
ValLossAUC_{\text{time,KAN}}\leq ValLossAUC_{\text{time,MLP}}.
$$

ProfilerPass 要求真实 phase-mapped kernel profiler：

```text
mapped_kernel_time_fraction >= 0.90
unknown_kernel_time_fraction <= 0.10
top3 phase time explain >= 0.70
```

### 1.3 v8.0 Strong Research Success

$$
\boxed{
\text{FormalSystemSuccess}
+
\text{ECE/NLL improvement}
+
\text{SampleEfficiencyOrRobustnessPass}
+
\text{ScalingSmokePass}
}
$$

其中：

$$
ECE_{\text{KAN}}\leq ECE_{\text{MLP}},
$$

$$
NLL_{\text{KAN}}\leq NLL_{\text{MLP}},
$$

并且至少一个扩展维度成立：

$$
AUC_{\text{data,KAN}}>AUC_{\text{data,MLP}},
$$

或：

$$
AccDrop_{\text{KAN}}<AccDrop_{\text{MLP}}
$$

for at least two noise settings.

---

## 2. v8.0 的核心思想：不是修一个点，而是建立四条并行主线

v8.0 不再采用“一轮只修一个 blocker”的线性方式。线性迭代太慢，而且容易把某个 near-pass 误当成终点。v8.0 改为四条并行主线，并用统一决策层合流。

### 主线 A：实验系统重构

目标是把 v7.9 的 metadata-first audit 升级成 code-native runner。所有候选必须从 `CandidateSpec` 构造，不能只在 postprocess 中重命名或标记。teacher contract 要分清 external teacher 与 self-bootstrap，strict contract 要成为 route 前置条件。

### 主线 B：teacher-free autonomous margin closure

目标是闭合 M13 从 `+0.01784` 到 `+0.0200` 的差距。先判断是 validation estimator、calibration/margin、representation rank、early dynamics 还是 hard-sample issue。然后针对性执行 self-bootstrap、objective repair、representation repair。

### 主线 C：system / kernel-native closure

目标是确保一旦 teacher-free macro pass 出现，candidate 能稳定进入 FullGridS2，并推进 S1、TimeAUC、phase-mapped profiler、fused/streaming package。这里不是等 task 结束后再补，而是和 task candidate factory 同步记录 system metrics。

### 主线 D：robustness / scaling / geometry validation

目标是防止只在 MNIST-family small split 上过线。只有 teacher-free macro + S2 成立后，才打开 sample efficiency、noise robustness、hidden/batch scaling、geometry Pareto。geometry 不是硬门槛，而是 Pareto 维度。

---

## 3. 当前问题的本质分析

### 3.1 表达力不是空的，但 teacher-free margin 不够厚

dense D3、C3、M12 都说明 KAN family 可以超过 MLP。M13 说明 teacher-free 路线也已经有稳定正信号。现在的问题不是有没有表达力，而是 teacher-free official margin 不够厚。`+0.01784` 已经有统计显著性，但没有满足实践阈值 `+0.0200`。这意味着我们不应该把路线推倒，也不应该用 external teacher 取巧，而应研究剩余 `0.00216` 的来源。

### 3.2 external teacher 不是答案，但它是重要诊断信号

`M12+C3` 比 M13 高约 `+0.00273`。这说明 C3 teacher 提供的是 mild boost，不是全部优势。这个 boost 刚好跨过 gate，因此它是重要诊断线索。v8.0 要问：

```text
C3 teacher 给的到底是什么？
是 margin smoothing？
是 calibration？
是 early optimization trajectory？
是 hard sample reweighting？
是 representation rank？
```

但 C3 teacher 不能进入 official route。

### 3.3 v7.9 的代码契约进步是真的，但机制实验还没开始

v7.9 已经有 CandidateMeta、candidate_registry、contract_validator、teacher_leak_audit。这非常重要，因为它避免了把 external-teacher candidate 混成 official success。但 v7.9 仍然主要调用旧 runner 并做 postprocess。SB/RR/SYS 候选多为 registry 占位。v8.0 必须把这些候选变成真实训练路径。

### 3.4 core 里有现成 primitive，但没有形成候选工厂

`dgkan_core` 中已有 SparseInterp、SparseSpline、DWM2Lite、DWM2Dense、GEMMNativeDepthwiseMix、RationalKATV2 等候选基础。它们不是随便的备选项，而是过去几轮反复出现的核心冲突的结构回应：

```text
SparseInterp / SparseSpline:
  针对 dense basis materialization 和 live-set。

DWM2Lite:
  针对 MLP-like compute path + KAN residual 的折中。

GEMMNativeDepthwiseMix:
  针对 einsum / fragmented op path。

DWM2Dense:
  针对 pre/post mix + channel function 的表达力保留。

RationalKATV2:
  针对更紧凑的 edge function family。
```

v8.0 必须把这些 primitive 纳入真正的 candidate factory，而不是继续只围绕 M13/A2S/M9 做小修。

### 3.5 system metrics 不能再拖到最后

v7.9 主 run 中 M13 memory max 在 S2 内，但 step max 漂到 `1.6199`，导致 FullGridS2 不稳。之前 v7.8 M13 曾有 S2 pass，说明可能存在 timing drift / measurement path / validation logging interference。v8.0 需要 phase-clean full-grid profiler，把 model primitive step、task trace time、validation/logging/sync 分开。

---

## 4. 禁止事项

第一，不允许把 `M12+C3` 写成 official success。所有 external teacher candidate 都只能进入 diagnostic table。

第二，不允许把 test gap 用作 selection。M13 test gap 高于 `+0.0200` 只能触发 validation robustness audit，不能直接把 old run 改判成功。

第三，不允许继续在 postprocess 中“发明”候选。候选必须由 `CandidateSpec -> CandidateFactory -> measured artifacts` 产生。

第四，不允许把 self-bootstrap 和 external teacher 混为一谈。self-bootstrap 可以作为 no-external route，但必须单独标记为 `self_teacher_used=1`，不能伪装成 pure CE-only。

第五，不允许以 `not_implemented` 候选失败作为科学结论。`not_implemented` 只能说明代码缺口。

第六，不允许继续粗扫 hidden/basis/depth。除非 gap attribution 指向 capacity/rank，否则粗扫不是主线。

第七，不允许在没有 GradPass 的情况下报告 macro success。所有 official candidates 必须重新做 gradient correctness。

第八，不允许只看 mean S2。FullGridS2 必须覆盖：

```text
MNIST / Fashion-MNIST / KMNIST
batch = 128 / 256 / 512
```

即 9/9 shapes。

第九，不允许在未完成 phase-mapped profiler 时声称 kernel bottleneck 被定位。

第十，不允许把 TimeAUC failure 直接归因于模型。必须拆解 step AUC、train-only time、validation/logging/sync、自举 teacher overhead、kernel launch。

---

## 5. Hypothesis Map：v8.0 要一次性回答的核心假设

### H0：当前实验系统是否足以支撑 official claim？

H0 不是模型假设，而是科学流程假设。如果 candidate 不是由真实 CandidateFactory 构造，或者 teacher contract / strict contract / artifact join 不完整，那么即使数值过线，也不能做 official claim。

H0 成立标准：

```text
candidate_registry_v2 join coverage = 100%
candidate_factory_audit coverage = 100%
teacher_contract_v2 coverage = 100%
contract_validator_v2 coverage = 100%
fake_proxy_nonzero_count = 0
all official candidates external_teacher_used = 0
```

### H1：M13 的 autonomous gap 是否来自 validation estimator？

M13 的 validation gap 低于 test gap。不能用 test 改判，但可以用 pre-registered validation shards 检查当前 validation split 是否低估。

H1 成立标准：

$$
\Delta Acc_{\text{macro,val-shard-mean,M13}}\geq0.0200,
$$

$$
CI_{95\%,shard}^{low}>0.
$$

如果 repeated validation mean 仍小于 `+0.0180`，则不是 split issue。

### H2：C3 teacher 的 mild boost 是否可由 no-external self-bootstrap 复现？

C3 teacher boost 约：

$$
\Delta_{\text{teacher}} \approx 0.00273.
$$

H2 假设这个 boost 主要来自 soft target、margin smoothing 或 trajectory stabilization，而非外部不可替代知识。

H2 成立标准：

$$
Acc_{\text{self-bootstrap}} - Acc_{\text{M13}}\geq0.0020,
$$

并且：

$$
\Delta Acc_{\text{macro,val,self-bootstrap}}\geq0.0200.
$$

同时：

```text
external_teacher_used = 0
GradPass = 1
FullGridS2Pass = 1
```

### H3：teacher-free gap 是否来自 representation / feature rank？

如果 self-bootstrap 不能补足差距，则可能缺的是表示结构。H3 要比较 M13、M12+C3、A2S/M9 near-pass 之间的 feature CKA、effective rank、margin、hard-sample overlap。

H3 representation gap 成立标准之一：

$$
CKA(M13,M12)<0.90,
$$

或：

$$
rank_{\text{M13}}<0.95\cdot rank_{\text{M12}},
$$

或：

$$
margin_{p10,M13}<0.95\cdot margin_{p10,M12}.
$$

若 H3 成立，则进入 core primitive representation repair，不再做 schedule 小修。

### H4：core primitive 能否在 teacher-free 条件下提供新的 margin？

H4 假设 `SparseInterpKANDense`、`SparseSplineKANDense`、`DWM2LiteDense`、`GEMMNativeDepthwiseMixDense`、`DWM2Dense`、`RationalKATV2Dense` 中至少一类能提供 teacher-free margin，并保持 S2。

H4 成立标准：

$$
Acc_{\text{RR}}-Acc_{\text{M13}}\geq0.0022,
$$

$$
\Delta Acc_{\text{macro,val,RR}}\geq0.0200,
$$

$$
r_{\text{mem,max}}\leq1.05,
$$

$$
r_{\text{step,max}}\leq1.50.
$$

如果 task 增益来自明显更重的 compute/live-set，且 S2 失败，则该 primitive 只能算 representation diagnostic。

### H5：S2 step drift 是否来自 measurement path，而非模型本体？

M13 在不同 run 中 step gate 有漂移。H5 假设一部分 S2 failure 来自 task-trace timing、validation/logging/sync、warmup/reps 或 profiler path 不一致。

H5 成立标准：

phase-clean full-grid profiler 中：

$$
r_{\text{step,max}}\leq1.50,
$$

但 task trace 或 non-clean run 中：

$$
r_{\text{step,max}}>1.50.
$$

如果 H5 成立，route 应记录为 timing-path issue，而不是 model primitive fail。

### H6：TimeAUC fail 是否来自 runtime path 而非 learning dynamics？

M13 的 step AUC 比 MLP 好，但 time AUC 失败。H6 假设模型每步学习不差，问题来自 runtime path。

H6 成立标准：

$$
ValLossAUC_{\text{step,KAN}}\leq ValLossAUC_{\text{step,MLP}},
$$

但：

$$
ValLossAUC_{\text{time,KAN}}>ValLossAUC_{\text{time,MLP}}.
$$

这时优先做 time accounting、profiler、kernel/fused path；不做 optimizer sweep。

### H7：teacher-free macro pass 后是否能系统化进入 S1 / TimeAUC？

H7 是系统合流假设。只要 candidate 过 teacher-free macro + S2，v8.0 要立即进入 full-grid S1 / TimeAUC / ProfilerPass，而不是再等下一版。

H7 成立标准：

$$
\text{TeacherFreeMacroPass}
\land
\text{FullGridS2Pass}
\rightarrow
\text{P8-P13 system closure opened}.
$$

---

## 6. Candidate 设计：从零散候选变成候选工厂

### 6.1 Baselines

```text
B0-MLP-AdamW
B2-MLP-C3-distill-diagnostic
M12-C3-distill-diagnostic
M13-teacher-free-official-baseline
A2S/M9-near-pass-structural-baseline
```

### 6.2 Pure CE teacher-free family

```text
TF0-M13-baseline
TF1-M13-validation-robustness-repro
TF2-M13-M5init-stabilized-no-teacher
TF3-M13-margin-stabilized-CE
TF4-M13-NLL-balanced-CE
TF5-M13-calibration-aware-CE
TF6-M13-warmup-stability-no-external
TF7-M13-SWA-final-averaging
TF8-M13-EMA-weights-no-logit-teacher
```

这里 TF7/TF8 不是 self-teacher，它们只是权重平均或 EMA weights，不使用 logits teacher。

### 6.3 No-external self-bootstrap family

```text
SB0-M13-EMA-self-teacher
SB1-M13-delayed-self-distill-40steps
SB2-M13-snapshot-self-distill-bestval
SB3-M13-dual-view-consistency
SB4-M13-SWA-logit-smoothing
SB5-M13-stopgrad-previous-epoch-teacher
SB6-M13-self-distill-temperature2
SB7-M13-self-distill-temperature4
```

这些 candidate 必须满足：

```text
external_teacher_used = 0
self_teacher_used = 1
self_teacher_source in {EMA, previous_checkpoint, same_run_snapshot, SWA}
```

### 6.4 Code-native representation family

```text
RR0-M13-current
RR1-M13-SparseInterpKANDense
RR2-M13-SparseSplineKANDense
RR3-M13-DWM2LiteDense-rbf-residual
RR4-M13-DWM2LiteDense-lut-residual
RR5-M13-GEMMNativeDepthwiseMixDense
RR6-M13-DWM2Dense-residual-lite
RR7-M13-RationalKATV2Dense-identity-residual
RR8-M13-feature-rank-preserving-init
RR9-M13-local-cross-feature-bridge-lite
RR10-A2S/M9-packed-generic-head-teacher-free-final
```

每个 RR candidate 必须先通过 implementation audit：

```text
dense_cls importable
forward works
manual backward or analytic grad checker available
nonKAN_param_count = 0
edge_param_count measured
kernel_path recorded
external_teacher_used = 0
```

### 6.5 System family

只有 teacher-free macro candidate 过线后，才进入 official system repair：

```text
SYS0-best-teacher-free-current
SYS1-phase-clean-timing
SYS2-root-input-lifetime-trim
SYS3-hidden-y-lifetime-trim
SYS4-streaming-root-input-cache
SYS5-streaming-hidden-y-cache
SYS6-fused-head-forward
SYS7-fused-linear-silu-backward
SYS8-dense-preserving-S1-combo
SYS9-time-accounting-clean-loop
```

---

## 7. 实验总流程

v8.0 不再按单线版本迭代，而按 Wave 执行。每个 Wave 都有并行分支和合流点。

```text
Wave 0:
  code-native infrastructure

Wave 1:
  evidence map and gap attribution

Wave 2:
  autonomous margin closure

Wave 3:
  official co-selection

Wave 4:
  system closure

Wave 5:
  robustness / scaling / geometry
```

下面是完整阶段。

---

## 8. P0：Code-native infrastructure

### 8.1 目标

P0 要把 v7.9 的 metadata-first audit 升级为 code-native experiment system。所有候选必须由 `CandidateSpec` 构造，不允许只 postprocess 旧结果。

### 8.2 必须实现

```text
CandidateSpecV2
CandidateFactory
TeacherContractV2
ContractValidatorV2
ArtifactJoinValidator
NoExternalRouteFilter
ImplementationStatusRegistry
```

`CandidateSpecV2` 至少包含：

```text
candidate_id
candidate_name
route_role
model_family
dense_cls_name
dense_cls_module_path
hidden_dim
basis_count
depth
init_policy
optimizer_policy
external_teacher_used
external_teacher_source
external_teacher_logits_used
external_teacher_forward_used
self_teacher_used
self_teacher_type
self_teacher_logits_used
self_teacher_forward_used
teacher_checkpoint_path
teacher_artifact_hash
no_external_teacher_eligible
pure_supervised_eligible
official_eligible
expected_manual_forward
expected_manual_backward
expected_manual_update
expected_nonkan_count
```

### 8.3 记录指标

```text
candidate_registry_v2.csv
candidate_factory_audit.csv
teacher_contract_v2.csv
contract_validator_v2.csv
artifact_join_coverage.csv
implementation_status_registry.csv
```

### 8.4 成立标准

CandidateFactoryPass：

```text
all measured candidates constructed_from_spec = 1
all measured rows join candidate_registry_v2 by candidate_id
no unknown candidate_id
route decision never infers teacher status from raw candidate_id
```

TeacherContractV2Pass：

```text
external teacher candidates official_eligible = 0
self-bootstrap candidates external_teacher_used = 0
self-bootstrap candidates no_external_teacher_eligible = 1
external_teacher_logits_used and self_teacher_logits_used are separate fields
```

StrictPass 定义：

$$
\text{StrictPass}
=
[\text{nonKAN}=0]
\land
[\text{head\_is\_kan}=1]
\land
[\text{manual\_forward}=1]
\land
[\text{manual\_backward}=1]
\land
[\text{manual\_update}=1]
\land
[\text{uses\_loss\_backward}=0].
$$

### 8.5 可视化

```text
p0_candidate_factory_graph.svg
p0_teacher_contract_matrix.svg
p0_strict_contract_heatmap.svg
p0_artifact_join_coverage.svg
```

---

## 9. P1：Reproduction and evidence map

### 9.1 目标

用 code-native runner 复现 B0、B2、M12、M13、A2S/M9。P1 的作用不是寻找新结果，而是确认重构后的 runner 没有改变已知事实。

### 9.2 必跑对象

```text
B0
B2
M12
M13
A2S
M9
```

### 9.3 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
train/val/test = 1536/512/512
steps = 240
bench batch = 128,256,512
grad batch = 8,128
bootstrap reps = 10000
```

### 9.4 记录指标

```text
val_acc
test_acc
val_gap_vs_MLP
test_gap_vs_MLP
ECE
NLL
CI95_low
Holm_p
GradPass
grad_relerr_max
memory_ratio_max
step_ratio_max
S2_shape_count
S1_shape_count
ValLossAUC_step
ValLossAUC_time
```

### 9.5 成立标准

M13 reproduction：

$$
|\Delta Acc_{\text{M13}}-0.01784|\leq0.004.
$$

M12 diagnostic reproduction：

$$
|\Delta Acc_{\text{M12+C3}}-0.02057|\leq0.004.
$$

A2S/M9 reproduction：

$$
|\Delta Acc_{\text{A2S/M9}}-0.01992|\leq0.004.
$$

### 9.6 可视化

```text
p1_reproduction_gap_bar.svg
p1_seedwise_gap_boxplot.svg
p1_efficiency_heatmap.svg
p1_m13_m12_a2s_m9_delta.svg
```

---

## 10. P2：Validation robustness audit

### 10.1 目标

判断 M13 / A2S/M9 的 near-pass 是否被 single validation split 低估。test 不能用于调参，但 test gap 高于 val gap时，必须审计 validation estimator。

### 10.2 实验设计

保持 test 完全不参与选择。从 train/val pool 中构造预注册 validation shards：

```text
shards = 5 or 10
balanced by class
fixed shard seeds
same train size budget
no test-based selection
```

### 10.3 必跑对象

```text
B0
M13
A2S
M9
M12 diagnostic only
```

### 10.4 记录指标

```text
candidate
split_id
seed
val_acc
val_loss
val_gap_vs_MLP
ECE
NLL
class_distribution
sample_count
macro_gap_split_mean
macro_gap_split_std
CI95_low_split
Holm_p_split
```

### 10.5 判断标准

Validation split effect：

$$
\Delta Acc_{\text{macro,val-shard-mean}}\geq0.0200,
$$

$$
CI_{95\%,split}^{low}>0.
$$

如果 repeated validation mean 仍低于：

$$
0.0180,
$$

则不是 split issue。

### 10.6 可视化

```text
p2_val_gap_by_split.svg
p2_split_distribution_ci.svg
p2_val_vs_test_gap_comparison.svg
p2_class_distribution_by_split.svg
```

---

## 11. P3：Autonomous gap attribution

### 11.1 目标

解释 `M13 -> M12+C3` 的 `+0.00273`，以及 `M13 -> A2S/M9` 的 near-pass 差异。P3 不修模型，只定位缺口来源。

### 11.2 必跑对象

```text
B0
B2
M13
M12+C3 diagnostic
A2S
M9
```

### 11.3 记录指标

```text
margin_mean
margin_p10
margin_p50
confidence_mean
wrong_confidence_mean
logit_norm_mean
feature_effective_rank
feature_CKA_M13_M12
feature_CKA_M13_A2S
feature_CKA_M13_M9
logit_KL_M12_M13
error_overlap_rate
hard_sample_gain
early_loss_slope
mid_loss_slope
late_loss_slope
class_pair_confusion_matrix
dataset_gap_contribution
```

### 11.4 判断标准

Calibration gap：

```text
error_overlap_rate >= 0.80
feature_CKA >= 0.95
NLL gap explains most delta
```

Representation gap：

$$
feature\_CKA<0.90
$$

or:

$$
rank_{M13}<0.95rank_{M12}.
$$

Margin gap：

$$
margin_{p10,M13}<0.95margin_{p10,M12}.
$$

Optimization timing gap：

```text
early_loss_slope_M13 worse
mid/late slope similar
```

### 11.5 可视化

```text
p3_feature_cka_heatmap.svg
p3_logit_kl_distribution.svg
p3_margin_distribution.svg
p3_error_overlap_venn.svg
p3_loss_slope_comparison.svg
p3_teacher_gap_decomposition_dashboard.svg
```

---

## 12. P4：No-external self-bootstrap

### 12.1 目标

用 no-external self-bootstrap 复现 external C3 teacher 的 mild boost。P4 是最重要的 teacher-free margin closure 主线之一。

### 12.2 候选

```text
SB0-M13-EMA-self-teacher
SB1-M13-delayed-self-distill-40steps
SB2-M13-snapshot-self-distill-bestval
SB3-M13-dual-view-consistency
SB4-M13-SWA-logit-smoothing
SB5-M13-stopgrad-previous-epoch-teacher
SB6-M13-self-distill-temperature2
SB7-M13-self-distill-temperature4
```

### 12.3 训练规则

self teacher 只能来自同一训练过程或同一架构自身：

```text
EMA weights
previous checkpoint
delayed snapshot
SWA logits
dual stochastic views
```

不得使用：

```text
C3
M12+C3
external checkpoint
test-selected checkpoint
```

### 12.4 记录指标

```text
self_teacher_type
temperature
alpha
val_acc
test_acc
val_gap_vs_MLP
macro_delta_vs_M13
ECE
NLL
ValLossAUC_step
ValLossAUC_time
memory_ratio_max
step_ratio_max
GradPass
grad_relerr_max
self_teacher_time_overhead
```

### 12.5 判断标准

Self-bootstrap useful：

$$
Acc_{\text{SB}}-Acc_{\text{M13}}\geq0.0020.
$$

No-external official pass：

$$
\Delta Acc_{\text{macro,val,SB}}\geq0.0200,
$$

$$
CI_{95,\text{macro}}^{low}>0,
$$

$$
p_{\text{Holm}}<0.05.
$$

Time overhead constraint：

$$
ValLossAUC_{\text{time,SB}}\leq1.10\cdot ValLossAUC_{\text{time,M13}}.
$$

### 12.6 可视化

```text
p4_self_bootstrap_gain.svg
p4_self_bootstrap_time_overhead.svg
p4_self_bootstrap_vs_c3_teacher.svg
p4_self_bootstrap_learning_curve.svg
```

---

## 13. P5：Targeted supervised teacher-free repair

### 13.1 目标

如果 P3 指向 calibration、margin 或 early dynamics，而不是 representation gap，则执行 supervised-only repair。这里不做泛化 optimizer sweep，只做机制导向修复。

### 13.2 候选

```text
TF2-M13-M5init-stabilized-no-teacher
TF3-M13-margin-stabilized-CE
TF4-M13-NLL-balanced-CE
TF5-M13-calibration-aware-CE
TF6-M13-warmup-stability-no-external
TF7-M13-SWA-final-averaging
TF8-M13-EMA-weights-no-logit-teacher
```

### 13.3 记录指标

```text
repair_type
external_teacher_used
self_teacher_used
val_acc
test_acc
val_gap_vs_MLP
test_gap_vs_MLP
ECE
NLL
ValLossAUC_step
ValLossAUC_time
grad_relerr_max
memory_ratio_max
step_ratio_max
macro_delta_vs_M13
margin_p10_delta
NLL_delta_vs_M13
```

### 13.4 判断标准

Repair useful：

$$
Acc_{\text{repair}}-Acc_{\text{M13}}\geq0.0022.
$$

Official macro pass：

$$
\Delta Acc_{\text{macro,val,repair}}\geq0.0200.
$$

Efficiency preservation：

$$
r_{\text{mem,max}}\leq1.05,
$$

$$
r_{\text{step,max}}\leq1.50.
$$

Grad preservation：

$$
grad\_relerr_{max}\leq10^{-4}.
$$

### 13.5 可视化

```text
p5_repair_gain_bar.svg
p5_task_efficiency_pareto.svg
p5_repair_auc_step_time.svg
p5_repair_nll_ece_bar.svg
p5_margin_repair_dashboard.svg
```

---

## 14. P6：Core primitive representation repair

### 14.1 目标

如果 P3 指向 representation gap，则把 `dgkan_core` 中已有的 low-live-set primitives 变成真实 teacher-free candidates。

### 14.2 候选

```text
RR0-M13-current
RR1-M13-SparseInterpKANDense
RR2-M13-SparseSplineKANDense
RR3-M13-DWM2LiteDense-rbf-residual
RR4-M13-DWM2LiteDense-lut-residual
RR5-M13-GEMMNativeDepthwiseMixDense
RR6-M13-DWM2Dense-residual-lite
RR7-M13-RationalKATV2Dense-identity-residual
RR8-M13-feature-rank-preserving-init
RR9-M13-local-cross-feature-bridge-lite
RR10-A2S/M9-packed-generic-head-teacher-free-final
```

### 14.3 Implementation audit

每个 RR candidate 必须先通过：

```text
dense_cls importable
forward works on all primary datasets
manual_backward or analytic gradient checker available
nonKAN_param_count = 0
edge_param_count measured
kernel_path recorded
external_teacher_used = 0
```

若 manual backward 不支持，candidate 只能进入 representation diagnostic，不能进入 official route。

### 14.4 记录指标

```text
dense_cls
structural_change
external_teacher_used
nonKAN_param_count
edge_param_delta
feature_rank
margin_p10
val_gap_vs_MLP
test_gap_vs_MLP
ECE
NLL
GradPass
grad_relerr_max
memory_ratio_max
step_ratio_max
S2_shape_count
```

### 14.5 判断标准

Representation repair useful：

$$
Acc_{\text{RR}}-Acc_{\text{M13}}\geq0.0022.
$$

Feature repair：

$$
rank_{\text{RR}}\geq1.05rank_{\text{M13}},
$$

或：

$$
margin_{p10,RR}\geq1.05margin_{p10,M13}.
$$

Official pass：

$$
\Delta Acc_{\text{macro,val,RR}}\geq0.0200,
$$

$$
r_{\text{mem,max}}\leq1.05,
$$

$$
r_{\text{step,max}}\leq1.50.
$$

### 14.6 可视化

```text
p6_representation_repair_gain.svg
p6_feature_rank_vs_acc.svg
p6_margin_vs_acc.svg
p6_structural_overhead_pareto.svg
p6_dense_cls_comparison_heatmap.svg
```

---

## 15. P7：10-seed official co-selection

### 15.1 目标

把 P4/P5/P6 的 best candidates 放到同一个 official 10-seed co-selection 中，不再以单分支 near-pass 下结论。

### 15.2 必跑对象

```text
B0
M13
best-SB
best-TF-repair
best-RR
A2S/M9 near-pass
M12+C3 diagnostic
B2 diagnostic
```

### 15.3 记录指标

```text
candidate
route_role
external_teacher_used
self_teacher_used
dataset
seed
val_acc
test_acc
val_gap_vs_MLP
test_gap_vs_MLP
CI95_low
Holm_p
seed_win_rate
ECE
NLL
ValLossAUC_step
ValLossAUC_time
memory_ratio_max
step_ratio_max
S2_shape_count
S1_shape_count
```

### 15.4 判断标准

Teacher-free official pass：

```text
external_teacher_used = 0
```

and:

$$
\Delta Acc_{\text{macro,val}}\geq0.0200,
$$

$$
CI_{95,\text{macro}}^{low}>0,
$$

$$
p_{\text{Holm}}<0.05,
$$

$$
\Delta Acc_{\text{macro,test}}\geq0.015.
$$

FullGridS2Pass：

$$
r_{\text{mem,max}}\leq1.05,
$$

$$
r_{\text{step,max}}\leq1.50.
$$

### 15.5 可视化

```text
p7_teacher_free_bootstrap_ci.svg
p7_seedwise_gap_boxplot.svg
p7_official_candidate_scorecard.svg
p7_task_efficiency_pareto.svg
```

---

## 16. P8：Phase-clean full-grid S2 profiler

### 16.1 目标

确认 candidate 的 S2 是否稳定，并拆分 task-trace step drift 与 primitive step。

### 16.2 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
batch sizes = 128,256,512
warmup = 50
reps = 200
profile mode = train_step_only
validation/logging disabled
no extra sync except measurement sync
```

### 16.3 记录指标

```text
memory_ratio
step_ratio
forward_ratio
backward_ratio
update_ratio
train_step_only_ms
validation_ms
logging_ms
cuda_sync_ms
S2_pass
S1_pass
```

### 16.4 判断标准

FullGridS2Pass：

```text
9/9 shapes pass
```

where:

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

Timing instability：

```text
phase-clean step max <= 1.50
but task-trace step max > 1.50
```

### 16.5 可视化

```text
p8_fullgrid_memory_heatmap.svg
p8_fullgrid_step_heatmap.svg
p8_task_trace_vs_phase_clean_step.svg
p8_step_instability_by_shape.svg
```

---

## 17. P9：Teacher-free time accounting

### 17.1 目标

解释为什么 step AUC 可以好于 MLP，但 wall-clock AUC 不好。P9 在 teacher-free candidate 上执行，避免 external teacher 干扰。

### 17.2 记录指标

```text
wall_clock_time_total
train_step_time
forward_time
backward_time
update_time
self_teacher_time
validation_time
logging_time
cuda_sync_time
data_loading_time
unknown_time_fraction
ValLossAUC_step
ValLossAUC_time_train_only
ValLossAUC_time_total
time_to_target_acc
```

### 17.3 判断标准

TimeAccountingPass：

```text
unknown_time_fraction <= 0.10
validation/logging/sync separated
self-teacher overhead separated if any
```

Teacher-free TimeAUCPass：

$$
ValLossAUC_{\text{time,total,TF}}\leq ValLossAUC_{\text{time,B0}}.
$$

如果：

$$
ValLossAUC_{\text{step,TF}}\leq ValLossAUC_{\text{step,B0}},
$$

但：

$$
ValLossAUC_{\text{time,TF}}>ValLossAUC_{\text{time,B0}},
$$

则进入 profiler/fused path，而不是 optimizer sweep。

### 17.4 可视化

```text
p9_teacher_free_time_breakdown.svg
p9_total_vs_train_only_auc.svg
p9_time_to_target.svg
p9_step_auc_vs_time_auc.svg
```

---

## 18. P10：Phase-mapped kernel profiler

### 18.1 目标

建立真实 kernel attribution。不能再用 op-count proxy 替代 kernel launch count。

### 18.2 必跑对象

```text
B0
M13
best-teacher-free
best-self-bootstrap
best-RR
M12+C3 diagnostic
```

### 18.3 记录指标

```text
kernel_count_total
mapped_kernel_time_fraction
unknown_kernel_time_fraction
kernel_count_forward
kernel_count_backward
kernel_count_update
kernel_count_validation
top_kernel_name_1
top_kernel_phase_1
top_kernel_time_1
small_kernel_count_under_10us
layout_conversion_count
cuda_memcpy_time
cuda_sync_time
```

### 18.4 判断标准

ProfilerPass：

```text
mapped_kernel_time_fraction >= 0.90
unknown_kernel_time_fraction <= 0.10
top3 phase time explain >= 0.70
```

Kernel fragmentation：

$$
N_{\text{small kernels under 10us}}\geq0.30N_{\text{kernel total}}.
$$

### 18.5 可视化

```text
p10_kernel_timeline.svg
p10_kernel_count_by_phase.svg
p10_top_kernel_time_bar.svg
p10_mapped_vs_unknown_kernel_time.svg
```

---

## 19. P11：S1 memory attribution

### 19.1 目标

在 teacher-free macro + S2 candidate 上定位 S1 memory gap，不能在 teacher-assisted candidate 上替代。

### 19.2 记录指标

```text
batch_size
memory_ratio
step_ratio
root_input_cache_MB
hidden_y_cache_MB
manual_cache_MB
optimizer_state_MB
head_temp_MB
allocator_padding_MB
top1_memory_source
top2_memory_source
top3_memory_source
unknown_memory_fraction
```

### 19.3 判断标准

S1 attribution pass：

```text
top3 memory sources identified
unknown_memory_fraction <= 0.10
```

Actionable source：

$$
\frac{M_{\text{source}}}{M_{\text{peak}}}\geq0.02.
$$

### 19.4 可视化

```text
p11_s1_memory_waterfall.svg
p11_bs512_memory_source.svg
p11_memory_ratio_by_batch.svg
```

---

## 20. P12：Dense-preserving fused / streaming package

### 20.1 目标

在不破坏 teacher-free macro signal 的条件下修 S1 / TimeAUC / forward bottleneck。

### 20.2 候选

```text
SYS0-best-teacher-free-current
SYS1-root-input-lifetime-trim
SYS2-hidden-y-lifetime-trim
SYS3-streaming-root-input-cache
SYS4-streaming-hidden-y-cache
SYS5-fused-head-forward
SYS6-fused-linear-silu-backward
SYS7-phase-mapped-fused-forward
SYS8-dense-preserving-S1-combo
SYS9-time-accounting-clean-loop
```

### 20.3 记录指标

```text
components
grad_relerr_max
grad_cos_min
macro_val_gap
test_gap
ECE
NLL
ValLossAUC_step
ValLossAUC_time
memory_ratio_max
step_ratio_max
S1_shape_count
S2_shape_count
kernel_count_total
mapped_kernel_time_fraction
materialized_tensor_count
macro_delta_vs_parent
```

### 20.4 判断标准

Task preservation：

$$
\Delta Acc_{\text{macro,val}}\geq0.0200,
$$

and:

$$
\Delta Acc_{\text{new}}\geq \Delta Acc_{\text{parent}}-0.003.
$$

FullGridS1：

$$
r_{\text{mem,max}}<1.00,
$$

$$
r_{\text{step,max}}\leq1.35.
$$

TimeAUCPass：

$$
ValLossAUC_{\text{time,new}}\leq ValLossAUC_{\text{time,B0}}.
$$

### 20.5 可视化

```text
p12_package_task_efficiency_pareto.svg
p12_s1_s2_shape_heatmap.svg
p12_time_auc_repair_bar.svg
p12_kernel_mapping_after_repair.svg
```

---

## 21. P13：Final task + system confirmation

### 21.1 目标

所有最终候选必须重新进行 10-seed task + system confirmation。

### 21.2 必跑对象

```text
B0
M13
best-teacher-free
best-no-external-self-bootstrap
best-RR
best-system-repaired
M12+C3 diagnostic
```

### 21.3 记录指标

```text
route_role
external_teacher_used
self_teacher_used
val_gap_vs_MLP
test_gap_vs_MLP
CI95_low
Holm_p
ECE
NLL
GradPass
FullGridS2Pass
FullGridS1Pass
TimeAUCPass
ProfilerPass
memory_ratio_max
step_ratio_max
ValLossAUC_time
```

### 21.4 判断标准

Minimum official success：

$$
\text{CodeNativePass}
\land
\text{TeacherFreeMacroSignificantPass}
\land
\text{GradPass}
\land
\text{FullGridS2Pass}.
$$

Formal success：

$$
\text{MinimumSuccess}
\land
\text{FullGridS1Pass}
\land
\text{TimeAUCPass}
\land
\text{ProfilerPass}.
$$

### 21.5 可视化

```text
p13_final_scorecard.svg
p13_system_pareto.svg
p13_route_tree.svg
p13_task_system_table.md
```

---

## 22. P14：Scaling / robustness / geometry

### 22.1 打开条件

Only open if：

```text
TeacherFreeMacroSignificantPass = true
FullGridS2Pass = true
GradPass = true
```

### 22.2 设置

```text
train_size = 256,512,1024,1536,4096
label_noise = 0.05,0.10,0.20
input_noise = 0.05,0.10
hidden_dim = 48,64,96,128
```

### 22.3 记录指标

```text
train_size
noise_type
hidden_dim
val_acc
test_acc
ECE
NLL
accuracy_drop
sample_efficiency_auc
robustness_auc
memory_ratio
step_ratio
geometry_metric_value
```

### 22.4 判断标准

Sample efficiency pass：

$$
AUC_{\text{data,KAN}}>AUC_{\text{data,MLP}}.
$$

Robustness pass：

$$
AccDrop_{\text{KAN}}<AccDrop_{\text{MLP}}
$$

for at least two settings.

Geometry useful if：

$$
Acc_{\lambda}\geq Acc_{\lambda=0}-0.005,
$$

and one of：

$$
ECE_{\lambda}<ECE_{\lambda=0},
$$

$$
NLL_{\lambda}<NLL_{\lambda=0},
$$

$$
ValLossAUC_{\lambda}<ValLossAUC_{\lambda=0}.
$$

### 22.5 可视化

```text
p14_accuracy_vs_train_size.svg
p14_noise_robustness.svg
p14_geometry_pareto_frontier.svg
```

---

## 23. Route Decision

### 23.1 Survivor types

```text
S0:
  CodeNative + TeacherFreeMacroSignificant + FullGridS1 + TimeAUCPass + ProfilerPass

S1:
  CodeNative + TeacherFreeMacroSignificant + FullGridS2 + TimeAUCPass

S2:
  CodeNative + TeacherFreeMacroSignificant + FullGridS2 but TimeAUC fail

S3:
  No-external self-bootstrap success

S4:
  Teacher-free near-pass only, external teacher pass

S5:
  External teacher only success

S6:
  Teacher-free macro pass but S2 fail

S7:
  GradFail

S8:
  Profiler/time incomplete

S9:
  Code contract incomplete
```

### 23.2 Route cases

```text
R1-TeacherFreeFullSystemAdvantage:
  S0. Official PureKAN-NG success.

R2-TeacherFreeS2QualityAdvantage:
  S2. Teacher-free quality/S2 success, system time still open.

R3-NoExternalSelfBootstrapSuccess:
  Self-bootstrap passes; report as no-external-teacher route.

R4-TeacherAssistedOnly:
  Only C3 teacher passes; official standalone goal not met.

R5-TeacherFreeNearPass:
  Teacher-free remains around +0.018 to +0.020; continue representation/initialization.

R6-KernelizationNeeded:
  Teacher-free macro pass but S1/time fail due to kernel/memory.

R7-NoReproduction:
  Teacher-free signal does not reproduce.

R8-CodeContractFail:
  Candidate metadata or strict contract incomplete; no scientific claim allowed.
```

### 23.3 必须记录字段

```text
candidate
route
external_teacher_used
self_teacher_used
strict_pass
grad_pass
teacher_free_macro_pass
fullgrid_s2_pass
fullgrid_s1_pass
time_auc_pass
profiler_pass
code_contract_pass
val_gap_vs_MLP
test_gap_vs_MLP
memory_ratio_max
step_ratio_max
primary_blocker
next_required_implementation
```

---

## 24. Artifact 要求

v8.0 必须落盘以下文件：

```text
run_manifest.json
candidate_registry_v2.csv
candidate_factory_audit.csv
teacher_contract_v2.csv
contract_validator_v2.csv
teacher_leak_audit_v2.csv
artifact_join_coverage.csv
p1_reproduction.csv
p2_validation_split_audit.csv
p3_gap_attribution.csv
p4_self_bootstrap.csv
p5_supervised_repair.csv
p6_core_primitive_repair.csv
p7_teacher_free_10seed.csv
p8_phase_clean_fullgrid.csv
p9_time_accounting.csv
p10_phase_mapped_profiler.csv
p11_s1_memory_attribution.csv
p12_fused_streaming_package.csv
p13_final_confirmation.csv
p14_scaling_robustness_geometry.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

failure taxonomy：

```text
F1_code_contract_fail
F2_candidate_factory_fail
F3_teacher_contract_fail
F4_teacher_free_macro_fail
F5_validation_split_underpowered
F6_self_bootstrap_fail
F7_representation_repair_fail
F8_external_teacher_only_success
F9_s2_fail
F10_s1_fail
F11_time_auc_fail
F12_grad_fail
F13_profiler_incomplete
F14_fake_or_proxy_violation
F15_artifact_missing
```

---

## 25. 第一轮执行顺序

第一轮不是从模型开始，而是从实验系统开始。

### Step 1：P0 Code-native runner

先实现 `CandidateFactory` 和 `TeacherContractV2`。没有这个，不做新 scientific claim。

### Step 2：P1 Reproduction

用新 runner 复现 M13、M12+C3、A2S/M9。确认 code-native runner 没改坏结果。

### Step 3：P2 Validation robustness

判断 M13 / A2S/M9 near-pass 是否是 validation split issue。

### Step 4：P3 Gap attribution

拆 `+0.00273` teacher boost 和 `+0.000078` A2S/M9 near-miss 到底来自哪里。

### Step 5：P4 Self-bootstrap

这是最重要的 no-external repair。它和 C3 teacher 的作用机制最接近，但不使用外部 teacher。

### Step 6：P6 Core primitive repair

如果 P3 指向 representation/rank/margin，立即把 SparseInterp、SparseSpline、DWM2Lite、GEMMNativeDepthwiseMix 等接入 CandidateFactory。

### Step 7：P7 10-seed co-selection

所有 near-pass candidates 统一 10-seed co-selection。`+0.0199` 不能算成功。

### Step 8：P8-P13 system closure

只有 teacher-free macro pass 后，才进入 official formal system route。

---

## 26. 停止条件

成功停止：

```text
CodeNativePass + TeacherFreeMacroSignificant + FullGridS2 + GradPass
CodeNativePass + TeacherFreeMacroSignificant + FullGridS1 + GradPass
CodeNativePass + TeacherFreeMacroSignificant + FullGridS1 + TimeAUCPass + ProfilerPass
```

失败停止：

```text
1. CandidateFactory 无法覆盖 measured candidates；
2. TeacherContractV2 不能区分 external teacher 与 self teacher；
3. M13 reproduction gap < +0.015；
4. repeated validation mean gap < +0.018；
5. all self-bootstrap repairs improve < +0.001；
6. all core primitive repairs improve < +0.001；
7. any official candidate uses external teacher；
8. S1 repair makes teacher-free macro gap < +0.0200；
9. GradPass fails and cannot be fixed by numerical equivalent rewrite；
10. no-fake/no-proxy audit fail。
```

---

## 27. 最终判断规则

### Case A：pure supervised teacher-free candidate crosses +0.020 and S2

这是最强结果。可写：

```text
PureKAN-NG teacher-free CE-trained official quality advantage is established.
```

### Case B：self-bootstrap crosses +0.020 and S2

可写：

```text
No-external-teacher autonomous self-bootstrap route succeeds.
```

但必须说明它不是 pure CE-only，而是 no-external training recipe。

### Case C：validation shard mean crosses +0.020 but old validation does not

说明 single validation split 可能低估 M13。必须预注册更稳健 validation protocol。不能 retroactively claim old run success。

### Case D：only external C3 teacher crosses +0.020

不能写 official success。只能写：

```text
PureKAN-NG is currently teacher-assisted only.
```

### Case E：teacher-free crosses macro but S2 fails

说明 autonomous quality advantage 成立，但 system candidate 尚未成立。进入 kernel/time/memory route。

### Case F：teacher-free remains around +0.0199

说明项目极接近但未科学闭合。下一步应基于 attribution 做 representation-level design，而不是 threshold chasing。

### Case G：code contract fails

不允许做 scientific claim。先修 runner。

---

## 28. 最终建议

v8.0 的一句话策略是：

$$
\boxed{
\text{从“版本补丁”升级为“code-native experimental platform + autonomous margin closure”。}
}
$$

现在的最短路线不是再调一个小参数，而是：

```text
1. 建真正 CandidateFactory；
2. 分离 external teacher 与 self teacher；
3. 做 validation robustness；
4. 做 M13/M12/A2S/M9 gap attribution；
5. 实现 no-external self-bootstrap；
6. 接入 core primitive representation repair；
7. 统一 10-seed co-selection；
8. 过线后一次性进入 S2/S1/TimeAUC/profiler system closure。
```

只有这样，才能把现在的：

$$
+0.01784 \text{ teacher-free near-pass}
$$

或：

$$
+0.019921875 \text{ structural near-pass}
$$

真正推进到：

$$
\geq+0.0200 \text{ teacher-free official success}.
$$

最终目标不变：

$$
\boxed{
\text{PureKAN-NG 自身，而不是 teacher-assisted student，系统性超过 MLP-AdamW。}
}
