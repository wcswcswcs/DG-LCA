# DG-KAN v7.7：M12 从 S2-Fair Minimum Success 到 S1/Time-AUC/System Advantage 的机制闭环实验计划

> 本计划基于 v7.6 clean 10-seed confirmation run 与后续 S1/cache repair 结果制定。当前结论已经不是“有没有 PureKAN-NG candidate”，而是：**M12 已经达到 v7.6 minimum success，但尚未达到 formal system success**。  
> v7.7 不做 loss、sampler、classwise、focal、target-margin 或普通 optimizer 小修。v7.7 只围绕三个本质问题展开：  
> 1. M12 的质量优势能否转化为 wall-clock / time-AUC 优势；  
> 2. M12 的 S2 能否升级为 S1，且不破坏 macro advantage 和 GradPass；  
> 3. M12 的剩余开销到底来自真实 kernel launch / op fragmentation / cache live-set / teacher objective，还是来自 KAN 数学本身。

---

## 0. 当前实验数据结论

### 0.1 v7.6 minimum success 已经达成

clean 10-seed confirmation run 中，M12 同时满足：

```text
StrictPass = true
GradPass = true
MacroSignificantPass_10seed = true
FairnessPass_vs_MLP_distill = true
FullGridS2Pass = true
success_v76_minimum = true
```

M12 相对 MLP-AdamW 的 task 结果为：

```text
B0 / MLP-AdamW:
  val acc  = 0.8449
  test acc = 0.7958
  ECE      = 0.0619
  NLL      = 0.5745

B2 / MLP + C3 logit distill:
  val acc  = 0.8503
  test acc = 0.8001
  val gap vs MLP = +0.0054
  ECE      = 0.0450
  NLL      = 0.4981

M12 / PureKAN-NG + C3 logit distill:
  val acc  = 0.8655
  test acc = 0.8154
  val gap vs MLP = +0.02057
  CI95 low = +0.01491
  Holm p   = 1.35e-06
  test gap = +0.01966
  ECE delta = -0.00415
  NLL delta = -0.10059
```

M12 相比同 teacher objective 的 MLP-distill 仍然更好：

```text
M12 gap vs B2 = +0.01517
M12 NLL delta vs B2 = -0.02413
```

这说明 C3 teacher distillation 本身确实提升 MLP，但提升不足以解释 M12 的优势。因此在当前 task family 上，M12 的优势不是 distillation-only。

M12 的 gradient correctness：

```text
GradPass = 6/6
max relerr = 8.07e-05
grad cos min = 0.999998
```

M12 的 full-grid S2：

```text
memory ratio mean = 1.0143
memory ratio max  = 1.0381
step ratio mean   = 1.1683
step ratio max    = 1.2591
S2 shapes = 9/9
```

因此当前可以写：

$$
\boxed{
\text{M12 已经达到 v7.6 minimum success。}
}
$$

---

### 0.2 但 v7.6 formal system success 未达成

v7.6 formal success 需要 FullGridS1Pass 和 TimeAUCPass，而当前二者均未达成：

```text
FullGridS1Pass = false
TimeAUCPass = false
success_v76_formal = false
```

S1 失败来自 memory，而不是 step：

```text
M12 S1 shapes = 3/9

bs128 memory ratio = 0.9962  -> S1 pass
bs256 memory ratio = 1.0086  -> S1 fail
bs512 memory ratio = 1.0381  -> S1 fail

step ratio max = 1.2591 <= 1.35
```

也就是说：

$$
\boxed{
\text{S1 blocker 已经收敛为 bs256/bs512 memory，而不是 step。}
}
$$

Time AUC 失败更关键：

```text
mean ValLossAUC_time:
  B0  = 0.1582
  M12 = 0.2941
```

因此：

$$
ValLossAUC_{time,M12} > ValLossAUC_{time,MLP}.
$$

这说明 M12 虽然 final quality 更好，但 wall-clock 训练效率还没有系统级优势。

---

### 0.3 追加 S1/cache repair 没有产生新路线

v7.6 后续尝试了 M14/M15/M16：

```text
M14:
  backward 重算 hidden-y，减少常驻 cache
  GradPass = 6/6
  task 与 M12 一致
  memory 有改善
  但 step fail / S2 不完整

M15:
  hidden-y half cache
  smoke relerr = 4.82e-02
  GradFail，停止

M16:
  只缓存第一层 y，第二层 y backward 重算
  GradPass = 6/6
  step 接近 S1
  但 bs512 memory 更差
```

这说明单纯 cache policy 小修无法闭合 formal success。需要更底层的 kernel/op attribution 与 dense-preserving fused package。

---

## 1. 当前问题的本质

### 1.1 当前已经不是“表达力是否存在”的问题

M12 在 10 seeds 下保持：

$$
\Delta Acc_{\text{macro,val}}=+0.02057,
$$

$$
CI_{95\%,macro}^{low}=+0.01491,
$$

$$
p_{\text{Holm}}=1.35\times10^{-6}.
$$

并且 M12 相比 MLP-distill 仍有：

$$
\Delta Acc_{\text{macro,val}}=+0.01517.
$$

因此当前问题不是 KAN basis 是否弱于 MLP，也不是 C3 teacher 是否完全解释优势。当前问题是：

$$
\boxed{
\text{如何把 M12 的质量优势转化为 S1 memory 与 wall-clock time-AUC 优势。}
}
$$

---

### 1.2 当前也不是普通 optimizer 小修问题

M12 的 step ratio 已经是：

$$
r_{\text{step,mean}}=1.1683,
$$

$$
r_{\text{step,max}}=1.2591.
$$

这已经明显低于 S2 step gate，也低于 S1 step gate：

$$
r_{\text{step}}\leq1.35.
$$

所以 current blocker 不是 optimizer update 太慢。真正问题是：

```text
1. bs256/bs512 memory 仍高于 1.00；
2. ValLossAUC_time 仍明显高于 MLP；
3. forward path / teacher objective / validation loop / profiler-task mismatch 尚未解释；
4. 真实 kernel launch / op-level profiler 仍未形成闭环。
```

---

### 1.3 TimeAUCPass 失败不能简单归因于 step ratio

M12 的 step ratio mean 是 `1.1683`，但 ValLossAUC_time 比 B0 高很多：

$$
\frac{0.2941}{0.1582}\approx1.86.
$$

这个比例远大于 step ratio 的 `1.1683`。因此 Time AUC 失败可能不是单纯 “每步慢 16.8%”，而可能来自：

```text
1. task trace wall-clock 计时包含 teacher forward / distillation overhead；
2. validation / trace logging / CUDA sync 被 M12 path 放大；
3. benchmark full-step path 与 task training path 不一致；
4. M12 early-stage val loss 更高，虽然 final acc/NLL 更好；
5. M12 loss curve 的优势出现在后期，因此 AUC_time 不占优；
6. profiler 没有真实拆出 kernel launch / op fragmentation，导致 forward_ratio 及 wall-clock 解释不足。
```

因此 v7.7 的第一本质任务不是“把 step 再降一点”，而是：

$$
\boxed{
\text{把 step profiler、task trace wall-clock、teacher cost、validation cost、kernel launch cost 对齐。}
}
$$

---

### 1.4 S1 memory fail 不能靠粗暴压缩解决

M12 当前距离 S1 memory 的差距是：

```text
bs256 memory ratio = 1.0086
bs512 memory ratio = 1.0381
```

要达到 FullGridS1：

$$
r_{\text{mem,max}}<1.00.
$$

bs512 需要降低：

$$
1-\frac{1.00}{1.0381}\approx3.67\%.
$$

这不是巨大 gap，但很脆。M14/M16 已经说明，简单 recompute / cache policy 会引入 step 或 memory 副作用。因此 S1 repair 必须是 dense-preserving、kernel-aware，而不是 aggressive compression。

---

## 2. v7.7 总体目标

v7.7 的目标是把 M12 从：

```text
S2-FairSystemMinimumSuccess
```

推进到至少：

```text
S1-or-TimeAUC-SystemCandidate
```

并最终争取：

```text
FullSystemAdvantage
```

v7.7 的最低目标：

$$
\boxed{
\text{ProfilerPass}
+
\text{TimeAccountingPass}
+
\text{M12-10seed-Reproducible}
+
\text{FullGridS2Pass}
}
$$

v7.7 的正式目标：

$$
\boxed{
\text{FullGridS1Pass}
+
\text{TimeAUCPass}
+
\text{FairnessPass}
+
\text{GradPass}
+
\text{MacroSignificantPass}
}
$$

v7.7 的强目标：

$$
\boxed{
\text{FullGridS1Pass}
+
\text{TimeAUCPass}
+
\text{ForwardRatioRepair}
+
\text{ScalingOrRobustnessPass}
+
\text{MechanismExplained}
}
$$

其中：

```text
ProfilerPass:
  真实 kernel launch / op-level profiler 可用，不以 proxy 替代。

TimeAccountingPass:
  task trace wall-clock 可以拆为 train/teacher/validation/logging/sync。

FullGridS1Pass:
  9/9 primary shapes memory < 1.00 且 step <= 1.35。

TimeAUCPass:
  M12 或 M12-derived candidate 的 ValLossAUC_time <= MLP-AdamW。
```

---

## 3. v7.7 禁止事项

第一，不允许回到 loss、sampler、class weight、focal、target-margin、classwise-safe 主线。当前 blocker 不是刷榜问题。

第二，不允许只看 final accuracy。M12 已经 final accuracy win；v7.7 必须看 time-AUC、time-to-target、S1 memory、profiler。

第三，不允许只看 mean ratio。S1/S2 都必须看 full-grid 9/9 shapes 与 worst shape。

第四，不允许把 kernel-count proxy、torch-op proxy、forward-GEMM proxy 冒充真实 kernel launch count。真实 profiler 不可用时必须写：

```text
metric_unavailable
```

第五，不允许在没有 P2/P3 attribution 的情况下写 fused package 原因。kernel repair 必须由 profiler 驱动。

第六，不允许用 FP16/half hidden-y cache 进入主线，除非先过：

$$
grad\_relerr_{max}\leq10^{-4}.
$$

M15 已经证明 naive half hidden-y cache 会 GradFail。

第七，不允许为了 S1 memory 牺牲 macro advantage。任何 S1 candidate 必须满足：

$$
\Delta Acc_{\text{macro,val}}\geq0.0200.
$$

第八，不允许把 teacher distillation cost 隐藏在 wall-clock 之外。必须同时报告：

```text
online_teacher_cost
offline_teacher_precompute_cost
inference_cost
train_step_cost_excluding_teacher
train_step_cost_including_teacher
```

第九，不允许把 posthoc calibration 当 architecture claim。posthoc 只能单独报告。

第十，不允许在 scaling / robustness 未测前，把 M12 外推为大任务通用优势。

---

## 4. 核心假设

### H1：M12 的 TimeAUCPass 失败主要来自时间核算或非训练主路径开销，而不完全来自 loss dynamics

M12 的 step ratio mean 只有 `1.1683`，但 ValLossAUC_time 是 B0 的约 `1.86x`。H1 假设：Time AUC 失败中有相当一部分来自 task trace wall-clock 中的 teacher、validation、logging、sync 或 path mismatch。

H1 成立标准：

将 wall-clock 分解为：

```text
train_forward_backward_update_time
teacher_forward_time
validation_time
logging_time
cuda_sync_time
data_loading_time
one_time_compile_warmup_time
```

若满足：

$$
\frac{T_{\text{non-train-overhead,M12}}}{T_{\text{wall-clock,M12}}}\geq0.25,
$$

或：

$$
\frac{T_{\text{teacher+validation+logging}}}{T_{\text{wall-clock,M12}}}\geq0.25,
$$

则 Time AUC fail 不是纯 primitive step fail，而是 time accounting / training loop problem。

若训练主路径仍满足：

$$
\frac{T_{\text{train-step,M12}}}{T_{\text{train-step,MLP}}}\leq1.25,
$$

但 wall-clock AUC fail，则 v7.7 应先修 loop accounting / teacher offline / validation schedule。

---

### H2：如果 AUC_step 也失败，则 M12 的 optimization dynamics 需要重新设计

H2 与 H1 互补。如果 M12 的 ValLossAUC_step 也明显差于 MLP，那么问题不是 wall-clock，而是训练曲线本身：

$$
ValLossAUC_{step,M12}>ValLossAUC_{step,MLP}.
$$

H2 成立标准：

$$
\frac{ValLossAUC_{step,M12}}{ValLossAUC_{step,MLP}}>1.10.
$$

若 H2 成立，需要研究 teacher distillation schedule / early CE dominance / M5 initialization / learning dynamics，而不是先写 kernel。

若 H2 不成立，而 AUC_time 失败，则优先做 runtime/profiler。

---

### H3：真实 kernel launch / op-level profiler 可以解释 M12 forward bottleneck

M12 的 forward ratio 之前仍偏高。H3 假设：forward overhead 主要来自 kernel launch fragmentation、head op fragmentation、layout conversion、小 kernel under-occupancy 或 eager dispatch，而不是 KAN 数学不可高效。

H3 成立标准：

ProfilerPass：

```text
real_kernel_launch_count_measured = 1
kernel_name_available = 1
kernel_duration_available = 1
phase_to_kernel_mapping_available = 1
top3_time_sources_explain >= 0.70
unknown_time_fraction <= 0.10
```

Actionable forward bottleneck：

$$
\frac{T_{\text{top forward source}}}{T_{\text{forward}}}\geq0.20.
$$

Kernel fragmentation：

$$
N_{\text{small kernels under 10us}}\geq0.30N_{\text{kernel total}}.
$$

Head fragmentation：

$$
N_{\text{head kernels}}\geq0.30N_{\text{kernel total}}.
$$

If H3 fails because profiler unavailable, no kernel-cause claim is allowed.

---

### H4：M12 S1 memory fail comes from cache/live-set scheduling rather than necessary expression state

S1 memory fail only occurs for bs256/512, and top memory sources include fused stack root input / hidden-y cache / optimizer state. H4 假设：这些 are implementation live-set sources, not essential expression state.

H4 成立标准：

Attribution identifies at least one actionable source satisfying:

$$
\frac{M_{\text{source}}}{M_{\text{peak}}}\geq0.02.
$$

A repair candidate must satisfy:

$$
r_{\text{mem,max,new}}<r_{\text{mem,max,M12}}-0.02.
$$

and:

$$
\Delta Acc_{\text{macro,val,new}}\geq\Delta Acc_{\text{macro,val,M12}}-0.003.
$$

If every memory source is optimizer state or unavoidable MLP-comparable state, S1 may require new parameter/state design rather than cache trimming.

---

### H5：Dense-preserving fused backward / streaming cache can recover S1 without harming GradPass

M14/M16 showed recompute policies can preserve task and GradPass, but step or memory worsens. H5 假设：the same idea can work if implemented as fused backward / streaming cache, not Python/Torch recompute.

H5 成立标准 for microkernel：

$$
\frac{T_{\text{candidate}}}{T_{\text{current}}}\leq0.80
$$

or:

$$
\frac{M_{\text{candidate}}}{M_{\text{current}}}\leq0.80.
$$

and:

$$
grad\_relerr_{max}\leq10^{-4}.
$$

Full package pass：

$$
r_{\text{mem,max}}\lt1.00,
$$

$$
r_{\text{step,max}}\leq1.35,
$$

$$
\Delta Acc_{\text{macro,val}}\geq0.0200.
$$

---

### H6：M12 的 system advantage must survive fair accounting against MLP-distill and teacher cost

Current B2 proves teacher helps MLP but does not close the gap. H6 假设：M12 advantage remains after including fair teacher and distillation accounting.

H6 成立标准：

M12 vs MLP-distill:

$$
Acc_{M12}-Acc_{B2}\geq0.010.
$$

NLL fairness:

$$
NLL_{M12}\leq NLL_{B2}+0.01.
$$

Teacher cost transparency:

```text
offline_teacher_precompute_sec recorded
online_teacher_forward_sec recorded
train_step_excluding_teacher recorded
train_step_including_teacher recorded
```

If MLP-distill closes the gap or teacher cost dominates wall-clock, route must be downgraded.

---

### H7：Scaling / robustness should be opened only after time/profiler ambiguity is resolved

H7 假设：M12 may be a small-task clean-data success, but scaling is meaningful only after we know whether the current wall-clock issue is real or measurement/loop overhead.

H7 gate open condition：

```text
P1 TimeAccountingPass = true
P3 ProfilerPass = true
P5 or P6 provides stable S2/S1 package
```

Then measure:

$$
AUC_{data,M12}>AUC_{data,MLP}
$$

or:

$$
AccDrop_{M12}<AccDrop_{MLP}
$$

for at least two robustness settings.

---

## 5. Candidate 设计

### 5.1 Baselines

```text
B0-MLP-AdamW-reference
B2-MLP-C3-logit-distill-T4-alpha025
B3-MLP-self-distill
B4-C3-teacher
B5-M12-final-clean
B6-M12-no-distill
B7-M12-offline-teacher-logit-cache
```

### 5.2 Time accounting candidates

```text
TA0-B0-MLP-clean-loop
TA1-B2-MLP-distill-online-teacher
TA2-B2-MLP-distill-offline-logits
TA3-M12-online-teacher
TA4-M12-offline-logits
TA5-M12-no-validation-during-train
TA6-M12-validation-every-20
TA7-M12-validation-every-10
TA8-M12-no-extra-sync
TA9-M12-train-only-timing
```

### 5.3 Profiler candidates

```text
PROF0-MLP-AdamW-reference
PROF1-B2-MLP-distill
PROF2-M12-final
PROF3-M12-no-distill
PROF4-M12-offline-logits
PROF5-M12-unfused-stack
PROF6-M12-no-packed-head
PROF7-A2S-reference
PROF8-D3-reference
PROF9-C3-teacher
```

### 5.4 Dense-preserving fused / streaming candidates

```text
FB0-M12-current
FB1-M12-fused-head-forward
FB2-M12-fused-head-backward
FB3-M12-fused-linear-silu-forward
FB4-M12-fused-linear-silu-backward
FB5-M12-streaming-manual-cache
FB6-M12-streaming-head-cache
FB7-M12-no-full-basis-materialization
FB8-M12-forward-kernel-count-trim
FB9-M12-dense-preserving-fused-backward-combo
FB10-M12-streaming-cache-fused-combo

A2S-FB0-A2S-current
A2S-FB1-A2S-fused-backward
A2S-FB2-A2S-streaming-cache
A2S-FB3-A2S-bs512-memory-specialized-tile

D3-FB0-D3-current
D3-FB1-D3-fused-backward
D3-FB2-D3-streaming-cache
D3-FB3-D3-dense-preserving-fused-package
```

### 5.5 S1 memory candidates

```text
S1-0-M12-current
S1-1-hidden-y-fused-recompute
S1-2-selective-y-cache-with-error-compensated-derivative
S1-3-root-input-lifetime-trim
S1-4-bs512-head-temp-streaming
S1-5-forward-temp-fusion
S1-6-optimizer-state-phase-separation
S1-7-packed-owner-state-trim
S1-8-allocator-padding-control
S1-9-M12-S1-dense-preserving-combo
```

### 5.6 Optimization dynamics candidates

These are not generic optimizer sweeps. They only run if H2 shows AUC_step failure.

```text
OD0-M12-current-alpha025-constant
OD1-M12-distill-alpha-warmup
OD2-M12-distill-alpha-cooldown
OD3-M12-CE-first-then-distill
OD4-M12-distill-target-NLL-balanced
OD5-M12-M5-init-with-shorter-distill
OD6-M12-480-step-curve-diagnostic
```

### 5.7 Scaling / robustness candidates

```text
R0-MLP-AdamW
R1-MLP-distill
R2-M12-final
R3-M12-offline-logits
R4-M12-S1-combo, if exists
R5-M12-fused-forward-combo, if exists
```

---

## 6. 实验阶段总览

v7.7 分为十四个阶段：

```text
P0: v7.6 clean result reproduction and provenance lock
P1: time accounting reconciliation
P2: AUC_step vs AUC_time dynamics decomposition
P3: true kernel launch / op-level profiler
P4: S1 memory source attribution
P5: dense-preserving fused backward / streaming cache microbench
P6: M12 fused/streaming full package
P7: offline teacher and fairness-time accounting
P8: optimization dynamics only if AUC_step fails
P9: 10-seed task/time re-confirmation
P10: S1 / S2 full-grid profiler
P11: scaling and robustness, gated
P12: geometry Pareto diagnostic, gated
P13: candidate co-selection and route decision
P14: artifact and failure audit
```

P1-P3 是 v7.7 的核心定位阶段。  
P5-P6 是实现阶段，但只允许由 P3/P4 的 attribution 驱动。  
P8 只有在 AUC_step 失败时打开。  
P11/P12 只有在 system ambiguity 解除后打开。

---

## 7. P0：v7.6 clean result reproduction and provenance lock

### 7.1 目的

P0 确认 v7.7 与 v7.6 clean run 可比，不寻找新结论。P0 必须复现 M12 的 task、grad、S2 和 fairness denominator。

### 7.2 必跑对象

```text
B0-MLP-AdamW-reference
B2-MLP-C3-logit-distill
M12-final-clean
M12-no-distill
C3-teacher
```

### 7.3 必须记录字段

```text
run_id
artifact_root
candidate
family
status
fake_data_used
proxy_row_used
uses_loss_backward
uses_torch_autograd_graph
manual_forward_available
manual_backward_available
manual_update_available
nonKAN_param_count
head_type
head_is_kan
teacher_used
teacher_candidate
distill_temperature
distill_alpha
grad_relerr_max
grad_cos_min
macro_val_gap
ci95_low
holm_p
test_gap
ECE_delta
NLL_delta
memory_ratio_mean
memory_ratio_max
step_ratio_mean
step_ratio_max
forward_ratio_mean
backward_ratio_mean
S2_shape_count
S1_shape_count
ValLossAUC_step
ValLossAUC_time
```

### 7.4 判断标准

P0 pass：

```text
fake_data_used = 0
proxy_row_used = 0
nonKAN_param_count = 0 for M12
manual_backward_available = 1
uses_loss_backward = 0 for M12
grad_relerr_max <= 1e-4
grad_cos_min >= 0.999
```

M12 task reproduction：

$$
|\Delta Acc_{\text{M12,v77}}-0.02057|\leq0.004.
$$

M12 efficiency reproduction：

$$
|r_{\text{mem,M12,v77}}-1.0143|\leq0.02.
$$

$$
|r_{\text{step,M12,v77}}-1.1683|\leq0.10.
$$

### 7.5 可视化

```text
p0_m12_reproduction_task_bar.svg
p0_m12_reproduction_efficiency_bar.svg
p0_contract_heatmap.svg
p0_m12_route_lineage.svg
```

---

## 8. P1：time accounting reconciliation

### 8.1 目的

P1 是 v7.7 最重要阶段。它要解释为什么 M12 step ratio 已经是 `1.1683`，但 ValLossAUC_time 仍是 MLP 的约 `1.86x`。P1 不改模型，只拆时间。

### 8.2 必跑对象

```text
B0-MLP-AdamW
B2-MLP-distill-online-teacher
B2-MLP-distill-offline-logits
M12-online-teacher
M12-offline-logits
M12-no-distill
```

### 8.3 设置

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0,1,2 initially
  0..9 after instrumentation stable

steps:
  240

trace_every:
  10

teacher modes:
  online teacher forward
  offline precomputed logits
  no teacher
```

### 8.4 必须记录字段

```text
candidate
dataset
seed
step
wall_clock_time_sec_total
train_step_time_sec
forward_time_sec
backward_time_sec
update_time_sec
teacher_forward_time_sec
distill_loss_time_sec
validation_time_sec
logging_time_sec
cuda_sync_time_sec
data_loading_time_sec
compile_warmup_time_sec
profiler_overhead_sec
examples_seen
samples_per_second_train_only
samples_per_second_end_to_end
val_loss
val_acc
ValLossAUC_time_total
ValLossAUC_time_train_only
ValLossAUC_step
```

### 8.5 判断标准

TimeAccountingPass：

```text
all major time components measured
unknown_time_fraction <= 0.10
train_only_time and end_to_end_time both reported
online/offline teacher modes separated
validation/logging/sync separated
```

If:

$$
\frac{T_{\text{teacher+validation+logging+sync}}}{T_{\text{wall-clock}}}\geq0.25,
$$

then TimeAUC fail is not purely primitive step fail.

If:

$$
ValLossAUC_{\text{time,train-only,M12}}\leq1.10\cdot ValLossAUC_{\text{time,train-only,MLP}}
$$

but:

$$
ValLossAUC_{\text{time,total,M12}}>1.25\cdot ValLossAUC_{\text{time,total,MLP}},
$$

then route is:

```text
time_accounting_or_loop_overhead
```

### 8.6 可视化

```text
p1_time_breakdown_stacked.svg
p1_total_vs_train_only_auc.svg
p1_online_vs_offline_teacher_time.svg
p1_validation_logging_sync_bar.svg
p1_wallclock_reconciliation_dashboard.svg
```

---

## 9. P2：AUC_step vs AUC_time dynamics decomposition

### 9.1 目的

P2 判断 TimeAUC fail 是 runtime 问题还是 optimization dynamics 问题。P2 不改 kernel，只分析曲线。

### 9.2 必跑对象

```text
B0-MLP-AdamW
B2-MLP-distill
M12-final
M12-no-distill
M12-offline-logits
```

### 9.3 必须记录字段

```text
candidate
dataset
seed
step
wall_clock_time_sec
val_loss
val_acc
train_loss
train_acc
ValLossAUC_step
ValLossAUC_time
ValAccAUC_step
ValAccAUC_time
early_loss_slope_step
mid_loss_slope_step
late_loss_slope_step
early_loss_slope_time
mid_loss_slope_time
late_loss_slope_time
time_to_mlp_final_acc
time_to_mlp_plus_005_acc
time_to_loss_threshold
```

### 9.4 判断标准

Optimization dynamics problem if:

$$
\frac{ValLossAUC_{\text{step,M12}}}{ValLossAUC_{\text{step,MLP}}}>1.10.
$$

Runtime problem if:

$$
ValLossAUC_{\text{step,M12}}\leq ValLossAUC_{\text{step,MLP}}
$$

but:

$$
ValLossAUC_{\text{time,M12}}>ValLossAUC_{\text{time,MLP}}.
$$

Late-only advantage if M12 final acc wins but early/mid slopes lose:

```text
early_loss_slope_step worse than MLP
late_acc better than MLP
```

This triggers P8 optimization dynamics only if the gap is not explained by runtime.

### 9.5 可视化

```text
p2_val_loss_vs_step.svg
p2_val_loss_vs_time.svg
p2_auc_step_vs_auc_time.svg
p2_loss_slope_by_phase.svg
p2_time_to_target.svg
```

---

## 10. P3：true kernel launch / op-level profiler

### 10.1 目的

P3 补齐真实 kernel/op attribution，避免继续用 proxy 判断。P3 必须回答 M12 的 forward、head、stack、distill 路径由哪些真实 CUDA kernels / PyTorch ops 构成。

### 10.2 工具优先级

```text
torch.profiler with CUDA activities
record_shapes = true
profile_memory = true
with_stack = true
NVTX ranges around manual phases

If available:
  nsys profile
  ncu selected kernel stats
  torch.cuda.memory_snapshot
  torch.cuda.memory._record_memory_history
```

Unavailable metrics must be written as:

```text
metric_unavailable
```

### 10.3 必跑对象

```text
PROF0-MLP-AdamW-reference
PROF1-B2-MLP-distill
PROF2-M12-final
PROF3-M12-no-distill
PROF4-M12-offline-logits
PROF5-M12-unfused-stack
PROF6-M12-no-packed-head
PROF7-A2S-reference
PROF8-D3-reference
PROF9-C3-teacher
```

### 10.4 Shape grid

```text
datasets:
  MNIST
  KMNIST

batch sizes:
  128
  512

depth:
  native

warmup:
  20 steps

profile steps:
  20 steps

repetitions:
  3
```

### 10.5 必须记录字段

Profiler availability：

```text
candidate
dataset
batch_size
profiler_backend
torch_profiler_available
nsys_available
ncu_available
cuda_activities_available
profile_memory_available
with_stack_available
nvtx_ranges_available
profiler_status
```

Kernel launch metrics：

```text
kernel_count_total
kernel_count_forward
kernel_count_backward
kernel_count_update
kernel_count_teacher
kernel_count_validation
kernel_count_head_forward
kernel_count_head_backward
kernel_count_stack_forward
kernel_count_stack_backward
unique_kernel_names_total
small_kernel_count_under_10us
small_kernel_time_total_ms
top_kernel_name_1
top_kernel_time_ms_1
top_kernel_calls_1
top_kernel_name_2
top_kernel_time_ms_2
top_kernel_calls_2
top_kernel_name_3
top_kernel_time_ms_3
top_kernel_calls_3
```

Op-level metrics：

```text
torch_op_count_total
torch_op_count_forward
torch_op_count_backward
torch_op_count_head
torch_op_count_stack
aten_mm_count
aten_addmm_count
aten_mul_count
aten_silu_count
aten_sigmoid_count
aten_sum_count
aten_copy_count
aten_contiguous_count
aten_view_reshape_count
layout_conversion_count
```

Phase timing：

```text
forward_total_ms
backward_total_ms
update_total_ms
teacher_forward_ms
distill_loss_ms
head_forward_ms
head_backward_ms
stack_forward_ms
stack_backward_ms
loss_delta_ms
optimizer_step_ms
cpu_dispatch_ms
cuda_kernel_time_ms
cuda_memcpy_time_ms
cuda_memset_time_ms
cuda_sync_time_ms
```

Memory profiler：

```text
self_cuda_memory_usage_MB
allocated_memory_peak_MB
reserved_memory_peak_MB
allocation_count
deallocation_count
memory_snapshot_events
largest_live_tensor_MB
top_memory_op_1
top_memory_op_2
top_memory_op_3
```

### 10.6 判断标准

ProfilerPass：

```text
kernel_count_total measured
kernel name available
kernel duration available
phase-to-kernel mapping available
top3 time sources explain >= 70% of step time
unknown_time_fraction <= 0.10
```

Forward bottleneck identified if:

$$
\frac{T_{\text{top forward source}}}{T_{\text{forward}}}\geq0.20.
$$

Kernel fragmentation identified if:

$$
N_{\text{small kernels under 10us}}\geq0.30N_{\text{kernel total}}.
$$

Head fragmentation identified if:

$$
N_{\text{head kernels}}\geq0.30N_{\text{kernel total}}.
$$

Layout bottleneck identified if:

$$
T_{\text{contiguous/copy/layout}}\geq0.10T_{\text{step}}.
$$

### 10.7 可视化

```text
p3_kernel_count_by_phase.svg
p3_top_kernel_time_bar.svg
p3_small_kernel_fragmentation.svg
p3_torch_op_count_stacked.svg
p3_forward_breakdown_m12_vs_mlp.svg
p3_layout_conversion_bar.svg
p3_kernel_timeline.svg
p3_profiler_status_dashboard.svg
```

---

## 11. P4：S1 memory source attribution

### 11.1 目的

P4 解释为什么 M12 仍不是 S1，特别是 bs256/bs512 memory > 1.00。P4 只做 attribution，不做修复。

### 11.2 必跑对象

```text
M12-final
M12-no-distill
M12-offline-logits
B0-MLP
B2-MLP-distill
```

### 11.3 Shape grid

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

batch sizes:
  128
  256
  512
```

### 11.4 必须记录字段

```text
candidate
dataset
batch_size
peak_allocated_MB
peak_reserved_MB
memory_ratio
step_ratio
forward_ratio
backward_ratio
update_ratio
root_input_cache_MB
hidden_y_cache_MB
head_temp_MB
stack_temp_MB
manual_cache_MB
optimizer_state_MB
packed_owner_state_MB
allocator_padding_MB
reserved_unallocated_MB
largest_live_tensor_MB
materialized_tensor_count
kernel_count_total
top1_memory_source
top2_memory_source
top3_memory_source
unknown_memory_fraction
```

### 11.5 判断标准

AttributionPass：

```text
top3 memory sources identified
unknown_memory_fraction <= 0.10
materialized_tensor_count measured
kernel_count_total measured or metric_unavailable explicitly
```

Actionable source:

$$
\frac{M_{\text{source}}}{M_{\text{peak}}}\geq0.02.
$$

S1 memory gap source explained if top actionable sources together exceed:

$$
r_{\text{mem,worst}}-1.00.
$$

### 11.6 可视化

```text
p4_s1_memory_waterfall.svg
p4_s1_gap_by_shape.svg
p4_bs512_memory_source_bar.svg
p4_cache_vs_optimizer_state.svg
p4_memory_source_heatmap.svg
```

---

## 12. P5：dense-preserving fused backward / streaming cache microbench

### 12.1 目的

P5 只对 P3/P4 找出的 bottleneck 写 microbench。它不是直接改 full package，而是验证 fused/streaming 是否真实降低 memory/time 且保持 GradPass。

### 12.2 必跑对象

```text
A2S-current
D3-current
M12-current
```

### 12.3 Microkernels

```text
MK0-current-head-forward
MK1-fused-head-forward
MK2-current-head-backward
MK3-fused-head-backward
MK4-current-linear-silu-forward
MK5-fused-linear-silu-forward
MK6-current-linear-silu-backward
MK7-fused-linear-silu-backward
MK8-streaming-manual-cache
MK9-streaming-head-cache
MK10-no-full-basis-materialization
MK11-fused-transform-mix-backward
MK12-bs512-memory-specialized-tile
MK13-forward-kernel-count-trim
MK14-dense-preserving-fused-backward-combo
```

### 12.4 必须记录字段

```text
microkernel
parent_candidate
input_shape
output_shape
dataset
batch_size
current_time_ms
candidate_time_ms
time_ratio_vs_current
current_memory_MB
candidate_memory_MB
memory_ratio_vs_current
forward_relerr
grad_relerr
grad_cos
macro_preservation_proxy
materialized_tensor_count_current
materialized_tensor_count_candidate
kernel_count_current
kernel_count_candidate
allocation_count_current
allocation_count_candidate
top_changed_kernel
top_removed_kernel
bandwidth_estimate_GBps
```

### 12.5 判断标准

Microkernel useful if:

$$
\frac{T_{\text{candidate}}}{T_{\text{current}}}\leq0.80
$$

or:

$$
\frac{M_{\text{candidate}}}{M_{\text{current}}}\leq0.80.
$$

Eligible for full package if:

```text
grad_relerr <= 1e-4
grad_cos >= 0.999
time_ratio_vs_current <= 0.90
memory_ratio_vs_current <= 0.95
kernel_count_candidate < kernel_count_current
```

Dense preservation proxy:

$$
\|f_{\text{candidate}}(x)-f_{\text{current}}(x)\|_{\infty}\leq10^{-5}
$$

or task-smoke delta:

$$
|\Delta Acc_{\text{smoke}}|\leq0.002.
$$

### 12.6 可视化

```text
p5_microkernel_memory_time_pareto.svg
p5_grad_correctness_lollipop.svg
p5_kernel_count_reduction_bar.svg
p5_materialized_tensor_reduction_bar.svg
p5_a2s_d3_m12_microbench_comparison.svg
```

---

## 13. P6：M12 fused / streaming full package

### 13.1 目的

P6 将 P5 有效 microkernels 集成到 full package，目标是 S1 或 time-AUC repair。P6 必须重新跑 task、GradPass、full-grid profiler。

### 13.2 Candidates

```text
F0-M12-current
F1-M12-fused-head-forward
F2-M12-fused-head-backward
F3-M12-fused-linear-silu-forward
F4-M12-fused-linear-silu-backward
F5-M12-streaming-cache
F6-M12-dense-preserving-fused-backward
F7-M12-fused-forward-combo
F8-M12-fused-backward-combo
F9-M12-fused-streaming-S1-combo
F10-M12-offline-teacher-fused-forward-combo
```

### 13.3 必须记录字段

```text
candidate
components
implementation_status
grad_relerr_max
grad_cos_min
macro_val_gap
test_gap
ECE_delta
NLL_delta
ValLossAUC_step
ValLossAUC_time
memory_ratio_mean
memory_ratio_max
step_ratio_mean
step_ratio_max
forward_ratio_mean
backward_ratio_mean
S1_shape_count
S2_shape_count
memory_improvement_vs_M12
step_improvement_vs_M12
forward_improvement_vs_M12
macro_delta_vs_M12
kernel_count_total
kernel_count_reduction_vs_M12
materialized_tensor_count
```

### 13.4 判断标准

Forward repair useful:

$$
r_{\text{forward,new}}\leq1.50
$$

or:

$$
r_{\text{forward,new}}\leq0.80r_{\text{forward,M12}}.
$$

S1 repair useful:

$$
r_{\text{mem,max,new}}<r_{\text{mem,max,M12}}-0.02.
$$

FullGridS1Pass:

$$
r_{\text{mem,max}}<1.00.
$$

$$
r_{\text{step,max}}\leq1.35.
$$

Task preservation:

$$
\Delta Acc_{\text{macro,val,new}}\geq0.0200.
$$

and:

$$
\Delta Acc_{\text{macro,val,new}}\geq
\Delta Acc_{\text{macro,val,M12}}-0.003.
$$

Time repair:

$$
ValLossAUC_{\text{time,new}}\leq ValLossAUC_{\text{time,B0}}.
$$

or:

$$
ValLossAUC_{\text{time,new}}\leq0.80ValLossAUC_{\text{time,M12}}.
$$

### 13.5 可视化

```text
p6_package_efficiency_task_pareto.svg
p6_s1_shape_pass_heatmap.svg
p6_task_preservation_bar.svg
p6_forward_repair_bar.svg
p6_kernel_count_vs_forward_ratio.svg
p6_auc_time_repair_bar.svg
```

---

## 14. P7：offline teacher and fairness-time accounting

### 14.1 目的

P7 将 C3 teacher distillation 的训练成本拆清。M12 的 task advantage 已经公平优于 MLP-distill，但 time accounting 还必须区分 online teacher 和 offline logits。

### 14.2 必跑对象

```text
B2-MLP-distill-online
B2-MLP-distill-offline
M12-online-teacher
M12-offline-logits
M12-no-distill
C3-teacher-precompute
```

### 14.3 必须记录字段

```text
candidate
teacher_mode
teacher_precompute_sec
teacher_storage_MB
online_teacher_forward_sec_per_step
distill_loss_sec_per_step
train_step_excluding_teacher_sec
train_step_including_teacher_sec
ValLossAUC_time_excluding_teacher
ValLossAUC_time_including_teacher
val_acc
test_acc
NLL
ECE
memory_ratio
step_ratio
```

### 14.4 判断标准

Fair time pass excluding teacher:

$$
ValLossAUC_{\text{time,M12,excluding-teacher}}
\leq
ValLossAUC_{\text{time,MLP-distill,excluding-teacher}}.
$$

Fair time pass including teacher:

$$
ValLossAUC_{\text{time,M12,including-teacher}}
\leq1.10
ValLossAUC_{\text{time,MLP-distill,including-teacher}}.
$$

If offline teacher solves time-AUC but online does not, official recipe must specify offline logit cache.

### 14.5 可视化

```text
p7_online_vs_offline_teacher_auc.svg
p7_teacher_cost_breakdown.svg
p7_fairness_time_pareto.svg
p7_storage_vs_time_tradeoff.svg
```

---

## 15. P8：optimization dynamics only if AUC_step fails

### 15.1 打开条件

Only open if P2 shows:

$$
ValLossAUC_{\text{step,M12}}>1.10ValLossAUC_{\text{step,MLP}}.
$$

### 15.2 目的

如果 M12 每 step 的 loss dynamics 也差，则 v7.7 需要研究训练目标，而不是 kernel。P8 不是普通 optimizer sweep，只针对 distillation dynamics。

### 15.3 Candidates

```text
OD0-M12-current-alpha025-constant
OD1-M12-distill-alpha-warmup
OD2-M12-distill-alpha-cooldown
OD3-M12-CE-first-then-distill
OD4-M12-distill-target-NLL-balanced
OD5-M12-M5-init-with-shorter-distill
OD6-M12-480-step-curve-diagnostic
```

### 15.4 必须记录字段

```text
candidate
distill_schedule
alpha_initial
alpha_final
temperature
val_acc
test_acc
NLL
ECE
ValLossAUC_step
ValLossAUC_time
time_to_target_acc
grad_relerr_max
memory_ratio
step_ratio
```

### 15.5 判断标准

AUC_step repair:

$$
ValLossAUC_{\text{step,new}}\leq ValLossAUC_{\text{step,MLP}}.
$$

Task preservation:

$$
\Delta Acc_{\text{macro,val,new}}\geq0.0200.
$$

No efficiency regression:

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

### 15.6 可视化

```text
p8_distill_schedule_auc_step.svg
p8_distill_schedule_loss_curves.svg
p8_task_vs_auc_pareto.svg
```

---

## 16. P9：10-seed task/time re-confirmation

### 16.1 目的

对 P6/P7/P8 产生的 final candidate 做 10-seed confirm。不能用 microbench 或 3-seed 直接下结论。

### 16.2 必跑对象

```text
B0-MLP-AdamW
B2-MLP-distill
M12-final
best-M12-fused-package
best-M12-offline-teacher-package
best-M12-S1-package, if exists
```

### 16.3 设置

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0..9

steps:
  240

trace_every:
  10

bootstrap:
  10000
```

### 16.4 必须记录字段

```text
candidate
dataset
seed
val_acc
test_acc
val_loss
test_loss
ECE
NLL
Brier
val_gap_vs_MLP
test_gap_vs_MLP
val_gap_vs_MLP_distill
CI95_low
CI95_high
Holm_p
ValLossAUC_step
ValLossAUC_time_total
ValLossAUC_time_train_only
time_to_target_acc
time_to_target_loss
memory_ratio_mean
memory_ratio_max
step_ratio_mean
step_ratio_max
forward_ratio_mean
backward_ratio_mean
```

### 16.5 判断标准

MacroSignificantPass:

$$
\Delta Acc_{\text{macro,val}}\geq0.0200.
$$

$$
CI_{95\%,macro}^{low}>0.
$$

$$
p_{\text{Holm}}<0.05.
$$

FairnessPass:

$$
Acc_{\text{candidate}}-Acc_{\text{MLP-distill}}\geq0.010.
$$

TimeAUCPass:

$$
ValLossAUC_{\text{time,candidate}}\leq ValLossAUC_{\text{time,MLP}}.
$$

S1/S2 gates as appropriate.

### 16.6 可视化

```text
p9_seedwise_val_gap_boxplot.svg
p9_bootstrap_ci_macro.svg
p9_auc_time_by_candidate.svg
p9_system_pareto.svg
```

---

## 17. P10：S1 / S2 full-grid profiler

### 17.1 目的

P10 验证 final candidate 是否真的保持 full-grid S2/S1。

### 17.2 Shape grid

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

batch:
  128
  256
  512

bench_warmup:
  20 or 50

bench_reps:
  100 or 200
```

### 17.3 必须记录字段

```text
candidate
dataset
batch_size
memory_ratio
step_ratio
forward_ratio
backward_ratio
update_ratio
memory_ratio_max
step_ratio_max
S2_pass
S1_pass
failure_reason
top_memory_source
kernel_count_total
materialized_tensor_count
```

### 17.4 判断标准

FullGridS2Pass:

```text
S2 shapes = 9/9
```

where:

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

FullGridS1Pass:

```text
S1 shapes = 9/9
```

where:

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

### 17.5 可视化

```text
p10_memory_heatmap.svg
p10_step_heatmap.svg
p10_s1_s2_boundary.svg
p10_shape_failure_reason.svg
```

---

## 18. P11：scaling and robustness, gated

### 18.1 打开条件

Only open if:

```text
P1 TimeAccountingPass = true
P3 ProfilerPass = true
P9 candidate remains MacroSignificant + S2
```

### 18.2 Settings

Train size:

```text
256
512
1024
1536
4096 if available
```

Noise:

```text
label_noise = 0.05, 0.10, 0.20
input_noise = 0.05, 0.10
random_erasing = small
mild_affine_shift
```

Scaling smoke:

```text
hidden_dim = 48,64,96,128
batch = 128,256,512
optional CIFAR10-small / EMNIST-subset
```

### 18.3 必须记录字段

```text
candidate
dataset
seed
train_size
noise_type
noise_level
hidden_dim
batch_size
val_acc
test_acc
val_loss
test_loss
ECE
NLL
accuracy_drop_vs_clean
sample_efficiency_auc
robustness_auc
memory_ratio
step_ratio
time_to_target_acc
```

### 18.4 判断标准

Sample efficiency pass:

$$
AUC_{data,M12}>AUC_{data,MLP}.
$$

Robustness pass:

$$
AccDrop_{M12}<AccDrop_{MLP}
$$

for at least two settings.

Scaling smoke pass:

$$
\Delta Acc_{\text{macro,val}}\geq0.010,
$$

$$
r_{\text{mem}}\leq1.10,
$$

$$
r_{\text{step}}\leq1.70.
$$

### 18.5 可视化

```text
p11_accuracy_vs_train_size.svg
p11_sample_efficiency_auc.svg
p11_noise_robustness.svg
p11_hidden_scaling_pareto.svg
```

---

## 19. P12：geometry Pareto diagnostic, gated

### 19.1 打开条件

Only open after P9 preserves MacroSignificant + S2/S1.

### 19.2 目的

Geometry 不是 hard gate。P12 只判断 weak geometry regularization 是否改善 ECE/NLL/AUC 而不伤 task/time。

### 19.3 Geometry losses

```text
basis_smoothness
edge_curvature
jacobian_norm
local_lipschitz
coefficient_tv
path_length
```

### 19.4 Lambda sweep

```text
lambda_geo:
  0
  1e-6
  3e-6
  1e-5
  3e-5
  1e-4
```

### 19.5 必须记录字段

```text
candidate
lambda_geo
geometry_loss_type
val_acc
test_acc
ECE
NLL
ValLossAUC_step
ValLossAUC_time
geometry_metric_value
basis_smoothness
edge_curvature
jacobian_norm
local_lipschitz
memory_ratio
step_ratio
```

### 19.6 判断标准

Geometry useful if:

$$
Acc_{\lambda}\geq Acc_{\lambda=0}-0.005
$$

and one of:

$$
ECE_{\lambda}<ECE_{\lambda=0},
$$

$$
NLL_{\lambda}<NLL_{\lambda=0},
$$

$$
ValLossAUC_{\lambda}<ValLossAUC_{\lambda=0}.
$$

Harmful if:

$$
Acc_{\lambda}<Acc_{\lambda=0}-0.02
$$

and no metric improves.

### 19.7 可视化

```text
p12_geometry_pareto_frontier.svg
p12_accuracy_vs_geometry.svg
p12_ece_nll_vs_geometry.svg
p12_val_loss_auc_vs_geometry.svg
```

---

## 20. P13：candidate co-selection and route decision

### 20.1 Survivor types

```text
S0:
  MacroSignificant + Fairness + FullGridS1 + TimeAUCPass + ProfilerPass

S1:
  MacroSignificant + Fairness + FullGridS2 + TimeAUCPass + ProfilerPass

S2:
  MacroSignificant + Fairness + FullGridS2, but TimeAUCPass fail

S3:
  TimeAUCPass but MacroSignificant unstable

S4:
  MLP-distill closes the gap

S5:
  S1 repair hurts macro

S6:
  Profiler attribution incomplete

S7:
  Kernelization improves speed but breaks GradPass

S8:
  Scaling / robustness fail

S9:
  No reproduction
```

### 20.2 必须记录字段

```text
candidate
survivor_type
strict_pass
grad_pass
macro_10seed_pass
fairness_pass
profiler_pass
time_accounting_pass
time_auc_pass
fullgrid_s2_pass
fullgrid_s1_pass
scaling_pass
robustness_pass
memory_ratio_mean
memory_ratio_max
step_ratio_mean
step_ratio_max
forward_ratio_mean
backward_ratio_mean
val_gap_vs_MLP
val_gap_vs_MLP_distill
test_gap_vs_MLP
CI95_low
Holm_p
ECE_delta
NLL_delta
ValLossAUC_time_delta
top_kernel_bottleneck
top_memory_bottleneck
route_recommendation
primary_blocker
next_required_implementation
```

### 20.3 Route cases

```text
R1-FullSystemAdvantage:
  S0. Claim robust system-level PureKAN-NG advantage.

R2-S2TimeAdvantage:
  S1. M12 is system useful but not S1.

R3-QualityOnlyAdvantage:
  S2. Accuracy/NLL advantage real, but wall-clock not closed.

R4-DistillationObjectiveExplainsAdvantage:
  MLP-distill closes the gap.

R5-TimeAccountingIssue:
  Time AUC fail mostly from teacher/validation/logging/sync.

R6-KernelizationNeeded:
  True profiler identifies bottleneck, fused package not yet integrated.

R7-S1MemoryBlocked:
  S1 memory cannot be reduced without hurting macro/GradPass.

R8-NoReproduction:
  M12 10-seed advantage does not reproduce.

R9-ProfilerIncomplete:
  No kernel-cause claim allowed.
```

### 20.4 可视化

```text
p13_system_pareto_accuracy_efficiency_time.svg
p13_survivor_type_dashboard.svg
p13_route_decision_tree.svg
p13_final_scorecard.svg
p13_kernelization_scorecard.svg
```

---

## 21. P14：artifact and failure audit

### 21.1 Required artifacts

```text
run_manifest.json
provenance_audit.csv
candidate_registry.csv
gate_config.json
p0_reproduction.csv
p1_time_accounting.csv
p2_auc_step_time_decomposition.csv
p3_true_kernel_profiler.csv
p4_s1_memory_attribution.csv
p5_dense_preserving_microbench.csv
p6_fused_streaming_package.csv
p7_teacher_time_accounting.csv
p8_optimization_dynamics.csv
p9_10seed_task_time_confirm.csv
p10_fullgrid_efficiency.csv
p11_scaling_robustness.csv
p12_geometry_pareto.csv
p13_candidate_selection.csv
failure_table.csv
route_decision.json
aggregate_decision.json
figures/
```

### 21.2 Failure taxonomy

```text
F1_macro_reproduction_fail
F2_fairness_fail_vs_mlp_distill
F3_time_accounting_incomplete
F4_auc_step_fail
F5_time_auc_fail
F6_true_profiler_unavailable
F7_kernel_count_not_measured
F8_op_level_attribution_incomplete
F9_s1_memory_fail
F10_s1_repair_hurts_macro
F11_grad_correctness_fail
F12_fused_microkernel_fail
F13_streaming_cache_hurts_expression
F14_teacher_cost_dominates
F15_scaling_fail
F16_robustness_fail
F17_geometry_hurts_expression
F18_fake_or_proxy_violation
F19_artifact_missing
```

### 21.3 Required figures

```text
figures/p13_system_pareto_accuracy_efficiency_time.svg
figures/p13_final_scorecard.svg
figures/p13_route_decision_tree.svg
figures/p13_kernelization_scorecard.svg
figures/failure_taxonomy_heatmap.svg
```

---

## 22. 第一轮推荐执行顺序

### Step 1：P1 time accounting reconciliation

先拆 M12 的 wall-clock。当前最大的矛盾是 step ratio 已经低，但 ValLossAUC_time 很差。没有这一步，无法判断该修 kernel、teacher、validation loop，还是 loss dynamics。

### Step 2：P2 AUC_step vs AUC_time

判断 Time AUC fail 是 runtime 还是 optimization dynamics。如果 AUC_step 也差，才打开 P8；否则先做 profiler/kernel。

### Step 3：P3 true kernel launch / op-level profiler

真实测 kernel launch，不再用 proxy。优先看 M12、B0、B2、M12-offline-logits、A2S/D3。

### Step 4：P4 S1 memory attribution

定位 bs256/bs512 memory > 1.00 的具体 live-set source。没有 attribution，不做 S1-cause claim。

### Step 5：P5 dense-preserving microbench

只针对 P3/P4 找到的 top bottleneck 做 fused/streaming microbench。

### Step 6：P6 full package

只有 P5 microkernel 过 GradPass 和 memory/time gate，才集成 full package。

### Step 7：P9 10-seed confirm

任何 final package 都必须重新 10-seed task/time confirm。

---

## 23. 停止条件

### 成功停止

出现以下任一情况，立即复盘：

```text
MacroSignificant + Fairness + FullGridS2 + TimeAUCPass + ProfilerPass
MacroSignificant + Fairness + FullGridS1 + ProfilerPass
MacroSignificant + Fairness + FullGridS1 + TimeAUCPass + ProfilerPass
```

### 失败停止

出现以下任一情况，停止对应路线：

```text
1. 10-seed macro gap < +0.015；
2. MLP-distill 与 M12 gap < +0.005；
3. TimeAccounting unknown_time_fraction > 0.25；
4. true profiler unavailable，且无法替代；
5. fused/streaming microkernel grad_relerr > 1e-4；
6. S1 repair 让 macro gap < +0.0200；
7. TimeAUC 比 MLP 差 > 20%，且 AUC_step 也失败；
8. no-fake/no-proxy audit fail。
```

---

## 24. 成功与失败解释规则

### Case A：Time AUC fail 来自 teacher / validation / logging / sync

如果 train-only AUC 接近 MLP，但 total wall-clock AUC 失败，则路线不是数学失败，而是 training loop / accounting failure。下一步应做 offline teacher、validation cadence、sync removal。

### Case B：AUC_step 失败

如果 M12 每 step 的 loss curve 就比 MLP 差，则说明 final accuracy win 可能是 late-stage / distillation effect，不是 faster convergence。下一步应做 distillation schedule 或 objective dynamics。

### Case C：Profiler 发现 forward kernel fragmentation

如果 small kernel launch / head fragmentation 主导 forward，则继续 fused head / fused linear-SiLU / forward kernel-count trim。

### Case D：S1 memory source 是 hidden-y cache

优先做 fused recompute / streaming y cache，而不是 naive half cache。M15 已经说明 naive half hidden-y 会 GradFail。

### Case E：S1 memory source 是 optimizer state

需要研究 parameter owner / optimizer state layout，不能用 SGD 破坏 task。必须保持 M12 macro gap。

### Case F：S1 repair 成功但 TimeAUCPass 仍失败

说明 memory 不是 wall-clock root cause。继续 P1/P3/P7。

### Case G：M12 keeps quality advantage but never gets TimeAUCPass

可声明 quality-efficient S2 candidate，但不能声明 full wall-clock system advantage。

---

## 25. 最终建议

v7.7 的一句话策略是：

$$
\boxed{
\text{先拆清 M12 的 time-AUC 失败，再用真实 kernel profiler 驱动 dense-preserving fused/streaming package，把 S2-minimum success 推向 S1/time-AUC system success。}
}
$$

当前不是失败，而是阶段升级：

```text
v7.6 minimum success:
  已经达成。

v7.6 formal success:
  未达成，因为 S1 与 TimeAUC fail。

v7.7 主线:
  time accounting + true profiler + dense-preserving fused/streaming kernel + S1 memory closure。
```

下一步不应该回到 loss/classwise/optimizer 小修，而应围绕：

```text
1. ValLossAUC_time 为什么失败；
2. task trace time 与 profiler step 为什么不一致；
3. M12 forward/kernel launch overhead 在哪里；
4. bs256/512 memory 为什么仍高于 1.00；
5. 哪个 fused/streaming kernel 能在不破坏 GradPass 和 macro 的前提下修复。
```

只有这些闭合，才能把：

```text
M12 is S2-FairSystemMinimumSuccess.
```

推进为：

```text
M12 is a FullSystemAdvantage candidate over MLP-AdamW.
```
