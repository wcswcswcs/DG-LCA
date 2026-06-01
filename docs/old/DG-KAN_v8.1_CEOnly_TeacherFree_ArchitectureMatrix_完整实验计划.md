# DG-KAN v8.1 CE-Only Teacher-Free Architecture Matrix：无 Teacher、无改 Loss、纯结构优势验证完整实验计划

> 本计划是对 v8.0/v7.9 之后路线的明确收紧。  
> **v8.1 官方主线不再研究 teacher，不再使用 self-teacher，不再改 loss。**  
> 所有 official success 必须来自：**同样 CE loss、同样无 teacher、同样训练预算、同样 AdamW-family fair setting 下，PureKAN-NG 结构本身相对 MLP-AdamW 的优势**。  
> 任何 external teacher、self-distill、EMA teacher、snapshot teacher、margin loss、calibration loss、NLL-balanced loss、label smoothing、focal loss、classwise loss、sampler 或 class weight 都不得进入 official route。它们不再是 v8.1 的实验主线，也不作为成功标准。

---

## 0. 当前判断

### 0.1 目标是否已经达到

当前尚未达到最终目标。已有结果说明，teacher-free PureKAN-NG 已有稳定正向信号，但还没有跨过官方 hard gate。

当前 official teacher-free baseline 是 `M13`：

```text
external_teacher_used = 0
self_teacher_used = 0
loss_type = CE
strict_pass = 1
grad_pass = 1
code_contract_pass = 1
```

但它没有通过 official macro gate：

```text
M13 macro val gap = +0.017838541666666666
official gate = +0.0200
gap to gate = 0.002161458333333334
```

即：

$$
0.0200-0.017838541666666666=0.002161458333333334.
$$

已有结构候选 `A2S/M9` 在 structural screen 中达到：

$$
\Delta Acc_{\text{macro,val}}=+0.019921875.
$$

它离 gate 只差：

$$
0.0200-0.019921875=0.000078125.
$$

但它仍未稳定跨过 gate，且 full-grid S2 还不稳。因此不能记为成功。

`M12+C3` 可以达到约：

$$
\Delta Acc_{\text{macro,val}}=+0.020572916666666666.
$$

但它使用 external teacher，所以只能作为历史诊断事实，不能进入 v8.1 official route。

### 0.2 当前最准确的科学状态

当前不是 “KAN 没有表达力”，也不是 “只差 optimizer”。更准确地说：

$$
\boxed{
\text{PureKAN-NG teacher-free CE-only 已经接近官方优势门槛，但自主结构 margin 仍不够厚。}
}
$$

现在的 blocker 是：

```text
1. CE-only teacher-free architecture margin 未稳定超过 +0.0200；
2. A2S/M9 已经接近 threshold，但 full-grid S2 不稳；
3. core primitive 候选还没有形成系统性 architecture matrix；
4. TimeAUC / profiler / S1 尚未形成 formal system closure；
5. 之前计划中 teacher/self-distill/loss repair 会污染最终结论，必须从 official route 移除。
```

### 0.3 本轮核心纠偏

从 v8.1 开始，官方目标明确改为：

$$
\boxed{
\text{CE-only}
+
\text{teacher-free}
+
\text{self-teacher-free}
+
\text{loss-unmodified}
+
\text{strict PureKAN}
+
\text{architecture-driven}
}
$$

换句话说，官方 candidate 必须满足：

```text
external_teacher_used = 0
self_teacher_used = 0
distill_loss_used = 0
special_loss_used = 0
margin_loss_used = 0
calibration_loss_used = 0
nll_balanced_loss_used = 0
label_smoothing_used = 0
focal_loss_used = 0
sampler_changed = 0
class_weight_used = 0
loss_type = CE
```

如果某个实验不满足这些条件，它不能进入 official route。v8.1 计划中不再为这些实验设置主线分支。

---

## 1. v8.1 总体目标

v8.1 的目标不是“把某个数字刷过 +0.0200”，而是回答：

$$
\boxed{
\text{PureKAN-NG 的结构本身，在标准 CE 监督下，是否能稳定超过 MLP-AdamW？}
}
$$

更完整地说，v8.1 要证明或证伪：

$$
\boxed{
\exists f_{\text{KAN}}\in\mathcal{F}_{\text{PureKAN}},
\quad
L = CE(y, f_{\text{KAN}}(x)),
\quad
\Delta Acc_{\text{macro,val}}\geq0.0200,
}
$$

同时满足：

$$
\boxed{
\text{StrictPass}
+
\text{GradPass}
+
\text{FullGridS2Pass}
+
\text{NoTeacherNoLossModificationPass}.
}
$$

v8.1 的官方比较对象是：

```text
B0-MLP-AdamW:
  standard MLP
  CE loss
  same train/val/test split
  same seeds
  same train budget

PureKAN candidates:
  CE loss
  no teacher
  no self-teacher
  no loss modification
  strict PureKAN
  manual forward/backward/update
```

### 1.1 Minimum Success

v8.1 minimum success 是：

$$
\boxed{
\text{CodeNativePass}
\land
\text{NoTeacherNoLossModificationPass}
\land
\text{StrictPass}
\land
\text{GradPass}
\land
\text{TeacherFreeCEMacroPass}
\land
\text{FullGridS2Pass}
}
$$

其中 task gate 为：

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

FullGridS2 为：

$$
r_{\text{mem,max}}\leq1.05,
$$

$$
r_{\text{step,max}}\leq1.50,
$$

并且是 9/9 shapes：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
batch sizes = 128, 256, 512
```

### 1.2 Formal System Success

v8.1 formal success 是：

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

ProfilerPass 要求真实 phase-mapped profiler：

```text
kernel_count_total measured
mapped_kernel_time_fraction >= 0.90
unknown_kernel_time_fraction <= 0.10
top3 phase time explain >= 0.70
```

### 1.3 Strong Success

v8.1 strong success 是：

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

## 2. 本轮明确禁止事项

v8.1 是 CE-only teacher-free architecture experiment。因此以下内容不得进入 official route：

### 2.1 禁止 teacher

```text
external teacher
C3 teacher
M12 teacher
MLP teacher
precomputed teacher logits
online teacher forward
teacher KL
teacher distillation
```

官方 candidate 必须满足：

```text
external_teacher_used = 0
external_teacher_logits_used = 0
external_teacher_forward_used = 0
teacher_checkpoint_path = none
teacher_artifact_hash = none
```

### 2.2 禁止 self-teacher

```text
EMA self-teacher
delayed self-teacher
snapshot self-distill
previous-epoch teacher
dual-view teacher logits
SWA logit smoothing
self-distillation KL
```

官方 candidate 必须满足：

```text
self_teacher_used = 0
self_teacher_logits_used = 0
self_teacher_forward_used = 0
```

### 2.3 禁止改 loss

v8.1 official route 只允许：

$$
L = CE(y,p_{\theta}(x)).
$$

不允许：

```text
margin-stabilized CE
NLL-balanced CE
calibration-aware CE
focal loss
label smoothing
classwise loss
target-margin loss
distillation loss
self-distillation loss
geometry loss as training objective
```

对应审计字段必须为：

```text
loss_type = CE
special_loss_used = 0
distill_loss_used = 0
margin_loss_used = 0
calibration_loss_used = 0
nll_balanced_loss_used = 0
label_smoothing_used = 0
focal_loss_used = 0
geometry_loss_used = 0
```

### 2.4 禁止调数据分布来刷 gate

不允许：

```text
class sampler
class weight
oversampling
undersampling
test-based selection
validation split after seeing test
```

允许做 validation robustness audit，但它必须是预注册的统计审计，不得用 test 选择 candidate。

### 2.5 禁止 optimizer 大扫

本轮不把 optimizer 作为主变量。官方主线保持 optimizer fair and fixed：

```text
B0:
  AdamW baseline, CE

KAN:
  ManualAdamW-equivalent update, CE
```

允许记录 optimizer-related diagnostics，例如 update time、optimizer state memory、update ratio，但不允许通过 optimizer sweep 作为主要成功路径。只有在 architecture candidate 已经过 macro gate 后，才允许做 system-level update implementation repair，例如 foreach / fused update / state layout trim；这属于系统实现，不属于训练 recipe 搜索。

---

## 3. 当前问题的本质

### 3.1 不是 optimizer 首要问题

多轮历史已经显示，普通 optimizer / lr / schedule / smoothing / input normalization 没有把 grouped/T3 路线修成 task success，dense KAN / strict head / A2S family 才带来真正 task signal。v7.0 的 dense D3 首次打开 basic accuracy gate，但 FastOpt 后仍不进 S2，说明 update overhead 不是唯一瓶颈。v7.2 的 NoSync / SGD / task repair也没有把 task + efficiency 同时闭合。v7.4-v7.5 的结构候选逐步把表达力迁移到 strict/S2 方向，说明本质在 architecture 和 implementation，而不是普通 optimizer 小调。

因此当前官方判断是：

$$
\boxed{
\text{optimizer 不是 v8.1 主变量。}
}
$$

如果有问题，也是在 system implementation 层面，例如 update state memory、update temp、kernel launch；不是 AdamW hyperparameter 本身。

### 3.2 当前核心是 CE-only architecture margin

M13 已经达到：

$$
\Delta Acc_{\text{macro,val}}=+0.017838541666666666.
$$

A2S/M9 structural screen 达到：

$$
\Delta Acc_{\text{macro,val}}=+0.019921875.
$$

这说明 CE-only teacher-free architecture 已经非常接近 official gate，但 margin 不稳。v8.1 必须回答：

```text
是 evaluation variance？
是 M13 结构略弱？
是 A2S/M9 结构已经对，但需要 full-grid S2 修复？
是 core primitive 里已有结构还没系统比较？
是 representation rank / margin / hard samples 的结构性差距？
```

这些都是 architecture / measurement / system 问题，不是 loss 或 teacher 问题。

### 3.3 为什么还要记录历史 teacher 结果

v8.1 不再研究 teacher，也不使用 teacher。历史 teacher 结果只作为 provenance，不作为实验变量。它唯一的作用是解释为什么我们知道 “只差一点 margin”：M12+C3 曾把同类结构推到 `+0.02057`，说明结构空间附近存在过线函数。但 v8.1 不使用它，也不把它作为候选。

在 v8.1 的 artifact 中，历史 teacher 只出现在：

```text
provenance/history table
excluded_candidate_table
failure/prohibition audit
```

不进入：

```text
candidate selection
route decision
success gate
co-selection
training recipe
```

---

## 4. v8.1 核心假设

### H0：实验代码是否能保证 no-teacher / no-loss-modification

这是前置假设。任何数值成功之前，必须先证明实验系统没有把 teacher 或 special loss 混入 official route。

H0 成立标准：

```text
candidate_factory_audit coverage = 100%
loss_contract_audit coverage = 100%
teacher_contract_audit coverage = 100%
all official candidates external_teacher_used = 0
all official candidates self_teacher_used = 0
all official candidates loss_type = CE
all official candidates special_loss_used = 0
all measured rows join candidate_registry by candidate_id
fake/proxy nonzero count = 0
```

如果 H0 不成立，即使 accuracy 过线，也不能做 scientific claim。

### H1：当前 gap 是否来自 validation estimator

M13 的 test gap 高于 validation gap，但 test 不能用于调参。因此需要验证 validation split 是否低估。

H1 成立标准：

在预注册 validation shards 上：

$$
\Delta Acc_{\text{macro,val-shard-mean}}\geq0.0200,
$$

且：

$$
CI_{95\%,split}^{low}>0.
$$

如果 repeated validation mean 仍低于：

$$
0.0180,
$$

则 validation 不是主因。

### H2：CE-only A2S/M9 是否已经是正确结构，只是 S2/timing 不稳

A2S/M9 的 `+0.019921875` 极接近 gate。H2 假设它代表正确 architecture direction，但需要 10-seed 和 full-grid S2 复核。

H2 成立标准：

10-seed official co-selection 中：

$$
\Delta Acc_{\text{macro,val,A2S/M9}}\geq0.0200,
$$

且：

$$
r_{\text{mem,max}}\leq1.05,
$$

$$
r_{\text{step,max}}\leq1.50.
$$

如果 task 过线但 S2 不过，则是 system candidate 未完成，不是 task direction 错。

### H3：core primitive representation matrix 可以提供 CE-only margin

v8.1 不改 loss，只改结构。候选来自 `dgkan_core` 中已有或可 code-native 实现的 KAN primitives：

```text
SparseInterpKANDense
SparseSplineKANDense
DWM2LiteDense
GEMMNativeDepthwiseMixDense
DWM2Dense
RationalKATV2 / RationalKAT-style variants
A2S/M9 packed generic head
```

H3 成立标准：

某个 CE-only no-teacher primitive candidate 满足：

$$
Acc_{\text{candidate}}-Acc_{\text{M13}}\geq0.0022,
$$

并且：

$$
\Delta Acc_{\text{macro,val,candidate}}\geq0.0200,
$$

$$
r_{\text{mem,max}}\leq1.05,
$$

$$
r_{\text{step,max}}\leq1.50.
$$

### H4：CE-only gap 体现在 representation rank / margin / hard samples

H4 是解释性假设。即使某个 architecture 过线，也要记录它为什么过线。

H4 成立标准之一：

Representation rank improvement：

$$
rank_{\text{candidate}}\geq1.05rank_{\text{M13}}.
$$

Margin improvement：

$$
margin_{p10,\text{candidate}}\geq1.05margin_{p10,\text{M13}}.
$$

Hard-sample improvement：

$$
Acc_{\text{hard-sample,candidate}}-Acc_{\text{hard-sample,M13}}\geq0.02.
$$

如果 task improvement 没有对应 rank/margin/hard-sample improvement，需记录为 mechanism unclear，不得声称已解释。

### H5：S2 漂移来自 timing path，而不是 CE-only architecture 必然过慢

M13 在 v7.8 曾经 FullGridS2 pass，v7.9 主 run step max 漂到 `1.6199`。H5 假设 S2 fail 可能来自 timing path，而不是 model primitive 必然慢。

H5 成立标准：

phase-clean full-grid profiler 中：

$$
r_{\text{step,max,phase-clean}}\leq1.50.
$$

如果 phase-clean pass 而 task-trace fail，则记录为 timing path issue。

### H6：TimeAUC fail 来自 runtime/system，而不是 CE learning dynamics

如果 CE-only KAN 的 step AUC 不差，但 time AUC 差，则不应回到 optimizer/loss，而应修 system path。

H6 成立标准：

$$
ValLossAUC_{\text{step,KAN}}\leq ValLossAUC_{\text{step,MLP}},
$$

但：

$$
ValLossAUC_{\text{time,KAN}}>ValLossAUC_{\text{time,MLP}}.
$$

此时下一步是 phase-mapped profiler / fused package，而不是改 loss 或 optimizer。

---

## 5. Candidate 设计

### 5.1 Baseline candidates

```text
B0-MLP-AdamW-CE
M13-CE-teacher-free-baseline
A2S-CE-teacher-free-nearpass
M9-CE-teacher-free-nearpass
```

历史 excluded diagnostic candidates 只用于 provenance，不参与 co-selection：

```text
B2-MLP-C3-distill-excluded
M12-C3-distill-excluded
```

### 5.2 CE-only architecture matrix candidates

```text
ARCH0-M13-baseline
ARCH1-A2S-packed-generic-head-CE
ARCH2-M9-packed-generic-head-CE
ARCH3-SparseInterpKANDense-CE
ARCH4-SparseSplineKANDense-CE
ARCH5-DWM2LiteDense-rbf-residual-CE
ARCH6-DWM2LiteDense-lut-residual-CE
ARCH7-GEMMNativeDepthwiseMixDense-CE
ARCH8-DWM2Dense-residual-lite-CE
ARCH9-RationalKATV2Dense-identity-residual-CE
ARCH10-feature-rank-preserving-init-CE
ARCH11-local-cross-feature-bridge-lite-CE
```

每个 ARCH candidate 必须满足：

```text
external_teacher_used = 0
self_teacher_used = 0
loss_type = CE
special_loss_used = 0
nonKAN_param_count = 0
manual_forward = 1
manual_backward = 1
manual_update = 1
```

### 5.3 System candidates after architecture pass

系统修复只在 CE-only architecture candidate 过 macro 或 near-pass 后打开：

```text
SYS0-best-CE-architecture-current
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

这些 candidate 不允许改变 loss，不允许加入 teacher，不允许改 task objective。

---

## 6. 实验阶段总览

v8.1 分为五个 Wave：

```text
Wave 0:
  CE-only code contract

Wave 1:
  CE-only evidence reproduction and validation robustness

Wave 2:
  CE-only architecture mechanism matrix

Wave 3:
  official co-selection and system closure

Wave 4:
  generalization validation
```

---

## 7. Wave 0：CE-only code contract

### P0：CandidateFactory + LossContract + TeacherContract

#### 目标

建立不可绕过的 official route 前置门禁。任何 official candidate 必须由 CandidateFactory 构造，并通过 teacher/loss/strict 三重审计。

#### 必须实现

```text
CandidateSpecV3
CandidateFactory
TeacherContractNoTeacher
LossContractCEOnly
StrictPureKANContract
ArtifactJoinValidator
ImplementationStatusRegistry
```

#### 每个 candidate 必须记录

```text
candidate_id
candidate_name
route_role
dense_cls_name
model_family
hidden_dim
basis_count
depth
external_teacher_used
external_teacher_logits_used
external_teacher_forward_used
self_teacher_used
self_teacher_logits_used
self_teacher_forward_used
loss_type
special_loss_used
distill_loss_used
margin_loss_used
calibration_loss_used
nll_balanced_loss_used
label_smoothing_used
focal_loss_used
geometry_loss_used
sampler_changed
class_weight_used
optimizer_family
optimizer_hparams_locked
nonKAN_param_count
manual_forward
manual_backward
manual_update
uses_loss_backward
uses_torch_autograd_graph
strict_pass
official_eligible
```

#### 判断标准

NoTeacherNoLossModificationPass：

```text
external_teacher_used = 0
self_teacher_used = 0
loss_type = CE
special_loss_used = 0
distill_loss_used = 0
margin_loss_used = 0
calibration_loss_used = 0
nll_balanced_loss_used = 0
label_smoothing_used = 0
focal_loss_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
```

StrictPass：

$$
\text{StrictPass}
=
[\text{nonKAN}=0]
\land
[\text{manual\_forward}=1]
\land
[\text{manual\_backward}=1]
\land
[\text{manual\_update}=1]
\land
[\text{uses\_loss\_backward}=0].
$$

#### 必须可视化

```text
p0_candidate_factory_graph.svg
p0_teacher_loss_contract_matrix.svg
p0_strict_contract_heatmap.svg
p0_artifact_join_coverage.svg
```

---

## 8. Wave 1：CE-only evidence reproduction and validation robustness

### P1：CE-only baseline reproduction

#### 目标

用 v8.1 的 CE-only contract 复现当前事实，确认新 runner 没有引入 teacher/loss 污染，也没有改变原始结果。

#### 必跑对象

```text
B0-MLP-AdamW-CE
M13-CE-teacher-free-baseline
A2S-CE-teacher-free-nearpass
M9-CE-teacher-free-nearpass
```

历史 excluded candidate 只读入 provenance，不训练、不 co-select：

```text
M12-C3-excluded
B2-C3-excluded
```

#### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
train/val/test = 1536/512/512
steps = 240
batch task = 128
bench batch = 128,256,512
grad batch = 8,128
bootstrap reps = 10000
```

#### 必须记录

```text
candidate_id
dataset
seed
val_acc
test_acc
val_loss
test_loss
train_acc
train_loss
val_gap_vs_MLP
test_gap_vs_MLP
ECE
AdaptiveECE
NLL
Brier
CI95_low
CI95_high
Holm_p
seed_win_rate
GradPass
grad_relerr_max
grad_cos_min
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

#### 判断标准

M13 reproduction：

$$
|\Delta Acc_{\text{M13}}-0.01784|\leq0.004.
$$

A2S/M9 reproduction：

$$
|\Delta Acc_{\text{A2S/M9}}-0.019921875|\leq0.004.
$$

NoTeacherNoLossModificationPass 必须为 1。

#### 必须可视化

```text
p1_macro_gap_bar.svg
p1_seedwise_gap_boxplot.svg
p1_bootstrap_ci_forest.svg
p1_efficiency_heatmap.svg
p1_val_loss_curve_step.svg
p1_val_loss_curve_time.svg
```

### P2：Validation robustness audit

#### 目标

判断 single validation split 是否低估 CE-only KAN candidate。该阶段不得使用 test selection。

#### 设计

构造 5 或 10 个预注册 validation shards：

```text
test set untouched
balanced by class
fixed shard seeds
same train budget
same CE loss
same candidate set
```

#### 必跑对象

```text
B0
M13
A2S
M9
best_ARCH candidates after P3 smoke, if available
```

#### 必须记录

```text
split_id
candidate_id
dataset
seed
class_distribution
sample_count
val_acc
val_loss
val_gap_vs_MLP
ECE
NLL
macro_gap_split_mean
macro_gap_split_std
CI95_low_split
Holm_p_split
```

#### 判断标准

Validation split effect 成立：

$$
\Delta Acc_{\text{macro,val-shard-mean}}\geq0.0200,
$$

且：

$$
CI_{95\%,split}^{low}>0.
$$

如果：

$$
\Delta Acc_{\text{macro,val-shard-mean}}<0.0180,
$$

则 validation 不是主要原因。

#### 必须可视化

```text
p2_val_gap_by_split.svg
p2_validation_ci_forest.svg
p2_class_distribution_heatmap.svg
p2_original_vs_sharded_validation.svg
```

---

## 9. Wave 2：CE-only architecture mechanism matrix

### P3：Architecture matrix smoke and implementation audit

#### 目标

把 core primitive 真正变成 CE-only official候选，而不是只写在计划里。

#### 必跑对象

```text
ARCH0-M13-baseline
ARCH1-A2S-packed-generic-head-CE
ARCH2-M9-packed-generic-head-CE
ARCH3-SparseInterpKANDense-CE
ARCH4-SparseSplineKANDense-CE
ARCH5-DWM2LiteDense-rbf-residual-CE
ARCH6-DWM2LiteDense-lut-residual-CE
ARCH7-GEMMNativeDepthwiseMixDense-CE
ARCH8-DWM2Dense-residual-lite-CE
ARCH9-RationalKATV2Dense-identity-residual-CE
ARCH10-feature-rank-preserving-init-CE
ARCH11-local-cross-feature-bridge-lite-CE
```

#### 实施要求

每个 candidate 先通过 implementation audit：

```text
dense_cls importable
forward works
manual backward available
manual update available
gradient checker available
nonKAN_param_count = 0
loss_type = CE
external_teacher_used = 0
self_teacher_used = 0
```

#### 必须记录

```text
candidate_id
dense_cls
kernel_path
implementation_status
manual_forward
manual_backward
manual_update
GradPass
grad_relerr_max
nonKAN_param_count
edge_param_count
basis_count
hidden_dim
depth
forward_smoke_pass
backward_smoke_pass
task_smoke_val_acc
task_smoke_val_gap
memory_ratio_smoke
step_ratio_smoke
```

#### 判断标准

ARCH candidate eligible：

```text
implementation_status = implemented
NoTeacherNoLossModificationPass = 1
StrictPass = 1
GradPass = 1
```

没有 manual backward 的 candidate 只能作为 non-official diagnostic，不进入 co-selection。

#### 必须可视化

```text
p3_architecture_implementation_matrix.svg
p3_grad_relerr_lollipop.svg
p3_smoke_task_efficiency_pareto.svg
```

### P4：CE-only architecture task matrix

#### 目标

在不改 loss、不用 teacher 的条件下，系统比较 architecture primitive 是否能提供足够 margin。

#### 设置

第一阶段：

```text
seeds = 0,1,2
steps = 240
datasets = MNIST,Fashion-MNIST,KMNIST
```

第二阶段只保留 top candidates：

```text
seeds = 0..9
```

#### 必须记录

```text
candidate_id
dataset
seed
val_acc
test_acc
val_gap_vs_MLP
test_gap_vs_MLP
ECE
NLL
Brier
ValLossAUC_step
ValLossAUC_time
feature_effective_rank
margin_mean
margin_p10
margin_p50
hard_sample_acc
class_pair_confusion
memory_ratio_max
step_ratio_max
S2_shape_count
GradPass
```

#### 判断标准

Architecture useful：

$$
Acc_{\text{candidate}}-Acc_{\text{M13}}\geq0.0022.
$$

Official macro pass：

$$
\Delta Acc_{\text{macro,val,candidate}}\geq0.0200.
$$

Efficiency preservation：

$$
r_{\text{mem,max}}\leq1.05,
$$

$$
r_{\text{step,max}}\leq1.50.
$$

#### 必须可视化

```text
p4_architecture_gap_bar.svg
p4_task_efficiency_pareto.svg
p4_feature_rank_vs_gap.svg
p4_margin_p10_vs_gap.svg
p4_hard_sample_improvement.svg
p4_class_pair_confusion_delta.svg
```

### P5：Representation and hard-sample attribution

#### 目标

解释为什么某些 architecture 有效，避免只看到 accuracy。

#### 必跑对象

```text
B0
M13
A2S
M9
best_ARCH_top3
```

#### 必须记录

```text
feature_effective_rank
feature_CKA_pairwise
logit_KL_pairwise
margin_mean
margin_p10
margin_p50
confidence_mean
wrong_confidence_mean
hard_sample_overlap_rate
hard_sample_gain_vs_M13
class_pair_confusion_matrix
dataset_gap_contribution
```

#### 判断标准

Representation improvement：

$$
rank_{\text{candidate}}\geq1.05rank_{\text{M13}}.
$$

Margin improvement：

$$
margin_{p10,\text{candidate}}\geq1.05margin_{p10,\text{M13}}.
$$

Hard-sample improvement：

$$
Acc_{\text{hard,candidate}}-Acc_{\text{hard,M13}}\geq0.02.
$$

如果 task improvement 没有任何 mechanism metric 支持，必须写：

```text
mechanism_explanation = unclear
```

#### 必须可视化

```text
p5_feature_cka_heatmap.svg
p5_feature_rank_bar.svg
p5_margin_distribution.svg
p5_hard_sample_overlap_matrix.svg
p5_dataset_gap_contribution.svg
```

---

## 10. Wave 3：Official co-selection and system closure

### P6：10-seed CE-only official co-selection

#### 目标

所有 near-pass architecture 同台 10-seed 对比，避免单分支误判。

#### 必跑对象

```text
B0
M13
A2S
M9
best_ARCH_1
best_ARCH_2
best_ARCH_3
```

#### 必须记录

```text
candidate_id
dataset
seed
val_acc
test_acc
val_gap_vs_MLP
test_gap_vs_MLP
CI95_low
CI95_high
Holm_p
seed_win_rate
ECE
NLL
Brier
GradPass
StrictPass
NoTeacherNoLossModificationPass
memory_ratio_max
step_ratio_max
S2_shape_count
S1_shape_count
ValLossAUC_step
ValLossAUC_time
```

#### 判断标准

Official CE-only pass：

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

同时：

```text
NoTeacherNoLossModificationPass = 1
StrictPass = 1
GradPass = 1
```

FullGridS2Pass：

$$
r_{\text{mem,max}}\leq1.05,
$$

$$
r_{\text{step,max}}\leq1.50.
$$

#### 必须可视化

```text
p6_official_scorecard.svg
p6_seedwise_gap_boxplot.svg
p6_bootstrap_ci_forest.svg
p6_task_efficiency_pareto.svg
p6_route_decision_tree.svg
```

### P7：Phase-clean full-grid S2 profiler

#### 目标

确认 S2 是否真实稳定，并区分 model primitive time 与 task loop overhead。

#### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
batch sizes = 128,256,512
warmup = 50
reps = 200
mode = train_step_only
validation/logging disabled
extra sync disabled except timing sync
```

#### 必须记录

```text
candidate_id
dataset
batch_size
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

#### 判断标准

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

Timing drift detected if：

```text
phase_clean_step_max <= 1.50
task_trace_step_max > 1.50
```

#### 必须可视化

```text
p7_memory_heatmap.svg
p7_step_heatmap.svg
p7_task_trace_vs_phase_clean.svg
p7_step_drift_by_shape.svg
```

### P8：Teacher-free time accounting

#### 目标

解释 step AUC 与 wall-clock AUC 的关系。官方 candidate 不使用 teacher，所以这里不需要 teacher time，只拆 system path。

#### 必须记录

```text
wall_clock_time_total
train_step_time
forward_time
backward_time
update_time
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

#### 判断标准

TimeAccountingPass：

```text
unknown_time_fraction <= 0.10
validation/logging/sync separated
```

TimeAUCPass：

$$
ValLossAUC_{\text{time,total,KAN}}\leq ValLossAUC_{\text{time,total,MLP}}.
$$

If:

$$
ValLossAUC_{\text{step,KAN}}\leq ValLossAUC_{\text{step,MLP}},
$$

but:

$$
ValLossAUC_{\text{time,KAN}}>ValLossAUC_{\text{time,MLP}},
$$

then system runtime is blocker, not optimizer/loss.

#### 必须可视化

```text
p8_time_breakdown_stacked.svg
p8_step_auc_vs_time_auc.svg
p8_train_only_vs_total_auc.svg
p8_time_to_target.svg
```

### P9：Phase-mapped kernel profiler

#### 目标

真实测 kernel launch / kernel time / phase mapping，不再使用 proxy。

#### 必须记录

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
top_kernel_name_2
top_kernel_phase_2
top_kernel_time_2
top_kernel_name_3
top_kernel_phase_3
top_kernel_time_3
small_kernel_count_under_10us
layout_conversion_count
cuda_memcpy_time
cuda_sync_time
```

#### 判断标准

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

#### 必须可视化

```text
p9_kernel_timeline.svg
p9_kernel_count_by_phase.svg
p9_top_kernel_time_bar.svg
p9_mapped_vs_unknown_kernel_time.svg
p9_small_kernel_fragmentation.svg
```

### P10：S1 memory attribution and dense-preserving system package

#### 目标

只在 CE-only macro + S2 candidate 上做 S1 / system repair，不改变 loss，不使用 teacher。

#### 必须记录 memory attribution

```text
root_input_cache_MB
hidden_y_cache_MB
manual_cache_MB
optimizer_state_MB
head_temp_MB
allocator_padding_MB
reserved_unallocated_MB
largest_live_tensor_MB
top1_memory_source
top2_memory_source
top3_memory_source
unknown_memory_fraction
```

Attribution pass：

```text
top3 memory sources identified
unknown_memory_fraction <= 0.10
```

#### System candidates

```text
SYS0-best-CE-architecture-current
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

#### 判断标准

Task preservation：

$$
\Delta Acc_{\text{macro,val,new}}\geq0.0200,
$$

and:

$$
\Delta Acc_{\text{new}}\geq\Delta Acc_{\text{parent}}-0.003.
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

#### 必须可视化

```text
p10_s1_memory_waterfall.svg
p10_system_package_pareto.svg
p10_s1_s2_shape_heatmap.svg
p10_time_auc_repair_bar.svg
p10_kernel_mapping_after_repair.svg
```

---

## 11. Wave 4：Generalization validation

### P11：Sample efficiency

#### 打开条件

```text
TeacherFreeCEMacroPass = 1
FullGridS2Pass = 1
GradPass = 1
```

#### 设置

```text
train_size = 256,512,1024,1536,4096
seeds = 0..9 for primary sizes
```

#### 必须记录

```text
train_size
candidate_id
dataset
seed
val_acc
test_acc
val_loss
test_loss
ECE
NLL
ValLossAUC_step
ValLossAUC_time
sample_efficiency_auc
memory_ratio
step_ratio
```

#### 判断标准

$$
AUC_{\text{data,KAN}}>AUC_{\text{data,MLP}}.
$$

#### 必须可视化

```text
p11_accuracy_vs_train_size.svg
p11_sample_efficiency_auc_bar.svg
p11_nll_vs_train_size.svg
```

### P12：Robustness

#### 设置

```text
label_noise = 0.05,0.10,0.20
input_noise = 0.05,0.10
random_erasing = small
affine_shift = mild
```

#### 必须记录

```text
noise_type
noise_level
clean_acc
noisy_acc
accuracy_drop
ECE_under_noise
NLL_under_noise
robustness_auc
memory_ratio
step_ratio
```

#### 判断标准

$$
AccDrop_{\text{KAN}}<AccDrop_{\text{MLP}}
$$

for at least two settings.

#### 必须可视化

```text
p12_accuracy_under_noise.svg
p12_accuracy_drop_bar.svg
p12_robustness_auc.svg
p12_ece_under_noise.svg
```

### P13：Geometry Pareto diagnostic

Geometry 只做 diagnostic，不作为 official hard gate，不进入 loss 训练主线。只允许 post-hoc 或 evaluation metric 记录。

#### 必须记录

```text
basis_smoothness
edge_curvature
jacobian_norm
local_lipschitz
path_length
val_acc
ECE
NLL
ValLossAUC_step
ValLossAUC_time
memory_ratio
step_ratio
```

#### 判断标准

Geometry Pareto useful：

$$
Acc_{\text{KAN}}\geq Acc_{\text{baseline}}-0.005,
$$

且至少一个指标改善：

$$
ECE_{\text{KAN}}<ECE_{\text{baseline}},
$$

或：

$$
NLL_{\text{KAN}}<NLL_{\text{baseline}},
$$

或：

$$
ValLossAUC_{\text{KAN}}<ValLossAUC_{\text{baseline}}.
$$

#### 必须可视化

```text
p13_geometry_pareto_frontier.svg
p13_accuracy_vs_geometry.svg
p13_ece_nll_vs_geometry.svg
```

---

## 12. Route decision

v8.1 route 必须明确区分成功类型，不能混淆 teacher / loss / structure。

### 12.1 Survivor types

```text
S0:
  CE-only teacher-free architecture + S1 + TimeAUC + ProfilerPass

S1:
  CE-only teacher-free architecture + S2 + TimeAUC

S2:
  CE-only teacher-free architecture + S2, but TimeAUC fail

S3:
  CE-only teacher-free macro pass, but S2 fail

S4:
  CE-only near-pass around +0.018 to +0.020

S5:
  external teacher or self teacher involved, excluded from official route

S6:
  special loss involved, excluded from official route

S7:
  GradFail

S8:
  code/contract incomplete

S9:
  no reproduction
```

### 12.2 Route cases

```text
R1-CEOnlyTeacherFreeFullSystemAdvantage:
  S0. Official success.

R2-CEOnlyTeacherFreeS2QualityAdvantage:
  S2. Quality/S2 success, system time still open.

R3-CEOnlyMacroPassButSystemBlocked:
  Macro pass but S2/S1/TimeAUC blocked.

R4-CEOnlyNearPass:
  Candidate remains between +0.018 and +0.020.

R5-ExcludedTeacherOrLossSuccess:
  Some excluded candidate passes, but official route fails.

R6-CodeContractFail:
  Cannot make scientific claim.

R7-NoReproduction:
  M13/A2S/M9 signal collapses.
```

### 12.3 route_decision.json 必须记录

```text
candidate_id
route
loss_type
external_teacher_used
self_teacher_used
special_loss_used
strict_pass
grad_pass
no_teacher_no_loss_pass
teacher_free_ce_macro_pass
fullgrid_s2_pass
fullgrid_s1_pass
time_auc_pass
profiler_pass
code_native_pass
val_gap_vs_MLP
test_gap_vs_MLP
CI95_low
Holm_p
memory_ratio_max
step_ratio_max
primary_blocker
next_required_implementation
```

---

## 13. 必须落盘 artifact

```text
run_manifest.json
candidate_registry_v8_1.csv
candidate_factory_audit.csv
teacher_contract_no_teacher.csv
loss_contract_ce_only.csv
strict_contract_validator.csv
artifact_join_coverage.csv
implementation_status_registry.csv

p0_contract_audit.csv
p1_reproduction.csv
p2_validation_robustness.csv
p3_architecture_implementation_audit.csv
p4_architecture_task_matrix.csv
p5_representation_attribution.csv
p6_official_co_selection.csv
p7_phase_clean_fullgrid.csv
p8_time_accounting.csv
p9_phase_mapped_profiler.csv
p10_s1_system_package.csv
p11_sample_efficiency.csv
p12_robustness.csv
p13_geometry_pareto.csv

route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

failure taxonomy：

```text
F1_teacher_used_official_violation
F2_self_teacher_used_official_violation
F3_special_loss_used_official_violation
F4_ce_only_macro_fail
F5_validation_split_underpowered
F6_architecture_matrix_fail
F7_grad_fail
F8_s2_fail
F9_s1_fail
F10_time_auc_fail
F11_profiler_incomplete
F12_code_contract_fail
F13_fake_or_proxy_violation
F14_artifact_missing
F15_not_implemented
```

---

## 14. 第一轮执行安排

第一轮不要再做一个小 probe，而要直接跑一个 CE-only architecture matrix slice。

### Slice 1：contract and reproduction

执行：

```text
P0 + P1
```

目的：

```text
确认 no teacher / no loss contract；
复现 M13/A2S/M9 near-pass；
确认新 runner 不引入污染。
```

停止条件：

```text
M13 reproduction gap < +0.015
或 contract fail
```

### Slice 2：validation robustness

执行：

```text
P2
```

目的：

```text
判断 +0.01784 / +0.019921875 是否被 single validation split 低估。
```

### Slice 3：architecture matrix

执行：

```text
P3 + P4 + P5
```

目的：

```text
系统比较 core primitive，不改 loss，不用 teacher。
```

### Slice 4：official co-selection

执行：

```text
P6
```

目的：

```text
所有 near-pass candidates 同台 10-seed 比较。
```

### Slice 5：system closure

执行：

```text
P7-P10
```

只对 P6 胜出的 CE-only teacher-free candidate 执行。

### Slice 6：generalization

执行：

```text
P11-P13
```

只在 minimum success 后打开。

---

## 15. 停止条件

### 15.1 成功停止

出现以下任一情况立即复盘：

```text
CE-only teacher-free candidate:
  macro gap >= +0.0200
  CI95 low > 0
  Holm p < 0.05
  GradPass = 1
  FullGridS2 = 1
```

Formal success：

```text
Minimum success + FullGridS1 + TimeAUCPass + ProfilerPass
```

### 15.2 失败停止

出现以下任一情况停止对应路线：

```text
1. official candidate uses external teacher；
2. official candidate uses self teacher；
3. official candidate uses non-CE / special loss；
4. M13 reproduction gap < +0.015；
5. repeated validation mean gap < +0.018；
6. all architecture candidates improve < +0.001；
7. all near-pass candidates fail GradPass；
8. all CE-only macro pass candidates fail S2；
9. code contract / no-fake / no-proxy audit fail。
```

---

## 16. 最终解释规则

### Case A：CE-only architecture crosses +0.020 and S2

这是 v8.1 最干净成功。可以写：

```text
PureKAN-NG architecture itself establishes a CE-only teacher-free advantage over MLP-AdamW.
```

### Case B：CE-only architecture crosses +0.020 but S2 fails

说明 task architecture 成立，但 system implementation 未闭合。下一步进入 kernel-native / S2 repair，不再改 loss。

### Case C：Only teacher or self-teacher candidate passes

不算 official success。写：

```text
Excluded teacher/self-teacher route can pass, but architecture-only goal remains unproven.
```

### Case D：Only special loss candidate passes

不算 official success。写：

```text
Loss recipe can pass, but CE-only architecture advantage remains unproven.
```

### Case E：All CE-only candidates remain +0.018 to +0.0199

说明路线非常接近，但尚未科学闭合。下一步应继续 architecture-level design，而不是 teacher/loss repair。

### Case F：Validation robustness passes

如果 repeated validation pass 但 original split fail，需要预注册新 validation protocol，不能 retroactively 宣称 old run 成功。

---

## 17. 最终建议

v8.1 的一句话策略是：

$$
\boxed{
\text{停止 teacher/loss 路线，集中做 CE-only teacher-free architecture matrix。}
}
$$

本轮要证明的不是：

```text
KAN + teacher 能不能赢；
KAN + special loss 能不能赢；
KAN + self-distill 能不能赢。
```

而是：

```text
KAN 结构本身，在标准 CE 和无 teacher 条件下，能不能稳定赢 MLP。
```

因此 v8.1 的最短路径是：

```text
1. no-teacher/no-loss contract；
2. M13/A2S/M9 reproduction；
3. validation robustness；
4. core primitive architecture matrix；
5. representation / hard-sample attribution；
6. 10-seed co-selection；
7. S2/S1/TimeAUC/profiler system closure。
```

只有这样，最终结论才干净：

$$
\boxed{
\text{PureKAN-NG 自身系统性优于 MLP-AdamW。}
}
