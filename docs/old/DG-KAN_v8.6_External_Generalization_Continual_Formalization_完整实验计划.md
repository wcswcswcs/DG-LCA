# DG-KAN v8.6 External Generalization and Continual Formalization：从 v8.5 Route-Level Strong Success 到跨任务公平结论的完整实验计划

> 本计划基于 v8.5 KANbeFair External-Fair Functional Advantage 的最新复盘制定。  
> v8.5 已经按文档判据完成 route-level strong success：SourceAudit、KANbeFair baseline reproduction、Adapter、Counter、PrimaryTransfer、ParameterFair、FLOPsFair、WallclockMemory、FunctionalCausality、Symbolic、Continual 均为 true，`success_v85_minimum = true`，`success_v85_external_formal = true`，`success_v85_strong = true`。  
> 但这不是“DG-KAN 已经在所有外部任务全面优于 MLP”的最终结论。v8.5 的强成功主要建立在 KANbeFair MNIST full protocol、symbolic stress 和当前 digit-group continual stress 上。v8.6 的任务是把这个结果从“当前文档判据 strong success”升级为“跨任务、跨公平 envelope、机制明确、边界清楚”的外部结论。

---

## 0. 当前状态判断

### 0.1 按 v8.5 文档判据，目标已经达成

v8.5 最新 route-level 状态为：

```text
SourceAuditPass = true
KANbeFair baseline reproduction pass = true
AdapterPass = true
CounterPass = true
PrimaryTransferPass = true
ParameterFairPass = true
FLOPsFairPass = true
WallclockMemoryPass = true
FunctionalCausalityPass = true
SymbolicPass = true
ContinualPass = true
success_v85_minimum = true
success_v85_external_formal = true
success_v85_strong = true
```

因此，如果问题是：

```text
v8.5 文档定义的 minimum / external formal / strong route 是否完成？
```

答案是：

$$
\boxed{
\text{完成。}
}
$$

### 0.2 但终极科学目标还没有完全完成

v8.5 的 strong success 有边界。它证明的是：

```text
DG1-FT7-KW6 hidden28 stride128
在 KANbeFair MNIST full protocol 下，
同时通过 primary、parameter、FLOPs、memory、wall-clock gate，
保留 functional geometry advantage，
并且通过当前 symbolic 和 digit-group continual stress artifact。
```

但它还不能推出：

```text
DG-KAN 在 KANbeFair 全部任务族上全面优于 MLP；
DG-KAN 已经解决所有 class-incremental continual learning；
functional update 在所有任务上都带来 task/geometry/robustness 一致收益；
当前 manual implementation 在复杂视觉、NLP、audio 上也会保持 wall-clock fair。
```

因此，v8.6 必须把结论从“一个外部框架下的 route-level strong pass”扩展成“任务族边界清楚的外部公平结论”。

### 0.3 当前最准确的问题定位

v8.5 之前的 blocker 是：

```text
外部公平 envelope 未完成；
hidden28 通过 params/FLOPs 后 wall-clock 失败；
functional causality / symbolic / continual stress 未闭合。
```

v8.5 之后，这些 blocker 在当前文档判据下已经被推进：

```text
stride128 / no-sync / loss-sync trim / paired guard forward / pre-holdout pairing
  使 wall-clock gate 闭合；

P9 functional causality controls
  证明 functional geometry 不是 no-op / random artifact；

P10 symbolic
  证明 functional geometry 在 symbolic family 有真实收益；

P11 anti-forgetting update rule
  让当前 digit-group continual stress 通过。
```

但新的核心问题是：

$$
\boxed{
\text{DG-KAN + functional update 的优势到底有多广，边界在哪里？}
}
$$

这正好对应 KANbeFair 论文的挑战：在参数量和 FLOPs 公平条件下，KAN 通常只在 symbolic formula representation 上优于 MLP，而在机器学习、视觉、NLP、音频任务上多数情况下弱于 MLP。v8.6 必须直接面对这个外部反证压力。

---

## 1. v8.6 总体目标

v8.6 的总体目标是：

$$
\boxed{
\text{把 v8.5 的 route-level strong success 扩展为跨任务、公平、可复现、边界清楚的 DG-KAN functional advantage。}
}
$$

v8.6 不再追求单一 MNIST full protocol 的成功，而要回答六个问题：

```text
Q1:
  v8.5 accepted route 是否能 fresh reproduce？
  不是复用上轮 artifact，而是在独立 out-dir 中重跑关键链条。

Q2:
  DG-KAN 在 KANbeFair 的更多任务族上是否仍然优于 MLP？
  包括 vision、symbolic、tabular / machine learning、NLP、audio、continual learning。

Q3:
  在 parameter-matched、FLOPs-matched、wall-clock-aware 三种公平 envelope 下，优势是否仍存在？

Q4:
  functional update 的几何优势是否跨任务存在？
  它是否真的关联 ECE/NLL/robustness/forgetting，而不是只降低内部 curvature metric？

Q5:
  P11 continual pass 是否只是当前 digit-group stress 的局部修复？
  old-class head-row restore + stack anchor 是否会产生偏向早期 task 的副作用？

Q6:
  如果 DG-KAN 只在 symbolic / geometry-sensitive / MNIST-like tasks 上优于 MLP，是否能明确写出边界，而不是继续强行宣称全面优越？
```

---

## 2. 继续坚持的硬约束

v8.6 继续严格禁止：

```text
external teacher
self teacher
teacher logits
distillation
self-distillation
label smoothing
focal loss
margin loss
calibration loss
NLL-balanced loss
geometry loss as training objective
sampler / class weight
oversampling / undersampling
test-based selection
CPU offload
optimizer hyperparameter sweep
```

允许 functional update，但必须是 update rule，不是 loss：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

允许的 functional update 形式为：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{\text{AdamW-equivalent}}
+
\lambda_t\Delta\theta_{\text{functional}}.
$$

不允许改写为：

$$
L=CE+\lambda L_{\text{geo}}.
$$

所有 official candidate 必须记录：

```text
loss_type = CE
geometry_loss_used = 0
external_teacher_used = 0
self_teacher_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
uses_loss_backward = 0
fake_data_used = 0
proxy_row_used = 0
```

---

## 3. 核心假设

### H0：v8.5 accepted route 可独立复现

H0 是前置假设。如果 v8.5 的 strong route 不能在 fresh run 复现，v8.6 不能继续扩大结论。

H0 成立标准：

在 fresh out-dir 中，`DG1-FT7-KW6 hidden28 stride128` 满足：

```text
PrimaryTransferPass = 1
ParameterFairPass = 1
FLOPsFairPass = 1
WallclockMemoryPass = 1
FunctionalCausalityPass = 1
```

核心数值要求：

$$
Acc_{\text{DG-Functional}}\geq Acc_{\text{KB-MLP}},
$$

$$
Params_{\text{DG-Functional}}\leq1.05Params_{\text{KB-MLP}},
$$

$$
FLOPs_{\text{DG-Functional}}\leq1.05FLOPs_{\text{KB-MLP}},
$$

$$
StepTime_{\text{DG-Functional}}\leq1.50StepTime_{\text{KB-MLP}},
$$

$$
R_{\text{curv,DG-Functional}}\leq0.90R_{\text{DG-Base}}.
$$

若 H0 不成立，v8.6 route 降级为：

```text
R8-NoReproduction
```

---

### H1：DG-KAN 的优势不是 MNIST-only

H1 检查 v8.5 的成功是否能扩展到多个 KANbeFair task family。

任务族至少包括：

```text
Vision:
  MNIST, Fashion-MNIST, KMNIST, CIFAR-10 if available in KANbeFair

Machine learning / tabular:
  at least two KANbeFair tabular datasets

Symbolic:
  KANbeFair symbolic function tasks

NLP:
  one text classification task if KANbeFair runnable

Audio:
  one audio classification task if KANbeFair runnable

Continual:
  digit-group and at least one alternative class-incremental split
```

H1 成立标准：

在至少两个非 symbolic task family 上：

$$
Acc_{\text{DG-Functional}}\geq Acc_{\text{KB-MLP}}.
$$

并且：

$$
Params_{\text{DG-Functional}}\leq1.05Params_{\text{KB-MLP}}
$$

或：

$$
FLOPs_{\text{DG-Functional}}\leq1.05FLOPs_{\text{KB-MLP}}.
$$

如果只在 MNIST / symbolic 上成立，则 route 写为：

```text
R4-SymbolicOrMNISTOnlyAdvantage
```

---

### H2：functional update 的几何收益跨任务存在

H2 不只要求 FT7 在 MNIST 上降低 curvature，还要求几何收益跨任务稳定。

H2 成立标准：

在至少 70% 的 tested tasks 上：

$$
R_{\text{curv,DG-Functional}}\leq0.90R_{\text{DG-Base}}.
$$

并且在至少 50% 的 tested tasks 上，几何收益伴随一个下游指标不劣：

$$
ECE_{\text{DG-Functional}}\leq ECE_{\text{DG-Base}},
$$

或：

$$
NLL_{\text{DG-Functional}}\leq NLL_{\text{DG-Base}},
$$

或：

$$
AccDrop_{\text{DG-Functional}}\leq AccDrop_{\text{DG-Base}}.
$$

如果 curvature 下降但所有下游指标不改善，route 写为：

```text
R5-GeometryOnlyNoGeneralization
```

---

### H3：P11 continual pass 不是偶然的 digit-group trick

v8.5 P11 的 pass 来自 update-rule 侧 anti-forgetting guard：

```text
old-class head-row restore
stack anchor strength
stride16 continual event
```

它没有 replay dataset，没有 teacher，没有 sampler/class weight，也没有改 CE loss。H3 要求这个机制在多个 class-incremental splits 上成立，并且不要只是偏向第一个 task。

H3 成立标准：

对至少三个 continual splits：

```text
split A: digits 0-2 / 3-5 / 6-9
split B: digits 0-4 / 5-9
split C: random balanced 3-task split
```

满足：

$$
Forgetting_{\text{DG-Functional}}\leq Forgetting_{\text{DG-Base}},
$$

and:

$$
FinalAvgAcc_{\text{DG-Functional}}\geq FinalAvgAcc_{\text{DG-Base}}.
$$

还必须满足 balanced criterion：

$$
\max_i Acc_i-\min_i Acc_i \leq 0.20.
$$

如果 forgetting 降低但只靠牺牲后续 task 或强偏向早期 task，则 H3 不成立。

---

### H4：external fair success 必须同时检查 parameters、FLOPs 与 wall-clock

H4 是 KANbeFair 的核心压力测试。不能只说参数少，不能只说 FLOPs 少，也不能只说 task acc 高。

H4 成立标准：

对每个任务族至少报告三条 envelope：

```text
parameter-matched
FLOPs-matched
wall-clock / memory-aware
```

至少一个 strong route 要同时满足：

$$
Params_{\text{DG}}\leq1.05Params_{\text{MLP}},
$$

$$
FLOPs_{\text{DG}}\leq1.05FLOPs_{\text{MLP}},
$$

$$
StepTime_{\text{DG}}\leq1.50StepTime_{\text{MLP}},
$$

$$
PeakMemory_{\text{DG}}\leq1.05PeakMemory_{\text{MLP}},
$$

and:

$$
Acc_{\text{DG}}\geq Acc_{\text{MLP}}.
$$

---

### H5：如果 KANbeFair 更广任务下 MLP 仍更强，必须承认边界

H5 是反证假设。v8.6 允许 negative result。

如果在多数非 symbolic tasks 上：

$$
Acc_{\text{DG-Functional}}<Acc_{\text{KB-MLP}}-0.01,
$$

即使 symbolic / MNIST 仍然成功，也必须写：

```text
DG-KAN functional advantage is task-family-limited.
```

这不是失败，而是边界清楚化。

---

## 4. Candidate 设计

### 4.1 Baselines

```text
KB-MLP:
  KANbeFair MLP baseline

KB-KAN:
  KANbeFair official KAN baseline

KB-BSpline-MLP:
  B-spline activation MLP if available

DG-Base:
  DG0-KW6 hidden28 stride128 base

DG-Functional:
  DG1-FT7-KW6 hidden28 stride128 accepted route

DG-NoOp:
  matched-overhead no-op control

DG-RandomFunc:
  random functional direction control

DG-ShuffledRoleFunc:
  shuffled role functional direction control
```

### 4.2 Continual variants

```text
CL0-DG-Base
CL1-DG-Functional-accepted
CL2-DG-Functional-no-old-head-restore
CL3-DG-Functional-no-stack-anchor
CL4-DG-Functional-stack-anchor-006
CL5-DG-Functional-stack-anchor-012
CL6-DG-Functional-stack-anchor-024
```

### 4.3 Fair envelope variants

```text
E0-default-KANbeFair
E1-parameter-matched
E2-FLOPs-matched
E3-wallclock-memory-aware
E4-joint-params-flops-wallclock
```

---

## 5. 实验阶段总览

v8.6 分为九个 Wave：

```text
Wave 0:
  v8.5 accepted route reproduction and contract audit

Wave 1:
  KANbeFair broader baseline reproduction

Wave 2:
  DG-KAN adapter hardening and counter validation

Wave 3:
  external multi-task fair evaluation

Wave 4:
  functional causality across tasks

Wave 5:
  continual learning formalization

Wave 6:
  system profiling and implementation bottleneck analysis

Wave 7:
  mechanism and boundary analysis

Wave 8:
  final route decision and artifact audit
```

---

## 6. Wave 0：v8.5 accepted route reproduction and contract audit

### P0：fresh reproduction of accepted v8.5 route

#### 目标

独立复现 v8.5 accepted route，确认不是一次性 artifact。

#### 必跑

```text
KB-MLP
DG-Base hidden28 stride128
DG-Functional hidden28 stride128
DG-NoOp
DG-RandomFunc
```

#### 必须记录

```text
candidate_id
dataset
seed
test_acc
delta_vs_KB_MLP
delta_vs_DG_Base
params
FLOPs
peak_memory_MB
step_time_ms
train_time_s
curvature_ratio_vs_DG_Base
functional_events
functional_update_time_ratio
primary_transfer_pass
parameter_fair_pass
flops_fair_pass
wallclock_pass
functional_causality_pass
fake_data_used
proxy_row_used
cpu_offload_used
```

#### 判断标准

FreshReproPass：

```text
primary_transfer_pass = 1
parameter_fair_pass = 1
flops_fair_pass = 1
wallclock_pass = 1
functional_causality_pass = 1
```

#### 可视化

```text
p0_reproduction_scorecard.svg
p0_task_system_geometry_pareto.svg
p0_step_time_repeat_boxplot.svg
p0_curvature_control_bar.svg
```

---

## 7. Wave 1：KANbeFair broader baseline reproduction

### P1：baseline reproduction across task families

#### 目标

复现 KANbeFair 自身 baseline，避免外部任务失败来自框架接入问题。

#### 必跑任务

```text
MNIST-family vision tasks
one or more tabular / machine learning datasets
symbolic formula representation
one NLP task if runnable
one audio task if runnable
continual learning protocol
```

#### 必须记录

```text
task_family
task_name
dataset_name
model_name
reported_metric
measured_metric
absolute_delta_from_reported
relative_delta_from_reported
params
FLOPs
train_time
test_time
peak_memory
seed
reproduction_pass
```

#### 判断标准

Classification reproduction：

$$
|Acc_{\text{repro}}-Acc_{\text{reported}}|\leq0.02.
$$

Symbolic reproduction：

$$
\frac{|RMSE_{\text{repro}}-RMSE_{\text{reported}}|}
{RMSE_{\text{reported}}}\leq0.20.
$$

#### 可视化

```text
p1_reproduction_vs_paper.svg
p1_task_family_reproduction_heatmap.svg
p1_params_flops_reproduction_table.md
```

---

## 8. Wave 2：DG-KAN adapter hardening and counter validation

### P2：adapter contract and manual path audit

#### 目标

确保 DG-KAN 在 KANbeFair 多任务输入输出形态下仍满足 PureKAN / functional contract。

#### 必须记录

```text
task_name
input_shape
output_shape
candidate_id
adapter_class
loss_type
metric_type
external_teacher_used
self_teacher_used
geometry_loss_used
sampler_changed
class_weight_used
cpu_offload_used
manual_forward
manual_backward
manual_update
uses_loss_backward
nonKAN_param_count
fake_data_used
proxy_row_used
adapter_pass
```

#### 判断标准

AdapterPass：

```text
loss_type is task-native standard objective
geometry_loss_used = 0
external_teacher_used = 0
self_teacher_used = 0
manual_forward = 1
manual_backward = 1
manual_update = 1
uses_loss_backward = 0
fake/proxy = 0
```

### P3：params / FLOPs / wall-clock counter validation

#### 目标

避免公平比较的 counter 自说自话。DG-KAN 的参数/FLOPs必须与 KANbeFair 计数口径对齐。

#### 必须记录

```text
model_name
task_name
params_total
params_trainable
params_KAN_edge
params_nonKAN
forward_FLOPs_formula
forward_FLOPs_counted
backward_FLOPs_estimate
peak_memory_MB
step_time_ms
train_time_s
counter_scope
counter_pass
```

#### 判断标准

CounterPass：

```text
params available for all models
FLOPs available for all models
counter scope explicitly documented
DG and KB baselines use same counting convention
```

#### 可视化

```text
p3_params_vs_acc.svg
p3_flops_vs_acc.svg
p3_time_memory_pareto.svg
p3_counter_scope_table.md
```

---

## 9. Wave 3：external multi-task fair evaluation

### P4：primary multi-task transfer

#### 目标

检查 DG-KAN 在更多 KANbeFair 任务上是否仍优于 MLP。

#### 必跑模型

```text
KB-MLP
KB-KAN
DG-Base
DG-Functional
DG-NoOp
DG-RandomFunc
```

#### 必须记录

```text
task_family
task_name
model
seed
params
FLOPs
train_time
step_time
peak_memory
val_metric
test_metric
delta_vs_KB_MLP
delta_vs_KB_KAN
delta_vs_DG_Base
ECE
NLL
curvature
jacobian_norm
local_lipschitz
functional_events
```

#### 判断标准

TaskFamilyPass：

$$
Metric_{\text{DG-Functional}}\geq Metric_{\text{KB-MLP}}
$$

for accuracy tasks, or:

$$
RMSE_{\text{DG-Functional}}\leq RMSE_{\text{KB-MLP}}
$$

for symbolic regression.

#### 可视化

```text
p4_task_family_win_loss_heatmap.svg
p4_delta_vs_mlp_by_task.svg
p4_task_geometry_pareto.svg
p4_task_system_pareto.svg
```

### P5：joint fairness envelope

#### 目标

做真正公平比较，而不是单点对比。

#### 每个任务必须构造

```text
parameter-matched model pair
FLOPs-matched model pair
wallclock-memory-aware model pair
joint envelope if possible
```

#### 必须记录

```text
task_name
envelope_type
model
params_ratio_vs_MLP
FLOPs_ratio_vs_MLP
step_ratio_vs_MLP
memory_ratio_vs_MLP
metric_delta_vs_MLP
pass
```

#### 判断标准

JointFairPass：

$$
Params_{\text{DG}}\leq1.05Params_{\text{MLP}},
$$

$$
FLOPs_{\text{DG}}\leq1.05FLOPs_{\text{MLP}},
$$

$$
StepTime_{\text{DG}}\leq1.50StepTime_{\text{MLP}},
$$

$$
Memory_{\text{DG}}\leq1.05Memory_{\text{MLP}},
$$

and:

$$
Metric_{\text{DG}}\geq Metric_{\text{MLP}}.
$$

#### 可视化

```text
p5_acc_vs_params_envelope.svg
p5_acc_vs_flops_envelope.svg
p5_acc_vs_wallclock_envelope.svg
p5_joint_fair_pass_matrix.svg
```

---

## 10. Wave 4：functional causality across tasks

### P6：functional causality on every task family

#### 目标

确认 FT7 不只是 MNIST 上的 functional artifact。

#### 必跑

```text
DG-Base
DG-Functional
DG-NoOp
DG-RandomFunc
DG-ShuffledRoleFunc
```

#### 必须记录

```text
task_name
seed
test_metric
delta_vs_base
curvature_ratio
jacobian_ratio
local_lipschitz_ratio
ECE_delta
NLL_delta
functional_update_time_ratio
bad_step_rate
holdout_descent_ratio
role_accept_rate
role_geometry_delta
causality_pass
```

#### 判断标准

CausalityPass：

$$
R_{\text{curv,DG-Functional}}<R_{\text{curv,DG-NoOp}},
$$

$$
R_{\text{curv,DG-Functional}}<R_{\text{curv,DG-RandomFunc}},
$$

and:

$$
Metric_{\text{DG-Functional}}\geq Metric_{\text{DG-NoOp}}-\epsilon.
$$

其中 classification 任务：

$$
\epsilon=0.002
$$

或 0.2 percentage point。

#### 可视化

```text
p6_causality_curvature_bar_by_task.svg
p6_noop_random_control_matrix.svg
p6_role_contribution_by_task.svg
p6_metric_vs_curvature_scatter.svg
```

---

## 11. Wave 5：continual learning formalization

### P7：multi-split continual learning stress

#### 目标

验证 v8.5 P11 的 anti-forgetting update-rule 不是单一 digit split 的偶然结果。

#### 必跑 splits

```text
Split A:
  digits 0-2 / 3-5 / 6-9

Split B:
  digits 0-4 / 5-9

Split C:
  random balanced 3-task split

Split D:
  interleaved hard split, e.g. 0,3,6 / 1,4,7 / 2,5,8,9
```

#### 必跑 candidates

```text
KB-MLP
KB-KAN
DG-Base
DG-Functional accepted
DG-Functional no-old-head-restore
DG-Functional no-stack-anchor
DG-Functional stack-anchor-006
DG-Functional stack-anchor-012
DG-Functional stack-anchor-024
```

#### 必须记录

```text
split_id
task_id
candidate
acc_after_each_task
final_task_acc_vector
final_avg_acc
forgetting_score
backward_transfer
forward_transfer
early_task_bias
late_task_bias
max_min_task_acc_gap
curvature_after_task
functional_event_count
anchor_strength
old_head_restore_used
```

#### 判断标准

ContinualFormalPass：

$$
Forgetting_{\text{DG-Functional}}\leq Forgetting_{\text{DG-Base}},
$$

$$
FinalAvgAcc_{\text{DG-Functional}}\geq FinalAvgAcc_{\text{DG-Base}},
$$

and:

$$
\max_i Acc_i-\min_i Acc_i\leq0.20.
$$

If forgetting improves but imbalance exceeds 0.20, route is:

```text
R5-ForgettingReducedButUnbalanced
```

#### 可视化

```text
p7_continual_accuracy_matrix.svg
p7_forgetting_by_split.svg
p7_final_task_balance_bar.svg
p7_anchor_strength_pareto.svg
p7_curvature_forgetting_scatter.svg
```

---

## 12. Wave 6：system profiling and implementation bottleneck analysis

### P8：external task time accounting

#### 目标

确认 DG-KAN 的 wall-clock pass 不只是 MNIST stride128 下的局部现象。

#### 必须记录

```text
task_name
candidate
wall_clock_total
train_step_time
forward_time
backward_time
base_update_time
functional_update_time
guard_forward_time
validation_time
logging_time
cuda_sync_time
unknown_time_fraction
ValLossAUC_time
TimeToTarget
```

#### 判断标准

TimeAccountingPass：

$$
unknown\_time\_fraction\leq0.10.
$$

WallClockPass：

$$
StepTime_{\text{DG}}\leq1.50StepTime_{\text{MLP}}.
$$

and:

$$
ValLossAUC_{\text{time,DG}}\leq1.05ValLossAUC_{\text{time,MLP}}.
$$

#### 可视化

```text
p8_time_breakdown_by_task.svg
p8_step_ratio_by_task.svg
p8_time_auc_by_task.svg
p8_functional_event_overhead.svg
```

### P9：phase-mapped profiler

#### 目标

把 manual DG path 的系统开销定位到 forward/backward/update/functional guard，而不是只看 total step。

#### 必须记录

```text
task_name
candidate
kernel_count_total
mapped_kernel_time_fraction
unknown_kernel_time_fraction
kernel_count_forward
kernel_count_backward
kernel_count_base_update
kernel_count_functional_update
kernel_count_guard_forward
small_kernel_count_under_10us
layout_conversion_count
cuda_memcpy_time
cuda_sync_time
top_kernel_phase_1
top_kernel_time_1
```

#### 判断标准

ProfilerPass：

```text
mapped_kernel_time_fraction >= 0.90
unknown_kernel_time_fraction <= 0.10
top3 phase time explain >= 0.70
```

#### 可视化

```text
p9_kernel_timeline_by_task.svg
p9_kernel_count_phase_bar.svg
p9_top_kernel_breakdown.svg
p9_small_kernel_fragmentation.svg
```

---

## 13. Wave 7：mechanism and boundary analysis

### P10：geometry-task relationship model

#### 目标

解释 functional geometry 到底什么时候有用。

#### 必须记录

```text
task_name
task_family
input_dim
num_classes
symbolic_smoothness
DG_curvature_delta
DG_jacobian_delta
DG_lipschitz_delta
accuracy_delta
ECE_delta
NLL_delta
robustness_delta
forgetting_delta
```

#### 判断标准

MechanismModelPass：

至少找到一个稳定关系：

$$
\Delta R_{\text{curv}}<0
\Rightarrow
\Delta ECE\leq0
$$

or:

$$
\Delta R_{\text{curv}}<0
\Rightarrow
\Delta RobustnessAUC\geq0
$$

in a clearly specified task subset.

#### 可视化

```text
p10_curvature_vs_accuracy_delta.svg
p10_curvature_vs_ece_delta.svg
p10_curvature_vs_robustness_delta.svg
p10_task_family_boundary_map.svg
```

### P11：negative result and boundary audit

#### 目标

如果 DG-KAN 在某些 task family 输给 MLP，必须明确边界，而不是回避。

#### 必须记录

```text
task_family
task_name
DG_vs_MLP_delta
params_ratio
FLOPs_ratio
step_ratio
geometry_delta
possible_reason
boundary_label
```

Boundary labels：

```text
symbolic_win
mnist_like_win
geometry_robustness_win
parameter_fair_only
flops_fail
wallclock_fail
mlp_dominates
continual_unbalanced
adapter_unstable
```

#### 可视化

```text
p11_boundary_matrix.svg
p11_route_by_task_family.svg
p11_negative_result_table.md
```

---

## 14. Wave 8：final route decision and artifact audit

### P12：route decision

v8.6 route 必须给出强弱分层，而不是只写 success / fail。

#### Route cases

```text
R1-BroadExternalFairFunctionalAdvantage:
  DG-KAN 在至少两个非 symbolic task family 中通过 joint fair envelope，functional causality 与 system pass 也成立。

R2-TaskFamilyLimitedFairAdvantage:
  DG-KAN 在 MNIST-like / symbolic / continual stress 中成立，但非 symbolic 多任务不广泛成立。

R3-SymbolicAndGeometryAdvantageOnly:
  task accuracy 不广泛赢 MLP，但 symbolic / geometry / robustness 有真实优势。

R4-ParameterFairButFLOPsOrWallclockFail:
  参数公平成立，但 FLOPs 或 wall-clock 不成立。

R5-ContinualUnbalanced:
  forgetting 降低但 task balance 不达标。

R6-FunctionalCausalityNotGeneral:
  MNIST causality pass，但其他任务不成立。

R7-NegativeExternalResult:
  多数外部任务 MLP 显著占优。

R8-NoReproduction:
  v8.5 accepted route 不能 fresh reproduce。

R9-CodeIntegrationFail:
  KANbeFair adapter / counters / baseline reproduction 不完整。
```

#### route_decision.json 必须记录

```text
route
v85_reproduction_pass
kanbefair_broad_reproduction_pass
primary_transfer_pass_count
joint_fair_pass_count
functional_causality_pass_count
symbolic_pass
continual_formal_pass
wallclock_pass_count
geometry_generalization_pass
best_candidate
best_task_family
boundary_label
primary_blocker
next_required_implementation
success_v86_minimum
success_v86_external_formal
success_v86_broad_strong
```

---

## 15. Required artifacts

v8.6 必须落盘：

```text
run_manifest.json
v85_reproduction.csv
kanbefair_broad_reproduction.csv
dgkan_adapter_contract_v86.csv
params_flops_counter_v86.csv
external_multitask_transfer.csv
joint_fair_envelope.csv
functional_causality_multitask.csv
continual_multisplit_stress.csv
time_accounting_external.csv
phase_mapped_profiler_external.csv
geometry_task_relationship.csv
negative_boundary_audit.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_v85_reproduction_fail
F2_kanbefair_baseline_reproduction_fail
F3_adapter_contract_fail
F4_counter_mismatch
F5_parameter_fair_fail
F6_flops_fair_fail
F7_wallclock_fail
F8_functional_causality_fail
F9_symbolic_fail
F10_continual_unbalanced
F11_external_multitask_fail
F12_profiler_incomplete
F13_teacher_or_loss_violation
F14_fake_or_proxy_violation
F15_artifact_missing
```

---

## 16. 第一轮执行顺序

### Step 1：fresh reproduce v8.5 accepted route

先确认 `DG1-FT7 hidden28 stride128` 在 fresh out-dir 中复现。若复现失败，停止 broad claim。

### Step 2：broad KANbeFair baseline reproduction

扩展 KANbeFair baseline 任务族。必须先复现 KB-MLP/KB-KAN，再接入 DG-KAN。

### Step 3：multi-task transfer

运行 DG-Base / DG-Functional / controls on all runnable KANbeFair tasks。

### Step 4：joint fairness envelope

对每个 task family 做参数、FLOPs、wall-clock 的 joint envelope。

### Step 5：functional causality cross-task

执行 no-op / random / shuffled role controls，不再只依赖 MNIST P9。

### Step 6：continual multi-split formalization

把 P11 从 current digit-group stress 扩展到多个 class-incremental split，并加入 balanced task criterion。

### Step 7：system profiling

对所有 pass / fail 任务做 time accounting 和 phase-mapped profiler，解释系统边界。

### Step 8：boundary route

不强行追全面成功；按任务族和公平条件输出边界。

---

## 17. 停止条件

### 17.1 成功停止

Broad success：

```text
v85 reproduction pass
at least two non-symbolic task families joint fair pass
functional causality pass in majority tasks
wallclock pass in majority tasks
continual formal pass
```

Task-family success：

```text
MNIST-like + symbolic + continual pass,
but broad non-symbolic tasks not enough
```

也可停止并写边界。

### 17.2 失败停止

```text
1. v8.5 accepted route cannot reproduce；
2. KANbeFair baseline cannot reproduce；
3. DG-KAN adapter violates no-teacher/no-loss/manual contract；
4. parameter/FLOPs counter cannot align；
5. DG-KAN loses to MLP by > 0.01 on most non-symbolic tasks；
6. functional causality disappears outside MNIST；
7. continual pass becomes unbalanced across splits；
8. wall-clock pass collapses on larger tasks；
9. fake/proxy/offload violation。
```

---

## 18. 最终解释规则

### Case A：DG-KAN passes broad external fair tests

可以声明：

```text
DG-KAN + functional update establishes broad external fair advantage over MLP on tested task families.
```

### Case B：DG-KAN passes MNIST-like / symbolic / continual but not broad non-symbolic tasks

应该声明：

```text
DG-KAN functional advantage is task-family-limited, strongest on MNIST-like, symbolic, and geometry-sensitive settings.
```

### Case C：DG-KAN improves geometry but task accuracy is not broadly better

应该声明：

```text
Functional update gives geometry advantage, but does not establish broad task superiority.
```

### Case D：Continual learning reduces forgetting but is unbalanced

应该声明：

```text
Anti-forgetting update rule reduces forgetting in current stress but does not solve balanced class-incremental continual learning.
```

### Case E：External tasks refute internal advantage

应该声明：

```text
Internal DG-KAN success does not generalize under broader KANbeFair fair comparison.
```

---

## 19. 最终建议

v8.6 的一句话策略是：

$$
\boxed{
\text{把 v8.5 的强成功从“文档判据内成立”推进到“跨任务、公平、边界明确”的外部结论。}
}
$$

现在不应该继续做：

```text
teacher
self-teacher
loss modification
optimizer sweep
只在 MNIST full protocol 上继续追数值
```

现在应该做：

```text
1. fresh reproduction；
2. KANbeFair broader task reproduction；
3. joint parameter/FLOPs/wall-clock fairness；
4. functional causality across task families；
5. continual multi-split formalization；
6. boundary-aware final route。
```

v8.6 的最终目标不是强行证明 DG-KAN 全面优于 MLP，而是给出可信边界：

$$
\boxed{
\text{DG-KAN + functional update 在哪些任务族和公平条件下真的优于 MLP，在哪些条件下不优。}
}
