# DG-KAN v8.4 Formal Functional Advantage：从 v8.3 Minimum Success 到 Formal / Strong Success 的完整实验计划

> 本计划基于 v8.3 System-Gated Functional Update Re-Entry 的最新真实结果制定。  
> v8.3 已经完成 **minimum success**：`KW6 hidden68` 作为 CE-only / no-teacher / no-loss / no-offload 的系统 base，通过 H0；`FT7` role-wise functional update 在 H0 base 上重新进入 full task，并通过 P7 10-seed task/geometry confirmation、P8 TimeAUC profiler、P9 scaling/robustness。  
> 但 v8.3 仍没有声明 formal strong success，runner 保守记录 `success_v83_formal = 0`。  
> 因此 v8.4 的目标不是继续做小修小补，而是把 v8.3 的 minimum success **形式化、复现、因果解释、系统化扩展**，最终判断 functional update 是否真的成为 PureKAN 的几何优势机制，而不是一次偶然的 postprocess / guard /小数据集现象。

---

## 0. 当前结论

### 0.1 v8.3 达到了什么

v8.3 当前最新 accepted route 是：

```text
route = S0-FunctionalReEntryMinimumSuccess
success_v83_minimum = 1
success_v83_formal = 0
```

已完成的关键链条是：

```text
KW6 hidden68 H0 base:
  macro = +0.0216796875
  memory/step inside system gate
  strict CE/no-teacher/no-loss/no-offload contract pass

FT7 role-wise functional update:
  enters full task after H0
  P7 10-seed functional confirmation pass
  task preserved
  curvature reduction about 23.8%

P8:
  TimeAUC profiler gate pass

P9:
  scaling / robustness grid pass
  sample efficiency AUC improved
  3/5 robustness settings improved
  system ratio remains safe
```

因此 v8.3 的科学结论是：

$$
\boxed{
\text{Functional update 已经以 system-gated 方式重新接回 PureKAN 路线，并完成 minimum success。}
}
$$

### 0.2 v8.3 还没有达到什么

v8.3 没有声明 formal strong success：

```text
success_v83_formal = 0
```

这不是说结果无效，而是说明当前 runner 和实验协议仍然保守：minimum success 已经达成，但 formal / strong success 还需要额外确认。v8.4 必须回答：

```text
1. 这个 minimum success 是否能独立复现？
2. route 中 success_v83_formal = 0 是因为确实缺 formal artifact，还是因为代码保守 hardcode？
3. FT7 的收益是否来自真正 functional geometry update，而不是 guard、stride、role budget、holdout check 或 measurement artifact？
4. FT7 是否在更多 seeds、更大样本、更强 robustness 和更严格 profiler 下仍然成立？
5. functional update 的几何改善是否有机制解释？
```

### 0.3 当前最准确的问题定位

v8.3 前的 blocker 是：

$$
\boxed{
\text{没有 H0 base system candidate，functional update 只能 one-step diagnostic。}
}
$$

v8.3 之后 blocker 已经变化为：

$$
\boxed{
\text{minimum success 已经出现，但 formalization、causality、reproducibility 和 scaling 还没有完全闭合。}
}
$$

所以 v8.4 不应再回到：

```text
teacher
self-teacher
distillation
special loss
sampler/class weight
optimizer hyperparameter sweep
大规模 architecture 搜索
```

v8.4 要做的是：

$$
\boxed{
\text{复现 v8.3 survivor}
+
\text{证明 FT7 的 functional 因果性}
+
\text{补齐 formal system artifacts}
+
\text{扩大 scaling / robustness / geometry 验证}
}
$$

---

## 1. v8.4 总体目标

v8.4 的总体目标是：

$$
\boxed{
\text{把 v8.3 的 system-gated functional minimum success 升级为可复现、可解释、可审计的 formal functional advantage。}
}
$$

v8.4 不把 final accuracy 作为唯一目标。v8.4 要同时证明四件事：

```text
A. Base validity:
  KW6 hidden68 确实是可靠 H0 base，不是偶然 hidden/seed artifact。

B. Functional causality:
  FT7 的收益来自 functional update 本身，而不是 guard/fallback/stride 等实现副作用。

C. Geometry-task Pareto:
  FT7 在不伤 task 的前提下稳定改善几何，并且几何改善与 ECE/NLL/robustness/efficiency 有一致关系。

D. Formal system closure:
  FT7 survivor 在严格 time accounting、phase-mapped profiler、full-grid S2/S1、scaling/robustness 下仍可用。
```

### 1.1 Minimum reproduction success

v8.4 首先要复现 v8.3 minimum success：

$$
\boxed{
\text{H0BasePass}
\land
\text{FT7P7Pass}
\land
\text{P8TimeAUCPass}
\land
\text{P9ScalingRobustnessPass}
}
$$

其中 H0 base 要求：

$$
\Delta Acc_{\text{macro,val,base}}\geq0.0200,
$$

$$
r_{\text{mem,max,base}}\leq1.05,
$$

$$
r_{\text{step,max,base}}\leq1.50.
$$

FT7 task preservation：

$$
Acc_{\text{FT7}}\geq Acc_{\text{base}}-0.005.
$$

Geometry improvement：

$$
R_{\text{curv,FT7}}\leq0.90R_{\text{curv,base}}.
$$

System preservation：

$$
r_{\text{mem,max,FT7}}\leq1.05,
$$

$$
r_{\text{step,max,FT7}}\leq1.50.
$$

### 1.2 Formal success

v8.4 formal success 是：

$$
\boxed{
\text{MinimumReproductionSuccess}
\land
\text{FormalProfilerPass}
\land
\text{FormalTimeAUCPass}
\land
\text{FunctionalCausalityPass}
}
$$

FormalTimeAUCPass：

$$
ValLossAUC_{\text{time,FT7}}\leq1.05ValLossAUC_{\text{time,base}}.
$$

Functional update overhead：

$$
\frac{T_{\text{functional update}}}{T_{\text{step}}}\leq0.10.
$$

Unknown time fraction：

$$
unknown\_time\_fraction\leq0.10.
$$

FormalProfilerPass：

```text
mapped_kernel_time_fraction >= 0.90
unknown_kernel_time_fraction <= 0.10
top3 phase time explain >= 0.70
functional_update_phase separated
```

FunctionalCausalityPass：

```text
FT7 full update > guarded no-op > random functional direction
FT7 geometry improvement remains after matched overhead control
FT7 role-wise update explains geometry reduction
```

### 1.3 Strong success

v8.4 strong success 是：

$$
\boxed{
\text{FormalSuccess}
+
\text{10/20-seed robustness}
+
\text{larger data scaling}
+
\text{S1 or near-S1 system closure}
}
$$

Strong geometry target：

$$
R_{\text{curv,FT7}}\leq0.80R_{\text{curv,base}}.
$$

Strong task target：

$$
Acc_{\text{FT7}}\geq Acc_{\text{base}}.
$$

Strong robustness target：

$$
AccDrop_{\text{FT7}}\leq AccDrop_{\text{base}}
$$

for at least 3 noise / perturbation settings.

---

## 2. 本轮明确禁止事项

v8.4 继续严格禁止以下内容进入 official route：

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

functional update 必须继续是 update rule，不是 loss：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

允许的 functional update 形式是：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{\text{task}}
+
\lambda_t \Delta\theta_{\text{functional}},
$$

但不得改写 objective：

$$
L \neq CE + \lambda L_{\text{geo}}.
$$

审计字段必须为：

```text
loss_type = CE
geometry_loss_used = 0
special_loss_used = 0
external_teacher_used = 0
self_teacher_used = 0
cpu_offload_used = 0
sampler_changed = 0
class_weight_used = 0
```

---

## 3. 当前结果的代码一致性检查目标

v8.4 必须先确认代码与结果是一致的。当前 `run_gafu_v83_real.py` 的核心设计是：

```text
P0:
  candidate registry / no-teacher-no-loss / functional update contract

P1:
  H0 base confirmation

P2:
  functional operator audit

P3:
  one-step functional direction audit

P4:
  multistep functional smoke

P5:
  functional task geometry co-selection

P6:
  functional confirm5 + full-grid S2

P7:
  functional confirm10

P8:
  functional time profiler

P9:
  scaling / robustness

route:
  success_v83_minimum = bool(p9_pass_rows)
  success_v83_formal = 0
```

这意味着 v8.4 的第一件事不是改模型，而是做 route semantics audit：

```text
1. success_v83_minimum 是否确实等价于 P9 pass？
2. success_v83_formal 为什么固定为 0？
3. P8/P9 pass 是否足以支持 formal success，还是缺更严格 profiler / S1 / time accounting？
4. P9 pass rows 是否真实来自 gated P8 survivor，而不是 postprocess artifact？
5. functional_full_task_opened 是否只有在 H0 pass 后才打开？
```

---

## 4. 核心假设

### H0：v8.3 minimum success 可以独立复现

H0 是 v8.4 的第一假设。如果 v8.3 minimum success 不能复现，不能继续谈 formalization。

H0 成立标准：

在 fresh run 中，`KW6 hidden68 + FT7 stride8 alpha15 role_budget0.15 pre2 train-budget fallback` 满足：

$$
\Delta Acc_{\text{macro,val,base}}\geq0.0200,
$$

$$
\Delta Acc_{\text{FT7-base}}\geq-0.005,
$$

$$
R_{\text{curv,FT7}}\leq0.90R_{\text{curv,base}},
$$

$$
r_{\text{mem,max,FT7}}\leq1.05,
$$

$$
r_{\text{step,max,FT7}}\leq1.50.
$$

并且：

```text
P7_confirm10_pass = 1
P8_TimeAUCProfilerPass = 1
P9_scaling_robustness_pass = 1
```

如果 H0 不成立，则 v8.3 accepted route 被降级为 single-run success，需要回到 P5/P7 reproduction，而不是继续 formal claim。

---

### H1：FT7 的收益来自 functional update，而不是 guard / overhead / stride artifact

FT7 使用 role-wise guarded functional update，包含：

```text
role budget
event stride
alpha multiplier
pre-holdout baseline
train-budget fallback
head / stack role handling
```

H1 假设：几何收益主要来自 functional direction，而不是这些 guard 的副作用。

H1 成立标准：

FT7 full candidate 相对 matched-overhead no-op control 满足：

$$
R_{\text{curv,FT7}}\leq0.90R_{\text{curv,no-op}}.
$$

FT7 相对 random / shuffled functional direction 满足：

$$
R_{\text{curv,FT7}}<R_{\text{curv,random}},
$$

且：

$$
Acc_{\text{FT7}}\geq Acc_{\text{random}}.
$$

Role ablation 必须显示至少一个 role 对 geometry reduction 有正贡献：

$$
\Delta R_{\text{curv,role}}>0.
$$

如果 matched-overhead no-op 也能达到同样 geometry reduction，则 functional causality 不成立。

---

### H2：FT7 的 geometry 改善与泛化/鲁棒性一致，而不是孤立指标

v8.3 P9 已显示 sample efficiency AUC 和 3/5 robustness settings 优于 base。H2 要求更严格地验证 geometry 不是“好看但无用”的指标。

H2 成立标准：

在 scaling / robustness / perturbation 中：

$$
\Delta R_{\text{curv}}<0
$$

并且至少满足一项：

$$
\Delta ECE\leq0,
$$

$$
\Delta NLL\leq0,
$$

$$
\Delta AccDrop\leq0.
$$

若 geometry 改善但 ECE/NLL/robustness 全部变差，则 functional update 被判为 geometry-only cosmetic improvement。

---

### H3：P8 TimeAUC pass 需要更细时间归因确认

v8.3 accepted route 报告 P8 pass，但 formal success 没声明。H3 假设 P8 pass 可能仍需更细的 phase accounting 才能升级 formal。

H3 成立标准：

严格 time accounting 中：

```text
unknown_time_fraction <= 0.10
functional update time separated
metric build time separated
holdout guard time separated
validation/logging/sync separated
```

并且：

$$
ValLossAUC_{\text{time,FT7}}\leq1.05ValLossAUC_{\text{time,base}}.
$$

If:

$$
ValLossAUC_{\text{step,FT7}}\leq ValLossAUC_{\text{step,base}},
$$

but:

$$
ValLossAUC_{\text{time,FT7}}>1.05ValLossAUC_{\text{time,base}},
$$

then bottleneck is system overhead, not functional direction.

---

### H4：formal success 需要明确 route semantics，而不是代码保守固定 0

当前代码中 `success_v83_formal = 0` 是保守记录。H4 要求 v8.4 重新定义 formal gate 并让 route_decision 显式区分：

```text
minimum_success
formal_time_profiler_success
formal_s1_success
strong_scaling_success
```

H4 成立标准：

`route_decision.json` 必须包含：

```text
success_v84_minimum
success_v84_formal_time_profiler
success_v84_formal_s1
success_v84_strong
```

并说明每个 false 的具体缺口。

---

### H5：FT7 仍需 S1 / memory formalization

v8.3 P9 system ratio 安全，但 formal strong success 不声明。H5 要求评估 FT7 是否进入 S1 或 near-S1。

H5 成立标准：

S1 pass：

$$
r_{\text{mem,max,FT7}}<1.00,
$$

$$
r_{\text{step,max,FT7}}\leq1.35.
$$

Near-S1 pass：

$$
r_{\text{mem,max,FT7}}\leq1.01,
$$

$$
r_{\text{step,max,FT7}}\leq1.35.
$$

若 S1 不成立，必须通过 live-set attribution 定位 functional update 额外 memory 来源。

---

## 5. Candidate 设计

### 5.1 Base candidates

```text
B0-MLP-AdamW-CE
KW6-hidden68-base
KW6-hidden68-base-reproduction
```

### 5.2 Functional survivor candidates

```text
FT7-accepted-stride8-alpha15-rolebudget015-pre2-trainbudget
FT7-reproduction-fresh
FT7-noop-matched-overhead-control
FT7-random-functional-direction-control
FT7-shuffled-role-functional-direction-control
FT7-head-only
FT7-stack-only
FT7-no-preholdout-control
FT7-no-trainbudget-fallback-control
```

### 5.3 Formalization candidates

```text
FORM0-FT7-current
FORM1-FT7-strict-time-accounting
FORM2-FT7-phase-mapped-profiler
FORM3-FT7-S1-live-set-attribution
FORM4-FT7-low-overhead-metric-build
FORM5-FT7-formalized-route
```

### 5.4 Scaling / robustness candidates

```text
GEN0-base
GEN1-FT7-current
GEN2-FT7-reproduction
```

---

## 6. 实验阶段总览

v8.4 分为八个 Wave：

```text
Wave 0:
  route / contract / code consistency audit

Wave 1:
  v8.3 survivor reproduction

Wave 2:
  functional causality and ablation

Wave 3:
  formal time / profiler / S1 system closure

Wave 4:
  geometry mechanism attribution

Wave 5:
  scaling / robustness / larger-budget validation

Wave 6:
  final 10/20-seed confirmation

Wave 7:
  route formalization and artifact audit
```

---

## 7. Wave 0：route / contract / code consistency audit

### P0：Contract and route semantics audit

#### 目标

确认 v8.3 accepted route 和 `run_gafu_v83_real.py` 的逻辑一致，且没有 teacher/loss/offload污染。

#### 必须记录

```text
candidate_id
functional_candidate_id
loss_type
geometry_loss_used
functional_update_is_update_rule
external_teacher_used
self_teacher_used
teacher_logits_used
sampler_changed
class_weight_used
cpu_offload_used
uses_loss_backward
nonKAN_param_count
manual_forward
manual_backward
manual_update
p7_pass_rows_count
p8_pass_rows_count
p9_pass_rows_count
success_v83_minimum_code_value
success_v83_formal_code_value
```

#### 判断标准

Contract pass：

```text
loss_type = CE
geometry_loss_used = 0
external_teacher_used = 0
self_teacher_used = 0
cpu_offload_used = 0
uses_loss_backward = 0
```

Route consistency pass：

```text
success_v83_minimum = bool(p9_pass_rows)
success_v83_formal = 0 is documented as conservative route field
functional_full_task_opened only after H0 pass
P9 only opens after P8 pass
```

#### 可视化

```text
p0_contract_heatmap.svg
p0_route_state_machine.svg
p0_pass_rows_flowchart.svg
p0_violation_matrix.svg
```

---

## 8. Wave 1：v8.3 survivor reproduction

### P1：Fresh reproduction of KW6 hidden68 base

#### 目标

确认 H0 base 不是一次性结果。

#### 设置

```text
candidate = KW6 hidden68
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
bench batch = 128,256,512
grad batch = 8,128
bootstrap reps = 10000
```

#### 必须记录

```text
macro_val_gap
CI95_low
Holm_p
test_gap
per_dataset_gap
per_seed_gap
ECE
NLL
Brier
GradPass
grad_relerr_max
memory_ratio_max
step_ratio_max
S2_shape_count
S1_shape_count
ValLossAUC_step
ValLossAUC_time
```

#### 判断标准

Base reproduction pass：

$$
\Delta Acc_{\text{macro,val}}\geq0.0200,
$$

$$
CI_{95\%,macro}^{low}>0,
$$

$$
r_{\text{mem,max}}\leq1.05,
$$

$$
r_{\text{step,max}}\leq1.50.
$$

#### 可视化

```text
p1_base_seedwise_gap_boxplot.svg
p1_base_dataset_gap_bar.svg
p1_base_s2_heatmap.svg
p1_base_task_system_pareto.svg
```

### P2：Fresh reproduction of FT7 accepted route

#### 目标

确认 FT7 accepted route 可复现。

#### 设置

```text
base = KW6 hidden68
functional = FT7 accepted setting
seeds = 0..9
datasets = MNIST,Fashion-MNIST,KMNIST
event_stride = 8
alpha_mult = 15
role_budget = 0.15 stack / 0.15 head
pre_holdout_every = 2
train_budget_fallback = 0.25
```

#### 必须记录

```text
functional_macro_delta_vs_base
functional_CI95_low
functional_ECE_delta_vs_base
functional_NLL_delta_vs_base
curvature_ratio_vs_base
geometry_reduction
memory_ratio_max
step_ratio_max
S2_shape_count
functional_update_time_ratio
bad_step_rate
holdout_descent_ratio
fallback_rate
role_accept_rate
```

#### 判断标准

FT7 reproduction pass：

$$
Acc_{\text{FT7}}\geq Acc_{\text{base}}-0.005,
$$

$$
R_{\text{curv,FT7}}\leq0.90R_{\text{curv,base}},
$$

$$
r_{\text{mem,max}}\leq1.05,
$$

$$
r_{\text{step,max}}\leq1.50.
$$

#### 可视化

```text
p2_ft7_task_delta_forest.svg
p2_ft7_geometry_delta_forest.svg
p2_ft7_role_accept_rate.svg
p2_ft7_system_overhead_bar.svg
```

---

## 9. Wave 2：functional causality and ablation

### P3：Matched overhead / no-op control

#### 目标

证明 geometry gain 不是因为额外 compute、holdout guard 或 timing overhead。

#### 候选

```text
FT7-full
FT7-noop-matched-overhead
FT7-random-direction
FT7-shuffled-role-direction
base-no-functional
```

#### 必须记录

```text
macro_delta_vs_base
curvature_ratio_vs_base
ECE_delta
NLL_delta
functional_update_time_ratio
metric_build_time
holdout_guard_time
random_direction_norm
accepted_update_norm
bad_step_rate
```

#### 判断标准

Functional causality pass：

$$
R_{\text{curv,FT7-full}}<R_{\text{curv,no-op}},
$$

$$
R_{\text{curv,FT7-full}}<R_{\text{curv,random}},
$$

且：

$$
Acc_{\text{FT7-full}}\geq Acc_{\text{no-op}}-0.002.
$$

If no-op gives same geometry improvement, FT7 causality fails.

#### 可视化

```text
p3_causality_curvature_bar.svg
p3_causality_task_geometry_pareto.svg
p3_matched_overhead_time_bar.svg
p3_random_vs_functional_direction.svg
```

### P4：Role ablation

#### 目标

解释 FT7 的 role-wise functional update 到底靠哪个 role 生效。

#### 候选

```text
FT7-full
FT7-head-only
FT7-stack-only
FT7-head-stack-swapped-budget
FT7-no-role-budget
```

#### 必须记录

```text
role_update_share_stack
role_update_share_head
role_accept_rate_stack
role_accept_rate_head
role_curvature_reduction_stack
role_curvature_reduction_head
role_task_delta_stack
role_task_delta_head
role_bad_step_rate
```

#### 判断标准

Role mechanism pass：

至少一个 role 满足：

$$
\Delta R_{\text{curv,role}}>0,
$$

且对应：

$$
AccDrop_{\text{role}}\leq0.003.
$$

如果 head-only 和 stack-only 都无几何收益，而 full 有收益，需要记录 role interaction effect。

#### 可视化

```text
p4_role_update_share_stacked.svg
p4_role_geometry_reduction_bar.svg
p4_role_task_delta_bar.svg
p4_role_interaction_matrix.svg
```

### P5：Guard / stride / fallback ablation

#### 目标

解释 accepted setting 为什么有效，避免把 guard 参数当黑箱。

#### 候选

```text
FT7-accepted
FT7-stride4-alpha15
FT7-stride8-alpha8
FT7-stride8-alpha15-no-pre2
FT7-stride8-alpha15-no-trainbudget
FT7-stride8-alpha15-pre1
FT7-stride8-alpha15-pre3
```

#### 必须记录

```text
event_count
accepted_event_count
role_accept_rate
fallback_rate
bad_step_rate
holdout_descent_ratio
curvature_ratio
macro_delta_vs_base
functional_update_time_ratio
ValLossAUC_time_ratio
```

#### 判断标准

Accepted setting remains best if it dominates alternatives on:

$$
R_{\text{curv}}\text{ lower},
$$

$$
AccDrop\leq0.005,
$$

$$
T_{\text{functional update}}/T_{\text{step}}\leq0.10.
$$

#### 可视化

```text
p5_stride_alpha_pareto.svg
p5_guard_ablation_bar.svg
p5_event_accept_timeline.svg
p5_bad_step_vs_curvature.svg
```

---

## 10. Wave 3：formal time / profiler / S1 closure

### P6：Strict time accounting

#### 目标

将 P8 从 “pass” 升级为 formal time accounting。

#### 必须记录

```text
wall_clock_total
train_step_time
forward_time
backward_time
update_time
functional_metric_build_time
functional_update_time
holdout_guard_time
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

Formal time pass：

$$
unknown\_time\_fraction\leq0.10,
$$

$$
ValLossAUC_{\text{time,FT7}}\leq1.05ValLossAUC_{\text{time,base}},
$$

$$
functional\_update\_time/T_{\text{step}}\leq0.10.
$$

#### 可视化

```text
p6_time_breakdown_stacked.svg
p6_functional_overhead_timeline.svg
p6_step_auc_vs_time_auc.svg
p6_train_only_vs_total_auc.svg
```

### P7：Phase-mapped profiler

#### 目标

确认 FT7 的 kernel / update overhead 可解释。

#### 必须记录

```text
kernel_count_total
mapped_kernel_time_fraction
unknown_kernel_time_fraction
kernel_count_forward
kernel_count_backward
kernel_count_base_update
kernel_count_functional_update
top_kernel_name_1
top_kernel_phase_1
top_kernel_time_1
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
functional_update_phase mapped
```

#### 可视化

```text
p7_kernel_timeline.svg
p7_kernel_count_by_phase.svg
p7_top_kernel_time_bar.svg
p7_functional_update_kernel_breakdown.svg
```

### P8：S1 / live-set attribution

#### 目标

判断 FT7 是否已接近 S1，并定位 S1 缺口。

#### 必须记录

```text
memory_ratio_max
step_ratio_max
root_input_cache_MB
backward_temp_MB
optimizer_state_MB
functional_metric_buffer_MB
functional_update_buffer_MB
holdout_guard_buffer_MB
allocator_padding_MB
reserved_unallocated_MB
top_memory_source_1
top_memory_source_2
top_memory_source_3
unknown_memory_fraction
```

#### 判断标准

S1 pass：

$$
r_{\text{mem,max}}<1.00,
$$

$$
r_{\text{step,max}}\leq1.35.
$$

Attribution pass：

```text
unknown_memory_fraction <= 0.10
top3 memory sources identified
```

#### 可视化

```text
p8_s1_memory_waterfall.svg
p8_memory_source_bar.svg
p8_s1_candidate_pareto.svg
```

---

## 11. Wave 4：geometry mechanism attribution

### P9：Edge-level geometry decomposition

#### 目标

确认 FT7 降低曲率的是哪些 edge / role / layer。

#### 必须记录

```text
layer_id
role
edge_group
curvature_base
curvature_ft7
curvature_delta
smoothness_base
smoothness_ft7
slope_p95_base
slope_p95_ft7
update_norm
accept_rate
task_descent_contribution
```

#### 判断标准

Geometry attribution pass：

至少 70% 的 curvature reduction 可归因到 top roles/layers：

$$
\frac{\sum_{i=1}^{k}\Delta R_{\text{curv},i}}
{\Delta R_{\text{curv,total}}}\geq0.70.
$$

#### 可视化

```text
p9_layer_curvature_delta_heatmap.svg
p9_role_curvature_waterfall.svg
p9_update_norm_vs_curvature_delta.svg
```

### P10：Function-space probe

#### 目标

确认参数曲率下降对应函数空间平滑，而不只是参数指标变化。

#### 必须记录

```text
input_noise_level
finite_difference_curvature
jacobian_norm
local_lipschitz
function_displacement_R2
path_length
margin_p10
hard_sample_acc
```

#### 判断标准

Function-space geometry pass：

至少两个指标改善：

$$
J_{\text{norm,FT7}}\leq0.95J_{\text{norm,base}},
$$

$$
Lip_{\text{local,FT7}}\leq0.95Lip_{\text{local,base}},
$$

$$
FD_{\text{curv,FT7}}\leq0.90FD_{\text{curv,base}}.
$$

#### 可视化

```text
p10_function_curvature_boxplot.svg
p10_jacobian_norm_bar.svg
p10_lipschitz_vs_noise.svg
p10_margin_hard_sample_scatter.svg
```

---

## 12. Wave 5：scaling / robustness / larger validation

### P11：Extended scaling confirmation

#### 目标

把 P9 scaling 从 smoke 扩展为 stronger evidence。

#### 设置

```text
train_size = 256,512,1024,1536,4096,8192
seeds = 0..9 for 1536
seeds = 0..4 for other sizes
```

#### 必须记录

```text
train_size
candidate
val_acc
test_acc
val_loss
test_loss
ECE
NLL
curvature
sample_efficiency_auc
memory_ratio
step_ratio
```

#### 判断标准

Scaling pass：

$$
AUC_{\text{data,FT7}}\geq AUC_{\text{data,base}}.
$$

and:

$$
R_{\text{curv,FT7}}\leq R_{\text{curv,base}}
$$

for at least 4/6 train sizes.

#### 可视化

```text
p11_accuracy_vs_train_size.svg
p11_sample_efficiency_auc.svg
p11_curvature_vs_train_size.svg
```

### P12：Extended robustness confirmation

#### 目标

确认 functional geometry 对扰动稳健性有真实作用。

#### 设置

```text
label_noise = 0.05,0.10,0.20
input_noise = 0.05,0.10,0.20
random_erasing = small,medium
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
curvature_under_noise
robustness_auc
memory_ratio
step_ratio
```

#### 判断标准

Robustness pass：

$$
AccDrop_{\text{FT7}}\leq AccDrop_{\text{base}}
$$

for at least 4 settings.

Strong robustness:

$$
RobustnessAUC_{\text{FT7}}\geq RobustnessAUC_{\text{base}}.
$$

#### 可视化

```text
p12_accuracy_drop_bar.svg
p12_robustness_auc.svg
p12_ece_under_noise.svg
p12_curvature_under_noise.svg
```

---

## 13. Wave 6：final confirmation

### P13：10-seed final reproduction

#### 目标

在 fresh out-dir 中复现整个 accepted chain。

#### 必跑

```text
B0
KW6 hidden68 base
FT7 accepted
matched-overhead no-op
random functional control
```

#### 必须记录

```text
H0BasePass
P7Confirm10Pass
P8TimePass
P9ScalingRobustnessPass
FunctionalCausalityPass
FormalProfilerPass
S1Pass
success_v84_minimum
success_v84_formal_time_profiler
success_v84_strong
```

#### 判断标准

v8.4 minimum：

```text
H0BasePass = 1
P7Confirm10Pass = 1
P8TimePass = 1
FunctionalCausalityPass = 1
```

v8.4 formal：

```text
v8.4 minimum
FormalProfilerPass = 1
FormalTimeAccountingPass = 1
```

v8.4 strong：

```text
v8.4 formal
ExtendedScalingPass = 1
ExtendedRobustnessPass = 1
FunctionSpaceGeometryPass = 1
```

#### 可视化

```text
p13_final_scorecard.svg
p13_route_decision_tree.svg
p13_task_geometry_system_pareto.svg
```

---

## 14. Route decision

v8.4 route 必须比 v8.3 更细，不再只有 minimum/formal 两个字段。

### 14.1 Survivor types

```text
S0:
  Functional formal + scaling/robustness strong success

S1:
  Functional formal success, strong validation open

S2:
  Functional minimum success reproduced, formal profiler incomplete

S3:
  Functional task/geometry success, TimeAUC fail

S4:
  Functional causality fail

S5:
  Functional geometry gain but no generalization benefit

S6:
  H0 base fail

S7:
  Contract violation

S8:
  Reproduction fail
```

### 14.2 route cases

```text
R1-FunctionalStrongGeometrySystemAdvantage:
  S0. 可声明 strong functional route.

R2-FunctionalFormalAdvantage:
  S1. 可声明 formal route.

R3-FunctionalMinimumReproduced:
  S2. 可声明 minimum reproduced, formal open.

R4-FunctionalTimeBlocked:
  S3. geometry/task 有效，但 wall-clock 仍未闭合.

R5-FunctionalCausalityUnproven:
  S4. FT7 可能只是 guard/overhead artifact.

R6-GeometryOnlyNoGeneralization:
  S5. 几何好看但泛化无证据.

R7-NoBaseSystemCandidate:
  S6. 回到 CE-only system closure.

R8-ContractFail:
  S7. 不允许 scientific claim.

R9-NoReproduction:
  S8. v8.3 accepted route 降级。
```

### 14.3 route_decision.json 必须记录

```text
route
base_candidate_id
functional_candidate_id
success_v84_minimum
success_v84_formal_time_profiler
success_v84_formal_s1
success_v84_strong
H0BasePass
P7Confirm10Pass
P8TimePass
P9ScalingRobustnessPass
FunctionalCausalityPass
FormalProfilerPass
FunctionSpaceGeometryPass
ExtendedScalingPass
ExtendedRobustnessPass
base_macro_gap
functional_acc_delta_vs_base
curvature_ratio_vs_base
ECE_delta_vs_base
NLL_delta_vs_base
memory_ratio_max
step_ratio_max
functional_update_time_ratio
unknown_time_fraction
primary_blocker
next_required_implementation
```

---

## 15. Required artifacts

v8.4 必须落盘：

```text
run_manifest.json
candidate_registry_v84.csv
contract_no_teacher_no_loss_functional_v84.csv
route_semantics_audit.csv

p0_contract_route_audit.csv
p1_base_reproduction.csv
p2_ft7_reproduction.csv
p3_causality_controls.csv
p4_role_ablation.csv
p5_guard_stride_ablation.csv
p6_time_accounting.csv
p7_phase_mapped_profiler.csv
p8_s1_liveset_attribution.csv
p9_edge_geometry_decomposition.csv
p10_function_space_probe.csv
p11_extended_scaling.csv
p12_extended_robustness.csv
p13_final_confirmation.csv

route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_base_reproduction_fail
F3_ft7_reproduction_fail
F4_functional_causality_fail
F5_geometry_no_generalization
F6_time_auc_fail
F7_profiler_incomplete
F8_s1_fail
F9_scaling_fail
F10_robustness_fail
F11_function_space_geometry_fail
F12_fake_or_proxy_violation
F13_artifact_missing
```

---

## 16. 第一轮执行顺序

### Step 1：P0 route semantics audit

先确认代码与实验声明一致。重点检查：

```text
success_v83_minimum = bool(p9_pass_rows)
success_v83_formal = 0
P9 only opens after P8 pass
functional_full_task only opens after H0 pass
```

### Step 2：P1/P2 fresh reproduction

复现 `KW6 hidden68 + FT7 accepted`。如果不能复现，停止 formalization。

### Step 3：P3/P4/P5 causality matrix

用 no-op、random、role ablation 和 guard ablation 检验 functional update 因果性。

### Step 4：P6/P7/P8 formal system closure

补严格 time accounting、phase-mapped profiler、S1/live-set attribution。

### Step 5：P9/P10 geometry mechanism

做 edge-level 和 function-space geometry attribution。

### Step 6：P11/P12 scaling/robustness extension

扩大 P9 的 scaling/robustness evidence。

### Step 7：P13 final confirmation

fresh 10-seed final chain，输出 v8.4 route。

---

## 17. 停止条件

### 17.1 成功停止

v8.4 minimum：

```text
H0BasePass = 1
P7Confirm10Pass = 1
P8TimePass = 1
FunctionalCausalityPass = 1
```

v8.4 formal：

```text
minimum + FormalProfilerPass + FormalTimeAccountingPass
```

v8.4 strong：

```text
formal + ExtendedScalingPass + ExtendedRobustnessPass + FunctionSpaceGeometryPass
```

### 17.2 失败停止

```text
1. KW6 hidden68 H0 base reproduction fail；
2. FT7 accepted route reproduction fail；
3. no-op / random control matches FT7 geometry improvement；
4. FT7 task delta < -0.005；
5. functional update time ratio > 0.10；
6. P8 TimeAUC fails again under strict accounting；
7. scaling/robustness extension fails all settings；
8. contract/fake/proxy/cpu-offload violation。
```

---

## 18. 最终解释规则

### Case A：FT7 reproduces and causality passes

说明 v8.3 的 minimum success 是真实的 functional geometry mechanism，不是 artifact。

### Case B：FT7 reproduces but causality fails

说明 functional route 可能依赖 guard/overhead/selection，而非 functional direction 本身。不能声明 functional advantage。

### Case C：FT7 reproduces and causality passes, but formal time fails

说明 functional update 有几何价值，但系统开销还需优化。

### Case D：FT7 improves geometry but no robustness/scaling benefit

说明几何指标可能是内部正则效果，尚未证明对 generalization 有用。

### Case E：v8.3 accepted route cannot reproduce

说明 v8.3 应降级为 single-run minimum success，需要重新找 functional survivor。

---

## 19. 最终建议

v8.4 的一句话策略是：

$$
\boxed{
\text{把 v8.3 的 minimum success 变成可复现、因果明确、系统可解释的 formal functional advantage。}
}
$$

现在不该再做：

```text
teacher
self-teacher
loss modification
optimizer sweep
大规模新 architecture 搜索
```

现在该做：

```text
1. fresh reproduction；
2. functional causality controls；
3. role-wise mechanism attribution；
4. formal time/profiler/S1 audit；
5. extended scaling/robustness；
6. final 10-seed route formalization。
```

v8.4 最终要证明的不只是：

$$
\text{FT7 让曲率下降。}
$$

而是：

$$
\boxed{
\text{FT7 是一个真实、可复现、低开销、task-safe 的 functional geometry update。}
}
