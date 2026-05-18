# DG-KAN v8.8 Broad External Generalization and Mechanism Closure：从 v8.7 Formal Selected Route 到跨任务强结论的完整实验计划

> 本计划基于 v8.7 Stability-First Functional Architecture 的真实复盘制定。  
> v8.7 的结果已经不是早期“能不能过线”的问题，而是进入了更严肃的阶段：**当前 selected route 在 formal 条件下已经成立，但 broad strong 仍未声明**。  
> v8.8 的目标不是继续小修小补，也不是继续围绕某个 MNIST-like 配置微调；v8.8 要把当前方法升级为一个跨任务、公平、可复现、机制可解释、边界清楚的 Functional PureKAN 方法。

---

## 0. 当前独立判断

### 0.1 v8.7 达到了什么

v8.7 最新 formal artifact 显示：

```text
success_v87_stability = true
success_v87_adaptive = true
success_v87_no_manual_tuning = true
success_v87_external_fair = true
success_v87_formal = true
success_v87_broad_strong = not_claimed
```

最终 route 是：

```text
route = R1-ExternalBoundarySelected
primary_blocker = none
next_required_implementation = optional_run_p10_p13_boundary_and_ablation_waves
```

最终 no-fake audit 为：

```text
rows_checked = 2829
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

这些结果说明，v8.7 已经完成了一个重要层级的目标：它不再是单点调参成功，而是完成了 stability、adaptive、no-manual-tuning、external-fair 和 formal route 的闭合。

### 0.2 但 v8.7 没有完成 broad strong

v8.7 明确不声明：

```text
success_v87_broad_strong = not_claimed
```

原因有三点：

```text
1. 当前 formal external 覆盖主要是 FMNIST / KMNIST，仍属于同一个 vision-family，不是两个以上非 symbolic task families。
2. continual_balance_multisplit.csv、robustness_perturbation.csv、functional_mechanism_attribution.csv、negative_boundary_audit.csv 当前仍是占位或 not_run。
3. 继续推进 broad strong 需要真实实现 P10-P13，而不能把未跑阶段写成成功。
```

因此，v8.7 的结论应该写成：

$$
\boxed{
\text{Formal selected route 已成立，但 broad strong external generalization 尚未证明。}
}
$$

### 0.3 当前最强数据

v8.7 的最终关键数据包括：

```text
P3 adaptive multistep:
  20-step curvature ratio = 0.896726
  50-step curvature ratio = 0.859501
  240-step curvature ratio = 0.807270
  3/3 pass

P4 no-manual selection:
  selected base = KW4 hidden28
  params ratio = 0.939568
  FLOPs ratio = 0.936347

P4 selected confirmation:
  DG functional delta vs KB-MLP = +0.279999
  curvature ratio = 0.772483

P5 timing:
  q90 step ratio = 0.933570
  max step ratio = 0.957862
  unknown fraction max = 0.066499

P6 compute:
  forward FLOPs ratio = 0.936347
  backward estimate ratio = 0.936347

P7 FMNIST:
  DG functional acc = 88.279998
  KB-MLP acc = 86.869997
  delta = +1.410002

P7 KMNIST:
  DG functional acc = 85.929996
  KB-MLP acc = 83.749998
  delta = +2.179998

P9 FMNIST causality:
  DG functional curvature = 0.657719
  RandomFunc curvature = 1.057890

P9 KMNIST causality:
  DG functional curvature = 0.725878
  RandomFunc curvature = 1.120099
```

这些数据支持一个明确判断：

$$
\boxed{
\text{当前方法在 vision-family 内已经同时出现 task、geometry、fairness、timing 的强证据。}
}
$$

但它还不支持：

$$
\boxed{
\text{DG-KAN + functional update 已经在广泛任务族上系统性超过 MLP。}
}
$$

---

## 1. v8.8 总体目标

v8.8 的总体目标是：

$$
\boxed{
\text{把 v8.7 的 formal selected route 扩展成跨任务、机制明确、边界清楚的 broad external conclusion。}
}
$$

这不是继续追一个更高的 MNIST / FMNIST / KMNIST 数字，而是回答以下问题：

```text
Q1:
  v8.7 selected route 是否在 independent rerun、更多 seeds、更多 timing protocols 下稳定？

Q2:
  adaptive functional controller 的收益是否来自真正 functional geometry update，
  而不是 event cadence、compiled/prewarm、fairness envelope 或 timing artifact？

Q3:
  DG-KAN + adaptive functional update 是否能跨出 MNIST-like vision-family？
  在 tabular、symbolic、continual、NLP/audio if runnable 上是否仍有公平优势？

Q4:
  当前 params / FLOPs / backward estimate / wall-clock / memory 的公平性是否足够硬？
  还是只在 forward FLOPs 或单一计数口径下好看？

Q5:
  functional geometry 改善是否与 ECE、NLL、robustness、forgetting、sample efficiency 有一致关系？
  还是只降低 curvature metric 但没有实际下游收益？

Q6:
  continual learning 是否真正 balanced？
  还是只降低 forgetting 但偏向 old tasks？

Q7:
  如果 broad external tasks 不通过，优势边界在哪里？
```

v8.8 允许出现 negative result。目标不是强行证明全面优越，而是给出可信边界。

---

## 2. 继续坚持的硬约束

v8.8 继续禁止以下行为进入 official route：

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

functional update 必须是 update rule，不是 loss。训练目标仍然是：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

允许的 functional update 是：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{\text{AdamW-equivalent}}
+
\Delta\theta_{\text{functional}}.
$$

不允许变成：

$$
L=CE+\lambda L_{\text{geo}}.
$$

所有 official rows 必须记录：

```text
loss_type = CE
geometry_loss_used = 0
external_teacher_used = 0
self_teacher_used = 0
teacher_logits_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
uses_loss_backward = 0
fake_data_used = 0
proxy_row_used = 0
```

---

## 3. 核心假设

### H0：v8.7 selected route 可以 independent reproduce

H0 是前置假设。v8.8 不能在旧 artifact 上继续扩大结论。

H0 成立标准：

在 fresh out-dir 中，selected route 满足：

$$
Acc_{\text{DG-Adaptive}}\geq Acc_{\text{KB-MLP}},
$$

$$
Params_{\text{DG-Adaptive}}\leq1.05Params_{\text{KB-MLP}},
$$

$$
FLOPs_{\text{DG-Adaptive}}\leq1.05FLOPs_{\text{KB-MLP}},
$$

$$
StepRatio_{\text{DG-Adaptive}}\leq1.50,
$$

$$
MemoryRatio_{\text{DG-Adaptive}}\leq1.05,
$$

$$
R_{\text{curv,DG-Adaptive}}\leq0.90R_{\text{DG-Base}}.
$$

并且：

```text
NoTeacherNoLossNoOffloadPass = 1
FunctionalCausalityPass = 1
NoFakeNoProxyPass = 1
```

如果 H0 不成立，v8.7 formal selected route 降级为：

```text
single-run formal success
```

不得进入 broad claim。

### H1：adaptive controller 比 fixed recipe 更稳定

v8.7 的进步是从 fixed `hidden28 + stride128` 走向 adaptive/no-manual selection。H1 要求证明 adaptive controller 不是只是另一个固定 recipe。

H1 成立标准：

在相同 task / seed / rerun / timing protocols 下：

$$
P_{\text{pass,adaptive}}\geq P_{\text{pass,fixed}},
$$

且：

$$
Var(\text{gate margin}_{\text{adaptive}})
<
Var(\text{gate margin}_{\text{fixed}}).
$$

其中 gate margin 定义为：

$$
m_{\text{gate}}
=
\min
\left(
Acc_{\text{DG}}-Acc_{\text{MLP}},
1.05-ParamsRatio,
1.05-FLOPsRatio,
1.50-StepRatio,
0.90-CurvatureRatio
\right).
$$

若 adaptive controller 与 fixed recipe pass rate 相同，但 variance 更低，也可视为稳定性改进。

### H2：compiled/prewarm/system-side repair 不是 functional causality 的替代品

v8.7 中 system-side repair 包括 compiled fused/prewarm 等，它们解决 timing stability，但不应被误认为 functional geometry 的来源。H2 要求分离：

```text
system repair effect
functional update effect
adaptive controller effect
```

H2 成立标准：

在相同 system path 下：

$$
R_{\text{curv,functional}}<R_{\text{curv,no-op}},
$$

$$
R_{\text{curv,functional}}<R_{\text{curv,random}},
$$

并且：

$$
Acc_{\text{functional}}\geq Acc_{\text{no-op}}-\epsilon.
$$

分类任务默认：

$$
\epsilon=0.002.
$$

如果 no-op 或 random control 也能获得同等 curvature improvement，则 functional causality 不成立。

### H3：vision-family 优势不是 broad external success

v8.7 已经在 FMNIST / KMNIST 上通过，但它们仍属于 vision-family。H3 要求扩展任务族。

H3 成立标准：

至少两个非 symbolic task families 满足 joint fair pass：

$$
Metric_{\text{DG-Adaptive}}\geq Metric_{\text{KB-MLP}},
$$

$$
ParamsRatio\leq1.05,
$$

$$
FLOPsRatio\leq1.05,
$$

$$
StepRatio\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

如果只在 MNIST-like / symbolic / geometry-sensitive tasks 上成功，则 route 应写为：

```text
TaskFamilyLimitedFunctionalAdvantage
```

而不是 broad strong。

### H4：training compute fairness 必须独立于 forward FLOPs

v8.7 P6 当前记录 forward/backward estimate ratio `0.936347`，这是很好的信号，但 v8.8 要进一步检查 kernel-time / event-time / guard-time。H4 成立标准：

至少满足以下三项中的两项：

$$
ForwardFLOPs_{\text{DG}}\leq1.05ForwardFLOPs_{\text{MLP}},
$$

$$
BackwardFLOPs_{\text{DG}}\leq1.50BackwardFLOPs_{\text{MLP}},
$$

$$
KernelTime_{\text{DG}}\leq1.50KernelTime_{\text{MLP}},
$$

$$
StepTime_{\text{DG}}\leq1.50StepTime_{\text{MLP}}.
$$

如果 forward/backward estimate pass，但 kernel time fail，则 route 应写为：

```text
ComputeCounterPassButKernelTimeFail
```

### H5：functional geometry 必须对应下游收益

H5 检查 geometry 是否只是内部指标。

H5 成立标准：

在 tested tasks 中，至少 70% 满足：

$$
R_{\text{curv,DG-Adaptive}}\leq0.90R_{\text{DG-Base}}.
$$

并且至少 50% 满足至少一个下游指标不劣化：

$$
ECE_{\text{DG-Adaptive}}\leq ECE_{\text{DG-Base}},
$$

或：

$$
NLL_{\text{DG-Adaptive}}\leq NLL_{\text{DG-Base}},
$$

或：

$$
RobustnessAUC_{\text{DG-Adaptive}}\geq RobustnessAUC_{\text{DG-Base}},
$$

或：

$$
Forgetting_{\text{DG-Adaptive}}\leq Forgetting_{\text{DG-Base}}.
$$

如果 curvature 下降但下游指标普遍变差，route 应写为：

```text
GeometryMetricOnly
```

### H6：continual learning 必须 balanced

v8.7 当前未运行 continual balance。v8.8 必须完成。

H6 成立标准：

对至少三个 splits：

$$
Forgetting_{\text{DG-Adaptive}}\leq Forgetting_{\text{DG-Base}},
$$

$$
FinalAvgAcc_{\text{DG-Adaptive}}\geq FinalAvgAcc_{\text{DG-Base}},
$$

并且：

$$
\max_i Acc_i-\min_i Acc_i\leq0.20.
$$

若 forgetting 降低但 task imbalance 超过 0.20，则 route 写为：

```text
ForgettingReducedButImbalanced
```

### H7：negative result 必须被允许

若多数非 symbolic tasks 上：

$$
Acc_{\text{DG-Adaptive}}<Acc_{\text{KB-MLP}}-0.01,
$$

则结论必须写成：

```text
DG-KAN functional advantage is task-family-limited.
```

这不是路线失败，而是边界明确化。

---

## 4. Candidate 设计

### 4.1 Core baselines

```text
KB-MLP:
  KANbeFair MLP baseline

KB-KAN:
  KANbeFair official KAN baseline

DG-Base:
  selected KW4 hidden28 base

DG-Fixed-FT7:
  old fixed stride / fixed role-budget recipe

DG-Adaptive-FT-P:
  v8.7 selected adaptive functional route

DG-NoOp:
  matched-overhead no-op control

DG-RandomFunc:
  random functional direction control

DG-ShuffledRoleFunc:
  shuffled role functional direction control
```

### 4.2 System controls

```text
SYS0-no-compiled-no-prewarm
SYS1-compiled-only
SYS2-prewarm-only
SYS3-compiled-prewarm
SYS4-loss-sync-trim
SYS5-no-sync-optimizer
```

These controls are not success candidates by themselves. They isolate system effects.

### 4.3 Functional controls

```text
FUNC0-no-functional
FUNC1-fixed-FT7
FUNC2-adaptive-FT-P
FUNC3-adaptive-no-task-budget
FUNC4-adaptive-no-role-score
FUNC5-adaptive-no-event-score
FUNC6-random-direction
FUNC7-shuffled-role-direction
FUNC8-matched-overhead-no-op
```

### 4.4 Continual candidates

```text
CL0-DG-Base
CL1-DG-Adaptive-FT-P
CL2-no-old-head-row-restore
CL3-no-stack-anchor
CL4-stack-anchor-006
CL5-stack-anchor-012
CL6-stack-anchor-024
CL7-new-task-bias-control
```

---

## 5. 实验阶段总览

v8.8 分为九个 Wave：

```text
Wave 0:
  selected route independent reproduction

Wave 1:
  system / functional factor isolation

Wave 2:
  robust timing and training compute formalization

Wave 3:
  broad KANbeFair baseline reproduction

Wave 4:
  multi-task external fair transfer

Wave 5:
  cross-task functional causality

Wave 6:
  robustness and calibration

Wave 7:
  continual balance formalization

Wave 8:
  mechanism attribution and boundary route
```

---

## 6. Wave 0：selected route independent reproduction

### P0：fresh reproduction of v8.7 selected route

#### 目标

独立复现 v8.7 selected route，量化 seed / rerun / protocol 稳定性。

#### 设置

```text
datasets = FMNIST, KMNIST, MNIST if available
seeds = 5 independent seeds
reruns = 3 independent out-dirs
timing_protocols = T0,T1,T2,T3,T4
```

#### 必跑对象

```text
KB-MLP
KB-KAN
DG-Base
DG-Adaptive-FT-P
DG-Fixed-FT7
DG-NoOp
DG-RandomFunc
```

#### 必须记录

```text
run_id
seed
dataset
candidate
test_acc
delta_vs_KB_MLP
delta_vs_DG_Base
params
FLOPs_forward
FLOPs_backward_estimate
step_ratio
memory_ratio
train_time_s
kernel_time_ms
curvature_ratio_vs_DG_Base
functional_event_count
functional_update_time_ratio
NoTeacherNoLossNoOffloadPass
FunctionalCausalityPass
fake_data_used
proxy_row_used
```

#### 判断标准

IndependentReproPass：

$$
Q_{50}(Acc_{\text{DG-Adaptive}}-Acc_{\text{KB-MLP}})>0,
$$

$$
Q_{90}(StepRatio)\leq1.50,
$$

$$
Q_{90}(MemoryRatio)\leq1.05,
$$

$$
Q_{50}(CurvatureRatio)\leq0.90.
$$

#### 必须可视化

```text
p0_acc_delta_rerun_boxplot.svg
p0_step_ratio_protocol_boxplot.svg
p0_memory_ratio_protocol_boxplot.svg
p0_curvature_ratio_rerun_boxplot.svg
p0_pass_rate_by_protocol.svg
```

---

## 7. Wave 1：system / functional factor isolation

### P1：system-side factor isolation

#### 目标

证明 system-side repair 解决的是 timing，不是 functional geometry 的来源。

#### 必跑对象

```text
SYS0-no-compiled-no-prewarm
SYS1-compiled-only
SYS2-prewarm-only
SYS3-compiled-prewarm
SYS4-loss-sync-trim
SYS5-no-sync-optimizer
```

#### 必须记录

```text
system_variant
step_ratio
memory_ratio
kernel_time_forward
kernel_time_backward
kernel_time_update
kernel_time_functional
unknown_time_fraction
test_acc
curvature_ratio
functional_events
```

#### 判断标准

System factor pass：

至少一个 system variant 解释 70% 以上 timing gain：

$$
\frac{\Delta StepTime_{\text{factor}}}{\Delta StepTime_{\text{total}}}\geq0.70.
$$

如果 system factor 也显著改变 curvature：

$$
|\Delta R_{\text{curv,system}}|>0.05,
$$

则必须在 route 中记录 system-confounded geometry。

#### 可视化

```text
p1_system_timing_waterfall.svg
p1_system_factor_pareto.svg
p1_system_curvature_confounding.svg
```

### P2：functional factor isolation

#### 目标

证明 adaptive functional update 的收益来自 event score / task budget / role score，而不是 no-op 或 random。

#### 必跑对象

```text
FUNC0-no-functional
FUNC1-fixed-FT7
FUNC2-adaptive-FT-P
FUNC3-adaptive-no-task-budget
FUNC4-adaptive-no-role-score
FUNC5-adaptive-no-event-score
FUNC6-random-direction
FUNC7-shuffled-role-direction
FUNC8-matched-overhead-no-op
```

#### 必须记录

```text
functional_variant
event_score_mean
event_count
role_budget_stack
role_budget_head
bad_step_rate
holdout_descent_ratio
test_acc
curvature_ratio
ECE
NLL
step_ratio
functional_update_time_ratio
```

#### 判断标准

Functional mechanism pass：

$$
R_{\text{curv,adaptive}}<R_{\text{curv,no-op}},
$$

$$
R_{\text{curv,adaptive}}<R_{\text{curv,random}},
$$

and:

$$
Acc_{\text{adaptive}}\geq Acc_{\text{no-op}}-0.002.
$$

Task-budget importance pass：

If removing task budget causes:

$$
BadStepRate_{\text{no-budget}}>BadStepRate_{\text{adaptive}},
$$

or:

$$
Acc_{\text{no-budget}}<Acc_{\text{adaptive}}-0.002,
$$

then task-budget is causally important.

#### 可视化

```text
p2_functional_curvature_bar.svg
p2_functional_task_geometry_pareto.svg
p2_event_score_timeline.svg
p2_role_budget_timeline.svg
p2_bad_step_rate_bar.svg
```

---

## 8. Wave 2：robust timing and training compute formalization

### P3：multi-protocol robust timing

#### 目标

把 v8.7 的 timing pass 推到 formal，多 protocol，不只看一个 high-rep protocol。

#### Protocols

```text
T0 = 50 warmup / 200 reps
T1 = 100 warmup / 500 reps
T2 = 200 warmup / 1000 reps
T3 = train-step-only phase-clean
T4 = full-loop with guard / validation / sync
```

#### 必须记录

```text
protocol_id
candidate
step_ratio
memory_ratio
forward_time
backward_time
base_update_time
functional_update_time
guard_time
cuda_sync_time
validation_time
unknown_time_fraction
kernel_time
```

#### 判断标准

FormalTimingPass：

$$
Q_{90}(StepRatio)\leq1.50.
$$

StrictTimingPass：

$$
\max(StepRatio)\leq1.50.
$$

TimeAccountingPass：

$$
unknown\_time\_fraction\leq0.10.
$$

#### 可视化

```text
p3_step_ratio_by_protocol.svg
p3_time_breakdown_stacked.svg
p3_unknown_time_fraction.svg
p3_q90_timing_gate.svg
```

### P4：training compute fairness

#### 目标

补齐 forward FLOPs 之外的训练计算公平性。

#### 必须记录

```text
candidate
forward_FLOPs
backward_FLOPs_estimate
update_FLOPs_estimate
functional_update_FLOPs_estimate
kernel_time_forward
kernel_time_backward
kernel_time_update
kernel_time_functional
train_step_energy_proxy_if_available
```

#### 判断标准

TrainingComputeFairPass：

至少满足三项中的两项：

$$
ForwardFLOPs_{\text{DG}}\leq1.05ForwardFLOPs_{\text{MLP}},
$$

$$
BackwardFLOPs_{\text{DG}}\leq1.50BackwardFLOPs_{\text{MLP}},
$$

$$
KernelTime_{\text{DG}}\leq1.50KernelTime_{\text{MLP}},
$$

$$
StepTime_{\text{DG}}\leq1.50StepTime_{\text{MLP}}.
$$

#### 可视化

```text
p4_forward_backward_flops_bar.svg
p4_kernel_time_by_phase.svg
p4_training_compute_vs_accuracy.svg
```

---

## 9. Wave 3：broad KANbeFair baseline reproduction

### P5：baseline reproduction across task families

#### 目标

复现 KANbeFair 任务族 baseline，避免因为外部 baseline 未复现而误判 DG-KAN 成败。

#### 任务族

```text
Vision:
  MNIST, Fashion-MNIST, KMNIST, CIFAR-10 if runnable

Tabular / machine learning:
  at least two KANbeFair tabular datasets

Symbolic:
  KANbeFair symbolic formula tasks

NLP:
  one text classification task if runnable

Audio:
  one audio classification task if runnable

Continual:
  at least four class-incremental splits
```

#### 必须记录

```text
task_family
task_name
model_name
reported_metric
measured_metric
absolute_delta_from_reported
relative_delta_from_reported
params
FLOPs
train_time
step_time
peak_memory
seed
reproduction_pass
```

#### 判断标准

Classification reproduction:

$$
|Acc_{\text{repro}}-Acc_{\text{reported}}|\leq0.02.
$$

Symbolic reproduction:

$$
\frac{|RMSE_{\text{repro}}-RMSE_{\text{reported}}|}
{RMSE_{\text{reported}}}\leq0.20.
$$

#### 可视化

```text
p5_reproduction_vs_paper.svg
p5_task_family_reproduction_heatmap.svg
p5_baseline_params_flops_table.md
```

---

## 10. Wave 4：multi-task external fair transfer

### P6：external task transfer

#### 目标

检查 DG-KAN 是否跨出 vision-family。

#### 必跑对象

```text
KB-MLP
KB-KAN
DG-Base
DG-Adaptive-FT-P
DG-Fixed-FT7
DG-NoOp
DG-RandomFunc
```

#### 必须记录

```text
task_family
task_name
candidate
seed
metric
delta_vs_KB_MLP
delta_vs_DG_Base
params_ratio
flops_ratio
training_compute_ratio
step_ratio
memory_ratio
curvature_ratio
ECE
NLL
functional_events
```

#### 判断标准

Task pass:

Classification:

$$
Acc_{\text{DG-Adaptive}}\geq Acc_{\text{KB-MLP}}.
$$

Regression / symbolic:

$$
RMSE_{\text{DG-Adaptive}}\leq RMSE_{\text{KB-MLP}}.
$$

Joint fair pass:

$$
ParamsRatio\leq1.05,
$$

$$
FLOPsRatio\leq1.05,
$$

$$
StepRatio\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

#### 可视化

```text
p6_task_family_win_loss_heatmap.svg
p6_delta_vs_mlp_by_task.svg
p6_task_family_pareto.svg
p6_joint_fair_pass_matrix.svg
```

---

## 11. Wave 5：cross-task functional causality

### P7：functional controls across tasks

#### 目标

证明 functional update 的因果性跨任务成立。

#### 必跑对象

```text
DG-Base
DG-Adaptive-FT-P
DG-NoOp
DG-RandomFunc
DG-ShuffledRoleFunc
```

#### 必须记录

```text
task_name
seed
metric
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

CausalityPass:

$$
R_{\text{curv,DG-Adaptive}}<R_{\text{curv,DG-NoOp}},
$$

$$
R_{\text{curv,DG-Adaptive}}<R_{\text{curv,DG-RandomFunc}},
$$

and:

$$
Metric_{\text{DG-Adaptive}}\geq Metric_{\text{DG-NoOp}}-\epsilon.
$$

Classification default:

$$
\epsilon=0.002.
$$

#### 可视化

```text
p7_causality_curvature_bar_by_task.svg
p7_noop_random_control_matrix.svg
p7_role_contribution_by_task.svg
p7_metric_vs_curvature_scatter.svg
```

---

## 12. Wave 6：robustness and calibration

### P8：robustness / perturbation

#### 目标

检查 functional geometry 是否带来稳定性，而不是只降低 curvature metric。

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
candidate
clean_acc
noisy_acc
accuracy_drop
ECE_under_noise
NLL_under_noise
curvature_under_noise
robustness_auc
functional_events
step_ratio
memory_ratio
```

#### 判断标准

RobustnessPass:

$$
AccDrop_{\text{DG-Adaptive}}\leq AccDrop_{\text{DG-Base}}
$$

for at least three settings.

CalibrationPass:

$$
ECE_{\text{DG-Adaptive}}\leq ECE_{\text{DG-Base}},
$$

or:

$$
NLL_{\text{DG-Adaptive}}\leq NLL_{\text{DG-Base}}
$$

in at least half of tested perturbations.

#### 可视化

```text
p8_accuracy_drop_bar.svg
p8_robustness_auc.svg
p8_ece_under_noise.svg
p8_curvature_under_noise.svg
```

---

## 13. Wave 7：continual balance formalization

### P9：multi-split continual stress

#### 目标

解决 v8.7 broad_strong 未声明的重要缺口：continual balance 尚未真实执行。

#### Splits

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

#### 必跑对象

```text
KB-MLP
KB-KAN
DG-Base
DG-Adaptive-FT-P
DG-Adaptive-no-old-head-restore
DG-Adaptive-no-stack-anchor
DG-Adaptive-new-task-bias-control
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
old_task_acc
new_task_acc
max_min_task_acc_gap
curvature_after_task
functional_events
anchor_strength
```

#### 判断标准

ContinualFormalPass:

$$
Forgetting_{\text{DG-Adaptive}}\leq Forgetting_{\text{DG-Base}},
$$

$$
FinalAvgAcc_{\text{DG-Adaptive}}\geq FinalAvgAcc_{\text{DG-Base}},
$$

and:

$$
\max_i Acc_i-\min_i Acc_i\leq0.20.
$$

#### 可视化

```text
p9_continual_accuracy_matrix.svg
p9_forgetting_by_split.svg
p9_task_balance_bar.svg
p9_curvature_forgetting_scatter.svg
```

---

## 14. Wave 8：mechanism attribution and boundary route

### P10：functional mechanism attribution

#### 目标

解释 adaptive functional update 到底靠哪些 role / event 生效。

#### 必须记录

```text
task_name
task_family
role
role_score
role_budget
role_accept_rate
role_update_norm
role_curvature_delta
role_task_delta
role_system_cost
event_score
event_threshold
event_interval
```

#### 判断标准

MechanismAttributionPass:

$$
\frac{\sum_{i=1}^{k}\Delta R_i}{\Delta R_{\text{total}}}\geq0.70.
$$

即 top roles/events 至少解释 70% geometry reduction。

#### 可视化

```text
p10_role_curvature_waterfall.svg
p10_event_score_timeline.svg
p10_role_budget_by_task.svg
p10_update_norm_vs_geometry_delta.svg
```

### P11：negative boundary audit

#### 目标

如果有任务失败，明确边界。

#### 必须记录

```text
task_family
task_name
DG_vs_MLP_delta
params_ratio
flops_ratio
training_compute_ratio
step_ratio
geometry_delta
failure_reason
boundary_label
```

Boundary labels:

```text
broad_fair_win
vision_family_win
mnist_like_win
symbolic_win
geometry_robustness_win
forward_flops_only
training_compute_fail
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

### P12：final route decision

#### Route cases

```text
R1-BroadExternalFairFunctionalAdvantage:
  At least two non-symbolic task families joint fair pass,
  functional causality pass in majority tasks,
  robust timing pass,
  training compute fair pass,
  continual formal pass.

R2-VisionFamilyExternalFairAdvantage:
  MNIST/FMNIST/KMNIST pass, but broad non-symbolic tasks not enough.

R3-SymbolicGeometryFunctionalAdvantage:
  symbolic / geometry / robustness success, but broad task accuracy not established.

R4-FormalButNotBroad:
  current v8.7 formal selected route reproduced,
  but P10-P13 broad evidence insufficient.

R5-TimingOrComputeBoundary:
  task and geometry pass, but robust timing or training compute fails.

R6-FunctionalCausalityBoundary:
  task passes, but no-op/random controls explain geometry equally well.

R7-ContinualUnbalanced:
  forgetting improves but task balance fails.

R8-NegativeExternalResult:
  DG-KAN loses to MLP in most external tasks.

R9-NoReproduction:
  v8.7 selected route cannot fresh reproduce.

R10-CodeOrContractFail:
  adapter / no-teacher / no-loss / no-fake contract fails.
```

#### route_decision.json 必须记录

```text
route
v87_reproduction_pass
adaptive_controller_pass
robust_timing_pass
training_compute_fair_pass
multi_task_fair_pass_count
functional_causality_pass_count
robustness_pass
continual_balance_pass
mechanism_attribution_pass
boundary_label
best_candidate
best_task_family
primary_blocker
next_required_implementation
success_v88_formal_reproduction
success_v88_broad_external
success_v88_strong
```

---

## 15. Required artifacts

v8.8 必须落盘：

```text
run_manifest.json
v87_fresh_reproduction.csv
system_factor_isolation.csv
functional_factor_isolation.csv
robust_timing_protocols.csv
training_compute_counter.csv
kanbefair_broad_reproduction.csv
external_multitask_transfer.csv
functional_causality_multitask.csv
robustness_perturbation.csv
continual_balance_multisplit.csv
functional_mechanism_attribution.csv
negative_boundary_audit.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy:

```text
F1_reproduction_fail
F2_system_factor_confounded
F3_functional_causality_fail
F4_robust_timing_fail
F5_training_compute_fair_fail
F6_baseline_reproduction_fail
F7_multitask_external_fail
F8_robustness_fail
F9_continual_unbalanced
F10_mechanism_attribution_incomplete
F11_geometry_no_downstream_gain
F12_teacher_or_loss_violation
F13_fake_or_proxy_violation
F14_adapter_or_counter_fail
F15_artifact_missing
```

---

## 16. 第一轮执行顺序

### Step 1：fresh reproduction

先执行 P0。如果 v8.7 selected route 不能 fresh reproduce，不进入 broad claim。

### Step 2：system / functional factor isolation

执行 P1/P2。确认 compiled/prewarm/system repair 与 functional geometry update 的因果边界。

### Step 3：formal timing / compute

执行 P3/P4。补多 protocol timing 与 training compute fairness。

### Step 4：broad KANbeFair reproduction

执行 P5。外部 baseline 复现不过，不评价 DG-KAN broad success。

### Step 5：multi-task transfer

执行 P6。把当前 vision-family result 推向更多 task families。

### Step 6：functional causality cross-task

执行 P7。no-op/random/shuffled role controls 不能只在 FMNIST/KMNIST 上过。

### Step 7：robustness / continual

执行 P8/P9。补齐 v8.7 未声明 broad strong 的关键缺口。

### Step 8：mechanism / boundary

执行 P10/P11/P12。输出 broad success、vision-family success、symbolic/geometry success 或 negative result。

---

## 17. 停止条件

### 17.1 成功停止

Broad strong success:

```text
fresh reproduction pass
robust timing pass
training compute fair pass
at least two non-symbolic task families joint fair pass
functional causality pass in majority tasks
robustness or continual formal pass
mechanism attribution pass
```

Formal reproduction success:

```text
fresh reproduction pass
system / functional factor isolation pass
robust timing pass
```

Task-family success:

```text
vision-family / symbolic tasks pass
but broad non-symbolic tasks not enough
```

也可停止，并写边界结论。

### 17.2 失败停止

```text
1. v8.7 selected route cannot reproduce；
2. functional causality fails against no-op/random controls；
3. robust timing Q90 step ratio > 1.50；
4. training compute fairness fails；
5. KANbeFair broad baseline cannot reproduce；
6. DG-KAN loses to MLP by > 1% on most non-symbolic tasks；
7. robustness and continual both fail；
8. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 18. 最终解释规则

### Case A：Broad external fair pass

可声明：

```text
DG-KAN + adaptive functional update establishes broad external fair advantage over MLP on tested task families.
```

### Case B：Vision-family pass only

必须声明：

```text
DG-KAN functional advantage is currently strongest in MNIST-like vision-family tasks.
```

### Case C：Geometry improves but task does not

必须声明：

```text
Functional update gives geometry advantage, but broad task superiority is not established.
```

### Case D：Continual improves forgetting but balance fails

必须声明：

```text
Anti-forgetting update reduces forgetting but does not solve balanced class-incremental continual learning.
```

### Case E：External negative result

必须声明：

```text
Internal and vision-family success do not generalize to broader external fair comparison.
```

---

## 19. 最终建议

v8.8 的一句话策略是：

$$
\boxed{
\text{把 v8.7 formal selected route 从 vision-family 成功扩展为可复现、因果明确、跨任务公平、边界清楚的外部结论。}
}
$$

现在不该做：

```text
继续只在 FMNIST/KMNIST/MNIST 上调参；
回 teacher；
改 loss；
做 optimizer sweep；
把 success_v87_formal 直接包装成 broad strong。
```

现在应该做：

```text
1. independent reproduction；
2. system/functional factor isolation；
3. robust timing and training compute fairness；
4. broad KANbeFair reproduction；
5. multi-task external fair transfer；
6. cross-task functional causality；
7. robustness and continual balance；
8. mechanism attribution and boundary audit。
```

最终目标不是强行证明 DG-KAN 全面优于 MLP，而是：

$$
\boxed{
\text{清楚回答 DG-KAN + adaptive functional update 在哪些公平条件和任务族上真正优于 MLP，在哪些条件下不优。}
}
