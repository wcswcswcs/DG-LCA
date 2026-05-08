# DG-KAN v8.3 System-Gated Functional Update Re-Entry：系统闭合后的函数空间更新与几何优势完整实验计划

> 本计划是对当前 v8.x 路线的重新收束。  
> 用户明确要求：**坚持 functional update，因为它是更好几何性质的基础；但不能回到 teacher、self-teacher、distillation、改 loss、sampler/class weight 或 optimizer 超参大扫。**  
> 因此 v8.3 的原则是：  
> **先用 CE-only / no-teacher / no-loss / strict PureKAN 路线闭合系统候选，再在这个系统候选上重新引入 functional update，并把 functional update 定义为函数空间几何更新机制，而不是更快的 AdamW 实现。**

---

## 0. 当前状态与问题定位

### 0.1 当前还没有达到最终目标

当前最接近目标的结果来自 v8.2 的 CE-only / no-teacher / no-loss 路线。v8.2 已经证明了三个重要事实：

```text
KC6:
  质量路径成立，macro gap = +0.024349，超过 +0.0200 hard gate；
  但 FullGridS2 不成立，memory max = 1.053081，step max = 1.513245，S2 shapes = 5/9。

KF4:
  系统路径成立，memory max = 1.046778，step max = 1.339375，S2 shapes = 9/9；
  但 macro gap = +0.018359，没有过 +0.0200 hard gate。

KF10:
  质量与 step 已接近合流，macro gap = +0.020313，step max = 1.388334；
  但 memory max = 1.053081，S2 shapes = 6/9。
```

所以当前不是 “KAN 没有表达力”，也不是 “必须靠 teacher / loss 才能过线”。当前准确的 blocker 是：

$$
\boxed{
\text{CE-only 质量路径与系统路径尚未稳定合流。}
}
$$

### 0.2 现在不应把 functional update 混入未闭合系统候选

v6.3/v6.4 早期 functional update 的经验非常重要：当时 `FunctionalDiag`、`FunctionalDataDiag`、`FunctionalResidualSobolev`、`FunctionalCoordinateAdam`、`FunctionalCoordinateAdanLite` 都没有通过质量 gate；最好的 FunctionalCoordinateAdam 也只有 `cos ≈ 0.49`、`bad ≈ 0.06`、`holdout ≈ 0.93`，而 ManualAdamW baseline 是 `cos = 1.0`、`bad = 0`、`holdout = 1.0`。这说明当时的 functional-no-autograd update 还不能作为主 task direction。

但这不意味着 functional update 路线应该放弃。更合理的解释是：functional update 不能在系统和任务候选都未稳定时承担所有训练任务。v8.3 把 functional update 重新定义为：

$$
\boxed{
\text{在已过系统 gate 的 PureKAN primitive 上，进行函数空间几何改良和 task-aware 更新。}
}
$$

也就是说，functional update 不再作为 “替代一切 AdamW 的裸更新” 直接上 full run；它要经过 direction audit、trust-region、geometry Pareto 和 task preservation gate。

### 0.3 v8.3 的目标分两层

v8.3 不是只做系统，也不是只做 functional update。它是一个 **system-gated functional re-entry** 计划：

```text
第一层：
  继续完成 CE-only/no-teacher/no-loss 系统候选闭合。
  目标是找到至少一个 macro + FullGridS2 candidate。

第二层：
  在这个候选上重新引入 functional update。
  目标是证明 functional update 能提供几何优势，并且不破坏 task、S2 和 TimeAUC。
```

因此 v8.3 的核心路线是：

$$
\boxed{
\text{CE-only system survivor}
\rightarrow
\text{functional direction audit}
\rightarrow
\text{task-aware functional update}
\rightarrow
\text{geometry-task-system Pareto confirmation}
}
$$

---

## 1. v8.3 总体目标

v8.3 的总目标是：

$$
\boxed{
\text{在严格 PureKAN、CE-only、no-teacher 条件下，建立可用的 functional update 路线。}
}
$$

更具体地说，v8.3 要回答两个问题：

### Q1：系统候选是否已经足够稳定，可以承载 functional update？

functional update 不能建立在一个系统不稳定的 primitive 上。v8.3 必须先得到或确认一个满足以下条件的 base candidate：

$$
\Delta Acc_{\text{macro,val}}\geq0.0200,
$$

$$
r_{\text{mem,max}}\leq1.05,
$$

$$
r_{\text{step,max}}\leq1.50,
$$

并且：

```text
NoTeacherNoLossPass = 1
StrictPass = 1
GradPass = 1
```

如果没有这样的 candidate，functional update 只能做 one-step / diagnostic，不进入 full task confirmation。

### Q2：functional update 是否能带来几何优势而不损伤任务和系统？

在 base candidate 上，functional update 必须满足：

$$
\Delta Acc_{\text{functional}} \geq \Delta Acc_{\text{base}} - 0.005,
$$

$$
ECE_{\text{functional}}\leq ECE_{\text{base}}+0.005,
$$

$$
NLL_{\text{functional}}\leq NLL_{\text{base}}+0.01,
$$

并且几何指标至少改善一项：

$$
R_{\text{curv,functional}}\leq0.8R_{\text{curv,base}},
$$

或：

$$
\phi'_{p95,\text{functional}}\leq0.9\phi'_{p95,\text{base}},
$$

或：

$$
J_{\text{norm,functional}}\leq0.9J_{\text{norm,base}}.
$$

系统开销不能破坏：

$$
r_{\text{mem,max,functional}}\leq1.05,
$$

$$
r_{\text{step,max,functional}}\leq1.50,
$$

$$
\frac{T_{\text{functional update}}}{T_{\text{base update}}}\leq1.10.
$$

---

## 2. 本轮明确禁止事项

v8.3 继续严格禁止以下内容进入 official route：

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
test-based selection
CPU offload
optimizer hyperparameter sweep
```

v8.3 允许的 functional update 不是改 loss。它是 update rule：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{\text{task}}
+
\lambda_t\Delta\theta_{\text{func}},
$$

其中 task loss 仍然是标准 CE：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

允许 functional update 的条件是：

```text
loss_type = CE
geometry_loss_used = 0
functional_update_used = 1
functional_update_is_update_rule = 1
special_loss_used = 0
external_teacher_used = 0
self_teacher_used = 0
```

换句话说，functional update 只能改变参数更新方向，不能改变训练 objective，也不能引入 teacher signal。

---

## 3. 核心假设

### H0：functional update 必须等待系统 base candidate

H0 是前置假设。functional update 只有在系统 base candidate 通过 minimum gate 后，才允许进入 full task。

H0 成立标准：

存在 base candidate 满足：

$$
\Delta Acc_{\text{macro,val}}\geq0.0200,
$$

$$
r_{\text{mem,max}}\leq1.05,
$$

$$
r_{\text{step,max}}\leq1.50,
$$

并且：

```text
NoTeacherNoLossPass = 1
StrictPass = 1
GradPass = 1
```

如果 H0 不成立，则只执行 P3 one-step functional direction audit 和 P4 geometry diagnostic，不跑 full functional training。

### H1：早期 functional update 失败是因为它被当成主 task direction，而不是因为 functional update 本身无价值

早期 functional update 的方向质量不够，FunctionalCoordinateAdam 的 cosine 只有约 `0.49`，bad step rate 约 `0.06`，holdout descent 约 `0.93`。这说明裸 functional update 不适合直接替代 task optimizer，但可能适合作为 task-aware correction。

v8.3 的 functional update 不直接替代 task direction，而是：

$$
d_{\text{v83}}
=
d_{\text{task}}
+
\lambda_t P_{\mathcal{T}}(d_{\text{func}}),
$$

其中 $P_{\mathcal{T}}$ 是 task trust-region projection。它保证 functional correction 不逆着 task descent。

H1 成立标准：

在 one-step audit 中，functional-corrected direction 满足：

$$
\cos(d_{\text{v83}},d_{\text{task}})\geq0.85,
$$

$$
\frac{\operatorname{HoldoutDescent}(d_{\text{v83}})}
{\operatorname{HoldoutDescent}(d_{\text{task}})}
\geq0.95,
$$

$$
BadStepRate(d_{\text{v83}})\leq0.02.
$$

如果这些不满足，functional update 不能进入 multi-step task。

### H2：functional update 的主要价值是几何，而不是 final accuracy

H2 假设 functional update 不一定显著提高 accuracy，但应该改善函数几何。

几何指标包括：

```text
edge smoothness norm
edge curvature norm
finite-difference curvature
basis slope p95
Jacobian norm
local Lipschitz estimate
Sobolev residual norm
function displacement R2
path length
margin stability
```

H2 成立标准：

在不损伤 task 的情况下，至少一个几何指标改善：

$$
R_{\text{curv,functional}}\leq0.8R_{\text{curv,base}},
$$

或：

$$
\phi'_{p95,\text{functional}}\leq0.9\phi'_{p95,\text{base}},
$$

或：

$$
J_{\text{norm,functional}}\leq0.9J_{\text{norm,base}}.
$$

同时：

$$
\Delta Acc_{\text{functional}}\geq\Delta Acc_{\text{base}}-0.005.
$$

### H3：task-aware functional update 可以避免几何-性能冲突

之前发现保几何可能伤害模型性能。H3 假设冲突来自 “几何更新无视 task descent”。通过 projection、trust region、event trigger、late-phase scheduling 可以减少冲突。

H3 成立标准：

与 naive functional update 相比，task-aware functional update 同时满足：

$$
BadStepRate_{\text{task-aware}}\leq0.5BadStepRate_{\text{naive}},
$$

$$
AccDrop_{\text{task-aware}}\leq0.5AccDrop_{\text{naive}},
$$

并且几何改善至少保留：

$$
\frac{\Delta R_{\text{geo,task-aware}}}{\Delta R_{\text{geo,naive}}}\geq0.5.
$$

### H4：functional update 不应破坏 S2 / TimeAUC

functional update 不能让系统退化。H4 成立标准：

$$
r_{\text{mem,max,functional}}\leq1.05,
$$

$$
r_{\text{step,max,functional}}\leq1.50,
$$

$$
\frac{T_{\text{functional update}}}{T_{\text{base update}}}\leq1.10.
$$

如果 functional update 的几何收益来自很重的 metric solve 或 full Jacobian materialization，则它不能进入 official route。

### H5：functional update 的方向质量要按 edge role 分析

KAN 参数不是普通 MLP 参数。不同 role 的 functional update 可能不同：

```text
input edge
stack mix
basis / residual coefficients
head edge
rational numerator / denominator
DWM2 pre/post mix
```

H5 成立标准：

必须记录 `role_update_share`、`role_descent_contribution`、`role_geometry_reduction`。如果某个 role 占据过大 update 但没有贡献 descent 或 geometry，则要对该 role 降权或冻结。

---

## 4. Candidate 设计

### 4.1 Base candidates

v8.3 的 base candidates 来自当前 CE-only system line：

```text
B0-MLP-AdamW-CE
KC6-quality-path-baseline
KW3-memory-repaired-quality-path
KW6-step-repaired-quality-path
KF4-system-path-baseline
KF10-quality-step-memory-fail-intermediate
best-KS-from-step-repair-if-available
best-KQ-from-KC6-fallback-if-available
```

其中只有满足 H0 的 candidate 才能进入 full functional update。

### 4.2 Functional direction candidates

functional update 候选分三类。

#### A 类：naive functional baseline，只做诊断

```text
F0-ManualAdamW-equivalent-baseline
F1-FunctionalDiag
F2-FunctionalDataDiag
F3-FunctionalResidualSobolev
F4-FunctionalCoordinateAdam
F5-FunctionalCoordinateAdanLite
```

这些用于复现早期失败，不能直接进入 official route，除非通过 H1。

#### B 类：task-aware functional correction

```text
FT0-TaskProjectedFunctionalDiag
FT1-TaskProjectedDataSobolev
FT2-TaskProjectedResidualSobolev
FT3-TrustRegionFunctionalCoordinateAdam
FT4-TrustRegionFunctionalCoordinateAdanLite
FT5-EventTriggeredFunctionalCorrection
FT6-LatePhaseFunctionalCorrection
FT7-RoleWiseFunctionalCorrection
```

这些是 v8.3 的主线。

#### C 类：functional replacement exploratory

```text
FR0-PureFunctionalDiag
FR1-PureFunctionalCoordinateAdam
FR2-PureFunctionalCoordinateAdanLite
FR3-PureFunctionalNaturalDiag
```

这些只做短程探索。它们必须先通过 very strict one-step gate 才能进入 longer smoke。它们不作为 v8.3 minimum success 主线。

### 4.3 Functional geometry metrics

对不同 primitive 使用不同几何指标。

#### RBF / ABRBF edge

$$
R_{\text{smooth}}
=
\sum_{i,o,q}
(\theta_{i,o,q+1}-\theta_{i,o,q})^2.
$$

$$
R_{\text{curv}}
=
\sum_{i,o,q}
(\theta_{i,o,q+2}-2\theta_{i,o,q+1}+\theta_{i,o,q})^2.
$$

#### SparseInterp / spline edge

$$
R_{\text{knot}}
=
\sum_{i,q}
(v_{i,q+1}-v_{i,q})^2.
$$

$$
R_{\text{knot-curv}}
=
\sum_{i,q}
(v_{i,q+2}-2v_{i,q+1}+v_{i,q})^2.
$$

#### DWM2 / rational edge

Use finite-difference function probes:

$$
R_{\text{func-curv}}
=
\mathbb{E}_{x,\epsilon}
\left[
\frac{
\phi(x+\epsilon)-2\phi(x)+\phi(x-\epsilon)
}{\epsilon^2}
\right]^2.
$$

#### Model-level geometry

$$
J_{\text{norm}}
=
\mathbb{E}_x
\|\nabla_x f_\theta(x)\|_F^2.
$$

$$
Lip_{\text{local}}
=
\mathbb{E}_{x,\delta}
\frac{
\|f_\theta(x+\delta)-f_\theta(x)\|_2
}{
\|\delta\|_2
}.
$$

---

## 5. 实验阶段总览

v8.3 分为七个 Wave：

```text
Wave 0:
  System base confirmation and contract lock

Wave 1:
  Functional update implementation audit

Wave 2:
  One-step functional direction audit

Wave 3:
  Multi-step trajectory and trust-region smoke

Wave 4:
  Functional task + geometry co-selection

Wave 5:
  5-seed / 10-seed confirmation

Wave 6:
  System / TimeAUC / scaling / robustness closure
```

---

## 6. Wave 0：System base confirmation and contract lock

### P0：No-teacher / no-loss / functional contract

#### 目标

确认所有 base 与 functional candidates 都满足本轮 contract。functional update 不能变成 special loss。

#### 必须记录

```text
candidate_id
base_candidate_id
functional_update_used
functional_update_type
functional_update_is_update_rule
loss_type
geometry_loss_used
special_loss_used
external_teacher_used
self_teacher_used
teacher_logits_used
sampler_changed
class_weight_used
cpu_offload_used
nonKAN_param_count
manual_forward
manual_backward
manual_update
uses_loss_backward
fake_data_used
proxy_row_used
```

#### 判断标准

Official functional candidate 必须满足：

```text
loss_type = CE
geometry_loss_used = 0
functional_update_is_update_rule = 1
external_teacher_used = 0
self_teacher_used = 0
special_loss_used = 0
cpu_offload_used = 0
nonKAN_param_count = 0
manual_forward = 1
manual_backward = 1
manual_update = 1
```

#### 可视化

```text
p0_contract_heatmap.svg
p0_functional_route_matrix.svg
p0_teacher_loss_violation_matrix.svg
```

### P1：System base candidate confirmation

#### 目标

确认 base candidate 是否满足 H0。没有 H0，functional update 不进入 full task。

#### 必跑对象

```text
B0
KC6
KW3
KW6
KF4
KF10
best-KS
best-KQ
```

#### 必须记录

```text
macro_gap
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

H0 pass：

$$
\Delta Acc_{\text{macro,val}}\geq0.0200,
$$

$$
r_{\text{mem,max}}\leq1.05,
$$

$$
r_{\text{step_max}}\leq1.50.
$$

如果没有 candidate 满足 H0，则只执行 P2/P3 one-step functional audit，不执行 full functional task.

#### 可视化

```text
p1_base_candidate_scorecard.svg
p1_quality_system_pareto.svg
p1_s2_shape_heatmap.svg
```

---

## 7. Wave 1：Functional update implementation audit

### P2：Functional operator implementation audit

#### 目标

确认 functional update 不是 loss、不是 autograd trick、不是 CPU heavy metric solve。它必须是 graph-free update rule。

#### 必须记录

```text
functional_update_type
uses_autograd_for_functional_metric
uses_loss_backward
uses_full_jacobian_materialization
uses_cpu_solve
metric_build_time_ms
metric_solve_time_ms
functional_update_time_ms
functional_memory_peak_MB
metric_condition_number
metric_solve_residual
fallback_rate
```

#### 判断标准

Implementation pass：

```text
uses_loss_backward = 0
uses_full_jacobian_materialization = 0
functional_update_time_ms <= 0.10 * base_step_time_ms
functional_memory_peak_ratio <= 1.05
metric_solve_residual <= 1e-4
```

若 functional rule 需要 full Jacobian materialization 或 CPU solve，则只能作为 diagnostic，不进入 official.

#### 可视化

```text
p2_functional_update_cost_bar.svg
p2_metric_condition_hist.svg
p2_metric_solve_residual.svg
p2_functional_memory_overhead.svg
```

---

## 8. Wave 2：One-step functional direction audit

### P3：One-step direction quality

#### 目标

先判断 functional direction 是否能在单步上保持 task descent。早期 functional update 失败主要是方向质量不够，因此这一步必须严格。

#### 必跑对象

```text
F0 ManualAdamW-equivalent baseline
F1 FunctionalDiag
F2 FunctionalDataDiag
F3 FunctionalResidualSobolev
F4 FunctionalCoordinateAdam
F5 FunctionalCoordinateAdanLite
FT0-FT7 task-aware variants
FR0-FR3 exploratory pure functional variants
```

#### 数据

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
batches per dataset = 20
holdout batches per dataset = 20
```

#### 必须记录

```text
manual_grad_norm
task_direction_norm
functional_direction_norm
corrected_direction_norm
cos_functional_with_task
cos_corrected_with_task
predicted_train_descent
actual_train_descent
actual_holdout_descent
holdout_descent_ratio_vs_task
bad_step_flag
bad_step_rate
update_over_param_norm
trust_region_scale
clip_rate
fallback_rate
role_update_share
role_descent_contribution
role_geometry_reduction
```

#### 判断标准

Functional direction pass：

$$
\cos(d_{\text{corrected}},d_{\text{task}})\geq0.85.
$$

$$
\frac{
\operatorname{HoldoutDescent}(d_{\text{corrected}})
}{
\operatorname{HoldoutDescent}(d_{\text{task}})
}
\geq0.95.
$$

$$
BadStepRate\leq0.02.
$$

Pure functional exploratory pass 更严格：

$$
BadStepRate\leq0.01,
$$

$$
\cos(d_{\text{functional}},d_{\text{task}})\geq0.75.
$$

若未通过，只能记录 diagnostic，不进入 P4。

#### 可视化

```text
p3_direction_cosine_heatmap.svg
p3_predicted_vs_actual_descent.svg
p3_bad_step_timeline.svg
p3_role_update_share_stacked.svg
p3_holdout_descent_ratio_boxplot.svg
```

---

## 9. Wave 3：Multi-step trajectory and trust-region smoke

### P4：20-step / 50-step functional smoke

#### 目标

验证 one-step pass 的 functional update 是否能在多步训练中不破坏轨迹，并产生几何改善。

#### 必跑对象

```text
base candidate
base + FT top3 from P3
base + naive functional best diagnostic
base + pure functional exploratory if P3 passed
```

#### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
steps = 20 and 50
trace_every = 1 or 5
```

#### 必须记录

```text
train_loss
val_loss
val_acc
ECE
NLL
margin_p10
geometry_smoothness
geometry_curvature
jacobian_norm
local_lipschitz
functional_update_norm
trust_region_scale
fallback_rate
bad_step_rate
trajectory_param_rel_l2_vs_base
trajectory_logit_KL_vs_base
geometry_reduction_vs_base
```

#### 判断标准

Multi-step pass：

$$
AccDrop_{50}\leq0.003.
$$

$$
ValLossIncrease_{50}\leq0.005.
$$

$$
BadStepRate_{50}\leq0.02.
$$

$$
\frac{R_{\text{geo,functional},50}}{R_{\text{geo,base},50}}\leq0.90.
$$

进入 P5 的候选必须满足 task preservation 与 geometry improvement 二者。

#### 可视化

```text
p4_loss_curve_overlay.svg
p4_accuracy_curve_overlay.svg
p4_geometry_curve_overlay.svg
p4_logit_kl_trajectory.svg
p4_trust_region_scale_timeline.svg
```

---

## 10. Wave 4：Functional task + geometry co-selection

### P5：3-seed full task functional co-selection

#### 目标

在已通过 P3/P4 的 functional candidates 上进行完整 240-step task run，比较 task、geometry、system 三维 Pareto。

#### 必跑对象

```text
B0 MLP-AdamW reference
base PureKAN candidate
base + FT best1
base + FT best2
base + FT best3
base + naive functional diagnostic if useful
```

#### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
task_steps = 240
bench batch = 128,256,512
trace_every = 10
```

#### 必须记录 task metrics

```text
val_acc
test_acc
macro_val_gap
test_gap
CI95_low_3seed
ECE
AdaptiveECE
NLL
Brier
margin_mean
margin_p10
hard_sample_acc
class_pair_confusion
ValLossAUC_step
ValLossAUC_time
time_to_target_acc
time_to_target_loss
```

#### 必须记录 geometry metrics

```text
edge_smoothness_norm
edge_curvature_norm
finite_difference_curvature_p95
phi_slope_p95
jacobian_norm
local_lipschitz
sobolev_residual_norm
function_displacement_R2
path_length
geometry_reduction_vs_base
```

#### 必须记录 system metrics

```text
memory_ratio_max
step_ratio_max
forward_ratio_mean
backward_ratio_mean
update_ratio_mean
functional_update_time_ms
functional_metric_build_time_ms
functional_metric_solve_time_ms
S2_shape_count
kernel_count_total if profiler available
```

#### 判断标准

Functional useful：

候选必须相对 base 至少满足一个主要收益，同时不破坏硬约束。

主要收益之一：

$$
R_{\text{curv,functional}}\leq0.8R_{\text{curv,base}},
$$

或：

$$
ECE_{\text{functional}}\leq ECE_{\text{base}}-0.005,
$$

或：

$$
ValLossAUC_{\text{step,functional}}\leq0.98ValLossAUC_{\text{step,base}},
$$

或：

$$
Acc_{\text{functional}}\geq Acc_{\text{base}}+0.003.
$$

硬约束：

$$
Acc_{\text{functional}}\geq Acc_{\text{base}}-0.005.
$$

$$
r_{\text{mem,max,functional}}\leq1.05.
$$

$$
r_{\text{step,max,functional}}\leq1.50.
$$

$$
\frac{T_{\text{functional update}}}{T_{\text{base update}}}\leq1.10.
$$

#### 可视化

```text
p5_task_geometry_pareto.svg
p5_accuracy_vs_curvature.svg
p5_ece_vs_curvature.svg
p5_val_auc_vs_geometry.svg
p5_system_overhead_bar.svg
p5_hard_sample_delta.svg
```

---

## 11. Wave 5：5-seed / 10-seed confirmation

### P6：5-seed functional confirmation

#### 目标

只对 P5 survivors 做 5-seed confirmation。

#### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
task_steps = 240
```

#### 必须记录

```text
macro_val_gap
CI95_low
Holm_p
test_gap
ECE_delta_vs_base
NLL_delta_vs_base
geometry_delta_vs_base
memory_ratio_max
step_ratio_max
S2_shape_count
TimeAUC_ratio
functional_overhead_ratio
```

#### 进入 P7 条件

$$
Acc_{\text{functional}}\geq Acc_{\text{base}}-0.005,
$$

$$
R_{\text{geo,functional}}\leq0.9R_{\text{geo,base}},
$$

$$
S2ShapeCount=9.
$$

### P7：10-seed final functional confirmation

#### 目标

验证 functional update 是否能作为 stable training route。

#### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
task_steps = 240
bench full-grid
bootstrap reps = 10000
```

#### 成功标准

Functional geometry success：

$$
Acc_{\text{functional}}\geq Acc_{\text{base}}-0.005,
$$

$$
CI_{95\%,AccDiff}^{low}\geq-0.005,
$$

$$
R_{\text{curv,functional}}\leq0.9R_{\text{curv,base}},
$$

$$
ECE_{\text{functional}}\leq ECE_{\text{base}}+0.005,
$$

$$
r_{\text{mem,max,functional}}\leq1.05,
$$

$$
r_{\text{step,max,functional}}\leq1.50.
$$

Strong functional success：

$$
Acc_{\text{functional}}\geq Acc_{\text{base}},
$$

$$
R_{\text{curv,functional}}\leq0.8R_{\text{curv,base}},
$$

$$
ValLossAUC_{\text{time,functional}}\leq ValLossAUC_{\text{time,base}}.
$$

#### 可视化

```text
p7_10seed_acc_delta_forest.svg
p7_10seed_geometry_delta_forest.svg
p7_seedwise_task_geometry_scatter.svg
p7_system_pareto_final.svg
```

---

## 12. Wave 6：System / TimeAUC / scaling / robustness closure

### P8：TimeAUC and profiler for functional survivor

#### 目标

functional update 不能只改善几何，必须保持 wall-clock competitiveness。

#### 必须记录

```text
ValLossAUC_step
ValLossAUC_time
train_step_time
forward_time
backward_time
update_time
functional_update_time
functional_metric_build_time
functional_metric_solve_time
validation_time
logging_time
unknown_time_fraction
kernel_count_total
mapped_kernel_time_fraction
small_kernel_count_under_10us
```

#### 判断标准

$$
ValLossAUC_{\text{time,functional}}\leq1.05ValLossAUC_{\text{time,base}}.
$$

$$
unknown\_time\_fraction\leq0.10.
$$

$$
functional\_update\_time\leq0.10T_{\text{step}}.
$$

#### 可视化

```text
p8_time_breakdown_stacked.svg
p8_functional_overhead_timeline.svg
p8_step_auc_vs_time_auc.svg
p8_kernel_count_by_phase.svg
```

### P9：Scaling / robustness with functional update

#### 目标

验证 functional geometry 是否带来更稳健的 generalization，而不是只让几何指标好看。

#### 设置

```text
train_size = 256,512,1024,1536,4096
label_noise = 0.05,0.10,0.20
input_noise = 0.05,0.10
seeds = 0..4 for scaling smoke
```

#### 必须记录

```text
train_size
noise_type
noise_level
val_acc
test_acc
accuracy_drop
ECE
NLL
geometry_metrics
sample_efficiency_auc
robustness_auc
memory_ratio
step_ratio
```

#### 判断标准

Sample efficiency benefit：

$$
AUC_{\text{data,functional}}\geq AUC_{\text{data,base}}.
$$

Robustness benefit：

$$
AccDrop_{\text{functional}}\leq AccDrop_{\text{base}}
$$

for at least two noise settings.

Geometry-generalization consistency：

$$
\Delta R_{\text{geo}}<0
$$

and:

$$
\Delta ECE\leq0
$$

or:

$$
\Delta NLL\leq0.
$$

#### 可视化

```text
p9_accuracy_vs_train_size.svg
p9_robustness_auc.svg
p9_geometry_vs_noise.svg
p9_ece_nll_vs_geometry.svg
```

---

## 13. Route decision

v8.3 的 route 不再把 functional update 与 AdamW implementation repair 混在一起。必须明确区分。

### Survivor types

```text
S0:
  Base CE-only system candidate + functional update strong geometry success + S2 + TimeAUC pass.

S1:
  Base CE-only system candidate + functional update geometry success + S2, TimeAUC not yet pass.

S2:
  Base candidate system pass, functional update one-step pass, but full task not opened.

S3:
  Functional update improves geometry but hurts task beyond allowed drop.

S4:
  Functional update task-neutral but no geometry benefit.

S5:
  Functional direction fails one-step descent gate.

S6:
  No base system candidate, functional full task gated.

S7:
  Functional implementation too expensive.

S8:
  Contract violation.
```

### route_decision.json 必须记录

```text
route
base_candidate_id
functional_candidate_id
base_macro_gap
functional_macro_gap
acc_delta_vs_base
geometry_delta_vs_base
ECE_delta_vs_base
NLL_delta_vs_base
memory_ratio_max
step_ratio_max
functional_update_time_ratio
bad_step_rate
holdout_descent_ratio
functional_direction_cos
S2_pass
TimeAUC_pass
ProfilerPass
primary_blocker
next_required_implementation
```

---

## 14. 必须落盘 artifact

```text
run_manifest.json
candidate_registry_v83_functional.csv
contract_no_teacher_no_loss_functional.csv
functional_update_contract.csv
p0_contract_audit.csv
p1_base_candidate_confirmation.csv
p2_functional_operator_audit.csv
p3_one_step_functional_direction.csv
p3_one_step_functional_direction_raw.csv
p4_multistep_functional_smoke.csv
p5_functional_task_geometry_coselection.csv
p5_functional_task_geometry_raw.csv
p6_functional_confirm5.csv
p6_functional_confirm5_raw.csv
p6_functional_fullgrid_s2.csv
p7_functional_confirm10.csv
p7_functional_confirm10_raw.csv
p8_functional_time_profiler.csv
p8_functional_time_profiler_raw.csv
p9_functional_scaling_robustness.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_no_base_system_candidate
F2_teacher_or_loss_violation
F3_functional_uses_loss_backward
F4_functional_full_jacobian_materialization
F5_functional_direction_bad_cosine
F6_functional_bad_step_rate_high
F7_functional_holdout_descent_fail
F8_functional_geometry_no_gain
F9_functional_task_drop
F10_functional_system_overhead
F11_grad_fail
F12_s2_fail
F13_time_auc_fail
F14_fake_or_proxy_violation
F15_artifact_missing
```

---

## 15. 第一轮执行顺序

### Step 1：确认 base system candidate

先执行 P0/P1。如果 KW/KC 系列仍没有 minimum candidate，functional update 只做 one-step diagnostic，不进入 full task。

### Step 2：实现 functional update contract

执行 P2，确保 functional update 是 update rule，不是 loss，不是 autograd trick，不是 CPU-heavy solver。

### Step 3：one-step direction audit

执行 P3。只有通过：

$$
\cos(d_{\text{corrected}},d_{\text{task}})\geq0.85,
$$

$$
BadStepRate\leq0.02,
$$

$$
HoldoutDescentRatio\geq0.95
$$

的 candidate 才能进入 P4。

### Step 4：multi-step smoke

执行 P4，观察 20/50 step 的 task、geometry、trajectory。凡是 task drop 超标或 geometry 无收益的 candidate 立即停止。

### Step 5：3-seed co-selection

执行 P5，做 task-geometry-system 三维 Pareto。

### Step 6：5-seed / 10-seed confirm

只确认 P5 survivor。

### Step 7：TimeAUC / scaling / robustness

只对 10-seed functional survivor 打开。

---

## 16. 成功与失败解释规则

### Case A：functional update 保持 task 并显著改善几何

这是 v8.3 的核心成功。说明 functional update 可以作为 PureKAN 的几何优势机制继续推进。

### Case B：functional update 改善几何但伤 task

说明几何和性能冲突仍存在，需要改 projection / trust region / role weighting，而不是直接放弃 functional update。

### Case C：functional update 不伤 task但也无几何收益

说明当前 functional metric 太弱或没有对准几何指标，需要重新设计 metric。

### Case D：functional direction one-step 就失败

说明当前 functional update 还不能进入 full task。下一步回到 direction construction，不跑大任务。

### Case E：没有 base system candidate

说明 v8.3 functional full-task 必须 gate。此时继续做 system closure，functional 只保留 one-step diagnostic。

### Case F：functional update 破坏 S2 / TimeAUC

说明 metric/update 实现太重，需要做 low-rank/diagonal/role-wise approximation，不应直接使用 dense functional metric。

---

## 17. 最终建议

v8.3 的一句话策略是：

$$
\boxed{
\text{先系统闭合，再以 task-aware functional update 重新引入几何优势。}
}
$$

这不是回到 teacher，也不是改 loss，也不是 optimizer sweep。它是把项目最初的 functional training 目标重新接回当前最强的 CE-only PureKAN 系统路线。

当前阶段应这样定位：

```text
KC/KW/KF line:
  负责得到可用的 CE-only/no-teacher system base。

Functional line:
  不再裸替代 AdamW；
  先做 task-aware correction；
  以几何优势、ECE/NLL、robustness 为主要收益；
  必须不破坏 task 和 S2。
```

最终要证明的是：

$$
\boxed{
\text{PureKAN 不仅能在 CE-only 下超过 MLP，而且 functional update 能带来 MLP/AdamW 不具备的几何优势。}
}
