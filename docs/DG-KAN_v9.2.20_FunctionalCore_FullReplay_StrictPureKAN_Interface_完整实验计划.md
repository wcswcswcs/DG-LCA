# DG-KAN v9.2.20 Functional-Core Full Replay 与 Strict PureKAN Interface Recovery 完整实验计划

> 本计划基于 v9.2.19 `FunctionalCore Rescue StrictPureKAN Interface` 的真实结果制定。  
> 本轮的核心立场是：**functional update 仍然是 DG-KAN 的核心，不能轻易放弃；但 v9.2.19 已经说明，不能再把 one-step curvature signal 当成 full functional success，也不能继续用旧的 output target / SNR gate / step fraction 小修。**
>
> v9.2.20 的任务是完成 v9.2.19 尚未完成的关键验证：  
>
> $$
> \boxed{
> \text{把 v8 strong-control replay 从 one-step 提升到 50/240-step 和 20-epoch full replay。}
> }
> $$
>
> 如果 v8 functional 在强控制下仍然成立，则 functional update 继续保留为主线，并把 v8 的有效机制迁移成 strict FC-PureKAN 内部的 functional interface。  
> 如果 v8 functional 在强控制下被 AdamWParallel / scalar LR controls 解释，则当前 functional 证据必须降级，functional 作为长期核心目标保留，但本轮 strict route 应回到 primitive / basis / AdamW-only full-pass。
>
> 本计划继续遵守：**no teacher、no self-teacher、no distillation、no loss modification、no label smoothing、no focal / margin / calibration loss、no sampler / class weight、no CPU offload、no fake / proxy rows、KAN path 不使用 PyTorch loss.backward graph、PureKANConv / PureKANFormer 继续 deferred。**

---

## 0. 当前结果的独立判断

### 0.1 v9.2.19 没有达到目标

v9.2.19 的 terminal route 是：

```text
route = R7-FunctionalPausedButNotAbandoned
base_candidate = LQ-t2-h256
success_v9219_functional_core_retained = false
success_v9219_strict_purekan_functional = false
success_v9219_external_ready = false
```

这不是 functional update 的失败终局，而是一个正确的暂停点。v9.2.19 相比 v9.2.18 已经有实质推进：它不再只是 source recap，而是新增了真实 v8 KW4 one-step strong-control replay，覆盖了 AdamWParallel、LRScale、NoOp、Random、Shuffled、Inverted controls。  
但它只完成 one-step replay，不是原计划要求的 20-epoch P1 strong-control replay，因此不能写：

```text
v8_strong_control_replay_pass = true
functional_core_retained = true
strict_purekan_functional = true
external_ready = true
```

### 0.2 v9.2.19 的正信号

v9.2.19 的 one-step replay 共测了：

```text
measured rows = 450
```

其中 functional 的正信号主要体现在 curvature。关键数值是：

```text
best functional curvature delta = -0.252045
best AdamWParallel curvature delta = -0.0399457
best LR curvature delta = -0.0399457
```

这说明 v8-style functional 在 one-step 几何指标上仍有非常明显的机制信号。这个信号不能忽视。它支持：

$$
\boxed{
\text{functional update 仍可能具有非 LR 的几何修正机制。}
}
$$

### 0.3 v9.2.19 的负信号

但是，CE tail 和 margin 的 aggregate 仍由 AdamWParallel / LR controls 领先：

```text
best functional CEp99 delta = -0.0565658
best AdamWParallel CEp99 delta = -0.0619887
best LR CEp99 delta = -0.0619887

best functional margin delta = +0.0467026
best AdamWParallel margin delta = +0.0515927
best LR margin delta = +0.0515927
```

这说明 one-step 下，functional 的几何信号强，但 task-tail / margin-tail 还没有超过强 optimizer controls。  
因此当前最准确的判断是：

$$
\boxed{
\text{functional 有 curvature 机制信号，但还没有证明它能超过 AdamWParallel / LR controls。}
}
$$

### 0.4 当前真正的问题

当前不是系统实现问题。LQ / A7c 路线已经证明 P4-qualified actuator 可以存在。  
当前也不是 output target 完全无效。v9.2.13-v9.2.14 已经证明 direct target 和 actuator controllability 有信号。  
当前也不是 v8 成功完全无效。v8 one-step replay 仍显示 curvature signal。

真正的问题是：

$$
\boxed{
\text{functional 的几何优势是否能从 one-step 机制信号转化为多步、全训练、强控制下的 task-safe advantage。}
}
$$

这正是 v9.2.20 必须回答的问题。

---

## 1. v9.2.20 总体目标

v9.2.20 的总体目标是：

$$
\boxed{
\text{用 full strong-control replay 决定 functional update 是否继续作为 DG-KAN 主线。}
}
$$

这个目标分成四个层级。

### 1.1 Functional retained success

如果 v8-style functional 在 50/240-step short-run 和 20-epoch full replay 中仍然满足：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005,
$$

并且至少一个核心机制指标超过 AdamWParallel 和 best LR control：

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamWParallel}},
$$

$$
CEp99_{\text{functional}}<CEp99_{\text{best LR}},
$$

或：

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamWParallel}},
$$

$$
MarginP10_{\text{functional}}>MarginP10_{\text{best LR}},
$$

或：

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamWParallel}},
$$

则 functional update 作为核心主线被保留。

### 1.2 Strict PureKAN functional interface success

如果 v8 functional survives strong controls，下一步必须把它迁移到 strict FC-PureKAN 内部，而不是继续依赖 transitional linear-SiLU stack。  
Strict PureKAN candidate 必须满足：

```text
external_residual_used = 0
ordinary_mlp_path_used = 0
edge_owned_param_fraction = 1
manual_forward = 1
manual_backward = 1
manual_update = 1
```

并且通过：

$$
forward_{q90}\leq1.25,
$$

$$
backward_{q90}\leq1.50,
$$

$$
step_{q90}\leq1.50,
$$

$$
memory\leq1.05.
$$

以及：

$$
near\_pass\_rate\geq0.80,
$$

$$
\Delta Acc_{\text{macro}}\geq-0.01.
$$

最后还必须在 paired replay 中满足：

$$
RealFunctional > AdamWParallel,
$$

$$
RealFunctional > best\ LR\ control.
$$

### 1.3 Base model full-pass success

Functional 是核心，但不能掩盖 base 没有 full-pass 的事实。  
因此 v9.2.20 还需要并行推进 AdamW-only base repair：

$$
\Delta Acc_{\text{macro,AdamW}}\geq0.
$$

如果 base full-pass 先达成，而 functional 仍不成立，则只能声明：

```text
Strict PureKAN base improved; functional advantage remains unproven.
```

### 1.4 External-ready success

只有当 strict PureKAN functional route 同时满足：

```text
P4 pass
P5 near-pass or full-pass
functional beats AdamWParallel / LR controls
short-run pass
full 10-seed pass
robustness pass
strong baseline challenge pass
```

才允许 external-ready。PureKANConv / PureKANFormer 继续 deferred。

---

## 2. 核心假设

### H1：v8 functional 的 one-step curvature signal 可以转化为 multi-step 几何优势

v9.2.19 已显示：

$$
\Delta Curvature_{\text{functional}}=-0.252045,
$$

而 AdamWParallel / LR controls 只有约：

$$
\Delta Curvature_{\text{control}}=-0.0399457.
$$

H1 假设：这个 one-step curvature signal 不是偶然，可以在 50/240-step 和 20-epoch full replay 中延续。

H1 成立标准：

在 50/240-step replay 中：

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamWParallel}},
$$

并且：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005.
$$

若 full replay 中 curvature 仍强但 task 无改善，则 functional 被归类为：

```text
GeometryOnlyFunctional
```

而不能直接写作 task advantage。

### H2：v8 functional 可能被 LR / extra-AdamW 解释

v9.2.19 one-step 中 CEp99 / margin aggregate 被 AdamWParallel / LR controls 领先。H2 假设：v8 functional 的部分成功可能来自 under-step 或 optimizer scale，而不是独立 functional direction。

H2 成立标准：

如果 full replay 中：

$$
MetricGain_{\text{functional}}
\leq
MetricGain_{\text{best LR control}}+\epsilon,
$$

则 v8 functional 被判定为 LR-equivalent。

默认阈值：

```text
epsilon_CEp99 = 0.02 relative
epsilon_margin = 0.005 absolute
epsilon_ECE = 0.005 absolute
epsilon_curvature = 0.02 relative
```

### H3：v8 functional 的真实机制来自 role-wise branch interface

v8 architecture 有 stack/head role separation、event guard、branch-like activity 与 effective derivative scale。H3 假设：functional 成功依赖这种 role-wise interface，而不是简单 output correction。

H3 成立标准：

如果 functional beats LR / AdamWParallel controls，并且满足：

```text
cos(functional, AdamW) < 0.95 for non-trivial event fraction
branch_ratio 与 curvature / CE-tail gain 正相关
effective_derivative_scale 在安全区间内
role update norm 对 stack/head 有清晰分工
```

则认定 v8 mechanism independent。

### H4：v9 LQ/A7c 失败来自 strict primitive 缺少 v8-like functional interface

如果 H1/H3 成立，但 v9 LQ/A7c 仍不成立，则说明 current strict primitive 缺少合适 interface，而不是 functional 概念错。

H4 成立标准：

```text
v8 survives strong controls = 1
v9 LQ/A7c functional control pass = 0
```

route 记为：

```text
FunctionalConceptValid_PrimitiveInterfaceMismatch
```

### H5：Strict PureKAN functional interface 必须是 edge-owned

任何迁移方案必须保持 strict PureKAN：

$$
y_c
=
\sum_j
\phi_{jc}(h_j),
$$

其中：

$$
\phi_{jc}(h)
=
\sum_k c_{jc,k}B_k(h).
$$

允许 task channel 和 functional channel 分开：

$$
\phi_{jc}(h)
=
\phi^{task}_{jc}(h)
+
\phi^{func}_{jc}(h),
$$

但二者都必须是 edge function 内部 channel，不能是 external residual。

### H6：如果 v8 full replay 失败，functional 不永久放弃，但当前路线必须重置

如果 v8 full replay 也无法超过 LR / AdamWParallel controls，则不能继续用 v8 旧成功背书 functional 主线。  
此时 functional 仍作为长期核心目标保留，但当前实现路线应 reset：

```text
target/gate patch 停止
strict interface 重新设计
basis / primitive factory 重新打开
AdamW-only full-pass repair 并行推进
```

---

## 3. 实验阶段

## P0：历史路线与最新 v9.2.19 边界复现

### 目标

确认 v9.2.19 的结论稳定：one-step strong-control replay 有 functional curvature signal，但 full replay 未完成。

### 必须记录

```text
route
base_candidate
v8_one_step_row_count
v8_one_step_best_functional_CEp99_delta
v8_one_step_best_functional_margin_delta
v8_one_step_best_functional_curvature_delta
v8_one_step_best_parallel_CEp99_delta
v8_one_step_best_parallel_margin_delta
v8_one_step_best_parallel_curvature_delta
v8_one_step_best_lr_CEp99_delta
v8_one_step_best_lr_margin_delta
v8_one_step_best_lr_curvature_delta
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R7-FunctionalPausedButNotAbandoned
v8_one_step_row_count = 450
functional curvature signal exists
full epoch v8 strong-control replay not done
fake_proxy_count = 0
```

### 可视化

```text
p0_v9219_boundary_dashboard.svg
p0_one_step_functional_vs_controls.svg
p0_curvature_ce_margin_tradeoff.svg
```

---

## P1：v8 50/240-step strong-control replay

### 目标

把 v9.2.19 的 one-step strong-control replay 提升到 short-run。  
这是本轮最重要的第一阶段。不能直接跳到 strict PureKAN interface。

### 候选

```text
V8-AdamWOnly
V8-FT7-RoleWiseFunctional
V8-Adaptive-FT-P
V8-NoOpMatchedOverhead
V8-RandomMatchedNorm
V8-ShuffledRoleMask
V8-InvertedRoleMask
V8-AdamWParallelSameNorm
V8-AdamWParallelTrustRatio-0.003
V8-AdamWParallelTrustRatio-0.01
V8-AdamWParallelTrustRatio-0.03
V8-LRScale-1.003
V8-LRScale-1.01
V8-LRScale-1.03
V8-LRScale-1.10
```

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
steps = 50,240
batch_size = 128
same initialization across branches = true
same batch sequence across branches = true
```

### 必须记录

```text
candidate
dataset
seed
steps
test_acc_proxy
holdout_loss
CEp99
margin_p10
wrong_confidence_p95
ECE_proxy
NLL_proxy
curvature
local_lipschitz
functional_event_count
event_coverage
branch_ratio
effective_derivative_scale
cos_functional_adamw
step_ratio_q90
memory_ratio
control_rank
beats_adamwparallel
beats_best_lr
```

### 判断标准

Short-run functional pass：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005.
$$

并且至少一个指标超过 AdamWParallel 和 best LR：

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamWParallel}},
$$

$$
CEp99_{\text{functional}}<CEp99_{\text{best LR}},
$$

或：

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamWParallel}},
$$

$$
MarginP10_{\text{functional}}>MarginP10_{\text{best LR}},
$$

或：

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamWParallel}}.
$$

System pass：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p1_short_run_functional_vs_controls.svg
p1_ce_margin_curvature_by_step.svg
p1_control_rank_by_dataset_step.svg
p1_branch_ratio_vs_gain.svg
p1_cos_with_adamw.svg
```

---

## P2：v8 20-epoch full strong-control replay

### 目标

若 P1 通过或出现强 curvature-only signal，执行 full replay。  
这是决定 functional core 是否重新 retained 的关键 gate。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
epochs = 20
batch_size = 128
controls = same as P1
```

### 必须记录

```text
candidate
dataset
seed
test_acc
val_acc
delta_vs_adamw
delta_vs_mlp_match
CEp99
margin_p10
wrong_confidence_p95
ECE
NLL
curvature
local_lipschitz
functional_event_count
event_coverage
branch_ratio
effective_derivative_scale
cos_functional_adamw
step_ratio_q90
memory_ratio
control_rank
beats_adamwparallel
beats_best_lr
```

### 判断标准

Full replay task-safety：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005.
$$

Functional retained pass：

Functional 必须超过 AdamWParallel 和 best LR control 至少一个机制指标：

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamWParallel}},
$$

$$
CEp99_{\text{functional}}<CEp99_{\text{best LR}},
$$

或：

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamWParallel}},
$$

$$
MarginP10_{\text{functional}}>MarginP10_{\text{best LR}},
$$

或：

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamWParallel}}.
$$

Strong retained pass：

至少一个任务维度也改善：

$$
\Delta Acc_{\text{macro,functional}}
-
\Delta Acc_{\text{macro,AdamW}}
\geq0.003,
$$

或：

$$
\Delta Acc_{\text{KMNIST,functional}}
-
\Delta Acc_{\text{KMNIST,AdamW}}
\geq0.005.
$$

### 可视化

```text
p2_full_replay_macro_delta.svg
p2_full_replay_kmnist_delta.svg
p2_functional_vs_controls_ce_tail.svg
p2_functional_vs_controls_curvature.svg
p2_seedwise_win_matrix.svg
```

---

## P3：v8 mechanism extraction

### 目标

如果 P1/P2 显示 v8 functional survives controls，必须提取机制，不能只保留结果。

### 必须记录

```text
role
role_update_norm
role_snr
role_curvature
role_event_frequency
branch_ratio
effective_derivative_scale
cos_functional_adamw
cos_functional_lrcontrol
cos_functional_random
functional_norm_vs_adamw
pre_holdout_loss
post_holdout_loss
CEp99_delta
margin_delta
curvature_delta
ECE_delta
NLL_delta
```

### 机制指标

Branch ratio：

$$
r_{\text{branch}}
=
\frac{\|\alpha F(h)\|}{\|h\|+\epsilon}.
$$

Effective derivative scale：

$$
s_{\text{eff}}
=
|\alpha|\cdot \phi'_{p95}.
$$

AdamW alignment：

$$
\cos_{\text{AdamW}}
=
\frac{
\langle \Delta\theta_{\text{functional}},\Delta\theta_{\text{AdamW}}\rangle
}{
\|\Delta\theta_{\text{functional}}\|\|\Delta\theta_{\text{AdamW}}\|+\epsilon
}.
$$

Non-LR mechanism pass：

```text
functional beats LR controls
cos_with_AdamW not always > 0.95
branch_ratio non-trivial
curvature/tail gain not reproduced by NoOp/Random/LR
```

### 可视化

```text
p3_rolewise_mechanism_heatmap.svg
p3_branch_ratio_vs_tail_gain.svg
p3_effective_derivative_scale_vs_curvature.svg
p3_cosine_distribution.svg
```

---

## P4：Strict FC-PureKAN functional interface design

### 目标

把 v8 的有效机制迁移到 strict FC-PureKAN 中。  
不允许复制 v8 transitional architecture；只能迁移 functional interface 机制。

### 候选 family A：Dual-role edge basis

输出层 edge function 分成 task channel 和 functional channel：

$$
\phi_{jc}(h)
=
\phi^{task}_{jc}(h)
+
\phi^{func}_{jc}(h).
$$

其中：

$$
\phi^{task}_{jc}(h)
=
c^{task}_{jc,0}B_0(h)
+
c^{task}_{jc,2}B_2(h),
$$

$$
\phi^{func}_{jc}(h)
=
c^{func}_{jc,a}B_a(h).
$$

候选：

```text
A1-T2-task + bounded-rational-functional
A2-T2-task + piecewise-linear-functional
A3-T2-task + centered-T2-functional
A4-T2-task + shared-RBF-functional
```

### 候选 family B：Derivative-scale controller

控制 functional channel 的 effective derivative scale：

$$
s_{\text{eff}}=|\alpha|\cdot B'_a(h)_{p95}.
$$

候选：

```text
B1-bounded-rational-derivative-controller
B2-piecewise-linear-derivative-controller
B3-normalized-T2-derivative-controller
```

### 候选 family C：Role-wise PureKAN metric

Functional 作为 role-wise metric，而不是 output target：

$$
\theta_{t+1}
=
\theta_t+
P_{\text{role}}(\theta,t)\Delta\theta_{\text{AdamW}}.
$$

但必须超过 scalar LR / AdamWParallel controls。

候选：

```text
C1-role-SNR metric
C2-role-curvature-damped metric
C3-tail-event role metric
C4-v8-FT7-style edge-role guard
```

### 候选 family D：Orthogonal non-AdamW correction

构造与 AdamW 不完全平行的 functional component：

$$
\Delta\theta_{\perp}
=
\Delta\theta_{\text{func}}
-
\frac{
\langle \Delta\theta_{\text{func}},\Delta\theta_{\text{AdamW}}\rangle
}{
\|\Delta\theta_{\text{AdamW}}\|^2+\epsilon
}
\Delta\theta_{\text{AdamW}}.
$$

### 必须记录

```text
candidate
full_edge_equivalence_pass
external_residual_used
ordinary_mlp_path_used
edge_owned_param_fraction
manual_forward
manual_backward
manual_update
GradRelErrMax
GradCosMin
synthetic_pairwise_R2
P4_forward_q90
P4_backward_q90
P4_step_q90
P4_memory
P5_nearpass_rate
macro_delta
functional_actuatability_R2
non_adamw_output_displacement
cos_with_adamw
paired_replay_vs_adamwparallel
paired_replay_vs_lrcontrol
```

### 判断标准

Contract pass：

```text
external_residual_used = 0
ordinary_mlp_path_used = 0
edge_owned_param_fraction = 1
manual_forward/backward/update = 1
```

P4 pass：

$$
forward_{q90}\leq1.25,
$$

$$
backward_{q90}\leq1.50,
$$

$$
step_{q90}\leq1.50,
$$

$$
memory\leq1.05.
$$

P5 near-pass：

$$
near\_pass\_rate\geq0.80,
$$

$$
\Delta Acc_{\text{macro}}\geq-0.01.
$$

Functional interface pass：

$$
RealFunctional > AdamWParallel,
$$

$$
RealFunctional > best\ LR\ control.
$$

### 可视化

```text
p4_interface_contract_heatmap.svg
p4_task_system_functional_pareto.svg
p4_non_adamw_displacement_rank.svg
p4_paired_replay_vs_controls.svg
```

---

## P5：Basis / primitive factory with functional actuatability

### 目标

如果 P4 没有 survivor，重新打开 basis / primitive factory。  
新 primitive 不只看 P4/P5，还必须看 functional actuatability。

### Candidate families

```text
B0-LQ-T2-current
B1-centered-T2
B2-normalized-T2
B3-Legendre2-only
B4-normalized-Legendre2
B5-bounded-rational-base
B6-piecewise-linear2-base
B7-shared-RBF4-base
B8-mixed-T2-rational
B9-mixed-T2-piecewise
B10-dual-role-T2-rational-functional
B11-dual-role-T2-piecewise-functional
B12-dual-role-T2-RBF-functional
```

### 必须记录

```text
basis_family
basis_formula
derivative_formula
requires_recursion
requires_dense_materialization
conditioning_pass
condition_number
dominant_basis_fraction
dead_basis_fraction
synthetic_pairwise_R2
local_bump_R2
GradRelErrMax
GradCosMin
P4_forward_q90
P4_backward_q90
P4_step_q90
P4_memory
P5_nearpass_rate
macro_delta
KMNIST_delta
functional_actuatability_R2
non_adamw_output_displacement
paired_replay_vs_adamwparallel
paired_replay_vs_lrcontrol
```

### 晋级标准

```text
conditioning pass
grad pass
P4 pass
P5 near-pass
functional actuatability pass
```

Functional actuatability pass：

$$
R^2_{\text{functional target fit}}\geq0.20,
$$

$$
r_z\geq0.05,
$$

and:

$$
RealFunctional > AdamWParallel,
$$

$$
RealFunctional > best\ LR\ control.
$$

### 可视化

```text
p5_basis_factory_task_system_functional_pareto.svg
p5_basis_conditioning_heatmap.svg
p5_functional_actuatability_matrix.svg
p5_kmnist_delta_vs_basis.svg
```

---

## P6：AdamW-only full-pass repair

### 目标

Functional 是核心，但 base 不够强会污染 functional 判断。  
并行推进 AdamW-only full-pass，禁止 teacher/loss/sampler。

### Allowed repairs

```text
orthogonal lift init
fan-in output scale
centered / normalized basis
basis-balanced init
hidden bracket h224/h256/h288
bounded rational base
piecewise local base
dual-role basis with functional channel frozen during AdamW-only phase
```

### 必须记录

```text
candidate
dataset
seed
test_acc
delta_vs_mlp
near_pass
full_pass
CEp99
margin_p10
ECE
NLL
basis_entropy
lift_condition_number
effective_rank
P4_step_q90
memory_ratio
```

### 判断标准

Full-pass：

$$
\Delta Acc_{\text{macro}}\geq0.
$$

Robust near-pass：

$$
near\_pass\_rate\geq0.80,
$$

$$
\Delta Acc_{\text{macro}}\geq-0.01.
$$

### 可视化

```text
p6_adamw_fullpass_gap.svg
p6_kmnist_miss_rows.svg
p6_ce_tail_margin.svg
p6_basis_entropy_vs_gap.svg
```

---

## P7：Full validation for functional survivors

### 目标

只有 P1/P2/P4/P5 出现 paired replay survivor 后，才打开 full validation。

### 设置

```text
short-run steps = 50,240
full-run epochs = 20
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
baseline = MLP-match
diagnostic baseline = QuadraticFeatureMLP
controls = AdamWOnly, AdamWParallel, best LR control, NoOp, Random, Shuffled
```

### 必须记录

```text
candidate
dataset
seed
val_acc
test_acc
delta_vs_mlp
delta_vs_adamw
delta_vs_adamwparallel
delta_vs_best_lr
delta_vs_quadratic_feature_mlp
CEp99
margin_p10
wrong_confidence_p95
ECE
NLL
curvature_ratio
local_lipschitz_ratio
event_count
event_coverage
bad_event_rate
step_ratio_q90
memory_ratio
```

### 判断标准

Task safety：

$$
\Delta Acc_{\text{functional-vs-AdamW}}\geq-0.005.
$$

Control superiority：

$$
MetricGain_{\text{functional}}>MetricGain_{\text{AdamWParallel}},
$$

$$
MetricGain_{\text{functional}}>MetricGain_{\text{best LR control}}.
$$

Macro improvement：

$$
\Delta Acc_{\text{macro,functional}}
-
\Delta Acc_{\text{macro,AdamW}}
\geq0.003.
$$

KMNIST repair：

$$
\Delta Acc_{\text{KMNIST,functional}}
-
\Delta Acc_{\text{KMNIST,AdamW}}
\geq0.005.
$$

Mechanism pass：

至少一个：

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamW}},
$$

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamW}},
$$

$$
ECE_{\text{functional}}\leq ECE_{\text{AdamW}},
$$

$$
NLL_{\text{functional}}\leq NLL_{\text{AdamW}},
$$

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

### 可视化

```text
p7_macro_delta_vs_controls.svg
p7_kmnist_repair_matrix.svg
p7_seedwise_win_matrix.svg
p7_task_geometry_pareto.svg
p7_ce_tail_margin_panel.svg
p7_ece_nll_panel.svg
```

---

## P8：Robustness / external-ready gate

### 目标

确认 functional advantage 不是 clean setting 偶然有效。

### 设置

```text
label_noise = 0.05,0.10,0.20
input_noise = 0.05,0.10
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
```

### 判断标准

Robustness pass：

$$
AccDrop_{\text{functional}}\leq AccDrop_{\text{AdamW}}.
$$

Noise-tail pass：

$$
CEp99_{\text{functional}}\leq CEp99_{\text{AdamW}}.
$$

External-ready：

```text
functional task-safe = 1
functional mechanism pass = 1
functional control pass = 1
functional system pass = 1
strong baseline challenge pass = 1
robustness pass = 1
```

### 可视化

```text
p8_noise_robustness_curve.svg
p8_strong_baseline_pareto.svg
p8_external_ready_scorecard.svg
```

---

## 4. Required artifacts

```text
run_manifest.json
contract_audit_v9220.csv
p0_v9219_boundary_reproduction.csv
p1_v8_short_run_strong_control_replay.csv
p2_v8_full_epoch_strong_control_replay.csv
p3_v8_mechanism_extraction.csv
p4_strict_purekan_functional_interface_design.csv
p5_basis_factory_functional_actuatability.csv
p6_adamw_only_fullpass_repair.csv
p7_functional_survivor_full_validation.csv
p8_robustness_external_ready.csv
paired_replay_branch_trace_v9220.csv
role_mechanism_trace_v9220.csv
purekan_interface_trace_v9220.csv
basis_functional_actuatability_trace_v9220.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9219_boundary_unstable
F3_v8_short_run_control_equivalent
F4_v8_full_replay_control_equivalent
F5_v8_lr_equivalent
F6_v8_mechanism_unattributed
F7_purekan_interface_contract_fail
F8_purekan_interface_p4_fail
F9_purekan_interface_p5_nearpass_fail
F10_functional_control_equivalent
F11_basis_factory_no_candidate
F12_adamw_fullpass_fail
F13_short_full_task_drop
F14_full_run_no_macro_kmnist_gain
F15_external_not_ready
F16_fake_or_proxy_violation
F17_artifact_missing
```

---

## 5. Route decision

### Route cases

```text
R1-v8FunctionalShortRunPass:
  v8 functional beats AdamWParallel / LR controls in 50/240-step replay.

R2-v8FunctionalFullReplayPass:
  v8 functional beats AdamWParallel / LR controls in 20-epoch replay.

R3-v8FunctionalIsLREquivalent:
  v8 functional explained by scalar LR / extra AdamW controls.

R4-FunctionalConceptValid_PrimitiveInterfaceMismatch:
  v8 survives controls, v9 current family fails; strict PureKAN interface redesign required.

R5-StrictPureKANInterfaceFunctionalPass:
  new strict PureKAN interface beats AdamWParallel / LR controls.

R6-NewPrimitiveFunctionalActuatabilityPass:
  basis/primitive factory finds P4/P5/function-actuatable candidate.

R7-AdamWFullPassNoFunctional:
  base reaches full-pass without functional; functional remains core but current evidence not restored.

R8-FunctionalPausedButNotAbandoned:
  evidence still insufficient; continue targeted replay/interface work, not old target patching.

R9-ExternalReady:
  strict FC-PureKAN functional route passes full task / robustness / strong baseline.
```

### route_decision.json 必须记录

```text
route
v8_short_run_pass
v8_full_replay_pass
v8_lr_equivalent
v8_mechanism_identified
v9_interface_pass
basis_factory_functional_pass
adamw_fullpass
functional_short_full_pass
external_ready
functional_core_retained
primary_blocker
next_required_implementation
success_v9220_functional_core_retained
success_v9220_strict_purekan_functional
success_v9220_external_ready
```

---

## 6. 第一轮执行顺序

```text
Step 1:
  P0 复现 v9.2.19 boundary。

Step 2:
  P1 先跑 v8 50/240-step strong-control replay。
  这是 one-step 到 full replay 之间的必要桥梁。

Step 3:
  如果 P1 有 survivor，进入 P2 20-epoch full strong-control replay。
  如果 P1 完全 control-equivalent，先不要做 P4/P5。

Step 4:
  P3 提取 v8 mechanism。
  必须知道 role-wise / branch-like / derivative-scale 机制是什么。

Step 5:
  P4 strict FC-PureKAN interface design。
  只迁移机制，不复制 transitional stack。

Step 6:
  P5 basis / primitive factory with functional actuatability。
  如果 P4 没 survivor，重新设计 primitive。

Step 7:
  P6 并行 AdamW-only full-pass repair。

Step 8:
  P7/P8 只对真正 paired replay survivor 打开。
```

---

## 7. 停止条件

### Functional retained success

```text
v8 short-run pass
v8 full replay pass
v8 mechanism identified
```

### Strict PureKAN functional success

```text
Functional retained success
+
new strict PureKAN interface passes contract/P4/P5
+
paired replay beats AdamWParallel / LR controls
+
short/full run task-safe mechanism gain
```

### Failure stop

```text
1. v9.2.19 boundary cannot be reproduced；
2. v8 short-run is control-equivalent；
3. v8 full replay is LR-equivalent；
4. v8 mechanism cannot be identified；
5. strict PureKAN interface violates contract；
6. no basis/primitive candidate passes P4/P5/function actuatability；
7. all functional candidates remain control-equivalent；
8. full run gives no macro/KMNIST/geometry/tail gain；
9. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 8. 最终解释规则

### Case A：v8 short/full strong-control replay passes

可以声明：

```text
Functional update remains a valid core mechanism; current v9 failure is primitive/interface mismatch.
```

### Case B：v8 full replay is LR-equivalent

必须声明：

```text
Previous functional evidence is downgraded under stronger controls.
```

但这不代表永远放弃 functional；它意味着当前实现不足，需要从 primitive/interface 重建。

### Case C：strict PureKAN interface passes

可以声明：

```text
Functional update has been recovered in strict FC-PureKAN under strong controls.
```

### Case D：new primitive AdamW full-pass, functional fails

必须声明：

```text
Base model improved; functional advantage remains unproven.
```

### Case E：all functional candidates control-equivalent

必须声明：

```text
Functional remains a long-term core objective, but current implementation route is exhausted; next cycle must redesign primitive/basis/interface.
```

---

## 9. 最终建议

v9.2.20 的一句话策略是：

$$
\boxed{
\text{functional 不放弃，但必须从 one-step curvature signal 走到 full strong-control replay；再把 v8 的有效 interface 迁移到 strict PureKAN。}
}
$$

当前最重要的不是继续 patch output target，也不是提前 Conv / Former，而是：

```text
1. v8 functional 的 curvature signal 能否在 50/240-step 与 20-epoch replay 中超过 AdamWParallel / LR？
2. v8 的 role-wise / branch-like / derivative-scale 机制到底是什么？
3. strict FC-PureKAN 如何合法表达这种 functional interface？
4. 新 primitive 是否同时满足 P4/P5/function actuatability？
5. functional 能否最终修复 KMNIST / CE-tail / margin-tail，并超过 strong controls？
```
