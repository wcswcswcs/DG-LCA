# DG-KAN v9.2.19 Functional-Core Rescue：v8 机制复验、Strict PureKAN Functional Interface 与 Non-LR Functional Advantage 完整实验计划

> 本计划基于 v8.3-v8.7 的 functional update 成功经验、v9.2.9-v9.2.18 的 strict FC-PureKAN functional 负结果，以及当前“functional update 是核心、不能轻易放弃”的路线要求制定。  
> 本计划的核心立场是：
>
> $$
> \boxed{
> \text{functional update 不应被放弃；但不能继续沿旧的 hand-designed output target / gate patch 路线小修。}
> }
> $$
>
> v9.2.19 的任务不是再写一个新 target，也不是再调 SNR threshold、event coverage、functional step fraction、LR 或 loss。  
> 本轮必须把问题提升到机制层：**v8 成功的 functional interface 到底是什么？它是否能经受 AdamWParallel / scalar LR 强控制？如果能，如何把这种机制迁移成 strict FC-PureKAN 内部的合法 functional interface？如果不能，如何重建 primitive，让它天然具备 non-LR functional actuatability？**
>
> 本计划继续遵守：**no teacher、no self-teacher、no distillation、no loss modification、no label smoothing、no focal/margin/calibration loss、no sampler/class weight、no CPU offload、no fake/proxy rows、KAN path 不使用 PyTorch loss.backward graph、PureKANConv / PureKANFormer 继续 deferred。**

---

## 0. 当前判断：functional update 不是失败，而是旧实现路径失败

### 0.1 不能轻易放弃 functional update

Functional update 是 DG-KAN 的核心，不应因为当前 v9 strict FC-PureKAN 上的连续负结果就被放弃。  
过去 v8.3-v8.7 的结果已经证明：在 transitional DG-KAN architecture 上，functional update 曾经带来 task-safe、geometry-positive、external-fair 的强信号。v8.3 完成了 system-gated functional re-entry minimum success；v8.7 在 selected `KW4 hidden28` route 下完成 FMNIST / KMNIST external fair formal protocol，并在 P9 functional causality 中显示 DG functional curvature 明显优于 RandomFunc。

但是，这些成功不是 strict FC-PureKAN 成功。它们属于 transitional / branch-like / role-wise architecture 上的 functional success。当前 v9.2 strict FC-PureKAN 的负结果说明：**把 functional update 简单移植成 output-space target、actuator correction 或 signal metric，并不能自动复现 v8 的优势。**

### 0.2 当前 v9.x 失败的准确含义

从 v9.2.9 到 v9.2.18，我们已经排除了很多表层解释：

```text
low-cost SNR instrumentation:
  已经可用，但 SNR gate 本身不能带来 causal advantage。

one-step safety:
  D9 可以做到 one-step safe，但 multi-step paired replay control-equivalent。

output-space target:
  direct oracle target 有用，但参数实现后打不过 controls。

actuator controllability:
  A7c / A4-family 已经能做到 P4-qualified + strong controllability。

control-resistant causality:
  RealFunctional 在 paired replay 中打不过 NoOp / Random / ShuffledTarget / AdamWParallel。

signal-aligned metric:
  raw survivor 扣除 AdamWParallel / scalar LR / extra AdamW equivalence 后 effective survivor = 0。
```

因此当前不能再说“functional 只是系统还没做好”或“actuator 不可控”。真正的问题是：

$$
\boxed{
\text{当前 strict LQ/A4/A7c family 中，functional update 没有产生独立于 AdamW/LR controls 的因果方向。}
}
$$

### 0.3 为什么 v8 成功而 v9 失败

v8 的 functional update 可能依赖一种 v9 还没有复现出来的 **functional interface**。这个 interface 不是 teacher，不是 loss，不是 sampler，而是 architecture 内部允许功能性修正安全进入的结构。v8 的 packed / cached / linear-SiLU stack + KAN-style head 有更明显的 role separation：

```text
stack role
head role
functional event branch
guarded update branch
effective derivative scale
branch activity ratio
```

v9 的 LQ/A7c 则更干净、更 strict，但可能缺少这种“可安全扰动而不等价于 AdamWParallel”的接口。  
所以 v9.2.19 不应暂停 functional，而应把 functional 主线升级为：

$$
\boxed{
\text{functional update = strict PureKAN 内部的 non-LR, role-aware, signal-causal interface。}
}
$$

---

## 1. v9.2.19 总体目标

v9.2.19 的总体目标是：

$$
\boxed{
\text{恢复 functional update 为主线，但必须证明它不是 LR/extra-AdamW disguise。}
}
$$

具体来说，本轮要回答七个问题。

### Q1：v8 functional success 是否能经受 v9-style strong controls？

v8 成功不能直接背书 v9，因为 v8 source artifact 缺少 AdamWParallel 与 scalar LR controls。v9.2.19 必须真正补跑：

```text
AdamWParallelSameNorm
AdamWParallelTrustRatio
LRScale 1.003 / 1.01 / 1.03 / 1.10
NoOpMatchedOverhead
RandomMatchedNorm
ShuffledRoleMask
InvertedRoleMask
```

如果 v8 FT7 / Adaptive functional 仍然超过这些 controls，则 functional concept 继续保留为核心主线。  
如果 v8 也被 LR/AdamWParallel 解释，则必须承认过去的 functional evidence 需要降级，并把 functional 改为 diagnostic，直到新的 primitive 产生 non-LR actuatability。

### Q2：v8 成功机制具体是什么？

不能只看 final acc。必须解释 v8 functional 的机制来源：

```text
role-wise update 是否真的必要？
stack/head 分工是否产生非 AdamW-parallel correction？
branch activity ratio 是否和收益相关？
effective derivative scale 是否提供安全扰动通道？
functional event 是否集中在 hard-mode / tail events？
curvature 改善是否伴随 ECE/NLL/CEp99/margin 改善？
```

### Q3：v9 strict PureKAN 是否缺少 v8-like interface？

如果 v8 在强 controls 下仍然有效，而 v9 LQ/A7c 无效，那么问题不是 functional 概念错，而是当前 strict FC-PureKAN primitive 缺少 functional interface。  
要证明这一点，必须建立 **functional interface metrics**：

```text
non-AdamW displacement rank
role-wise controllability
safe orthogonal output movement
branch activity ratio
effective derivative scale
paired replay control margin
```

### Q4：如何在 strict FC-PureKAN 内重建 v8-like functional interface？

v9.2.19 不允许引入普通 residual、MLP hidden path、Conv、Former 或外部 preprocessor。  
但允许在 edge-function 系统内部构造合法的 **edge-owned functional interface**：

```text
dual-role edge basis
gated edge-owned actuator channel
task channel + functional channel
basis-normalized derivative controller
bounded rational branch channel
piecewise local correction channel
role-wise metric controller
```

它们必须仍可写为：

$$
y_j=\sum_i\phi_{ij}(x_i),
$$

或两层 FullEdge composition：

$$
h_j=\sum_i\phi^0_{ij}(x_i),
$$

$$
y_c=\sum_j\phi^1_{jc}(h_j).
$$

### Q5：functional update 应如何避免退化为 LR / AdamWParallel？

所有 functional candidate 必须和 scalar LR / extra AdamW controls 比：

$$
MetricGain_{\text{functional}}
>
MetricGain_{\text{best LR control}}
+
\epsilon.
$$

如果不能超过，它不是 functional advantage。

### Q6：functional 能否优先修复当前 hard mode？

v9 strict base 的主要缺口不是 MNIST easy mode，而是 KMNIST / CE-tail / margin-tail / calibration / curvature。  
Functional success 必须优先体现在这些机制上：

$$
\Delta Acc_{\text{KMNIST,functional}}
-
\Delta Acc_{\text{KMNIST,AdamW}}
\geq0.005,
$$

或：

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamW}},
$$

或：

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamW}},
$$

或：

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

### Q7：什么时候才允许 PureKANConv / PureKANFormer？

本轮仍不允许打开 Conv / Former。只有 FC-PureKAN 达到：

```text
P4 pass
P5 robust near-pass or full-pass
functional beats AdamWParallel / LR controls
short-run and full-run task-safe mechanism gain
external fair ready
```

才允许进入 extension。

---

## 2. 核心假设

### H1：functional update 的真实价值在 v8 中存在，但依赖 role-wise branch interface

H1 认为 v8 成功不是单纯 LR effect，而是由 role-wise branch interface 支撑：

```text
stack/head role separation
functional event guard
branch activity ratio
effective derivative scale
task-safe projection / holdout guard
```

H1 成立标准：

在 v8 strong-control replay 中，FT7 / Adaptive functional 满足：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005,
$$

并且至少一个机制指标同时超过 AdamWParallel 与 best LR control：

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

### H2：如果 v8 成功也被 LR controls 解释，则当前 functional 证据必须降级

H2 是一个严肃停止条件。  
如果 v8 FT7 / Adaptive functional 无法超过 best LR / AdamWParallel control，则过去的 functional success 不能再作为核心贡献直接引用。

H2 成立标准：

$$
MetricGain_{\text{functional}}
\leq
MetricGain_{\text{best LR control}}+\epsilon.
$$

其中：

```text
epsilon_CEp99 = 0.02 relative
epsilon_margin = 0.005 absolute
epsilon_ECE = 0.005 absolute
epsilon_curvature = 0.02 relative
```

### H3：v9 LQ/A7c 失败来自 functional interface mismatch，而不是 PureKAN 不可能 functional

H3 认为：LQ/A7c 是好的 task learner / controllable actuator，但不是好的 functional interface。  
它能移动 output，却不能产生 non-LR, control-resistant causality。

H3 成立标准：

如果 v8 survives controls，但 v9 LQ/A7c 不通过，则记录：

```text
FunctionalConceptValid_PrimitiveInterfaceMismatch = 1
```

并进入 strict PureKAN functional interface redesign，而不是暂停 functional。

### H4：strict PureKAN functional interface 必须内生在 edge basis 中

H4 要求任何 v8 机制迁移都不能依赖 external residual / MLP stack。  
合法 interface 必须满足：

```text
edge_owned_params = all trainable params
ordinary_mlp_path_used = 0
external_residual_used = 0
manual_forward/backward/update = 1
P4 pass
P5 near-pass
```

### H5：新的 functional candidate 必须先在 paired replay 中超过 controls，再进入 full training

不能再从 direct oracle、target fit、one-step safety 直接进入 full run。  
Paired replay 是硬 gate：

$$
RealFunctional > AdamWParallel,
$$

$$
RealFunctional > best\ LR\ control,
$$

$$
RealFunctional > RandomMatchedNorm,
$$

$$
RealFunctional > NoOpMatchedOverhead.
$$

### H6：functional 是核心，但也必须接受 falsifiable gate

Functional 是核心，不等于不可证伪。  
本轮如果 v8 strong-control replay 通过，则 functional 主线继续。  
如果 v8 也被 LR controls 解释，则下一步仍保留 functional 作为长期目标，但当前实现必须重置为 primitive/basis 问题，而不是继续改 gate。

---

## 3. 设计总览：三条并行主线

v9.2.19 不再只做一个 runner。它分三条主线，互相约束。

### 主线 A：v8 strong-control replay

目标：

```text
确认 v8 functional success 是否独立于 AdamWParallel / LR controls。
```

如果 A 成功，functional concept 保留为核心。  
如果 A 失败，functional evidence 降级。

### 主线 B：v8 mechanism extraction

目标：

```text
从 v8 成功中抽取可迁移机制：
role separation、branch activity、derivative scale、guard、event、curvature/tail response。
```

不能只迁移 architecture，要迁移机制。

### 主线 C：strict FC-PureKAN functional interface redesign

目标：

```text
在 strict PureKAN 内部实现 v8-like functional interface，
并用 P4/P5 + paired replay controls 检验。
```

如果 C 成功，functional update 重新成为 strict FC-PureKAN 主线。  
如果 C 失败，但 A 成功，说明 primitive mismatch，还要继续 basis/interface redesign。  
如果 A/C 都失败，则当前 functional line 进入 reset，但不是永久放弃。

---

## 4. 阶段 P0：统一历史审计与合同重标注

### 目标

明确 v8/v9 的成功类型，避免把 transitional success 当作 strict PureKAN success，也避免把 v9 strict failure 错写成 functional 永久失败。

### 必须记录

```text
version
route
candidate
architecture_family
strict_full_edge_purekan
transitional_route
functional_type
teacher_used
loss_modified
controls_used
adamwparallel_control_used
lr_control_used
p4_pass
p5_nearpass
p5_fullpass
functional_pass
external_fair_pass
broad_strong_claimed
no_fake_proxy
```

### 判定标准

历史路线必须分为：

```text
Strict FC-PureKAN success
Transitional functional success
Diagnostic assisted success
System-only success
Functional paused / not established
```

### 可视化

```text
p0_version_route_lattice.svg
p0_success_definition_matrix.svg
p0_v8_v9_contract_gap_table.md
```

---

## 5. 阶段 P1：v8 strong-control replay

### 目标

真正补跑 v8 FT7 / Adaptive functional under v9-style controls。  
这是回答“functional update 是否应该继续作为核心”的第一硬 gate。

### 候选

```text
V8-B0-AdamWOnly
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

### 数据集与 seeds

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
epochs = 20
batch_size = 128
protocol = KANbeFair-compatible where applicable
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
step_ratio_q90
memory_ratio
control_rank
real_beats_adamwparallel
real_beats_best_lr
```

### 判断标准

v8 functional strong-control pass：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005.
$$

并且至少一个机制指标超过 AdamWParallel 与 best LR：

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

系统 gate：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p1_v8_functional_vs_strong_controls.svg
p1_v8_control_rank_by_dataset.svg
p1_v8_ce_margin_curvature_panel.svg
p1_v8_task_geometry_pareto.svg
```

---

## 6. 阶段 P2：v8 机制归因

### 目标

如果 P1 中 v8 functional survives controls，必须解释它为什么有效。  
如果 P1 中 v8 fails controls，也要解释它是如何被 LR / AdamWParallel 解释掉的。

### 必须记录

```text
candidate
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

### 机制定义

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

### 判断标准

v8 independent mechanism pass：

```text
cos_with_AdamW is not always > 0.95
functional beats LR controls
branch activity is non-trivial
curvature/tail improvement is not reproduced by NoOp/Random/LR
```

v8 LR-equivalent：

```text
cos_with_AdamW > 0.95
best LR control matches or beats functional
branch activity does not explain gains
```

### 可视化

```text
p2_rolewise_mechanism_heatmap.svg
p2_branch_ratio_vs_gain.svg
p2_cosine_with_adamw_histogram.svg
p2_effective_derivative_scale_trace.svg
p2_event_frequency_vs_tail_gain.svg
```

---

## 7. 阶段 P3：strict FC-PureKAN functional interface 设计

### 目标

把 v8 的机制迁移到 strict FC-PureKAN，而不是复制 v8 transitional architecture。

### 候选 families

#### F1：Dual-role edge basis interface

在输出层 edge basis 中分成 task channel 和 functional channel：

$$
\phi_{jc}(h)
=
c^{task}_{jc,0}B_0(h)
+
c^{task}_{jc,2}B_2(h)
+
c^{func}_{jc,a}B_a(h).
$$

Functional update 只作用于 $c^{func}_{jc,a}$，AdamW 主要作用于 task channel。  
这不是 external residual，因为所有通道仍在 edge function 内。

Candidates：

```text
F1a-T2-task + bounded-rational-functional
F1b-T2-task + piecewise-linear-functional
F1c-T2-task + centered-T2-functional
F1d-T2-task + shared-RBF-functional
```

#### F2：Derivative-scale controller

控制 edge basis derivative p95，使 functional update 有安全可调的扰动幅度：

$$
s_{\text{eff}}=|\alpha|\cdot B'_a(h)_{p95}.
$$

Candidates：

```text
F2a-bounded-rational-derivative-controller
F2b-piecewise-linear-derivative-controller
F2c-normalized-T2-derivative-controller
```

#### F3：Role-wise metric inside PureKAN

Functional 不再是 output target，而是 role-wise update metric：

$$
\theta_{t+1}
=
\theta_t+
P_{\text{role}}(\theta,t)\Delta\theta_{\text{AdamW}}.
$$

但 $P_{\text{role}}$ 必须不是 scalar LR disguise。它只能按 edge role / basis role 调整：

```text
lift role
task T2 role
functional actuator role
output edge role
```

Candidates：

```text
F3a-role-SNR metric
F3b-role-curvature damped metric
F3c-tail-event role metric
F3d-v8-FT7-style edge-role guard
```

#### F4：Non-AdamW orthogonal functional branch

如果 v8 mechanism 表明 functional direction 与 AdamW 不完全平行，则在 strict FC-PureKAN 中构造：

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

必须通过 paired replay：

$$
RealFunctional_\perp > Random_\perp.
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
p3_purekan_interface_contract_heatmap.svg
p3_interface_task_system_pareto.svg
p3_non_adamw_displacement_rank.svg
p3_paired_replay_vs_controls.svg
```

---

## 8. 阶段 P4：basis / primitive factory with functional actuatability gate

### 目标

不再只追 task/P4，而是把 functional actuatability 作为 primitive 晋级条件之一。

### Candidate basis families

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

Basis candidate 进入 full functional route 前必须同时满足：

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
p4_basis_factory_task_system_functional_pareto.svg
p4_basis_conditioning_heatmap.svg
p4_functional_actuatability_matrix.svg
p4_kmnist_delta_vs_basis.svg
```

---

## 9. 阶段 P5：AdamW-only full-pass repair 并行线

### 目标

Functional 是核心，但不能用 functional 掩盖 base 不够强。  
并行推进 AdamW-only full-pass，避免把 near-pass 的不稳定性误归因于 functional。

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

### Forbidden

```text
teacher
distillation
label smoothing
loss change
sampler
class weight
functional update
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
p5_adamw_fullpass_gap.svg
p5_kmnist_miss_rows.svg
p5_ce_tail_margin.svg
p5_basis_entropy_vs_gap.svg
```

---

## 10. 阶段 P6：short-run / full-run functional validation

### 目标

只有 P1 或 P3/P4 产生 paired replay survivor 后，才打开 short-run 和 full-run。  
不允许再从 direct oracle 或 target fit 直接进入 full run。

### Short-run 设置

```text
steps = 50,240
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
controls = AdamWOnly, AdamWParallel, best LR control, NoOp, Random, Shuffled
```

### Full-run 设置

```text
epochs = 20
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
baseline = MLP-match
diagnostic baseline = QuadraticFeatureMLP
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
p6_macro_delta_vs_controls.svg
p6_kmnist_repair_matrix.svg
p6_seedwise_win_matrix.svg
p6_task_geometry_pareto.svg
p6_ce_tail_margin_panel.svg
p6_ece_nll_panel.svg
```

---

## 11. 阶段 P7：robustness / external-ready gate

### 目标

验证 functional advantage 是否不只是 clean setting 偶然有效。

### Robustness 设置

```text
label_noise = 0.05,0.10,0.20
input_noise = 0.05,0.10
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
```

### Strong baseline

```text
MLP-match
same-shape MLP
hidden-bracket MLP
QuadraticFeatureMLP
LQ/A7c AdamW-only
functional candidate
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
p7_noise_robustness_curve.svg
p7_strong_baseline_pareto.svg
p7_external_ready_scorecard.svg
```

---

## 12. Required artifacts

```text
run_manifest.json
contract_audit_v9219.csv
p0_history_unified_audit.csv
p1_v8_strong_control_replay.csv
p2_v8_mechanism_attribution.csv
p3_purekan_functional_interface_redesign.csv
p4_basis_factory_functional_actuatability.csv
p5_adamw_only_fullpass_repair.csv
p6_functional_short_full_validation.csv
p7_robustness_external_ready.csv
paired_replay_branch_trace_v9219.csv
role_mechanism_trace_v9219.csv
purekan_interface_trace_v9219.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v8_strong_control_replay_missing
F3_v8_functional_lr_equivalent
F4_v8_mechanism_unattributed
F5_v9_interface_contract_fail
F6_v9_interface_p4_fail
F7_v9_interface_p5_nearpass_fail
F8_functional_control_equivalent
F9_basis_factory_no_candidate
F10_adamw_fullpass_fail
F11_short_run_task_drop
F12_full_run_no_macro_kmnist_gain
F13_external_not_ready
F14_fake_or_proxy_violation
F15_artifact_missing
```

---

## 13. Route decision

### Route cases

```text
R1-v8FunctionalSurvivesStrongControls:
  v8 functional beats AdamWParallel and LR controls.

R2-v8FunctionalIsLREquivalent:
  v8 functional explained by scalar LR / extra AdamW controls.

R3-FunctionalConceptValid_PrimitiveInterfaceMismatch:
  v8 survives controls, v9 current family fails; need strict PureKAN interface redesign.

R4-StrictPureKANInterfaceFunctionalPass:
  new strict PureKAN interface beats AdamWParallel / LR controls.

R5-NewPrimitiveFunctionalActuatabilityPass:
  basis/primitive factory finds candidate with P4/P5/function actuatability.

R6-AdamWFullPassNoFunctional:
  base reaches full-pass without functional; functional remains core research but not current evidence.

R7-FunctionalPausedButNotAbandoned:
  no current runnable evidence, but v8 unresolved; keep functional as research core and implement missing replay.

R8-FunctionalEvidenceDowngraded:
  v8 fails strong controls; current functional evidence downgraded.

R9-ExternalReady:
  strict FC-PureKAN functional route passes full task / robustness / strong baseline.
```

### route_decision.json 必须记录

```text
route
v8_strong_control_replay_pass
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
success_v9219_functional_core_retained
success_v9219_strict_purekan_functional
success_v9219_external_ready
```

---

## 14. 第一轮执行顺序

```text
Step 1:
  P0 统一历史审计。
  明确 v8 是 transitional success，不是 strict PureKAN success。

Step 2:
  P1 补跑 v8 strong-control replay。
  这是当前最重要的实验，不能跳过。

Step 3:
  P2 做 v8 机制归因。
  如果 v8 survives controls，抽取 role / branch / derivative 机制。

Step 4:
  P3 在 strict FC-PureKAN 内重建设计 functional interface。
  不允许 external residual / ordinary MLP path。

Step 5:
  P4 basis / primitive factory 加入 functional actuatability gate。
  不再只看 P4/P5。

Step 6:
  P5 并行修 AdamW-only full-pass。
  base 不能靠 functional 掩盖。

Step 7:
  P6 只对 paired replay survivor 做 short/full functional validation。

Step 8:
  P7 robustness / external-ready。
```

---

## 15. 停止条件

### Functional retained success

```text
v8 survives strong controls
and v8 mechanism identified
```

### Strict PureKAN functional success

```text
Functional retained success
+
new strict PureKAN interface passes P4/P5
+
paired replay beats AdamWParallel / LR controls
+
short/full run task-safe mechanism gain
```

### External-ready success

```text
Strict PureKAN functional success
+
robustness pass
+
strong baseline challenge pass
```

### Failure stop

```text
1. v8 strong-control replay cannot be implemented；
2. v8 functional is LR-equivalent；
3. v8 mechanism cannot be identified；
4. strict PureKAN interface fails contract；
5. no basis/primitive candidate passes P4/P5/functional actuatability；
6. all functional candidates remain control-equivalent；
7. full run gives no macro/KMNIST/geometry/tail gain；
8. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 16. 最终解释规则

### Case A：v8 survives strong controls

可以声明：

```text
Functional update remains a valid core mechanism; v9 failure is primitive/interface mismatch.
```

下一步继续 strict PureKAN functional interface redesign。

### Case B：v8 is LR-equivalent

必须声明：

```text
Prior functional evidence is downgraded under stronger controls.
```

但这仍不代表永远放弃 functional；它意味着 current implementation 不足，后续要从 primitive/interface 重建。

### Case C：strict PureKAN interface passes

可以声明：

```text
Functional update has been recovered in strict FC-PureKAN under strong controls.
```

### Case D：new primitive reaches AdamW full-pass but functional fails

必须声明：

```text
Base model improved; functional advantage remains unproven.
```

### Case E：all fail

必须声明：

```text
Functional remains a long-term core objective, but current evidence does not support continuing target/gate patches; next cycle must redesign primitive/basis.
```

---

## 17. 最终建议

v9.2.19 的一句话策略是：

$$
\boxed{
\text{不放弃 functional；先复验 v8 强控制，再把 v8 的有效机制迁移为 strict PureKAN 内部的 functional interface。}
}
$$

当前最重要的不是继续 patch v9 target，也不是提前 Conv / Former，而是：

```text
1. v8 functional 是否真的超过 AdamWParallel / LR controls？
2. v8 的 role-wise / branch-like 机制是什么？
3. strict FC-PureKAN 如何合法表达这种 functional interface？
4. 新 primitive 是否同时满足 P4/P5/function actuatability？
5. functional 能否最终超过 AdamWParallel / best LR control，并修复 KMNIST / CE-tail / margin-tail？
```
