# DG-KAN v6.1 Graph-Free Analytic Adjoint 实验计划

## 0. 本轮计划的重新定位

本轮计划的核心不是继续调 `DWM2 / RationalKAT / LUT / LightSmooth / U-FULL` 的超参数，而是重新定义 DG-KAN 的训练系统。此前许多实验虽然使用了 functional update rule，但梯度仍然大多来自 PyTorch autograd 的 `loss.backward()`。这意味着 forward 中产生的 basis、einsum、exp、rational 中间量、LUT 中间量等仍可能被 PyTorch backward graph 保存。只要这一点没有被消除，DG-KAN 就很难满足最终目标中的 backward 显存要求。

新的终极目标是：

$$
\boxed{
\text{构建一个无 non-KAN 参数的 PureKAN functional training system，}
}
$$

$$
\boxed{
\text{其 forward 时间/显存接近 MLP，backward 时间接近 MLP，backward 显存低于 MLP，}
}
$$

$$
\boxed{
\text{收敛速度快，并在 accuracy / AUC / ECE / geometry 上超过 MLP-AdamW 与 PureKAN-AdamW。}
}
$$

为了达到这个目标，v6.1 必须把以下两件事严格区分：

```text
functional update rule:
  拿到梯度以后如何更新参数，例如 Sobolev、TFU、FCAdam、LightSmooth。

analytic adjoint / graph-free training:
  梯度本身如何计算，是否依赖 PyTorch backward graph。
```

过去许多失败并不能完全证明某个 primitive 不可能高效，因为它们是在 autograd prototype 下被测量的。v6.1 的原则是：

$$
\boxed{
\text{任何效率结论，只承认 graph-free / manual-adjoint 路径下的结果。}
}
$$

如果某个候选仍然依赖 `loss.backward()` 或完整 PyTorch autograd graph，它只能作为机制 reference，不能作为终极系统候选。

---

## 1. 对已有结果的重新解释

### 1.1 v5.8 / v5.9 失败的真正含义

v5.8 和 v5.9 的 P0 都确认了新 primitive 在结构上是 strict PureKAN：edge-covered、nonKAN-free、rollback-exact。DWM2、RationalKAT、LUTKAN 等候选没有重新引入 MLP stem/head/LN 这类 non-KAN 参数。这个结果说明我们在模型纯度上没有倒退。

但是 v5.8 / v5.9 的效率阶段没有 survivor。v5.9 中，`DWM2-lite-RBFK2` 已经把 saved tensor volume 从 Dense-ABRBF 约 `90.20 MB` 降到约 `17.95 MB`，但它仍然是大约 `4.04x` forward、`2.81x` backward、`1.91x` backward memory。这个结果说明 DWM2-lite 已经减少了 PyTorch autograd 保存的张量，但还没有从根本上摆脱 autograd graph。

因此，v5.9 的失败应重写为：

$$
\boxed{
\text{当前 PyTorch-autograd prototype 没有满足效率门槛。}
}
$$

而不应直接写成：

$$
\boxed{
\text{DWM2 / LUT / RationalKAT 本身不可能满足效率门槛。}
}
$$

因为真正的 DG-KAN 终极实现应该使用解析梯度、手写反传、流式梯度累积和必要时的 recomputation。

### 1.2 当前代码实现的核心风险

当前基础 `RBFDense` 的 forward 仍是典型 autograd-friendly 写法：

```python
basis = self.basis(x)
out = torch.einsum("bik,oik->bo", basis, self.coeff)
```

这会显式构造：

```text
basis: [B, d, K]
```

如果直接进入 PyTorch backward，系统很可能会保存 basis 或其相关中间量。对于 Dense RBF / Dense AB-RBF，这个问题尤其严重，因为 forward 复杂度是：

$$
O(B d_{in} d_{out} K),
$$

而 MLP/Linear 是：

$$
O(B d_{in} d_{out}).
$$

所以 Dense RBF / Dense AB-RBF 不能作为终极高效系统。它们的价值是 mechanism reference / teacher，而不是最终 primitive。

### 1.3 v5.4 custom backward 给出的重要正信号

v5.4 的 custom backward audit 已经显示，手写或 recompute backward 可以大幅减少 saved tensor。比如 DepthwiseMix 的 saved tensor 从 autograd 约 `7.27 MB` 降到 custom backward 约 `1.09 MB`，RationalKAT 也从约 `1.53 MB` 降到约 `0.23 MB`。这说明 custom backward / recompute / streaming 并不是附属优化，而是达到终极目标的必要条件。

因此 v6.1 不再允许：

```text
P1 效率不过 -> gate 掉 custom backward
```

相反，custom backward / manual adjoint 是 v6.1 的第一主线。

---

## 2. 新的技术路线：Graph-Free Analytic Adjoint DG-KAN

### 2.1 基本思想

DG-KAN 的真正优势应该来自解析结构。以 RBF edge 为例：

$$
y_{bo}
=
\sum_{i,k} B_k(x_{bi}) c_{oik}.
$$

如果上游梯度为：

$$
\delta_{bo}=\frac{\partial L}{\partial y_{bo}},
$$

那么 coefficient 梯度可以解析写成：

$$
\frac{\partial L}{\partial c_{oik}}
=
\sum_b \delta_{bo} B_k(x_{bi}).
$$

输入梯度可以解析写成：

$$
\frac{\partial L}{\partial x_{bi}}
=
\sum_{o,k}\delta_{bo} c_{oik} B'_k(x_{bi}).
$$

所以终极实现不应该保存 PyTorch autograd graph，而应该执行：

```text
manual forward:
  计算 logits，同时保存极小 cache，或保存用于 recomputation 的输入。

manual CE delta:
  delta_logits = softmax(logits) - one_hot(y)

manual backward:
  从 output KAN 到 input KAN 手写反传。

manual update:
  用 functional rule / Adam-like rule / LightSmooth rule 更新 edge 参数。
```

v6.1 的核心训练循环应该长这样：

```python
with torch.no_grad():
    logits, cache = model.forward_manual(x)
    delta = softmax_cross_entropy_delta(logits, y)
    grad_pack = model.backward_manual(delta, cache)
    optimizer_manual.step(grad_pack)
```

禁止在终极候选中使用：

```python
loss.backward()
```

### 2.2 何为 graph-free 合格

一个候选必须满足：

```text
uses_loss_backward = False
uses_torch_autograd_graph = False
saved_tensor_total_bytes ≈ 0 or strictly bounded by small manual cache
grad_coeff_relerr vs autograd reference <= 1e-4
grad_input_relerr vs autograd reference <= 1e-4
```

如果使用 `torch.autograd.graph.saved_tensors_hooks` 审计，候选的 saved tensor 应该满足：

$$
\text{saved_tensor_total_bytes}_{KAN}
\ll
\text{saved_tensor_total_bytes}_{MLP}.
$$

如果某候选必须依赖 PyTorch autograd graph，它只能进入：

```text
mechanism reference
accuracy teacher
offline diagnostic
```

不能进入终极系统候选。

---

## 3. v6.1 的 primitive 优先级

### 3.1 第一优先级：SparseInterpKAN / SparseSplineKAN

Dense RBF 的根本问题是显式展开完整 $K$ 维 basis。SparseInterpKAN 用 piecewise-linear 或 compact local basis 取代 dense RBF。每个输入只激活两个 knot：

$$
q_i = \left\lfloor \frac{x_i-x_{min}}{\Delta}\right\rfloor,
$$

$$
t_i = \frac{x_i-(x_{min}+q_i\Delta)}{\Delta}.
$$

局部函数为：

$$
r_i(x_i) = (1-t_i) v_{i,q_i} + t_i v_{i,q_i+1}.
$$

加上 base path 后：

$$
z_i = a_i x_i + b_i + s_i r_i(x_i).
$$

再用 edge-system 内部 mixing：

$$
y = zW.
$$

计算复杂度是：

$$
O(Bd) + O(Bdh),
$$

主计算是 GEMM，接近 MLP。反传也只更新每个 channel 的两个 knot：

$$
\frac{\partial L}{\partial v_{i,q_i}}
+=
\delta z_i (1-t_i),
$$

$$
\frac{\partial L}{\partial v_{i,q_i+1}}
+=
\delta z_i t_i.
$$

输入梯度为：

$$
\frac{\partial r_i}{\partial x_i}
=
\frac{v_{i,q_i+1}-v_{i,q_i}}{\Delta}.
$$

SparseInterpKAN 的优势：

```text
不构造 [B,d,K] dense basis;
只保存或重算 q/t;
manual backward 简单;
和 LightSmooth / knot finite-difference geometry 天然匹配;
最有希望实现 backward 显存低于 MLP。
```

这是 v6.1 的第一主线。

### 3.2 第二优先级：DWM2-lite manual adjoint

DWM2-lite 之前已经减少 saved tensors，但仍在 autograd prototype 下不过 gate。v6.1 中它作为第二主线，用来验证：

```text
channel-wise AB-RBF residual + single GEMM mixing
```

在 manual adjoint 下能否达到效率门槛。

DWM2 的核心结构：

$$
z_i = \text{base}_i(x_i) + \text{residual}_i(x_i),
$$

$$
y = zW.
$$

如果 residual 仍用小 $K$ RBF，则 forward 为：

$$
O(BdK)+O(Bdh).
$$

若用 manual backward，每步只需重算小 $K$ basis，不必保存完整 autograd graph。

### 3.3 第三优先级：RationalKAT-lite manual adjoint

Rational/KAT 的工程形态最接近 MLP：

$$
z_i = r_i(x_i),
$$

$$
y = zW.
$$

计算复杂度是：

$$
O(Bd(m+n)) + O(Bdh).
$$

但 Rational 的 accuracy recipe 和 functional metric 还不稳定。因此 v6.1 中 RationalKAT-lite 的定位是：

```text
efficiency / memory candidate;
manual adjoint feasibility candidate;
不是当前 functional update 主线。
```

只有当 RationalKAT-lite 在 Adam-like manual training 下接近 MLP accuracy，才继续做 functional geometry。

### 3.4 Dense RBF / Dense AB-RBF 的新定位

Dense RBF / Dense AB-RBF 只保留为：

```text
mechanism reference;
accuracy teacher;
geometry teacher;
LightSmooth reference;
```

不再作为终极系统候选。

---

## 4. v6.1 实验总流程

v6.1 分成十个阶段。所有阶段都必须同时记录 task、geometry、efficiency 和 graph-free 指标。只有前一阶段过 gate，下一阶段才运行。

```text
P0: Graph-free implementation invariants
P1: Analytic gradient correctness
P2: Graph-free memory / time microbenchmark
P3: Primitive efficiency selection
P4: Manual training smoke
P5: Manual Adam-like convergence baseline
P6: Functional update without autograd
P7: Acceleration and fast convergence package
P8: Geometry maintenance / LightSmooth manual path
P9: 3-seed joint task-efficiency selection
P10: 5-seed / 10-seed confirm
```

---

## 5. P0：Graph-Free Implementation Invariants

### 5.1 目的

P0 不看 accuracy。它只确认候选是否真的不依赖 PyTorch backward graph。

### 5.2 方法

候选：

```text
MLP-AdamW-autograd-reference
Dense-ABRBF-autograd-reference
SparseInterpKAN-manual
DWM2-lite-RBFK2-manual
DWM2-lite-LUTK8-manual
RationalKAT-lite-manual
```

P0 对每个候选执行：

```text
forward_manual
manual_loss_delta
backward_manual
manual_update_noop
rollback
saved_tensors_hooks audit
```

### 5.3 必须记录

```text
uses_loss_backward
uses_torch_autograd_grad
uses_torch_autograd_graph
manual_forward_available
manual_backward_available
manual_update_available
nonKAN_param_count
edge_param_count
coverage_edge
coverage_base
coverage_residual
coverage_mixing
rollback_max_abs_error
saved_tensor_count
saved_tensor_total_bytes
saved_tensor_largest_bytes
manual_cache_bytes
cache_tensor_shapes_top20
```

### 5.4 Gate

候选通过 P0 必须满足：

$$
\text{uses_loss_backward}=0,
$$

$$
\text{uses_torch_autograd_graph}=0,
$$

$$
\text{nonKAN_param_count}=0,
$$

$$
\text{coverage_edge}=1.0,
$$

$$
\text{rollback_error}<10^{-8}.
$$

P0 如果不过，不允许进入 P1。

### 5.5 可视化

画一张 graph-free invariant dashboard：

```text
primitive vs uses_loss_backward
primitive vs saved_tensor_total_bytes
primitive vs manual_cache_bytes
primitive vs coverage_edge
primitive vs rollback_error
```

这张图必须清楚区分：

```text
autograd reference
manual candidate
```

---

## 6. P1：Analytic Gradient Correctness

### 6.1 目的

确认 manual adjoint 计算的梯度和 PyTorch autograd reference 一致。这个阶段允许用 autograd reference，但只用于对比，不用于候选效率。

### 6.2 测试对象

对每个 primitive 做三类测试：

```text
single layer gradient test
single residual block gradient test
full PureKAN gradient test
```

每类测试使用小 batch：

```text
B = 8 / 16
d = 16 / 64
K = 2 / 4 / 8 / 16
```

### 6.3 必须记录

```text
grad_coeff_relerr
grad_coeff_max_abs_error
grad_coeff_cosine
grad_input_relerr
grad_input_max_abs_error
grad_input_cosine
grad_base_relerr, if base exists
grad_mixing_relerr, if mixing exists
grad_residual_relerr
grad_norm_manual
grad_norm_autograd
```

### 6.4 公式

相对误差：

$$
\operatorname{relerr}(g_m,g_a)
=
\frac{\|g_m-g_a\|_2}{\|g_a\|_2+\epsilon}.
$$

梯度方向一致性：

$$
\cos(g_m,g_a)
=
\frac{g_m^Tg_a}{\|g_m\|\|g_a\|+\epsilon}.
$$

### 6.5 Gate

Exploratory gate：

$$
\operatorname{relerr}<10^{-4},
$$

$$
\cos>0.999.
$$

Final gate：

$$
\operatorname{relerr}<10^{-5},
$$

$$
\cos>0.9999.
$$

### 6.6 可视化

必须画：

```text
gradient relative error heatmap by primitive and role
gradient cosine bar chart
manual vs autograd gradient norm scatter
max absolute error distribution
```

---

## 7. P2：Graph-Free Memory / Time Microbenchmark

### 7.1 目的

P2 是 v6.1 的第一个真正效率门槛。只测 graph-free/manual candidates，不允许把 autograd candidate 当最终候选。

### 7.2 测试配置

数据维度：

```text
MNIST-like: input_dim = 784
hidden_dim = 64 / 96
depth = 2 / 4
batch_size = 128 / 256 / 512
```

候选：

```text
MLP-autograd-reference
SparseInterpKAN-K8-manual
SparseInterpKAN-K16-manual
SparseInterpKAN-K32-manual
DWM2-lite-RBFK2-manual
DWM2-lite-RBFK4-manual
DWM2-lite-LUTK8-manual
RationalKAT-lite-manual
Dense-ABRBF-autograd-reference
```

### 7.3 必须记录

```text
forward_time_ms
manual_backward_time_ms
optimizer_update_time_ms
step_time_ms
forward_memory_peak_mb
backward_memory_peak_mb
step_memory_peak_mb
reserved_memory_mb
manual_cache_bytes
saved_tensor_total_bytes
activation_cache_bytes
workspace_temp_bytes
kernel_count
num_gemm_calls
num_exp_calls
num_einsum_calls
num_scatter_add_calls
num_index_select_calls
bandwidth_estimate_gb_s
```

每项都要同时记录 ratio：

```text
forward_time_ratio_vs_mlp
backward_time_ratio_vs_mlp
step_time_ratio_vs_mlp
forward_memory_ratio_vs_mlp
backward_memory_ratio_vs_mlp
step_memory_ratio_vs_mlp
```

### 7.4 Gate

Exploratory efficiency gate：

$$
\frac{T_{forward,KAN}}{T_{forward,MLP}} \leq 1.50,
$$

$$
\frac{T_{backward,KAN}}{T_{backward,MLP}} \leq 1.80,
$$

$$
\frac{M_{backward,KAN}}{M_{backward,MLP}} \leq 1.00.
$$

Final efficiency gate：

$$
\frac{T_{forward,KAN}}{T_{forward,MLP}} \leq 1.25,
$$

$$
\frac{T_{backward,KAN}}{T_{backward,MLP}} \leq 1.30,
$$

$$
\frac{M_{backward,KAN}}{M_{backward,MLP}} \leq 0.80.
$$

P2 没有 survivor，则不允许进入 task recipe。此时必须回到 primitive / manual backward 实现。

### 7.5 可视化

必须画：

```text
efficiency Pareto: forward ratio vs backward memory ratio
step-time stacked bar: forward / backward / update
memory decomposition stacked bar: manual cache / workspace / optimizer state / saved tensors
primitive size scaling: batch size vs time / memory
kernel breakdown heatmap
```

---

## 8. P3：Primitive Efficiency Selection

### 8.1 目的

P3 在 P2 survivor 中选择最值得进入训练的 primitive。这里仍然不看 full training，只做短 batch 稳定性、数值稳定性和 simple loss descent。

### 8.2 必测候选

```text
SparseInterpKAN-K8-linear+silu
SparseInterpKAN-K16-linear+silu
SparseInterpKAN-K32-linear+silu
DWM2-lite-RBFK2-linear+silu
DWM2-lite-LUTK8-linear+silu
RationalKAT-lite-groups8
RationalKAT-lite-groups16
```

### 8.3 必须记录

```text
one_step_train_descent
one_step_val_descent
bad_step_rate
logit_norm
hidden_norm
margin_mean
margin_p10
effective_rank_input
effective_rank_block
effective_rank_output
knot_occupancy_entropy
knot_dead_fraction
basis_oog_fraction, for RBF
rational_den_min, for Rational
rational_den_condition, for Rational
```

### 8.4 Gate

候选必须满足：

```text
one_step_train_descent > 0
one_step_val_descent >= - small tolerance
bad_step_rate <= 0.05
no NaN / Inf
knot_dead_fraction <= 0.30
```

---

## 9. P4：Manual Training Smoke

### 9.1 目的

验证 manual adjoint path 能训练，而不是只会算一阶梯度。

### 9.2 方法

只用 P2/P3 survivor。每个候选跑：

```text
MNIST
Fashion-MNIST
KMNIST
seeds = 0,1,2
steps = 100 / 500
```

训练方式：

```text
Manual-SGD
Manual-AdamLike
Manual-NesterovAdamLike
```

此阶段不加 LightSmooth，不加复杂 functional geometry，只验证任务学习。

### 9.3 必须记录

```text
train_loss_curve
val_loss_curve
val_loss_auc_step
val_loss_auc_time
test_acc_at_100_steps
test_acc_at_500_steps
ECE
NLL
margin_mean
margin_p10
effective_rank
manual_backward_time_ms
step_time_ms
backward_memory_mb
```

### 9.4 Gate

短程 gate：

$$
\text{val_loss_auc_time}_{KAN}
\leq
1.25\cdot\text{val_loss_auc_time}_{MLP}.
$$

Accuracy gate：

$$
Acc_{KAN,500}
\geq
Acc_{MLP,500}-0.02.
$$

Memory gate 仍要求：

$$
M_{backward,KAN} \leq M_{backward,MLP}.
$$

---

## 10. P5：Manual Adam-Like Convergence Baseline

### 10.1 目的

v6.1 必须先证明 graph-free PureKAN 的任务学习能接近 MLP，然后才谈 functional update。P5 是第一个 full-budget task training 阶段。

### 10.2 方法

候选：

```text
MLP-AdamW-autograd
SparseInterpKAN-best-ManualAdam
DWM2-best-ManualAdam
RationalKAT-best-ManualAdam
Dense-ABRBF-AdamW-teacher, reference only
```

数据：

```text
MNIST
Fashion-MNIST
KMNIST
```

seed：

```text
0,1,2
```

### 10.3 必须记录

```text
test_acc
val_auc_step
val_auc_time
ECE
NLL
margin_mean
margin_p10
effective_rank
classwise_accuracy
confusion_matrix
step_time_ms
forward_time_ms
manual_backward_time_ms
optimizer_time_ms
peak_memory_mb
backward_memory_ratio
```

### 10.4 Gate

候选必须至少满足：

$$
Acc_{KAN}\geq Acc_{MLP}-0.01,
$$

$$
\text{val_auc_time}_{KAN}\leq1.25\cdot\text{val_auc_time}_{MLP},
$$

$$
ECE_{KAN}\leq ECE_{MLP}+0.02,
$$

$$
\frac{M_{backward,KAN}}{M_{backward,MLP}}\leq1.0.
$$

否则不进入 P6 functional update。

---

## 11. P6：Functional Update Without Autograd

### 11.1 目的

只有在 graph-free manual Adam-like 能学任务后，才引入 functional geometry。P6 不能再依赖 autograd 梯度。

### 11.2 候选 functional rule

```text
Manual-Functional-L2
Manual-Functional-dataSob
Manual-Functional-sparseSobolev
Manual-FCAdam-coordinate
Manual-FLD-AdanLite-coordinate
Manual-LightSmooth-event-lowfreq
```

对于 SparseInterpKAN，geometry 可以定义为 knot finite-difference：

$$
R_{smooth}
=
\sum_{i,q}(v_{i,q+1}-v_{i,q})^2.
$$

也可以加入二阶差分：

$$
R_{curv}
=
\sum_{i,q}(v_{i,q+2}-2v_{i,q+1}+v_{i,q})^2.
$$

### 11.3 必须记录

```text
smoothness_norm
curvature_norm
phi_like_slope_p95
finite_difference_curvature_p95
val_loss_auc_time
acc
ECE
NLL
backward_memory_ratio
step_time_ratio
functional_update_time_ms
```

### 11.4 Gate

Functional rule 必须相比 ManualAdam baseline 同时满足：

$$
Acc_{functional}\geq Acc_{ManualAdam}-0.005,
$$

$$
ECE_{functional}\leq ECE_{ManualAdam},
$$

$$
R_{curv,functional}\leq0.8R_{curv,ManualAdam},
$$

$$
\text{step_time_ratio increment}\leq1.10.
$$

---

## 12. P7：Acceleration and Fast Convergence Package

### 12.1 目的

用户终极目标里包含“收敛速度快”。因此 v6.1 不能只看 final accuracy。P7 系统验证加速收敛。

### 12.2 候选 optimizer dynamics

全部都必须是 manual / graph-free：

```text
ManualAdam
ManualNesterovAdam
ManualAdanLite
ManualWinLite
ManualLookaheadAdam
ManualAdamRestart
ManualAdanRestart
ManualFCAdam-AdanLite
```

### 12.3 加速指标

必须记录：

```text
time_to_target_loss
time_to_target_acc
epoch_to_target_loss
epoch_to_target_acc
val_loss_auc_step
val_loss_auc_time
early_loss_slope_step
early_loss_slope_time
best_acc_by_time_budget
best_val_loss_by_time_budget
restart_count
restart_reason
momentum_norm
velocity_norm
update_cos_previous
update_cos_gradient
```

### 12.4 Gate

加速候选必须满足：

$$
T_{target,KAN}\leq T_{target,MLP},
$$

或者在 early exploratory stage：

$$
T_{target,KAN}\leq1.15T_{target,MLP}.
$$

同时不得牺牲 memory：

$$
M_{backward,KAN}\leq M_{backward,MLP}.
$$

### 12.5 可视化

必须画：

```text
loss vs step
loss vs wall-clock time
accuracy vs wall-clock time
time-to-target bar chart
early slope comparison
momentum norm trajectory
restart marker plot
```

---

## 13. P8：Manual LightSmooth / Geometry Maintenance

### 13.1 目的

LightSmooth 不再作为主训练器，而是低频 geometry maintenance。它必须在 graph-free path 下运行。

### 13.2 流程

```text
train with best manual optimizer
if geometry debt high and task slack positive:
    apply manual LightSmooth
    run short refresh
```

### 13.3 触发条件

定义 geometry debt：

$$
D_g(t)=\frac{R_{curv}(t)}{R_{curv}(0)+\epsilon}.
$$

定义 task slack：

$$
S_t(t)=Acc(t)-Acc_{target\_floor}.
$$

触发条件：

```text
D_g > threshold
S_t > 0
val_loss plateau
cooldown passed
```

### 13.4 必须记录

```text
smooth_event_count
accepted_event_count
rejected_event_count
smooth_eta
acc_drop_after_smooth
KL_after_smooth
logit_drift_after_smooth
curvature_reduction_after_smooth
refresh_recovery_acc
geometry_retention_after_refresh
smoothing_time_ms
smoothing_memory_mb
amortized_overhead_ratio
```

### 13.5 Gate

LightSmooth 必须满足：

$$
\Delta Acc_{smooth+refresh}\geq -0.005,
$$

$$
R_{curv,after}\leq0.8R_{curv,before},
$$

$$
\text{amortized_time_overhead}\leq0.05.
$$

---

## 14. P9：3-Seed Joint Task-Efficiency Selection

### 14.1 目的

P9 只扩展 P5-P8 中同时满足 task / geometry / efficiency 的候选。

### 14.2 方法

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0,1,2

baselines:
  MLP-AdamW
  PureKAN-AdamW-autograd, if available
  Dense-ABRBF-teacher, reference only

candidates:
  best graph-free primitive + best manual optimizer
  best graph-free primitive + best accelerated manual optimizer
  best graph-free primitive + functional geometry maintenance
```

### 14.3 通过标准

候选必须满足：

$$
Acc_{KAN}\geq Acc_{MLP},
$$

$$
\text{val_auc_time}_{KAN}\leq\text{val_auc_time}_{MLP},
$$

$$
ECE_{KAN}\leq ECE_{MLP},
$$

$$
M_{backward,KAN}\leq0.9M_{backward,MLP},
$$

$$
T_{step,KAN}\leq1.25T_{step,MLP}.
$$

---

## 15. P10：5-Seed / 10-Seed Final Confirm

P10 只有在 P9 通过后运行。

### 15.1 5-seed confirm

```text
seeds = 0,1,2,3,4
```

要求 paired CI：

$$
CI_{95}(Acc_{KAN}-Acc_{MLP})>0
$$

或至少：

$$
CI_{95}(Acc_{KAN}-Acc_{MLP})\geq -0.002
$$

同时：

$$
CI_{95}(M_{backward,KAN}/M_{backward,MLP})<1.0.
$$

### 15.2 10-seed final

```text
seeds = 0..9
```

最终必须报告：

```text
mean / std
paired delta
bootstrap CI
failure cases
classwise robustness
speed and memory distribution
```

---

## 16. 统一记录字段

所有阶段至少记录以下字段。

### 16.1 Graph-free 字段

```text
uses_loss_backward
uses_autograd_grad
uses_autograd_graph
saved_tensor_count
saved_tensor_total_bytes
manual_cache_bytes
manual_cache_shapes
rollback_error
grad_relerr_vs_autograd
grad_cos_vs_autograd
```

### 16.2 效率字段

```text
forward_time_ms
backward_time_ms
optimizer_time_ms
step_time_ms
forward_memory_mb
backward_memory_mb
step_memory_mb
reserved_memory_mb
ratio_forward_time_vs_mlp
ratio_backward_time_vs_mlp
ratio_step_time_vs_mlp
ratio_backward_memory_vs_mlp
kernel_count
num_gemm_calls
num_exp_calls
num_einsum_calls
num_scatter_add_calls
```

### 16.3 任务字段

```text
train_loss
val_loss
test_loss
test_acc
val_loss_auc_step
val_loss_auc_time
ECE
NLL
margin_mean
margin_p10
classwise_acc
confusion_matrix
```

### 16.4 表示字段

```text
effective_rank_input
effective_rank_block
effective_rank_output
class_centroid_separation
feature_norm_mean
feature_norm_p95
hidden_drift
logit_drift
```

### 16.5 KAN / geometry 字段

```text
base_norm
residual_norm
base_over_residual
base_ablation_drop
residual_ablation_drop
knot_occupancy_entropy
knot_dead_fraction
basis_occupancy_entropy
out_of_grid_fraction
smoothness_norm
curvature_norm
phi_like_slope_p95
jacobian_condition_proxy
LightSmooth_event_count
LightSmooth_geometry_reduction
```

### 16.6 收敛加速字段

```text
time_to_target_loss
time_to_target_acc
early_loss_slope_step
early_loss_slope_time
best_acc_under_time_budget
restart_count
restart_reason
momentum_norm
velocity_norm
update_cos_prev
update_cos_grad
```

---

## 17. 必须可视化的图

### 17.1 Graph-free correctness dashboard

```text
primitive vs grad_relerr
primitive vs grad_cos
primitive vs saved_tensor_total_bytes
primitive vs manual_cache_bytes
```

### 17.2 Efficiency dashboard

```text
forward_time_ratio vs backward_memory_ratio
step_time stacked bar
memory stacked bar
batch-size scaling curve
hidden-dim scaling curve
```

### 17.3 Task-efficiency Pareto

横轴：

$$
\text{step time ratio vs MLP}
$$

纵轴：

$$
\text{accuracy delta vs MLP}
$$

点大小：

$$
\text{backward memory ratio}
$$

颜色：

```text
primitive family
```

### 17.4 Convergence speed plots

```text
val loss vs step
val loss vs wall-clock time
accuracy vs wall-clock time
time-to-target box plot
early slope bar chart
```

### 17.5 Geometry plots

```text
curvature_norm vs epoch
smoothness_norm vs epoch
geometry reduction vs accuracy delta
LightSmooth event markers over training curve
```

### 17.6 Representation plots

```text
effective rank vs epoch
margin p10 vs epoch
class centroid separation vs epoch
classwise accuracy heatmap
```

### 17.7 Failure taxonomy heatmap

行：

```text
primitive / optimizer
```

列：

```text
graph-free fail
grad correctness fail
forward time fail
backward memory fail
task fail
ECE fail
geometry fail
convergence speed fail
```

---

## 18. 决策树

### Case A：manual SparseInterpKAN 通过效率和任务 gate

这时 SparseInterpKAN 成为 v6.x 主线。继续做 functional geometry 和 acceleration。

### Case B：manual primitive 通过效率但 accuracy 不够

说明 primitive 形态可能对，但 recipe/capacity 不够。优先调：

```text
hidden_dim
knot_count
base path type
residual scale
normalization-free scaling
optimizer dynamics
```

不要回到 Dense RBF。

### Case C：manual primitive accuracy 够但效率不够

说明 manual backward 没实现到位。继续优化：

```text
cache design
streaming scatter-add
kernel fusion
Triton custom kernel
segment recomputation
```

### Case D：没有 primitive 通过 P2 效率门槛

暂停 functional optimizer，重新设计 primitive。不要运行 P4-P10。

### Case E：task 和 efficiency 都通过，但 functional geometry 不提升

保留 graph-free PureKAN task learner，LightSmooth 降级为 offline diagnostic。此时先完成高效 PureKAN task system，再做 geometry。

---

## 19. 本轮计划的最终判断

v6.1 的核心不是继续“调 optimizer”，而是实现真正的：

$$
\boxed{
\text{graph-free analytic-adjoint PureKAN training。}
}
$$

只有这一步成立，DG-KAN 才有可能满足：

```text
forward 接近 MLP
backward 接近 MLP
backward 显存低于 MLP
收敛更快
accuracy / AUC / ECE / geometry 超过 MLP-AdamW 与 PureKAN-AdamW
```

因此，v6.1 的最高优先级是：

```text
1. 不再使用 loss.backward() 作为终极候选路径。
2. 实现 manual forward / backward / update。
3. 先验证梯度正确性和显存收益。
4. 再验证任务收敛和加速。
5. 最后才引入 functional geometry maintenance。
```

一句话总结：

$$
\boxed{
\text{functional update 必须和 analytic adjoint 绑定；否则它只是昂贵 autograd 原型上的更新规则。}
}
$$
