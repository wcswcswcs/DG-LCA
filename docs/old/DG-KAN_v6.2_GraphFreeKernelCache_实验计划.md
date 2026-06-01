# DG-KAN v6.2 Graph-Free Kernel / Cache Redesign 实验计划

## 0. 终极目标重新固定

当前项目的终极目标不是“让某个 KAN 版本能跑”，也不是“只要 geometry 好就算成功”。终极目标必须固定为：

> 构建一个无 non-KAN 参数的 PureKAN functional training system，其 forward 时间/显存接近 MLP，backward 时间接近 MLP，backward 显存低于 MLP，收敛速度快，并在 accuracy / AUC / ECE / geometry 上超过 MLP-AdamW 与 PureKAN-AdamW。

因此，一个候选系统必须同时满足四类要求：

1. **结构纯度**：所有可学习参数都属于 KAN edge system，`nonKAN = 0`，没有 MLP stem/head/LN 参数作为隐藏支撑。
2. **训练路径纯度**：训练不依赖 `loss.backward()` 和 PyTorch autograd backward graph，必须使用 analytic adjoint / manual backward / streaming update。
3. **效率约束**：forward / backward / step time 接近 MLP，backward memory 低于 MLP 或至少明显低于 MLP-autograd reference。
4. **任务与几何目标**：accuracy、validation-loss AUC、ECE、geometry、收敛速度都要优于 `MLP-AdamW` 和 `PureKAN-AdamW`。

v6.1 已经把“functional update rule”和“graph-free analytic adjoint”拆开验证，这是正确方向。但 v6.1 也说明：当前 manual primitive 还没有进入 MLP-like efficiency envelope，因此还不能进入 task recipe / functional optimizer / LightSmooth 的正式阶段。

---

## 1. v6.1 结果复盘与当前判断

### 1.1 已经确认的正进展

v6.1 最重要的进展是：我们第一次把训练路径从 PyTorch autograd 中拆出来，建立了真正的 graph-free manual primitive benchmark。P0 显示手写 manual primitives 满足：

```text
nonKAN = 0
loss.backward = 0
saved = 0
rollback = 0
```

其中 `SparseInterpKAN-manual`、`DWM2-lite-RBFK2-manual`、`DWM2-lite-LUTK8-manual`、`RationalKAT-lite-manual` 都通过了 graph-free invariants。P1 更关键：所有 manual primitive 的 analytic gradient correctness 都达到：

```text
max coeff rel = 0
min coeff cos = 1.0
max input rel = 0
min input cos = 1.0
```

这说明当前 blocker 不再是“manual adjoint 公式错了”。至少在 P1 的测试范围内，coefficient gradient 和 input gradient 已经和 autograd 对齐。

因此当前正确结论是：

$$
\boxed{\text{graph-free analytic adjoint 路线成立，manual gradient correctness 已经通过。}}
$$

### 1.2 失败点已经从 backward graph 转移到 primitive / cache / kernel efficiency

P2 中没有任何 graph-free primitive 通过效率 gate。最接近的是 `DWM2-lite-RBFK2-manual`：

```text
forward ratio = 2.1639
backward ratio = 2.0543
backward memory ratio = 1.1747
step ratio = 1.5584
cache = 1.08 MB
```

这个结果有两个含义。

第一，manual graph-free path 已经比 dense RBF / dense AB-RBF 好很多。Dense-ABRBF-autograd-reference 的 forward ratio 是 `12.4203`，backward ratio 是 `6.5919`，step ratio 是 `6.2057`。所以 graph-free + lite primitive 不是没用。

第二，离终极目标仍然不够。当前最接近的 DWM2-lite-RBFK2 仍然有约 `2.16x` forward 和 `2.05x` backward，backward memory 也还是 `1.17x`，没有达到“forward 接近 MLP、backward memory 低于 MLP”的目标。

所以 v6.1 的核心结论应写成：

$$
\boxed{\text{graph-free 方向正确，但当前 manual primitive 的 kernel/cache 还没有 MLP-like。}}
$$

### 1.3 为什么这是进展而不是原地失败

此前 v5.8 / v5.9 的效率失败有一个根本不确定性：它们很多还依赖 PyTorch autograd graph，所以 backward memory 失败可能来自 autograd 保存中间图，而不是 primitive 本身。v6.1 排除了这个混淆。

现在我们知道：即使 `loss.backward=0`、`saved=0`、manual gradient 正确，primitive 仍然慢。这把问题缩小到更具体的工程与数学实现层面：

```text
1. manual forward/backward 的 kernel 数量是否太多；
2. sparse/interp 操作是否用了低效 scatter/gather；
3. cache 是否仍然保存了过多 per-layer tensor；
4. Python 层循环是否主导时间；
5. DWM2 / SparseInterp / Rational 的公式是否没有写成 GEMM-native 或 fused path；
6. MLP baseline 是高度优化 GEMM，而 manual primitive 还没有等价 kernel。
```

这说明下一步不是继续发明 optimizer，而是做 **Graph-Free Kernel / Cache Redesign**。

---

## 2. 当前 functional update 的重新定位

v6.1 之后，必须把系统拆成三层：

### 2.1 Functional update rule

这是参数更新公式，例如：

$$
\Delta \theta = -\eta M^{-1}g.
$$

之前我们试过 Sobolev、TFU、FNG、FCAdam、LightSmooth、BFT 等。这些回答的是：拿到梯度之后如何更新。

### 2.2 Analytic adjoint / graph-free gradient

这是如何得到 $g$。对 RBF edge，手写 adjoint 可以写成：

$$
y_{bo}=\sum_{i,k}B_k(x_{bi})c_{oik},
$$

$$
\frac{\partial L}{\partial c_{oik}}=\sum_b \delta_{bo}B_k(x_{bi}),
$$

$$
\frac{\partial L}{\partial x_{bi}}=\sum_{o,k}\delta_{bo}c_{oik}B_k'(x_{bi}).
$$

v6.1 的 P1 已经验证了这一层对若干 primitive 是正确的。

### 2.3 Efficient primitive / kernel path

这是当前真正 blocker。即使 analytic gradient 正确，如果 forward / backward 由大量 Python-level op、scatter/gather、pow/exp、小 kernel、临时 cache 组成，也无法接近 MLP 的 GEMM path。

当前 v6.2 的主问题是：

$$
\boxed{\text{如何把 graph-free analytic adjoint 写成 MLP-like kernel path。}}
$$

---

## 3. 当前关键假设

v6.2 不再只验证一个方向，而是并行验证多个可能原因。每个假设都必须有对应实验和可判定结果。

### H1：manual primitives 慢主要来自 Python / 小 kernel / 未融合操作

如果 forward/backward 由很多小操作组成，例如：

```text
per-layer loops
per-channel loops
index/gather/scatter
separate basis / derivative / update kernels
```

那么即使理论复杂度低，实际 wall-clock 也会远慢于 MLP。

验证方式：记录 `kernel_count`、`small_kernel_count`、`Python loop count`、`torch op count`、`forward_op_breakdown`，并对比 fused/vectorized 版本。

### H2：SparseInterpKAN 理论 sparse，但当前实现没有真正 fused

SparseInterp 的理论是每个输入只激活两个 knots：

$$
r_i(x_i)=(1-w_i)v_{i,q_i}+w_i v_{i,q_i+1}.
$$

理论复杂度接近：

$$
O(Bd)+O(Bdh).
$$

但 P2 中 SparseInterpKAN 的 forward 约 `6.4x`，backward 约 `6.2x`，说明它可能没有真正利用 sparse locality，而是被 `floor/clamp/gather/scatter`、分散 kernel 或 Python 实现拖慢。

验证方式：写 `SparseInterp-v2`，把 bin index 与 interpolation weight 的计算、前向插值、反向 knot gradient 累积统一成少数 fused kernel 或至少 vectorized batched kernel。

### H3：DWM2-lite-RBFK2 是当前最接近方向，但 RBF exp 仍然贵

DWM2-lite-RBFK2 是 P2 最接近的候选：step ratio `1.5584`，backward memory `1.1747`。它说明 DWM2-lite 的结构是对的，但 forward/backward 时间仍然约 `2x`。

可能原因是 RBF 的 `exp` 仍然贵，哪怕 $K=2$ 也需要额外 nonlinear eval。下一步要测试：

```text
DWM2-poly2
DWM2-hardswish
DWM2-piecewise-linear
DWM2-table-lookup
DWM2-rational-lite
```

核心判断：如果去掉 `exp` 后速度显著下降到 `1.3x` 以内，说明 RBF kernel 是主要时间瓶颈；如果没有改善，说明 mixing / cache / implementation 是主因。

### H4：cache MB 固定为 1.08 MB，说明 manual cache 仍有冗余

P2 中所有 manual primitives 的 `cache MB` 都是 `1.08 MB`，这很可疑。不同 primitive 的 cache 结构理论上应该不同：SparseInterp 应该只保存或重算 index/weight，Rational 可能只保存 input，DWM2-RBFK2 只需保存 normalized input 或少量 basis stats。

如果 cache MB 固定，可能说明 ManualStack 在所有 primitive 上统一保存了某些 full hidden / per-layer tensors。v6.2 必须拆 cache：

```text
cache_x_MB
cache_hidden_MB
cache_indices_MB
cache_weights_MB
cache_delta_MB
cache_logits_MB
cache_misc_MB
```

目标不是只看总 cache，而是知道 cache 为什么固定。

### H5：MLP reference 太强，必须确保 fair baseline

MLP 的 PyTorch Linear + SiLU + LayerNorm 走高度优化 GEMM / fused kernels。manual primitive 如果用 Python + torch ops，是不公平的。v6.2 必须同时保留：

```text
MLP-autograd-reference
MLP-manual-reference
MLP-graphfree-linear-manual
```

如果 graph-free MLP manual 也比 autograd MLP 慢很多，说明手写 runtime 本身不成熟；如果 graph-free MLP manual 接近 MLP autograd，而 KAN 仍慢，才说明 KAN primitive 本身慢。

### H6：当前 P2 gate 可能过早阻断 task recipe

v6.1 里 P3-P10 全部 gated，因为 P2 没 survivor。但 DWM2-lite-RBFK2 已经很接近 step ratio `1.56`，backward memory `1.17`。完全阻断任务训练会导致无法判断它是否具有更快收敛或更高 accuracy，从而抵消计算慢一点的成本。

v6.2 应该增加 “near-miss track”：对最接近效率门槛的候选，即使 P2 final gate 没过，也允许进入小规模 task smoke。目标不是 confirm，而是判断是否有 convergence compensation：

$$
T_{wall\text{-}target}=T_{step}\times N_{steps\text{-}to\text{-}target}.
$$

如果 step 慢 `1.5x`，但 step-to-target 少 `2x`，它仍然可能满足“收敛速度快”。

---

## 4. v6.2 总体策略

v6.2 的原则是：

```text
不要回到 PyTorch autograd；
不要继续 dense RBF；
不要急着 functional optimizer；
先证明 graph-free primitive 的 kernel/cache 能接近 MLP；
同时对 near-miss primitive 做最小收敛补偿验证。
```

也就是说，v6.2 要同时做两件事：

1. **Kernel / cache 修复**：让 graph-free primitive 更快更省显存。
2. **Near-miss convergence check**：如果某候选效率没完全过，但已经接近，则验证是否靠更快收敛补偿 wall-clock。

---

## 5. 实验阶段设计

## P0：Graph-free implementation manifest 与 cache 拆解

### 目的

确认所有候选都是真正 graph-free，并把 cache / kernel 来源拆清楚。

### 方法

必须跑：

```text
MLP-autograd-reference
MLP-manual-linear-reference
Dense-ABRBF-autograd-reference
DWM2-lite-RBFK2-manual-v61
DWM2-lite-RBFK2-manual-cacheMin
DWM2-lite-poly2-manual
SparseInterpKAN-K8-manual-v61
SparseInterpKAN-K8-manual-vectorized
RationalKAT-lite-manual-v61
RationalKAT-lite-manual-fastpoly
```

### 必须记录

```text
primitive_name
implementation_version
uses_loss_backward
uses_torch_autograd_graph
uses_custom_autograd_function
uses_manual_adjoint
edge_param_count
nonKAN_param_count
manual_cache_total_MB
cache_x_MB
cache_hidden_MB
cache_index_MB
cache_weight_MB
cache_delta_MB
cache_logits_MB
cache_misc_MB
kernel_count_forward
kernel_count_backward
python_loop_count_forward
python_loop_count_backward
op_count_exp
op_count_pow
op_count_gather
op_count_scatter
op_count_gemm
rollback_error
grad_check_available
```

### 成功标准

P0 必须满足：

```text
nonKAN_param_count = 0
uses_loss_backward = 0
uses_torch_autograd_graph = 0
manual_cache_total_MB is finite
rollback_error < 1e-8
```

如果 `MLP-manual-linear-reference` 本身比 `MLP-autograd-reference` 慢超过 `1.5x`，必须暂停 KAN primitive 比较，先修手写 runtime。

### 可视化

必须画：

```text
cache_decomposition_stacked_bar.svg
kernel_count_forward_backward_bar.svg
manual_vs_autograd_mlp_runtime.svg
op_breakdown_by_primitive.svg
```

---

## P1：Analytic gradient correctness 与 streaming update correctness

### 目的

v6.1 已经通过 gradient correctness，但 v6.2 新增 primitive / cacheMin / vectorized / fastpoly 后必须重新验证。

### 方法

对每个 primitive，在 batch size `32 / 128 / 512`、hidden dim `64 / 128` 上比较 manual gradient 与 autograd gradient。即使最终训练不用 autograd，gradient check 仍然用小 batch autograd reference。

### 必须记录

```text
primitive
shape_id
batch_size
hidden_dim
input_dim
num_classes
coeff_grad_relerr
coeff_grad_cos
input_grad_relerr
input_grad_cos
base_grad_relerr, if applicable
base_grad_cos, if applicable
mixing_grad_relerr, if applicable
mixing_grad_cos, if applicable
manual_forward_relerr
manual_forward_max_abs
manual_backward_nan_count
manual_update_nan_count
```

### 成功标准

```text
coeff_grad_relerr < 1e-5 or coeff_grad_cos > 0.99999
input_grad_relerr < 1e-5 or input_grad_cos > 0.99999
manual_forward_relerr < 1e-6
no NaN / Inf
```

如果某 primitive 只在 float32 下过，不在 mixed precision 下过，需要标记为 `fp32_only`，不能进入最终效率确认。

### 可视化

```text
grad_relerr_by_primitive.svg
grad_cos_by_primitive.svg
forward_relerr_heatmap.svg
```

---

## P2：Graph-free efficiency profiler v2

### 目的

在完全 graph-free path 下，重新测量 forward / backward / update / step 的时间和显存，并拆分瓶颈。

### 方法

采用 cold / warm 分离。每个 primitive 先 warmup `50` steps，再测 steady-state `200` steps。计时必须分成：

```text
manual_forward_time
loss_delta_time
manual_backward_time
manual_update_time
optimizer_state_time
total_step_time
```

内存必须分成：

```text
forward_peak_memory
backward_peak_memory
update_peak_memory
manual_cache_memory
optimizer_state_memory
workspace_memory
```

### 候选

核心候选：

```text
DWM2-lite-RBFK2-cacheMin
DWM2-lite-poly2
DWM2-lite-piecewiseLinear
DWM2-lite-fastRational
SparseInterpKAN-K8-vectorized
SparseInterpKAN-K8-fusedIndex
RationalKAT-lite-fastpoly
ManualLinear-MLP-reference
```

### Gate

探索门槛：

$$
\frac{T_{forward}}{T_{MLP}} \leq 1.75,
$$

$$
\frac{T_{backward}}{T_{MLP}} \leq 1.75,
$$

$$
\frac{M_{backward}}{M_{MLP}} \leq 1.05,
$$

$$
\frac{T_{step}}{T_{MLP}} \leq 1.50.
$$

最终门槛：

$$
\frac{T_{forward}}{T_{MLP}} \leq 1.25,
$$

$$
\frac{T_{backward}}{T_{MLP}} \leq 1.30,
$$

$$
\frac{M_{backward}}{M_{MLP}} \leq 0.80,
$$

$$
\frac{T_{step}}{T_{MLP}} \leq 1.25.
$$

### Near-miss 规则

如果某候选满足：

```text
step ratio <= 1.75
backward memory ratio <= 1.25
manual gradient correct
```

即使没有通过最终门槛，也进入 P4 near-miss convergence smoke。

### 可视化

```text
forward_backward_step_ratio_bar.svg
memory_ratio_bar.svg
time_breakdown_stacked_bar.svg
cache_vs_runtime_scatter.svg
kernel_count_vs_forward_time.svg
efficiency_pareto_step_vs_bmem.svg
```

---

## P3：Kernel / cache ablation

### 目的

找出 P2 中每个 near-miss primitive 具体慢在哪里，而不是只看整体 ratio。

### DWM2-lite 分解

对 DWM2-lite-RBFK2：

```text
basis eval only
base path only
mixing GEMM only
manual backward input only
manual backward coeff only
manual update only
```

同时测试替代 nonlinearity：

```text
RBFK2
poly2
piecewiseLinear
fastRational
hardswish
```

### SparseInterp 分解

对 SparseInterp：

```text
bin index compute only
interpolation weight compute only
forward interpolation only
knot gradient accumulation only
input gradient only
mixing GEMM only
```

必须比较：

```text
index saved vs recomputed
scatter_add vs segmented reduce
vectorized gather vs fused kernel
K=8 / K=16 / K=32
```

### RationalKAT-lite 分解

对 RationalKAT-lite：

```text
numerator eval
 denominator eval
 division
 derivative
 mixing GEMM
 manual coeff grad
```

测试：

```text
Horner polynomial eval
precomputed powers
grouped vectorized eval
fast reciprocal approximation, diagnostic only
```

### 必须记录

```text
component_time_ms
component_memory_MB
component_kernel_count
component_temp_bytes
component_fraction_of_forward
component_fraction_of_backward
```

### 可视化

```text
component_runtime_waterfall_by_primitive.svg
component_memory_waterfall_by_primitive.svg
nonlinearity_runtime_comparison.svg
scatter_vs_segmented_reduce.svg
```

---

## P4：Near-miss convergence smoke

### 目的

验证“效率接近但未完全过 gate”的候选是否能通过更快收敛补偿 step time。这个阶段不做 full confirm，只做短程判断。

### 方法

候选来自 P2 near-miss：预期至少包括 `DWM2-lite-RBFK2-cacheMin` 或其替代版本。

对每个候选跑：

```text
MNIST
Fashion-MNIST
KMNIST
seeds = 0,1,2
train budget = 20 epochs or fixed 1000 steps
```

对照：

```text
MLP-AdamW-autograd-reference
MLP-manual-linear-reference
PureKAN-AdamW-autograd-reference, optional
Dense-ABRBF-teacher, optional
```

### 必须记录

```text
train_loss_curve_by_step
val_loss_curve_by_step
test_acc_by_epoch
ECE_by_epoch
NLL_by_epoch
margin_mean_by_epoch
margin_p10_by_epoch
feature_rank_by_epoch
step_time_ms
wall_clock_time_sec
time_to_target_loss
time_to_target_acc
steps_to_target_loss
steps_to_target_acc
val_loss_auc_by_step
val_loss_auc_by_time
```

### 目标定义

如果候选 step time 是 MLP 的 $r_t$ 倍，则它必须在 step 数上减少足够多：

$$
\frac{N_{target,KAN}}{N_{target,MLP}} < \frac{1}{r_t}.
$$

也就是：

$$
T_{target,KAN}=r_tN_{target,KAN}<N_{target,MLP}=T_{target,MLP}.
$$

### 成功标准

P4 通过需满足：

```text
至少两个数据集 time_to_target_loss <= MLP
至少两个数据集 val_loss_auc_by_time >= MLP
accuracy gap <= 2%
ECE <= MLP + 0.03
backward memory ratio <= 1.25
```

如果 P4 失败，但 P2 效率已经非常接近，则说明 recipe / optimizer 不够；如果 P4 成功，则该候选进入 P5。

### 可视化

```text
loss_vs_step.svg
loss_vs_time.svg
accuracy_vs_time.svg
time_to_target_bar.svg
auc_by_step_vs_auc_by_time.svg
step_ratio_vs_step_saving_scatter.svg
```

---

## P5：Acceleration package for graph-free candidates

### 目的

在 graph-free primitive 上验证是否可以通过加速优化器获得更快收敛。注意：这一步仍然不依赖 PyTorch autograd。

### 方法

对 P4 near-miss / survivor 候选，比较：

```text
Manual-SGD-momentum
Manual-AdamW
Manual-NesterovAdamW
Manual-AdanLite
Manual-WinLite
Manual-LookaheadAdam
Manual-AdamW-restart
Manual-AdanLite-restart
```

这些优化器都只使用 manual gradient，不使用 autograd graph。

### 关键记录

```text
optimizer_name
momentum_norm
rms_norm
update_norm
update_over_param
cos_update_grad
cos_update_previous
restart_count
restart_reason
lookahead_sync_count
loss_slope_early
loss_slope_middle
time_to_target_loss
time_to_target_acc
```

### Gate

一个 optimizer 进入 P6 必须满足：

```text
time_to_target_loss improves over Manual-AdamW by >= 15%
final acc gap vs MLP <= 1.5%
ECE <= MLP + 0.03
no unstable restart explosion
```

### 可视化

```text
optimizer_loss_slope.svg
optimizer_time_to_target.svg
momentum_norm_trace.svg
restart_marker_loss_curve.svg
update_cosine_trace.svg
```

---

## P6：Manual functional geometry maintenance smoke

### 目的

只有在 P2/P4/P5 找到效率与收敛候选后，才重新引入 functional geometry maintenance。此阶段验证：graph-free primitive 能否在不增加显存/时间过多的情况下做 residual geometry maintenance。

### 方法

比较：

```text
Manual-AdamW-only
Manual-AdamW + low-frequency LightSmooth
Manual-AdanLite + low-frequency LightSmooth
Manual-AdamW + residual curvature penalty, cheap
Manual-AdamW + post-epoch residual projection, cheap
```

LightSmooth 必须也是 graph-free / no-autograd，不允许调用 PyTorch backward。

### 记录指标

```text
smoothing_event_count
smoothing_time_ms
smoothing_memory_MB
amortized_time_overhead
amortized_memory_overhead
acc_before_smooth
acc_after_smooth
acc_after_refresh
KL_before_after
logit_drift
phi_residual_before_after
curvature_residual_before_after
geometry_retention_after_refresh
```

### Gate

```text
amortized_time_overhead <= 10%
amortized_memory_overhead <= 10%
acc_after_refresh >= no_smoothing - 0.005
phi_residual_reduction >= 10% or curvature_residual_reduction >= 25%
```

### 可视化

```text
smoothing_event_timeline.svg
geometry_before_after_smooth.svg
refresh_recovery_curve.svg
amortized_overhead_bar.svg
```

---

## P7：3-seed joint selection

### 目的

把所有维度合并：task、efficiency、memory、convergence、geometry。只保留真正可能接近终极目标的候选。

### 候选

来自 P5/P6 survivors，最多保留 `3` 个：

```text
best_graphfree_primitive_manual_adamw
best_graphfree_primitive_accelerated
best_graphfree_primitive_accelerated_lightsmooth
```

### 数据集

```text
MNIST
Fashion-MNIST
KMNIST
```

### seeds

```text
0,1,2
```

### 记录指标

任务：

```text
test_acc
val_loss_auc_step
val_loss_auc_time
ECE
NLL
margin_mean
margin_p10
classwise_acc
```

效率：

```text
forward_time_ratio
backward_time_ratio
step_time_ratio
backward_memory_ratio
manual_cache_MB
optimizer_state_MB
activation_saved_bytes
```

收敛：

```text
time_to_target_loss
time_to_target_acc
steps_to_target_loss
steps_to_target_acc
early_loss_slope
middle_loss_slope
```

几何：

```text
phi_residual_p95
curvature_residual
jacobian_condition
geometry_reduction_vs_manual_adamw
```

结构：

```text
edge_param_count
nonKAN_param_count
base/residual contribution, if applicable
knot occupancy entropy
active knot fraction
dead knot fraction
```

### Gate

进入 5-seed confirm 必须满足：

```text
nonKAN = 0
uses_loss_backward = 0
accuracy >= MLP - 1%
val_loss_auc_time >= MLP
ECE <= MLP + 0.02
step_time_ratio <= 1.50
backward_memory_ratio <= 1.00
至少一个 geometry 指标优于 MLP 或 PureKAN-AdamW
```

### 可视化

```text
joint_scorecard.svg
task_efficiency_pareto.svg
memory_accuracy_pareto.svg
time_to_target_paired_bar.svg
geometry_vs_accuracy_pareto.svg
```

---

## P8：5-seed confirm

### 目的

确认 P7 survivor 不是 seed 偶然。

### 设置

```text
seeds = 0,1,2,3,4
methods:
  MLP-AdamW-autograd-reference
  MLP-manual-reference, if P0 valid
  best graph-free PureKAN candidate
  best graph-free PureKAN + acceleration
  best graph-free PureKAN + acceleration + LightSmooth, if P6 passed
```

### 成功标准

```text
paired acc delta vs MLP >= 0 on at least two datasets
paired val_loss_auc_time delta >= 0 on at least two datasets
backward_memory_ratio <= 1.0 mean and <= 1.2 max
step_time_ratio <= 1.5 mean
ECE not worse than MLP
geometry improves over MLP/PureKAN baseline
```

### 可视化

```text
paired_delta_accuracy_ci.svg
paired_delta_auc_time_ci.svg
memory_ratio_seed_scatter.svg
step_time_seed_scatter.svg
calibration_curve_comparison.svg
```

---

## P9：10-seed final confirm

### 目的

只有 P8 通过才跑。确认最终候选是否真正推进终极目标。

### 成功标准

最终候选必须满足：

$$
\operatorname{Acc}_{KAN} \geq \operatorname{Acc}_{MLP},
$$

$$
\operatorname{AUCtime}_{KAN} \geq \operatorname{AUCtime}_{MLP},
$$

$$
\operatorname{ECE}_{KAN} \leq \operatorname{ECE}_{MLP},
$$

$$
\frac{M_{backward,KAN}}{M_{backward,MLP}} < 1.0,
$$

并且目标方向是：

$$
\frac{M_{backward,KAN}}{M_{backward,MLP}} \leq 0.8.
$$

如果候选 accuracy 与 AUC 过，但 backward memory 只是 `0.95x - 1.0x`，可以称为 **near-final graph-free PureKAN**，但不能称为完全达成终极目标。

---

## 6. 关键 CSV 字段规范

为了避免后续结果难分析，v6.2 每个 run 至少写出下面字段。

### 通用字段

```text
stage
package
primitive
method
dataset
seed
shape_id
batch_size
hidden_dim
depth
edge_param_count
nonKAN_param_count
uses_loss_backward
uses_torch_autograd_graph
manual_adjoint
manual_update
```

### 效率字段

```text
forward_time_ms
loss_time_ms
backward_time_ms
update_time_ms
step_time_ms
forward_time_ratio
backward_time_ratio
step_time_ratio
forward_peak_memory_MB
backward_peak_memory_MB
update_peak_memory_MB
backward_memory_ratio
manual_cache_total_MB
cache_x_MB
cache_hidden_MB
cache_index_MB
cache_weight_MB
cache_delta_MB
cache_misc_MB
kernel_count_forward
kernel_count_backward
op_count_exp
op_count_pow
op_count_gather
op_count_scatter
op_count_gemm
```

### 梯度正确性字段

```text
coeff_grad_relerr
coeff_grad_cos
input_grad_relerr
input_grad_cos
base_grad_relerr
base_grad_cos
mixing_grad_relerr
mixing_grad_cos
forward_relerr
forward_max_abs
```

### 训练字段

```text
train_loss
val_loss
test_acc
ECE
NLL
val_auc_by_step
val_auc_by_time
steps_to_target_loss
time_to_target_loss
steps_to_target_acc
time_to_target_acc
early_loss_slope
middle_loss_slope
```

### 几何字段

```text
phi_residual_p95
curvature_residual
jacobian_condition
knot_occupancy_entropy
active_knot_fraction
dead_knot_fraction
geometry_reduction_vs_baseline
```

### 加速优化字段

```text
optimizer_type
momentum_norm
rms_norm
update_norm
update_over_param
cos_update_grad
cos_update_previous
restart_count
restart_reason
lookahead_sync_count
```

---

## 7. 必须生成的图

v6.2 必须生成这些图，缺任何一类都不能说分析完成。

### 7.1 Graph-free efficiency dashboard

包含：

```text
forward ratio bar
backward ratio bar
step ratio bar
backward memory ratio bar
manual cache MB bar
```

目的：判断是否真的接近 MLP。

### 7.2 Cache decomposition

堆叠柱状图：

```text
x cache
hidden cache
index cache
weight cache
delta cache
misc cache
```

目的：解释 cache MB 为什么固定，找出可删项。

### 7.3 Component runtime waterfall

对每个 near-miss primitive 画 waterfall：

```text
basis/index eval
activation transform
mixing GEMM
input gradient
coeff gradient
update
```

目的：知道应该 fusion 哪一段。

### 7.4 Step vs wall-clock convergence

画：

```text
val loss vs step
val loss vs wall-clock time
accuracy vs wall-clock time
```

目的：判断 slow step 是否被 faster convergence 补偿。

### 7.5 Task-efficiency Pareto

横轴：

$$
\text{step time ratio}
$$

纵轴：

$$
\text{test accuracy}
$$

点大小：

$$
\text{backward memory ratio}
$$

目的：选择候选。

### 7.6 Memory-accuracy Pareto

横轴：

$$
\text{backward memory ratio}
$$

纵轴：

$$
\text{test accuracy}
$$

颜色表示 primitive。

### 7.7 Acceleration dynamics

画：

```text
momentum norm trace
RMS norm trace
update cosine trace
restart markers on loss curve
```

目的：判断加速方案是否稳定。

---

## 8. 最终决策树

v6.2 的最终决策应该按下面顺序，而不是只看一个指标。

### Case A：存在 graph-free efficiency survivor

如果某 primitive 满足：

```text
forward <= 1.25x
backward <= 1.30x
backward memory <= 0.80x
step <= 1.25x
```

则直接进入 P4-P9，重点验证 task 与 convergence。

### Case B：没有 final survivor，但存在 near-miss candidate

如果最优候选类似当前 DWM2-lite-RBFK2：

```text
step <= 1.75x
backward memory <= 1.25x
```

则进入 near-miss convergence smoke。若 time-to-target 赢 MLP，可以继续；若不赢，回到 kernel redesign。

### Case C：manual MLP reference 也很慢

说明 manual runtime 本身不成熟。优先修 runtime，不评价 KAN primitive。

### Case D：manual primitive 效率过了，但 task 不过

说明 primitive 表达力 / recipe 不够。进入 task recipe / acceleration，而不是继续 kernel。

### Case E：task 过了、效率过了，但 geometry 不过

说明可以重新加入 LightSmooth / residual geometry maintenance。

---

## 9. 当前最优先的三项工作

### 第一优先：DWM2-lite-RBFK2-cacheMin

这是 v6.1 当前最接近效率目标的候选。必须拆解它的 forward/backward time，确认是否能从 `2.16x / 2.05x` 压到 `1.5x` 以下。

### 第二优先：SparseInterpKAN-vectorized / fused

SparseInterp 理论最符合 backward memory 低于 MLP 的目标，但 v6.1 实现非常慢。它需要 kernel-level 重写，而不是继续用现有实现判断路线失败。

### 第三优先：ManualLinear reference

如果手写 Linear graph-free 不能接近 MLP autograd，就说明 benchmark runtime 不公平，必须先修 manual framework。

---

## 10. 总结

v6.1 之后，我们终于站在了更正确的道路上：

$$
\boxed{\text{graph-free analytic adjoint 已经验证正确。}}
$$

但同时也暴露了新的硬问题：

$$
\boxed{\text{当前 manual primitive 仍然不是 MLP-like kernel path。}}
$$

所以 v6.2 的核心不是继续调 functional optimizer，而是：

$$
\boxed{\text{把 graph-free primitive 从“正确”推进到“高效”。}}
$$

只有当 graph-free primitive 先过 efficiency gate，functional update、LightSmooth、加速收敛实验才有进入最终系统的意义。
