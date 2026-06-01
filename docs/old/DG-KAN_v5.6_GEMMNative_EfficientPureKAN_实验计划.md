# DG-KAN v5.6：GEMM-Native Efficient PureKAN 下一步实验计划

## 0. 当前终极目标

本阶段必须以新的终极目标作为硬约束，而不是只追求某个 optimizer 在 MNIST-family 上的 accuracy 或 geometry：

$$
\boxed{\text{构建一个无 non-KAN 参数的 PureKAN functional training system。}}
$$

它必须同时满足：

$$
\boxed{\text{forward 时间/显存接近 MLP，backward 时间接近 MLP，backward 显存低于 MLP。}}
$$

并且：

$$
\boxed{\text{收敛速度快，accuracy / AUC / ECE / geometry 超过 MLP-AdamW 与 PureKAN-AdamW。}}
$$

因此，v5.6 的评价顺序必须改变。以前我们经常先问 “某个 functional update 是否能改善 geometry”，现在必须先问：

```text
这个 PureKAN primitive 是否真的有 MLP-like compute path？
```

如果 primitive 本身的 forward / backward / memory 不接近 MLP，那么后续 functional optimizer、LightSmooth、NFS、TAN、FNG 都不能算通向终极目标的主线。它们最多只能作为机制诊断。

---

## 1. v5.5 结果复盘与重新定性

### 1.1 v5.5 做对了什么

v5.5 的价值在于，它第一次把问题拆成了明确的 kernel-first 实验，而不是继续让 optimizer 和 primitive 混在一起。P0 证明候选 primitive 都保持在 audited edge system 内，`nonKAN = 0`，rollback 也正常。候选包括：

```text
RBFOnly-Dense
ABRBF-Dense
ABRBF-DepthwiseMix
ABRBF-CPRank4 / 8 / 16
RationalKAT-AB
```

这说明实验对象符合 PureKAN 约束：可学习参数不再依赖普通 MLP stem/head/LN。

P1/P2/P3/P4 进一步暴露了三个有用信号：

```text
DWM-2-compiled 有真实 kernel-path gain。
CP-1-two-stage-gemm / CP-3-compiled 修复了部分 rank-speed 行为。
RK-3-compiled 是最清楚的 memory/kernel signal。
```

这说明我们不是没有高效方向，而是还没有把 “高效 kernel path” 和 “足够 task accuracy” 同时跑通。

### 1.2 v5.5 没过的地方

v5.5 的核心失败是：

$$
\boxed{\text{没有任何 primitive 同时通过 efficiency gate 和 accuracy gate。}}
$$

P1 没有 final MLP-like efficiency survivor。Dense RBF / Dense AB-RBF 过慢且显存高，理论上也不可能接近 MLP，因为它们需要展开 basis 维度 $K$：

$$
\text{Dense RBF cost} = O(B d_{in} d_{out} K),
$$

而 MLP linear layer 是：

$$
\text{MLP cost} = O(B d_{in} d_{out}).
$$

P5 则说明，虽然 CP 在 MNIST/Fashion 上有较好的 task signal，但 KMNIST 仍低于需要的 frontier；DWM 和 RationalKAT 的 kernel path 有速度/显存信号，但 recipe 还没有释放 accuracy。

所以 v5.5 的真正结论不是 “efficient PureKAN 不行”，而是：

$$
\boxed{\text{kernel path 已经出现，但 task recipe 还没修好。}}
$$

### 1.3 现在不要做什么

v5.5 之后不应该立刻回到：

```text
Sobolev / TFU / FNG / FCAdam / NFS / LightSmooth 大扫
```

也不应该继续把 Dense AB-RBF 当作最终系统。Dense AB-RBF 是很好的机制 reference，但它不满足终极效率目标。

下一轮必须聚焦：

$$
\boxed{\text{先让一个 GEMM-native / compiled efficient primitive 同时过 task 和 efficiency。}}
$$

只有做到这一点，functional optimizer 才有继续投入的意义。

---

## 2. 当前代码实现需要重点审视的地方

### 2.1 Dense RBFDense 的天然限制

当前基础 `RBFDense` 的 forward 逻辑是：

```text
basis = exp(-0.5 * ((x - centers) / width)^2)
out = einsum("bik,oik->bo", basis, coeff)
```

这会显式形成：

$$
\text{basis} \in \mathbb R^{B \times d_{in} \times K}.
$$

然后执行 edge-wise dense aggregation：

$$
y_o = \sum_{i,k} B_k(x_i)c_{oik}.
$$

因此它无法天然满足：

$$
T_{forward,KAN} \approx T_{forward,MLP}.
$$

它的价值是机制研究，不是最终高效 primitive。

### 2.2 Efficient primitive 必须进入 core，而不是只在 runner 中临时实现

v5.5 P0 显示 efficient primitives 已被审计：`ABRBFDepthwiseMixDense`、`CPABRBFDense`、`RationalKATDense` 都保持 edge-only 参数约束。但下一轮仍然必须做 core consistency hardening：

```text
1. 实际训练 import 的 dgkan_core.py 必须包含这些 primitive。
2. edge_named_params / base_named_params / rbf_residual_named_params / mixing_named_params 必须统一。
3. 所有 runner 不能自己偷偷定义临时 helper primitive。
4. P0 必须输出每个参数组的 numel、requires_grad、optimizer group、functional group。
```

如果 primitive 只存在于 runner/helper 中，后续 functional update、LightSmooth、custom backward、profiling 都会产生分裂结果。

### 2.3 custom backward 不能只看 saved bytes

v5.4/v5.5 的 custom backward 结果说明，仅仅减少 saved tensors 不等于降低 measured peak memory。可能原因包括：

```text
CUDA workspace / temporary tensor 仍然很大；
recompute backward 触发额外 kernel 和中间 buffer；
PyTorch memory allocator 的 reserved memory 没下降；
compile graph 与 eager graph 的 workspace 不同；
profiling 没拆开 forward activation、backward temp、optimizer state。
```

因此 v5.6 必须把 memory 拆成：

```text
forward activation bytes
saved tensor bytes
backward temporary bytes
optimizer state bytes
workspace / reserved delta
peak allocated MB
peak reserved MB
```

否则会误判 “显存优化是否有效”。

---

## 3. v5.6 的核心假设

v5.6 不再问：

```text
functional update 还要怎么修？
```

而是先问：

```text
哪一种 PureKAN edge primitive 有机会满足 MLP-like efficiency，同时保住 task accuracy？
```

我建议 v5.6 同时保留三条候选路线，但优先级不同。

### 3.1 第一候选：DepthwiseMix，修 task recipe

DepthwiseMix 理论上最符合 MLP-like compute：

$$
\tilde x_i = f_i(x_i),
$$

$$
y = W\tilde x.
$$

计算量接近：

$$
O(B d K) + O(B d h).
$$

它避免了 Dense RBF 的 $O(B d h K)$。v5.5 里 `DWM-2-compiled` 已经暴露出真实 kernel-path gain，说明这条路值得优先救。

但 DWM 的问题是 task accuracy 不够。下一轮应该重点修：

```text
channel function capacity；
LinearMix 初始化；
base path 类型；
normalization / input scaling；
DWM residual depth；
DWM mixing width；
learning rate recipe。
```

### 3.2 第二候选：RationalKAT-AB，修 recipe 和 kernel

RationalKAT 理论上最接近 MLP，因为它是：

```text
elementwise rational activation + GEMM mixing
```

计算量接近：

$$
O(Bd(m+n)) + O(Bdh).
$$

v5.5 的 `RK-3-compiled` 是最清楚的 memory/kernel signal，但 compact recipe 远低于 MLP，尤其 KMNIST。下一轮要先把它作为 AdamW / edge-system primitive 修通，不要立刻上 functional update。

RationalKAT 的下一步不是 “tangent functional metric”，而是：

```text
recipe repair first, functional metric later。
```

### 3.3 第三候选：CP-ABRBF，保留为 accuracy reference

CP-ABRBF 在 MNIST/Fashion 上有较好的 task signal，但 v5.5 显示它仍不满足最终 efficiency envelope。它可以作为中间 reference：

```text
如果 DWM/RK 很快但 accuracy 差，CP 告诉我们需要多少 capacity；
如果 CP accuracy 好但效率不够，它不是终极 primitive。
```

CP 只有在 rank-speed monotonic、accuracy 接近 MLP 且 memory 进一步下降时，才进入最终候选。

---

## 4. v5.6 成功标准

v5.6 不允许只看 accuracy，也不允许只看 step time。每个候选必须同时通过三类 gate。

### 4.1 Efficiency gate

探索门槛：

$$
\frac{T_{forward,KAN}}{T_{forward,MLP}} \leq 1.75,
$$

$$
\frac{T_{backward,KAN}}{T_{backward,MLP}} \leq 1.75,
$$

$$
\frac{M_{backward,KAN}}{M_{backward,MLP}} \leq 1.25,
$$

$$
\frac{T_{step,KAN}}{T_{step,MLP}} \leq 1.75.
$$

最终门槛：

$$
\frac{T_{forward,KAN}}{T_{forward,MLP}} \leq 1.25,
$$

$$
\frac{T_{backward,KAN}}{T_{backward,MLP}} \leq 1.40,
$$

$$
\frac{M_{backward,KAN}}{M_{backward,MLP}} \leq 0.80,
$$

$$
\frac{T_{step,KAN}}{T_{step,MLP}} \leq 1.30.
$$

最终目标要求 backward memory 低于 MLP，所以 $0.80$ 是 final target；探索阶段可以先允许 $1.25$。

### 4.2 Task gate

候选至少要满足：

$$
Acc_{KAN} \geq Acc_{MLP} - 0.5\%.
$$

并且在 KMNIST 上不能低于：

$$
Acc_{KAN} \geq Acc_{MLP} - 1.0\%.
$$

如果一个 candidate 只在 MNIST/Fashion 接近 MLP，但 KMNIST 大幅掉点，则不能作为 all-dataset survivor。

### 4.3 Geometry / calibration gate

进入 functional / smoothing 阶段前，AdamW recipe candidate 只需要记录 geometry，不需要强制超过 MLP。进入 final candidate 后，必须满足：

$$
ECE_{KAN} \leq ECE_{MLP} + 0.02,
$$

并至少满足一个 geometry 条件：

$$
\phi_{residual,KAN} \leq 0.9 \phi_{RBFOnly},
$$

或：

$$
\operatorname{curv}_{residual,KAN} \leq 0.9 \operatorname{curv}_{RBFOnly}.
$$

如果 candidate 是 RationalKAT，则用：

```text
r_prime_p95
r_double_prime_p95
denominator condition
```

替代 RBF residual geometry。

---

## 5. 实验计划

## P0：core / runner / profiler 一致性硬检查

P0 不是形式检查，而是防止后续结果分裂。每个 primitive 必须输出完整 manifest。

### 运行对象

```text
MLP-reference
RBFOnly-Dense
ABRBF-Dense
DWM-compiled-current
DWM-recipe-candidates
CP-two-stage / CP-compiled
RationalKAT-compiled-current
RationalKAT-recipe-candidates
```

### 必须记录

```text
primitive_class
source_file
is_core_defined
is_runner_defined
edge_param_count
base_param_count
rbf_param_count
mixing_param_count
nonKAN_param_count
functional_param_coverage
edge_param_coverage
mixing_param_coverage
rollback_max_abs_error
uses_einsum
uses_exp
uses_gemm
uses_compile
compile_backend
kernel_count_forward
kernel_count_backward
```

### 判定

P0 通过条件：

```text
nonKAN_param_count = 0
edge_param_coverage = 1.0
rollback_max_abs_error = 0
primitive_class 来自 core，不来自 runner helper
所有参数都属于 audited edge system
```

如果发现 runner-only primitive，必须先合并到 core，再跑 P1。

### 可视化

画一张 `parameter_manifest_stacked_bar`：

```text
x-axis: primitive
y-axis: parameter count
stack: base / RBF / mixing / rational / nonKAN
```

再画一张 `kernel_path_heatmap`：

```text
rows: primitive
cols: einsum / GEMM / exp / compile / custom backward / fused path
```

---

## P1：phase-separated efficiency profiler v2

P1 只测效率，不看 accuracy。

### 测试设置

每个 primitive 在相同 shape 下测：

```text
batch_size in {128, 256, 512}
hidden_dim in {64, 96}
depth in {2, 4}
basis_count or groups in candidate-specific grid
```

MLP-reference 使用相同 hidden_dim/depth。

### 候选

```text
DWM-2-compiled-current
DWM-compiled-noExpCache
DWM-compiled-smallK
DWM-compiled-wideMix
DWM-compiled-channelNorm

RK-3-compiled-current
RK-compiled-groups4
RK-compiled-groups8
RK-compiled-groups16
RK-compiled-linear+silu
RK-compiled-siluOnly

CP-1-two-stage-r8/r16/r32
CP-3-compiled-r8/r16/r32

RBFOnly-Dense
ABRBF-Dense
MLP-reference
```

Dense RBF/ABRBF 只作为 reference，不作为 efficiency survivor。

### 必须记录

```text
forward_time_ms
backward_time_ms
optimizer_time_ms
step_time_ms
forward_memory_mb
backward_peak_allocated_mb
backward_peak_reserved_mb
optimizer_state_mb
activation_saved_bytes
basis_tensor_bytes
workspace_temp_bytes
kernel_count_forward
kernel_count_backward
num_gemm_calls
num_einsum_calls
num_exp_calls
compile_time_sec
steady_state_step_time_ms
```

### 判定

P1 分成 exploratory 和 final 两个 gate。

探索通过：

```text
forward <= 1.75x MLP
backward <= 1.75x MLP
backward_memory <= 1.25x MLP
step <= 1.75x MLP
```

最终通过：

```text
forward <= 1.25x MLP
backward <= 1.40x MLP
backward_memory <= 0.80x MLP
step <= 1.30x MLP
```

P1 只需要产生 exploratory survivor 即可进入 P2/P3/P4 recipe repair。

### 可视化

必须画：

```text
efficiency_radar_by_primitive
forward_backward_memory_pareto
step_time_vs_backward_memory
kernel_count_breakdown
saved_tensor_vs_peak_memory
shape_scaling_curve
```

其中 `saved_tensor_vs_peak_memory` 要专门检查：saved bytes 降了但 peak memory 不降的情况。

---

## P2：DepthwiseMix recipe repair

DWM 是 v5.6 第一优先级，因为它理论上最符合：

$$
O(BdK)+O(Bdh).
$$

P2 目标是把 DWM 的 task accuracy 拉近 MLP，同时不破坏 P1 efficiency。

### 变量

```text
basis_count K in {4, 8, 12, 16}
hidden_dim in {64, 96, 128}
mixing_type in {linear, linear_silu, gated_linear}
channel_function in {rbf, abrbf_linear, abrbf_silu, abrbf_linear_silu}
normalization in {fixednorm, channel_scale, pre_mixing_norm}
init_scale in {small, mlp_matched, residual_matched}
lr in {5e-4, 1e-3, 2e-3}
weight_decay in {0, 1e-4}
```

### 运行方式

先跑：

```text
seeds = 0,1,2
short budget = 8 epochs
```

只保留同时满足：

```text
accuracy gap <= 2% on all datasets
step <= 1.75x MLP
backward memory <= 1.25x MLP
```

的候选进入 full-budget。

### 必须记录

```text
test_acc
val_loss_auc
train_loss_auc
ECE
NLL
margin_mean
margin_p10
classwise_acc
input_feature_rank
block_feature_rank
output_logit_norm
channel_function_norm
mixing_weight_norm
mixing_condition
base_over_residual_norm
basis_occupancy_entropy
out_of_grid_fraction
forward/backward/step ratios
backward_memory_ratio
```

### 重点诊断

如果 DWM accuracy 低，必须判断是哪种失败：

```text
channel_function_underfit:
  channel_function_norm 很小，basis occupancy 低

mixing_underfit:
  channel function 有输出，但 mixing weight rank 低，class margin 弱

normalization_failure:
  feature rank 崩，pre/post norm 分布漂移

recipe_failure:
  train acc 低但 capacity 指标正常
```

### 可视化

```text
DWM_accuracy_vs_efficiency
DWM_channel_norm_vs_margin
DWM_mixing_rank_curve
DWM_basis_occupancy_heatmap
DWM_classwise_error_heatmap
```

---

## P3：RationalKAT recipe repair

RationalKAT 是最接近最终高效 primitive 的路线，但 v5.5 中 task quality 很弱。P3 的目标是先让 RationalKAT-AB 在 AdamW 下接近 MLP，再考虑 functional update。

### 变量

```text
groups in {4, 8, 16}
rational_order in {(5,4), (4,3), (3,2)}
base_path in {linear, silu, linear_silu}
init in {kat_default, identity_like, small_residual}
lr in {5e-4, 1e-3, 2e-3, 3e-3}
betas in {(0.9,0.999), (0.95,0.98)}
weight_decay in {0, 1e-4}
compile in {eager, compiled}
```

### 必须记录

```text
test_acc
val_loss_auc
ECE
NLL
margin_mean
margin_p10
rational_denominator_min
rational_denominator_p01
rational_denominator_condition
r_prime_p95
r_double_prime_p95
rational_group_function_diversity
base_over_rational_norm
mixing_weight_norm
forward/backward/step ratios
backward_memory_ratio
```

### 判定

RationalKAT 进入 P5 的条件：

```text
acc >= MLP - 1% on MNIST/Fashion
acc >= MLP - 1.5% on KMNIST
ECE <= MLP + 0.05
denominator_condition stable
step <= 1.75x MLP
backward_memory <= 1.25x MLP
```

### 可视化

```text
Rational_accuracy_vs_groups
Rational_denominator_safety_curve
Rational_derivative_distribution
Rational_group_diversity_heatmap
Rational_efficiency_accuracy_pareto
```

---

## P4：CP-ABRBF compute and capacity repair

CP 是 accuracy backup。它不是首选最终 primitive，但能告诉我们 efficient primitive 需要多少 expressivity。

### 变量

```text
rank in {4, 8, 16, 32}
implementation in {two_stage_gemm, compiled}
basis_count in {8, 12, 16}
base_path in {linear, silu, linear_silu}
hidden_dim in {64, 96}
```

### 必须记录

```text
rank
actual_compute_rank
rank_speed_monotonicity
forward/backward/step ratios
backward_memory_ratio
test_acc
val_loss_auc
ECE
base_over_rbf
high_mode_energy
lowrank_factor_norms
factor_condition
```

### 判定

CP 只有在下面条件满足时进入 P5：

```text
rank-speed monotonicity = true
acc >= MLP - 1% on all datasets
step <= 1.75x MLP
backward_memory <= 1.25x MLP
```

如果 rank 越小不越快，说明实现仍没有利用低秩结构，暂停 CP。

### 可视化

```text
CP_rank_vs_speed
CP_rank_vs_accuracy
CP_factor_norm_stacked
CP_memory_breakdown
```

---

## P5：joint task-efficiency selection

P5 把 P2/P3/P4 的候选放到统一门槛下比较。

### 方法集合

```text
MLP-AdamW-reference
RBFOnly-Dense-AdamW-reference
ABRBF-Dense-AdamW-reference
DWM-best candidates
RationalKAT-best candidates
CP-best candidates
```

### 运行设置

```text
seeds = 0,1,2
epochs = full compact budget
MNIST / Fashion-MNIST / KMNIST
```

### 必须记录

```text
test_acc
train_acc
val_loss_auc
train_loss_auc
ECE
NLL
margin_mean
margin_p10
classwise_acc
feature_rank_input
feature_rank_block
feature_rank_output
geometry metrics
forward_time_ratio
backward_time_ratio
step_time_ratio
backward_memory_ratio
activation_saved_bytes
optimizer_state_bytes
```

### P5 survivor 条件

候选必须满足：

```text
acc >= MLP - 0.5% on MNIST/Fashion
acc >= MLP - 1.0% on KMNIST
ECE <= MLP + 0.03
step <= 1.75x MLP
backward_memory <= 1.25x MLP
nonKAN = 0
edge coverage = 1.0
```

如果 P5 没有 survivor，停止 functional optimizer，不跑 P6-P10。

### 可视化

```text
joint_accuracy_efficiency_pareto
accuracy_gap_vs_step_ratio
accuracy_gap_vs_backward_memory
classwise_accuracy_difference
feature_rank_vs_accuracy
ECE_vs_accuracy
```

---

## P6：custom backward and memory reduction for survivors only

P6 只对 P5 survivors 做。不要再对所有 primitive 大扫 custom backward。

### 目标

把 backward memory 从 exploratory gate 推到 final target：

$$
M_{backward,KAN} \leq 0.8 M_{backward,MLP}.
$$

### 变体

```text
autograd baseline
compiled autograd
recompute backward
streaming stats backward
fused backward, if available
checkpointed forward
```

### 必须记录

```text
grad_rel_error
grad_cosine
param_update_rel_error
saved_tensor_bytes
activation_saved_bytes
workspace_temp_bytes
peak_allocated_mb
peak_reserved_mb
backward_time_ms
step_time_ms
kernel_count_backward
```

### 判定

```text
grad_rel_error <= 1e-5
grad_cosine >= 0.9999
backward_memory <= 0.8x MLP, final target
or <= 1.0x MLP, exploratory target
backward_time <= 1.4x MLP
```

### 可视化

```text
memory_component_breakdown
saved_bytes_vs_peak_allocated
backward_time_vs_memory
custom_backward_error_plot
```

---

## P7：functional / LightSmooth compatibility smoke

P7 只有在 P5/P6 有 survivor 时运行。

### 目标

确认 efficient primitive 是否还能支持 functional geometry maintenance。

### 方法

```text
best efficient primitive AdamW
best efficient primitive + LightSmooth single event
best efficient primitive + one-cycle smooth-refresh
```

### 必须记录

```text
acc_drop
KL
logit_drift
geometry_reduction
ECE_change
smoothing_time_sec
smoothing_memory_ratio
amortized_step_overhead
```

### 判定

```text
acc_drop <= 0.005
KL <= 0.005
logit_drift <= 0.03
geometry reduction > 0
amortized overhead <= 1.10x
```

如果 LightSmooth 不兼容，但 primitive 本身满足 efficiency/task gate，仍保留 primitive；functional smoothing 降级为后续研究。

---

## P8：functional training smoke

P8 不再测试旧 U-FULL / TFU / FNG 大扫，只测试最小可行 functional route。

候选：

```text
Functional-coordinate Adam on efficient primitive
Functional-coordinate Adam + phased geometry
LightSmooth maintenance only
```

目标是判断：

```text
能否在不牺牲效率的前提下改善 AUC/ECE/geometry。
```

记录：

```text
val_loss_auc
time_to_relaxed_loss
step_time_ratio
backward_memory_ratio
ECE
geometry
rank/margin
functional_state_memory
```

判定：

```text
AUC improves over AdamW primitive
ECE improves
geometry improves
step_time_ratio increase <= 1.10
backward_memory_ratio increase <= 1.05
```

---

## P9：5-seed confirm

只有 P5/P6/P7/P8 全部产生 survivor 时才运行。

运行：

```text
seeds = 0..4
methods:
  MLP-AdamW
  best efficient PureKAN AdamW
  best efficient PureKAN functional / LightSmooth candidate
  dense ABRBF reference optional
```

必须记录 paired delta：

```text
acc delta vs MLP
AUC delta vs MLP
ECE delta vs MLP
step ratio vs MLP
backward memory ratio vs MLP
geometry delta vs dense ABRBF / RBFOnly
```

成功条件：

```text
acc >= MLP on at least 2/3 datasets
acc >= MLP - 0.5% on remaining dataset
AUC better than MLP on at least 2/3 datasets
ECE better than MLP on at least 2/3 datasets
backward memory <= MLP, exploratory
step <= 1.3x MLP
```

---

## P10：10-seed final confirm

P10 是最终确认，不是调参。

运行：

```text
seeds = 0..9
only final selected candidate
```

最终成功标准：

```text
nonKAN = 0
forward <= 1.25x MLP
backward <= 1.40x MLP
backward memory <= 0.80x MLP
step <= 1.30x MLP
accuracy >= MLP-AdamW
AUC >= MLP-AdamW
ECE <= MLP-AdamW
geometry better than MLP / PureKAN-AdamW reference
```

如果 P10 通过，才可以 claim：

$$
\boxed{\text{Efficient PureKAN functional training system achieved.}}
$$

如果 P10 不通过，只 claim：

```text
Efficient PureKAN primitive candidate identified, but full functional training not solved.
```

---

## 6. 统一记录字段

所有阶段必须统一输出以下字段。

### 6.1 task metrics

```text
dataset
method
seed
epochs
train_size
hidden_dim
depth
primitive_type
test_acc
train_acc
val_loss
val_loss_auc
train_loss_auc
NLL
ECE
margin_mean
margin_p10
classwise_acc_0...9
```

### 6.2 efficiency metrics

```text
forward_time_ms
backward_time_ms
optimizer_time_ms
step_time_ms
forward_time_ratio_vs_mlp
backward_time_ratio_vs_mlp
step_time_ratio_vs_mlp
forward_memory_mb
backward_peak_allocated_mb
backward_peak_reserved_mb
backward_memory_ratio_vs_mlp
activation_saved_bytes
basis_tensor_bytes
workspace_temp_bytes
optimizer_state_bytes
kernel_count_forward
kernel_count_backward
num_gemm_calls
num_einsum_calls
num_exp_calls
```

### 6.3 PureKAN / edge metrics

```text
nonKAN_param_count
edge_param_count
base_param_count
rbf_param_count
mixing_param_count
edge_coverage
base_coverage
rbf_coverage
mixing_coverage
base_over_rbf_norm
base_ablation_drop
rbf_ablation_drop
mixing_weight_norm
mixing_rank
```

### 6.4 geometry metrics

```text
phi_total_p95
phi_residual_p95
curvature_total
curvature_residual
sobolev_residual_norm
jacobian_condition
basis_occupancy_entropy
out_of_grid_fraction
high_sobolev_eigen_energy
rational_r_prime_p95
rational_r_double_prime_p95
rational_denominator_condition
```

### 6.5 functional / smoothing metrics

```text
functional_method
functional_state_memory_mb
cos_function_with_adam
function_R2_with_adam
LightSmooth_acc_drop
LightSmooth_KL
LightSmooth_logit_drift
LightSmooth_geometry_reduction
LightSmooth_time_sec
LightSmooth_memory_ratio
amortized_overhead
```

---

## 7. 必须生成的可视化

### 7.1 efficiency dashboard

图包括：

```text
forward ratio by primitive
backward ratio by primitive
step ratio by primitive
backward memory ratio by primitive
```

这张图直接回答候选是否接近 MLP。

### 7.2 task-efficiency Pareto

横轴：

$$
\text{step time ratio vs MLP}
$$

纵轴：

$$
\text{accuracy gap vs MLP}
$$

点大小：

$$
\text{backward memory ratio}
$$

颜色：

```text
primitive type
```

### 7.3 memory decomposition plot

stacked bar：

```text
activation saved bytes
basis tensor bytes
workspace temp bytes
optimizer state bytes
other allocated memory
```

### 7.4 recipe repair heatmaps

DWM：

```text
basis_count x hidden_dim -> accuracy gap
basis_count x hidden_dim -> step ratio
mixing_type x lr -> accuracy gap
```

RationalKAT：

```text
groups x lr -> accuracy
order x groups -> denominator condition
base_path x init -> ECE
```

CP：

```text
rank x basis_count -> accuracy
rank x basis_count -> step time
rank x implementation -> memory
```

### 7.5 classwise failure heatmap

每个 candidate 对每个 dataset 画：

```text
class index vs accuracy delta relative to MLP
```

这能判断 KMNIST 是整体 underfit 还是特定类别崩。

### 7.6 geometry-efficiency Pareto

横轴：

$$
\text{backward memory ratio}
$$

纵轴：

$$
\text{residual geometry reduction}
$$

点大小：

$$
\text{accuracy gap}
$$

这张图用来判断 LightSmooth / functional update 是否值得引入。

---

## 8. 失败后的决策规则

### 情况 A：DWM 通过 task + efficiency

如果 DWM 通过 P5，并且 P6 custom backward 能进一步降低 memory，则 v5.6 主线转为：

```text
DWM PureKAN as efficient primitive
LightSmooth optional
functional optimizer optional
```

后续进入 5-seed / CIFAR-small。

### 情况 B：RationalKAT 通过 task + efficiency

如果 RationalKAT 通过 P5，则优先级最高，因为它最接近 MLP-like primitive。下一步才重新设计 Rational-specific functional metric。

### 情况 C：CP 通过 task 但 efficiency 不够

CP 作为 accuracy reference 保留，但不进入 final system。继续寻找更 GEMM-native primitive。

### 情况 D：没有 primitive 通过 P5

如果 P5 没有 survivor，停止 optimizer 搜索，转向新 primitive 设计：

```text
piecewise-linear edge + GEMM
low-degree polynomial edge + GEMM
rational activation + structured mixing
spline-depthwise + GEMM
```

不要继续在 Dense RBF / Dense AB-RBF 上投入 functional optimizer。

---

## 9. v5.6 的最终输出

v5.6 最终必须输出：

```text
1. primitive_efficiency_report.md
2. task_efficiency_pareto.csv
3. memory_breakdown.csv
4. recipe_repair_summary.csv
5. candidate_selection.json
6. failure_taxonomy.csv
7. next_route_decision.json
```

`next_route_decision.json` 只能有以下几种结论：

```text
DWM_mainline
RationalKAT_mainline
CP_accuracy_reference_only
No_efficient_primitive_yet
Proceed_to_functional_training
```

---

## 10. 本轮最重要的判断

v5.6 的任务不是证明 functional update 终于成功，而是先证明：

$$
\boxed{\text{存在一个 strict PureKAN edge primitive，能以接近 MLP 的效率学到接近 MLP 的任务性能。}}
$$

如果这件事做不到，functional update 不可能满足终极目标。

如果这件事做到，才进入下一阶段：

$$
\boxed{\text{在高效 primitive 上引入 functional training / LightSmooth / analytic adjoint。}}
$$

所以 v5.6 的核心路线是：

```text
Kernel first.
Recipe second.
Functional optimizer third.
```
