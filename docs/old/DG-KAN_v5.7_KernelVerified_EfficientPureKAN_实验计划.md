# DG-KAN v5.7 Kernel-Verified Efficient PureKAN 实验计划

## 0. 终极目标与本轮定位

本项目的终极目标固定为：

$$
\boxed{
\text{构建一个无 non-KAN 参数的 PureKAN functional training system，}
}
$$

$$
\boxed{
\text{其 forward 时间/显存接近 MLP，backward 时间接近 MLP，backward 显存低于 MLP，收敛速度快，}
}
$$

$$
\boxed{
\text{并在 accuracy / AUC / ECE / geometry 上超过 MLP-AdamW 与 PureKAN-AdamW。}
}
$$

v5.6 的结果说明，当前还不能继续推进 functional optimizer / LightSmooth 主线。原因不是 functional update 没意义，而是目前还没有一个同时满足 **task + efficiency** 的 PureKAN primitive。v5.7 因此改成 **Kernel-Verified Efficient PureKAN**：先把高效 primitive 的 kernel path、profiler、accuracy recipe 三者打通，再重新接回 functional training。

本轮不追求再发明一个新的 $S^{-1}g$ 或新的 smoothing controller。v5.7 的核心问题是：

$$
\boxed{
\text{有没有一个 strict PureKAN edge primitive，既能接近 MLP 的计算/显存，又能接近或超过 MLP 的任务表现？}
}
$$

只有这个问题得到肯定答案，后续 functional update 才值得继续投入。

---

## 1. v5.6 结果复盘与诊断

### 1.1 已经确认的正面进展

v5.6 的 P0 说明 core-level primitive 已经进入主代码路径，并且不再是 runner-only helper。参与检查的 primitive 包括：

```text
RBFOnly-Dense
ABRBF-Dense
GEMMNativeDepthwiseMixDense
GEMMNativeCPABRBFDense
GEMMNativeRationalKATDense
```

P0 中所有非参考 PureKAN primitive 都满足：

```text
core = 1
nonKAN = 0
rollback = 0
edge parameters are covered
```

这说明当前失败不应首先归因于 hidden MLP stem/head、learnable LayerNorm、rollback、coverage 等老问题。

### 1.2 当前最重要的失败

v5.6 的最终状态是：

```text
stop_after_p4_no_recipe_survivor
```

P5 没有 all-dataset survivor，P6-P10 都没有继续运行。失败诊断中主要失败类型是：

```text
F2_memory_fail = 82
F3_accuracy_recipe_unproven = 61
F1_efficiency_fail = 33
F4_joint_gate_failed = 20
```

这意味着问题不是单点失败，而是 **效率、显存、任务 recipe 和 joint gate 同时没有闭环**。

### 1.3 三类 primitive 的现状

#### Dense RBF / Dense AB-RBF

Dense RBF / Dense AB-RBF 仍然是机制参考，不是最终高效路线。它的 forward 需要显式或隐式处理 basis 维度：

$$
B_{bik}=\exp\left(-\frac{(x_{bi}-c_k)^2}{2\sigma^2}\right),
$$

$$
y_{bo}=\sum_{i,k}B_{bik}c_{oik}.
$$

复杂度是：

$$
O(Bd_{in}d_{out}K),
$$

而 MLP 是：

$$
O(Bd_{in}d_{out}).
$$

所以 dense RBF / dense AB-RBF 很难满足 MLP-like forward。它们应该继续作为：

```text
mechanism reference
accuracy reference
geometry reference
```

但不应作为最终 efficient system。

#### DepthwiseMix

DepthwiseMix 理论上应该是最接近 MLP 结构的 RBF-like primitive：

$$
\tilde x_i=f_i(x_i),
$$

$$
y=W\tilde x.
$$

理论复杂度应该是：

$$
O(BdK)+O(Bdh),
$$

其中第二项是 GEMM。它应该比 full RBFDense 的 $O(Bd_{in}d_{out}K)$ 更接近 MLP。

但 v5.6 出现了非常异常的 profiler 信号：`DWM-compiled-current` 的 forward / backward time 远大于 MLP，P2 recipe 中 step ratio 也很大。这说明至少有一种可能：

```text
compile overhead 被计入 steady-state；
或 torch.compile 发生动态 shape / graph break / repeated recompile；
或 DWM 的实现没有真正落到 GEMM-native path；
或 profiler 没有分开 cold compile 与 warm steady-state。
```

因此 v5.7 必须先做 profiler correctness audit，而不能直接用 v5.6 的 DWM time 断定 DepthwiseMix 不可行。

#### RationalKAT

RationalKAT 的理论优势是：

$$
\tilde x_i=r_g(x_i),
$$

$$
y=W\tilde x.
$$

主计算仍然是 GEMM，参数量也接近 MLP：

$$
O(dh)+O(g(m+n)).
$$

v5.6 中 RationalKAT 的 backward memory 比较有潜力，但 forward / backward / step time 和 accuracy recipe 没有过 gate。当前结论不是 RationalKAT 失败，而是：

$$
\boxed{
\text{RationalKAT 是最有希望的 MLP-like primitive 之一，但 recipe 和 kernel path 尚未调通。}
}
$$

#### CP-ABRBF

CP-ABRBF 目前是 task signal 最好的 efficient candidate 之一，但仍然 memory / time 不够。它可以作为 accuracy reference，但不能作为最终效率主线，除非 rank-speed monotonicity 和 memory gate 同时修好。

---

## 2. v5.7 总原则

v5.7 的原则是：

```text
1. Profiler correctness first.
2. Kernel path second.
3. Task recipe third.
4. Functional optimizer last.
```

也就是：

$$
\boxed{
\text{没有 efficiency survivor，就不跑 functional optimizer。}
}
$$

本轮所有方法必须同时接受三类 gate：

### 2.1 Strict PureKAN gate

候选必须满足：

```text
nonKAN_params = 0
edge_coverage = 1.0
rollback_error = 0
all trainable params belong to audited edge system
```

如果候选需要 normalization / mixing / gain 参数，那么这些参数必须明确属于 edge system，而不能成为普通 non-KAN 参数。

### 2.2 Efficiency gate

最终目标 gate：

$$
\frac{T_{fwd,KAN}}{T_{fwd,MLP}}\leq 1.25,
$$

$$
\frac{M_{fwd,KAN}}{M_{fwd,MLP}}\leq 1.25,
$$

$$
\frac{T_{bwd,KAN}}{T_{bwd,MLP}}\leq 1.40,
$$

$$
\frac{M_{bwd,KAN}}{M_{bwd,MLP}}\leq 0.80.
$$

探索阶段可以允许：

$$
\frac{T_{step,KAN}}{T_{step,MLP}}\leq 2.0,
$$

但只要进入 5-seed / 10-seed confirm，必须接近最终 gate。

### 2.3 Task / geometry gate

候选至少要满足：

```text
accuracy gap vs MLP-AdamW <= 1.0% exploratory
accuracy gap vs MLP-AdamW <= 0.5% confirm
AUC >= MLP-AdamW - small tolerance
ECE <= MLP-AdamW + 0.02
geometry not worse than MLP / RBFOnly reference
```

最终目标要求：

$$
\operatorname{Acc}_{KAN}\geq \operatorname{Acc}_{MLP-AdamW},
$$

$$
\operatorname{ECE}_{KAN}\leq \operatorname{ECE}_{MLP-AdamW},
$$

$$
\operatorname{AUC}_{KAN}\geq \operatorname{AUC}_{MLP-AdamW},
$$

并且 edge geometry 优于 MLP 或至少优于对应 AdamW PureKAN reference。

---

## 3. 代码实现审计与改进方向

### 3.1 Profiler 必须重写为 cold / warm 分离

v5.6 中 `DWM-compiled-current` 和 `RationalKAT-compiled-current` 出现极端时间比。下一轮必须将 profiler 拆成：

```text
compile_time_ms
cold_forward_ms
warm_forward_ms
warm_backward_ms
optimizer_ms
step_ms
```

所有 compiled / torch.compile / Triton 路径都必须：

```text
预热至少 20 steps；
正式计时 100 steps；
每次 timing 前后 torch.cuda.synchronize；
单独记录 graph_break_count / recompile_count；
固定 batch shape；
固定 dtype；
关闭数据加载扰动；
分开 forward-only、backward-only、optimizer-only。
```

如果 `warm_forward_ms` 与 `cold_forward_ms` 差距巨大，则当前 P1/P2 结果不能作为 steady-state efficiency 结论，只能作为 profiler warning。

### 3.2 DWM 的实现应强制 GEMM-native

DepthwiseMix 的正确结构应是：

$$
u = f_{dw}(x),
$$

$$
y = \nu W^T.
$$

其中 $f_{dw}$ 是 channel-wise edge function，$W$ 是 edge-owned mixing matrix。不要形成：

$$
[B,d,h,K]
$$

这样的高维 tensor。推荐实现：

```python
u = depthwise_function(x)       # [B, d]
y = torch.matmul(u, W.T)        # [B, h]
```

DWM 的关键 audit 是：

```text
num_exp_calls
basis_tensor_bytes
num_gemm_calls
num_einsum_calls
kernel_count
```

如果 DWM 里出现 `einsum` 或巨大 basis tensor，就说明没有真正 GEMM-native。

### 3.3 RationalKAT 必须检查 group broadcast 与 backend

RationalKAT 理论上应该是：

```text
elementwise/group rational activation
+ GEMM mixing
```

必须确认没有：

```text
不必要的 unsqueeze / squeeze 造成 kernel explosion；
Python loop over groups；
每 batch 重新创建 KAT module；
torch.compile graph break；
Triton fallback 到 torch eager；
保存过多 denominator intermediate。
```

记录：

```text
actual_backend = torch / triton / compiled
module_creation_count
rational_forward_kernel_count
denominator_tensor_bytes
saved_activation_bytes
compile_graph_count
```

### 3.4 CP-ABRBF 必须强制 rank-speed monotonicity

如果 CP rank 下降不带来速度下降，说明 compute graph 没有真正利用低秩。必须记录：

```text
rank
parameter_count
FLOPs_estimate
actual_step_ms
actual_bwd_mem
rank_speed_monotonic
rank_memory_monotonic
```

如果 `rank4` 不比 `rank8` / `rank16` 更快，CP 只能保留为 accuracy reference。

### 3.5 Custom backward 只对 efficiency survivor 做

v5.4/v5.5 的 custom backward 还没有真正解决 measured peak memory。v5.7 不再对所有 primitive 写 custom backward，只对通过 P4 的 survivor 做。否则会浪费大量工程时间。

---

## 4. 实验阶段设计

## P0：Core / Runner / Profiler 一致性硬检查

### 目标

确认 v5.7 的所有 candidate 是 strict PureKAN，并且 profiler 不再混入 compile overhead 或 graph break。

### 方法

```text
MLP-reference
RBFOnly-Dense-reference
ABRBF-Dense-reference
DWM-current
DWM-warm-compiled
DWM-gemm-native
RationalKAT-current
RationalKAT-compiled
RationalKAT-grouped-gemm
CP-two-stage-r4/r8/r16
```

### 必须记录

```text
nonKAN_params
edge_param_count
mixing_param_count
edge_coverage
mixing_coverage
rollback_error
compile_graph_count
recompile_count
graph_break_count
actual_backend
```

### 通过条件

```text
nonKAN_params = 0
edge_coverage = 1.0
rollback_error = 0
recompile_count = 0 after warmup
```

如果 P0 不通过，不进入 P1。

---

## P1：Phase-separated efficiency profiler v3

### 目标

重新测量真实 steady-state efficiency，尤其排查 v5.6 中 DWM / RationalKAT compiled time 极端异常的原因。

### 计时协议

每个方法执行：

```text
compile warmup steps = 20
measurement steps = 100
batch_size = 256
fixed shape
cuda sync before/after every timing region
repeat = 3
```

分开记录：

```text
forward only
loss + backward
optimizer step
full train step
```

### 记录指标

```text
forward_time_ms_cold
forward_time_ms_warm
backward_time_ms_warm
optimizer_time_ms
step_time_ms
forward_peak_allocated_mb
backward_peak_allocated_mb
optimizer_peak_allocated_mb
reserved_mb
activation_saved_bytes
basis_tensor_bytes
workspace_temp_bytes
parameter_bytes
gradient_bytes
optimizer_state_bytes
kernel_count
GEMM_count
einsum_count
exp_count
compile_time_ms
graph_break_count
recompile_count
```

### 可视化

1. **Cold vs warm timing plot**
   - x-axis: primitive
   - y-axis: time ratio vs MLP
   - bars: cold forward / warm forward / backward / step

2. **Memory decomposition stacked bar**
   - activation saved bytes
   - basis tensor bytes
   - workspace temp bytes
   - params / grads / optimizer states

3. **Kernel path heatmap**
   - methods vs kernel categories
   - cells: GEMM / exp / einsum / custom kernel / graph breaks

### 通过条件

Exploratory gate：

$$
T_{step}/T_{MLP}\leq 2.0,
$$

$$
M_{bwd}/M_{MLP}\leq 1.2.
$$

Final-efficiency candidate gate：

$$
T_{step}/T_{MLP}\leq 1.5,
$$

$$
M_{bwd}/M_{MLP}\leq 1.0.
$$

---

## P2：DepthwiseMix kernel and recipe repair

### 目标

判断 DWM 是真正的效率路线，还是只是 profiler 偶然 pocket。

### 候选

```text
DWM-0 current
DWM-1 no compile, pure eager vectorized
DWM-2 torch.compile warm steady-state
DWM-3 GEMM-native matmul only
DWM-4 fused depthwise activation + GEMM
DWM-5 custom backward recompute, only if P1 warm path passes
```

### recipe sweep

```text
K in {4, 8}
base path in {linear, silu, linear+silu}
mixing width in {1x, 2x}
residual scale init in {0.1, 0.3, 1.0}
normalization in {FixedNorm, edge-owned channel gain}
lr in {1e-3, 2e-3, 3e-3}
weight_decay in {0, 1e-4}
```

### 数据集

```text
MNIST
Fashion-MNIST
KMNIST
```

先跑：

```text
seeds = 0,1,2
short budget = 8 epochs or matched steps
```

### 必须记录

```text
test_acc
val_loss_auc
ECE
NLL
margin_mean
margin_p10
feature_effective_rank
class_centroid_separation
mixing_weight_norm
depthwise_function_norm
base_path_norm
residual_path_norm
step_time_ratio
bwd_memory_ratio
```

### 可视化

1. **DWM recipe heatmap**
   - x-axis: K / base path / mix width
   - y-axis: dataset
   - color: accuracy gap vs MLP

2. **DWM task-efficiency Pareto**
   - x-axis: step time ratio
   - y-axis: accuracy
   - point size: backward memory ratio

3. **DWM representation plot**
   - epoch vs feature rank
   - epoch vs margin p10
   - epoch vs class centroid separation

### 通过条件

```text
accuracy gap vs MLP <= 2% exploratory
step_time_ratio <= 2.0
bwd_memory_ratio <= 1.2
ECE not worse than MLP by more than 0.03
```

如果 DWM 速度过线但 accuracy 不过，继续做 recipe repair；如果 DWM accuracy 也完全不接近，则降级为 kernel reference。

---

## P3：RationalKAT kernel and recipe repair

### 目标

RationalKAT 是理论上最接近 MLP-like 的 primitive。本阶段重点修 recipe，不做 functional update。

### 候选

```text
RK-0 current
RK-1 eager clean
RK-2 compiled warm steady-state
RK-3 grouped rational + GEMM
RK-4 fused rational activation
RK-5 identity-initialized rational residual
```

### recipe sweep

```text
groups in {4, 8, 16, 32}
rational degree in {(5,4), (3,2)}
base path in {linear, silu, linear+silu}
residual init scale in {0.05, 0.1, 0.3}
mixing width in {1x, 2x}
lr in {1e-3, 2e-3, 3e-3, 5e-3}
weight_decay in {0, 1e-4}
```

### 必须记录

```text
test_acc
val_loss_auc
ECE
NLL
rational_denominator_min
rational_denominator_p01
rational_denominator_condition
r_prime_p95
r_double_prime_p95
activation_output_norm
mixing_weight_norm
feature_rank
margin_p10
step_time_ratio
bwd_memory_ratio
backend
kernel_count
```

### 可视化

1. **Rational safety dashboard**
   - denominator p01 over epoch
   - r_prime_p95 over epoch
   - r_double_prime_p95 over epoch

2. **Rational recipe heatmap**
   - groups / residual scale / lr vs accuracy

3. **Rational efficiency Pareto**
   - step ratio vs accuracy
   - color: groups
   - size: memory ratio

### 通过条件

```text
accuracy gap vs MLP <= 2%
step_time_ratio <= 2.0
bwd_memory_ratio <= 1.0 exploratory
no denominator safety failure
```

如果 RationalKAT 速度过线但 accuracy 差，则重点修初始化 / base path / group recipe；不要进入 functional update。

---

## P4：CP-ABRBF compute and accuracy reference

### 目标

CP 不是当前最高优先级效率路线，但它在任务上有信号，适合作为 accuracy reference。

### 候选

```text
CP-rank4
CP-rank8
CP-rank16
CP-rank32 optional
CP-two-stage-gemm
CP-compiled
```

### 必须记录

```text
rank
parameter_count
FLOPs_estimate
actual_forward_time
actual_backward_time
actual_step_time
actual_backward_memory
rank_speed_monotonic
rank_memory_monotonic
accuracy
ECE
AUC
feature_rank
margin_p10
```

### 通过条件

CP 若要保留为 efficient candidate，必须满足：

$$
T_{rank4}<T_{rank8}<T_{rank16}
$$

或至少：

$$
T_{rank4}<T_{rank16}.
$$

如果 rank-speed 不单调，说明计算图没有真正利用低秩，CP 只保留为 task reference。

---

## P5：Joint task-efficiency selection

### 目标

只比较 P2/P3/P4 过 exploratory gate 的候选，不允许把所有方法都带入 confirm。

### 方法池

```text
MLP-AdamW reference
RBFOnly-Dense reference
ABRBF-Dense reference
best DWM candidate
best RationalKAT candidate
best CP candidate
```

### 数据集

```text
MNIST
Fashion-MNIST
KMNIST
```

### 设置

```text
seeds = 0,1,2
train budget = matched epochs + matched steps both recorded
```

### 记录指标

任务：

```text
test_acc
val_loss_auc
train_loss_auc
ECE
NLL
margin_mean
margin_p10
classwise_accuracy
feature_effective_rank
class_centroid_separation
```

效率：

```text
forward_time_ratio
backward_time_ratio
step_time_ratio
forward_memory_ratio
backward_memory_ratio
activation_saved_bytes
kernel_count
```

结构：

```text
edge_param_count
mixing_param_count
base_path_norm
residual_path_norm
rational_safety or rbf_geometry
```

### 可视化

1. **Joint Pareto plot**
   - x-axis: step time ratio
   - y-axis: accuracy
   - point size: backward memory ratio
   - color: primitive

2. **Accuracy-efficiency table by dataset**
   - rows: candidate
   - columns: acc gap / ECE / step / bmem

3. **Failure reason heatmap**
   - datasets vs methods
   - cells: efficiency fail / accuracy fail / memory fail / ECE fail

### survivor 条件

Exploratory survivor：

```text
all datasets accuracy gap <= 2%
step_time_ratio <= 2.0
bwd_memory_ratio <= 1.2
```

Confirm survivor：

```text
all datasets accuracy gap <= 1%
step_time_ratio <= 1.5
bwd_memory_ratio <= 1.0
```

如果 P5 无 survivor，则不跑 P6-P10。

---

## P6：Custom backward / analytic adjoint memory reduction, survivor only

### 目标

只对 P5 survivor 写 custom backward，避免对失败 primitive 浪费工程时间。

### 设计

对于 DWM / RBF-like primitive：

```text
不要保存 basis tensor；
backward recompute depthwise function / basis；
streaming accumulation of coeff gradients；
only save x and compact edge metadata。
```

对于 RationalKAT：

```text
不要保存 full denominator intermediate；
recompute rational activation in backward；
only save x, group index, numerator/denominator params。
```

### 记录指标

```text
grad_rel_error
grad_cosine
forward_memory_before/after
backward_memory_before/after
saved_activation_bytes_before/after
backward_time_before/after
step_time_before/after
```

### 通过条件

```text
grad_rel_error <= 1e-4
grad_cosine >= 0.999
backward_memory_ratio <= 0.8 target, <=1.0 exploratory
backward_time_ratio <= 1.5
```

---

## P7：Functional / LightSmooth compatibility smoke

### 目标

只有 P5/P6 有 survivor 后，才检查 functional training 或 LightSmooth 是否兼容该 primitive。

### 方法

```text
survivor-AdamW
survivor-functional-coordinate Adam
survivor-LightSmooth-one-event
survivor-LightSmooth-one-cycle
```

### 记录指标

```text
accuracy
AUC
ECE
geometry metric appropriate to primitive
step_time_ratio
memory_ratio
smoothing_overhead
function_drift
logit_KL
```

### 通过条件

Functional / LightSmooth 不能破坏 P5 的 task-efficiency survivor：

```text
accuracy drop <= 0.5%
step overhead <= 10% amortized
memory overhead <= 10%
geometry improves by meaningful amount
```

---

## P8：3-seed candidate selection

### 目标

将 P5/P6/P7 survivor 做 3-seed full-budget 选择。

### 候选上限

最多：

```text
2 primitive candidates
1 functional/LightSmooth variant each
MLP reference
```

不要再扩大。

### 记录指标

```text
all task metrics
all efficiency metrics
all geometry metrics
paired delta vs MLP
paired delta vs PureKAN-AdamW dense reference
```

### 通过条件

```text
acc >= MLP-AdamW - 0.5%
AUC >= MLP-AdamW
ECE <= MLP-AdamW + 0.02
step_time_ratio <= 1.5
bwd_memory_ratio <= 1.0
```

---

## P9：5-seed confirm

### 目标

对 P8 winner 做正式 5-seed confirm。

### 通过条件

$$
\Delta Acc_{candidate-MLP}\geq 0
$$

或至少 paired CI 不显著低于 MLP，同时：

$$
\frac{T_{step}}{T_{MLP}}\leq 1.5,
$$

$$
\frac{M_{bwd}}{M_{MLP}}\leq 1.0.
$$

如果 P9 过，再进入 P10。

---

## P10：10-seed final confirm

### 目标

验证终极目标的候选系统是否成立。

### 必须报告

```text
accuracy mean/std
paired acc delta vs MLP-AdamW
paired acc delta vs PureKAN-AdamW
AUC delta
ECE delta
geometry delta
forward time ratio
backward time ratio
step time ratio
backward memory ratio
wall-clock time-to-target
```

最终 claim 只有在以下条件同时成立时允许：

$$
\operatorname{Acc}_{candidate}\geq\operatorname{Acc}_{MLP-AdamW},
$$

$$
\operatorname{AUC}_{candidate}\geq\operatorname{AUC}_{MLP-AdamW},
$$

$$
\operatorname{ECE}_{candidate}\leq\operatorname{ECE}_{MLP-AdamW},
$$

$$
\frac{T_{fwd}}{T_{MLP}}\leq 1.25,
$$

$$
\frac{T_{bwd}}{T_{MLP}}\leq 1.40,
$$

$$
\frac{M_{bwd}}{M_{MLP}}\leq 0.80.
$$

如果 accuracy 过但 memory 不过，不能 claim 终极目标，只能 claim architecture-positive。

---

## 5. 本轮必须新增的 artifact

v5.7 必须输出：

```text
p0_core_profiler_consistency.csv
p1_phase_efficiency_v3.csv
p1_cold_warm_compile_audit.csv
p1_memory_decomposition.csv
p2_dwm_kernel_recipe.csv
p2_dwm_recipe_heatmap.csv
p3_rational_kernel_recipe.csv
p3_rational_safety.csv
p4_cp_rank_speed.csv
p5_joint_task_efficiency_selection.csv
p6_custom_backward_memory.csv
p7_functional_lightsmooth_compat.csv
p8_candidate_selection3.csv
p9_confirm5.csv
p10_confirm10.csv
failure_table.csv
failure_by_method.csv
failure_by_type.csv
route_decision.json
aggregate_decision.json
figures/
```

---

## 6. Failure taxonomy

v5.7 使用以下失败标签：

```text
F0_core_invariant_fail
F1_profiler_cold_warm_confounded
F2_graph_break_or_recompile
F3_forward_time_fail
F4_backward_time_fail
F5_backward_memory_fail
F6_accuracy_recipe_fail
F7_ece_or_calibration_fail
F8_rank_margin_representation_fail
F9_custom_backward_incorrect
F10_functional_or_lightsmooth_breaks_efficiency
F11_no_joint_survivor
```

每个失败必须写明：

```text
method
dataset
stage
primary_failure
secondary_failure
numeric_reason
```

---

## 7. 最终决策规则

### 情况 A：DWM 或 RationalKAT 通过 P5

继续 P6 custom backward，再 P7 functional compatibility。

这代表：

$$
\boxed{
\text{找到 MLP-like PureKAN primitive 的候选。}
}
$$

### 情况 B：只有 CP 通过 task，但不通过 efficiency

CP 保留为 accuracy reference，不进入终极系统主线。下一步继续修 DWM/RationalKAT 或重新设计 primitive。

### 情况 C：没有任何 primitive 通过 P5

停止 functional optimizer；下一步转向新的 primitive 设计，例如：

```text
edge-owned low-rank MLP-like basis
rational-spline hybrid
piecewise-linear KAN edge with fused GEMM
learned lookup-table / spline LUT kernel
```

### 情况 D：primitive 通过 P5，但 custom backward 不能降显存

不能 claim backward memory goal。继续 custom kernel / recompute / checkpoint design，不进入 final confirm。

---

## 8. 本轮最重要的判断

v5.7 的核心不是证明某个 optimizer 更好，而是确认：

$$
\boxed{
\text{是否存在一个 strict PureKAN primitive，真正具备 MLP-like kernel path 和可接受 accuracy。}
}
$$

如果不存在，终极目标暂时不能通过 functional update 解决。因为再好的 optimizer 也无法让一个 forward/backward 天然过重的 primitive 满足：

```text
forward close to MLP
backward close to MLP
backward memory lower than MLP
```

所以 v5.7 的总原则是：

$$
\boxed{
\text{primitive efficiency 是 functional optimizer 的前置条件。}
}
$$
