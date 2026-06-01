# DG-KAN v6.3：Graph-Free Fused Kernel 与收敛加速联合实验计划

> 本计划基于 v6.2 结果重新制定。v6.2 已经证明 analytic adjoint / graph-free 方向是正确的，但也证明当前 manual primitive 还没有达到终极系统所需的 kernel/cache 效率，也没有用更快收敛补偿 step-time 开销。因此 v6.3 不再继续泛泛地调 functional update，而是把问题拆成三层同时验证：graph-free kernel、primitive 表达/recipe、以及收敛加速。

---

## 0. 终极目标与当前差距

我们的终极目标固定为：

$$
\boxed{
\text{构建一个无 non-KAN 参数的 PureKAN functional training system，}
}
$$

并且该系统必须满足：

$$
\boxed{
\text{forward 时间/显存接近 MLP，backward 时间接近 MLP，backward 显存低于 MLP，收敛速度快，}
}
$$

同时在以下指标上超过 `MLP-AdamW` 与 `PureKAN-AdamW`：

$$
\boxed{
\text{accuracy / validation-loss AUC / ECE / geometry。}
}
$$

v6.1 和 v6.2 已经把一个核心误区拆开：

```text
functional update rule != graph-free training system
```

v6.1 证明 manual analytic adjoint 的梯度是正确的，但没有 primitive 通过 graph-free efficiency gate。v6.2 进一步拆了 cache、kernel/op count，并允许 near-miss candidate 做短程收敛补偿测试。结果说明：

```text
1. graph-free analytic adjoint 已经不是主要瓶颈；
2. 当前 blocker 是 manual primitive 的 kernel/cache 效率和 task recipe；
3. near-miss candidate 尚未靠更快收敛补偿 step-time 开销；
4. functional optimizer / LightSmooth 仍应继续 gated，不能在 primitive 未达标前大规模展开。
```

所以 v6.3 的问题不是：

```text
functional update 公式还要怎么调？
```

而是：

$$
\boxed{
\text{如何把 graph-free analytic adjoint 写成真正 MLP-like 的 primitive，并让它在 wall-clock 上收敛更快？}
}
$$

---

## 1. v6.2 结果复盘与关键判断

### 1.1 已经站住的部分

v6.2 的 P0/P1 是强正信号：manual primitive 不依赖 `loss.backward()`，cache manifest 有记录，gradient correctness 全部通过。也就是说，我们已经有资格说：

$$
\boxed{
\text{graph-free analytic-adjoint 路线实现上是可行的。}
}
$$

这和之前 PyTorch autograd prototype 的状态完全不同。现在每个候选都必须明确记录：

```text
uses_loss_backward = 0
uses_torch_autograd_graph = 0
manual_forward_available = 1
manual_backward_available = 1
manual_update_available = 1
```

### 1.2 P2 efficiency 的真实含义

v6.2 P2 中，最接近终极目标的候选是 `DWM2-lite-poly2` 和 `DWM2-lite-RBFK2-cacheMin`。关键数字是：

| primitive | forward ratio | backward ratio | step ratio | backward memory ratio | cache ratio |
|---|---:|---:|---:|---:|---:|
| MLP-manual-linear-reference | 0.5698 | 0.4572 | 0.4455 | 1.0071 | 1.0755 |
| DWM2-lite-poly2 | 1.6616 | 1.6062 | 1.2415 | 1.1731 | 1.0755 |
| DWM2-lite-RBFK2-cacheMin | 2.0948 | 1.8540 | 1.4367 | 1.4954 | 1.0755 |
| DWM2-lite-fastRational | 2.3495 | 2.4963 | 1.8142 | 1.5795 | 1.0755 |
| SparseInterpKAN-K8-fusedIndex | 6.2795 | 6.2028 | 4.3895 | 1.6454 | 1.0755 |

这组结果有三层含义。

第一，`DWM2-lite-poly2` 是目前最接近 efficiency envelope 的候选。它的 step ratio 已经是 `1.2415`，但 backward memory ratio 仍然是 `1.1731`，没有达到最终目标中的 backward memory below MLP。

第二，`MLP-manual-linear-reference` 明显比 MLP autograd reference 更快。它的 step ratio 是 `0.4455`，说明手写 graph-free runtime 本身可以很快。这也提醒我们：后续必须同时使用两个 MLP baseline。

```text
MLP-autograd-reference:
  当前 PyTorch 实用 baseline。

MLP-manual-linear-reference:
  graph-free lower-bound baseline。
```

如果只对比 autograd MLP，DWM2-poly2 接近；如果对比 manual MLP，DWM2-poly2 仍然偏慢。终极目标应至少超过 autograd MLP，并逐步逼近 manual MLP。

第三，SparseInterpKAN 理论上应该 sparse，但当前 `vectorized / fusedIndex` 仍然是 `6x` 左右 forward/backward。这说明它不是理论路线失败，而是实现路径没有真正 fused，可能存在过多 gather/scatter、小 kernel、Python-loop 或非连续 memory access。

### 1.3 P4 near-miss convergence 的真实含义

v6.2 允许 near-miss candidate 进入 P4 wall-clock 收敛补偿测试，这是对的。结果显示：

```text
Fashion:
  DWM2-lite-poly2 可以过 P4 单数据集 gate，step ratio = 1.2415，AUC-time 轻微正。

KMNIST:
  DWM2-lite-poly2 / RBFK2-cacheMin 都没有过，accuracy gap 和 AUC-time 都不足。

MNIST:
  near-miss 仍未形成 all-dataset survivor。
```

因此，v6.2 的结论不是“路线错了”，而是：

$$
\boxed{
\text{graph-free route 正确，但当前 DWM2-lite 系列只在 Fashion 有局部收敛补偿，不能跨数据集成立。}
}
$$

P5-P9 被 gate 掉是合理的，因为 P4 没有 all-dataset survivor。但下一轮不应该简单重复 P4，而应该扩大假设空间，系统验证为什么 near-miss 不能变成 winner。

---

## 2. 当前核心缺陷：不是公式，而是系统闭环

### 2.1 缺陷一：primitive 的计算路径还不是 kernel-native

虽然我们已经不用 PyTorch backward 图，但当前 manual primitive 仍然可能由许多 PyTorch tensor op 组成，例如：

```text
elementwise transform
small gather/scatter
small matmul
manual loop
临时 tensor 分配
```

这种实现是 graph-free，但不是 kernel-native。graph-free 只解决 autograd saved tensor 问题，不自动保证：

$$
T_{forward}\approx T_{MLP},\quad T_{backward}\approx T_{MLP}.
$$

所以 v6.3 必须把 profiler 从“总时间”拆到 component-level：

```text
transform time
mixing GEMM time
manual adjoint time
parameter update time
kernel launch count
temporary allocation bytes
```

### 2.2 缺陷二：DWM2-poly2 效率好，但表达可能不够

`DWM2-lite-poly2` 是 P2 最接近效率门槛的候选，但 P4 只在 Fashion 过，KMNIST 不过。这说明 polynomial residual 的计算轻，但可能不足以表达 KMNIST 所需的局部非线性。

因此，v6.3 不能只优化 `poly2`，而应验证：

```text
poly2 是否是 under-expressive？
RBFK2 是否是 expressive but too slow？
fastRational 是否能在 expressivity 与 compute 之间折中？
SparseInterp 是否只因实现不好而慢？
```

### 2.3 缺陷三：当前 convergence compensation 测试太窄

v6.2 的 P4 只做了 compact near-miss smoke，结果没有 survivor。但这还不能证明 “加速收敛不能补偿 step-time”。它只能说明当前 recipe 不够。

根据近年的加速优化经验，真正实用的加速并不是静态 preconditioner，而是 adaptive dynamics、Nesterov/Adan/Win-like momentum、restart、time-to-target controller。v6.3 应该让 near-miss candidate 测试更多 optimizer dynamics：

```text
ManualAdam
ManualNesterovAdam
ManualAdanLite
ManualWinLite
Lookahead
Restart-on-plateau
Role-wise LR schedule
```

并用 wall-clock target 而不是只看 final accuracy。

### 2.4 缺陷四：gate 仍然太二段式

过去很多计划采用：

```text
先过效率 gate，再跑 task。
```

这个逻辑能避免浪费资源，但现在 `DWM2-poly2` 已经是 near-miss。对于 near-miss，我们需要额外保留一条路径：

```text
效率略未过，但可能通过更快收敛补偿。
```

因此 v6.3 采用双通道 gate：

```text
Efficiency-survivor track:
  直接进入 task recipe。

Near-miss acceleration track:
  允许 step ratio <= 1.6 且 backward memory <= 1.25 的候选进入 time-to-target 验证。
```

---

## 3. v6.3 总体路线

v6.3 命名为：

```text
DG-KAN v6.3 Graph-Free Fused Kernel + Accelerated Convergence
```

核心原则是：

$$
\boxed{
\text{先让 graph-free primitive 真正 kernel-native，再用 acceleration 验证能否 wall-clock 赢 MLP。}
}
$$

具体路线分成四条假设。

---

## 4. 核心假设

### H1：当前 manual primitive 慢，主要是 kernel launch / tensor temporary，不是数学复杂度

如果 H1 成立，则 `torch.compile`、Triton fused forward/backward、或手写 fused CUDA 能显著降低：

$$
T_{forward},\quad T_{backward},\quad M_{workspace}.
$$

验证方式：component profiler、kernel count、op count、temporary bytes。

成功信号：同一 primitive 的 fused 版本相对 current manual 版本满足：

$$
\frac{T_{step}^{fused}}{T_{step}^{manual}}<0.75,
$$

且：

$$
\frac{M_{backward}^{fused}}{M_{backward}^{manual}}<0.9.
$$

### H2：DWM2-poly2 是 efficiency candidate，但 task 表达不足

如果 H2 成立，则扩展为 `poly2+gate`、`poly3`、`poly2+silu_base` 会提高 KMNIST accuracy，但不能显著破坏效率。

成功信号：

$$
Acc_{KMNIST}^{poly2+variant} - Acc_{KMNIST}^{poly2} > 1\%.
$$

同时：

$$
T_{step}/T_{MLP}<1.35,
$$

$$
M_{backward}/M_{MLP}<1.20.
$$

### H3：SparseInterp 理论上最适合终极目标，但当前实现没有真正 sparse/fused

如果 H3 成立，则 fused SparseInterp 应显著优于 current vectorized/fusedIndex 版本，至少满足：

$$
T_{forward}^{fusedSparse}<2T_{MLP},
$$

$$
M_{backward}^{fusedSparse}<M_{MLP}.
$$

如果 fused 后仍慢，则 SparseInterp 不是当前硬件/实现条件下的主线。

### H4：near-miss candidate 可以通过 accelerated recipe 弥补 step-time 开销

如果 H4 成立，候选即使 step ratio 为 `1.2-1.5`，也可以因为更快下降满足：

$$
\text{time-to-target}_{KAN}<\text{time-to-target}_{MLP}.
$$

必须用以下指标判断：

```text
time_to_target_loss
time_to_target_acc
val_loss_auc_by_time
early_loss_slope_by_time
```

而不能只看 final accuracy。

---

## 5. P0：实现与 baseline contract

### 5.1 目标

P0 不跑任务结论，只冻结 runtime baseline 和 graph-free invariants。v6.3 以前常出现一个隐含问题：到底拿哪个 MLP 当效率基准？v6.3 必须显式区分。

### 5.2 必跑对象

```text
MLP-autograd-reference
MLP-manual-linear-reference
DWM2-lite-poly2-current
DWM2-lite-RBFK2-cacheMin
DWM2-lite-fastRational
SparseInterpKAN-K8-current
RationalKAT-lite-fastpoly
```

### 5.3 必须记录

```text
uses_loss_backward
uses_torch_autograd_graph
manual_forward_available
manual_backward_available
manual_update_available
edge_param_count
nonKAN_param_count
coverage_edge
rollback_max_abs_error
cache_total_MB
cache_x_MB
cache_hidden_MB
cache_index_MB
cache_weight_MB
cache_delta_MB
```

### 5.4 通过条件

所有 graph-free candidate 必须满足：

$$
\text{uses\_loss\_backward}=0,
$$

$$
\text{nonKAN\_param\_count}=0,
$$

$$
\text{coverage\_edge}=1.
$$

如果任一 candidate 失败，不进入 P1。

---

## 6. P1：component-level kernel/cache profiler

### 6.1 目标

P1 要回答：当前慢在哪里。

不再只记录总 forward/backward，而是拆成：

```text
preprocess / transform / mixing / adjoint / update / temporary allocation
```

### 6.2 方法

每个 candidate 做以下 profile：

```text
batch size = 128, 256, 512
hidden_dim = 64
num_classes = 10
depth = 2, 4
warmup steps = 20
measured steps = 100
```

分别记录 cold 与 warm：

```text
cold_forward_time_ms
warm_forward_time_ms
cold_backward_time_ms
warm_backward_time_ms
compile_time_ms
recompile_count
graph_break_count
```

### 6.3 必须记录的效率指标

```text
forward_time_ms
manual_backward_time_ms
update_time_ms
step_time_ms
forward_ratio_vs_mlp_autograd
backward_ratio_vs_mlp_autograd
step_ratio_vs_mlp_autograd
forward_ratio_vs_mlp_manual
backward_ratio_vs_mlp_manual
step_ratio_vs_mlp_manual
peak_allocated_MB
peak_reserved_MB
backward_memory_ratio_vs_mlp_autograd
backward_memory_ratio_vs_mlp_manual
workspace_temp_MB
cache_total_MB
```

### 6.4 必须记录的 kernel/op 指标

```text
kernel_count_forward
kernel_count_backward
kernel_count_update
op_count_gemm
op_count_elementwise
op_count_exp
op_count_pow
op_count_gather
op_count_scatter
op_count_index_select
op_count_scatter_add
num_tensor_allocations
largest_temp_tensor_MB
```

### 6.5 P1 可视化

必须画：

```text
component runtime waterfall
kernel count stacked bar
cache decomposition stacked bar
forward/backward/step ratio dashboard
memory ratio vs step ratio scatter
op-count heatmap
cold-vs-warm timing plot
```

### 6.6 P1 判定

定义两个 gate。

#### Efficiency survivor gate

$$
T_{step}/T_{MLP\_autograd}<1.20,
$$

$$
M_{backward}/M_{MLP\_autograd}<1.00.
$$

#### Near-miss gate

$$
T_{step}/T_{MLP\_autograd}<1.60,
$$

$$
M_{backward}/M_{MLP\_autograd}<1.25.
$$

P1 允许 near-miss 进入 P3/P4，但不能进入 final confirm。

---

## 7. P2：kernel repair package

### 7.1 目标

P2 针对 P1 中的 top candidates 做 kernel repair，不跑完整任务。重点是把当前 manual primitive 变成真正 fused / streaming / low-cache primitive。

### 7.2 候选一：DWM2-poly2 fused

当前最接近 efficiency 的 candidate 是 `DWM2-lite-poly2`。P2 对它做：

```text
DWM2-poly2-current
DWM2-poly2-no-temp
DWM2-poly2-fused-forward
DWM2-poly2-fused-adjoint
DWM2-poly2-fused-forward-adjoint
DWM2-poly2-compiled
```

要验证：

```text
是否减少 elementwise kernel 数量；
是否减少 temporary tensor；
是否改善 step ratio；
是否保持 gradient correctness。
```

### 7.3 候选二：DWM2-RBFK2 exp 替代

RBFK2 仍有表达价值，但 exp 可能是 forward bottleneck。测试：

```text
RBFK2-exact-exp
RBFK2-fast-exp-approx
RBFK2-poly-exp-approx
RBFK2-lut-exp
RBFK2-piecewise-exp
```

记录：

```text
approx_error_max
approx_error_mean
forward_time_ms
backward_time_ms
accuracy_smoke
```

### 7.4 候选三：SparseInterp fused kernel

当前 SparseInterp 慢得不合理。P2 不再使用现有 vectorized/fusedIndex 作为最终判断，而是实现两个极简版本：

```text
SparseInterp-gather2-no-scatter-training
SparseInterp-fused-adjoint-scatter
```

第一版只测 forward + input backward，不更新 knots，用来确认 forward path 是否能做到快。第二版再做 knot gradient scatter。

### 7.5 P2 通过条件

对每个候选，必须同时满足：

$$
\text{grad relerr}<10^{-5},
$$

$$
\text{grad cosine}>0.9999,
$$

以及至少满足 near-miss gate。

如果没有候选进入 near-miss gate，v6.3 直接停止，不进入 P3/P4。

---

## 8. P3：minimal task recipe repair

### 8.1 目标

P3 只跑通过 P1/P2 的 efficiency-survivor 或 near-miss candidate。目标不是找最终超参，而是判断 candidate 是否具备基本任务学习能力。

### 8.2 数据集

```text
MNIST
Fashion-MNIST
KMNIST
train / val / test = 1536 / 512 / 512
seeds = 0,1,2
```

### 8.3 方法

```text
MLP-AdamW-autograd-reference
MLP-manual-linear-reference
DWM2-poly2-best
DWM2-RBFK2-best
DWM2-fastRational-best
SparseInterp-best, if P2 near-miss
RationalKAT-fastpoly-best, if P2 near-miss
```

### 8.4 Optimizer recipes

每个 candidate 先只测下面四个：

```text
ManualAdam
ManualAdamW
ManualNesterovAdam
ManualAdanLite
```

不在 P3 使用 LightSmooth 或 functional geometry update。

### 8.5 记录指标

任务：

```text
train_loss_curve
val_loss_curve
test_acc
val_acc
val_auc_by_step
val_auc_by_time
ECE
NLL
margin_mean
margin_p10
classwise_acc
```

效率：

```text
step_time_ms
forward_time_ms
backward_time_ms
update_time_ms
peak_allocated_MB
backward_memory_ratio
samples_per_second
```

收敛：

```text
early_loss_slope_step
early_loss_slope_time
time_to_mlp_final_loss
time_to_mlp_final_acc
time_to_relaxed_loss
time_to_relaxed_acc
reached_target_loss
reached_target_acc
```

表示：

```text
effective_rank_input
effective_rank_block
effective_rank_output
class_centroid_separation
feature_norm_mean
feature_norm_p95
```

### 8.6 可视化

```text
loss vs step
loss vs wall-clock
accuracy vs wall-clock
val_auc_by_time bar
step-time vs accuracy Pareto
time-to-target Kaplan curve
classwise accuracy heatmap
feature-rank trajectory
margin trajectory
```

### 8.7 P3 通过条件

P3 candidate 必须至少在两个数据集满足：

$$
Acc_{candidate}\geq Acc_{MLP}-1.5\%,
$$

且至少在一个数据集满足：

$$
\text{time-to-target}_{candidate}<\text{time-to-target}_{MLP}.
$$

如果没有 candidate 通过，进入 P8 failure diagnosis，不继续 P4-P7。

---

## 9. P4：acceleration package

### 9.1 目标

P4 针对 P3 过线或 near-pass 的候选，系统验证加速收敛方案。

这里的重点是：

$$
\boxed{
\text{即使 step 更慢，也要证明 wall-clock 更快或 AUC-by-time 更好。}
}
$$

### 9.2 加速方法

测试以下 optimizer dynamics：

```text
A0 ManualAdamW
A1 ManualNesterovAdamW
A2 ManualAdanLite
A3 ManualWinLite
A4 Lookahead-ManualAdamW
A5 Restart-ManualAdamW
A6 WarmupCosine-ManualAdamW
A7 RoleWiseLR-InputHeavy
A8 RoleWiseLR-OutputWarmup
```

其中 RoleWiseLR 的动机是：之前 AdamW trajectory forensic 显示 early update share 主要集中在 input role，所以需要验证：

$$
\eta_{input} > \eta_{block} > \eta_{output}
$$

是否能提升 early representation learning。

### 9.3 记录指标

```text
optimizer_state_norm_m
optimizer_state_norm_v
nesterov_lookahead_norm
adan_diff_momentum_norm
win_weight_decay_coupling_norm
restart_count
restart_reason
role_lr_input
role_lr_block
role_lr_output
role_update_share_input
role_update_share_block
role_update_share_output
```

收敛指标必须按 step 和 wall-clock 双记录：

```text
loss_auc_by_step
loss_auc_by_time
time_to_target_loss
steps_to_target_loss
time_to_target_acc
steps_to_target_acc
early_loss_slope_step
early_loss_slope_time
```

### 9.4 可视化

```text
optimizer dynamics trace
role update share stacked area
time-to-target bar
loss vs time with restart markers
AUC-by-time Pareto
restart reason histogram
```

### 9.5 P4 通过条件

候选要进入 P5，必须满足：

$$
\text{time-to-target}_{candidate}<0.9\cdot\text{time-to-target}_{MLP}
$$

或：

$$
\text{val AUC by time}_{candidate}>\text{val AUC by time}_{MLP}+5\%.
$$

同时：

$$
Acc_{candidate}\geq Acc_{MLP}-1\%.
$$

---

## 10. P5：geometry / LightSmooth compatibility smoke

### 10.1 目标

只有 P4 出现 task + time survivor 后，才进入 P5。P5 不再试图让 LightSmooth 拯救 task 失败候选。它只回答：

```text
高效 graph-free candidate 是否能接入低频 geometry maintenance？
```

### 10.2 方法

```text
Best candidate from P4
Best candidate + one LightSmooth event
Best candidate + event-driven LightSmooth, max events = 2
MLP reference
```

### 10.3 记录指标

```text
phi_residual
curvature_residual
sobolev_residual_norm
logit_drift
KL_teacher_student
acc_drop_after_smooth
refresh_recovery
smooth_event_count
accepted_smooth_count
smoothing_time_ms
amortized_smoothing_overhead
```

### 10.4 可视化

```text
geometry curve with smoothing markers
accuracy curve with smoothing markers
logit drift vs geometry reduction scatter
smoothing overhead bar
```

### 10.5 通过条件

$$
\Delta Acc_{smooth}\geq -0.5\%,
$$

$$
\phi\_residual\text{ reduction}>10\%,
$$

$$
\text{amortized overhead}<5\%.
$$

---

## 11. P6：functional update without autograd smoke

### 11.1 目标

P6 是回到 functional update 的第一步。只有 P4/P5 通过后才跑。

这一步验证：

```text
在 graph-free primitive 上，manual gradient + functional metric update 是否能替代 ManualAdamW 的一部分更新。
```

### 11.2 方法

```text
ManualAdamW baseline
FunctionalDiag
FunctionalDataDiag
FunctionalResidualSobolev
FunctionalCoordinateAdam
FunctionalCoordinateAdanLite
```

### 11.3 记录指标

```text
manual_grad_norm
functional_direction_norm
cos_with_manual_adam_direction
predicted_descent
actual_train_descent
actual_holdout_descent
bad_step_rate
update_over_param_norm
role_update_share
```

### 11.4 可视化

```text
functional direction cosine heatmap
predicted vs actual descent scatter
bad-step timeline
role update share comparison
```

### 11.5 通过条件

$$
\text{bad step rate}<5\%,
$$

$$
\cos(d_{functional},d_{ManualAdam})>0.5
$$

或：

$$
\text{holdout descent}_{functional}>\text{holdout descent}_{ManualAdam}.
$$

---

## 12. P7：3-seed joint candidate selection

### 12.1 目标

P7 是 v6.3 的第一轮正式 selection。候选必须来自 P4/P5/P6 survivor。

### 12.2 数据集

```text
MNIST
Fashion-MNIST
KMNIST
train / val / test = 6000 / 1000 / 1000
seeds = 0,1,2
```

### 12.3 对照

```text
MLP-AdamW-autograd-reference
MLP-manual-linear-reference
Best graph-free PureKAN ManualAdamW
Best graph-free PureKAN accelerated
Best graph-free PureKAN accelerated + LightSmooth
Best graph-free PureKAN functional, if P6 passes
```

### 12.4 指标

必须完整记录：

```text
accuracy
acc std
validation-loss AUC by step
validation-loss AUC by time
ECE
NLL
margin mean / p10
feature rank
geometry metrics
forward / backward / step time
peak memory
backward memory ratio
samples/sec
time-to-target loss / acc
```

### 12.5 通过条件

P7 进入 P8 的候选必须满足 all-dataset：

$$
Acc_{candidate}\geq Acc_{MLP}-0.5\%,
$$

$$
\text{time-to-target}_{candidate}\leq\text{time-to-target}_{MLP},
$$

$$
M_{backward}/M_{MLP}\leq1.0.
$$

如果还做不到 backward memory below MLP，则可以进入 exploratory confirm，但不能 claim 终极目标。

---

## 13. P8 / P9：5-seed 与 10-seed confirm

### P8 5-seed

只扩 P7 survivor。

```text
seeds = 0,1,2,3,4
```

通过条件：

$$
\Delta Acc_{candidate-MLP}\geq0
$$

或 paired CI 不显著低于 MLP，同时：

$$
\text{time-to-target}_{candidate}<\text{time-to-target}_{MLP}.
$$

### P9 10-seed

只扩 P8 clean survivor。

```text
seeds = 0..9
```

终极目标 claim 需要：

$$
Acc_{candidate}>Acc_{MLP},
$$

$$
AUC_{time,candidate}>AUC_{time,MLP},
$$

$$
ECE_{candidate}<ECE_{MLP},
$$

$$
M_{backward,candidate}<M_{backward,MLP},
$$

且：

$$
T_{forward,candidate}\approx T_{forward,MLP},\quad T_{backward,candidate}\approx T_{backward,MLP}.
$$

---

## 14. P10：failure diagnosis

如果 v6.3 失败，不允许只写“又失败了”。必须分类。

### 14.1 Failure taxonomy

```text
F1_kernel_launch_overhead:
  kernel_count 太高，单 op 太碎。

F2_transform_cost:
  transform 函数本身太贵，例如 exp/rational/gather。

F3_cache_memory:
  cache 或 workspace 仍高于 MLP。

F4_task_underfit:
  efficiency 过了，但 accuracy 差。

F5_convergence_slow:
  final accuracy 可接受，但 time-to-target 输。

F6_acceleration_unstable:
  Nesterov/Adan/Win-like dynamics 提速失败或 seed unstable。

F7_geometry_incompatible:
  task/efficiency 过了，但 geometry 完全无法维护。

F8_manual_runtime_framework:
  manual MLP reference 本身异常，说明 benchmark framework 不公平。
```

### 14.2 诊断图

```text
failure heatmap by primitive and stage
kernel-count vs time scatter
transform-cost vs accuracy scatter
memory vs cache decomposition
step-time vs time-to-target scatter
accuracy vs geometry vs memory Pareto
```

---

## 15. v6.3 决策树

### Case A：DWM2-poly2 fused 过 efficiency + acceleration

结论：

```text
DWM2-poly2 becomes the first graph-free efficient PureKAN primitive.
Next: add LightSmooth and functional update smoke.
```

### Case B：DWM2-poly2 过 efficiency 但 task 不够

结论：

```text
poly2 is efficient but under-expressive.
Next: poly3 / gated poly / fastRational, not RBF dense.
```

### Case C：SparseInterp fused 后过 efficiency

结论：

```text
SparseInterp becomes the best terminal primitive candidate.
Next: task recipe and knot geometry.
```

### Case D：所有 primitive 仍不接近 MLP

结论：

```text
Current PureKAN primitive family still misses terminal efficiency target.
Functional optimizer remains gated.
Need either a lower-level custom kernel implementation or a different primitive family.
```

### Case E：primitive 过效率但收敛不补偿

结论：

```text
Graph-free efficiency alone insufficient.
Need acceleration / role-wise training / better initialization.
```

---

## 16. 最终总结

v6.3 的关键不是继续证明 analytic adjoint 正确。这个 v6.1/v6.2 已经证明了。

v6.3 的关键是回答：

$$
\boxed{
\text{graph-free PureKAN 能不能同时做到 kernel-efficient 和 convergence-efficient？}
}
$$

如果可以，才进入最终 functional optimizer。否则，所有 functional update / LightSmooth / geometry maintenance 都只能算机制研究，不能 claim 终极目标。

