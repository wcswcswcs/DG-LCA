# DG-KAN v5.9 Memory-First Efficient PureKAN Primitive 实验计划

## 0. 终极目标与本轮定位

当前终极目标必须保持不变：

$$
\boxed{
\text{构建一个无 non-KAN 参数的 PureKAN functional training system，}
}
$$

并且它必须满足：

$$
\boxed{
\text{forward 时间/显存接近 MLP，backward 时间接近 MLP，backward 显存低于 MLP，收敛速度快，}
}
$$

$$
\boxed{
\text{并在 accuracy / AUC / ECE / geometry 上超过 MLP-AdamW 与 PureKAN-AdamW。}
}
$$

v5.8 的结果说明，我们现在还没有到 functional optimizer 阶段。更准确地说，当前瓶颈是：

$$
\boxed{
\text{还没有一个 strict PureKAN primitive 同时通过效率门槛和任务门槛。}
}
$$

因此 v5.9 不应该继续做 U-FULL、LightSmooth、FNG、TFU 或 smoothing controller。v5.9 的目标是先把 primitive 的效率与显存问题拆开，找出到底是 **计算图结构**、**autograd 保存张量**、**kernel path**、**benchmark 混入错误**，还是 **primitive 本身的数学形式** 导致所有候选在 P1 被 memory gate 卡死。

v5.9 的核心原则是：

```text
Profiler truth first.
Memory cause second.
Custom backward third.
Task recipe fourth.
Functional optimizer last.
```

如果 P1/P2 不能证明某个 primitive 具有接近 MLP 的真实 forward/backward 可能性，后续不进入 task recipe，更不进入 functional update。

---

## 1. v5.8 结果复盘与关键判断

### 1.1 P0 是好消息：strict PureKAN primitive 已经能进入 core

v5.8 已经把这些候选加入 core 并通过 manifest：

```text
DWM2Dense
RationalKATV2Dense
LUTKANDense
CPABRBF reference
Dense ABRBF reference
```

P0 显示它们满足：

```text
edge-covered
nonKAN = 0
rollback = 0
```

这说明当前失败不是因为又混进 MLP stem/head/LN，也不是因为 rollback 或 parameter discovery 明显错误。v5.8 的实现状态比早期阶段更干净。

但 P0 只能证明 **参数归属正确**，不能证明 **计算路径高效**，也不能证明 **backward 显存合理**。

---

### 1.2 P1 是核心失败：所有 primitive 都被 memory gate 卡死

v5.8 的 P1 没有任何 exploratory survivor，也没有 final survivor。典型结果是：

| primitive | forward ratio | backward ratio | backward memory ratio | step ratio | bottleneck |
|---|---:|---:|---:|---:|---|
| DWM2-prePostMix | 6.18 | 3.14 | 3.18 | 2.95 | memory |
| DWM2-residual | 6.39 | 3.44 | 3.20 | 3.15 | memory |
| DWM2-gated | 7.16 | 3.92 | 3.32 | 3.54 | memory |
| RationalKAT-AB-v2 | 7.78 | 4.95 | 2.25 | 4.22 | memory |
| LUTKAN-AB | 11.84 | 3.13 | 2.39 | 3.59 | memory |
| CP-ABRBF-reference | 13.74 | 3.72 | 3.99 | 4.26 | memory |

这说明三个事实。

第一，当前新增 primitive 虽然参数量更低，但 **PyTorch 计算图并没有变成 MLP-like path**。如果真正是 MLP-like，forward ratio 不应该长期维持在 $6\times$ 到 $12\times$。

第二，所有候选最主要的 failure type 是 memory fail，这说明 backward 保存的中间张量或 workspace 是当前最关键问题。

第三，P2/P3/P4 没有继续跑是正确的。没有效率 survivor 时，继续做 recipe 会浪费实验资源。

---

### 1.3 v5.8 的失败不是“DWM2 / Rational / LUT 思想全错”，而是当前实现没有证明 kernel path

DWM2 的理论路径应该接近：

$$
O(BdK)+O(Bdh),
$$

而 MLP 是：

$$
O(Bdh).
$$

如果 $K$ 小，且实现是 fused / vectorized / GEMM-native，那么 DWM2 应该远小于 dense RBF 的：

$$
O(BdhK).
$$

但 v5.8 DWM2 仍然是 $6\times$ forward 和 $3\times$ memory。这说明当前实现至少有一个问题：

```text
1. forward 仍有大量非 GEMM elementwise / reshape / temporary tensor；
2. backward 保存了完整 intermediate；
3. torch.compile / eager path 没真正融合；
4. benchmark 把不该算入 steady-state 的开销算进去了；
5. MLP baseline 太小，kernel launch overhead 主导；
6. primitive 的数学形式本身仍然太复杂。
```

v5.9 必须逐项排查，而不是继续换 optimizer。

---

## 2. 代码实现层面的重新审视

### 2.1 Dense RBF / Dense AB-RBF 已经完成机制任务，不再作为终极候选

Dense RBF 的计算形式是：

$$
B_{bik}=\exp\left(-\frac{(x_{bi}-c_k)^2}{2\sigma^2}\right),
$$

$$
y_{bo}=\sum_{i,k}B_{bik}c_{oik}.
$$

其复杂度是：

$$
O(Bd_{in}d_{out}K).
$$

MLP 是：

$$
O(Bd_{in}d_{out}).
$$

因此 dense RBF / dense AB-RBF 无论怎么调，都很难满足终极效率目标。它们应该保留为：

```text
mechanism reference
teacher
accuracy / geometry upper reference
```

而不再作为最终模型。

---

### 2.2 现在必须把 primitive 定义成 “forward equation + backward saved tensors + optimizer states”

以前我们只问：

```text
这个 primitive 的 forward 数学形式是什么？
```

现在必须同时问：

```text
forward 需要几个 kernel？
forward 需要哪些 temporary tensors？
backward autograd 保存了哪些 tensors？
custom backward 能不能只保存 x / logits / small stats？
optimizer state 是每 parameter 还是每 function coordinate？
```

一个 primitive 如果 forward 公式看起来轻，但 backward 需要保存大 tensor，也不能作为终极候选。

因此 v5.9 要把每个 primitive 的内存分解成：

$$
M_{total}=M_{params}+M_{optimizer}+M_{activations}+M_{saved\ tensors}+M_{workspace}+M_{fragmentation}.
$$

只有知道 memory fail 的具体来源，才能决定是 custom backward、fused kernel、recompute，还是直接放弃该 primitive。

---

### 2.3 目前最值得继续的候选顺序

v5.9 按以下优先级推进。

#### 第一优先级：DWM2-lite

DWM2 的思想仍然最接近 AB-RBF 机制，同时计算复杂度理论上能接近 MLP。下一轮要把它从当前的 DWM2-prePostMix / gated / residual 改成更克制的 lite 版本：

$$
\tilde x = x + s\cdot r(x),
$$

$$
y = W\tilde x.
$$

其中 $r(x)$ 是低成本 channel-wise KAN residual。默认只测试：

```text
K = 2, 4, 8
base = linear + silu
residual = piecewise / LUT / tiny RBF
mixing = single GEMM
no gated controller initially
```

DWM2-lite 的目标不是最高表达力，而是先证明：

$$
\boxed{\text{KAN-like edge transform 可以在 MLP-like compute path 中存在。}}
$$

#### 第二优先级：RationalKAT-v2-lite

Rational/KAT 的优势是 elementwise rational + GEMM，理论上最接近 MLP：

$$
y = W r(x).
$$

但 v5.8 的 RationalKAT-v2 forward / backward 仍然过慢，说明当前 rational implementation 或 grouping recipe 没释放出来。v5.9 只保留 identity-like、small residual 的 RationalKAT-lite：

$$
r(x)=x+s\cdot q(x),
$$

其中 $q(x)$ 是小幅 rational residual。先不做 functional update，只修：

```text
initialization
scale
groups
compiled/eager consistency
backward saved tensors
accuracy recipe
```

#### 第三优先级：LUTKAN-v2

LUTKAN 原本应该比 RBF 更省，因为它避免 exp，使用 bin index / interpolation。v5.8 中 LUTKAN-AB 仍然很慢，说明实现可能没有写到 lookup-native 路径。v5.9 只保留一个很小的 LUTKAN-v2：

```text
piecewise linear bins
store bin index as int16 or recompute
linear interpolation
single GEMM mix
custom backward prototype
```

如果 LUTKAN-v2 仍然 forward > $2\times$ MLP，则暂停。

#### 第四优先级：CP-ABRBF

CP 作为 expressivity-preserving reference 保留，但不是第一效率主线。除非 rank-speed / memory 单调性明确，否则不继续扩。

---

## 3. v5.9 总体实验策略

v5.9 分成两个大阶段。

第一阶段是 **efficiency forensics**。不训练，不看 accuracy，只回答每个 primitive 为什么慢、为什么占显存。

第二阶段是 **minimal task recipe**。只允许通过 efficiency forensic 的候选进入任务训练。

执行原则：

```text
没有 P1/P2 efficiency evidence，不跑 P3 recipe。
没有 P3 task evidence，不跑 P4 custom backward full confirm。
没有 P4 survivor，不跑 functional update / LightSmooth。
```

---

# 4. P0: Core / Runner / Primitive Manifest Hardening

## 4.1 目的

确保 v5.8 新增的 primitive 不是只在 runner 中临时存在，而是 core-level 一致实现。

## 4.2 必跑 primitive

```text
MLP-reference
Dense-ABRBF-reference
DWM2-prePostMix-current
DWM2-lite-RBFK2
DWM2-lite-RBFK4
DWM2-lite-LUTK8
RationalKAT-v2-current
RationalKAT-lite-groups8
RationalKAT-lite-groups16
LUTKAN-current
LUTKAN-v2-linearInterp
CP-ABRBF-r4-reference
```

## 4.3 必须记录

```text
primitive_name
class_name
backend
device
batch_size
hidden_dim
depth
basis_count_or_bins
groups
edge_param_count
base_param_count
residual_param_count
mixing_param_count
nonKAN_param_count
edge_coverage
base_coverage
residual_coverage
mixing_coverage
rollback_max_error
uses_custom_backward
uses_torch_compile
graph_break_count
recompile_count
```

## 4.4 P0 通过标准

```text
nonKAN_param_count = 0
edge_coverage = 1.0
rollback_max_error < 1e-8
graph_break_count = 0 for compiled candidates
```

如果 P0 失败，不进入 P1。

---

# 5. P1: Profiler Truth and Memory Decomposition

## 5.1 目的

v5.8 的所有候选都被 memory gate 卡死。P1 要回答：

```text
memory fail 来自哪里？
是 activation saved tensor？
是 optimizer state？
是 temporary workspace？
是 CUDA reserved fragmentation？
还是 benchmark 统计方式不对？
```

## 5.2 方法

每个 primitive 分别测：

```text
forward only
forward + loss
backward only
optimizer step only
full train step
inference eval step
```

并且分别记录 cold / warm：

```text
cold_forward_ms
warm_forward_ms
cold_backward_ms
warm_backward_ms
compile_time_ms
```

如果使用 `torch.compile`，必须丢弃前若干 warmup steps，不得把 compile time 算入 steady-state。

## 5.3 Saved tensor hook

使用 `torch.autograd.graph.saved_tensors_hooks` 记录 backward 保存的 tensor：

```text
saved_tensor_count
saved_tensor_total_bytes
saved_tensor_largest_bytes
saved_tensor_shapes_top10
saved_tensor_dtype_histogram
```

这一步是 v5.9 的关键。没有 saved tensor breakdown，就无法判断 custom backward 是否值得做。

## 5.4 必须记录的效率指标

```text
forward_time_ms
backward_time_ms
optimizer_time_ms
step_time_ms
inference_time_ms
peak_allocated_mb
peak_reserved_mb
activation_saved_mb
saved_tensor_total_mb
workspace_temp_mb
optimizer_state_mb
param_mb
kernel_count
gemm_count
einsum_count
exp_count
pow_count
index_select_count
compile_time_ms
cold_warm_ratio
```

## 5.5 P1 exploratory gate

为了避免过早淘汰候选，P1 分 exploratory 和 final 两级。

Exploratory gate：

$$
T_{forward}/T_{MLP} \leq 3.0,
$$

$$
T_{backward}/T_{MLP} \leq 3.0,
$$

$$
M_{backward}/M_{MLP} \leq 2.0.
$$

Final gate：

$$
T_{forward}/T_{MLP} \leq 1.25,
$$

$$
T_{backward}/T_{MLP} \leq 1.40,
$$

$$
M_{backward}/M_{MLP} \leq 1.00,
$$

with final target:

$$
M_{backward}/M_{MLP} \leq 0.80.
$$

P1 只要求 exploratory survivor 进入 P2。

## 5.6 必须画图

```text
cold_vs_warm_timing_plot.svg
memory_decomposition_stacked_bar.svg
saved_tensor_top_shapes.svg
kernel_count_heatmap.svg
forward_backward_step_ratio_dashboard.svg
```

---

# 6. P2: Custom Backward / Recompute Feasibility

## 6.1 目的

如果 P1 显示 memory fail 主要来自 saved tensors，则测试 custom backward 是否能解决。

## 6.2 候选

只对 P1 exploratory survivor 做 P2。默认最多三个：

```text
DWM2-lite-best
RationalKAT-lite-best
LUTKAN-v2-best
```

每个比较：

```text
autograd
checkpoint/recompute
custom autograd recompute
streaming backward stats
```

## 6.3 DWM2 custom backward 设计

DWM2-lite 的 forward：

$$
\tilde x = x + s r(x),
$$

$$
y = W\tilde x.
$$

Backward 只需要保存：

```text
x
W
small config
```

不保存完整 intermediate residual basis。如果是 LUT residual，可以只保存 bin index 或重算 bin：

```text
store: x or int16 bin index
recompute: interpolation weights
```

## 6.4 RationalKAT custom backward 设计

RationalKAT-lite 的 forward：

$$
r(x)=x+s q(x),
$$

$$
y=W r(x).
$$

Backward 只保存：

```text
x
rational parameters
W
```

并重算 numerator / denominator。必须记录 denominator safety：

```text
den_min
den_p01
den_condition
r_prime_p95
r_double_prime_p95
```

## 6.5 LUTKAN custom backward 设计

LUTKAN-v2 的 forward：

$$
r(x)= (1-t)v_j + t v_{j+1},
$$

where:

$$
j=\operatorname{bin}(x),\quad t=\frac{x-x_j}{x_{j+1}-x_j}.
$$

Backward 可以只保存 $j$ 和 $t$，或者只保存 $x$ 并重算。

## 6.6 必须记录

```text
rel_error_forward
rel_error_grad_input
rel_error_grad_params
cos_grad_input
cos_grad_params
saved_tensor_total_mb
peak_allocated_mb
backward_time_ms
step_time_ms
custom_backward_compile_time_ms
```

## 6.7 P2 gate

```text
rel_error_grad_params < 1e-4
cos_grad_params > 0.999
backward_memory_ratio_vs_mlp <= 1.25 exploratory
backward_memory_ratio_vs_mlp <= 1.00 final
backward_time_ratio_vs_mlp <= 2.0 exploratory
```

如果 custom backward 减少 saved tensor，但 step time 爆炸超过 $4\times$，该路线只保留为 diagnostic。

---

# 7. P3: Minimal Task Recipe for Efficient Survivors

## 7.1 目的

只让 P2 过线的 primitive 进入任务训练，避免在不可能满足效率目标的模型上浪费 recipe 搜索。

## 7.2 Baselines

```text
MLP-AdamW
PureKAN-RBFOnly-Dense-AdamW, reference only
Dense-ABRBF-AdamW, mechanism reference only
best primitive from P2
```

## 7.3 DWM2-lite recipe grid

```text
K in {2,4,8}
residual_scale in {0.05,0.10,0.20}
base in {linear_silu, silu, linear}
mix_width in {hidden, 2hidden}
norm in {FixedNorm, channelNorm}
lr in {1e-3, 2e-3, 3e-3}
```

Keep grid compact. Start with seeds `0,1,2`.

## 7.4 RationalKAT-lite recipe grid

```text
groups in {4,8,16,32}
rational_mode in {identity_residual, silu_residual, linear_silu}
residual_scale in {0.03,0.05,0.10}
lr in {1e-3,2e-3,3e-3}
denominator_damping in {1e-3,1e-2}
```

## 7.5 LUTKAN-v2 recipe grid

```text
bins in {8,16,32}
interpolation in {linear, cubic_hermite_optional}
residual_scale in {0.05,0.10,0.20}
base in {linear_silu, silu}
lr in {1e-3,2e-3}
```

## 7.6 Task metrics

```text
test_acc
val_loss
val_loss_auc
ECE
NLL
margin_mean
margin_p10
feature_effective_rank
class_centroid_separation
classwise_accuracy
train_to_val_gap
convergence_epoch_to_target
```

## 7.7 Efficiency metrics during training

```text
forward_time_ratio
backward_time_ratio
step_time_ratio
backward_memory_ratio
activation_saved_mb
saved_tensor_total_mb
optimizer_state_mb
samples_per_sec
```

## 7.8 P3 survivor gate

For exploratory survivor:

```text
accuracy gap vs MLP <= 2.0 percentage points on all datasets
val_loss_auc not worse than MLP by more than 5%
ECE <= MLP + 0.03
backward_memory_ratio <= 1.25
step_time_ratio <= 2.0
```

For final survivor:

```text
accuracy >= MLP-AdamW or gap <= 0.5 percentage points
AUC >= MLP-AdamW
ECE <= MLP-AdamW
backward_memory_ratio <= 1.0, target <= 0.8
step_time_ratio <= 1.25
```

## 7.9 必须画图

```text
task_efficiency_pareto.svg
recipe_heatmap_accuracy.svg
recipe_heatmap_memory.svg
recipe_heatmap_step_time.svg
classwise_accuracy_failure_heatmap.svg
margin_rank_vs_accuracy.svg
```

---

# 8. P4: Primitive-Specific Geometry and LightSmooth Compatibility

## 8.1 目的

只有 P3 有 survivor 时才跑。不要在效率不合格的 primitive 上做 LightSmooth。

## 8.2 测试内容

对 P3 survivor 做：

```text
single-event geometry maintenance
one-cycle train -> smooth -> refresh
no multicycle yet
```

## 8.3 指标

```text
acc_drop
KL
logit_drift
phi_edge_reduction
curvature_edge_reduction
geometry_retention_after_refresh
memory_ratio
smoothing_time_sec
amortized_step_time_ratio
```

## 8.4 Gate

```text
acc_drop <= 0.005
KL <= 0.005
logit_drift <= 0.03
geometry reduction > 0
amortized overhead <= 10%
```

如果 P4 不过，仍可保留 primitive 作为 efficient PureKAN candidate，但不进入 functional geometry maintenance。

---

# 9. P5: Functional Training Smoke, Only if P3/P4 Survive

## 9.1 目的

测试该 primitive 是否能支持 functional training，而不是只靠 AdamW。

## 9.2 方法

```text
AdamW baseline
functional-coordinate Adam
task-aware diagonal update
residual-only functional smoothing
hybrid AdamW-task + functional residual maintenance
```

注意：这里的 `hybrid` 不是 non-KAN hybrid。所有参数仍在 PureKAN edge system 内。含义是：

```text
base/mix path 用 Adam-like task dynamics；
residual path 用 functional geometry maintenance。
```

## 9.3 指标

```text
accuracy
val_auc
ECE
geometry
backward_memory_ratio
step_time_ratio
functional_update_overhead
convergence_speed_to_target
```

## 9.4 Gate

```text
functional variant >= AdamW primitive accuracy - 0.5%
AUC >= AdamW primitive
ECE <= AdamW primitive
geometry better than AdamW primitive
step_time_ratio overhead <= 1.15x primitive AdamW
```

---

# 10. P6: 5-Seed Confirmation

Only run if P3 produces a task-efficiency survivor.

## 10.1 Methods

```text
MLP-AdamW
best efficient PureKAN primitive AdamW
best efficient PureKAN primitive + LightSmooth, optional
best efficient PureKAN primitive functional, optional if P5 passes
Dense-ABRBF reference, optional diagnostic only
```

## 10.2 Datasets

```text
MNIST
Fashion-MNIST
KMNIST
```

## 10.3 Metrics

```text
mean_acc
acc_std
paired_acc_delta_vs_MLP
val_auc_delta_vs_MLP
ECE_delta_vs_MLP
backward_memory_ratio
step_time_ratio
convergence_epoch_to_target
classwise_accuracy
feature_rank
margin_p10
geometry metrics
```

## 10.4 Gate

```text
accuracy >= MLP or paired CI excludes worse than -0.5%
AUC >= MLP
ECE <= MLP
backward memory <= MLP, target <= 0.8 MLP
step time <= 1.25 MLP
time-to-target <= MLP or not worse by more than 10%
```

---

# 11. P7: 10-Seed Final Confirm

Only run after P6 passes.

## 11.1 必须确认

```text
same datasets
seeds = 0..9
same hardware
same profiler protocol
same batch sizes
```

## 11.2 Final claim requires

$$
\operatorname{Acc}_{PureKAN} \geq \operatorname{Acc}_{MLP-AdamW},
$$

$$
\operatorname{AUC}_{PureKAN} \geq \operatorname{AUC}_{MLP-AdamW},
$$

$$
\operatorname{ECE}_{PureKAN} \leq \operatorname{ECE}_{MLP-AdamW},
$$

$$
\frac{T_{forward,PureKAN}}{T_{forward,MLP}} \leq 1.25,
$$

$$
\frac{T_{backward,PureKAN}}{T_{backward,MLP}} \leq 1.40,
$$

$$
\frac{M_{backward,PureKAN}}{M_{backward,MLP}} \leq 1.00,
$$

with target:

$$
\frac{M_{backward,PureKAN}}{M_{backward,MLP}} \leq 0.80.
$$

If P7 passes, then and only then can we claim progress toward the terminal system.

---

# 12. Failure Interpretation Rules

## 12.1 If P1 fails again

Conclusion:

```text
Current primitive implementations are not MLP-like.
Do not run task recipe.
Work on kernel / custom backward / primitive simplification.
```

Most likely action:

```text
Drop dense RBF / CP.
Focus on RationalKAT-lite and LUTKAN-v2.
```

## 12.2 If P1 passes but P2 fails

Conclusion:

```text
Forward path is plausible, backward memory is still the blocker.
```

Action:

```text
Implement custom backward or recompute strategy.
Do not tune accuracy yet.
```

## 12.3 If P2 passes but P3 fails

Conclusion:

```text
Primitive is efficient but lacks task recipe or capacity.
```

Action:

```text
Tune initialization, residual scale, groups / bins / K, learning rate, normalization.
Do not add functional update yet.
```

## 12.4 If P3 passes but P4 fails

Conclusion:

```text
Primitive can be efficient and accurate, but geometry maintenance is not integrated.
```

Action:

```text
Keep primitive as main candidate; treat LightSmooth as optional offline posthoc.
```

## 12.5 If P3 and P4 pass but P5 fails

Conclusion:

```text
Functional-from-scratch optimizer still unsolved, but efficient PureKAN primitive exists.
```

Action:

```text
Report efficient PureKAN-AdamW as architecture result.
Continue functional optimizer separately.
```

---

# 13. Required Figures

The v5.9 result package must include:

```text
figures/p1_memory_decomposition_stacked.svg
figures/p1_saved_tensor_shapes_top10.svg
figures/p1_kernel_count_heatmap.svg
figures/p1_cold_warm_timing.svg
figures/p2_custom_backward_memory_vs_time.svg
figures/p2_grad_correctness_scatter.svg
figures/p3_task_efficiency_pareto.svg
figures/p3_recipe_accuracy_heatmap.svg
figures/p3_recipe_memory_heatmap.svg
figures/p3_classwise_failure_heatmap.svg
figures/p4_geometry_cost_pareto.svg
figures/p4_refresh_retention_curve.svg
figures/p6_paired_delta_dashboard.svg
```

---

# 14. Required Tables / Artifacts

```text
p0_core_manifest.csv
p1_phase_efficiency.csv
p1_memory_decomposition.csv
p1_saved_tensor_audit.csv
p1_kernel_breakdown.csv
p2_custom_backward_correctness.csv
p2_custom_backward_memory.csv
p3_recipe_grid.csv
p3_task_efficiency_selection.csv
p4_lightsmooth_compatibility.csv
p5_functional_smoke.csv
p6_confirm5.csv
p7_confirm10.csv
failure_table.csv
route_decision.json
aggregate_decision.json
```

Each failure row must include:

```text
primitive
backend
dataset
stage
failure_type
primary_bottleneck
secondary_bottleneck
exact_metric_values
recommended_action
```

---

# 15. Final Recommendation

v5.8 failed early, but that is useful. It tells us that the next step is not optimizer tuning. The current bottleneck is primitive efficiency and backward memory.

The v5.9 recommendation is:

$$
\boxed{
\text{Stop functional optimizer work until at least one primitive passes P1/P2 efficiency gates.}
}
$$

$$
\boxed{
\text{Focus on DWM2-lite, RationalKAT-lite, and LUTKAN-v2 with saved-tensor-aware profiling.}
}
$$

$$
\boxed{
\text{Only after a primitive is MLP-like in compute/memory should LightSmooth or functional training return.}
}
$$

This is the shortest path toward the terminal goal: a strict PureKAN system that is not only accurate and geometric, but also genuinely competitive with MLP in forward time, backward time, and backward memory.
