# DG-KAN v6.0 Efficient Functional PureKAN：高效 primitive、加速收敛与 functional training 重设计实验计划

## 0. 终极目标

本阶段必须重新对齐终极目标：

> 构建一个无 non-KAN 参数的 PureKAN functional training system，其 forward 时间/显存接近 MLP，backward 时间接近 MLP，backward 显存低于 MLP，收敛速度快，并在 accuracy / AUC / ECE / geometry 上超过 MLP-AdamW 与 PureKAN-AdamW。

因此，任何候选方法都不能只在 accuracy 或 geometry 上成立，也不能只在 one-step 或 one-cycle 上成立。最终系统必须同时满足：

$$
\frac{T_{forward,KAN}}{T_{forward,MLP}} \le 1.25,
$$

$$
\frac{M_{forward,KAN}}{M_{forward,MLP}} \le 1.25,
$$

$$
\frac{T_{backward,KAN}}{T_{backward,MLP}} \le 1.40,
$$

$$
\frac{M_{backward,KAN}}{M_{backward,MLP}} \le 0.80,
$$

并且在任务指标上至少满足：

$$
Acc_{KAN} \ge Acc_{MLP-AdamW},
$$

$$
AUC_{KAN} \ge AUC_{MLP-AdamW},
$$

$$
ECE_{KAN} \le ECE_{MLP-AdamW},
$$

同时 geometry 不劣化，最好能满足：

$$
\phi'_{p95,KAN} < \phi'_{p95,PureKAN-AdamW},
$$

$$
\kappa(J)_{KAN} < \kappa(J)_{PureKAN-AdamW}.
$$

这意味着 v6.0 的主线不再是“继续调 functional optimizer”，也不再是“继续让 Dense AB-RBF 变快”。Dense RBF / Dense AB-RBF 已经完成 mechanism reference 的使命：它证明了 PureKAN edge 需要 base + residual decomposition；但 Dense edge-wise basis expansion 的复杂度天然太高，不能作为最终高效系统。

v6.0 的目标是回答一个更根本的问题：

$$
\boxed{
\text{是否存在一个 strict PureKAN primitive，既接近 MLP 的计算/显存，又能承载 functional geometry？}
}
$$

只有这个问题先得到正答案，functional update / LightSmooth / GA-FU 才值得重新进入主线。

---

## 1. v5.9 结果复盘与当前判断

### 1.1 v5.9 的正向进展

v5.9 并不是无意义失败。它做对了三件事。

第一，P0 core manifest 通过，说明新增候选仍然保持 strict PureKAN：

```text
nonKAN = 0
edge coverage = 1.0
rollback = 0
```

第二，DWM2-lite 相比 Dense AB-RBF 和旧 DWM2 明显减少了保存的中间张量。例如 Dense AB-RBF 的 saved tensor volume 约为 90 MB，而 DWM2-lite-RBFK2 约为 18 MB。这说明 DWM2-lite 的方向确实在降低 memory pressure。

第三，v5.9 引入了 saved tensor audit，这非常重要。我们现在终于不只是知道“显存爆了”，而是可以定位 backward 到底保存了哪些 tensor。

### 1.2 v5.9 的失败点

v5.9 P1 没有任何 exploratory survivor。DWM2-lite-RBFK2 是最近的候选，但仍然大约是：

```text
forward ratio ≈ 4.0x MLP
backward ratio ≈ 2.8x MLP
backward memory ratio ≈ 1.9x MLP
step ratio ≈ 2.6x MLP
```

失败统计中，backward memory fail 是主导项，其次是 forward time fail。P2 custom backward / recompute 没有运行，因为 P1 没有 survivor。

这个 gating 逻辑需要修正：

$$
\boxed{
\text{如果 P1 因 backward memory 失败，那么 P2 custom backward 不应该被 gate 掉。}
}
$$

custom backward 正是用来解决 backward memory 的；因此 v5.9 最大的问题不是“没有继续跑”，而是 **决策流程把可能解决 P1 失败的实验挡在 P1 后面**。

### 1.3 当前是否在正确道路上？

答案是：**方向总体是对的，但 v5.9 之后必须换一个更根本的策略。**

正确的部分是：

```text
kernel-first 是对的；
functional optimizer 必须 gated behind efficient primitive 是对的；
Dense AB-RBF 只能当 mechanism reference 是对的；
saved tensor audit 是必须的。
```

需要改变的部分是：

```text
不能继续只围绕 DWM2-lite-RBFK2 小修；
不能让 P1 no survivor 阻止 custom backward 诊断；
不能只测 primitive 效率而不测 accelerated convergence；
不能只靠 RBF/LUT/Rational 三个旧候选。
```

v6.0 的判断是：

$$
\boxed{
\text{我们需要从 “efficient primitive search” 升级为 “efficient primitive + custom backward + accelerated convergence co-design”。}
}
$$

---

## 2. 当前核心假设

v6.0 不再只测一个假设，而是系统验证多组可能性。每个假设都必须有明确实验来证伪。

### H1：P1 失败部分来自 profiler / kernel path，而不是理论复杂度

DWM2-lite 理论上应该接近：

$$
O(BdK)+O(Bdh),
$$

其中 $O(Bdh)$ 是 GEMM 主项。若实现真正 GEMM-native，它不该比 MLP 慢 4 倍以上。因此需要确认：

```text
是否仍有大量小 kernel launch？
是否 basis / LUT / rational 部分没有 fuse？
是否 torch.compile 的 warm path 没被正确测量？
是否 batch/hidden 太小导致 launch overhead 主导？
是否 profiler 把 optimizer 或 sync 混入 forward？
```

### H2：backward memory 失败主要来自 autograd saved tensors，而不是模型本身

如果 autograd 保存了完整 basis、interp weights、temporary basis tensor 或 gather outputs，那么 memory ratio 会很高。custom backward / recomputation / streaming stats 应该能显著改善：

$$
M_{backward,KAN}^{custom} < M_{backward,KAN}^{autograd}.
$$

如果 custom backward 后 memory 仍然高，说明 memory 来自 workspace / GEMM / optimizer states，而不是 saved tensors。

### H3：RBF / LUT 的 channel-wise residual 仍然太慢，必须转向 sparse interpolation primitive

RBF 要计算 exp，LUT 如果实现为 dense interpolation 或保存完整 table activation，也会慢。真正接近 MLP 的一维 KAN primitive 应该是 sparse two-bin interpolation：

$$
q = \left\lfloor \frac{x-x_{min}}{\Delta} \right\rfloor,
$$

$$
w = \frac{x-x_q}{\Delta},
$$

$$
r(x)=(1-w)v_q + wv_{q+1}.
$$

该 residual 的 forward 是 $O(Bd)$，backward 只需要保存 $q,w$ 或重算它们，不需要保存 $[B,d,K]$。

### H4：Rational/KAT 仍然是最终高效候选，但目前 recipe 不对

Rational/KAT 的计算形态最接近 MLP：

$$
x \rightarrow r(x) \rightarrow Wx,
$$

主计算仍然是 GEMM。它之前的问题是 accuracy recipe 弱、functional metric 不干净，而不是理论效率不行。因此 v6.0 要重新测试：

```text
identity-like rational init
small residual rational branch
better group size
better denominator damping
better output scale
accelerated optimizer recipe
```

### H5：收敛速度不能靠 AdamW 默认 recipe，需要显式验证加速优化器

终极目标要求“收敛速度快”。因此 v6.0 不能只比较最终 accuracy，也必须比较：

$$
T_{target},
$$

即达到某个 validation loss / accuracy target 所需 step 和 wall-clock time。

需要测试：

```text
AdamW
NesterovAdamW
AdanLite
WinAdamW
Lookahead-AdamW
restart-on-loss-spike
```

这些优化器只作用于 edge-system 参数，不引入 non-KAN 参数。

### H6：早期 geometry gate 太严会阻碍 learning，geometry 应 late consolidation

已有结果多次显示：early task learning 可能需要更粗糙的函数。v6.0 中 geometry 不再作为 early hard gate，而是分阶段：

```text
early: 关注 task loss, rank, margin, time-to-target
middle: 控制 residual geometry 不爆
late: 做 LightSmooth / residual smoothing consolidation
```

### H7：Dense AB-RBF 仍应作为 teacher/reference，而不是删除

Dense AB-RBF 虽然不是终极高效系统，但它证明了 edge decomposition 的有效性。它应继续作为：

```text
mechanism reference
accuracy teacher
function displacement teacher
geometry teacher
```

但它不能进入最终 efficiency claim。

---

## 3. v6.0 候选方法

v6.0 将候选分为四类：reference、repair、new primitive、optimizer/recipe。

### 3.1 Reference methods

这些不一定满足终极目标，但必须保留用于判断方向。

```text
MLP-AdamW
MLP-AdanLite
PureKAN-RBFOnly-Dense-AdamW
PureKAN-Dense-ABRBF-AdamW
DWM2-lite-RBFK2-autograd
RationalKAT-lite-autograd
```

### 3.2 Repair candidates

这些是对已有 v5.9 候选的修复。

```text
DWM2-lite-RBFK2-customBackward-recompute
DWM2-lite-RBFK2-customBackward-streaming
DWM2-lite-LUTK8-customBackward-indexOnly
RationalKAT-lite-customBackward
RationalKAT-lite-compiled-warm
```

### 3.3 New efficient primitive candidates

#### A. SparseInterpKAN

这是 v6.0 的新增核心候选。形式为：

$$
z_i = a_i x_i + b_i + s_i r_i(x_i),
$$

其中：

$$
r_i(x_i)=(1-w_i)v_{i,q_i}+w_i v_{i,q_i+1}.
$$

然后：

$$
y = Wz.
$$

它的计算复杂度：

$$
O(Bd)+O(Bdh),
$$

不会构造 dense basis tensor。

#### B. SparseSplineKAN

SparseInterpKAN 的二阶版本。使用 piecewise linear 或 cubic Hermite interpolation，但仍然必须保存稀疏 index，而不是 one-hot basis。

#### C. RationalKAT-AB-v3

Rational residual + base path + GEMM mixing。它强调 identity-like initialization：

$$
r(x) \approx x + \epsilon \psi(x),
$$

其中 $\epsilon$ 初始很小，防止 early training destabilization。

#### D. CP-ABRBF-light

CP-ABRBF 保留作为表达力 reference。只有当 rank-speed monotonicity 和 memory 明确改善时，才继续进入 task gate。

### 3.4 Accelerated optimizer candidates

这些用于测试收敛速度，不是替代 primitive。

```text
AdamW
NesterovAdamW
AdanLite
WinAdamW
Lookahead-AdamW
AdamW + restart-on-val-loss
AdanLite + late LightSmooth
```

---

## 4. 实验总流程

v6.0 分为十个阶段。不同于 v5.9，P2 custom backward 不再完全 gated by P1 survivor，而是对 P1 的 top-N failure candidates 强制执行。

```text
P0: Core / manifest / strict PureKAN invariants
P1: Profiler truth, roofline and size-scaling audit
P2: Custom backward / recompute / streaming mandatory audit
P3: SparseInterpKAN / SparseSplineKAN primitive implementation and efficiency test
P4: RationalKAT-v3 recipe and efficiency repair
P5: Accelerated convergence recipe audit
P6: Joint task-efficiency-convergence gate
P7: Geometry / LightSmooth compatibility for survivors
P8: Functional training smoke for survivors only
P9: 3-seed candidate selection
P10: 5-seed and 10-seed final confirmation
```

---

## 5. P0：Core / Manifest / Strict PureKAN Invariants

### 5.1 目的

确认所有候选都是 strict PureKAN，不引入普通 MLP stem/head/LN 可学习参数，并且参数发现函数能正确区分：

```text
edge params
base params
residual params
mixing params
functional params
optimizer params
```

### 5.2 必跑方法

```text
MLP-reference
Dense-ABRBF-reference
DWM2-lite-RBFK2
DWM2-lite-LUTK8
SparseInterpKAN-K8
SparseInterpKAN-K16
SparseSplineKAN-K8
RationalKAT-AB-v3-g8
RationalKAT-AB-v3-g16
CP-ABRBF-light-r4
```

### 5.3 记录指标

```text
method
primitive_class
edge_param_count
base_param_count
residual_param_count
mixing_param_count
nonKAN_param_count
functional_coverage
edge_coverage
base_coverage
residual_coverage
mixing_coverage
rollback_max_abs_error
parameter_dtype
trainable_center_count
trainable_width_count
has_dense_basis_tensor
has_custom_backward
```

### 5.4 通过标准

strict PureKAN 候选必须满足：

$$
nonKAN=0,
$$

$$
coverage=1.0,
$$

$$
rollback\_error < 10^{-8}.
$$

如果某个候选需要 learnable normalization、普通 Linear head 或普通 MLP stem 才能运行，则降级为 diagnostic，不允许进入终极系统 claim。

---

## 6. P1：Profiler Truth, Roofline and Size-Scaling Audit

### 6.1 目的

判断当前效率失败到底是：

```text
kernel launch overhead
basis construction
GEMM cost
saved tensor memory
workspace memory
autograd overhead
compile/cold-start artifact
```

还是 primitive 理论复杂度本身不可接受。

### 6.2 Profiling grid

每个候选必须在不同规模下测试：

```text
batch_size = 64, 128, 256, 512
hidden_dim = 64, 128, 256
input_dim = 784
num_classes = 10
precision = fp32, bf16 optional
backend = eager, compile-warm, custom if available
```

### 6.3 记录指标

#### 时间

```text
cold_forward_ms
warm_forward_ms
backward_ms
optimizer_ms
step_ms
compile_time_ms
sync_count
```

#### 显存

```text
forward_peak_allocated_mb
backward_peak_allocated_mb
peak_reserved_mb
saved_tensor_total_mb
saved_tensor_count
largest_saved_tensor_mb
workspace_temp_mb
optimizer_state_mb
activation_saved_mb
```

#### kernel breakdown

```text
kernel_count
num_gemm_calls
num_einsum_calls
num_exp_calls
num_gather_calls
num_scatter_add_calls
num_custom_kernel_calls
graph_break_count
recompile_count
```

#### roofline proxy

```text
estimated_flops
estimated_memory_bytes
arithmetic_intensity
achieved_gflops_per_sec
bandwidth_gb_per_sec
```

### 6.4 P1 exploratory gate

P1 不再要求直接达到 final gate。探索 gate 为：

$$
T_{forward}/T_{MLP} \le 2.5,
$$

$$
T_{backward}/T_{MLP} \le 2.5,
$$

$$
M_{backward}/M_{MLP} \le 2.0.
$$

如果候选没有通过探索 gate，但它是某类路线的 closest candidate，仍然进入 P2 custom backward audit。

### 6.5 可视化

必须画：

```text
cold vs warm timing bar
forward/backward/optimizer time stacked bar
memory decomposition stacked bar
saved tensor shape top-10 table
kernel count heatmap
roofline proxy scatter
size scaling curve: batch_size vs time ratio
size scaling curve: hidden_dim vs time ratio
```

---

## 7. P2：Custom Backward / Recompute / Streaming Mandatory Audit

### 7.1 目的

修正 v5.9 的 gating 问题。P2 不是 P1 survivor 的后续，而是 **P1 memory failure 的必做诊断**。

### 7.2 必跑候选

不论 P1 是否通过，以下 top-N 进入 P2：

```text
DWM2-lite-RBFK2
DWM2-lite-LUTK8
SparseInterpKAN-K8
SparseSplineKAN-K8
RationalKAT-AB-v3-g8
```

### 7.3 自定义 backward 版本

每个候选至少比较：

```text
autograd
recompute backward
streaming coeff-gradient backward
index-only backward, for sparse interpolation
fused forward + analytic backward, if available
```

### 7.4 正确性指标

```text
grad_rel_error
param_grad_cosine
input_grad_cosine
max_abs_grad_error
finite_grad_rate
```

要求：

$$
grad\_rel\_error < 10^{-4},
$$

$$
cos(grad_{custom}, grad_{autograd}) > 0.999.
$$

### 7.5 效率指标

```text
backward_time_ratio_vs_mlp
backward_memory_ratio_vs_mlp
saved_tensor_total_mb
saved_tensor_total_mb_reduction
workspace_temp_mb
step_time_ratio
```

### 7.6 P2 gate

探索 gate：

$$
M_{backward}/M_{MLP} \le 1.2,
$$

$$
T_{backward}/T_{MLP} \le 2.0.
$$

强 gate：

$$
M_{backward}/M_{MLP} \le 0.8,
$$

$$
T_{backward}/T_{MLP} \le 1.4.
$$

### 7.7 可视化

```text
autograd vs custom backward memory bar
grad correctness scatter
saved tensor reduction waterfall
backward memory vs backward time Pareto
```

---

## 8. P3：SparseInterpKAN / SparseSplineKAN 新 primitive 验证

### 8.1 目的

测试一个真正避免 dense basis expansion 的 KAN primitive。v5.9 的 DWM2-lite 仍然没有通过 forward/memory envelope，因此 v6.0 必须引入新的稀疏插值式 primitive。

### 8.2 SparseInterpKAN 公式

对每个 channel：

$$
z_i = a_i x_i + b_i + s_i r_i(x_i),
$$

其中：

$$
r_i(x_i) = (1-w_i)v_{i,q_i}+w_i v_{i,q_i+1}.
$$

再做 edge-system mixing：

$$
y = Wz.
$$

这里 $W$ 是 KAN edge system 的 mixing parameter，不是 ordinary non-KAN Linear head。

### 8.3 关键实现约束

```text
不构造 one-hot basis
不构造 [B,d,K] dense tensor
保存 q,w 或重算 q,w
v_i 通过 scatter_add 获得梯度
mixing 使用单 GEMM
```

### 8.4 候选配置

```text
SparseInterp-K8-linearBase
SparseInterp-K16-linearBase
SparseInterp-K8-linear+siluBase
SparseInterp-K16-linear+siluBase
SparseSpline-K8-cubicLite
SparseSpline-K16-cubicLite
```

### 8.5 记录指标

```text
same efficiency metrics as P1/P2
interp_bin_occupancy_entropy
empty_bin_fraction
out_of_grid_fraction
residual_norm
base_norm
residual_over_base
function_total_variation
function_second_difference_energy
```

### 8.6 P3 gate

效率探索 gate：

$$
T_{forward}/T_{MLP} \le 1.8,
$$

$$
M_{backward}/M_{MLP} \le 1.0.
$$

任务 smoke gate：

```text
single-seed 10-epoch acc gap vs MLP <= 5 percentage points
ECE not worse than MLP by more than 0.10
loss curve finite and decreasing
```

---

## 9. P4：RationalKAT-v3 Recipe Repair

### 9.1 目的

Rational/KAT 仍然是最可能接近 MLP compute 的 primitive，但之前 recipe 弱。P4 只修 AdamW / accelerated recipe，不做 functional update。

### 9.2 候选配置

```text
groups = 4, 8, 16, 32
mode = identity_rational, linear_silu, swish, gelu_like
residual_scale_init = 0.05, 0.10, 0.20
mixing_init = identity_like, orthogonal, xavier
learning_rate = 1e-3, 2e-3, 3e-3
optimizer = AdamW, AdanLite, WinAdamW
```

### 9.3 记录指标

```text
accuracy
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
group_function_diversity
mixing_weight_norm
residual_scale
```

### 9.4 可视化

```text
recipe heatmap: groups x residual_scale
rational denominator safety curve
accuracy vs denominator condition scatter
ECE vs r_prime_p95 scatter
```

### 9.5 P4 gate

必须满足：

$$
Acc_{RK} \ge Acc_{MLP} - 0.02,
$$

$$
T_{step,RK}/T_{step,MLP} \le 1.5,
$$

$$
M_{backward,RK}/M_{backward,MLP} \le 1.0.
$$

---

## 10. P5：Accelerated Convergence Recipe Audit

### 10.1 目的

终极目标包含“收敛速度快”，因此每个高效 primitive 必须验证加速收敛，而不是只验证最终 accuracy。

### 10.2 候选 optimizer

```text
AdamW
NesterovAdamW
AdanLite
WinAdamW
Lookahead-AdamW
AdamW + gradient restart
AdanLite + restart
```

### 10.3 收敛指标

#### step-based

```text
steps_to_90pct_final_acc
steps_to_target_val_loss
epochs_to_target_val_loss
val_loss_auc_step
val_acc_auc_step
early_loss_slope_0_20
early_loss_slope_20_100
```

#### wall-clock

```text
time_to_target_val_loss_sec
time_to_target_acc_sec
val_loss_auc_time
samples_per_second
```

#### optimizer dynamics

```text
update_norm
update_over_param_norm
momentum_norm
rms_norm
restart_count
restart_reason
gradient_noise_scale
cos_update_grad
cos_update_prev_update
```

### 10.4 Target definitions

For each dataset, define target using MLP-AdamW baseline:

$$
L_{target}=1.05 L_{MLP-final}.
$$

And accuracy target:

$$
Acc_{target}=Acc_{MLP-final}-0.01.
$$

### 10.5 P5 gate

Candidate passes convergence gate if:

$$
time\_to\_target_{KAN} \le 1.25 \cdot time\_to\_target_{MLP},
$$

and:

$$
val\_loss\_auc\_time(KAN) \le val\_loss\_auc\_time(MLP).
$$

### 10.6 可视化

```text
validation loss vs step
validation loss vs wall-clock
accuracy vs wall-clock
time-to-target bar
optimizer state norm trace
restart marker plot
```

---

## 11. P6：Joint Task-Efficiency-Convergence Gate

### 11.1 目的

只让真正有希望满足终极目标的候选进入 geometry / functional update。

### 11.2 必须对比方法

```text
MLP-AdamW
MLP-AdanLite
Dense-ABRBF-AdamW reference
DWM2-lite best
SparseInterp best
SparseSpline best
RationalKAT-v3 best
CP-light best, optional
```

### 11.3 通过标准

候选必须同时满足：

$$
T_{forward}/T_{MLP} \le 1.5,
$$

$$
T_{backward}/T_{MLP} \le 1.8,
$$

$$
M_{backward}/M_{MLP} \le 1.0,
$$

$$
Acc_{KAN} \ge Acc_{MLP} - 0.02,
$$

$$
ECE_{KAN} \le ECE_{MLP} + 0.05,
$$

$$
time\_to\_target_{KAN} \le 1.5 \cdot time\_to\_target_{MLP}.
$$

如果没有候选通过 P6，则不进入 P7-P10，并且结论是：

```text
当前 strict PureKAN high-efficiency primitive 未成立；
需要重新设计 primitive，而不是 functional optimizer。
```

---

## 12. P7：Geometry / LightSmooth Compatibility for Survivors

### 12.1 目的

只有 P6 survivor 才进入 geometry maintenance。LightSmooth 不再作为主 optimizer，只是低频 residual geometry maintenance。

### 12.2 实验

```text
no smoothing
single LightSmooth event
one-cycle train -> smooth -> refresh
late-only LightSmooth
```

### 12.3 指标

```text
acc_drop
KL_teacher_student
logit_drift
hidden_sketch_drift optional
geometry_reduction
geometry_retention_after_refresh
refresh_recovery_acc
refresh_recovery_loss
smoothing_memory_ratio
smoothing_time_overhead
```

### 12.4 Gate

$$
acc\_drop \le 0.005,
$$

$$
KL \le 0.005,
$$

$$
geometry\_reduction \ge 0.10,
$$

$$
amortized\_time\_overhead \le 0.10.
$$

---

## 13. P8：Functional Training Smoke for Survivors Only

### 13.1 目的

验证 efficient primitive 上是否能重新引入 functional update。

### 13.2 方法

```text
AdamW / accelerated optimizer baseline
Functional-coordinate Adam
Residual-only functional update
Late-phase functional smoothing
Functional update + restart
```

### 13.3 指标

```text
same task/convergence/efficiency metrics
functional_direction_cos_with_adam
function_displacement_R2
residual_geometry_reduction
bad_step_rate
fallback_rate
```

### 13.4 Gate

候选必须比 accelerated AdamW baseline 至少满足一项：

```text
same accuracy, better geometry
same geometry, faster time-to-target
better ECE with no accuracy drop
```

否则 functional update 不进入 confirm。

---

## 14. P9：3-seed Candidate Selection

### 14.1 方法

对 P6-P8 survivors 跑：

```text
seeds = 0,1,2
datasets = MNIST, Fashion-MNIST, KMNIST
```

### 14.2 必须保留 baseline

```text
MLP-AdamW
MLP-best-accelerated
PureKAN-RBFOnly-AdamW
Dense-ABRBF-AdamW reference
best efficient primitive AdamW/accelerated
best efficient primitive + LightSmooth
best efficient primitive + functional update, if any
```

### 14.3 选择标准

需要同时满足：

```text
mean accuracy >= MLP-AdamW - 1%
val_loss_auc_time <= MLP-AdamW
ECE <= MLP-AdamW + 0.03
backward memory <= MLP
step time <= 1.5x MLP
```

---

## 15. P10：5-seed / 10-seed Final Confirm

### 15.1 5-seed confirm

扩展到：

```text
seeds = 0..4
```

验证：

```text
paired accuracy delta
paired time-to-target delta
paired memory ratio
paired ECE delta
paired geometry delta
```

### 15.2 10-seed confirm

只有 5-seed 同时过 task / efficiency / convergence / geometry gate，才跑：

```text
seeds = 0..9
```

### 15.3 Final success definition

最终成功必须满足：

$$
Acc_{KAN} \ge Acc_{MLP-AdamW},
$$

$$
AUC_{time,KAN} \le AUC_{time,MLP-AdamW},
$$

$$
ECE_{KAN} \le ECE_{MLP-AdamW},
$$

$$
M_{backward,KAN} \le 0.8M_{backward,MLP},
$$

$$
T_{step,KAN} \le 1.25T_{step,MLP},
$$

and geometry improves:

$$
GeometryScore_{KAN} > GeometryScore_{MLP/PureKAN-AdamW}.
$$

---

## 16. 总指标表

v6.0 所有实验必须统一记录以下指标。

### 16.1 Task metrics

```text
test_acc
val_acc
train_acc
val_loss
test_loss
val_loss_auc_step
val_loss_auc_time
NLL
ECE
Brier score
margin_mean
margin_p10
classwise_accuracy
```

### 16.2 Convergence metrics

```text
steps_to_target_loss
time_to_target_loss
epochs_to_target_loss
steps_to_target_acc
time_to_target_acc
early_loss_slope
mid_loss_slope
late_loss_slope
bad_step_rate
loss_spike_count
restart_count
```

### 16.3 Efficiency metrics

```text
forward_time_ms
backward_time_ms
optimizer_time_ms
step_time_ms
samples_per_second
forward_peak_memory_mb
backward_peak_memory_mb
peak_reserved_mb
backward_memory_ratio_vs_mlp
step_time_ratio_vs_mlp
activation_saved_bytes
saved_tensor_total_bytes
largest_saved_tensor_bytes
workspace_temp_bytes
optimizer_state_bytes
```

### 16.4 Kernel metrics

```text
kernel_count
num_gemm_calls
num_exp_calls
num_gather_calls
num_scatter_add_calls
num_custom_kernel_calls
graph_break_count
recompile_count
compile_time_ms
cold_warm_ratio
```

### 16.5 KAN structure metrics

```text
edge_param_count
base_param_count
residual_param_count
mixing_param_count
base_norm
residual_norm
residual_over_base
base_ablation_drop
residual_ablation_drop
function_diversity
bin_occupancy_entropy
out_of_grid_fraction
```

### 16.6 Geometry metrics

For RBF / AB-RBF:

```text
phi_rbf_p95
curvature_rbf
sobolev_rbf_norm
phi_total_p95
jacobian_condition
high_sobolev_eigen_energy
```

For SparseInterp / SparseSpline:

```text
total_variation
second_difference_energy
slope_p95
curvature_proxy
bin_jump_p95
```

For Rational/KAT:

```text
r_prime_p95
r_double_prime_p95
denominator_min
denominator_p01
denominator_condition
group_function_diversity
```

### 16.7 Functional update metrics

```text
functional_direction_cos_with_adam
function_displacement_R2
bad_functional_step_rate
fallback_rate
trust_clip_rate
accepted_smoothing_events
rejected_smoothing_events
geometry_retention_after_refresh
```

---

## 17. 必须生成的可视化

### 17.1 Efficiency dashboard

```text
forward/backward/optimizer time stacked bar
backward memory decomposition stacked bar
saved tensor top-k shapes table
kernel count heatmap
cold vs warm timing plot
```

### 17.2 Task-efficiency Pareto

横轴：

$$
step\_time\_ratio
$$

纵轴：

$$
test\_acc
$$

点大小：

$$
backward\_memory\_ratio
$$

颜色：primitive family。

### 17.3 Convergence plots

```text
val loss vs step
val loss vs wall-clock
accuracy vs wall-clock
time-to-target bar
loss slope comparison
```

### 17.4 Geometry-efficiency Pareto

横轴：

$$
backward\_memory\_ratio
$$

纵轴：

$$
geometry\_improvement
$$

点大小：accuracy drop。

### 17.5 Primitive diagnostics

```text
SparseInterp bin occupancy heatmap
Rational denominator safety curve
base/residual norm trajectory
ablation drop trajectory
feature rank trajectory
classwise accuracy heatmap
```

### 17.6 Gate dashboard

每个候选生成四栏：

```text
task gate
efficiency gate
convergence gate
geometry gate
```

每栏标记 pass / fail / near-pass，并给出 failure reason。

---

## 18. 失败后的决策规则

### Case A：没有 P1/P2 efficiency candidate

结论：

```text
当前 primitive family 不满足终极目标。
停止 optimizer / LightSmooth。
继续设计 primitive / kernel。
```

### Case B：有 efficiency candidate，但 task recipe 失败

结论：

```text
先修 recipe / initialization / optimization。
不进入 functional update。
```

### Case C：有 task + efficiency candidate，但 convergence 慢

结论：

```text
优先研究 AdanLite / WinAdamW / restart / schedule。
不先做 geometry smoothing。
```

### Case D：task + efficiency + convergence 都过，但 geometry 不够

结论：

```text
进入 LightSmooth / functional geometry maintenance。
```

### Case E：全部通过

结论：

```text
进入 5-seed / 10-seed confirm，并开始 CIFAR-small precheck。
```

---

## 19. v6.0 预期最可能成功的路线

目前我认为最可能的优先级是：

```text
1. SparseInterpKAN / SparseSplineKAN with custom backward
2. RationalKAT-AB-v3 with better recipe and accelerated optimizer
3. DWM2-lite-RBFK2 with custom backward, as repair path
4. CP-ABRBF-light as accuracy reference, not final efficiency route
```

最重要的 pivot 是：

$$
\boxed{
\text{不要再试图让 Dense RBF 变快；要让 KAN primitive 本身变成 sparse / GEMM-native / custom-backward friendly。}
}
$$

---

## 20. 最终总结

v5.9 的失败不意味着我们不在正确道路上。它说明我们已经把问题从“functional update 怎么调”推进到了更根本的系统问题：

$$
\boxed{
\text{有没有一个 strict PureKAN primitive 能同时满足 efficiency、task learning、convergence、geometry？}
}
$$

v6.0 的计划因此更全面：它不只验证 primitive 效率，也验证 custom backward、加速收敛、任务 recipe、geometry compatibility 和最终多 seed 稳定性。

只要没有 efficient primitive survivor，就坚决不进入 functional optimizer。只要有 efficient primitive survivor，就必须立刻验证 accelerated convergence，而不是只看最终 accuracy。

这才符合终极目标。
