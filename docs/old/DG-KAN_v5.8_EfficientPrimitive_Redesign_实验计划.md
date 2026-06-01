# DG-KAN v5.8 Efficient Primitive Redesign 实验计划

## 0. 终极目标与本轮重新定位

本阶段必须以以下终极目标作为硬约束，而不是只追求单个实验表格里的 accuracy 或 geometry：

> 构建一个无 non-KAN 参数的 PureKAN functional training system，其 forward 时间/显存接近 MLP，backward 时间接近 MLP，backward 显存低于 MLP，收敛速度快，并在 accuracy / AUC / ECE / geometry 上超过 MLP-AdamW 与 PureKAN-AdamW。

因此 v5.8 的核心原则是：

$$
\boxed{\text{没有 MLP-like primitive，就不进入 functional optimizer。}}
$$

v5.7 的结果已经很清楚：当前的 DWM、RationalKAT、CP 这些 efficient PureKAN primitive 都已经能作为 strict PureKAN core object 被审计，但没有一个 primitive 通过 task + efficiency 联合门槛。失败最多的是 backward memory，其次是 accuracy recipe，再其次是 forward/backward time。这意味着下一步不是继续调 U-FULL、LightSmooth、NFS 或 Sobolev，而是要重新设计 PureKAN primitive 的计算结构和训练 recipe。

本计划将 v5.8 定义为：

$$
\boxed{\text{Efficient Primitive First, Functional Optimizer Later.}}
$$

v5.8 不再试图让 Dense RBF / Dense AB-RBF 成为终极系统。Dense AB-RBF 继续作为 mechanism reference / teacher，因为它已经证明 edge decomposition 是正确方向；但最终候选必须是 GEMM-native、LUT-native 或 fused-native 的高效 primitive。

---

## 1. v5.7 结果的深度分析

### 1.1 P0 说明实现对象已经变干净

v5.7 的 P0 表明，以下 primitive 都能以 strict PureKAN 形式存在：

```text
RBFOnly-Dense-reference
ABRBF-Dense-reference
GEMMNativeDepthwiseMixDense
GEMMNativeRationalKATDense
GEMMNativeCPABRBFDense
```

它们的可学习参数都在 audited edge system 内，`nonKAN = 0`，rollback 为 0，graph break 记录为 0。这说明当前失败不再是因为又混入了普通 MLP stem/head/LN 参数，也不是因为 rollback 或 core manifest 没接好。

但 P0 只是说明候选对象合法，不说明它高效或有效。

### 1.2 P1 说明没有任何 primitive 达到 MLP-like efficiency

P1 的结果显示没有 exploratory survivor，也没有 final survivor。典型信号如下：

```text
RBFOnly-Dense-reference:
  forward ≈ 3.60x MLP
  backward ≈ 1.85x MLP
  backward memory ≈ 1.66x MLP
  step ≈ 1.75x MLP

DWM-gemm-native:
  forward ≈ 6.21x MLP
  backward ≈ 1.95x MLP
  backward memory ≈ 1.89x MLP
  step ≈ 2.08x MLP

DWM-warm-compiled:
  backward memory ≈ 0.13x MLP
  but forward ≈ 8.44x MLP
  step ≈ 2.61x MLP

RationalKAT-compiled:
  backward memory ≈ 0.13x MLP
  but forward ≈ 9.18x MLP
  step ≈ 2.85x MLP
```

这说明当前效率问题不是单一问题。不同路线有不同瓶颈：

```text
Dense RBF / Dense AB-RBF:
  结构性 K 维展开导致 forward/memory 都重。

DWM:
  理论上应该 GEMM-native，但当前实现仍然 forward 太慢，bmem 或 time 不稳定。

RationalKAT:
  memory path 有潜力，但 forward/time 和 accuracy recipe 没释放出来。

CP-ABRBF:
  任务信号相对较好，但 memory 和 rank-speed 行为仍然不足。
```

因此不能简单说“DWM 不行”或“RationalKAT 不行”。更准确地说：

$$
\boxed{\text{当前 primitive 的理论计算形态和实际 kernel path 尚未一致。}}
$$

### 1.3 P2 说明 DWM 的主要问题是 task recipe + time，不只是表达力

DWM 的最好 recipe 仍然有明显 task gap：

```text
Fashion-MNIST:
  best DWM ≈ 0.7682, gap ≈ 2.86 points

KMNIST:
  best DWM ≈ 0.6465, gap ≈ 5.01 points

MNIST:
  best DWM ≈ 0.8750, gap ≈ 2.80 points
```

DWM 当前 step ≈ 2.28x，backward memory ≈ 1.31x。这说明即便 DWM 被当成高效 primitive，它现在也没有达到终极目标要求：task 不够，time 不够，memory 也只是 exploratory 可接受，不是 final pass。

DWM 现在不应继续靠单纯改 `K`、`scale`、`lr`。它需要重新审视结构：

1. depthwise function 是否过于逐通道，缺少足够的 cross-channel nonlinear interaction；
2. mixing 是否只有后置 GEMM，导致非线性组合不足；
3. current channelNorm / FixedNorm 是否让输入分布对 RBF / base path 不友好；
4. DWM 的 compiled path 是否真的是 warm steady-state，而不是 Python / dispatch overhead 主导。

### 1.4 P3 说明 RationalKAT 有 memory 潜力，但 recipe 远未充分

RationalKAT 的理论优势是：elementwise/group rational activation + GEMM mixing。理想复杂度接近：

$$
O(Bd(m+n))+O(Bdh),
$$

而不是 Dense RBF 的：

$$
O(BdhK).
$$

但是 v5.7 里 RationalKAT 当前依然没有通过 P3。Fashion 接近一些，KMNIST 仍明显不足。更重要的是，compiled RationalKAT 虽然有低 backward memory 信号，但 forward time 仍然远离 MLP-like。

因此 RationalKAT 的下一步不是 functional update，而是三件事：

```text
1. kernel：确认 rational activation 是否 fused；
2. recipe：identity-like / silu-like 初始化是否充分；
3. architecture：grouping / residual scale / mixing depth 是否正确。
```

### 1.5 P4 说明 CP 仍是 accuracy reference，不是效率 final

CP-two-stage 的 accuracy 比 DWM / RationalKAT 更稳，尤其 MNIST 与 Fashion。但 CP 的 memory ratio 仍然高，rank-speed 和 rank-memory 没有形成最终的 MLP-like envelope。因此 CP 目前更像：

$$
\boxed{\text{expressivity-preserving compressed reference}}
$$

而不是终极高效实现。

CP 可以继续保留为 teacher / accuracy reference，但如果 rank4、rank8、rank16 的 speed/memory 不呈稳定单调改善，就不应该作为效率主线。

### 1.6 v5.7 的真正结论

v5.7 的最终结论不是“所有方向失败”。它说明的是：

$$
\boxed{\text{当前没有 strict PureKAN primitive 同时通过效率和任务门槛。}}
$$

这意味着：

```text
1. Dense AB-RBF 继续做 mechanism reference。
2. DWM 是当前第一优先效率路线，但需要结构和 kernel 双修。
3. RationalKAT 是最终 MLP-like 潜力路线，但 recipe 没修好。
4. CP 是保表达力的压缩 reference，但不是最终效率路线。
5. functional optimizer / LightSmooth 继续 gated，不能提前进入主线。
```

---

## 2. 代码实现层面的关键判断

### 2.1 Dense RBF path 的结构性瓶颈

基础 RBFDense 的 forward 是：

$$
B_{bik}=\exp\left(-\frac{(x_{bi}-c_k)^2}{2\sigma^2}\right),
$$

$$
y_{bo}=\sum_{i,k}B_{bik}c_{oik}.
$$

实现上需要构造：

```text
basis: [B, d_in, K]
```

然后做：

```text
einsum("bik,oik->bo")
```

所以 Dense RBF 的主复杂度是：

$$
O(Bd_{in}d_{out}K),
$$

而 MLP 是：

$$
O(Bd_{in}d_{out}).
$$

因此 Dense RBF / Dense AB-RBF 不能通过小修 kernel 成为终极系统。它们只能作为：

```text
mechanism reference
teacher model
geometry / smoothing audit platform
```

### 2.2 DepthwiseMix 的理论路径正确，但实现和 recipe 还没有对齐

DepthwiseMix 的目标是把 Dense edge-wise function：

$$
y_o=\sum_{i,k}B_k(x_i)c_{oik}
$$

改成：

$$
\tilde x_i=f_i(x_i),
$$

$$
y=W\tilde x.
$$

这样复杂度应当接近：

$$
O(BdK)+O(Bdh).
$$

如果 DWM 的实际 forward 仍然是 6x 到 8x MLP，说明当前实现没有充分吃到 GEMM-native 优势。可能原因包括：

```text
basis calculation 没有 fused；
small tensor operations 太多；
exp / nonlinear transform 没有批量化好；
mixing 前后发生了额外 layout / transpose；
torch.compile 仍有 graph guard / dispatch overhead；
benchmark 使用的小维度让 kernel launch overhead 主导。
```

v5.8 必须把 DWM 拆成可测组件：

```text
basis / channel function time
mixing GEMM time
normalization time
Python overhead
kernel launch count
activation save memory
```

### 2.3 RationalKAT 不能继续用“当前 recipe”判死刑

RationalKAT 理论上最接近 MLP，因为主计算是：

```text
elementwise/group rational activation + GEMM
```

但当前 recipe 的 accuracy gap 大，说明它可能缺：

```text
identity-like initialization
base + rational residual decomposition
group count / group sharing balance
rational denominator damping
small residual scale warmup
post-activation mixing depth
teacher distillation
```

RationalKAT 的 functional metric 之前不干净，因此 v5.8 只做 AdamW / AdamW-like task recipe repair，不进入 functional update。

### 2.4 CP-ABRBF 的低秩计算图仍需验证

CP 的核心是：

$$
c_{oik}=\sum_{r=1}^R U_{or}V_{ir}W_{kr}.
$$

如果实现正确，rank 越低应该越快、越省内存。v5.7 里 CP 仍然被 memory 阻断，说明要硬查：

```text
rank4 是否真的比 rank8 / rank16 快；
是否仍然 materialize 了 [B,d,K] 或 [B,d,R,K]；
是否把低秩分解写成多个小 op 而不是 GEMM/bmm；
backward 是否保存了不必要的中间 tensor。
```

---

## 3. v5.8 的核心改进方向

v5.8 不再把目标写成“找到更好的 functional update”。新的目标是：

$$
\boxed{\text{先找到一个 strict PureKAN primitive，满足 task + efficiency envelope。}}
$$

只有它通过之后，才进入 functional optimizer / LightSmooth。

### 3.1 第一主线：DWM-v2，修结构而不是只修参数

DWM-v2 需要比当前 DepthwiseMix 更有表达力，同时保持 GEMM-native：

$$
h_1=W_1x,
$$

$$
\tilde h=f_{\theta}(\operatorname{FixedNorm}(h_1)),
$$

$$
y=W_2\tilde h.
$$

这里 $W_1,W_2$ 必须被定义为 KAN edge-system 的 mixing 参数，不算 non-KAN。这个结构比当前逐通道 transform 更接近 MLP 的两层表达力，同时仍然是：

$$
\text{GEMM} + \text{elementwise KAN function} + \text{GEMM}.
$$

DWM-v2 的候选包括：

```text
DWM2-A: preMix -> channel AB function -> postMix
DWM2-B: grouped preMix -> group AB function -> postMix
DWM2-C: gated channel function, y = W2( f(h) * g(h) )
DWM2-D: residual form, y = W2( h + s f(h) )
```

这条线的目标是用更合理的结构弥补当前 DWM task gap。

### 3.2 第二主线：RationalKAT-AB recipe repair

RationalKAT-AB 应该被视为最终效率候选，而不是 v5.7 的失败项。v5.8 要做 identity-like recipe repair：

$$
f(x)=x+s\cdot r_{\theta}(x),
$$

其中 $s$ 从小值 warmup：

$$
s_0\in\{0.05,0.1,0.2\}.
$$

Rational denominator 必须安全：

$$
D(x)>\epsilon.
$$

Rational derivative 也要被监控：

$$
r'_{p95},\quad r''_{p95}.
$$

这一线成功的标准不是先 geometry，而是：

```text
accuracy 接近 MLP；
step / memory 接近 MLP；
ECE 不坏；
denominator / derivative 不爆。
```

### 3.3 第三主线：LUT / Piecewise Linear KAN primitive

这是 v5.8 新增的方向。原因是 RBF 的 exp 和 K 维 dense basis 很难满足 MLP-like forward，而 RationalKAT 需要复杂 denominator 和 Triton path。Piecewise-linear / LUT KAN 可以用更简单的插值实现：

$$
f(x)=a_j + \frac{x-t_j}{t_{j+1}-t_j}(a_{j+1}-a_j),
\quad x\in[t_j,t_{j+1}].
$$

如果采用 channel-wise / grouped LUT + GEMM mixing，复杂度是：

$$
O(Bd)+O(Bdh),
$$

而不是：

$$
O(BdhK).
$$

LUT-KAN 的优势是：

```text
forward 不需要 exp；
backward 可以极低显存，只保存 bin index 或重算；
函数几何可以通过一阶/二阶 finite difference 正则；
functional smoothing 可以变成 cheap difference smoothing。
```

这条路线可能比 RBF 更接近终极目标。

### 3.4 第四主线：benchmark protocol 修正

当前 MLP baseline 很小，很多 primitive 的 kernel overhead 会被放大。因此 v5.8 必须同时跑两个 benchmark：

```text
Small-task benchmark:
  与当前 Fashion / KMNIST / MNIST 设置一致，看真实任务表现。

Kernel-scale benchmark:
  B 更大，hidden_dim 更大，看 steady-state GEMM-native 路线是否真的接近 MLP。
```

否则小模型上 2x-3x 的 overhead 可能来自 kernel launch，而不是理论复杂度；反过来，大模型上如果仍然慢，就说明 primitive 真的不合格。

---

## 4. v5.8 实验阶段

## P0: Core implementation and manifest hardening

P0 的目标是确保所有候选 primitive 都是 strict PureKAN，并且真正属于 core-level 实现，而不是 runner-only helper。

必须检查：

```text
primitive_class
edge_param_count
base_param_count
residual_param_count
mixing_param_count
nonKAN_param_count
edge_coverage
mixing_coverage
rollback_error
graph_break_count
recompile_count
```

P0 方法：

```text
MLP-reference
Dense-ABRBF-reference
DWM-v1-current
DWM2-prePostMix
DWM2-gated
DWM2-residual
RationalKAT-AB-v2
LUTKAN-AB
CP-ABRBF-reference
```

P0 通过条件：

```text
nonKAN_param_count = 0
edge_coverage = 1.0
mixing_coverage = 1.0, if mixing exists
rollback_error < 1e-8
no graph break in declared compiled path
```

如果 P0 不通过，该 primitive 不进入任何性能或任务实验。

---

## P1: Component-level efficiency decomposition

P1 不看 accuracy，只测 primitive 的组成成本。每个 primitive 必须拆成：

```text
normalization time
basis / activation time
mixing GEMM time
post-processing time
backward activation grad time
backward param grad time
optimizer step time
```

记录：

```text
forward_time_ms
backward_time_ms
optimizer_time_ms
step_time_ms
peak_allocated_mb
peak_reserved_mb
activation_saved_bytes
workspace_temp_bytes
basis_tensor_bytes
kernel_count
gemm_count
exp_count
index_select_count
gather_count
compile_time_ms
cold_forward_ms
warm_forward_ms
recompile_count
graph_break_count
```

P1 benchmark 设置：

```text
Small:
  B = 256
  hidden_dim = 64
  depth = 4

Medium:
  B = 512
  hidden_dim = 128
  depth = 4

Large:
  B = 1024
  hidden_dim = 256
  depth = 4
```

P1 通过分两级。

Exploratory gate：

$$
T_{fwd}/T_{MLP}\leq 2.0,
$$

$$
T_{bwd}/T_{MLP}\leq 2.0,
$$

$$
M_{bwd}/M_{MLP}\leq 1.25.
$$

Final efficiency gate：

$$
T_{fwd}/T_{MLP}\leq 1.25,
$$

$$
T_{bwd}/T_{MLP}\leq 1.40,
$$

$$
M_{bwd}/M_{MLP}\leq 0.80.
$$

如果没有任何 candidate 通过 exploratory gate，就不进入 P2/P3 accuracy recipe，先回代码实现。

---

## P2: DWM-v2 recipe repair

P2 只跑 P1 通过 exploratory 的 DWM-v2 变体。

候选：

```text
DWM2-prePostMix-K4-scale0.1
DWM2-prePostMix-K8-scale0.1
DWM2-prePostMix-K8-scale0.3
DWM2-gated-K8-scale0.1
DWM2-gated-K8-scale0.3
DWM2-residual-K8-scale0.1
DWM2-residual-K12-scale0.1
```

训练配置：

```text
optimizer = AdamW
lr in {1e-3, 2e-3, 3e-3}
weight_decay in {1e-4, 3e-4}
residual_scale warmup in {none, linear 20%, cosine 20%}
FixedNorm / channelNorm / edgeRMSNorm diagnostic
seeds = 0,1,2
```

记录任务指标：

```text
test_acc
val_loss
val_auc
ECE
NLL
margin_mean
margin_p10
feature_effective_rank
class_centroid_separation
classwise_accuracy
```

记录结构指标：

```text
preMix_norm
postMix_norm
channel_function_norm
gate_mean, if gated
gate_saturation_fraction
base_over_residual
activation_input_p01/p50/p99
activation_output_p01/p50/p99
```

P2 通过条件：

```text
Fashion / MNIST gap vs MLP <= 1.0 point
KMNIST gap vs MLP <= 2.0 points
ECE <= MLP + 0.03
P1 exploratory efficiency still holds under training config
```

---

## P3: RationalKAT-AB-v2 recipe repair

P3 的目标是先让 RationalKAT-AB 在 AdamW 下接近 MLP，再谈 functional update。

候选结构：

```text
RK2-identityResidual:
  f(x)=x+s*r(x)

RK2-basePlusRational:
  f(x)=a*x+b*silu(x)+s*r(x)

RK2-gatedRational:
  f(x)=x+s*g(x)*r(x)
```

搜索变量：

```text
groups in {4, 8, 16, 32}
mode in {silu, gelu_like, tanh_silu}
residual_scale_init in {0.05, 0.1, 0.2}
residual_scale_warmup in {10%, 20%, 40%}
denominator_damping in {1e-3, 1e-2, 1e-1}
lr in {1e-3, 2e-3, 3e-3}
```

必须记录 Rational 安全指标：

```text
den_actual_min_batch
den_actual_p01_batch
den_actual_condition_batch
den_actual_min_grid
den_actual_p01_grid
r_prime_p95
r_double_prime_p95
group_function_diversity
rational_residual_norm
base_over_rational
```

P3 通过条件：

```text
no denominator safety failure
Fashion / MNIST gap vs MLP <= 1 point
KMNIST gap vs MLP <= 2 points
ECE <= MLP + 0.03
P1 exploratory efficiency holds
```

---

## P4: LUT-KAN / Piecewise Linear primitive feasibility

P4 是 v5.8 新增核心实验。

LUT-KAN edge function：

$$
f(x)=a_j+\omega(a_{j+1}-a_j),
\quad \omega=\frac{x-t_j}{t_{j+1}-t_j}.
$$

候选：

```text
LUT-channelwise + GEMM
LUT-grouped + GEMM
LUT-basePlusResidual + GEMM
LUT-gated + GEMM
```

网格数：

```text
G in {8, 16, 32}
```

正则：

$$
R_1=\sum_j(a_{j+1}-a_j)^2,
$$

$$
R_2=\sum_j(a_{j+1}-2a_j+a_{j-1})^2.
$$

训练配置：

```text
AdamW first
no functional update in P4
seeds = 0,1,2
```

P4 的关键指标：

```text
LUT bin occupancy entropy
dead bin fraction
out_of_grid_fraction
slope_p95
curvature_fd_p95
activation_saved_bytes
bin_index_saved_bytes
recompute_backward_available
```

P4 通过条件：

```text
P1 exploratory efficiency pass
accuracy gap within P2/P3 bounds
backward memory <= MLP or close to MLP
no severe dead-bin collapse
```

如果 LUT-KAN 通过 P4，它将成为 v5.8 的优先 final primitive，因为它最有可能做到 low-memory backward。

---

## P5: Joint task-efficiency selection

P5 汇总 P2/P3/P4 的候选。

方法：

```text
MLP-AdamW
Dense-ABRBF-reference-AdamW
DWM2-best
RationalKAT-AB-v2-best
LUT-KAN-best
CP-ABRBF-best-reference
```

数据集：

```text
MNIST
Fashion-MNIST
KMNIST
```

seeds：

```text
0,1,2
```

P5 score：

$$
Score = AccScore + AUCScore + ECEScore + GeometryScore + EfficiencyScore.
$$

其中：

$$
EfficiencyScore=-\log\left(\frac{T_{step}}{T_{MLP}}\right)-\log\left(\frac{M_{bwd}}{M_{MLP}}\right).
$$

P5 survivor 必须满足：

```text
all datasets pass task gate
all datasets pass exploratory efficiency gate
no dataset has ECE worse than MLP by > 0.03
no dataset has backward memory ratio > 1.25
```

只有 P5 survivor 才进入 P6 custom backward。

---

## P6: Custom backward / analytic adjoint memory reduction

P6 只对 P5 survivor 做，不对失败 primitive 浪费工程资源。

候选 backward：

```text
Autograd baseline
Recompute backward
Streaming param-gradient backward
Fused backward, if implemented
```

对 DWM / RationalKAT / LUT 分别测：

```text
activation_saved_bytes
intermediate_saved_bytes
workspace_temp_bytes
param_grad_time
input_grad_time
peak_allocated_mb
peak_reserved_mb
```

梯度正确性必须满足：

$$
\frac{\|g_{custom}-g_{autograd}\|}{\|g_{autograd}\|+\epsilon}<10^{-4},
$$

$$
\cos(g_{custom},g_{autograd})>0.999.
$$

P6 通过条件：

```text
backward memory <= MLP, exploratory
backward memory <= 0.8 * MLP, final target
backward time <= 1.4 * MLP
step time <= 1.5 * MLP, exploratory
```

---

## P7: Functional / LightSmooth compatibility smoke

P7 只在 P5/P6 后运行。

目标不是立刻证明 functional optimizer，而是验证高效 primitive 能否承载低成本 geometry maintenance。

候选：

```text
LightSmooth-residual, if residual component exists
Finite-difference smoothing, for LUT
Derivative penalty projection, for RationalKAT
No smoothing baseline
```

记录：

```text
acc_drop
KL_teacher_student
logit_relative_drift
geometry_reduction
curvature_reduction
smoothing_time
smoothing_memory_ratio
amortized_step_overhead
```

P7 通过条件：

```text
acc_drop <= 0.005
KL <= 0.005
logit_drift <= 0.03
geometry_reduction >= 0.05 or curvature_reduction >= 0.20
amortized overhead <= 10%
```

---

## P8: Functional training smoke

P8 只做 smoke，不做大规模 search。

候选 functional route：

```text
Functional-coordinate Adam on efficient primitive
Residual-only functional smoothing
AdamW task + functional geometry maintenance
```

目标：

```text
确认 functional component 不破坏 efficiency envelope；
确认 geometry 有可测改善；
确认 accuracy/AUC 没明显下降。
```

P8 不通过，则保留 efficient primitive + AdamW 作为 architecture milestone，不继续 functional optimizer。

---

## P9: 5-seed confirmation

P9 只确认 P5/P6/P7/P8 之后仍然活着的候选。

方法：

```text
MLP-AdamW
Best efficient PureKAN-AdamW
Best efficient PureKAN + LightSmooth
Dense-ABRBF-reference, optional teacher only
```

数据集：

```text
MNIST
Fashion-MNIST
KMNIST
```

seeds：

```text
0..4
```

必须记录：

```text
accuracy mean/std
paired delta vs MLP
val_loss_auc
ECE
NLL
geometry metrics
step_time
backward_memory
forward_memory
```

P9 通过条件：

```text
accuracy >= MLP - 0.5 point on all datasets
AUC >= MLP or no worse by > 3%
ECE <= MLP + 0.02
step_time <= 1.5x MLP exploratory
backward_memory <= 1.0x MLP exploratory
```

---

## P10: 10-seed final confirmation

P10 只有 P9 通过才运行。

最终 claim 需要：

```text
accuracy >= MLP-AdamW and >= PureKAN-AdamW, mean over 10 seeds
AUC better than MLP-AdamW
ECE better than MLP-AdamW
geometry better than MLP-compatible baseline
forward time <= 1.25x MLP
backward time <= 1.40x MLP
backward memory <= 0.80x MLP
```

如果 task 过但 efficiency 不过，结论是：

```text
PureKAN architecture milestone, not final system.
```

如果 efficiency 过但 task 不过，结论是：

```text
kernel primitive milestone, recipe still unsolved.
```

如果两者都不过，结论是：

```text
current primitive family insufficient; move to alternative KAN primitive.
```

---

## 5. 必须记录的指标

### 5.1 任务与泛化指标

```text
test_acc
val_acc
train_acc
val_loss
test_loss
val_loss_auc
NLL
ECE
Brier score
margin_mean
margin_p10
classwise_accuracy
confusion_matrix
```

### 5.2 表示学习指标

```text
feature_effective_rank_input
feature_effective_rank_block
feature_effective_rank_output
class_centroid_separation
within_class_variance
between_class_variance
activation_mean/std/p01/p50/p99
role_update_share_input/block/output
```

### 5.3 KAN edge 指标

```text
edge_param_count
base_param_count
residual_param_count
mixing_param_count
base_over_residual_norm
base_ablation_drop
residual_ablation_drop
channel_function_norm
group_function_diversity
```

### 5.4 几何指标

```text
phi_total_p95
phi_residual_p95
curvature_total_p95
curvature_residual_p95
sobolev_residual_norm
jacobian_condition
finite_difference_slope_p95, for LUT
finite_difference_curvature_p95, for LUT
r_prime_p95, for RationalKAT
r_double_prime_p95, for RationalKAT
```

### 5.5 效率指标

```text
forward_time_ms
backward_time_ms
optimizer_time_ms
step_time_ms
forward_memory_mb
backward_memory_mb
peak_allocated_mb
peak_reserved_mb
activation_saved_bytes
basis_tensor_bytes
bin_index_saved_bytes
workspace_temp_bytes
compile_time_ms
cold_forward_ms
warm_forward_ms
cold_warm_ratio
graph_break_count
recompile_count
kernel_count
gemm_count
exp_count
gather_count
index_select_count
```

### 5.6 custom backward 正确性指标

```text
grad_rel_error
grad_cosine
param_grad_rel_error
input_grad_rel_error
finite_difference_grad_check
```

---

## 6. 必须画的图

### 6.1 Efficiency decomposition dashboard

每个 primitive 画 stacked bar：

```text
forward components:
  norm / basis-or-activation / mixing / other

backward components:
  activation grad / param grad / input grad / optimizer

memory components:
  saved activations / basis tensor / workspace / parameters / optimizer state
```

目标：直接看哪个部分导致 fail。

### 6.2 Task-efficiency Pareto

横轴：

$$
T_{step}/T_{MLP}
$$

纵轴：

$$
Acc-Acc_{MLP}
$$

点大小：

$$
M_{backward}/M_{MLP}.
$$

颜色表示 primitive family：DWM / RationalKAT / CP / LUT / Dense reference。

### 6.3 Memory-time frontier

横轴：

$$
M_{backward}/M_{MLP}
$$

纵轴：

$$
T_{backward}/T_{MLP}.
$$

标出 final target 区域：

$$
M_{backward}/M_{MLP}\leq0.8,
$$

$$
T_{backward}/T_{MLP}\leq1.4.
$$

### 6.4 Recipe heatmaps

对 DWM / RationalKAT / LUT 分别画：

```text
x-axis: lr / residual scale / K / groups / grid
 y-axis: recipe variant
 cell: accuracy gap, ECE, step ratio, memory ratio
```

### 6.5 Classwise failure heatmap

画每个 dataset 的 classwise accuracy difference：

$$
Acc_{class}^{candidate}-Acc_{class}^{MLP}.
$$

如果 KMNIST 某些字符类系统性失败，说明是 representation / grouping recipe 问题，而不是单纯 kernel 问题。

### 6.6 Geometry-efficiency plot

横轴：

$$
M_{backward}/M_{MLP}
$$

纵轴：

$$
\phi_{residual}\text{ reduction}
$$

点大小：accuracy gap。

### 6.7 Cold vs warm timing plot

每个 compiled primitive 画：

```text
cold_forward
warm_forward
compile_time
cold_warm_ratio
```

避免 v5.6/v5.7 中 compile overhead 和 steady-state 混淆。

---

## 7. 决策规则

### 7.1 如果 DWM-v2 过 task 但效率不过

结论：

```text
DWM-v2 architecture is useful but kernel still wrong.
```

行动：继续 kernel / fused implementation，不进入 functional update。

### 7.2 如果 DWM-v2 过效率但 task 不过

结论：

```text
DWM-v2 compute path is viable but expressivity insufficient.
```

行动：加 grouped nonlinear interaction / gated residual / teacher distillation。

### 7.3 如果 RationalKAT 过效率但 task 不过

结论：

```text
RationalKAT is the likely final efficient primitive, but recipe is underfit.
```

行动：继续 identity init / group / denominator / residual-scale repair。

### 7.4 如果 LUT-KAN 过效率和 task

结论：

```text
LUT-KAN becomes primary final primitive candidate.
```

行动：立刻做 custom backward + functional smoothing compatibility。

### 7.5 如果没有 primitive 过 P1 exploratory

结论：

```text
current primitive family cannot meet terminal efficiency target.
```

行动：不要跑 accuracy / functional optimizer，直接回 kernel primitive design。

### 7.6 如果 P5 没有 survivor

结论：

```text
no current strict PureKAN primitive satisfies task-efficiency joint requirement.
```

行动：停止 v5.8，写 failure diagnosis，不进入 P6-P10。

---

## 8. v5.8 最终预期产物

本轮必须生成：

```text
p0_core_manifest.csv
p1_efficiency_decomposition.csv
p1_component_timing.csv
p1_memory_decomposition.csv
p2_dwm_v2_recipe.csv
p3_rationalkat_v2_recipe.csv
p4_lutkan_feasibility.csv
p5_joint_task_efficiency.csv
p6_custom_backward_audit.csv
p7_lightsmooth_compatibility.csv
p8_functional_training_smoke.csv
p9_confirm5.csv
p10_confirm10.csv
failure_table.csv
failure_by_primitive.csv
failure_by_dataset.csv
failure_by_gate.csv
route_decision.json
aggregate_decision.json
figures/
```

并且 `route_decision.json` 必须明确给出以下之一：

```text
A_DWM_v2_survivor
B_RationalKAT_survivor
C_LUTKAN_survivor
D_CP_reference_only
E_no_efficient_primitive
```

---

## 9. 最终总结

v5.7 不是没有进展，而是把问题进一步钉死在 primitive 层面：

$$
\boxed{\text{functional optimizer 不能拯救一个不满足效率和任务门槛的 primitive。}}
$$

v5.8 的目标是找到一个真正可进入终极系统的 PureKAN primitive。Dense AB-RBF 继续作为 teacher / reference；DWM-v2、RationalKAT-AB-v2、LUT-KAN 是主要候选；CP 是保表达力的参考路线。

本轮只有一个问题：

$$
\boxed{\text{是否存在一个 strict PureKAN primitive，能接近 MLP 的时间/显存，并保持足够 accuracy？}}
$$

如果答案是否定的，就不应该继续 functional optimizer；如果答案是肯定的，才进入下一阶段：

$$
\boxed{\text{在该 primitive 上加入 functional geometry maintenance，并冲击最终目标。}}
$$
