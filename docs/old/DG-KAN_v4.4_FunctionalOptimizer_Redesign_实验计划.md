# DG-KAN v4.4 Functional Optimizer Redesign 实验计划

## 0. 当前判断：不能再把 functional update 当作 Sobolev preconditioner 调参

这份计划的目标不是继续微调 `Sobolev alpha / beta`、`trust_radius`、`branch_final_scale`、`diagwarmup` 或固定 shallow/deep 分工。v4.3 已经给出足够强的证据：当前 PureKAN functional optimization 的失败不是单一实现 bug，也不是缺少 normalization，而是 optimizer 定义本身不完整。

当前 PureKAN 的严格设定是：

```text
input_kan.coeff
blocks.{i}.kan.coeff
output_kan.coeff
```

这三类 KAN coefficient 是唯一可学习参数；没有 learnable LayerNorm，没有 Linear stem/head，没有 ordinary non-KAN 参数。PureKAN-AdamW 和所有 functional candidates 更新的是同一批 coefficient，只是更新规则不同。因此，PureKAN-AdamW 能训练而 functional update 不能训练，说明问题集中在 optimizer dynamics，而不是模型完全没表达力。

v4.3 的核心结果可以概括为：

```text
FPA:
  比 Sobolev-only / CFT 更接近 AdamW 方向，尤其 FPA-edge-prox 的 Adam alignment 较高。
  但短程训练严重 underfit，说明“靠近 AdamW 方向”本身不够。

EK-FNG:
  edge-feature / task curvature 是有意义的方向。
  但仍然无法缩小 KMNIST 与 PureKAN-AdamW 的差距。

CFT:
  output-only 和 coordinated target fitting 仍然不稳定或 underfit。

BFT / trust gate:
  能避免 catastrophic divergence，但会把 update 变弱。

结论:
  当前 functional update 仍然是 local proposal 或 local preconditioner，
  不是一个能完成深度表示学习的 optimizer。
```

PureKAN functional optimizer 需要重新定义为：

$$
\text{functional optimizer}
=
\text{task-learning dynamics}
+
\text{function-space coordinate/control}
+
\text{long-horizon representation support}.
$$

之前的更新基本是：

$$
\Delta a=-\eta S^{-1}g,
$$

或者：

$$
\Delta a_l=-\eta_l(G_l+\lambda_lS_l+\rho I)^{-1}g_l.
$$

这类公式的问题是，$S_l$ 主要是函数平滑先验，$G_l$ 目前也只是局部或近似 task metric。它们没有完整捕捉 AdamW 在 PureKAN 里表现出的三种能力：

```text
1. 长期时间尺度：momentum / RMS / per-parameter scale adaptation；
2. 表示学习动力学：input -> hidden -> logits 的多层 co-adaptation；
3. 容忍“暂时不平滑但任务有效”的函数形状。
```

所以 v4.4 的目标不是“让 Sobolev 更好”，而是回答一个更根本的问题：

$$
\boxed{
\text{PureKAN 需要怎样的 functional-coordinate optimizer，才能超过 PureKAN-AdamW？}
}
$$

---

## 1. v4.4 的三个核心假设

### 1.1 假设 A：AdamW 的成功来自时间尺度，而不是 raw parameter coordinate 本身

PureKAN-AdamW 能训练，并不一定说明 raw coefficient coordinate 是最好的函数空间坐标。它可能只是说明 AdamW 的长期动力学足够强：

```text
momentum
RMS normalization
per-parameter adaptive learning rate
noise averaging
implicit warmup / scale control
```

因此，下一步应该测试 **functional-coordinate Adam**，而不是继续用 $S^{-1}g$ 直接替代 AdamW。

核心思路是把 KAN coefficient 先变换到一个函数空间白化坐标 $u$，然后在 $u$ 上用 AdamW：

$$
S+\rho I = LL^\top,
$$

$$
a = L^{-\top}u.
$$

在这个参数化下，$u$ 的 Euclidean step 对应 $a$ 的 Sobolev geometry。于是我们不再写：

$$
\Delta a=-\eta S^{-1}g,
$$

而是让 optimizer 学：

$$
u_{t+1}=\operatorname{AdamW}(u_t, \nabla_uL),
$$

再映射回：

$$
a_{t+1}=L^{-\top}u_{t+1}.
$$

如果这条路成功，说明之前失败不是因为 functional geometry 没价值，而是因为我们把 functional geometry 作为一次性 preconditioner，而不是作为 optimizer coordinate。

---

### 1.2 假设 B：functional update 应该模仿 AdamW 的有用函数位移，而不是模仿 AdamW 的参数位移

FPA 在 v4.3 中尝试保持 Adam-like direction，但短程训练仍然不够。一个可能原因是它对齐的是参数 update，而不是 layer function update。

对 KAN layer：

$$
Y_l = \Phi_l A_l^\top.
$$

AdamW 的一次更新产生一个真实的 layer output displacement：

$$
\Delta Y_l^{Adam}
=Y_l(A_l+\Delta A_l^{Adam})-Y_l(A_l).
$$

真正值得模仿的不是 $\Delta A_l^{Adam}$，而是 $\Delta Y_l^{Adam}$，因为深度网络的表示学习发生在 activation/function space。v4.4 要测试一种 **shadow-Adam functional distillation**：每一步用 shadow AdamW 产生一个 teacher functional displacement，但最终写回的更新由 functional least squares 产生。

即解：

$$
\Delta A_l
=
\arg\min_{\Delta A}
\left[
\|\Phi_l\Delta A^\top-\Delta Y_l^{Adam}\|_F^2
+
\lambda_l\|\Delta A\|_{S_l}^2
+
\rho\|\Delta A\|_F^2
\right].
$$

这个更新的含义是：

```text
AdamW 负责告诉我们“任务上有用的函数位移是什么”；
functional solver 负责用平滑、可控的 KAN coefficient 实现这个位移。
```

这和之前的 FTF 不同。FTF 用的是手写 target $Y_l^\star$，容易造成 cross-layer drift；这里用 AdamW 的真实局部函数位移做 teacher，至少从 PureKAN 已知可训练的轨迹中学习。

---

### 1.3 假设 C：PureKAN 需要 global output-space functional step，而不是 layer-local proposal

FTF one-step 很强但训练崩，BFT 稳定但弱，说明 layer-local target fitting 与全网络 loss 不一致。下一步要测试更接近全网络 trust-region 的 **Global Functional Kernel Step**。

把当前 batch 上的 logits 线性化：

$$
z(\theta+\Delta\theta)
\approx
z(\theta)+J\Delta\theta.
$$

我们不在参数空间直接解，而是在 output function space 中解低秩 trust-region 问题：

$$
\Delta\theta
=
-M^{-1}J^\top
\left(JM^{-1}J^\top+\mu H_z^{-1}\right)^{-1}r.
$$

其中：

```text
M = function-space metric，包含 Sobolev / basis covariance / damping；
J = logits 对 KAN coefficients 的 Jacobian；
H_z = cross-entropy 的 logit Hessian 或 Fisher；
r = logits residual / natural-gradient residual。
```

这条路线成本更高，但它直接回答：之前失败是否来自缺少全网络 downstream sensitivity。

---

## 2. v4.4 的方法候选

### 2.1 FC-Adam-v2：Functional-Coordinate Adam

FC-Adam-v2 是本轮最基础的 candidate。它不再把 Sobolev inverse 当更新，而是把 Sobolev / basis metric 变成坐标变换。

候选包括：

```text
FCAdam-L2:
  使用 RBF L2 Gram 白化。

FCAdam-H1:
  使用 L2 + alpha H1 Sobolev Gram 白化。

FCAdam-H1-low:
  使用较弱 Sobolev，避免过度平滑。

FCAdam-dataGram:
  使用 batch/EMA basis covariance 白化。

FCAdam-dataSob:
  使用 data basis covariance + light Sobolev。
```

关键是：AdamW 的 momentum/RMS 状态在 $u$ 坐标中维护，而不是在 raw $a$ 坐标中维护。

实验要明确记录：

```text
u_grad_norm
u_update_norm
a_update_norm
functional_update_norm
cos(raw AdamW update, FCAdam update)
cos(function displacement AdamW, function displacement FCAdam)
Adam state norm in u-coordinate
basis metric condition
coordinate reconstruction error
```

成功信号不是立刻几何最优，而是：

```text
short-horizon accuracy 接近或超过 PureKAN-AdamW；
val-loss AUC 不再大幅负；
feature rank 不塌；
phi/J 不像 AdamW 那样极端，但也不牺牲 task learning。
```

---

### 2.2 SFD：Shadow-Adam Functional Distillation

SFD 每一步维护一个 shadow AdamW 状态，但不直接提交 AdamW 参数。它只用 shadow AdamW 生成 teacher displacement。

对每个 layer，记录：

$$
\Delta Y_l^{Adam}=Y_l(A_l+\Delta A_l^{Adam})-Y_l(A_l).
$$

然后求：

$$
\Delta A_l^{SFD}
=
\arg\min_{\Delta A}
\left[
\|\Phi_l\Delta A^\top-\operatorname{stopgrad}(\Delta Y_l^{Adam})\|_F^2
+
\lambda_l\|\Delta A\|_{S_l}^2
+
\rho\|\Delta A\|_F^2
\right].
$$

SFD 有三种强度：

```text
SFD-direct:
  拟合 AdamW layer output displacement。

SFD-prox:
  拟合 AdamW displacement，但限制每步 functional norm。

SFD-residual:
  只拟合 AdamW update 中与当前 functional optimizer 不同的 residual。
```

SFD 不应该被解释成最终纯理论 optimizer，而是一个诊断和过渡工具。它回答：

$$
\boxed{
\text{如果 functional update 模仿 AdamW 的函数位移，是否能训练 PureKAN？}
}
$$

如果 SFD 成功，说明我们之前的 analytic metrics 缺的是 AdamW 的 task-learning dynamics。如果 SFD 失败，说明问题可能更深，涉及 architecture 或 basis parameterization。

---

### 2.3 GFK：Global Functional Kernel Step

GFK 是最接近全网络 functional natural gradient 的 candidate。它使用 batch 上的 logits Jacobian，避免每层独立 target fitting 的 drift。

简化目标为：

$$
\min_{\Delta\theta}
\left[
\frac{1}{2}\|J\Delta\theta-r\|_{H_z}^2
+
\frac{\lambda}{2}\|\Delta\theta\|_M^2
\right].
$$

解的 output-space 形式为：

$$
\Delta\theta
=
M^{-1}J^\top
\left(JM^{-1}J^\top+\lambda H_z^{-1}\right)^{-1}r.
$$

候选：

```text
GFK-output-only:
  只更新 output_kan，作为 sanity。

GFK-block-output:
  更新 blocks + output，不更新 input。

GFK-all-lowrank:
  input/block/output 全部参与，但用 low-rank batch solve。

GFK-hybrid-step:
  用 GFK 每 N 步替代一次普通 functional step。
```

GFK 的目标不是马上成为最终高效 optimizer，而是判断：

```text
是否只有 global downstream sensitivity 才能让 PureKAN functional update 学起来。
```

如果 GFK 能在短程训练中接近 AdamW，那么后续再考虑近似化、缓存、低秩化。

---

### 2.4 AB-KAN：Affine-Basis PureKAN

另一个需要正视的问题是当前 RBF basis 可能对 functional optimizer 不友好。AdamW 可以在 raw coefficient space 中慢慢学出近似线性/仿射结构，但 Sobolev/TFU/FNG 可能会压制这种早期表示。

因此 v4.4 加入一个 architecture/coordinate 诊断：在每条 edge 的 basis 中加入 explicit constant / linear basis：

$$
\phi(x)=a_0+a_1x+\sum_m c_mB_m(x).
$$

这仍然是 PureKAN，因为 $a_0,a_1,c_m$ 都是 KAN edge coefficients，不引入 Linear layer 或 non-KAN 参数。

候选：

```text
PureKAN-RBF:
  当前版本。

PureKAN-AB-RBF:
  RBF + constant + linear basis。

PureKAN-AB-RBF-orth:
  对 basis 做 Gram-Schmidt / metric whitening。
```

如果 AB-KAN 显著改善 functional optimizer，却不显著改变 AdamW，则说明 functional update 的缺陷部分来自 basis coordinate conditioning。

---

## 3. 实验总原则

本轮不允许直接跑 10 seeds。必须按如下逻辑推进：

```text
先做 one-step / multi-step direction audit；
再做 3-seed short-horizon；
只有明显接近 PureKAN-AdamW 的候选，才能进入 full-budget；
只有 full-budget 3-seed 接近或超过 AdamW，才进入 5-seed / 10-seed。
```

v4.4 的判断标准不是“比旧 Sobolev 好”，而是必须对比：

```text
PureKAN-AdamW
Hybrid-DGKAN-UFULL-f085
MLP-AdamW
D0-allFullSobolev
D6-allTaskAware
F4-FNG-leftFullRight
FPA / EK-FNG best from v4.3
```

最终目标是：

$$
\boxed{
\operatorname{Acc}_{PureKANFunctional}
>
\max(
\operatorname{Acc}_{PureKANAdamW},
\operatorname{Acc}_{HybridDGKAN},
\operatorname{Acc}_{MLPAdamW}
)
}
$$

同时要求：

```text
val-loss AUC 不劣于 AdamW；
ECE 不劣于 AdamW；
phi_prime_p95 / Jacobian 明显优于 AdamW；
没有 non-KAN trainable params；
所有 KAN coefficient update coverage = 1.0。
```

---

## 4. P0：实现不变量与旧失败复现

### 4.1 目的

P0 不是训练性能测试，而是保证 v4.4 的新实现没有污染 PureKAN 的严格设定，并且旧失败可以复现。

### 4.2 必跑方法

```text
PureKAN-AdamW
D0-allFullSobolev
D6-allTaskAware
F4-FNG-leftFullRight
FCAdam-L2-smoke
FCAdam-H1-smoke
SFD-direct-smoke
GFK-output-only-smoke
GFK-all-lowrank-smoke
PureKAN-AB-RBF-AdamW-smoke
PureKAN-AB-RBF-FCAdam-smoke
```

### 4.3 必须记录

```text
learnable_nonKAN_params
raw_nonKAN_params
functional_coverage
input_coeff_seen
block_coeff_seen
output_coeff_seen
basis_type
basis_count
has_affine_basis
alpha_trainable
norm_mode
optimizer_state_type
adam_state_in_function_coordinate
shadow_adam_state_exists
metric_condition_input
metric_condition_block_mean
metric_condition_output
coordinate_reconstruction_error
rollback_error_for_shadow_step
```

### 4.4 通过条件

所有 strict PureKAN functional rows 必须满足：

```text
learnable_nonKAN_params = 0
functional_coverage = 1.0
input/block/output coeff seen = 1.0
alpha_trainable = 0
coordinate_reconstruction_error < 1e-6
rollback_error < 1e-6
metric conditions finite
```

如果 P0 不通过，后续不跑。

---

## 5. P1：AdamW functional trajectory audit

### 5.1 目的

这一步是 v4.4 最重要的诊断。过去我们只比较 update direction，而没有真正看 AdamW 在 function space 中做了什么。P1 要回答：

$$
\boxed{
\text{AdamW 到底怎样改变 input/block/output KAN functions？}
}
$$

### 5.2 方法

对 PureKAN-AdamW 跑若干短轨迹，每个 step 记录 AdamW 实际更新前后的 functional displacement。

对每层 $l$ 记录：

$$
\Delta Y_l^{Adam}
=Y_l(\theta_{t+1}^{Adam})-Y_l(\theta_t),
$$

$$
\Delta z^{Adam}
=z(\theta_{t+1}^{Adam})-z(\theta_t).
$$

然后用当前 functional candidates 去拟合或投影这些 displacement。

### 5.3 记录指标

```text
adam_delta_coeff_norm_by_role
adam_delta_function_norm_by_role
adam_delta_function_over_coeff_norm
adam_delta_sobolev_norm_by_role
adam_delta_margin_change
adam_delta_feature_rank_change
adam_delta_class_separation_change
adam_delta_basis_occupancy_change
adam_delta_phi_prime_change
adam_delta_jacobian_change
```

对每个 functional proposal 记录：

```text
cos_coeff_with_adam
cos_function_with_adam
function_displacement_r2_vs_adam
projection_error_vs_adam
projection_error_by_role
margin_change_error_vs_adam
rank_change_error_vs_adam
actual_train_descent
actual_holdout_descent
```

### 5.4 关键图

```text
AdamW functional displacement norm by role over training
cos_function_with_adam heatmap
function R2 vs train descent scatter
margin change AdamW vs candidate scatter
feature rank change AdamW vs candidate scatter
role-wise displacement stacked area plot
```

### 5.5 判定

如果某个 candidate 的 function-space alignment 高，但短程训练差，说明它缺长期状态或 step-scale。

如果 function-space alignment 低，说明 proposal 方向本身仍错。

---

## 6. P2：one-step 与 5-step horizon audit

### 6.1 目的

v4.3 说明 one-step 方向不够。P2 要同时看 1-step 和 5-step。一个 optimizer 如果只能单步下降但 5 步后 representation rank / margin 坏掉，不能进入训练。

### 6.2 方法组

```text
PureKAN-AdamW-one-step
D0-allFullSobolev
D6-allTaskAware
F4-FNG-leftFullRight
FPA-edge-prox
EKFNG-leftRightFull-lowSob
FCAdam-L2
FCAdam-H1-low
FCAdam-dataSob
SFD-direct
SFD-prox
SFD-residual
GFK-output-only
GFK-block-output
GFK-all-lowrank
AB-RBF-FCAdam-H1-low
```

### 6.3 指标

每个方法在同一初始模型和同一 batch 序列上跑 1 step 与 5 steps，记录：

```text
train_loss_descent_1step
holdout_loss_descent_1step
train_loss_descent_5step
holdout_loss_descent_5step
bad_step_rate
cumulative_margin_change
cumulative_rank_change
cumulative_logit_drift
cumulative_activation_drift
cos_to_adam_per_step
function_r2_to_adam_per_step
phi_prime_p95_change
jacobian_change
basis_occupancy_entropy_change
```

### 6.4 通过条件

一个方法进入 P3 必须满足：

```text
bad_step_rate = 0
holdout_loss_descent_5step >= 0
function_r2_to_adam_mean > 0.5 或 train_loss_descent_5step 接近 AdamW
feature_rank_change 不低于 AdamW 的 70%
margin_change 不低于 AdamW 的 70%
phi_prime_p95 不超过 AdamW 的 90%
```

---

## 7. P3：3-seed short-horizon 训练

### 7.1 目的

只让 P2 通过的候选进入 P3。P3 不是最终性能确认，而是看候选是否能形成有效表示。

### 7.2 设置

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0, 1, 2

budget:
  short-horizon, 例如 full budget 的 30%-40%

baselines:
  PureKAN-AdamW
  MLP-AdamW
  Hybrid-DGKAN-UFULL-f085
  D0-allFullSobolev
  D6-allTaskAware
  F4-FNG-leftFullRight
```

### 7.3 记录指标

```text
test_acc
val_loss
val_loss_auc
val_acc_auc
train_loss_curve
holdout_loss_curve
ECE
NLL
classwise_accuracy
margin_mean
margin_p10
feature_effective_rank_input
feature_effective_rank_block_mean
feature_effective_rank_output
class_centroid_separation
basis_occupancy_entropy
rbf_dead_basis_fraction
out_of_grid_fraction
phi_prime_p95
phi_prime_max
jacobian_condition
curvature_energy
step_time_ms
memory_peak_mb
optimizer_state_memory_mb
```

### 7.4 通过条件

候选进入 P4 必须满足：

```text
MNIST gap vs PureKAN-AdamW < 2%
Fashion gap vs PureKAN-AdamW < 2%
KMNIST gap vs PureKAN-AdamW < 3%
val-loss AUC 不比 D6 差
margin_p10 不低于 PureKAN-AdamW 的 80%
feature rank 不低于 PureKAN-AdamW 的 80%
ECE 不劣于 PureKAN-AdamW
```

如果没有候选满足，停止，不跑 5-seed。

---

## 8. P4：FC-Adam-v2 深入验证

### 8.1 目的

如果 FC-Adam 在 P3 有信号，P4 专门验证它是不是解决了“functional coordinate + Adam time-scale”的问题。

### 8.2 变体

```text
FCAdam-L2
FCAdam-H1-low
FCAdam-H1-mid
FCAdam-dataGram
FCAdam-dataSob
FCAdam-dataSob-cosineDecay
FCAdam-AB-RBF-H1-low
```

### 8.3 额外指标

```text
u_coordinate_grad_norm
u_coordinate_update_norm
u_adam_m_norm
u_adam_v_norm
u_step_rms
coordinate_condition
basis_whitening_error
function_norm_per_u_norm
sobolev_energy_curve
AdamW_coordinate_alignment
```

### 8.4 判断

如果 FCAdam 接近 AdamW accuracy 但几何更好，这是非常强的路线。它说明 functional update 不应该是一次性 preconditioner，而应该是坐标重参数化后的 Adam-like optimizer。

---

## 9. P5：SFD 深入验证

### 9.1 目的

如果 SFD 在 P3 有信号，P5 用来判断 functional update 是否能通过模仿 AdamW function trajectory 来训练。

### 9.2 变体

```text
SFD-direct-lambda0
SFD-direct-lightSob
SFD-prox-lightSob
SFD-prox-midSob
SFD-residual-lightSob
SFD-residual-with-rank-preserve
SFD-residual-with-margin-preserve
```

### 9.3 额外指标

```text
teacher_function_displacement_norm
student_function_displacement_norm
teacher_student_function_r2
teacher_student_margin_r2
teacher_student_rank_r2
SFD_fit_residual
SFD_sobolev_penalty
SFD_step_shrink_rate
shadow_adam_state_norm
```

### 9.4 判断

如果 SFD 成功，但 FCAdam 不成功，说明 AdamW 的 function trajectory 是关键，而坐标变换不够。

如果 SFD 不成功，但 FCAdam 成功，说明长期 Adam dynamics 比单步 teacher displacement 更关键。

如果二者都失败，说明可能需要 global kernel / architecture change。

---

## 10. P6：GFK 深入验证

### 10.1 目的

如果 GFK 有信号，P6 用来验证 global downstream sensitivity 是否是缺失主因。

### 10.2 变体

```text
GFK-output-only
GFK-block-output
GFK-all-lowrank-rank16
GFK-all-lowrank-rank32
GFK-all-every4steps
GFK-all-plus-FCAdam-between
```

### 10.3 额外指标

```text
Jv_norm
JT_v_norm
kernel_condition
kernel_eig_min
kernel_eig_max
cg_residual
cg_iterations
predicted_logit_improvement
actual_logit_improvement
predicted_loss_descent
actual_loss_descent
output_trust_radius_used
```

### 10.4 判断

如果 GFK 明显优于 layer-local candidates，说明下一步应该用 global functional natural gradient 方向，不再继续 layerwise preconditioner。

---

## 11. P7：full-budget 3-seed candidate selection

P7 只跑 P4/P5/P6 中最好的 2-3 个候选。不能超过 3 个，否则又会变成搜索。

候选必须包括：

```text
PureKAN-AdamW
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
best_FCAdam_or_SFD_or_GFK
second_best_if_needed
```

记录所有 P3 指标，并额外记录：

```text
time_to_target_loss
step_time
memory
optimizer_state_memory
wallclock_to_AdamW_acc
```

进入 P8 的条件：

```text
至少一个候选在 3 个数据集上平均 acc >= PureKAN-AdamW；
且没有任何数据集 gap > 1.5%；
且 ECE / phi / Jacobian 至少两个指标优于 AdamW。
```

---

## 12. P8：5-seed confirm

P8 使用 seeds 0..4，确认 P7 的最佳候选。

### 12.1 必须记录

```text
paired_acc_delta_vs_PureKAN_AdamW
paired_auc_delta_vs_PureKAN_AdamW
paired_ece_delta_vs_PureKAN_AdamW
paired_margin_delta_vs_PureKAN_AdamW
paired_rank_delta_vs_PureKAN_AdamW
paired_phi_delta_vs_PureKAN_AdamW
paired_jacobian_delta_vs_PureKAN_AdamW
```

### 12.2 通过条件

```text
mean acc > PureKAN-AdamW on at least 2/3 datasets
no dataset acc gap worse than -0.5%
AUC >= AdamW on at least 2/3 datasets
ECE improved on at least 2/3 datasets
phi/J improved on all datasets
```

如果通过，再进入 10-seed。

---

## 13. P9：10-seed final confirm

P9 是最终确认，不再调参。

```text
seeds = 0..9
methods:
  PureKAN-AdamW
  MLP-AdamW
  Hybrid-DGKAN-UFULL-f085
  final_v4.4_candidate
```

必须输出：

```text
final_scorecard.csv
paired_delta_vs_purekan_adamw.csv
paired_delta_vs_mlp.csv
paired_delta_vs_hybrid.csv
wallclock_summary.csv
geometry_summary.csv
representation_summary.csv
failure_table.csv
```

最终 claim 只有在以下条件满足时才成立：

$$
\boxed{
\text{PureKAN functional optimizer is solved.}
}
$$

条件：

```text
1. 平均 accuracy 超过 PureKAN-AdamW；
2. 平均 accuracy 超过 Hybrid-DGKAN-UFULL-f085 或至少不低于 0.5%；
3. 平均 accuracy 超过 MLP-AdamW 或在 harder dataset 上超过；
4. AUC / ECE / geometry 至少两个维度稳定优于 AdamW；
5. nonKAN params = 0；
6. functional coverage = 1.0；
7. 没有 catastrophic seed。
```

---

## 14. 必须画的图

### 14.1 optimizer trajectory 图

```text
train loss curve
val loss curve
test acc curve
val acc curve
AUC bar
```

### 14.2 function-space alignment 图

```text
cos_function_with_adam over steps
function_displacement_R2 over steps
projection_error_vs_accuracy_gap scatter
role-wise function displacement stacked area
```

### 14.3 representation 图

```text
feature effective rank by layer over epochs
class centroid separation over epochs
margin mean and margin p10 curves
classwise accuracy heatmap
```

### 14.4 geometry 图

```text
phi_prime_p95 curve
Jacobian condition curve
Sobolev energy curve
basis occupancy entropy curve
basis dead fraction curve
```

### 14.5 optimizer dynamics 图

```text
Adam/RMS state norm in functional coordinate
u_update_norm vs a_update_norm
accepted eta / step scale curve
metric condition spectrum heatmap
CG residual curve for GFK
```

### 14.6 Pareto 图

```text
accuracy vs phi_prime_p95
accuracy vs ECE
accuracy vs wallclock
accuracy vs function_displacement_R2
AUC vs step_time
```

---

## 15. 失败解释表

每个失败候选必须自动归类：

```text
F1: one-step direction wrong
F2: one-step right but 5-step bad
F3: function-space Adam alignment low
F4: function-space Adam alignment high but accuracy low
F5: feature rank collapse
F6: margin not improving
F7: geometry over-regularization
F8: update too damped
F9: output-only learning / hidden underfit
F10: global kernel condition bad
F11: wall-clock infeasible
```

不要只写 “fail”。每个 fail 都必须对应至少一个具体指标。

---

## 16. 当前不再推进的方向

以下方向在 v4.4 暂停：

```text
继续调 Sobolev alpha / beta
继续调 branch_final_scale
继续调 diagwarmup / smooth transition
继续固定 shallow/deep 手写分工
继续 CFT-output-only
继续 BFT trust gate 微调
继续只看 one-step descent
```

这些方向已经不能解释 PureKAN-AdamW 与 functional candidates 的主要差距。

---

## 17. 最终预期结论模板

### 情况 A：FCAdam 成功

结论：

```text
PureKAN 需要 Adam-like temporal dynamics；
function-space geometry 应作为 coordinate system，而不是 direct inverse preconditioner。
```

### 情况 B：SFD 成功

结论：

```text
PureKAN functional optimizer 缺的是 AdamW 的 function trajectory；
可通过 functional distillation 把 AdamW trajectory 转化为 smooth KAN coefficient update。
```

### 情况 C：GFK 成功

结论：

```text
layer-local functional update 不够；
PureKAN 需要 global output-space functional natural gradient。
```

### 情况 D：AB-KAN 成功

结论：

```text
当前 RBF-only basis coordinate 对 functional optimizer 不友好；
添加 affine basis / orthogonal basis 后 functional update 才能释放 PureKAN 表达力。
```

### 情况 E：全部失败

结论：

```text
PureKAN functional optimization 暂未解决；
当前证据支持 Hybrid-DGKAN branch functional update，
但不支持 full PureKAN functional optimizer。
```

这时必须停止继续 optimizer 小修，转向模型结构或理论重新建模。

---

## 18. 一句话总结

v4.4 的核心不再是寻找更好的 $S^{-1}g$，而是验证下面三种更根本的可能：

$$
\boxed{
\text{functional coordinate + Adam dynamics}
}
$$

$$
\boxed{
\text{AdamW function trajectory distillation}
}
$$

$$
\boxed{
\text{global output-space functional natural gradient}
}
$$

如果这三条都无法让 PureKAN functional optimizer 接近 PureKAN-AdamW，那么就应承认：当前 PureKAN full functional optimization 还没有被解决，而此前的成功必须严格限定为 Hybrid-DGKAN residual branch functional update。
