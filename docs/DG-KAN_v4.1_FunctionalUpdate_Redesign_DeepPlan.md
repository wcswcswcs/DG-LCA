# DG-KAN v4.1：Functional Update 重设计与 PureKAN 恢复实验计划

> 版本：v4.1  
> 目标：不再继续微调 Sobolev / TFU / FNG 的局部超参，而是重新设计 PureKAN 的 functional update。  
> 公式格式：Typora 友好，仅使用 `$...$` 和 `$$...$$`。  
> 当前判断：
>
> $$
> \boxed{
> \text{Sobolev-only U-FULL 不是 PureKAN optimizer。}
> }
> $$
>
> $$
> \boxed{
> \text{当前 TFU/FNG 仍是局部 preconditioner，不是完整深度网络 optimizer。}
> }
> $$
>
> 下一步主线应从：
>
> $$
> \Delta a=-\eta S^{-1}g
> $$
>
> 转向：
>
> $$
> \boxed{
> \text{Functional Target Fitting}
> +
> \text{Functional-coordinate Adam}
> +
> \text{network-function trust region}
> }
> $$

---

# 1. 当前实验结论：我们到底学到了什么

v3.9 的结果非常重要，因为它把很多“可能是实现问题”的解释排除了。PureKAN strict functional variants 已经满足：

```text
nonKAN params = 0
functional coverage = 1.000
input coeff / block coeff / output coeff 都被覆盖
FNG metric condition finite
run errors = 0
```

所以现在不能再把 PureKAN functional update 的失败主要归咎于：

```text
alpha 没 fixed
coeff 没被识别
还有 non-KAN 参数
normalization hook 没接上
FNG metric 没被调用
```

v3.9 还给出了三个关键结论。

第一，**normalization 必要，但不是主因**。`NoNorm-AdamW` 是硬失败边界，说明没有 normalization 不行；但 `AffineNorm / ScalarGain` 没有救回 PureKAN functional training，所以缺少 learnable LN 不是核心瓶颈。

第二，**FNG/KFAC-style metric 能修复一部分局部方向，但没有形成可训练 optimizer**。P1 里 FNG variants 修复了一些 bad-step case，但没有任何候选通过 strict direction gate；P3 relaxed diagnostic 也没有 formal survivor 进入后续确认。

第三，**PureKAN-AdamW 仍然可训练**，所以 PureKAN 架构本身不是没有表达力。当前失败主要来自 optimizer metric / dynamics，而不是 PureKAN 无法表达任务。

因此当前结论应该写成：

$$
\boxed{
\text{PureKAN functional optimization remains unsolved.}
}
$$

但它不应该写成：

$$
\boxed{
\text{PureKAN architecture failed.}
}
$$

更准确地说：

$$
\boxed{
\text{当前 functional update 没有提供 AdamW 所具备的深度表示学习动力学。}
}
$$

---

# 2. 为什么当前 functional update 难调：根本缺陷

## 2.1 Sobolev metric 是 smoothness prior，不是 task geometry

当前 U-FULL 的核心是：

$$
\Delta a=-\eta S^{-1}g,
$$

其中：

$$
S=
\int BB^\top
+
\alpha\int B'B'^\top
+
\beta\int B''B''^\top.
$$

这个 $S$ 控制的是一维 edge function 的平滑性：

```text
函数幅度
一阶导数
二阶曲率
局部 Jacobian
```

但分类任务真正需要的是：

```text
loss 下降
class margin 增大
representation 变得可分
logit boundary 变好
```

因此，$S^{-1}g$ 很容易成为“几何漂亮但任务不足”的方向。  
在 Hybrid-DGKAN 里，stem/head/LayerNorm/AdamW 已经承担了主要 representation learning，KAN branch 只需要做 residual correction，所以 Sobolev functional update 有用。  
但在 PureKAN 里，input KAN、block KAN、output KAN 都要靠 functional update 完成完整表示学习，这时 Sobolev-only 方向就不够了。

---

## 2.2 当前 TFU/FNG 仍然太局部

TFU/FNG 的形式大致是：

$$
\Delta a_l
=
-\eta_l
\left(
G_l+\lambda_lS_l+\rho I
\right)^{-1}g_l.
$$

问题是当前的 $G_l$ 仍然是局部 proxy，例如：

```text
data diagonal metric
basis occupancy metric
layer-local feature covariance
left/right KFAC-like approximation
```

这些 metric 没有充分捕捉：

```text
这个 layer 的 coefficient update 对最终 logits 的真实影响；
早期 layer update 对后续 layer input distribution 的影响；
多层 KAN composition 的 gauge / scale coupling；
representation 长期演化的稳定性。
```

因此，P1 里 one-step descent 可以被修复，但 P2/P3 训练仍追不上 AdamW。  
这说明：

$$
\boxed{
\text{local descent correctness 不等于 long-horizon representation learning correctness。}
}
$$

---

## 2.3 PureKAN 的 deep composition 需要全网络 function-space trust region

PureKAN 不是一个单层 KAN regression problem。它是：

$$
x
\rightarrow
h_0=\operatorname{KAN}_{in}(x)
\rightarrow
h_1
\rightarrow
\cdots
\rightarrow
z=\operatorname{KAN}_{out}(h_L).
$$

因此每层 update 都改变下一层的输入分布。固定地在每条 edge 的一维 coordinate 上做 Sobolev regularization，会忽略这种深层 composition 的耦合。

更合理的 functional objective 应该约束的是全网络输出或中间 activation 的变化：

$$
\|\delta f_\theta(x)\|^2,
\quad
\|\delta h_l(x)\|^2,
\quad
\|\delta \phi_l\|^2_{\mathcal H}.
$$

也就是说，functional update 不应该只回答：

```text
这个 edge function 是否平滑？
```

而要回答：

```text
这次 update 是否让整个网络函数朝任务目标移动？
```

---

## 2.4 AdamW 的优势不是几何，而是长期 optimizer dynamics

PureKAN-AdamW 能训练，说明 AdamW 虽然没有函数空间几何，却具备：

```text
per-parameter RMS scale adaptation
momentum / temporal averaging
容忍粗糙但有用的函数
长期 representation learning dynamics
```

我们之前直接尝试 Adam-style UO 会失控，是因为 raw Adam moment 和 Sobolev direction 冲突。  
但这并不意味着 functional update 不需要时间尺度。相反，它说明：

$$
\boxed{
\text{functional update 需要自己的 temporal dynamics，而不是 raw Adam moment。}
}
$$

正确做法应该是：

```text
先构造 functional/task descent direction
再在 functional metric 下做 normalized temporal smoothing
```

而不是：

```text
直接对 raw coefficient gradient 做 Adam
```

---

# 3. 新设计方向总览

v4.1 不再把 functional update 等同于 Sobolev inverse。新方向分为三条主线：

```text
A. FTF: Functional Target Fitting
B. FC-Adam: Functional-coordinate Adam
C. FTR / Global FGN: network-function trust-region update
```

这三条路线回答三个不同问题。

---

## 3.1 FTF：Functional Target Fitting

FTF 的核心思想是：  
不要把 coefficient update 看成 preconditioned gradient，而是把每层 KAN 当成一个函数拟合器，让它拟合一个由 downstream credit 给出的局部 target。

对某一层 KAN：

$$
Y_l = \Psi_l A_l^\top,
$$

其中 $\Psi_l$ 是 RBF basis design matrix，$A_l$ 是 coefficient matrix。

反向传播得到该层输出端 credit：

$$
\delta_l = \frac{\partial L}{\partial Y_l}.
$$

我们希望 layer output 沿负梯度方向移动：

$$
Y_l^\star = Y_l - \tau_l \delta_l.
$$

于是定义 target change：

$$
T_l = Y_l^\star - Y_l = -\tau_l\delta_l.
$$

然后直接求解 functional ridge regression：

$$
\Delta A_l
=
\arg\min_{\Delta A}
\left[
\|\Psi_l \Delta A_l^\top - T_l\|_F^2
+
\lambda_l\|\Delta A_l\|_{S_l}^2
+
\rho\|\Delta A_l\|_F^2
\right].
$$

这和 $S^{-1}g$ 的根本区别是：  
FTF 直接在 layer output function space 里拟合任务目标，而不是只对 coefficient gradient 做平滑。

如果 $P_l=\text{in_dim}\times K$ 很大，可以用 dual solve：

$$
\Delta A_l^\top
=
S_l^{-1}\Psi_l^\top
\left(
\Psi_l S_l^{-1}\Psi_l^\top+\lambda I
\right)^{-1}
T_l.
$$

这样 solve 的主矩阵是 batch-size 维度，而不是 coefficient 维度，适合 input KAN 这种高维层。

---

## 3.2 FC-Adam：Functional-coordinate Adam

PureKAN-AdamW 能训练，说明 AdamW 的长期动力学非常重要。  
但是直接在 raw coefficient 上用 AdamW，不是 function-space invariant。

FC-Adam 的思路是：  
把 coefficient 重参数化到 functional-whitened coordinate：

$$
a = L^{-T}u,
$$

其中：

$$
S+\rho I=LL^\top.
$$

这样：

$$
\|a\|_S^2
=
\|u\|_2^2.
$$

然后在 $u$ 上使用 AdamW：

$$
u_{t+1}=\operatorname{AdamW}(u_t,\nabla_u L).
$$

这样 AdamW 的 momentum / RMS dynamics 被保留下来，但坐标是 function-space whitened 的。  
这不是旧的 UO-PGAdam，因为我们不是先算 Sobolev direction 再塞进 Adam moment，而是直接把参数坐标换成 functional coordinate。

FC-Adam 要回答的问题是：

$$
\boxed{
\text{PureKAN 需要的是 functional preconditioner，还是 functional parameterization + Adam dynamics？}
}
$$

---

## 3.3 FTR / Global FGN：全网络函数 trust region

最完整的 functional update 应该约束全网络输出：

$$
f_\theta(x).
$$

线性化网络：

$$
f_{\theta+\Delta}(x)
\approx
f_\theta(x)+J_\theta(x)\Delta.
$$

定义局部目标：

$$
\min_\Delta
\left[
g^\top\Delta
+
\frac{1}{2\eta}
\|J_\theta\Delta\|_{H_z}^2
+
\frac{\lambda}{2}\|\Delta\|_S^2
+
\frac{\rho}{2}\|\Delta\|^2
\right].
$$

对应线性系统：

$$
\left(
J^\top H_zJ+\lambda S+\rho I
\right)\Delta
=
-g.
$$

这个路线最接近真正的 functional natural gradient。  
它开销更大，不应该先作为默认 optimizer，而应该作为 diagnostic：  
如果 FTR one-step 很好，而 layer-wise FTF/FC-Adam 不够，说明问题在 cross-layer coupling。

---

# 4. v4.1 实验总策略

v4.1 的基本原则是：

$$
\boxed{
\text{先证明 update 是任务有效的，再谈几何稳定。}
}
$$

所以不再使用“phi/J 漂亮”作为主 gate。主 gate 改成：

```text
actual train/val descent
local target fit quality
long-horizon AUC
final accuracy
representation separability
```

几何指标仍然记录，但作为 constraint，而不是主要优化目标。

---

# 5. 实验对象与对照

## 5.1 数据集

```text
MNIST
Fashion-MNIST
KMNIST
```

暂时不进入 CIFAR。  
原因是 PureKAN 在 MNIST-family 上还没有 functional optimizer 解；CIFAR 只会叠加 ConvStem / scaling / basis interface 问题。

---

## 5.2 模型

主模型：

```text
PureKAN
hidden_dim = 64
depth = 2 or 4
basis_count = 16
alpha = fixed1
norm = FixedNorm
no trainable non-KAN params
```

保留可选 capacity sanity：

```text
hidden_dim = 96
basis_count = 24
depth = 4
```

但不能先用大模型掩盖 optimizer 问题。  
主表以 h64/b16/depth4 为准。

---

## 5.3 必须对照

```text
PureKAN-AdamW
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
D0-allFullSobolev
D6-allTaskAware
F4-FNG-leftFullRight
```

新增候选：

```text
FTF-output-only
FTF-blocks-output
FTF-all-sequential
FTF-all-simultaneous
FC-Adam
FC-Adam+SobDecay
FTR-CG-small
```

---

# 6. P0：实现与数学一致性检查

## 6.1 目标

确认 FTF / FC-Adam / FTR 的实现是数学上正确的，不直接进入性能结论。

## 6.2 检查项

### PureKAN 参数约束

```text
learnable_nonKAN_params = 0
functional_param_coverage = 1.000
input_kan_covered = 1
block_kan_covered = 1
output_kan_covered = 1
alpha_trainable = 0
alpha_value = 1.0
```

### FTF cache 检查

每个 KAN layer 需要记录：

```text
layer/input_shape
layer/output_shape
basis_design_shape
credit_shape
target_delta_shape
role = input / block / output
```

### FTF solve 检查

```text
ridge_matrix_condition
dual_or_primal_solve
ridge_residual_norm
target_fit_mse_before
target_fit_mse_after
target_fit_R2
```

其中：

$$
R^2_{fit}
=
1-
\frac{
\|\Psi\Delta A^\top-T\|^2
}{
\|T\|^2+\epsilon
}.
$$

P0 通过条件：

```text
all finite
target_fit_R2 > 0.5 on smoke batch
actual update norm finite
no NaN / Inf
```

### FC-Adam 检查

```text
functional_coordinate_param_count = raw_coeff_param_count
S_cholesky_condition finite
u_to_a_reconstruction_error < 1e-5
grad_chain_rule_error < 1e-4
```

### FTR 检查

```text
CG residual decrease
JVP/VJP finite
predicted functional output change finite
```

---

# 7. P1：单层与单步方向审计

## 7.1 目标

回答：哪个方向是真正的 task descent？

## 7.2 方法

每个数据集 seed0，抽一个 batch，比较：

```text
AdamW-one-step
D0-allFullSobolev
D6-allTaskAware
F4-FNG-leftFullRight
FTF-output-only
FTF-blocks-output
FTF-all-simultaneous
FTF-all-sequential
FC-Adam-one-step
FTR-CG-small
```

## 7.3 记录指标

### 方向质量

```text
p1/train_descent
p1/val_descent
p1/bad_train_step
p1/bad_val_step
p1/g_dot_delta
p1/predicted_descent
p1/actual_descent
p1/predicted_actual_ratio
p1/raw_direction_cos
```

其中：

$$
\text{train descent}
=
L_{before}^{train}-L_{after}^{train}.
$$

$$
\text{predicted/actual ratio}
=
\frac{
-g^\top\Delta
}{
L(\theta)-L(\theta+\Delta)+\epsilon
}.
$$

### Layer-local FTF 指标

```text
ftf/input_fit_R2
ftf/block_fit_R2_mean
ftf/output_fit_R2
ftf/input_target_norm
ftf/block_target_norm
ftf/output_target_norm
ftf/input_residual_after
ftf/output_residual_after
```

### Activation trust 指标

```text
act/input_delta_h_norm_ratio
act/block_delta_h_norm_ratio_mean
act/output_delta_logit_norm_ratio
act/max_layer_delta_norm_ratio
```

定义：

$$
r_{\Delta h,l}
=
\frac{
\|\Delta h_l\|
}{
\|h_l\|+\epsilon
}.
$$

### Functional complexity

```text
func/sobolev_norm_delta
func/high_freq_energy_delta
func/phi_prime_p95_before
func/phi_prime_p95_after
func/jac_condition_before
func/jac_condition_after
```

## 7.4 P1 通过条件

一个候选要进入 P2，必须满足：

```text
train_descent > 0 on all 3 datasets
val_descent >= -0.01 on all 3 datasets
bad_train_step = 0
target_fit_R2_mean > 0.6 for FTF candidates
max_layer_delta_norm_ratio < 0.25
```

注意：不再要求 raw/preconditioned cosine > 0.5。  
理由是 natural-gradient / target-fitting direction 可以和 raw gradient 有低 cosine，只要 $g^\top\Delta<0$ 且 actual descent 好即可。

---

# 8. P2：短训练 micro-run

## 8.1 目标

确认 P1 的好方向是否能变成长程训练信号。

## 8.2 设置

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
epochs = short budget:
  MNIST = 8
  Fashion = 12
  KMNIST = 8
train/val/test = 6000/1000/1000
```

## 8.3 方法

```text
PureKAN-AdamW
D0-allFullSobolev
D6-allTaskAware
F4-FNG-leftFullRight
FTF-output-only
FTF-blocks-output
FTF-all-simultaneous
FTF-all-sequential
FC-Adam
FC-Adam+SobDecay
FTR-CG-small
```

## 8.4 记录指标

### 任务指标

```text
task/train_loss_curve
task/val_loss_curve
task/test_acc
task/test_loss
task/best_val_acc
task/best_epoch
task/acc_gap_vs_purekan_adamw
task/acc_gap_vs_mlp_adamw
task/acc_gap_vs_hybrid_ufull
```

### AUC 指标

```text
conv/val_loss_auc
conv/val_acc_auc
conv/AUC_improvement_vs_purekan_adamw
conv/AUC_improvement_vs_D0
conv/AUC_improvement_vs_D6
```

### FTF/FTR 过程指标

```text
ftf/fit_R2_curve_by_role
ftf/target_norm_curve_by_role
ftf/accepted_step_ratio
ftf/ridge_condition_curve
ftr/cg_iterations
ftr/cg_residual_final
ftr/predicted_actual_ratio_curve
```

### Representation 指标

```text
repr/effective_rank_input
repr/effective_rank_block_mean
repr/effective_rank_output_input
repr/class_separation_ratio
repr/within_class_variance
repr/between_class_distance
repr/logit_margin_mean
repr/logit_margin_p05
```

Class separation ratio：

$$
R_{sep}
=
\frac{
\operatorname{mean}_{c\neq c'}\|\mu_c-\mu_{c'}\|^2
}{
\operatorname{mean}_c \operatorname{tr}(\Sigma_c)+\epsilon
}.
$$

### 函数复杂度指标

```text
func/phi_prime_p95
func/phi_prime_max
func/curvature_energy
func/high_frequency_energy
func/basis_occupancy_entropy
func/dead_basis_fraction
func/out_of_grid_fraction
```

### Geometry 指标

```text
geom/max_jac_condition
geom/credit_amplification_p95
geom/jacobian_condition_per_layer
```

### Compute 指标

```text
compute/step_time_ms
compute/solve_time_ms
compute/cache_time_ms
compute/memory_peak_mb
compute/time_ratio_vs_adamw
compute/samples_per_sec
```

## 8.5 P2 通过条件

候选进入 P3，需要：

```text
MNIST acc gap vs PureKAN-AdamW < 2%
Fashion acc gap vs PureKAN-AdamW < 2%
KMNIST acc gap vs PureKAN-AdamW < 3%
val_loss_auc better than D6 on all datasets
no catastrophic ECE regression
phi_prime_p95 not more than 1.5x AdamW
```

如果没有候选通过 P2：

```text
停止大规模训练；
进入 P2-failure diagnosis。
```

---

# 9. P2-failure diagnosis：如果仍然失败，必须回答的问题

如果 P2 没有候选接近 AdamW，不要继续调参。必须定位失败类型。

## 9.1 FTF target fit 很好但训练差

判定：

```text
fit_R2 > 0.8
one-step descent > 0
short-run acc still bad
```

解释：

```text
layer-local target fitting 会造成 cross-layer drift。
```

下一步：

```text
改为 sequential output-to-input update
每层 update 后重算后续 activations
加入 activation trust region
```

## 9.2 FTF fit_R2 差

判定：

```text
fit_R2 < 0.5
ridge residual high
```

解释：

```text
basis/design matrix 不能拟合 downstream target。
```

下一步：

```text
增加 basis_count
改 centers / width
使用 adaptive centers
加入 residual linear skip
```

## 9.3 FC-Adam 接近 AdamW，FTF 不行

解释：

```text
主要缺的是 AdamW 的 temporal dynamics，而不是 functional target solve。
```

下一步：

```text
FC-Adam 进入主线；
FTF 做 regularized local correction。
```

## 9.4 FTR-CG 好但 FTF/FC-Adam 不好

解释：

```text
问题是 cross-layer coupling，layer-wise update 不够。
```

下一步：

```text
研究 low-rank global FTR / block-coordinate FTR。
```

---

# 10. P3：候选 refinement

P3 只对 P2 survivor 做小范围 refinement，不再大扫。

## 10.1 FTF 关键参数

```text
target_step_tau in {0.05, 0.10, 0.20}
sobolev_lambda in {0.0, 1e-4, 1e-3, 1e-2}
activation_trust_radius in {0.05, 0.10, 0.20}
sequential_mode in {simultaneous, output_to_input}
```

## 10.2 FC-Adam 关键参数

```text
functional_whitening = Sobolev / dataGram+Sobolev
adam_lr in {3e-4, 1e-3, 3e-3}
functional_weight_decay in {0, 1e-4, 1e-3}
gradient_clip in {off, global_norm_1.0}
```

## 10.3 FTR 关键参数

```text
cg_iters in {5, 10}
damping rho in {1e-2, 1e-1}
sobolev_lambda in {0, 1e-4, 1e-3}
trust_radius in {0.05, 0.10}
```

## 10.4 P3 记录指标

除 P2 指标外，还必须记录：

```text
hyperparam/tau
hyperparam/lambda
hyperparam/trust_radius
hyperparam/cg_iters
update/rejection_rate
update/backtracking_count
update/clipped_step_fraction
```

## 10.5 P3 进入 P4 条件

```text
MNIST gap vs AdamW < 1%
Fashion gap vs AdamW < 1%
KMNIST gap vs AdamW < 2%
AUC not worse than AdamW by more than 5%
geometry not catastrophically worse than AdamW
```

---

# 11. P4：5-seed confirm

## 11.1 方法

最多选 2 个候选进入 P4。

```text
PureKAN-AdamW
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
D6-allTaskAware
best candidate 1
best candidate 2
```

## 11.2 数据集与 seeds

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2,3,4
```

## 11.3 P4 成功标准

一个 candidate 成为 serious PureKAN functional optimizer，需要：

```text
test_acc >= PureKAN-AdamW - 0.5% on all datasets
test_acc >= MLP-AdamW - 0.5% on all datasets
test_acc >= Hybrid-DGKAN-UFULL - 0.5% on all datasets
val_loss_auc not worse than PureKAN-AdamW by more than 5%
ECE <= PureKAN-AdamW or within 10%
phi_prime_p95 <= 1.5x PureKAN-AdamW
```

如果 candidate 超过 PureKAN-AdamW：

```text
enter P5 10-seed final confirm
```

如果 candidate 接近但没超过：

```text
report as near-PureKAN functional optimizer, not solved
```

---

# 12. P5：10-seed final confirm

只有 P4 通过才跑。

## 12.1 方法

```text
PureKAN-AdamW
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
final PureKAN functional candidate
```

## 12.2 Seeds

```text
seeds = 0..9
```

## 12.3 最终 clean claim 条件

必须满足：

$$
\operatorname{Acc}_{candidate}
\geq
\max(
\operatorname{Acc}_{PureKAN-AdamW},
\operatorname{Acc}_{MLP-AdamW},
\operatorname{Acc}_{Hybrid-DGKAN-UFULL}
)
$$

在至少两个数据集上严格超过，并且第三个数据集不低于 `0.5%`。

同时：

```text
AUC not worse than AdamW
ECE better or equal
geometry better than AdamW or at least controlled
no implementation fallback dominates
```

如果不能满足，不允许写：

```text
PureKAN functional optimizer solved
```

只能写：

```text
PureKAN functional optimizer remains unsolved;
best candidate improves specific diagnostics but does not replace AdamW.
```

---

# 13. 必须生成的可视化

## 13.1 Direction dashboard

```text
figures/p1_direction_descent_scatter.png
```

x-axis:

```text
predicted descent
```

y-axis:

```text
actual train/val descent
```

颜色：

```text
method
```

目的：看谁是真正下降方向。

---

## 13.2 Target-fitting dashboard

```text
figures/ftf_target_fit_by_role.png
```

包括：

```text
fit_R2 by input/block/output
target_norm by role
residual_after / target_norm
ridge_condition
```

目的：判断 FTF 是否真的拟合了 local target。

---

## 13.3 Accuracy / AUC curves

```text
figures/p2_loss_curves_by_dataset.png
figures/p2_acc_curves_by_dataset.png
```

必须包含：

```text
PureKAN-AdamW
D0
D6
best FTF
best FC-Adam
best FTR
```

---

## 13.4 Representation dashboard

```text
figures/representation_effective_rank.png
figures/class_separation_ratio.png
figures/logit_margin_distribution.png
```

目的：判断 functional update 是否学出了好的 hidden representation，而不只是降低了几何指标。

---

## 13.5 Geometry vs accuracy Pareto

```text
figures/geometry_accuracy_pareto.png
```

x-axis:

```text
test_acc
```

y-axis:

```text
phi_prime_p95 或 max_jac_condition
```

目的：避免再次被“几何漂亮但 accuracy 低”误导。

---

## 13.6 Functional spectrum

```text
figures/functional_spectrum_by_layer.png
```

记录：

```text
coefficient energy by basis index
high-frequency energy
basis occupancy entropy
dead basis fraction
```

目的：看 functional update 是否过度压制高频表达。

---

## 13.7 Compute dashboard

```text
figures/compute_breakdown.png
```

包括：

```text
step_time_ms
solve_time_ms
cache_time_ms
memory_peak_mb
time_to_target_val_loss
```

目的：确认候选不是用不可接受的计算成本换结果。

---

# 14. 必须输出的 CSV / JSON

```text
results/v4_1/
  p0_invariants.csv
  p1_direction_audit.csv
  p2_micro_run_scorecard.csv
  p2_failure_diagnosis.csv
  p3_refinement_scorecard.csv
  p4_confirm5_scorecard.csv
  p5_confirm10_scorecard.csv
  functional_target_fit.csv
  representation_audit.csv
  geometry_audit.csv
  compute_audit.csv
  failure_table.csv
  aggregate_decision.json
```

`aggregate_decision.json` 至少包含：

```text
purekan_functional_solved
best_candidate
candidate_type = FTF / FC-Adam / FTR / none
passes_p1_direction
passes_p2_micro_run
passes_p4_confirm5
passes_p5_confirm10
main_failure_type
recommended_next_step
```

---

# 15. 推荐实验顺序

不要一口气跑全部。推荐顺序：

```text
Step 1:
  Implement P0 / P1 only.
  看 FTF / FC-Adam / FTR 哪个方向有 actual descent。

Step 2:
  只把 P1 过的候选跑 P2。
  如果 P1 全不过，停止训练，修 direction。

Step 3:
  P2 只选最多 3 个候选进入 P3。
  不要继续扫旧 Sobolev / D6 / depth-wise。

Step 4:
  P3 只选最多 2 个候选进入 P4。
  如果没有候选接近 AdamW，明确写 failure。

Step 5:
  P4 通过才进入 P5 10-seed。
```

---

# 16. 这轮之后可能的三种结论

## 16.1 最好情况

```text
FC-Adam 或 FTF/FTR 超过 PureKAN-AdamW。
```

结论：

$$
\boxed{
\text{PureKAN functional optimizer solved.}
}
$$

此时可以重新推进：

```text
Rational functional update
CIFAR-small
scaling
```

---

## 16.2 中等情况

```text
candidate 接近 AdamW，但没超过。
```

结论：

$$
\boxed{
\text{Functional update can approach AdamW when redesigned as target fitting / functional-coordinate optimization, but not yet surpass it.}
}
$$

下一步：

```text
capacity / basis / adaptive centers / longer training
```

---

## 16.3 失败情况

```text
FTF / FC-Adam / FTR 都明显低于 AdamW。
```

结论：

$$
\boxed{
\text{PureKAN full functional training remains unsolved; current evidence supports hybrid DG-KAN as the practical path.}
}
$$

这时不要继续微调 optimizer，而要重新审视：

```text
PureKAN architecture
basis parameterization
lack of linear mixing / normalization
task difficulty
```

---

# 17. 最终一句话

v4.1 的核心不是再修一个 optimizer 超参，而是回答：

$$
\boxed{
\text{PureKAN 需要的是 Sobolev preconditioner，还是 functional-coordinate Adam，还是 functional target fitting？}
}
$$

如果这个问题回答清楚，我们才能决定 functional update 是否能成为真正的 PureKAN optimizer，而不是只作为 Hybrid-DGKAN 的 branch optimizer。
