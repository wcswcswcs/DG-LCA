# DG-KAN v3.6 具体实验计划：Rational U-FULL 修复、U-FULL 失败诊断与 Pure-KAN 验证

> 版本：v3.6 draft  
> 日期：2026-05-02  
> 目标：这轮不再继续泛泛扫 optimizer，而是围绕三个非常具体的问题做实验：  
>
> 1. **Rational / KAT 上 functional update 为什么没有跑通，当前实现是不是 metric 错配或 bug。**  
> 2. **U-FULL 还有哪些没考虑的失败机制，尤其 f100、CIFAR-small、Rational U-FULL 的失败原因。**  
> 3. **能否实现没有非 KAN 参数的 Pure-KAN / Pure-DGKAN，让所有可学习参数都是 KAN coefficient，从而真正测试 functional update 是否能支撑纯 KAN 网络。**  
>
> 公式格式：Typora 友好，全文只使用 `$...$` 和 `$$...$$`，不使用 `方括号公式`。

---

## 0. 当前背景与本轮核心判断

v3.5 已经给出一个很清楚的局面：RBF-DGKAN 的 `U-FULL-f085-alphaFixed1` 仍然是当前最稳的统一 Pareto profile；`branch_final_scale=1.0` 的 `U-FULL-f100` 在 seed0 看起来很强，但 5-seed 后在 KMNIST 上没有通过，说明 branch scale 增大并不等价于有用表达力增强。CIFAR-small 上，ConvStem-DGKAN 的 U-FULL 明显失败，表现为 branch/A 和 no-KAN 极低、AUC 为负，说明 KAN branch 没有有效参与任务。Rational/KAT 的 AdamW 版本在 Fashion/KMNIST 上有可训练信号，但 Rational U-FULL 的 AUC 为负、Gram condition 明显偏高、clip 开始介入，说明当前 Rational functional update 还没有跑通。

因此本轮不能把 Rational/KAT AdamW 解释成“替代 U-FULL 主线”，也不能把 Rational U-FULL 失败解释成“U-FULL 只能用于 RBF”。更准确的判断是：

$$
\boxed{
\text{RBF-U-FULL 是已验证的机制平台；Rational/KAT 需要 primitive-specific functional metric。}
}
$$

当前 Rational U-FULL 的问题很可能不是普通调参问题，而是 metric 定义错配：RBF 的 Sobolev Gram 被硬套到了 Rational numerator / denominator 参数上。RBF edge function 是线性 basis 展开：

$$
\phi(t)=\sum_m a_m B_m(t),
$$

所以 Sobolev Gram

$$
M=
\int BB^\top dt
+
\alpha\int B'B'^\top dt
+
\beta\int B''B''^\top dt
+
\rho I
$$

是自然的。但 Rational/KAT 的函数更接近：

$$
r(x;\theta)=\frac{P(x)}{D(x)}
$$

或 KAT 实现中的分组 rational activation。它对 denominator 参数是非线性的，对 numerator 和 denominator 也是耦合的。因此 Rational 的 functional metric 应该基于函数对参数的 tangent：

$$
J_\theta(x)=\frac{\partial r(x;\theta)}{\partial \theta}.
$$

更合理的 metric 是：

$$
M_\theta
=
\int J_\theta(x)^\top J_\theta(x)dx
+
\alpha\int \frac{\partial J_\theta(x)^\top}{\partial x}
\frac{\partial J_\theta(x)}{\partial x}dx
+
\rho I.
$$

这就是本轮的第一条主线：**把 Rational functional update 从 polynomial coefficient metric 改成 tangent Sobolev metric**。

第二条主线是审视 U-FULL：`f100` 失败、CIFAR 失败、Rational 失败不是同一种失败。我们要把它们拆开诊断：

```text
f100 failure:
  branch norm 更大，但任务对齐可能变差。

CIFAR failure:
  branch under-active，可能是 ConvStem feature 分布、RBF basis occupancy、early full-Sobolev 过保守共同导致。

Rational failure:
  functional metric 与 rational 参数几何不匹配，condition 高，direction 可能错误。
```

第三条主线是 Pure-KAN：如果我们一直在 hybrid 架构里讨论“非 KAN 参数是否需要 AdamW”，结论会被 stem/head/LayerNorm 干扰。最直接的验证是实现一个没有 learnable non-KAN 参数的 Pure-KAN classifier。这样所有可学习参数都是 KAN coefficients，U-FULL 不再需要依赖 AdamW rest optimizer。

---

## 1. 本轮研究问题

### 1.1 Rational functional update 是否只是实现错了？

本轮要明确回答：

$$
\boxed{
\text{Rational/KAT U-FULL 当前失败，是 primitive 本身不适合 functional update，还是 metric 实现不对？}
}
$$

优先假设是 metric 实现不对。需要验证以下具体可能性：

```text
H1: 当前 polynomial Sobolev Gram condition 太高，导致 direction 不可靠。
H2: numerator 和 denominator 被分开 precondition，但真实 Rational 函数需要 joint metric。
H3: 当前 denominator safety 只看参数，不看实际 denominator function。
H4: branch 内 linear mixing 没有纳入 functional branch update，导致 activation functional update 和 linear AdamW 目标错位。
H5: 固定 grid metric 没有匹配实际 input distribution。
```

### 1.2 U-FULL 是否还漏了重要机制？

本轮要明确回答：

$$
\boxed{
\text{U-FULL 的成功和失败是否由 branch norm 决定，还是由 branch-task alignment 决定？}
}
$$

f100 的结果显示 `branch/A` 上升不一定带来 no-KAN 上升，所以必须补充：

```text
KAN correct-class delta
KAN margin contribution
branch-to-margin alignment
class-wise no-KAN drop
per-layer branch/no-KAN contribution
```

CIFAR 失败则要补充：

```text
KAN input distribution
RBF basis occupancy
dead basis fraction
out-of-grid fraction
ConvStem output norm
per-layer branch contribution
functional update norm
```

### 1.3 Pure-KAN 是否能用 U-FULL 训练？

本轮要明确回答：

$$
\boxed{
\text{如果模型里所有可学习参数都是 KAN coefficients，functional update 是否能独立支撑训练？}
}
$$

如果 PureKAN-UFULL 成功，说明 functional update 可以成为纯 KAN 网络的核心 optimizer。如果失败，则说明当前 U-FULL 更适合作为 hybrid branch optimizer，而不是完整网络 optimizer。

---

## 2. 实现改动总览

本轮建议新增或修改以下文件：

```text
experiments/dgkan_core.py
  增加 Rational tangent Sobolev metric
  增加 Rational actual denominator audit
  增加 Rational branch-linear functional update 选项
  修改 coefficient_named_params 为 module-type based 或至少兼容 pure KAN
  增加 PureKANClassifier / PureResidualKANBlock / FixedNorm
  增加 basis occupancy / input distribution audit
  增加 branch alignment audit

experiments/run_gafu_v36.py
  新增 packages:
    V3_6_P0_IMPL_SMOKE
    V3_6_P1_RATIONAL_METRIC_AUDIT
    V3_6_P2_RATIONAL_TANGENT_TOY
    V3_6_P3_RATIONAL_DGKAN_TANGENT3
    V3_6_P4_RATIONAL_DGKAN_CONFIRM
    V3_6_P5_UFULL_F100_FAILURE_AUDIT
    V3_6_P6_CIFAR_BRANCH_AUDIT
    V3_6_P7_PUREKAN_SMOKE
    V3_6_P8_PUREKAN_3SEED
    V3_6_P9_PUREKAN_5SEED

experiments/analyze_gafu_v36.py
  聚合 direction audit、shadow-step audit、branch alignment、basis occupancy、pure KAN scorecard
```

---

## 3. 关键实现要求

### 3.1 改掉 name-only 的 KAN coefficient 识别

当前参数识别容易依赖命名规则，例如只识别：

```text
kan.coeff
kan.act.weight_numerator
kan.act.weight_denominator
kan.kat_group.weight_numerator
kan.kat_group.weight_denominator
```

本轮 PureKAN 和 Rational branch-level update 会让命名变复杂。建议改成 module-type based：

```python
def coefficient_named_params(model):
    out = []
    for module_name, module in model.named_modules():
        if isinstance(module, RBFDense):
            out.append((f"{module_name}.coeff", module.coeff))
        if isinstance(module, KAT_Group_like):
            out.append((f"{module_name}.weight_numerator", module.weight_numerator))
            out.append((f"{module_name}.weight_denominator", module.weight_denominator))
    return out
```

如果第三方 KAT class 不方便 isinstance，则使用 duck typing：

```python
hasattr(module, "weight_numerator") and hasattr(module, "weight_denominator")
```

这个改动是 P0 的 hard gate。否则 PureKAN 的 input/output KAN coeff 可能漏更新。

---

### 3.2 Rational actual denominator audit

当前 denominator safety 不能只看参数：

$$
1+|\theta_{den}|.
$$

要在 batch 和 grid 上计算实际 denominator function：

$$
D_g(x)
$$

并记录：

```text
rational/den_actual_min_batch
rational/den_actual_p01_batch
rational/den_actual_median_batch
rational/den_actual_max_batch
rational/den_actual_condition_batch
rational/den_actual_min_grid
rational/den_actual_p01_grid
```

如果 KAT denominator 形式是第三方实现内部逻辑，需要写一个与 forward 一致的 helper。若一开始无法完全复刻，就至少通过 autograd wrapper 对 `act(x)` 的内部 denominator proxy 做审计，不能继续只看参数绝对值。

---

### 3.3 Rational tangent Sobolev metric

本轮新增三个 metric mode：

```text
rational_poly_separate
rational_poly_diag
rational_tangent_joint
rational_tangent_diag
```

#### 3.3.1 当前 separate polynomial metric

这个保留为对照，不作为推荐：

$$
M^{poly}_{num}
=
\int p_i(x)p_j(x)dx
+
\alpha\int p_i'(x)p_j'(x)dx
+
\rho I.
$$

denominator 单独一个：

$$
M^{poly}_{den}
=
\int q_i(x)q_j(x)dx
+
\alpha\int q_i'(x)q_j'(x)dx
+
\rho I.
$$

它的问题是没有 numerator-denominator coupling。

#### 3.3.2 Joint tangent metric

对每个 group，把参数拼起来：

$$
\theta_g=[a_{g,0},...,a_{g,m},b_{g,1},...,b_{g,n}].
$$

在 grid 或 batch input 上计算：

$$
J_g(x_q)=\frac{\partial r_g(x_q;\theta_g)}{\partial \theta_g}.
$$

然后构造：

$$
M_g
=
\Delta x\sum_q J_g(x_q)^\top J_g(x_q)
+
\alpha\Delta x\sum_q J'_g(x_q)^\top J'_g(x_q)
+
\rho I.
$$

其中：

$$
J'_g(x_q)=\frac{\partial}{\partial x}J_g(x_q).
$$

第一版可以只做 $J^\top J+\rho I$，不加 $J'_x$，这样先验证 direction 是否正常：

```text
rational_tangent_joint_L2
```

第二版再加 input-derivative：

```text
rational_tangent_joint_H1
```

如果计算 $J'_g$ 太慢，可以先用 finite difference：

$$
J'_g(x)\approx\frac{J_g(x+\epsilon)-J_g(x-\epsilon)}{2\epsilon}.
$$

建议：

```text
grid_size = 128 for tangent smoke
grid_size = 256 for 3-seed
rho in {1e-2, 1e-1}
```

P0 要求 tangent Gram condition 小于当前 polynomial Gram 的一半，或者至少低于 $1e4$。

---

### 3.4 Rational branch linear update

当前 Rational branch 是：

$$
y=W r(x;\theta)+b.
$$

当前 functional update 主要作用于 $\theta$，而 $W,b$ 仍然属于 non-coefficient 参数。新增两个 branch update 设定：

```text
Rational-UFULL-act-only:
  只更新 numerator / denominator
  branch linear W/b 仍然 AdamW

Rational-UFULL-branch:
  numerator / denominator 用 tangent metric
  branch linear W 用 activation covariance metric
  branch linear bias 用 diagonal update
```

对 branch linear 的 metric：

$$
G_r=\mathbb E[r(x)r(x)^\top]+\rho I.
$$

更新：

$$
\Delta W=-\eta_W \nabla_W L G_r^{-1}.
$$

这个实验是为了判断 current Rational U-FULL 是否因为 branch 内部一半 AdamW、一半 functional 而错位。

---

### 3.5 PureKANClassifier

实现一版没有 learnable non-KAN 参数的纯 KAN。

#### 3.5.1 模型结构

```text
x
  -> input KAN: input_dim -> hidden_dim
  -> fixed normalization
  -> residual KAN blocks
  -> fixed normalization
  -> output KAN: hidden_dim -> num_classes
```

数学形式：

$$
h_0=\operatorname{KAN}_{in}(x)
$$

$$
h_{l+1}=h_l+s_l\operatorname{KAN}_l(\operatorname{FixedNorm}(h_l))
$$

$$
z=\operatorname{KAN}_{out}(\operatorname{FixedNorm}(h_L)).
$$

其中 `FixedNorm` 没有 learnable $\gamma,\beta$：

$$
\operatorname{FixedNorm}(h)=\frac{h-\mu(h)}{\sigma(h)+\epsilon}.
$$

#### 3.5.2 learnable 参数限制

PureKAN 必须满足：

```text
learnable_nonkan_param_count = 0
learnable_kan_param_count > 0
```

可选 bias 处理有两种：

```text
PureKAN-no-bias:
  RBFDense 不带 bias

PureKAN-constant-basis-bias:
  bias 作为 constant basis coefficient
```

第一轮建议用 no-bias，避免争议。

#### 3.5.3 参数识别

PureKAN 的 input/output layer 也必须被 `coefficient_named_params()` 正确识别。P0 要输出：

```text
purekan/learnable_total
purekan/learnable_kan_coeff
purekan/learnable_nonkan
purekan/coeff_param_names
```

---

## 4. P0：实现 smoke 与一致性检查

### 4.1 目标

P0 不做方法结论，只确认实现没有明显 bug：

```text
Rational tangent metric 能构建；
actual denominator audit 有值；
branch linear update 能记录；
PureKAN 没有 non-KAN learnable 参数；
coefficient_named_params 没漏参数；
所有 smoke run 无 NaN。
```

### 4.2 smoke 设置

```text
datasets = Fashion-MNIST, KMNIST
train/val/test = 512/128/128
epochs = 1
seeds = 0
hidden_dim = 32
depth = 2
basis_count = 8
rational_groups = 4
```

### 4.3 方法

```text
Rational-AdamW-smoke
Rational-UFULL-poly-separate-smoke
Rational-UFULL-tangent-joint-smoke
Rational-UFULL-branch-smoke
PureKAN-AdamW-smoke
PureKAN-UFULL-smoke
Hybrid-UFULL-smoke
```

### 4.4 必须记录

```text
run_failed
nan_or_inf_count
learnable_total_params
learnable_kan_params
learnable_nonkan_params
coeff_param_count
coeff_param_names_hash
rational_metric_condition
rational_metric_eig_min
rational_metric_eig_max
rational_den_actual_min_batch
rational_den_actual_p01_batch
rational_den_actual_condition_batch
trust_clip_rate
update_over_coeff_norm_p95
purekan_nonkan_param_count
```

### 4.5 通过条件

P0 通过要求：

$$
\text{run failures}=0.
$$

$$
\text{all tracked metrics finite}.
$$

PureKAN 必须满足：

$$
\text{learnable non-KAN params}=0.
$$

Rational tangent metric 至少要满足：

$$
\operatorname{cond}(M_{tangent}) < 10^5
$$

作为 smoke 上限。如果它仍然和当前 polynomial metric 一样高，需要先调 $\rho$ 和 grid，不进入 P1。

---

## 5. P1：Rational current metric audit

### 5.1 目标

P1 不直接追 accuracy，而是判断 current Rational U-FULL 失败是否来自 metric condition 和 direction 错配。

### 5.2 设置

```text
datasets = Fashion-MNIST, KMNIST
model = Rational-DGKAN
hidden_dim = 96
depth = 4
groups = 8
alpha_mode = fixed1
epochs = 20 for KMNIST, 30 for Fashion
seeds = 0,1,2
```

### 5.3 方法

```text
Rational-AdamW
Rational-UFULL-poly-separate-current
Rational-UFULL-poly-diag
Rational-UFULL-poly-rho1e-2
Rational-UFULL-poly-rho1e-1
Rational-UFULL-identity
Rational-UFULL-legendre-separate
```

解释：

```text
poly-separate-current:
  当前实现，作为失败复现。

poly-diag:
  只用 polynomial Gram diagonal，看 full inverse 是否方向有问题。

rho1e-2 / rho1e-1:
  检查 condition 是否主因。

identity:
  不做 metric precondition，只做 functional coefficient SGD-style update。

legendre-separate:
  用 orthogonal polynomial basis 替代 monomial / abs-power，检查 basis conditioning。
```

### 5.4 记录指标

每个 run 必须记录：

```text
task/test_acc
task/val_auc
task/ece
branch/branch_over_adamw
branch/no_kan_drop
branch/no_kan_drop_over_adamw
kan/rational_den_actual_min_batch
kan/rational_den_actual_p01_batch
kan/rational_r_prime_p95
kan/rational_r_double_prime_p95
metric/condition
metric/eig_min
metric/eig_max
metric/clip_rate
direction/raw_grad_norm
direction/precond_direction_norm
direction/precond_over_raw
direction/cos_raw_precond
direction/update_over_coeff_norm_p95
shadow/predicted_descent
shadow/actual_descent
shadow/bad_shadow_step_rate
```

### 5.5 可视化

P1 dashboard 必须包括：

```text
metric condition by method
AUC improvement vs metric condition
clip rate vs metric condition
precond_over_raw vs AUC
cos_raw_precond vs AUC
den_actual_p01 over epoch
r_prime_p95 over epoch
shadow predicted vs actual descent scatter
```

### 5.6 判定

如果 `poly-diag` 或 `rho1e-1` 明显好于 `poly-separate-current`，例如：

$$
\operatorname{AUCImp}_{diag} > \operatorname{AUCImp}_{current} + 3\%
$$

且：

$$
\operatorname{cond}(M)<10^4,
$$

则说明 current full polynomial inverse 是主要问题。

如果所有 polynomial variants 都不行，但 AdamW 正常，则进入 P2 tangent metric，不再继续调 polynomial。

---

## 6. P2：Rational tangent metric toy sanity

### 6.1 目标

在小任务上确认 tangent metric 方向是否正确。不要先在完整分类模型上试错。

### 6.2 Toy 任务

构造 group rational activation 拟合：

```text
target_1: y = silu(x)
target_2: y = tanh(x) + 0.2 sin(3x)
target_3: y = smoothstep(x)
```

输入：

$$
x\in[-2.5,2.5].
$$

方法：

```text
Adam on rational params
poly-separate-current
poly-diag
tangent-joint-L2
tangent-joint-H1
tangent-diag
```

### 6.3 记录指标

```text
train_mse
val_mse
loss_auc
r_prime_p95
r_double_prime_energy
metric_condition
cos_raw_precond
predicted_actual_descent_corr
bad_step_rate
den_actual_min
den_actual_p01
num_update_norm
den_update_norm
num_den_update_cos
```

### 6.4 可视化

```text
target vs fitted function
loss curve
r_prime curve
denominator curve over x
metric eig spectrum
gradient direction cosine heatmap
predicted vs actual descent scatter
```

### 6.5 通过条件

Tangent metric 进入 P3 的条件：

$$
\operatorname{bad\_step\_rate}<5\%.
$$

$$
\operatorname{AUC}_{tangent}\leq1.05\operatorname{AUC}_{Adam}
$$

或至少优于 current polynomial：

$$
\operatorname{AUC}_{tangent}<0.8\operatorname{AUC}_{poly-current}.
$$

并且：

$$
\operatorname{denominator\_p01}>0.5.
$$

---

## 7. P3：Rational-DGKAN tangent metric 3-seed

### 7.1 目标

验证 tangent metric 能否让 Rational U-FULL 从负 AUC / high-condition 状态恢复。

### 7.2 设置

```text
datasets = Fashion-MNIST, KMNIST
model = Rational-DGKAN
hidden_dim = 96
depth = 4
groups = 8
alpha_mode = fixed1
branch_final_scale = 1.0
seeds = 0,1,2
```

### 7.3 方法

```text
Rational-AdamW
Rational-UFULL-poly-current
Rational-UFULL-tangent-act-L2
Rational-UFULL-tangent-act-H1
Rational-UFULL-tangent-branch-L2
Rational-UFULL-tangent-branch-H1
Rational-UFULL-tangent-diag
RBF-UFULL-f085-reference
```

### 7.4 记录指标

除了 P1 指标，还要记录：

```text
branch_linear/update_norm
branch_linear/update_over_param
branch_linear/cov_condition
branch_linear/cos_raw_precond
rational/num_update_norm
rational/den_update_norm
rational/num_den_update_ratio
rational/num_den_update_cos
rational/group_function_diversity
rational/group_dead_fraction
```

### 7.5 判定

Rational-UFULL-tangent 初步成功要求：

Fashion / KMNIST 都满足：

$$
\text{AUC improvement} > 0.
$$

$$
\text{test acc gap vs Rational-AdamW} < 1.0\%.
$$

$$
\text{trust clip rate}<2\%.
$$

$$
\operatorname{cond}(M_{tangent})<10^4.
$$

如果 tangent-act 成功但 tangent-branch 失败，说明 branch linear covariance update 不成熟，保留 act-only。  
如果 tangent-branch 成功且 expression/ECE 更好，则 Rational functional update 的主线改为 branch-level metric。

---

## 8. P4：Rational-DGKAN 5-seed confirm

P4 只在 P3 有候选通过时运行。

### 8.1 方法

```text
Rational-AdamW
Rational-UFULL-best-tangent
RBF-UFULL-f085
MLP-AdamW
```

### 8.2 设置

```text
datasets = Fashion-MNIST, KMNIST
seeds = 0..4
```

### 8.3 成功标准

Rational functional update 不能只看 accuracy。它必须满足：

$$
\text{acc gap vs Rational-AdamW}<0.5\%
$$

$$
\text{AUC improvement vs Rational-AdamW}>3\%
$$

$$
\text{ECE reduction vs Rational-AdamW}>5\%
$$

$$
\text{clip rate}<2\%
$$

并且 wall-clock 不得超过 Rational-AdamW 太多：

$$
\text{time ratio}<1.5.
$$

如果这个不成立，Rational/KAT 仍然只能作为 AdamW scaling primitive，不能 claim functional update 成功。

---

## 9. P5：U-FULL f100 failure audit

### 9.1 目标

解释为什么 `branch_final_scale=1.0` seed0 好但 5-seed 不稳，以及为什么 branch/A 上升但 no-KAN 下降。

### 9.2 方法

```text
AdamW-alphaFixed1
U-FULL-f085
U-FULL-f100
F-V3-HARD Fashion only
K-V3-FULL-base KMNIST only
```

### 9.3 设置

```text
datasets = Fashion-MNIST, KMNIST
seeds = 0..4
reuse existing checkpoints if available
```

### 9.4 新增审计指标

每个 epoch 或 final audit 记录：

```text
branch/layer_k_norm_ratio
branch/layer_k_no_kan_drop_proxy
branch/layer_k_logit_delta_norm
branch/layer_k_margin_contribution
branch/correct_class_delta_mean
branch/wrong_class_delta_mean
branch/margin_alignment_cos
branch/classwise_no_kan_drop
branch/classwise_acc_delta_full_vs_noKAN
branch/confidence_correct_delta
branch/confidence_wrong_delta
```

定义 margin contribution：

$$
\Delta margin
=
[z_y-\max_{c\ne y}z_c]_{full}
-
[z_y-\max_{c\ne y}z_c]_{noKAN}.
$$

定义 branch-to-margin alignment：

$$
\cos_{align}
=
\frac{\langle \Delta z_{KAN}, e_y-e_{c^*}\rangle}
{\|\Delta z_{KAN}\|\cdot\|e_y-e_{c^*}\|+\epsilon}.
$$

其中 $c^*$ 是 no-KAN logits 中最大的错误类。

### 9.5 可视化

```text
branch/A vs noKAN scatter
branch/A vs margin contribution scatter
f085 vs f100 class-wise noKAN heatmap
correct-class delta distribution
margin contribution distribution
ECE reliability diagram for f085/f100
per-layer branch ratio bars
per-layer margin contribution bars
```

### 9.6 判定

如果 f100 的 branch/A 更高但 margin contribution 更低，则确认：

$$
\boxed{
\text{f100 failure = branch amplitude increase without task-aligned contribution.}
}
$$

如果 f100 在某些 class 上 noKAN 大幅下降，则需要写 class-specific failure。

---

## 10. P6：CIFAR U-FULL branch under-active audit

### 10.1 目标

解释 ConvStem-DGKAN 上 U-FULL 为什么 branch/noKAN 极低。不要先调 optimizer，先找原因。

### 10.2 方法

```text
ConvStem-DGKAN-AdamW
ConvStem-DGKAN-UFULL-f085
ConvStem-DGKAN-UFULL-f100
ConvStem-DGKAN-UFULL-diag-warmup-probe
ConvStem-DGKAN-UFULL-branchboost-probe
```

后两个只作为 probe，不进入结论。

### 10.3 设置

```text
dataset = CIFAR10 small
train/val/test = 10000/2000/2000 或当前设置
seeds = 0,1,2
```

### 10.4 必须记录

```text
convstem/output_mean
convstem/output_std
convstem/output_p95
kan_input/layer_k_mean
kan_input/layer_k_std
kan_input/layer_k_p01
kan_input/layer_k_p99
rbf/basis_occupancy_mean
rbf/basis_occupancy_min
rbf/dead_basis_fraction
rbf/out_of_grid_fraction
branch/layer_k_ratio
branch/layer_k_noKAN_proxy
update/raw_grad_norm_layer_k
update/precond_direction_norm_layer_k
update/update_over_coeff_norm_layer_k
head/margin_contribution
head/logit_norm
```

### 10.5 可视化

```text
KAN input histogram by layer
RBF basis occupancy heatmap
dead basis fraction by layer
branch ratio by layer over epoch
noKAN drop over epoch
raw_grad vs precond_norm by layer
ConvStem output norm vs branch ratio scatter
```

### 10.6 判定

如果 basis occupancy 异常，例如：

$$
\text{dead basis fraction}>0.3
$$

或：

$$
\text{out-of-grid fraction}>0.1,
$$

则 CIFAR failure 主要是 basis-interface mismatch。

如果 occupancy 正常但 update norm 极低，则是 functional step too conservative。

如果 branch norm 正常但 noKAN 极低，则是 branch task alignment failure。

---

## 11. P7：PureKAN smoke

### 11.1 目标

确认 PureKAN 实现真的没有 non-KAN learnable 参数，并且能跑通一轮训练。

### 11.2 设置

```text
datasets = Fashion-MNIST, KMNIST
train/val/test = 512/128/128
epochs = 1
seeds = 0
hidden_dim = 32
depth = 2
basis_count = 8
```

### 11.3 方法

```text
PureKAN-AdamW-smoke
PureKAN-UFULL-smoke
Hybrid-UFULL-smoke
MLP-AdamW-smoke
```

### 11.4 记录指标

```text
purekan/learnable_total
purekan/learnable_kan_coeff
purekan/learnable_nonkan
purekan/has_linear
purekan/has_layernorm_params
purekan/has_bias_params
run_failed
nan_count
val_loss
test_acc
metric_condition
trust_clip_rate
```

### 11.5 通过条件

$$
\text{learnable non-KAN params}=0.
$$

$$
\text{run failures}=0.
$$

$$
\text{all metrics finite}.
$$

---

## 12. P8：PureKAN 3-seed capability test

### 12.1 目标

判断 PureKAN 是否有基本任务能力，以及 PureKAN-UFULL 是否接近 PureKAN-AdamW。

### 12.2 设置

```text
datasets = MNIST, Fashion-MNIST, KMNIST
train/val/test = 6000/1000/1000
hidden_dim = 64, 96
depth = 2, 4
basis_count = 16, 24
seeds = 0,1,2
```

第一轮建议先固定：

```text
hidden_dim = 64
depth = 2
basis_count = 16
```

如果完全欠拟合，再加 capacity。

### 12.3 方法

```text
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
PureKAN-AdamW
PureKAN-UFULL
PureKAN-UFULL-diagwarmup
```

### 12.4 记录指标

```text
task/test_acc
task/val_auc
task/ece
purekan/param_count
purekan/coeff_update_norm
purekan/update_over_coeff_norm
kan/phi_prime_p95
kan/jacobian_condition
basis/active_basis_fraction
basis/dead_basis_fraction
basis/out_of_grid_fraction
branch/noKAN_equivalent
logit/margin
```

PureKAN 没有 disable_kan 的同义操作时，记录 layer ablation：

```text
disable_residual_block_k
disable_input_kan
disable_output_kan
```

### 12.5 判定

PureKAN-UFULL 初步成功：

$$
\text{acc gap vs PureKAN-AdamW}<1.5\%
$$

$$
\text{AUC improvement vs PureKAN-AdamW}>3\%
$$

$$
\phi'\text{ reduction}>15\%
$$

如果 PureKAN-AdamW 自己很差，则说明纯 KAN 架构能力不足，先不要评价 U-FULL。

---

## 13. P9：PureKAN 5-seed confirm

只在 P8 通过后运行。

### 13.1 方法

```text
PureKAN-AdamW
PureKAN-UFULL-best
Hybrid-UFULL-f085
MLP-AdamW
```

### 13.2 设置

```text
datasets = Fashion-MNIST, KMNIST
seeds = 0..4
best capacity from P8
```

### 13.3 成功标准

如果 PureKAN-UFULL 满足：

$$
\text{acc gap vs PureKAN-AdamW}<1.0\%
$$

$$
\text{AUC improvement}>5\%
$$

$$
\text{geometry reduction}>15\%
$$

则可以 claim：

```text
Pure KAN can be trained by functional update.
```

如果 PureKAN-UFULL 失败但 PureKAN-AdamW 成功，说明 functional update 对 full pure KAN 仍不够。  
如果两者都失败，说明架构不够，不是 optimizer 单独问题。

---

## 14. 最终决策规则

### 14.1 Rational functional update

结论分三类：

```text
A. Tangent metric succeeds:
   Rational/KAT functional update becomes viable.
   下一步进入 5/10-seed 和 CIFAR scaling。

B. Tangent metric improves but not enough:
   Rational/KAT remains scaling primitive under AdamW.
   functional metric 作为 ongoing research。

C. Tangent metric also fails:
   Rational/KAT functional update 暂停。
   结论是 current rational parameterization 不适合当前 functional update。
```

### 14.2 U-FULL failure diagnosis

必须产出明确 failure mode：

```text
f100:
  amplitude-not-alignment 或 class-specific failure

CIFAR:
  basis mismatch / functional step too conservative / branch-task alignment failure

Rational:
  metric mismatch / condition failure / denominator instability / branch-linear mismatch
```

### 14.3 PureKAN

结论分三类：

```text
A. PureKAN-UFULL succeeds:
   functional update 可以支撑无 non-KAN 参数网络。

B. PureKAN-AdamW succeeds but UFULL fails:
   纯 KAN 架构可行，但 U-FULL 仍需改。

C. PureKAN-AdamW fails:
   当前 pure architecture 不足，不能用于评价 optimizer。
```

---

## 15. 必须输出的文件

每个 package 输出：

```text
runs.csv
summary_by_method.csv
failure_table.csv
direction_audit.csv
shadow_step_audit.csv
branch_alignment_audit.csv
basis_occupancy_audit.csv
metric_spectrum.csv
```

总报告输出：

```text
docs/DG-KAN_v3.6_RationalUFull_PureKAN_结果复盘.md
results/gafu_v3_6_decision.json
```

图表输出：

```text
plots/rational_metric_condition_vs_auc.png
plots/rational_direction_cos_heatmap.png
plots/rational_denominator_curve.png
plots/f085_f100_branch_alignment.png
plots/cifar_basis_occupancy_heatmap.png
plots/cifar_branch_layer_curve.png
plots/purekan_scorecard.png
plots/purekan_auc_geometry_pareto.png
```

---

## 16. 本轮不做什么

本轮不要做：

```text
继续大范围 branch_final_scale sweep
继续 PGAdam / PostAdam
继续 Full AFU non-KAN functional update
直接把 Rational AdamW 当主线替代
直接跑 CIFAR 大规模 confirm
```

原因是当前问题不是缺少更多 profile，而是需要解释为什么 U-FULL 在三个场景失败，以及如何给 Rational 设计正确 functional metric。

---

## 17. 一句话总结

本轮主线是：

$$
\boxed{
\text{把 U-FULL 从 RBF 机制平台推进到 primitive-specific functional update。}
}
$$

具体就是：

```text
RBF:
  已经有 U-FULL-f085 机制结果，继续做 failure audit。

Rational/KAT:
  重新设计 tangent functional metric，而不是硬套 polynomial Sobolev。

PureKAN:
  去掉 non-KAN 参数干扰，测试 functional update 是否能支撑纯 KAN 网络。

CIFAR:
  先做 branch / basis / interface 诊断，不急着调 profile。
```
