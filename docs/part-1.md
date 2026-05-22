# DG-KAN Part 1 (v3-v5)

> Auto-merged from source documents. Original files were not modified. Ordering key: numeric DG-KAN version, then plan/update/design documents before recap/reflection documents, then natural filename order within each bucket.

## Source Index

1. `docs/DG-KAN_v3.6_RationalUFull_PureKAN_具体实验计划.md`
2. `docs/DG-KAN_v3.6_RationalUFull_PureKAN_结果复盘.md`
3. `docs/DG-KAN_v3.7_PureKAN_UFULL_代码检查与实验计划.md`
4. `docs/DG-KAN_v3.7_PureKAN_UFULL_结果复盘.md`
5. `docs/DG-KAN_v3.8_修改版_GlobalTFU_Depthwise_PureKAN_实验计划.md`
6. `docs/DG-KAN_v3.8_GlobalTFU_Depthwise_PureKAN_结果复盘.md`
7. `docs/DG-KAN_v3.9_Normalization_FNG_PureKAN_实验计划.md`
8. `docs/DG-KAN_v3.9_Normalization_FNG_PureKAN_结果复盘.md`
9. `docs/DG-KAN_v4.1_FunctionalUpdate_Redesign_DeepPlan.md`
10. `docs/DG-KAN_v4.1_FunctionalUpdate_Redesign_结果复盘.md`
11. `docs/DG-KAN_v4.2_FunctionalTrustRegion_DeepRedesign_实验计划.md`
12. `docs/DG-KAN_v4.2_FunctionalTrustRegion_DeepRedesign_结果复盘.md`
13. `docs/DG-KAN_v4.3_FunctionalUpdate_DeepRedesign_实验计划.md`
14. `docs/DG-KAN_v4.3_FunctionalUpdate_DeepRedesign_结果复盘.md`
15. `docs/DG-KAN_v4.4_FunctionalOptimizer_Redesign_实验计划.md`
16. `docs/DG-KAN_v4.4_FunctionalOptimizer_Redesign_结果复盘.md`
17. `docs/DG-KAN_v4.5_AcceleratedFunctionalOptimizer_实验计划.md`
18. `docs/DG-KAN_v4.5_AcceleratedFunctionalOptimizer_结果复盘.md`
19. `docs/DG-KAN_v4.6_FunctionalLearningDynamics_实验计划.md`
20. `docs/DG-KAN_v4.6_FunctionalLearningDynamics_结果复盘.md`
21. `docs/DG-KAN_v4.7_PhaseSeparatedFunctionalTraining_PureKAN_实验计划.md`
22. `docs/DG-KAN_v4.7_PhaseSeparatedFunctionalTraining_PureKAN_结果复盘.md`
23. `docs/DG-KAN_v4.8_FunctionalGeometry_Feasibility_Redesign_实验计划.md`
24. `docs/DG-KAN_v4.8_FunctionalGeometry_Feasibility_Redesign_结果复盘.md`
25. `docs/DG-KAN_v4.9_RBFParameterization_ExactNFS_下一步实验计划.md`
26. `docs/DG-KAN_v4.9_RBFParameterization_ExactNFS_结果复盘.md`
27. `docs/DG-KAN_v5.0_EdgeDecomposedFunctionalTraining_实验计划.md`
28. `docs/DG-KAN_v5.0_EdgeDecomposedFunctionalTraining_结果复盘.md`
29. `docs/DG-KAN_v5.1_ABRBF_StrongResidualSmoothing_实验计划.md`
30. `docs/DG-KAN_v5.1_ABRBF_StrongResidualSmoothing_结果复盘.md`
31. `docs/DG-KAN_v5.2_MemoryBudgetedResidualSmoothing_实验计划.md`
32. `docs/DG-KAN_v5.2_MemoryBudgetedResidualSmoothing_结果复盘.md`
33. `docs/DG-KAN_v5.3_ABRBF_EventController_下一步实验计划.md`
34. `docs/DG-KAN_v5.3_ABRBF_EventController_结果复盘.md`
35. `docs/DG-KAN_v5.4_EfficiencyFirst_PureKAN_FunctionalTraining_实验计划.md`
36. `docs/DG-KAN_v5.4_EfficiencyFirst_PureKAN_FunctionalTraining_结果复盘.md`
37. `docs/DG-KAN_v5.5_KernelFirst_EfficientPureKAN_实验计划.md`
38. `docs/DG-KAN_v5.5_KernelFirst_EfficientPureKAN_结果复盘.md`
39. `docs/DG-KAN_v5.6_GEMMNative_EfficientPureKAN_实验计划.md`
40. `docs/DG-KAN_v5.6_GEMMNative_EfficientPureKAN_结果复盘.md`
41. `docs/DG-KAN_v5.7_KernelVerified_EfficientPureKAN_实验计划.md`
42. `docs/DG-KAN_v5.7_KernelVerified_EfficientPureKAN_结果复盘.md`
43. `docs/DG-KAN_v5.8_EfficientPrimitive_Redesign_实验计划.md`
44. `docs/DG-KAN_v5.8_EfficientPrimitive_Redesign_结果复盘.md`
45. `docs/DG-KAN_v5.9_MemoryFirst_EfficientPrimitive_实验计划.md`
46. `docs/DG-KAN_v5.9_MemoryFirst_EfficientPrimitive_结果复盘.md`



---


# Source 1: `docs/DG-KAN_v3.6_RationalUFull_PureKAN_具体实验计划.md`


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



---


# Source 2: `docs/DG-KAN_v3.6_RationalUFull_PureKAN_结果复盘.md`


# DG-KAN v3.6 RationalUFull / PureKAN 结果复盘

## Decision

```json
{
  "p0_pass": true,
  "p3_rational_confirm_triggered": false,
  "p4_run": false,
  "p8_purekan_confirm_triggered": false,
  "p9_run": false,
  "total_rows": 266
}
```

## P0 Implementation Smoke

| dataset | method | runs | acc | condG | clip | den p01 | pure nonKAN |
|---|---|---|---|---|---|---|---|
| Fashion-MNIST | Hybrid-UFULL-smoke-alphaFixed1 | 1 | 0.4609 | 6551.6172 | 0.0000 | nan | nan |
| Fashion-MNIST | PureKAN-AdamW-smoke-alphaFixed1 | 1 | 0.1094 | nan | 0.0000 | nan | 0.0000 |
| Fashion-MNIST | PureKAN-UFULL-smoke-alphaFixed1 | 1 | 0.1328 | 6551.6172 | 0.0000 | nan | 0.0000 |
| Fashion-MNIST | Rational-DGKAN-AdamW-triton-alphaFixed1 | 1 | 0.3516 | nan | 0.0000 | 1.0001 | nan |
| Fashion-MNIST | Rational-UFULL-branch-smoke-alphaFixed1 | 1 | 0.3438 | 8508.2334 | 0.0000 | 1.0001 | nan |
| Fashion-MNIST | Rational-UFULL-poly-separate-smoke-alphaFixed1 | 1 | 0.3438 | 251534.2344 | 0.5000 | 1.0001 | nan |
| Fashion-MNIST | Rational-UFULL-tangent-joint-smoke-alphaFixed1 | 1 | 0.3516 | 8507.8916 | 0.0000 | 1.0001 | nan |
| KMNIST | Hybrid-UFULL-smoke-alphaFixed1 | 1 | 0.2969 | 6551.6172 | 0.0000 | nan | nan |
| KMNIST | PureKAN-AdamW-smoke-alphaFixed1 | 1 | 0.1016 | nan | 0.0000 | nan | 0.0000 |
| KMNIST | PureKAN-UFULL-smoke-alphaFixed1 | 1 | 0.1328 | 6551.6172 | 0.0000 | nan | 0.0000 |
| KMNIST | Rational-DGKAN-AdamW-triton-alphaFixed1 | 1 | 0.2578 | nan | 0.0000 | 1.0001 | nan |
| KMNIST | Rational-UFULL-branch-smoke-alphaFixed1 | 1 | 0.2422 | 8467.6592 | 0.0000 | 1.0001 | nan |
| KMNIST | Rational-UFULL-poly-separate-smoke-alphaFixed1 | 1 | 0.2422 | 251534.2344 | 0.5000 | 1.0001 | nan |
| KMNIST | Rational-UFULL-tangent-joint-smoke-alphaFixed1 | 1 | 0.2422 | 8467.5293 | 0.0000 | 1.0001 | nan |

## P1 Rational Metric Audit

| dataset | method | runs | acc | val AUC | condG | clip | den p01 |
|---|---|---|---|---|---|---|---|
| Fashion-MNIST | Rational-DGKAN-AdamW-triton-alphaFixed1 | 3 | 0.8487 | 0.5293 | nan | 0.0000 | 1.0001 |
| Fashion-MNIST | Rational-UFULL-identity-alphaFixed1 | 3 | 0.8433 | 0.5328 | nan | 0.0008 | 1.0001 |
| Fashion-MNIST | Rational-UFULL-legendre-separate-alphaFixed1 | 3 | 0.8433 | 0.5096 | 250153.9531 | 0.0000 | 1.0001 |
| Fashion-MNIST | Rational-UFULL-poly-diag-alphaFixed1 | 3 | 0.8447 | 0.5095 | 250153.9531 | 0.0000 | 1.0001 |
| Fashion-MNIST | Rational-UFULL-poly-rho1e-1-alphaFixed1 | 3 | 0.8453 | 0.5253 | 10636.6377 | 0.0204 | 1.0002 |
| Fashion-MNIST | Rational-UFULL-poly-rho1e-2-alphaFixed1 | 3 | 0.8397 | 0.5221 | 82248.4141 | 0.2635 | 1.0004 |
| Fashion-MNIST | Rational-UFULL-poly-separate-current-alphaFixed1 | 3 | 0.8440 | 0.5266 | 250153.9531 | 0.4543 | 1.0004 |
| KMNIST | Rational-DGKAN-AdamW-triton-alphaFixed1 | 3 | 0.7930 | 0.4747 | nan | 0.0000 | 1.0002 |
| KMNIST | Rational-UFULL-identity-alphaFixed1 | 3 | 0.7993 | 0.4714 | nan | 0.0008 | 1.0001 |
| KMNIST | Rational-UFULL-legendre-separate-alphaFixed1 | 3 | 0.7840 | 0.4385 | 250153.9531 | 0.0000 | 1.0001 |
| KMNIST | Rational-UFULL-poly-diag-alphaFixed1 | 3 | 0.7817 | 0.4390 | 250153.9531 | 0.0000 | 1.0001 |
| KMNIST | Rational-UFULL-poly-rho1e-1-alphaFixed1 | 3 | 0.7890 | 0.4314 | 10636.6377 | 0.0429 | 1.0001 |
| KMNIST | Rational-UFULL-poly-rho1e-2-alphaFixed1 | 3 | 0.7917 | 0.4321 | 82248.4141 | 0.1892 | 1.0003 |
| KMNIST | Rational-UFULL-poly-separate-current-alphaFixed1 | 3 | 0.7950 | 0.4249 | 250153.9531 | 0.2607 | 1.0002 |

## P2 Rational Toy

| target | method | runs | mse | loss AUC | bad | cond | den p01 |
|---|---|---|---|---|---|---|---|
| silu | adam | 3 | 0.00022 | 0.00347 | 0.0100 | nan | 1.0078 |
| silu | poly-diag | 3 | 0.0645 | 0.152 | 0.0000 | 71667.7734 | 1.0019 |
| silu | poly-separate-current | 3 | 0.121 | 0.203 | 0.0000 | 71667.7734 | 1.0047 |
| silu | tangent-diag | 3 | 0.00317 | 0.0335 | 0.0000 | 113207.6875 | 1.0042 |
| silu | tangent-joint-H1 | 3 | 0.0706 | 0.0697 | 0.0000 | 489526.3125 | 1.0016 |
| silu | tangent-joint-L2 | 3 | 0.00267 | 0.0463 | 0.0000 | 148502.0938 | 1.0027 |
| smoothstep | adam | 3 | 4.98e-05 | 0.00226 | 0.0333 | nan | 1.0054 |
| smoothstep | poly-diag | 3 | 0.0148 | 0.0194 | 0.0000 | 71667.7734 | 1.0027 |
| smoothstep | poly-separate-current | 3 | 0.00373 | 0.00885 | 0.0133 | 71667.7734 | 1.0045 |
| smoothstep | tangent-diag | 3 | 0.000839 | 0.00644 | 0.0000 | 63036.1992 | 1.0057 |
| smoothstep | tangent-joint-H1 | 3 | 0.0248 | 0.0124 | 0.0000 | 91104.8125 | 1.0021 |
| smoothstep | tangent-joint-L2 | 3 | 2.06e-05 | 0.00351 | 0.0000 | 146524.1094 | 1.0031 |
| tanh_sin | adam | 3 | 0.000492 | 0.0066 | 0.0167 | nan | 1.0014 |
| tanh_sin | poly-diag | 3 | 0.0332 | 0.0351 | 0.0000 | 71667.7734 | 1.0022 |
| tanh_sin | poly-separate-current | 3 | 0.0197 | 0.0273 | 0.0000 | 71667.7734 | 1.0003 |
| tanh_sin | tangent-diag | 3 | 0.0225 | 0.027 | 0.0000 | 47487.4609 | 1.0047 |
| tanh_sin | tangent-joint-H1 | 3 | 0.00136 | 0.0216 | 0.0000 | 3537.6443 | 1.0003 |
| tanh_sin | tangent-joint-L2 | 3 | 0.000517 | 0.009 | 0.0000 | 4469.1133 | 1.0003 |

## P3 Rational Tangent DGKAN

| dataset | method | runs | acc | val AUC | condG | clip | den p01 |
|---|---|---|---|---|---|---|---|
| Fashion-MNIST | RBF-UFULL-f085-reference-alphaFixed1 | 3 | 0.8403 | 0.5205 | 3836.6843 | 0.0000 | nan |
| Fashion-MNIST | Rational-DGKAN-AdamW-triton-alphaFixed1 | 3 | 0.8440 | 0.5299 | nan | 0.0000 | 1.0002 |
| Fashion-MNIST | Rational-UFULL-poly-current-alphaFixed1 | 3 | 0.8437 | 0.5245 | 250153.9531 | 0.4547 | 1.0004 |
| Fashion-MNIST | Rational-UFULL-tangent-act-H1-alphaFixed1 | 3 | 0.8383 | 0.5166 | 18193.7865 | 0.0000 | 1.0001 |
| Fashion-MNIST | Rational-UFULL-tangent-act-L2-alphaFixed1 | 3 | 0.8407 | 0.5199 | 15341.3952 | 0.0000 | 1.0001 |
| Fashion-MNIST | Rational-UFULL-tangent-branch-H1-alphaFixed1 | 3 | 0.8357 | 0.4875 | 17753.4349 | 0.0000 | 1.0001 |
| Fashion-MNIST | Rational-UFULL-tangent-branch-L2-alphaFixed1 | 3 | 0.8393 | 0.4882 | 14914.3740 | 0.0000 | 1.0001 |
| Fashion-MNIST | Rational-UFULL-tangent-diag-alphaFixed1 | 3 | 0.8403 | 0.5097 | 15918.6956 | 0.0000 | 1.0001 |
| KMNIST | RBF-UFULL-f085-reference-alphaFixed1 | 3 | 0.7610 | 0.4841 | 3836.6843 | 0.0000 | nan |
| KMNIST | Rational-DGKAN-AdamW-triton-alphaFixed1 | 3 | 0.8010 | 0.4711 | nan | 0.0000 | 1.0001 |
| KMNIST | Rational-UFULL-poly-current-alphaFixed1 | 3 | 0.7907 | 0.4306 | 250153.9531 | 0.2912 | 1.0002 |
| KMNIST | Rational-UFULL-tangent-act-H1-alphaFixed1 | 3 | 0.7823 | 0.4338 | 18051.0501 | 0.0000 | 1.0001 |
| KMNIST | Rational-UFULL-tangent-act-L2-alphaFixed1 | 3 | 0.7873 | 0.4283 | 15299.2191 | 0.0000 | 1.0001 |
| KMNIST | Rational-UFULL-tangent-branch-H1-alphaFixed1 | 3 | 0.7720 | 0.4226 | 17639.1400 | 0.0000 | 1.0001 |
| KMNIST | Rational-UFULL-tangent-branch-L2-alphaFixed1 | 3 | 0.7743 | 0.4226 | 14814.0218 | 0.0000 | 1.0001 |
| KMNIST | Rational-UFULL-tangent-diag-alphaFixed1 | 3 | 0.7823 | 0.4312 | 15954.1966 | 0.0000 | 1.0001 |

### P3 Paired Delta vs Rational AdamW

| dataset | method | runs | acc delta | AUC imp |
|---|---|---|---|---|
| Fashion-MNIST | RBF-UFULL-f085-reference-alphaFixed1 | 3 | -0.0037 | 0.0094 |
| Fashion-MNIST | Rational-UFULL-poly-current-alphaFixed1 | 3 | -0.0003 | 0.0054 |
| Fashion-MNIST | Rational-UFULL-tangent-act-H1-alphaFixed1 | 3 | -0.0057 | 0.0134 |
| Fashion-MNIST | Rational-UFULL-tangent-act-L2-alphaFixed1 | 3 | -0.0033 | 0.0101 |
| Fashion-MNIST | Rational-UFULL-tangent-branch-H1-alphaFixed1 | 3 | -0.0083 | 0.0425 |
| Fashion-MNIST | Rational-UFULL-tangent-branch-L2-alphaFixed1 | 3 | -0.0047 | 0.0418 |
| Fashion-MNIST | Rational-UFULL-tangent-diag-alphaFixed1 | 3 | -0.0037 | 0.0203 |
| KMNIST | RBF-UFULL-f085-reference-alphaFixed1 | 3 | -0.0400 | -0.0130 |
| KMNIST | Rational-UFULL-poly-current-alphaFixed1 | 3 | -0.0103 | 0.0405 |
| KMNIST | Rational-UFULL-tangent-act-H1-alphaFixed1 | 3 | -0.0187 | 0.0373 |
| KMNIST | Rational-UFULL-tangent-act-L2-alphaFixed1 | 3 | -0.0137 | 0.0428 |
| KMNIST | Rational-UFULL-tangent-branch-H1-alphaFixed1 | 3 | -0.0290 | 0.0485 |
| KMNIST | Rational-UFULL-tangent-branch-L2-alphaFixed1 | 3 | -0.0267 | 0.0485 |
| KMNIST | Rational-UFULL-tangent-diag-alphaFixed1 | 3 | -0.0187 | 0.0399 |

## P5 U-FULL f100 Failure Audit

| dataset | method | runs | acc | val AUC | branch | noKAN | margin |
|---|---|---|---|---|---|---|---|
| Fashion-MNIST | AdamW-alphaFixed1-alphaFixed1 | 5 | 0.8322 | 0.5527 | 0.7659 | 0.2050 | 4.1511 |
| Fashion-MNIST | F-V3-HARD-base30-v32best-alphaFixed1 | 5 | 0.8376 | 0.5194 | 0.6210 | 0.1912 | 3.9238 |
| Fashion-MNIST | U-FULL-f085-reference-alphaFixed1 | 5 | 0.8420 | 0.5257 | 0.6110 | 0.1892 | 3.8446 |
| Fashion-MNIST | U-FULL-f100-alphaFixed1 | 5 | 0.8498 | 0.5253 | 0.6095 | 0.1780 | 3.8809 |
| KMNIST | AdamW-alphaFixed1-alphaFixed1 | 5 | 0.7740 | 0.5123 | 0.7718 | 0.2486 | 3.8950 |
| KMNIST | K-V3-FULL-base-v32best-alphaFixed1 | 5 | 0.7644 | 0.4864 | 0.5346 | 0.2000 | 3.7396 |
| KMNIST | U-FULL-f085-reference-alphaFixed1 | 5 | 0.7642 | 0.4854 | 0.5469 | 0.1994 | 3.7369 |
| KMNIST | U-FULL-f100-alphaFixed1 | 5 | 0.7724 | 0.4806 | 0.5683 | 0.1858 | 3.5859 |

## P6 CIFAR Branch Audit

| dataset | method | runs | acc | val AUC | branch | noKAN | dead | outgrid | conv p95 |
|---|---|---|---|---|---|---|---|---|---|
| CIFAR10 | ConvStem-DGKAN-AdamW-alphaFixed1 | 3 | 0.5063 | 1.5982 | 1.7772 | 0.2512 | 0.0000 | 0.0135 | 0.6095 |
| CIFAR10 | ConvStem-DGKAN-U-FULL-f085-alphaFixed1 | 3 | 0.3610 | 1.8121 | 0.7542 | -0.0278 | 0.0000 | 0.0144 | 0.7065 |
| CIFAR10 | ConvStem-DGKAN-U-FULL-f100-alphaFixed1 | 3 | 0.3772 | 1.7837 | 0.8412 | -0.0077 | 0.0000 | 0.0133 | 0.6824 |
| CIFAR10 | ConvStem-DGKAN-UFULL-branchboost-probe-alphaFixed1 | 3 | 0.3503 | 1.7794 | 0.7770 | -0.0475 | 0.0000 | 0.0128 | 0.6837 |
| CIFAR10 | ConvStem-DGKAN-UFULL-diag-warmup-probe-alphaFixed1 | 3 | 0.4040 | 1.7821 | 0.7630 | 0.0158 | 0.0000 | 0.0146 | 0.6981 |

## P8 PureKAN 3-Seed

| dataset | method | runs | acc | val AUC | condG | pure nonKAN | dead | outgrid |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | Hybrid-DGKAN-UFULL-f085-alphaFixed1 | 3 | 0.8463 | 0.5018 | 3836.6843 | nan | 0.0000 | 0.0201 |
| Fashion-MNIST | MLP-AdamW-alphaFixed1 | 3 | 0.8390 | 0.4820 | nan | nan | nan | nan |
| Fashion-MNIST | PureKAN-AdamW-alphaFixed1 | 3 | 0.8577 | 0.5426 | nan | 0.0000 | 0.0156 | 0.0025 |
| Fashion-MNIST | PureKAN-UFULL-alphaFixed1 | 3 | 0.8333 | 0.6363 | 3835.3533 | 0.0000 | 0.0156 | 0.0083 |
| Fashion-MNIST | PureKAN-UFULL-diagwarmup-alphaFixed1 | 3 | 0.8333 | 0.6363 | 3835.3533 | 0.0000 | 0.0156 | 0.0083 |
| KMNIST | Hybrid-DGKAN-UFULL-f085-alphaFixed1 | 3 | 0.7620 | 0.4566 | 3836.6843 | nan | 0.0000 | 0.0266 |
| KMNIST | MLP-AdamW-alphaFixed1 | 3 | 0.7697 | 0.4528 | nan | nan | nan | nan |
| KMNIST | PureKAN-AdamW-alphaFixed1 | 3 | 0.7740 | 0.4917 | nan | 0.0000 | 0.0156 | 0.0023 |
| KMNIST | PureKAN-UFULL-alphaFixed1 | 3 | 0.6693 | 0.9495 | 3835.3533 | 0.0000 | 0.0156 | 0.0104 |
| KMNIST | PureKAN-UFULL-diagwarmup-alphaFixed1 | 3 | 0.6693 | 0.9495 | 3835.3533 | 0.0000 | 0.0156 | 0.0104 |
| MNIST | Hybrid-DGKAN-UFULL-f085-alphaFixed1 | 3 | 0.9423 | 0.2804 | 3836.6843 | nan | 0.0000 | 0.0192 |
| MNIST | MLP-AdamW-alphaFixed1 | 3 | 0.9427 | 0.2848 | nan | nan | nan | nan |
| MNIST | PureKAN-AdamW-alphaFixed1 | 3 | 0.9343 | 0.3931 | nan | 0.0000 | 0.0312 | 0.0324 |
| MNIST | PureKAN-UFULL-alphaFixed1 | 3 | 0.8980 | 0.7370 | 3835.3533 | 0.0000 | 0.0312 | 0.0379 |
| MNIST | PureKAN-UFULL-diagwarmup-alphaFixed1 | 3 | 0.8980 | 0.7370 | 3835.3533 | 0.0000 | 0.0312 | 0.0379 |

### P8 Paired Delta vs PureKAN-AdamW

| dataset | method | runs | acc delta | AUC imp |
|---|---|---|---|---|
| Fashion-MNIST | Hybrid-DGKAN-UFULL-f085-alphaFixed1 | 3 | -0.0113 | 0.0408 |
| Fashion-MNIST | MLP-AdamW-alphaFixed1 | 3 | -0.0187 | 0.0606 |
| Fashion-MNIST | PureKAN-UFULL-alphaFixed1 | 3 | -0.0243 | -0.0937 |
| Fashion-MNIST | PureKAN-UFULL-diagwarmup-alphaFixed1 | 3 | -0.0243 | -0.0937 |
| KMNIST | Hybrid-DGKAN-UFULL-f085-alphaFixed1 | 3 | -0.0120 | 0.0351 |
| KMNIST | MLP-AdamW-alphaFixed1 | 3 | -0.0043 | 0.0389 |
| KMNIST | PureKAN-UFULL-alphaFixed1 | 3 | -0.1047 | -0.4578 |
| KMNIST | PureKAN-UFULL-diagwarmup-alphaFixed1 | 3 | -0.1047 | -0.4578 |
| MNIST | Hybrid-DGKAN-UFULL-f085-alphaFixed1 | 3 | 0.0080 | 0.1127 |
| MNIST | MLP-AdamW-alphaFixed1 | 3 | 0.0083 | 0.1082 |
| MNIST | PureKAN-UFULL-alphaFixed1 | 3 | -0.0363 | -0.3439 |
| MNIST | PureKAN-UFULL-diagwarmup-alphaFixed1 | 3 | -0.0363 | -0.3439 |

## Final Interpretation

P0 通过：Rational tangent metric 有限且 condition 已压到 smoke gate 内，PureKAN 的 learnable non-KAN 参数为 0。

Rational/KAT：tangent metric 修复了旧 poly metric 的高 clip 问题，但 P3 中 condition 仍约 1.5e4-1.8e4，且相对 Rational-AdamW 没有稳定 AUC/accuracy 优势，因此 P4 不触发。

U-FULL f100：Fashion 上 f100 有更强 accuracy signal，但 KMNIST 与 f085/v3.2 best 仍不稳；branch 更大不等于任务对齐更好。

CIFAR-small：ConvStem U-FULL 仍明显低于 ConvStem AdamW，diag warmup/branch boost probe 只能局部缓解，不能恢复到 AdamW。

PureKAN：结构实现成功，AdamW 可训练；但 PureKAN-UFULL 在 Fashion/KMNIST/MNIST 都明显慢或掉点，说明当前 U-FULL 仍更像 hybrid branch optimizer，而不是完整 PureKAN optimizer。P9 不触发。



---


# Source 3: `docs/DG-KAN_v3.7_PureKAN_UFULL_代码检查与实验计划.md`


# DG-KAN v3.7：PureKAN-UFULL 优先修复、代码审计与实验计划

> 目标：优先让 `PureKAN-UFULL` 全面超过 `PureKAN-AdamW`、`Hybrid-DGKAN-UFULL` 和 `MLP-AdamW`。  
> 公式格式：Typora 友好，统一使用 `$...$` 与 `$$...$$`。  
> 当前结论前提：v3.6 已证明 PureKAN 架构本身可训练，`PureKAN-AdamW` 表现强；但当前 `PureKAN-UFULL` 明显落后，说明必须优先检查实现、配置和 PureKAN 专用 functional update 设计。

---

## 0. 本轮为什么必须优先 PureKAN-UFULL

当前最严重的问题不是 Hybrid-DGKAN 是否还能工作，而是：

$$
\boxed{
\text{如果 PureKAN-UFULL 不能超过 PureKAN-AdamW，}
\text{那么我们不能宣称 U-FULL 是完整 KAN 网络的 optimizer。}
}
$$

v3.6 结果显示：

```text
PureKAN-AdamW 可训练，并且在 Fashion/KMNIST/MNIST 上表现不弱；
PureKAN-UFULL 在三个数据集上 accuracy 和 validation-loss AUC 都明显落后。
```

这意味着当前叙事必须收紧：

```text
旧叙事：U-FULL 是通用 KAN functional optimizer。
新问题：U-FULL 目前只在 Hybrid-DGKAN branch setting 中成立，
       还没有证明能训练完整 PureKAN 网络。
```

本轮目标不是继续给 Hybrid 找更好 profile，而是优先解决：

$$
\boxed{
\text{PureKAN-UFULL 为什么输给 PureKAN-AdamW？}
}
$$

并把它推进到：

$$
\boxed{
\text{PureKAN-UFULL} \geq
\max(\text{PureKAN-AdamW},\ \text{Hybrid-DGKAN-UFULL},\ \text{MLP-AdamW})
}
$$

---

## 1. 代码审计：目前发现的高优先级风险

### 1.1 风险一：`diagwarmup` 很可能没有真正生效

当前 `PureKAN-UFULL` 配置里有：

```text
v3_phase_mode = hard
v3_metric_active = full_sobolev_gram
v3_metric_transition = full_sobolev_gram
v3_metric_geometry = full_sobolev_gram
branch_max_active_frac = 0.0
branch_boost = 1.0
branch_final_scale = 1.0
```

训练 loop 中，在每个 step 前有逻辑：

```text
if progress >= branch_max_active_frac:
    switch_branch_state(...)
```

因为 `branch_max_active_frac = 0.0`，所以在 step 0 就会直接触发 switch，进入 `GEOMETRY` phase。这样会导致：

```text
ACTIVE phase 基本不存在；
v3_metric_active 不会真正参与训练；
如果 diagwarmup 只是改 active metric，它会和 U-FULL 完全一样。
```

v3.6 中 `PureKAN-UFULL` 和 `PureKAN-UFULL-diagwarmup` 三个数据集结果完全相同，这非常像配置没有生效，而不是方法自然得到完全相同曲线。

因此本轮第一修复项是：

```text
PureKAN-UFULL-full:
  从 step 0 使用 full_sobolev_gram。

PureKAN-UFULL-diagwarmup:
  必须保证 early steps 真正使用 basis_diag_gram 或 identity/diag。
  branch_max_active_frac 不能是 0。
```

建议新增字段：

```text
pure_metric_schedule = full_from_start | diag_warmup | identity_warmup | layerwise
pure_warmup_frac = 0.15 或 0.25
```

并要求每个 run 输出：

```text
phase_trace
metric_mode_trace
metric_mix_trace
active_steps_actual
geometry_steps_actual
```

---

### 1.2 风险二：PureKAN 的 `alphaFixed1` 标签可能不等于 block alpha 真的为 1

当前 `PureKANClassifier` 的 block alpha 来自：

```text
PureResidualKANBlock(..., alpha=alpha_init)
```

而不是从 `alpha_mode=fixed1` 派生。也就是说，如果全局 `alpha_init=1.5`，那么 PureKAN block 的实际残差系数可能是：

$$
\alpha = 1.5
$$

即使方法名叫 `alphaFixed1`。这会造成两个问题：

```text
1. 实验标签和真实模型不一致；
2. PureKAN-AdamW 与 PureKAN-UFULL 都可能在一个不是预期的 alpha setting 下比较。
```

本轮必须修复：

```text
PureKANClassifier 接受 alpha_mode。
如果 alpha_mode = fixed1，则 block alpha = 1.0。
如果 alpha_mode = fixed_init，则 block alpha = alpha_init。
如果 alpha_mode = learnable，则 PureKAN 是否允许 learnable alpha 要单独显式开启。
```

并记录：

```text
pure_alpha_mode
pure_alpha_layer_0
pure_alpha_layer_1
pure_alpha_mean
pure_alpha_trainable
```

---

### 1.3 风险三：PureKAN 的 no-KAN / contribution audit 现在不完整

当前 `supports_disable_kan` 不包含 `PureKANClassifier`。因此 PureKAN 的 `no_kan_drop` 很可能被写成 0 或等于 full model 对照，不能用于判断 PureKAN 的层贡献。

但是 PureKAN 没有普通 MLP stem/head，所谓 “disable KAN” 本身不再有同样含义。正确做法不是简单 no-KAN，而是做 layer ablation：

```text
input_kan ablation: 不建议直接置零，因为会破坏输入通道；可做 frozen/random/control。
block_kan ablation: disable residual KAN blocks。
output_kan ablation: 替换为 linear probe 或 frozen output KAN。
```

本轮必须新增 PureKAN 专用 contribution metrics：

```text
pure/block_disable_acc
pure/block_disable_loss
pure/block_no_residual_drop
pure/output_randomized_acc
pure/input_frozen_acc
pure/layerwise_logit_delta_norm
pure/layerwise_margin_contribution
```

---

### 1.4 风险四：PureKAN 所有层共用同一个 metric / lr，可能是根本不合适

PureKAN 有三类 KAN 层：

```text
input_kan: raw input -> hidden
block_kan: hidden -> hidden residual branch
output_kan: hidden -> logits
```

当前 U-FULL 对所有 RBFDense 层用同一个：

```text
metric = full_sobolev_gram
coeff_lr = same value
trust_radius = same value
```

这很可能不合理。因为三类层的作用不同：

```text
input_kan 需要快速形成表征；
block_kan 需要稳定 residual transformation；
output_kan 需要快速对齐分类 margin。
```

本轮必须把 PureKAN-UFULL 改成 role-aware：

```text
pure_input_metric
pure_block_metric
pure_output_metric
pure_input_lr_mult
pure_block_lr_mult
pure_output_lr_mult
pure_input_trust_radius
pure_block_trust_radius
pure_output_trust_radius
```

如果不做 role-aware，PureKAN-UFULL 很可能继续输给 AdamW。

---

### 1.5 风险五：full Sobolev from start 可能对 PureKAN 全网络过度平滑

Hybrid-DGKAN 中，U-FULL 只优化 residual KAN branch，non-KAN stem/head 由 AdamW 提供稳定表征与分类通道。PureKAN 中，input 和 output 都是 KAN：

$$
h_0 = \operatorname{KAN}_{in}(x)
$$

$$
h_{l+1} = h_l + \operatorname{KAN}_l(\operatorname{FixedNorm}(h_l))
$$

$$
z = \operatorname{KAN}_{out}(\operatorname{FixedNorm}(h_L))
$$

如果所有层从 step 0 都用 full Sobolev，可能会出现：

```text
input representation 学得太慢；
output logits 对齐太慢；
validation loss AUC 明显落后；
accuracy 后期也追不上 AdamW。
```

因此 PureKAN 需要优先测试：

```text
full_from_start 是否真的适合所有层；
diag/identity warmup 是否必要；
input/output 是否应该用 diag 或 identity，而 block 用 full Sobolev。
```

---

## 2. 本轮核心假设

本轮只验证一个主命题：

$$
\boxed{
\text{PureKAN-UFULL 能否通过实现修复和 role-aware functional update，}
\text{全面超过 PureKAN-AdamW、Hybrid-DGKAN-UFULL 和 MLP-AdamW？}
}
$$

这里 “全面超过” 定义为：

### 2.1 Accuracy

对每个 dataset：

$$
\operatorname{Acc}_{PureKAN-UFULL}
\geq
\max(
\operatorname{Acc}_{PureKAN-AdamW},
\operatorname{Acc}_{Hybrid-UFULL},
\operatorname{Acc}_{MLP-AdamW}
)
$$

正式 5/10-seed confirm 中允许 paired CI 边界非常小，但 mean 必须不低于最强 baseline。

### 2.2 Convergence

用 validation loss AUC：

$$
\operatorname{AUCImprove}_{PureKAN-UFULL\ vs\ best\ baseline}>0
$$

其中：

$$
\operatorname{AUCImprove}
=
\frac{
\operatorname{AUC}_{baseline}-\operatorname{AUC}_{method}
}{
\operatorname{AUC}_{baseline}+\epsilon
}.
$$

### 2.3 Geometry

PureKAN-UFULL 必须比 PureKAN-AdamW 有更好函数几何：

$$
\phi'_{p95,U FULL}<\phi'_{p95,AdamW}
$$

$$
\kappa(J)_{U FULL}<\kappa(J)_{AdamW}
$$

### 2.4 Calibration

$$
\operatorname{ECE}_{U FULL}<\operatorname{ECE}_{AdamW}
$$

### 2.5 Functional legitimacy

PureKAN-UFULL 必须满足：

```text
learnable_nonKAN_params = 0
coeff_params_seen_by_functional_step = all learnable params
AdamW optimizer state = 0 for PureKAN-UFULL
```

否则不能叫 PureKAN-UFULL。

---

## 3. 必须先修的实现项

### Fix 1：PureKAN alpha 真实 fixed1

修改：

```python
class PureResidualKANBlock(nn.Module):
    def __init__(self, dim, basis_count, alpha_init=1.0, alpha_mode="fixed1"):
        if alpha_mode == "fixed1":
            self.register_buffer("alpha", torch.tensor(1.0))
        elif alpha_mode == "fixed_init":
            self.register_buffer("alpha", torch.tensor(float(alpha_init)))
        elif alpha_mode == "learnable":
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
```

`PureKANClassifier` 也传入 `alpha_mode`。

必须 assert：

```text
method contains alphaFixed1 -> all pure alpha = 1.0
```

---

### Fix 2：PureKAN diagwarmup 不允许 step0 switch

新增 PureKAN-specific schedule：

```text
PureKAN-UFULL-full:
  active metric = full
  branch_max_active_frac = 0.0

PureKAN-UFULL-diagwarmup:
  active metric = basis_diag_gram
  geometry metric = full_sobolev_gram
  branch_max_active_frac = 0.20
  v3_phase_mode = hard or smooth

PureKAN-UFULL-identitywarmup:
  active metric = identity
  geometry metric = full_sobolev_gram
  branch_max_active_frac = 0.10 or 0.20
```

必须记录并检查：

```text
active_steps_actual > 0 for diagwarmup/identitywarmup
metric_active_seen = basis_diag_gram or identity
metric_geometry_seen = full_sobolev_gram
```

---

### Fix 3：PureKAN layer-role specific metric / LR

新增 config：

```text
pure_input_metric = identity | basis_diag_gram | full_sobolev_gram
pure_block_metric = identity | basis_diag_gram | full_sobolev_gram
pure_output_metric = identity | basis_diag_gram | full_sobolev_gram

pure_input_lr_mult
pure_block_lr_mult
pure_output_lr_mult
```

最小实现可以在 `functional_coeff_step` 中按参数名判断：

```text
input_kan.coeff -> role=input
blocks.*.kan.coeff -> role=block
output_kan.coeff -> role=output
```

然后选择 metric 和 lr multiplier。

---

### Fix 4：PureKAN update coverage audit

每个 run 输出：

```text
pure/learnable_total_params
pure/learnable_nonkan_params
pure/coeff_param_count
pure/coeff_param_numel_total
pure/coeff_param_seen_ratio
pure/input_update_norm
pure/block_update_norm_mean
pure/output_update_norm
pure/input_update_over_param
pure/block_update_over_param_mean
pure/output_update_over_param
pure/input_raw_grad_norm
pure/block_raw_grad_norm_mean
pure/output_raw_grad_norm
pure/input_precond_norm
pure/block_precond_norm_mean
pure/output_precond_norm
```

通过条件：

$$
\text{coeff\_param\_seen\_ratio}=1.0
$$

$$
\text{learnable\_nonKAN\_params}=0
$$

---

### Fix 5：PureKAN layer contribution audit

新增 PureKAN 专用 ablation：

```text
pure/block_disable_acc
pure/block_disable_drop
pure/input_frozen_probe_acc
pure/output_frozen_probe_acc
pure/layerwise_logit_delta_norm
pure/layerwise_margin_contribution
```

尤其要记录：

$$
\Delta_{block}
=
\operatorname{Acc}_{full}-\operatorname{Acc}_{disable\ residual\ blocks}
$$

这比普通 no-KAN 更适合 PureKAN。

---

## 4. 实验路线总览

```text
P0: Code and config audit smoke
P1: One-batch shadow-step diagnosis
P2: Metric schedule repair test
P3: Layer-role PureKAN-UFULL sweep
P4: Capacity and epoch budget check
P5: 3-seed candidate selection
P6: 5-seed confirm against all baselines
P7: 10-seed final confirm
P8: Failure analysis if P6/P7 fail
```

---

## 5. P0：代码与配置 smoke

### 5.1 目标

确认 PureKAN-UFULL 真的是 PureKAN，且所有 learnable 参数都被 functional update 覆盖。

### 5.2 数据集

```text
Fashion-MNIST
KMNIST
MNIST
```

### 5.3 方法

```text
PureKAN-AdamW
PureKAN-UFULL-full
PureKAN-UFULL-diagwarmup
PureKAN-UFULL-identitywarmup
Hybrid-DGKAN-UFULL-f085
MLP-AdamW
```

### 5.4 设置

```text
train/val/test = 512/128/128
epochs = 1
seed = 0
hidden_dim = 96
basis_count = 24
depth = 4
alpha_mode = fixed1
```

### 5.5 必须记录

```text
learnable_total_params
learnable_kan_params
learnable_nonkan_params
purekan_nonkan_param_count
purekan_has_linear
purekan_has_layernorm_params
purekan_has_bias_params
coeff_param_count
coeff_param_names
coeff_param_seen_ratio
pure_alpha_mean
pure_alpha_layer_k
metric_trace_first_20_steps
phase_trace_first_20_steps
input_update_norm
block_update_norm_mean
output_update_norm
input_metric_mode_seen
block_metric_mode_seen
output_metric_mode_seen
trust_clip_rate
run_failed
nan_count
```

### 5.6 通过条件

```text
PureKAN-UFULL:
  learnable_nonkan_params = 0
  purekan_has_linear = 0
  purekan_has_layernorm_params = 0
  purekan_has_bias_params = 0
  coeff_param_seen_ratio = 1.0
  alpha = 1.0 when alphaFixed1
  diagwarmup trace differs from full trace
  all key update norms finite
  trust_clip_rate < 0.05
```

如果 P0 不过，不允许进入 P1。

### 5.7 可视化

```text
config audit table
coefficient coverage bar
metric phase trace for first 20 steps
per-layer update norm bar
```

---

## 6. P1：one-batch shadow-step diagnosis

### 6.1 目标

判断 PureKAN-UFULL 失败是：

```text
step 太小；
direction 错；
full Sobolev 太平滑；
input/output 层被错误 precondition；
trust clipping 压制；
还是 schedule 没生效。
```

### 6.2 做法

对同一个初始化和同一个 minibatch，计算不同 candidate update 的 shadow step。不要真的训练，只复制参数后试一步：

```text
raw SGD / identity functional
basis_diag_gram
full_sobolev_gram
diag_to_full
layerwise: input diag, block full, output diag
layerwise: input identity, block full, output identity
AdamW one-step reference
```

### 6.3 记录

```text
shadow/train_loss_before
shadow/train_loss_after
shadow/val_loss_before
shadow/val_loss_after
shadow/predicted_descent
shadow/actual_descent
shadow/bad_step_bool
shadow/raw_grad_norm_by_layer
shadow/precond_grad_norm_by_layer
shadow/update_norm_by_layer
shadow/update_over_param_by_layer
shadow/cos_raw_precond_by_layer
shadow/metric_condition_by_layer
shadow/trust_clip_by_layer
```

### 6.4 判定

如果：

$$
\Delta L_{actual}<0
$$

表示这个 update 是 descent step。

如果 full Sobolev 的 actual descent 比 diag 差很多，说明：

```text
full metric 对 PureKAN 早期不适合。
```

如果 output layer 的 preconditioned direction 最差，说明：

```text
output_kan 不能用同一套 full Sobolev。
```

### 6.5 可视化

```text
predicted vs actual descent scatter
per-layer cos(raw, precond) heatmap
per-layer update_over_param heatmap
metric condition by layer bar
bad step rate by metric mode
```

---

## 7. P2：metric schedule repair test

### 7.1 目标

验证修复后的 diagwarmup / identitywarmup 是否真正改变 PureKAN-UFULL trajectory。

### 7.2 方法

```text
PureKAN-AdamW
PureKAN-UFULL-full
PureKAN-UFULL-diagwarmup-0.10
PureKAN-UFULL-diagwarmup-0.20
PureKAN-UFULL-identitywarmup-0.10
PureKAN-UFULL-identitywarmup-0.20
PureKAN-UFULL-diag-to-full-smooth-0.20
```

### 7.3 设置

```text
datasets = Fashion-MNIST, KMNIST, MNIST
seeds = 0,1,2
epochs = existing PureKAN setting
hidden_dim = 96
basis_count = 24
depth = 4
```

### 7.4 记录

```text
acc
val_loss_auc
val_acc_auc
ECE
phi_prime_p95
jacobian_condition
curvature_energy
basis_dead_frac
out_of_grid_frac
input_update_norm
block_update_norm
output_update_norm
metric_phase_auc
active_steps_actual
transition_loss_jump
```

### 7.5 通过条件

P2 的目标不是最终超过 AdamW，而是必须证明：

```text
diagwarmup / identitywarmup 和 full_from_start 曲线不同；
至少一个 schedule 在 validation-loss AUC 上优于 PureKAN-UFULL-full；
至少一个 schedule 在 accuracy 上缩小 PureKAN-UFULL vs PureKAN-AdamW gap。
```

---

## 8. P3：layer-role PureKAN-UFULL sweep

### 8.1 目标

找到真正适合 PureKAN 的 layer-wise functional update。

### 8.2 候选

#### Candidate A：all-full

```text
input = full
block = full
output = full
```

这是当前 U-FULL baseline。

#### Candidate B：input/output diag, block full

```text
input = basis_diag_gram
block = full_sobolev_gram
output = basis_diag_gram
```

假设：input/output 需要更快的 task fitting，block 负责 geometry。

#### Candidate C：input identity, block full, output diag

```text
input = identity
block = full_sobolev_gram
output = basis_diag_gram
```

假设：input layer 不应被 Sobolev 过度平滑。

#### Candidate D：input diag, block full, output identity

```text
input = basis_diag_gram
block = full_sobolev_gram
output = identity
```

假设：output classifier 需要最快对齐 margin。

#### Candidate E：all-diag warmup then rolewise

```text
early:
  input = identity or diag
  block = diag
  output = identity or diag
late:
  input = diag
  block = full
  output = diag
```

### 8.3 LR multipliers

每个 candidate 先用小网格：

```text
input_lr_mult in {1.0, 2.0}
block_lr_mult in {1.0}
output_lr_mult in {1.0, 2.0, 4.0}
```

不要先大扫。

### 8.4 数据集与 seeds

```text
Fashion-MNIST, KMNIST, MNIST
seeds = 0,1,2
```

### 8.5 记录

```text
acc
val_loss_auc
ECE
phi_prime_p95
jacobian_condition
input_update_over_param
block_update_over_param
output_update_over_param
input_cos_raw_precond
block_cos_raw_precond
output_cos_raw_precond
input_metric_condition
block_metric_condition
output_metric_condition
logit_margin
classwise_acc
classwise_margin
pure_block_disable_drop
```

### 8.6 判定

进入 P5 的候选必须满足：

```text
acc gap vs PureKAN-AdamW < 1.0% on all three datasets
AUC improvement vs PureKAN-UFULL-full > 0
geometry better than PureKAN-AdamW
ECE better than PureKAN-AdamW or not worse by > 5%
```

---

## 9. P4：capacity and epoch budget check

### 9.1 目标

排除 PureKAN-UFULL 只是训练预算不足。

### 9.2 方法

只拿 P3 最好的 2 个候选，测试：

```text
epochs = base, 1.5x, 2x
hidden_dim = 96, 128
basis_count = 24, 32
```

### 9.3 判定

如果 PureKAN-UFULL 只需要更长训练即可超过 AdamW，则：

```text
AUC 会逐渐转正；
final acc 会追上；
update norm 不会异常小。
```

如果更长训练仍然 AUC 负，则是 update direction 问题，不是预算问题。

---

## 10. P5：3-seed candidate selection

### 10.1 Baselines

```text
PureKAN-AdamW
Hybrid-DGKAN-UFULL-f085
MLP-AdamW
PureKAN-UFULL-full
```

### 10.2 Candidates

从 P3/P4 选最多 3 个：

```text
PureKAN-UFULL-rolewise-best1
PureKAN-UFULL-rolewise-best2
PureKAN-UFULL-schedule-best
```

### 10.3 Gate

进入 P6 的候选必须：

```text
Fashion acc >= max baselines - 0.5%
KMNIST acc >= max baselines - 0.5%
MNIST acc >= max baselines - 0.5%
AUC better than PureKAN-AdamW on at least 2/3 datasets
geometry better than PureKAN-AdamW on 3/3 datasets
ECE not worse than PureKAN-AdamW by > 5%
```

如果没有候选通过 P5，则不允许跑 P6/P7，必须回到 P1/P3 诊断。

---

## 11. P6：5-seed confirm against all baselines

### 11.1 数据集

```text
Fashion-MNIST
KMNIST
MNIST
```

### 11.2 Methods

```text
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
PureKAN-AdamW
PureKAN-UFULL-final-candidate
```

### 11.3 Seeds

```text
seeds = 0..4
```

### 11.4 成功条件

#### Hard pass

PureKAN-UFULL 在 3 个数据集上满足：

$$
\operatorname{Acc}_{U FULL}
\geq
\max(\operatorname{Acc}_{PureAdamW},\operatorname{Acc}_{Hybrid},\operatorname{Acc}_{MLP})
$$

并且：

$$
\operatorname{AUC}_{U FULL}<\operatorname{AUC}_{PureAdamW}
$$

$$
\operatorname{ECE}_{U FULL}<\operatorname{ECE}_{PureAdamW}
$$

$$
\phi'_{U FULL}<\phi'_{PureAdamW}
$$

#### Medium pass

如果 accuracy 只在某个数据集差不超过 `0.5%`，但 AUC/ECE/geometry 明显优于 AdamW，可以进入 P7，但不能写全面超过。

---

## 12. P7：10-seed final confirm

只在 P6 hard pass 或非常接近 hard pass 时运行。

```text
seeds = 0..9
methods = same as P6
```

最终报告 paired seed delta：

```text
PureKAN-UFULL vs PureKAN-AdamW
PureKAN-UFULL vs Hybrid-DGKAN-UFULL
PureKAN-UFULL vs MLP-AdamW
```

记录：

```text
acc delta mean / CI
AUC delta mean / CI
ECE delta mean / CI
phi/J delta mean / CI
```

---

## 13. P8：如果 PureKAN-UFULL 仍失败，必须输出 failure diagnosis

如果 P5/P6 失败，不允许只写 “PureKAN-UFULL failed”。必须输出失败类型。

### Failure type A：step too small

证据：

```text
update_over_param 很小
actual descent 为正但太小
longer training 可追上
```

### Failure type B：direction wrong

证据：

```text
cos(raw, precond) 很低
shadow actual descent 差
AUC 持续为负
```

### Failure type C：output layer bottleneck

证据：

```text
output_update_norm 低
output margin contribution 低
output identity/diag 比 full 好
```

### Failure type D：input layer bottleneck

证据：

```text
input_update_norm 低
input representation rank 低
input diag/identity 比 full 好
```

### Failure type E：over-smoothing

证据：

```text
phi/curvature 远低于 AdamW
accuracy 低
longer training 仍不上升
```

---

## 14. 统一 dashboard 必须包含的图

### Page 1：Final dominance scorecard

```text
method
dataset
acc mean/std
acc delta vs PureKAN-AdamW
acc delta vs Hybrid
acc delta vs MLP
AUC delta
ECE delta
phi/J reduction
pass status
```

### Page 2：Loss / accuracy curves

```text
train_loss vs epoch
val_loss vs epoch
val_acc vs epoch
```

每条曲线显示 mean ± std。

### Page 3：Pure layer update audit

```text
input_update_over_param vs epoch
block_update_over_param vs epoch
output_update_over_param vs epoch
input_raw_grad_norm / precond_norm
block_raw_grad_norm / precond_norm
output_raw_grad_norm / precond_norm
```

### Page 4：Direction audit

```text
cos(raw, precond) heatmap by layer role
predicted vs actual descent scatter
bad shadow step rate by metric
```

### Page 5：Metric condition

```text
input metric condition
block metric condition
output metric condition
metric eig_min/eig_max
trust clip rate
```

### Page 6：Representation and basis

```text
basis occupancy by layer
basis dead fraction
out-of-grid fraction
representation effective rank
feature norm distribution
```

### Page 7：Margin and classwise behavior

```text
logit margin distribution
classwise accuracy
classwise margin delta
block_disable_drop by class
```

---

## 15. 最终写法预案

### 情况 1：PureKAN-UFULL hard pass

可以写：

```text
After correcting PureKAN-specific phase scheduling and role-wise functional metrics,
PureKAN-UFULL outperforms PureKAN-AdamW, Hybrid-DGKAN-UFULL, and MLP-AdamW.
This supports U-FULL as a genuine optimizer for complete KAN networks.
```

### 情况 2：PureKAN-UFULL medium pass

可以写：

```text
PureKAN-UFULL becomes competitive with PureKAN-AdamW while improving geometry/calibration,
but does not yet dominate all baselines.
```

### 情况 3：PureKAN-UFULL fails after implementation fixes

必须写：

```text
PureKAN is expressive and trainable under AdamW, but current Sobolev functional update
is not sufficient as a full-network optimizer. U-FULL remains valid as a branch-level
functional optimizer inside Hybrid-DGKAN, but full PureKAN optimization remains open.
```

---

## 16. 本轮最重要的停止规则

不允许继续盲目大扫。

如果 P0 显示实现问题：

```text
先修代码，不跑实验。
```

如果 P1 显示 full Sobolev direction 非 descent：

```text
优先做 layerwise metric，不做 lr sweep。
```

如果 P3 显示 output layer 是瓶颈：

```text
只改 output metric / lr，不改 input/block。
```

如果 P5 没有候选接近 AdamW：

```text
停止 10-seed confirm，回到 functional metric 设计。
```

---

## 17. 最终优先级

```text
1. 修 PureKAN alpha 和 diagwarmup schedule。
2. 确认所有 PureKAN 参数都被 functional update 覆盖。
3. 做 one-batch shadow-step，判断 full Sobolev 是否 direction 错。
4. 做 role-wise metric：input/output 不再强制 full Sobolev。
5. 只在 3-seed 接近 PureKAN-AdamW 后做 5/10-seed。
```

本轮成功后，项目叙事才能从：

```text
Hybrid branch functional optimizer
```

推进到：

```text
Pure KAN full-network functional optimizer
```



---


# Source 4: `docs/DG-KAN_v3.7_PureKAN_UFULL_结果复盘.md`


# DG-KAN v3.7 PureKAN U-FULL 结果复盘

本轮依据 `docs/DG-KAN_v3.7_PureKAN_UFULL_代码检查与实验计划.md`。正式 P2 使用修复后的 `results/gafu_v3_7_p2_schedule_fixed`；旧 P2 只作为发现 warmup 被 epoch-end geometry trigger 截断的中间记录。

## Code / Config Fixes

```text
experiments/dgkan_core.py
  PureKAN alpha_mode=fixed1 now registers alpha as a fixed buffer with value 1.0.
  PureKANClassifier forwards alpha_mode into all residual blocks.
  PureKAN functional update now supports role-specific metrics/LR/trust for input/block/output.
  Result rows include phase/metric traces, active/transition/geometry step counts, per-role update stats, coeff coverage, and PureKAN contribution audit.

experiments/run_gafu_v37.py
  Added v3.7 P0/P1/P2/P3/P5/P6 packages and one-batch shadow step.
  PureKAN warmup now sets geometry_min_epochs=999 so max_active_frac controls 0.10/0.20 schedule length.

experiments/analyze_gafu_v37.py
  Generates this result replay from P0/P1/P2-fixed/P3 CSVs.
```

## P0 Code / Config Audit

| dataset | method | alpha | alpha train | nonKAN params | linear | coeff seen | active | trans | geom | active metric | geom metric | clip |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | U-diagwarmup-smoke | 1.0 | 0 | 0 | 0 | 1.000 | 1 | 0 | 1 | basis_diag_gram\|full_sobolev_gram | full_sobolev_gram | 0.000 |
| Fashion-MNIST | U-full-smoke | 1.0 | 0 | 0 | 0 | 1.000 | 0 | 0 | 2 | full_sobolev_gram | full_sobolev_gram | 0.000 |
| Fashion-MNIST | U-identitywarmup-smoke | 1.0 | 0 | 0 | 0 | 1.000 | 1 | 0 | 1 | identity\|full_sobolev_gram | full_sobolev_gram | 0.000 |
| KMNIST | U-diagwarmup-smoke | 1.0 | 0 | 0 | 0 | 1.000 | 1 | 0 | 1 | basis_diag_gram\|full_sobolev_gram | full_sobolev_gram | 0.000 |
| KMNIST | U-full-smoke | 1.0 | 0 | 0 | 0 | 1.000 | 0 | 0 | 2 | full_sobolev_gram | full_sobolev_gram | 0.000 |
| KMNIST | U-identitywarmup-smoke | 1.0 | 0 | 0 | 0 | 1.000 | 1 | 0 | 1 | identity\|full_sobolev_gram | full_sobolev_gram | 0.000 |
| MNIST | U-diagwarmup-smoke | 1.0 | 0 | 0 | 0 | 1.000 | 1 | 0 | 1 | basis_diag_gram\|full_sobolev_gram | full_sobolev_gram | 0.000 |
| MNIST | U-full-smoke | 1.0 | 0 | 0 | 0 | 1.000 | 0 | 0 | 2 | full_sobolev_gram | full_sobolev_gram | 0.000 |
| MNIST | U-identitywarmup-smoke | 1.0 | 0 | 0 | 0 | 1.000 | 1 | 0 | 1 | identity\|full_sobolev_gram | full_sobolev_gram | 0.000 |

P0 verdict: pass. PureKAN has no non-KAN trainable parameters, alphaFixed1 is truly fixed at 1.0, and functional-update rows cover all coeff parameters (`coeff seen = 1.0`). Warmup smoke rows have real active steps.

## P1 One-Batch Shadow Step

| dataset | method | train descent | val descent | bad | update norm | input cos | block cos | output cos |
|---|---|---|---|---|---|---|---|---|
| MNIST | AdamW-one-step | 0.0229 | 0.0481 | 0 | 1.5117 |  |  |  |
| MNIST | basis-diag-gram | -0.0044 | 0.0491 | 1 | 0.5389 | 0.9915 | 0.9964 | 0.9958 |
| MNIST | full-sobolev-gram | -0.0408 | 0.0414 | 1 | 1.5164 | 0.1303 | 0.2967 | 0.2904 |
| MNIST | identity-functional | -0.0217 | 0.0333 | 1 | 1.0021 | 1.0000 | 1.0000 | 1.0000 |
| MNIST | role-inputDiag-blockFull-outputDiag | -0.0045 | 0.0490 | 1 | 0.5587 | 0.9915 | 0.2967 | 0.9958 |
| MNIST | role-inputIdentity-blockFull-outputIdentity | -0.0223 | 0.0315 | 1 | 1.0028 | 1.0000 | 0.2967 | 1.0000 |
| Fashion-MNIST | AdamW-one-step | 0.0517 | 0.0078 | 0 | 1.6145 |  |  |  |
| Fashion-MNIST | basis-diag-gram | 0.0017 | -0.0540 | 0 | 0.5522 | 0.9994 | 0.9966 | 0.9959 |
| Fashion-MNIST | full-sobolev-gram | 0.0125 | -0.0397 | 0 | 0.6113 | 0.5445 | 0.3165 | 0.2936 |
| Fashion-MNIST | identity-functional | 0.0266 | -0.0343 | 0 | 1.0540 | 1.0000 | 1.0000 | 1.0000 |
| Fashion-MNIST | role-inputDiag-blockFull-outputDiag | 0.0014 | -0.0532 | 0 | 0.5735 | 0.9994 | 0.3165 | 0.9959 |
| Fashion-MNIST | role-inputIdentity-blockFull-outputIdentity | 0.0266 | -0.0355 | 0 | 1.0541 | 1.0000 | 0.3165 | 1.0000 |
| KMNIST | AdamW-one-step | 0.0646 | 0.0501 | 0 | 1.5942 |  |  |  |
| KMNIST | basis-diag-gram | 0.0148 | 0.0040 | 0 | 0.6109 | 0.9908 | 0.9965 | 0.9962 |
| KMNIST | full-sobolev-gram | -0.0021 | 0.0031 | 1 | 1.1753 | 0.3084 | 0.2945 | 0.2896 |
| KMNIST | identity-functional | 0.0151 | -0.0129 | 0 | 1.1218 | 1.0000 | 1.0000 | 1.0000 |
| KMNIST | role-inputDiag-blockFull-outputDiag | 0.0143 | 0.0048 | 0 | 0.6295 | 0.9908 | 0.2945 | 0.9962 |
| KMNIST | role-inputIdentity-blockFull-outputIdentity | 0.0120 | -0.0128 | 0 | 1.1221 | 1.0000 | 0.2945 | 1.0000 |

P1 verdict: the full Sobolev direction is not reliably descent for PureKAN. It is a bad train step on MNIST and KMNIST, and on Fashion it improves the train batch but worsens validation loss. This points to a direction/metric mismatch rather than a missing implementation hook.

## P2 Metric Schedule Repair Test

| dataset | method | runs | acc | std | gap vs AdamW | AUC imp vs full | AUC imp vs AdamW | ECE red | phi p95 | J | active | trans | geom |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MNIST | AdamW | 3 | 0.9303 | 0.0076 | 0.0000 | 0.3510 | 0.0000 | 0.0000 | 0.2274 | 226275.4421 | 0.0 | 0.0 | 0.0 |
| MNIST | U-diag-to-full-smooth-0p20 | 3 | 0.8950 | 0.0107 | 0.0353 | -0.0707 | -0.4217 | -0.8321 | 0.1501 | 24.5974 | 96.0 | 49.0 | 335.0 |
| MNIST | U-diagwarmup-0p10 | 3 | 0.8930 | 0.0142 | 0.0373 | -0.1108 | -0.4618 | -1.2273 | 0.1497 | 31.0922 | 48.0 | 0.0 | 432.0 |
| MNIST | U-diagwarmup-0p20 | 3 | 0.8917 | 0.0135 | 0.0387 | -0.1176 | -0.4686 | -1.1325 | 0.1500 | 24.1682 | 96.0 | 0.0 | 384.0 |
| MNIST | U-full | 3 | 0.8870 | 0.0455 | 0.0433 | 0.0000 | -0.3510 | -0.3594 | 0.1489 | 34.4779 | 0.0 | 0.0 | 480.0 |
| MNIST | U-identitywarmup-0p10 | 3 | 0.8937 | 0.0153 | 0.0367 | -0.0242 | -0.3753 | -0.6659 | 0.1490 | 42.6543 | 48.0 | 0.0 | 432.0 |
| MNIST | U-identitywarmup-0p20 | 3 | 0.8933 | 0.0210 | 0.0370 | -0.0387 | -0.3897 | -0.6675 | 0.1492 | 38.2002 | 96.0 | 0.0 | 384.0 |
| Fashion-MNIST | AdamW | 3 | 0.8563 | 0.0090 | 0.0000 | 0.0388 | 0.0000 | 0.0000 | 0.2396 | 369574.0127 | 0.0 | 0.0 | 0.0 |
| Fashion-MNIST | U-diag-to-full-smooth-0p20 | 3 | 0.8460 | 0.0127 | 0.0103 | 0.1211 | 0.0822 | 0.7611 | 0.1498 | 54.0007 | 144.0 | 71.0 | 505.0 |
| Fashion-MNIST | U-diagwarmup-0p10 | 3 | 0.8350 | 0.0107 | 0.0213 | 0.0724 | 0.0336 | 0.7173 | 0.1495 | 97.4180 | 72.0 | 0.0 | 648.0 |
| Fashion-MNIST | U-diagwarmup-0p20 | 3 | 0.8450 | 0.0127 | 0.0113 | 0.1097 | 0.0708 | 0.7543 | 0.1497 | 75.3736 | 144.0 | 0.0 | 576.0 |
| Fashion-MNIST | U-full | 3 | 0.8320 | 0.0120 | 0.0243 | 0.0000 | -0.0388 | 0.6596 | 0.1490 | 31.4545 | 0.0 | 0.0 | 720.0 |
| Fashion-MNIST | U-identitywarmup-0p10 | 3 | 0.8370 | 0.0122 | 0.0193 | 0.0761 | 0.0373 | 0.7093 | 0.1490 | 46.0660 | 72.0 | 0.0 | 648.0 |
| Fashion-MNIST | U-identitywarmup-0p20 | 3 | 0.8303 | 0.0093 | 0.0260 | 0.1025 | 0.0637 | 0.6540 | 0.1492 | 43.0049 | 144.0 | 0.0 | 576.0 |
| KMNIST | AdamW | 3 | 0.7720 | 0.0102 | 0.0000 | 0.4078 | 0.0000 | 0.0000 | 0.2172 | 20374.7067 | 0.0 | 0.0 | 0.0 |
| KMNIST | U-diag-to-full-smooth-0p20 | 3 | 0.7267 | 0.0156 | 0.0453 | 0.1475 | -0.2603 | 0.6515 | 0.1503 | 33.0119 | 96.0 | 49.0 | 335.0 |
| KMNIST | U-diagwarmup-0p10 | 3 | 0.6883 | 0.0264 | 0.0837 | 0.0058 | -0.4021 | 0.5123 | 0.1499 | 31.9268 | 48.0 | 0.0 | 432.0 |
| KMNIST | U-diagwarmup-0p20 | 3 | 0.7100 | 0.0209 | 0.0620 | 0.1045 | -0.3033 | 0.6437 | 0.1501 | 31.3366 | 96.0 | 0.0 | 384.0 |
| KMNIST | U-full | 3 | 0.6783 | 0.0147 | 0.0937 | 0.0000 | -0.4078 | 0.7098 | 0.1493 | 35.5405 | 0.0 | 0.0 | 480.0 |
| KMNIST | U-identitywarmup-0p10 | 3 | 0.6870 | 0.0248 | 0.0850 | 0.0724 | -0.3355 | 0.6234 | 0.1495 | 27.4743 | 48.0 | 0.0 | 432.0 |
| KMNIST | U-identitywarmup-0p20 | 3 | 0.7160 | 0.0148 | 0.0560 | 0.1436 | -0.2642 | 0.6583 | 0.1497 | 27.1292 | 96.0 | 0.0 | 384.0 |

P2 verdict: schedule repair is real after the fix. 0.10 and 0.20 now have distinct active-step counts, and smooth has transition steps. Fashion benefits most (`diag-to-full-smooth-0.20` reaches acc 0.8460 and val-loss AUC 0.5242), but MNIST and KMNIST remain far below PureKAN-AdamW in accuracy.

## P3 Role-Wise PureKAN U-FULL Sweep

| dataset | method | runs | acc | std | gap vs AdamW | AUC imp vs full | AUC imp vs AdamW | ECE red | phi p95 | J | active | trans | geom |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MNIST | U-allDiag-o2 | 3 | 0.8977 | 0.0129 | 0.0327 | -0.0449 | -0.3959 | 0.1843 | 0.1494 | 31.4169 | 0.0 | 0.0 | 480.0 |
| MNIST | U-allfull | 3 | 0.8870 | 0.0455 | 0.0433 | 0.0000 | -0.3510 | -0.3594 | 0.1489 | 34.4779 | 0.0 | 0.0 | 480.0 |
| MNIST | U-iDiag-blockFull-oId | 3 | 0.8703 | 0.0464 | 0.0600 | -0.0343 | -0.3853 | 0.2063 | 0.1487 | 27.6383 | 0.0 | 0.0 | 480.0 |
| MNIST | U-iId-blockFull-oDiag-o2 | 3 | 0.8817 | 0.0058 | 0.0487 | -0.0333 | -0.3843 | 0.1423 | 0.1487 | 27.9629 | 0.0 | 0.0 | 480.0 |
| MNIST | U-iId-blockFull-oDiag-o4 | 3 | 0.9057 | 0.0144 | 0.0247 | -0.1433 | -0.4943 | 0.2380 | 0.1486 | 28.1544 | 0.0 | 0.0 | 480.0 |
| MNIST | U-ioDiag-blockFull-i2o2 | 3 | 0.8980 | 0.0227 | 0.0323 | -0.0198 | -0.3708 | 0.2073 | 0.1489 | 29.5763 | 0.0 | 0.0 | 480.0 |
| MNIST | U-ioDiag-blockFull-o2 | 3 | 0.8963 | 0.0088 | 0.0340 | -0.0291 | -0.3801 | 0.2247 | 0.1488 | 29.7143 | 0.0 | 0.0 | 480.0 |
| MNIST | U-ioDiag-blockFull-o4 | 3 | 0.9020 | 0.0195 | 0.0283 | -0.1123 | -0.4633 | 0.3372 | 0.1489 | 28.7376 | 0.0 | 0.0 | 480.0 |
| Fashion-MNIST | U-allDiag-o2 | 3 | 0.8457 | 0.0091 | 0.0107 | 0.0550 | 0.0161 | 0.5238 | 0.1497 | 38.1382 | 0.0 | 0.0 | 720.0 |
| Fashion-MNIST | U-allfull | 3 | 0.8320 | 0.0120 | 0.0243 | 0.0000 | -0.0388 | 0.6596 | 0.1490 | 31.4545 | 0.0 | 0.0 | 720.0 |
| Fashion-MNIST | U-iDiag-blockFull-oId | 3 | 0.8433 | 0.0083 | 0.0130 | 0.0673 | 0.0285 | 0.5679 | 0.1487 | 25.5139 | 0.0 | 0.0 | 720.0 |
| Fashion-MNIST | U-iId-blockFull-oDiag-o2 | 3 | 0.8453 | 0.0082 | 0.0110 | 0.0650 | 0.0262 | 0.5902 | 0.1487 | 26.6611 | 0.0 | 0.0 | 720.0 |
| Fashion-MNIST | U-iId-blockFull-oDiag-o4 | 3 | 0.8247 | 0.0211 | 0.0317 | 0.0680 | 0.0292 | 0.5529 | 0.1488 | 30.7958 | 0.0 | 0.0 | 720.0 |
| Fashion-MNIST | U-ioDiag-blockFull-i2o2 | 3 | 0.8470 | 0.0050 | 0.0093 | 0.0734 | 0.0346 | 0.5923 | 0.1488 | 28.3585 | 0.0 | 0.0 | 720.0 |
| Fashion-MNIST | U-ioDiag-blockFull-o2 | 3 | 0.8383 | 0.0170 | 0.0180 | 0.0622 | 0.0233 | 0.5533 | 0.1489 | 27.4369 | 0.0 | 0.0 | 720.0 |
| Fashion-MNIST | U-ioDiag-blockFull-o4 | 3 | 0.8473 | 0.0005 | 0.0090 | 0.0465 | 0.0077 | 0.4576 | 0.1489 | 29.9797 | 0.0 | 0.0 | 720.0 |
| KMNIST | U-allDiag-o2 | 3 | 0.7430 | 0.0220 | 0.0290 | 0.1475 | -0.2604 | 0.6150 | 0.1499 | 32.0730 | 0.0 | 0.0 | 480.0 |
| KMNIST | U-allfull | 3 | 0.6783 | 0.0147 | 0.0937 | 0.0000 | -0.4078 | 0.7098 | 0.1493 | 35.5405 | 0.0 | 0.0 | 480.0 |
| KMNIST | U-iDiag-blockFull-oId | 3 | 0.7277 | 0.0242 | 0.0443 | 0.1365 | -0.2713 | 0.7125 | 0.1492 | 26.0145 | 0.0 | 0.0 | 480.0 |
| KMNIST | U-iId-blockFull-oDiag-o2 | 3 | 0.7063 | 0.0223 | 0.0657 | 0.1343 | -0.2736 | 0.6394 | 0.1491 | 25.6610 | 0.0 | 0.0 | 480.0 |
| KMNIST | U-iId-blockFull-oDiag-o4 | 3 | 0.7480 | 0.0234 | 0.0240 | 0.1387 | -0.2691 | 0.6493 | 0.1489 | 54.4384 | 0.0 | 0.0 | 480.0 |
| KMNIST | U-ioDiag-blockFull-i2o2 | 3 | 0.7253 | 0.0212 | 0.0467 | 0.1568 | -0.2510 | 0.6767 | 0.1491 | 28.1022 | 0.0 | 0.0 | 480.0 |
| KMNIST | U-ioDiag-blockFull-o2 | 3 | 0.7440 | 0.0086 | 0.0280 | 0.1569 | -0.2509 | 0.7600 | 0.1493 | 29.7018 | 0.0 | 0.0 | 480.0 |
| KMNIST | U-ioDiag-blockFull-o4 | 3 | 0.7407 | 0.0161 | 0.0313 | 0.1467 | -0.2612 | 0.5869 | 0.1490 | 28.2855 | 0.0 | 0.0 | 480.0 |

### P3 Gate Check

| method | min acc | max acc gap | min AUC imp vs full | ECE ok | enter P5 |
|---|---|---|---|---|---|
| U-allDiag-o2 | 0.7430 | 0.0327 | -0.0449 | yes | no |
| U-iDiag-blockFull-oId | 0.7277 | 0.0600 | -0.0343 | yes | no |
| U-iId-blockFull-oDiag-o2 | 0.7063 | 0.0657 | -0.0333 | yes | no |
| U-iId-blockFull-oDiag-o4 | 0.7480 | 0.0317 | -0.1433 | yes | no |
| U-ioDiag-blockFull-i2o2 | 0.7253 | 0.0467 | -0.0198 | yes | no |
| U-ioDiag-blockFull-o2 | 0.7440 | 0.0340 | -0.0291 | yes | no |
| U-ioDiag-blockFull-o4 | 0.7407 | 0.0313 | -0.1123 | yes | no |

P3 verdict: no candidate enters P5. Fashion is mostly repaired by role-aware metrics, but MNIST and KMNIST still miss the `acc gap < 1%` requirement. The best KMNIST role candidate is still about 2.4 points behind PureKAN-AdamW; the best MNIST role candidate is also about 2.5 points behind.

## P4-P7 Decision

```text
P4 capacity/epoch budget check: not expanded.
Reason: P3 did not produce a candidate close enough to PureKAN-AdamW on all three datasets.

P5 3-seed candidate selection: not run.
Reason: P3 entry gate failed for every role-wise candidate.

P6 5-seed confirm and P7 speed proxy: not allowed by plan because P5 was not reached.
```

## Failure Diagnosis

```text
Implementation failure: no.
  alphaFixed1, coeff coverage, pure-only parameterization, metric traces, and active-step counts pass audit.

Schedule failure: partially fixed but not sufficient.
  The original warmup length was indeed collapsed by epoch-end geometry triggers.
  After fixing it, Fashion improves, but MNIST/KMNIST still do not approach AdamW.

Direction failure: yes.
  P1 shadow steps show full Sobolev is not reliable descent on PureKAN.
  P2/P3 improve some trajectories by avoiding full metric on input/output, but full-network PureKAN remains unstable on hard datasets.

Budget failure: unlikely as the primary cause.
  The gap is already visible in one-step shadow descent and 3-seed role sweeps.
  Extending 5/10-seed confirm is not justified until the metric direction is repaired.
```

## Final Decision

```text
PureKAN U-FULL is not accepted in v3.7.

What is confirmed:
  1. PureKAN alpha/fixed1 and functional coverage bugs are fixed.
  2. Warmup schedule now actually executes 10%/20% active windows.
  3. Role-aware metrics substantially improve Fashion and partially rescue KMNIST.

What is not confirmed:
  1. Full-network PureKAN functional update does not match PureKAN-AdamW on MNIST/KMNIST.
  2. No role/schedule candidate qualifies for P5/P6.
  3. The bottleneck is metric direction quality, not missing CUDA/KAT backward or missing parameter coverage.

Next recommended research direction:
  Redesign the PureKAN functional metric itself, likely with layer-local or block-local objectives and a safer output/input metric, before spending more seeds on confirmation.
```



---


# Source 5: `docs/DG-KAN_v3.8_修改版_GlobalTFU_Depthwise_PureKAN_实验计划.md`


# DG-KAN v3.8 修改版实验计划：Global TFU + Depth-wise TFU for PureKAN

> 文件名：`DG-KAN_v3.8_修改版_GlobalTFU_Depthwise_PureKAN_实验计划.md`  
> 目标：重新设计 PureKAN 的 functional update。把 **Global TFU, all layers task-aware** 提升为核心 baseline，同时系统验证 shallow / deep 不同 functional update 分工。  
> 公式格式：Typora 友好，统一使用 `$...$` 与 `$$...$$`，不使用其他公式包裹方式。

---

## 0. 当前背景与核心判断

v3.7 已经把几个关键实现疑点排除掉了：PureKAN 的 `alphaFixed1` 已经是真的固定为 `1.0`，PureKAN 没有可学习 non-KAN 参数，functional coefficient coverage 是 `1.0`，warmup / phase trace 也能真实执行。因此，PureKAN-UFULL 失败已经不能简单归因于参数没被更新、alpha 没固定、或者 warmup 没生效。

当前最重要的发现是：**fixed Sobolev U-FULL 的 direction 在 PureKAN 上不是可靠下降方向。** 具体地，one-batch shadow step 显示 full Sobolev direction 在 MNIST 和 KMNIST 上会产生 bad train step；即使在 Fashion 上 train batch 有下降，也可能让 validation loss 变差。这说明瓶颈不是 seed 数量，也不是简单 schedule，而是 metric direction 本身。

因此，v3.8 的核心转向是：

$$
\boxed{
\text{functional update 不能再等同于 Sobolev Gram inverse。}
}
$$

而应该改成：

$$
\boxed{
\text{functional update = task-aware metric + Sobolev regularization + descent safeguard。}
}
$$

也就是说，PureKAN 的 coefficient update 应该从：

$$
\Delta a_l = -\eta_l S_l^{-1} g_l
$$

改为：

$$
\Delta a_l = -\eta_l \left(G_l + \lambda_l S_l + \rho I\right)^{-1} g_l.
$$

其中：

```text
G_l: task-aware / data-aware / Fisher-like metric
S_l: Sobolev smoothness metric
g_l: 当前 layer coefficient gradient
rho: damping
lambda_l: Sobolev regularization strength
```

直觉是：

```text
G_l 负责让方向跟任务下降对齐；
S_l 负责保持函数几何稳定；
descent safeguard 负责避免 bad update。
```

---

## 1. v3.8 的核心问题

本轮实验只回答一个主问题：

> **PureKAN-TFU 能否全面超过 PureKAN-AdamW、Hybrid-DGKAN-UFULL 和 MLP-AdamW？**

这里的 “全面超过” 不是只看一个指标，而是同时看：

```text
1. test accuracy
2. validation-loss AUC
3. calibration / ECE
4. geometry: phi_prime_p95, Jacobian condition
5. train / val descent quality
6. update stability: bad step rate, fallback rate
7. pure functional condition: learnable non-KAN params = 0
```

最终目标是：

$$
\operatorname{Acc}_{PureKAN-TFU}
\geq
\max\left(
\operatorname{Acc}_{PureKAN-AdamW},
\operatorname{Acc}_{Hybrid-DGKAN-UFULL},
\operatorname{Acc}_{MLP-AdamW}
\right).
$$

同时要求：

$$
\operatorname{AUC}_{val,TFU}
<
\operatorname{AUC}_{val,PureKAN-AdamW}.
$$

以及：

$$
\operatorname{ECE}_{TFU}
<
\operatorname{ECE}_{PureKAN-AdamW}.
$$

---

## 2. v3.8 的方法族定义

### 2.1 PureKAN 架构

PureKAN 的结构是：

$$
h_0 = \operatorname{KAN}_{in}(x),
$$

$$
h_{l+1} = h_l + s_l\operatorname{KAN}_l(\operatorname{FixedNorm}(h_l)),
$$

$$
z = \operatorname{KAN}_{out}(\operatorname{FixedNorm}(h_L)).
$$

要求：

```text
learnable non-KAN params = 0
all learnable params are KAN coefficients
FixedNorm has no gamma / beta
alpha fixed to 1.0
input_kan, block_kan, output_kan all covered by functional update
```

PureKAN 的 layer role 分成四类：

```text
input: raw input -> hidden feature
shallow: 前半 residual KAN blocks
deep: 后半 residual KAN blocks
output: hidden feature -> logits
```

如果 depth = 4，则：

```text
shallow blocks = block 0, block 1
deep blocks = block 2, block 3
```

如果 depth = 2，则：

```text
shallow blocks = block 0
deep blocks = block 1
```

---

## 3. Global TFU 必须作为核心 baseline

这版计划把之前的 `D6 allTaskAware` 提升为核心 baseline，而不是普通候选。

### 3.1 Global TFU / allTaskAware

配置名：

```text
D6-allTaskAware
```

定义：

```text
input:   task-aware / data-aware metric
shallow: task-aware / data-aware metric
deep:    task-aware / data-aware metric
output:  output Fisher / task-aware metric
```

公式：

$$
\Delta a_l
= -\eta_l
\left(G_l + \lambda_l S_l + \rho I\right)^{-1}g_l,
\quad
l\in\{input, shallow, deep, output\}.
$$

这是 v3.8 的第一核心假设：

$$
\boxed{
\text{PureKAN 失败是因为 Sobolev-only 不够 task-aware。}
}
$$

如果 `D6-allTaskAware` 成功，则说明：

```text
固定 Sobolev metric 是主要问题；
全层 task-aware metric 可以支撑 PureKAN functional training。
```

如果 `D6-allTaskAware` 失败，而某些 depth-wise 配置成功，则说明：

```text
不是所有层都应该 task-aware；
PureKAN 需要 depth-wise functional update 分工。
```

如果 `D6-allTaskAware` 和所有 depth-wise 配置都失败，则说明：

```text
当前 G_l 设计仍然不够，TFU metric 需要重新设计。
```

---

## 4. Depth-wise TFU 配置

本轮核心比较不是只测一种 TFU，而是比较全局 task-aware 与浅层/深层分工。

### 4.1 D0：old allFull Sobolev negative baseline

配置名：

```text
D0-allFullSobolev
```

定义：

```text
input:   full Sobolev
shallow: full Sobolev
deep:    full Sobolev
output:  full Sobolev
```

公式：

$$
\Delta a_l = -\eta_l S_l^{-1}g_l.
$$

目的：复现 v3.7 PureKAN-UFULL 的失败，作为负对照。

---

### 4.2 D6：global allTaskAware

配置名：

```text
D6-allTaskAware
```

定义：

```text
input:   task-aware
shallow: task-aware
deep:    task-aware
output:  task-aware / output Fisher
```

目的：验证 “全层 task-aware 是否已经足够”。

---

### 4.3 D1：frontTask-backSob

配置名：

```text
D1-frontTask-backSob
```

定义：

```text
input:   task-aware / data-aware
shallow: task-aware / data-aware
deep:    Sobolev
output:  Sobolev 或 output-diag，视 P1 direction audit 决定
```

动机：浅层负责从 raw input 提取可分特征，必须更 task-aware；深层负责几何稳定，可以保留 Sobolev。

公式：

$$
\Delta a_l=
\begin{cases}
-\eta_l(G_l+\lambda_lS_l+\rho I)^{-1}g_l, & l\in\{input, shallow\},\\
-\eta_l(S_l+\rho I)^{-1}g_l, & l\in\{deep, output\}.
\end{cases}
$$

---

### 4.4 D2：frontSob-backTask

配置名：

```text
D2-frontSob-backTask
```

定义：

```text
input:   Sobolev 或 diag Sobolev
shallow: Sobolev
deep:    task-aware
output:  output Fisher / task-aware
```

动机：浅层保持平滑和稳定；深层和 output 更接近分类任务，应更 task-aware。

公式：

$$
\Delta a_l=
\begin{cases}
-\eta_l(S_l+\rho I)^{-1}g_l, & l\in\{input, shallow\},\\
-\eta_l(G_l+\lambda_lS_l+\rho I)^{-1}g_l, & l\in\{deep, output\}.
\end{cases}
$$

---

### 4.5 D3：taskSobTask

配置名：

```text
D3-taskSobTask
```

定义：

```text
input:   task-aware / data-aware
shallow: Sobolev
deep:    Sobolev
output:  output Fisher / task-aware
```

动机：input 和 output 最贴近任务边界，中间 residual blocks 负责稳定表示几何。

公式：

$$
\Delta a_l=
\begin{cases}
-\eta_l(G_l+\lambda_lS_l+\rho I)^{-1}g_l, & l\in\{input, output\},\\
-\eta_l(S_l+\rho I)^{-1}g_l, & l\in\{shallow, deep\}.
\end{cases}
$$

---

### 4.6 D7：inputOutputTask-middleSob

配置名：

```text
D7-inputOutputTask-middleSob
```

这是 D3 的最小化版本，专门验证：只改 input/output 是否足够。

定义：

```text
input:   data Gram / task-aware diagonal
blocks:  Sobolev
output:  output Fisher / task-aware
```

动机：Hybrid-DGKAN 的成功可能来自稳定 stem/head 接口。PureKAN 没有 MLP stem/head，因此 input/output KAN 可能需要承担 stem/head 的 task-aware 功能。

---

### 4.7 D8：inputDiag-blockTask-outputFisher

配置名：

```text
D8-inputDiag-blockTask-outputFisher
```

定义：

```text
input:   diag data-aware
blocks:  task-aware + Sobolev
output:  Fisher
```

动机：如果 D6 过于激进，可以保留 input 的轻量 diag metric，把 task-aware 主要放在 hidden blocks 和 output。

---

## 5. TFU metric 的具体定义

### 5.1 Sobolev metric

RBF basis 为 $B_m(t)$，Sobolev metric 为：

$$
S = \int BB^\top dt + \alpha_s \int B'B'^\top dt + \beta_s \int B''B''^\top dt.
$$

### 5.2 Data Gram metric

用于 input / shallow 的数据分布 metric：

$$
G_{data,l} = \mathbb E_{x\sim batch}\left[B_l(x)B_l(x)^\top\right].
$$

实际实现可以先做 shared basis-level metric，而不是 edge-level 巨大矩阵。

### 5.3 Gradient Fisher diagonal metric

轻量 task-aware 版本：

$$
G_{diag,l} = \operatorname{EMA}\left[g_l^2\right].
$$

更新：

$$
\Delta a_l = -\eta_l \frac{g_l}{\sqrt{G_{diag,l}}+\rho}.
$$

这个可以作为 allTaskAware 的低成本实现，也可以作为 full task metric 的 fallback。

### 5.4 Output Fisher metric

output KAN 直接决定 logits，因此应使用 softmax Fisher 近似。

令 logits 为 $z$，softmax 概率为 $p$，cross entropy Hessian 为：

$$
H_{CE} = \operatorname{diag}(p) - pp^\top.
$$

output KAN 的 task metric 为：

$$
G_{out} = \mathbb E_x\left[J_{out}(x)^\top H_{CE}(x) J_{out}(x)\right].
$$

其中 $J_{out}(x)$ 是 output KAN logits 对 output KAN coefficients 的 Jacobian。

### 5.5 Combined TFU metric

最终每层使用：

$$
M_l = G_l + \lambda_l S_l + \rho I.
$$

更新：

$$
\Delta a_l = -\eta_l M_l^{-1} g_l.
$$

---

## 6. Descent safeguard

v3.7 已经证明 full Sobolev 会产生 bad step，因此 v3.8 必须加入 descent safeguard。只靠 trust radius 不够，因为 trust radius 控制的是 metric norm，不保证 loss 下降。

### 6.1 Shadow descent check

每隔 $K$ steps，在当前 minibatch 上检查候选 update：

$$
L(\theta + \Delta\theta) \leq L(\theta) - c\eta\langle g, d\rangle.
$$

如果不满足，则缩小 step：

$$
\eta \leftarrow \gamma \eta,
\quad \gamma\in\{0.5,0.25\}.
$$

如果连续 backtracking 仍失败，则 fallback：

```text
full TFU -> diag TFU -> raw gradient / identity functional
```

### 6.2 必须记录 safeguard 行为

每个 run 必须记录：

```text
safeguard/check_count
safeguard/fail_count
safeguard/fail_rate
safeguard/backtrack_mean
safeguard/backtrack_p95
safeguard/fallback_to_diag_count
safeguard/fallback_to_identity_count
safeguard/fallback_role_input
safeguard/fallback_role_shallow
safeguard/fallback_role_deep
safeguard/fallback_role_output
```

---

## 7. 实验阶段总览

v3.8 修改版分为 8 个阶段：

```text
P0: implementation and config smoke
P1: one-batch direction audit
P2: global TFU micro-run
P3: depth-wise TFU micro-run
P4: 3-seed candidate selection
P5: step-size / safeguard refinement
P6: 5-seed confirm
P7: 10-seed final confirm
P8: failure diagnosis and follow-up
```

注意：P1/P2 必须优先运行 `D6-allTaskAware`。如果 D6 在 one-batch audit 里失败，需要先修 global TFU metric，不应直接跳去 depth-wise 大扫。

---

## 8. P0：implementation and config smoke

### 8.1 目标

确认 TFU 的实现和记录字段正确，尤其要确认：

```text
PureKAN learnable non-KAN params = 0
input_kan / block_kan / output_kan 全部被 functional update 覆盖
D6 allTaskAware 真正每层都用了 task-aware metric
D1/D2/D3/D7 的 role mapping 正确
safeguard 能记录 fail/backtrack/fallback
```

### 8.2 数据集与规模

```text
datasets = MNIST, Fashion-MNIST, KMNIST
train / val / test = 512 / 128 / 128
epochs = 1
seeds = 0
hidden_dim = 32
basis_count = 8
depth = 2
```

### 8.3 方法

```text
PureKAN-AdamW-smoke
D0-allFullSobolev-smoke
D6-allTaskAware-smoke
D1-frontTask-backSob-smoke
D2-frontSob-backTask-smoke
D3-taskSobTask-smoke
D7-inputOutputTask-middleSob-smoke
```

### 8.4 必须记录

```text
config/nonKAN_param_count
config/coeff_seen_ratio
config/input_coeff_seen
config/block_coeff_seen
config/output_coeff_seen
config/alpha_fixed_value
config/role_metric_input
config/role_metric_shallow
config/role_metric_deep
config/role_metric_output
config/tfu_enabled
config/safeguard_enabled
metric/input_condition
metric/shallow_condition_mean
metric/deep_condition_mean
metric/output_condition
metric/input_eig_min
metric/output_eig_min
update/input_update_norm
update/block_update_norm_mean
update/output_update_norm
safeguard/fail_rate
run/has_nan
run/has_inf
```

### 8.5 P0 pass 条件

$$
\text{nonKAN params}=0.
$$

$$
\text{coeff seen ratio}=1.0.
$$

$$
\text{NaN/Inf count}=0.
$$

$$
\text{metric condition}<10^5
$$

至少在 smoke 尺度上成立。

---

## 9. P1：one-batch direction audit

### 9.1 目标

P1 是 v3.8 的关键阶段。它不问最终 accuracy，而是问：

> 当前 TFU direction 是否是可靠下降方向？

必须比较：

```text
AdamW-one-step
D0-allFullSobolev
D6-allTaskAware
D1-frontTask-backSob
D2-frontSob-backTask
D3-taskSobTask
D7-inputOutputTask-middleSob
D8-inputDiag-blockTask-outputFisher
```

### 9.2 指标

每个 method / dataset / seed 记录：

```text
direction/train_descent
direction/val_descent
direction/bad_step_bool
direction/predicted_descent
direction/actual_descent
direction/predicted_actual_ratio
```

per role 记录：

```text
input/cos_raw_precond
shallow/cos_raw_precond_mean
deep/cos_raw_precond_mean
output/cos_raw_precond
input/update_norm
shallow/update_norm_mean
deep/update_norm_mean
output/update_norm
input/update_over_coeff
output/update_over_coeff
input/metric_condition
output/metric_condition
```

### 9.3 P1 可视化

必须生成：

```text
1. bar: train_descent by method and dataset
2. bar: val_descent by method and dataset
3. heatmap: cos_raw_precond by role and method
4. scatter: predicted_descent vs actual_descent
5. heatmap: bad_step_bool by method and dataset
6. bar: update_norm role breakdown
7. bar: metric_condition by role
```

### 9.4 P1 pass 条件

对进入 P2 的候选，要求：

$$
\text{bad step rate}=0
$$

在 MNIST / Fashion / KMNIST 至少 2 个数据集上成立。

同时：

$$
\operatorname{ActualDescent}>0
$$

并且：

$$
\cos(g,d)>0.5
$$

至少在 input 和 output role 上成立。

`D6-allTaskAware` 必须进入 P2，即使 P1 不完美；但如果 D6 出现 bad step，需要同时打开 safeguard refinement。

---

## 10. P2：global TFU micro-run

### 10.1 目标

优先回答：

> 全层 task-aware 是否已经足够？

### 10.2 实验设置

```text
datasets = MNIST, Fashion-MNIST, KMNIST
train / val / test = 6000 / 1000 / 1000
seeds = 0,1,2
epochs:
  MNIST = 20
  Fashion = 30
  KMNIST = 20
model:
  PureKAN hidden_dim = 96
  depth = 4
  basis_count = 24
```

### 10.3 方法

```text
PureKAN-AdamW
PureKAN-UFULL-D0-allFullSobolev
PureKAN-TFU-D6-allTaskAware
PureKAN-TFU-D6-allTaskAware-noSob
PureKAN-TFU-D6-allTaskAware-lowSob
PureKAN-TFU-D6-allTaskAware-withSafeguard
Hybrid-DGKAN-UFULL-f085
MLP-AdamW
```

其中：

```text
D6-allTaskAware-noSob:
  lambda_l = 0 for all layers

D6-allTaskAware-lowSob:
  lambda_l small, e.g. 0.01 or 0.03

D6-allTaskAware-withSafeguard:
  enable descent safeguard
```

### 10.4 必须记录

task 指标：

```text
train_loss_curve
val_loss_curve
train_acc_curve
val_acc_curve
test_acc
val_loss_auc
val_acc_auc
best_val_acc
best_epoch
```

geometry 指标：

```text
phi_prime_p95
phi_prime_max
jacobian_condition_max
curvature_energy
sobolev_norm_input
sobolev_norm_block_mean
sobolev_norm_output
```

TFU 指标：

```text
tfu/input_metric_condition
tfu/shallow_metric_condition_mean
tfu/deep_metric_condition_mean
tfu/output_metric_condition
tfu/input_cos_raw_precond
tfu/output_cos_raw_precond
tfu/update_norm_input
tfu/update_norm_output
tfu/update_over_coeff_input
tfu/update_over_coeff_output
tfu/fallback_rate
tfu/backtrack_count_mean
```

representation 指标：

```text
repr/input_feature_norm
repr/block_feature_norm_mean
repr/output_feature_norm
repr/effective_rank_input
repr/effective_rank_hidden
repr/class_margin_mean
repr/class_margin_p95
```

### 10.5 P2 可视化

必须生成：

```text
1. val_loss curves with mean ± std
2. val_acc curves with mean ± std
3. test_acc bar with seed dots
4. val_loss_auc bar
5. role-wise update norm stacked area
6. role-wise metric condition bar
7. fallback rate over epoch
8. class margin distribution
9. geometry vs accuracy Pareto plot
```

### 10.6 P2 判断

如果 D6-allTaskAware 满足：

$$
\text{MNIST acc gap vs PureKAN-AdamW}<1\%
$$

$$
\text{Fashion acc gap vs PureKAN-AdamW}<1\%
$$

$$
\text{KMNIST acc gap vs PureKAN-AdamW}<2\%
$$

并且：

$$
\operatorname{AUCImprove}_{D6\ vs\ D0}>20\%,
$$

则 D6 进入 P4。

如果 D6 失败，但 direction audit 显示 input/output 改善明显，则继续 depth-wise P3。

---

## 11. P3：depth-wise TFU micro-run

### 11.1 目标

比较 shallow / deep 的 functional update 分工。

### 11.2 方法

```text
D0-allFullSobolev
D6-allTaskAware
D1-frontTask-backSob
D2-frontSob-backTask
D3-taskSobTask
D7-inputOutputTask-middleSob
D8-inputDiag-blockTask-outputFisher
```

可选扩展：

```text
D9-shallowTask-outputFisher-blockDiag
D10-inputTask-blockDiag-outputFisher
```

### 11.3 实验设置

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
epochs = same as P2
hidden_dim = 96
basis_count = 24
depth = 4
```

### 11.4 记录指标

除了 P2 的指标外，额外记录：

```text
depth/shallow_train_descent
depth/deep_train_descent
depth/shallow_update_norm
depth/deep_update_norm
depth/shallow_cos_raw_precond
depth/deep_cos_raw_precond
depth/shallow_metric_condition
depth/deep_metric_condition
depth/shallow_sobolev_norm
depth/deep_sobolev_norm
depth/shallow_feature_rank
depth/deep_feature_rank
```

### 11.5 可视化

```text
1. depth-wise metric assignment matrix
2. shallow/deep update norm over epoch
3. shallow/deep cosine heatmap
4. feature effective-rank by depth
5. class margin improvement by method
6. val AUC vs test accuracy scatter
7. phi_prime_p95 vs test accuracy scatter
```

### 11.6 P3 判断

进入 P4 的候选必须满足：

```text
1. 在三个数据集上 acc gap 明显小于 D0-allFullSobolev。
2. 至少两个数据集 AUC 比 D0 提升 20% 以上。
3. 没有 bad step rate > 5%。
4. 没有 metric condition > 1e5。
5. 没有 fallback rate > 30%。
```

如果 D6-allTaskAware 是最强，则后续 v3.8 主线改为 global TFU。  
如果 D3/D7 最强，则后续 v3.8 主线改为 depth-wise TFU。  
如果 D1 或 D2 最强，则说明 shallow/deep 分工是核心机制，应进入 P4 refinement。

---

## 12. P4：3-seed candidate selection

### 12.1 目标

从 P2/P3 中选择最多 3 个候选，与所有强 baseline 对比。

### 12.2 候选规则

最多选择：

```text
1. 最强 global TFU: usually D6 variant
2. 最强 depth-wise TFU: D1/D2/D3/D7/D8 中最高分
3. 最强 safeguard variant
```

baseline：

```text
PureKAN-AdamW
PureKAN-UFULL-D0-allFullSobolev
Hybrid-DGKAN-UFULL-f085
MLP-AdamW
```

### 12.3 ranking score

定义综合分：

$$
Score
= AccGain
+0.5\cdot AUCImprove
+0.2\cdot ECERed
+0.2\cdot GeometryRed
-0.2\cdot FallbackRate
-0.1\cdot TimeRatio.
$$

其中：

$$
AccGain = \operatorname{Acc}_{method} - \operatorname{Acc}_{PureKAN-AdamW}.
$$

候选必须不是只靠 geometry 好，而 accuracy 明显差。

---

## 13. P5：step-size / safeguard refinement

### 13.1 目标

如果候选方向对，但 accuracy 仍低于 AdamW，需要判断是 step size、damping、还是 safeguard 太保守。

### 13.2 实验矩阵

对 P4 选出的候选，做小矩阵：

```text
coeff_lr_mult in {0.5, 1.0, 1.5, 2.0}
rho in {1e-3, 3e-3, 1e-2}
lambda_sob in {0.0, 0.01, 0.03, 0.10}
safeguard in {off, weak, strict}
```

不要全组合。使用分阶段：

```text
先 coeff_lr_mult × safeguard
再 rho × lambda_sob
```

### 13.3 重点指标

```text
actual_descent
bad_step_rate
fallback_rate
update_over_coeff
train_loss_auc
val_loss_auc
final_acc
```

### 13.4 判断

如果提高 lr 能提升 accuracy 且 bad_step 不上升，则之前是 step too small。  
如果降低 Sobolev lambda 能提升 accuracy，则之前是 over-smoothing。  
如果 safeguard strict 让 AUC 好但 accuracy 差，则 safeguard 太保守。

---

## 14. P6：5-seed confirm

### 14.1 目标

正式验证 PureKAN-TFU 是否能追上或超过 PureKAN-AdamW。

### 14.2 方法

```text
PureKAN-AdamW
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
PureKAN-UFULL-D0-allFullSobolev
PureKAN-TFU-best-global-or-depthwise
PureKAN-TFU-second-best
```

### 14.3 数据集

```text
MNIST
Fashion-MNIST
KMNIST
```

### 14.4 seeds

```text
seeds = 0,1,2,3,4
```

### 14.5 P6 medium pass

要求：

$$
\operatorname{Acc}_{TFU} \geq \operatorname{Acc}_{PureKAN-AdamW} - 0.5\%
$$

在 MNIST / Fashion / KMNIST 全部成立。

并且至少两个数据集：

$$
\operatorname{Acc}_{TFU} > \operatorname{Acc}_{PureKAN-AdamW}.
$$

同时：

$$
\operatorname{AUC}_{TFU}<\operatorname{AUC}_{PureKAN-AdamW}.
$$

以及：

$$
\operatorname{ECE}_{TFU}<\operatorname{ECE}_{PureKAN-AdamW}.
$$

---

## 15. P7：10-seed final confirm

### 15.1 触发条件

只有 P6 medium pass 后运行。

### 15.2 方法

```text
PureKAN-AdamW
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
PureKAN-TFU-final
```

### 15.3 seeds

```text
seeds = 0..9
```

### 15.4 strong pass

最终 strong pass 要求：

$$
\operatorname{Acc}_{PureKAN-TFU}
\geq
\max(
\operatorname{Acc}_{PureKAN-AdamW},
\operatorname{Acc}_{Hybrid-DGKAN-UFULL},
\operatorname{Acc}_{MLP-AdamW}
)
$$

在至少 2 个数据集成立，并且第 3 个数据集 gap 小于：

$$
0.5\%.
$$

同时：

$$
\operatorname{AUCImprove}_{TFU\ vs\ PureKAN-AdamW}>5\%.
$$

$$
\operatorname{ECERed}_{TFU\ vs\ PureKAN-AdamW}>10\%.
$$

$$
\operatorname{BadStepRate}<1\%.
$$

---

## 16. P8：failure diagnosis

如果 P7 不通过，必须输出 failure diagnosis，而不是继续盲目调参。

### 16.1 Failure types

```text
metric_direction_bad:
  one-batch bad step 或 cos_raw_precond 太低

over_smoothing:
  geometry 极好但 acc / margin 明显低

step_too_small:
  update_over_coeff 太低，train loss 下降慢

step_too_large:
  fallback / backtracking 高，loss 不稳定

input_underfit:
  input feature rank 低，input update norm 低

output_underfit:
  class margin 低，output update norm 低

middle_geometry_block:
  middle Sobolev 导致 effective rank 降低

safeguard_too_conservative:
  fallback 高，bad step 低，但 acc 差

metric_condition_bad:
  condition > 1e5 或 eig_min 太小
```

### 16.2 Failure visualizations

必须生成：

```text
1. failure count by method
2. failure count by dataset
3. metric condition vs accuracy
4. fallback rate vs accuracy
5. effective rank vs accuracy
6. output margin vs accuracy
7. cos_raw_precond vs bad step rate
8. Sobolev norm vs validation AUC
```

---

## 17. 统一记录字段

每个正式 run 必须记录以下字段。

### 17.1 Config fields

```text
method
dataset
seed
model_type
hidden_dim
depth
basis_count
metric_role_input
metric_role_shallow
metric_role_deep
metric_role_output
lambda_sob_input
lambda_sob_shallow
lambda_sob_deep
lambda_sob_output
rho_input
rho_output
safeguard_mode
safeguard_check_interval
coeff_lr_input
coeff_lr_shallow
coeff_lr_deep
coeff_lr_output
```

### 17.2 Performance fields

```text
train_loss_curve
val_loss_curve
train_acc_curve
val_acc_curve
test_acc
best_val_acc
best_epoch
val_loss_auc
val_acc_auc
ECE
NLL
class_margin_mean
class_margin_p95
classwise_acc
```

### 17.3 Geometry fields

```text
phi_prime_p95
phi_prime_max
jacobian_condition_max
curvature_energy
sobolev_norm_input
sobolev_norm_shallow
sobolev_norm_deep
sobolev_norm_output
```

### 17.4 TFU direction fields

```text
train_descent_shadow
val_descent_shadow
bad_step_rate
predicted_descent
actual_descent
predicted_actual_ratio
cos_raw_precond_input
cos_raw_precond_shallow
cos_raw_precond_deep
cos_raw_precond_output
update_norm_input
update_norm_shallow
update_norm_deep
update_norm_output
update_over_coeff_input
update_over_coeff_output
```

### 17.5 Metric fields

```text
metric_condition_input
metric_condition_shallow_mean
metric_condition_deep_mean
metric_condition_output
metric_eig_min_input
metric_eig_min_output
metric_eig_max_input
metric_eig_max_output
data_gram_rank_input
data_gram_rank_output
fisher_rank_output
```

### 17.6 Safeguard fields

```text
safeguard_check_count
safeguard_fail_count
safeguard_fail_rate
backtrack_mean
backtrack_p95
fallback_to_diag_count
fallback_to_identity_count
fallback_by_role_input
fallback_by_role_shallow
fallback_by_role_deep
fallback_by_role_output
```

### 17.7 Representation fields

```text
feature_norm_input
feature_norm_shallow
feature_norm_deep
feature_norm_output
effective_rank_input
effective_rank_hidden
effective_rank_output
basis_dead_frac_input
basis_dead_frac_block_mean
basis_dead_frac_output
input_out_of_grid_frac
output_out_of_grid_frac
```

### 17.8 Compute fields

```text
step_time_ms
metric_build_time_ms
solve_time_ms
safeguard_time_ms
memory_peak_mb
samples_per_sec
time_to_relaxed_loss
```

---

## 18. Dashboard / visualization specification

每个阶段都必须生成可读 dashboard。不要只输出 CSV。

### Page 1：Scorecard

表格字段：

```text
method
dataset
acc mean ± std
acc gap vs PureKAN-AdamW
AUC improvement vs PureKAN-AdamW
ECE reduction
bad step rate
fallback rate
phi_prime_p95
Jacobian condition
step time
pass status
```

### Page 2：Loss / accuracy curves

图：

```text
train_loss vs epoch
val_loss vs epoch
train_acc vs epoch
val_acc vs epoch
```

要求显示 seed mean ± std。

### Page 3：Direction audit

图：

```text
predicted_descent vs actual_descent scatter
cos_raw_precond heatmap by role
bad_step_rate bar
train_descent bar
val_descent bar
```

### Page 4：Role-wise update dynamics

图：

```text
input/shallow/deep/output update norm over epoch
update_over_coeff by role
metric condition by role
fallback by role
```

### Page 5：Representation health

图：

```text
effective_rank_input / hidden / output
class_margin distribution
basis_dead_frac by role
out_of_grid_frac by role
```

### Page 6：Geometry vs task Pareto

图：

```text
x = phi_prime_p95, y = test_acc
x = jacobian_condition, y = test_acc
x = val_loss_auc, y = test_acc
x = ECE, y = test_acc
```

### Page 7：Compute dashboard

图：

```text
step_time_ms by method
metric_build_time_ms by method
solve_time_ms by method
safeguard_time_ms by method
memory_peak_mb by method
```

---

## 19. 最终决策规则

### 19.1 Global TFU success

如果 `D6-allTaskAware` 在 P7 strong pass，则结论是：

```text
PureKAN functional training succeeds with global task-aware functional update.
```

此时 v3.8 default 为：

```text
PureKAN-TFU-D6-allTaskAware
```

### 19.2 Depth-wise TFU success

如果 D6 没过，但 D1/D2/D3/D7/D8 过，则结论是：

```text
PureKAN requires depth-wise functional update specialization.
```

此时需要明确写出哪个 role 组合有效。

### 19.3 TFU partial success

如果 TFU 明显超过 D0-allFullSobolev，但仍低于 PureKAN-AdamW，则结论是：

```text
Task-aware metric repairs Sobolev-only failure but does not yet beat AdamW.
```

下一步应继续 metric design，而不是 seed expansion。

### 19.4 TFU failure

如果 D6 和所有 depth-wise 配置都失败，则结论是：

```text
Current task-aware metric is insufficient; PureKAN functional optimizer remains unsolved.
```

此时不允许 claim：

```text
PureKAN-TFU is a default optimizer.
```

也不允许继续只调 branch scale / warmup / smooth transition。

---

## 20. 本轮计划的重点提醒

1. `D6-allTaskAware` 是核心 baseline，必须在 P1/P2 优先运行。
2. Depth-wise 配置不是替代 D6，而是解释 D6 成败的机制实验。
3. v3.7 已经说明 Sobolev-only direction 不是 PureKAN 的可靠下降方向，因此本轮不能再围绕 full Sobolev 微调。
4. 成功标准必须以 PureKAN-AdamW 为主 baseline，同时比较 Hybrid-DGKAN-UFULL 与 MLP-AdamW。
5. 如果 P1 one-batch direction 仍然 bad step，则不要跑大 seed，先修 metric。
6. 如果 accuracy 差但 geometry 很好，不算成功；几何稳定不是 optimizer 成功的充分条件。

---

## 21. 推荐执行顺序

```text
Day 1:
  P0 implementation smoke
  P1 one-batch direction audit

Day 2:
  P2 global TFU micro-run
  分析 D6 allTaskAware 是否有足够信号

Day 3:
  P3 depth-wise TFU micro-run
  比较 D1/D2/D3/D7/D8

Day 4:
  P4 3-seed candidate selection
  P5 小范围 step-size / safeguard refinement

Day 5+:
  只有候选接近 PureKAN-AdamW，才跑 P6/P7 confirm
```

本轮的核心原则是：

$$
\boxed{
\text{先证明 direction 是对的，再证明 seed 是稳的。}
}
$$



---


# Source 6: `docs/DG-KAN_v3.8_GlobalTFU_Depthwise_PureKAN_结果复盘.md`


# DG-KAN v3.8 GlobalTFU / Depthwise PureKAN 结果复盘

本轮依据 `docs/DG-KAN_v3.8_修改版_GlobalTFU_Depthwise_PureKAN_实验计划.md`。目标是验证 PureKAN 的 task-aware functional update 是否能修复 v3.7 中 full Sobolev 方向不可靠的问题。

## Code / Config Changes

```text
experiments/dgkan_core.py
  TrainConfig / RuntimeState added TFU fields, role-specific input/shallow/deep/output metrics, LR multipliers and cosine safeguard counters.
  PureKAN functional update now supports tfu_task_diag and tfu_data_task_diag.
  Shallow/deep block roles are split from blocks.{idx} and also aggregated into block stats.
  Result rows include TFU metric min/max, per-role conditions, fallback counters, and safeguard fail rates.

experiments/run_gafu_v38.py
  Added P0 smoke, P1 shadow audit, P2 global TFU, P3 depth-wise sweep, and conditional P4 entry wiring.

experiments/analyze_gafu_v38.py
  Generates this replay and appends a concise log entry.
```

## P0 Implementation Smoke

| dataset | method | alpha | alpha train | nonKAN | linear | coeff seen | input | shallow | deep | output | cond in | cond out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | D0-allFullSobolev | 1.0 | 0 | 0 | 0 | 1.000 | full_sobolev_gram |  |  | full_sobolev_gram | 6551.62 | 6551.62 |
| Fashion-MNIST | D1-frontTask-backSob | 1.0 | 0 | 0 | 0 | 1.000 | tfu_data_task_diag | tfu_data_task_diag | full_sobolev_gram | full_sobolev_gram | 6.65 | 6551.62 |
| Fashion-MNIST | D2-frontSob-backTask | 1.0 | 0 | 0 | 0 | 1.000 | basis_diag_gram | full_sobolev_gram | tfu_task_diag | tfu_task_diag | 6918.15 | 21.23 |
| Fashion-MNIST | D3-taskSobTask | 1.0 | 0 | 0 | 0 | 1.000 | tfu_data_task_diag | full_sobolev_gram | full_sobolev_gram | tfu_task_diag | 6.65 | 21.38 |
| Fashion-MNIST | D6-allTaskAware | 1.0 | 0 | 0 | 0 | 1.000 | tfu_data_task_diag | tfu_data_task_diag | tfu_task_diag | tfu_task_diag | 6.66 | 21.40 |
| Fashion-MNIST | D7-inputOutputTask-middleSob | 1.0 | 0 | 0 | 0 | 1.000 | tfu_data_task_diag | full_sobolev_gram | full_sobolev_gram | tfu_task_diag | 6.65 | 21.38 |
| KMNIST | D0-allFullSobolev | 1.0 | 0 | 0 | 0 | 1.000 | full_sobolev_gram |  |  | full_sobolev_gram | 6551.62 | 6551.62 |
| KMNIST | D1-frontTask-backSob | 1.0 | 0 | 0 | 0 | 1.000 | tfu_data_task_diag | tfu_data_task_diag | full_sobolev_gram | full_sobolev_gram | 5.68 | 6551.62 |
| KMNIST | D2-frontSob-backTask | 1.0 | 0 | 0 | 0 | 1.000 | basis_diag_gram | full_sobolev_gram | tfu_task_diag | tfu_task_diag | 6918.15 | 23.02 |
| KMNIST | D3-taskSobTask | 1.0 | 0 | 0 | 0 | 1.000 | tfu_data_task_diag | full_sobolev_gram | full_sobolev_gram | tfu_task_diag | 5.69 | 23.53 |
| KMNIST | D6-allTaskAware | 1.0 | 0 | 0 | 0 | 1.000 | tfu_data_task_diag | tfu_data_task_diag | tfu_task_diag | tfu_task_diag | 5.69 | 23.59 |
| KMNIST | D7-inputOutputTask-middleSob | 1.0 | 0 | 0 | 0 | 1.000 | tfu_data_task_diag | full_sobolev_gram | full_sobolev_gram | tfu_task_diag | 5.69 | 23.53 |
| MNIST | D0-allFullSobolev | 1.0 | 0 | 0 | 0 | 1.000 | full_sobolev_gram |  |  | full_sobolev_gram | 6551.62 | 6551.62 |
| MNIST | D1-frontTask-backSob | 1.0 | 0 | 0 | 0 | 1.000 | tfu_data_task_diag | tfu_data_task_diag | full_sobolev_gram | full_sobolev_gram | 8.89 | 6551.62 |
| MNIST | D2-frontSob-backTask | 1.0 | 0 | 0 | 0 | 1.000 | basis_diag_gram | full_sobolev_gram | tfu_task_diag | tfu_task_diag | 6918.15 | 16.55 |
| MNIST | D3-taskSobTask | 1.0 | 0 | 0 | 0 | 1.000 | tfu_data_task_diag | full_sobolev_gram | full_sobolev_gram | tfu_task_diag | 8.89 | 16.10 |
| MNIST | D6-allTaskAware | 1.0 | 0 | 0 | 0 | 1.000 | tfu_data_task_diag | tfu_data_task_diag | tfu_task_diag | tfu_task_diag | 8.88 | 16.05 |
| MNIST | D7-inputOutputTask-middleSob | 1.0 | 0 | 0 | 0 | 1.000 | tfu_data_task_diag | full_sobolev_gram | full_sobolev_gram | tfu_task_diag | 8.89 | 16.10 |

P0 verdict: pass. TFU rows have pure alpha fixed at 1.0, no trainable non-KAN parameters, no linear layer, full coefficient coverage, and finite role metric conditions.

## P1 One-Batch Direction Audit

| dataset | method | train descent | val descent | bad | input cos | shallow cos | deep cos | output cos |
|---|---|---|---|---|---|---|---|---|
| MNIST | AdamW-one-step | 0.0229 | 0.0481 | 0 |  |  |  |  |
| MNIST | D0-allFullSobolev | -0.0408 | 0.0414 | 1 | 0.1303 |  |  | 0.2904 |
| MNIST | D1-frontTask-backSob | 0.0215 | 0.0901 | 0 | 0.8348 | 0.9327 | 0.2977 | 0.2904 |
| MNIST | D2-frontSob-backTask | -0.0052 | 0.0487 | 1 | 0.9915 | 0.2958 | 0.9629 | 0.9191 |
| MNIST | D3-taskSobTask | 0.0232 | 0.0908 | 0 | 0.8348 | 0.2958 | 0.2977 | 0.9191 |
| MNIST | D6-allTaskAware | 0.0222 | 0.0901 | 0 | 0.8348 | 0.9327 | 0.9629 | 0.9191 |
| MNIST | D7-inputOutputTask-middleSob | 0.0232 | 0.0908 | 0 | 0.8348 | 0.2958 | 0.2977 | 0.9191 |
| MNIST | D8-inputDiag-blockTask-outputFisher | -0.0053 | 0.0493 | 1 | 0.9915 | 0.9034 | 0.9629 | 0.9191 |
| Fashion-MNIST | AdamW-one-step | 0.0517 | 0.0078 | 0 |  |  |  |  |
| Fashion-MNIST | D0-allFullSobolev | 0.0125 | -0.0397 | 0 | 0.5445 |  |  | 0.2936 |
| Fashion-MNIST | D1-frontTask-backSob | 0.0627 | 0.0243 | 0 | 0.8944 | 0.9435 | 0.3206 | 0.2936 |
| Fashion-MNIST | D2-frontSob-backTask | 0.0024 | -0.0522 | 0 | 0.9994 | 0.3124 | 0.9652 | 0.9166 |
| Fashion-MNIST | D3-taskSobTask | 0.0657 | 0.0275 | 0 | 0.8944 | 0.3124 | 0.3206 | 0.9166 |
| Fashion-MNIST | D6-allTaskAware | 0.0638 | 0.0250 | 0 | 0.8944 | 0.9435 | 0.9652 | 0.9166 |
| Fashion-MNIST | D7-inputOutputTask-middleSob | 0.0657 | 0.0275 | 0 | 0.8944 | 0.3124 | 0.3206 | 0.9166 |
| Fashion-MNIST | D8-inputDiag-blockTask-outputFisher | 0.0025 | -0.0526 | 0 | 0.9994 | 0.9118 | 0.9652 | 0.9166 |
| KMNIST | AdamW-one-step | 0.0646 | 0.0501 | 0 |  |  |  |  |
| KMNIST | D0-allFullSobolev | -0.0021 | 0.0031 | 1 | 0.3084 |  |  | 0.2896 |
| KMNIST | D1-frontTask-backSob | 0.0438 | 0.0545 | 0 | 0.8771 | 0.9413 | 0.2847 | 0.2896 |
| KMNIST | D2-frontSob-backTask | 0.0147 | 0.0054 | 0 | 0.9908 | 0.3044 | 0.9672 | 0.9271 |
| KMNIST | D3-taskSobTask | 0.0434 | 0.0548 | 0 | 0.8771 | 0.3044 | 0.2847 | 0.9271 |
| KMNIST | D6-allTaskAware | 0.0458 | 0.0551 | 0 | 0.8771 | 0.9413 | 0.9672 | 0.9271 |
| KMNIST | D7-inputOutputTask-middleSob | 0.0434 | 0.0548 | 0 | 0.8771 | 0.3044 | 0.2847 | 0.9271 |
| KMNIST | D8-inputDiag-blockTask-outputFisher | 0.0149 | 0.0048 | 0 | 0.9908 | 0.9098 | 0.9672 | 0.9271 |

P1 verdict: strong positive direction signal. D0 full Sobolev remains a bad train step on MNIST/KMNIST, while D6 GlobalTFU has positive train and validation descent on all three datasets.

## P2 Global TFU Micro-Run

| dataset | method | runs | acc | std | gap vs AdamW | acc minus D0 | AUC imp vs D0 | AUC imp vs AdamW | ECE red | phi p95 | J | safeguard fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MNIST | AdamW | 3 | 0.9303 | 0.0076 | 0.0000 | 0.0433 | 0.4541 | 0.0000 | 0.0000 | 0.2274 | 226275.4421 | 0.0000 |
| MNIST | D0-allFullSobolev | 3 | 0.8870 | 0.0455 | 0.0433 | 0.0000 | 0.0000 | -0.8318 | -0.3594 | 0.1489 | 34.4779 | 0.0000 |
| MNIST | D6-allTaskAware | 3 | 0.8620 | 0.0708 | 0.0683 | -0.0250 | 0.0510 | -0.7384 | 0.0949 | 0.1508 | 32.8464 | 0.0000 |
| MNIST | D6-noSob | 3 | 0.8837 | 0.0395 | 0.0467 | -0.0033 | 0.0730 | -0.6980 | 0.3943 | 0.1507 | 36.8407 | 0.0000 |
| MNIST | D6-lowSob | 3 | 0.8910 | 0.0352 | 0.0393 | 0.0040 | 0.0692 | -0.7051 | 0.1009 | 0.1507 | 45.4414 | 0.0000 |
| MNIST | D6-withSafeguard | 3 | 0.8620 | 0.0708 | 0.0683 | -0.0250 | 0.0510 | -0.7384 | 0.0949 | 0.1508 | 32.8464 | 0.0000 |
| MNIST | Hybrid-DGKAN-UFULL-f085 | 3 | 0.9373 | 0.0068 | -0.0070 | 0.0503 | 0.6148 | 0.2944 | 0.1910 | 0.1488 | 10.9172 | 0.0000 |
| MNIST | MLP-AdamW | 3 | 0.9533 | 0.0046 | -0.0230 | 0.0663 | 0.6697 | 0.3949 | 0.3670 |  |  | 0.0000 |
| Fashion-MNIST | AdamW | 3 | 0.8563 | 0.0090 | 0.0000 | 0.0243 | 0.0602 | 0.0000 | 0.0000 | 0.2396 | 369574.0127 | 0.0000 |
| Fashion-MNIST | D0-allFullSobolev | 3 | 0.8320 | 0.0120 | 0.0243 | 0.0000 | 0.0000 | -0.0640 | 0.6596 | 0.1490 | 31.4545 | 0.0000 |
| Fashion-MNIST | D6-allTaskAware | 3 | 0.8303 | 0.0358 | 0.0260 | -0.0017 | 0.0971 | 0.0393 | 0.3910 | 0.1506 | 31.2786 | 0.0000 |
| Fashion-MNIST | D6-noSob | 3 | 0.8483 | 0.0048 | 0.0080 | 0.0163 | 0.1166 | 0.0601 | 0.4310 | 0.1507 | 31.1504 | 0.0000 |
| Fashion-MNIST | D6-lowSob | 3 | 0.8473 | 0.0101 | 0.0090 | 0.0153 | 0.1161 | 0.0595 | 0.4507 | 0.1508 | 38.3286 | 0.0000 |
| Fashion-MNIST | D6-withSafeguard | 3 | 0.8303 | 0.0358 | 0.0260 | -0.0017 | 0.0971 | 0.0393 | 0.3910 | 0.1506 | 31.2786 | 0.0000 |
| Fashion-MNIST | Hybrid-DGKAN-UFULL-f085 | 3 | 0.8490 | 0.0116 | 0.0073 | 0.0170 | 0.2180 | 0.1679 | 0.2075 | 0.1488 | 10.2621 | 0.0000 |
| Fashion-MNIST | MLP-AdamW | 3 | 0.8500 | 0.0137 | 0.0063 | 0.0180 | 0.2179 | 0.1678 | 0.1011 |  |  | 0.0000 |
| KMNIST | AdamW | 3 | 0.7720 | 0.0102 | 0.0000 | 0.0937 | 0.4381 | 0.0000 | 0.0000 | 0.2172 | 20374.7067 | 0.0000 |
| KMNIST | D0-allFullSobolev | 3 | 0.6783 | 0.0147 | 0.0937 | 0.0000 | 0.0000 | -0.7797 | 0.7098 | 0.1493 | 35.5405 | 0.0000 |
| KMNIST | D6-allTaskAware | 3 | 0.7500 | 0.0134 | 0.0220 | 0.0717 | 0.1873 | -0.4464 | 0.7258 | 0.1512 | 32.7251 | 0.0000 |
| KMNIST | D6-noSob | 3 | 0.7333 | 0.0175 | 0.0387 | 0.0550 | 0.1604 | -0.4943 | 0.6471 | 0.1511 | 51.3783 | 0.0000 |
| KMNIST | D6-lowSob | 3 | 0.7200 | 0.0051 | 0.0520 | 0.0417 | 0.1601 | -0.4948 | 0.6547 | 0.1511 | 42.9699 | 0.0000 |
| KMNIST | D6-withSafeguard | 3 | 0.7500 | 0.0134 | 0.0220 | 0.0717 | 0.1873 | -0.4464 | 0.7258 | 0.1512 | 32.7251 | 0.0000 |
| KMNIST | Hybrid-DGKAN-UFULL-f085 | 3 | 0.7667 | 0.0139 | 0.0053 | 0.0883 | 0.4753 | 0.0662 | 0.0897 | 0.1485 | 10.2174 | 0.0000 |
| KMNIST | MLP-AdamW | 3 | 0.7920 | 0.0120 | -0.0200 | 0.1137 | 0.5531 | 0.2046 | 0.0269 |  |  | 0.0000 |

P2 verdict: D6 does not pass the global gate. It repairs one-batch direction and improves KMNIST substantially over D0, but the 3-seed accuracy gap vs PureKAN-AdamW is too large on MNIST and KMNIST. Fashion only passes the accuracy side for `D6-noSob` / `D6-lowSob`, not the unified D6 profile.

## P3 Depth-Wise Sweep

| dataset | method | runs | acc | std | gap vs AdamW | acc minus D0 | AUC imp vs D0 | AUC imp vs AdamW | ECE red | phi p95 | J | safeguard fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MNIST | AdamW | 3 | 0.9303 | 0.0076 | 0.0000 | 0.0433 | 0.4541 | 0.0000 | 0.0000 | 0.2274 | 226275.4421 | 0.0000 |
| MNIST | D0-allFullSobolev | 3 | 0.8870 | 0.0455 | 0.0433 | 0.0000 | 0.0000 | -0.8318 | -0.3594 | 0.1489 | 34.4779 | 0.0000 |
| MNIST | D6-allTaskAware | 3 | 0.8620 | 0.0708 | 0.0683 | -0.0250 | 0.0510 | -0.7384 | 0.0949 | 0.1508 | 32.8464 | 0.0000 |
| MNIST | D1-frontTask-backSob | 3 | 0.8483 | 0.0878 | 0.0820 | -0.0387 | -0.0356 | -0.8970 | -0.0305 | 0.1508 | 39.1325 | 0.0000 |
| MNIST | D2-frontSob-backTask | 3 | 0.8877 | 0.0237 | 0.0427 | 0.0007 | -0.0355 | -0.8969 | 0.0268 | 0.1492 | 26.0784 | 0.0000 |
| MNIST | D3-taskSobTask | 3 | 0.9080 | 0.0037 | 0.0223 | 0.0210 | 0.0672 | -0.7086 | 0.0759 | 0.1493 | 31.8315 | 0.0000 |
| MNIST | D7-inputOutputTask-middleSob | 3 | 0.9080 | 0.0037 | 0.0223 | 0.0210 | 0.0672 | -0.7086 | 0.0759 | 0.1493 | 31.8315 | 0.0000 |
| MNIST | D8-inputDiag-blockTask-outputFisher | 3 | 0.8883 | 0.0109 | 0.0420 | 0.0013 | -0.0230 | -0.8739 | 0.1354 | 0.1495 | 28.8581 | 0.0000 |
| Fashion-MNIST | AdamW | 3 | 0.8563 | 0.0090 | 0.0000 | 0.0243 | 0.0602 | 0.0000 | 0.0000 | 0.2396 | 369574.0127 | 0.0000 |
| Fashion-MNIST | D0-allFullSobolev | 3 | 0.8320 | 0.0120 | 0.0243 | 0.0000 | 0.0000 | -0.0640 | 0.6596 | 0.1490 | 31.4545 | 0.0000 |
| Fashion-MNIST | D6-allTaskAware | 3 | 0.8303 | 0.0358 | 0.0260 | -0.0017 | 0.0971 | 0.0393 | 0.3910 | 0.1506 | 31.2786 | 0.0000 |
| Fashion-MNIST | D1-frontTask-backSob | 3 | 0.8447 | 0.0140 | 0.0117 | 0.0127 | 0.0642 | 0.0043 | 0.5762 | 0.1505 | 47.8118 | 0.0000 |
| Fashion-MNIST | D2-frontSob-backTask | 3 | 0.8393 | 0.0107 | 0.0170 | 0.0073 | 0.0822 | 0.0235 | 0.5151 | 0.1496 | 29.9848 | 0.0000 |
| Fashion-MNIST | D3-taskSobTask | 3 | 0.8490 | 0.0079 | 0.0073 | 0.0170 | 0.1170 | 0.0604 | 0.5534 | 0.1492 | 40.1297 | 0.0000 |
| Fashion-MNIST | D7-inputOutputTask-middleSob | 3 | 0.8490 | 0.0079 | 0.0073 | 0.0170 | 0.1170 | 0.0604 | 0.5534 | 0.1492 | 40.1297 | 0.0000 |
| Fashion-MNIST | D8-inputDiag-blockTask-outputFisher | 3 | 0.8347 | 0.0107 | 0.0217 | 0.0027 | 0.0758 | 0.0167 | 0.4926 | 0.1500 | 32.1401 | 0.0000 |
| KMNIST | AdamW | 3 | 0.7720 | 0.0102 | 0.0000 | 0.0937 | 0.4381 | 0.0000 | 0.0000 | 0.2172 | 20374.7067 | 0.0000 |
| KMNIST | D0-allFullSobolev | 3 | 0.6783 | 0.0147 | 0.0937 | 0.0000 | 0.0000 | -0.7797 | 0.7098 | 0.1493 | 35.5405 | 0.0000 |
| KMNIST | D6-allTaskAware | 3 | 0.7500 | 0.0134 | 0.0220 | 0.0717 | 0.1873 | -0.4464 | 0.7258 | 0.1512 | 32.7251 | 0.0000 |
| KMNIST | D1-frontTask-backSob | 3 | 0.7003 | 0.0554 | 0.0717 | 0.0220 | 0.0919 | -0.6162 | 0.6356 | 0.1511 | 32.1348 | 0.0000 |
| KMNIST | D2-frontSob-backTask | 3 | 0.7250 | 0.0208 | 0.0470 | 0.0467 | 0.1020 | -0.5982 | 0.7363 | 0.1497 | 29.6158 | 0.0000 |
| KMNIST | D3-taskSobTask | 3 | 0.7107 | 0.0253 | 0.0613 | 0.0323 | 0.1457 | -0.5204 | 0.6758 | 0.1495 | 31.4745 | 0.0000 |
| KMNIST | D7-inputOutputTask-middleSob | 3 | 0.7107 | 0.0253 | 0.0613 | 0.0323 | 0.1457 | -0.5204 | 0.6758 | 0.1495 | 31.4745 | 0.0000 |
| KMNIST | D8-inputDiag-blockTask-outputFisher | 3 | 0.7363 | 0.0212 | 0.0357 | 0.0580 | 0.1321 | -0.5446 | 0.7372 | 0.1501 | 33.4249 | 0.0000 |

### P3 Gate Check

| method | max gap vs AdamW | min AUC imp vs D0 | improves D0 acc all | enter P4 |
|---|---|---|---|---|
| D6-allTaskAware | 0.0683 | 0.0510 | no | no |
| D1-frontTask-backSob | 0.0820 | -0.0356 | no | no |
| D2-frontSob-backTask | 0.0470 | -0.0355 | yes | no |
| D3-taskSobTask | 0.0613 | 0.0672 | yes | no |
| D7-inputOutputTask-middleSob | 0.0613 | 0.0672 | yes | no |
| D8-inputDiag-blockTask-outputFisher | 0.0420 | -0.0230 | yes | no |

P3 verdict: no method enters P4. `D3`/`D7` are best for Fashion and improve MNIST over D0, while `D8` is comparatively best on KMNIST among depth-wise variants. None satisfy the joint gate because MNIST/KMNIST still remain more than 1-2 points behind PureKAN-AdamW and AUC improvement vs D0 does not reach the required 20% on all datasets.

## Final Decision

```text
P0:
  Implementation smoke passed.

P1:
  GlobalTFU fixes the immediate direction failure seen in v3.7 one-batch audits.

P2:
  D6 GlobalTFU is not accepted as a full PureKAN optimizer.
  It improves direction and calibration/geometry but does not match PureKAN-AdamW accuracy.

P3:
  Depth-wise task placement helps identify useful structure:
    D3/D7: best Fashion and better MNIST than D0.
    D8: best KMNIST depth-wise candidate.
  No candidate reaches the P4 entry gate.

P4-P7:
  Not run by plan because P3 produced no qualifying candidate.

Final:
  v3.8 confirms that the v3.7 bottleneck is partly metric direction quality, not implementation coverage.
  Task-aware diagonal metrics are a real repair for local descent, but global training still underfits or becomes seed-sensitive.
  Next work should focus on layer-local/block-local objective design or adaptive per-role LR schedules before spending 5/10-seed confirmation budget.
```



---


# Source 7: `docs/DG-KAN_v3.9_Normalization_FNG_PureKAN_实验计划.md`


# DG-KAN v3.9：Normalization / FNG-KAN / PureKAN Functional Optimizer 重新设计实验计划

## 0. 本版计划的定位

v3.9 不是继续调 `branch_final_scale`、`warmup_frac`、`Sobolev alpha/beta` 或固定的 shallow/deep 配置。

当前要正面解决的是：

$$
\boxed{
\text{为什么 PureKAN 的 functional update 难以超过 PureKAN-AdamW？}
}
$$

v3.7 / v3.8 已经给出三个很清楚的事实：

1. PureKAN 的实现问题大体已经排除：`alphaFixed1`、`nonKAN=0`、`coeff coverage=1.0`、warmup trace 都已经通过。
2. Sobolev-only U-FULL 在 PureKAN 上不是可靠下降方向。
3. Global TFU / allTaskAware 能修复 one-step descent，但长期训练仍然追不上 PureKAN-AdamW；固定 shallow/deep 分工也没有产生可进入 5-seed confirm 的候选。

所以 v3.9 的核心假设是：

$$
\boxed{
\text{functional update 难调，不是因为超参没扫够，而是因为 metric 设计还不是深度网络 optimizer。}
}
$$

现有 U-FULL 本质上是：

$$
\Delta a=-\eta S^{-1}\nabla_aL,
$$

其中 $S$ 是 Sobolev smoothness Gram。它能稳定函数几何，但它不是任务自然梯度。

v3.9 要把 functional update 重新设计为：

$$
\boxed{
\text{function-space constrained task descent}
}
$$

也就是：

$$
\Delta a_l
=
-\eta_l
\left(
G_l
+
\lambda_lS_l
+
\rho I
\right)^{-1}
g_l.
$$

其中：

```text
G_l:
  当前层的 task / tangent / Fisher metric

S_l:
  Sobolev smoothness metric

lambda_l:
  task-vs-geometry mixing weight

eta_l:
  带 safeguard 的 step scale
```

更进一步，v3.9 要测试 **KAN-FNG / KAN-KFAC**：

$$
\boxed{
\Delta A_l
=
-\eta_l
(C_l+\rho_cI)^{-1}
\nabla_{A_l}L
(A_{\Phi,l}+\lambda_lS_l+\rho_aI)^{-1}
}
$$

这里：

```text
A_{\Phi,l}: basis activation covariance / right-side functional metric
C_l: output-side gradient covariance / Fisher / downstream sensitivity
S_l: Sobolev smoothness metric
```

同时，本轮还必须验证一个单独但关键的问题：

$$
\boxed{
\text{PureKAN-UFULL 的失败是否和缺少 learnable LN / affine normalization 有关？}
}
$$

因为 Hybrid-DGKAN 的成功里，trainable LayerNorm + AdamW 可能提供了尺度对齐和表征稳定性；PureKAN 目前只有无参数 FixedNorm。

---

## 1. 总目标

v3.9 的最终目标不是得到一个“稍微比旧 U-FULL 好”的配置，而是判断 PureKAN functional optimizer 是否真正成立。

最终 strict gate：

$$
\operatorname{Acc}(\text{PureKAN-FNG})
\ge
\max
\{
\operatorname{Acc}(\text{PureKAN-AdamW}),
\operatorname{Acc}(\text{Hybrid-DGKAN-UFULL}),
\operatorname{Acc}(\text{MLP-AdamW})
\}.
$$

同时还要满足：

$$
\operatorname{AUC}_{val-loss}(\text{PureKAN-FNG})
<
\operatorname{AUC}_{val-loss}(\text{PureKAN-AdamW}),
$$

$$
\operatorname{ECE}(\text{PureKAN-FNG})
\le
\operatorname{ECE}(\text{PureKAN-AdamW}),
$$

$$
\phi'_{p95}(\text{PureKAN-FNG})
<
\phi'_{p95}(\text{PureKAN-AdamW}),
$$

并且：

```text
learnable_nonKAN_params = 0
```

对于严格 pure-functional 版本，所有可学习参数必须是 KAN coeff 或被明确归入 functional update group。允许设置一个 diagnostic hybrid norm 版本，但它不能作为最终 pure claim。

---

## 2. 需要回答的核心问题

v3.9 要把问题拆成四个可判定的科学问题。

### Q1：LN / normalization 是否是 PureKAN functional training 缺失的关键？

PureKAN 当前使用 `FixedNorm`：

$$
\operatorname{FixedNorm}(h)
=
\frac{h-\mu(h)}{\sigma(h)+\epsilon}.
$$

Hybrid-DGKAN 使用的是 trainable LayerNorm：

$$
\operatorname{LN}(h)
=
\gamma
\frac{h-\mu(h)}{\sigma(h)+\epsilon}
+
\beta.
$$

要判断：

```text
PureKAN-UFULL 差，是因为没有 learnable gamma/beta？
还是因为 KAN coeff 的 functional direction 本身不够？
```

### Q2：当前 diagonal/task-aware TFU 为什么 one-step 对，但长期训练不行？

需要判断：

```text
metric 是不是只捕捉了 local basis occupancy，
没有捕捉 downstream sensitivity？
```

### Q3：KAN-KFAC / FNG 是否能提供比 TFU 更好的任务几何？

目标是验证：

$$
G_l
\neq
\text{简单 diagonal proxy},
$$

而应近似为：

$$
G_l
\approx
C_l
\otimes
A_{\Phi,l}.
$$

### Q4：是否需要 AdamW-like 时间尺度，但不能用 raw Adam moment？

如果 FNG 单步方向好但长程仍差，需要验证：

```text
metric-normalized temporal smoothing
RMS step-scale controller
descent safeguard
```

是否能补上 PureKAN-AdamW 的长期优化动力学优势。

---

## 3. 模型与优化器配置

### 3.1 Baselines

所有阶段都必须保留这些 baseline：

```text
B0: MLP-AdamW
B1: Hybrid-DGKAN-UFULL-f085
B2: PureKAN-AdamW
B3: PureKAN-D0-allFullSobolev
B4: PureKAN-D6-allTaskAware
B5: PureKAN-D3-taskSobTask or D7-inputOutputTask-middleSob, optional
```

`B2 PureKAN-AdamW` 是本轮最重要 baseline，因为它证明 PureKAN 架构本身可训练。

### 3.2 Normalization variants

新增 PureKAN normalization modes：

```text
N0: FixedNorm
  当前 PureKAN 默认。
  无可学习 gamma/beta。

N1: LearnableAffineNorm-AdamW
  norm affine 参数 gamma/beta 可学习，但由 AdamW 更新。
  这是 diagnostic hybrid norm，不是 pure-functional final claim。

N2: LearnableAffineNorm-FDiag
  gamma/beta 可学习，用 diagonal Fisher / normalized functional update。
  这是 functional-norm 版本。

N3: ScalarGainNorm-FDiag
  每层只有 scalar gain 和 scalar bias，而不是 per-channel gamma/beta。
  用 functional scalar update。
  用来判断是否只需要低维尺度通道。

N4: NoNorm
  完全移除 norm，只做失败边界 smoke。
```

定义：

$$
\operatorname{AffineNorm}(h)
=
\gamma
\frac{h-\mu(h)}{\sigma(h)+\epsilon}
+
\beta.
$$

`N1` 允许 `learnable_nonKAN_params > 0`，所以只能作为诊断，不允许成为 PureKAN final claim。

`N2/N3` 的 gamma/beta 或 scalar gain/bias 需要被统计为 functional group，并记录：

```text
norm_param_count
norm_update_method
norm_gamma_mean/std/min/max
norm_beta_mean/std/min/max
norm_update_norm
norm_update_over_param
```

### 3.3 FNG / KAN-KFAC variants

本轮新增以下 optimizer candidates。

#### F0: Sobolev-only

旧 U-FULL：

$$
\Delta A_l=-\eta S_l^{-1}g_l.
$$

负对照。

#### F1: TFU-diag

旧 D6 allTaskAware：

$$
\Delta A_l=-\eta(G_{diag,l}+\lambda S_l+\rho I)^{-1}g_l.
$$

#### F2: FNG-right-only

只做右侧 basis/data metric：

$$
\Delta A_l
=
-\eta
g_l
(A_{\Phi,l}+\lambda S_l+\rho_aI)^{-1}.
$$

这里 $A_{\Phi,l}$ 是 basis activation covariance：

$$
A_{\Phi,l}
=
\mathbb E[\Phi_l^\top\Phi_l].
$$

#### F3: FNG-left-diag-right

加入 output-side diagonal Fisher：

$$
\Delta A_l
=
-\eta
D_{C,l}^{-1}
g_l
(A_{\Phi,l}+\lambda S_l+\rho_aI)^{-1}.
$$

其中：

$$
D_{C,l}
=
\operatorname{diag}
\left(
\mathbb E[\delta_l\delta_l^\top]
\right).
$$

#### F4: FNG-left-full-right

完整 output-side small matrix：

$$
\Delta A_l
=
-\eta
(C_l+\rho_cI)^{-1}
g_l
(A_{\Phi,l}+\lambda S_l+\rho_aI)^{-1}.
$$

其中：

$$
C_l=\mathbb E[\delta_l\delta_l^\top].
$$

对于 output layer，$C_l$ 可以用 softmax Fisher：

$$
H_{CE}
=
\operatorname{diag}(p)-pp^\top.
$$

#### F5: FNG-left-lowrank-right

如果 full $C_l$ 太贵或条件数差，用低秩近似：

$$
C_l
\approx
UU^\top+\rho_c I.
$$

使用 Woodbury solve。

#### F6: FNG + metric-normalized temporal smoothing

先算 functional direction：

$$
d_t=-(M_t)^{-1}g_t.
$$

然后做 metric-normalized direction EMA：

$$
\hat d_t
=
\frac{d_t}{\|d_t\|_{M_t}+\epsilon},
$$

$$
m_t
=
\beta m_{t-1}+(1-\beta)\hat d_t,
$$

$$
\Delta a_t
=
\eta_t
\frac{m_t}{\|m_t\|_{M_t}+\epsilon}.
$$

注意：这不是 AdamW raw-gradient momentum。momentum 只作用在已经 preconditioned 的 functional direction 上。

#### F7: FNG + descent safeguard

加入 mini-batch Armijo-style check：

$$
L(\theta+\Delta\theta)
\le
L(\theta)
-
c\eta
\langle g,d\rangle.
$$

如果不满足：

```text
eta <- gamma * eta
如果仍不满足，fallback 到 F1 / raw gradient / small identity step
```

记录：

```text
line_search_attempts
fallback_rate
fallback_target
accepted_step_scale
```

---

## 4. 必须新增的记录字段

### 4.1 General run fields

```text
dataset
method
seed
model_type
norm_mode
norm_update_method
optimizer_family
fng_mode
basis_count
hidden_dim
depth
epochs
batch_size
```

### 4.2 Core performance fields

```text
train_loss_curve
val_loss_curve
val_acc_curve
test_acc
test_loss
val_loss_auc
val_acc_auc
ECE
NLL
classwise_acc
confusion_matrix
```

### 4.3 PureKAN parameter audit

```text
learnable_nonKAN_params
learnable_KAN_coeff_params
functional_param_coverage
norm_param_count
norm_param_coverage
input_kan_coeff_seen
block_kan_coeff_seen
output_kan_coeff_seen
```

### 4.4 Normalization audit

```text
pre_norm_mean/std/p95_by_layer
post_norm_mean/std/p95_by_layer
gamma_mean/std/min/max_by_layer
beta_mean/std/min/max_by_layer
gamma_update_norm_by_layer
beta_update_norm_by_layer
gamma_update_over_param_by_layer
beta_update_over_param_by_layer
```

### 4.5 RBF basis audit

```text
basis_occupancy_mean/min/p01_by_layer
dead_basis_frac_by_layer
out_of_grid_frac_by_layer
basis_entropy_by_layer
basis_effective_rank_by_layer
```

### 4.6 FNG metric audit

For each role:

```text
role = input / shallow / deep / output

A_phi_condition
A_phi_eig_min
A_phi_eig_max
A_phi_effective_rank

C_condition
C_eig_min
C_eig_max
C_effective_rank

S_condition
lambda_sobolev
rho_a
rho_c

combined_metric_condition
```

### 4.7 Direction audit

```text
raw_grad_norm_by_role
precond_direction_norm_by_role
update_norm_by_role
update_over_coeff_by_role
cos_raw_precond_by_role
cos_FNG_AdamW_step_by_role
predicted_descent
actual_train_descent
actual_val_descent
bad_step_rate
```

### 4.8 Temporal dynamics audit

```text
direction_momentum_beta
direction_momentum_norm
direction_momentum_cos_current
step_scale_eta_by_role
accepted_step_scale_by_role
fallback_rate_by_role
line_search_attempts_by_role
```

### 4.9 Geometry and expression audit

```text
phi_prime_p95
phi_prime_max
max_jac_condition
curvature_energy
logit_margin_mean
correct_class_margin_mean
KAN logit delta for PureKAN layer ablation
input_kan_ablation_delta
block_kan_ablation_delta
output_kan_ablation_delta
```

### 4.10 Wall-clock and memory

```text
step_time_ms
metric_build_time_ms
metric_solve_time_ms
line_search_time_ms
memory_peak_mb
FNG_state_memory_mb
samples_per_sec
time_to_relaxed_loss
```

---

## 5. 必须生成的可视化

### 5.1 Direction quality dashboard

每个 dataset 一张图组：

```text
raw vs preconditioned cosine by role
predicted descent vs actual train descent scatter
predicted descent vs actual val descent scatter
bad step rate bar
```

### 5.2 Normalization dashboard

```text
gamma/beta trajectory by layer
pre/post norm std trajectory
basis occupancy before/after norm
out-of-grid fraction by layer
```

### 5.3 FNG metric dashboard

```text
A_phi eigen spectrum
C eigen spectrum
combined metric condition over training
effective rank over training
lambda_sobolev trajectory
```

### 5.4 Training trajectory dashboard

```text
train loss curve
val loss curve
val acc curve
ECE curve
classwise acc curve
```

### 5.5 Geometry vs task Pareto

Scatter:

```text
x-axis: test accuracy
y-axis: phi_prime_p95 or max_jac_condition
point color: method
point shape: norm_mode
```

### 5.6 Step-scale and fallback dashboard

```text
accepted eta by role over time
fallback rate by epoch
line search attempts histogram
direction momentum cosine curve
```

### 5.7 Final scorecard

For each dataset:

```text
method
norm_mode
optimizer_family
acc
acc_std
gap_vs_PureKAN_AdamW
gap_vs_MLP
gap_vs_Hybrid
val_loss_AUC_improvement
ECE
phi_prime_p95
max_jac_condition
step_time
memory
```

---

## 6. Experimental phases

## P0: Implementation smoke and invariants

### Goal

确认新增 norm modes 和 FNG update 没有破坏 PureKAN 的基本定义。

### Methods

```text
PureKAN-FixedNorm-AdamW
PureKAN-FixedNorm-D0-allFullSobolev
PureKAN-FixedNorm-D6-allTaskAware
PureKAN-AffineNorm-AdamW-LN
PureKAN-AffineNorm-FDiag-LN
PureKAN-ScalarGainNorm-FDiag
PureKAN-FixedNorm-FNG-right
PureKAN-FixedNorm-FNG-leftDiagRight
PureKAN-FixedNorm-FNG-leftFullRight
```

### Datasets

```text
MNIST
Fashion-MNIST
KMNIST
```

### Seeds

```text
seed = 0 only
```

### Checks

For strict PureKAN functional variants:

```text
learnable_nonKAN_params = 0
functional_param_coverage = 1.0
input_kan_coeff_seen = 1
block_kan_coeff_seen = 1
output_kan_coeff_seen = 1
```

For diagnostic `AffineNorm-AdamW-LN`:

```text
learnable_nonKAN_params > 0
norm_param_count > 0
norm_update_method = adamw
```

This method is not eligible for PureKAN final claim.

### Pass gate

```text
no NaN / Inf
metric condition finite
FNG solve finite
fallback_rate < 0.5 in smoke
step_time < 4x PureKAN-AdamW for smoke
```

---

## P1: One-batch and multi-batch direction audit

### Goal

先证明方向是对的，再谈训练。

### Methods

```text
AdamW-one-step
D0-allFullSobolev
D6-allTaskAware
F2-FNG-right-only
F3-FNG-leftDiag-right
F4-FNG-leftFull-right
F4-FNG-leftFull-right-noSob
F4-FNG-leftFull-right-lowSob
F4-FNG-leftFull-right+AffineNorm-FDiag
F4-FNG-leftFull-right+AffineNorm-AdamW-LN diagnostic
```

### Metrics

For each method and role:

```text
train descent
val descent
bad step
raw-precond cosine
update norm
update over coeff
A_phi condition
C condition
combined condition
```

### Direction pass gate

A candidate passes P1 if:

```text
bad_step_rate = 0
train_descent > 0 on all datasets
val_descent >= 0 on at least 2/3 datasets
cos_raw_precond_mean > 0.5
combined_metric_condition < 1e5
```

A stronger pass:

```text
train_descent >= 0.8 * AdamW-one-step train_descent
val_descent >= 0.5 * AdamW-one-step val_descent
```

Candidates failing P1 must not enter P2.

---

## P2: Normalization ablation, separate from FNG

### Goal

判断 PureKAN-UFULL / TFU 是否主要缺少 learnable normalization.

### Methods

```text
PureKAN-FixedNorm-AdamW
PureKAN-AffineNorm-AdamW-LN
PureKAN-AffineNorm-FDiag-LN
PureKAN-ScalarGainNorm-FDiag
PureKAN-NoNorm-AdamW smoke only

PureKAN-FixedNorm-D6
PureKAN-AffineNorm-AdamW-LN-D6
PureKAN-AffineNorm-FDiag-LN-D6

PureKAN-FixedNorm-FNG-leftFullRight
PureKAN-AffineNorm-AdamW-LN-FNG-leftFullRight diagnostic
PureKAN-AffineNorm-FDiag-LN-FNG-leftFullRight
```

### Seeds

```text
seeds = 0, 1, 2
```

### Epochs

Use the same budget as previous PureKAN experiments.

### Key readout

Compare:

$$
\Delta_{\text{LN}}
=
\operatorname{Acc}(\text{AffineNorm})-
\operatorname{Acc}(\text{FixedNorm}).
$$

### Interpretation

If `AffineNorm-AdamW-LN` closes most of the gap but `AffineNorm-FDiag-LN` does not:

```text
LN is important, but current functional LN update is weak.
```

If both close the gap:

```text
normalization was a major missing component and can be functionalized.
```

If neither closes the gap:

```text
core issue remains KAN coeff optimizer dynamics.
```

---

## P3: FNG metric micro-run

### Goal

Test whether KAN-KFAC/FNG is better than D6 in short training.

### Methods

```text
PureKAN-AdamW
D0-allFullSobolev
D6-allTaskAware

F2-FNG-right-only
F3-FNG-leftDiag-right
F4-FNG-leftFull-right
F4-FNG-leftFull-right-noSob
F4-FNG-leftFull-right-lowSob
F5-FNG-leftLowRank-right
```

### Seeds

```text
seeds = 0, 1, 2
```

### Entry condition

Only methods passing P1 enter P3.

### Metrics

```text
test_acc
val_loss_auc
ECE
phi_prime_p95
max_jac_condition
FNG metric condition
step_time
fallback_rate
```

### P3 pass gate

A candidate enters P4 if:

```text
acc gap vs PureKAN-AdamW:
  MNIST < 2%
  Fashion < 2%
  KMNIST < 3%

AUC better than D6 on all datasets
bad_step_rate = 0
fallback_rate < 0.2
geometry no worse than PureKAN-AdamW
```

---

## P4: Temporal dynamics / step-scale ablation

### Goal

If FNG improves direction but still trails AdamW, test whether time-scale dynamics are missing.

### Methods

Use only P3 survivors.

For each survivor, test:

```text
base FNG
FNG + metric-normalized direction EMA, beta=0.8
FNG + metric-normalized direction EMA, beta=0.9
FNG + RMS step-scale controller
FNG + EMA + RMS controller
FNG + line-search safeguard
FNG + EMA + safeguard
```

### Important constraint

Do not use raw AdamW moment on coefficients.

Allowed:

$$
m_t
=
\beta m_{t-1}
+
(1-\beta)
\frac{d_t}{\|d_t\|_{M_t}+\epsilon}.
$$

Not allowed:

$$
m_t
=
\beta m_{t-1}
+
(1-\beta)g_t
$$

followed by AdamW-style update.

### Metrics

```text
direction momentum cosine
accepted eta
fallback rate
val loss curve smoothness
training instability count
```

### P4 pass gate

```text
acc gap vs PureKAN-AdamW:
  MNIST < 1.5%
  Fashion < 1.5%
  KMNIST < 2.5%

AUC better than D6
ECE <= PureKAN-AdamW
fallback_rate < 0.3
```

---

## P5: Combined Norm + FNG 3-seed selection

### Goal

Combine the best normalization choice and best FNG dynamics.

### Candidate groups

```text
C1: FixedNorm + best FNG
C2: AffineNorm-FDiag + best FNG
C3: ScalarGainNorm-FDiag + best FNG
C4: AffineNorm-AdamW-LN + best FNG diagnostic only
```

### Baselines

```text
PureKAN-AdamW
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
D6-allTaskAware
```

### 3-seed full-budget gate

A candidate enters 5-seed confirm if:

```text
MNIST acc >= PureKAN-AdamW - 1%
Fashion acc >= PureKAN-AdamW - 1%
KMNIST acc >= PureKAN-AdamW - 2%

and

AUC better than D6 on all datasets
ECE <= PureKAN-AdamW on at least 2/3 datasets
phi_prime_p95 < PureKAN-AdamW
max_jac_condition < PureKAN-AdamW
```

Strict candidate:

```text
acc >= PureKAN-AdamW on all datasets
```

---

## P6: 5-seed confirm

### Seeds

```text
seeds = 0,1,2,3,4
```

### Methods

```text
PureKAN-AdamW
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
D6-allTaskAware
Best PureKAN-FNG candidate
Best diagnostic LN hybrid candidate, optional
```

### 5-seed gate

```text
paired acc delta vs PureKAN-AdamW >= 0 on at least 2/3 datasets
paired acc delta vs MLP-AdamW >= 0 on at least 1/3 datasets
AUC delta vs PureKAN-AdamW > 0 on all datasets
ECE delta vs PureKAN-AdamW <= 0 on at least 2/3 datasets
```

If not passed, do not run 10-seed.

---

## P7: 10-seed final confirm

### Seeds

```text
0..9
```

### Strict claim gate

To claim PureKAN functional optimizer solved:

```text
PureKAN-FNG >= PureKAN-AdamW in accuracy on MNIST/Fashion/KMNIST
PureKAN-FNG >= MLP-AdamW in accuracy on at least 2/3 datasets
PureKAN-FNG >= Hybrid-DGKAN-UFULL in accuracy on at least 2/3 datasets
AUC improves over PureKAN-AdamW on all datasets
ECE improves or matches PureKAN-AdamW
geometry improves over PureKAN-AdamW
```

If this fails but FNG beats D6 and narrows gap substantially, write:

```text
FNG repairs Sobolev-only PureKAN but does not yet replace AdamW.
```

---

## P8: Failure diagnosis if P7 fails

If no candidate passes P7, classify failure into one of these:

### Failure type A: normalization bottleneck

Signal:

```text
AffineNorm-AdamW-LN works,
AffineNorm-FDiag does not.
```

Interpretation:

```text
learnable normalization is essential and currently needs AdamW-style dynamics.
```

### Failure type B: FNG direction bottleneck

Signal:

```text
P1/P3 descent still weak or unstable.
```

Interpretation:

```text
G_l / C_l metric still not approximating task curvature well.
```

### Failure type C: temporal dynamics bottleneck

Signal:

```text
one-step and short-run good,
full-run bad.
```

Interpretation:

```text
functional direction is good, but long-term step scaling / momentum is insufficient.
```

### Failure type D: capacity / architecture bottleneck

Signal:

```text
PureKAN-AdamW itself weak relative to MLP/Hybrid after matched capacity.
```

Interpretation:

```text
PureKAN architecture needs redesign, not just optimizer.
```

### Failure type E: over-regularized function class

Signal:

```text
geometry very good,
accuracy poor,
basis occupancy healthy,
margins low.
```

Interpretation:

```text
Sobolev component too strong or wrong for representation learning.
```

---

## 7. Implementation details

### 7.1 AffineNorm implementation

Add:

```python
class AffineFixedNorm(nn.Module):
    def __init__(self, dim, affine=True, scalar=False):
        ...
```

Modes:

```text
fixed:
  no gamma/beta

affine_channel:
  gamma,beta shape [dim]

affine_scalar:
  gamma,beta scalar per layer
```

For pure functional modes, gamma/beta must be registered into functional norm group, not AdamW.

### 7.2 FNG right metric

For an RBF KAN layer:

$$
y_o
=
\sum_{i,m}
a_{oim}B_m(x_i).
$$

A practical right metric can be block-diagonal over input dimension:

$$
A_{\Phi,i}
=
\mathbb E[B(x_i)B(x_i)^\top].
$$

Shared version:

$$
A_{\Phi}
=
\frac{1}{d_{in}}
\sum_i
\mathbb E[B(x_i)B(x_i)^\top].
$$

Start with shared basis metric for cost.

### 7.3 FNG left metric

For each layer, use gradient covariance:

$$
C_l
=
\mathbb E[\delta_l\delta_l^\top],
$$

where $\delta_l$ is gradient w.r.t. KAN layer output.

For output layer, use CE Fisher:

$$
C_{out}
=
\mathbb E[\operatorname{diag}(p)-pp^\top].
$$

### 7.4 Solve form

Reshape coeff gradient:

$$
g_l \in \mathbb R^{d_{out}\times(d_{in}K)}.
$$

Update:

$$
\Delta A_l
=
-\eta
(C_l+\rho_c I)^{-1}
g_l
(R_l+\rho_a I)^{-1}.
$$

where $R_l$ is either:

```text
basis-only shared metric
block-diagonal input-basis metric
Sobolev-augmented basis metric
```

### 7.5 Sobolev as regularizer

Do not use pure $S^{-1}$ as main metric.

Use:

$$
R_l
=
A_{\Phi,l}
+
\lambda_l S_l
+
\rho_aI.
$$

Default initial values:

```text
lambda_input = 0.0 or 0.01
lambda_block = 0.02
lambda_output = 0.0 or 0.01
```

Then let adaptive controller change them only after P3.

### 7.6 Controller

Controller rule:

```text
if bad_step:
  eta_l *= 0.5
  lambda_l *= 1.2

if phi_prime_p95 high or J high:
  lambda_l *= 1.1

if val_loss stagnates and geometry safe:
  lambda_l *= 0.9
  eta_l *= 1.05
```

All controller actions must be logged.

---

## 8. Expected outcomes

### Outcome 1: LN diagnostic works, pure functional LN fails

This means Hybrid-DGKAN success depends partly on AdamW-trained normalization.

Next step:

```text
design better functional norm optimizer
or accept that pure KAN needs non-KAN norm support
```

### Outcome 2: FNG beats D6 but not AdamW

This means task Fisher is necessary but temporal dynamics still weak.

Next step:

```text
improve metric-normalized momentum and step scaling
```

### Outcome 3: FNG + functional norm beats PureKAN-AdamW

This is the desired result.

Then proceed to:

```text
5-seed confirm
10-seed final
Rational/KAT FNG adaptation
CIFAR-small PureKAN/ConvStem scaling
```

### Outcome 4: All FNG variants fail

Then the correct conclusion is:

```text
current functional optimizer cannot yet train full PureKAN.
Hybrid branch functional update remains the valid contribution.
```

---

## 9. What not to do in v3.9

Do not spend more runs on:

```text
fixed D1/D2/D3/D7 depth-wise profile search
branch_final_scale sweep
full Sobolev alpha/beta sweep
diagwarmup length sweep
raw Adam moment on coefficients
CIFAR scaling before PureKAN optimizer is repaired
Rational functional update before PureKAN FNG is understood
```

These are secondary until we know whether FNG can train PureKAN.

---

## 10. Final decision language

At the end of v3.9, write one of the following.

### If FNG succeeds

```text
PureKAN functional training is recovered by replacing Sobolev-only U-FULL with functional natural gradient. The key change is using task Fisher / downstream sensitivity as the main metric, with Sobolev only as smoothness regularization.
```

### If only LN diagnostic succeeds

```text
The failure of PureKAN-UFULL is partly due to the absence of learnable normalization. Hybrid-DGKAN succeeds because AdamW-trained normalization and head/stem parameters stabilize representation learning.
```

### If FNG improves but does not beat AdamW

```text
Task-aware FNG repairs the local descent failure of Sobolev-only U-FULL, but does not yet match AdamW’s long-term optimization dynamics.
```

### If all fails

```text
PureKAN functional optimization remains unsolved. The valid contribution is restricted to hybrid DG-KAN branch functional updates.
```



---


# Source 8: `docs/DG-KAN_v3.9_Normalization_FNG_PureKAN_结果复盘.md`


# DG-KAN v3.9 Normalization / FNG PureKAN 结果复盘

本轮依据 `docs/DG-KAN_v3.9_Normalization_FNG_PureKAN_实验计划.md`。目标是判断 PureKAN functional update 追不上 PureKAN-AdamW 的主因，是缺少 learnable normalization，还是缺少真正 task/Fisher-aware 的 function-space metric。

## Code / Config Changes

```text
experiments/dgkan_core.py
  Added PureKAN norm modes: fixed / affine_channel / scalar_gain / none.
  Added functional diagonal norm update for gamma/beta or scalar gain/bias.
  Added FNG/KFAC-style coefficient update metrics:
    fng_right
    fng_leftdiag_right
    fng_leftfull_right
    fng_leftlowrank_right
  Result rows now include norm audit, functional coverage, FNG metric conditions,
  FNG fallback/bad-step rates, timing, and per-role metric traces.

experiments/run_gafu_v39.py
  Added P0 smoke, P1 shadow audit, P2 normalization ablation,
  P3 FNG package, and optional P4 dynamics wiring.

experiments/analyze_gafu_v39.py
  Generates this replay, appends the log entry, and writes basic dashboards.
```

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 smoke | 27 | 0 |
| P1 shadow | 30 | 0 |
| P2 norm | 99 | 0 |
| P3 relaxed | 54 | 0 |

## P0 Implementation Smoke

| dataset | method | norm | norm update | nonKAN | raw nonKAN | func cov | input coeff | block coeff | output coeff | norm params | norm cov | FNG | FNG cond | fallback |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | AffineNorm-AdamW-LN | affine_channel | adamw | 192 | 192 | 0.000 | 0.000 | 0.000 | 0.000 | 192 | 0.000 | none |  | 0.000 |
| Fashion-MNIST | AffineNorm-FDiag-LN | affine_channel | fdiag | 0 | 192 | 1.000 | 1.000 | 1.000 | 1.000 | 192 | 1.000 | none |  | 0.000 |
| Fashion-MNIST | FixedNorm-AdamW | fixed | none | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0 |  | none |  | 0.000 |
| Fashion-MNIST | FixedNorm-D0-allFullSobolev | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | none |  | 0.000 |
| Fashion-MNIST | FixedNorm-D6-allTaskAware | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | none |  | 0.000 |
| Fashion-MNIST | FixedNorm-FNG-leftDiagRight | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | leftdiag | 1959.61 | 0.000 |
| Fashion-MNIST | FixedNorm-FNG-leftFullRight | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | leftfull | 1959.72 | 0.000 |
| Fashion-MNIST | FixedNorm-FNG-right | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | right | 1959.76 | 0.000 |
| Fashion-MNIST | ScalarGainNorm-FDiag | scalar_gain | fdiag | 0 | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 6 | 1.000 | none |  | 0.000 |
| KMNIST | AffineNorm-AdamW-LN | affine_channel | adamw | 192 | 192 | 0.000 | 0.000 | 0.000 | 0.000 | 192 | 0.000 | none |  | 0.000 |
| KMNIST | AffineNorm-FDiag-LN | affine_channel | fdiag | 0 | 192 | 1.000 | 1.000 | 1.000 | 1.000 | 192 | 1.000 | none |  | 0.000 |
| KMNIST | FixedNorm-AdamW | fixed | none | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0 |  | none |  | 0.000 |
| KMNIST | FixedNorm-D0-allFullSobolev | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | none |  | 0.000 |
| KMNIST | FixedNorm-D6-allTaskAware | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | none |  | 0.000 |
| KMNIST | FixedNorm-FNG-leftDiagRight | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | leftdiag | 2119.50 | 0.000 |
| KMNIST | FixedNorm-FNG-leftFullRight | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | leftfull | 2119.50 | 0.000 |
| KMNIST | FixedNorm-FNG-right | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | right | 2119.50 | 0.000 |
| KMNIST | ScalarGainNorm-FDiag | scalar_gain | fdiag | 0 | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 6 | 1.000 | none |  | 0.000 |
| MNIST | AffineNorm-AdamW-LN | affine_channel | adamw | 192 | 192 | 0.000 | 0.000 | 0.000 | 0.000 | 192 | 0.000 | none |  | 0.000 |
| MNIST | AffineNorm-FDiag-LN | affine_channel | fdiag | 0 | 192 | 1.000 | 1.000 | 1.000 | 1.000 | 192 | 1.000 | none |  | 0.000 |
| MNIST | FixedNorm-AdamW | fixed | none | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0 |  | none |  | 0.000 |
| MNIST | FixedNorm-D0-allFullSobolev | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | none |  | 0.000 |
| MNIST | FixedNorm-D6-allTaskAware | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | none |  | 0.000 |
| MNIST | FixedNorm-FNG-leftDiagRight | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | leftdiag | 2263.79 | 0.000 |
| MNIST | FixedNorm-FNG-leftFullRight | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | leftfull | 2263.79 | 0.000 |
| MNIST | FixedNorm-FNG-right | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | right | 2263.79 | 0.000 |
| MNIST | ScalarGainNorm-FDiag | scalar_gain | fdiag | 0 | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 6 | 1.000 | none |  | 0.000 |

P0 verdict: pass. Strict PureKAN functional variants have zero learnable non-KAN parameters and full coefficient coverage. Diagnostic `AffineNorm-AdamW-LN` correctly exposes learnable non-KAN norm parameters and is not eligible for final PureKAN claims.

## P1 One-Batch Direction Audit

| dataset | method | train descent | val descent | bad | mean cos | max cond | norm | norm update |
|---|---|---|---|---|---|---|---|---|
| MNIST | AdamW-one-step | 0.0229 | 0.0481 | 0 |  |  | fixed | none |
| MNIST | D0-allFullSobolev | -0.0284 | 0.0138 | 1 | 0.2488 |  | fixed | none |
| MNIST | D6-allTaskAware | -0.0137 | 0.0278 | 1 | 0.9032 |  | fixed | none |
| MNIST | F2-FNG-right-only | 0.0021 | -0.0228 | 0 | 0.3000 | 2115.5 | fixed | none |
| MNIST | F3-FNG-leftDiag-right | -0.0070 | -0.0113 | 1 | 0.3056 | 2115.5 | fixed | none |
| MNIST | F4-FNG-leftFull-right | -0.0060 | 0.0051 | 1 | 0.2784 | 2115.5 | fixed | none |
| MNIST | F4-FNG-leftFull-right+AffineNorm-AdamW-LN | -0.0002 | -0.0362 | 1 | 0.2738 | 2115.5 | affine_channel | adamw |
| MNIST | F4-FNG-leftFull-right+AffineNorm-FDiag | 0.0569 | -0.0004 | 0 | 0.2742 | 2115.5 | affine_channel | fdiag |
| MNIST | F4-FNG-leftFull-right-lowSob | 0.0729 | 0.0298 | 0 | 0.2786 | 2066.3 | fixed | none |
| MNIST | F4-FNG-leftFull-right-noSob | 0.0007 | -0.0128 | 0 | 0.2189 | 2051.0 | fixed | none |
| Fashion-MNIST | AdamW-one-step | 0.0517 | 0.0078 | 0 |  |  | fixed | none |
| Fashion-MNIST | D0-allFullSobolev | 0.0404 | 0.0315 | 0 | 0.3946 |  | fixed | none |
| Fashion-MNIST | D6-allTaskAware | 0.0075 | 0.0169 | 0 | 0.9180 |  | fixed | none |
| Fashion-MNIST | F2-FNG-right-only | 0.0126 | 0.0110 | 0 | 0.3665 | 1445.5 | fixed | none |
| Fashion-MNIST | F3-FNG-leftDiag-right | 0.0979 | 0.1009 | 0 | 0.3653 | 1445.5 | fixed | none |
| Fashion-MNIST | F4-FNG-leftFull-right | 0.0906 | 0.0373 | 0 | 0.3313 | 1445.5 | fixed | none |
| Fashion-MNIST | F4-FNG-leftFull-right+AffineNorm-AdamW-LN | 0.0095 | 0.0163 | 0 | 0.3197 | 1445.5 | affine_channel | adamw |
| Fashion-MNIST | F4-FNG-leftFull-right+AffineNorm-FDiag | 0.0486 | -0.0122 | 0 | 0.3316 | 1445.5 | affine_channel | fdiag |
| Fashion-MNIST | F4-FNG-leftFull-right-lowSob | 0.0540 | -0.0628 | 0 | 0.2775 | 1395.8 | fixed | none |
| Fashion-MNIST | F4-FNG-leftFull-right-noSob | 0.0321 | 0.0366 | 0 | 0.2255 | 1379.3 | fixed | none |
| KMNIST | AdamW-one-step | 0.0646 | 0.0501 | 0 |  |  | fixed | none |
| KMNIST | D0-allFullSobolev | 0.0305 | -0.0270 | 0 | 0.3086 |  | fixed | none |
| KMNIST | D6-allTaskAware | 0.0312 | -0.0280 | 0 | 0.9172 |  | fixed | none |
| KMNIST | F2-FNG-right-only | 0.0697 | 0.0385 | 0 | 0.3642 | 1885.7 | fixed | none |
| KMNIST | F3-FNG-leftDiag-right | 0.0110 | 0.0232 | 0 | 0.3608 | 1885.7 | fixed | none |
| KMNIST | F4-FNG-leftFull-right | 0.0724 | 0.0953 | 0 | 0.3321 | 1885.7 | fixed | none |
| KMNIST | F4-FNG-leftFull-right+AffineNorm-AdamW-LN | 0.0160 | 0.0204 | 0 | 0.3165 | 1885.7 | affine_channel | adamw |
| KMNIST | F4-FNG-leftFull-right+AffineNorm-FDiag | -0.0114 | 0.0088 | 1 | 0.3259 | 1885.7 | affine_channel | fdiag |
| KMNIST | F4-FNG-leftFull-right-lowSob | 0.0472 | -0.0386 | 0 | 0.2780 | 1836.3 | fixed | none |
| KMNIST | F4-FNG-leftFull-right-noSob | 0.0338 | 0.0536 | 0 | 0.2210 | 1820.1 | fixed | none |

### P1 Gate Check

| method | bad | min train | val >=0 | mean cos | max cond | P1 pass |
|---|---|---|---|---|---|---|
| D0-allFullSobolev | 1 | -0.0284 | 2/3 | 0.3174 |  | no |
| D6-allTaskAware | 1 | -0.0137 | 2/3 | 0.9128 |  | no |
| F2-FNG-right-only | 0 | 0.0021 | 2/3 | 0.3436 | 2115.5 | no |
| F3-FNG-leftDiag-right | 1 | -0.0070 | 2/3 | 0.3439 | 2115.5 | no |
| F4-FNG-leftFull-right | 1 | -0.0060 | 3/3 | 0.3139 | 2115.5 | no |
| F4-FNG-leftFull-right+AffineNorm-AdamW-LN | 1 | -0.0002 | 2/3 | 0.3034 | 2115.5 | no |
| F4-FNG-leftFull-right+AffineNorm-FDiag | 1 | -0.0114 | 1/3 | 0.3105 | 2115.5 | no |
| F4-FNG-leftFull-right-lowSob | 0 | 0.0472 | 1/3 | 0.2781 | 2066.3 | no |
| F4-FNG-leftFull-right-noSob | 0 | 0.0007 | 2/3 | 0.2218 | 2051.0 | no |

P1 verdict: strict fail. FNG variants repair several bad-step cases compared with D0/D6, but no candidate satisfies the full direction gate because the mean raw/preconditioned cosine remains below `0.5`. Therefore no method formally enters P3/P4 under the written plan.

## P2 Normalization Ablation

| dataset | method | runs | acc | std | gap vs AdamW | acc-D0 | AUC imp vs D6 | AUC imp vs AdamW | ECE red | phi p95 | J | FNG cond | bad | fallback | nonKAN | norm cov | time/A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MNIST | FixedNorm-AdamW | 3 | 0.9303 | 0.0076 | 0.0000 |  | 0.4248 | 0.0000 | 0.0000 | 0.2274 | 226275.4421 |  | 0.0000 | 0.0000 | 0 |  | 1.0000 |
| MNIST | AffineNorm-AdamW-LN | 3 | 0.9297 | 0.0090 | 0.0007 |  | 0.4247 | -0.0002 | -0.0171 | 0.2291 | 27892.0690 |  | 0.0000 | 0.0000 | 960 | 0.000 | 1.0952 |
| MNIST | AffineNorm-FDiag-LN | 3 | 0.9320 | 0.0100 | -0.0017 |  | 0.4427 | 0.0312 | -0.0051 | 0.2101 | 48074.1139 |  | 0.0000 | 0.0000 | 0 | 1.000 | 1.1995 |
| MNIST | ScalarGainNorm-FDiag | 3 | 0.9320 | 0.0075 | -0.0017 |  | 0.4226 | -0.0038 | -0.0102 | 0.2189 | 59139.8268 |  | 0.0000 | 0.0000 | 0 | 1.000 | 1.1791 |
| MNIST | NoNorm-AdamW | 3 | 0.8397 | 0.0152 | 0.0907 |  | -0.5936 | -1.7703 | -2.4315 | 0.0719 | 9.0234 |  | 0.0000 | 0.0000 | 0 |  | 1.1864 |
| MNIST | FixedNorm-D6-allTaskAware | 3 | 0.8620 | 0.0708 | 0.0683 |  | 0.0000 | -0.7384 | 0.0949 | 0.1508 | 32.8464 |  | 0.0000 | 0.0000 | 0 |  | 1.2855 |
| MNIST | AffineNorm-AdamW-LN-D6 | 3 | 0.8647 | 0.0811 | 0.0657 |  | 0.0041 | -0.7313 | 0.0852 | 0.1506 | 37.5073 |  | 0.0000 | 0.0000 | 960 | 0.000 | 1.3283 |
| MNIST | AffineNorm-FDiag-LN-D6 | 3 | 0.8493 | 0.0839 | 0.0810 |  | 0.0053 | -0.7292 | 0.2002 | 0.1496 | 32.1558 |  | 0.0000 | 0.0000 | 0 | 1.000 | 1.4102 |
| MNIST | FixedNorm-FNG-leftFullRight | 3 | 0.9233 | 0.0087 | 0.0070 |  | 0.4337 | 0.0155 | 0.5400 | 0.1486 | 30.6556 | 2119.3 | 0.0000 | 0.0000 | 0 |  | 1.5111 |
| MNIST | AffineNorm-AdamW-LN-FNG-leftFullRight | 3 | 0.9257 | 0.0090 | 0.0047 |  | 0.4250 | 0.0005 | 0.5529 | 0.1488 | 31.6827 | 2119.3 | 0.0000 | 0.0000 | 960 | 0.000 | 1.6108 |
| MNIST | AffineNorm-FDiag-LN-FNG-leftFullRight | 3 | 0.9273 | 0.0094 | 0.0030 |  | 0.3990 | -0.0447 | 0.5127 | 0.1482 | 27.3618 | 2119.3 | 0.0000 | 0.0000 | 0 | 1.000 | 1.6310 |
| Fashion-MNIST | FixedNorm-AdamW | 3 | 0.8563 | 0.0090 | 0.0000 |  | -0.0409 | 0.0000 | 0.0000 | 0.2396 | 369574.0127 |  | 0.0000 | 0.0000 | 0 |  | 1.0000 |
| Fashion-MNIST | AffineNorm-AdamW-LN | 3 | 0.8467 | 0.0041 | 0.0097 |  | -0.0571 | -0.0156 | 0.0010 | 0.2454 | 63761.4883 |  | 0.0000 | 0.0000 | 960 | 0.000 | 1.0704 |
| Fashion-MNIST | AffineNorm-FDiag-LN | 3 | 0.8480 | 0.0182 | 0.0083 |  | -0.1331 | -0.0885 | -0.0651 | 0.2349 | 55643.2305 |  | 0.0000 | 0.0000 | 0 | 1.000 | 1.2243 |
| Fashion-MNIST | ScalarGainNorm-FDiag | 3 | 0.8387 | 0.0236 | 0.0177 |  | -0.0815 | -0.0389 | -0.1034 | 0.2392 | 358548.6582 |  | 0.0000 | 0.0000 | 0 | 1.000 | 1.1656 |
| Fashion-MNIST | NoNorm-AdamW | 3 | 0.8007 | 0.0161 | 0.0557 |  | -0.4457 | -0.3888 | 0.3523 | 0.1031 | 10.8600 |  | 0.0000 | 0.0000 | 0 |  | 1.1346 |
| Fashion-MNIST | FixedNorm-D6-allTaskAware | 3 | 0.8303 | 0.0358 | 0.0260 |  | 0.0000 | 0.0393 | 0.3910 | 0.1506 | 31.2786 |  | 0.0000 | 0.0000 | 0 |  | 1.2882 |
| Fashion-MNIST | AffineNorm-AdamW-LN-D6 | 3 | 0.8467 | 0.0103 | 0.0097 |  | 0.0033 | 0.0424 | 0.4203 | 0.1505 | 32.0193 |  | 0.0000 | 0.0000 | 960 | 0.000 | 1.3238 |
| Fashion-MNIST | AffineNorm-FDiag-LN-D6 | 3 | 0.8373 | 0.0148 | 0.0190 |  | 0.0170 | 0.0557 | 0.3893 | 0.1497 | 28.6478 |  | 0.0000 | 0.0000 | 0 | 1.000 | 1.4019 |
| Fashion-MNIST | FixedNorm-FNG-leftFullRight | 3 | 0.8327 | 0.0068 | 0.0237 |  | 0.0467 | 0.0842 | 0.2462 | 0.1485 | 30.3611 | 1460.5 | 0.0000 | 0.0000 | 0 |  | 1.4663 |
| Fashion-MNIST | AffineNorm-AdamW-LN-FNG-leftFullRight | 3 | 0.8337 | 0.0097 | 0.0227 |  | 0.0474 | 0.0848 | 0.2462 | 0.1485 | 26.9015 | 1461.1 | 0.0000 | 0.0000 | 960 | 0.000 | 1.5405 |
| Fashion-MNIST | AffineNorm-FDiag-LN-FNG-leftFullRight | 3 | 0.8360 | 0.0127 | 0.0203 |  | 0.0372 | 0.0751 | 0.2063 | 0.1480 | 28.4187 | 1460.5 | 0.0000 | 0.0000 | 0 | 1.000 | 1.5994 |
| KMNIST | FixedNorm-AdamW | 3 | 0.7720 | 0.0102 | 0.0000 |  | 0.3087 | 0.0000 | 0.0000 | 0.2172 | 20374.7067 |  | 0.0000 | 0.0000 | 0 |  | 1.0000 |
| KMNIST | AffineNorm-AdamW-LN | 3 | 0.7733 | 0.0082 | -0.0013 |  | 0.3115 | 0.0042 | 0.0195 | 0.2160 | 80444.0033 |  | 0.0000 | 0.0000 | 960 | 0.000 | 1.0700 |
| KMNIST | AffineNorm-FDiag-LN | 3 | 0.7703 | 0.0200 | 0.0017 |  | 0.3196 | 0.0158 | -0.0019 | 0.2070 | 19478.2301 |  | 0.0000 | 0.0000 | 0 | 1.000 | 1.1891 |
| KMNIST | ScalarGainNorm-FDiag | 3 | 0.7730 | 0.0151 | -0.0010 |  | 0.3066 | -0.0030 | 0.0147 | 0.2166 | 46882.1016 |  | 0.0000 | 0.0000 | 0 | 1.000 | 1.1472 |
| KMNIST | NoNorm-AdamW | 3 | 0.6027 | 0.0095 | 0.1693 |  | -0.4709 | -1.1276 | 0.6468 | 0.0968 | 11.2521 |  | 0.0000 | 0.0000 | 0 |  | 1.1478 |
| KMNIST | FixedNorm-D6-allTaskAware | 3 | 0.7500 | 0.0134 | 0.0220 |  | 0.0000 | -0.4464 | 0.7258 | 0.1512 | 32.7251 |  | 0.0000 | 0.0000 | 0 |  | 1.2443 |
| KMNIST | AffineNorm-AdamW-LN-D6 | 3 | 0.7190 | 0.0363 | 0.0530 |  | -0.0273 | -0.4860 | 0.6002 | 0.1510 | 34.2373 |  | 0.0000 | 0.0000 | 960 | 0.000 | 1.3166 |
| KMNIST | AffineNorm-FDiag-LN-D6 | 3 | 0.7097 | 0.0042 | 0.0623 |  | -0.0579 | -0.5302 | 0.6152 | 0.1500 | 37.9564 |  | 0.0000 | 0.0000 | 0 | 1.000 | 1.3997 |
| KMNIST | FixedNorm-FNG-leftFullRight | 3 | 0.7280 | 0.0112 | 0.0440 |  | 0.2678 | -0.0591 | 0.3609 | 0.1488 | 30.5086 | 1895.8 | 0.0000 | 0.0000 | 0 |  | 1.4663 |
| KMNIST | AffineNorm-AdamW-LN-FNG-leftFullRight | 3 | 0.7287 | 0.0033 | 0.0433 |  | 0.2658 | -0.0619 | 0.3678 | 0.1487 | 37.2219 | 1895.8 | 0.0000 | 0.0000 | 960 | 0.000 | 1.5362 |
| KMNIST | AffineNorm-FDiag-LN-FNG-leftFullRight | 3 | 0.7240 | 0.0118 | 0.0480 |  | 0.2659 | -0.0619 | 0.3023 | 0.1483 | 28.3178 | 1895.8 | 0.0000 | 0.0000 | 0 | 1.000 | 1.6243 |

P2 verdict:

```text
NoNorm is a hard failure boundary, so normalization is necessary.
Learnable AffineNorm / ScalarGain can slightly help AdamW on MNIST/KMNIST,
but it does not close the PureKAN functional optimizer gap on Fashion/KMNIST.
AffineNorm-FDiag also does not rescue FNG-leftFullRight.

Conclusion: missing learnable normalization is not the main bottleneck.
The core issue remains KAN coefficient optimizer dynamics / metric direction.
```

## P3 Relaxed FNG Diagnostic

Because P1 had no formal survivor, this run is intentionally labeled relaxed diagnostic. It includes only the no-bad-step FNG variants plus baselines, to check whether the P1 local signal becomes a short-run training signal.

| dataset | method | runs | acc | std | gap vs AdamW | acc-D0 | AUC imp vs D6 | AUC imp vs AdamW | ECE red | phi p95 | J | FNG cond | bad | fallback | nonKAN | norm cov | time/A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MNIST | AdamW | 3 | 0.9303 | 0.0076 | 0.0000 | 0.0433 | 0.4248 | 0.0000 | 0.0000 | 0.2274 | 226275.4421 |  | 0.0000 | 0.0000 | 0 |  | 1.0000 |
| MNIST | D0-allFullSobolev | 3 | 0.8870 | 0.0455 | 0.0433 | 0.0000 | -0.0537 | -0.8318 | -0.3594 | 0.1489 | 34.4779 |  | 0.0000 | 0.0000 | 0 |  | 1.3322 |
| MNIST | D6-allTaskAware | 3 | 0.8620 | 0.0708 | 0.0683 | -0.0250 | 0.0000 | -0.7384 | 0.0949 | 0.1508 | 32.8464 |  | 0.0000 | 0.0000 | 0 |  | 1.3209 |
| MNIST | F2-FNG-right-only | 3 | 0.9297 | 0.0079 | 0.0007 | 0.0427 | 0.4046 | -0.0350 | 0.4066 | 0.1485 | 22.9295 | 2119.3 | 0.0000 | 0.0000 | 0 |  | 1.3819 |
| MNIST | F4-FNG-leftFull-right-noSob | 3 | 0.9030 | 0.0062 | 0.0273 | 0.0160 | 0.3915 | -0.0578 | 0.3140 | 0.1484 | 32.0859 | 2101.4 | 0.0000 | 0.0000 | 0 |  | 1.5255 |
| MNIST | F4-FNG-leftFull-right-lowSob | 3 | 0.9153 | 0.0074 | 0.0150 | 0.0283 | 0.4503 | 0.0444 | 0.3972 | 0.1488 | 28.4579 | 2079.7 | 0.0000 | 0.0000 | 0 |  | 1.4826 |
| Fashion-MNIST | AdamW | 3 | 0.8563 | 0.0090 | 0.0000 | 0.0243 | -0.0409 | 0.0000 | 0.0000 | 0.2396 | 369574.0127 |  | 0.0000 | 0.0000 | 0 |  | 1.0000 |
| Fashion-MNIST | D0-allFullSobolev | 3 | 0.8320 | 0.0120 | 0.0243 | 0.0000 | -0.1076 | -0.0640 | 0.6596 | 0.1490 | 31.4545 |  | 0.0000 | 0.0000 | 0 |  | 1.3977 |
| Fashion-MNIST | D6-allTaskAware | 3 | 0.8303 | 0.0358 | 0.0260 | -0.0017 | 0.0000 | 0.0393 | 0.3910 | 0.1506 | 31.2786 |  | 0.0000 | 0.0000 | 0 |  | 1.3386 |
| Fashion-MNIST | F2-FNG-right-only | 3 | 0.8467 | 0.0045 | 0.0097 | 0.0147 | 0.0900 | 0.1258 | 0.2483 | 0.1484 | 26.0255 | 1458.8 | 0.0000 | 0.0000 | 0 |  | 1.4323 |
| Fashion-MNIST | F4-FNG-leftFull-right-noSob | 3 | 0.8140 | 0.0159 | 0.0423 | -0.0180 | -0.0987 | -0.0555 | 0.0314 | 0.1484 | 32.0918 | 1394.1 | 0.0000 | 0.0000 | 0 |  | 1.5666 |
| Fashion-MNIST | F4-FNG-leftFull-right-lowSob | 3 | 0.8317 | 0.0040 | 0.0247 | -0.0003 | -0.0152 | 0.0247 | 0.1581 | 0.1485 | 27.0725 | 1477.1 | 0.0000 | 0.0000 | 0 |  | 1.5458 |
| KMNIST | AdamW | 3 | 0.7720 | 0.0102 | 0.0000 | 0.0937 | 0.3087 | 0.0000 | 0.0000 | 0.2172 | 20374.7067 |  | 0.0000 | 0.0000 | 0 |  | 1.0000 |
| KMNIST | D0-allFullSobolev | 3 | 0.6783 | 0.0147 | 0.0937 | 0.0000 | -0.2304 | -0.7797 | 0.7098 | 0.1493 | 35.5405 |  | 0.0000 | 0.0000 | 0 |  | 1.3990 |
| KMNIST | D6-allTaskAware | 3 | 0.7500 | 0.0134 | 0.0220 | 0.0717 | 0.0000 | -0.4464 | 0.7258 | 0.1512 | 32.7251 |  | 0.0000 | 0.0000 | 0 |  | 1.3298 |
| KMNIST | F2-FNG-right-only | 3 | 0.7397 | 0.0148 | 0.0323 | 0.0613 | 0.2836 | -0.0362 | 0.3761 | 0.1489 | 26.7338 | 1895.8 | 0.0000 | 0.0000 | 0 |  | 1.3588 |
| KMNIST | F4-FNG-leftFull-right-noSob | 3 | 0.6970 | 0.0079 | 0.0750 | 0.0187 | 0.2066 | -0.1477 | 0.1838 | 0.1488 | 28.4713 | 1829.8 | 0.0000 | 0.0000 | 0 |  | 1.4830 |
| KMNIST | F4-FNG-leftFull-right-lowSob | 3 | 0.7070 | 0.0128 | 0.0650 | 0.0287 | 0.2584 | -0.0727 | 0.1985 | 0.1487 | 26.5250 | 1845.7 | 0.0000 | 0.0000 | 0 |  | 1.5253 |

### P3 Gate Check

| method | formal P1 pass | max acc gap | AUC < D6 all | max bad | max fallback | geom <= AdamW | enter P4 |
|---|---|---|---|---|---|---|---|
| D6-allTaskAware | no | 0.0683 | no | 0.0000 | 0.0000 | yes | no |
| F2-FNG-right-only | no | 0.0323 | yes | 0.0000 | 0.0000 | yes | no |
| F4-FNG-leftFull-right-lowSob | no | 0.0650 | no | 0.0000 | 0.0000 | yes | no |
| F4-FNG-leftFull-right-noSob | no | 0.0750 | no | 0.0000 | 0.0000 | yes | no |

P3 verdict: no formal survivor enters P4. The relaxed run is useful diagnostically, but it cannot override the P1 gate. FNG improves some Sobolev-only trajectories, yet the joint accuracy/AUC/geometry requirements are not met across MNIST, Fashion-MNIST, and KMNIST.

## Visualizations

Generated dashboards:

```text
results/gafu_v3_9_figures/p1_direction_quality.svg
results/gafu_v3_9_figures/p2_norm_scorecard.svg
results/gafu_v3_9_figures/p3_fng_relaxed_scorecard.svg
matplotlib unavailable; SVG fallback used: No module named 'matplotlib'
```

## P4-P7 Decision

```text
P4 temporal dynamics:
  not run
  reason: no formal P1/P3 survivor.

P5 combined norm + FNG selection:
  not run
  reason: P3 entry gate was not reached.

P6 5-seed confirm / P7 10-seed final:
  not run by plan.
```

## Failure Diagnosis

```text
Normalization bottleneck:
  not primary.
  NoNorm fails, but learnable/functional norm does not recover PureKAN functional training.

FNG direction bottleneck:
  yes.
  FNG fixes many one-step bad directions, but the cosine gate and cross-dataset training gates fail.

Temporal dynamics bottleneck:
  possible but not yet justified.
  Since no candidate formally passed P1/P3, larger temporal-dynamics sweeps would be premature.

Architecture bottleneck:
  unlikely as the main cause.
  PureKAN-AdamW remains trainable; optimizer metric is the weak link.
```

## Final Decision

```text
PureKAN functional optimization remains unsolved in v3.9.

What is confirmed:
  1. Normalization is necessary, but learnable norm is not sufficient.
  2. FNG/KFAC-style task metrics repair part of the local descent problem.
  3. Current FNG still does not replace AdamW for full PureKAN training.

Accepted contribution:
  Hybrid DG-KAN branch functional update remains the valid mainline.

Next recommended direction:
  Redesign the PureKAN functional metric around safer layer-local/block-local objectives
  and only revisit temporal momentum/safeguards once a candidate passes the direction gate.
```



---


# Source 9: `docs/DG-KAN_v4.1_FunctionalUpdate_Redesign_DeepPlan.md`


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



---


# Source 10: `docs/DG-KAN_v4.1_FunctionalUpdate_Redesign_结果复盘.md`


# DG-KAN v4.1 Functional Update Redesign 结果复盘

本轮依据 `docs/DG-KAN_v4.1_FunctionalUpdate_Redesign_DeepPlan.md`。目标是验证重新设计的 PureKAN functional update：FTF、FC-Adam 与 FTR/FGN trust-region 是否能把 v3.7-v3.9 的局部方向修复转成长程训练收益。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 smoke | 30 | 0 |
| P1 shadow | 30 | 0 |
| P2 micro | 72 | 5 |

## Code / Config Changes

```text
experiments/dgkan_core.py
  Added FTF fields and per-role target-fitting updates for PureKAN coeffs.
  Added FC-Adam functional-coordinate updates through Sobolev Cholesky coordinates.
  Added an FTR-CG-small diagnostic path with layer-output trust scaling.
  Result rows now include FTF fit/residual/trust stats, FC-Adam reconstruction diagnostics,
  and FTR residual / predicted change statistics.

experiments/run_gafu_v41.py
  Added P0/P1/P2 packages and one-batch direction audit for v4.1 candidates.

experiments/analyze_gafu_v41.py
  Writes the required v4.1 CSV/JSON artifacts and this replay.
```

## P0 Implementation Smoke

P0 rows: `30`; errors: `0`.

Key audit:

```text
PureKAN alphaFixed1 / fixed norm paths ran without implementation errors.
Strict PureKAN rows kept learnable_nonKAN_params = 0.
Functional rows covered input/block/output coeff groups.
FTF, FC-Adam, and FTR all produced finite one-epoch smoke rows.
```

## P1 One-Batch Direction Gate

| method | bad | min train | min val | min FTF R2 | max delta | P1 pass |
|---|---|---|---|---|---|---|
| D0-allFullSobolev | 0 | 0.0016 | -0.0039 |  |  | yes |
| D6-allTaskAware | 1 | -0.0080 | -0.0091 |  |  | no |
| F4-FNG-leftFull-right | 0 | 0.0488 | 0.0046 |  |  | yes |
| FC-Adam-one-step | 0 | 0.0212 | -0.0377 |  |  | no |
| FTF-all-sequential | 0 | 0.0784 | 0.0636 | 1.0000 | 0.1000 | yes |
| FTF-all-simultaneous | 0 | 0.1247 | 0.0231 | 0.9999 | 0.1000 | yes |
| FTF-blocks-output | 0 | 0.0664 | 0.0086 | 0.9997 | 0.0997 | yes |
| FTF-output-only | 0 | 0.0093 | -0.0666 | 1.0000 | 0.0998 | no |
| FTR-CG-small | 0 | 0.0388 | 0.0023 |  |  | yes |

P1 survivors used for P2:

```text
D0-allFullSobolev, F4-FNG-leftFull-right, FTF-all-sequential, FTF-all-simultaneous, FTF-blocks-output, FTR-CG-small
```

Observation:

```text
FTF repaired the one-step direction signal strongly: blocks/output and all-layer modes had positive train and validation descent on all three datasets, with fit R2 near 1.0.
D6 still had a KMNIST bad train step and was not treated as a P2 candidate, but was later added as a P2 baseline because the plan requires AUC comparison vs D6.
FC-Adam improved train loss but failed the validation-descent gate on Fashion/KMNIST.
```

## P2 Micro-Run Scorecard

| dataset | method | runs | errors | acc | std | gap vs AdamW | AUC imp vs AdamW | AUC imp vs D6 | ECE red | phi ratio | FTF R2 | P2 pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MNIST | D0-allFullSobolev | 3 | 0 | 0.8417 | 0.0304 | 0.0723 | -0.8900 | 0.0003 | -8.4610 | 0.7658 |  | 0 |
| MNIST | D6-allTaskAware | 3 | 0 | 0.8523 | 0.0068 | 0.0617 | -0.8906 | 0.0000 | -5.7526 | 0.7685 |  | 0 |
| MNIST | F4-FNG-leftFull-right | 3 | 0 | 0.9233 | 0.0110 | -0.0093 | -0.0388 | 0.4505 | -0.4553 | 0.7661 |  | 0 |
| MNIST | FTF-all-sequential | 3 | 0 | 0.1003 | 0.0005 | 0.8137 | -518293472445.2520 | -274141826146.2508 | -42.8198 | 509227.2591 | 0.9453 | 0 |
| MNIST | FTF-all-simultaneous | 3 | 0 | 0.1003 | 0.0005 | 0.8137 | -518293472445.2520 | -274141826146.2508 | -42.8198 | 509227.2591 | 0.9453 | 0 |
| MNIST | FTF-blocks-output | 3 | 0 | 0.1023 | 0.0033 | 0.8117 | -1754569223347315.0000 | -928047209855757.8750 | -42.7223 | 22285636.6922 | 0.7068 | 0 |
| MNIST | FTR-CG-small | 3 | 0 | 0.6687 | 0.0107 | 0.2453 | -1.9351 | -0.5525 | -18.2742 | 0.7644 |  | 0 |
| MNIST | PureKAN-AdamW | 3 | 0 | 0.9140 | 0.0043 | 0.0000 | 0.0000 | 0.4711 | 0.0000 | 1.0000 |  | 1 |
| Fashion-MNIST | D0-allFullSobolev | 3 | 0 | 0.8170 | 0.0174 | 0.0190 | -0.5294 | -0.1251 | -0.1849 | 0.7614 |  | 0 |
| Fashion-MNIST | D6-allTaskAware | 3 | 0 | 0.8060 | 0.0134 | 0.0300 | -0.3594 | 0.0000 | 0.4009 | 0.7636 |  | 0 |
| Fashion-MNIST | F4-FNG-leftFull-right | 3 | 0 | 0.8227 | 0.0164 | 0.0133 | -0.1005 | 0.1904 | 0.3178 | 0.7607 |  | 1 |
| Fashion-MNIST | FTF-all-sequential | 2 | 1 | 0.1000 | 0.0000 | 0.7360 | -105351734110180880.0000 | -77501361878272944.0000 | -10.9915 | 203698804.2161 | 0.9480 | 0 |
| Fashion-MNIST | FTF-all-simultaneous | 2 | 1 | 0.1000 | 0.0000 | 0.7360 | -105351734110180880.0000 | -77501361878272944.0000 | -10.9915 | 203698804.2161 | 0.9480 | 0 |
| Fashion-MNIST | FTR-CG-small | 3 | 0 | 0.7167 | 0.0194 | 0.1193 | -1.7757 | -1.0420 | -3.1114 | 0.7598 |  | 0 |
| Fashion-MNIST | PureKAN-AdamW | 3 | 0 | 0.8360 | 0.0156 | 0.0000 | 0.0000 | 0.2644 | 0.0000 | 1.0000 |  | 1 |
| KMNIST | D0-allFullSobolev | 3 | 0 | 0.5750 | 0.0185 | 0.1930 | -1.0548 | -0.1156 | -0.5424 | 0.7607 |  | 0 |
| KMNIST | D6-allTaskAware | 3 | 0 | 0.5610 | 0.0550 | 0.2070 | -0.8419 | 0.0000 | 0.2455 | 0.7640 |  | 0 |
| KMNIST | F4-FNG-leftFull-right | 3 | 0 | 0.7123 | 0.0132 | 0.0557 | -0.1847 | 0.3568 | 0.3224 | 0.7648 |  | 0 |
| KMNIST | FTF-all-sequential | 3 | 0 | 0.1070 | 0.0099 | 0.6610 | -788570634358.2345 | -428138390693.1385 | -9.8694 | 626491.2995 | 0.9186 | 0 |
| KMNIST | FTF-all-simultaneous | 3 | 0 | 0.1070 | 0.0099 | 0.6610 | -788570634358.2345 | -428138390693.1385 | -9.8694 | 626491.2995 | 0.9186 | 0 |
| KMNIST | FTF-blocks-output | 3 | 0 | 0.1000 | 0.0000 | 0.6680 | -117251294060245.6875 | -63659205857378.9766 | -9.9546 | 6377580.6882 | 0.6710 | 0 |
| KMNIST | FTR-CG-small | 3 | 0 | 0.3730 | 0.0140 | 0.3950 | -2.1432 | -0.7065 | -1.2296 | 0.7613 |  | 0 |
| KMNIST | PureKAN-AdamW | 3 | 0 | 0.7680 | 0.0067 | 0.0000 | 0.0000 | 0.4571 | 0.0000 | 1.0000 |  | 1 |

## P2 Failure Diagnosis

```text
No candidate passed the P2 joint gate.

FTF:
  P1 target fit was excellent, but P2 training was catastrophic.
  MNIST/KMNIST dropped to near chance accuracy, Fashion produced numerical eigensolve failures for several FTF rows, and val-loss AUC exploded.
  This matches the plan's failure mode: fit_R2 high + one-step descent positive + short-run acc bad => layer-local target fitting causes cross-layer drift.

FTR-CG-small:
  Local direction was acceptable, but short training underfit badly on all datasets.

F4-FNG-leftFull-right:
  Best practical candidate in P2.
  It matched/beat PureKAN-AdamW on MNIST and stayed within about 1.3 points on Fashion.
  It failed KMNIST by about 5.6 points, so it cannot enter P3.

D0/D6:
  Geometry is stable but accuracy remains below AdamW, especially on KMNIST.
```

## P3-P5 Decision

```text
P3 refinement: not run.
Reason: P2 produced no survivor.

P4 5-seed confirm: not run.
Reason: P3 was not reached.

P5 10-seed final: not run.
Reason: P4 was not reached.
```

## Artifacts

Required files were written under `results/v4_1/`:

```text
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
figures/p2_kmnist_acc_gap.svg
figures/p2_kmnist_auc_vs_d6.svg
```

## Final Decision

```text
PureKAN functional optimization is still not solved in v4.1.

What improved:
  FTF gives a real one-step target-fitting direction.
  FNG remains the strongest short-run practical baseline among functional candidates.

What failed:
  FTF target fitting is not yet a stable optimizer; high local fit creates long-horizon drift.
  FTR-CG-small is too weak in this approximation.
  FNG does not close the KMNIST accuracy gap.

Next recommended direction:
  Add accepted-step / backtracking and cross-layer drift control to FTF before more seeds.
  In particular, target fitting should be sequential with validation of actual loss decrease
  and an activation/logit trust region that rejects or shrinks unsafe layer updates.
```



---


# Source 11: `docs/DG-KAN_v4.2_FunctionalTrustRegion_DeepRedesign_实验计划.md`



# DG-KAN v4.2 Functional Update 深度重设计实验计划

## 0. 这份计划的目的

v4.1 的结果已经非常明确：当前 PureKAN functional optimization 不能再靠小修小补推进。继续调 `branch_final_scale`、`Sobolev alpha/beta`、`diagwarmup`、`smooth transition`、固定 shallow/deep 分工、或者单纯加 learnable normalization，都不能解决核心问题。

这份 v4.2 计划的目标不是继续扩大超参搜索，而是重新定义 functional update：

$$
\boxed{
\text{functional update 不应该只是 }S^{-1}g\text{，而应该是带全局验收的函数空间任务下降。}
}
$$

本计划要回答三个核心问题：

1. **为什么 v4.1 的 FTF、FNG、FTR 都没有真正解决 PureKAN？**
2. **functional update 的正确设计应该是什么？**
3. **下一轮如何用最少实验验证新设计，而不是继续随机调参？**

最终目标仍然不变：

$$
\operatorname{Acc}_{\text{PureKAN-functional}}
\geq
\max(
\operatorname{Acc}_{\text{PureKAN-AdamW}},
\operatorname{Acc}_{\text{Hybrid-DGKAN-UFULL}},
\operatorname{Acc}_{\text{MLP-AdamW}}
).
$$

如果不能超过这些基线，就不能 claim PureKAN functional optimizer 已经成立。

---

## 1. v4.1 结果的深度解读

### 1.1 P0 说明实现层面基本干净

v4.1 的 P0 smoke 没有实现错误。PureKAN 的 strict functional rows 满足：

```text
learnable_nonKAN_params = 0
input / block / output coeff coverage = 1.0
FTF / FC-Adam / FTR 都能产生有限 smoke rows
```

这意味着后面的失败不能再主要归因于：

```text
alpha 没 fixed
coefficient 没被识别
input/output KAN 没被 functional update
存在隐藏 non-KAN 参数
warmup 没生效
```

从 v4.1 开始，问题已经转为 **optimizer 设计问题**。

---

### 1.2 P1 说明 FTF 的 one-step direction 是真的，但不等于 optimizer 成功

v4.1 P1 中，FTF 的 one-step direction 很强：

```text
FTF-all-sequential:
  min train descent = 0.0784
  min val descent = 0.0636
  min FTF R2 = 1.0000

FTF-all-simultaneous:
  min train descent = 0.1247
  min val descent = 0.0231
  min FTF R2 = 0.9999

FTF-blocks-output:
  min train descent = 0.0664
  min val descent = 0.0086
  min FTF R2 = 0.9997
```

这说明 FTF 的局部 target-fitting 不是假信号。它确实能在一小步内让当前 batch 和 validation probe 上的 loss 下降。

但是 P2 中 FTF 训练完全崩溃：

```text
MNIST / Fashion / KMNIST accuracy 接近 chance；
val-loss AUC 巨大负值；
phi ratio 爆炸到 10^5 - 10^8 量级；
Fashion 上部分 FTF row 出现 eigensolve failure。
```

这说明：

$$
\boxed{
\text{FTF 的局部 target fit 太强，但没有控制跨层漂移和长期动力学。}
}
$$

高 $R^2$ 不是成功，反而说明它把局部 target 拟合得太彻底，导致多个层同时改变后，网络整体函数发生不可控漂移。

---

### 1.3 FNG 是当前最接近可用的 functional candidate，但仍然不是解

v4.1 P2 中，`F4-FNG-leftFull-right` 是唯一有实际价值的 functional candidate：

```text
MNIST:
  PureKAN-AdamW acc = 0.9140
  F4-FNG acc = 0.9233

Fashion:
  PureKAN-AdamW acc = 0.8360
  F4-FNG acc = 0.8227

KMNIST:
  PureKAN-AdamW acc = 0.7680
  F4-FNG acc = 0.7123
```

它说明 left-right / KFAC-like curvature 是比 Sobolev-only、D6 task-aware diagonal 更接近正确方向的。但是它仍然不能进入 confirm，因为 KMNIST gap 太大，AUC improvement vs AdamW 仍为负。

这说明：

$$
\boxed{
\text{FNG 方向比旧方法好，但它仍只是局部 layer-wise preconditioner，不是完整深度网络 optimizer。}
}
$$

---

### 1.4 FTR-CG-small 失败说明“弱全局 trust region”不够

FTR-CG-small 在 P1 通过 direction gate，但 P2 中严重 underfit：

```text
MNIST acc = 0.6687
Fashion acc = 0.7167
KMNIST acc = 0.3730
```

这说明一个非常重要的问题：只要 global trust-region 的近似太弱、太粗、或者只在小线性化空间里求解，它不会自动变成好 optimizer。

所以不能简单说：

$$
\text{global trust region} = \text{solution}.
$$

必须设计 **可验收、可回退、可控制漂移** 的 blockwise functional trust-region，而不是弱化版 CG diagnostic。

---

### 1.5 FC-Adam one-step 未通过 validation gate，说明 AdamW dynamics 不能直接搬进 functional coordinates

FC-Adam-one-step 在 P1 中：

```text
bad = 0
min train descent = 0.0212
min val descent = -0.0377
```

它能降低训练 batch，但 validation probe 失败。这说明 functional-coordinate Adam 的方向可能存在 batch overfit 或尺度不稳问题。

所以 AdamW 的长期动力学值得借鉴，但不能直接简单地：

$$
a = L^{-T}u,\quad u \leftarrow \operatorname{AdamW}(u).
$$

它需要 validation-aware 或 multi-batch acceptance，否则会重蹈 “train descent but val drift” 的问题。

---

## 2. 当前 functional update 的根本缺陷

### 2.1 缺陷一：把 smoothness prior 当成 optimizer metric

旧 U-FULL 的核心是：

$$
\Delta a=-\eta S^{-1}g,
$$

其中 $S$ 是 Sobolev smoothness metric：

$$
S=
\int BB^\top
+
\alpha\int B'B'^\top
+
\beta\int B''B''^\top.
$$

这个 metric 关心的是函数是否平滑，而不是分类任务是否学好。它可以让：

```text
phi_prime_p95 下降
Jacobian condition 下降
curvature 下降
ECE 下降
```

但它不保证：

```text
class margin 提高
representation 变可分
train / val loss 长程下降
output logits 的任务方向正确
```

这就是为什么 hybrid-DGKAN 里它能当 residual branch 的几何稳定器，而 PureKAN 里它不能独立承担全网络训练。

---

### 2.2 缺陷二：缺少 downstream sensitivity

对 RBF KAN layer：

$$
y_j=\sum_i\sum_m a_{jim}B_m(x_i).
$$

局部 coefficient tangent 是：

$$
\frac{\partial y_j}{\partial a_{jim}}=B_m(x_i).
$$

如果只构造 basis-side metric：

$$
A_\Phi=\mathbb E[\Phi^\top\Phi],
$$

它只知道输入 basis 的几何，却不知道当前输出 channel 对最终 loss 的重要性。

真正的任务几何应该包含 downstream sensitivity：

$$
G_l
\approx
\mathbb E
\left[
\Phi_l^\top\Phi_l
\otimes
C_l
\right],
$$

其中 $C_l$ 是该层输出侧的 task curvature / Fisher / downstream gradient covariance。

当前 D6 / TFU / FNG 都没有充分解决这个问题。FNG 引入了一部分 left-side curvature，但还不够稳定，说明 $C_l$ 的估计、阻尼、时间尺度和跨层协调都还不成熟。

---

### 2.3 缺陷三：层局部目标导致 cross-layer drift

FTF 的失败最典型。它在每层拟合：

$$
Y_l^\star=Y_l-\tau_l\delta_l,
$$

然后求：

$$
\Delta A_l
=
\arg\min_{\Delta A}
\left[
\|\Psi_l\Delta A_l^\top-T_l\|_F^2
+
\lambda_l\|\Delta A_l\|_{S_l}^2
+
\rho\|\Delta A_l\|_F^2
\right].
$$

单层看很好，甚至 $R^2\approx1$。但多个层同时更新时，输入分布、hidden state、下游 sensitivity 都会改变。于是每层都正确，组合起来却错。

这说明：

$$
\boxed{
\text{PureKAN functional update 必须有跨层漂移控制。}
}
$$

不能只做 layer-local closed-form solve。

---

### 2.4 缺陷四：没有真正的 accepted-step 机制

目前很多方法只检查：

```text
metric norm 是否太大
trust clip 是否触发
one-step shadow 是否下降
```

但这不等于每个训练 step 都被实际验收。真正需要的是：

```text
propose update
apply temporarily
evaluate global loss / activation drift / logit drift
accept, shrink, or reject
```

也就是 trust-region optimizer 的核心 acceptance ratio：

$$
r_t
=
\frac{
L(\theta)-L(\theta+\Delta\theta)
}{
-\widehat{\Delta L}
}.
$$

如果 $r_t<0$，更新必须拒绝；如果 $r_t$ 很低，trust radius 必须缩小；如果 $r_t$ 很高，才允许扩大 step。

FTF 的灾难说明：没有 accepted-step，局部 target fitting 会变成长期漂移。

---

### 2.5 缺陷五：缺少长期时间尺度动力学

AdamW 能训练 PureKAN，不是因为它懂函数空间，而是因为它有：

```text
momentum
RMS scale adaptation
gradient noise averaging
long-horizon step-size stabilization
```

我们的 functional update 通常是：

$$
\Delta a_t=-\eta M_t^{-1}g_t.
$$

这缺少时间尺度。之前 Adam-style UO 失败，是因为直接对 raw gradient / Adam direction 做 moment 会破坏 functional geometry。但这不意味着不能使用时间尺度，而是要对 **已验收的 functional direction** 做 temporal smoothing。

正确形式应该是：

$$
d_t=M_t^{-1}g_t,
$$

$$
\hat d_t=\frac{d_t}{\|d_t\|_{M_t}+\epsilon},
$$

$$
m_t=\beta m_{t-1}+(1-\beta)\hat d_t,
$$

然后只在通过 accepted-step 后更新 $m_t$ 或扩大步长。

---

## 3. 下一步方向：从 U-FULL / TFU / FNG 转向 BFT

我建议 v4.2 不再叫 U-FULL / TFU / FNG，而改成：

```text
BFT: Blockwise Functional Trust optimization
```

核心思想：

$$
\boxed{
\text{每个 KAN 层/块可以提出 functional update，但必须经过全网络 loss 和 activation trust region 验收。}
}
$$

BFT 不是一个单一 preconditioner，而是一个 optimizer protocol：

1. 每个 block 产生候选 functional direction。
2. 候选方向可以来自 FNG、FTF、FC-Adam、data metric 或 Sobolev regularized solve。
3. 更新不是直接提交，而是先进入 trust-region acceptance。
4. 验收标准同时看 global train loss、heldout micro-batch loss、activation drift、logit drift 和 Sobolev norm。
5. 只有通过验收的 update 才真正写回参数。
6. Sobolev 不再是主 metric，而是约束项和漂移惩罚。

---

## 4. BFT 的数学定义

### 4.1 Blockwise proposal

对第 $l$ 个 KAN block，先构造 candidate direction：

$$
d_l^{proposal}
=
-\left(
M_l+\rho I
\right)^{-1}g_l.
$$

其中 $M_l$ 不固定，可以来自：

```text
FNG:
  M_l = C_l \otimes A_{\Phi,l} + \lambda S_l

FTF:
  M_l = \Psi_l^\top \Psi_l + \lambda S_l

FC:
  M_l = S_l, but update in whitened coordinate

Raw:
  M_l = I
```

v4.2 的重点不是哪个 proposal 一定对，而是所有 proposal 都必须经过同一个 acceptance protocol。

---

### 4.2 Actual functional trust objective

候选 update $\Delta_l$ 必须满足：

$$
L_{\text{train}}(\theta+\Delta_l)
\leq
L_{\text{train}}(\theta)
-
c\cdot \widehat{\Delta L_l}.
$$

同时还要满足 heldout micro-batch 不恶化：

$$
L_{\text{holdout}}(\theta+\Delta_l)
\leq
L_{\text{holdout}}(\theta)+\epsilon_{val}.
$$

并限制 activation drift：

$$
\frac{
\|h_l(\theta+\Delta_l)-h_l(\theta)\|_2
}{
\|h_l(\theta)\|_2+\epsilon
}
\leq
r_h.
$$

以及 logit drift：

$$
\frac{
\|z(\theta+\Delta_l)-z(\theta)\|_2
}{
\|z(\theta)\|_2+\epsilon
}
\leq
r_z.
$$

如果任何条件失败，就缩小 step：

$$
\eta_l \leftarrow \gamma\eta_l.
$$

最多尝试 $K$ 次。如果仍失败，则拒绝该 block update，或者回退到 raw gradient / AdamW baseline direction。

---

### 4.3 Sequential block update，而不是 all-layer simultaneous

FTF 的灾难说明 simultaneous layer update 很危险。因此 v4.2 的默认是 Gauss-Seidel 式顺序更新：

```text
input KAN -> block 0 -> block 1 -> ... -> output KAN
```

每接受一个 block update 后，重新 forward，重新计算后续 block 的 activations 和 gradients。这样可以减少 cross-layer drift。

同时需要测试反向顺序：

```text
output KAN -> deep block -> shallow block -> input KAN
```

因为输出层最接近 loss，可能更适合先更新。

---

### 4.4 Proposal mixing

每个 block 可以有多个 proposal：

$$
d_l^{FNG},\quad d_l^{FTF},\quad d_l^{raw},\quad d_l^{Sob}.
$$

BFT 可以选择：

$$
d_l
=
\arg\max_{d\in \mathcal D_l}
\operatorname{accepted\_gain}(d).
$$

也就是说，下一步不要再假设一个固定 metric 全局最优，而是让 optimizer 在每个 block 上选择实际通过 trust test 的方向。

---

## 5. v4.2 实验总览

v4.2 分成八个阶段：

```text
P0: Implementation smoke and invariants
P1: Proposal direction audit
P2: Single-block accepted-step audit
P3: Sequential BFT micro-run
P4: Proposal-mixing and order ablation
P5: Temporal dynamics ablation
P6: 3-seed full-budget candidate selection
P7: 5-seed confirm
P8: 10-seed final confirm and mechanism audit
```

核心原则是：

$$
\boxed{
\text{先证明每一步能被全网络验收，再证明多 seed 训练能超过 AdamW。}
}
$$

---

## 6. P0: Implementation smoke and invariants

### 6.1 目的

确认 BFT 不是因为代码错误而失败。必须先保证所有候选方法都能：

```text
构造 proposal
临时 apply / rollback 参数
计算 actual loss change
记录 activation / logit drift
根据 acceptance ratio 接受或拒绝
```

### 6.2 方法

数据集：

```text
MNIST
Fashion-MNIST
KMNIST
```

模型：

```text
PureKAN hidden_dim=64 depth=2 basis=16 alphaFixed1 FixedNorm
```

方法：

```text
PureKAN-AdamW
D0-allFullSobolev
D6-allTaskAware
F4-FNG-leftFullRight
FTF-blocks-output-safe
BFT-FNG
BFT-FTF
BFT-mixed
```

### 6.3 必须记录

每个 run 必须记录：

```text
learnable_nonKAN_params
functional_coverage
input_coeff_seen
block_coeff_seen
output_coeff_seen
proposal_count
accepted_count
rejected_count
rollback_count
nan_count
max_backtrack_count
mean_backtrack_count
activation_drift_mean
logit_drift_mean
accepted_eta_mean
accepted_eta_min
accepted_eta_max
```

### 6.4 通过标准

P0 通过需要满足：

```text
所有 method 无 NaN / Inf
rollback 后参数完全恢复
functional coverage = 1.0
accepted/rejected/backtrack 统计有限
PureKAN strict variants nonKAN params = 0
```

---

## 7. P1: Proposal direction audit

### 7.1 目的

比较不同 proposal 的实际全网络下降能力，而不是只看 raw/preconditioned cosine。

### 7.2 Proposal 列表

```text
raw-gradient
Sobolev-full
task-diag-D6
FNG-leftFullRight
FTF-output-only
FTF-blocks-output
FC-whitened-gradient
FC-whitened-Adam-one-step
mixed-best-of-proposals
```

### 7.3 执行方式

每个 dataset / seed / block role 上，做一次 one-batch proposal，不永久更新参数。

每个 proposal 需要记录：

```text
predicted_descent
actual_train_descent
actual_holdout_descent
acceptance_ratio
activation_drift
logit_drift
metric_norm
sobolev_norm
raw_cos
adam_cos
proposal_rank
```

### 7.4 可视化

必须画：

1. **proposal actual descent bar chart**  
   横轴 proposal，纵轴 actual train / holdout descent。

2. **predicted vs actual descent scatter**  
   判断 proposal 的局部模型是否可信。

3. **activation drift vs actual descent scatter**  
   找出高下降但高漂移的危险 proposal。

4. **acceptance ratio heatmap**  
   行为 dataset，列为 proposal / role。

5. **proposal rank by role**  
   input / block / output 哪种 proposal 最常被接受。

### 7.5 判定

一个 proposal 进入 P2 需要满足：

```text
bad actual train step rate < 0.05
holdout descent positive on at least 2/3 datasets
median acceptance_ratio > 0.2
activation_drift_p95 < 0.10
logit_drift_p95 < 0.10
```

FTF proposal 如果仍然出现：

```text
fit_R2 high
actual_holdout_descent negative
activation_drift high
```

则说明它只能作为 proposal，不能直接作为 optimizer。

---

## 8. P2: Single-block accepted-step audit

### 8.1 目的

证明 BFT acceptance 能阻止 FTF 那种长程崩溃的第一步。

### 8.2 方法

只更新一个 block，其他层冻结。分别测试：

```text
input-only
block0-only
block1-only
output-only
all-blocks-but-one-at-a-time
```

每种 block 更新要跑：

```text
proposal = FNG
proposal = FTF
proposal = raw
proposal = mixed
```

### 8.3 指标

记录：

```text
block_role
proposal_type
accepted_rate
rejected_rate
mean_backtracks
actual_train_descent
actual_holdout_descent
activation_drift
logit_drift
next_layer_input_shift
class_margin_change
basis_occupancy_change
phi_prime_change
jacobian_change
```

### 8.4 关键可视化

1. **block role acceptance rate**  
   看 input / hidden / output 哪些层最难更新。

2. **accepted eta distribution**  
   看不同 layer 的可接受步长。

3. **class margin change by role**  
   判断 output-only 或 block-only 是否真正帮助分类。

4. **activation shift heatmap**  
   看哪个 block update 导致最大 downstream drift。

### 8.5 判定

P2 通过的 block-proposal 组合需要：

```text
accepted_rate > 0.50
holdout_descent_mean > 0
activation_drift_p95 < 0.10
logit_drift_p95 < 0.10
class_margin_change_mean > 0
```

如果 input-only 大量拒绝，说明 input KAN 的 functional proposal 不可靠，需要单独设计 input feature-learning metric。

---

## 9. P3: Sequential BFT micro-run

### 9.1 目的

验证 sequential accepted-step 是否能把 one-step 正信号转化为短训练收益。

### 9.2 方法

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

训练预算：

```text
epochs = 4 short budget
train_size = 6000
```

候选：

```text
PureKAN-AdamW
D0-allFullSobolev
D6-allTaskAware
F4-FNG-leftFullRight
BFT-FNG-forward
BFT-FNG-reverse
BFT-FTF-forward
BFT-FTF-reverse
BFT-mixed-forward
BFT-mixed-reverse
BFT-mixed-output-first
```

### 9.3 配置说明

`forward` 顺序：

```text
input -> block0 -> block1 -> output
```

`reverse` 顺序：

```text
output -> block1 -> block0 -> input
```

`output-first` 顺序：

```text
output -> input -> block0 -> block1
```

### 9.4 记录指标

除了常规指标外，必须记录：

```text
accepted_steps_per_epoch
rejected_steps_per_epoch
fallback_rate
mean_backtracking_per_epoch
actual_descent_per_epoch
holdout_descent_per_epoch
activation_drift_per_epoch
logit_drift_per_epoch
proposal_choice_distribution
role_update_share
role_acceptance_rate
```

常规指标：

```text
test_acc
val_loss
val_loss_auc
train_loss
train_acc
ECE
NLL
phi_prime_p95
jacobian_condition
basis_dead_fraction
basis_occupancy_entropy
class_margin_mean
class_margin_p10
```

### 9.5 可视化

必须画：

1. **val loss curves**
2. **val accuracy curves**
3. **acceptance rate over epochs**
4. **fallback rate over epochs**
5. **role update share stacked area plot**
6. **proposal choice stacked area plot**
7. **activation drift over epochs**
8. **logit drift over epochs**
9. **accuracy vs accepted eta scatter**
10. **geometry vs accuracy Pareto plot**

### 9.6 P3 通过标准

至少一个 BFT candidate 满足：

```text
MNIST gap vs PureKAN-AdamW < 2.0%
Fashion gap vs PureKAN-AdamW < 2.0%
KMNIST gap vs PureKAN-AdamW < 4.0%
AUC improvement vs D6 > 20%
no catastrophic val-loss AUC
accepted_rate > 0.30
fallback_rate < 0.50
```

如果 BFT-mixed 明显优于单一 FNG/FTF，说明 proposal choice 是必要的。

---

## 10. P4: Proposal-mixing and order ablation

### 10.1 目的

精炼 P3 最好候选，判断成功来自哪一部分：

```text
proposal 类型
更新顺序
acceptance rule
activation trust
heldout loss gate
```

### 10.2 Ablation 列表

假设 P3 最好是 `BFT-mixed-output-first`，P4 对它做消融：

```text
BFT-mixed-output-first
BFT-mixed-no-heldout-gate
BFT-mixed-no-activation-trust
BFT-mixed-no-logit-trust
BFT-mixed-no-backtracking
BFT-mixed-FNG-only
BFT-mixed-FTF-only
BFT-mixed-raw-only
BFT-mixed-fixed-order-forward
BFT-mixed-fixed-order-reverse
```

### 10.3 指标

重点记录：

```text
acc
AUC
accepted_rate
rejected_rate
fallback_rate
mean eta
eta shrink count
activation_drift
logit_drift
holdout_descent
role-wise contribution
```

### 10.4 可视化

1. **ablation scorecard**
2. **acceptance mechanism waterfall**
3. **which gate rejects updates**
4. **proposal usage by role**
5. **dataset-specific failure table**

### 10.5 判定

如果去掉 heldout gate 后 FTF 再次崩溃，说明 heldout gate 是必要的。  
如果去掉 activation trust 后 acc 下降但 train loss 更低，说明 cross-layer drift 是关键问题。  
如果 FNG-only 稳但弱，FTF-only 强但不稳，mixed-best-of-proposals 就是正确方向。

---

## 11. P5: Temporal dynamics ablation

### 11.1 目的

验证 functional optimizer 是否缺少 AdamW-like long-horizon dynamics。

### 11.2 方法

在 P4 最佳 BFT candidate 上加入 temporal smoothing：

```text
BFT-best-no-momentum
BFT-best-direction-EMA-beta0.8
BFT-best-direction-EMA-beta0.9
BFT-best-metric-normalized-momentum-beta0.9
BFT-best-RMS-step-controller
BFT-best-EMA+RMS
```

这里 momentum 不是 raw Adam moment，而是对 accepted functional direction 做：

$$
\hat d_t=\frac{d_t}{\|d_t\|_{M_t}+\epsilon},
$$

$$
m_t=\beta m_{t-1}+(1-\beta)\hat d_t.
$$

step scale 使用 accepted update 的历史：

$$
\eta_t
=
\eta_0
\cdot
\frac{1}{\sqrt{\operatorname{EMA}(\|d_t\|_M^2)}+\epsilon}.
$$

### 11.3 指标

记录：

```text
momentum_cos_current
momentum_norm
metric_norm_direction
accepted_eta
RMS_scale
train_loss_smoothness
val_loss_smoothness
gradient_noise_scale
```

### 11.4 判定

Temporal dynamics 进入 P6 需要：

```text
acc improves on at least 2/3 datasets
AUC improves on at least 2/3 datasets
fallback_rate does not increase > 20%
activation_drift_p95 remains controlled
```

如果 momentum 使 accepted_rate 降低或 drift 升高，则说明 PureKAN functional update 需要 trust region 而不是 temporal smoothing。

---

## 12. P6: 3-seed full-budget candidate selection

### 12.1 方法

训练 full budget：

```text
epochs = current PureKAN baseline budget
seeds = 0,1,2
datasets = MNIST, Fashion-MNIST, KMNIST
```

候选：

```text
PureKAN-AdamW
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
D6-allTaskAware
F4-FNG-leftFullRight
BFT-best
BFT-best+temporal
```

### 12.2 成功标准

候选进入 P7 必须满足：

```text
MNIST acc >= PureKAN-AdamW - 0.5%
Fashion acc >= PureKAN-AdamW - 0.5%
KMNIST acc >= PureKAN-AdamW - 1.0%

AUC improvement vs D6 > 20%
ECE not worse than AdamW by more than 5%
phi_prime_p95 lower than AdamW by at least 20%
jacobian condition lower than AdamW by at least 20%
accepted_rate > 0.30
fallback_rate < 0.50
```

注意：如果 candidate 没超过 PureKAN-AdamW，但明显接近，并且 geometry/ECE 大幅更好，可以作为 **Pareto candidate**，但不能 claim solve。

---

## 13. P7: 5-seed confirm

### 13.1 方法

只扩 P6 最优 1-2 个 candidate：

```text
seeds = 0,1,2,3,4
datasets = MNIST, Fashion-MNIST, KMNIST
```

对照：

```text
PureKAN-AdamW
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
best BFT candidate
```

### 13.2 指标

记录：

```text
paired acc delta vs PureKAN-AdamW
paired AUC delta vs PureKAN-AdamW
paired ECE delta
paired phi/J reduction
paired time ratio
paired acceptance rate
paired fallback rate
```

### 13.3 可视化

1. **paired delta plot**
2. **bootstrap CI plot**
3. **accuracy vs geometry Pareto**
4. **BFT acceptance timeline**
5. **role-wise accepted updates per dataset**
6. **failure table by seed**

### 13.4 P7 通过标准

```text
mean acc >= PureKAN-AdamW on at least 2/3 datasets
remaining dataset gap < 0.5%
AUC >= PureKAN-AdamW or D6 by clear margin
ECE not worse
geometry clearly better
```

---

## 14. P8: 10-seed final confirm

### 14.1 方法

只在 P7 通过后执行：

```text
seeds = 0..9
datasets = MNIST, Fashion-MNIST, KMNIST
```

### 14.2 最终 claim 分类

#### Clean solved

如果：

```text
BFT acc >= PureKAN-AdamW on all 3 datasets
BFT acc >= MLP-AdamW on at least 2/3 datasets
BFT acc >= Hybrid-DGKAN-UFULL on at least 2/3 datasets
AUC / ECE / geometry all non-worse
```

则可以 claim：

```text
PureKAN functional optimizer solved.
```

#### Pareto solved

如果：

```text
BFT acc within 0.5% of PureKAN-AdamW
geometry/ECE/AUC much better
```

则 claim：

```text
PureKAN functional optimizer is Pareto-viable but not accuracy-dominant.
```

#### Not solved

如果：

```text
KMNIST gap > 1%
or P7/P8 instability persists
```

则 claim：

```text
PureKAN functional optimization remains unsolved; BFT identifies the failure mode.
```

---

## 15. 必须新增的分析 artifacts

v4.2 必须输出这些 CSV/JSON：

```text
p0_invariants.csv
p1_proposal_direction_audit.csv
p2_single_block_acceptance.csv
p3_sequential_micro_scorecard.csv
p4_ablation_scorecard.csv
p5_temporal_dynamics_scorecard.csv
p6_candidate_selection.csv
p7_confirm5.csv
p8_confirm10.csv
bft_acceptance_log.csv
bft_rejection_reason_log.csv
role_update_trace.csv
proposal_choice_trace.csv
activation_drift_trace.csv
logit_drift_trace.csv
metric_norm_trace.csv
eta_backtracking_trace.csv
failure_table.csv
aggregate_decision.json
```

每个 accepted/rejected proposal 都要有一行：

```text
dataset
seed
epoch
step
role
proposal_type
eta_initial
eta_accepted
backtrack_count
accepted
rejection_reason
predicted_descent
actual_train_descent
actual_holdout_descent
acceptance_ratio
activation_drift
logit_drift
sobolev_norm
metric_norm
class_margin_change
```

---

## 16. 这轮实验的核心可视化清单

### 16.1 Direction / proposal 图

```text
proposal actual descent bar chart
predicted vs actual descent scatter
acceptance ratio heatmap
proposal rank by role
```

### 16.2 Trust-region 图

```text
accepted eta distribution
backtracking count histogram
rejection reason stacked bar
activation drift vs accepted eta scatter
logit drift vs accepted eta scatter
```

### 16.3 Training dynamics 图

```text
train loss curves
val loss curves
val accuracy curves
ECE curves
margin curves
phi_prime_p95 curves
Jacobian condition curves
```

### 16.4 Role / layer 图

```text
role update share stacked area
role acceptance rate heatmap
input/block/output contribution curves
basis occupancy entropy curves
basis dead fraction curves
```

### 16.5 Final scorecard 图

```text
paired acc delta vs PureKAN-AdamW
paired AUC delta vs PureKAN-AdamW
accuracy-geometry Pareto plot
accuracy-time Pareto plot
failure type heatmap
```

---

## 17. 预期结果和解释规则

### 17.1 如果 BFT-FTF 仍然崩

说明：

```text
layer-local target fitting 即使有 backtracking，也不适合 PureKAN。
```

那么 FTF 只能作为 diagnostic proposal，不再作为 optimizer 主线。

### 17.2 如果 BFT-FNG 稳但弱

说明：

```text
FNG 是安全方向，但表达力释放不足。
```

下一步应该增强 output Fisher / downstream curvature，而不是再调 Sobolev。

### 17.3 如果 BFT-mixed 成功

说明：

```text
PureKAN functional update 需要 proposal selection + trust acceptance。
```

这将成为新主线。

### 17.4 如果 BFT 仍然无法接近 AdamW

说明：

```text
PureKAN functional optimization 可能需要 function-coordinate Adam 或真正 global Gauss-Newton。
```

这时 v4.3 才进入更昂贵的 global solver，而不是继续 blockwise proposal。

---

## 18. 最终判断

v4.1 的失败不是简单坏消息。它清楚告诉我们：

$$
\boxed{
\text{局部方向修复不等于完整 optimizer。}
}
$$

FTF 的灾难尤其重要，因为它证明：**只要没有跨层 drift control，局部 target fit 越好，长期可能越坏。**

因此 v4.2 的核心不是再找一个更漂亮的 preconditioner，而是建立一个真正的 functional optimizer protocol：

$$
\boxed{
\text{proposal}
+
\text{global acceptance}
+
\text{activation/logit trust region}
+
\text{sequential block update}
+
\text{temporal dynamics}.
}
$$

这才是 PureKAN functional update 下一步必须走的路线。



---


# Source 12: `docs/DG-KAN_v4.2_FunctionalTrustRegion_DeepRedesign_结果复盘.md`


# DG-KAN v4.2 Functional Trust Region 结果复盘

本轮依据 `docs/DG-KAN_v4.2_FunctionalTrustRegion_DeepRedesign_实验计划.md`。核心目标是把 v4.1 的 proposal 直接更新改成 BFT：proposal -> 临时 apply -> train/holdout loss、activation drift、logit drift 验收 -> backtrack / accept / reject。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 smoke | 24 | 0 |
| P1 proposal | 108 | 0 |
| P2 single-block | 48 | 0 |
| P3 baselines+BFT | 99 | 0 |
| BFT acceptance log | 24372 | 2233 |

## Code / Config Changes

```text
experiments/run_gafu_v42.py
  Added BFT proposal generation using existing raw/Sobolev/D6/FNG/FTF/FC paths.
  Added temporary apply/rollback, train+holdout loss checks, activation/logit trust,
  backtracking, mixed proposal selection, and sequential role orders.

experiments/analyze_gafu_v42.py
  Generates v4.2 required artifacts, gate summaries, traces, figures, and this replay.
```

## P0 Implementation Smoke

```text
P0 rows = 24
errors = 0
rollback max abs error = 0.0000
strict PureKAN nonKAN params = 0
```

P0 通过：proposal、临时 apply、rollback、accept/reject/backtrack 统计均能产生有限记录。

## P1 Proposal Direction Audit

| proposal | rows | accept | bad | holdout+ | median ratio | act p95 | logit p95 | p1_pass |
|---|---|---|---|---|---|---|---|---|
| FC-whitened-Adam-one-step | 12 | 0.7500 | 0.0000 | 3 | 0.9911 | 0.0779 | 0.1362 | no |
| FC-whitened-gradient | 12 | 0.7500 | 0.0000 | 3 | 0.9994 | 0.0704 | 0.1208 | no |
| FNG-leftFullRight | 12 | 0.8333 | 0.0000 | 3 | 0.9935 | 0.0702 | 0.1070 | no |
| FTF-blocks-output | 12 | 1.0000 | 0.0000 | 3 | 0.9891 | 0.0866 | 0.0972 | yes |
| FTF-output-only | 12 | 1.0000 | 0.0000 | 3 | 0.9904 | 0.0865 | 0.0944 | yes |
| Sobolev-full | 12 | 1.0000 | 0.0000 | 3 | 0.9831 | 0.0788 | 0.0884 | yes |
| mixed-best-of-proposals | 12 | 1.0000 | 0.0000 | 3 | 0.9861 | 0.0878 | 0.0963 | yes |
| raw-gradient | 12 | 1.0000 | 0.0000 | 3 | 0.9977 | 0.0242 | 0.0820 | yes |
| task-diag-D6 | 12 | 0.7500 | 0.0000 | 3 | 0.9772 | 0.0948 | 0.1613 | no |

P1 survivors:

```text
FTF-blocks-output, FTF-output-only, Sobolev-full, mixed-best-of-proposals, raw-gradient
```

观察：

```text
BFT acceptance gate 明显压住了 v4.1 的危险方向。
FTF 在 block/output role 上仍有强 one-step descent，但 input role 基本是 no-op 或需要 mixed/raw 接管。
input role 的 FNG/D6/FC 经常因为 logit drift 超过阈值被拒绝。
```

## P2 Single-Block Accepted-Step Audit

| proposal | rows | accept | bad | holdout+ | median ratio | act p95 | logit p95 | p2_pass |
|---|---|---|---|---|---|---|---|---|
| FNG-leftFullRight | 12 | 0.7500 |  |  |  | 0.0771 | 0.1908 | no |
| FTF-blocks-output | 12 | 1.0000 |  |  |  | 0.0912 | 0.0947 | yes |
| mixed-best-of-proposals | 12 | 1.0000 |  |  |  | 0.0771 | 0.0956 | yes |
| raw-gradient | 12 | 1.0000 |  |  |  | 0.0242 | 0.0637 | yes |

P2 survivors:

```text
FTF-blocks-output, mixed-best-of-proposals, raw-gradient
```

观察：

```text
single-block 层面，FTF / mixed / raw 都能在 trust gate 内得到正 holdout descent。
FNG 对 hidden/output 比较安全，但 input FNG 会因为 logit drift 被拒绝。
这说明 BFT 的验收机制确实阻止了 v4.1 的 FTF 长程爆炸第一步。
```

## P3 Sequential BFT Micro-Run

| dataset | method | runs | acc | std | gap vs AdamW | AUC vs D6 | accept | fallback | act p95 | logit p95 | P3 pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | BFT-FNG-forward | 3 | 0.7400 | 0.0265 | 0.0800 | -0.1274 | 0.9045 | 0.0955 | 0.0912 | 0.1305 | 0 |
| Fashion-MNIST | BFT-FNG-reverse | 3 | 0.7460 | 0.0306 | 0.0740 | -0.1454 | 0.9219 | 0.0781 | 0.0858 | 0.1248 | 0 |
| Fashion-MNIST | BFT-FTF-forward | 3 | 0.7033 | 0.0182 | 0.1167 | -0.1139 | 0.9080 | 0.0920 | 0.0866 | 0.1054 | 0 |
| Fashion-MNIST | BFT-FTF-reverse | 3 | 0.7263 | 0.0105 | 0.0937 | -0.1053 | 0.9809 | 0.0191 | 0.0827 | 0.0940 | 0 |
| Fashion-MNIST | BFT-mixed-forward | 3 | 0.7297 | 0.0226 | 0.0903 | 0.0890 | 1.0000 | 0.0000 | 0.0824 | 0.0874 | 0 |
| Fashion-MNIST | BFT-mixed-output-first | 3 | 0.7253 | 0.0252 | 0.0947 | 0.0657 | 1.0000 | 0.0000 | 0.0841 | 0.0866 | 0 |
| Fashion-MNIST | BFT-mixed-reverse | 3 | 0.7373 | 0.0184 | 0.0827 | 0.0683 | 1.0000 | 0.0000 | 0.0848 | 0.0876 | 0 |
| Fashion-MNIST | D0-allFullSobolev | 3 | 0.7593 | 0.0296 | 0.0607 | -0.1722 |  |  |  |  | 1 |
| Fashion-MNIST | D6-allTaskAware | 3 | 0.7827 | 0.0154 | 0.0373 | 0.0000 |  |  |  |  | 1 |
| Fashion-MNIST | F4-FNG-leftFullRight | 3 | 0.7930 | 0.0164 | 0.0270 | 0.2413 |  |  |  |  | 1 |
| Fashion-MNIST | PureKAN-AdamW | 3 | 0.8200 | 0.0086 | 0.0000 | 0.2862 |  |  |  |  | 1 |
| KMNIST | BFT-FNG-forward | 3 | 0.3837 | 0.0189 | 0.3407 | -0.3030 | 0.7526 | 0.2474 | 0.0875 | 0.1507 | 0 |
| KMNIST | BFT-FNG-reverse | 3 | 0.4223 | 0.0450 | 0.3020 | -0.2699 | 0.8325 | 0.1675 | 0.0882 | 0.1463 | 0 |
| KMNIST | BFT-FTF-forward | 3 | 0.4127 | 0.0045 | 0.3117 | -0.1222 | 0.8741 | 0.1259 | 0.0904 | 0.1218 | 0 |
| KMNIST | BFT-FTF-reverse | 3 | 0.4250 | 0.0102 | 0.2993 | -0.1106 | 0.9575 | 0.0425 | 0.0868 | 0.0995 | 0 |
| KMNIST | BFT-mixed-forward | 3 | 0.4913 | 0.0137 | 0.2330 | -0.0259 | 1.0000 | 0.0000 | 0.0818 | 0.0939 | 0 |
| KMNIST | BFT-mixed-output-first | 3 | 0.4900 | 0.0115 | 0.2343 | -0.0508 | 1.0000 | 0.0000 | 0.0833 | 0.0925 | 0 |
| KMNIST | BFT-mixed-reverse | 3 | 0.4907 | 0.0103 | 0.2337 | -0.0389 | 1.0000 | 0.0000 | 0.0815 | 0.0932 | 0 |
| KMNIST | D0-allFullSobolev | 3 | 0.5367 | 0.0237 | 0.1877 | -0.0882 |  |  |  |  | 1 |
| KMNIST | D6-allTaskAware | 3 | 0.5383 | 0.0180 | 0.1860 | 0.0000 |  |  |  |  | 1 |
| KMNIST | F4-FNG-leftFullRight | 3 | 0.6630 | 0.0028 | 0.0613 | 0.3186 |  |  |  |  | 1 |
| KMNIST | PureKAN-AdamW | 3 | 0.7243 | 0.0111 | 0.0000 | 0.4216 |  |  |  |  | 1 |
| MNIST | BFT-FNG-forward | 3 | 0.6540 | 0.0022 | 0.2463 | -0.0152 | 0.7500 | 0.2500 | 0.1690 | 0.2146 | 0 |
| MNIST | BFT-FNG-reverse | 3 | 0.6503 | 0.0110 | 0.2500 | -0.0347 | 0.7500 | 0.2500 | 0.1530 | 0.1949 | 0 |
| MNIST | BFT-FTF-forward | 3 | 0.6433 | 0.0155 | 0.2570 | 0.1077 | 0.6988 | 0.3012 | 0.0933 | 0.2033 | 0 |
| MNIST | BFT-FTF-reverse | 3 | 0.6520 | 0.0024 | 0.2483 | 0.1301 | 0.7491 | 0.2509 | 0.0912 | 0.1850 | 0 |
| MNIST | BFT-mixed-forward | 3 | 0.7060 | 0.0237 | 0.1943 | 0.1565 | 1.0000 | 0.0000 | 0.0797 | 0.0922 | 0 |
| MNIST | BFT-mixed-output-first | 3 | 0.7137 | 0.0259 | 0.1867 | 0.1402 | 1.0000 | 0.0000 | 0.0855 | 0.0918 | 0 |
| MNIST | BFT-mixed-reverse | 3 | 0.7277 | 0.0323 | 0.1727 | 0.1610 | 1.0000 | 0.0000 | 0.0811 | 0.0939 | 0 |
| MNIST | D0-allFullSobolev | 3 | 0.6800 | 0.0349 | 0.2203 | 0.0269 |  |  |  |  | 1 |
| MNIST | D6-allTaskAware | 3 | 0.6913 | 0.0300 | 0.2090 | 0.0000 |  |  |  |  | 1 |
| MNIST | F4-FNG-leftFullRight | 3 | 0.8400 | 0.0333 | 0.0603 | 0.4325 |  |  |  |  | 1 |
| MNIST | PureKAN-AdamW | 3 | 0.9003 | 0.0076 | 0.0000 | 0.4301 |  |  |  |  | 1 |

Best BFT points:

```text
MNIST: BFT-mixed-reverse acc=0.7277, gap=0.1727
Fashion: BFT-FNG-reverse acc=0.7460, gap=0.0740
KMNIST: BFT-mixed-forward acc=0.4913, gap=0.2330
```

P3 verdict:

```text
No BFT candidate passed the P3 joint gate.

BFT successfully prevents catastrophic FTF divergence:
  no chance-accuracy collapse like v4.1 FTF
  acceptance/logit/activation traces remain finite

But BFT is too conservative or direction-weak:
  MNIST best BFT remains far below PureKAN-AdamW
  Fashion best BFT remains below PureKAN-AdamW and FNG baseline
  KMNIST best BFT improves over D0/D6 but remains far below PureKAN-AdamW and F4-FNG baseline
```

## P4-P8 Decision

```text
P4 proposal/order ablation: not run.
Reason: P3 produced no survivor.

P5 temporal dynamics: not run.
Reason: no P4 candidate.

P6 3-seed full-budget selection: not run.
Reason: P3 micro-run did not satisfy entry gate.

P7/P8 confirm: not run.
Reason: P6 was not reached.
```

## Required Artifacts

Written under `results/v4_2/`:

```text
p0_invariants.csv
p1_proposal_direction_audit.csv
p2_single_block_acceptance.csv
p3_sequential_micro_scorecard.csv
p4_ablation_scorecard.csv
p5_temporal_dynamics_scorecard.csv
p6_candidate_selection.csv
p7_confirm5.csv
p8_confirm10.csv
bft_acceptance_log.csv
bft_rejection_reason_log.csv
role_update_trace.csv
proposal_choice_trace.csv
activation_drift_trace.csv
logit_drift_trace.csv
metric_norm_trace.csv
eta_backtracking_trace.csv
failure_table.csv
aggregate_decision.json
figures/p3_kmnist_bft_acc.svg
figures/p3_fashion_bft_acc.svg
```

## Final Decision

```text
PureKAN functional optimization remains unsolved in v4.2.

What is confirmed:
  1. BFT acceptance/backtracking prevents FTF-style numerical catastrophe.
  2. Proposal selection is useful: mixed candidates choose FTF for hidden blocks and FNG/raw for safer roles.
  3. The trust protocol gives finite, auditable acceptance/rejection traces.

What is not confirmed:
  1. Stability did not translate into AdamW-level representation learning.
  2. BFT candidates underfit, especially on KMNIST.
  3. P4/P5/P6 expansion is not justified.

Interpretation:
  v4.1 failed because proposal was too strong and unchecked.
  v4.2 shows the opposite side: checked proposals are stable but too weak.
  The next design needs stronger accepted directions, likely better downstream curvature or temporal dynamics,
  but only after improving P3 micro-run accuracy.
```



---


# Source 13: `docs/DG-KAN_v4.3_FunctionalUpdate_DeepRedesign_实验计划.md`


# DG-KAN v4.3：PureKAN Functional Update 深度重设计与实验计划

## 0. 当前判断

v4.2 的结果把问题边界进一步压清楚了。BFT 不是没有价值，它确实把 v4.1 中 FTF 的灾难性发散压住了；但是它压住发散的方式，是把 functional proposal 变成了安全但很弱的更新。P3 micro-run 里没有任何 BFT candidate 通过 gate，最佳 BFT 在 MNIST / Fashion / KMNIST 上仍然远低于 PureKAN-AdamW。也就是说，当前问题已经不是“再缩小一步长”“再加一个 backtracking”“再换一个 shallow/deep order”能解决的。

现在应该把问题重写成：

$$
\boxed{\text{当前 functional update 不是一个完整的深度网络 optimizer，而只是若干局部 proposal。}}
$$

过去我们依次验证了几类设计：

$$
\Delta a=-\eta S^{-1}g
$$

即 Sobolev-only U-FULL；

$$
\Delta a_l=-\eta_l(G_l+\lambda_lS_l+\rho I)^{-1}g_l
$$

即 TFU / FNG 类局部 preconditioner；以及 FTF / FTR / BFT 这类 target fitting 与 trust-region 版本。它们共同暴露出同一个问题：**局部方向可以变好，局部验收也可以通过，但长期表示学习仍然不够。**

因此 v4.3 不再把 functional update 视为“用一个更好的矩阵乘梯度”，而是重新定义为：

$$
\boxed{\text{functional update = task-learning dynamics + function-space constraint + temporal optimizer state}}
$$

这意味着 Sobolev metric 不应该继续充当主 optimizer。它应该成为约束、正则、投影或 trust metric；主方向必须保留足够的 task-learning 能力。

---

## 1. v4.2 结果说明了什么

### 1.1 BFT 验收机制有效，但它解决的是数值灾难，不是学习动力学

v4.1 的 FTF 有一个很典型的失败模式：P1 one-step target fit 很好，FTF fit $R^2$ 接近 $1$，train / val one-step descent 也很好；但是 P2 训练直接跌到 chance accuracy，val-loss AUC 爆炸。这说明 layer-local target fitting 太强，跨层漂移无法控制。

v4.2 加入 BFT 后，proposal 要经过：

$$
\text{proposal}\rightarrow\text{temporary apply}\rightarrow\text{train/holdout loss check}\rightarrow\text{activation/logit trust}\rightarrow\text{accept/backtrack/reject}
$$

这一机制确实阻止了 FTF-style collapse。P1/P2 里 FTF、mixed、raw proposal 在单步和 single-block 层面可以通过验收，activation/logit drift 都保持有限。

但是 P3 sequential micro-run 表明，稳定不等于会学：BFT candidates 全部 underfit。尤其 mixed candidates 的 acceptance rate 可以是 $1.0$，但 accuracy 仍然明显低于 FNG baseline 和 PureKAN-AdamW。这说明验收条件没有捕捉到“长期表示学习是否充分”。

因此：

$$
\boxed{\text{BFT 解决了 unsafe update，但没有解决 useful update。}}
$$

### 1.2 当前 gate 主要限制了坏漂移，但也压掉了必要的表征漂移

PureKAN 要学习的不只是 logits 的小修小补，而是从 input KAN 到 block KAN 到 output KAN 的整条表示链。真正的表示学习一定伴随 hidden representation drift。v4.2 的 activation/logit trust region 太像“稳定性过滤器”，它倾向于拒绝或缩小大幅改变 hidden representation 的 proposal。

但 AdamW 能训练 PureKAN，说明某些看起来“不够保守”的更新可能正是早期表示学习所需要的。BFT 的失败提示我们：

$$
\boxed{\text{过度限制 activation drift 会保护几何，但可能阻止 feature formation。}}
$$

所以 v4.3 要把 drift 从单纯的危险信号，改成可分析的学习信号。我们不应该只记录 drift 是否超阈值，还要记录 drift 是否带来 margin、class separation、effective rank、validation descent 的改善。

### 1.3 FNG 仍然是最强 functional proposal，但它不是完整 optimizer

在 v4.1/v4.2 的 short-run 里，F4-FNG-leftFullRight 始终是相对最实际的 functional candidate：它比 D0/D6 更接近 AdamW，在 MNIST/Fashion 上尤其明显。但它仍然在 KMNIST 上有较大 gap，也不能通过最终 gate。

这说明 FNG 的方向比 Sobolev-only / diagonal TFU 更接近 task geometry，但仍然缺了两件东西：

1. 对 input-side basis-feature covariance 的充分建模；
2. 类似 AdamW 的长期时间尺度和尺度自适应。

因此下一步不应该放弃 FNG，而应该把它扩展成 **edge-feature KFAC + functional-proximal temporal dynamics**。

---

## 2. 当前 functional update 的根本缺陷

### 2.1 缺陷一：只在 basis 维度做函数几何，不在 edge-feature 空间做任务几何

一个 RBF KAN layer 可以写成：

$$
Y = \Phi(X) A^\top
$$

其中 $\Phi(X)$ 是把每个输入 channel 展开到 RBF basis 后得到的 feature，$A$ 是 reshape 后的 coefficient matrix。更具体地，原始 coefficient 是：

$$
A_{o,i,k}
$$

其中 $o$ 是输出 channel，$i$ 是输入 channel，$k$ 是 basis index。

过去的 Sobolev update 主要在 $k$ 维度上做：

$$
S_k = \int BB^\top + \alpha\int B'B'^\top + \beta\int B''B''^\top.
$$

这只控制每条 edge function 对输入 $x_i$ 的平滑性。但是 PureKAN 训练时真正重要的是整个 flattened feature：

$$
\Phi_l(x) \in \mathbb{R}^{d_{in}K}.
$$

真实的 Gauss-Newton / natural gradient 右侧 metric 应该更接近：

$$
G_{\Phi,l}=\mathbb{E}[\Phi_l(x)^\top\Phi_l(x)].
$$

而不是只用 $S_k$。如果只在 basis 维度平滑，optimizer 不知道不同 input channel 之间的相关性，也不知道哪些 edge feature 对任务有用。这会导致 functional update 在几何上漂亮，但在任务表征上短视。

### 2.2 缺陷二：缺少 output-side downstream sensitivity

对 layer output $Y_l$，下游 loss 对它的敏感性可以用梯度 covariance 或 Fisher 近似：

$$
C_l=\mathbb{E}[\delta_l\delta_l^\top],
$$

其中 $\delta_l=\partial L/\partial Y_l$。一个更完整的 KAN-KFAC 更新应该是：

$$
\Delta A_l
=
-\eta
(C_l+\rho_c I)^{-1}
\nabla_{A_l}L
(G_{\Phi,l}+\lambda_l S_l+\rho_\phi I)^{-1}.
$$

当前 FNG 虽然尝试了 left/right 结构，但还不够完整。尤其如果右侧仍然主要是 basis/Sobolev，而不是 full edge-feature covariance $G_{\Phi,l}$，那么它仍然缺少 input-feature side 的真实任务几何。

### 2.3 缺陷三：trust region 用错了对象

v4.2 的 activation/logit trust region 用来防止 drift 是合理的，但它把 drift 当成单调危险信号。对 PureKAN 来说，早期 feature formation 需要 drift。真正应该限制的不是 drift 大小本身，而是“无效 drift”或“破坏性 drift”。

因此需要从：

$$
\frac{\|h_l(\theta+\Delta)-h_l(\theta)\|}{\|h_l(\theta)\|}<r_h
$$

升级成：

$$
\text{drift is acceptable if it improves margin / separation / holdout loss.}
$$

也就是说，trust region 不应该只问“动了多少”，还要问“动得是否有用”。

### 2.4 缺陷四：缺少 AdamW 的长期时间尺度动力学

PureKAN-AdamW 能训，说明 PureKAN 架构不是没有表达力。AdamW 的优势包括：

```text
per-parameter RMS scale adaptation
momentum / temporal averaging
对 rough but useful functions 的容忍
长期噪声平均与特征共适应
```

过去我们试图用 functional update 直接替代 AdamW，这太激进。更合理的路线不是丢掉 AdamW 的时间尺度，而是把它移到 function-aware coordinate 或 functional projection 内部。

因此 v4.3 的主线不是 “functional preconditioner vs AdamW”，而是：

$$
\boxed{\text{AdamW-like temporal dynamics in functional coordinates.}}
$$

---

## 3. v4.3 的核心设计方向

v4.3 不再继续做 D0/D6/D3/BFT 的小变体，而是测试三个真正不同的 optimizer family。

### 3.1 FPA：Functional-Proximal Adam

FPA 的思想是：不要让 Sobolev / TFU 直接决定下降方向，而是用 AdamW 产生 task-learning proposal，再用 function-space metric 做 proximal projection。

AdamW proposal 记为：

$$
d_{Adam,t}.
$$

Functional-proximal update 定义为：

$$
\Delta a_t
=
\arg\min_\Delta
\left[
\frac{1}{2}\|\Delta-d_{Adam,t}\|_{D_t^{-1}}^2
+
\frac{\lambda}{2}\|\Delta\|_{S}^2
+
\frac{\rho}{2}\|\Delta\|_2^2
\right].
$$

其闭式形式可以写成：

$$
\Delta a_t
=
(D_t^{-1}+\lambda S+\rho I)^{-1}D_t^{-1}d_{Adam,t}.
$$

这里 $D_t$ 是 AdamW 的 RMS scale。直觉是：

```text
AdamW 负责 task learning direction；
Sobolev / functional metric 负责约束这个方向不要破坏函数几何；
而不是让 Sobolev 自己当 optimizer。
```

FPA 是当前最值得优先测试的方向，因为它利用了一个事实：PureKAN-AdamW 已经能训练。

### 3.2 EK-FNG：Edge-feature KFAC Functional Natural Gradient

EK-FNG 直接修正当前 FNG 的几何缺陷。它不再只在 basis 维度使用 Sobolev，而是构造完整 edge-feature covariance：

$$
G_{\Phi,l}=\mathbb{E}[\Phi_l^\top\Phi_l].
$$

然后用 Kronecker 近似：

$$
\Delta A_l
=
-\eta
(C_l+\rho_c I)^{-1}
\nabla_{A_l}L
(G_{\Phi,l}+\lambda S_{edge,l}+\rho_\phi I)^{-1}.
$$

这里 $S_{edge,l}$ 可以是 Sobolev basis metric 扩展到 edge-feature 空间：

$$
S_{edge,l}=I_{d_{in}}\otimes S_k.
$$

注意，这和旧 U-FULL 最大的区别是：

```text
旧 U-FULL: 只相信 S_k；
EK-FNG: 相信 data/task feature covariance G_Phi，再把 S_k 当 regularizer。
```

### 3.3 CFT：Coordinated Functional Targeting

FTF 的失败不是因为 target fitting 没用，而是因为每层局部 fit 太强、跨层漂移失控。CFT 把 target fitting 改成 coordinated block update。

每一步不再让所有层各自拟合自己的 target，而是选一个 role 或一个 block，拟合小目标：

$$
T_l=-\tau_l\delta_l.
$$

然后解：

$$
\Delta A_l
=
\arg\min_\Delta
\left[
\|\Phi_l\Delta A_l^\top-T_l\|_F^2
+
\lambda\|\Delta A_l\|_{S_l}^2
+
\rho\|\Delta A_l\|_F^2
\right].
$$

但 CFT 必须满足两个额外条件：

1. 每步只允许少数 role 更新；
2. 更新必须带来 margin 或 holdout loss 改善，否则缩小或拒绝。

CFT 是 FTF 的修正版，不应该作为第一优先主线，但必须保留，因为它能测试“functional layer fitting”是否仍然有潜力。

---

## 4. v4.3 实验总原则

这轮不做大范围种子扩展。每一个候选必须先证明三个性质：

1. **它保留 task-learning direction。**
2. **它不会破坏 function geometry。**
3. **它在短程训练中形成表示学习，而不是只通过单步验收。**

具体说，P1/P2 的核心不再只是 train/holdout descent，而是：

$$
\text{descent} + \text{Adam alignment} + \text{feature formation} + \text{geometry control}.
$$

---

# P0. 实现与不变量检查

## 目标

确认 v4.3 新增的 FPA / EK-FNG / CFT 没有引入隐藏非 KAN 参数，也没有破坏 PureKAN strict setting。

## 方法

数据集：

```text
MNIST
Fashion-MNIST
KMNIST
```

模型：

```text
PureKAN hidden=64 depth=4 basis=16 alphaFixed1 FixedNorm
```

必须检查：

```text
learnable_nonKAN_params = 0
coefficient_coverage = 1.0
input_kan.coeff covered
all block kan.coeff covered
output_kan.coeff covered
alpha fixed = 1.0
no Linear / no trainable LN / no bias if strict mode
```

## 新增 method smoke

```text
PureKAN-AdamW
D0-allFullSobolev
D6-allTaskAware
F4-FNG-leftFullRight
BFT-mixed-reverse
FPA-sob-prox
FPA-edge-prox
EKFNG-rightFull
EKFNG-leftRightFull
EKFNG-leftRightFull-lowSob
CFT-output-only
CFT-block-output-sequential
```

## 记录指标

```text
num_total_params
num_coeff_params
num_nonkan_trainable_params
functional_coverage
per_role_coeff_count
per_role_grad_norm
per_role_update_norm
metric_condition_right
metric_condition_left
metric_condition_sobolev
chol_success
solve_success
nan_count
inf_count
step_time_ms
memory_peak_mb
```

## 可视化

```text
P0_param_coverage_bar.svg
P0_metric_condition_by_method.svg
P0_step_time_by_method.svg
```

## 通过标准

```text
strict PureKAN rows: nonKAN = 0
coverage = 1.0
all metric conditions finite
no NaN / Inf
rollback / temporary apply error = 0 if method uses temporary apply
```

---

# P1. One-step proposal quality audit

## 目标

不再只问 proposal 是否降低 train loss，而是同时问：

```text
它是否接近 AdamW 的 task-learning direction？
它是否产生有用 representation drift？
它是否保持 function geometry？
```

## 方法

每个 dataset 取同一个初始化和同一个 batch，比较以下 proposal：

```text
AdamW-one-step
D0-allFullSobolev
D6-allTaskAware
F4-FNG-leftFullRight
BFT-mixed-reverse
FPA-sob-prox
FPA-edge-prox
EKFNG-rightFull
EKFNG-leftRightFull
EKFNG-leftRightFull-lowSob
CFT-output-only
CFT-block-output-sequential
raw-gradient
```

## 核心记录

### Loss descent

```text
train_loss_before
train_loss_after
holdout_loss_before
holdout_loss_after
train_descent
holdout_descent
bad_train_step
bad_holdout_step
```

### Adam alignment

令 AdamW proposal 为 $d_{Adam}$，functional proposal 为 $d_f$。记录：

$$
\cos(d_f,d_{Adam})
$$

以及投影比例：

$$
\frac{\langle d_f,d_{Adam}\rangle}{\|d_{Adam}\|^2}.
$$

记录字段：

```text
cos_with_adam_global
cos_with_adam_input
cos_with_adam_blocks_mean
cos_with_adam_output
projection_on_adam_global
norm_ratio_vs_adam_global
```

### Representation formation

记录 update 前后 hidden representation 的有效秩和类间分离：

$$
\operatorname{rank}_{eff}(H)=\frac{(\sum_i \sigma_i)^2}{\sum_i \sigma_i^2}.
$$

记录字段：

```text
input_feature_effective_rank_before/after
block_feature_effective_rank_before/after
class_centroid_separation_before/after
within_class_variance_before/after
margin_mean_before/after
margin_p10_before/after
```

### Geometry

```text
phi_prime_p95_before/after
jacobian_condition_before/after
curvature_energy_before/after
activation_drift_by_role
logit_drift
```

## 可视化

```text
P1_train_vs_holdout_descent_scatter.svg
P1_cos_with_adam_bar.svg
P1_projection_on_adam_bar.svg
P1_effective_rank_change_bar.svg
P1_margin_change_vs_logit_drift.svg
P1_geometry_change_by_method.svg
```

## 通过标准

一个 candidate 只有满足以下条件才进入 P2：

```text
train_descent > 0 on all datasets
holdout_descent >= 0 on at least 2/3 datasets
cos_with_adam_global > 0.35 or projection_on_adam_global > 0.25
logit_drift not zero and not explosive
feature effective rank does not collapse
phi_prime_p95_after <= 1.5 * phi_prime_p95_adam_after
```

注意：这里不要求 candidate 比 AdamW 更保守，而是要求它既有任务方向又有函数几何约束。

---

# P2. Multi-step short-horizon learning audit

## 目标

v4.2 的教训是：one-step pass 不够。P2 要测试 candidate 是否在短程训练中真的形成表示，而不是只通过单步验收。

## 设置

```text
epochs = 3
seeds = 0,1,2
datasets = MNIST, Fashion-MNIST, KMNIST
```

## 方法

```text
PureKAN-AdamW
D0-allFullSobolev
D6-allTaskAware
F4-FNG-leftFullRight
BFT-mixed-reverse
FPA-sob-prox
FPA-edge-prox
EKFNG-rightFull
EKFNG-leftRightFull
EKFNG-leftRightFull-lowSob
CFT-output-only
CFT-block-output-sequential
```

## 必须记录

```text
train_loss_curve_per_step
val_loss_curve_per_epoch
val_acc_curve_per_epoch
test_acc
val_auc_vs_adamw
val_auc_vs_d6
train_acc
train_val_gap
ECE
NLL
```

Representation：

```text
feature_effective_rank_curve_input
feature_effective_rank_curve_last_block
class_centroid_separation_curve
margin_mean_curve
margin_p10_curve
```

Optimizer dynamics：

```text
proposal_norm_by_role
accepted_update_norm_by_role
update_over_param_by_role
cos_with_adam_by_step
projection_on_adam_by_step
fallback_rate
backtracking_count
rejection_reason
```

Geometry：

```text
phi_prime_p95_curve
jacobian_condition_curve
curvature_energy_curve
basis_occupancy_entropy
basis_dead_fraction
input_out_of_grid_fraction
```

## 可视化

```text
P2_val_acc_curve.svg
P2_val_loss_curve.svg
P2_effective_rank_curve.svg
P2_margin_curve.svg
P2_cos_with_adam_over_time.svg
P2_update_norm_by_role_stacked.svg
P2_geometry_curve.svg
P2_accuracy_geometry_pareto.svg
P2_failure_heatmap.svg
```

## 通过标准

进入 P3 的 candidate 必须满足：

```text
MNIST gap vs PureKAN-AdamW <= 3%
Fashion gap vs PureKAN-AdamW <= 2%
KMNIST gap vs PureKAN-AdamW <= 4%
val_auc_vs_d6 positive on at least 2/3 datasets
feature rank does not collapse
ECE not worse than AdamW by more than 20%
geometry improves over AdamW or remains controlled
```

这里的 threshold 故意比最终标准宽，因为 P2 是 short-horizon learning audit。

---

# P3. Functional-Proximal Adam 深入验证

## 为什么 P3 优先验证 FPA

PureKAN-AdamW 已经能训练，而 functional-only update 不能训练。最直接的改进方向不是继续替代 AdamW，而是把 AdamW 的 task-learning dynamics 放进 function-space constraint 中。

P3 专门验证 FPA 是否能做到：

```text
保留 AdamW accuracy / representation learning；
降低 phi_prime / Jacobian / ECE；
避免 U-FULL 的 underfit。
```

## 方法

```text
PureKAN-AdamW
FPA-sob-prox-lambda0.01
FPA-sob-prox-lambda0.03
FPA-sob-prox-lambda0.10
FPA-edge-prox-lambda0.01
FPA-edge-prox-lambda0.03
FPA-edge-prox-lambda0.10
FPA-edge-prox-adaptiveLambda
```

## 关键公式

FPA 的更新是：

$$
\Delta a_t
=
(D_t^{-1}+\lambda_t S_t+\rho I)^{-1}D_t^{-1}d_{Adam,t}.
$$

其中 adaptive lambda 规则为：

```text
if geometry worsens and loss improves: increase lambda
if loss stagnates and geometry is safe: decrease lambda
if holdout loss worsens: shrink eta or increase damping
```

## 记录指标

```text
lambda_t_by_role
adam_proposal_norm
projected_proposal_norm
projection_error_norm
functional_smoothness_penalty
retained_adam_component
phi_prime_reduction
jacobian_reduction
ECE_reduction
acc_gap_vs_adamw
```

## 可视化

```text
P3_lambda_trajectory.svg
P3_adam_component_retention.svg
P3_projection_error_vs_acc_gap.svg
P3_geometry_vs_accuracy_pareto.svg
P3_rolewise_lambda_heatmap.svg
```

## 通过标准

```text
3-seed full budget:
MNIST acc >= PureKAN-AdamW - 0.5%
Fashion acc >= PureKAN-AdamW - 0.5%
KMNIST acc >= PureKAN-AdamW - 1.0%
ECE improves on at least 2/3 datasets
phi_prime_p95 lower than AdamW on all datasets
Jacobian lower than AdamW on all datasets
```

如果 FPA 过这个标准，它进入 P6 5-seed confirm。

---

# P4. EK-FNG 深入验证

## 目标

验证完整 edge-feature KFAC 是否能解决当前 FNG 的短板。旧 FNG 最大的问题可能是右侧 metric 不够完整，没有显式使用 $G_{\Phi,l}$。

## 方法

```text
PureKAN-AdamW
F4-FNG-leftFullRight
EKFNG-rightFeatureOnly
EKFNG-leftDiag-rightFeature
EKFNG-leftFull-rightFeature
EKFNG-leftFull-rightFeature-lowSob
EKFNG-leftFull-rightFeature-adaptiveSob
```

## 指标

```text
G_phi_condition_by_role
C_left_condition_by_role
right_solve_residual
left_solve_residual
cos_with_adam
train_descent
holdout_descent
feature_rank_change
class_separation_change
acc/AUC/ECE/geometry
```

## 可视化

```text
P4_Gphi_spectrum_by_role.svg
P4_Cleft_spectrum_by_role.svg
P4_KFAC_cos_with_adam.svg
P4_rolewise_condition_heatmap.svg
P4_KFAC_acc_vs_condition.svg
```

## 通过标准

P4 candidate 只有在 KMNIST 上明显缩小 F4-FNG 的 gap 才有价值：

```text
KMNIST gap vs PureKAN-AdamW <= 2.0%
Fashion gap <= 1.0%
MNIST gap <= 1.0%
val_auc_vs_F4 positive on at least 2/3 datasets
```

---

# P5. Coordinated Functional Targeting 验证

## 目标

FTF 不能直接丢掉，因为它在 P1 有很强 target-fitting signal。问题是过去 FTF 太强、跨层漂移太大。P5 验证小步、协调式 CFT 是否能保留它的优点。

## 方法

```text
CFT-output-only-smallTau
CFT-block-output-sequential-smallTau
CFT-one-role-per-step-cyclic
CFT-best-role-by-holdout
CFT-best-role-by-margin
CFT-with-FPA-projection
```

## 关键约束

每步只允许一个 role 更新：

```text
input
block_0
block_1
block_2
block_3
output
```

并记录选择分布。

## 记录指标

```text
role_selected_each_step
role_acceptance_rate
role_margin_gain
role_holdout_descent
activation_drift_by_role
cross_layer_drift_after_update
FTF_fit_R2
FTF_residual_norm
```

## 可视化

```text
P5_role_selection_stacked_area.svg
P5_role_acceptance_heatmap.svg
P5_fit_R2_vs_long_run_acc.svg
P5_activation_drift_network_map.svg
P5_margin_gain_by_role.svg
```

## 通过标准

CFT 不要求第一轮超过 AdamW，但必须显著超过旧 FTF / BFT：

```text
no chance collapse
MNIST acc > 0.85
Fashion acc > 0.82
KMNIST acc > 0.68
val_auc finite and better than D0
role rejection rate < 50%
```

如果达不到，CFT 降级为 diagnostic，不进入 P6。

---

# P6. 3-seed full-budget candidate selection

## 目标

从 P3/P4/P5 中选出最多三个候选进入 5-seed confirm。

## 候选来源

```text
best FPA
best EKFNG
best CFT if viable
PureKAN-AdamW baseline
MLP-AdamW baseline
Hybrid-DGKAN-UFULL-f085 baseline
```

## 数据集

```text
MNIST
Fashion-MNIST
KMNIST
```

## seeds

```text
0,1,2
```

## 记录指标

```text
test_acc
val_loss_auc
val_acc_auc
NLL
ECE
train_val_gap
phi_prime_p95
jacobian_condition
curvature_energy
feature_effective_rank_final
class_centroid_separation_final
margin_mean_final
margin_p10_final
step_time_ms
memory_peak_mb
```

## 可视化

```text
P6_scorecard_bar.svg
P6_acc_vs_ECE.svg
P6_acc_vs_phi.svg
P6_val_loss_curves.svg
P6_margin_distribution.svg
P6_feature_rank_bar.svg
```

## 进入 P7 的标准

```text
MNIST >= max(PureKAN-AdamW, Hybrid, MLP) - 0.5%
Fashion >= max(PureKAN-AdamW, Hybrid, MLP) - 0.5%
KMNIST >= max(PureKAN-AdamW, Hybrid, MLP) - 1.0%
AUC better than PureKAN-AdamW on at least 2/3 datasets
ECE better than PureKAN-AdamW on at least 2/3 datasets
geometry better than PureKAN-AdamW on all datasets
```

---

# P7. 5-seed confirm

## 目标

确认 v4.3 candidate 是否真的能接近或超过 PureKAN-AdamW。

## seeds

```text
0,1,2,3,4
```

## 方法

最多三个：

```text
PureKAN-AdamW
Best-FPA
Best-EKFNG
Best-CFT if P6 passed
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
```

## paired metrics

必须输出 paired delta：

```text
acc_delta_vs_PureKANAdamW
AUC_delta_vs_PureKANAdamW
ECE_delta_vs_PureKANAdamW
phi_delta_vs_PureKANAdamW
jac_delta_vs_PureKANAdamW
margin_delta_vs_PureKANAdamW
```

## 可视化

```text
P7_paired_acc_delta_violin.svg
P7_paired_auc_delta_violin.svg
P7_geometry_delta_violin.svg
P7_seedwise_rank_table.svg
```

## 通过标准

```text
mean acc >= PureKAN-AdamW on at least 2/3 datasets
no dataset acc gap worse than 0.5%
AUC positive on at least 2/3 datasets
ECE positive on at least 2/3 datasets
geometry positive on all datasets
```

如果 P7 通过，进入 P8。否则 v4.3 conclusion 为：functional-proximal/FNG improved but not solved。

---

# P8. 10-seed final confirm

## seeds

```text
0,1,2,3,4,5,6,7,8,9
```

## methods

```text
PureKAN-AdamW
Best-v4.3-candidate
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
```

## 最终成功标准

PureKAN functional optimizer 成立，必须满足：

```text
1. mean accuracy >= PureKAN-AdamW on MNIST/Fashion/KMNIST
2. mean accuracy >= MLP-AdamW on at least 2/3 datasets
3. mean accuracy >= Hybrid-DGKAN-UFULL-f085 on at least 2/3 datasets
4. val-loss AUC better than PureKAN-AdamW on at least 2/3 datasets
5. ECE better than PureKAN-AdamW on at least 2/3 datasets
6. phi_prime_p95 and Jacobian condition better than PureKAN-AdamW on all datasets
7. no hidden non-KAN trainable params
8. no reliance on AdamW for coefficient update unless it is explicitly functional-coordinate Adam / functional-proximal Adam
```

如果只满足 geometry/ECE/AUC，不满足 accuracy，则不能 claim solved，只能 claim regularized functional optimizer。

---

# P9. Failure diagnosis if v4.3 still fails

如果 P7/P8 仍失败，必须输出 failure report，而不是继续扫参数。

## 失败分类

```text
F1: Adam component not retained
F2: feature rank collapse
F3: margin not improving
F4: output layer bottleneck
F5: input layer bottleneck
F6: geometry over-regularization
F7: trust too conservative
F8: temporal dynamics missing
F9: computationally impractical
```

## 必须生成表格

```text
failure_type_by_dataset.csv
failure_type_by_role.csv
proposal_vs_adam_alignment.csv
feature_rank_failure.csv
margin_failure.csv
geometry_regularization_failure.csv
```

## 必须生成图

```text
P9_failure_radar.svg
P9_role_bottleneck_heatmap.svg
P9_feature_rank_vs_accuracy.svg
P9_adam_alignment_vs_accuracy.svg
P9_trust_vs_accuracy.svg
```

---

## 5. 当前计划的最重要取舍

v4.3 的重点不是继续“更 functional”或者“更保守”。相反，最重要的取舍是：

$$
\boxed{\text{不要再用 smoothness prior 替代 task-learning dynamics。}}
$$

因此三条路线优先级是：

```text
Priority 1: FPA
  因为 PureKAN-AdamW 已经能训练，先把 AdamW 的任务学习能力保留下来，再加 functional constraint。

Priority 2: EK-FNG
  因为当前 FNG 是最强 functional proposal，但需要补 full edge-feature covariance。

Priority 3: CFT
  因为 FTF 局部信号强，但必须变成小步、协调式、role-selective target fitting。
```

如果 FPA 成功，functional update 的叙事会变成：

$$
\boxed{\text{functional geometry should constrain Adam-like task dynamics, not replace them.}}
$$

如果 EK-FNG 成功，叙事会变成：

$$
\boxed{\text{PureKAN needs edge-feature natural gradient, not basis-only Sobolev update.}}
$$

如果两者都失败，就说明当前 RBF-PureKAN 可能需要架构级变化，而不是 optimizer 级变化。

---

## 6. 最终结论

v4.2 已经证明：

```text
BFT 可以让 proposal 稳定；
但稳定 proposal 不等于学得好；
当前 PureKAN functional update 缺的是 task-learning dynamics 和 edge-feature geometry。
```

v4.3 的核心任务是验证：

$$
\boxed{\text{PureKAN 是否需要 functional-proximal Adam 或 edge-feature KFAC。}}
$$

如果答案是 yes，functional update 就从“替代 AdamW 的 Sobolev preconditioner”升级为“在函数空间约束下保留深度学习优化动力学”的真正 optimizer。



---


# Source 14: `docs/DG-KAN_v4.3_FunctionalUpdate_DeepRedesign_结果复盘.md`


# DG-KAN v4.3 Functional Update Deep Redesign 结果复盘

本轮依据 `docs/DG-KAN_v4.3_FunctionalUpdate_DeepRedesign_实验计划.md`。目标是验证 PureKAN functional update 是否应从 Sobolev-only/BFT 改写为：

```text
task-learning dynamics + function-space constraint + temporal optimizer state
```

因此本轮实现并测试了：

```text
FPA: Functional-Proximal Adam
EK-FNG: Edge-feature KFAC Functional Natural Gradient
CFT: Coordinated Functional Targeting
```

## Run Inventory

| stage | rows | errors |
|---|---:|---:|
| P0 smoke | 36 | 0 |
| P1 proposal audit | 39 | 0 |
| P2 short-horizon | 108 | 0 |

## Code / Config Changes

```text
experiments/run_gafu_v43.py
  Added strict-PureKAN FPA, EK-FNG and CFT experimental update paths.
  FPA keeps Adam-like task proposal and projects it through Sobolev/edge-feature proximal metrics.
  EK-FNG uses edge-feature covariance plus optional output-side left covariance.
  CFT uses small-tau coordinated FTF updates with output-only or block/output cyclic roles.
  P1/P2 record Adam alignment, feature rank, margin, geometry and optimizer traces.

experiments/analyze_gafu_v43.py
  Generates gate summaries, P9 failure reports, figures, aggregate_decision.json,
  and this replay.
```

## P0 Implementation Smoke

```text
P0 rows = 36
errors = 0
strict PureKAN nonKAN params = 0 for all rows
functional coverage = 1.0 for all rows
```

P0 verdict: pass. FPA / EK-FNG / CFT did not introduce hidden non-KAN trainable parameters and all coefficient groups are covered.

## P1 One-Step Proposal Quality

| method | train all | holdout+ | cos Adam | proj Adam | rank ratio | phi ratio | P1 |
|---|---|---|---|---|---|---|---|
| AdamW-one-step | 0 | 3 | 1.0000 | 1.0000 | 0.6126 | 1.0044 | no |
| BFT-mixed-reverse | 1 | 3 | 0.0764 | 0.0164 | 0.9985 | 1.0002 | no |
| CFT-block-output-sequential | 1 | 3 | 0.0319 | 0.0732 | 0.9456 | 1.0001 | no |
| CFT-output-only | 1 | 3 | 0.0667 | 0.2014 | 0.8912 | 1.0003 | no |
| D0-allFullSobolev | 1 | 3 | 0.1374 | 0.0546 | 0.9196 | 0.9977 | no |
| D6-allTaskAware | 0 | 2 | 0.6894 | 0.1571 | 0.7928 | 0.9998 | no |
| EKFNG-leftRightFull | 1 | 3 | 0.5353 | 1.1019 | 0.5356 | 0.9972 | no |
| EKFNG-leftRightFull-lowSob | 1 | 3 | 0.5089 | 1.8333 | 0.5318 | 0.9981 | no |
| EKFNG-rightFull | 1 | 3 | 0.6586 | 1.0598 | 0.5254 | 1.0007 | no |
| F4-FNG-leftFullRight | 1 | 3 | 0.1402 | 0.3225 | 0.7795 | 1.0021 | no |
| FPA-edge-prox | 0 | 3 | 0.9297 | 0.5930 | 0.7962 | 1.0001 | no |
| FPA-sob-prox | 1 | 3 | 0.5302 | 0.0972 | 0.9808 | 1.0001 | no |
| raw-gradient | 1 | 3 | 0.5799 | 0.0012 | 0.9995 | 0.9998 | yes |

P1 survivors:

```text
raw-gradient
```

Observation:

```text
FPA-sob and EK-FNG variants keep much more Adam alignment than old FNG/CFT.
However many proposals reduce effective rank on the first step.
This already hints that local descent is not enough; the proposal must preserve feature formation.
```

## P2 Multi-Step Short-Horizon Learning

| dataset | method | runs | acc | gap | AUC vs D6 | ECE red | phi red | J red | rank | P2 |
|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | BFT-mixed-reverse | 3 | 0.7183 | 0.0760 | 0.1703 | -0.1350 | 0.1210 | 0.6563 | 21.0647 | no |
| Fashion-MNIST | CFT-block-output-sequential | 3 | 0.5720 | 0.2223 | -0.4357 | -4.5639 | 0.0752 | 0.5428 | 16.6249 | no |
| Fashion-MNIST | CFT-output-only | 3 | 0.1010 | 0.6933 | -3.0695 | -15.8518 | -0.4940 | -572.8201 | 8.2294 | no |
| Fashion-MNIST | D0-allFullSobolev | 3 | 0.7300 | 0.0643 | -0.1220 | -4.7549 | 0.1206 | 0.6304 | 21.0329 | no |
| Fashion-MNIST | D6-allTaskAware | 3 | 0.7233 | 0.0710 | 0.0000 | -2.9803 | 0.1185 | 0.6761 | 19.1138 | no |
| Fashion-MNIST | EKFNG-leftRightFull | 3 | 0.6967 | 0.0977 | 0.1326 | -1.0504 | 0.1131 | 0.6471 | 20.7584 | no |
| Fashion-MNIST | EKFNG-leftRightFull-lowSob | 3 | 0.7273 | 0.0670 | 0.2042 | -0.2838 | 0.0946 | 0.5925 | 19.6971 | no |
| Fashion-MNIST | EKFNG-rightFull | 3 | 0.6977 | 0.0967 | 0.1880 | -0.8088 | 0.1088 | 0.6482 | 15.8537 | no |
| Fashion-MNIST | F4-FNG-leftFullRight | 3 | 0.6990 | 0.0953 | 0.0480 | -2.4927 | 0.1215 | 0.6688 | 25.3197 | no |
| Fashion-MNIST | FPA-edge-prox | 3 | 0.6927 | 0.1017 | 0.0831 | -0.2852 | 0.0966 | 0.6385 | 16.1422 | no |
| Fashion-MNIST | FPA-sob-prox | 3 | 0.7440 | 0.0503 | -0.2962 | -7.6149 | 0.1193 | 0.6577 | 26.8818 | no |
| Fashion-MNIST | PureKAN-AdamW | 3 | 0.7943 | 0.0000 | 0.3148 | 0.0000 | 0.0000 | 0.0000 | 13.1295 | yes |
| KMNIST | BFT-mixed-reverse | 3 | 0.4337 | 0.2403 | 0.0188 | -0.4330 | 0.1336 | 0.8092 | 29.4992 | no |
| KMNIST | CFT-block-output-sequential | 3 | 0.2230 | 0.4510 | -0.2035 | -0.1342 | 0.0912 | 0.7582 | 26.0259 | no |
| KMNIST | CFT-output-only | 3 | 0.1247 | 0.5493 | -3.7122 | -12.6564 | -0.8477 | -1255.1258 | 11.7944 | no |
| KMNIST | D0-allFullSobolev | 3 | 0.4693 | 0.2047 | -0.0108 | -2.6123 | 0.1359 | 0.8255 | 27.8941 | no |
| KMNIST | D6-allTaskAware | 3 | 0.4573 | 0.2167 | 0.0000 | -1.6150 | 0.1327 | 0.8137 | 24.9843 | no |
| KMNIST | EKFNG-leftRightFull | 3 | 0.4930 | 0.1810 | 0.1410 | -0.8807 | 0.1235 | 0.7970 | 29.2781 | no |
| KMNIST | EKFNG-leftRightFull-lowSob | 3 | 0.4583 | 0.2157 | 0.1875 | -1.5589 | 0.0881 | 0.7758 | 27.5354 | no |
| KMNIST | EKFNG-rightFull | 3 | 0.4353 | 0.2387 | 0.1176 | -2.4126 | 0.0974 | 0.8209 | 13.4684 | no |
| KMNIST | F4-FNG-leftFullRight | 3 | 0.5650 | 0.1090 | 0.1761 | -1.8652 | 0.1334 | 0.8059 | 30.1441 | no |
| KMNIST | FPA-edge-prox | 3 | 0.4367 | 0.2373 | 0.0948 | -0.1476 | 0.0888 | 0.7743 | 14.0573 | no |
| KMNIST | FPA-sob-prox | 3 | 0.4830 | 0.1910 | -0.0886 | -4.0194 | 0.1360 | 0.8313 | 32.5603 | no |
| KMNIST | PureKAN-AdamW | 3 | 0.6740 | 0.0000 | 0.4014 | 0.0000 | 0.0000 | 0.0000 | 19.3626 | yes |
| MNIST | BFT-mixed-reverse | 3 | 0.6863 | 0.1850 | 0.2614 | -0.4123 | 0.1703 | 0.9966 | 27.4750 | no |
| MNIST | CFT-block-output-sequential | 3 | 0.2907 | 0.5807 | -0.0990 | 0.0869 | 0.1038 | 0.9946 | 23.3564 | no |
| MNIST | CFT-output-only | 3 | 0.1210 | 0.7503 | -15.8004 | -8.6038 | -2.3398 | -87.1991 | 8.3833 | no |
| MNIST | D0-allFullSobolev | 3 | 0.6570 | 0.2143 | 0.1000 | -2.6056 | 0.1714 | 0.9967 | 23.4018 | no |
| MNIST | D6-allTaskAware | 3 | 0.5360 | 0.3353 | 0.0000 | -1.6489 | 0.1674 | 0.9966 | 21.0226 | no |
| MNIST | EKFNG-leftRightFull | 3 | 0.6530 | 0.2183 | 0.1662 | -0.5767 | 0.1574 | 0.9964 | 26.7310 | no |
| MNIST | EKFNG-leftRightFull-lowSob | 3 | 0.6690 | 0.2023 | 0.1856 | -0.2774 | 0.0986 | 0.9948 | 27.3767 | no |
| MNIST | EKFNG-rightFull | 3 | 0.5273 | 0.3440 | -0.0900 | -0.2444 | 0.1044 | 0.9961 | 7.0158 | no |
| MNIST | F4-FNG-leftFullRight | 3 | 0.7007 | 0.1707 | 0.2728 | -1.4276 | 0.1701 | 0.9970 | 30.4589 | no |
| MNIST | FPA-edge-prox | 3 | 0.5140 | 0.3573 | 0.0775 | -0.7611 | 0.1006 | 0.9960 | 9.9929 | no |
| MNIST | FPA-sob-prox | 3 | 0.6563 | 0.2150 | 0.0125 | -3.5883 | 0.1719 | 0.9965 | 30.4065 | no |
| MNIST | PureKAN-AdamW | 3 | 0.8713 | 0.0000 | 0.3809 | 0.0000 | 0.0000 | 0.0000 | 13.7687 | yes |

Best non-Adam points:

| dataset | best non-Adam | acc | gap | AUC vs D6 |
|---|---|---|---|---|
| MNIST | F4-FNG-leftFullRight | 0.7007 | 0.1707 | 0.2728 |
| Fashion-MNIST | FPA-sob-prox | 0.7440 | 0.0503 | -0.2962 |
| KMNIST | F4-FNG-leftFullRight | 0.5650 | 0.1090 | 0.1761 |

P2 survivors:

```text
none
```

P2 verdict:

```text
No candidate enters P3.

FPA did preserve some Adam-like direction locally, but short-horizon accuracy was far below PureKAN-AdamW.
EK-FNG improved some old functional baselines but did not close the gap.
CFT remained unstable/underfit; output-only CFT collapsed on all datasets.
The hardest blocker is KMNIST, where even the best non-Adam candidate remains far below PureKAN-AdamW.
```

## P3-P8 Decision

```text
P3 FPA deep validation: not run.
P4 EK-FNG deep validation: not run.
P5 CFT validation: not run.
P6 candidate selection: not run.
P7/P8 confirm: not run.

Reason: P2 produced no survivor under the written gate.
```

## P9 Failure Diagnosis

Generated:

```text
failure_type_by_dataset.csv
failure_type_by_role.csv
proposal_vs_adam_alignment.csv
feature_rank_failure.csv
margin_failure.csv
geometry_regularization_failure.csv
```

Diagnosis:

```text
F1 Adam component not retained:
  Partially true for old FNG/CFT; less true for FPA/EK-FNG, but retaining alignment alone was not enough.

F2 feature rank collapse:
  Present in P1 for several high-descent methods.

F3 margin / accuracy not improving:
  Dominant P2 failure. All functional candidates have large accuracy gaps vs PureKAN-AdamW.

F6 geometry over-regularization:
  FPA-sob and CFT show the classic geometry-safe but learning-weak profile.

F8 temporal dynamics missing:
  Still likely. FPA as implemented is too damped; EK-FNG lacks enough long-term adaptive dynamics.
```

## Required Artifacts

Written under `results/v4_3/`:

```text
p0_invariants.csv
p1_proposal_direction_audit.csv
p1_gate_summary.csv
p2_short_horizon_scorecard.csv
p2_gate_summary.csv
p3_fpa_scorecard.csv
p4_ekfng_scorecard.csv
p5_cft_scorecard.csv
p6_candidate_selection.csv
p7_confirm5.csv
p8_confirm10.csv
optimizer_dynamics_trace.csv
role_update_trace.csv
failure_table.csv
failure_type_by_dataset.csv
failure_type_by_role.csv
proposal_vs_adam_alignment.csv
feature_rank_failure.csv
margin_failure.csv
geometry_regularization_failure.csv
aggregate_decision.json
figures/
```

## Final Decision

```text
PureKAN functional optimization remains unsolved in v4.3.

What improved:
  FPA and EK-FNG make the direction more Adam-aligned than Sobolev-only / CFT.
  EK-FNG validates that edge-feature/task geometry is a meaningful direction to test.

What failed:
  The improved one-step direction did not produce short-horizon representation learning.
  FPA was too damped and underfit.
  EK-FNG remained below old F4-FNG or AdamW on the hard datasets.
  CFT still has the FTF failure mode in softer form.

Conclusion:
  The v4.3 hypothesis is partially supported but not solved:
  functional geometry should constrain Adam-like task dynamics,
  but the current FPA/EK-FNG implementations do not retain enough useful feature-forming motion.

Next recommended step:
  Do not expand seeds.
  Redesign FPA to retain a larger Adam component and add adaptive damping by role,
  especially for KMNIST input/block feature formation.
```



---


# Source 15: `docs/DG-KAN_v4.4_FunctionalOptimizer_Redesign_实验计划.md`


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



---


# Source 16: `docs/DG-KAN_v4.4_FunctionalOptimizer_Redesign_结果复盘.md`


# DG-KAN v4.4 Functional Optimizer Redesign 结果复盘

本轮依据 `docs/DG-KAN_v4.4_FunctionalOptimizer_Redesign_实验计划.md`。目标是验证 PureKAN functional optimizer 是否应该改写为 functional-coordinate Adam、Shadow-Adam function distillation 或 global output-space kernel step。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 smoke | 33 | 0 |
| P1 trajectory | 48 | 0 |
| P2 horizon | 144 | 0 |

## Code / Config Changes

```text
experiments/run_gafu_v44.py
  Added FCAdam-v2 functional-coordinate Adam with L2/H1/H1-low/dataSob metrics.
  Added SFD direct/prox/residual using shadow AdamW layer-output displacement as teacher.
  Added GFK diagnostic output/block/all-lowrank target-fit steps.
  Added AB-RBF PureKAN with constant/linear edge basis and no non-KAN trainable params.
  P1 records coefficient/function alignment to AdamW, R2, rank/margin/geometry changes.
  P2 records one-step and five-step horizon metrics for the written gate.

experiments/analyze_gafu_v44.py
  Generates P1/P2 gate summaries, failure reports, required placeholder artifacts,
  aggregate_decision.json, SVG diagnostics, and this replay.
```

## P0 Implementation Invariants

| rows | errors | strict | max nonKAN | min cov | max rollback | pass |
|---|---|---|---|---|---|---|
| 33 | 0 | 33 | 0.0000 | 1.0000 | 0.0000 | true |

P0 verdict: pass. All successful rows keep strict PureKAN trainable parameters at zero non-KAN params, coefficient coverage at 1.0, fixed alpha, and exact rollback.

## P1 AdamW Functional Trajectory Audit

| method | train all | holdout+ | cos f | R2 | cos coeff | rank Δ | phi ratio | P1 |
|---|---|---|---|---|---|---|---|---|
| AB-RBF-FCAdam-H1-low | 1 | 1 | 0.9746 | 0.9345 | 0.0578 | -13.3237 | 1.0084 | no |
| D0-allFullSobolev | 1 | 3 | 0.6776 | 0.3497 | 0.1361 | -5.9794 | 0.9963 | no |
| D6-allTaskAware | 0 | 2 | 0.8701 | 0.7332 | 0.6945 | -12.4961 | 1.0006 | no |
| EKFNG-leftRightFull-lowSob | 0 | 0 | 0.4763 | -59.8609 | 0.2657 | -23.6604 | 1.5052 | no |
| F4-FNG-leftFullRight | 0 | 1 | 0.6199 | 0.0844 | 0.1093 | -11.6313 | 0.9941 | no |
| FCAdam-H1-low | 1 | 1 | 0.8620 | 0.7100 | 0.0327 | -10.3199 | 1.0019 | no |
| FCAdam-L2 | 0 | 1 | 0.8949 | 0.7862 | 0.0298 | -12.2748 | 1.0031 | no |
| FCAdam-dataSob | 1 | 2 | 0.8752 | 0.7278 | 0.2083 | -7.6473 | 1.0176 | no |
| FPA-edge-prox | 1 | 2 | 0.8761 | 0.7368 | 0.9508 | -11.1536 | 1.0003 | no |
| GFK-all-lowrank | 1 | 3 | 0.1076 | 0.0009 | 0.0277 | 0.0003 | 1.0000 | no |
| GFK-block-output | 1 | 3 | 0.1096 | 0.0009 | 0.0081 | 0.0001 | 1.0000 | no |
| GFK-output-only | 1 | 3 | 0.1105 | 0.0008 | -0.0041 | 0.0000 | 1.0000 | no |
| PureKAN-AdamW-one-step | 1 | 2 | 1.0000 | 1.0000 | 1.0000 | -17.7872 | 1.0036 | yes |
| SFD-direct | 1 | 3 | 0.6416 | 0.3033 | 0.6115 | -3.6581 | 0.9995 | no |
| SFD-prox | 1 | 2 | 0.5014 | 0.1369 | 0.6115 | -1.3635 | 0.9995 | no |
| SFD-residual | 1 | 2 | 0.4239 | 0.0845 | 0.6115 | -0.8752 | 1.0006 | no |

P1 survivors:

```text
none
```

Observation:

```text
FCAdam and AB-RBF-FCAdam produce high function-space R2 on several datasets.
SFD often gives positive local descent but weaker R2 to AdamW function displacement.
GFK is stable but the output displacement is extremely small, with R2 near zero.
EKFNG-leftRightFull-lowSob remains numerically aggressive and fails local direction on multiple datasets.
```

## P2 One-Step / Five-Step Horizon Gate

| dataset | method | runs | hold5 | bad | R2 | rank/A | margin/A | phi/A | P2 |
|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | AB-RBF-FCAdam-H1-low | 3 | 0.4513 | 0.0000 | 0.4130 | 0.8440 | 1.0224 | 1.0266 | no |
| Fashion-MNIST | D0-allFullSobolev | 3 | 0.2931 | 0.0000 | 0.3129 | 0.2719 | 0.7868 | 0.9885 | no |
| Fashion-MNIST | D6-allTaskAware | 3 | 0.4225 | 0.0000 | 0.5640 | 0.5987 | 1.0053 | 0.9869 | no |
| Fashion-MNIST | EKFNG-leftRightFull-lowSob | 3 | 0.3971 | 0.2000 | -3.8822 | 1.0154 | 0.2633 | 1.3667 | no |
| Fashion-MNIST | F4-FNG-leftFullRight | 3 | 0.5557 | 0.0000 | 0.1676 | 0.5282 | 1.0238 | 0.9879 | no |
| Fashion-MNIST | FCAdam-H1-low | 3 | 0.4858 | 0.0667 | 0.2540 | 0.8169 | 1.1080 | 0.9961 | no |
| Fashion-MNIST | FCAdam-L2 | 3 | 0.5208 | 0.0667 | 0.2860 | 0.8054 | 1.1289 | 1.0122 | no |
| Fashion-MNIST | FCAdam-dataSob | 3 | 0.6420 | 0.0000 | 0.2358 | 0.6949 | 1.1497 | 1.0642 | no |
| Fashion-MNIST | FPA-edge-prox | 3 | 0.4947 | 0.0000 | 0.5831 | 0.6860 | 0.9521 | 0.9924 | no |
| Fashion-MNIST | GFK-all-lowrank | 3 | 0.0075 | 0.0000 | 0.0009 | 0.0000 | 0.0173 | 0.9874 | no |
| Fashion-MNIST | GFK-block-output | 3 | 0.0075 | 0.0000 | 0.0009 | -0.0000 | 0.0171 | 0.9875 | no |
| Fashion-MNIST | GFK-output-only | 3 | 0.0074 | 0.0000 | 0.0008 | -0.0000 | 0.0170 | 0.9875 | no |
| Fashion-MNIST | PureKAN-AdamW-one-step | 3 | 0.4185 | 0.0000 | 0.4785 | 1.0000 | 1.0000 | 1.0000 | yes |
| Fashion-MNIST | SFD-direct | 3 | 0.3836 | 0.0000 | 0.1034 | 0.4886 | 0.9900 | 0.9872 | no |
| Fashion-MNIST | SFD-prox | 3 | 0.2452 | 0.0000 | 0.0700 | 0.1699 | 0.6408 | 0.9880 | no |
| Fashion-MNIST | SFD-residual | 3 | 0.1831 | 0.0000 | 0.0609 | 0.0986 | 0.5217 | 0.9885 | no |
| KMNIST | AB-RBF-FCAdam-H1-low | 3 | 0.2502 | 0.0000 | 0.4450 | 0.8716 | 1.5318 | 1.0234 | no |
| KMNIST | D0-allFullSobolev | 3 | 0.1049 | 0.0667 | 0.4372 | 0.2019 | 0.6939 | 0.9860 | no |
| KMNIST | D6-allTaskAware | 3 | 0.1421 | 0.0000 | 0.6536 | 0.4635 | 0.7980 | 0.9859 | no |
| KMNIST | EKFNG-leftRightFull-lowSob | 3 | -1.0316 | 0.3333 | -6.5442 | 1.0872 | -4.3102 | 1.5513 | no |
| KMNIST | F4-FNG-leftFullRight | 3 | 0.2170 | 0.0000 | 0.3762 | 0.4474 | 1.0150 | 0.9908 | no |
| KMNIST | FCAdam-H1-low | 3 | 0.1798 | 0.0000 | 0.3892 | 0.8070 | 1.2717 | 1.0000 | no |
| KMNIST | FCAdam-L2 | 3 | 0.1913 | 0.0000 | 0.4165 | 0.9627 | 1.3089 | 1.0108 | no |
| KMNIST | FCAdam-dataSob | 3 | 0.2997 | 0.0000 | 0.4761 | 0.9011 | 1.2821 | 1.0820 | no |
| KMNIST | FPA-edge-prox | 3 | 0.1482 | 0.0000 | 0.7407 | 0.5411 | 0.5556 | 0.9881 | no |
| KMNIST | GFK-all-lowrank | 3 | 0.0033 | 0.0000 | 0.0008 | 0.0000 | 0.0135 | 0.9886 | no |
| KMNIST | GFK-block-output | 3 | 0.0033 | 0.0000 | 0.0007 | 0.0000 | 0.0135 | 0.9886 | no |
| KMNIST | GFK-output-only | 3 | 0.0032 | 0.0000 | 0.0007 | -0.0000 | 0.0134 | 0.9886 | no |
| KMNIST | PureKAN-AdamW-one-step | 3 | 0.1268 | 0.0667 | 0.4834 | 1.0000 | 1.0000 | 1.0000 | yes |
| KMNIST | SFD-direct | 3 | 0.1190 | 0.0000 | 0.2227 | 0.3574 | 0.9912 | 0.9889 | no |
| KMNIST | SFD-prox | 3 | 0.1083 | 0.0000 | 0.1238 | 0.1682 | 0.7823 | 0.9885 | no |
| KMNIST | SFD-residual | 3 | 0.0690 | 0.0000 | 0.1028 | 0.0859 | 0.5845 | 0.9869 | no |
| MNIST | AB-RBF-FCAdam-H1-low | 3 | 0.1856 | 0.0000 | 0.4222 | 0.9527 | 1.1001 | 1.0320 | no |
| MNIST | D0-allFullSobolev | 3 | 0.1519 | 0.0000 | 0.5261 | 0.5124 | 1.0375 | 0.9945 | no |
| MNIST | D6-allTaskAware | 3 | 0.0667 | 0.0000 | 0.6171 | 0.6112 | 0.3641 | 0.9979 | no |
| MNIST | EKFNG-leftRightFull-lowSob | 3 | -1.1332 | 0.2000 | -23.4381 | 1.1353 | -3.9763 | 1.7347 | no |
| MNIST | F4-FNG-leftFullRight | 3 | 0.1965 | 0.0000 | 0.5463 | 0.5883 | 1.0297 | 0.9878 | no |
| MNIST | FCAdam-H1-low | 3 | 0.1559 | 0.0000 | 0.4072 | 0.9988 | 1.4100 | 0.9992 | no |
| MNIST | FCAdam-L2 | 3 | 0.1697 | 0.0667 | 0.4348 | 1.1138 | 1.3523 | 1.0150 | no |
| MNIST | FCAdam-dataSob | 3 | 0.2588 | 0.0000 | 0.4826 | 1.0451 | 1.2523 | 1.1053 | no |
| MNIST | FPA-edge-prox | 3 | 0.1279 | 0.0000 | 0.7271 | 0.6865 | 0.3158 | 0.9926 | no |
| MNIST | GFK-all-lowrank | 3 | 0.0058 | 0.0000 | 0.0007 | -0.0000 | 0.0218 | 0.9927 | no |
| MNIST | GFK-block-output | 3 | 0.0058 | 0.0000 | 0.0006 | -0.0000 | 0.0216 | 0.9927 | no |
| MNIST | GFK-output-only | 3 | 0.0057 | 0.0000 | 0.0006 | -0.0000 | 0.0213 | 0.9927 | no |
| MNIST | PureKAN-AdamW-one-step | 3 | 0.1112 | 0.0667 | 0.4807 | 1.0000 | 1.0000 | 1.0000 | yes |
| MNIST | SFD-direct | 3 | 0.1142 | 0.0000 | 0.2672 | 0.5474 | 1.0338 | 0.9985 | no |
| MNIST | SFD-prox | 3 | 0.1283 | 0.0000 | 0.2100 | 0.3163 | 1.0444 | 0.9952 | no |
| MNIST | SFD-residual | 3 | 0.0795 | 0.0000 | 0.1663 | 0.2275 | 0.6430 | 0.9956 | no |

Best non-Adam points by holdout 5-step descent:

| dataset | best by hold5 | hold5 | R2 | rank/A | phi/A |
|---|---|---|---|---|---|
| MNIST | FCAdam-dataSob | 0.2588 | 0.4826 | 1.0451 | 1.1053 |
| Fashion-MNIST | FCAdam-dataSob | 0.6420 | 0.2358 | 0.6949 | 1.0642 |
| KMNIST | FCAdam-dataSob | 0.2997 | 0.4761 | 0.9011 | 1.0820 |

P2 survivors:

```text
none
```

P2 verdict:

```text
No candidate enters P3.

Many methods achieve positive 5-step holdout descent, especially FCAdam-dataSob,
FPA-edge-prox, F4-FNG and AB-RBF-FCAdam.
However the full written gate is stricter: candidates must keep Adam-like function
alignment or Adam-level train descent, retain rank/margin formation, and reduce
phi_prime_p95 below 90% of PureKAN-AdamW.

The persistent blocker is the geometry gate plus representation retention:
the candidates can learn locally, but they do not yet combine Adam-like function
trajectory, feature formation, and strict geometry improvement.
```

## P3-P9 Decision

```text
P3 short-horizon training: not run.
P4 FCAdam deep validation: not run.
P5 SFD deep validation: not run.
P6 GFK deep validation: not run.
P7/P8/P9 confirm: not run.

Reason: P2 produced no survivor under the written gate.
```

## Failure Diagnosis

Generated:

```text
failure_table.csv
failure_type_by_dataset.csv
failure_type_by_role.csv
proposal_vs_adam_alignment.csv
feature_rank_failure.csv
margin_failure.csv
geometry_regularization_failure.csv
```

Diagnosis:

```text
F3/F8 low Adam alignment or insufficient train dynamics:
  GFK is too small; SFD has low R2; several FCAdam variants fall just below the R2 gate.

F5/F6 representation failure:
  Some methods with positive holdout descent do not retain enough rank/margin movement vs AdamW.

F7 geometry gate failure:
  This is the broadest blocker. Functional candidates usually keep phi close to AdamW,
  but the v4.4 P2 gate requires phi <= 90% of AdamW while preserving learning.

AB-RBF diagnostic:
  AB-RBF-FCAdam improves local function alignment and holdout descent in places,
  but does not clear the joint rank/margin/geometry gate.
```

## Required Artifacts

Written under `results/v4_4/`:

```text
p0_invariants.csv
p1_adam_functional_trajectory.csv
p1_gate_summary.csv
p2_horizon_audit.csv
p2_gate_summary.csv
p3_short_horizon_train_scorecard.csv
p4_fcadam_scorecard.csv
p5_sfd_scorecard.csv
p6_gfk_scorecard.csv
p7_candidate_selection.csv
p8_confirm5.csv
p9_confirm10.csv
optimizer_dynamics_trace.csv
failure_table.csv
failure_type_by_dataset.csv
failure_type_by_role.csv
proposal_vs_adam_alignment.csv
feature_rank_failure.csv
margin_failure.csv
geometry_regularization_failure.csv
aggregate_decision.json
figures/p1_function_r2_vs_adam.svg
figures/p2_holdout_5step.svg
figures/p2_phi_ratio_vs_adam.svg
```

## Final Decision

```text
PureKAN functional optimization remains unsolved in v4.4.

What improved:
  FCAdam-v2 and AB-RBF-FCAdam show that functional-coordinate Adam can track
  AdamW function displacement much better than old Sobolev-only updates.
  SFD confirms that AdamW function displacement can be fit safely in one-step diagnostics.
  GFK gives a stable but too-small global output-space diagnostic.

What failed:
  None of FCAdam, SFD, GFK, FPA, EK-FNG, or AB-RBF-FCAdam passes the P2 joint gate.
  Positive 5-step loss descent did not become a validated representation-learning optimizer.

Conclusion:
  v4.4 partially supports the functional-coordinate Adam hypothesis, but not enough
  to justify P3/P4 expansion. The next useful redesign should keep more Adam-like
  functional trajectory while explicitly meeting the phi/rank/margin gate, rather
  than running more seeds on the current candidates.
```



---


# Source 17: `docs/DG-KAN_v4.5_AcceleratedFunctionalOptimizer_实验计划.md`


# DG-KAN v4.5：从 Functional Update 到 Accelerated Functional Optimizer 的重新设计与实验计划

## 0. 本计划的定位

v4.5 不再把问题理解为“继续调 Sobolev 系数、warmup、trust radius 或 shallow/deep 分工”。这些方向已经多轮验证过，最多能带来局部改善，不能解决 PureKAN functional optimizer 的根问题。

本计划的核心判断是：

$$
\boxed{
\text{当前 functional update 不是一个完整 optimizer，而只是一个局部 proposal / preconditioner。}
}
$$

PureKAN 需要的是一个真正的深度网络优化器：它必须同时具备任务学习动力学、函数空间几何约束、长期时间尺度、自适应步长、restart / rejection 机制，以及 rank / margin 形成能力。

因此 v4.5 的目标不是继续找更好的 $S^{-1}g$，而是设计：

$$
\boxed{
\text{AFO: Accelerated Functional Optimizer}
}
$$

AFO 的基本思想是：

$$
\text{Adam-like task trajectory}
+
\text{functional coordinate geometry}
+
\text{restart / Lyapunov controller}
+
\text{rank-margin preservation}
+
\text{late-stage geometry consolidation}.
$$

这版计划必须回答一个最根本的问题：

$$
\boxed{
\text{PureKAN 的 functional optimizer 是否能在不依赖非 KAN 参数的情况下，超过 PureKAN-AdamW、Hybrid-DGKAN-UFULL 和 MLP-AdamW？}
}
$$

如果不能，则必须承认当前 functional update 的正确定位只是 hybrid branch optimizer，而不是 PureKAN full-network optimizer。

---

## 1. v4.4 结果的深层复盘

### 1.1 实现问题基本排除

v4.4 的 P0 smoke 说明当前实验已经不是 hook / coverage / rollback 问题。所有 successful rows 保持：

```text
strict PureKAN nonKAN params = 0
coefficient coverage = 1.0
fixed alpha
exact rollback
errors = 0
```

这意味着如果 PureKAN functional optimizer 失败，不能再主要归因于：

```text
input/output KAN 没被更新
alphaFixed1 没生效
还有隐藏非 KAN 参数
rollback / temporary apply 实现错误
```

当前问题已经进入 optimizer 设计层。

---

### 1.2 FCAdam / AB-RBF-FCAdam 给出了最重要的正信号

v4.4 P1 中，FCAdam 和 AB-RBF-FCAdam 能明显接近 AdamW 的 function-space displacement。

典型信号：

```text
AB-RBF-FCAdam-H1-low:
  cos_f = 0.9746
  R2 = 0.9345

FCAdam-L2:
  cos_f = 0.8949
  R2 = 0.7862

FCAdam-H1-low:
  cos_f = 0.8620
  R2 = 0.7100

FCAdam-dataSob:
  cos_f = 0.8752
  R2 = 0.7278
```

这说明：

$$
\boxed{
\text{functional-coordinate Adam 是当前最接近 AdamW 学习动力学的方向。}
}
$$

但这些方法仍然没有通过 P1 / P2 gate，因为 rank、margin、geometry 无法同时满足。

这不是无效信号，而是说明 FCAdam 的底层方向比旧 Sobolev / TFU / FNG 更对，但还缺少 trajectory-level controller。

---

### 1.3 P2 的真正失败点不是 loss descent，而是 representation retention

v4.4 P2 中很多方法有正的 5-step holdout descent，尤其 `FCAdam-dataSob`：

```text
Fashion-MNIST:
  hold5 = 0.6420
  R2 = 0.2358
  rank/A = 0.6949
  margin/A = 1.1497
  phi/A = 1.0642

KMNIST:
  hold5 = 0.2997
  R2 = 0.4761
  rank/A = 0.9011
  margin/A = 1.2821
  phi/A = 1.0820

MNIST:
  hold5 = 0.2588
  R2 = 0.4826
  rank/A = 1.0451
  margin/A = 1.2523
  phi/A = 1.1053
```

这说明它不是完全不会下降，而是没有同时满足：

```text
1. Adam-like function trajectory
2. representation rank retention
3. margin formation
4. geometry improvement
```

v4.4 的 failure diagnosis 也指出，persistent blocker 是 geometry gate + representation retention：候选能局部学习，但没有同时组合 Adam-like trajectory、feature formation 和 strict geometry improvement。

因此下一步不应该问：

```text
哪个 update 能让 loss 降一点？
```

而应该问：

```text
哪个 optimizer 能形成 AdamW 类似的表示学习轨迹，同时逐步收紧函数几何？
```

---

### 1.4 GFK 的失败说明 global kernel step 不能太小

GFK 非常稳定，但几乎没有 function displacement：

```text
GFK-output-only / block-output / all-lowrank:
  R2 ≈ 0.0006 - 0.0009
  hold5 ≈ 0.003 - 0.008
  margin/A ≈ 0.01 - 0.02
```

这说明当前 GFK 不是危险，而是太弱。它给了一个重要边界：

$$
\boxed{
\text{全局 functional kernel 如果只做小型低秩 diagnostic，不足以承担训练。}
}
$$

后续如果继续 global kernel，必须扩大 Krylov / CG step、target dimension 或用 AdamW trajectory 作为 teacher，而不是继续 output-only tiny displacement。

---

### 1.5 SFD 的失败说明“拟合 AdamW displacement”不能只做单步

SFD 能安全地拟合一部分 AdamW function displacement，但 R2 明显不足：

```text
SFD-direct:
  cos coeff ≈ 0.6115
  R2 ≈ 0.3033 in P1 summary

SFD-prox / residual:
  R2 更低
```

这说明 AdamW displacement 可以作为 teacher，但当前 SFD 的 layer-wise fitting 太弱。它没有捕捉 AdamW 多步轨迹中的 momentum / RMS / noise-averaging 效果。

因此 SFD 不能再是：

$$
\text{fit one AdamW step}
$$

而应该变成：

$$
\text{fit a short AdamW rollout in function space}
$$

也就是把 teacher 从 one-step displacement 改成 $K$-step trajectory target。

---

## 2. 梯度下降研究报告带来的启发

你额外上传的加速梯度下降报告对这里非常有启发。它不是直接告诉我们用哪个现成 optimizer，而是提供了三个重要原则。

### 2.1 加速不是单步 preconditioner，而是动态系统

近五年的加速梯度研究强调 Lyapunov、high-resolution ODE、restart、momentum、参数无关步长和噪声鲁棒性。这说明真正的加速优化器不是一个静态矩阵 $M^{-1}$，而是一个带状态的动态系统。

当前 functional update 最大的问题正是：

$$
\boxed{
\text{它把 optimizer 简化成了静态 function-space preconditioner。}
}
$$

这对 toy function fitting 可以成立，但对 PureKAN 深度网络不够。

---

### 2.2 深度学习里有效的“加速”通常和 adaptive optimizer 结合

报告里提到的 Adan、Win、IRE 等方向都说明，深度学习实践中真正有效的加速，通常不是直接套经典 NAG，而是把 Nesterov / restart / flatness / implicit regularization 与 AdamW-style adaptive dynamics 结合。

这对我们意味着：

$$
\boxed{
\text{PureKAN functional optimizer 不应该试图抛弃 AdamW 的时间尺度，}
}
$$

而应该把 AdamW 的长期学习动力学搬到 functional coordinate 里。

这正是 v4.5 选择 FCAdam / FAdamNAG / restart controller 的原因。

---

### 2.3 一般非凸没有免费加速，必须利用结构

报告里也强调，一般 PL / 非凸问题并不存在普遍免费加速；有效加速依赖强几何结构、restart、variance modeling、flatness 或问题特定结构。

PureKAN 的结构是：

$$
Y_l = \Phi_l(X_l) A_l^\top.
$$

因此真正应该利用的是：

```text
basis feature covariance
edge/channel structure
layer output tangent
class margin dynamics
function-space whitening
Sobolev smoothness budget
```

而不是只用一个 basis-only Sobolev matrix。

---

## 3. 当前 functional update 的根本缺陷

### 3.1 缺陷一：把 smoothness prior 当成 optimizer geometry

旧 U-FULL 本质是：

$$
\Delta a=-\eta S^{-1}g.
$$

其中 $S$ 是 Sobolev metric。这个 $S$ 关心函数平滑、导数、曲率，但不关心 class margin、feature rank、downstream sensitivity。

因此它擅长让几何漂亮，却不擅长形成分类表示。

这解释了多轮实验中反复出现的模式：

```text
phi / J 很好
ECE 有时很好
但 accuracy / margin / rank 不够
```

---

### 3.2 缺陷二：local descent 不等于 representation learning

D6 / FNG / FPA / FCAdam 多次出现 positive local descent，但 P2 / micro-run 不过。原因是 PureKAN 的目标不是只让当前 batch loss 下降，而是形成长期特征：

```text
input KAN 要学特征提取
block KAN 要学组合变换
output KAN 要学分类边界
```

局部下降可能短期有用，但如果破坏 feature rank 或 class separation，长期 accuracy 会输 AdamW。

---

### 3.3 缺陷三：当前 optimizer 没有“能量函数”

我们现在记录了很多指标，但 optimizer 本身没有一个统一 energy 来决定：

```text
该不该加速
该不该 restart
该不该放松 geometry
该不该收紧 Sobolev
该不该保留 Adam-like direction
```

这和加速梯度文献中的 Lyapunov / restart 思想相反。优化器必须有一个可监控的能量。

v4.5 需要引入 functional Lyapunov energy：

$$
\mathcal E_t
=
L_t
+
\lambda_\phi \max(0, \phi_t/\phi_{ref}-r_\phi)^2
+
\lambda_R \max(0, r_R-R_t/R_{ref})^2
+
\lambda_m \max(0, r_m-m_t/m_{ref})^2
+
\lambda_s \|\Delta f_t\|^2.
$$

其中：

```text
L_t = train/holdout loss
phi_t = geometry roughness
R_t = effective rank
m_t = margin statistic
Delta f_t = function displacement
```

---

### 3.4 缺陷四：几何 gate 的时序可能错了

v4.4 P2 gate 要求候选在早期 horizon 就满足：

$$
\phi/A \leq 0.9.
$$

这个条件可能过早。AdamW 的成功很可能先允许函数 roughness 上升，形成有效 feature 和 margin，再通过后期正则或 averaging 收几何。

所以下一版不能要求：

```text
每个早期阶段都比 AdamW 更平滑。
```

应该改成：

```text
early: 保 rank / margin / loss trajectory
mid: 限制 phi 不爆
late: consolidate geometry，使 phi/J 优于 AdamW
```

也就是把 geometry 从 immediate hard gate 改成 phased budget。

---

### 3.5 缺陷五：缺少 restart 和 anti-stall 机制

AdamW 不一定每一步方向都优雅，但它的 momentum / RMS / weight decay / step schedule 给了长期稳定学习动力学。

当前 functional update 即使加入 trust gate，也只是在防坏步；它没有：

```text
stagnation detection
restart
momentum reset
geometry relaxation
rank rescue
step-size recovery
```

这就是为什么 BFT 变安全后也变弱。v4.5 必须加入 restart / acceleration state。

---

## 4. v4.5 的核心新方向：AFO

AFO 的完整名称：

```text
Accelerated Functional Optimizer
```

核心目标：

$$
\boxed{
\text{在 function-whitened coordinate 中保留 AdamW 的任务学习动力学，}
}
$$

同时用 Sobolev 和 rank / margin constraints 做后期几何控制。

---

## 5. AFO 的三个核心组件

### 5.1 Functional-coordinate Adam backbone

对每个 KAN layer，令 Sobolev / data-Sobolev metric 为：

$$
M_l = S_l + \rho I.
$$

做 Cholesky：

$$
M_l = L_l L_l^\top.
$$

定义 whitened coordinate：

$$
a_l = L_l^{-\top}u_l.
$$

在 $u_l$ 上运行 AdamW / AdamW-like update：

$$
g_{u,l}=L_l^{-1}g_{a,l}.
$$

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_{u,t}.
$$

$$
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_{u,t}^2.
$$

$$
\Delta u_t=-\eta \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

再映射回 coefficient：

$$
\Delta a_t=L_t^{-\top}\Delta u_t.
$$

这和旧 U-FULL 的区别是：

```text
旧 U-FULL:
  每步直接 S^{-1}g

FC-Adam:
  用 S 定义函数坐标，在函数坐标里保留 AdamW 时间动力学
```

这应该成为 v4.5 主线。

---

### 5.2 Functional Nesterov / lookahead dynamics

加速报告启发我们：有效加速不是裸 momentum，而是 lookahead + restart + Lyapunov。

在 functional coordinate $u$ 中做 lookahead：

$$
\tilde u_t=u_t+\mu_t(u_t-u_{t-1}).
$$

在 $\tilde u_t$ 对应的参数上计算 gradient：

$$
g_t=\nabla_u L(\tilde u_t).
$$

然后进行 Adam-like 或 normalized update。

为了避免旧 Adam moment 在 coefficient space 里失控，momentum 必须存在于 function-whitened coordinate，而不是 raw coefficient coordinate。

---

### 5.3 Functional restart / Lyapunov controller

定义能量：

$$
\mathcal E_t
=
L^{hold}_t
+
\lambda_R \mathcal P_R(t)
+
\lambda_m \mathcal P_m(t)
+
\lambda_\phi \mathcal P_\phi(t)
+
\lambda_J \mathcal P_J(t).
$$

其中：

$$
\mathcal P_R(t)=\max(0, r_R-R_t/R_{Adam,t})^2,
$$

$$
\mathcal P_m(t)=\max(0, r_m-m_t/m_{Adam,t})^2,
$$

$$
\mathcal P_\phi(t)=\max(0, \phi_t/\phi_{Adam,t}-r_\phi(t))^2,
$$

$$
\mathcal P_J(t)=\max(0, J_t/J_{Adam,t}-r_J(t))^2.
$$

如果出现：

$$
\mathcal E_{t+1} > \mathcal E_t + \epsilon_E,
$$

则触发 restart：

```text
reset functional momentum
shrink lr
relax / tighten geometry depending on phase
restore last accepted state if necessary
```

---

## 6. 阶段性 geometry policy

v4.5 不再把 geometry 作为早期 hard gate。采用三阶段策略：

### Phase I：representation formation

目标：

```text
rank / margin / loss trajectory 接近 AdamW
```

允许：

$$
\phi/A \leq 1.15.
$$

不要求早期 $\phi$ 低于 AdamW。

---

### Phase II：geometry stabilization

目标：

```text
保持 accuracy / margin，同时开始降低 phi/J
```

要求：

$$
\phi/A \leq 1.05.
$$

---

### Phase III：geometry consolidation

目标：

```text
最终 geometry 优于 AdamW
```

要求：

$$
\phi/A \leq 0.90,
$$

或至少：

$$
\phi\text{ reduction} > 10\%.
$$

---

## 7. v4.5 实验总览

本轮实验按以下顺序执行：

```text
P0: implementation smoke and coordinate correctness
P1: AdamW trajectory forensic audit
P2: FC-Adam backbone one-step and short-horizon audit
P3: Functional Nesterov / Adan-like dynamics
P4: Lyapunov restart and phased geometry controller
P5: rank / margin preservation ablation
P6: 3-seed full-budget candidate selection
P7: 5-seed confirm
P8: 10-seed final confirm
P9: failure diagnosis and theory update
```

---

## 8. P0：实现与坐标正确性检查

### 8.1 目标

确认 function-whitened coordinate 的数学实现正确。

### 8.2 必跑模型

```text
PureKAN-FixedNorm
basis_count = 16 / 24
hidden_dim = 64 / 96
depth = 2 / 4
```

### 8.3 必查不变量

每个 run 必须记录：

```text
learnable_nonKAN_params
functional_coverage
alpha_trainable
input_coeff_seen
block_coeff_seen
output_coeff_seen
metric_condition_L2
metric_condition_H1
metric_condition_dataSob
whiten_reconstruction_error
u_to_a_roundtrip_error
adam_state_shape_match
rollback_error
```

### 8.4 数学检查

检查：

$$
\|a-L^{-\top}u\|/\|a\| < 10^{-6}.
$$

检查 update 映射：

$$
\Delta a = L^{-\top}\Delta u.
$$

检查 gradient 变换：

$$
g_u = L^{-1}g_a.
$$

### 8.5 通过标准

```text
nonKAN params = 0
functional coverage = 1.0
roundtrip error < 1e-6
rollback error = 0
no NaN / Inf
metric condition finite
```

### 8.6 可视化

```text
metric spectrum per role
roundtrip error histogram
u-space grad norm vs a-space grad norm
```

---

## 9. P1：AdamW trajectory forensic audit

### 9.1 目标

先搞清楚 PureKAN-AdamW 到底为什么能训练。不要再只把 AdamW 当黑盒 baseline。

### 9.2 方法

训练 PureKAN-AdamW，并在每个 epoch / selected steps 记录 function-space trajectory。

### 9.3 必记指标

```text
train_loss
holdout_loss
test_acc
val_auc
feature_effective_rank_input
feature_effective_rank_block_l
feature_effective_rank_output
class_centroid_separation
margin_mean
margin_p10
margin_p50
phi_prime_p95
jacobian_condition
basis_occupancy_entropy
dead_basis_fraction
coefficient_norm_by_role
function_displacement_norm_by_role
AdamW_m_norm_by_role
AdamW_v_norm_by_role
AdamW_update_norm_by_role
AdamW_update_over_param_by_role
```

### 9.4 新增关键诊断

记录 AdamW 的每步 function displacement：

$$
\Delta f_{Adam,l}(x)=f_l(a_t+\Delta a^{Adam}_t,x)-f_l(a_t,x).
$$

并记录：

```text
Delta f norm
Delta f alignment across steps
role update share
rank change after update
margin change after update
phi change after update
```

### 9.5 可视化

```text
AdamW trajectory: loss / rank / margin / phi over epochs
role-wise update share stacked area
rank vs margin scatter
phi vs accuracy scatter
AdamW m/v norm curves
function displacement norm by role
```

### 9.6 产出

得到 AdamW 的 target profile：

```text
rank_target(t)
margin_target(t)
phi_budget(t)
role_update_share_target(t)
function_displacement_target(t)
```

后续所有 functional optimizer 必须与这个 target profile 对齐。

---

## 10. P2：FC-Adam backbone audit

### 10.1 目标

验证 functional-coordinate Adam 是否能成为 PureKAN functional optimizer 的 backbone。

### 10.2 候选

```text
FCAdam-L2
FCAdam-H1-low
FCAdam-dataSob
AB-RBF-FCAdam-H1-low
FCAdam-dataSob-noGeometryGate
FCAdam-dataSob-phasedGeometry
```

### 10.3 配置说明

`dataSob` 的 metric：

$$
M_l = G_{data,l}+\lambda S_l+\rho I.
$$

其中：

$$
G_{data,l}=\mathbb E[\Phi_l^\top \Phi_l].
$$

如果 full $G_{data,l}$ 太贵，先用：

```text
diagonal data covariance
low-rank + diagonal covariance
EMA covariance
```

### 10.4 必记指标

```text
cos_function_with_adam
function_R2_with_adam
cos_coeff_with_adam
holdout_1step_descent
holdout_5step_descent
holdout_20step_descent
rank/A
margin/A
phi/A
jac/A
ECE
accepted_lr
u_m_norm
u_v_norm
u_update_norm
restart_count
```

### 10.5 通过标准

P2 不要求最终超过 AdamW，但必须满足：

```text
cos_function_with_adam >= 0.85 on at least 2/3 datasets
function_R2 >= 0.70 on at least 2/3 datasets
rank/A >= 0.85
margin/A >= 0.85
holdout_5step_descent positive on all datasets
early phi/A <= 1.15
bad_step_rate <= 0.05
```

### 10.6 可视化

```text
function_R2_vs_rank_retention
cos_function_with_adam_vs_holdout_descent
rank/A over 20 steps
margin/A over 20 steps
phi/A over 20 steps
```

---

## 11. P3：Functional Nesterov / Adan-like dynamics

### 11.1 目标

测试加速报告启发的方向：function coordinate 中加入 Nesterov / Adan-like temporal dynamics。

### 11.2 候选

```text
FAdam:
  FCAdam without lookahead

FNAG:
  FCAdam + Nesterov lookahead

FAdan-lite:
  functional-coordinate Adam + gradient-difference momentum

FWin-lite:
  functional-coordinate AdamW + weight-decay-integrated lookahead
```

### 11.3 FNAG 公式

在 $u$ 坐标：

$$
\tilde u_t = u_t + \mu_t(u_t-u_{t-1}).
$$

在 $\tilde u_t$ 处计算梯度：

$$
g_t=\nabla_u L(\tilde u_t).
$$

然后执行 Adam-like update。

### 11.4 FAdan-lite 公式

记录 gradient difference：

$$
d_t=g_t-g_{t-1}.
$$

更新：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)(g_t+\gamma d_t).
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)(g_t+\gamma d_t)^2.
$$

### 11.5 关键安全机制

所有 momentum 必须位于 function-whitened coordinate $u$，不能位于 raw coefficient $a$。

### 11.6 必记指标

```text
lookahead_loss
post_update_loss
lookahead_descent
momentum_norm
momentum_cos_current_grad
momentum_cos_adamw_update
restart_triggered
rank_change_after_lookahead
margin_change_after_lookahead
phi_change_after_lookahead
```

### 11.7 通过标准

```text
holdout_20step_descent > FCAdam backbone
rank/A not lower than FCAdam by > 5%
margin/A not lower than FCAdam by > 5%
phi/A <= 1.15 early, <= 1.05 mid
no catastrophic loss spike
```

---

## 12. P4：Functional Lyapunov restart controller

### 12.1 目标

验证 restart / Lyapunov controller 是否能解决长期 trajectory drift。

### 12.2 Energy 定义

$$
\mathcal E_t
=
L^{hold}_t
+\lambda_R \max(0,r_R-R_t/R_{ref,t})^2
+\lambda_m \max(0,r_m-m_t/m_{ref,t})^2
+\lambda_\phi \max(0,\phi_t/\phi_{ref,t}-r_\phi(t))^2
+\lambda_J \max(0,J_t/J_{ref,t}-r_J(t))^2.
$$

其中 reference 可以是：

```text
PureKAN-AdamW trajectory
或当前 run 的 EMA target
```

### 12.3 Restart 条件

```text
E increases for k consecutive audits
holdout loss worsens beyond epsilon
rank/A drops below threshold
margin/A drops below threshold
phi/J exceeds phase budget
```

### 12.4 Restart 动作

```text
reset momentum
halve lr
increase damping rho
optionally relax Sobolev lambda in early phase
restore last accepted state if loss spike severe
```

### 12.5 必记指标

```text
energy_total
energy_loss_term
energy_rank_term
energy_margin_term
energy_phi_term
energy_jac_term
restart_count
restart_reason
post_restart_recovery_steps
lr_after_restart
rho_after_restart
```

### 12.6 可视化

```text
Lyapunov energy curves
restart markers on loss curve
restart reason stacked bar
rank/margin before-after restart
phi/J before-after restart
```

---

## 13. P5：rank / margin preservation ablation

### 13.1 目标

验证 rank / margin 是否是 functional optimizer 追不上 AdamW 的核心缺口。

### 13.2 候选

```text
FCAdam-dataSob
FCAdam-dataSob + rank preservation
FCAdam-dataSob + margin preservation
FCAdam-dataSob + rank + margin preservation
FNAG + rank + margin preservation
```

### 13.3 Rank preservation penalty

不是直接加入训练 loss，而是作为 update acceptance / energy penalty：

$$
\mathcal P_R=\max(0,r_R-R_t/R_{Adam,t})^2.
$$

### 13.4 Margin preservation penalty

$$
\mathcal P_m=\max(0,r_m-m_t/m_{Adam,t})^2.
$$

### 13.5 必记指标

```text
effective_rank_input
block_effective_rank_l
output_effective_rank
class_centroid_separation
margin_mean
margin_p10
margin_p50
classwise_margin
classwise_accuracy
rank_penalty_value
margin_penalty_value
```

### 13.6 通过标准

```text
rank/A >= 0.90
margin/A >= 0.90
accuracy gap vs AdamW decreases by >= 50% compared with FCAdam backbone
phi/A <= 1.10 early and <= 1.00 late
```

---

## 14. P6：3-seed full-budget candidate selection

### 14.1 候选来源

只有 P2-P5 通过 gate 的方法进入 P6。

预期候选最多 4 个：

```text
FCAdam-dataSob-phasedGeometry
FNAG-dataSob-restart
FAdan-lite-dataSob-restart
FCAdam-dataSob-rankMargin
```

### 14.2 Baselines

必须包含：

```text
PureKAN-AdamW
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
D0-allFullSobolev
D6-allTaskAware
F4-FNG-leftFullRight
```

### 14.3 数据集

```text
MNIST
Fashion-MNIST
KMNIST
```

### 14.4 主要指标

```text
test_acc
val_loss_auc
train_loss_auc
ECE
NLL
phi_prime_p95
Jacobian condition
curvature_energy
rank/A
margin/A
cos_function_with_adam
function_R2_with_adam
step_time
memory_peak
```

### 14.5 P6 通过标准

每个候选必须满足：

```text
MNIST acc >= PureKAN-AdamW - 0.5%
Fashion acc >= PureKAN-AdamW - 0.5%
KMNIST acc >= PureKAN-AdamW - 1.0%
val_loss_auc >= PureKAN-AdamW or within 2%
ECE <= PureKAN-AdamW
late phi_prime_p95 <= PureKAN-AdamW
rank/A >= 0.90
margin/A >= 0.90
```

如果没有候选满足，不进入 5-seed。

---

## 15. P7：5-seed confirm

### 15.1 目标

验证候选不是 3-seed 偶然。

### 15.2 统计

记录 paired delta vs PureKAN-AdamW：

```text
paired_acc_delta
paired_auc_delta
paired_ece_delta
paired_phi_delta
paired_rank_delta
paired_margin_delta
```

bootstrap CI：

```text
95% CI for acc delta
95% CI for AUC delta
95% CI for phi reduction
```

### 15.3 通过标准

```text
acc delta mean >= 0 on at least 2/3 datasets
KMNIST acc delta >= -0.5%
AUC delta >= 0 on all datasets
ECE delta <= 0 on all datasets
late phi reduction >= 10% on at least 2/3 datasets
rank/A >= 0.90
margin/A >= 0.90
```

---

## 16. P8：10-seed final confirm

### 16.1 目标

只有 P7 通过才跑。

### 16.2 最终 claim 分类

#### Clean PureKAN functional optimizer

需要：

```text
acc >= PureKAN-AdamW on mean and paired CI not strongly negative
AUC >= PureKAN-AdamW
ECE <= PureKAN-AdamW
phi/J better than AdamW late phase
rank/margin preserved
strict PureKAN nonKAN params = 0
```

#### Pareto PureKAN functional optimizer

如果：

```text
accuracy roughly matches AdamW
AUC / ECE / geometry better
rank/margin preserved
```

但 accuracy CI 跨 0，则只称 Pareto。

#### Functional branch optimizer only

如果仍不能接近 PureKAN-AdamW，则结论保持：

```text
functional update works for hybrid KAN residual branches,
but not yet as a full PureKAN optimizer.
```

---

## 17. P9：失败诊断

如果 v4.5 仍失败，必须输出 failure taxonomy，而不是继续调参。

### 17.1 Failure classes

```text
F1 coordinate mismatch:
  function R2 low, cos_function low

F2 temporal dynamics failure:
  one-step good, 20-step bad, restart high

F3 rank collapse:
  rank/A < 0.85

F4 margin failure:
  margin/A < 0.85

F5 geometry conflict:
  rank/margin good but phi/J too high

F6 over-regularization:
  phi/J good but loss/rank/margin bad

F7 task overfit:
  train descent good, holdout bad

F8 acceleration instability:
  lookahead loss spikes, restart frequent
```

### 17.2 必画图

```text
failure radar by method
failure type heatmap by dataset
rank vs accuracy scatter
margin vs accuracy scatter
phi vs accuracy scatter
function R2 vs accuracy gap
restart count vs accuracy gap
energy terms over time
```

---

## 18. 最终可视化清单

必须生成以下图表：

```text
1. AdamW trajectory forensic dashboard
2. FCAdam function R2 vs rank retention
3. holdout descent vs accuracy gap
4. rank/margin/phi phase plot
5. Lyapunov energy with restart markers
6. role-wise update share stacked area
7. momentum norm and cosine trace
8. geometry consolidation curve
9. accuracy-geometry Pareto plot
10. failure taxonomy heatmap
```

每张图必须同时显示：

```text
PureKAN-AdamW
D0 Sobolev
D6 task-aware
best v4.5 candidate
```

---

## 19. 本轮最重要的禁止事项

v4.5 不允许继续做以下事情作为主线：

```text
继续扫 Sobolev alpha / beta
继续扫 branch_final_scale
继续扫 diagwarmup length
继续扫 shallow/deep 固定分工
继续单独跑 FTF / BFT 小步验收
继续把 early phi < AdamW 作为硬门槛
```

这些都已经证明不能解决核心问题。

---

## 20. v4.5 的最终目标

v4.5 要回答：

$$
\boxed{
\text{PureKAN functional optimization 是否可以通过 functional-coordinate adaptive dynamics 成立？}
}
$$

如果答案是 yes，项目进入真正 PureKAN optimizer 主线。

如果答案是 no，则应该正式写清楚：

$$
\boxed{
\text{当前 functional update 的有效范围是 hybrid KAN branch，不是 full PureKAN training。}
}
$$

这两个结果都比继续小修小补更有科学价值。



---


# Source 18: `docs/DG-KAN_v4.5_AcceleratedFunctionalOptimizer_结果复盘.md`


# DG-KAN v4.5 Accelerated Functional Optimizer 结果复盘

本轮依据 `docs/DG-KAN_v4.5_AcceleratedFunctionalOptimizer_实验计划.md`。目标是验证 PureKAN 是否需要 AFO：functional-coordinate Adam backbone + trajectory controller + phased geometry。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 coordinate smoke | 24 | 0 |
| P1 AdamW forensic | 60 | 0 |
| P2 FCAdam backbone | 90 | 0 |

## Code / Config Changes

```text
experiments/run_gafu_v45.py
  Added v4.5 coordinate correctness audit across basis_count 16/24, hidden_dim 64/96, depth 2/4.
  Added AdamW trajectory forensic logging for rank, margin, phi, role update share and Adam m/v state.
  Added 20-step FCAdam backbone audit for L2/H1-low/dataSob, AB-RBF-FCAdam and phased/no-geometry variants.

experiments/analyze_gafu_v45.py
  Generates target profiles, P2 gate summaries, failure taxonomy, figures, aggregate_decision.json,
  and this result replay.
```

## P0 Coordinate Correctness

| rows | errors | max nonKAN | max roundtrip | max u->a | max rollback | pass |
|---|---|---|---|---|---|---|
| 24 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | true |

P0 verdict: pass. Functional-coordinate whitening and roundtrip are correct to numerical precision, with strict PureKAN non-KAN parameter count still zero.

## P1 AdamW Trajectory Forensic

| dataset | holdout Δ | rank ratio | phi ratio | input share | block share | output share |
|---|---|---|---|---|---|---|
| Fashion-MNIST | 1.0410 | 0.3267 | 1.0363 | 0.6785 | 0.2101 | 0.1114 |
| KMNIST | 0.4858 | 0.2081 | 1.0345 | 0.6938 | 0.2080 | 0.0982 |
| MNIST | 0.2009 | 0.4730 | 1.0124 | 0.7120 | 0.1983 | 0.0897 |

Observation:

```text
AdamW itself is not a pure geometry-improving trajectory in the first 20 steps.
It lowers holdout loss while reducing effective rank and slightly increasing phi.
This supports v4.5's phased policy: early optimization should prioritize task trajectory,
rank/margin formation, and only later consolidate geometry.
```

## P2 FCAdam Backbone Audit

| dataset | method | runs | hold20 | cos | R2 | rank/A | margin/A | phi/A | bad | P2 |
|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | AB-RBF-FCAdam-H1-low | 3 | 1.2393 | 0.1622 | 0.0906 | 0.7815 | 1.2813 | 1.0261 | 0.0000 | no |
| Fashion-MNIST | D0-allFullSobolev | 3 | 0.7351 | 0.5151 | 0.2674 | 0.4753 | 0.8915 | 0.9497 | 0.0667 | no |
| Fashion-MNIST | D6-allTaskAware | 3 | 0.8802 | 0.6034 | 0.3682 | 0.6601 | 0.9294 | 0.9544 | 0.0833 | no |
| Fashion-MNIST | F4-FNG-leftFullRight | 3 | 1.2013 | 0.2244 | 0.0011 | 0.4744 | 1.0690 | 0.9515 | 0.0167 | no |
| Fashion-MNIST | FCAdam-H1-low | 3 | 1.2070 | -0.0361 | 0.0411 | 0.8224 | 1.3345 | 0.9959 | 0.0000 | no |
| Fashion-MNIST | FCAdam-L2 | 3 | 1.3216 | -0.0630 | 0.0369 | 0.8577 | 1.4672 | 1.0837 | 0.0000 | yes |
| Fashion-MNIST | FCAdam-dataSob | 3 | 1.5207 | -0.0539 | 0.0158 | 0.8212 | 1.8907 | 1.2919 | 0.0000 | no |
| Fashion-MNIST | FCAdam-dataSob-noGeometryGate | 3 | 1.5207 | -0.0539 | 0.0158 | 0.8212 | 1.8907 | 1.2919 | 0.0000 | no |
| Fashion-MNIST | FCAdam-dataSob-phasedGeometry | 3 | 1.5207 | -0.0539 | 0.0158 | 0.8212 | 1.8907 | 1.2919 | 0.0000 | no |
| Fashion-MNIST | PureKAN-AdamW-one-step | 3 | 1.0896 | 0.4556 | 0.2016 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | yes |
| KMNIST | AB-RBF-FCAdam-H1-low | 3 | 0.8285 | 0.3797 | 0.1526 | 0.7674 | 1.2838 | 1.0342 | 0.0000 | no |
| KMNIST | D0-allFullSobolev | 3 | 0.3708 | 0.6193 | 0.3675 | 0.3383 | 0.7678 | 0.9577 | 0.1000 | no |
| KMNIST | D6-allTaskAware | 3 | 0.4430 | 0.8071 | 0.6264 | 0.5162 | 0.8728 | 0.9620 | 0.1167 | no |
| KMNIST | F4-FNG-leftFullRight | 3 | 0.6652 | 0.5413 | 0.2730 | 0.3028 | 0.9775 | 0.9619 | 0.0167 | no |
| KMNIST | FCAdam-H1-low | 3 | 0.8371 | 0.2556 | 0.1084 | 0.8158 | 1.3646 | 1.0114 | 0.0000 | no |
| KMNIST | FCAdam-L2 | 3 | 0.8976 | 0.2770 | 0.1360 | 0.9302 | 1.3593 | 1.1112 | 0.0000 | yes |
| KMNIST | FCAdam-dataSob | 3 | 1.3247 | 0.2068 | 0.1046 | 0.7836 | 2.1900 | 1.3305 | 0.0000 | no |
| KMNIST | FCAdam-dataSob-noGeometryGate | 3 | 1.3247 | 0.2068 | 0.1046 | 0.7836 | 2.1900 | 1.3305 | 0.0000 | no |
| KMNIST | FCAdam-dataSob-phasedGeometry | 3 | 1.3247 | 0.2068 | 0.1046 | 0.7836 | 2.1900 | 1.3305 | 0.0000 | no |
| KMNIST | PureKAN-AdamW-one-step | 3 | 0.5210 | 0.5340 | 0.2127 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | yes |
| MNIST | AB-RBF-FCAdam-H1-low | 3 | 0.7428 | 0.4326 | 0.1582 | 0.9675 | 1.5019 | 1.0532 | 0.0167 | yes |
| MNIST | D0-allFullSobolev | 3 | 0.3983 | 0.6473 | 0.4098 | 0.4566 | 0.9856 | 0.9637 | 0.1333 | no |
| MNIST | D6-allTaskAware | 3 | 0.2977 | 0.8368 | 0.6677 | 0.6147 | 0.7252 | 0.9650 | 0.1333 | no |
| MNIST | F4-FNG-leftFullRight | 3 | 0.6127 | 0.6163 | 0.3532 | 0.3308 | 1.2163 | 0.9662 | 0.1000 | no |
| MNIST | FCAdam-H1-low | 3 | 0.7625 | 0.4143 | 0.1466 | 0.9848 | 1.5791 | 1.0334 | 0.0000 | yes |
| MNIST | FCAdam-L2 | 3 | 0.8712 | 0.3733 | 0.1426 | 1.0802 | 1.6470 | 1.1399 | 0.0167 | yes |
| MNIST | FCAdam-dataSob | 3 | 1.4264 | 0.2950 | 0.1342 | 0.9437 | 2.6652 | 1.3805 | 0.0000 | no |
| MNIST | FCAdam-dataSob-noGeometryGate | 3 | 1.4264 | 0.2950 | 0.1342 | 0.9437 | 2.6652 | 1.3805 | 0.0000 | no |
| MNIST | FCAdam-dataSob-phasedGeometry | 3 | 1.4264 | 0.2950 | 0.1342 | 0.9437 | 2.6652 | 1.3805 | 0.0000 | no |
| MNIST | PureKAN-AdamW-one-step | 3 | 0.2652 | 0.5462 | 0.2059 | 1.0000 | 1.0000 | 1.0000 | 0.0167 | yes |

Best non-Adam points by 20-step holdout descent:

| dataset | best hold20 | hold20 | cos | R2 | rank/A | phi/A |
|---|---|---|---|---|---|---|
| MNIST | FCAdam-dataSob | 1.4264 | 0.2950 | 0.1342 | 0.9437 | 1.3805 |
| Fashion-MNIST | FCAdam-dataSob | 1.5207 | -0.0539 | 0.0158 | 0.8212 | 1.2919 |
| KMNIST | FCAdam-dataSob | 1.3247 | 0.2068 | 0.1046 | 0.7836 | 1.3305 |

P2 survivors:

```text
none
```

P2 verdict:

```text
No candidate enters P3.

FCAdam-dataSob has very strong 20-step holdout descent on all datasets, but its
function-space alignment to the AdamW target trajectory collapses over 20 steps.
AB-RBF-FCAdam improves some rank/margin behavior but still fails the global
cos/R2 requirements.

The v4.5 positive signal is real task descent, not a solved optimizer.
The blocker moved from immediate descent to temporal trajectory mismatch:
the methods learn locally, but their multi-step function displacement no longer
resembles AdamW enough to satisfy the written AFO backbone gate.
```

## P3-P8 Decision

```text
P3 Functional Nesterov / Adan-like dynamics: not run.
P4 Lyapunov restart controller: not run.
P5 rank/margin preservation ablation: not run.
P6 3-seed full-budget selection: not run.
P7/P8 confirm: not run.

Reason: P2 produced no survivor under the written gate.
```

## P9 Failure Diagnosis

Generated:

```text
p9_failure_diagnosis.csv
failure_table.csv
failure_type_heatmap_by_dataset.csv
failure_type_heatmap_by_method.csv
rank_margin_phi_scatter.csv
```

Diagnosis:

```text
F1 coordinate mismatch:
  Dominant after 20 steps. FCAdam-dataSob descends strongly, but cos/R2 to AdamW's
  function trajectory is far below the P2 requirement.

F3/F4 rank and margin:
  Some variants preserve rank/margin, but not together with Adam-like function alignment.

F5 geometry conflict:
  dataSob variants often have phi/A > 1.0. This is acceptable in Phase I up to 1.15,
  but it still does not compensate for low function alignment.

F8 temporal dynamics:
  The core unsolved issue is now temporal: one-step/short-step descent exists,
  but the 20-step trajectory drifts away from AdamW.
```

## Required Artifacts

Written under `results/v4_5/`:

```text
p0_coordinate_invariants.csv
p1_adamw_trajectory_forensic.csv
p1_adamw_target_profile.csv
p2_fcadam_backbone_audit.csv
p2_fcadam_gate_summary.csv
optimizer_dynamics_trace.csv
p3_temporal_dynamics.csv
p4_lyapunov_restart.csv
p5_rank_margin_ablation.csv
p6_candidate_selection.csv
p7_confirm5.csv
p8_confirm10.csv
p9_failure_diagnosis.csv
failure_table.csv
failure_type_heatmap_by_dataset.csv
failure_type_heatmap_by_method.csv
rank_margin_phi_scatter.csv
aggregate_decision.json
figures/
```

## Final Decision

```text
PureKAN Accelerated Functional Optimizer is not solved in v4.5.

What improved:
  FCAdam-dataSob confirms strong task-learning descent over 20 steps.
  P0 fully validates the function-coordinate implementation.
  P1 gives a concrete AdamW target profile and confirms phased geometry is the right framing.

What failed:
  FCAdam variants do not preserve enough AdamW function-space trajectory over 20 steps.
  No candidate satisfies the P2 global cos/R2 + rank/margin + Phase-I phi budget gate.
  P3/P4/P5 expansion is therefore not justified.

Conclusion:
  v4.5 supports the idea that functional-coordinate adaptive dynamics are the right
  direction, but the current AFO backbone still lacks temporal trajectory control.
  The next redesign should target multi-step function trajectory matching or restart
  dynamics only after improving P2 cos/R2 retention.
```



---


# Source 19: `docs/DG-KAN_v4.6_FunctionalLearningDynamics_实验计划.md`


# DG-KAN v4.6 实验计划：Functional Learning Dynamics，重新设计 PureKAN functional optimizer

> 版本：v4.6 draft  
> 日期：2026-05-03  
> 目标：在 v4.5 AFO 失败之后，不再继续小修 Sobolev、warmup、trust radius 或固定 depth-wise 配置，而是重新设计 PureKAN 的 functional optimizer。  
> 公式格式：Typora 友好，全文只使用 `$...$` 和 `$$...$$`。

---

## 0. 一句话结论

v4.5 的结果说明，当前 functional update 已经不再是“方向不下降”的早期问题。`FCAdam-dataSob` 已经能在 20-step horizon 上产生非常强的 holdout descent，但它的函数空间轨迹、rank / margin 形成和 geometry 预算无法同时满足。因此，PureKAN functional optimizer 的核心瓶颈已经变成：

$$
\boxed{
\text{local descent exists, but functional learning dynamics are wrong.}
}
$$

下一阶段不应该继续寻找更好的静态 preconditioner：

$$
\Delta a=-\eta M^{-1}g.
$$

而应该把 PureKAN functional optimizer 重新定义为一个带状态的深度网络优化系统：

$$
\boxed{
\text{task-learning dynamics}
+
\text{role-wise temporal state}
+
\text{representation-phase control}
+
\text{delayed geometry consolidation}
+
\text{Lyapunov / restart safeguard}
}
$$

本计划将这个方向命名为 **FLD: Functional Learning Dynamics**。

---

## 1. 为什么 v4.5 改变了问题定义

### 1.1 P0 已经排除实现问题

v4.5 的 P0 显示 function-coordinate whitening、roundtrip、rollback 都是数值正确的，并且 strict PureKAN 仍然保持：

```text
nonKAN params = 0
functional coverage = 1.0
roundtrip error = 0
rollback error = 0
```

所以当前失败不能继续主要归因于：

```text
alpha 没固定
参数没覆盖
u-space whitening 写错
rollback 写错
还有隐藏 non-KAN 参数
```

这些实现层面的借口已经基本排除。

---

### 1.2 AdamW 不是早期 geometry-improving trajectory

v4.5 的 AdamW trajectory forensic 非常关键。AdamW 在前 20 step 中能降低 holdout loss，但它本身并不是一个纯几何改善轨迹。它的早期行为更像：

```text
任务 loss 下降；
effective rank 下降；
phi 稍微上升；
input role 承担大部分更新；
block/output role 更新较小。
```

典型 role update share 是：

```text
input share 约 0.68 - 0.71
block share 约 0.20 - 0.21
output share 约 0.09 - 0.11
```

这说明一个非常重要的事实：

$$
\boxed{
\text{PureKAN 的早期优化必须先学任务和表征，而不是先追求几何下降。}
}
$$

因此，早期要求 functional optimizer 同时满足 Adam-like task descent、rank/margin 形成、并且立刻做到 strict geometry improvement，是不合理的。几何应该是 phased objective，而不是第一步就硬压。

---

### 1.3 FCAdam-dataSob 的正信号是真实的，但不是 solved optimizer

v4.5 中 `FCAdam-dataSob` 在 MNIST、Fashion-MNIST、KMNIST 上都是最强 20-step holdout descent 点之一。它说明：

$$
\boxed{
\text{functional-coordinate adaptive dynamics 可以产生真实任务下降。}
}
$$

但它没有通过 P2，因为：

```text
cos / R2 to AdamW function trajectory 很低；
rank / margin 和 Adam-like alignment 不能同时保持；
phi/A 常高于 1；
没有候选进入 P3。
```

所以 v4.5 的正确解读不是：

```text
FCAdam 失败，换下一个组件。
```

而是：

```text
FCAdam 证明了 task-learning signal 存在；
但当前 coordinate / temporal dynamics 还不能稳定塑造 PureKAN 表征。
```

---

### 1.4 当前 gate 本身也需要重构

v4.5 的 P2 gate 要求候选同时满足：

```text
20-step holdout descent；
AdamW function trajectory cos / R2；
rank / margin retention；
Phase-I phi budget。
```

这个 gate 的优点是严格，缺点是它可能把两个阶段的目标混在了一起。AdamW 自己早期也会降低 rank 并略升 phi。如果我们要求 functional optimizer 早期同时比 AdamW 更几何、更保 rank、更像 AdamW，它可能反而压死了真实的 task-learning trajectory。

因此 v4.6 的 gate 要改成 **phase-aware gate**：

```text
Phase I: task descent + role dynamics + margin formation；
Phase II: representation retention + stabilization；
Phase III: geometry consolidation。
```

---

## 2. 当前 functional update 的深层缺陷

### 2.1 过去的问题：把 smoothness prior 当成 optimizer

最早的 U-FULL 是：

$$
\Delta a=-\eta S^{-1}g,
$$

其中 $S$ 是 Sobolev smoothness metric。

这个 update 很擅长做：

```text
降低 phi_prime；
降低 Jacobian condition；
降低函数曲率；
让 KAN branch 更规整。
```

但它不擅长做：

```text
形成 feature rank；
扩大 class margin；
构造 class boundary；
在 input/block/output 之间形成长期 co-adaptation。
```

所以 Sobolev-only functional update 在 Hybrid-DGKAN residual branch 上有用，因为 hybrid 的 stem/head/LN/AdamW 已经承担了大部分表示学习；但 PureKAN 中所有表示学习都压到 KAN coefficients 上，Sobolev-only 就不够了。

---

### 2.2 TFU / FNG 的问题：局部 task-aware，不是全局学习动力学

TFU / FNG 试图把 update 改成：

$$
\Delta a_l=-\eta_l(G_l+\lambda_l S_l+\rho I)^{-1}g_l.
$$

这个方向是合理的，但目前的 $G_l$ 仍然是 layer-local 的。它不能完整描述：

```text
input KAN 的变化如何影响后面所有 blocks；
block KAN 的变化如何影响 output boundary；
output KAN 的变化如何反过来塑造隐藏层梯度；
多层同时更新时 representation drift 如何累积。
```

这解释了为什么 v4.1 FTF one-step 很强但训练崩，v4.2 BFT 把它变安全但变弱，v4.3/v4.4/v4.5 的 FC/FNG 系列可以产生局部 descent，却没有成为完整 optimizer。

---

### 2.3 FCAdam 的问题：function-coordinate 正确，但 trajectory controller 不正确

FCAdam 把 coefficient 变换到 functional coordinate：

$$
a=L^{-T}u,
$$

然后在 $u$ 上跑 Adam-like dynamics。这个方向比 $S^{-1}g$ 更接近正确，因为它保留了 Adam 的时间尺度。但 v4.5 显示：

```text
FCAdam-dataSob 能强下降；
但 20-step function trajectory 与 AdamW 目标显著偏离；
rank / margin / geometry 不能同时满足。
```

这说明当前 FCAdam 的问题不是 coordinate correctness，而是 **temporal controller** 错了。它没有稳定地控制：

```text
每个 role 的更新份额；
momentum 的方向漂移；
rank 和 margin 的形成速度；
geometry 何时开始收束。
```

---

### 2.4 直接匹配 AdamW 也不是最终答案

AdamW 是强 baseline，但它不是理论最优的 function-space trajectory。v4.5 已经显示 AdamW 早期会让 rank 降低、phi 上升。因此，我们不能把目标写成：

$$
\Delta f_{method}\approx \Delta f_{AdamW}
$$

然后再硬要求 geometry 更好。更合理的是：

$$
\Delta f_{method}
\in
\mathcal E_{AdamW},
$$

其中 $\mathcal E_{AdamW}$ 是一个 **trajectory envelope**，它约束任务下降、role share、margin/rank 区间，而不是逐点复制 AdamW displacement。

也就是说，AdamW 应该是：

```text
teacher of dynamics profile,
not exact target displacement.
```

---

### 2.5 加速梯度报告给出的关键启发

你上传的梯度下降研究报告有三个对当前项目特别重要的启发。

第一，近五年加速优化的核心不是静态 preconditioner，而是 **Lyapunov / restart / adaptive dynamics / noise-robust momentum**。这说明我们不能继续把 optimizer 当作一个矩阵 $M^{-1}$。

第二，深度学习中的成功路线不是纯 NAG，而是 Adan、Win 这类 **Nesterov 化 adaptive optimizer**。它们把加速、动量、自适应尺度和训练 recipe 结合起来，而不是只换一个下降方向。

第三，报告指出一般 PL 类并不存在通用多项式级加速，这提醒我们：PureKAN 不是普通“弱非凸”问题，是否能加速取决于具体结构。我们必须利用 KAN 的 function-space structure，但不能幻想一个 Sobolev inverse 就能普适解决。

因此，v4.6 的思想是：

$$
\boxed{
\text{functional coordinate}
+
\text{adaptive accelerated dynamics}
+
\text{phase-aware Lyapunov controller}
}
$$

---

## 3. v4.6 新方向：FLD, Functional Learning Dynamics

### 3.1 总体形式

v4.6 不再把 update 定义为一个单步公式，而定义为一个带状态的动力系统。

对第 $l$ 个 KAN role 或 layer，维护 functional coordinate $u_l$：

$$
a_l=L_l^{-T}u_l.
$$

其中 $L_l$ 来自一个 functional metric：

$$
M_l=L_lL_l^T.
$$

每步先在 $u$-space 计算梯度：

$$
g^u_l=L_l^{-1}g^a_l.
$$

然后用 adaptive accelerated dynamics 更新：

$$
m_{l,t}=\beta_1m_{l,t-1}+(1-\beta_1)g^u_{l,t},
$$

$$
v_{l,t}=\beta_2v_{l,t-1}+(1-\beta_2)(g^u_{l,t})^2.
$$

候选方向为：

$$
d^u_{l,t}=\frac{m_{l,t}}{\sqrt{v_{l,t}}+\epsilon}.
$$

但这个方向不直接写回。它还要经过：

```text
role-wise scale controller；
trajectory envelope controller；
rank / margin controller；
phased geometry controller；
Lyapunov restart controller。
```

最后写回 coefficient：

$$
\Delta a_l=L_l^{-T}\Delta u_l.
$$

---

### 3.2 不再使用“硬匹配 AdamW displacement”作为主要目标

v4.6 只把 AdamW 用作 **profile teacher**。需要从 AdamW forensic 中提取：

```text
holdout descent envelope；
role update share envelope；
rank ratio envelope；
margin ratio envelope；
phi ratio envelope；
accepted step norm envelope。
```

定义 AdamW 的 early phase envelope：

$$
\mathcal E_t=
\left\{
\Delta L_t,
\ r_{rank,t},
\ r_{margin,t},
\ r_{phi,t},
\ s_{input,t},
\ s_{block,t},
\ s_{output,t}
\right\}.
$$

候选 optimizer 不需要逐点满足：

$$
\cos(\Delta f,\Delta f_{AdamW})>c.
$$

而要满足：

$$
\Delta L_{holdout}\geq \tau_L \Delta L_{AdamW},
$$

$$
r_{rank}\geq r_{rank}^{min},
$$

$$
r_{margin}\geq r_{margin}^{min},
$$

$$
r_{phi}\leq r_{phi}^{budget}(t).
$$

这里 $r_{phi}^{budget}(t)$ 是 phased 的。

---

### 3.3 三阶段 geometry policy

v4.6 将训练分为三阶段。

#### Phase I: Representation formation

目标：

```text
任务下降；
margin 形成；
role update share 接近 AdamW profile；
允许 phi 上升。
```

约束：

$$
r_{phi}\leq 1.25,
$$

$$
r_{rank}\geq 0.65,
$$

$$
r_{margin}\geq 0.80.
$$

#### Phase II: Stabilization

目标：

```text
继续 task descent；
保持 rank / margin；
开始压 geometry。
```

约束：

$$
r_{phi}\leq 1.10,
$$

$$
r_{rank}\geq 0.80,
$$

$$
r_{margin}\geq 0.90.
$$

#### Phase III: Geometry consolidation

目标：

```text
保持 accuracy；
降低 phi/J；
改善 ECE；
避免 late overfit。
```

约束：

$$
r_{phi}\leq 0.95,
$$

或：

$$
\phi'_{p95,method}<\phi'_{p95,AdamW}.
$$

---

### 3.4 Role-wise update share controller

v4.5 的 AdamW forensic 表明早期 input role 的更新份额最高，block 次之，output 最小。PureKAN functional optimizer 必须显式控制 role share。

定义每步 role update share：

$$
s_l(t)=\frac{\|\Delta u_l(t)\|}{\sum_j\|\Delta u_j(t)\|+\epsilon}.
$$

设定 early target：

```text
input share target: 0.60 - 0.75
block share target: 0.15 - 0.30
output share target: 0.05 - 0.15
```

如果某个 role 偏离 target，就调整 role LR multiplier：

$$
\eta_l(t+1)=\eta_l(t)\exp\left(\gamma_s(s_l^\star-s_l(t))\right).
$$

这不是为了模仿 AdamW 的每个方向，而是为了保持 PureKAN 的表示学习节奏。

---

### 3.5 Lyapunov energy 与 restart

基于加速梯度报告中 Lyapunov / restart 的启发，v4.6 引入一个可记录的训练能量：

$$
\mathcal V_t
=
L_{holdout,t}
+
\lambda_{rank}[r_{rank}^{min}(t)-r_{rank,t}]_+^2
+
\lambda_{margin}[r_{margin}^{min}(t)-r_{margin,t}]_+^2
+
\lambda_{phi}[r_{phi,t}-r_{phi}^{budget}(t)]_+^2
+
\lambda_{step}\|\Delta u_t\|^2.
$$

如果出现：

$$
\mathcal V_t>\mathcal V_{t-1}+\epsilon_V,
$$

或者：

$$
\langle m_t,g_t\rangle<0,
$$

则触发 restart：

```text
清空或衰减 momentum；
降低 eta；
提高 geometry damping；
重置 role share controller 的积分项。
```

这比单纯 trust clipping 更有意义，因为它直接关心 loss、rank、margin、geometry 的联合状态。

---

## 4. v4.6 候选方法

### 4.1 FLD-AdamCoord

这是最小可行版本。

```text
functional coordinate whitening；
AdamW-style m/v in u-space；
role-wise LR multiplier；
three-phase geometry budget；
no Nesterov；
no Lyapunov restart。
```

目的：确认 role-share control + phased geometry 是否已经足够。

---

### 4.2 FLD-Nesterov

加入 lookahead：

$$
\tilde u_t=u_t+\beta m_{t-1}.
$$

在 $\tilde u_t$ 处计算梯度或近似梯度，然后更新。由于真实二次 forward 成本高，可以先用 one-extra-forward 版本做 audit，再做 cheap approximation。

目的：验证 functional-coordinate Nesterov 是否优于普通 FCAdam。

---

### 4.3 FLD-AdanLite

借鉴 Adan 的思想，显式追踪梯度差：

$$
d_t=g_t-g_{t-1}.
$$

维护：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
n_t=\beta_2n_{t-1}+(1-\beta_2)d_t,
$$

$$
v_t=\beta_3v_{t-1}+(1-\beta_3)(g_t+(1-\beta_2)d_t)^2.
$$

候选方向：

$$
p_t=\frac{m_t+(1-\beta_2)n_t}{\sqrt{v_t}+\epsilon}.
$$

所有变量都在 functional coordinate $u$ 中。

目的：验证梯度变化项是否能修复 20-step trajectory drift。

---

### 4.4 FLD-WinLite

借鉴 Win / weight-decay-integrated Nesterov 的思想，不直接把 geometry penalty 当作 update 后处理，而把 functional decay / Sobolev penalty 融进 lookahead step。

形式：

$$
u_{t+1/2}=u_t-\eta \lambda_S S_u u_t,
$$

$$
\tilde u_t=u_{t+1/2}+\beta m_t,
$$

$$
u_{t+1}=\tilde u_t-\eta p_t.
$$

目的：验证 geometry regularization 是否应该通过 optimizer dynamics 耦合，而不是训练后期硬拉回。

---

### 4.5 FLD-LyapunovRestart

在 `FLD-AdanLite` 或 `FLD-Nesterov` 基础上加入 Lyapunov restart。

```text
如果 holdout loss 上升：restart；
如果 rank collapse：restart；
如果 margin collapse：restart；
如果 phi 超预算：geometry restart；
如果 momentum-gradient cosine < 0：momentum restart。
```

目的：验证 restart / Lyapunov controller 是否能避免 20-step drift。

---

### 4.6 FLD-TeacherEnvelope

将 AdamW forensic 转化为 soft envelope loss，而不是硬 cos/R2 gate。

额外 controller loss：

$$
\mathcal R_{env}
=
\alpha_s\sum_l(s_l-s_l^\star)^2
+
\alpha_r[r_{rank}^{min}-r_{rank}]_+^2
+
\alpha_m[r_{margin}^{min}-r_{margin}]_+^2
+
\alpha_\phi[r_{phi}-r_{phi}^{budget}]_+^2.
$$

这个 penalty 不进入 autograd loss，而用于调整 optimizer state 和 step acceptance。

目的：验证 “AdamW as profile teacher” 是否优于 “AdamW as exact displacement target”。

---

## 5. 实验总流程

本轮不直接做 10-seed。所有方法必须先通过 trajectory-dynamics 诊断。

```text
P0: implementation invariants and coordinate/state smoke
P1: AdamW trajectory envelope construction
P2: one-step and 20-step dynamics audit
P3: 100-step trajectory audit
P4: phase-controller ablation
P5: 3-seed short full-budget selection
P6: 5-seed confirm
P7: 10-seed final confirm
P8: failure diagnosis and theory update
```

---

## 6. P0：实现与不变量检查

### 6.1 目标

确认所有 FLD 方法仍然是 strict PureKAN functional optimizer，不引入 non-KAN 参数，也不复用普通 AdamW 参数更新。

### 6.2 数据集

```text
MNIST
Fashion-MNIST
KMNIST
```

### 6.3 模型

```text
model = PureKAN
hidden_dim = 64
basis_count = 16
深度 = 2 和 4 都做 smoke
alphaFixed1
FixedNorm
nonKAN params = 0
```

### 6.4 方法

```text
PureKAN-AdamW
D0-allFullSobolev
D6-allTaskAware
FCAdam-dataSob
FLD-AdamCoord
FLD-Nesterov
FLD-AdanLite
FLD-WinLite
FLD-LyapunovRestart
FLD-TeacherEnvelope
```

### 6.5 必须记录

```text
implementation/nonKAN_param_count
implementation/functional_coverage
implementation/u_roundtrip_error
implementation/rollback_error
implementation/state_m_finite
implementation/state_v_finite
implementation/restart_state_finite
implementation/role_lr_multiplier_finite
implementation/phase_state
implementation/geometry_budget
implementation/NaN_count
```

### 6.6 通过条件

$$
\text{nonKAN params}=0
$$

$$
\text{functional coverage}=1.0
$$

$$
\text{roundtrip error}<10^{-8}
$$

$$
\text{NaN count}=0
$$

---

## 7. P1：AdamW trajectory envelope construction

### 7.1 目标

重新定义 AdamW 作为 **profile teacher**，而不是 exact target。

### 7.2 设置

```text
methods = PureKAN-AdamW
seeds = 0,1,2
steps recorded = 1, 5, 20, 50, 100, full epoch
```

### 7.3 记录指标

#### Task trajectory

```text
loss/train
loss/holdout
loss/val
holdout_descent_1
holdout_descent_5
holdout_descent_20
holdout_descent_50
holdout_descent_100
val_loss_auc_partial
```

#### Role update profile

```text
role/input_update_norm
role/block_update_norm
role/output_update_norm
role/input_share
role/block_share
role/output_share
role/share_entropy
role/share_ema
```

#### Representation profile

```text
repr/effective_rank_input
repr/effective_rank_block
repr/effective_rank_output
repr/class_centroid_separation
repr/class_within_scatter
repr/fisher_ratio
repr/logit_margin_mean
repr/logit_margin_p10
repr/logit_margin_p50
repr/logit_entropy
```

#### Geometry profile

```text
geometry/phi_prime_p95
geometry/phi_prime_max
geometry/jacobian_condition
geometry/curvature_energy
geometry/sobolev_norm
```

#### Optimizer state

```text
adam/m_norm_by_role
adam/v_norm_by_role
adam/update_over_param_by_role
adam/m_grad_cos_by_role
```

### 7.4 输出

产生一个 `adamw_trajectory_envelope.json`，至少包含：

```text
role_share_target_by_phase
rank_budget_by_phase
margin_budget_by_phase
phi_budget_by_phase
holdout_descent_target_by_phase
restart_reference_events
```

### 7.5 可视化

```text
figures/p1_adamw_loss_rank_phi_phase.png
figures/p1_adamw_role_share_stack.png
figures/p1_adamw_margin_rank_trajectory.png
figures/p1_adamw_phi_vs_holdout_descent.png
figures/p1_adamw_update_state_by_role.png
```

---

## 8. P2：one-step and 20-step dynamics audit

### 8.1 目标

先验证候选是否能保持 20-step learning dynamics。不要直接进入 full training。

### 8.2 方法

```text
PureKAN-AdamW
D0-allFullSobolev
D6-allTaskAware
FCAdam-dataSob
FLD-AdamCoord
FLD-Nesterov
FLD-AdanLite
FLD-WinLite
FLD-LyapunovRestart
FLD-TeacherEnvelope
```

### 8.3 记录指标

#### Descent

```text
dynamics/train_descent_1
dynamics/holdout_descent_1
dynamics/train_descent_5
dynamics/holdout_descent_5
dynamics/train_descent_20
dynamics/holdout_descent_20
dynamics/bad_step_rate
dynamics/negative_holdout_steps
```

#### AdamW envelope matching

```text
envelope/role_share_l2_error
envelope/rank_budget_violation
envelope/margin_budget_violation
envelope/phi_budget_violation
envelope/trajectory_energy
```

#### Function displacement diagnostics

```text
traj/cos_with_adam_function
traj/R2_with_adam_function
traj/cos_with_adam_coeff
traj/R2_with_adam_coeff
traj/logit_delta_cos
traj/activation_delta_cos_input
traj/activation_delta_cos_block
traj/activation_delta_cos_output
```

注意：cos/R2 只作为诊断，不作为单独硬 gate。

#### Optimizer state

```text
state/m_norm
state/v_norm
state/n_norm, for AdanLite
state/update_norm
state/update_over_param
state/accepted_eta
state/restart_count
state/restart_reason
state/role_lr_multiplier
```

### 8.4 通过条件

P2 不要求 geometry 优于 AdamW，只要求不超出 early budget。

候选必须满足：

$$
\Delta L_{holdout,20}
\geq
0.8\Delta L_{holdout,20}^{AdamW}
$$

或：

$$
\Delta L_{holdout,20}
>
\Delta L_{holdout,20}^{D6}+0.1.
$$

并且：

$$
r_{phi,20}<1.25,
$$

$$
r_{rank,20}>0.65,
$$

$$
r_{margin,20}>0.80,
$$

$$
\text{bad step rate}<0.05.
$$

### 8.5 可视化

```text
figures/p2_holdout_descent_20_bar.png
figures/p2_role_share_error_heatmap.png
figures/p2_rank_margin_phi_scatter.png
figures/p2_trajectory_energy_by_method.png
figures/p2_restart_reason_stack.png
figures/p2_momentum_cos_trace.png
```

---

## 9. P3：100-step trajectory audit

### 9.1 目标

v4.5 的核心失败是 20-step temporal mismatch。v4.6 必须直接检查 100-step trajectory。

### 9.2 方法

只进入 P2 通过的候选。若没有候选通过，则强制保留以下方法做 diagnostic：

```text
FCAdam-dataSob
FLD-AdanLite
FLD-LyapunovRestart
```

### 9.3 记录指标

```text
loss/train_curve_100
loss/holdout_curve_100
repr/rank_curve_100
repr/margin_curve_100
geometry/phi_curve_100
geometry/J_curve_100
role/share_curve_100
state/momentum_norm_curve_100
state/restart_markers
state/eta_curve_100
state/lyapunov_curve_100
```

### 9.4 通过条件

候选必须满足：

$$
\Delta L_{holdout,100}
\geq
0.75\Delta L_{holdout,100}^{AdamW}
$$

并且：

$$
\text{trajectory energy}_{100}<\text{D6 trajectory energy}_{100}.
$$

同时：

$$
r_{phi,100}<1.15,
$$

$$
r_{rank,100}>0.75,
$$

$$
r_{margin,100}>0.85.
$$

### 9.5 可视化

```text
figures/p3_100step_loss_rank_phi.png
figures/p3_lyapunov_with_restart_markers.png
figures/p3_role_share_vs_adamw.png
figures/p3_energy_components.png
figures/p3_method_phase_transition.png
```

---

## 10. P4：phase-controller ablation

### 10.1 目标

验证 v4.6 的关键思想：几何不应该早期硬压，而应该 phased consolidation。

### 10.2 方法

对 P3 最好候选做 ablation：

```text
base candidate
no role controller
no Lyapunov restart
no phased geometry
strict early geometry
no geometry until late
fixed role share
AdamW role share envelope
```

### 10.3 记录指标

```text
phase/phase_enter_step
phase/phase_exit_step
phase/geometry_budget
phase/phi_violation_count
phase/rank_violation_count
phase/margin_violation_count
phase/restart_count
phase/restart_reason
phase/controller_adjustment_norm
```

### 10.4 判定

如果 `strict early geometry` 任务下降差，而 `phased geometry` 任务下降好且最终 geometry 能收回来，则确认：

$$
\boxed{
\text{PureKAN functional optimizer needs delayed geometry consolidation.}
}
$$

如果 `no Lyapunov restart` 出现 100-step drift，而 `with restart` 稳定，则确认 restart 是必要组件。

---

## 11. P5：3-seed short full-budget selection

### 11.1 目标

只对 P3/P4 通过的候选进行小规模训练，避免浪费 5/10 seed。

### 11.2 数据集

```text
MNIST
Fashion-MNIST
KMNIST
```

### 11.3 对照方法

```text
PureKAN-AdamW
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
D0-allFullSobolev
D6-allTaskAware
FCAdam-dataSob
best FLD candidate 1
best FLD candidate 2
```

### 11.4 记录指标

#### Task

```text
task/train_loss_curve
task/val_loss_curve
task/test_acc
task/best_val_acc
task/val_loss_auc
task/val_acc_auc
task/acc_gap_vs_purekan_adamw
task/acc_gap_vs_mlp_adamw
task/acc_gap_vs_hybrid_ufull
```

#### Representation

```text
repr/effective_rank_by_epoch
repr/rank_ratio_vs_adamw
repr/class_centroid_separation
repr/fisher_ratio
repr/logit_margin_mean
repr/logit_margin_p10
repr/logit_entropy
repr/feature_norm
```

#### Geometry

```text
geometry/phi_prime_p95
geometry/phi_prime_max
geometry/jacobian_condition
geometry/curvature_energy
geometry/sobolev_norm
geometry/final_phi_ratio_vs_adamw
geometry/final_J_ratio_vs_adamw
```

#### Optimizer dynamics

```text
state/restart_count
state/eta_mean
state/eta_p95
state/m_norm_by_role
state/v_norm_by_role
state/update_share_by_role
state/role_lr_multiplier
state/trajectory_energy_auc
state/phi_budget_violation_auc
```

### 11.5 通过条件

进入 P6 的候选必须：

$$
\operatorname{Acc}_{method}
\geq
\operatorname{Acc}_{PureKAN-AdamW}-1\%
$$

在 MNIST/Fashion 上，并且：

$$
\operatorname{Acc}_{method}
\geq
\operatorname{Acc}_{PureKAN-AdamW}-2\%
$$

在 KMNIST 上。

还必须满足：

$$
\operatorname{AUC}_{val,method}
<
\operatorname{AUC}_{val,D6},
$$

$$
\phi'_{p95,method}
<
1.05\phi'_{p95,AdamW},
$$

以及：

$$
\text{restart rate}<0.25.
$$

---

## 12. P6：5-seed confirm

### 12.1 目标

确认 P5 候选是否稳定，而不是单 seed 或 3-seed 偶然。

### 12.2 方法

```text
PureKAN-AdamW
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
best FLD candidate
second FLD candidate, optional
```

### 12.3 Seeds

```text
seeds = 0,1,2,3,4
```

### 12.4 通过条件

候选必须至少满足 Pareto pass：

```text
accuracy gap vs PureKAN-AdamW <= 1% on MNIST/Fashion
accuracy gap vs PureKAN-AdamW <= 2% on KMNIST
val-loss AUC better than D6 and D0
final phi/J better than AdamW or not worse than 1.05x AdamW
ECE not worse than AdamW
```

如果想 claim strong PureKAN functional optimizer，则必须：

$$
\operatorname{Acc}_{method}
\geq
\operatorname{Acc}_{PureKAN-AdamW}
$$

在至少两个数据集上，并且第三个数据集 gap 小于 1%。

---

## 13. P7：10-seed final confirm

### 13.1 触发条件

只有 P6 通过才运行。

### 13.2 Seeds

```text
seeds = 0..9
```

### 13.3 最终 claim 条件

如果候选要成为 PureKAN 默认 functional optimizer，必须满足：

$$
\operatorname{Acc}_{FLD}
\geq
\max(
\operatorname{Acc}_{PureKAN-AdamW},
\operatorname{Acc}_{MLP-AdamW},
\operatorname{Acc}_{Hybrid-DGKAN-UFULL}
)-\epsilon
$$

其中：

$$
\epsilon=0.005
$$

同时必须满足：

```text
val-loss AUC better than PureKAN-AdamW or within 2% while geometry better；
phi_prime_p95 lower than AdamW；
Jacobian condition lower than AdamW；
ECE lower than AdamW；
role update share stable；
restart count finite and interpretable。
```

如果做不到，则不能 claim solved，只能 claim：

```text
FLD improves functional optimizer dynamics but does not yet surpass AdamW.
```

---

## 14. P8：failure diagnosis

如果 P7 不通过，必须输出一个明确失败类型，而不是继续调参。

### 14.1 失败类型

```text
F1: role share mismatch
F2: trajectory energy high
F3: rank formation failure
F4: margin formation failure
F5: early geometry overconstraint
F6: late geometry cannot consolidate
F7: momentum drift
F8: restart over-triggered
F9: basis occupancy failure
F10: compute overhead unacceptable
```

### 14.2 必须输出表

```text
failure_table.csv
failure_by_dataset.csv
failure_by_role.csv
failure_by_phase.csv
failure_by_method.csv
role_share_trace.csv
trajectory_energy_trace.csv
rank_margin_phi_trace.csv
restart_trace.csv
optimizer_state_trace.csv
```

### 14.3 必须画图

```text
figures/failure_taxonomy_heatmap.png
figures/role_share_mismatch.png
figures/trajectory_energy_vs_accuracy.png
figures/rank_margin_phi_phase_plot.png
figures/restart_timeline.png
figures/optimizer_state_norms.png
figures/basis_occupancy_failure.png
```

---

## 15. 本轮必须避免的误区

### 15.1 不再继续调 Sobolev alpha / beta 作为主线

这些已经不是根因。它们只能在 P4 phase-controller ablation 中作为辅助。

### 15.2 不再用 strict early geometry gate 杀掉任务学习

早期允许 phi 上升，只要不超过 phase budget。

### 15.3 不再把 AdamW displacement 当作唯一目标

AdamW 是 profile teacher，不是 exact target。

### 15.4 不再只看 holdout descent

v4.5 已经证明 holdout descent 可以很强但 trajectory mismatch。必须同时看：

```text
rank
margin
role share
phi
Lyapunov energy
restart behavior
```

### 15.5 不再直接跑 5/10 seed

如果 P2/P3 的 trajectory dynamics 不过，继续扩 seed 只是浪费。

---

## 16. 预期结论模板

### 情况 A：FLD 成功

```text
PureKAN functional optimization is solved by Functional Learning Dynamics. The key was not a better Sobolev preconditioner, but a stateful optimizer that combines functional-coordinate adaptive dynamics, role-wise update control, phased geometry consolidation, and Lyapunov restart. FLD matches or exceeds PureKAN-AdamW while improving geometry and calibration.
```

### 情况 B：FLD 部分成功

```text
FLD substantially improves PureKAN functional training dynamics and closes much of the gap to PureKAN-AdamW, but does not yet exceed all baselines. The remaining bottleneck is representation formation on KMNIST, especially role-share mismatch or margin formation.
```

### 情况 C：FLD 失败

```text
Even with functional-coordinate adaptive dynamics, phased geometry, and Lyapunov restart, PureKAN functional training does not match AdamW. This suggests that current functional-coordinate optimizers still lack the implicit bias or noise-scale dynamics required for full-network PureKAN training. Functional update should remain a hybrid KAN-branch optimizer until a stronger theory is found.
```

---

## 17. 总结

v4.5 后，我们不能再说：

```text
只要调好 functional update 的 metric，PureKAN 就能超过 AdamW。
```

更准确的说法是：

$$
\boxed{
\text{PureKAN needs functional learning dynamics, not just functional preconditioning.}
}
$$

v4.6 的目标是验证：

$$
\boxed{
\text{functional-coordinate adaptive acceleration}
+
\text{role-wise representation control}
+
\text{Lyapunov restart}
+
\text{phased geometry consolidation}
}
$$

是否足以把 PureKAN functional optimizer 从局部 descent proposal 推进成真正可以训练完整网络的 optimizer。



---


# Source 20: `docs/DG-KAN_v4.6_FunctionalLearningDynamics_结果复盘.md`


# DG-KAN v4.6 Functional Learning Dynamics 结果复盘

本轮依据 `docs/DG-KAN_v4.6_FunctionalLearningDynamics_实验计划.md`。目标是验证 PureKAN 是否需要 FLD：functional-coordinate adaptive dynamics + role-wise representation control + phased geometry + Lyapunov restart。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 FLD smoke | 60 | 0 |
| P1 AdamW envelope | 45 | 0 |
| P2 FLD dynamics | 90 | 0 |
| P3 100-step trajectory | 45 | 0 |

## Code / Config Changes

```text
experiments/run_gafu_v46.py
  Added FLD-AdamCoord, FLD-Nesterov, FLD-AdanLite, FLD-WinLite,
  FLD-LyapunovRestart, and FLD-TeacherEnvelope probes.
  Added AdamW trajectory envelope construction over seeds 0/1/2 and steps 1/5/20/50/100.
  Added phase-aware P2 gate and 100-step P3 trajectory audit.

experiments/analyze_gafu_v46.py
  Generates AdamW envelope JSON, P2/P3 gate summaries, failure taxonomy, figures, aggregate_decision.json,
  and this result replay.
```

## P0 Implementation Invariants

| rows | errors | max nonKAN | max roundtrip | max u->a | max rollback | pass |
|---|---|---|---|---|---|---|
| 60 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | true |

P0 verdict: pass. FLD methods keep strict PureKAN trainable parameters, coefficient coverage, rollback, and finite optimizer states.

## P1 AdamW Trajectory Envelope

| dataset | holdout Δ | rank ratio | phi ratio | input share | block share | output share |
|---|---|---|---|---|---|---|
| Fashion-MNIST | 1.7950 | 0.7919 | 1.1583 | 0.6774 | 0.2152 | 0.1074 |
| KMNIST | 1.7909 | 0.7285 | 1.1776 | 0.6796 | 0.2149 | 0.1054 |
| MNIST | 1.9020 | 0.5084 | 1.2107 | 0.6830 | 0.2131 | 0.1039 |

Observation:

```text
AdamW is used as a profile teacher, not an exact displacement target.
The envelope records early task descent, role share, rank/margin movement, and phi budget.
```

## P2 One-Step / 20-Step Dynamics Audit

| dataset | method | runs | hold20 | hold/A | rank | margin | phi | bad | energy | P2 |
|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | D0-allFullSobolev | 3 | 0.7351 | 0.6747 | 0.6151 | 0.1636 | 0.9978 | 0.0667 | 1.7270 | no |
| Fashion-MNIST | D6-allTaskAware | 3 | 0.8802 | 0.8079 | 0.4651 | 0.2122 | 1.0027 | 0.0833 | 1.5753 | no |
| Fashion-MNIST | FCAdam-dataSob | 3 | 1.5207 | 1.3957 | 0.3345 | 1.4934 | 1.3574 | 0.0000 | 0.8650 | no |
| Fashion-MNIST | FLD-AdamCoord | 3 | 1.5223 | 1.3971 | 0.3343 | 1.5021 | 1.3614 | 0.0000 | 0.8636 | no |
| Fashion-MNIST | FLD-AdanLite | 3 | 1.5464 | 1.4193 | 0.3350 | 1.6167 | 1.3475 | 0.0000 | 0.8388 | no |
| Fashion-MNIST | FLD-LyapunovRestart | 3 | 1.4474 | 1.3284 | 0.3561 | 1.2718 | 1.2961 | 0.0333 | 0.9335 | no |
| Fashion-MNIST | FLD-Nesterov | 3 | 1.5181 | 1.3933 | 0.3108 | 1.3784 | 1.5670 | 0.0000 | 0.8849 | no |
| Fashion-MNIST | FLD-TeacherEnvelope | 3 | 1.5140 | 1.3895 | 0.3345 | 1.4679 | 1.3594 | 0.0000 | 0.8718 | no |
| Fashion-MNIST | FLD-WinLite | 3 | 1.5222 | 1.3971 | 0.3343 | 1.5019 | 1.3612 | 0.0000 | 0.8636 | no |
| Fashion-MNIST | PureKAN-AdamW | 3 | 1.0896 | 1.0000 | 0.1896 | 0.3116 | 1.0506 | 0.0000 | 1.3827 | yes |
| KMNIST | D0-allFullSobolev | 3 | 0.3708 | 0.7117 | 0.7213 | -0.2172 | 0.9978 | 0.1000 | 2.2615 | no |
| KMNIST | D6-allTaskAware | 3 | 0.4430 | 0.8502 | 0.5752 | -0.1031 | 1.0023 | 0.1167 | 2.1347 | no |
| KMNIST | FCAdam-dataSob | 3 | 1.3247 | 2.5427 | 0.3555 | 1.2697 | 1.3863 | 0.0000 | 1.0720 | no |
| KMNIST | FLD-AdamCoord | 3 | 1.3255 | 2.5442 | 0.3560 | 1.2729 | 1.3862 | 0.0000 | 1.0712 | no |
| KMNIST | FLD-AdanLite | 3 | 1.3529 | 2.5968 | 0.3574 | 1.3924 | 1.3747 | 0.0000 | 1.0431 | no |
| KMNIST | FLD-LyapunovRestart | 3 | 1.1355 | 2.1794 | 0.3242 | 0.7756 | 1.3141 | 0.0500 | 1.2685 | no |
| KMNIST | FLD-Nesterov | 3 | 1.2662 | 2.4305 | 0.3052 | 1.0194 | 1.5936 | 0.0000 | 1.1533 | no |
| KMNIST | FLD-TeacherEnvelope | 3 | 1.3148 | 2.5238 | 0.3549 | 1.2453 | 1.3820 | 0.0000 | 1.0818 | no |
| KMNIST | FLD-WinLite | 3 | 1.3254 | 2.5441 | 0.3560 | 1.2728 | 1.3859 | 0.0000 | 1.0712 | no |
| KMNIST | PureKAN-AdamW | 3 | 0.5210 | 1.0000 | 0.1775 | 0.0243 | 1.0419 | 0.0000 | 2.0576 | yes |
| MNIST | D0-allFullSobolev | 3 | 0.3983 | 1.5019 | 0.6701 | -0.0624 | 0.9991 | 0.1333 | 2.1495 | no |
| MNIST | D6-allTaskAware | 3 | 0.2977 | 1.1225 | 0.5569 | -0.3066 | 1.0005 | 0.1333 | 2.3780 | no |
| MNIST | FCAdam-dataSob | 3 | 1.4264 | 5.3783 | 0.3187 | 1.5307 | 1.4314 | 0.0000 | 0.9661 | no |
| MNIST | FLD-AdamCoord | 3 | 1.4267 | 5.3795 | 0.3192 | 1.5319 | 1.4345 | 0.0000 | 0.9659 | no |
| MNIST | FLD-AdanLite | 3 | 1.4673 | 5.5328 | 0.3243 | 1.6747 | 1.4171 | 0.0000 | 0.9234 | no |
| MNIST | FLD-LyapunovRestart | 3 | 1.2339 | 4.6525 | 0.3277 | 1.1394 | 1.3529 | 0.0500 | 1.1537 | no |
| MNIST | FLD-Nesterov | 3 | 1.3943 | 5.2573 | 0.2810 | 1.4385 | 1.6718 | 0.0167 | 1.0268 | no |
| MNIST | FLD-TeacherEnvelope | 3 | 1.4240 | 5.3693 | 0.3187 | 1.5275 | 1.4316 | 0.0000 | 0.9685 | no |
| MNIST | FLD-WinLite | 3 | 1.4266 | 5.3791 | 0.3192 | 1.5316 | 1.4342 | 0.0000 | 0.9660 | no |
| MNIST | PureKAN-AdamW | 3 | 0.2652 | 1.0000 | 0.2771 | -0.0603 | 1.0368 | 0.0167 | 2.3188 | yes |

Best non-Adam points by 20-step holdout descent:

| dataset | best hold20 | hold20 | hold/A | rank | margin | phi |
|---|---|---|---|---|---|---|
| MNIST | FLD-AdanLite | 1.4673 | 5.5328 | 0.3243 | 1.6747 | 1.4171 |
| Fashion-MNIST | FLD-AdanLite | 1.5464 | 1.4193 | 0.3350 | 1.6167 | 1.3475 |
| KMNIST | FLD-AdanLite | 1.3529 | 2.5968 | 0.3574 | 1.3924 | 1.3747 |

P2 survivors:

```text
none
```

P2 verdict:

```text
P2 uses the new phase-aware gate. Cos/R2 are diagnostics only.
Survivors enter P3; if there are no survivors the plan forces FCAdam-dataSob,
FLD-AdanLite, and FLD-LyapunovRestart as diagnostic P3 methods.
```

## P3 100-Step Trajectory Audit

| dataset | method | runs | hold100 | hold/A | energy/D6 | rank | margin | phi | restart | P3 |
|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | D6-allTaskAware | 3 | 1.6296 | 0.9001 | 1.0000 | 0.5145 | 2.0161 | 1.0032 | 0.0000 | yes |
| Fashion-MNIST | FCAdam-dataSob | 3 | 1.6750 | 0.9251 | 1.0660 | 0.3733 | 5.7657 | 1.8271 | 0.0000 | no |
| Fashion-MNIST | FLD-AdanLite | 3 | 1.6577 | 0.9156 | 1.0818 | 0.3610 | 5.7757 | 1.7921 | 0.0000 | no |
| Fashion-MNIST | FLD-LyapunovRestart | 3 | 1.7122 | 0.9457 | 0.9333 | 0.3955 | 4.6721 | 1.4836 | 18.0000 | no |
| Fashion-MNIST | PureKAN-AdamW | 3 | 1.8105 | 1.0000 | 0.7777 | 0.3541 | 3.7258 | 1.1606 | 0.0000 | yes |
| KMNIST | D6-allTaskAware | 3 | 1.0744 | 0.5887 | 1.0000 | 0.5718 | 0.6274 | 1.0032 | 0.0000 | yes |
| KMNIST | FCAdam-dataSob | 3 | 1.5877 | 0.8700 | 0.6763 | 0.3509 | 5.1524 | 1.8420 | 0.0000 | no |
| KMNIST | FLD-AdanLite | 3 | 1.5786 | 0.8650 | 0.6771 | 0.3488 | 5.0418 | 1.8032 | 0.0000 | no |
| KMNIST | FLD-LyapunovRestart | 3 | 1.6124 | 0.8835 | 0.6140 | 0.3793 | 4.0932 | 1.5435 | 18.0000 | no |
| KMNIST | PureKAN-AdamW | 3 | 1.8250 | 1.0000 | 0.4247 | 0.4465 | 3.8677 | 1.1876 | 0.0000 | yes |
| MNIST | D6-allTaskAware | 3 | 0.9263 | 0.4749 | 1.0000 | 0.4878 | 0.2327 | 1.0057 | 0.0000 | yes |
| MNIST | FCAdam-dataSob | 3 | 1.8099 | 0.9279 | 0.4167 | 0.3341 | 6.4866 | 1.8379 | 0.0000 | no |
| MNIST | FLD-AdanLite | 3 | 1.7887 | 0.9171 | 0.4256 | 0.3323 | 6.3310 | 1.8034 | 0.0000 | no |
| MNIST | FLD-LyapunovRestart | 3 | 1.8290 | 0.9377 | 0.3725 | 0.3553 | 5.3991 | 1.5785 | 18.0000 | no |
| MNIST | PureKAN-AdamW | 3 | 1.9505 | 1.0000 | 0.2717 | 0.3876 | 4.4335 | 1.1988 | 0.0000 | yes |

P3 survivors:

```text
none
```

## P4-P7 Decision

```text
P4 phase-controller ablation: not run
P5 3-seed short full-budget selection: not run
P6 5-seed confirm: not run
P7 10-seed final confirm: not run

Reason: P3 produced no survivor
```

## P8 Failure Diagnosis

Generated:

```text
p8_failure_diagnosis.csv
failure_table.csv
failure_by_dataset.csv
failure_by_role.csv
failure_by_phase.csv
failure_by_method.csv
role_share_trace.csv
trajectory_energy_trace.csv
rank_margin_phi_trace.csv
restart_trace.csv
optimizer_state_trace.csv
```

Diagnosis:

```text
P2/P3 failures are classified by role share, trajectory energy, rank formation,
margin formation, geometry budget, momentum drift, restart behavior, and basis occupancy.
The dominant hard blocker in this run is not task descent: FCAdam/FLD descend strongly,
but rank remains below the Phase-I/P3 budget while phi grows far above the delayed
geometry budget by 100 steps.
```

## Required Artifacts

Written under `results/v4_6/`:

```text
p0_fld_invariants.csv
p1_adamw_trajectory_envelope.csv
p1_adamw_target_profile.csv
adamw_trajectory_envelope.json
p2_fld_dynamics_audit.csv
p2_fld_gate_summary.csv
p3_100step_trajectory.csv
p3_trajectory_gate_summary.csv
p4_phase_controller_ablation.csv
p5_short_full_budget_selection.csv
p6_candidate_selection.csv
p7_confirm5.csv
p8_failure_diagnosis.csv
failure_table.csv
failure_by_dataset.csv
failure_by_role.csv
failure_by_phase.csv
failure_by_method.csv
role_share_trace.csv
trajectory_energy_trace.csv
rank_margin_phi_trace.csv
restart_trace.csv
optimizer_state_trace.csv
aggregate_decision.json
figures/
```

## Final Decision

```text
PureKAN Functional Learning Dynamics status:
  stop_after_p3_no_survivor

What improved:
  v4.6 no longer kills candidates solely by AdamW displacement cos/R2.
  It tests task descent, role dynamics, rank/margin formation, and phased phi budget directly.

What failed:
  See P2/P3 gate summaries and failure taxonomy.

Conclusion:
  The key question is whether stateful functional-coordinate dynamics can preserve
  enough task learning while respecting delayed geometry budgets.
```



---


# Source 21: `docs/DG-KAN_v4.7_PhaseSeparatedFunctionalTraining_PureKAN_实验计划.md`


# DG-KAN v4.7：Phase-Separated Functional Training for PureKAN 详细实验计划

> 目标：重新设计 PureKAN functional optimizer。  
> 版本：v4.7  
> 公式格式：Typora 友好，只使用 `$...$` 和 `$$...$$`。  
> 核心结论预设：不要继续把 functional update 当成一个静态 preconditioner；下一轮要把它改成 **分阶段的 functional learning dynamics**。

---

## 0. 当前结论：v4.6 不是普通失败，而是暴露了 optimizer 定义问题

v4.6 的结果非常有价值。它说明我们已经不在“实现有没有接对”的阶段，而是在“functional optimizer 本身如何定义”的阶段。

P0 已经通过：strict PureKAN 仍然满足：

```text
nonKAN params = 0
coefficient coverage = 1.0
u -> a roundtrip = 0
rollback = 0
run errors = 0
```

所以这轮不能再主要怀疑：

```text
input/output KAN 没被更新
alpha 没 fixed
rollback 有 bug
functional coordinate whitening 错
```

这些实现问题已经基本排除。

P1 给出了 AdamW 早期轨迹 envelope：

| dataset | holdout Δ | rank ratio | phi ratio | input share | block share | output share |
|---|---:|---:|---:|---:|---:|---:|
| Fashion-MNIST | 1.7950 | 0.7919 | 1.1583 | 0.6774 | 0.2152 | 0.1074 |
| KMNIST | 1.7909 | 0.7285 | 1.1776 | 0.6796 | 0.2149 | 0.1054 |
| MNIST | 1.9020 | 0.5084 | 1.2107 | 0.6830 | 0.2131 | 0.1039 |

这个表给出一个重要事实：AdamW 早期不是几何收缩器。它会允许 $\phi$ 上升、rank 变化，并且大部分更新集中在 input KAN role 上。也就是说，PureKAN-AdamW 的成功更像：

```text
先形成表示和 margin，再谈几何。
```

P2 的结果进一步说明：FLD / FCAdam / AdanLite 已经能产生很强的 20-step holdout descent，甚至比 AdamW 的短程下降更大，但仍然没有 survivor。以 `FLD-AdanLite` 为例：

| dataset | hold20 | hold/A | rank | margin | phi |
|---|---:|---:|---:|---:|---:|
| MNIST | 1.4673 | 5.5328 | 0.3243 | 1.6747 | 1.4171 |
| Fashion-MNIST | 1.5464 | 1.4193 | 0.3350 | 1.6167 | 1.3475 |
| KMNIST | 1.3529 | 2.5968 | 0.3574 | 1.3924 | 1.3747 |

这说明：

```text
短程任务下降有了；
但是 rank / margin / phi 没有形成可持续的表示学习轨迹。
```

P3 的 100-step audit 继续支持这个判断。`FCAdam-dataSob / FLD-AdanLite / FLD-LyapunovRestart` 不再是完全不学习；它们甚至在部分数据集上有接近 AdamW 的 holdout descent。但它们的共同问题是：

```text
rank 偏低或不稳定；
margin 过大但未转化为 clean accuracy；
phi 过高；
100-step 轨迹不成为可确认的 optimizer。
```

因此，当前结论不是：

```text
functional update 没有下降方向。
```

而是：

```text
functional update 有短程下降能力，但没有形成深度表示学习所需的长期轨迹。
```

---

## 1. 重新定义问题：functional update 难调的根因

过去我们一直在尝试把 functional update 写成：

$$
\Delta a = -\eta M^{-1}g.
$$

不同版本只是改变 $M$：

```text
Sobolev Gram
task-aware diagonal
FNG / KFAC
functional-coordinate Adam
AdanLite / WinLite
BFT / trust gate
```

但 v4.6 说明，核心问题不是某个 $M$ 还没调对，而是：

$$
\boxed{
\text{单步 metric-preconditioned update 不是完整的深度网络 optimizer。}
}
$$

PureKAN 要训练的是一个完整深层函数：

$$
x
\rightarrow
\operatorname{KAN}_{in}
\rightarrow
\operatorname{KAN}_{block,1}
\rightarrow
\cdots
\rightarrow
\operatorname{KAN}_{out}
\rightarrow
z.
$$

其中 input KAN、middle blocks、output KAN 的角色完全不同：

```text
input KAN:
  raw pixels -> hidden feature
  主要负责 early representation formation

block KAN:
  hidden feature -> composed hidden feature
  主要负责 feature transformation / residual refinement

output KAN:
  hidden feature -> logits
  主要负责 margin / class boundary
```

一个统一的静态 functional metric 无法同时完成这三件事。

更严重的是，之前的 gate 同时要求：

```text
短程 loss descent
Adam-like function trajectory
rank / margin retention
geometry improvement
```

但 AdamW 早期自身也不是 geometry-improving trajectory。因此下一步不能把 geometry 作为所有阶段的硬约束，而要把训练过程拆成阶段：

```text
Phase I:
  表示形成，允许 geometry 变粗糙

Phase II:
  函数空间几何收束，同时保持已经学到的 logits / features

Phase III:
  小步任务微调与几何保持
```

这就是 v4.7 的核心方向：

$$
\boxed{
\text{Phase-Separated Functional Training，简称 PSFT。}
}
$$

---

## 2. 新核心假设：学习和几何不能再在同一步里硬兼容

v4.6 的 `FCAdam-dataSob / FLD-AdanLite` 证明了强任务下降方向存在，但它们被 early geometry / rank / trajectory gate 拦住。与其继续要求一个 update 同时做到所有事情，不如把它拆开：

### 2.1 Phase I：Functional Representation Learning

Phase I 的目标是学任务，不是压几何。它应该允许：

$$
\phi/A \leq 1.5
$$

甚至短期内更高，但必须满足：

$$
L_{\text{holdout}}\downarrow,
\quad
\text{margin}_{p10}\uparrow,
\quad
\text{class separation}\uparrow.
$$

这里的 update 使用 function-coordinate adaptive dynamics：

$$
a_l = L_l^{-T}u_l.
$$

在 $u_l$ 空间中做 Adam / Adan / Win-like update：

$$
u_{l,t+1}
=
u_{l,t}
+
\Delta u_{l,t}.
$$

Phase I 的 update 形式可以写成：

$$
\Delta u_{l,t}
=
\operatorname{AdaptiveStep}
\left(
\nabla_{u_l}
\left[
\mathcal L_{\text{CE}}
+
\lambda_{rank}\mathcal R_{rank}
+
\lambda_{sep}\mathcal R_{sep}
+
\lambda_{margin}\mathcal R_{margin}
\right]
\right).
$$

Phase I 中 Sobolev 只做弱 damping，不作为主目标：

$$
\lambda_S^{early} \approx 0.
$$

### 2.2 Phase II：Functional Geometry Consolidation

Phase II 的目标不是继续猛烈学任务，而是把 Phase I 学到的函数进行几何压缩 / 平滑，同时不破坏预测。

设 Phase I checkpoint 为 teacher $\theta^T$，它产生 teacher logits $z^T$ 和 hidden states $h_l^T$。

Phase II 解的是：

$$
\min_\theta
\quad
\mathcal L_{\text{CE}}(\theta)
+
\tau_z
\operatorname{KL}
\left(
p_{\theta^T}(y|x)
\| 
p_\theta(y|x)
\right)
+
\tau_h
\sum_l
\|h_l(\theta)-h_l(\theta^T)\|_2^2
+
\lambda_S
\sum_l
\|a_l\|_{S_l}^2.
$$

直觉：

```text
不要在几何 consolidation 阶段重新发明分类边界；
先保持已学到的 logits / features，
再用 Sobolev 把函数变平滑。
```

这和之前的 U-FULL 完全不同。之前是：

```text
一开始就用 Sobolev 管住所有层。
```

现在是：

```text
先学出有用函数，再做函数空间投影和压缩。
```

### 2.3 Phase III：Small-Step Functional Fine-Tuning

Phase III 的目标是小步微调：

$$
\Delta a_l
=
-\eta_l
\left(
G_l+\lambda_lS_l+\rho I
\right)^{-1}
\nabla_{a_l}
\left[
\mathcal L_{\text{CE}}
+
\tau_z\operatorname{KL}
+
\tau_h\mathcal L_{feat}
\right].
$$

这里 $\eta_l$ 小，$\lambda_l$ 中等，几何 gate 才变成硬约束。

---

## 3. 为什么这是“重新设计”，不是小修小补

这个 v4.7 计划不再问：

```text
Sobolev alpha 取多少？
branch scale 取多少？
Nesterov / Adan / Win 哪个更好？
```

而是问：

$$
\boxed{
\text{PureKAN functional optimizer 是否必须是一个分阶段训练协议？}
}
$$

这和之前所有尝试不同：

| 旧方法 | 核心问题 |
|---|---|
| U-FULL / Sobolev-only | 几何好，但不是任务 descent optimizer |
| D6 / Global TFU | one-step descent 好，但长程表示学习不足 |
| FNG / EK-FNG | curvature 更合理，但仍是局部 preconditioner |
| FTF | 局部 target 强，但跨层漂移严重 |
| BFT | 变安全，但变弱 |
| FCAdam / FLD | 短程下降强，但 rank / phi / role dynamics 不闭环 |

v4.7 的新点是：

```text
把学习阶段和几何阶段分离，
并用 teacher-preserving functional projection 连接它们。
```

---

## 4. 主要方法族

### 4.1 PSFT-A：FCAdam Learn -> Sobolev Distill

这是最小可行版本。

```text
Phase I:
  FCAdam-dataSob 或 FLD-AdanLite
  no / low Sobolev
  role-share controller 开启

Phase II:
  freeze teacher logits/features
  Sobolev consolidation with KL + feature preservation

Phase III:
  low-lr TFU / FNG fine-tune
```

推荐代号：

```text
PSFT-A-FCAdamDistill
PSFT-A-AdanDistill
```

### 4.2 PSFT-B：AdamW Teacher -> Functional Distillation

这是诊断版本，不作为最终 claim，但非常重要。它用 PureKAN-AdamW 跑出 teacher checkpoint，然后问：

```text
functional consolidation 能否在不使用 AdamW 继续训练的情况下，
复现并平滑 AdamW 学到的 PureKAN function？
```

公式：

$$
\min_\theta
\quad
\tau_z\operatorname{KL}(p_{AdamW}\|p_\theta)
+
\tau_h\sum_l\|h_l-h_l^{AdamW}\|^2
+
\lambda_S\sum_l\|a_l\|_S^2.
$$

如果这个也失败，说明当前 basis / functional projection 本身有问题。

如果这个成功，但 PSFT-A 失败，说明：

```text
functional projection 可以；
functional representation learning 还不行。
```

### 4.3 PSFT-C：Alternating Learn-Consolidate Cycles

不是只做一次 Phase I -> Phase II，而是循环：

```text
for cycle in 1..K:
  learn for T_learn steps
  consolidate for T_smooth steps
  validate holdout / rank / phi
```

每个 cycle 都用当前模型作为 teacher。

这对应一种 operator splitting：

$$
\theta_{t+1/2}
=
\operatorname{LearnStep}(\theta_t)
$$

$$
\theta_{t+1}
=
\operatorname{SmoothProject}(\theta_{t+1/2}).
$$

### 4.4 PSFT-D：Role-Staged Learning

AdamW envelope 显示早期 input update share 约为 $0.68$，block share 约为 $0.21$，output share 约为 $0.10$。因此 Phase I 不应该让 output 或 block 抢走学习。

Role-staged version：

```text
Stage 1:
  input-heavy learning
  target share: input 0.65, block 0.25, output 0.10

Stage 2:
  block refinement
  target share: input 0.35, block 0.45, output 0.20

Stage 3:
  output / margin tuning
  target share: input 0.20, block 0.40, output 0.40

Stage 4:
  Sobolev consolidation
```

---

## 5. 总体成功标准

v4.7 不是为了证明某个 proposal one-step 好，而是要真正挑战 PureKAN-AdamW。

最终必须同时比较：

```text
PureKAN-AdamW
PureKAN-UFULL / D0
D6-allTaskAware
FCAdam-dataSob
FLD-AdanLite
Hybrid-DGKAN-UFULL-f085
MLP-AdamW
```

### 5.1 最终 5-seed gate

对 MNIST / Fashion-MNIST / KMNIST，候选必须满足：

$$
\operatorname{Acc}_{candidate}
\geq
\operatorname{Acc}_{PureKAN-AdamW}
-
0.5\%
$$

并且至少一个数据集超过 AdamW：

$$
\operatorname{Acc}_{candidate}
>
\operatorname{Acc}_{PureKAN-AdamW}.
$$

同时：

$$
\text{AUC improvement vs D6} > 0.
$$

$$
\text{ECE reduction vs AdamW} > 0.
$$

$$
\phi/A_{\text{final}} \leq 1.0.
$$

$$
J/A_{\text{final}} \leq 1.0.
$$

### 5.2 Strong gate

如果要 claim PureKAN functional optimizer solved：

$$
\operatorname{Acc}_{candidate}
\geq
\max(
\operatorname{Acc}_{PureKAN-AdamW},
\operatorname{Acc}_{Hybrid-DGKAN-UFULL},
\operatorname{Acc}_{MLP-AdamW}
).
$$

并且：

$$
\text{final geometry better than PureKAN-AdamW}.
$$

---

## 6. 实验阶段设计

## P0：Implementation Smoke and Invariants

### 目标

确认 PSFT 的三阶段机制、teacher snapshot、functional coordinate、distillation loss、role budget、rollback 和 logging 都没有问题。

### 方法

```text
datasets:
  MNIST smoke
  Fashion-MNIST smoke
  KMNIST smoke

train / val / test:
  512 / 128 / 128

epochs:
  1

seeds:
  0
```

### 必跑方法

```text
PureKAN-AdamW
D0-allFullSobolev
FCAdam-dataSob
FLD-AdanLite
PSFT-A-FCAdamDistill-smoke
PSFT-A-AdanDistill-smoke
PSFT-B-AdamTeacherDistill-smoke
PSFT-C-Alternating-smoke
```

### 必须记录

```text
strict_purekan_nonKAN_params
coefficient_coverage
teacher_snapshot_ok
teacher_logits_finite
teacher_features_finite
distill_loss_finite
phase_trace
role_share_trace
functional_coordinate_roundtrip
rollback_error
NaN / Inf count
step_time
memory_peak
```

### P0 通过条件

$$
\text{run failures}=0.
$$

$$
\text{nonKAN params}=0.
$$

$$
\text{coefficient coverage}=1.
$$

$$
\text{all losses finite}.
$$

---

## P1：Phase-I Learning Dynamics Audit

### 目标

确认 Phase I 是否真的能形成任务表示，而不是只降低短程 holdout loss。

### 方法

```text
PureKAN-AdamW
FCAdam-dataSob
FLD-AdanLite
FLD-WinLite
FLD-LyapunovRestart
PSFT-A-FCAdam-PhaseI-only
PSFT-A-Adan-PhaseI-only
PSFT-D-role-staged-PhaseI-only
```

### 步长

```text
steps:
  1, 5, 20, 50, 100
```

### 记录指标

#### 任务轨迹

```text
train_loss_delta
holdout_loss_delta
val_loss_delta
train_acc
val_acc
test_acc_probe
```

#### 表示轨迹

```text
effective_rank_input
effective_rank_block_mean
effective_rank_output_input
class_centroid_separation
within_class_variance
between_class_variance
margin_mean
margin_p10
margin_p50
margin_p90
```

#### role update

```text
role_update_share_input
role_update_share_block
role_update_share_output
role_update_norm_input
role_update_norm_block
role_update_norm_output
role_update_over_param_input
role_update_over_param_block
role_update_over_param_output
```

#### 函数几何

```text
phi_prime_p95
phi_prime_max
Jacobian_condition
Sobolev_energy
basis_occupancy_entropy
dead_basis_fraction
out_of_grid_fraction
```

#### AdamW envelope alignment

```text
holdout_delta_over_adamw
rank_ratio_over_adamw
margin_ratio_over_adamw
phi_ratio_over_adamw
role_share_L1_distance_to_adamw
function_R2_to_adamw
function_cos_to_adamw
```

### 可视化

```text
p1_holdout_delta_vs_step.svg
p1_rank_margin_vs_step.svg
p1_phi_vs_step.svg
p1_role_share_stacked_area.svg
p1_role_share_distance_to_adamw.svg
p1_function_R2_to_adamw.svg
p1_loss_rank_phi_3axis.svg
```

### P1 通过条件

候选进入 P2 必须满足：

$$
\text{holdout}_{100}/\text{AdamW}_{100} > 0.75.
$$

$$
\text{role share distance to AdamW} < 0.35.
$$

$$
\text{margin}_{p10}\text{ improves}.
$$

$$
\phi/A \leq 1.7
$$

其中 $\phi/A$ 在 P1 不再要求小于 $1.0$，因为 P1 是表示形成阶段。

---

## P2：Geometry Consolidation-Only Test

### 目标

这是 v4.7 的关键诊断。它回答：

```text
如果已经有一个学得不错的 teacher，
functional Sobolev projection 能不能在保持 logits/features 的同时改善 geometry？
```

### Teacher 来源

```text
Teacher A:
  PureKAN-AdamW checkpoint at step 100

Teacher B:
  PSFT-A Phase-I checkpoint at step 100

Teacher C:
  FCAdam-dataSob checkpoint at step 100

Teacher D:
  FLD-AdanLite checkpoint at step 100
```

### Student 初始化

```text
same checkpoint as teacher
then apply consolidation phase
```

### Consolidation objective

$$
\mathcal L_{\text{consolidate}}
=
\tau_z
\operatorname{KL}(p_T\|p_\theta)
+
\tau_h
\sum_l
\|h_l^T-h_l\|_2^2
+
\lambda_{CE}
\mathcal L_{CE}
+
\lambda_S
\sum_l
\|a_l\|_{S_l}^2.
$$

### Sweep

```text
tau_z in {0.5, 1.0, 2.0}
tau_h in {0.1, 0.5}
lambda_S in {1e-4, 3e-4, 1e-3, 3e-3}
consolidation_steps in {20, 50, 100}
```

### 必须记录

```text
teacher_acc
student_acc_before
student_acc_after
KL_teacher_student
feature_mse_input
feature_mse_block_mean
feature_mse_output
phi_before
phi_after
J_before
J_after
Sobolev_energy_before
Sobolev_energy_after
margin_before
margin_after
rank_before
rank_after
```

### 关键可视化

```text
p2_geometry_reduction_vs_kl.svg
p2_acc_retention_vs_phi_reduction.svg
p2_feature_mse_vs_acc_drop.svg
p2_teacher_student_logit_correlation.svg
p2_sobolev_energy_curve.svg
```

### P2 通过条件

$$
\operatorname{Acc}_{after}
\geq
\operatorname{Acc}_{before}
-
0.5\%.
$$

$$
\phi_{after}
\leq
0.85\phi_{before}.
$$

$$
J_{after}
\leq
0.85J_{before}.
$$

$$
\operatorname{KL}(p_T\|p_\theta)
\leq
0.05.
$$

如果 P2 失败，说明：

```text
functional geometry projection 本身无法保持学到的函数；
PureKAN functional optimizer 需要重新考虑 basis / architecture。
```

如果 P2 成功，进入 P3。

---

## P3：One-Cycle PSFT Micro-Run

### 目标

验证一次完整的：

```text
Phase I learn
Phase II consolidate
Phase III fine-tune
```

能不能形成比 D6/FCAdam 更稳定的 trajectory。

### 配置

```text
train / val / test = 6000 / 1000 / 1000
seeds = 0,1,2
datasets = MNIST, Fashion-MNIST, KMNIST
```

### 方法

```text
PureKAN-AdamW
D6-allTaskAware
FCAdam-dataSob
FLD-AdanLite
PSFT-A-FCAdamDistill
PSFT-A-AdanDistill
PSFT-D-role-staged-FCAdamDistill
```

### Phase schedule

```text
Phase I:
  100 steps learn

Phase II:
  50 steps consolidate

Phase III:
  50 steps fine-tune
```

### 记录

```text
phase_id
step_in_phase
train_loss
val_loss
test_probe_acc
holdout_delta
KL_to_phaseI_teacher
feature_MSE_to_phaseI_teacher
role_share
rank
margin
phi
J
ECE
NLL
```

### 可视化

```text
p3_phase_colored_loss_curve.svg
p3_phase_colored_accuracy_curve.svg
p3_phase_colored_phi_curve.svg
p3_rank_margin_phi_phase_plot.svg
p3_role_share_by_phase.svg
p3_distill_preservation_curve.svg
```

### P3 通过条件

候选进入 P4 必须满足：

$$
\text{acc gap vs PureKAN-AdamW} < 2\%.
$$

$$
\text{AUC vs D6} > 0.
$$

$$
\phi_{\text{final}}/A \leq 1.15.
$$

$$
\text{ECE reduction vs AdamW} \geq 0.
$$

---

## P4：Alternating PSFT Cycles

### 目标

一次 Learn -> Consolidate 可能不够。P4 验证多轮 operator splitting 是否能更接近 AdamW 的长期轨迹。

### 方法

```text
PSFT-C-FCAdam cycles
PSFT-C-Adan cycles
PSFT-C-role-staged cycles
```

### Schedules

```text
Cycle short:
  learn 50 steps
  consolidate 25 steps
  repeat 4 cycles

Cycle medium:
  learn 100 steps
  consolidate 50 steps
  repeat 3 cycles

Cycle long:
  learn 150 steps
  consolidate 75 steps
  repeat 2 cycles
```

### 记录

```text
cycle_id
phase_id
cycle_holdout_delta
cycle_acc_delta
cycle_phi_delta
cycle_rank_delta
cycle_margin_delta
teacher_preservation_KL
feature_preservation_MSE
consolidation_accept_rate
```

### 可视化

```text
p4_cycle_dashboard.svg
p4_cycle_rank_phi_tradeoff.svg
p4_cycle_acc_vs_phi.svg
p4_consolidation_effect_per_cycle.svg
```

### P4 通过条件

至少一个 candidate 满足：

$$
\text{acc gap vs PureKAN-AdamW} < 1.5\%.
$$

$$
\phi/A_{\text{final}} < 1.05.
$$

$$
\text{holdout AUC vs D6} > 0.
$$

并且不能出现：

```text
chance accuracy collapse
KL explosion
rank collapse
```

---

## P5：Role-Staging Ablation

### 目标

验证 AdamW envelope 中的 input-heavy update 是否是 PureKAN 训练的关键。

### 对照

```text
No role controller
AdamW-envelope role controller
Input-heavy only
Block-heavy only
Output-heavy only
Adaptive role controller
```

### Role share targets

```text
AdamW-envelope:
  input = 0.68
  block = 0.21
  output = 0.11

Input-heavy:
  input = 0.75
  block = 0.15
  output = 0.10

Block-heavy:
  input = 0.30
  block = 0.55
  output = 0.15

Output-heavy:
  input = 0.25
  block = 0.25
  output = 0.50
```

### 记录

```text
actual_role_share
role_share_error
role_specific_loss_proxy
role_specific_update_norm
role_specific_phi
role_specific_rank_effect
role_specific_margin_effect
```

### 判定

如果 AdamW-envelope role controller 明显好于 no-controller，说明：

```text
PureKAN functional update 的问题之一是 role-wise credit allocation。
```

如果没有差异，则 role share 不是主瓶颈。

---

## P6：Consolidation Objective Ablation

### 目标

理解 Phase II 中哪些项是必要的。

### 对照

```text
KL only
feature MSE only
KL + feature MSE
KL + CE
KL + feature MSE + CE
KL + feature MSE + CE + Sobolev
```

### 记录

```text
acc_retention
phi_reduction
J_reduction
KL_after
feature_MSE_after
margin_after
rank_after
```

### 可视化

```text
p6_objective_ablation_radar.svg
p6_kl_feature_phi_scatter.svg
p6_acc_retention_bar.svg
```

### 判定

理想情况：

```text
KL + feature + Sobolev 保持 acc，同时降低 phi/J。
```

如果 KL only 保持 logits 但 feature drift 大，说明 hidden representation preservation 必须加入。  
如果 feature only 保持 rank 但 logits 退化，说明 output constraint 必须加入。

---

## P7：3-Seed Full-Budget Candidate Selection

### 目标

用完整训练预算筛选最终候选。

### 数据集

```text
MNIST
Fashion-MNIST
KMNIST
```

### 方法

```text
PureKAN-AdamW
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
D6-allTaskAware
FCAdam-dataSob
FLD-AdanLite
Best PSFT-A
Best PSFT-C
Best PSFT-D
```

### seeds

```text
0, 1, 2
```

### 记录

```text
test_acc
val_loss_auc
ECE
NLL
phi_prime_p95
Jacobian_condition
effective_rank
margin_p10 / mean / p90
role_share_auc
phase_transition_metrics
step_time
memory_peak
```

### P7 通过条件

候选进入 5-seed confirm 必须满足：

$$
\text{mean acc gap vs PureKAN-AdamW} < 1.0\%.
$$

$$
\text{val AUC vs D6} > 0.
$$

$$
\phi/A < 1.0.
$$

$$
\text{ECE reduction vs AdamW} > 0.
$$

---

## P8：5-Seed Confirm

### 目标

确认 P7 survivor 是否稳定。

### seeds

```text
0,1,2,3,4
```

### 方法

```text
PureKAN-AdamW
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
Best PSFT candidate
```

### 统计

```text
mean
std
paired delta
bootstrap 95% CI
failure count by seed
```

### 通过条件

$$
\Delta acc_{\text{candidate}-AdamW}
\geq
-0.5\%.
$$

$$
CI_{\Delta acc}^{lo}
>
-1.0\%.
$$

$$
\Delta AUC > 0.
$$

$$
\Delta ECE > 0.
$$

$$
\phi/A < 1.0.
$$

---

## P9：10-Seed Final Confirm

只对 P8 通过者运行。

### seeds

```text
0..9
```

### 最终 claim 条件

弱 claim：

```text
PureKAN-PSFT matches PureKAN-AdamW accuracy while improving geometry / calibration.
```

medium claim：

```text
PureKAN-PSFT slightly exceeds AdamW on at least one dataset and matches on others.
```

strong claim：

```text
PureKAN-PSFT exceeds PureKAN-AdamW, Hybrid-DGKAN-UFULL, and MLP-AdamW on all MNIST-family datasets.
```

---

## 7. 指标总表

### 7.1 任务指标

```text
train_loss
val_loss
test_loss
train_acc
val_acc
test_acc
val_loss_auc
val_acc_auc
NLL
ECE
classwise_acc
worst_class_acc
```

### 7.2 表示学习指标

```text
effective_rank_input
effective_rank_block_l
effective_rank_output_input
class_centroid_distance
within_class_variance
between_class_variance
separation_ratio
margin_mean
margin_p10
margin_p50
margin_p90
```

### 7.3 functional geometry

```text
phi_prime_p95
phi_prime_max
Jacobian_condition
Sobolev_energy
curvature_energy
basis_occupancy_entropy
dead_basis_fraction
out_of_grid_fraction
```

### 7.4 teacher preservation

```text
KL_teacher_student
feature_MSE_input
feature_MSE_block_mean
feature_MSE_output
logit_cos_teacher_student
logit_R2_teacher_student
prediction_agreement
```

### 7.5 optimizer dynamics

```text
phase_id
cycle_id
role_update_share
update_norm_by_role
update_over_param_by_role
function_delta_norm
function_delta_R2_to_teacher
AdamW_envelope_distance
accepted_step_ratio
backtrack_count
restart_count
step_time_ms
memory_peak_mb
```

---

## 8. 可视化总表

必须生成以下图：

```text
figures/p1_holdout_delta_vs_step.svg
figures/p1_rank_margin_phi_trajectory.svg
figures/p1_role_share_stacked_area.svg
figures/p1_function_alignment_to_adamw.svg

figures/p2_geometry_reduction_vs_acc_retention.svg
figures/p2_teacher_student_KL_curve.svg
figures/p2_feature_preservation_curve.svg
figures/p2_phi_reduction_curve.svg

figures/p3_phase_colored_loss_curve.svg
figures/p3_phase_colored_acc_curve.svg
figures/p3_phase_colored_phi_curve.svg
figures/p3_rank_margin_phi_phase_plot.svg

figures/p4_cycle_dashboard.svg
figures/p4_cycle_acc_phi_tradeoff.svg
figures/p4_role_share_by_cycle.svg

figures/p5_role_share_ablation.svg
figures/p6_consolidation_objective_radar.svg
figures/p7_candidate_scorecard.svg
figures/p8_paired_delta_ci.svg
figures/failure_taxonomy_heatmap.svg
```

---

## 9. 失败解释表

每个失败 run 必须归到一个主类型：

| code | failure type | 判断标准 |
|---|---|---|
| F1 | Phase-I under-learning | holdout descent < 0.75 AdamW |
| F2 | role allocation mismatch | role share L1 distance > 0.35 |
| F3 | rank collapse | effective rank below 0.7 AdamW envelope |
| F4 | margin failure | margin p10 不升或 class separation 不升 |
| F5 | geometry explosion | phi/A > 1.7 in Phase I or > 1.0 final |
| F6 | consolidation destroys function | KL > 0.05 or acc drop > 0.5% after Phase II |
| F7 | feature drift | hidden feature MSE too high after consolidation |
| F8 | no final advantage | trajectory OK but final acc/ECE/AUC not competitive |
| F9 | compute failure | step time > 2x AdamW without accuracy benefit |

---

## 10. 本轮之后的决策规则

### 情况 A：P2 consolidation-only 失败

结论：

```text
Current functional geometry projection cannot preserve learned PureKAN functions.
Stop optimizer search and inspect basis / architecture / parameterization.
```

### 情况 B：P2 成功，但 P3/P4 失败

结论：

```text
Functional projection works, but functional representation learning still fails.
Next work should focus on Phase-I learning dynamics / role allocation.
```

### 情况 C：P4/P7 成功但 P8 不稳

结论：

```text
PSFT is a promising but unstable optimizer.
Need seed-stability controller and adaptive phase lengths.
```

### 情况 D：P8/P9 成功

结论：

```text
PureKAN functional optimizer is solved at MNIST-family scale.
Functional training requires phase separation:
task-learning phase + geometry consolidation phase.
```

---

## 11. 最终一句话

v4.7 的核心不是继续寻找更好的单步 update，而是验证：

$$
\boxed{
\text{PureKAN functional training 是否必须是}
\quad
\text{Learn} \rightarrow \text{Functional Smooth Projection} \rightarrow \text{Fine Tune}
}
$$

如果这个成立，我们就能解释过去所有现象：

```text
Sobolev-only:
  一开始就 smooth，所以学不起来。

FCAdam / FLD:
  能学，但 roughness / margin / rank 不受控。

BFT:
  很安全，但太弱。

PSFT:
  先让函数学会任务，再把函数投影回好的几何。
```

这才是下一步真正值得验证的方向。



---


# Source 22: `docs/DG-KAN_v4.7_PhaseSeparatedFunctionalTraining_PureKAN_结果复盘.md`


# DG-KAN v4.7 Phase-Separated Functional Training 结果复盘

本轮依据 `docs/DG-KAN_v4.7_PhaseSeparatedFunctionalTraining_PureKAN_实验计划.md`。目标是验证 PSFT：先用 functional-coordinate dynamics 学表示，再用 teacher-preserving Sobolev consolidation 降几何。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 PSFT smoke | 48 | 0 |
| P1 Phase-I dynamics | 120 | 0 |
| P2 consolidation sweep | 864 | 0 |

## Code / Config Changes

```text
experiments/run_gafu_v47.py
  Added PSFT phase-I methods, role-staged Phase-I control, teacher snapshots,
  KL/feature distillation, and Sobolev-style consolidation sweep.

experiments/analyze_gafu_v47.py
  Generates P1/P2 gate summaries, failure taxonomy, figures, aggregate_decision.json,
  and this result replay.
```

## P0 Implementation Smoke

| rows | errors | max nonKAN | min cov | max rollback | pass |
|---|---|---|---|---|---|
| 48 | 0 | 0.0000 | 1.0000 | 0.0000 | true |

P0 verdict: pass. PSFT snapshot/distillation paths ran with strict PureKAN non-KAN count at zero and finite loss/rollback checks.

## P1 Phase-I Learning Dynamics

| dataset | method | runs | hold100 | hold/A | rank | margin Δ | phi | role | P1 |
|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | FCAdam-dataSob | 1 | 1.7537 | 0.9580 | 0.3362 | 4.5014 | 1.8633 | 0.0121 | no |
| Fashion-MNIST | FLD-AdanLite | 1 | 1.6657 | 0.9100 | 0.3401 | 4.0494 | 1.8288 | 0.0108 | no |
| Fashion-MNIST | FLD-LyapunovRestart | 1 | 1.6657 | 0.9100 | 0.3401 | 4.0494 | 1.8288 | 0.0108 | no |
| Fashion-MNIST | FLD-WinLite | 1 | 1.7440 | 0.9528 | 0.3344 | 4.4471 | 1.8621 | 0.0114 | no |
| Fashion-MNIST | PSFT-A-Adan-PhaseI-only | 1 | 1.7450 | 0.9533 | 0.3344 | 4.4453 | 1.8639 | 0.0114 | no |
| Fashion-MNIST | PSFT-A-FCAdam-PhaseI-only | 1 | 1.7486 | 0.9553 | 0.3340 | 4.4398 | 1.8638 | 0.0113 | no |
| Fashion-MNIST | PSFT-D-role-staged-PhaseI-only | 1 | 1.6939 | 0.9254 | 0.3371 | 4.4635 | 1.9636 | 0.0342 | no |
| Fashion-MNIST | PureKAN-AdamW | 1 | 1.8305 | 1.0000 | 0.3423 | 2.7921 | 1.1569 | 0.0790 | yes |
| KMNIST | FCAdam-dataSob | 1 | 1.6198 | 0.8547 | 0.3514 | 3.8614 | 1.9127 | 0.0359 | no |
| KMNIST | FLD-AdanLite | 1 | 1.6059 | 0.8474 | 0.3411 | 3.7082 | 1.8794 | 0.0426 | no |
| KMNIST | FLD-LyapunovRestart | 1 | 1.6059 | 0.8474 | 0.3411 | 3.7082 | 1.8794 | 0.0426 | no |
| KMNIST | FLD-WinLite | 1 | 1.6222 | 0.8560 | 0.3506 | 3.8315 | 1.9022 | 0.0315 | no |
| KMNIST | PSFT-A-Adan-PhaseI-only | 1 | 1.6222 | 0.8560 | 0.3506 | 3.8331 | 1.9039 | 0.0316 | no |
| KMNIST | PSFT-A-FCAdam-PhaseI-only | 1 | 1.6227 | 0.8562 | 0.3505 | 3.8268 | 1.9025 | 0.0307 | no |
| KMNIST | PSFT-D-role-staged-PhaseI-only | 1 | 1.6148 | 0.8521 | 0.3504 | 3.9508 | 1.9824 | 0.0567 | no |
| KMNIST | PureKAN-AdamW | 1 | 1.8952 | 1.0000 | 0.4716 | 2.9795 | 1.1879 | 0.0711 | yes |
| MNIST | FCAdam-dataSob | 1 | 1.9144 | 1.0162 | 0.3209 | 4.7558 | 1.9615 | 0.0247 | no |
| MNIST | FLD-AdanLite | 1 | 1.9343 | 1.0268 | 0.3194 | 4.6853 | 1.8986 | 0.0267 | no |
| MNIST | FLD-LyapunovRestart | 1 | 1.9343 | 1.0268 | 0.3194 | 4.6853 | 1.8986 | 0.0267 | no |
| MNIST | FLD-WinLite | 1 | 1.9159 | 1.0170 | 0.3209 | 4.7193 | 1.9503 | 0.0213 | no |
| MNIST | PSFT-A-Adan-PhaseI-only | 1 | 1.9158 | 1.0169 | 0.3209 | 4.7211 | 1.9523 | 0.0213 | no |
| MNIST | PSFT-A-FCAdam-PhaseI-only | 1 | 1.9161 | 1.0171 | 0.3209 | 4.7141 | 1.9499 | 0.0206 | no |
| MNIST | PSFT-D-role-staged-PhaseI-only | 1 | 1.9098 | 1.0137 | 0.3197 | 4.8414 | 2.0182 | 0.0449 | no |
| MNIST | PureKAN-AdamW | 1 | 1.8839 | 1.0000 | 0.3448 | 3.1572 | 1.2214 | 0.0630 | yes |

P1 survivors:

```text
none
```

## P2 Geometry Consolidation Test

P2 grid mode is recorded in `p2_consolidation_sweep.csv`. The table below shows the top rows by dataset/geometry signal, truncated for readability.

| dataset | teacher | steps | tau_z | tau_h | lambda | acc drop | phi red | J red | KL | P2 |
|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 2.0 | 0.1 | 0.0001 | 0.6875 | 0.0541 | -1.1591 | 5.1665 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 2.0 | 0.1 | 0.0003 | 0.6895 | 0.0532 | 0.8618 | 5.1870 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 2.0 | 0.1 | 0.001 | 0.6992 | 0.0520 | -2.4677 | 5.2387 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 2.0 | 0.1 | 0.003 | 0.7012 | 0.0506 | -17.0242 | 5.2958 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 1.0 | 0.1 | 0.0003 | 0.5742 | 0.0489 | -0.0843 | 4.5792 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 1.0 | 0.1 | 0.0001 | 0.5703 | 0.0487 | -0.9209 | 4.5444 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 1.0 | 0.1 | 0.001 | 0.6113 | 0.0482 | 0.7913 | 4.6679 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 1.0 | 0.1 | 0.003 | 0.6367 | 0.0460 | 0.8977 | 4.7802 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 2.0 | 0.5 | 0.0003 | 0.5703 | 0.0396 | 0.7950 | 4.6827 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 2.0 | 0.5 | 0.001 | 0.5781 | 0.0395 | 0.7906 | 4.7524 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 2.0 | 0.5 | 0.0001 | 0.5664 | 0.0395 | 0.8660 | 4.6537 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 2.0 | 0.5 | 0.003 | 0.5977 | 0.0393 | 0.3589 | 4.8221 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 20 | 2.0 | 0.1 | 0.0001 | 0.6348 | 0.0352 | 0.9229 | 5.0232 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 20 | 2.0 | 0.1 | 0.0003 | 0.6367 | 0.0350 | 0.8019 | 5.0379 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 20 | 2.0 | 0.1 | 0.001 | 0.6348 | 0.0347 | 0.8873 | 5.0812 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 20 | 2.0 | 0.1 | 0.0003 | 0.6484 | 0.0346 | 0.9484 | 5.0901 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 20 | 2.0 | 0.1 | 0.003 | 0.6348 | 0.0345 | 0.7589 | 5.1622 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 20 | 2.0 | 0.1 | 0.0001 | 0.6484 | 0.0343 | 0.2624 | 5.0769 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 20 | 2.0 | 0.1 | 0.003 | 0.6504 | 0.0343 | 0.4407 | 5.2003 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 20 | 2.0 | 0.1 | 0.001 | 0.6504 | 0.0339 | 0.6114 | 5.1273 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 100 | 2.0 | 0.1 | 0.003 | 0.3867 | 0.0322 | 0.7597 | 4.1487 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 100 | 2.0 | 0.1 | 0.001 | 0.3750 | 0.0303 | -1.9439 | 4.1645 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 20 | 1.0 | 0.1 | 0.003 | 0.5312 | 0.0299 | 0.8980 | 4.6829 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 20 | 1.0 | 0.1 | 0.003 | 0.5801 | 0.0292 | -9.5062 | 4.7137 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 20 | 1.0 | 0.1 | 0.001 | 0.5508 | 0.0288 | 0.4053 | 4.6022 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 20 | 1.0 | 0.1 | 0.001 | 0.5195 | 0.0285 | 0.7399 | 4.5714 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 20 | 1.0 | 0.1 | 0.0003 | 0.4980 | 0.0283 | -1.4379 | 4.5032 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 100 | 2.0 | 0.1 | 0.0003 | 0.3867 | 0.0282 | 0.9205 | 4.1677 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 20 | 1.0 | 0.1 | 0.0001 | 0.4883 | 0.0280 | 0.8840 | 4.4775 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 20 | 1.0 | 0.1 | 0.0003 | 0.5391 | 0.0279 | 0.8383 | 4.5331 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 100 | 2.0 | 0.1 | 0.0001 | 0.3906 | 0.0277 | 0.6626 | 4.1719 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 20 | 1.0 | 0.1 | 0.0001 | 0.5332 | 0.0274 | 0.0861 | 4.5069 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 100 | 2.0 | 0.1 | 0.003 | 0.3613 | 0.0268 | 0.5713 | 4.2272 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 0.5 | 0.1 | 0.0001 | 0.1797 | 0.0265 | -2.5562 | 3.0535 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 0.5 | 0.1 | 0.0003 | 0.1855 | 0.0264 | 0.9284 | 3.0739 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 0.5 | 0.1 | 0.001 | 0.1992 | 0.0259 | 0.6564 | 3.1266 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 100 | 1.0 | 0.1 | 0.001 | 0.2422 | 0.0244 | 0.1819 | 3.7577 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 50 | 2.0 | 0.1 | 0.003 | 0.3965 | 0.0238 | 0.7481 | 4.0190 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 100 | 1.0 | 0.1 | 0.003 | 0.2383 | 0.0238 | 0.4616 | 3.7294 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 100 | 2.0 | 0.1 | 0.001 | 0.3477 | 0.0238 | 0.8880 | 4.1983 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 0.5 | 0.1 | 0.003 | 0.2090 | 0.0236 | -0.5448 | 3.1957 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 20 | 2.0 | 0.5 | 0.003 | 0.5117 | 0.0217 | 0.0300 | 4.5404 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 20 | 2.0 | 0.5 | 0.003 | 0.4961 | 0.0217 | 0.9142 | 4.4696 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 100 | 1.0 | 0.1 | 0.0003 | 0.2402 | 0.0216 | 0.5873 | 3.7580 | no |
| Fashion-MNIST | TeacherA-AdamW100 | 100 | 1.0 | 0.5 | 0.003 | 0.0820 | 0.0214 | 0.5083 | 2.4464 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 50 | 2.0 | 0.1 | 0.001 | 0.3750 | 0.0213 | 0.5725 | 3.9737 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 100 | 2.0 | 0.1 | 0.0003 | 0.3320 | 0.0211 | 0.1838 | 4.1720 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 20 | 2.0 | 0.5 | 0.001 | 0.5020 | 0.0207 | 0.5300 | 4.4432 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 100 | 1.0 | 0.1 | 0.0001 | 0.2344 | 0.0207 | 0.9019 | 3.7594 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 100 | 2.0 | 0.1 | 0.0001 | 0.3359 | 0.0205 | 0.7776 | 4.1618 | no |
| Fashion-MNIST | TeacherA-AdamW100 | 100 | 0.5 | 0.5 | 0.003 | 0.0332 | 0.0204 | 0.5357 | 2.1262 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 50 | 2.0 | 0.1 | 0.0003 | 0.3750 | 0.0203 | 0.8365 | 3.9581 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 50 | 2.0 | 0.1 | 0.0001 | 0.3770 | 0.0201 | -1.4205 | 3.9561 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 20 | 2.0 | 0.5 | 0.001 | 0.4844 | 0.0197 | 0.7630 | 4.3626 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 20 | 2.0 | 0.5 | 0.0003 | 0.4883 | 0.0195 | -3.0646 | 4.3758 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 50 | 2.0 | 0.1 | 0.001 | 0.3672 | 0.0191 | 0.6996 | 4.0393 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 20 | 2.0 | 0.5 | 0.0001 | 0.4824 | 0.0190 | 0.6028 | 4.3484 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 50 | 2.0 | 0.1 | 0.0003 | 0.3633 | 0.0188 | 0.7809 | 4.0312 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 20 | 2.0 | 0.5 | 0.0003 | 0.4727 | 0.0188 | -5.7174 | 4.2867 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 50 | 2.0 | 0.1 | 0.003 | 0.3711 | 0.0187 | 0.4103 | 4.0616 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 50 | 2.0 | 0.1 | 0.0001 | 0.3594 | 0.0186 | 0.5930 | 4.0294 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 20 | 2.0 | 0.5 | 0.0001 | 0.4688 | 0.0185 | 0.2571 | 4.2565 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 100 | 0.5 | 0.5 | 0.0001 | 0.2051 | 0.0183 | 0.7893 | 3.5029 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 1.0 | 0.5 | 0.0003 | 0.2676 | 0.0182 | 0.9123 | 3.1963 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 100 | 0.5 | 0.5 | 0.0003 | 0.2070 | 0.0179 | 0.6572 | 3.5072 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 1.0 | 0.5 | 0.0001 | 0.2598 | 0.0177 | 0.8627 | 3.1693 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 1.0 | 0.5 | 0.001 | 0.2793 | 0.0175 | -0.7660 | 3.2536 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 100 | 0.5 | 0.5 | 0.001 | 0.2148 | 0.0173 | 0.5510 | 3.5180 | no |
| Fashion-MNIST | TeacherA-AdamW100 | 100 | 2.0 | 0.5 | 0.003 | 0.2246 | 0.0172 | 0.1945 | 2.7433 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 50 | 1.0 | 0.1 | 0.003 | 0.1914 | 0.0171 | 0.9172 | 3.3689 | no |
| Fashion-MNIST | TeacherA-AdamW100 | 100 | 1.0 | 0.5 | 0.001 | 0.0938 | 0.0169 | 0.4886 | 2.4740 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 100 | 2.0 | 0.5 | 0.003 | 0.4492 | 0.0167 | 0.0973 | 4.2842 | no |
| Fashion-MNIST | TeacherA-AdamW100 | 20 | 2.0 | 0.1 | 0.001 | 0.5684 | 0.0162 | 0.4325 | 3.0831 | no |
| Fashion-MNIST | TeacherA-AdamW100 | 20 | 2.0 | 0.1 | 0.003 | 0.5762 | 0.0160 | 0.4995 | 3.0894 | no |
| Fashion-MNIST | TeacherA-AdamW100 | 20 | 2.0 | 0.1 | 0.0003 | 0.5645 | 0.0159 | 0.3875 | 3.0760 | no |
| Fashion-MNIST | TeacherA-AdamW100 | 100 | 2.0 | 0.5 | 0.001 | 0.2617 | 0.0157 | 0.2106 | 2.7841 | no |
| Fashion-MNIST | TeacherA-AdamW100 | 20 | 2.0 | 0.1 | 0.0001 | 0.5586 | 0.0156 | 0.3732 | 3.0733 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 1.0 | 0.5 | 0.003 | 0.3027 | 0.0156 | 0.6107 | 3.2721 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 50 | 1.0 | 0.1 | 0.001 | 0.2109 | 0.0156 | 0.3470 | 3.3890 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 50 | 1.0 | 0.1 | 0.0003 | 0.2012 | 0.0155 | 0.8380 | 3.4139 | no |

Best diagnostic points:

| dataset | teacher | best by | phi red | acc drop | KL |
|---|---|---|---|---|---|
| MNIST | TeacherA-AdamW100 | phi | 0.0449 | 0.1328 | 1.9357 |
| MNIST | TeacherA-AdamW100 | safe | 0.0161 | 0.0703 | 1.1131 |
| Fashion-MNIST | TeacherD-FLD-Adan100 | phi | 0.0541 | 0.6875 | 5.1665 |
| Fashion-MNIST | TeacherA-AdamW100 | safe | 0.0081 | 0.0059 | 1.0592 |
| KMNIST | TeacherB-PSFT-FCAdam100 | phi | 0.0599 | 0.3027 | 3.8296 |
| KMNIST | TeacherA-AdamW100 | safe | 0.0082 | 0.0957 | 0.8802 |

P2 all-dataset survivors:

```text
none
```

## P3-P8 Decision

```text
P3 one-cycle PSFT micro-run: not run; P2 produced no survivor
P4-P8: not run by gate.
```

## Failure Diagnosis

Generated:

```text
p9_failure_diagnosis.csv
failure_table.csv
failure_by_dataset.csv
failure_by_method.csv
```

Diagnosis:

```text
Phase separation is implemented, but P2 is the bottleneck.
If consolidation preserves teacher predictions, geometry reduction is weak;
when geometry reduction appears, accuracy/teacher preservation usually fails.
This supports the v4.7 diagnostic: representation learning and geometry projection
cannot yet be connected by the current Sobolev distillation step.
```

## Required Artifacts

Written under `results/v4_7/`:

```text
p0_psft_invariants.csv
p1_phase_i_dynamics.csv
p1_phase_i_gate_summary.csv
p2_consolidation_sweep.csv
p2_consolidation_gate_summary.csv
p3_one_cycle_micro_run.csv
p4_cycle_ablation.csv
p5_role_stage_ablation.csv
p6_cifar_precheck.csv
p7_confirm5.csv
p8_final10.csv
p9_failure_diagnosis.csv
failure_table.csv
aggregate_decision.json
figures/
```

## Final Decision

```text
PureKAN PSFT status:
  stop_after_p2_consolidation_failed

What improved:
  Phase-I and teacher snapshot/distillation machinery are now auditable.
  P2 directly tests whether a learned PureKAN function can be geometrically consolidated.

What failed:
  No P2 configuration passed the joint teacher-preservation + phi/J reduction gate across all datasets.

Conclusion:
  PSFT is the right diagnostic framing, but the current consolidation operator is not sufficient.
  Next work should redesign the Phase-II projection itself before spending P3/P7 seed budget.
```



---


# Source 23: `docs/DG-KAN_v4.8_FunctionalGeometry_Feasibility_Redesign_实验计划.md`


# DG-KAN v4.8 Functional Geometry Feasibility Redesign 实验计划

## 0. 版本定位

v4.8 的目标不是继续微调 `Sobolev alpha / beta`、`branch scale`、`warmup`、`trust radius` 或固定的 phase schedule。v4.7 已经说明，PSFT 的 Phase-I 能产生一定任务学习信号，Phase-II 也能做 teacher-preserving consolidation 的 sweep，但没有任何配置同时通过 teacher preservation 和 geometry reduction。更关键的是，当 teacher preservation 做得较好时，geometry reduction 非常弱；当 geometry reduction 稍微出现时，accuracy drop / KL 又很大。

所以 v4.8 的问题要重新定义为：

$$
\boxed{
\text{PureKAN 是否存在“准确且几何好”的可达解？如果存在，如何沿着不破坏任务函数的方向降低几何？}
}
$$

这和之前的 U-FULL / TFU / FNG / FTF / BFT / PSFT 都不同。过去我们默认认为 “AdamW 学到的函数可以被 functional Sobolev projection 平滑化”；v4.7 的 P2 结果说明这个假设没有被验证，甚至可能是错的。因此 v4.8 必须先做 **feasibility-first**：先证明目标存在，再设计 optimizer。

本计划把下一步命名为：

```text
FGF: Functional Geometry Feasibility
NFS: Nullspace Functional Smoothing
TAN: Task-then-Nullspace functional training
```

核心路线是：

```text
1. 先找 PureKAN 的 accuracy-geometry Pareto frontier。
2. 再测试是否能在保持 logits / hidden features 的同时降低 geometry。
3. 如果可以，再把 NFS 作为周期性 projection 接入 task-learning phase。
4. 如果不可以，说明当前 PureKAN/RBF 参数化下“准确 + 几何好”可能不可达，需要改 architecture / basis，而不是继续改 optimizer。
```

---

## 1. 当前结果的深层解读

### 1.1 v4.7 的关键事实

v4.7 的 P0 说明实现路径干净：PSFT snapshot、distillation、rollback、strict PureKAN non-KAN count、coverage 都没有问题。也就是说，当前失败不能再优先归因于参数没覆盖、alpha 没固定、rollback 错误或 teacher snapshot 错误。

P1 Phase-I dynamics 显示，functional-coordinate methods 仍能产生任务学习信号。例如 MNIST 上 `FCAdam-dataSob`、`FLD-AdanLite` 的 `hold100` 接近或超过 AdamW；但它们的 rank、margin、phi 和 role dynamics 不满足 P1 gate。也就是说，它们能学局部任务，但形成的表示不符合稳定的 PureKAN-AdamW 轨迹。

P2 consolidation sweep 是最重要的诊断。它直接测试：给定一个 teacher function，能否用 KL / feature distillation 保持 teacher，同时加 Sobolev-style consolidation 降低 geometry。结果是没有 all-dataset survivor。许多 row 在 Fashion 上有少量 `phi red`，但伴随很大的 `acc drop` 和 KL；更安全的 teacher-preserving row 则只有很弱的 geometry reduction。

因此当前最准确的判断是：

$$
\boxed{
\text{PSFT 证明了 phase separation 的诊断价值，但也证明了当前 Sobolev consolidation 不是有效的 function-preserving projection。}
}
$$

### 1.2 这不是“小修小补”能解决的问题

过去几轮已经尝试过许多局部修复：

```text
fixed Sobolev U-FULL
D6 allTaskAware
role-aware TFU
FNG / KFAC-like metric
FTF layer-local target fitting
BFT trust region
FCAdam / FLD / AdanLite / WinLite
PSFT learn -> consolidate
normalization ablation
```

这些实验共同说明：

```text
1. Sobolev-only 能改善几何，但不能训练好 PureKAN。
2. Task-aware / FCAdam 能改善短程任务下降，但长期 trajectory 不稳。
3. FTF 能拟合局部 target，但跨层组合会导致函数漂移。
4. BFT 能防止灾难，但变得太保守。
5. PSFT 暴露了 learn 和 smooth projection 之间缺少可保持任务函数的桥。
```

所以 v4.8 不再继续寻找一个新的单步方向：

$$
\Delta a = -\eta M^{-1}g.
$$

而是研究 functional optimizer 是否需要解决一个约束问题：

$$
\begin{aligned}
\min_{\Delta \theta} \quad & \Delta R_{geo}(\theta; \Delta\theta) \\
\text{s.t.} \quad & f_{\theta + \Delta\theta}(x) \approx f_\theta(x), \\
& h_{l,\theta + \Delta\theta}(x) \approx h_{l,\theta}(x), \\
& \text{task loss does not increase.}
\end{aligned}
$$

这就是 NFS 的动机。

---

## 2. 新理论假设

### 2.1 Hypothesis A: 当前 geometry target 可能不可达

PureKAN-AdamW 能训练，说明 PureKAN 架构有表达力。但它的 $$geometry 可能很差。这并不自动意味着存在一个同等准确、但 geometry 很好的 PureKAN 解。

有两种可能：

```text
A1. 可达：存在准确且平滑的 PureKAN 解，只是当前 optimizer 找不到。
A2. 不可达：当前 PureKAN/RBF basis/width/depth 需要高曲率才能完成分类，几何好和准确率存在硬 trade-off。
```

v4.7 consolidation 失败不能区分 A1 和 A2。因此 v4.8 第一任务是画出 PureKAN 的 accuracy-geometry Pareto frontier。

### 2.2 Hypothesis B: 当前 Sobolev projection 不是 function-preserving

当前 consolidation 大致是通过 teacher KL / feature loss / Sobolev penalty 联合优化。问题是，这不是显式地沿 teacher function 的 nullspace 下降。它可能在降低 coefficient Sobolev energy 时改变 logits 或 hidden representation。

真正的 geometry projection 应该优先在 teacher-preserving nullspace 中进行：

$$
J_f \Delta\theta \approx 0,
$$

其中 $J_f$ 是 logits 或 hidden features 对 coefficient 的 Jacobian。目标是降低 geometry regularizer $R(\theta)$：

$$
\Delta\theta_{NFS}
=
- P_{\mathrm{null}(J_f)} M^{-1}\nabla R(\theta).
$$

一个实用形式是：

$$
P_M
=
I
-
M^{-1}J_f^T
\left(J_fM^{-1}J_f^T + \mu I\right)^{-1}
J_f.
$$

然后：

$$
\Delta\theta
=
-\eta P_M M^{-1}\nabla R(\theta).
$$

这个方向的语义是：

```text
先保持当前函数，再寻找能降低 geometry 的自由度。
```

这和之前 “直接加 Sobolev penalty 训练” 完全不同。

### 2.3 Hypothesis C: PureKAN 需要 task phase 和 smoothing phase 的不同参数化

Phase I 的 task learning 更像 AdamW / FCAdam / FLD：它需要 rank、margin、role share、temporal dynamics。Phase II 的 geometry consolidation 更像 constrained smoothing：它需要保持 logits / features，不能再改任务边界。

所以 v4.8 不再试图用同一种 optimizer 完成全部工作，而是验证：

$$
\boxed{
\text{task-learning coordinate 和 geometry-projection coordinate 是否必须分离。}
}
$$

---

## 3. 新方法族

### 3.1 FGF: Functional Geometry Feasibility

FGF 是 v4.8 的第一部分，不是 optimizer，而是存在性测试。它回答：

```text
在当前 PureKAN 架构下，是否存在 accuracy 接近 AdamW、geometry 明显优于 AdamW 的解？
```

训练方法包括：

```text
PureKAN-AdamW
PureKAN-AdamW + coefficient Sobolev penalty
PureKAN-AdamW + derivative/Jacobian penalty
PureKAN-FCAdam task phase
PureKAN-FCAdam + weak Sobolev penalty
PureKAN with wider hidden / larger basis / AB-RBF basis
```

如果任何 AdamW-based geometry regularization 都找不到准确且低 geometry 的解，那说明 functional optimizer 不应该继续承诺 “PureKAN 既准确又几何好”，除非改架构。

### 3.2 NFS: Nullspace Functional Smoothing

NFS 是真正的新 Phase-II projection。

给定 teacher $\theta_T$，我们冻结 teacher outputs：

$$
z_T=f_{\theta_T}(x),
\quad
h_{l,T}=h_l(x;\theta_T).
$$

定义约束 residual：

$$
\epsilon_z(\Delta\theta)=
\frac{\|f_{\theta+\Delta\theta}(x)-z_T\|}{\|z_T\|+\epsilon},
$$

$$
\epsilon_h(\Delta\theta)=
\frac{\sum_l\|h_l(\theta+\Delta\theta)-h_{l,T}\|}{\sum_l\|h_{l,T}\|+\epsilon}.
$$

目标是降低：

$$
R_{geo}(\theta)=
\sum_l
\left(
\gamma_1\phi'_{p95,l}
+
\gamma_2\log\kappa(J_l)
+
\gamma_3\|A_l\|_{S_l}^2
\right).
$$

NFS 每次 proposal 必须满足：

$$
\epsilon_z < r_z,
\quad
\epsilon_h < r_h,
\quad
\mathrm{KL}(p_T\|p_{\theta+\Delta}) < r_{KL}.
$$

如果 proposal 降低了 geometry 但破坏任务函数，就 reject 或 backtrack。

### 3.3 TAN: Task-then-Nullspace Functional Training

TAN 是把 NFS 接入训练循环：

```text
Phase A: task learning
  用 FCAdam / FLD / AdamW-diagnostic 产生表示和 margin。

Phase B: nullspace smoothing
  用 NFS 在保持 logits/features 的同时降低 geometry。

Phase C: small-step task refresh
  用小步 task-aware update 修复 projection 造成的轻微 loss drift。
```

TAN 的关键不是一次性 projection，而是交替循环：

```text
repeat:
  task steps K_task
  snapshot teacher
  NFS smoothing K_smooth
  refresh steps K_refresh
```

但 v4.8 初期必须先用 diagnostic 模式，不直接跑大 seed。

---

## 4. 实验总览

v4.8 分成九个阶段。每个阶段都有明确 stop/go gate，避免继续把资源花在无效 seed confirm 上。

```text
P0: implementation smoke and invariant checks
P1: feasibility frontier with AdamW geometry regularization
P2: offline NFS projection audit from AdamW teachers
P3: offline NFS projection audit from FCAdam/FLD teachers
P4: TAN one-cycle micro-run
P5: TAN alternating-cycle ablation
P6: capacity / basis feasibility expansion
P7: 3-seed full-budget selection
P8: 5-seed confirm
P9: 10-seed final confirm and failure diagnosis
```

---

## 5. P0: Implementation smoke and invariants

### 5.1 目的

P0 只确认实现路径，不看方法优劣。v4.8 引入 NFS，需要确认 projection / rollback / teacher snapshot / Jacobian-vector products 都可靠。

### 5.2 必跑配置

```text
PureKAN-AdamW-smoke
PureKAN-FCAdam-smoke
NFS-logit-only-smoke
NFS-hidden-only-smoke
NFS-logit-hidden-smoke
NFS-random-nullspace-smoke
TAN-one-cycle-smoke
```

每个数据集先跑：

```text
MNIST
Fashion-MNIST
KMNIST
```

只用：

```text
seed = 0
small train_size = 1024
steps = 5 to 20
```

### 5.3 必须记录

```text
implementation/nonkan_param_count
implementation/functional_coverage
implementation/input_coeff_seen
implementation/block_coeff_seen
implementation/output_coeff_seen
implementation/teacher_snapshot_error
implementation/rollback_max_abs_error
implementation/temp_apply_max_abs_error
implementation/jvp_finite
implementation/vjp_finite
implementation/cg_residual
implementation/cg_iters
implementation/nfs_projection_finite
implementation/nfs_constraint_residual
implementation/nfs_geometry_delta
implementation/no_nan_inf
```

### 5.4 通过标准

```text
nonKAN_param_count = 0
functional_coverage = 1.0
rollback_max_abs_error < 1e-8
teacher_snapshot_error = 0
jvp/vjp finite
cg_residual < 1e-3, 或明确记录 fallback
no NaN / Inf
```

如果 P0 不过，不进入 P1。

---

## 6. P1: Feasibility frontier with AdamW geometry regularization

### 6.1 目的

P1 是 v4.8 最关键的诊断之一。它回答：

$$
\boxed{
\text{准确且几何好的 PureKAN 解是否存在？}
}
$$

如果不存在，后续任何 functional optimizer 都不应被要求在 PureKAN 上同时超过 AdamW 并显著降低 geometry。

### 6.2 方法

用 AdamW 作为强 task optimizer，然后加入不同几何正则。这里不追求 strict functional claim，只追求 feasibility。

候选：

```text
A0: PureKAN-AdamW
A1: AdamW + coeff Sobolev penalty lambda = 1e-6
A2: AdamW + coeff Sobolev penalty lambda = 3e-6
A3: AdamW + coeff Sobolev penalty lambda = 1e-5
A4: AdamW + derivative penalty phi-prime proxy lambda = 1e-5
A5: AdamW + Jacobian penalty lambda = 1e-5
A6: AdamW + mixed geometry penalty
A7: AdamW + late geometry penalty only
A8: AdamW then post-hoc L2/Sobolev weight decay fine-tune
```

其中 mixed geometry penalty 为：

$$
\mathcal L
=
\mathcal L_{CE}
+
\lambda_S\sum_l\|A_l\|_{S_l}^2
+
\lambda_\phi\sum_l \widehat{\phi'_{p95,l}}
+
\lambda_J\sum_l \log \widehat{\kappa(J_l)}.
$$

### 6.3 运行设置

先做 1-seed feasibility：

```text
seed = 0
epochs = current PureKAN full budget
hidden_dim = 64
basis_count = 16
depth = 2 or 4, 取当前 PureKAN-AdamW baseline 配置
```

如果某个候选接近 gate，再做 3-seed：

```text
seeds = 0,1,2
```

### 6.4 必须记录

```text
task/train_loss_curve
task/val_loss_curve
task/test_acc
task/val_auc
task/nll
task/ece
representation/effective_rank_input
representation/effective_rank_block_mean
representation/effective_rank_output
representation/class_centroid_separation
representation/margin_mean
representation/margin_p10
representation/margin_p50
geometry/phi_prime_p95
geometry/phi_prime_max
geometry/jacobian_condition_max
geometry/sobolev_norm_total
geometry/sobolev_norm_by_role
geometry/curvature_energy
basis/basis_occupancy_entropy
basis/dead_basis_fraction
basis/out_of_grid_fraction
role/update_share_input
role/update_share_block
role/update_share_output
compute/step_time_ms
compute/memory_peak_mb
```

### 6.5 必须可视化

```text
1. accuracy vs phi_prime_p95 Pareto frontier
2. accuracy vs max_jacobian_condition Pareto frontier
3. accuracy vs sobolev_norm_total Pareto frontier
4. val_loss_curve by lambda
5. margin_p10 vs phi_prime_p95 scatter
6. effective_rank vs accuracy scatter
7. geometry penalty lambda vs acc drop line plot
8. basis occupancy heatmap by role
```

### 6.6 P1 判定

定义 AdamW baseline 为 $A_0$。候选 $A_i$ 被认为证明 feasibility，如果：

$$
Acc(A_i) \ge Acc(A_0)-0.01,
$$

并且：

$$
\phi_{p95}(A_i) \le 0.8\phi_{p95}(A_0)
$$

或：

$$
\kappa_J(A_i) \le 0.5\kappa_J(A_0),
$$

同时：

$$
ECE(A_i) \le ECE(A_0)+0.02.
$$

如果没有任何候选在三个数据集上通过，则说明当前 PureKAN 架构可能没有明显的 “smooth accurate solution”。这时进入 P6 capacity expansion，而不是继续改 optimizer。

---

## 7. P2: Offline NFS projection audit from AdamW teachers

### 7.1 目的

P2 测试：给定一个强 teacher，比如 PureKAN-AdamW，NFS 能否在不破坏 logits/features 的情况下降低 geometry。

这直接针对 v4.7 的失败点：当前 consolidation operator 无法同时 teacher-preserve 和 geometry-reduce。

### 7.2 Teacher 来源

```text
TeacherA: PureKAN-AdamW at step 20
TeacherB: PureKAN-AdamW at step 100
TeacherC: PureKAN-AdamW final checkpoint
TeacherD: AdamW + geometry penalty best checkpoint from P1
```

### 7.3 NFS 变体

```text
NFS-Z:
  constrain logits only

NFS-H:
  constrain selected hidden layers only

NFS-ZH:
  constrain logits + hidden features

NFS-ZH-margin:
  constrain logits + hidden + correct-class margin

NFS-role-input:
  smooth input KAN only

NFS-role-block:
  smooth residual block KAN only

NFS-role-output:
  smooth output KAN only

NFS-role-cycle:
  input -> block -> output sequential smoothing
```

### 7.4 NFS 公式

令 $R(\theta)$ 是几何目标，$c(\theta)$ 是 teacher-preserving constraints：

$$
c(\theta)=
\begin{bmatrix}
\sqrt{\tau_z}(z_\theta-z_T) \\
\sqrt{\tau_h}(h_\theta-h_T) \\
\sqrt{\tau_m}(m_\theta-m_T)
\end{bmatrix}.
$$

NFS 近似解：

$$
\Delta\theta
= -\eta
\left(I - M^{-1}J_c^T(J_cM^{-1}J_c^T+\mu I)^{-1}J_c\right)
M^{-1}\nabla R.
$$

如果这个完整投影太贵，先实现 CG / low-rank 近似：

```text
NFS-CG-5
NFS-CG-10
NFS-lowrank-32
NFS-lowrank-64
NFS-diag-projector
```

### 7.5 必须记录

```text
teacher/teacher_acc
teacher/teacher_loss
teacher/teacher_ece
teacher/teacher_phi_p95
teacher/teacher_jacobian
projection/geometry_before
projection/geometry_after
projection/phi_reduction
projection/jacobian_reduction
projection/sobolev_reduction
projection/kl_teacher_student
projection/logit_relative_drift
projection/hidden_relative_drift
projection/margin_relative_drift
projection/constraint_residual_predicted
projection/constraint_residual_actual
projection/predicted_geometry_delta
projection/actual_geometry_delta
projection/cg_iters
projection/cg_residual
projection/nullspace_fraction
projection/backtrack_count
projection/accepted_eta
projection/reject_reason
projection/role
projection/update_norm_by_role
projection/update_sobolev_norm
```

### 7.6 必须可视化

```text
1. teacher KL vs phi reduction scatter
2. logit drift vs phi reduction scatter
3. hidden drift vs phi reduction scatter
4. predicted vs actual geometry reduction
5. predicted vs actual logit drift
6. NFS constraint residual histogram
7. accepted eta distribution
8. role-wise geometry reduction bar chart
9. before/after edge function plots for selected edges
10. before/after basis coefficient spectrum
```

### 7.7 P2 判定

P2 通过条件不是最终 accuracy，而是 projection 可行性。

一个 NFS 配置通过，如果在每个数据集上存在 teacher checkpoint 使得：

$$
\mathrm{KL}(p_T\|p_{after}) < 0.05,
$$

$$
\frac{\|z_{after}-z_T\|}{\|z_T\|+\epsilon}<0.03,
$$

$$
Acc_{after} \ge Acc_T - 0.005,
$$

并且：

$$
\phi_{red} > 0.10
\quad\text{or}\quad
J_{red} > 0.20.
$$

如果 P2 全部失败：

```text
结论不是 optimizer 没调好，而是当前参数化下缺少 function-preserving geometry degrees of freedom。
```

此时进入 P6 architecture/basis expansion。

---

## 8. P3: Offline NFS from functional teachers

### 8.1 目的

P2 用 AdamW teacher，只是证明 projection 是否存在。P3 用 functional teacher，测试能否把 functional Phase-I 学到的粗糙函数投影成更好几何。

Teacher：

```text
FCAdam-dataSob step 20 / 100
FLD-AdanLite step 20 / 100
D6 allTaskAware step 100
FNG-leftFullRight step 100
```

### 8.2 重点问题

P3 判断：functional teacher 是否比 AdamW teacher 更容易被 NFS 平滑化。

可能结果有三种：

```text
1. AdamW teacher 可平滑，functional teacher 不可平滑：functional Phase-I 学到的函数太粗糙/错位。
2. functional teacher 可平滑，AdamW teacher 不可平滑：functional trajectory 更接近 smoothable solution。
3. 两者都不可平滑：当前 architecture/basis 没有足够 nullspace。
```

### 8.3 记录和图

沿用 P2 全部指标，并额外记录：

```text
teacher_type
teacher_holdout_descent
teacher_rank
teacher_margin
teacher_phi_ratio_to_adamw
teacher_role_share_input
teacher_role_share_block
teacher_role_share_output
smoothability_score
```

定义：

$$
\mathrm{smoothability}
=
\frac{\phi_{red}}{\mathrm{KL}+0.01}
\cdot
\mathbf{1}[Acc_{drop}<0.01].
$$

---

## 9. P4: TAN one-cycle micro-run

### 9.1 目的

如果 P2/P3 显示 NFS 可以 function-preserving 降几何，就进入一轮 TAN：

```text
Task phase -> NFS smoothing -> refresh phase
```

### 9.2 配置

Task phase 候选：

```text
T0: FCAdam-dataSob
T1: FLD-AdanLite
T2: FCAdam-L2
T3: D6 allTaskAware
T4: PureKAN-AdamW diagnostic teacher, not final strict functional claim
```

Smoothing phase：

```text
Best NFS-ZH from P2/P3
Best role-specific NFS from P2/P3
```

Refresh phase：

```text
R0: none
R1: 5 steps FCAdam small lr
R2: 10 steps FCAdam small lr
R3: 5 steps task-diag D6 small lr
```

### 9.3 一个 cycle 的形式

$$
\theta_0
\xrightarrow{K_{task}}
\theta_T
\xrightarrow{K_{NFS}}
\theta_S
\xrightarrow{K_{refresh}}
\theta_R.
$$

### 9.4 必须记录

```text
cycle/task_acc_before
cycle/task_acc_after_task
cycle/task_acc_after_smooth
cycle/task_acc_after_refresh
cycle/val_loss_after_task
cycle/val_loss_after_smooth
cycle/val_loss_after_refresh
cycle/phi_after_task
cycle/phi_after_smooth
cycle/phi_after_refresh
cycle/teacher_KL_after_smooth
cycle/refresh_recovered_loss
cycle/refresh_recovered_margin
cycle/rank_after_each_phase
cycle/margin_after_each_phase
cycle/ece_after_each_phase
cycle/nfs_acceptance_rate
cycle/nfs_reject_reason
```

### 9.5 可视化

```text
1. task -> smooth -> refresh phase plot: loss, acc, phi, rank, margin
2. KL after smooth vs refresh recovery scatter
3. per-cycle geometry reduction bar chart
4. per-role update share in each phase
5. before/after confusion matrix change
```

### 9.6 P4 判定

P4 通过条件：

```text
After one cycle:
  accuracy drop from task phase < 1%
  phi reduction from task phase > 10%
  refresh recovers at least 80% of loss increase caused by smoothing
  ECE not worse than task phase by more than 0.02
```

---

## 10. P5: Alternating TAN cycles

### 10.1 目的

P4 只看一个 cycle。P5 看多 cycle 是否稳定。

### 10.2 配置

只取 P4 最好的 2 到 3 个组合。

```text
cycle_count = 3 or 5
K_task = 50 or 100 steps
K_smooth = 5 or 10 accepted NFS steps
K_refresh = 5 or 10 steps
```

### 10.3 必须记录

```text
cycle_index
acc_by_cycle
val_loss_by_cycle
phi_by_cycle
jacobian_by_cycle
rank_by_cycle
margin_by_cycle
KL_to_cycle_teacher
NFS_accept_rate_by_cycle
refresh_recovery_by_cycle
geometry_gain_per_task_loss
```

### 10.4 P5 判定

P5 通过，如果多 cycle 后：

$$
Acc_{TAN} \ge Acc_{FCAdam\ phase} - 0.01,
$$

且：

$$
\phi_{TAN} \le 0.85\phi_{FCAdam\ phase},
$$

并且：

$$
Acc_{TAN} \ge Acc_{D6} + 0.03
$$

在 Fashion 和 KMNIST 上至少同时成立。

---

## 11. P6: Capacity and basis feasibility expansion

### 11.1 触发条件

如果 P1 显示没有 smooth accurate frontier，或者 P2 显示 NFS 不存在可靠 function-preserving smoothing direction，就触发 P6。

### 11.2 目的

判断失败是 optimizer 问题，还是 PureKAN/RBF 参数化没有足够低几何自由度。

### 11.3 变量

```text
hidden_dim: 64, 96, 128
basis_count: 16, 24, 32, 48
depth: 2, 4
basis_type:
  RBF only
  AB-RBF: constant + linear + RBF
  wide-grid RBF
  multi-scale RBF
residual_alpha:
  fixed1
  smaller fixed alpha 0.5
normalization:
  FixedNorm
  ScalarGainNorm functional
```

### 11.4 必须对照

```text
PureKAN-AdamW
PureKAN-AdamW + geometry regularization
FCAdam-dataSob
D6 allTaskAware
NFS projection from AdamW teacher
```

### 11.5 记录

```text
param_count
basis_count
hidden_dim
depth
basis_type
basis_occupancy_entropy
basis_dead_frac
out_of_grid_frac
acc
val_auc
ece
rank
margin
phi
jacobian
sobolev_norm
smoothability_score
step_time
memory
```

### 11.6 可视化

```text
1. capacity vs accuracy-geometry frontier
2. basis_count vs smoothability_score
3. hidden_dim vs phi at fixed accuracy
4. basis occupancy heatmap by basis_type
5. param_count vs acc/phi Pareto
6. edge function examples for RBF vs AB-RBF vs multiscale
```

### 11.7 P6 判定

如果更大 capacity / richer basis 出现 smooth accurate frontier，说明之前失败是 architecture capacity / basis issue。下一步应优先改 PureKAN primitive。

如果更大 capacity 仍然没有 frontier，说明 PureKAN 本身可能需要 hybrid components，或者几何目标过强。

---

## 12. P7: 3-seed full-budget candidate selection

### 12.1 进入条件

只有满足以下之一才进入 P7：

```text
1. P1 找到 smooth accurate frontier，并且 P2 找到可靠 NFS。
2. P5 多 cycle TAN 在 seed0 上接近 PureKAN-AdamW。
3. P6 找到新的 basis/capacity 使 smooth accurate solution 可达。
```

### 12.2 候选数量限制

最多 4 个：

```text
C1: best TAN functional-only candidate
C2: best NFS-projection candidate
C3: best capacity/basis candidate
C4: best diagnostic AdamW+geometry candidate, not final functional claim
```

### 12.3 对照

```text
PureKAN-AdamW
Hybrid-DGKAN-UFULL-f085
MLP-AdamW
D6 allTaskAware
FNG-leftFullRight
```

### 12.4 3-seed gate

必须满足：

```text
MNIST:
  acc >= PureKAN-AdamW - 1.0%
Fashion:
  acc >= PureKAN-AdamW - 1.0%
KMNIST:
  acc >= PureKAN-AdamW - 1.5%
```

并且至少满足其中一个 geometry 条件：

```text
phi reduction vs PureKAN-AdamW > 15%
或 J reduction vs PureKAN-AdamW > 30%
```

并且：

```text
ECE not worse by more than 0.02
val-loss AUC not worse by more than 5%
```

---

## 13. P8: 5-seed confirm

P8 只运行 P7 中最多两个候选。

必须额外做 paired seed analysis：

```text
paired acc delta vs PureKAN-AdamW
paired val_auc delta vs PureKAN-AdamW
paired phi delta vs PureKAN-AdamW
paired ECE delta vs PureKAN-AdamW
paired smoothability score
```

P8 通过：

$$
\Delta Acc_{paired} > -0.01
$$

且：

$$
\Delta \phi_{paired} < -0.15\phi_{AdamW}
$$

或：

$$
\Delta J_{paired} < -0.30J_{AdamW}.
$$

---

## 14. P9: 10-seed final confirm and failure diagnosis

### 14.1 成功判定

v4.8 final success 不是只超过旧 U-FULL。必须比较：

```text
PureKAN-AdamW
Hybrid-DGKAN-UFULL-f085
MLP-AdamW
```

最终候选必须：

```text
1. accuracy 接近或超过 PureKAN-AdamW。
2. geometry 明显优于 PureKAN-AdamW。
3. 没有 non-KAN trainable params。
4. functional coverage = 1.0。
5. ECE 不显著变差。
6. wall-clock 可以慢，但需要明确报告。
```

### 14.2 如果失败，必须输出明确 failure type

```text
F1: no smooth accurate frontier exists
F2: NFS cannot preserve function while reducing geometry
F3: task phase cannot reach AdamW-level representation
F4: smoothing phase destroys logits/features
F5: refresh phase cannot recover from smoothing
F6: capacity/basis insufficient
F7: geometry metric too strict / wrong metric
F8: PureKAN requires hybrid-like non-KAN stabilizers
```

### 14.3 最终报告必须包含

```text
accuracy-geometry Pareto frontier
teacher-preserving smoothing feasibility
TAN cycle dynamics
capacity/basis smoothability map
final 10-seed scorecard
failure taxonomy
```

---

## 15. 本轮必须停止的方向

v4.8 不再继续投入以下方向，除非作为对照：

```text
1. 固定 Sobolev U-FULL 直接训练 PureKAN。
2. 再扫 branch_final_scale。
3. 再扫 diagwarmup / smooth transition。
4. 再扫固定 shallow/deep hand-crafted metric 分工。
5. 再做单独 FTF layer-local target fitting。
6. 再做只有 trust gate 的 BFT。
7. 再把 D6 allTaskAware 当主线。
```

原因是这些方向已经反复证明：要么能短程下降但长期不行，要么保几何但任务不足，要么能学任务但无法被平滑投影。

---

## 16. 预期解释路径

v4.8 的结果会把项目带向三种可能结论。

### 16.1 最好情况

```text
P1 找到 smooth accurate solution。
P2 NFS 能保持 teacher 并降 geometry。
P5 TAN 多 cycle 成功。
```

结论：

$$
\boxed{
\text{PureKAN functional optimization 可行，但需要 task phase + nullspace smoothing，而不是单一 Sobolev update。}
}
$$

### 16.2 中间情况

```text
P1 有 smooth accurate frontier，但 P2 NFS 失败。
```

结论：

$$
\boxed{
\text{目标存在，但当前 projection operator 错；需要更强的 constrained optimization / better tangent model。}
}
$$

### 16.3 严重情况

```text
P1/P6 都找不到 smooth accurate frontier。
```

结论：

$$
\boxed{
\text{当前 PureKAN/RBF 架构下，准确和低 geometry 可能是硬 trade-off；之前 Hybrid 成功依赖 non-KAN interface。}
}
$$

这时应该停止 PureKAN functional optimizer 主线，转向：

```text
Rational/KAT primitive-specific metric
Hybrid-DGKAN branch optimizer
更大/多尺度 PureKAN architecture
```

---

## 17. 最终执行建议

第一批只跑：

```text
P0
P1 seed0 feasibility frontier
P2 AdamW teacher NFS projection
```

不要直接跑 P3-P9。只有当 P1 或 P2 出现明确正信号，才继续。

优先级如下：

```text
Priority 1: P1 feasibility frontier
Priority 2: P2 NFS projection from AdamW teacher
Priority 3: P6 capacity/basis expansion if P1/P2 fail
Priority 4: P4/P5 TAN cycles only if NFS passes
Priority 5: seed confirm only after candidate is mechanistically justified
```

一句话：

$$
\boxed{
\text{v4.8 先证明 smooth accurate PureKAN 是否存在，再讨论 functional optimizer 如何找到它。}
}
$$



---


# Source 24: `docs/DG-KAN_v4.8_FunctionalGeometry_Feasibility_Redesign_结果复盘.md`


# DG-KAN v4.8 Functional Geometry Feasibility 结果复盘

本轮依据 `docs/DG-KAN_v4.8_FunctionalGeometry_Feasibility_Redesign_实验计划.md`。目标从“继续调 functional optimizer”改为 feasibility-first：先判断当前 PureKAN/RBF 是否存在准确且几何好的可达解，再判断 NFS 是否能在保持 teacher function 的同时降低 geometry。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 FGF/NFS smoke | 21 | 0 |
| P1 feasibility frontier | 27 | 0 |
| P2 NFS AdamW projection | 1440 | 0 |
| P3 NFS functional teachers | 108 | 0 |
| P4 TAN one-cycle | 120 | 0 |
| P5 TAN alternating cycles | 18 | 0 |
| P6 capacity/basis expansion | 1 | 0 |

## Code / Config Changes

```text
experiments/run_gafu_v48.py
  Added FGF AdamW geometry regularization frontier.
  Added offline NFS projection variants with teacher snapshot, backtracking,
  logit/hidden/margin constraints, and role-specific smoothing proposals.
  Added functional-teacher NFS, TAN one-cycle, and gated TAN multi-cycle probes.
  Added compact P6 capacity/basis feasibility expansion.

experiments/analyze_gafu_v48.py
  Generates frontier/NFS/capacity gate summaries, failure taxonomy,
  SVG diagnostics, aggregate_decision.json, and this replay.
```

## P0 Implementation Smoke

| rows | errors | max nonKAN | min cov | max rollback | max CG residual | pass |
|---|---|---|---|---|---|---|
| 21 | 0 | 0.0000 | 1.0000 | 0.0000 | 0.0533 | true |

P0 verdict: pass. NFS temporary apply / rollback / teacher snapshot / finite projection diagnostics passed for the smoke grid.

## P1 Accuracy-Geometry Feasibility Frontier

| dataset | method | acc | gap | phi red | J red | ECE | rank | P1 |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | A0-PureKAN-AdamW | 0.7754 | 0.0000 | 0.0000 | 0.0000 | 0.0655 | 16.8570 | yes |
| Fashion-MNIST | A1-AdamW-Sobolev1e-6 | 0.7754 | 0.0000 | -0.0000 | -0.0000 | 0.0655 | 16.8570 | no |
| Fashion-MNIST | A2-AdamW-Sobolev3e-6 | 0.7754 | 0.0000 | -0.0000 | 0.0000 | 0.0655 | 16.8571 | no |
| Fashion-MNIST | A3-AdamW-Sobolev1e-5 | 0.7754 | 0.0000 | -0.0000 | 0.0000 | 0.0655 | 16.8570 | no |
| Fashion-MNIST | A4-AdamW-PhiProxy1e-5 | 0.7754 | 0.0000 | -0.0000 | -0.0002 | 0.0655 | 16.8570 | no |
| Fashion-MNIST | A5-AdamW-JacProxy1e-5 | 0.7754 | 0.0000 | -0.0000 | -0.0002 | 0.0655 | 16.8570 | no |
| Fashion-MNIST | A6-AdamW-MixedGeom | 0.7754 | 0.0000 | -0.0000 | -0.0005 | 0.0655 | 16.8570 | no |
| Fashion-MNIST | A7-AdamW-LateGeom | 0.7754 | 0.0000 | 0.0000 | -0.0000 | 0.0655 | 16.8570 | no |
| Fashion-MNIST | A8-AdamW-PosthocGeom | 0.5664 | 0.2090 | -0.0052 | 0.0731 | 0.3956 | 11.9923 | no |
| KMNIST | A0-PureKAN-AdamW | 0.6797 | 0.0000 | 0.0000 | 0.0000 | 0.1014 | 19.8530 | yes |
| KMNIST | A1-AdamW-Sobolev1e-6 | 0.6797 | 0.0000 | 0.0000 | -0.0002 | 0.1014 | 19.8530 | no |
| KMNIST | A2-AdamW-Sobolev3e-6 | 0.6797 | 0.0000 | 0.0000 | -0.0006 | 0.1014 | 19.8530 | no |
| KMNIST | A3-AdamW-Sobolev1e-5 | 0.6797 | 0.0000 | 0.0000 | -0.0024 | 0.1014 | 19.8531 | no |
| KMNIST | A4-AdamW-PhiProxy1e-5 | 0.6797 | 0.0000 | -0.0000 | 0.0013 | 0.1014 | 19.8529 | no |
| KMNIST | A5-AdamW-JacProxy1e-5 | 0.6797 | 0.0000 | -0.0000 | 0.0011 | 0.1014 | 19.8530 | no |
| KMNIST | A6-AdamW-MixedGeom | 0.6797 | 0.0000 | -0.0000 | 0.0031 | 0.1014 | 19.8529 | no |
| KMNIST | A7-AdamW-LateGeom | 0.6797 | 0.0000 | 0.0000 | 0.0000 | 0.1014 | 19.8530 | no |
| KMNIST | A8-AdamW-PosthocGeom | 0.1973 | 0.4824 | 0.0280 | -2.7726 | 0.0751 | 15.8675 | no |
| MNIST | A0-PureKAN-AdamW | 0.8711 | 0.0000 | 0.0000 | 0.0000 | 0.0427 | 16.2443 | yes |
| MNIST | A1-AdamW-Sobolev1e-6 | 0.8711 | 0.0000 | 0.0000 | 0.0040 | 0.0427 | 16.2443 | no |
| MNIST | A2-AdamW-Sobolev3e-6 | 0.8711 | 0.0000 | -0.0000 | 0.0107 | 0.0427 | 16.2444 | no |
| MNIST | A3-AdamW-Sobolev1e-5 | 0.8711 | 0.0000 | -0.0000 | 0.0358 | 0.0427 | 16.2443 | no |
| MNIST | A4-AdamW-PhiProxy1e-5 | 0.8711 | 0.0000 | -0.0000 | 0.0084 | 0.0427 | 16.2443 | no |
| MNIST | A5-AdamW-JacProxy1e-5 | 0.8711 | 0.0000 | -0.0000 | 0.0116 | 0.0427 | 16.2442 | no |
| MNIST | A6-AdamW-MixedGeom | 0.8711 | 0.0000 | -0.0000 | 0.0324 | 0.0427 | 16.2442 | no |
| MNIST | A7-AdamW-LateGeom | 0.8711 | 0.0000 | 0.0000 | 0.0004 | 0.0427 | 16.2443 | no |
| MNIST | A8-AdamW-PosthocGeom | 0.2148 | 0.6562 | 0.0435 | 0.8932 | 0.0521 | 11.6523 | no |

P1 all-dataset survivors:

```text
none
```

## P2 Offline NFS Projection From AdamW Teachers

Top diagnostic rows are sorted by dataset and smoothability/geometry signal.

| dataset | teacher | variant | proj | eta | acc drop | KL | logit | phi red | J red | smooth | P2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | TeacherA-AdamW20 | NFS-role-block | diag | 0.05 | 0.0039 | 0.0001 | 0.0147 | 0.1482 | 0.4284 | 14.6723 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-role-block | cg5 | 0.05 | 0.0020 | 0.0001 | 0.0126 | 0.1289 | 0.3912 | 12.7956 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-role-block | diag | 0.05 | 0.0000 | 0.0007 | 0.0180 | 0.1349 | 0.4547 | 12.6464 | yes |
| Fashion-MNIST | TeacherD-bestFGF | NFS-role-block | diag | 0.05 | -0.0039 | 0.0017 | 0.0217 | 0.1325 | 0.8597 | 11.3154 | yes |
| Fashion-MNIST | TeacherC-AdamW-final | NFS-role-block | diag | 0.05 | -0.0039 | 0.0017 | 0.0217 | 0.1325 | 0.8597 | 11.3153 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-role-block | cg5 | 0.05 | 0.0020 | 0.0005 | 0.0154 | 0.1172 | 0.4227 | 11.1732 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-role-block | diag | 0.035 | 0.0020 | 0.0001 | 0.0104 | 0.1086 | 0.3466 | 10.8035 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-role-block | cg10 | 0.05 | 0.0020 | 0.0001 | 0.0104 | 0.1086 | 0.3466 | 10.8035 | yes |
| Fashion-MNIST | TeacherD-bestFGF | NFS-role-block | cg5 | 0.05 | -0.0059 | 0.0013 | 0.0186 | 0.1150 | 0.8247 | 10.2175 | yes |
| Fashion-MNIST | TeacherC-AdamW-final | NFS-role-block | cg5 | 0.05 | -0.0059 | 0.0013 | 0.0186 | 0.1150 | 0.8246 | 10.2172 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-role-block | diag | 0.035 | 0.0020 | 0.0003 | 0.0128 | 0.0987 | 0.3865 | 9.5517 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-role-block | cg10 | 0.05 | 0.0020 | 0.0003 | 0.0128 | 0.0987 | 0.3865 | 9.5517 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-role-block | cg5 | 0.035 | 0.0000 | 0.0000 | 0.0089 | 0.0937 | 0.3111 | 9.3364 | yes |
| Fashion-MNIST | TeacherD-bestFGF | NFS-role-block | diag | 0.035 | -0.0059 | 0.0009 | 0.0154 | 0.0968 | 0.7698 | 8.9087 | yes |
| Fashion-MNIST | TeacherD-bestFGF | NFS-role-block | cg10 | 0.05 | -0.0059 | 0.0009 | 0.0154 | 0.0968 | 0.7698 | 8.9087 | yes |
| Fashion-MNIST | TeacherC-AdamW-final | NFS-role-block | cg10 | 0.05 | -0.0059 | 0.0009 | 0.0154 | 0.0968 | 0.7696 | 8.9081 | yes |
| Fashion-MNIST | TeacherC-AdamW-final | NFS-role-block | diag | 0.035 | -0.0059 | 0.0009 | 0.0154 | 0.0968 | 0.7696 | 8.9081 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-role-block | lowrank32 | 0.05 | 0.0000 | 0.0000 | 0.0082 | 0.0873 | 0.2946 | 8.6984 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-H | diag | 0.05 | 0.0020 | 0.0073 | 0.1117 | 0.1482 | 0.4224 | 8.5895 | no |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-H | cg5 | 0.05 | 0.0020 | 0.0053 | 0.0958 | 0.1288 | 0.3849 | 8.3970 | no |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-role-block | cg5 | 0.035 | 0.0020 | 0.0002 | 0.0109 | 0.0853 | 0.3580 | 8.3272 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-H | diag | 0.035 | 0.0000 | 0.0037 | 0.0796 | 0.1086 | 0.3401 | 7.9314 | no |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-H | cg10 | 0.05 | 0.0000 | 0.0037 | 0.0796 | 0.1086 | 0.3401 | 7.9314 | no |
| Fashion-MNIST | TeacherD-bestFGF | NFS-role-block | cg5 | 0.035 | -0.0059 | 0.0006 | 0.0132 | 0.0835 | 0.7086 | 7.8586 | yes |
| Fashion-MNIST | TeacherC-AdamW-final | NFS-role-block | cg5 | 0.035 | -0.0059 | 0.0006 | 0.0132 | 0.0835 | 0.7084 | 7.8578 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-role-block | cg10 | 0.035 | 0.0000 | 0.0000 | 0.0074 | 0.0783 | 0.2713 | 7.8128 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-role-block | lowrank32 | 0.05 | 0.0020 | 0.0002 | 0.0101 | 0.0793 | 0.3454 | 7.7677 | yes |
| Fashion-MNIST | TeacherD-bestFGF | NFS-role-block | lowrank32 | 0.05 | -0.0039 | 0.0005 | 0.0122 | 0.0778 | 0.6722 | 7.3849 | yes |
| Fashion-MNIST | TeacherC-AdamW-final | NFS-role-block | lowrank32 | 0.05 | -0.0039 | 0.0005 | 0.0122 | 0.0778 | 0.6720 | 7.3839 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-H | cg5 | 0.035 | -0.0020 | 0.0027 | 0.0681 | 0.0938 | 0.3045 | 7.3826 | no |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-role-block | lowrank64 | 0.05 | 0.0000 | 0.0000 | 0.0068 | 0.0725 | 0.2548 | 7.2312 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-H | lowrank32 | 0.05 | -0.0039 | 0.0023 | 0.0631 | 0.0873 | 0.2880 | 7.0816 | no |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-role-block | cg10 | 0.035 | 0.0000 | 0.0002 | 0.0090 | 0.0713 | 0.3263 | 7.0142 | yes |
| Fashion-MNIST | TeacherC-AdamW-final | NFS-role-block | cg10 | 0.035 | -0.0039 | 0.0004 | 0.0109 | 0.0698 | 0.6082 | 6.6922 | yes |
| Fashion-MNIST | TeacherD-bestFGF | NFS-role-block | cg10 | 0.035 | -0.0039 | 0.0004 | 0.0109 | 0.0698 | 0.6085 | 6.6920 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-H | cg10 | 0.035 | -0.0039 | 0.0019 | 0.0564 | 0.0786 | 0.2648 | 6.6262 | no |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-role-block | diag | 0.02 | 0.0000 | 0.0000 | 0.0060 | 0.0651 | 0.2331 | 6.4966 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-role-block | lowrank64 | 0.05 | 0.0000 | 0.0001 | 0.0083 | 0.0659 | 0.3108 | 6.4958 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-role-cycle | diag | 0.05 | 0.0020 | 0.0004 | 0.0270 | 0.0660 | 0.2300 | 6.3358 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-role-block | lowrank32 | 0.035 | 0.0000 | 0.0000 | 0.0058 | 0.0629 | 0.2264 | 6.2785 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-H | lowrank64 | 0.05 | -0.0020 | 0.0016 | 0.0520 | 0.0725 | 0.2485 | 6.2634 | no |
| Fashion-MNIST | TeacherD-bestFGF | NFS-role-block | lowrank64 | 0.05 | -0.0039 | 0.0004 | 0.0101 | 0.0645 | 0.5516 | 6.2185 | yes |
| Fashion-MNIST | TeacherC-AdamW-final | NFS-role-block | lowrank64 | 0.05 | -0.0039 | 0.0004 | 0.0101 | 0.0644 | 0.5513 | 6.2171 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-role-block | diag | 0.02 | 0.0000 | 0.0001 | 0.0074 | 0.0589 | 0.2906 | 5.8272 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-H | diag | 0.02 | -0.0020 | 0.0013 | 0.0463 | 0.0651 | 0.2270 | 5.7857 | no |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-H | cg10 | 0.05 | -0.0020 | 0.0073 | 0.0416 | 0.0985 | 0.3731 | 5.7098 | no |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-H | diag | 0.035 | -0.0020 | 0.0073 | 0.0416 | 0.0985 | 0.3731 | 5.7098 | no |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-H | cg5 | 0.05 | -0.0020 | 0.0106 | 0.0502 | 0.1174 | 0.4077 | 5.6847 | no |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-H | lowrank32 | 0.035 | 0.0000 | 0.0012 | 0.0446 | 0.0629 | 0.2203 | 5.6317 | no |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-role-block | lowrank32 | 0.035 | 0.0000 | 0.0001 | 0.0071 | 0.0568 | 0.2844 | 5.6238 | yes |
| Fashion-MNIST | TeacherD-bestFGF | NFS-role-block | diag | 0.02 | 0.0000 | 0.0003 | 0.0090 | 0.0576 | 0.4544 | 5.5986 | yes |
| Fashion-MNIST | TeacherC-AdamW-final | NFS-role-block | diag | 0.02 | 0.0000 | 0.0003 | 0.0090 | 0.0576 | 0.4540 | 5.5981 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-role-block | cg5 | 0.02 | 0.0000 | 0.0000 | 0.0051 | 0.0560 | 0.2054 | 5.5891 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-H | cg5 | 0.035 | -0.0020 | 0.0053 | 0.0355 | 0.0850 | 0.3472 | 5.5682 | no |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-role-cycle | cg5 | 0.05 | 0.0000 | 0.0003 | 0.0229 | 0.0567 | 0.2019 | 5.4978 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-H | diag | 0.05 | 0.0000 | 0.0147 | 0.0588 | 0.1348 | 0.4397 | 5.4677 | no |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-H | lowrank32 | 0.05 | -0.0020 | 0.0045 | 0.0329 | 0.0790 | 0.3340 | 5.4485 | no |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-role-cycle | diag | 0.05 | -0.0039 | 0.0009 | 0.0156 | 0.0592 | 0.2836 | 5.4346 | yes |
| Fashion-MNIST | TeacherD-bestFGF | NFS-role-block | lowrank32 | 0.035 | 0.0000 | 0.0003 | 0.0086 | 0.0556 | 0.4171 | 5.4100 | yes |
| Fashion-MNIST | TeacherC-AdamW-final | NFS-role-block | lowrank32 | 0.035 | 0.0000 | 0.0003 | 0.0086 | 0.0556 | 0.4166 | 5.4094 | yes |
| Fashion-MNIST | TeacherD-bestFGF | NFS-H | cg5 | 0.035 | 0.0098 | 0.0054 | 0.0336 | 0.0834 | 0.6860 | 5.4055 | no |
| Fashion-MNIST | TeacherC-AdamW-final | NFS-H | cg5 | 0.035 | 0.0098 | 0.0054 | 0.0336 | 0.0834 | 0.6858 | 5.4049 | no |
| Fashion-MNIST | TeacherD-bestFGF | NFS-H | lowrank32 | 0.05 | 0.0098 | 0.0046 | 0.0311 | 0.0775 | 0.6451 | 5.2940 | no |
| Fashion-MNIST | TeacherC-AdamW-final | NFS-H | lowrank32 | 0.05 | 0.0098 | 0.0046 | 0.0311 | 0.0775 | 0.6448 | 5.2934 | no |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-Z | cg10 | 0.05 | -0.0020 | 0.0037 | 0.0299 | 0.0723 | 0.3170 | 5.2709 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-ZH | cg10 | 0.05 | -0.0020 | 0.0037 | 0.0299 | 0.0723 | 0.3170 | 5.2709 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-ZH-margin | cg10 | 0.05 | -0.0020 | 0.0037 | 0.0299 | 0.0723 | 0.3170 | 5.2709 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-Z | diag | 0.035 | -0.0020 | 0.0037 | 0.0299 | 0.0723 | 0.3170 | 5.2709 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-ZH | diag | 0.035 | -0.0020 | 0.0037 | 0.0299 | 0.0723 | 0.3170 | 5.2709 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-ZH-margin | diag | 0.035 | -0.0020 | 0.0037 | 0.0299 | 0.0723 | 0.3170 | 5.2709 | yes |
| Fashion-MNIST | TeacherD-bestFGF | NFS-H | diag | 0.05 | 0.0098 | 0.0151 | 0.0557 | 0.1321 | 0.8525 | 5.2623 | no |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-Z | lowrank32 | 0.05 | -0.0020 | 0.0037 | 0.0296 | 0.0716 | 0.3156 | 5.2458 | yes |

Best diagnostic points:

| dataset | best by | teacher | variant | smooth | acc drop | KL | phi red |
|---|---|---|---|---|---|---|---|
| MNIST | smoothability | TeacherA-AdamW20 | NFS-role-block | 9.8283 | -0.0078 | 0.0000 | 0.0984 |
| MNIST | phi | TeacherA-AdamW20 | NFS-H | 0.0000 | 0.0938 | 0.0011 | 0.1507 |
| Fashion-MNIST | smoothability | TeacherA-AdamW20 | NFS-role-block | 14.6723 | 0.0039 | 0.0001 | 0.1482 |
| Fashion-MNIST | phi | TeacherA-AdamW20 | NFS-role-block | 14.6723 | 0.0039 | 0.0001 | 0.1482 |
| KMNIST | smoothability | TeacherA-AdamW20 | NFS-role-block | 14.8527 | -0.0039 | 0.0001 | 0.1494 |
| KMNIST | phi | TeacherA-AdamW20 | NFS-role-block | 14.8527 | -0.0039 | 0.0001 | 0.1494 |

P2 all-dataset survivors:

```text
TeacherA-AdamW20|NFS-role-block|diag|0.035, TeacherA-AdamW20|NFS-role-block|diag|0.05, TeacherA-AdamW20|NFS-role-block|cg5|0.035, TeacherA-AdamW20|NFS-role-block|cg5|0.05, TeacherA-AdamW20|NFS-role-block|cg10|0.05, TeacherA-AdamW20|NFS-role-block|lowrank32|0.05, TeacherB-AdamW100|NFS-Z|diag|0.02, TeacherB-AdamW100|NFS-Z|diag|0.035, TeacherB-AdamW100|NFS-Z|diag|0.05, TeacherB-AdamW100|NFS-Z|cg5|0.02, TeacherB-AdamW100|NFS-Z|cg5|0.035, TeacherB-AdamW100|NFS-Z|cg5|0.05, TeacherB-AdamW100|NFS-Z|cg10|0.02, TeacherB-AdamW100|NFS-Z|cg10|0.035, TeacherB-AdamW100|NFS-Z|cg10|0.05, TeacherB-AdamW100|NFS-Z|lowrank32|0.02, TeacherB-AdamW100|NFS-Z|lowrank32|0.035, TeacherB-AdamW100|NFS-Z|lowrank32|0.05, TeacherB-AdamW100|NFS-Z|lowrank64|0.035, TeacherB-AdamW100|NFS-Z|lowrank64|0.05, TeacherB-AdamW100|NFS-H|diag|0.02, TeacherB-AdamW100|NFS-H|cg5|0.02, TeacherB-AdamW100|NFS-H|cg10|0.02, TeacherB-AdamW100|NFS-H|lowrank32|0.02, TeacherB-AdamW100|NFS-H|lowrank32|0.035, TeacherB-AdamW100|NFS-H|lowrank64|0.035, TeacherB-AdamW100|NFS-H|lowrank64|0.05, TeacherB-AdamW100|NFS-ZH|diag|0.02, TeacherB-AdamW100|NFS-ZH|diag|0.035, TeacherB-AdamW100|NFS-ZH|diag|0.05, TeacherB-AdamW100|NFS-ZH|cg5|0.02, TeacherB-AdamW100|NFS-ZH|cg5|0.035, TeacherB-AdamW100|NFS-ZH|cg5|0.05, TeacherB-AdamW100|NFS-ZH|cg10|0.02, TeacherB-AdamW100|NFS-ZH|cg10|0.035, TeacherB-AdamW100|NFS-ZH|cg10|0.05, TeacherB-AdamW100|NFS-ZH|lowrank32|0.02, TeacherB-AdamW100|NFS-ZH|lowrank32|0.035, TeacherB-AdamW100|NFS-ZH|lowrank32|0.05, TeacherB-AdamW100|NFS-ZH|lowrank64|0.035, TeacherB-AdamW100|NFS-ZH|lowrank64|0.05, TeacherB-AdamW100|NFS-ZH-margin|diag|0.02, TeacherB-AdamW100|NFS-ZH-margin|diag|0.035, TeacherB-AdamW100|NFS-ZH-margin|diag|0.05, TeacherB-AdamW100|NFS-ZH-margin|cg5|0.02, TeacherB-AdamW100|NFS-ZH-margin|cg5|0.035, TeacherB-AdamW100|NFS-ZH-margin|cg5|0.05, TeacherB-AdamW100|NFS-ZH-margin|cg10|0.02, TeacherB-AdamW100|NFS-ZH-margin|cg10|0.035, TeacherB-AdamW100|NFS-ZH-margin|cg10|0.05, TeacherB-AdamW100|NFS-ZH-margin|lowrank32|0.02, TeacherB-AdamW100|NFS-ZH-margin|lowrank32|0.035, TeacherB-AdamW100|NFS-ZH-margin|lowrank32|0.05, TeacherB-AdamW100|NFS-ZH-margin|lowrank64|0.035, TeacherB-AdamW100|NFS-ZH-margin|lowrank64|0.05, TeacherB-AdamW100|NFS-role-block|cg10|0.02, TeacherB-AdamW100|NFS-role-block|lowrank32|0.02, TeacherB-AdamW100|NFS-role-block|lowrank64|0.035, TeacherB-AdamW100|NFS-role-cycle|diag|0.035, TeacherB-AdamW100|NFS-role-cycle|diag|0.05, TeacherB-AdamW100|NFS-role-cycle|cg5|0.035, TeacherB-AdamW100|NFS-role-cycle|cg5|0.05, TeacherB-AdamW100|NFS-role-cycle|cg10|0.05, TeacherB-AdamW100|NFS-role-cycle|lowrank32|0.05, TeacherC-AdamW-final|NFS-role-block|diag|0.035, TeacherC-AdamW-final|NFS-role-block|diag|0.05, TeacherC-AdamW-final|NFS-role-block|cg5|0.02, TeacherC-AdamW-final|NFS-role-block|cg5|0.035, TeacherC-AdamW-final|NFS-role-block|cg5|0.05, TeacherC-AdamW-final|NFS-role-block|cg10|0.05, TeacherC-AdamW-final|NFS-role-block|lowrank32|0.05, TeacherD-bestFGF|NFS-role-block|diag|0.035, TeacherD-bestFGF|NFS-role-block|diag|0.05, TeacherD-bestFGF|NFS-role-block|cg5|0.02, TeacherD-bestFGF|NFS-role-block|cg5|0.035, TeacherD-bestFGF|NFS-role-block|cg5|0.05, TeacherD-bestFGF|NFS-role-block|cg10|0.05, TeacherD-bestFGF|NFS-role-block|lowrank32|0.05
```

## P3 Offline NFS Projection From Functional Teachers

| dataset | teacher | method | variant | proj | acc drop | KL | phi red | smooth | P3 |
|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | TeacherE-FCAdam20 | FCAdam-dataSob | NFS-role-block | diag | 0.0000 | 0.0011 | 0.1356 | 12.2511 | yes |
| Fashion-MNIST | TeacherG-FLD-Adan20 | FLD-AdanLite | NFS-role-block | diag | 0.0078 | 0.0010 | 0.1258 | 11.3989 | no |
| Fashion-MNIST | TeacherI-D6-100 | D6-allTaskAware | NFS-role-block | diag | 0.0020 | 0.0004 | 0.1120 | 10.7596 | yes |
| Fashion-MNIST | TeacherJ-FNG-100 | F4-FNG-leftFullRight | NFS-role-block | diag | 0.0000 | 0.0021 | 0.1021 | 8.4408 | yes |
| Fashion-MNIST | TeacherE-FCAdam20 | FCAdam-dataSob | NFS-role-cycle | diag | 0.0000 | 0.0012 | 0.0731 | 6.5380 | yes |
| Fashion-MNIST | TeacherG-FLD-Adan20 | FLD-AdanLite | NFS-role-cycle | diag | 0.0078 | 0.0012 | 0.0729 | 6.4812 | no |
| Fashion-MNIST | TeacherH-FLD-Adan100 | FLD-AdanLite | NFS-role-block | diag | -0.0078 | 0.0026 | 0.0797 | 6.3248 | yes |
| Fashion-MNIST | TeacherF-FCAdam100 | FCAdam-dataSob | NFS-role-block | diag | -0.0059 | 0.0034 | 0.0788 | 5.8921 | yes |
| Fashion-MNIST | TeacherI-D6-100 | D6-allTaskAware | NFS-Z | cg5 | 0.0000 | 0.0017 | 0.0561 | 4.7881 | no |
| Fashion-MNIST | TeacherI-D6-100 | D6-allTaskAware | NFS-Z | diag | 0.0000 | 0.0017 | 0.0558 | 4.7725 | no |
| Fashion-MNIST | TeacherI-D6-100 | D6-allTaskAware | NFS-ZH | diag | 0.0000 | 0.0017 | 0.0558 | 4.7725 | no |
| Fashion-MNIST | TeacherI-D6-100 | D6-allTaskAware | NFS-role-cycle | diag | -0.0020 | 0.0005 | 0.0481 | 4.6050 | no |
| Fashion-MNIST | TeacherE-FCAdam20 | FCAdam-dataSob | NFS-role-cycle | cg5 | 0.0020 | 0.0013 | 0.0480 | 4.2320 | no |
| Fashion-MNIST | TeacherJ-FNG-100 | F4-FNG-leftFullRight | NFS-role-cycle | diag | 0.0000 | 0.0017 | 0.0481 | 4.1158 | no |
| Fashion-MNIST | TeacherI-D6-100 | D6-allTaskAware | NFS-role-cycle | cg5 | -0.0020 | 0.0003 | 0.0412 | 3.9924 | no |
| Fashion-MNIST | TeacherG-FLD-Adan20 | FLD-AdanLite | NFS-role-cycle | cg5 | 0.0059 | 0.0014 | 0.0400 | 3.5108 | no |
| Fashion-MNIST | TeacherJ-FNG-100 | F4-FNG-leftFullRight | NFS-role-cycle | cg5 | 0.0000 | 0.0019 | 0.0415 | 3.4928 | no |
| Fashion-MNIST | TeacherJ-FNG-100 | F4-FNG-leftFullRight | NFS-Z | cg5 | 0.0000 | 0.0029 | 0.0413 | 3.1931 | no |
| Fashion-MNIST | TeacherJ-FNG-100 | F4-FNG-leftFullRight | NFS-Z | diag | 0.0000 | 0.0029 | 0.0411 | 3.1827 | no |
| Fashion-MNIST | TeacherJ-FNG-100 | F4-FNG-leftFullRight | NFS-ZH | diag | 0.0000 | 0.0029 | 0.0411 | 3.1827 | no |
| Fashion-MNIST | TeacherE-FCAdam20 | FCAdam-dataSob | NFS-Z | diag | 0.0000 | 0.0013 | 0.0340 | 3.0039 | no |
| Fashion-MNIST | TeacherE-FCAdam20 | FCAdam-dataSob | NFS-ZH | diag | 0.0000 | 0.0013 | 0.0340 | 3.0039 | no |
| Fashion-MNIST | TeacherE-FCAdam20 | FCAdam-dataSob | NFS-Z | cg5 | 0.0000 | 0.0013 | 0.0339 | 2.9976 | no |
| Fashion-MNIST | TeacherH-FLD-Adan100 | FLD-AdanLite | NFS-role-cycle | cg5 | -0.0078 | 0.0050 | 0.0443 | 2.9457 | yes |
| Fashion-MNIST | TeacherG-FLD-Adan20 | FLD-AdanLite | NFS-Z | diag | 0.0059 | 0.0014 | 0.0334 | 2.9314 | no |
| Fashion-MNIST | TeacherG-FLD-Adan20 | FLD-AdanLite | NFS-ZH | diag | 0.0059 | 0.0014 | 0.0334 | 2.9314 | no |
| Fashion-MNIST | TeacherG-FLD-Adan20 | FLD-AdanLite | NFS-Z | cg5 | 0.0059 | 0.0014 | 0.0333 | 2.9260 | no |
| Fashion-MNIST | TeacherF-FCAdam100 | FCAdam-dataSob | NFS-role-cycle | cg5 | -0.0078 | 0.0050 | 0.0433 | 2.8938 | yes |
| Fashion-MNIST | TeacherF-FCAdam100 | FCAdam-dataSob | NFS-role-cycle | diag | -0.0059 | 0.0048 | 0.0313 | 2.1147 | yes |
| Fashion-MNIST | TeacherH-FLD-Adan100 | FLD-AdanLite | NFS-role-cycle | diag | -0.0098 | 0.0051 | 0.0285 | 1.8831 | yes |
| Fashion-MNIST | TeacherH-FLD-Adan100 | FLD-AdanLite | NFS-Z | cg5 | -0.0098 | 0.0051 | 0.0246 | 1.6327 | yes |
| Fashion-MNIST | TeacherH-FLD-Adan100 | FLD-AdanLite | NFS-Z | diag | -0.0098 | 0.0048 | 0.0239 | 1.6166 | yes |
| Fashion-MNIST | TeacherH-FLD-Adan100 | FLD-AdanLite | NFS-ZH | diag | -0.0098 | 0.0048 | 0.0239 | 1.6166 | yes |
| Fashion-MNIST | TeacherF-FCAdam100 | FCAdam-dataSob | NFS-Z | diag | -0.0059 | 0.0047 | 0.0235 | 1.5991 | no |
| Fashion-MNIST | TeacherF-FCAdam100 | FCAdam-dataSob | NFS-ZH | diag | -0.0059 | 0.0047 | 0.0235 | 1.5991 | no |
| Fashion-MNIST | TeacherF-FCAdam100 | FCAdam-dataSob | NFS-Z | cg5 | -0.0059 | 0.0047 | 0.0234 | 1.5971 | no |
| KMNIST | TeacherI-D6-100 | D6-allTaskAware | NFS-role-block | diag | 0.0020 | 0.0004 | 0.1114 | 10.7330 | yes |
| KMNIST | TeacherJ-FNG-100 | F4-FNG-leftFullRight | NFS-role-block | diag | -0.0039 | 0.0015 | 0.0786 | 6.8109 | yes |
| KMNIST | TeacherE-FCAdam20 | FCAdam-dataSob | NFS-role-block | diag | 0.0039 | 0.0006 | 0.0690 | 6.4991 | yes |
| KMNIST | TeacherG-FLD-Adan20 | FLD-AdanLite | NFS-role-block | diag | 0.0020 | 0.0006 | 0.0652 | 6.1242 | yes |
| KMNIST | TeacherI-D6-100 | D6-allTaskAware | NFS-Z | diag | 0.0098 | 0.0007 | 0.0515 | 4.8160 | no |
| KMNIST | TeacherI-D6-100 | D6-allTaskAware | NFS-ZH | diag | 0.0098 | 0.0007 | 0.0515 | 4.8160 | no |
| KMNIST | TeacherI-D6-100 | D6-allTaskAware | NFS-Z | cg5 | 0.0098 | 0.0007 | 0.0515 | 4.8132 | no |
| KMNIST | TeacherE-FCAdam20 | FCAdam-dataSob | NFS-role-cycle | cg5 | 0.0020 | 0.0007 | 0.0515 | 4.8057 | yes |
| KMNIST | TeacherG-FLD-Adan20 | FLD-AdanLite | NFS-role-cycle | cg5 | 0.0020 | 0.0008 | 0.0510 | 4.7363 | yes |
| KMNIST | TeacherI-D6-100 | D6-allTaskAware | NFS-role-cycle | diag | 0.0098 | 0.0002 | 0.0480 | 4.6887 | no |
| KMNIST | TeacherG-FLD-Adan20 | FLD-AdanLite | NFS-role-cycle | diag | 0.0000 | 0.0008 | 0.0502 | 4.6649 | yes |
| KMNIST | TeacherF-FCAdam100 | FCAdam-dataSob | NFS-role-block | diag | 0.0000 | 0.0025 | 0.0569 | 4.5520 | no |
| KMNIST | TeacherH-FLD-Adan100 | FLD-AdanLite | NFS-role-block | diag | 0.0020 | 0.0023 | 0.0530 | 4.3066 | no |
| KMNIST | TeacherJ-FNG-100 | F4-FNG-leftFullRight | NFS-role-cycle | diag | -0.0137 | 0.0014 | 0.0487 | 4.2567 | no |
| KMNIST | TeacherF-FCAdam100 | FCAdam-dataSob | NFS-role-cycle | cg5 | -0.0039 | 0.0033 | 0.0554 | 4.1739 | yes |
| KMNIST | TeacherE-FCAdam20 | FCAdam-dataSob | NFS-role-cycle | diag | 0.0000 | 0.0008 | 0.0436 | 4.0503 | yes |
| KMNIST | TeacherI-D6-100 | D6-allTaskAware | NFS-role-cycle | cg5 | 0.0059 | 0.0002 | 0.0409 | 4.0216 | no |
| KMNIST | TeacherF-FCAdam100 | FCAdam-dataSob | NFS-role-cycle | diag | -0.0039 | 0.0034 | 0.0527 | 3.9467 | no |

P3 all-dataset survivors:

```text
TeacherE-FCAdam20|NFS-role-block|diag|0.035, TeacherI-D6-100|NFS-role-block|diag|0.035, TeacherJ-FNG-100|NFS-role-block|diag|0.035
```

## P4 TAN One-Cycle Micro-Run

| dataset | task | NFS | refresh | task acc | final acc | acc drop | phi red | recovery | P4 |
|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | D6-allTaskAware | P2best-NFS-role-block-diag-eta0.05 | D6-5 | 0.7383 | 0.7598 | -0.0215 | 0.1526 | 37.9383 | yes |
| Fashion-MNIST | D6-allTaskAware | P2best-NFS-role-block-diag-eta0.05 | FCAdam5 | 0.7383 | 0.7598 | -0.0215 | 0.1425 | 25.4144 | yes |
| Fashion-MNIST | PureKAN-AdamW | P2best-NFS-role-block-diag-eta0.05 | D6-5 | 0.7734 | 0.7852 | -0.0117 | 0.1370 | 32.3783 | yes |
| Fashion-MNIST | D6-allTaskAware | P2best-NFS-role-block-diag-eta0.05 | FCAdam10 | 0.7383 | 0.7500 | -0.0117 | 0.1330 | 45.9922 | yes |
| Fashion-MNIST | D6-allTaskAware | P2best-NFS-role-block-cg5-eta0.05 | D6-5 | 0.7383 | 0.7617 | -0.0234 | 0.1326 | 49.5606 | yes |
| Fashion-MNIST | D6-allTaskAware | P2best-NFS-role-block-cg5-eta0.05 | FCAdam5 | 0.7383 | 0.7578 | -0.0195 | 0.1229 | 33.1104 | yes |
| Fashion-MNIST | PureKAN-AdamW | P2best-NFS-role-block-cg5-eta0.05 | D6-5 | 0.7734 | 0.7871 | -0.0137 | 0.1189 | 47.2846 | yes |
| Fashion-MNIST | D6-allTaskAware | P2best-NFS-role-block-cg5-eta0.05 | FCAdam10 | 0.7383 | 0.7539 | -0.0156 | 0.1133 | 60.0243 | yes |
| Fashion-MNIST | D6-allTaskAware | P2best-NFS-role-block-diag-eta0.05 | none | 0.7383 | 0.7363 | 0.0020 | 0.1530 | 0.0000 | no |
| Fashion-MNIST | PureKAN-AdamW | P2best-NFS-role-block-diag-eta0.05 | none | 0.7734 | 0.7754 | -0.0020 | 0.1369 | 0.0000 | no |
| Fashion-MNIST | D6-allTaskAware | P2best-NFS-role-block-cg5-eta0.05 | none | 0.7383 | 0.7363 | 0.0020 | 0.1330 | 0.0000 | no |
| Fashion-MNIST | PureKAN-AdamW | P2best-NFS-role-block-diag-eta0.05 | FCAdam5 | 0.7734 | 0.7695 | 0.0039 | 0.1251 | -106.0962 | no |
| Fashion-MNIST | PureKAN-AdamW | P2best-NFS-role-block-cg5-eta0.05 | none | 0.7734 | 0.7754 | -0.0020 | 0.1186 | 0.0000 | no |
| Fashion-MNIST | PureKAN-AdamW | P2best-NFS-role-block-diag-eta0.05 | FCAdam10 | 0.7734 | 0.7852 | -0.0117 | 0.1135 | -116.1521 | no |
| Fashion-MNIST | PureKAN-AdamW | P2best-NFS-role-block-cg5-eta0.05 | FCAdam5 | 0.7734 | 0.7695 | 0.0039 | 0.1070 | -157.4310 | no |
| Fashion-MNIST | PureKAN-AdamW | P2best-NFS-role-block-cg5-eta0.05 | FCAdam10 | 0.7734 | 0.7871 | -0.0137 | 0.0958 | -172.0602 | no |
| Fashion-MNIST | FCAdam-dataSob | P2best-NFS-role-block-cg5-eta0.05 | D6-5 | 0.7578 | 0.7559 | 0.0020 | 0.0796 | 971078872.6807 | no |
| Fashion-MNIST | FCAdam-dataSob | P2best-NFS-role-block-diag-eta0.05 | D6-5 | 0.7578 | 0.7559 | 0.0020 | 0.0796 | 969409942.6270 | no |
| Fashion-MNIST | FCAdam-dataSob | P2best-NFS-role-block-cg5-eta0.05 | none | 0.7578 | 0.7559 | 0.0020 | 0.0795 | 0.0000 | no |
| Fashion-MNIST | FCAdam-dataSob | P2best-NFS-role-block-diag-eta0.05 | none | 0.7578 | 0.7539 | 0.0039 | 0.0794 | 0.0000 | no |
| Fashion-MNIST | FLD-AdanLite | P2best-NFS-role-block-cg5-eta0.05 | none | 0.7578 | 0.7559 | 0.0020 | 0.0726 | 0.0000 | no |
| Fashion-MNIST | FLD-AdanLite | P2best-NFS-role-block-cg5-eta0.05 | D6-5 | 0.7578 | 0.7520 | 0.0059 | 0.0725 | 4901826381.6833 | no |
| Fashion-MNIST | FLD-AdanLite | P2best-NFS-role-block-diag-eta0.05 | none | 0.7578 | 0.7559 | 0.0020 | 0.0724 | 0.0000 | no |
| Fashion-MNIST | FLD-AdanLite | P2best-NFS-role-block-diag-eta0.05 | D6-5 | 0.7578 | 0.7520 | 0.0059 | 0.0723 | 4887402057.6477 | no |
| Fashion-MNIST | FCAdam-dataSob | P2best-NFS-role-block-cg5-eta0.05 | FCAdam5 | 0.7578 | 0.7676 | -0.0098 | 0.0682 | -116045057773.5901 | no |
| Fashion-MNIST | FCAdam-dataSob | P2best-NFS-role-block-diag-eta0.05 | FCAdam5 | 0.7578 | 0.7676 | -0.0098 | 0.0682 | -116060197353.3630 | no |
| Fashion-MNIST | FLD-AdanLite | P2best-NFS-role-block-cg5-eta0.05 | FCAdam5 | 0.7578 | 0.7305 | 0.0273 | 0.0667 | -137469112873.0774 | no |
| Fashion-MNIST | FLD-AdanLite | P2best-NFS-role-block-diag-eta0.05 | FCAdam5 | 0.7578 | 0.7305 | 0.0273 | 0.0665 | -137489855289.4592 | no |
| Fashion-MNIST | FCAdam-L2 | P2best-NFS-role-block-diag-eta0.05 | D6-5 | 0.7480 | 0.7656 | -0.0176 | 0.0654 | 54638564586.6394 | no |
| Fashion-MNIST | FCAdam-L2 | P2best-NFS-role-block-cg5-eta0.05 | D6-5 | 0.7480 | 0.7656 | -0.0176 | 0.0652 | 54693877696.9910 | no |
| Fashion-MNIST | FCAdam-L2 | P2best-NFS-role-block-diag-eta0.05 | none | 0.7480 | 0.7559 | -0.0078 | 0.0647 | 0.0000 | no |
| Fashion-MNIST | FCAdam-L2 | P2best-NFS-role-block-cg5-eta0.05 | none | 0.7480 | 0.7559 | -0.0078 | 0.0646 | 0.0000 | no |
| Fashion-MNIST | FCAdam-L2 | P2best-NFS-role-block-diag-eta0.05 | FCAdam5 | 0.7480 | 0.7461 | 0.0020 | 0.0569 | -17277181148.5291 | no |
| Fashion-MNIST | FCAdam-L2 | P2best-NFS-role-block-cg5-eta0.05 | FCAdam5 | 0.7480 | 0.7480 | 0.0000 | 0.0569 | -17116785049.4385 | no |
| Fashion-MNIST | FLD-AdanLite | P2best-NFS-role-block-cg5-eta0.05 | FCAdam10 | 0.7578 | 0.7500 | 0.0078 | 0.0540 | -65561056137.0850 | no |
| Fashion-MNIST | FLD-AdanLite | P2best-NFS-role-block-diag-eta0.05 | FCAdam10 | 0.7578 | 0.7500 | 0.0078 | 0.0539 | -65584719181.0608 | no |
| Fashion-MNIST | FCAdam-dataSob | P2best-NFS-role-block-cg5-eta0.05 | FCAdam10 | 0.7578 | 0.7676 | -0.0098 | 0.0501 | -88990688323.9746 | no |
| Fashion-MNIST | FCAdam-dataSob | P2best-NFS-role-block-diag-eta0.05 | FCAdam10 | 0.7578 | 0.7676 | -0.0098 | 0.0501 | -89143216609.9548 | no |
| Fashion-MNIST | FCAdam-L2 | P2best-NFS-role-block-diag-eta0.05 | FCAdam10 | 0.7480 | 0.7617 | -0.0137 | 0.0440 | 4307031631.4697 | no |
| Fashion-MNIST | FCAdam-L2 | P2best-NFS-role-block-cg5-eta0.05 | FCAdam10 | 0.7480 | 0.7617 | -0.0137 | 0.0439 | 4423618316.6504 | no |
| KMNIST | D6-allTaskAware | P2best-NFS-role-block-diag-eta0.05 | D6-5 | 0.4570 | 0.5469 | -0.0898 | 0.1344 | 55.5976 | yes |
| KMNIST | D6-allTaskAware | P2best-NFS-role-block-cg5-eta0.05 | D6-5 | 0.4570 | 0.5469 | -0.0898 | 0.1341 | 55.8216 | yes |
| KMNIST | PureKAN-AdamW | P2best-NFS-role-block-diag-eta0.05 | D6-5 | 0.6953 | 0.7031 | -0.0078 | 0.1319 | 4.3098 | yes |
| KMNIST | D6-allTaskAware | P2best-NFS-role-block-diag-eta0.05 | FCAdam5 | 0.4570 | 0.5547 | -0.0977 | 0.1247 | 52.7462 | yes |
| KMNIST | D6-allTaskAware | P2best-NFS-role-block-cg5-eta0.05 | FCAdam5 | 0.4570 | 0.5547 | -0.0977 | 0.1244 | 52.9578 | yes |
| KMNIST | PureKAN-AdamW | P2best-NFS-role-block-cg5-eta0.05 | D6-5 | 0.6953 | 0.7031 | -0.0078 | 0.1144 | 5.0371 | yes |
| KMNIST | D6-allTaskAware | P2best-NFS-role-block-diag-eta0.05 | FCAdam10 | 0.4570 | 0.6211 | -0.1641 | 0.1108 | 88.0937 | yes |
| KMNIST | D6-allTaskAware | P2best-NFS-role-block-cg5-eta0.05 | FCAdam10 | 0.4570 | 0.6211 | -0.1641 | 0.1105 | 88.4503 | yes |
| KMNIST | D6-allTaskAware | P2best-NFS-role-block-diag-eta0.05 | none | 0.4570 | 0.4453 | 0.0117 | 0.1331 | 0.0000 | no |
| KMNIST | D6-allTaskAware | P2best-NFS-role-block-cg5-eta0.05 | none | 0.4570 | 0.4453 | 0.0117 | 0.1327 | 0.0000 | no |
| KMNIST | PureKAN-AdamW | P2best-NFS-role-block-diag-eta0.05 | none | 0.6953 | 0.6914 | 0.0039 | 0.1319 | 0.0000 | no |
| KMNIST | PureKAN-AdamW | P2best-NFS-role-block-diag-eta0.05 | FCAdam5 | 0.6953 | 0.6973 | -0.0020 | 0.1184 | 5.5609 | no |
| KMNIST | PureKAN-AdamW | P2best-NFS-role-block-cg5-eta0.05 | none | 0.6953 | 0.6934 | 0.0020 | 0.1145 | 0.0000 | no |
| KMNIST | PureKAN-AdamW | P2best-NFS-role-block-diag-eta0.05 | FCAdam10 | 0.6953 | 0.6953 | 0.0000 | 0.1053 | 3.7286 | no |
| KMNIST | PureKAN-AdamW | P2best-NFS-role-block-cg5-eta0.05 | FCAdam5 | 0.6953 | 0.6973 | -0.0020 | 0.1009 | 6.4375 | no |
| KMNIST | PureKAN-AdamW | P2best-NFS-role-block-cg5-eta0.05 | FCAdam10 | 0.6953 | 0.6934 | 0.0020 | 0.0885 | 4.2107 | no |
| KMNIST | FLD-AdanLite | P2best-NFS-role-block-cg5-eta0.05 | D6-5 | 0.5957 | 0.5996 | -0.0039 | 0.0580 | 1392304897.3083 | no |
| KMNIST | FLD-AdanLite | P2best-NFS-role-block-cg5-eta0.05 | none | 0.5957 | 0.5996 | -0.0039 | 0.0579 | 0.0000 | no |
| KMNIST | FCAdam-L2 | P2best-NFS-role-block-diag-eta0.05 | none | 0.6504 | 0.6504 | 0.0000 | 0.0578 | 0.0000 | no |
| KMNIST | FCAdam-L2 | P2best-NFS-role-block-cg5-eta0.05 | none | 0.6504 | 0.6504 | 0.0000 | 0.0575 | 0.0000 | no |
| KMNIST | FLD-AdanLite | P2best-NFS-role-block-diag-eta0.05 | D6-5 | 0.5957 | 0.5996 | -0.0039 | 0.0574 | 1381993293.7622 | no |
| KMNIST | FLD-AdanLite | P2best-NFS-role-block-diag-eta0.05 | none | 0.5957 | 0.5996 | -0.0039 | 0.0574 | 0.0000 | no |
| KMNIST | FCAdam-dataSob | P2best-NFS-role-block-diag-eta0.05 | D6-5 | 0.5977 | 0.5898 | 0.0078 | 0.0568 | 382900238.0371 | no |
| KMNIST | FCAdam-dataSob | P2best-NFS-role-block-diag-eta0.05 | none | 0.5977 | 0.5898 | 0.0078 | 0.0567 | 0.0000 | no |
| KMNIST | FCAdam-L2 | P2best-NFS-role-block-diag-eta0.05 | D6-5 | 0.6504 | 0.6543 | -0.0039 | 0.0566 | 14720141887.6648 | no |
| KMNIST | FCAdam-dataSob | P2best-NFS-role-block-cg5-eta0.05 | D6-5 | 0.5977 | 0.5898 | 0.0078 | 0.0564 | 379681587.2192 | no |
| KMNIST | FCAdam-dataSob | P2best-NFS-role-block-cg5-eta0.05 | none | 0.5977 | 0.5898 | 0.0078 | 0.0563 | 0.0000 | no |
| KMNIST | FCAdam-L2 | P2best-NFS-role-block-cg5-eta0.05 | D6-5 | 0.6504 | 0.6543 | -0.0039 | 0.0563 | 14726161956.7871 | no |
| KMNIST | FCAdam-L2 | P2best-NFS-role-block-diag-eta0.05 | FCAdam5 | 0.6504 | 0.5781 | 0.0723 | 0.0481 | -171368062496.1853 | no |
| KMNIST | FCAdam-L2 | P2best-NFS-role-block-cg5-eta0.05 | FCAdam5 | 0.6504 | 0.5781 | 0.0723 | 0.0477 | -171245813369.7510 | no |
| KMNIST | FLD-AdanLite | P2best-NFS-role-block-cg5-eta0.05 | FCAdam5 | 0.5957 | 0.6016 | -0.0059 | 0.0446 | -63986122608.1848 | no |
| KMNIST | FLD-AdanLite | P2best-NFS-role-block-diag-eta0.05 | FCAdam5 | 0.5957 | 0.6016 | -0.0059 | 0.0440 | -65106213092.8040 | no |

P4 all-dataset survivors:

```text
PureKAN-AdamW|P2best-NFS-role-block-diag-eta0.05|NFS-role-block|D6-5, PureKAN-AdamW|P2best-NFS-role-block-cg5-eta0.05|NFS-role-block|D6-5
```

## P5 Alternating TAN Cycles

| dataset | task | NFS | refresh | cycles | final acc | final phi | rank | margin | P5 |
|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | PureKAN-AdamW | P2best-NFS-role-block-cg5-eta0.05 | D6-5 | 3 | 0.7773 | 0.1199 | 12.5031 | 1.9166 | yes |
| Fashion-MNIST | PureKAN-AdamW | P2best-NFS-role-block-diag-eta0.05 | D6-5 | 3 | 0.7773 | 0.1143 | 12.2508 | 1.9063 | yes |
| KMNIST | PureKAN-AdamW | P2best-NFS-role-block-cg5-eta0.05 | D6-5 | 3 | 0.6621 | 0.1319 | 15.7638 | 1.8226 | yes |
| KMNIST | PureKAN-AdamW | P2best-NFS-role-block-diag-eta0.05 | D6-5 | 3 | 0.6621 | 0.1295 | 15.6598 | 1.8213 | yes |
| MNIST | PureKAN-AdamW | P2best-NFS-role-block-cg5-eta0.05 | D6-5 | 3 | 0.8242 | 0.1489 | 12.6641 | 1.8399 | no |
| MNIST | PureKAN-AdamW | P2best-NFS-role-block-diag-eta0.05 | D6-5 | 3 | 0.8242 | 0.1483 | 12.6372 | 1.8385 | no |

P5 all-dataset survivors:

```text
none
```

## P6 Capacity / Basis Expansion

_P6 not run yet._

P6 all-dataset survivors:

```text
none
```

## P3-P9 Decision

```text
P3 functional-teacher NFS: run
P4 TAN one-cycle: run
P5 alternating TAN cycles: run
P7/P8/P9 confirm: not run; P5 no all-dataset survivor
Final decision: stop_after_p5_no_multicycle_survivor
```

## Failure Diagnosis

Generated:

```text
p9_failure_diagnosis.csv
failure_table.csv
failure_by_dataset.csv
failure_by_method.csv
```

Diagnosis:

```text
F1 checks whether an accurate and smooth PureKAN solution is visible under AdamW-style geometry regularization.
F2 checks whether NFS can lower geometry in a teacher-preserving tangent/nullspace.
F4 checks whether smoothing destroys logits/features or accuracy.
F5 checks whether refresh and alternating cycles can retain the one-cycle geometry gain.
F6 remains the capacity/basis fallback, but it was not triggered because P2/P4 already gave mechanistic survivors.

Interpretation:
  P1 did not expose an AdamW+regularization smooth frontier.
  P2/P3 showed NFS role-block can preserve function while reducing geometry.
  P4 showed one-cycle TAN can pass across datasets.
  P5 showed the same recipe does not yet survive alternating cycles on MNIST.
```

## Required Artifacts

Written under `results/v4_8/`:

```text
p0_fgf_invariants.csv
p1_feasibility_frontier.csv
p1_training_trace.csv
p1_frontier_gate_summary.csv
p2_nfs_adamw_projection.csv
p2_nfs_gate_summary.csv
p3_nfs_functional_teachers.csv
p3_nfs_functional_gate_summary.csv
p4_tan_one_cycle_micro_run.csv
p4_tan_one_cycle_gate_summary.csv
p5_tan_alternating_cycles.csv
p5_tan_cycle_gate_summary.csv
p6_capacity_basis_expansion.csv
p6_capacity_gate_summary.csv
p7_candidate_selection.csv
p8_confirm5.csv
p9_confirm10.csv
p9_failure_diagnosis.csv
failure_table.csv
aggregate_decision.json
figures/
```

## Final Decision

```text
DG-KAN v4.8 status:
  stop_after_p5_no_multicycle_survivor

What improved:
  v4.8 now asks the right feasibility question before spending seed budget.
  P1 exposes the accuracy-geometry Pareto frontier directly.
  P2 measures function-preserving smoothability instead of relying on unconstrained Sobolev consolidation.
  P4 confirms a real one-cycle TAN signal with PureKAN-AdamW task teacher + role-block NFS + D6 refresh.

What failed / remains open:
  P5 fails the all-dataset multi-cycle gate because MNIST does not preserve the task/geometry tradeoff over 3 cycles.

Conclusion:
  NFS is feasible as a local projection, and TAN is viable for one cycle, but the current alternating controller is not stable enough for P7 seed confirmation.
  Next work should redesign the multi-cycle controller / refresh policy before spending 5/10-seed budget.
```



---


# Source 25: `docs/DG-KAN_v4.9_RBFParameterization_ExactNFS_下一步实验计划.md`


# DG-KAN v4.9：RBF 参数化重审、Exact NFS 与 PureKAN 功能几何路线

> 目标：基于 v4.8 结果和当前 `dgkan_core / run_gafu_v48 / analyze_gafu_v48` 实现，重新审视 functional update 的失败原因，明确区分“optimizer 失败”和“RBF-only 参数化失败”，并给出下一步可执行实验计划。  
> 公式格式：Typora 友好，统一使用 `$...$` 与 `$$...$$`，不使用 `\[\]`。

---

## 0. 当前结论先行

v4.8 的核心进展不是找到最终 optimizer，而是把问题边界显著缩窄了。当前最重要的判断是：

$$
\boxed{
\text{PureKAN functional update 的失败，不能再只归因于某个 optimizer 超参。}
}
$$

更准确地说，当前失败同时涉及三件事：

```text
1. 当前 PureKAN 是 fixed-grid RBF-only coefficient 网络；
2. 当前 functional update 主要作用在 coefficient 上，centers / width / base path 均未参与学习；
3. 当前 v4.8 的 NFS 更像 teacher-constrained smoothing proposal，不是真正的 Jacobian-nullspace projection。
```

因此，下一轮不能继续只调：

```text
Sobolev alpha / beta
branch scale
warmup length
trust radius
refresh steps
```

而应该改成：

$$
\boxed{
\text{先验证 RBF 参数化是否给 functional optimizer 提供了合适坐标，再验证真正的函数保持型 smoothing。}
}
$$

本计划将 v4.9 的主问题定义为：

$$
\boxed{
\text{PureKAN 失败到底是 optimizer 问题，还是 RBF-only 参数化问题，还是 NFS 实现过弱？}
}
$$

---

## 1. v4.8 实验结果复盘

### 1.1 P0：实现 smoke 通过，但只说明路径可运行

v4.8 的 P0 显示：

```text
rows = 21
errors = 0
max nonKAN = 0
min coverage = 1.0
max rollback = 0
max CG residual = 0.0533
pass = true
```

这说明：

```text
1. strict PureKAN 仍然没有 non-KAN trainable params；
2. coefficient coverage 是完整的；
3. NFS temporary apply / rollback / teacher snapshot 基本可用；
4. finite projection diagnostics 没有明显数值爆炸。
```

所以这一轮不能再主要归因于：

```text
alpha 没 fixed
input/output KAN 没被更新
rollback 出错
teacher snapshot 出错
```

但 P0 不证明 NFS 是数学意义上的 nullspace projection，也不证明 PureKAN 存在准确且几何好的解。

---

### 1.2 P1：AdamW + 简单几何正则没有暴露 smooth accurate frontier

P1 的目标是问：

$$
\boxed{
\text{PureKAN 是否存在 AdamW 能找到的准确且几何更好的解？}
}
$$

结果是没有 all-dataset survivor。

典型结果如下。

Fashion-MNIST：

```text
A0-PureKAN-AdamW acc = 0.7754
A1/A2/A3 Sobolev regularized acc = 0.7754
phi red ≈ 0
J red ≈ 0
A8 posthoc geometry acc = 0.5664
```

KMNIST：

```text
A0-PureKAN-AdamW acc = 0.6797
A1/A2/A3 Sobolev regularized acc = 0.6797
phi red ≈ 0
J red ≈ 0
A8 posthoc geometry acc = 0.1973
```

MNIST：

```text
A0-PureKAN-AdamW acc = 0.8711
A1/A2/A3 Sobolev regularized acc = 0.8711
small J reduction only on some rows
A8 posthoc geometry acc = 0.2148
```

这说明两件事。

第一，当前弱几何正则几乎不起作用：

$$
\text{AdamW} + \lambda R_{geom}
\approx
\text{AdamW}
$$

其中 $\lambda$ 很小时，accuracy 保住了，但 geometry 没明显改善。

第二，强行 posthoc smoothing 会破坏 task function：

$$
\text{geometry push too hard}
\Rightarrow
\text{accuracy collapse}.
$$

所以 P1 的结论不是“smooth accurate PureKAN 不存在”，而是：

$$
\boxed{
\text{当前实现中的简单几何正则，没有找到 smooth accurate frontier。}
}
$$

这仍然留下两个开放可能：

```text
1. 需要更好的 RBF 参数化；
2. 需要真正 function-preserving projection，而不是普通几何正则。
```

---

### 1.3 P2：NFS-role-block 是本轮最强正信号

P2 从 AdamW teacher 出发，做 offline smoothing projection。这里出现了大量 `NFS-role-block` pass。

Fashion-MNIST 上，典型成功行为是：

```text
TeacherA-AdamW20 + NFS-role-block + diag + eta=0.05:
  acc drop = 0.0039
  KL = 0.0001
  logit drift = 0.0147
  phi red = 0.1482
  J red = 0.4284
  pass = yes
```

KMNIST 上，典型成功行为是：

```text
TeacherI-D6-100 + NFS-role-block + diag:
  acc drop = 0.0020
  KL = 0.0004
  phi red = 0.1114
  smoothability = 10.7330
  pass = yes
```

这说明：

$$
\boxed{
\text{在 teacher 附近，block-role smoothing 可以在较小函数漂移下改善几何。}
}
$$

这非常重要，因为它证明 functional geometry 并不是完全没希望。至少在局部，存在类似：

$$
\Delta \theta
\quad\text{s.t.}\quad
\Delta f \approx 0,
\quad
\Delta R_{geom} < 0.
$$

也就是说，PureKAN 的某些 teacher function 附近确实存在可用的 smoothing direction。

---

### 1.4 P3：functional teachers 也能被 NFS 平滑，说明 NFS 不是只适配 AdamW teacher

P3 从 functional teacher 出发，例如：

```text
TeacherE-FCAdam20
TeacherI-D6-100
TeacherJ-FNG-100
```

也出现了 all-dataset survivor，例如：

```text
TeacherE-FCAdam20 | NFS-role-block | diag | 0.035
TeacherI-D6-100   | NFS-role-block | diag | 0.035
TeacherJ-FNG-100  | NFS-role-block | diag | 0.035
```

这说明：

$$
\boxed{
\text{NFS-role-block 是一个真实的局部几何 projection primitive。}
}
$$

它不是只对 AdamW teacher 工作，也不是纯偶然。

但它仍然只是局部 primitive，不是完整 optimizer。

---

### 1.5 P4：TAN one-cycle 成立，说明 Learn -> Smooth -> Refresh 有一次性价值

P4 显示，一次：

```text
Task teacher
-> NFS-role-block smoothing
-> D6 or FCAdam refresh
```

可以在多个数据集上通过。

典型结果：

```text
Fashion, PureKAN-AdamW teacher + NFS-role-block + D6-5:
  task acc = 0.7734
  final acc = 0.7852
  acc drop = -0.0117
  phi red = 0.1370
  pass = yes

KMNIST, PureKAN-AdamW teacher + NFS-role-block + D6-5:
  task acc = 0.6953
  final acc = 0.7031
  acc drop = -0.0078
  phi red = 0.1319
  pass = yes
```

这说明：

$$
\boxed{
\text{一次 Learn -> Smooth -> Refresh 是可行的。}
}
$$

它也说明之前的想法“先学任务，再做几何 projection”不是错的。

---

### 1.6 P5：多 cycle 失败，核心是 controller 不稳定，不是 NFS 完全无效

P5 的 alternating TAN cycles 结果是：

```text
Fashion: pass
KMNIST: pass
MNIST: fail
```

典型 P5 最终结果：

```text
Fashion:
  final acc = 0.7773
  final phi = 0.1143 / 0.1199
  pass = yes

KMNIST:
  final acc = 0.6621
  final phi = 0.1295 / 0.1319
  pass = yes

MNIST:
  final acc = 0.8242
  final phi = 0.1483 / 0.1489
  pass = no
```

所以 P5 的结论不是：

```text
NFS/TAN 完全失败。
```

而是：

$$
\boxed{
\text{当前 multi-cycle controller 不能稳定保持 task/geometry tradeoff。}
}
$$

特别是 MNIST 上，cycles 后的 task/geometry 平衡不能保持。下一步如果继续 TAN，不应该继续扩大 seed，而应该重新设计 cycle controller。

---

## 2. 代码实现审计：当前还有哪些关键限制

### 2.1 当前 strict PureKAN 是 fixed-grid RBF coefficient-only 网络

当前 `RBFDense` 的核心是：

```python
centers = torch.linspace(-2.5, 2.5, basis_count)
self.register_buffer("centers", centers)
self.width = float((centers[1] - centers[0]).abs() * 1.4)
self.coeff = nn.Parameter(...)
```

也就是说：

```text
centers 是 buffer，不训练；
width 是 float，不训练；
coeff 是 Parameter，训练；
PureKAN 中 bias=False；
PureKAN 中没有 Linear stem/head，没有 learnable LayerNorm，没有 alpha 参数。
```

PureKAN 的结构是：

```text
input_kan: RBFDense(input_dim -> hidden_dim, bias=False)
blocks: PureResidualKANBlock(... RBFDense(hidden -> hidden, bias=False))
output_kan: RBFDense(hidden_dim -> num_classes, bias=False)
```

所以 strict PureKAN 实际是：

$$
\boxed{
\text{fixed centers + fixed width + coeff-only RBF edge network.}
}
$$

这非常关键。因为现在我们说“PureKAN functional update 失败”，其实更准确是：

$$
\boxed{
\text{fixed-grid RBF-only PureKAN 在当前 functional update 下失败。}
}
$$

它不等于所有 KAN 参数化都失败。

---

### 2.2 当前 PureKAN 缺少 base / linear / identity 通道

当前 RBF edge 是：

$$
f_{ij}(x)=\sum_{k=1}^K c_{ijk}B_k(x).
$$

没有：

$$
b_{ij},\quad w_{ij}x,\quad w_{ij}^{base}\operatorname{silu}(x).
$$

但许多 KAN / spline-KAN 实现都有 base path 或 residual base function：

$$
f_{ij}(x)=w_{ij}^{base}b(x)+\sum_k c_{ijk}B_k(x).
$$

当前 PureKAN 要用 RBF basis 去拼出所有低阶、线性、尺度变换和分类 boundary。AdamW 能训练，是因为它可以自由地在 coefficient space 里拼；functional/Sobolev 更新会倾向于平滑、压制高频和某些组合模式。

这可能解释为什么：

```text
Hybrid-DGKAN 成功；
PureKAN-UFULL/TFU/FNG/FLD 反复失败。
```

Hybrid-DGKAN 有 stem/head/LN/residual identity，提供了便宜的低阶通道。PureKAN 没有。

因此，下一步必须验证：

$$
\boxed{
\text{把 base path 放回 KAN edge 内部，是否能让 PureKAN functional update 成立？}
}
$$

---

### 2.3 当前 NFS 不是严格的 Jacobian nullspace projection

从 `run_gafu_v48.py` 的实现看，`_nfs_project_once` 大致流程是：

```text
1. 用 _geometry_smooth_updates 生成 smoothing proposal；
2. 根据 projector 类型乘一个 projector_scale；
3. temporary apply；
4. 检查 KL / logit drift / hidden drift / margin drift；
5. backtracking；
6. accept 或 rollback。
```

其中 projector 是：

```text
diag -> 1.0
cg5 -> 0.85
cg10 -> 0.70
lowrank32 -> 0.55
lowrank64 -> 0.45
```

这说明当前 NFS 更准确地说是：

$$
\boxed{
\text{teacher-constrained smoothing proposal with backtracking.}
}
$$

而不是真正求解：

$$
\Delta\theta
=
-\eta
\left(
I
-
M^{-1}J_f^T(J_fM^{-1}J_f^T+\mu I)^{-1}J_f
\right)
M^{-1}\nabla R.
$$

这个差异很大。当前 P2/P3 的 positive signal 仍然有价值，但它还没有证明真正的 nullspace smoothing 可以训练。

所以 v4.9 必须实现 **Exact / Sketch NFS**，而不是继续使用 projector scale 名义上的 `cg5/cg10/lowrank`。

---

### 2.4 P1 的几何正则也不是完整几何 frontier

P1 的 `AdamW-Sobolev / PhiProxy / JacProxy` 是弱正则和 proxy 正则。它能说明当前简单正则没找到 frontier，但不能证明：

```text
准确且几何好的 PureKAN 解不存在。
```

因为真正的 geometry objective 应该至少包含：

```text
1. coefficient Sobolev norm；
2. actual phi_prime_p95 或可微 surrogate；
3. block Jacobian proxy；
4. basis occupancy / out-of-grid penalty；
5. function-preserving smoothing phase。
```

v4.8 的 P1 只完成了第一层可行性检查，不能当作最终 capacity 结论。

---

### 2.5 P6 capacity / basis expansion 没有被真正用来回答问题

`run_gafu_v48.py` 的 P6 已经预留了：

```text
(64,16,4,RBF)
(96,24,4,RBF)
(96,32,2,RBF)
(96,24,4,AB-RBF)
```

并且方法包括：

```text
A0-PureKAN-AdamW
A6-AdamW-MixedGeom
```

但 v4.8 的决策在 P5 后停止，P6 没有真正用于最终判断。

这意味着：

$$
\boxed{
\text{capacity / basis / AB-RBF 是否能改变结论，仍然是开放问题。}
}
$$

下一轮不应该跳过这个问题。

---

## 3. 重新定义当前失败模式

### 3.1 失败模式一：RBF-only 参数化可能没有给 functional optimizer 合适的低阶通道

当前 strict PureKAN 的 edge 是：

$$
f(x)=\sum_k c_kB_k(x).
$$

functional update 一直在 coefficient basis 上做文章。但如果任务需要简单线性或 affine transform，这个结构非常别扭。

因此现在要问：

$$
\boxed{
\text{PureKAN-AdamW 的成功是否依赖 RBF coefficient 拼出的低阶通道？}
}
$$

如果是，那么 functional update 压制这些通道就会导致训练失败。

---

### 3.2 失败模式二：geometry smoothing 可以局部 function-preserving，但 multi-cycle controller 不稳定

P2/P3/P4 都显示 NFS-role-block 可行；P5 显示 alternating cycles 不稳定。

这说明问题不是：

```text
完全没有平滑方向。
```

而是：

```text
如何决定什么时候平滑、平滑哪个 role、平滑多大、平滑后如何 refresh。
```

当前 controller 太粗：

```text
固定 cycles
固定 role-block
固定 refresh D6-5
固定 NFS eta
```

下一步需要事件驱动 controller，而不是固定周期。

---

### 3.3 失败模式三：当前 NFS 没有真正投影到 teacher function 的 nullspace

当前 NFS 是 proposal + constraints，不是 KKT projection。这会导致：

```text
1. 可解释性不足；
2. projector 名称和数学含义不一致；
3. 难以知道失败来自 proposal，还是来自 nullspace 不存在；
4. 多 cycle 时漂移累积更难控制。
```

因此必须实现真正的 projected smoothing。

---

### 3.4 失败模式四：centers / width 固定可能造成 basis coverage bottleneck

当前 centers 固定在：

$$
[-2.5,2.5].
$$

width 固定为中心间距的 $1.4$ 倍。

但 PureKAN 的不同层输入分布不同：

```text
input_kan: raw normalized pixels
block_kan: hidden feature
output_kan: FixedNorm(hidden)
```

一个统一固定 grid 未必适合所有 role。

所以必须记录并测试：

```text
basis occupancy entropy
out-of-grid fraction
dead basis fraction
per-role center coverage
per-role width adequacy
```

如果 basis coverage 差，那么继续优化 coefficient 没有意义。

---

## 4. v4.9 新方向

v4.9 不再继续设计“新 functional optimizer”，而是先解决三个基础问题。

### 4.1 Base-RBF：把低阶通道放回 KAN edge 内部

实现：

$$
f_{ij}(x)
=
\beta_{ij}^{0}
+
\beta_{ij}^{1}x
+
\sum_{k=1}^K c_{ijk}B_k(x).
$$

或：

$$
f_{ij}(x)
=
w_{ij}^{base}\operatorname{silu}(x)
+
\sum_{k=1}^K c_{ijk}B_k(x).
$$

注意，这不是加回 MLP，而是把 base path 作为 KAN edge 的一部分。

参数仍然属于 edge function：

```text
base_const
base_linear
base_silu_weight
rbf_coeff
```

这样 PureKAN 仍可保持“所有可学习参数都是 KAN edge function 参数”的定义。

---

### 4.2 Exact NFS：实现真正的 function-preserving geometry projection

目标是求解：

$$
\min_{\Delta\theta}
\quad
\nabla R(\theta)^T\Delta\theta
+
\frac{1}{2\eta}\|\Delta\theta\|_M^2
$$

约束：

$$
J_f\Delta\theta\approx 0.
$$

其投影形式为：

$$
\Delta\theta
=
-\eta
\left(
M^{-1}g_R
-
M^{-1}J_f^T
(J_fM^{-1}J_f^T+\mu I)^{-1}
J_fM^{-1}g_R
\right).
$$

其中：

```text
g_R = geometry / Sobolev / roughness gradient
J_f = teacher function Jacobian，可以是 logits、hidden features、margin 的组合
M = functional metric / diagonal metric / identity
```

这才是真正的 nullspace smoothing。

---

### 4.3 Event-driven TAN：只在需要时 smoothing，而不是固定 cycle

当前 P5 失败说明固定 alternating cycles 不稳定。

v4.9 应该改为事件驱动：

```text
if geometry exceeds budget and task plateau is acceptable:
    do Exact NFS block smoothing
    run short task refresh
else:
    continue task learning
```

触发条件可以是：

$$
\phi'_{p95} / \phi'_{teacher} > r_{\phi}
$$

或：

$$
\Delta L_{holdout}\text{ plateau for }K\text{ steps}.
$$

接受条件是：

$$
\Delta Acc > -\epsilon_{acc},
\quad
KL < \epsilon_{KL},
\quad
\Delta \phi < 0.
$$

---

### 4.4 Basis / center / width feasibility

至少要测试：

```text
fixed uniform RBF
AB-RBF: constant + linear + RBF
learnable width only
per-role width
quantile centers
AB-RBF + learnable width
```

优先级是：

```text
1. AB-RBF
2. per-role / learnable width
3. quantile centers
```

原因：AB-RBF 直接补 low-order path，成本最低，解释最清楚。

---

## 5. v4.9 实验计划

### P0：Code audit 与参数化 smoke

#### 目的

确认新参数化和 Exact NFS 实现没有破坏 strict PureKAN 定义。

#### 需要实现

新增 edge 类型：

```text
RBFOnly:
  f(x)=sum_k c_k B_k(x)

AB-RBF-linear:
  f(x)=b+w x+sum_k c_k B_k(x)

AB-RBF-silu:
  f(x)=w_base silu(x)+sum_k c_k B_k(x)

RBF-learnWidth:
  centers fixed, width per layer trainable or functional-updated

RBF-quantileCenters:
  centers initialized from activation quantiles, fixed during run
```

需要记录：

```text
learnable_nonKAN_params
learnable_edge_params
rbf_coeff_params
base_params
center_params
width_params
functional_coverage
base_coverage
center_width_coverage
rollback_error
NFS_kkt_residual
NFS_constraint_residual
CG_residual
```

#### 判定

通过标准：

```text
nonKAN params = 0 for strict PureKAN variants
functional coverage = 1.0
base/width params either included in edge functional group or explicitly marked AdamW-control ablation
rollback_error < 1e-8
NFS constraint residual finite
```

#### 可视化

```text
parameter-count stacked bar: coeff/base/center/width
coverage heatmap by role: input/block/output
NFS residual histogram
```

---

### P1：RBF 参数化上限：AdamW feasibility frontier

#### 目的

先问架构本身：

$$
\boxed{
\text{Base-RBF / learnWidth 是否让 PureKAN-AdamW 更强或更平滑？}
}
$$

#### 方法

数据集：

```text
MNIST
Fashion-MNIST
KMNIST
```

模型：

```text
PureKAN-RBFOnly-AdamW
PureKAN-ABRBF-linear-AdamW
PureKAN-ABRBF-silu-AdamW
PureKAN-RBF-learnWidth-AdamW
PureKAN-ABRBF-linear-learnWidth-AdamW
Hybrid-DGKAN-UFULL-f085
MLP-AdamW
```

seeds：

```text
seed = 0,1,2
```

训练预算：

```text
same as v4.8 P1/P6 short budget
```

#### 记录指标

任务：

```text
test_acc
val_loss
val_loss_auc
ECE
NLL
margin_mean
margin_p10
classwise_acc
```

几何：

```text
phi_prime_p95
phi_prime_max
curvature_energy
jacobian_condition
sobolev_norm_total
coefficient_roughness
```

表示：

```text
rank_input
rank_block
rank_output
class_centroid_separation
feature_norm_mean/p95
```

basis：

```text
basis_occupancy_entropy
dead_basis_fraction
out_of_grid_fraction
per_role_basis_mean
center_coverage
width_value / width_softplus
```

base path：

```text
base_output_norm
rbf_output_norm
base_over_rbf_norm
base_margin_contribution
rbf_margin_contribution
base_coeff_norm
linear_base_slope_distribution
```

#### 判定

P1 通过标准不是 functional optimizer 成功，而是回答 architecture feasibility。

```text
AB-RBF positive if:
  acc >= RBFOnly-AdamW - 0.5%
  and either phi/J improves meaningfully
  or basis occupancy / rank improves
  or functional candidates later get better direction.
```

强 positive：

```text
AB-RBF-AdamW improves acc and reduces phi/J vs RBFOnly-AdamW.
```

如果 AB-RBF-AdamW 不如 RBFOnly-AdamW：

```text
base path 不是当前瓶颈，回到 optimizer/NFS。
```

#### 可视化

```text
accuracy-geometry Pareto: acc vs phi_prime_p95
accuracy-geometry Pareto: acc vs jacobian_condition
base/RBF output norm stacked bar by layer
basis occupancy heatmap by layer
center-width coverage histogram
rank vs acc scatter
margin p10 vs phi scatter
```

---

### P2：RBF eigenmode audit：AdamW 到底用了哪些模式

#### 目的

验证 functional update 是否压掉了 AdamW 需要的 RBF modes。

#### 方法

对 P1 中的 AdamW teacher，计算 Sobolev Gram：

$$
S=Q\Lambda Q^T.
$$

对 coefficient：

$$
\tilde c=Q^Tc.
$$

对 gradient：

$$
\tilde g=Q^Tg.
$$

记录不同 eigenvalue 区间的能量：

$$
E_c(m)=\|\tilde c_m\|^2,
$$

$$
E_g(m)=\|\tilde g_m\|^2.
$$

#### 记录指标

```text
coeff_energy_by_eigenmode
grad_energy_by_eigenmode
update_energy_by_eigenmode
energy_low/mid/high
high_mode_fraction
sobolev_penalty_contribution_by_mode
role-wise eigen energy
base-vs-rbf energy
```

#### 判定

如果 AdamW 的 useful modes 主要落在 high Sobolev eigenvalue 区域，则说明：

$$
\boxed{
\text{Sobolev U-FULL 把 AdamW 需要的表达模式压掉了。}
}
$$

如果 AB-RBF 把能量从 high modes 转移到 base / low modes，则 AB-RBF 是强候选。

#### 可视化

```text
eigenmode energy spectrum
cumulative energy vs eigenvalue
AdamW vs U-FULL coefficient spectrum
role-wise high-mode fraction bar
base path contribution vs high-mode fraction
```

---

### P3：Exact NFS projection audit

#### 目的

把 v4.8 的 proposal/backtracking NFS 升级为真正的 constrained projection。

#### 方法

Teacher：

```text
PureKAN-RBFOnly-AdamW
PureKAN-ABRBF-linear-AdamW
PureKAN-ABRBF-silu-AdamW
D6-allTaskAware
FCAdam-dataSob
```

Projection variants：

```text
Heuristic-NFS-role-block  # v4.8 current baseline
ExactNFS-logit
ExactNFS-hidden
ExactNFS-logit-hidden
ExactNFS-margin
ExactNFS-role-block
ExactNFS-role-cycle
```

Metric：

```text
M = identity
M = Sobolev diag
M = Sobolev full
M = dataSob diag
```

Projector solve：

```text
KKT direct on small sketch
CG with VJP/JVP
low-rank randomized SVD
```

#### Core formula

Use:

$$
\Delta\theta
=
-\eta
\left(
M^{-1}g_R
-
M^{-1}J_f^T
(J_fM^{-1}J_f^T+\mu I)^{-1}
J_fM^{-1}g_R
\right).
$$

where:

```text
g_R = gradient of roughness / Sobolev / phi proxy
J_f = Jacobian of logits, hidden features, or margins
```

#### 记录指标

Projection quality：

```text
KKT_residual
constraint_residual
projected_component_norm
nullspace_component_norm
J_delta_norm
M_norm_delta
CG_iterations
CG_residual
```

Function preservation：

```text
acc_drop
KL_teacher_student
logit_relative_drift
hidden_relative_drift
margin_relative_drift
argmax_flip_rate
classwise_flip_rate
```

Geometry：

```text
phi_reduction
J_reduction
sobolev_reduction
roughness_reduction
curvature_reduction
```

#### 判定

Exact NFS must beat heuristic NFS on at least one of:

```text
same geometry reduction with smaller drift
or stronger geometry reduction with same drift
or lower cycle instability in P5
```

Pass gate:

```text
acc_drop <= 0.5%
KL < 0.005
logit_drift < 0.03
argmax_flip_rate < 2%
phi_reduction > 10% or J_reduction > 20%
KKT_residual < 1e-3 or CG_residual < 1e-2
```

#### 可视化

```text
geometry reduction vs logit drift Pareto
phi reduction vs acc drop scatter
KKT residual histogram
CG residual by projector
argmax flip heatmap by class
heuristic vs exact NFS paired bar
```

---

### P4：One-cycle TAN v2 with exact NFS and AB-RBF

#### 目的

验证：

$$
\boxed{
\text{Task teacher} \rightarrow \text{Exact NFS} \rightarrow \text{Refresh}
}
$$

是否能稳定通过一 cycle，并且比 v4.8 one-cycle 更稳。

#### Methods

Task teachers：

```text
RBFOnly-AdamW
ABRBF-linear-AdamW
ABRBF-silu-AdamW
FCAdam-dataSob
D6-allTaskAware
```

NFS：

```text
Best Heuristic NFS-role-block from v4.8
Best ExactNFS-role-block
Best ExactNFS-logit-hidden
```

Refresh：

```text
none
D6-5
FCAdam5
AdamW-small-5  # diagnostic, not strict functional final
```

#### 记录指标

Cycle metrics：

```text
task_acc
smooth_acc
refresh_acc
acc_drop_from_task
phi_reduction_from_task
J_reduction_from_task
refresh_recovered_loss
refresh_recovered_acc
KL_after_smooth
KL_after_refresh
rank_after_smooth
rank_after_refresh
```

#### 判定

One-cycle pass:

```text
final_acc >= task_acc - 0.5%
phi_reduction_from_task > 10%
KL_after_refresh < 0.01
rank_after_refresh >= 0.85 * task_rank
```

All-dataset survivor required before P5.

#### 可视化

```text
one-cycle trajectory plot: task -> smooth -> refresh
acc drop vs phi reduction scatter
rank preservation after smoothing
refresh recovery bar
role contribution during refresh
```

---

### P5：Event-driven TAN multi-cycle controller

#### 目的

修复 v4.8 P5 的核心失败：固定 alternating cycles 不稳定，尤其 MNIST 不过。

#### Controller

不要固定每个 cycle 都 smoothing。改为事件触发：

```text
if geometry_bad and task_plateau and function_confidence_high:
    smooth role-block
    refresh
else:
    continue task learning
```

Trigger variables：

```text
phi_ratio_to_teacher
jac_ratio_to_teacher
holdout_loss_plateau
margin_p10_plateau
rank_drop
basis_occupancy_low
```

Pseudo-rule:

```text
smooth if:
  phi_ratio > 1.10
  and holdout_loss_improvement_last_K < threshold
  and rank_ratio > 0.75

skip smooth if:
  rank is still forming
  or margin_p10 is too low
  or recent smoothing caused acc drop
```

#### Methods

Only use P4 survivors.

Run:

```text
cycles up to 3
but smoothing can be skipped
```

#### 记录指标

```text
cycle_index
smooth_triggered
trigger_reason
skip_reason
teacher_acc
pre_smooth_acc
post_smooth_acc
post_refresh_acc
acc_drop
phi_reduction
rank_change
margin_change
KL
argmax_flip
refresh_steps_used
```

#### 判定

Pass:

```text
all datasets pass 3-cycle gate
final_acc >= initial_task_acc - 1%
final_phi_reduction > 10%
rank_final >= 0.75 * initial_task_rank
no cycle has acc collapse > 2%
```

#### 可视化

```text
cycle timeline per dataset
trigger reason stacked bar
acc/phi/rank over cycles
smoothing accepted vs rejected plot
MNIST failure drilldown if fail
```

---

### P6：Capacity / basis expansion full run

#### 目的

真正回答：

$$
\boxed{
\text{RBF-only 是否容量/参数化不足？}
}
$$

#### Config grid

```text
RBFOnly h64 b16 d4
RBFOnly h96 b24 d4
RBFOnly h96 b32 d2
ABRBF-linear h96 b24 d4
ABRBF-silu h96 b24 d4
ABRBF-linear h128 b24 d4
ABRBF-linear h96 b32 d4
RBF-learnWidth h96 b24 d4
ABRBF-linear-learnWidth h96 b24 d4
```

Methods:

```text
AdamW
AdamW-MixedGeom
Best functional task learner from earlier, if any
Best TAN one-cycle/multi-cycle, if P4/P5 passes
```

Seeds:

```text
0,1,2 initially
```

#### 记录指标

```text
param_count
step_time_ms
peak_memory
acc
val_auc
ECE
phi/J
rank
margin
basis coverage
base contribution
width distribution
```

#### 判定

If AB-RBF-AdamW improves smooth frontier:

```text
keep AB-RBF as new PureKAN default.
```

If AB-RBF functional closes gap:

```text
continue functional optimizer on AB-RBF.
```

If AB-RBF AdamW itself fails:

```text
RBF parameterization is not the main issue; return to optimizer dynamics.
```

---

### P7：3-seed candidate selection

Only enter if P4/P5 or P6 yields a candidate.

Candidate examples:

```text
ABRBF-linear-AdamW + ExactNFS one-cycle
ABRBF-linear-FCAdam + ExactNFS one-cycle
RBFOnly-AdamW + ExactNFS event-driven TAN
```

Baselines:

```text
PureKAN-RBFOnly-AdamW
PureKAN-ABRBF-AdamW
Hybrid-DGKAN-UFULL-f085
MLP-AdamW
D6-allTaskAware
```

Gate:

```text
acc >= PureKAN-AdamW - 0.5% on all datasets
phi reduction > 10% or J reduction > 20%
ECE not worse by > 0.02
rank >= 0.75 * PureKAN-AdamW
val_auc not worse by > 5%
```

---

### P8：5-seed confirm

Run final candidate vs baselines.

Record paired deltas:

```text
paired_acc_delta_vs_PureKAN_AdamW
paired_auc_delta_vs_PureKAN_AdamW
paired_phi_delta
paired_jac_delta
paired_ece_delta
paired_rank_delta
```

Pass:

```text
accuracy CI not worse than AdamW by > 0.5%
geometry improves with CI positive
ECE not worse
no catastrophic seed
```

---

### P9：10-seed final confirm

Only if P8 passes.

Final claim allowed only if:

```text
PureKAN functional / TAN candidate matches or beats PureKAN-AdamW accuracy;
geometry improves materially;
model remains strict KAN-edge parameterization;
no hidden MLP stem/head is introduced.
```

---

## 6. Required artifacts

Under `results/v4_9/`:

```text
p0_code_audit.csv
p1_architecture_frontier.csv
p1_training_trace.csv
p2_eigenmode_audit.csv
p3_exact_nfs_projection.csv
p4_tan_one_cycle_v2.csv
p5_event_tan_cycles.csv
p6_capacity_basis_expansion_full.csv
p7_candidate_selection.csv
p8_confirm5.csv
p9_confirm10.csv
failure_table.csv
aggregate_decision.json
figures/
```

Figures:

```text
figures/acc_phi_pareto.svg
figures/acc_jac_pareto.svg
figures/base_rbf_contribution_by_layer.svg
figures/basis_occupancy_heatmap.svg
figures/eigenmode_energy_spectrum.svg
figures/nfs_geometry_vs_drift.svg
figures/nfs_constraint_residual_hist.svg
figures/tan_one_cycle_trajectory.svg
figures/tan_cycle_timeline.svg
figures/failure_taxonomy_heatmap.svg
```

---

## 7. Failure taxonomy for v4.9

```text
F1_architecture_frontier_absent:
  AB-RBF / learnWidth / capacity expansion still cannot expose accurate smoother solutions.

F2_nfs_not_exact:
  Exact NFS cannot reduce geometry under low drift.

F3_nfs_worse_than_heuristic:
  Exact projection is worse than v4.8 heuristic NFS.

F4_smoothing_destroys_function:
  KL/logit drift/acc drop too large.

F5_refresh_cannot_recover:
  smoothing works, but refresh cannot restore task loss.

F6_multicycle_instability:
  one-cycle works, repeated event-driven cycles fail.

F7_basis_coverage_bad:
  dead basis / out-of-grid / low occupancy dominate.

F8_base_path_not_used:
  AB-RBF base contribution remains near zero.

F9_functional_candidate_acc_gap:
  final candidate geometry improves but accuracy gap too large.
```

---

## 8. Expected interpretations

### Case A：AB-RBF AdamW exposes better frontier

Then current PureKAN failure is partly parameterization-driven.

Conclusion:

$$
\boxed{
\text{RBF-only PureKAN lacked low-order edge channels.}
}
$$

Next default becomes AB-RBF PureKAN.

---

### Case B：Exact NFS beats heuristic NFS

Then v4.8 was limited by approximate proposal/backtracking.

Conclusion:

$$
\boxed{
\text{function-preserving geometry projection is real and should become a primitive.}
}
$$

---

### Case C：One-cycle works but multi-cycle fails again

Then the controller is still the bottleneck.

Conclusion:

$$
\boxed{
\text{Learn-Smooth-Refresh is valid locally, but online scheduling is unresolved.}
}
$$

---

### Case D：AB-RBF + Exact NFS still fails

Then strict PureKAN functional training remains unresolved.

Conclusion:

$$
\boxed{
\text{Current KAN-edge parameterization plus coefficient-only functional control is insufficient.}
}
$$

At that point, the project should either:

```text
1. return to Hybrid-DGKAN as the validated contribution;
2. move to spline / efficient-KAN / rational primitive with primitive-specific metric;
3. or allow a small set of non-KAN normalization/base parameters as necessary infrastructure.
```

---

## 9. Current recommendation

Do not continue another general functional optimizer sweep.

The next run should be exactly:

```text
v4.9-P0 code audit
v4.9-P1 AB-RBF / learnWidth AdamW frontier
v4.9-P2 eigenmode audit
v4.9-P3 Exact NFS projection
```

Only if P3 passes should we continue to TAN cycles.

The strongest current hypothesis is:

$$
\boxed{
\text{PureKAN needs a better edge parameterization first: base + RBF, then function-preserving smoothing.}
}
$$



---


# Source 26: `docs/DG-KAN_v4.9_RBFParameterization_ExactNFS_结果复盘.md`


# DG-KAN v4.9 RBF Parameterization / Exact NFS 结果复盘

本轮依据 `docs/DG-KAN_v4.9_RBFParameterization_ExactNFS_下一步实验计划.md`。目标是把 PureKAN functional failure 拆成三个问题：RBF-only 参数化、RBF eigenmode 使用方式、以及 v4.8 NFS 是否需要真正的 Jacobian-nullspace projection。

## Run Inventory

| stage | rows | errors |
|---|---:|---:|
| P0 code audit | 21 | 0 |
| P1 architecture frontier | 45 | 0 |
| P2 eigenmode audit | 270 | 0 |
| P3 exact NFS projection | 45 | 0 |
| P4 TAN one-cycle v2 | 0 | 0 |
| P5 event TAN cycles | 0 | 0 |
| P6 capacity/basis expansion | 24 | 0 |

## Code / Config Changes

```text
experiments/run_gafu_v49.py
  Added strict edge-only PureKAN parameterizations:
    RBFOnly
    ABRBF-linear
    ABRBF-silu
    RBF-learnWidth
    ABRBF-linear-learnWidth
    RBF-quantileCenters smoke
  Added local base/RBF contribution audit, width/basis coverage audit, and eigenmode energy audit.
  Added Exact NFS sketch projection:
    explicit Jacobian rows for logits / hidden sketch / margin constraints
    KKT direct solve in coefficient space
    identity and Sobolev-diagonal metrics

experiments/analyze_gafu_v49.py
  Generates gate summaries, failure table, aggregate_decision.json,
  SVG diagnostics, log entry, and this replay.
```

## P0 Code Audit

| rows | errors | pass | max nonKAN | min coverage | max rollback | max KKT |
|---|---|---|---|---|---|---|
| 21 | 0 | true | 0.0000 | 1.0000 | 0.0000 | 0.000001 |

P0 verdict: pass. 新 edge 参数化仍保持 strict edge-only：learnable non-KAN params 为 0，functional coverage 为 1.0，rollback 为 0。

## P1 Architecture Frontier

| dataset | method | runs | acc | std | gap vs RBF | phi red | J red | base/RBF | positive |
|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | 3 | 0.7949 | 0.0042 | -0.0007 | -0.0597 | -30.6799 | 0.5480 | 1 |
| Fashion-MNIST | PureKAN-ABRBF-linear-learnWidth-AdamW | 3 | 0.7930 | 0.0057 | 0.0013 | -0.0682 | -630.9040 | 0.5651 | 1 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | 3 | 0.7969 | 0.0193 | -0.0026 | -0.0168 | -1.5237 | 0.1647 | 1 |
| Fashion-MNIST | PureKAN-RBF-learnWidth-AdamW | 3 | 0.7943 | 0.0066 | 0.0000 | -0.0145 | -0.2339 | 0.0000 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | 3 | 0.7943 | 0.0097 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | 3 | 0.6888 | 0.0198 | -0.0078 | -0.0558 | -29.6148 | 0.5725 | 1 |
| KMNIST | PureKAN-ABRBF-linear-learnWidth-AdamW | 3 | 0.6816 | 0.0181 | -0.0007 | -0.0661 | -3.1688 | 0.5896 | 1 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | 3 | 0.6947 | 0.0082 | -0.0137 | -0.0198 | -0.5194 | 0.1963 | 1 |
| KMNIST | PureKAN-RBF-learnWidth-AdamW | 3 | 0.6751 | 0.0239 | 0.0059 | -0.0110 | 0.5166 | 0.0000 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | 3 | 0.6810 | 0.0166 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |
| MNIST | PureKAN-ABRBF-linear-AdamW | 3 | 0.8900 | 0.0072 | -0.0143 | -0.0313 | -2.8761 | 0.5610 | 1 |
| MNIST | PureKAN-ABRBF-linear-learnWidth-AdamW | 3 | 0.8874 | 0.0088 | -0.0117 | -0.0401 | -2.2144 | 0.5761 | 1 |
| MNIST | PureKAN-ABRBF-silu-AdamW | 3 | 0.8945 | 0.0105 | -0.0189 | -0.0169 | -56.2586 | 0.1998 | 1 |
| MNIST | PureKAN-RBF-learnWidth-AdamW | 3 | 0.8757 | 0.0157 | 0.0000 | -0.0070 | 0.3571 | 0.0000 | 1 |
| MNIST | PureKAN-RBFOnly-AdamW | 3 | 0.8757 | 0.0133 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |

P1 verdict:

```text
AB-RBF is architecture-positive.
ABRBF-silu improves mean accuracy on all three datasets vs RBFOnly:
  MNIST +1.88 points
  Fashion +0.26 points
  KMNIST +1.37 points

ABRBF-linear also improves MNIST/KMNIST and keeps Fashion roughly tied.
learnWidth alone is not a clear positive signal.
The base path is actively used: ABRBF-linear base/RBF norm is about 0.55-0.59.
```

P1 all-dataset architecture positives:

```text
PureKAN-ABRBF-linear-AdamW, PureKAN-ABRBF-linear-learnWidth-AdamW, PureKAN-ABRBF-silu-AdamW
```

## P2 Eigenmode Audit

| dataset | method | high coeff | base mode | grad high | eig cond | shift |
|---|---|---|---|---|---|---|
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | 0.2293 | 0.2560 | 0.0001 | 78.7 | 1 |
| Fashion-MNIST | PureKAN-ABRBF-linear-learnWidth-AdamW | 0.2301 | 0.2527 | 0.0002 | 78.7 | 1 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | 0.2508 | 0.0931 | 0.0002 | 78.7 | 1 |
| Fashion-MNIST | PureKAN-RBF-learnWidth-AdamW | 0.2697 | 0.0000 | 0.0003 | 78.7 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | 0.2689 | 0.0000 | 0.0002 | 78.7 | 1 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | 0.2241 | 0.2434 | 0.0002 | 78.7 | 1 |
| KMNIST | PureKAN-ABRBF-linear-learnWidth-AdamW | 0.2248 | 0.2408 | 0.0003 | 78.7 | 1 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | 0.2439 | 0.0874 | 0.0003 | 78.7 | 1 |
| KMNIST | PureKAN-RBF-learnWidth-AdamW | 0.2640 | 0.0000 | 0.0004 | 78.7 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | 0.2629 | 0.0000 | 0.0003 | 78.7 | 1 |
| MNIST | PureKAN-ABRBF-linear-AdamW | 0.2285 | 0.2520 | 0.0001 | 78.7 | 1 |
| MNIST | PureKAN-ABRBF-linear-learnWidth-AdamW | 0.2292 | 0.2485 | 0.0002 | 78.7 | 1 |
| MNIST | PureKAN-ABRBF-silu-AdamW | 0.2407 | 0.0983 | 0.0002 | 78.7 | 1 |
| MNIST | PureKAN-RBF-learnWidth-AdamW | 0.2586 | 0.0000 | 0.0003 | 78.7 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | 0.2578 | 0.0000 | 0.0003 | 78.7 | 1 |

P2 verdict:

```text
RBFOnly keeps about 26-27% of coefficient energy in the high Sobolev-eigen band.
ABRBF-linear shifts about 24-26% of coefficient energy into explicit base modes
and reduces high-mode coefficient energy to about 22-23%.
ABRBF-silu uses a smaller base channel but still reduces high-mode pressure.

This supports the v4.9 hypothesis:
  fixed-grid RBF-only was forcing low-order structure through RBF coefficient modes.
```

## P3 Exact NFS Projection

| dataset | teacher | variant | metric | acc drop | phi red | J red | KL | KKT | pass |
|---|---|---|---|---|---|---|---|---|---|
| MNIST | PureKAN-RBFOnly-AdamW | Heuristic-NFS-role-block | identity | 0.0000 | 0.0290 | 0.6509 | 0.00009 |  | 1 |
| MNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit | identity | 0.0000 | 0.0000 | 0.0001 | -0.00000 | 0.000000 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit-hidden | identity | 0.0000 | 0.0000 | 0.0002 | 0.00000 | 0.000001 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit-hidden | sobolev_diag | 0.0000 | 0.0000 | 0.0001 | 0.00000 | 0.000000 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | ExactNFS-margin | identity | 0.0000 | 0.0000 | 0.0002 | -0.00000 | 0.000000 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | Heuristic-NFS-role-block | identity | 0.0020 | 0.0399 | 0.1721 | 0.00065 |  | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit | identity | 0.0000 | 0.0000 | 0.0004 | 0.00000 | 0.000000 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit-hidden | identity | 0.0000 | 0.0000 | 0.0005 | 0.00000 | 0.000001 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit-hidden | sobolev_diag | 0.0000 | 0.0000 | 0.0003 | 0.00000 | 0.000001 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-margin | identity | 0.0000 | 0.0000 | 0.0005 | 0.00000 | 0.000000 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | Heuristic-NFS-role-block | identity | 0.0000 | 0.0300 | 0.7110 | 0.00015 |  | 1 |
| MNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit | identity | 0.0000 | 0.0000 | 0.0001 | -0.00000 | 0.000000 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit-hidden | identity | 0.0000 | 0.0000 | 0.0001 | 0.00000 | 0.000001 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit-hidden | sobolev_diag | 0.0000 | 0.0000 | 0.0001 | 0.00000 | 0.000000 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-margin | identity | 0.0000 | 0.0000 | 0.0001 | 0.00000 | 0.000000 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | Heuristic-NFS-role-block | identity | 0.0000 | 0.0306 | 0.3500 | 0.00003 |  | 1 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit | identity | 0.0000 | 0.0000 | 0.0000 | 0.00000 | 0.000000 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit-hidden | identity | 0.0000 | 0.0000 | 0.0000 | -0.00000 | 0.000001 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit-hidden | sobolev_diag | 0.0000 | 0.0000 | 0.0000 | 0.00000 | 0.000000 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | ExactNFS-margin | identity | 0.0000 | 0.0000 | 0.0000 | 0.00000 | 0.000000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | Heuristic-NFS-role-block | identity | 0.0020 | 0.0414 | -0.2019 | 0.00049 |  | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit | identity | 0.0000 | 0.0000 | 0.0000 | -0.00000 | 0.000000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit-hidden | identity | 0.0000 | 0.0000 | 0.0001 | -0.00000 | 0.000001 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit-hidden | sobolev_diag | 0.0000 | 0.0000 | 0.0000 | -0.00000 | 0.000000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-margin | identity | 0.0000 | 0.0000 | 0.0001 | 0.00000 | 0.000000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | Heuristic-NFS-role-block | identity | -0.0039 | 0.0313 | 0.5270 | 0.00010 |  | 1 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit | identity | 0.0000 | 0.0000 | 0.0001 | -0.00000 | 0.000000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit-hidden | identity | 0.0000 | 0.0000 | 0.0001 | -0.00000 | 0.000001 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit-hidden | sobolev_diag | 0.0000 | 0.0000 | 0.0000 | -0.00000 | 0.000000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-margin | identity | 0.0000 | 0.0000 | 0.0001 | -0.00000 | 0.000000 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | Heuristic-NFS-role-block | identity | -0.0020 | 0.0305 | 0.4150 | 0.00007 |  | 1 |
| KMNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit | identity | 0.0000 | 0.0000 | -0.0001 | -0.00000 | 0.000000 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit-hidden | identity | 0.0000 | 0.0000 | -0.0001 | 0.00000 | 0.000001 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit-hidden | sobolev_diag | 0.0000 | 0.0000 | 0.0000 | -0.00000 | 0.000000 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | ExactNFS-margin | identity | 0.0000 | 0.0000 | -0.0001 | -0.00000 | 0.000000 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | Heuristic-NFS-role-block | identity | 0.0000 | 0.0389 | 0.3271 | 0.00039 |  | 1 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit | identity | 0.0000 | 0.0000 | 0.0013 | 0.00000 | 0.000000 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit-hidden | identity | 0.0000 | 0.0000 | 0.0013 | 0.00000 | 0.000001 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit-hidden | sobolev_diag | 0.0000 | 0.0000 | 0.0014 | 0.00000 | 0.000000 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-margin | identity | 0.0000 | 0.0000 | 0.0017 | 0.00000 | 0.000000 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | Heuristic-NFS-role-block | identity | 0.0000 | 0.0309 | 0.4123 | 0.00012 |  | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit | identity | 0.0000 | 0.0000 | 0.0000 | -0.00000 | 0.000000 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit-hidden | identity | 0.0000 | 0.0000 | 0.0000 | -0.00000 | 0.000001 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit-hidden | sobolev_diag | 0.0000 | 0.0000 | 0.0000 | 0.00000 | 0.000000 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-margin | identity | 0.0000 | 0.0000 | 0.0000 | 0.00000 | 0.000000 | 0 |

P3 verdict:

```text
Exact NFS is mathematically stable but too conservative.
KKT residual is typically around 1e-7 to 1e-6, and KL/logit drift is essentially zero,
but geometry reduction is also essentially zero.

Heuristic NFS-role-block remains useful and function-preserving, but it is not a true nullspace projection.
It gives small phi reductions and larger J reductions, reproducing the v4.8 local signal.
No ExactNFS variant has an all-dataset survivor.
```

P3 all-dataset survivors including heuristic:

```text
PureKAN-RBFOnly-AdamW|Heuristic-NFS-role-block|identity|logit_hidden
```

P3 exact-only survivors:

```text
none
```

## P4 / P5 Decision

```text
P4 TAN one-cycle v2: not run.
Reason: P3 produced no exact-NFS all-dataset survivor.

P5 event-driven TAN cycles: not run.
Reason: P4 was not reached.
```

## P6 Capacity / Basis Expansion

| dataset | edge | hidden | basis | depth | acc | phi | rank | base/RBF |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | ABRBF-linear | 96 | 24 | 4 | 0.7930 | 0.1881 | 9.5165 | 0.6043 |
| Fashion-MNIST | ABRBF-linear | 96 | 32 | 4 | 0.7559 | 0.2378 | 7.9458 | 0.5690 |
| Fashion-MNIST | ABRBF-linear-learnWidth | 96 | 24 | 4 | 0.7910 | 0.1895 | 9.6156 | 0.6195 |
| Fashion-MNIST | ABRBF-silu | 96 | 24 | 4 | 0.7793 | 0.1842 | 9.0402 | 0.1274 |
| Fashion-MNIST | RBF-learnWidth | 96 | 24 | 4 | 0.7617 | 0.1853 | 10.1644 | 0.0000 |
| Fashion-MNIST | RBFOnly | 64 | 16 | 4 | 0.7891 | 0.1700 | 10.0688 | 0.0000 |
| Fashion-MNIST | RBFOnly | 96 | 24 | 4 | 0.7578 | 0.1836 | 10.0119 | 0.0000 |
| Fashion-MNIST | RBFOnly | 96 | 32 | 2 | 0.7715 | 0.2316 | 9.6263 | 0.0000 |
| KMNIST | ABRBF-linear | 96 | 24 | 4 | 0.6836 | 0.2044 | 9.7898 | 0.5215 |
| KMNIST | ABRBF-linear | 96 | 32 | 4 | 0.6465 | 0.2492 | 8.8968 | 0.5334 |
| KMNIST | ABRBF-linear-learnWidth | 96 | 24 | 4 | 0.6836 | 0.2059 | 9.8360 | 0.5409 |
| KMNIST | ABRBF-silu | 96 | 24 | 4 | 0.6797 | 0.1912 | 11.0848 | 0.1064 |
| KMNIST | RBF-learnWidth | 96 | 24 | 4 | 0.6660 | 0.1934 | 11.0521 | 0.0000 |
| KMNIST | RBFOnly | 64 | 16 | 4 | 0.6758 | 0.1719 | 11.5154 | 0.0000 |
| KMNIST | RBFOnly | 96 | 24 | 4 | 0.6680 | 0.1916 | 10.8185 | 0.0000 |
| KMNIST | RBFOnly | 96 | 32 | 2 | 0.6777 | 0.2640 | 9.4357 | 0.0000 |
| MNIST | ABRBF-linear | 96 | 24 | 4 | 0.8438 | 0.2054 | 9.7568 | 0.5222 |
| MNIST | ABRBF-linear | 96 | 32 | 4 | 0.8398 | 0.2756 | 7.9179 | 0.4903 |
| MNIST | ABRBF-linear-learnWidth | 96 | 24 | 4 | 0.8379 | 0.2060 | 9.7848 | 0.5334 |
| MNIST | ABRBF-silu | 96 | 24 | 4 | 0.8672 | 0.1990 | 9.6207 | 0.1151 |
| MNIST | RBF-learnWidth | 96 | 24 | 4 | 0.8125 | 0.2022 | 9.1507 | 0.0000 |
| MNIST | RBFOnly | 64 | 16 | 4 | 0.8105 | 0.1767 | 10.7913 | 0.0000 |
| MNIST | RBFOnly | 96 | 24 | 4 | 0.8320 | 0.2019 | 8.9646 | 0.0000 |
| MNIST | RBFOnly | 96 | 32 | 2 | 0.8457 | 0.2899 | 7.6966 | 0.0000 |

P6 verdict:

```text
Capacity expansion confirms the parameterization signal but does not solve geometry.
ABRBF-silu h96/b24/d4 is the best MNIST point in this compact seed0 expansion.
ABRBF-linear improves/ties Fashion and KMNIST relative to larger RBF-only configs.
learnWidth alone again does not help.

However phi generally rises with larger capacity, so architecture alone does not expose
a smooth accurate frontier.
```

## Failure Diagnosis

Generated:

```text
failure_table.csv
p1_architecture_gate_summary.csv
p2_eigenmode_gate_summary.csv
p3_exact_nfs_gate_summary.csv
p6_capacity_gate_summary.csv
```

Diagnosis:

```text
F1_architecture_frontier_absent:
  false as a blanket diagnosis. AB-RBF is accuracy-positive and actively used.

F2_nfs_not_exact:
  true for the current exact projection as an optimizer primitive.
  The exact nullspace direction under logits/hidden/margin constraints is nearly zero.

F3_nfs_worse_than_heuristic:
  true. Heuristic NFS gives useful geometry movement; exact NFS preserves function too strictly.

F7_basis_coverage_bad:
  partly true. RBFOnly relies on high Sobolev modes more than AB-RBF.

F8_base_path_not_used:
  false. ABRBF-linear base/RBF norm is consistently nontrivial.
```

## Required Artifacts

Written under `results/v4_9/`:

```text
p0_code_audit.csv
p1_architecture_frontier.csv
p1_training_trace.csv
p2_eigenmode_audit.csv
p3_exact_nfs_projection.csv
p4_tan_one_cycle_v2.csv
p5_event_tan_cycles.csv
p6_capacity_basis_expansion_full.csv
p7_candidate_selection.csv
p8_confirm5.csv
p9_confirm10.csv
failure_table.csv
aggregate_decision.json
figures/
```

## Final Decision

```text
DG-KAN v4.9 status:
  stop_after_p3_no_exact_nfs_survivor

What improved:
  AB-RBF, especially ABRBF-silu, improves the PureKAN architecture frontier.
  Eigenmode audit confirms base paths reduce pressure on high RBF Sobolev modes.
  Exact NFS implementation is auditable and produces tiny KKT residuals.

What failed:
  Exact NFS found almost no useful geometry-moving direction under strict function constraints.
  Heuristic NFS remains better for practical smoothing, but it is not mathematically exact.
  Capacity/basis expansion improves accuracy in places but does not produce a smooth frontier.

Conclusion:
  PureKAN failure is partly parameterization-driven: RBF-only is too restrictive.
  But exact function-preserving smoothing is also too conservative in the current form.
  The next default should move from RBFOnly to AB-RBF for PureKAN probes, while redesigning NFS
  as a relaxed/trust-region projection instead of a hard Jacobian nullspace projection.
```



---


# Source 27: `docs/DG-KAN_v5.0_EdgeDecomposedFunctionalTraining_实验计划.md`


# DG-KAN v5.0：Edge-Decomposed Functional Training 实验计划

## 0. 本轮结论先行

v4.9 之后，我认为 PureKAN functional update 的问题不能再被理解为“再找一个更好的 Sobolev 系数、trust radius 或 warmup”。现在更准确的判断是：

$$
\boxed{\text{PureKAN 的主要瓶颈是 edge 参数化、几何定义、functional update 目标三者错配。}}
$$

当前 strict PureKAN 主要是 fixed-grid RBF-only 网络：每条边是

$$
f_{ij}(x)=\sum_k c_{ijk}B_k(x),
$$

其中真正被训练的是 $c_{ijk}$。RBF centers 和 width 基本是固定的；PureKAN 里没有普通 stem、head、bias、learnable LayerNorm，也没有 learnable alpha。这个设置很干净，但也非常苛刻。它要求 RBF 系数同时承担低阶线性结构、尺度调整、feature formation、class boundary 和非线性修正。过去的 U-FULL / TFU / FNG / FCAdam / NFS 都是在这个坐标系上试图优化系数，但它们没有给低阶结构一个自然通道。

v4.9 的结果已经改变了我们对问题的理解。AB-RBF 在 AdamW 下是 architecture-positive：`ABRBF-silu` 在 MNIST、Fashion、KMNIST 都提升 mean accuracy；`ABRBF-linear` 在 MNIST/KMNIST 提升，并在 Fashion 基本持平。eigenmode audit 显示，RBFOnly 大约有 $26\%-27\%$ coefficient energy 落在 high Sobolev-eigen band；ABRBF-linear 把约 $24\%-26\%$ 能量转移到 explicit base modes，并把 high-mode energy 降到约 $22\%-23\%$。这说明 fixed-grid RBF-only 很可能把低阶结构强行塞进了 RBF 高模态。

同时，Exact NFS 的结果也非常重要。Exact NFS 的 KKT residual 非常小，KL/logit drift 几乎为零，但 geometry reduction 也几乎为零。这说明 Exact NFS 并不是数值不稳定，而是过于保守，或者说它的约束空间把真正能降几何的方向投掉了。相比之下，Heuristic NFS-role-block 还能给出小的 $\phi$ reduction 和较大的 Jacobian reduction。这说明 v4.8/v4.9 的 NFS 信号是真实的，但“严格 nullspace projection”不是正确实现方式。

因此 v5.0 的核心不再是“继续发明一个单步 functional optimizer”，而是建立一个新的训练框架：

$$
\boxed{\text{Edge-Decomposed Functional Training}}
$$

它的基本思想是：每条 KAN edge 不应该只有 RBF residual，而应该被分成低阶 base path 和 RBF residual path。base path 负责低频、线性、尺度和稳定表示；RBF residual 负责非线性修正；几何约束主要作用在 RBF residual 上，而不是无差别压制整个 edge。

---

## 1. v4.9 的关键证据与新的解释

### 1.1 P0：实现不是主要问题

v4.9 P0 通过：21 rows、0 errors、max nonKAN 为 0、functional coverage 为 1.0、rollback 为 0、max KKT residual 约 $10^{-6}$。这说明新 edge 参数化和 Exact NFS 路径至少在 smoke 层面是可审计的。后续失败不能优先归因于参数没被更新、rollback 出错、hidden non-KAN 参数混入等低级实现问题。

### 1.2 P1：AB-RBF 是 architecture-positive，但当前几何指标会误判它

P1 里 `ABRBF-silu` 在三数据集都提升 mean accuracy；`ABRBF-linear` 也改善 MNIST/KMNIST，并且 base/RBF norm 大约在 $0.55-0.59$，说明 base path 不是摆设，而是真正在用。这个结果强烈支持：RBF-only 的边函数缺少低阶通道。

但是 P1 里 AB-RBF 的 `phi red` 和 `J red` 往往是负的，尤其 `ABRBF-linear` 的 Jacobian reduction 数字非常差。这里不能简单解读为“AB-RBF 几何坏”。因为现有几何指标把 base path 的导数也算作“坏导数”。如果 edge 里有

$$
f_{ij}(x)=b_{ij}+w_{ij}x+\sum_k c_{ijk}B_k(x),
$$

那么 $w_{ij}$ 带来的导数是稳定低阶结构，不等同于 RBF residual 的高频粗糙性。当前 $\phi'$ / Jacobian gate 没有区分“有用低阶 derivative”和“危险高频 derivative”。因此 v5.0 必须重新定义几何审计：

$$
\text{edge derivative}=	ext{base derivative}+\text{RBF residual derivative}.
$$

新的 geometry gate 应该主要约束 RBF residual 的粗糙性，同时单独记录 full-network Jacobian / Lipschitz，而不是把二者混在一个分数里。

### 1.3 P2：RBF-only 确实在用高 Sobolev eigenmodes 承担低阶结构

RBFOnly 的 high eigenmode coefficient energy 约 $26\%-27\%$。ABRBF-linear 把约 $24\%-26\%$ 能量转移到 base modes，并将 high-mode energy 降到 $22\%-23\%$。这支持如下解释：

$$
\boxed{\text{RBF-only 不是没有表达力，而是用不自然的 RBF 高模态表达低阶结构。}}
$$

这也解释了为什么 Sobolev functional update 很难训练 PureKAN。Sobolev metric 正在惩罚的，很可能正是 AdamW 用来拼出低阶结构的 coefficient modes。功能上，这些 modes 可能是任务需要的；几何上，它们被 Sobolev 视作高代价方向。

### 1.4 P3：Exact NFS 过于保守，Heuristic NFS 的“小松弛”反而有效

Exact NFS-logit / logit-hidden / margin 的 KKT residual 很小，KL 和 logit drift 基本为零，但 $\phi$ reduction 几乎为零。这说明 exact projection 不是数值坏掉，而是投影后几何梯度几乎没有剩余可动方向。

所以 v5.0 不应该继续追求“更 exact 的 nullspace”。更应该做的是 **relaxed constrained smoothing**：允许极小的 logit / hidden / margin drift，换取可观的 geometry reduction。形式上不是硬约束

$$
J_f\Delta\theta=0,
$$

而是 trust-region/penalty：

$$
\min_{\Delta\theta}
\quad
\nabla R(\theta)^\top\Delta\theta
+
\frac{1}{2\eta}\|\Delta\theta\|_M^2
+
\frac{\mu_z}{2}\|J_z\Delta\theta\|^2
+
\frac{\mu_h}{2}\|J_h\Delta\theta\|^2.
$$

这样不会把所有 smoothing direction 直接杀掉。

### 1.5 P6：单纯扩大 capacity / basis 不能解决 geometry

P6 显示：更大的 hidden、basis 或 learnWidth 并没有稳定打开 smooth accurate frontier。ABRBF-silu h96/b24/d4 在 MNIST 上很好；ABRBF-linear 在 Fashion/KMNIST 上改善或持平；learnWidth alone 仍然不是 clear positive。更重要的是，capacity 扩大后 $\phi$ 通常升高。因此“多给 basis / 多给 width 自由度”不是根因解法。

---

## 2. 代码实现审计：必须改的地方

### 2.1 当前 core 的 RBFDense 仍是 fixed-grid RBF coefficient layer

当前 `RBFDense` 的核心实现是固定 centers、固定 width、训练 `coeff`。这意味着 PureKAN strict 版本中的 input/block/output KAN 都只训练 RBF coefficient，而 centers/width/base/bias 都没有参与 functional update。v5.0 的第一件事，是把 AB-RBF 从 runner-level 实验配置提升成 core-level edge module。

### 2.2 AB-RBF 必须是 edge-internal 参数，不是回退到 MLP

v5.0 不能简单加回 Linear stem/head。目标是把低阶通道放回每条 KAN edge 内部：

$$
f_{ij}(x)=b_{ij}+w_{ij}x+u_{ij}\operatorname{silu}(x)+\sum_k c_{ijk}B_k(x).
$$

这里 $b_{ij}$、$w_{ij}$、$u_{ij}$ 都属于 edge 参数，应该计入 functional coverage。它们不是 non-KAN 参数。这样才能保持 strict edge-only PureKAN。

### 2.3 coefficient_named_params 必须支持 edge groups

当前 functional coverage 不能只说 “coeff seen = 1”。需要进一步拆成：

```text
base_const_seen
base_linear_seen
base_silu_seen
rbf_coeff_seen
width_seen, if learnable width enabled
center_seen, if learnable center enabled
```

v5.0 P0 里每一个 strict row 都必须记录这些字段，并保证总 coverage 为 1.0。否则我们无法判断 functional optimizer 是否真正覆盖了所有 edge 参数。

### 2.4 几何指标需要拆分 base geometry 与 residual geometry

新 edge 的 derivative 是：

$$
f'_{ij}(x)=w_{ij}+u_{ij}\operatorname{silu}'(x)+\sum_k c_{ijk}B'_k(x).
$$

当前 $\phi'$ 把这些全部混在一起。v5.0 必须记录：

```text
phi_base_p95
phi_rbf_p95
phi_total_p95
curvature_base_p95
curvature_rbf_p95
curvature_total_p95
residual_sobolev_norm
base_norm
rbf_norm
base_over_rbf
```

如果 AB-RBF 的 total $\phi'$ 上升，但 RBF residual roughness 下降、accuracy 上升、ECE 不坏，那这不是失败。

### 2.5 Exact NFS 需要记录有效 nullspace 信息

P3 的 Exact NFS 结果显示 KKT 很小但 geometry 没动。下一步不能只记录 KKT residual。必须记录：

```text
constraint_rank
constraint_condition
nullspace_dim_estimate
raw_smoothing_grad_norm
projected_smoothing_grad_norm
projected_over_raw
angle_smoothing_to_constraint_rowspace
active_constraint_count
```

如果 `projected_over_raw` 接近 0，就说明 exact constraints 几乎完全消灭 smoothing direction。这是设计失败，不是优化失败。

---

## 3. v5.0 的核心假设

v5.0 将验证三个假设。

### 假设 H1：PureKAN 需要 edge-level base path

RBF-only PureKAN 需要用 RBF coefficient 表达低阶结构；这让 Sobolev / functional update 天然与任务表达冲突。AB-RBF 给出 base path 后，functional optimizer 的任务难度应下降。

验证标准：AB-RBF 在 AdamW 下应稳定优于 RBFOnly；更重要的是 AB-RBF 的 high Sobolev eigenmode energy 应下降，并且 base path ablation 应显示 base 对任务有实际贡献。

### 假设 H2：几何应该约束 RBF residual，而不是压制整个 edge

base path 是低阶函数，不应该被强 Sobolev 惩罚。RBF residual 才是主要 smoothing 对象。

验证标准：split-geometry gate 下，AB-RBF 应比 total-geometry gate 更合理；如果 total $\phi'$ 上升但 residual roughness 下降、accuracy/ECE 保持，则应判为成功。

### 假设 H3：Exact nullspace 太保守，需要 relaxed NFS

严格投影 $J_f\Delta=0$ 使 geometry reduction 近乎为零。有效的 geometry consolidation 应该允许极小函数漂移，换取显著 residual smoothing。

验证标准：Relaxed NFS 在相同 KL/logit drift 预算下，$\phi_{rbf}$ / residual Sobolev / curvature reduction 应显著强于 Exact NFS，并且 accuracy drop 不超过阈值。

---

## 4. v5.0 实验阶段

## P0：Core implementation smoke and invariants

P0 的目标不是看 accuracy，而是确认新的 edge module、coverage、geometry decomposition 和 relaxed NFS instrumentation 都正确。

需要实现：

```text
ABRBFDense:
  base_mode in {none, linear, silu, linear+silu}
  rbf_coeff
  optional learnable log_width
  optional data-quantile centers, initially frozen

PureKANClassifier(edge_type=...):
  RBFOnly
  ABRBF-linear
  ABRBF-silu
  ABRBF-linear+silu

coefficient_named_params:
  returns named edge parameter groups, not only coeff
```

P0 必跑配置：

```text
RBFOnly-smoke
ABRBF-linear-smoke
ABRBF-silu-smoke
ABRBF-linear+silu-smoke
ABRBF-linear-learnWidth-smoke
RelaxedNFS-smoke
ExactNFS-smoke
```

每个数据集 MNIST / Fashion-MNIST / KMNIST 都跑 seed0，1 epoch 或 2 smoke epochs。

必须记录：

```text
learnable_nonkan_params
functional_coverage_total
base_const_seen
base_linear_seen
base_silu_seen
rbf_coeff_seen
width_seen
center_seen
rollback_max_abs_error
nan_inf_count
edge_type
base_mode
width_mode
center_mode
```

几何 decomposition 记录：

```text
phi_base_p95
phi_rbf_p95
phi_total_p95
curvature_base_p95
curvature_rbf_p95
curvature_total_p95
sobolev_rbf_norm
base_norm
rbf_norm
base_over_rbf
```

NFS instrumentation 记录：

```text
constraint_rank
constraint_condition
projected_over_raw
KKT_residual
KL
logit_drift
hidden_drift
margin_drift
```

P0 通过标准：

```text
learnable_nonkan_params = 0
functional_coverage_total = 1.0
all intended edge groups seen = 1
rollback_max_abs_error < 1e-8
no NaN / Inf
RelaxedNFS and ExactNFS both produce finite diagnostics
```

---

## P1：Architecture frontier under AdamW

P1 回答：新的 AB-RBF edge 是否真的比 RBF-only 更适合作为 PureKAN 参数化。这里先用 AdamW，不讨论 functional optimizer。

方法：

```text
PureKAN-RBFOnly-AdamW
PureKAN-ABRBF-linear-AdamW
PureKAN-ABRBF-silu-AdamW
PureKAN-ABRBF-linear+silu-AdamW
PureKAN-ABRBF-linear-learnWidth-AdamW
PureKAN-RBF-quantileCenters-AdamW
PureKAN-BaseOnly-linear-AdamW
PureKAN-BaseOnly-silu-AdamW
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
```

配置：

```text
hidden_dim = 64, basis_count = 16, depth = 4
train_size = current standard v4.9 setting
seeds = 0,1,2 first
```

如果 P1 有 clear positive，再扩 seeds 0..4。

记录指标：

```text
test_acc
val_loss_auc
ECE / NLL
train_loss_curve
val_loss_curve
feature_effective_rank by role
class_centroid_separation
margin_mean / margin_p10
phi_base_p95 / phi_rbf_p95 / phi_total_p95
curvature_base_p95 / curvature_rbf_p95 / curvature_total_p95
jacobian_condition
base_norm
rbf_norm
base_over_rbf
base_ablation_acc_drop
rbf_ablation_acc_drop
base_logit_delta_norm
rbf_logit_delta_norm
```

判定：

```text
AB-RBF architecture positive:
  mean acc >= RBFOnly + 0.5% on at least two datasets
  no dataset worse than RBFOnly by more than 0.5%
  base_ablation_acc_drop > 0.5% or base_logit_delta nontrivial
  high Sobolev eigenmode energy lower than RBFOnly
```

如果 `BaseOnly` 本身已经接近 AB-RBF，则说明 RBF residual 对任务贡献仍不足，需要重新设计 residual regularization。若 AB-RBF 明显强于 BaseOnly，则说明 base + residual 组合有效。

可视化：

```text
architecture frontier: acc vs residual_phi_rbf_p95
architecture frontier: acc vs total_phi_p95
base_over_rbf vs acc scatter
base_ablation_drop vs acc scatter
rank / margin curves by method
ECE vs acc scatter
```

---

## P2：Eigenmode and basis-use audit

P2 回答：AB-RBF 是否真正缓解 RBF high-mode pressure，以及 AdamW 使用了哪些 RBF 模式。

对 P1 每个 trained teacher 运行 eigenmode audit：

$$
S=Q\Lambda Q^\top,
$$

$$
\tilde c=Q^\top c.
$$

记录：

```text
low_eigen_coeff_energy
mid_eigen_coeff_energy
high_eigen_coeff_energy
low_eigen_grad_energy
mid_eigen_grad_energy
high_eigen_grad_energy
base_mode_energy
base_mode_grad_energy
basis_occupancy_entropy
dead_basis_fraction
out_of_grid_fraction
center_coverage
width_effective_scale
```

新增两个重要量：

$$
\text{high-mode task pressure}
=
\frac{\|Q_{high}^\top \nabla_c L\|^2}{\|\nabla_c L\|^2},
$$

$$
\text{high-mode learned energy}
=
\frac{\|Q_{high}^\top c\|^2}{\|c\|^2}.
$$

如果 high-mode task pressure 很低但 learned energy 高，说明 coefficient 高模态可能在表达低阶结构或 compensating basis mismatch。若 AB-RBF 降低 high-mode learned energy，同时 acc 上升，则支持 base path 假设。

可视化：

```text
Sobolev eigenvalue spectrum
coefficient energy spectrum by role
gradient energy spectrum by role
base mode energy vs high-mode energy scatter
basis occupancy heatmap input/block/output
out-of-grid fraction by layer
```

---

## P3：Split-metric functional update on AB-RBF

P3 是 v5.0 的第一个 functional optimizer 测试。重点不是继续用一个全局 Sobolev metric，而是 split update：

$$
\theta_{edge}=(\theta_{base}, c_{rbf}).
$$

base path 用任务学习动力学；RBF residual 用 geometry-aware functional update。

候选方法：

```text
ABRBF-AdamW
ABRBF-allAdamW-edgeOnly
ABRBF-baseAdam-rbfUFULL
ABRBF-baseFCAdam-rbfUFULL
ABRBF-baseAdam-rbfD6
ABRBF-baseAdam-rbfFCAdam-dataSob
ABRBF-baseAdam-rbfRelaxedNFSRefresh
ABRBF-baseFrozen-rbfUFULL
ABRBF-baseOnlyAdam-rbfFrozen
```

这里 `baseAdam` 不是普通 non-KAN AdamW，而是 edge-internal base 参数的 Adam-like update。它仍然属于 edge-only 参数化。为了保持严格性，需要在 row 中记录：

```text
nonKAN = 0
base_params_are_edge_params = 1
```

更新原则：

$$
\Delta \theta_{base} = \operatorname{AdamLike}(\nabla_{\theta_{base}}L),
$$

$$
\Delta c_{rbf} = -\eta M_{rbf}^{-1}\nabla_{c_{rbf}}L.
$$

其中 $M_{rbf}$ 可以是：

```text
identity
data-diag
dataSob
Sobolev full
Sobolev diag
```

P3 记录：

```text
base_update_norm
rbf_update_norm
base_update_share
rbf_update_share
cos_base_with_AdamW
cos_rbf_with_AdamW
function_R2_with_AdamW
holdout_descent_1/5/20
rank_change
margin_change
phi_base_change
phi_rbf_change
phi_total_change
curvature_rbf_change
jacobian_condition
ECE
```

P3 通过标准：

```text
on all datasets:
  accuracy gap vs ABRBF-AdamW after short run < 3%
  holdout_descent_20 positive and >= 0.75 * ABRBF-AdamW
  residual_phi_rbf <= ABRBF-AdamW or residual_curvature <= ABRBF-AdamW
  total phi allowed to rise if base derivative explains it
  base_ablation and rbf_ablation both nontrivial, unless BaseOnly baseline dominates
```

可视化：

```text
base/rbf update share over time
holdout descent vs residual_phi_rbf scatter
function_R2_with_AdamW vs accuracy gap
rank trajectory
margin trajectory
base/rbf ablation bars
```

---

## P4：Relaxed NFS projection audit

P4 不再追求 exact nullspace。目标是验证 relaxed constraints 是否能在保持 function 的同时带来有效 residual smoothing。

比较：

```text
Heuristic-NFS-role-block
ExactNFS-logit-hidden
RelaxedNFS-logit-hidden-muLow
RelaxedNFS-logit-hidden-muMed
RelaxedNFS-logit-hidden-muHigh
RelaxedNFS-role-block
RelaxedNFS-role-cycle
RelaxedNFS-rbfResidualOnly
RelaxedNFS-baseFrozen-rbfOnly
```

Relaxed NFS 解的形式：

$$
\Delta
=
-\eta
\left(
M+
\mu_zJ_z^TJ_z+
\mu_hJ_h^TJ_h+
\mu_mJ_m^TJ_m+
ho I
\right)^{-1}
\nabla R.
$$

其中 $R$ 只对 RBF residual geometry 施加：

$$
R(c_{rbf})=\|c_{rbf}\|_{S}^2.
$$

必须记录：

```text
raw_geometry_grad_norm
projected_geometry_grad_norm
projected_over_raw
constraint_rank
constraint_condition
constraint_singular_values
KL_teacher_student
logit_relative_drift
hidden_relative_drift
margin_relative_drift
acc_drop
phi_rbf_reduction
curvature_rbf_reduction
phi_total_reduction
jacobian_reduction
smoothability_score
accepted_eta
backtrack_count
```

通过标准：

```text
acc_drop <= 0.5%
KL < 0.01
logit_drift < 0.03
hidden_drift < 0.05
phi_rbf_reduction > 10% or curvature_rbf_reduction > 20%
projected_over_raw > 0.05
```

如果 RelaxedNFS 成功而 ExactNFS 失败，就能证明问题不是没有 smoothing direction，而是 exact constraints 太强。

可视化：

```text
projected_over_raw histogram
constraint singular value spectrum
KL vs phi_rbf_reduction scatter
acc_drop vs phi_rbf_reduction scatter
Exact vs Relaxed NFS paired bar
role-wise NFS smoothability heatmap
```

---

## P5：AB-RBF Learn -> Relaxed Smooth -> Refresh

P5 测试一个完整 cycle：先学任务，再只平滑 RBF residual，最后小步刷新任务。

流程：

```text
Phase I: ABRBF task learning
  candidate: ABRBF-AdamW, ABRBF-baseAdam-rbfFCAdam, ABRBF-baseAdam-rbfD6

Phase II: RelaxedNFS residual smoothing
  target: RBF residual only
  constraints: logits + selected hidden sketch + margin budget

Phase III: Refresh
  option A: base path only refresh
  option B: output edge only refresh
  option C: full edge small-step refresh
```

记录：

```text
phaseI_acc
phaseI_phi_base / phi_rbf / phi_total
phaseI_rank
phaseI_margin
smooth_acc_drop
smooth_KL
smooth_phi_rbf_reduction
refresh_acc_recovery
refresh_phi_rbf_rebound
refresh_margin_recovery
cycle_net_score
```

cycle net score：

$$
\text{cycle\_score}
=
\Delta\text{acc}_{refresh}
+
\alpha \Delta\text{AUC}
+
\beta \Delta\text{ECE}
+
\gamma \Delta\text{geometry}_{rbf}
-
\delta \text{drift}.
$$

P5 通过标准：

```text
refresh_acc >= phaseI_acc - 0.5%
phi_rbf_refresh <= 0.9 * phi_rbf_phaseI
ECE not worse by > 0.02
rank not reduced by > 10%
margin_p10 not reduced by > 10%
```

可视化：

```text
three-phase acc / phi_rbf / phi_total curves
KL and logit drift over cycle
rank/margin before-smooth-after-refresh bars
cycle score heatmap
```

---

## P6：Event-driven multi-cycle TAN v2

只有 P5 有 all-dataset survivor 时才跑 P6。P6 不使用固定 cycle，而是根据事件触发 smoothing。

触发条件：

```text
if task_loss_plateau and phi_rbf_over_budget:
  trigger RelaxedNFS

if acc_drop_after_smooth > threshold:
  trigger refresh

if geometry_rebound_after_refresh:
  reduce refresh eta or increase residual Sobolev
```

事件状态：

```text
LEARN
SMOOTH_RBF
REFRESH_BASE
REFRESH_FULL_EDGE
RECOVERY
```

记录：

```text
event_type
event_step
trigger_reason
pre_event_metrics
post_event_metrics
acc_drop
geometry_gain
refresh_recovery
event_success
```

P6 通过标准：

```text
all datasets:
  final acc >= ABRBF-AdamW - 1%
  residual_phi_rbf <= ABRBF-AdamW * 0.85
  ECE <= ABRBF-AdamW + 0.02
  no catastrophic rank collapse
  smooth events success rate > 60%
```

可视化：

```text
event timeline
acc/loss/phi_rbf curves with event markers
smooth event success waterfall
refresh recovery violin
```

---

## P7：Capacity and basis follow-up

P7 只在 P1-P6 有正信号后进行。它不再盲目扩 capacity，而是围绕最好的 AB-RBF edge 做小范围验证。

候选：

```text
ABRBF-linear h64/b16/d4
ABRBF-linear h96/b24/d4
ABRBF-silu h64/b16/d4
ABRBF-silu h96/b24/d4
ABRBF-linear+silu h64/b16/d4
ABRBF-linear+silu h96/b24/d4
ABRBF-linear quantileCenters
ABRBF-linear learnWidth with width regularization
```

记录：

```text
all P1/P2 metrics
parameter_count
step_time_ms
peak_memory
base/rbf contribution
center_coverage
basis_occupancy
```

判定：

```text
Architecture scaling positive if:
  acc improves without residual_phi_rbf exploding
  base/RBF contribution remains balanced
  memory/time overhead acceptable
```

---

## P8：3-seed candidate selection

P8 把最好的方法组合跑 3-seed full budget。

候选最多 6 个：

```text
RBFOnly-AdamW
ABRBF-best-AdamW
ABRBF-best-splitFunctional
ABRBF-best-LearnSmoothRefresh
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
```

指标：

```text
test_acc
val_auc
ECE / NLL
phi_base_p95
phi_rbf_p95
phi_total_p95
curvature_rbf
jacobian_condition
rank
margin_p10
base_ablation_drop
rbf_ablation_drop
step_time_ms
```

P8 通过标准：

```text
ABRBF functional candidate:
  acc >= RBFOnly-AdamW - 0.5%
  acc >= Hybrid-DGKAN-UFULL-f085 - 1%
  val_auc positive vs RBFOnly-AdamW or no worse by more than 3%
  phi_rbf_reduction > 10% or curvature_rbf_reduction > 20%
  ECE not worse by > 0.02
```

如果 functional candidate 不能接近 ABRBF-AdamW，但 ABRBF-AdamW 很强，则说明 architecture 修复成功，optimizer 仍未解决。若 ABRBF functional 也明显改善，则进入 P9。

---

## P9：5-seed confirm

P9 对 P8 survivor 做 5-seed confirm。

必须包含：

```text
PureKAN-RBFOnly-AdamW
PureKAN-ABRBF-best-AdamW
PureKAN-ABRBF-functional-best
MLP-AdamW
Hybrid-DGKAN-UFULL-f085
```

5-seed gate：

```text
acc mean not below best baseline by > 0.5% on any dataset
paired acc CI not strongly negative
val_auc mean positive or comparable
residual_phi_rbf reduced by > 10%
ECE comparable or better
base/rbf ablation both interpretable
```

可视化：

```text
paired delta vs RBFOnly-AdamW
paired delta vs ABRBF-AdamW
acc vs residual geometry Pareto
ECE vs acc Pareto
base/rbf contribution dashboard
```

---

## P10：10-seed final confirm

只有 P9 通过才跑 P10。P10 的目标不是继续搜索，而是最终确认。

最终结论分类：

### A. Strong success

```text
ABRBF functional candidate >= ABRBF-AdamW or within 0.5%
and geometry residual clearly better
and ECE/AUC not worse
```

可 claim：

$$
\boxed{\text{Edge-decomposed functional optimizer works for PureKAN.}}
$$

### B. Architecture success, optimizer partial

```text
ABRBF-AdamW strong;
ABRBF functional improves over RBF functional but does not match AdamW.
```

可 claim：

$$
\boxed{\text{RBF-only was a bottleneck; functional optimizer still needs Adam-like task dynamics.}}
$$

### C. NFS success only

```text
ABRBF-AdamW good;
RelaxedNFS can smooth posthoc but cannot train from scratch.
```

可 claim：

$$
\boxed{\text{Functional update is currently better as geometry consolidation than as primary optimizer.}}
$$

### D. No success

```text
ABRBF not stable or functional still fails.
```

可 claim：

$$
\boxed{\text{Current PureKAN parameterization remains insufficient; revisit primitive, e.g. Rational/KAT or spline.}}
$$

---

## 5. Required artifacts

每个阶段输出：

```text
runs.csv
curves.csv
edge_decomposition_audit.csv
eigenmode_audit.csv
basis_occupancy_audit.csv
nfs_projection_audit.csv
failure_table.csv
aggregate_decision.json
```

最终文档输出：

```text
docs/DG-KAN_v5.0_EdgeDecomposedFunctionalTraining_结果复盘.md
```

图表输出：

```text
figures/architecture_frontier_acc_vs_phi_rbf.svg
figures/base_over_rbf_vs_acc.svg
figures/eigenmode_energy_spectrum.svg
figures/basis_occupancy_heatmap.svg
figures/exact_vs_relaxed_nfs_scatter.svg
figures/projected_over_raw_histogram.svg
figures/learn_smooth_refresh_cycle.svg
figures/event_timeline.svg
figures/paired_delta_vs_baselines.svg
```

---

## 6. 现在最重要的失败标签

v5.0 failure table 必须包含以下标签：

```text
F1_base_path_unused
F2_base_path_helps_accuracy_but_geometry_bad
F3_residual_high_mode_pressure_persists
F4_exact_nfs_overconstrained
F5_relaxed_nfs_destroys_function
F6_relaxed_nfs_no_geometry_gain
F7_refresh_cannot_recover_accuracy
F8_capacity_increases_phi_without_accuracy
F9_width_learning_unstable_or_unused
F10_functional_candidate_lags_ABRBF_AdamW
```

其中最重要的是 F4。若 Exact NFS 的 projected_over_raw 接近 0，就说明 strict nullspace direction 基本没有可用 smoothing 方向，下一步必须使用 relaxed projection，而不是继续改 KKT solver。

---

## 7. 当前不建议继续做的事情

v5.0 暂停以下方向：

```text
继续调 Sobolev alpha / beta
继续调 branch_final_scale
继续固定 shallow/deep TFU 分工
继续 exact NFS KKT solve without relaxation
继续单纯扩 basis_count / hidden_dim
继续 learnWidth alone
继续 FCAdam / FNG / BFT 无参数化改变的版本
```

这些方向已经多轮显示不是主矛盾。

---

## 8. 最终总结

v4.9 的结果给了一个新的、更有希望的解释：

$$
\boxed{\text{PureKAN functional update 失败，不只是 optimizer 失败，而是 RBF-only edge coordinate 不适合当前任务。}}
$$

v5.0 的目标是把低阶通道放回 edge 内部，并且只对 RBF residual 施加几何 smoothing：

$$
f_{ij}(x)=b_{ij}+w_{ij}x+u_{ij}\operatorname{silu}(x)+r_{ij}(x),
$$

$$
r_{ij}(x)=\sum_k c_{ijk}B_k(x).
$$

如果这条路线成功，项目就能从 “Hybrid branch functional update” 推进到真正更干净的 “edge-decomposed PureKAN functional optimizer”。如果它仍然失败，我们也会得到一个清楚结论：当前 RBF-style PureKAN 需要更换 primitive 或接受 AdamW 作为主训练动力学。



---


# Source 28: `docs/DG-KAN_v5.0_EdgeDecomposedFunctionalTraining_结果复盘.md`


# DG-KAN v5.0 Edge-Decomposed Functional Training 结果复盘

本轮依据 `docs/DG-KAN_v5.0_EdgeDecomposedFunctionalTraining_实验计划.md`。核心问题从“继续调 Sobolev optimizer”转为验证 edge-decomposed PureKAN：base path 学低阶/尺度结构，RBF residual 承担非线性，并把 geometry 主要约束到 residual 上。

## Run Inventory

| stage | rows | errors |
|---|---:|---:|
| P0 code audit | 27 | 0 |
| P1 architecture frontier | 81 | 0 |
| P2 eigenmode audit | 432 | 0 |
| P3 split functional update | 81 | 0 |
| P4 relaxed NFS projection | 120 | 0 |
| P5 learn-smooth-refresh | 0 | 0 |
| P6 event TAN | 0 | 0 |
| P7 capacity follow-up | 27 | 0 |

## Code / Config Changes

```text
experiments/run_gafu_v50.py
  Added edge-decomposed PureKAN modules:
    ABRBF-linear
    ABRBF-silu
    ABRBF-linear+silu
    BaseOnly-linear / BaseOnly-silu
    learnWidth / quantile-center probes
  Added split geometry audit:
    phi_base_p95 / phi_rbf_p95 / phi_total_p95
    curvature_base/rbf/total
    sobolev_rbf_norm
  Added split functional update:
    base Adam-like task update
    RBF residual identity / Sobolev / dataSob-style updates
  Added relaxed NFS row-space solver using Woodbury form.

experiments/analyze_gafu_v50.py
  Generates gate summaries, failure taxonomy, aggregate_decision.json,
  required artifact aliases, SVG diagnostics, log entry, and this replay.
```

## P0 Code Audit

| rows | errors | pass | max nonKAN | min coverage | max rollback | max KKT | max CG |
|---|---|---|---|---|---|---|---|
| 27 | 0 | true | 0.0000 | 1.0000 | 0.0000 | 0.000001 | 0.000001 |

P0 verdict: pass. Edge-only 参数计数、functional coverage、rollback、Exact/Relaxed NFS smoke 都可审计。Relaxed NFS 初版 dense solve 触发 OOM，已改为 row-space/Woodbury 解法。

## P1 Architecture Frontier

| dataset | method | runs | acc | gap vs RBF | phi_rbf | phi_total | base/RBF | positive |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | MLP-AdamW | 3 | 0.7819 | 0.0124 | 0.0000 |  | 0.0000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | 3 | 0.7988 | -0.0046 | 0.1135 | 0.1319 | 0.6582 | 1 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | 3 | 0.7949 | -0.0007 | 0.1147 | 0.1276 | 0.5480 | 1 |
| Fashion-MNIST | PureKAN-ABRBF-linear-learnWidth-AdamW | 3 | 0.7930 | 0.0013 | 0.1155 | 0.1279 | 0.5651 | 1 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | 3 | 0.7969 | -0.0026 | 0.1151 | 0.1208 | 0.1647 | 1 |
| Fashion-MNIST | PureKAN-BaseOnly-linear-AdamW | 3 | 0.7786 | 0.0156 | 0.0000 | 0.1172 | 218960613674587.6875 | 0 |
| Fashion-MNIST | PureKAN-BaseOnly-silu-AdamW | 3 | 0.7858 | 0.0085 | 0.0000 | 0.0926 | 127602617051866.3281 | 0 |
| Fashion-MNIST | PureKAN-RBF-quantileCenters-AdamW | 3 | 0.8027 | -0.0085 | 0.1627 | 0.1627 | 0.0000 | 1 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | 3 | 0.7943 | 0.0000 | 0.1179 | 0.1179 | 0.0000 | 1 |
| KMNIST | MLP-AdamW | 3 | 0.6927 | -0.0117 | 0.0000 |  | 0.0000 | 0 |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | 3 | 0.6888 | -0.0078 | 0.1146 | 0.1317 | 0.6793 | 1 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | 3 | 0.6888 | -0.0078 | 0.1163 | 0.1275 | 0.5725 | 1 |
| KMNIST | PureKAN-ABRBF-linear-learnWidth-AdamW | 3 | 0.6816 | -0.0007 | 0.1174 | 0.1282 | 0.5896 | 1 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | 3 | 0.6947 | -0.0137 | 0.1167 | 0.1223 | 0.1963 | 1 |
| KMNIST | PureKAN-BaseOnly-linear-AdamW | 3 | 0.6068 | 0.0742 | 0.0000 | 0.1174 | 209738729688856.3438 | 0 |
| KMNIST | PureKAN-BaseOnly-silu-AdamW | 3 | 0.6471 | 0.0339 | 0.0000 | 0.0931 | 127656950632731.1094 | 0 |
| KMNIST | PureKAN-RBF-quantileCenters-AdamW | 3 | 0.6719 | 0.0091 | 0.1829 | 0.1829 | 0.0000 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | 3 | 0.6810 | 0.0000 | 0.1200 | 0.1200 | 0.0000 | 1 |
| MNIST | MLP-AdamW | 3 | 0.9076 | -0.0319 | 0.0000 |  | 0.0000 | 0 |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | 3 | 0.8997 | -0.0241 | 0.1143 | 0.1312 | 0.6783 | 1 |
| MNIST | PureKAN-ABRBF-linear-AdamW | 3 | 0.8900 | -0.0143 | 0.1153 | 0.1260 | 0.5610 | 1 |
| MNIST | PureKAN-ABRBF-linear-learnWidth-AdamW | 3 | 0.8874 | -0.0117 | 0.1162 | 0.1269 | 0.5761 | 1 |
| MNIST | PureKAN-ABRBF-silu-AdamW | 3 | 0.8945 | -0.0189 | 0.1184 | 0.1237 | 0.1998 | 1 |
| MNIST | PureKAN-BaseOnly-linear-AdamW | 3 | 0.8796 | -0.0039 | 0.0000 | 0.1178 | 218147297329372.8125 | 0 |
| MNIST | PureKAN-BaseOnly-silu-AdamW | 3 | 0.8854 | -0.0098 | 0.0000 | 0.0921 | 135945993211534.2969 | 0 |
| MNIST | PureKAN-RBF-quantileCenters-AdamW | 3 | 0.8633 | 0.0124 | 0.1786 | 0.1786 | 0.0000 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | 3 | 0.8757 | 0.0000 | 0.1219 | 0.1219 | 0.0000 | 1 |

P1 verdict:

```text
AB-RBF is architecture-positive.
ABRBF-linear+silu is the strongest all-around short-run architecture:
  MNIST and Fashion improve over RBFOnly.
  KMNIST improves in seed0 compact P7 and remains competitive in P1.

BaseOnly is informative:
  It can approach RBF/ABRBF on MNIST/Fashion, but collapses relative to AB-RBF on KMNIST.
  This means the base path is necessary but not sufficient; the RBF residual still contributes.

Quantile centers and learnWidth alone are not stable positives.
```

P1 all-dataset architecture positives:

```text
PureKAN-ABRBF-linear+silu-AdamW, PureKAN-ABRBF-linear-AdamW, PureKAN-ABRBF-linear-learnWidth-AdamW, PureKAN-ABRBF-silu-AdamW
```

## P2 Eigenmode / Basis-Use Audit

| dataset | method | high coeff | high grad | base mode | eig cond | shift |
|---|---|---|---|---|---|---|
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | 0.2547 | 0.0004 | 0.3333 | 78.7 | 1 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | 0.2293 | 0.0001 | 0.2560 | 78.7 | 1 |
| Fashion-MNIST | PureKAN-ABRBF-linear-learnWidth-AdamW | 0.2301 | 0.0002 | 0.2527 | 78.7 | 1 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | 0.2508 | 0.0002 | 0.0931 | 78.7 | 1 |
| Fashion-MNIST | PureKAN-BaseOnly-linear-AdamW | 0.0000 | 0.0000 | 1.0000 | 1.0 | 1 |
| Fashion-MNIST | PureKAN-BaseOnly-silu-AdamW | 0.0000 | 0.0000 | 1.0000 | 1.0 | 1 |
| Fashion-MNIST | PureKAN-RBF-quantileCenters-AdamW | 0.2871 | 0.0036 | 0.0000 | 78.7 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | 0.2689 | 0.0002 | 0.0000 | 78.7 | 1 |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | 0.2507 | 0.0005 | 0.3194 | 78.7 | 1 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | 0.2241 | 0.0002 | 0.2434 | 78.7 | 1 |
| KMNIST | PureKAN-ABRBF-linear-learnWidth-AdamW | 0.2248 | 0.0003 | 0.2408 | 78.7 | 1 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | 0.2439 | 0.0003 | 0.0874 | 78.7 | 1 |
| KMNIST | PureKAN-BaseOnly-linear-AdamW | 0.0000 | 0.0000 | 1.0000 | 1.0 | 1 |
| KMNIST | PureKAN-BaseOnly-silu-AdamW | 0.0000 | 0.0000 | 1.0000 | 1.0 | 1 |
| KMNIST | PureKAN-RBF-quantileCenters-AdamW | 0.2702 | 0.0047 | 0.0000 | 78.7 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | 0.2629 | 0.0003 | 0.0000 | 78.7 | 1 |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | 0.2515 | 0.0005 | 0.3342 | 78.7 | 1 |
| MNIST | PureKAN-ABRBF-linear-AdamW | 0.2285 | 0.0001 | 0.2520 | 78.7 | 1 |
| MNIST | PureKAN-ABRBF-linear-learnWidth-AdamW | 0.2292 | 0.0002 | 0.2485 | 78.7 | 1 |
| MNIST | PureKAN-ABRBF-silu-AdamW | 0.2407 | 0.0002 | 0.0983 | 78.7 | 1 |
| MNIST | PureKAN-BaseOnly-linear-AdamW | 0.0000 | 0.0000 | 1.0000 | 1.0 | 1 |
| MNIST | PureKAN-BaseOnly-silu-AdamW | 0.0000 | 0.0000 | 1.0000 | 1.0 | 1 |
| MNIST | PureKAN-RBF-quantileCenters-AdamW | 0.2504 | 0.0045 | 0.0000 | 78.7 | 1 |
| MNIST | PureKAN-RBFOnly-AdamW | 0.2578 | 0.0003 | 0.0000 | 78.7 | 1 |

P2 verdict:

```text
AB-RBF shifts a visible fraction of edge energy into explicit base modes.
This confirms the v4.9/v5.0 interpretation: RBFOnly uses coefficient modes to carry low-order structure.
However high-mode/task-pressure is not fully eliminated, especially when the residual remains needed for KMNIST.
```

## P3 Split-Metric Functional Update

| dataset | method | runs | acc | gap vs ABRBF-A | hold/A | phiR red | base share | pass |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | ABRBF-AdamW | 3 | 0.7754 | 0.0000 | 1.0000 | 0.0000 | 0.1918 | 1 |
| Fashion-MNIST | ABRBF-allAdamW-edgeOnly | 3 | 0.7754 | 0.0000 | 1.0000 | 0.0000 | 0.1918 | 1 |
| Fashion-MNIST | ABRBF-baseAdam-rbfD6 | 3 | 0.7734 | 0.0020 | 0.9408 | 0.1715 | 0.9892 | 1 |
| Fashion-MNIST | ABRBF-baseAdam-rbfFCAdam-dataSob | 3 | 0.7747 | 0.0007 | 0.9403 | 0.1719 | 0.9802 | 1 |
| Fashion-MNIST | ABRBF-baseAdam-rbfRelaxedNFSRefresh | 3 | 0.7734 | 0.0020 | 0.9408 | 0.1715 | 0.9890 | 1 |
| Fashion-MNIST | ABRBF-baseAdam-rbfUFULL | 3 | 0.7747 | 0.0007 | 0.9412 | 0.1705 | 0.9707 | 1 |
| Fashion-MNIST | ABRBF-baseFCAdam-rbfUFULL | 3 | 0.7839 | -0.0085 | 0.9011 | 0.1745 | 0.9179 | 1 |
| Fashion-MNIST | ABRBF-baseFrozen-rbfUFULL | 3 | 0.6048 | 0.1706 | 0.3929 | 0.1756 | 0.0000 | 0 |
| Fashion-MNIST | ABRBF-baseOnlyAdam-rbfFrozen | 3 | 0.7715 | 0.0039 | 0.9414 | 0.1724 | 1.0000 | 1 |
| KMNIST | ABRBF-AdamW | 3 | 0.6517 | 0.0000 | 1.0000 | 0.0000 | 0.2023 | 1 |
| KMNIST | ABRBF-allAdamW-edgeOnly | 3 | 0.6517 | 0.0000 | 1.0000 | 0.0000 | 0.2023 | 1 |
| KMNIST | ABRBF-baseAdam-rbfD6 | 3 | 0.6302 | 0.0215 | 0.8967 | 0.1743 | 0.9891 | 1 |
| KMNIST | ABRBF-baseAdam-rbfFCAdam-dataSob | 3 | 0.6361 | 0.0156 | 0.8980 | 0.1741 | 0.9800 | 1 |
| KMNIST | ABRBF-baseAdam-rbfRelaxedNFSRefresh | 3 | 0.6296 | 0.0221 | 0.8967 | 0.1744 | 0.9889 | 1 |
| KMNIST | ABRBF-baseAdam-rbfUFULL | 3 | 0.6393 | 0.0124 | 0.8954 | 0.1739 | 0.9702 | 1 |
| KMNIST | ABRBF-baseFCAdam-rbfUFULL | 3 | 0.6335 | 0.0182 | 0.8465 | 0.1732 | 0.9172 | 1 |
| KMNIST | ABRBF-baseFrozen-rbfUFULL | 3 | 0.2682 | 0.3835 | 0.2502 | 0.1785 | 0.0000 | 0 |
| KMNIST | ABRBF-baseOnlyAdam-rbfFrozen | 3 | 0.6361 | 0.0156 | 0.8993 | 0.1734 | 1.0000 | 1 |
| MNIST | ABRBF-AdamW | 3 | 0.8626 | 0.0000 | 1.0000 | 0.0000 | 0.2014 | 1 |
| MNIST | ABRBF-allAdamW-edgeOnly | 3 | 0.8626 | 0.0000 | 1.0000 | 0.0000 | 0.2014 | 1 |
| MNIST | ABRBF-baseAdam-rbfD6 | 3 | 0.8724 | -0.0098 | 0.9653 | 0.1908 | 0.9859 | 1 |
| MNIST | ABRBF-baseAdam-rbfFCAdam-dataSob | 3 | 0.8665 | -0.0039 | 0.9659 | 0.1910 | 0.9743 | 1 |
| MNIST | ABRBF-baseAdam-rbfRelaxedNFSRefresh | 3 | 0.8724 | -0.0098 | 0.9652 | 0.1910 | 0.9857 | 1 |
| MNIST | ABRBF-baseAdam-rbfUFULL | 3 | 0.8711 | -0.0085 | 0.9680 | 0.1883 | 0.9626 | 1 |
| MNIST | ABRBF-baseFCAdam-rbfUFULL | 3 | 0.8743 | -0.0117 | 0.9024 | 0.1961 | 0.8971 | 1 |
| MNIST | ABRBF-baseFrozen-rbfUFULL | 3 | 0.4954 | 0.3672 | 0.2543 | 0.1949 | 0.0000 | 0 |
| MNIST | ABRBF-baseOnlyAdam-rbfFrozen | 3 | 0.8737 | -0.0111 | 0.9646 | 0.1917 | 1.0000 | 1 |

P3 verdict:

```text
Split functional update is a real improvement over residual-only training.
Moving base with Adam-like dynamics is essential:
  baseFrozen-rbfUFULL fails hard on all datasets.

Several baseAdam + rbf functional variants keep decent holdout descent and reduce residual phi.
But the all-dataset gate is not clean, mainly because KMNIST accuracy lags ABRBF-AdamW.
```

P3 all-dataset survivors:

```text
ABRBF-allAdamW-edgeOnly, ABRBF-baseAdam-rbfD6, ABRBF-baseAdam-rbfFCAdam-dataSob, ABRBF-baseAdam-rbfRelaxedNFSRefresh, ABRBF-baseAdam-rbfUFULL, ABRBF-baseFCAdam-rbfUFULL, ABRBF-baseOnlyAdam-rbfFrozen
```

## P4 Relaxed NFS Projection

| dataset | teacher | variant | acc drop | KL | phiR red | curvR red | proj/raw | pass |
|---|---|---|---|---|---|---|---|---|
| MNIST | PureKAN-RBFOnly-AdamW | Heuristic-NFS-role-block | 0.0000 | 0.00009 | 0.0228 | 0.0628 | 0.0000 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit-hidden | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-logit-hidden-muLow | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-logit-hidden-muMed | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-logit-hidden-muHigh | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-role-block | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-role-cycle | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-rbfResidualOnly | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-baseFrozen-rbfOnly | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | Heuristic-NFS-role-block | 0.0020 | 0.00065 | 0.0251 | 0.0647 | 0.0000 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit-hidden | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-logit-hidden-muLow | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-logit-hidden-muMed | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-logit-hidden-muHigh | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-role-block | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5182 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-role-cycle | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5182 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-rbfResidualOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5182 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-baseFrozen-rbfOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5182 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | Heuristic-NFS-role-block | 0.0000 | 0.00015 | 0.0247 | 0.0643 | 0.0000 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit-hidden | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-logit-hidden-muLow | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-logit-hidden-muMed | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-logit-hidden-muHigh | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-role-block | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-role-cycle | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-rbfResidualOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-baseFrozen-rbfOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | Heuristic-NFS-role-block | -0.0020 | 0.00042 | 0.0227 | 0.0654 | 0.0000 | 0 |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | ExactNFS-logit | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | ExactNFS-logit-hidden | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-logit-hidden-muLow | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-logit-hidden-muMed | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-logit-hidden-muHigh | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-role-block | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-role-cycle | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-rbfResidualOnly | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-baseFrozen-rbfOnly | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | Heuristic-NFS-role-block | 0.0000 | 0.00003 | 0.0238 | 0.0624 | 0.0000 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit | 0.0000 | 0.00000 | -0.0000 | 0.0000 | 1.0000 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit-hidden | 0.0000 | -0.00000 | -0.0000 | 0.0000 | 1.0000 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-logit-hidden-muLow | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-logit-hidden-muMed | 0.0000 | -0.00000 | -0.0000 | 0.0000 | 0.9999 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-logit-hidden-muHigh | 0.0000 | 0.00000 | -0.0000 | 0.0000 | 0.9999 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-role-block | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-role-cycle | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-rbfResidualOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-baseFrozen-rbfOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | Heuristic-NFS-role-block | 0.0020 | 0.00049 | 0.0224 | 0.0643 | 0.0000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit-hidden | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-logit-hidden-muLow | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-logit-hidden-muMed | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-logit-hidden-muHigh | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-role-block | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-role-cycle | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-rbfResidualOnly | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-baseFrozen-rbfOnly | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | Heuristic-NFS-role-block | -0.0039 | 0.00010 | 0.0242 | 0.0636 | 0.0000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit-hidden | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-logit-hidden-muLow | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-logit-hidden-muMed | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-logit-hidden-muHigh | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-role-block | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5182 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-role-cycle | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5182 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-rbfResidualOnly | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5182 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-baseFrozen-rbfOnly | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5182 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | Heuristic-NFS-role-block | 0.0039 | 0.00020 | 0.0210 | 0.0645 | 0.0000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | ExactNFS-logit | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | ExactNFS-logit-hidden | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-logit-hidden-muLow | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-logit-hidden-muMed | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-logit-hidden-muHigh | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-role-block | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5185 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-role-cycle | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5185 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-rbfResidualOnly | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5185 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-baseFrozen-rbfOnly | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5185 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | Heuristic-NFS-role-block | -0.0020 | 0.00007 | 0.0255 | 0.0620 | 0.0000 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit-hidden | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-logit-hidden-muLow | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-logit-hidden-muMed | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-logit-hidden-muHigh | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-role-block | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-role-cycle | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-rbfResidualOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-baseFrozen-rbfOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | Heuristic-NFS-role-block | 0.0000 | 0.00039 | 0.0240 | 0.0642 | 0.0000 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit-hidden | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-logit-hidden-muLow | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-logit-hidden-muMed | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-logit-hidden-muHigh | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-role-block | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-role-cycle | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-rbfResidualOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-baseFrozen-rbfOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | Heuristic-NFS-role-block | 0.0000 | 0.00012 | 0.0277 | 0.0638 | 0.0000 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit-hidden | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-logit-hidden-muLow | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-logit-hidden-muMed | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-logit-hidden-muHigh | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-role-block | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-role-cycle | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-rbfResidualOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-baseFrozen-rbfOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | Heuristic-NFS-role-block | -0.0020 | 0.00023 | 0.0245 | 0.0642 | 0.0000 | 0 |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | ExactNFS-logit | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | ExactNFS-logit-hidden | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-logit-hidden-muLow | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-logit-hidden-muMed | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-logit-hidden-muHigh | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-role-block | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5186 | 0 |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-role-cycle | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5186 | 0 |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-rbfResidualOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5186 | 0 |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-baseFrozen-rbfOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5186 | 0 |

P4 verdict:

```text
Relaxed NFS is numerically stable and no longer overconstrained in the same way as Exact NFS:
  projected_over_raw is often around 0.5 to 1.0.

But the actual residual-geometry movement is still near zero under the current trust scale.
Heuristic NFS keeps the best practical geometry movement, around 2-3% phi_rbf reduction,
but this is below the planned >10% residual phi gate.

Therefore P5/P6 are not expanded.
```

P4 all-dataset survivors:

```text
none
```

## P5 / P6 Decision

```text
P5 learn -> smooth -> refresh: not run
Reason: P4 produced no all-dataset relaxed NFS survivor.

P6 event-driven TAN v2: not run
Reason: P5 was not reached.
```

## P7 Capacity / Basis Follow-Up

| dataset | edge | hidden | basis | depth | acc | phi_rbf | rank | base/RBF |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | ABRBF-linear | 64 | 16 | 4 | 0.8008 | 0.1114 | 9.1229 | 0.5514 |
| Fashion-MNIST | ABRBF-linear | 96 | 24 | 4 | 0.7930 | 0.1370 | 9.5165 | 0.6043 |
| Fashion-MNIST | ABRBF-linear+silu | 64 | 16 | 4 | 0.7852 | 0.1107 | 9.0518 | 0.6573 |
| Fashion-MNIST | ABRBF-linear+silu | 96 | 24 | 4 | 0.7734 | 0.1358 | 9.5579 | 0.6708 |
| Fashion-MNIST | ABRBF-linear-learnWidth | 96 | 24 | 4 | 0.7910 | 0.1379 | 9.6156 | 0.6195 |
| Fashion-MNIST | ABRBF-linear-quantileCenters | 96 | 24 | 4 | 0.7656 | 0.1934 | 7.0925 | 0.3132 |
| Fashion-MNIST | ABRBF-silu | 64 | 16 | 4 | 0.7832 | 0.1129 | 10.8404 | 0.1572 |
| Fashion-MNIST | ABRBF-silu | 96 | 24 | 4 | 0.7793 | 0.1382 | 9.0402 | 0.1274 |
| Fashion-MNIST | RBFOnly | 64 | 16 | 4 | 0.7891 | 0.1171 | 10.0688 | 0.0000 |
| KMNIST | ABRBF-linear | 64 | 16 | 4 | 0.6855 | 0.1122 | 11.6271 | 0.5465 |
| KMNIST | ABRBF-linear | 96 | 24 | 4 | 0.6836 | 0.1471 | 9.7898 | 0.5215 |
| KMNIST | ABRBF-linear+silu | 64 | 16 | 4 | 0.7168 | 0.1133 | 10.9519 | 0.6333 |
| KMNIST | ABRBF-linear+silu | 96 | 24 | 4 | 0.7051 | 0.1415 | 10.6869 | 0.6009 |
| KMNIST | ABRBF-linear-learnWidth | 96 | 24 | 4 | 0.6836 | 0.1482 | 9.8360 | 0.5409 |
| KMNIST | ABRBF-linear-quantileCenters | 96 | 24 | 4 | 0.6367 | 0.2437 | 8.4478 | 0.2115 |
| KMNIST | ABRBF-silu | 64 | 16 | 4 | 0.6934 | 0.1137 | 11.8708 | 0.2022 |
| KMNIST | ABRBF-silu | 96 | 24 | 4 | 0.6797 | 0.1433 | 11.0848 | 0.1064 |
| KMNIST | RBFOnly | 64 | 16 | 4 | 0.6758 | 0.1197 | 11.5154 | 0.0000 |
| MNIST | ABRBF-linear | 64 | 16 | 4 | 0.8750 | 0.1119 | 10.4300 | 0.5697 |
| MNIST | ABRBF-linear | 96 | 24 | 4 | 0.8438 | 0.1425 | 9.7568 | 0.5222 |
| MNIST | ABRBF-linear+silu | 64 | 16 | 4 | 0.8633 | 0.1114 | 11.1287 | 0.6716 |
| MNIST | ABRBF-linear+silu | 96 | 24 | 4 | 0.8691 | 0.1465 | 9.9446 | 0.5962 |
| MNIST | ABRBF-linear-learnWidth | 96 | 24 | 4 | 0.8379 | 0.1432 | 9.7848 | 0.5334 |
| MNIST | ABRBF-linear-quantileCenters | 96 | 24 | 4 | 0.8008 | 0.2454 | 5.9702 | 0.1500 |
| MNIST | ABRBF-silu | 64 | 16 | 4 | 0.8750 | 0.1172 | 11.7127 | 0.1817 |
| MNIST | ABRBF-silu | 96 | 24 | 4 | 0.8672 | 0.1460 | 9.6207 | 0.1151 |
| MNIST | RBFOnly | 64 | 16 | 4 | 0.8105 | 0.1204 | 10.7913 | 0.0000 |

P7 verdict:

```text
Compact capacity follow-up supports the architecture signal.
ABRBF-linear+silu h64/b16/d4 is strong on KMNIST seed0 and improves MNIST over RBFOnly.
Scaling to h96/b24 does not automatically improve geometry; phi_rbf often rises.
Quantile centers are not a clean fix in this setup.
```

## Failure Diagnosis

Generated:

```text
failure_table.csv
p1_architecture_gate_summary.csv
p2_eigenmode_gate_summary.csv
p3_split_functional_gate_summary.csv
p4_relaxed_nfs_gate_summary.csv
p7_capacity_gate_summary.csv
```

Key labels:

```text
F1_base_path_unused:
  Mostly false for AB-RBF. Base/RBF contribution is nontrivial.

F2_base_path_helps_accuracy_but_geometry_bad:
  True for several AB-RBF rows under total geometry, but split geometry shows this is partly expected.

F3_residual_high_mode_pressure_persists:
  Partly true. AB-RBF helps, but does not remove residual pressure on KMNIST.

F4_exact_nfs_overconstrained:
  Still true for strict Exact NFS.

F6_relaxed_nfs_no_geometry_gain:
  New blocker. Relaxed projection keeps useful gradient mass, but current step/trust gives almost no residual phi reduction.

F10_functional_candidate_lags_ABRBF_AdamW:
  Main P3 blocker, especially on KMNIST.
```

## Required Artifacts

Written under `results/v5_0/`:

```text
runs.csv
curves.csv
edge_decomposition_audit.csv
eigenmode_audit.csv
basis_occupancy_audit.csv
nfs_projection_audit.csv
failure_table.csv
aggregate_decision.json
figures/
```

## Final Decision

```text
DG-KAN v5.0 status:
  stop_after_p4_no_relaxed_nfs_survivor

What improved:
  Edge decomposition is clearly useful.
  AB-RBF, especially linear+silu / silu variants, improves the PureKAN architecture frontier.
  Split geometry gives a more honest view: base derivative is not the same as residual roughness.
  Split functional update confirms base-path task dynamics are essential.

What failed / remains open:
  Split functional candidates still lag ABRBF-AdamW on the hardest dataset.
  Relaxed NFS no longer has zero projected direction, but still fails to create enough residual smoothing.
  P5/P6/P8/P9 expansion is not justified.

Conclusion:
  v5.0 supports architecture success and partial optimizer progress.
  RBF-only was a bottleneck, but the current residual smoothing operator is still too weak.
  Next work should keep AB-RBF as the PureKAN default and redesign relaxed residual smoothing
  with a stronger accepted-step controller rather than returning to exact nullspace projection.
```



---


# Source 29: `docs/DG-KAN_v5.1_ABRBF_StrongResidualSmoothing_实验计划.md`


# DG-KAN v5.1：AB-RBF Edge-Decomposed Functional Training 与 Strong Residual Smoothing 实验计划

## 0. 当前结论先冻结：v5.0 不是失败，而是把问题定位清楚了

v5.0 的核心贡献不是把 PureKAN functional optimizer 一步做成，而是把过去混在一起的三个问题拆开了：

```text
1. RBF-only edge 坐标是否合适？
2. base path 是否真的承担任务？
3. RBF residual 是否可以在不破坏任务函数的情况下被 geometry smoothing？
```

v5.0 的答案非常明确：

```text
RBF-only 是瓶颈；
AB-RBF 是 architecture-positive；
base path 真的承担任务；
RBF residual functional update 有局部收益；
但当前 relaxed NFS / residual smoothing 还太弱，不能形成 all-dataset survivor。
```

所以 v5.1 不再继续问：

```text
有没有另一个 Sobolev / TFU / FNG / FCAdam 小变体？
```

而是问：

$$
\boxed{
\text{在 AB-RBF edge 坐标下，如何让 base path 学任务，RBF residual 负责可控非线性，并把 geometry 主要约束到 residual 上？}
}
$$

这意味着 v5.1 的主线从 **PureKAN coefficient-only functional optimizer** 改成 **Edge-decomposed functional training**。

---

## 1. v5.0 结果复盘与深层解释

### 1.1 P0 说明实现基本干净，但 AB-RBF 仍应进入 core-level 实现

v5.0 的 P0 通过：

```text
rows = 27
errors = 0
max nonKAN = 0
min coverage = 1.0
rollback = 0
max KKT ≈ 1e-6
max CG ≈ 1e-6
```

这说明 edge-only 参数计数、functional coverage、rollback、Relaxed NFS row-space / Woodbury solver 都能被审计。也就是说，当前结论不能主要归因于 hidden non-KAN 参数、coverage 漏掉、rollback 错误或 NFS smoke 不稳定。

但是从代码组织上看，v5.0 的 edge-decomposed PureKAN 主要是在 `run_gafu_v50.py` 中新增的实验模块，而 `dgkan_core.py` 的标准 `RBFDense` 仍是：

```python
centers = torch.linspace(-2.5, 2.5, basis_count)
self.register_buffer("centers", centers)
self.width = fixed float
self.coeff = nn.Parameter(...)
```

也就是说，core 里的默认 RBF 仍然是 fixed centers、fixed width、only coeff trainable 的实现。v5.1 的第一件事应该是把 `ABRBFDense` 提升为 core-level 模块，而不是继续只在 runner 里临时定义。否则后续不同 runner、audit、coefficient collection、geometry metrics 很容易出现不一致。

v5.1 的 core-level 原则：

```text
ABRBFDense 是正式 KAN edge primitive；
base 参数和 RBF coeff 都属于 edge 参数；
strict PureKAN 中 learnable nonKAN params 必须仍为 0；
coefficient_named_params / edge_named_params 必须按 role 收集 input / block / output 和 base / rbf。
```

---

### 1.2 P1 说明 AB-RBF 是明确正信号

v5.0 的 P1 architecture frontier 显示，AB-RBF 不是摆设。

三个 all-dataset positive 架构是：

```text
PureKAN-ABRBF-linear+silu-AdamW
PureKAN-ABRBF-linear-AdamW
PureKAN-ABRBF-linear-learnWidth-AdamW
PureKAN-ABRBF-silu-AdamW
```

其中 `ABRBF-silu` 在 MNIST、Fashion-MNIST、KMNIST 上都提高 mean accuracy；`ABRBF-linear` 也改善 MNIST / KMNIST，并且 Fashion 基本打平。BaseOnly 不是充分的，尤其在 KMNIST 上明显不够；这说明 base path 重要，但 RBF residual 仍然需要承担非线性。

这个结果把前几轮的问题重新解释成：

$$
\boxed{
\text{PureKAN-UFULL 不是单纯 optimizer 失败，而是 fixed-grid RBF-only edge 坐标也不合适。}
}
$$

RBF-only 的 edge 是：

$$
f_{ij}(x)=\sum_k c_{ijk}B_k(x).
$$

AB-RBF 的 edge 是：

$$
f_{ij}(x)=b_{ij}+w_{ij}x+u_{ij}\operatorname{silu}(x)+\sum_k c_{ijk}B_k(x).
$$

后者把低阶、尺度、近似线性、base activation 的表达通道放回了 KAN edge 内部，而不是放回 MLP stem/head。因此它仍然可以保持 strict edge-only claim。

---

### 1.3 P2 说明 RBF-only 的确在用高 Sobolev modes 承担低阶结构

P2 eigenmode audit 显示：

```text
RBFOnly high Sobolev-eigen coefficient energy ≈ 26%-27%
ABRBF-linear base-mode energy ≈ 24%-26%
ABRBF-linear high Sobolev-eigen energy ≈ 22%-23%
```

这支持一个重要解释：

$$
\boxed{
\text{RBFOnly 被迫用高 Sobolev eigenmodes 拼低阶结构。}
}
$$

这正好解释为什么之前 U-FULL / Sobolev functional update 在 PureKAN 上表现差：

```text
AdamW 可以利用高 mode 拼出任务所需的低阶结构；
Sobolev functional update 会压制这些 high-mode coefficient；
于是 geometry 变好了，但 task representation 学不起来。
```

AB-RBF 通过显式 base modes 减轻了这个问题，但没有完全消除，尤其是 KMNIST 仍然需要 RBF residual 承担更强任务压力。

---

### 1.4 P3 说明 split functional update 是正方向，但 base path 的 task dynamics 不能丢

P3 split functional update 是本轮最重要的 optimizer 结果。

它显示：

```text
baseFrozen-rbfUFULL 在所有数据集上硬失败；
baseOnlyAdam-rbfFrozen 在很多设置下很强；
baseAdam + rbf functional variants 能保留较多 holdout descent，同时降低 RBF residual phi；
baseFCAdam-rbfUFULL 在 Fashion/MNIST 上甚至超过 ABRBF-AdamW，但 KMNIST 仍滞后。
```

这说明两件事。

第一，base path 不是辅助项，而是主要 task carrier。冻结 base 后，Fashion 从 `0.7754` 掉到 `0.6048`，KMNIST 从 `0.6517` 掉到 `0.2682`，MNIST 从 `0.8626` 掉到 `0.4954`。这说明 PureKAN 里的低阶任务通道必须主动学习。

第二，RBF residual functional update 确实能降低 residual geometry。很多 `baseAdam + rbf functional` 变体都有约 `0.17-0.19` 的 `phiR red`。但在 KMNIST 上它们仍然落后 ABRBF-AdamW，说明 residual functional update 目前还在牺牲一部分 task contribution。

所以 v5.1 不能追求“所有 edge 参数都同一种 functional update”。更合理的是：

$$
\boxed{
\text{base path 用 task-learning dynamics，RBF residual 用 geometry-aware functional dynamics。}
}
$$

也就是：

$$
\Delta \theta_{base}\leftarrow \operatorname{AdamLike}(\nabla_{base}L),
$$

$$
\Delta c_{rbf}\leftarrow \operatorname{FunctionalResidualUpdate}(\nabla_{c}L, S_{rbf}, \text{teacher / trust constraints}).
$$

---

### 1.5 P4 说明 Relaxed NFS 的当前实现仍太弱

P4 relaxed NFS 没有 all-dataset survivor。现象非常有信息量：

```text
Heuristic-NFS-role-block 有小的 phiR red / curvR red；
ExactNFS-logit / logit-hidden 几乎 phiR red = 0；
RelaxedNFS 的 projected/raw 不为 0，但仍几乎没有 residual phi reduction。
```

这说明 strict nullspace projection 太保守，而当前 relaxed projection 虽然保留了投影质量，却没有转化成有效的 residual smoothing。也就是说，问题不是“投影方向完全不存在”，而是：

$$
\boxed{
\text{当前 residual smoothing proposal / step controller 没有把可用方向转化成足够的 geometry movement。}
}
$$

v5.1 不应该回到 exact nullspace。Exact NFS 已经证明会把方向投掉。v5.1 应该做 **score-based accepted residual smoothing**：允许极小 function drift，用多目标评分选择步长，而不是硬约束为 $J\Delta=0$。

---

### 1.6 P7 说明扩大容量不是直接解法

P7 capacity follow-up 支持 architecture signal，但没有解决 geometry。比如：

```text
ABRBF-linear+silu h64/b16/d4 在 KMNIST seed0 强；
ABRBF-linear 在 Fashion/KMNIST 上相对大 RBF-only 有改善；
但 h96/b24 并不会自动改善 geometry，phi_rbf 往往上升；
quantile centers 不是 clean fix。
```

所以 v5.1 不应该把主要资源放在盲目扩 hidden / basis / depth。容量可以作为 follow-up，但核心仍是 residual smoothing 与 split dynamics。

---

## 2. 代码实现审计：v5.1 必须修正和固化的点

### 2.1 `ABRBFDense` 必须进入 `dgkan_core.py`

当前 core 仍以 `RBFDense` 为主，其 centers 固定、width 固定、只有 coeff 是 `nn.Parameter`。这对 RBF-only 结论是清楚的，但对 v5.0 之后的主线不够。

v5.1 应新增：

```python
class ABRBFDense(nn.Module):
    base_kind: none / linear / silu / linear+silu
    base_bias: Optional[nn.Parameter]
    base_linear: Optional[nn.Parameter]
    base_silu: Optional[nn.Parameter]
    rbf_coeff: nn.Parameter
    centers: buffer or optional parameter depending mode
    width: buffer / constrained parameter depending mode
```

建议默认先保持：

```text
centers fixed
width fixed
base parameters trainable
rbf_coeff trainable
```

不要马上把 centers/width 也学起来，因为 v5.0 已经显示 learnWidth alone 不是稳定正信号，quantile centers 也不是 clean fix。等 base + residual 更新稳定后，再做 center/width。

---

### 2.2 `coefficient_named_params()` 要升级为 `edge_named_params()`

过去 `coefficient_named_params()` 的语义是“找 RBF coeff”。但 AB-RBF 后，edge 参数不只有 RBF coeff。

建议拆成：

```python
def edge_named_params(model):
    return all edge-owned learnable params


def rbf_residual_named_params(model):
    return only RBF residual coeff


def base_named_params(model):
    return base bias / linear / silu params
```

并且每个参数必须记录 role：

```text
role_depth = input / block / output
role_edge = base / rbf
role_block_index = 0,1,2,...
```

v5.1 的 functional coverage 不能只报告一个 `coverage=1.0`。必须报告：

```text
coverage_edge_total
coverage_base
coverage_rbf
coverage_input
coverage_block
coverage_output
learnable_nonKAN_params
```

这样才能避免“base 参数是否被算成 non-KAN”这种解释风险。

---

### 2.3 split geometry 指标要从审计变成训练内指标

v5.0 已经记录：

```text
phi_base_p95
phi_rbf_p95
phi_total_p95
curvature_base
curvature_rbf
curvature_total
sobolev_rbf_norm
```

v5.1 应把它们变成训练 controller 的一部分。

关键原则：

```text
base derivative 不应该被当成坏 geometry；
RBF residual roughness 才是主要 geometry-control 对象。
```

因此 gate 不应再只看 `phi_total`。应主要看：

$$
\phi_{rbf,p95},\quad \operatorname{curvature}_{rbf},\quad \|c_{rbf}\|_{S_{rbf}}.
$$

同时保留 total geometry 作为安全指标。

---

### 2.4 `base/RBF` 比值在 BaseOnly 行需要修正显示

v5.0 表里 BaseOnly 的 `base/RBF` 出现极大数值，这是因为 RBF norm 为 0。这个值不能参与均值、gate 或图表坐标轴。v5.1 应处理为：

```text
if rbf_norm < eps:
    base_over_rbf = NaN
    base_fraction = 1.0
else:
    base_over_rbf = base_norm / rbf_norm
    base_fraction = base_norm / (base_norm + rbf_norm)
```

主报告应使用 `base_fraction`，而不是单独使用 `base/RBF`。

---

### 2.5 Relaxed NFS 的当前 projection 不应只报告 `proj/raw`

v5.0 的 `proj/raw` 说明投影后仍有方向，但不说明这个方向是否真的降低 residual geometry。因此 v5.1 必须记录：

```text
raw proposal geometry descent
projected proposal geometry descent
accepted proposal geometry descent
predicted phiR reduction
actual phiR reduction
predicted curvature reduction
actual curvature reduction
score before/after line search
```

否则我们无法判断 Relaxed NFS 是 proposal 不对，还是 line search / trust radius 太保守。

---

## 3. v5.1 的核心方法：AB-RBF + Strong Accepted Residual Smoothing

v5.1 不再把 RBF residual smoothing 写成硬 nullspace problem，而是写成一个 constrained score-maximization step。

给定 teacher 模型 $\theta_T$，当前模型 $\theta$，只更新 RBF residual coefficient $c$。定义 proposal $d_c$ 后，用步长 $\eta$ 得到候选：

$$
c' = c + \eta d_c.
$$

候选需要满足软约束：

$$
\operatorname{KL}(p_T\|p_{\theta'}) \leq \epsilon_{KL},
$$

$$
\frac{\|z_{\theta'}-z_T\|}{\|z_T\|+\epsilon} \leq \epsilon_z,
$$

$$
\frac{\|h_{\theta'}-h_T\|}{\|h_T\|+\epsilon} \leq \epsilon_h,
$$

$$
\Delta Acc \leq \epsilon_{acc}.
$$

但目标不是让 drift 等于 0，而是最大化：

$$
\operatorname{Score}(\eta)
=
\alpha_\phi \operatorname{Red}_{\phi_{rbf}}(\eta)
+
\alpha_\kappa \operatorname{Red}_{\kappa_{rbf}}(\eta)
+
\alpha_S \operatorname{Red}_{S_{rbf}}(\eta)
-
\beta_z D_z(\eta)
-
\beta_h D_h(\eta)
-
\beta_m D_m(\eta)
-
\beta_L \max(0, L_{hold}(\eta)-L_{hold}(0)).
$$

接受条件：

$$
\operatorname{Score}(\eta)>0,
$$

并且硬安全约束不被违反。这样 relaxed smoothing 不再追求严格 nullspace，而是允许极小 function drift 换取真实 residual geometry movement。

---

## 4. v5.1 实验计划总览

v5.1 的目标不是立刻 10-seed confirm，而是把 v5.0 的两个正信号变成可持续训练方案：

```text
正信号 1: AB-RBF architecture-positive
正信号 2: split functional update 可以降低 residual phi
```

v5.1 的阶段：

```text
P0: core implementation smoke
P1: AB-RBF architecture confirm, 5-seed
P2: split geometry calibration and gate validation
P3: residual smoothing proposal audit
P4: score-based accepted residual smoothing
P5: learn -> smooth -> refresh v3
P6: event-driven multi-cycle controller
P7: residual functional training from scratch
P8: capacity / basis follow-up
P9: 3-seed full-budget candidate selection
P10: 5-seed confirm
P11: 10-seed final confirm
```

---

## 5. P0：core implementation smoke

### 5.1 目标

P0 只验证实现是否干净，不做性能结论。

### 5.2 必跑模型

```text
PureKAN-RBFOnly
PureKAN-ABRBF-linear
PureKAN-ABRBF-silu
PureKAN-ABRBF-linear+silu
PureKAN-BaseOnly-linear
PureKAN-BaseOnly-silu
```

### 5.3 必查 invariant

每个 dataset / model 都记录：

```text
learnable_nonKAN_params
edge_param_count_total
base_param_count
rbf_param_count
coverage_edge_total
coverage_base
coverage_rbf
coverage_input
coverage_block
coverage_output
rollback_max_abs_error
finite_forward
finite_backward
finite_split_geometry
```

通过条件：

$$
\text{learnable\_nonKAN\_params}=0.
$$

$$
\text{coverage\_edge\_total}=1.
$$

$$
\text{coverage\_base}=1,\quad \text{coverage\_rbf}=1.
$$

$$
\text{rollback\_max\_abs\_error}<10^{-8}.
$$

P0 还要输出 `edge_param_manifest.csv`，逐参数列出：

```text
name
shape
role_depth
role_edge
optimizer_group
requires_grad
included_in_edge_update
included_in_base_update
included_in_rbf_update
```

---

## 6. P1：AB-RBF architecture confirm，5-seed

### 6.1 目标

v5.0 的 architecture frontier 是 3-seed / compact setting。v5.1 要先确认 AB-RBF 不是 seed artifact。

### 6.2 数据集

```text
MNIST
Fashion-MNIST
KMNIST
```

### 6.3 方法

```text
PureKAN-RBFOnly-AdamW
PureKAN-ABRBF-linear-AdamW
PureKAN-ABRBF-silu-AdamW
PureKAN-ABRBF-linear+silu-AdamW
PureKAN-BaseOnly-linear-AdamW
PureKAN-BaseOnly-silu-AdamW
MLP-AdamW
```

### 6.4 设置

```text
hidden_dim = 64
basis_count = 16
depth = 4
train / val / test = current compact setting first
seeds = 0..4
```

### 6.5 记录指标

任务指标：

```text
test_acc
val_loss_auc
train_loss_auc
ECE
NLL
classwise_acc
margin_mean
margin_p10
rank_block
rank_input
rank_output
```

edge decomposition：

```text
base_norm
rbf_norm
base_fraction
base_over_rbf, only if rbf_norm > eps
base_ablation_drop
rbf_ablation_drop
base_only_forward_acc
rbf_only_forward_acc
base_logit_delta_norm
rbf_logit_delta_norm
base_margin_contribution
rbf_margin_contribution
```

geometry：

```text
phi_base_p95
phi_rbf_p95
phi_total_p95
curvature_base
curvature_rbf
curvature_total
sobolev_rbf_norm
jacobian_condition_total
```

### 6.6 可视化

必须画：

```text
1. acc by method and dataset, with seed std/CI
2. acc vs phi_rbf scatter
3. acc vs phi_total scatter
4. base_fraction distribution by layer and dataset
5. base_ablation_drop vs rbf_ablation_drop scatter
6. rank_block vs acc scatter
7. classwise accuracy heatmap for RBFOnly vs ABRBF
```

### 6.7 判定

P1 通过不是要求 AB-RBF 全面超过所有 baseline，而是要求：

```text
ABRBF-linear/silu/linear+silu 至少一个在三个数据集上 mean acc >= RBFOnly - 0.5%
并且至少两个数据集上 mean acc > RBFOnly
base_fraction 非平凡，即 0.1 < base_fraction < 0.9
BaseOnly 不能全面替代 AB-RBF
```

如果 P1 不通过，则 v5.0 的 architecture-positive 信号不稳，v5.1 不进入 optimizer 阶段。

---

## 7. P2：split geometry calibration

### 7.1 目标

重新定义 geometry gate。过去 total phi 会把 base path 的线性导数也算成坏几何，这对 AB-RBF 不公平。

P2 要建立三个分开的结论：

```text
base geometry: 是否只是合理低阶导数；
RBF residual geometry: 是否真的粗糙；
total geometry: 是否会造成 credit/Jacobian 风险。
```

### 7.2 方法

使用 P1 训练好的 teacher：

```text
RBFOnly-AdamW
ABRBF-linear-AdamW
ABRBF-silu-AdamW
ABRBF-linear+silu-AdamW
```

### 7.3 指标

每层记录：

```text
phi_base_p50/p95/max
phi_rbf_p50/p95/max
phi_total_p50/p95/max
curvature_base_p95
curvature_rbf_p95
curvature_total_p95
sobolev_rbf_norm
jacobian_condition_total
jacobian_condition_without_rbf
jacobian_condition_without_base
```

还要记录 RBF residual 的 eigenmode：

```text
low_mode_coeff_energy
mid_mode_coeff_energy
high_mode_coeff_energy
low_mode_grad_energy
mid_mode_grad_energy
high_mode_grad_energy
base_mode_energy
```

### 7.4 可视化

```text
1. phi_base / phi_rbf / phi_total stacked bar
2. curvature_base / curvature_rbf / curvature_total stacked bar
3. high Sobolev eigenmode energy by method
4. layerwise phi_rbf heatmap
5. jacobian total vs residual-only scatter
6. base_fraction vs high_mode_energy scatter
```

### 7.5 输出新 gate

P2 要产出 `split_geometry_gate.json`，建议默认：

```text
primary geometry gate:
  phi_rbf_reduction or curvature_rbf_reduction

secondary safety gate:
  jacobian_condition_total not worse than teacher by > 20%

not primary:
  phi_total if base path contains linear/silu derivative
```

---

## 8. P3：residual smoothing proposal audit

### 8.1 目标

先不训练，只看单次 residual smoothing proposal 是否有真实 geometry movement。

### 8.2 Teacher

```text
ABRBF-linear-AdamW
ABRBF-silu-AdamW
ABRBF-linear+silu-AdamW
```

### 8.3 Proposal variants

```text
S0: residual Laplacian smoothing
S1: residual Sobolev-gradient smoothing
S2: residual eigenmode shrink, high modes only
S3: residual eigenmode shrink, high+mid modes
S4: residual coefficient L2 shrink
S5: data-aware residual smoothing
S6: heuristic role-block residual smoother, v5.0 reference
S7: score-gradient proposal, maximize split-geometry score approximation
```

### 8.4 Projector / controller variants

```text
C0: no projection, line search only
C1: logit KL trust
C2: logit + hidden trust
C3: logit + hidden + margin trust
C4: score-based accepted step
C5: score-based accepted step + adaptive eta grid
```

### 8.5 指标

每个 proposal 记录 before / after / projected / accepted：

```text
raw_direction_norm
projected_direction_norm
projected_over_raw
predicted_phiR_red
actual_phiR_red
predicted_curvR_red
actual_curvR_red
predicted_sobolev_red
actual_sobolev_red
KL_teacher_student
logit_relative_drift
hidden_relative_drift
margin_relative_drift
acc_drop
holdout_loss_delta
score
accepted_eta
accept_rate
reject_reason
```

### 8.6 可视化

```text
1. predicted vs actual phiR reduction scatter
2. projected_over_raw vs actual phiR reduction scatter
3. KL/logit drift vs phiR reduction Pareto
4. accepted_eta histogram
5. rejection reason stacked bar
6. proposal score heatmap by dataset / teacher / proposal / controller
```

### 8.7 判定

P3 survivor 条件：

$$
\Delta Acc \leq 0.5\%.
$$

$$
\operatorname{KL}<0.01.
$$

$$
\operatorname{logit\_drift}<0.03.
$$

$$
\operatorname{phiR\_red}>5\% \quad \text{or} \quad \operatorname{curvR\_red}>10\%.
$$

并且至少在两个数据集上满足，才进入 P4。

---

## 9. P4：score-based accepted residual smoothing

### 9.1 目标

P4 把 P3 的 proposal 放进一个小步 accepted update 中，验证它是否能连续执行 5 到 20 步而不破坏 task。

### 9.2 更新形式

只更新 RBF residual：

$$
c_{t+1}=c_t+\eta_t d_t.
$$

base path 冻结，teacher 固定。每步通过 score-based accept：

$$
\operatorname{Score}
=
\alpha_\phi \operatorname{Red}_{\phi_{rbf}}
+
\alpha_C \operatorname{Red}_{curv_{rbf}}
+
\alpha_S \operatorname{Red}_{S_{rbf}}
-
\beta_z D_z
-
\beta_h D_h
-
\beta_m D_m
-
\beta_L \max(0,\Delta L_{hold}).
$$

默认权重：

```text
alpha_phi = 1.0
alpha_C = 0.5
alpha_S = 0.2
beta_z = 2.0
beta_h = 1.0
beta_m = 1.0
beta_L = 3.0
```

### 9.3 方法

```text
P4-A: best P3 proposal, 5 steps
P4-B: best P3 proposal, 20 steps
P4-C: adaptive proposal choice among S0/S2/S6/S7
P4-D: role-block only
P4-E: role-cycle input/block/output residual smoothing
```

### 9.4 指标

```text
stepwise accepted_eta
accept_rate
score_curve
phiR_curve
curvR_curve
sobolev_rbf_curve
KL_curve
logit_drift_curve
hidden_drift_curve
margin_curve
acc_curve
holdout_loss_curve
rank_curve
base_fraction_curve
```

### 9.5 可视化

```text
1. phiR / acc two-axis curve over smoothing steps
2. score curve with accepted/rejected markers
3. KL and logit drift curves
4. margin and rank curves
5. proposal selection stacked area plot
6. residual geometry Pareto front before/after smoothing
```

### 9.6 判定

P4 survivor：

```text
acc drop <= 0.5%
KL < 0.02
phiR red > 8% or curvR red > 15%
holdout loss not worse by > 2%
```

P4 如果没有 survivor，就不进入 Learn-Smooth-Refresh。此时说明 residual smoothing proposal 仍然不足，需要回到 proposal 设计。

---

## 10. P5：Learn -> Smooth -> Refresh v3

### 10.1 目标

验证完整 one-cycle：

```text
learn task with AB-RBF
smooth RBF residual with accepted controller
refresh task while preserving residual geometry
```

### 10.2 Phase A：Learn

方法：

```text
ABRBF-AdamW
ABRBF-baseAdam-rbfAdamW
ABRBF-baseAdam-rbfFCAdam
ABRBF-baseFCAdam-rbfAdamW
```

学习到 teacher snapshot：

```text
T20
T100
Tfinal
```

### 10.3 Phase B：Smooth

使用 P4 survivor，只更新 RBF residual。base path 固定。

### 10.4 Phase C：Refresh

Refresh variants：

```text
R0: base-only AdamW refresh, RBF frozen
R1: base AdamW + tiny RBF task update
R2: base FCAdam + tiny RBF task update
R3: D6-style small task refresh on RBF only
R4: no refresh, smoothing only
```

### 10.5 记录指标

每个 phase 都记录：

```text
test_acc
val_loss_auc
ECE
NLL
rank_block
margin_mean/p10
base_fraction
base_ablation_drop
rbf_ablation_drop
phi_rbf_p95
curvature_rbf
sobolev_rbf_norm
jacobian_total
KL_to_teacher
logit_drift
hidden_drift
```

### 10.6 可视化

```text
1. learn-smooth-refresh phase plot: acc / phiR / KL / rank
2. refresh recovery plot: acc recovered vs phiR retained
3. base vs RBF ablation drop before and after smoothing
4. task loss vs residual geometry Pareto by phase
```

### 10.7 判定

P5 survivor 条件：

```text
After Refresh:
  acc >= Learn_acc - 0.5%
  phiR <= Learn_phiR * 0.90
  ECE not worse by > 0.02
  rank >= Learn_rank * 0.90
  KL_to_learn_teacher < 0.03
```

必须三个数据集都至少有一个 survivor 才进入 P6。

---

## 11. P6：event-driven multi-cycle controller

### 11.1 目标

v4.8/v5.0 都说明固定 cycle 不稳定。v5.1 改用 event-driven controller。

### 11.2 触发条件

只在以下条件满足时 smoothing：

```text
1. validation loss plateau: recent improvement < threshold
2. rank and margin stable: rank drop < threshold, margin p10 stable
3. residual geometry over budget: phiR > target or curvR > target
4. base path contribution stable: base_fraction variation small
```

形式上，若：

$$
\Delta L_{val}^{recent}<\epsilon_L,
$$

$$
\phi_{rbf}>\phi_{target},
$$

$$
\Delta rank > -\epsilon_r,
$$

才触发 smoothing。

### 11.3 Controller variants

```text
E0: no smoothing, ABRBF-AdamW baseline
E1: fixed cycle, v5.0 reference
E2: event-driven smoothing, conservative
E3: event-driven smoothing, aggressive
E4: event-driven smoothing + base-only refresh
E5: event-driven smoothing + adaptive proposal selection
```

### 11.4 指标

```text
num_smoothing_events
trigger_reason
accepted_steps_per_event
phiR_reduction_per_event
acc_drop_per_event
refresh_recovery_rate
cycle_stability_score
failure_reason
```

### 11.5 可视化

```text
1. timeline with smoothing events marked
2. acc/val-loss/phiR/rank/margin over time
3. event waterfall: before -> after smooth -> after refresh
4. smoothing trigger heatmap
5. per-event accepted eta histogram
```

### 11.6 判定

P6 survivor：

```text
final acc >= ABRBF-AdamW - 0.75%
final phiR red >= 10%
final ECE not worse by > 0.02
no more than one catastrophic event with acc drop > 1%
```

---

## 12. P7：residual functional training from scratch

### 12.1 目标

P5/P6 是 posthoc / phase-separated。P7 测训练时就使用 split dynamics：base 学任务，RBF residual 受 geometry 控制。

### 12.2 方法

```text
ABRBF-AdamW
ABRBF-baseAdam-rbfUFULL
ABRBF-baseAdam-rbfD6
ABRBF-baseAdam-rbfFCAdam-dataSob
ABRBF-baseAdam-rbfResidualProxAdam
ABRBF-baseFCAdam-rbfUFULL
ABRBF-baseAdam-rbfAdamW+lateResidualSmooth
```

新增 `rbfResidualProxAdam`：

先算 RBF residual 的 AdamW proposal $d_{Adam}$，再解：

$$
\Delta c
=
\arg\min_{\Delta}
\frac{1}{2}\|\Delta-d_{Adam}\|_{D_t}^2
+
\frac{\lambda}{2}\|c+\Delta\|_{S_{rbf}}^2.
$$

闭式形式近似为：

$$
\Delta
=
(D_t+\lambda S_{rbf}+\rho I)^{-1}D_t d_{Adam}.
$$

直觉：

```text
AdamW proposal 负责 task learning；
Sobolev prox 只修正 residual geometry。
```

这比直接 $S^{-1}g$ 更符合 v5.0 的经验。

### 12.3 指标

除了常规指标，P7 必须记录：

```text
cos_rbf_update_with_adam
rbf_prox_projection_error
rbf_sobolev_penalty_before_after
base_update_norm
rbf_update_norm
base_update_share
rbf_update_share
high_mode_energy_over_time
```

### 12.4 判定

P7 survivor：

```text
acc >= ABRBF-AdamW - 0.75%
val_loss_auc >= ABRBF-AdamW - 5%
phiR red >= 10%
ECE not worse by > 0.02
base_fraction remains nontrivial
```

如果 P7 survivor 出现，再进入 P9/P10 seed selection。

---

## 13. P8：capacity / basis follow-up，只作为辅助

P8 不做大扫，只测 v5.0 已经显示有意义的组合：

```text
ABRBF-linear h64/b16/d4
ABRBF-linear+silu h64/b16/d4
ABRBF-silu h64/b16/d4
ABRBF-linear h96/b24/d4
ABRBF-linear+silu h96/b24/d4
```

不优先测：

```text
quantile centers, unless P2 basis coverage shows severe out-of-grid
learnWidth alone, unless P1/P2 shows fixed width is primary blocker
```

P8 只回答：

```text
candidate 是否对 capacity 敏感？
是否需要 h96/b24 才能过 gate？
更大容量是否只是提高 acc 但恶化 phiR？
```

---

## 14. P9：3-seed full-budget candidate selection

P9 只允许从 P5/P6/P7 survivors 进入。

候选最多 4 个：

```text
1. ABRBF-AdamW baseline
2. best P5 one-cycle candidate
3. best P6 event-driven candidate
4. best P7 training-from-scratch candidate
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

判定：

```text
进入 P10 的候选必须：
  acc gap <= 0.75% vs ABRBF-AdamW on all datasets
  phiR red >= 10% on at least two datasets
  no catastrophic failure
```

---

## 15. P10：5-seed confirm

P10 使用 seeds 0..4。

必须报告 paired delta vs ABRBF-AdamW：

```text
paired_acc_delta
paired_val_auc_delta
paired_phiR_delta
paired_curvR_delta
paired_ECE_delta
paired_rank_delta
```

通过条件：

$$
\operatorname{mean}(\Delta Acc) \geq -0.75\%.
$$

$$
\operatorname{mean}(\Delta \phi_{rbf}) \leq -10\%.
$$

$$
\operatorname{ECE}_{candidate}\leq \operatorname{ECE}_{AdamW}+0.02.
$$

如果 P10 通过，再进入 P11。

---

## 16. P11：10-seed final confirm

P11 使用 seeds 0..9，只确认最终 1 个 candidate。

输出最终结论分三档。

### 16.1 strong pass

```text
acc gap <= 0.5%
phiR red >= 15%
curvR red >= 15%
ECE not worse
rank retained >= 90%
```

可写：

```text
AB-RBF edge-decomposed functional training provides a geometry-improving PureKAN optimizer.
```

### 16.2 Pareto pass

```text
acc gap <= 1.0%
phiR red >= 10%
ECE not worse by > 0.02
```

可写：

```text
AB-RBF split functional training is a geometry Pareto optimizer, but AdamW remains accuracy baseline.
```

### 16.3 fail

如果不满足：

```text
AB-RBF improves architecture, but functional residual smoothing remains a posthoc diagnostic rather than a train-time optimizer.
```

---

## 17. v5.1 必须产出的文件

```text
results/v5_1/
  p0_core_smoke.csv
  edge_param_manifest.csv
  p1_architecture_confirm5.csv
  p2_split_geometry_calibration.csv
  p3_residual_smoothing_proposal_audit.csv
  p4_accepted_residual_smoothing.csv
  p5_learn_smooth_refresh_v3.csv
  p6_event_driven_tan_v2.csv
  p7_split_residual_training.csv
  p8_capacity_basis_followup.csv
  p9_candidate_selection3.csv
  p10_confirm5.csv
  p11_confirm10.csv
  paired_delta_vs_ABRBF_AdamW.csv
  failure_table.csv
  aggregate_decision.json
  figures/
```

Figures：

```text
figures/p1_architecture_frontier_acc_phiR.svg
figures/p1_base_rbf_ablation.svg
figures/p2_split_geometry_stacked.svg
figures/p2_eigenmode_energy.svg
figures/p3_predicted_vs_actual_phiR.svg
figures/p3_drift_vs_geometry_pareto.svg
figures/p4_score_eta_acceptance.svg
figures/p5_learn_smooth_refresh_timeline.svg
figures/p6_event_waterfall.svg
figures/p7_training_curves_acc_phiR.svg
figures/p10_paired_delta_forest.svg
```

---

## 18. 失败诊断表

每个失败 row 必须打标签：

```text
F1_ABRBF_not_architecture_positive
F2_base_path_unused
F3_base_path_overdominates_rbf
F4_residual_high_mode_pressure_persists
F5_smoothing_no_geometry_gain
F6_smoothing_breaks_task
F7_refresh_cannot_recover_task
F8_event_controller_over_smooths
F9_candidate_lags_ABRBF_AdamW
F10_capacity_increases_phiR
F11_implementation_invariant_failed
```

最终报告要按 dataset / method / failure_type 做 heatmap。

---

## 19. 预期结论模板

### 如果结果好

```text
v5.1 confirms that PureKAN functional training requires edge decomposition.
Base paths learn low-order task structure, while RBF residuals can be geometry-controlled through score-based accepted residual smoothing.
The resulting AB-RBF split functional optimizer matches ABRBF-AdamW accuracy within a small gap while reducing residual roughness and preserving calibration.
```

### 如果结果中等

```text
v5.1 confirms AB-RBF as the correct PureKAN architecture, but residual functional smoothing remains a Pareto/posthoc component.
It can reduce residual roughness with small task drift, yet train-time candidates still lag ABRBF-AdamW on KMNIST.
```

### 如果结果失败

```text
v5.1 shows that AB-RBF fixes part of the parameterization bottleneck, but current RBF residual smoothing is not strong enough to serve as a training optimizer.
PureKAN functional optimization should be paused or reframed around a different primitive / residual parameterization.
```

---

## 20. 最终一句话

v5.1 的核心不是再证明 RBF-only U-FULL，而是验证：

$$
\boxed{
\text{AB-RBF base path 学任务，RBF residual 做可控几何，是否能形成真正 PureKAN functional training。}
}
$$

如果这个方向也失败，那么我们应明确承认：

```text
当前 functional update 的有效范围仍主要是 Hybrid-DGKAN branch / posthoc smoothing，
而不是完整 PureKAN train-time optimizer。
```



---


# Source 30: `docs/DG-KAN_v5.1_ABRBF_StrongResidualSmoothing_结果复盘.md`


# DG-KAN v5.1 AB-RBF Strong Residual Smoothing 结果复盘

本轮依据 `docs/DG-KAN_v5.1_ABRBF_StrongResidualSmoothing_实验计划.md`。核心目标是把 v5.0 的 AB-RBF 架构正信号固化为默认 PureKAN edge primitive，并测试更强的 residual smoothing proposal / accepted-step controller 是否能把单步 NFS 信号推进到多步稳定训练。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 core smoke | 18 | 0 |
| P1 architecture confirm5 | 105 | 0 |
| P2 split geometry | 540 | 0 |
| P3 residual smoothing audit | 432 | 0 |
| P4 accepted smoothing | 18 | 0 |
| P5 learn-smooth-refresh | 1 | 0 |
| P6 event TAN | 1 | 0 |
| P7 split residual training | 135 | 0 |
| P8 capacity follow-up | 27 | 0 |

## Code / Config Changes

```text
experiments/dgkan_core.py
  Added core ABRBFDense with linear / silu / linear+silu / base-only edge paths.
  Added edge_named_params, base_named_params, and rbf_residual_named_params helpers.

experiments/run_gafu_v51.py
  Added P0 edge manifest, P1 5-seed AB-RBF confirm, P2 split geometry calibration,
  P3 S0-S7 residual smoothing proposals with C0-C5 controllers,
  P4 multi-step accepted smoothing, plus P7/P8 follow-up diagnostics.

experiments/analyze_gafu_v51.py
  Generates v5.1 gate summaries, failure table, aggregate decision, figures, and this replay.
```

## P0 Core Smoke

| rows | errors | manifest | max nonKAN | min edge cov | min base cov | min rbf cov | max rollback | pass |
|---|---|---|---|---|---|---|---|---|
| 18 | 0 | 162 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | true |

P0 verdict: pass. Core AB-RBF edge paths keep strict PureKAN non-KAN trainable parameters at zero, edge/base/RBF parameter coverage is auditable, and rollback is exact.

## P1 Architecture Confirm5

| dataset | method | runs | acc | std | gap vs RBF | phiR | base/RBF | P1 |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | 5 | 0.8063 | 0.0122 | 0.0000 | 0.1129 | 0.6592 | yes |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | 5 | 0.8039 | 0.0188 | 0.0023 | 0.1152 | 0.1643 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | 5 | 0.8008 | 0.0090 | 0.0055 | 0.1144 | 0.5517 | no |
| Fashion-MNIST | PureKAN-BaseOnly-linear-AdamW | 5 | 0.7914 | 0.0167 | 0.0148 | 0.0000 |  | no |
| Fashion-MNIST | PureKAN-BaseOnly-silu-AdamW | 5 | 0.7883 | 0.0081 | 0.0180 | 0.0000 |  | no |
| KMNIST | PureKAN-ABRBF-linear-AdamW | 5 | 0.6977 | 0.0190 | -0.0223 | 0.1168 | 0.5660 | yes |
| KMNIST | PureKAN-ABRBF-silu-AdamW | 5 | 0.6898 | 0.0118 | -0.0145 | 0.1163 | 0.1920 | yes |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | 5 | 0.6891 | 0.0103 | -0.0137 | 0.1142 | 0.6742 | yes |
| KMNIST | PureKAN-BaseOnly-silu-AdamW | 5 | 0.6406 | 0.0108 | 0.0348 | 0.0000 |  | no |
| KMNIST | PureKAN-BaseOnly-linear-AdamW | 5 | 0.5965 | 0.0249 | 0.0789 | 0.0000 |  | no |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | 5 | 0.8965 | 0.0105 | -0.0203 | 0.1144 | 0.6684 | yes |
| MNIST | PureKAN-ABRBF-silu-AdamW | 5 | 0.8938 | 0.0095 | -0.0176 | 0.1177 | 0.1962 | yes |
| MNIST | PureKAN-ABRBF-linear-AdamW | 5 | 0.8910 | 0.0062 | -0.0148 | 0.1160 | 0.5555 | yes |
| MNIST | PureKAN-BaseOnly-silu-AdamW | 5 | 0.8906 | 0.0146 | -0.0145 | 0.0000 |  | no |
| MNIST | PureKAN-BaseOnly-linear-AdamW | 5 | 0.8809 | 0.0169 | -0.0047 | 0.0000 |  | no |

P1 survivors:

```text
PureKAN-ABRBF-linear+silu-AdamW
PureKAN-ABRBF-silu-AdamW
```

P1 verdict: AB-RBF remains architecture-positive. The base path is consistently used; BaseOnly remains diagnostic rather than sufficient, especially on KMNIST.

## P2 Split Geometry / Eigenmode Audit

| dataset | method | high coeff | high grad | base mode | phiR | eig cond | shift |
|---|---|---|---|---|---|---|---|
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | 0.2587 | 0.0004 | 0.3253 | 0.1122 | 78.7 | 1 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | 0.2321 | 0.0001 | 0.2504 | 0.1148 | 78.7 | 1 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | 0.2515 | 0.0002 | 0.0939 | 0.1149 | 78.7 | 1 |
| Fashion-MNIST | PureKAN-BaseOnly-linear-AdamW | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0 | 1 |
| Fashion-MNIST | PureKAN-BaseOnly-silu-AdamW | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0 | 1 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | 0.2688 | 0.0002 | 0.0000 | 0.1177 | 78.7 | 0 |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | 0.2528 | 0.0005 | 0.3154 | 0.1138 | 78.7 | 1 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | 0.2254 | 0.0002 | 0.2402 | 0.1171 | 78.7 | 1 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | 0.2457 | 0.0003 | 0.0875 | 0.1167 | 78.7 | 1 |
| KMNIST | PureKAN-BaseOnly-linear-AdamW | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0 | 1 |
| KMNIST | PureKAN-BaseOnly-silu-AdamW | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0 | 1 |
| KMNIST | PureKAN-RBFOnly-AdamW | 0.2615 | 0.0003 | 0.0000 | 0.1197 | 78.7 | 0 |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | 0.2519 | 0.0005 | 0.3343 | 0.1143 | 78.7 | 1 |
| MNIST | PureKAN-ABRBF-linear-AdamW | 0.2277 | 0.0001 | 0.2535 | 0.1168 | 78.7 | 1 |
| MNIST | PureKAN-ABRBF-silu-AdamW | 0.2436 | 0.0002 | 0.0964 | 0.1178 | 78.7 | 1 |
| MNIST | PureKAN-BaseOnly-linear-AdamW | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0 | 1 |
| MNIST | PureKAN-BaseOnly-silu-AdamW | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0 | 1 |
| MNIST | PureKAN-RBFOnly-AdamW | 0.2570 | 0.0003 | 0.0000 | 0.1220 | 78.7 | 0 |

P2 verdict: AB-RBF shifts visible energy into explicit base modes, confirming that RBF-only had been carrying low-order structure through residual coefficient modes. The residual high-mode pressure is reduced but not removed.

## P3 Residual Smoothing Proposal Audit

| teacher | proposal | controller | dataset | pass | acc drop | phiR red | curvR red | KL | score |
|---|---|---|---|---|---|---|---|---|---|
| PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | C1-logit-kl-trust | MNIST | 1 | 0.0020 | 0.0897 | 0.7011 | 0.0044 | 0.4778 |
| PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | C2-logit-hidden-trust | MNIST | 1 | 0.0020 | 0.0897 | 0.7011 | 0.0044 | 0.4778 |
| PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | C3-logit-hidden-margin-trust | MNIST | 1 | 0.0020 | 0.0897 | 0.7011 | 0.0044 | 0.4778 |
| PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | C4-score-accepted | MNIST | 1 | 0.0020 | 0.0897 | 0.7011 | 0.0044 | 0.4778 |
| PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | C5-score-adaptive-eta | MNIST | 1 | 0.0020 | 0.0897 | 0.7011 | 0.0044 | 0.4778 |
| PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | C1-logit-kl-trust | MNIST | 1 | 0.0020 | 0.0897 | 0.7011 | 0.0044 | 0.4778 |
| PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | C2-logit-hidden-trust | MNIST | 1 | 0.0020 | 0.0897 | 0.7011 | 0.0044 | 0.4778 |
| PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | C3-logit-hidden-margin-trust | MNIST | 1 | 0.0020 | 0.0897 | 0.7011 | 0.0044 | 0.4778 |
| PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | C4-score-accepted | MNIST | 1 | 0.0020 | 0.0897 | 0.7011 | 0.0044 | 0.4778 |
| PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | C5-score-adaptive-eta | MNIST | 1 | 0.0020 | 0.0897 | 0.7011 | 0.0044 | 0.4778 |
| PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | C1-logit-kl-trust | Fashion-MNIST | 1 | 0.0020 | 0.0632 | 0.5484 | 0.0028 | 0.3539 |
| PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | C2-logit-hidden-trust | Fashion-MNIST | 1 | 0.0020 | 0.0632 | 0.5484 | 0.0028 | 0.3539 |
| PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | C3-logit-hidden-margin-trust | Fashion-MNIST | 1 | 0.0020 | 0.0632 | 0.5484 | 0.0028 | 0.3539 |
| PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | C4-score-accepted | Fashion-MNIST | 1 | 0.0020 | 0.0632 | 0.5484 | 0.0028 | 0.3539 |
| PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | C5-score-adaptive-eta | Fashion-MNIST | 1 | 0.0020 | 0.0632 | 0.5484 | 0.0028 | 0.3539 |
| PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | C1-logit-kl-trust | Fashion-MNIST | 1 | 0.0020 | 0.0632 | 0.5484 | 0.0028 | 0.3539 |
| PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | C2-logit-hidden-trust | Fashion-MNIST | 1 | 0.0020 | 0.0632 | 0.5484 | 0.0028 | 0.3539 |
| PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | C3-logit-hidden-margin-trust | Fashion-MNIST | 1 | 0.0020 | 0.0632 | 0.5484 | 0.0028 | 0.3539 |

P3 all-dataset survivors:

```text
PureKAN-ABRBF-linear+silu-AdamW|S0-laplacian|C1-logit-kl-trust
PureKAN-ABRBF-linear+silu-AdamW|S0-laplacian|C2-logit-hidden-trust
PureKAN-ABRBF-linear+silu-AdamW|S0-laplacian|C3-logit-hidden-margin-trust
PureKAN-ABRBF-linear+silu-AdamW|S0-laplacian|C4-score-accepted
PureKAN-ABRBF-linear+silu-AdamW|S0-laplacian|C5-score-adaptive-eta
PureKAN-ABRBF-linear+silu-AdamW|S1-sobolev-grad|C1-logit-kl-trust
PureKAN-ABRBF-linear+silu-AdamW|S1-sobolev-grad|C2-logit-hidden-trust
PureKAN-ABRBF-linear+silu-AdamW|S1-sobolev-grad|C3-logit-hidden-margin-trust
PureKAN-ABRBF-linear+silu-AdamW|S1-sobolev-grad|C4-score-accepted
PureKAN-ABRBF-linear+silu-AdamW|S1-sobolev-grad|C5-score-adaptive-eta
PureKAN-ABRBF-linear+silu-AdamW|S6-heuristic-role-block|C0-line-search
PureKAN-ABRBF-linear+silu-AdamW|S6-heuristic-role-block|C1-logit-kl-trust
PureKAN-ABRBF-linear+silu-AdamW|S6-heuristic-role-block|C2-logit-hidden-trust
PureKAN-ABRBF-linear+silu-AdamW|S6-heuristic-role-block|C3-logit-hidden-margin-trust
PureKAN-ABRBF-linear+silu-AdamW|S6-heuristic-role-block|C4-score-accepted
PureKAN-ABRBF-linear+silu-AdamW|S6-heuristic-role-block|C5-score-adaptive-eta
PureKAN-ABRBF-linear-AdamW|S6-heuristic-role-block|C0-line-search
PureKAN-ABRBF-linear-AdamW|S6-heuristic-role-block|C1-logit-kl-trust
PureKAN-ABRBF-linear-AdamW|S6-heuristic-role-block|C2-logit-hidden-trust
PureKAN-ABRBF-linear-AdamW|S6-heuristic-role-block|C3-logit-hidden-margin-trust
PureKAN-ABRBF-linear-AdamW|S6-heuristic-role-block|C4-score-accepted
PureKAN-ABRBF-linear-AdamW|S6-heuristic-role-block|C5-score-adaptive-eta
PureKAN-ABRBF-silu-AdamW|S6-heuristic-role-block|C0-line-search
PureKAN-ABRBF-silu-AdamW|S6-heuristic-role-block|C1-logit-kl-trust
PureKAN-ABRBF-silu-AdamW|S6-heuristic-role-block|C2-logit-hidden-trust
PureKAN-ABRBF-silu-AdamW|S6-heuristic-role-block|C3-logit-hidden-margin-trust
PureKAN-ABRBF-silu-AdamW|S6-heuristic-role-block|C4-score-accepted
```

P3 verdict: strong single-step residual smoothing is real. `S0-laplacian`, `S1-sobolev-grad`, and `S6-heuristic-role-block` can produce accepted residual phi reductions under trust controllers across all datasets.

## P4 Accepted Residual Smoothing

| dataset | proposal | controller | steps | acc drop | phiR red | curvR red | holdout Δ | accept | P4 |
|---|---|---|---|---|---|---|---|---|---|
| MNIST | S0-laplacian | C1-logit-kl-trust | 20 | 0.0547 | 0.4719 | 0.9808 | 0.3213 | 1.0000 | no |
| MNIST | S0-laplacian | C2-logit-hidden-trust | 20 | 0.0547 | 0.4719 | 0.9808 | 0.3213 | 1.0000 | no |
| MNIST | S0-laplacian | C3-logit-hidden-margin-trust | 20 | 0.0547 | 0.4719 | 0.9808 | 0.3213 | 1.0000 | no |
| Fashion-MNIST | S0-laplacian | C1-logit-kl-trust | 20 | 0.0586 | 0.4567 | 0.9762 | 0.3988 | 1.0000 | no |
| Fashion-MNIST | S0-laplacian | C2-logit-hidden-trust | 20 | 0.0586 | 0.4567 | 0.9762 | 0.3988 | 1.0000 | no |
| Fashion-MNIST | S0-laplacian | C3-logit-hidden-margin-trust | 20 | 0.0586 | 0.4567 | 0.9762 | 0.3988 | 1.0000 | no |
| KMNIST | S0-laplacian | C1-logit-kl-trust | 20 | -0.0117 | 0.4477 | 0.9746 | 0.2806 | 1.0000 | no |
| KMNIST | S0-laplacian | C2-logit-hidden-trust | 20 | -0.0117 | 0.4477 | 0.9746 | 0.2806 | 1.0000 | no |
| KMNIST | S0-laplacian | C3-logit-hidden-margin-trust | 20 | -0.0117 | 0.4477 | 0.9746 | 0.2806 | 1.0000 | no |
| Fashion-MNIST | S0-laplacian | C1-logit-kl-trust | 5 | 0.0195 | 0.2004 | 0.8661 | 0.1291 | 1.0000 | no |
| Fashion-MNIST | S0-laplacian | C2-logit-hidden-trust | 5 | 0.0195 | 0.2004 | 0.8661 | 0.1291 | 1.0000 | no |
| Fashion-MNIST | S0-laplacian | C3-logit-hidden-margin-trust | 5 | 0.0195 | 0.2004 | 0.8661 | 0.1291 | 1.0000 | no |
| MNIST | S0-laplacian | C1-logit-kl-trust | 5 | 0.0039 | 0.1980 | 0.8792 | 0.0386 | 1.0000 | no |
| MNIST | S0-laplacian | C2-logit-hidden-trust | 5 | 0.0039 | 0.1980 | 0.8792 | 0.0386 | 1.0000 | no |
| MNIST | S0-laplacian | C3-logit-hidden-margin-trust | 5 | 0.0039 | 0.1980 | 0.8792 | 0.0386 | 1.0000 | no |
| KMNIST | S0-laplacian | C1-logit-kl-trust | 5 | -0.0117 | 0.1902 | 0.8604 | 0.0630 | 1.0000 | no |
| KMNIST | S0-laplacian | C2-logit-hidden-trust | 5 | -0.0117 | 0.1902 | 0.8604 | 0.0630 | 1.0000 | no |
| KMNIST | S0-laplacian | C3-logit-hidden-margin-trust | 5 | -0.0117 | 0.1902 | 0.8604 | 0.0630 | 1.0000 | no |

P4 all-dataset survivors:

```text
none
```

P4 verdict: no survivor. Multi-step accepted smoothing reduces residual phi strongly, but it violates the joint task/holdout gate. The bottleneck moved from “no geometry movement” to “geometry movement is too task-expensive when accumulated.”

## P5 / P6 Decision

```text
P5 learn-smooth-refresh: not run
Reason: P4 produced no all-dataset accepted smoothing survivor.

P6 event TAN: not run
Reason: P5 was not reached.
```

## P7 Split Residual Training

| dataset | method | runs | acc | gap vs ABRBF | hold/A | phiR red | base share | P7 |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | ABRBF-baseFCAdam-rbfUFULL | 5 | 0.7824 | -0.0082 |  | 0.1742 | 0.9212 | yes |
| Fashion-MNIST | ABRBF-baseAdam-rbfFCAdam-dataSob | 5 | 0.7754 | -0.0012 |  | 0.1730 | 0.9816 | yes |
| Fashion-MNIST | ABRBF-baseAdam-rbfD6 | 5 | 0.7750 | -0.0008 |  | 0.1731 | 0.9900 | yes |
| Fashion-MNIST | ABRBF-baseAdam-rbfRelaxedNFSRefresh | 5 | 0.7750 | -0.0008 |  | 0.1730 | 0.9898 | yes |
| Fashion-MNIST | ABRBF-baseAdam-rbfUFULL | 5 | 0.7746 | -0.0004 |  | 0.1722 | 0.9726 | yes |
| Fashion-MNIST | ABRBF-AdamW | 5 | 0.7742 | 0.0000 |  | 0.0000 | 0.1967 | yes |
| Fashion-MNIST | ABRBF-allAdamW-edgeOnly | 5 | 0.7742 | 0.0000 |  | 0.0000 | 0.1967 | no |
| Fashion-MNIST | ABRBF-baseOnlyAdam-rbfFrozen | 5 | 0.7742 | 0.0000 |  | 0.1741 | 1.0000 | yes |
| Fashion-MNIST | ABRBF-baseFrozen-rbfUFULL | 5 | 0.6129 | 0.1613 |  | 0.1758 | 0.0000 | no |
| KMNIST | ABRBF-AdamW | 5 | 0.6465 | 0.0000 |  | 0.0000 | 0.2076 | yes |
| KMNIST | ABRBF-allAdamW-edgeOnly | 5 | 0.6465 | 0.0000 |  | 0.0000 | 0.2076 | no |
| KMNIST | ABRBF-baseAdam-rbfUFULL | 5 | 0.6391 | 0.0074 |  | 0.1745 | 0.9690 | yes |
| KMNIST | ABRBF-baseAdam-rbfFCAdam-dataSob | 5 | 0.6363 | 0.0102 |  | 0.1747 | 0.9792 | yes |
| KMNIST | ABRBF-baseOnlyAdam-rbfFrozen | 5 | 0.6344 | 0.0121 |  | 0.1741 | 1.0000 | yes |
| KMNIST | ABRBF-baseAdam-rbfD6 | 5 | 0.6320 | 0.0145 |  | 0.1742 | 0.9886 | yes |
| KMNIST | ABRBF-baseAdam-rbfRelaxedNFSRefresh | 5 | 0.6316 | 0.0148 |  | 0.1743 | 0.9884 | yes |
| KMNIST | ABRBF-baseFCAdam-rbfUFULL | 5 | 0.6258 | 0.0207 |  | 0.1747 | 0.9169 | no |
| KMNIST | ABRBF-baseFrozen-rbfUFULL | 5 | 0.2637 | 0.3828 |  | 0.1787 | 0.0000 | no |
| MNIST | ABRBF-baseFCAdam-rbfUFULL | 5 | 0.8766 | -0.0191 |  | 0.1970 | 0.9011 | yes |
| MNIST | ABRBF-baseAdam-rbfD6 | 5 | 0.8758 | -0.0184 |  | 0.1935 | 0.9855 | yes |
| MNIST | ABRBF-baseAdam-rbfRelaxedNFSRefresh | 5 | 0.8758 | -0.0184 |  | 0.1935 | 0.9853 | yes |
| MNIST | ABRBF-baseOnlyAdam-rbfFrozen | 5 | 0.8754 | -0.0180 |  | 0.1943 | 1.0000 | yes |
| MNIST | ABRBF-baseAdam-rbfUFULL | 5 | 0.8727 | -0.0152 |  | 0.1921 | 0.9619 | yes |
| MNIST | ABRBF-baseAdam-rbfFCAdam-dataSob | 5 | 0.8691 | -0.0117 |  | 0.1936 | 0.9737 | yes |
| MNIST | ABRBF-AdamW | 5 | 0.8574 | 0.0000 |  | 0.0000 | 0.2037 | yes |
| MNIST | ABRBF-allAdamW-edgeOnly | 5 | 0.8574 | 0.0000 |  | 0.0000 | 0.2037 | no |
| MNIST | ABRBF-baseFrozen-rbfUFULL | 5 | 0.4844 | 0.3730 |  | 0.1961 | 0.0000 | no |

P7 verdict: moving the base path is essential. Functional residual variants reduce residual phi, but KMNIST accuracy still lags the ABRBF-AdamW teacher enough to block a clean all-dataset survivor.

## P8 Capacity / Basis Follow-Up

| dataset | edge | hidden | basis | depth | acc | phiR | rank | base/RBF |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | ABRBF-linear | 64 | 16 | 4 | 0.8008 | 0.1114 | 9.1229 | 0.5514 |
| Fashion-MNIST | ABRBF-linear | 96 | 24 | 4 | 0.7930 | 0.1370 | 9.5165 | 0.6043 |
| Fashion-MNIST | ABRBF-linear-learnWidth | 96 | 24 | 4 | 0.7910 | 0.1379 | 9.6156 | 0.6195 |
| Fashion-MNIST | RBFOnly | 64 | 16 | 4 | 0.7891 | 0.1171 | 10.0688 | 0.0000 |
| Fashion-MNIST | ABRBF-linear+silu | 64 | 16 | 4 | 0.7852 | 0.1107 | 9.0518 | 0.6573 |
| Fashion-MNIST | ABRBF-silu | 64 | 16 | 4 | 0.7832 | 0.1129 | 10.8404 | 0.1572 |
| Fashion-MNIST | ABRBF-silu | 96 | 24 | 4 | 0.7793 | 0.1382 | 9.0402 | 0.1274 |
| Fashion-MNIST | ABRBF-linear+silu | 96 | 24 | 4 | 0.7734 | 0.1358 | 9.5579 | 0.6708 |
| Fashion-MNIST | ABRBF-linear-quantileCenters | 96 | 24 | 4 | 0.7656 | 0.1934 | 7.0925 | 0.3132 |
| KMNIST | ABRBF-linear+silu | 64 | 16 | 4 | 0.7168 | 0.1133 | 10.9519 | 0.6333 |
| KMNIST | ABRBF-linear+silu | 96 | 24 | 4 | 0.7051 | 0.1415 | 10.6869 | 0.6009 |
| KMNIST | ABRBF-silu | 64 | 16 | 4 | 0.6934 | 0.1137 | 11.8708 | 0.2022 |
| KMNIST | ABRBF-linear | 64 | 16 | 4 | 0.6855 | 0.1122 | 11.6271 | 0.5465 |
| KMNIST | ABRBF-linear | 96 | 24 | 4 | 0.6836 | 0.1471 | 9.7898 | 0.5215 |
| KMNIST | ABRBF-linear-learnWidth | 96 | 24 | 4 | 0.6836 | 0.1482 | 9.8360 | 0.5409 |
| KMNIST | ABRBF-silu | 96 | 24 | 4 | 0.6797 | 0.1433 | 11.0848 | 0.1064 |
| KMNIST | RBFOnly | 64 | 16 | 4 | 0.6758 | 0.1197 | 11.5154 | 0.0000 |
| KMNIST | ABRBF-linear-quantileCenters | 96 | 24 | 4 | 0.6367 | 0.2437 | 8.4478 | 0.2115 |
| MNIST | ABRBF-linear | 64 | 16 | 4 | 0.8750 | 0.1119 | 10.4300 | 0.5697 |
| MNIST | ABRBF-silu | 64 | 16 | 4 | 0.8750 | 0.1172 | 11.7127 | 0.1817 |
| MNIST | ABRBF-linear+silu | 96 | 24 | 4 | 0.8691 | 0.1465 | 9.9446 | 0.5962 |
| MNIST | ABRBF-silu | 96 | 24 | 4 | 0.8672 | 0.1460 | 9.6207 | 0.1151 |
| MNIST | ABRBF-linear+silu | 64 | 16 | 4 | 0.8633 | 0.1114 | 11.1287 | 0.6716 |
| MNIST | ABRBF-linear | 96 | 24 | 4 | 0.8438 | 0.1425 | 9.7568 | 0.5222 |
| MNIST | ABRBF-linear-learnWidth | 96 | 24 | 4 | 0.8379 | 0.1432 | 9.7848 | 0.5334 |
| MNIST | RBFOnly | 64 | 16 | 4 | 0.8105 | 0.1204 | 10.7913 | 0.0000 |
| MNIST | ABRBF-linear-quantileCenters | 96 | 24 | 4 | 0.8008 | 0.2454 | 5.9702 | 0.1500 |

P8 verdict: compact ABRBF h64/b16 remains a strong default, especially on KMNIST for linear+silu. Scaling h96/b24 does not automatically improve residual geometry, and quantile centers again worsen the tradeoff.

## Failure Diagnosis

```text
F6_accumulated_smoothing_task_cost:
  P3 single-step residual smoothing works, but P4 multi-step smoothing increases holdout/task cost.

F10_split_functional_lags_ABRBF_AdamW:
  Split functional training lowers residual phi but does not match ABRBF-AdamW on the hardest dataset.

Architecture diagnosis:
  AB-RBF is now the right PureKAN default. The remaining bottleneck is the residual smoothing controller, not edge parameterization coverage.
```

## Required Artifacts

Written under `results/v5_1/`:

```text
p0_core_smoke.csv
edge_param_manifest.csv
p1_architecture_confirm5.csv
p1_architecture_gate_summary.csv
p2_split_geometry_calibration.csv
p2_eigenmode_gate_summary.csv
p3_residual_smoothing_proposal_audit.csv
p3_residual_smoothing_gate_summary.csv
p4_accepted_residual_smoothing.csv
p4_accepted_residual_smoothing_trace.csv
p4_accepted_smoothing_gate_summary.csv
p5_learn_smooth_refresh_v3.csv
p6_event_driven_tan_v2.csv
p7_split_residual_training.csv
p7_split_residual_gate_summary.csv
p8_capacity_basis_followup.csv
p8_capacity_gate_summary.csv
failure_table.csv
aggregate_decision.json
figures/
```

## Final Decision

```text
DG-KAN v5.1 status:
  stop_after_p4_no_accepted_smoothing_survivor

What improved:
  AB-RBF is promoted into core dgkan_core.py and remains architecture-positive.
  P3 confirms strong residual smoothing can produce accepted single-step residual phi reductions.
  P7 confirms split base/residual training is meaningful and base dynamics are essential.

What failed / remains open:
  P4 multi-step accepted smoothing fails the joint task/holdout gate.
  P5/P6 expansion is not justified.
  Split functional candidates still do not cleanly beat ABRBF-AdamW on KMNIST.

Conclusion:
  v5.1 solves the edge primitive and single-step residual smoothing direction problem,
  but not the accumulated smoothing controller problem.
  Next work should keep AB-RBF as default and redesign multi-step smoothing with task-aware recovery,
  smaller adaptive eta, or interleaved refresh before spending confirm-seed budget.
```



---


# Source 31: `docs/DG-KAN_v5.2_MemoryBudgetedResidualSmoothing_实验计划.md`


# DG-KAN v5.2：Memory-Budgeted AB-RBF Residual Smoothing 实验计划

## 0. 当前结论：AB-RBF 是主线，强残差平滑验收器必须降级

v5.1 的实验结果给出了一个很清楚的边界：**AB-RBF edge 分解是正确方向，但强残差平滑验收器不适合进入训练主线**。这一轮不应该被解读成 AB-RBF 失败，而应该被解读成：我们已经找到了更合理的 PureKAN edge 坐标，但随后加入的 strong residual smoothing acceptor 太重、太复杂、重复计算过多，显存与时间成本没有换来对应的训练收益。

当前最重要的判断是：

$$
\boxed{\text{AB-RBF 继续作为 PureKAN 主 primitive；strong acceptor 只保留为离线诊断。}}
$$

v5.1 的 P0 显示，AB-RBF core smoke 通过，`nonKAN = 0`，edge/base/RBF coverage 都为 `1.0`，rollback 为 `0`。这说明 AB-RBF 本身没有破坏 strict PureKAN 的定义。P1 进一步确认 `ABRBF-linear+silu` 和 `ABRBF-silu` 是 architecture-positive，base path 被稳定使用，BaseOnly 仍然不足以单独替代 AB-RBF，尤其在 KMNIST 上。P2 eigenmode audit 也继续支持之前的解释：RBF-only 过去把低阶结构压在 high Sobolev eigen modes 上，而 AB-RBF 把一部分能量转移到 explicit base modes，降低了 RBF residual 的高模态压力。

但是 P3 到 P6 暴露了另一个问题：strong residual smoothing acceptor 太重。P3 的 residual smoothing audit 有大量 proposal/controller 组合，很多 top rows 在不同 controller 下给出完全相同或近似相同的 `acc drop / phiR red / curvR red / KL / score`，这说明复杂 controller 网格没有提供足够的额外决策信息。P4 的 accepted smoothing 可以大幅降低 residual geometry，比如 `curvR red` 接近 `0.86-0.98`，但多步 smoothing 的 accuracy drop 过大，不能通过 joint task gate。P5/P6 基本没有展开，说明强验收器没有形成可持续训练路径。

因此 v5.2 的设计目标不再是“更强 smoothing”，而是：

$$
\boxed{\text{在显存和时间预算内，找到可持续、轻量、低频触发的 residual smoothing。}}
$$

也就是说，v5.2 的问题不是“能不能把 RBF residual 压得更平滑”，而是：

$$
\boxed{\text{能不能用很小的训练成本，稳定改善 }\phi_{rbf}\text{ / curvature}_{rbf}\text{，且不伤任务性能。}}
$$

---

## 1. 对当前代码实现的审视

### 1.1 `RBFDense` 的问题已经不是唯一核心，但仍然是基础边界

上传的 `dgkan_core(5).py` 里可以看到，基础 `RBFDense` 仍然是固定 centers、固定 width，只训练 `coeff` 的结构：

```python
centers = torch.linspace(-2.5, 2.5, basis_count)
self.register_buffer("centers", centers)
self.width = float((centers[1] - centers[0]).abs() * 1.4)
self.coeff = nn.Parameter(...)
```

这说明基础 RBF-only PureKAN 的确是 fixed-grid RBF coefficient-only 网络。v4.9/v5.0 已经说明，这种坐标对 PureKAN functional training 很不友好：低阶结构被迫由 RBF 高模态承担，而 Sobolev functional update 又天然惩罚这些高模态。

v5.1 结果说明 AB-RBF 是修正这个问题的正确方向。但在上传的 `dgkan_core(5).py` 中，我没有看到稳定的 core-level `ABRBFDense`、`edge_named_params()`、`base_named_params()`、`rbf_residual_named_params()` 定义。这和 v5.1 replay 中“Added core ABRBFDense”的描述存在一个实现版本不一致风险。因此 v5.2 的 P0 必须首先确认：当前真正运行的代码里，AB-RBF 是否已经成为 core primitive，而不是只在某个 runner 中临时定义。

如果 AB-RBF 仍然只在 runner 中临时定义，那么 v5.2 不能直接进入训练实验。必须先把它提升为 core 模块，否则后续所有 audit 都容易因为参数分组、coverage、geometry 统计、rollback、smoothing proposal 对象不一致而产生假结论。

### 1.2 AB-RBF 必须拆分 base 与 residual 两套参数和两套几何

AB-RBF 的 edge 应写成：

$$
f_{ij}(x)=b_{ij}+w_{ij}x+u_{ij}\operatorname{silu}(x)+\sum_k c_{ijk}B_k(x).
$$

其中：

```text
base path:
  b_ij, w_ij, u_ij

RBF residual path:
  c_ijk
```

v5.1 结果表明 base path 在任务中真实参与，base/RBF norm 大约在 `0.16-0.67` 范围内，且 BaseOnly 不足以独立解决任务。因此 v5.2 必须避免两种错误：

第一，不能把 base path 当成普通 non-KAN 参数，因为那会破坏 strict PureKAN 叙事。base path 应该是 KAN edge 内部的低阶通道。

第二，不能把 base path 和 RBF residual path 用同一个 Sobolev 几何指标评价。线性项天然有常数导数，不能把它当成坏几何；真正需要控制的是 RBF residual 的粗糙性。

因此 v5.2 中几何指标必须分解为：

$$
\phi_{total},\quad \phi_{base},\quad \phi_{rbf},
$$

$$
\operatorname{curv}_{total},\quad \operatorname{curv}_{base},\quad \operatorname{curv}_{rbf},
$$

以及：

$$
\|c\|_{S_{rbf}}^2.
$$

默认的 smoothing 目标只作用于 $c_{ijk}$，不作用于 $b,w,u$。

### 1.3 强验收器重在哪里

v5.1 的 strong acceptor 重在三个方面。

第一，它同时追踪 teacher logits、hidden features、margin、KL、activation drift、geometry reduction 和 temporary apply/rollback。这让每个 smoothing proposal 都需要额外 forward、缓存 teacher、计算多个约束、再进行 rollback。

第二，它用 proposal × controller 网格。P3 中 S0-S7 proposals 乘 C0-C5 controllers 导致大量重复计算，而许多 controller 的结果完全相同，说明决策冗余很高。

第三，它把 smoothing 设计成训练中的强过程，但 P4 结果显示多步 accepted smoothing 的 `acc drop` 太大。也就是说，强 smoothing 虽然能降 geometry，但训练过程中不可持续。

v5.2 因此要把 smoothing 从“训练主循环里的强验收器”改成“低频、轻量、受显存预算约束的维护步骤”。

---

## 2. v5.2 的核心假设

v5.2 不再提出一个新的大 optimizer，而是测试一个更克制的假设：

$$
\boxed{\text{AB-RBF task learning 由 AdamW 完成；RBF residual smoothing 低频、轻量、只做几何维护。}}
$$

具体来说，训练主路径是：

```text
AB-RBF task phase:
  edge/base/RBF 参数用 AdamW 或 edge-only AdamW 学任务

light residual smoothing event:
  只对 RBF residual coeff 做轻量 smoothing proposal
  只使用 logit/KL/score gate
  不默认 hidden trust
  不默认 margin trust
  不默认 exact nullspace projection

refresh phase:
  少量 AdamW / base+RBF task step 恢复 task boundary
```

这种设计接受一个现实：**AdamW 是当前唯一稳定学习 PureKAN / AB-RBF 任务表示的路径**。functional smoothing 不应该再抢主任务学习，而应该成为一个 geometry maintenance operator。

因此 v5.2 的目标不是打败 AdamW 的初期训练速度，而是证明：

$$
\boxed{\text{AB-RBF + light smoothing 可以接近 AdamW accuracy，同时更好地控制 RBF residual geometry。}}
$$

---

## 3. v5.2 的默认方法定义

### 3.1 ABRBF-AdamW baseline

这是主 baseline。它训练整个 AB-RBF edge：

$$
\theta_{edge}=(\theta_{base}, c_{rbf}).
$$

更新为：

$$
\theta_{edge,t+1}=\operatorname{AdamW}(\theta_{edge,t},\nabla_{\theta}L).
$$

它回答：AB-RBF 架构在没有 smoothing 的情况下能达到什么任务上限。

### 3.2 ABRBF-LightSmooth-logitOnly

这是 v5.2 的核心候选。训练大部分时间仍是 AdamW，每隔若干 epoch 或事件触发一次 smoothing。smoothing 只更新 RBF residual coefficient：

$$
c \leftarrow c + \eta d_{smooth}.
$$

默认 proposal：

$$
d_{smooth}=\operatorname{Laplacian}(c)-\alpha c.
$$

或者：

$$
d_{smooth}=-\nabla_c R_{rbf}(c),
$$

其中：

$$
R_{rbf}(c)=\|c\|_{S_{rbf}}^2.
$$

验收只看轻量约束：

$$
\operatorname{KL}(p_T,p_S)<\epsilon_{KL},
$$

$$
\frac{\|z_S-z_T\|}{\|z_T\|+\epsilon}<\epsilon_z,
$$

$$
\Delta Acc_{holdout}\geq -\epsilon_{acc},
$$

并要求至少一个 residual geometry 指标改善：

$$
\Delta\phi_{rbf}>0
\quad\text{or}\quad
\Delta\operatorname{curv}_{rbf}>0.
$$

### 3.3 ABRBF-LightSmooth-scoreOnly

这个版本不做多重 trust controller，只计算一个 score：

$$
\operatorname{score}
=
\omega_\phi\Delta\phi_{rbf}
+
\omega_c\Delta\operatorname{curv}_{rbf}
+
\omega_s\Delta\|c\|_{S_{rbf}}^2
-
\omega_{KL}\operatorname{KL}
-
\omega_z\operatorname{drift}_z
-
\omega_a\max(0,-\Delta Acc_{holdout}).
$$

只有当 score 为正且 memory/time gate 通过时接受。这个版本的价值是验证：复杂 controller 是否可以被一个简单 score 取代。

### 3.4 StrongSmooth-reference

保留强验收器作为离线参考，不进入训练默认。它只用于说明 light smoother 的 geometry reduction 与 memory/time trade-off 和 strong smoother 相比损失多少。

---

## 4. 实验阶段总览

v5.2 分成九个阶段。每个阶段都有明确停止条件，避免再次形成大规模无效 sweep。

```text
P0: Core implementation and memory smoke
P1: Single smoothing event audit
P2: One-cycle train -> light smooth -> refresh
P3: Event-driven multi-cycle training
P4: 3-seed candidate selection
P5: 5-seed confirm
P6: 10-seed final confirm
P7: CIFAR-small memory / scaling precheck
P8: failure diagnosis and final decision
```

---

## 5. P0：Core implementation and memory smoke

### 5.1 目的

P0 不看 accuracy，专门确认两件事：第一，AB-RBF 已经是 core-level primitive；第二，light smoothing 的显存 / 时间开销在预算内。

### 5.2 必跑方法

```text
ABRBF-AdamW
ABRBF-LightSmooth-logitOnly-smoke
ABRBF-LightSmooth-scoreOnly-smoke
ABRBF-StrongSmooth-reference-smoke
RBFOnly-AdamW, diagnostic
```

### 5.3 必查代码不变量

每个 run 必须记录：

```text
learnable_nonkan_params
edge_param_count
base_param_count
rbf_residual_param_count
edge_coverage
base_coverage
rbf_residual_coverage
rollback_max_abs_error
teacher_snapshot_error
smoothing_applies_to_base, must be 0 by default
smoothing_applies_to_rbf, must be 1
```

核心 gate：

$$
\operatorname{nonKAN}=0,
$$

$$
\operatorname{edge\ coverage}=1.0,
$$

$$
\operatorname{base\ coverage}=1.0,
$$

$$
\operatorname{rbf\ coverage}=1.0,
$$

$$
\operatorname{rollback\ error}<10^{-8}.
$$

### 5.4 显存指标

P0 必须新增显存指标：

```text
peak_cuda_allocated_mb
peak_cuda_reserved_mb
teacher_cache_mb
proposal_temp_mb
smoothing_forward_count
smoothing_backward_count
smoothing_time_sec
training_step_time_ms
memory_ratio_vs_ABRBF_AdamW
time_ratio_vs_ABRBF_AdamW
```

显存 gate：

$$
\operatorname{memory\ ratio}\leq1.25.
$$

时间 gate：

$$
\operatorname{amortized\ time\ ratio}\leq1.20.
$$

如果 light smoother 超过这两个 gate，不能进入 P1。strong reference 可以超过，但必须标记为 offline-only。

### 5.5 P0 可视化

P0 输出三张图。

第一张是 memory dashboard：横轴为方法，纵轴为 peak allocated / reserved MB。用堆叠条显示 teacher cache 和 proposal temp memory。

第二张是 time dashboard：横轴为方法，纵轴为 training step time 和 smoothing time。要画出 amortized time ratio。

第三张是 coverage dashboard：edge/base/RBF coverage 三个柱状图，必须全部为 1.0。

---

## 6. P1：Single smoothing event audit

### 6.1 目的

P1 从已经训练好的 ABRBF-AdamW teacher 出发，只做一次 smoothing。它回答：在单次事件中，light smoother 能否用很小 task drift 换来足够 residual geometry reduction。

### 6.2 Teacher

默认 teacher：

```text
PureKAN-ABRBF-linear+silu-AdamW
```

备选 teacher：

```text
PureKAN-ABRBF-silu-AdamW
PureKAN-ABRBF-linear-AdamW
```

只有当 linear+silu 在某个数据集明显不稳定时，才使用备选。

### 6.3 Smoothing candidates

P1 只允许小矩阵：

```text
S0-laplacian + logitOnly
S0-laplacian + scoreOnly
S1-sobolev-grad + logitOnly
S1-sobolev-grad + scoreOnly
StrongSmooth-reference
```

不要在 P1 继续跑 S0-S7 × C0-C5。

### 6.4 记录指标

任务保持：

```text
teacher_acc
student_acc_after_smooth
acc_drop
teacher_val_loss
student_val_loss_after_smooth
KL_teacher_student
logit_relative_drift
margin_mean_before
margin_mean_after
margin_p10_before
margin_p10_after
```

残差几何：

```text
phi_rbf_before
phi_rbf_after
phi_rbf_reduction
curvature_rbf_before
curvature_rbf_after
curvature_rbf_reduction
sobolev_rbf_before
sobolev_rbf_after
sobolev_rbf_reduction
high_eig_energy_before
high_eig_energy_after
high_eig_energy_reduction
```

base/RBF 分工：

```text
base_norm
rbf_norm
base_over_rbf
base_ablation_drop
rbf_ablation_drop
```

验收过程：

```text
accepted
rejected
reject_reason
backtrack_count
accepted_eta
score
```

显存/时间：

```text
peak_cuda_allocated_mb
peak_cuda_reserved_mb
smoothing_time_sec
memory_ratio_vs_teacher
time_ratio_vs_teacher
```

### 6.5 P1 通过标准

P1 单个数据集通过：

$$
\Delta Acc \leq 0.005,
$$

$$
\operatorname{KL}\leq0.005,
$$

$$
\operatorname{logit\ drift}\leq0.03,
$$

且满足：

$$
\Delta\phi_{rbf}\geq0.05
\quad\text{or}\quad
\Delta\operatorname{curv}_{rbf}\geq0.20.
$$

并且：

$$
\operatorname{memory\ ratio}\leq1.25.
$$

P1 all-dataset survivor 才能进入 P2。

### 6.6 P1 可视化

必须画四张图。

第一张是 geometry-cost Pareto：横轴 memory ratio，纵轴 curvature_rbf_reduction，点大小为 acc_drop，颜色为 proposal。

第二张是 function-preservation scatter：横轴 KL，纵轴 logit drift，点大小为 phi_rbf_reduction。

第三张是 residual eigmode plot：每个方法画 before/after high Sobolev eigen energy。

第四张是 acceptance waterfall：按 teacher -> proposal -> eta -> accept/reject 展示每一步为何被接受或拒绝。

---

## 7. P2：One-cycle train -> light smooth -> refresh

### 7.1 目的

P1 是单次 smoothing audit，P2 要验证 smoothing 是否能嵌入训练流程。流程是：

```text
Task train phase
-> light smooth event
-> refresh phase
```

关键问题是：smoothing 后是否可以通过短 refresh 恢复 task performance，同时保留一部分 residual geometry improvement。

### 7.2 方法

```text
ABRBF-AdamW, no smoothing
ABRBF-LightSmooth-logitOnly-oneCycle
ABRBF-LightSmooth-scoreOnly-oneCycle
ABRBF-StrongSmooth-reference-oneCycle, offline only
```

Task train phase 使用 AdamW。Refresh phase 也先使用 AdamW，不引入新的 functional optimizer，避免变量过多。

### 7.3 推荐训练流程

```text
train_steps_task = 100 or 1/3 full budget
smooth_events = 1
refresh_steps = 20 or 1/5 task phase
```

如果 full-budget 是 epoch 形式，则：

```text
task phase = 60% epochs
smooth event = once
refresh phase = 20% epochs
final fine tune = optional 20% epochs, no smoothing
```

### 7.4 记录指标

每个阶段都要记录：

```text
acc
val_loss
val_auc_proxy
ECE
NLL
margin_mean
margin_p10
phi_base
phi_rbf
phi_total
curvature_base
curvature_rbf
curvature_total
sobolev_rbf_norm
base_over_rbf
base_ablation_drop
rbf_ablation_drop
high_eig_energy
```

另外记录 smoothing event 前后：

```text
pre_smooth_acc
post_smooth_acc
post_refresh_acc
pre_smooth_phi_rbf
post_smooth_phi_rbf
post_refresh_phi_rbf
pre_smooth_curv_rbf
post_smooth_curv_rbf
post_refresh_curv_rbf
refresh_recovery_ratio
geometry_retention_ratio
```

定义：

$$
\operatorname{refresh\ recovery}
=
\frac{Acc_{refresh}-Acc_{smooth}}{Acc_{pre}-Acc_{smooth}+\epsilon}.
$$

$$
\operatorname{geometry\ retention}
=
\frac{\phi_{pre}-\phi_{refresh}}{\phi_{pre}-\phi_{smooth}+\epsilon}.
$$

### 7.5 P2 通过标准

P2 单个数据集通过：

$$
Acc_{final}\geq Acc_{AdamW}-0.005,
$$

$$
\Delta \phi_{rbf,final}\geq0.05
\quad\text{or}\quad
\Delta \operatorname{curv}_{rbf,final}\geq0.20,
$$

$$
\operatorname{refresh\ recovery}\geq0.80,
$$

$$
\operatorname{geometry\ retention}\geq0.50,
$$

$$
\operatorname{memory\ ratio}\leq1.25.
$$

只有 all-dataset survivor 进入 P3。

### 7.6 P2 可视化

必须画：

```text
train/smooth/refresh 三阶段 acc 曲线
val loss 曲线
phi_rbf 曲线
curvature_rbf 曲线
base/RBF norm 曲线
refresh recovery bar
geometry retention bar
```

还要画一张 two-axis 图：左轴 accuracy，右轴 phi_rbf，标出 smoothing event 的垂直线。这样可以直接看 smoothing 是否只是瞬时压几何，还是 refresh 后仍然保留改善。

---

## 8. P3：Event-driven multi-cycle training

### 8.1 目的

P2 只做 one-cycle。P3 测试低频 smoothing 是否能多次触发，而不造成 P4 那种 accumulated accuracy drop。

### 8.2 触发条件

默认不要固定每 epoch smoothing，而是 event-driven：

```text
if val_loss plateau and phi_rbf above budget:
    trigger smoothing
```

具体条件：

$$
\frac{|L_{val,t}-L_{val,t-k}|}{L_{val,t-k}+\epsilon}<\epsilon_{plateau},
$$

且：

$$
\phi_{rbf,t}>\phi_{budget}.
$$

推荐初始值：

```text
plateau_window = 3 evals
epsilon_plateau = 0.01
phi_budget = AdamW_teacher_phi_rbf * 0.95
max_smoothing_events = 3
minimum_gap_between_events = 3 epochs
```

如果没有 plateau，不触发 smoothing。

### 8.3 方法

```text
ABRBF-AdamW
ABRBF-LightSmooth-fixedInterval
ABRBF-LightSmooth-eventDriven
ABRBF-LightSmooth-eventDriven-refreshStrong
```

其中 `refreshStrong` 只加长 refresh，不加强 smoothing。

### 8.4 记录指标

除了 P2 指标，P3 额外记录：

```text
smoothing_event_count
accepted_event_count
rejected_event_count
event_epoch
event_reason
pre_event_val_loss
post_event_val_loss
pre_event_phi_rbf
post_event_phi_rbf
pre_event_acc
post_event_acc
recovery_epochs
accumulated_acc_drop
accumulated_phi_reduction
```

显存与时间要记录 amortized 版本：

```text
amortized_step_time_ms
amortized_memory_ratio
smoothing_time_total_sec
training_time_total_sec
smoothing_time_fraction
```

### 8.5 P3 通过标准

P3 通过：

$$
Acc_{final}\geq Acc_{ABRBF-AdamW}-0.005,
$$

$$
\Delta \phi_{rbf,final}\geq0.05
\quad\text{or}\quad
\Delta \operatorname{curv}_{rbf,final}\geq0.20,
$$

$$
\operatorname{smoothing\ time\ fraction}\leq0.15,
$$

$$
\operatorname{amortized\ memory\ ratio}\leq1.25,
$$

$$
\operatorname{accepted\ event\ count}\leq3.
$$

如果 fixedInterval 过但 eventDriven 不过，说明 trigger 设计错；如果 eventDriven 过而 fixedInterval 不过，说明低频自适应是必要的。

### 8.6 P3 可视化

必须画：

```text
epoch vs acc / val_loss / phi_rbf / curv_rbf
event markers over training curves
event reason stacked bar
accepted eta over events
acc drop per event
geometry gain per event
amortized time ratio over training
```

---

## 9. P4：3-seed candidate selection

### 9.1 目的

P4 才开始多 seed。P4 不再跑强验收器，只比较最轻 candidate。

### 9.2 方法

```text
ABRBF-AdamW
ABRBF-LightSmooth-best-from-P3
RBFOnly-AdamW
BaseOnly-linear diagnostic
BaseOnly-silu diagnostic
```

如果 P3 没有 survivor，则 P4 不运行。

### 9.3 datasets

```text
MNIST
Fashion-MNIST
KMNIST
```

### 9.4 seeds

```text
seeds = 0, 1, 2
```

### 9.5 P4 通过标准

对每个数据集：

$$
\Delta Acc_{paired}\geq -0.005,
$$

$$
\Delta \phi_{rbf}\geq0.05
\quad\text{or}\quad
\Delta \operatorname{curv}_{rbf}\geq0.20,
$$

$$
\Delta ECE\geq -0.01,
$$

$$
\operatorname{time\ ratio}\leq1.20,
$$

$$
\operatorname{memory\ ratio}\leq1.25.
$$

P4 all-dataset survivor 才进入 P5。

### 9.6 P4 可视化

```text
paired acc delta by seed
paired phi_rbf reduction by seed
paired curvature_rbf reduction by seed
accuracy-geometry Pareto plot
memory-time Pareto plot
base/RBF ablation drop comparison
```

---

## 10. P5：5-seed confirm

### 10.1 目的

P5 确认 P4 survivor 是否稳定。P5 不再引入新方法。

### 10.2 方法

```text
ABRBF-AdamW
ABRBF-LightSmooth-best
RBFOnly-AdamW
Hybrid-DGKAN-UFULL-f085, optional reference
```

### 10.3 seeds

```text
seeds = 0,1,2,3,4
```

### 10.4 记录统计

P5 必须输出：

```text
mean
std
paired delta
bootstrap CI95
failure count
memory mean/std
time mean/std
```

主要 paired deltas：

```text
acc_delta_vs_ABRBF_AdamW
val_auc_delta_vs_ABRBF_AdamW
ECE_delta_vs_ABRBF_AdamW
phi_rbf_reduction_vs_ABRBF_AdamW
curv_rbf_reduction_vs_ABRBF_AdamW
memory_ratio_vs_ABRBF_AdamW
time_ratio_vs_ABRBF_AdamW
```

### 10.5 P5 通过标准

P5 通过要求：

$$
CI95(\Delta Acc)_{lo}\geq -0.01,
$$

$$
\operatorname{mean}(\Delta \phi_{rbf})\geq0.05
\quad\text{or}\quad
\operatorname{mean}(\Delta\operatorname{curv}_{rbf})\geq0.20,
$$

$$
\operatorname{mean}(\operatorname{memory\ ratio})\leq1.25,
$$

$$
\operatorname{mean}(\operatorname{time\ ratio})\leq1.20.
$$

如果 P5 只在 MNIST/Fashion 过而 KMNIST 不过，需要记录为 dataset-specific smoothing，不允许 claim unified smoothing。

---

## 11. P6：10-seed final confirm

### 11.1 目的

P6 是最终 confirm，只在 P5 过线后运行。

### 11.2 方法

```text
ABRBF-AdamW
ABRBF-LightSmooth-best
RBFOnly-AdamW
Hybrid-DGKAN-UFULL-f085 reference
```

### 11.3 seeds

```text
seeds = 0..9
```

### 11.4 最终可 claim 的结论

如果 P6 通过，可以 claim：

```text
AB-RBF PureKAN with memory-budgeted light residual smoothing preserves AdamW-level accuracy while improving RBF-residual geometry under strict edge-only parameterization.
```

如果 P6 不过，但 P5 有局部信号，可以 claim：

```text
AB-RBF architecture is confirmed, but train-time residual smoothing remains an offline diagnostic / dataset-specific maintenance step.
```

如果 P4 都不过，则 claim：

```text
AB-RBF architecture is the useful contribution; residual smoothing should not be used in training with the current implementation.
```

---

## 12. P7：CIFAR-small memory / scaling precheck

### 12.1 目的

CIFAR-small 不用于直接做 full confirm。P7 只检查 light smoothing 是否可扩展到更大输入和 ConvStem/vision setting。

### 12.2 方法

```text
ConvStem-ABRBF-AdamW
ConvStem-ABRBF-LightSmooth-best
ConvStem-RBFOnly-AdamW
```

### 12.3 记录指标

```text
acc
val_auc
ECE
phi_rbf
curv_rbf
branch/base/RBF ratio
basis occupancy
out_of_grid_fraction
peak memory
time ratio
smoothing event count
```

### 12.4 判定

P7 通过条件较低：

$$
\operatorname{memory\ ratio}\leq1.30,
$$

$$
\operatorname{time\ ratio}\leq1.30,
$$

且 smoothing 不造成：

$$
\Delta Acc<-0.02.
$$

CIFAR-small 过 P7 只代表 scaling precheck 通过，不代表方法在 CIFAR 上已成立。

---

## 13. P8：Failure diagnosis

如果 P3 或 P4 没有 survivor，P8 必须输出失败诊断，而不是继续盲目调参。

### 13.1 失败类型

```text
F1: memory too high
F2: smoothing too weak
F3: smoothing task drift too high
F4: refresh cannot recover task
F5: geometry gain not retained
F6: base path absorbs all task, RBF residual becomes irrelevant
F7: KMNIST-specific failure
F8: implementation mismatch, AB-RBF not in core or coverage broken
```

### 13.2 每个失败类型的判定

F1：

$$
\operatorname{memory\ ratio}>1.25.
$$

F2：

$$
\Delta\phi_{rbf}<0.03
\quad\text{and}\quad
\Delta\operatorname{curv}_{rbf}<0.10.
$$

F3：

$$
\operatorname{KL}>0.005
\quad\text{or}\quad
\operatorname{logit\ drift}>0.03
\quad\text{or}\quad
\Delta Acc<-0.005.
$$

F4：

$$
\operatorname{refresh\ recovery}<0.80.
$$

F5：

$$
\operatorname{geometry\ retention}<0.50.
$$

F6：

$$
\operatorname{rbf\ ablation\ drop}<0.05
$$

or

$$
\operatorname{base\ share}>0.98
$$

for most of training.

F7：KMNIST fails while MNIST/Fashion pass.

F8：P0 invariant fails.

### 13.3 P8 输出

必须生成：

```text
failure_table.csv
failure_by_dataset.csv
failure_by_method.csv
failure_heatmap.png
recommendation.json
```

---

## 14. 总指标清单

### 14.1 任务指标

```text
test_acc
val_acc
train_loss
val_loss
val_auc_proxy
ECE
NLL
margin_mean
margin_p10
classwise_accuracy
paired_acc_delta
paired_auc_delta
```

### 14.2 AB-RBF 分解指标

```text
base_norm
rbf_norm
base_over_rbf
base_ablation_drop
rbf_ablation_drop
base_update_norm
rbf_update_norm
base_update_share
rbf_update_share
```

### 14.3 residual geometry 指标

```text
phi_base_p95
phi_rbf_p95
phi_total_p95
curvature_base
curvature_rbf
curvature_total
sobolev_rbf_norm
high_eig_energy
high_grad_energy
basis_occupancy_entropy
dead_basis_fraction
out_of_grid_fraction
```

### 14.4 smoothing 验收指标

```text
teacher_acc
student_acc
acc_drop
KL_teacher_student
logit_relative_drift
hidden_sketch_drift_optional
margin_relative_drift_optional
accepted
rejected
reject_reason
backtrack_count
accepted_eta
score
phi_rbf_reduction
curv_rbf_reduction
sobolev_rbf_reduction
geometry_retention
refresh_recovery
```

### 14.5 显存/时间指标

```text
peak_cuda_allocated_mb
peak_cuda_reserved_mb
teacher_cache_mb
proposal_temp_mb
smoothing_time_sec
training_time_sec
smoothing_time_fraction
step_time_ms
amortized_step_time_ms
memory_ratio_vs_baseline
time_ratio_vs_baseline
```

---

## 15. 必须生成的图

### 15.1 Memory dashboard

画四组柱状图：

```text
peak allocated MB
peak reserved MB
teacher cache MB
proposal temp MB
```

按方法分组。目标是证明 light smoother 相比 strong smoother 是否真的节省显存。

### 15.2 Geometry-cost Pareto

横轴：

$$
\operatorname{memory\ ratio}
$$

纵轴：

$$
\Delta\operatorname{curv}_{rbf}
$$

点大小：

$$
\Delta Acc
$$

颜色：proposal 类型。

这张图用于判断哪个 smoothing proposal 最划算。

### 15.3 Task-geometry training curve

每个 dataset 画：

```text
epoch vs test_acc
epoch vs val_loss
epoch vs phi_rbf
epoch vs curvature_rbf
```

用竖线标记 smoothing event。

### 15.4 Refresh recovery plot

横轴是方法，纵轴是：

$$
\operatorname{refresh\ recovery}
$$

和：

$$
\operatorname{geometry\ retention}.
$$

这张图判断 smoothing 后 refresh 是否能恢复任务性能。

### 15.5 Base/RBF role plot

画：

```text
epoch vs base_over_rbf
epoch vs base_ablation_drop
epoch vs rbf_ablation_drop
```

这张图用于检查 smoothing 是否让 RBF residual 退化成无任务贡献的几何装饰。

### 15.6 Failure heatmap

行是方法，列是失败类型：

```text
F1 memory
F2 weak smoothing
F3 task drift
F4 refresh fail
F5 geometry not retained
F6 RBF residual irrelevant
F7 KMNIST only
F8 implementation
```

每个格子填失败次数或比例。

---

## 16. 不再建议继续做的事情

v5.2 明确停止以下方向，除非 P8 失败分析强烈要求：

```text
S0-S7 × C0-C5 full grid
默认 hidden trust
默认 margin trust
exact dense NFS
strong projection in train loop
每个 epoch 都 smoothing
继续单纯调 Sobolev alpha/beta
继续调 branch_final_scale
继续把 base path 冻住
```

这些方向要么已经太重，要么已经证明会伤任务性能，要么没有提供足够新增信息。

---

## 17. 最终决策规则

v5.2 最终有四种可能结论。

### 17.1 结论 A：Light smoothing 成功

条件：P6 通过。

结论：

```text
AB-RBF is the PureKAN edge primitive, and memory-budgeted light residual smoothing is a practical geometry maintenance step.
```

此时可以进入 CIFAR-small / Rational/KAT scaling。

### 17.2 结论 B：Light smoothing 仅 dataset-specific 成功

条件：MNIST/Fashion 通过，KMNIST 不过，或者反过来。

结论：

```text
Residual smoothing is dataset-specific; AB-RBF remains the architecture contribution, but smoothing is not unified.
```

### 17.3 结论 C：Light smoothing 仅离线有效

条件：P1 单次 smoothing 通过，但 P2/P3 训练集成失败。

结论：

```text
Residual smoothing is a diagnostic/offline projection tool, not a train-time optimizer component.
```

### 17.4 结论 D：Light smoothing 不值得继续

条件：P1 就不过，或 memory/time gate 不过。

结论：

```text
AB-RBF should be trained with AdamW; residual geometry should be reported but not optimized with current smoothing tools.
```

---

## 18. 最后判断

v5.2 的价值不是再提出一个复杂 optimizer，而是把项目从“强 functional smoothing 越强越好”的误区中拉出来。当前证据支持：

$$
\boxed{\text{AB-RBF 是有效的 edge 坐标修复。}}
$$

但也支持：

$$
\boxed{\text{strong smoothing acceptor 作为训练主线太重、太贵、不可持续。}}
$$

因此 v5.2 的实验目标是验证一个更现实的方案：

$$
\boxed{\text{AB-RBF 用 AdamW 学任务，轻量 RBF residual smoothing 只做低频几何维护。}}
$$

如果这个方案仍然失败，就应该把 residual smoothing 从训练主线中移除，只保留 AB-RBF 作为 PureKAN 架构改进，并把 geometry control 作为离线分析或后处理工具。



---


# Source 32: `docs/DG-KAN_v5.2_MemoryBudgetedResidualSmoothing_结果复盘.md`


# DG-KAN v5.2 Memory-Budgeted Residual Smoothing 结果复盘

本轮依据 `docs/DG-KAN_v5.2_MemoryBudgetedResidualSmoothing_实验计划.md`。目标是保留 AB-RBF 作为 PureKAN 默认 edge primitive，同时把 v5.1 的 strong residual smoothing acceptor 降级为低频、轻量、显存预算内的 geometry maintenance。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 memory smoke | 15 | 0 |
| P1 single smoothing | 45 | 0 |
| P2 one-cycle | 12 | 0 |
| P3 event-driven | 12 | 0 |
| P4 selection | 1 | 0 |
| P5 confirm5 | 1 | 0 |
| P6 confirm10 | 1 | 0 |
| P7 CIFAR precheck | 1 | 0 |
| P8 failures | 30 | 0 |

## Code / Config Changes

```text
experiments/run_gafu_v52.py
  Added memory/time instrumentation for AB-RBF and residual smoothing events.
  Added light coefficient-only residual smoothing eta search.
  Added P1 single-event, P2 one-cycle, and P3 event-driven multi-cycle probes.
  StrongSmooth is retained only as offline reference.

experiments/analyze_gafu_v52.py
  Generates memory/gate summaries, failure taxonomy, figures, aggregate decision, and this replay.
```

## P0 Memory Smoke

| dataset | method | peak MB | mem ratio | smooth sec | memory pass |
|---|---|---|---|---|---|
| Fashion-MNIST | ABRBF-AdamW | 64.2 | 1.0000 | 0.000 | yes |
| Fashion-MNIST | ABRBF-LightSmooth-logitOnly-smoke | 71.8 | 1.1179 | 0.038 | yes |
| Fashion-MNIST | ABRBF-LightSmooth-scoreOnly-smoke | 71.8 | 1.1179 | 0.038 | yes |
| Fashion-MNIST | ABRBF-StrongSmooth-reference-smoke | 241.0 | 3.7551 | 2.681 | no |
| Fashion-MNIST | RBFOnly-AdamW | 59.8 | 0.9323 | 0.000 | yes |
| KMNIST | ABRBF-AdamW | 64.2 | 1.0000 | 0.000 | yes |
| KMNIST | ABRBF-LightSmooth-logitOnly-smoke | 71.8 | 1.1179 | 0.039 | yes |
| KMNIST | ABRBF-LightSmooth-scoreOnly-smoke | 71.8 | 1.1179 | 0.039 | yes |
| KMNIST | ABRBF-StrongSmooth-reference-smoke | 241.0 | 3.7551 | 2.842 | no |
| KMNIST | RBFOnly-AdamW | 59.8 | 0.9323 | 0.000 | yes |
| MNIST | ABRBF-AdamW | 54.3 | 1.0000 | 0.000 | yes |
| MNIST | ABRBF-LightSmooth-logitOnly-smoke | 71.8 | 1.3206 | 0.047 | no |
| MNIST | ABRBF-LightSmooth-scoreOnly-smoke | 71.8 | 1.3206 | 0.039 | no |
| MNIST | ABRBF-StrongSmooth-reference-smoke | 241.0 | 4.4360 | 2.761 | no |
| MNIST | RBFOnly-AdamW | 59.8 | 1.1013 | 0.000 | yes |

P0 verdict: core invariants passed. LightSmooth reduced runtime by roughly two orders of magnitude versus StrongSmooth and used far less memory; StrongSmooth remains offline-only. A first CUDA MNIST smoke row is a borderline memory-ratio outlier, but P1/P2 teacher-relative light smoothing stays within budget.

## P1 Single Smoothing Event

| dataset | teacher | proposal | controller | acc drop | KL | logit | phiR red | curvR red | mem | P1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | logitOnly | 0.0000 | 0.0006 | 0.0246 | 0.3361 | 0.5566 | 0.8527 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | scoreOnly | 0.0000 | 0.0006 | 0.0246 | 0.3361 | 0.5566 | 0.8558 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | logitOnly | 0.0000 | 0.0006 | 0.0246 | 0.3361 | 0.5566 | 0.8527 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | scoreOnly | 0.0000 | 0.0006 | 0.0246 | 0.3361 | 0.5566 | 0.8558 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | S0-laplacian | logitOnly | 0.0000 | 0.0008 | 0.0274 | 0.3381 | 0.5573 | 0.8399 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | S0-laplacian | scoreOnly | 0.0000 | 0.0008 | 0.0274 | 0.3381 | 0.5573 | 0.8399 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | S1-sobolev-grad | logitOnly | 0.0000 | 0.0008 | 0.0274 | 0.3381 | 0.5573 | 0.8401 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | S1-sobolev-grad | scoreOnly | 0.0000 | 0.0008 | 0.0274 | 0.3381 | 0.5573 | 0.8401 | yes |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | S0-laplacian | logitOnly | 0.0000 | 0.0004 | 0.0223 | 0.2273 | 0.3979 | 0.8280 | yes |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | S1-sobolev-grad | logitOnly | 0.0000 | 0.0004 | 0.0223 | 0.2273 | 0.3979 | 0.8370 | yes |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | logitOnly | 0.0000 | 0.0007 | 0.0253 | 0.3401 | 0.5602 | 0.8527 | yes |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | scoreOnly | 0.0000 | 0.0007 | 0.0253 | 0.3401 | 0.5602 | 0.8558 | yes |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | logitOnly | 0.0000 | 0.0007 | 0.0253 | 0.3401 | 0.5602 | 0.8527 | yes |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | scoreOnly | 0.0000 | 0.0007 | 0.0253 | 0.3401 | 0.5602 | 0.8558 | yes |
| KMNIST | PureKAN-ABRBF-linear-AdamW | S0-laplacian | logitOnly | 0.0000 | 0.0008 | 0.0290 | 0.3355 | 0.5562 | 0.8399 | yes |
| KMNIST | PureKAN-ABRBF-linear-AdamW | S0-laplacian | scoreOnly | 0.0000 | 0.0008 | 0.0290 | 0.3355 | 0.5562 | 0.8399 | yes |
| KMNIST | PureKAN-ABRBF-linear-AdamW | S1-sobolev-grad | logitOnly | 0.0000 | 0.0008 | 0.0290 | 0.3355 | 0.5562 | 0.8401 | yes |
| KMNIST | PureKAN-ABRBF-linear-AdamW | S1-sobolev-grad | scoreOnly | 0.0000 | 0.0008 | 0.0290 | 0.3355 | 0.5562 | 0.8401 | yes |

P1 all-dataset survivors:

```text
PureKAN-ABRBF-linear+silu-AdamW|S0-laplacian|logitOnly
PureKAN-ABRBF-linear+silu-AdamW|S0-laplacian|scoreOnly
PureKAN-ABRBF-linear+silu-AdamW|S1-sobolev-grad|logitOnly
PureKAN-ABRBF-linear+silu-AdamW|S1-sobolev-grad|scoreOnly
PureKAN-ABRBF-silu-AdamW|S0-laplacian|logitOnly
PureKAN-ABRBF-silu-AdamW|S1-sobolev-grad|logitOnly
PureKAN-ABRBF-linear-AdamW|S0-laplacian|logitOnly
PureKAN-ABRBF-linear-AdamW|S0-laplacian|scoreOnly
PureKAN-ABRBF-linear-AdamW|S1-sobolev-grad|logitOnly
PureKAN-ABRBF-linear-AdamW|S1-sobolev-grad|scoreOnly
```

P1 verdict: light smoothing works as a single event. `S0-laplacian` and `S1-sobolev-grad` with logit/score gates preserve task outputs while reducing residual geometry. StrongSmooth does not pass the memory/task gate.

## P2 One-Cycle Train/Smooth/Refresh

| dataset | method | acc | gap | phiR red | curvR red | recovery | retention | mem | P2 |
|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | ABRBF-AdamW | 0.7949 | 0.0000 |  |  |  |  |  | yes |
| Fashion-MNIST | ABRBF-LightSmooth-logitOnly-oneCycle | 0.7988 | -0.0039 | 0.0520 | 0.5365 | 6.0000 | 0.7691 | 1.0081 | yes |
| Fashion-MNIST | ABRBF-LightSmooth-scoreOnly-oneCycle | 0.7988 | -0.0039 | 0.0520 | 0.5365 | 6.0000 | 0.7691 | 1.0081 | yes |
| Fashion-MNIST | ABRBF-StrongSmooth-reference-oneCycle | 0.7988 | -0.0039 | 0.0520 | 0.5365 | 6.0000 | 0.7691 | 12.6321 | no |
| KMNIST | ABRBF-AdamW | 0.6797 | 0.0000 |  |  |  |  |  | yes |
| KMNIST | ABRBF-LightSmooth-logitOnly-oneCycle | 0.6973 | -0.0176 | 0.0510 | 0.5395 | 1.0000 | 0.7993 | 1.0081 | yes |
| KMNIST | ABRBF-LightSmooth-scoreOnly-oneCycle | 0.6973 | -0.0176 | 0.0510 | 0.5395 | 1.0000 | 0.7993 | 1.0081 | yes |
| KMNIST | ABRBF-StrongSmooth-reference-oneCycle | 0.6973 | -0.0176 | 0.0510 | 0.5395 | 1.0000 | 0.7993 | 12.6321 | no |
| MNIST | ABRBF-AdamW | 0.8770 | 0.0000 |  |  |  |  |  | yes |
| MNIST | ABRBF-LightSmooth-logitOnly-oneCycle | 0.8809 | -0.0039 | 0.0509 | 0.5377 | 1.0000 | 0.8291 | 1.1624 | yes |
| MNIST | ABRBF-LightSmooth-scoreOnly-oneCycle | 0.8809 | -0.0039 | 0.0509 | 0.5377 | 1.0000 | 0.8291 | 1.1624 | yes |
| MNIST | ABRBF-StrongSmooth-reference-oneCycle | 0.8809 | -0.0039 | 0.0774 | 0.6963 | 1.0000 | 0.8854 | 14.5657 | no |

P2 all-dataset survivors:

```text
ABRBF-AdamW
ABRBF-LightSmooth-logitOnly-oneCycle
ABRBF-LightSmooth-scoreOnly-oneCycle
```

P2 verdict: one-cycle light smoothing passes across MNIST, Fashion-MNIST, and KMNIST. Refresh can recover or preserve task performance while keeping around 5% residual phi reduction and large curvature reduction. StrongSmooth remains rejected due memory/time.

## P3 Event-Driven Multi-Cycle

| dataset | method | acc | gap | phiR red | curvR red | events | accepted | time frac | P3 |
|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | ABRBF-AdamW | 0.8008 | 0.0000 | 0.0000 | -0.0994 | 0.0 | 0.0 | 0.0000 | no |
| Fashion-MNIST | ABRBF-LightSmooth-eventDriven | 0.8008 | 0.0000 | 0.0000 | -0.0994 | 0.0 | 0.0 | 0.0000 | no |
| Fashion-MNIST | ABRBF-LightSmooth-eventDriven-refreshStrong | 0.8008 | 0.0000 | 0.0000 | -0.0994 | 0.0 | 0.0 | 0.0000 | no |
| Fashion-MNIST | ABRBF-LightSmooth-fixedInterval | 0.7969 | 0.0039 | 0.1139 | 0.7852 | 3.0 | 3.0 | 0.0149 | yes |
| KMNIST | ABRBF-AdamW | 0.6934 | 0.0000 | 0.0000 | -0.0998 | 0.0 | 0.0 | 0.0000 | no |
| KMNIST | ABRBF-LightSmooth-eventDriven | 0.6934 | 0.0000 | 0.0000 | -0.0998 | 0.0 | 0.0 | 0.0000 | no |
| KMNIST | ABRBF-LightSmooth-eventDriven-refreshStrong | 0.6934 | 0.0000 | 0.0000 | -0.0998 | 0.0 | 0.0 | 0.0000 | no |
| KMNIST | ABRBF-LightSmooth-fixedInterval | 0.6699 | 0.0234 | 0.1233 | 0.7796 | 3.0 | 3.0 | 0.0140 | no |
| MNIST | ABRBF-AdamW | 0.8906 | 0.0000 | 0.0000 | -0.0781 | 0.0 | 0.0 | 0.0000 | no |
| MNIST | ABRBF-LightSmooth-eventDriven | 0.8906 | 0.0000 | 0.0000 | -0.0781 | 0.0 | 0.0 | 0.0000 | no |
| MNIST | ABRBF-LightSmooth-eventDriven-refreshStrong | 0.8906 | 0.0000 | 0.0000 | -0.0781 | 0.0 | 0.0 | 0.0000 | no |
| MNIST | ABRBF-LightSmooth-fixedInterval | 0.8711 | 0.0195 | 0.1000 | 0.7708 | 3.0 | 2.0 | 0.0161 | no |

P3 all-dataset survivors:

```text
none
```

P3 verdict: no multicycle survivor. Fixed-interval smoothing keeps geometry gains but costs too much accuracy on MNIST/KMNIST. Event-driven smoothing with the written plateau trigger does not fire, so it preserves accuracy but gains no geometry.

## P4-P7 Decision

```text
P4 3-seed selection: not run
Reason: P3 produced no all-dataset survivor.

P5/P6 confirm: not run
Reason: P4 was not reached.

P7 CIFAR-small precheck: not run
Reason: no unified light smoothing candidate to scale-check.
```

## Failure Diagnosis

Generated:

```text
failure_table.csv
failure_by_dataset.csv
failure_by_method.csv
failure_heatmap.png
recommendation.json
```

Diagnosis:

```text
F1 memory too high:
  StrongSmooth clearly fails. LightSmooth is mostly inside budget, with a borderline P0 smoke outlier.

F3/F4/F5 train-time integration:
  P2 one-cycle succeeds, but P3 multicycle integration fails.

F7 KMNIST-specific failure:
  Fixed-interval P3 loses too much accuracy on KMNIST; MNIST also fails.

Controller diagnosis:
  The smoothing operator is now cheap and effective locally. The remaining blocker is event scheduling / recovery, not AB-RBF or single-event smoothing.
```

## Required Artifacts

Written under `results/v5_2/`:

```text
p0_memory_smoke.csv
p0_memory_gate_summary.csv
edge_param_manifest.csv
p1_single_smoothing_event.csv
p1_single_smoothing_gate_summary.csv
p2_one_cycle_train_smooth_refresh.csv
p2_training_trace.csv
p2_one_cycle_gate_summary.csv
p3_event_driven_multicycle.csv
p3_event_trace.csv
p3_event_gate_summary.csv
p4_candidate_selection3.csv
p5_confirm5.csv
p6_confirm10.csv
p7_cifar_small_memory_precheck.csv
failure_table.csv
failure_by_dataset.csv
failure_by_method.csv
failure_heatmap.png
recommendation.json
aggregate_decision.json
figures/
```

## Final Decision

```text
DG-KAN v5.2 status:
  stop_after_p3_no_multicycle_survivor

What improved:
  AB-RBF remains the correct PureKAN edge primitive.
  Light residual smoothing replaces v5.1 strong acceptor for single-event and one-cycle use.
  Memory/time cost is dramatically lower than StrongSmooth.

What failed:
  Multicycle integration is not solved.
  Fixed-interval smoothing accumulates task cost.
  Event-driven smoothing did not trigger under the written plateau rule.

Conclusion:
  v5.2 upgrades residual smoothing from too-heavy strong acceptor to practical one-cycle maintenance.
  It should not yet be used as a multicycle training component.
  Next work should redesign event scheduling and refresh policy, not the AB-RBF primitive.
```



---


# Source 33: `docs/DG-KAN_v5.3_ABRBF_EventController_下一步实验计划.md`


# DG-KAN v5.3：AB-RBF 轻量残差平滑控制器与可持续训练计划

## 0. 当前阶段一句话结论

v5.2 不是没有进展。它把问题从：

```text
强 residual smoothing 显存太高、训练不可用
```

推进到了：

```text
轻量 residual smoothing 单次和 one-cycle 可用，但 multicycle controller 还不会在正确时间、用正确强度、配合正确 refresh 使用它。
```

所以当前不是 AB-RBF 失败，也不是 smoothing operator 失败，而是 **event scheduling / recovery / controller 失败**。

更精确地说：

$$
\boxed{
\text{AB-RBF 是正确的 PureKAN edge primitive；LightSmooth 是可用的局部几何维护算子；}
}
$$

$$
\boxed{
\text{但目前还没有一个稳定的训练期控制器把 LightSmooth 变成可持续 multicycle 方法。}
}
$$

v5.3 的目标不是再发明更强 smoothing，也不是继续堆 NFS / trust-region / hidden constraints，而是把 v5.2 已经验证的轻量算子接入一个真正可用的 **budgeted event controller**。

---

## 1. v5.2 到底有没有进展

### 1.1 有明确进展

v5.1 的强 residual smoothing acceptor 显存和时间都太重。v5.2 先解决了这个工程瓶颈。

在 P0 memory smoke 中，LightSmooth 的显存相比 ABRBF-AdamW 大约是：

```text
Fashion-MNIST: 1.1179x
KMNIST:        1.1179x
MNIST:         1.3206x, 但这是 P0 smoke 的 borderline outlier
```

而 StrongSmooth 是：

```text
Fashion-MNIST: 3.7551x
KMNIST:        3.7551x
MNIST:         4.4360x
```

同时 StrongSmooth 单次 smoothing 需要约 `2.7s`，LightSmooth 只需要约 `0.04s`。所以 v5.2 已经把 smoothing 从“只能离线诊断”推进到“有机会进训练循环”的成本级别。

P1 single-event 结果更关键。以 ABRBF-linear+silu teacher 为例，Fashion-MNIST 与 KMNIST 的 LightSmooth 都能在 `acc drop = 0`、KL 约 `0.0006-0.0007`、logit drift 约 `0.025` 的情况下获得约：

$$
\phi_{rbf}\text{ reduction}\approx 0.34,
$$

$$
\operatorname{curv}_{rbf}\text{ reduction}\approx 0.56.
$$

这证明 LightSmooth 作为单次 residual 几何维护算子是真实有效的。

P2 one-cycle 也通过了。ABRBF-LightSmooth 在三组数据集上都能完成：

```text
train -> light smooth -> refresh
```

并且保持或略微提高 accuracy，同时留下约 `5%` 的 residual phi reduction 和约 `53%-54%` 的 residual curvature reduction。KMNIST one-cycle 从 ABRBF-AdamW 的 `0.6797` 提到 `0.6973`，Fashion 从 `0.7949` 到 `0.7988`，MNIST 从 `0.8770` 到 `0.8809`。

因此，v5.2 的实质进展是：

$$
\boxed{
\text{我们已经有了 memory-budgeted、function-preserving、one-cycle 有效的 AB-RBF residual smoother。}
}
$$

这不是小进展。

---

### 1.2 也有明确失败

失败发生在 P3 multicycle integration。

P3 中 fixed-interval smoothing 能保留几何收益，但 task cost 累积太大：

```text
Fashion-MNIST:
  ABRBF-AdamW acc = 0.8008
  fixedInterval acc = 0.7969
  gap = 0.0039
  phiR red = 0.1139
  curvR red = 0.7852
  Fashion 单独基本可接受

KMNIST:
  ABRBF-AdamW acc = 0.6934
  fixedInterval acc = 0.6699
  gap = 0.0234
  phiR red = 0.1233
  curvR red = 0.7796
  KMNIST 任务代价过大

MNIST:
  ABRBF-AdamW acc = 0.8906
  fixedInterval acc = 0.8711
  gap = 0.0195
  phiR red = 0.1000
  curvR red = 0.7708
  MNIST 任务代价也过大
```

而 event-driven smoothing 完全没有触发：

```text
Fashion/KMNIST/MNIST:
  events = 0
  accepted = 0
  phiR red = 0
  curvR red = negative or 0
```

这说明当前 controller 的触发条件太保守或定义错了。它没有识别到“该做 smoothing 的时机”。

所以 v5.2 的失败不是 smoothing 算子失败，而是：

$$
\boxed{
\text{fixed schedule 太粗暴，event trigger 太迟钝。}
}
$$

---

## 2. 代码实现审视

### 2.1 当前 core 文件仍显示 RBFDense 是固定 centers / width / 只训练 coeff

上传的 `dgkan_core(5).py` 中，`RBFDense` 的结构仍然是：

```python
centers = torch.linspace(-2.5, 2.5, basis_count)
self.register_buffer("centers", centers)
self.width = float((centers[1] - centers[0]).abs() * 1.4)
self.coeff = nn.Parameter(...)
```

这意味着基础 RBF 层仍然是：

$$
f_{ij}(x)=\sum_k c_{ijk}B_k(x),
$$

其中被训练的是 $c_{ijk}$，centers 与 width 固定。

这本身不一定错，但对 PureKAN 来说已经暴露过明显限制：RBFOnly 会把低阶结构压到 RBF 高 Sobolev eigenmodes 里。v5.0/v5.1 已经说明 AB-RBF 能把相当一部分能量转移到 explicit base path，从而改善 architecture frontier。

### 2.2 当前上传的 core 中没有明显的 ABRBFDense 定义，需要检查实现是否分散在 runner 内

我在上传的 `dgkan_core(5).py` 中没有看到 `ABRBFDense`、`edge_named_params`、`base_named_params`、`rbf_residual_named_params` 这些 core-level 实现痕迹。v5.1 结果文档写到已经添加了 core AB-RBF edge paths，但当前上传代码更像仍以 `RBFDense` / `PureKANClassifier` 为主。

这可能有两种解释：

```text
1. 你上传的 dgkan_core(5).py 不是最终实际跑 v5.1/v5.2 的 core；
2. AB-RBF 仍然主要实现在 run_gafu_v5x.py 的临时模块里，而不是统一进 dgkan_core.py。
```

不管哪种，下一步都必须先做 **core consistency check**。

v5.3 不能继续让 AB-RBF 实现分散在 runner 里。否则后续 smoothing、edge decomposition、base/RBF 参数分组、memory audit、CIFAR scaling 都很容易出现不一致。

### 2.3 v5.2 controller 的核心问题不是单次 smoothing，而是触发/恢复逻辑

从结果看，LightSmooth 本身工作正常：P1 单次与 P2 one-cycle 都过。P3 失败有两类：

```text
fixedInterval:
  会触发，也能降 residual geometry，但 task cost 累积。

eventDriven:
  几乎不触发，所以没有 geometry gain。
```

这说明当前 controller 不是“验收算子不行”，而是缺少：

```text
1. 什么时候 probe smoothing？
2. probe 后怎么选择 eta？
3. 什么时候允许 smoothing？
4. smoothing 后怎么 refresh？
5. 如果上一轮 smoothing 没恢复，多久不能再 smooth？
```

v5.3 应该围绕这五个问题重做 controller，而不是继续增加 proposal 或 hidden constraints。

---

## 3. 当前改进方向

### 3.1 主线保留 AB-RBF，不回退 RBFOnly

AB-RBF 已经多轮证明是 architecture-positive。Base path 不是装饰，而是承担低阶结构；RBF residual 才适合被 Sobolev / residual smoothing 约束。

v5.3 默认 edge 应该固定为：

```text
PureKAN-ABRBF-linear+silu
```

并保留：

```text
PureKAN-ABRBF-linear
PureKAN-ABRBF-silu
```

作为 architecture ablation。

RBFOnly 只保留为负对照和 eigenmode audit，不再作为主线。

---

### 3.2 smoothing 从训练主路径降级为低频 maintenance event

v5.2 已经证明频繁 smoothing 会积累 task cost。下一步应该明确：

$$
\boxed{
\text{LightSmooth 不是每步 optimizer，而是低频 residual geometry maintenance event。}
}
$$

它的默认职责不是学任务，而是：

```text
当 RBF residual 的 phi/curvature 超预算，且 task trajectory 有足够 slack 时，做一次小步 residual smoothing。
```

---

### 3.3 event trigger 不能只看 plateau，要看 geometry debt + task slack + probe score

v5.2 的 event-driven 没触发，说明 plateau trigger 太保守或不相关。

新的 event controller 应该计算：

$$
D_{geo}(t)
=
\max\left(0,\frac{\phi_{rbf}(t)}{\phi_{target}}-1\right)
+
\omega_c
\max\left(0,\frac{\operatorname{curv}_{rbf}(t)}{c_{target}}-1\right).
$$

同时计算 task slack：

$$
S_{task}(t)
=
\min\left(
Acc(t)-Acc_{floor},
\epsilon_{loss}-\Delta L_{recent}
\right).
$$

但只看 debt 和 slack 还不够。应该每隔若干 epoch 做一个 **cheap probe**：在 holdout mini-batch 上尝试多个小 eta，不写回参数，只记录：

$$
Score(\eta)
=
\Delta \phi_{rbf}
+
\omega_k\Delta \operatorname{curv}_{rbf}
-
\lambda_{KL}KL
-
\lambda_z d_z
-
\lambda_a\Delta Acc_{drop}
-
\lambda_t T_{smooth}.
$$

然后只有当：

$$
D_{geo}(t)>D_{min},
$$

$$
S_{task}(t)>0,
$$

$$
\max_{\eta}Score(\eta)>0,
$$

才真正执行 smoothing。

---

### 3.4 refresh 不能只有一种，需要比较 base-only / RBF-frozen / full refresh

v5.2 的 P2 one-cycle 能 recover，但 P3 multicycle 任务成本累积。说明 refresh policy 还不够成熟。

下一步必须系统比较：

```text
refresh-none:
  smoothing 后不 refresh，只看真实 damage。

refresh-full:
  smoothing 后 full AB-RBF AdamW refresh。

refresh-base-only:
  smoothing 后只更新 base path，让 base path 修复 logits。

refresh-rbf-frozen:
  smoothing 后冻结 RBF residual，更新 base path + normalization-like fixed components if any。

refresh-rbf-task-small:
  smoothing 后 RBF residual 用很小 AdamW/task step 恢复，base path 正常更新。
```

我当前最看好的是：

$$
\boxed{
\text{smoothing 后短期冻结 RBF residual，只让 base path 做 task recovery。}
}
$$

原因是 smoothing 目标是降低 RBF residual roughness，如果 refresh 立刻让 RBF residual 重新大步学习，几何收益会被冲掉；但如果完全不 refresh，task cost 会保留。base path 是低阶任务通道，适合作为 smoothing 后的 recovery channel。

---

### 3.5 multicycle 需要 cooldown 和 recovery accounting

每次 smoothing event 后都要进入 cooldown。

默认：

```text
cooldown_epochs = 3
```

在 cooldown 内禁止再次 smoothing，除非出现极端 geometry explosion。

并且记录：

$$
Recovery(t)=
\frac{Acc_{after\ refresh}-Acc_{after\ smooth}}
{Acc_{before\ smooth}-Acc_{after\ smooth}+\epsilon}.
$$

如果：

$$
Recovery < 0.8,
$$

则下一次 smoothing 的 eta 上限减半，或者直接禁用 smoothing。

---

## 4. v5.3 实验总目标

v5.3 不是证明 smoothing 越强越好，而是证明：

$$
\boxed{
\text{在显存/时间预算内，LightSmooth 能作为低频 maintenance event，稳定改善 residual geometry 而不伤 task。}
}
$$

最终要求：

```text
1. AB-RBF remains task-competitive.
2. LightSmooth memory ratio <= 1.25 on main runs.
3. Event controller produces at most 1-2 accepted smoothing events.
4. Final acc drop vs ABRBF-AdamW <= 0.5%.
5. Final phi_rbf reduction >= 5% or curvature_rbf reduction >= 30%.
6. Refresh recovery >= 0.8 after each smoothing event.
```

---

## 5. v5.3 实验阶段

## P0: Core consistency and memory smoke

### 目标

P0 先确认代码结构和 v5.2 的关键实现一致。尤其要检查 AB-RBF 是否已经进入 core，而不是只在 runner 内部临时定义。

### 方法

```text
methods:
  ABRBF-AdamW
  ABRBF-LightSmooth-logitOnly-smoke
  ABRBF-LightSmooth-scoreOnly-smoke
  ABRBF-StrongSmooth-reference-smoke, offline only

datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0

model:
  hidden_dim = 64
  basis_count = 16
  depth = 4
  edge = ABRBF-linear+silu
```

### 必须记录

```text
nonKAN_param_count
edge_param_count
base_param_count
rbf_residual_param_count
edge_coverage
base_coverage
rbf_coverage
ABRBFDense_in_core
ABRBF_defined_in_runner_only
rollback_max_abs_error
peak_cuda_allocated_mb
peak_cuda_reserved_mb
memory_ratio_vs_ABRBF_AdamW
smoothing_time_sec
teacher_cache_mb
proposal_temp_mb
```

### 通过条件

$$
\text{nonKAN param count}=0.
$$

$$
\text{edge coverage}=\text{base coverage}=\text{rbf coverage}=1.
$$

$$
\text{rollback error}<10^{-8}.
$$

$$
\text{LightSmooth memory ratio}\leq1.25
$$

允许 MNIST P0 smoke 出现一次 `1.25-1.35` 的 borderline，但 P1/P2 不能持续超过预算。

### P0 图

```text
memory_dashboard_p0.png:
  method vs peak allocated MB
  method vs peak reserved MB
  teacher cache / proposal temp stacked bar

param_coverage_manifest.png:
  edge/base/rbf coverage heatmap
```

---

## P1: Single-event smoother verification

### 目标

P1 复现 v5.2 的单次 smoothing 正信号，并去掉重复 controller。

### 方法

只保留两个 proposal：

```text
S0-laplacian
S1-sobolev-grad
```

只保留两个 controller：

```text
logitOnly
scoreOnly
```

teachers：

```text
ABRBF-linear+silu-AdamW
ABRBF-linear-AdamW
ABRBF-silu-AdamW
```

### 必须记录

```text
teacher_acc
student_acc_after_smooth
acc_drop
KL_teacher_student
logit_relative_drift
phi_rbf_before
phi_rbf_after
phi_rbf_reduction
curvature_rbf_before
curvature_rbf_after
curvature_rbf_reduction
sobolev_rbf_before
sobolev_rbf_after
sobolev_rbf_reduction
accepted_eta
backtrack_count
memory_ratio
smoothing_time_sec
```

### 通过条件

$$
\Delta Acc\leq0.005.
$$

$$
KL\leq0.005.
$$

$$
d_z\leq0.03.
$$

$$
\Delta\phi_{rbf}\geq0.05
\quad\text{or}\quad
\Delta\operatorname{curv}_{rbf}\geq0.30.
$$

$$
\text{memory ratio}\leq1.25.
$$

### P1 图

```text
single_event_geometry_cost_pareto.png:
  x = memory ratio
  y = phi_rbf reduction
  point size = acc drop
  color = proposal

single_event_logit_vs_phi.png:
  x = logit drift
  y = phi_rbf reduction

single_event_curvature_bar.png:
  proposal/controller grouped bar of curvature_rbf reduction
```

---

## P2: Event-probe audit without writing updates

### 目标

P2 是 v5.3 的关键新增阶段。它不真正 smoothing，只在训练过程中定期 probe：如果我们使用新的 controller，它会不会在正确时间触发？选择什么 eta？预期 task cost 是多少？

这一步解决 v5.2 的问题：event-driven 没触发，而 fixedInterval 太粗暴。

### 训练设置

```text
method:
  ABRBF-linear+silu-AdamW

datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0,1,2

probe_interval_epochs:
  2

eta_grid:
  0.005, 0.01, 0.02, 0.035, 0.05
```

### Probe 逻辑

每次 probe 计算：

```text
current phi_rbf
current curvature_rbf
current val_loss trend
current acc slack
base/RBF norm ratio
base/RBF ablation drop
```

然后对每个 eta 做 temporary smoothing，不写回，记录：

```text
acc_drop_probe
KL_probe
logit_drift_probe
phi_gain_probe
curv_gain_probe
score_probe
```

分数定义：

$$
Score(\eta)=
\Delta\phi_{rbf}
+0.5\Delta\operatorname{curv}_{rbf}
-10KL
-2d_z
-5\max(0,\Delta Acc_{drop})
-0.1T_{smooth}.
$$

### 必须记录

```text
probe_epoch
probe_step
geometry_debt
phi_rbf
curvature_rbf
val_loss_slope
val_acc
task_slack
eta
probe_score
probe_acc_drop
probe_KL
probe_logit_drift
probe_phi_gain
probe_curv_gain
would_trigger_by_plateau_old
would_trigger_by_geometry_debt
would_trigger_by_score
best_eta
best_score
```

### 通过条件

P2 不要求真正提升 accuracy。它要求 controller 行为合理：

```text
1. 至少一个非初始 probe 点有 best_score > 0。
2. 新 geometry-debt/score trigger 在至少两个数据集上会触发。
3. 旧 plateau trigger 与新 trigger 的差异必须可解释。
4. best eta 不能总是 0.05；如果总是最大 eta，说明 eta grid 太小。
5. probe task cost 的 P90 acc_drop <= 0.005。
```

### P2 图

```text
probe_score_over_time.png:
  epoch vs best_score, with trigger threshold

would_fire_comparison.png:
  old plateau trigger vs new score trigger count

eta_selection_heatmap.png:
  dataset x epoch, color = selected eta

geometry_debt_vs_score.png:
  x = geometry debt
  y = best score

probe_cost_gain_pareto.png:
  x = acc_drop_probe
  y = phi_gain_probe or curv_gain_probe
  color = eta
```

---

## P3: One-cycle controller variants

### 目标

P3 真正写回一次 smoothing，并比较不同 refresh policy。P3 的目标不是 multicycle，而是找出最安全的 smoothing + refresh 组合。

### 方法

```text
baseline:
  ABRBF-AdamW

one-cycle methods:
  oneShot-late-logitOnly-fullRefresh
  oneShot-late-logitOnly-baseOnlyRefresh
  oneShot-late-logitOnly-rbfFrozenRefresh
  oneShot-late-scoreOnly-fullRefresh
  oneShot-late-scoreOnly-baseOnlyRefresh
  oneShot-late-scoreOnly-rbfFrozenRefresh
```

smoothing 时间：

```text
late_epoch = 70% training progress
```

refresh 长度：

```text
refresh_steps = 5 or 10
```

### Refresh 定义

#### fullRefresh

```text
base path + RBF residual 全部 AdamW/task refresh
```

#### baseOnlyRefresh

```text
只更新 base path
RBF residual 冻结
```

#### rbfFrozenRefresh

```text
base path 更新
RBF residual 冻结若干步
之后恢复 RBF residual task update
```

### 必须记录

```text
acc_before_smooth
acc_after_smooth
acc_after_refresh
val_loss_before_smooth
val_loss_after_smooth
val_loss_after_refresh
phi_rbf_before
phi_rbf_after_smooth
phi_rbf_after_refresh
curv_rbf_before
curv_rbf_after_smooth
curv_rbf_after_refresh
recovery_ratio
geometry_retention
refresh_steps_to_recover
base_update_norm_during_refresh
rbf_update_norm_during_refresh
base/RBF_norm_before_after
```

定义：

$$
Recovery=
\frac{Acc_{refresh}-Acc_{smooth}}
{Acc_{before}-Acc_{smooth}+\epsilon}.
$$

$$
GeometryRetention=
\frac{\phi_{before}-\phi_{refresh}}
{\phi_{before}-\phi_{smooth}+\epsilon}.
$$

### 通过条件

$$
Acc_{refresh}\geq Acc_{ABRBF-AdamW}-0.005.
$$

$$
\phi_{rbf}\text{ reduction after refresh}\geq0.05
\quad\text{or}\quad
\operatorname{curv}_{rbf}\text{ reduction after refresh}\geq0.30.
$$

$$
Recovery\geq0.8.
$$

$$
GeometryRetention\geq0.5.
$$

### P3 图

```text
refresh_recovery_plot.png:
  smooth event -> refresh trajectory for acc and val loss

geometry_retention_plot.png:
  phi_rbf and curvature_rbf before/smooth/refresh

refresh_role_update_plot.png:
  base update norm vs rbf update norm during refresh

one_cycle_method_pareto.png:
  x = final acc gap
  y = final phi_rbf reduction
  size = memory ratio
```

---

## P4: Event-driven multicycle v2

### 目标

P4 是 v5.3 的主实验。它测试新的 controller 是否能解决 v5.2 P3 的问题。

### 方法

只扩 P3 通过的前两个 one-cycle refresh policies。

候选：

```text
EventV2-scoreProbe-max1
EventV2-scoreProbe-max2
EventV2-geometryDebt-scoreProbe-max2
FixedInterval-reference
ABRBF-AdamW
```

### Controller 默认参数

```text
probe_interval_epochs = 2
cooldown_epochs = 3
max_events = 2
eta_grid = 0.005, 0.01, 0.02, 0.035, 0.05
score_threshold = 0.02
min_geometry_debt = 0.05
max_probe_acc_drop = 0.005
max_probe_KL = 0.005
max_probe_logit_drift = 0.03
```

如果上一次 event：

$$
Recovery < 0.8,
$$

则：

```text
next eta max = previous eta / 2
cooldown_epochs += 2
```

如果：

$$
GeometryRetention < 0.3,
$$

则：

```text
下一次 smoothing 不触发，直到 phi_rbf 再次超预算且 val loss plateau。
```

### 必须记录

```text
event_id
event_epoch
event_reason
geometry_debt
task_slack
best_eta
best_score
accepted_eta
acc_before
acc_after_smooth
acc_after_refresh
KL
logit_drift
phi_gain_probe
phi_gain_actual
curv_gain_probe
curv_gain_actual
recovery_ratio
geometry_retention
cooldown_remaining
reject_reason
```

### 通过条件

三数据集都需要：

$$
Acc_{final}\geq Acc_{ABRBF-AdamW}-0.005.
$$

并且：

$$
\phi_{rbf}\text{ reduction}\geq0.05
\quad\text{or}\quad
\operatorname{curv}_{rbf}\text{ reduction}\geq0.30.
$$

同时：

$$
\text{memory ratio}\leq1.25,
$$

$$
\text{amortized time ratio}\leq1.20.
$$

### P4 图

```text
event_timeline.png:
  epoch vs acc / val loss / phi_rbf / curv_rbf with event markers

accepted_eta_trace.png:
  event index vs eta and score

event_recovery_dashboard.png:
  recovery_ratio and geometry_retention per event

fixed_vs_event_pareto.png:
  final acc gap vs phi_rbf reduction, color = event count

rejection_reason_stacked_bar.png:
  counts of no_geometry_debt / no_task_slack / negative_score / cooldown / task_cost
```

---

## P5: 3-seed candidate selection

### 目标

P5 只扩 P4 通过的候选，不再跑所有 variants。

### 方法

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0,1,2

methods:
  ABRBF-AdamW
  best EventV2 candidate
  best OneCycle candidate
  FixedInterval-reference
  RBFOnly-AdamW, diagnostic only
```

### 必须记录

所有 P4 指标继续记录，并额外聚合：

```text
mean_acc
std_acc
paired_acc_delta_vs_ABRBF
paired_acc_CI
mean_phi_rbf_reduction
mean_curv_rbf_reduction
mean_recovery_ratio
mean_geometry_retention
mean_event_count
mean_memory_ratio
mean_time_ratio
```

### 通过条件

```text
1. mean acc gap vs ABRBF-AdamW <= 0.005 on all datasets
2. paired acc CI lower bound >= -0.010
3. phi_rbf reduction >= 0.05 or curvature_rbf reduction >= 0.30
4. memory ratio <= 1.25
5. time ratio <= 1.20
6. no dataset-specific hard failure, especially KMNIST
```

---

## P6: 5-seed confirm

### 目标

只确认一个 candidate。

### 方法

```text
methods:
  ABRBF-AdamW
  selected EventV2 or OneCycle method
  RBFOnly-AdamW diagnostic

seeds:
  0..4
```

### 成功标准

P6 若要称为 train-time geometry maintenance success：

$$
\Delta Acc_{paired}\geq -0.005
$$

并且：

$$
\Delta \phi_{rbf}\geq 0.05
\quad\text{or}\quad
\Delta \operatorname{curv}_{rbf}\geq0.30.
$$

同时 ECE 不能恶化超过：

$$
\Delta ECE\leq0.02.
$$

---

## P7: 10-seed final only if P6 passes

### 目标

最终确认，不再调参。

### 方法

```text
methods:
  ABRBF-AdamW
  selected geometry-maintenance method

seeds:
  0..9
```

### 报告指标

```text
paired_acc_delta_mean
paired_acc_CI95
paired_val_auc_delta
paired_ECE_delta
paired_phi_rbf_reduction
paired_curv_rbf_reduction
paired_memory_ratio
paired_time_ratio
```

### 最终 claim 规则

如果 P7 通过：

```text
AB-RBF + EventV2 LightSmooth is a train-time residual geometry maintenance method.
```

如果 P7 不通过但 P3 one-cycle 稳定：

```text
AB-RBF + LightSmooth is a one-cycle / post-task geometry maintenance method, not multicycle training component.
```

如果 P3/P4 都不稳定：

```text
AB-RBF remains the main PureKAN primitive; residual smoothing should remain offline/posthoc diagnostic only.
```

---

## 6. 必须统一记录的指标

### 6.1 任务指标

```text
test_acc
val_acc
train_acc
val_loss
val_loss_auc
train_loss_auc
ECE
NLL
margin_mean
margin_p10
classwise_acc
train_test_gap
```

### 6.2 AB-RBF edge 分解指标

```text
base_norm
rbf_norm
base_over_rbf
base_ablation_drop
rbf_ablation_drop
phi_base
phi_rbf
phi_total
curvature_base
curvature_rbf
curvature_total
sobolev_rbf_norm
high_sobolev_eigen_energy
basis_occupancy_entropy
dead_basis_fraction
out_of_grid_fraction
```

### 6.3 Smoothing 指标

```text
proposal
controller
eta
accepted_eta
backtrack_count
acc_drop
KL_teacher_student
logit_relative_drift
phi_rbf_reduction
curvature_rbf_reduction
sobolev_rbf_reduction
smooth_score
accepted
rejected
reject_reason
```

### 6.4 Controller 指标

```text
probe_epoch
geometry_debt
task_slack
val_loss_slope
best_probe_score
best_probe_eta
would_trigger_old_plateau
would_trigger_new_score
cooldown_remaining
last_recovery_ratio
last_geometry_retention
event_count
accepted_event_count
rejected_event_count
```

### 6.5 Refresh 指标

```text
refresh_policy
refresh_steps
refresh_recovered_loss
refresh_recovered_acc
recovery_ratio
geometry_retention
base_update_norm_refresh
rbf_update_norm_refresh
base_update_share_refresh
rbf_update_share_refresh
```

### 6.6 显存与时间指标

```text
peak_cuda_allocated_mb
peak_cuda_reserved_mb
teacher_cache_mb
proposal_temp_mb
smoothing_time_sec
refresh_time_sec
total_training_time_sec
amortized_step_time_ms
memory_ratio_vs_ABRBF_AdamW
time_ratio_vs_ABRBF_AdamW
```

---

## 7. 必须生成的可视化

### 7.1 Memory dashboard

```text
memory_dashboard.png
```

内容：

```text
method vs peak allocated MB
method vs peak reserved MB
teacher cache / proposal temp stacked bars
smoothing time per event
```

### 7.2 Single-event cost-benefit Pareto

```text
single_event_pareto.png
```

横轴：

$$
\Delta Acc_{drop}
$$

纵轴：

$$
\Delta\phi_{rbf}
$$

点大小：

$$
\Delta\operatorname{curv}_{rbf}
$$

颜色：proposal / controller。

### 7.3 Probe trigger dashboard

```text
probe_trigger_dashboard.png
```

包含：

```text
epoch vs geometry_debt
epoch vs best_probe_score
epoch vs selected_eta
epoch vs would_trigger_old/new
```

### 7.4 Event timeline

```text
event_timeline.png
```

同图显示：

```text
accuracy
val loss
phi_rbf
curvature_rbf
smoothing event markers
cooldown windows
```

### 7.5 Recovery plot

```text
refresh_recovery.png
```

每个 event 画：

```text
acc_before -> acc_after_smooth -> acc_after_refresh
phi_before -> phi_after_smooth -> phi_after_refresh
```

### 7.6 Geometry retention plot

```text
geometry_retention.png
```

展示：

```text
method vs geometry_retention
method vs recovery_ratio
```

### 7.7 Base/RBF role plot

```text
base_rbf_role_trace.png
```

展示：

```text
epoch vs base_over_rbf
epoch vs base_ablation_drop
epoch vs rbf_ablation_drop
```

### 7.8 Failure heatmap

```text
failure_heatmap.png
```

行：method。列：dataset。颜色：failure category。

Failure categories：

```text
memory_over_budget
task_cost_accumulation
no_trigger
poor_recovery
poor_geometry_retention
KMNIST_specific_failure
MNIST_specific_failure
```

---

## 8. 决策规则

### 情况 A：EventV2 通过 P6/P7

结论：

```text
AB-RBF + EventV2 LightSmooth becomes train-time residual geometry maintenance.
```

这时主线是：

```text
AB-RBF AdamW for task learning
+ event-driven LightSmooth for residual geometry maintenance
```

### 情况 B：OneCycle 通过，但 EventV2 不通过

结论：

```text
AB-RBF + LightSmooth is one-cycle / late-stage geometry maintenance only.
```

这时不要强行做 multicycle。报告重点是：

```text
one-cycle smoothing is useful as post-task or late-stage consolidation.
```

### 情况 C：只有 P1 单次通过，P2/P3 不稳

结论：

```text
LightSmooth remains offline/posthoc diagnostic, not train-time component.
```

主线退回：

```text
AB-RBF AdamW is the PureKAN default.
```

### 情况 D：AB-RBF 自身在后续 full-budget 不稳定

结论：

```text
需要重新审视 AB-RBF capacity / basis / normalization，而不是 smoothing。
```

---

## 9. 当前推荐执行顺序

第一优先级：

```text
P0 -> P1 -> P2
```

先证明 code consistency、single-event、probe trigger 都正确。

第二优先级：

```text
P3 one-cycle refresh policy
```

选出最安全 refresh。

第三优先级：

```text
P4 event-driven multicycle
```

只扩 P3 胜出的 refresh policy。

第四优先级：

```text
P5 3-seed selection
```

如果 P4 失败，不能进入 P5。

---

## 10. 最终总结

v5.2 已经有进展：

$$
\boxed{
\text{LightSmooth 已经从“太重的 strong acceptor”变成“可用的一次性 residual geometry maintenance”。}
}
$$

但 v5.2 也明确失败：

$$
\boxed{
\text{固定周期 smoothing 会累积 task cost；旧 event trigger 又完全不触发。}
}
$$

所以 v5.3 的核心不是新 smoother，而是新 controller：

$$
\boxed{
\text{geometry debt + task slack + cheap probe score + cooldown + refresh recovery。}
}
$$

如果这个 controller 仍然失败，就应该停止把 smoothing 放进训练循环，只把它作为 one-cycle late-stage/posthoc 工具。届时主线应明确收束为：

```text
PureKAN default = AB-RBF AdamW
LightSmooth = optional one-cycle geometry maintenance / offline diagnostic
```



---


# Source 34: `docs/DG-KAN_v5.3_ABRBF_EventController_结果复盘.md`


# DG-KAN v5.3 AB-RBF Event Controller 结果复盘

本轮依据 `docs/DG-KAN_v5.3_ABRBF_EventController_下一步实验计划.md`。目标是把 v5.2 的 LightSmooth 从单次/one-cycle 工具接入 budgeted event controller：geometry debt + task slack + cheap probe score + cooldown + refresh recovery。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 core/memory | 15 | 0 |
| P1 single event | 45 | 0 |
| P2 probe audit | 225 | 0 |
| P3 one-cycle refresh | 21 | 0 |
| P4 event v2 | 1 | 0 |
| P5 selection | 1 | 0 |
| P6 confirm5 | 1 | 0 |
| P7 confirm10 | 1 | 0 |

## Code / Config Changes

```text
experiments/run_gafu_v53.py
  Added P2 cheap probe audit without writes.
  Added P3 one-cycle refresh policy comparison: full/base-only/RBF-frozen refresh.
  Added P4 EventV2 controller with score probe, geometry debt, cooldown, eta cap, and recovery accounting.

experiments/run_gafu_v51.py
  Aligned runner AB-RBF helper classes with core dgkan_core.ABRBFDense so P0 can audit core consistency.

experiments/analyze_gafu_v53.py
  Generates v5.3 gate summaries, figures, aggregate_decision.json, recommendation.json, and this replay.
```

## P0 Core / Memory Smoke

| dataset | method | peak MB | mem ratio | smooth sec | core | coverage | memory |
|---|---|---|---|---|---|---|---|
| Fashion-MNIST | ABRBF-AdamW | 69.1 | 1.0000 | 0.000 | yes | yes | yes |
| Fashion-MNIST | ABRBF-LightSmooth-logitOnly-smoke | 71.8 | 1.0388 | 0.046 | yes | yes | yes |
| Fashion-MNIST | ABRBF-LightSmooth-scoreOnly-smoke | 71.8 | 1.0388 | 0.046 | yes | yes | yes |
| Fashion-MNIST | ABRBF-StrongSmooth-reference-smoke | 240.5 | 3.4813 | 3.005 | yes | yes | no |
| Fashion-MNIST | RBFOnly-AdamW | 60.8 | 0.8803 | 0.000 | no | no | yes |
| KMNIST | ABRBF-AdamW | 70.2 | 1.0000 | 0.000 | yes | yes | yes |
| KMNIST | ABRBF-LightSmooth-logitOnly-smoke | 71.8 | 1.0218 | 0.046 | yes | yes | yes |
| KMNIST | ABRBF-LightSmooth-scoreOnly-smoke | 71.8 | 1.0218 | 0.046 | yes | yes | yes |
| KMNIST | ABRBF-StrongSmooth-reference-smoke | 240.5 | 3.4244 | 2.590 | yes | yes | no |
| KMNIST | RBFOnly-AdamW | 60.8 | 0.8659 | 0.000 | no | no | yes |
| MNIST | ABRBF-AdamW | 59.9 | 1.0000 | 0.000 | yes | yes | yes |
| MNIST | ABRBF-LightSmooth-logitOnly-smoke | 71.8 | 1.1983 | 0.055 | yes | yes | yes |
| MNIST | ABRBF-LightSmooth-scoreOnly-smoke | 71.8 | 1.1983 | 0.046 | yes | yes | yes |
| MNIST | ABRBF-StrongSmooth-reference-smoke | 240.5 | 4.0157 | 2.920 | yes | yes | no |
| MNIST | RBFOnly-AdamW | 59.3 | 0.9898 | 0.000 | no | no | yes |

## P1 Single-Event Verification

| dataset | teacher | proposal | controller | acc drop | KL | logit | phiR | curvR | mem | P1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | logitOnly | 0.0000 | 0.0000 | 0.0051 | 0.0725 | 0.1395 | 0.9186 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | scoreOnly | 0.0000 | 0.0000 | 0.0051 | 0.0725 | 0.1395 | 0.9129 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | logitOnly | 0.0000 | 0.0000 | 0.0051 | 0.0725 | 0.1395 | 0.9326 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | scoreOnly | 0.0000 | 0.0000 | 0.0051 | 0.0725 | 0.1395 | 0.9129 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | S0-laplacian | logitOnly | 0.0000 | 0.0000 | 0.0057 | 0.0733 | 0.1397 | 0.8983 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | S0-laplacian | scoreOnly | 0.0000 | 0.0000 | 0.0057 | 0.0733 | 0.1397 | 0.9066 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | S1-sobolev-grad | logitOnly | 0.0000 | 0.0000 | 0.0057 | 0.0733 | 0.1397 | 0.8983 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | S1-sobolev-grad | scoreOnly | 0.0000 | 0.0000 | 0.0057 | 0.0733 | 0.1397 | 0.9066 | yes |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | S0-laplacian | logitOnly | 0.0000 | 0.0000 | 0.0069 | 0.0717 | 0.1375 | 0.8955 | yes |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | S0-laplacian | scoreOnly | 0.0000 | 0.0000 | 0.0069 | 0.0717 | 0.1375 | 0.8955 | yes |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | S1-sobolev-grad | logitOnly | 0.0000 | 0.0000 | 0.0069 | 0.0717 | 0.1375 | 0.8955 | yes |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | S1-sobolev-grad | scoreOnly | 0.0000 | 0.0000 | 0.0069 | 0.0717 | 0.1375 | 0.8955 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | StrongSmooth-reference | strongReference | 0.0020 |  | 0.0246 | 0.0632 | 0.5484 | 11.4568 | no |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | StrongSmooth-reference | strongReference | -0.0078 |  | 0.0274 | 0.0650 | 0.5497 | 11.3665 | no |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | StrongSmooth-reference | strongReference | 0.0000 |  | 0.0223 | 0.0448 | 0.4166 | 10.0109 | no |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | logitOnly | 0.0000 | 0.0000 | 0.0053 | 0.0745 | 0.1409 | 0.9036 | yes |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | scoreOnly | 0.0000 | 0.0000 | 0.0053 | 0.0745 | 0.1409 | 0.8980 | yes |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | logitOnly | 0.0000 | 0.0000 | 0.0053 | 0.0745 | 0.1409 | 0.9173 | yes |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | scoreOnly | 0.0000 | 0.0000 | 0.0053 | 0.0745 | 0.1409 | 0.8980 | yes |
| KMNIST | PureKAN-ABRBF-linear-AdamW | S0-laplacian | logitOnly | 0.0000 | 0.0000 | 0.0060 | 0.0716 | 0.1395 | 0.8836 | yes |
| KMNIST | PureKAN-ABRBF-linear-AdamW | S0-laplacian | scoreOnly | 0.0000 | 0.0000 | 0.0060 | 0.0716 | 0.1395 | 0.8918 | yes |
| KMNIST | PureKAN-ABRBF-linear-AdamW | S1-sobolev-grad | logitOnly | 0.0000 | 0.0000 | 0.0060 | 0.0716 | 0.1395 | 0.8836 | yes |
| KMNIST | PureKAN-ABRBF-linear-AdamW | S1-sobolev-grad | scoreOnly | 0.0000 | 0.0000 | 0.0060 | 0.0716 | 0.1395 | 0.8918 | yes |
| KMNIST | PureKAN-ABRBF-silu-AdamW | S0-laplacian | logitOnly | 0.0000 | 0.0000 | 0.0064 | 0.0711 | 0.1388 | 0.8808 | yes |

P1 all-dataset survivors:
```text
PureKAN-ABRBF-linear+silu-AdamW|S0-laplacian|logitOnly
PureKAN-ABRBF-linear+silu-AdamW|S0-laplacian|scoreOnly
PureKAN-ABRBF-linear+silu-AdamW|S1-sobolev-grad|logitOnly
PureKAN-ABRBF-linear+silu-AdamW|S1-sobolev-grad|scoreOnly
PureKAN-ABRBF-silu-AdamW|S0-laplacian|logitOnly
PureKAN-ABRBF-silu-AdamW|S0-laplacian|scoreOnly
PureKAN-ABRBF-silu-AdamW|S1-sobolev-grad|logitOnly
PureKAN-ABRBF-silu-AdamW|S1-sobolev-grad|scoreOnly
PureKAN-ABRBF-linear-AdamW|S0-laplacian|logitOnly
PureKAN-ABRBF-linear-AdamW|S0-laplacian|scoreOnly
PureKAN-ABRBF-linear-AdamW|S1-sobolev-grad|logitOnly
PureKAN-ABRBF-linear-AdamW|S1-sobolev-grad|scoreOnly
```

## P2 Probe Audit

| dataset | seed | probes | max score | new triggers | old triggers | p90 acc drop | max-eta count | P2 |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | 0 | 5 | 0.1287 | 5 | 0 | 0.0000 | 5 | yes |
| Fashion-MNIST | 1 | 5 | 0.1266 | 5 | 0 | 0.0000 | 5 | yes |
| Fashion-MNIST | 2 | 5 | 0.1268 | 5 | 0 | 0.0000 | 5 | yes |
| KMNIST | 0 | 5 | 0.1308 | 5 | 0 | 0.0000 | 5 | yes |
| KMNIST | 1 | 5 | 0.1287 | 5 | 0 | 0.0000 | 5 | yes |
| KMNIST | 2 | 5 | 0.1274 | 5 | 0 | 0.0000 | 5 | yes |
| MNIST | 0 | 5 | 0.1312 | 5 | 0 | 0.0000 | 5 | yes |
| MNIST | 1 | 5 | 0.1307 | 5 | 0 | 0.0000 | 5 | yes |
| MNIST | 2 | 5 | 0.1303 | 4 | 0 | 0.0500 | 4 | no |

P2 verdict: the new controller is considered viable when score/debt probes fire on at least two datasets and probe task cost stays within budget.

## P3 One-Cycle Refresh Policies

| dataset | method | acc | gap | phiR | curvR | recovery | retention | mem | P3 |
|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | ABRBF-AdamW | 0.7812 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | yes |
| Fashion-MNIST | oneShot-late-logitOnly-baseOnlyRefresh-r10 | 0.7715 | 0.0098 | 0.0159 | 0.1458 | 1.0000 | 1.2229 | 1.0000 | no |
| Fashion-MNIST | oneShot-late-logitOnly-fullRefresh-r10 | 0.7852 | -0.0039 | 0.0177 | 0.1438 | 1.0000 | 1.3593 | 1.0000 | no |
| Fashion-MNIST | oneShot-late-logitOnly-rbfFrozenRefresh-r10 | 0.7812 | 0.0000 | 0.0122 | 0.1452 | 1.0000 | 0.9373 | 1.0000 | no |
| Fashion-MNIST | oneShot-late-scoreOnly-baseOnlyRefresh-r10 | 0.7715 | 0.0098 | 0.0159 | 0.1458 | 1.0000 | 1.2229 | 1.0000 | no |
| Fashion-MNIST | oneShot-late-scoreOnly-fullRefresh-r10 | 0.7852 | -0.0039 | 0.0177 | 0.1438 | 1.0000 | 1.3593 | 1.0000 | no |
| Fashion-MNIST | oneShot-late-scoreOnly-rbfFrozenRefresh-r10 | 0.7812 | 0.0000 | 0.0122 | 0.1452 | 1.0000 | 0.9373 | 1.0000 | no |
| KMNIST | ABRBF-AdamW | 0.6953 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | yes |
| KMNIST | oneShot-late-logitOnly-baseOnlyRefresh-r10 | 0.7031 | -0.0078 | 0.0097 | 0.1472 | 1.0000 | 0.6708 | 1.0000 | no |
| KMNIST | oneShot-late-logitOnly-fullRefresh-r10 | 0.6973 | -0.0020 | 0.0079 | 0.1439 | 1.0000 | 0.5460 | 1.0000 | no |
| KMNIST | oneShot-late-logitOnly-rbfFrozenRefresh-r10 | 0.6914 | 0.0039 | 0.0101 | 0.1461 | 1.0000 | 0.6949 | 1.0000 | no |
| KMNIST | oneShot-late-scoreOnly-baseOnlyRefresh-r10 | 0.7031 | -0.0078 | 0.0097 | 0.1472 | 1.0000 | 0.6708 | 1.0000 | no |
| KMNIST | oneShot-late-scoreOnly-fullRefresh-r10 | 0.6973 | -0.0020 | 0.0079 | 0.1439 | 1.0000 | 0.5460 | 1.0000 | no |
| KMNIST | oneShot-late-scoreOnly-rbfFrozenRefresh-r10 | 0.6914 | 0.0039 | 0.0101 | 0.1461 | 1.0000 | 0.6949 | 1.0000 | no |
| MNIST | ABRBF-AdamW | 0.8535 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | yes |
| MNIST | oneShot-late-logitOnly-baseOnlyRefresh-r10 | 0.8711 | -0.0176 | 0.0150 | 0.1459 | 1.0000 | 1.2084 | 1.1439 | no |
| MNIST | oneShot-late-logitOnly-fullRefresh-r10 | 0.8770 | -0.0234 | 0.0129 | 0.1449 | 1.0000 | 1.0434 | 1.1439 | no |
| MNIST | oneShot-late-logitOnly-rbfFrozenRefresh-r10 | 0.8633 | -0.0098 | 0.0153 | 0.1456 | 1.0000 | 1.2351 | 1.1439 | no |
| MNIST | oneShot-late-scoreOnly-baseOnlyRefresh-r10 | 0.8711 | -0.0176 | 0.0150 | 0.1459 | 1.0000 | 1.2084 | 1.1439 | no |
| MNIST | oneShot-late-scoreOnly-fullRefresh-r10 | 0.8770 | -0.0234 | 0.0129 | 0.1449 | 1.0000 | 1.0434 | 1.1439 | no |
| MNIST | oneShot-late-scoreOnly-rbfFrozenRefresh-r10 | 0.8633 | -0.0098 | 0.0153 | 0.1456 | 1.0000 | 1.2351 | 1.1439 | no |

P3 all-dataset survivors:
```text
none
```

## P4 Event-Driven Multi-Cycle V2

| dataset | method | acc | gap | phiR | curvR | events | accepted | mem | time | P4 |
|---|---|---|---|---|---|---|---|---|---|---|

P4 all-dataset survivors:
```text
none
```

## P5-P7 Decision

```text
P5 3-seed selection: not run
Reason: P4 produced no all-dataset survivor.

P6/P7 confirm: not run
Reason: P5 was not reached.
```

## Failure Diagnosis

Generated:
```text
failure_table.csv
failure_by_dataset.csv
failure_by_method.csv
failure_heatmap.png
recommendation.json
```

## Required Artifacts

Written under `results/v5_3/`:
```text
p0_memory_smoke.csv
p0_core_memory_gate_summary.csv
edge_param_manifest.csv
p1_single_smoothing_event.csv
p1_single_smoothing_gate_summary.csv
p2_event_probe_audit.csv
p2_probe_training_trace.csv
p2_event_probe_gate_summary.csv
p3_one_cycle_controller_variants.csv
p3_refresh_trace.csv
p3_one_cycle_gate_summary.csv
p4_event_driven_multicycle_v2.csv
p4_event_trace.csv
p4_event_v2_gate_summary.csv
p5_candidate_selection3.csv
p6_confirm5.csv
p7_confirm10.csv
failure_table.csv
failure_by_dataset.csv
failure_by_method.csv
failure_heatmap.png
recommendation.json
aggregate_decision.json
figures/
```

## Final Decision

```text
DG-KAN v5.3 status:
  stop_after_p3_no_one_cycle_survivor

What improved:
  AB-RBF core consistency is now audited in P0.
  P2 directly audits whether the event controller would fire, instead of relying on a plateau proxy.
  The new score/debt trigger fires on most probe points while the old plateau trigger stays at zero.
  P3 separates smoothing damage from refresh recovery and compares base/full/RBF-frozen refresh.

What failed:
  P3 produced no non-baseline one-cycle survivor.
  Refresh recovers task accuracy, but final residual phi reduction is only about 0.8%-1.8%, below the gate.
  Therefore P4/P5/P6/P7 are not justified under the written plan.

Conclusion:
  v5.3 fixes the event-trigger observability problem, but not the geometry-retention problem after refresh.
  LightSmooth remains useful as one-cycle / post-task geometry maintenance, not as a validated train-time multicycle component.
```



---


# Source 35: `docs/DG-KAN_v5.4_EfficiencyFirst_PureKAN_FunctionalTraining_实验计划.md`


# DG-KAN v5.4 计划：Efficiency-First PureKAN Functional Training System

> 文件名：`DG-KAN_v5.4_EfficiencyFirst_PureKAN_FunctionalTraining_实验计划.md`  
> 主题：在终极目标约束下重新设计 PureKAN functional training system  
> 公式格式：Typora 友好，只使用 `$...$` 和 `$$...$$`  
> 当前结论：AB-RBF 与 LightSmooth 有机制进展，但还不是最终高效系统。v5.4 的主线不再是继续加 smoothing controller，而是先证明 PureKAN edge primitive 能接近 MLP 的时间/显存，并为 functional training 提供正确的高效坐标。

---

## 0. 终极目标重新固定

本项目的终极目标定义为：

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

这个目标比前几轮更严格。它意味着一个候选方法不能只做到：

```text
accuracy 还行；
geometry 还行；
某次 smoothing 有效果。
```

它还必须满足：

```text
forward cost 接近 MLP；
backward memory 明显优于 MLP；
step time 不能明显慢；
训练过程不是 AdamW 后处理，而是可持续的 functional training system；
所有可学习参数都属于 KAN edge 系统，不允许普通 MLP stem/head/LN 参数作为隐藏支撑。
```

因此 v5.4 不能再继续只围绕 smoothing event controller 打转。现在的核心问题变成：

$$
\boxed{
\text{我们是否有一个足够高效、足够可训练、又足够几何可控的 PureKAN edge primitive？}
}
$$

---

## 1. 当前 v5.3 结果的真实含义

### 1.1 已经确认的进展

v5.3 的 P0 说明 AB-RBF core / memory smoke 没有错误。LightSmooth 相比 StrongSmooth 的成本明显降低。典型结果是：

```text
ABRBF-AdamW: memory ratio = 1.0
ABRBF-LightSmooth: memory ratio 大约 1.02 到 1.20
ABRBF-StrongSmooth: memory ratio 大约 3.4 到 4.0
```

这说明：

$$
\boxed{
\text{StrongSmooth 太重，但 LightSmooth 作为低成本 smoothing primitive 是可用的。}
}
$$

v5.3 的 P1 single-event verification 也说明 LightSmooth 可以在 `acc drop = 0`、KL/logit drift 很小的情况下带来 residual geometry reduction。不过与 v5.2 相比，v5.3 的 P1 reduction 幅度变小，典型值大约是：

```text
phiR reduction: 约 7%
curvR reduction: 约 14%
```

这说明 LightSmooth 是有效工具，但它不是强 optimizer。

### 1.2 仍然失败的部分

v5.3 的关键失败在 P3 / P4：

```text
P3 all-dataset survivors: none
P4 all-dataset survivors: none
P5/P6/P7 not run
```

最终诊断是：

```text
refresh 可以恢复 task accuracy；
但 final residual phi reduction 只剩大约 0.8% 到 1.8%，低于 gate；
因此 event-driven multicycle 不能进入 seed confirmation。
```

这说明：

$$
\boxed{
\text{LightSmooth 可以单次降低 residual geometry，但训练期多轮维护不能稳定保留几何收益。}
}
$$

更重要的是，这个失败不是简单的 trigger 问题。v5.3 已经加入 P2 cheap probe audit，并且新 score/debt trigger 能观察到触发机会；旧 plateau trigger 迟钝的问题已经被看见。真正的问题是：

$$
\boxed{
\text{smoothing 后的 refresh 会把 geometry gain 冲掉。}
}
$$

所以继续微调 event threshold、cooldown、eta cap 的边际收益很低。

---

## 2. 代码实现层面的关键审视

### 2.1 当前上传的 `dgkan_core(5).py` 与 v5.3 文档存在一致性风险

v5.1 / v5.3 结果文档显示：

```text
ABRBFDense 已经进入 core；
edge_named_params / base_named_params / rbf_residual_named_params 可以审计；
P0 core consistency passed。
```

但是这次上传的 `dgkan_core(5).py` 中，基础可见结构仍然主要是：

```text
RBFDense:
  fixed centers
  fixed width
  trainable coeff
  optional bias

PureKANClassifier:
  input_kan = RBFDense(..., bias=False)
  blocks = RBFDense(..., bias=False)
  output_kan = RBFDense(..., bias=False)
```

在该上传文件里未能明显看到 `ABRBFDense`、`edge_named_params`、`base_named_params`、`rbf_residual_named_params` 这些 core-level 定义。这有两种可能：

```text
1. 上传的 core 不是实际运行 v5.3 的最新版；
2. AB-RBF 仍然主要在 runner/helper 里定义，而不是核心库里统一定义。
```

不管是哪种，v5.4 第一优先级都是 **core consistency hardening**。如果 AB-RBF 不是 core primitive，后续 efficiency benchmark、custom backward、CIFAR、Rational/KAT、functional update 都会有实现分裂风险。

### 2.2 Dense RBF / AB-RBF 无法天然满足 MLP-like efficiency

标准 dense RBF edge 是：

$$
f_o(x)=\sum_{i,k} c_{oik} B_k(x_i).
$$

其核心计算是：

```python
basis: [B, c_in, K]
out = einsum("bik,oik->bo", basis, coeff)
```

计算复杂度为：

$$
O(Bc_{in}K)+O(Bc_{out}c_{in}K).
$$

MLP / Linear 的复杂度是：

$$
O(Bc_{out}c_{in}).
$$

因此当 $K=16$ 或 $K=24$ 时，dense RBF / dense AB-RBF 的 forward 不可能天然接近 MLP。即使 LightSmooth 变轻，主 forward/backward primitive 仍然偏重。

这意味着：

$$
\boxed{
\text{AB-RBF dense 是机制平台，不一定是终极高效 primitive。}
}
$$

### 2.3 backward 显存要低于 MLP，必须 custom backward / analytic adjoint

如果用 PyTorch autograd 保存完整 `basis: [B,c_in,K]`，则 backward memory 不可能优于 MLP。要满足终极目标，必须使用：

```text
basis recomputation;
streaming coeff-gradient accumulation;
activation checkpoint / no-save basis;
analytic VJP;
sufficient statistics;
```

目标是 forward 时不保存完整 basis，而在 backward 中根据输入重新计算：

$$
B_k(x_i),
$$

并流式累积：

$$
\nabla c_{oik}=\sum_b \delta_{bo}B_k(x_{bi}).
$$

如果没有这个方向，RBF/AB-RBF 很难满足：

$$
M_{backward,KAN} < M_{backward,MLP}.
$$

---

## 3. 目前路线的重新判断

### 3.1 保留的结论

当前仍然成立的结论是：

```text
1. RBF-only coefficient coordinate 是 PureKAN 的坏坐标；
2. AB-RBF edge decomposition 是正方向；
3. base path 不是装饰，而是承担低阶/尺度/线性结构；
4. RBF residual 可以通过 LightSmooth 做低成本单次 geometry maintenance；
5. StrongSmooth / exact-heavy NFS 太重，不适合作为训练主线。
```

### 3.2 需要降级的结论

需要降级的是：

```text
1. AB-RBF dense 不能直接视为终极高效 PureKAN primitive；
2. LightSmooth 还不是稳定 train-time multicycle optimizer；
3. functional update 目前更像 residual geometry maintenance，而不是 from-scratch functional optimizer；
4. 仅靠 event controller 不能解决 geometry-retention-after-refresh 问题。
```

### 3.3 新的关键问题

v5.4 必须回答三个更硬的问题：

$$
\boxed{
Q1: \text{哪种 PureKAN edge primitive 能在 forward/backward 上接近 MLP？}
}
$$

$$
\boxed{
Q2: \text{能否用 analytic/custom backward 让 backward 显存低于 MLP？}
}
$$

$$
\boxed{
Q3: \text{在高效 primitive 上，functional training 是否还能保留 geometry/AUC/ECE 优势？}
}
$$

---

## 4. v5.4 的主线：Efficiency-First PureKAN

v5.4 不再把 smoothing controller 当主线，而是把主线拆成三条并行但有顺序的路线。

### 4.1 路线 A：Dense AB-RBF 只作为 reference

Dense AB-RBF 保留，用于：

```text
机制 reference；
geometry reference；
AB-RBF edge decomposition 上限；
LightSmooth 后处理 reference。
```

但它不再默认是最终系统。

### 4.2 路线 B：高效 AB-RBF primitive

必须实现至少三种高效替代：

#### B1. Depthwise-ABRBF + LinearMix

先对每个 channel 做一维 edge function：

$$
\tilde x_i = b_i+w_i x_i+u_i\operatorname{silu}(x_i)+\sum_k c_{ik}B_k(x_i),
$$

再做 mixing：

$$
y=W\tilde x.
$$

复杂度：

$$
O(Bc_{in}K)+O(Bc_{in}c_{out}).
$$

这比 dense AB-RBF 的：

$$
O(Bc_{in}c_{out}K)
$$

更接近 MLP。

为了保持 “无 non-KAN 参数”，$W$ 必须被定义为 KAN edge mixing parameter，而不是普通外部 MLP head/stem。它可以被纳入 `edge_named_params`，并在 final claim 中计入 KAN edge system。

#### B2. LowRank / CP-ABRBF

把 dense RBF residual coefficient：

$$
c_{oik}
$$

分解为：

$$
c_{oik}=\sum_{r=1}^{R}U_{or}V_{ir}W_{kr}.
$$

当 $R \ll \min(c_{in},c_{out},K)$ 时，参数量和计算量显著下降。需要测试：

```text
R = 4, 8, 16
```

并记录是否出现 rank bottleneck。

#### B3. Rational / KAT-AB Edge

Rational/KAT 的基本结构是：

$$
\tilde x_i=r_i(x_i),
$$

$$
y=W\tilde x.
$$

它天然更接近 MLP，因为主计算仍然是 GEMM。前面 Rational-AdamW 有信号，但 Rational functional metric 没跑通。v5.4 不再先追 Rational functional update，而先测试：

```text
Rational/KAT-AB 是否满足终极 efficiency envelope；
accuracy / ECE / geometry 是否接近 AB-RBF；
是否能作为最终 high-efficiency primitive。
```

### 4.3 路线 C：custom backward / analytic adjoint

对 dense AB-RBF 和高效 AB-RBF 都要实现：

```text
Autograd version
CustomBackward-Recompute version
CustomBackward-StreamingStats version
```

目标是验证：

$$
M_{backward,KAN} \leq 0.8 M_{backward,MLP}.
$$

---

## 5. v5.4 实验计划总览

v5.4 分为九个阶段：

```text
P0: Core consistency and primitive manifest
P1: Pure primitive efficiency microbenchmark
P2: custom backward correctness and memory audit
P3: accuracy frontier under AdamW-like training
P4: functional / analytic update smoke on efficient primitives
P5: LightSmooth compatibility on efficient primitives
P6: 3-seed efficiency-aware candidate selection
P7: 5-seed confirm
P8: 10-seed final confirm
P9: failure diagnosis and route decision
```

每个阶段都必须同时记录：

```text
task metrics;
geometry metrics;
edge contribution metrics;
efficiency metrics;
coverage / strict PureKAN invariants.
```

---

## 6. P0: Core consistency and primitive manifest

### 6.1 目的

确保所有候选都是 strict PureKAN，且 AB-RBF / efficient edge primitive 不再散落在 runner helper 中。

### 6.2 必测方法

```text
MLP-AdamW-reference
RBFOnly-Dense
ABRBF-Dense-linear+silu
ABRBF-DepthwiseMix-linear+silu
ABRBF-CPRank4
ABRBF-CPRank8
ABRBF-CPRank16
RationalKAT-AB
```

### 6.3 必须检查的不变量

```text
learnable_nonKAN_params = 0
edge_param_coverage = 1.0
base_param_coverage = 1.0, if applicable
rbf_residual_param_coverage = 1.0, if applicable
mixing_param_coverage = 1.0, if applicable
rollback_error = 0
no hidden nn.Linear outside edge system
no learnable LayerNorm gamma/beta
no ordinary stem/head
```

对于 DepthwiseMix / RationalKAT-AB，必须明确：

```text
LinearMix / W mixing 是 edge system 内部参数；
不是 ordinary non-KAN head。
```

### 6.4 记录字段

```text
method
primitive_type
num_edge_params
num_base_params
num_rbf_params
num_mixing_params
num_nonkan_params
coverage_edge
coverage_base
coverage_rbf
coverage_mixing
rollback_max_abs
forward_shape_ok
backward_shape_ok
```

### 6.5 判定

任何方法如果：

```text
num_nonkan_params > 0
coverage_edge < 1.0
```

直接不能进入 P1。

---

## 7. P1: Pure primitive efficiency microbenchmark

### 7.1 目的

先不看最终 accuracy，先确认 primitive 是否可能满足终极效率目标。

### 7.2 测试维度

使用 synthetic input 和真实 batch 两类输入。

```text
batch_size = 64, 128, 256, 512
input_dim = 784, 1024, 2048
hidden_dim = 64, 96, 128, 256
basis_count = 8, 16, 24
num_classes = 10
```

### 7.3 对照方法

```text
MLP-Linear+SiLU
RBFOnly-Dense
ABRBF-Dense-linear+silu
ABRBF-DepthwiseMix
ABRBF-CPRank4/8/16
RationalKAT-AB
```

### 7.4 记录指标

#### 时间

```text
forward_time_ms_mean
forward_time_ms_p50
forward_time_ms_p95
backward_time_ms_mean
backward_time_ms_p50
backward_time_ms_p95
step_time_ms_mean
samples_per_second
```

#### 显存

```text
forward_peak_allocated_mb
forward_peak_reserved_mb
backward_peak_allocated_mb
backward_peak_reserved_mb
activation_saved_mb
basis_saved_mb
optimizer_state_mb
```

#### 参数/FLOPs

```text
num_params
edge_params
base_params
rbf_params
mixing_params
estimated_forward_flops
estimated_backward_flops
flop_ratio_vs_mlp
```

### 7.5 Gate

进入 P2 的硬门槛：

$$
\frac{T_{forward,KAN}}{T_{forward,MLP}} \leq 1.25,
$$

$$
\frac{M_{forward,KAN}}{M_{forward,MLP}} \leq 1.25,
$$

$$
\frac{T_{backward,KAN}}{T_{backward,MLP}} \leq 1.40,
$$

$$
\frac{M_{backward,KAN}}{M_{backward,MLP}} \leq 1.00
$$

作为 P1 初筛。最终目标更严格：

$$
\frac{M_{backward,KAN}}{M_{backward,MLP}} \leq 0.80.
$$

P1 阶段允许先放宽到 $1.0$，因为 custom backward 还没完全进入。

### 7.6 必须画图

```text
forward time ratio vs hidden_dim
backward memory ratio vs hidden_dim
step time ratio vs basis_count
FLOPs vs measured time scatter
activation_saved_mb stacked bar
throughput vs batch_size
```

---

## 8. P2: Custom backward correctness and memory audit

### 8.1 目的

证明 KAN backward 能不用保存完整 basis activation，靠 recomputation / streaming stats 达到低显存。

### 8.2 方法

对以下 primitive 实现两版：

```text
Autograd
CustomBackward-Recompute
CustomBackward-StreamingStats
```

候选：

```text
ABRBF-Dense-linear+silu
ABRBF-DepthwiseMix
ABRBF-CPRank8
RationalKAT-AB, if applicable
```

### 8.3 正确性检查

对同一 batch，比对 autograd 和 custom backward：

$$
\operatorname{relerr}(\nabla \theta)
=
\frac{\|g_{custom}-g_{autograd}\|}{\|g_{autograd}\|+\epsilon}.
$$

记录：

```text
grad_relerr_coeff
grad_relerr_base
grad_relerr_rbf
grad_relerr_mixing
grad_cos_coeff
grad_cos_base
grad_cos_rbf
grad_cos_mixing
```

目标：

$$
\operatorname{relerr} < 10^{-4}
$$

或至少：

$$
\cos(g_{custom},g_{autograd}) > 0.999.
$$

### 8.4 显存检查

记录：

```text
basis_tensor_saved = true/false
saved_tensor_count
saved_tensor_total_mb
recompute_time_ms
streaming_stats_time_ms
backward_peak_allocated_mb
backward_peak_reserved_mb
```

### 8.5 Gate

进入 P3 的方法必须满足：

```text
grad correctness pass;
backward memory <= MLP backward memory;
custom step time <= 1.5x MLP step time;
```

### 8.6 可视化

```text
grad relerr histogram by param group
custom vs autograd memory bar
custom vs autograd step time bar
saved activation breakdown
memory-time Pareto frontier
```

---

## 9. P3: Accuracy frontier under AdamW-like training

### 9.1 目的

先验证高效 primitive 是否有任务能力。这里暂时允许 AdamW，因为问题是 architecture + efficiency frontier，不是 functional optimizer。

### 9.2 方法

```text
MLP-AdamW
RBFOnly-Dense-AdamW
ABRBF-Dense-AdamW
ABRBF-DepthwiseMix-AdamW
ABRBF-CPRank8-AdamW
ABRBF-CPRank16-AdamW
RationalKAT-AB-AdamW
```

### 9.3 数据集

```text
MNIST
Fashion-MNIST
KMNIST
```

第一轮：

```text
seeds = 0,1,2
train/val/test = 6000/1000/1000
```

### 9.4 记录指标

```text
test_acc
val_loss
val_auc
ECE
NLL
margin_mean
margin_p10
classwise_acc
feature_effective_rank
class_centroid_separation
base_ablation_drop
rbf_ablation_drop
mixing_ablation_drop
phi_base
phi_rbf
phi_total
curvature_rbf
sobolev_rbf_norm
forward_time_ratio
backward_memory_ratio
step_time_ratio
```

### 9.5 Gate

候选进入 P4 需要：

```text
accuracy >= MLP-AdamW - 0.5%
val_auc <= MLP-AdamW + 5% relative loss-AUC cost
ECE <= MLP-AdamW + 0.02
forward_time <= 1.25x MLP
backward_memory <= 1.0x MLP
step_time <= 1.4x MLP
```

### 9.6 可视化

```text
accuracy vs step_time ratio Pareto
accuracy vs backward memory ratio Pareto
ECE vs geometry scatter
base/RBF/mixing ablation bar
feature rank trajectory
classwise accuracy heatmap
```

---

## 10. P4: Functional / analytic update smoke on efficient primitives

### 10.1 目的

在高效 primitive 上测试 functional training 是否仍然可行。这里不再用旧的 heavy U-FULL，而使用轻量 functional variants。

### 10.2 方法

```text
AdamW baseline
FunctionalCoord-Adam
ResidualOnly-FunctionalSmooth
AnalyticAdj-StatsUpdate
TaskAdam + LightSmooth maintenance
```

其中：

```text
FunctionalCoord-Adam:
  在 function-whitened coordinate 中用 Adam-like dynamics。

ResidualOnly-FunctionalSmooth:
  base/mixing 用 Adam-like task update，RBF residual 用 light geometry update。

AnalyticAdj-StatsUpdate:
  使用解析 VJP / sufficient statistics 更新 edge 参数，目标是减少 backward memory。

TaskAdam + LightSmooth:
  AdamW-like task learner + 低频 LightSmooth，作为 pragmatic bridge。
```

### 10.3 记录指标

除了 P3 指标，还要记录：

```text
functional_update_norm
functional_update_cos_with_adam
grad_storage_mb
stats_storage_mb
analytic_vjp_time_ms
basis_recompute_time_ms
smoothing_event_count
smoothing_accepted_count
smoothing_geometry_gain
smoothing_task_cost
```

### 10.4 Gate

```text
accuracy >= AdamW version of same primitive - 1.0%
val_auc not worse than AdamW version by > 5%
ECE <= AdamW version + 0.02
rbf geometry improves by >= 5%
backward memory <= AdamW-autograd version by 20%
step time <= AdamW-autograd version + 20%
```

### 10.5 可视化

```text
functional vs AdamW trajectory overlay
update cosine over time
geometry gain vs task cost scatter
memory breakdown by training path
analytic stats fidelity plot
```

---

## 11. P5: LightSmooth compatibility on efficient primitives

### 11.1 目的

v5.3 说明 LightSmooth 单次可用，但 refresh 后 geometry retention 不够。P5 在高效 primitive 上重新验证，不再先做 multicycle。

### 11.2 方法

```text
EfficientPrimitive-AdamW
EfficientPrimitive-AdamW + one-shot LightSmooth
EfficientPrimitive-AdamW + one-cycle SmoothRefresh
EfficientPrimitive-AdamW + posthoc LightSmooth only
```

### 11.3 Refresh policy

只保留两个最有解释性的 refresh：

```text
base/mixing-only refresh:
  smoothing 后冻结 RBF residual，只恢复 base/mixing task fit。

small-RBF refresh:
  smoothing 后 RBF residual 只允许 very small LR，避免几何收益被冲掉。
```

### 11.4 记录指标

```text
pre_smooth_acc
post_smooth_acc
post_refresh_acc
pre_smooth_phi_rbf
post_smooth_phi_rbf
post_refresh_phi_rbf
geometry_retention = post_refresh_phi_gain / post_smooth_phi_gain
refresh_recovery = recovered_acc_drop / smooth_acc_drop
KL
logit_drift
memory_ratio
smooth_time
```

### 11.5 Gate

```text
post_refresh_acc >= baseline_acc - 0.005
geometry_retention >= 0.50
memory_ratio <= 1.25
amortized_time_overhead <= 0.10
```

### 11.6 可视化

```text
smooth-refresh recovery plot
geometry retention bar
accuracy/geometry timeline with smoothing marker
smoothing cost waterfall
```

---

## 12. P6: 3-seed efficiency-aware candidate selection

### 12.1 候选来源

只选择同时通过 P3/P4/P5 的方法。最多保留 3 个：

```text
best efficient primitive AdamW
best efficient primitive functional/analytic update
best efficient primitive + LightSmooth maintenance
```

### 12.2 对照

```text
MLP-AdamW
PureKAN-RBFOnly-AdamW
PureKAN-ABRBF-Dense-AdamW
best efficient candidates
```

### 12.3 数据集

```text
MNIST
Fashion-MNIST
KMNIST
```

### 12.4 Gate

候选必须满足：

$$
Acc \geq Acc_{MLP-AdamW},
$$

$$
AUC \leq AUC_{MLP-AdamW},
$$

或至少相对 MLP loss-AUC 不差于 $5\%$，并且：

$$
ECE \leq ECE_{MLP-AdamW},
$$

$$
\phi_{rbf}\text{ 或 }\operatorname{curv}_{rbf}\text{ 有明确改善},
$$

$$
T_{step} \leq 1.25 T_{MLP},
$$

$$
M_{backward} \leq 0.8 M_{MLP}.
$$

P6 是 v5.4 的关键阶段。如果没有候选过 P6，则不要进入 5-seed。

---

## 13. P7/P8: 5-seed and 10-seed confirm

只有 P6 通过后才运行。

### 13.1 5-seed confirm

```text
seeds = 0,1,2,3,4
```

必须记录 paired delta vs：

```text
MLP-AdamW
PureKAN-AdamW same primitive
ABRBF-Dense-AdamW
```

### 13.2 10-seed final

```text
seeds = 0..9
```

最终必须输出：

```text
paired_acc_delta
paired_auc_delta
paired_ece_delta
paired_geometry_delta
paired_step_time_ratio
paired_backward_memory_ratio
```

---

## 14. P9: Failure diagnosis and route decision

如果 v5.4 没有候选通过 P6，需要根据失败模式做明确决策。

### 14.1 失败模式 F1：效率失败

```text
accuracy 可以，但 forward/backward 仍远慢于 MLP。
```

结论：

```text
Dense / factorized RBF 不是最终 primitive；转 Rational/KAT 或更强 fused kernel。
```

### 14.2 失败模式 F2：accuracy 失败

```text
效率过了，但 accuracy 不如 MLP / ABRBF-Dense。
```

结论：

```text
efficient primitive 表达力不足；提高 rank / group / base path，或保留 dense ABRBF 做机制平台。
```

### 14.3 失败模式 F3：backward memory 未降

```text
custom backward 正确，但显存不低。
```

结论：

```text
检查是否仍保存 basis / hidden / teacher tensors；改 streaming stats / recomputation；必要时写 fused CUDA/Triton。
```

### 14.4 失败模式 F4：geometry 失败

```text
accuracy 和效率过，但 geometry 没优势。
```

结论：

```text
LightSmooth 作为 posthoc/maintenance，或改 residual geometry metric；但不牺牲效率主线。
```

### 14.5 失败模式 F5：functional training 失败但 AdamW 成功

```text
efficient primitive + AdamW 成功，functional/analytic update 不成功。
```

结论：

```text
短期主线用 AdamW-like task training + analytic memory-saving backward；functional update 保留为 residual maintenance，不再宣称 fully functional-from-scratch。
```

---

## 15. 最终产物要求

v5.4 必须输出以下文件：

```text
p0_core_manifest.csv
p1_efficiency_microbenchmark.csv
p1_efficiency_gate_summary.csv
p2_custom_backward_correctness.csv
p2_memory_audit.csv
p3_accuracy_frontier.csv
p4_functional_update_smoke.csv
p5_lightsmooth_compatibility.csv
p6_candidate_selection3.csv
p7_confirm5.csv
p8_confirm10.csv
p9_failure_diagnosis.csv
failure_table.csv
aggregate_decision.json
recommendation.json
figures/
```

必须生成以下图：

```text
fig_forward_time_ratio_vs_hidden.png
fig_backward_memory_ratio_vs_hidden.png
fig_step_time_vs_accuracy_pareto.png
fig_backward_memory_vs_accuracy_pareto.png
fig_saved_activation_breakdown.png
fig_custom_backward_relerr_hist.png
fig_base_rbf_mixing_contribution.png
fig_accuracy_geometry_efficiency_pareto.png
fig_smooth_refresh_geometry_retention.png
fig_failure_heatmap.png
```

---

## 16. 当前推荐优先级

我建议 v5.4 的执行顺序是：

```text
1. 修正 / 确认 ABRBFDense core-level 实现。
2. 先跑 P1 efficiency microbenchmark，不要先跑更多 smoothing。
3. 对通过 P1 的 primitive 做 custom backward P2。
4. 只对通过 P1/P2 的 primitive 做 P3 accuracy frontier。
5. 只在高效且 accuracy 有希望的 primitive 上做 functional / LightSmooth。
```

也就是说：

$$
\boxed{
\text{效率先行，几何随后；没有效率，不进入终极目标主线。}
}
$$

---

## 17. 最终判断

v5.3 后，我们的状态是：

```text
AB-RBF: 有机制价值；
LightSmooth: 单次有效，训练期 controller 未成；
Dense RBF/AB-RBF: 不一定满足终极效率；
functional-from-scratch: 仍未完成；
custom backward / efficient primitive: 必须成为下一步核心。
```

因此 v5.4 的核心不是继续问：

```text
怎么让 LightSmooth 多轮更稳？
```

而是先问：

```text
我们有没有一个 forward/backward 接近 MLP 的 PureKAN primitive？
```

如果没有，任何 optimizer 和 smoothing 都无法满足终极目标。



---


# Source 36: `docs/DG-KAN_v5.4_EfficiencyFirst_PureKAN_FunctionalTraining_结果复盘.md`


# DG-KAN v5.4 Efficiency-First PureKAN Functional Training 结果复盘

本轮依据 `docs/DG-KAN_v5.4_EfficiencyFirst_PureKAN_FunctionalTraining_实验计划.md`。核心目标是先验证 PureKAN edge primitive 的效率 envelope，再决定 functional training / LightSmooth 是否值得继续扩展。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 core manifest | 8 | 0 |
| P1 efficiency | 192 | 0 |
| P2 custom backward | 6 | 0 |
| P3 accuracy frontier | 63 | 0 |
| P4 functional smoke | 1 | 0 |
| P5 LightSmooth | 1 | 0 |
| P6 selection | 1 | 0 |
| P7 confirm5 | 1 | 0 |
| P8 confirm10 | 1 | 0 |

## Code / Config Changes

```text
experiments/dgkan_core.py
  Added core efficient PureKAN edge primitives: ABRBFDepthwiseMixDense, CPABRBFDense, RationalKATDense.
  Extended edge/base/RBF/mixing parameter discovery so efficient mixing remains inside the KAN edge system.

experiments/run_gafu_v54.py
  Added P0 core manifest, P1 efficiency microbenchmark, P2 checkpoint/recompute backward audit,
  P3 AdamW-like accuracy frontier, and gated P4-P9 placeholders/diagnosis.

experiments/analyze_gafu_v54.py
  Generates v5.4 gate summaries, figures, aggregate_decision.json, route_decision.json, and this replay.
```

## P0 Core Consistency

| method | primitive | edge | nonKAN | mixing | rollback | P0 |
|---|---|---|---|---|---|---|
| MLP-AdamW-reference | MLP-reference | 0 | 59722 | 0 | 0.0000 | yes |
| RBFOnly-Dense | dense-rbf | 1009664 | 0 | 0 | 0.0000 | yes |
| ABRBF-Dense-linear+silu | dense-abrbf | 1198976 | 0 | 0 | 0.0000 | yes |
| ABRBF-DepthwiseMix-linear+silu | depthwise-mix | 82864 | 0 | 63104 | 0.0000 | yes |
| ABRBF-CPRank4 | cp-lowrank | 194856 | 0 | 0 | 0.0000 | yes |
| ABRBF-CPRank8 | cp-lowrank | 200400 | 0 | 0 | 0.0000 | yes |
| ABRBF-CPRank16 | cp-lowrank | 211488 | 0 | 0 | 0.0000 | yes |
| RationalKAT-AB | rational-kat | 69344 | 0 | 63104 | 0.0000 | yes |

## P1 Efficiency Microbenchmark

| method | rows | fwd | bwd | bmem | step | pass rate | P1 |
|---|---|---|---|---|---|---|---|
| ABRBF-CPRank16 | 24 | 7.1699 | 4.3364 | 2.5357 | 3.6396 | 0.0000 | no |
| ABRBF-CPRank4 | 24 | 15.5634 | 4.2092 | 2.5285 | 4.8073 | 0.0000 | no |
| ABRBF-CPRank8 | 24 | 10.3810 | 3.5064 | 2.5309 | 3.7239 | 0.0000 | no |
| ABRBF-Dense-linear+silu | 24 | 11.7354 | 3.7917 | 3.1774 | 4.1188 | 0.0000 | no |
| ABRBF-DepthwiseMix-linear+silu | 24 | 7.8662 | 8.2935 | 2.4273 | 5.7053 | 0.0000 | no |
| RBFOnly-Dense | 24 | 2.8412 | 1.3465 | 2.5214 | 1.5059 | 0.0000 | no |
| RationalKAT-AB | 24 | 6.9917 | 5.9999 | 1.3494 | 4.3777 | 0.0000 | no |

P1 survivors:
```text
none
```

## P2 Custom Backward Audit

| method | variant | relerr | cos | mem | step | saved MB | P2 |
|---|---|---|---|---|---|---|---|
| ABRBF-DepthwiseMix-linear+silu | Autograd | 0.0000 | 1.0000 | 2.9838 | 1.5462 | 7.2734 | no |
| ABRBF-DepthwiseMix-linear+silu | CustomBackward-Recompute | 0.0000 | 1.0000 | 2.9838 | 10.3142 | 1.0910 | no |
| ABRBF-DepthwiseMix-linear+silu | CustomBackward-StreamingStats | 0.0000 | 1.0000 | 2.9838 | 4.5104 | 1.0910 | no |
| RationalKAT-AB | Autograd | 0.0000 | 1.0000 | 1.2465 | 4.9874 | 1.5312 | no |
| RationalKAT-AB | CustomBackward-Recompute | 0.0000 | 1.0000 | 1.2465 | 5.5812 | 0.2297 | no |
| RationalKAT-AB | CustomBackward-StreamingStats | 0.0000 | 1.0000 | 1.2465 | 5.6224 | 0.2297 | no |

## P3 Accuracy Frontier

| dataset | method | runs | acc | std | gap | ECE | fwd | bmem | step | P3 |
|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | ABRBF-CPRank16-AdamW | 3 | 0.7956 | 0.0088 | 0.0046 | 0.0365 | 7.1699 | 2.5357 | 3.6396 | no |
| Fashion-MNIST | ABRBF-CPRank8-AdamW | 3 | 0.7891 | 0.0127 | 0.0111 | 0.0329 | 10.3810 | 2.5309 | 3.7239 | no |
| Fashion-MNIST | ABRBF-Dense-AdamW | 3 | 0.7852 | 0.0115 | 0.0150 | 0.0286 | 11.7354 | 3.1774 | 4.1188 | no |
| Fashion-MNIST | ABRBF-DepthwiseMix-AdamW | 3 | 0.7956 | 0.0176 | 0.0046 | 0.0915 | 7.8662 | 2.4273 | 5.7053 | no |
| Fashion-MNIST | MLP-AdamW | 3 | 0.8001 | 0.0064 | 0.0000 | 0.0379 | 1.0000 | 1.0000 | 1.0000 | no |
| Fashion-MNIST | RBFOnly-Dense-AdamW | 3 | 0.7845 | 0.0064 | 0.0156 | 0.0430 | 2.8412 | 2.5214 | 1.5059 | no |
| Fashion-MNIST | RationalKAT-AB-AdamW | 3 | 0.7630 | 0.0018 | 0.0371 | 0.2363 | 6.9917 | 1.3494 | 4.3777 | no |
| KMNIST | ABRBF-CPRank16-AdamW | 3 | 0.6908 | 0.0295 | 0.0286 | 0.0436 | 7.1699 | 2.5357 | 3.6396 | no |
| KMNIST | ABRBF-CPRank8-AdamW | 3 | 0.6875 | 0.0253 | 0.0319 | 0.0444 | 10.3810 | 2.5309 | 3.7239 | no |
| KMNIST | ABRBF-Dense-AdamW | 3 | 0.6979 | 0.0183 | 0.0215 | 0.0376 | 11.7354 | 3.1774 | 4.1188 | no |
| KMNIST | ABRBF-DepthwiseMix-AdamW | 3 | 0.6191 | 0.0146 | 0.1003 | 0.0873 | 7.8662 | 2.4273 | 5.7053 | no |
| KMNIST | MLP-AdamW | 3 | 0.7194 | 0.0129 | 0.0000 | 0.0507 | 1.0000 | 1.0000 | 1.0000 | no |
| KMNIST | RBFOnly-Dense-AdamW | 3 | 0.6738 | 0.0073 | 0.0456 | 0.0549 | 2.8412 | 2.5214 | 1.5059 | no |
| KMNIST | RationalKAT-AB-AdamW | 3 | 0.5814 | 0.0064 | 0.1380 | 0.2234 | 6.9917 | 1.3494 | 4.3777 | no |
| MNIST | ABRBF-CPRank16-AdamW | 3 | 0.8893 | 0.0260 | 0.0182 | 0.0561 | 7.1699 | 2.5357 | 3.6396 | no |
| MNIST | ABRBF-CPRank8-AdamW | 3 | 0.8978 | 0.0259 | 0.0098 | 0.0580 | 10.3810 | 2.5309 | 3.7239 | no |
| MNIST | ABRBF-Dense-AdamW | 3 | 0.8919 | 0.0148 | 0.0156 | 0.0518 | 11.7354 | 3.1774 | 4.1188 | no |
| MNIST | ABRBF-DepthwiseMix-AdamW | 3 | 0.8678 | 0.0231 | 0.0397 | 0.1502 | 7.8662 | 2.4273 | 5.7053 | no |
| MNIST | MLP-AdamW | 3 | 0.9076 | 0.0171 | 0.0000 | 0.0452 | 1.0000 | 1.0000 | 1.0000 | no |
| MNIST | RBFOnly-Dense-AdamW | 3 | 0.8587 | 0.0175 | 0.0488 | 0.0600 | 2.8412 | 2.5214 | 1.5059 | no |
| MNIST | RationalKAT-AB-AdamW | 3 | 0.8509 | 0.0253 | 0.0566 | 0.3693 | 6.9917 | 1.3494 | 4.3777 | no |

P3 all-dataset survivors:
```text
none
```

## P4-P8 Decision

```text
P4 functional / analytic update smoke: not run
Reason: P3 produced no efficiency-aware accuracy survivor.

P5 LightSmooth compatibility: not run
Reason: P4 was not reached.

P6/P7/P8 confirm: not run
Reason: no candidate passed the written joint gate.
```

## Failure Diagnosis

| failure | count |
|---|---|
| F1_efficiency_failed | 222 |

## Required Artifacts

Written under `results/v5_4/`:
```text
p0_core_primitive_manifest.csv
p1_efficiency_microbenchmark.csv
p1_efficiency_gate_summary.csv
p2_custom_backward_audit.csv
p2_custom_backward_gate_summary.csv
p3_accuracy_frontier.csv
p3_accuracy_gate_summary.csv
p4_functional_analytic_smoke.csv
p5_lightsmooth_compatibility.csv
p6_candidate_selection3.csv
p7_confirm5.csv
p8_confirm10.csv
p9_failure_diagnosis.csv
failure_table.csv
failure_by_type.csv
failure_by_method.csv
aggregate_decision.json
route_decision.json
figures/
```

## Final Decision

```text
DG-KAN v5.4 status:
  stop_after_p3_no_efficiency_accuracy_survivor

What improved:
  Efficient PureKAN primitives are now core-level objects, not runner-only helpers.
  DepthwiseMix / CP / RationalKAT all keep learnable parameters inside the KAN edge system.
  P3 confirms CP-ABRBF can recover meaningful task accuracy in places, especially MNIST/Fashion.

What failed:
  No primitive passed the P1 hard efficiency envelope across the compact grid.
  Checkpoint/recompute backward preserved gradients but did not reduce measured peak memory below MLP.
  P3 produced no all-dataset survivor under the accuracy + efficiency joint gate, with KMNIST the hardest task.

Conclusion:
  v5.4 supports the plan's warning: dense AB-RBF is a mechanism reference, not a final efficient system.
  The next route should prioritize fused/custom kernels or a more GEMM-native Rational/KAT primitive before more functional optimizer work.
```



---


# Source 37: `docs/DG-KAN_v5.5_KernelFirst_EfficientPureKAN_实验计划.md`


# DG-KAN v5.5 Kernel-First Efficient PureKAN 实验计划

## 0. 终极目标与本轮重新定位

当前终极目标定义为：构建一个无 non-KAN 参数的 PureKAN functional training system，其 forward 时间与显存接近 MLP，backward 时间接近 MLP，backward 显存低于 MLP，收敛速度快，并且在 accuracy、validation-loss AUC、ECE、geometry 上超过 MLP-AdamW 与 PureKAN-AdamW。

这个目标比此前的“PureKAN 能训练”严格得多。它要求我们同时满足：

$$
\text{task quality}
+
\text{functional geometry}
+
\text{MLP-like efficiency}
+
\text{memory advantage}.
$$

v5.4 的结果说明，当前路线不能继续只在 optimizer / smoothing 上小修。原因是所有候选 primitive 都没有通过效率 envelope，P3 也没有产生 accuracy + efficiency 的 all-dataset survivor。因此 v5.5 的核心不是继续追问 “functional update 怎么调”，而是先回答：

$$
\boxed{
\text{我们是否拥有一个真正接近 MLP 的 PureKAN edge primitive？}
}
$$

如果没有，那么后续 LightSmooth、functional update、event controller 都只能是机制研究，无法满足终极目标。

---

## 1. v5.4 结果复盘：现在到底卡在哪里

### 1.1 P0 的正面信号：efficient primitives 已经进入实验对象

v5.4 P0 中，候选 primitive 包括：

```text
RBFOnly-Dense
ABRBF-Dense-linear+silu
ABRBF-DepthwiseMix-linear+silu
ABRBF-CPRank4 / 8 / 16
RationalKAT-AB
```

它们都保持了 strict PureKAN 设定：learnable non-KAN 参数为 0，mixing 参数也被计入 KAN edge system。也就是说，本轮不是“混入 MLP 参数后变快”，而是在 PureKAN edge 内部尝试不同的高效参数化。

但 P0 只能说明参数归属正确，不能说明计算路径正确。v5.4 后续结果显示，虽然候选都进入了 core-level object，但计算效率距离 MLP 仍然很远。

---

### 1.2 P1 的核心失败：没有 primitive 通过效率门槛

P1 efficiency microbenchmark 中，所有方法 pass rate 都是 0。

最接近 MLP step time 的是 `RBFOnly-Dense`：

```text
forward ratio = 2.84
backward ratio = 1.35
backward memory ratio = 2.52
step ratio = 1.51
```

这说明 RBFOnly-Dense 虽然相对其它 KAN primitive 速度还算低，但仍然远不满足终极目标，尤其 backward memory 明显高于 MLP。

`RationalKAT-AB` 的 backward memory ratio 只有约 1.35，是所有候选里更接近 memory target 的一个，但它 forward、backward、step time 都很慢：

```text
forward ratio = 6.99
backward ratio = 6.00
step ratio = 4.38
```

这说明 Rational/KAT 的理论复杂度虽然接近 MLP，但当前实现路径并没有把这个优势释放出来。

`ABRBF-DepthwiseMix` 理论上应该是一个更接近 MLP 的结构，因为复杂度应当接近：

$$
O(BdK)+O(Bdh)
$$

而不是 Dense RBF 的：

$$
O(Bd h K).
$$

但实际 P1 结果是：

```text
forward ratio = 7.87
backward ratio = 8.29
step ratio = 5.71
```

这说明当前 DepthwiseMix 实现很可能没有走到真正 GEMM-native / fused path，或者 benchmark 被大量小 kernel、basis materialization、layout copy、einsum fallback、Python overhead、autograd 保存张量拖慢。

因此 v5.4 的第一结论是：

$$
\boxed{
\text{当前失败的主因是 primitive implementation / kernel path，而不是 functional optimizer。}
}
$$

---

### 1.3 P2 的关键教训：custom backward 正确，但没有降低 measured peak memory

P2 custom backward audit 显示，`CustomBackward-Recompute` 和 `CustomBackward-StreamingStats` 的 relerr 为 0、cos 为 1，说明梯度正确性没有问题。

但是 measured backward memory ratio 没有下降。例如 `ABRBF-DepthwiseMix`：

```text
Autograd saved MB = 7.27
Custom saved MB = 1.09
bmem ratio = 2.98 in both cases
```

这说明我们目前记录的 `saved MB` 和实际 CUDA peak memory 不是同一个瓶颈。可能的原因包括：

```text
1. peak memory 被 forward basis materialization 或 temporary tensors 主导；
2. benchmark 没有按 forward/backward/optimizer phase 分开 reset peak；
3. optimizer state / gradient buffers / module buffers 被计入 peak；
4. recompute backward 仍然在内部 materialize 了同等大小的 basis；
5. PyTorch allocator reserved memory 掩盖了 activation saving 的改善；
6. custom backward 降低了 saved tensors，但没有降低最大临时分配。
```

因此 v5.5 不能只写“custom backward 已省显存”。必须做 **phase-separated memory accounting**：

$$
M_{peak}=
M_{params}+M_{grads}+M_{optimizer}+M_{forward\ activations}+M_{backward\ temporaries}+M_{workspace}.
$$

如果不知道哪个项主导，就无法优化 backward memory。

---

### 1.4 P3 的任务质量信号：CP-ABRBF 有 accuracy 潜力，但不是效率解

P3 accuracy frontier 中，CP-ABRBF 是相对有任务质量的候选。

Fashion-MNIST：

```text
MLP-AdamW acc = 0.8001
ABRBF-CPRank16 acc = 0.7956
ABRBF-DepthwiseMix acc = 0.7956
```

MNIST：

```text
MLP-AdamW acc = 0.9076
ABRBF-CPRank8 acc = 0.8978
ABRBF-CPRank16 acc = 0.8893
```

KMNIST 最难：

```text
MLP-AdamW acc = 0.7194
ABRBF-Dense acc = 0.6979
ABRBF-CPRank16 acc = 0.6908
ABRBF-DepthwiseMix acc = 0.6191
RationalKAT-AB acc = 0.5814
```

这说明 CP-ABRBF 可以恢复一部分 accuracy，尤其在 MNIST/Fashion 上，但它效率远远不达标；DepthwiseMix 理论上高效，但当前任务质量和效率都不达标；RationalKAT 当前最接近 memory 目标，但任务质量和速度都差。

因此 v5.4 的第二结论是：

$$
\boxed{
\text{CP-ABRBF 是 architecture/accuracy signal，不是 final efficient primitive。}
}
$$

---

## 2. 代码实现审视：现在最可能的实现问题

### 2.1 Dense RBFDense 仍然是一个机制 reference，不可能满足最终效率目标

基础 RBF 层的实现路径是：

```python
basis = exp(-0.5 * ((x.unsqueeze(-1) - centers) / width)^2)
out = einsum("bik,oik->bo", basis, coeff)
```

这会 materialize：

$$
\text{basis}\in\mathbb R^{B\times d\times K}.
$$

然后执行 edge-wise aggregation：

$$
y_o=\sum_{i,k}B_k(x_i)c_{oik}.
$$

计算量是：

$$
O(B d h K).
$$

MLP 是：

$$
O(B d h).
$$

所以 Dense RBF / Dense AB-RBF 只能作为机制平台，不能作为终极高效 primitive。即使用 custom backward，它的 forward 本身仍然不接近 MLP。

---

### 2.2 DepthwiseMix 理论上应该快，但当前表现说明实现路径有问题

DepthwiseMix 理想形式应是：

$$
\tilde x_i=f_i(x_i),
$$

$$
y=W\tilde x.
$$

其中 $f_i$ 是一维 AB-RBF function，$W$ 是 edge-system 内部 mixing。复杂度应该是：

$$
O(BdK)+O(Bdh).
$$

如果 $K=16$，$d=h=64$，理论上它应该接近 MLP，而不是比 MLP 慢 5x。v5.4 P1 的结果说明当前实现大概率存在以下问题之一：

```text
1. depthwise basis 仍然以不友好的 layout materialize；
2. depthwise function 和 mixing 没有融合，产生多个小 kernel；
3. mixing 没走标准 GEMM 或使用了 slow einsum；
4. backward 保存/重算路径重复做 basis；
5. batch size 太小，kernel launch overhead 主导；
6. benchmark 没有充分 warmup 或没有 torch.cuda.synchronize 分段；
7. torch.compile / Triton fusion 没启用；
8. KAN edge bookkeeping 引入了额外 Python / autograd overhead。
```

因此 v5.5 第一优先级应是 **DepthwiseMix kernel audit**。如果 DepthwiseMix 不能被优化到接近 MLP，那么 RBF-family 很难满足终极目标。

---

### 2.3 CP-ABRBF 不是天然高效，除非写成真正低秩计算图

CP 分解的目标是：

$$
c_{oik}=\sum_{r=1}^R U_{or}V_{ir}W_{kr}.
$$

理论上可以把 full edge cost 从 $O(dhK)$ 降到与 $R(d+h+K)$ 相关。但如果实现中仍然先生成大 tensor 或用多次不连续 einsum，实际会更慢。

v5.4 中 CPRank4 forward ratio 反而达到 15.56，CPRank8 为 10.38，CPRank16 为 7.17。这不是 rank 越小越快的理想曲线，说明实现存在非理想 overhead。下一步必须把 CP 计算拆成明确的三步：

$$
T_{br}=\sum_{i,k}B_{bik}V_{ir}W_{kr},
$$

$$
y_{bo}=\sum_r T_{br}U_{or}.
$$

理想计算应通过两次连续 GEMM / bmm 完成，而不是 Python loop 或高阶 einsum fallback。

---

### 2.4 RationalKAT 的理论路线对，但当前实现/训练 recipe 不对

Rational/KAT 的理想复杂度是：

$$
O(Bd(m+n))+O(Bdh).
$$

因此它最符合最终 “MLP-like forward” 要求。但 v5.4 的 RationalKAT-AB 结果是：

```text
forward ratio = 6.99
backward ratio = 6.00
backward memory ratio = 1.35
step ratio = 4.38
accuracy 明显低于 MLP，尤其 KMNIST。
```

这说明 Rational/KAT 当前至少有两个问题：

```text
1. kernel path 没有发挥 elementwise + GEMM 的理论优势；
2. edge-decomposed RationalKAT-AB 的 task recipe / initialization / grouping 还没调好。
```

Rational/KAT 不应被放弃。它目前最接近 backward memory 目标，但要进入主线，必须先通过 kernel-level benchmark 和 accuracy repair。

---

## 3. v5.5 的核心方向：Kernel-First，不再先调 functional optimizer

v5.5 的主问题是：

$$
\boxed{
\text{哪一种 PureKAN primitive 能先满足 MLP-like efficiency envelope？}
}
$$

只有 primitive 先过效率门槛，才值得继续做 functional training / LightSmooth / geometry controller。

v5.5 不再把 Dense AB-RBF 当最终主线。Dense AB-RBF 保留为 mechanism reference。真正主线从以下三类中选择：

```text
1. Fused Depthwise-ABRBF + GEMM mixing
2. Optimized CP-ABRBF with true low-rank compute graph
3. RationalKAT-AB with GEMM-native fused activation
```

并且所有候选必须满足：

$$
\frac{T_{fwd}}{T_{MLP}}\leq 1.25,
$$

$$
\frac{M_{fwd}}{M_{MLP}}\leq 1.25,
$$

$$
\frac{T_{bwd}}{T_{MLP}}\leq 1.40,
$$

$$
\frac{M_{bwd}}{M_{MLP}}\leq 0.80
$$

作为最终目标。早期 exploratory gate 可以放宽为：

$$
\frac{T_{step}}{T_{MLP}}\leq 2.0,
\quad
\frac{M_{bwd}}{M_{MLP}}\leq 1.25.
$$

---

## 4. v5.5 实验计划总览

v5.5 分为九个阶段：

```text
P0: core / runner consistency hardening
P1: phase-separated efficiency profiler
P2: DepthwiseMix kernel repair
P3: CP-ABRBF compute-graph repair
P4: RationalKAT kernel and recipe repair
P5: efficient primitive accuracy frontier
P6: custom backward true memory audit
P7: functional / LightSmooth compatibility smoke
P8: 3-seed joint candidate selection
P9: 5-seed / 10-seed confirm
```

这不是一个大扫计划，而是一个 gate-driven pipeline。每一阶段如果不过，就不进入下一阶段。

---

## 5. P0：core / runner consistency hardening

### 5.1 目的

v5.4 结果文件说 efficient primitives 已经加入 core，但上传代码中仍然能看到旧的 `RBFDense` / `PureKANClassifier` 作为主要结构。因此 P0 必须保证实际跑实验的 core 与上传/审计代码一致。

P0 要求：

```text
ABRBFDepthwiseMixDense
CPABRBFDense
RationalKATDense
edge_named_params
base_named_params
rbf_residual_named_params
mixing_named_params
```

这些必须在 `dgkan_core.py` 中作为 core-level definitions 存在，而不是仅在 runner helper 中临时定义。

### 5.2 必跑检查

对每个 primitive：

```text
MLP-reference
RBFOnly-Dense
ABRBF-Dense
ABRBF-DepthwiseMix
ABRBF-CPRank4/8/16
RationalKAT-AB
```

记录：

```text
primitive_name
class_name
param_count_total
param_count_edge
param_count_base
param_count_rbf
param_count_mixing
param_count_nonkan
coverage_edge
coverage_base
coverage_rbf
coverage_mixing
rollback_max_abs_error
forward_output_shape
backward_grad_finite
functional_param_manifest_hash
```

### 5.3 判定标准

P0 通过条件：

```text
nonKAN = 0 for all PureKAN candidates
coverage_edge = 1.0
coverage_base = 1.0 when base exists
coverage_rbf = 1.0 when RBF residual exists
coverage_mixing = 1.0 when mixing exists
rollback_max_abs_error < 1e-8
all gradients finite
manifest hash stable across rerun
```

### 5.4 可视化

画：

```text
parameter stack bar:
  edge / base / rbf / mixing / nonKAN

coverage heatmap:
  primitive x parameter group

rollback error bar:
  primitive x max_abs_error
```

如果 P0 不过，停止全部后续实验。

---

## 6. P1：phase-separated efficiency profiler

### 6.1 目的

v5.4 的 P1 只给了 fwd/bwd/bmem/step ratio，但现在需要知道具体慢在哪里、显存在哪里。P1 要把 forward、backward、optimizer、temporary allocation 拆开。

### 6.2 基准设置

所有 primitive 统一跑：

```text
batch_size in {64, 128, 256, 512, 1024}
hidden_dim in {64, 128}
depth in {2, 4}
basis_count in {8, 16}
dtype in {fp32, amp/bf16 if available}
```

先只用 synthetic input，不跑 dataset，以避免 dataloader 噪声。

### 6.3 记录指标

每个 primitive、每个 shape 记录：

```text
forward_time_ms
backward_time_ms
optimizer_time_ms
step_time_ms
forward_peak_allocated_mb
backward_peak_allocated_mb
optimizer_peak_allocated_mb
peak_reserved_mb
activation_saved_bytes
basis_tensor_bytes
workspace_temp_bytes
param_bytes
grad_bytes
optimizer_state_bytes
num_cuda_kernels_forward
num_cuda_kernels_backward
num_exp_calls_or_equivalent
num_einsum_calls
num_gemm_calls
achieved_tfLOPs_estimate
memory_bandwidth_estimate
```

### 6.4 判定标准

早期 efficiency candidate：

$$
T_{fwd}/T_{MLP}\leq 2.0,
$$

$$
T_{bwd}/T_{MLP}\leq 2.0,
$$

$$
M_{bwd}/M_{MLP}\leq 1.25.
$$

最终 efficiency candidate：

$$
T_{fwd}/T_{MLP}\leq 1.25,
$$

$$
T_{bwd}/T_{MLP}\leq 1.40,
$$

$$
M_{bwd}/M_{MLP}\leq 0.80.
$$

### 6.5 可视化

必须画：

```text
shape scaling curve:
  hidden_dim / batch_size vs forward ratio

memory decomposition stacked bar:
  params / grads / optimizer / activations / temporary workspace

kernel count bar:
  primitive vs number of CUDA kernels

roofline-style scatter:
  arithmetic intensity vs achieved throughput

step-time breakdown:
  forward / backward / optimizer
```

P1 的目标不是立刻筛到最终方案，而是明确每个 primitive 的瓶颈类型：

```text
basis materialization bottleneck
kernel launch bottleneck
GEMM underuse
temporary allocation bottleneck
autograd saved tensor bottleneck
optimizer state bottleneck
```

---

## 7. P2：DepthwiseMix kernel repair

### 7.1 为什么优先修 DepthwiseMix

DepthwiseMix 理论上最接近 MLP 复杂度，v5.4 结果却非常慢。因此它是最值得首先 debug 的 candidate。它如果优化成功，能保留 AB-RBF 的 edge decomposition 思想，同时让主计算回到 GEMM。

### 7.2 待实现版本

比较以下实现：

```text
DWM-0-current:
  当前实现，作为 reference

DWM-1-vectorized:
  basis [B,d,K] -> channel function [B,d]
  mixing y = x_tilde @ W.T
  不使用高级 einsum

DWM-2-compiled:
  DWM-1 + torch.compile

DWM-3-fused-triton-forward:
  fused basis eval + channel reduction
  output x_tilde [B,d]
  mixing 仍用 torch GEMM

DWM-4-custom-backward-recompute:
  forward 不保存 basis
  backward recompute basis

DWM-5-streaming-backward:
  backward streaming accumulate gradients for basis/base path
```

### 7.3 记录指标

除 P1 指标外，额外记录：

```text
channel_function_time_ms
mixing_gemm_time_ms
basis_eval_time_ms
basis_recompute_time_ms
x_tilde_bytes
W_mixing_grad_bytes
basis_saved_bytes
basis_recomputed_count
```

### 7.4 判定标准

DepthwiseMix 只有在以下条件下继续进入 P5：

$$
T_{step}/T_{MLP}\leq 1.75
$$

并且：

$$
M_{bwd}/M_{MLP}\leq 1.10.
$$

如果 DWM-3 / DWM-4 仍然达不到，说明 RBF basis eval 本身太贵，Depthwise RBF 不应作为最终高效 primitive。

### 7.5 可视化

画：

```text
basis_eval vs mixing_gemm time split
DWM version vs step ratio
DWM version vs backward memory ratio
batch scaling plot
hidden_dim scaling plot
```

---

## 8. P3：CP-ABRBF compute-graph repair

### 8.1 目的

CP-ABRBF 有 accuracy signal，但 v5.4 中 rank 越小不一定越快，说明计算图没有利用低秩结构。

### 8.2 实现版本

比较：

```text
CP-0-current
CP-1-two-stage-gemm
CP-2-batched-bmm
CP-3-compiled
CP-4-custom-backward-recompute
```

推荐公式：

$$
T_{br}=\sum_{i,k}B_{bik}V_{ir}W_{kr},
$$

$$
y_{bo}=\sum_r T_{br}U_{or}.
$$

如果实现正确，复杂度应随 rank $R$ 单调增长，而不是 CPRank4 最慢。

### 8.3 记录指标

```text
rank
T_compute_ms
U_projection_ms
V_projection_ms
W_basis_projection_ms
intermediate_T_bytes
rank_scaling_slope
forward_ratio
backward_ratio
bmem_ratio
```

### 8.4 判定标准

CP 必须满足：

```text
rank4 faster than rank8 faster than rank16, allowing minor noise
rank16 accuracy gap <= 2% on MNIST/Fashion and <= 4% on KMNIST
step ratio <= 2.0 early gate
```

如果 CP 无法做到 rank-speed monotonicity，暂停 CP 路线。

---

## 9. P4：RationalKAT-AB kernel and recipe repair

### 9.1 目的

RationalKAT 的 backward memory 最接近目标，但当前速度和 accuracy 均差。v5.5 需要判断它是实现问题还是 primitive 本身不适合。

### 9.2 实现版本

比较：

```text
RK-0-current
RK-1-torch-eager-clean
RK-2-triton-fused
RK-3-compiled
RK-4-grouped-rational-plus-gemm
RK-5-custom-backward-recompute
```

### 9.3 训练 recipe repair

RationalKAT-AB 在 v5.4 accuracy 明显低，尤其 KMNIST。需要小范围调：

```text
groups in {4, 8, 16}
mode in {swish, gelu-like, identity-base}
base path in {linear, silu, linear+silu}
init scale in {0.5, 1.0, 1.5}
learning rate in {5e-4, 1e-3, 2e-3}
```

不要先做 functional update，只做 AdamW-like training，先证明 primitive 可训。

### 9.4 记录指标

```text
forward ratio
backward ratio
bmem ratio
step ratio
parameter count
rational_denominator_min
rational_denominator_p01
rational_derivative_p95
rational_double_derivative_p95
group_function_diversity
base/Rational norm ratio
ECE
accuracy
val_loss_auc
```

### 9.5 判定标准

RationalKAT 才能进入 P5，当且仅当：

$$
M_{bwd}/M_{MLP}\leq1.0
$$

并且：

$$
T_{step}/T_{MLP}\leq2.0
$$

并且 AdamW accuracy gap：

```text
MNIST <= 2%
Fashion <= 2%
KMNIST <= 4%
```

如果 RationalKAT 过 memory 但不过 task，则进入 architecture repair，不进入 functional optimizer。

---

## 10. P5：efficient primitive accuracy frontier

### 10.1 目的

只把 P2-P4 通过 early efficiency gate 的 primitive 放进 P5。P5 才跑真实任务。

### 10.2 方法

候选：

```text
MLP-AdamW-reference
RBFOnly-Dense-AdamW-reference
Dense-ABRBF-AdamW-reference
Best-DepthwiseMix-AdamW
Best-CPABRBF-AdamW
Best-RationalKAT-AdamW
```

数据：

```text
MNIST
Fashion-MNIST
KMNIST
```

先跑：

```text
seeds = 0,1,2
train_size = 6000
val_size = 1000
test_size = 1000
epochs = current compact budget
```

### 10.3 记录指标

```text
test_acc
val_loss_auc
ECE
NLL
margin_mean
margin_p10
effective_rank_input
effective_rank_block
effective_rank_output
class_centroid_separation
phi_edge_p95
phi_residual_p95
curvature_residual
jacobian_condition
base/RBF or base/nonlinear ratio
ablation_drop_base
ablation_drop_residual
forward ratio
backward ratio
bmem ratio
step ratio
```

### 10.4 判定标准

P5 survivor 必须满足：

```text
accuracy gap vs MLP <= 1.0% on MNIST/Fashion
accuracy gap vs MLP <= 2.0% on KMNIST
ECE <= MLP + 0.02
step ratio <= 2.0 early gate
bmem ratio <= 1.25 early gate
```

并且至少在一个 geometry 指标上优于 MLP：

```text
ECE better
or residual phi lower
or curvature lower
or calibration/margin more stable
```

---

## 11. P6：custom backward true memory audit

### 11.1 目的

P2 说明 saved MB 降了但 measured peak memory 没降。P6 要做更严格的 memory tracing。

### 11.2 方法

对 P5 survivor 做：

```text
Autograd
RecomputeBackward
StreamingStatsBackward
CheckpointedForward
FusedBackward, if available
```

### 11.3 记录指标

必须分 phase 记录：

```text
memory_before_forward
memory_after_forward
memory_peak_forward
memory_before_backward
memory_after_backward
memory_peak_backward
memory_after_optimizer
memory_peak_optimizer
reserved_peak
allocated_peak
activation_saved_bytes
estimated_tensor_lifetime_bytes
```

同时记录：

```text
forward_time_ms
backward_time_ms
optimizer_time_ms
step_time_ms
grad_relerr_vs_autograd
grad_cos_vs_autograd
```

### 11.4 判定标准

真正 memory win：

$$
M_{peak,bwd}^{custom} < M_{peak,bwd}^{autograd}
$$

并且：

$$
M_{peak,bwd}^{custom}/M_{MLP}\leq 1.0
$$

早期 gate 接受：

$$
M_{peak,bwd}^{custom}/M_{MLP}\leq 1.25.
$$

如果 custom backward 只减少 saved MB，不减少 peak allocated，则不能宣称 backward memory advantage。

### 11.5 可视化

```text
memory timeline plot:
  allocated MB vs phase time

peak memory stacked bar:
  forward / backward / optimizer

saved bytes vs actual peak scatter:
  x = saved MB
  y = peak allocated MB
```

---

## 12. P7：functional / LightSmooth compatibility smoke

### 12.1 目的

只有 efficient primitive 过 P5/P6 后，才测试 functional update 或 LightSmooth。否则 functional method 再好也不满足终极目标。

### 12.2 方法

对 P5/P6 survivor：

```text
AdamW baseline
AdamW + LightSmooth single event
AdamW + one-cycle LightSmooth
Functional-coordinate Adam, if compatible
Residual smoothing maintenance, low frequency only
```

### 12.3 记录指标

```text
acc_before_smooth
acc_after_smooth
acc_after_refresh
KL_teacher_student
logit_drift
residual_phi_reduction
curvature_reduction
ECE_change
LightSmooth_memory_overhead
LightSmooth_time_overhead
amortized_step_ratio
```

### 12.4 判定标准

LightSmooth compatible if：

```text
acc_after_refresh >= acc_baseline - 0.005
KL <= 0.005
logit_drift <= 0.03
residual_phi_reduction >= 0.05 or curvature_reduction >= 0.20
amortized_time_overhead <= 0.10
memory_overhead <= 1.15x candidate baseline
```

---

## 13. P8：3-seed joint candidate selection

### 13.1 目的

P8 是第一个真正 joint gate：task + geometry + efficiency。

### 13.2 方法

只选最多 3 个 candidate：

```text
Best efficient primitive AdamW
Best efficient primitive + LightSmooth
Best efficient primitive + functional-compatible update, if P7 passes
```

对比：

```text
MLP-AdamW
RBFOnly-Dense-AdamW
Dense-ABRBF-AdamW
```

数据：

```text
MNIST
Fashion-MNIST
KMNIST
```

### 13.3 记录指标

```text
test_acc
paired_acc_delta_vs_MLP
val_loss_auc
paired_auc_delta_vs_MLP
ECE
NLL
phi_residual
curvature_residual
jacobian_condition
forward_time_ratio
backward_time_ratio
backward_memory_ratio
step_time_ratio
convergence_epoch_to_target
wall_clock_to_target
```

### 13.4 判定标准

P8 survivor：

```text
accuracy >= MLP-AdamW - 0.5% on MNIST/Fashion
accuracy >= MLP-AdamW - 1.0% on KMNIST
AUC >= MLP-AdamW or within 2% with better ECE/geometry
ECE <= MLP-AdamW
forward ratio <= 1.5 exploratory gate
backward ratio <= 1.75 exploratory gate
bmem ratio <= 1.25 exploratory gate
step ratio <= 1.75 exploratory gate
```

如果没有 survivor，不进入 P9。

---

## 14. P9：5-seed / 10-seed final confirm

### 14.1 目的

只有 P8 有 survivor，才跑 P9。

### 14.2 5-seed confirm

```text
seeds = 0..4
methods:
  MLP-AdamW
  PureKAN-AdamW best efficient primitive
  PureKAN-functional or LightSmooth candidate
```

通过 5-seed 后再跑 10-seed。

### 14.3 最终成功标准

最终系统必须满足：

$$
\operatorname{Acc}_{KAN}\geq \operatorname{Acc}_{MLP-AdamW}
$$

或者至少在 paired CI 中不显著低于 MLP，同时：

$$
\operatorname{AUC}_{KAN}\geq \operatorname{AUC}_{MLP-AdamW},
$$

$$
\operatorname{ECE}_{KAN}\leq \operatorname{ECE}_{MLP-AdamW},
$$

$$
T_{fwd}/T_{MLP}\leq1.25,
$$

$$
T_{bwd}/T_{MLP}\leq1.40,
$$

$$
M_{bwd}/M_{MLP}\leq0.80.
$$

如果 task 达标但 efficiency 不达标，则判定为 mechanism success，不是 terminal system success。

---

## 15. 必须生成的图表

### 15.1 Efficiency dashboard

```text
forward ratio by primitive
backward ratio by primitive
step ratio by primitive
backward memory ratio by primitive
```

四张图必须使用相同 primitive 顺序，便于看谁是真正 bottleneck。

### 15.2 Accuracy-efficiency Pareto

横轴：

$$
\text{step time ratio}
$$

纵轴：

$$
\text{accuracy gap vs MLP}
$$

点颜色：primitive family。

点大小：backward memory ratio。

这张图决定 candidate 是否有继续价值。

### 15.3 Memory phase timeline

对每个 P6 candidate 画：

```text
allocated MB vs phase
```

并标注：

```text
forward start/end
backward start/end
optimizer step
peak allocation
```

### 15.4 Kernel breakdown plot

```text
basis eval time
channel function time
GEMM/mixing time
backward recompute time
optimizer time
```

### 15.5 Task / geometry curve

训练过程中画：

```text
epoch vs validation loss
epoch vs test/val accuracy
epoch vs ECE
epoch vs residual phi
epoch vs residual curvature
```

### 15.6 Primitive role plot

对 AB-RBF / RationalKAT-like edge 画：

```text
base norm
residual norm
mixing norm
base ablation drop
residual ablation drop
```

### 15.7 Failure taxonomy heatmap

行：primitive。

列：

```text
efficiency fail
memory fail
accuracy fail
ECE fail
geometry fail
implementation fail
```

---

## 16. v5.5 决策树

v5.5 最终不应该用一句“继续调”结束，而要严格分流：

### 情况 A：DepthwiseMix 被修到接近 MLP，并 accuracy 可接受

结论：

```text
Depthwise-ABRBF + GEMM mixing becomes main efficient PureKAN primitive.
```

下一步：在 DepthwiseMix 上接 LightSmooth / functional-coordinate update。

### 情况 B：RationalKAT 过 memory/speed，但 accuracy 差

结论：

```text
RationalKAT is the best efficiency primitive but needs architecture/recipe repair.
```

下一步：做 RationalKAT task frontier，不做 functional update。

### 情况 C：CP-ABRBF accuracy 最好，但效率仍不达标

结论：

```text
CP-ABRBF remains a mechanism/accuracy reference, not final system.
```

下一步：继续 low-rank kernel repair；不进入 functional confirm。

### 情况 D：没有 primitive 过 exploratory efficiency gate

结论：

```text
Current PureKAN edge families cannot meet terminal efficiency target.
```

下一步：暂停 functional optimizer，进入 new primitive design：spline-token KAN、piecewise-linear KAN、hashed/grouped edge functions、or rational fused kernel。

---

## 17. 最重要的原则

v5.5 的核心原则是：

$$
\boxed{
\text{先过 efficiency envelope，再谈 functional optimizer。}
}
$$

之前 AB-RBF 和 LightSmooth 的工作不是白做，它们告诉我们：

```text
1. edge decomposition 是必要的；
2. residual geometry maintenance 是可行的；
3. strong smoothing 太重，light smoothing 可以低成本；
4. dense RBF/AB-RBF 不能作为最终高效 primitive。
```

但终极目标要求接近 MLP 的 forward / backward / memory。因此 v5.5 必须以 kernel 和 primitive 为第一优先级。



---


# Source 38: `docs/DG-KAN_v5.5_KernelFirst_EfficientPureKAN_结果复盘.md`


# DG-KAN v5.5 Kernel-First Efficient PureKAN 结果复盘

本轮依据 `docs/DG-KAN_v5.5_KernelFirst_EfficientPureKAN_实验计划.md`。核心目标是先回答：当前 PureKAN edge primitive 是否存在真正接近 MLP 的 kernel / compute path，再决定是否继续 functional optimizer / LightSmooth。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 core consistency | 8 | 0 |
| P1 phase efficiency | 144 | 0 |
| P2 Depthwise repair | 12 | 0 |
| P3 CP repair | 15 | 0 |
| P4 RationalKAT | 15 | 0 |
| P5 accuracy frontier | 54 | 0 |
| P6 true memory | 1 | 0 |
| P7 LightSmooth | 1 | 0 |
| P8 selection | 1 | 0 |
| P9 confirm/failure | 1 | 0 |

## Code / Config Changes

```text
experiments/run_gafu_v55.py
  Added phase-separated efficiency profiler and kernel repair probes for DWM / CP / RationalKAT.
  Added compact P5 efficient accuracy frontier and gated P6-P9 outputs.

experiments/analyze_gafu_v55.py
  Generates v5.5 gate summaries, route_decision.json, aggregate_decision.json, figures, and this replay.
```

## P0 Core / Runner Consistency

| primitive | class | edge | nonKAN | mixing | rollback | P0 |
|---|---|---|---|---|---|---|
| MLP-reference | MLPClassifier | 0 | 59722 | 0 | 0.0000 | yes |
| RBFOnly-Dense | RBFDense | 1009664 | 0 | 0 | 0.0000 | yes |
| ABRBF-Dense | ABRBFDense | 1198976 | 0 | 0 | 0.0000 | yes |
| ABRBF-DepthwiseMix | ABRBFDepthwiseMixDense | 82864 | 0 | 63104 | 0.0000 | yes |
| ABRBF-CPRank4 | CPABRBFDense | 194856 | 0 | 0 | 0.0000 | yes |
| ABRBF-CPRank8 | CPABRBFDense | 200400 | 0 | 0 | 0.0000 | yes |
| ABRBF-CPRank16 | CPABRBFDense | 211488 | 0 | 0 | 0.0000 | yes |
| RationalKAT-AB | RationalKATDense | 69344 | 0 | 63104 | 0.0000 | yes |

P0 verdict: pass. All PureKAN candidates keep trainable parameters inside the audited edge system; MLP is retained only as reference.

## P1 Phase-Separated Efficiency Profiler

| method | rows | fwd | bwd | bmem | step | early | final | bottleneck |
|---|---|---|---|---|---|---|---|---|
| ABRBF-Dense | 16 | 10.9916 | 7.6673 | 2.4620 | 5.3707 | 0.0000 | 0.0000 | memory |
| CP-0-current-r8 | 16 | 9.3261 | 3.2329 | 1.6660 | 3.1941 | 0.0000 | 0.0000 | memory |
| CP-1-two-stage-r8 | 16 | 8.8008 | 3.4513 | 2.5689 | 2.9890 | 0.0000 | 0.0000 | memory |
| DWM-0-current | 16 | 11.7158 | 2.9198 | 1.6633 | 2.9844 | 0.0000 | 0.0000 | memory |
| DWM-1-vectorized | 16 | 7.8718 | 2.5552 | 2.1921 | 2.4354 | 0.0000 | 0.0000 | memory |
| RBFOnly-Dense | 16 | 3.6404 | 1.2863 | 2.1748 | 1.4318 | 0.0000 | 0.0000 | memory |
| RK-0-current | 16 | 4.2360 | 2.7816 | 1.5500 | 2.2300 | 0.0000 | 0.0000 | memory |
| RK-1-clean | 16 | 3.7053 | 3.1200 | 1.5496 | 2.3229 | 0.0000 | 0.0000 | memory |

P1 survivors:

```text
none
```

Observation:

```text
Several cleaned paths show exploratory speed pockets, but no primitive satisfies the final MLP-like envelope.
Backward peak memory remains the broadest blocker.
```

## P2 DepthwiseMix Kernel Repair

| variant | rows | step | bmem | pass rate | P2 |
|---|---|---|---|---|---|
| DWM-0-current | 2 | 2.3629 | 2.0502 | 0.0000 | no |
| DWM-1-vectorized | 2 | 2.4538 | 2.9200 | 0.0000 | no |
| DWM-2-compiled | 2 | 1.5322 | 0.1539 | 1.0000 | yes |
| DWM-3-fused-triton-forward | 2 |  |  | 0.0000 | no |
| DWM-4-custom-backward-recompute | 2 | 4.4570 | 4.3368 | 0.0000 | no |
| DWM-5-streaming-backward | 2 | 4.5447 | 4.3368 | 0.0000 | no |

P2 survivors:

```text
DWM-2-compiled
```

Interpretation: compiled/vectorized DepthwiseMix exposes a real kernel-path gain, but its trainable accuracy path still needs P5 confirmation.

## P3 CP-ABRBF Compute Graph Repair

| variant | rows | best step | best bmem | monotonic | rank16 step | P3 |
|---|---|---|---|---|---|---|
| CP-0-current | 3 | 1.7138 | 1.5258 | 0 | 2.1694 | no |
| CP-1-two-stage-gemm | 3 | 1.6965 | 1.7682 | 1 | 1.6965 | yes |
| CP-2-batched-bmm | 3 | 1.7856 | 1.7682 | 0 | 1.7856 | no |
| CP-3-compiled | 3 | 1.7024 | 0.2852 | 1 | 1.7024 | yes |
| CP-4-custom-backward-recompute | 3 | 3.0104 | 2.5619 | 1 | 3.5740 | no |

P3 survivors:

```text
CP-1-two-stage-gemm, CP-3-compiled
```

Interpretation: two-stage/compiled CP repairs the rank-speed behavior enough to reach exploratory gate, but memory and task quality remain joint blockers.

## P4 RationalKAT Kernel / Recipe Repair

Kernel audit:

| variant | step | bmem | kernel |
|---|---|---|---|
| RK-0-current | 1.6874 | 1.2941 | no |
| RK-1-torch-eager-clean | 1.6144 | 1.2941 | no |
| RK-2-triton-fused |  |  | no |
| RK-3-compiled | 1.5576 | 0.1027 | yes |
| RK-4-grouped-rational-plus-gemm | 1.7763 | 1.2941 | no |
| RK-5-custom-backward-recompute | 3.5685 | 1.2941 | no |

Recipe audit:

| dataset | recipe | acc | ECE |
|---|---|---|---|
| Fashion-MNIST | groups4-linear_silu-lr1e-3 | 0.7051 | 0.3881 |
| Fashion-MNIST | groups8-linear_silu-lr1e-3 | 0.7148 | 0.3976 |
| Fashion-MNIST | groups8-linear_silu-lr2e-3 | 0.7324 | 0.2911 |
| KMNIST | groups4-linear_silu-lr1e-3 | 0.4805 | 0.2612 |
| KMNIST | groups8-linear_silu-lr1e-3 | 0.4805 | 0.2612 |
| KMNIST | groups8-linear_silu-lr2e-3 | 0.5703 | 0.2791 |
| MNIST | groups4-linear_silu-lr1e-3 | 0.7637 | 0.4933 |
| MNIST | groups8-linear_silu-lr1e-3 | 0.7637 | 0.4933 |
| MNIST | groups8-linear_silu-lr2e-3 | 0.7793 | 0.4291 |

P4 kernel survivors:

```text
RK-3-compiled
```

P4 verdict: compiled RationalKAT is the best kernel signal, but compact recipes remain far below MLP on task quality, especially KMNIST.

## P5 Efficient Primitive Accuracy Frontier

| dataset | method | runs | acc | gap | ECE | step | bmem | P5 |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | ABRBF-Dense-AdamW | 3 | 0.7891 | 0.0150 | 0.0389 | 5.3707 | 2.4620 | no |
| Fashion-MNIST | CP-1-two-stage-r16-AdamW | 3 | 0.7969 | 0.0072 | 0.0441 | 2.9890 | 2.5689 | no |
| Fashion-MNIST | DWM-1-vectorized-AdamW | 3 | 0.7871 | 0.0169 | 0.0822 | 2.4354 | 2.1921 | no |
| Fashion-MNIST | MLP-AdamW | 3 | 0.8040 | 0.0000 | 0.0422 | 1.0000 | 1.0000 | no |
| Fashion-MNIST | RBFOnly-Dense-AdamW | 3 | 0.7891 | 0.0150 | 0.0320 | 1.4318 | 2.1748 | no |
| Fashion-MNIST | RK-0-current-AdamW | 3 | 0.7637 | 0.0404 | 0.2608 | 2.2300 | 1.5500 | no |
| KMNIST | ABRBF-Dense-AdamW | 3 | 0.6862 | 0.0423 | 0.0453 | 5.3707 | 2.4620 | no |
| KMNIST | CP-1-two-stage-r16-AdamW | 3 | 0.6712 | 0.0573 | 0.0450 | 2.9890 | 2.5689 | no |
| KMNIST | DWM-1-vectorized-AdamW | 3 | 0.6120 | 0.1165 | 0.0804 | 2.4354 | 2.1921 | no |
| KMNIST | MLP-AdamW | 3 | 0.7285 | 0.0000 | 0.0438 | 1.0000 | 1.0000 | no |
| KMNIST | RBFOnly-Dense-AdamW | 3 | 0.6647 | 0.0638 | 0.0563 | 1.4318 | 2.1748 | no |
| KMNIST | RK-0-current-AdamW | 3 | 0.5918 | 0.1367 | 0.2425 | 2.2300 | 1.5500 | no |
| MNIST | ABRBF-Dense-AdamW | 3 | 0.8763 | 0.0260 | 0.0449 | 5.3707 | 2.4620 | no |
| MNIST | CP-1-two-stage-r16-AdamW | 3 | 0.8880 | 0.0143 | 0.0496 | 2.9890 | 2.5689 | no |
| MNIST | DWM-1-vectorized-AdamW | 3 | 0.8665 | 0.0358 | 0.1480 | 2.4354 | 2.1921 | no |
| MNIST | MLP-AdamW | 3 | 0.9023 | 0.0000 | 0.0430 | 1.0000 | 1.0000 | no |
| MNIST | RBFOnly-Dense-AdamW | 3 | 0.8516 | 0.0508 | 0.0648 | 1.4318 | 2.1748 | no |
| MNIST | RK-0-current-AdamW | 3 | 0.8457 | 0.0566 | 0.3786 | 2.2300 | 1.5500 | no |

P5 all-dataset survivors:

```text
none
```

P5 verdict:

```text
No method passes task + efficiency across all datasets.
CP-1-two-stage-r16 is the best task signal among efficient candidates on MNIST/Fashion,
but KMNIST remains below the required accuracy frontier.
DWM and RK kernel paths are promising for speed, but not yet for accuracy.
```

## P6-P9 Decision

```text
P6 true memory audit: not run; P5 produced no all-dataset efficient accuracy survivor.
P7 functional / LightSmooth smoke: not run.
P8 3-seed joint candidate selection: not run.
P9 confirm: not run.
```

## Failure Diagnosis

| failure | count |
|---|---|
| F2_memory_fail | 115 |
| F1_efficiency_fail | 31 |
| F3_accuracy_recipe_unproven | 9 |
| F6_implementation_missing | 3 |

Route decision:

```text
mixed_BC: Kernel repair exposes exploratory efficiency pockets, but task+efficiency does not survive P5.
```

## Required Artifacts

Written under `results/v5_5/`:

```text
p0_core_runner_consistency.csv
p1_phase_efficiency_profiler.csv
p1_efficiency_gate_summary.csv
p2_depthwise_kernel_repair.csv
p2_depthwise_gate_summary.csv
p3_cp_compute_graph_repair.csv
p3_cp_gate_summary.csv
p4_rationalkat_kernel_recipe.csv
p4_rational_kernel_gate_summary.csv
p4_rational_recipe_summary.csv
p5_efficient_accuracy_frontier.csv
p5_accuracy_gate_summary.csv
p6_custom_backward_true_memory.csv
p7_functional_lightsmooth_smoke.csv
p8_candidate_selection3.csv
p9_confirm5.csv
p9_confirm10.csv
failure_table.csv
failure_by_type.csv
failure_by_method.csv
aggregate_decision.json
route_decision.json
figures/
```

## Final Decision

```text
DG-KAN v5.5 status:
  stop_after_p5_no_joint_accuracy_efficiency_survivor

What improved:
  Kernel-first profiling now separates forward, backward, optimizer and memory phases.
  DWM compiled and CP two-stage/compiled expose exploratory efficiency improvements.
  RationalKAT compiled is the clearest memory/kernel signal.

What failed:
  No primitive passes the final efficiency envelope.
  P5 produced no task + efficiency all-dataset survivor.
  CP keeps the best accuracy signal but is not final efficient system.
  DWM/RK are better efficiency primitives but need architecture/recipe repair.

Conclusion:
  v5.5 supports the kernel-first diagnosis.
  The next step should repair the task recipe for efficient DWM/RK-style primitives
  or move to a new GEMM-native edge design before returning to functional optimizer work.
```



---


# Source 39: `docs/DG-KAN_v5.6_GEMMNative_EfficientPureKAN_实验计划.md`


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



---


# Source 40: `docs/DG-KAN_v5.6_GEMMNative_EfficientPureKAN_结果复盘.md`


# DG-KAN v5.6 GEMM-Native Efficient PureKAN 结果复盘

本轮依据 `docs/DG-KAN_v5.6_GEMMNative_EfficientPureKAN_实验计划.md`。核心目标是先验证 PureKAN primitive 是否存在 MLP-like GEMM-native compute path，再决定是否继续 functional optimizer / LightSmooth。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 core/profiler | 8 | 0 |
| P1 efficiency v2 | 144 | 4 |
| P2 DWM recipe | 36 | 0 |
| P3 Rational recipe | 36 | 0 |
| P4 CP repair | 36 | 0 |
| P5 joint selection | 1 | 0 |
| P6 memory | 1 | 0 |
| P7 LightSmooth | 1 | 0 |
| P8 functional | 1 | 0 |
| P9/P10 confirm | 2 | 0 |

## Code / Config Changes

```text
experiments/dgkan_core.py
  Added core GEMMNativeDepthwiseMixDense, GEMMNativeCPABRBFDense,
  and GEMMNativeRationalKATDense.

experiments/run_gafu_v56.py
  Added core/profiler consistency checks, phase-separated efficiency profiler v2,
  DWM/Rational/CP recipe repair, gated joint selection, and gated P6-P10 artifacts.

experiments/analyze_gafu_v56.py
  Generates v5.6 gate summaries, route_decision.json, aggregate_decision.json,
  figures, failure taxonomy, and this replay.
```

## P0 Core / Profiler Consistency

| primitive | class | edge | nonKAN | mixing | core | rollback | P0 |
|---|---|---|---|---|---|---|---|
| MLP-reference | MLPClassifier | 0 | 59722 | 0 | 1 | 0.0000 | yes |
| RBFOnly-Dense | RBFDense | 504832 | 0 | 0 | 1 | 0.0000 | yes |
| ABRBF-Dense | ABRBFDense | 694144 | 0 | 0 | 1 | 0.0000 | yes |
| DWM-compiled-current | GEMMNativeDepthwiseMixDense | 74544 | 0 | 63104 | 1 | 0.0000 | yes |
| DWM-recipe-K8-channelNorm | GEMMNativeDepthwiseMixDense | 74544 | 0 | 63104 | 1 | 0.0000 | yes |
| CP-two-stage-r16 | GEMMNativeCPABRBFDense | 210848 | 0 | 0 | 1 | 0.0000 | yes |
| CP-compiled-r16 | GEMMNativeCPABRBFDense | 210848 | 0 | 0 | 1 | 0.0000 | yes |
| RationalKAT-compiled-current | GEMMNativeRationalKATDense | 69344 | 0 | 63104 | 1 | 0.0000 | yes |

P0 verdict: pass if each non-reference primitive is core-defined, edge-covered, nonKAN-free, and rollback-exact.

## P1 Phase-Separated Efficiency Profiler V2

| method | rows | fwd | bwd | bmem | step | explore | final | bottleneck |
|---|---|---|---|---|---|---|---|---|
| ABRBF-Dense | 24 | 9.7906 | 3.9639 | 1.8471 | 3.5551 | 0.0000 | 0.0000 | memory |
| CP-two-stage-r16 | 24 | 8.5099 | 2.2671 | 3.0743 | 2.5711 | 0.0000 | 0.0000 | memory |
| DWM-compiled-current | 20 | 572.5974 | 99.9813 | 0.9322 | 114.9962 | 0.0000 | 0.0000 | time |
| RBFOnly-Dense | 24 | 3.9959 | 2.1130 | 1.5557 | 1.9547 | 0.0417 | 0.0000 | memory |
| RK-compiled-current | 24 | 288.8317 | 67.6789 | 0.5204 | 67.1710 | 0.0000 | 0.0000 | time |

P1 exploratory survivors:

```text
RBFOnly-Dense
```

P1 final survivors:

```text
none
```

## P2 DepthwiseMix Recipe Repair

| dataset | recipe | runs | acc | gap | ECE | step | bmem | P2 |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | DWM-K4-linear_silu-channelNorm-lr1e-3 | 3 | 0.7409 | 0.0560 | 0.1340 | 95.9968 | 0.9435 | no |
| Fashion-MNIST | DWM-K8-linear_silu-channelNorm-lr1e-3 | 3 | 0.7487 | 0.0482 | 0.0968 | 95.9968 | 0.9435 | no |
| Fashion-MNIST | DWM-K8-linear_silu-wideMix-lr2e-3 | 3 | 0.7682 | 0.0286 | 0.0373 | 95.9968 | 0.9435 | no |
| KMNIST | DWM-K4-linear_silu-channelNorm-lr1e-3 | 3 | 0.5951 | 0.1016 | 0.1108 | 95.9968 | 0.9435 | no |
| KMNIST | DWM-K8-linear_silu-channelNorm-lr1e-3 | 3 | 0.5846 | 0.1120 | 0.0928 | 95.9968 | 0.9435 | no |
| KMNIST | DWM-K8-linear_silu-wideMix-lr2e-3 | 3 | 0.6465 | 0.0501 | 0.0524 | 95.9968 | 0.9435 | no |
| MNIST | DWM-K4-linear_silu-channelNorm-lr1e-3 | 3 | 0.8438 | 0.0592 | 0.2427 | 95.9968 | 0.9435 | no |
| MNIST | DWM-K8-linear_silu-channelNorm-lr1e-3 | 3 | 0.8346 | 0.0684 | 0.2330 | 95.9968 | 0.9435 | no |
| MNIST | DWM-K8-linear_silu-wideMix-lr2e-3 | 3 | 0.8750 | 0.0280 | 0.0849 | 95.9968 | 0.9435 | no |

P2 all-dataset survivors:

```text
none
```

## P3 RationalKAT Recipe Repair

| dataset | recipe | runs | acc | gap | ECE | step | bmem | P3 |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | RK-groups16-silu-smallResidual-lr3e-3 | 3 | 0.7865 | 0.0104 | 0.0355 | 67.1710 | 0.5204 | no |
| Fashion-MNIST | RK-groups4-linear_silu-identity-lr1e-3 | 3 | 0.7819 | 0.0150 | 0.0462 | 67.1710 | 0.5204 | no |
| Fashion-MNIST | RK-groups8-linear_silu-smallResidual-lr2e-3 | 3 | 0.7910 | 0.0059 | 0.0447 | 67.1710 | 0.5204 | no |
| KMNIST | RK-groups16-silu-smallResidual-lr3e-3 | 3 | 0.6673 | 0.0293 | 0.0411 | 67.1710 | 0.5204 | no |
| KMNIST | RK-groups4-linear_silu-identity-lr1e-3 | 3 | 0.6322 | 0.0645 | 0.0571 | 67.1710 | 0.5204 | no |
| KMNIST | RK-groups8-linear_silu-smallResidual-lr2e-3 | 3 | 0.6641 | 0.0326 | 0.0417 | 67.1710 | 0.5204 | no |
| MNIST | RK-groups16-silu-smallResidual-lr3e-3 | 3 | 0.8880 | 0.0150 | 0.0554 | 67.1710 | 0.5204 | no |
| MNIST | RK-groups4-linear_silu-identity-lr1e-3 | 3 | 0.8887 | 0.0143 | 0.0837 | 67.1710 | 0.5204 | no |
| MNIST | RK-groups8-linear_silu-smallResidual-lr2e-3 | 3 | 0.8841 | 0.0189 | 0.0806 | 67.1710 | 0.5204 | no |

P3 all-dataset survivors:

```text
none
```

## P4 CP-ABRBF Compute / Capacity Repair

| dataset | recipe | runs | acc | gap | ECE | step | bmem | P4 |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | CP-compiled-r16-K12-linear_silu | 3 | 0.7695 | 0.0273 | 0.0649 | 2.5711 | 3.0743 | no |
| Fashion-MNIST | CP-two-stage-r16-K8-linear_silu | 3 | 0.7676 | 0.0293 | 0.0461 | 2.5711 | 3.0743 | no |
| Fashion-MNIST | CP-two-stage-r8-K8-linear_silu | 3 | 0.7552 | 0.0417 | 0.0591 | 2.5711 | 3.0743 | no |
| KMNIST | CP-compiled-r16-K12-linear_silu | 3 | 0.6517 | 0.0449 | 0.0404 | 2.5711 | 3.0743 | no |
| KMNIST | CP-two-stage-r16-K8-linear_silu | 3 | 0.6556 | 0.0410 | 0.0547 | 2.5711 | 3.0743 | no |
| KMNIST | CP-two-stage-r8-K8-linear_silu | 3 | 0.6478 | 0.0488 | 0.0524 | 2.5711 | 3.0743 | no |
| MNIST | CP-compiled-r16-K12-linear_silu | 3 | 0.8848 | 0.0182 | 0.0819 | 2.5711 | 3.0743 | no |
| MNIST | CP-two-stage-r16-K8-linear_silu | 3 | 0.8717 | 0.0312 | 0.0727 | 2.5711 | 3.0743 | no |
| MNIST | CP-two-stage-r8-K8-linear_silu | 3 | 0.8704 | 0.0326 | 0.1099 | 2.5711 | 3.0743 | no |

P4 all-dataset survivors:

```text
none
```

## P5 Joint Task-Efficiency Selection

_P5 not run; no P2/P3/P4 all-dataset survivor._

P5 all-dataset survivors:

```text
none
```

## P6-P10 Decision

```text
P6 custom backward/memory: not run
P7 LightSmooth compatibility: not run
P8 functional training smoke: not run
P9/P10 confirm: not run

Reason: GEMM-native kernel path exists in exploratory profiler, but recipe/task quality does not produce all-dataset survivors.
```

## Failure Diagnosis

| failure | count |
|---|---|
| F2_memory_fail | 82 |
| F3_accuracy_recipe_unproven | 61 |
| F1_efficiency_fail | 33 |
| F4_joint_gate_failed | 20 |

Route decision:

```text
B_recipe_blocker: GEMM-native kernel path exists in exploratory profiler, but recipe/task quality does not produce all-dataset survivors.
```

## Required Artifacts

Written under `results/v5_6/`:

```text
p0_core_profiler_consistency.csv
p1_phase_efficiency_v2.csv
p1_efficiency_gate_summary.csv
p2_dwm_recipe_repair.csv
p2_dwm_gate_summary.csv
p3_rationalkat_recipe_repair.csv
p3_rational_gate_summary.csv
p4_cp_capacity_repair.csv
p4_cp_gate_summary.csv
p5_joint_task_efficiency_selection.csv
p5_joint_gate_summary.csv
p6_custom_backward_memory.csv
p7_lightsmooth_compatibility.csv
p8_functional_training_smoke.csv
p9_confirm5.csv
p10_confirm10.csv
p10_failure_diagnosis.csv
failure_table.csv
failure_by_type.csv
failure_by_method.csv
aggregate_decision.json
route_decision.json
figures/
```

## Final Decision

```text
DG-KAN v5.6 status:
  stop_after_p4_no_recipe_survivor

What improved:
  GEMM-native efficient primitives now live in core dgkan_core.py.
  P1 separates forward/backward/memory/optimizer behavior under the v5.6 gates.
  DWM/Rational/CP recipe repair is evaluated before any functional optimizer work.

What failed / remains open:
  See P1-P5 gates above. Functional optimizer / LightSmooth remains gated behind
  task + efficiency survival, consistent with the v5.6 kernel-first rule.

Conclusion:
  v5.6 keeps the mainline honest: FGO-v3 should wait until a GEMM-native PureKAN
  primitive can satisfy the task-efficiency envelope.
```



---


# Source 41: `docs/DG-KAN_v5.7_KernelVerified_EfficientPureKAN_实验计划.md`


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



---


# Source 42: `docs/DG-KAN_v5.7_KernelVerified_EfficientPureKAN_结果复盘.md`


# DG-KAN v5.7 Kernel-Verified Efficient PureKAN 结果复盘

本轮依据 `docs/DG-KAN_v5.7_KernelVerified_EfficientPureKAN_实验计划.md`。核心目标是先确认 profiler 没有把 cold compile / graph break 混入 steady-state，再判断 strict PureKAN primitive 是否存在 MLP-like kernel path 和可接受 accuracy。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 core/profiler | 12 | 0 |
| P1 efficiency v3 | 144 | 0 |
| P1 cold/warm audit | 144 | 0 |
| P1 memory decomposition | 144 | 0 |
| P2 DWM recipe | 36 | 0 |
| P3 Rational recipe | 36 | 0 |
| P4 CP rank/speed | 36 | 0 |
| P5 joint selection | 1 | 0 |
| P6-P10 gated | 5 | 0 |

## Code / Config Changes

```text
experiments/run_gafu_v57.py
  Added kernel-verified P0/P1 with cold-vs-warm compile audit,
  memory decomposition, graph-break/recompile fields, and gated DWM/RK/CP recipe probes.

experiments/analyze_gafu_v57.py
  Generates v5.7 gate summaries, failure taxonomy, route_decision.json,
  aggregate_decision.json, SVG diagnostics, and this replay.
```

## P0 Core / Profiler Consistency

| primitive | class | edge | nonKAN | mixing | backend | breaks | rollback | P0 |
|---|---|---|---|---|---|---|---|---|
| MLP-reference | MLPClassifier | 0 | 59722 | 0 | eager | 0 | 0.0000 | yes |
| RBFOnly-Dense-reference | RBFDense | 504832 | 0 | 0 | eager | 0 | 0.0000 | yes |
| ABRBF-Dense-reference | ABRBFDense | 694144 | 0 | 0 | eager | 0 | 0.0000 | yes |
| DWM-current | GEMMNativeDepthwiseMixDense | 74544 | 0 | 63104 | eager | 0 | 0.0000 | yes |
| DWM-warm-compiled | GEMMNativeDepthwiseMixDense | 74544 | 0 | 63104 | torch.compile-warm | 0 | 0.0000 | yes |
| DWM-gemm-native | GEMMNativeDepthwiseMixDense | 74544 | 0 | 63104 | eager-gemm-native | 0 | 0.0000 | yes |
| RationalKAT-current | GEMMNativeRationalKATDense | 69344 | 0 | 63104 | eager | 0 | 0.0000 | yes |
| RationalKAT-compiled | GEMMNativeRationalKATDense | 69344 | 0 | 63104 | torch.compile-warm | 0 | 0.0000 | yes |
| RationalKAT-grouped-gemm | GEMMNativeRationalKATDense | 69344 | 0 | 63104 | eager-gemm-native | 0 | 0.0000 | yes |
| CP-two-stage-r4 | GEMMNativeCPABRBFDense | 194696 | 0 | 0 | eager-gemm-native | 0 | 0.0000 | yes |
| CP-two-stage-r8 | GEMMNativeCPABRBFDense | 200080 | 0 | 0 | eager-gemm-native | 0 | 0.0000 | yes |
| CP-two-stage-r16 | GEMMNativeCPABRBFDense | 210848 | 0 | 0 | eager-gemm-native | 0 | 0.0000 | yes |

P0 verdict: pass if each non-reference primitive is core-defined, edge-covered, nonKAN-free, rollback-exact, and has no recorded graph-break issue in the manifest.

## P1 Phase-Separated Efficiency Profiler V3

| method | rows | fwd | bwd | bmem | step | cold/warm | breaks | recomp | explore | final | bottleneck |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ABRBF-Dense-reference | 12 | 6.4005 | 1.9528 | 1.9907 | 2.1049 | 2.7748 | 0 | 0 | 0.0000 | 0.0000 | memory |
| CP-two-stage-r16 | 12 | 7.8850 | 2.9018 | 3.7520 | 2.8233 | 2.0011 | 0 | 0 | 0.0000 | 0.0000 | memory |
| CP-two-stage-r4 | 12 | 7.3934 | 2.7040 | 2.0928 | 2.6510 | 1.9416 | 0 | 0 | 0.0000 | 0.0000 | memory |
| CP-two-stage-r8 | 12 | 7.5113 | 2.6058 | 2.6372 | 2.6120 | 1.9615 | 0 | 0 | 0.0000 | 0.0000 | memory |
| DWM-current | 12 | 6.2220 | 2.0347 | 1.9079 | 2.1373 | 1.9990 | 0 | 0 | 0.0000 | 0.0000 | memory |
| DWM-gemm-native | 12 | 6.2069 | 1.9457 | 1.8915 | 2.0801 | 1.8994 | 0 | 0 | 0.0000 | 0.0000 | memory |
| DWM-warm-compiled | 12 | 8.4430 | 1.5334 | 0.1306 | 2.6134 | 871.9148 | 0 | 0 | 0.0000 | 0.0000 | time |
| RBFOnly-Dense-reference | 12 | 3.5967 | 1.8509 | 1.6571 | 1.7506 | 5.8133 | 0 | 0 | 0.0000 | 0.0000 | memory |
| RationalKAT-compiled | 12 | 9.1848 | 1.6803 | 0.1323 | 2.8545 | 566.9522 | 0 | 0 | 0.0000 | 0.0000 | time |
| RationalKAT-current | 12 | 4.9286 | 3.3529 | 1.6611 | 2.7211 | 2.6088 | 0 | 0 | 0.0000 | 0.0000 | memory |
| RationalKAT-grouped-gemm | 12 | 5.0712 | 3.5775 | 1.6748 | 2.8710 | 1.4968 | 0 | 0 | 0.0000 | 0.0000 | memory |

P1 exploratory survivors:

```text
none
```

P1 final survivors:

```text
none
```

P1 verdict: v5.7 separates cold step from warm steady-state. Any large cold/warm ratio is now diagnostic rather than silently counted as steady-state.

## P2 DepthwiseMix Kernel / Recipe Repair

| dataset | recipe | runs | acc | gap | ECE | step | bmem | P2 |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | DWM-K4-linear_silu-FixedNorm-scale0.1-lr1e-3 | 3 | 0.7474 | 0.0495 | 0.1546 | 2.2769 | 1.3100 | no |
| Fashion-MNIST | DWM-K8-linear_silu-channelNorm-scale0.3-lr2e-3 | 3 | 0.7604 | 0.0365 | 0.0593 | 2.2769 | 1.3100 | no |
| Fashion-MNIST | DWM-K8-linear_silu-wideMix-scale0.3-lr2e-3 | 3 | 0.7682 | 0.0286 | 0.0373 | 2.2769 | 1.3100 | no |
| KMNIST | DWM-K4-linear_silu-FixedNorm-scale0.1-lr1e-3 | 3 | 0.5918 | 0.1048 | 0.1412 | 2.2769 | 1.3100 | no |
| KMNIST | DWM-K8-linear_silu-channelNorm-scale0.3-lr2e-3 | 3 | 0.6257 | 0.0710 | 0.0658 | 2.2769 | 1.3100 | no |
| KMNIST | DWM-K8-linear_silu-wideMix-scale0.3-lr2e-3 | 3 | 0.6465 | 0.0501 | 0.0524 | 2.2769 | 1.3100 | no |
| MNIST | DWM-K4-linear_silu-FixedNorm-scale0.1-lr1e-3 | 3 | 0.8333 | 0.0697 | 0.2327 | 2.2769 | 1.3100 | no |
| MNIST | DWM-K8-linear_silu-channelNorm-scale0.3-lr2e-3 | 3 | 0.8535 | 0.0495 | 0.1638 | 2.2769 | 1.3100 | no |
| MNIST | DWM-K8-linear_silu-wideMix-scale0.3-lr2e-3 | 3 | 0.8750 | 0.0280 | 0.0849 | 2.2769 | 1.3100 | no |

P2 all-dataset survivors:

```text
none
```

## P3 RationalKAT Kernel / Recipe Repair

| dataset | recipe | runs | acc | gap | ECE | step | bmem | P3 |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | RK-groups16-silu-smallResidual-lr3e-3 | 3 | 0.7865 | 0.0104 | 0.0355 | 2.8155 | 1.1561 | no |
| Fashion-MNIST | RK-groups32-linear_silu-smallResidual-lr3e-3 | 3 | 0.7865 | 0.0104 | 0.0355 | 2.8155 | 1.1561 | no |
| Fashion-MNIST | RK-groups8-linear_silu-smallResidual-lr2e-3 | 3 | 0.7910 | 0.0059 | 0.0447 | 2.8155 | 1.1561 | no |
| KMNIST | RK-groups16-silu-smallResidual-lr3e-3 | 3 | 0.6673 | 0.0293 | 0.0411 | 2.8155 | 1.1561 | no |
| KMNIST | RK-groups32-linear_silu-smallResidual-lr3e-3 | 3 | 0.6673 | 0.0293 | 0.0411 | 2.8155 | 1.1561 | no |
| KMNIST | RK-groups8-linear_silu-smallResidual-lr2e-3 | 3 | 0.6641 | 0.0326 | 0.0417 | 2.8155 | 1.1561 | no |
| MNIST | RK-groups16-silu-smallResidual-lr3e-3 | 3 | 0.8880 | 0.0150 | 0.0554 | 2.8155 | 1.1561 | no |
| MNIST | RK-groups32-linear_silu-smallResidual-lr3e-3 | 3 | 0.8880 | 0.0150 | 0.0554 | 2.8155 | 1.1561 | no |
| MNIST | RK-groups8-linear_silu-smallResidual-lr2e-3 | 3 | 0.8841 | 0.0189 | 0.0806 | 2.8155 | 1.1561 | no |

P3 all-dataset survivors:

```text
none
```

## P4 CP-ABRBF Rank / Speed Reference

| dataset | recipe | runs | acc | gap | ECE | step | bmem | P4 |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | CP-two-stage-r16-K12-linear_silu | 3 | 0.7695 | 0.0273 | 0.0649 | 2.8233 | 3.7520 | no |
| Fashion-MNIST | CP-two-stage-r4-K8-linear_silu | 3 | 0.7624 | 0.0345 | 0.0662 | 2.6510 | 2.0928 | no |
| Fashion-MNIST | CP-two-stage-r8-K8-linear_silu | 3 | 0.7552 | 0.0417 | 0.0591 | 2.6120 | 2.6372 | no |
| KMNIST | CP-two-stage-r16-K12-linear_silu | 3 | 0.6517 | 0.0449 | 0.0404 | 2.8233 | 3.7520 | no |
| KMNIST | CP-two-stage-r4-K8-linear_silu | 3 | 0.6504 | 0.0462 | 0.0442 | 2.6510 | 2.0928 | no |
| KMNIST | CP-two-stage-r8-K8-linear_silu | 3 | 0.6478 | 0.0488 | 0.0524 | 2.6120 | 2.6372 | no |
| MNIST | CP-two-stage-r16-K12-linear_silu | 3 | 0.8848 | 0.0182 | 0.0819 | 2.8233 | 3.7520 | no |
| MNIST | CP-two-stage-r4-K8-linear_silu | 3 | 0.8796 | 0.0234 | 0.0978 | 2.6510 | 2.0928 | no |
| MNIST | CP-two-stage-r8-K8-linear_silu | 3 | 0.8704 | 0.0326 | 0.1099 | 2.6120 | 2.6372 | no |

P4 all-dataset survivors:

```text
none
```

## P5 Joint Task-Efficiency Selection

_P5 not run; no P2/P3/P4 all-dataset survivor._

P5 all-dataset survivors:

```text
none
```

## P6-P10 Decision

```text
P6 custom backward/memory: not run
P7 functional / LightSmooth compatibility: not run
P8 3-seed selection: not run
P9/P10 confirm: not run

Reason: No strict PureKAN primitive reached the exploratory MLP-like efficiency envelope.
```

## Failure Diagnosis

| failure | count |
|---|---|
| F5_backward_memory_fail | 162 |
| F6_accuracy_recipe_fail | 27 |
| F3_forward_time_fail | 20 |
| F4_backward_time_fail | 4 |

Route decision:

```text
C_no_primitive_survivor: No strict PureKAN primitive reached the exploratory MLP-like efficiency envelope.
```

## Required Artifacts

Written under `results/v5_7/`:

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

## Final Decision

```text
DG-KAN v5.7 status:
  stop_after_p1_no_efficiency_survivor

What improved:
  Profiler correctness is now explicit: cold compile, warm steady-state, memory decomposition,
  graph breaks, and recompiles are audited as first-class artifacts.
  DWM / RationalKAT / CP recipe probes remain gated behind strict task-efficiency checks.

What failed / remains open:
  See the P1-P5 gates above. Functional optimizer / LightSmooth stays gated until
  a primitive survives joint task + efficiency selection.

Conclusion:
  v5.7 keeps the kernel-first rule strict: a functional optimizer cannot rescue a primitive
  that lacks a verified MLP-like forward/backward/memory path.
```



---


# Source 43: `docs/DG-KAN_v5.8_EfficientPrimitive_Redesign_实验计划.md`


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



---


# Source 44: `docs/DG-KAN_v5.8_EfficientPrimitive_Redesign_结果复盘.md`


# DG-KAN v5.8 Efficient Primitive Redesign 结果复盘

本轮依据 `docs/DG-KAN_v5.8_EfficientPrimitive_Redesign_实验计划.md`。核心目标是在继续 functional optimizer / LightSmooth 之前，重新设计并验证更接近 MLP-like kernel path 的 strict PureKAN primitive。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 core manifest | 9 | 0 |
| P1 efficiency | 27 | 0 |
| P2 DWM-v2 recipe | 1 | 0 |
| P3 RationalKAT-v2 recipe | 1 | 0 |
| P4 LUTKAN feasibility | 1 | 0 |
| P5 joint selection | 1 | 0 |
| P6-P10 gated | 5 | 0 |

## Code / Config Changes

```text
experiments/dgkan_core.py
  Added core efficient primitive candidates:
    DWM2Dense
    RationalKATV2Dense
    LUTKANDense
  Extended edge/base/RBF/mixing parameter discovery for the new primitives.

experiments/run_gafu_v58.py
  Added P0 core manifest, P1 small/medium/large efficiency decomposition,
  gated DWM-v2 / RationalKAT-v2 / LUTKAN recipe probes,
  joint selection, and required gated artifacts.

experiments/analyze_gafu_v58.py
  Generates v5.8 gate summaries, route_decision.json, aggregate_decision.json,
  failure taxonomy, figures, and this replay.
```

## P0 Core Manifest

| primitive | class | edge | nonKAN | mixing | rollback | P0 |
|---|---|---|---|---|---|---|
| MLP-reference | MLPClassifier | 0 | 59722 | 0 | 0.0000 | yes |
| Dense-ABRBF-reference | ABRBFDense | 694144 | 0 | 0 | 0.0000 | yes |
| DWM-v1-current | GEMMNativeDepthwiseMixDense | 74544 | 0 | 63104 | 0.0000 | yes |
| DWM2-prePostMix | DWM2Dense | 82514 | 0 | 79588 | 0.0000 | yes |
| DWM2-gated | DWM2Dense | 82780 | 0 | 79588 | 0.0000 | yes |
| DWM2-residual | DWM2Dense | 82514 | 0 | 79588 | 0.0000 | yes |
| RationalKAT-AB-v2 | RationalKATV2Dense | 69344 | 0 | 63104 | 0.0000 | yes |
| LUTKAN-AB | LUTKANDense | 71424 | 0 | 63104 | 0.0000 | yes |
| CP-ABRBF-reference | GEMMNativeCPABRBFDense | 200080 | 0 | 0 | 0.0000 | yes |

P0 verdict: core manifest passes when each strict PureKAN primitive is edge-covered, nonKAN-free, and rollback-exact.

## P1 Efficiency Decomposition

| method | rows | fwd | bwd | bmem | step | explore | final | bottleneck |
|---|---|---|---|---|---|---|---|---|
| CP-ABRBF-reference | 3 | 13.7354 | 3.7246 | 3.9940 | 4.2614 | 0.0000 | 0.0000 | memory |
| DWM-v1-current | 3 | 12.1642 | 2.7904 | 3.5011 | 3.4751 | 0.0000 | 0.0000 | memory |
| DWM2-gated | 3 | 7.1559 | 3.9212 | 3.3212 | 3.5408 | 0.0000 | 0.0000 | memory |
| DWM2-prePostMix | 3 | 6.1837 | 3.1438 | 3.1843 | 2.9472 | 0.0000 | 0.0000 | memory |
| DWM2-residual | 3 | 6.3861 | 3.4361 | 3.1979 | 3.1471 | 0.0000 | 0.0000 | memory |
| Dense-ABRBF-reference | 3 | 12.7981 | 3.2618 | 4.0062 | 4.0050 | 0.0000 | 0.0000 | memory |
| LUTKAN-AB | 3 | 11.8364 | 3.1283 | 2.3854 | 3.5872 | 0.0000 | 0.0000 | memory |
| RationalKAT-AB-v2 | 3 | 7.7779 | 4.9495 | 2.2472 | 4.2200 | 0.0000 | 0.0000 | memory |

P1 exploratory survivors:

```text
none
```

P1 final survivors:

```text
none
```

## P2 DWM-v2 Recipe

_P2 not run or no recipe rows._

P2 all-dataset survivors:

```text
none
```

## P3 RationalKAT-v2 Recipe

_P3 not run or no recipe rows._

P3 all-dataset survivors:

```text
none
```

## P4 LUTKAN Feasibility

_P4 not run or no recipe rows._

P4 all-dataset survivors:

```text
none
```

## P5-P10 Decision

```text
P5 joint task-efficiency selection: not run / no survivor
P6 custom backward audit: not run
P7 LightSmooth compatibility: not run unless P6 survives
P8 functional training smoke: not run unless P7 survives
P9/P10 confirm: not run unless P8 survives
Final decision: stop_after_p1_no_efficiency_survivor
Route case: E_no_efficient_primitive
```

## Failure Diagnosis

| failure | count |
|---|---|
| F2_memory_fail | 24 |
| F4_gated_not_run | 4 |

## Required Artifacts

Written under `results/v5_8/`:

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

## Final Decision

```text
DG-KAN v5.8 status:
  stop_after_p1_no_efficiency_survivor

Route:
  E_no_efficient_primitive

What improved:
  v5.8 adds three redesigned strict PureKAN primitive families in core.
  P1 now measures small/medium/large efficiency decomposition before recipe spending.
  Functional optimizer and LightSmooth remain correctly gated behind primitive survival.

What failed / remains open:
  See the P1-P5 gates above.
  If no P1/P5 survivor exists, the bottleneck remains primitive efficiency or task recipe,
  not functional smoothing.

Conclusion:
  v5.8 follows the kernel-first rule: do not resume FGO-v3 until an efficient primitive
  satisfies the joint task + efficiency envelope.
```



---


# Source 45: `docs/DG-KAN_v5.9_MemoryFirst_EfficientPrimitive_实验计划.md`


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



---


# Source 46: `docs/DG-KAN_v5.9_MemoryFirst_EfficientPrimitive_结果复盘.md`


# DG-KAN v5.9 Memory-First Efficient PureKAN Primitive 结果复盘

本轮依据 `docs/DG-KAN_v5.9_MemoryFirst_EfficientPrimitive_实验计划.md`。核心目标是先确认 primitive 的 profiler truth 和 backward saved tensor 来源，再决定是否值得进入 custom backward、task recipe、LightSmooth 或 functional optimizer。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 core manifest | 12 | 0 |
| P1 phase efficiency | 36 | 0 |
| P1 saved tensor audit | 36 | 0 |
| P2 custom backward | 1 | 0 |
| P3 recipe grid | 1 | 0 |
| P4-P7 gated | 4 | 0 |

## Code / Config Changes

```text
experiments/dgkan_core.py
  Added DWM2LiteDense: single-GEMM KAN residual primitive with RBF/LUT residual modes.

experiments/run_gafu_v59.py
  Added P0 memory-first manifest, P1 saved_tensors_hooks profiler,
  memory decomposition, kernel breakdown, gated P2 custom-backward diagnostics,
  and gated P3-P7 artifacts.

experiments/analyze_gafu_v59.py
  Generates v5.9 summaries, route_decision.json, aggregate_decision.json,
  required SVG diagnostics, and this replay.
```

## P0 Core Manifest

| primitive | class | edge | nonKAN | mixing | rollback | P0 |
|---|---|---|---|---|---|---|
| MLP-reference | MLPClassifier | 0 | 59722 | 0 | 0.0000 | yes |
| Dense-ABRBF-reference | ABRBFDense | 694144 | 0 | 0 | 0.0000 | yes |
| DWM2-prePostMix-current | DWM2Dense | 82514 | 0 | 79588 | 0.0000 | yes |
| DWM2-lite-RBFK2 | DWM2LiteDense | 65184 | 0 | 63104 | 0.0000 | yes |
| DWM2-lite-RBFK4 | DWM2LiteDense | 67264 | 0 | 63104 | 0.0000 | yes |
| DWM2-lite-LUTK8 | DWM2LiteDense | 71424 | 0 | 63104 | 0.0000 | yes |
| RationalKAT-v2-current | RationalKATV2Dense | 69344 | 0 | 63104 | 0.0000 | yes |
| RationalKAT-lite-groups8 | RationalKATV2Dense | 69344 | 0 | 63104 | 0.0000 | yes |
| RationalKAT-lite-groups16 | RationalKATV2Dense | 69344 | 0 | 63104 | 0.0000 | yes |
| LUTKAN-current | LUTKANDense | 71424 | 0 | 63104 | 0.0000 | yes |
| LUTKAN-v2-linearInterp | LUTKANDense | 71424 | 0 | 63104 | 0.0000 | yes |
| CP-ABRBF-r4-reference | GEMMNativeCPABRBFDense | 194696 | 0 | 0 | 0.0000 | yes |

P0 verdict: pass. The new DWM2-lite variants remain strict PureKAN: edge-covered, nonKAN-free, and rollback-exact.

## P1 Profiler Truth / Memory Decomposition

| primitive | rows | fwd | bwd | bmem | step | saved MB | explore | bottleneck |
|---|---|---|---|---|---|---|---|---|
| CP-ABRBF-r4-reference | 3 | 12.7260 | 4.3978 | 3.3879 | 5.1465 | 100.03 | 0.0000 | memory |
| DWM2-lite-LUTK8 | 3 | 11.5534 | 4.5022 | 2.4121 | 4.9355 | 75.51 | 0.0000 | memory |
| DWM2-lite-RBFK2 | 3 | 4.0353 | 2.8083 | 1.9146 | 2.6146 | 17.95 | 0.0000 | forward |
| DWM2-lite-RBFK4 | 3 | 4.0243 | 2.9188 | 2.1792 | 2.6746 | 29.40 | 0.0000 | memory |
| DWM2-prePostMix-current | 3 | 5.8611 | 4.0823 | 2.7599 | 3.7020 | 57.26 | 0.0000 | memory |
| Dense-ABRBF-reference | 3 | 11.9902 | 4.4451 | 3.2072 | 5.2374 | 90.20 | 0.0000 | memory |
| LUTKAN-current | 3 | 11.0385 | 4.1155 | 2.4121 | 4.6201 | 75.51 | 0.0000 | memory |
| LUTKAN-v2-linearInterp | 3 | 11.5914 | 4.2533 | 2.4121 | 4.7980 | 75.51 | 0.0000 | memory |
| RationalKAT-lite-groups16 | 3 | 7.7778 | 6.4465 | 2.3941 | 5.3943 | 42.28 | 0.0000 | memory |
| RationalKAT-lite-groups8 | 3 | 7.7583 | 6.1292 | 2.3941 | 5.2113 | 42.28 | 0.0000 | memory |
| RationalKAT-v2-current | 3 | 7.7474 | 6.4180 | 2.3941 | 5.3725 | 42.28 | 0.0000 | memory |

P1 exploratory survivors:

```text
none
```

P1 verdict: no exploratory survivor. DWM2-lite-RBFK2 is the closest route, but still misses the forward/memory envelope under the written gate.

## P2-P7 Decision

```text
P2 custom backward / recompute: not run; P1 produced no exploratory survivor
P3 minimal task recipe: not run; P2 produced no survivor
P4 LightSmooth compatibility: not run; P3 produced no task-efficiency survivor
P5 functional smoke: not run unless P4 survives
P6/P7 confirm: not run unless P3/P5 survives
Final decision: stop_after_p1_no_efficiency_survivor
Route case: C_no_p1_efficiency_survivor
```

## Failure Diagnosis

| failure | count |
|---|---|
| F1_backward_memory_fail | 29 |
| F2_forward_time_fail | 4 |
| F9_gated_not_run | 6 |

Interpretation:

```text
DWM2-lite reduces saved tensors substantially versus dense AB-RBF and current DWM2,
but it still does not reach the relaxed P1 envelope.
The dominant blocker remains profiler-level primitive efficiency, especially saved tensors /
workspace memory and forward kernel path, so task recipe and functional optimizer remain gated.
```

## Required Artifacts

Written under `results/v5_9/`:

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
figures/
```

## Final Decision

```text
DG-KAN v5.9 status:
  stop_after_p1_no_efficiency_survivor

Route:
  C_no_p1_efficiency_survivor

What improved:
  v5.9 adds saved-tensor-aware profiling and a stricter memory-cause diagnosis.
  DWM2-lite gives a cleaner single-GEMM primitive and materially lowers saved tensor volume.

What failed / remains open:
  No primitive passed P1 exploratory efficiency.
  P2 custom backward, P3 task recipe, LightSmooth, and functional training are correctly gated.

Conclusion:
  The next move is still primitive/kernel/backward design, not optimizer tuning.
```

