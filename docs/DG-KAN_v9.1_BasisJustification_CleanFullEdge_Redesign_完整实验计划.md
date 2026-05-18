# DG-KAN v9.1 Basis Justification and Clean FullEdge Redesign 完整实验计划

> 本计划针对当前最关键的新问题：**FullEdge PureKAN 的基函数选取缺乏 justification**。  
> v9.0 已经完成 CleanCE transitional route，并且 FullEdge manual implementation、gradcheck、functional causality、robust timing 都通过；但 FullEdge external task fair 与 training compute fair 都失败。  
> 这意味着我们不能简单说“FullEdge PureKAN 不行”，更准确地说是：**当前 FullEdge basis / parameterization / compute design 没有被充分论证，也没有证明它是适合 vision classification 与 external fair comparison 的基函数族。**  
> v9.1 的目标不是继续盲试 `poly2_silu6 / compact_poly_silu3 / rbf4 / spline4`，而是建立一个可审计的 **basis justification protocol**，让每一种基函数进入 full task 前都有理论、诊断、优化、几何和系统层面的理由。

---

## 0. 当前问题的独立判断

### 0.1 v9.0 已经证明了什么

v9.0 的真实 route 是：

```text
route = R3-CleanTransitionalOnlySuccess
success_v90_clean_ce = true
success_v90_contract_hardening = true
success_v90_core_skeleton = true
success_v90_full_edge = false
success_v90_external_fair = false
success_v90_strong = false
```

这说明：

```text
CleanCE transitional route 成立；
FullEdge implementation / GradPass 成立；
FullEdge functional causality controls 成立；
FullEdge robust timing 成立；
但 FullEdge external task fair 失败；
FullEdge training compute fair 失败。
```

v9.0 FullEdge candidates 包括：

```text
DG-FullEdge-Poly2Silu6-h5
DG-FullEdge-CompactPolySilu3-h11
DG-FullEdge-CompactPolySilu3-h11x2
DG-FullEdge-RBF4-h8
DG-FullEdge-Spline4-h8
```

三任务 best FullEdge functional candidate 均低于 KB-MLP：

```text
MNIST:
  best = CompactPolySilu3-h11x2+Functional
  acc = 94.949996
  delta vs KB-MLP = -2.039999
  FLOPs ratio = 1.550366
  memory ratio = 1.864129
  curvature ratio = 0.522096

Fashion-MNIST:
  best = CompactPolySilu3-h11+Functional
  acc = 85.670000
  delta vs KB-MLP = -1.309997
  FLOPs ratio = 1.529181
  memory ratio = 1.856530
  curvature ratio = 0.386515

KMNIST:
  best = CompactPolySilu3-h11x2+Functional
  acc = 78.679997
  delta vs KB-MLP = -4.510003
  FLOPs ratio = 1.550366
  memory ratio = 1.862592
  curvature ratio = 0.480719
```

这个结果说明 functional update 确实能压低 curvature，但 task accuracy 和 compute fairness 同时失败。

### 0.2 不能直接得出的结论

不能直接说：

```text
FullEdge PureKAN 本质上不如 MLP；
functional update 没用；
KAN edge function 没表达力；
应该退回 transitional linear-SiLU stack。
```

因为当前 FullEdge 的基函数选择没有经过系统 justification。`poly2_silu6`、`compact_poly_silu3`、`rbf4`、`spline4` 只是候选集合，不是经过诊断筛选后的最优 basis family。

当前更准确的结论是：

$$
\boxed{
\text{v9.0 失败的是当前 FullEdge basis family 与 parameterization，不是 FullEdge PureKAN 终极路线本身。}
}
$$

### 0.3 为什么“基函数 justification”是核心问题

FullEdge layer 的形式是：

$$
y_j=\sum_i \phi_{ij}(x_i).
$$

所有表达力都压在 $\phi_{ij}$ 的基函数族上。如果 $\phi_{ij}$ 的 basis 选错，就会同时带来三个问题：

```text
1. task expressivity 不够；
2. optimization conditioning 变差；
3. compute / memory materialization 太重。
```

因此，v9.1 必须先回答：

$$
\boxed{
\text{什么基函数适合当前任务、当前公平预算、当前 functional update？}
}
$$

而不是继续把各种 basis 当成并列 sweep 项。

---

## 1. v9.1 总体目标

v9.1 的总体目标是：

$$
\boxed{
\text{建立一个有 justification 的 FullEdge basis selection protocol，并用它重新设计 Clean FullEdge PureKAN。}
}
$$

最终要证明或证伪：

```text
1. 哪种 basis family 适合 vision classification；
2. 哪种 basis family 适合 symbolic / smooth function representation；
3. 哪种 basis family 与 functional update 的 geometry objective 匹配；
4. 哪种 basis family 能在 parameter / FLOPs / wall-clock fair envelope 内运行；
5. 当前 FullEdge 失败是 basis 选择问题、parameterization 问题、optimization 问题，还是 FullEdge 形式本身的问题。
```

v9.1 的 minimum success 是：

$$
\text{BasisJustificationPass}
\land
\text{CleanFullEdgeContractPass}
\land
\text{GradPass}
\land
\text{ExternalFairTaskPass}
\land
\text{FunctionalCausalityPass}.
$$

v9.1 的 strong success 是：

$$
\text{MinimumSuccess}
+
\text{TrainingComputeFairPass}
+
\text{RobustnessOrSymbolicPass}
+
\text{MechanismAttributionPass}.
$$

---

## 2. 本轮硬约束

v9.1 继续禁止：

```text
external teacher
self teacher
teacher logits
distillation
self-distillation
label smoothing
focal loss
margin loss
calibration loss
NLL-balanced loss
geometry loss as training objective
sampler / class weight
oversampling / undersampling
test-based selection
CPU offload
optimizer hyperparameter sweep
```

训练目标必须是：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

Functional update 只能是 update rule：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{\text{AdamW-equivalent}}
+
\Delta\theta_{\text{functional}}.
$$

不允许：

$$
L=CE+\lambda L_{\text{geo}}.
$$

所有 official candidates 必须记录：

```text
loss_type = CE
label_smoothing = 0
geometry_loss_used = 0
external_teacher_used = 0
self_teacher_used = 0
teacher_logits_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
uses_loss_backward = 0
fake_data_used = 0
proxy_row_used = 0
```

---

## 3. Basis justification 的五个维度

每个 basis family 进入 full task 之前，必须通过五维审计。

### 3.1 Approximation justification

基函数必须说明它能表示什么函数类。

#### Polynomial / compact polynomial basis

适合低阶局部非线性、便宜、容易做 manual backward。当前 `compact_poly_silu3` 属于这一类：

$$
\phi(x)=a_1x+a_2\operatorname{SiLU}(x)+a_3x^2.
$$

问题是 monomial $x^2$ 可能 conditioning 不好，输入分布稍变就造成尺度不稳。因此 v9.1 不再只用 monomial，而要引入 normalized orthogonal polynomial：

$$
\phi(x)=a_0+a_1P_1(\tilde{x})+a_2P_2(\tilde{x})+a_3\operatorname{SiLU}(\tilde{x}),
$$

其中：

$$
\tilde{x}=\frac{x-\mu}{\sigma+\epsilon}.
$$

可选 $P_k$：

```text
Legendre P1/P2/P3
Chebyshev T1/T2/T3
Hermite H1/H2/H3
```

#### RBF basis

适合局部平滑函数，天然有局部支持倾向：

$$
\phi(x)=\sum_{k=1}^{K}c_k\exp\left(-\frac{(x-\mu_k)^2}{2\sigma^2}\right).
$$

问题是 dense full-edge RBF 会构造 $B\times d_{in}\times d_{out}\times K$ 级别的 basis，compute 很重。因此 RBF 只能以 shared-basis / grouped / low-rank 方式进入 official。

#### Spline / piecewise basis

适合 KAN 原始动机和 symbolic function representation。KANbeFair 论文认为 KAN 在 symbolic formula representation 上的优势主要来自 B-spline activation，因此 spline 是唯一有强外部 justification 的 basis family。Spline 的问题是 vision classification 未必受益，而且 knot / binning / active interval 的 kernelization 复杂。

Piecewise linear / cubic basis：

$$
\phi(x)=\sum_k c_kB_k(x).
$$

其中 $B_k$ 是 local support basis。

#### Rational / DWM-like basis

适合高曲率、分段变化、具有分母调制的函数：

$$
\phi(x)=\frac{p(x)}{1+|q(x)|}.
$$

优点是表达力强，缺点是梯度和稳定性较难控制。只作为 diagnostic，不直接进入 official，除非先通过 stability audit。

### 3.2 Optimization justification

每个 basis 必须记录 conditioning：

```text
basis activation mean/std
basis covariance condition number
gradient norm by basis channel
dead basis fraction
dominant basis fraction
grad cosine vs CE task direction
```

如果某个 basis 的 condition number 太高：

$$
\kappa(B^\top B)>10^4,
$$

或 dead basis fraction 超过：

$$
0.30,
$$

则不能进入 full task official candidate。

### 3.3 Functional update compatibility

Functional update 的几何目标通常是 curvature / slope / Lipschitz。Basis 必须能支持稳定的 geometry direction。

几何指标包括：

$$
R_{\text{coef-curv}}
=
\sum_k(c_{k+2}-2c_{k+1}+c_k)^2.
$$

函数有限差分曲率：

$$
R_{\text{func-curv}}
=
\mathbb{E}_{x,\epsilon}
\left[
\frac{\phi(x+\epsilon)-2\phi(x)+\phi(x-\epsilon)}{\epsilon^2}
\right]^2.
$$

Jacobian proxy：

$$
J_{\text{norm}}=
\mathbb{E}_x\|\nabla_x f_\theta(x)\|^2.
$$

Basis compatibility pass：

$$
R_{\text{curv,functional}}\leq0.90R_{\text{curv,base}},
$$

and:

$$
Acc_{\text{functional}}\geq Acc_{\text{base}}-0.005.
$$

### 3.4 System justification

Basis 必须有 materialization-free 或 bounded-materialization path。

每个 basis 必须记录：

```text
forward FLOPs
backward FLOPs estimate
basis materialized tensor shape
peak memory
step time
kernel count
basis eval time
basis backward time
functional update time
```

System pass：

$$
ForwardFLOPsRatio\leq1.05,
$$

$$
BackwardFLOPsRatio\leq1.50,
$$

$$
StepRatio\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

如果 basis 必须 materialize：

$$
B\times d_{in}\times d_{out}\times K,
$$

且 memory ratio 超过 1.05，则只能是 diagnostic。

### 3.5 Task-family justification

不同 task family 允许不同 basis priority，但必须预注册，不能事后挑。

```text
Vision MNIST-family:
  priority = normalized compact polynomial / shared-basis polynomial / low-rank edge basis

Symbolic:
  priority = spline / B-spline / RBF

Tabular:
  priority = spline / monotone piecewise / RBF diagnostic

NLP/audio:
  priority = compact polynomial / low-rank shared-basis, unless input representation suggests locality
```

---

## 4. 核心假设

### H1：v9.0 FullEdge 失败不是 FullEdge 形式失败，而是 basis parameterization 未 justified

H1 成立标准：

至少一个 justified basis 在同样 clean contract 下满足：

$$
Acc_{\text{FullEdge+Functional}}\geq Acc_{\text{KB-MLP}}-0.005
$$

并且比 v9.0 best FullEdge 提升：

$$
Acc_{\text{new}}-Acc_{\text{v90-best}}\geq0.010.
$$

若所有 justified basis 仍低于 MLP 超过 1%，则 FullEdge 形式本身可能不适合当前 vision tasks。

### H2：normalized orthogonal polynomial 比 monomial compact poly 更稳定

H2 比较：

```text
monomial x, x^2, x^3
Legendre P1/P2/P3
Chebyshev T1/T2/T3
Hermite H1/H2/H3
```

成立标准：

$$
\kappa_{\text{orthogonal}}\leq0.5\kappa_{\text{monomial}},
$$

$$
DeadBasisFrac_{\text{orthogonal}}\leq DeadBasisFrac_{\text{monomial}},
$$

and:

$$
Acc_{\text{orthogonal}}\geq Acc_{\text{monomial}}.
$$

### H3：B-spline / piecewise basis 应优先用于 symbolic，不应默认用于 vision

H3 成立标准：

Symbolic tasks 上：

$$
RMSE_{\text{spline}}\leq RMSE_{\text{MLP}},
$$

or:

$$
RMSE_{\text{spline}}\leq RMSE_{\text{poly}}.
$$

Vision tasks 上如果：

$$
Acc_{\text{spline}}<Acc_{\text{poly}}-0.005,
$$

则 spline 不作为 vision primary basis，只保留 symbolic route。

### H4：shared-basis / low-rank coefficient factorization 是 FullEdge compute fair 的必要条件

Dense edge coefficient：

$$
C\in\mathbb{R}^{d_{in}\times d_{out}\times K}
$$

太重。v9.1 引入：

$$
C_{i,j,k}
=
\sum_{r=1}^{R}u_{i,r}v_{j,r}w_{k,r}.
$$

或 shared basis：

$$
\phi_{ij}(x)=\sum_k c_{ij,k}B_k(x),
$$

其中 $B_k(x)$ 只按 input channel materialize：

$$
B\in\mathbb{R}^{B\times d_{in}\times K},
$$

然后用 matrix contraction 得到 output。

H4 成立标准：

$$
MemoryRatio_{\text{shared/lowrank}}\leq1.05,
$$

$$
StepRatio_{\text{shared/lowrank}}\leq1.50,
$$

and:

$$
Acc_{\text{shared/lowrank}}\geq Acc_{\text{dense-full-edge}}-0.005.
$$

### H5：Functional update 应作用在 residual / high-curvature subspace，而不是所有 basis 参数

H5 假设对所有 coefficient 做 smoothing 会伤 task；应只对 residual or high curvature channel 做 functional correction。

成立标准：

Residual-subspace functional update 满足：

$$
Acc_{\text{residual-func}}\geq Acc_{\text{allparam-func}},
$$

and:

$$
R_{\text{curv,residual-func}}\leq R_{\text{base}}\cdot0.90.
$$

---

## 5. Candidate 设计

### 5.1 Basis families

```text
BAS0-MonomialCompactPoly3
BAS1-NormalizedLegendre3
BAS2-NormalizedChebyshev3
BAS3-NormalizedHermite3
BAS4-SharedRBF4
BAS5-SharedRBF8
BAS6-PiecewiseLinear4
BAS7-CubicBSpline4
BAS8-CubicBSpline8
BAS9-RationalPade2
```

### 5.2 Parameterization variants

```text
P0-DenseFullEdge
P1-SharedBasisFullEdge
P2-LowRankCP-r2
P3-LowRankCP-r4
P4-GroupedSharedBasis-g4
P5-GroupedSharedBasis-g8
P6-ResidualEdgeOwned
P7-FanInNormalized
```

### 5.3 Functional update variants

```text
F0-NoFunctional
F1-AllParamSecondDiff
F2-ResidualSubspaceSecondDiff
F3-HighCurvatureOnlyFunctional
F4-RoleBudgetedFunctional
F5-NoOpMatchedOverhead
F6-RandomFunctionalDirection
```

### 5.4 Official candidate naming

Example:

```text
FE-Legendre3-SharedBasis-FanNorm-h32-F2
FE-BSpline4-SharedBasis-FanNorm-h32-F2
FE-RBF4-LowRankCP-r4-FanNorm-h32-F3
```

Candidate ID 必须包含：

```text
basis family
parameterization
normalization
hidden
functional mode
```

---

## 6. 实验阶段总览

v9.1 分为八个 Wave：

```text
Wave 0:
  Basis taxonomy and implementation audit

Wave 1:
  One-layer basis diagnostics

Wave 2:
  FullEdge small-task smoke and gradcheck

Wave 3:
  Basis x parameterization matrix on MNIST/FMNIST/KMNIST

Wave 4:
  Functional subspace compatibility

Wave 5:
  Materialization-free compute repair

Wave 6:
  External fair validation against KB-MLP / KB-KAN

Wave 7:
  Symbolic / robustness / boundary route
```

---

## 7. Wave 0：Basis taxonomy and implementation audit

### P0：basis registry audit

#### 目标

所有 basis 必须有公式、参数量、FLOPs、support、geometry compatibility 说明。

#### 必须记录

```text
basis_id
basis_name
formula
support_type
local_support
learnable_centers
learnable_widths
parameter_count_per_edge
forward_flops_per_edge
backward_flops_per_edge
materialized_shape
geometry_metric_supported
functional_direction_supported
official_eligible
```

#### 判断标准

BasisJustificationPass：

```text
formula recorded
parameter count recorded
FLOPs recorded
geometry metric supported
manual backward implemented or explicitly diagnostic-only
```

#### 可视化

```text
p0_basis_taxonomy_table.md
p0_basis_flops_params_bar.svg
p0_basis_support_diagram.svg
```

---

## 8. Wave 1：One-layer basis diagnostics

### P1：basis activation and conditioning audit

#### 目标

在真实 data distribution 上检查 basis 的数值稳定性。

#### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
input source = raw flattened input and transitional hidden features
batch size = 512
seeds = 0,1,2
```

#### 必须记录

```text
basis_id
dataset
input_source
activation_mean
activation_std
activation_p01
activation_p99
basis_cov_condition
dead_basis_fraction
dominant_basis_fraction
grad_norm_mean
grad_norm_cv
slope_p95
curvature_p95
```

#### 判断标准

ConditioningPass：

$$
\kappa(B^\top B)\leq10^4.
$$

$$
DeadBasisFraction\leq0.30.
$$

$$
DominantBasisFraction\leq0.70.
$$

#### 可视化

```text
p1_basis_activation_histograms.svg
p1_basis_condition_bar.svg
p1_dead_dominant_basis_heatmap.svg
p1_slope_curvature_boxplot.svg
```

### P2：one-layer target fit diagnostic

#### 目标

判断 basis 的基本拟合能力，不进入 official success。

#### 诊断 targets

```text
identity
silu
quadratic
piecewise ramp
local bump
sinusoidal
symbolic polynomial
random projection target from transitional hidden
```

#### 必须记录

```text
basis_id
target_type
fit_MSE
fit_R2
coeff_norm
curvature
condition_number
train_time
```

#### 判断标准

FitPass diagnostic：

$$
R^2\geq0.95
$$

for simple targets identity / quadratic / silu。

RBF / spline 必须在 local bump 上优于 compact poly：

$$
MSE_{\text{RBF/spline}}<MSE_{\text{poly}}.
$$

#### 可视化

```text
p2_fit_r2_by_basis.svg
p2_fit_curves_examples.svg
p2_coeff_norm_vs_fit.svg
```

---

## 9. Wave 2：FullEdge smoke and gradcheck

### P3：manual gradcheck

#### 目标

所有 official basis 必须通过 manual backward。

#### 必须记录

```text
basis_id
parameterization
GradRelErrMax
GradCosMin
OutputAbsDiffMax
DxAbsDiffMax
ParamGradAbsDiffMax
GradPass
```

#### 判断标准

$$
GradRelErrMax\leq10^{-4}.
$$

$$
GradCosMin\geq0.999.
$$

### P4：small-task overfit smoke

#### 目标

确认 basis 能拟合小 batch，不是训练完全不可达。

#### 设置

```text
dataset = MNIST
train subset = 512
steps = 300
no functional update
```

#### 判断标准

OverfitPass：

$$
TrainAcc\geq0.98.
$$

If a basis cannot overfit 512 samples, do not run full external validation.

#### 可视化

```text
p4_small_overfit_curve.svg
p4_train_acc_final_bar.svg
```

---

## 10. Wave 3：Basis x parameterization matrix

### P5：full task matrix

#### 目标

比较 basis 和 parameterization，找出合理 FullEdge candidate。

#### Matrix

```text
Basis:
  BAS0-BAS8

Parameterization:
  DenseFullEdge
  SharedBasis
  LowRankCP-r2/r4
  GroupedSharedBasis-g4/g8
  FanInNormalized
  ResidualEdgeOwned
```

#### 必须记录

```text
candidate_id
basis_id
parameterization
dataset
seed
test_acc
delta_vs_KB_MLP
params_ratio
forward_flops_ratio
backward_flops_ratio
step_ratio
memory_ratio
curvature
ECE
NLL
GradPass
```

#### 判断标准

PromotionPass：

$$
Acc_{\text{candidate}}\geq Acc_{\text{KB-MLP}}-0.005.
$$

$$
ParamsRatio\leq1.05.
$$

$$
StepRatio\leq1.50.
$$

$$
MemoryRatio\leq1.05.
$$

如果没有任何 basis 达到 PromotionPass，说明 FullEdge 参数化仍不成熟，不能继续只调 functional update。

#### 可视化

```text
p5_basis_param_task_heatmap.svg
p5_task_system_pareto.svg
p5_basis_family_win_loss.svg
p5_failure_reason_stacked.svg
```

---

## 11. Wave 4：Functional subspace compatibility

### P6：functional causality by basis

#### 目标

验证每种 basis 的 functional update 是否有意义。

#### 必跑

```text
Base
Functional
NoOp
RandomFunc
ShuffledRole
AllParamFunctional
ResidualSubspaceFunctional
HighCurvatureOnlyFunctional
```

#### 必须记录

```text
basis_id
functional_mode
test_acc
delta_vs_base
curvature_ratio
ECE_delta
NLL_delta
bad_step_rate
holdout_descent_ratio
functional_update_time
```

#### 判断标准

CausalityPass：

$$
R_{\text{curv,Functional}}<R_{\text{curv,NoOp}}.
$$

$$
R_{\text{curv,Functional}}<R_{\text{curv,Random}}.
$$

$$
Acc_{\text{Functional}}\geq Acc_{\text{NoOp}}-0.002.
$$

FunctionalUsefulPass：

$$
R_{\text{curv,Functional}}\leq0.90R_{\text{base}}.
$$

$$
Acc_{\text{Functional}}\geq Acc_{\text{base}}-0.005.
$$

#### 可视化

```text
p6_functional_curvature_by_basis.svg
p6_functional_task_geometry_pareto.svg
p6_noop_random_control_matrix.svg
```

---

## 12. Wave 5：Materialization-free compute repair

### P7：shared basis / low-rank compute audit

#### 目标

解决 v9.0 FullEdge FLOPs / memory failure。

#### 必须记录

```text
candidate_id
materialized_tensor_shape
materialized_MB
forward_flops_ratio
backward_flops_ratio
kernel_time_ratio
step_ratio
memory_ratio
```

#### 判断标准

ComputeRepairPass：

$$
ForwardFLOPsRatio\leq1.05.
$$

$$
BackwardFLOPsRatio\leq1.50.
$$

$$
StepRatio\leq1.50.
$$

$$
MemoryRatio\leq1.05.
$$

#### 可视化

```text
p7_materialization_memory_bar.svg
p7_compute_repair_pareto.svg
p7_kernel_time_by_phase.svg
```

---

## 13. Wave 6：External fair validation

### P8：KANbeFair external fair comparison

#### 目标

重新与 KB-MLP / KB-KAN 比较。

#### 必跑

```text
KB-MLP
KB-KAN
CleanTransitional
BestFullEdgeBase
BestFullEdgeFunctional
NoOp
RandomFunc
```

#### Tasks

```text
MNIST
Fashion-MNIST
KMNIST
symbolic tasks if available
tabular tasks if runnable
```

#### 判断标准

FullEdgeExternalFairPass：

$$
Acc_{\text{FullEdgeFunctional}}\geq Acc_{\text{KB-MLP}}.
$$

$$
ParamsRatio\leq1.05.
$$

$$
ForwardFLOPsRatio\leq1.05.
$$

$$
BackwardFLOPsRatio\leq1.50.
$$

$$
StepRatio\leq1.50.
$$

$$
MemoryRatio\leq1.05.
$$

#### 可视化

```text
p8_external_fair_scorecard.svg
p8_acc_vs_params.svg
p8_acc_vs_flops.svg
p8_boundary_by_task.svg
```

---

## 14. Wave 7：Symbolic / robustness / boundary

### P9：symbolic validation

#### 目标

验证 spline / RBF 的 external justification。

#### 判断标准

$$
RMSE_{\text{FullEdgeSpline/RBF}}\leq RMSE_{\text{KB-MLP}}.
$$

or:

$$
RMSE_{\text{FullEdgeSpline/RBF}}\leq RMSE_{\text{CleanTransitional}}.
$$

### P10：robustness

#### 判断标准

$$
AccDrop_{\text{Functional}}\leq AccDrop_{\text{Base}}
$$

for at least 3 perturbation settings.

### P11：boundary audit

Boundary labels：

```text
basis_not_conditioned
basis_fits_simple_targets_only
basis_task_fail
basis_compute_fail
basis_geometry_only
vision_poly_win
symbolic_spline_win
mlp_dominates
full_edge_success
```

---

## 15. Required artifacts

```text
run_manifest.json
basis_registry_audit.csv
basis_activation_conditioning.csv
basis_fit_diagnostics.csv
basis_gradcheck.csv
basis_small_overfit.csv
basis_parameterization_matrix.csv
functional_basis_causality.csv
materialization_free_compute_audit.csv
external_fair_validation.csv
symbolic_validation.csv
robustness_validation.csv
basis_boundary_audit.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

---

## 16. Final route cases

```text
R1-FullEdgeBasisJustifiedBroadSuccess:
  justified full-edge basis passes external fair gate in at least two task families.

R2-FullEdgeVisionBasisSuccess:
  justified full-edge basis passes MNIST-family but not broad tasks.

R3-SymbolicSplineOnlySuccess:
  spline/RBF justified only on symbolic tasks.

R4-BasisComputeFail:
  basis has task signal but compute/memory fair fails.

R5-BasisTaskFail:
  basis is compute fair but cannot match MLP task.

R6-GeometryOnly:
  functional update improves geometry but not task.

R7-TransitionalOnly:
  clean transitional remains best; full-edge basis families fail.

R8-BasisJustificationIncomplete:
  basis selected without passing Wave 0-2 justification.
```

---

## 17. 第一轮执行顺序

```text
Step 1:
  P0 basis registry audit。
  先补公式、support、params、FLOPs、geometry compatibility。

Step 2:
  P1/P2 one-layer diagnostics。
  不跑 full task，先判断 basis 是否数值稳定、能拟合基本目标。

Step 3:
  P3/P4 gradcheck + small overfit。
  没过的小 basis 直接淘汰。

Step 4:
  P5 basis x parameterization matrix。
  只让 justified basis 进入 full task。

Step 5:
  P6 functional subspace compatibility。
  判断 functional update 应作用在哪个 basis subspace。

Step 6:
  P7 materialization-free compute repair。
  解决 v9.0 的 FLOPs/memory failure。

Step 7:
  P8 external fair validation。
  与 KB-MLP / KB-KAN 重新比较。

Step 8:
  P9/P10/P11 symbolic / robustness / boundary。
```

---

## 18. 最终建议

v9.1 的一句话策略是：

$$
\boxed{
\text{先证明为什么选这个 basis，再证明它能训练、能高效、能外部公平地赢 MLP。}
}
$$

现在不应该继续：

```text
随便加一个 basis；
继续手工调 h/basis_count；
只看 task accuracy；
只看 curvature；
只跑 MNIST；
把 v9.0 FullEdge failure 解释成 PureKAN 不行。
```

现在应该做：

```text
1. basis taxonomy；
2. basis conditioning；
3. basis one-layer fit；
4. full-edge gradcheck；
5. small overfit；
6. basis x parameterization matrix；
7. functional subspace causality；
8. materialization-free compute repair；
9. external fair validation。
```

最终目标不是“找到一个看起来能跑的 basis”，而是：

$$
\boxed{
\text{建立一个有理论、数值、任务、几何、系统五重 justification 的 PureKAN basis。}
}
