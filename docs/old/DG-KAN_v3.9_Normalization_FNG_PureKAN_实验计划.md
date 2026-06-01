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
