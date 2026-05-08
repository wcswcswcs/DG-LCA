
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
