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
