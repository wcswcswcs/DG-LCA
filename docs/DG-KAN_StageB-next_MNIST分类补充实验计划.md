# DG-KAN Stage B-next：MNIST small 分类失败补充实验计划

> 版本：v0.1  
> 目标：解释 Stage B-next 中 MNIST small hybrid functional update 未追上 AdamW 的原因，并在新增分类数据集上验证该现象是 MNIST 特例、分类任务共性，还是当前 DG-KAN / functional update 配置问题。  
> 范围：本计划仍属于 Stage B functional-space update 研究，不引入 learned router。所有 credit 仍来自真实 BP / autograd，重点只研究 KAN edge coefficients 的更新几何。  
> 公式格式：Typora 友好，使用 `$...$` 与 `$$...$$`，不使用 `\[...\]`。

---

## 1. 背景与当前问题

Stage B-next 的回归实验已经显示，KAN edge coefficients 的 functional-space update 有明确价值：`diag` 在回归任务上收敛最快，`sobolev` 在多个回归任务上最终 test loss 最好，`diag_to_sobolev` 能较好结合二者优势。Two Moons 分类任务中，hybrid functional update 也能接近 AdamW 的最终 accuracy，同时显著降低 Jacobian condition、`phi_prime_p95` 和坏步数量。

但是 MNIST small 分类任务出现了一个值得深挖的失败现象。当前配置为：

```text
MNIST train / val / test = 6000 / 1000 / 1000
hidden_dim = 32
depth = 2
basis_count = 8
alpha_init = 1.0
epochs = 5
```

在这个设置下，`all_adamw` 结果为：

| method | test acc | test loss | val AUC | train acc | max J cond | phi_prime_p95 |
|---|---:|---:|---:|---:|---:|---:|
| all_adamw | 0.9133 | 0.2731 | 0.3548 | 0.9689 | 1.79 | 0.0440 |

而 hybrid functional update 的典型结果为：

| method | test acc | test loss | val AUC | train acc | max J cond | phi_prime_p95 |
|---|---:|---:|---:|---:|---:|---:|
| hybrid_diag | 0.8987 | 0.3876 | 0.5466 | 0.9284 | 1.12 | 0.0137 |
| hybrid_sobolev | 0.8993 | 0.3882 | 0.5476 | 0.9287 | 1.12 | 0.0139 |
| hybrid_diag_to_sobolev | 0.8990 | 0.3880 | 0.5469 | 0.9287 | 1.12 | 0.0139 |

这个现象不能简单解释为泛化失败。因为 hybrid 的 train accuracy 也明显低于 AdamW：

$$
0.9689 - 0.9287 \approx 0.0402
$$

也就是说，MNIST small 上的 hybrid 失败更像是：

$$
\boxed{\text{optimization / capacity / update strength failure}}
$$

而不是：

$$
\boxed{\text{generalization failure}}
$$

更关键的是，hybrid 的几何指标非常“稳定”：

$$
\text{Jacobian cond}_{hybrid}\approx 1.12 < 1.79
$$

$$
\phi'_{p95,hybrid}\approx 0.014 < 0.044
$$

这说明 hybrid 不是不稳定，而是可能过度保守。它可能让 KAN branch 的有效函数变化太弱，从而抑制了模型对 MNIST 分类边界的表达能力。

因此，本补充实验的核心目标不是继续泛泛比较 optimizer，而是要定位 MNIST small 失败的机制。

---

## 2. 本补充实验要回答的问题

本计划要回答七个具体问题。

第一，hybrid functional update 是否让 KAN branch 过度低活跃？如果是，应该能观察到 hybrid 的 branch output norm、residual update norm、logit contribution 明显低于 AdamW。

第二，functional update 是否只是有效步长太小？如果是，增大 KAN coefficient learning rate 后，train accuracy、branch output norm、`phi_prime_p95` 应该上升，并且 accuracy gap 应明显缩小。

第三，classification 任务是否需要 Adam-style moment dynamics？如果是，Sobolev / diag preconditioner 加上 Adam moments 后，应能追近 `all_adamw`，同时保持较低 Jacobian condition 和较低 `phi_prime_p95`。

第四，当前 MNIST small 失败是否来自模型容量或训练预算不足？如果是，增加 hidden dim、basis count、depth 或 epochs 后，hybrid train accuracy 应显著上升。

第五，RBF basis 覆盖是否不匹配 hidden activation 分布？如果是，应该观察到 active basis fraction 偏低、basis occupancy 不均衡、hidden inputs 大量落在 grid 外或集中在少数 centers 上。

第六，这个失败是否是 MNIST small 特例？需要在 Fashion-MNIST、KMNIST、CIFAR-10 small 等新增分类数据集上重复诊断，观察 hybrid 的欠拟合、几何稳定和 branch under-activation 是否复现。

第七，最终应该形成什么默认分类训练方案？候选包括：

$$
\text{KAN coeff: diag}\rightarrow\text{Sobolev} + \text{non-KAN params: AdamW}
$$

或：

$$
\text{KAN coeff: Sobolev-preconditioned Adam} + \text{non-KAN params: AdamW}
$$

本实验要为 Stage C 之前的 DG-KAN 默认分类配置给出明确建议。

---

## 3. 总体实验设计原则

本补充实验仍然坚持 hybrid 原则：functional update 只作用于 KAN edge coefficients，非 KAN 参数使用 AdamW。非 KAN 参数包括 stem、head、LayerNorm 参数、残差尺度参数以及其他普通线性参数。

普通全 AdamW 是主 baseline：

$$
\text{all_adamw}: \quad \theta_{all}\leftarrow \operatorname{AdamW}(\nabla_\theta \mathcal L)
$$

hybrid functional update 的基本形式是：

$$
\text{KAN coeff}: \quad a\leftarrow a - \eta_a P\nabla_a\mathcal L
$$

$$
\text{non-KAN params}: \quad \psi\leftarrow \operatorname{AdamW}(\nabla_\psi \mathcal L)
$$

其中 $P$ 可以是 diagonal preconditioner、Sobolev preconditioner、`diag_to_sobolev` schedule 或 Sobolev-preconditioned Adam direction。

本实验不把 `all_functional_other_sgd` 当作主方法。它只作为失败锚点保留少量 run，用于确认“functional update 不能粗暴替代所有参数优化器”。

所有关键结果必须进行 3 seed 探索。如果某个配置接近成功，需要用 5 seed 复核。每个 run 必须记录完整 W&B config、训练曲线、几何指标、branch utilization 指标、basis coverage 指标、update strength 指标和 per-class 指标。

---

## 4. 主架构定义

### 4.1 MNIST-like flat DG-KAN classifier

MNIST、Fashion-MNIST、KMNIST、EMNIST 使用同一个 flat-stem residual DG-KAN classifier，以便和当前 MNIST small 结果直接比较。

输入图像展平后进入 stem：

$$
x\in\mathbb R^{784}
$$

$$
h_0 = W_{stem}x + b_{stem}
$$

然后经过 $K$ 个 residual DG-KAN block：

$$
h_{k+1}=h_k+\alpha_k\operatorname{KAN}_k(\operatorname{LN}(h_k))
$$

KAN block 为：

$$
\operatorname{KAN}_k(z)_j = \sum_i \phi_{k,ji}(z_i)
$$

RBF edge function 为：

$$
\phi_{ji}(t)=\sum_{m=1}^{M}a_{jim}\exp(-\gamma(t-c_m)^2)
$$

最后使用 linear head：

$$
\hat y = W_{head}h_K+b_{head}
$$

默认配置从当前 MNIST small 开始：

```text
hidden_dim = 32
depth = 2
basis_count = 8
alpha_init = 1.0
epochs = 5
```

后续 sweep 会扩展：

```text
hidden_dim = 32, 64, 128
depth = 2, 4
basis_count = 8, 16, 32
epochs = 5, 20
```

### 4.2 CIFAR-10 small conv-stem DG-KAN classifier

CIFAR-10 不建议一开始直接使用 flatten stem，因为 flatten stem 会让图像结构丢失，导致结论混杂。CIFAR-10 small 使用轻量 conv stem：

$$
x \rightarrow \operatorname{ConvStem}(x) \rightarrow \operatorname{Pool} \rightarrow h_0
$$

然后接 residual DG-KAN blocks：

$$
h_{k+1}=h_k+\alpha_k\operatorname{KAN}_k(\operatorname{LN}(h_k))
$$

CIFAR-10 small 的主目的不是追求 SOTA，而是验证 MNIST 上观察到的 hybrid under-activation 是否在更复杂视觉特征上复现。

---

## 5. 数据集与 split

### 5.1 主诊断数据集：MNIST small

沿用当前失败设置，保证结果可比：

```text
train / val / test = 6000 / 1000 / 1000
classes = 10
epochs default = 5
```

MNIST small 是本补充实验的中心，因为目标是解释当前失败。

### 5.2 新增分类数据集

为了判断 MNIST 失败是数据集特例还是分类任务共性，本实验增加以下数据集。

#### Fashion-MNIST small

```text
train / val / test = 6000 / 1000 / 1000
classes = 10
image shape = 28 x 28
```

Fashion-MNIST 与 MNIST 形状相同，但类别边界更复杂。它用于测试：hybrid 在更难但同维度数据上是否继续 underfit。

#### KMNIST small

```text
train / val / test = 6000 / 1000 / 1000
classes = 10
image shape = 28 x 28
```

KMNIST 的符号结构不同于数字和衣物，用于测试 MNIST 的低复杂数字结构是否造成特殊偏差。

#### EMNIST balanced small，可选

```text
train / val / test = 12000 / 2000 / 2000
classes = 47
image shape = 28 x 28
```

EMNIST balanced 用于测试类别数增加后，functional update 是否更难训练。如果资源有限，EMNIST 放到第二轮。

#### CIFAR-10 small

```text
train / val / test = 12000 / 2000 / 2000
classes = 10
image shape = 32 x 32 x 3
stem = lightweight conv stem
```

CIFAR-10 small 用于测试 DG-KAN 在更复杂视觉输入上的 branch utilization 和 update strength。

---

## 6. 方法集合

本实验比较以下方法。为了避免 run 数爆炸，不是每个 phase 都跑全部方法；每个 phase 会指定主方法和必要对照。

### 6.1 主 baseline：all_adamw

所有参数都用 AdamW：

```text
method = all_adamw
kan_coeff_optimizer = adamw
nonkan_optimizer = adamw
```

这是分类任务的主性能 baseline。所有 hybrid 的 accuracy、loss AUC、train accuracy、几何指标都必须与它比较。

### 6.2 失败锚点：all_functional_other_sgd

KAN coefficients 使用 functional update，非 KAN 参数使用 SGD：

```text
method = all_functional_other_sgd
kan_coeff_optimizer = diag / sobolev
nonkan_optimizer = sgd
```

该方法已在 Two Moons 和 MNIST 失败。后续只少量运行，用于确认新数据集上是否同样失败，不作为主比较对象。

### 6.3 hybrid_diag

KAN coefficients 使用 diagonal functional preconditioner：

$$
a\leftarrow a-\eta_a\operatorname{diag}(M+\rho I)^{-1}\nabla_a\mathcal L
$$

非 KAN 参数使用 AdamW。

### 6.4 hybrid_sobolev

KAN coefficients 使用 Sobolev preconditioner：

$$
a\leftarrow a-\eta_a(M_{Sob}+\rho I)^{-1}\nabla_a\mathcal L
$$

其中：

$$
M_{mn}
=
\int B_mB_n
+
\alpha\int B'_mB'_n
+
\beta\int B''_mB''_n
$$

默认先使用 Stage B-next 稳定区：

```text
sobolev_alpha = 1e-2
sobolev_beta = 0
rho = 1e-3
```

如果分类任务中仍然过保守，需要降低 $
alpha$ 或增大学习率，而不是盲目增加平滑项。

### 6.5 hybrid_diag_to_sobolev

前 $T_w$ 步使用 diagonal preconditioner，之后切换到 full Sobolev preconditioner：

$$
P_t =
\begin{cases}
\operatorname{diag}(M+\rho I)^{-1}, & t<T_w \\
(M+\rho I)^{-1}, & t\ge T_w
\end{cases}
$$

其中：

$$
T_w = \lfloor f_w T \rfloor
$$

默认 sweep：

```text
warmup_frac = 0.05, 0.10, 0.25
```

### 6.6 hybrid_sobolev_momentum

KAN coefficients 使用 Sobolev-preconditioned momentum：

$$
m_t=\mu m_{t-1}+(1-\mu)\nabla_a\mathcal L
$$

$$
d_t=(M+\rho I)^{-1}m_t
$$

$$
a_{t+1}=a_t-\eta_a d_t
$$

非 KAN 参数使用 AdamW。该方法测试：classification 任务是否缺少 momentum dynamics。

### 6.7 hybrid_sobolev_adam_moment

KAN coefficients 使用 Adam-style moments，再经过 Sobolev preconditioner：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)\nabla_a\mathcal L
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)(\nabla_a\mathcal L)^2
$$

$$
\tilde g_t=\frac{m_t}{\sqrt{v_t}+\epsilon}
$$

$$
d_t=(M+\rho I)^{-1}\tilde g_t
$$

$$
a_{t+1}=a_t-\eta_a d_t
$$

这个方法是本补充实验最重要的新候选之一。它测试：

$$
\boxed{\text{Adam dynamics} + \text{Sobolev geometry}}
$$

是否能同时获得分类性能和几何稳定性。

---

## 7. W&B 全局 config 要求

每个 run 必须记录以下 config。

```text
exp/stage = stage_b_next_mnist_classification_drilldown
exp/phase = M0 / M1 / M2 / M3 / M4 / M5 / M6 / M7
exp/method = all_adamw / hybrid_diag / hybrid_sobolev / hybrid_diag_to_sobolev / hybrid_sobolev_momentum / hybrid_sobolev_adam_moment / all_functional_other_sgd
exp/seed = 0 / 1 / 2 / 3 / 4
```

数据配置：

```text
data/name = mnist_small / fashion_mnist_small / kmnist_small / emnist_balanced_small / cifar10_small
data/train_size
data/val_size
data/test_size
data/batch_size_train
data/batch_size_eval
data/augmentation
```

模型配置：

```text
model/name = residual_dgkan_classifier
model/stem_type = flat / conv
model/hidden_dim
model/depth
model/basis_count
model/rbf_grid_range
model/rbf_gamma_scale
model/alpha_init
model/alpha_learnable
model/num_params
model/trainable_params
```

优化配置：

```text
optim/coeff_update = adamw / diag / sobolev / diag_to_sobolev / sobolev_momentum / sobolev_adam_moment
optim/nonkan_update = adamw / sgd
optim/coeff_lr
optim/rest_lr
optim/weight_decay
optim/warmup_frac
optim/beta1
optim/beta2
optim/momentum_mu
optim/epochs
optim/max_steps
```

metric 配置：

```text
metric/type = diag / sobolev
metric/sobolev_alpha
metric/sobolev_beta
metric/rho
metric/precond_condition
```

basis 配置：

```text
basis/type = rbf
basis/count
basis/grid_min
basis/grid_max
basis/gamma_scale
basis/adaptive_centers = true / false
```

---

## 8. 全局指标记录要求

每个 run 都必须记录以下指标。不要只记录 final accuracy，否则无法解释 MNIST 失败机制。

### 8.1 任务与收敛指标

```text
train/loss
train/acc
val/loss
val/acc
test/loss
test/acc
conv/loss_auc_train
conv/loss_auc_val
conv/steps_to_target_val_acc
conv/steps_to_target_val_loss
conv/time_to_target_sec
conv/loss_at_10pct_steps
conv/loss_at_25pct_steps
conv/loss_at_50pct_steps
conv/loss_at_75pct_steps
conv/plateau_step
```

目标阈值推荐相对 AdamW 定义：

$$
\operatorname{Acc}_{target}=\operatorname{Acc}_{AdamW,final}-0.01
$$

以及：

$$
\mathcal L_{target}=1.05\times\mathcal L_{AdamW,final}
$$

如果 hybrid 达不到 target，应记录 `steps_to_target = inf` 或用固定大值标记。

### 8.2 泛化与欠拟合指标

```text
generalization/train_test_gap_acc
generalization/val_train_gap_acc
generalization/train_test_gap_loss
generalization/val_train_gap_loss
underfit/train_acc_gap_vs_adamw
underfit/train_loss_gap_vs_adamw
underfit/test_acc_gap_vs_adamw
underfit/test_loss_gap_vs_adamw
```

这里最重要的是：如果 hybrid 的 `train_acc_gap_vs_adamw` 很大，则属于优化/表达不足，而不是泛化失败。

### 8.3 Branch utilization 指标

每层记录：

$$
r_{branch,k}
=
\frac{\|\alpha_k\operatorname{KAN}_k(\operatorname{LN}(h_k))\|}
{\|h_k\|+\epsilon}
$$

W&B keys：

```text
branch/layer_{k}/output_norm_ratio
branch/layer_{k}/output_norm
branch/layer_{k}/residual_input_norm
branch/layer_{k}/alpha
branch/layer_{k}/alpha_grad_norm
branch/global/mean_output_norm_ratio
branch/global/max_output_norm_ratio
```

Residual update ratio：

$$
r_{\Delta h,k}
=
\frac{\|h_{k+1}-h_k\|}{\|h_k\|+\epsilon}
$$

W&B keys：

```text
residual/layer_{k}/delta_h_norm_ratio
residual/global/mean_delta_h_norm_ratio
```

### 8.4 Logit contribution ablation

每个 eval epoch，除了正常 forward，还要做两种 ablation：

1. 关闭 KAN branch：

$$
h_{k+1}=h_k
$$

或关闭所有 DG-KAN residual updates，只保留 stem + head。

2. 只记录 KAN branch 对 logits 的变化：

$$
\Delta z = z_{full} - z_{no\_kan}
$$

W&B keys：

```text
ablation/no_kan_branch_val_acc
ablation/no_kan_branch_val_loss
ablation/no_kan_branch_test_acc
ablation/no_kan_branch_test_loss
ablation/kan_branch_logit_delta_norm
ablation/kan_branch_margin_contribution
ablation/acc_drop_when_disabling_kan
ablation/loss_increase_when_disabling_kan
```

如果 hybrid 关闭 KAN branch 后 accuracy 下降很小，而 AdamW 下降明显，则说明 hybrid 下 KAN branch 没有被充分利用。

### 8.5 KAN edge function 几何指标

```text
kan/phi_prime_mean
kan/phi_prime_p95
kan/phi_prime_max
kan/phi_double_prime_energy_mean
kan/phi_double_prime_energy_max
kan/sobolev_norm_mean
kan/sobolev_norm_max
kan/edge_function_norm_mean
kan/edge_function_norm_max
kan/edge_saturation_rate
```

这些指标不能单独解释好坏。它们要和 accuracy、branch utilization、Jacobian condition 一起看。

### 8.6 Jacobian 与 credit amplification 指标

```text
jacobian/sigma_max
jacobian/sigma_min
jacobian/condition
jacobian/condition_p95
credit/amplification_mean
credit/amplification_p95
credit/amplification_max
```

如果 hybrid 的 condition 很低但 train acc 低，说明几何稳定可能是以表达能力不足为代价。

### 8.7 Update strength 指标

Coefficient update ratio：

$$
r_{\Delta a}=\frac{\|\Delta a\|}{\|a\|+\epsilon}
$$

Function update ratio：

$$
r_{\Delta F}=\frac{\|\Delta F(h)\|}{\|F(h)\|+\epsilon}
$$

W&B keys：

```text
update/coeff_delta_over_coeff_norm
update/function_delta_over_function_norm
update/raw_grad_norm_coeff
update/precond_grad_norm_coeff
update/precond_over_raw_grad_norm
update/nonkan_update_norm
update/coeff_update_norm
update/total_update_norm
```

如果 functional update 的 direction 与 AdamW direction 比较可得，还要记录：

```text
update/cos_func_vs_adamw
update/norm_ratio_func_vs_adamw
update/function_delta_ratio_func_vs_adamw
```

### 8.8 Adam moment 指标

只对 `hybrid_sobolev_momentum` 和 `hybrid_sobolev_adam_moment` 记录：

```text
moment/m_norm
moment/v_norm
moment/update_norm_after_moment
moment/update_norm_after_precond
moment/cos_moment_direction_vs_raw_grad
moment/cos_final_direction_vs_adamw
```

### 8.9 Basis coverage 指标

对每层 KAN 输入 $z_k=\operatorname{LN}(h_k)$ 记录分布：

```text
kan_input/layer_{k}/mean
kan_input/layer_{k}/std
kan_input/layer_{k}/p01
kan_input/layer_{k}/p05
kan_input/layer_{k}/p50
kan_input/layer_{k}/p95
kan_input/layer_{k}/p99
kan_input/layer_{k}/out_of_grid_fraction
```

对 RBF basis 激活记录：

```text
basis/layer_{k}/activation_entropy
basis/layer_{k}/active_basis_fraction
basis/layer_{k}/dead_basis_fraction
basis/layer_{k}/center_occupancy_min
basis/layer_{k}/center_occupancy_max
basis/layer_{k}/center_occupancy_cv
basis/layer_{k}/mean_max_basis_activation
```

如果 active basis fraction 很低，说明 RBF centers / gamma / grid range 可能不匹配 hidden distribution。

### 8.10 Per-class 分类指标

```text
class/acc_class_{c}
class/loss_class_{c}
class/margin_class_{c}
class/confusion_matrix
class/worst_class_acc
class/mean_class_acc
```

如果 hybrid 只在某些类别失败，应该通过 confusion matrix 解释，而不是只看平均 accuracy。

### 8.11 Descent alignment 指标

```text
descent/pred
descent/actual
descent/ratio
descent/ratio_median
descent/ratio_p10
descent/pred_actual_corr
descent/sign_agreement
descent/sign_agreement_rate
descent/pred_negative_actual_positive_count
descent/bad_step_total_positive_loss_increase
descent/bad_step_max_loss_increase
```

Stage B-next 已经显示 sign agreement 容易饱和，因此后续主要看 `ratio_median`、`pred_actual_corr` 和坏步幅度。

### 8.12 资源指标

```text
perf/step_time_ms
perf/samples_per_sec
memory/peak_allocated_mb
memory/peak_reserved_mb
```

---

## 9. W&B Dashboard 设计

本补充实验需要单独建一个 dashboard，至少包含 9 个页面。

### Page 1：Overview

展示所有数据集和方法的总览。必须包含：

1. method vs test accuracy bar chart；
2. method vs test loss bar chart；
3. method vs val loss AUC bar chart；
4. method vs train accuracy bar chart；
5. method vs Jacobian condition bar chart；
6. method vs `phi_prime_p95` bar chart；
7. method vs branch output norm ratio bar chart；
8. pass / fail decision table。

主要判断：hybrid 是性能失败、优化失败，还是只是在几何上更稳。

### Page 2：MNIST Failure Diagnosis

该页面专门解释 MNIST small。必须包含：

1. train / val loss curves；
2. train / val accuracy curves；
3. train acc gap vs AdamW over time；
4. test acc gap vs AdamW final bar；
5. branch output norm ratio over time；
6. no-KAN branch ablation accuracy；
7. `phi_prime_p95` over time；
8. Jacobian condition over time；
9. coefficient update norm ratio over time。

如果 hybrid train acc 低且 branch output ratio 低，就支持 under-active KAN branch 假设。

### Page 3：Branch Utilization

展示 KAN branch 是否真正参与分类。图包括：

1. layer-wise branch output norm ratio heatmap；
2. residual delta-h norm ratio heatmap；
3. alpha value and alpha gradient curves；
4. no-KAN branch accuracy drop bar chart；
5. KAN branch logit delta norm curve；
6. margin contribution bar chart。

需要按 method 对比：`all_adamw`、`hybrid_diag`、`hybrid_sobolev`、`hybrid_diag_to_sobolev`、`hybrid_sobolev_adam_moment`。

### Page 4：Update Strength and Direction

展示 functional update 是否太弱或方向偏离 AdamW。图包括：

1. coefficient update norm ratio over time；
2. function update norm ratio over time；
3. raw grad norm vs preconditioned grad norm scatter；
4. `cos_func_vs_adamw` over time；
5. `norm_ratio_func_vs_adamw` over time；
6. coefficient lr heatmap：x=`coeff_lr`，y=`rest_lr`，color=`test_acc`；
7. coefficient lr heatmap：color=`branch/output_norm_ratio`；
8. coefficient lr heatmap：color=`jacobian/condition`。

### Page 5：Adam Moment + Functional Geometry

展示 momentum / Adam moment 是否解决分类优化不足。图包括：

1. method vs test acc；
2. method vs val AUC；
3. method vs train acc；
4. moment norm over time；
5. final direction cosine vs AdamW；
6. final Jacobian condition；
7. final `phi_prime_p95`；
8. performance-geometry Pareto plot：x=`jacobian/condition`，y=`test_acc`。

如果 `hybrid_sobolev_adam_moment` 接近 AdamW accuracy，同时保持较低 condition，则该方向成为新的默认分类方法。

### Page 6：Capacity and Budget

展示容量和训练时间是否影响 hybrid。图包括：

1. hidden_dim vs test acc heatmap；
2. basis_count vs test acc heatmap；
3. depth vs test acc bar chart；
4. epochs vs train acc curve；
5. train acc vs branch output norm scatter；
6. train acc vs `phi_prime_p95` scatter；
7. loss AUC vs final test acc scatter。

### Page 7：Basis Coverage

展示 RBF basis 是否覆盖 hidden activation。图包括：

1. KAN input p01/p05/p95/p99 over time；
2. out-of-grid fraction over time；
3. active basis fraction over time；
4. dead basis fraction over time；
5. basis activation entropy over time；
6. center occupancy histogram；
7. grid_range / gamma_scale heatmap：color=`test_acc`；
8. grid_range / gamma_scale heatmap：color=`active_basis_fraction`。

### Page 8：Dataset Generalization

比较 MNIST、Fashion-MNIST、KMNIST、CIFAR-10 small。图包括：

1. dataset × method test acc heatmap；
2. dataset × method train acc heatmap；
3. dataset × method val AUC heatmap；
4. dataset × method Jacobian condition heatmap；
5. dataset × method branch output ratio heatmap；
6. dataset × method `phi_prime_p95` heatmap；
7. worst-class accuracy by dataset and method。

该页面回答：MNIST failure 是孤例还是分类共性。

### Page 9：Failure Table

每个失败 run 都写入 failure table。字段包括：

```text
failure/type
failure/dataset
failure/method
failure/seed
failure/metric_name
failure/metric_value
failure/threshold
failure/diagnosis
failure/recommended_action
```

推荐 failure types：

```text
underfit_train_acc_gap
kan_branch_underactive
functional_update_too_small
basis_coverage_failure
moment_missing
capacity_insufficient
sobolev_oversmooth
jacobian_too_flat
classification_hybrid_not_competitive
```

---

## 10. 实验 Phase 设计

本补充实验分为 8 个 phase。它们不是随意 sweep，而是按因果诊断顺序推进。

---

## Phase M0：MNIST failure reproduction with full diagnostics

### 目标

复现 MNIST small 当前失败，并补齐 branch activity、basis coverage、update strength、ablation 和 per-class 指标。该 phase 用来确认后续分析的 instrumentation 正常。

### 数据集

```text
MNIST small: train / val / test = 6000 / 1000 / 1000
```

### 模型

```text
hidden_dim = 32
depth = 2
basis_count = 8
alpha_init = 1.0
epochs = 5
```

### 方法

```text
all_adamw
all_functional_other_sgd
hybrid_diag
hybrid_sobolev
hybrid_diag_to_sobolev
```

### Seeds

探索阶段使用：

```text
seeds = 0, 1, 2
```

如果复现与旧结果一致，之后关键配置扩展到：

```text
seeds = 0, 1, 2, 3, 4
```

### 必须记录的重点指标

除全局指标外，M0 特别关注：

```text
underfit/train_acc_gap_vs_adamw
branch/global/mean_output_norm_ratio
ablation/acc_drop_when_disabling_kan
update/coeff_delta_over_coeff_norm
update/function_delta_over_function_norm
basis/global/active_basis_fraction
basis/global/dead_basis_fraction
kan/phi_prime_p95
jacobian/condition
class/confusion_matrix
```

### 解释逻辑

如果 hybrid 复现以下特征：

$$
\text{train acc gap vs AdamW} > 0.03
$$

并且：

$$
\text{branch output ratio}_{hybrid} < 0.5 \times \text{branch output ratio}_{AdamW}
$$

则初步支持 KAN branch under-active 假设。

如果 hybrid 的 `active_basis_fraction` 低于 0.3，或者 out-of-grid fraction 高于 0.1，则同时支持 basis coverage failure 假设。

---

## Phase M1：Branch activity and ablation audit

### 目标

直接回答：hybrid functional update 是否让 KAN branch 没有充分参与分类？

### 数据集

```text
MNIST small
Fashion-MNIST small
```

先在 MNIST 复现，再在 Fashion-MNIST 验证。

### 模型

使用默认小模型：

```text
hidden_dim = 32
depth = 2
basis_count = 8
epochs = 5
```

### 方法

```text
all_adamw
hybrid_diag
hybrid_sobolev
hybrid_diag_to_sobolev
```

### 关键记录

每个 eval epoch 执行 no-KAN ablation。

正常模型：

$$
z_{full}=f(x)
$$

关闭 KAN branch：

$$
z_{no\_kan}=f_{no\_kan}(x)
$$

记录：

$$
\Delta z = z_{full}-z_{no\_kan}
$$

W&B keys：

```text
ablation/no_kan_branch_val_acc
ablation/no_kan_branch_test_acc
ablation/acc_drop_when_disabling_kan
ablation/kan_branch_logit_delta_norm
ablation/kan_branch_margin_contribution
branch/layer_{k}/output_norm_ratio
residual/layer_{k}/delta_h_norm_ratio
```

### 关键可视化

1. method vs `acc_drop_when_disabling_kan`；
2. method vs `branch/global/mean_output_norm_ratio`；
3. layer-wise branch output ratio heatmap；
4. logit delta norm over epochs；
5. test accuracy vs branch output ratio scatter。

### 判定标准

如果 hybrid 的 `acc_drop_when_disabling_kan` 很小，例如：

$$
\Delta \text{acc}_{disableKAN}<0.01
$$

而 AdamW 的下降明显，例如：

$$
\Delta \text{acc}_{disableKAN}>0.03
$$

则说明 hybrid 下 KAN branch 没有充分参与分类。

如果 hybrid 的 branch output ratio 低于 AdamW 的 50%，同时 train acc gap 大，则 MNIST 失败主要是 branch under-activation。

---

## Phase M2：Effective update strength and learning-rate sweep

### 目标

回答：MNIST hybrid 是否只是 KAN coefficient 更新太弱？

### 数据集

```text
MNIST small
```

若 MNIST 发现有效配置，再迁移到：

```text
Fashion-MNIST small
KMNIST small
```

### 方法

```text
hybrid_diag
hybrid_sobolev
hybrid_diag_to_sobolev
```

### Sweep

KAN coefficient learning rate：

```text
coeff_lr = 0.03, 0.10, 0.30, 1.00
```

non-KAN AdamW learning rate：

```text
rest_lr = 3e-4, 1e-3, 3e-3
```

warmup for `diag_to_sobolev`：

```text
warmup_frac = 0.05, 0.10, 0.25
```

为了控制 run 数，第一轮采用如下矩阵：

```text
hybrid_diag: coeff_lr x rest_lr
hybrid_sobolev: coeff_lr x rest_lr
hybrid_diag_to_sobolev: coeff_lr in {0.10, 0.30, 1.00}, rest_lr in {1e-3, 3e-3}, warmup_frac in {0.05, 0.10, 0.25}
```

### 必须记录

```text
train/acc
test/acc
conv/loss_auc_val
branch/global/mean_output_norm_ratio
update/coeff_delta_over_coeff_norm
update/function_delta_over_function_norm
kan/phi_prime_p95
jacobian/condition
jacobian/sigma_max
jacobian/sigma_min
stability/loss_spike_count
descent/bad_step_total_positive_loss_increase
```

### 可视化

1. heatmap：x=`coeff_lr`，y=`rest_lr`，color=`test/acc`；
2. heatmap：x=`coeff_lr`，y=`rest_lr`，color=`train/acc`；
3. heatmap：x=`coeff_lr`，y=`rest_lr`，color=`branch/global/mean_output_norm_ratio`；
4. heatmap：x=`coeff_lr`，y=`rest_lr`，color=`jacobian/condition`；
5. scatter：`branch output ratio` vs `test acc`；
6. scatter：`phi_prime_p95` vs `test acc`；
7. line：`coeff_lr` vs `update/function_delta_over_function_norm`。

### 判定标准

如果增大 `coeff_lr` 后：

$$
\text{train acc gap vs AdamW}<0.01
$$

并且：

$$
\text{test acc gap vs AdamW}<0.02
$$

同时：

$$
\text{Jacobian cond}_{hybrid}<\text{Jacobian cond}_{AdamW}
$$

则说明原失败主要是 KAN coefficient 有效步长不足，functional update 本身仍然成立。

如果增大 `coeff_lr` 后 train acc 上升，但 Jacobian condition 和 `phi_prime_p95` 爆炸，则说明 functional update 的稳定优势来自强约束，必须在表达与几何之间找 Pareto 点。

如果无论 `coeff_lr` 如何，train acc 都不上升，则需要进入 M3 / M4 / M5 继续排查。

---

## Phase M3：Adam-style moment with Sobolev geometry

### 目标

回答：classification 任务是否需要 Adam 的 moment dynamics，而 Sobolev 只负责函数空间几何？

### 数据集

```text
MNIST small
Fashion-MNIST small
KMNIST small
```

### 方法

```text
all_adamw
hybrid_sobolev
hybrid_sobolev_momentum
hybrid_sobolev_adam_moment
hybrid_diag_to_sobolev
hybrid_diag_to_sobolev_adam_moment
```

### 关键配置

使用 M2 中找到的较好 `coeff_lr` 和 `rest_lr`。如果 M2 还没有完成，初始使用：

```text
coeff_lr = 0.10, 0.30
rest_lr = 1e-3
sobolev_alpha = 1e-2
rho = 1e-3
beta1 = 0.9
beta2 = 0.999
```

### 必须记录

```text
moment/m_norm
moment/v_norm
moment/update_norm_after_moment
moment/update_norm_after_precond
moment/cos_moment_direction_vs_raw_grad
moment/cos_final_direction_vs_adamw
update/cos_func_vs_adamw
update/norm_ratio_func_vs_adamw
train/acc
test/acc
conv/loss_auc_val
branch/global/mean_output_norm_ratio
kan/phi_prime_p95
jacobian/condition
```

### 可视化

1. method vs test acc；
2. method vs val AUC；
3. method vs train acc；
4. method vs Jacobian condition；
5. method vs `phi_prime_p95`；
6. performance-geometry Pareto：x=`jacobian/condition`，y=`test/acc`；
7. update direction cosine vs AdamW over epochs；
8. moment norm over time。

### 判定标准

如果 `hybrid_sobolev_adam_moment` 满足：

$$
\text{test acc gap vs AdamW}<0.01
$$

且：

$$
\text{Jacobian cond}_{method}<0.8\times\text{Jacobian cond}_{AdamW}
$$

或：

$$
\phi'_{p95,method}<0.8\times\phi'_{p95,AdamW}
$$

则说明分类任务需要：

$$
\boxed{\text{Adam dynamics} + \text{functional geometry}}
$$

这将成为 Stage B 的重要新结论。

如果 Adam moment 仍无法提升 train acc，则说明失败更可能来自容量、basis coverage 或 architecture。

---

## Phase M4：Capacity and training budget sweep

### 目标

回答：MNIST hybrid 是否因为模型容量太小或训练太短而失败？

### 数据集

```text
MNIST small
Fashion-MNIST small
```

### 方法

选择 M2 / M3 中表现最好的两个 hybrid 方法，加上 `all_adamw`。默认候选：

```text
all_adamw
hybrid_sobolev
hybrid_sobolev_adam_moment
hybrid_diag_to_sobolev
```

### Sweep

为了避免组合爆炸，采用 staged matrix。

第一组：扩大 hidden dim。

```text
hidden_dim = 32, 64, 128
basis_count = 8
depth = 2
epochs = 5
```

第二组：增加 basis count。

```text
hidden_dim = 64
basis_count = 8, 16, 32
depth = 2
epochs = 5
```

第三组：增加 depth。

```text
hidden_dim = 64
basis_count = 16
depth = 2, 4
epochs = 5
```

第四组：增加训练预算。

```text
hidden_dim = 64
basis_count = 16
depth = 4
epochs = 5, 20
```

### 必须记录

```text
train/acc
test/acc
conv/loss_auc_val
conv/steps_to_target_val_acc
branch/global/mean_output_norm_ratio
kan/phi_prime_p95
kan/phi_double_prime_energy_mean
jacobian/condition
update/function_delta_over_function_norm
basis/active_basis_fraction
```

### 可视化

1. hidden_dim vs test acc heatmap；
2. basis_count vs test acc heatmap；
3. depth vs test acc bar；
4. epochs vs train acc curve；
5. train acc vs branch output ratio scatter；
6. test acc vs `phi_prime_p95` scatter；
7. test acc vs Jacobian condition Pareto。

### 判定标准

如果增加 capacity 后 hybrid 追上 AdamW，则原失败主要是容量 / 表达不足。

如果延长 epochs 后 hybrid 追上 AdamW，但 5 epoch 不行，则原失败主要是收敛速度不足。

如果 capacity 和 budget 都增加后仍明显低于 AdamW，则需要重点检查 basis coverage 和 update dynamics。

---

## Phase M5：RBF basis coverage and grid/gamma audit

### 目标

回答：MNIST hybrid 失败是否因为 RBF basis 没有覆盖 hidden activation 分布？

### 数据集

```text
MNIST small
Fashion-MNIST small
KMNIST small
```

### 方法

使用 M2 / M3 中最有希望的 hybrid 方法，以及 `all_adamw` 对照。

### Sweep

```text
grid_range = [-2,2], [-3,3], [-4,4]
rbf_gamma_scale = 0.25, 0.5, 1.0, 2.0
basis_count = 8, 16, 32
adaptive_centers = false, true
```

第一轮不做全排列，推荐：

```text
basis_count = 8, grid_range sweep, gamma_scale sweep
basis_count = 16, best grid_range, gamma_scale sweep
adaptive_centers = true only for best two static configs
```

### 必须记录

```text
kan_input/layer_{k}/p01
kan_input/layer_{k}/p05
kan_input/layer_{k}/p95
kan_input/layer_{k}/p99
kan_input/layer_{k}/out_of_grid_fraction
basis/layer_{k}/active_basis_fraction
basis/layer_{k}/dead_basis_fraction
basis/layer_{k}/activation_entropy
basis/layer_{k}/center_occupancy_cv
basis/layer_{k}/mean_max_basis_activation
train/acc
test/acc
branch/global/mean_output_norm_ratio
kan/phi_prime_p95
jacobian/condition
```

### 可视化

1. KAN input percentile curves；
2. out-of-grid fraction over epochs；
3. basis activation entropy over epochs；
4. center occupancy histogram；
5. grid range × gamma heatmap：test acc；
6. grid range × gamma heatmap：active basis fraction；
7. active basis fraction vs train acc scatter；
8. out-of-grid fraction vs test acc scatter。

### 判定标准

如果当前失败配置存在：

$$
\text{active basis fraction}<0.3
$$

或：

$$
\text{out-of-grid fraction}>0.1
$$

并且调整 grid / gamma 后 train acc 和 test acc 明显提升，则 basis coverage 是主要失败原因。

如果 basis coverage 一直正常，但 hybrid 仍欠拟合，则转向 update dynamics / capacity 解释。

---

## Phase M6：Small subset memorization test

### 目标

区分正则化过强、更新过弱和真正表达能力不足。一个方法如果不能 memorize 很小的训练子集，说明它存在明显优化或容量问题。

### 数据集

从 MNIST 中取：

```text
train_subset = 256, 512, 1024
val = 1000
test = 1000
```

不使用 data augmentation。

### 方法

```text
all_adamw
hybrid_sobolev
hybrid_sobolev_adam_moment
hybrid_diag_to_sobolev
```

### 训练预算

训练足够长：

```text
epochs = 50
```

### 必须记录

```text
train/acc
train/loss
val/acc
test/acc
conv/steps_to_99pct_train_acc
branch/global/mean_output_norm_ratio
kan/phi_prime_p95
jacobian/condition
basis/active_basis_fraction
```

### 可视化

1. train accuracy over epochs；
2. steps to 99% train accuracy；
3. method vs final train acc；
4. branch output ratio vs train acc；
5. phi_prime_p95 vs train acc。

### 判定标准

如果 AdamW 可以 100% memorize，而 hybrid 无法达到 99% train acc，则 functional update 抑制了表达或优化。

如果 hybrid 也能 memorize，但在 6000 样本上追不上，则说明问题是训练预算、mini-batch optimization 或数据规模下的动态，而不是基本表达能力。

---

## Phase M7：新增分类数据集外推验证

### 目标

判断 MNIST small 失败是否是特例，或者 hybrid functional update 在分类任务中普遍存在 under-activation / underfit。

### 数据集

第一轮：

```text
MNIST small
Fashion-MNIST small
KMNIST small
```

第二轮：

```text
CIFAR-10 small
EMNIST balanced small optional
```

### 方法

选择前面 phase 中最有希望的配置。至少包括：

```text
all_adamw
hybrid_sobolev
hybrid_diag_to_sobolev
hybrid_sobolev_adam_moment
```

如果 M2 显示 diag 更强，则加入：

```text
hybrid_diag
```

### 模型

MNIST-like 数据：

```text
flat stem residual DG-KAN
hidden_dim = best from M4
basis_count = best from M4 / M5
depth = best from M4
epochs = 20 if 5 insufficient
```

CIFAR-10 small：

```text
conv stem residual DG-KAN
hidden_dim = 64 or 128
basis_count = 16
depth = 4
```

### 必须记录

```text
train/acc
test/acc
conv/loss_auc_val
underfit/train_acc_gap_vs_adamw
underfit/test_acc_gap_vs_adamw
branch/global/mean_output_norm_ratio
ablation/acc_drop_when_disabling_kan
kan/phi_prime_p95
jacobian/condition
basis/active_basis_fraction
class/worst_class_acc
class/confusion_matrix
```

### 可视化

1. dataset × method test acc heatmap；
2. dataset × method train acc heatmap；
3. dataset × method train acc gap vs AdamW heatmap；
4. dataset × method branch output ratio heatmap；
5. dataset × method Jacobian condition heatmap；
6. dataset × method `phi_prime_p95` heatmap；
7. worst-class accuracy by method；
8. confusion matrix per dataset for AdamW vs best hybrid。

### 判定标准

如果 hybrid 只在 MNIST small 失败，而在 Fashion-MNIST / KMNIST 追近 AdamW，则 MNIST failure 可能来自当前 split、短训练或 basis 配置。

如果 hybrid 在所有分类任务上都 train acc 低且 branch output ratio 低，则说明 functional update 在分类任务中普遍偏保守，需要 Adam moment 或 branch scaling 作为默认组件。

如果 `hybrid_sobolev_adam_moment` 在多个分类数据集上接近 AdamW，同时保持更低 Jacobian condition 和更低 `phi_prime_p95`，则该方法成为 Stage B 分类默认方案。

---

## 11. 执行顺序与 run 预算

为了避免无控制地扩大实验量，本计划按三轮执行。

### Round 1：MNIST failure instrumentation

目的：先把 MNIST 失败机制确认清楚。

执行：

```text
M0: 5 methods x 3 seeds = 15 runs
M1: 4 methods x 2 datasets x 3 seeds = 24 runs
M2: focused lr sweep on MNIST, about 36 runs
```

Round 1 结束后必须回答：

1. hybrid 是否 branch under-active？
2. 增大 coeff_lr 是否能提升 train acc？
3. 失败是否主要是 update strength？

### Round 2：Moment、容量、basis coverage

执行：

```text
M3: 6 methods x 3 datasets x 3 seeds = 54 runs
M4: staged capacity sweep, about 48 runs
M5: basis coverage sweep, about 36 runs
M6: memorization test, about 36 runs
```

Round 2 结束后必须回答：

1. Adam moments 是否必要？
2. 增加容量 / 训练预算是否解决问题？
3. RBF basis coverage 是否是失败原因？
4. hybrid 是否能 memorize 小数据集？

### Round 3：新增分类数据集验证

执行：

```text
M7: best 4 methods x 3 or 4 datasets x 5 seeds = 60 to 80 runs
```

Round 3 结束后给出分类任务结论。

---

## 12. 成功标准

### 12.1 机制解释成功

如果实验能把 MNIST 失败归因到以下任一清晰机制，则本补充实验成功：

1. KAN branch under-active；
2. functional update effective step 太小；
3. 缺少 Adam-style moment dynamics；
4. 模型容量 / 训练预算不足；
5. RBF basis coverage 不匹配；
6. Sobolev smoothness 过强导致分类表达不足。

要求不是所有机制都成立，而是要通过指标排除错误解释。

### 12.2 方法改进成功

如果找到新 hybrid 方法满足：

$$
\text{test acc gap vs AdamW}<0.01
$$

并且：

$$
\text{train acc gap vs AdamW}<0.01
$$

同时满足至少一个几何优势：

$$
\text{Jacobian cond}_{hybrid}<0.8\times\text{Jacobian cond}_{AdamW}
$$

或：

$$
\phi'_{p95,hybrid}<0.8\times\phi'_{p95,AdamW}
$$

则认为分类 hybrid functional update 修复成功。

### 12.3 跨数据集成功

如果 best hybrid 在至少两个新增分类数据集上满足：

$$
\text{test acc gap vs AdamW}<0.02
$$

并且：

$$
\text{train acc gap vs AdamW}<0.02
$$

同时几何指标优于 AdamW，则可以声称：

> Functional geometry can be useful for KAN coefficients in classification when combined with proper optimization dynamics.

### 12.4 失败但有价值

如果所有修复都无法追上 AdamW，但能清楚证明 hybrid 的失败来自过度保守的 KAN branch 或缺少 moment dynamics，也仍然是有价值结果。此时 Stage B 结论应写为：

> Functional-space update is strong for regression and improves geometry in simple classification, but current classification optimization requires additional adaptive dynamics or branch-utilization control.

---

## 13. 失败诊断表

每个 run 如果触发下列条件，应写入 W&B failure table。

| failure type | 触发条件 | 解释 |
|---|---|---|
| `underfit_train_acc_gap` | train acc gap vs AdamW > 0.03 | 训练集未充分拟合 |
| `kan_branch_underactive` | branch output ratio < 0.5 x AdamW | KAN branch 贡献太弱 |
| `no_kan_ablation_small_drop` | disable KAN 后 acc drop < 0.01 | KAN branch 未被利用 |
| `functional_update_too_small` | function update ratio < 0.5 x AdamW | 有效函数步长不足 |
| `basis_coverage_failure` | active basis fraction < 0.3 或 out-of-grid > 0.1 | RBF basis 覆盖不足 |
| `moment_missing` | Adam moment 版本显著优于无 moment | 分类需要 adaptive dynamics |
| `capacity_insufficient` | 增大容量后性能显著提升 | 模型或 basis 太小 |
| `sobolev_oversmooth` | phi_prime / curvature 太低且 train acc 低 | 平滑过强 |
| `jacobian_too_flat` | condition 接近 1 但 train acc 低 | 表达被压平 |
| `classification_hybrid_not_competitive` | test acc gap > 0.03 | 分类任务性能不够 |

---

## 14. 最终预期结论形式

本补充实验结束后，应该能给出以下三类结论之一。

### 结论 A：修复成功

如果 Sobolev + Adam moment 或其他 hybrid 方案追近 AdamW，同时保持更健康的几何，则 Stage B 分类结论升级为：

> KAN edge coefficients benefit from functional-space geometry, but classification requires Adam-style temporal adaptivity. The best update is not pure functional SGD but Adam dynamics preconditioned by Sobolev geometry.

### 结论 B：部分成功

如果 hybrid 在 Two Moons / Fashion-MNIST 有效，但 MNIST / CIFAR-10 仍落后，则结论为：

> Functional update improves local geometry and can match AdamW in simple classification, but robustness across classification tasks depends on branch utilization, basis coverage, and update dynamics.

### 结论 C：失败但解释清楚

如果 hybrid 在所有分类数据集上都不能追近 AdamW，但指标显示它稳定且 under-active，则结论为：

> Current functional update acts as a strong geometry stabilizer but suppresses KAN branch expressivity in classification. It remains valuable for regression and geometry control, but not yet as a standalone classification optimizer for DG-KAN.

无论哪种结论，都能推进 Stage C，因为 Stage C 的 analytic KAN adjoint / credit routing 可以在已经明确的 update geometry 基础上继续实验。区别只在于 Stage C 分类默认 update 选用什么配置。

