# DG-KAN Optimizer v3.5 下一步实验计划：U-FULL f1 默认、泛化确认、CIFAR 预检、速度优化与 Rational/KAT 路线

> 版本：v3.5 draft  
> 日期：2026-05-02  
> 目标：在 v3.4 之后冻结主 optimizer 结构，把 `branch_final_scale = 1.0` 设为 U-FULL 默认，系统验证它在泛化、scaling、速度和 RBF 替代路线上的可行性。  
> 公式格式：Typora 友好，全文只使用 `$...$` 和 `$$...$$`。  
> 关键词：U-FULL, branch_final_scale=1, Hybrid GA-FU, full Sobolev from start, alphaFixed1, small-data, label-noise, CIFAR-small, ConvStem-DGKAN, true-Gram speed, Rational/KAT。

---

## 0. 这份计划的核心判断

v3.4 之后，optimizer 结论已经比较清楚：

```text
KAN coeff:
  适合 Sobolev functional update。

non-KAN params:
  当前仍然需要 AdamW。

true all-functional update:
  以当前 non-KAN metrics 还不可行。

partial AFU:
  HeadCov / LNOnly 有分析价值，但不替代主线。
```

因此下一步不再继续大规模搜索 optimizer 组件，而是冻结主线结构：

```text
Hybrid GA-FU:
  KAN coeff = full Sobolev functional update
  non-KAN params = AdamW
  alpha_mode = fixed1
```

用户要求把：

```text
branch_final_scale = 1.0
```

设为默认 U-FULL。因此 v3.5 的主 profile 定义为：

```text
U-FULL-f100
```

它表示：

```text
full_sobolev_gram from start
alpha_mode = fixed1
branch_final_scale = 1.0
KAN coeff = Sobolev functional update
non-KAN params = AdamW
```

本轮最重要的科学问题变成：

$$
\boxed{
\text{如果不再压低 KAN branch，U-FULL-f100 能否仍然保持 accuracy、AUC、geometry、ECE 的 Pareto 优势？}
}
$$

也就是说，我们要验证：

```text
branch_final_scale=1.0 是否是一个更自然、更统一、更可解释的默认设置。
```

但也要诚实承认风险：v3.4 中 `branch_final_scale=0.875 / 0.90` 已经显示，进一步增强 branch 会提高 expression，但可能牺牲 accuracy / AUC。因此 v3.5 不应假设 f1 一定成功，而应把它作为正式 default hypothesis 来验证。

---

## 1. v3.5 的主线定义

### 1.1 DG-KAN block

DG-KAN block 写成：

$$
h_{k+1}
=
h_k
+
\alpha b_t \operatorname{KAN}(\operatorname{LN}(h_k)).
$$

v3.5 默认：

$$
\alpha=1.
$$

同时：

$$
b_t = 1.0.
$$

因此默认 forward 为：

$$
h_{k+1}
=
h_k
+
\operatorname{KAN}(\operatorname{LN}(h_k)).
$$

这就是 `alphaFixed1 + branch_final_scale=1.0` 的含义。

### 1.2 KAN coefficient update

KAN edge function：

$$
\phi_{ji}(t)
=
\sum_m a_{jim}B_m(t).
$$

Sobolev Gram：

$$
M
=
\int B(t)B(t)^\top dt
+
\alpha_s\int B'(t)B'(t)^\top dt
+
\beta_s\int B''(t)B''(t)^\top dt.
$$

v3.5 U-FULL 默认从训练一开始就使用 full Sobolev Gram：

$$
a_{t+1}
=
a_t
-
\eta_a(t)(M+\rho I)^{-1}\nabla_a L_t.
$$

不再使用 early diag phase，不再使用 hard transition，也不使用 smooth transition。这样 profile 更简单：

```text
ACTIVE / TRANSITION / GEOMETRY 不再作为主 schedule。
full Sobolev Gram from start 是唯一 KAN coefficient metric。
```

### 1.3 non-KAN 参数更新

non-KAN 参数继续使用 AdamW：

$$
\theta_{t+1}
=
\operatorname{AdamW}(\theta_t,\nabla_\theta L_t).
$$

这里包括：

```text
stem
head
LayerNorm
KAN bias
其他非 KAN coefficient 参数
```

注意：v3.4 已经测试过当前版本的 true all-functional update，结果不够好。因此 v3.5 不再把 all-functional update 当主线，只保留为历史边界结论。

---

## 2. 为什么把 branch_final_scale 设为 1.0

### 2.1 动机

`branch_final_scale=0.85` 是一个经验折中点。它让 KMNIST 的 no-KAN expression 从 v3.2 的约 `0.663` 提升到约 `0.696`，但严格 `0.700` threshold 仍差一点。

把 `branch_final_scale=1.0` 设为默认的动机是：

```text
1. 更自然：residual branch 默认不再人为缩放。
2. 更统一：不需要解释为什么最终只用 85% KAN branch。
3. 更表达：更可能让 no-KAN drop / margin contribution 达到 expression gate。
4. 更贴近未来 scaling：ConvStem / Rational / KAT 路线中，默认 branch 不压缩更容易作为基础架构描述。
```

### 2.2 风险

风险也很明确：

```text
1. branch 过强可能导致 AUC 变差。
2. KAN margin 变大不一定带来 accuracy 提升。
3. phi_prime / Jacobian 可能恶化。
4. calibration 可能变差。
5. CIFAR-small 上可能更不稳定。
```

所以 v3.5 的关键不是盲目证明 f1，而是用严格实验回答：

$$
\boxed{
\text{f1 的表达力收益是否大于它带来的 AUC / geometry / calibration 代价？}
}
$$

---

# 第一部分：冻结 v3.4 结论并定义 v3.5 default

## 3. P0：v3.4 结论冻结与 U-FULL-f100 配置检查

### 3.1 目标

P0 不是训练主实验，而是把 v3.4 的边界结论写清楚，同时确保 v3.5 代码配置没有混淆。

v3.4 需要冻结的结论：

```text
1. Hybrid optimizer remains mainline.
2. true all-functional update with current non-KAN metrics is not viable.
3. U-FULL-f085 was the previous unified Pareto profile.
4. v3.5 changes only one default: branch_final_scale from 0.85 to 1.00.
```

### 3.2 代码配置

新增或确认：

```text
profile_name = U-FULL-f100
alpha_mode = fixed1
branch_final_scale = 1.0
branch_boost = 1.0
coeff_lr_boost = 1.0
gafu_v3_enabled = true
v3_metric_active = full_sobolev_gram
v3_metric_transition = full_sobolev_gram
v3_metric_geometry = full_sobolev_gram
v3_phase_mode = hard or none
```

这里为了避免 schedule 混淆，推荐：

```text
branch_schedule = none
```

如果 runner 必须使用 `geometry_aware`，也要确保：

```text
current_branch_scale = 1.0
metric_mix = 1.0
phase = GEOMETRY
switch_step = 0
```

### 3.3 必须记录

每个 run 必须写入：

```text
profile_name
alpha_mode
alpha_trainable
branch_final_scale
branch_boost
branch_schedule
v3_metric_active
v3_metric_transition
v3_metric_geometry
v3_phase_final
metric_mix_auc
kan_coeff_optimizer
nonkan_optimizer
```

### 3.4 P0 通过条件

P0 通过要求：

```text
1. U-FULL-f100 的 config row 中 branch_final_scale 必须为 1.0。
2. alpha_trainable = false。
3. alpha_final_mean = 1.0。
4. KAN coeff 走 full_sobolev_gram。
5. non-KAN 参数仍走 AdamW。
6. trust_clip_rate 近似为 0，除非显式启用 trust safety。
```

---

# 第二部分：U-FULL-f100 clean confirm

## 4. P1：U-FULL-f100 smoke 与 seed0 sanity

### 4.1 目标

P1 只确认 f1 default 不会立即导致数值异常或明显 branch explosion。

### 4.2 设置

```text
datasets = Fashion-MNIST, KMNIST
methods = AdamW, U-FULL-f085-reference, U-FULL-f100
seeds = 0
Fashion epochs = 30
KMNIST epochs = 20
model = h96 / depth4 / basis24
alpha_mode = fixed1
```

### 4.3 必须记录

任务指标：

```text
train_loss
val_loss
test_loss
train_acc
val_acc
test_acc
val_loss_auc
val_acc_auc
```

表达力指标：

```text
branch/A
no_kan_test_acc
no_kan_acc_drop
no_kan_drop_over_adamw
kan_logit_delta_norm_mean
kan_logit_delta_norm_p95
kan_margin_contribution_mean
kan_margin_contribution_p95
```

几何指标：

```text
phi_prime_p95
phi_prime_max
jacobian_condition_max
curvature_energy
sobolev_norm_mean
```

稳定性指标：

```text
nan_or_inf_count
loss_spike_count
trust_clip_rate
update_over_coeff_norm_mean
update_over_coeff_norm_p95
```

### 4.4 P1 通过条件

U-FULL-f100 进入 P2 的条件：

Fashion：

$$
\operatorname{Acc}_{f100}
\geq
\operatorname{Acc}_{AdamW}-0.5\%.
$$

KMNIST：

$$
\operatorname{Acc}_{f100}
\geq
\operatorname{Acc}_{AdamW}-1.0\%.
$$

两个数据集还需要：

$$
\operatorname{AUCImprove}>5\%.
$$

$$
\phi\text{ red}>20\%.
$$

$$
J\text{ red}>20\%.
$$

并且：

$$
0.5<\operatorname{branch/A}<0.95.
$$

如果 branch/A 超过 `1.0`，但 accuracy / AUC 都更好，仍可进入 P2，但必须标记：

```text
branch_high_watch = true
```

---

## 5. P2：U-FULL-f100 5-seed clean confirm

### 5.1 目标

P2 判断 f1 是否能替代 f085 成为 unified profile。

### 5.2 方法

```text
AdamW-alphaFixed1
U-FULL-f085-reference
U-FULL-f100
F-V3-HARD-base30, Fashion optional upper bound
```

### 5.3 设置

```text
datasets = Fashion-MNIST, KMNIST
seeds = 0,1,2,3,4
Fashion epochs = 30
KMNIST epochs = 20
train / val / test = 6000 / 1000 / 1000
model = h96 / depth4 / basis24
```

### 5.4 主要 scorecard

每个 method 输出：

```text
runs
test_acc_mean
test_acc_std
acc_gap_vs_adamw
val_auc_improvement_vs_adamw
phi_prime_reduction_vs_adamw
jac_reduction_vs_adamw
branch_over_adamw
ece_reduction_vs_adamw
no_kan_drop_over_adamw
kan_margin_contribution_mean
time_ratio_vs_adamw
step_time_ms
```

### 5.5 Paired analysis

对每个 seed 计算：

$$
\Delta Acc_s
=
Acc_{method,s}-Acc_{AdamW,s}.
$$

$$
\Delta AUC_s
=
AUC_{AdamW,s}-AUC_{method,s}.
$$

$$
\Delta noKAN_s
=
noKAN_{method,s}-noKAN_{AdamW,s}.
$$

输出：

```text
paired_acc_delta_mean
paired_acc_delta_ci95
paired_auc_delta_mean
paired_auc_delta_ci95
paired_noKAN_delta_mean
paired_noKAN_delta_ci95
```

### 5.6 P2 判定

U-FULL-f100 如果满足以下条件，则进入 P3 10-seed final：

Fashion：

$$
Acc_{f100}\geq Acc_{AdamW}.
$$

KMNIST：

$$
Acc_{f100}\geq Acc_{AdamW}-0.5\%.
$$

两个数据集：

$$
\operatorname{AUCImprove}>8\%.
$$

$$
\phi\text{ red}>25\%.
$$

$$
J\text{ red}>20\%.
$$

$$
ECE\text{ red}>10\%.
$$

表达力：

$$
noKAN\text{ ratio}>0.70.
$$

如果 U-FULL-f100 满足表达力但 AUC / accuracy 明显低于 f085，则它不能替代 f085，只能作为 expression-heavy profile。

---

## 6. P3：U-FULL-f100 10-seed final confirm

### 6.1 触发条件

只有 P2 通过后才运行 P3。

### 6.2 方法

```text
AdamW-alphaFixed1
U-FULL-f085-reference
U-FULL-f100
F-V3-HARD-base30, Fashion optional
```

### 6.3 设置

```text
seeds = 0,1,2,3,4,5,6,7,8,9
```

### 6.4 Final decision

可能出现三种结论：

#### 结论 A：f100 成为新 default

如果 U-FULL-f100 在两个数据集上都满足：

```text
acc not worse than AdamW
AUC improvement > 8%
phi/J reductions passed
ECE reduction > 10%
noKAN ratio > 0.70
```

并且相对 f085 没有明显劣化：

$$
Acc_{f100}\geq Acc_{f085}-0.3\%.
$$

则：

```text
v3.5 unified default = U-FULL-f100
```

#### 结论 B：f100 是 expression-heavy profile，f085 保持 Pareto default

如果 f100 的 noKAN 更高，但 accuracy/AUC 不如 f085，则：

```text
U-FULL-f085 = unified Pareto default
U-FULL-f100 = expression-heavy ablation
```

#### 结论 C：f100 失败

如果 f100 造成 branch/A 过高、AUC 下降或 accuracy 明显掉，则：

```text
f100 not adopted
f085 remains default
```

---

# 第三部分：small-data 与 label-noise 泛化确认

## 7. P4：small-data generalization confirm

### 7.1 目标

验证 U-FULL-f100 或最终选定的 U-FULL profile 是否在小数据下保留 geometry / calibration / generalization 优势。

### 7.2 方法

```text
AdamW-alphaFixed1
U-FULL-selected
U-FULL-f085-reference, if selected is f100
F-V3-HARD-base30, Fashion optional
```

### 7.3 数据设置

```text
datasets = Fashion-MNIST, KMNIST
train_size = 500, 1000, 2000, 6000
val_size = 1000
test_size = 1000
seeds = 0,1,2,3,4
```

训练 epoch 建议：

```text
train_size = 500:
  epochs = 50

train_size = 1000:
  epochs = 40

train_size = 2000:
  epochs = 30

train_size = 6000:
  Fashion epochs = 30
  KMNIST epochs = 20
```

如果希望保持 step budget 可比，可以额外输出 matched-step analysis。

### 7.4 必须记录

任务指标：

```text
train_acc
val_acc
test_acc
train_loss
val_loss
test_loss
best_val_acc
best_epoch
```

泛化指标：

```text
train_test_gap_acc
train_test_gap_loss
val_test_gap_loss
val_train_gap_loss
overfit_epoch
early_stop_gap
```

收敛指标：

```text
val_loss_auc
val_acc_auc
loss_at_10pct_steps
loss_at_25pct_steps
loss_at_50pct_steps
loss_at_75pct_steps
```

校准指标：

```text
ECE
NLL
confidence_correct_mean
confidence_wrong_mean
reliability_bins
```

KAN 表达力指标：

```text
branch/A
no_kan_acc_drop
no_kan_drop_over_adamw
kan_logit_delta_norm
kan_margin_contribution
branch_ratio_per_layer
```

几何指标：

```text
phi_prime_p95
phi_prime_max
jacobian_condition_max
curvature_energy
sobolev_norm_mean
```

### 7.5 可视化

必须生成以下图：

```text
small_data_acc_curve.png
  x = train_size
  y = test_acc_mean ± std
  color = method

small_data_ece_curve.png
  x = train_size
  y = ECE_mean ± std

small_data_auc_curve.png
  x = train_size
  y = val_loss_auc_mean

train_test_gap_curve.png
  x = train_size
  y = train_test_gap_loss

noKAN_vs_train_size.png
  x = train_size
  y = no_kan_drop_over_adamw

geometry_vs_train_size.png
  x = train_size
  y = phi_prime_p95 and jacobian_condition
```

还要生成 Pareto 图：

```text
x = test_acc
y = ECE
size = noKAN
color = method
```

以及：

```text
x = test_acc
y = phi_prime_p95
size = branch/A
color = method
```

### 7.6 判定

Small-data pass 条件：

Fashion：

$$
Acc_{U-FULL}\geq Acc_{AdamW}-0.5\%.
$$

KMNIST：

$$
Acc_{U-FULL}\geq Acc_{AdamW}-1.0\%.
$$

并且至少两个 train size 上：

$$
ECE_{U-FULL}<ECE_{AdamW}.
$$

$$
\phi'_{U-FULL}<\phi'_{AdamW}.
$$

$$
J_{U-FULL}<J_{AdamW}.
$$

如果 U-FULL 在小数据下 accuracy 稍低但 ECE / geometry / gap 更好，应报告为：

```text
small-data Pareto improvement
```

而不是 clean accuracy win。

---

## 8. P5：label-noise generalization confirm

### 8.1 目标

验证 U-FULL 是否能利用 geometry / calibration 优势减少 noisy-label 过拟合。

### 8.2 设置

```text
datasets = Fashion-MNIST, KMNIST
noise_type = symmetric label noise
noise_rate = 0.2, 0.4
train_size = 6000
val_size = 1000 clean
test_size = 1000 clean
seeds = 0,1,2,3,4
```

### 8.3 方法

```text
AdamW-alphaFixed1
U-FULL-selected
U-FULL-f085-reference, if needed
```

不加入 SNR，不加入 noise-specific optimizer。目标是验证 fixed U-FULL 的泛化，不是做 noisy-label 最优。

### 8.4 必须记录

准确率：

```text
clean_train_acc
noisy_train_acc
clean_val_acc
clean_test_acc
worst_class_acc
classwise_acc
```

loss：

```text
clean_train_loss
noisy_train_loss
val_loss
test_loss
val_loss_auc
```

noise 记忆指标：

```text
noise_memorization_rate
clean_recovery_rate
noisy_label_fit_rate
clean_label_fit_rate
memorization_gap
```

定义：

$$
\operatorname{memorization\_gap}
=
Acc_{noisy\ labels,train}
-
Acc_{clean\ labels,train}.
$$

校准：

```text
ECE
NLL
confidence_on_clean_correct
confidence_on_noisy_wrong
```

KAN / geometry：

```text
branch/A
no_kan_drop
kan_margin_contribution
phi_prime_p95
jacobian_condition
curvature_energy
```

### 8.5 可视化

```text
noise_clean_acc_bar.png
  x = method
  y = clean_test_acc
  facet = dataset / noise_rate

noise_memorization_bar.png
  x = method
  y = noise_memorization_rate

noise_ece_bar.png
  x = method
  y = ECE

noise_loss_curve.png
  y = train noisy loss, val clean loss

confidence_hist_clean_vs_noisy.png
  histogram confidence for clean-correct and noisy-wrong samples

phi_vs_noise_rate.png
  x = noise_rate
  y = phi_prime_p95

noKAN_vs_noise_rate.png
  x = noise_rate
  y = no_kan_drop
```

### 8.6 判定

Label-noise sanity pass：

Fashion：

$$
Acc_{clean,U-FULL}\geq Acc_{clean,AdamW}-0.5\%.
$$

KMNIST：

$$
Acc_{clean,U-FULL}\geq Acc_{clean,AdamW}-1.5\%.
$$

同时：

$$
ECE_{U-FULL}<ECE_{AdamW}.
$$

$$
\operatorname{memorization\_gap}_{U-FULL}
<
\operatorname{memorization\_gap}_{AdamW}.
$$

如果 clean accuracy 没赢，但 ECE / memorization 明显更好，可以写：

```text
U-FULL improves noisy-label calibration and memorization behavior, but not clean accuracy.
```

---

# 第四部分：CIFAR-small / ConvStem-DGKAN precheck

## 9. P6：CIFAR-small ConvStem-DGKAN precheck

### 9.1 目标

验证 U-FULL 是否能离开 28x28 灰度数据，进入自然图像小规模 setting。

这不是追 SOTA，而是回答：

$$
\boxed{
\text{DG-KAN + U-FULL 是否能在 ConvStem vision setting 中正常训练，并保留 geometry / calibration 优势？}
}
$$

### 9.2 模型设计

建议使用轻量 ConvStem：

```text
input image
  -> Conv 3x3, channels 32, stride 1
  -> BatchNorm or LayerNorm2d
  -> SiLU
  -> Conv 3x3, channels 64, stride 2
  -> Norm
  -> SiLU
  -> Conv 3x3, channels 96, stride 2
  -> Norm
  -> SiLU
  -> global average pooling or flatten projection
  -> DG-KAN blocks h96 / depth4 / basis24
  -> classifier head
```

为了和 Fashion/KMNIST 保持可比，DG-KAN 部分优先使用：

```text
hidden_dim = 96
depth = 4
basis_count = 24
alpha_mode = fixed1
branch_final_scale = selected U-FULL default
```

### 9.3 数据设置

```text
dataset = CIFAR-10
train_size = 10000 first
val_size = 5000
test_size = 10000
seeds = 0,1,2
```

如果 3-seed smoke 通过，再扩：

```text
train_size = 50000
seeds = 0,1,2
```

数据增强分两档：

```text
AugLite:
  random crop with padding 4
  horizontal flip
  normalize

NoAug:
  normalize only
```

先用 AugLite 作为主 setting，NoAug 作为机制审计。

### 9.4 方法

```text
ConvStem-MLP-AdamW
ConvStem-DGKAN-AdamW
ConvStem-DGKAN-U-FULL-selected
ConvStem-DGKAN-StaticFunctional
optional ConvStem-DGKAN-F-HARD-like
```

这里 `F-HARD-like` 不是主线，只用于判断是否 CIFAR 需要 early diag/branch schedule。

### 9.5 必须记录

任务指标：

```text
train_acc
val_acc
test_acc
train_loss
val_loss
test_loss
best_val_acc
best_epoch
```

收敛：

```text
val_loss_auc
val_acc_auc
steps_to_adamw_final_val_loss
time_to_adamw_final_val_loss
loss_at_10pct_steps
loss_at_25pct_steps
loss_at_50pct_steps
loss_at_75pct_steps
```

KAN 表达力：

```text
branch/A
branch_ratio_per_layer
no_kan_test_acc
no_kan_acc_drop
no_kan_drop_over_adamw
kan_logit_delta_norm
kan_margin_contribution
```

几何：

```text
phi_prime_p95
phi_prime_max
jacobian_condition_sampled
curvature_energy
sobolev_norm_mean
credit_amplification_proxy
```

CIFAR-specific：

```text
classwise_acc
confusion_matrix
augmentation_sensitivity
feature_norm_mean
feature_norm_std
representation_effective_rank
linear_probe_head_acc, optional
```

compute：

```text
step_time_ms
epoch_time_sec
samples_per_sec
peak_memory_mb
precond_solve_time_ms
metric_build_time_ms
forward_time_ms
backward_time_ms
optimizer_time_ms
```

### 9.6 可视化

```text
cifar_scorecard.png
  table-like plot for acc/AUC/ECE/geometry/time

cifar_loss_curves.png
  val loss vs epoch, mean ± std

cifar_acc_curves.png
  val acc vs epoch, mean ± std

cifar_branch_layer_curves.png
  branch ratio per layer vs epoch

cifar_noKAN_bar.png
  no-KAN drop by method

cifar_geometry_curves.png
  phi_prime_p95 and jacobian_condition vs epoch

cifar_classwise_heatmap.png
  classwise accuracy and confusion matrix

cifar_compute_breakdown.png
  stacked bar: forward/backward/optimizer/precond time

cifar_pareto_acc_time.png
  x = step_time_ms
  y = test_acc
  size = ECE reduction
  color = method
```

### 9.7 判定

CIFAR-small precheck pass：

```text
run failures = 0
NaN/Inf = 0
U-FULL test acc gap vs DGKAN-AdamW < 2%
U-FULL val AUC not worse than DGKAN-AdamW by more than 5%
phi/J geometry better than DGKAN-AdamW
branch/A between 0.4 and 1.1
no-KAN drop finite and meaningful
```

如果 U-FULL accuracy 明显低但 geometry 很好，诊断为：

```text
CIFAR branch under-expression or rest representation mismatch.
```

如果 branch/A 高且 geometry 变坏，诊断为：

```text
CIFAR branch over-expression, f1 not safe for ConvStem.
```

---

# 第五部分：true-Gram implementation speed optimization

## 10. P7：true-Gram overhead profiling

### 10.1 目标

当前 true-Gram 方法 step time 明显高于 AdamW。P7 的目标不是改变算法，而是降低实现 overhead。

### 10.2 Profiling setting

使用两个代表性 setting：

```text
Fashion-MNIST h96/d4/b24
CIFAR-small ConvStem-DGKAN h96/d4/b24
```

方法：

```text
AdamW
U-FULL-selected
```

### 10.3 计时分解

必须记录：

```text
total_step_time_ms
data_load_time_ms
forward_time_ms
loss_time_ms
backward_time_ms
kan_coeff_grad_time_ms
precond_solve_time_ms
nonkan_optimizer_time_ms
metric_build_time_ms
eval_audit_time_ms
```

显存：

```text
peak_allocated_mb
peak_reserved_mb
activation_memory_estimate
optimizer_state_memory_mb
kan_coeff_memory_mb
metric_cache_memory_mb
```

### 10.4 优化候选

#### 10.4.1 Gram cache 固定化

因为 centers、width、basis_count 固定，Gram 可缓存：

```text
cache key:
  basis_count
  width
  alpha_geo
  beta_geo
  rho
  dtype
  device
```

期望：

```text
metric_build_time_ms 接近 0。
```

#### 10.4.2 batched Cholesky solve 优化

当前 gradient shape：

```text
out_dim x in_dim x basis_count
```

reshape 为：

```text
(out_dim * in_dim) x basis_count
```

统一 solve：

$$
D = G A^{-1}.
$$

需要比较：

```text
torch.cholesky_solve
torch.linalg.solve_triangular twice
precomputed inverse matmul
```

因为 basis_count 只有 24，预计算 inverse 可能更快：

$$
D = G A^{-1}.
$$

但要审计数值误差：

$$
\frac{\|D_{inv}-D_{chol}\|}{\|D_{chol}\|+\epsilon}.
$$

#### 10.4.3 mixed precision audit

测试：

```text
model fp32, metric fp32
model amp, metric fp32
model amp, metric fp16 not recommended unless stable
```

记录：

```text
acc
AUC
phi/J
NaN
direction_relerr
step_time
memory
```

#### 10.4.4 audit frequency reduction

很多 geometry audit 很慢。测试：

```text
audit_every_epoch = 1
audit_every_epoch = 2
audit_every_epoch = 5
```

训练结果不应变，因为 U-FULL full-start 不依赖 online trigger。audit frequency 只影响 logging 成本。

#### 10.4.5 einsum rewrite

RBF Dense 当前可能用：

```text
basis: batch x in x basis
coeff: out x in x basis
einsum -> batch x out
```

测试是否可以 rewrite 为：

```text
basis_flat = basis.reshape(batch, in*basis)
coeff_flat = coeff.reshape(out, in*basis)
out = basis_flat @ coeff_flat.T
```

记录 forward speed 与数值误差。

### 10.5 可视化

```text
speed_breakdown_before_after.png
  stacked bar by method

precond_solve_method_bar.png
  compare cholesky_solve / inverse_matmul / triangular

step_time_vs_acc.png
  x = step_time_ms
  y = test_acc

memory_breakdown.png
  stacked memory components

direction_error_vs_speed.png
  x = direction_relerr
  y = speedup
```

### 10.6 判定

Speed optimization pass：

$$
\text{step_time_ratio}_{U-FULL}
<
1.20\times\text{AdamW}
$$

或者至少：

$$
\text{optimizer overhead reduced by }30\%.
$$

如果 accuracy/AUC/geometry 变化超过 small tolerance，则优化不能采用。

Tolerance：

```text
acc change < 0.2%
AUC change < 1%
phi/J reduction change < 5%
no NaN / Inf
```

---

# 第六部分：Rational / KAT 替代 RBF 的路线

## 11. P8：Rational / KAT scaling precheck

### 11.1 目标

RBF-DGKAN 表达能力强，但参数和计算重：

$$
O(c_{out}c_{in}K).
$$

Rational / KAT 的结构更像：

```text
group rational activation -> Linear
```

参数量接近：

$$
O(c_{out}c_{in}+g(m+n)).
$$

P8 的目标是验证：

$$
\boxed{
\text{Rational/KAT 是否能保留足够的 DG-KAN branch 表达力，同时显著降低参数、显存和 step time。}
}
$$

这不是直接替代当前 RBF 结论，而是 scaling 路线预检。

### 11.2 Rational branch 定义

输入 hidden：

$$
x\in\mathbb R^d.
$$

按 group 分组：

$$
x = (x^{(1)},\dots,x^{(g)}).
$$

每组使用一个 rational function：

$$
r_q(t)
=
\frac{
p_q(t)
}{
1+|q_q(t)|
}.
$$

其中：

$$
p_q(t)=\sum_{i=0}^{m}a_{qi}t^i.
$$

$$
q_q(t)=\sum_{j=1}^{n}b_{qj}t^j.
$$

Rational branch：

$$
u = r(x)
$$

然后：

$$
y = W u + b.
$$

Residual block：

$$
h_{k+1}=h_k+\alpha b_t W r(\operatorname{LN}(h_k)).
$$

v3.5 默认：

$$
\alpha=1,\quad b_t=1.
$$

### 11.3 方法矩阵

先在 Fashion/KMNIST 小规模验证：

```text
RBF-U-FULL-f100
RBF-U-FULL-f085-reference
Rational-DGKAN-AdamW
Rational-DGKAN-Hybrid
Rational-DGKAN-FunctionalRational, optional
MLP-AdamW
```

其中：

```text
Rational-DGKAN-Hybrid:
  rational parameters and linear W initially use AdamW
  optional only KAN-like branch scale / geometry audit

Rational-DGKAN-FunctionalRational:
  rational numerator/denominator parameters use diagonal Fisher or coefficient norm preconditioner
  only after AdamW version is stable
```

不建议一开始给 rational denominator 做激进 functional update，因为 denominator 可能带来数值不稳定。

### 11.4 Rational-specific safety metrics

必须记录：

```text
rational/denominator_min
rational/denominator_p01
rational/denominator_p05
rational/r_prime_p95
rational/r_prime_max
rational/r_double_prime_p95
rational/group_function_diversity
rational/group_output_norm
rational/numerator_norm
rational/denominator_coeff_norm
rational/saturation_fraction
```

其中 denominator safety：

$$
den(t)=1+|q(t)|.
$$

要求：

$$
\min den(t) > \epsilon.
$$

虽然定义中有 `1+abs`，仍需检查是否出现过大 derivative 或 saturation。

### 11.5 对比指标

参数 / 计算：

```text
param_count
trainable_param_count
KAN_or_Rational_branch_param_count
FLOPs_estimate
activation_memory_mb
step_time_ms
optimizer_state_memory_mb
peak_memory_mb
```

任务：

```text
test_acc
val_loss_auc
ECE
NLL
classwise_acc
```

表达力：

```text
branch/A
no_kan_drop
no_kan_drop_over_adamw
logit_delta_norm
margin_contribution
```

几何：

```text
RBF:
  phi_prime_p95
  jacobian_condition

Rational:
  r_prime_p95
  sampled_jacobian_condition
```

### 11.6 Fashion/KMNIST 判定

Rational 进入 CIFAR precheck 的条件：

```text
acc gap vs RBF-U-FULL < 1.0%
AUC not worse than RBF-U-FULL by more than 5%
step_time < 0.7 x RBF-U-FULL
peak_memory < 0.7 x RBF-U-FULL
ECE not worse than RBF by more than 5%
no denominator / derivative instability
```

如果 Rational accuracy 明显低，但 speed/memory 大幅好，则定位为：

```text
engineering scaling candidate
```

而不是 replacement。

### 11.7 CIFAR Rational precheck

如果 Fashion/KMNIST 通过，再上 CIFAR-small：

```text
ConvStem-RBF-DGKAN-U-FULL
ConvStem-Rational-DGKAN-AdamW
ConvStem-Rational-DGKAN-Hybrid
```

目标不是追 SOTA，而是看 scaling 是否更顺：

```text
step_time
memory
stability
branch expression
accuracy gap
```

### 11.8 可视化

```text
param_vs_acc.png
  x = parameter_count
  y = test_acc

step_time_vs_acc.png
  x = step_time_ms
  y = test_acc

memory_vs_acc.png
  x = peak_memory
  y = test_acc

branch_expression_compare.png
  y = no_kan_drop_over_adamw

rational_denominator_safety.png
  denominator_min / p01 over epochs

rational_derivative_curve.png
  r_prime_p95 and r_prime_max over epochs

rational_group_diversity_heatmap.png
  group x layer diversity
```

---

# 第七部分：统一 logging 与 failure table

## 12. 所有实验必须记录的核心字段

每个 run 至少记录：

```text
dataset
method
profile_name
seed
train_size
val_size
test_size
epochs
model_type
hidden_dim
depth
basis_count
alpha_mode
branch_final_scale
kan_primitive_type
optimizer_structure
kan_coeff_update
nonkan_update
```

任务：

```text
train_acc_final
val_acc_final
test_acc_final
train_loss_final
val_loss_final
test_loss_final
val_loss_auc
val_acc_auc
best_val_acc
best_epoch
```

比较：

```text
acc_gap_vs_adamw
val_auc_improvement_vs_adamw
ece_reduction_vs_adamw
phi_reduction_vs_adamw
jac_reduction_vs_adamw
branch_over_adamw
no_kan_drop_over_adamw
```

几何：

```text
phi_prime_p95
phi_prime_max
jacobian_condition_max
curvature_energy
sobolev_norm_mean
```

表达力：

```text
branch_ratio_mean
branch_ratio_per_layer
no_kan_acc_drop
kan_logit_delta_norm_mean
kan_margin_contribution_mean
```

校准：

```text
ECE
NLL
confidence_correct_mean
confidence_wrong_mean
```

计算：

```text
step_time_ms
epoch_time_sec
total_train_time_sec
samples_per_sec
peak_memory_mb
precond_solve_time_ms
metric_build_time_ms
```

---

## 13. Failure tags

每个 run 需要自动打标签：

| failure tag | 条件 |
|---|---|
| `run_failed` | 训练异常 |
| `nan_or_inf` | 出现 NaN / Inf |
| `accuracy_gap_large` | acc gap 超过阈值 |
| `auc_no_gain` | AUC improvement 小于 0 |
| `expression_low` | noKAN ratio 小于 0.7 |
| `branch_overactive` | branch/A 大于 1.1 |
| `branch_underactive` | branch/A 小于 0.45 |
| `geometry_bad` | phi/J reduction 未达阈值 |
| `ece_worse` | ECE 差于 AdamW |
| `wallclock_slow` | time-to-target 慢于 AdamW 且无其他明显收益 |
| `stem_underfit` | stem functional / ConvStem 导致 train acc 明显低 |
| `rational_denominator_risk` | rational denominator / derivative 异常 |
| `speed_regression` | step time 未改善反而更慢 |

输出：

```text
failure_table.csv
failure_summary_by_method.csv
failure_summary_by_dataset.csv
```

可视化：

```text
failure_counts_by_method.png
failure_counts_by_dataset.png
failure_cooccurrence_heatmap.png
```

---

# 第八部分：最终决策规则

## 14. 决策一：U-FULL-f100 是否成为默认

### f100 成为默认

如果 P3 10-seed 满足：

```text
Fashion clean pass
KMNIST medium or clean pass
noKAN ratio > 0.70 on both
AUC > 8% on both
geometry and ECE passed
```

则：

```text
default unified profile = U-FULL-f100
```

### f085 保持默认

如果 f100 expression 更强但 accuracy/AUC 弱于 f085，则：

```text
default unified profile = U-FULL-f085
f100 = expression-heavy ablation
```

### dataset-specific 仍保留

如果 Fashion hard 明显优于 U-FULL，则：

```text
Fashion best profile = F-V3-HARD-base30
Unified profile = U-FULL-selected
```

这两者可以同时存在。

---

## 15. 决策二：是否推进 CIFAR

CIFAR precheck 进入下一阶段的条件：

```text
U-FULL selected does not collapse.
acc gap vs DGKAN-AdamW < 2%.
geometry/ECE positive.
branch/A finite and interpretable.
step time overhead documented.
```

如果 CIFAR 上 U-FULL 失败，优先诊断：

```text
ConvStem representation
RBF branch compute cost
branch scale f1 是否过强
capacity / head mismatch
```

不要立刻回到 optimizer 大扫。

---

## 16. 决策三：是否推进 Rational/KAT

Rational/KAT 进入 scaling 主线的条件：

```text
step_time and memory significantly better than RBF
accuracy gap acceptable
no denominator / derivative instability
branch expression meaningful
```

如果 Rational 速度好但 accuracy 差，则保留为 engineering candidate。

如果 Rational accuracy 接近 RBF，则：

```text
Stage scaling default may switch from RBF-DGKAN to Rational-DGKAN.
```

---

# 第九部分：建议执行顺序

## 17. 最小执行顺序

建议按以下顺序运行，不要并行大扫：

```text
P0: config freeze
P1: U-FULL-f100 seed0 sanity
P2: U-FULL-f100 5-seed confirm
P3: U-FULL-f100 10-seed final, only if P2 passes
P4: small-data confirm
P5: label-noise confirm
P6: CIFAR-small precheck
P7: speed profiling and optimization
P8: Rational/KAT precheck
```

如果 P1 中 f100 明显失败，则立即停止 f100 confirm，回到：

```text
U-FULL-f085 as default
```

然后继续 P4-P8。

---

## 18. 本轮不再做什么

本轮明确不再做：

```text
Full AFU
StemOnly functional update
PGAdam / NormPGAdam / PostAdam
HeadCov as main optimizer
LNOnly as main optimizer
branch_final_scale large sweep
momentum / SNR / FSAM / data metric
```

这些已经是边界或支线，不进入主线。

---

# 19. 最终预期输出

最终应输出：

```text
docs/DG-KAN_Optimizer_v3.5_UFULL_f1_结果复盘.md

results/gafu_v3_5_p1_seed0/
results/gafu_v3_5_p2_confirm5/
results/gafu_v3_5_p3_confirm10/
results/gafu_v3_5_p4_small_data/
results/gafu_v3_5_p5_label_noise/
results/gafu_v3_5_p6_cifar_small/
results/gafu_v3_5_p7_speed/
results/gafu_v3_5_p8_rational/
```

每个目录包含：

```text
runs.csv
summary_by_method.csv
paired_delta.csv
failure_table.csv
curves/
plots/
aggregate_summary.json
```

最终报告必须回答：

```text
1. branch_final_scale=1.0 能否成为 U-FULL 默认？
2. U-FULL 在 small-data / label-noise 下是否仍保持 Pareto 优势？
3. U-FULL 能否进入 CIFAR-small / ConvStem-DGKAN？
4. true-Gram overhead 能否优化到可接受范围？
5. Rational/KAT 是否值得替代 RBF 进入 scaling？
```

---

# 20. 最终一句话

v3.5 的核心不是继续发明 optimizer，而是验证一个更自然的默认：

$$
\boxed{
\text{U-FULL-f100}
=
\text{full Sobolev from start}
+
\alpha=1
+
b=1
+
\text{Hybrid optimizer}.
}
$$

如果它通过，DG-KAN 的 optimizer 叙事会更简单：

```text
KAN branch 不再需要人为缩放；
KAN edge 用函数空间 Sobolev update；
普通神经网络参数用 AdamW；
统一 profile 可用于后续泛化与 scaling。
```

如果它不通过，则 v3.4 的结论仍然稳固：

```text
U-FULL-f085 是统一 Pareto profile；
f100 只是 expression-heavy ablation。
```
