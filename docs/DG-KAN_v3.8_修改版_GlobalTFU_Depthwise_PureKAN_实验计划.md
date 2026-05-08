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

