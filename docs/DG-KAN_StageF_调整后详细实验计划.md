# DG-KAN Stage F 调整后详细实验计划：收敛动力学与优化效率

> 文件名：`DG-KAN_StageF_调整后详细实验计划.md`  
> 版本：v1.0  
> 目标：在 Stage D/E 已经打通 `AnalyticAdj / FirstOrderDecomp + functional update` 的基础上，系统验证 DG-KAN functional-space update 是否带来更快收敛、更稳定优化、更健康几何，以及这些收益在视觉分类任务中是否真实存在。  
> 公式格式：Typora 友好，统一使用 `$...$` 和 `$$...$$`。

---

## 0. Stage F 的重新定位

Stage F 不再被定义为“证明 functional update 一定比 AdamW 快”。根据目前 Stage D/E 的结果，更准确的定位应该是：

> **研究 DG-KAN functional-space update 的收敛动力学，找出它在什么条件下比 AdamW 更快、更稳，什么条件下只是几何更好但最终 accuracy 不够。**

当前已经明确的前提是：

1. `AnalyticAdj-DGKAN` 与 `FullBP-DGKAN` 在 MNIST / Fashion-MNIST / KMNIST 上训练结果几乎等价，说明 analytic primitive-adjoint 已经可以作为主 credit 实现。
2. `FirstOrderDecomp-DGKAN` 已经完成真正 joint training，相对 `AnalyticAdj-DGKAN` 的 accuracy gap 小于 $1\%$，可以作为低成本候选。
3. `DGKAN-AdamW` 在 8 epoch 分类实验中 accuracy 仍然更高，但它的 branch ratio、$\phi'_{p95}$ 和 Jacobian condition 更激进。
4. Stage E 显示 `AnalyticAdj-DGKAN` 是当前最强 low-memory 路线，peak memory 相对 FullBP 降低约 $44.5\%$，而 dense `SufficientStats` 当前显存收益不够强。

因此 Stage F 的核心问题不是 credit 是否正确，而是：

$$
\boxed{
\text{functional-space update 能否在保持更健康几何的同时，追上或改善 AdamW 的收敛效率？}
}
$$

---

## 1. Stage F 要回答的具体问题

Stage F 要回答六个问题。

### 1.1 Functional update 是否更快收敛？

这里的“更快”不能只看最终 accuracy，而要看：

$$
\text{loss AUC}
$$

$$
\text{steps-to-target}
$$

$$
\text{time-to-target}
$$

如果一个方法 final accuracy 接近 AdamW，但 val loss AUC 更低，说明它在训练过程中更快达到较好区域。反过来，如果 final accuracy 高但 AUC 差，说明它可能只是后期追上。

---

### 1.2 Functional update 是否允许更大学习率？

Stage B 的函数拟合实验显示 `diag` 和 `diag-to-Sobolev` 可能改善早期优化。Stage F 要在分类任务中验证：

$$
\eta_{max}^{functional} > \eta_{max}^{AdamW}
$$

是否成立。

这里的 $\eta_{max}$ 是 largest stable learning rate，即在不发散、不出现严重 loss spike、最终性能不崩的前提下可使用的最大学习率。

---

### 1.3 Functional update 的几何优势是否仍然存在？

D0/D1 已经显示 AdamW 更激进，通常有更高的 branch ratio、$\phi'_{p95}$ 和 Jacobian condition。Stage F 要看在优化更充分后，functional update 是否仍然保持：

$$
\phi_{p95}^{' func}<\phi_{p95}^{' AdamW}
$$

$$
\kappa(J)^{func}<\kappa(J)^{AdamW}
$$

$$
A_{p95}^{func}<A_{p95}^{AdamW}
$$

其中 credit amplification 为：

$$
A_k=\frac{\|g_k\|}{\|g_{k+1}\|+\epsilon}
$$

---

### 1.4 Functional update 的性能差距来自欠拟合、branch under-active，还是超参没调好？

Stage B/C 中多次出现 branch under-active。Stage F 必须持续记录：

$$
r_{branch,k}
=
\frac{
\|\alpha_k\operatorname{KAN}_k(\operatorname{LN}(h_k))\|
}{
\|h_k\|+\epsilon
}
$$

以及 no-KAN ablation：

$$
\Delta_{noKAN}
=
\operatorname{Acc}_{full}-\operatorname{Acc}_{noKAN}
$$

如果 functional update accuracy 低，同时 branch ratio 和 no-KAN drop 都低，说明问题不是泛化，而是 KAN branch 没被充分利用。

---

### 1.5 AnalyticAdj / FirstOrderDecomp 对收敛速度有没有影响？

Credit 侧要比较：

| 方法 | 作用 |
|---|---|
| FullBP-DGKAN | autograd BP baseline |
| AnalyticAdj-DGKAN | 主 credit 实现 |
| FirstOrderDecomp-DGKAN | 低成本近似候选 |
| IdentityAdj-DGKAN | 强 baseline，测试 identity credit 是否足够 |

如果这些方法收敛曲线几乎一致，说明收敛瓶颈主要在 update geometry，而不是 credit transport。

---

### 1.6 Functional update 的收益是否能推广到更真实视觉任务？

Stage F 主任务仍然是 MNIST / Fashion-MNIST / KMNIST，因为已有 D/E 结果可对齐。但 Stage F 需要设计一个进入 CIFAR-10 small 的门槛。只有当 Fashion / KMNIST 至少达到 medium pass，才进入 CIFAR-10 small。

---

## 2. 方法定义

Stage F 比较以下方法。

### 2.1 AdamW baseline

所有参数使用 AdamW：

$$
\theta\leftarrow\theta-\eta\operatorname{AdamW}(\nabla_\theta\mathcal L)
$$

这是当前最强 accuracy baseline。

---

### 2.2 FullBP functional update

使用普通 autograd full BP credit，但 KAN coefficients 使用 functional update：

$$
a\leftarrow a-\eta(M+\rho I)^{-1}\nabla_a\mathcal L
$$

非 KAN 参数使用 AdamW。

---

### 2.3 AnalyticAdj functional update

使用 KAN analytic adjoint 替代 KAN primitive autograd backward：

$$
g_{x_i}=\sum_jg_{y_j}\phi'_{ji}(x_i)
$$

KAN coefficients 使用 functional update，非 KAN 参数使用 AdamW。

这是 Stage F 的主方法。

---

### 2.4 FirstOrderDecomp functional update

使用 span-2 first-order approximation：

$$
g_k^{first}=g_{k+2}+\Delta_k(g_{k+2})+\Delta_{k+1}(g_{k+2})
$$

忽略 cross term。KAN coefficients 使用 functional update，非 KAN 参数使用 AdamW。

这是低成本候选，不是默认主方法。

---

### 2.5 IdentityAdj functional update

直接使用 identity credit：

$$
g_k=g_{k+1}
$$

它是强 baseline，用来判断 residual DG-KAN 是否已经把 credit transport 稳定到 identity 附近。

---

### 2.6 Functional update variants

Stage F 至少比较以下 update：

| update | 定义 | 作用 |
|---|---|---|
| `diag` | $\operatorname{diag}(M+\rho I)^{-1}\nabla_a\mathcal L$ | early speed |
| `sobolev` | $(M+\rho I)^{-1}\nabla_a\mathcal L$ | final quality / geometry |
| `diag_to_sobolev` | 先 diag 后 Sobolev | 当前默认 |
| `coeff_adamw` | KAN coeff 也用 AdamW | strong optimizer baseline |

`diag_to_sobolev` 定义为：

前 $T_w$ steps：

$$
d_t=\operatorname{diag}(M+\rho I)^{-1}\nabla_a\mathcal L
$$

之后：

$$
d_t=(M+\rho I)^{-1}\nabla_a\mathcal L
$$

---

## 3. 数据集与模型配置

### 3.1 Stage F 主数据集

Stage F 第一轮使用：

| 数据集 | 作用 |
|---|---|
| MNIST | 简单任务，快速看收敛速度 |
| Fashion-MNIST | 中等视觉分类，主诊断任务 |
| KMNIST | 更难分类任务，暴露 functional update 与 AdamW 差距 |

每个数据集使用与 Stage D 对齐的 split：

```text
train / val / test = 6000 / 1000 / 1000
```

正式确认使用 $5$ seeds：

```text
seeds = 0,1,2,3,4
```

探索阶段可使用 $2$ 或 $3$ seeds。

---

### 3.2 Stage F 主模型

与 Stage D/D1 对齐：

```text
model = residual DG-KAN
depth = 4
hidden_dim = 64
basis_count = 16
alpha_init = 1.5
batch_size = 256
epochs = 8 / 20
```

其中 block 为：

$$
h_{k+1}=h_k+\alpha_k\operatorname{KAN}_k(\operatorname{LN}(h_k))
$$

KAN edge function 为：

$$
\phi_{ji}(t)=\sum_m a_{jim}B_m(t)
$$

---

### 3.3 Stage F 扩展模型

如果主模型上发现 medium pass，再扩展：

```text
depth = 8
hidden_dim = 128
basis_count = 16
```

这个设置与 Stage E memory accounting 对齐。

---

### 3.4 CIFAR-10 small 进入条件

只有当 F1/F2 在 Fashion-MNIST 或 KMNIST 上满足 medium pass，才进入 CIFAR-10 small。

CIFAR-10 small 初始模型：

```text
Conv stem -> residual DG-KAN blocks -> linear head
hidden_dim = 128
depth = 4
basis_count = 16
connectivity = grouped or dense small
```

---

## 4. Stage F 分阶段实验设计

Stage F 分为 F0 到 F5。每个阶段有不同作用。

---

## 4.1 F0：复现与 baseline lock

### 目标

确认 Stage D/D1 中关键 baseline 可复现，并固定比较基准。

### 方法

使用 Stage D/D1 当前配置：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
methods = AdamW, FullBP-functional, AnalyticAdj-functional, FirstOrderDecomp-functional, IdentityAdj-functional
update = diag_to_sobolev for functional methods
```

训练：

```text
epochs = 8
seeds = 0,1,2,3,4
```

### 必须确认

1. `AnalyticAdj-DGKAN` 与 `FullBP-DGKAN` 接近；
2. `FirstOrderDecomp-DGKAN` 相对 AnalyticAdj gap < $1\%$；
3. `DGKAN-AdamW` 仍然是强 accuracy baseline；
4. functional methods 的 branch ratio、$\phi'_{p95}$、Jacobian condition 低于 AdamW 或至少更稳。

### F0 通过标准

$$
\text{test acc gap}_{AnalyticAdj,FullBP}<0.5\%
$$

$$
\text{test acc gap}_{FirstOrder,AnalyticAdj}<1\%
$$

否则不能进入大规模 Stage F sweep。

---

## 4.2 F1：Functional update regime sweep

### 目标

找出 functional update 在视觉分类任务上的最佳速度-精度-几何 Pareto 区域。

### 核心问题

现有 D0/D1 显示 AdamW accuracy 更高，但几何更激进。F1 要回答：

> 通过调 `coeff_lr`、`rest_lr`、`alpha`、`warmup_frac`、Sobolev metric，能否让 functional update 接近 AdamW accuracy，同时保持几何优势？

---

### 搜索空间

为了避免 full grid 爆炸，F1 分两层。

#### F1-search：粗搜索

```text
datasets = Fashion-MNIST, KMNIST
method = AnalyticAdj-DGKAN
update = diag_to_sobolev
seeds = 0,1
```

搜索：

```text
coeff_lr = 0.3, 0.5, 0.7, 1.0
rest_lr = 0.001, 0.003, 0.006
alpha_init = 1.0, 1.5, 2.0
warmup_frac = 0.05, 0.10, 0.25
sobolev_alpha = 0, 1e-3, 1e-2, 1e-1
rho = 1e-4, 1e-3, 1e-2
```

不要全排列。采用 staged search：

1. 固定 `sobolev_alpha=1e-2, rho=1e-3`，扫 `coeff_lr/rest_lr/alpha/warmup`；
2. 选 top 5 config 后扫 `sobolev_alpha/rho`；
3. 每个 dataset 保留 top 3 config 进入 F1-confirm。

---

#### F1-confirm：5-seed 复核

```text
datasets = MNIST, Fashion-MNIST, KMNIST
methods = selected functional configs + AdamW baseline
seeds = 0,1,2,3,4
epochs = 8 and 20
```

同时跑 8 epoch 和 20 epoch 是为了区分：

- early convergence advantage；
- final performance advantage；
- 是否只是训练更慢。

---

### F1 选择标准

每个候选 config 都计算：

$$
S=	ext{AccScore}+\lambda_1\text{AUCScore}+\lambda_2\text{GeometryScore}+\lambda_3\text{BranchScore}
$$

其中：

$$
\text{AccScore}=-\max(0,\operatorname{Acc}_{AdamW}-\operatorname{Acc}_{method})
$$

$$
\text{AUCScore}=\frac{\operatorname{AUC}_{AdamW}-\operatorname{AUC}_{method}}{\operatorname{AUC}_{AdamW}+\epsilon}
$$

$$
\text{GeometryScore}=\frac{1}{3}
\left(
\frac{\phi'_{AdamW}-\phi'_{method}}{\phi'_{AdamW}+\epsilon}
+
\frac{\kappa_{AdamW}-\kappa_{method}}{\kappa_{AdamW}+\epsilon}
+
\frac{A_{AdamW}-A_{method}}{A_{AdamW}+\epsilon}
\right)
$$

BranchScore 鼓励 branch 既不 under-active 也不过强：

$$
\text{BranchScore}=-\left|\frac{r_{branch}^{method}}{r_{branch}^{AdamW}}-r^*\right|
$$

推荐：

$$
r^*\in[0.4,0.8]
$$

---

### F1 成功标准

Weak：找到至少一个 functional config 满足：

$$
\text{test acc gap vs AdamW}<2\%
$$

且：

$$
\phi_{p95}^{' func}<\phi_{p95}^{' AdamW}
$$

Medium：在 Fashion 或 KMNIST 上满足：

$$
\text{test acc gap vs AdamW}<1.5\%
$$

且：

$$
\text{val AUC improvement}>5\%
$$

或：

$$
\text{time-to-target improvement}>10\%
$$

Strong：在 MNIST / Fashion / KMNIST 三个数据集上同时满足：

$$
\text{test acc gap vs AdamW}<1\%
$$

$$
\text{val AUC improvement}>10\%
$$

$$
\phi'_{p95}\text{ reduction}>20\%
$$

$$
\kappa(J)\text{ reduction}>20\%
$$

---

## 4.3 F2：正式收敛速度比较

### 目标

对 F1 选出的配置进行公平收敛比较。

### 对比方法

| 方法 | 说明 |
|---|---|
| DGKAN-AdamW | 强 accuracy baseline |
| FullBP-DGKAN + diag-to-Sobolev | functional update baseline |
| AnalyticAdj-DGKAN + diag-to-Sobolev | 主方法 |
| FirstOrderDecomp-DGKAN + diag-to-Sobolev | 低成本候选 |
| IdentityAdj-DGKAN + diag-to-Sobolev | credit simplification baseline |
| DGKAN-Diag | early speed baseline |
| DGKAN-Sobolev | final-quality baseline |

---

### 三类公平比较

#### 1. Step-matched

所有方法训练相同步数：

```text
steps = fixed by epochs = 8 / 20
```

报告 final metrics 和 loss AUC。

---

#### 2. Wall-clock-matched

每个方法训练相同 wall-clock 时间：

```text
budget_sec = AdamW 8 epoch wall-clock
```

报告 budget 内最佳 val/test 性能。

---

#### 3. Target-matched

定义目标：

$$
\operatorname{Acc}_{target}=\operatorname{Acc}_{AdamW}^{final}-\delta
$$

推荐：

- MNIST: $\delta=0.5\%$；
- Fashion-MNIST: $\delta=1.0\%$；
- KMNIST: $\delta=1.5\%$。

也定义 loss 目标：

$$
\mathcal L_{target}=1.05\times\mathcal L_{AdamW}^{final}
$$

记录：

$$
\text{steps-to-target}
$$

$$
\text{time-to-target}
$$

没有达到目标的 run 记录为 censored，并在 W&B 里单独标记。

---

### F2 关键输出

F2 最终必须产出以下表格：

| dataset | method | final acc | val AUC | steps-to-target | time-to-target | step ms | peak MB | $\phi'_{p95}$ | J cond | branch / AdamW |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

---

## 4.4 F3：Largest stable LR / stability sweep

### 目标

验证 functional update 是否比 AdamW 更稳定，能否承受更大 update strength。

### 扫描对象

对每个方法扫主学习率：

```text
scale = 0.25, 0.5, 1.0, 2.0, 4.0
```

其中：

$$
\eta_{coeff}=\eta_{coeff}^{base}\times scale
$$

$$
\eta_{rest}=\eta_{rest}^{base}\times scale
$$

也可以分开扫：

```text
coeff_lr_scale = 0.5,1,2,4
rest_lr_scale = 0.5,1,2
```

---

### 发散判定

一个 run 被判定为 unstable，如果满足任一条件：

$$
\text{NaN or Inf count}>0
$$

$$
\mathcal L_t > 5\times \mathcal L_0
$$

$$
\phi^{'}_{p95}>5\times\phi _{p95}^{' baseline}
$$

$$
\kappa(J)>10\times\kappa(J)^{baseline}
$$

$$
\text{loss spike count}>N_{spike}
$$

默认：

$$
N_{spike}=5
$$

---

### F3 输出

每个方法记录：

```text
conv/largest_stable_coeff_lr
conv/largest_stable_rest_lr
stability/divergence_rate
stability/loss_spike_count
stability/bad_steps
stability/max_phi_prime_p95
stability/max_jacobian_condition
```

### F3 成功标准

Functional update 的 largest stable coeff lr 至少比 AdamW 高：

$$
\eta_{max}^{func} > \eta_{max}^{AdamW}
$$

或者在相同 learning-rate scale 下有更低：

$$
\text{loss spike count}
$$

$$
\phi'_{p95}
$$

$$
\kappa(J)
$$

---

## 4.5 F4：收敛机制审计

### 目标

解释 functional update 为什么快或不快。F4 不主要跑新模型，而是对 F0–F3 的 run 做机制分析。

---

### 核心分析 1：branch utilization 与收敛

画：

$$
r_{branch}/r_{branch}^{AdamW}
\quad\text{vs}\quad
\text{val AUC}
$$

以及：

$$
r_{branch}/r_{branch}^{AdamW}
\quad\text{vs}\quad
\text{test acc}
$$

判断：

- branch 太低：under-active；
- branch 太高：over-active；
- 中间区间：Pareto。

---

### 核心分析 2：函数复杂度与泛化 / 收敛

画：

$$
\phi'_{p95}\quad\text{vs}\quad\text{val AUC}
$$

$$
\int|\phi''|^2\quad\text{vs}\quad\text{test acc}
$$

$$
\kappa(J)\quad\text{vs}\quad\text{loss spike count}
$$

---

### 核心分析 3：descent quality

记录并分析：

$$
\Delta\mathcal L_{pred}
$$

$$
\Delta\mathcal L_{actual}
$$

$$
r_{descent}=\frac{\Delta\mathcal L_{actual}}{\Delta\mathcal L_{pred}+\epsilon}
$$

W&B keys：

```text
functional/descent_ratio_mean
functional/descent_ratio_median
functional/descent_ratio_p10
functional/descent_pred_actual_corr
functional/bad_step_total_positive_loss_increase
```

---

### 核心分析 4：AdamW 与 functional update 的 tradeoff

特别关注 AdamW 是否只是通过更激进 branch 获得 accuracy。

需要画：

1. test acc vs $\phi'_{p95}$；
2. test acc vs Jacobian condition；
3. branch ratio vs no-KAN drop；
4. no-KAN drop vs test acc；
5. val AUC vs branch ratio。

如果 AdamW accuracy 高，但 $\phi'$、J cond、credit amplification 明显更高，则说明 AdamW 在用更激进的几何换性能。

---

## 4.6 F5：CIFAR-10 small 进入实验

### 目标

在 Fashion / KMNIST 上找到可用 functional regime 后，进入 CIFAR-10 small。

### 进入条件

满足以下任一条件才进入 F5：

1. Fashion / KMNIST 上 functional update 达到 medium pass；
2. 或者在至少一个数据集上 time-to-target improvement > $10\%$，且 test acc gap < $2\%$。

---

### 模型

```text
Conv stem -> residual DG-KAN blocks -> linear head
hidden_dim = 128
depth = 4
basis_count = 16
alpha_init = best from F1
update = best from F1/F2
credit = AnalyticAdj
```

---

### CIFAR baseline

| 方法 | 说明 |
|---|---|
| small CNN | 最低 baseline |
| ResMLP / MLP-Mixer small | token/channel mixing baseline |
| ConvStem-DGKAN-AdamW | strong optimizer baseline |
| ConvStem-DGKAN-AnalyticAdj-Func | 主方法 |
| ConvStem-DGKAN-FirstOrder-Func | 低成本候选 |

---

### CIFAR 指标

除 Stage F 通用指标外，记录：

```text
vision/test_acc
vision/train_test_gap
vision/corruption_acc_gaussian
vision/corruption_acc_blur
vision/corruption_acc_rotation
vision/ece
vision/params
vision/flops
vision/throughput
```

F5 不是要求马上超过 ResNet，而是验证 DG-KAN functional update 在自然图像上是否仍然稳定可训练。

---

## 5. W&B config 规范

所有 Stage F runs 必须记录以下 config。

```text
stage = F
phase = F0 / F1_search / F1_confirm / F2 / F3 / F4 / F5
dataset = mnist / fashion_mnist / kmnist / cifar10_small
seed = 0..4
model_family = residual_dgkan / convstem_dgkan
depth
hidden_dim
basis_count
alpha_init
alpha_learnable
primitive_mode = autograd / analytic
credit_mode = fullbp / identity / analytic / first_order
update_type = adamw / diag / sobolev / diag_to_sobolev
coeff_lr
rest_lr
warmup_frac
sobolev_alpha
sobolev_beta
rho
batch_size
epochs
scheduler_type
lr_scale
comparison_mode = step_matched / wall_clock_matched / target_matched
```

---

## 6. W&B metrics 详细规范

### 6.1 收敛指标

```text
conv/loss_auc_train
conv/loss_auc_val
conv/acc_auc_val
conv/steps_to_target_val_loss
conv/steps_to_target_val_acc
conv/time_to_target_val_loss
conv/time_to_target_val_acc
conv/loss_at_10pct_steps
conv/loss_at_25pct_steps
conv/loss_at_50pct_steps
conv/loss_at_75pct_steps
conv/val_acc_at_10pct_steps
conv/val_acc_at_25pct_steps
conv/val_acc_at_50pct_steps
conv/val_acc_at_75pct_steps
conv/plateau_step
conv/plateau_val_loss
```

---

### 6.2 任务性能

```text
train/loss
val/loss
test/loss
train/acc
val/acc
test/acc
compare/test_acc_gap_vs_adamw
compare/test_acc_gap_vs_fullbp_func
compare/test_acc_gap_vs_analytic_func
compare/val_auc_gap_vs_adamw
compare/val_auc_gap_vs_analytic_func
```

---

### 6.3 时间与显存

```text
perf/step_time_ms
perf/samples_per_sec
perf/time_total_sec
perf/time_to_target_sec
memory/peak_allocated_mb
memory/peak_reserved_mb
memory/reduction_vs_fullbp_pct
memory/reduction_vs_adamw_pct
```

---

### 6.4 Functional update 指标

```text
functional/update_norm_coeff
functional/update_norm_function
functional/update_to_coeff_norm_ratio
functional/update_to_function_norm_ratio
functional/pred_descent
functional/actual_descent
functional/descent_ratio
functional/descent_sign_agreement_rate
functional/descent_pred_actual_corr
functional/bad_step_count
functional/bad_step_total_positive_loss_increase
```

---

### 6.5 Metric / preconditioner 指标

```text
metric/precond_condition
metric/eig_min
metric/eig_max
metric/eig_p05
metric/eig_p95
metric/update_amplification
metric/damping_rho
metric/sobolev_alpha
metric/sobolev_beta
```

---

### 6.6 KAN geometry 指标

```text
kan/phi_prime_mean
kan/phi_prime_p95
kan/phi_prime_max
kan/curvature_mean
kan/curvature_p95
kan/sobolev_norm_mean
kan/sobolev_norm_p95
kan/edge_function_norm
kan/edge_saturation_rate
basis/active_basis_fraction
basis/dead_basis_fraction
basis/out_of_grid_fraction
```

---

### 6.7 Jacobian / credit 指标

```text
jacobian/condition_mean
jacobian/condition_max
jacobian/sigma_max_mean
jacobian/sigma_min_mean
credit/amplification_mean
credit/amplification_p95
credit/noise_gain_p95
credit/identity_relerr
credit/analytic_relerr
credit/first_order_relerr
credit/cos_vs_fullbp
credit/relerr_vs_fullbp
```

---

### 6.8 Branch utilization 指标

```text
branch/mean_output_norm_ratio
branch/branch_over_adamw
branch/layer_{k}/output_norm_ratio
ablation/no_kan_val_acc
ablation/no_kan_test_acc
ablation/no_kan_acc_drop
ablation/kan_branch_logit_delta_norm
ablation/kan_branch_margin_contribution
```

---

### 6.9 Stability 指标

```text
stability/nan_or_inf_count
stability/diverged
stability/loss_spike_count
stability/grad_norm_p95
stability/update_norm_p95
stability/max_phi_prime_p95
stability/max_jacobian_condition
stability/largest_stable_coeff_lr
stability/largest_stable_rest_lr
```

---

## 7. W&B Dashboard 设计

### Page F0：Overview

包含：

1. method × dataset 的 test acc 表；
2. val AUC bar；
3. time-to-target bar；
4. peak memory bar；
5. pass / fail scorecard；
6. failure table。

---

### Page F1：Convergence curves

包含：

1. train loss curves；
2. val loss curves；
3. val acc curves；
4. smoothed loss curves；
5. loss_at_10/25/50/75 pct bar。

图中必须同时显示 AdamW、AnalyticAdj functional、FirstOrder functional、FullBP functional。

---

### Page F2：Target-matched comparison

包含：

1. steps-to-target-val-loss；
2. steps-to-target-val-acc；
3. time-to-target-val-loss；
4. time-to-target-val-acc；
5. censored runs 标记。

---

### Page F3：Speed-accuracy-memory Pareto

画三类 Pareto：

1. x = step time，y = test acc，size = peak memory；
2. x = val AUC，y = test acc，color = method；
3. x = peak memory，y = time-to-target，color = method。

---

### Page F4：Functional geometry

包含：

1. $\phi'_{p95}$ over training；
2. curvature over training；
3. Sobolev norm over training；
4. Jacobian condition over training；
5. credit amplification p95 over training。

---

### Page F5：Branch utilization

包含：

1. branch ratio over training；
2. branch / AdamW by method；
3. no-KAN acc drop；
4. KAN logit delta norm；
5. branch ratio vs test acc scatter；
6. branch ratio vs $\phi'_{p95}$ scatter。

---

### Page F6：Learning-rate stability

包含：

1. lr scale heatmap：method × scale -> final acc；
2. lr scale heatmap：method × scale -> divergence rate；
3. largest stable LR bar；
4. loss spike count bar；
5. max Jacobian condition by lr scale。

---

### Page F7：Descent quality

包含：

1. predicted vs actual descent scatter；
2. descent ratio distribution；
3. bad step count by method；
4. bad step positive loss increase；
5. update norm vs actual loss delta。

---

### Page F8：Mechanism correlations

包含：

1. $\phi'_{p95}$ vs val AUC；
2. Jacobian condition vs val AUC；
3. credit amplification vs loss spike count；
4. branch ratio vs no-KAN drop；
5. curvature vs test acc；
6. branch ratio vs time-to-target。

---

## 8. Failure table

所有失败必须记录为结构化表。

| failure type | 触发条件 | 解释 |
|---|---|---|
| `functional_underfit` | train acc 显著低于 AdamW | functional update 太保守 |
| `branch_underactive` | branch / AdamW < 0.25 | KAN branch 没充分参与 |
| `branch_overactive` | branch / AdamW > 1.5 | KAN branch 过强 |
| `phi_prime_explosion` | $\phi'_{p95}$ 高于 AdamW $2\times$ | edge derivative 失控 |
| `jacobian_unstable` | J cond 高于 AdamW $2\times$ | 局部几何不稳 |
| `loss_spike` | loss spike count 超阈值 | 优化不稳定 |
| `no_convergence_gain` | AUC / time-to-target 不优于 AdamW | 没有快收敛优势 |
| `accuracy_gap_too_large` | test gap > 2% | 性能不足 |
| `geometry_no_better` | $\phi'$ / J cond 不优于 AdamW | 几何优势不存在 |
| `credit_not_bottleneck` | credit variants 曲线一致 | 收敛瓶颈不在 credit |

failure table 字段：

```text
failure/type
failure/phase
failure/dataset
failure/method
failure/seed
failure/metric_name
failure/metric_value
failure/threshold
failure/diagnosis
failure/recommended_action
```

---

## 9. Stage F 决策规则

### 9.1 若 Functional update 接近 AdamW 且更快

如果满足：

$$
\text{test acc gap}<1\%
$$

$$
\text{val AUC improvement}>10\%
$$

$$
\phi'_{p95}\text{ and }\kappa(J)\text{ lower than AdamW}
$$

则 Stage F 强通过，可以进入 Stage G 泛化实验，并把 `diag-to-Sobolev` 作为主优化器。

---

### 9.2 若 Functional update 精度接近但不更快

如果：

$$
\text{test acc gap}<1\%-2\%
$$

但：

$$
\text{val AUC improvement}\leq0
$$

同时几何明显更好，则结论是：

> Functional update 的主要优势是几何稳定，不是快收敛。

此时 Stage G/H 仍然值得做，因为泛化和抗遗忘可能受益。

---

### 9.3 若 Functional update 明显慢且精度差

如果：

$$
\text{test acc gap}>2\%
$$

且：

$$
\text{val AUC worse than AdamW}
$$

则 Stage F 失败。需要回到 F1 调 branch activity / learning rate / alpha，或把 functional update 降级为 KAN coeff regularizer。

---

### 9.4 若 AdamW 更强但几何更差

如果 AdamW accuracy 更高，但：

$$
\phi'_{p95}^{AdamW}\gg\phi'_{p95}^{func}
$$

$$
\kappa(J)^{AdamW}\gg\kappa(J)^{func}
$$

则不要直接判 functional update 失败。应进入 Stage G/H，测试泛化和抗遗忘。

---

## 10. 第一轮最小执行包

Stage F 第一轮建议执行以下实验。

### F0 minimal

```text
datasets = MNIST, Fashion-MNIST, KMNIST
methods = AdamW, FullBP-functional, AnalyticAdj-functional, FirstOrder-functional, Identity-functional
seeds = 0..4
epochs = 8
```

目标：复现 D1，并补全 Stage F convergence metrics。

---

### F1 search minimal

```text
datasets = Fashion-MNIST, KMNIST
method = AnalyticAdj-functional
seeds = 0,1
coeff_lr = 0.3, 0.5, 0.7, 1.0
rest_lr = 0.001, 0.003, 0.006
alpha = 1.0, 1.5, 2.0
warmup_frac = 0.05, 0.10, 0.25
```

先固定：

```text
sobolev_alpha = 1e-2
rho = 1e-3
```

目标：找出每个 dataset 的 top 3 config。

---

### F1 confirm minimal

```text
datasets = MNIST, Fashion-MNIST, KMNIST
methods = AdamW + top functional configs
seeds = 0..4
epochs = 8 and 20
```

目标：确认 functional update 是否能追近 AdamW，并保留几何优势。

---

### F2 formal comparison minimal

```text
methods = AdamW, AnalyticAdj-best, FirstOrder-best, FullBP-functional, Identity-functional
comparison = step-matched + target-matched
```

目标：正式报告 convergence / target metrics。

---

### F3 LR stability minimal

```text
datasets = Fashion-MNIST, KMNIST
methods = AdamW, AnalyticAdj-best
lr_scale = 0.25,0.5,1,2,4
```

目标：比较 largest stable LR 和 loss spike。

---

## 11. Stage F 最终输出

Stage F 完成后应产出：

1. `stage_f_runs.csv`：所有 run；
2. `stage_f_summary_by_method.csv`：按 method 聚合；
3. `stage_f_summary_by_dataset.csv`：按 dataset 聚合；
4. `stage_f_target_matched.csv`：steps/time to target；
5. `stage_f_lr_stability.csv`：largest stable LR；
6. `stage_f_failure_table.csv`；
7. W&B dashboard；
8. 一页结论表：

| conclusion | answer |
|---|---|
| functional update 是否更快 | yes / no / conditional |
| 是否追上 AdamW accuracy | yes / no / dataset-specific |
| 几何是否更稳定 | yes / no |
| branch 是否充分利用 | yes / no |
| credit 是否是瓶颈 | yes / no |
| 是否进入 Stage G/H/I | yes / no |

---

## 12. 当前预期结果与解释方式

基于 Stage D/E 当前结果，最可能出现三种情况。

### 情况 A：Functional update 精度追近 AdamW，AUC 也更好

这是最强结果。说明 `diag-to-Sobolev` 不仅几何更好，而且优化更快。可以把 Stage F 作为主贡献之一。

---

### 情况 B：Functional update 精度追近 AdamW，但 AUC 不更好

这仍然有价值。说明 functional update 更偏向几何稳定，不一定快。后续应重点验证 Stage G 泛化和 Stage H 抗遗忘。

---

### 情况 C：Functional update 追不上 AdamW

如果 branch under-active，则继续调 branch / coeff_lr / alpha；如果 branch 正常但仍追不上，则 functional update 可能更适合作为 regularizer，而不是主 optimizer。

---

## 13. 最终一句话

Stage F 的核心不是重复证明 analytic adjoint 正确，而是回答：

$$
\boxed{
\text{DG-KAN 的 functional-space update 是否在视觉分类中带来真实优化优势？}
}
$$

如果答案是 yes，那么 DG-KAN 的主线变成：

$$
\text{analytic local credit}
+
\text{functional update}
+
\text{fast convergence}
+
\text{low memory}
$$

如果答案是 no，但几何优势明显，那么后续重点转向：

$$
\text{generalization}
+
\text{anti-forgetting}
+
\text{stable credit transport}
$$

而不是继续强推“快收敛”。
