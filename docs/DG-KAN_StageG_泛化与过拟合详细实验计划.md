# DG-KAN Stage G 详细实验计划：Generalization, Overfitting, Robustness and Calibration

> 文件名：`DG-KAN_StageG_泛化与过拟合详细实验计划.md`  
> 阶段：Stage G  
> 目标：验证 DG-KAN 的 **functional-space update + analytic local credit** 是否带来更好的泛化、不易过拟合、鲁棒性与校准，而不是仅仅表现为欠拟合或几何指标更好。  
> 公式格式：Typora 友好，统一使用 `$...$` 和 `$$...$$`。

---

## 0. Stage G 的核心问题

Stage F 已经说明，静态 functional update 不是自动优于 AdamW；它需要 branch-aware / geometry-aware schedule 才能释放优势。Fashion-MNIST 上 geometry-aware schedule 已经能同时做到：accuracy 超过 AdamW、val-loss AUC 更低、`phi_prime_p95` 更低、Jacobian condition 更低。KMNIST 上也明显优于 static functional update，但仍有约 $1\%$ 左右的 accuracy gap。

因此 Stage G 不再问：

> functional update 能不能训练？

这个问题已经由 Stage D/F 给出正面答案。Stage G 要问的是：

> **DG-KAN 的函数空间更新是否真的更泛化、更不容易过拟合、更鲁棒、更可校准？**

更准确地说，Stage G 要区分三种情况：

1. **真泛化更好**：train performance 接近 AdamW，但 test / OOD / corruption / calibration 更好；
2. **只是欠拟合**：train performance 明显差，所以 test gap 看起来更小；
3. **几何好但任务无收益**：`phi'`、Jacobian、curvature 更稳，但 test acc / robustness / calibration 没有实际改善。

Stage G 的核心判定不是单一 test accuracy，而是：

$$
\text{generalization gain}
+
\text{non-underfitting evidence}
+
\text{geometry control}
+
\text{branch utilization}
$$

同时成立。

---

## 1. Stage G 的研究假设

### 1.1 主假设

DG-KAN 的 KAN edge function 是可学习一维函数：

$$
\phi_{ji}(t)=\sum_m a_{jim}B_m(t)
$$

functional-space update 使用 metric 修正 coefficient gradient：

$$
a
\leftarrow
 a-
\eta(M+\rho I)^{-1}\nabla_a\mathcal L
$$

其中 Sobolev metric 可以写成：

$$
\|\phi\|_{\mathcal H}^2
=
\int \phi(t)^2dt
+
\alpha\int \phi'(t)^2dt
+
\beta\int \phi''(t)^2dt
$$

因此 functional update 可能带来三个泛化相关效果：

1. 控制 edge function 的过大斜率：

   $$
   \phi'_{p95},\quad \phi'_{max}
   $$

2. 控制 edge function 的过大曲率：

   $$
   \int |\phi''(t)|^2dt
   $$

3. 控制局部 credit transport 的放大：

   $$
   A_k=\frac{\|g_k\|}{\|g_{k+1}\|+\epsilon}
   $$

主假设是：

$$
\boxed{
\text{functional-space update controls function complexity and credit geometry,}
\quad
\text{therefore it may improve generalization under limited data, noisy labels, and distribution shifts.}
}
$$

---

### 1.2 反假设

Stage G 必须主动检验几个反假设。

#### 反假设 A：泛化好只是欠拟合

如果 DG-KAN functional update 的 train accuracy 显著低于 AdamW，那么 test gap 小并不说明泛化好。

因此必须记录：

$$
\text{train acc},\quad \text{val acc},\quad \text{test acc}
$$

以及：

$$
\text{train-test gap}_{acc}=\operatorname{Acc}_{train}-\operatorname{Acc}_{test}
$$

$$
\text{train-test gap}_{loss}=\mathcal L_{test}-\mathcal L_{train}
$$

#### 反假设 B：branch 没被用起来

如果 KAN branch under-active，即：

$$
\text{branch / AdamW}<0.3
$$

或者 no-KAN ablation drop 很小：

$$
\Delta_{noKAN}
=
\operatorname{Acc}_{full}-\operatorname{Acc}_{noKAN}
\approx0
$$

那么不能说 KAN functional update 改善了泛化，因为 KAN branch 可能根本没参与任务。

#### 反假设 C：只是强正则导致 accuracy 下降

如果 Sobolev / geometry-aware update 让 curvature 很低，但 train loss 也降不下去，则说明过度平滑。

触发条件：

$$
\operatorname{Acc}_{train}^{DG-KAN}
<
\operatorname{Acc}_{train}^{AdamW}-2\%
$$

且：

$$
\phi'_{p95}\ll \phi'^{AdamW}_{p95}
$$

或 curvature 显著过低。

---

## 2. Stage G 的方法设置

Stage G 不再大范围探索 credit 方法。默认 credit 主线是：

$$
\boxed{\text{AnalyticAdj-DGKAN}}
$$

因为 Stage D/E 已经证明 analytic primitive backward 与 FullBP 对齐，并且 AnalyticAdj 是当前最强 low-memory 路线。

FirstOrderDecompAdj 可以作为候选方法保留，但不是 Stage G 的主线。learned router 不进入 Stage G 主实验。

---

### 2.1 主方法

#### Method 1：DGKAN-AdamW

所有参数使用 AdamW，是强 accuracy baseline。

#### Method 2：AnalyticAdj-DGKAN + Static Functional

KAN coefficients 使用 static `diag_to_sobolev`，非 KAN 参数使用 AdamW。

用于判断：静态 functional update 的泛化和几何收益。

#### Method 3：AnalyticAdj-DGKAN + Geometry-Aware Functional

Stage F 推荐主方法。

Fashion-MNIST 默认：

```text
branch_schedule = geometry_aware
branch_boost = 2.0
coeff_lr_boost = 1.5
branch_max_active_frac = 0.50
branch_final_scale = 0.8
phi_high = 0.052
jac_high = 4.5
branch_high = 0.30
```

KMNIST 默认：

```text
branch_schedule = geometry_aware
branch_boost = 1.5
coeff_lr_boost = 1.2
branch_max_active_frac = 0.35
branch_final_scale = 0.9
phi_high = 0.055
jac_high = 4.5
branch_high = 0.30
```

#### Method 4：FirstOrderDecomp-DGKAN + Geometry-Aware Functional

候选方法。用于验证 first-order credit 是否在泛化任务中与 AnalyticAdj 保持一致。

#### Method 5：ResMLP / KAN-AdamW baseline

用于控制架构差异。

---

### 2.2 不进入主线的方法

以下方法只作为必要时的 ablation：

| 方法 | 原因 |
|---|---|
| HiddenRouterAdj | 之前实验显示 hidden-only router 学不好 correction |
| DecompRouterAdj | v4 显示 fidelity 好但 micro-training gain 很小，且成本高于 first-order |
| Dense SufficientStats | 当前 dense 版本显存收益不强，只做 Stage E 后续优化 |
| RKHS update | 当前 RKHS metric 不稳定，不进入主线 |

---

## 3. 数据集与阶段安排

Stage G 采用逐级推进，不一开始上 CIFAR。

### 3.1 主数据集

| 数据集 | 作用 |
|---|---|
| Fashion-MNIST | Stage F 已强通过，是验证泛化的第一主数据集 |
| KMNIST | 更难，Stage F 中接近通过但仍有 accuracy gap，是压力测试主数据集 |
| MNIST | 仅 sanity，不作为强证据 |

### 3.2 扩展数据集

| 数据集 | 进入条件 |
|---|---|
| CIFAR-10 small | Fashion/KMNIST 上至少 medium pass 后进入 |
| CIFAR-10 full | CIFAR-10 small 通过后进入 |
| CIFAR-100 small | CIFAR-10 full 或 small 有正面结果后进入 |

---

## 4. Stage G 的实验总结构

Stage G 分成七个连续实验包：

| Package | 名称 | 目的 |
|---|---|---|
| G0 | Baseline reproduction and train-gap calibration | 复现 Stage F 默认配置，建立 clean baseline |
| G1 | Small-data generalization | 检查小样本泛化 |
| G2 | Label-noise robustness | 检查是否更少记忆噪声标签 |
| G3 | Input corruption / OOD robustness | 检查分布扰动鲁棒性 |
| G4 | Calibration and confidence | 检查 ECE / NLL / confidence |
| G5 | Matched-train-performance analysis | 排除欠拟合假象 |
| G6 | Function-complexity mechanism audit | 建立泛化与函数复杂度的关系 |
| G7 | CIFAR-10 small entry gate | 决定是否进入真实视觉数据 |

---

## 5. G0：Baseline reproduction and train-gap calibration

### 5.1 目标

G0 的目标是复现 Stage F 的默认 setting，并额外记录泛化相关指标。

G0 不用于提出新 claim，而是确认：

1. AdamW baseline 稳定；
2. geometry-aware functional setting 稳定；
3. branch utilization 正常；
4. train-test gap 可用；
5. no-KAN ablation 可用。

---

### 5.2 实验设置

数据集：

```text
Fashion-MNIST
KMNIST
MNIST sanity optional
```

训练集大小：

```text
train = 6000
val = 1000
test = 1000
```

模型：

```text
residual DG-KAN
depth = 4
hidden_dim = 64
basis_count = 16
alpha_init = 1.5
```

seeds：

```text
0,1,2,3,4
```

方法：

```text
DGKAN-AdamW
AnalyticAdj-StaticFunctional
AnalyticAdj-GeometryAwareFunctional
FirstOrderDecomp-GeometryAwareFunctional optional
ResMLP-AdamW optional
```

---

### 5.3 必须记录的指标

#### Performance

```text
train/loss
val/loss
test/loss
train/acc
val/acc
test/acc
```

#### Generalization gap

```text
generalization/train_test_gap_loss
generalization/val_train_gap_loss
generalization/train_test_gap_acc
generalization/val_train_gap_acc
```

公式：

$$
\text{train-test gap}_{acc}
=
\operatorname{Acc}_{train}-\operatorname{Acc}_{test}
$$

$$
\text{train-test gap}_{loss}
=
\mathcal L_{test}-\mathcal L_{train}
$$

#### Branch utilization

```text
branch/mean_output_norm_ratio
branch/branch_over_adamw
ablation/no_kan_test_acc
ablation/no_kan_acc_drop
ablation/kan_logit_delta_norm
```

#### KAN complexity

```text
kan/phi_prime_p95
kan/phi_prime_max
kan/curvature_mean
kan/curvature_p95
kan/sobolev_norm_mean
kan/basis_active_fraction
kan/out_of_grid_fraction
```

#### Credit geometry

```text
jacobian/condition_mean
jacobian/condition_max
credit/amplification_p95
credit/noise_gain_p95
```

---

### 5.4 G0 成功条件

G0 通过要求：

1. Fashion-MNIST 上 geometry-aware functional 的 test acc 不低于 AdamW $-1\%$；
2. KMNIST 上 geometry-aware functional 的 test acc 不低于 AdamW $-1.5\%$；
3. `branch / AdamW` 不低于 $0.5$；
4. `no-KAN acc drop` 不低于 AdamW 的 $50\%$；
5. `phi_prime_p95` 与 Jacobian condition 显著低于 AdamW。

如果 G0 不复现 Stage F，则先不进入 G1/G2。

---

## 6. G1：Small-data generalization

### 6.1 目标

G1 验证在小样本场景下，functional-space update 是否比 AdamW 更不容易过拟合。

核心问题：

> 当训练数据变少时，DG-KAN 的函数空间正则是否能让 test accuracy / NLL / calibration 更好？

---

### 6.2 数据设置

训练集大小：

$$
N\in\{250,500,1000,2000,4000,6000\}
$$

验证集和测试集固定：

```text
val = 1000
test = 1000
```

每个 $N$ 重新采样 class-balanced train subset。

数据集：

```text
Fashion-MNIST
KMNIST
MNIST sanity optional
```

seeds：

```text
0,1,2,3,4
```

---

### 6.3 方法比较

```text
DGKAN-AdamW
AnalyticAdj-StaticFunctional
AnalyticAdj-GeometryAwareFunctional
FirstOrderDecomp-GeometryAwareFunctional optional
ResMLP-AdamW optional
```

所有方法使用相同数据 split。

---

### 6.4 指标

#### Main metrics

```text
small_data/train_size
small_data/test_acc
small_data/test_loss
small_data/val_acc
small_data/val_loss
small_data/train_acc
small_data/train_loss
small_data/train_test_gap_acc
small_data/train_test_gap_loss
small_data/acc_gap_vs_adamw
small_data/loss_gap_vs_adamw
```

#### Sample efficiency

设目标 accuracy 为 AdamW full-data accuracy 的若干比例：

$$
\operatorname{Acc}_{target}=0.95\cdot\operatorname{Acc}_{AdamW,N=6000}
$$

记录：

```text
small_data/samples_to_target_acc
small_data/samples_to_target_loss
```

#### Overfitting slope

对每个方法拟合：

$$
\text{gap}(N)=aN^{-b}+c
$$

记录：

```text
small_data/overfit_slope_b
small_data/gap_area_under_curve
```

#### Geometry and branch

```text
branch/branch_over_adamw
ablation/no_kan_acc_drop
kan/phi_prime_p95
kan/curvature_mean
jacobian/condition_max
credit/amplification_p95
```

---

### 6.5 可视化

W&B Page G1 应包含：

1. **Accuracy vs train size**  
   x-axis: train size；y-axis: test acc；method as color。

2. **Train-test gap vs train size**  
   判断是否只是欠拟合。

3. **Test loss vs train size**  
   分类 accuracy 之外看 NLL。

4. **Sample efficiency curve**  
   看达到 target acc 需要多少样本。

5. **Branch ratio vs train size**  
   检查小数据下 KAN branch 是否 under-active。

6. **Curvature / phi_prime vs test gap scatter**  
   看函数复杂度是否和泛化 gap 相关。

---

### 6.6 G1 成功标准

Weak success：

在 Fashion-MNIST 上至少三个 train size 满足：

$$
\operatorname{Acc}_{DG-KAN}\geq \operatorname{Acc}_{AdamW}-1\%
$$

并且：

$$
\text{train-test gap}_{DG-KAN}<\text{train-test gap}_{AdamW}
$$

Medium success：

在 Fashion-MNIST 和 KMNIST 上均满足：

$$
\text{gap reduction}>10\%
$$

且不存在明显欠拟合：

$$
\operatorname{Acc}_{train}^{DG-KAN}
\geq
\operatorname{Acc}_{train}^{AdamW}-2\%
$$

Strong success：

DG-KAN 在小数据下同时满足：

$$
\text{test acc higher or equal}
$$

$$
\text{train-test gap lower by } >20\%
$$

$$
\text{samples-to-target lower by } >20\%
$$

且 branch utilization 正常：

$$
0.5<\text{branch / AdamW}<1.2
$$

---

## 7. G2：Label-noise robustness

### 7.1 目标

G2 验证 functional-space update 是否更不容易记忆噪声标签。

核心问题：

> 当训练标签含噪时，DG-KAN 是否更少 memorization，并在 clean test set 上保持更高性能？

---

### 7.2 噪声设置

使用 symmetric label noise。

噪声比例：

$$
\eta\in\{0.1,0.2,0.4\}
$$

对于每个训练样本，以概率 $\eta$ 将 label 随机替换成其他类别。

验证集和测试集保持 clean。

数据集：

```text
Fashion-MNIST
KMNIST
```

seeds：

```text
0,1,2,3,4
```

---

### 7.3 指标

#### Clean performance

```text
noise/clean_val_acc
noise/clean_test_acc
noise/clean_test_loss
noise/acc_gap_vs_clean_training
noise/acc_gap_vs_adamw
```

#### Noisy train behavior

```text
noise/noisy_train_acc_against_noisy_labels
noise/noisy_train_acc_against_clean_labels
noise/noise_memorization_rate
noise/noise_fit_rate
noise/clean_subset_train_acc
noise/corrupted_subset_train_acc_clean_label
noise/corrupted_subset_train_acc_noisy_label
```

定义噪声记忆率：

$$
\text{memorization rate}
=
\Pr
\left[
\hat y_i = y_i^{noisy}
\mid
 y_i^{noisy}\neq y_i^{clean}
\right]
$$

干净标签恢复率：

$$
\text{clean recovery rate}
=
\Pr
\left[
\hat y_i = y_i^{clean}
\mid
 y_i^{noisy}\neq y_i^{clean}
\right]
$$

#### Geometry and complexity

```text
kan/phi_prime_p95
kan/curvature_mean
jacobian/condition_max
branch/branch_over_adamw
ablation/no_kan_acc_drop
```

#### Training dynamics

```text
conv/loss_auc_val_clean
conv/time_to_clean_val_target
stability/loss_spike_count
```

---

### 7.4 可视化

W&B Page G2 应包含：

1. **Clean test acc vs label noise ratio**；
2. **Memorization rate vs epoch**；
3. **Clean recovery rate vs epoch**；
4. **Train noisy acc vs clean test acc scatter**；
5. **phi_prime / curvature vs memorization rate**；
6. **Per-class confusion under label noise**；
7. **Calibration under noise**。

---

### 7.5 G2 成功标准

Weak success：

在 Fashion-MNIST 上至少一个 noise ratio 满足：

$$
\operatorname{Acc}_{clean}^{DG-KAN}
>
\operatorname{Acc}_{clean}^{AdamW}
$$

或 clean accuracy 不低于 AdamW $-1\%$，同时 memorization rate 更低。

Medium success：

在 Fashion-MNIST 和 KMNIST 上，DG-KAN 满足：

$$
\text{memorization rate reduction}>15\%
$$

同时：

$$
\text{clean test acc gap}<1.5\%
$$

Strong success：

DG-KAN 在 $\eta=0.2$ 和 $0.4$ 下均满足：

$$
\operatorname{Acc}_{clean}^{DG-KAN}
>
\operatorname{Acc}_{clean}^{AdamW}
$$

并且：

$$
\text{memorization rate reduction}>25\%
$$

---

## 8. G3：Input corruption / OOD robustness

### 8.1 目标

G3 验证 functional-space update 是否改善输入扰动下的鲁棒性。

训练使用 clean train set，测试使用 corruption / shifted test set。

---

### 8.2 Corruption types

对 Fashion-MNIST / KMNIST 使用：

| corruption | severity |
|---|---|
| Gaussian noise | $1,2,3,4,5$ |
| Gaussian blur | $1,2,3,4,5$ |
| rotation | $\pm 10^\circ, \pm 20^\circ, \pm 30^\circ$ |
| translation | $2,4,6$ pixels |
| contrast shift | $0.5,0.75,1.25,1.5$ |
| occlusion / cutout | small / medium / large |

后续 CIFAR-10 可使用 CIFAR-10-C 类似 corruption。

---

### 8.3 指标

```text
corruption/acc_clean
corruption/acc_mean
corruption/acc_worst
corruption/loss_mean
corruption/loss_worst
corruption/relative_drop_mean
corruption/relative_drop_worst
corruption/type_{name}/severity_{s}/acc
corruption/type_{name}/severity_{s}/loss
```

定义平均 corruption accuracy：

$$
\operatorname{Acc}_{corr,mean}
=
\frac{1}{|\mathcal C|}
\sum_{c\in\mathcal C}
\operatorname{Acc}_c
$$

相对下降：

$$
\Delta_{corr}
=
\operatorname{Acc}_{clean}-\operatorname{Acc}_{corr,mean}
$$

---

### 8.4 可视化

W&B Page G3 应包含：

1. **Corruption robustness heatmap**：method × corruption severity；
2. **Clean vs corrupted accuracy scatter**；
3. **Worst corruption bar chart**；
4. **Corruption drop vs phi_prime_p95 scatter**；
5. **Corruption drop vs curvature scatter**；
6. **Representative corrupted image panels**。

---

### 8.5 G3 成功标准

Weak success：

DG-KAN corruption mean accuracy 不低于 AdamW $-1\%$，同时 clean accuracy gap 不超过 $1.5\%$。

Medium success：

DG-KAN 满足：

$$
\Delta_{corr}^{DG-KAN}<\Delta_{corr}^{AdamW}
$$

且 corruption mean accuracy 高于 AdamW 或持平。

Strong success：

DG-KAN 在至少两个 corruption types 上显著优于 AdamW，并且：

$$
\operatorname{Acc}_{clean}^{DG-KAN}
\geq
\operatorname{Acc}_{clean}^{AdamW}-1\%
$$

---

## 9. G4：Calibration and confidence

### 9.1 目标

G4 验证 DG-KAN 是否因函数复杂度更可控而具有更好的概率校准。

这很重要，因为 functional update 可能不提高 accuracy，但可能降低过度自信。

---

### 9.2 指标

```text
calibration/ece
calibration/mce
calibration/nll
calibration/brier_score
calibration/mean_confidence_correct
calibration/mean_confidence_incorrect
calibration/confidence_gap
calibration/entropy_mean
```

Expected Calibration Error：

$$
\operatorname{ECE}
=
\sum_{b=1}^{B}
\frac{|S_b|}{N}
\left|
\operatorname{acc}(S_b)-\operatorname{conf}(S_b)
\right|
$$

Brier score：

$$
\operatorname{Brier}
=
\frac{1}{N}\sum_{n=1}^{N}\|p_n-y_n\|^2
$$

---

### 9.3 设置

在 G0/G1/G2/G3 所有 eval 中同步记录 calibration。

重点比较：

1. clean test calibration；
2. small-data calibration；
3. label-noise calibration；
4. corruption calibration。

---

### 9.4 可视化

W&B Page G4 应包含：

1. **Reliability diagram**；
2. **Confidence histogram**；
3. **ECE vs method bar**；
4. **NLL vs accuracy scatter**；
5. **ECE vs phi_prime_p95 scatter**；
6. **Confidence on wrong predictions**。

---

### 9.5 G4 成功标准

Weak success：

DG-KAN ECE 低于 AdamW，且 clean test acc gap < $1.5\%$。

Medium success：

DG-KAN 在 clean、small-data、noise 或 corruption 中至少两个设置 ECE 均低于 AdamW $>10\%$。

Strong success：

DG-KAN 同时满足：

$$
\text{ECE reduction}>20\%
$$

$$
\text{NLL lower or equal}
$$

$$
\text{accuracy gap}<1\%
$$

---

## 10. G5：Matched-train-performance analysis

### 10.1 目标

G5 是 Stage G 最重要的防伪实验。它用于排除：

> DG-KAN 泛化 gap 小只是因为 train performance 低。

---

### 10.2 方法

对于每个 method 的训练过程，保存 checkpoints：

```text
epoch checkpoints
best val checkpoint
matched train acc checkpoint
matched train loss checkpoint
```

选择 AdamW 和 DG-KAN 在相同 train accuracy 或相同 train loss 下比较 test performance。

例如选择：

$$
\operatorname{Acc}_{train}^{AdamW}(t_a)
\approx
\operatorname{Acc}_{train}^{DG-KAN}(t_d)
$$

然后比较：

$$
\operatorname{Acc}_{test}^{AdamW}(t_a)
$$

和：

$$
\operatorname{Acc}_{test}^{DG-KAN}(t_d)
$$

---

### 10.3 指标

```text
matched/train_acc_target
matched/train_loss_target
matched/adamw_test_acc_at_target
matched/dgkan_test_acc_at_target
matched/test_acc_gain_at_matched_train_acc
matched/test_loss_gain_at_matched_train_loss
matched/ece_gain_at_matched_train_acc
matched/phi_prime_at_target
matched/curvature_at_target
```

---

### 10.4 可视化

1. **Train acc vs test acc trajectory**；
2. **Train loss vs test loss trajectory**；
3. **Matched train acc comparison bars**；
4. **Matched train loss comparison bars**；
5. **Generalization gap along training**。

---

### 10.5 成功标准

DG-KAN 泛化 claim 只有在 matched-train analysis 中仍成立，才算强。

Medium success：

$$
\operatorname{Acc}_{test}^{DG-KAN}
>
\operatorname{Acc}_{test}^{AdamW}
$$

at matched train accuracy，或：

$$
\mathcal L_{test}^{DG-KAN}<\mathcal L_{test}^{AdamW}
$$

at matched train loss。

如果 DG-KAN 只在 final checkpoint gap 更小，但 matched train performance 下没有优势，则不能 claim 真泛化更好。

---

## 11. G6：Function-complexity mechanism audit

### 11.1 目标

G6 不是单独训练实验，而是对 G0-G5 的所有结果做机制分析。

核心问题：

> 泛化、鲁棒性、校准是否与 KAN edge function 的复杂度控制有关？

---

### 11.2 机制指标

```text
mechanism/phi_prime_p95
mechanism/phi_prime_max
mechanism/curvature_mean
mechanism/curvature_p95
mechanism/sobolev_norm_mean
mechanism/basis_active_fraction
mechanism/out_of_grid_fraction
mechanism/jacobian_condition
mechanism/credit_amplification_p95
mechanism/branch_over_adamw
mechanism/no_kan_acc_drop
```

---

### 11.3 相关性分析

计算 Pearson / Spearman correlation：

```text
corr(phi_prime_p95, test_acc)
corr(phi_prime_p95, train_test_gap)
corr(curvature, train_test_gap)
corr(jacobian_condition, corruption_drop)
corr(branch_ratio, test_acc)
corr(no_kan_drop, test_acc)
corr(ece, phi_prime_p95)
```

---

### 11.4 可视化

W&B Page G6 应包含：

1. **phi_prime vs train-test gap scatter**；
2. **curvature vs test loss scatter**；
3. **Jacobian condition vs corruption drop scatter**；
4. **branch ratio vs test acc scatter**；
5. **no-KAN drop vs test acc scatter**；
6. **parallel coordinate plot**：method / accuracy / gap / phi / curvature / branch / ECE。

---

### 11.5 判定

如果 DG-KAN 泛化更好，同时存在：

$$
\phi'_{p95}\downarrow
$$

$$
\text{curvature}\downarrow
$$

$$
\kappa(J)\downarrow
$$

但 branch utilization 正常，则支持：

> functional-space update 通过控制函数复杂度改善泛化。

如果泛化没有改善，但几何改善明显，则结论应写成：

> functional update improves geometry but does not necessarily improve generalization under this setting.

---

## 12. G7：CIFAR-10 small entry gate

### 12.1 目标

G7 决定是否把 Stage G 推进到 CIFAR-10 small。

---

### 12.2 进入条件

只有在 Fashion-MNIST 或 KMNIST 至少达到 Stage G medium success 后进入 CIFAR-10 small。

条件：

1. Fashion or KMNIST small-data / noise / corruption 至少一个 package medium pass；
2. geometry-aware functional 的 branch utilization 正常；
3. no-KAN ablation 证明 KAN branch 有贡献；
4. Stage F 的 convergence 不明显差于 AdamW。

---

### 12.3 CIFAR-10 small 设置

模型：

```text
ConvStem-DGKAN
conv stem -> residual DG-KAN blocks -> head
```

初始配置：

```text
train = 10000
val = 5000
test = 10000
hidden_dim = 128
depth = 4
basis_count = 16
grouped KAN optional
```

方法：

```text
SmallCNN-AdamW
ConvStem-DGKAN-AdamW
ConvStem-DGKAN-AnalyticAdj-GeometryAwareFunctional
ConvStem-DGKAN-FirstOrder optional
```

指标沿用 G0-G6。

---

## 13. W&B config 规范

每个 run 必须记录以下 config。

```text
stage = G
package = G0/G1/G2/G3/G4/G5/G6/G7
experiment_name
dataset
train_size
val_size
test_size
label_noise_ratio
corruption_type
corruption_severity
seed
model_family
hidden_dim
depth
basis_count
alpha_init
connectivity = dense/grouped
credit_mode = analytic/first_order/fullbp/identity
update_mode = adamw/static_functional/geometry_aware
branch_schedule
branch_boost
coeff_lr_boost
branch_final_scale
branch_max_active_frac
phi_high
jac_high
branch_high
coeff_lr
rest_lr
warmup_frac
sobolev_alpha
sobolev_beta
rho
batch_size
epochs
```

---

## 14. W&B Dashboard 总设计

### Page G0：Stage G overview

包括：

1. method × dataset summary table；
2. test acc mean ± std；
3. train-test gap mean ± std；
4. ECE / NLL；
5. corruption mean acc；
6. label-noise clean acc；
7. pass/fail scorecard。

---

### Page G1：Small-data generalization

包括：

1. accuracy vs train size；
2. train-test gap vs train size；
3. sample-to-target curve；
4. test loss vs train size；
5. branch ratio vs train size；
6. curvature vs small-data gap。

---

### Page G2：Label noise

包括：

1. clean test acc vs noise ratio；
2. memorization rate vs epoch；
3. clean recovery rate vs epoch；
4. noisy train acc vs clean test acc；
5. confidence on noisy examples；
6. per-class confusion under noise。

---

### Page G3：Corruption / OOD

包括：

1. corruption heatmap；
2. severity curves；
3. mean corruption acc bar；
4. worst corruption acc bar；
5. corruption drop vs phi_prime scatter；
6. example corrupted images。

---

### Page G4：Calibration

包括：

1. reliability diagram；
2. confidence histogram；
3. ECE bar；
4. NLL vs accuracy scatter；
5. Brier score bar；
6. wrong prediction confidence。

---

### Page G5：Matched-train analysis

包括：

1. train acc vs test acc trajectory；
2. train loss vs test loss trajectory；
3. matched train acc test comparison；
4. matched train loss test comparison；
5. gap over training。

---

### Page G6：Mechanism audit

包括：

1. phi_prime vs train-test gap；
2. curvature vs test loss；
3. Jacobian condition vs corruption drop；
4. branch ratio vs test acc；
5. no-KAN drop vs test acc；
6. parallel coordinate plot。

---

### Page G7：Failure analysis

Failure table 包含：

| failure type | 含义 |
|---|---|
| `underfit_not_generalization` | train acc 明显低，不能说泛化好 |
| `branch_underactive` | branch / AdamW 太低 |
| `no_kan_no_contribution` | no-KAN drop 太小 |
| `over_smoothing` | curvature 太低且 train loss 不降 |
| `no_generalization_gain` | test gap / robustness 没提升 |
| `noise_memorization_not_reduced` | 噪声标签记忆没有降低 |
| `corruption_not_robust` | corruption acc 不如 baseline |
| `calibration_not_improved` | ECE / NLL 不优 |
| `geometry_good_task_bad` | 几何好但任务性能差 |
| `schedule_unstable` | branch 或 Jacobian 失控 |

---

## 15. Stage G 统计与报告规范

### 15.1 多 seed

正式结论必须使用：

```text
seeds = 0,1,2,3,4
```

报告：

```text
mean
std
min
max
95% bootstrap confidence interval
```

---

### 15.2 Paired comparison

由于不同方法使用相同 seed 和 data split，应报告 paired differences：

$$
\Delta_i
=
M_i^{DG-KAN}-M_i^{AdamW}
$$

并报告：

```text
paired mean difference
paired std
paired sign count
paired bootstrap CI
```

---

### 15.3 不使用单一最好 seed

任何 claim 都不能基于单 seed 或 best seed。必须基于 mean ± std。

---

## 16. Stage G 总成功标准

### Weak pass

满足任一：

1. 小数据下 test acc 不低于 AdamW $-1\%$，且 train-test gap 更低；
2. label noise 下 clean test acc 不低于 AdamW $-1\%$，且 memorization rate 更低；
3. corruption mean accuracy 不低于 AdamW $-1\%$，且 geometry 更稳；
4. ECE 低于 AdamW $>10\%$，且 accuracy gap < $1.5\%$。

---

### Medium pass

在 Fashion-MNIST 和 KMNIST 上至少两个 package 通过 weak，并且满足：

$$
\text{branch / AdamW}>0.5
$$

$$
\Delta_{noKAN}^{DG-KAN}>0.5\Delta_{noKAN}^{AdamW}
$$

$$
\operatorname{Acc}_{train}^{DG-KAN}
\geq
\operatorname{Acc}_{train}^{AdamW}-2\%
$$

---

### Strong pass

满足：

1. small-data test acc 高于 AdamW；
2. label-noise clean acc 高于 AdamW；
3. corruption robustness 高于 AdamW；
4. ECE / NLL 更好；
5. matched-train analysis 仍显示 test gain；
6. geometry 指标更稳定；
7. branch utilization 正常。

---

## 17. Stage G 的决策规则

### 情况 A：泛化提升明显

若 Stage G medium / strong pass，则可以进入 Stage H 抗遗忘，并把 functional update 的优势描述为：

$$
\text{better optimization geometry}
+
\text{better generalization}
$$

---

### 情况 B：只有几何改善，没有泛化改善

如果：

$$
\phi'\downarrow,
\quad
\kappa(J)\downarrow,
\quad
\text{but test / robustness not improved}
$$

则结论应收缩为：

> functional update improves geometry but does not automatically improve generalization.

后续优先做 Stage E / Stage H，而不是继续 Stage G。

---

### 情况 C：看似泛化好，但 train performance 明显低

如果 matched-train analysis 不支持 DG-KAN，则不能 claim 泛化好。

应诊断为：

```text
underfit_not_generalization
```

然后回到 Stage F 调 branch schedule / capacity。

---

### 情况 D：branch 没贡献

如果 no-KAN drop 很小，则不能 claim KAN 表达力或泛化。

应回到 Stage F 调 branch utilization。

---

## 18. 第一轮最小执行包

为了避免 Stage G 一次性过大，第一轮只做：

### G0

Fashion-MNIST / KMNIST baseline reproduction。

### G1

Small-data：

$$
N\in\{500,1000,2000,6000\}
$$

Fashion + KMNIST。

### G2

Label noise：

$$
\eta\in\{0.2,0.4\}
$$

Fashion + KMNIST。

### G4

Calibration 同步记录，不单独跑。

### G5

Matched-train analysis 基于 G1/G2 checkpoints 离线分析。

G3 corruption 可作为第二轮。

---

## 19. 预期结论模板

### 若成功

> Geometry-aware functional update improves generalization under limited-data and noisy-label settings. Compared with AdamW, DG-KAN maintains comparable training accuracy while reducing train-test gap, label-noise memorization, $
\phi'_{p95}$, Jacobian condition, and calibration error. Matched-train analysis confirms that the improvement is not merely underfitting.

### 若部分成功

> DG-KAN functional update consistently improves function-space and credit geometry, and improves convergence on selected tasks. However, generalization gains are dataset- and regime-dependent. In harder settings, the method trades a small accuracy gap for lower function complexity and better calibration.

### 若失败

> Functional-space update improves geometry but does not automatically improve generalization. The main bottleneck is either branch under-utilization or over-smoothing. Further gains require better branch schedules, stronger capacity, or task-specific functional metrics.

---

## 20. 最终定位

Stage G 的目标不是证明 DG-KAN 一定泛化更好，而是严谨回答：

$$
\boxed{
\text{Does function-space geometry control translate into real generalization benefits?}
}
$$

如果答案是肯定的，DG-KAN 的主线将从：

$$
\text{faster and more stable optimization}
$$

扩展到：

$$
\text{better generalization and robustness}
$$

如果答案是否定的，DG-KAN 仍然可以保留：

$$
\text{analytic adjoint}
+
\text{low-memory primitive backward}
+
\text{geometry-controlled optimization}
$$

作为主要贡献。Stage G 的价值在于把“几何更好”是否真的转化为“泛化更好”说清楚。
