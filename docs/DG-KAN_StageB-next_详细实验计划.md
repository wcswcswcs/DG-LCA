# DG-KAN Stage B-next 详细实验计划

> 版本：v0.1  
> 对象：DG-LCA / DG-KAN 中的 **Stage B：KAN-like functional-space update** 后续实验  
> 定位：实验执行协议，不涉及具体代码实现  
> 公式格式：Typora 友好，使用 `$...$` 与 `$$...$$`，不使用方括号形式 display math  
> 核心目标：在 Stage B pilot 的基础上，系统验证 **functional-space preconditioning 是否真正适合 KAN edge functions**，并明确 `diag`、`Sobolev`、`RKHS` 的角色、适用条件、速度/最终质量/几何稳定性的 tradeoff。

---

## 0. 本计划为什么要重写 Stage B-next

Stage B pilot 已经给出一个重要信号：KAN-like functional update 不是空想。回归任务中，`diag` 通常收敛最快，`Sobolev` 通常最终 test loss 最好，而 `RKHS` 因 metric 条件数过高明显劣化。Two Moons 分类任务中，直接把 functional update 扩展到所有参数会欠拟合；改成 **KAN coefficients 使用 functional update，非 KAN 参数使用 Adam/AdamW** 后，性能明显恢复。

因此 Stage B-next 不再问一个泛泛的问题：

$$
\text{functional update 是否比 Adam 好？}
$$

而是问更细的四个问题：

1. **functional update 是否应该只作用在 KAN edge coefficients 上？**
2. **`diag` 是否适合作为快速 warmup preconditioner？**
3. **`Sobolev` 是否适合作为最终质量与几何稳定性的 preconditioner？**
4. **`RKHS` 是否只是当前实现病态，还是在合理 conditioning 后仍然不如 Sobolev / diag？**

Stage B-next 的主实验不训练 learned router，也不测试 sequential routing。所有 credit 仍然来自真实 BP / local exact VJP。这样可以隔离变量，只考察 KAN edge function 的更新几何。

---

## 1. Stage B-next 的核心假设

### 1.1 假设 H1：functional update 更适合 KAN edge coefficients，而不是所有参数

KAN-like 层中的边函数写成：

$$
\phi_{ji}(t)=\sum_{m=1}^{M}a_{jim}B_m(t)
$$

其中 $a_{jim}$ 是 edge function 的 basis coefficient。functional-space update 的对象是这些 coefficient，因为它们对应的是一个真实的一维函数 $\phi_{ji}$。因此，Stage B-next 的默认训练策略是：

$$
\text{KAN coefficients: functional preconditioned update}
$$

$$
\text{non-KAN parameters: AdamW}
$$

非 KAN 参数包括分类 head、stem、normalization 参数、普通 linear projection 等。它们没有天然的一维函数空间 metric，所以不应该强行使用 KAN 的 Sobolev / RKHS metric。

这一点要通过 ablation 验证，而不是只作为假设接受。Stage B-next 会比较：

| 设置 | KAN coefficients | non-KAN parameters | 目的 |
|---|---|---|---|
| all AdamW | AdamW | AdamW | 强 baseline |
| all functional + SGD rest | functional | SGD | 复现欠拟合失败模式 |
| hybrid diag | diag functional | AdamW | 验证 functional update 是否适合 KAN coeff |
| hybrid Sobolev | Sobolev functional | AdamW | 验证几何稳定性 |
| hybrid diag→Sobolev | schedule functional | AdamW | 验证速度与最终质量结合 |

---

### 1.2 假设 H2：`diag` 是快速 warmup 方法

`diag` preconditioner 只做 coefficient-wise 缩放：

$$
M=\operatorname{diag}(m_1,m_2,\dots,m_M)
$$

更新为：

$$
a_m\leftarrow a_m-\eta\frac{1}{m_m+\rho}\frac{\partial \mathcal L}{\partial a_m}
$$

它不建模 basis 之间的函数空间耦合，但代价低、条件数易控。Stage B pilot 显示它在回归任务上的 loss AUC 很强。因此 Stage B-next 要验证：

$$
\text{diag 是否适合前期快速下降？}
$$

如果成立，`diag` 不一定是最终最好的 optimizer，但可以作为 warmup 阶段。

---

### 1.3 假设 H3：`Sobolev` 是最终质量和局部几何稳定性方法

Sobolev metric 定义：

$$
\|\phi\|_{\mathcal H}^2
=
\int \phi(t)^2dt
+
\alpha\int \phi'(t)^2dt
+
\beta\int \phi''(t)^2dt
$$

对应 basis coefficient 的 Gram matrix：

$$
M_{mn}
=
\int B_mB_n
+
\alpha\int B'_mB'_n
+
\beta\int B''_mB''_n
$$

更新为：

$$
a\leftarrow a-
\eta(M_{\text{Sob}}+\rho I)^{-1}\nabla_a\mathcal L
$$

Stage B pilot 显示 Sobolev 在多个回归任务上最终 test loss 最好，并且在 Two Moons hybrid 设置中可以达到 Adam 的 accuracy，同时 Jacobian condition 和 $\phi'_\text{p95}$ 更低。

Stage B-next 要进一步验证：

$$
\text{Sobolev 是否在更广任务上提供更好的最终质量与几何稳定性？}
$$

---

### 1.4 假设 H4：`diag → Sobolev` schedule 可能同时获得速度和最终质量

基于 pilot：

- `diag`：早期下降快；
- `Sobolev`：最终 test loss 和几何稳定性更好。

因此引入 schedule：

$$
\text{first }T_w\text{ steps: diag}
\quad\rightarrow\quad
\text{remaining steps: Sobolev}
$$

Stage B-next 要判断该 schedule 是否能在下列指标上同时接近最优：

1. loss AUC 接近或优于 `diag`；
2. final test loss 接近或优于 `Sobolev`；
3. Jacobian condition 接近或优于 `Sobolev`；
4. bad steps 不增加；
5. 不显著增加 wall-clock。

---

### 1.5 假设 H5：RKHS 当前失败主要来自 conditioning，而不是理论上不可用

RKHS update 的理想形式是：

$$
\Delta\phi_{ji}(t)
=
-
\eta\mathbb E[g_jK(t,x_i)]
$$

但 pilot 中 RKHS preconditioner 的条件数约为：

$$
\kappa(M_{\text{RKHS}})\approx 4720.9
$$

远高于 diag / Sobolev 的约 $29.65$。Stage B-next 不把 RKHS 作为主方法，而是作为 **quarantine / repair track**。只有当：

$$
\kappa(M_{\text{RKHS}}+\rho I)<500
$$

时，才允许 RKHS 进入正式对比。否则只报告它作为失败案例。

---

## 2. 主架构与训练对象

Stage B-next 使用 DG-KAN / KAN-like block 作为主架构。一个 KAN-like block 定义为：

$$
y_j=\sum_i\phi_{ji}(x_i)
$$

其中：

$$
\phi_{ji}(t)=\sum_m a_{jim}B_m(t)
$$

对于深层模型，推荐 residual DG-KAN 形式：

$$
h_{k+1}=h_k+\alpha_k\operatorname{KAN}_k(\operatorname{LN}(h_k))
$$

其中：

- $h_k\in\mathbb R^d$ 是完整 hidden state；
- $\operatorname{KAN}_k$ 是 full hidden-state KAN-like mapping；
- $\alpha_k$ 是 residual scale；
- LayerNorm 保持 KAN edge function 输入范围稳定；
- Stage B-next 只改变 KAN edge coefficients 的更新方式。

如果模型中还有非 KAN 参数，例如输入 stem、classification head、LayerNorm 参数、residual scale $\alpha_k$，默认使用 AdamW。

---

## 3. 实验分组总览

Stage B-next 分为五组实验。五组实验之间不是松散提纲，而是递进关系。

| 实验组 | 名称 | 目的 | 是否主实验 |
|---|---|---|---|
| B0 | Pilot reproduction & calibration | 复现实验、校准指标、确认 logging 正确 | 是，作为基准 |
| B1 | diag→Sobolev schedule | 验证速度与最终质量能否结合 | 是，核心实验 |
| B2 | hybrid classification | 验证 functional update 只作用于 KAN coeff 的分类效果 | 是，核心实验 |
| B3 | Sobolev metric sweep | 找到 Sobolev 的稳定区域和超参规律 | 是，核心实验 |
| B4 | RKHS conditioning repair | 判断 RKHS 是否可修复 | 否，隔离实验 |

执行顺序为：

$$
\boxed{
B0\rightarrow B1\rightarrow B2\rightarrow B3\rightarrow B4
}
$$

B4 不应阻塞 B1-B3。如果 RKHS 继续失败，不影响 Stage C 进入 analytic KAN adjoint / learned router。

---

## 4. 数据集与任务设计

### 4.1 回归任务

回归任务用于精确分析函数拟合质量、curvature、descent alignment 和 final test loss。

#### Task R1：低频一维函数

$$
y=\sin x
$$

目的：验证简单平滑函数上 functional update 是否稳定。

#### Task R2：中频一维函数

$$
y=\sin x+0.3\sin(5x)
$$

目的：验证 Sobolev 是否能在需要更高频成分时保持最终拟合质量。

#### Task R3：二维可加函数

$$
y=\sin x_1+\cos x_2
$$

目的：测试多输入下 KAN edge functions 的耦合和泛化。

#### Task R4：带噪声版本

对 R1-R3 增加噪声：

$$
y_{noise}=y+\epsilon,
\quad
\epsilon\sim\mathcal N(0,\sigma^2)
$$

建议：

$$
\sigma\in\{0.01,0.05,0.1\}
$$

目的：判断 Sobolev 是否真的有泛化 / 平滑优势。如果 Sobolev 只是更好优化干净函数，它在 noisy test 上未必占优；如果它有正则化作用，noisy setting 下 test loss / gap 应更好。

#### Task R5：extrapolation split

训练区间：

$$
x\in[-\pi,\pi]
$$

测试区间：

$$
x\in[-1.5\pi,1.5\pi]
$$

目的：检验函数空间 metric 是否改善外推，而不只是插值。

---

### 4.2 分类任务

分类任务用于验证 hybrid functional update 是否在真实 decision boundary 上有效。

#### Task C1：Two Moons

作用：sanity classification。保留它，因为它能快速暴露 underfit、bad steps、Jacobian condition 等问题。

#### Task C2：MNIST

作用：简单真实分类任务。验证 hybrid 在 KAN-like classifier 上是否可用。

#### Task C3：Fashion-MNIST

作用：比 MNIST 更难，更能观察收敛速度与泛化差异。

#### Task C4：CIFAR-10 small

作用：更接近视觉任务。建议使用 conv stem 或 patch stem，然后接 residual DG-KAN blocks。

CIFAR-10 small 不要求第一轮达到 SOTA，只要求比较不同 update geometry 的相对稳定性和成本。

---

## 5. 方法与 baseline

Stage B-next 必须比较的不只是 Adam vs Sobolev，而是一组清晰的方法族。

### 5.1 Coefficient-space baseline

#### Method M1：coeff_sgd

$$
a\leftarrow a-\eta\nabla_a\mathcal L
$$

用途：最基础 baseline。预期不一定强，但用于判断 preconditioning 是否必要。

#### Method M2：coeff_adam

KAN coefficients 使用 Adam，非 KAN 参数也使用 Adam 或 AdamW。

用途：主要强 baseline。

#### Method M3：coeff_adamw

KAN coefficients 与非 KAN 参数都使用 AdamW。

用途：分类任务主 baseline。对于带 weight decay 的设置，AdamW 更公平。

---

### 5.2 Functional preconditioner

#### Method M4：diag

KAN coefficients 使用 diagonal preconditioner：

$$
a\leftarrow a-
\eta\operatorname{diag}(M+\rho I)^{-1}\nabla_a\mathcal L
$$

非 KAN 参数默认 AdamW。

#### Method M5：sobolev

KAN coefficients 使用 Sobolev preconditioner：

$$
a\leftarrow a-
\eta(M_{\text{Sob}}+\rho I)^{-1}\nabla_a\mathcal L
$$

非 KAN 参数默认 AdamW。

#### Method M6：diag_to_sobolev

前 $T_w$ steps 使用 diag，之后切到 Sobolev：

$$
P_t=
\begin{cases}
P_{\text{diag}}, & t<T_w \\
P_{\text{Sob}}, & t\geq T_w
\end{cases}
$$

其中：

$$
P_{\text{diag}}=\operatorname{diag}(M+\rho I)^{-1}
$$

$$
P_{\text{Sob}}=(M_{\text{Sob}}+\rho I)^{-1}
$$

#### Method M7：soft_mix_diag_sobolev

连续混合：

$$
P_t=(1-\lambda_t)P_{\text{diag}}+\lambda_tP_{\text{Sob}}
$$

其中 $\lambda_t$ 从 $0$ 增长到 $1$。建议使用 linear schedule 或 sigmoid schedule。

---

### 5.3 RKHS quarantine track

#### Method M8：rkhs_raw

复现当前 RKHS 设置，作为失败基线。

#### Method M9：rkhs_damped

使用更强 damping：

$$
P_{\text{RKHS}}=(M_{\text{RKHS}}+\rho I)^{-1}
$$

扫：

$$
\rho\in\{10^{-4},10^{-3},10^{-2},10^{-1},1.0\}
$$

#### Method M10：rkhs_clipped

对 eigenvalue 做 clipping：

$$
\lambda_i\leftarrow\max(\lambda_i,\lambda_{min})
$$

或限制有效条件数：

$$
\kappa(M_{\text{RKHS}}+\rho I)\leq\kappa_{max}
$$

建议：

$$
\kappa_{max}\in\{100,500,1000\}
$$

RKHS 只有在 condition number 达标时，才进入正式 loss / accuracy 比较。

---

## 6. W&B config 设计

每个 run 必须记录完整 config，避免后期无法解释差异。

### 6.1 实验元信息

| key | 示例 | 说明 |
|---|---|---|
| `exp/stage` | `B_next` | 实验阶段 |
| `exp/group` | `B1_diag_to_sobolev_regression` | 实验组 |
| `exp/task_type` | `regression` / `classification` | 任务类型 |
| `exp/method` | `sobolev` | 方法 |
| `exp/seed` | `0` | 随机种子 |
| `exp/run_name_manual` | `B1_sin5_diag2sob_Tw500_seed0` | 人工可读名称 |

---

### 6.2 数据配置

| key | 说明 |
|---|---|
| `data/name` | 数据集名称，例如 `sin1d`, `sin5`, `sincos2d`, `two_moons`, `mnist` |
| `data/train_size` | 训练样本数 |
| `data/val_size` | 验证样本数 |
| `data/test_size` | 测试样本数 |
| `data/noise_sigma` | 回归噪声标准差 |
| `data/extrapolation_split` | 是否使用外推测试 |
| `data/batch_size` | batch size |
| `data/normalization` | 数据标准化方式 |

---

### 6.3 模型配置

| key | 说明 |
|---|---|
| `model/name` | `residual_dgkan`, `dgkan_mlp`, `kan_classifier` |
| `model/depth` | KAN block 数 |
| `model/hidden_dim` | hidden dimension |
| `model/basis_type` | `bspline`, `rbf`, `piecewise_linear` |
| `model/basis_count` | basis 数量 $M$ |
| `model/residual_alpha_init` | residual scale 初始值 |
| `model/use_layernorm` | 是否使用 LayerNorm |
| `model/nonkan_param_count` | 非 KAN 参数量 |
| `model/kan_coeff_count` | KAN coefficient 数量 |

---

### 6.4 更新方法配置

| key | 说明 |
|---|---|
| `update/kan_method` | `adam`, `adamw`, `diag`, `sobolev`, `diag_to_sobolev`, `soft_mix_diag_sobolev`, `rkhs` |
| `update/nonkan_method` | `adamw`, `adam`, `sgd` |
| `update/coeff_lr` | KAN coefficient 学习率 |
| `update/nonkan_lr` | 非 KAN 参数学习率 |
| `update/weight_decay` | AdamW weight decay |
| `update/grad_clip` | gradient clipping |
| `update/schedule_type` | `none`, `hard_switch`, `soft_mix`, `rho_decay` |
| `update/warmup_steps` | diag warmup steps $T_w$ |
| `update/mix_schedule` | `linear`, `sigmoid`, `cosine` |

---

### 6.5 Metric / preconditioner 配置

| key | 说明 |
|---|---|
| `metric/type` | `diag`, `sobolev`, `rkhs` |
| `metric/damping_rho` | damping $\rho$ |
| `metric/sobolev_alpha` | Sobolev 一阶导权重 $\alpha$ |
| `metric/sobolev_beta` | Sobolev 二阶导权重 $\beta$ |
| `metric/rkhs_kernel` | kernel 类型 |
| `metric/rkhs_lengthscale` | kernel lengthscale |
| `metric/eigen_clip_enabled` | 是否 eigenvalue clipping |
| `metric/condition_cap` | 最大允许条件数 |

---

### 6.6 训练预算

| key | 说明 |
|---|---|
| `train/max_steps` | 最大训练步数 |
| `train/max_epochs` | 最大 epoch |
| `train/eval_every` | 评估频率 |
| `train/audit_every` | Jacobian / curvature audit 频率 |
| `train/early_stop_enabled` | 是否 early stopping |
| `resource/precision` | `fp32`, `bf16` 等 |
| `resource/gpu_type` | GPU 类型 |

---

## 7. 每个 run 必须记录的指标

### 7.1 任务性能指标

| metric | 频率 | 说明 |
|---|---:|---|
| `train/loss` | every step | 训练 loss |
| `val/loss` | eval | 验证 loss |
| `test/loss` | final | 测试 loss |
| `train/acc` | eval | 分类训练准确率 |
| `val/acc` | eval | 分类验证准确率 |
| `test/acc` | final | 分类测试准确率 |
| `generalization/train_test_gap_loss` | final | `test/loss - train/loss` |
| `generalization/val_train_gap_loss` | eval | `val/loss - train/loss` |
| `generalization/val_train_gap_acc` | eval | `train/acc - val/acc` |

在判断 Sobolev 是否“泛化更好”时，不能只看 `test/loss`，还要看 gap。如果 Sobolev 的 train loss 和 test loss 都更低，则说明它拟合质量更好；如果 train loss 相近但 test loss 更低，才是更强的泛化证据。

---

### 7.2 收敛速度指标

#### loss AUC

loss AUC 定义为：

$$
\text{loss\_auc}
=
\frac{1}{T}\sum_{t=1}^{T}\mathcal L_t
$$

记录：

| metric | 说明 |
|---|---|
| `conv/loss_auc_train` | train loss AUC |
| `conv/loss_auc_val` | val loss AUC |
| `conv/loss_auc_normalized_vs_adam` | 相对 Adam 的 AUC 比例 |

#### steps-to-target

设 Adam final loss 为参考：

$$
\mathcal L_{target}=1.05\times\mathcal L_{Adam,final}
$$

分类任务也可设：

$$
\text{Acc}_{target}=\text{Acc}_{Adam,final}-0.01
$$

记录：

| metric | 说明 |
|---|---|
| `conv/steps_to_target_loss` | 达到目标 loss 的 step |
| `conv/steps_to_target_acc` | 达到目标 accuracy 的 step |
| `conv/time_to_target_loss_sec` | 达到目标 loss 的 wall-clock |
| `conv/time_to_target_acc_sec` | 达到目标 accuracy 的 wall-clock |

#### fixed-step loss

记录：

| metric | 说明 |
|---|---|
| `conv/loss_at_10pct_steps` | 10% 训练步数时 loss |
| `conv/loss_at_25pct_steps` | 25% 训练步数时 loss |
| `conv/loss_at_50pct_steps` | 50% 训练步数时 loss |
| `conv/loss_at_75pct_steps` | 75% 训练步数时 loss |

这些指标用来判断某方法是前期快、后期慢，还是前期慢、最终好。

---

### 7.3 Descent alignment 指标

Functional update 需要证明局部更新方向没有系统性错误。

定义 predicted descent：

$$
\Delta \mathcal L_{pred}
=
D_F\mathcal L[\Delta F]
$$

定义 actual descent：

$$
\Delta \mathcal L_{actual}
=
\mathcal L_{after}-\mathcal L_{before}
$$

记录：

| metric | 说明 |
|---|---|
| `descent/pred` | predicted descent |
| `descent/actual` | actual descent |
| `descent/ratio` | $\Delta L_{actual}/(\Delta L_{pred}+\epsilon)$ |
| `descent/sign_agreement` | predicted 与 actual 是否同号 |
| `descent/sign_agreement_rate` | 最近窗口 sign agreement 比例 |
| `descent/pred_actual_corr` | predicted / actual 相关性 |
| `descent/pred_neg_actual_pos_count` | 预测下降但实际上升次数 |
| `descent/bad_step_total_positive_loss_increase` | 坏步实际上升总量 |
| `descent/bad_step_max_loss_increase` | 最大坏步幅度 |

pilot 中 `sign_agreement=1.0` 已经饱和，所以 B-next 必须重点看 `descent/ratio` 和 `pred_actual_corr`。

---

### 7.4 KAN edge function 几何指标

| metric | 说明 |
|---|---|
| `kan/phi_prime_mean` | 平均 $|\phi'|$ |
| `kan/phi_prime_p95` | $|\phi'|$ 95 分位 |
| `kan/phi_prime_max` | 最大 $|\phi'|$ |
| `kan/phi_double_prime_energy_mean` | 平均 $\int|\phi''|^2$ |
| `kan/phi_double_prime_energy_max` | 最大 $\int|\phi''|^2$ |
| `kan/sobolev_norm_mean` | 平均 Sobolev norm |
| `kan/sobolev_norm_max` | 最大 Sobolev norm |
| `kan/coefficient_norm` | coefficient norm |
| `kan/edge_update_norm_coeff` | coefficient update norm |
| `kan/edge_update_norm_function` | function-space update norm |
| `kan/edge_saturation_rate` | edge function 是否进入饱和区 |

注意：curvature 不是越低越好。如果任务需要高频，例如 $\sin x+0.3\sin 5x$，合理的 curvature 是必要的。因此需要增加 target-normalized curvature：

$$
\text{excess\_curvature}
=
\max(0, C_{model}-C_{target})
$$

记录：

| metric | 说明 |
|---|---|
| `kan/target_curvature` | 目标函数 curvature |
| `kan/excess_curvature` | 超出目标的 curvature |

---

### 7.5 Jacobian / credit stability 指标

KAN 的 credit transport 由 $\phi'(x)$ 控制：

$$
g_{x_i}=\sum_jg_j\phi'_{ji}(x_i)
$$

因此要记录：

| metric | 说明 |
|---|---|
| `jacobian/sigma_max` | 局部 Jacobian 最大奇异值 |
| `jacobian/sigma_min` | 局部 Jacobian 最小奇异值 |
| `jacobian/condition` | $\sigma_{max}/(\sigma_{min}+\epsilon)$ |
| `credit/amplification_mean` | $\|g_x\|/\|g_y\|$ 均值 |
| `credit/amplification_p95` | amplification 95 分位 |
| `credit/amplification_max` | 最大 amplification |

Two Moons pilot 中 Sobolev 的几何稳定性主要看：

1. `jacobian/condition` 低于 Adam；
2. `kan/phi_prime_p95` 低于 Adam；
3. `descent/pred_neg_actual_pos_count` 更低；
4. accuracy 不损失。

---

### 7.6 Metric conditioning 指标

| metric | 说明 |
|---|---|
| `metric/precond_condition` | $\kappa(M+\rho I)$ |
| `metric/eig_min` | 最小特征值 |
| `metric/eig_max` | 最大特征值 |
| `metric/eig_p05` | 特征值 5 分位 |
| `metric/eig_p95` | 特征值 95 分位 |
| `metric/update_amplification` | $\|P\nabla\|/\|\nabla\|$ |
| `metric/noise_amplification_proxy` | 小梯度方向被放大的程度 |
| `metric/damping_rho` | 当前 damping |
| `metric/sobolev_alpha` | 当前 $\alpha$ |
| `metric/sobolev_beta` | 当前 $\beta$ |

对于 RKHS，必须记录：

| metric | 说明 |
|---|---|
| `rkhs/kernel_lengthscale` | kernel lengthscale |
| `rkhs/raw_condition` | 未修正条件数 |
| `rkhs/damped_condition` | damping 后条件数 |
| `rkhs/clipped_condition` | clipping 后条件数 |
| `rkhs/allowed_in_main_comparison` | 是否进入正式比较，0/1 |

---

### 7.7 资源与成本指标

| metric | 说明 |
|---|---|
| `perf/step_time_ms` | 单步耗时 |
| `perf/samples_per_sec` | 吞吐 |
| `compute/preconditioner_time_ms` | 构造/应用 preconditioner 耗时 |
| `compute/functional_update_time_ms` | functional update 耗时 |
| `compute/adam_update_time_ms` | Adam update 耗时 |
| `memory/peak_allocated_mb` | peak memory |
| `memory/preconditioner_mb` | preconditioner 存储 |
| `memory/kan_coeff_mb` | KAN coefficients 存储 |

Functional update 如果性能略好但耗时极高，也不能算实际优势。

---

## 8. W&B Dashboard 设计

Stage B-next 应该建立一个独立 dashboard，至少包含八个页面。

---

### Page 1：Stage B-next Overview

目标：快速比较所有方法的最终效果、速度、稳定性。

图表：

| 图 | x | y | color / group |
|---|---|---|---|
| Final test loss | method | `test/loss` | task |
| Final test acc | method | `test/acc` | task |
| Loss AUC | method | `conv/loss_auc_val` | task |
| Steps-to-target | method | `conv/steps_to_target_loss` | task |
| Geometry summary | method | `jacobian/condition` | task |
| Preconditioner condition | method | `metric/precond_condition` | task |

必须有一个 summary table，字段包括：

```text
method
task
seed_mean_test_loss
seed_std_test_loss
loss_auc
steps_to_target
jacobian_condition
phi_prime_p95
curvature
precond_condition
bad_steps
step_time_ms
```

---

### Page 2：Regression Functional Update

目标：分析回归任务中 diag、Sobolev、diag→Sobolev 的速度与最终质量。

图表：

1. train loss curve：按 method 分组；
2. val loss curve：按 method 分组；
3. test loss bar chart；
4. loss AUC bar chart；
5. steps-to-target bar chart；
6. fixed-step loss line：10%、25%、50%、75%；
7. task × method heatmap：颜色为 normalized test loss；
8. task × method heatmap：颜色为 normalized loss AUC。

关键看法：

- 如果 diag 的 AUC 明显低，但 final test loss 不如 Sobolev，说明 diag 是 warmup 方法；
- 如果 diag→Sobolev 的 AUC 接近 diag，final test loss 接近 Sobolev，则 schedule 成立。

---

### Page 3：Classification Hybrid Update

目标：验证 hybrid 是否是分类任务正确形态。

图表：

1. accuracy curve：Adam all params vs hybrid diag vs hybrid Sobolev；
2. loss curve；
3. final acc bar；
4. loss AUC bar；
5. confusion matrix，可选；
6. underfit diagnostic：train acc vs val acc；
7. non-KAN optimizer ablation：other SGD vs other AdamW。

必须单独展示：

```text
all_functional_other_sgd
hybrid_diag_other_adamw
hybrid_sobolev_other_adamw
all_adamw
```

如果 `all_functional_other_sgd` 几何很好但 accuracy 差，应标记为欠拟合案例，而不是 functional update 局部方向错误。

---

### Page 4：diag→Sobolev Schedule Analysis

目标：判断 schedule 是否真的结合了两者优势。

图表：

1. warmup steps $T_w$ vs final test loss；
2. warmup steps $T_w$ vs loss AUC；
3. warmup steps $T_w$ vs Jacobian condition；
4. schedule transition vertical line：训练曲线上标出从 diag 切到 Sobolev 的 step；
5. hard switch vs soft mix 对比；
6. $\lambda_t$ 曲线与 loss 曲线叠图。

推荐 sweep：

$$
T_w\in\{0,0.1T,0.25T,0.5T,0.75T\}
$$

其中 $T_w=0$ 是纯 Sobolev，$T_w=T$ 是纯 diag。

soft mix：

$$
\lambda_t\in\{\text{linear},\text{sigmoid},\text{cosine}\}
$$

---

### Page 5：Function Geometry

目标：判断不同方法对 KAN edge functions 的影响。

图表：

1. $\phi'_\text{p95}$ over time；
2. $\phi'_\text{max}$ over time；
3. curvature over time；
4. Sobolev norm over time；
5. coefficient norm over time；
6. function update norm vs coefficient update norm；
7. curvature vs test loss Pareto；
8. $\phi'_\text{p95}$ vs Jacobian condition scatter。

解释规则：

- $\phi'$ 过大可能导致 credit amplification；
- curvature 过低可能欠拟合高频任务；
- curvature 过高可能过拟合或产生不稳定；
- Sobolev 的目标不是简单压低 curvature，而是让 curvature 与任务需求匹配。

---

### Page 6：Jacobian and Credit Stability

目标：验证 functional update 是否改善 credit transport 相关几何。

图表：

1. Jacobian condition over time；
2. $\sigma_{max}$ over time；
3. $\sigma_{min}$ over time；
4. credit amplification mean / p95 over time；
5. method × task heatmap：Jacobian condition；
6. method × task heatmap：credit amplification p95。

关键判断：

如果某方法 accuracy / loss 与 Adam 接近，但：

$$
\kappa(J) < \kappa(J_{Adam})
$$

且：

$$
\phi'_\text{p95}<\phi'_{\text{p95},Adam}
$$

则它在局部几何上更健康。

---

### Page 7：Descent Alignment

目标：判断 predicted descent 与 actual descent 是否一致。

图表：

1. predicted vs actual descent scatter；
2. descent ratio histogram；
3. bad step count over training；
4. bad step severity bar；
5. sign agreement rate over time；
6. pred_actual_corr over time。

因为 pilot 中 sign agreement 饱和，B-next 要重点看：

```text
descent/ratio_median
descent/ratio_p10
descent/pred_actual_corr
descent/bad_step_total_positive_loss_increase
```

---

### Page 8：Metric Conditioning and RKHS Quarantine

目标：判断 preconditioner 是否病态。

图表：

1. condition number bar：diag / Sobolev / RKHS；
2. eigenvalue spectrum；
3. damping $\rho$ vs condition number heatmap；
4. RKHS lengthscale vs condition number heatmap；
5. condition number vs test loss scatter；
6. update amplification histogram。

RKHS 必须有 gating：

如果：

$$
\kappa(M_{\text{RKHS}}+\rho I)>500
$$

则该 run 标记为：

```text
rkhs/allowed_in_main_comparison = 0
```

不进入正式方法对比，只进入 failure analysis。

---

## 9. B0：Pilot reproduction & calibration

B0 的目标是复现 Stage B pilot，并确认所有新增指标可用。B0 不需要大规模 sweep，但每个已有任务必须至少跑 3 seeds，正式结论使用 5 seeds。

### 9.1 任务

| task | 说明 |
|---|---|
| `sin1d` | 低频回归 |
| `sin5` | 中频回归 |
| `sincos2d` | 二维回归 |
| `two_moons` | 分类 sanity |

### 9.2 方法

| method | 说明 |
|---|---|
| `coeff_adam` | baseline |
| `diag` | 复现快速 preconditioner |
| `sobolev` | 复现最终质量 preconditioner |
| `rkhs_raw` | 复现失败案例 |

### 9.3 必须新增的记录

B0 不只是复现实验，还必须补充原 pilot 缺失的指标：

```text
generalization/train_test_gap_loss
descent/ratio_median
descent/ratio_p10
descent/pred_actual_corr
descent/bad_step_total_positive_loss_increase
kan/target_curvature
kan/excess_curvature
metric/eig_min
metric/eig_max
metric/update_amplification
perf/step_time_ms
```

### 9.4 B0 成功标准

B0 通过当且仅当：

1. 新指标没有缺失；
2. 复现出 diag AUC 快、Sobolev final test loss 好、RKHS 条件数高的基本趋势；
3. W&B dashboard 可以按 task / method / seed 聚合。

如果 B0 不能复现 pilot，不能进入 B1-B4。

---

## 10. B1：diag→Sobolev schedule 实验

B1 是 Stage B-next 的第一个核心实验。它要验证：

$$
\text{diag warmup} \rightarrow \text{Sobolev finetune}
$$

是否同时获得 diag 的速度和 Sobolev 的最终质量。

### 10.1 实验设置

任务：

| task | 第一轮是否必做 |
|---|---|
| sin1d | 是 |
| sin5 | 是 |
| sincos2d | 是 |
| noisy sin5 | 是 |
| Two Moons | 是 |
| MNIST | 第二轮 |
| Fashion-MNIST | 第二轮 |

方法：

| method | 说明 |
|---|---|
| AdamW all | 强 baseline |
| diag | 纯 diag |
| Sobolev | 纯 Sobolev |
| hard diag→Sobolev | hard switch schedule |
| soft diag→Sobolev | continuous mix schedule |

### 10.2 warmup sweep

设总训练步数为 $T$。扫：

$$
T_w\in\{0,0.1T,0.25T,0.5T,0.75T,T\}
$$

其中：

- $T_w=0$ 等价于纯 Sobolev；
- $T_w=T$ 等价于纯 diag。

### 10.3 soft mix sweep

$$
P_t=(1-\lambda_t)P_{diag}+\lambda_tP_{Sob}
$$

比较：

| schedule | 定义 |
|---|---|
| linear | $\lambda_t=t/T$ |
| sigmoid | 慢启动，中段快速切换 |
| cosine | 平滑切换 |

### 10.4 B1 主要指标

必须上传：

```text
conv/loss_auc_val
conv/steps_to_target_loss
conv/time_to_target_loss_sec
test/loss
test/acc
jacobian/condition
kan/phi_prime_p95
kan/phi_double_prime_energy_mean
descent/sign_agreement_rate
descent/ratio_median
metric/precond_condition
perf/step_time_ms
```

### 10.5 B1 成功标准

#### 弱成功

存在某个 diag→Sobolev schedule，使得：

$$
\text{test loss} \leq 1.1\times \text{test loss}_{Sobolev}
$$

且：

$$
\text{loss AUC} \leq 1.1\times \text{loss AUC}_{diag}
$$

#### 中等成功

存在 schedule 同时满足：

$$
\text{test loss} \leq \text{test loss}_{Sobolev}
$$

或最多差 $5\%$，并且：

$$
\text{loss AUC} < \text{loss AUC}_{Sobolev}
$$

同时：

$$
\kappa(J) < \kappa(J_{Adam})
$$

#### 强成功

存在 schedule 同时满足：

1. final test loss 不差于 Sobolev；
2. loss AUC 不差于 diag；
3. steps-to-target 优于 Adam；
4. Jacobian condition 低于 Adam；
5. no increase in bad steps；
6. wall-clock overhead 小于 $1.3\times$ AdamW。

---

## 11. B2：classification hybrid update 实验

B2 的目标是验证：functional update 应用于 KAN coefficients，非 KAN 参数使用 AdamW，这是否是分类任务上的正确默认设置。

### 11.1 任务

第一轮：

| task | 目的 |
|---|---|
| Two Moons | sanity，复现 hybrid 现象 |
| MNIST | 简单真实分类 |
| Fashion-MNIST | 稍难真实分类 |

第二轮：

| task | 目的 |
|---|---|
| CIFAR-10 small | 图像结构测试 |

### 11.2 方法

| method | KAN coeff | non-KAN params |
|---|---|---|
| all_adamw | AdamW | AdamW |
| all_functional_other_sgd | functional | SGD |
| hybrid_diag | diag | AdamW |
| hybrid_sobolev | Sobolev | AdamW |
| hybrid_diag_to_sobolev | diag→Sobolev | AdamW |

### 11.3 学习率 sweep

KAN coefficient lr：

$$
\eta_{coeff}\in\{0.003,0.01,0.03,0.1,0.3\}
$$

non-KAN AdamW lr：

$$
\eta_{nonkan}\in\{10^{-4},3\times10^{-4},10^{-3}\}
$$

正式比较时，每个方法应使用各自 validation-selected 最佳 lr，但要报告 tuning budget，避免 unfair advantage。

### 11.4 B2 必须记录的 underfit 指标

为了区分“几何稳定”与“欠拟合”，必须记录：

```text
train/acc
val/acc
test/acc
train/loss
val/loss
generalization/val_train_gap_acc
model/logit_norm
model/class_margin_mean
model/class_margin_p10
kan/edge_update_norm_coeff
kan/edge_update_norm_function
```

如果 train acc 也低，说明是 underfit；如果 train acc 高、val acc 低，才是泛化问题。

### 11.5 B2 成功标准

#### 弱成功

hybrid Sobolev 或 hybrid diag→Sobolev 满足：

$$
\text{test acc} \geq \text{AdamW acc}-2\%
$$

并且 Jacobian condition 或 $\phi'_\text{p95}$ 明显低于 AdamW。

#### 中等成功

满足：

$$
\text{test acc} \geq \text{AdamW acc}-1\%
$$

且：

$$
\text{loss AUC} \leq 1.25\times \text{AdamW loss AUC}
$$

并且：

$$
\kappa(J)<\kappa(J_{AdamW})
$$

#### 强成功

hybrid diag→Sobolev 同时满足：

1. test acc 不低于 AdamW；
2. loss AUC 接近 AdamW 或更低；
3. Jacobian condition 更低；
4. $\phi'_\text{p95}$ 更低；
5. bad steps 更少；
6. 5 seeds 上稳定。

---

## 12. B3：Sobolev metric sweep

B3 的目标是系统理解 Sobolev 的超参数区域，而不是只报告单点结果。

### 12.1 Sweep 参数

$$
\alpha\in\{0,10^{-4},10^{-3},10^{-2},10^{-1},1\}
$$

$$
\beta\in\{0,10^{-5},10^{-4},10^{-3},10^{-2},10^{-1}\}
$$

$$
\rho\in\{10^{-6},10^{-5},10^{-4},10^{-3},10^{-2},10^{-1}\}
$$

不建议全网格一次性跑完。建议采用两阶段：

1. coarse sweep：少量 seeds，找稳定区域；
2. focused sweep：稳定区域内 5 seeds。

### 12.2 关键图

必须画：

1. $\alpha,\beta$ heatmap：颜色为 test loss；
2. $\alpha,\beta$ heatmap：颜色为 loss AUC；
3. $\alpha,\beta$ heatmap：颜色为 Jacobian condition；
4. $\rho$ vs precond condition；
5. precond condition vs test loss scatter；
6. curvature vs test loss Pareto。

### 12.3 B3 成功标准

B3 不要求某个单点最佳，而要求发现一个稳定区域。

#### 弱成功

存在多个 Sobolev 配置满足：

$$
\text{test loss} < \text{Adam test loss}
$$

或分类中：

$$
\text{test acc} \geq \text{Adam acc}-2\%
$$

#### 中等成功

存在连续超参区域满足：

$$
\kappa(M+\rho I)<100
$$

且 final quality 不差于 Adam。

#### 强成功

存在 Sobolev 区域同时满足：

1. final test loss / acc 不差于 Adam；
2. loss AUC 不显著更差；
3. Jacobian condition 更低；
4. $\phi'_\text{p95}$ 更低；
5. seeds 方差小。

---

## 13. B4：RKHS conditioning repair

B4 是隔离实验，不应阻塞主线。

### 13.1 目标

判断 RKHS 是因为当前 metric 病态而失败，还是即使修正 conditioning 后仍不适合当前 KAN setup。

### 13.2 修复策略

#### Damping sweep

$$
\rho\in\{10^{-4},10^{-3},10^{-2},10^{-1},1.0\}
$$

#### Lengthscale sweep

若使用 RBF kernel：

$$
K(t,s)=\exp\left(-\frac{(t-s)^2}{2\ell^2}\right)
$$

扫：

$$
\ell\in\{0.05,0.1,0.2,0.5,1.0,2.0\}
$$

#### Eigenvalue clipping

限制：

$$
\kappa(M+\rho I)\leq\kappa_{max}
$$

其中：

$$
\kappa_{max}\in\{100,500,1000\}
$$

### 13.3 RKHS 进入正式比较的门槛

只有满足：

$$
\kappa(M_{RKHS}+\rho I)<500
$$

并且：

$$
\text{update amplification}<10
$$

才允许进入正式 comparison table。

否则标记为：

```text
rkhs/status = ill_conditioned_failed
```

---

## 14. 公平比较规则

Stage B-next 必须遵守公平比较，否则容易误判。

### 14.1 每个方法都要做学习率 sweep

不能只给 functional update 调学习率，而不给 AdamW 调。正式结果应使用 validation-selected best config，但报告 sweep range。

### 14.2 step-matched 与 wall-clock-matched 都要报告

Step-matched 回答：同样 update 次数，谁更好。

Wall-clock-matched 回答：考虑 preconditioner 开销后，谁更好。

记录：

```text
compare/step_matched_test_loss
compare/wallclock_matched_test_loss
compare/wallclock_overhead_vs_adamw
```

### 14.3 多 seed

探索阶段可以 3 seeds，正式结论必须 5 seeds。

报告：

$$
\text{mean}\pm\text{std}
$$

并展示 worst seed。

### 14.4 不只看 best run

所有结论必须使用 group mean，而不是单个 best run。

---

## 15. 结果解释规则

### 15.1 什么情况下说 `diag` 有优势

只有当：

$$
\text{loss AUC}_{diag}<\text{loss AUC}_{Adam}
$$

且：

$$
\text{steps-to-target}_{diag}<\text{steps-to-target}_{Adam}
$$

才能说 diag 收敛更快。

如果 final test loss 不如 Sobolev，则 diag 的定位是 warmup / fast optimizer，而不是最终最佳 optimizer。

---

### 15.2 什么情况下说 Sobolev 泛化更好

如果只看到：

$$
\text{test loss}_{Sob}<\text{test loss}_{Adam}
$$

只能说 Sobolev 的最终 test performance 更好。

若还满足：

$$
\text{train loss}_{Sob}\approx\text{train loss}_{Adam}
$$

但：

$$
\text{test loss}_{Sob}<\text{test loss}_{Adam}
$$

或者：

$$
\text{generalization gap}_{Sob}<\text{generalization gap}_{Adam}
$$

才可以说 Sobolev 更有泛化优势。

---

### 15.3 什么情况下说 Sobolev 几何更稳定

需要至少满足两个以上：

$$
\kappa(J_{Sob})<\kappa(J_{Adam})
$$

$$
\phi'_{p95,Sob}<\phi'_{p95,Adam}
$$

$$
\text{credit amplification}_{p95,Sob}<\text{credit amplification}_{p95,Adam}
$$

$$
\text{bad steps}_{Sob}<\text{bad steps}_{Adam}
$$

且任务 performance 不显著下降。

---

### 15.4 什么情况下说 functional update 是明显优势

分三档。

#### minor advantage

final performance 接近 Adam，几何指标略好，但收敛速度不如 Adam。

#### moderate advantage

至少一个维度有 $20\%$ 以上改善，例如：

$$
\text{loss AUC reduction}>20\%
$$

或：

$$
\text{test loss reduction}>20\%
$$

或：

$$
\kappa(J)\text{ reduction}>20\%
$$

且其他主要指标不明显恶化。

#### strong advantage

同时满足：

1. final test performance 更好或持平；
2. loss AUC 更低或接近；
3. Jacobian condition 更低；
4. $\phi'$ / credit amplification 更可控；
5. wall-clock overhead 可接受。

---

## 16. Failure analysis

每次失败都要记录到 W&B failure table。

| failure type | 触发条件 | 解释 |
|---|---|---|
| `underfit_nonkan_optimizer` | train acc 也低 | 非 KAN 参数优化器太弱 |
| `oversmooth_function` | train loss 高且 curvature 很低 | Sobolev 过强，表达力被压制 |
| `curvature_explosion` | curvature 和 $\phi'$ 同时高 | 函数过度弯曲或不稳定 |
| `bad_descent` | predicted descent < 0 but actual > 0 | 局部线性假设或更新过大失败 |
| `metric_ill_conditioned` | condition number 超阈值 | preconditioner 病态 |
| `update_amplification` | $\|P\nabla\|/\|\nabla\|$ 过高 | 噪声被放大 |
| `slow_convergence` | AUC 显著差于 Adam | 收敛慢 |
| `no_generalization_gain` | test loss 不好且 gap 不低 | 没有泛化收益 |
| `overhead_too_high` | wall-clock 显著过高 | 实用性不足 |

failure table 字段：

```text
failure/type
failure/task
failure/method
failure/seed
failure/step
failure/metric_name
failure/metric_value
failure/threshold
failure/diagnosis
failure/recommended_action
```

---

## 17. Stage B-next 最终通过标准

Stage B-next 不是要求所有方法都赢 Adam，而是要判断 functional update 这条线是否值得进入 Stage C / D。

### 17.1 Weak pass

满足：

1. diag 或 Sobolev 在回归任务上至少一个维度明显优于 Adam；
2. hybrid classification 能恢复到 Adam accuracy 的 $98\%$ 以上；
3. RKHS 失败原因能被明确归因到 conditioning。

### 17.2 Moderate pass

满足：

1. Sobolev 在多数回归任务 final test loss 优于 Adam；
2. diag 在多数回归任务 loss AUC 优于 Adam；
3. hybrid Sobolev 在 Two Moons / MNIST / Fashion-MNIST 上 accuracy 接近 Adam；
4. Sobolev 的 Jacobian condition 或 $\phi'_\text{p95}$ 明显低于 Adam；
5. diag→Sobolev schedule 至少在两个任务上同时改善 AUC 和 final quality。

### 17.3 Strong pass

满足：

1. diag→Sobolev 在回归和分类任务上成为 Pareto 最优或接近 Pareto 最优；
2. final test performance 不低于 Adam；
3. loss AUC 不高于 Adam 或只轻微高于 Adam；
4. Jacobian condition、credit amplification、$\phi'$ 明显更稳定；
5. wall-clock overhead $<1.3\times$ AdamW；
6. 5 seeds 下稳定。

若达到 moderate pass，即可进入 Stage C：analytic KAN adjoint / local credit transport。若只达到 weak pass，仍可进入 Stage C，但论文 claim 要保守。若 Stage B-next 失败，则 DG-KAN 的 functional-space update 部分需要重构，Stage C 可作为纯 adjoint / credit transport 实验继续，但不能再把 functional update 当核心优势。

---

## 18. 第一轮执行建议

如果资源有限，第一轮不要一次性跑完整 B-next。建议先跑以下最小但有信息量的组合。

### Round 1：Regression schedule core

任务：

```text
sin1d
sin5
sincos2d
```

方法：

```text
AdamW all
diag
Sobolev
diag_to_sobolev hard switch
```

warmup：

$$
T_w\in\{0.1T,0.25T,0.5T\}
$$

seeds：

$$
3\text{ exploratory},\quad 5\text{ final}
$$

重点看：

```text
test/loss
conv/loss_auc_val
conv/steps_to_target_loss
jacobian/condition
kan/phi_prime_p95
descent/ratio_median
```

---

### Round 2：Two Moons + MNIST hybrid

任务：

```text
two_moons
mnist
```

方法：

```text
all_adamw
all_functional_other_sgd
hybrid_diag
hybrid_sobolev
hybrid_diag_to_sobolev
```

重点看：

```text
test/acc
conv/loss_auc_val
generalization/val_train_gap_acc
jacobian/condition
kan/phi_prime_p95
descent/pred_neg_actual_pos_count
```

---

### Round 3：Sobolev focused sweep

基于 Round 1/2 选出的最佳任务和模型，扫：

$$
\alpha,\beta,\rho
$$

目标是找到稳定区域，而不是单点最优。

---

### Round 4：RKHS repair only if needed

如果论文中仍想保留 RKHS，才跑 B4。否则可以把 RKHS 作为 failure analysis 写入。

---

## 19. 最终建议

Stage B-next 的关键不是证明某个名字叫 Sobolev 的方法必然最好，而是要把 functional update 的作用边界讲清楚。

当前最可能成立的结论是：

> **KAN edge functions 适合使用函数空间预条件更新。`diag` 提供快速尺度修正，`Sobolev` 提供更好的最终函数质量与局部几何控制，`diag→Sobolev` 可能是最优折中。非 KAN 参数应保留 AdamW。RKHS 当前需要先解决 metric conditioning。**

如果 Stage B-next 证明这个结论稳定成立，那么 DG-KAN 后续 Stage C / D 就有坚实基础：先用 analytic KAN adjoint 或 learned router 解决 credit transport，再把 credit 与 functional-space update 组合起来。
