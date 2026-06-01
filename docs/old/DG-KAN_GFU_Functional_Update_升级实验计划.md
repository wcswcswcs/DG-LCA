# DG-KAN Functional Update 升级实验计划：GFU 可行性分析与执行方案

> 文件名：`DG-KAN_GFU_Functional_Update_升级实验计划.md`  
> 主题：在进入 Stage I Vision Scaling 前，升级 DG-KAN 的 functional update  
> 核心方法：Geometry-aware Functional Update, GFU  
> 公式格式：Typora 友好，统一使用 `$...$` 和 `$$...$$`  
> 当前主线：`AnalyticAdj + functional-space update + geometry-aware controller + low-memory primitive`

---

## 0. 总体判断

我认为这个 GFU 方案**值得做，而且应该在 Stage I 之前做**。原因是：当前 DG-KAN 的 credit 侧已经比较扎实，AnalyticAdj 与 FullBP 基本等价，低显存方面 AnalyticAdj 也已经有强结果；现在真正限制后续视觉扩展的是 KAN coefficients 的 optimizer dynamics，而不是 learned router 或 analytic adjoint 正确性。

当前已有结果说明：

1. 静态 functional update 太保守，容易导致 KAN branch under-active；
2. branch-aware / geometry-aware schedule 可以释放 functional update 的优势；
3. Fashion-MNIST 上已经出现强结果：accuracy 超过 AdamW，val-loss AUC 明显更好，$\phi'_{p95}$ 和 Jacobian condition 明显降低；
4. KMNIST 上仍有约 $1\%$ accuracy gap，说明当前 schedule 还没有完全解决 harder classification 的表达力 / 几何折中；
5. Stage G 说明 geometry-aware functional update 在 calibration、label-noise clean accuracy、小数据泛化上有真实收益，但 corruption robustness 仍不足；
6. 继续粗扫 `branch_high`、`jac_high`、`final_scale` 的收益已经进入平台期，因此需要升级 optimizer 机制，而不是继续只调 schedule 阈值。

所以，GFU 的实验目标不应该是泛泛地“再发明一个 optimizer”，而应该是：

$$
\boxed{
\text{在保持 DG-KAN 几何稳定性的前提下，进一步提升 branch utilization、收敛速度、泛化和 harder-task accuracy。}
}
$$

更直接地说：

> GFU 要解决的问题是：现有 geometry-aware schedule 已经能控制 $\phi'$ 和 Jacobian，但它对 KMNIST 等 harder task 仍然偏保守；GFU 要把 branch activation、functional step size、trust region、metric strength 和 late smoothness 放进一个可审计的闭环控制器。

---

## 1. 方案可行性分析

### 1.1 Trust-region functional step：高可行性，优先做

当前 functional update 的基本形式是：

$$
a_{t+1}
=
a_t
-
\eta(M+\rho I)^{-1}\nabla_a L
$$

这个方向通常不是问题，真正的问题是步长。步长太小，KAN branch under-active；步长太大，$\phi'$、Jacobian condition 和 credit amplification 会恶化。因此，引入 trust-region 是合理的。

GFU 中先计算 functional direction：

$$
p_t
=
-(M_t+\rho_t I)^{-1}g_t
$$

其中：

$$
g_t=\nabla_a L_t
$$

然后用 $M$-norm 约束函数空间步长：

$$
\|p_t\|_{M_t}
=
\sqrt{p_t^\top M_t p_t}
$$

若：

$$
\eta_t\|p_t\|_{M_t}>\tau_t
$$

则缩放：

$$
p_t
\leftarrow
p_t
\frac{\tau_t}{\eta_t\|p_t\|_{M_t}+\epsilon}
$$

这个组件非常适合现在的情况，因为你已经在 Stage B/F/G 里记录了 predicted descent、actual descent、bad steps、$\phi'$、Jacobian、branch ratio 等指标，可以直接把原来的 audit 指标升级为 online controller。

可行性判断：

| 维度 | 判断 |
|---|---|
| 实现难度 | 中等 |
| 风险 | 低 |
| 预期收益 | 高，尤其是抑制 over-active branch 和允许更激进 early update |
| 是否优先 | 是，GFU-lite 第一优先级 |

---

### 1.2 Layer-wise branch target controller：高价值，中等风险

已有实验反复说明，branch utilization 是 classification 性能的关键。定义：

$$
r_{branch,k}
=
\frac{
\|\alpha_k b_{k,t}\operatorname{KAN}_k(\operatorname{LN}(h_k))\|
}{
\|h_k\|+\epsilon
}
$$

过去的全局 schedule 只能粗略设定 `branch_boost` 和 `branch_final_scale`。GFU 更进一步，给每层设置目标：

$$
r_{branch,k}^{\star}(t)
$$

并调节每层 coefficient learning rate multiplier：

$$
m_{k,t+1}
=
m_{k,t}
\exp
\left(
\gamma_r
\log
\frac{r_{branch,k}^{\star}(t)+\epsilon}{r_{branch,k,t}+\epsilon}
\right)
$$

同时加 geometry safety factor：

$$
s_{geom,k,t}
=
\min
\left(
1,
\frac{\tau_{\phi}}{\phi'_{p95,k,t}+\epsilon},
\frac{\tau_J}{\kappa(J_{k,t})+\epsilon},
\frac{\tau_s}{|\alpha_k|b_{k,t}\phi'_{p95,k,t}+\epsilon}
\right)^{\gamma_g}
$$

最终：

$$
m_{k,t+1}
\leftarrow
m_{k,t+1}s_{geom,k,t}
$$

这个组件很有价值，因为它直接针对当前 functional update 的主要失败模式：

$$
\text{branch under-active}
\quad
\text{or}
\quad
\text{branch over-active}
$$

可行性判断：

| 维度 | 判断 |
|---|---|
| 实现难度 | 中等 |
| 风险 | 中等，可能振荡 |
| 预期收益 | 高，尤其是 KMNIST remaining gap |
| 是否优先 | 是，但先控制 coeff lr multiplier，再控制 branch scale |

建议先只调 `coeff_lr_multiplier`，不要一开始同时调 `branch_scale`，否则系统自由度太多，失败时难以定位。

---

### 1.3 Metric continuation：高可行性，低风险

现有 `diag_to_sobolev` 是 hard switch：

$$
\operatorname{diag}(M)^{-1}
\rightarrow
M^{-1}
$$

GFU 可以改成软混合：

$$
P_t
=
(1-\lambda_t)\operatorname{diag}(M_t)^{-1}
+
\lambda_t M_t^{-1}
$$

更新方向：

$$
p_t=-P_t g_t
$$

其中 $\lambda_t$ 随训练状态变化。早期 $\lambda_t$ 小，强调快速下降和 branch 激活；后期 $\lambda_t$ 大，强调 Sobolev 几何控制。

可行性判断：

| 维度 | 判断 |
|---|---|
| 实现难度 | 低 |
| 风险 | 低 |
| 预期收益 | 中等 |
| 是否优先 | 是，作为 GFU-lite 的一部分 |

---

### 1.4 Data-weighted metric：有潜力，但要防过拟合

当前 Sobolev metric 多是 grid-based：

$$
M_{grid}
=
\int B(t)B(t)^\top dt
+
\alpha\int B'(t)B'(t)^\top dt
+
\beta\int B''(t)B''(t)^\top dt
$$

但训练时实际输入分布不是均匀 grid，而是：

$$
x_i\sim\mathcal D_{k,t}
$$

因此可以引入 data-weighted metric：

$$
M_{data,k,t}
=
\mathbb E_{x_i\sim\mathcal D_{k,t}}
\left[
B(x_i)B(x_i)^\top
\right]
$$

用 EMA 更新：

$$
\widehat M_{data,k,t}
=
\mu \widehat M_{data,k,t-1}
+
(1-\mu)M_{data,k}^{batch}
$$

混合 metric：

$$
M_{k,t}
=
(1-\lambda_{data,t})M_{grid,k}
+
\lambda_{data,t}\widehat M_{data,k,t}
+
\alpha_tR_{1,k}
+
\beta_tR_{2,k}
+
\rho_{k,t}I
$$

这个组件有潜力提升 harder task，因为它让函数空间 metric 更贴近实际 activation distribution。但也有风险：data metric 过强会让函数只在训练分布高密度区域稳定，而在分布外不受约束，可能损害泛化和 corruption robustness。

可行性判断：

| 维度 | 判断 |
|---|---|
| 实现难度 | 中等 |
| 风险 | 中等偏高 |
| 预期收益 | 中等到高，尤其是 harder data |
| 是否优先 | 放在 GFU-data，不要一开始打开 |

建议先用小 $\lambda_{data}$：

$$
\lambda_{data}\in\{0.1,0.25,0.5\}
$$

并且强制保留 grid Sobolev 项，避免完全 data-local。

---

### 1.5 M-aware momentum：高风险，需要慎重

之前 naive Adam-style moment 造成过 branch explosion。因此 momentum 不能直接在 coefficient Euclidean space 里做。

安全版本是对 functional direction 做 momentum：

$$
p_t=-M_t^{-1}g_t
$$

$$
v_t=\mu v_{t-1}+(1-\mu)p_t
$$

再做 $M$-norm clipping：

$$
v_t
\leftarrow
v_t
\min
\left(
1,
\frac{\tau_t}{\eta_t\|v_t\|_{M_t}+\epsilon}
\right)
$$

更新：

$$
a_{t+1}=a_t+\eta_t v_t
$$

这个比 naive Adam 更安全，因为 momentum 累积的是函数空间方向，而不是 raw coefficient gradient。

可行性判断：

| 维度 | 判断 |
|---|---|
| 实现难度 | 中等 |
| 风险 | 中等偏高 |
| 预期收益 | 不确定，可能帮助 KMNIST |
| 是否优先 | 第三阶段再做 |

GFU-momentum 必须在 GFU-lite / GFU-data 稳定后再打开。若出现：

$$
\text{branch / AdamW}>1.2
$$

或：

$$
\kappa(J)>1.5\kappa(J)_{AdamW}
$$

立即视为 momentum failure。

---

### 1.6 Proximal smoothness / spectral filtering：适合 late phase，不适合 early phase

Sobolev preconditioning 不是显式 regularization。如果要进一步改善泛化，可以加入 late proximal shrink：

$$
\tilde a_{t+1}=a_t+\eta p_t
$$

$$
a_{t+1}
=
(I+2\eta\lambda R)^{-1}\tilde a_{t+1}
$$

其中 $R$ 可以是 $R_1$ 或 $R_2$。

它的作用是后期收缩不必要的高曲率函数变化，尤其适合 label-noise / small-data generalization。但如果太早启用，容易造成 underfit。

可行性判断：

| 维度 | 判断 |
|---|---|
| 实现难度 | 低到中 |
| 风险 | 中等，容易欠拟合 |
| 预期收益 | 泛化 / label noise 上可能有收益 |
| 是否优先 | 放在 GFU-prox，只在 late phase 打开 |

---

## 2. 新实验总目标

GFU 实验不是为了证明“一个新 optimizer 全面赢 AdamW”。更合理的目标是：

$$
\boxed{
\text{验证 GFU 是否能改善 DG-KAN 的 speed-accuracy-geometry-generalization Pareto。}
}
$$

也就是说，至少要同时看四类结果：

1. **收敛速度**：val-loss AUC、steps-to-target、time-to-target；
2. **最终性能**：test acc、test loss、gap vs AdamW；
3. **几何稳定性**：$\phi'_{p95}$、Jacobian condition、credit amplification、curvature；
4. **泛化与校准**：train-test gap、label-noise clean acc、ECE、small-data accuracy。

GFU 的直接目标不是“每个数据集都超过 AdamW”，而是找到：

$$
\text{AdamW-level accuracy}
+
\text{better convergence AUC}
+
\text{lower derivative / Jacobian complexity}
+
\text{better calibration / noise robustness}
$$

---

## 3. 实验总原则

### 3.1 固定 credit 主线

GFU 实验中，默认使用：

$$
\boxed{
\text{credit source} = \text{AnalyticAdj}
}
$$

不要再把 learned router 混进来。原因是 learned router 目前不是主贡献，且会引入额外不确定性。

保留可选 ablation：

$$
\text{FirstOrderDecompAdj}
$$

但 GFU 主实验先以 AnalyticAdj 为准。

---

### 3.2 非 KAN 参数继续使用 AdamW

分类任务中，非 KAN 参数仍然使用 AdamW：

$$
\boxed{
\text{non-KAN params} = \text{AdamW}
}
$$

GFU 只更新 KAN edge coefficients：

$$
a_{jim}
$$

这承接已有结论：functional update 适合 KAN edge functions，不适合替代所有参数 optimizer。

---

### 3.3 从 GFU-lite 到 GFU-full 逐步打开组件

不要一次打开所有机制。实验应按下面顺序：

| 版本 | 打开的组件 |
|---|---|
| GFU-lite | metric continuation + trust-region + branch target controller |
| GFU-data | GFU-lite + data-weighted metric |
| GFU-momentum | GFU-data + functional-direction momentum |
| GFU-prox | GFU-momentum + late proximal smoothness |
| GFU-full | 所有组件，若前面组件均通过 |

每个版本必须和上一版本比较，否则无法知道哪个组件有贡献。

---

## 4. 方法定义

### 4.1 Baselines

必须包括以下方法：

| 方法 | 说明 |
|---|---|
| AdamW | 所有参数 AdamW，强 accuracy baseline |
| Static Functional | 当前静态 diag-to-Sobolev |
| Current Geometry-aware | Stage F 最优 geometry-aware schedule |
| AnalyticAdj + Static Functional | credit 主线 baseline |
| AnalyticAdj + Geometry-aware | 当前主方法 baseline |

---

### 4.2 GFU variants

| 方法 | 说明 |
|---|---|
| GFU-lite | trust region + branch controller + metric continuation |
| GFU-lite no trust | 去掉 trust-region |
| GFU-lite no branch controller | 去掉 branch target controller |
| GFU-data | 加 data-weighted metric |
| GFU-data no grid | 只用 data metric，风险对照 |
| GFU-momentum | 加 functional-direction momentum |
| GFU-prox | late proximal smoothness |
| GFU-full | 当前最强组合 |

---

## 5. 数据集与模型配置

### 5.1 第一阶段数据集

先做：

```text
Fashion-MNIST
KMNIST
```

原因：

- Fashion 上当前 geometry-aware 已经强通过，可以测试 GFU 是否保持优势；
- KMNIST 仍有约 $1\%$ gap，是 GFU 的主要挑战。

MNIST 只做 smoke，不作为核心结论。

---

### 5.2 第二阶段数据集

若第一阶段通过，再做：

```text
MNIST full sanity
EMNIST Balanced small
CIFAR-10 small ConvStem-DGKAN
```

CIFAR-10 small 只有在 GFU 在 Fashion / KMNIST 上表现稳定后进入。

---

### 5.3 默认模型

第一阶段：

```text
model = residual DG-KAN
depth = 4
hidden_dim = 64
basis_count = 16
alpha_init = 1.5
train / val / test = 6000 / 1000 / 1000
batch_size = 256
epochs = 20
seeds = 0,1,2,3,4
```

KMNIST 如果继续存在 gap，可加：

```text
hidden_dim = 96, 128
basis_count = 24, 32
epochs = 30
```

但模型容量 sweep 应在 GFU-lite 过后再做，避免把 optimizer 和 capacity 混在一起。

---

## 6. W&B / CSV config 记录规范

每个 run 必须记录：

```text
exp/stage = GFU
exp/package = GFU0 / GFU1 / ...
data/name
data/train_size
data/val_size
data/test_size
model/depth
model/hidden_dim
model/basis_count
model/alpha_init
credit/source = analytic_adj / first_order
update/type = adamw / static_func / geometry_aware / gfu_lite / gfu_data / ...
gfu/use_trust_region
gfu/use_branch_controller
gfu/use_metric_continuation
gfu/use_data_metric
gfu/use_momentum
gfu/use_prox
gfu/rho_init
gfu/trust_radius_init
gfu/trust_radius_min
gfu/trust_radius_max
gfu/branch_target_early
gfu/branch_target_mid
gfu/branch_target_late
gfu/controller_gamma
gfu/data_metric_lambda
gfu/data_metric_ema
gfu/momentum_mu
gfu/prox_lambda
gfu/prox_start_frac
optim/rest_lr
optim/coeff_lr_base
optim/weight_decay
train/epochs
train/batch_size
seed
```

---

## 7. 必须记录的指标

### 7.1 任务与收敛指标

```text
train/loss
train/acc
val/loss
val/acc
test/loss
test/acc
conv/train_loss_auc
conv/val_loss_auc
conv/val_auc_improvement_vs_adamw
conv/steps_to_target_val_loss
conv/steps_to_target_val_acc
conv/time_to_target_val_loss
conv/time_to_target_val_acc
conv/loss_at_10pct_steps
conv/loss_at_25pct_steps
conv/loss_at_50pct_steps
conv/loss_at_75pct_steps
```

特别重要的是：

$$
\text{val AUC improvement vs AdamW}
$$

和：

$$
\text{time-to-target}
$$

因为 GFU 单步可能比 AdamW 贵，不能只看 step-wise AUC。

---

### 7.2 Geometry 指标

```text
kan/phi_prime_p95
kan/phi_prime_max
kan/curvature_mean
kan/curvature_p95
kan/sobolev_norm_mean
kan/high_freq_energy
jacobian/condition_mean
jacobian/condition_max
credit/amplification_mean
credit/amplification_p95
credit/noise_gain_p95
geometry/effective_alpha_phi_p95
```

其中：

$$
\text{effective alpha-phi}
=
|\alpha_k|b_{k,t}\phi'_{p95,k}
$$

必须记录，因为它更接近实际 branch Jacobian 强度。

---

### 7.3 Branch utilization 指标

```text
branch/layer_k/target_ratio
branch/layer_k/actual_ratio
branch/layer_k/target_error
branch/layer_k/controller_multiplier
branch/layer_k/branch_scale
branch/layer_k/effective_coeff_lr
branch/global/branch_over_adamw
branch/global/no_kan_acc_drop
branch/global/kan_logit_delta_norm
branch/global/underactive_rate
branch/global/overactive_rate
```

定义：

$$
\text{target error}_{k,t}
=
\frac{
r_{branch,k,t}-r^\star_{branch,k,t}
}{
r^\star_{branch,k,t}+\epsilon
}
$$

---

### 7.4 Trust-region / descent 指标

```text
gfu/layer_k/rho
gfu/layer_k/trust_radius
gfu/layer_k/update_m_norm
gfu/layer_k/trust_clip_rate
gfu/global/trust_clip_rate
descent/predicted_delta
descent/actual_delta
descent/ratio
descent/ratio_ema
descent/ratio_p10
descent/ratio_p90
descent/bad_step_rate
descent/rho_increase_count
descent/trust_shrink_count
descent/trust_expand_count
```

核心是：

$$
q_t
=
\frac{
\Delta L_{actual,t}
}{
\Delta L_{pred,t}+\epsilon
}
$$

如果 $q_t$ 经常小于 0，说明 trust-region 或 metric 有问题。

---

### 7.5 Metric 指标

```text
metric/layer_k/cond
metric/layer_k/eig_min
metric/layer_k/eig_max
metric/layer_k/data_mix_lambda
metric/layer_k/data_metric_cond
metric/layer_k/grid_metric_cond
metric/layer_k/rho
metric/layer_k/diag_to_full_lambda
metric/global/max_cond
metric/global/mean_cond
```

如果 `data_metric_cond` 明显高于 grid metric，且出现 overfit，需要降低 data mix。

---

### 7.6 Momentum / prox 指标

如果启用 momentum：

```text
momentum/layer_k/buffer_m_norm
momentum/layer_k/direction_cos_current
momentum/layer_k/momentum_to_current_ratio
momentum/global/explosion_count
```

如果启用 prox：

```text
prox/layer_k/shrink_norm
prox/layer_k/shrink_ratio
prox/global/shrink_rate
smooth/high_freq_energy_before
smooth/high_freq_energy_after
smooth/curvature_before
smooth/curvature_after
```

---

### 7.7 泛化与校准指标

```text
generalization/train_test_gap_loss
generalization/train_test_gap_acc
calibration/ece
calibration/nll
calibration/brier
noise/clean_test_acc
noise/noisy_train_acc
noise/noise_memorization_rate
noise/clean_recovery_rate
corruption/mean_acc
corruption/worst_acc
```

GFU 若号称泛化更好，必须记录这些指标，避免把欠拟合误判为泛化。

---

### 7.8 Cost 指标

```text
compute/step_time_ms
compute/functional_update_time_ms
compute/metric_build_time_ms
compute/trust_region_time_ms
compute/controller_time_ms
compute/overhead_vs_adamw
memory/peak_allocated_mb
memory/metric_state_mb
memory/controller_state_mb
memory/momentum_state_mb
```

GFU 的额外开销必须可控。建议 gate：

$$
\text{step time overhead vs current geometry-aware}<1.2\times
$$

---

## 8. W&B dashboard 设计

### Page 1：GFU scorecard

展示每个方法在 Fashion / KMNIST 上的总览：

| method | test acc | acc gap | val AUC improvement | $\phi'$ reduction | J reduction | ECE reduction | step time |
|---|---:|---:|---:|---:|---:|---:|---:|

目标：一眼判断 GFU 是否优于 current geometry-aware。

---

### Page 2：Convergence dynamics

图：

1. train loss curve；
2. val loss curve；
3. val AUC bar；
4. steps-to-target bar；
5. time-to-target bar；
6. loss-at-fixed-fraction plot。

必须同时画 step-based 和 wall-clock-based 曲线。

---

### Page 3：Branch controller

图：

1. layer-wise target vs actual branch ratio；
2. global branch / AdamW over time；
3. controller multiplier over time；
4. branch scale over time；
5. no-KAN drop by method；
6. branch ratio vs test acc scatter。

目标：判断 GFU 是否真的控制 branch，而不是偶然。

---

### Page 4：Trust-region and descent

图：

1. predicted vs actual descent scatter；
2. descent ratio histogram；
3. trust radius over time；
4. trust clip rate over time；
5. rho over time；
6. bad step rate by method。

目标：判断 trust-region 是否稳定 optimizer，而不是过度保守。

---

### Page 5：Metric diagnostics

图：

1. metric condition over time；
2. data mix lambda over time；
3. data metric condition vs grid metric condition；
4. diag-to-full lambda over time；
5. metric condition vs val loss scatter；
6. metric condition vs $\phi'$ scatter。

---

### Page 6：Geometry

图：

1. $\phi'_{p95}$ over training；
2. curvature over training；
3. Jacobian condition over training；
4. effective alpha-phi over training；
5. credit amplification p95 over training；
6. geometry metrics vs test acc Pareto plot。

---

### Page 7：Generalization

图：

1. train-test gap loss；
2. train-test gap acc；
3. ECE by method；
4. label-noise clean acc；
5. corruption mean acc；
6. curvature vs train-test gap scatter；
7. $\phi'$ vs ECE scatter。

---

### Page 8：Component ablation

图：

1. GFU-lite vs no-trust vs no-branch；
2. GFU-lite vs GFU-data；
3. GFU-data vs GFU-momentum；
4. GFU-momentum vs GFU-prox；
5. per-component gain table。

---

### Page 9：Failure analysis

Failure table 包含：

```text
failure/type
failure/dataset
failure/method
failure/seed
failure/step
failure/layer
failure/metric_name
failure/metric_value
failure/threshold
failure/diagnosis
failure/recommended_action
```

---

## 9. 实验包设计

## GFU-0：Smoke 与实现正确性

### 目标

确认 GFU-lite 不会破坏现有 functional update pipeline。

### 设置

```text
datasets = Two Moons, MNIST smoke, Fashion-MNIST smoke
depth = 2
hidden_dim = 32
basis_count = 8
epochs = 2
seeds = 0,1
```

### 方法

```text
AdamW
current geometry-aware
GFU-lite
GFU-lite no trust
GFU-lite no branch controller
```

### 成功标准

$$
\text{NaN/Inf count}=0
$$

$$
\text{test acc gap vs current geometry-aware}<1\%
$$

$$
\text{bad step rate}_{GFU}<\text{bad step rate}_{current}+5\%
$$

如果 GFU-lite smoke 不稳定，停止后续实验。

---

## GFU-1：主任务对比，Fashion / KMNIST

### 目标

验证 GFU-lite 是否比当前 geometry-aware schedule 更好。

### 设置

```text
datasets = Fashion-MNIST, KMNIST
depth = 4
hidden_dim = 64
basis_count = 16
alpha_init = 1.5
train / val / test = 6000 / 1000 / 1000
epochs = 20
batch_size = 256
seeds = 0,1,2,3,4
```

### 方法

```text
AdamW
static functional
current geometry-aware best
GFU-lite
GFU-lite no trust
GFU-lite no branch controller
```

### Fashion 成功标准

当前 best Fashion geometry-aware 已经很强，因此 GFU 不能退步。GFU-lite 至少应满足：

$$
\text{test acc}\geq \text{current geometry-aware}-0.3\%
$$

$$
\text{val AUC improvement vs AdamW}>15\%
$$

$$
\phi'_{p95}\text{ reduction vs AdamW}>20\%
$$

$$
\text{J condition reduction vs AdamW}>35\%
$$

强成功：

$$
\text{test acc}>\text{current geometry-aware}
$$

且：

$$
\text{step time overhead}<1.2\times
$$

### KMNIST 成功标准

当前 best KMNIST gap 约 $1.06\%$。GFU-lite 目标是缩小 gap：

$$
\text{test acc gap vs AdamW}<0.8\%
$$

并且：

$$
\text{val AUC improvement vs AdamW}>10\%
$$

$$
\phi'_{p95}\text{ reduction}>30\%
$$

$$
\text{J condition reduction}>15\%
$$

强成功：

$$
\text{test acc gap vs AdamW}<0.5\%
$$

---

## GFU-2：Trust-region ablation

### 目标

判断 trust-region 是否真的改善 step-size control。

### 设置

使用 GFU-1 的 Fashion / KMNIST 配置。

### 方法

```text
GFU-lite
GFU-lite no trust
GFU-lite M-norm clipping only
GFU-lite descent-ratio damping only
GFU-lite M-norm clipping + descent-ratio damping
```

### 关键指标

```text
descent/bad_step_rate
descent/ratio_median
descent/ratio_p10
gfu/trust_clip_rate
gfu/rho_increase_count
branch/branch_over_adamw
kan/phi_prime_p95
jacobian/condition_max
test/acc
val_auc
```

### 判定

Trust-region 有价值，如果：

$$
\text{bad step rate 降低}
$$

且：

$$
\text{geometry spike 减少}
$$

并且 test acc / AUC 不明显退步。

若 `trust_clip_rate > 0.8` 且 branch under-active，则 trust radius 太小。

---

## GFU-3：Branch controller ablation

### 目标

判断 layer-wise branch target controller 是否比固定 branch schedule 更好。

### 设置

```text
datasets = Fashion-MNIST, KMNIST
methods = current geometry-aware, GFU fixed target, GFU layer-wise target
```

### Target schedules

测试：

```text
target schedule A:
  early = 0.60
  mid = 0.45
  late = 0.35

target schedule B:
  early = 0.70
  mid = 0.50
  late = 0.40

target schedule C:
  early = 0.55
  mid = 0.45
  late = 0.35
```

### 指标

```text
branch/layer_k/target_ratio
branch/layer_k/actual_ratio
branch/layer_k/target_error
branch/global/underactive_rate
branch/global/overactive_rate
test/acc
val_auc
phi_prime_p95
jacobian_condition
no_kan_drop
```

### 判定

Controller 有价值，如果：

1. branch target error 明显低于固定 schedule；
2. no-KAN drop 接近 AdamW 或 current geometry-aware；
3. accuracy 或 AUC 提升；
4. $\phi'$ / Jacobian 不恶化。

---

## GFU-4：Data-weighted metric 实验

### 目标

判断 data-weighted metric 是否改善 harder task，尤其 KMNIST。

### 设置

在 GFU-lite 最佳配置上加入 data metric。

### 方法

```text
GFU-lite
GFU-data lambda=0.1
GFU-data lambda=0.25
GFU-data lambda=0.5
GFU-data lambda schedule 0.5 -> 0.1
GFU-data no grid
```

### 指标

```text
metric/data_mix_lambda
metric/data_metric_cond
metric/grid_metric_cond
metric/global/max_cond
test/acc
val_auc
train_test_gap_loss
ece
corruption_mean_acc
phi_prime_p95
curvature
```

### 判定

Data metric 有价值，如果：

$$
\text{KMNIST test acc gap 缩小}
$$

或：

$$
\text{val AUC / ECE / train-test gap 改善}
$$

且：

$$
\text{data metric condition 不失控}
$$

`GFU-data no grid` 若表现明显过拟合，则说明 grid Sobolev prior 必须保留。

---

## GFU-5：M-aware momentum 实验

### 目标

测试是否能引入 Adam-like dynamics，同时避免 naive Adam moment 的 branch explosion。

### 方法

```text
GFU-data
GFU-data + functional-direction momentum mu=0.8
GFU-data + functional-direction momentum mu=0.9
GFU-data + M-aware Adam + trust region
naive Sobolev Adam moment
```

### 指标

```text
momentum/buffer_m_norm
momentum/direction_cos_current
momentum/momentum_to_current_ratio
branch/branch_over_adamw
geometry/effective_alpha_phi_p95
jacobian/condition_max
kan/phi_prime_p95
val_auc
test_acc
bad_step_rate
```

### 判定

Momentum 成功，如果：

$$
\text{val AUC improvement 提升}
$$

或：

$$
\text{KMNIST acc gap 缩小}
$$

且不出现：

$$
\text{branch / AdamW}>1.2
$$

或：

$$
\kappa(J)>1.5\kappa(J)_{AdamW}
$$

如果 naive momentum 爆炸而 M-aware momentum 不爆炸，这是很有价值的机制证据。

---

## GFU-6：Late proximal smoothness / spectral filtering

### 目标

测试 explicit smoothness 是否进一步改善泛化和 label-noise robustness。

### 设置

在 GFU-data 或 GFU-momentum 最佳配置上做 late-phase smoothness。

### 方法

```text
GFU-best
GFU-best + prox lambda=1e-4
GFU-best + prox lambda=1e-3
GFU-best + prox lambda=1e-2
GFU-best + spectral filtering late
```

只在训练后半段启用：

$$
t/T > 0.5
$$

或：

$$
t/T > 0.7
$$

### 指标

```text
test_acc
val_auc
train_test_gap_loss
label_noise_clean_acc
label_noise_memorization_rate
ece
curvature_mean
high_freq_energy
branch_ratio
no_kan_drop
```

### 判定

Prox / spectral filtering 成功，如果它改善：

1. ECE；
2. label-noise clean accuracy；
3. train-test gap；
4. curvature / high-frequency energy；

同时不造成：

$$
\text{train acc gap}>2\%
$$

或 branch under-active。

---

## GFU-7：Generalization stress

### 目标

验证 GFU 是否比当前 geometry-aware 在泛化上更强。

### 设置

```text
datasets = Fashion-MNIST, KMNIST
train_size = 500, 1000, 2000, 6000
label_noise = 0.1, 0.2, 0.4
corruption eval = same as Stage G
methods = AdamW, current geometry-aware, GFU-best
```

### 指标

```text
small_data/test_acc
small_data/train_test_gap_loss
noise/clean_test_acc
noise/noise_memorization_rate
noise/clean_recovery_rate
corruption/mean_acc
corruption/worst_acc
calibration/ece
curvature
phi_prime_p95
```

### 判定

GFU 泛化更强，如果：

$$
\text{GFU clean test acc} \geq \text{current geometry-aware}
$$

并且在至少两个 stress setting 中：

$$
\text{ECE lower}
$$

或：

$$
\text{train-test gap lower}
$$

或：

$$
\text{label-noise clean acc higher}
$$

---

## GFU-8：CIFAR-10 small readiness

### 目标

决定 GFU 是否可以进入 Stage I Vision Scaling 的 CIFAR-10 small。

### 条件

只有当以下条件满足时进入 CIFAR：

1. Fashion 不低于 current geometry-aware；
2. KMNIST acc gap vs AdamW < $1\%$ 或 GFU 比 current geometry-aware 明显好；
3. GFU cost overhead < $1.25\times$；
4. ECE / geometry 不退步。

### CIFAR-10 small 设置

```text
model = ConvStem-DGKAN
depth = 4
hidden_dim = 128
basis_count = 16
grouped KAN optional
methods = AdamW, current geometry-aware, GFU-best
```

### 指标

```text
cifar/test_acc
cifar/val_auc
cifar/ece
cifar/branch_over_adamw
cifar/no_kan_drop
cifar/phi_prime_p95
cifar/jacobian_condition
cifar/memory_peak
cifar/step_time
```

---

## 10. 通过标准

### 10.1 GFU-lite 通过

Fashion：

$$
\text{test acc}\geq\text{current geometry-aware}-0.3\%
$$

KMNIST：

$$
\text{test acc gap vs AdamW}<0.8\%
$$

同时：

$$
\text{val AUC improvement vs AdamW}>10\%
$$

$$
\phi'_{p95}\text{ reduction}>25\%
$$

$$
\text{J condition reduction}>15\%
$$

---

### 10.2 GFU-data 通过

GFU-data 必须优于 GFU-lite 至少一个方面：

1. KMNIST acc gap 缩小；
2. ECE 降低；
3. label-noise clean acc 提升；
4. train-test gap 降低；

并且不能造成 metric condition 或 overfitting failure。

---

### 10.3 GFU-momentum 通过

Momentum 只有在以下条件满足时保留：

$$
\text{AUC improvement 增加}
$$

或：

$$
\text{KMNIST accuracy gap 缩小}
$$

同时：

$$
\text{branch / AdamW}<1.2
$$

$$
\kappa(J)<1.5\kappa(J)_{AdamW}
$$

---

### 10.4 GFU-prox 通过

GFU-prox 主要为泛化服务。通过标准：

$$
\text{label-noise clean acc 提升}
$$

或：

$$
\text{ECE 降低}
$$

或：

$$
\text{train-test gap 降低}
$$

且：

$$
\text{train acc 不明显下降}
$$

---

### 10.5 GFU 强通过

GFU-best 若满足：

Fashion：

$$
\text{test acc}\geq\text{AdamW}
$$

KMNIST：

$$
\text{test acc gap vs AdamW}<0.5\%
$$

并且两个数据集都满足：

$$
\text{val AUC improvement}>10\%
$$

$$
\phi'_{p95}\text{ reduction}>25\%
$$

$$
\text{J condition reduction}>20\%
$$

$$
\text{ECE reduction}>10\%
$$

则可作为 Stage I Vision Scaling 的默认 functional optimizer。

---

## 11. Failure table

每个失败都要记录：

| failure type | 触发条件 | 解释 |
|---|---|---|
| `branch_underactive` | branch / AdamW < 0.4 | KAN branch 没充分参与 |
| `branch_overactive` | branch / AdamW > 1.2 | branch 过强 |
| `geometry_spike` | J cond > 1.5x AdamW | 几何失控 |
| `phi_prime_spike` | $\phi'_{p95}$ > 1.2x AdamW | derivative 失控 |
| `trust_too_conservative` | clip rate > 0.8 且 branch under-active | trust radius 太小 |
| `trust_too_loose` | bad step rate 高或 geometry spike | trust radius 太大 |
| `descent_ratio_bad` | ratio p10 < 0 | 一阶预测失效 |
| `data_metric_overfit` | train loss 好但 test gap 变大 | data metric 过拟合 |
| `momentum_explosion` | branch / AdamW > 1.5 | momentum 失控 |
| `prox_oversmooth` | curvature 低但 train acc 大幅下降 | smoothness 太强 |
| `no_convergence_gain` | AUC 无提升 | 没有收敛优势 |
| `no_generalization_gain` | 泛化指标无提升 | 没有泛化优势 |

---

## 12. 决策规则

### 情况 A：GFU-lite 已经明显优于当前 geometry-aware

则：

1. GFU-lite 成为 Stage F/G 默认方法；
2. GFU-data / momentum / prox 作为增量 ablation；
3. 可以直接进入 CIFAR-10 small readiness。

---

### 情况 B：GFU-lite 只改善 KMNIST，但 Fashion 不如当前 best

则：

1. 保留 dataset-specific GFU；
2. Fashion 使用 current geometry-aware；
3. KMNIST 使用 GFU-lite；
4. 暂不做统一 optimizer claim。

---

### 情况 C：GFU-data 提升泛化但降低 clean accuracy

则 GFU-data 定位为泛化 / label-noise 模块，不作为 clean accuracy 默认方法。

---

### 情况 D：M-aware momentum 仍然不稳

则停止 momentum 主线，只保留 trust-region + branch controller。

---

### 情况 E：GFU 无法缩小 KMNIST gap

如果 GFU 后 KMNIST 仍卡在约 $1\%$ gap，则说明 bottleneck 可能不是 update 机制，而是：

1. 模型容量；
2. basis count；
3. rest optimizer；
4. training schedule；
5. data augmentation。

此时应进入 capacity / architecture refinement，而不是继续优化 functional update。

---

## 13. 最终建议

我建议按下面顺序执行：

1. **GFU-0 smoke**：确认稳定；
2. **GFU-1 主任务对比**：Fashion / KMNIST 5 seeds；
3. **GFU-2 trust-region ablation**：判断 step control 是否有用；
4. **GFU-3 branch controller ablation**：判断 layer-wise target 是否有用；
5. **GFU-4 data metric**：专攻 KMNIST / 泛化；
6. **GFU-5 momentum**：只有前面稳定后再做；
7. **GFU-6 prox / spectral filtering**：专攻泛化和 noise；
8. **GFU-7 generalization stress**：复核 Stage G；
9. **GFU-8 CIFAR readiness**：决定是否进入 Stage I。

最关键的判断标准是：

$$
\boxed{
\text{GFU 是否能把 KMNIST 的约 }1\%\text{ accuracy gap 缩小，同时保留 Stage F/G 的 AUC、几何和校准优势。}
}
$$

如果能做到，GFU 就值得成为 DG-KAN 进入 Vision Scaling 前的默认 optimizer。

---

## 14. 一句话总结

这个 GFU 方案总体可行，但必须增量验证。最值得优先做的是：

$$
\boxed{
\text{trust-region functional step}
+
\text{layer-wise branch target controller}
+
\text{soft metric continuation}
}
$$

Data-weighted metric、M-aware momentum 和 proximal smoothness 都有潜力，但风险更高，必须在 GFU-lite 稳定后逐步打开。成功后，DG-KAN 的 functional update 将从“手工 schedule”升级为真正的 geometry-aware functional optimizer，为 Stage I Vision Scaling 提供更稳的训练基础。
