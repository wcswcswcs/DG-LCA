# DG-KAN Functional Update v2 全面升级实验计划

> 文件名：`DG-KAN_Functional_Update_v2_全面升级实验计划.md`  
> 对象：DG-KAN / DG-LCA functional update 后续升级实验  
> 定位：在 current geometry-aware functional update 与 GFU 实验之后，设计一套面向 Stage I Vision Scaling 前的全面 functional optimizer 升级路线  
> 公式格式：Typora 友好，统一使用 `$...$` 和 `$$...$$`  
> 核心目标：把当前有效但偏手工的 geometry-aware functional update，升级成一个可解释、可审计、可扩展的 **Functional Geometry Optimizer v2**，简称 **FGO-v2**。它不再是单纯的 Sobolev 预条件，也不是 GFU-lite 全家桶，而是围绕 **function-space trust、data-aware metric、functional-direction momentum、edge-SNR、Functional-SAM、target solve、function-space averaging** 形成的分阶段优化体系。

---

## 0. 当前结论与为什么需要全面升级

当前 DG-KAN 的主线已经从 learned router 转向下面这个闭环：

$$
\text{KAN primitive-adjoint co-design}
+
\text{functional-space update}
+
\text{analytic local credit}
+
\text{low-memory sufficient statistics}
$$

我们已经有了几条比较稳的结论。

第一，`AnalyticAdj-DGKAN` 是当前最可靠的 credit / low-memory 路线。KAN primitive 的 analytic adjoint 与 autograd 对齐，且在 Stage E 中带来了明显的 peak memory reduction。Dense sufficient statistics 证明了统计量公式正确，但 dense 形式本身不是强低显存实现。

第二，Stage F/G 证明 **current geometry-aware functional update** 是当前 clean / generalization 主线。它在 Fashion-MNIST 上已经同时优于 AdamW 的 test accuracy、validation loss AUC、$\phi'_{p95}$、Jacobian condition 和 ECE；在 KMNIST 上仍有约 $1\%$ clean accuracy gap，但 AUC、几何和 calibration 明显优于 AdamW。

第三，GFU 第一版没有成为默认方法。GFU-lite 的 trust-region 和 branch controller 过于保守，主要失败类型是 `branch_underactive` 与 `trust_too_conservative`。这说明问题不是方向错误，而是 controller 过强、信号太粗、不同组件耦合方式不对。

第四，GFU 实验也筛出了有价值的组件。`data-weighted metric` 对 GFU-lite 有增益，但必须保留 grid Sobolev prior；`functional-direction momentum` 在 KMNIST 上有效，尤其 $\mu=0.8$ 能把 clean gap 从 current geometry-aware 的约 $1.02\%$ 缩到约 $0.76\%$；Fashion label-noise 下 GFU-best 有很强收益，尤其 high noise 时 clean accuracy 和 memorization 都明显改善。

因此下一步不应继续“小修小补”地调一个阈值，也不应简单把 GFU-lite 扩大 sweep。新的实验应该回答一个更大问题：

$$
\boxed{
\text{能否把 DG-KAN functional update 升级为真正的函数空间优化器，}
\text{在 clean accuracy、AUC、泛化、抗噪、几何稳定之间形成更好的 Pareto？}
}
$$

本计划将 current geometry-aware schedule 作为可靠底座，把 GFU 中有效组件重新组合，并引入 Functional-SAM、SNR-adaptive edge update、Block-level functional target solve、function-space EMA / spectral regularization 等机制，形成 **FGO-v2**。

---

## 1. 总体原则

FGO-v2 的设计不再把所有组件一次性打开，而是采用 **底座稳定、逐层升级、每个组件必须过 gate** 的策略。

当前默认底座是：

$$
\boxed{
\text{AnalyticAdj credit}
+
\text{current geometry-aware functional update}
+
\text{non-KAN AdamW}
}
$$

这个底座已经在 Fashion-MNIST 和 KMNIST 上跑通，因此任何新组件都必须至少回答三个问题。

第一，它是否改善 clean training 的速度-精度-几何 Pareto？

第二，它是否改善 small-data、label-noise、calibration 等泛化场景？

第三，它是否带来明显额外 cost、bad steps、branch over-active 或 under-active？

FGO-v2 的原则是：

1. **不放弃 current geometry-aware**。它是当前最强 clean 主线，新方法先作为叠加组件，不直接替代。
2. **不使用 hard branch target controller 作为默认**。第一版 GFU 的 layer-wise branch target controller 太强，后续只作为 soft multiplier / soft regularizer。
3. **trust-region 不做高频强裁剪**。目标 clip rate 是 $5\%$ 到 $20\%$，如果 clip rate 长期超过 $40\%$，说明 optimizer 被压死。
4. **momentum 只能作用在 functional direction 上**。naive coefficient Adam / Sobolev Adam 已经多次爆炸，因此禁止作为主线。
5. **data metric 必须混合 grid prior**。纯 data metric 或 no-grid metric 会降低全局函数约束，暂不进入主线。
6. **regularization 模块后期启用**。Functional-SAM、spectral shrink、proximal smoothing 主要用于泛化和 late phase，不能一开始就抑制 branch activation。
7. **所有结果必须同时报告 branch utilization**。没有 branch ratio / no-KAN drop 的 accuracy 结果不能解释为 KAN 有效。

---

## 2. FGO-v2 的核心数学对象

DG-KAN block 为：

$$
h_{k+1}=h_k+\alpha_k b_{k,t}\operatorname{KAN}_k(\operatorname{LN}(h_k))
$$

其中 $b_{k,t}$ 是 optimizer 控制的 branch scale。KAN edge function 为：

$$
\phi_{ji}(t)=\sum_{m=1}^{M}a_{jim}B_m(t)
$$

KAN coefficient gradient 为：

$$
g_{a,k,t}=\nabla_{a_k}L_t
$$

基础 Sobolev metric 为：

$$
M_{grid}=M_0+\alpha_s R_1+\beta_s R_2+\rho I
$$

其中：

$$
M_0=\int B(t)B(t)^\top dt
$$

$$
R_1=\int B'(t)B'(t)^\top dt
$$

$$
R_2=\int B''(t)B''(t)^\top dt
$$

Data-weighted metric 为：

$$
M_{data,t}=\mathbb E_{x\sim \mathcal D_t}[B(x)B(x)^\top]
$$

FGO-v2 的混合 metric 为：

$$
M_{t}=(1-\lambda_{data,t})M_{grid}+\lambda_{data,t}M_{data,t}+\rho_t I
$$

functional direction 为：

$$
p_t=-M_t^{-1}g_{a,t}
$$

若使用 functional-direction momentum：

$$
v_t=\mu v_{t-1}+(1-\mu)p_t
$$

最终更新为：

$$
a_{t+1}=a_t+\eta_t\operatorname{clip}_{M_t}(v_t,\tau_t)
$$

其中 M-norm clipping 为：

$$
\operatorname{clip}_{M}(v,\tau)=
v\cdot \min\left(1,\frac{\tau}{\sqrt{v^\top Mv}+\epsilon}\right)
$$

FGO-v2 还维护 branch activity：

$$
r_{branch,k,t}=
\frac{
\|\alpha_k b_{k,t}\operatorname{KAN}_k(\operatorname{LN}(h_k))\|
}{
\|h_k\|+\epsilon
}
$$

以及 effective derivative scale：

$$
s_{k,t}=|\alpha_k b_{k,t}|\cdot \phi'_{p95,k,t}
$$

核心目标不是让所有几何指标越小越好，而是找到：

$$
\boxed{
\text{branch utilization 足够高}
\quad + \quad
\phi' / J / curvature 足够可控
}
$$

---

## 3. 方法族定义

本轮实验会比较以下方法族。

### 3.1 Baseline 方法

| 方法 | 说明 |
|---|---|
| `adamw` | 全参数 AdamW，强 accuracy baseline |
| `static_func` | 静态 diag-to-Sobolev functional update |
| `geometry_func_current` | Stage F/G 当前最强 geometry-aware schedule |
| `analytic_adj_geometry` | AnalyticAdj credit + current geometry-aware update，默认主线 |

### 3.2 GFU-v1 参考方法

| 方法 | 说明 |
|---|---|
| `gfu_lite` | trust-region + branch target controller + soft continuation |
| `gfu_data` | gfu_lite + data-weighted metric |
| `gfu_momentum` | gfu_data + functional-direction momentum |
| `gfu_prox` | gfu_momentum + late proximal / spectral smoothing |

这批方法只作为参考，不作为默认主线，因为第一版 GFU 实验已经显示 GFU-lite 太保守。

### 3.3 FGO-v2 主方法

| 方法 | 说明 |
|---|---|
| `fgo_relaxed_trust` | current geometry-aware + relaxed trust + descent damping |
| `fgo_data_metric` | relaxed trust + mixed data/grid metric |
| `fgo_momentum` | fgo_data_metric + functional-direction momentum |
| `fgo_snr` | fgo_momentum + SNR-adaptive edge update |
| `fgo_fsam` | fgo_momentum + Functional-SAM |
| `fgo_target_solve` | fgo_momentum + early block-level functional target solve |
| `fgo_regularized` | fgo_momentum + function EMA / spectral shrink / prox late |
| `fgo_full` | 只在组件通过后组合，不在第一轮直接跑 |

---

## 4. 数据集、模型与基本训练配置

第一阶段仍然使用灰度视觉任务，因为它们已经形成稳定结论：

```text
datasets = Fashion-MNIST, KMNIST
train / val / test = 6000 / 1000 / 1000
seeds = 0,1,2,3,4
```

Fashion-MNIST 用于验证 clean / small-data / label-noise 的收益，因为当前 geometry-aware 已经非常强。KMNIST 用于验证 hard-task clean accuracy gap 能否继续缩小。

模型默认：

```text
model = residual DG-KAN
credit = AnalyticAdj
hidden_dim = 64
basis_count = 16
depth = 4
alpha_init = 1.5
batch_size = 256
epochs = 20
non-KAN optimizer = AdamW
rest_lr = 0.003
```

KMNIST 的候选扩展：

```text
hidden_dim = 64, 96
basis_count = 16, 24
epochs = 20, 30
```

只有当 FGO-v2 在 Fashion/KMNIST 上过 gate 后，才进入：

```text
CIFAR-10 small ConvStem-DGKAN
```

---

## 5. 总体实验路线

本计划分成八个 package。它们不是并列大扫，而是递进判断。

| Package | 目的 | 进入下一步条件 |
|---|---|---|
| V2-0 | 复现 current geometry-aware 与 GFU-v1 关键结果 | 指标对齐，确认无实现偏差 |
| V2-1 | relaxed trust / descent damping | trust 不再过保守，clip rate 合理 |
| V2-2 | data-weighted metric 重组 | 提升 KMNIST 或 label-noise，不牺牲 Fashion |
| V2-3 | functional-direction momentum | 缩小 KMNIST clean gap，避免 branch explosion |
| V2-4 | SNR-adaptive edge update | 提升 branch 有效性，降低 noisy edge 更新 |
| V2-5 | Functional-SAM | 提升泛化、label-noise、calibration |
| V2-6 | Target solve | 改善 hard task branch under-active / AUC |
| V2-7 | Regularization stack | function EMA / spectral / prox，用于 noisy-label 与 small-data |
| V2-8 | Selection / CIFAR readiness | 选择 clean default、hard-task default、noise default |

每个 package 都会与 `geometry_func_current` 比较，而不是只和 AdamW 比较。因为 current geometry-aware 是我们现在真正要超越的内部基线。

---

## 6. Package V2-0：复现实验与统一基线

### 6.1 目的

V2-0 用于固定基准，避免后续组件结果不可比。它复现：

1. `AdamW`；
2. `current geometry-aware`；
3. `GFU-lite`；
4. `GFU-data lambda=0.25`；
5. `GFU-data + functional momentum mu=0.8`。

### 6.2 设置

```text
datasets = Fashion-MNIST, KMNIST
seeds = 0,1,2,3,4
epochs = 20
model = depth4 hidden64 basis16 alpha1.5
```

### 6.3 需要记录的指标

#### 性能与收敛

```text
task/train_acc
task/val_acc
task/test_acc
task/test_loss
conv/train_loss_auc
conv/val_loss_auc
conv/val_auc_improvement_vs_adamw
conv/val_auc_improvement_vs_geometry_current
conv/steps_to_adamw_final_val_loss
conv/time_to_adamw_final_val_loss
```

#### 几何

```text
kan/phi_prime_p95
kan/phi_prime_max
kan/curvature_mean
kan/sobolev_norm_mean
jacobian/condition_mean
jacobian/condition_max
credit/amplification_p95
credit/noise_gain_p95
```

#### branch utilization

```text
branch/mean_output_norm_ratio
branch/branch_over_adamw
branch/effective_alpha_mean
ablation/no_kan_test_acc
ablation/no_kan_acc_drop
ablation/no_kan_drop_over_adamw
ablation/kan_logit_delta_norm
```

#### optimizer diagnostics

```text
functional/update_norm_coeff
functional/update_norm_M
functional/descent_pred
functional/descent_actual
functional/descent_ratio
functional/bad_step_rate
schedule/branch_scale
schedule/coeff_lr_multiplier
schedule/phase_id
```

### 6.4 可视化

W&B 建立 Page V2-0：

1. `test_acc` bar：AdamW / geometry-current / GFU variants；
2. `val_loss_auc` bar；
3. `test_acc vs val_loss_auc` Pareto scatter；
4. `phi_prime_p95` 和 `jacobian_condition` bar；
5. branch ratio / no-KAN drop paired plot；
6. failure table。

### 6.5 成功判定

V2-0 不要求新方法胜出，只要求复现当前已知结论：

Fashion：

$$
\text{geometry-current test acc} \geq \text{AdamW test acc}
$$

且：

$$
\text{geometry-current val AUC improvement} > 10\%
$$

KMNIST：

$$
\text{geometry-current acc gap vs AdamW} \approx 1\%
$$

如果复现偏差大于 $0.5\%$ accuracy 或 AUC 偏差大于 $3\%$，后续 package 不应继续。

---

## 7. Package V2-1：Relaxed trust-region 与 descent damping

### 7.1 背景

GFU-lite 的一个主要问题是 trust clipping 太高。Fashion 上 clip rate 高达 $0.6+$，导致 branch 被压住。V2-1 的目标是重新设计 trust，使它只在坏步或几何风险时介入，而不是长期裁剪 functional direction。

### 7.2 方法

以 current geometry-aware 为底座，不使用 hard branch target controller，只加入 relaxed trust。

functional direction：

$$
p_t=-M_t^{-1}g_t
$$

M-norm：

$$
\|p_t\|_{M_t}=\sqrt{p_t^\top M_t p_t}
$$

如果：

$$
\eta_t\|p_t\|_{M_t}>\tau_t
$$

则裁剪：

$$
p_t\leftarrow p_t\frac{\tau_t}{\eta_t\|p_t\|_{M_t}+\epsilon}
$$

但 trust radius 不固定，而是根据 descent ratio 更新：

$$
q_t=\frac{\Delta L_{actual,t}}{\Delta L_{pred,t}+\epsilon}
$$

如果 $q_t<0$ 或 actual loss 上升，则：

$$
\tau_{t+1}=c_{shrink}\tau_t
$$

如果 $0.5<q_t<1.5$ 且几何稳定，则：

$$
\tau_{t+1}=c_{expand}\tau_t
$$

### 7.3 实验变量

```text
trust_radius_init = 0.03, 0.05, 0.08
trust_radius_min = 0.005
trust_radius_max = 0.20, 0.30
shrink = 0.7, 0.8
expand = 1.05, 1.10
use_descent_damping = true / false
```

### 7.4 记录指标

```text
trust/clip_rate_global
trust/clip_rate_layer_k
trust/radius_mean
trust/radius_min
trust/radius_max
trust/shrink_count
trust/expand_count
trust/clip_due_to_mnorm
trust/clip_due_to_bad_descent
trust/clip_due_to_geometry

functional/descent_ratio_mean
functional/descent_ratio_p10
functional/descent_ratio_p50
functional/descent_ratio_p90
functional/bad_step_count
functional/bad_step_total_positive_loss
functional/pred_actual_corr

branch/branch_over_adamw
kan/phi_prime_p95
jacobian/condition_max
conv/val_loss_auc
task/test_acc
```

### 7.5 可视化

Page V2-1：

1. trust clip rate over epochs；
2. trust radius over epochs；
3. descent ratio histogram；
4. test acc vs clip rate scatter；
5. branch ratio vs clip rate scatter；
6. geometry spike count by method；
7. AUC improvement vs trust clip rate。

### 7.6 成功标准

Relaxed trust 通过条件：

$$
0.05 \leq \text{trust clip rate} \leq 0.25
$$

同时：

$$
\text{test acc gap vs geometry-current} < 0.3\%
$$

并且：

$$
\text{bad step rate} \leq \text{geometry-current bad step rate}
$$

若 trust clip rate $>0.4$ 且 branch/AdamW 下降，则判定为 `trust_too_conservative`。

---

## 8. Package V2-2：Data-weighted Sobolev metric 重组

### 8.1 背景

GFU-data 表明 data-weighted metric 有增益，但单独作为 GFU-lite 组件仍没有超过 geometry-aware。V2-2 不再把 data metric 放在 GFU-lite 上，而是把它叠加到 current geometry-aware + relaxed trust 上。

混合 metric 为：

$$
M_t=(1-\lambda_t)M_{grid}+\lambda_t M_{data,t}+\rho_t I
$$

其中 $M_{data,t}$ 用 batch basis statistics 的 EMA：

$$
M_{data,t}=\mu M_{data,t-1}+(1-\mu)B(x_t)^\top B(x_t)
$$

### 8.2 实验变量

```text
data_lambda_init = 0.05, 0.10, 0.25
data_lambda_final = 0.00, 0.05, 0.10
data_metric_ema = 0.90, 0.95, 0.98
update_interval = 1 epoch, 2 epochs
keep_grid_prior = true
no_grid = false except diagnostic only
```

推荐主候选：

```text
lambda_init = 0.10 or 0.25
lambda_final = 0.05
data_metric_ema = 0.95
```

### 8.3 记录指标

```text
metric/data_lambda_current
metric/layer_k/data_condition
metric/layer_k/grid_condition
metric/layer_k/mixed_condition
metric/layer_k/eig_min
metric/layer_k/eig_max
metric/layer_k/data_grid_alignment
metric/layer_k/update_cos_grid_vs_mixed
metric/layer_k/update_norm_ratio_mixed_vs_grid
trust/clip_rate_global
functional/descent_ratio
conv/val_loss_auc
task/test_acc
```

`data_grid_alignment` 定义为：

$$
\operatorname{align}(M_{data},M_{grid})
=
\frac{\langle M_{data},M_{grid}\rangle_F}{\|M_{data}\|_F\|M_{grid}\|_F+\epsilon}
$$

### 8.4 可视化

Page V2-2：

1. data lambda schedule over epochs；
2. mixed metric condition over epochs；
3. update cosine grid vs mixed；
4. test acc vs mixed condition；
5. val AUC vs data lambda；
6. data-grid alignment heatmap by layer。

### 8.5 成功标准

Data metric 通过条件：

Fashion：

$$
\text{test acc} \geq \text{geometry-current test acc} - 0.3\%
$$

KMNIST：

$$
\text{acc gap vs AdamW} < 0.8\%
$$

或者相对 geometry-current：

$$
\text{test acc gain} > 0.2\%
$$

同时：

$$
\text{mixed condition} < 500
$$

且：

$$
\text{trust clip rate}<0.3
$$

---

## 9. Package V2-3：Functional-direction momentum

### 9.1 背景

GFU 实验已经明确：naive coefficient Adam / Sobolev Adam 会爆炸，functional-direction momentum 是更合理方向。V2-3 将 momentum 集成到 relaxed trust + data metric 之后。

先算：

$$
p_t=-M_t^{-1}g_t
$$

再做：

$$
v_t=\mu v_{t-1}+(1-\mu)p_t
$$

然后做 M-norm trust clipping。

### 9.2 实验变量

```text
momentum_mu = 0.5, 0.7, 0.8, 0.9
warmup_steps_before_momentum = 0.10, 0.25 epoch fraction
momentum_reset_on_switch = true / false
momentum_reset_on_geometry_spike = true
```

重点候选：

```text
mu = 0.8
start_after = 0.10 training fraction
reset_on_geometry_spike = true
```

### 9.3 记录指标

```text
momentum/mu
momentum/layer_k/direction_cos_current
momentum/layer_k/direction_cos_prev
momentum/layer_k/buffer_m_norm
momentum/layer_k/buffer_to_step_norm_ratio
momentum/reset_count
momentum/spike_reset_count
functional/update_m_norm
trust/clip_rate
branch/branch_over_adamw
kan/phi_prime_p95
jacobian/condition_max
```

Direction cosine：

$$
\cos(p_t,v_{t-1})=
\frac{\langle p_t,v_{t-1}\rangle}{\|p_t\|\|v_{t-1}\|+\epsilon}
$$

### 9.4 可视化

Page V2-3：

1. test acc vs momentum $\mu$；
2. AUC vs momentum $\mu$；
3. branch ratio vs momentum $\mu$；
4. direction cosine over time；
5. momentum buffer norm over time；
6. geometry spike count by $\mu$；
7. comparison with naive Adam negative control。

### 9.5 成功标准

Momentum 通过条件：

KMNIST：

$$
\text{acc gap vs AdamW}<0.75\%
$$

且：

$$
\text{val AUC improvement}>10\%
$$

Fashion：

$$
\text{test acc} \geq \text{geometry-current}-0.3\%
$$

同时：

$$
\text{branch/AdamW}\in[0.6,0.9]
$$

$$
\text{bad step rate}<1\%
$$

若 $\phi'_{p95}$ 或 Jacobian condition 超过 AdamW，则判定为 `momentum_overactive`。

---

## 10. Package V2-4：SNR-adaptive edge update

### 10.1 背景

不同 KAN edge 的 credit 可靠性不同。统一更新所有 edge 会导致：

1. 高 SNR edge 更新不够；
2. 低 SNR edge 拟合 batch noise；
3. branch ratio 上升但 no-KAN drop 不上升。

因此引入 edge-level SNR。

对 edge $(j,i)$ 的 coefficient gradient：

$$
s_{ji,t}\in\mathbb R^M
$$

维护 EMA：

$$
\mu_{ji,t}=\lambda\mu_{ji,t-1}+(1-\lambda)s_{ji,t}
$$

$$
q_{ji,t}=\lambda q_{ji,t-1}+(1-\lambda)s_{ji,t}^2
$$

$$
v_{ji,t}=q_{ji,t}-\mu_{ji,t}^2
$$

定义：

$$
\operatorname{SNR}_{ji}=\frac{\|\mu_{ji,t}\|_2}{\sqrt{\operatorname{mean}(v_{ji,t})}+\epsilon}
$$

学习率 multiplier：

$$
m^{lr}_{ji}=\operatorname{clip}\left(\frac{\operatorname{SNR}_{ji}}{\tau_{snr}},m_{min},m_{max}\right)
$$

### 10.2 实验变量

```text
snr_ema = 0.90, 0.95, 0.98
snr_tau = median_per_layer, p60_per_layer, fixed_1.0
lr_mult_min = 0.5
lr_mult_max = 1.5, 2.0
rho_mult_enable = false first round
beta_mult_enable = false first round
```

第一轮只调 lr multiplier，不调 $\rho$ 和 $\beta$。第二轮再加入 damping adaptation。

### 10.3 记录指标

```text
snr/global/mean
snr/global/p10
snr/global/p50
snr/global/p90
snr/global/low_snr_fraction
snr/global/high_snr_fraction
snr/layer_k/mean
snr/layer_k/p10
snr/layer_k/p90
snr/layer_k/low_fraction
update/snr_lr_mult_mean
update/snr_lr_mult_p10
update/snr_lr_mult_p90
update/snr_lr_mult_max
branch/snr_weighted_branch_ratio
branch/no_kan_drop
```

SNR-weighted branch ratio：

$$
r^{snr}_{branch,k}=
\frac{\sum_{ji}\operatorname{SNR}_{ji}\|\Delta h_{ji}\|}{\sum_{ji}\operatorname{SNR}_{ji}+\epsilon}
$$

### 10.4 可视化

Page V2-4：

1. edge SNR histogram by layer；
2. low-SNR fraction over training；
3. SNR multiplier distribution；
4. low-SNR fraction vs train-test gap；
5. no-KAN drop vs SNR-weighted branch ratio；
6. test acc vs high-SNR edge fraction；
7. noisy-label memorization vs low-SNR fraction。

### 10.5 成功标准

SNR-adaptive update 通过条件：

Clean KMNIST：

$$
\text{acc gap vs AdamW}<0.7\%
$$

Noisy-label Fashion / KMNIST：

$$
\text{clean test acc} > \text{geometry-current clean test acc}
$$

且：

$$
\text{noise memorization} < \text{geometry-current memorization}
$$

如果 SNR-adaptive 提高 branch ratio 但 no-KAN drop 不升，则判定为 `branch_nonuseful_growth`。

---

## 11. Package V2-5：Functional-SAM

### 11.1 背景

Stage G 显示 geometry-aware 对 calibration 和 label-noise 有明显优势，但 corruption robustness 不是强项。Functional-SAM 不主要服务 clean train loss，而服务 flatness 和泛化。

普通 SAM：

$$
\min_\theta \max_{\|\epsilon\|\leq\rho} L(\theta+\epsilon)
$$

Functional-SAM 使用 function-space norm：

$$
\min_a \max_{\delta a^\top M\delta a\leq \rho^2}L(a+\delta a)
$$

一阶最坏扰动为：

$$
\delta a_{adv}=\rho\frac{M^{-1}g}{\sqrt{g^\top M^{-1}g}+\epsilon}
$$

### 11.2 实验变量

```text
rho_func = 0.01, 0.03, 0.05
sam_interval = 4, 8
sam_start_frac = 0.25, 0.50
metric = grid_sobolev, mixed_data_grid
apply_to = KAN coefficients only
combine_with = fgo_momentum best
```

### 11.3 记录指标

```text
fsam/rho
fsam/interval
fsam/start_frac
fsam/adv_norm_M
fsam/base_loss
fsam/adv_loss
fsam/sharpness_gap
fsam/grad_cos_base_adv
fsam/update_norm_M
fsam/extra_step_time_ms

calibration/ece
label_noise/clean_test_acc
label_noise/noise_memorization
small_data/test_acc
corruption/mean_acc
```

Sharpness gap：

$$
\Delta L_{sharp}=L(a+\delta a_{adv})-L(a)
$$

### 11.4 可视化

Page V2-5：

1. sharpness gap over epochs；
2. sharpness gap vs ECE；
3. sharpness gap vs label-noise clean acc；
4. SAM interval cost bar；
5. clean acc vs label-noise acc Pareto；
6. FSAM vs geometry-current by small-data curves。

### 11.5 成功标准

Functional-SAM 不要求 clean accuracy 全面提升。它通过的条件是：

Label-noise Fashion：

$$
\text{clean acc gain vs geometry-current}>1\%
$$

或者：

$$
\text{memorization reduction}>10\%
$$

KMNIST：

$$
\text{clean acc gap vs geometry-current}<0.5\%
$$

且 ECE improvement 增强。

Cost gate：

$$
\text{step time overhead}<1.4\times
$$

否则只作为 robustness ablation。

---

## 12. Package V2-6：Block-level functional target solve

### 12.1 背景

KMNIST 的剩余 gap 可能来自 branch 参与度不足，而不是几何阈值。Target solve 直接让 KAN branch output 朝局部目标移动。

对 block：

$$
h_{out}=h+\alpha b(z),\quad z=\operatorname{LN}(h)
$$

希望：

$$
\Delta h_{out}^{(n)}\approx-\eta g_{out}^{(n)}
$$

所以 branch target：

$$
\Delta b^{(n)}\approx-\frac{\eta}{\alpha}g_{out}^{(n)}
$$

令设计矩阵 $X$ 由 $B_m(z_i^{(n)})$ 组成，则：

$$
\Delta a_j^*=(X^\top X+\lambda M)^{-1}X^\top y_j
$$

其中：

$$
y_j=-\frac{\eta}{\alpha}g_j
$$

### 12.2 实验变量

```text
apply_layers = last_1, last_2, all_even_layers
target_solve_frac = 0.10, 0.25, 0.50
ridge_lambda = 1e-3, 1e-2, 1e-1
gamma_start = 0.25, 0.50
gamma_final = 0.0
apply_start_frac = 0.0, 0.10
apply_end_frac = 0.25, 0.50
```

混合更新：

$$
\Delta a=(1-\gamma_t)\Delta a_{func}+\gamma_t\Delta a_{target}
$$

### 12.3 记录指标

```text
target_solve/layer_k/residual_mse
target_solve/layer_k/ridge_condition
target_solve/layer_k/delta_norm_M
target_solve/layer_k/target_delta_norm
target_solve/layer_k/actual_branch_delta_norm
target_solve/layer_k/target_actual_cos
target_solve/global/actual_descent
branch/branch_over_adamw
ablation/no_kan_drop
conv/val_loss_auc
task/test_acc
kan/phi_prime_p95
jacobian/condition_max
```

### 12.4 可视化

Page V2-6：

1. target residual MSE by layer；
2. target solve condition number；
3. branch ratio over epochs；
4. no-KAN drop over epochs；
5. target actual branch delta scatter；
6. test acc vs target_solve_frac；
7. geometry vs target_solve_frac。

### 12.5 成功标准

Target solve 主要针对 KMNIST：

$$
\text{acc gap vs AdamW}<0.5\%
$$

且：

$$
\text{branch/AdamW}\in[0.75,1.0]
$$

同时：

$$
\text{J condition reduction}>10\%
$$

如果 target solve 提高 branch 但 $\phi'$ 或 J condition 超过 AdamW，则判定为 `target_solve_overactive`。

---

## 13. Package V2-7：Function-space EMA / spectral regularization / prox

### 13.1 背景

GFU6 显示 prox / spectral 对 clean accuracy 不是默认方法，但 Fashion label-noise 下可能有价值。V2-7 将它定位为泛化增强，不追 clean 主指标。

### 13.2 方法

#### Function EMA

$$
a^{ema}_{t+1}=\tau a^{ema}_t+(1-\tau)a_t
$$

因为 basis 固定，这等价于函数 EMA：

$$
\phi^{ema}(t)=\tau\phi^{ema}(t)+(1-\tau)\phi(t)
$$

#### Spectral shrink

若：

$$
R=Q\Lambda Q^\top
$$

$$
a=Qz
$$

则 late shrink 为：

$$
z_i\leftarrow\frac{1}{1+\lambda\Lambda_i}z_i
$$

#### Proximal smoothness

$$
a_{t+1}=(I+2\eta\lambda R)^{-1}\tilde a_{t+1}
$$

### 13.3 实验变量

```text
ema_tau = 0.99, 0.995, 0.999
ema_start_frac = 0.25, 0.50
spectral_lambda = 1e-4, 1e-3, 1e-2
spectral_start_frac = 0.50, 0.70
prox_lambda = 1e-4, 1e-3, 1e-2
prox_start_frac = 0.50, 0.70
```

### 13.4 记录指标

```text
ema/test_acc
ema/val_loss
ema/ece
ema/phi_prime_p95
ema/jacobian_condition
ema/raw_minus_ema_acc
ema/raw_minus_ema_jcond
spectral/high_freq_energy
spectral/low_freq_energy
spectral/shrink_ratio
prox/shrink_norm
prox/curvature_reduction
```

### 13.5 可视化

Page V2-7：

1. raw vs EMA test acc；
2. raw vs EMA ECE；
3. high-frequency energy over training；
4. label-noise clean acc by regularizer；
5. memorization vs spectral shrink；
6. curvature vs train-test gap。

### 13.6 成功标准

用于 label-noise：

$$
\text{clean acc gain vs geometry-current}>1\%
$$

或：

$$
\text{memorization reduction}>10\%
$$

Clean setting 不允许：

$$
\text{test acc drop}>0.5\%
$$

否则该 regularizer 只用于 noisy-label / small-data。

---

## 14. Package V2-8：方法选择与 CIFAR readiness

### 14.1 目标

V2-8 不训练新模型，而是聚合 V2-1 到 V2-7 的结果，选出三个 default：

| Default | 用途 |
|---|---|
| `FGO-clean-default` | clean Fashion / KMNIST / CIFAR 默认 |
| `FGO-hard-default` | KMNIST / harder task 默认 |
| `FGO-noise-default` | label-noise / small-data 默认 |

### 14.2 选择规则

Clean default 需要满足：

Fashion：

$$
\text{test acc}\geq\text{geometry-current}-0.3\%
$$

KMNIST：

$$
\text{gap vs AdamW}<0.75\%
$$

并且：

$$
\text{val AUC improvement vs AdamW}>10\%
$$

$$
\phi'\text{ reduction}>20\%
$$

$$
J\text{ reduction}>10\%
$$

Noise default 需要满足：

$$
\text{label-noise clean acc}>
\text{geometry-current clean acc}
$$

并且：

$$
\text{ECE}<\text{geometry-current ECE}
$$

CIFAR readiness 需要：

$$
\text{Fashion clean pass}
$$

$$
\text{KMNIST gap}<1\%
$$

$$
\text{no geometry spike}
$$

$$
\text{step time overhead}<1.3\times \text{geometry-current}
$$

### 14.3 可视化

Page V2-8：

1. method scorecard；
2. clean accuracy / AUC / geometry radar；
3. noisy-label scorecard；
4. cost vs gain Pareto；
5. selected default table；
6. CIFAR readiness gate。

---

## 15. 全局 W&B Dashboard 设计

### Page 1：FGO-v2 Overview

展示所有方法的 scorecard：

```text
test_acc
val_AUC
AUC_improvement_vs_AdamW
acc_gap_vs_AdamW
phi_prime_reduction
J_reduction
ECE_reduction
branch_over_AdamW
no_KAN_drop
step_time_ratio
```

### Page 2：Clean Training Pareto

图：

1. test acc vs val AUC；
2. test acc vs J condition；
3. test acc vs $\phi'_{p95}$；
4. AUC improvement vs branch ratio；
5. no-KAN drop vs branch ratio。

### Page 3：Functional Optimizer Internals

图：

1. update M-norm over epochs；
2. descent ratio histogram；
3. bad step rate；
4. trust radius over time；
5. trust clip rate over time；
6. effective coefficient lr over time。

### Page 4：Metric Dynamics

图：

1. data lambda over time；
2. metric condition over time；
3. update cosine grid vs mixed；
4. metric eigenvalue spectrum；
5. mixed condition vs test acc scatter。

### Page 5：Momentum Dynamics

图：

1. direction cosine current vs buffer；
2. buffer M-norm；
3. momentum reset count；
4. momentum $\mu$ vs test acc；
5. momentum $\mu$ vs geometry spike。

### Page 6：Edge SNR

图：

1. SNR histogram by layer；
2. low-SNR fraction over epochs；
3. SNR lr multiplier distribution；
4. no-KAN drop vs high-SNR fraction；
5. noisy-label memorization vs low-SNR fraction。

### Page 7：Functional-SAM / Sharpness

图：

1. sharpness gap over epochs；
2. sharpness gap vs ECE；
3. sharpness gap vs label-noise clean acc；
4. FSAM overhead bar；
5. rho sweep Pareto。

### Page 8：Target Solve

图：

1. target residual MSE；
2. ridge condition；
3. branch activation curve；
4. target actual branch delta scatter；
5. target_solve_frac vs accuracy；
6. target_solve_frac vs geometry。

### Page 9：Generalization and Noise

图：

1. small-data accuracy curves；
2. label-noise clean accuracy；
3. memorization rate；
4. ECE by noise level；
5. curvature vs train-test gap；
6. raw vs EMA comparison。

### Page 10：Failure Analysis

结构化 failure table：

| failure type | 说明 |
|---|---|
| `branch_underactive` | branch/AdamW 太低，no-KAN drop 小 |
| `branch_overactive` | branch/AdamW 太高，J / phi spike |
| `trust_too_conservative` | clip rate 高且 branch 低 |
| `trust_too_loose` | bad step 或 geometry spike |
| `data_metric_bad_condition` | mixed condition 过高 |
| `momentum_overactive` | momentum 导致 branch / geometry 爆 |
| `snr_suppresses_branch` | SNR update 导致 branch 不参与 |
| `fsam_too_slow` | FSAM overhead 过高 |
| `target_solve_overactive` | target solve 提升 branch 但几何崩 |
| `prox_too_strong` | smoothness 太强导致欠拟合 |
| `no_clean_gain` | clean accuracy 没有收益 |
| `no_generalization_gain` | 泛化压力没收益 |
| `cifar_not_ready` | CIFAR readiness gate 未通过 |

---

## 16. 成功标准总表

### 16.1 Weak success

满足：

$$
\text{FGO-v2 clean acc gap vs geometry-current}<0.5\%
$$

且在至少一个设置中：

$$
\text{val AUC improvement vs AdamW}>10\%
$$

$$
\phi'\text{ reduction}>20\%
$$

$$
J\text{ reduction}>10\%
$$

### 16.2 Medium success

Fashion：

$$
\text{test acc}\geq\text{geometry-current test acc}
$$

KMNIST：

$$
\text{acc gap vs AdamW}<0.75\%
$$

同时：

$$
\text{val AUC improvement}>10\%
$$

$$
\text{ECE reduction}>10\%
$$

### 16.3 Strong success

满足：

$$
\text{Fashion test acc} \geq \text{AdamW}
$$

$$
\text{KMNIST acc gap vs AdamW}<0.5\%
$$

$$
\text{label-noise clean acc gain vs geometry-current}>1\%
$$

$$
\text{step time overhead}<1.3\times
$$

并且：

$$
\text{no geometry spike}
$$

这时 FGO-v2 可以作为 Stage I / CIFAR 默认方法。

---

## 17. 预期结论模板

如果 V2 实验成功，我们可以写：

> We upgrade DG-KAN functional updates from a fixed Sobolev preconditioner to an online functional geometry optimizer. By combining relaxed M-norm trust control, data/grid mixed functional metrics, functional-direction momentum, and edge-level reliability control, FGO-v2 improves the clean accuracy and convergence-geometry Pareto over the previous geometry-aware schedule. The gains are most pronounced on hard tasks and noisy-label regimes, while preserving lower $\phi'$, lower Jacobian condition, and better calibration than AdamW.

中文：

> 我们将 DG-KAN 的 functional update 从固定 Sobolev 预条件器升级为在线函数空间几何优化器。通过 relaxed M-norm trust、data/grid 混合函数 metric、functional-direction momentum 和 edge-level reliability control，FGO-v2 相比现有 geometry-aware schedule 进一步改善了精度、收敛和几何稳定性的 Pareto，尤其在 harder task 和 noisy-label 设置中收益明显，同时保持低 $\phi'$、低 Jacobian condition 和更好的校准。

如果 V2 没有全面超过 current geometry-aware，也不是完全失败。应写：

> The first-generation GFU components reveal that overly strong trust control and hard branch targets suppress KAN branch utilization. Data-weighted metrics and functional-direction momentum are beneficial, but the current geometry-aware schedule remains the strongest clean baseline. This suggests that functional update gains depend less on global optimizer complexity and more on preserving a delicate branch-geometry balance.

---

## 18. 最终建议

本计划的核心判断是：

$$
\boxed{
\text{不要用 GFU-lite 全家桶替代 current geometry-aware，}
\text{而要把有效组件重组为 FGO-v2。}
}
$$

第一轮最推荐跑的组合是：

```text
V2-0 reproduce baselines
V2-1 relaxed trust
V2-2 data metric lambda=0.10/0.25
V2-3 functional-direction momentum mu=0.8
```

如果这三个组件组合后在 KMNIST 上把 gap 压到 $0.75\%$ 以内，再启动：

```text
V2-4 SNR-adaptive edge update
V2-5 Functional-SAM
V2-7 EMA / spectral regularization for noisy-label
```

如果仍然停留在 $1\%$ gap 附近，则不要继续复杂化 optimizer，而应考虑：

```text
hidden_dim / basis_count 增大
ConvStem-DGKAN 前的小规模视觉验证
Low-rank / grouped KAN 结构改造
```

最终目标不是追求一个很复杂的 optimizer 名字，而是把 DG-KAN 的 functional update 变成真正可控的函数空间训练机制：

$$
\boxed{
\text{用 AnalyticAdj 保证 credit 正确，}
\text{用 functional metric 保证更新几何，}
\text{用 branch / SNR / trust / SAM 保证泛化和稳定。}
}
$$
