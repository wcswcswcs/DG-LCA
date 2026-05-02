# DG-KAN FGO-v3 后续实验计划：从“大一统 GFU”转向证据驱动的模块化 Functional Optimizer

> 文件名：`DG-KAN_FGO-v3_后续实验计划.md`  
> 目标：解释 FGO-v2 没达到预期的原因，并设计下一轮更有针对性的 functional update 升级实验。  
> 公式格式：Typora 友好，统一使用 `$...$` 与 `$$...$$`。  
> 核心判断：FGO-v2 不是方向失败，而是“全家桶式组合”过早。下一步不应继续堆组件，而应把已经有证据的组件拆开、修正、重组，形成 clean / hard-task / noisy-label 三条 profile。

---

## 0. 当前 FGO-v2 结果说明了什么

FGO-v2 的目标很大：

$$
\text{FGO-v2}
=
\text{geometry-aware}
+
\text{relaxed trust}
+
\text{mixed data/grid metric}
+
\text{functional-direction momentum}
+
\text{edge-SNR}
+
\text{Functional-SAM}
+
\text{target solve}
+
\text{EMA / prox / spectral}
$$

这条路线的动机是合理的：我们希望把 current geometry-aware functional update 从手工 schedule 升级成更完整的 function-space optimizer。  
但实验结果显示，FGO-v2 没有形成新的 clean default。Fashion-MNIST clean 仍然是 current geometry-aware 最强；KMNIST clean 最接近的是此前 GFU-v1 的 functional-direction momentum，gap 约 $0.76\%$，但 AUC、Jacobian 和 $\phi'$ 的综合 gate 没完全通过。FGO-v2 的 `cifar_ready=false` 是合理的。

这轮结果不是“functional optimizer 方向失败”，而是说明：

$$
\boxed{
\text{全量组合太早，组件之间相互干扰。}
}
$$

尤其是：

1. **GFU-lite / FGO-lite 太保守**。Trust clip 和 branch target controller 叠加后，容易压制 KAN branch，使 branch under-active。
2. **relaxed trust 修复了 clip rate，但没有自动带来 clean accuracy**。这说明 trust-region 是安全阀，不是主要收益来源。
3. **data metric 有 accuracy 信号，但 no-grid 会破坏几何**。KMNIST `no-grid` accuracy 高，但 $\phi'$ 和 Jacobian 明显变差，不能作为默认。
4. **functional-direction momentum 是 hard-task 上最有价值的 clean 组件**。但它必须作用在 functional direction 上，不能退回 naive coefficient Adam。
5. **SNR-adaptive update 是 label-noise 上最清楚的收益来源**，尤其 Fashion noise=0.2/0.4；但它有明显时间开销，也带来 geometry spike 风险。
6. **FSAM / prox / spectral 不适合 clean default**，但在 label-noise 或泛化 stress 中有价值。
7. **corruption robustness 没被改善**，所以 FGO-v2 不能被描述为全面鲁棒 optimizer。

因此，下一轮应该把问题重新拆成三个 profile：

$$
\boxed{
\text{FGO-v3-clean}
}
$$

用于 clean / hard-task accuracy；

$$
\boxed{
\text{FGO-v3-noise}
}
$$

用于 label-noise 与 memorization；

$$
\boxed{
\text{FGO-v3-cost}
}
$$

用于控制 SNR / FSAM 这类组件的时间开销。

这比继续跑一个更大的 “FGO-v2++” 更合理。

---

## 1. 失败原因分析

### 1.1 GFU-lite / FGO-lite 的主要问题是过度保守

GFU-lite 原本包含：

```text
soft metric continuation
trust-region functional step
layer-wise branch target controller
```

但实验中它的主要失败类型是：

```text
branch_underactive
trust_too_conservative
```

这说明它没有把 KAN branch 激活起来。  
对 residual DG-KAN 来说，branch 太弱会导致：

$$
h_{k+1}
\approx
h_k
$$

即使 $\phi'$ 和 Jacobian condition 很漂亮，模型实际表达能力也不够。

我们真正需要控制的是：

$$
r_{\text{branch},k}
=
\frac{
\|\alpha_k \operatorname{KAN}_k(\operatorname{LN}(h_k))\|
}{
\|h_k\|+\epsilon
}
$$

而不是仅仅压低：

$$
\phi'_{p95},\quad \kappa(J)
$$

GFU-lite 的问题是：它更像是在保护几何，而不是在服务 loss 下降和 branch utilization。

---

### 1.2 Hard branch target controller 不如简单 geometry-aware schedule

layer-wise branch target controller 的想法是合理的：不同层需要不同 branch ratio。  
但是当前实现更像强目标追踪器，它试图把每层 branch ratio 拉向某个预设曲线。

这会带来两个问题。

第一，branch ratio 本身不是最终目标。真正目标是：

$$
\text{loss descent}
+
\text{branch contribution}
+
\text{geometry stability}
$$

如果只追 branch ratio，可能压制有效更新。

第二，和 trust-region 叠加后，controller 会进一步降低 effective step。  
这就是为什么很多 FGO-lite / branch-controller 配置出现 branch under-active。

因此下一轮不再使用 hard branch target controller。  
新的策略是：

> 保留 current geometry-aware schedule 的全局 branch activation 逻辑，只把 branch controller 降级为 soft multiplier regularizer。

也就是说，它只能小幅调节：

$$
\eta_{\text{coeff},k}
$$

或：

$$
b_{k,t}
$$

不能强行覆盖主 schedule。

---

### 1.3 Data metric 有信号，但必须保留 grid Sobolev prior

data-weighted metric 的基本形式是：

$$
M_{\text{mixed},t}
=
(1-\lambda_t)M_{\text{grid}}
+
\lambda_t M_{\text{data},t}
$$

其中：

$$
M_{\text{data},t}
=
\mathbb E_{x\sim\mathcal D_t}
\left[
B(x)B(x)^\top
\right]
$$

FGO-v2 显示 data metric 有 accuracy 信号，尤其在 KMNIST clean 中。但 `no-grid` 版本虽然可能提高 accuracy，却明显损害 $\phi'$ 和 Jacobian condition。

这说明：

$$
\boxed{
\text{data metric 可以让 update 更贴近当前激活分布，但 grid prior 负责全域函数稳定性。}
}
$$

因此下一轮必须禁止 no-grid 作为默认。  
data metric 必须以 mixed 形式进入：

$$
M_t
=
M_{\text{grid}}
+
\lambda_t \widetilde M_{\text{data},t}
+
\alpha R_1
+
\beta R_2
+
\rho I
$$

其中 $\widetilde M_{\text{data},t}$ 需要 trace-normalization 和 condition clipping，防止 data metric 主导整个函数空间几何。

---

### 1.4 Functional-direction momentum 是正确方向，但要做成 clean hard-task profile

实验反复说明 naive Adam moment 会爆炸。  
原因是 Adam 在 coefficient Euclidean space 中累积动量，而 coefficient space 不是 KAN edge function 的自然几何。

正确方式是先计算 functional direction：

$$
p_t
=
-M_t^{-1}g_t
$$

再对 functional direction 做 momentum：

$$
v_t
=
\mu v_{t-1}
+
(1-\mu)p_t
$$

最后进行 M-norm clipping：

$$
v_t
\leftarrow
v_t
\cdot
\min
\left(
1,
\frac{\tau_t}{\|v_t\|_{M_t}+\epsilon}
\right)
$$

这类 momentum 在 KMNIST 上已经把 gap 从 geometry-aware 的约 $1.02\%$ 降到约 $0.76\%$。  
但 FGO-v2 里完整组合后并没有进一步放大收益，说明 momentum 要以 clean profile 的方式独立验证，而不是埋在一堆组件里面。

---

### 1.5 SNR-adaptive update 是 noise profile，不是 clean default

SNR-adaptive edge update 的核心思想是按 edge-level credit 信噪比调节更新强度。它适合 noisy-label，因为 noisy credit 的 edge 更新应该更保守。

定义 edge gradient 统计：

$$
s_{ji,t}
=
\nabla_{a_{ji}} L_t
$$

EMA mean 与 variance：

$$
\mu_{ji,t}
=
\beta\mu_{ji,t-1}
+
(1-\beta)s_{ji,t}
$$

$$
q_{ji,t}
=
\beta q_{ji,t-1}
+
(1-\beta)s_{ji,t}^2
$$

$$
v_{ji,t}
=
q_{ji,t}
-
\mu_{ji,t}^2
$$

edge SNR：

$$
\operatorname{SNR}_{ji}
=
\frac{\|\mu_{ji,t}\|_2}
{\sqrt{\operatorname{mean}(v_{ji,t})}+\epsilon}
$$

更新倍率：

$$
m_{ji}^{snr}
=
\operatorname{clip}
\left(
\left(
\frac{\operatorname{SNR}_{ji}}{\tau_{snr}+\epsilon}
\right)^\gamma,
m_{\min},
m_{\max}
\right)
$$

FGO-v2 说明 SNR 是 label-noise 最清楚的有效组件，尤其 Fashion noise=0.4。  
但它 step time 较高，约 $1.4\times$ 到 $1.7\times$，并且 failure table 有 geometry spike。

所以 SNR 不进入 clean default。它进入：

$$
\boxed{
\text{FGO-v3-noise}
}
$$

并且必须做 cost-aware 优化。

---

### 1.6 FSAM / spectral / prox 是 stress 组件，不是 clean 主优化器

Functional-SAM、prox 和 spectral regularization 都偏向泛化控制。  
它们可以改善某些 label-noise 设置，但在 clean task 上没有超过 current geometry-aware。

因此下一轮不能把它们放进 clean default。  
它们只进入 noise / stress profile。

---

## 2. FGO-v3 的总体策略

FGO-v3 不再是一个全组件 optimizer，而是三个 profile。

### 2.1 FGO-v3-clean

目标是解决 KMNIST clean 剩余 gap，同时不损害 Fashion clean。

它基于 current geometry-aware，而不是替代 current geometry-aware：

$$
\text{FGO-v3-clean}
=
\text{current geometry-aware}
+
\text{safe mixed data metric}
+
\text{functional-direction momentum}
+
\text{relaxed safety trust}
$$

它不使用：

```text
hard branch target controller
SNR edge update
FSAM
prox
spectral
no-grid metric
naive Adam
```

### 2.2 FGO-v3-noise

目标是 label-noise clean accuracy、memorization reduction 和 calibration。

它基于 geometry-aware 或 FGO-v3-clean，再加入：

$$
\text{SNR-adaptive edge update}
+
\text{late spectral / prox}
+
\text{optional FSAM}
$$

它不追求 clean task 默认最优。

### 2.3 FGO-v3-cost

目标是降低 SNR / FSAM 的额外开销。

它测试：

```text
SNR update interval
grouped SNR
low-frequency SNR updates
FSAM interval
FSAM start fraction
```

如果某个组件收益明显但太慢，则必须通过 FGO-v3-cost 才能进入 Stage I。

---

## 3. FGO-v3-clean 设计

### 3.1 基础训练公式

当前 geometry-aware 的 KAN coefficient update 是：

$$
a_{t+1}
=
a_t
+
\eta_t p_t
$$

其中：

$$
p_t
=
-P_t g_t
$$

$P_t$ 在 early phase 更接近 diagonal preconditioner，在 late phase 更接近 Sobolev inverse。

FGO-v3-clean 保留这个结构，但将 metric 换成安全混合 metric：

$$
M_t
=
M_{\text{grid}}
+
\lambda_t \widetilde M_{\text{data},t}
+
\alpha R_1
+
\beta R_2
+
\rho I
$$

其中 data metric 使用 trace normalization：

$$
\widetilde M_{\text{data},t}
=
\frac{
\operatorname{tr}(M_{\text{grid}})
}{
\operatorname{tr}(M_{\text{data},t})+\epsilon
}
M_{\text{data},t}
$$

并做 eigenvalue clipping：

$$
\lambda_i(\widetilde M_{\text{data}})
\leftarrow
\operatorname{clip}
\left(
\lambda_i,
\lambda_{\min}^{clip},
\lambda_{\max}^{clip}
\right)
$$

最后：

$$
p_t=-M_t^{-1}g_t
$$

或 metric continuation：

$$
p_t
=
-
\left[
(1-\omega_t)\operatorname{diag}(M_t)^{-1}
+
\omega_t M_t^{-1}
\right]
g_t
$$

---

### 3.2 Functional-direction momentum

Momentum 只作用于 functional direction：

$$
v_t
=
\mu v_{t-1}
+
(1-\mu)p_t
$$

然后更新：

$$
a_{t+1}
=
a_t
+
\eta_t v_t
$$

为避免 momentum 早期干扰 branch activation，设置 start fraction：

$$
t_{\text{mom,start}}
=
\gamma_{\text{mom}}T
$$

在此之前：

$$
v_t=p_t
$$

之后才启用 momentum。

---

### 3.3 Relaxed safety trust

FGO-v3-clean 不用强 trust clipping。  
Trust 只作为安全阀。

定义：

$$
\|v_t\|_{M_t}
=
\sqrt{
v_t^\top M_t v_t
}
$$

若：

$$
\eta_t \|v_t\|_{M_t}>\tau_t
$$

则：

$$
v_t
\leftarrow
v_t
\frac{\tau_t}{\eta_t\|v_t\|_{M_t}+\epsilon}
$$

但 $\tau_t$ 的控制目标不是“尽量多裁剪”，而是让 clip rate 维持在：

$$
0.03 \leq r_{\text{clip}} \leq 0.15
$$

如果：

$$
r_{\text{clip}}>0.2
$$

且没有 bad step / geometry spike，则说明 trust 太保守，应增大 $\tau_t$：

$$
\tau_{t+1}
=
c_{\text{expand}}\tau_t
$$

如果出现：

$$
\Delta L_{\text{actual}}>0
$$

或：

$$
\kappa(J)>\tau_J
$$

或：

$$
\phi'_{p95}>\tau_\phi
$$

则 shrink：

$$
\tau_{t+1}
=
c_{\text{shrink}}\tau_t
$$

---

### 3.4 Clean profile 默认配置

Fashion-MNIST 保持 current geometry-aware 默认，不强行替换：

```text
branch_boost = 2.0
coeff_lr_boost = 1.5
branch_max_active_frac = 0.50
branch_final_scale = 0.8
phi_high = 0.052
jac_high = 4.5
branch_high = 0.30
```

KMNIST 的 clean profile 从当前最佳 geometry-aware 出发：

```text
branch_boost = 1.5
coeff_lr_boost = 1.2
branch_max_active_frac = 0.35
branch_final_scale = 0.9
phi_high = 0.055
jac_high = 4.5
branch_high = 0.30
```

在此基础上加：

```text
data_lambda_start in {0.05, 0.10, 0.25}
data_lambda_final in {0.00, 0.05, 0.10}
momentum_mu in {0.6, 0.8, 0.9}
momentum_start_frac in {0.10, 0.25, 0.50}
trust_clip_target in {0.05, 0.10, 0.15}
```

---

## 4. FGO-v3-noise 设计

### 4.1 目标

FGO-v3-noise 只服务 noisy-label setting。  
它的目标不是 clean default，而是：

1. noisy-label 下 clean test accuracy；
2. memorization reduction；
3. calibration；
4. $\phi'$ 和 curvature 控制。

---

### 4.2 SNR update

在 geometry-aware 或 FGO-v3-clean 的基础上加入 SNR multiplier：

$$
\Delta a_{ji}
=
-\eta_t
m_{ji}^{snr}
M_t^{-1}g_{ji}
$$

其中：

$$
m_{ji}^{snr}
=
\operatorname{clip}
\left(
\left(
\frac{\operatorname{SNR}_{ji}}{\tau_{snr}+\epsilon}
\right)^\gamma,
m_{\min},
m_{\max}
\right)
$$

如果是低 SNR edge，则：

$$
m_{ji}^{snr}<1
$$

更新更保守；如果是高 SNR edge，则：

$$
m_{ji}^{snr}>1
$$

允许更积极学习。

---

### 4.3 Late spectral / prox

高 noise 下，后期启用 spectral shrink：

$$
a \leftarrow (I+\lambda R)^{-1}a
$$

或 prox：

$$
a_{t+1}
=
\arg\min_a
\frac{1}{2}\|a-\tilde a_{t+1}\|^2
+
\eta\lambda a^\top R a
$$

闭式为：

$$
a_{t+1}
=
(I+2\eta\lambda R)^{-1}
\tilde a_{t+1}
$$

它只在后期启用：

$$
t/T > 0.5
$$

或：

$$
\text{noise setting enabled}
$$

不进入 clean default。

---

### 4.4 Noise profile 默认候选

Fashion noise：

```text
base = geometry-aware
snr_mode = fixed15 or p60_15
snr_update_interval = 1 or 4
spectral_late = optional
prox_lambda in {1e-4, 1e-3}
```

KMNIST noise：

```text
base = geometry-aware
snr_mode = p60_15
snr_update_interval in {1, 4, 8}
spectral_late = optional
```

KMNIST noise 的目标不强制 memorization reduction，因为目前实验已经显示 memorization 难降。  
KMNIST 目标更务实：

$$
\text{clean test acc improvement vs AdamW}
$$

$$
\text{ECE reduction}
$$

如果同时降低 memorization，则作为额外收益。

---

## 5. FGO-v3-cost 设计

### 5.1 为什么需要成本实验

SNR 和 FSAM 都有成本问题。  
如果 noisy-label 准确率提升但 step time $1.7\times$，它不适合 Vision Scaling 默认，只适合 special setting。

因此 FGO-v3-cost 单独验证成本-收益 Pareto。

---

### 5.2 SNR cost reduction

测试：

```text
snr_update_interval = 1, 2, 4, 8
snr_grouping = edge, output_channel, layer
snr_samples = full_batch, half_batch, quarter_batch
```

如果每步都算 edge-level SNR 太贵，可以只每 $K$ step 更新一次：

$$
m_{ji,t}^{snr}
=
m_{ji,t-K}^{snr}
$$

或按 group 统计：

$$
\operatorname{SNR}_{G}
=
\frac{1}{|G|}
\sum_{(j,i)\in G}
\operatorname{SNR}_{ji}
$$

---

### 5.3 FSAM cost reduction

若保留 Functional-SAM，只测试：

```text
sam_interval = 8, 16
sam_start_frac = 0.5
rho_func = 0.003, 0.005
```

如果 step time 超过：

$$
1.3\times
$$

且 label-noise clean acc gain < $1\%$，则 FSAM 暂停。

---

## 6. 实验计划总览

本轮实验命名为：

$$
\boxed{
\text{FGO-v3：Profiled Functional Optimizer Repair}
}
$$

它不是全量 sweep，而是有明确目标的三阶段实验。

---

## 7. V3-0：复现与基准固定

### 7.1 目标

固定本轮所有 comparison baseline，避免和前轮结果混淆。

必须复现：

1. Fashion current geometry-aware；
2. KMNIST current geometry-aware；
3. KMNIST GFU-v1 momentum $\mu=0.8$；
4. Fashion / KMNIST SNR noisy-label baseline；
5. AdamW baseline。

---

### 7.2 数据集

```text
Fashion-MNIST
KMNIST
```

clean：

```text
train / val / test = 6000 / 1000 / 1000
epochs = 20
seeds = 0,1,2,3,4
```

noise：

```text
label_noise = 0.2, 0.4
epochs = 20
seeds = 0,1,2,3,4
```

---

### 7.3 记录指标

必须记录：

```text
task/test_acc
task/val_loss_auc
task/test_loss
compare/gap_vs_adamw
compare/gain_vs_geometry
kan/phi_prime_p95
kan/curvature_mean
jacobian/max_condition
branch/branch_over_adamw
ablation/no_kan_acc_drop
calibration/ece
compute/step_time_ms
```

noise 额外记录：

```text
noise/clean_test_acc
noise/noise_memorization_rate
noise/clean_recovery_rate
noise/noisy_train_acc_against_noisy_labels
noise/noisy_train_acc_against_clean_labels
```

---

### 7.4 成功标准

复现误差不应超过：

$$
|\Delta \text{test acc}|<0.3\%
$$

如果复现不稳定，则不得进入 V3-1/V3-2。

---

## 8. V3-1：Clean hard-task optimizer repair

### 8.1 目标

针对 KMNIST clean 剩余 gap，测试：

$$
\text{geometry-aware}
+
\text{safe mixed data metric}
+
\text{functional-direction momentum}
+
\text{relaxed trust}
$$

能否把 KMNIST gap 压到：

$$
<0.5\%
$$

同时保持 AUC / geometry 优势。

---

### 8.2 方法

比较：

```text
AdamW
geometry-current
GFU-v1 momentum mu=0.8
FGO-v3-clean-data-only
FGO-v3-clean-momentum-only
FGO-v3-clean-data+momentum
FGO-v3-clean-data+momentum+trust
```

不包含：

```text
SNR
FSAM
prox
spectral
hard branch target controller
no-grid metric
```

---

### 8.3 搜索空间

固定 base：

```text
branch_schedule = geometry_aware
branch_boost = 1.5
coeff_lr_boost = 1.2
branch_max_active_frac = 0.35
phi_high = 0.055
jac_high = 4.5
branch_high = 0.30
branch_final_scale = 0.9 or 1.0
```

搜索：

```text
coeff_lr = 0.3, 0.4, 0.5
rest_lr = 0.002, 0.003, 0.004
data_lambda_start = 0.05, 0.10, 0.25
data_lambda_final = 0.00, 0.05, 0.10
momentum_mu = 0.6, 0.8, 0.9
momentum_start_frac = 0.10, 0.25, 0.50
trust_clip_target = 0.05, 0.10, 0.15
```

采用两阶段搜索。

第一阶段用 seeds：

```text
0,1
```

快速筛选。  
第二阶段对 top configs 用 seeds：

```text
0,1,2,3,4
```

复核。

---

### 8.4 指标

性能：

```text
task/test_acc
task/val_acc
task/test_loss
conv/val_loss_auc
conv/steps_to_target_val_loss
conv/time_to_target_val_loss
```

几何：

```text
kan/phi_prime_p95
kan/phi_prime_max
kan/curvature_mean
jacobian/max_condition
credit/amplification_p95
```

branch：

```text
branch/branch_over_adamw
branch/mean_output_norm_ratio
ablation/no_kan_acc_drop
ablation/kan_logit_delta_norm
```

metric：

```text
metric/data_lambda
metric/data_condition
metric/mixed_condition
metric/eig_min
metric/eig_max
metric/grid_data_trace_ratio
metric/data_grid_alignment
```

momentum：

```text
momentum/enabled
momentum/mu
momentum/start_frac
momentum/direction_cos_current
momentum/m_norm
momentum/update_angle_to_nonmomentum
```

trust：

```text
trust/clip_rate
trust/target_clip_rate
trust/radius
trust/radius_expand_count
trust/radius_shrink_count
trust/bad_step_count
descent/actual_delta
descent/pred_delta
descent/ratio
```

---

### 8.5 可视化

Dashboard 页面必须包含：

1. **Clean accuracy Pareto**  
   x-axis: test acc gap vs AdamW；y-axis: val AUC improvement；color: method；size: J reduction。

2. **Geometry Pareto**  
   x-axis: $\phi'_{p95}$ reduction；y-axis: Jacobian reduction；color: test acc gap。

3. **Branch utilization plot**  
   x-axis: branch / AdamW；y-axis: no-KAN drop；color: test acc。

4. **Metric condition panel**  
   data $\lambda$ vs mixed condition；mixed condition vs accuracy；mixed condition vs J condition。

5. **Momentum stability panel**  
   momentum $\mu$ vs test acc；momentum update angle vs bad step rate；momentum M-norm over time。

6. **Trust effectiveness panel**  
   trust clip rate over time；clip rate vs accuracy；bad step count vs trust radius。

7. **Loss curves**  
   val loss curves for AdamW / geometry-current / best FGO-v3-clean。

---

### 8.6 成功标准

Weak success：

$$
\text{test acc gap vs AdamW}<0.8\%
$$

且：

$$
\text{val AUC improvement vs AdamW}>8\%
$$

Medium success：

$$
\text{test acc gap vs AdamW}<0.5\%
$$

$$
\text{val AUC improvement}>10\%
$$

$$
\phi'_{p95}\text{ reduction}>25\%
$$

$$
\text{Jacobian reduction}>15\%
$$

Strong success：

$$
\text{test acc gap vs AdamW}<0.3\%
$$

$$
\text{val AUC improvement}>10\%
$$

$$
\text{ECE reduction}>10\%
$$

$$
\text{branch / AdamW}\in[0.70,0.95]
$$

$$
\text{trust clip rate}<0.20
$$

且没有 geometry spike。

---

## 9. V3-2：Data metric safety audit

### 9.1 目标

专门解释 data metric 的收益与风险。  
FGO-v2 显示 `no-grid` 能提高 KMNIST accuracy，但会损害 $\phi'$ 和 Jacobian。  
本实验要判断是否能通过 normalization / condition clipping 保留 accuracy signal，同时避免几何变坏。

---

### 9.2 方法

比较：

```text
grid only
mixed raw lambda=0.10
mixed raw lambda=0.25
mixed trace-normalized lambda=0.10
mixed trace-normalized lambda=0.25
mixed eig-clipped lambda=0.10
mixed eig-clipped lambda=0.25
no-grid stress only
```

---

### 9.3 公式

trace-normalized data metric：

$$
\widetilde M_{\text{data}}
=
\frac{
\operatorname{tr}(M_{\text{grid}})
}{
\operatorname{tr}(M_{\text{data}})+\epsilon
}
M_{\text{data}}
$$

eigen-clipped data metric：

$$
M_{\text{data}}
=
Q\Lambda Q^\top
$$

$$
\Lambda_i
\leftarrow
\operatorname{clip}(\Lambda_i,\lambda_{\min},\lambda_{\max})
$$

mixed metric：

$$
M_{\text{mixed}}
=
M_{\text{grid}}
+
\lambda \widetilde M_{\text{data}}
+
\alpha R_1
+
\beta R_2
+
\rho I
$$

---

### 9.4 指标

```text
metric/mixed_condition
metric/data_condition
metric/data_grid_alignment
metric/update_cos_grid_vs_mixed
metric/update_norm_ratio_mixed_vs_grid
task/test_acc
conv/val_auc
kan/phi_prime_p95
jacobian/max_condition
```

其中：

$$
\text{data-grid alignment}
=
\frac{
\langle M_{\text{grid}}, M_{\text{data}}\rangle_F
}{
\|M_{\text{grid}}\|_F\|M_{\text{data}}\|_F+\epsilon
}
$$

---

### 9.5 成功标准

Data metric 只有在同时满足下列条件时才进入 clean profile：

$$
\text{test acc gain vs geometry-current}>0.2\%
$$

$$
\text{J reduction vs AdamW}>10\%
$$

$$
\phi'_{p95}\text{ reduction vs AdamW}>20\%
$$

$$
\text{mixed condition}<300
$$

否则只保留为 ablation。

---

## 10. V3-3：Functional momentum isolation

### 10.1 目标

解释为什么 GFU-v1 momentum 有效，而 FGO-v2 momentum 没有形成新的默认。  
本实验只研究 momentum，不引入 SNR / FSAM / prox。

---

### 10.2 方法

比较：

```text
geometry-current
GFU-v1 momentum mu=0.8 reproduction
functional direction momentum mu=0.6
functional direction momentum mu=0.8
functional direction momentum mu=0.9
functional direction momentum with start_frac=0.10
functional direction momentum with start_frac=0.25
whitened functional Adam-lite
naive Sobolev Adam negative control
```

---

### 10.3 公式

functional direction：

$$
p_t=-M_t^{-1}g_t
$$

momentum：

$$
v_t=\mu v_{t-1}+(1-\mu)p_t
$$

update：

$$
a_{t+1}=a_t+\eta_t v_t
$$

whitened Adam-lite：

$$
\tilde g_t=M_t^{-1/2}g_t
$$

在 $\tilde g_t$ 上做 Adam normalization：

$$
u_t=\frac{m_t}{\sqrt{s_t}+\epsilon}
$$

再映射：

$$
p_t=-M_t^{-1/2}u_t
$$

并做 M-norm clipping。

---

### 10.4 指标

```text
momentum/direction_cos_current
momentum/direction_cos_prev
momentum/m_norm
momentum/update_angle_to_base
momentum/effective_step_ratio
trust/clip_rate
stability/bad_step_count
branch/branch_over_adamw
kan/phi_prime_p95
jacobian/max_condition
task/test_acc
conv/val_auc
```

---

### 10.5 成功标准

Momentum 成功不是只看 accuracy。必须满足：

$$
\text{test acc gain vs geometry-current}>0.2\%
$$

$$
\text{bad step count}=0
$$

$$
\text{branch / AdamW}\in[0.70,0.95]
$$

$$
\text{J reduction vs AdamW}>10\%
$$

若 naive Adam 爆炸而 functional momentum 稳定，应记录为机制证据。

---

## 11. V3-4：Label-noise optimizer profile

### 11.1 目标

建立专门的 noisy-label functional optimizer，而不是用 clean optimizer 处理 noise。

---

### 11.2 数据集与设置

```text
Fashion-MNIST
KMNIST
label_noise = 0.1, 0.2, 0.4
seeds = 0,1,2,3,4
epochs = 20
```

---

### 11.3 方法

```text
AdamW
geometry-current
FGO-v3-clean best
SNR fixed15
SNR p60_15
SNR + spectral late
SNR + prox late
SNR + FSAM sparse
spectral only
prox only
```

---

### 11.4 指标

clean performance：

```text
noise/clean_test_acc
noise/clean_test_loss
compare/clean_acc_gain_vs_adamw
compare/clean_acc_gain_vs_geometry
```

memorization：

```text
noise/noisy_train_acc_against_noisy_labels
noise/noisy_train_acc_against_clean_labels
noise/noise_memorization_rate
noise/clean_recovery_rate
noise/corrupted_subset_train_acc_clean_label
noise/corrupted_subset_train_acc_noisy_label
```

calibration：

```text
calibration/ece
calibration/nll
calibration/brier_score
```

SNR：

```text
snr/global/mean
snr/global/p10
snr/global/p90
snr/low_snr_fraction
snr/high_snr_fraction
snr/lr_mult_mean
snr/lr_mult_p90
```

regularization：

```text
spectral/high_freq_energy
prox/shrink_norm
fsam/sharpness_gap
fsam/adv_loss_gap
```

cost：

```text
compute/step_time_ms
compute/time_ratio_vs_geometry
```

---

### 11.5 可视化

1. **Noise robustness curves**  
   x-axis: noise rate；y-axis: clean test acc。

2. **Memorization curve**  
   x-axis: epoch；y-axis: noise memorization rate。

3. **Calibration bar**  
   ECE by method and noise level。

4. **SNR distribution panel**  
   SNR histogram over edge groups。

5. **Clean acc vs memorization Pareto**  
   x-axis: noise memorization；y-axis: clean test acc。

6. **Cost vs gain plot**  
   x-axis: time ratio；y-axis: clean acc gain over geometry。

---

### 11.6 成功标准

Fashion noise success：

$$
\text{clean acc gain vs geometry-current}>1\%
$$

or:

$$
\text{memorization reduction vs geometry-current}>10\%
$$

and:

$$
\text{ECE reduction vs AdamW}>30\%
$$

KMNIST noise success：

$$
\text{clean acc gain vs AdamW}>2\%
$$

$$
\text{ECE reduction vs AdamW}>20\%
$$

Memorization reduction is optional for KMNIST because previous results show it is hard.

Cost condition:

$$
\text{step time ratio}<1.5
$$

If step time ratio $>1.5$, method remains stress-only.

---

## 12. V3-5：SNR cost reduction

### 12.1 目标

SNR 是 noisy-label 强组件，但成本偏高。  
本实验专门优化成本。

---

### 12.2 方法

比较：

```text
edge-level SNR every step
edge-level SNR interval=2
edge-level SNR interval=4
edge-level SNR interval=8
output-channel grouped SNR
layer-level SNR
half-batch SNR
quarter-batch SNR
```

---

### 12.3 指标

```text
compute/step_time_ms
compute/time_ratio_vs_geometry
snr/staleness_steps
snr/grouping_mode
noise/clean_test_acc
noise/memorization_rate
calibration/ece
kan/phi_prime_p95
jacobian/max_condition
```

---

### 12.4 成功标准

SNR-cost profile 通过需要：

$$
\text{retain at least }80\%\text{ of clean acc gain}
$$

and:

$$
\text{step time ratio}<1.25
$$

If SNR interval=4 or grouped SNR keeps most gain, it becomes the default noisy-label implementation.

---

## 13. V3-6：CIFAR readiness decision

### 13.1 目标

决定是否进入 CIFAR-10 small ConvStem-DGKAN，以及用哪个 optimizer 作为默认。

---

### 13.2 候选

```text
current geometry-aware
FGO-v3-clean best
FGO-v3-noise best
GFU-v1 momentum mu=0.8
```

---

### 13.3 Readiness gate

进入 CIFAR clean default 的条件：

$$
\text{Fashion clean no regression}
$$

$$
\text{KMNIST clean gap vs AdamW}<0.8\%
$$

$$
\text{val AUC improvement vs AdamW}>8\%
$$

$$
\text{J reduction}>10\%
$$

$$
\text{time ratio}<1.25
$$

如果 FGO-v3-clean 不满足这些条件，则 CIFAR 默认仍使用 current geometry-aware。  
FGO-v3-clean 只作为 ablation。

进入 CIFAR noisy-label / robustness ablation 的条件：

$$
\text{Fashion noise clean acc gain vs geometry}>1\%
$$

or:

$$
\text{memorization reduction}>10\%
$$

并且：

$$
\text{time ratio}<1.5
$$

---

## 14. 统一 W&B / CSV 指标表

### 14.1 任务与收敛

```text
train/loss
val/loss
test/loss
train/acc
val/acc
test/acc
conv/train_loss_auc
conv/val_loss_auc
conv/steps_to_target_val_loss
conv/time_to_target_val_loss
```

### 14.2 泛化与校准

```text
generalization/train_test_gap_acc
generalization/train_test_gap_loss
calibration/ece
calibration/nll
calibration/brier_score
```

### 14.3 KAN 几何

```text
kan/phi_prime_p95
kan/phi_prime_max
kan/curvature_mean
kan/sobolev_norm
jacobian/max_condition
jacobian/mean_condition
credit/amplification_p95
```

### 14.4 Branch utilization

```text
branch/mean_output_norm_ratio
branch/branch_over_adamw
branch/effective_alpha_mean
ablation/no_kan_acc_drop
ablation/kan_logit_delta_norm
```

### 14.5 Metric

```text
metric/data_lambda
metric/data_condition
metric/mixed_condition
metric/eig_min
metric/eig_max
metric/data_grid_alignment
metric/update_cos_grid_vs_mixed
metric/update_norm_ratio_mixed_vs_grid
```

### 14.6 Trust

```text
trust/clip_rate
trust/target_clip_rate
trust/radius
trust/shrink_count
trust/expand_count
descent/pred_delta
descent/actual_delta
descent/ratio
descent/bad_step_count
```

### 14.7 Momentum

```text
momentum/enabled
momentum/mu
momentum/start_frac
momentum/direction_cos_current
momentum/direction_cos_prev
momentum/m_norm
momentum/update_angle_to_base
```

### 14.8 SNR

```text
snr/mode
snr/update_interval
snr/grouping_mode
snr/global_mean
snr/global_p10
snr/global_p90
snr/low_snr_fraction
snr/high_snr_fraction
snr/lr_mult_mean
snr/lr_mult_p90
```

### 14.9 Noise

```text
noise/rate
noise/clean_test_acc
noise/noise_memorization_rate
noise/clean_recovery_rate
noise/noisy_train_acc_against_noisy_labels
noise/noisy_train_acc_against_clean_labels
```

### 14.10 Cost

```text
compute/step_time_ms
compute/time_ratio_vs_geometry
compute/throughput_samples_per_sec
memory/peak_allocated_mb
```

---

## 15. Dashboard 设计

### Page 1：FGO-v3 overview

展示所有方法的：

```text
test acc
val AUC
ECE
phi_prime_p95
Jacobian condition
step time
```

用一个 scorecard 标注：

```text
clean pass
noise pass
cost pass
cifar readiness
```

---

### Page 2：Clean hard-task Pareto

Fashion / KMNIST clean 的 Pareto 图：

```text
x = test acc gap vs AdamW
y = val AUC improvement
color = method
size = Jacobian reduction
```

再加：

```text
branch / AdamW vs test acc
no-KAN drop vs test acc
```

---

### Page 3：Metric safety

展示：

```text
data lambda vs test acc
data lambda vs mixed condition
mixed condition vs Jacobian condition
data-grid alignment vs update angle
```

---

### Page 4：Functional momentum

展示：

```text
momentum mu vs test acc
momentum m_norm over time
update angle to base over time
bad step count by momentum
```

---

### Page 5：Trust region

展示：

```text
clip rate over time
trust radius over time
clip rate vs branch ratio
descent ratio histogram
bad step count
```

---

### Page 6：Label-noise profile

展示：

```text
noise rate vs clean test acc
noise memorization rate over epochs
ECE by noise rate
SNR distribution
clean acc vs memorization Pareto
```

---

### Page 7：Cost profile

展示：

```text
time ratio vs clean acc gain
time ratio vs noise acc gain
step time by method
SNR interval vs performance
```

---

### Page 8：Failure table

结构化失败表：

```text
branch_underactive
branch_overactive
geometry_spike
metric_ill_conditioned
trust_too_conservative
momentum_unstable
snr_too_slow
snr_geometry_spike
noise_no_gain
clean_no_gain
cifar_not_ready
```

---

## 16. Failure table 规则

### 16.1 branch_underactive

触发：

$$
\text{branch / AdamW}<0.5
$$

且：

$$
\text{test acc gap vs AdamW}>1\%
$$

---

### 16.2 branch_overactive

触发：

$$
\text{branch / AdamW}>1.2
$$

或：

$$
\text{no-KAN drop high but J condition worsens}
$$

---

### 16.3 geometry_spike

触发：

$$
\kappa(J)>\kappa(J)_{\text{AdamW}}
$$

or:

$$
\phi'_{p95}>\phi'_{p95,\text{AdamW}}
$$

---

### 16.4 metric_ill_conditioned

触发：

$$
\kappa(M_{\text{mixed}})>500
$$

or:

$$
\lambda_{\min}(M_{\text{mixed}})<10^{-5}
$$

---

### 16.5 trust_too_conservative

触发：

$$
\text{clip rate}>0.3
$$

and:

$$
\text{branch / AdamW}<0.6
$$

---

### 16.6 momentum_unstable

触发：

$$
\text{bad step count}>0
$$

or:

$$
\text{update M-norm spike}>3\times\text{median}
$$

---

### 16.7 snr_too_slow

触发：

$$
\text{time ratio}>1.5
$$

and noise gain insufficient.

---

## 17. 最终决策规则

### 17.1 Clean default

Clean default 按以下优先级选择：

1. 若 FGO-v3-clean medium success，则作为 Stage I clean default；
2. 否则保留 current geometry-aware；
3. GFU-v1 momentum 作为 KMNIST / hard-task ablation；
4. SNR / FSAM / prox 不进入 clean default。

---

### 17.2 Noise default

Noise default 按以下规则选择：

1. Fashion noise：优先 SNR fixed15 / p60_15；
2. KMNIST noise=0.2：优先 SNR p60_15；
3. KMNIST noise=0.4：若 SNR 不如 geometry-current，则保留 geometry-current；
4. spectral / prox 作为 secondary ablation。

---

### 17.3 CIFAR readiness

进入 CIFAR-10 small 之前，至少需要：

$$
\text{Fashion clean no regression}
$$

$$
\text{KMNIST clean gap}<0.8\%
$$

or current geometry-aware remains default.

如果 FGO-v3-clean 未通过，不阻止 CIFAR。  
因为 Stage G 已经通过 CIFAR entry gate；只是 CIFAR default 不使用 FGO-v3-clean。

---

## 18. 本轮最终预期

FGO-v3 的目标不是一次性证明一个万能 optimizer，而是做出更准确的分工：

```text
Clean / default:
  current geometry-aware or FGO-v3-clean if it passes

Hard-task clean:
  FGO-v3-clean data+momentum profile

Label-noise:
  FGO-v3-noise SNR / spectral profile

Corruption:
  do not claim unless separately improved

Vision scaling:
  use clean default first; use noise profile only in noisy-label ablation
```

如果 FGO-v3-clean 成功，则我们可以升级主线说：

> Geometry-aware functional update can be further improved by safe data/grid metric mixing and functional-direction momentum, closing the hard-task accuracy gap while preserving better credit geometry.

如果 FGO-v3-clean 失败，但 FGO-v3-noise 成功，则叙事变为：

> Current geometry-aware remains the clean default, while SNR-adaptive functional update provides a specialized noisy-label robustness profile.

如果二者都失败，则继续保留 current geometry-aware，FGO-v2/v3 作为 negative result，进入 CIFAR-10 small 前不再继续优化 functional update。

---

## 19. 一句话总结

FGO-v2 没达到要求，不是因为 functional optimizer 的方向错了，而是因为它把太多组件一次性堆在 clean default 上，导致 branch under-active、trust 过保守、data metric 与 grid prior 失衡、SNR/FSAM 成本过高。FGO-v3 应该转向 profile 化设计：clean profile 只保留 safe data metric、functional-direction momentum 和 relaxed trust；noise profile 专门使用 SNR / spectral / FSAM；cost profile 专门压低这些组件的开销。最终是否进入 Stage I，不取决于 FGO-v3 是否“全赢”，而取决于它是否在对应 profile 上通过清楚的 accuracy / AUC / geometry / cost gate。
