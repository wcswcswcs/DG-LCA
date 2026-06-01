# DG-KAN FGO-v4：实验结果分析与下一步实验计划

> 版本：v4 planning  
> 依据：FGO-v3 实验结果与前序 Stage F/G/GFU/FGO-v2 结论  
> 公式格式：Typora 友好，统一使用 `$...$` 与 `$$...$$`  
> 核心目标：不再继续堆叠一个“大一统 optimizer”，而是把 FGO-v3 暴露出的有效信号拆成可控 profile：clean hard-task、noisy-label、cost-aware SNR、CIFAR readiness。

---

## 0. 执行摘要

FGO-v3 的结果说明：**FGO-v3-clean 还没有资格替代 current geometry-aware functional update 作为 clean 默认方法**，但它已经比 FGO-v2 清楚很多。它把问题拆成了三个稳定结论。

第一，clean task 上，current geometry-aware 仍然是最可靠默认方法。Fashion-MNIST 上 current geometry-aware 仍是 clean 最强；KMNIST 上 FGO-v3-clean full 把 gap 压到 $0.76\%$，通过 weak clean gate，但没有超过 AdamW，也没有达到 medium / strong gate。

第二，data metric 的信号非常明确：**no-grid data metric 有强 accuracy signal，但会牺牲全域几何**。KMNIST no-grid stress 达到 $0.8724$，gap vs AdamW 只有 $0.36\%$，但 Jacobian reduction 只有 $2.0\%$，$\phi'$ reduction 只有 $13.3\%$。这说明 data metric 不是没用，而是不能作为裸 default。

第三，noisy-label profile 是 FGO-v3 最明确的正结果。Fashion 和 KMNIST 在 label-noise 下都得到 clean accuracy / ECE 改善；但 SNR 成本没有解决，许多有效 SNR 变体仍然在 $1.4\times$ 到 $2.0\times$ step-time 区间。因此 SNR 是 noisy-label candidate，不是 clean default。

因此，FGO-v4 不应继续扩大 FGO-v3-clean full，而应进入 profile 化升级：

$$
\boxed{
\text{FGO-v4-clean}
=
\text{geometry-aware}
+
\text{data-pulse}
+
\text{grid re-anchor}
+
\text{functional-direction momentum}
+
\text{relaxed safety trust}
}
$$

$$
\boxed{
\text{FGO-v4-noise}
=
\text{geometry-aware}
+
\text{cost-aware SNR}
+
\text{late spectral/prox regularization}
}
$$

$$
\boxed{
\text{FGO-v4-readiness}
=
\text{clean profile selection}
+
\text{noise profile selection}
+
\text{CIFAR entry decision}
}
$$

FGO-v4 的核心不是“再发明一个更复杂 optimizer”，而是回答三个具体问题：

1. 能否利用 no-grid / data metric 的 accuracy signal，同时通过 late grid re-anchor 把 $\phi'$ 和 Jacobian 拉回来？
2. 能否把 SNR noisy-label 收益压到可接受成本，例如 $\leq 1.3\times$ geometry-aware step time？
3. clean default 是否仍保持 current geometry-aware，还是 FGO-v4-clean 能真正替代它进入 Stage I / CIFAR small？

---

## 1. FGO-v3 实验结果分析

### 1.1 Clean baseline 复现说明主线没有漂移

FGO-v3 复现了关键 baseline。Fashion-MNIST 上 current geometry-aware 仍是 clean 最强方法：

```text
Fashion current geometry-aware:
  test acc = 0.8466
  AUC red = 17.8%
  phi' red = 27.3%
  J red = 50.4%
  branch / AdamW = 0.523
```

FGO-v3-clean full 在 Fashion 上 test acc 为 $0.8378$，比 current geometry-aware 低 $0.88\%$，因此不能替换 Fashion clean default。

KMNIST 上，FGO-v3-clean full 有价值：

```text
KMNIST FGO-v3-clean full:
  test acc = 0.8684
  gap vs AdamW = 0.76%
  gain vs geometry-current = 0.26%
  AUC red = 11.4%
  phi' red = 31.3%
  J red = 11.1%
  branch / AdamW = 0.709
  trust clip = 0.021
```

这说明 FGO-v3-clean full 是 KMNIST hard-task candidate，但它仍没达到 medium gate。主要原因是：

$$
\text{gap vs AdamW}=0.76\% > 0.5\%
$$

且：

$$
\text{J reduction}=11.1\% < 15\%
$$

因此它是 weak pass，不是 clean default。

---

### 1.2 Data metric 的核心矛盾：accuracy signal 与 geometry safety 冲突

V3-2 的最重要现象是 KMNIST no-grid stress：

```text
KMNIST no-grid:
  test acc = 0.8724
  gap vs AdamW = 0.36%
  gain vs geometry = 0.66%
  AUC red = 14.7%
  phi' red = 13.3%
  J red = 2.0%
  mixed cond = 183.0
```

它说明：data metric 确实更贴合 hard-task 的训练分布，能带来明显 accuracy signal。但是去掉 grid Sobolev prior 后，函数空间的全域约束变弱，导致 $\phi'$ 和 Jacobian 的几何优势几乎消失。

所以 no-grid 不是失败，而是告诉我们一个方向：

$$
\boxed{
\text{data metric should be used as an early pulse, not as a permanent default.}
}
$$

FGO-v4 因此引入 **data-pulse to grid re-anchor**：早期允许 data metric 更强，帮助 branch 学到任务；后期把 metric 重新锚定到 grid / Sobolev prior，恢复函数复杂度控制。

---

### 1.3 Functional momentum 不是单独有效，而是依赖 metric / schedule 组合

V3-3 显示，单独的 simple functional momentum 没能复制 GFU-momentum 的收益。GFU-momentum 的有效性来自 data metric、controller、trust、schedule 的组合，而不是只加一个 $\mu$ buffer。

同时，naive Sobolev Adam 再次爆炸：

```text
naive Sobolev Adam:
  test acc = 0.7874
  branch / AdamW = 138.9x
  bad step = 0.071
  phi' / J 全面崩坏
```

这说明 FGO-v4 不能引入 raw Adam-style normalization。保留的只有 **functional-direction momentum**：

$$
p_t = -M_t^{-1}g_t
$$

$$
v_t = \mu v_{t-1} + (1-\mu)p_t
$$

然后对 $v_t$ 做 M-norm safety clipping：

$$
v_t \leftarrow v_t \cdot \min\left(1, \frac{\tau}{\|v_t\|_{M_t}+\epsilon}\right)
$$

---

### 1.4 Noisy-label profile 成功，但成本没有解决

V3-4 是本轮最强正结果。

Fashion label noise：

```text
noise 0.1 best = spectral:
  clean test acc = 0.8234
  gain vs geometry = 0.90%
  gain vs AdamW = 3.60%
  ECE red vs AdamW = 67.3%
  time / geometry = 1.20

noise 0.2 best = SNR + sparse FSAM:
  clean test acc = 0.7940
  gain vs geometry = 0.94%
  gain vs AdamW = 6.18%
  time / geometry = 1.46

noise 0.4 best = SNR fixed15 full:
  clean test acc = 0.6778
  gain vs geometry = 2.26%
  gain vs AdamW = 8.60%
  time / geometry = 2.01
```

KMNIST label noise：

```text
noise 0.2 best = SNR p60:
  clean test acc = 0.7566
  gain vs geometry = 1.64%
  gain vs AdamW = 4.08%
  ECE red vs AdamW = 36.9%
  memorization = 1.000

noise 0.4 best = SNR + spectral:
  clean test acc = 0.6010
  gain vs geometry = 0.80%
  gain vs AdamW = 5.16%
  ECE red vs AdamW = 36.1%
  memorization = 1.000
```

结论是：SNR profile 对 noisy-label clean accuracy 和 calibration 有真实收益，但 KMNIST 不减少 memorization，且成本偏高。

V3-5 cost audit 显示：

- Fashion high-noise 下 `SNR i4 edge quarter` 最强，clean acc 到 $0.6922$，但 time / geometry = $1.95$；
- `SNR i4 output` 成本降到 $1.45\times$，但仍超过 $1.25\times$；
- KMNIST 里所有有收益的 SNR 变体也明显超过 $1.25\times$。

因此 V4-noise 的目标不是再证明 SNR 有用，而是降低 SNR 成本。

---

### 1.5 CIFAR readiness 仍未通过

自动选择给出：

```text
V3_6/cifar_ready = false
reason = fashion_ok=True; kmnist_ok=False
```

更重要的是人工判定：KMNIST no-grid 不能作为 clean default，因为它违反 no-grid stress-only 约束。当前 Stage I / CIFAR clean default 仍应该是 current geometry-aware functional update，而不是 FGO-v3-clean full 或 no-grid。

FGO-v4 的任务是判断：是否能构造一个既接近 no-grid accuracy，又保住 geometry 的 clean profile。如果能，才有资格替代 current geometry-aware；否则 Stage I 仍使用 current geometry-aware，FGO-v4 只作为 hard-task / noisy-label ablation。

---

## 2. FGO-v4 的核心假设

### 2.1 Clean hard-task 假设

V3 的 KMNIST no-grid 结果提示：hard-task accuracy 可能需要更 data-local 的 metric，而 current geometry-aware / mixed metric 太偏全域平滑，导致 branch 参与度和任务拟合仍不足。

但裸 no-grid 破坏几何。因此 FGO-v4-clean 的假设是：

$$
\boxed{
\text{early data-local metric pulse improves hard-task accuracy,}
\quad
\text{late grid re-anchor restores geometry.}
}
$$

具体形式：

$$
M_t = (1-\lambda_t)M_{grid} + \lambda_t M_{data}
$$

其中 $\lambda_t$ 不再固定，而是先高后低：

$$
\lambda_t =
\begin{cases}
\lambda_{early}, & t < T_{pulse} \\
\text{decay to } \lambda_{final}, & T_{pulse}\leq t < T_{anchor} \\
\lambda_{final}, & t \geq T_{anchor}
\end{cases}
$$

FGO-v4-clean 要验证：

$$
\text{KMNIST gap vs AdamW}<0.5\%
$$

同时：

$$
\phi'\text{ reduction}>20\%,\quad J\text{ reduction}>10\%-15\%
$$

如果做不到，则 current geometry-aware 继续作为 clean default。

---

### 2.2 Momentum 组合假设

V3 显示 simple momentum 不足，GFU-momentum 的收益来自组合。FGO-v4 不再做单独 momentum，而是只测试：

$$
\text{data-pulse metric} + \text{functional-direction momentum} + \text{relaxed safety trust}
$$

方向先通过 metric 得到：

$$
p_t = -M_t^{-1}g_t
$$

再做 functional momentum：

$$
v_t=\mu v_{t-1}+(1-\mu)p_t
$$

并对 $v_t$ 做 M-norm 限制：

$$
\|v_t\|_{M_t}\leq \tau_t
$$

momentum 必须满足两个安全条件：

$$
\text{branch / AdamW}<1.0
$$

$$
\kappa(J)<1.1\times \kappa(J)_{AdamW}
$$

---

### 2.3 Noise profile 假设

SNR 对 label-noise 有效，但成本高。FGO-v4-noise 的假设是：

$$
\boxed{
\text{SNR does not need per-edge full refresh every step.}
}
$$

可以通过以下方式降低成本：

1. 分组 SNR：edge $\rightarrow$ output-channel $\rightarrow$ layer；
2. 降低更新频率：每 $K$ steps 刷新一次；
3. 样本 sketch：只用 batch 的一部分估计 SNR；
4. top-SNR / low-SNR selective update：只对不稳定 edge 进行细粒度 SNR；
5. 与 late spectral / prox 结合，减少 FSAM 依赖。

目标是：

$$
\text{time ratio}\leq1.3\times
$$

同时保留 noisy-label clean acc gain 的大部分：

$$
\text{gain retained}\geq 80\%
$$

---

## 3. FGO-v4 方法族

### 3.1 Baseline 方法

FGO-v4 每个实验都必须包含以下 baseline：

| 方法 | 作用 |
|---|---|
| AdamW | accuracy baseline |
| current geometry-aware | 当前 clean default |
| FGO-v3-clean full | KMNIST weak-pass candidate |
| no-grid stress | accuracy upper signal，不可默认 |
| SNR previous best | noisy-label upper signal |

---

### 3.2 FGO-v4-clean profile

FGO-v4-clean 是 clean hard-task 候选。它不替代 Fashion 默认，主要用于 KMNIST 和后续 CIFAR hard setting。

其形式为：

$$
\boxed{
\text{FGO-v4-clean}
=
\text{geometry-aware}
+
\text{data-pulse metric}
+
\text{functional momentum}
+
\text{grid re-anchor}
+
\text{relaxed trust}
}
$$

#### 3.2.1 Data-pulse metric

$$
M_t=(1-\lambda_t)M_{grid}+\lambda_t M_{data}
$$

搜索：

```text
lambda_early = 0.25, 0.50, 0.75, 1.00
lambda_final = 0.00, 0.05, 0.10
pulse_frac = 0.15, 0.25, 0.35
anchor_decay = linear, cosine
```

其中 $\lambda_{early}=1.00$ 表示 early no-grid pulse，但必须 late re-anchor，否则不允许作为 default。

#### 3.2.2 Grid re-anchor

re-anchor 后增加 Sobolev / grid weight，并可选 late proximal shrink：

$$
a_{t+1} = (I+2\eta\lambda_{prox}R)^{-1}\tilde a_{t+1}
$$

搜索：

```text
prox_lambda = 0, 1e-5, 3e-5, 1e-4
prox_start_frac = pulse_frac + 0.25
```

#### 3.2.3 Functional-direction momentum

$$
v_t=\mu v_{t-1}+(1-\mu)p_t
$$

搜索：

```text
momentum_mu = 0.0, 0.6, 0.8
momentum_start_frac = 0.05, 0.15
reset_on_reanchor = true, false
```

默认建议：$\mu=0.8$，但在 re-anchor 时重置或衰减动量，避免 data-metric phase 的方向带到 grid phase 后造成 geometry mismatch。

#### 3.2.4 Relaxed trust

trust-region 只作为安全阀，不作为强裁剪器。

目标 clip rate：

$$
0.00\leq \text{clip rate}\leq0.10
$$

搜索：

```text
trust_radius = 0.10, 0.15, 0.20
trust_mode = off, safety_only, damping_only, safety_plus_damping
```

若 clip rate 超过 $0.20$ 且 branch_underactive，则判定为 trust_too_conservative。

---

### 3.3 FGO-v4-noise profile

FGO-v4-noise 只用于 noisy-label / memorization / calibration，不用于 clean default。

其形式为：

$$
\boxed{
\text{FGO-v4-noise}
=
\text{geometry-aware}
+
\text{SNR-lite}
+
\text{late spectral/prox}
+
\text{optional sparse FSAM}
}
$$

#### 3.3.1 SNR-lite

原始 edge SNR 定义：

$$
\operatorname{SNR}_{ji}
=
\frac{\|\mu_{ji}\|}{\sqrt{\operatorname{mean}(v_{ji})}+\epsilon}
$$

SNR-lite 支持以下近似：

```text
grouping = edge, output_channel, layer
update_interval = 1, 2, 4, 8, 16
sample_frac = 0.25, 0.50, 1.00
snr_percentile = p50, p60, p75
```

边级 full SNR 作为上界，output/channel/layer SNR 作为低成本候选。

#### 3.3.2 SNR multiplier

每个 edge / group 的学习率 multiplier：

$$
m_{snr}=	ext{clip}\left(\frac{\operatorname{SNR}}{\tau_{snr}}, m_{min}, m_{max}\right)
$$

更新：

$$
\Delta a_{ji} = -\eta m_{snr,ji}(M+\rho I)^{-1}g_{ji}
$$

搜索：

```text
m_min = 0.5, 0.7
m_max = 1.2, 1.5
snr_ema_beta = 0.90, 0.95, 0.98
```

#### 3.3.3 Late spectral/prox

noisy-label 下 late spectral / prox 有价值，但 clean 下不适合作默认。V4-noise 只在中后期打开：

```text
spectral_start_frac = 0.50, 0.70
prox_start_frac = 0.50, 0.70
prox_lambda = 1e-5, 3e-5, 1e-4
spectral_lambda = 1e-4, 3e-4, 1e-3
```

#### 3.3.4 Sparse FSAM

FSAM 不是 clean 默认，只在 Fashion noise $\eta=0.1/0.2$ 或 stress 中保留：

```text
fsam_interval = 8, 16
fsam_sample_frac = 0.25, 0.50
fsam_rho = 0.001, 0.003
```

---

## 4. 实验总体路线

FGO-v4 分成六个 package。

```text
V4-0: baseline reproduction and sanity
V4-1: clean data-pulse metric curriculum
V4-2: clean re-anchor / momentum / trust ablation
V4-3: noise SNR-lite cost-performance optimization
V4-4: profile stress validation
V4-5: CIFAR readiness and final method selection
```

每个 package 都要求 5 seeds。搜索阶段可以用 3 seeds，confirm 必须用 5 seeds。

---

## 5. V4-0：baseline reproduction and sanity

### 5.1 目标

复现 FGO-v3 的关键 baseline，确保新代码没有漂移。

### 5.2 数据集

```text
Fashion-MNIST
KMNIST
```

### 5.3 方法

```text
AdamW
current geometry-aware
FGO-v3-clean full
GFU-momentum mu=0.8
no-grid stress
SNR previous best
```

### 5.4 必须记录

```text
task/train_loss
task/val_loss
task/test_acc
conv/val_auc
geometry/phi_prime_p95
geometry/max_jac_condition
branch/branch_over_adamw
ablation/no_kan_drop
calibration/ece
optimizer/trust_clip_rate
compute/step_time_ms
```

### 5.5 通过标准

复现误差应小于：

$$
|\Delta \text{test acc}| < 0.3\%
$$

$$
|\Delta \text{AUC red}| < 2\%
$$

若复现失败，则不进入 V4-1。

---

## 6. V4-1：Clean data-pulse metric curriculum

### 6.1 目标

验证 early no-grid / data metric 的 accuracy signal 是否能通过 late grid re-anchor 转化为 safe clean optimizer。

### 6.2 核心问题

V3 no-grid 的问题是：

$$
\text{accuracy high},\quad \text{geometry weak}
$$

V4-1 要验证是否能得到：

$$
\text{accuracy close to no-grid},\quad \text{geometry close to geometry-aware}
$$

### 6.3 数据集

主数据集：

```text
KMNIST
```

Fashion 只做 sanity，不作为主要优化对象，因为 Fashion current geometry-aware 已经强。

### 6.4 模型

```text
Residual DG-KAN
depth = 4
hidden_dim = 64
basis_count = 16
alpha_init = 1.5
train / val / test = 6000 / 1000 / 1000
epochs = 20
batch_size = 256
seeds search = 0,1,2
seeds confirm = 0,1,2,3,4
```

### 6.5 搜索空间

```text
lambda_early = 0.25, 0.50, 0.75, 1.00
lambda_final = 0.00, 0.05, 0.10
pulse_frac = 0.15, 0.25, 0.35
anchor_decay = linear, cosine
momentum_mu = 0.0, 0.8
momentum_reset_on_anchor = true, false
trust_radius = 0.10, 0.15, 0.20
prox_lambda = 0, 3e-5, 1e-4
```

为了避免组合爆炸，第一轮采用分阶段搜索。

第一步只扫：

```text
lambda_early, lambda_final, pulse_frac
momentum_mu = 0
trust_radius = 0.15
prox_lambda = 0
```

第二步固定 top-6 metric curriculum，再扫：

```text
momentum_mu = 0.0, 0.8
momentum_reset_on_anchor = true, false
prox_lambda = 0, 3e-5, 1e-4
trust_radius = 0.10, 0.15, 0.20
```

### 6.6 关键指标

任务指标：

```text
task/test_acc
task/val_loss
conv/val_auc
compare/gap_vs_adamw
compare/gain_vs_geometry_current
```

metric 指标：

```text
metric/data_lambda_current
metric/data_grid_alignment
metric/grid_data_trace_ratio
metric/mixed_condition
metric/eig_min
metric/eig_max
metric/reanchor_step
```

phase 指标：

```text
phase/pulse_frac
phase/reanchor_progress
phase/anchor_decay_type
phase/metric_phase_id
```

geometry 指标：

```text
geometry/phi_prime_p95
geometry/phi_prime_max
geometry/max_jac_condition
geometry/credit_amplification_p95
geometry/curvature_energy
```

branch 指标：

```text
branch/branch_over_adamw
branch/mean_output_norm_ratio
branch/no_kan_drop
branch/kan_logit_delta_norm
```

optimizer 指标：

```text
optimizer/trust_clip_rate
optimizer/momentum_mu
optimizer/momentum_reset_count
optimizer/descent_ratio_median
optimizer/bad_step_rate
```

### 6.7 可视化

W&B 页面需要包含：

1. **Data pulse timeline**：$\lambda_t$ 随 epoch 变化；
2. **Accuracy vs geometry Pareto**：x = gap vs AdamW，y = J reduction，点大小 = branch / AdamW；
3. **No-grid recovery plot**：比较 no-grid、current geometry、data-pulse reanchor 的 accuracy / $\phi'$ / J；
4. **Metric condition over time**；
5. **Branch utilization over time**；
6. **Val loss curves**；
7. **Failure type by config**。

### 6.8 成功标准

Weak pass：

$$
\text{KMNIST gap vs AdamW}<0.8\%
$$

$$
\text{AUC red vs AdamW}>10\%
$$

$$
\text{trust clip}<0.10
$$

Medium pass：

$$
\text{KMNIST gap vs AdamW}<0.5\%
$$

$$
\phi'\text{ red}>20\%
$$

$$
J\text{ red}>10\%
$$

Strong pass：

$$
\text{KMNIST gap vs AdamW}<0.3\%
$$

$$
\text{AUC red}>12\%
$$

$$
\phi'\text{ red}>25\%
$$

$$
J\text{ red}>15\%
$$

如果 accuracy 达到 no-grid 水平但 $J$ reduction 低于 $5\%$，则判为 `data_metric_geometry_failure`，不能进入 default。

---

## 7. V4-2：Clean re-anchor / momentum / trust ablation

### 7.1 目标

解释 V4-1 中哪些组件真正有效。尤其要回答：

1. 收益是否来自 data-pulse 本身？
2. 收益是否来自 functional momentum？
3. late grid re-anchor 是否真的恢复 geometry？
4. trust-region 是否只是安全阀，而不是过度裁剪？

### 7.2 方法

从 V4-1 top-3 clean configs 中选出候选，做 ablation：

```text
full candidate
minus data-pulse
minus re-anchor
minus momentum
minus trust
minus prox
fixed lambda baseline
no-grid permanent baseline
```

### 7.3 指标

除了 V4-1 指标，还要记录 component gain：

```text
ablation/gain_from_data_pulse
ablation/gain_from_reanchor
ablation/gain_from_momentum
ablation/gain_from_trust
ablation/gain_from_prox
```

定义：

$$
\text{gain}_{component}=\text{metric}_{full}-\text{metric}_{minus\ component}
$$

其中 metric 可以是 test acc、AUC、$\phi'$ reduction、J reduction。

### 7.4 通过标准

如果 full candidate 过 gate，但去掉 re-anchor 后几何崩坏，则说明 re-anchor 是必要组件。

如果 full candidate 只比 no-grid 好在几何，但 test acc 明显掉，则说明 re-anchor 太强，需要降低 $\lambda_{final}$ 或 prox。

如果 momentum ablation 不影响结果，则 momentum 不进入 default。

---

## 8. V4-3：Noise SNR-lite cost-performance optimization

### 8.1 目标

把 V3 的 noisy-label SNR 收益压到可接受成本。

当前问题：

$$
\text{SNR works, but time ratio often } 1.4\times - 2.0\times
$$

V4-3 目标：

$$
\text{time ratio}\leq1.3\times
$$

同时保留大部分 clean accuracy gain。

### 8.2 数据集与噪声

```text
Fashion-MNIST noise = 0.2, 0.4
KMNIST noise = 0.2, 0.4
```

### 8.3 搜索空间

```text
snr_grouping = edge, output_channel, layer
snr_update_interval = 1, 2, 4, 8, 16
snr_sample_frac = 0.25, 0.50, 1.00
snr_percentile = p50, p60, p75
snr_ema_beta = 0.90, 0.95, 0.98
snr_lr_min = 0.5, 0.7
snr_lr_max = 1.2, 1.5
spectral_start_frac = 0.50, 0.70
prox_lambda = 0, 3e-5, 1e-4
```

分两轮。

第一轮 cost search：

```text
固定 noisy-label best 上界，扫 grouping / interval / sample_frac。
目标是找到 time <= 1.3x 的候选。
```

第二轮 quality recovery：

```text
在低成本候选上加入 spectral / prox / percentile 调节。
目标是恢复 accuracy gain 和 ECE gain。
```

### 8.4 必须记录

Noisy-label task：

```text
noise/rate
noise/clean_test_acc
noise/noisy_train_acc_against_noisy_labels
noise/noisy_train_acc_against_clean_labels
noise/noise_memorization_rate
noise/clean_recovery_rate
noise/corrupted_subset_train_acc_clean_label
noise/corrupted_subset_train_acc_noisy_label
```

SNR：

```text
snr/grouping
snr/update_interval
snr/sample_frac
snr/staleness_steps
snr/update_rate
snr/mean
snr/p10
snr/p50
snr/p90
snr/low_snr_fraction
snr/high_snr_fraction
snr/lr_mult_mean
snr/lr_mult_p90
```

Cost：

```text
compute/step_time_ms
compute/time_ratio_vs_geometry
compute/snr_update_time_ms
compute/snr_overhead_pct
memory/peak_mb
memory/snr_state_mb
```

Calibration：

```text
calibration/ece
calibration/nll
calibration/brier_score
```

### 8.5 可视化

1. **Accuracy gain vs time ratio Pareto**；
2. **ECE reduction vs time ratio Pareto**；
3. **Memorization rate vs clean acc**；
4. **SNR staleness vs test acc**；
5. **Grouping comparison heatmap**；
6. **Noise rate curves**；
7. **Cost breakdown stacked bar**。

### 8.6 成功标准

Fashion noise medium / high：

$$
\text{clean acc gain vs geometry}>1.0\%
$$

$$
\text{ECE red vs AdamW}>50\%
$$

$$
\text{time ratio}\leq1.3
$$

KMNIST noise：

$$
\text{clean acc gain vs geometry}>0.8\%
$$

$$
\text{ECE red vs AdamW}>25\%
$$

$$
\text{time ratio}\leq1.4
$$

如果 accuracy gain 过线但 time ratio $>1.5$，判为 `snr_cost_failure`，只保留 stress。

---

## 9. V4-4：Profile stress validation

### 9.1 目标

在 clean 和 noisy profile 之外，测试 small-data、calibration、corruption 是否有副作用。

### 9.2 方法

比较：

```text
AdamW
current geometry-aware
FGO-v4-clean candidate
FGO-v4-noise candidate
```

### 9.3 Stress 设置

Small data：

```text
train_size = 500, 1000, 2000, 6000
```

Corruption：

```text
Gaussian noise severity 1..5
blur severity 1..5
rotation severity 1..5
translation severity 1..5
contrast severity 1..5
cutout severity 1..3
```

Calibration：

```text
ECE
NLL
Brier score
confidence histogram
```

### 9.4 成功标准

FGO-v4-clean 不能明显损害 small-data / calibration：

$$
\text{small-data acc gap vs geometry}<0.5\%
$$

$$
\text{ECE not worse than geometry by } >5\%
$$

Corruption 只要求不明显恶化：

$$
\text{corruption mean acc gap vs AdamW}<1\%
$$

如果 clean candidate 在 KMNIST 上更好但 Fashion small-data 明显变差，则它只能作为 KMNIST hard-task profile，不能成为 universal default。

---

## 10. V4-5：CIFAR readiness and final method selection

### 10.1 目标

决定是否进入 CIFAR-10 small ConvStem-DGKAN，以及带哪些方法进入。

### 10.2 选择规则

Clean default selection：

```text
if FGO-v4-clean medium pass on KMNIST and not worse than Fashion geometry:
    clean default = FGO-v4-clean
else:
    clean default = current geometry-aware
```

Hard-task candidate：

```text
best method with:
  KMNIST gap < 0.8%
  AUC red > 10%
  phi' red > 25%
  J red > 10%
```

Noise candidate：

```text
best method with:
  noisy-label clean acc gain > 1%
  ECE red > 25%
  time ratio <= 1.4
```

CIFAR readiness：

```text
Fashion clean ok
KMNIST hard-task ok
No catastrophic corruption loss
Step time ratio <= 1.3 for clean default
```

### 10.3 CIFAR small 初始方法包

如果 readiness 通过，进入 CIFAR-10 small 的方法包为：

```text
AdamW
current geometry-aware
FGO-v4-clean candidate
FGO-v4-noise candidate only for noisy-label CIFAR ablation
```

不进入 CIFAR clean 默认的方法：

```text
permanent no-grid
naive Adam-style functional optimizer
full edge SNR if time ratio > 1.5
FSAM unless explicitly noise/cost ablation
```

---

## 11. W&B Dashboard 设计

### Page 1：FGO-v4 scorecard

展示每个 package 的 pass/fail：

```text
V4-0 reproducibility
V4-1 clean data-pulse
V4-2 ablation explanation
V4-3 noise cost optimization
V4-4 stress validation
V4-5 readiness
```

每个 method 的主表字段：

```text
test_acc
gap_vs_adamw
gain_vs_geometry
val_auc_red
phi_red
J_red
branch_over_adamw
ECE_red
step_time_ratio
failure_type
```

---

### Page 2：Clean accuracy-geometry Pareto

图：

```text
x = gap_vs_adamw
y = J_reduction
color = method profile
size = phi_reduction
shape = dataset
```

目标：看 clean profile 是否真的接近 no-grid accuracy，同时保留 geometry。

---

### Page 3：Metric curriculum dynamics

曲线：

```text
metric/data_lambda_current over epoch
metric/mixed_condition over epoch
metric/data_grid_alignment over epoch
branch/branch_over_adamw over epoch
geometry/max_jac_condition over epoch
```

对比：

```text
no-grid permanent
trace010
FGO-v4 data-pulse
current geometry-aware
```

---

### Page 4：Momentum and trust diagnostics

图：

```text
optimizer/trust_clip_rate over epoch
optimizer/descent_ratio_median over epoch
optimizer/bad_step_rate over epoch
momentum/direction_cos_current over epoch
momentum/reset_count
```

重点判断：trust 是否只是安全阀，还是过度裁剪。

---

### Page 5：Noisy-label profile

图：

```text
clean_test_acc vs noise_rate
noise_memorization_rate vs noise_rate
ECE vs noise_rate
time_ratio vs noise_rate
```

另做 Pareto：

```text
x = time_ratio_vs_geometry
y = clean_acc_gain_vs_geometry
color = SNR grouping
size = ECE reduction
```

---

### Page 6：SNR cost audit

图：

```text
snr_update_interval vs clean_acc
snr_grouping vs time_ratio heatmap
snr_sample_frac vs ECE_reduction
snr_staleness_steps vs acc_gain
```

---

### Page 7：Stress validation

图：

```text
small_data_curve
corruption_mean_acc_bar
calibration_reliability_diagram
train_test_gap_curve
```

---

### Page 8：Failure table

Failure types：

```text
branch_underactive
branch_overactive
geometry_spike
phi_prime_spike
data_metric_geometry_failure
reanchor_too_strong
momentum_mismatch
trust_too_conservative
snr_cost_failure
snr_stale_failure
noise_memorization_not_reduced
corruption_regression
cifar_not_ready
```

定义：

```text
branch_underactive:
  branch / AdamW < 0.5 and acc gap > 1%

branch_overactive:
  branch / AdamW > 1.2 or phi' red < 0

data_metric_geometry_failure:
  acc improves but J red < 5% or phi' red < 10%

reanchor_too_strong:
  geometry improves but train acc / branch drops significantly

trust_too_conservative:
  clip rate > 0.2 and branch_underactive

snr_cost_failure:
  noisy acc gain positive but time ratio > 1.5
```

---

## 12. 第一轮最小执行包

如果资源有限，先执行以下最小包。

### 12.1 V4-mini-clean

```text
dataset = KMNIST
methods:
  AdamW
  current geometry-aware
  no-grid stress
  FGO-v3-clean full
  FGO-v4 data-pulse configs
seeds search = 0,1,2
seeds confirm = 0,1,2,3,4
```

搜索：

```text
lambda_early = 0.5, 1.0
lambda_final = 0.0, 0.05
pulse_frac = 0.15, 0.25
momentum_mu = 0.0, 0.8
prox_lambda = 0, 3e-5
trust_radius = 0.15
```

目标：验证是否能把 no-grid 的 $0.36\%$ gap 信号转成 safe profile。

---

### 12.2 V4-mini-noise

```text
dataset = Fashion-MNIST
noise = 0.2, 0.4
methods:
  AdamW
  geometry-aware
  SNR previous best
  SNR-lite output/layer grouping
  SNR-lite edge interval 4/8 sample 0.5
```

目标：找到 time ratio $\leq1.3$ 且保留 $80\%$ gain 的 SNR-lite。

---

### 12.3 V4-mini-stress

使用 V4-mini-clean top-2 和 V4-mini-noise top-2，在 Fashion/KMNIST 做：

```text
small train size = 1000, 6000
corruption mean acc
ECE
```

---

## 13. 决策规则

### 13.1 Clean default 决策

如果 FGO-v4-clean 在 KMNIST 满足：

$$
\text{gap vs AdamW}<0.5\%
$$

$$
\text{AUC red}>10\%
$$

$$
\phi'\text{ red}>20\%
$$

$$
J\text{ red}>10\%
$$

且 Fashion 不低于 current geometry-aware 超过 $0.3\%$，则：

```text
clean default = FGO-v4-clean
```

否则：

```text
clean default = current geometry-aware
hard-task candidate = best FGO-v4-clean
```

---

### 13.2 Noise default 决策

如果 SNR-lite 满足：

$$
\text{clean acc gain vs geometry}>1\%
$$

$$
\text{time ratio}<1.3\text{ to }1.4
$$

$$
\text{ECE red vs AdamW}>25\%
$$

则：

```text
noise default = SNR-lite profile
```

否则：

```text
noise default = geometry-aware
SNR = stress-only ablation
```

---

### 13.3 CIFAR readiness 决策

进入 CIFAR-10 small clean 实验的条件：

```text
Fashion clean strong or unchanged
KMNIST hard-task gap < 0.8%
step time ratio <= 1.3 for clean default
no catastrophic corruption regression
```

如果不满足，继续 clean/noise profile，而不是贸然进入 Stage I。

---

## 14. 预期结论模板

### 若 FGO-v4-clean 成功

可以写：

> Early data-local functional metric provides task-adaptive descent, while late grid/Sobolev re-anchoring restores derivative and Jacobian control. This closes the KMNIST clean accuracy gap without sacrificing DG-KAN's geometry advantage.

中文：

> 早期 data-local metric 帮助 hard task 激活 KAN branch，后期 grid/Sobolev re-anchor 恢复函数复杂度控制，从而在不牺牲几何优势的情况下缩小 KMNIST clean gap。

---

### 若 FGO-v4-clean 失败但 noise 成功

可以写：

> Data-pulse clean optimization remains insufficient to replace the current geometry-aware default, but SNR-adaptive functional updates provide a robust noisy-label profile with improved clean accuracy and calibration.

中文：

> FGO-v4-clean 仍不足以替代 current geometry-aware，但 SNR-adaptive functional update 已经形成明确的 noisy-label profile，可用于 label-noise 和 calibration 场景。

---

### 若 SNR-lite 成本仍失败

可以写：

> Edge-level SNR is effective but currently too expensive. It should remain a stress-time or noisy-label-specific method until grouped/fused SNR kernels are implemented.

中文：

> Edge-level SNR 有效果，但工程成本仍太高。在 grouped/fused SNR 实现之前，它只能作为 noisy-label stress 方法，而不是默认优化器。

---

## 15. 最终建议

FGO-v3 的结果已经说明：继续堆大一统 optimizer 没有意义。下一步应明确拆成两个 profile：

$$
\boxed{
\text{clean profile: data-pulse + grid re-anchor + momentum}
}
$$

$$
\boxed{
\text{noise profile: cost-aware SNR + late spectral/prox}
}
$$

clean profile 的目标是解决 KMNIST 剩余 gap，noise profile 的目标是把 label-noise 收益做成低成本可用方法。两者不能混成一个默认 optimizer。

目前最稳决策仍然是：

```text
Stage I clean default = current geometry-aware functional update
Hard-task ablation = FGO-v3/FGO-v4 clean candidate
Noisy-label ablation = SNR profile
```

只有当 FGO-v4-clean 真的把 KMNIST gap 压到 $0.5\%$ 内，并保持 $\phi'$ / Jacobian 优势，才允许替代 current geometry-aware 进入 CIFAR clean default。

