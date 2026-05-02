# DG-KAN 下一轮实验计划：GA-FU 统一 Optimizer 收敛验证

> 文件名：`DG-KAN_GA-FU_统一Optimizer下一轮实验计划.md`  
> 目标：停止继续堆叠新 optimizer 组件，固定并验证一个统一主 optimizer：**GA-FU, Geometry-Aware Functional Update**。  
> 公式格式：Typora 友好，统一使用 `$...$` 与 `$$...$$`。

---

## 0. 本轮实验为什么要这样设计

过去几轮 GFU / FGO 实验已经说明一个很清楚的问题：我们不是缺少新组件，而是缺少一个被固定、被命名、被严格复核的主 optimizer。继续往 functional update 里叠加 data metric、SNR、FSAM、prox、target solve、controller、trust 等组件，会让实验空间越来越大，但核心问题反而不清楚。

现在最关键的问题是：

$$
\boxed{
\text{如何让 KAN branch 足够参与任务，同时不让 } \phi',\ \kappa(J),\ \text{credit amplification 失控。}
}
$$

这就是 **effective functional step control problem**。

如果 KAN branch 太弱：

$$
\text{branch under-active}
\Rightarrow
\text{accuracy 不够，no-KAN ablation drop 很小。}
$$

如果 KAN branch 太强：

$$
\text{branch over-active}
\Rightarrow
\phi'_{p95}\uparrow,
\quad
\kappa(J)\uparrow,
\quad
\text{credit amplification}\uparrow.
$$

因此，本轮实验不再继续验证大量新技术，而是把当前最稳的 **geometry-aware functional update** 固定为统一 optimizer，命名为 **GA-FU**，并系统回答：

> GA-FU 能否成为 DG-KAN 在 clean vision classification 上的默认 optimizer？

这里的“默认”不是指每个指标都必须超过 AdamW，而是指它在 accuracy、convergence、geometry、branch utilization 和 calibration 上形成稳定的 Pareto 优势。

---

## 1. 本轮实验的核心命题

本轮只验证一个主命题：

> **GA-FU 是否能在 Fashion-MNIST 和 KMNIST 上稳定达到 AdamW 级别 accuracy，同时获得更好的 validation loss AUC、更低 $\phi'_{p95}$、更低 Jacobian condition、更稳定 branch utilization 和更好的 calibration。**

形式化地，GA-FU 的目标不是简单最大化 accuracy，而是优化以下多目标：

$$
\max \operatorname{Acc}_{test}
$$

$$
\min \operatorname{AUC}_{val\ loss}
$$

$$
\min \phi'_{p95}
$$

$$
\min \kappa(J)
$$

$$
\text{keep } r_{branch}/r_{branch}^{AdamW}\in[0.5,0.9]
$$

$$
\min \operatorname{ECE}
$$

其中 branch ratio 定义为：

$$
r_{branch,k}
=
\frac{
\|\alpha_k b_{k,t}\operatorname{KAN}_k(\operatorname{LN}(h_k))\|
}{
\|h_k\|+\epsilon
}.
$$

相对 AdamW 的 branch utilization 为：

$$
\operatorname{branch/AdamW}
=
\frac{r_{branch}^{GA-FU}}{r_{branch}^{AdamW}+\epsilon}.
$$

---

## 2. 统一 optimizer：GA-FU 的固定定义

### 2.1 总体定义

本轮把 **GA-FU** 定义为：

$$
\boxed{
\text{GA-FU}
=
\text{early branch activation}
+
\text{diag-to-Sobolev functional update}
+
\text{online geometry trigger}
+
\text{late branch scaling}
}
$$

GA-FU 不包含以下组件作为默认项：

```text
data metric
SNR
FSAM
prox / spectral smoothing
target solve
hard branch target controller
naive Adam moment
```

这些组件只能作为后续 profile 或 ablation，不进入本轮主 optimizer。

---

### 2.2 非 KAN 参数更新

非 KAN 参数继续使用 AdamW：

$$
\theta_{nonKAN,t+1}
=
\operatorname{AdamW}(\theta_{nonKAN,t},\nabla_{\theta_{nonKAN}}L).
$$

这里包括：

```text
stem
head
LayerNorm parameters
alpha parameters
bias parameters not in KAN edge coefficients
```

---

### 2.3 KAN coefficient functional update

KAN edge function 为：

$$
\phi_{ji}(t)
=
\sum_{m=1}^{M} a_{jim}B_m(t).
$$

普通 coefficient update 是：

$$
a\leftarrow a-\eta\nabla_a L.
$$

GA-FU 使用函数空间预条件更新：

$$
a\leftarrow a-\eta P_t \nabla_a L.
$$

早期 active phase 使用 diagonal functional preconditioner：

$$
P_t=\operatorname{diag}(M+\rho I)^{-1}.
$$

后期 geometry phase 使用 Sobolev preconditioner：

$$
P_t=(M+\rho I)^{-1}.
$$

Sobolev metric 为：

$$
M
=
\int B(t)B(t)^\top dt
+
\alpha_s\int B'(t)B'(t)^\top dt
+
\beta_s\int B''(t)B''(t)^\top dt.
$$

---

### 2.4 Early branch activation

在 active phase 中，forward block 使用 boosted branch scale：

$$
h_{k+1}
=
h_k
+
\alpha_k b_{boost}\operatorname{KAN}_k(\operatorname{LN}(h_k)).
$$

同时 KAN coefficient learning rate 乘以：

$$
c_{boost}>1.
$$

因此 early phase 的 KAN coefficient 更新为：

$$
a_{t+1}
=
a_t
-
\eta_0 c_{boost}
\operatorname{diag}(M+\rho I)^{-1}\nabla_a L.
$$

目的是避免静态 functional update 的 branch under-active 问题。

---

### 2.5 Online geometry trigger

GA-FU 不使用固定切换时间作为唯一依据，而是根据几何信号切换。

触发条件包括：

$$
\phi'_{p95}>\tau_{\phi}
$$

或：

$$
\kappa(J)>\tau_J
$$

或：

$$
r_{branch}>\tau_b
$$

或达到最大 active fraction：

$$
t>T_{max\ active}.
$$

触发后进入 Sobolev phase：

$$
a_{t+1}
=
a_t
-
\eta_0 c_{final}
(M+\rho I)^{-1}\nabla_a L.
$$

branch scale 进入 late final scale：

$$
b_t\rightarrow b_{final}.
$$

如果触发后几何仍然超阈值，则允许 shrink：

$$
b_t\leftarrow \max(b_{min},\gamma_{shrink}b_t).
$$

---

## 3. 本轮固定配置

### 3.1 Fashion-MNIST 默认 GA-FU

Fashion-MNIST 当前已有 strong signal，因此本轮不再搜索大范围参数，只做稳定性复核。

```text
dataset = Fashion-MNIST
optimizer = GA-FU
branch_boost = 2.0
coeff_lr_boost = 1.5
branch_max_active_frac = 0.50
branch_final_scale = 0.8
phi_high = 0.052
jac_high = 4.5
branch_high = 0.30
coeff_lr_base = 0.3
rest_lr = 0.003
```

这个配置的目标是：

$$
\text{test acc}\geq\text{AdamW}
$$

$$
\text{val AUC improvement}>10\%
$$

$$
\phi'_{p95}\text{ reduction}>20\%
$$

$$
\kappa(J)\text{ reduction}>30\%.
$$

---

### 3.2 KMNIST 默认 GA-FU

KMNIST 更难，不能使用过强 branch boost。

```text
dataset = KMNIST
optimizer = GA-FU
branch_boost = 1.5
coeff_lr_boost = 1.2
branch_max_active_frac = 0.35
branch_final_scale = 0.9
phi_high = 0.055
jac_high = 4.5
branch_high = 0.30
coeff_lr_base = 0.3 或 0.4
rest_lr = 0.003
```

本轮保留两个 KMNIST 版本：

```text
GA-FU-KM-acc:
  branch_final_scale = 1.0
  目标是更高 accuracy

GA-FU-KM-balanced:
  branch_final_scale = 0.9
  目标是 accuracy / geometry 折中
```

但默认报告主结果使用：

```text
GA-FU-KM-balanced
```

---

## 4. 实验阶段总览

本轮实验分为五个阶段。

```text
P0：配置冻结与 smoke
P1：clean default 5/10-seed confirm
P2：target-matched convergence analysis
P3：branch / geometry mechanism audit
P4：fixed-GA-FU generalization sanity
P5：profile ablation only if P1-P4 passed
```

其中 P5 不是新技术主线，只用于确认是否需要 hard-task momentum / noise SNR profile。

---

## 5. P0：配置冻结与 smoke

### 5.1 目标

P0 只验证配置、日志和指标是否正常，不做结论。

### 5.2 数据集

```text
Fashion-MNIST smoke
KMNIST smoke
train / val / test = 512 / 128 / 128
epochs = 1
seeds = 0
```

### 5.3 方法

```text
AdamW
Static functional
GA-FU
```

### 5.4 必须检查

```text
NaN / Inf count = 0
run_failed = 0
schedule/geometry_switch_step exists
schedule/geometry_switch_reason exists
branch/global/mean_output_norm_ratio finite
kan/phi_prime_p95 finite
jacobian/max_condition finite
```

### 5.5 P0 通过条件

$$
\text{run failures}=0
$$

$$
\text{all key metrics finite}
$$

$$
\text{GA-FU test acc gap vs static functional}<5\%
$$

---

## 6. P1：Clean default confirm

### 6.1 目标

P1 是本轮核心实验。它验证 GA-FU 是否能作为 clean default optimizer。

### 6.2 数据集

```text
Fashion-MNIST
KMNIST
optional sanity: MNIST
```

主报告只使用 Fashion-MNIST 和 KMNIST。

### 6.3 模型

```text
model = residual DG-KAN
depth = 4
hidden_dim = 64
basis_count = 16
alpha_init = 1.5
train / val / test = 6000 / 1000 / 1000
batch_size = 256
eval_batch_size = 512
audit_batch_size = 256
```

### 6.4 训练设置

```text
epochs = 20
seeds = 0,1,2,3,4
optional stronger confirm: seeds = 0..9
```

如果资源允许，建议：

```text
first pass: seeds 0..4
confirm pass: seeds 0..9 only for final selected methods
```

### 6.5 方法对照

P1 只比较这些方法，不再加入 SNR / FSAM / data metric / prox。

| method | 说明 |
|---|---|
| `AdamW` | DG-KAN 全参数 AdamW，强 accuracy baseline |
| `StaticFunctional` | diag-to-Sobolev，无 branch schedule |
| `GA-FU` | 本轮主 optimizer |
| `DGKAN-AdamW` | 和 AdamW 相同，可合并命名；保留便于日志一致 |

如果已有 FullBP / AnalyticAdj 区别，本轮默认使用：

```text
credit = AnalyticAdj
```

除非做对照，否则不再单独比较 FullBP / FirstOrder。

---

### 6.6 P1 记录指标

#### 任务性能

```text
task/train_loss
task/val_loss
task/test_loss
task/train_acc
task/val_acc
task/test_acc
task/best_val_acc
task/best_epoch
task/final_epoch
compare/test_acc_gap_vs_adamw
compare/test_loss_gap_vs_adamw
```

定义：

$$
\text{test acc gap vs AdamW}
=
\operatorname{Acc}_{AdamW}-\operatorname{Acc}_{method}.
$$

正数表示 method 低于 AdamW，负数表示 method 高于 AdamW。

---

#### 收敛指标

```text
conv/train_loss_auc
conv/val_loss_auc
conv/val_acc_auc
conv/steps_to_adamw_final_val_loss
conv/steps_to_adamw_final_val_acc
conv/time_to_adamw_final_val_loss_sec
conv/time_to_adamw_final_val_acc_sec
conv/loss_at_10pct_steps
conv/loss_at_25pct_steps
conv/loss_at_50pct_steps
conv/loss_at_75pct_steps
conv/val_auc_improvement_vs_adamw
```

AUC improvement 定义为：

$$
\operatorname{AUCImprove}
=
\frac{
\operatorname{AUC}_{AdamW}-\operatorname{AUC}_{method}
}{
\operatorname{AUC}_{AdamW}+\epsilon
}.
$$

---

#### Wall-clock 指标

```text
compute/step_time_ms
compute/epoch_time_sec
compute/total_train_time_sec
compute/time_to_target_val_loss_sec
compute/time_to_target_val_acc_sec
compute/throughput_samples_per_sec
compute/time_ratio_vs_adamw
```

因为 functional update 的单步可能比 AdamW 慢，所以本轮必须同时报告 step-matched 和 wall-clock-matched 结果。

---

#### KAN branch 指标

```text
branch/global/mean_output_norm_ratio
branch/global/branch_over_adamw
branch/layer_{k}/output_norm_ratio
branch/layer_{k}/branch_scale
branch/layer_{k}/effective_alpha_mean
ablation/no_kan_test_acc
ablation/no_kan_acc_drop
ablation/no_kan_drop_over_adamw
ablation/kan_logit_delta_norm
ablation/kan_margin_contribution
```

No-KAN drop：

$$
\Delta_{noKAN}
=
\operatorname{Acc}_{full}-\operatorname{Acc}_{noKAN}.
$$

如果：

$$
\Delta_{noKAN}\approx0
$$

说明 KAN branch 没有参与任务。

---

#### Geometry 指标

```text
kan/phi_prime_p95
kan/phi_prime_max
kan/curvature_mean
kan/curvature_p95
kan/sobolev_norm_mean
kan/sobolev_norm_max
jacobian/condition_mean
jacobian/condition_max
credit/amplification_mean
credit/amplification_p95
credit/noise_gain_p95
```

需要记录相对 AdamW reduction：

$$
\operatorname{PhiRed}
=
\frac{\phi'_{AdamW}-\phi'_{method}}{\phi'_{AdamW}+\epsilon}.
$$

$$
\operatorname{JRed}
=
\frac{\kappa(J)_{AdamW}-\kappa(J)_{method}}{\kappa(J)_{AdamW}+\epsilon}.
$$

---

#### Schedule 指标

```text
schedule/phase_id
schedule/branch_scale
schedule/coeff_lr_multiplier
schedule/effective_coeff_lr
schedule/geometry_switch_step
schedule/geometry_switch_epoch
schedule/geometry_switch_reason
schedule/geometry_current_branch_scale
schedule/geometry_shrink_events
schedule/geometry_last_phi_p95
schedule/geometry_last_jac_condition
schedule/geometry_last_branch_proxy
schedule/active_phase_fraction_actual
```

这些指标用于判断 GA-FU 是否按预期运行。

---

#### Calibration / generalized performance

```text
calibration/ece
calibration/nll
calibration/confidence_correct
calibration/confidence_wrong
generalization/train_test_gap_acc
generalization/train_test_gap_loss
generalization/val_test_gap_loss
```

---

### 6.7 P1 可视化

P1 的 W&B dashboard 至少包含以下页面。

#### Page 1：Clean default scorecard

表格列：

```text
method
dataset
test_acc_mean
test_acc_std
acc_gap_vs_adamw
val_auc_improvement
phi_prime_reduction
J_reduction
branch_over_adamw
ECE_reduction
pass_status
```

#### Page 2：Loss / accuracy curves

图：

```text
train_loss vs epoch
val_loss vs epoch
val_acc vs epoch
test_acc vs epoch if eval_test_every_epoch enabled
```

每张图按 method 分组，并显示 seed mean ± std。

#### Page 3：Target-matched convergence

图：

```text
steps_to_adamw_final_val_loss bar
time_to_adamw_final_val_loss bar
steps_to_adamw_final_val_acc bar
time_to_adamw_final_val_acc bar
```

#### Page 4：Branch utilization

图：

```text
branch_over_adamw over epoch
no_kan_acc_drop bar
branch_ratio vs test_acc scatter
branch_ratio vs phi_prime_p95 scatter
```

#### Page 5：Geometry safety

图：

```text
phi_prime_p95 over epoch
jacobian_condition over epoch
credit_amplification_p95 over epoch
curvature_mean over epoch
```

#### Page 6：Schedule behavior

图：

```text
branch_scale over epoch
coeff_lr_multiplier over epoch
switch_step histogram
switch_reason pie chart
shrink_events bar
```

#### Page 7：Pareto plot

至少包括：

```text
x = val_loss_auc, y = test_acc, color = method, size = J_reduction
x = step_time_ms, y = test_acc, color = method, size = memory
x = phi_prime_p95, y = test_acc, color = method
```

---

### 6.8 P1 成功标准

#### Weak pass

GA-FU 在 Fashion 和 KMNIST 上均满足：

$$
\text{test acc gap vs AdamW}<1.5\%
$$

$$
\text{val AUC improvement}>5\%
$$

$$
\phi'_{p95}\text{ reduction}>20\%
$$

$$
\kappa(J)\text{ reduction}>10\%
$$

并且：

$$
0.45<\text{branch/AdamW}<0.95.
$$

---

#### Medium pass

GA-FU 满足：

Fashion：

$$
\text{test acc}_{GA-FU}\geq\text{test acc}_{AdamW}
$$

KMNIST：

$$
\text{test acc gap vs AdamW}<1.0\%
$$

同时两个数据集均满足：

$$
\text{val AUC improvement}>8\%
$$

$$
\phi'_{p95}\text{ reduction}>25\%
$$

$$
\kappa(J)\text{ reduction}>15\%.
$$

---

#### Strong pass

GA-FU 在 Fashion 和 KMNIST 上均满足：

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
\kappa(J)\text{ reduction}>20\%
$$

$$
\text{ECE reduction}>10\%.
$$

并且 time-to-target 不劣于 AdamW：

$$
\text{time-to-target}_{GA-FU}
\leq
1.1\times
\text{time-to-target}_{AdamW}.
$$

---

## 7. P2：Target-matched convergence analysis

### 7.1 目标

P2 不再问最终 accuracy，而是问：

> GA-FU 是否真正更快达到同等目标？

### 7.2 Target 定义

对每个 dataset 和 seed，定义 AdamW 的目标：

$$
L^*_{AdamW}=L_{val,AdamW}^{final}.
$$

$$
A^*_{AdamW}=A_{val,AdamW}^{final}.
$$

同时定义 relaxed target：

$$
L^{relaxed}=1.05L^*_{AdamW}.
$$

$$
A^{relaxed}=A^*_{AdamW}-0.005.
$$

记录 GA-FU 达到这些目标所需的 step 和 wall-clock。

### 7.3 指标

```text
target/steps_to_adamw_final_val_loss
target/time_to_adamw_final_val_loss
target/steps_to_relaxed_val_loss
target/time_to_relaxed_val_loss
target/steps_to_adamw_final_val_acc
target/time_to_adamw_final_val_acc
target/reached_final_loss_bool
target/reached_final_acc_bool
```

### 7.4 可视化

```text
Kaplan-style target reach curve
bar: mean steps_to_target by method
bar: mean time_to_target by method
scatter: val_auc vs time_to_target
```

### 7.5 判定

如果 GA-FU 的 AUC 更好，但 time-to-target 更差，则不能 claim fast convergence，只能 claim smoother / better trajectory。

Fast convergence claim 成立需要：

$$
\text{time-to-target}_{GA-FU}
<
\text{time-to-target}_{AdamW}
$$

或至少：

$$
\text{steps-to-target}_{GA-FU}
<
\text{steps-to-target}_{AdamW}
$$

且 step-time overhead 不超过：

$$
1.2\times.
$$

---

## 8. P3：Branch / geometry mechanism audit

### 8.1 目标

P3 解释 GA-FU 是否真正解决了 effective functional step control problem。

### 8.2 关键分析

#### Branch activation trajectory

观察：

$$
\text{branch/AdamW}(t)
$$

是否从 early phase 的低值逐渐进入合理区间：

$$
0.5\leq \text{branch/AdamW}\leq0.9.
$$

#### Geometry trajectory

观察：

$$
\phi'_{p95}(t)
$$

$$
\kappa(J_t)
$$

是否在 active phase 后被收住。

#### No-KAN contribution

记录：

$$
\Delta_{noKAN}(t)
=
\operatorname{Acc}_{full}(t)-\operatorname{Acc}_{noKAN}(t).
$$

如果 branch ratio 提高，但 no-KAN drop 不提高，说明 branch 变大但没有任务贡献。

### 8.3 指标

```text
mechanism/branch_activation_auc
mechanism/branch_over_adamw_final
mechanism/no_kan_drop_final
mechanism/no_kan_drop_over_adamw
mechanism/phi_prime_peak_epoch
mechanism/jacobian_peak_epoch
mechanism/geometry_recovery_after_switch
mechanism/switch_to_final_acc_gain
```

### 8.4 可视化

```text
branch_over_adamw vs epoch
no_kan_drop vs epoch
phi_prime_p95 vs epoch
jacobian_condition vs epoch
vertical line: switch epoch
```

### 8.5 判定

GA-FU 机制成功要求：

$$
\text{branch/AdamW}_{final}>0.5
$$

$$
\Delta_{noKAN}^{GA-FU}>0.7\Delta_{noKAN}^{AdamW}
$$

$$
\phi'_{p95}^{GA-FU}<0.8\phi'_{p95}^{AdamW}
$$

$$
\kappa(J)^{GA-FU}<0.85\kappa(J)^{AdamW}
$$

至少在 Fashion 上必须满足；KMNIST 可作为 weaker pass。

---

## 9. P4：Fixed-GA-FU generalization sanity

### 9.1 目标

P4 不引入新 optimizer，只用固定 GA-FU 复查它是否保留 Stage G 的泛化 / calibration 优势。

### 9.2 实验设置

#### Small-data

```text
dataset = Fashion-MNIST, KMNIST
train_size = 500, 1000, 2000, 6000
methods = AdamW, StaticFunctional, GA-FU
seeds = 0,1,2,3,4
epochs = 20
```

#### Label-noise sanity

这里只做 sanity，不做 SNR。

```text
dataset = Fashion-MNIST, KMNIST
noise = 0.2, 0.4
methods = AdamW, GA-FU
seeds = 0,1,2,3,4
epochs = 20
```

### 9.3 指标

```text
generalization/train_test_gap_loss
generalization/train_test_gap_acc
generalization/ece
noise/clean_test_acc
noise/noisy_train_acc_against_noisy_labels
noise/noisy_train_acc_against_clean_labels
noise/noise_memorization_rate
noise/clean_recovery_rate
kan/curvature_mean
kan/phi_prime_p95
jacobian/condition_max
```

### 9.4 可视化

```text
small-data acc curve: train_size vs test_acc
small-data ECE curve
label-noise clean acc bar
label-noise memorization bar
curvature vs train-test gap scatter
phi_prime_p95 vs ECE scatter
```

### 9.5 判定

P4 不是为了证明新 SOTA，而是确保固定 GA-FU 没有失去 Stage G 的优势。

GA-FU generalization sanity pass：

Fashion：

$$
\text{test acc}_{GA-FU}\geq \text{test acc}_{AdamW}-0.5\%
$$

KMNIST：

$$
\text{test acc gap}<1.5\%
$$

同时：

$$
\operatorname{ECE}_{GA-FU}<\operatorname{ECE}_{AdamW}
$$

$$
\text{train-test gap}_{loss,GA-FU}<\text{train-test gap}_{loss,AdamW}.
$$

---

## 10. P5：Profile ablation，只在主线通过后运行

### 10.1 原则

P5 不属于主实验。只有 P1-P4 表明 GA-FU 已经稳定后，才允许做 profile ablation。

本轮最多保留两个 profile：

```text
GA-FU-M:
  GA-FU + functional-direction momentum
  只用于 KMNIST hard-task ablation

GA-FU-SNR:
  GA-FU + SNR-adaptive update
  只用于 label-noise ablation
```

不再运行：

```text
FSAM
prox/spectral clean default
data metric no-grid
target solve
hard branch target controller
naive Adam moment
```

### 10.2 GA-FU-M

目的：检查 momentum 是否能缩小 KMNIST 剩余 gap。

```text
dataset = KMNIST
methods = AdamW, GA-FU, GA-FU-M
mu = 0.8
epochs = 20
seeds = 0..4
```

通过条件：

$$
\text{test acc gain vs GA-FU}>0.3\%
$$

且：

$$
\phi'_{p95}\text{ reduction vs AdamW}>20\%
$$

$$
\kappa(J)\text{ reduction vs AdamW}>10\%.
$$

---

### 10.3 GA-FU-SNR

目的：检查 SNR profile 是否只在 noisy-label 中有价值。

```text
dataset = Fashion-MNIST
noise = 0.4
methods = AdamW, GA-FU, GA-FU-SNR
seeds = 0..4
epochs = 20
```

通过条件：

$$
\text{clean test acc gain vs GA-FU}>1\%
$$

$$
\text{time ratio vs GA-FU}<1.5
$$

并且：

$$
\text{ECE not worse than GA-FU by more than }5\%.
$$

---

## 11. Failure table

本轮必须保留结构化 failure table。每个 run 都要根据以下规则打标签。

### 11.1 Failure types

| failure type | 触发条件 | 解释 |
|---|---|---|
| `accuracy_gap_too_large` | test acc gap vs AdamW > threshold | 精度不够 |
| `no_convergence_gain` | val AUC improvement <= 0 | 没有收敛优势 |
| `wall_clock_no_gain` | time-to-target 不优于 AdamW | 实际时间无优势 |
| `branch_underactive` | branch/AdamW < 0.5 and no-KAN drop low | KAN branch 没用起来 |
| `branch_overactive` | branch/AdamW > 1.1 | branch 过强 |
| `geometry_not_preserved` | phi/J reduction 不达标 | 几何优势消失 |
| `phi_prime_spike` | phi_prime_p95 > 1.2 AdamW | derivative 失控 |
| `jacobian_spike` | max J cond > 1.2 AdamW | Jacobian 失控 |
| `switch_too_early` | switch epoch < 10% training | active phase 太短 |
| `switch_too_late` | switch epoch > max_active_frac + tolerance | Sobolev phase 太晚 |
| `no_kan_no_contribution` | no-KAN drop < 0.5 AdamW | KAN branch 任务贡献不足 |
| `calibration_worse` | ECE > AdamW | 校准变差 |
| `seed_unstable` | std test acc > 0.8% | seed 方差太大 |

### 11.2 Failure aggregation

输出：

```text
failure_table.csv
failure_summary_by_method.csv
failure_summary_by_dataset.csv
```

必须可视化：

```text
failure type counts by method
failure type counts by dataset
accuracy_gap_too_large vs branch_underactive scatter
geometry_not_preserved vs branch_overactive scatter
```

---

## 12. 统计与报告规范

### 12.1 Seed aggregation

所有正式结论必须使用：

```text
mean ± std over seeds
```

并记录：

```text
95% bootstrap confidence interval
```

建议对关键比较做 paired seed difference：

$$
\Delta_s
=
Metric_{GA-FU,s}-Metric_{AdamW,s}.
$$

报告：

$$
\bar\Delta,
\quad
\operatorname{std}(\Delta),
\quad
\operatorname{CI}_{95\%}(\Delta).
$$

---

### 12.2 不允许的 claim

如果 P1 没有同时通过 accuracy 和 AUC gate，不允许 claim：

```text
GA-FU converges faster than AdamW.
```

如果 P3 没有通过 branch/no-KAN gate，不允许 claim：

```text
GA-FU effectively uses KAN branch.
```

如果 P4 没有通过 matched generalization / ECE gate，不允许 claim：

```text
GA-FU improves generalization.
```

如果 P5 没有运行，不允许 claim：

```text
momentum / SNR is part of the optimizer.
```

---

## 13. 最终决策规则

### 13.1 GA-FU 成为 clean default

如果满足 medium pass，则：

```text
Stage I / CIFAR clean default = GA-FU
```

### 13.2 GA-FU 作为 Pareto optimizer

如果 GA-FU 满足：

$$
\text{test acc gap}<1.5\%
$$

$$
\text{val AUC improvement}>5\%
$$

$$
\text{geometry reductions significant}
$$

但没有达到 medium pass，则：

```text
GA-FU = geometry-stable Pareto optimizer
AdamW = accuracy baseline
```

### 13.3 GA-FU 失败

如果 GA-FU 在 KMNIST 上仍然：

$$
\text{test acc gap}>1.5\%
$$

或：

$$
\text{AUC improvement}\leq0
$$

则下一步不再继续调 trigger，而应该回到：

```text
model capacity
basis count
rest optimizer LR
training length / LR decay
```

而不是发明新 optimizer 组件。

---

## 14. 推荐执行命令结构

可以新增 runner：

```text
experiments/run_gafu_consolidation.py
```

或者在 `run_stage_f.py` 中新增：

```text
F7_gafu_consolidation
F7_gafu_target_matched
F7_gafu_mechanism
F7_gafu_generalization_sanity
F7_gafu_profile_ablation
```

建议命令：

```bash
python experiments/run_gafu_consolidation.py \
  --out-dir results/gafu_consolidation \
  --packages P0,P1,P2,P3,P4 \
  --datasets fashion_mnist,kmnist \
  --methods adamw,static_func,gafu \
  --seeds 0,1,2,3,4 \
  --epochs 20 \
  --train-size 6000 \
  --val-size 1000 \
  --test-size 1000 \
  --hidden-dim 64 \
  --depth 4 \
  --basis-count 16 \
  --batch-size 256 \
  --eval-batch-size 512 \
  --audit-batch-size 256 \
  --device cuda \
  --continue-on-error
```

如果资源允许，二次确认：

```bash
python experiments/run_gafu_consolidation.py \
  --out-dir results/gafu_consolidation_10seed \
  --packages P1,P2,P3 \
  --datasets fashion_mnist,kmnist \
  --methods adamw,gafu \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --epochs 20 \
  --device cuda \
  --continue-on-error
```

---

## 15. 本轮结束后应该产出的文件

```text
results/gafu_consolidation/
  gafu_runs.csv
  gafu_summary_by_method.csv
  gafu_summary_by_dataset.csv
  gafu_target_matched.csv
  gafu_branch_geometry_audit.csv
  gafu_generalization_sanity.csv
  gafu_failure_table.csv
  aggregate_summary.json
  runs/*/summary.json
```

`aggregate_summary.json` 至少包含：

```text
GA_FU_clean_default_pass
GA_FU_medium_pass
GA_FU_strong_pass
fashion_pass_level
kmnist_pass_level
recommended_stage_i_default
recommended_profile_ablation
main_failure_type
```

---

## 16. 本轮预期结论模板

如果结果好，结论写成：

> GA-FU, a geometry-aware functional update with early branch activation and late Sobolev control, provides a stable optimizer for DG-KAN. It matches or exceeds AdamW-level accuracy on Fashion-MNIST and approaches AdamW on KMNIST, while substantially reducing KAN derivative magnitude, Jacobian condition, and improving validation-loss AUC and calibration.

如果结果中等，结论写成：

> GA-FU is a reliable geometry-stable Pareto optimizer. It does not universally surpass AdamW accuracy, but it consistently improves validation-loss AUC and credit geometry while keeping accuracy within a small gap. It should be used as the default DG-KAN functional optimizer for geometry-sensitive experiments, with AdamW retained as the accuracy baseline.

如果结果不好，结论写成：

> GA-FU improves geometry but fails to provide a stable convergence or accuracy advantage. The remaining bottleneck is likely not trigger scheduling, but model capacity, rest-optimizer coupling, or the basis/branch parameterization. Further optimizer component proliferation is not recommended before those factors are isolated.

---

## 17. 一句话总结

本轮实验的目的不是再发明新 optimizer，而是回答：

$$
\boxed{
\text{GA-FU 是否可以作为 DG-KAN 的统一 clean optimizer？}
}
$$

如果答案是“是”，后续 Stage I / CIFAR clean scaling 就应该默认使用 GA-FU。  
如果答案是“部分是”，GA-FU 就是 geometry-stable Pareto optimizer，AdamW 继续作为 accuracy baseline。  
如果答案是“否”，下一步应停止 optimizer 扩展，转向模型容量、basis、rest optimizer 与训练长度的结构性分析。
