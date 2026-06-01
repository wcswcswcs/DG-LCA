# DG-KAN v13.7：Boundary-Conditioned Population-Risk Training + Basis-Cover Plasticity 计划

> 版本：v13.7 execution plan  
> 目标：基于 v13.6 `PopRiskSNR + BasisCoverBoundary` 结果、Generalization paper 的 signal/reservoir + population-risk SNR 观点，以及 Deep Manifold 的 boundary-conditioned iteration / moving node cover 观点，重置 functional update 线。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no active B-spline；no label-informed initialization；no teacher/distillation/loss modification/sampler/class weight/dataset-name branch；CEp99/NLL/ECE/LineC hard target 只能作为 audit/gate，不能生成方向；validation/test/future/query batch 不能参与 direction generation。  
> 核心改变：不再把 functional update 当成 one-shot perturbation，而是把它定义为 **loss-interface-generic 的 boundary-conditioned population-risk training process**。

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的总目标是构建一个可以替代或超越 MLP 的 strict FC-PureKAN 系统：

$$
\boxed{
\text{label-free strict FC-PureKAN substrate/base}
+
\text{loss-interface-generic functional update}
>
\text{same base + ordinary backprop / AdamW controls}
}
$$

最终 claim 必须同时满足：

```text
1. 表达力不打折，最好优于同规模 MLP；
2. forward / backward / step / memory 与 MLP 可比；
3. 收敛轨迹健康，AUC-step / AUC-time 不输；
4. 几何健康：真实信号进入 signal channel，噪声不泄漏进 signal channel，reservoir 不困住有用信号；
5. functional update 的收益必须击败 NoOp / RandomMatchedNorm / AdamWParallel / ShuffledGradient / SameActiveFraction controls；
6. functional 不是 CE-specific trick，而是通过统一 loss interface 接收当前训练边界条件。
```

本项目不是在 MNIST / Fashion-MNIST / KMNIST 上打榜。这些数据集只作为 failure slices，用于暴露训练几何、效率、tail 和 signal/reservoir 问题。

## 0.2 当前进展重新锚定

经过 v13.6，当前真实状态是：

```text
1. Historical B320/FHQ 曾证明 PureKAN 工程上可以做到很强效率/任务，但由于 label-informed init 已被禁止，不能作为 official claim。
2. Rational 仍是唯一稳定 active substrate family；Non-RAT family 仍不能进入 functional proof。
3. v13.4 证明 Rational 可执行 basis-channel movement：S1C 成立。
4. v13.5 证明 oracle DeltaZ upper bound route 不成立。
5. v13.6 证明 one-shot population-risk SNR mask / basis-channel SNR / cover-boundary 仍不成立。
6. MLP functional analog 仍没有 positive。
```

所以当前核心问题不是：

```text
有没有更多 O7/O8/O9 operator target；
有没有更复杂 response dictionary；
有没有更多 BN/BM parameter metric；
有没有更多 token / alias。
```

而是：

$$
\boxed{
\text{如何把训练时可见的 population signal，}
\text{变成持续的 basis-cover 边界条件，而不是一次性更新事件。}
}
$$

---

# 1. v13.6 独立分析

## 1.1 v13.6 的事实

v13.6 已经实现了：

```text
1. loss-interface cotangent per-example；
2. per-example gradient statistics；
3. parameter-SNR；
4. basis-channel-SNR；
5. basis-cover boundary guard；
6. MLP / KAN / Rational / Non-RAT scouts；
7. matched controls；
8. Soft / EMA3RoleNorm / role-wise threshold fallback；
9. tau / cap / norm sensitivity fallback；
10. Line S substrate/base scout。
```

最终仍然是：

```text
route = R4-PopRiskSNRNoGo
minimum_success = S1-EfficientSubstrate
official_success_reached = 0
promotion_allowed = 0
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
mlp_snr_pass_rows = 0
kan_parameter_snr_pass_rows = 0
kan_basis_snr_pass_rows = 0
nonrat_substrate_snr_gate_pass_count = 0
```

最接近的 rows 有明显几何改善，但 source advantage 太小。例如：

```text
KAN-BasisCoverBoundary / X4 / CE:
  source_vs_best ≈ 0.00038
  CouplingR2_delta ≈ +2.04
  NoiseSignalLeak_delta ≈ -0.0189
  Reservoir_delta ≈ -0.343
  CEp99_delta ≈ -0.955
  pass = 0
```

这说明 v13.6 不是“完全没动”，而是：

```text
几何 audit 可以变好；
cover boundary 不是全拒绝；
Rational substrate-SNR 可见；
但这些没有转化成 task/future advantage。
```

## 1.2 v13.6 失败的本质

v13.6 的失败不是因为：

```text
1. artifact 缺失；
2. forbidden information；
3. readout-feature proxy；
4. CE-only fallback 不完整；
5. cover boundary 全部拒绝；
6. SNR threshold 没扫；
7. Non-RAT 完全没 telemetry。
```

这些解释都已经被复核排除。

真正失败是：

$$
\boxed{
\text{one-shot SNR mask 只筛掉了一部分 noisy update，}
\text{但没有构造持续的 signal-channel training schedule。}
}
$$

它更像是 “AdamW update 的局部过滤器”，而不是 “boundary-conditioned training process”。

## 1.3 和文档启发的冲突

Generalization paper 的 population-risk gate来自 realized training path 上的 per-example gradient drift/diffusion：

$$
\bar g_B=\frac{1}{b}\sum_i g_i,
$$

$$
\Sigma_B=\frac{1}{b}\sum_i (g_i-\bar g_B)(g_i-\bar g_B)^T,
$$

$$
A_B=\bar g_B\bar g_B^T-\frac{1}{b-1}\Sigma_B.
$$

对 diagonal preconditioner，最简单 gate 是：

$$
\mu_k^2>\frac{\sigma_k^2}{b-1}.
$$

但是 v13.6 把这个思想实现成了单步 mask / single event。  
Deep Manifold 的启发则是：网络不是单步 fixed operator，而是在 moving covers 和 boundary-conditioned iteration 中构造 fixed-point regions。也就是说，functional update 应该是：

$$
\boxed{
\text{持续施加的边界条件}
}
$$

而不是：

$$
\boxed{
\text{某一次 parameter perturbation}
}
$$

因此 v13.7 的核心变化是：从 one-shot PopRiskSNR 转向 **boundary-conditioned population-risk training**。

---

# 2. 各条线进展百分比：v13.5 后估计 vs v13.6 后估计

| 线 | 上次估计 | 当前估计 | 变化 | 判断 |
|---|---:|---:|---:|---|
| 代码 / provenance / finalizer 审计 | 99% | 99% | 0 | 工程闭包强，不是 blocker |
| Historical FHQ / B320-current 工程能力 | 85% frozen | 85% frozen | 0 | 历史强，但 label-informed init 禁用 |
| Label-free FHQ / A-DYN monitor | 15% | 12% | -3 | 已不是主线 |
| Line C 几何审计 | 88% | 88% | 0 | 审计成熟，但不能作为 direction source |
| Rational efficient substrate | 85% | 85% | 0 | 仍是唯一稳定 substrate family |
| Rational S1C channel controllability | 60% | 60% | 0 | 可执行 channel movement，但价值未成立 |
| Non-RAT substrate / S1C | 20%-30% | 15%-25% | -5 | v13.6 Line S 仍 0 pass |
| Full basis-parameter writeback | 80% | 82% | +2 | v13.6 full_parameter_update_rows 已闭合 |
| Operator-level basis-channel solve | 50% | 45% | -5 | oracle/SNR target 都未产生 future advantage |
| Oracle value target | 0%-5% | 0%-5% | 0 | v13.5 已 no-go |
| Population-risk SNR implementation | 0% | 55% | +55 | 实现与 fallback 完成 |
| Population-risk SNR mechanism success | 0% | 5% | +5 | 0 pass；只留下负结论 |
| Basis-cover boundary implementation | 0% | 50% | +50 | 22/28 或 49/56 accepted，不是 no-op |
| Basis-cover boundary mechanism success | 0% | 5%-10% | +5 | 几何改善有，但 source advantage 不够 |
| MLP analog functional | 10%-15% | 8%-12% | -3 | MLP-SNR / HiddenChannelSNR 0 pass |
| Basis-specific functional official | 0%-5% | 0%-5% | 0 | S5 仍为 0 |
| 整体 next-gen MLP claim | 36%-42% | 34%-40% | -2 | 方向更清楚，但 one-shot SNR route 被证伪 |

---

# 3. 当前真正卡在哪里

## 3.1 第一层 blocker：one-shot update 不够

我们已经反复验证：

```text
response dictionary
basis-natural parameter metric
operator DeltaZ
oracle DeltaZ
population-risk SNR one-shot mask
basis-cover boundary one-shot guard
```

都不能稳定产生 future advantage。这说明问题不是单个 update direction 是否存在，而是：

$$
\boxed{
\text{functional update 需要作用于训练过程的边界条件，}
\text{而不是训练过程中的一次扰动。}
}
$$

## 3.2 第二层 blocker：basis cover 没有动态 plasticity schedule

Deep Manifold 强调 plasticity 会先升后降，node covers 的 orientation 每次迭代都在变。  
目前我们的 cover-boundary 只做了静态 guard：

```text
Rational den/slope/curvature guard
Fourier high-frequency guard
Chebyshev high-degree guard
RBF occupancy/width guard
Wavelet scale/support guard
```

但它没有分阶段：

```text
早期：允许 cover 展开；
中期：把 signal-rich cover 加固；
后期：抑制 noise-leak / tail drift。
```

所以 v13.7 必须把 functional update 从 `event` 改成 `phase schedule`。

## 3.3 第三层 blocker：Non-RAT 仍不是 functional substrate

v13.6 Line S 证明：

```text
D-RAT: 4/4 substrate-SNR pass
D-CHE / D-FOU / D-RBF / D-WAV: 0 pass
```

而且 Non-RAT 的 channel_snr_entropy 通常不是 0，说明不是全退化；它们仍被 workspace / efficiency / task-health gate 拒绝。  
所以 Non-RAT 需要的是 **substrate architecture repair**，不是直接 functional proof。

## 3.4 第四层 blocker：MLP analog 没有 generic positive

MLP-SNR 也 0 pass。这说明当前 functional update 不应被描述为通用 optimizer trick。  
如果后续 KAN + basis-cover schedule work，而 MLP 不 work，才可能证明 KAN 显式 basis cover 提供 functional leverage。

---

# 4. v13.7 总体思路

v13.7 不再问：

```text
哪一个 one-shot update 能过 synthetic 5/7？
```

而问：

```text
一个训练过程如果持续应用 population-risk boundary condition，
是否能比 AdamW 更快形成 stable signal channel？
```

新定义：

$$
\boxed{
\text{Functional update}
=
\text{Population-risk signal selection}
+
\text{Basis-cover boundary schedule}
}
$$

训练形式不再是：

$$
\theta'=\theta+\Delta\theta_{func}.
$$

而是：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{task,t}^{AdamW}
+
\lambda_t
\Delta\theta_{boundary,t}^{SNR+cover}.
$$

或者更简单地，functional 作为 AdamW 的 preconditioned mask：

$$
\Delta\theta_t
=
-\eta_t
\left[
P_{cover,t}
P_{SNR,t}
M_{basis,t}^{-1}
g_t
\right].
$$

其中：

```text
P_SNR,t:
  来自 per-example gradient mean/variance/covariance，决定哪些方向有 population signal。

M_basis,t:
  来自 basis telemetry，决定每类 basis 的自然尺度。

P_cover,t:
  来自 basis-cover boundary schedule，防止 cover collapse / noise leak / high-frequency / high-curvature explosion。
```

---

# 5. 核心假设

## H1：v13.6 失败是因为 one-shot，而不是 SNR 原理错误

如果把 SNR gate 放进连续训练步骤，而不是 synthetic one-step future probe，它可能改善 AUC-time、CEp99、LineC。  
验证：run multi-step SNR schedule。

## H2：MLP-SNR 是必须的基线

如果 MLP-SNR 也有效，那么 functional update 是 generic optimizer，不是 KAN-specific。  
如果 MLP-SNR 无效但 KAN-BasisSNR 有效，才说明 KAN basis cover 提供优势。

## H3：Basis-cover boundary 需要 phase schedule

同一个 basis 在训练早期和后期需要不同的边界条件。  
早期过强 boundary 会阻碍 learning，晚期过弱 boundary 会让 noise leak 和 tail drift 增加。

## H4：Non-RAT 的首要任务是 substrate repair，不是 functional proof

Non-RAT 只有在满足 substrate-health 后才进入 basis-SNR training。  
否则 functional failure 只是在坏 substrate 上重复失败。

## H5：LineC 只能做审计

LineC / CEp99 / NLL / ECE 不参与方向生成。  
它们只用于判断 functional 是否伤害 signal/reservoir/tail/calibration。

---

# 6. v13.7 实验总览

v13.7 分为七条线：

```text
Line R:
  Implementation / provenance / per-example gradient audit。

Line M:
  MLP continuous PopRisk-SNR training baseline。

Line K:
  Rational continuous PopRisk-SNR training baseline。

Line B:
  Rational basis-channel SNR + cover-boundary schedule。

Line D:
  Non-RAT substrate architecture repair + SNR scout。

Line C:
  Manifold-channel audit。

Line Z:
  Finalizer / no-go / next hypothesis。
```

---

# 7. Line R：实现审计

## 7.1 目标

确认 v13.7 不再是 one-shot update，而是真正多步训练过程中的 per-example gradient SNR / boundary schedule。

## 7.2 必须记录

`v137_implementation_readback.csv`

```text
runner_file
function_name
line_start
line_end
uses_loss_backward
uses_per_example_gradient
uses_generic_loss_interface
uses_ce_specific_formula
uses_validation_for_direction
uses_test_for_direction
uses_future_for_direction
uses_linec_for_direction
updates_named_parameters
optimizer_state_updated
snr_state_persistent
cover_state_persistent
schedule_phase
```

Gate：

```text
uses_ce_specific_formula = 0
uses_validation_for_direction = 0
uses_test_for_direction = 0
uses_future_for_direction = 0
uses_linec_for_direction = 0
snr_state_persistent = 1
```

如果 `snr_state_persistent = 0`，说明它还是 one-shot，不允许进入 v13.7 official。

---

# 8. Line M：MLP continuous PopRisk-SNR baseline

## 8.1 目标

先验证 Generalization paper 的可执行主张：

$$
q_k=1\left\{\mu_k^2>\tau\frac{\sigma_k^2}{b-1}\right\}.
$$

不是看 one-step synthetic，而是看连续训练。

## 8.2 方法

比较：

```text
MLP-AdamW
MLP-AdamW-SNRHard
MLP-AdamW-SNRSoft
MLP-AdamW-SNREMA
MLP-AdamW-SNRRoleNorm
```

数据：

```text
synthetic X1-X7
MNIST / Fashion-MNIST / KMNIST small-budget triage
```

训练预算：

```text
synthetic:
  steps = 200
  seeds = 0,1,2

real triage:
  train_size = 1024
  val/test = 512
  epochs = 3
  seeds = 0,1,2
```

## 8.3 指标

`v137_mlp_snr_training.csv`

```text
dataset_or_task
seed
method
loss_interface
step
snr_active_fraction
snr_active_fraction_by_layer
removed_update_norm_fraction
cos_snr_adamw
train_loss
val_loss
val_acc
val_loss_auc_step
val_loss_auc_time
CEp99
NLL
ECE
LineC_CouplingR2
LineC_NoiseSignalLeak
LineC_ReservoirRatio
step_time_ratio
memory_ratio
```

## 8.4 Gate

Synthetic MLP-SNR exploratory pass：

$$
\Delta AUC_{time} \le -0.02
$$

or

$$
source\_vs\_AdamW \ge 0.005
$$

并且：

$$
CEp99_{delta} \le 0.05,
$$

$$
ECE_{delta} \le 0.02.
$$

Real triage pass：

```text
mean_delta_vs_AdamW >= 0
AUC_time_ratio_vs_AdamW <= 1.00
CEp99_delta <= 0.05
ECE_delta <= 0.02
```

Interpretation：

```text
MLP-SNR pass:
  functional has generic optimizer value.

MLP-SNR fail:
  generic parameter SNR not sufficient; KAN-specific basis cover must carry the advantage.
```

---

# 9. Line K：Rational continuous PopRisk-SNR baseline

## 9.1 目标

在唯一稳定 substrate family Rational 上测试连续 SNR training 是否比 one-shot effective。

方法：

```text
RAT-AdamW
RAT-ParameterSNRHard
RAT-ParameterSNRSoft
RAT-ParameterSNREMA
RAT-GroupSNR
RAT-GroupSNREMA
```

这里 `GroupSNR` 按 Rational group / numerator / denominator / readout group 统计：

$$
SNR_g=\frac{\|\mu_g\|_2^2}{\operatorname{tr}(\Sigma_g)/(b-1)+\epsilon}.
$$

## 9.2 指标

`v137_rational_snr_training.csv`

```text
candidate
task_or_dataset
seed
method
step
group_snr
group_snr_active
num_snr_active
den_snr_active
readout_snr_active
den_p01
den_p99
r_prime_p99
r_double_prime_p99
tangent_condition
group_diversity
train_loss
val_loss
val_acc
AUC_time
CEp99
NLL
ECE
LineC_CouplingR2
NoiseSignalLeak
ReservoirRatio
```

## 9.3 Gate

Rational SNR substrate pass：

```text
mean_delta_vs_RAT_AdamW >= 0
AUC_time_ratio_vs_RAT_AdamW <= 1.00
LineC_CouplingR2_delta >= 0
NoiseSignalLeak_delta <= 0
ReservoirRatio_delta <= 0
CEp99_delta <= 0.05
```

---

# 10. Line B：Rational basis-channel SNR + cover-boundary schedule

## 10.1 目标

把 population-risk signal selection 和 basis-cover boundary stabilization 合并。

## 10.2 Phase schedule

```text
Phase 1: Plasticity-open
  steps 0-20%
  SNR gate loose
  cover boundary weak
  目标：不阻碍 signal channel 形成

Phase 2: Cover-alignment
  steps 20%-70%
  SNR gate moderate
  basis group SNR + cover guard
  目标：让 signal-rich channel 稳定

Phase 3: Fixed-point consolidation
  steps 70%-100%
  SNR gate strict
  cover boundary stronger
  目标：抑制 noise leak / tail drift / high curvature
```

## 10.3 Boundary definitions

### Rational boundary

```text
den_p01_floor
r_prime_p99_cap
r_double_prime_p99_cap
group_diversity_floor
readout_rational_coupling_cap
```

Boundary debt：

$$
D_{RAT}
=
w_d[\tau_d-den_{p01}]_+
+
w_1[r'_{p99}-\tau_1]_+
+
w_2[r''_{p99}-\tau_2]_+
+
w_g[\tau_g-diversity]_+.
$$

### Fourier boundary

```text
high_freq_energy_cap
phase_drift_cap
band_snr_floor
```

### Chebyshev boundary

```text
high_degree_energy_cap
degree_snr_floor
recurrence_max_abs_cap
```

### RBF boundary

```text
center_occupancy_floor
width_condition_cap
out_of_grid_fraction_cap
```

### Wavelet boundary

```text
scale_energy_balance
support_overlap_cap
local_tail_coverage_floor
```

## 10.4 Candidate methods

```text
RAT-BasisSNR
RAT-BasisSNR-CoverWeak
RAT-BasisSNR-CoverPhaseSchedule
RAT-BasisSNR-CoverConsolidateOnly
RAT-BasisSNR-CoverNoPlasticity, diagnostic
```

## 10.5 Gate

Synthetic 5/7 pass required before real short-run:

```text
>= 5 of X1-X7 pass
>= 2 seeds or >= 2 substrate candidates per family
source_vs_best_control >= 0.005
CEp99_delta <= 0.05
NLL_delta <= 0.02
ECE_delta <= 0.02
NoiseSignalLeak_delta <= 0
ReservoirRatio_delta <= 0
```

No real short-run before synthetic 5/7.

---

# 11. Line D：Non-RAT substrate architecture repair + SNR scout

## 11.1 目标

Non-RAT 不直接 functional proof。先修 substrate-health：

```text
Chebyshev:
  degree-energy substrate with lower lifetime and no task collapse.

Fourier:
  low-frequency + identity residual substrate, no high-frequency leak.

RBF/FastKAN:
  compact center/width substrate with occupancy balance.

Wavelet:
  local hat/triangle substrate with scale/support balance.
```

## 11.2 Methods

```text
CHE-SNR4-degreeLowRankResidual
CHE-SNR5-degreeLateEnable
FOU-SNR4-lowFreqIdentityResidual
FOU-SNR5-bandLimitedResidual
RBF-SNR4-compactOccupancyRepair
RBF-SNR5-widthConditionedLocalBump
WAV-SNR4-hatScaleBalanced
WAV-SNR5-localSupportOccupancy
```

## 11.3 Gate

Substrate-health pass：

```text
workspace_raw_ratio <= 1.25
workspace_incremental_ratio <= 2.0
step_ratio <= 1.75
mean_delta_vs_MLP >= -0.05
worst_delta_vs_MLP >= -0.10
AUC_time_ratio_vs_MLP <= 2.0
LineC_pass_rate >= 0.30
channel_snr_entropy > 0.10
cover_rejection_fraction <= 0.70
```

Only after this can Non-RAT enter Line B.

---

# 12. Line C：Audit only

Line C records:

```text
CouplingR2
CouplingCorr
NoiseSignalLeak
RealSignalReservoirRatio
KernelDrift
CEp99
NLL
ECE
Brier
margin_p10
```

Rules:

```text
LineC cannot generate direction.
LineC cannot select candidate during training.
LineC can reject candidate after audit.
```

---

# 13. 必须生成的可视化

```text
fig_mlp_snr_vs_adamw_loss_time.svg
fig_rational_parameter_snr_vs_basis_snr.svg
fig_basis_snr_active_fraction_by_phase.svg
fig_cover_boundary_debt_by_phase.svg
fig_signal_reservoir_trajectory.svg
fig_noise_leak_vs_snr_active_fraction.svg
fig_nonrat_substrate_health_matrix.svg
fig_mlp_vs_rational_snr_comparison.svg
fig_synthetic_5of7_heatmap.svg
fig_real_triage_if_opened.svg
```

---

# 14. Failure -> Codex 自动尝试方向

## Case A：MLP-SNR fail，RAT-SNR fail

不要再调 SNR threshold。直接进入：

```text
R0-PopRiskSNRImplementationNoGo
```

然后执行：

```text
1. verify per-example gradients against autograd baseline;
2. verify AB formula on tiny linear model with known solution;
3. run noisy-label toy where SNR should help;
4. if still fail, stop SNR route and return substrate/base architecture.
```

## Case B：MLP-SNR pass，RAT-SNR fail

说明 generic optimizer 机制存在，但 KAN substrate / telemetry 有问题。Codex 先尝试：

```text
1. reduce basis cover guard strength;
2. compare parameter-SNR vs group-SNR;
3. inspect den/r'/r''/group diversity rejection;
4. run Rational no-cover SNR;
5. if no-cover SNR works, cover boundary is too strong;
6. if parameter-SNR works but group-SNR fails, group telemetry wrong.
```

## Case C：MLP-SNR fail，RAT basis-SNR pass

这是最有价值路线。Codex 必须继续：

```text
1. expand to two more Rational substrates;
2. run real triage;
3. run Non-RAT SNR scout;
4. measure basis-specific advantage vs MLP analog.
```

## Case D：RAT basis-SNR works but cover boundary hurts

调整：

```text
1. boundary only in Phase 3;
2. use weak symmetric boundary in Phase 1/2;
3. record cover debt but do not project until consolidation.
```

## Case E：Cover boundary improves geometry but hurts source

这就是 v13.6 模式。Codex 不准继续小修 boundary，必须 test:

```text
1. Phase schedule;
2. delayed boundary;
3. separate task-SNR and boundary update frequencies;
4. if still fail, boundary is audit-only, not active update.
```

## Case F：Non-RAT substrate remains WorkspaceOnly

不要 functional。先做:

```text
1. lifetime repair;
2. task-health repair;
3. substrate-SNR telemetry audit;
4. if no substrate after two rounds, freeze that family for current milestone.
```

---

# 15. Official success definitions

## S1：SNR implementation sanity

```text
per-example gradients verified
AB / SNR formula verified on toy linear
MLP-SNR runner executes
KAN-SNR runner executes
no forbidden info
```

## S2：MLP generic SNR positive

```text
MLP-SNR beats MLP-AdamW on synthetic >= 5/7
or real triage mean/AUC/CEp99 gate
```

## S3：KAN basis-SNR positive

```text
KAN-BasisSNR beats same KAN AdamW on synthetic >= 5/7
and beats parameter-SNR controls
```

## S4：KAN basis-cover boundary positive

```text
BasisSNR+Cover beats BasisSNR without worsening source
LineC improves
tail/calibration non-harm
```

## S5：Real short-run official

```text
real triage 3x3 pass
source_vs_AdamW/control positive
AUC_time not worse
CEp99/NLL/ECE non-harm
LineC non-harm or improvement
step/memory overhead acceptable
```

---

# 16. Stop / go rules

```text
If S1 fail:
  implementation bug; fix before science conclusion.

If S1 pass but S2/S3 both fail:
  record PopRiskSNRNoGo_CurrentImplementation.
  Do not continue threshold/grid search.
  Return to substrate/base architecture or new theory.

If S2 pass but S3 fail:
  functional is generic but not KAN-specific; continue MLP optimizer line separately.

If S3 pass:
  open KAN functional re-entry.

If S4 pass:
  open real short-run.

If S5 pass:
  begin 5-seed / 10-seed confirmation.
```

---

# 17. 当前最终判断

v13.6 不是没意义，但它是 **one-shot SNR no-go**，不是 functional update no-go。

下一步必须从：

```text
one-shot SNR update
```

变成：

```text
boundary-conditioned population-risk training process
```

这才真正吸收了两篇文档的启发：

```text
Generalization:
  signal channel / reservoir / population-risk SNR gate。

Deep Manifold:
  moving covers / boundary-conditioned fixed-point iteration / plasticity schedule。
```

