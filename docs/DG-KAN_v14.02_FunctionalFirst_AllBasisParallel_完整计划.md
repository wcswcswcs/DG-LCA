# DG-KAN v14.2：Functional-First + All-Basis Parallel Substrate Repair 完整计划

> 版本：v14.2  
> 生成目的：修正 v14.1 只强调 functional-first、忽略 v14.0 中其他经典基函数修复的问题。  
> 核心态度：functional update 仍然是突破口，但 functional update 不能悬空；它必须在多个可控 basis substrate 上验证。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no label-informed initialization；B-spline 不作为 active family；functional direction 不使用 validation/test/future/query batch，不使用 CEp99 / NLL / ECE / LineC hard target 生成方向；这些指标只作为 audit / gate。

---

# 0. 项目总目标与当前进展

## 0.1 总目标

DG-KAN 的总目标不是找一个单独更快的 KAN basis，也不是找一个只在某个 diagnostic score 上好看的 functional trick。总目标是：

$$
\boxed{
\text{label-free strict FC-PureKAN substrate/base}
+
\text{loss-interface-generic functional update}
>
\text{same substrate/base + ordinary AdamW/backprop}
}
$$

并且这个结果必须满足：

```text
1. 表达力不打折；
2. forward / backward / step / memory 与 MLP 可比；
3. 收敛轨迹不慢，AUC-step / AUC-time 不坏；
4. 几何更健康：真实信号进入 signal channel，噪声不进入 signal channel，真实信号不困在 reservoir；
5. functional update 的收益必须独立于 AdamWParallel / RandomMatchedNorm / NoOp / SNR-only / LR controls；
6. 不靠 CE-specific trick，不靠 dataset-specific controller。
```

最终要证明的是：

$$
\boxed{
\Delta_{\text{functional}}
=
\left[
\text{KAN/Basis + FMS}
-
\text{KAN/Basis + AdamW}
\right]
>
0
}
$$

并且如果 MLP 上也存在同类收益，还必须进一步证明 KAN / basis 坐标提供额外 leverage：

$$
\boxed{
\Delta_{\text{KAN-specific}}
=
(\text{Basis-FMS}-\text{Basis-AdamW})
-
(\text{MLP-FMS}-\text{MLP-AdamW})
>0.
}
$$

---

## 0.2 当前真实进展

最近几轮给出的结论需要降噪理解：

```text
1. B320-current / FHQ 历史上很强，但含 label-informed init，不能作为 official label-free claim。
2. Rational 是当前唯一稳定接近 functional substrate 的 active basis family。
3. RBF / FastKAN、Chebyshev、Fourier、Wavelet 不能放弃；之前多数失败来自旧实现、workspace/lifetime、expression 或 task-health blocker，不足以判死。
4. MLP PopRisk / SNR 一度有 generic 信号，但 10-seed 没确认；它仍是必要 control，而不是 KAN success。
5. response dictionary、oracle DeltaZ、cover_purity、A-RCF/A-CPF、K-token、O-token、BN/BM metric 等路线已经多次失败，不能继续当主线。
6. functional update 如果要作为突破口，必须变成 persistent training boundary / metric state，而不是一次性参数扰动。
```

---

## 0.3 v14.1 的缺口与 v14.2 的修正

v14.1 做对了一个方向：把 functional update 重新定义为 **Functional Metric State / training boundary**，而不是一次性 geometry perturbation。

但 v14.1 漏掉了 v14.0 的关键要求：

```text
经典 basis family 必须继续并行修复：
  Rational
  RBF / FastKAN
  Chebyshev
  Fourier
  Wavelet

B-spline 当前仍 frozen，但 MatrixKAN / KAN-SA / local support 思想可以作为 implementation inspiration。
```

因此 v14.2 的核心修正是：

$$
\boxed{
\text{Functional-first，但不是 basis-blind。}
}
$$

也就是说：

```text
functional update 是突破口；
basis substrate 是 functional update 的作用对象；
每个 active basis 都必须推进到：
  EfficientSubstrate / HealthyBase / FunctionalSubstrate / RejectedForThisVersion。
```

---

# 1. v14.2 的总策略

v14.2 不再走两种错误路线。

第一种错误路线：

```text
先把某个 basis 单独调成 healthy base，再谈 functional。
```

这会无限拖延 functional update。

第二种错误路线：

```text
不管 basis 是否可控，直接用 functional update 试图修一切。
```

这会在坏 substrate 上制造无意义的 functional no-go。

v14.2 采用三层结构：

```text
Gate S: Efficient / controllable substrate
  basis 至少要跑得动、有基本表达、有 telemetry、有最小 task-health。

Gate F: Functional Metric State repair
  在 substrate 上测试 persistent FMS 是否改善 training path。

Gate H: Healthy / official base
  basis + FMS 或 basis-only 通过 task / AUC / LineC / tail / controls。
```

---

# 2. 核心科学假设

## H1：functional update 的突破点不是一次性 perturbation，而是 persistent metric state

旧 functional update 形式：

$$
\theta'=\theta+\Delta \theta_{\text{func}}.
$$

v14.2 要测试的新形式：

$$
c_{r,t+1}
=
\beta c_{r,t}
+
(1-\beta)\operatorname{clip}(U_{r,t},u_{\min},u_{\max}),
$$

$$
\Delta\theta_t
=
-\eta_t
T(c_t)
P_{\text{safe},t}
M_{\text{Adam},t}^{-1/2}
m_t.
$$

其中：

```text
c_t:
  persistent functional metric state。

U_{r,t}:
  role / layer / basis group 的 population-risk utility。

T(c_t):
  根据 functional state 调整 plasticity / learning rate / trust region。

P_safe,t:
  basis-specific safety projection，只防灾难，不生成任务方向。

m_t, M_Adam,t:
  ordinary AdamW moment / second moment。
```

这里的 functional update 不替代 AdamW，而是改变 AdamW 的训练边界和长期 metric。

---

## H2：loss-agnostic 应改为 loss-interface-generic，而不是 label-blind

允许：

```text
通过统一 loss interface 读取当前训练 loss 的 output cotangent；
计算 per-example gradient mean / variance / covariance；
用 population-risk SNR 判断哪些训练信号具有 coherent drift。
```

禁止：

```text
hardcode CE-specific formula；
用 CEp99 / NLL / ECE / LineC hard target 生成方向；
用 validation/test/future/query batch；
用 dataset-name branch。
```

对每个 role / group $r$：

$$
g_{i,r}=J_{\theta_r}(x_i)^T\delta_i,
$$

$$
\mu_r=\frac{1}{b}\sum_i g_{i,r},
$$

$$
\Sigma_r=\frac{1}{b}\sum_i(g_{i,r}-\mu_r)(g_{i,r}-\mu_r)^T.
$$

定义 population-risk utility：

$$
U_r
=
\mu_r^T M_r^{-1}\mu_r
-
\frac{1}{b-1}
\operatorname{tr}(M_r^{-1}\Sigma_r).
$$

最简单的 diagonal rule 是：

$$
\mu_{r,k}^2 >
\tau\frac{\sigma_{r,k}^2}{b-1}.
$$

---

## H3：basis 的职责不是自己完全 healthy，而是成为可控 substrate

一个 basis 不必单独解决所有 AUC / tail / LineC 问题，但必须满足：

```text
1. 运行成本接近 MLP；
2. 表达力不明显打折；
3. 任务训练不灾难；
4. 有 family-specific telemetry；
5. FMS 能作用到真实参数或真实 basis channel；
6. safety projection 不会拒绝大多数更新。
```

因此，basis 失败要分清：

```text
EfficiencyBlocked
ExpressionBlocked
TaskHealthBlocked
TelemetryBlocked
FunctionalControllabilityBlocked
RejectedForThisVersion
```

不能笼统写“family failed”。

---

# 3. 进度表与预算分配

## 3.1 当前完成度估计

| 线 | 当前完成度 | 判断 |
|---|---:|---|
| 代码 / provenance / finalizer 审计 | 99% | 工程审计很强，不是 blocker |
| Historical FHQ/B320-current | 85% frozen | 历史强，但 label-informed init 禁用 |
| Rational substrate | 80%-85% | 当前唯一稳定 substrate 起点 |
| RBF/FastKAN substrate | 20%-30% | 需要重启 FastKAN-style no-materialize 路线 |
| Chebyshev substrate | 25%-35% | exact/correctness 有基础，但 lifetime/task-health 未闭合 |
| Fourier substrate | 25%-35% | lowfreq 计算可快，但 expression/task/cover 未闭合 |
| Wavelet substrate | 20%-30% | local path 有信号，但 task-health 差 |
| Functional Metric State | 0%-5% | 新定义尚未验证 |
| MLP generic FMS control | 15%-20% | PopRisk/SNR 有过信号，但 10-seed 未确认 |
| Line C geometry audit | 88% | 审计可用，不做方向源 |
| Overall next-gen MLP claim | 25%-35% | 必须重建 functional + all-basis substrate |

---

## 3.2 计算预算

v14.2 不允许 functional 和 basis 相互吞噬预算，采用固定比例：

```text
Line F Functional Metric State:
  35%

Line D All-basis substrate repair:
  40%

Line M MLP generic control:
  10%

Line C/R/Z audit, finalizer, figures:
  15%
```

Line D 内部分配：

```text
Rational:
  30% of Line D

RBF / FastKAN:
  25% of Line D

Chebyshev:
  15% of Line D

Fourier:
  15% of Line D

Wavelet:
  15% of Line D
```

B-spline：

```text
active_budget = 0
status = frozen
allowed_use = implementation inspiration only
```

---

# 4. Line R：代码、信息源、实现语义审计

## 4.1 目标

确认下一轮实验真的测试 v14.2 的核心机制，而不是旧 token / proxy / artifact 伪装。

## 4.2 必须审查

Codex 必须在复盘文件中写清楚以下核心代码路径：

```text
1. FMS state 的创建、更新、写回。
2. per-example gradient 采集是否来自统一 loss interface。
3. FMS 是否写真实参数 / optimizer state，而非 feature table proxy。
4. MLP-FMS 与 KAN-FMS 是否共用同一 loss-interface contract。
5. 每个 basis 的 telemetry 是否来自 train-stream / model state，不来自 LineC / validation / test。
6. CEp99 / NLL / ECE / LineC 是否只做 audit。
7. AdamWParallel / RandomMatchedNorm / NoOp / SNR-only controls 是否同窗口同预算。
8. B-spline 是否没有 active budget。
```

## 4.3 必须生成 artifact

```text
v142_code_path_manifest.csv
v142_functional_metric_state_readback.md
v142_loss_interface_audit.csv
v142_basis_family_implementation_manifest.csv
v142_forbidden_information_audit.csv
v142_control_semantics_audit.csv
```

硬门：

```text
如果 FMS 只是 one-shot mask 或 readout-feature proxy:
  route = R0-FMSNotImplemented

如果 any official candidate 使用 label-informed init:
  route = R0-LabelInformedInitViolation

如果 CEp99/NLL/ECE/LineC 进入 direction source:
  route = R0-AuditMetricUsedAsDirection
```

---

# 5. Line F：Functional Metric State 主线

## 5.1 目标

验证 functional update 本身是否有突破，而不是验证某个 geometry score 是否可局部调高。

核心问题：

$$
\boxed{
\text{Persistent FMS 是否能改变训练路径，并稳定优于 ordinary AdamW？}
}
$$

---

## 5.2 FMS 方法族

### F0：AdamW baseline

```text
普通 AdamW / manual AdamW。
```

### F1：Prior SNR one-shot / continuous reference

```text
保留 v13.7 / v13.8 最强 SNR/Blend 作为 reference。
不能当作 v14.2 新机制。
```

### F2：ParameterFMS

每个 named parameter 一个 functional state：

$$
c_{k,t+1}=\beta c_{k,t}+(1-\beta)\operatorname{clip}(U_{k,t}).
$$

更新：

$$
\Delta\theta_{k,t}=-\eta \cdot \operatorname{softplus}(c_{k,t}) \cdot AdamWUpdate_{k,t}.
$$

### F3：LayerFMS

每个 layer 一个 state，适合 MLP control 与 all-basis shared control。

### F4：RoleFMS

KAN / basis 角色：

```text
readout
basis coefficient
projection
numerator
denominator
center
width
frequency
degree
scale
support
```

每个 role 一个 state。

### F5：BasisGroupFMS

每个 basis group / channel 一个 state。

### F6：LowRankFMS

维护低秩 state：

$$
C_t=D_t+U_tU_t^T.
$$

但只能在 F2-F5 有 positive signal 后打开。

### F7：PhaseScheduleFMS

分阶段：

```text
Phase 1: plasticity-open
  FMS 只做弱 gating，不阻碍 early signal discovery。

Phase 2: alignment
  FMS 增强 coherent population-risk direction。

Phase 3: consolidation
  FMS 加强 noise / curvature / tail safety。
```

---

## 5.3 MLP FMS 对照

必须先跑：

```text
M0 MLP-AdamW
M1 MLP-best-prior-SNR
M2 MLP-ParameterFMS
M3 MLP-LayerFMS
M4 MLP-LowRankFMS
M5 MLP-PhaseScheduleFMS
```

目的：

```text
判断 functional update 是否作为 generic optimizer 成立。
```

MLP 成功不是 KAN 成功；它只给出 generic baseline。

---

## 5.4 Rational FMS

Rational 作为当前最稳定 substrate，必须跑：

```text
R0 Rational-AdamW
R1 Rational-PriorSNR
R2 Rational-ParameterFMS
R3 Rational-RoleFMS
R4 Rational-GroupFMS
R5 Rational-BasisChannelFMS
R6 Rational-FMS + denominator/slope safety
R7 Rational-FMS + delayed readout-basis coupling
R8 Rational-FMS + phase schedule
```

Rational telemetry：

```text
den_p01
den_p99
r_prime_p99
r_double_prime_p99
group_diversity
num_update_norm
den_update_norm
num_den_cosine
readout_basis_coupling
basis_channel_snr
```

---

## 5.5 FMS 进入其他 basis 的条件

其他 basis 只有通过 Line D substrate gate 后才能进入 FMS。

```text
RBF/FastKAN:
  center / width / occupancy telemetry 通过。

Chebyshev:
  degree energy / recurrence stability 通过。

Fourier:
  frequency band / phase stability 通过。

Wavelet:
  scale / support occupancy 通过。
```

不能把 task-collapsed basis 送进 FMS 后写成 functional no-go。

---

## 5.6 Functional Gate

### Exploration gate

Synthetic X1-X7 中：

```text
>= 5/7 task families pass
source_vs_best_control >= 0.002
AUCtime_ratio <= 1.05
CEp99_delta <= 0.10
NLL_delta <= 0.05
ECE_delta <= 0.05
LineC majority pass
step_time_overhead <= 15%
memory_overhead <= 15%
```

### Official gate

Real 3x3 triage：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 1024
val/test = 512
epochs = 8 or fixed step budget
```

要求：

$$
source\_vs\_best\_control \ge 0.005,
$$

$$
AUCtime\_ratio \le 1.0,
$$

$$
CEp99\_delta \le 0.05,
$$

$$
NLL\_delta \le 0.02,
$$

$$
ECE\_delta \le 0.02,
$$

$$
LineC\_pass\_rate \ge 0.60.
$$

10-seed 只在 3x3 official 过后打开。

---

# 6. Line D：All-Basis Substrate Repair

## 6.1 总目标

Line D 不是和 functional 抢方向，而是为 functional 提供多个可控 substrate。

每个 active basis 必须输出：

```text
FamilyStatus in:
  EfficientSubstrate
  HealthyBase
  FunctionalSubstrate
  EfficiencyBlocked
  ExpressionBlocked
  TaskHealthBlocked
  TelemetryBlocked
  RejectedForThisVersion
```

---

## 6.2 通用 substrate gate

进入 FMS 前必须满足：

$$
step\_ratio \le 1.75,
$$

$$
memory\_ratio \le 1.75,
$$

$$
mean\_delta\_vs\_MLP \ge -0.05,
$$

$$
worst\_delta\_vs\_MLP \ge -0.10,
$$

$$
AUCtime\_ratio \le 2.0,
$$

$$
LineC\_pass\_rate \ge 0.30.
$$

这是 minimal substrate gate，不是 final healthy gate。

Healthy gate 更严格：

$$
step\_ratio \le 1.25,
$$

$$
memory\_ratio \le 1.25,
$$

$$
mean\_delta\_vs\_MLP \ge 0,
$$

$$
worst\_delta\_vs\_MLP \ge -0.02,
$$

$$
AUCtime\_ratio \le 1.0,
$$

$$
LineC\_pass\_rate \ge 0.60.
$$

---

## 6.3 Rational / Group Rational

### 当前状态

Rational 是最接近 substrate 的 family，但旧 D-RAT token 已经边际收益低。下一步必须对齐 group rational / KAT-style GPU 路线，而不是继续自造局部 alias。

### 候选

```text
RAT-A GroupRational-Horner-g8
RAT-B GroupRational-Horner-g16
RAT-C GroupRational-TritonEval-g8
RAT-D GroupRational-TritonEval-g16
RAT-E RationalKAT-cu-compatible-shape
RAT-F DenominatorRecomputeBackward
RAT-G ReadoutBasisDecoupledRational
RAT-H MinimalRationalSubstrateForFMS
```

### 必须记录

```text
group_count
degree_num
degree_den
horner_enabled
triton_eval_enabled
division_count
den_p01
den_p99
den_condition
r_prime_p99
r_double_prime_p99
group_diversity
dead_group_fraction
num_update_norm
den_update_norm
readout_basis_coupling
functional_state_rejection_fraction
```

### 失败后 Codex 先尝试

```text
如果 denominator unsafe:
  先修 denominator guard / init，不做 task search。

如果 step/memory fail:
  Horner fused；
  recompute denominator；
  no power materialization；
  group rational not per-edge rational。

如果 task ok 但 LineC fail:
  进入 Rational-FMS safety projection，而不是 CE tuning。

如果 FMS rejection > 0.80:
  说明 substrate 不可控，回到 group structure。
```

---

## 6.4 RBF / FastKAN

### 当前状态

RBF/FastKAN 之前不能按旧 dense RBF 失败判死。FastKAN-style 路线需要重启为高优先级 substrate。

### 候选

```text
RBF-A FastKAN-Gaussian-K4-fixed-center
RBF-B FastKAN-Gaussian-K8-fixed-center
RBF-C CompactRBF-Triangular-K4
RBF-D CompactRBF-GaussianLocal-K4-active2
RBF-E CompactRBF-GaussianLocal-K8-active4
RBF-F RBF-as-SplineApprox-noDenseBasis
RBF-G QuantileCenterRBF-fixedWidth
RBF-H RBF-MinimalSubstrateForFMS
```

### 必须记录

```text
K_total
K_active
center_trainable
width_trainable
center_init
width_init
exp_count_per_sample
approx_exp_enabled
active_center_entropy
dead_center_fraction
out_of_grid_fraction
width_p01
width_p99
center_condition
basis_materialized_shape
dense_basis_materialized
```

### FMS telemetry

```text
center_snr
width_snr
active_center_snr
occupancy_debt
width_condition_debt
out_of_grid_debt
```

### 失败后 Codex 先尝试

```text
如果 exp 成本高:
  active K 降到 2/4；
  固定 center/width；
  triangular compact bump；
  approximate exp 只能 diagnostic。

如果 dead_center_fraction 高:
  train-stream quantile centers；
  增大 width；
  occupancy rebalance。

如果 expression fail:
  K_active 从 2 增到 4；
  不直接回 dense K。

如果 task/AUC fail:
  先测 FMS center/width gating；
  不按 dataset tuning。
```

---

## 6.5 Chebyshev

### 候选

```text
CHE-A Chebyshev-K3-recurrence
CHE-B Chebyshev-K4-recurrence
CHE-C Chebyshev-K6-lowrank
CHE-D Chebyshev-degree-energy-capped
CHE-E Chebyshev-late-high-degree-enable
CHE-F Chebyshev-MinimalSubstrateForFMS
```

### 必须记录

```text
degree_K
recurrence_time_ms
basis_condition_proxy
input_clamp_rate
degree_energy_0
degree_energy_1
degree_energy_high
degree_energy_high_ratio
high_degree_update_norm
degreewise_snr
recurrence_stability
```

### FMS telemetry

```text
degree_snr
high_degree_noise_leak_proxy
degree_energy_debt
degree_churn
```

### 失败后 Codex 先尝试

```text
如果 workspace fail:
  recurrence no-materialize；
  recompute derivative；
  no dense degree tensor。

如果 high_degree_energy 过高:
  degree damping；
  late-enable high degree；
  degree-wise FMS cap。

如果 expression fail:
  先 K3->K4；
  K6 只能在 A1 pass 后跑。

如果 task/AUC fail:
  degree energy schedule；
  不是调 CE。
```

---

## 6.6 Fourier

### 候选

```text
FOU-A Fourier-lowfreq-K2
FOU-B Fourier-lowfreq-K4
FOU-C Fourier-phase-amplitude-shared-K4
FOU-D Fourier-bandlimited-residual-K4
FOU-E Fourier-late-highfreq-enable
FOU-F Fourier-MinimalSubstrateForFMS
```

### 必须记录

```text
frequency_count
sincos_time_ms
phase_drift
amplitude_norm
frequency_energy_low
frequency_energy_mid
frequency_energy_high
high_freq_ratio
bandwise_snr
NoiseSignalLeak
E4_high_frequency_delta
```

### FMS telemetry

```text
band_snr
high_frequency_noise_guard
phase_stability_debt
frequency_band_churn
```

### 失败后 Codex 先尝试

```text
如果 expression fail:
  K2 -> K4；
  phase-amplitude sharing；
  identity residual。

如果 NoiseSignalLeak high:
  high-frequency guard；
  bandwise SNR；
  late high-frequency enable。

如果 memory/lifetime fail:
  fused sincos；
  derivative recompute；
  no dense frequency tensor。
```

---

## 6.7 Wavelet

### 候选

```text
WAV-A Haar-K4
WAV-B TriangularWavelet-K4
WAV-C HatWavelet-local-K4
WAV-D CompactSplineWavelet-K4
WAV-E ScaleDiversityLocalWavelet
WAV-F Wavelet-MinimalSubstrateForFMS
```

### 必须记录

```text
scale_count
shift_count
learnable_scale
learnable_shift
scale_energy
support_overlap
local_support_fraction
active_support_entropy
dead_support_fraction
local_tail_coverage
wavelet_eval_time_ms
```

### FMS telemetry

```text
scale_snr
support_snr
scale_energy_debt
support_overlap_debt
local_tail_snr
```

### 失败后 Codex 先尝试

```text
如果 efficiency fail:
  Haar / triangular；
  fixed scales；
  no Morlet/MexicanHat mainline。

如果 task collapse:
  local support occupancy repair；
  scale energy cap；
  LineC audit。

如果 expression fail:
  modest scale diversity；
  not heavy exp/sin path。
```

---

## 6.8 B-spline

```text
status = Frozen
active_budget = 0
```

但允许吸收以下 implementation ideas：

```text
no dense basis materialization；
uniform local support；
matrix representation；
systolic / sparse lifetime thinking；
two-bin / four-bin interpolation idea。
```

不能重启 old B-spline search，除非用户明确解除 frozen 状态。

---

# 7. Line M：MLP Generic FMS Control

## 7.1 目标

判断 FMS 是否是 generic optimizer 机制。

## 7.2 方法

```text
M0 MLP-AdamW
M1 MLP-PriorSNR
M2 MLP-ParameterFMS
M3 MLP-LayerFMS
M4 MLP-LowRankFMS
M5 MLP-PhaseScheduleFMS
```

## 7.3 Gate

10-seed confirm 前先 3x3 triage：

```text
source_vs_adamw >= 0.005
AUCtime_ratio <= 1.0
CEp99_delta <= 0.05
NLL_delta <= 0.02
ECE_delta <= 0.02
```

如果 MLP-FMS 成功但 KAN/basis-FMS 不成功：

```text
route = R3-GenericFunctionalOnlyKANSpecificNotEstablished
```

如果 MLP-FMS 和 basis-FMS 都失败：

```text
route = R4-FMSNoGoCurrentDefinition
```

---

# 8. Line C：Geometry / Manifold-Channel Audit

Line C 只做审计，不做 direction source。

## 8.1 必须记录

```text
CouplingR2
CouplingCorr
NoiseSignalLeak
RealSignalReservoirRatio
signal_mass_topk
reservoir_fraction
KernelDrift
CEp99
NLL
ECE
Brier
margin_p10
wrong_conf_p95
```

## 8.2 公式

Train-probe coupling：

$$
CouplingR^2
=
1-
\frac{\|\Delta U_Q-A_t\Delta U_B\|_F^2}
{\|\Delta U_Q\|_F^2+\epsilon}.
$$

Signal / reservoir audit：

```text
signal channel:
  top eigenspace of windowed train-probe sketch.

reservoir:
  low-eigen / test-invisible complement.
```

## 8.3 Gate

Geometry improvement 不要求 kernel drift 小；要求：

```text
NoiseSignalLeak 不上升；
RealSignalReservoirRatio 不上升；
CouplingR2 不塌；
tail / calibration 不坏。
```

---

# 9. 实验流程

## P0：Unified code/provenance audit

执行 Line R，确认信息源和 FMS 语义。

## P1：All-basis substrate scout

并行跑：

```text
Rational
RBF/FastKAN
Chebyshev
Fourier
Wavelet
```

输出 family status。

## P2：FMS minimal implementation

跑：

```text
MLP-FMS
Rational-FMS
```

确认 persistent state 是否真写 optimizer / parameter update。

## P3：Synthetic X1-X7 FMS training

比较：

```text
AdamW
PriorSNR
FMS-Parameter
FMS-Role/Layer
FMS-Basis
```

要求至少 synthetic 5/7 才允许 real triage。

## P4：All-basis FMS entry

只有 substrate gate pass 的 basis 进入 FMS。

## P5：Real 3x3 triage

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
```

## P6：5-seed / 10-seed confirm

只在 P5 通过后打开。

---

# 10. 统一 artifact

## 10.1 CSV / JSON

```text
v142_project_progress.csv
v142_code_path_manifest.csv
v142_loss_interface_audit.csv
v142_functional_metric_state_trace.csv
v142_per_example_gradient_stats.csv
v142_fms_update_trace.csv
v142_mlp_fms_results.csv
v142_basis_substrate_status.csv
v142_basis_family_telemetry.csv
v142_basis_fms_results.csv
v142_linec_audit.csv
v142_controls.csv
v142_failure_table.csv
v142_route_decision.json
v142_no_go_boundary.md
v142_next_hypothesis_queue.md
```

## 10.2 关键字段

```text
method
family
candidate_id
dataset
seed
loss_interface
train_step
phase
fms_state_type
snr_active_fraction
utility_mean
utility_p10
utility_p90
state_c_mean
state_c_p90
state_update_norm
adamw_update_norm
fms_update_norm
cos_fms_adamw
basis_safety_rejection_fraction
source_vs_adamw
source_vs_best_control
AUCtime_ratio
CEp99_delta
NLL_delta
ECE_delta
CouplingR2_delta
NoiseSignalLeak_delta
ReservoirRatio_delta
step_time_ratio
memory_ratio
failure_reason
```

---

# 11. 必须可视化

```text
fig_v142_progress_by_line.svg
fig_v142_fms_state_trace_by_method.svg
fig_v142_mlp_vs_basis_fms.svg
fig_v142_basis_family_status_heatmap.svg
fig_v142_substrate_pareto.svg
fig_v142_snr_utility_distribution.svg
fig_v142_fms_update_cosine.svg
fig_v142_linec_signal_reservoir_dashboard.svg
fig_v142_tail_calibration_dashboard.svg
fig_v142_failure_taxonomy.svg
```

---

# 12. 决策路由

```text
R0-ImplementationViolation
R1-NoEfficientSubstrate
R2-SubstrateExistsFMSNoSyntheticGain
R3-GenericFunctionalOnlyKANSpecificNotEstablished
R4-FMSNoGoCurrentDefinition
R5-BasisSubstrateButFunctionalBlocked
R6-NonRATStillWorkspaceOrTaskBlocked
S1-EfficientSubstrateFound
S2-FMSGenericPositive
S3-KANSpecificFMSPositive
S4-BasisFamilyNearPass
S5-OfficialFunctionalSuccess
```

---

# 13. Codex 自动尝试规则

## 13.1 如果 MLP-FMS 不过

Codex 先尝试：

```text
1. FMS state beta in {0.9,0.99}；
2. phase schedule on/off；
3. layerwise vs parameterwise；
4. lowrank only if layerwise有 positive；
5. 不允许回到 CE-tail direction。
```

如果仍不过：

```text
route = R4-FMSNoGoCurrentDefinition
```

## 13.2 如果 MLP-FMS 过但 Rational-FMS 不过

Codex 必须做：

```text
1. Rational FMS rejection audit；
2. role signal mass audit；
3. numerator/denominator/readout update separation；
4. denominator safety loosen/tighten diagnostic；
5. group granularity g8/g16。
```

不能立刻回到 basis architecture search。

## 13.3 如果 Rational 过但 RBF/Cheby/Fourier/Wavelet 不过

Codex 必须输出：

```text
basis_specific_blocker_matrix
```

并对每个 family 执行其 family-specific fallback，不得写成“all failed”。

## 13.4 如果某 basis substrate 不过

Codex 不准跑该 basis 的 FMS official proof，只能跑：

```text
substrate repair
telemetry diagnostic
minimal smoke
```

## 13.5 如果 FMS 改善 task 但 LineC / tail 坏

Codex 必须先做：

```text
1. audit-only failure localization；
2. safety projection rejection / leak source；
3. role/basis state over-amplification audit；
4. reduced-plasticity phase schedule。
```

不能用 CEp99 / LineC 作为 direction target。

---

# 14. 本轮成功标准

最小成功：

```text
1. FMS implementation contract pass；
2. MLP-FMS 或 Rational-FMS 至少一个 synthetic 5/7 positive；
3. All-basis substrate map 完成；
4. 每个 active basis 有明确 status。
```

重要成功：

```text
Rational-FMS 3x3 real triage pass
且
MLP-FMS 对照允许分解 KAN-specific gain。
```

重大成功：

```text
至少一个 non-Rational basis 通过 substrate gate，
并且它的 basis-specific FMS synthetic 5/7 pass。
```

Official success：

```text
同一 basis 上：
Basis + FMS
>
Basis + AdamW / controls

满足 task / AUC / tail / calibration / LineC / efficiency 全门槛。
```

---

# 15. 最终判断

v14.2 的核心不是在 v14.1 和 v14.0 之间二选一。

它的核心是：

$$
\boxed{
\text{Functional update 作为突破口继续主投，}
\text{但所有 active basis substrate 必须并行推进，}
\text{因为 functional update 必须在多个可控 basis 上证明其机制。}
}
$$

如果 functional update 只在 MLP 上成立，它是 generic optimizer。  
如果只在 Rational 上成立，它可能是 Rational-specific。  
如果能在 RBF/FastKAN、Chebyshev、Fourier、Wavelet 中至少再打开一个，它才开始接近 **KAN basis-general functional geometry advantage**。

因此 v14.2 的真正目标是：

```text
Functional-first；
All-basis parallel；
Geometry-audited；
No CE-specific trick；
No label-informed init；
No token search。
```
