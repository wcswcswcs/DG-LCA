# DG-KAN v12.34.2：All-Basis Efficient Substrate + Basis-Specific Functional Repair 完整计划

> 版本：v12.34.2 execution plan  
> 目的：修正 v12.34.1 过度聚焦 Rational 的问题。Rational、Chebyshev、Fourier、RBF/FastKAN、Wavelet 都应采用同一个研究范式：**基函数先成为高效率 substrate，不要求它单独解决全部 task / AUC / tail / LineC；随后用 basis-specific、loss-agnostic functional update 修复训练健康与几何问题。**  
> B-spline：按当前决策继续 frozen，不进入 active budget。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no fake / proxy / CPU offload；不使用 label-informed initialization；functional direction 不使用 label、CE vector、permuted-label CE、validation/test、future outcome、query batch 或 LineC hard target。CE / NLL / ECE / CEp99 只能作为审计和坏化约束，不能作为方向源。

---

# 0. 项目总目标与当前进展

## 0.1 总目标

DG-KAN 的总目标不是让某个 toy 数据集指标好看，也不是只证明某个 KAN primitive 比 MLP 慢一点但 accuracy 高一点。项目最终要证明：

$$
\boxed{
\text{label-free strict FC-PureKAN base + loss-agnostic functional update}
>
\text{same base + ordinary backprop / AdamW controls}
}
$$

并且这个系统必须同时满足：

```text
1. 表达力不打折，最好比同规模 MLP 略强；
2. forward / backward / step / memory 与 MLP 可比；
3. 训练轨迹健康，AUC-step / AUC-time 不劣于 MLP；
4. 几何健康，train-probe coupling、signal/reservoir/noise、tail、calibration 不坏；
5. functional update 的收益必须打过 NoOp / RandomMatchedNorm / AdamWParallel / SNR-only / MLP analog controls；
6. functional update 必须 loss-agnostic，不能针对 CE / NLL / ECE / CEp99 设计方向。
```

## 0.2 当前关键事实

当前历史 FHQ / B320-current 证明过 PureKAN 系统效率可以很强，但由于 label-informed trainprobe initialization 已经被禁止，它只能作为历史 reference，不再作为 official base。v12.26.1 以后，官方路线必须是 label-free。

v12.31-v12.33 的核心进展是 classic no-BSpline basis 线已经推进到更深的层面：

```text
Rational:
  workspace 和 telemetry 已经明显打开，当前最接近成为 efficient substrate。
  但 task / AUC / LineC / tail 不能同位。

Chebyshev / Fourier:
  exact no-materialize kernel correctness 与 A4 smoke 已过。
  但 full-step incremental memory lifetime 未闭合。

RBF / FastKAN:
  compact/fused efficiency 有过路径，但 expression / compact capacity 仍不足。

Wavelet:
  local/hat wavelet efficiency 有过路径，但 task / LineC 不稳。

MLP functional:
  M-G/M-H/M-I family 没有 generic positive，说明当前 loss-agnostic functional update 不是一个容易直接套到 MLP 上的通用 optimizer trick。
```

因此 v12.34.2 的核心修正是：

$$
\boxed{
\text{不仅 Rational，所有 active basis 都应进入：efficient substrate + basis-specific functional repair。}
}
$$

这意味着不能再要求每个 basis 单独把全部 task / AUC / tail / LineC 都修好后才允许 functional update。相反，我们要将问题拆成两层：

```text
第一层：basis substrate。
  基函数负责高效率、表达力、最小可训练性、不灾难性坏化。

第二层：basis-specific functional repair。
  functional update 负责在该 substrate 上修复 AUC / tail / LineC / calibration / signal-reservoir co-location。
```

---

# 1. 为什么不能只对 Rational 这样做

上一版 v12.34.1 把 Rational 定义为 efficient substrate，并计划让 Rational-specific functional update 修复 tail/AUC/LineC。这是正确的，但不应只限于 Rational。

原因有三点。

## 1.1 基函数本体与 functional repair 的职责应统一

如果我们要求 Rational 自己不解决 tail/AUC/LineC，而允许 functional update 修，那么同样应该允许 Chebyshev / Fourier / RBF / Wavelet 这样做。否则实验标准不一致：

```text
Rational:
  efficiency + minimal trainability 即可进入 functional repair。

其他 basis:
  必须单独成为 healthy base 才能进入 functional repair。
```

这会不公平地压低其他 family。

## 1.2 当前 blocker 很多不是 basis 数学表达力，而是训练健康

Rational 的问题是 task / LineC / tail 不同位；Chebyshev / Fourier 的问题是 exact kernel 已过但 full-step lifetime / training dynamics 未闭合；Wavelet 的问题是 local support 能跑但 task geometry 不稳；RBF/FastKAN 的问题是 compact capacity 和 occupancy 没形成表达覆盖。

这些问题不一定应该全由 basis 本体解决。functional update 的研究目标本来就是改善训练几何。如果 basis 已经提供可训练坐标，functional repair 就应该被允许介入。

## 1.3 这能更好地区分“基函数贡献”和“functional update 贡献”

如果每个 basis 都先作为 substrate，再加 basis-specific functional repair，我们就能比较：

```text
basis-only contribution:
  Basis + AdamW vs MLP + AdamW

functional contribution:
  Basis + Functional vs Basis + AdamW

basis-specific synergy:
  (Basis + Functional) - (Basis + AdamW)
  与
  (MLP + Functional) - (MLP + AdamW)
```

这正好符合项目原始目标：区分好处来自 KAN 还是 functional update。

---

# 2. 新的统一判断框架

## 2.1 三层 gate

所有 active no-BSpline basis 都采用三层 gate。

### Gate S：Efficient Substrate Gate

这是进入 basis-specific functional repair 的门槛。它不要求 basis 自己成为 official healthy base，只要求：

$$
T_{step}/T_{MLP} \leq 1.50
$$

$$
M_{peak}/M_{MLP} \leq 1.50
$$

$$
ExpressionGate = 1
$$

$$
\Delta Acc_{mean} \geq -0.05
$$

$$
\Delta Acc_{worst} \geq -0.10
$$

并且没有灾难性数值问题：

```text
no NaN / Inf
no exploding logits
CEp99_delta <= +5.0 exploratory
LineC measured available
functional telemetry available
```

解释：Substrate gate 是“可承载 functional repair”的门，不是 official base gate。

### Gate H：Healthy Base Gate

这是 basis 单独成为 base 的门槛：

$$
T_{step}/T_{MLP} \leq 1.25
$$

$$
M_{peak}/M_{MLP} \leq 1.25
$$

$$
\Delta Acc_{mean} \geq 0
$$

$$
\Delta Acc_{worst} \geq -0.003
$$

$$
AUC_{time}/AUC_{time,MLP} \leq 1.0
$$

$$
CEp99_{basis} \leq CEp99_{MLP} + 0.05
$$

$$
LineC_{basis} \text{ passes majority or all-pass gate.}
$$

Gate H 不是 functional repair 的前置条件，但如果 basis-only 通过，说明该 family 本身就是 base candidate。

### Gate F：Functional Repair Gate

basis-specific functional update 成功必须满足：

$$
\Delta Acc_{func-vs-base} \geq -0.003
$$

$$
AUCtime_{func} \leq AUCtime_{base}
$$

$$
CEp99_{func} \leq CEp99_{base}+0.05
$$

$$
NLL_{func} \leq NLL_{base}+0.02
$$

$$
ECE_{func} \leq ECE_{base}+0.02
$$

$$
CouplingR^2_{func} \geq CouplingR^2_{base}+0.02
$$

$$
NoiseSignalLeak_{func} \leq NoiseSignalLeak_{base}-0.01
$$

$$
RealSignalReservoirRatio_{func} \leq RealSignalReservoirRatio_{base}-0.01
$$

并且必须满足：

```text
beats NoOpMatchedOverhead
beats RandomMatchedNorm
beats AdamWParallelDirection
beats SNR-only
beats GeometryOnlyNoSNR
beats MLPAnalog or MLP functional matched diagnostic, where applicable
```

### Gate O：Official Basis + Functional Success

最终 family success 可以有两种路径：

```text
Path 1: Basis-only success
  Gate H pass。

Path 2: Substrate + functional success
  Gate S pass + Gate F pass。
```

也就是说，basis 不一定要单独成为 healthy base；它可以作为 efficient substrate，由 functional repair 完成训练健康闭合。

---

# 3. 当前各 family 的统一定位

## 3.1 Rational

当前状态：最接近 substrate。workspace 和 telemetry 已经打开，但 task/AUC/LineC/tail 不同位。

Rational 的 substrate 目标：

```text
1. workspace strict pass；
2. expression pass；
3. mean task signal 不灾难；
4. denominator / derivative / tangent telemetry 可用；
5. CEp99 / AUC 不灾难到 functional 无法修。
```

Rational 的 functional repair 应聚焦：

```text
denominator-slope guard；
r' / r'' tangent trust region；
group diversity transport；
readout-rational decoupling；
tangent condition stabilization；
LineC-stable tangent mixing。
```

不能做：

```text
用 CEp99 梯度修 tail；
用 NLL / ECE 直接设计方向；
按数据集调 denominator；
把 output-geometry token 小网格当机制。
```

## 3.2 Chebyshev

当前状态：exact no-materialize correctness / A4 smoke 已过；主要 blocker 是 full-step incremental memory lifetime 和 task trajectory。

Chebyshev 的 substrate 目标：

```text
1. true fused recurrence path；
2. no dense basis materialization；
3. degree-energy telemetry；
4. expression pass；
5. task 不灾难。
```

Chebyshev telemetry：

```text
degree_energy_k；
high_degree_energy_ratio；
recurrence_max_abs；
tangent_condition；
top_eigen_share；
LineC pass count；
NoiseSignalLeak；
RealSignalReservoirRatio。
```

Chebyshev-specific functional repair：

```text
degree-energy damping；
late-enable high-degree；
role-wise degree energy cap；
Chebyshev tangent trust region；
high-degree noise-leak veto；
low-degree signal reinforcement。
```

## 3.3 Fourier

当前状态：exact no-materialize correctness / A4 smoke 已过，但 low-frequency compact expression / memory lifetime / task geometry 仍不足。

Fourier substrate 目标：

```text
1. fused sincos or recurrence-like low-frequency path；
2. no dense frequency materialization；
3. expression battery pass；
4. frequency energy telemetry；
5. NoiseSignalLeak 不灾难。
```

Fourier telemetry：

```text
frequency_energy_k；
high_freq_energy_ratio；
phase_drift；
spectral_entropy；
signal_band_mass；
noise_band_mass；
LineC coupling by frequency band。
```

Fourier-specific functional repair：

```text
frequency-band damping；
late-enable high-frequency residual；
phase-stability correction；
high-frequency noise-leak veto；
low-frequency signal-channel transport。
```

注意：不能简单加高频来追 expression。高频必须通过 NoiseSignalLeak / CEp99 / LineC audit。

## 3.4 RBF / FastKAN

当前状态：compact/fused L3 path 有过，主要 blocker 是 A4 expression / compact capacity。

RBF/FastKAN substrate 目标：

```text
1. compact active-center path；
2. K_active small but expression battery 不塌；
3. center occupancy 不死；
4. width condition 安全；
5. OOG fraction 可控。
```

RBF telemetry：

```text
active_center_entropy；
dead_center_fraction；
out_of_grid_fraction；
width_p01 / width_p99；
center_usage_by_class-free window；
exp_count_per_sample；
center-width condition；
LineC pass count。
```

RBF-specific functional repair：

```text
center occupancy rebalance；
width smoothing / width cap；
OOG boundary repair；
local curvature smoothing with compensation；
active-center diversity transport；
noise-leak veto for over-localized centers。
```

## 3.5 Wavelet

当前状态：local/hat wavelet path 有 efficiency，某些 A4 过，task / LineC 不稳。

Wavelet substrate 目标：

```text
1. local support kernel；
2. scale / shift telemetry；
3. local tail coverage；
4. task 不灾难；
5. LineC 不灾难。
```

Wavelet telemetry：

```text
active_support_count；
scale_energy；
scale_dead_fraction；
local_tail_coverage；
scale_condition；
support_overlap_entropy；
LineC by local support group；
NoiseSignalLeak local/global split。
```

Wavelet-specific functional repair：

```text
scale-energy balance；
local support occupancy repair；
local tail correction；
scale diversity transport；
support-overlap guard；
noise-leak veto for overly local basis。
```

---

# 4. 实验总流程

v12.34.2 分为七条线。

```text
Line D-S: substrate gate for all active basis families。
Line D-T: family-specific telemetry。
Line B-F: basis-specific functional repair。
Line D-K: kernel / lifetime repair。
Line M: MLP functional closure / comparison。
Line C: LineC audit and geometry diagnostics。
Line R/Z: code review, provenance, finalizer, no-go boundary。
```

---

# 5. Line D-S：All-Basis Substrate Gate

## 5.1 目标

不是找一个 family，而是让每个 active family 都进入明确状态：

```text
SubstratePass
SubstrateNearPass
EfficiencyBlocked
ExpressionBlocked
TaskCatastrophicBlocked
TelemetryMissingBlocked
Frozen
```

## 5.2 候选

```text
D-RAT: Rational D-RAT24..D-RAT33 plus new D-RAT34..D-RAT39。
D-CHE: Chebyshev D-CHE12..D-CHE15 plus D-CHE16..D-CHE19。
D-FOU: Fourier D-FOU12..D-FOU15 plus D-FOU16..D-FOU19。
D-RBF: RBF/FastKAN D-RBF11 plus D-RBF12..D-RBF16。
D-WAV: Wavelet D-WAV10 plus D-WAV11..D-WAV15。
```

B-spline 不进入 active candidates。

## 5.3 记录字段

```text
family
candidate_id
dataset
seed
step_ratio_vs_mlp
raw_memory_ratio_vs_mlp
incremental_memory_ratio_vs_mlp
expression_pass
mean_delta_vs_mlp
worst_delta_vs_mlp
AUC_step_ratio_vs_mlp
AUC_time_ratio_vs_mlp
CEp99_delta_vs_mlp
NLL_delta_vs_mlp
ECE_delta_vs_mlp
LineC_pass_count
LineC_total
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
telemetry_available
substrate_gate_pass
substrate_near_pass
healthy_base_gate_pass
failure_reason
```

## 5.4 Substrate pass 标准

$$
step\_ratio \le 1.50
$$

$$
raw\_memory\_ratio \le 1.50
$$

$$
incremental\_memory\_ratio \le 2.50
$$

$$
ExpressionPass = 1
$$

$$
mean\_delta \ge -0.05
$$

$$
worst\_delta \ge -0.10
$$

$$
CEp99\_delta \le 5.0
$$

$$
telemetry\_available = 1
$$

Substrate near-pass 可放宽 memory，但不能放宽 expression 与 telemetry：

$$
step\_ratio \le 1.75
$$

$$
incremental\_memory\_ratio \le 3.50
$$

$$
ExpressionPass = 1
$$

$$
telemetry\_available = 1
$$

## 5.5 失败后 Codex 自动尝试

```text
EfficiencyBlocked:
  run lifetime waterfall；
  try recompute derivative；
  delay optimizer state allocation；
  fuse readout grad；
  rerun exact same task probe only after workspace fix。

ExpressionBlocked:
  increase compact capacity within memory budget；
  add low-rank residual basis；
  run expression battery E1/E6/E8；
  if expression fix breaks efficiency, route = ExpressionEfficiencyConflict。

TaskCatastrophicBlocked:
  run shorter LR / slow-start / residual-scale warmup；
  record whether task catastrophe is early loss explosion or late tail drift；
  do not tune per dataset。

TelemetryMissingBlocked:
  implement family telemetry before any functional repair。
```

---

# 6. Line D-T：Family-Specific Telemetry

## 6.1 目标

functional repair 不能没有观测对象。每个 basis family 必须暴露自己的 loss-agnostic internal geometry telemetry。

## 6.2 Rational telemetry

记录：

```text
den_p01_batch
den_p99_batch
den_condition
r_prime_p95
r_prime_p99
r_double_prime_p95
r_double_prime_p99
tangent_condition
tangent_rank
tangent_top_eigen_share
group_function_diversity
readout_rational_coupling
per_group_update_norm
per_sample_rational_displacement
```

## 6.3 Chebyshev telemetry

记录：

```text
degree_energy_k0_to_kK
high_degree_energy_ratio
recurrence_max_abs
recurrence_overflow_rate
degree_tangent_condition
degree_top_eigen_share
degree_usage_entropy
role_degree_energy
```

## 6.4 Fourier telemetry

记录：

```text
frequency_energy_k
high_freq_energy_ratio
low_freq_signal_mass
high_freq_noise_mass
phase_drift
spectral_entropy
frequency_tangent_condition
bandwise_LineC
```

## 6.5 RBF/FastKAN telemetry

记录：

```text
active_center_entropy
dead_center_fraction
out_of_grid_fraction
center_width_condition
width_p01
width_p99
center_update_norm
local_curvature_proxy
center_occupancy_by_window
```

## 6.6 Wavelet telemetry

记录：

```text
active_support_count
support_overlap_entropy
scale_energy
scale_dead_fraction
scale_condition
local_tail_coverage
local_global_signal_split
support_group_LineC
```

## 6.7 通用 telemetry 相关性分析

对每个 family，分析 telemetry 与审计指标的关系：

```text
Spearman(telemetry, CEp99_delta)
Spearman(telemetry, AUC_time_ratio)
Spearman(telemetry, LineC_pass_count)
Spearman(telemetry, NoiseSignalLeak)
Spearman(telemetry, RealSignalReservoirRatio)
```

必须生成：

```text
fig_family_telemetry_vs_tail.svg
fig_family_telemetry_vs_auc.svg
fig_family_telemetry_vs_linec.svg
fig_family_telemetry_correlation_heatmap.svg
```

---

# 7. Line B-F：Basis-Specific Functional Repair

## 7.1 总原则

Functional repair 只允许在 SubstratePass 或 SubstrateNearPass 上运行。它的职责是修复健康问题，而不是让一个完全坏的 basis 起死回生。

Functional direction 必须：

```text
loss_agnostic_direction = 1
label_used_for_direction = 0
ce_vector_used_for_direction = 0
validation_used_for_commit = 0
test_used_for_commit = 0
query_batch_used_for_direction = 0
linec_hard_target_used_for_direction = 0
dataset_name_branch = 0
```

## 7.2 通用 functional repair 框架

对 base update：

$$
\theta_{t+1}=\theta_t+\Delta\theta_{task}+\lambda_t\Delta\theta_{func}
$$

其中：

```text
Delta theta task:
  ordinary AdamW / manual AdamW。

Delta theta func:
  basis-specific, telemetry-derived, loss-agnostic maintenance event。

lambda_t:
  determined by train-stream precommit geometry safety, not by CE/label。
```

## 7.3 Family repair candidates

### Rational functional repair

```text
B-RAT1-TangentTrustRegionNoCE
B-RAT2-DenSlopeGuardNoCE
B-RAT3-GroupDiversityTransport
B-RAT4-ReadoutRationalDecouple
B-RAT5-LineCStableTangentMix
```

### Chebyshev functional repair

```text
B-CHE1-DegreeEnergyDamping
B-CHE2-HighDegreeLateEnable
B-CHE3-DegreeTangentTrustRegion
B-CHE4-RoleDegreeEnergyCap
B-CHE5-HighDegreeNoiseLeakVeto
```

### Fourier functional repair

```text
B-FOU1-FrequencyBandDamping
B-FOU2-PhaseStabilityCorrection
B-FOU3-LowFreqSignalTransport
B-FOU4-HighFreqNoiseLeakVeto
B-FOU5-LateEnableHighFreqResidual
```

### RBF/FastKAN functional repair

```text
B-RBF1-CenterOccupancyRebalance
B-RBF2-WidthConditionGuard
B-RBF3-OOGBoundaryRepair
B-RBF4-LocalCurvatureSmoothCompensated
B-RBF5-ActiveCenterDiversityTransport
```

### Wavelet functional repair

```text
B-WAV1-ScaleEnergyBalance
B-WAV2-LocalSupportOccupancyRepair
B-WAV3-LocalTailCoverageGuard
B-WAV4-SupportOverlapEntropyGuard
B-WAV5-ScaleDiversityTransport
```

## 7.4 Functional repair P3 cloned audit

必须记录：

```text
family
base_candidate_id
functional_candidate_id
dataset
seed
window
lambda
source_vs_noop
source_vs_random
source_vs_adamwparallel
source_vs_snr
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
AUC_time_delta_proxy
CEp99_delta_audit
NLL_delta_audit
ECE_delta_audit
telemetry_delta_summary
control_gap
p3_pass
p3_fail_reason
```

## 7.5 Functional repair P4 short-run

只有 P3 过，才允许 P4。P4 必须比较：

```text
Basis + AdamW
Basis + NoOpMatchedOverhead
Basis + RandomMatchedNorm
Basis + AdamWParallelDirection
Basis + SNR-only
Basis + BasisSpecificFunctionalRepair
MLP + matched functional, if relevant
```

P4 gate：

$$
source\_vs\_best\_control \ge 0.005
$$

$$
AUCtime_{func} \le AUCtime_{base}
$$

$$
CEp99_{func} \le CEp99_{base}+0.05
$$

$$
LineC_{func} \ge LineC_{base}
$$

$$
train\_shuffle\_robust = 1
$$

---

# 8. Line D-K：Kernel / Lifetime Repair

## 8.1 目标

对 Chebyshev / Fourier / RBF / Wavelet 来说，不能只说“basis 不行”。v12.32-v12.33 显示 exact correctness 可能已经过，但 full-step incremental memory lifetime 仍失败。因此必须继续 kernel/lifetime 修复。

## 8.2 记录字段

```text
family
candidate_id
basis_activation_live_bytes
basis_derivative_live_bytes
readout_grad_live_bytes
coeff_grad_live_bytes
optimizer_state_live_bytes
temporary_workspace_live_bytes
linec_hook_live_bytes
largest_live_tensor_name
largest_live_tensor_shape
raw_peak_ratio
incremental_peak_ratio
step_ratio
lifetime_gate_pass
```

## 8.3 必须可视化

```text
fig_lifetime_waterfall_by_family.svg
fig_largest_live_tensor_table.md
fig_incremental_memory_vs_step.svg
fig_readout_grad_recompute_effect.svg
```

## 8.4 失败后 Codex 自动尝试

```text
basis_derivative too large:
  recompute derivative in backward；
  don't materialize derivative tensor。

readout_grad too large:
  fused readout grad；
  chunked reduction；
  recompute activation。

optimizer state too large:
  delayed allocation；
  foreach disabled audit；
  low precision optimizer state audit。

temporary workspace too large:
  static workspace reuse；
  single buffer scheduler；
  block-level streaming。
```

---

# 9. Line M：MLP Functional Closure / Comparison

## 9.1 目标

继续验证 functional update 是否是通用机制。

当前 MLP functional family 没有 generic positive。因此 v12.34.2 只做 closure，不继续扩大 objective 网格。

## 9.2 必跑

```text
M-J1-CloneProbeCovarianceTransportV2
M-J2-UnlabeledOptimizerObservableTransportV2
M-J3-ArchitectureNeutralSNRTransportV2
```

## 9.3 判断

如果仍然 0 pass，则记录：

```text
MLPFunctionalNoGo_CurrentLossAgnosticObservableFamily_v2
```

这不代表 functional update 永远不可能在 MLP 上 work，只代表当前 observable family 不成立。之后 functional 预算应转向 basis-specific repair。

---

# 10. Line C：Manifold-Channel Geometry Diagnostics

继续作为审计，不作为方向源。

## 10.1 记录指标

```text
CouplingR2
CouplingCorr
NoiseSignalLeak
RealSignalReservoirRatio
KernelDrift
CEp99
NLL
ECE
MarginP10
signal_effective_rank
reservoir_fraction
signal_top_eigen_share
```

## 10.2 解释规则

```text
Good substrate:
  efficient + expression + non-catastrophic task + LineC measured。

Good functional repair:
  task/control non-harm + LineC improvement + tail/calibration non-harm。

Bad candidate:
  task gain but NoiseSignalLeak increases;
  LineC gain but tail explodes;
  output calibration improves but task/LineC collapses。
```

---

# 11. 统一 artifacts

必须写出：

```text
v12342_family_substrate_summary.csv
v12342_family_telemetry.csv
v12342_family_telemetry_correlations.csv
v12342_basis_functional_p3.csv
v12342_basis_functional_p4.csv
v12342_kernel_lifetime_waterfall.csv
v12342_mlp_functional_closure.csv
v12342_linec_audit.csv
v12342_failure_table.csv
v12342_route_decision.json
v12342_next_hypothesis_queue.md
v12342_code_review_packet.zip
```

---

# 12. 必须生成的图

```text
fig_progress_by_family.svg
fig_family_substrate_pareto.svg
fig_family_telemetry_vs_tail.svg
fig_family_telemetry_vs_linec.svg
fig_functional_control_gap_by_family.svg
fig_kernel_lifetime_waterfall_by_family.svg
fig_mlp_functional_no_go.svg
fig_linec_task_tail_colocation.svg
fig_route_dashboard.svg
```

---

# 13. Route 定义

```text
S1-AnyFamilySubstratePass:
  至少一个 active basis 通过 Substrate Gate。

S2-AnyFamilyFunctionalRepairP3:
  至少一个 basis-specific functional repair 通过 P3。

S3-AnyFamilyFunctionalRepairP4:
  至少一个 basis-specific functional repair 通过 P4。

S4-AnyFamilyHealthyBase:
  至少一个 family basis-only 通过 Healthy Base Gate。

S5-OfficialBasisFunctionalSuccess:
  label-free strict FC-PureKAN base/substrate + functional repair 通过 official controls。

R1-NoSubstratePass:
  没有 active family 通过 substrate gate。

R2-SubstrateButNoFunctionalRepair:
  有 substrate，但 no functional P3/P4。

R3-KernelLifetimeBlocked:
  主要失败来自 full-step lifetime。

R4-TaskLineCTailNotColocated:
  有 task 或 LineC 单项信号，但不能同位。

R5-MLPFunctionalNoGo_BasisSpecificOnly:
  MLP functional closure 失败，后续 functional 预算转向 basis-specific。
```

---

# 14. 下一步执行优先级

## 14.1 第一优先级：All-basis substrate map

先跑所有 active family 的 substrate gate，不再只跑 Rational。

## 14.2 第二优先级：Family telemetry

没有 telemetry 的 family 不允许进入 functional repair。

## 14.3 第三优先级：Basis-specific functional repair

只在 substrate / near-substrate 上运行，不在完全坏的 candidate 上浪费预算。

## 14.4 第四优先级：Kernel lifetime repair

Chebyshev / Fourier / RBF / Wavelet 的 lifetime blocker 必须继续修。

## 14.5 第五优先级：MLP functional closure

不再主投 MLP functional，只完成 no-go / transfer 判断。

---

# 15. 最终总结

v12.34.2 的核心修正是：

$$
\boxed{
\text{所有 active basis 都应先作为 efficient substrate，}
\text{再由 basis-specific functional update 修复训练健康与几何。}
}
$$

这比只让 Rational 进入 functional repair 更一致、更公平，也更符合项目目标。基函数的第一职责是提供高效、表达力足够、可训练的坐标系统；functional update 的职责是让这个坐标系统在训练中更健康、更稳定、更少噪声泄漏、更好地形成 signal channel。

因此下一轮不应继续平均撒网，也不应只盯 Rational。应按 family-specific blocker 并行推进：

```text
Rational:
  substrate 已最接近，进入 internal telemetry + functional repair。

Chebyshev / Fourier:
  修 lifetime，同时准备 degree/frequency functional repair。

RBF / FastKAN:
  修 compact expression，同时准备 center-width functional repair。

Wavelet:
  修 task/LineC，同时准备 scale/support functional repair。

MLP:
  做 functional closure，不再主投。
```

最终成功不要求每个 basis 单独健康，而是要求：

$$
\boxed{
\text{至少一个 label-free strict FC-PureKAN substrate + loss-agnostic basis-specific functional repair}
\text{打过同 base AdamW 和 matched controls。}
}
$$
