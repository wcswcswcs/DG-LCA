# DG-KAN v12.36：Substrate-Health First + Basis-Channel Co-location Functional Repair 完整计划

> 生成时间：2026-05-27  
> 基于：v12.35 `AllBasisSubstrateHealth FunctionalColocation` 结果复盘  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；不使用 label-informed initialization；functional direction 不使用 label、CE vector、permuted-label CE、validation/test、future outcome、query batch 或 LineC hard target；CE / NLL / ECE / CEp99 只能作为审计与坏化约束，不能作为 functional direction source。

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是找到某个局部好看的 geometry score。项目总目标是：

$$
\boxed{
\text{构建 label-free strict FC-PureKAN base/substrate，}
\text{并用 loss-agnostic functional update 获得比 ordinary backprop 更好的训练几何和模型。}
}
$$

最终要证明的是：

$$
\boxed{
\text{KAN base/substrate} + \text{functional update}
>
\text{same KAN base/substrate} + \text{ordinary AdamW/backprop controls}
}
$$

其中“更好”必须同时满足：

```text
1. 表达力不打折；
2. forward / backward / step / memory 接近 MLP；
3. 任务收敛轨迹不差，AUC-step / AUC-time 不输；
4. LineC / manifold-channel geometry 更健康；
5. functional update 的收益不能被 NoOp / RandomMatchedNorm / AdamWParallel / SNR-only / MLP analog controls 解释；
6. functional direction 必须 loss-agnostic，不允许针对 CE / NLL / ECE / CEp99 设计方向。
```

当前需要特别强调：

```text
B320-current / FHQ 历史上证明了 strict PureKAN 工程效率可以非常强，
但它依赖 label-informed trainprobe initialization，已经不能作为 official label-free claim。
```

因此当前 official 主线是：

$$
\boxed{
\text{label-free classic no-BSpline basis substrate}
+
\text{basis-specific functional repair}
}
$$

B-spline 继续 frozen，不参与 active budget。

---

# 1. v12.35 当前状态与各线完成度

下面的百分比不是官方字段，而是根据 gate 通过情况、代码审计完整度、机制清晰度和距离 official success 的综合估计。括号内为相对 v12.34.2 的变化。

| 线 | v12.34.2 | v12.35 | 变化 | 当前判断 |
|---|---:|---:|---:|---|
| Line R：代码 / provenance / finalizer 审计 | 97% | 98% | +1 | v12.35 required artifact、implementation readback、code review surface 基本闭合。 |
| Historical FHQ / B320-current 工程能力 | 85% frozen | 85% frozen | 0 | 历史参考，不能 official。 |
| Label-free FHQ / A-DYN monitor | 26% | 25% | -1 | 仍无 label-free near-anchor，只保留低预算 monitor。 |
| Line C：Manifold-Channel 几何审计 | 87% | 88% | +1 | 审计更稳定，能暴露 task/LineC 分裂，但不能作为 direction source。 |
| Precommit / loss-agnostic value source | 20% | 18% | -2 | response dictionary 0 pass，当前 value source 更不乐观。 |
| Generic MLP functional update | 23% | 23% | 0 | M-J v3 仍 0 pass，generic observable family no-go。 |
| KAN / basis-specific functional official | 8% | 7% | -1 | P3/P4 仍 0 pass；不是没有执行，而是没有有效 channel co-location。 |
| Classic no-BSpline portfolio 总体 | 67% | 68% | +1 | substrate map 和 health gate 更清楚；能力没有质变。 |
| Rational family | 84% | 85% | +1 | 唯一稳定 substrate family；11/44 pass 全来自 Rational。 |
| Chebyshev family | 46% | 44% | -2 | 无 substrate-health pass；仍缺 task-health row。 |
| Fourier family | 49% | 47% | -2 | lifetime/task-health repair后仍 task/AUC/LineC 崩。 |
| RBF / FastKAN family | 45% | 43% | -2 | LineC 局部有信号，但 task/AUC 严重坏化。 |
| Wavelet family | 42% | 40% | -2 | task/AUC/LineC 坏化，不能进入 functional P3。 |
| Response dictionary / co-location modeling | 新增 | 22% | +22 | artifact 已建立，但 270 rows 中 0 pass。 |
| Non-RAT task-health repair | 新增 | 20% | +20 | 40 rows 0 pass，只证明问题边界。 |
| 整体 next-gen MLP claim | 51%-55% | 50%-54% | -1 | 路线更清楚，但 official base/functional 仍未闭合。 |

## 1.1 v12.35 的最终状态

v12.35 最终状态是：

```text
route = R2-SubstrateButNoFunctionalRepair
route_detail = R4-RationalOnlySubstrate also applies;
               R3-NonRATLifetimeTaskHealthSplit also applies;
               R5-MLPFunctionalNoGo also applies;
               ResponseDictionaryNoGo
minimum_success = S1-SubstrateHealthPass
official_success_reached = 0
promotion_allowed = 0
basis_substrate_health_rows = 44
substrate_health_gate_pass_count = 11
substrate_health_near_pass_count = 16
healthy_base_gate_pass_count = 0
nonrat_substrate_health_pass_count = 0
basis_response_dictionary_rows = 270
response_dictionary_pass_count = 0
basis_functional_p3_pass_count = 0
basis_functional_p4_pass_count = 0
nonrat_task_health_repair_pass_count = 0
mlp_functional_mj_v3_pass_rows = 0
provenance_violation_count = 0
```

这说明：

$$
\boxed{
\text{v12.35 只达到 S1：存在 active basis substrate。}
}
$$

还没有达到：

```text
S2: basis-specific functional P3 pass；
S3: basis-specific functional P4 pass；
S4: Non-RAT task-health substrate；
S5: official functional success。
```

---

# 2. v12.35 独立分析

## 2.1 有进展，但不是能力突破

v12.35 的进展主要是“判定体系”和“失败边界”的进展，而不是模型能力的进展。

它做对了几件事：

```text
1. 用 v12.35 Gate S/H 重新审计 substrate-health，而不是只看 workspace pass。
2. 证明 substrate-health pass 仍然只来自 Rational。
3. 建立了 basis response dictionary，但没有把旧 P3 行伪装成新训练成功。
4. 重算 basis-functional P3/P4 gate，结果仍然 0 pass。
5. 执行 Non-RAT task-health repair audit，结果仍然 0 pass。
6. MLP M-J v3 closure monitor 仍然 0 pass。
7. provenance、manifest、code review surface 均通过。
```

所以，这轮没有乱报成功，是好事。但它也说明：继续在同类 token 上微调，基本不会打开 S5。

## 2.2 Rational-only substrate 的意义

v12.35 Gate S/H 的 family summary 是：

```text
D-RAT:
  rows = 16
  substrate health pass = 11
  near pass = 16
  healthy base pass = 0
  best mean delta = 0.01318359375
  best worst delta = -0.01171875
  best LineC pass rate = 0.9259259259259259

D-CHE:
  substrate health pass = 0
  best LineC pass rate = 0.0

D-FOU:
  substrate health pass = 0
  best mean delta = -0.314453125
  best worst delta = -0.50390625
  best LineC pass rate = 0.0

D-RBF:
  substrate health pass = 0
  best mean delta = -0.4535590277777778
  best worst delta = -0.6796875
  best LineC pass rate = 0.5185185185185185

D-WAV:
  substrate health pass = 0
  best LineC pass rate = 0.0
```

这说明：

$$
\boxed{
\text{当前只有 Rational 可以作为 usable substrate 的起点。}
}
$$

但是 Rational 仍不是 healthy base，因为：

```text
healthy_base_gate_pass_count = 0
```

所以 Rational 的角色应该是：

```text
efficient substrate / repair target
```

不是：

```text
official base success
```

## 2.3 Response dictionary 的失败非常关键

v12.35 建立了 basis response dictionary：

```text
basis_response_dictionary_rows = 270
basis_response_dictionary_executed_rows = 195
response_dictionary_pass_count = 0
```

最接近的 Rational rows 也只有：

```text
max CouplingR2 delta ≈ 0.00153
min NoiseSignalLeak delta ≈ -0.000715
min Reservoir delta ≈ -0.001576
```

而 gate 需要的量级大约是：

```text
CouplingR2 delta >= 0.01
NoiseSignalLeak delta <= negative threshold
RealSignalReservoirRatio delta <= negative threshold
```

这说明当前 functional candidates 的问题不是“tail 安全门太严”，而是：

$$
\boxed{
\text{它们几乎没有真正移动 signal/reservoir/noise channel。}
}
$$

这也解释了为什么 P3/P4 复核为 0：

```text
basis_functional_p3_rows = 270
basis_functional_p3_pass_count = 0
basis_functional_p4_rows = 195
basis_functional_p4_pass_count = 0
```

## 2.4 Non-RAT 不能直接进入 functional repair

Non-RAT task-health repair 结果：

```text
D-CHE: pass = 0, LineC = 0
D-FOU: pass = 0, best mean = -0.314, worst = -0.504, min AUC-time = 4.30, LineC = 0
D-RBF: pass = 0, best mean = -0.454, worst = -0.680, min AUC-time = 7.53, LineC = 0.519
D-WAV: pass = 0, best mean = -0.547, worst = -0.672, min AUC-time = 10.24, LineC = 0
```

这说明 foreach-off / lifetime-open 只是系统层进展，不是 substrate-health 进展。尤其 RBF 有一些 LineC 局部信号，但 task/AUC 太坏，不能作为 functional substrate。

因此：

$$
\boxed{
\text{Non-RAT 下一步不能直接 functional P3，必须先解决 task-health substrate。}
}
$$

## 2.5 MLP functional no-go 的含义

MLP M-J v3 结果：

```text
M-J1: pass = 0, mean source_vs_control = -0.0014468
M-J2: pass = 0, mean source_vs_control = -0.0010127
M-J3: pass = 0, mean source_vs_control = -0.0013021
```

这说明当前 MLP generic loss-agnostic functional observable family 仍然 no-go。后续 functional 不应该继续作为通用 MLP optimizer trick 平均扩展；更应该成为：

```text
basis-specific channel repair
```

---

# 3. 当前真正问题

v12.35 暴露的核心问题是：

$$
\boxed{
\text{有 substrate，但没有 channel co-location value source。}
}
$$

更具体：

```text
1. Rational 是唯一稳定 substrate，但 functional response dictionary 不支持构造 official repair。
2. Non-RAT 的 lifetime 或 workspace 可以局部打开，但 task/AUC/LineC substrate health 不成立。
3. MLP generic functional no-go，说明当前 functional 思路不是通用 optimizer trick。
4. Basis-specific functional P3/P4 没有移动 CouplingR2 / NoiseSignalLeak / ReservoirRatio 到足够量级。
```

所以继续做：

```text
B-RAT6 / B-RAT7；
B-FOU6 / B-FOU7；
M-J4 / M-J5；
D-CHE21 / D-FOU21 / D-RBF18 / D-WAV17 同类 token
```

很可能只是低价值网格搜索。

---

# 4. 下一步核心假设

v12.36 不再以“扩 token”为主，而是验证三个机制假设。

## H1：Rational 需要 internal-channel response dictionary，而不是 output-level response dictionary

当前 response dictionary 多数是 output / LineC 级别响应。Rational 的有效 functional repair 可能必须直接操控：

```text
denominator channel
derivative channel
tangent channel
group diversity channel
readout-rational coupling channel
```

假设：

$$
\boxed{
\text{Rational 的 functional repair 失败，是因为 response dictionary 没有覆盖内部 rational channel。}
}
$$

## H2：Non-RAT 需要先做 task-health substrate，而不是 lifetime-only substrate

Non-RAT 的 foreach-off / lifetime-open 后 task/AUC/LineC 崩坏。假设：

$$
\boxed{
\text{Non-RAT 缺少的是 task-health-preserving substrate formulation，}
\text{不是 functional repair 事件本身。}
}
$$

## H3：Basis-specific functional update 需要先有 response model，再有 candidate construction

当前做法仍然太像“先试一个 functional token，再看 P3”。v12.36 改成：

$$
\boxed{
\text{先建立 basis actuator response model，}
\text{再解 constrained channel co-location repair。}
}
$$

---

# 5. v12.36 总体路线

v12.36 设为五条线：

```text
Line S:
  Substrate-health map v3。
  目标：把 usable substrate 与 workspace-only substrate 严格分开。

Line Q:
  Basis-channel response dictionary v2。
  目标：从 output-level response 升级为 internal-channel response。

Line B:
  Basis-specific constrained functional repair v2。
  目标：从 response model 解修复方向，而不是再扩 token。

Line N:
  Non-RAT task-health substrate redesign。
  目标：让至少一个 Non-RAT family 达到 substrate-health near pass。

Line M:
  MLP functional closure monitor。
  目标：保持 no-go 复核，不再主投 generic MLP functional。

Line R:
  Code/provenance/readback/finalizer。
```

---

# 6. Line S：Substrate-health map v3

## 6.1 目标

v12.35 已经证明 workspace-only pass 不够。Line S 要构造更清楚的三层分类：

```text
WorkspaceOnly:
  只证明跑得动。

UsableSubstrate:
  workspace + minimal task/AUC/LineC 不灾难。

HealthyBase:
  task/AUC/LineC/tail 已接近 MLP，可直接 promotion。
```

## 6.2 候选

### Rational candidates

保留 v12.35 pass/near 的 Rational：

```text
D-RAT25,D-RAT26,D-RAT27,D-RAT28,D-RAT29,D-RAT30,D-RAT31,D-RAT35,D-RAT36,D-RAT37,D-RAT39
```

新增机制候选不超过 4 个：

```text
D-RAT41-InternalTangentTelemetrySubstrate
D-RAT42-DenDerivativeSlowStartSubstrate
D-RAT43-GroupDiversitySubstrate
D-RAT44-ReadoutRationalDecoupledSubstrate
```

这些不是 token 小修，而是必须写明：

```text
which internal rational channel it changes;
what telemetry it expects to improve;
which response model it will feed.
```

### Non-RAT candidates

每个 family 最多 2 个：

```text
D-CHE21-DegreeEnergyTaskHealth
D-CHE22-LateHighDegreeSubstrate

D-FOU21-LowFreqIdentityResidualTaskHealth
D-FOU22-BandLimitedResidualTaskHealth

D-RBF18-CompactCenterTaskHealth
D-RBF19-OccupancyBalancedTaskHealth

D-WAV17-HatLocalTaskHealth
D-WAV18-ScaleStableLocalTaskHealth
```

## 6.3 指标

每个 candidate 必须记录：

```text
family
candidate
workspace_raw_ratio
workspace_incremental_ratio
step_ratio
mean_delta_vs_MLP
worst_delta_vs_MLP
AUC_step_ratio
AUC_time_ratio
LineC_pass_rate
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
CEp99_delta
NLL_delta
ECE_delta
tail_bad_rate
```

Rational 额外记录：

```text
den_p01
den_p99
r_prime_p95
r_double_prime_p95
tangent_condition
tangent_top_eigen_share
group_diversity
readout_rational_coupling
per_group_update_norm
```

Chebyshev 额外记录：

```text
degree_energy_ratio
high_degree_energy
degree_tangent_condition
```

Fourier 额外记录：

```text
freq_band_energy
high_freq_energy_ratio
phase_drift
spectral_entropy
```

RBF 额外记录：

```text
center_occupancy_entropy
dead_center_fraction
width_condition
out_of_grid_fraction
```

Wavelet 额外记录：

```text
scale_energy_entropy
support_occupancy
support_overlap
scale_dead_fraction
```

## 6.4 Gate S3：Usable Substrate

一个 candidate 进入 functional response dictionary 必须满足：

$$
workspace\_raw\_ratio \le 1.10,
$$

$$
workspace\_incremental\_ratio \le 1.90,
$$

$$
step\_ratio \le 1.90,
$$

$$
mean\_delta\_vs\_MLP \ge -0.04,
$$

$$
worst\_delta\_vs\_MLP \ge -0.08,
$$

$$
AUCtime\_ratio\_vs\_MLP \le 2.00,
$$

$$
LineC\_pass\_rate \ge 0.30.
$$

这是 substrate gate，不是 final base gate。它的目标是阻止“跑得快但已经坏掉”的 candidate 进入 functional。

## 6.5 Healthy Base Gate

如果 candidate 自己能达到：

$$
mean\_delta\_vs\_MLP \ge -0.005,
$$

$$
worst\_delta\_vs\_MLP \ge -0.015,
$$

$$
AUCtime\_ratio\_vs\_MLP \le 1.05,
$$

$$
LineC\_pass\_rate \ge 0.80,
$$

$$
CEp99\_delta \le 0.05,
$$

则为 HealthyBase，可直接进入 confirm，同时也可以作为 functional anchor。

## 6.6 不满足条件时 Codex 先做什么

如果 Rational 只有 WorkspaceOnly：

```text
1. 输出 rational_internal_telemetry_failure.csv；
2. 选择 den / derivative / tangent / group diversity 哪个 channel 最异常；
3. 进入 Line Q，不得继续只扩 D-RAT alias。
```

如果 Non-RAT workspace 过但 task/AUC/LineC 崩：

```text
1. 输出 nonrat_task_health_collapse_table.csv；
2. 判断崩坏来自 task、AUC、LineC、tail 哪一项；
3. 执行本 family 的 task-health substrate repair，不进入 functional P3。
```

如果所有 family 都无 UsableSubstrate：

```text
route = R1-NoUsableSubstrate
stop functional official
只允许低预算 code/audit monitor
```

---

# 7. Line Q：Basis-channel response dictionary v2

## 7.1 目标

v12.35 的 response dictionary 是 output-level / LineC-level，结果 0 pass。v12.36 要建立 basis-channel response dictionary：

$$
\boxed{
\text{actuator} \rightarrow \text{basis internal telemetry} \rightarrow \text{LineC/task/tail audit}
}
$$

## 7.2 输入

只使用通过 Gate S3 的 UsableSubstrate candidates。

## 7.3 Actuator families

### Rational actuator channels

```text
A-RAT-den-slope
A-RAT-rprime-cap
A-RAT-rdoubleprime-cap
A-RAT-tangent-condition
A-RAT-group-diversity
A-RAT-readout-rational-decouple
A-RAT-lowrank-tangent-rotation
```

### Chebyshev actuator channels

```text
A-CHE-degree-energy-shift
A-CHE-high-degree-delay
A-CHE-degree-tangent-condition
```

### Fourier actuator channels

```text
A-FOU-band-energy-shift
A-FOU-phase-stabilize
A-FOU-highfreq-veto
```

### RBF actuator channels

```text
A-RBF-center-occupancy
A-RBF-width-condition
A-RBF-oog-boundary
```

### Wavelet actuator channels

```text
A-WAV-scale-balance
A-WAV-support-overlap
A-WAV-local-tail-coverage
```

## 7.4 记录字段

```text
family
base_candidate
actuator
dataset
seed
pre_state_hash
actuator_norm
task_delta_proxy
AUC_proxy_delta
CEp99_delta
NLL_delta
ECE_delta
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
control_gap
basis_channel_delta_json
internal_telemetry_before_json
internal_telemetry_after_json
pass_response_gate
```

## 7.5 Response Pass Gate

必须满足：

$$
\Delta CouplingR^2 \ge 0.005
$$

或

$$
\Delta NoiseSignalLeak \le -0.002
$$

或

$$
\Delta RealSignalReservoirRatio \le -0.002.
$$

并且：

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
control\_gap \ge 0.
$$

这是 response dictionary gate，不是 final P3 gate。目标是先找到有物理响应的 actuator。

## 7.6 不满足条件时 Codex 先做什么

如果 Rational response 仍 0 pass：

```text
1. 检查 actuator 是否真的改变 den/r'/r''/tangent/group diversity；
2. 若 basis_channel_delta 近 0，则 route = ActuatorWeak；
3. 若 basis_channel_delta 非 0 但 LineC 不动，则 route = ChannelNotCoupledToLineC；
4. 若 LineC 动但 task/tail 坏，则 route = UnsafeChannelRepair；
5. 每个 route 必须输出下一步机制假设，不允许只写 no-go。
```

---

# 8. Line B：Basis-specific constrained functional repair v2

## 8.1 目标

只有 Line Q 有 response pass 的 family/candidate 才能进入 Line B。Line B 不再手写 token，而是从 response dictionary 解约束组合：

$$
\min_a
\quad
-\alpha \widehat{\Delta CouplingR^2}(a)
+
\beta \widehat{\Delta NoiseSignalLeak}(a)
+
\gamma \widehat{\Delta ReservoirRatio}(a)
+
\eta \widehat{TailRisk}(a)
+
\lambda \|a\|_2^2.
$$

其中 $a$ 是 actuator 组合权重。

## 8.2 约束

$$
\widehat{CEp99\_delta}(a) \le 0.05,
$$

$$
\widehat{NLL\_delta}(a) \le 0.02,
$$

$$
\widehat{ECE\_delta}(a) \le 0.02,
$$

$$
\widehat{AUCtime\_delta}(a) \le 0.05,
$$

$$
control\_gap(a) \ge 0.
$$

## 8.3 P3 Gate

P3 pass 必须满足：

$$
\Delta CouplingR^2 \ge 0.01,
$$

$$
\Delta NoiseSignalLeak \le -0.005
\quad \text{or} \quad
\Delta RealSignalReservoirRatio \le -0.005,
$$

$$
control\_gap \ge 0.002,
$$

并且：

$$
CEp99\_delta \le 0.05,
$$

$$
NLL\_delta \le 0.02,
$$

$$
ECE\_delta \le 0.02.
$$

## 8.4 P4 Gate

P4 short-run 必须满足：

```text
source_vs_noop >= 0
source_vs_best_control >= 0.003
LineC majority pass
CEp99 / NLL / ECE non-harm
AUCtime non-harm
train-shuffle robust >= 2/3
```

## 8.5 不满足条件时 Codex 先做什么

如果 P3 fail due to weak geometry delta：

```text
return to Line Q; response dictionary insufficient
```

如果 P3 pass but P4 fail due to task/control：

```text
run policy-aware P3/P4 alignment:
  event + planned post-event optimizer policy
  event + cooldown
  event + smaller lambda
```

如果 P4 fail due to tail/calibration：

```text
do not CE-tune;
run loss-agnostic tail proxy:
  logit entropy floor
  top1-top2 gap guard
  unlabeled confidence drift cap
```

---

# 9. Line N：Non-RAT task-health substrate repair

## 9.1 目标

Non-RAT 不允许直接进入 functional P3，除非先有 task-health substrate。Line N 针对每个 Non-RAT family 做少量机制级修复。

## 9.2 Chebyshev

假设：global polynomial support 导致 early trajectory / reservoir 不健康。

候选：

```text
D-CHE23-DegreeEnergyLateEnable
D-CHE24-InputBoundedDegreeMix
D-CHE25-RolewiseDegreeEnergyCap
```

记录：

```text
degree_energy_ratio
high_degree_energy
tangent_condition
LineC_pass_rate
AUCtime
```

## 9.3 Fourier

假设：low-frequency compact path expression 不够，高频又引入 noise risk。

候选：

```text
D-FOU23-IdentityResidualLowFreq
D-FOU24-BandLimitedMultiScale
D-FOU25-LateHighFreqResidual
```

记录：

```text
freq_band_energy
high_freq_energy_ratio
NoiseSignalLeak
AUCtime
```

## 9.4 RBF / FastKAN

假设：compact RBF center coverage 不足导致 expression/task 不稳。

候选：

```text
D-RBF20-OccupancyBalancedCenters
D-RBF21-CompactBumpNoExp
D-RBF22-LocalLinearRBFHybrid
```

记录：

```text
center_occupancy_entropy
dead_center_fraction
width_condition
out_of_grid_fraction
```

## 9.5 Wavelet

假设：local support / scale coverage 与 task signal 不匹配。

候选：

```text
D-WAV19-ScaleDiversityLocalHat
D-WAV20-SupportOverlapGuard
D-WAV21-TailLocalSupportFrame
```

记录：

```text
scale_energy_entropy
support_overlap
local_tail_coverage
```

## 9.6 Gate

Non-RAT task-health substrate pass:

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
LineC\_pass\_rate \ge 0.30,
$$

$$
workspace\_incremental\_ratio \le 2.0.
$$

---

# 10. Line M：MLP functional closure monitor

## 10.1 目标

不再主投 MLP functional。只在每轮保留一个 closure monitor，防止误判 functional 是 KAN-specific 还是 generic。

## 10.2 候选

最多 2 个：

```text
M-K1-ResponseDictionaryStyleMLPProbe
M-K2-BasislessChannelCoLocationProbe
```

## 10.3 Gate

若 MLP monitor 仍 0 pass，则维持：

```text
MLPFunctionalNoGo_CurrentLossAgnosticObservableFamily
```

并把 functional 主预算保留给 basis-specific。

---

# 11. Line R：代码与实现审计

Codex 必须在复盘里写清楚以下代码路径：

```text
1. 每个 substrate candidate 的 PrimitiveSpec / kernel path。
2. 是否使用 label-informed init。
3. workspace / task / LineC 是否使用同一 candidate id。
4. response dictionary 是否只使用 loss-agnostic probe。
5. Line Q/B 是否有 CE/NLL/ECE/CEp99 direction leakage。
6. controls 是否 matched。
7. P4 是否 gated by P3。
```

缺失任一项，route 必须降级为：

```text
R0-CodeReviewSurfaceIncomplete
```

---

# 12. 必须生成的 artifacts

```text
v1236_route_decision.json
v1236_progress_by_line.csv
v1236_substrate_health_v3.csv
v1236_family_substrate_summary.csv
v1236_basis_channel_telemetry.csv
v1236_response_dictionary_v2.csv
v1236_response_dictionary_summary.csv
v1236_basis_functional_p3_v2.csv
v1236_basis_functional_p4_v2.csv
v1236_nonrat_task_health_repair_v2.csv
v1236_mlp_functional_monitor.csv
v1236_failure_table.csv
v1236_code_provenance_audit.csv
v1236_required_artifact_manifest.csv
v1236_code_review_packet.zip
```

---

# 13. 必须可视化

```text
fig_v1236_progress_by_line.svg
fig_v1236_substrate_health_family_matrix.svg
fig_v1236_rational_internal_channel_response.svg
fig_v1236_response_dictionary_delta_scatter.svg
fig_v1236_functional_p3_gate_waterfall.svg
fig_v1236_nonrat_task_health_pareto.svg
fig_v1236_mlp_functional_monitor.svg
fig_v1236_failure_taxonomy.svg
```

---

# 14. Stop / go route

## Success routes

```text
S1-UsableSubstrate:
  至少一个 active basis 通过 Gate S3。

S2-ResponseDictionaryOpened:
  至少一个 basis response dictionary pass。

S3-BasisFunctionalP3Opened:
  至少一个 basis-specific functional repair 通过 P3。

S4-BasisFunctionalP4Opened:
  至少一个 basis-specific functional repair 通过 P4。

S5-OfficialFunctionalSuccess:
  label-free strict FC-PureKAN substrate + functional repair 通过 official controls。
```

## Failure routes

```text
R1-NoUsableSubstrate:
  没有任何 family 通过 Gate S3。

R2-SubstrateButNoResponse:
  有 substrate，但 response dictionary 0 pass。

R3-ResponseButNoFunctionalP3:
  有 response，但 functional P3 0 pass。

R4-NonRATTaskHealthMissing:
  Non-RAT 仍无 task-health substrate。

R5-MLPFunctionalNoGo:
  MLP monitor 仍 0 pass。

R6-LowValueGridSearchDetected:
  只是扩同类 token，没有新机制，禁止继续。
```

---

# 15. 最终总结

v12.35 没有失败到没有价值。它回答了一个关键问题：

$$
\boxed{
\text{当前不是缺更多 basis token，而是缺 basis-channel co-location value source。}
}
$$

下一步 v12.36 的重点不是继续扩 B-RAT/B-FOU/M-J，而是：

```text
1. 用 substrate-health gate 区分真正可修的 substrate；
2. 用 internal channel telemetry 建立 basis response dictionary；
3. 用 constrained solver 构造 functional repair；
4. Non-RAT 先修 task-health substrate；
5. MLP functional 只做 closure monitor。
```
