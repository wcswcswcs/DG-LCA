# DG-KAN v14.3：Functional Value / Basis Constraint 分离 + All-Basis Substrate 并行推进完整计划

> 版本：v14.3 execution plan  
> 生成时间：2026-05-29  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no active B-spline budget；no teacher；no distillation；no sampler / class weight；no dataset-name branch；no label-informed initialization；functional direction 不使用 validation/test/future/query batch；LineC / CEp99 / NLL / ECE / Brier 只能作为 audit/gate，不能作为方向源；MLP-FMS positive 只能说明 generic optimizer/control，不得写成 KAN-specific promotion。

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是找到一个只在局部几何指标上好看的更新。项目真正要证明的是：

$$
\boxed{
\text{label-free strict FC-PureKAN substrate/base}
+
\text{functional update}
>
\text{同一 substrate/base + ordinary backprop / AdamW controls}
}
$$

这里的 “$>$” 必须同时包含：

```text
1. 表达力不打折；
2. forward / backward / step / memory 与 MLP 可比；
3. 训练轨迹不慢，AUC-step / AUC-time 不坏；
4. 几何更健康：train-probe coupling、signal / reservoir / noise、tail、calibration 不坏；
5. functional update 的收益必须打过 NoOp / RandomMatchedNorm / AdamWParallel / matched-norm controls；
6. functional update 不能是 CE-specific trick，也不能依赖 label-informed init、validation/test/future/query batch 或 dataset-name branch。
```

## 0.2 v14.2 真实状态

v14.2 执行了：

```text
functional-first persistent metric state, FMS；
all-basis substrate map / remap / sweep；
Rational focused repair；
R6/R7/R8 denominator / readout / phase repair；
R9 role-separated FMS；
R10 train-stream agreement-gated safety projection；
Non-RAT cross-plan substrate repair audit。
```

但最终没有达成 KAN-specific S3/S4/S5：

```text
official route = R4-FMSNoGoCurrentDefinition
best repair route = R3-GenericFunctionalOnlyKANSpecificNotEstablished
promotion_allowed = 0
real_short_run_open_allowed = 0
```

v14.2 的正信号是：

```text
1. FMS runner / artifact / persistent state surface 已经可执行；
2. MLP-FMS 在 repair 中能达到 5/7，说明 generic FMS signal 存在；
3. Rational / D-RAT 局部 strong-source rows 存在；
4. D-RAT30 是当前唯一通过 v14.2 substrate + healthy gate 的 family；
5. all-basis substrate remap 证明 Non-RAT fail 不是单个 candidate 名称选错导致。
```

v14.2 的硬失败是：

```text
1. official non-amortized FMS 中 MLP-FMS、Rational-FMS 都是 0/7；
2. amortized FMS 能让 MLP-FMS 到 5/7，但 Rational 只有 1/7 或 2/7；
3. D-RAT strong-source rows 被 LineC / tail / overhead gate 拒绝；
4. R6/R7/R8/R9/R10 都没有打开 Rational gate；
5. Non-RAT 在 v14.2 strict substrate gate 下没有 family pass；
6. real short-run 不允许打开。
```

## 0.3 v14.2 后的关键判断

过去几个版本把 functional update 当作：

```text
一个产生 value 的方向源：
  basis/group telemetry -> FMS state -> update direction。
```

v14.2 说明这条路不成立：basis/group telemetry 作为 value source 时，会丢失或扭曲 generic population-risk signal。MLP-FMS 已经显示 generic value signal 可见；Rational-FMS 失败说明 KAN basis telemetry 当前没有把这个 value signal 转成稳定 KAN-specific advantage。

因此 v14.3 的核心重置是：

$$
\boxed{
\text{functional value source 与 basis constraint / geometry safety 必须分离。}
}
$$

新的原则是：

```text
1. value path：由 generic loss-interface FMS / PopRisk signal 产生；
2. constraint path：由 basis telemetry 提供 trust region / safety projection / plasticity boundary；
3. basis telemetry 不再直接替代 generic value source；
4. KAN-specific advantage 应来自：在不破坏 generic value 的前提下，basis constraint 让训练几何更健康。
```

这不是调一个新 token，而是改变 functional update 的因果结构。

---

# 1. 各线当前进展百分比

这些百分比是基于 v14.2 gate、artifact 完整性、机制清晰度和离 official success 的距离做出的估计，不是 artifact 中的官方字段。

| 线 | 当前完成度 | 判断 |
|---|---:|---|
| Line R：代码 / provenance / finalizer 审计 | 99% | 工程闭包强，required artifact 与 forbidden audit 基本不是 blocker。 |
| Historical FHQ / B320-current 工程能力 | 85% frozen | 历史强，但 label-informed init 禁用，不能 official。 |
| Line C：Manifold-Channel Geometry Audit | 88% | 审计稳定，能暴露 LineC / noise / reservoir / tail 问题，但不能做 direction source。 |
| PopRisk / FMS 实现面 | 88% | persistent state、per-example gradient telemetry、amortized refresh 已实现。 |
| Generic MLP-FMS | 45% | repair 中有 5/7 positive，说明 generic signal 存在；但不是 KAN-specific，且还需 full-budget / seed robustness。 |
| FMS as KAN-specific value source | 8% | Rational 最好 2/7，R6-R10 均未打开；当前 value-source 设计不成立。 |
| Rational substrate | 85% | 唯一稳定 substrate / healthy family，D-RAT30 pass；但 FMS 不能修成 KAN-specific success。 |
| Rational FMS functional | 10% | 有局部 strong-source rows，但 LineC / tail / overhead / family coverage 不闭合。 |
| RBF / FastKAN substrate | 20% | 有旧 workspace/local efficiency 信号，但 v14.2 strict substrate pass=0，task/AUC/LineC 不健康。 |
| Chebyshev substrate | 15% | exact no-materialize rows 存在，但 incremental memory 高，source/telemetry fail。 |
| Fourier substrate | 20% | raw/step 局部接近，曾有 P3 repair 60 rows但 P3 pass=0；incremental memory 仍高。 |
| Wavelet substrate | 15% | local workspace 有弱信号，但 step/memory/task/AUC/LineC 不闭合。 |
| Non-RAT functional | 0%-5% | 没有 strict substrate pass，不允许 official FMS proof。 |
| Overall next-gen MLP claim | 32%-38% | generic FMS 有正信号，但 KAN-specific + all-basis substrate 没闭合。 |

---

# 2. v14.2 结果独立分析

## 2.1 v14.2 有进展吗？

有，但不是 KAN 能力进展。它的进展主要是把问题拆清楚了。

v14.2 official 的最终 route 是：

```text
R4-FMSNoGoCurrentDefinition
basis_substrate_pass_count = 1
healthy_base_pass_count = 1
mlp_fms_task_pass_count = 0
rational_fms_task_pass_count = 0
nonrat_fms_task_pass_count_total = 0
real_short_run_open_allowed = 0
```

这说明 official run 本身没有 functional success。

但 v14.2 的 repair 发现：

```text
amortized persistent FMS interval=40:
  MLP-FMS task pass = 5/7
  Rational-FMS task pass = 1/7

D-RAT28 focused repair:
  Rational task pass = 2/7

interval=80 phase repair:
  MLP-FMS task pass = 6/7
  Rational-FMS task pass = 2/7

R10 train-stream agreement safety:
  Rational task pass = 0/7
```

所以 v14.2 的真实结论是：

$$
\boxed{
\text{FMS 有 generic value signal，但 KAN/Rational 没有形成 KAN-specific value。}
}
$$

## 2.2 为什么不能把 MLP-FMS positive 写成成功？

因为项目目标不是 generic optimizer paper，而是：

$$
\text{PureKAN substrate + functional update}
>
\text{same substrate + AdamW controls}.
$$

MLP-FMS positive 只能说明：

```text
functional metric state 作为 generic optimizer/control 有潜力。
```

它不能说明：

```text
KAN basis coordinate 提供了 functional leverage。
```

只有当：

$$
(\text{Rational-FMS}-\text{Rational-AdamW})
>
(\text{MLP-FMS}-\text{MLP-AdamW})
$$

并且同时满足 LineC / tail / AUC / controls，才能 claim KAN-specific functional success。

v14.2 没有达到这个条件。

## 2.3 为什么 Rational 有 strong-source rows 仍不能算进展？

v14.2 多处显示 Rational 局部 row 很强。例如 D-RAT28 的 X2 / X4 source 很高，但 LineC=0；R7/R8 有 source 很强的 rows，但 LineC / tail / overhead gate fail；R10 也能产生 source-positive rows，但没有 task-family pass。

这说明：

```text
Rational substrate 能被 FMS 推动；
但推动方向不是稳定的 signal-channel improvement；
它经常只是 task/source 局部改善，同时破坏 LineC / tail / overhead。
```

因此当前 Rational FMS blocker 不是“完全没有 source”，而是：

$$
\boxed{
\text{source signal 与 geometry/tail safety 没有 co-location。}
}
$$

## 2.4 Non-RAT 线怎么理解？

v14.2 all-basis remap 后：

```text
D-RAT: pass
D-RBF: fail step/memory/task/AUC
D-CHE: source row missing / telemetry fail
D-FOU: source row missing / telemetry fail
D-WAV: source row missing / telemetry fail
```

进一步 cross-plan Non-RAT audit 显示：

```text
v12.34.2 foreach-off repair 有 workspace pass rows；
但按 v14.2 strict gate 重新判定后 strict pass = 0；
D-FOU / D-RBF / D-WAV raw memory 与 step 局部接近，但 incremental_memory_ratio > 1.75；
D-CHE exact no-materialize rows 的 best incremental_memory_ratio 仍高达 4.2293；
D-FOU 曾执行 P3 repair 60 rows，但 P3 pass = 0。
```

因此 Non-RAT 当前不能进入 official FMS proof。不是因为我们忘了它们，而是它们还没有成为可控 substrate。

---

# 3. 当前真正问题：Value / Constraint 混淆

v14.2 之前，我们反复试图让 basis/group telemetry 既提供 value，又提供 safety：

```text
basis telemetry -> FMS state -> update value direction
basis telemetry -> safety / trust region
```

这导致一个问题：当 basis telemetry 不成熟时，它会把 generic population-risk signal 弄坏。v14.2 正好证明了这一点：

```text
MLP-FMS positive；
Rational-FMS weak；
Rational denominator / phase / role / agreement 小修都不能稳定打开 gate；
Non-RAT substrate fail。
```

所以 v14.3 要把 functional update 拆成两条路径：

$$
\boxed{
\Delta\theta_{FMS}
=
\Pi_{basis-safe}
\left(
\Delta\theta_{generic-FMS}
\right)
}
$$

其中：

```text
generic-FMS:
  从 loss-interface population-risk signal 产生 value update；
  不读 CEp99 / NLL / ECE / LineC；
  可以在 MLP 与 KAN 上共享。

basis-safe projection:
  只做 trust region / telemetry safety / plasticity boundary；
  不改变 value source 的核心方向；
  必须记录投影前后 cosine 和 value retention。
```

如果投影后 generic value 被破坏，说明 basis substrate 或 basis safety 设计不成熟；不能继续把问题归咎于 functional value source。

---

# 4. v14.3 核心假设

## H1：generic FMS value source 是有用的，但 KAN basis route 把它破坏了

v14.2 中 MLP-FMS 在 repair 下能达到 5/7，而 Rational 只有 1/7 或 2/7。最自然的假设是：

$$
\boxed{
\text{FMS 的 value signal 可见，}
\text{但 KAN-specific transform / basis update 把 value signal 损失或污染。}
}
$$

验证方式：

```text
同一个 generic-FMS update，依次比较：
1. 不投影；
2. identity projection；
3. basis trust-region projection；
4. denominator/slope projection；
5. delayed readout-basis projection；
6. random matched projection。
```

关键指标：

$$
\cos(\Delta\theta_{projected},\Delta\theta_{generic})
$$

$$
value\_retention
=
\frac{\langle \Delta\theta_{projected}, \Delta\theta_{generic}\rangle}{\|\Delta\theta_{generic}\|^2+\epsilon}
$$

如果 `value_retention < 0.7`，basis projection 本身就是 blocker。

## H2：Rational 的 LineC / tail failure 是 projection failure，不是 value failure

Rational rows often have source positive but LineC/tail bad. 这说明 value 不是完全缺失。需要判断：

```text
是 generic value update 本身造成 LineC/tail坏？
还是 basis-specific projection/telemetry造成 LineC/tail坏？
```

实验上需要记录：

```text
before_projection LineC/tail audit；
after_projection LineC/tail audit；
projection_rejection_reason；
which roles were scaled / clipped。
```

## H3：Non-RAT 不应进入 FMS proof，直到具备 substrate-health

Non-RAT 当前 strict substrate pass=0。继续对它们跑 FMS proof 只会制造噪声。v14.3 对 Non-RAT 只做 substrate repair：

```text
RBF/FastKAN:
  compact Gaussian / local bump / active center substrate。

Chebyshev:
  recurrence no-materialize / degree-energy bounded substrate。

Fourier:
  low-frequency band-limited phase-stable substrate。

Wavelet:
  local hat / support-stable substrate。
```

## H4：Functional update 的突破要靠训练过程，而不是单步 event

FMS 是 persistent state，不能只看 20-step compute-budgeted synthetic。v14.3 需要使用两级 budget：

```text
scout:
  fast 20-40 step, X1-X7 seed0, CE/Brier。

mechanism-confirm:
  top candidate 200-step, seeds 0,1,2, CE/Brier。
```

只有 mechanism-confirm 过 synthetic >=5/7，才允许 real 3x3 short-run。

---

# 5. 实验总览

v14.3 分为七条线：

```text
Line R: Code / provenance / implementation readback。
Line G: Generic FMS value baseline and MLP control。
Line K: Value-preserving basis constraint on Rational。
Line D: All-basis substrate repair, Non-RAT 不直接 functional。
Line C: Manifold-Channel audit。
Line X: Synthetic + real gate。
Line Z: finalizer / route / no-go / next hypothesis。
```

预算建议：

```text
Line G + K functional mechanism: 45%
Line D all-basis substrate repair: 35%
Line C/R/Z audit/finalizer: 20%
```

Line D 内部：

```text
Rational: 25% of Line D
RBF/FastKAN: 30% of Line D
Chebyshev: 15% of Line D
Fourier: 15% of Line D
Wavelet: 15% of Line D
```

---

# 6. Line R：代码与实现审计

## 6.1 目标

确保 v14.3 的核心语义不是换名小修：

```text
generic value path 与 basis constraint path 必须在代码中分离；
CEp99 / NLL / ECE / LineC 不得进入 direction；
MLP-FMS positive 不得写成 KAN success；
Non-RAT substrate fail 不得进入 official FMS proof。
```

## 6.2 必须落盘

```text
v143_code_path_manifest.csv
v143_functional_value_constraint_manifest.csv
v143_loss_interface_audit.csv
v143_forbidden_information_audit.csv
v143_projection_semantics_audit.csv
v143_required_manifest.csv
```

## 6.3 必须字段

```text
method
uses_label_for_init
uses_validation_for_direction
uses_test_for_direction
uses_future_for_direction
uses_query_for_direction
uses_linec_for_direction
uses_cep99_for_direction
uses_nll_for_direction
uses_ece_for_direction
value_source_type
constraint_source_type
projection_applied
projection_uses_audit_metric
promotion_allowed
```

## 6.4 Gate

如果任何 candidate 满足：

```text
uses_linec_for_direction = 1
uses_cep99_for_direction = 1
uses_validation_for_direction = 1
uses_future_for_direction = 1
```

则 route 必须为：

```text
R0-ForbiddenDirectionSource
```

---

# 7. Line G：Generic FMS value baseline

## 7.1 目标

确认 FMS 的 generic value source 到底有多强，并建立所有 KAN-specific 结果的 reference。

## 7.2 方法

比较：

```text
G0-AdamW
G1-PriorSNRReference
G2-ParameterFMS
G3-LayerFMS
G4-RoleFMS
G5-AmortizedParameterFMS
G6-AmortizedLayerFMS
G7-PhaseScheduleGenericFMS
GCTRL-RandomMatchedNorm
```

在 MLP 上跑：

```text
synthetic X1..X7
seeds 0,1,2
loss_interfaces CE,Brier
train_steps = 200
batch_size = 32
```

## 7.3 指标

```text
source_vs_best_control
AUCtime_ratio
CEp99_delta
NLL_delta
ECE_delta
LineC_pass
per_example_gradient_overhead
fms_update_interval
fms_state_norm
active_fraction
value_state_entropy
```

## 7.4 Gate

Generic FMS positive：

$$
\text{synthetic task-family pass count} \ge 5/7.
$$

每个 pass family 需要：

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
ECE\_delta \le 0.02.
$$

注意：Generic FMS positive 不是 KAN success，只是 control baseline。

---

# 8. Line K：Value-preserving basis constraint on Rational

## 8.1 目标

在 Rational 上测试：generic FMS value update 能否在 basis constraints 后保留 value，同时改善或不伤 LineC/tail。

## 8.2 方法族

以同一个 generic FMS value update 为起点：

```text
K0-RAT-AdamW
K1-RAT-GenericFMS-NoProjection
K2-RAT-GenericFMS-IdentityProjectionAudit
K3-RAT-GenericFMS-DenSlopeTrustRegion
K4-RAT-GenericFMS-ReadoutBasisTrustRegion
K5-RAT-GenericFMS-RoleWisePlasticityTrustRegion
K6-RAT-GenericFMS-DelayedBasisConstraint
K7-RAT-GenericFMS-PhaseScheduleConstraint
K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint
KCTRL-RandomMatchedProjection
```

这些方法不允许把 basis telemetry 用作 value source。它们只能用于 constraint / trust region / projection。

## 8.3 新增核心指标

```text
generic_value_norm
projected_value_norm
cos_projected_vs_generic
value_retention
projection_rejection_fraction
projection_role_scale_denominator
projection_role_scale_readout
projection_role_scale_numerator
projection_role_scale_projection
pre_projection_source_vs_control
post_projection_source_vs_control
pre_projection_LineC
post_projection_LineC
pre_projection_CEp99_delta
post_projection_CEp99_delta
```

## 8.4 Gate K-SignalRetention

任何 basis projection 要进入 task gate，必须满足：

$$
cos(\Delta\theta_{proj},\Delta\theta_{generic}) \ge 0.60,
$$

$$
value\_retention \ge 0.70,
$$

$$
projection\_rejection\_fraction \le 0.50.
$$

如果不满足，则 route：

```text
R2-GenericValueKilledByBasisProjection
```

## 8.5 Gate K-Synthetic

KAN-specific synthetic pass 要求：

$$
\text{task-family pass count} \ge 5/7,
$$

并且：

$$
\Delta_{KAN-specific}
=
(\text{RAT-FMS}-\text{RAT-AdamW})
-
(\text{MLP-FMS}-\text{MLP-AdamW})
> 0.
$$

同时：

```text
LineC majority pass；
CEp99 / NLL / ECE non-harm；
source_vs_best_control >= 0.005；
AUCtime_ratio <= 1.0。
```

---

# 9. Line D：All-basis substrate repair

## 9.1 总原则

All-basis line 不是 optional。Functional-first 不等于 Rational-only。每个 active non-BSpline basis 必须持续推进：

```text
Rational
RBF / FastKAN
Chebyshev
Fourier
Wavelet
```

B-spline 继续 frozen，不进入 active budget。

每个 family 的目标不是自己成为 perfect base，而是成为：

$$
\boxed{
\text{efficient controllable substrate}
}
$$

## 9.2 Substrate gate

每个 candidate 要进入 FMS proof，必须满足：

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

如果 substrate gate 不过，则只能跑 substrate repair / telemetry diagnostic，不能 official functional proof。

## 9.3 Rational / Group Rational

### 目标

Rational 是当前唯一稳定 substrate，但 v14.2 说明现有 G16 Triton path 不能产生 KAN-specific FMS success。下一步不再在 R6-R10 上小修，而是补齐真正的 substrate variants：

```text
D-RAT-Horner-g8
D-RAT-Horner-g16
D-RAT-TritonEval-g8
D-RAT-TritonEval-g16 existing reference
D-RAT-ReadoutBasisDecoupled-v2
D-RAT-DenSlopeTelemetry-v2
```

### 指标

```text
group_count
uses_horner
uses_triton_eval
step_ratio
memory_ratio
incremental_memory_ratio
den_p01
den_p99
r_prime_p99
r_double_prime_p99
group_diversity
LineC_pass_rate
```

### 失败后 Codex 先尝试

```text
1. g8 unavailable -> implement registry + minimal Horner/Triton vertical slice，不混写 v14.2 proof。
2. denominator safe 但 LineC fail -> readout-basis decouple，不调 CE-tail。
3. source positive but tail bad -> basis constraint projection，不用 CEp99 direction。
```

## 9.4 RBF / FastKAN

### 目标

RBF/FastKAN 不能再用旧 dense basis 判断失败。v14.3 要做 FastKAN-style compact substrate：

```text
D-RBF-GaussianK4NoDense
D-RBF-CompactBumpK4
D-RBF-ActiveCenterK8
D-RBF-WidthConditionGuard
D-RBF-CenterOccupancyFMSReady
```

### 关键技术难点

```text
1. 不构造 dense [B,d,K] basis；
2. center / width / basis / grad 尽量 fused 或 no-materialize；
3. K 不靠大规模增加；
4. expression repair 不能破坏 L3 efficiency。
```

### 指标

```text
center_occupancy_entropy
empty_center_fraction
width_condition
out_of_grid_fraction
basis_usage_entropy
expression_E1_E6_E8_delta
step_ratio
memory_ratio
LineC_pass_rate
```

### 失败后 Codex 先尝试

```text
1. expression fail -> K_active 2->4->8 only if memory stays <=1.75。
2. OOG high -> center/width train-stream stats repair。
3. LineC fail -> occupancy rebalance，不调 dataset-specific hyperparam。
```

## 9.5 Chebyshev

### 目标

Chebyshev 需要 no-materialize recurrence + degree-energy bound：

```text
D-CHE-RecurrenceK3NoMaterialize
D-CHE-DegreeEnergyDamped
D-CHE-HighDegreeLateEnable
D-CHE-DegreeWiseFMSReady
```

### 指标

```text
degree_energy_k
degree_energy_high_ratio
recurrence_stability
incremental_memory_ratio
expression_E1_E6_E8_delta
AUCtime_ratio
LineC_pass_rate
```

### 失败后 Codex 先尝试

```text
1. incremental memory high -> recurrence derivative recompute。
2. task collapse -> high-degree damping / late-enable。
3. expression fail -> bounded degree K，不直接加大 K。
```

## 9.6 Fourier

### 目标

Fourier 需要低频、band-limited、phase-stable substrate：

```text
D-FOU-LowFreqK2IdentityResidual
D-FOU-BandLimitedK4
D-FOU-PhaseAmplitudeShared
D-FOU-HighFreqLeakGuard
D-FOU-BandwiseFMSReady
```

### 指标

```text
band_energy
high_freq_ratio
phase_drift
frequency_noise_leak
expression_E4_highfreq_delta
expression_E6_E8_delta
AUCtime_ratio
LineC_pass_rate
```

### 失败后 Codex 先尝试

```text
1. expression fail -> small low-rank residual，不直接高频扩张。
2. noise leak high -> high frequency quarantine。
3. memory high -> fused sincos / no-materialize derivative。
```

## 9.7 Wavelet

### 目标

Wavelet 保留 local support / scale-stable 路线：

```text
D-WAV-HatLocalK4
D-WAV-TriangularSupportK4
D-WAV-ScaleEnergyBalance
D-WAV-SupportOverlapGuard
D-WAV-ScaleWiseFMSReady
```

### 指标

```text
scale_energy
support_overlap
local_tail_coverage
support_dead_fraction
basis_usage_entropy
AUCtime_ratio
LineC_pass_rate
```

### 失败后 Codex 先尝试

```text
1. support dead -> scale diversity + occupancy balance。
2. task collapse -> reduce scale overlap / local-tail coverage。
3. memory high -> support-index no-materialize path。
```

---

# 10. Line C：Manifold-Channel audit

Line C 继续作为 audit，不作为 direction source。

## 必须记录

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
```

## 可视化

```text
fig_linec_by_method.svg
fig_noise_reservoir_tradeoff.svg
fig_source_vs_linec_scatter.svg
fig_tail_vs_source_scatter.svg
fig_projection_value_retention_vs_linec.svg
```

---

# 11. Line X：Synthetic 与 real gate

## 11.1 Synthetic gate

Synthetic X1-X7：

```text
seeds = 0,1,2
loss_interfaces = CE,Brier
train_steps = 200
batch_size = 32
```

Synthetic S3：

```text
>=5/7 task-family pass；
每个 family 至少 2/3 seeds 或 CE/Brier 双接口之一稳定；
source_vs_best_control >= 0.005；
AUCtime_ratio <= 1.0；
CEp99/NLL/ECE non-harm；
LineC majority pass。
```

## 11.2 Real short-run gate

只有 synthetic S3 达成后，才允许 real 3x3：

```text
MNIST / Fashion-MNIST / KMNIST
seeds = 0,1,2
train_size = 1024
val/test = 512
```

Real S5：

```text
source_vs_control >= 0.005 mean；
AUCtime_ratio <= 1.0；
CEp99_delta <= 0.05；
NLL_delta <= 0.02；
ECE_delta <= 0.02；
LineC majority/all pass；
step/memory overhead within budget。
```

---

# 12. Final route definitions

```text
R0-ForbiddenDirectionSource:
  uses validation/test/future/query/LineC/CEp99/NLL/ECE for direction。

R1-NoGenericFMSValue:
  MLP / generic FMS cannot reach synthetic >=5/7。

R2-GenericValueKilledByBasisProjection:
  generic value exists but basis projection retention/cosine fails。

R3-RationalLineCTailUnsafe:
  Rational preserves source but fails LineC/tail/overhead。

R4-NonRATSubstrateMissing:
  Non-RAT still no substrate gate pass。

R5-GenericOnlyNoKANSpecific:
  MLP-FMS works, KAN-FMS does not exceed generic baseline。

S1-GenericFMSPositive:
  MLP/generic FMS reaches synthetic >=5/7。

S2-RationalValuePreserved:
  Rational projection preserves generic value retention/cosine。

S3-KANSpecificSyntheticPass:
  Rational/KAN reaches synthetic >=5/7 and positive KAN-specific delta。

S4-RealShortRunOpened:
  synthetic S3 opens real short-run。

S5-OfficialFunctionalSuccess:
  real 3x3 pass with controls, LineC, tail, efficiency。
```

---

# 13. Required artifacts

```text
v143_route_decision.json
v143_project_progress.csv
v143_code_path_manifest.csv
v143_loss_interface_audit.csv
v143_forbidden_information_audit.csv
v143_functional_value_constraint_manifest.csv
v143_generic_fms_results.csv
v143_generic_fms_controls.csv
v143_rational_projection_audit.csv
v143_rational_fms_results.csv
v143_projection_value_retention.csv
v143_basis_substrate_status.csv
v143_all_basis_substrate_repair.csv
v143_basis_family_telemetry.csv
v143_nonrat_substrate_repair.csv
v143_linec_audit.csv
v143_tail_calibration_audit.csv
v143_failure_table.csv
v143_no_go_boundary.md
v143_next_hypothesis_queue.md
v143_required_manifest.csv
v143_code_review_packet.zip
```

---

# 14. Required figures

```text
fig_progress_by_line.svg
fig_generic_fms_vs_rational_fms.svg
fig_value_retention_projection.svg
fig_projection_rejection_by_role.svg
fig_source_linec_tail_scatter.svg
fig_all_basis_substrate_matrix.svg
fig_nonrat_memory_task_linec.svg
fig_synthetic_task_family_heatmap.svg
fig_route_dashboard.svg
```

---

# 15. Codex 执行要求

Codex 不能再做：

```text
1. 继续 R6/R7/R8/R9/R10 小修；
2. 继续 cover_purity / cover_churn objective；
3. 继续 O7/O8/O9 / oracle DeltaZ；
4. 继续 response dictionary；
5. 继续 BN/BM parameter metric fallback；
6. Non-RAT 未过 substrate gate 就跑 official FMS；
7. MLP-only positive 写成 KAN success。
```

Codex 必须先执行：

```text
1. Generic FMS 200-step / 3-seed confirmation；
2. Rational generic-value-preserving projection audit；
3. Rational K1-K8 synthetic confirmation；
4. All-basis substrate repair，尤其 RBF/FastKAN 与 Rational g8/Horner/Triton variants；
5. Non-RAT strict substrate recheck；
6. LineC/tail audit；
7. final route / no-go / next hypothesis。
```

如果 Codex 只新增小 token 而不执行 value/constraint 分离，route 必须是：

```text
R0-TokenSearchNotMechanismReset
```

---

# 16. 最终判断

v14.2 告诉我们：

```text
1. Functional-first 是对的；
2. FMS 作为 persistent training boundary 有 generic positive；
3. 但把 basis/group telemetry 当 value source 是错的；
4. Rational substrate 能产生 local source，但 LineC/tail/overhead 不闭合；
5. Non-RAT 仍不是 functional substrate；
6. 下一步必须把 value source 与 basis constraint 分开，同时继续 all-basis substrate repair。
```

v14.3 的目标不是再找一个小修 token，而是回答：

$$
\boxed{
\text{KAN basis 能否在不破坏 generic FMS value 的情况下，}
\text{通过 basis constraint 获得额外几何与训练优势？}
}
$$

如果答案是否定的，functional update 作为 KAN-specific breakthrough 就必须收缩；如果答案是肯定的，DG-KAN 才真正有下一步。
