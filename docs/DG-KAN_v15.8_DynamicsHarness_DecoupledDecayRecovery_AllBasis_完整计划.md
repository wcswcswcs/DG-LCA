# DG-KAN v15.8：Training-Dynamics Harness + Decoupled Decay Recovery + All-Basis 并行加速完整计划

> 生成时间：2026-05-31 Asia/Singapore  
> 基于：v15.7 `SourceHazardFactorization / CarrierReparam / AllBasisAcceleration` 实验结果复盘  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`  
> 硬约束：strict FC-PureKAN；no active B-spline；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no label-informed init；no validation/test/future/query direction；LineC / CEp99 / NLL / ECE / AUCtime / Brier 只能作为 audit / gate / failure taxonomy，不能生成方向；不新增 G-token / action bank / controller / reset route；promotion fail-closed；exploration continue-open。

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是找到一个局部看起来好的 update。总目标是：

$$
\boxed{
\text{在 strict FC-PureKAN base 上，证明 functional update 能产生 control-resistant、可迁移、低成本、几何健康的训练收益。}
}
$$

最终要证明的是：

$$
\boxed{
\text{PureKAN base + functional update}
>
\text{PureKAN base + ordinary AdamW / strong optimizer controls}
}
$$

并且必须同时满足：

```text
1. 表达力不打折；
2. forward / backward / step / memory 与 MLP 可比；
3. 收敛路径更健康，AUC-step / AUC-time 不输；
4. 几何更好：train-probe coupling、signal/reservoir/noise、tail、calibration 不坏；
5. 收益必须击败 NoOp / RandomMatched / AdamWParallel / SameActiveFraction / GenericOptimizerState / MLP analog controls；
6. 不依赖 label-informed init；
7. 不依赖 CE-tail / LineC / validation / test / future / dataset branch 生成方向。
```

## 0.2 v15.7 当前真实状态

v15.7 真实执行了 Line R/S/U2/K/G/B/C/P/D/M/X/Z，并生成 required artifacts。核心 route 为：

```text
route = R4-StabilizationKillsSource
minimum_success = S1-SourceSignalObservable
promotion_allowed = 0
real_lite_pass_count = 0 / 9
source_vs_best_control_mean = 0.0
bad_event_fraction = 1.0
```

v15.7 的最重要结果不是“没有信号”，而是：

```text
1. split-consensus / G7R 类 signal 仍然存在；
2. 但当前 carrier reparameterization 没有把 source 从 hazard 中剥离出来；
3. 在 M0..M5 六种 metric/space 下，真正 source-carrying sources U0/G7R、U1/Q2、U2/Q4、U3/Q5 都没有合法 decoupling support；
4. U4 random 与 U5 AdamW control 反而能在 U2 readback 上出现 decoupling support，说明 metric-space 不是完全不能表达 decoupling；
5. 失败更像是 G7R/Q source-carrying direction 本身绑定在 hazard tangent 上；
6. v15.7 内继续 G9/G10、N7/Q6、action/controller/reset 都会变成 audit-directed search。
```

## 0.3 本轮重新解释：不能只盯局部坏化

v15.6 / v15.7 之前的解释是：

```text
source 与 hazard 共线；
去 hazard 会杀 source；
当前 carrier 可能不行。
```

这只说对了一半。更深一层是：

$$
\boxed{
\text{局部 bad event 未必代表 update 错；它可能是有用塑性扰动带来的 training debt。}
}
$$

也就是说，G7R 的局部 bad event 可能有三种含义：

```text
A. unrecovered damage：坏化不可恢复，最终无价值；
B. productive plasticity：短期坏化，但后续训练恢复并保留 source；
C. control-equivalent noise：局部 positive 被 controls 解释。
```

v15.7 证明了静态分离失败，但还没有回答：

$$
\boxed{
\text{G7R 造成的 hazard 是不可恢复 damage，还是可偿还的 training debt？}
}
$$

v15.8 的核心就是把判断从 “单步是否安全” 改成 “长期训练动力学是否更好”。

---

# 1. 当前各线进展百分比

| 线 | 当前完成度 | 判断 |
|---|---:|---|
| 代码 / provenance / finalizer 审计 | 99% | required / forbidden / no-action-search / contract 基本闭合 |
| Historical B320 / FHQ anchor | 85% frozen | 历史强，但 label-informed init 禁用，不能作 official final claim |
| Line C 几何审计 | 88% | 审计成熟；只能解释失败，不能生成方向 |
| PopRisk / FU / FMS infrastructure | 95% | per-example gradient、second moment、projection telemetry、controls 都成熟 |
| Current FU / FMS / proximal / ST-FU family | 0%-3% | v15.0-v15.3 基本 no-go |
| Split-consensus signal observability | 60% | G7R source signal 可观测，v15.7 仍有 source lineage |
| Split-consensus source strength | 45%-50% | source signal 真实，但 official source_vs_best_control 未形成 promotion row |
| Static source-hazard separability | 0%-5% | v15.6/v15.7 都显示 source/hazard 静态分离失败 |
| Dynamic recovery / debt-repayment understanding | 0%-5% | v15.8 新主线，尚未执行 |
| Decoupled weight decay as recovery mechanism | 5%-10% | 已作为 deconfound/control 存在，但未作为 recovery/consolidation 主机制 |
| D-CHE substrate | 85% | 当前最强 Non-RAT carrier，历史 9/9 eligibility |
| D-CHE stable real-transfer FU | 10%-15% | G7R/G7S/N/Q/K 均未开 S4/S5 |
| Rational substrate | 85% | 稳定；reset / optimizer-state route 已被 generic confound 打回 |
| D-FOU substrate | 20%-30% | v15.7 仍未打开 >=6/9；历史 6/9 未稳定复现 |
| D-RBF / FastKAN substrate | 20%-30% | workspace/telemetry 有线索，task-health 不稳 |
| D-WAV substrate | 15%-20% | 弱线索，低预算保留 |
| Non-D-CHE official FU proof | 0%-5% | 当前不可打开 |
| Official S5 functional success | 0% | 尚未达成 |
| 整体 next-gen MLP claim | 28%-36% | 有 signal，无 dynamic harness / stability / recovery；不能 promotion |

---

# 2. v15.7 结果独立分析

## 2.1 有进展吗？

有，但不是能力进展。v15.7 的价值是把静态 source-hazard 解耦路线基本排除了。

最新人工复核说明，U2 gate 初版误把 U4 RandomSameSourceNorm 与 U5 AdamW control 的 decoupling support 计入 S2，修复后只有 U0/G7R 与 U1..U3 Q-source readback eligible；结果 `eligible_decomposition_support_rows = 0`，route 回到 `R4-StabilizationKillsSource`。这说明先前的 S2 并不是合法 source-carrying 解耦，而是 control-only support。

## 2.2 关键数据

Line U2 的 source-hazard metric-space colinearity：

```text
line_u2_rows = 378
source_hazard_colinear_rows = 275
eligible_decomposition_support_rows = 0
mean_hazard_overlap = 0.7798357372044058
mean_source_retention_after_null = 0.2547888235569247
```

其中真正 source-carrying sources：

```text
U0 G7R-original-replay:
  hazard_overlap = 0.9800781298566748
  source_retention_after_null = 0.02022690376290686
  support_rows = 0 / 54

U1 Q2:
  hazard_overlap = 0.980966752326047
  source_retention_after_null = 0.019371475361333945
  support_rows = 0 / 54

U2 Q4:
  hazard_overlap = 0.9807703351532971
  source_retention_after_null = 0.0195556350212832
  support_rows = 0 / 54

U3 Q5:
  hazard_overlap = 0.9735685178527126
  source_retention_after_null = 0.034100033058267504
  support_rows = 0 / 54
```

而 controls：

```text
U4 RandomSubspaceSameRank:
  hazard_overlap = 0.0071097214154347225
  source_retention_after_null = 0.9999322653920563
  support_rows = 54 / 54

U5 D-CHE-AdamW:
  hazard_overlap = 0.5414301264617178
  source_retention_after_null = 0.685259434360045
  support_rows = 46 / 54
```

这说明 metric-space 本身有能力表达 decoupling，但 **G7R/Q source-carrying direction 当前绑定在 hazard tangent 上**。

## 2.3 这是不是说明 G7R 不行？

还不能这么说。v15.7 只证明：

```text
静态投影 / carrier reparameterization 没能分离 source 和 hazard。
```

它没有证明：

```text
G7R 的短期 hazard 长期一定不可恢复。
```

因此，v15.8 不应该继续静态分离小修，也不应该直接杀掉 G7R，而应该做 **long-horizon training dynamics audit**。

---

# 3. 当前真正 blocker

当前 blocker 不是：

```text
缺二阶矩；
缺 cautious alignment；
缺 decoupled weight decay；
缺 proximal solver；
缺 split-transfer objective；
缺 split-consensus signal；
缺 trust scalar；
缺 hazard-null projection；
缺 carrier reparameterization；
缺 matched controls；
缺 fallback ladder。
```

当前真正 blocker 是：

$$
\boxed{
\text{我们还不知道 G7R 的 bad event 是不可恢复 damage，还是可偿还 training debt。}
}
$$

如果是 unrecovered damage，则当前 D-CHE/G7R 需要 substrate/base-level 重构。  
如果是 productive plasticity，则 functional update 的正确形态不是“每步都安全”，而是：

$$
\boxed{
\text{functional perturbation}
+
\text{recovery dynamics}
+
\text{decoupled decay consolidation}
}
$$

---

# 4. v15.8 核心假设

## H1：G7R 的 local bad event 可能是 recoverable training debt

G7R 当前表现为：

```text
source signal 强；
hazard / bad event 强；
静态分离失败。
```

这可能意味着 G7R 是 productive perturbation。必须测：

```text
G7R pulse 后，普通训练能否在 H=20/50/100 steps 内恢复 tail/LineC；
source 是否仍然保留；
AUCtime 是否最终优于 matched controls。
```

## H2：weight decay / decoupled decay 可能是 recovery/consolidation 机制，而不只是 control

此前 decoupled decay 主要作为 deconfound/control：

```text
decay-only 是否解释 FU gain？
FU + decay 是否只是 regularization？
```

这还不够。v15.8 把 decay 提升为训练动力学机制：

$$
\boxed{
\text{FU pulse 负责制造 plasticity；decoupled decay 负责偿还 training debt。}
}
$$

## H3：坏 update 不一定要避免，但坏 debt 必须可偿还

定义短期 debt：

$$
D_{tail}(t,H)=\sum_{\tau=t}^{t+H}\max(0, Tail(\tau)-Tail(t)).
$$

定义恢复：

$$
R_{tail}(t,H)=Tail(t+1)-Tail(t+H).
$$

类似记录：

$$
D_{LineC}, D_{AUC}, D_{calib}, R_{LineC}, R_{AUC}, R_{calib}.
$$

一个 functional update 可以短期产生 debt，但必须满足：

```text
source retained；
debt recovered；
controls cannot explain；
long-horizon AUC/time not worse。
```

---

# 5. Line R：代码 / provenance / 禁止项审计

## 5.1 目标

确认 v15.8 不是 action search，也不是 audit-directed update。

## 5.2 必须记录

```text
required_artifact_missing_count
forbidden_information_violation_count
no_action_search_violation_count
cpu_offload_used
fake_proxy_used
uses_validation_test_future_query_for_direction
uses_LineC_CEp99_NLL_ECE_AUCtime_Brier_for_direction
uses_dataset_name_branch
uses_seed_specific_scale
uses_label_informed_init
```

## 5.3 hard stop

只要下面任一项非零，立即 hard stop：

```text
required_artifact_missing_count > 0
forbidden_information_violation_count > 0
no_action_search_violation_count > 0
uses_validation_test_future_query_for_direction = 1
uses_LineC_CEp99_NLL_ECE_AUCtime_Brier_for_direction = 1
uses_dataset_name_branch = 1
uses_seed_specific_scale = 1
cpu_offload_used = 1
fake_proxy_used = 1
```

---

# 6. Line T：G7R pulse + recovery dynamics audit

## 6.1 目标

判断 G7R 的局部 hazard 是否是 recoverable training debt。

## 6.2 固定方法

不新增 G9/G10，不改 G7R direction。只比较 schedule：

```text
T0-D-CHE-AdamW
T1-G7R-PulseOnce-then-AdamWRecovery
T2-G7R-PulseEvery50-then-AdamWRecovery
T3-G7R-PulseEarlyOnly-then-AdamWRecovery
T4-G7R-PulseMidOnly-then-AdamWRecovery
T5-G7R-PulseLateOnly-then-AdamWRecovery
TCTRL-RandomMatchedPulse-then-AdamWRecovery
TCTRL-NoOpMatchedOverhead
TCTRL-AdamWExtraStepsMatchedTime
```

注意：T1..T5 不是新 update token。它们只改变 **同一个 G7R pulse 的时间安排**，目的是测训练动力学，不是找好动作。

## 6.3 训练阶段定义

```text
early: step in first 20% budget
mid: step in 40%-60% budget
late: step in 75%-90% budget
```

不按 dataset / seed 分支。

## 6.4 Recovery horizons

每次 pulse 后记录：

```text
horizon = 1, 5, 20, 50, 100 steps
```

必须记录每个 horizon 的：

```text
source_vs_best_control_h
AUCtime_ratio_h
CEp99_delta_h
NLL_delta_h
ECE_delta_h
Brier_delta_h
LineC_pass_h
CouplingR2_delta_h
NoiseSignalLeak_delta_h
RealSignalReservoirRatio_delta_h
train_loss_delta_h
val_loss_delta_h audit only
margin_p10_delta_h
logit_rms_delta_h
entropy_delta_h
```

## 6.5 判断标准

### Productive plasticity exploration

```text
source_vs_best_control_h100 >= 0.005
tail_debt_recovery_rate >= 0.60
LineC_debt_recovery_rate >= 0.60
AUCtime_ratio_h100 <= 1.05
controls fail
```

### Damage no-go

```text
source retained but tail/LineC debt recovery_rate < 0.30
or
source not retained and bad debt remains
```

### Control-equivalent no-go

```text
random matched pulse or AdamW extra-step matched-time explains source and recovery.
```

## 6.6 可视化

```text
fig_t_pulse_source_over_horizon.svg
fig_t_tail_debt_recovery.svg
fig_t_linec_debt_recovery.svg
fig_t_auc_debt_vs_source_retention.svg
fig_t_pulse_stage_comparison.svg
fig_t_controls_recovery_overlay.svg
```

---

# 7. Line W：Decoupled weight decay / damping recovery

## 7.1 目标

把 weight decay 从“control / deconfound”升级为 recovery / consolidation 机制。

核心问题：

$$
\boxed{
\text{G7R pulse 的 bad debt 能否被 decoupled decay / role-wise damping 偿还？}
}
$$

## 7.2 分离三类更新

每步训练必须分解为：

$$
\Delta\theta_t
=
\Delta\theta_{AdamW,t}
+
\Delta\theta_{FU,t}
+
\Delta\theta_{decay,t}.
$$

其中：

```text
AdamW / gradient:
  普通下降。

FU / G7R pulse:
  制造 split-consensus source movement。

Decoupled decay:
  不参与 FU direction，不伪装成 source；只做 shrink / damping / recovery。
```

## 7.3 预注册 decay mechanisms

```text
W0-NoExtraDecay-AdamWDefaultOnly
W1-GlobalDecoupledWeightDecayRecovery
W2-DegreeWiseDecoupledDecay
W3-HighDegreeExtraDecay
W4-ReadoutBasisDecoupledDecay
W5-SignalReservoirDualDecay
W6-RationalNoRegressionDecayMonitor
```

W1..W5 不允许按 dataset/seed 调参，只能使用全局固定设置。

## 7.4 需要记录的 decay 指标

```text
decay_update_norm
fu_update_norm
adam_update_norm
decay_to_fu_norm_ratio
decay_vs_fu_cosine
decay_vs_adam_cosine
rolewise_decay_norm
low_degree_decay_norm
high_degree_decay_norm
readout_decay_norm
basis_decay_norm
source_retention_after_decay
hazard_proxy_after_decay
tail_debt_after_decay
linec_debt_after_decay
```

## 7.5 比较矩阵

必须同时跑：

```text
AdamW only
Decay-only
G7R pulse only
G7R pulse + default AdamW recovery
G7R pulse + W1
G7R pulse + W2
G7R pulse + W3
G7R pulse + W4
G7R pulse + W5
Random pulse + same decay
NoOp + same decay
MLP same pulse/decay analog
```

## 7.6 成功标准

Decay recovery exploration：

```text
G7R + decay source_retention_h100 >= 0.50
G7R + decay tail_debt_recovery_rate >= 0.60
G7R + decay LineC_debt_recovery_rate >= 0.60
G7R + decay source_vs_best_control_h100 >= 0.005
Decay-only does not explain source
Random pulse + same decay does not explain source
MLP analog does not fully explain KAN-specific delta
```

如果 decay-only 成功：

```text
route = R-W-DecoupledDecayOnlyExplainsGain
```

如果 G7R + decay 成功但 MLP analog 也成功：

```text
route = S-W-GenericFunctionalDecayDynamics
```

如果 D-CHE G7R + decay 明显强于 MLP / random / decay-only：

```text
route = S-W-KANSpecificFunctionalDecayRecovery
```

---

# 8. Line H：Long-horizon consolidation

## 8.1 目标

判断短期 bad event 是否能在更长 training horizon 中转为长期收益。

## 8.2 设置

在 low-budget real-lite 后，选择 top candidates 进入：

```text
train_steps = 400, 800
pulse schedules = T1, T2, T3
recovery = AdamW, W1, W2, W3, W4, W5
```

不使用 validation/test/future/query 生成方向。validation/test 只用于最终 audit。

## 8.3 必须记录

```text
source_vs_best_control_curve
AUCtime_curve
CEp99_curve
NLL_curve
ECE_curve
LineC_curve
Debt_tail_curve
Debt_linec_curve
Recovery_tail_curve
Recovery_linec_curve
source_retention_curve
parameter_norm_curve
decay_norm_curve
basis_degree_energy_curve
high_degree_fraction_curve
```

## 8.4 成功标准

Long-horizon exploration：

```text
source_vs_best_control_final >= 0.005
AUCtime_ratio_final <= 1.05
tail_debt_final <= 0.40 * tail_debt_peak
LineC_debt_final <= 0.40 * LineC_debt_peak
control_equivalent_fraction <= 0.50
```

---

# 9. Line B：Train-stream bad-event proxy 继续 readback，不做 direction

## 9.1 目标

继续评估 bad-event proxy 是否有助于理解 debt，但不允许用它生成 update direction。

## 9.2 记录

```text
split_loss_disagreement
recovery_lag
logit_rms_drift
entropy_collapse
margin_p10_drift
update_cosine_to_adam
loss_q90_q95
projection_retention_drift
source_retention_horizon
hazard_proxy_horizon
```

## 9.3 判断

如果 proxy leaveout min AUC 仍 <0.60：

```text
不能用于 gating；
只能用于 failure taxonomy。
```

如果 proxy leaveout min AUC >=0.65：

```text
可以作为下一版 recovery monitor；
仍不能在 v15.8 里生成方向。
```

---

# 10. Line D：All-basis substrate 并行推进

## 10.1 目标

D-CHE 不能成为唯一 carrier。继续推进 D-FOU / D-RBF / D-WAV，但不允许没过 substrate gate 就进入 official FU proof。

## 10.2 Candidates

```text
D-FOU77-LowFreqIdentityResidualV6
D-FOU78-BandwiseConsensusMetricV3
D-FOU79-PhaseStableBandMixV3
D-FOU80-NoMaterializeLifetimeV5
D-FOU81-HighFrequencyQuarantineV3

D-RBF75-ActiveCenterOccupancyV5
D-RBF76-WidthConditionGuardV5
D-RBF77-CompactBumpNoDenseV5
D-RBF78-GaussianLocalK4TaskHealthV3
D-RBF79-CenterSplitConsensusMetricV2

D-WAV65-TriangularSupportV6
D-WAV66-ScaleOccupancyV4
D-WAV67-SupportOverlapDampingV4
D-WAV68-LocalTailCoverageAuditV3

D-CHE-no-regression
Rational-no-regression
```

## 10.3 Substrate exploration gate

```text
family_dataset_seed_pass_count >= 6/9
step_ratio <= 1.75
memory_ratio <= 1.75
mean_delta_vs_MLP >= -0.05
worst_delta_vs_MLP >= -0.10
LineC_pass_rate >= 0.30
```

Official FU eligibility：

```text
family_dataset_seed_pass_count = 9/9
```

未过 substrate gate时：

```text
不能进入 official FU proof；
不能 hard stop 整轮；
必须写 family-specific failure taxonomy。
```

---

# 11. Line M：MLP / generic controls

## 11.1 目标

防止把 generic optimizer / generic decay dynamics 写成 KAN-specific。

## 11.2 Controls

```text
MLP-AdamW
MLP-CautiousAdamW
MLP-MGUP
MLP-G7AnalogPulse
MLP-G7AnalogPulsePlusDecay
MLP-RandomPulseSameNorm
MLP-NoOpMatchedOverhead
```

## 11.3 KAN-specific delta

$$
\Delta_{KAN-specific}
=
(G_{KAN}-C_{KAN})-(G_{MLP}-C_{MLP}).
$$

Exploration positive only if：

```text
Delta_KAN_specific > 0
MLP/generic controls do not explain source + recovery
```

---

# 12. Line C：Geometry / tail / transfer audit

Line C 仍然只做 audit。

必须记录：

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
AUCtime_ratio
```

可视化：

```text
fig_c_source_vs_tail_debt.svg
fig_c_source_vs_linec_debt.svg
fig_c_recovery_curves.svg
fig_c_signal_reservoir_over_horizon.svg
fig_c_decay_recovery_overlay.svg
```

---

# 13. Route definitions

```text
S1-SourceSignalObservable:
  G7R / split-consensus source signal confirmed.

S2-ProductivePlasticityObserved:
  short-term debt appears but long-horizon recovery preserves source.

S3-DecoupledDecayRecoveryPositive:
  G7R + decoupled decay recovery beats controls and repays debt.

S4-RealTransferExplorationPositive:
  real_lite >=6/9 with source retained and debt recovered.

S5-OfficialFunctionalSuccess:
  official 9/9 full gate.

R1-UnrecoveredDamage:
  source/hazard remains bad after recovery horizon.

R2-DecayedAwaySource:
  decay repays debt but removes source.

R3-DecayOnlyExplainsGain:
  decay-only explains source/recovery.

R4-RandomPulseExplainsGain:
  random matched pulse + same recovery explains result.

R5-GenericMLPExplainsGain:
  MLP analog explains gain.

R6-AllBasisCarrierBlocked:
  non-D-CHE families fail substrate gate.

R7-CurrentSplitConsensusDynamicsNoGo:
  no productive plasticity and no decay recovery under legal settings.
```

---

# 14. Success standards

## 14.1 S2 productive plasticity

```text
source_vs_best_control_h100 >= 0.005
tail_debt_recovery_rate >= 0.60
LineC_debt_recovery_rate >= 0.60
AUCtime_ratio_h100 <= 1.05
random/control fail
```

## 14.2 S3 decoupled decay recovery

```text
G7R + decay source_retention_h100 >= 0.50
G7R + decay source_vs_best_control_h100 >= 0.005
tail_debt_final <= 0.40 * tail_debt_peak
LineC_debt_final <= 0.40 * LineC_debt_peak
Decay-only does not explain
Random pulse + same decay does not explain
MLP analog does not fully explain
```

## 14.3 S4 real-transfer exploration

```text
real_lite_pass_count >= 6/9
source_vs_best_control_mean >= 0.005
AUCtime_median <= 1.05
tail debt recovered
LineC debt recovered
controls fail
```

## 14.4 S5 official success，不降低

```text
real_dataset_seed_pass_count = 9/9
source_vs_best_control >= 0.005
AUCtime_ratio <= 1.0
CEp99_delta <= 0.05
NLL_delta <= 0.02
ECE_delta <= 0.02
LineC_pass = 1
step_time_ratio <= 1.25
memory_ratio <= 1.25
controls fail
forbidden audit pass
no-action-search audit pass
code review pass
promotion_allowed = 1
```

---

# 15. Stop / continue 制度

## 15.1 Promotion fail-closed

S5 不到，永远不能 promotion。

## 15.2 Exploration continue-open

这些不能 hard stop：

```text
Line T shows unrecovered damage；
Line W decay-only explains gain；
Line H long-horizon fails；
Line B proxy leaveout fail；
Line D all-basis fail；
MLP/generic controls positive；
D-CHE real-lite <4/9；
LineC/tail/AUC immediate fail；
overhead high；
source killed by decay；
random pulse explains result。
```

它们必须进入 fallback ladder / failure taxonomy / exhaustion certificate，而不能直接停。

## 15.3 真正 hard stop

只有这些可以 hard stop：

```text
required_artifact_missing_count > 0
forbidden_information_violation_count > 0
no_action_search_violation_count > 0
direction uses validation/test/future/query
direction uses LineC/CEp99/NLL/ECE/AUCtime/Brier
dataset-name branch
seed-specific scale
action token / controller / action bank / reset route
fake / proxy / CPU offload
```

---

# 16. 加速执行计划

v15.8 必须并行执行。

```text
GPU group 0:
  Line T G7R pulse + AdamW recovery horizons

GPU group 1:
  Line W G7R pulse + decoupled decay recovery

GPU group 2:
  Line H long-horizon consolidation for top T/W candidates

GPU group 3:
  Line D D-FOU / D-RBF substrate acceleration

GPU group 4 if available:
  D-WAV + D-CHE/Rational no-regression + Line M controls
```

最低完成合同：

```text
Line R:
  all audits pass.

Line T:
  T0..T5 + controls across 3x3 real-lite; horizons 1/5/20/50/100 recorded.

Line W:
  W0..W6 + decay-only / random-pulse / NoOp controls.

Line H:
  at least top-2 T/W candidates x 400-step recovery; if positive, extend to 800-step.

Line B:
  readback-only proxy table with leaveout.

Line D:
  D-FOU and D-RBF full rows; D-WAV low-budget; D-CHE/Rational no-regression.

Line M:
  MLP/generic controls for every positive-looking result.

Line C:
  debt / recovery / geometry / tail audit curves.

Line Z:
  route, no-go, failure taxonomy, exhaustion certificate, next hypothesis queue.
```

---

# 17. 必须生成的可视化

```text
fig_v158_progress_dashboard.svg
fig_t_pulse_source_over_horizon.svg
fig_t_tail_debt_recovery.svg
fig_t_linec_debt_recovery.svg
fig_t_stage_timing_comparison.svg
fig_w_decay_norm_decomposition.svg
fig_w_decay_recovery_vs_source.svg
fig_h_long_horizon_source_tail_linec.svg
fig_d_allbasis_substrate_status.svg
fig_m_kan_specific_delta.svg
fig_c_signal_reservoir_recovery.svg
fig_route_decision_tree.svg
```

---

# 18. 最终判断

v15.7 不是完全没价值。它证明：

```text
1. source signal 并没有完全消失；
2. 静态 source-hazard 分离失败；
3. 当前 carrier reparameterization 没有找到合法 decoupling；
4. controls 显示 metric-space 有 decoupling 能力，但 source-carrying direction 本身被 hazard 绑定；
5. 继续 G-token / trust scalar / nullspace 小修没有意义。
```

但 v15.7 还没有回答：

$$
\boxed{
\text{G7R 的 bad event 是不可恢复 damage，还是可恢复 training debt？}
}
$$

v15.8 的核心就是把 functional update 从“单步安全更新”重写为：

$$
\boxed{
\text{functional perturbation}
+
\text{recovery dynamics}
+
\text{decoupled decay consolidation}.
}
$$

如果 v15.8 证明 G7R debt 可被偿还，functional update 主线继续推进。  
如果 v15.8 证明 debt 不可偿还，或者收益被 decay-only/random/MLP controls 解释，就必须关闭当前 split-consensus dynamics family，转向 substrate/base-level redesign，而不能再继续小修。
