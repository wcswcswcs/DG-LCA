# DG-KAN v15.9：Multi-Line Functional Dynamics + MLP Active + LQ Reanchor + All-Basis 并行加速完整计划

> 版本：v15.9 execution plan  
> 生成时间：2026-05-31（Asia/Singapore）  
> 目标：停止单线等待式推进，恢复 **多线并行 functional update**。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no active B-spline；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific scale；no fake / proxy / CPU offload；functional direction 不使用 validation / test / future / query；LineC / CEp99 / NLL / ECE / AUCtime / Brier 只能作为 audit / gate / failure taxonomy，不能生成方向。

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是寻找一个局部看起来漂亮的 update。总目标是：

$$
\boxed{
\text{在 strict FC-PureKAN / KAN-like carrier 上，通过 functional update 改善训练动力学，}
\text{最终优于 ordinary backprop / AdamW controls。}
}
$$

这里“优于”必须同时包含：

```text
1. source gain：
   update 后模型获得 controls 不能解释的真实训练/泛化收益。

2. trajectory gain：
   即使短期产生 bad debt，长期训练轨迹仍更好。

3. efficiency：
   step / memory / AUC-time 不能靠大量额外计算换取。

4. geometry / tail safety：
   LineC / tail / calibration 可以短期产生 debt，但必须能恢复；
   official promotion 时必须满足严格 gate。

5. carrier validity：
   D-CHE / MLP / LQ / Rational / D-FOU / D-RBF / D-WAV 的结果要分开解释，
   不能把一个 carrier 的局部 positive 写成整体成功。
```

最终要证明的是：

$$
\boxed{
\text{Base + functional dynamics}
>
\text{Base + ordinary optimizer / matched controls}
}
$$

而不是只证明：

$$
\text{某个单步 update 的局部 source 为正。}
$$

---

## 0.2 当前进展简明版

当前最重要的事实是：

```text
1. D-CHE 是当前最强 Non-RAT carrier，历史上已达到 9/9 substrate eligibility。
2. D-CHE split-consensus G7R / metric-only update 曾经出现 source-positive、control-resistant 信号。
3. v15.5 / v15.6 证明：这个 source signal 和 tail / LineC bad event 绑定很紧。
4. v15.8 进一步证明：简单 pulse + recovery、decoupled decay、long horizon H=400/800 没有把 bad debt 转成 productive plasticity。
5. D-FOU / D-RBF / D-WAV 当前仍未稳定到 >=6/9 substrate exploration gate。
6. Rational 仍是稳定 monitor，但 reset / optimizer-state route 被 generic controls 解释过，不应重启。
7. LQ 线历史上有 repaired base survivor 与 late-attach 思路，但后来没有被继续系统推进。
8. MLP functional update 线被我过度降级成 control，这是错误的；必须恢复为主动机制发现线。
```

因此 v15.9 的核心不是继续写一个 D-CHE G-token，也不是坐等一个 D-CHE recovery 实验结束，而是：

$$
\boxed{
\text{D-CHE dynamics、MLP dynamics、LQ late-attach、Rational monitor、All-basis substrate 同时推进。}
}
$$

---

# 1. 本轮核心判断：functional update 应该从“单步好坏”升级为“训练动力学”

## 1.1 不能再只看单步 bad event

过去很多 gate 隐含了一个假设：

```text
如果 functional update 立即造成 tail / LineC / AUC bad event，
那这个 update 就是坏的。
```

这个假设现在必须放松。真实训练可能是：

```text
短期扰动 -> recovery -> consolidation -> 长期收益
```

因此，v15.9 不再只问：

```text
这个 update 当下安全不安全？
```

而问：

```text
这个 update 造成的 bad debt 能不能被后续训练偿还？
source 是否能留下？
controls 是否不能解释？
```

定义短期 debt：

$$
D_m(t, h)
=
\max_{1 \le \tau \le h}
\left[
m(t+\tau)-m_{\text{baseline}}(t+\tau)
\right]_+,
$$

其中 $m$ 可以是：

```text
CEp99 / NLL / ECE / LineC badness / AUC debt / logit RMS debt / margin debt
```

定义恢复率：

$$
R_m(t,h)
=
\frac{D_m(t,h)-D_m(t+h,h_{\text{final}})}
{D_m(t,h)+\epsilon}.
$$

定义 source retention：

$$
S_{\text{ret}}(t,h)
=
\frac{Source(t+h)}
{Source(t+1)+\epsilon}.
$$

一个短期坏 update 只有在满足下面条件时才算 productive perturbation：

```text
1. source 在长期仍保留；
2. tail / LineC / AUC debt 被偿还；
3. controls 不能解释；
4. 不是 decay-only / MLP generic / random pulse 同样能做到。
```

---

## 1.2 weight decay 很重要，但不是唯一偿还机制

本轮把 recovery mechanism 分成多类：

```text
R0 Ordinary AdamW recovery
R1 Decoupled weight decay recovery
R2 Role-wise / basis-wise decoupled decay
R3 Momentum / EMA recovery
R4 LR cooldown / restart recovery
R5 Lookahead / slow-weight consolidation
R6 Schedule-free / long-EMA style recovery
R7 NoOp / abstention after pulse
R8 Random pulse + same recovery controls
```

其中 weight decay / decoupled decay 的角色是：

$$
\boxed{
\text{functional pulse 负责制造 useful plasticity，}
\text{recovery mechanism 负责偿还 bad debt 并巩固 source。}
}
$$

但不能把 weight decay 本身的收益写成 functional update 成功。必须同时比较：

```text
FU-only
Recovery-only
FU + recovery
RandomFU + same recovery
NoOp + same recovery
MLP + same recovery
```

---

# 2. v15.9 不是单线计划，而是五条 functional 线并行

本轮不允许“先等 D-CHE 跑完，再决定 MLP / LQ / all-basis”。所有线并行执行。

```text
Line A: D-CHE G7R / split-consensus productive perturbation dynamics
Line B: MLP functional dynamics active line
Line C: LQ repaired-anchor / snapshot-late-attach functional line
Line D: Rational monitor + dynamics sanity replay
Line E: All-basis substrate acceleration
Line F: Cross-line comparison / mechanism routing
Line Z: route / no-go / exhaustion / next queue
```

---

# 3. Line A：D-CHE split-consensus dynamics harness

## 3.1 目标

Line A 继续使用 D-CHE 作为当前最强 KAN carrier，但不再试图让 G7R 单步安全。它只回答：

$$
\boxed{
\text{G7R source signal 是否能通过后续 recovery dynamics 变成长期收益？}
}
$$

## 3.2 候选程序

所有候选都是预注册 global dynamics program，不是 action bank，不做 dataset/seed 分支。

```text
A0-D-CHE-AdamW
A1-D-CHE-G7R-PulseOnce-AdamWRecovery
A2-D-CHE-G7R-PulseEvery50-AdamWRecovery
A3-D-CHE-G7R-EarlyPulseOnly-AdamWRecovery
A4-D-CHE-G7R-MidPulseOnly-AdamWRecovery
A5-D-CHE-G7R-LatePulseOnly-AdamWRecovery

A6-D-CHE-G7R-Pulse-GlobalDecoupledDecayRecovery
A7-D-CHE-G7R-Pulse-DegreeWiseDecayRecovery
A8-D-CHE-G7R-Pulse-HighDegreeExtraDecayRecovery
A9-D-CHE-G7R-Pulse-ReadoutBasisDecoupledDecayRecovery

A10-D-CHE-G7R-Pulse-MomentumEMARecovery
A11-D-CHE-G7R-Pulse-LRCooldownRecovery
A12-D-CHE-G7R-Pulse-LookaheadConsolidation
A13-D-CHE-G7R-Pulse-ScheduleFreeLongEMARecovery

ACTRL1-D-CHE-RandomMatchedPulse-SameAdamWRecovery
ACTRL2-D-CHE-RandomMatchedPulse-SameDecayRecovery
ACTRL3-D-CHE-NoOpMatchedOverhead
ACTRL4-D-CHE-AdamWExtraStepsMatchedTime
```

A6-A13 不是“再找好动作”。它们都使用同一个 G7R pulse，只改变后续 recovery dynamics。

## 3.3 记录指标

每个 candidate 必须在 horizon：

```text
h = 1, 5, 20, 50, 100, 400, 800
```

记录：

```text
source_vs_best_control_h
source_retention_h
tail_debt_peak
tail_debt_final_h
tail_debt_recovery_rate_h
LineC_debt_peak
LineC_debt_final_h
LineC_debt_recovery_rate_h
AUC_debt_peak
AUC_debt_final_h
AUC_debt_recovery_rate_h
CEp99_curve
NLL_curve
ECE_curve
Brier_curve
LineC_curve
margin_p10_curve
logit_RMS_curve
entropy_curve
degree_energy_curve
high_degree_energy_fraction_curve
decay_update_norm_curve
adam_update_norm_curve
fu_update_norm_curve
decay_vs_fu_cosine_curve
step_time_ratio_curve
memory_ratio_curve
```

## 3.4 判断标准

Exploration positive：

```text
source_vs_best_control_h100 >= 0.005
source_retention_h100 >= 0.40
tail_debt_recovery_rate_h100 >= 0.40
LineC_debt_recovery_rate_h100 >= 0.40
random matched controls do not explain
```

Meaningful positive：

```text
source_vs_best_control_h400 >= 0.005
source_retention_h400 >= 0.50
tail_debt_final_h400 <= 0.50 * tail_debt_peak
LineC_debt_final_h400 <= 0.50 * LineC_debt_peak
AUCtime_h400 <= 1.05
controls fail
```

No-go：

```text
source retained but debt not recovered:
  route = R-A1-ProductiveSignalUnstable

debt recovered but source killed:
  route = R-A2-RecoveryKillsSource

random pulse + same recovery explains:
  route = R-A3-GenericPerturbationRecovery

decay-only explains:
  route = R-A4-DecayOnlyExplains

all recovery mechanisms fail:
  route = R-A5-DCHEG7RDynamicsNoGo
```

---

# 4. Line B：MLP functional dynamics active line

## 4.1 为什么 MLP 不能只当 control

MLP functional update 线必须恢复成主动机制发现线，因为它能回答：

```text
1. productive bad update / recovery 这种机制是否 general？
2. 如果 MLP 上能成功而 D-CHE 不成功，说明问题可能是 KAN carrier；
3. 如果 MLP 和 D-CHE 都失败，说明当前 functional dynamics 定义不成立；
4. 如果 D-CHE 成功而 MLP 不成功，才说明 KAN carrier 有独特价值。
```

## 4.2 MLP 候选程序

```text
B0-MLP-AdamW
B1-MLP-CautiousAdamW
B2-MLP-MGUP
B3-MLP-SplitConsensusMetric
B4-MLP-FunctionalPulseOnce-AdamWRecovery
B5-MLP-FunctionalPulseEvery50-AdamWRecovery
B6-MLP-FunctionalPulse-GlobalDecayRecovery
B7-MLP-FunctionalPulse-MomentumEMARecovery
B8-MLP-FunctionalPulse-LRCooldownRecovery
B9-MLP-FunctionalPulse-LookaheadConsolidation
B10-MLP-AmortizedPersistentFMS-Recheck
BCTRL1-MLP-RandomMatchedPulse-SameRecovery
BCTRL2-MLP-NoOpMatchedOverhead
BCTRL3-MLP-AdamWExtraStepsMatchedTime
```

MLP functional pulse 的 definition 必须和 D-CHE 尽量结构同构：

```text
split-consensus signal subspace；
train-stream gradients only；
no validation/test/future/query；
no LineC/tail/AUC audit direction。
```

## 4.3 MLP 成功解释规则

```text
Case B-success-only:
  MLP succeeds, D-CHE fails.
  结论：functional dynamics 机制可能成立，但 KAN carrier 不合适。

Case D-CHE-success-only:
  D-CHE succeeds, MLP fails.
  结论：KAN basis provides carrier advantage。

Case both succeed:
  结论：generic functional dynamics / optimizer mechanism；仍有价值，但不是 KAN-specific。

Case both fail:
  结论：current dynamics definition no-go。
```

## 4.4 MLP 记录指标

和 Line A 相同，但加入：

```text
hidden_rank_curve
hidden_covariance_drift
feature_norm_curve
classifier_weight_norm_curve
MLP_source_vs_DCHE_source
MLP_debt_recovery_vs_DCHE_recovery
```

---

# 5. Line C：LQ repaired-anchor / snapshot-late-attach functional line

## 5.1 为什么恢复 LQ 线

LQ 历史上不是废线。它曾经有过 repaired base survivor，也有 snapshot late-attach / TPEA-off equivalence 思路。当前 v15 后期过度依赖 D-CHE，可能错过了 LQ 作为 carrier 的价值。

Line C 的目标不是直接 claim LQ success，而是：

$$
\boxed{
\text{恢复 LQ as monitor / carrier candidate，检查 functional dynamics 是否更容易被 LQ 承载。}
}
$$

## 5.2 阶段 C1：LQ re-anchor

候选：

```text
C0-HistoricalLQReference
C1-CurrentLQReproduction
C2-ExactProtocolLQReplay
C3-RepairedLQ-FaninOutputScaleConfirmed
C4-RepeatedCurrentLQ
```

必须记录：

```text
near_pass_rate
macro_delta_vs_MLP
step_q90
memory_ratio
CEp99
margin_p10
ECE
NLL
basis_entropy
lift_condition_number
effective_rank
protocol_hash
candidate_config_hash
data_split_hash
MLP_match_hash
```

通过：

```text
near_pass_rate >= 0.80
macro_delta_vs_MLP >= -0.01
step_q90 <= 1.50
memory_ratio <= 1.05
```

如果 LQ base 不过，Line C functional 不执行，但不能停止其它线。

## 5.3 阶段 C2：snapshot late attach

只在 C1 pass 后执行。

要求 inactive attach 等价：

$$
\max_x
\|z_{\text{late-attach-off}}(x)-z_{\text{LQ checkpoint}}(x)\|_\infty
\leq 10^{-6}.
$$

参数不漂移：

$$
\|\theta_T^{off}-\theta_T^{LQ}\|_\infty
\leq 10^{-6}.
$$

optimizer state 不漂移：

$$
\|m_T^{off}-m_T^{LQ}\|_\infty
\leq 10^{-6}.
$$

## 5.4 阶段 C3：LQ functional dynamics

```text
C5-LQ-G7Analog-PulseOnce-AdamWRecovery
C6-LQ-G7Analog-Pulse-DecayRecovery
C7-LQ-G7Analog-Pulse-MomentumRecovery
C8-LQ-SplitConsensusMetric
C9-LQ-SnapshotLateAttachFunctionalPulse
CCTRL-RandomMatchedPulse-SameRecovery
```

成功标准同 Line A，但必须额外满足：

```text
attach equivalence preserved before active event
no external residual
functional params edge-owned
inactive path exact
```

---

# 6. Line D：Rational monitor + dynamics sanity replay

## 6.1 目标

Rational 不重启 reset / optimizer-state transport route。Line D 只做：

```text
1. no-regression monitor；
2. 同样的 pulse+recovery dynamics sanity check；
3. 判断 productive perturbation 是否只在 D-CHE 出现。
```

候选：

```text
D0-RAT-AdamW
D1-RAT-SplitConsensusMetricPulse-AdamWRecovery
D2-RAT-SplitConsensusMetricPulse-DecayRecovery
D3-RAT-SplitConsensusMetricPulse-MomentumRecovery
DCTRL-RAT-RandomMatchedPulse-SameRecovery
```

如果 Rational 出现 positive，也不能直接 promotion，必须对比 D-CHE / MLP / LQ，并且不能走 reset / controller / action bank。

---

# 7. Line E：All-basis substrate acceleration

## 7.1 目标

D-CHE 不能成为唯一 carrier。D-FOU / D-RBF / D-WAV 必须继续 substrate-only acceleration，但不过 gate 不能进入 official functional proof。

## 7.2 候选方向

### Fourier / D-FOU

```text
D-FOU82-LowFreqIdentityResidualV5
D-FOU83-BandwiseSNRWarmupV5
D-FOU84-PhaseStableBandMixV5
D-FOU85-NoMaterializeLifetimeV5
D-FOU86-HighFrequencyQuarantineV5
```

### RBF / FastKAN

```text
D-RBF80-ActiveCenterOccupancyV5
D-RBF81-WidthConditionGuardV5
D-RBF82-CompactBumpNoDenseV5
D-RBF83-GaussianLocalK4TaskHealthV5
D-RBF84-IdentityResidualWidthWarmupV5
```

### Wavelet / D-WAV

```text
D-WAV69-TriangularSupportV5
D-WAV70-ScaleOccupancyV5
D-WAV71-SupportOverlapDampingV5
D-WAV72-LocalTailCoverageAuditV5
```

## 7.3 Substrate gates

Exploration substrate gate：

```text
family_dataset_seed_pass_count >= 6/9
step_ratio <= 1.75
memory_ratio <= 1.75
mean_delta_vs_MLP >= -0.05
LineC_pass_rate >= 0.30
```

Official FMS/FU eligibility：

```text
family_dataset_seed_pass_count = 9/9
step_ratio <= 1.50
memory_ratio <= 1.50
mean_delta_vs_MLP >= -0.02
LineC_pass_rate >= 0.50
```

D-FOU / D-RBF / D-WAV 未到 exploration gate，不允许 official FU proof，但不能让整轮 hard stop。

---

# 8. Line F：Cross-line routing / mechanism interpretation

Line F 聚合 A/B/C/D/E 结果，形成真正的判断，而不是单线自说自话。

## 8.1 关键比较

```text
D-CHE vs MLP:
  判定 KAN-specific vs generic dynamics。

D-CHE vs LQ:
  判定 carrier coordinate 是否关键。

D-CHE vs Rational:
  判定 Rational 是否更稳定但 source 弱。

D-CHE vs D-FOU/RBF/WAV:
  判定是否值得切换 carrier。

FU+recovery vs recovery-only:
  判定 functional pulse 是否提供独立 source。

FU+decay vs decay-only:
  判定 weight decay 是否只是 regularizer gain。

FU+momentum vs momentum-only:
  判定 recovery 是否来自 generic optimizer dynamics。
```

## 8.2 路由结果

```text
R-F1-DCHESpecificProductiveDynamics
R-F2-GenericMLPFunctionalDynamics
R-F3-CarrierDependentLQOrRationalBetter
R-F4-RecoveryOnlyExplainsGain
R-F5-AllFunctionalDynamicsNoGo
R-F6-AllBasisCarrierBlocked
R-F7-NeedSubstrateArchitectureReset
```

---

# 9. 统一指标表

所有线必须统一记录下列指标，避免每条线各写各的。

## 9.1 Task / source

```text
source_vs_best_control_h1
source_vs_best_control_h20
source_vs_best_control_h50
source_vs_best_control_h100
source_vs_best_control_h400
source_vs_best_control_h800
source_retention_h100
source_retention_h400
source_retention_h800
real_lite_pass_count
dataset_seed_pass_count
```

## 9.2 Debt / recovery

```text
tail_debt_peak
tail_debt_final_h100
tail_debt_final_h400
tail_debt_final_h800
tail_debt_recovery_rate_h100
tail_debt_recovery_rate_h400
LineC_debt_peak
LineC_debt_final_h100
LineC_debt_recovery_rate_h100
AUC_debt_peak
AUC_debt_recovery_rate_h100
debt_integral_tail
debt_integral_LineC
debt_integral_AUC
```

## 9.3 Recovery mechanism

```text
recovery_type
decay_update_norm
momentum_update_norm
adam_update_norm
fu_update_norm
decay_to_fu_norm_ratio
recovery_update_to_fu_cosine
rolewise_decay_norm
high_degree_decay_norm
low_degree_decay_norm
ema_norm
lookahead_sync_count
lr_cooldown_factor
```

## 9.4 Carrier telemetry

```text
carrier_type
basis_family
degree_energy
high_degree_energy_fraction
degree_entropy
basis_occupancy_entropy
projection_retention
split_consensus_snr
effective_rank
condition_number
rational_den_p01
rational_derivative_p99
rbf_width_p01
rbf_width_p99
fourier_high_freq_ratio
wavelet_scale_occupancy
```

## 9.5 Controls / specificity

```text
random_matched_explains
decay_only_explains
recovery_only_explains
MLP_generic_explains
control_equivalent_fraction
kan_specific_delta
mlp_functional_delta
lq_functional_delta
rational_functional_delta
```

## 9.6 Efficiency

```text
step_time_ratio
memory_ratio
AUCtime_ratio
extra_compute_overhead
recovery_overhead
pulse_overhead
samples_per_second
```

---

# 10. 必须生成的可视化

```text
fig_01_multiline_progress_dashboard.svg
fig_02_dche_vs_mlp_vs_lq_source_retention.svg
fig_03_tail_debt_recovery_curves.svg
fig_04_linec_debt_recovery_curves.svg
fig_05_recovery_mechanism_comparison.svg
fig_06_decay_only_vs_fu_decay_decomposition.svg
fig_07_mlp_functional_dynamics_dashboard.svg
fig_08_lq_reanchor_and_late_attach_dashboard.svg
fig_09_allbasis_substrate_heatmap.svg
fig_10_carrier_specificity_matrix.svg
fig_11_source_debt_phase_portrait.svg
fig_12_route_decision_ladder.svg
```

---

# 11. Execution contract：不能再让 Codex 坐等单线结果

## 11.1 必须并行执行

```text
GPU group 0:
  Line A D-CHE dynamics T/A6-A13

GPU group 1:
  Line B MLP functional dynamics

GPU group 2:
  Line C LQ reanchor + snapshot late attach if C1 pass

GPU group 3:
  Line E D-FOU + D-RBF substrate acceleration

GPU group 4:
  Line D Rational monitor + D-WAV + no-regression monitors
```

没有 GPU group 4 时，D-WAV 和 Rational monitor 可以低预算排队，但不能被写成 fail。

## 11.2 最低完成合同

一轮 v15.9 不允许只跑 D-CHE 后停止。最低必须完成：

```text
Line A:
  A0-A13 + ACTRL1-ACTRL4, horizons 1/5/20/50/100；
  top-2 positive-looking candidates extend to h=400；
  if h=400 positive, extend to h=800。

Line B:
  B0-B10 + controls；
  at least h=100 for all；
  top-2 extend to h=400。

Line C:
  C0-C4 LQ reanchor；
  if pass, C5-C9 late-attach functional；
  if fail, write LQBaseNotReanchored but continue other lines。

Line D:
  D0-D3 Rational monitor；
  no reset / no controller。

Line E:
  D-FOU and D-RBF full substrate rows；
  D-WAV low-budget rows；
  D-CHE no-regression。

Line F:
  cross-line route / specificity matrix。

Line Z:
  no-go / next queue / exhaustion certificate.
```

## 11.3 Budget exhaustion certificate

如果某条线未跑，必须写：

```text
line_name
planned_rows
executed_rows
planned_horizons
executed_horizons
planned_datasets
executed_datasets
planned_seeds
executed_seeds
budget_kind
budget_consumed
why_deferred
does_defer_affect_route
next_priority
```

不能只写：

```text
budget exhausted
```

没有 exhaustion certificate 的停止是 invalid early stop。

---

# 12. Stop / continue 制度

## 12.1 Promotion fail-closed

S5 official success 仍然严格：

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
promotion_allowed = 1
```

## 12.2 Exploration continue-open

这些情况不能 hard stop：

```text
D-CHE fails；
MLP succeeds；
MLP fails；
LQ reanchor fails；
Rational monitor fails；
D-FOU/RBF/WAV substrate fail；
decay-only explains one branch；
recovery-only explains one branch；
LineC/tail/AUC immediate fail；
bad debt persists on one carrier；
source killed by one recovery mechanism；
h=100 fails but h=400 not yet run；
h=400 fails but Line B / Line C not yet run。
```

这些只能进入 fallback ladder / route taxonomy，不能阻止其它线执行。

## 12.3 真正 hard stop

只有以下情况可以 hard stop：

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

# 13. 结果解释规则

## 13.1 如果 D-CHE 失败，但 MLP 成功

结论不是 functional update 失败，而是：

```text
functional dynamics may be generic and carrier-dependent；
KAN / D-CHE carrier is failing to harness it。
```

下一步应改 carrier / substrate，不应继续 D-CHE 小修。

## 13.2 如果 MLP 失败，但 D-CHE 成功

这是强 KAN-specific signal：

```text
D-CHE / KAN carrier provides useful functional dynamics。
```

再进入 stricter S4/S5 confirmation。

## 13.3 如果 D-CHE 和 MLP 都成功

结论是 generic functional optimizer / dynamics 进展：

```text
valuable, but not KAN-specific。
```

仍可作为项目成果，但 KAN-specific claim 要谨慎。

## 13.4 如果 LQ 成功而 D-CHE 失败

说明 carrier coordinate 很关键：

```text
D-CHE source-hazard绑定可能是 carrier 问题；
LQ late-attach 更适合 functional dynamics。
```

## 13.5 如果 decay-only / recovery-only 解释收益

不能写 functional update success：

```text
route = R-RecoveryOnlyExplainsGain
```

但可作为 optimizer / training dynamics insight 保留。

## 13.6 如果所有线都失败

只有在所有 line contract 完成后才能写：

```text
R15_9-AllFunctionalDynamicsNoGo
```

如果 Line B 或 Line C 未执行，则不能写 global no-go，只能写：

```text
R15_9-IncompleteFunctionalDynamicsCoverage
```

---

# 14. Codex 执行禁止事项

Codex 不许：

```text
1. 新增 G9/G10/FU9/FU10/F-CHE8/F-CHE9。
2. 使用 action bank / controller / learned selector。
3. reset route。
4. strength / lambda / interval / threshold 大网格。
5. 按 dataset / seed 写 branch。
6. 使用 validation/test/future/query 生成方向。
7. 使用 LineC / CEp99 / NLL / ECE / AUCtime / Brier 生成方向。
8. 把 MLP generic positive 写成 KAN-specific。
9. 把 LQ substrate/base success 写成 functional success。
10. 把 decay-only / recovery-only success 写成 functional success。
11. 把 substrate-only rows 写成 promotion。
12. 缺 Line B / Line C 时写 global no-go。
```

Codex 必须：

```text
1. 同时推进 D-CHE、MLP、LQ、Rational、all-basis。
2. 明确记录每条线执行覆盖。
3. 每个 positive-looking result 都有 matched controls。
4. 每个 fail 都有 failure taxonomy。
5. 每个 deferred 都有 budget exhaustion certificate。
6. 最终 route 必须区分 carrier-specific / generic / recovery-only / incomplete coverage。
```

---

# 15. 本轮最重要的科学问题

v15.9 的核心问题不是：

```text
G7R 能不能立即安全？
```

而是：

$$
\boxed{
\text{functional update 是否可以通过训练动力学被 harness，}
\text{并且这种机制是在 D-CHE、MLP、LQ、Rational 中哪一类 carrier 上成立？}
}
$$

如果答案是：

```text
只在 MLP 成立：
  functional dynamics 是 generic optimizer direction，KAN carrier 需要重做。

只在 D-CHE 成立：
  D-CHE split-consensus 是 KAN-specific carrier。

只在 LQ 成立：
  LQ late-attach carrier 需要重启为主线。

所有都不成立：
  current functional dynamics family no-go，转向 substrate/base-level architecture。
```

---

# 16. 最终总结

v15.9 的最大修正是：

```text
不再坐等 D-CHE 单线；
不再把 MLP 当附属 control；
不再把 LQ 遗忘；
不再把 weight decay 当唯一 recovery；
不再用单步 bad event 否定 functional update；
不再允许 Codex 主线失败后直接 no-go。
```

本轮必须并行验证：

```text
1. D-CHE productive perturbation；
2. MLP functional dynamics；
3. LQ snapshot late-attach；
4. Rational monitor；
5. all-basis substrate；
6. recovery / debt repayment mechanisms；
7. cross-line causal specificity。
```

只有这样，才能真正回答：

$$
\boxed{
\text{functional update 是 KAN-specific breakthrough、generic optimizer insight、carrier problem，还是 current family no-go。}
}
$$
