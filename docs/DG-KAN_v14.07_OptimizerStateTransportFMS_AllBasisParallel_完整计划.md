
# DG-KAN v14.7：Optimizer-State Transport FMS Official Confirmation + All-Basis Parallel Substrate 完整计划

生成日期：2026-05-29（Asia/Singapore）  
适用对象：Codex / 后续执行者 / 项目复盘者  
公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是继续寻找一个局部有效的 functional action。项目的最终目标是：

$$
\boxed{
\text{在 label-free strict FC-PureKAN / KAN-substrate 上，}
\text{通过 functional update 获得比普通 AdamW / backprop 更好的训练过程与模型。}
}
$$

“更好”必须同时满足：

```text
1. 表达力不打折；
2. forward / backward / step / memory 与 MLP 可比；
3. 收敛轨迹更快或至少不慢；
4. 几何更健康：train-probe coupling、signal/reservoir/noise、tail、calibration 不坏；
5. functional update 的收益必须独立于 AdamWParallel、NoOp、RandomMatched、best LR / best control；
6. functional direction 不使用 validation / test / future / query；
7. LineC / CEp99 / NLL / ECE / AUCtime / Brier 只能作为 audit / gate，不能作为 direction source；
8. 不使用 label-informed initialization；
9. 不按 dataset / seed 定制规则；
10. 不重蹈 v9 系列“找好动作 / 找 controller / 找 threshold”的老路。
```

最终要证明的是：

$$
\boxed{
\text{same KAN substrate + functional update}
>
\text{same KAN substrate + ordinary AdamW / controls}.
}
$$

不是证明：

$$
\text{某个 diagnostic action 在某个 seed 上好看。}
$$

---

## 0.2 当前阶段：v14.6.1 的真实状态

v14.6.1 的目标不是继续寻找 action，而是回答：

$$
\boxed{
\text{v14.5 剩余的 Fashion-MNIST seed2 与 KMNIST seed2 是否真是 AUCtime-only trajectory-cost states，}
\text{以及预注册机制 probe 是否能把 coverage 从 7/9 推到 9/9。}
}
$$

v14.6.1 分成两个阶段。

### 初版 v14.6.1

初版 v14.6.1 做了正确的事情：

```text
1. 锁定 v14.4 / v14.5 已有 action-bank rows；
2. 不新增 K-RT8 / K-AUC5 / K-FL3；
3. 不调 strength / lambda / lr / refresh；
4. 不执行 controller；
5. 不用 LineC / CEp99 / NLL / ECE / AUCtime 生成 direction；
6. 只做 missing-state trajectory autopsy 和机制 sufficiency probe。
```

初版结果：

```text
route = R2-MechanismUpperBoundInsufficient
coverage_before = 7 / 9
coverage_after = 7 / 9
controller_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

原因：

```text
P1 optimizer-state transport:
  initially unavailable because historical artifacts lacked AdamW moment / RMS snapshots.

P2 NoOp-aware commit:
  coverage_if_applied = 7 / 9, insufficient.

P3 micro-horizon loss integral:
  median_spearman = 0.142472385108414
  false_accept_count = 5
  insufficient for controller.

P4 delayed projection:
  retrospective rows only, not pre-registered k=1/2/4 probe.
```

初版结论是：不能启动 controller，不能新增 action，必须补齐 optimizer-state telemetry。

### 继续后的 P1 optimizer-state telemetry

用户要求“没有达成则继续”后，合法下一步不是新增 action，而是补齐 P1 所需 optimizer-state telemetry：

```text
AdamW moments
RMS state
post-event projected gradient alignment
moment_staleness_before/after
rms_mismatch_before/after
```

新增 fixed P1 state transport modes：

```text
none
zero_moment_reset
partial_moment_interpolation
rms_recompute_microbatch
moment_transport_projected_grad
```

关键结果：

```text
zero_moment_reset:
  dataset-seed pass = 9 / 9
  pass rows = 18
  mean source = 1.261272

none:
  dataset-seed pass = 3 / 9

partial_moment_interpolation:
  dataset-seed pass = 2 / 9

rms_recompute_microbatch:
  dataset-seed pass = 0 / 9
  mean source strongly negative

moment_transport_projected_grad:
  dataset-seed pass = 1 / 9
```

row-level 关键现象：

```text
Fashion-MNIST seed2:
  AUCDebt before = 0.232769
  AUCDebt after = 0.000000
  source change = +0.848225
  CEp99 change = -12.081368
  NLL change = -0.848225
  ECE change = -0.124221
  LineC change = 0
  pass = 1

KMNIST seed2:
  AUCDebt before = 0.122462
  AUCDebt after = 0.000000
  source change = +1.209654
  CEp99 change = -10.987492
  NLL change = -1.209654
  ECE change = -0.031164
  LineC change = 0
  pass = 1
```

optimizer-state mismatch telemetry：

```text
Fashion-MNIST seed2:
  moment_staleness = 1.174842
  rms_mismatch = 0.905346

KMNIST seed2:
  moment_staleness = 1.020316
  rms_mismatch = 1.065835
```

updated diagnostic route：

```text
route = S4c-MechanismSufficientDiagnostic
minimum_success = S4c-MechanismSufficientDiagnostic
coverage_before = 7 / 9
coverage_after = 9 / 9
best_p1_state_transport_mode = zero_moment_reset
mechanism_probe_gate_pass_count = 1
missing_state_classes = C1-AUCOnlyTrajectoryCost
controller_executed = 0
controller_not_executed_reason = mechanism_sufficient_diagnostic_no_controller_in_v1461
action_search_violation_count = 0
forbidden_information_violation_count = 0
required_artifact_missing_count = 0
official_s5_reached = 0
promotion_allowed = 0
```

## 0.3 当前最重要的科学结论

v14.6.1 的核心进展不是“找到了一个好 action”。  
核心进展是：

$$
\boxed{
\text{FMS event 后，一阶 AdamW momentum stale 是剩余 AUC-only trajectory cost 的主要机制证据。}
}
$$

更具体地说：

```text
1. v14.5 剩余 2/9 不是 LineC blocker；
2. 也不是 source 完全没有；
3. 也不是 tail/calibration endpoint 必然坏；
4. 它们是 AUCtime-only trajectory-cost states；
5. zero_moment_reset 清零 affected coordinates 的 AdamW exp_avg 后，diagnostic coverage 从 7/9 到 9/9；
6. RMS-only recompute 退化严重，所以证据更指向 first-moment stale，不是简单 second-moment mismatch。
```

这说明 functional update 的下一步应该是：

$$
\boxed{
\text{FMS weight update + optimizer-state transport 的联合机制。}
}
$$

而不是：

```text
继续寻找更多好 action；
继续扩 K-RT / K-AUC token；
继续调 controller；
继续改 LineC / tail / AUC audit target。
```

---

# 1. 各条线当前进展百分比

这些百分比不是 artifact 中的官方字段，而是基于 gate、artifact 完整性、机制清晰度和距离 official success 的综合估计。

| 线 | 当前完成度 | 判断 |
|---|---:|---|
| Line R：代码 / provenance / finalizer 审计 | 99% | 工程闭包强，不是当前 blocker |
| Historical FHQ / B320-current | 85% frozen | 历史强，但 label-informed init 已禁用，不能 official |
| Line C 几何审计 | 88% | 审计稳定；当前失败基本不是 LineC 主导 |
| PopRisk / FMS 实现面 | 94% | persistent state、per-example gradient、projection trace、real runner、optimizer-state probe 已成熟 |
| Generic MLP-FMS | 45% | 有 generic signal，但不能写成 KAN success |
| Rational substrate | 85% | 当前最稳定 substrate |
| Rational-FMS synthetic | 70% | 已达 S3，KAN-specific synthetic signal成立 |
| Rational-FMS real-transfer official | 65% | actual best 6/9；diagnostic after zero_moment_reset 为 9/9；official S5 未确认 |
| Optimizer-state transport mechanism | 75% diagnostic / 0% official | zero_moment_reset diagnostic 成立，但未进入 official full runner |
| Train-stream controller | 0% | 当前不应启动 controller；不是 selection problem |
| Action-bank route | 关闭 | action bank 只能做 upper-bound diagnostic，不能作为搜索空间 |
| Wavelet substrate | 35%-40% | 有线索，但 v14.5 best 仍不足，不可 FMS proof |
| Fourier substrate | 35%-40% | single-config 3/9，audit best 4/9，仍非 substrate success |
| RBF / FastKAN substrate | 25% | compact workspace 有进展，但 task-health collapse |
| Chebyshev substrate | 25% | workspace / exact path有进展，task-health / LineC 不闭合 |
| Non-RAT official FMS | 0%-5% | 没有 full substrate，不允许 official proof |
| Official S5 functional success | 0% | 仍未 official 达成 |
| 整体 next-gen MLP claim | 48%-55% | 比 v14.5 更近，但必须先做 official confirmation，不能 promotion |

---

# 2. 独立分析：v14.6.1 到底说明了什么

## 2.1 有进展吗？

有，而且是最近几轮中很关键的一次进展。

v14.5 的状态是：

```text
actual real best = 6 / 9
extended legal diagnostic bank oracle = 7 / 9
controller_executed = 0
```

v14.6.1 初版先正确避免 action search，确认：

```text
remaining 2/9 = C1-AUCOnlyTrajectoryCost
NoOp-aware commit insufficient
micro-horizon integral proxy insufficient
delayed projection retrospective only
```

继续后的 P1 telemetry 又进一步确认：

```text
zero_moment_reset 可以把 coverage_after 推到 9/9
```

这不是旧式 action search，因为：

```text
1. 没有新增 K-token；
2. 没有调 scale/lambda/lr/refresh；
3. 没有启动 controller；
4. P1 mode 只改变 AdamW optimizer-state handling，不改变 FMS functional direction；
5. 不使用 validation/test/future/query；
6. 不使用 LineC / CEp99 / NLL / ECE / AUCtime 生成 direction。
```

所以这轮不是“找到一个好动作”，而是定位到一个真实机制：

$$
\boxed{
\text{FMS 改变参数后，AdamW 的一阶动量仍沿旧坐标推进，造成 trajectory lag / AUC debt。}
}
$$

## 2.2 为什么还不能算成功？

因为 v14.6.1 的成功是 diagnostic，不是 official promotion。

原因有四个：

```text
1. zero_moment_reset 是 P1 fixed-mode diagnostic，不是完整 v14.7 official FMS training protocol；
2. 需要把 optimizer-state transport 纳入正式 runner 的 pre-registered mechanism；
3. 需要重跑 full 3x3 official gate；
4. 需要与 AdamWParallel / NoOp / RandomMatched / best controls 做完整对照，并生成完整 artifact surface。
```

因此当前只能写：

```text
S4c-MechanismSufficientDiagnostic
```

不能写：

```text
S5-OfficialFunctionalSuccess
```

## 2.3 当前真正卡在哪里？

现在 blocker 已经从：

```text
是否有 action 覆盖 9/9？
是否需要 controller？
是否缺 LineC？
是否缺 tail trust？
```

变成：

$$
\boxed{
\text{optimizer-state transport 是否能作为 official, precommit-safe, train-stream-only 的 FMS 机制稳定通过 real 3x3。}
}
$$

更具体地说，下一步要回答：

```text
1. zero_moment_reset 是否在 full official runner 中仍然 9/9？
2. 它是否真的只作用于 affected FMS coordinates？
3. 它是否会伤害原本已 pass 的 7/9？
4. 它是否在不同 FMS methods / Rational substrate 上稳定？
5. 它的 overhead 是否可接受？
6. 它是否比 NoOp / AdamW reset / Random reset / full-moment reset 更有特异性？
```

如果这些都过，它才是 functional update 的真正机制突破。

---

# 3. v9 系列教训：本轮必须避免重犯

v9 系列的主错误是：

$$
\boxed{
\text{action-centric 幻觉：以为 functional 成败取决于找到 / 选择好动作，}
\text{而不是先证明动作的因果价值、合法可观测性、物化执行和 runtime 闭环。}
}
$$

v14.7 必须把 v9 教训写成硬约束：

```text
1. action bank 只能是 upper-bound diagnostic，不是搜索空间；
2. oracle bank < 9/9 时 controller 必须关闭；
3. 不能因为局部 positive row 扩 K-token；
4. 新机制必须来自 missing-state causal decomposition，而不是 local positive row；
5. 每个机制必须有 apply / outcome / control / runtime materialization；
6. risk、value、support、control 必须分开；
7. dataset / seed 只能诊断，不能作为 branch；
8. NoOp-safe / harmless-null / control-equivalent 不能算 good；
9. 若 mechanism probe 不能提高 upper bound，就必须 stop，不许继续小修。
```

v14.6.1 符合这些原则：

```text
1. 没有新增 action；
2. 没有启动 controller；
3. 没有调 threshold；
4. P1 机制来自 missing-state autopsy；
5. P1 机制只改变 optimizer-state transport；
6. diagnostic success 没写成 promotion。
```

v14.7 也必须继续保持这些约束。

---

# 4. 下一步核心方案：v14.7

新计划命名为：

```text
DG-KAN v14.7
Optimizer-State Transport FMS Official Confirmation
+
All-Basis Parallel Substrate
```

核心不是继续 action search，而是将 v14.6.1 的 P1 机制正式化：

$$
\boxed{
\text{FMS event 不只是改参数，还必须同步处理 optimizer state。}
}
$$

具体来说，普通 AdamW 维护一阶与二阶状态：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

FMS event 使部分参数进入新的 functional coordinate / trust region 后，如果 $m_t$ 仍保留旧方向，就可能产生 stale momentum：

$$
\cos(m_t, g_{post}) < 0
$$

或者：

$$
\|m_t\| \gg \|g_{post}\|
$$

造成训练路径 recovery lag，最终体现在 AUCtime debt。

v14.7 的主假设是：

$$
\boxed{
\text{对 FMS-affected coordinates 执行 zero-moment reset，}
\text{可以消除 FMS 后 stale first-momentum 造成的 AUCtime-only trajectory debt。}
}
$$

---

# 5. Line R：Implementation / provenance / no-action-search audit

## 5.1 目标

确认 v14.7 不是 action search，也不是 controller tuning，而是 optimizer-state transport official confirmation。

## 5.2 必须检查

```text
1. 不新增 K-RT8 / K-AUC / K-FL / action token；
2. 不新增 action bank；
3. 不启动 controller；
4. 不按 dataset / seed 分支；
5. 不使用 validation/test/future/query；
6. 不使用 LineC / CEp99 / NLL / ECE / AUCtime / Brier 生成 direction；
7. optimizer-state transport 只根据 FMS affected coordinates 与 train-stream state 执行；
8. zero_moment_reset 不改变 FMS direction；
9. all-basis substrate 继续并行审计。
```

## 5.3 必须记录 artifact

```text
v147_no_action_search_audit.csv
v147_forbidden_information_audit.csv
v147_optimizer_state_transport_manifest.csv
v147_fms_affected_coordinate_manifest.csv
v147_code_review_manifest.csv
v147_required_artifact_manifest.csv
```

## 5.4 Pass / fail

如果出现任一情况，直接 route：

```text
R0-ActionSearchViolation
```

触发条件：

```text
1. 新增 action token；
2. 调 K-RT strength / lambda / lr / refresh 网格；
3. 启动 controller；
4. 用 audit metric 生成 direction；
5. 按 dataset / seed 写 branch；
6. 把 diagnostic success 写成 promotion。
```

---

# 6. Line K：Official optimizer-state transport FMS

## 6.1 目标

把 v14.6.1 P1 diagnostic 中的 `zero_moment_reset` 变成 official FMS mechanism，并与 controls 做完整比较。

## 6.2 方法组

必须至少比较：

```text
K0-RAT-AdamW
K1-RAT-FMS-NoStateTransport
K2-RAT-FMS-ZeroMomentReset-AffectedOnly
K3-RAT-FMS-ZeroMomentReset-AllFMSRoles
K4-RAT-FMS-PartialMomentInterpolation
K5-RAT-FMS-RMSRecomputeMicrobatch
K6-RAT-FMS-MomentTransportProjectedGrad
KCTRL-RandomMomentResetMatchedFraction
KCTRL-ZeroMomentResetRandomCoords
KCTRL-FullAdamWStateReset
KCTRL-NoOpMatchedOverhead
```

注意：

```text
K2 是主候选；
K3/K4/K5/K6 是机制对照；
KCTRL rows 不能 promotion；
K3 如果过而 K2 不过，说明 affected coordinate identification 不够；
KCTRL 如果过，说明 reset 本身可能是 generic optimizer trick，不是 FMS-specific。
```

## 6.3 zero_moment_reset 定义

对 FMS event 影响的参数坐标集合 $\mathcal{I}_{FMS}$，执行：

$$
m_t[i] \leftarrow 0,\quad i\in \mathcal{I}_{FMS}.
$$

不改变：

$$
v_t[i]
$$

不改变：

$$
\theta_t[i]
$$

不改变 FMS direction。

也就是说：

```text
zero_moment_reset 只清除 stale first moment；
不重新计算 RMS；
不改变 weight update；
不改变 loss；
不使用 audit target。
```

## 6.4 affected coordinate 选择

必须写清楚 affected coordinate 来自：

```text
1. FMS writeback trace；
2. projection nonzero indices；
3. basis-role affected groups；
4. train-stream FMS active mask。
```

禁止：

```text
1. 根据 dataset / seed 选 coordinates；
2. 根据 CEp99 / NLL / ECE / AUCtime 选 coordinates；
3. 根据 LineC 选 coordinates；
4. 根据 validation/test 选 coordinates。
```

## 6.5 必须记录字段

```text
dataset
seed
method
fms_method
state_transport_mode
affected_coord_count
affected_coord_fraction
moment_norm_before
moment_norm_after
rms_norm_before
rms_norm_after
moment_staleness_before
moment_staleness_after
rms_mismatch_before
rms_mismatch_after
cos_moment_postgrad_before
cos_moment_postgrad_after
source_vs_best_control
AUCtime_ratio
CEp99_delta
NLL_delta
ECE_delta
LineC_pass
step_time_ratio
memory_ratio
controller_executed
promotion_allowed
```

## 6.6 Official S5 gate

S5 需要：

```text
real_dataset_seed_pass_count = 9 / 9
source_vs_best_control >= 0.005
AUCtime_ratio <= 1.0
CEp99_delta <= 0.05
NLL_delta <= 0.02
ECE_delta <= 0.02
LineC_pass = 1
step_time_ratio <= 1.25
memory_ratio <= 1.25
forbidden audit pass
code review pass
promotion_allowed = 1
```

不允许降低 gate。

## 6.7 关键 failure route

```text
R1-ZeroMomentResetNotReplicated:
  K2 < 9/9。

R2-ResetWorksButNotFMSSpecific:
  K2 过，但 random moment reset / full state reset 也过。

R3-AffectedCoordinateIdentificationFail:
  K2 失败，但 K3 all-FMS-role reset 过。

R4-RMSMismatchDominant:
  K5 RMS recompute 过而 K2 不过。

R5-ProjectedMomentTransportDominant:
  K6 过而 K2 不过。

S5-OfficialFunctionalSuccess:
  K2 过，controls 不过，all gates pass。
```

---

# 7. Line M：MLP optimizer-state analog control

## 7.1 目标

判断 zero_moment_reset 是 KAN/FMS-specific 机制，还是 generic optimizer-state trick。

## 7.2 方法

在 MLP-FMS / MLP-SNRBlend 上执行：

```text
M0-MLP-AdamW
M1-MLP-FMS-NoStateTransport
M2-MLP-FMS-ZeroMomentResetAffected
M3-MLP-FMS-ZeroMomentResetRandomCoords
M4-MLP-FMS-FullAdamWStateReset
```

## 7.3 判断

如果 MLP-M2 也稳定 9/9，而 Rational-K2 没有明显更强：

```text
这是 generic optimizer-state mechanism；
不能 claim KAN-specific。
```

如果 Rational-K2 过，MLP-M2 不过：

```text
KAN/FMS-specific mechanism 成立可能性更强。
```

如果两者都不过：

```text
v14.6.1 diagnostic 可能不 robust，需要重新审计。
```

---

# 8. Line D：All-basis substrate parallel repair

## 8.1 目标

不能因为 Rational-FMS 已接近 S5 就停止其他 basis。所有 active non-BSpline family 继续推进 substrate，尤其 Fourier、Wavelet、RBF/FastKAN、Chebyshev。

## 8.2 Family status

```text
Rational:
  主 FMS carrier，必须做 no-regression monitor。

Wavelet:
  有历史线索，但 v14.5 best only 1/9；
  继续做 D-WAV17/18/19 hardening。

Fourier:
  single-config 3/9，audit best 4/9；
  当前最值得继续的 Non-RAT substrate 之一。

RBF / FastKAN:
  compact workspace open but task-health collapse；
  重点修 center occupancy / width condition / identity residual。

Chebyshev:
  compact workspace open but task-health / LineC fail；
  重点修 low-degree identity residual / high-degree late-enable / degree energy damping。

B-spline:
  frozen。
```

## 8.3 必须执行的 substrate candidates

### Rational monitor

```text
D-RAT-Monitor-v147
```

### Wavelet

```text
D-WAV20-TriangleSupportStable
D-WAV21-ScaleOccupancyHardening
D-WAV22-LocalTailCoverageGuard
```

### Fourier

```text
D-FOU20-LowFreqIdentityResidual
D-FOU21-BandwiseSNRWarmup
D-FOU22-PhaseStableBandMix
```

### RBF / FastKAN

```text
D-RBF20-CompactBumpIdentityResidual
D-RBF21-ActiveCenterOccupancyRepair
D-RBF22-WidthConditionGuard
```

### Chebyshev

```text
D-CHE20-LowDegreeIdentityResidual
D-CHE21-HighDegreeLateEnable
D-CHE22-DegreeEnergyDamping
```

## 8.4 Substrate gate

A candidate becomes substrate only if:

```text
step_ratio <= 1.75
memory_ratio <= 1.75
mean_delta_vs_mlp >= -0.05
worst_delta_vs_mlp >= -0.10
AUCtime_ratio <= 2.0
LineC_pass_rate >= 0.30
```

Strict substrate:

```text
step_ratio <= 1.25
memory_ratio <= 1.25
mean_delta_vs_mlp >= -0.02
worst_delta_vs_mlp >= -0.05
AUCtime_ratio <= 1.25
LineC_pass_rate >= 0.50
```

If substrate gate fails:

```text
Do not run official FMS proof for that family.
```

---

# 9. Line T：Trajectory / optimizer-state autopsy

## 9.1 目标

确认 zero_moment_reset 的机制解释，而不是只看 final gate。

## 9.2 必须记录

```text
AUC_debt_before
AUC_debt_after
recovery_lag_steps
loss_spike_after_fms
loss_spike_after_state_transport
moment_staleness_before
moment_staleness_after
rms_mismatch_before
rms_mismatch_after
cos_moment_postgrad_before
cos_moment_postgrad_after
cos_update_grad_before
cos_update_grad_after
```

## 9.3 可视化

```text
fig_v147_auc_debt_before_after.svg
fig_v147_moment_staleness_before_after.svg
fig_v147_rms_mismatch_before_after.svg
fig_v147_cos_moment_postgrad.svg
fig_v147_recovery_lag_by_dataset_seed.svg
fig_v147_source_auc_tail_linec_matrix.svg
```

## 9.4 成功解释

如果 K2 succeeds：

```text
FMS event creates a functional coordinate shift;
AdamW first-moment state remains aligned with pre-event trajectory;
zeroing affected first moment prevents stale momentum from producing AUC debt.
```

如果 K5 succeeds instead:

```text
The blocker is RMS scale mismatch, not first-moment stale.
```

If no state transport succeeds:

```text
v14.6.1 diagnostic overfit to artifact; mechanism not robust.
```

---

# 10. Line Z：Finalizer / route / stop-go

## 10.1 Required artifacts

```text
v147_route_decision.json
v147_real_results.csv
v147_real_summary.csv
v147_optimizer_state_transport_manifest.csv
v147_fms_affected_coordinate_manifest.csv
v147_state_transport_controls.csv
v147_mlp_state_transport_control.csv
v147_all_basis_substrate_status.csv
v147_trajectory_autopsy.csv
v147_forbidden_information_audit.csv
v147_no_action_search_audit.csv
v147_required_artifact_manifest.csv
v147_code_review_packet.zip
fig_v147_auc_debt_before_after.svg
fig_v147_state_transport_gate_matrix.svg
fig_v147_controls_comparison.svg
fig_v147_all_basis_status.svg
```

## 10.2 Stop-go

```text
If K2 9/9 and controls fail:
  S5 candidate, promotion allowed only after audit/code review.

If K2 <9/9:
  stop zero_moment_reset route; do not add action tokens.

If controls also pass:
  generic reset trick; no KAN-specific claim.

If K3/K5/K6 beat K2:
  update mechanism interpretation.

If all state transport fail:
  return to substrate or FMS design, not action search.
```

---

# 11. v14.7 最终结论模板

## Case A：Official S5

```text
Rational-FMS + affected-coordinate zero_moment_reset reaches S5.
Functional update success is optimizer-state-aware FMS, not action search.
```

## Case B：Diagnostic fails to reproduce

```text
v14.6.1 S4c diagnostic did not reproduce in official runner.
The stale-momentum hypothesis is not robust.
Do not continue with state reset variants unless new mechanism evidence appears.
```

## Case C：Generic optimizer-state reset

```text
Zero moment reset works for MLP / random controls too.
This is a generic optimizer-state trick, not KAN-specific functional geometry.
```

## Case D：Affected coordinate issue

```text
Affected coordinate reset fails but broader FMS-role reset works.
The blocker is identifying FMS-affected coordinates, not state transport itself.
```

## Case E：RMS / projected gradient alternative

```text
RMS or projected-gradient state transport beats zero moment.
Update mechanism interpretation; still no action search.
```

---

# 12. 最终判断

v14.6.1 不是 S5，但它是最近最重要的机制定位之一：

$$
\boxed{
\text{剩余 2/9 不是缺好动作，}
\text{而是 FMS event 后 optimizer first moment stale 造成的 AUC-only trajectory debt。}
}
$$

下一步必须把这个机制做成 official confirmation。  
不能回到 v9 式 action search，不能新增 K-token，不能启动 controller。  
如果 v14.7 成功，functional update 的真正突破点会被重新定义为：

$$
\boxed{
\text{functional parameter update}
+
\text{optimizer-state transport}
=
\text{trajectory-safe functional training}.
}
$$
