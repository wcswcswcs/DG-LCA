# DG-KAN v9.2.43 Legal Controller Calibration 与 Paired Replay Closure 完整实验计划

> 本计划基于 v9.2.42 `Fresh Multi-Stratum Control-Gap Functional Validation` 的真实复盘制定。  
> v9.2.42 的 terminal route 是：
>
> ```text
> route = R5-OracleHighLegalScoreLow
> base_candidate = LQ-t2-h256
> success_v9242_strict_purekan_functional = False
> success_v9242_full_functional = False
> success_v9242_external_ready = False
> ```
>
> v9.2.42 的关键事实是：
>
> ```text
> fresh rows = 8640
> fresh real events = 1440
> signal strata = 8
> attach candidates = 3
>
> frozen S7 on fresh rows:
>   AUC = 0.752942
>   corr = 0.338265
>   precision = 0.906977
>   coverage = 0.029861 < 0.03
>   bad-event = 0.0
>   frozen_s7_pass = 0
>
> score matrix:
>   no legal score official-pass
>   S13-Oracle pass only, but posthoc/oracle, not official
>
> role-wise:
>   A3-LateAttachRoleWiseFT7EdgeCarrier carrier pass = 1
>   rolewise value pass = 0
>
> downstream:
>   LDO / LSO / paired replay not opened
>
> blocker:
>   oracle_good_events_exist_but_legal_scores_failed
> ```
>
> 因此，本轮的核心判断是：
>
> $$
> \boxed{
> \text{functional carrier 已经能产生 good events；真正 blocker 是 legal controller 找不到足够多、可泛化、control-resistant 的 good events。}
> }
> $$
>
> v9.2.43 不应继续修 base，也不应回到 dataset-specific tuning，更不应只把 S7 coverage 阈值从 $0.029861$ 人工放宽到 $0.03$。  
> v9.2.43 必须把问题升级为：**如何在不使用 dataset name、不使用 validation/test metric、不使用 posthoc outcome at commit 的条件下，校准一个 legal controller，使它稳定达到 precision / coverage / bad-event / leave-out / paired-replay gate。**
>
> 本计划采用并行 runner，一次性执行：
>
> ```text
> Lane A:
>   v9.2.42 boundary reproduction and score-gap autopsy
>
> Lane B:
>   frozen S7 threshold-free diagnostic and calibration split
>
> Lane C:
>   legal controller calibration family
>
> Lane D:
>   fresh multi-stratum expansion v2
>
> Lane E:
>   leave-dataset-out / leave-stratum-out
>
> Lane F:
>   official paired replay
>
> Lane G:
>   short-run scout if paired replay passes
>
> Lane H:
>   fallback: carrier reset only if oracle drops or paired replay remains control-equivalent
> ```

---

## 0. 硬约束

本计划继续遵守：

```text
no teacher
no self-teacher
no distillation
no loss modification
no label smoothing
no focal / margin / calibration loss
no sampler / class weight
no CPU offload
no fake / proxy rows
KAN path 不使用 PyTorch loss.backward graph
official controller 不使用 dataset_name 分支
PureKANConv / PureKANFormer 继续 deferred
```

Functional update 仍然是 update rule，不是 loss：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{\text{AdamW}}
+
\Delta\theta_{\text{functional}}.
$$

任务目标保持标准 CE：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

数据集只能作为诊断切片，不能作为 official controller 条件：

```text
allowed:
  report MNIST / Fashion-MNIST / KMNIST slice metrics
  report failure distribution by dataset
  report leave-dataset-out generalization
  report signal-stratum composition by dataset
  report dataset-specific diagnostic plots

forbidden:
  if dataset == Fashion: use controller A
  if dataset == KMNIST: use primitive B
  if dataset == MNIST: abstain
  tune threshold separately per dataset
  promote a score because it rescues only one dataset
  use validation/test metric at commit time
  use posthoc replay outcome at commit time
```

---

# Part I. 当前结果的独立判断

## 1. v9.2.42 没有达到 strict functional success

v9.2.42 的 `R5-OracleHighLegalScoreLow` 说明：

```text
good functional events exist
but legal controller does not yet identify them robustly enough
```

它不是：

```text
strict PureKAN functional success
full functional success
external-ready success
```

原因很直接：frozen S7 虽然 AUC、precision、bad-event 都很好，但 coverage 只有 `0.029861`，略低于 $0.03$ gate；其他 legal scores 没有 official-pass；LDO/LSO/paired replay 因 no legal value score survivor 没有打开。

因此不能把 v9.2.42 写成成功。它只是把问题从：

```text
value score 是否能在 source rows 上成立？
```

推进到：

```text
legal score 是否能在 fresh multi-stratum rows 上稳定形成 enough coverage，并泛化到 leave-out？
```

## 2. v9.2.42 的真实进展

v9.2.42 比 v9.2.41 更可信，原因是它不再只使用单一 source stratum，而是完成了：

```text
fresh rows = 8640
fresh real events = 1440
signal strata = 8
attach candidates = 3
```

这证明 fresh multi-stratum expansion 本身是可行的。更重要的是，frozen S7 在 fresh rows 上没有崩：

$$
AUC=0.752942,
$$

$$
Precision=0.906977,
$$

$$
BadEvent=0.
$$

它只差 coverage：

$$
Coverage=0.029861<0.03.
$$

这个结果非常关键。它说明 v9.2.41 的 S7 不是纯 source-row artifact。它在 fresh rows 上仍有很强排序能力和 precision，但 official gate 没过，因此不能宣称成功。

## 3. 当前真正 blocker

当前 blocker 不是 base / attach / carrier。它也不是 “functional 没有 good events”。因为 oracle pass 说明 good events 存在。当前 blocker 是：

$$
\boxed{
\text{legal score 的 coverage / calibration / leave-out 泛化尚未闭合。}
}
$$

更具体地说，存在三个子问题：

```text
1. Coverage cliff:
   S7 precision 很高，但 coverage 低于 gate 约 0.000139。
   这说明 score 非常保守，或者 threshold / uncertainty calibration 不稳。

2. Legal-vs-oracle gap:
   S13 oracle pass，但 legal scores fail。
   这说明可选 good events 存在，但 legal controller 没有稳定找到它们。

3. Role-wise insufficiency:
   A3 carrier pass，但 rolewise value pass = 0。
   这说明简单 role-wise carrier 不自动带来 value observability，需要更明确的 role-wise control-gap score 或 calibration。
```

## 4. 为什么仍然感觉进展慢

现在慢的原因不是基础设施没做完，而是我们正在靠严格 gate 避免“局部信号包装成成功”。  
但执行方式必须改变。v9.2.42 已经做了并行 fresh rows，但 v9.2.43 还要进一步并行：

```text
1. 不只看 one threshold，而看 threshold-free PR curve 和 lower-bound precision curve。
2. 不只试 S7，而是做 legal controller family。
3. 不只测 value observability，而是同轮测 LDO/LSO 和 paired replay scout。
4. 不等下一轮再测 calibration，而是直接把 calibration split / heldout split 写进 runner。
```

本轮不能再做“再微调一个 threshold”的小补丁；要把 controller calibration 作为一个正式研究对象。

---

# Part II. v9.2.43 总体目标

v9.2.43 的总体目标是：

$$
\boxed{
\text{把 oracle-good but legal-score-low 的状态，推进到 legal calibrated controller + leave-out + paired replay closure。}
}
$$

目标分为六层。

## 1. Boundary and gap attribution success

必须复现 v9.2.42 boundary，并解释为什么 frozen S7 只有 coverage fail：

```text
S7 AUC high
S7 precision high
S7 bad-event low
S7 coverage slightly below gate
oracle pass
legal scores fail
```

必须判断 coverage fail 属于：

```text
G1-threshold_borderline:
  S7 的 precision/coverage curve 在 0.03 附近连续，稍微 calibration 就能满足。

G2-score_overconservative:
  S7 排序好但过度保守，top region precision 过高而 coverage 不足。

G3-family_coverage_gap:
  S7 只接受少数 family/strata，导致 accepted_signal_strata 不足。

G4-uncertainty_miscalibration:
  score 均值有效，但 uncertainty / LCB 太保守。

G5-legal_feature_missing:
  oracle-good events 依赖 legal score 未使用的 role/control-gap features。

G6-true_controller_gap:
  legal features 本身不足，oracle-good events 无法从 commit-time signal 识别。
```

## 2. Legal controller calibration success

至少一个 legal controller 必须在 fresh heldout rows 上满足：

$$
AUC(Y_{\text{beat}})\geq0.70
$$

or:

$$
Corr(S,V_{\text{grounded}})\geq0.35.
$$

Accept / abstain gate：

$$
Precision\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05.
$$

Coverage 不能通过直接放宽 gate 获得。必须通过预注册 calibration rule 产生：

```text
threshold selected on calibration rows only
tested on heldout rows
no dataset name
no validation/test metric
no posthoc outcome at commit
```

## 3. Multi-family coverage success

v9.2.42 已经测到 8 strata，但 frozen S7 可能只接受很少的 event families。v9.2.43 要求：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
each accepted stratum coverage >= 0.005
no single family contributes > 60% accepted events
```

这避免 controller 只在一个狭窄 source mode 上成功。

## 4. Leave-out success

Leave-dataset-out：

$$
Acc_{\text{heldout,Real}}\geq Acc_{\text{heldout,AdamW}}-0.005.
$$

At least two held-out datasets satisfy：

$$
BeatRate_{\text{heldout,Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{heldout,Real vs bestLR}}\geq0.50.
$$

Leave-stratum-out：

At least $70\%$ held-out strata task-safe and：

$$
CEp99_{\text{heldout,Real}}\leq CEp99_{\text{AdamW}}+\epsilon.
$$

and：

$$
BeatRate_{\text{heldout-stratum,Real vs AdamWParallel}}\geq0.50.
$$

## 5. Official paired replay success

Only after legal controller + leave-out pass, official paired replay opens.

Paired replay pass：

$$
BeatRate_{\text{macro,Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro,Real vs bestLR}}\geq0.60.
$$

Task safety：

$$
Acc_{\text{Real,slice}}\geq Acc_{\text{AdamW,slice}}-0.005.
$$

System：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

## 6. Short-run scout success

Only after paired replay pass：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005.
$$

At least one mechanism improvement：

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamW}},
$$

or:

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamW}},
$$

or:

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

---

# Part III. 核心假设

## H1：Frozen S7 的失败主要是 coverage calibration，而不是排序能力失败

证据：

$$
AUC=0.752942,
$$

$$
Precision=0.906977,
$$

$$
BadEvent=0,
$$

but:

$$
Coverage=0.029861.
$$

H1 成立标准：

在 threshold-free curve 上，存在 legal calibration threshold $\tau$，使 heldout rows 满足：

$$
Precision(\tau)\geq0.75,
$$

$$
Coverage(\tau)\in[0.03,0.15],
$$

$$
BadEventRate(\tau)\leq0.05.
$$

如果 calibration rows 上可以但 heldout rows 上不可以，则 H1 失败，说明 calibration overfit。

## H2：Oracle-good events 可以由 legal features 近似恢复

v9.2.42 的 route 是 OracleHighLegalScoreLow。H2 认为 legal features 仍有足够信息，只是当前 S7/S9/S10/S11/S12 没有正确校准。

H2 成立标准：

至少一个 legal controller 在 heldout rows 上达到：

$$
AUC\geq0.70
$$

or:

$$
Precision\geq0.75 \land Coverage\geq0.03.
$$

If oracle pass but all legal calibrated controllers fail：

```text
controller_feature_gap confirmed
```

## H3：two-stage controller 比 single-score threshold 更可能过 gate

Frozen S7 top region precision 很高但 coverage 略低。Two-stage rule 可以保留 high precision core，同时用 uncertainty / family reliability 扩展 coverage。

Two-stage form：

$$
Accept(e)
=
Core(e)
\lor
[
Border(e)
\land ReliableFamily(e)
\land LowRisk(e)
].
$$

Where:

$$
Core(e)=\mathbb{1}[S_7(e)\geq\tau_{core}],
$$

$$
Border(e)=\mathbb{1}[\tau_{border}\leq S_7(e)<\tau_{core}].
$$

H3 成立标准：

Two-stage controller heldout precision/coverage/bad-event gate pass and LDO/LSO pass.

## H4：role-wise value failed because role score was not control-gap calibrated

A3 carrier can move, but rolewise value pass = 0. H4 says role-wise needs value calibration:

$$
S_{\text{role-gap}}
=
\alpha_s S_{\text{stack-control-gap}}
+
\alpha_h S_{\text{head-control-gap}}
-
R_{\text{risk}}.
$$

H4 成立标准：

Role-gap score improves AUC or paired beat rate over non-role score by at least $0.05$.

## H5：如果 legal calibration still fails but oracle remains high, next step is controller representation, not carrier reset

If:

```text
oracle pass = 1
all legal calibrated controllers fail
carrier pass = 1
```

then blocker is not carrier actuatability but legal feature/controller representation.

## H6：if oracle drops under expanded rows, carrier mechanism must reset

If fresh v2 oracle no longer has sufficient precision/coverage, then current carrier is not producing robust good events and should be reset toward role-wise FT7 mechanism.

---

# Part IV. 并行执行设计

v9.2.43 runner 必须并行执行：

```text
Lane A:
  Boundary reproduction and S7 coverage autopsy.

Lane B:
  Threshold-free PR / coverage curve for frozen S7.

Lane C:
  Legal controller calibration family:
    C0-FrozenS7
    C1-CalibratedS7
    C2-TwoStageS7FamilyReliability
    C3-ControlGapLCB
    C4-RoleGapCalibrated
    C5-FamilyReliabilityGap
    C6-HybridMonotoneCalibrated

Lane D:
  Fresh multi-stratum expansion v2.

Lane E:
  LDO/LSO.

Lane F:
  Paired replay scout and official replay.

Lane G:
  Short-run scout if paired replay passes.

Lane H:
  Fallback decision:
    controller representation vs carrier reset.
```

Gate discipline：

```text
diagnostic rows can be computed early
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier pass
  legal value controller pass
  LDO/LSO pass
```

---

# Part V. Candidate and controller design

## 1. Base and attach

Official base：

```text
R2-LQ-fanin-output-scale-confirmed
```

Attach candidates：

```text
A1-LateAttachZeroLinearTail
A2-LateAttachControlGapChannel
A3-LateAttachRoleWiseFT7EdgeCarrier
A5-HybridRoleWiseControlGap
```

## 2. Legal controllers

### C0：FrozenS7Reference

Exactly v9.2.41/v9.2.42 S7.  
No calibration. Used as reference.

### C1：CalibratedS7Threshold

Threshold selected on calibration rows only.

Calibration objective：

$$
\max_{\tau} Coverage_{\text{cal}}(\tau)
$$

subject to：

$$
Precision_{\text{cal}}(\tau)\geq0.80,
$$

$$
BadEventRate_{\text{cal}}(\tau)\leq0.03,
$$

$$
Coverage_{\text{cal}}(\tau)\in[0.03,0.15].
$$

Evaluation uses heldout rows.

### C2：TwoStageS7FamilyReliability

Accept if high-confidence S7 core or reliable borderline family:

$$
Accept(e)=
\mathbb{1}[S_7(e)\geq\tau_c]
\lor
\mathbb{1}[
S_7(e)\geq\tau_b
\land Reliability(family(e))\geq r_0
\land Risk(e)\leq\rho
].
$$

Family does not include dataset name:

$$
family(e)
=
(stratum(e),horizon(e),attach(e),risk\_bucket(e),role\_bucket(e)).
$$

### C3：ControlGapLCBCalibrated

$$
S_{gap}
=
LCB(Gain_{\text{Real}})
-
UCB(\max(Gain_{\text{AdamWParallel}},Gain_{\text{bestLR}})).
$$

Accept if：

$$
S_{gap}>\tau.
$$

### C4：RoleGapCalibrated

$$
S_{role}
=
\alpha_s S_{\text{stack-gap}}
+
\alpha_h S_{\text{head-gap}}
-
\lambda R_{\text{risk}}.
$$

### C5：FamilyReliabilityGap

$$
S_{family}
=
\mathbb{E}[V\mid family]
-
\kappa \operatorname{Std}(V\mid family).
$$

### C6：HybridMonotoneCalibrated

A monotone rule using only：

```text
S7
control_gap_score
tail_value_score
role_gap_score
uncertainty
branch_ratio
effective_derivative
family_reliability
```

No learned black-box with hidden dataset features.  
No dataset name.

### C7：Oracle

Posthoc diagnostic only.  
Never official.

---

# Part VI. 实验阶段

## P0：v9.2.42 boundary reproduction

### 目标

复现 v9.2.42 boundary。

### 必须记录

```text
route
source_route_v9241
fresh_row_count
fresh_real_event_count
signal_strata_count
attach_candidate_count
frozen_s7_auc
frozen_s7_corr
frozen_s7_precision
frozen_s7_coverage
frozen_s7_bad_event
frozen_s7_pass
legal_score_survivor_count
oracle_pass
oracle_precision
oracle_coverage
rolewise_carrier_pass
rolewise_value_pass
paired_replay_opened
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R5-OracleHighLegalScoreLow
frozen S7 near-pass but coverage fail
oracle pass = 1
no legal official survivor
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_s7_nearpass_failure.svg
p0_oracle_legal_gap.svg
```

---

## P1：S7 coverage cliff and PR-curve autopsy

### 目标

解释 S7 为什么差一点 coverage。不能只说 “差 0.000139”。

### 必须记录

```text
score_id
threshold
precision
coverage
bad_event_rate
recall
accepted_count
accepted_good_count
accepted_bad_count
accepted_strata_count
accepted_family_count
max_family_share
AUC
corr
calibration_split
heldout_split
```

### 判断标准

P1 pass：

```text
S7 threshold-free PR curve computed
coverage cliff mode assigned
family concentration measured
no dataset name used
```

Borderline if：

$$
\exists \tau:
Precision(\tau)\geq0.75,\quad
Coverage(\tau)\in[0.03,0.15],\quad
BadEvent(\tau)\leq0.05.
$$

But this is diagnostic unless $\tau$ is selected on calibration and evaluated on heldout.

### 可视化

```text
p1_s7_precision_coverage_curve.svg
p1_s7_bad_event_curve.svg
p1_s7_family_concentration.svg
p1_s7_threshold_margin.svg
```

---

## P2：Fresh multi-stratum expansion v2

### 目标

增加 heldout power，避免 controller calibration 在 8640 rows 上过拟合。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7
horizons = 20,80,240,640
signal_strata = S1-S8
attach_candidates = A1,A2,A3,A5
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random
```

### 必须记录

```text
row_id
dataset
seed
horizon
signal_stratum
attach_candidate
branch
event_family
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
real_gain
adamwparallel_gain
bestlr_gain
control_gap
task_safe
bad_event
r_z_tail
r_perp_tail
cos_real_adamw
cos_real_bestlr
branch_ratio
effective_derivative
uncertainty
posthoc_value
fake_proxy_flag
```

### 判断标准

Fresh v2 pass：

```text
fresh_real_event_count >= 2000
measured_signal_strata_count >= 6
attach_candidates_measured >= 4
fake/proxy/offload = 0
```

Carrier remains active：

$$
r_{z,\text{tail}}\geq0.10,
$$

$$
r_{\perp,\text{tail}}\geq0.10.
$$

Task safety：

$$
BadEventRate\leq0.05.
$$

### 可视化

```text
p2_event_coverage_by_stratum.svg
p2_event_coverage_by_family.svg
p2_carrier_movement_by_attach.svg
p2_control_gap_distribution.svg
```

---

## P3：Legal controller calibration matrix

### 目标

并行测试 C0-C7，形成 legal controller survivor 或明确 controller feature gap。

### Split design

使用 event rows 划分，不按 dataset 特化：

```text
Calibration split:
  used to select thresholds / monotone coefficients

Heldout split:
  used for official score pass

Leave-dataset-out:
  separately evaluated in P4

Leave-stratum-out:
  separately evaluated in P4
```

### 必须记录

```text
controller_id
attach_candidate
calibration_split_id
heldout_split_id
thresholds
coefficients_if_any
precision_cal
coverage_cal
bad_event_cal
precision_heldout
coverage_heldout
bad_event_heldout
AUC_heldout
corr_heldout
accepted_strata_count
accepted_family_count
max_family_share
dataset_name_used
posthoc_used_at_commit
validation_used
test_used
official_eligible
gate_missing_reason
```

### 判断标准

Official controller pass：

$$
AUC_{\text{heldout}}\geq0.70
$$

or:

$$
Corr_{\text{heldout}}\geq0.35.
$$

and：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{heldout}}\leq0.05.
$$

Multi-family coverage：

```text
accepted_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

Legality：

```text
dataset_name_used = 0
posthoc_used_at_commit = 0
validation_used = 0
test_used = 0
```

Oracle interpretation：

```text
If C7 oracle pass and C0-C6 fail:
  controller representation blocker.

If C7 oracle fail:
  carrier mechanism blocker.
```

### 可视化

```text
p3_controller_matrix_auc_corr.svg
p3_controller_precision_coverage.svg
p3_controller_family_coverage.svg
p3_oracle_vs_legal_gap.svg
p3_controller_legality_matrix.svg
```

---

## P4：Leave-dataset-out and leave-stratum-out

### 目标

验证 controller 不依赖 dataset 或单一 stratum。

### 设置

Leave-dataset-out：

```text
calibrate on MNIST + Fashion, evaluate KMNIST
calibrate on MNIST + KMNIST, evaluate Fashion
calibrate on Fashion + KMNIST, evaluate MNIST
```

Leave-stratum-out：

```text
calibrate on all but one signal stratum
evaluate held-out stratum
```

### 必须记录

```text
split_type
heldout
controller_id
attach_candidate
threshold
precision
coverage
bad_event_rate
task_safe
CEp99_delta
margin_delta
ECE_delta
NLL_delta
curvature_delta
beats_adamwparallel
beats_bestlr
dataset_name_used
shuffle_control_pass
```

### 判断标准

LDO pass：

$$
Acc_{\text{heldout,Real}}\geq Acc_{\text{heldout,AdamW}}-0.005.
$$

At least two held-out datasets：

$$
BeatRate_{\text{Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{Real vs bestLR}}\geq0.50.
$$

LSO pass：

At least $70\%$ held-out strata task-safe and：

$$
CEp99_{\text{heldout,Real}}\leq CEp99_{\text{AdamW}}+\epsilon.
$$

and：

$$
BeatRate_{\text{heldout-stratum,Real vs AdamWParallel}}\geq0.50.
$$

### 可视化

```text
p4_leave_dataset_out_matrix.svg
p4_leave_stratum_out_matrix.svg
p4_hidden_dataset_tuning_audit.svg
p4_leaveout_failure_modes.svg
```

---

## P5：Official paired replay

### 目标

验证 RealFunctional 是否在 strong controls 下有局部因果优势。

### 设置

```text
base = R2 repaired base checkpoint
attach_candidate = best P4 survivor
controller_id = best legal controller survivor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, InvertedRole, ValueScoreShuffled, FunctionalChannelShuffled, TailMaskShuffled, RoleScoreShuffled
```

### 必须记录

```text
attach_candidate
controller_id
dataset
seed
horizon
signal_stratum
event_family
branch
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
real_beats_adamwparallel
real_beats_bestlr
real_beats_random
real_beats_noop
task_safe
event_count
coverage
bad_event_rate
step_q90
memory_ratio
base_checkpoint_hash
```

### 判断标准

Paired replay pass：

$$
BeatRate_{\text{macro,Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro,Real vs bestLR}}\geq0.60.
$$

Task safety：

$$
Acc_{\text{each slice,Real}}\geq Acc_{\text{AdamW}}-0.005.
$$

Shuffle controls fail：

```text
ValueScoreShuffled = fail
FunctionalChannelShuffled = fail
TailMaskShuffled = fail
RoleScoreShuffled = fail
DatasetRouteShuffled = fail
EventRouteShuffled = fail
InvertedRoleMask = fail
```

System：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p5_official_paired_replay_pareto.svg
p5_macro_beat_rate.svg
p5_signal_stratum_win_matrix.svg
p5_shuffle_control_matrix.svg
p5_system_gate_distribution.svg
```

---

## P6：Short-run scout

### 目标

如果 P5 pass，验证局部 paired replay 优势能否在连续训练中保持。

### 设置

```text
steps = 50,240,640
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, ValueScoreShuffled
```

### 必须记录

```text
candidate
dataset
seed
steps
train_loss
holdout_loss
val_acc_proxy
CEp99
margin_p10
ECE_proxy
NLL_proxy
curvature
local_lipschitz
basis_usage_entropy
functional_channel_usage_entropy
event_count
coverage
bad_event_rate
step_ratio_q90
memory_ratio
base_checkpoint_hash
```

### 判断标准

Task safety：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005.
$$

Control superiority：

$$
MetricGain_{\text{functional}}>MetricGain_{\text{AdamWParallel}},
$$

$$
MetricGain_{\text{functional}}>MetricGain_{\text{bestLR}}.
$$

Mechanism pass, at least one：

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamW}},
$$

or:

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamW}},
$$

or:

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

### 可视化

```text
p6_short_run_task_mechanism_pareto.svg
p6_short_run_controls.svg
p6_event_timeline.svg
p6_ce_tail_margin_panel.svg
```

---

## P7：Full 10-seed validation

### 目标

如果 short-run pass，验证 full functional route。

### 设置

```text
epochs = 20
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
baseline = MLP-match
diagnostic baseline = QuadraticFeatureMLP
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole
```

### 必须记录

```text
candidate
dataset
seed
val_acc
test_acc
delta_vs_mlp
delta_vs_adamw
delta_vs_adamwparallel
delta_vs_bestlr
delta_vs_quadratic_feature_mlp
near_pass
full_pass
CEp99
margin_p10
wrong_confidence_p95
ECE
NLL
curvature_ratio
local_lipschitz_ratio
basis_usage_entropy
functional_channel_usage_entropy
event_count
coverage
bad_event_rate
step_ratio_q90
memory_ratio
base_checkpoint_hash
```

### 判断标准

Task safety：

$$
\Delta Acc_{\text{functional-vs-AdamW}}\geq-0.005.
$$

Macro improvement：

$$
\Delta Acc_{\text{macro,functional}}
-
\Delta Acc_{\text{macro,AdamW}}
\geq0.003.
$$

Hard-stratum repair：

$$
\Delta Acc_{\text{hard-stratum,functional}}
-
\Delta Acc_{\text{hard-stratum,AdamW}}
\geq0.005.
$$

Control superiority：

$$
MetricGain_{\text{functional}}>MetricGain_{\text{AdamWParallel}},
$$

$$
MetricGain_{\text{functional}}>MetricGain_{\text{bestLR}}.
$$

---

## P8：Robustness / strong baseline / external-ready

### 目标

确认 functional advantage 不是 clean MNIST-family artifact。

### 设置

```text
label_noise = 0.05,0.10,0.20
input_noise = 0.05,0.10
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
strong baselines = MLP-match, hidden-bracket MLP, QuadraticFeatureMLP, repaired LQ AdamW-only
```

### 判断标准

Robustness pass：

$$
AccDrop_{\text{functional}}\leq AccDrop_{\text{AdamW}}.
$$

Noise-tail pass：

$$
CEp99_{\text{functional}}\leq CEp99_{\text{AdamW}}.
$$

Strong baseline pass：

$$
Acc_{\text{functional}}\geq Acc_{\text{QuadraticFeatureMLP}}-0.005
$$

or:

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{QuadraticFeatureMLP}}.
$$

---

# Part VII. Required artifacts

```text
run_manifest.json
contract_audit_v9243.csv
p0_v9242_boundary_reproduction.csv
p1_s7_coverage_cliff_autopsy.csv
p2_fresh_multistratum_expansion_v2.csv
p3_legal_controller_calibration_matrix.csv
p4_leave_dataset_and_stratum_out.csv
p5_official_paired_replay.csv
p6_short_run_functional_validation.csv
p7_full_10seed_functional_validation.csv
p8_robustness_external_ready.csv
s7_pr_curve_trace_v9243.csv
controller_calibration_trace_v9243.csv
fresh_event_trace_v9243.csv
oracle_legal_gap_trace_v9243.csv
leaveout_trace_v9243.csv
paired_replay_branch_trace_v9243.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9242_boundary_unstable
F3_dataset_tuning_detected
F4_s7_coverage_cliff_unattributed
F5_fresh_rows_insufficient
F6_calibrated_controller_overfit
F7_all_legal_controllers_fail
F8_oracle_low_carrier_mechanism_fail
F9_oracle_high_legal_feature_gap
F10_leave_dataset_out_fail
F11_leave_stratum_out_fail
F12_paired_replay_control_equivalent
F13_shuffle_control_pass
F14_functional_lr_equivalent
F15_short_run_task_drop
F16_full_run_no_macro_hard_stratum_gain
F17_strong_baseline_explains_gain
F18_robustness_fail
F19_external_not_ready
F20_fake_or_proxy_violation
F21_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-S7CoverageBorderlineCalibrated:
  frozen S7 was borderline; calibrated S7 passes heldout without dataset tuning.

R2-TwoStageControllerPass:
  two-stage S7 + family reliability controller passes legal value gate.

R3-ControlGapLCBControllerPass:
  control-gap LCB controller passes legal value gate.

R4-RoleGapControllerPass:
  role-wise calibrated control-gap controller passes legal value gate.

R5-FamilyReliabilityControllerPass:
  family reliability controller passes legal value gate.

R6-OracleHighLegalFeatureGap:
  oracle passes but all legal controllers fail; controller feature representation is blocker.

R7-OracleLowCarrierMechanismReset:
  oracle fails; current carrier does not produce enough good events.

R8-LeaveDatasetOutPass:
  legal controller generalizes across held-out datasets.

R9-LeaveStratumOutPass:
  legal controller generalizes across held-out signal strata.

R10-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R11-ControlEquivalentAgain:
  value controller passes local gate but paired replay remains control-equivalent.

R12-StrictPureKANFunctionalShortRunPass:
  short-run task-safe mechanism gain.

R13-StrictPureKANFunctionalFullPass:
  full 10-seed macro / hard-stratum / geometry gain.

R14-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` 必须记录：

```text
route
v9242_boundary_pass
dataset_tuning_detected
s7_coverage_cliff_mode
fresh_real_event_count
measured_signal_strata_count
best_controller_id
controller_value_pass
controller_auc
controller_corr
accepted_precision
accepted_coverage
accepted_bad_event_rate
accepted_strata_count
accepted_family_count
oracle_upper_bound_pass
oracle_legal_gap
leave_dataset_out_pass
leave_stratum_out_pass
paired_replay_pass
short_run_pass
full_run_pass
external_ready
primary_blocker
next_required_implementation
success_v9243_strict_purekan_functional
success_v9243_full_functional
success_v9243_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 S7 coverage cliff autopsy
  P2 fresh multi-stratum expansion v2
  P3 legal controller calibration matrix

Batch 2:
  P4 leave-dataset-out / leave-stratum-out
  P5 paired replay scout for controller survivors

Batch 3:
  official P5 paired replay
  P6 short-run if paired replay passes

Batch 4:
  P7 full 10-seed
  P8 robustness / strong baseline
```

Gate rule：

```text
P3/P5 diagnostic rows may be measured before all gates finish.
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier pass
  legal value controller pass
  LDO/LSO pass
```

---

# Part X. 停止条件

## Minimum diagnostic success

```text
v9.2.42 boundary reproduced
S7 coverage cliff attributed
fresh multi-stratum v2 measured
controller calibration matrix completed
oracle/legal gap measured
no fake/proxy/offload/loss/teacher violation
```

## Legal controller success

```text
Minimum diagnostic success
+
at least one legal controller passes heldout observability
+
precision / coverage / bad-event gate pass
+
accepted strata/family coverage pass
```

## Local functional success

```text
Legal controller success
+
LDO / LSO pass
+
official paired replay beats AdamWParallel / bestLR
```

## Full functional success

```text
Local functional success
+
short/full run task-safe mechanism gain
+
macro or hard-stratum improvement
```

## Failure stop

```text
1. v9.2.42 boundary cannot be reproduced；
2. S7 coverage cliff cannot be attributed；
3. fresh expansion cannot produce enough rows / strata；
4. all calibrated legal controllers fail heldout value gate；
5. oracle high but legal features cannot recover good events；
6. oracle low, meaning carrier mechanism lacks good events；
7. leave-dataset-out fails；
8. leave-stratum-out fails；
9. paired replay remains control-equivalent；
10. shuffle controls pass, indicating overfit；
11. short-run task drops；
12. full run gives no macro / hard-stratum / geometry gain；
13. functional breaks system gate；
14. gains are explained by QuadraticFeatureMLP；
15. any teacher/loss/fake/proxy/offload violation occurs。
```

---

# Part XI. 最终解释规则

## Case A：Calibrated S7 passes

可以声明：

```text
v9.2.42 was a coverage calibration failure; S7 ordering was valid.
```

但不能声明 strict success unless LDO/LSO and paired replay pass.

## Case B：Two-stage controller passes

可以声明：

```text
High-precision S7 core plus reliable borderline family expands coverage without dataset tuning.
```

但仍需 paired replay.

## Case C：Oracle high but all legal controllers fail

必须声明：

```text
Good functional events exist, but current legal feature set cannot identify them.
```

下一步应设计 richer train-stream value features，不应修 base or attach.

## Case D：Oracle low

必须声明：

```text
Current carrier does not generate enough robust control-resistant good events.
```

下一步 reset carrier / role-wise mechanism.

## Case E：LDO/LSO fail

必须声明：

```text
Controller is not dataset-agnostic or stratum-agnostic enough.
```

不能用 dataset-specific tuning 写成功。

## Case F：Paired replay passes

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full validation.

---

# Part XII. 最终建议

v9.2.43 的一句话策略是：

$$
\boxed{
\text{不要再扩大基础设施；把 oracle-good / legal-low 问题正式变成 controller calibration 与 leave-out paired replay 闭环。}
}
$$

当前最关键的问题不是：

```text
base 是否稳定；
attach 是否污染；
carrier 是否 silent；
Fashion 怎么调；
KMNIST 怎么调；
MNIST 是否 abstain；
是否再换一个 basis。
```

而是：

```text
1. Frozen S7 为什么 coverage 卡在 0.029861？
2. 这个 coverage miss 是否能通过预注册 calibration 在 heldout rows 上解决？
3. 两阶段 controller 是否能保持 precision，同时把 coverage 稳定推过 0.03？
4. legal controller 能否跨 dataset / stratum 泛化？
5. paired replay 是否真的击败 AdamWParallel / bestLR？
6. 如果 oracle 高而 legal 低，缺的是哪些 train-stream value features？
7. 如果 oracle 低，当前 carrier 是否必须 reset？
```

v9.2.43 的结果将给出清晰分叉：

```text
if calibrated legal controller + LDO/LSO + paired replay pass:
  strict PureKAN functional gets first local causal evidence.

if calibrated controller passes local but LDO/LSO fail:
  score is not general enough; no dataset tuning allowed.

if oracle high but legal controllers fail:
  controller feature representation is blocker.

if oracle low:
  carrier mechanism is blocker.
```
