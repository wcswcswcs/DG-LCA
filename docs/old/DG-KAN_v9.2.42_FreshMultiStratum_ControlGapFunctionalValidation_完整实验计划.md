# DG-KAN v9.2.42 Fresh Multi-Stratum Control-Gap Functional Validation 完整实验计划

> 本计划基于 v9.2.41 `Control-Gap Value Alignment 与 RoleWise Late-Attach` 的真实复盘制定。  
> v9.2.41 的 terminal route 是：
>
> ```text
> route = R3-ControlGapScorePass
> base_candidate = LQ-t2-h256
> success_v9241_strict_purekan_functional = False
> success_v9241_full_functional = False
> success_v9241_external_ready = False
> ```
>
> v9.2.41 的最重要事实是：**value score 第一次过了局部 observability gate，但还没有通过 leave-dataset-out / leave-stratum-out，因此不能算 strict functional success。**
>
> 关键数据：
>
> ```text
> P1 value failure autopsy pass = 1
> primary mode = F5-score_sign_mismatch
> primary fraction = 0.95
>
> best legal score = S7-HybridMonotoneLegal
> value observability pass = 1
> AUC = 0.736068
> corr = 0.343287
> accepted precision = 0.777778
> coverage = 0.04
> bad-event = 0.0
>
> role-wise late attach audit pass = 1
> oracle upper bound pass = 1
> oracle precision = 1.0
> oracle coverage = 0.12
> shuffle controls = fail
>
> leave-dataset-out pass = 0
> leave-stratum-out = not evaluable
> reason: source rows have only one signal stratum
> current blocker = leave_dataset_or_stratum_out_failed_after_value_score_pass
> ```
>
> v9.2.42 的核心不是继续修 base，也不是再调 Fashion / KMNIST / MNIST；当前必须把 v9.2.41 的局部 score success 变成 **fresh、multi-stratum、leave-out 可验证的 functional causality**。  
>
> 本轮必须加快，不再一轮只测一个小 gate。v9.2.42 采用并行 runner：
>
> ```text
> Lane A:
>   fresh multi-stratum event expansion
>
> Lane B:
>   frozen S7 confirmation
>
> Lane C:
>   role-wise / control-gap score variants
>
> Lane D:
>   leave-dataset-out and leave-stratum-out
>
> Lane E:
>   official paired replay
>
> Lane F:
>   short-run scout
>
> Lane G:
>   mechanism reset fallback if oracle drops
> ```
>
> 但并行不等于越 gate。所有 diagnostic rows 可以提前测，official success 仍必须满足预注册 gate。

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

forbidden:
  if dataset == Fashion: use controller A
  if dataset == KMNIST: use primitive B
  if dataset == MNIST: abstain
  tune threshold separately per dataset
  choose score because it rescues only one dataset
```

---

# Part I. 对 v9.2.41 的独立判断

## 1. v9.2.41 没有达到 strict functional success

v9.2.41 的 route 是 `R3-ControlGapScorePass`，这个 route 比 v9.2.40 的 `R11-BasePreservedButValueUnobservable` 明显前进，但它不是终点。它只说明：

```text
在 v9.2.40 source-measured carrier/value rows 上，
找到了一个 legal value score，可以局部预测 Real-vs-control value。
```

它没有证明：

```text
fresh event rows 上仍成立；
多个 signal strata 上仍成立；
held-out dataset 上仍成立；
held-out stratum 上仍成立；
paired replay 中 RealFunctional 能打过 AdamWParallel / bestLR；
short-run / full-run 能成立。
```

因此正确判断是：

$$
\boxed{
\text{v9.2.41 是局部 value score 突破，不是 strict functional success。}
}
$$

## 2. v9.2.41 的真实推进

v9.2.40 之前，我们卡在 base / attach / carrier。v9.2.40 第一次同时通过：

```text
repaired base robust pass
snapshot attach pass
inactive equivalence pass
no-event replay pass
functional carrier pass
```

v9.2.41 则进一步解决了 v9.2.40 的 score 方向错配问题。P1 显示 primary failure 是 `F5-score_sign_mismatch`，说明 v9.2.40 的 negative corr 不是纯噪声，而是 score construction 的方向性错误。P2 的 `S7-HybridMonotoneLegal` 过了 observability：

$$
AUC=0.736068,
$$

$$
Precision=0.777778,
$$

$$
Coverage=0.04,
$$

$$
BadEvent=0.
$$

这非常关键。它说明：

$$
\boxed{
\text{active functional carrier 中确实存在可由 legal signal score 捕捉的 good events。}
}
$$

同时 P4 oracle upper bound pass，oracle precision 为 $1.0$、coverage 为 $0.12$，说明 good events 不是不存在，而是过去 score 没找到或没有泛化验证。

## 3. 当前真正 blocker

当前 blocker 不是：

```text
base 不稳；
attach 污染；
carrier silent；
functional update 必然没用；
control-gap score 完全无效。
```

当前 blocker 是：

$$
\boxed{
\text{S7 的 value score 只在 source rows / 单 stratum 上成立，还没有证明 dataset-agnostic 和 stratum-agnostic。}
}
$$

尤其需要重视两点。

第一，P5 leave-dataset-out pass = 0。说明 S7 可能捕捉到了 source row distribution，而不是泛化的 functional value mechanism。

第二，leave-stratum-out 不可评估，因为 v9.2.40 source rows 只有一个 signal stratum。这意味着 v9.2.41 不能证明 score 在不同 event families 上有效。

所以现在不能继续在同一批 source rows 上做更多后验 score polishing。必须 fresh rerun 并扩展 signal strata。

## 4. 为什么仍然感觉慢

慢的根源已经不是 base infrastructure，而是我们对 functional causality 的每个前置条件都要求真实 gate。这个要求是必要的，但执行方式必须改变。v9.2.42 必须做到：

```text
1. fresh rows
2. multi-stratum
3. frozen score confirmation
4. role-wise candidate
5. leave-out
6. paired replay
```

同一轮并行完成。不能再按：

```text
v9.2.42 只扩 events
v9.2.43 再测 score
v9.2.44 再 LDO
v9.2.45 再 paired replay
```

这样会太慢。

## 5. 是否在正确道路上

是。因为现在终于到了 functional update 的核心问题：**能不能在不使用 dataset name、不改 loss、不加 teacher 的情况下，用训练流内可计算的 signal score 选择能击败 optimizer controls 的 functional event。**

这与终极目标一致。终极目标不是 MNIST-family 打榜，而是：

```text
Clean FullEdge PureKAN
Graph-Free Manual Training
Kernel-Native Efficiency
Task-Safe Functional Update
External Fair Advantage
Scalable Conv / Transformer Extension
```

当前 v9.2.41 的位置是：

```text
base / attach / carrier:
  已打开

value score:
  局部打开

generalization / paired replay:
  未打开
```

因此下一步不是回头修 base，而是用 fresh multi-stratum 实验证明或证伪 value mechanism。

---

# Part II. v9.2.42 总体目标

v9.2.42 的总体目标是：

$$
\boxed{
\text{把 v9.2.41 的局部 S7 value score success，升级为 fresh multi-stratum leave-out paired replay success。}
}
$$

目标分为六层。

## 1. Fresh row success

v9.2.42 必须新增 fresh rows，而不是只复用 v9.2.40 source rows。  
最小 fresh measurement：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
signal_strata >= 4
attach_candidates >= 3
score_candidates >= 5
```

Fresh row gate：

```text
fresh_row_count >= 3 * 5 * 4 * 4 * 3
no fake/proxy/offload = 0
```

## 2. Multi-stratum success

至少要有四类 signal strata：

```text
S1-CEHardTail
S2-MarginTail
S3-ControlGapPositive
S4-RoleWiseCurvature
S5-UncertaintyLCB
S6-FamilyValueReliable
S7-HighDerivativeBranch
S8-OrthogonalTailNonAdamW
```

成功标准：

```text
measured_signal_strata_count >= 4
accepted_signal_strata_count >= 2
each accepted stratum coverage >= 0.01
```

## 3. Frozen S7 confirmation success

v9.2.41 的 S7 必须冻结，不能再根据 new rows 调公式。  
Frozen S7 必须在 fresh rows 上满足：

$$
AUC(Y_{\text{beat}})\geq0.70
$$

or:

$$
Corr(S_7,V_{\text{grounded}})\geq0.35.
$$

Accept/abstain gate：

$$
Precision\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05.
$$

如果 frozen S7 fail，但 a pre-registered new score passes，则 route 只能写：

```text
R2-NewScorePassFrozenS7Fail
```

不能写成 S7 confirmed。

## 4. Leave-out success

必须通过 leave-dataset-out 与 leave-stratum-out。

Leave-dataset-out：

$$
Acc_{\text{heldout,Real}}\geq Acc_{\text{heldout,AdamW}}-0.005.
$$

至少两个 held-out datasets 满足：

$$
BeatRate_{\text{heldout,Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{heldout,Real vs bestLR}}\geq0.50.
$$

Leave-stratum-out：

至少 $70\%$ held-out strata task-safe，且：

$$
CEp99_{\text{heldout,Real}}\leq CEp99_{\text{AdamW}}+\epsilon.
$$

并且：

$$
BeatRate_{\text{heldout-stratum,Real vs AdamWParallel}}\geq0.50.
$$

## 5. Official paired replay success

Paired replay 必须 beating strong controls：

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

Anti-overfit controls must fail：

```text
ValueScoreShuffled = fail
FunctionalChannelShuffled = fail
TailMaskShuffled = fail
RoleScoreShuffled = fail
DatasetRouteShuffled = fail
EventRouteShuffled = fail
InvertedRoleMask = fail
```

## 6. Short-run scout success

Only after paired replay pass，short-run scout 打开：

```text
steps = 50,240,640
seeds = 0,1,2
datasets = MNIST,Fashion-MNIST,KMNIST
```

Short-run success：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005,
$$

and at least one mechanism gain：

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

## H1：v9.2.41 的 S7 不是偶然 source-row overfit

H1 认为 S7 的 AUC/precision 成立，是因为它捕捉了 monotone control-gap / role / tail-risk signal，而不是 source-row artifact。

H1 成立标准：

fresh rows 上：

$$
AUC(S7)\geq0.70,
$$

$$
Precision(S7)\geq0.75,
$$

$$
Coverage(S7)\in[0.03,0.15].
$$

H1 失败标准：

```text
AUC(S7) < 0.60
or precision < 0.60
or only one dataset / one stratum works
```

## H2：v9.2.41 LDO fail 是因为 source rows stratum support 太窄，不是 score mechanism 必然不泛化

v9.2.41 LSO 不可评估，因为只有一个 signal stratum。H2 认为多 stratum fresh rows 会改善 LDO/LSO。

H2 成立标准：

```text
measured_signal_strata_count >= 4
LDO pass = 1
LSO pass = 1
```

## H3：role-wise late attach 可以增加 held-out robustness

v8/v9.2.20 的 functional core 曾主要来自 FT7 RoleWiseFunctional。v9.2.41 role-wise audit pass 但还没有 fresh multi-stratum official validation。H3 认为 A3/A5 role-wise late attach 能提高 LDO/LSO 和 paired replay。

H3 成立标准：

Role-wise candidate 相比 non-role candidate：

$$
AUC_{\text{role}}-AUC_{\text{non-role}}\geq0.05
$$

or:

$$
BeatRate_{\text{role}}-BeatRate_{\text{non-role}}\geq0.05.
$$

## H4：oracle-good events 存在，但 legal score 仍可能缺 calibration

v9.2.41 oracle precision $1.0$、coverage $0.12$，说明 good events 存在。H4 认为 legal score 的主要问题可能是 threshold / uncertainty calibration，而不是 carrier mechanism。

H4 成立标准：

fresh oracle upper bound remains:

$$
OraclePrecision\geq0.75,
$$

$$
OracleCoverage\in[0.03,0.15].
$$

If oracle pass but all legal scores fail：

```text
controller calibration blocker
```

If oracle fail：

```text
carrier mechanism blocker
```

## H5：不能用 dataset branch 修 LDO

即使 S7 在某个 dataset 上明显更好，也不能写 dataset-specific route。只能用 signal features：

```text
control_gap_score
tail_value_score
role_score
uncertainty
branch_ratio
effective_derivative
family_reliability
```

H5 成立标准：

```text
dataset_name_used = 0
seed_id_used = 0
class_specific_hand_rule = 0
```

---

# Part IV. 并行执行设计

v9.2.42 不再单线执行。所有 lane 在同一个 runner 内并行落盘：

```text
Lane A:
  fresh multi-stratum carrier/value measurement

Lane B:
  frozen S7 confirmation

Lane C:
  score matrix S7/S9/S10/S11/S12

Lane D:
  role-wise late attach fresh validation

Lane E:
  oracle upper-bound and shuffle controls

Lane F:
  LDO / LSO

Lane G:
  paired replay scout and official replay

Lane H:
  short-run scout if paired replay pass
```

Gate discipline：

```text
diagnostic rows can be computed early
official_eligible = 1 only if all upstream gates pass
```

This accelerates experiment without faking route success.

---

# Part V. Candidate and score design

## 1. Base

Official base remains:

```text
R2-LQ-fanin-output-scale-confirmed
```

from v9.2.40 repaired robust pass.

## 2. Attach candidates

### A0：v9.2.40 best attach survivor

Reference carrier.

### A1：LateAttachZeroLinearTail

Same as v9.2.40 simple carrier.  
Used to test whether S7 generalizes independent of role-wise carrier.

### A2：LateAttachControlGapChannel

Functional branch explicitly computes control gap lower-bound.

### A3：LateAttachRoleWiseFT7EdgeCarrier

Edge-owned role-wise functional carrier：

$$
\phi_{ij}(h)
=
\phi^{base}_{ij}(h;\theta_T^\star)
+
\lambda(e)
[
\alpha_s(e)\phi^{F,s}_{ij}(h;\theta_{F,s})
+
\alpha_h(e)\phi^{F,h}_{ij}(h;\theta_{F,h})
].
$$

Role weights use only signal features：

$$
\alpha_r(e)
=
f(
CEp99,
MarginP10,
Curvature,
BranchRatio,
EffectiveDerivative,
ControlGap,
Uncertainty
).
$$

No dataset name.

### A4：FamilyValueLateAttach

Family:

$$
family(e)=(stratum(e),horizon(e),risk\_bucket(e),role\_bucket(e)).
$$

### A5：Hybrid RoleWise + ControlGap

Role-wise carrier with control-gap lower-bound accept rule.

---

## 3. Score candidates

### S7：Frozen HybridMonotoneLegal

Exactly freeze v9.2.41 S7.  
No changes allowed.

### S9：ControlGapLCB

$$
S_9
=
LCB(Gain_{\text{Real}})
-
UCB(\max(Gain_{\text{AdamWParallel}},Gain_{\text{bestLR}})).
$$

### S10：RoleWiseControlGap

$$
S_{10}
=
\alpha_s S_{\text{stack-gap}}
+
\alpha_h S_{\text{head-gap}}.
$$

### S11：FamilyReliabilityGap

$$
S_{11}
=
\mathbb{E}[V\mid family]
-
\kappa\operatorname{Std}[V\mid family].
$$

### S12：HybridMonotoneFreshPreRegistered

A monotone rule using only:

```text
control_gap_score
tail_value_score
role_score
uncertainty
branch_ratio
effective_derivative
family_reliability
```

No dataset name, no validation/test metric, no posthoc outcome at commit.

### S13：Oracle

Posthoc diagnostic only.  
Never official.

---

# Part VI. 实验阶段

## P0：v9.2.41 boundary reproduction

### 目标

复现 v9.2.41 boundary。

### 必须记录

```text
route
source_route_v9240
value_failure_mode
best_legal_score
best_score_auc
best_score_corr
accepted_precision
accepted_coverage
accepted_bad_event_rate
rolewise_attach_pass
oracle_upper_bound_pass
oracle_precision
oracle_coverage
shuffle_controls_pass
leave_dataset_out_pass
leave_stratum_out_status
primary_blocker
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R3-ControlGapScorePass
value observability pass = 1
LDO pass = 0
LSO not evaluable or fail
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_gate_ladder_value_to_leaveout.svg
p0_score_success_but_leaveout_fail.svg
```

---

## P1：Fresh multi-stratum event expansion

### 目标

生成 fresh rows，补足 signal strata，避免继续在 v9.2.40 source rows 上过拟合。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
signal_strata = S1-S8
attach_candidates = A1,A2,A3,A4,A5
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
event_accepted_by_source_gate
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
family_id
posthoc_value
fake_proxy_flag
```

### 判断标准

Fresh measurement pass：

```text
fresh_row_count >= 3 * 5 * 4 * 4 * 3
measured_signal_strata_count >= 4
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
p1_fresh_event_coverage_by_stratum.svg
p1_carrier_movement_distribution.svg
p1_control_gap_by_stratum.svg
p1_task_safety_by_attach.svg
```

---

## P2：Frozen S7 confirmation

### 目标

不改 S7，直接在 fresh rows 上验证 v9.2.41 score 是否泛化。

### 必须记录

```text
score_id = S7-FrozenHybridMonotoneLegal
row_id
score_value
grounded_value
Y_beat
accepted
precision
coverage
bad_event_rate
corr
auc
dataset_name_used
posthoc_used_at_commit
validation_used
test_used
```

### 判断标准

S7 confirmation pass：

$$
AUC(S7)\geq0.70
$$

or:

$$
Corr(S7,V)\geq0.35.
$$

Accept gate：

$$
Precision\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05.
$$

Legality：

```text
dataset_name_used = 0
posthoc_used_at_commit = 0
validation_used = 0
test_used = 0
```

### 可视化

```text
p2_s7_score_vs_value.svg
p2_s7_precision_coverage.svg
p2_s7_by_dataset_and_stratum.svg
```

---

## P3：Parallel score matrix

### 目标

并行测试 S7/S9/S10/S11/S12/S13，避免一轮只试一个 score。

### 必须记录

```text
score_id
attach_candidate
signal_stratum
horizon
corr
auc
precision
coverage
bad_event_rate
accepted_event_count
legal_score
official_eligible
dataset_name_used
posthoc_used_at_commit
validation_used
test_used
gate_missing_reason
```

### 判断标准

Diagnostic score pass：

$$
AUC\geq0.60
$$

or:

$$
Corr\geq0.20.
$$

Official score pass：

$$
AUC\geq0.70
$$

or:

$$
Corr\geq0.35.
$$

Precision / coverage / bad-event gate as above.

Oracle interpretation：

```text
If S13 oracle pass but legal scores fail:
  controller blocker.

If S13 oracle fail:
  carrier mechanism blocker.
```

### 可视化

```text
p3_score_matrix_auc_corr.svg
p3_score_matrix_precision_coverage.svg
p3_oracle_vs_legal_gap.svg
p3_score_legality_matrix.svg
```

---

## P4：Role-wise late attach fresh validation

### 目标

验证 A3/A5 role-wise mechanism 是否提升 value generalization。

### 必须记录

```text
attach_candidate
role_channels
edge_owned_param_fraction
manual_forward
manual_backward
manual_update
inactive_equivalence_pass
no_event_preservation_pass
carrier_pass
r_z_tail
r_perp_tail
role_stack_score
role_head_score
role_weight_stack
role_weight_head
value_auc
value_corr
precision
coverage
bad_event_rate
```

### 判断标准

Role-wise attach pass：

```text
strict PureKAN contract pass = 1
inactive equivalence pass = 1
no-event preservation pass = 1
carrier pass = 1
```

Role-wise value pass：

$$
AUC_{\text{role}}\geq0.70
$$

or:

$$
Corr_{\text{role}}\geq0.35.
$$

Role-wise improvement diagnostic：

$$
AUC_{\text{role}}-AUC_{\text{non-role}}\geq0.05.
$$

### 可视化

```text
p4_rolewise_value_score.svg
p4_role_contribution_breakdown.svg
p4_rolewise_vs_nonrole_pareto.svg
```

---

## P5：Leave-dataset-out and leave-stratum-out

### 目标

验证 dataset-agnostic 和 stratum-agnostic。

### 设置

Leave-dataset-out：

```text
train/calibrate on MNIST + Fashion, evaluate KMNIST
train/calibrate on MNIST + KMNIST, evaluate Fashion
train/calibrate on Fashion + KMNIST, evaluate MNIST
```

Leave-stratum-out：

```text
train/calibrate on all but one signal stratum
evaluate held-out stratum
```

### 必须记录

```text
split_type
heldout
score_id
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

At least two held-out datasets satisfy：

$$
BeatRate_{\text{Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{Real vs bestLR}}\geq0.50.
$$

LSO pass：

At least $70\%$ held-out strata task-safe and CE-tail non-worse：

$$
CEp99_{\text{heldout,Real}}\leq CEp99_{\text{AdamW}}+\epsilon.
$$

### 可视化

```text
p5_leave_dataset_out_matrix.svg
p5_leave_stratum_out_matrix.svg
p5_hidden_dataset_tuning_audit.svg
```

---

## P6：Official paired replay

### 目标

验证 RealFunctional 是否在 strong controls 下有局部因果优势。

### 设置

```text
base = R2 repaired base checkpoint
attach_candidate = best P5 survivor
score_id = best legal score survivor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, InvertedRole, ValueScoreShuffled, FunctionalChannelShuffled, TailMaskShuffled, RoleScoreShuffled
```

### 必须记录

```text
attach_candidate
score_id
dataset
seed
horizon
signal_stratum
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
p6_official_paired_replay_pareto.svg
p6_macro_beat_rate.svg
p6_signal_stratum_win_matrix.svg
p6_shuffle_control_matrix.svg
p6_system_gate_distribution.svg
```

---

## P7：Short-run scout

### 目标

如果 P6 pass，验证连续训练中机制是否保持。

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

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamW}},
$$

$$
ECE_{\text{functional}}\leq ECE_{\text{AdamW}},
$$

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

### 可视化

```text
p7_short_run_task_mechanism_pareto.svg
p7_short_run_controls.svg
p7_event_timeline.svg
p7_ce_tail_margin_panel.svg
```

---

## P8：Full 10-seed validation

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

## P9：Robustness / strong baseline / external-ready

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
contract_audit_v9242.csv
p0_v9241_boundary_reproduction.csv
p1_fresh_multistratum_event_expansion.csv
p2_frozen_s7_confirmation.csv
p3_parallel_score_matrix.csv
p4_rolewise_late_attach_fresh_validation.csv
p5_leave_dataset_and_stratum_out.csv
p6_official_paired_replay.csv
p7_short_run_functional_validation.csv
p8_full_10seed_functional_validation.csv
p9_robustness_external_ready.csv
fresh_event_trace_v9242.csv
frozen_s7_trace_v9242.csv
score_matrix_trace_v9242.csv
rolewise_attach_trace_v9242.csv
oracle_upper_bound_trace_v9242.csv
leaveout_trace_v9242.csv
paired_replay_branch_trace_v9242.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9241_boundary_unstable
F3_dataset_tuning_detected
F4_fresh_rows_insufficient
F5_multistratum_coverage_fail
F6_frozen_s7_fail
F7_all_legal_scores_fail
F8_oracle_low_carrier_mechanism_fail
F9_oracle_high_legal_controller_fail
F10_rolewise_attach_fail
F11_leave_dataset_out_fail
F12_leave_stratum_out_fail
F13_paired_replay_control_equivalent
F14_shuffle_control_pass
F15_functional_lr_equivalent
F16_short_run_task_drop
F17_full_run_no_macro_hard_stratum_gain
F18_strong_baseline_explains_gain
F19_robustness_fail
F20_external_not_ready
F21_fake_or_proxy_violation
F22_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-FrozenS7FreshPass:
  S7 from v9.2.41 passes on fresh multi-stratum rows.

R2-NewScorePassFrozenS7Fail:
  frozen S7 fails, but another pre-registered legal score passes.

R3-MultiStratumValuePass:
  value observability passes across at least two accepted signal strata.

R4-RoleWiseValuePass:
  role-wise late attach improves value and passes gates.

R5-OracleHighLegalScoreLow:
  good events exist, but legal controller fails.

R6-OracleLowCarrierMechanismReset:
  oracle cannot find enough good events; carrier mechanism must reset.

R7-LeaveDatasetOutPass:
  controller generalizes without dataset-specific tuning.

R8-LeaveStratumOutPass:
  score generalizes across held-out signal strata.

R9-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R10-BaseAttachCarrierValidButValueUnstable:
  base/attach/carrier still valid, but value score not robust.

R11-StrictPureKANFunctionalShortRunPass:
  short-run task-safe mechanism gain.

R12-StrictPureKANFunctionalFullPass:
  full 10-seed macro / hard-stratum / geometry gain.

R13-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` 必须记录：

```text
route
v9241_boundary_pass
dataset_tuning_detected
fresh_row_count
measured_signal_strata_count
accepted_signal_strata_count
frozen_s7_pass
best_score_id
best_attach_candidate
value_observability_pass
value_auc
value_corr
accepted_precision
accepted_coverage
accepted_bad_event_rate
oracle_upper_bound_pass
rolewise_value_pass
leave_dataset_out_pass
leave_stratum_out_pass
paired_replay_pass
short_run_pass
full_run_pass
external_ready
primary_blocker
next_required_implementation
success_v9242_strict_purekan_functional
success_v9242_full_functional
success_v9242_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 fresh multi-stratum event expansion
  P2 frozen S7 confirmation
  P3 parallel score matrix
  P4 role-wise late attach validation

Batch 2:
  P5 leave-dataset-out / leave-stratum-out
  P6 paired replay scout for eligible scores

Batch 3:
  official P6 paired replay
  P7 short-run if paired replay passes

Batch 4:
  P8 full 10-seed
  P9 robustness / strong baseline
```

Gate rule：

```text
P1-P4 can run in parallel.
P6 diagnostic can run before P5 finishes.
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier pass
  value observability pass
  LDO/LSO pass
```

---

# Part X. 停止条件

## Minimum diagnostic success

```text
v9.2.41 boundary reproduced
fresh multi-stratum rows measured
frozen S7 tested
score matrix completed
role-wise attach audited
oracle upper-bound measured
no fake/proxy/offload/loss/teacher violation
```

## Value mechanism success

```text
Minimum diagnostic success
+
at least one legal score passes observability
+
precision / coverage / bad-event gate pass
+
LDO / LSO pass
```

## Local functional success

```text
Value mechanism success
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
1. v9.2.41 boundary cannot be reproduced；
2. fresh multi-stratum expansion cannot produce enough strata；
3. frozen S7 fails and all legal pre-registered scores fail；
4. oracle upper bound fails；
5. role-wise late attach cannot be implemented as strict PureKAN；
6. leave-dataset-out fails；
7. leave-stratum-out fails；
8. paired replay remains control-equivalent；
9. shuffle controls pass, indicating overfit；
10. short-run task drops；
11. full run gives no macro / hard-stratum / geometry gain；
12. functional breaks system gate；
13. gains are explained by QuadraticFeatureMLP；
14. any teacher/loss/fake/proxy/offload violation occurs。
```

---

# Part XI. 最终解释规则

## Case A：frozen S7 passes

可以声明：

```text
v9.2.41 S7 was not source-row overfit; it generalizes to fresh multi-stratum rows.
```

但不能声明 strict functional success，除非 LDO/LSO 和 paired replay 也通过。

## Case B：S7 fails but another legal score passes

必须声明：

```text
v9.2.41 discovered the right value family, but S7 itself was not robust enough.
```

下一步应 promote new score only if LDO/LSO pass.

## Case C：oracle high but legal score fail

必须声明：

```text
Good functional events exist, but current legal controller cannot find them robustly.
```

下一步修 controller calibration / uncertainty.

## Case D：oracle low

必须声明：

```text
Carrier movement exists, but current carrier does not produce enough control-resistant good events.
```

下一步 reset carrier / role-wise mechanism.

## Case E：LDO / LSO fail

必须声明：

```text
The score is not dataset-agnostic or stratum-agnostic enough for official success.
```

不能用 dataset-specific tuning 写成功。

## Case F：paired replay passes

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full validation.

---

# Part XII. 最终建议

v9.2.42 的一句话策略是：

$$
\boxed{
\text{不要再在 source rows 上打磨 score；必须 fresh multi-stratum 验证 S7/role-wise/control-gap 是否真的泛化。}
}
$$

当前最关键的问题不是：

```text
base 是否稳定；
attach 是否污染；
carrier 是否 silent；
Fashion 怎么调；
KMNIST 怎么调；
MNIST 是否 abstain。
```

而是：

```text
1. v9.2.41 的 S7 是否能在 fresh rows 上复现？
2. value score 是否跨 signal strata 泛化？
3. role-wise late attach 是否提高 leave-out robustness？
4. oracle-good events 是否仍然存在？
5. legal score 能否找到 oracle-good events？
6. LDO/LSO 能否通过？
7. official paired replay 能否打过 AdamWParallel / bestLR？
```

v9.2.42 的结果将给出明确分叉：

```text
if S7/role-wise passes LDO+LSO+paired replay:
  strict PureKAN functional gets local causal evidence.

if oracle high but legal scores fail:
  controller calibration is blocker.

if oracle low:
  carrier mechanism is blocker.

if LDO/LSO fail:
  score is not general enough; no dataset tuning allowed.
```
