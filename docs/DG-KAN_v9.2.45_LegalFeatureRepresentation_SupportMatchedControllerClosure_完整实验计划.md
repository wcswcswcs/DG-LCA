# DG-KAN v9.2.45 Legal Feature Representation 与 Support-Matched Controller Closure 完整实验计划

> 本计划基于 v9.2.44 `Safe Good-Event Support Recovery 与 Carrier Mechanism Reset` 的真实复盘制定。  
> v9.2.44 的 terminal route 是：
>
> ```text
> route = R8-OracleHighLegalFeatureGap
> base_candidate = LQ-t2-h256
> success_v9244_strict_purekan_functional = False
> success_v9244_full_functional = False
> success_v9244_external_ready = False
> ```
>
> v9.2.44 的核心事实是：
>
> ```text
> P1 failure decomposition pass = 1
> primary mode = A6-risk_score_missing
> bad-event attribution fraction = 1.0
> oracle-collapse attribution fraction = 1.0
>
> P2 risk-first support survivor:
>   best = F1-BranchRatioRiskSafe
>   oracle precision = 0.967480
>   oracle coverage = 0.040039
>   oracle bad-event = 0.032520
>
> P3 carrier reset:
>   no route-level carrier pass
>   best = A1-RiskBoundedTailCarrier
>   oracle support exists
>   carrier active = 0
>   r_perp_tail = 0.084411 < 0.10
>
> P4 legal controller:
>   best = C1-RiskFirstS7
>   precision = 0.794118
>   coverage = 0.022135 < 0.03
>   bad-event = 0.117647 > 0.05
>
> current blocker:
>   safe_good_support_recovered_but_legal_controller_failed
> ```
>
> 因此 v9.2.45 的核心任务不再是继续修 base、attach、carrier，也不是继续在同一个 S7 threshold 上小修小补。当前最本质的问题是：
>
> $$
> \boxed{
> \text{oracle/risk filter 能找到 safe-good support，但 legal controller 缺少足够的 commit-time sufficient statistics。}
> }
> $$
>
> 也就是说，现在不是 “good events 不存在”，而是 “good events 的可观测法律特征还不够”。v9.2.45 要把问题从 **oracle support recovery** 推进到 **legal feature representation recovery**：
>
> $$
> \boxed{
> \text{用 train-stream legal features 逼近 F1-BranchRatioRiskSafe 的 safe-good support，并通过 leave-out 与 paired replay。}
> }
> $$

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
  report bad-event / oracle-collapse distribution by dataset
  report leave-dataset-out generalization
  report signal-stratum and family composition by dataset
  report per-dataset diagnostic plots

forbidden:
  if dataset == Fashion: use controller A
  if dataset == KMNIST: use primitive B
  if dataset == MNIST: abstain
  tune threshold separately per dataset
  promote a controller because it rescues only one dataset
  use validation/test metric at commit time
  use posthoc replay outcome at commit time for official controller
```

---

# Part I. 对 v9.2.44 的独立判断

## 1. v9.2.44 没有达到目标

v9.2.44 没有 strict PureKAN functional success。它只完成了 risk-first support recovery 的关键一步，但没有完成 legal controller、leave-out、paired replay、short-run、full-run。

这轮不能声明：

```text
strict PureKAN functional causal evidence
full functional success
external-ready success
Beyond-MLP success
```

原因是 official legal controller 仍未通过。best controller `C1-RiskFirstS7` 虽然 precision 达到 `0.794118`，但 coverage 只有 `0.022135`，低于 `0.03` gate；bad-event 为 `0.117647`，高于 `0.05` gate。因此它不能进入 leave-out 或 paired replay。

## 2. v9.2.44 的真实进展

v9.2.43 的问题是 fresh v2 中 bad-event rate 超过 gate，oracle 也失效。v9.2.44 证明这个 collapse 不是不可修复的机制崩溃，而是 risk feature 缺失导致的 support 污染。P1 把 primary mode 归因为 `A6-risk_score_missing`，bad-event 与 oracle-collapse attribution fraction 都是 `1.0`。

最重要的是 P2：

```text
F1-BranchRatioRiskSafe:
  oracle precision = 0.967480
  coverage = 0.040039
  bad-event = 0.032520
```

这已经满足 safe-good oracle support 的核心门槛：

$$
OraclePrecision_{\text{safe-good}}\geq0.75,
$$

$$
OracleCoverage_{\text{safe-good}}\in[0.03,0.15],
$$

$$
OracleBadEventRate\leq0.05.
$$

这说明：

$$
\boxed{
\text{当前 functional carrier/event population 中仍然存在足够 safe 且 control-resistant 的 good events。}
}
$$

因此 v9.2.43 的悲观结论 “carrier mechanism 必须全面 reset” 被 v9.2.44 部分纠正：至少在 F1 risk-first support 下，good events 仍存在。现在真正 blocker 从 carrier reset 转为 legal feature representation。

## 3. 当前真正 blocker

当前 blocker 是：

$$
\boxed{
\text{oracle support recovered, but legal controller cannot reproduce it safely and with enough coverage.}
}
$$

也可以写成：

$$
\boxed{
\text{F1-BranchRatioRiskSafe 是一个好 support，但 C1-RiskFirstS7 不是一个足够好的 legal approximation。}
}
$$

其中包含三个子问题。

### 3.1 Legal controller 没有学到 F1 support 的边界

F1 oracle support 的 bad-event 是 `0.032520`，但 C1 的 bad-event 是 `0.117647`。这说明 C1 的 risk filter 没有正确排除 unsafe support，或者它在 heldout split 上没有保持 F1 的 safe boundary。

需要记录：

$$
Overlap(C1,F1)
=
\frac{|Accept_{C1}\cap Accept_{F1}|}
{|Accept_{C1}\cup Accept_{F1}|}.
$$

如果 overlap 很低，则 C1 不是 F1 的合法近似。

### 3.2 Legal controller coverage 不足

C1 coverage 是 `0.022135`，低于 gate。它可能过度保守，也可能把安全事件筛掉了。需要画 precision-coverage-bad-event 三维曲线，而不是只看一个 threshold。

### 3.3 Carrier reset 没有产生 route-level pass

P3 中 `A1-RiskBoundedTailCarrier` 有 oracle support，但 carrier active = `0`，`r_perp_tail=0.084411<0.10`。这说明 A1 可能是一个 risk-clean support filter，但不是一个合格的 active functional carrier。v9.2.45 不应把 A1 直接 promote 为 carrier；应把它拆成两个角色：

```text
A1 as support filter:
  useful diagnostic.

A1 as carrier:
  failed carrier-active gate.
```

---

# Part II. v9.2.45 总体目标

v9.2.45 的总体目标是：

$$
\boxed{
\text{把 F1 risk-first oracle support 转化为 legal train-stream controller，并关闭 leave-out + paired replay。}
}
$$

目标分为七层。

## 1. Legal-feature gap attribution success

必须解释为什么 F1 oracle pass，而 C1 legal fail。至少归因到以下机制之一：

```text
G1-feature_missing:
  C1 缺少 F1 的关键 branch/risk feature。

G2-threshold_miscalibration:
  C1 score 排序可用，但 threshold 在 heldout 上错位。

G3-risk_boundary_mismatch:
  C1 的 risk predictor 与 F1 safe boundary 不一致。

G4-family_imbalance:
  C1 accepted events 被少数 event family 主导，导致 heldout bad-event 升高。

G5-control_gap_mismatch:
  C1 能预测 task-safe，但不能预测 Real-vs-control gap。

G6-carrier_support_mismatch:
  C1 在一个 carrier 上有效，但跨 attach/carrier 后失败。

G7-legal_infeasible:
  F1 依赖 posthoc or non-commit-time information，无法 legalize。

G8-split_variance:
  good-event support 太薄，heldout split 方差导致 coverage / bad-event 不稳。
```

成功标准：

```text
>= 90% C1 false positives assigned to primary failure mode
>= 90% C1 false negatives assigned to primary failure mode
dataset_name_used = 0
```

## 2. Legal sufficient statistic success

必须实现一组 legal train-stream features，使其能近似 F1 safe-good oracle support。定义：

$$
Y_{\text{F1-support}}
=
\mathbb{1}[e\in Accept_{F1}].
$$

Legal feature representation success：

$$
AUC(S_{\text{legal}},Y_{\text{F1-support}})\geq0.75,
$$

且对真正 safe-good label：

$$
AUC(S_{\text{legal}},Y_{\text{safe-good}})\geq0.70
$$

or:

$$
Corr(S_{\text{legal}},V_{\text{safe-grounded}})\geq0.35.
$$

## 3. Heldout legal controller success

在 calibration split 上选 threshold，在 heldout split 上评价。不得直接用 test / validation / posthoc commit metric。

Heldout gate：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{heldout}}\leq0.05.
$$

Multi-support gate：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

## 4. System gate

Legal feature factory 不能把 system cost 拉爆。额外 legal controller 成本必须满足：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

如果需要 train-stream micro-probe，则 probe amortized overhead 需要单独记录：

$$
Overhead_{\text{legal-probe}}\leq0.20.
$$

超过该值只能作为 diagnostic，不得 official promote。

## 5. Leave-out success

Leave-dataset-out：

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

Leave-stratum-out：

At least $70\%$ held-out strata task-safe and：

$$
CEp99_{\text{heldout,Real}}\leq CEp99_{\text{AdamW}}+\epsilon,
$$

$$
BeatRate_{\text{heldout-stratum,Real vs AdamWParallel}}\geq0.50.
$$

## 6. Official paired replay success

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

Shuffle controls must fail：

```text
ValueScoreShuffled = fail
FunctionalChannelShuffled = fail
TailMaskShuffled = fail
RoleScoreShuffled = fail
DatasetRouteShuffled = fail
EventRouteShuffled = fail
InvertedRoleMask = fail
RiskScoreShuffled = fail
BranchRatioShuffled = fail
```

## 7. Short-run scout success

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

## H1：v9.2.44 已经证明 carrier/event population 中存在 safe-good support

H1 基于 F1-BranchRatioRiskSafe：

$$
Precision=0.967480,
$$

$$
Coverage=0.040039,
$$

$$
BadEvent=0.032520.
$$

H1 成立标准：

在 v9.2.45 fresh split 中，F1 或 F1-equivalent support 仍满足：

$$
OraclePrecision\geq0.75,
$$

$$
OracleCoverage\in[0.03,0.15],
$$

$$
OracleBadEventRate\leq0.05.
$$

H1 失败说明 v9.2.44 的 support recovery 是 split-specific，需要回到 carrier/support design。

## H2：C1 失败是 legal feature representation gap，而不是 good-event absence

C1 已经有 precision `0.794118`，说明它不是完全无信号；但 coverage 和 bad-event 同时失败。H2 认为 C1 的问题是缺少 branch/risk/family sufficient statistics。

H2 成立标准：

新 legal feature set 能同时把：

$$
BadEventRate: 0.117647 \rightarrow \leq0.05,
$$

并把：

$$
Coverage: 0.022135 \rightarrow \geq0.03.
$$

## H3：Risk-first controller 必须拆成 RiskSafe、ValuePositive、ControlResistant 三段

单一 score 难以同时控制风险与 value。v9.2.45 的 official controller 应采用：

$$
Accept(e)
=
RiskSafe(e)
\land
ValuePositive(e)
\land
ControlResistant(e).
$$

其中：

$$
RiskSafe(e)=\mathbb{1}[S_{\text{risk}}(e)\leq\rho],
$$

$$
ValuePositive(e)=\mathbb{1}[S_{\text{value}}(e)\geq\tau],
$$

$$
ControlResistant(e)=\mathbb{1}[S_{\text{gap}}(e)>0].
$$

H3 成立标准：

三段式 controller 相比 C1：

$$
BadEventRate_{\text{tri-stage}} < BadEventRate_{C1},
$$

$$
Coverage_{\text{tri-stage}} > Coverage_{C1},
$$

且 heldout pass。

## H4：Branch-ratio 是关键 risk sufficient statistic，但必须合法化

F1 名称是 `BranchRatioRiskSafe`，说明 branch ratio 可能是 risk support 的核心。必须确认 branch ratio 是否为 commit-time legal feature，而不是 posthoc artifact。

H4 成立标准：

```text
branch_ratio_available_pre_commit = 1
branch_ratio_no_validation_test = 1
branch_ratio_no_posthoc_replay = 1
```

且 branch-ratio-only 或 branch-ratio+risk score 在 heldout 上有：

$$
AUC(Y_{\text{safe-good}})\geq0.60
$$

or:

$$
BadEventRate\leq0.05 \text{ at usable coverage}.
$$

## H5：如果 F1 support remains high but all legal controllers fail，问题是 legal feature set 不足

此时不应继续调 threshold，也不应修 base/attach。下一步要设计 richer train-stream legal features。

## H6：如果 F1 support collapses，问题回到 carrier/support stability

如果 F1 在 fresh split 上也不能维持 oracle support，则 v9.2.44 的 support recovery 只是 sample-specific；需要重新做 carrier mechanism reset。

## H7：dataset tuning 仍然禁止

即使 legal feature gap集中出现在某个 dataset，official controller 仍不能使用 dataset name。所有 features 必须是 event-level、signal-level、risk-level、role-level。

---

# Part IV. 并行执行设计

v9.2.45 必须并行，不再一轮只测一个 gate。Runner 同时执行：

```text
Lane A:
  v9.2.44 boundary reproduction and F1/C1 gap autopsy

Lane B:
  F1 support stability on fresh split

Lane C:
  legal feature factory:
    branch ratio
    risk LCB
    control gap LCB
    role gap
    family reliability
    drift-diffusion / SNR
    train-stream micro-probe

Lane D:
  tri-stage controller calibration

Lane E:
  legal-vs-oracle support matching

Lane F:
  leave-dataset-out / leave-stratum-out

Lane G:
  paired replay scout and official replay

Lane H:
  short-run scout if paired replay passes

Lane I:
  fallback:
    if oracle support collapses -> carrier reset
    if oracle high legal low -> legal feature representation
```

Gate discipline：

```text
diagnostic rows may be measured early
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  oracle support pass
  legal controller pass
  LDO/LSO pass
```

---

# Part V. Legal feature factory

## 1. Base and carrier

Official base remains：

```text
R2-LQ-fanin-output-scale-confirmed
```

Carrier candidates：

```text
A0-current-v9244
A3-RoleWiseFT7ResetCarrier
A6-HybridRoleControlRiskCarrier
```

A1 can be used as support filter diagnostic, but cannot be promoted as carrier unless it passes carrier active:

$$
r_{z,\text{tail}}\geq0.10,
$$

$$
r_{\perp,\text{tail}}\geq0.10.
$$

## 2. Legal feature candidates

### LF0：Current C1 features

Reference only.

### LF1：BranchRatioRisk Feature

$$
S_{\text{branch-risk}}
=
a_1 BranchRatio
+
a_2 EffectiveDerivative
+
a_3 TailActivationMass
-
a_4 Uncertainty.
$$

No dataset name.

### LF2：Risk LCB Feature

$$
S_{\text{risk}}
=
w_1 CEp99
+
w_2 WrongConfidenceP95
-
w_3 MarginP10
+
w_4 Uncertainty
+
w_5 Curvature.
$$

RiskSafe：

$$
RiskSafe(e)=\mathbb{1}[S_{\text{risk}}\leq\rho].
$$

### LF3：ControlGap LCB Feature

$$
S_{\text{gap}}
=
LCB(Gain_{\text{Real}})
-
UCB(\max(Gain_{\text{AdamWParallel}},Gain_{\text{bestLR}})).
$$

### LF4：Role-Gap Feature

$$
S_{\text{role-gap}}
=
\alpha_sS_{\text{stack-gap}}
+
\alpha_hS_{\text{head-gap}}
-
R_{\text{risk}}.
$$

### LF5：Family Reliability Feature

Family must not contain dataset name：

$$
family(e)
=
(stratum,horizon,risk\_bucket,role\_bucket,attach\_type).
$$

Reliability：

$$
Rel(f)
=
\mathbb{E}[Y_{\text{safe-good}}\mid f]
-
\kappa\sqrt{\operatorname{Var}(Y_{\text{safe-good}}\mid f)}.
$$

### LF6：Drift-Diffusion / SNR Feature

Inspired by signal-channel reasoning, define:

$$
S_{\text{snr}}
=
\frac{\|\mu_{\text{train-stream}}\|^2}
{\sigma^2_{\text{train-stream}}+\epsilon}.
$$

This is not a loss and not a teacher. It is a commit-time update statistic computed from train stream only.

### LF7：Train-Stream Micro-Probe Feature

Use only train split. Split batch into update batch and probe batch:

```text
update batch:
  used to form candidate functional update

probe batch:
  used to estimate train-stream population-risk proxy
```

Legal constraints：

```text
no validation
no test
no dataset name
no posthoc replay outcome
probe overhead measured
```

Feature：

$$
S_{\text{probe}}
=
-\Delta CE_{\text{probe}}
+
\lambda_m\Delta Margin_{\text{probe}}
-
\lambda_r Risk_{\text{probe}}.
$$

Official only if：

$$
Overhead_{\text{probe}}\leq0.20.
$$

---

# Part VI. Controller candidates

## C0：C1-RiskFirstS7 reference

Current best failed legal controller.

## C1：BranchRiskTriStage

$$
Accept(e)
=
RiskSafe_{\text{branch}}(e)
\land
S7(e)\geq\tau
\land
S_{\text{gap}}(e)>0.
$$

## C2：RiskLCB-ControlGap

$$
Accept(e)
=
\mathbb{1}[S_{\text{risk}}\leq\rho]
\land
\mathbb{1}[S_{\text{gap}}\geq\tau_g].
$$

## C3：RoleGapRiskFirst

$$
Accept(e)
=
RiskSafe(e)
\land
S_{\text{role-gap}}(e)\geq\tau_r.
$$

## C4：FamilyBalancedRiskS7

$$
Accept(e)
=
RiskSafe(e)
\land
S7(e)\geq\tau
\land
Rel(family(e))\geq r_0
\land
FamilyBalance(e).
$$

## C5：SNR-GatedControlGap

$$
Accept(e)
=
S_{\text{snr}}(e)\geq\tau_s
\land
S_{\text{gap}}(e)>0
\land
RiskSafe(e).
$$

## C6：TrainStreamProbeController

$$
Accept(e)
=
S_{\text{probe}}(e)\geq\tau_p
\land
RiskSafe(e)
\land
ControlResistant(e).
$$

## C7：HybridLegalMonotone

A monotone controller using only:

```text
branch_ratio
effective_derivative
tail_activation_mass
risk_lcb
control_gap_lcb
role_gap
family_reliability
snr
train_stream_probe
```

No hidden black-box with dataset features.

## C8：Oracle

Posthoc diagnostic only. Never official.

---

# Part VII. 实验阶段

## P0：v9.2.44 boundary reproduction

### 目标

复现 v9.2.44 boundary，确认不是解析或 source artifact 问题。

### 必须记录

```text
route
source_route_v9243
primary_mode
risk_filter_survivor
oracle_precision
oracle_coverage
oracle_bad_event
best_carrier
carrier_active
r_z_tail
r_perp_tail
best_legal_controller
legal_precision
legal_coverage
legal_bad_event
primary_blocker
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R8-OracleHighLegalFeatureGap
F1 oracle support pass = 1
C1 legal controller pass = 0
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_oracle_vs_legal_gate_ladder.svg
p0_support_recovered_controller_failed.svg
```

---

## P1：F1 vs C1 legal feature gap autopsy

### 目标

解释 F1 support 为什么好，C1 controller 为什么失败。

### 必须记录

```text
row_id
event_family
signal_stratum
horizon
attach_candidate
F1_accept
C1_accept
safe_good
bad_event
real_beats_adamwparallel
real_beats_bestlr
branch_ratio
effective_derivative
tail_activation_mass
risk_lcb
control_gap_lcb
role_gap
family_reliability
S7_score
C1_score
false_positive_type
false_negative_type
failure_mode
```

### 判断标准

P1 pass：

```text
C1 false positives attributed fraction >= 0.90
C1 false negatives attributed fraction >= 0.90
F1/C1 overlap measured
dataset_name_used = 0
```

Key metrics：

$$
Jaccard(F1,C1),
$$

$$
FPBadRate(C1),
$$

$$
FNSafeGoodRate(C1),
$$

$$
AUC(S_{\text{branch-risk}},Y_{\text{safe-good}}).
$$

### 可视化

```text
p1_f1_c1_venn.svg
p1_false_positive_bad_event.svg
p1_false_negative_safe_good.svg
p1_feature_gap_heatmap.svg
p1_branch_ratio_vs_risk.svg
```

---

## P2：Fresh support stability

### 目标

确认 F1 support 在 fresh split 上是否稳定，防止 v9.2.44 support recovery 是 split-specific。

### 设置

```text
base = R2 repaired base
carrier = A0-current, A3-rolewise, A6-hybrid
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7
horizons = 20,80,240,640
signal_strata = S1-S8
branches = RealFunctional, AdamWParallel, bestLR, NoOp, Random
```

### 必须记录

```text
row_id
fresh_split_id
carrier_id
dataset
seed
horizon
signal_stratum
event_family
F1_accept
safe_good
bad_event
oracle_precision
oracle_coverage
oracle_bad_event
accepted_strata_count
accepted_family_count
max_family_share
r_z_tail
r_perp_tail
step_q90
memory_ratio
```

### 判断标准

F1 support stability pass：

$$
OraclePrecision_{\text{F1}}\geq0.75,
$$

$$
OracleCoverage_{\text{F1}}\in[0.03,0.15],
$$

$$
OracleBadEventRate_{\text{F1}}\leq0.05.
$$

Multi-family：

```text
accepted_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

### 可视化

```text
p2_f1_support_stability.svg
p2_support_by_family.svg
p2_support_by_stratum.svg
p2_fresh_split_oracle_ci.svg
```

---

## P3：Legal feature factory audit

### 目标

实现 LF1-LF7，并评估它们是否能预测 F1 support 与 safe-good label。

### 必须记录

```text
feature_id
feature_available_pre_commit
uses_dataset_name
uses_validation
uses_test
uses_posthoc
feature_compute_overhead
memory_overhead
AUC_to_F1_support
AUC_to_safe_good
corr_to_safe_grounded_value
precision_at_gate
coverage_at_gate
bad_event_at_gate
```

### 判断标准

Legal feature pass：

```text
feature_available_pre_commit = 1
uses_dataset_name = 0
uses_validation = 0
uses_test = 0
uses_posthoc = 0
```

Predictive gate：

$$
AUC(S_{\text{feature}},Y_{\text{safe-good}})\geq0.60
$$

or:

$$
AUC(S_{\text{feature}},Y_{\text{F1-support}})\geq0.75.
$$

System gate：

$$
FeatureOverhead\leq0.20.
$$

### 可视化

```text
p3_feature_predictivity_matrix.svg
p3_feature_overhead_pareto.svg
p3_feature_legality_matrix.svg
```

---

## P4：Tri-stage legal controller calibration

### 目标

基于 P3 legal features 校准 C1-C7，不能用 dataset-specific threshold。

### Split design

```text
calibration rows:
  choose thresholds / monotone coefficients

heldout rows:
  evaluate official pass

leave-dataset-out:
  evaluated in P5

leave-stratum-out:
  evaluated in P5
```

### 必须记录

```text
controller_id
features_used
thresholds
coefficients
calibration_split_id
heldout_split_id
precision_cal
coverage_cal
bad_event_cal
precision_heldout
coverage_heldout
bad_event_heldout
auc_heldout
corr_heldout
accepted_strata_count
accepted_family_count
max_family_share
dataset_name_used
posthoc_used_at_commit
validation_used
test_used
official_eligible
```

### 判断标准

Heldout controller pass：

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

Multi-family：

```text
accepted_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

### 可视化

```text
p4_controller_precision_coverage_bad.svg
p4_controller_auc_corr.svg
p4_controller_family_coverage.svg
p4_controller_vs_f1_overlap.svg
```

---

## P5：Leave-dataset-out / leave-stratum-out

### 目标

证明 controller 不是 dataset-specific 或 single-stratum overfit。

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
carrier_id
controller_id
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
CEp99_{\text{heldout,Real}}\leq CEp99_{\text{AdamW}}+\epsilon,
$$

$$
BeatRate_{\text{heldout-stratum,Real vs AdamWParallel}}\geq0.50.
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
carrier_id = best P5 survivor
controller_id = best P5 survivor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRiskScore, ShuffledBranchRatio, ShuffledValueScore, FunctionalChannelShuffled, TailMaskShuffled, RoleScoreShuffled, DatasetRouteShuffled, EventRouteShuffled, InvertedRoleMask
```

### 必须记录

```text
carrier_id
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
ShuffledRiskScore = fail
ShuffledBranchRatio = fail
ShuffledValueScore = fail
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

如果 P6 pass，验证局部 paired replay 优势能否在连续训练中保持。

### 设置

```text
steps = 50,240,640
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRiskScore, ShuffledBranchRatio
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
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRiskScore
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

# Part VIII. Required artifacts

```text
run_manifest.json
contract_audit_v9245.csv
p0_v9244_boundary_reproduction.csv
p1_f1_c1_legal_feature_gap_autopsy.csv
p2_fresh_support_stability.csv
p3_legal_feature_factory_audit.csv
p4_tristage_legal_controller_calibration.csv
p5_leave_dataset_and_stratum_out.csv
p6_official_paired_replay.csv
p7_short_run_functional_validation.csv
p8_full_10seed_functional_validation.csv
p9_robustness_external_ready.csv
f1_c1_overlap_trace_v9245.csv
legal_feature_trace_v9245.csv
feature_overhead_trace_v9245.csv
controller_calibration_trace_v9245.csv
leaveout_trace_v9245.csv
paired_replay_branch_trace_v9245.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9244_boundary_unstable
F3_dataset_tuning_detected
F4_f1_support_unstable
F5_legal_feature_gap_unattributed
F6_branch_ratio_not_legal
F7_legal_feature_system_too_expensive
F8_all_legal_features_fail_predictivity
F9_tri_stage_controller_fail
F10_oracle_high_legal_feature_gap
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

# Part IX. Route decision

```text
R1-F1SupportStable:
  F1-BranchRatioRiskSafe support remains oracle-safe-good on fresh split.

R2-LegalFeatureGapAttributed:
  C1 failure explained by missing feature / calibration / family imbalance / risk boundary mismatch.

R3-BranchRatioLegalFeaturePass:
  branch-ratio risk feature is commit-time legal and predictive.

R4-TrainStreamProbeFeaturePass:
  train-stream micro-probe passes predictivity and system overhead gates.

R5-TriStageControllerPass:
  RiskSafe + ValuePositive + ControlResistant controller passes heldout legal gate.

R6-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R7-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R8-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R9-OracleHighLegalFeatureGap:
  F1 support exists, but legal features still cannot identify it.

R10-FeatureSystemTooExpensive:
  legal feature works but exceeds system overhead.

R11-SupportUnstableCarrierReset:
  F1 support collapses on fresh split; return to carrier/support reset.

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
v9244_boundary_pass
dataset_tuning_detected
f1_support_stability_pass
f1_oracle_precision
f1_oracle_coverage
f1_oracle_bad_event
f1_c1_jaccard
legal_feature_gap_mode
best_legal_feature_id
legal_feature_predictivity_pass
legal_feature_overhead
best_controller_id
tri_stage_controller_pass
controller_auc
controller_corr
accepted_precision
accepted_coverage
accepted_bad_event_rate
accepted_strata_count
accepted_family_count
leave_dataset_out_pass
leave_stratum_out_pass
paired_replay_pass
short_run_pass
full_run_pass
external_ready
primary_blocker
next_required_implementation
success_v9245_strict_purekan_functional
success_v9245_full_functional
success_v9245_external_ready
```

---

# Part X. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 F1/C1 legal feature gap autopsy
  P2 fresh support stability
  P3 legal feature factory audit

Batch 2:
  P4 tri-stage legal controller calibration
  P5 leave-dataset-out / leave-stratum-out
  P6 paired replay scout

Batch 3:
  official P6 paired replay
  P7 short-run if paired replay passes

Batch 4:
  P8 full 10-seed
  P9 robustness / strong baseline
```

Gate rule：

```text
P4/P6 diagnostic rows may be measured before all gates finish.
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  oracle support pass
  legal controller pass
  LDO/LSO pass
```

---

# Part XI. 停止条件

## Minimum diagnostic success

```text
v9.2.44 boundary reproduced
F1/C1 legal feature gap attributed
F1 fresh support stability measured
legal feature factory completed
no fake/proxy/offload/loss/teacher violation
```

## Legal controller success

```text
Minimum diagnostic success
+
F1 support stable
+
at least one legal feature passes predictivity
+
tri-stage controller passes heldout precision / coverage / bad-event gate
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
1. v9.2.44 boundary cannot be reproduced；
2. F1 support collapses on fresh split；
3. legal feature gap cannot be attributed；
4. branch-ratio / risk feature is not commit-time legal；
5. all legal features fail predictivity；
6. only effective feature is too expensive；
7. tri-stage controller fails heldout gate；
8. leave-dataset-out fails；
9. leave-stratum-out fails；
10. paired replay remains control-equivalent；
11. shuffle controls pass, indicating overfit；
12. short-run task drops；
13. full run gives no macro / hard-stratum / geometry gain；
14. functional breaks system gate；
15. gains are explained by QuadraticFeatureMLP；
16. any teacher/loss/fake/proxy/offload violation occurs。
```

---

# Part XII. 最终解释规则

## Case A：F1 support stable + tri-stage controller passes

可以声明：

```text
v9.2.44 failed because legal controller missed the oracle support; v9.2.45 recovered a legal commit-time approximation.
```

但不能声明 strict functional success unless LDO/LSO and paired replay pass.

## Case B：F1 support stable but all legal features fail

必须声明：

```text
Safe-good events exist, but current legal train-stream feature set cannot identify them.
```

下一步设计 richer legal features，不修 base/attach.

## Case C：F1 support collapses

必须声明：

```text
v9.2.44 support recovery was not stable.
```

下一步回到 carrier/support reset。

## Case D：legal feature works but too expensive

必须声明：

```text
Value is observable but not system-legal.
```

下一步做 feature kernelization / amortization。

## Case E：LDO/LSO fail

必须声明：

```text
Controller is not dataset-agnostic or stratum-agnostic enough.
```

不能用 dataset-specific tuning 写成功。

## Case F：paired replay passes

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full validation。

---

# Part XIII. 最终建议

v9.2.45 的一句话策略是：

$$
\boxed{
\text{不要再盲目 reset carrier；F1 已恢复 oracle support。现在要恢复 legal feature representation。}
}
$$

当前最关键的问题不是：

```text
base 是否稳定；
attach 是否污染；
carrier 是否 silent；
S7 coverage 是否差一点；
Fashion 怎么调；
KMNIST 怎么调；
MNIST 是否 abstain；
是否再换普通 basis。
```

而是：

```text
1. F1-BranchRatioRiskSafe 为什么能恢复 oracle safe-good support？
2. C1-RiskFirstS7 为什么不能合法复现 F1 support？
3. branch ratio / risk LCB / control-gap LCB 是否是 commit-time legal sufficient statistics？
4. 三段式 RiskSafe + ValuePositive + ControlResistant controller 能否同时过 precision / coverage / bad-event？
5. 这个 controller 能否 leave-dataset-out 和 leave-stratum-out？
6. official paired replay 能否打过 AdamWParallel / bestLR？
```

v9.2.45 的结果会给出清晰分叉：

```text
if legal feature + controller + LDO/LSO + paired replay pass:
  strict PureKAN functional gets local causal evidence.

if F1 support stable but legal features fail:
  controller feature representation is the blocker.

if F1 support collapses:
  carrier/support stability remains the blocker.

if paired replay remains control-equivalent:
  legal value gate is not enough; functional update still lacks independent causal advantage.
```
