# DG-KAN v9.2.44 Safe Good-Event Support Recovery 与 Carrier Mechanism Reset 并行实验计划

> 本计划基于 v9.2.43 `Legal Controller Calibration 与 Paired Replay Closure` 的真实复盘制定。  
> v9.2.43 的 terminal route 是：
>
> ```text
> route = R7-OracleLowCarrierMechanismReset
> base_candidate = LQ-t2-h256
> success_v9243_strict_purekan_functional = False
> success_v9243_full_functional = False
> success_v9243_external_ready = False
> ```
>
> v9.2.43 的关键事实是：
>
> ```text
> P0:
>   source route = R5-OracleHighLegalScoreLow
>   frozen S7 coverage fail
>   oracle pass
>   fake/proxy = 0
>
> P1:
>   s7_coverage_cliff_mode = G1-threshold_borderline
>   source threshold-free point:
>     coverage = 0.03125
>     precision = 0.911111
>     bad-event = 0.0
>
> P2 fresh v2:
>   fresh rows = 18432
>   fresh real events = 3072
>   strata = 8
>   attach = 4
>   fresh bad-event rate = 0.07682291666666667 > 0.05
>   fresh_v2_pass = 0
>
> P3 controller calibration:
>   best legal controller = C1-CalibratedS7Threshold
>   heldout corr = 0.391692
>   heldout precision = 0.544118
>   heldout bad-event = 0.455882
>   official pass = 0
>
> Fresh v2 oracle:
>   oracle precision = 0.375
>   oracle coverage = 0.083333
>   oracle bad-event = 0.625
>   oracle upper bound pass = 0
>
> Downstream:
>   LDO / LSO / paired replay / short-run / full-run = not_run
>
> current blocker:
>   fresh_v2_bad_event_rate_above_gate
> ```
>
> 这不是普通的 controller calibration 失败。v9.2.42 还是 “oracle high, legal low”；v9.2.43 在更宽 fresh v2 event population 上连 oracle 都掉了。  
> 因此 v9.2.44 的核心任务必须从 “继续校准 legal score” 转为：
>
> $$
> \boxed{
> \text{恢复 safe good-event support：先证明 carrier/event family 中存在足够 safe 且 control-resistant 的 good events，再谈 legal controller。}
> }
> $$
>
> 本轮继续加速，采用并行 runner。所有 diagnostic rows 可以并行测，但 official success 仍必须严格 gate，不允许把 gate-blocked downstream 写成通过。

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
  report per-dataset diagnostic plots

forbidden:
  if dataset == Fashion: use controller A
  if dataset == KMNIST: use primitive B
  if dataset == MNIST: abstain
  tune threshold separately per dataset
  promote a carrier because it rescues only one dataset
  use validation/test metric at commit time
  use posthoc replay outcome at commit time for official controller
```

---

# Part I. 对 v9.2.43 的独立判断

## 1. v9.2.43 没有达到目标

v9.2.43 没有达到 strict PureKAN functional success。它没有打开 leave-out、paired replay、short-run、full-run、external-ready。原因不是 “没有 legal controller survivor” 这么简单，而是更前置的 fresh event support 已经失败：

$$
BadEventRate_{\text{fresh v2}}=0.0768229>0.05.
$$

并且 fresh v2 oracle upper bound 也失败：

$$
Precision_{\text{oracle}}=0.375,
$$

$$
BadEventRate_{\text{oracle}}=0.625.
$$

所以不能把 v9.2.43 解释成 “S7 阈值差一点”。v9.2.42 的 frozen S7 coverage 只差 $0.000139$，但 v9.2.43 的 fresh v2 说明更宽事件分布中，good events 不再稳健存在，或者 carrier/event families 的风险结构发生了明显变化。

## 2. v9.2.43 的真实进展

v9.2.43 的价值在于它推翻了一个危险的乐观解释。

v9.2.42 的状态是：

```text
S7 fresh near-pass
oracle pass
legal score low
```

这很容易让人以为只需要 calibration。v9.2.43 做了更大 fresh v2，发现：

```text
oracle pass = 0
fresh safety pass = 0
best legal controller precision/bad-event 也明显失败
```

这说明问题不只是 threshold / calibration，而是：

$$
\boxed{
\text{当前 carrier/event distribution 的 safe good-event support 不稳定。}
}
$$

这是一个更本质的结论。它让我们避免继续在同一个 score 上做微调，避免把 source-row 或 narrow-support success 包装成 functional causality。

## 3. 当前真正 blocker

当前 blocker 不是：

```text
base 不稳；
snapshot attach 污染；
carrier silent；
S7 公式只差一点；
某个数据集需要单独调；
paired replay 已经证明失败。
```

当前 blocker 是：

$$
\boxed{
\text{fresh expanded event population 中，carrier 产生的 events 不是稳定 safe-good support。}
}
$$

具体分解为三个问题。

### 3.1 Safety support 不足

Fresh v2 的总体 bad-event rate 超过 gate：

$$
BadEventRate=0.0768229.
$$

这说明 active carrier 在扩大 seed / stratum / attach support 后，任务风险开始变成主 blocker。v9.2.40 / v9.2.41 的 bad-event 低，不能外推到更宽 fresh v2。

### 3.2 Oracle support 失效

如果 oracle 都只有 precision $0.375$ 且 bad-event $0.625$，说明在当前 fresh v2 event set 中，即使 posthoc 选择，也没有足够安全、能赢 controls 的 events。此时继续修 legal controller 没有意义。

### 3.3 Controller shape 与 accept gate 脱节

Best legal controller 的 heldout corr 是 $0.391692$，按相关性看似有 signal，但 accept precision 只有 $0.544118$，bad-event 高达 $0.455882$。这说明“score 形状相关”不等于“可用于 accept / abstain 的安全 controller”。必须把 ranking、threshold、risk、family support 分开评估。

## 4. 进展是否缓慢

进展确实慢，但 v9.2.43 是一次重要的负结果，不是无意义失败。它把下一步决策从 “继续校准 legal score” 改成 “carrier/event support reset”。  
慢的主要原因是过去每次只解决一个前置 gate。v9.2.44 必须改成并行 falsification：

```text
同时测：
  carrier family reset
  risk-gated event support
  oracle support
  legal controller
  leave-out
  paired replay scout

而不是：
  本轮只修 carrier
  下轮只测 oracle
  再下轮只测 controller
```

但并行不等于放松标准。所有 official route 仍然必须满足预注册 gate。

## 5. 是否在正确道路上

是，但当前分叉非常明确。

过去路线已经把 infrastructure 逐步闭合：

```text
base repaired and robust
snapshot attach implemented
inactive equivalence pass
no-event replay pass
carrier active
local S7 value score discovered
fresh v2 expanded
```

现在的问题已经逼近 functional update 的核心：

```text
是否存在足够多 legal, safe, control-resistant good events？
```

这个问题正是 functional update 是否有独特价值的核心。如果 good events 存在但 legal controller 找不到，那么继续做 controller representation。如果 good events 连 oracle 都找不到，那么 carrier mechanism 必须 reset。

v9.2.43 的 fresh v2 结果更接近第二种情况，所以 v9.2.44 必须优先恢复 carrier/event support，而不是继续做 score 微调。

---

# Part II. v9.2.44 总体目标

v9.2.44 的总体目标是：

$$
\boxed{
\text{在不使用 dataset-specific tuning 的前提下，恢复并验证 safe good-event support，然后再打开 legal controller 与 paired replay。}
}
$$

这个目标分为六层。

## 1. Fresh v2 failure attribution success

必须解释 v9.2.43 的 fresh safety / oracle collapse。  
至少归因到以下机制之一：

```text
A1-unsafe_attach_family:
  某些 attach candidate 明显抬高 bad-event。

A2-unsafe_signal_stratum:
  某些 signal strata 产生大量 unsafe movement。

A3-horizon_risk_explosion:
  长 horizon 或短 horizon 导致 risk 激增。

A4-rolewise_branch_misaligned:
  role-wise carrier 能动，但 stack/head contribution 与 task safety/value 不一致。

A5-control_gap_absent:
  RealFunctional 有 task movement，但 control gap 大多不为正。

A6-risk_score_missing:
  good-event ranking 有 signal，但没有包含 risk predictor，因此 accepted events high bad-event。

A7-family_support_thin:
  good events 只存在于少数 family，fresh v2 扩展后被 unsafe families 稀释。

A8-carrier_mechanism_true_fail:
  所有 family / strata / horizon 下 oracle 都低，说明 current carrier 缺少 robust control-resistant value。
```

成功标准：

```text
>= 90% bad events assigned to primary failure mode
>= 90% oracle-fail events assigned to primary failure mode
dataset_name not used in attribution rule
```

## 2. Safe good-event support recovery success

要在 fresh rows 上先恢复 oracle / support，而不是直接调 controller。

定义：

$$
Y_{\text{safe-good}}
=
\mathbb{1}
[
RealFunctional \text{ beats AdamWParallel and bestLR}
]
\cdot
\mathbb{1}
[
BadEvent=0
].
$$

Support recovery pass：

$$
OraclePrecision_{\text{safe-good}}\geq0.75,
$$

$$
OracleCoverage_{\text{safe-good}}\in[0.03,0.15],
$$

$$
OracleBadEventRate\leq0.05.
$$

并且 multi-family：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

如果 oracle support 不能恢复，legal controller 不允许 official 打开。

## 3. Carrier mechanism reset success

至少一个 carrier / event-family candidate 必须同时满足：

```text
inactive equivalence pass = 1
no-event preservation pass = 1
carrier active = 1
safe-good oracle support pass = 1
system pass = 1
```

Carrier active：

$$
r_{z,\text{tail}}\geq0.10,
$$

$$
r_{\perp,\text{tail}}\geq0.10.
$$

System：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

## 4. Legal controller success

只有 support recovery pass 后才评估 official legal controller。  
Heldout value gate：

$$
AUC(Y_{\text{safe-good}})\geq0.70
$$

or:

$$
Corr(S,V_{\text{safe-grounded}})\geq0.35.
$$

Accept gate：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{heldout}}\leq0.05.
$$

Legality：

```text
dataset_name_used = 0
posthoc_used_at_commit = 0
validation_used = 0
test_used = 0
```

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
```

---

# Part III. 核心假设

## H1：v9.2.43 的失败来自 unsafe event-support expansion，而不是 S7 本身彻底无效

v9.2.42 frozen S7 在 fresh rows 上仍有：

$$
AUC=0.752942,
$$

$$
Precision=0.906977,
$$

$$
BadEvent=0.
$$

v9.2.43 的 fresh v2 更大，但 bad-event 和 oracle 同时恶化。H1 认为问题来自扩展后的 event support，而不是 S7 排序能力完全不存在。

H1 成立标准：

```text
存在至少一个 dataset-agnostic risk/event-family filter，
使 oracle safe-good support pass，
且 S7 / S7-derived score 在该 support 上 AUC >= 0.70 或 precision >= 0.75。
```

H1 失败标准：

```text
所有 pre-registered support filters 下 oracle 都 fail。
```

## H2：risk gating 必须在 controller 前面，而不是附属 metric

v9.2.43 best legal controller corr 高但 accept bad-event 高。说明 ranking score 不能替代 risk filter。  
H2 认为 accept rule 应拆成：

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
RiskSafe(e)=\mathbb{1}[R(e)\leq \rho],
$$

$$
ValuePositive(e)=\mathbb{1}[S_{\text{value}}(e)\geq\tau],
$$

$$
ControlResistant(e)=\mathbb{1}[S_{\text{gap}}(e)>0].
$$

H2 成立标准：

risk-first controller 相比 score-only controller：

$$
BadEventRate_{\text{risk-first}}
<
BadEventRate_{\text{score-only}},
$$

且 precision / coverage gate 不下降到不可用。

## H3：carrier reset 应围绕 role-wise + control-gap，不应回到 arbitrary basis sweep

v9.2.43 不要求回到 “随便换 basis”。当前 base 和 attach 已闭合，问题是 carrier mechanism。  
Carrier reset 应限于：

```text
role-wise FT7
control-gap bounded channel
risk-bounded tail channel
family-balanced carrier
uncertainty-LCB carrier
```

而不是按 dataset 或随意 basis 扩展。

H3 成立标准：

至少一个 role/control/risk carrier 恢复 oracle support，并在 legal controller + leave-out 中优于 current carrier。

## H4：如果 oracle support 恢复但 legal controller fail，问题转为 legal feature representation

如果：

```text
oracle safe-good support pass = 1
legal controller pass = 0
```

说明 carrier 中有 good events，legal controller 识别不到。下一步应设计 richer train-stream features，而不是继续改 carrier。

## H5：如果 oracle support 仍 fail，current functional mechanism 必须 reset

如果：

```text
all carriers / event filters oracle fail
```

说明 current snapshot late-attach carrier family 不能产生足够 robust control-resistant value。下一步必须回到 v8-FT7 role-wise mechanism 重新设计 strict PureKAN carrier，不能继续 controller calibration。

## H6：dataset tuning 仍然禁止

即使某个 dataset slice 上 risk-safe support pass，也不能形成 official route。Official controller 必须通过 LDO/LSO。

---

# Part IV. 并行执行设计

v9.2.44 runner 必须并行执行以下 lanes：

```text
Lane A:
  v9.2.43 boundary reproduction and failure attribution

Lane B:
  bad-event / oracle-collapse decomposition by attach, stratum, horizon, family

Lane C:
  risk-first event support filters

Lane D:
  carrier reset matrix

Lane E:
  oracle safe-good support audit

Lane F:
  legal controller calibration only on support survivors

Lane G:
  leave-dataset-out / leave-stratum-out

Lane H:
  paired replay scout and official paired replay

Lane I:
  short-run scout if paired replay passes
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

# Part V. Candidate and controller design

## 1. Base

Official base remains：

```text
R2-LQ-fanin-output-scale-confirmed
```

## 2. Attach / carrier candidates

### A0：CurrentCarrier-v9243

Reference only.  
Used for failure attribution.

### A1：RiskBoundedTailCarrier

Accepts tail events only if pre-event risk is below threshold：

$$
R(e)
=
w_1 CEp99
+
w_2 WrongConfidenceP95
-
w_3 MarginP10
+
w_4 Uncertainty.
$$

No dataset name.

### A2：ControlGapBoundedCarrier

Carrier active only if train-stream control-gap lower bound is positive：

$$
LCB(Gain_{\text{Real}})
-
UCB(\max(Gain_{\text{AdamWParallel}},Gain_{\text{bestLR}}))
>0.
$$

### A3：RoleWiseFT7ResetCarrier

Strict edge-owned role-wise late attach：

$$
\phi_{ij}(h)
=
\phi^{base}_{ij}(h;\theta_T^\star)
+
\lambda(e)
[
\alpha_s(e)\phi_{ij}^{F,s}(h;\theta_{F,s})
+
\alpha_h(e)\phi_{ij}^{F,h}(h;\theta_{F,h})
].
$$

Role weights use only：

```text
CEp99
MarginP10
Curvature
BranchRatio
EffectiveDerivative
ControlGap
Uncertainty
```

### A4：FamilyBalancedCarrier

Restricts event selection so that no single event family dominates：

$$
family(e)=(stratum,horizon,risk\_bucket,role\_bucket,attach).
$$

Promotion requires：

```text
max_family_share <= 0.60
accepted_family_count >= 4
```

### A5：UncertaintyLCBCarrier

Uses lower-confidence bound：

$$
S_{\text{LCB}}
=
\mu_{\text{value}}
-
\kappa\sigma_{\text{value}}
-
R_{\text{risk}}.
$$

### A6：HybridRoleControlRiskCarrier

Accept if all three hold：

$$
S_{\text{role-gap}}>0,
$$

$$
S_{\text{control-gap}}>0,
$$

$$
R_{\text{risk}}\leq\rho.
$$

---

## 3. Controllers

### C0：Current v9.2.43 Best Legal Controller

Reference only.

### C1：RiskFirstS7

$$
Accept(e)=RiskSafe(e)\land S7(e)\geq\tau.
$$

### C2：RiskFirstControlGap

$$
Accept(e)=RiskSafe(e)\land S_{\text{gap}}(e)>0.
$$

### C3：RiskFirstRoleGap

$$
Accept(e)=RiskSafe(e)\land S_{\text{role-gap}}(e)>0.
$$

### C4：TwoStageRiskFamily

$$
Accept(e)
=
Core(e)
\lor
[
Border(e)
\land
ReliableFamily(e)
\land
RiskSafe(e)
].
$$

### C5：FamilyBalancedLCB

$$
Accept(e)=
S_{\text{LCB}}(e)>0
\land FamilyBalance(e).
$$

### C6：Oracle

Posthoc diagnostic only.  
Never official.

---

# Part VI. 实验阶段

## P0：v9.2.43 boundary reproduction

### 目标

复现 v9.2.43 boundary，确认本轮不是解析错误。

### 必须记录

```text
route
source_route_v9242
s7_coverage_cliff_mode
fresh_rows
fresh_real_events
strata_count
attach_count
fresh_bad_event_rate
fresh_v2_pass
best_legal_controller
heldout_corr
heldout_precision
heldout_bad_event
oracle_precision
oracle_coverage
oracle_bad_event
oracle_pass
paired_replay_opened
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R7-OracleLowCarrierMechanismReset
fresh_v2_bad_event_rate > 0.05
oracle pass = 0
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_gate_ladder_from_v9242_to_v9243.svg
p0_oracle_collapse_recap.svg
```

---

## P1：Fresh v2 failure decomposition

### 目标

解释 fresh v2 为什么 bad-event 与 oracle 同时恶化。

### 必须记录

```text
row_id
dataset
seed
horizon
signal_stratum
attach_candidate
event_family
branch
real_gain
adamwparallel_gain
bestlr_gain
control_gap
bad_event
task_safe
CEp99_delta
margin_delta
ECE_delta
NLL_delta
curvature_delta
r_z_tail
r_perp_tail
cos_real_adamw
cos_real_bestlr
risk_score
uncertainty
role_stack_score
role_head_score
failure_mode
```

### 判断标准

P1 pass：

```text
bad-event attribution fraction >= 0.90
oracle-fail attribution fraction >= 0.90
at least attach/stratum/horizon/family decomposition measured
dataset_name not used in route
```

### 可视化

```text
p1_bad_event_by_attach.svg
p1_bad_event_by_stratum_horizon.svg
p1_oracle_fail_by_family.svg
p1_control_gap_vs_bad_event.svg
p1_risk_score_distribution.svg
p1_failure_mode_sankey.svg
```

---

## P2：Risk-first support filters

### 目标

在不改 carrier 的情况下，先判断是否能通过 dataset-agnostic risk/support filter 恢复 oracle safe-good support。

### 必须记录

```text
filter_id
risk_features
thresholds
calibration_split
heldout_split
oracle_precision
oracle_coverage
oracle_bad_event
legal_precision
legal_coverage
legal_bad_event
accepted_strata_count
accepted_family_count
max_family_share
dataset_name_used
```

### 判断标准

Risk filter diagnostic pass：

$$
OraclePrecision\geq0.75,
$$

$$
OracleCoverage\in[0.03,0.15],
$$

$$
OracleBadEventRate\leq0.05.
$$

Official support pass also requires：

```text
accepted_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
dataset_name_used = 0
```

### 可视化

```text
p2_risk_filter_precision_coverage.svg
p2_oracle_support_recovery.svg
p2_family_balance.svg
```

---

## P3：Carrier reset matrix

### 目标

并行测试 A1-A6，判断是否有 carrier 能恢复 safe-good support。

### 设置

```text
base = R2 repaired base checkpoint
carriers = A0,A1,A2,A3,A4,A5,A6
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7
horizons = 20,80,240,640
signal_strata = S1-S8
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random
```

### 必须记录

```text
carrier_id
row_id
dataset
seed
horizon
signal_stratum
event_family
inactive_equivalence_pass
no_event_preservation_pass
carrier_active
r_z_tail
r_perp_tail
real_gain
adamwparallel_gain
bestlr_gain
control_gap
bad_event
task_safe
oracle_safe_good
step_q90
memory_ratio
```

### 判断标准

Carrier support pass：

$$
OraclePrecision_{\text{safe-good}}\geq0.75,
$$

$$
OracleCoverage_{\text{safe-good}}\in[0.03,0.15],
$$

$$
OracleBadEventRate\leq0.05.
$$

Carrier system pass：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

Carrier active：

$$
r_{z,\text{tail}}\geq0.10,
$$

$$
r_{\perp,\text{tail}}\geq0.10.
$$

### 可视化

```text
p3_carrier_oracle_support_matrix.svg
p3_carrier_task_safety_pareto.svg
p3_carrier_control_gap_distribution.svg
p3_carrier_system_pareto.svg
```

---

## P4：Legal controller after support recovery

### 目标

只有 P2/P3 support survivor 才进入 legal controller calibration。

### 必须记录

```text
controller_id
carrier_id
calibration_split
heldout_split
thresholds
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

Controller pass：

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
p4_controller_matrix_auc_corr.svg
p4_controller_precision_coverage_bad.svg
p4_controller_family_coverage.svg
p4_legal_vs_oracle_gap.svg
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
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, InvertedRole, ValueScoreShuffled, FunctionalChannelShuffled, TailMaskShuffled, RoleScoreShuffled
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

如果 P6 pass，验证局部 paired replay 优势能否在连续训练中保持。

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
contract_audit_v9244.csv
p0_v9243_boundary_reproduction.csv
p1_fresh_v2_failure_decomposition.csv
p2_risk_first_support_filters.csv
p3_carrier_reset_matrix.csv
p4_legal_controller_after_support_recovery.csv
p5_leave_dataset_and_stratum_out.csv
p6_official_paired_replay.csv
p7_short_run_functional_validation.csv
p8_full_10seed_functional_validation.csv
p9_robustness_external_ready.csv
fresh_v2_failure_trace_v9244.csv
risk_filter_trace_v9244.csv
carrier_reset_trace_v9244.csv
oracle_support_trace_v9244.csv
legal_controller_trace_v9244.csv
leaveout_trace_v9244.csv
paired_replay_branch_trace_v9244.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9243_boundary_unstable
F3_dataset_tuning_detected
F4_fresh_v2_failure_unattributed
F5_risk_filter_no_support_recovery
F6_all_carriers_oracle_support_fail
F7_carrier_system_fail
F8_carrier_silent
F9_legal_controller_fail_after_oracle_support
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

# Part VIII. Route decision

```text
R1-FreshV2FailureAttributed:
  v9.2.43 fresh safety / oracle collapse is explained.

R2-RiskFilterSupportRecovered:
  risk-first event support restores oracle safe-good support.

R3-CarrierResetSupportRecovered:
  at least one reset carrier restores safe-good support.

R4-LegalControllerPassAfterSupport:
  legal controller passes after support recovery.

R5-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R6-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R7-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R8-OracleHighLegalFeatureGap:
  safe-good events exist, but legal controller cannot identify them.

R9-OracleLowCarrierMechanismReset:
  all carrier/event supports fail oracle; mechanism must reset deeper.

R10-BaseAttachCarrierValidButUnsafe:
  base/attach/carrier remain valid, but risk cannot be controlled.

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
v9243_boundary_pass
dataset_tuning_detected
fresh_v2_failure_mode
bad_event_attribution_pass
oracle_collapse_attribution_pass
best_risk_filter
risk_filter_support_pass
best_carrier_id
carrier_support_pass
oracle_precision
oracle_coverage
oracle_bad_event
best_controller_id
legal_controller_pass
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
success_v9244_strict_purekan_functional
success_v9244_full_functional
success_v9244_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 fresh v2 failure decomposition
  P2 risk-first support filters
  P3 carrier reset matrix

Batch 2:
  P4 legal controller on support survivors
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
P3/P4/P6 diagnostic rows may be measured before all gates finish.
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

# Part X. 停止条件

## Minimum diagnostic success

```text
v9.2.43 boundary reproduced
fresh v2 failure attributed
risk support filters measured
carrier reset matrix measured
oracle/legal gap measured
no fake/proxy/offload/loss/teacher violation
```

## Support recovery success

```text
Minimum diagnostic success
+
at least one risk filter or carrier restores oracle safe-good support
+
multi-stratum / multi-family coverage pass
```

## Legal controller success

```text
Support recovery success
+
at least one legal controller passes heldout observability
+
precision / coverage / bad-event gate pass
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
1. v9.2.43 boundary cannot be reproduced；
2. fresh v2 failure cannot be attributed；
3. all risk filters fail to recover oracle support；
4. all carrier reset candidates fail oracle support；
5. carrier support recovers but all legal controllers fail；
6. oracle high but legal features cannot recover good events；
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

## Case A：Risk filter restores oracle and legal controller passes

可以声明：

```text
v9.2.43 failed because fresh v2 expanded unsafe support; risk-first support recovery fixed it.
```

但不能声明 strict success unless LDO/LSO and paired replay pass.

## Case B：Carrier reset restores oracle support

可以声明：

```text
current carrier family was unsafe; reset carrier restored safe good-event support.
```

但仍需 legal controller and paired replay.

## Case C：Oracle support recovers but legal controller fails

必须声明：

```text
Good events exist, but current legal train-stream features cannot identify them.
```

下一步设计 richer legal features，不修 base.

## Case D：All oracle support fails

必须声明：

```text
Current carrier/event family cannot produce enough safe control-resistant good events.
```

下一步做 deeper role-wise FT7 carrier reset，而不是 controller calibration.

## Case E：LDO/LSO fail

必须声明：

```text
Controller or support recovery is not dataset-agnostic / stratum-agnostic enough.
```

不能用 dataset-specific tuning 写成功.

## Case F：Paired replay passes

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full validation.

---

# Part XII. 最终建议

v9.2.44 的一句话策略是：

$$
\boxed{
\text{不要继续在 unsafe fresh support 上校准 score；先恢复 safe good-event support，再谈 legal controller 与 paired replay。}
}
$$

当前最关键的问题不是：

```text
base 是否稳定；
attach 是否污染；
carrier 是否 silent；
S7 coverage 是否差 0.000139；
Fashion 怎么调；
KMNIST 怎么调；
MNIST 是否 abstain；
是否再换一个普通 basis。
```

而是：

```text
1. v9.2.43 fresh v2 为什么 bad-event rate 超 gate？
2. 为什么 oracle upper bound 从 v9.2.42 pass 变成 fresh v2 fail？
3. 哪些 attach / stratum / horizon / family 产生 unsafe events？
4. risk-first support filter 是否能恢复 safe good-event oracle？
5. 如果 risk filter 不够，role-wise / control-gap / LCB carrier reset 是否能恢复 support？
6. support 恢复后，legal controller 是否能在 heldout / LDO / LSO 下找到 good events？
7. official paired replay 是否能打过 AdamWParallel / bestLR？
```

v9.2.44 的结果将给出清晰分叉：

```text
if risk/carrier restores oracle support and legal controller passes:
  enter leave-out and paired replay closure.

if oracle support recovers but legal controller fails:
  controller feature representation is blocker.

if oracle support never recovers:
  current carrier mechanism is the blocker; deeper FT7-style carrier reset required.

if paired replay passes:
  strict PureKAN functional obtains local causal evidence under strong controls.
```
