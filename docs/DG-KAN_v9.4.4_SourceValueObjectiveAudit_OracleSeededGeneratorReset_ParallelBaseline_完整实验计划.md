# DG-KAN v9.4.4 Source Value Objective Audit / Oracle-Seeded Generator Reset / Parallel Baseline 实验计划

> 本计划基于 v9.4.3 `Source Frontier Recovery / Direct Generator Effect Certificate / Parallel Validation` 的真实执行结果制定。  
> v9.4.4 的目标不是继续调 AP0j、certificate threshold 或 legal selector 单特征，而是回答一个更根本的问题：
>
> $$
> \boxed{
> \text{当前失败到底是 source objective 定义错、AP0 source frontier 真不存在、legal selector 看不见，还是 generator 破坏价值？}
> }
> $$
>
> 本轮继续遵守：不按 dataset 调参，不用 teacher / distillation / loss modification，不把 oracle / diagnostic / Base-Acc Sentinel 写成 official controller success。  
> 数据集可以用于诊断不同分布上的失败模式，但不能用 dataset_name 作为 controller、selector、generator 或 threshold 的分支条件。

---

# 0. 当前结论摘要

v9.4.3 的真实结果是：

```text
route = R2-FullAP0OracleSourceFrontierAbsent
base_candidate = LQ-t2-h256
success_v9430_strict_purekan_functional = False
success_v9430_full_functional = False
success_v9430_external_ready = False
```

但我不建议把这个 route 简单理解成“AP0 全体已经没有任何好 action”。更准确的独立判断是：

$$
\boxed{
\text{在 v9.4.3 的 source-frontier protocol 下，AP0 oracle panel 只有 borderline signal，legal selector 与 direct generator 都没有形成 h20-positive / h240-safe frontier。}
}
$$

v9.4.3 的关键事实如下：

```text
P0:
  v9.4.2 source outcome materializer 没有回退；
  source outcome rows = 7020 / 7020。

P1:
  best oracle panel = PANEL-ORC64；
  h20 weak CP = 0.578125；
  h20 V_ctrl LCB = -0.020407727203802614；
  h240 long-risk = 0.0；
  diagnostic oracle pass = 0。

  PANEL-ORC128:
  h20 weak CP = 0.6875；
  h20 V_ctrl LCB = -0.041945819477686164；
  h240 long-risk = 0.2109375；
  仍不能 official。

P1 legal selector:
  best legal panel = PANEL-LGL64-AdamWConflictLow；
  h20 weak CP = 0.171875；
  h20 V_ctrl LCB = -0.3014420568912851；
  h240 long-risk = 0.890625；
  legal selector pass = 0。

P2:
  PANEL-ORC64 AP0 same-panel baseline 完整；
  expected / actual rows = 1152 / 1152；
  branch / horizon completion = 1.0 / 1.0；
  source survivor pass = 0。

P3:
  AP0b-AP0f transform damage 仍明显；
  paired horizon = 780；
  Damage median = -0.1004650485701859；
  source-positive lost rate = 0.9215686274509803。

P4:
  AP0g-AP0k direct generator 已真实 materialize；
  generated actions = 60；
  branch-horizon rows = 1620 / 1620；
  best primitive = AP0j-LowRankEdgeSafeSource；
  h20 weak CP = 0.25；
  h20 V_ctrl LCB = -0.20831185402129682；
  h240 long-risk = 0.8333333333333334；
  direct generator pass = 0。

P5:
  effect certificate 未过；
  AUC weak CP = 0.5069044879171462；
  AUC strong CP = 0.6217142857142857；
  AUC long-risk = 0.5581680996666082；
  monotone sign pass = 0。

P8 Base-Acc Sentinel:
  datasets = MNIST, Fashion-MNIST, KMNIST；
  seeds = 0,1,2,3,4；
  row count = 45；
  mean test acc LQ = 0.64921875；
  mean test acc MatchedMLP = 0.5609375；
  LQ - MLP = 0.08828125；
  LQ - QuadraticFeatureMLP = 0.23776041666666664；
  LQ catastrophic fail = 0；
  AdamWStrongLRGridMLP diagnostic materialized = 0。
```

这说明 v9.4.3 有进展，但进展主要是 **失败边界定位**，不是 functional success。它完成了 multi-panel source frontier、direct generator smoke、certificate audit、5-seed Base-Acc Sentinel；但没有 controller、selected runtime、LDO/LSO、paired replay 或 short/full training success。

---

# 1. 这轮为什么感觉慢

你觉得慢是合理的。当前连续多轮的问题不是“完全没做事”，而是推进方式过于串行：

```text
v9.4.0:
  发现 first-N source panel 是 biased convenience slice；
  legal selector 看不见 full AP0 oracle frontier；
  AP0b-AP0f generator 未 materialize。

v9.4.1:
  修 stratified source panel；
  materialize AP0b-AP0f payload/certificate；
  但 source outcome rows = 0。

v9.4.2:
  修 source outcome rows 到 7020 / 7020；
  发现 stratified source panel 和 AP0b-AP0f generated source 都 value-poor。

v9.4.3:
  跑 multi-panel AP0 oracle frontier 和 AP0g-AP0k direct generator；
  发现 oracle panel 只有 borderline signal；
  legal selector、direct generator、certificate 都不过。
```

这类进展在科学上有价值，因为它没有把 fake/proxy/diagnostic 写成 success；但执行效率上确实太慢。v9.4.4 必须采用并行矩阵，而不是继续“一轮发现一个 blocker”。

本轮计划的原则是：

```text
1. 同时验证 full AP0 source frontier、source objective、legal selector、generator、certificate、Base-Acc、runtime。
2. 每个路径先做 single-action / mini-panel preflight，再做 official matrix。
3. 不再等完整大 runner 跑完才发现某个表为空或某个 gate 定义不匹配。
4. 不继续 AP0j threshold / certificate threshold / single feature topK 的小修小补。
```

---

# 2. v9.4.3 的本质问题

## 2.1 weak CP 和 V_ctrl LCB 出现了结构性张力

v9.4.3 最重要的现象不是 legal selector 失败，而是 oracle panel 的标签与 value 不一致：

```text
PANEL-ORC64:
  h20 weak CP = 0.578125
  h20 V_ctrl LCB = -0.020408
  h240 long-risk = 0.0

PANEL-ORC128:
  h20 weak CP = 0.6875
  h20 V_ctrl LCB = -0.041946
  h240 long-risk = 0.210938
```

这意味着：

$$
\boxed{
\text{weak CP 命中率接近或超过阈值，不代表 } V_{ctrl} \text{ 的 lower bound 为正。}
}
$$

所以 v9.4.4 必须首先审计：

```text
1. weak_CP label 是否过于离散，无法反映 value distribution；
2. V_ctrl LCB 是否被少数 outliers 拉负；
3. h20 value 与 h240 risk 是否存在 tradeoff；
4. 当前 source survivor gate 是否把“接近可用”的 frontier 判成 absent；
5. CP label、V_ctrl、horizon robust 三者是否应拆成不同对象。
```

不能直接把 `R2-FullAP0OracleSourceFrontierAbsent` 当成 AP0 死刑。

## 2.2 legal selector 与 oracle 差距仍巨大

当前 best legal panel：

```text
PANEL-LGL64-AdamWConflictLow:
  h20 weak CP = 0.171875
  h20 V_ctrl LCB = -0.301442
  h240 long-risk = 0.890625
```

与 oracle ORC64 对比：

$$
\Delta weakCP_{h20}=0.578125-0.171875=0.40625.
$$

与 ORC128 对比：

$$
\Delta weakCP_{h20}=0.6875-0.171875=0.515625.
$$

也就是说，legal selector 不是差一点，而是没有看到 oracle source frontier 的主要结构。

## 2.3 direct generator 不是 value-producing generator

AP0g-AP0k 的 direct generator 已经真实 materialize，说明 implementation blocker 推进了。但 best AP0j 结果是：

```text
h20 weak CP = 0.25
h20 V_ctrl LCB = -0.208312
h240 long-risk = 0.833333
```

它不但 h20 value 负，h240 risk 也高。这说明当前 direct generator 仍像 heuristic action builder，不像真正求解：

$$
\max_{\Delta\theta}
V_{h20}(\Delta\theta)
-
\lambda_r Risk_{h240}(\Delta\theta)
-
\lambda_c Cost(\Delta\theta)
$$

的 constrained update generator。

## 2.4 certificate 仍不是 effect certificate

P5 certificate AUC：

```text
AUC weak CP = 0.506904
AUC strong CP = 0.621714
AUC long-risk = 0.558168
monotone sign pass = 0
```

这说明 certificate 目前仍只是 construction descriptor，而不是 effect-valid statistic。它没有稳定预测：

```text
h20 value；
h240 long-risk；
weak / strong CP；
horizon robust survivor。
```

## 2.5 Base-Acc Sentinel 是健康信号，不是 functional success

v9.4.3 有数据集训练，但只是 Base-Acc Sentinel：

```text
LQ-t2-h256 mean test acc = 0.64921875
MatchedMLP mean test acc = 0.5609375
LQ - MLP = +0.08828125
```

这说明 LQ base 在固定 sentinel 配置下没有 catastrophic fail，并且比 MatchedMLP 高约 8.83 个百分点。但这不是 official functional training，也不是强 MLP baseline comparison，因为：

```text
1. system controller 没有打开；
2. short/full functional training 没有打开；
3. AdamWStrongLRGridMLP diagnostic 未 materialize；
4. Base-Acc Sentinel 没有用于 selector/controller；
5. 绝对 acc 偏低，不是打榜结果；
6. 当前项目目标不是在这些数据集上刷榜。
```

因此 acc 结论只能写成：

$$
\boxed{
\text{LQ base health positive, functional controller evidence absent.}
}
$$

---

# 3. v9.4.4 总体目标

v9.4.4 的总体目标是：

$$
\boxed{
\text{用并行实验判定 source failure 的根因，并尝试从 heuristic generator 转向 objective-solving generator。}
}
$$

本轮不追求直接 full success。最低有效推进是把当前模糊的 blocker 拆成可行动的根因：

```text
A. Source objective mismatch:
   weak CP 与 V_ctrl LCB 冲突，gate 定义或 value 估计需要重构。

B. Full AP0 source frontier genuinely absent:
   即使 exhaustive oracle search，也没有 h20-positive / h240-safe source。

C. Legal selector opacity:
   oracle source 存在，但 commit-time legal features 找不到。

D. Generator destruction:
   oracle-selected AP0 source 本身可用，但 generator transform 后价值被破坏。

E. Direct generator objective failure:
   generator 能产 payload，但没有解对 source value objective。

F. Certificate failure:
   certificate 与实际 effect 无单调关系。

G. Runtime boundary:
   只有在 source/controller survivor 出现后，才测 selected runtime；否则只做 microbench。
```

---

# 4. 成功定义

## 4.1 Source survivor gate

一个 source action set $A$ 在 v9.4.4 中成为 source survivor，需要同时满足：

$$
WeakCP_{h20}(A)\ge 0.60,
$$

$$
LCB(V_{ctrl,h20}(A))>0,
$$

$$
LongRisk_{h240}(A)\le 0.10,
$$

$$
SupportBalance(A)=1.
$$

其中：

$$
V_{ctrl}(a,h)
=
V_{RealFunctional}(a,h)
-
\max_{b\in Controls}V_b(a,h).
$$

Controls 至少包括：

```text
AdamWParallel
bestLR
NoOp
Random
ShuffledPayload
```

如果 $WeakCP_{h20}$ 过线但 $LCB(V_{ctrl,h20})\le0$，则不能 official；必须进入 source objective mismatch audit。

## 4.2 Legal selector gate

Legal selector 不允许使用 outcome、dataset_name、future step、validation/test metric。它必须满足：

$$
WeakCP_{h20}^{heldout}\ge0.60,
$$

$$
LCB(V_{ctrl,h20}^{heldout})>0,
$$

$$
LongRisk_{h240}^{heldout}\le0.10,
$$

$$
Coverage_{action}\in[0.01,0.15].
$$

若只是 oracle selector 过线，不能 official。

## 4.3 Generator survivor gate

Direct generator 或 transform generator 必须在 same-panel source outcomes 中满足：

```text
generated action apply replay pass = 1
branch/horizon completion = 1
h20 weak CP >= 0.60
h20 V_ctrl LCB > 0
h240 long-risk <= 0.10
source-positive lost rate <= 0.20 for transform generators
```

## 4.4 Certificate effect-valid gate

Certificate $Cert(a)$ 必须满足：

$$
AUC(Cert, WeakCP_{h20})\ge0.75,
$$

$$
AUC(Cert, LongRisk_{h240})\ge0.75,
$$

并且 sign 必须单调：

```text
higher value_cert -> higher h20 V_ctrl
higher risk_cert -> higher h240 long-risk
higher support_cert -> lower calibration drift
```

还必须满足：

```text
P(WeakCP | cert pass) >= 2 * P(WeakCP | cert fail)
P(LongRisk | cert pass) <= 0.5 * P(LongRisk | cert fail)
```

## 4.5 Base-Acc Sentinel gate

Base-Acc Sentinel 不是 functional success gate。它只用于 health monitoring：

```text
LQ catastrophic fail = 0
LQ mean test acc not worse than MatchedMLP by more than 0.01
train/val/test curves finite
no NaN / Inf
manual training contract pass
```

Strong baseline diagnostic 需要 materialize：

```text
AdamWStrongLRGridMLP diagnostic materialized = 1
QuadraticFeatureMLP materialized = 1
same seeds and fixed grid
dataset_specific_tuning = 0
```

---

# 5. 核心假设

## H1：v9.4.3 的 route 可能是 borderline oracle fail，而不是 complete AP0 death

PANEL-ORC64 的 h20 weak CP 是 `0.578125`，只差 `0.021875` 到 `0.60`；h20 V_ctrl LCB 是 `-0.020408`，也只是略负。H1 认为需要做 exhaustive oracle frontier 和 value distribution audit，不能仅凭一个 panel route 判死刑。

H1 成立标准：

```text
exhaustive AP0 source frontier 找到至少一个 K 或 Pareto set:
  h20 weak CP >= 0.60
  h20 V_ctrl LCB > 0
  h240 long-risk <= 0.10
```

H1 失败标准：

```text
all K in {16,32,64,128,256,512}
and all Pareto oracle scores
均无法同时满足 source survivor gate。
```

## H2：weak CP 与 V_ctrl LCB 的冲突是当前 source objective 的核心问题

H2 成立标准：

```text
存在 panel:
  weak_CP_h20 >= 0.60
  but V_ctrl_lcb_h20 <= 0

且 V_ctrl distribution 显示：
  mean/median 与 LCB 或 tail outliers 分离；
  或 weak CP label 与 actual V_ctrl correlation < 0.35。
```

H2 失败标准：

```text
weak CP、V_ctrl LCB、long-risk 三者高度一致；
不存在 objective mismatch。
```

## H3：legal source selector 失败来自 feature sigma-algebra 不足，不是 K 太小

H3 成立标准：

```text
legal selectors across K=16..512 均无法达到:
  h20 weak CP >= 0.45
  h20 V_ctrl LCB > 0
  h240 long-risk <= 0.20

best legal AUC weak CP < 0.70
best legal AUC long-risk < 0.70
```

H3 失败标准：

```text
某个 legal selector 在不同 K 和 heldout split 上稳定过 weak gate。
```

## H4：AP0b-AP0f transform generator 失败主要来自 value destruction

H4 成立标准：

```text
oracle-selected source panel 在 AP0 baseline 下有 source survivor；
但生成后:
  source-positive lost rate > 0.50
  Damage median < -0.05
```

H4 失败标准：

```text
source 本身不好，不能归因于 generator；
或 generator 能 preserve source positives。
```

## H5：AP0g-AP0k direct generator 没有求解正确 objective

H5 成立标准：

```text
direct generator h20 V_ctrl LCB < 0
and h240 long-risk > 0.20
and certificate cannot predict failure。
```

H5 失败标准：

```text
某个 direct generator 在 pre-registered panel 上满足 source survivor gate。
```

## H6：effect-valid certificate 需要绑定 action effect，而不是 construction heuristic

H6 成立标准：

```text
current certificate AUC weak CP <= 0.60
or monotone sign pass = 0
or P_weak_CP_given_cert_pass ≈ P_weak_CP_given_cert_fail。
```

H6 失败标准：

```text
redesigned certificate achieves:
  AUC weak CP >= 0.75
  AUC long-risk >= 0.75
  monotone sign pass = 1。
```

## H7：Base LQ 没崩，但强 MLP 对照仍不完整

H7 成立标准：

```text
Base-Acc Sentinel LQ >= MatchedMLP - 0.01
but AdamWStrongLRGridMLP diagnostic materialized = 0。
```

H7 失败标准：

```text
StrongLRGridMLP materialized 后显著超过 LQ；
或 LQ catastrophic fail。
```

## H8：v9.4.4 必须并行化，否则会继续一轮一个 blocker

H8 成立标准：

```text
single-action preflight
mini-panel preflight
official matrix
base acc sentinel
runtime microbench
all run as independent lanes with early stop manifests。
```

H8 失败标准：

```text
仍然一个大 runner 串行执行，直到末尾才发现 rows=0 或 certificate fail。
```

---

# 6. 数据合同

## 6.1 Source action table

每个 source action 必须记录：

```text
source_action_id
source_panel_id
action_id
candidate_id
event_id
dataset
seed
step
family_id
bucket_id
horizon
primitive_id
source_selection_rule
source_selection_legal
source_selection_oracle
source_selection_diagnostic_only
payload_hash
payload_norm
payload_linf
payload_role_entropy
action_apply_error_linf
action_apply_cosine
commit_time_available
uses_dataset_name
uses_outcome_at_commit
uses_future_step
```

## 6.2 Source outcome table

每个 action × branch × horizon 必须记录：

```text
source_action_id
generated_action_id
primitive_id
branch
horizon
CE_delta
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
task_safe_label
bad_event_label
null_event_label
weak_CP_label
strong_CP_label
long_risk_label
V_ctrl
V_ctrl_component_CE
V_ctrl_component_margin
V_ctrl_component_ECE
V_ctrl_component_NLL
V_ctrl_component_curvature
outcome_row_id
outcome_hash
metric_nan_count
metric_inf_count
```

## 6.3 Frontier table

每 oracle / legal / direct generator panel 必须记录：

```text
panel_id
selector_id
selector_type
K
action_count
h20_weak_CP
h20_strong_CP
h20_V_ctrl_mean
h20_V_ctrl_median
h20_V_ctrl_lcb
h20_bad_event
h20_null_rate
h80_weak_CP
h80_V_ctrl_lcb
h240_weak_CP
h240_V_ctrl_lcb
h240_long_risk
horizon_robust_coverage
support_balance_pass
family_count
max_family_share
dataset_count
max_dataset_share
gate_pass
reason_if_fail
```

## 6.4 Certificate table

```text
certificate_id
certificate_version
primitive_id
action_id
payload_hash
value_cert
risk_cert
support_cert
cost_cert
horizon_cert
linearity_error_cert
adamw_conflict_cert
tail_margin_cert
certificate_pass
certificate_compute_ms
uses_outcome_at_commit
uses_dataset_name
```

## 6.5 Base-Acc Sentinel table

```text
dataset
seed
model_id
model_class
hidden_dim
param_count
optimizer
lr
weight_decay
epochs
train_acc
val_acc
test_acc
train_CE
val_CE
test_CE
ECE
NLL
CEp99
time_to_target
step_time_q90
memory_peak
catastrophic_fail
used_for_controller
```

Models:

```text
LQ-t2-h256
MatchedMLP
QuadraticFeatureMLP
AdamWStrongLRGridMLP
```

StrongLRGrid 只能使用 pre-registered global grid，不得按 dataset 调参。

## 6.6 Runtime table

```text
runtime_candidate_id
primitive_id
controller_id
selected_path
step_id
active_step
candidate_count
feature_compute_ms
certificate_compute_ms
score_accept_ms
payload_apply_ms
base_step_ms
total_step_ms
step_ratio
memory_ratio
kernel_count
sync_count
audit_outside_timed
diagnostic_only
official_runtime
```

---

# 7. 实验阶段

---

## P0：v9.4.3 boundary reproduction and route reinterpretation

### 目标

复现 v9.4.3 boundary，确认当前不是 materializer 回退、fake/proxy、dataset leakage 或 Base-Acc regression。然后重新解释 route，避免过早把 AP0 全体判死。

### 必须记录

```text
route_v9430
source_outcome_materializer_closed_from_v9420
PANEL_ORC64_h20_weak_CP
PANEL_ORC64_h20_V_ctrl_lcb
PANEL_ORC64_h240_long_risk
PANEL_ORC128_h20_weak_CP
PANEL_ORC128_h20_V_ctrl_lcb
PANEL_ORC128_h240_long_risk
best_legal_panel_id
best_legal_h20_weak_CP
best_legal_h20_V_ctrl_lcb
best_legal_h240_long_risk
direct_generated_action_count
direct_branch_horizon_rows
best_direct_primitive_id
best_direct_h20_weak_CP
best_direct_h20_V_ctrl_lcb
best_direct_h240_long_risk
certificate_AUC_weak_CP
base_acc_sentinel_complete
mean_test_acc_LQ
mean_test_acc_MLP
system_legal_controller_pass
fake_proxy_count
```

### 判断标准

P0 pass：

```text
v9.4.3 metrics reproduced；
fake/proxy/offload = 0；
route reinterpreted as "borderline/absent under current source protocol"；
not interpreted as "all possible AP0 value absent" without P1 exhaustive audit。
```

### 可视化

```text
p0_v9430_route_ladder.svg
p0_oracle_vs_legal_vs_direct_summary.svg
p0_base_acc_sentinel_bar.svg
```

---

## P1：full AP0 source objective decomposition

### 目标

解释为什么 weak CP 接近或过线但 $V_{ctrl}$ LCB 仍为负。P1 不做 generator，不做 controller；只分析 source objective。

### 必须记录

对所有已 materialized AP0 source candidates：

```text
weak_CP_h20
strong_CP_h20
V_ctrl_h20
V_ctrl_h20_mean
V_ctrl_h20_median
V_ctrl_h20_lcb
V_ctrl_h20_p10
V_ctrl_h20_p90
V_ctrl_h20_outlier_count
V_ctrl_h20_negative_tail_mass
weak_CP_h80
V_ctrl_h80
weak_CP_h240
V_ctrl_h240
long_risk_h240
branch_dominance
control_branch_winner
```

核心诊断：

$$
Corr(WeakCP_{h20}, V_{ctrl,h20}),
$$

$$
P(V_{ctrl,h20}>0 \mid WeakCP_{h20}=1),
$$

$$
P(LongRisk_{h240}=1 \mid WeakCP_{h20}=1).
$$

### 判断标准

P1 objective mismatch pass：

```text
weak_CP 和 V_ctrl_h20 correlation < 0.35
or weak_CP pass panel has V_ctrl_lcb <= 0
or V_ctrl negative tail explains LCB fail。
```

如果 P1 成立，下一步不能继续只追 weak CP；必须重构 source objective 为 value-first / risk-constrained。

### 可视化

```text
p1_weakcp_vs_vctrl_scatter.svg
p1_vctrl_distribution_orc64_orc128.svg
p1_control_branch_winner_stacked.svg
p1_h20_h240_tradeoff_phase.svg
p1_lcb_outlier_influence_plot.svg
```

---

## P2：exhaustive AP0 source frontier enumeration

### 目标

v9.4.3 只报告若干 panel。P2 要在 full AP0 source universe 上枚举 topK / Pareto / support-balanced oracle，判断是否真的不存在 source survivor。

### Selector families

```text
ORC-A: weak CP oracle
ORC-B: V_ctrl_h20 oracle
ORC-C: V_ctrl_h20 with h240 long-risk penalty
ORC-D: horizon-robust oracle
ORC-E: support-balanced value-risk oracle
ORC-F: minimal-family-concentration oracle
ORC-G: Pareto oracle over {V_h20, WeakCP_h20, LongRisk_h240, Support}
```

Score examples：

$$
S_{VR}(a)=LCB(V_{ctrl,h20}(a))-\lambda_r LongRisk_{h240}(a)+\lambda_s Support(a).
$$

$$
S_{HR}(a)=\min_{h\in\{20,80,240\}}LCB(V_{ctrl,h}(a))-\lambda_r LongRisk_{h240}(a).
$$

K values：

```text
K = 16, 32, 64, 96, 128, 192, 256, 384, 512
```

### 必须记录

```text
selector_id
K
h20_weak_CP
h20_V_ctrl_mean
h20_V_ctrl_lcb
h20_bad_event
h20_null
h80_weak_CP
h80_V_ctrl_lcb
h240_weak_CP
h240_long_risk
horizon_robust_coverage
support_balance_pass
family_count
max_family_share
dataset_count
max_dataset_share
pass_level
```

### 判断标准

P2 weak source survivor：

```text
h20_weak_CP >= 0.60
h20_V_ctrl_lcb > 0
h240_long_risk <= 0.10
support_balance_pass = 1
```

P2 strong source survivor：

```text
h20_weak_CP >= 0.75
h20_V_ctrl_lcb > 0.05
h240_long_risk <= 0.05
horizon_robust_coverage >= 0.03
support_balance_pass = 1
```

P2 fail：

```text
no selector at any K satisfies weak source survivor。
```

### 可视化

```text
p2_topk_frontier_h20_vctrl.svg
p2_topk_frontier_longrisk.svg
p2_pareto_surface_vctrl_risk_support.svg
p2_oracle_selector_heatmap.svg
p2_support_balance_by_K.svg
```

---

## P3：source objective reset decision

### 目标

根据 P1/P2，决定 v9.4.4 后续走向：

```text
A. objective mismatch；
B. source frontier genuinely absent；
C. frontier exists but legal selector opaque；
D. generator destroys frontier。
```

### 判断逻辑

```text
if P2 weak source survivor pass:
    route_to = P4 legal selector and P5 oracle-seeded generator
elif P1 objective mismatch pass:
    route_to = source objective redesign, redefine CP/V/risk certificate
else:
    route_to = source primitive reset, AP0 source frontier exhausted under current objective
```

### 必须记录

```text
source_objective_route
weakCP_Vctrl_mismatch
exhaustive_oracle_pass
exhaustive_oracle_best_selector
exhaustive_oracle_best_K
frontier_absence_confidence
objective_redesign_required
```

### 可视化

```text
p3_route_decision_tree.svg
p3_objective_mismatch_dashboard.svg
```

---

## P4：legal source selector next-generation capacity audit

### 目标

只在 P2 发现 oracle survivor 或 P1 显示 objective mismatch 后继续。P4 不调 official controller，只评估 legal features 的信息上限。

### Feature groups

#### LS-A：gradient/action alignment

```text
cos_delta_negative_grad
cos_delta_adamw
projected_CE_descent
grad_norm_ratio
adamw_conflict_score
```

$$
\widehat{\Delta CE}_{lin}(a)=g_t^\top \Delta\theta_a.
$$

#### LS-B：tail-margin response

```text
margin_p10_current
tail_CE_p99_current
predicted_margin_gain_tail
tail_wrong_confidence
tail_sample_overlap
```

#### LS-C：source support memory

```text
family_support_lcb
horizon_support_lcb
neighbor_value_lcb
neighbor_longrisk_ucb
effective_sample_size
```

#### LS-D：payload geometry

```text
payload_norm
payload_linf
last_edge_fraction
low_rank_energy
role_entropy
sparsity
```

#### LS-E：linearization reliability

```text
linearized_CE_delta
linearized_margin_delta
linearization_error_ucb
state_NLL
state_margin_p10
```

### 必须记录

```text
feature_id
feature_group
AUC_weak_CP
AUC_Vctrl_positive
AUC_longrisk
PR_lift_weak_CP
PR_lift_longrisk
topK_h20_weak_CP
topK_h20_V_ctrl_lcb
topK_h240_long_risk
leave_dataset_drop
leave_family_drop
feature_cost_ms_q90
uses_dataset_name
uses_outcome_at_commit
```

### 判断标准

P4 capacity pass：

```text
AUC_weak_CP >= 0.75
AUC_longrisk >= 0.70
top64_h20_V_ctrl_lcb > 0
top64_h240_long_risk <= 0.15
feature_cost_ms_q90 <= 0.20
uses_dataset_name = 0
uses_outcome_at_commit = 0
```

P4 fail：

```text
best legal topK still h20 weak CP < 0.45
or V_ctrl_lcb <= 0
or long-risk > 0.30。
```

### 可视化

```text
p4_legal_feature_auc_bar.svg
p4_legal_topK_frontier.svg
p4_feature_cost_vs_auc.svg
p4_leaveout_feature_drop.svg
```

---

## P5：oracle-seeded generator preservation test

### 目标

把 generator 问题和 source selection 问题解耦。使用 oracle panels 只作 diagnostic input，不得 official。若 generator 连 oracle-good source 都保不住，问题在 generator；若 generator 能保住 oracle source，问题在 legal source selector。

### Panels

```text
PANEL-ORC64
PANEL-ORC128
PANEL-VCTRL64
PANEL-HR64
PANEL-LEGAL64
PANEL-RND64
```

Generators tested：

```text
AP0b-AP0f existing transform generators
AP0g-AP0k direct generators
AP0l-AP0q new objective-solving generators
```

### 必须记录

```text
source_panel_id
generator_id
source_h20_weak_CP
source_h20_V_ctrl_lcb
source_h240_longrisk
generated_h20_weak_CP
generated_h20_V_ctrl_lcb
generated_h240_longrisk
damage_mean
damage_median
source_positive_lost_rate
source_negative_fixed_rate
payload_apply_error_linf
branch_horizon_completion
```

Damage：

$$
Damage(a)=V_{generated}(a)-V_{source}(a).
$$

### 判断标准

Generator preserve pass：

```text
source_positive_lost_rate <= 0.20
Damage_median >= -0.02
generated_h20_V_ctrl_lcb > 0
generated_h240_longrisk <= 0.15
```

Generator destructive fail：

```text
source panel survivor pass = 1
but generated panel survivor pass = 0
and source_positive_lost_rate > 0.50。
```

### 可视化

```text
p5_source_to_generated_damage_heatmap.svg
p5_generator_preservation_by_panel.svg
p5_source_positive_lost_rate.svg
p5_generated_vctrl_distribution.svg
```

---

## P6：direct value-producing generator reset

### 目标

不再做 heuristic direct generator，而是实现 objective-solving source generator。v9.4.4 至少并行测试 6 个新 primitives。

---

## AP0l：Linearized Trust-Region Source

目标：

$$
\min_{\Delta\theta}
g_t^\top \Delta\theta
+
\frac{\lambda}{2}\|\Delta\theta\|^2
$$

subject to：

$$
\|\Delta\theta\|\le r,
$$

$$
\cos(\Delta\theta,\Delta\theta_{AdamW})\ge c_{min},
$$

$$
TailRisk_{lin}(\Delta\theta)\le \tau_r.
$$

记录 certificate：

```text
linearized_CE_gain
trust_region_radius
adamw_alignment
tail_risk_linearized
norm_ratio
```

## AP0m：Tail-Safe Projected Descent Source

先生成 descent direction，再投影掉会伤害 tail margin 的方向：

$$
\Delta\theta = Proj_{\mathcal{C}_{tail}}(-g_t)
$$

where：

$$
\mathcal{C}_{tail}
=
\{\Delta\theta:
J_{tail}\Delta\theta \ge -\epsilon_{tail}\}.
$$

记录：

```text
tail_margin_projection_norm
tail_constraint_violation_before
tail_constraint_violation_after
CE_descent_after_projection
```

## AP0n：AdamW Residual Value Corrector

生成 AdamW 的低秩 residual，而不是独立 payload：

$$
\Delta\theta_{AP0n}
=
\Delta\theta_{AdamW}
+
P_{lowrank}(\Delta\theta_{func}-\Delta\theta_{AdamW}).
$$

目标是减少 generator destructiveness。

记录：

```text
residual_norm
residual_rank
cos_with_adamw
cos_with_negative_grad
projected_descent_gain
```

## AP0o：Horizon-Conservative Blend Source

用 h20 value 和 h240 risk 的 certificate 产生 conservative blend：

$$
\Delta\theta
=
\alpha \Delta\theta_{value}
+
(1-\alpha)\Delta\theta_{safe},
$$

where：

$$
\alpha
=
clip\left(
\frac{RiskBudget-EstimatedRisk}{RiskBudget},
0,
1
\right).
$$

记录：

```text
alpha
estimated_h20_value
estimated_h240_risk
risk_budget
blend_norm
```

## AP0p：Support-Memory Meta Source

只在 family/horizon/source-neighborhood 有足够 support 时生成：

$$
LCB(Support(a))\ge\tau_s.
$$

记录：

```text
support_neighbor_count
support_value_lcb
support_longrisk_ucb
family_horizon_count
```

## AP0q：NoOp-Guarded Sparse Last-Edge Source

只允许 last-edge sparse payload，并且必须 beat NoOp certificate：

$$
LCB(V_{Real}-V_{NoOp})>0.
$$

记录：

```text
last_edge_sparsity
noop_gain_lcb
payload_apply_cost
```

---

### P6 必须记录

```text
generator_id
generated_action_count
payload_tensor_written
certificate_tensor_written
payload_hash_missing_count
certificate_hash_missing_count
action_apply_error_linf_max
commit_time_available
uses_dataset_name
uses_outcome_at_commit
h20_weak_CP
h20_V_ctrl_lcb
h240_long_risk
source_survivor_pass
```

### P6 pass

```text
at least one AP0l-AP0q:
  generated_action_count >= 64
  action_apply_error_linf_max <= 1e-6
  h20_weak_CP >= 0.60
  h20_V_ctrl_lcb > 0
  h240_long_risk <= 0.10
  uses_dataset_name = 0
  uses_outcome_at_commit = 0
```

### 可视化

```text
p6_generator_value_risk_frontier.svg
p6_generator_certificate_components.svg
p6_ap0l_to_ap0q_comparison.svg
p6_payload_geometry_vs_value.svg
```

---

## P7：effect-valid certificate redesign

### 目标

certificate 必须从 construction-valid 变成 effect-valid。P7 只对 P5/P6 survivors 运行。

### Certificate components

```text
C_value:
  linearized_CE_gain
  projected_margin_gain
  noop_gain_lcb
  adamw_parallel_gain_lcb

C_risk:
  tail_margin_violation_ucb
  h240_longrisk_estimate
  curvature_growth_estimate
  update_norm_tail_ratio

C_support:
  effective_sample_size
  family_horizon_support
  neighbor_value_lcb
  leaveout_stability_score

C_cost:
  payload_apply_ms_estimate
  certificate_compute_ms
  memory_delta
```

Certificate score：

$$
CertScore(a)
=
LCB(C_{value}(a))
-
\lambda_r UCB(C_{risk}(a))
+
\lambda_s LCB(C_{support}(a))
-
\lambda_c C_{cost}(a).
$$

### 必须记录

```text
certificate_id
component_values
component_ablation
AUC_weak_CP
AUC_Vctrl_positive
AUC_longrisk
P_weak_CP_given_cert_pass
P_weak_CP_given_cert_fail
P_longrisk_given_cert_pass
P_longrisk_given_cert_fail
monotone_sign_pass
calibration_ECE
leave_dataset_drop
leave_family_drop
```

### P7 pass

```text
AUC_weak_CP >= 0.75
AUC_longrisk >= 0.75
monotone_sign_pass = 1
P_weak_CP_given_cert_pass >= 2 * P_weak_CP_given_cert_fail
P_longrisk_given_cert_pass <= 0.5 * P_longrisk_given_cert_fail
certificate_compute_ms_q90 <= 0.20
```

### 可视化

```text
p7_certificate_reliability_curve.svg
p7_certificate_component_ablation.svg
p7_certscore_vs_vctrl.svg
p7_certscore_vs_longrisk.svg
```

---

## P8：minimal source certificate controller

### 目标

只在 P2/P6/P7 有 survivor 后运行 controller。Controller 必须最小、可解释、cross-fitted，不能堆很多 feature。

Controller form：

$$
Accept(a)=1
\iff
LCB(V_{cert}(a))>0
\land
UCB(Risk_{cert}(a))\le\tau_r
\land
LCB(Support_{cert}(a))\ge\tau_s
\land
Cost(a)\le C_{max}.
$$

### Cross-fit

```text
seed folds:
  train/calibration/heldout rotate

leave-dataset-out:
  MNIST holdout
  Fashion-MNIST holdout
  KMNIST holdout

leave-family-out:
  high-volume family holdout

leave-horizon-context-out:
  h20/h80/h240 stress
```

### 必须记录

```text
controller_id
primitive_id
certificate_id
fold_id
accepted_count
coverage
h20_weak_CP
h20_V_ctrl_lcb
h240_long_risk
precision_control_positive
bad_event_rate
null_rate
support_balance_pass
thresholds
dataset_name_used
outcome_at_commit_used
```

### P8 pass

```text
dataset_name_used = 0
outcome_at_commit_used = 0
coverage in [0.01, 0.15]
h20_weak_CP >= 0.60
h20_V_ctrl_lcb > 0
h240_long_risk <= 0.10
support_balance_pass = 1
leave_dataset_drop <= 0.10
```

### 可视化

```text
p8_controller_frontier.svg
p8_crossfit_metric_distribution.svg
p8_leaveout_heatmap.svg
p8_threshold_stability.svg
```

---

## P9：selected source online runtime

### 目标

只有 P8 controller pass 后才 official 测 selected runtime。没有 selected controller 时只做 microbench。

### 必须记录

```text
runtime_candidate_id
primitive_id
controller_id
selected_controller_used
payload_apply_used
certificate_compute_used
step_count
active_step_count
accepted_count
feature_compute_ms_q90
certificate_compute_ms_q90
score_accept_ms_q90
payload_apply_ms_q90
base_train_step_ms_q90
total_step_ms_q90
step_ratio_q90
memory_ratio
kernel_count
sync_count
diagnostic_only
official_runtime
```

### P9 pass

```text
selected_controller_used = 1
diagnostic_only = 0
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
payload_apply_error_linf_max <= 1e-6
```

### 可视化

```text
p9_runtime_waterfall.svg
p9_payload_apply_distribution.svg
p9_step_ratio_over_time.svg
```

---

## P10：Base-Acc Sentinel + Strong Baseline Diagnostic

### 目标

回答“有在数据集上训练吗、acc 如何、和 MLP 比如何”，但隔离它与 controller。Base-Acc 是健康监控，不是 functional success。

### 设置

Datasets：

```text
MNIST
Fashion-MNIST
KMNIST
```

Seeds：

```text
0,1,2,3,4,5,6,7,8,9
```

Models：

```text
LQ-t2-h256
MatchedMLP
QuadraticFeatureMLP
AdamWStrongLRGridMLP
```

Rules：

```text
same seed schedule
same train budget
pre-registered global LR grid
no dataset-specific tuning
not used for controller
not used for source selector
not used for certificate threshold
```

### 必须记录

```text
dataset
seed
model_id
train_acc
val_acc
test_acc
train_CE
val_CE
test_CE
ECE
NLL
CEp99
step_time_q90
memory_ratio
time_to_90pct_train_acc
time_to_target_val_acc
catastrophic_fail
```

### P10 health pass

```text
LQ_catastrophic_fail = 0
LQ_mean_test_acc >= MatchedMLP_mean_test_acc - 0.01
AdamWStrongLRGridMLP_diagnostic_materialized = 1
QuadraticFeatureMLP_materialized = 1
base_acc_used_for_controller = 0
```

### 可视化

```text
p10_test_acc_by_dataset_model.svg
p10_train_val_test_curves.svg
p10_lq_minus_mlp_gap_by_seed.svg
p10_time_to_target.svg
p10_ece_nll_comparison.svg
```

---

## P11：system integration boundary

### 目标

把 P8 controller、P9 runtime、P10 base health 放在同一 system gate 中，但不把 Base-Acc 用于 controller。

### 必须记录

```text
system_candidate_id
primitive_id
controller_id
certificate_id
runtime_candidate_id
base_candidate
controller_pass
runtime_pass
base_acc_sentinel_pass
strong_baseline_diagnostic_complete
official_eligible
system_legal_controller_pass
reason_if_fail
```

### P11 pass

```text
controller_pass = 1
runtime_pass = 1
base_acc_sentinel_pass = 1
official_eligible = 1
system_legal_controller_pass = 1
```

If P11 fails, downstream stays not_run.

---

## P12：leave-out / paired replay / short-full boundary

### 目标

只有 P11 pass 后打开。v9.4.4 可以预生成 diagnostic scaffolding，但不能 official。

Official stages：

```text
P12a leave-dataset-out
P12b leave-family/stratum-out
P12c official paired replay
P12d short-run functional training
P12e full-run sample efficiency / calibration / robustness
P12f continual / anti-forgetting
```

### Paired replay branches

```text
RealFunctional
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledPayload
ShuffledCertificate
ShuffledSourceSelector
```

### Pass

```text
RealFunctional beats AdamWParallel and bestLR on paired replay；
shuffle controls fail；
LDO/LSO no support collapse；
short/full run improves at least one non-accuracy axis:
  calibration
  CEp99
  sample efficiency
  robustness
  continual retention
without losing primary acc。
```

---

# 8. 并行执行方案

## Batch A：data/objective audit

```text
P0 boundary reproduction
P1 source objective decomposition
P2 exhaustive AP0 source frontier enumeration
P3 source objective reset decision
```

Expected wallclock should be dominated by reading existing outcome tables, not new training.

## Batch B：legal selector capacity

```text
P4 legal source selector features
leave-dataset / leave-family capacity
feature cost audit
```

Can run in parallel with Batch C.

## Batch C：generator matrix

```text
P5 oracle-seeded generator preservation
P6 AP0l-AP0q direct generator reset
single-action preflight
mini-panel preflight
official 64/128-action smoke
```

Preflight required:

```text
single_action_payload_write_pass = 1
single_action_apply_replay_pass = 1
single_action_branch_horizon_rows > 0
mini_panel_rows_expected = mini_panel_rows_actual
```

## Batch D：certificate matrix

```text
P7 effect-valid certificate redesign
component ablation
monotone sign audit
```

Only uses P5/P6 generated outcomes.

## Batch E：Base-Acc Sentinel

```text
P10 10-seed fixed-config training
StrongLRGridMLP diagnostic
QuadraticFeatureMLP diagnostic
```

Runs independently. Results cannot influence controller.

## Batch F：runtime microbench

```text
payload apply microbench for AP0l-AP0q
certificate compute cost
selected runtime only if P8 pass
```

## Batch G：system boundary

```text
P8 controller
P9 runtime
P11 system gate
P12 downstream boundary
```

---

# 9. Route decision table

## R0-ObjectiveMismatch

Condition：

```text
weak CP near/pass but V_ctrl LCB negative；
P1 objective mismatch pass = 1。
```

Next：

```text
Redefine source objective;
separate CP label from value gate;
do not tune thresholds blindly。
```

## R1-FullAP0SourceFrontierAbsent

Condition：

```text
P2 exhaustive oracle fail:
  no K / Pareto set satisfies source survivor gate。
```

Next：

```text
Stop AP0 source selector search;
source primitive/generator must be rebuilt from optimization objective。
```

## R2-LegalSourceSelectorOpaque

Condition：

```text
P2 oracle survivor pass;
P4 legal selector fail。
```

Next：

```text
Do not run controller;
implement new legal effect features or certificate-producing generator。
```

## R3-GeneratorDestructive

Condition：

```text
Oracle source panel pass;
generated panel fail;
source-positive lost rate > 0.50。
```

Next：

```text
Stop transform generator;
use value-preserving or direct constrained generator。
```

## R4-DirectGeneratorObjectiveFail

Condition：

```text
AP0l-AP0q direct generators fail h20 V_ctrl or h240 longrisk。
```

Next：

```text
Redesign generator objective;
do not tune AP0j/AP0k thresholds。
```

## R5-CertificateEffectFail

Condition：

```text
Generator has frontier;
certificate AUC / monotonicity fail。
```

Next：

```text
Certificate redesign;
no controller threshold search。
```

## R6-ControllerSupportFail

Condition：

```text
certificate passes local;
controller fails heldout / LDO / support。
```

Next：

```text
support calibration / uncertainty model redesign。
```

## R7-RuntimeFail

Condition：

```text
controller pass;
selected runtime step_ratio_q90 > 1.50。
```

Next：

```text
payload apply / certificate compute / scheduler optimization。
```

## R8-SystemReady

Condition：

```text
controller pass;
runtime pass;
base health pass;
official eligible = 1。
```

Next：

```text
Open LDO/LSO, paired replay, short/full training。
```

---

# 10. 停止条件

## 必须停止小修小补的条件

```text
certificate AUC weak CP < 0.60 after P7；
direct generator h20 V_ctrl LCB < 0 after P6；
P2 exhaustive oracle fail；
legal selector topK h20 weak CP < 0.30 and V_ctrl LCB < 0；
Base-Acc Sentinel catastrophic fail。
```

一旦触发，不允许继续：

```text
AP0j threshold tuning
certificate threshold tuning
PayloadLinf topK tuning
StateNLL topK tuning
dataset-specific branch
posthoc oracle promotion
```

## 可进入 controller 的条件

```text
P2 or P6 source survivor pass；
P7 certificate effect-valid pass；
P4 or P7 legal feature cost pass；
no dataset leakage；
no outcome-at-commit leakage。
```

## 可进入 selected runtime 的条件

```text
P8 controller pass；
selected primitive and certificate fixed；
no diagnostic-only path。
```

## 可进入 paired replay / short-full 的条件

```text
P11 system_legal_controller_pass = 1。
```

---

# 11. 最终预期

v9.4.4 的最好结果不是“直接 full functional success”，而是：

```text
1. 明确 AP0 source frontier 是真的 absent、borderline 还是 objective mismatch；
2. 如果 AP0 source frontier 存在，知道 legal selector 是否能找到；
3. 如果 legal selector 找不到，知道 oracle-seeded generator 是否能 preserve value；
4. 如果 direct generator 失败，知道是 h20 value fail、h240 risk fail，还是 certificate fail；
5. Base-Acc Sentinel 与 StrongLRGridMLP 对照补齐；
6. 下一步不再凭感觉继续 AP0j/AP0k/threshold patch。
```

理想 route：

```text
R8-SystemReady
```

更现实但仍有价值的 route：

```text
R0-ObjectiveMismatch
or
R1-FullAP0SourceFrontierAbsent
or
R3-GeneratorDestructive
or
R4-DirectGeneratorObjectiveFail
```

这些 route 都比“继续调一个 feature/threshold”更有价值，因为它们能决定下一阶段到底是：

```text
source objective reset；
source primitive reset；
legal selector / certificate reset；
generator optimization reset；
runtime optimization。
```
