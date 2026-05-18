# DG-KAN v9.4.3 Source Frontier Recovery / Direct Generator / Effect-Valid Certificate / Parallel Validation 完整实验计划

> 本计划基于 v9.4.2 `Source Outcome Materializer Closure / Value-Producing Source Triage` 的真实结果制定。  
> v9.4.2 的最大进展是把 v9.4.1 的 `0-row source outcome materializer` 修通，完整落盘 `7020 / 7020` source outcome rows。  
> v9.4.2 的最大失败是：一旦 outcome 真正可测，stratified source panel、AP0b-AP0f generated source actions 和 v9410 certificate 全部没有形成 value-positive / horizon-safe frontier。

---

# 0. 一句话判断

v9.4.2 不是“完全没进展”。它完成了一个关键测量闭环：source outcome materializer 从 `0` 行推进到 `7020` 行，并且质量审计通过。但用户觉得慢是合理的，因为这轮没有推进到 controller、runtime、paired replay 或 full training；它只是把下一层 failure 从“无法评估”变成“可测后发现 source panel / generator / certificate 都不够”。

本轮不能继续做 threshold patch，也不能继续微调 AP0b-AP0f。v9.4.3 必须直接回答一个更本质的问题：

$$
\boxed{
\text{functional source action 的好 frontier 到底能否被 legal selector 找到，或者能否被 direct generator 主动生成？}
}
$$

如果只能用 oracle outcome 找到好 source，但 legal selector 和 direct generator 都找不到，那么 AP0/AP0b-AP0f 路线应暂停，转向新的 functional primitive。  
如果 legal selector 或 direct generator 能找到 h20-positive、h240-safe 的 source actions，再进入 certificate/controller/runtime。

---

# 1. v9.4.2 独立数据判断

## 1.1 v9.4.2 的真实进展

v9.4.1 的核心 blocker 是：

```text
source outcome materializer expected = 7020
source outcome materializer actual = 0
```

v9.4.2 已经真实修通：

```text
generated_action_count_input = 260
branch_count = 9
horizon_count = 3
branch_horizon_row_count_expected = 7020
branch_horizon_row_count_actual = 7020
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
secondary_delta_completion_rate = 1
rows_per_sec_total = 23.6331087573844
wallclock_sec = 297.04090443905443
row_count_failed/retried/unresolved = 0 / 0 / 0
quality_audit_pass = 1
missing_branch/horizon/secondary = 0 / 0 / 0
metric_nan/inf = 0 / 0
label_exclusivity_violation = 0
duplicate_outcome_row_id_count = 0
```

这说明 v9.4.2 的进展是 **measurement capability progress**。从科学角度，这是有价值的：之前不知道 generated source action 是好是坏，现在可以在 full branch/horizon outcome 上测出来。

## 1.2 v9.4.2 的真实失败

### 1.2.1 source AP0 same-panel baseline 已经不好

P3 same-panel AP0 baseline：

```text
source_action_count = 52
branch_horizon_row_count_expected/actual = 1404 / 780
AP0_weak_CP_precision_h20 = 0.11153846153846154
AP0_strong_CP_precision_h20 = 0.08076923076923077
AP0_weak_CP_precision_h80 = 0.13076923076923078
AP0_weak_CP_precision_h240 = 0.15
AP0_long_risk_rate_h240 = 0.8038461538461539
AP0_V_ctrl_lcb_h20 = -1.29393883306781
AP0_V_ctrl_lcb_all = -1.2099435679819464
AP0_horizon_robust_action_count = 8
AP0_horizon_robust_coverage = 0.15384615384615385
source_ap0_baseline_useful = 0
source_ap0_baseline_bad = 1
```

这里有一个需要独立指出的细节：P3 AP0 baseline 的 `expected/actual = 1404 / 780`，不是 full `9 branches × 3 horizons` 的完整 AP0 baseline。v9.4.2 的 damage matrix 也使用 `paired_horizon_count = 780`，因此 same-panel AP0 与 generated source 的对比是可用的，但它仍不应被扩大成“全 AP0 universe 都坏”。

更关键的是：这个 source panel 是 v9.4.1 中修过代表性的 `PANEL-S256` 体系下的 source input，但代表性 sampling 本来就会接近 base rate，而不是自动选到 frontier。v9.4.0 已经显示 full AP0 universe 中有 oracle source frontier，而当前 representative panel 的 h20 weak CP 只有 `0.1115`，这说明：

$$
\boxed{
\text{representative panel 适合评估分布，不适合作为 value-producing source input。}
}
$$

v9.4.2 的 route 写成 `source_panel_value_poor_despite_stratification` 是合理的，但更深的解释不是“stratification 无效”，而是：**stratification 目标错了**。它只保证 covariate representativeness，不保证 value enrichment。

### 1.2.2 AP0b-AP0f generated source 没有 immediate direction

P4 generated h20：

| primitive | weak CP | strong CP | bad-event | V_ctrl LCB | pass |
|---|---:|---:|---:|---:|---:|
| AP0b-LastEdgeLinearizedDescentSource | 0.096154 | 0.057692 | 0.288462 | -1.505584 | 0 |
| AP0c-AdamWResidualOrthogonalSource | 0.096154 | 0.096154 | 0.326923 | -1.499603 | 0 |
| AP0d-TailMarginRepairSource | 0.173077 | 0.134615 | 0.365385 | -1.801472 | 0 |
| AP0e-CurvatureGuardedLowRankEdgeSource | 0.019231 | 0.019231 | 0.384615 | -2.044336 | 0 |
| AP0f-SupportMemorySource | 0.115385 | 0.076923 | 0.365385 | -1.603251 | 0 |

best primitive 是 AP0d，但它只是 weak CP `0.1731`，bad-event `0.3654`，V_ctrl LCB `-1.8015`。这不是“阈值调一下”的问题。若要形成 deployable source frontier，h20 至少需要可见的正 value，而当前 best primitive 的 value lower bound 是强负数。

### 1.2.3 generator 有 damage，但不是唯一主因

P5：

```text
paired_horizon_count = 780
Damage_mean = -0.09249623563492862
Damage_median = -0.1004650485701859
source_positive_horizon_count = 102
source_positive_lost_after_generation_rate = 0.9215686274509803
source_negative_fixed_after_generation_rate = 0.13569321533923304
generator_value_preserving_pass = 0
generator_damage_fail = 1
```

这说明 AP0b-AP0f transform 会严重丢掉 source-positive rows，`source_positive_lost_after_generation_rate = 0.9216` 非常糟糕。  
但因为 P3 已经显示 source AP0 panel 本身 value-poor，所以不能把 failure 全归因于 transform。v9.4.2 的真实结论应该是两层同时失败：

$$
\boxed{
\text{source input 不够好，generator transform 还会进一步破坏正向 horizon rows。}
}
$$

### 1.2.4 horizon safety 失败非常严重

P6：

```text
selected_primitive_id = AP0d-TailMarginRepairSource
weak_CP_precision_h80 = 0.19230769230769232
weak_CP_precision_h240 = 0.11538461538461539
long_risk_rate_h240 = 0.8653846153846154
horizon_robust_action_coverage = 0.038461538461538464
horizon_extension_pass = 0
```

`h240 long-risk = 0.8654` 基本说明 AP0d 的 long horizon 不可用。即使 h20 能勉强找到一些 weak CP，长 horizon 也会被 risk 吃掉。

### 1.2.5 certificate 不是 effect-valid

P7：

```text
AUC_certificate_weak_CP = 0.47741176470588237
AUC_certificate_strong_CP = 0.45378723404255317
AUC_certificate_longrisk = 0.4895340819542947
P_weak_CP_given_cert_pass = 0.12757201646090535
P_weak_CP_given_cert_fail = 0.12849162011173185
P_longrisk_given_cert_pass = 0.25925925925925924
P_longrisk_given_cert_fail = 0.2849162011173184
Lift_weak = 0.9928430846305243
Lift_longrisk = 0.9099491648511256
monotone_sign_pass = 0
certificate_sufficiency_pass = 0
```

这几乎是“证书无信息”：cert pass 和 cert fail 的 weak CP 率几乎一样。继续调 certificate threshold 没意义。

### 1.2.6 Base-Acc Sentinel 有训练，但不是 official functional result

v9.4.2 有在 MNIST / Fashion-MNIST / KMNIST 上训练，但只是 Base-Acc Sentinel：

```text
sentinel_row_count = 27
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
model_count = 3
sentinel_complete = 1
mean_test_acc_LQ = 0.6553819444444444
mean_test_acc_MLP = 0.5559895833333334
LQ_minus_MLP_mean_test_acc = 0.09939236111111105
LQ_catastrophic_fail = 0
base_acc_used_for_controller = 0
```

结论：

```text
LQ-t2-h256 fixed-config sentinel 平均 test acc = 65.54%
MatchedMLP fixed-config sentinel 平均 test acc = 55.60%
LQ - MLP = +9.94 个百分点
```

这说明 base LQ 没有 catastrophic fail，并且在这个固定 sentinel 设置下比 MatchedMLP 好。但绝对 acc 偏低，且没有 official functional controller，因此不能写成“DG-KAN functional 已经超过 MLP”。

---

# 2. 为什么用户会感觉“还是很慢”

这个感觉是对的。当前慢的原因不是实验没有信息，而是执行模式过于串行：

```text
v9.4.0：发现 source generator 没 materialize；
v9.4.1：materialize source generator，但 outcome rows = 0；
v9.4.2：修通 outcome rows，但发现 source panel / generator / cert 都不好。
```

这每一轮都是真实推进，但每轮只清一个 blocker，所以从系统目标看显得慢。v9.4.3 必须改变执行方式：

```text
1. 不再一个大 runner 串行跑到底；
2. 先做单 action / 单 branch / 单 horizon preflight；
3. 多 panel、多 source selector、多 generator、多 certificate 并行矩阵；
4. 每个分支有硬 kill gate；
5. Base-Acc Sentinel 与 source/generator/certificate 实验并行；
6. runtime 只对 survivor 测，不在 upstream fail 时浪费大跑。
```

---

# 3. v9.4.3 总体目标

v9.4.3 的目标不是修 AP0d threshold，也不是继续加一个小 certificate。它要验证以下第一性命题：

$$
\boxed{
\text{是否存在 legal source selector 或 direct source generator，能产生 h20-positive 且 h240-safe 的 functional source actions？}
}
$$

v9.4.3 分两条路线并行：

```text
Route A: Source Frontier Recovery
  从 full AP0 universe 中，用 legal commit-time selector 找到接近 oracle frontier 的 source actions。

Route B: Direct Generator Reset
  不依赖 AP0 existing source 是否好，直接从 state/gradient/tail/control conflict 生成 source action。
```

只要 A 或 B 任一路线产生 survivor，再进入：

```text
Effect-valid certificate
Minimal source controller
Selected runtime
Leave-out / paired replay boundary
Short/full training boundary
```

---

# 4. 明确不做什么

v9.4.3 不做：

```text
1. 不继续调 AP0d threshold；
2. 不继续调 v9410 certificate threshold；
3. 不把 representative panel 当成 value source selector；
4. 不把 Base-Acc Sentinel 当作 functional success；
5. 不按 dataset 调 source selector、generator 或 certificate；
6. 不使用 test/validation outcome at commit time；
7. 不用 teacher / distillation / auxiliary loss / loss modification；
8. 不用 oracle-selected rows 训练 official selector；
9. 不把 diagnostic oracle route 写成 official；
10. 不在 h20/h80/h240 source frontier 未过时打开 selected runtime / paired replay / full training。
```

允许做：

```text
1. oracle panel 作为 upper-bound diagnostic；
2. dataset-level failure diagnosis，但不能 dataset-specific tuning；
3. legal selector feature capacity audit；
4. direct generator smoke；
5. source-to-generated damage matrix；
6. effect certificate calibration；
7. Base-Acc Sentinel 固定配置健康监控；
8. 多 worker / 多 shard / 多 panel 并行验证。
```

---

# 5. 核心假设

## H1：v9.4.2 的 source panel 失败不是“AP0 full universe 没好 source”，而是 representative panel 没有 value enrichment

背景：v9.4.0 已经显示 full AP0 universe 有 oracle source frontier；v9.4.2 的 stratified panel h20 weak CP 只有 `0.1115`。代表性采样会接近 base rate，而不会自动命中 frontier。

H1 成立标准：

```text
oracle source panel weak CP precision >= 0.60
random / representative panel weak CP precision <= 0.20
legal selector panel weak CP precision 明显低于 oracle panel
```

H1 失败标准：

```text
oracle source panel 在 same protocol 下也 weak CP < 0.30 或 V_ctrl LCB <= 0
```

若 H1 失败，说明 AP0 full source frontier 在 v9.4.2 protocol 下可能不成立，需要回到 full universe oracle consistency audit。

## H2：当前 legal source selector 不够，不应继续单特征 top-K

H2 成立标准：

```text
PayloadLinf / StateNLL / CEp99 / margin / payload norm 等单特征 selector top-K weak CP < 0.30
V_ctrl LCB <= 0
long-risk h240 > 0.30
```

H2 失败标准：

```text
某个 legal selector top-K source panel 达到：
  h20 weak CP >= 0.40
  V_ctrl LCB h20 > 0
  h240 long-risk <= 0.25
```

## H3：AP0b-AP0f generator transform 会破坏 source-positive rows

H3 成立标准：

```text
source_positive_lost_after_generation_rate >= 0.50
Damage_median < -0.02
对 oracle-good source panel 也出现明显 lost
```

H3 失败标准：

```text
在 oracle-good source panel 上，某 generator：
  Damage_median >= -0.02
  source_positive_lost_rate <= 0.25
  generated weak CP >= 0.70 * source weak CP
```

## H4：需要 direct source generator，而不是只变换 AP0 source payload

H4 成立标准：

```text
direct generator AP0g-AP0k 至少一个达到：
  h20 weak CP >= 0.30
  h20 V_ctrl LCB > 0
  h20 bad-event <= 0.20
  h240 long-risk <= 0.30
```

H4 失败标准：

```text
所有 direct generator h20 weak CP < 0.20 且 V_ctrl LCB <= 0
```

## H5：effect-valid certificate 必须预测 outcome，而不是描述 construction

H5 成立标准：

```text
AUC weak CP >= 0.70
AUC long-risk >= 0.70 when scored as risk
Lift weak >= 2.0
Lift long-risk <= 0.70
P_weak_CP_given_cert_pass >= 0.35
P_longrisk_given_cert_pass <= 0.20
monotone_sign_pass = 1
```

H5 失败标准：

```text
certificate AUC around 0.5, or cert-pass weak CP 与 cert-fail 无差异。
```

## H6：Base-Acc Sentinel 只做健康监控，不做 controller tuning

H6 成立标准：

```text
base_acc_sentinel_complete = 1
base_acc_used_for_controller = 0
same_seed_schedule = 1
same_budget = 1
dataset_specific_tuning = 0
LQ_catastrophic_fail = 0
```

H6 失败标准：

```text
Base-Acc 结果用于选择 source selector / generator / certificate threshold。
```

---

# 6. 数据合同

## 6.1 source panel contract

每个 source panel 必须记录：

```text
panel_id
panel_type
source_selector_id
selection_mode
uses_outcome_for_selection
uses_dataset_name
uses_validation_or_test
uses_future_step
source_action_count
action_ids
family_distribution
step_bucket_distribution
score_bucket_distribution
payload_bucket_distribution
PSI_vs_full
KL_vs_full
max_family_gap
max_step_bucket_gap
max_score_bucket_gap
max_payload_bucket_gap
```

Panel 类型：

```text
REP: representative / stratified diagnostic panel
ORC: oracle diagnostic panel, not official
LGL: legal commit-time selector panel
DIR: direct generator input panel
RND: random baseline panel
```

Official selector 必须：

```text
uses_outcome_for_selection = 0
uses_dataset_name = 0
uses_validation_or_test = 0
uses_future_step = 0
```

## 6.2 source outcome contract

每个 row 必须记录：

```text
source_action_id
generated_action_id
primitive_id
branch
horizon
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
V_ctrl
weak_CP_label
strong_CP_label
bad_event_label
null_event_label
long_risk_label
horizon_robust_label
beats_adamwparallel
beats_bestlr
beats_noop
beats_random
outcome_row_id
horizon_state_hash
branch_identity_hash
```

Quality pass：

```text
missing_branch_count = 0
missing_horizon_count = 0
missing_secondary_delta_count = 0
metric_nan_count = 0
metric_inf_count = 0
label_exclusivity_violation = 0
duplicate_outcome_row_id_count = 0
```

## 6.3 source-to-generated damage contract

必须记录：

```text
source_action_id
generated_action_id
primitive_id
horizon
V_ctrl_source
V_ctrl_generated
Damage = V_ctrl_generated - V_ctrl_source
weak_CP_source
weak_CP_generated
strong_CP_source
strong_CP_generated
long_risk_source
long_risk_generated
source_positive_lost
generated_fixed_source_negative
payload_cosine_source_generated
payload_norm_ratio
certificate_pass
```

## 6.4 certificate contract

每个 certificate row 必须记录：

```text
certificate_id
certificate_schema_version
primitive_id
payload_hash
certificate_hash
commit_time_available
uses_dataset_name
uses_outcome_at_commit
uses_future_step
linearized_CE_delta
linearized_margin_delta
gradient_alignment
adamw_conflict
tail_margin_guard
curvature_guard
support_lcb
bad_ucb
null_ucb
horizon_risk_ucb
cost_estimate_ms
certificate_score
certificate_pass
```

Effect validity 必须用 outcome 后验审计，但 certificate 本身不能用 outcome at commit。

## 6.5 Base-Acc Sentinel contract

必须记录：

```text
dataset
seed
model_id
model_class
params
train_steps
wallclock_sec
train_acc
val_acc
test_acc
train_loss
val_loss
test_loss
ECE
NLL
CEp99
margin_p10
step_time_ms_q90
memory_mb_peak
same_budget
same_seed_schedule
hyperparams_fixed_before_run
dataset_specific_tuning
used_for_controller
```

---

# 7. 实验阶段

---

## P0：v9.4.2 boundary reanalysis and AP0 baseline completeness audit

### 目标

复现 v9.4.2 的 terminal boundary，并专门审计 P3 AP0 baseline `expected/actual = 1404 / 780` 是否会影响 route 判断。

### 假设

H0：v9.4.2 的 7020-row generated source outcome materializer 已闭合；P3 AP0 baseline 虽非 full 9-branch，但 same-panel damage comparison 仍可用。

### 必须记录

```text
route_v9420
source_outcome_materialized
branch_horizon_row_count_expected
branch_horizon_row_count_actual
quality_audit_pass
P3_AP0_expected_rows
P3_AP0_actual_rows
P3_AP0_missing_branch_count
P3_AP0_missing_horizon_count
P3_AP0_missing_by_branch
P3_AP0_missing_by_horizon
P3_AP0_metrics_recomputed_on_common_rows
P3_AP0_metrics_recomputed_on_full_available_rows
same_panel_comparison_valid
```

### 判断标准

P0 pass：

```text
v9420 source_outcome_materialized = 1
v9420 generated source rows = 7020 / 7020
quality_audit_pass = 1
P3_AP0_common_rows >= 780
same_panel_comparison_valid = 1
```

若 P3 AP0 missing rows 系统性影响 metrics，则 route 改为：

```text
R0-AP0BaselineIncompleteNeedsRerun
```

### 可视化

```text
p0_v9420_boundary_ladder.svg
p0_ap0_baseline_completion_heatmap.svg
p0_common_row_vs_all_available_metrics.svg
```

---

## P1：multi-panel source frontier recovery

### 目标

区分“representative panel value-poor”与“full AP0 source frontier absent”。v9.4.3 不再只用一个 stratified panel，而是并行构造多种 source panel。

### Panel candidates

```text
PANEL-RND64: random 64 actions
PANEL-REP64: representative stratified 64 actions
PANEL-REP256: representative stratified 256 actions
PANEL-ORC64: oracle weak-CP diagnostic top64, not official
PANEL-ORC128: oracle weak-CP diagnostic top128, not official
PANEL-LGL64-StateTail: legal state-tail selector top64
PANEL-LGL64-GradientAlignment: legal gradient-alignment selector top64
PANEL-LGL64-AdamWConflictLow: legal AdamW-conflict-low top64
PANEL-LGL64-TailRiskLow: legal tail-risk-low top64
PANEL-LGL64-HybridMonotone: legal monotone source score top64
PANEL-DIV64: diversity-constrained legal panel
```

Legal hybrid source score：

$$
S_{src}(a)=
\alpha_1 Alignment(a)
-\alpha_2 TailRisk(a)
-\alpha_3 AdamWConflict(a)
+\alpha_4 SupportLCB(a)
-\alpha_5 Cost(a).
$$

Constraints：

```text
alpha_i >= 0
feature groups <= 5
thresholds calibrated only on calibration split
no dataset branch
```

### 必须记录

```text
panel_id
panel_type
source_action_count
selector_id
selector_feature_set
uses_outcome_for_selection
uses_dataset_name
PSI_vs_full
KL_vs_full
max_family_gap
max_step_bucket_gap
max_payload_bucket_gap
weak_CP_precision_h20
weak_CP_precision_h80
weak_CP_precision_h240
strong_CP_precision_h20
V_ctrl_lcb_h20
V_ctrl_lcb_all
long_risk_rate_h240
horizon_robust_coverage
support_balance_pass
```

### 判断标准

P1 diagnostic oracle pass：

```text
PANEL-ORC64 weak_CP_precision_h20 >= 0.60
PANEL-ORC64 V_ctrl_lcb_h20 > 0
PANEL-ORC64 long_risk_rate_h240 <= 0.25
```

P1 legal selector weak pass：

```text
any PANEL-LGL64:
  weak_CP_precision_h20 >= 0.30
  V_ctrl_lcb_h20 > 0
  bad_event_h20 <= 0.20
  long_risk_rate_h240 <= 0.30
```

P1 strong pass：

```text
any PANEL-LGL64:
  weak_CP_precision_h20 >= 0.50
  V_ctrl_lcb_h20 > 0
  weak_CP_precision_h240 >= 0.30
  long_risk_rate_h240 <= 0.20
  support_balance_pass = 1
```

### 可视化

```text
p1_panel_value_frontier.svg
p1_oracle_vs_legal_selector_gap.svg
p1_panel_representativeness_vs_value.svg
p1_source_panel_horizon_curve.svg
p1_legal_feature_auc_and_topk_precision.svg
```

---

## P2：same-panel AP0 baseline full branch/horizon rerun

### 目标

对 P1 选出的 panel 统一跑 AP0 same-panel baseline，确保 source quality 判断不是 v9.4.2 P3 partial baseline 的副作用。

### 范围

```text
Panels:
  ORC64, REP64, REP256, best 3 legal panels, RND64
Branches:
  RealAP0, AdamWParallel, bestLR, NoOp, Random, ShuffledAP0Payload
Horizons:
  20, 80, 240
```

### 必须记录

```text
panel_id
source_action_count
expected_rows
actual_rows
branch_completion_rate
horizon_completion_rate
weak_CP_precision_h20/h80/h240
strong_CP_precision_h20/h80/h240
V_ctrl_lcb_h20/h80/h240/all
bad_event_rate_h20/h80/h240
null_rate_h20/h80/h240
long_risk_rate_h240
horizon_robust_action_count
horizon_robust_coverage
```

### 判断标准

P2 pass：

```text
for each selected panel:
  expected_rows = actual_rows
  branch_completion_rate = 1.0
  horizon_completion_rate = 1.0
```

Source survivor：

```text
weak_CP_precision_h20 >= 0.30
V_ctrl_lcb_h20 > 0
long_risk_rate_h240 <= 0.30
```

Source strong survivor：

```text
weak_CP_precision_h20 >= 0.50
V_ctrl_lcb_h20 > 0
weak_CP_precision_h240 >= 0.30
long_risk_rate_h240 <= 0.20
```

### 可视化

```text
p2_ap0_panel_baseline_horizon_curves.svg
p2_ap0_source_value_distribution.svg
p2_source_panel_longrisk_heatmap.svg
p2_source_survivor_table.svg
```

---

## P3：AP0b-AP0f generator damage matrix across panels

### 目标

v9.4.2 只在一个 source panel 上发现 generator damage。v9.4.3 要判断 damage 是否对 oracle-good / legal-good source 也成立。

### 范围

```text
Generators:
  AP0b-LastEdgeLinearizedDescentSource
  AP0c-AdamWResidualOrthogonalSource
  AP0d-TailMarginRepairSource
  AP0e-CurvatureGuardedLowRankEdgeSource
  AP0f-SupportMemorySource
Panels:
  ORC64, REP64, best legal panel, RND64
Horizons:
  20, 80, 240
```

### 必须记录

```text
panel_id
primitive_id
source_action_count
generated_action_count
paired_horizon_count
Damage_mean
Damage_median
Damage_lcb
source_positive_horizon_count
source_positive_lost_after_generation_rate
source_negative_fixed_after_generation_rate
generated_weak_CP_h20/h80/h240
generated_V_ctrl_lcb_h20/all
generated_bad_event_h20
generated_long_risk_h240
payload_cosine_source_generated_mean
payload_norm_ratio_mean
```

### 判断标准

Generator preserve pass：

```text
on ORC64 or legal-source-survivor panel:
  Damage_median >= -0.02
  source_positive_lost_after_generation_rate <= 0.25
  generated_weak_CP_h20 >= 0.70 * source_weak_CP_h20
  generated_long_risk_h240 <= source_long_risk_h240 + 0.05
```

Generator improve pass：

```text
on REP64 or RND64:
  generated_weak_CP_h20 >= source_weak_CP_h20 + 0.10
  generated_V_ctrl_lcb_h20 > source_V_ctrl_lcb_h20
  generated_bad_event_h20 <= source_bad_event_h20
```

If all AP0b-AP0f fail, stop patching AP0b-AP0f.

### 可视化

```text
p3_generator_damage_matrix.svg
p3_source_positive_lost_by_generator.svg
p3_source_to_generated_value_scatter.svg
p3_payload_geometry_vs_damage.svg
```

---

## P4：direct source generator reset AP0g-AP0k

### 目标

如果 legal source selector 找不到 good AP0 source，不能只变换 bad source。需要 direct generator 从当前 state/gradient/tail conflict 中直接生成 source action。

### Direct generators

#### AP0g-GradientAlignedLastEdgeSource

生成方向：

$$
\Delta\theta_{g}= -\eta_g P_{edge}(\nabla_{\theta} CE_{tail}).
$$

证书：

```text
linearized_CE_delta < 0
cos_delta_negative_grad >= tau_cos
payload_norm <= tau_norm
```

#### AP0h-TailMarginConservativeSource

目标：只修 low-margin tail samples，限制 full-batch disturbance。

$$
\Delta\theta_h = \eta_h P_{tail}(\nabla Margin_{p10}) - \lambda P_{nonTail}(\Delta\theta).
$$

证书：

```text
predicted_margin_p10_gain > 0
non_tail_interference <= tau_interfere
horizon_tail_risk_ucb <= tau_h
```

#### AP0i-AdamWResidualBlendSource

目标：functional action 只补 AdamW 不覆盖的 residual direction。

$$
\Delta\theta_i = \Pi_{\perp \Delta\theta_{AdamW}}(\Delta\theta_{tail}) + \rho \Delta\theta_{AdamW}.
$$

证书：

```text
cos_delta_adamw in [tau_low, tau_high]
adamw_conflict <= tau_conflict
projected_descent < 0
```

#### AP0j-LowRankEdgeSafeSource

目标：限制 action 到 low-rank edge subspace，降低 long-horizon risk。

$$
\Delta W = U_r S_r V_r^T,
\quad r \le r_{max}.
$$

证书：

```text
rank <= r_max
spectral_norm <= tau_spec
curvature_guard <= tau_curv
```

#### AP0k-NoOpGuardedMicroSource

目标：小幅度可撤销 update，只作为 safe source existence test。

$$
\Delta\theta_k = \epsilon \cdot sign(-\nabla CE_{tail}) \odot mask_{edge}.
$$

证书：

```text
norm_ratio <= tau_micro
linearized_CE_delta < 0
predicted_bad_ucb <= tau_b
```

### 必须记录

```text
primitive_id
generator_id
action_count
payload_hash_missing
certificate_hash_missing
action_apply_error_linf_max
action_apply_cosine_min
h20_weak_CP_precision
h20_strong_CP_precision
h20_bad_event_rate
h20_null_rate
h20_V_ctrl_lcb
h80_weak_CP_precision
h240_weak_CP_precision
h240_long_risk_rate
horizon_robust_coverage
cost_ms_q90
```

### 判断标准

Direct generator weak pass：

```text
h20_weak_CP_precision >= 0.30
h20_V_ctrl_lcb > 0
h20_bad_event_rate <= 0.20
h240_long_risk_rate <= 0.30
action_apply_error_linf_max <= 1e-6
```

Direct generator strong pass：

```text
h20_weak_CP_precision >= 0.50
h80_weak_CP_precision >= 0.35
h240_weak_CP_precision >= 0.25
h240_long_risk_rate <= 0.20
horizon_robust_coverage >= 0.05
cost_ms_q90 <= 0.20
```

### 可视化

```text
p4_direct_generator_value_frontier.svg
p4_direct_generator_horizon_curves.svg
p4_direct_generator_cost_vs_value.svg
p4_direct_generator_bad_null_longrisk.svg
```

---

## P5：effect-valid certificate redesign

### 目标

v9410 certificate AUC 接近随机。v9.4.3 的 certificate 必须预测 effect，而不是 construction validity。

### Certificate candidates

```text
CERT0-v9410-reference
CERT1-LinearizedEffectCert
CERT2-TailMarginEffectCert
CERT3-AdamWConflictEffectCert
CERT4-HorizonRiskEffectCert
CERT5-MinimalHybridEffectCert
```

Minimal hybrid score：

$$
CertScore(a)=
\beta_1(-\widehat{\Delta CE}_{lin})
+\beta_2\widehat{\Delta Margin}_{tail}
-\beta_3 AdamWConflict
-\beta_4 TailRiskUCB
-\beta_5 Cost.
$$

Constraints：

```text
beta_i >= 0
feature groups <= 5
commit-time available = 1
no outcome at commit
```

### 必须记录

```text
certificate_id
primitive_id
feature_groups
commit_time_available
uses_outcome_at_commit
AUC_weak_CP
AUC_strong_CP
AUC_longrisk
P_weak_CP_given_cert_pass
P_weak_CP_given_cert_fail
P_longrisk_given_cert_pass
P_longrisk_given_cert_fail
Lift_weak
Lift_longrisk
monotone_sign_pass
calibration_to_heldout_drift
feature_cost_ms_q90
```

### 判断标准

P5 weak pass：

```text
AUC_weak_CP >= 0.70
Lift_weak >= 2.0
AUC_longrisk >= 0.65
Lift_longrisk <= 0.80
monotone_sign_pass = 1
feature_cost_ms_q90 <= 0.20
```

P5 strong pass：

```text
AUC_weak_CP >= 0.78
AUC_longrisk >= 0.75
P_weak_CP_given_cert_pass >= 0.40
P_longrisk_given_cert_pass <= 0.15
calibration_to_heldout_drift <= 0.07
```

### 可视化

```text
p5_certificate_auc_bar.svg
p5_cert_pass_vs_fail_cp_longrisk.svg
p5_certificate_calibration_curve.svg
p5_certificate_ablation_waterfall.svg
```

---

## P6：minimal source certificate controller

### 目标

只在 P1/P4/P5 产生 survivor 后运行。controller 不能补救 bad source/generator；它只能选择已经有正向 frontier 的 candidates。

### Controller form

$$
Accept(a)=1
\iff
LCB(V_{h20}(a))>0
\land
UCB(Bad_{h20}(a))\le \tau_b
\land
UCB(LongRisk_{h240}(a))\le \tau_h
\land
CertScore(a)\ge \tau_c
\land
Cost(a)\le C_{max}.
$$

### 必须记录

```text
controller_id
primitive_id
certificate_id
source_panel_id
calibration_split_id
heldout_split_id
accepted_count_cal
accepted_count_heldout
coverage_heldout
weak_CP_precision_h20_heldout
strong_CP_precision_h20_heldout
bad_event_h20_heldout
long_risk_h240_heldout
V_ctrl_lcb_h20_heldout
horizon_robust_coverage_heldout
precision_lcb
bad_event_ucb
long_risk_ucb
accepted_family_count
accepted_signal_strata_count
max_family_share
max_stratum_share
uses_dataset_name
uses_outcome_at_commit
```

### 判断标准

P6 weak controller pass：

```text
coverage_heldout >= 0.03
weak_CP_precision_h20_heldout >= 0.50
bad_event_h20_heldout <= 0.15
long_risk_h240_heldout <= 0.25
V_ctrl_lcb_h20_heldout > 0
support_balance_pass = 1
```

P6 official-prep pass：

```text
coverage_heldout in [0.03, 0.15]
weak_CP_precision_h20_heldout >= 0.75
bad_event_h20_heldout <= 0.05
long_risk_h240_heldout <= 0.15
horizon_robust_coverage_heldout >= 0.03
precision_lcb >= 0.75
bad_event_ucb <= 0.05
```

### 可视化

```text
p6_controller_precision_coverage_longrisk_frontier.svg
p6_calibration_to_heldout_drift.svg
p6_controller_support_balance.svg
p6_controller_oracle_overlap.svg
```

---

## P7：selected source online runtime

### 目标

只有 P6 pass 后测 selected runtime。不能对 failed controller 测 runtime 并写 system progress。

### 必须记录

```text
runtime_candidate_id
controller_id
primitive_id
certificate_id
selected_action_count
step_count
active_step_count
zero_candidate_controller_kernel_count
controller_launches_per_active_step_q90
feature_compute_time_ms_q90
certificate_compute_time_ms_q90
score_accept_time_ms_q90
payload_apply_time_ms_q90
base_train_step_time_ms_q90
total_step_time_ms_q90
step_ratio_q90
memory_ratio
payload_apply_error_linf_max
payload_apply_cosine_min
```

### 判断标准

P7 pass：

```text
zero_candidate_controller_kernel_count = 0
controller_launches_per_active_step_q90 <= 2
feature_compute_time_ms_q90 + certificate_compute_time_ms_q90 <= 0.20
payload_apply_time_ms_q90 <= 0.20
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
payload_apply_error_linf_max <= 1e-6
```

### 可视化

```text
p7_selected_runtime_component_waterfall.svg
p7_step_ratio_distribution.svg
p7_payload_apply_error_hist.svg
```

---

## P8：Base-Acc Sentinel extended, not controller tuning

### 目标

回应“是否在数据集上训练、acc 如何、和 MLP 比如何”。继续运行固定配置健康监控，但严格隔离，不用于 source selector / generator / certificate。

### 设置

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2,3,4
models:
  LQ-t2-h256
  MatchedMLP
  QuadraticFeatureMLP
  AdamWStrongLRGridMLP diagnostic
budgets:
  same steps
  same batch schedule
  same optimizer family
  no dataset-specific tuning
```

### 必须记录

```text
dataset
seed
model_id
train_acc
val_acc
test_acc
train_loss
val_loss
test_loss
ECE
NLL
CEp99
margin_p10
wallclock_sec
step_time_ms_q90
memory_ratio
same_seed_schedule
same_budget
hyperparams_fixed_before_run
dataset_specific_tuning
base_acc_used_for_controller
```

### 判断标准

Sentinel healthy：

```text
sentinel_complete = 1
LQ_catastrophic_fail = 0
base_acc_used_for_controller = 0
```

Base comparison diagnostic：

```text
LQ_minus_MLP_mean_test_acc reported
LQ_minus_QuadraticFeatureMLP_mean_test_acc reported
no selector/controller threshold changed based on this result
```

### 可视化

```text
p8_base_acc_by_dataset_seed.svg
p8_lq_vs_mlp_acc_gap.svg
p8_lq_vs_quadratic_mlp_gap.svg
p8_base_calibration_ece.svg
```

---

## P9：system integration boundary

### 目标

组合 P6 selected controller 与 P7 runtime，判定是否可以进入 LDO/LSO 和 paired replay。

### 必须记录

```text
system_candidate_id
controller_id
primitive_id
certificate_id
runtime_candidate_id
decision_gate_pass
runtime_gate_pass
source_lifecycle_pass
outcome_materializer_pass
certificate_effect_valid_pass
base_acc_sentinel_complete
base_acc_used_for_controller
official_eligible
system_legal_controller_pass
reason_if_fail
```

### 判断标准

System pass：

```text
decision_gate_pass = 1
runtime_gate_pass = 1
source_lifecycle_pass = 1
certificate_effect_valid_pass = 1
base_acc_used_for_controller = 0
official_eligible = 1
system_legal_controller_pass = 1
```

If fail, downstream remains not_run.

### 可视化

```text
p9_system_gate_dashboard.svg
p9_decision_runtime_joint_frontier.svg
```

---

## P10：leave-dataset-out / leave-stratum-out boundary

### 目标

只在 P9 pass 后执行。验证 selector/generator/certificate 不是 dataset-specific artifact。

### 必须记录

```text
split_type
heldout_dataset_or_stratum
controller_id
primitive_id
certificate_id
accepted_count
coverage
weak_CP_precision_h20
bad_event_h20
long_risk_h240
V_ctrl_lcb_h20
step_ratio_q90
dataset_name_used
```

### 判断标准

LDO weak pass：

```text
all heldout datasets coverage > 0
at least 2/3 heldout datasets weak_CP_precision_h20 >= 0.50
bad_event_h20 <= 0.15
long_risk_h240 <= 0.30
```

LDO strong pass：

```text
all heldout datasets weak_CP_precision_h20 >= 0.75
bad_event_h20 <= 0.05
long_risk_h240 <= 0.15
```

### 可视化

```text
p10_leave_dataset_matrix.svg
p10_leave_stratum_matrix.svg
p10_dataset_tuning_audit.svg
```

---

## P11：official paired replay boundary

### 目标

只在 P9/P10 pass 后打开。验证 selected functional source update 是否有 causal advantage。

### Branches

```text
RealFunctional
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledPayload
ShuffledCertificate
ShuffledSelectorScore
ShuffledTailMask
ShuffledHorizonRisk
```

### 必须记录

```text
dataset
seed
horizon
branch
controller_id
primitive_id
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
real_beats_adamwparallel
real_beats_bestlr
real_beats_noop
real_beats_random
shuffle_control_fail
```

### 判断标准

Paired replay pass：

```text
RealFunctional beats AdamWParallel >= 0.60 macro
RealFunctional beats bestLR >= 0.60 macro
shuffle controls fail
Acc_real >= Acc_AdamW - 0.005 on every slice
```

### 可视化

```text
p11_paired_replay_branch_matrix.svg
p11_real_vs_control_delta_distribution.svg
p11_shuffle_control_failure_dashboard.svg
```

---

## P12：short/full training and MLP comparison boundary

### 目标

只有 P11 pass 后进入。回答真正的“和 MLP 比如何”。

### 设置

```text
models:
  LQ-t2-h256 + selected functional system
  LQ-t2-h256 AdamW-only
  MatchedMLP
  StrongLRGridMLP
  QuadraticFeatureMLP

datasets:
  MNIST, Fashion-MNIST, KMNIST

metrics:
  final test acc
  best val acc
  ValLossAUC_step
  ValLossAUC_time
  time_to_target
  steps_to_target
  ECE
  NLL
  CEp99
  margin_p10
  robustness perturbation
  continual retention if enabled
```

### 判断标准

Short-run pass：

```text
Functional LQ >= AdamW-only LQ - 0.005 on all datasets
Functional LQ beats matched MLP on at least 2/3 datasets
ValLossAUC_step improves vs AdamW-only LQ
```

Full-run pass：

```text
Functional LQ mean test acc > MatchedMLP mean test acc with CI lower bound > 0
Functional LQ not worse than StrongLRGridMLP by more than 0.005 unless sample-efficiency / calibration advantage is significant
ECE or CEp99 improves vs matched MLP
step_ratio_q90 <= 1.50
```

### 可视化

```text
p12_acc_by_dataset_seed.svg
p12_val_loss_auc_step_time.svg
p12_time_to_target.svg
p12_calibration_ece_nll.svg
p12_lq_functional_vs_mlp_summary.svg
```

---

# 8. 并行执行计划

v9.4.3 必须避免“一轮一个 blocker”。建议并行如下：

## Batch A：materializer / baseline preflight

```text
A1 single action × single branch × h20 preflight
A2 8 actions × all branches × h20 preflight
A3 8 actions × all branches × all horizons preflight
A4 selected panels full rows
```

Kill gate：

```text
任何阶段 actual rows = 0 立即停止该 branch，不跑大 runner。
```

## Batch B：multi-panel AP0 baseline

并行跑：

```text
RND64
REP64
ORC64
best legal top64 panels
```

输出 P1/P2 panel source value dashboard。

## Batch C：generator matrix

并行跑：

```text
AP0b-AP0f on ORC64
AP0b-AP0f on REP64
AP0b-AP0f on best legal panel
AP0g-AP0k direct generator smoke
```

输出 damage matrix 和 direct generator frontier。

## Batch D：certificate matrix

只对 Batch C survivor 跑：

```text
CERT1-CERT5
calibration/heldout split
component ablation
```

## Batch E：Base-Acc Sentinel

并行独立运行，不影响 selector/controller：

```text
LQ-t2-h256
MatchedMLP
QuadraticFeatureMLP
StrongLRGridMLP diagnostic
```

## Batch F：runtime microbench

只对 P6 selected survivor 运行；如果 P6 没 survivor，runtime 不转 official。

---

# 9. Route decision

v9.4.3 结束时必须落入以下之一：

```text
R1-SourceOutcomeMaterializerRegression
  7020-row closure 回退或 quality audit fail。

R2-FullAP0OracleSourceFrontierAbsent
  oracle panel 在 same protocol 下也无 positive source frontier。

R3-LegalSourceSelectorOpaqueButOracleSourceExists
  oracle source panel strong，但 legal selector 找不到。

R4-AP0bAP0fTransformDamagePrimary
  source panel good，但 AP0b-AP0f generator 严重破坏 source positives。

R5-DirectGeneratorImmediatePassCertificateFail
  direct generator h20 positive，但 certificate 不可用。

R6-DirectGeneratorLongRiskFail
  h20 pass，但 h240 long-risk 高。

R7-EffectCertificateControllerPassRuntimeFail
  decision pass，但 selected runtime 不过。

R8-SystemLegalSourceControllerPass
  decision + certificate + runtime 全过，可进入 LDO/paired replay。

R9-BaseAccRegression
  Base-Acc Sentinel catastrophic fail，需先修 base/training health。
```

---

# 10. 最终成功条件

v9.4.3 的最低有效成功：

```text
1. multi-panel source frontier 被量化；
2. representative panel 与 value-source panel 的区别被证明；
3. 至少一个 legal source selector 或 direct generator 被真实测量；
4. AP0b-AP0f damage 是否 primary 被判清；
5. certificate 是否 effect-valid 被判清；
6. Base-Acc Sentinel 继续完整且隔离；
7. route 不再停在 “row sink / materializer missing”。
```

v9.4.3 的强成功：

```text
1. legal source selector 或 direct generator 产生 h20-positive source frontier；
2. h240 long-risk 可控；
3. effect certificate 有正 lift；
4. minimal controller heldout 过 weak decision gate；
5. selected runtime step_ratio_q90 <= 1.50；
6. system_legal_controller_pass = 1。
```

---

# 11. 本计划的核心原则

v9.4.3 的核心不是“多试几个 generator”。核心是把问题拆成四个不可混淆的对象：

```text
1. Source selector 是否能找到好 source；
2. Generator 是否保留或创造 value；
3. Certificate 是否预测 effect；
4. Controller/runtime 是否能 legal deploy。
```

只要其中任一个上游对象没过，就不要把下游写成 success。

最终判断公式：

$$
\boxed{
\text{Functional success}
\neq
\text{generator materialized}
\neq
\text{outcome rows materialized}
\neq
\text{base acc sentinel healthy}
}
$$

必须同时满足：

$$
\boxed{
\text{legal source frontier}
+
\text{effect-valid certificate}
+
\text{selected controller}
+
\text{system runtime}
+
\text{causal replay advantage}
}
$$
