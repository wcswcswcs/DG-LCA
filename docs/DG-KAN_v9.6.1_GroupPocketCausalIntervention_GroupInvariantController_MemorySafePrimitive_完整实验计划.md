# DG-KAN v9.6.1 Group-Pocket Causal Intervention / Group-Invariant Controller / Memory-Safe Primitive 完整实验计划

> 本计划基于 v9.6.0 `Group-Pocket Deconfounded Rank / Memory-Safe LongRisk Primitive` 的真实执行结果制定。  
> v9.6.0 的 route 是：
>
> ```text
> route = R1-GroupPocketMechanismUnresolved
> base_candidate = LQ-t2-h256
> success_v9600_strict_purekan_functional = False
> success_v9600_full_functional = False
> success_v9600_external_ready = False
> ```
>
> v9.6.1 不再做 `GIR9 / RC7 / APGD2` 阈值小修，也不继续把 pooled TopK 当作进展。  
> 本轮的核心问题是：
>
> $$
> \boxed{
> \text{好动作是否真的只存在于 payload0 这个局部口袋？还是我们的 ranker 只学到了一个局部口袋？}
> }
> $$
>
> 以及：
>
> $$
> \boxed{
> \text{能否把现有 strong TopK signal 转成跨 group 稳定、可上线的 accepted region？}
> }
> $$

---

# 0. v9.6.0 结果的独立判断

v9.6.0 有真实进展，但不是 functional success。它推进了三件事：

```text
1. group pocket autopsy 真实运行；
2. two-score / pairwise rank 继续显示强 TopK diagnostic；
3. APGD1-APGD8 工程链路和 branch-horizon replay 完整闭合。
```

但它也暴露了更深的问题：

```text
1. payload0 pocket drop 很强，但机制没有被解释清楚；
2. matched-pair evidence 完全缺失；
3. ranker 的 TopK 好看，但 group leaveout 仍不稳；
4. certificate 从 TopK 转 accepted region 后崩；
5. APGD 生成动作仍 value-negative / high-longrisk。
```

v9.6.0 的关键数字：

```text
P1 group pocket:
  dominant_axis = payload_norm_bucket
  dominant_group = payload0
  max_drop = 0.8505747126436781
  payload_norm_bucket_drop_explained_fraction = 0.18181818181818182
  failure_attribution_fraction = 0.03689567430025445
  matched_pair_count = 0
  p1_pass = 0

P2 target:
  best = ValuePositiveNoLongRisk
  count = 97
  coverage = 0.03372739916550765
  positive_group_coverage = 1.0
  official_density_pass = 0

P4 two-score ranker:
  best = VR4-PairwiseValue-RiskVeto
  TopK87 GradeAB precision = 0.8390804597701149
  V LCB = 0.13479400136362346
  longrisk UCB = 0.0
  leaveout drop = 0.3448275862068965
  max group share = 1.0

P5 pairwise ranker:
  best = GIR9-PairwiseWithinGroupRanker
  TopK87 precision = 0.8505747126436781
  V LCB = 0.15630165181090255
  longrisk UCB = 0.03389313880385762
  LDO / LSO drop = 0.3563218390804597 / 0.3563218390804597

P6 certificate:
  best = RC7-TopKFixedCountGroupBalanced
  accepted = 87
  coverage = 0.10046189376443418
  GradeAB precision = 0.28735632183908044
  V LCB = 0.03544770490685384
  longrisk UCB = 0.369780988566077

P13 APGD:
  best = APGD2-MemorySafeResidualBlend
  GradeAB precision = 0.0625
  V LCB = -0.875224386072408
  h240 longrisk UCB = 0.8560881119635937
  longrisk created rate = 0.75
```

我的独立判断是：

$$
\boxed{
\text{v9.6.0 证明 rank signal 不是完全没有；但它仍是局部口袋，不是跨 group 稳定规则。}
}
$$

这不是“没进展”。它把问题从“有没有 rank signal”推进到了“rank signal 为什么只在某些 group 里成立”。这一步很重要，因为我们的终极目标不是在某个 payload bucket、某个 dataset、某个 stratum 上挑好动作，而是做一个 dataset-agnostic、group-stable、system-legal 的 functional update。

---

# 1. v9.6.1 总体目标

v9.6.1 的总目标不是继续提高 pooled TopK，也不是调 APGD2，而是回答三个第一性问题：

```text
Q1: payload0 pocket 是真实因果机制，还是 ranker / sampling / score distribution 造成的局部现象？
Q2: 如果现有动作里确实存在可用区域，能否构造 group-stable accepted region？
Q3: 如果 existing-action route 仍不稳，能否生成 memory-safe / offdiag-safe / longrisk-safe 的新动作？
```

本轮强目标：

$$
GroupStableControllerPass = 1
$$

$$
AcceptedCount \ge 87
$$

$$
Precision_{GradeAB,heldout} \ge 0.75
$$

$$
V^{LCB}_{integrated,heldout} > 0
$$

$$
LongRisk^{UCB}_{h240,heldout} \le 0.05
$$

$$
LDOdrop_{max} \le 0.10
$$

$$
LSOdrop_{max} \le 0.10
$$

$$
StepRatio_{q90} \le 1.50
$$

最低有效推进目标：

```text
1. group pocket 机制被解释，或者明确判定为不可解释 confounded pocket；
2. matched-pair / intervention panel 不再为 0；
3. existing-action ranker 的 max group share 降到 <= 0.35；
4. ranker 的 leaveout drop 从 0.35 降到 <= 0.20；
5. certificate 的 accepted region longrisk UCB 从 0.37 降到 <= 0.12；
6. APGD / APGE 类生成动作至少不再制造 0.75 级别 longrisk；
7. 若 controller 仍不过，明确路线停在 group-invariant rule absent，而不是继续未知状态。
```

---

# 2. 关键原则

## 2.1 不按数据集调参

允许诊断：

```text
per dataset
per seed
per stratum
per family
per payload_norm_bucket
per step_bucket
per hard_tail_bucket
per source primitive
```

不允许 official controller 使用：

```text
dataset_name branch
seed branch
manual per-dataset threshold
manual per-dataset topK
validation/test labels
future outcome
outcome-derived score
old table result
```

## 2.2 不再只看 AUC

v9.6.0 以前多次出现：

```text
AUC 很高，但 TopK / accepted region 失败。
```

因此 v9.6.1 所有 rank / certificate 必须同时报告：

```text
AUC
TopK64 / TopK87 precision
accepted region precision
V LCB
longrisk UCB
bad/null UCB
max group share
worst group precision
LDO / LSO drop
coverage
calibration ECE
```

AUC 只能作为辅助指标，不能作为 pass 条件。

## 2.3 好动作必须服务终极目标

好动作不是 acc 短期上涨，而是：

```text
短期有用；
中期不坏；
长期安全；
不破坏旧 family / old stratum；
不造成 cover collapse；
不走 noise / reservoir 方向；
能在训练当下被合法证据选中；
成本接近 MLP。
```

对应向量：

$$
Q(a)=
(V_{20},V_{80},V_{240},Bad,Null,LongRisk,MemoryRisk,CoverRisk,SignalScore,Cost).
$$

---

# 3. 核心假设

## H1：payload0 pocket 可能是真实机制，但目前证据不足

v9.6.0 显示：

```text
payload0_removal_precision_drop = 0.8505747126436781
within_payload0_precision = 0.8505747126436781
cross_payload_precision = 0.0
```

但同时：

```text
payload_norm_bucket_drop_explained_fraction = 0.18181818181818182
failure_attribution_fraction = 0.03689567430025445
matched_pair_count = 0
```

所以不能直接说：

```text
payload0 是因果机制。
```

H1 成立标准：

```text
matched_pair_count >= 128
or interventional_payload_norm_panel_count >= 256
payload0 causal lift sign-consistent across >= 4 axes
payload0 removal precision drop remains >= 0.50 after confound adjustment
within-payload0 precision >= 0.75
cross-payload rescued precision >= 0.30 after normalized intervention
```

H1 失败标准：

```text
matched / intervention 后 payload0 lift 消失；
payload0 只由 step/family/source/hash availability confound 解释；
payload0 不是机制，而是 convenience pocket。
```

## H2：existing-action legal rank 是最接近成功的路线，但必须 group-stable

H2 成立标准：

```text
TopK87 GradeAB precision >= 0.75
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.05
max_group_share <= 0.35
LDO_drop_max <= 0.10
LSO_drop_max <= 0.10
worst_group_precision >= 0.60
```

H2 失败标准：

```text
TopK 很强但 max group share 仍 > 0.50；
LDO/LSO drop 仍 > 0.20；
heldout accepted region precision < 0.75；
longrisk UCB > 0.10。
```

## H3：rank-safe certificate 失败不是校准小问题，而是 accepted region 形状错

H3 成立标准：

```text
TopK diagnostic good but frozen accepted region bad；
accepted region GradeAB precision < 0.75；
accepted region longrisk UCB > 0.10；
threshold perturbation 不改变结论。
```

H3 失败标准：

```text
通过 group-conformal calibration 后 accepted region 过线。
```

## H4：APGD 失败主因是 longrisk / offdiag / memory 没有被生成器真正约束

H4 成立标准：

```text
APGD/APGA generated actions longrisk_created_rate >= 0.50
population_risk_offdiag_fail 与 longrisk 绑定
memory-safe value-positive density < 0.03
source-to-generated damage V LCB < 0
```

H4 失败标准：

```text
APGE 新 primitive 通过 memory/offdiag constraints 后产生 GradeAB precision >= 0.25、longrisk UCB <= 0.20 的 early frontier。
```

## H5：好几何结构要从 group 口袋转为 group 不变规则

Deep Manifold 视角下，网络训练是在移动局部 cover、改变坐标、积累或释放曲率，并逐渐形成稳定区域。若一个规则只在 payload0 生效，就更像局部 cover 的 pocket，而不是全局稳定的训练几何。Generalization 视角下，动作要走 signal channel，而不是只在训练误差或局部 pocket 里有效的 reservoir/noise 方向。

H5 成立标准：

```text
rank score 在不同 payload / dataset / stratum / family 里方向一致；
value score 与 longrisk veto 可以分离；
memory/offdiag-safe 条件能解释 longrisk；
accepted region 中 old-family failure 和 old-stratum failure 显著低于 rejected region。
```

---

# 4. 实验设计总览

v9.6.1 分成 15 个阶段。前 8 个阶段可以并行跑，避免继续“一轮只清一个 blocker”。

```text
P0  boundary reproduction
P1  group pocket intervention panel
P2  matched-pair and causal attribution
P3  group-balanced target and support map v2
P4  legal feature deconfounding v3
P5  value-rank + risk-veto + memory-veto ranker
P6  group-invariant pairwise ranker v3
P7  rank-safe certificate v9
P8  existing-action minimal controller
P9  selected runtime preflight
P10 APGA/APGD OOD causal autopsy v2
P11 APGE1-APGE8 memory/offdiag-safe primitive
P12 APGE branch-horizon outcome
P13 APGE geometry / damage / longrisk audit
P14 system gate and leaveout boundary
P15 paired replay / short-full boundary
```

---

# 5. P0：v9.6.0 boundary reproduction

## 目标

确认 v9.6.1 没有跳过 v9.6.0 的失败边界。

## 必须记录

```text
source_route_v9600
system_legal_controller_pass_v9600
p1_group_pocket_pass_v9600
payload_norm_bucket_drop_explained_fraction_v9600
payload0_removal_precision_drop_v9600
matched_pair_count_v9600
best_two_score_ranker_v9600
best_two_score_TopK87_precision_v9600
best_two_score_leaveout_drop_v9600
best_pairwise_ranker_v9600
best_pairwise_TopK87_precision_v9600
best_pairwise_LDO_drop_v9600
rank_safe_certificate_pass_v9600
best_apgd_primitive_v9600
best_apgd_gradeab_precision_v9600
best_apgd_V_lcb_v9600
best_apgd_longrisk_ucb_v9600
```

## 判断标准

P0 pass：

```text
source_route_v9600 = R1-GroupPocketMechanismUnresolved
system_legal_controller_pass_v9600 = 0
matched_pair_count_v9600 = 0
best_pairwise_LDO_drop_v9600 = 0.3563218390804597
best_apgd_V_lcb_v9600 < 0
```

## 可视化

```text
p0_v9600_boundary_summary.svg
p0_ranker_progress_v9580_v9590_v9600.svg
p0_generator_failure_progress_apga_apgd.svg
```

---

# 6. P1：group pocket intervention panel

## 目标

不要继续只做 observational autopsy。v9.6.1 必须制造可比较的 intervention panel，直接问：

```text
payload_norm_bucket / payload0 到底是不是产生好动作的机制？
```

## 实验对象

选择以下 source actions：

```text
A. payload0 high-rank top candidates
B. payload0 low-rank candidates
C. non-payload0 matched high-rank candidates
D. non-payload0 random matched candidates
E. value-positive no-longrisk targets
F. high-longrisk negatives
```

每个 source action 生成三类 clone：

```text
1. no-transform clone：验证 replay 等价；
2. norm-normalized clone：保持方向，改变 norm bucket；
3. direction-preserved small-scale clone：保持方向，缩小到 safe norm；
4. direction-shuffled negative control：保持 norm，打乱方向；
5. memory-preserving projection clone：投影掉 old-family high-risk 分量。
```

这些 clone 只能用于机制诊断，不能直接 promotion 成 official controller，除非后续 P8/P14 过 gate。

## 必须记录

```text
source_action_id
clone_action_id
clone_type
source_payload_norm_bucket
clone_payload_norm_bucket
source_payload_norm
clone_payload_norm
payload_direction_cosine_source_clone
payload_linf_error_no_transform
state_before_hash
branch_config_hash
horizon_config_hash
memory_projection_applied
offdiag_projection_applied
branch_horizon_rows_expected
branch_horizon_rows_actual
V20_ctrl
V80_ctrl
V240_ctrl
V_integrated
GradeAB_label
LongRisk_h240
bad_event
null_event
old_family_fail
old_stratum_fail
population_risk_offdiag_fail
cover_entropy_delta
basis_effective_rank_delta
```

## 判断标准

P1 pass：

```text
intervention_clone_count >= 512
no_transform_metric_abs_diff_max = 0
branch_horizon_completion = 1
matched source/clone rows >= 512
payload_norm_bucket_intervention_effect_measured = 1
negative_control_divergence_present = 1
```

H1 strong pass：

```text
payload0 -> nonpayload0 normalized clone retains GradeAB precision drop <= 0.15
or nonpayload0 -> payload0 normalized clone gains GradeAB lift >= 0.30
and longrisk does not increase above 0.10
```

H1 falsified：

```text
payload0 effect disappears after source/action/family/step matching;
normalized clones do not reproduce payload0 lift;
matched-pair lift sign inconsistent.
```

## 可视化

```text
p1_payload_bucket_intervention_flow.svg
p1_norm_bucket_counterfactual_lift.svg
p1_source_clone_value_delta.svg
p1_payload_norm_vs_longrisk_by_clone_type.svg
p1_negative_control_divergence.svg
```

---

# 7. P2：matched-pair causal attribution

## 目标

v9.6.0 matched_pair_count = 0。v9.6.1 必须用更宽松但仍合法的 matching，或者用 P1 intervention clone 构造可比对的 pair。

## Matching 层级

按优先级：

```text
Level 1 exact:
  same dataset, seed, family, step_bucket, horizon_source, score_bucket, hard_tail_bucket, AdamWConflict bucket

Level 2 coarsened exact:
  same family, step_bucket, hard_tail_bucket, value_rank_bucket, memory_risk_bucket

Level 3 propensity:
  propensity score over legal pre-commit features, caliper <= 0.05

Level 4 intervention:
  source/clone pair from P1
```

不能用 outcome labels 参与 matching。

## 必须记录

```text
pair_id
pair_level
source_action_id_A
source_action_id_B
group_A
group_B
matching_features
propensity_A
propensity_B
propensity_gap
score_A
score_B
GradeAB_A
GradeAB_B
V_integrated_A
V_integrated_B
LongRisk_A
LongRisk_B
old_family_fail_A/B
population_risk_offdiag_fail_A/B
pair_lift_gradeab
pair_lift_V
pair_lift_longrisk
sign_test_p_value
bootstrap_ci_lift
```

## 判断标准

P2 pass：

```text
matched_pair_count_total >= 256
exact_or_coarsened_pair_count >= 64
intervention_pair_count >= 128
mean_V_lift_CI_low > 0 for proposed mechanism
longrisk_lift_CI_high <= 0.05
sign_test_p_value <= 0.05
```

P2 fail：

```text
matched_pair_count_total < 128
or V lift not positive
or longrisk lift high
or pair results disagree with P1 intervention.
```

## 可视化

```text
p2_matching_balance_love_plot.svg
p2_pair_lift_forest.svg
p2_payload0_vs_counterfactual_pairs.svg
p2_propensity_overlap.svg
```

---

# 8. P3：group-balanced target and support map v2

## 目标

`ValuePositiveNoLongRisk` 有 weak density：count = 97，coverage = 0.0337，但 official density pass = 0。P3 要搞清楚它为什么不能 official：是 group min density、bad/null/support、还是某些 axes 支持不足。

## Target 候选

```text
T1: GradeAB
T2: ValuePositiveNoLongRisk
T3: MemorySafeValuePositiveNoLongRisk
T4: CoverStableValuePositiveNoLongRisk
T5: OffdiagSafeValuePositiveNoLongRisk
T6: GradeAB + MemorySafe
T7: GradeAB + OffdiagSafe
T8: ValuePositiveNoLongRisk + max_group_share constraint
T9: SoftGrade score quantile target
T10: Multi-grade accepted target A/B/C with risk veto
```

## 必须记录

```text
target_id
action_count
coverage
global_precision
V_integrated_mean
V_integrated_LCB
h20_V_LCB
h80_V_LCB
h240_V_LCB
h240_longrisk_rate
h240_longrisk_UCB
bad_event_rate
bad_event_UCB
null_rate
null_UCB
memory_fail_rate
cover_collapse_rate
offdiag_fail_rate
positive_group_coverage_by_dataset
positive_group_coverage_by_payload_bucket
positive_group_coverage_by_family
positive_group_coverage_by_stratum
min_group_density
max_group_share
worst_group_precision
```

## 判断标准

P3 official target pass：

```text
action_count >= 87
coverage >= 0.03
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.05
bad_event_UCB <= 0.05
null_UCB <= 0.15
positive_group_coverage_by_dataset = 1
positive_group_coverage_by_payload_bucket >= 0.80
positive_group_coverage_by_family >= 0.60
max_group_share <= 0.35
```

Weak pass：

```text
action_count >= 87
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.10
positive_group_coverage_by_dataset = 1
```

## 可视化

```text
p3_target_density_lattice.svg
p3_target_group_support_heatmap.svg
p3_value_risk_memory_target_frontier.svg
p3_grade_target_overlap_upset.svg
```

---

# 9. P4：legal feature deconfounding v3

## 目标

`ControlTransferImprovement` 的 within-group AUC 很高，但 longrisk UCB 高。P4 要把 value signal、risk signal、memory signal 分开。

## Feature groups

```text
Group V: value signal
  ControlTransferImprovement
  projected_CE_descent
  margin_improvement_estimate
  train_to_control_transfer_delta

Group R: longrisk signal
  longrisk_proxy_tail
  h240_tail_margin_proxy
  hard_tail_fraction_change_proxy

Group M: memory signal
  old_family_similarity
  old_stratum_gradient_conflict
  forget_risk_proxy
  population_risk_offdiag_proxy

Group C: cover signal
  basis_effective_rank_proxy
  cover_entropy_proxy
  payload_locality_entropy

Group S: signal/reservoir
  batch_gradient_mean_square
  batch_gradient_variance
  snr_gate_score
  offdiag_agreement_score

Group Cost:
  feature_ms_q90
  payload_apply_estimate
  action_shape_id
```

## 必须记录

```text
feature_id
feature_group
legal_green_or_red
uses_outcome_derived_field
uses_dataset_name
uses_validation_test
uses_future_outcome
within_group_auc_mean
within_group_auc_min
TopK64_precision_global
TopK64_precision_group_balanced
TopK87_precision_group_balanced
TopK64_V_LCB
TopK64_longrisk_UCB
TopK64_memory_fail_UCB
TopK64_cover_fail_UCB
worst_group_precision
max_group_share
LDO_drop
LSO_drop
feature_cost_q90_ms
```

## 判断标准

P4 pass：

```text
best legal feature group combination has:
TopK87_precision_group_balanced >= 0.60
V_LCB > 0
longrisk_UCB <= 0.10
max_group_share <= 0.40
LDO_drop <= 0.20
LSO_drop <= 0.20
feature_cost_q90_ms <= budget
```

P4 fail：

```text
feature signal remains high-AUC / high-risk;
or group-balanced precision < 0.50;
or LDO/LSO drop > 0.30.
```

## 可视化

```text
p4_feature_group_auc_topk_matrix.svg
p4_value_vs_longrisk_feature_tradeoff.svg
p4_feature_group_drop_by_axis.svg
p4_legal_field_red_green_ledger.svg
```

---

# 10. P5：value-rank + risk-veto + memory-veto ranker

## 目标

不要把 value 和 risk 混在一个分数里。v9.6.1 使用三段式：

```text
先按 value rank 找候选；
再用 longrisk veto；
再用 memory/offdiag veto；
最后做 group-balanced acceptance。
```

## Rank 形式

$$
Score_V(a)=f_V(x_a)
$$

$$
Veto_R(a)=1[Risk_R(a)\le \tau_R]
$$

$$
Veto_M(a)=1[Risk_M(a)\le \tau_M]
$$

$$
Accept(a)=TopK_{group-balanced}(Score_V(a)) \land Veto_R(a) \land Veto_M(a)
$$

## Rank candidates

```text
VRM1: value rank + longrisk veto
VRM2: value rank + memory veto
VRM3: value rank + longrisk + memory veto
VRM4: value rank + offdiag veto
VRM5: value rank + cover veto
VRM6: group-DRO value rank + risk/memory veto
VRM7: pairwise value rank within group + global risk veto
VRM8: group-calibrated score with per-group quota
```

## 必须记录

```text
ranker_id
feature_set
training_split
calibration_split
heldout_split
accepted_count
coverage
GradeAB_precision
V_integrated_LCB
h240_longrisk_UCB
bad_UCB
null_UCB
memory_fail_UCB
cover_fail_UCB
population_risk_offdiag_fail_UCB
max_group_share
worst_group_precision
LDO_drop_max
LSO_drop_max
dataset_name_used
outcome_field_used
score_component_ablation
```

## 判断标准

P5 pass：

```text
accepted_count >= 87
coverage >= 0.03
GradeAB_precision >= 0.75
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.05
memory_fail_UCB <= 0.10
max_group_share <= 0.35
LDO_drop_max <= 0.10
LSO_drop_max <= 0.10
```

Weak pass：

```text
GradeAB_precision >= 0.70
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.10
LDO_drop_max <= 0.20
LSO_drop_max <= 0.20
```

## 可视化

```text
p5_value_rank_risk_veto_frontier.svg
p5_ranker_group_share_bar.svg
p5_veto_ablation_waterfall.svg
p5_calibration_to_heldout_drift.svg
p5_leaveout_drop_by_axis.svg
```

---

# 11. P6：group-invariant pairwise ranker v3

## 目标

在 v9.5.9 / v9.6.0 中，GIR9 的 TopK 很强但 drop 仍为 0.356。P6 要测试更硬的 group-invariant training。

## 方法

```text
1. group-adversarial penalty；
2. per-group pairwise loss equalization；
3. worst-group validation selection；
4. payload_norm_bucket deconfounding；
5. leave-one-axis-out training；
6. group-balanced TopK acceptance。
```

## Loss

$$
L = L_{pairwise} + \lambda_1 L_{worst-group} + \lambda_2 L_{group-share} + \lambda_3 L_{risk-veto}.
$$

其中：

$$
L_{worst-group}=\max_g L_g.
$$

$$
L_{group-share}=\sum_g \max(0, Share_g-\tau_g)^2.
$$

## 必须记录

```text
ranker_id
pair_count
positive_pair_count
negative_pair_count
group_axes_used_for_training
group_axes_heldout
pairwise_auc_global
pairwise_auc_worst_group
TopK87_precision
TopK87_V_LCB
TopK87_longrisk_UCB
worst_group_precision
max_group_share
LDO_drop_by_axis
LSO_drop_by_axis
score_distribution_by_group
```

## 判断标准

P6 pass 与 P5 相同，但额外要求：

```text
pairwise_auc_worst_group >= 0.70
score_distribution_KS_max <= 0.25
max_group_share <= 0.35
```

## 可视化

```text
p6_pairwise_score_by_group.svg
p6_worst_group_training_curve.svg
p6_group_adversarial_ablation.svg
p6_payload_bucket_score_distribution.svg
```

---

# 12. P7：rank-safe certificate v9

## 目标

把 TopK diagnostic 转成 frozen accepted region。v9.6.0 RC7 失败的核心是 accepted region precision 低、longrisk 高。P7 使用 group-conformal risk bound，而不是单阈值。

## Certificate form

$$
Cert(a)=
1[
Score_V(a) \ge q_V(g)
]
\land
1[
Risk_R(a) \le q_R(g)]
\land
1[
Risk_M(a) \le q_M(g)]
\land
1[Support_g(a) \ge s_{min}].
$$

其中 $g$ 不是 dataset dispatch，而是 calibration group axis 的统计分桶。threshold 必须由 calibration split 冻结。

## 必须记录

```text
certificate_id
ranker_source
calibration_split
heldout_split
conformal_group_axis
qV_by_group
qR_by_group
qM_by_group
accepted_count_cal
accepted_count_heldout
coverage_heldout
GradeAB_precision_heldout
V_integrated_LCB_heldout
h240_longrisk_UCB_heldout
memory_fail_UCB_heldout
bad_UCB_heldout
null_UCB_heldout
ECE
worst_group_precision
max_group_share
LDO_drop
LSO_drop
```

## 判断标准

P7 pass：

```text
accepted_count_heldout >= 87
coverage_heldout >= 0.03
GradeAB_precision_heldout >= 0.75
V_integrated_LCB_heldout > 0
h240_longrisk_UCB_heldout <= 0.05
memory_fail_UCB_heldout <= 0.10
bad_UCB_heldout <= 0.05
null_UCB_heldout <= 0.15
LDO_drop <= 0.10
LSO_drop <= 0.10
```

## 可视化

```text
p7_certificate_calibration_curves.svg
p7_conformal_thresholds_by_group.svg
p7_accepted_region_composition.svg
p7_topk_vs_accepted_region_comparison.svg
```

---

# 13. P8：existing-action minimal controller

## 目标

如果 P5/P6/P7 至少 weak pass，则构造最小 online controller。该 controller 只能接受 existing AP0 action，不能使用 outcome-derived field。

## Controller

$$
Accept(a)=Cert(a)\land Cost(a)\le C_{max}.
$$

## 必须记录

```text
controller_id
ranker_id
certificate_id
accepted_count
coverage
GradeAB_precision
V_integrated_LCB
h20/h80/h240 V_LCB
h240_longrisk_UCB
memory_fail_UCB
cover_fail_UCB
bad_UCB
null_UCB
max_group_share
worst_group_precision
LDO_drop
LSO_drop
uses_dataset_name
uses_outcome_at_commit
uses_future_outcome
feature_cost_q90_ms
payload_apply_cost_q90_ms
```

## 判断标准

P8 pass：同 P7，并且：

```text
uses_dataset_name = 0
uses_outcome_at_commit = 0
feature_cost_q90_ms <= budget
```

## 可视化

```text
p8_controller_gate_flow.svg
p8_controller_accept_reject_map.svg
p8_controller_group_balance.svg
p8_controller_feature_cost.svg
```

---

# 14. P9：selected runtime preflight

## 目标

只有 P8 pass 后才测 selected runtime。不能提前把 runtime preflight 写成 system pass。

## 必须记录

```text
controller_id
selected_action_count_per_step
active_step_count
zero_candidate_step_count
controller_feature_time_ms_q50/q90/q99
certificate_time_ms_q50/q90/q99
payload_apply_time_ms_q50/q90/q99
kernel_launch_count
sync_count
allocation_count
step_ratio_q50/q90/q99
memory_ratio
selected_runtime_pass
```

## 判断标准

P9 pass：

```text
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
zero_candidate_controller_launch_count = 0
old_runtime_reused = 0
measured_selected_controller_runtime = 1
```

## 可视化

```text
p9_runtime_breakdown_stacked_bar.svg
p9_step_ratio_distribution.svg
p9_payload_apply_time_by_shape.svg
p9_active_step_controller_launches.svg
```

---

# 15. P10：APGA/APGD OOD causal autopsy v2

## 目标

P10 不再生成新 primitive，而是解释 APGA/APGD 为什么大量制造 longrisk。

## 必须记录

```text
generator_family
source_action_id
generated_action_id
source_group
generated_group
source_payload_norm_bucket
generated_payload_norm_bucket
source_memory_risk
generated_memory_risk
source_offdiag_risk
generated_offdiag_risk
source_cover_entropy
generated_cover_entropy
source_basis_rank
generated_basis_rank
V_damage
longrisk_created
memory_fail_created
offdiag_fail_created
cover_fail_created
dominant_damage_mode
```

## 判断标准

P10 pass：

```text
assigned_damage_fraction >= 0.90
longrisk_created explained by memory/offdiag/cover >= 0.75
specific generator component attribution >= 0.60
```

## 可视化

```text
p10_generator_damage_sankey.svg
p10_longrisk_created_by_component.svg
p10_source_generated_geometry_delta.svg
p10_offdiag_memory_cover_correlation.svg
```

---

# 16. P11：APGE1-APGE8 memory/offdiag-safe primitive

## 目标

只有在 P10 明确 damage 机制后，才生成 APGE。APGE 不是 APGD 小变体，而是显式约束 memory/offdiag/cover。

## APGE primitive family

```text
APGE1: MemoryNullspaceProjectedResidual
APGE2: OffdiagSafePopulationRiskGate
APGE3: CoverEntropyPreservingEdgeUpdate
APGE4: OldFamilyOrthogonalizedFunctionalDelta
APGE5: SymmetricBoundarySmallStepUpdate
APGE6: SignalChannelSNRPreconditionedDelta
APGE7: ValueRankSeededMemorySafeBlend
APGE8: NegativeControlShuffledMemoryProjection
```

## 生成规则

每个 APGE action 必须产生：

```text
payload tensor
certificate tensor
memory projection trace
offdiag risk trace
cover preservation trace
signal SNR trace
cost trace
```

## 必须记录

```text
primitive_id
generated_action_count
payload_hash_missing_count
certificate_hash_missing_count
action_apply_error_linf_max
memory_projection_error
offdiag_projection_error
cover_entropy_preservation_score
basis_rank_preservation_score
snr_score
estimated_payload_apply_ms
```

## 判断标准

P11 implementation pass：

```text
generated_action_count >= 512
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_error_linf_max <= 1e-6
negative_control_present = 1
```

## 可视化

```text
p11_apge_primitive_family_map.svg
p11_projection_error_hist.svg
p11_memory_cover_preservation_by_primitive.svg
```

---

# 17. P12：APGE branch-horizon outcome

## 目标

对 APGE 真实 materialize branch-horizon outcomes，不能用 proxy。

## Branches

```text
RealAPGE
AdamWOnly
AdamWParallel
BestLR
NoOp
Random
ShuffledAPGEPayload
CertificatePassNoPayload
```

Horizons：

```text
h20
h80
h240
```

## 必须记录

```text
branch_horizon_rows_expected
branch_horizon_rows_actual
branch_completion_rate
horizon_completion_rate
secondary_delta_completion_rate
quality_audit_pass
rows_per_sec
unresolved_exception_count
duplicate_row_count
label_exclusivity_violation_count
```

## 判断标准

P12 pass：

```text
branch_horizon_rows_actual = branch_horizon_rows_expected
quality_audit_pass = 1
unresolved_exception_count = 0
no fake/proxy rows
```

---

# 18. P13：APGE outcome / geometry pass

## 目标

判断 APGE 是否真正生成 value-positive、horizon-safe、memory-safe、cover-safe 动作。

## 必须记录

```text
primitive_id
accepted_count
GradeAB_precision
V_integrated_LCB
h20_V_LCB
h80_V_LCB
h240_V_LCB
h240_longrisk_UCB
bad_UCB
null_UCB
memory_fail_UCB
cover_fail_UCB
offdiag_fail_UCB
new_positive_created_rate
longrisk_created_rate
memory_fail_created_rate
cover_fail_created_rate
source_to_generated_damage_LCB
```

## 判断标准

APGE weak pass：

```text
GradeAB_precision >= 0.25
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.20
longrisk_created_rate <= 0.25
```

APGE strong pass：

```text
GradeAB_precision >= 0.50
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.10
memory_fail_UCB <= 0.10
cover_fail_UCB <= 0.10
new_positive_created_rate >= 0.20
```

Official candidate pass：

```text
accepted_count >= 87
GradeAB_precision >= 0.75
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.05
LDO/LSO drop <= 0.10
```

## 可视化

```text
p13_apge_value_risk_frontier.svg
p13_apge_memory_cover_frontier.svg
p13_apge_source_to_generated_damage.svg
p13_apge_primitive_comparison.svg
```

---

# 19. P14：system gate and leaveout boundary

## 目标

若 P8/P9 或 P13 过线，打开 official system boundary。否则只写 not_run。

## 必须记录

```text
system_candidate_id
controller_id
primitive_id
certificate_id
runtime_candidate_id
official_eligible
system_legal_controller_pass
leave_dataset_out_pass
leave_stratum_out_pass
leave_payload_bucket_out_pass
leave_family_out_pass
paired_replay_opened
reason_if_not_opened
```

## 判断标准

P14 pass：

```text
official_eligible = 1
system_legal_controller_pass = 1
selected_runtime_pass = 1
LDO_drop <= 0.10
LSO_drop <= 0.10
leave_payload_bucket_out_drop <= 0.10
```

---

# 20. P15：paired replay / short-full boundary

## 目标

只有 P14 pass 后，才打开 paired replay 和 short/full training。

## Paired replay branches

```text
RealFunctional
AdamWOnly
AdamWParallel
BestLR
NoOp
Random
ShuffledPayload
ShuffledScore
```

## 必须记录

```text
paired_replay_row_count
real_beats_adamwparallel_rate
real_beats_bestlr_rate
real_beats_noop_rate
real_beats_random_rate
shuffled_payload_pass_rate
V_integrated_LCB
h20/h80/h240 V_LCB
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
calibration_delta
robustness_delta
sample_efficiency_delta
runtime_step_ratio
memory_ratio
```

## 判断标准

Paired replay pass：

```text
real_beats_adamwparallel_rate >= 0.55
real_beats_bestlr_rate >= 0.55
real_beats_noop_rate >= 0.60
real_beats_random_rate >= 0.60
shuffled_payload_pass_rate <= 0.10
V_integrated_LCB > 0
bad/null/longrisk gates pass
```

Short/full boundary pass：

```text
short_run_acc_not_worse = 1
sample_efficiency_improved = 1 or calibration/robustness improved = 1
step_ratio_q90 <= 1.50
no dataset-specific tuning
```

---

# 21. 并行执行安排

```text
Batch A:
  P0 boundary reproduction
  P1 group pocket intervention panel
  P2 matched-pair causal attribution
  P3 group-balanced target map

Batch B:
  P4 legal feature deconfounding
  P5 value-rank + risk/memory veto
  P6 group-invariant pairwise ranker
  P7 rank-safe certificate

Batch C:
  P10 APGA/APGD OOD autopsy
  P11 APGE implementation
  P12 APGE branch-horizon outcome
  P13 APGE geometry pass

Batch D:
  P8 existing-action controller if P5/P6/P7 pass
  P9 selected runtime if P8 pass
  P14 system gate if P8/P9 or P13 pass

Batch E:
  P15 paired replay and short/full only if P14 pass
```

这样安排的目的：不再一轮只清一个 blocker。P1/P2 解释 pocket，P5/P6/P7 尝试 existing-action route，P10-P13 同时追 generated-action route。

---

# 22. 必须产出的 artifact

```text
p0_v9600_boundary_reproduction.csv
p1_group_pocket_intervention_panel.csv
p1_intervention_clone_trace.csv
p2_matched_pair_causal_attribution.csv
p2_pair_lift_trace.csv
p3_group_balanced_target_density_v2.csv
p4_legal_feature_deconfounding_v3.csv
p5_value_rank_risk_memory_veto_ranker.csv
p6_group_invariant_pairwise_ranker_v3.csv
p7_rank_safe_certificate_v9.csv
p8_existing_action_minimal_controller.csv
p9_selected_runtime_preflight.csv
p10_apga_apgd_ood_causal_autopsy_v2.csv
p11_apge_memory_offdiag_safe_primitive.csv
p12_apge_branch_horizon_outcome.csv
p13_apge_outcome_geometry_pass.csv
p14_system_leaveout_boundary.csv
p15_paired_replay_short_full_boundary.csv
base_acc_sentinel_v9610.csv
no_fake_audit_v9610.csv
contract_audit_v9610.csv
failure_taxonomy_v9610.csv
route_decision_v9610.json
```

---

# 23. 关键可视化总表

```text
p1_payload_bucket_intervention_flow.svg
p1_norm_bucket_counterfactual_lift.svg
p2_pair_lift_forest.svg
p3_target_group_support_heatmap.svg
p4_value_vs_longrisk_feature_tradeoff.svg
p5_value_rank_risk_veto_frontier.svg
p5_leaveout_drop_by_axis.svg
p6_payload_bucket_score_distribution.svg
p7_topk_vs_accepted_region_comparison.svg
p9_runtime_breakdown_stacked_bar.svg
p10_generator_damage_sankey.svg
p13_apge_value_risk_frontier.svg
p14_system_gate_waterfall.svg
p15_paired_replay_branch_comparison.svg
```

---

# 24. Stop / Pivot 条件

## 24.1 若 P1/P2 显示 payload0 是因果机制

继续：

```text
用 payload0 mechanism 设计 group-invariant feature；
但 controller 仍不能使用 payload0 作为单一硬规则；
必须通过 P5/P6/P7 的 group-stable gate。
```

## 24.2 若 P1/P2 显示 payload0 只是 confounded pocket

停止：

```text
停止围绕 payload_norm_bucket 做 rank patch；
转向 memory/offdiag/signal mechanism；
P5/P6 中移除 payload_norm_bucket 作为主特征。
```

## 24.3 若 P5/P6/P7 existing-action controller 过线

继续：

```text
打开 P8/P9/P14；
但 APGE 仍保留 diagnostic。
```

## 24.4 若 existing-action controller 仍 group unstable

转向：

```text
停止 existing AP0 selection 主线；
APGE generated-action route 成为主线；
同时保留 legal rank 作为 diagnostic / teacher sandbox，不 official。
```

## 24.5 若 APGE 仍生成 high-longrisk

转向更根本 primitive：

```text
不再做 residual blend / projection 小变体；
改成 population-risk gated optimizer update：
只更新满足 SNR / offdiag agreement / memory-safe 的 edge groups。
```

---

# 25. 成功路线定义

v9.6.1 不要求 full external success，但要求把 route 明确落入以下之一：

```text
R1: GroupPocketMechanismResolved_ControllerStillFail
R2: GroupPocketConfounded_StopPayloadPocketRoute
R3: ExistingActionGroupStableControllerPass
R4: ExistingActionControllerFail_APGEWeakPass
R5: APGEGeneratedFrontierPass
R6: BothExistingAndGeneratedFail_PrimitiveResetToPopulationRiskGate
R7: SystemPass_OpenPairedReplay
```

最理想结果：

```text
R7-SystemPass_OpenPairedReplay
```

可接受科学进展：

```text
R2 或 R6，只要机制证据清楚，且下一步不再围绕错误路线打转。
```

---

# 26. 一句话总结

v9.6.1 的目标不是让 `GIR9` 或 `APGD2` “再好一点”，而是彻底回答：

$$
\boxed{
\text{当前强 rank signal 到底是局部 payload pocket，还是可以变成跨 group 稳定的 functional update rule？}
}
$$

如果答案是前者，停止 existing-action rank patch；如果答案是后者，把它变成 group-stable controller；如果两者都不行，转向真正的 population-risk / memory-safe / offdiag-safe primitive，而不是继续增加小变体。
