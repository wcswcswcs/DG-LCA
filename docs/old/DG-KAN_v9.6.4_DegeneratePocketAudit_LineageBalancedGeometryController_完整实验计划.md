# DG-KAN v9.6.4 Degenerate Pocket Audit / Lineage-Balanced Geometry Controller / Value-Preserving Primitive 完整实验计划

> 本计划基于 v9.6.3 的真实执行结果制定。v9.6.4 不继续小修 `GIR9 / RANK-D / CERT11 / APGH7`，也不把 `payload_norm_bucket=payload0` 继续当作可用机制。v9.6.4 的核心任务是先判断 v9.6.3 的 accepted region 到底是一个无意义的 group collapse、一个单一 lineage 的局部口袋，还是一个真正有用的 memory/offdiag-safe 几何区域。只有这个判断清楚后，才允许继续 existing-action controller 或 generated-action primitive。

---

## 0. 一句话目标

v9.6.4 的目标是：

$$
\boxed{
\text{把“强 TopK 但 group pocket”拆成可解释的 lineage / memory / offdiag / value-risk 结构，并判断 existing-action controller 是否还有 official 路线。}
}
$$

更直白地说，本轮要回答：

```text
v9.6.3 里那 87 个看起来很好的 accepted actions，
到底是：

1. 一个统计口袋，不能用；
2. 一个单一 candidate/source lineage 的重复区域，不能推广；
3. 一个 memory/offdiag-safe 的真实好几何区域，可以转成 controller；
4. 还是三者混在一起，需要重新设计 action generator。
```

---

## 1. v9.6.3 的独立判断

v9.6.3 的 route 是：

```text
route = R2-hidden_multiaxis_pocket_explains_rank
system_legal_controller_pass = 0
primary_blocker = multi_axis_group_pocket_explains_rank
secondary_blocker = apgh_generated_frontier_fail
```

这轮不是失败回退。它有三个真实推进：

```text
1. P1 证明 accepted-region scope 没有 duplicate / row-action / universe mismatch。
2. P2 证明 v9.6.2 的强 accepted region 可以被 group pocket 解释。
3. P10/P11 证明 APGH1-APGH8 的 payload / certificate / action apply / branch-horizon replay 都完整闭合。
```

但我不完全接受 “payload_norm_bucket=payload0 explains rank” 这个表述。原因是 v9.6.3 P1 里有一个非常危险的细节：

```text
payload_norm_bucket=payload0:
  accepted share = 1.0
  base share = 1.0

action_norm_bucket=anorm0:
  accepted share = 1.0
  base share = 1.0

candidate_origin=canonical_AP0:
  accepted share = 1.0
  base share = 1.0
```

如果一个 group 在整个 evaluation universe 里本来就是 100%，它不能解释为什么 accepted region 好。它只能说明这个字段没有区分力。这样的字段应该被标为 `degenerate_axis`，不能作为 terminal causal explanation。

真正值得追的是这些非退化 group：

```text
memory_bucket=memory_fail_0:
  accepted share = 1.0
  base share ≈ 0.1777

offdiag_bucket=offdiag_fail_0:
  accepted share = 1.0
  base share ≈ 0.1380

dataset_id=KMNIST:
  accepted share ≈ 0.4138
  base share ≈ 0.2879
```

这意味着 v9.6.3 不能简单解读成 “rank 被 payload pocket 解释，所以 existing-action route 死了”。更准确的判断是：

$$
\boxed{
\text{accepted region 不是 scope 错；但它高度集中在少数几类几何安全 / lineage / group 区域。}
}
$$

本轮还必须特别审计一个异常：

```text
accepted_action_count = 87
accepted_unique_action_count = 87
accepted_unique_event_count = 87
accepted_unique_candidate_count = 1
```

如果 87 个 action 只来自 1 个 candidate template 或 source lineage，那么这不是一个可部署 controller，而是一个局部 action family。即使它 precision 高，也不能代表通用 functional update 规则。

---

## 2. v9.6.4 不继续做什么

v9.6.4 禁止继续走这些路线：

```text
1. 不继续调 RANK-D threshold。
2. 不继续调 CERT11 accepted count。
3. 不继续调 APGH7 blend / scale / SNR 参数。
4. 不把 payload_norm_bucket=payload0 当 selector。
5. 不把 base share = 1.0 的 group 当 pocket explanation。
6. 不只看 TopK87 precision。
7. 不用 pooled precision 覆盖 LDO / LSO / lineage drop。
8. 不在 APGH value-negative / high-longrisk 后继续加 APGH 小变体。
9. 不打开 paired replay / short-full training，除非 existing-action controller 或 generated primitive 先过 system gate。
```

---

## 3. v9.6.4 总体实验目标

v9.6.4 有两个并行主线。

### 主线 A：existing-action route

判断 canonical AP0 里的已有好动作，是否能通过合法、跨 group 稳定、lineage-balanced 的规则选出来。

成功形式是：

$$
Accept(a)=1
$$

当且仅当：

$$
ValueRank(a) \ge q_v
$$

$$
RiskVeto(a)=0
$$

$$
MemoryVeto(a)=0
$$

$$
OffdiagVeto(a)=0
$$

$$
LineageBalance(a)=1
$$

$$
Cost(a) \le C_{max}.
$$

### 主线 B：generated-action route

如果 existing-action route 被证明只是局部 lineage/pocket，则必须生成新的动作。新动作不能只是合法 payload，而要满足：

```text
1. value direction 不丢；
2. longrisk 不升；
3. memory/offdiag 不破坏；
4. cover 不塌缩；
5. 与 AdamW 不冲突；
6. runtime 可承受。
```

---

## 4. 本轮核心假设

### H1：v9.6.3 的 terminal pocket explanation 混入了退化轴

如果某个 group 的 base share 本身为 1.0，则该 group 不能作为解释变量。

成立标准：

```text
存在 base_share >= 0.95 的 axis，被 P2 选为 best_axis_set；
adjusted_explanation 去掉 degenerate axes 后，best_drop_explained_fraction 明显下降；
nondegenerate_axis_explanation 与 raw P2 conclusion 不一致。
```

失败标准：

```text
去掉所有 base_share >= 0.95 的 axis 后，仍有非退化 group 能解释 accepted region，并且该 group 不使用 forbidden selector field。
```

### H2：accepted region 可能是单一 candidate/source lineage，而非通用规则

成立标准：

```text
accepted_unique_candidate_count <= 3；
accepted_unique_source_payload_hash_count <= 3；
accepted_lineage_entropy <= 1.0；
leave-lineage-out drop >= 0.25。
```

失败标准：

```text
accepted actions 分布在足够多 candidate/source lineage；
leave-lineage-out drop <= 0.10；
max lineage share <= 0.25。
```

### H3：memory/offdiag-safe 可能是真正有用的好几何信号

成立标准：

```text
memory_safe 与 offdiag_safe 在 matched family / step / dataset / action-norm 条件下仍提升 GradeAB；
P(GradeAB | memory_safe & offdiag_safe) 显著高于 base；
P(longrisk | memory_safe & offdiag_safe) 显著低于 base；
该关系在 LDO / LSO / leave-family-out 中稳定。
```

失败标准：

```text
memory/offdiag 只是与某个 source lineage 或 dataset pocket 共现；
去掉 lineage/pocket 后 lift 消失。
```

### H4：TopK 强不等于 accepted region 可用

成立标准：

```text
TopK precision 高，但 frozen accepted region 的 V_LCB <= 0 或 longrisk_UCB > gate；
accepted region 对 group / lineage drop 明显。
```

v9.6.3 已经强烈支持这个假设，因此 v9.6.4 必须用 frozen accepted region，而不是只用 TopK。

### H5：APGH 失败的主因是 value direction lost，而不是实现问题

成立标准：

```text
APGH action apply error = 0；
branch-horizon rows 完整；
但 source-to-generated Damage V LCB < 0；
dominant damage mode = D2-value-direction-lost；
longrisk created rate 高。
```

v9.6.3 已支持该假设。本轮要进一步拆 APGH 哪一步丢 value：source selection、projection、memory/offdiag veto、SNR scaling、AdamW compatibility，还是 certificate pass condition。

---

## 5. 关键指标定义

### 5.1 退化轴指标

对每个 group axis $g$：

$$
BaseShare(g)=\frac{|\{a:g(a)=g^*\}|}{|A|}
$$

$$
AcceptedShare(g)=\frac{|\{a:Accept(a)=1,g(a)=g^*\}|}{|\{a:Accept(a)=1\}|}
$$

定义退化轴：

$$
DegenerateAxis(g)=1 \iff BaseShare(g) \ge 0.95.
$$

退化轴不能作为 causal explanation，也不能作为 selector。

### 5.2 adjusted pocket explanation

原始 drop：

$$
DropRaw(g)=Precision(Accept)-Precision(Accept \setminus g).
$$

调整后 drop：

$$
DropAdj(g)=DropRaw(g)\cdot 1[BaseShare(g)<0.95]\cdot 1[AllowedAsSelector(g)=1].
$$

如果 $DropRaw$ 高但 $DropAdj=0$，则说明这是不可用解释。

### 5.3 lineage balance

令 $L(a)$ 是 source candidate / payload / event family / action template 的组合 lineage。

$$
MaxLineageShare=\max_l \frac{|\{a:Accept(a)=1,L(a)=l\}|}{|\{a:Accept(a)=1\}|}.
$$

$$
LineageEntropy=-\sum_l p_l\log p_l.
$$

controller 最低要求：

```text
MaxLineageShare <= 0.25
LineageEntropy >= 2.0
accepted_unique_lineage_count >= 8
```

若做不到，就只能算 diagnostic。

### 5.4 好动作等级

本轮不只用 GradeAB。每个 action 记录：

```text
V_integrated_LCB
h20_value
h80_value
h240_value
h240_longrisk_UCB
bad_event_UCB
null_event_UCB
memory_fail
old_family_fail
offdiag_fail
cover_entropy_delta
basis_effective_rank_delta
CEp99_delta
margin_p10_delta
runtime_cost
```

GoodGeometry-A：

```text
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.05
bad_event_UCB <= 0.05
null_event_UCB <= 0.15
memory_fail = 0
offdiag_fail = 0
cover_entropy_delta >= 0 或在容忍范围内
```

GoodGeometry-B：

```text
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.10
memory_fail = 0
offdiag_fail = 0
bad/null 不越界
```

GoodGeometry-C：

```text
短期 value 正，但至少一个中长期 / memory / offdiag 指标未闭合。
```

Official controller 只能接受 A/B，不允许把 C 当成功。

---

# Part I：Existing-Action Route

## P0. Boundary reproduction

### 目标

复现 v9.6.3 boundary，确保没有跳过 v9.6.2 payload-pocket-stopped 与 v9.6.3 multi-axis pocket conclusion。

### 输入

```text
source_v9630 = results/.../v9630_group_stable_accepted_region_memory_offdiag_primitive_first_20260515T130000Z
source_v9620 = results/.../v9620_confounder_purged_legal_rank_poprisk_geometry_primitive_first_20260515T120000Z
source_v9610 = results/.../v9610_group_pocket_causal_intervention_group_invariant_controller_memory_safe_primitive_first_20260515T110000Z
canonical_ap0_table = v9480/v9490 canonical full outcome table
```

### 必须记录

```text
route_v9630
system_legal_controller_pass_v9630
accepted_action_count_v9630
accepted_unique_action_count_v9630
accepted_unique_candidate_count_v9630
GradeAB_precision_v9630
V_LCB_v9630
h240_longrisk_UCB_v9630
best_ranker_v9630
best_certificate_v9630
best_apgh_primitive_v9630
```

### 判断标准

P0 pass：

```text
v9630 route 可复现；
accepted region / APGH / system gate 与 v9.6.3 一致；
no fake / no proxy / no old-table official。
```

---

## P1. Degenerate pocket audit

### 目标

判断 v9.6.3 的 pocket explanation 是否被 base share = 1.0 的退化轴污染。

### 做法

对所有 axis 重新计算：

```text
payload_norm_bucket
action_norm_bucket
candidate_origin
candidate_id
source_candidate_id
source_payload_hash
event_family
dataset_id
seed
step_bucket
memory_bucket
offdiag_bucket
cover_bucket
SNR_bucket
AdamW_conflict_bucket
```

每个 axis 输出：

```text
base_share
accepted_share
lift
raw_drop_if_removed
adjusted_drop_if_removed
is_degenerate_axis
is_forbidden_selector
is_commit_time_legal
```

### 可视化

```text
fig_p1_base_share_vs_accepted_share.svg
fig_p1_raw_drop_vs_adjusted_drop.svg
fig_p1_degenerate_axis_table.svg
fig_p1_group_entropy_by_axis.svg
```

### 判断标准

P1 pass-A：退化污染成立。

```text
best raw explanation 使用 base_share >= 0.95 axis；
best adjusted explanation 改变；
raw route conclusion 需要降级。
```

P1 pass-B：非退化解释成立。

```text
存在 base_share < 0.80 且 adjusted_drop >= 0.20 的 legal/diagnostic axis；
该 axis 在 leaveout 中仍有 lift。
```

P1 fail：

```text
无法解释 accepted region，或所有解释都来自 forbidden/degenerate axis。
```

---

## P2. Accepted lineage audit

### 目标

解释 `accepted_unique_candidate_count = 1` 是否意味着 accepted region 是单一 candidate/source lineage。

### 需要记录

```text
action_id
candidate_id
event_id
source_action_id
source_candidate_id
source_payload_hash
payload_hash
action_template_id
primitive_origin
family_id
dataset_id
seed
step_id
step_bucket
lineage_id
```

定义：

$$
LineageId=hash(source\_candidate,source\_payload,action\_template,family,step\_bucket).
$$

### 指标

```text
accepted_unique_lineage_count
accepted_lineage_entropy
max_lineage_share
leave_lineage_out_precision_drop
leave_lineage_out_V_drop
leave_lineage_out_longrisk_increase
candidate_to_action_expansion_ratio
candidate_id_collision_count
```

### 可视化

```text
fig_p2_accepted_lineage_sankey.svg
fig_p2_lineage_share_bar.svg
fig_p2_leave_lineage_out_drop.svg
fig_p2_candidate_action_event_map.svg
```

### 判断标准

P2 pass-GENERAL：

```text
accepted_unique_lineage_count >= 8
max_lineage_share <= 0.25
leave_lineage_out_precision_drop <= 0.10
leave_lineage_out_V_drop <= 0.05
```

P2 fail-LINEAGE：

```text
accepted_unique_lineage_count < 4
or max_lineage_share > 0.50
or leave_lineage_out_precision_drop > 0.20
```

如果 P2 fail-LINEAGE，则 existing-action controller 不能 official，即使 precision / V / longrisk 看起来好。

---

## P3. Memory/offdiag geometry causality audit

### 目标

判断 memory-safe / offdiag-safe 是否是真正的好几何信号，而不是 lineage/pocket 共现。

### 做法

构造 matched comparison：

```text
same dataset_id
same seed bucket
same event_family
same step_bucket
same action_norm_bucket
same value-rank bucket
memory/offdiag status 不同
```

如果完全配不出 pairs，使用 nearest-neighbor matching：

```text
distance = zscore(action_norm) + zscore(payload_norm) + zscore(state_NLL) + zscore(AdamW_conflict) + family mismatch penalty
```

### 指标

```text
matched_pair_count_memory
matched_pair_count_offdiag
matched_pair_count_joint
GradeAB_lift_memory_safe
V_lift_memory_safe
longrisk_drop_memory_safe
GradeAB_lift_offdiag_safe
V_lift_offdiag_safe
longrisk_drop_offdiag_safe
joint_memory_offdiag_lift
```

### 可视化

```text
fig_p3_memory_offdiag_matched_lift.svg
fig_p3_longrisk_by_memory_offdiag.svg
fig_p3_V_distribution_by_geometry_status.svg
fig_p3_cover_memory_offdiag_scatter.svg
```

### 判断标准

P3 pass：

```text
matched_pair_count_joint >= 64
joint_memory_offdiag_lift_precision >= 0.20
joint_memory_offdiag_V_lift_LCB > 0
longrisk_drop_UCB > 0
leave_dataset_out_lift_drop <= 0.10
leave_stratum_out_lift_drop <= 0.10
```

P3 fail：

```text
matched pairs 不足；
或 lift 只在某个 lineage / dataset / payload pocket 存在；
或 memory/offdiag safe 不能降低 longrisk。
```

---

## P4. Out-of-pocket target map

### 目标

定义不依赖退化 axis、不依赖 payload0、不依赖单 lineage 的 target support map。

### Target variants

T4.1 ValuePositiveNoLongRiskLineageBalanced：

```text
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.05
max_lineage_share <= 0.25
```

T4.2 MemoryOffdiagSafeValue：

```text
V_integrated_LCB > 0
memory_fail = 0
offdiag_fail = 0
h240_longrisk_UCB <= 0.05
```

T4.3 GradeABBalanced：

```text
GradeAB = 1
max group share <= 0.25
positive group coverage >= 0.50
```

T4.4 OutOfPocketGradeAB：

```text
GradeAB = 1
not in forbidden pocket
not in degenerate-only explanation
lineage balanced
```

### 指标

```text
target_count
target_coverage
V_LCB
longrisk_UCB
bad_UCB
null_UCB
positive_group_coverage
max_group_share
max_lineage_share
LDO_drop
LSO_drop
```

### 判断标准

P4 pass：

```text
target_count >= 87
coverage >= 0.03
V_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
max_lineage_share <= 0.25
LDO_drop <= 0.10
LSO_drop <= 0.10
```

P4 weak pass：

```text
target_count >= 64
V_LCB > 0
longrisk_UCB <= 0.05
但 coverage 或 group balance 仍不足。
```

P4 fail：

```text
没有任何 target 同时满足 value/risk/support/group balance。
```

---

## P5. Group-deconfounded legal ranker v4

### 目标

把 value rank、risk veto、memory/offdiag veto、lineage diversity 分开，而不是塞进一个 opaque score。

### Rank 形式

$$
Score(a)=ValueRank(a)-\lambda_R Risk(a)-\lambda_M MemoryRisk(a)-\lambda_O OffdiagRisk(a)-\lambda_C Cost(a).
$$

接受规则：

$$
Accept(a)=1
$$

当且仅当：

```text
ValueRank(a) >= q_v
Risk(a) <= q_r
MemoryRisk(a) <= q_m
OffdiagRisk(a) <= q_o
LineageDiversityConstraint satisfied
Cost(a) <= Cmax
```

### Candidate rankers

```text
R4A value-only legal rank
R4B value + longrisk veto
R4C value + longrisk + memory veto
R4D value + longrisk + memory + offdiag veto
R4E pairwise within lineage
R4F pairwise leave-lineage-out trained
R4G group DRO ranker
R4H conformal group-balanced ranker
```

### 必须记录

```text
TopK64 / TopK87 / TopK128 precision
V_LCB
longrisk_UCB
bad_UCB
null_UCB
max_group_share
max_lineage_share
LDO_drop
LSO_drop
leave_lineage_out_drop
score monotonicity
feature ablation
single-feature baselines
```

### 可视化

```text
fig_p5_ranker_precision_vs_group_drop.svg
fig_p5_value_risk_frontier.svg
fig_p5_rank_score_hist_by_grade.svg
fig_p5_ablation_waterfall.svg
fig_p5_lineage_balanced_topk_map.svg
```

### 判断标准

P5 pass：

```text
TopK87 precision >= 0.75
V_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
max_group_share <= 0.25
max_lineage_share <= 0.25
LDO_drop <= 0.10
LSO_drop <= 0.10
leave_lineage_out_drop <= 0.10
```

P5 weak pass：

```text
TopK87 precision >= 0.70
V_LCB > 0
longrisk_UCB <= 0.10
但 group/lineage drop 仍稍高。
```

---

## P6. Rank-safe certificate v12

### 目标

把 P5 ranker 转成 frozen accepted-region controller，而不是只报告 TopK。

### 设计

```text
Calibration split:
  freeze q_v, q_r, q_m, q_o, lineage cap, accepted count。

Heldout split:
  只执行 frozen rule，不调 threshold。

Leaveout split:
  leave-dataset-out
  leave-stratum-out
  leave-family-out
  leave-lineage-out
```

### 指标

```text
accepted_count
coverage
precision
V_LCB
longrisk_UCB
bad_UCB
null_UCB
max_group_share
max_lineage_share
LDO_drop
LSO_drop
LFO_drop
LLO_drop
ECE 仅作辅助，不作为唯一 pass
```

### 判断标准

P6 pass：

```text
accepted_count >= 87
coverage >= 0.03
precision >= 0.75
V_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
max_group_share <= 0.25
max_lineage_share <= 0.25
LDO/LSO/LFO/LLO drop <= 0.10
```

如果 P6 pass，则打开 P7/P8。

如果 P6 fail 但 P5 pass，则说明 TopK 到 accepted-region 转换仍断裂，需要重新设计 certificate，不允许打开 runtime。

---

## P7. Existing-action minimal controller

### 目标

在 P6 pass 后，构造最小 online controller。

### Controller 限制

```text
feature groups <= 5
no dataset_name
no outcome-derived field
no payload_norm_bucket selector
no degenerate axis selector
no candidate_id direct selector
no source_payload_hash direct selector
```

### Controller 形式

```text
value score
risk veto
memory veto
offdiag veto
lineage diversity cap
runtime cost cap
```

### 指标

```text
controller_feature_cost_ms_q90
payload_apply_ms_q90
step_ratio_q90
memory_ratio
accepted_per_active_step
no_event_preservation_pass
base_adamw_equivalence_zero_event_steps
```

### 判断标准

P7 pass：

```text
P6 pass
feature_cost_ms_q90 <= budget
no_event_preservation_pass = 1
base_adamw_equivalence_zero_event_steps = 1
```

---

## P8. Selected runtime measurement

### 目标

真实测量 selected controller runtime，不用 estimate。

### 指标

```text
online_runtime_measured = 1
selected_controller_in_timed_path = 1
audit_outside_timed_path = 1
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
payload_apply_q90
kernel_count
sync_count
zero_candidate_launch_count
active_step_launch_q90
```

### 判断标准

P8 pass：

```text
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
zero_candidate_launch_count = 0
no CPU offload
no proxy rows
```

---

# Part II：Generated-Action Route

## P9. APGH damage decomposition

### 目标

解释 APGH 为什么失败，尤其是 `D2-value-direction-lost` 和 longrisk created。

### 分解维度

```text
source action quality
value direction alignment
AdamW compatibility
memory/offdiag preservation
cover entropy preservation
basis rank preservation
SNR gate
risk veto
certificate pass condition
projection step
scaling step
```

### 指标

```text
Damage_V_LCB per primitive
source_positive_preserved_rate
new_positive_created_rate
longrisk_created_rate
value_direction_cosine_before_after
AdamW_conflict_delta
memory_fail_delta
offdiag_fail_delta
cover_entropy_delta
basis_rank_delta
```

### 可视化

```text
fig_p9_source_to_generated_damage_matrix.svg
fig_p9_value_direction_cosine_hist.svg
fig_p9_longrisk_created_by_transform_step.svg
fig_p9_memory_offdiag_delta_by_primitive.svg
```

### 判断标准

P9 pass：

```text
dominant failure assigned fraction >= 0.80
per-primitive failure reason recorded
clear stop/go decision for each APGH primitive
```

---

## P10. APGL1-APGL8 value-preserving geometry primitive

### 目标

不再生成“合法 payload”，而是生成 value-preserving、memory/offdiag-safe、cover-stable 的动作。

### Primitive family

```text
APGL1-SourceReplayPreserver:
  no-transform / tiny legal perturbation，用来确认 source value can be preserved。

APGL2-PopRiskProjectedEdgeDelta:
  使用 population-risk SNR 规则，只保留多个样本一致支持的方向。

APGL3-MemoryOffdiagNullspaceProjection:
  将 delta 投影到 old-family / offdiag risk 的近似 nullspace。

APGL4-CoverEntropyBoundedDelta:
  限制 cover entropy 和 basis effective rank 变化。

APGL5-BoundarySymmetricResidual:
  使用正负对称边界，防止单向漂移。

APGL6-ValueRiskTwoHeadProjectedDelta:
  分离 value head 和 risk veto head。

APGL7-LineageDiverseEnsembleIntersection:
  只保留多个 lineage / family 同时支持的 delta。

APGL8-ShuffledPayloadNegativeControl:
  shape-preserving shuffled negative control。
```

### Preflight

```text
single action preflight
three action preflight
sixteen action preflight
no-transform equivalence
negative control divergence
payload hash binding
certificate tensor binding
action apply L∞ = 0 or within tolerance
```

### 判断标准

P10 pass：

```text
generated actions = 512
payload/certificate hash missing = 0
action apply L∞ max <= tolerance
preflight all pass
negative controls diverge
```

---

## P11. APGL branch-horizon smoke outcome

### 目标

真实 materialize generated APGL outcomes。

### Branches

```text
RealAPGL
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledAPGLPayload
CertificatePassNoPayload
```

### Horizons

```text
h20
h80
h240
```

### 指标

```text
generated_action_count
branch_horizon_rows_expected
branch_horizon_rows_actual
branch_completion
horizon_completion
missing_secondary_delta_count
label_exclusivity_violation_count
duplicate_row_count
rows_per_sec
```

### 判断标准

P11 pass：

```text
actual rows = expected rows
unresolved exception = 0
quality audit pass = 1
```

---

## P12. APGL outcome / geometry pass

### 目标

判断 APGL 是否真正产生新好动作。

### 指标

```text
GradeAB precision
GoodGeometry-A precision
GoodGeometry-B precision
V_integrated_LCB
h240_longrisk_UCB
bad_UCB
null_UCB
memory_fail_rate
offdiag_fail_rate
cover_collapse_rate
source_positive_preserved_rate
new_positive_created_rate
negative_control_pass
```

### 判断标准

APGL weak pass：

```text
accepted_count >= 64
GradeAB precision >= 0.50
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.10
negative control does not pass
```

APGL official candidate pass：

```text
accepted_count >= 87
coverage >= 0.03
GradeAB precision >= 0.75
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
memory/offdiag fail low
negative control fail
```

---

## P13. APGL certificate / controller boundary

如果 P12 pass，训练 APGL certificate。

### 指标

```text
TopK precision
frozen accepted precision
V_LCB
longrisk_UCB
bad_UCB
null_UCB
LDO_drop
LSO_drop
leave_lineage_out_drop
feature cost
payload apply cost
```

### 判断标准

P13 pass：同 P6 / P8。

---

# Part III：Downstream Gates

## P14. Official leaveout

只有 P7/P8 或 P12/P13 过后运行。

```text
leave-dataset-out
leave-stratum-out
leave-family-out
leave-lineage-out
```

通过标准：

```text
precision >= 0.75
V_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
no split drop > 0.10
```

## P15. Official paired replay

只有 system controller pass 后运行。

必须比较：

```text
RealFunctional
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledPayload
```

通过标准：

```text
RealFunctional beats AdamWParallel rate high
RealFunctional beats bestLR rate high
RealFunctional beats NoOp/Random/ShuffledPayload
V_LCB > 0
bad/null/longrisk controlled
```

## P16. Short/full training boundary

只有 P15 pass 后打开。

记录：

```text
test_acc
val_loss_auc_step
val_loss_auc_time
time_to_target
steps_to_target
NLL
ECE
CEp99
margin_p10
hard-stratum acc
memory retained acc
forgetting
forward transfer
backward transfer
step_ratio_q90
memory_ratio
```

注意：这些结果不能用于调 controller，只能用于最后验证。

---

## 6. 并行执行策略

v9.6.4 必须并行，不能一轮只发现一个 blocker。

### Lane A：Existing-action rank

```text
P1 degenerate pocket audit
P2 accepted lineage audit
P3 memory/offdiag causality
P4 target support
P5 ranker
P6 certificate
```

### Lane B：Generated-action primitive

```text
P9 APGH damage autopsy
P10 APGL implementation
P11 APGL outcomes
P12 APGL geometry pass
```

### Lane C：Runtime

```text
P7 controller cost preflight
P8 selected runtime only if P6/P13 pass
```

### Lane D：Sentinel / no-regression

```text
Base-Acc Sentinel continuation
no-fake audit
contract audit
old table quarantine
manual training audit
```

---

## 7. Early-stop 规则

### Stop existing-action route

如果同时满足：

```text
P2 fail-LINEAGE；
P3 memory/offdiag lift 不成立；
P5/P6 group-stable accepted region 不过；
```

则停止 existing-action controller route，转向 generated-action route。

### Stop APGL route

如果 APGL 仍满足：

```text
best generated V_LCB < 0；
longrisk_UCB > 0.50；
new positive created rate <= 0.05；
negative control 与 real primitive 差别不大；
```

则停止 APGL 小变体，转向更底层的 source action / update rule redesign。

### Open downstream

只有当：

```text
P6 或 P13 pass；
P8 selected runtime pass；
contract audit pass；
no fake / no proxy / no CPU offload；
```

才打开 P14-P16。

---

## 8. 必须落盘的 artifact

```text
p0_boundary_reproduction_v9640.csv
p1_degenerate_pocket_audit_v9640.csv
p2_accepted_lineage_audit_v9640.csv
p3_memory_offdiag_causality_v9640.csv
p4_out_of_pocket_target_map_v9640.csv
p5_group_deconfounded_ranker_v4.csv
p6_rank_safe_certificate_v12.csv
p7_existing_action_controller_boundary_v9640.csv
p8_selected_runtime_trace_v9640.csv
p9_apgh_damage_decomposition_v9640.csv
p10_apgl_primitive_implementation_v9640.csv
p11_apgl_branch_horizon_outcome_v9640.csv
p12_apgl_geometry_outcome_v9640.csv
p13_apgl_certificate_controller_v9640.csv
p14_leaveout_boundary_v9640.csv
p15_paired_replay_boundary_v9640.csv
p16_short_full_boundary_v9640.csv
base_acc_sentinel_v9640.csv
contract_audit_v9640.csv
no_fake_audit_v9640.csv
route_decision_v9640.json
run_manifest_v9640.json
```

---

## 9. 预期 route 决策

### R1-DegeneratePocketArtifact

```text
P1 发现 best explanation 全来自 base_share=1.0 退化轴。
下一步：重算 nondegenerate pocket / lineage route。
```

### R2-LineageCollapsedAcceptedRegion

```text
P2 发现 accepted region 由单一 candidate/source lineage 主导。
下一步：stop existing-action controller 或强制 lineage-balanced rank。
```

### R3-MemoryOffdiagRealMechanism

```text
P3 证明 memory/offdiag safe 是跨 group 稳定好几何信号。
下一步：继续 P5/P6 controller。
```

### R4-ExistingActionControllerPassRuntimeBlocked

```text
P6 pass 但 P8 runtime fail。
下一步：runtime-specific repair，不再动 science controller。
```

### R5-ExistingActionControllerPass

```text
P6/P8 pass。
下一步：official leaveout + paired replay。
```

### R6-APGLGeneratedFrontierPass

```text
APGL 生成出 value-positive / horizon-safe / memory-offdiag-safe frontier。
下一步：APGL certificate/controller/runtime。
```

### R7-ExistingAndGeneratedBothFail

```text
existing-action route 被 lineage/group instability 阻断；
generated-action route 仍 value-negative/high-longrisk。
下一步：reset action source/update rule，不再继续 APG* 小变体。
```

---

## 10. 最终判断标准

v9.6.4 最低有效进展：

```text
明确 v9.6.3 pocket conclusion 是否被 degenerate axes 污染；
明确 accepted region 是否 single-lineage collapse；
明确 memory/offdiag 是否真实好几何信号；
明确 APGH value-direction-lost 的具体发生位置；
APGL 完整落盘并给出 go/no-go。
```

v9.6.4 强进展：

```text
P6 existing-action certificate pass；
或 P12 APGL generated frontier pass。
```

v9.6.4 official success：

```text
P6/P13 controller pass；
P8 selected runtime pass；
P14 leaveout pass；
P15 paired replay pass。
```

如果只是再次得到：

```text
TopK precision 高；
但 accepted region group/lineage unstable；
APGL generated action value-negative/high-longrisk；
```

则必须停止 v9.x 的 existing-action rank / APG* primitive 小变体路线，进入更底层的 update rule redesign。
