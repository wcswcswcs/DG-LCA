# DG-KAN v9.6.0 Group-Pocket Deconfounded Rank / Memory-Safe LongRisk Primitive / Parallel Closure 完整实验计划

> 本计划基于 v9.5.9 `Group-Invariant Legal Rank / Existing-Action Controller / Generator OOD Triage` 的真实执行结果制定。  
> v9.5.9 的 terminal route 是：
>
> ```text
> route = R1-GroupSpecificRankPocket
> base_candidate = LQ-t2-h256
> success_v9590_strict_purekan_functional = False
> success_v9590_full_functional = False
> success_v9590_external_ready = False
> primary_blocker = group_specific_rank_pocket_group_invariant_rank_failed
> ```
>
> v9.6.0 不再小修 APGA、RC7、GIR9 或单一 TopK threshold。  
> 本轮目标是把 v9.5.9 的强 pooled legal rank signal 变成 group-stable accepted region；如果做不到，就用严格证据判定 existing-action selection 主线暂时不可部署，并转向 memory-safe / longrisk-veto-effective generated primitive。

---

# 0. 总体判断与本轮核心问题

v9.5.9 的关键进展不是 functional success，而是把问题定位得更准：

```text
1. legal rank 已经不是完全没信号；
2. pooled TopK 已经能找到高 GradeAB、正 V、低 longrisk 的区域；
3. 但这个区域高度集中在 payload_norm_bucket / payload0；
4. group-invariant ranker 已把 LDO/LSO drop 从约 0.885 降到约 0.356；
5. 但 0.356 仍然太高，不能 official；
6. rank-safe certificate 的 heldout accepted region 质量不足；
7. APGA 生成器主要失败于 longrisk veto ineffective 和 generated OOD payload；
8. memory/cover blocker 继续成立，population_risk_offdiag_fail 与 longrisk 强相关。
```

因此，v9.6.0 的核心问题不是：

```text
GIR9 threshold 能不能再调一点？
RC7 的 accepted count 能不能改一下？
APGA7 能不能再加一个小变体？
AUC 能不能再涨一点？
```

真正的问题是：

$$
\boxed{
\text{好动作是否能被一个不依赖 group pocket 的训练当下规则选出来？}
}
$$

以及：

$$
\boxed{
\text{如果不能选出来，能否直接生成 memory-safe、cover-safe、longrisk-veto-effective 的新动作？}
}
$$

---

# 1. 本轮不做什么

v9.6.0 明确禁止以下路线：

```text
1. 不继续调 R2 / GIR9 / RC7 的单阈值；
2. 不把 pooled TopK precision 当成 success；
3. 不把 AUC 当成 controller pass；
4. 不把 payload_norm_bucket / payload0 作为 deployed branch rule；
5. 不按 dataset 调 threshold、ranker、certificate 或 runtime route；
6. 不使用 outcome-derived fields，包括 V_integrated、risk_score、GradeAB、future label；
7. 不用 validation/test metric at commit time；
8. 不打开 paired replay、short/full training，除非 controller + runtime 先过；
9. 不新增没有机制假设的小 APG/APGA/APGM 变体；
10. 不把 generated primitive 的 implementation pass 写成 science pass。
```

允许做：

```text
1. 用 group 只做诊断、split、worst-group evaluation、training reweighting；
2. official score 不允许使用 dataset_name / group_id / target label / future outcome；
3. 可以使用 legal commit-time continuous features，如 action norm、AdamW alignment、control-transfer estimate、memory-safe proxy、cover proxy、population-risk offdiag proxy；
4. 可以做 group-balanced training 和 adversarial group-invariance regularization；
5. 可以做 leave-group-out / leave-dataset-out / leave-stratum-out；
6. 可以做 generated primitive sandbox，但必须真实 payload、真实 branch-horizon rows、真实 no-fake audit。
```

---

# 2. v9.6.0 总体目标

v9.6.0 的总体目标是：

$$
\boxed{
\text{把 v9.5.9 的 group-specific legal rank pocket 变成 group-stable legal accepted region；若失败，则重置为 memory-safe longrisk primitive。}
}
$$

本轮分两条并行路线。

第一条是 existing-action legal rank 路线：

```text
canonical AP0 actions 已经存在；
legal rank 有 pooled TopK signal；
现在要消掉 payload_norm_bucket / payload0 pocket；
让 accepted region 跨 group、dataset、stratum 稳定。
```

第二条是 generated primitive 路线：

```text
APGA 失败主要来自 longrisk veto ineffective 与 OOD payload；
下一步只允许围绕 memory-safe、cover-preserving、offdiag-risk-veto 机制重做生成器；
不再盲目添加小变体。
```

强目标：

$$
SystemLegalControllerPass=1
$$

$$
TopK87Precision_{GradeAB}\ge0.75
$$

$$
V_{integrated}^{LCB}>0
$$

$$
LongRisk_{240}^{UCB}\le0.05
$$

$$
Bad^{UCB}\le0.05
$$

$$
Null^{UCB}\le0.15
$$

$$
LDO/LSO\ Drop_{max}\le0.15
$$

$$
StepRatio_{q90}\le1.50
$$

最低有效推进目标：

```text
1. group pocket 的来源被解释清楚；
2. group-balanced target density 是否足够被判定；
3. GIR9/R2/ControlTransferImprovement 的 group drop 被拆成 value drop、risk drop、support drop；
4. 至少一个 legal ranker 在 leave-payload-bucket-out 上 drop 降到 <=0.25；
5. rank-safe certificate 的 heldout accepted region longrisk UCB 降到 <=0.10；
6. APGA OOD 的主要维度被量化；
7. 新 generated primitive 只有在 memory/cover/longrisk 机制明确后才 materialize。
```

---

# 3. 核心假设

## H1：当前 rank failure 主要不是没有信号，而是 TopK group concentration

v9.5.9 显示：

```text
max_topK_group_share = 1.0
max_drop = 0.8505747126436781
dominant_axis = payload_norm_bucket
dominant_group = payload0
dominant_miss_reason = GDF4-topk-group-concentration
```

H1 认为，如果不消掉这个 concentration，任何 TopK precision 都不能 official。

H1 成立标准：

```text
P1 group autopsy shows:
  payload_norm_bucket explains >= 0.60 of drop;
  removing / balancing payload0 reduces TopK precision by >= 0.30;
  within-payload0 precision is high but cross-bucket precision is low.
```

H1 失败标准：

```text
payload0 只是表面 proxy；
真正 drop 来自 dataset、stratum、family、step bucket 或 horizon bucket。
```

若 H1 失败，v9.6.0 route 应改为对应真实 group axis，不继续围绕 payload_norm_bucket 修。

## H2：存在 group-stable legal rank，但需要 value score 与 risk veto 分离

v9.5.8/v9.5.9 都显示 `ControlTransferImprovement` 对 value 很强，但会带来 longrisk。v9.5.9 P3 中该 feature within-group AUC 很高，但 TopK64 group-balanced longrisk UCB 仍高。

H2 认为单一 score 不够，必须拆成：

$$
Score_{value}(a)
$$

和：

$$
RiskVeto(a)
$$

接受条件应为：

$$
Accept(a)=1
\iff
Rank_{value}(a)\le K
\land
RiskVeto(a)=0
\land
MemoryVeto(a)=0
\land
Cost(a)\le C_{max}.
$$

H2 成立标准：

```text
value-only ranker TopK precision high but longrisk high；
risk veto 单独能把 longrisk UCB 从 >0.30 降到 <=0.10；
combined ranker LDO/LSO drop <=0.25；
TopK87 V LCB > 0。
```

H2 失败标准：

```text
risk veto 一加，coverage 或 value 直接崩；
说明 value 与 longrisk 在当前 feature 空间无法分离。
```

## H3：memory / population-risk offdiag failure 是 longrisk 的主要机制之一

v9.5.9 P8 显示：

```text
best_subcomponent = population_risk_offdiag_fail
P(longrisk | subfail) = 0.873054114158636
P(subfail | longrisk) = 1.0
memory_safe_value_positive_density = 0.021362229102167184
```

H3 认为 longrisk 不是随机坏，而是动作进入 reservoir/noise 或破坏 old family / old stratum 的结果。

H3 成立标准：

```text
population_risk_offdiag_fail explains >=0.80 longrisk；
old-family memory fail explains >=0.60 longrisk；
memory-safe value-positive density can be raised above 0.03 after group balancing or generator reset。
```

H3 失败标准：

```text
longrisk 与 memory/offdiag/cover 无关；
需要寻找新的 longrisk source，例如 optimizer-state interaction 或 horizon-specific batch drift。
```

## H4：APGA generated frontier 失败不是实现问题，而是 OOD + veto ineffective

v9.5.9 P7 显示：

```text
longrisk_created_rate = 0.66796875
Damage_V_integrated_LCB = -0.24198602000541142
D7-longrisk-veto-ineffective fraction = 0.66796875
D2-generated-OOD-payload fraction = 0.279296875
```

H4 认为 APGA 不该继续小修，除非先量化 OOD 与 veto 失效。

H4 成立标准：

```text
APGA generated payload 在 norm / cosine / edge-group / memory / cover 分布上明显偏离 canonical good actions；
OOD score 与 longrisk_created 强相关；
longrisk veto 对 generated payload 的 false negative 率 >=0.50。
```

H4 失败标准：

```text
APGA payload 不 OOD；
失败来自 branch-horizon materializer 或 target label mismatch。
```

## H5：existing-action path 比 generated-action path 更接近 system controller

v9.5.9 的 existing-action ranker 已经达到：

```text
TopK87 GradeAB precision = 0.8505747126436781
V_integrated LCB = 0.15630165181090255
h240 longrisk UCB = 0.03389313880385762
```

只是 LDO/LSO drop 仍为 `0.3563218390804597`。

H5 认为 v9.6.0 主线应优先修 group-stability，而不是立即大规模重写 generator。

H5 成立标准：

```text
经过 group deconfounding 后，existing-action ranker drop <=0.15；
certificate accepted region pass；
selected runtime pass。
```

H5 失败标准：

```text
所有 legal ranker drop 仍 >0.25；
或 accepted region longrisk UCB 仍 >0.10；
则 existing-action route 暂停，转向 generated primitive。
```

---

# 4. 实验总览

v9.6.0 分为 16 个阶段：

```text
P0  v9.5.9 boundary reproduction
P1  group pocket causal autopsy
P2  group-balanced target density and support map
P3  legal feature invariance / deconfounding v2
P4  two-score ranker: value rank + risk veto
P5  group-stable pairwise ranker v2
P6  rank-safe certificate v8
P7  existing-action minimal controller candidate
P8  selected existing-action runtime preflight
P9  APGA OOD / damage decomposition v2
P10 memory / cover / offdiag blocker anatomy v4
P11 APGD1-APGD8 memory-safe longrisk primitive design
P12 APGD branch-horizon smoke outcome
P13 APGD outcome / geometry / OOD pass
P14 integrated route decision
P15 official leaveout + paired replay boundary
P16 Base-Acc Sentinel continuation
```

P0-P8 是 existing-action rank 主线。  
P9-P13 是 generated primitive 主线。  
P14-P16 是 gate / downstream。  
两条主线并行执行，但只有 P7/P8 或 P13 过线后才能打开 P15。

---

# 5. P0 v9.5.9 boundary reproduction

## 目标

确认 v9.5.9 的 failure boundary 稳定复现，避免在错误输入上继续。

## 必须读取 artifact

```text
route_decision_v9590.json
p1_group_drop_autopsy.csv
p2_group_balanced_target_density.csv
p3_feature_invariance.csv
p4_group_invariant_rank.csv
p5_rank_safe_certificate_v7.csv
p7_apga_ood_damage_autopsy.csv
p8_memory_cover_blocker_v3.csv
base_acc_sentinel_v9590.csv
contract_audit_v9590.csv
no_fake_audit_v9590.csv
```

## 必须记录指标

```text
source_route_v9590
p0_pass
group_specific_rank_pocket
max_topK_group_share
max_drop
dominant_drop_axis
dominant_drop_group
best_GIR_id
best_GIR_TopK87_precision
best_GIR_V_integrated_LCB
best_GIR_h240_longrisk_UCB
best_GIR_LDO_drop
best_GIR_LSO_drop
rank_safe_certificate_pass
existing_action_controller_pass
apga_ood_damage_autopsy_pass
memory_cover_blocker_confirmed
system_legal_controller_pass
```

## 成立标准

```text
p0_pass = 1
source route = R1-GroupSpecificRankPocket
canonical truth / no-fake / no-proxy audits pass
```

## 可视化

```text
v9.5.8 -> v9.5.9 drop reduction bar chart
P4 ranker TopK precision / V / longrisk / drop scatter
P5 certificate accepted region quality bar chart
```

---

# 6. P1 group pocket causal autopsy

## 目标

判断 rank pocket 到底是 payload_norm_bucket 导致，还是 payload_norm_bucket 只是其它因素的 proxy。

## 实验设计

构造多个 group axis：

```text
payload_norm_bucket
payload_linf_bucket
action_norm_bucket
step_bucket
event_family
candidate_family
horizon_source
dataset
seed
state_NLL_bucket
hard_tail_fraction_bucket
old_family_memory_bucket
population_risk_offdiag_bucket
```

对每个 axis 做：

```text
1. TopK share；
2. TopK GradeAB precision；
3. TopK V LCB；
4. TopK longrisk UCB；
5. leave-one-group-out drop；
6. within-group rank AUC；
7. cross-group transfer AUC；
8. Simpson reversal test；
9. matched-pair comparison within same family/step/state bucket。
```

## 关键公式

令 group 为 $g$，rank score 为 $s(a)$。

TopK group share：

$$
Share_g(K)=\frac{|\{a\in TopK(s):group(a)=g\}|}{K}
$$

leave-one-group-out drop：

$$
Drop_g = Precision_{pooled}(TopK)-Precision_{leave\ g\ out}(TopK)
$$

matched pair lift：

$$
Lift_{matched}(g)=
E[Y(a_i)-Y(a_j)\mid group(a_i)=g, group(a_j)\neq g, match(i,j)]
$$

## 必须记录指标

```text
axis
num_groups
max_topK_group_share
entropy_topK_group_distribution
max_leave_group_drop
mean_leave_group_drop
within_group_auc_mean
cross_group_auc_mean
matched_pair_lift
matched_pair_p_value
simpson_reversal_flag
dominant_axis
dominant_group
failure_mode
```

## 判断标准

P1 pass 条件：

```text
dominant group pocket 被 >=0.90 attribution coverage 解释；
明确区分 causal group 与 proxy group；
如果 payload_norm_bucket 仍是 dominant，则必须确认其是否只是 action norm / family / memory 的 proxy。
```

P1 fail 条件：

```text
归因 coverage < 0.90；
多个 axis 互相冲突，无法判断 rank pocket 来源；
matched-pair test 无法区分 payload0 与其它 confounder。
```

## 可视化

```text
TopK group share heatmap
leave-one-group-out drop bar chart
within-group vs cross-group AUC scatter
payload_norm_bucket × memory_fail_bucket confusion heatmap
matched-pair lift forest plot
```

---

# 7. P2 group-balanced target density and support map

## 目标

判断当前 canonical AP0 中是否存在足够多、跨 group 分布合理的可选动作，而不是只在某个 pocket 里存在。

## Target 候选

至少扫描以下 target：

```text
GradeAB
MemorySafeGradeB
ValuePositiveNoLongRisk
ValuePositiveMemorySafeNoLongRisk
ValuePositiveNoLongRiskNoBadNoNull
T5-like relaxed geometry
T5 + memory safe
T5 + population risk offdiag pass
T5 + cover safe
T5 + group balanced support
```

## 必须记录指标

```text
target_id
global_count
global_coverage
per_group_min_count
per_group_min_coverage
per_group_precision_mean
per_group_precision_min
max_group_share
V_integrated_LCB
h20/h80/h240 V_LCB
h240_longrisk_UCB
bad_UCB
null_UCB
memory_fail_UCB
cover_collapse_UCB
population_risk_offdiag_fail_UCB
support_family_count
support_dataset_count
support_stratum_count
density_pass
group_balanced_density_pass
```

## 判断标准

Official target density pass：

$$
Coverage\ge0.03
$$

$$
V_{integrated}^{LCB}>0
$$

$$
LongRisk_{240}^{UCB}\le0.05
$$

$$
Bad^{UCB}\le0.05
$$

$$
Null^{UCB}\le0.15
$$

$$
MaxGroupShare\le0.35
$$

$$
MinGroupCoverage\ge0.01
$$

Weak pass：

```text
Coverage >= 0.03
V_integrated_LCB > 0
LongRisk_UCB <= 0.10
MaxGroupShare <= 0.50
```

## 可视化

```text
target coverage vs V LCB scatter
target coverage vs longrisk UCB scatter
group-balanced density heatmap
target membership Venn diagram: GradeAB / ValuePositiveNoLongRisk / MemorySafe / T5
```

---

# 8. P3 legal feature invariance / deconfounding v2

## 目标

把 current legal features 分成三类：

```text
1. value features；
2. risk veto features；
3. confounder / pocket features。
```

不能再把所有 feature 混成一个 score。

## Feature families

```text
F_value:
  ControlTransferImprovement
  AdamWAlignmentGain
  LinearizedCEImprovement
  LinearizedMarginImprovement
  LeaveOneOutTransferApprox

F_risk:
  LongRiskProxy
  PopulationRiskOffdiagFailProxy
  OldFamilyMemoryFailProxy
  HardTailCoverCollapseProxy
  CEp99TailWorsenProxy

F_support:
  family_support_count
  stratum_support_count
  horizon_support_count
  payload_bucket_support_count

F_cost:
  feature_ms
  payload_apply_ms
  memory_delta

F_forbidden_diagnostic_only:
  V_integrated
  risk_score
  GradeAB
  future outcome labels
  dataset-specific branch labels
```

## 实验设计

对每个 feature 记录：

```text
pooled AUC
within-group AUC mean
cross-group AUC mean
TopK64 precision
TopK64 V LCB
TopK64 longrisk UCB
TopK64 max group share
group-balanced TopK64 precision
group-balanced TopK64 V LCB
group-balanced TopK64 longrisk UCB
leave-one-axis drop
```

同时做 residualization：

$$
f_{resid}(a)=f(a)-E[f(a)\mid group(a)]
$$

但 deployed score 不能直接使用 group id；residualization 只作为 diagnostic 和 training-time debiasing reference。

## 判断标准

Feature invariant weak pass：

```text
within_group_auc_mean >= 0.70
cross_group_auc_mean >= 0.65
TopK64 group-balanced precision >= 0.50
TopK64 longrisk UCB <= 0.20
max_group_share <= 0.50
```

Feature invariant official pass：

```text
TopK87 GradeAB precision >= 0.75
V_integrated_LCB > 0
longrisk UCB <= 0.05
LDO/LSO drop <= 0.15
max_group_share <= 0.35
```

## 可视化

```text
feature AUC vs TopK precision scatter
feature TopK precision vs longrisk scatter
residualized feature before/after group share histogram
feature correlation graph with group axes
```

---

# 9. P4 two-score ranker: value rank + risk veto

## 目标

验证 “value signal 和 risk signal 分裂” 的假设。不要再用一个 score 同时排序 value 和 risk。

## Ranker 形式

定义：

$$
S_V(a)=f_V(x_a)
$$

$$
S_R(a)=f_R(x_a)
$$

$$
S_M(a)=f_M(x_a)
$$

接受规则：

$$
Accept(a)=1
\iff
S_V(a)\in TopK
\land
S_R(a)\le \tau_R
\land
S_M(a)\le \tau_M
\land
Cost(a)\le C_{max}
$$

其中：

```text
S_V: value rank
S_R: longrisk / bad risk veto
S_M: memory / cover veto
```

## 候选模型

```text
VR1-ControlTransferOnly
VR2-ControlTransfer+AdamWAlignment
VR3-LinearizedCE+Margin
VR4-LeaveOneOutTransferApprox
RV1-PopulationRiskOffdiagVeto
RV2-OldFamilyMemoryVeto
RV3-HardTailCoverVeto
RV4-CombinedLongRiskVeto
MR1-MemoryAndCoverVeto
MR2-CostConstrainedMemoryVeto
COMBO1-VR1+RV1
COMBO2-VR2+RV1+MR1
COMBO3-VR4+RV4+MR2
```

## 必须记录指标

```text
model_id
score_fields
red_field_count
TopK64/87/128 precision
TopK64/87/128 V_integrated_LCB
TopK64/87/128 longrisk_UCB
TopK64/87/128 bad_UCB
TopK64/87/128 null_UCB
max_group_share
LDO_drop_max
LSO_drop_max
leave_payload_bucket_drop_max
leave_family_drop_max
feature_cost_q90_ms
memory_ratio_estimate
```

## 判断标准

P4 weak pass：

```text
TopK87 GradeAB precision >= 0.75
V_integrated_LCB > 0
h240 longrisk_UCB <= 0.10
LDO/LSO drop <= 0.25
max_group_share <= 0.50
```

P4 official candidate pass：

```text
TopK87 GradeAB precision >= 0.75
V_integrated_LCB > 0
h240 longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
LDO/LSO drop <= 0.15
max_group_share <= 0.35
feature_cost_q90_ms <= budget
```

## 可视化

```text
value score vs risk score quadrant plot
accepted region group distribution
TopK K-sweep: precision/V/longrisk/drop vs K
risk veto ablation waterfall
```

---

# 10. P5 group-stable pairwise ranker v2

## 目标

在 P4 的 two-score 结构基础上，训练或构造 group-stable ranker，专门压 LDO/LSO drop。

## 训练设置

ranker 不能使用：

```text
dataset_name
group_id
future outcome
V_integrated
risk_score
GradeAB
validation/test metrics
```

允许使用 group 作为 training/evaluation split：

```text
batch reweighting
pairwise within-group loss
pairwise cross-group consistency loss
worst-group validation
adversarial group prediction penalty
```

Loss：

$$
L = L_{pairwise} + \lambda_1 L_{worst-group} + \lambda_2 L_{group-confusion} + \lambda_3 L_{risk-veto}
$$

其中：

$$
L_{pairwise}=\sum_{(i,j)}\max(0,1-s_i+s_j)
$$

$$
L_{worst-group}=\max_g L_g
$$

$$
L_{group-confusion}= - H(\hat g \mid s)
$$

risk veto loss：

$$
L_{risk-veto}=\sum_i y^{risk}_i \max(0,\tau_R-S_R(i))
$$

## 候选 ranker

```text
GIR10-WorstGroupPairwiseRanker
GIR11-AdversarialGroupInvariantRanker
GIR12-ValueRiskSeparatedRanker
GIR13-GroupBalancedConformalRanker
GIR14-LeavePayloadBucketOutRanker
GIR15-FamilyStratumBalancedRanker
```

## 必须记录指标

```text
ranker_id
red_field_count
train_group_axes
TopK87 GradeAB precision
TopK87 V_integrated_LCB
TopK87 h240 longrisk_UCB
TopK87 bad_UCB
TopK87 null_UCB
max_topK_group_share
entropy_topK_group_distribution
LDO_drop_max
LSO_drop_max
leave_payload_bucket_drop_max
leave_family_drop_max
leave_step_bucket_drop_max
worst_group_precision
worst_group_V_LCB
worst_group_longrisk_UCB
```

## 判断标准

Group-stable ranker pass：

```text
TopK87 GradeAB precision >= 0.75
V_integrated_LCB > 0
h240 longrisk_UCB <= 0.05
LDO_drop_max <= 0.15
LSO_drop_max <= 0.15
leave_payload_bucket_drop_max <= 0.15
max_topK_group_share <= 0.35
worst_group_precision >= 0.50
```

Weak pass：

```text
TopK87 precision >= 0.75
V LCB > 0
longrisk UCB <= 0.10
max drop <= 0.25
```

## 可视化

```text
ranker comparison radar chart
leave-group drop heatmap
TopK group entropy curve
worst-group precision/V/longrisk table
```

---

# 11. P6 rank-safe certificate v8

## 目标

把 P5 ranker 变成可 official 的 accepted region，而不是只保留 TopK diagnostic。

## Certificate 形式

Certificate 不再要求输出校准概率。它可以是 rank-safe set：

$$
Cert(a)=1
\iff
s_V(a)\ge q_V
\land
s_R(a)\le q_R
\land
s_M(a)\le q_M
\land
support(a)\ge q_S
\land
cost(a)\le q_C
$$

阈值必须只在 calibration split 冻结一次。

## 候选 certificate

```text
RC8-GroupStableTopKFixedCount
RC9-WorstGroupConformalRiskBound
RC10-ValueRiskTwoScoreConformal
RC11-MemoryCoverVetoCertificate
RC12-CostConstrainedGroupStableCertificate
RC13-LeavePayloadBucketRobustCertificate
```

## 必须记录指标

```text
certificate_id
source_ranker_id
thresholds
calibration_accepted_count
heldout_accepted_count
coverage_heldout
GradeAB_precision_heldout
V_integrated_LCB_heldout
h240_longrisk_UCB_heldout
bad_UCB_heldout
null_UCB_heldout
max_group_share_heldout
LDO_drop_max
LSO_drop_max
leave_payload_bucket_drop_max
feature_cost_q90_ms
certificate_compute_ms_q90
```

## 判断标准

Certificate official pass：

$$
Accepted_{heldout}\ge87
$$

$$
0.03\le Coverage\le0.15
$$

$$
Precision_{GradeAB}\ge0.75
$$

$$
V_{integrated}^{LCB}>0
$$

$$
LongRisk_{240}^{UCB}\le0.05
$$

$$
Bad^{UCB}\le0.05
$$

$$
Null^{UCB}\le0.15
$$

$$
Drop_{LDO/LSO}\le0.15
$$

如果 coverage 和 value pass，但 risk UCB 在 $0.05$ 到 $0.10$ 之间，记为 weak diagnostic，不许打开 P7 official controller。

## 可视化

```text
accepted region K sweep
calibration vs heldout quality plot
risk conformal bound plot
accepted group distribution heatmap
```

---

# 12. P7 existing-action minimal controller candidate

## 目标

只在 P5/P6 过线时，构建 official candidate controller。

## Controller 定义

Controller 必须是最小化形式：

$$
Controller(a)=Cert(a)
$$

不允许再加隐藏后验判断。

## 必须记录指标

```text
controller_id
ranker_id
certificate_id
feature_list
red_field_count
accepted_count
coverage
precision
V_integrated_LCB
h20/h80/h240 V_LCB
h240_longrisk_UCB
bad_UCB
null_UCB
memory_fail_UCB
cover_fail_UCB
support_family_count
support_dataset_count
support_stratum_count
LDO_drop_max
LSO_drop_max
compute_cost_ms_q90
```

## 判断标准

Controller pass 等于 P6 official pass 加上：

```text
red_field_count = 0
uses_dataset_name_for_controller = 0
uses_future_outcome_for_features = 0
uses_validation_or_test_for_controller = 0
```

## 可视化

```text
controller accepted rows timeline
accepted vs rejected geometry distribution
controller component ablation
```

---

# 13. P8 selected existing-action runtime preflight

## 目标

测 selected controller 的真实 online runtime，不再只测 diagnostic ranker。

## Runtime path

必须包括：

```text
feature compute
rank score compute
certificate compute
payload lookup
payload apply
base AdamW step
post-step audit subset
```

不得包括：

```text
offline materializer
future horizon replay
paired replay
oracle label computation
```

## 必须记录指标

```text
step_count
active_step_count
accepted_step_count
feature_compute_ms_q50/q90/q99
rank_compute_ms_q50/q90/q99
certificate_compute_ms_q50/q90/q99
payload_lookup_ms_q90
payload_apply_ms_q90
base_step_ms_q90
controller_step_ratio_q90
memory_ratio
kernel_count
sync_count
zero_candidate_launch_count
selected_runtime_pass
```

## 判断标准

$$
StepRatio_{q90}\le1.50
$$

$$
MemoryRatio\le1.05
$$

```text
zero_candidate_launch_count = 0
payload_apply_error_linf_max <= 1e-6
no_event_preservation_pass = 1
```

## 可视化

```text
runtime waterfall chart
per-step latency histogram
active vs inactive step runtime scatter
memory ratio trace
```

---

# 14. P9 APGA OOD / damage decomposition v2

## 目标

在不急着新增 generator 的情况下，先解释 APGA 为什么毁掉 good geometry。

## OOD 指标

比较三类 action：

```text
canonical good AP0 actions
canonical bad/longrisk AP0 actions
APGA generated actions
```

记录：

```text
payload_norm
payload_linf
payload_sparsity
edge_group_norm
basis_group_norm
action_adamw_cosine
action_control_transfer_score
action_memory_similarity
population_risk_offdiag_proxy
cover_delta_proxy
state_NLL_bucket
hard_tail_fraction_bucket
```

## 必须记录指标

```text
OOD_MMD
OOD_Mahalanobis
OOD_payload_norm_KS
OOD_edge_group_KS
OOD_memory_similarity_KS
P(longrisk_created | OOD_high)
P(OOD_high | longrisk_created)
longrisk_veto_false_negative_rate
source_good_destroyed_rate
new_positive_created_rate
Damage_V_integrated_LCB
```

## 判断标准

APGA OOD confirmed：

```text
OOD_MMD significant；
P(longrisk_created | OOD_high) >= 0.50；
longrisk_veto_false_negative_rate >= 0.50。
```

If not confirmed：

```text
investigate materializer / label / target mismatch rather than generator OOD。
```

## 可视化

```text
PCA/UMAP of action payload geometry
AP0-good vs APGA feature distribution histograms
OOD score vs longrisk scatter
veto false negative confusion matrix
```

---

# 15. P10 memory / cover / offdiag blocker anatomy v4

## 目标

把 longrisk 的机制拆清楚，尤其是 memory、cover、population-risk offdiag 三者的关系。

## 记录对象

```text
canonical AP0 actions
existing-rank accepted candidates
APGA generated actions
future APGD generated actions
```

## 指标

```text
old_family_fail
old_stratum_fail
memory_margin_p10_delta
forget_risk_score
cover_entropy_delta
basis_effective_rank_delta
hard_tail_cover_entropy_delta
population_risk_offdiag_score
signal_reservoir_ratio
gradient_mean_square
gradient_variance_scaled
SNR_gate = mu^2 - sigma^2/(b-1)
```

## 公式

SNR gate：

$$
SNR_k=\mu_k^2-\frac{\sigma_k^2}{b-1}
$$

Population-risk offdiag proxy：

$$
\Omega_B=\frac{1}{b(b-1)}\sum_{a\ne c} r_a^T K_{ac} r_c
$$

Memory-safe condition：

$$
MemorySafe(a)=1
\iff
ForgetRisk^{UCB}(a)\le \tau_F
\land
OldFamilyFail(a)=0
\land
OldStratumFail(a)=0
$$

## 判断标准

Memory/cover official blocker confirmed：

```text
P(longrisk | memory/offdiag/cover subfail) >= 0.80
P(subfail | longrisk) >= 0.80
memory-safe value-positive density < 0.03
```

If memory-safe value-positive density >=0.03：

```text
existing-action controller should prioritize memory-safe subset。
```

## 可视化

```text
longrisk attribution waterfall
memory/offdiag/cover Venn diagram
SNR vs longrisk scatter
cover entropy delta vs GradeAB plot
```

---

# 16. P11 APGD1-APGD8 memory-safe longrisk primitive design

## 目标

只有在 P5/P6 existing-action path 未过，且 P9/P10 解释清楚 APGA OOD/longrisk 后，才 materialize APGD。

APGD 不是 APGA 小变体，而是围绕三个硬约束生成动作：

```text
1. 不偏离 AP0-good payload geometry；
2. 不破坏 old family / old stratum；
3. longrisk veto 在生成时就生效。
```

## APGD primitive 候选

```text
APGD1-AP0GoodAnchorProjection
APGD2-MemorySafeResidualBlend
APGD3-PopRiskOffdiagPositiveGate
APGD4-CoverEntropyPreservingUpdate
APGD5-AdamWCompatibleLowNormTrustRegion
APGD6-ValueRiskTwoScoreProjectedUpdate
APGD7-OldFamilyConstrainedEdgeUpdate
APGD8-NegativeControlShuffledAnchor
```

## 生成约束

对每个 generated action $a'$：

$$
\|a'-a_{anchor}\|_2 \le \epsilon_{anchor}
$$

$$
Cos(a',a_{AdamW})\ge \tau_{cos}
$$

$$
ForgetRiskProxy(a')\le \tau_F
$$

$$
OffdiagFailProxy(a')=0
$$

$$
CoverCollapseProxy(a')=0
$$

$$
PayloadNormBucket(a') \notin OODHigh
$$

## 必须记录指标

```text
primitive_id
generated_action_count
anchor_source
payload_hash_missing
certificate_hash_missing
action_apply_error_linf_max
no_transform_equivalence
negative_control_divergence
precommit_certificate_fields
OOD_score
memory_proxy
cover_proxy
offdiag_proxy
cost_proxy
```

## 判断标准

Implementation pass：

```text
generated_action_count >= 512
payload/certificate hash missing = 0
action_apply_error_linf_max <= 1e-6
negative control fails
```

不在 P11 判断 science pass；science pass 在 P13。

---

# 17. P12 APGD branch-horizon smoke outcome

## 目标

对 APGD generated actions 真实跑 branch-horizon outcomes。

## 分支

至少包括：

```text
RealAPGD
AdamWOnly
AdamWParallel
BestLR
NoOp
RandomPayload
ShuffledAPGD
CertificatePassNoPayload
```

## Horizons

```text
h20
h80
h240
```

## 必须记录指标

```text
row_count_expected
row_count_actual
branch_completion_rate
horizon_completion_rate
secondary_delta_completion_rate
unresolved_exception_count
label_exclusivity_violation_count
duplicate_row_id_count
NaN/Inf count
rows_per_sec
quality_audit_pass
```

## 判断标准

```text
row_count_actual = row_count_expected
quality_audit_pass = 1
unresolved_exception_count = 0
```

## 可视化

```text
branch completion matrix
rows/sec per primitive
horizon metric distributions
```

---

# 18. P13 APGD outcome / geometry / OOD pass

## 目标

判断 APGD 是否真的生成了 value-positive、memory-safe、longrisk-safe 动作。

## 必须记录指标

```text
primitive_id
GradeAB_precision
T5_precision
ValuePositiveNoLongRisk_precision
V_integrated_LCB
h20/h80/h240 V_LCB
h240_longrisk_UCB
bad_UCB
null_UCB
memory_fail_UCB
cover_fail_UCB
population_risk_offdiag_fail_UCB
OOD_score_mean
OOD_high_rate
new_positive_created_rate
longrisk_created_rate
source_good_destroyed_rate
Damage_V_integrated_LCB
```

## 判断标准

APGD weak pass：

```text
GradeAB precision >= 0.25
V_integrated_LCB > 0
h240 longrisk_UCB <= 0.10
new_positive_created_rate >= 0.10
OOD_high_rate <= 0.20
```

APGD official candidate pass：

```text
accepted_count >= 87
coverage >= 0.03
GradeAB precision >= 0.75
V_integrated_LCB > 0
h240 longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
memory_fail_UCB <= 0.05
OOD_high_rate <= 0.10
```

## 可视化

```text
primitive outcome radar chart
source-to-generated damage waterfall
OOD score vs value plot
memory/cover/offdiag fail heatmap
```

---

# 19. P14 integrated route decision

## 目标

统一 existing-action route 和 generated-action route，避免两条线互相抢功或误报。

## Route 规则

```text
R0-BoundaryReproductionFail:
  P0 fail。

R1-GroupPocketMechanismUnresolved:
  P1 归因不足。

R2-GroupBalancedTargetAbsent:
  P2 没有 weak target。

R3-LegalRankGroupStablePass:
  P5/P6/P7 pass，P8 pending or pass。

R4-LegalRankGroupStableFail:
  P5/P6 fail 且 drop > 0.25。

R5-RankCertificateAcceptedRegionFail:
  P5 pass but P6 fail。

R6-ExistingActionRuntimeFail:
  P7 pass but P8 fail。

R7-APGAOODMechanismUnresolved:
  P9/P10 无法解释 generated failure。

R8-APGDGeneratedFrontierPass:
  P13 pass。

R9-APGDGeneratedFrontierFail:
  P13 fail。

R10-SystemLegalControllerPass:
  controller + runtime pass。
```

## 必须输出

```text
route_decision_v9600.json
failure_taxonomy_v9600.csv
next_required_implementation
stop_conditions
allowed_downstream_gates
```

---

# 20. P15 official leaveout + paired replay boundary

## 打开条件

只在以下条件同时成立时打开：

```text
source_controller_pass = 1
selected_runtime_pass = 1
system_legal_controller_pass = 1
no_fake = 1
no_proxy = 1
red_field_count = 0
```

## Leaveout

记录：

```text
leave_dataset_out_precision
leave_dataset_out_V_LCB
leave_dataset_out_longrisk_UCB
leave_stratum_out_precision
leave_stratum_out_V_LCB
leave_stratum_out_longrisk_UCB
leave_payload_bucket_out_precision
leave_payload_bucket_out_V_LCB
leave_payload_bucket_out_longrisk_UCB
```

pass：

```text
all drop <= 0.15
all V_LCB > 0
all longrisk_UCB <= 0.05
```

## Paired replay

比较：

```text
RealFunctional
AdamWOnly
AdamWParallel
BestLR
NoOp
RandomPayload
ShuffledPayload
```

记录：

```text
final_acc_delta
CE_AUC_delta
NLL_delta
ECE_delta
CEp99_delta
margin_p10_delta
hard_tail_acc_delta
runtime_delta
memory_delta
```

pass：

```text
RealFunctional beats AdamWParallel and BestLR on value/risk composite；
ShuffledPayload fails；
NoOp fails；
runtime still <= 1.50 step ratio。
```

---

# 21. P16 Base-Acc Sentinel continuation

## 目标

继续健康检查，但不用于 selector/controller。

## 数据集

```text
MNIST
Fashion-MNIST
KMNIST
```

## 模型

```text
LQ-t2-h256
MatchedMLP
QuadraticFeatureMLP
AdamWStrongLRGridMLP
```

## 记录指标

```text
dataset
seed
model
test_acc
val_acc
train_acc
NLL
ECE
CEp99
steps
wallclock
catastrophic_fail
```

## 判断

```text
base_acc_used_for_controller = 0
```

Sentinel 只用于确认 base 没崩；不能作为 functional success。

---

# 22. 并行执行方案

为了避免“一轮只暴露一个 blocker”，v9.6.0 必须并行：

```text
Batch A: P0/P1/P2
  group pocket autopsy + target density。

Batch B: P3/P4/P5
  legal features + two-score rank + group-stable ranker。

Batch C: P6/P7/P8
  certificate + existing-action controller + runtime preflight。

Batch D: P9/P10
  APGA OOD + memory/cover/offdiag anatomy。

Batch E: P11/P12/P13
  APGD generation + branch-horizon outcome + generated frontier。

Batch F: P16
  Base-Acc Sentinel continuation。
```

P11-P13 不需要等 P6 完全失败才开始，但 P13 不能 official promotion，除非 P9/P10 已解释 APGA failure 且 APGD obeys no-fake/no-proxy/action-apply contracts。

---

# 23. 必须落盘 artifacts

```text
p0_v9590_boundary_reproduction.csv
p1_group_pocket_causal_autopsy.csv
p1_matched_pair_lift_trace.csv
p2_group_balanced_target_density_map.csv
p3_feature_invariance_deconfounding_v2.csv
p4_two_score_ranker_value_risk_veto.csv
p5_group_stable_pairwise_ranker_v2.csv
p6_rank_safe_certificate_v8.csv
p7_existing_action_minimal_controller.csv
p8_selected_existing_action_runtime_preflight.csv
p9_apga_ood_damage_decomposition_v2.csv
p10_memory_cover_offdiag_blocker_v4.csv
p11_apgd_memory_safe_longrisk_primitive.csv
p12_apgd_branch_horizon_smoke_outcome.csv
p13_apgd_outcome_geometry_ood_pass.csv
p14_integrated_route_decision_v9600.json
p15_leaveout_paired_replay_boundary.csv
p16_base_acc_sentinel_v9600.csv
contract_audit_v9600.csv
no_fake_audit_v9600.csv
provenance_audit_v9600.csv
failure_taxonomy_v9600.csv
run_manifest_v9600.json
```

---

# 24. 最终成功标准

v9.6.0 可以声明 **system-legal local functional controller candidate**，当且仅当：

```text
1. canonical truth base ready；
2. no fake / no proxy / no CPU offload；
3. red field count = 0；
4. controller accepted region pass；
5. LDO/LSO/drop gates pass；
6. selected runtime pass；
7. Base-Acc Sentinel not catastrophic；
8. no dataset-specific dispatch。
```

v9.6.0 不允许声明：

```text
strict PureKAN functional success；
full functional success；
external ready；
Beyond-MLP final success；
```

除非 P15 official paired replay 与 short/full boundary 也真实打开并过线。

---

# 25. 最终判断模板

本轮最终分析必须明确写：

```text
1. group pocket 是否被解释；
2. existing-action legal rank 是否还能作为主线；
3. group-invariant ranker 是否从 diagnostic 变成 official candidate；
4. rank-safe certificate 是否把 TopK 变成 accepted region；
5. APGA failure 是否来自 OOD / longrisk veto / memory；
6. APGD 是否生成了新 frontier；
7. 是否可以打开 selected runtime；
8. 是否可以打开 leaveout / paired replay / short-full。
```

如果 P5/P6/P8 失败，但 P13 成功：

```text
route = R8-APGDGeneratedFrontierPass
next = APGD certificate/controller/runtime
```

如果 P5/P6 成功，但 P8 失败：

```text
route = R6-ExistingActionRuntimeFail
next = selected runtime kernel/path optimization
```

如果 P5/P6/P13 全失败：

```text
route = R11-LegalRankAndGeneratedPrimitiveBothFail
next = revisit geometry objective / candidate source definition
```

