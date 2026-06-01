# DG-KAN v9.5.9 Group-Invariant Legal Rank / Existing-Action Controller / Generator OOD Triage 完整实验计划

> 本计划基于 v9.5.8 `Group-Stable Legal Rank / Memory-Safe Primitive` 的真实执行结果制定。v9.5.8 的 route 是 `R1-LegalRankStillGroupUnstable`，说明当前已经不是 canonical outcome table、payload replay、field legality、APGA implementation 或 branch-horizon materializer 的问题，而是：legal rank 在 pooled TopK 上已经很强，但对 leave-dataset-out / leave-stratum-out 极不稳定；同时 APGA 新生成动作仍然 value-negative / high-longrisk。
>
> v9.5.9 不继续小修 APGA、CERT-R2 或单个 rank threshold，而是直接检验一个更根本的问题：
>
> $$
> \boxed{\text{好动作的 legal rank 是否只是少数 group 的局部规律，还是能转成 group-invariant controller？}}
> $$
>
> 本计划中所有公式使用 `$...$` 或 `$$...$$`，Typora 友好。

---

# 0. 执行摘要

v9.5.8 的关键信息如下：

```text
route = R1-LegalRankStillGroupUnstable
base_candidate = LQ-t2-h256
strict_purekan_functional = False
full_functional = False
external_ready = False
```

v9.5.8 的强进展不是 APGA，而是 legal rank。最好的 `R2-ValueRankLongRiskVeto` 在 TopK87 上达到：

```text
TopK87 GradeAB precision = 0.8505747126436781
TopK87 V_integrated LCB = 0.13419830825382068
TopK87 h240 longrisk UCB = 0.0
```

但同一个 ranker 的 group stability 彻底不过：

```text
LDO drop max = 0.8854166666666666
LSO drop max = 0.8854166666666666
group_stable_rank_weak_pass = 0
```

这意味着 pooled TopK 已经能找到一批好动作，但这批动作很可能集中在某些 dataset / family / stratum / step bucket / action source 的局部区域里。它不是可以直接 online 部署的通用规则。

APGA1-APGA8 的工程链路也闭合：

```text
generated_actions = 512
payload_hash_missing = 0
certificate_hash_missing = 0
action_apply_linf_max = 0.0
branch_horizon_rows = 12288 / 12288
quality_audit_pass = 1
```

但最好的 APGA7 仍然失败：

```text
GradeAB precision = 0.03125
V_integrated LCB = -0.3580921649906729
h240 longrisk UCB = 0.8150608413036654
```

因此 v9.5.9 的总目标不是“再调 APGA7”或“再调 R2 threshold”，而是并行回答四个问题：

```text
1. legal rank 为什么 pooled TopK 强、LDO/LSO 却崩？
2. 是否存在不使用 dataset-specific branch 的 group-invariant legal rank？
3. existing AP0 actions 是否已经足够形成一个 group-stable controller？
4. APGA 为什么生成动作远离好动作区域，是否应该暂时停止 generated-action 主线？
```

v9.5.9 的最低有效推进是：即使不能 system pass，也必须输出一个明确判定：

```text
Case A: existing-action legal rank 可以通过 group-stable controller gate；
Case B: legal rank 只能做 pooled diagnostic，不能 official；
Case C: target/action density 本身在 group-balanced 约束下不足；
Case D: generator OOD / destructive 是主因，必须重置生成器目标；
Case E: memory-cover 机制解释不够，必须回到几何定义。
```

---

# 1. 对 v9.5.8 的独立判断

## 1.1 有进展，但不是最终能力进展

v9.5.8 的实质进展有三点。

第一，legacy rank 的非法强信号已经被持续隔离。field legality ledger 仍然通过，legacy rank 强但含 red fields，best legal score variant 仍很弱：

```text
legacy best TopK64 GradeAB precision = 0.890625
legacy best red field count = 2
best legal variant TopK64 GradeAB precision = 0.09375
best legal variant V_integrated LCB = -0.3660500491083909
```

这说明我们没有再把 outcome-derived rank 当成 official。

第二，legal rank 不再完全无信号。R2/R1/R5 这些 ranker 在 pooled TopK87 上能抓到 high GradeAB、positive value、low longrisk 的区域。这个结果比 v9.4.x / v9.5.0 的“legal completely opaque”强很多。

第三，APGA 的失败不再是 implementation failure。payload、certificate、action apply、branch-horizon materializer 全都闭合；失败发生在 outcome：value negative、longrisk high。

## 1.2 当前最大问题不是 AUC，也不是 TopK，而是 group stability

如果只看 pooled TopK，v9.5.8 已经很像成功：

$$
Precision_{TopK87}=0.8506,
\quad
V_{LCB}=0.1342,
\quad
LongRisk_{UCB}=0.
$$

但 LDO/LSO drop 接近 0.885，说明这个 ranker 一旦离开训练过的 group，就几乎丢掉了它的好区域。

这不是小校准问题。它更像：

$$
\boxed{\text{ranker 学到的是 group-specific pocket，不是 universal geometry rule。}}
$$

也就是说，它可能知道“某几个 family / stratum / step bucket 里的动作好”，但不知道“为什么这些动作好”。这和我们的终极目标不一致，因为终极目标要求 dataset-agnostic、leaveout-stable、paired-replay-valid 的 functional update。

## 1.3 当前 generator 方向也不是差一点

APGA7 的结果是：

```text
GradeAB precision = 0.03125
V_integrated LCB = -0.3581
h240 longrisk UCB = 0.8151
longrisk created rate = 0.703125
```

这不是 threshold 能修的问题。APGA 现在做到了“合法生成”，但没有做到“生成好的训练形状”。它把一些 signal / memory / cover 的想法写进了 payload，但生成出的动作仍然远离 existing good-action manifold。

所以 v9.5.9 必须先做 APGA OOD / damage triage：

```text
APGA 是把 source-side good pocket 破坏了？
APGA 是产生了 payload distribution shift？
APGA 是 value direction 错？
APGA 是 memory guard 过弱？
APGA 是 cover / rank 保护没有真实约束？
```

如果这些问题不拆清楚，继续 APGA9/APGA10 就只是换名字。

## 1.4 现在离目标还差多远

已经完成或基本完成：

```text
canonical outcome table
payload replay
action apply
field legality ledger
legacy leakage isolation
Geometry / Grade ledger
legal rank pooled TopK signal
APGA implementation / materializer
Base-Acc Sentinel health
```

仍然没完成：

```text
group-stable legal rank
minimal geometry controller
selected runtime
official leaveout
official paired replay
short/full functional training
generated-action positive frontier
memory/cover preserving primitive
```

当前不是“快成功”。更准确是：

$$
\boxed{\text{我们第一次看到了可用 rank 的影子，但它还没有跨 group 稳定。}}
$$

---

# 2. v9.5.9 总体目标

v9.5.9 的总体目标是：

$$
\boxed{\text{把 pooled legal rank 变成 group-invariant legal controller，或者明确证明它不能转正。}}
$$

同时，v9.5.9 要把 generated-action 方向从“继续造 APGA 小变体”改为：

$$
\boxed{\text{先解释 APGA 为什么离开好动作区域，再决定是否重置生成器。}}
$$

本轮强目标：

```text
group_stable_legal_rank_pass = 1
minimal_geometry_controller_pass = 1
selected_runtime_pass = 1
system_legal_controller_pass = 1
```

本轮最低有效推进目标：

```text
1. group drop autopsy 完成，并能解释 >= 80% LDO/LSO drop；
2. 输出 group-balanced target density map；
3. 训练并冻结至少 3 类 group-invariant legal ranker；
4. 明确 existing-action controller 是否可行；
5. APGA OOD / damage matrix 完整；
6. 若 generated path 继续，APGI/APGC 不再只是 payload builder，而要有 memory-cover constraints；
7. 所有 controller 评估均必须使用 frozen split，不能用 dataset-specific threshold；
8. 如果 controller 不过，P14/P15/P16 不打开。
```

---

# 3. 本轮硬约束

继续遵守：

```text
no teacher for official controller
no self-teacher
no distillation promoted to official
no loss modification
no label smoothing / focal / margin loss
no dataset-name branch in selector/controller
no validation/test metric at commit time
no future outcome feature
no outcome-at-commit feature
no old table official
no fake/proxy rows
no CPU offload
manual forward / manual backward / manual AdamW update remains required
```

允许使用：

```text
legacy rank as diagnostic only
green-field distillation sandbox diagnostic only
legal features computed at commit time
group labels for audit and split, but not as dataset-specific dispatch
family/stratum/step bucket diagnostics
group-balanced calibration
conformal or Wilson bounds computed on calibration split
heldout frozen evaluation
negative controls
Base-Acc Sentinel isolated health check
```

特别禁止：

```text
不能因为 R2 TopK87 好就直接打开 controller；
不能因为 CERT-R2 TopK64 好就忽略 heldout V LCB 为负；
不能调 APGA7 threshold 写成 generator pass；
不能让 LDO/LSO drop 用 pooled metrics 掩盖；
不能按 dataset 定不同阈值；
不能用 GradeAB / V_integrated / risk_score 这类结果字段进入 official rank。
```

---

# 4. 核心假设

## H1：当前 rank failure 主要来自 group-specific pocket，而不是 rank 完全无效

H1 认为 pooled TopK 强、LDO/LSO 崩，是因为 ranker 抓住了少数 group 中的好动作，但没有学到跨 group 的共同条件。

成立标准：

```text
P1 group drop autopsy identifies dominant drop groups;
TopK composition max_group_share > 0.40 or entropy too low;
within-good-groups precision high, heldout groups precision low;
feature distribution PSI/KL across good/bad groups high;
rank score calibration differs across groups.
```

失败标准：

```text
TopK group composition balanced;
每个 group 都有足够 positive；
但 LDO/LSO 仍崩。
```

如果 H1 失败，说明问题可能是 label/target instability 或 value estimate variance，而不是 group pocket。

## H2：存在 group-invariant legal rank，但需要 group-balanced objective

H2 认为 R2/R1/R5 已经有信息，只是训练目标过于 pooled。用 group-balanced loss、min-group value/risk bound、group-normalized scores 后，可以降低 LDO/LSO drop。

成立标准：

```text
TopK87 GradeAB precision >= 0.65
V_integrated LCB > 0
h240 longrisk UCB <= 0.05
bad UCB <= 0.05
null UCB <= 0.15
LDO_drop_max <= 0.20
LSO_drop_max <= 0.20
max_family_share <= 0.35
```

强成立标准：

```text
TopK87 GradeAB precision >= 0.75
V_integrated LCB >= 0.05
h240 longrisk UCB <= 0.03
LDO/LSO drop <= 0.15
```

失败标准：

```text
任何 group-balanced legal ranker 都无法同时保持 value > 0 与 longrisk <= 0.05；
或者 LDO/LSO drop 仍 > 0.50。
```

## H3：existing-action controller 可能比 generated-action primitive 更接近 success

H3 认为 current AP0 / existing action universe 已经有可用 rank pockets，先把 existing-action selector 做稳，比继续生成 APGA 动作更合理。

成立标准：

```text
P6 existing-action controller pass;
P7 runtime preflight pass;
P8 leaveout pass;
P9 paired replay scout pass.
```

失败标准：

```text
existing-action group-balanced rank 仍不能过 LDO/LSO；
accepted region 不足或 V LCB 为负；
```

若 H3 失败，主线转到 APGI/APGC generator reset。

## H4：APGA 失败来自 generated action OOD 或 memory-cover damage

成立标准：

```text
APGA generated actions 在 payload/stat/geometry distribution 上与 GradeAB existing actions PSI/KL 高；
source-to-generated Damage LCB < 0；
new_positive_created_rate 很低；
longrisk_created_rate 高；
cover/memory deltas 与 longrisk 强相关。
```

失败标准：

```text
APGA generated actions 与 good existing actions 分布接近，但仍 outcome bad；
```

若 H4 失败，说明 geometry metrics 仍缺关键维度。

## H5：memory 是 longrisk 的主要原因，但当前 memory 指标还不够

v9.5.7 memory explained fraction 是 0.7638，v9.5.8 纳入更多 generated rows 后是 0.6981，接近但略低于 0.70。H5 认为 memory 仍是主要因素，但需要拆分 old family、old stratum、hard-tail memory、cover entropy 四类。

成立标准：

```text
at least one memory subcomponent explains >= 0.75 high-longrisk rows；
P(longrisk | memory_subfail) >= 0.80；
P(memory_subfail | longrisk) >= 0.65；
```

失败标准：

```text
所有 memory subcomponent 都 < 0.60；
longrisk 主要由其他因素解释。
```

---

# 5. 分阶段实验计划

## P0：v9.5.8 boundary reproduction

### 目标

确认本轮没有跳过 v9.5.8 的核心失败边界。

### 必须记录

```text
source_route_v9580
field_legality_ledger_pass
legacy_rank_leakage_explained
target_density_pass
legal_feature_weak_pass
group_stable_rank_weak_pass
best_ranker_id
best_ranker_TopK87_GradeAB_precision
best_ranker_V_integrated_LCB
best_ranker_h240_longrisk_UCB
best_ranker_LDO_drop_max
best_ranker_LSO_drop_max
distillation_pass
apga_implementation_pass
apga_branch_horizon_pass
apga_weak_pass
certificate_official_pass
system_legal_controller_pass
```

### 判断标准

P0 pass：

```text
source_route_v9580 = R1-LegalRankStillGroupUnstable
group_stable_rank_weak_pass = 0
apga_weak_pass = 0
certificate_official_pass = 0
system_legal_controller_pass = 0
```

### 可视化

```text
p0_v9580_boundary_dashboard.svg
p0_gate_waterfall.svg
```

---

## P1：Group drop autopsy

### 目标

解释为什么 pooled TopK87 很强，但 LDO/LSO drop 高达约 0.885。P1 是本轮第一科学核心。

### 方法

对 R1/R2/R5/R7 和 v9.5.7 UB-M1 ranker 做 group-wise 评估。group 只用于诊断和 leaveout，不用于 dataset-specific controller。

Group axes：

```text
dataset
seed
event_family
signal_stratum
step_bucket
score_bucket
payload_norm_bucket
memory_fail_bucket
longrisk_bucket
old_family_bucket
source_action_family
action_origin = AP0 / APY / APG / APGS / APGR / APGM / APGA
```

### 必须记录

```text
ranker_id
group_axis
group_id
group_action_count
group_GradeAB_count
group_MemorySafeGradeB_count
group_T5_count
group_positive_density
TopK_global_overlap
TopK_group_count
TopK_group_share
TopK_group_GradeAB_precision
TopK_group_V_integrated_LCB
TopK_group_h240_longrisk_UCB
TopK_group_bad_UCB
TopK_group_null_UCB
rank_score_mean
rank_score_std
score_shift_vs_global
feature_PSI_vs_global
feature_KL_vs_global
LDO_drop
LSO_drop
miss_reason
```

### 主要计算

TopK group concentration：

$$
Share_g=\frac{|TopK\cap g|}{|TopK|}.
$$

Group drop：

$$
Drop_g=Precision_{pooled}-Precision_{leaveout(g)}.
$$

Group score shift：

$$
Shift_g=\frac{|\mu_g-\mu_{global}|}{\sigma_{global}+\epsilon}.
$$

### 判断标准

P1 pass：

```text
identified_drop_group_fraction >= 0.80
at least one dominant_drop_axis found
failure_mode assigned for >= 90% drop cases
```

Failure modes：

```text
GDF1-positive-density-missing-in-heldout-group
GDF2-score-calibration-shift
GDF3-feature-distribution-shift
GDF4-topk-group-concentration
GDF5-memory-risk-shift
GDF6-action-origin-shift
GDF7-label-definition-instability
GDF8-small-sample-LCB-collapse
GDF9-dataset-specific-pocket
GDF10-stratum-specific-pocket
```

### 可视化

```text
p1_ldo_lso_drop_heatmap.svg
p1_topk_group_composition.svg
p1_group_score_shift_violin.svg
p1_feature_shift_psi_matrix.svg
p1_drop_failure_modes.svg
```

---

## P2：Group-balanced target density map

### 目标

判断 official target 在 group-balanced 条件下是否有足够 density。若 target 只在少数组里有正例，则任何 controller 都会在 LDO/LSO 崩。

### Target families

```text
GradeA
GradeB
GradeAB
MemorySafeGradeB
T5-like clean geometry
T5-relaxed
ValuePositiveNoLongRisk
ValuePositiveMemorySafe
HorizonSafeValue
GroupStableGradeAB
```

### 必须记录

```text
target_id
global_count
global_coverage
global_V_integrated_LCB
global_h240_longrisk_UCB
global_bad_UCB
global_null_UCB
group_axis
group_min_count
group_min_density
group_median_density
group_density_cv
num_groups_with_zero_positive
positive_group_coverage
support_balance_pass
max_group_share
```

### 判断标准

Weak density pass：

```text
global_coverage >= 0.03
group_min_density >= 0.005
positive_group_coverage >= 0.70
max_group_share <= 0.35
```

Official density pass：

```text
global_coverage >= 0.03
group_min_density >= 0.01
positive_group_coverage >= 0.85
global_V_integrated_LCB > 0
global_h240_longrisk_UCB <= 0.05
global_bad_UCB <= 0.05
global_null_UCB <= 0.15
```

### 可视化

```text
p2_target_density_by_group.svg
p2_target_quality_vs_density.svg
p2_group_positive_coverage_bar.svg
p2_target_support_balance.svg
```

---

## P3：Legal feature invariance and deconfounding

### 目标

把 legal feature 分成“跨 group 有意义的信号”和“只在某些 group 有用的信号”。

### Feature groups

```text
linearized_action_effect
control_transfer_improvement
longrisk_veto_features
memory_fail_features
cover_preservation_features
curvature_fixedpoint_features
population_risk_offdiag_features
old_family_margin_features
hard_tail_margin_features
cost_features
```

### 必须记录

```text
feature_id
feature_group
commit_time_legal
cost_ms_q90
AUC_global_GradeAB
AUC_within_group_mean
AUC_within_group_min
TopK64_global_precision
TopK64_group_balanced_precision
V_LCB_global
V_LCB_group_balanced
longrisk_UCB_global
longrisk_UCB_group_balanced
feature_group_shift_PSI
feature_group_shift_KL
sign_consistency_rate
monotone_sign_pass
```

### 判断标准

Feature invariant pass：

```text
AUC_within_group_mean >= 0.70
AUC_within_group_min >= 0.55
sign_consistency_rate >= 0.80
TopK64_group_balanced_precision >= 0.50
longrisk_UCB_group_balanced <= 0.10
cost_ms_q90 <= 0.50
```

### 可视化

```text
p3_feature_auc_global_vs_within_group.svg
p3_feature_sign_consistency.svg
p3_feature_cost_quality_frontier.svg
p3_feature_shift_vs_drop.svg
```

---

## P4：Group-invariant legal ranker

### 目标

训练或构造不使用结果字段、不过度依赖 group pocket 的 legal ranker。P4 是本轮第二科学核心。

### Rankers

```text
GIR1-GroupQuantileNormalizedValueRiskRank
GIR2-MinGroupLCBRank
GIR3-ValueRankLongRiskVetoGroupBalanced
GIR4-MemoryVetoGroupBalanced
GIR5-CoverMemoryValueRank
GIR6-DistributionallyRobustLinearRank
GIR7-GroupAdversarialSmallMLPDiagnostic
GIR8-LegalDistilledStudentNoRedNoGroupName
GIR9-PairwiseWithinGroupRanker
GIR10-GroupConformalRiskRank
```

### 禁止项

```text
不能输入 dataset_name；
不能输入 outcome-derived V_integrated / GradeAB / risk_score；
不能用 heldout group threshold；
不能把 group id 当 controller branch；
```

Group 只能用于训练时的 balancing / audit，不能用于 dataset-specific dispatch。

### 必须记录

```text
ranker_id
input_feature_count
red_field_count
group_name_used_as_feature
calibration_split_id
heldout_split_id
TopK64_precision
TopK87_precision
TopK87_V_integrated_LCB
TopK87_h240_longrisk_UCB
TopK87_bad_UCB
TopK87_null_UCB
TopK87_memory_fail_UCB
coverage_at_87
max_group_share
rank_entropy
LDO_drop_max
LSO_drop_max
within_group_precision_min
within_group_V_LCB_min
within_group_longrisk_UCB_max
```

### 判断标准

P4 weak pass：

```text
red_field_count = 0
group_name_used_as_feature = 0
TopK87_precision >= 0.65
TopK87_V_integrated_LCB > 0
TopK87_h240_longrisk_UCB <= 0.05
TopK87_bad_UCB <= 0.05
TopK87_null_UCB <= 0.15
LDO_drop_max <= 0.20
LSO_drop_max <= 0.20
max_group_share <= 0.35
```

P4 strong pass：

```text
TopK87_precision >= 0.75
TopK87_V_integrated_LCB >= 0.05
TopK87_h240_longrisk_UCB <= 0.03
LDO_drop_max <= 0.15
LSO_drop_max <= 0.15
```

### 可视化

```text
p4_ranker_pareto_precision_risk_drop.svg
p4_group_balanced_topk_composition.svg
p4_rank_score_calibration_by_group.svg
p4_ranker_leaveout_matrix.svg
```

---

## P5：Rank-safe certificate v7

### 目标

把 ranker 从 diagnostic 排名变成可部署 accepted-region 规则。重点不是 ECE 好看，而是 frozen heldout accepted region 过 value / risk / support gate。

### Candidate certificates

```text
RC7-TopKFixedCountGroupBalanced
RC7-ValueLCBLongRiskUCBConformal
RC7-GroupMinRiskBound
RC7-MemoryRiskVetoCertificate
RC7-CoverMemoryValueCertificate
RC7-CostConstrainedRankCertificate
RC7-IntersectionOfValueAndRiskRanks
```

### 必须记录

```text
certificate_id
ranker_id
accepted_count_cal
accepted_count_heldout
coverage_heldout
GradeAB_precision_heldout
V_integrated_LCB_heldout
h240_longrisk_UCB_heldout
bad_UCB_heldout
null_UCB_heldout
memory_fail_UCB_heldout
max_group_share_heldout
LDO_drop_max
LSO_drop_max
ECE
Brier
rank_calibration_error
cost_ms_q90
```

### 判断标准

P5 pass：

```text
accepted_count_heldout >= 87
coverage_heldout >= 0.03
GradeAB_precision_heldout >= 0.70
V_integrated_LCB_heldout > 0
h240_longrisk_UCB_heldout <= 0.05
bad_UCB_heldout <= 0.05
null_UCB_heldout <= 0.15
memory_fail_UCB_heldout <= 0.10
LDO_drop_max <= 0.20
LSO_drop_max <= 0.20
cost_ms_q90 <= 0.50
```

### 可视化

```text
p5_certificate_accepted_region_dashboard.svg
p5_certificate_heldout_vs_calibration.svg
p5_certificate_group_stability.svg
p5_certificate_cost_quality.svg
```

---

## P6：Existing-action minimal geometry controller

### 目标

优先检验 existing AP0/action universe 是否已经足够构成 functional controller。只有 existing-action controller 不可行，才把主资源转到 generated-action primitive。

### Controller form

Controller 不使用 dataset name，不使用 future outcome，不使用 result labels。

$$
Accept(a)=1
$$

当且仅当：

$$
RankScore(a) \ge q_{cal}
$$

$$
ValueLCB_{cal}(a)>0
$$

$$
LongRiskUCB_{cal}(a)\le \tau_L
$$

$$
MemoryRiskUCB_{cal}(a)\le \tau_M
$$

$$
Cost(a)\le C_{max}.
$$

### 必须记录

```text
controller_id
ranker_id
certificate_id
accepted_count
coverage
GradeAB_precision
V_integrated_LCB
h240_longrisk_UCB
bad_UCB
null_UCB
memory_fail_UCB
support_balance
max_group_share
max_family_share
LDO_pass
LSO_pass
feature_cost_ms_q90
payload_apply_ms_q90_estimate
```

### 判断标准

P6 pass：

```text
accepted_count >= 87
coverage >= 0.03
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
LDO_pass = 1
LSO_pass = 1
support_balance = 1
```

### 可视化

```text
p6_controller_quality_dashboard.svg
p6_controller_group_support.svg
p6_controller_value_risk_tradeoff.svg
```

---

## P7：APGA OOD / damage autopsy

### 目标

解释 APGA 为什么生成失败。P7 不生成新 primitive，只做原因拆解。

### 必须记录

```text
primitive_id
source_action_id
generated_action_id
source_rank_score
generated_rank_score
source_grade
generated_grade
source_V_integrated
generated_V_integrated
source_longrisk
generated_longrisk
payload_norm_shift
payload_linf_shift
payload_cosine_to_source
payload_cosine_to_adamw
cover_delta_shift
memory_delta_shift
basis_rank_delta
hardtail_entropy_delta
feature_distribution_PSI_to_GradeAB
feature_distribution_KL_to_GradeAB
new_positive_created
positive_lost
longrisk_created
Damage_V_integrated
Damage_memory
Damage_cover
```

### 判断标准

P7 pass：

```text
APGA_failure_attribution_fraction >= 0.80
at least one dominant damage mode identified
```

Damage modes：

```text
D1-source-good-destroyed
D2-generated-OOD-payload
D3-memory-guard-too-weak
D4-cover-collapse
D5-value-direction-wrong
D6-adamw-conflict
D7-longrisk-veto-ineffective
D8-certificate-misaligned
D9-source-target-mismatch
```

### 可视化

```text
p7_apga_source_to_generated_waterfall.svg
p7_payload_distribution_shift.svg
p7_geometry_damage_heatmap.svg
p7_longrisk_created_by_primitive.svg
p7_source_generated_rank_score_scatter.svg
```

---

## P8：Memory / cover blocker anatomy v3

### 目标

重新拆解 longrisk 的来源。v9.5.8 memory explained fraction = 0.6981，接近阈值但不够。P8 将 memory 拆成更细维度。

### Memory subcomponents

```text
old_family_margin_fail
old_stratum_margin_fail
hard_tail_memory_fail
cover_entropy_collapse
basis_rank_collapse
signal_reservoir_leak
population_risk_offdiag_fail
```

### 必须记录

```text
subcomponent_id
high_longrisk_count
subfail_count
longrisk_and_subfail_count
P_longrisk_given_subfail
P_subfail_given_longrisk
V_integrated_LCB_for_subfail
GradeAB_precision_for_subfail
memory_safe_value_positive_count
memory_safe_value_positive_density
```

### 判断标准

P8 pass：

```text
one_or_more_subcomponent_P_longrisk_given_subfail >= 0.80
one_or_more_subcomponent_P_subfail_given_longrisk >= 0.65
memory_safe_value_positive_density >= 0.03
```

若 density < 0.03，但 count near threshold，则记为 diagnostic only。

### 可视化

```text
p8_memory_subcomponent_explanation.svg
p8_memory_vs_longrisk_sankey.svg
p8_memory_safe_value_density.svg
p8_cover_memory_phase_plot.svg
```

---

## P9：APGI / APGC generator reset only if needed

### 目标

只有 P6 existing-action controller 不过，且 P7/P8 指出 generated path 可修，才运行 P9。P9 的 generator 不再只是合法 payload builder，而必须直接约束 memory、cover、rank score 和 longrisk。

### Primitive family

```text
APGI1-GroupInvariantRankProjectedUpdate
APGI2-MemorySubspaceNullProjection
APGI3-CoverEntropyPreservingUpdate
APGI4-BasisRankPreservingUpdate
APGI5-PopRiskOffDiagSignalGate
APGI6-AdamWCompatibleValueRiskIntersection
APGI7-SymmetricBoundaryDampedMemoryUpdate
APGI8-NegativeControlShuffledPayload
```

### 生成约束

每个 generated action 必须在 commit time 写出：

```text
rank_score_before
rank_score_after_pred
memory_subrisk_pred
cover_collapse_pred
basis_rank_delta_pred
adamw_conflict_pred
payload_cost_pred
certificate_hash
payload_hash
```

### 必须记录

```text
primitive_id
generated_action_count
payload_hash_missing
certificate_hash_missing
action_apply_linf_max
preflight_pass
negative_control_divergence
```

### P9 仅做 implementation/preflight 判断

P9 pass：

```text
generated_action_count = 512
payload_hash_missing = 0
certificate_hash_missing = 0
action_apply_linf_max <= 1e-7
negative_control_divergence = 1
```

---

## P10：APGI/APGC branch-horizon outcome

### 目标

如果 P9 运行，则测 generated actions 是否真正形成 new frontier。

### 必须记录

```text
primitive_id
expected_rows
actual_rows
branch_completion_rate
horizon_completion_rate
quality_audit_pass
GradeAB_precision
MemorySafeGradeB_precision
V_integrated_LCB
h240_longrisk_UCB
bad_UCB
null_UCB
memory_fail_UCB
new_positive_created_rate
longrisk_created_rate
source_to_generated_damage_LCB
```

### 判断标准

P10 weak pass：

```text
GradeAB_precision >= 0.20
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.10
new_positive_created_rate >= 0.10
longrisk_created_rate <= 0.10
```

P10 official candidate pass：

```text
GradeAB_precision >= 0.70
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
```

### 可视化

```text
p10_apgi_outcome_by_primitive.svg
p10_generated_frontier_value_risk.svg
p10_generated_damage_waterfall.svg
```

---

## P11：Selected controller runtime preflight

### 目标

只有 P6 或 P10 产生 selected controller 后运行。测在线训练时实际成本，不把 offline materializer 混进去。

### 必须记录

```text
controller_id
ranker_id
certificate_id
runtime_mode
online_steps
active_steps
zero_candidate_steps
feature_compute_ms_q90
rank_compute_ms_q90
certificate_compute_ms_q90
payload_apply_ms_q90
controller_launches_per_active_step_q90
step_ratio_q50
step_ratio_q90
step_ratio_q99
memory_ratio
materializer_in_timed_path
old_step_ratio_reused
```

### 判断标准

P11 pass：

```text
materializer_in_timed_path = 0
old_step_ratio_reused = 0
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
payload_apply_ms_q90 <= base_step_ms_q90 * 0.25
```

### 可视化

```text
p11_runtime_breakdown.svg
p11_step_ratio_distribution.svg
p11_active_step_launches.svg
```

---

## P12：System legal controller boundary

### 目标

把 P6/P10/P11 的结果合并，判断是否进入 official system candidate。

### 必须记录

```text
system_candidate_id
controller_id
ranker_id
certificate_id
primitive_id
source = existing_action / generated_action
accepted_count
coverage
precision_GradeAB
V_integrated_LCB
h240_longrisk_UCB
bad_UCB
null_UCB
support_balance
LDO_pass
LSO_pass
step_ratio_q90
memory_ratio
contract_audit_pass
no_fake_no_proxy
```

### 判断标准

P12 pass：

```text
accepted_count >= 87
coverage >= 0.03
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
LDO_pass = 1
LSO_pass = 1
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
contract_audit_pass = 1
no_fake_no_proxy = 1
```

---

## P13：Leave-dataset-out / leave-stratum-out official

### 目标

只有 P12 pass 后运行。证明不是 dataset-specific route，不在数据集上调榜。

### 设置

```text
LDO:
  train/calibrate on MNIST + Fashion, evaluate KMNIST
  train/calibrate on MNIST + KMNIST, evaluate Fashion
  train/calibrate on Fashion + KMNIST, evaluate MNIST

LSO:
  leave one signal stratum / family / step bucket out
```

### 必须记录

```text
split_type
heldout_group
controller_id
accepted_count
coverage
GradeAB_precision
V_integrated_LCB
h240_longrisk_UCB
bad_UCB
null_UCB
CEp99_delta
margin_delta
ECE_delta
NLL_delta
beats_adamwparallel
beats_bestlr
beats_noop
beats_random
shuffle_control_pass
```

### 判断标准

```text
LDO: all heldout datasets task-safe, at least 2/3 positive V LCB
LSO: >= 70% heldout strata pass value/risk gate
shuffle_control_pass = 1
```

### 可视化

```text
p13_ldo_matrix.svg
p13_lso_matrix.svg
p13_shuffle_control.svg
```

---

## P14：Official paired replay

### 目标

只有 P12/P13 pass 后运行。验证 RealFunctional 是否真的比强对照好。

### Branches

```text
RealFunctional
AdamWOnly
AdamWParallel
BestLR
NoOp
RandomPayload
ShuffledFunctionalPayload
```

### Horizons

```text
20, 80, 240, optional 480
```

### 必须记录

```text
branch
horizon
CE_delta
CEp99_delta
margin_p10_delta
NLL_delta
ECE_delta
acc_delta
hard_tail_acc_delta
V_ctrl
beats_adamwparallel
beats_bestlr
beats_noop
beats_random
beats_shuffled
runtime_step_ratio
```

### 判断标准

```text
RealFunctional beats AdamWParallel rate >= 0.50
RealFunctional beats BestLR rate >= 0.50
RealFunctional beats NoOp rate >= 0.60
ShuffledFunctionalPayload fails advantage
h240 longrisk remains <= 0.05
```

---

## P15：Short/full training boundary

### 目标

只有 P14 pass 后运行。不是打榜，而是确认 functional update 在真实训练中不会只局部有效。

### 必须记录

```text
dataset
seed
model
controller_id
final_train_acc
final_val_acc
final_test_acc
best_val_acc
time_to_target
steps_to_target
CEp99
NLL
ECE
hard_tail_acc
continual_retained_acc
forgetting_rate
step_ratio_q90
memory_ratio
```

### Baselines

```text
LQ-t2-h256 AdamW
MatchedMLP
AdamWStrongLRGridMLP
QuadraticFeatureMLP
NoFunctionalController
ShuffledFunctionalController
```

### 判断标准

```text
no catastrophic fail
RealFunctional >= LQ AdamW on macro metrics
RealFunctional not worse than AdamWStrongLRGridMLP by more than 0.005 unless geometry metrics improve strongly
sample efficiency or calibration/robustness improvement present
```

---

# 6. 并行执行安排

Batch 1：快速诊断，必须并行。

```text
P0 boundary reproduction
P1 group drop autopsy
P2 group-balanced target density map
P3 feature invariance / deconfounding
P7 APGA OOD / damage autopsy
P8 memory / cover blocker anatomy
Base-Acc Sentinel continuation
```

Batch 2：rank/controller 路线。

```text
P4 group-invariant legal ranker
P5 rank-safe certificate
P6 existing-action minimal geometry controller
```

Batch 3：generated-action 路线，仅在 P6 不过或需要补 density 时运行。

```text
P9 APGI/APGC implementation
P10 APGI/APGC branch-horizon outcome
```

Batch 4：系统路线，仅在 P6 或 P10 pass 后运行。

```text
P11 selected runtime
P12 system legal controller
P13 leaveout
P14 paired replay
P15 short/full training
```

这样安排的原因是：v9.5.8 已经说明 existing-action rank 有强信号，所以不能把资源全花在新 generator 上；同时 APGA 失败必须 autopsy，不能盲目造新 primitive。

---

# 7. Artifact 清单

```text
p0_v9580_boundary_reproduction.csv
p1_group_drop_autopsy.csv
p1_group_drop_failure_modes.csv
p2_group_balanced_target_density.csv
p3_feature_invariance_deconfounding.csv
p4_group_invariant_rankers.csv
p5_rank_safe_certificate_v7.csv
p6_existing_action_controller.csv
p7_apga_ood_damage_autopsy.csv
p8_memory_cover_blocker_v3.csv
p9_apgi_apgc_implementation.csv
p10_apgi_apgc_branch_horizon_outcome.csv
p11_selected_runtime_preflight.csv
p12_system_legal_controller_boundary.csv
p13_leaveout_official.csv
p14_official_paired_replay.csv
p15_short_full_training_boundary.csv
base_acc_sentinel_v9590.csv
contract_audit_v9590.csv
no_fake_audit_v9590.csv
route_decision_v9590.json
failure_taxonomy_v9590.csv
run_manifest_v9590.json
```

---

# 8. Route decision

```text
R0-BoundaryRegression
  if P0 fails

R1-GroupSpecificRankPocket
  if P1 shows TopK concentrated in groups and P4 cannot reduce LDO/LSO drop

R2-GroupBalancedTargetDensityInsufficient
  if P2 shows no target with enough group-balanced support

R3-GroupInvariantLegalRankPassExistingActionControllerPending
  if P4 passes but P5/P6 not yet run

R4-ExistingActionControllerPassRuntimePending
  if P6 passes but P11 not run

R5-SystemLegalControllerPassPairedReplayPending
  if P12 passes but P14 not run

R6-APGAGeneratorOODDestructive
  if P7 attributes APGA failure to OOD/damage

R7-MemoryCoverBlockerConfirmedGeneratorResetRequired
  if P8 confirms memory/cover dominates longrisk and existing action controller fails

R8-GeneratedPrimitiveStillFails
  if P9/P10 implementation passes but generated outcome fails

R9-FunctionalLocalCausalPassShortFullPending
  if P14 paired replay passes

R10-ExternalReadyCandidate
  if P15 short/full + leaveout + paired replay + runtime all pass
```

---

# 9. 最重要的停止条件

立即停止小修路线的条件：

```text
1. P4 LDO/LSO drop 仍 > 0.50：停止调 rank threshold，转机制解释。
2. P5 heldout V LCB 仍为负：停止调 certificate ECE，转 accepted-region 分解。
3. P7 发现 APGA OOD/damage：停止 APGA 小变体，转 generator reset。
4. P2 group-balanced target density < 0.03 且 generator 失败：回到 target / geometry definition，而不是加 controller。
5. 任何阶段出现 red field / outcome-derived field：该结果只能 diagnostic，不能 official。
```

---

# 10. 本轮预期结论形式

v9.5.9 不强求成功，但必须明确回答：

```text
1. R2/R1/R5 的强 TopK 是否能变成 group-stable legal rank？
2. 如果不能，是 target density 问题、feature shift 问题、还是 group pocket 问题？
3. existing AP0 actions 是否足够形成 system controller？
4. APGA 是否因 OOD / memory-cover damage 而失败？
5. 下一步应该主攻 existing-action controller，还是 generated-action primitive reset？
```

最低可接受结论：

$$
\boxed{\text{明确排除一种主线，并保留一种可验证主线。}}
$$

最理想结论：

$$
\boxed{\text{existing-action group-invariant legal controller 过 P6/P11/P12，进入 paired replay。}}
$$

