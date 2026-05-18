# DG-KAN v9.7.6 Core77 LDO / Existing-Action Density 与机制验证 / 最小 Controller 实验计划

> 本计划基于 v9.7.5 `ExistingAction LDO / Core Mechanism / Horizon Transfer Reformulation` 的真实结果制定。  
> v9.7.6 不继续小修 `ExactTransfer`、`WindowedTransfer`、`OldRank threshold`、`Expansion10` 或 APG/APGU 生成器变体。  
> 本轮核心目标是把 v9.7.5 暴露出的真正问题拆清楚：**已有高质量动作，但它们跨数据集支持不均；我们还不知道这是因为动作数量太少、选择规则不稳定，还是评价方式本身太容易被 support shift 触发。**

---

## 0. 一句话目标

v9.7.6 要回答的问题是：

$$
\boxed{
\text{Core77 / OldOnly 这批高质量动作，能否变成跨数据集、跨模板、训练当下可部署的最小 functional update 选择规则？}
}
$$

更直白地说，本轮不再问：

```text
ExactTransfer 阈值再调一下行不行？
WT80 再换一个窗口行不行？
Expansion10 换十个动作行不行？
APGU/APGV 再造一批动作行不行？
```

而是问：

```text
1. Core77 为什么每个数据集内 precision 都是 1.0，却 LDO 仍然很高？
2. 这个 LDO 是真正的跨数据集失败，还是 support / coverage / score scale 共同造成的结构性惩罚？
3. OldOnly 为什么高质量且跨 dataset/template/family，但 ExactTransfer / WT80 解释不了？
4. 旧 rank 的好动作能否被一组更简单、更合法、更稳定的训练当下信号解释？
5. 如果当前 2876 个 action 的高质量区域就是太稀疏，是否应该扩大 AP0 自然候选流，而不是继续手写 APG 生成器？
```

本轮最低有效推进不是 system pass，而是把 route 明确落入下面几种之一：

```text
Case A: LDO 主要是 support / sample-size artifact，可通过更稳健的 dataset-blind support rule 修复。
Case B: LDO 是真实跨数据集不稳定，existing-action route 暂时不能 official。
Case C: OldRank 的高质量区域能被简单合法机制解释，可进入 minimal controller。
Case D: OldRank 仍不可解释，但扩大自然 AP0 candidate 流后 Core density 足够，转向 density scaling。
Case E: 旧 action 选择与自然候选扩展都失败，generated route 只有在新数学目标出现后才重开。
```

---

## 1. v9.7.5 的独立判断

### 1.1 这轮不是没进展

v9.7.5 的 route 是：

```text
route = R2-ExistingActionHighQualityButLDOBlocked
system_legal_controller_pass = 0
generated_route_status = stopped_no_new_objective
```

这不是 functional success，也不是 controller success。但它有几个非常重要的推进。

第一，Core77 的质量被确认很强：

```text
Core77_count = 77
Core77_precision = 1.0
Core77_V_LCB = 0.16402807605399972
Core77_longrisk_UCB = 0.0
Core77_LDO_drop = 0.4415584415584416
```

这说明 Core77 不是垃圾动作。它的问题是 LDO，而不是 value 或 longrisk。

第二，Core77 的 LDO 不是一个简单 hidden pocket：

```text
H1a_dataset_shift_or_support_shift = 1
H1b_hidden_group_concentration = 0
H1c_structural_not_random = 1
max_family_share = 0.1818
max_template_share = 0.0130
max_step_bucket_share = 0.1169
```

这说明它不是单 family、单 template、单 step bucket 的局部口袋。它更像是跨数据集 support / value 分布不均。

第三，OldOnly 仍然非常值得研究：

```text
OldOnly_count = 69
OldOnly_precision = 0.855072463768116
OldOnly_V_LCB = 0.13450092269587752
OldOnly_longrisk_UCB = 0.0
OldOnly_dataset_count = 3
OldOnly_template_count = 69
OldOnly_family_count = 34
OldOnly_single_group_pocket = 0
```

这说明 OldOnly 不是一个单 group 假象。它跨 3 个数据集、69 个模板、34 个 family，仍然是高质量区域。

第四，ExactOnly / WT80 与真实 value 错配：

```text
OldOnly_mean_actual_V = 0.18785132119344358
ExactOnly_mean_actual_V = -0.25884870676985144
OldOnly_mean_WT80 = 0.010067179508415913
ExactOnly_mean_WT80 = 0.10113173453042902
```

这说明 ExactTransfer / WindowedTransfer 现在偏向选择“看起来安全、即时响应高，但真实长期价值低”的动作。它不是一个简单可修的阈值问题。

第五，Expansion10 不是主因：

```text
E0-Core77-only accepted = 77, precision = 1.0, LDO = 0.441558
E1-old-rank-top10 accepted = 87, precision = 0.885057, LDO = 0.390805
E2/E3/E5/E7 full87 LDO 都仍约 0.390805
```

换十个补充动作并不能解决 LDO。Core77 本身已经 LDO fail。

第六，OldRank ablation 没有给出机制：

```text
best_ablation_id = A0-OldRank
best_precision = 0.8735632183908046
best_V_LCB = 0.14111334880346277
best_LDO_drop = 0.37931034482758624
best_OldOnly_overlap = 1.0
mechanism_pass_count = 0
```

也就是说，旧 rank 仍然是强 diagnostic，但不能直接转成公式、controller 或 generator objective。

### 1.2 我不完全接受的表面解读

报告把 primary blocker 写成：

```text
existing_action_ldo_blocked
```

这个是对的。但更深一层应该写成：

$$
\boxed{
\text{已有高质量动作，但这些动作的支持度和价值在不同数据分布下不均；我们还没有找到稳定解释它们的训练当下规则。}
}
$$

这有三层含义：

```text
1. 不是没有好动作。
2. 不是 expansion10 坏。
3. 不是 ExactTransfer 没测准。
4. 真正问题是：好动作为什么好，以及它们能否跨数据分布稳定地被识别。
```

### 1.3 现在最危险的误区

v9.7.5 后最危险的误区有三个。

第一个误区是继续调 ExactTransfer / WT80：

```text
ExactOnly / WT80 的响应值更高，但 actual V 为负。
```

这说明目标错配，不是阈值错。

第二个误区是继续换 expansion10：

```text
所有补 10 的 full87 方案 LDO 都仍高。
```

这说明补充动作不是主因。

第三个误区是继续造 APG/APGU 小变体：

```text
generated_route_status = stopped_no_new_objective
APGU/APGV/APGW/APGX run = 0
direct_solved_sandbox_allowed = 0
```

没有新目标前，generated route 应该继续停止。

---

## 2. 本轮总体实验结构

v9.7.6 采用三条线并行。

### A 线：解释 Core77 的 LDO

目标：判断 Core77 的 LDO 是真实泛化失败，还是 support / sample-size / leaveout 定义导致的结构性惩罚。

### B 线：解释 OldOnly / OldRank 的机制

目标：找出 OldOnly 为什么好，并判断是否能用少量合法训练当下字段复现 OldRank 的高质量区域。

### C 线：判断是否需要扩大自然 AP0 action 流

目标：如果现有 2876 个 action 里干净动作就是太少，就不要继续调 controller，而是扩大自然 AP0 候选流，验证好动作密度是否随 stream 增长而稳定出现。

三条线必须同时跑，不能串行等一个大 runner 结束后才发现下一个 blocker。

---

## 3. v9.7.6 硬约束

继续遵守：

```text
no teacher
no self-teacher
no distillation
no loss modification
no auxiliary loss
no label smoothing
no focal / margin / calibration loss
no sampler / class weight
no CPU offload
no fake rows
no proxy rows
KAN path 不使用 PyTorch loss.backward graph
official controller 不使用 dataset_name 分支
official controller 不使用 validation/test metric
official controller 不使用 future outcome
official controller 不使用 outcome-derived fields
generated route 在没有新 objective evidence 前继续停止
```

允许做：

```text
按 dataset 做诊断，但不能按 dataset 调参；
按 dataset 做 leaveout 分析，但不能把 dataset_id 输入 controller；
用 outcome label 做离线 autopsy，但不能作为 commit-time feature；
用 Core77 / OldOnly / ExactOnly 做对照分析；
扩大 AP0 自然 candidate stream，但不能使用 dataset-specific rule；
用 candidate/action/value/risk 结果训练 diagnostic model，但 official score 只能用 green fields。
```

---

## 4. 成功门槛

任何 controller / accepted region 必须满足：

$$
N_{accept} \ge 87
$$

$$
Precision_{GradeAB} \ge 0.75
$$

$$
LCB(V_{integrated}) > 0
$$

$$
UCB(LongRisk_{240}) \le 0.05
$$

$$
UCB(Bad) \le 0.05
$$

$$
UCB(Null) \le 0.15
$$

$$
UCB(MemoryFail) \le 0.05
$$

$$
UCB(OffdiagFail) \le 0.05
$$

$$
LDO_{drop} \le 0.10
$$

$$
LSO_{drop} \le 0.10
$$

$$
LTO_{drop} \le 0.10
$$

$$
LFO_{drop} \le 0.10
$$

$$
StepRatio_{q90} \le 1.50
$$

其中：

```text
LDO = leave-dataset-out，即留出一个数据集后评估是否仍稳。
LSO = leave-stratum-out，即留出一个难度/状态分层后是否仍稳。
LTO = leave-template-out，即留出一个 action template 后是否仍稳。
LFO = leave-family-out，即留出一个 event/action family 后是否仍稳。
```

v9.7.6 增加一个新门槛：

$$
SupportStabilityPass=1
$$

它要求 accepted region 不是只在某个 split 中因为支持度不足而被 LDO 惩罚。具体定义见 P1。

---

## 5. P0：边界复现与 field legality

### 目标

确认 v9.7.6 没有跳过 v9.7.5 的失败边界，也没有把非法字段带入 official path。

### 要读取的 artifact

```text
v9.7.5 route_decision_v9750.json
p1_core77_ldo_autopsy_v9750.csv
p2_oldonly_mechanism_anatomy_v9750.csv
p3_transfer_mismatch_deep_autopsy_v9750.csv
p4_dataset_blind_stability_repair_v9750.csv
p5_core77_expansion_v2_v9750.csv
p6_oldrank_mechanism_ablation_v2_v9750.csv
```

### 必须记录

```text
source_route_v9750
system_legal_controller_pass_v9750
generated_route_status_v9750
Core77_precision
Core77_LDO_drop
OldOnly_precision
ExactOnly_precision
best_stability_precision
best_stability_LDO_drop
best_ablation_precision
best_ablation_LDO_drop
field_green_count
field_yellow_count
field_red_count
dataset_name_used_in_controller
outcome_derived_field_used_in_controller
fake_row_count
proxy_row_count
cpu_offload_used
```

### 通过标准

```text
P0_pass = 1
field_red_count = 0
dataset_name_used_in_controller = 0
outcome_derived_field_used_in_controller = 0
fake_row_count = 0
proxy_row_count = 0
```

---

## 6. P1：Core77 LDO 定义审计

### 为什么要做

v9.7.5 显示 Core77 在每个 dataset 内 precision 都是 1.0，但 LDO drop 很高。这很反常。我们必须确认 LDO 到底在惩罚什么：

```text
1. precision 下降；
2. value LCB 下降；
3. coverage / support 不足；
4. macro averaging 造成的统计惩罚；
5. leaveout split 中 accepted count 太少；
6. dataset 间 score scale 不一致；
7. 真实跨数据集失败。
```

### 假设

#### H1：Core77 的 LDO 主要是 support-size penalty

成立条件：

```text
per-dataset precision 都高；
per-dataset V LCB 都为正；
但某些 heldout split accepted count 太少；
bootstrap LDO 高主要来自低 support split。
```

#### H2：Core77 的 LDO 是真实 score/rule instability

成立条件：

```text
即使控制 accepted count / support，leaveout precision 或 V 仍显著下降；
同一规则在 heldout dataset 排名位置明显漂移；
score distribution PSI 高且无法用 dataset-blind transform 修复。
```

### 实验设计

对 Core77 执行四种 LDO 计算：

```text
LDO_raw：沿用旧定义。
LDO_precision_only：只看 precision drop。
LDO_value_only：只看 V LCB drop。
LDO_support_adjusted：对 accepted count 做 Wilson / bootstrap 校正。
LDO_equal_count：每个 dataset 采样相同数量 accepted actions 后再算 drop。
```

### 必须记录

```text
Core77_count_total
Core77_count_by_dataset
Core77_precision_by_dataset
Core77_V_LCB_by_dataset
Core77_longrisk_UCB_by_dataset
Core77_bad_UCB_by_dataset
Core77_null_UCB_by_dataset
Core77_memory_UCB_by_dataset
Core77_offdiag_UCB_by_dataset
LDO_raw
LDO_precision_only
LDO_value_only
LDO_support_adjusted
LDO_equal_count
bootstrap_LDO_p50
bootstrap_LDO_p95
bootstrap_prob_LDO_gt_0.10
bootstrap_prob_LDO_gt_0.20
min_dataset_support
max_dataset_support
support_imbalance_ratio
```

### 可视化

```text
fig_p1_core77_dataset_support_bar.svg
fig_p1_core77_dataset_V_lcb_bar.svg
fig_p1_ldo_metric_decomposition_waterfall.svg
fig_p1_bootstrap_ldo_distribution.svg
fig_p1_support_vs_ldo_scatter.svg
```

### 判断标准

P1 支持 “support-size penalty” 的标准：

$$
Precision_{dataset,min} \ge 0.90
$$

$$
LCB(V)_{dataset,min} > 0
$$

$$
LDO_{support\_adjusted} \le 0.10
$$

且：

```text
LDO_raw 高，但 LDO_support_adjusted 低。
```

P1 支持 “真实 LDO instability” 的标准：

$$
LDO_{support\_adjusted} > 0.20
$$

或：

$$
LDO_{equal\_count} > 0.20
$$

如果 P1 判定 LDO 是 support artifact，v9.7.6 可以尝试 support-aware accepted region；如果 P1 判定 LDO 是真实 instability，就必须停止 Core77 直接 controller route，转向机制发现或扩大候选流。

---

## 7. P2：Core77 density 与 action universe 是否太小

### 为什么要做

Core77 只有 77 个动作，距离 87 个最低 accepted count 差 10 个。多轮实验反复证明，随便补 10 个动作不能解决 LDO。现在要问：

```text
当前 2876 action universe 是否太小？
如果自然 AP0 stream 扩大，Core-like 高质量动作会不会按稳定比例出现？
```

这不是生成新 APG primitive，而是扩大自然 AP0 candidate stream。它不引入新手工动作，只验证现有 action 机制是否有足够密度。

### 实验设计

构造 AP0 natural stream extension：

```text
保留同一 base candidate：LQ-t2-h256；
保留同一 manual training contract；
保留同一 candidate lifecycle；
新增 seeds 或更长 train stream；
不按 dataset 调 threshold；
所有新 actions 走同一 canonical outcome materializer；
只对少量 staged sample 先 preflight，避免一次大跑浪费。
```

分三档：

```text
S1: +1 seed / 每 dataset 少量 steps，目标新增约 1000 actions；
S2: +3 seeds，目标新增约 3000 actions；
S3: full extension，目标 action count 至少翻倍。
```

每档先跑 preflight：

```text
100 actions -> 300 actions -> full stage
```

### 必须记录

```text
new_action_count
new_candidate_count
new_event_count
core_like_count
core_like_rate
GradeAB_count
ValuePositiveNoLongRisk_count
MemoryOffdiagCore_count
per_dataset_core_like_count
per_dataset_core_like_rate
per_template_core_like_count
per_family_core_like_count
CoreLike_precision
CoreLike_V_LCB
CoreLike_longrisk_UCB
CoreLike_bad_UCB
CoreLike_null_UCB
CoreLike_LDO_drop
materializer_rows
rows_per_sec
OOM_count
label_violation_count
```

### Core-like 定义

Core-like 动作不是直接复制 Core77 ID，而是满足：

$$
Precision_{GradeAB}=1 \text{ on calibration label}
$$

$$
LCB(V)>0
$$

$$
UCB(LongRisk)=0
$$

$$
UCB(Bad)=0
$$

$$
UCB(Null)=0
$$

$$
UCB(MemoryFail)=0
$$

$$
UCB(OffdiagFail)=0
$$

对于 single action 不能直接算 precision，因此实际用 outcome label：

```text
GradeAB = 1
LongRisk = 0
Bad = 0
Null = 0
MemoryFail = 0
OffdiagFail = 0
V > 0
```

### 可视化

```text
fig_p2_action_universe_growth_curve.svg
fig_p2_core_like_density_by_stage.svg
fig_p2_core_like_density_by_dataset.svg
fig_p2_density_vs_ldo.svg
fig_p2_materializer_throughput.svg
```

### 判断标准

如果扩展后：

$$
CoreLikeCount \ge 174
$$

且：

$$
LDO_{CoreLike} \le 0.10
$$

则说明现有 AP0 route 可能是 density-limited，可以继续 existing-action controller。

如果扩展后 Core-like rate 仍低，或 LDO 仍高：

```text
existing AP0 action universe 不是简单密度问题；必须转向机制解释或新 update rule。
```

---

## 8. P3：OldOnly 机制的 matched contrast

### 为什么要做

OldOnly 是当前最重要的谜题：

```text
OldOnly precision = 0.855
OldOnly V LCB = 0.1345
OldOnly longrisk = 0
OldOnly 跨 dataset/template/family
ExactOnly 几乎全错
```

如果解释不了 OldOnly，OldRank 就只能是 diagnostic，不能变成 controller 或 generator objective。

### 实验设计

为每个 OldOnly action 找 matched negative actions：

```text
匹配 dataset；
匹配 step bucket；
匹配 family；
匹配 template type；
匹配 payload norm bucket；
匹配 action norm bucket；
匹配 memory/offdiag state；
尽量匹配 AdamW cosine。
```

构造对比集合：

```text
OldOnly vs ExactOnly
OldOnly vs Neither
OldOnly vs RandomLegalSafe
OldOnly vs Core77
OldOnly vs Intersection
```

### 比较字段

只比较 commit-time green/yellow diagnostic 字段，不把 outcome-derived 字段当作 candidate feature：

```text
action norm
payload norm
payload linf
AdamW cosine
action/gradient alignment
old family response
memory buffer response
hard-tail response
WT1 / WT5 / WT20 / WT80 response vector
exact transfer mean
exact transfer variance
per-example response sign agreement
support count
template diversity
family diversity
step position
pre-update CE / margin / NLL
pre-update hard-tail fraction
pre-update memory fail proxy
pre-update offdiag proxy
```

### 必须记录

```text
matched_pair_count
matched_success_rate
OldOnly_minus_ExactOnly_feature_delta_mean
OldOnly_minus_Neither_feature_delta_mean
effect_size_by_feature
AUC_by_feature
TopK87_precision_by_feature
TopK87_V_LCB_by_feature
TopK87_LDO_by_feature
feature_stability_by_dataset
feature_stability_by_template
feature_stability_by_family
```

### 可视化

```text
fig_p3_oldonly_matched_contrast_forest.svg
fig_p3_feature_effect_size_heatmap.svg
fig_p3_oldonly_vs_exactonly_response_vector.svg
fig_p3_per_example_response_sign_agreement.svg
fig_p3_oldonly_mechanism_sankey.svg
```

### 判断标准

P3 机制 pass 条件：

```text
至少 2 个 non-outcome legal features 在 matched contrast 中 effect size >= 0.5；
这些 features 的 TopK87 precision >= 0.75；
V LCB > 0；
longrisk UCB <= 0.05；
LDO drop <= 0.15；
并且不使用 dataset_name、future outcome 或 GradeAB label。
```

如果 P3 找不到任何机制，但 OldOnly 质量继续强，则结论为：

```text
OldOnly 是真实 high-quality diagnostic，但当前 landed feature set 仍无法解释。
```

---

## 9. P4：OldRank score 机制替代实验

### 目标

把 OldRank 拆成更少、更清楚的组件，避免继续把 `OldRank` 当黑箱。

### 方法

构建一组极简 rankers：

```text
R0-OldRank：原始参考。
R1-ValueProxyOnly：只用 value-like commit-time proxy。
R2-RiskVetoOnly：只用 longrisk/memory/offdiag veto。
R3-ValueProxy + HardRiskVeto。
R4-ValueProxy + SupportPenalty。
R5-ResponseVectorShapeRank：只用 response vector shape，不用 outcome。
R6-OldFamilyStableRank：只用 old-family/memory response。
R7-PerExampleAgreementRank：只用 response sign agreement。
R8-MinimalTwoFeatureRank：P3 找到的两个最稳 feature。
R9-MinimalThreeFeatureRank：P3 找到的三个最稳 feature。
```

### 必须记录

```text
ranker_id
feature_count
red_field_count
accepted_count
precision
V_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
LDO_drop
LSO_drop
LTO_drop
LFO_drop
Core77_overlap
OldOnly_overlap
ExactOnly_overlap
feature_cost_q90
```

### 可视化

```text
fig_p4_ranker_ablation_frontier.svg
fig_p4_precision_vs_ldo_tradeoff.svg
fig_p4_oldonly_overlap_by_ranker.svg
fig_p4_feature_count_vs_quality.svg
```

### 判断标准

Minimal mechanism pass：

$$
FeatureCount \le 3
$$

$$
N_{accept} \ge 87
$$

$$
Precision \ge 0.75
$$

$$
LCB(V)>0
$$

$$
UCB(LongRisk) \le 0.05
$$

$$
LDO_{drop} \le 0.10
$$

如果只有 R0-OldRank 过 quality 但 LDO 高，而 minimal features 全失败，则 OldRank 不能转 controller。

---

## 10. P5：Dataset-blind support-aware accepted region

### 为什么要做

如果 P1 证明 LDO 有很大 support / sample-size 成分，就可以尝试 dataset-blind support-aware rule。这里不能按 dataset 调参，只能用通用支持度原则。

### 方法

构造 accepted rule：

```text
先选 Core-like high-quality actions；
要求 accepted set 在 generic split 上有足够 support；
generic split 不能是 dataset_id，而是 action family、template、step bucket、memory/offdiag state、score quantile；
如果某个 split support 不足，就不强行补该 dataset，而是降低整体 accept count 或扩大 natural stream。
```

规则候选：

```text
S1-Core77Raw
S2-Core77SupportAdjusted
S3-Core77EqualCountBootstrap
S4-Core77PlusOldOnlySupportBalanced
S5-OldRankWithSupportFloor
S6-MinimalFeatureRankWithSupportFloor
S7-DensityScaledCoreLikeNaturalStream
```

### 必须记录

```text
accepted_count
coverage
precision
V_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
LDO_raw
LDO_support_adjusted
LDO_equal_count
LSO_drop
LTO_drop
LFO_drop
min_generic_group_support
max_generic_group_share
score_quantile_support
feature_cost_q90
```

### 判断标准

Support-aware weak pass：

```text
accepted_count >= 87
precision >= 0.75
V_LCB > 0
longrisk/bad/memory/offdiag UCB <= 0.05
null UCB <= 0.15
LDO_support_adjusted <= 0.10
LSO/LTO/LFO <= 0.10
```

如果 `LDO_raw` 高但 `LDO_support_adjusted` 低，本轮必须明确记录：

```text
official 是否允许 support-adjusted LDO？
```

如果不允许，仍不能 system pass，但可以作为 route 变更依据。

---

## 11. P6：Core density 扩展后的 controller 候选

### 前置条件

P2 至少 S1 或 S2 完成，并证明 natural stream 扩展后 Core-like density 增加。

### 实验设计

在 expanded AP0 natural stream 上重做：

```text
Core-like target
OldOnly-like target
MinimalFeatureRank
OldRank diagnostic
SupportAwareRank
```

### 必须记录

```text
expanded_action_count
expanded_core_like_count
expanded_core_like_rate
expanded_rank_precision
expanded_rank_V_LCB
expanded_rank_longrisk_UCB
expanded_rank_LDO
expanded_rank_LSO
expanded_rank_LTO
expanded_rank_LFO
comparison_to_original_2876
```

### 判断标准

如果 expanded stream 上：

$$
N_{accept} \ge 174
$$

并且所有 gate 过，则说明 previous failure 是 action density / sample-size 问题。

如果 expanded stream 仍只有少量 Core-like，或仍 LDO 高，则 existing-action natural route 不够。

---

## 12. P7：Controller boundary

### 目标

只有 P4/P5/P6 至少一个 weak pass 后，才打开 minimal controller。

### Controller 形式

尽量简单：

```text
Accept(a) = RankScore(a) >= threshold
            and RiskVeto(a) = 0
            and MemoryVeto(a) = 0
            and OffdiagVeto(a) = 0
            and Cost(a) <= Cmax
```

不允许：

```text
dataset-specific threshold
dataset branch
future outcome
GradeAB at commit
V_integrated at commit
old table outcome
```

### 必须记录

```text
controller_id
feature_list
red_field_count
yellow_field_count
threshold_source
calibration_split
heldout_split
accepted_count
coverage
precision
V_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
LDO_drop
LSO_drop
LTO_drop
LFO_drop
feature_cost_q90
```

### 通过标准

所有门槛见第 4 节。

如果 P7 不过，P8 runtime 不打开。

---

## 13. P8：Selected runtime boundary

### 前置条件

```text
P7_controller_pass = 1
```

### 要记录

```text
selected_controller_id
accepted_steps
accepted_actions
feature_compute_ms_q90
rank_score_ms_q90
veto_ms_q90
payload_apply_ms_q90
kernel_launch_count_q90
step_ratio_q50
step_ratio_q90
step_ratio_q99
memory_ratio_q90
empty_step_kernel_count
active_step_launches_q90
```

### 通过标准

$$
StepRatio_{q90} \le 1.50
$$

$$
MemoryRatio_{q90} \le 1.05
$$

且：

```text
no CPU offload
no fake runtime
no diagnostic-derived runtime
```

---

## 14. P9：Generated route stop / reopen decision

### 当前判断

v9.7.5 后 generated route 继续停止：

```text
new_objective_evidence = 0
APGU/APGV/APGW/APGX run = 0
direct_solved_sandbox_allowed = 0
```

v9.7.6 不允许直接新增 APGY/APGZ。

### 允许重开的条件

只有以下任一条件成立，generated route 才能重开：

```text
1. P3/P4 找到明确、简单、合法的 OldOnly mechanism；
2. P2/P6 证明 natural stream density 不足但 mechanism 明确；
3. P5/P7 证明 existing-action controller 仍失败，但 failure 可归因为 candidate density，不是 mechanism absence；
4. 出现新的直接优化目标，不是 APG 小变体。
```

### 重开后生成器必须满足

```text
不再是 residual blend / projection 小变体；
必须直接优化 P3/P4 找到的机制；
必须先 1-action / 8-action / 64-action preflight；
必须比较 source-to-generated damage；
必须有 negative control。
```

---

## 15. P10：Paired replay boundary

### 前置条件

```text
P7_controller_pass = 1
P8_selected_runtime_pass = 1
```

### 分支

```text
RealFunctional
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledFunctionalPayload
```

### 必须记录

```text
branch
horizon
dataset
seed
template
accepted_count
V_integrated
CEp99_delta
margin_p10_delta
NLL_delta
ECE_delta
hard_tail_delta
memory_delta
longrisk
bad
null
runtime
```

### 通过标准

```text
RealFunctional beats AdamWParallel / bestLR / NoOp / Random；
ShuffledFunctionalPayload 不通过；
LDO / LSO / LTO / LFO 稳定；
runtime gate 保持通过。
```

---

## 16. P11：Short / full boundary

### 前置条件

```text
P10_paired_replay_pass = 1
```

### 任务

继续只做固定配置评估，不按数据集调参：

```text
MNIST
Fashion-MNIST
KMNIST
seeds 0..9
MatchedMLP
AdamWStrongLRGridMLP
LQ base
LQ + selected functional update
```

### 指标

```text
train acc
val acc
test acc
CE
NLL
ECE
CEp99
margin_p10
time_to_target
steps_to_target
sample_efficiency
hard-stratum acc
forgetting / old-family retained acc
runtime step ratio
peak memory
```

### 通过标准

不能只看 acc。必须：

```text
functional update 在至少一个核心维度有稳定优势；
不能显著降低 calibration / robustness / forgetting；
不能靠 dataset-specific tuning；
不能比 MLP runtime envelope 过慢。
```

---

## 17. 可视化清单

必须生成：

```text
fig_p1_ldo_definition_decomposition.svg
fig_p1_core77_support_vs_ldo.svg
fig_p1_core77_bootstrap_ldo.svg
fig_p2_core_density_growth.svg
fig_p2_core_density_by_dataset.svg
fig_p3_oldonly_matched_contrast_forest.svg
fig_p3_response_vector_oldonly_exactonly.svg
fig_p4_ranker_ablation_frontier.svg
fig_p4_oldonly_overlap_heatmap.svg
fig_p5_support_aware_frontier.svg
fig_p6_expanded_stream_density_comparison.svg
fig_p7_controller_gate_waterfall.svg
fig_p8_runtime_breakdown.svg
fig_p9_generated_route_decision_tree.svg
fig_p10_paired_replay_branch_comparison.svg
fig_p11_short_full_summary.svg
```

---

## 18. Stop / pivot 条件

### Stop 1：Core77 LDO 是真实 instability

如果：

```text
LDO_support_adjusted > 0.20
LDO_equal_count > 0.20
```

则不能继续 Core77 direct controller。

### Stop 2：OldOnly 仍无机制

如果 P3/P4 找不到任何简单合法机制：

```text
mechanism_pass_count = 0
```

则 OldRank 只能 diagnostic，不得 official。

### Stop 3：natural stream density 不增长

如果 expanded AP0 stream 后：

```text
CoreLikeRate 不增加；
CoreLike LDO 仍高；
```

则 existing-action route 应进入 terminal boundary。

### Pivot 1：support-adjusted LDO 低但 raw LDO 高

需要决定 official gate 是否允许支持度校正。若不允许，不能 pass，但路线可以转向扩大 action universe。

### Pivot 2：natural stream expansion 成功

如果好动作密度随 action universe 增长稳定出现，则 v9.7.7 应转向 high-throughput AP0 candidate harvesting + minimal controller。

### Pivot 3：OldOnly mechanism 被解释

如果找到简单机制，则 v9.7.7 应转向 mechanism-based controller + selected runtime。

---

## 19. 最终期望输出

v9.7.6 结束时必须给出下列结论之一：

```text
R1-Core77LDOSupportArtifact_ControllerCandidate
R2-Core77TrueLDOFail_ExistingActionRouteBlocked
R3-OldOnlyMechanismFound_MinimalControllerCandidate
R4-OldOnlyMechanismAbsent_OldRankDiagnosticOnly
R5-NaturalAP0DensityScalingWorks
R6-NaturalAP0DensityScalingFails
R7-SystemPass_OpenPairedReplay
```

最理想是：

```text
R7-SystemPass_OpenPairedReplay
```

但可接受科学进展是：

```text
R2 / R4 / R6
```

只要它们能明确告诉我们：继续 existing-action route 是否还有意义。

---

## 20. 一句话总结

v9.7.6 的核心不是再调一个分数，而是回答：

$$
\boxed{
\text{当前高质量 existing actions 到底是“可扩展的训练规律”，还是“当前 action universe 里的有限诊断现象”？}
}
$$

如果是前者，我们继续走 existing-action minimal controller。  
如果是后者，就必须停止围绕 OldRank / Core77 小修，转向更根本的 action 生成原则或更大规模的自然 AP0 action universe。
