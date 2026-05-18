# DG-KAN v9.6.8 Dataset-Invariant Core-Expansion Controller / Score Transport / Generator Route Decision 完整实验计划

> 本计划基于 v9.6.7 `Dataset-Shift Deconfounded Rank / Core-Expansion Controller / Generated Route Stop` 的真实结果制定。  
> v9.6.8 不继续小修 `R7A`、`CERT15C`、`T2.3`、`T2.6` 或 APGU。  
> 本轮要回答一个更根本的问题：
>
> $$
> \boxed{\text{能否在不按数据集调参的前提下，把现有好动作区域扩展成跨数据集稳定、足够干净、数量足够的 accepted region？}}
> $$
>
> 如果答案仍是否定的，本轮必须给出明确停机结论：existing-action controller route 暂停，generated-action route 只能在重新定义生成目标后重启。

---

# 0. 给没有背景的人看的说明

我们现在做的是 DG-KAN 的 functional update。普通训练每一步会按 AdamW 改一次参数。我们的想法是在少数训练步里额外加一个小的参数改动，看看它能不能让 KAN 训练得更稳、更快、更不容易忘旧知识，并且最终比 MLP 或强 AdamW 对照更有优势。

每个“动作”可以理解为：

```text
在某个训练步，是否额外给模型参数加这一小块变化？
```

一个好动作不能只看准确率。一个好动作必须同时满足：

```text
短期有帮助；
中期不变坏；
长期不产生 long-risk；
bad/null 低；
不破坏 old family / old stratum；
不破坏 memory/offdiag 几何；
训练当下能用合法信息识别；
运行成本不能太高；
换数据集、换 stratum、换 template 后仍然大体成立。
```

这里几个常用词的意思：

```text
GradeAB：动作质量分层里的好动作集合。A/B 表示 value 正、longrisk 低、风险较干净。
V_integrated_LCB：动作收益的保守下界。大于 0 表示有保守正收益。
h240 longrisk UCB：长时间后出问题的风险上界。越低越好。
bad/null UCB：坏事件和无效事件的风险上界。越低越好。
LDO：leave-dataset-out。留一个数据集不参与调规则，看规则在这个数据集上是否仍然稳。
LSO：leave-stratum-out。留一个 stratum 不参与调规则，看规则是否仍然稳。
LTO：leave-template-out。留一个动作模板不参与调规则，看规则是否仍然稳。
accepted region：最终被规则选中、准备在线上真正执行的一批动作。
controller：训练时决定“要不要接受这个动作”的规则。
certificate：对动作是否安全有用的训练当下证明，不允许使用未来结果字段。
```

---

# 1. v9.6.7 结果的独立判断

## 1.1 这轮有进展，但不是能力成功

v9.6.7 的 route 是：

```text
route = R2-DatasetShiftSolvedButTargetAbsent
strict_purekan_functional = False
full_functional = False
external_ready = False
system_legal_controller_pass = 0
```

这说明没有 selected controller，没有 selected runtime，没有 paired replay，也没有 short/full training。

但是 v9.6.7 不是原地踏步。它把 v9.6.6 的问题拆得更清楚了：

```text
1. Dataset-LDO failure 被进一步量化；
2. score scale shift 是主 failure class；
3. MNIST 还有 target density shift；
4. Core + Expansion target 仍没有 official pass；
5. feature normalization / risk veto 能压 longrisk 或 LDO，但会损坏 precision/value；
6. CERT15C 没有形成可用 accepted region；
7. generated route stop-rule 继续触发，APGU 没有盲跑。
```

v9.6.7 的关键不是“没有进展”，而是：

$$
\boxed{\text{我们已经知道 dataset shift 是怎么表现的，但还没有把它转成合法、稳定、足够干净的动作选择规则。}}
$$

## 1.2 P1 的真实含义：不是某个数据集没有好动作，而是分数尺度和支持度不一样

P1 显示：

```text
dataset_count = 3
dominant_LDO_failure_class = score_scale_shift
dominant_failure_class_fraction = 0.6666666666666666
assigned_failure_fraction = 1.0
dominant_heldout_dataset = Fashion-MNIST
dominant_precision_drop = 0.37931034482758624
no_dataset_specific_selector_produced = 1
```

按数据集看：

```text
Fashion-MNIST:
  action = 1226
  T2.3 count = 34
  GradeAB count = 36
  R5B mean = -6.0217
  TopK87 count/precision = 34 / 1.0
  failure = score_scale_shift

KMNIST:
  action = 828
  T2.3 count = 28
  GradeAB count = 28
  R5B mean = -3.7326
  TopK87 count/precision = 36 / 0.7778
  failure = score_scale_shift

MNIST:
  action = 822
  T2.3 count = 15
  GradeAB count = 15
  R5B mean = -4.0680
  TopK87 count/precision = 17 / 0.8235
  failure = target_density_shift
```

我的理解：

```text
Fashion-MNIST 不是没有好动作；它有 34-36 个好动作，而且 precision 甚至很高。
KMNIST 也不是完全没好动作。
MNIST 的问题更偏 target density：好动作数量本身少。
```

所以不能做 dataset-specific threshold，但可以做 dataset-agnostic 的分数运输、分位数归一、support-aware quota，让同一条规则在不同分布下保持同样语义。

## 1.3 P2 的真实含义：干净集合太少，数量够的集合不稳

P2 的 best target 是 `T2.3-MemoryOffdiagCore`：

```text
accepted = 77
coverage = 0.026773296244784424
GradeAB = 1.0
V_integrated_LCB = 0.16402807605399972
h240_longrisk_UCB = 0.0
bad/null/memory/offdiag UCB = 0.0 / 0.0 / 0.0 / 0.0
LDO_drop = 0.39080459770114945
```

它非常干净，但 count 只有 77。若 coverage 下限是 0.03，2876 个 action 至少需要：

$$
N_{min}=\lceil 0.03\times 2876\rceil=87.
$$

T2.3 还差：

$$
87-77=10.
$$

v9.6.7 还给了两个接近点：

```text
T2.5-CoreRiskCleanExpansion:
  accepted = 87
  GradeAB = 0.9080
  V LCB = 0.1567
  longrisk = 0.0
  null UCB = 0.1527
  offdiag UCB = 0.1820
  LDO = 0.4138

T2.6-R5BHighScoreRiskClean:
  accepted = 87
  GradeAB = 0.8736
  V LCB = 0.1411
  risk/bad/null/memory/offdiag UCB = 0
  LDO = 0.3793
```

这说明核心冲突不是完全没好动作，而是：

$$
\boxed{\text{干净 target 缺 10 个左右；补足 coverage 后，LDO 或 null/offdiag 又坏。}}
$$

## 1.4 P3/P4/P5 的真实含义：风险压下去了，但 value 一起被压坏

P3 best feature `F3A-raw-value-proxy`：

```text
TopK87 precision = 0.39080459770114945
V LCB = -0.0285391768504236
longrisk UCB = 0.33129982990159157
LDO drop = 0.13793103448275862
```

P3 `F3D-risk-normalized-value`：

```text
TopK87 precision = 0.367816091954023
V LCB = -0.024880225746581084
longrisk UCB = 0.0
LDO drop = 0.09195402298850575
```

P4 best ranker `R7A-ValueRankOnly`：

```text
accepted = 87
precision = 0.39080459770114945
V LCB = -0.0285391768504236
longrisk UCB = 0.33129982990159157
```

P5 best certificate `CERT15C-topK64-high-precision`：

```text
accepted = 64
coverage = 0.022253129346314324
precision = 0.484375
V LCB = -0.012902744078215206
longrisk UCB = 0.37383303237969157
bad/null UCB = 0.12180505748880108 / 0.09866091513007542
```

这些结果说明：

```text
1. 只追 value，会带来 longrisk；
2. 加 risk-normalization，可以降低 longrisk 和 LDO；
3. 但加完以后 precision 和 V LCB 掉到不能用；
4. certificate 只能挑到 64 个，而且 V 仍为负、longrisk 仍高。
```

这不是调阈值问题。它说明当前 score 的 value 维度和 risk 维度还没有被正确分离。

## 1.5 generated route 的结论：暂停是对的

v9.6.7 继续触发 generated route stop-rule：

```text
generated_route_stop_triggered = 1
APGU_run = 0
no_fake_APGU_rows = 1
new_positive_created_rate_mean = 0.020833333333333332
longrisk_created_rate_mean = 0.671875
```

连续失败 family：

```text
APGH:
  GradeAB = 0.046875
  V LCB = -0.4881
  longrisk UCB = 0.7869

APGL:
  GradeAB = 0.03125
  V LCB = -0.5827
  longrisk UCB = 0.7869

APGT:
  GradeAB = 0.03125
  V LCB = -0.6095
  longrisk UCB = 0.7869
```

这说明继续盲目新增 APGU/APGV 只会浪费时间。生成器现在不是“差一点”，而是反复生成 value-negative / high-longrisk 动作。

---

# 2. 当前项目进度

## 2.1 已经可靠的部分

```text
1. canonical outcome table 可信；
2. no-transform replay 可信；
3. action apply 可信；
4. Base-Acc Sentinel 健康；
5. no-fake / no-proxy / no CPU offload 守住；
6. dataset 没有进入 official selector/controller；
7. outcome-derived fields 没有进入 official path；
8. generated route stop-rule 正确执行，没有 blind run。
```

## 2.2 仍未完成的部分

```text
1. 没有 template-balanced official target；
2. 没有 dataset-invariant accepted region；
3. 没有 rank-safe certificate；
4. 没有 existing-action minimal controller；
5. 没有 selected runtime；
6. 没有 paired replay；
7. 没有 short/full functional training；
8. generated-action route 暂停且需要重设生成目标。
```

## 2.3 离终极目标的距离

离 system-legal local functional controller 还差：

```text
1. accepted_count >= 87；
2. GradeAB precision >= 0.75；
3. V_integrated_LCB > 0；
4. longrisk/bad/null/memory/offdiag UCB 全部受控；
5. LDO/LSO/LTO drop <= 0.10 到 0.15；
6. selected runtime step_ratio_q90 <= 1.50。
```

离 strict PureKAN functional causal evidence 还差：

```text
1. official paired replay；
2. RealFunctional beats AdamWParallel / bestLR / NoOp / Random；
3. shuffled payload fail；
4. leave-dataset-out / leave-stratum-out / leave-template-out 全部稳定。
```

离 external-ready Beyond-MLP 还差：

```text
1. short/full training；
2. sample efficiency；
3. calibration / robustness；
4. continual / anti-forgetting；
5. strong baseline 排除；
6. external reproducibility。
```

---

# 3. v9.6.8 总体目标

v9.6.8 不再问：

```text
R7A threshold 能不能调一下？
CERT15C 能不能接受更多？
T2.3 阈值能不能放松一点？
APGU 要不要继续跑？
```

v9.6.8 要问：

$$
\boxed{\text{在不按数据集调参的情况下，能否把干净但不足量的核心动作，扩展成足量、干净、跨数据集稳定的 accepted region？}}
$$

本轮分成两条主线。

## 主线 A：existing-action route

目标是使用已经存在的 canonical AP0 action，构建一个可部署的动作选择规则。

核心思路：

```text
先用极干净核心动作做 anchor；
再从相邻动作里补足 coverage；
补充动作必须经过 bad/null/longrisk/memory/offdiag 多重否决；
分数必须做 dataset-agnostic transport，而不是 dataset-specific threshold；
最后用 heldout + LDO + LSO + LTO 验证。
```

## 主线 B：generated-action route

目标不是继续跑 APGU，而是判断生成路线是否应暂停，或者必须用全新目标重启。

核心思路：

```text
APGH/APGL/APGT 已连续失败；
APGU 不运行是正确的；
本轮只做 generator failure notebook 和新 objective spec；
除非 existing-action route 完全失败且新 objective 有清楚证据，否则不生成 APGU/APGV。
```

---

# 4. v9.6.8 核心假设

## H1：LDO failure 主要由 score scale shift 和 target density shift 共同造成

不是某个数据集没有好动作，也不是 dataset-specific rule 缺失。

成立标准：

```text
per-dataset target count 与 R5B/R7 score scale 均有明显差异；
score normalization 能减少 threshold transfer error；
但如果不做 density-aware expansion，coverage 仍不足。
```

失败标准：

```text
score scale 归一化后仍完全无法解释 LDO；
或某 dataset 在所有 target 下都没有可用动作。
```

## H2：核心动作可以作为 anchor，但必须谨慎扩展

T2.3 这类核心动作很干净，但只有 77 个；需要至少补 10 个动作。

成立标准：

```text
核心 target: precision = 1.0, V_LCB > 0, longrisk/bad/null/memory/offdiag = 0；
expansion candidate 可以补到 >=87；
补充后仍满足 precision/value/risk/bad/null/memory/offdiag gate；
LDO drop 降到 <=0.15。
```

失败标准：

```text
只要补足 87，就必然引入 bad/null/offdiag 或 LDO 崩；
说明当前 action universe 的干净可用 support 不足。
```

## H3：value rank、risk veto、memory/offdiag veto 必须分开学习和组合

当前单一 score 会在 precision/value/risk 之间互相破坏。

成立标准：

```text
value_score 单独提高 V，但 longrisk 高；
risk_veto 单独能压 longrisk；
memory/offdiag_veto 能压长期风险；
三者分层组合后，TopK/accepted region 同时满足 value 和 risk。
```

失败标准：

```text
任何 veto 一加，V_LCB 或 precision 必然转负；
说明现有 features 无法支持 accepted region。
```

## H4：generated route 应继续暂停，除非有新的生成目标证据

成立标准：

```text
APGH/APGL/APGT damage notebook 仍显示 longrisk_created 高、V_LCB 负；
新 objective spec 没有可验证的 positive preflight；
APGU 保持 not_run，无 fake rows。
```

失败标准：

```text
找到一个新生成目标可以在 preflight 中证明 value-preserving 和 longrisk-safe；
才允许重启 APGU/APGV。
```

---

# 5. P0：Boundary reproduction 与字段合法性审计

## 目标

确认 v9.6.7 的结果被稳定复现，并且本轮没有使用不允许的信息。

## 输入

```text
v9.6.7 route_decision
canonical AP0 ledger
grade ledger
feature ledger
ranker input
certificate input
Base-Acc Sentinel hash
```

## 必须记录

```text
source_route_v9670
canonical_table_hash
feature_ledger_hash
ranker_input_hash
certificate_input_hash
no_fake
no_proxy
cpu_offload_used
old_table_official_violation_count
dataset_name_used_by_controller_count
outcome_derived_field_used_count
validation_test_used_by_controller_count
future_outcome_used_by_feature_count
```

## 判断标准

P0 pass：

```text
source_route_v9670 = R2-DatasetShiftSolvedButTargetAbsent
no_fake = 1
no_proxy = 1
cpu_offload_used = 0
dataset_name_used_by_controller_count = 0
outcome_derived_field_used_count = 0
old_table_official_violation_count = 0
```

P0 fail：

```text
任何 fake/proxy/old table/outcome-derived/dataset selector 进入 official path。
```

## 可视化

```text
fig_p0_route_waterfall_v9280_to_v9680.svg
fig_p0_field_legality_matrix.svg
```

---

# 6. P1：Dataset score transport audit

## 目标

把 v9.6.7 的 score-scale shift 拆成可处理的分数运输问题。注意：这里不是按 dataset 调阈值，而是测试一个不使用 dataset 名称的通用分数归一规则是否能让分数语义一致。

## 核心思想

训练当下的 score 在不同数据集上尺度不同。若直接用统一阈值，可能在一个数据集上选太多，在另一个数据集上选太少。我们允许使用不含 dataset 名称的分位数、rank、robust z-score、score-to-support mapping，因为这些是每个 batch / event stream 内部可计算的统计，不是手写 dataset branch。

## 候选运输方式

```text
S0 raw score
S1 robust z-score: (score - median) / IQR
S2 within-window percentile rank
S3 family-normalized percentile
S4 template-normalized percentile
S5 memory/offdiag-safe subset percentile
S6 isotonic score-to-risk transport trained without dataset id
S7 quantile mapping with shared calibration, no dataset branch
S8 conformal rank score with fixed global quota
```

## 必须记录

对每个 transport：

```text
transport_id
uses_dataset_name
uses_outcome_field
feature_cost_ms_q90
score_mean_by_dataset
score_std_by_dataset
score_PSI_by_dataset
TopK87_count_by_dataset
TopK87_precision_by_dataset
TopK87_V_LCB_by_dataset
TopK87_longrisk_UCB_by_dataset
threshold_transfer_precision_by_dataset
LDO_drop
LSO_drop
LTO_drop
max_dataset_share
min_dataset_support
```

## 判断标准

P1 strong pass：

```text
uses_dataset_name = 0
uses_outcome_field = 0
LDO_drop <= 0.10
TopK87_precision_macro >= 0.75
V_LCB_macro > 0
longrisk_UCB_macro <= 0.05
bad_UCB_macro <= 0.05
null_UCB_macro <= 0.15
```

P1 weak pass：

```text
LDO_drop <= 0.15
TopK87_precision_macro >= 0.65
V_LCB_macro > 0
longrisk_UCB_macro <= 0.10
```

P1 fail：

```text
score transport 降低 LDO，但 precision/value 变负；
或 score transport 保住 precision/value，但 LDO 仍 > 0.25。
```

## 可视化

```text
fig_p1_score_distribution_by_dataset_before_after.svg
fig_p1_quantile_transport_curve.svg
fig_p1_topk_count_precision_by_dataset.svg
fig_p1_score_transport_pareto_precision_vs_LDO.svg
fig_p1_threshold_transfer_heatmap.svg
```

---

# 7. P2：Core + Expansion target lattice v3

## 目标

解决“干净但少 / 数量够但不稳”的冲突。

## 核心定义

核心动作：

```text
memory/offdiag clean
longrisk = 0
bad = 0
null = 0
V_LCB > 0
GradeAB = 1
```

扩展动作：

```text
和核心动作在 score、memory/offdiag、horizon 行为上相邻；
允许 GradeAB < 1，但必须满足风险边界；
用于把 accepted count 从 77 补到 >=87。
```

## 候选 target

```text
T3.1 CoreOnly: 保留 T2.3 类核心动作
T3.2 Core + nearest 10 by transported score
T3.3 Core + nearest 10 by memory/offdiag-safe value
T3.4 Core + nearest 10 by conformal low-risk score
T3.5 Core + quota expansion per dataset-free score stratum
T3.6 Core + expansion with bad/null veto before rank
T3.7 Core + expansion with memory/offdiag veto before rank
T3.8 Core + expansion with leaveout-stability penalty
T3.9 Core + expansion but cap per family/template/score bucket
T3.10 Core + expansion ensemble intersection
```

## 必须记录

```text
target_id
accepted_count
coverage
GradeAB_precision
V_integrated_LCB
h20/h80/h240 V_LCB
h240_longrisk_UCB
bad_UCB
null_UCB
memory_fail_UCB
offdiag_fail_UCB
candidate_template_count
max_template_share
max_dataset_share
LDO_drop
LSO_drop
LTO_drop
support_by_dataset
support_by_template
support_by_memory_bucket
support_by_offdiag_bucket
```

## 判断标准

P2 strong pass：

```text
accepted_count >= 87
coverage >= 0.03
GradeAB_precision >= 0.75
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
memory_fail_UCB <= 0.05
offdiag_fail_UCB <= 0.05
LDO_drop <= 0.10
LSO_drop <= 0.10
LTO_drop <= 0.10
max_template_share <= 0.25
```

P2 weak pass：

```text
accepted_count >= 87
GradeAB_precision >= 0.70
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.10
bad_UCB <= 0.08
null_UCB <= 0.18
LDO_drop <= 0.15
```

P2 fail：

```text
没有任何 target 同时满足 coverage、V、longrisk、bad/null、memory/offdiag 和 LDO。
```

## 可视化

```text
fig_p2_core_expansion_frontier_count_vs_cleanliness.svg
fig_p2_target_parallel_coordinates.svg
fig_p2_dataset_support_heatmap.svg
fig_p2_template_support_heatmap.svg
fig_p2_value_risk_null_3d_frontier.svg
fig_p2_core_to_expansion_nearest_neighbor_graph.svg
```

---

# 8. P3：Bad/null 与 memory/offdiag 的先后顺序实验

## 目标

v9.6.7 说明 risk/memory/offdiag veto 能降低 longrisk 或 LDO，但会破坏 precision/value。本阶段要查清楚：veto 应该在 rank 前、rank 后，还是作为 separate gate。

## 候选策略

```text
V0 value rank only
V1 risk veto before value rank
V2 risk veto after value rank
V3 bad/null veto before value rank
V4 bad/null veto after value rank
V5 memory/offdiag veto before value rank
V6 memory/offdiag veto after value rank
V7 two-stage: core first, expansion second
V8 three-stage: core -> value expansion -> risk cleanup
V9 constrained optimizer: maximize value subject to risk/bad/null/memory/offdiag constraints
```

## 必须记录

```text
strategy_id
accepted_count
GradeAB_precision
V_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
veto_removed_count
veto_removed_good_count
veto_removed_bad_count
veto_false_positive_rate
veto_false_negative_rate
LDO_drop
support_by_dataset
```

## 判断标准

P3 pass：

```text
存在策略满足 P2 weak pass；
并且 veto_false_positive_rate <= 0.25；
veto_false_negative_rate <= 0.25。
```

P3 fail：

```text
所有 veto 顺序都表现为：risk 降了但 value 变负，或 value 保住但 risk/bad/null 失控。
```

## 可视化

```text
fig_p3_veto_order_waterfall.svg
fig_p3_veto_removed_good_bad_bar.svg
fig_p3_value_vs_longrisk_after_veto.svg
fig_p3_bad_null_memory_offdiag_sankey.svg
```

---

# 9. P4：Dataset-invariant ranker v2，不使用 dataset name

## 目标

训练一个或多个不使用 dataset 名称、不使用 outcome-derived 字段的 ranker，让 score transport + core expansion + veto 的组合形成稳定 accepted region。

## Ranker 类型

```text
R8A transported value rank
R8B transported value + risk veto
R8C transported value + bad/null veto
R8D transported value + memory/offdiag veto
R8E constrained rank: value score with all veto constraints
R8F pairwise rank within score-stratum, no dataset id
R8G leave-dataset-adversarial ranker, but no dataset feature at inference
R8H conformal set ranker with global quota
R8I core-expansion ranker: first select core, then expansion
R8J Pareto ranker: value, risk, bad/null, memory/offdiag separate axes
```

## 注意

允许训练时做 leave-dataset adversarial 或 reweighting，但推理时不能读 dataset name。必须落盘：

```text
uses_dataset_name_at_inference = 0
uses_outcome_field_at_inference = 0
```

## 必须记录

```text
ranker_id
feature_list
red_field_count
amber_field_count
green_field_count
feature_cost_ms_q90
TopK64/87 precision
accepted_count
coverage
V_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
LDO_drop
LSO_drop
LTO_drop
max_dataset_share
max_template_share
worst_dataset_precision
worst_dataset_V_LCB
worst_dataset_longrisk_UCB
```

## 判断标准

P4 strong pass：

```text
accepted_count >= 87
GradeAB_precision >= 0.75
V_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
memory/offdiag_UCB <= 0.05
LDO/LSO/LTO_drop <= 0.10
red_field_count = 0
uses_dataset_name_at_inference = 0
```

P4 weak pass：

```text
accepted_count >= 87
GradeAB_precision >= 0.70
V_LCB > 0
longrisk_UCB <= 0.10
LDO/LSO/LTO_drop <= 0.15
red_field_count = 0
```

P4 fail：

```text
best ranker 仍出现 precision < 0.60 或 V_LCB <= 0 或 LDO_drop > 0.20。
```

## 可视化

```text
fig_p4_ranker_pareto_precision_value_LDO.svg
fig_p4_ranker_worst_dataset_panel.svg
fig_p4_feature_importance_legal_only.svg
fig_p4_rank_score_by_dataset_after_transport.svg
fig_p4_accepted_region_risk_stack.svg
```

---

# 10. P5：Rank-safe certificate v16

## 目标

把 P4 的 ranker 冻结成一个真实 controller candidate。这里不能只看 TopK 表，而要看 frozen accepted region。

## Certificate 类型

```text
C16A fixed TopK87 global
C16B fixed TopK87 with core-expansion split
C16C conformal risk bound
C16D value LCB + risk UCB bound
C16E leaveout robust certificate
C16F quota-free score threshold with transport
C16G topK64 high precision diagnostic only
C16H hybrid: core all + expansion if bound passes
```

## 必须记录

```text
certificate_id
source_ranker_id
accepted_count_heldout
coverage_heldout
GradeAB_precision_heldout
V_LCB_heldout
longrisk_UCB_heldout
bad_UCB_heldout
null_UCB_heldout
memory_UCB_heldout
offdiag_UCB_heldout
LDO_drop
LSO_drop
LTO_drop
ECE
calibration_slope
worst_dataset_precision
worst_dataset_V_LCB
worst_dataset_longrisk_UCB
```

## 判断标准

P5 strong pass：

```text
accepted_count >= 87
coverage >= 0.03
GradeAB_precision >= 0.75
V_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
memory/offdiag_UCB <= 0.05
LDO/LSO/LTO_drop <= 0.10
```

P5 weak pass：

```text
accepted_count >= 87
GradeAB_precision >= 0.70
V_LCB > 0
longrisk_UCB <= 0.10
bad_UCB <= 0.08
null_UCB <= 0.18
LDO/LSO/LTO_drop <= 0.15
```

P5 fail：

```text
TopK diagnostic 强，但 frozen accepted region 无法通过 weak pass。
```

## 可视化

```text
fig_p5_certificate_threshold_sensitivity.svg
fig_p5_accepted_region_by_dataset.svg
fig_p5_accepted_region_by_template.svg
fig_p5_calibration_reliability.svg
fig_p5_heldout_vs_calibration_drift.svg
```

---

# 11. P6：Existing-action minimal controller

## 目标

只有 P5 pass 后才运行。把 certificate 转成训练流里的最小 online controller。

## Controller 限制

```text
no dataset_name
no outcome-derived field
no old table
no validation/test at commit
no payload_norm_bucket shortcut
no candidate_id shortcut
no source_payload_hash shortcut
feature groups <= 6
```

## Controller 形式

```text
score_transport
core_anchor_check
value_rank
risk_veto
bad_null_veto
memory_offdiag_veto
cost_cap
```

接受条件：

$$
Accept(a)=1
$$

当且仅当：

$$
CoreOrExpansion(a)=1,
$$

$$
ValueRank(a)\ge q_v,
$$

$$
RiskVeto(a)=0,
$$

$$
BadNullVeto(a)=0,
$$

$$
MemoryOffdiagVeto(a)=0,
$$

$$
Cost(a)\le C_{max}.
$$

## 必须记录

```text
controller_id
accepted_count_stream
accepted_count_heldout
feature_cost_ms_q90
payload_apply_ms_q90
controller_decision_ms_q90
accepted_per_active_step
zero_event_preservation_pass
base_adamw_equivalence_zero_event_steps
red_field_count
dataset_name_used_count
```

## 判断标准

P6 pass：

```text
P5 strong or weak pass
red_field_count = 0
dataset_name_used_count = 0
feature_cost_ms_q90 <= budget
zero_event_preservation_pass = 1
base_adamw_equivalence_zero_event_steps = 1
```

P6 fail：

```text
P5 未过，或 controller 使用了 forbidden fields，或在线成本明显过高。
```

## 可视化

```text
fig_p6_controller_decision_flow.svg
fig_p6_stream_accepted_timeline.svg
fig_p6_feature_cost_breakdown.svg
```

---

# 12. P7：Selected runtime measurement

## 目标

只有 P6 pass 后运行。真实测 selected controller runtime，不用 estimate。

## 必须记录

```text
online_runtime_measured
selected_controller_in_timed_path
audit_outside_timed_path
step_ratio_q90
step_ratio_mean
memory_ratio
payload_apply_q90
feature_compute_q90
controller_kernel_count
sync_count
zero_candidate_launch_count
active_step_launch_q90
no_cpu_offload
no_proxy_rows
```

## 判断标准

P7 pass：

```text
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
zero_candidate_launch_count = 0
no_cpu_offload = 1
no_proxy_rows = 1
```

P7 fail：

```text
selected controller 过线但 runtime step_ratio_q90 > 1.50，或 memory_ratio > 1.05。
```

## 可视化

```text
fig_p7_runtime_component_stack.svg
fig_p7_step_ratio_distribution.svg
fig_p7_active_step_launch_hist.svg
fig_p7_payload_apply_cost_timeline.svg
```

---

# 13. P8：Generated route stop-rule enforcement

## 目标

保持 generated route 不盲跑。APGU 只有在新 objective 有可验证证据时才允许运行。

## 必须记录

```text
family_count
consecutive_failure_family_count
APGH/APGL/APGT latest metrics
APGU_run
APGU_not_run_reason
no_fake_APGU_rows
new_objective_evidence_present
```

## 判断标准

P8 pass：

```text
generated_route_stop_triggered = 1
APGU_run = 0
no_fake_APGU_rows = 1
```

或者：

```text
new_objective_evidence_present = 1
preflight_value_preserving = 1
preflight_longrisk_safe = 1
then APGU_run allowed
```

P8 fail：

```text
stop-rule 已触发仍然生成 blind APGU rows；
或 fake/proxy APGU rows 出现。
```

## 可视化

```text
fig_p8_generated_family_stop_dashboard.svg
fig_p8_longrisk_created_by_family.svg
fig_p8_damage_mode_pareto.svg
```

---

# 14. P9：Generated failure mechanism notebook

## 目标

虽然不盲跑 APGU，但必须继续解释为什么 APGH/APGL/APGT 反复失败。这个 notebook 不为了 promotion，只为了决定下一代 generator 是否值得重启。

## 必须记录

```text
family
primitive
source_action_quality
value_direction_cosine
AdamW_alignment
memory_delta
offdiag_delta
cover_entropy_delta
basis_rank_delta
longrisk_created_rate
new_positive_created_rate
Damage_V_LCB
dominant_failure_mode
```

## 判断标准

P9 pass：

```text
damage rows 完整；
dominant failure assigned fraction >= 0.80；
给出 clear stop/go recommendation。
```

P9 fail：

```text
failure mode 未能归因；
或 materializer 缺失。
```

## 可视化

```text
fig_p9_source_to_generated_damage_sankey.svg
fig_p9_value_direction_lost_hist.svg
fig_p9_memory_offdiag_delta_by_family.svg
fig_p9_new_positive_vs_longrisk_scatter.svg
```

---

# 15. P10：System boundary decision

## 目标

统一判断：是否进入 paired replay / short-full，还是继续停在 target/rank/certificate 层。

## Route 规则

```text
R0-BoundaryRegression:
  P0 fail。

R1-ScoreTransportHelpsTargetStillAbsent:
  P1 pass or weak pass, but P2 no weak target。

R2-CoreExpansionTargetReadyCertificatePending:
  P2 pass, P5 not yet pass。

R3-ExistingActionControllerCandidateReady:
  P5/P6 pass, P7 pending。

R4-SystemLegalControllerPass:
  P6 and P7 pass。

R5-DatasetInvariantRankImpossibleCurrentFeatures:
  P1-P5 fail, and no transport/veto strategy works。

R6-GeneratedRouteStoppedNoNewObjective:
  existing route fail, generated stop remains triggered, no new objective evidence。

R7-GeneratedRouteRestartAllowed:
  new generator objective passes preflight, APGU/APGV allowed in next run。
```

## 必须输出

```text
route_decision_v9680.json
failure_taxonomy_v9680.csv
stop_conditions_v9680.json
allowed_next_gates_v9680.json
```

---

# 16. P11：Paired replay boundary

## 打开条件

只在以下条件同时成立时打开：

```text
P6 existing-action controller pass = 1
P7 selected runtime pass = 1
no_fake = 1
no_proxy = 1
red_field_count = 0
```

## 必须比较

```text
RealFunctional
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledPayload
```

## 必须记录

```text
beats_AdamWParallel_rate
beats_bestLR_rate
beats_NoOp_rate
beats_Random_rate
shuffled_payload_fail_rate
V_LCB_by_branch
bad/null/longrisk_by_branch
h20/h80/h240 metrics
LDO/LSO/LTO pass
```

## 判断标准

P11 pass：

```text
beats_AdamWParallel_rate >= 0.60
beats_bestLR_rate >= 0.55
beats_NoOp_rate >= 0.65
beats_Random_rate >= 0.65
shuffled_payload_fail_rate >= 0.95
V_LCB > 0
longrisk/bad/null controlled
```

## 可视化

```text
fig_p11_branch_winrate_matrix.svg
fig_p11_real_vs_controls_value_distribution.svg
fig_p11_shuffle_control_dashboard.svg
```

---

# 17. P12：Short/full training boundary

## 打开条件

只在 P11 pass 后打开。

## 目的

验证 functional update 是否真正改善训练，而不是只在 branch replay 中有效。

## 必须记录

```text
train_acc
val_acc
test_acc
CE
NLL
ECE
CEp99
margin_p10
time_to_target
steps_to_target
sample_efficiency
hard_stratum_acc
noise_robust_acc
continual_retained_acc
forgetting_rate
runtime_step_ratio
memory_ratio
```

## Baselines

```text
MatchedMLP
AdamWStrongLRGridMLP
QuadraticFeatureMLP
KAN base without functional update
KAN + NoOp controller
KAN + shuffled payload
```

## 判断标准

不能只靠 acc。至少要满足：

```text
functional KAN beats KAN base on at least one non-acc metric；
functional KAN does not worsen ECE/NLL/CEp99；
functional KAN beats AdamWParallel/bestLR in paired replay；
functional KAN step_ratio_q90 <= 1.50；
no dataset-specific rule used。
```

## 可视化

```text
fig_p12_learning_curves.svg
fig_p12_time_to_target.svg
fig_p12_sample_efficiency.svg
fig_p12_ece_nll_cep99.svg
fig_p12_continual_forgetting_matrix.svg
fig_p12_runtime_quality_pareto.svg
```

---

# 18. 并行执行安排

为了避免继续“一轮只发现一个 blocker”，v9.6.8 必须并行。

## Batch A：立即跑，无需新 materializer

```text
P0 boundary reproduction
P1 score transport audit
P2 core + expansion target lattice v3
P3 veto order experiment
Base-Acc Sentinel reuse audit
```

## Batch B：rank / certificate 并行

```text
P4 dataset-invariant ranker v2
P5 rank-safe certificate v16
```

P4 可以在 P1/P2 的候选 score 生成后立即启动，不等完整 P3。

## Batch C：generated route audit

```text
P8 generated stop-rule enforcement
P9 generated failure mechanism notebook
```

P8/P9 不等 P4/P5，保持并行。

## Batch D：条件触发

```text
P6 controller
P7 selected runtime
P11 paired replay
P12 short/full training
```

只有上游 gate pass 才触发。

---

# 19. 必须落盘的 artifacts

```text
p0_boundary_reproduction_v9680.csv
p0_field_legality_audit_v9680.csv
p1_score_transport_audit_v9680.csv
p1_score_distribution_by_dataset_v9680.csv
p2_core_expansion_target_lattice_v3_v9680.csv
p2_target_support_by_dataset_template_v9680.csv
p3_veto_order_experiment_v9680.csv
p4_dataset_invariant_ranker_v2_v9680.csv
p4_ranker_feature_ledger_v9680.csv
p5_rank_safe_certificate_v16_v9680.csv
p5_certificate_threshold_sensitivity_v9680.csv
p6_existing_action_minimal_controller_v9680.csv
p7_selected_controller_runtime_v9680.csv
p8_generated_route_stop_rule_v9680.csv
p9_generated_failure_mechanism_notebook_v9680.csv
p10_route_decision_v9680.json
p11_paired_replay_boundary_v9680.csv
p12_short_full_boundary_v9680.csv
base_acc_sentinel_v9680.csv
contract_audit_v9680.csv
no_fake_audit_v9680.csv
failure_taxonomy_v9680.csv
run_manifest_v9680.json
```

---

# 20. 必须可视化清单

```text
fig_p0_route_waterfall_v9280_to_v9680.svg
fig_p1_score_distribution_by_dataset_before_after.svg
fig_p1_quantile_transport_curve.svg
fig_p1_topk_count_precision_by_dataset.svg
fig_p1_threshold_transfer_heatmap.svg
fig_p2_core_expansion_frontier_count_vs_cleanliness.svg
fig_p2_target_parallel_coordinates.svg
fig_p2_dataset_support_heatmap.svg
fig_p2_template_support_heatmap.svg
fig_p2_core_to_expansion_nearest_neighbor_graph.svg
fig_p3_veto_order_waterfall.svg
fig_p3_veto_removed_good_bad_bar.svg
fig_p3_value_vs_longrisk_after_veto.svg
fig_p4_ranker_pareto_precision_value_LDO.svg
fig_p4_worst_dataset_panel.svg
fig_p4_feature_importance_legal_only.svg
fig_p5_certificate_threshold_sensitivity.svg
fig_p5_accepted_region_by_dataset.svg
fig_p5_calibration_reliability.svg
fig_p6_controller_decision_flow.svg
fig_p7_runtime_component_stack.svg
fig_p8_generated_family_stop_dashboard.svg
fig_p9_source_to_generated_damage_sankey.svg
fig_p11_branch_winrate_matrix.svg
fig_p12_learning_curves.svg
```

---

# 21. Stop conditions

## Stop score transport patching

```text
best transport LDO_drop > 0.20
or TopK87 precision < 0.60
or V_LCB <= 0
```

如果触发，不再调 score normalization。

## Stop core-expansion target patching

```text
no target with accepted_count >= 87
and GradeAB_precision >= 0.70
and V_LCB > 0
and longrisk_UCB <= 0.10
```

或者：

```text
clean target count remains < 87
and every expansion target violates bad/null/memory/offdiag/LDO。
```

## Stop existing-action route

```text
P1-P5 全部失败；
且没有任何 weak pass；
且 best accepted region precision < 0.60 或 V_LCB <= 0。
```

如果触发，existing-action route 暂停，转入 functional update primitive redesign。

## Stop generated blind variants

```text
generated_route_stop_triggered = 1
and no new_objective_evidence_present
```

则 APGU/APGV 不运行。

---

# 22. 本轮最终必须回答的问题

v9.6.8 结束时必须回答：

```text
1. Score scale shift 是否可以通过 dataset-agnostic transport 解决？
2. 干净核心动作能不能通过受控 expansion 补足到 87 个？
3. Risk/bad/null/memory/offdiag veto 应该在 rank 前还是 rank 后？
4. 是否存在一个不使用 dataset name 的 accepted region？
5. P5 certificate 是否能打开 P6 controller？
6. generated route 是否继续停止？
7. 如果 P1-P5 全失败，existing-action route 是否应该暂停？
8. 下一步是 selected runtime，还是 primitive redesign？
```

---

# 23. v9.6.8 最重要的一句话

v9.6.8 不是再调一个阈值，也不是继续盲跑 generator。

本轮要做的是：

$$
\boxed{\text{用 score transport + core expansion + 多重 veto，把局部好动作转成跨数据集稳定的 accepted region；若做不到，就停止 existing-action patching。}}
$$
