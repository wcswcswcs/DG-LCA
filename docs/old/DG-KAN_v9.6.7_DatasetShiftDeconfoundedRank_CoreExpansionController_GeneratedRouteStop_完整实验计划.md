# DG-KAN v9.6.7 Dataset-Shift Deconfounded Rank / Core-Expansion Controller / Generated Route Stop 完整实验计划

> 本计划基于 v9.6.6 的真实结果制定。  
> v9.6.7 不继续小修 `R6E`、`CERT14A`、`APGU`，也不按数据集写规则。  
> 本轮的核心目标是：把 v9.6.6 中“局部很强、跨数据集不稳”的 existing-action 排名，变成一个 **不使用数据集名字、跨数据集稳定、能冻结成 accepted region、能进入 runtime 测量** 的 functional update 选择规则。  
> 如果做不到，就要明确判定：当前 AP0 existing-action 路线虽然有局部信号，但还不能成为 system-legal controller。

---

# 0. 先把话说清楚：我们现在到底在做什么

训练 KAN 时，每一步本来会按 AdamW 改一次参数。我们想额外加一个小的参数改动。每个这样的额外改动叫一个 **action**，也就是：

```text
在某个训练步，除了 AdamW 之外，再额外加这一小块参数变化。
```

我们要解决的问题不是：

```text
MNIST / Fashion-MNIST / KMNIST 上 acc 能不能更高？
```

而是：

```text
训练当下，能不能知道哪些额外参数改动值得加？
加了以后是不是比 AdamW / bestLR / NoOp / Random 都好？
短期有帮助吗？
中期会不会坏？
长期会不会产生 long-risk？
会不会忘旧知识？
会不会太慢？
换一个数据集、换一个数据片区后还成立吗？
```

所以本项目的 functional update 成功，不是“某个数据集 acc 涨了”，而是：

$$
\boxed{
\text{能在训练当下合法地挑出一批额外参数改动，它们稳定、便宜、可复现、跨 group 有效，并且打过强对照。}
}
$$

这里的 **group** 指的是我们用来检查泛化稳定性的分组，比如 dataset、stratum、family、template、payload bucket、memory bucket。  
重要原则是：

```text
可以用 group 做诊断和 leaveout；
不可以在 official controller 里写 if dataset == xxx 这种规则。
```

---

# 1. v9.6.6 的独立判断

## 1.1 这轮有进展，但不是能力成功

v9.6.6 的 route 是：

```text
route = R2-NoTemplateBalancedTargetStill
strict_purekan_functional = False
full_functional = False
external_ready = False
```

它没有选出 official controller，也没有测 selected runtime，更没有打开 paired replay / short-full training。

但它有真实进展：

```text
1. v9.6.5 的边界被复现，没有跳过旧 blocker；
2. LDO failure 被拆到 dataset 级别，dominant heldout dataset 是 Fashion-MNIST；
3. Core + Expansion target lattice 证明：coverage 够的 target 仍不够干净，干净 target 又不够多；
4. memory/offdiag 作为 risk veto 有真实诊断信号；
5. dataset-invariant ranker 能降低一部分 LDO drop，但牺牲了 precision / value / risk；
6. generated route 连续失败，blind variant stop-rule 触发，APGU 按计划不跑。
```

所以这轮不是空转。它把问题从：

```text
template-lineage collapse？
memory/offdiag 有没有用？
existing-action rank 是不是完全没信号？
generated primitive 是不是继续试？
```

推进到：

```text
dataset leaveout 不稳；
template-balanced target 仍不存在；
memory/offdiag 适合作为 risk veto，但不是 value scorer；
generated blind variants 应该停止。
```

## 1.2 最关键的新事实：LDO 主要卡在 dataset-level shift

v9.6.6 的 P1 显示：

```text
LDO primary axis = dataset_id
dominant heldout dataset = Fashion-MNIST
dominant precision drop = 0.37931034482758624
explained fraction = 1.0
```

这说明，v9.6.5 的 R5B / CERT13 之所以 LDO 不稳，主要不是 template collapse，也不是 action apply / replay 问题，而是 **数据集之间的分布差异影响了 ranker**。

但本轮也明确：

```text
no_forbidden_dataset_specific_selector_produced = 1
```

也就是说，dataset 只用于诊断，不进入 selector。这个是正确的。我们不能因为 Fashion-MNIST 是 dominant heldout，就写一个 Fashion-MNIST 专用规则。

## 1.3 Existing-action 路线仍然最接近成功

v9.6.5 的 R5B / CERT13 已经能选出一批质量很高的 existing actions：

```text
TopK87 precision = 0.873563
V_integrated_LCB = 0.141113
h240 longrisk UCB = 0.0
candidate_template_count = 87
LTO drop = 0.01149
```

v9.6.6 进一步做 dataset-invariant ranker，best 是：

```text
R6E-GroupAdversarialScoreNormalization
TopK87 precision = 0.459770
V LCB = 0.002219
longrisk UCB = 0.152675
LDO drop = 0.160920
```

这个结果有两个含义：

```text
1. 强行做 dataset-invariant 后，LDO drop 确实下降了一部分；
2. 但 precision / value / longrisk 明显变差。
```

因此现在不是“没有 rank 信号”，而是：

$$
\boxed{
\text{强 rank 信号和 dataset-invariance 发生冲突。}
}
$$

R5B 类 ranker 质量高但 LDO 不稳；R6E 类 ranker LDO 稍稳但质量不够。

## 1.4 Target 层仍然卡在“干净但少，数量够但脏”

P2 的 best target 是：

```text
T2.1-CoreExpansionValueNoLongRisk
accepted = 97
coverage = 0.033727
GradeAB = 0.814433
V LCB = 0.166222
longrisk UCB = 0.0
bad UCB = 0.150521
null UCB = 0.189235
LDO drop = 0.413793
```

它的好处是：

```text
数量够；
value 正；
longrisk 低。
```

但它的问题是：

```text
bad UCB 太高；
null UCB 太高；
LDO drop 太高。
```

clean targets：

```text
T2.2-CoreClean accepted = 79
T2.3-MemoryOffdiagCore accepted = 77
bad/null/longrisk = 0
```

它们的质量更干净，但 coverage 小于 $0.03$。如果 canonical action count 是 $2876$，coverage 下限 $0.03$ 对应：

$$
N_{min}=\lceil 0.03 \times 2876\rceil = 87.
$$

所以：

```text
T2.2 缺 8 个；
T2.3 缺 10 个；
T2.1 数量够，但坏动作和空动作太多。
```

这就是 v9.6.6 的核心目标冲突。

## 1.5 memory/offdiag 是有效风险线索，但不是完整 selector

P3 显示：

```text
matched joint GradeAB lift = 0.191436
V lift = 0.290439
longrisk drop = 0.906693
legal proxy AUC_longrisk = 0.874089
```

这很重要。它说明 memory/offdiag 不是空信号。它确实和 longrisk、安全性、value 有联系。

但 P3 也显示：

```text
TopK longrisk before/after veto = 21 / 19
removal fraction = 0.095238
```

也就是说，memory/offdiag 单独做 TopK removal 不够强。我的判断是：

$$
\boxed{
\text{memory/offdiag 适合作为 risk veto，不适合作为单独的 value ranker。}
}
$$

换句话说，它能告诉我们“哪些动作容易长期出问题”，但不能单独告诉我们“哪些动作最值得加”。

## 1.6 generated route 该停，不应继续盲目变体

v9.6.6 的 P8 触发 generated route stop-rule：

```text
APGH / APGL / APGT 连续满足：
GradeAB < 0.10
V LCB < 0
longrisk UCB > 0.50
new positive < 0.05
longrisk created > 0.50
```

damage consolidation：

```text
dominant damage = D7-longrisk-created
mean longrisk created rate = 0.727539
mean Damage V LCB = -0.605683
```

这说明 generated-action 路线现在不是“差一点”。连续多轮 APG/APGA/APGD/APGE/APGF/APGH/APGL/APGT 都是：

```text
工程能跑；
payload 能生成；
branch-horizon 能落盘；
但生成动作 value-negative / high-longrisk。
```

因此 v9.6.7 不应该继续 APGU blind variant。除非我们有新机制，否则 generated route 应该保持停止，只做必要的 autopsy 和极小 preflight。

---

# 2. 当前项目进度

## 2.1 已经比较可信的东西

```text
1. canonical outcome universe 可信；
2. no-transform replay / old table quarantine 已解决；
3. action apply / payload replay / branch-horizon materializer 可信；
4. Base-Acc Sentinel 健康；
5. existing-action universe 里确实有高质量局部动作；
6. template collapse 已经被重新定义后解除；
7. memory/offdiag 与 longrisk 有强关系；
8. existing-action rank 有强局部信号；
9. generated blind variants 应该停止。
```

## 2.2 还没有完成的东西

```text
1. 没有 template-balanced + dataset-LDO-stable target；
2. 没有 legal ranker 同时满足 precision、value、risk、bad/null、LDO/LSO/LTO；
3. 没有 rank-safe certificate；
4. 没有 minimal controller；
5. 没有 selected runtime；
6. 没有 official paired replay；
7. 没有 short/full functional training；
8. generated-action route 没有 positive frontier。
```

## 2.3 现在真正卡在哪里

当前最准确 blocker 是：

$$
\boxed{
\text{existing-action good region 很强，但 dataset leaveout 不稳；generated-action 路线连续失败。}
}
$$

拆成四个具体卡点：

```text
B1. Dataset shift:
    Fashion-MNIST leaveout 是主 drop 来源。
    不能按 dataset 调参，但必须诊断 score / support / target density 差异。

B2. Target density vs clean quality:
    够数量的 target 有 bad/null/LDO 问题；
    干净 target 数量不足。

B3. Risk veto integration:
    memory/offdiag 有强诊断价值；
    但现在的 veto 没有足够改变 TopK accepted region。

B4. Generated route:
    多个 primitive family 连续 high-longrisk；
    应触发 stop-rule，不继续 blind variant。
```

---

# 3. v9.6.7 的总体目标

v9.6.7 的目标不是再调一个 threshold，也不是新增 APGU 小变体。

本轮要做三件更根本的事：

```text
1. 把 dataset-LDO drop 拆清楚：到底是 score shift、support shift、target density shift，还是 bad/null shift？
2. 把 clean core target 和 expansion target 合起来：既满足 coverage，又不引入 bad/null/longrisk。
3. 把 value rank、bad/null veto、longrisk veto、memory/offdiag veto 分开建模，再组合成 frozen accepted region。
```

一句话：

$$
\boxed{
\text{v9.6.7 要把 existing-action strong local rank 转成 dataset-invariant accepted region；若失败，则明确停止 existing-action controller promotion。}
}
$$

---

# 4. 本轮硬约束

继续遵守：

```text
no teacher
no self-teacher
no distillation as training trick
no loss modification
no label smoothing
no focal / margin auxiliary loss
no sampler / class weight
no CPU offload
no fake / proxy rows
official controller 不使用 dataset_name 分支
official controller 不使用 validation/test/future outcome
official controller 不使用 outcome-derived fields
old outcome table 不进入 official
payload_norm_bucket 不作为 selector
candidate_id 不作为 selector
source_payload_hash 不作为 selector
generated route stop-rule 不得被绕过
```

允许：

```text
dataset 作为 leaveout / autopsy 轴；
dataset 作为 offline group-DRO / worst-group training 的 group label；
但最终 score function 不读取 dataset_name；
memory/offdiag legal proxy 作为 risk veto；
template / family / stratum 作为 diversity audit；
calibration split 训练 legal ranker；
heldout split 冻结评估；
LDO / LSO / LTO / leave-family-out 作为 pass gate；
runtime 在 controller 过线后立即 measured。
```

---

# 5. 本轮核心假设

## H1：LDO failure 主要来自 score distribution shift，而不是好动作完全不存在

解释：

```text
如果 Fashion-MNIST leaveout 后 precision 掉很多，可能不是 Fashion 没有好动作；
而是 rank score 在 Fashion 上尺度不同，导致同一个 threshold / topK 不再合适。
```

记录：

```text
per_dataset_score_mean
per_dataset_score_std
per_dataset_score_quantiles
per_dataset_target_density
per_dataset_GradeAB_precision
per_dataset_bad_rate
per_dataset_null_rate
per_dataset_longrisk_rate
threshold_transfer_error
score_PSI
score_KL
score_Wasserstein
```

成立标准：

```text
目标正例在每个 dataset 都存在；
但 raw score 分布差异导致 threshold transfer fail；
经过不使用 dataset_name 的 robust normalization 后 LDO drop 明显降低。
```

失败标准：

```text
某个 heldout dataset 的 target density 本身不足；
或者 legal features 在该 dataset 上无法区分 good/bad。
```

## H2：Core + Expansion 可以解决“干净但少 / 数量够但脏”的冲突

解释：

```text
Core 是非常干净的动作，比如 T2.2/T2.3；
Expansion 是接近 core，但需要通过 bad/null/memory/offdiag veto 的动作。
```

目标：

```text
Core 保证质量；
Expansion 补足 coverage；
Veto 负责不引入 longrisk/bad/null。
```

成立标准：

```text
accepted_count >= 87
coverage >= 0.03
GradeAB precision >= 0.75
V_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
LDO/LSO/LTO drop <= 0.10
```

失败标准：

```text
core + expansion 一旦补到 87 个，bad/null 或 LDO 必然超标。
```

## H3：memory/offdiag 应作为风险否决，不应作为价值排序

解释：

```text
memory/offdiag safe 对 longrisk 有强关系；
但它本身不能保证 value positive。
```

成立标准：

```text
memory/offdiag veto 可以显著降低 longrisk 和 bad/null；
同时不把 accepted_count 压低到 87 以下；
且不让 V_LCB 变负。
```

失败标准：

```text
veto 后 longrisk 变化小；
或者 veto 后 coverage / value 崩。
```

## H4：existing-action route 比 generated-action route 更值得优先推进

解释：

```text
existing-action rank 已经多轮出现高质量局部区域；
generated-action 多轮连续 value-negative / high-longrisk。
```

成立标准：

```text
existing-action ranker 在 v9.6.7 出现一个 weak/strong controller candidate；
generated route 仍保持 stop-rule，不浪费大量算力。
```

失败标准：

```text
existing-action route 在清掉 dataset shift 后仍无 accepted region；
此时才重启 generated route，但必须换机制，不允许 blind variant。
```

---

# 6. 实验阶段

## P0. Boundary 与 artifact 审计

### 目标

确认 v9.6.6 的输入边界没有漂移。

### 必须记录

```text
source_v9660_route
canonical_outcome_table_hash
grade_ledger_hash
feature_ledger_hash
ranker_input_hash
certificate_input_hash
Base-Acc Sentinel hash
no_fake_proxy_cpu_offload flags
old_table_official_violation_count
dataset_name_used_by_controller_count
outcome_derived_field_used_count
generated_stop_rule_status
```

### 判断标准

P0 pass：

```text
source route = R2-NoTemplateBalancedTargetStill
canonical outcome ready = 1
no fake/proxy/cpu offload = 1
old table official = 0
dataset_name controller = 0
outcome-derived official field = 0
generated stop-rule preserved = 1
```

### 可视化

```text
fig_p0_artifact_dependency_graph.svg
fig_p0_field_legality_table.svg
```

---

## P1. Dataset-LDO failure autopsy v2

### 目标

把 Fashion-MNIST LDO drop 拆成四类：

```text
1. target density shift；
2. score scale shift；
3. bad/null distribution shift；
4. memory/offdiag distribution shift。
```

### 必须记录

对每个 dataset：

```text
dataset_id
action_count
target_T2_1_count
target_T2_2_count
target_T2_3_count
GradeAB_count
ValuePositiveNoLongRisk_count
bad_count
null_count
longrisk_count
memory_fail_rate
offdiag_fail_rate
R5B_score_mean/std/q10/q50/q90
R6E_score_mean/std/q10/q50/q90
score_PSI_vs_global
score_KL_vs_global
score_Wasserstein_vs_global
TopK87_count
TopK87_precision
TopK87_V_LCB
TopK87_longrisk_UCB
TopK87_bad_UCB
TopK87_null_UCB
threshold_transfer_precision
threshold_transfer_coverage
```

### 关键分析

```text
A. 如果 Fashion-MNIST target density 够，但 threshold transfer 差：
   说明是 score shift。

B. 如果 Fashion-MNIST target density 不够：
   说明是 action population / target density problem。

C. 如果 Fashion-MNIST bad/null 高：
   说明 target definition 太宽。

D. 如果 Fashion-MNIST memory/offdiag fail 高：
   说明 risk veto 应该更早进入。
```

### 判断标准

P1 pass：

```text
dominant LDO failure class assigned
assigned fraction >= 0.80
per-dataset target density and score shift recorded
no dataset-specific selector produced
```

### 可视化

```text
fig_p1_per_dataset_target_density_bar.svg
fig_p1_score_distribution_by_dataset.svg
fig_p1_threshold_transfer_heatmap.svg
fig_p1_LDO_drop_waterfall.svg
fig_p1_dataset_bad_null_longrisk_stack.svg
```

---

## P2. Core + Expansion target lattice v2

### 目标

构造一个不是单一标签的目标结构：

```text
Core:
  干净但少。

Expansion:
  数量补足，但必须通过 risk / bad / null / memory / offdiag gate。

Reject:
  value 低、longrisk 高、bad/null 高、memory/offdiag fail。
```

### 候选目标

```text
Core-1:
  T2.2-CoreClean

Core-2:
  T2.3-MemoryOffdiagCore

Expansion-1:
  ValuePositiveNoLongRisk
  且 bad_proxy <= threshold
  且 null_proxy <= threshold

Expansion-2:
  GradeAB
  且 memory/offdiag safe
  且 low hard-tail risk

Expansion-3:
  R5B high score
  且 R6E normalized high score
  且 bad/null proxy low
```

### 必须记录

```text
target_id
core_count
expansion_count
accepted_count_total
coverage
GradeAB precision
V_integrated_LCB
h20_V_LCB
h80_V_LCB
h240_V_LCB
h240_longrisk_UCB
bad_UCB
null_UCB
memory_fail_UCB
offdiag_fail_UCB
per_dataset_precision
per_dataset_coverage
LDO_drop
LSO_drop
LTO_drop
max_dataset_share
max_template_share
max_family_share
```

### 判断标准

Strong target pass：

```text
accepted_count >= 87
coverage >= 0.03
GradeAB precision >= 0.75
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
memory_fail_UCB <= 0.05
offdiag_fail_UCB <= 0.05
LDO_drop <= 0.10
LSO_drop <= 0.10
LTO_drop <= 0.10
max_dataset_share <= 0.50
max_template_share <= 0.25
```

Weak target pass：

```text
accepted_count >= 87
GradeAB precision >= 0.70
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.10
bad_UCB <= 0.08
null_UCB <= 0.18
LDO_drop <= 0.20
```

### 可视化

```text
fig_p2_core_expansion_venn.svg
fig_p2_precision_coverage_curve.svg
fig_p2_bad_null_vs_expansion_size.svg
fig_p2_LDO_drop_vs_expansion_size.svg
fig_p2_target_risk_table.svg
```

---

## P3. Legal feature normalization and deconfounding v3

### 目标

不使用 dataset_name，但修复 score scale shift。

### 可用方法

```text
1. global robust z-score:
   使用 calibration stream 的 median / MAD。

2. event-local percentile:
   在同一 active step / event family 的候选中做 percentile。

3. template-diversity cap:
   不直接用 candidate_id，当作 support cap。

4. memory/offdiag risk-normalized score:
   value score 与 risk score 分离。

5. distributionally robust training:
   训练时用 dataset 作为 group 权重；
   推理时不读 dataset_name。
```

### 禁止方法

```text
if dataset == Fashion-MNIST then threshold = ...
dataset-specific topK
dataset-specific calibration threshold
payload_norm_bucket selector
candidate_id selector
future outcome feature
GradeAB label as feature
```

### 必须记录

```text
feature_name
feature_legality
normalization_method
uses_dataset_name_at_commit
uses_outcome_field
raw_AUC
normalized_AUC
TopK87_precision
TopK87_V_LCB
TopK87_longrisk_UCB
TopK87_bad_UCB
TopK87_null_UCB
LDO_drop
score_shift_before
score_shift_after
feature_cost_ms_q90
```

### 判断标准

P3 pass：

```text
at least one normalized legal feature or feature group:
  TopK87 precision >= 0.65
  V_LCB > 0
  longrisk_UCB <= 0.10
  LDO_drop <= 0.20
  no red field
  no dataset-name commit feature
```

### 可视化

```text
fig_p3_feature_shift_before_after.svg
fig_p3_feature_cost_vs_precision.svg
fig_p3_legal_feature_heatmap.svg
```

---

## P4. Value-rank + bad/null/risk/memory veto ranker

### 目标

不要再用一个分数硬选动作。把决策拆成四步：

```text
1. value rank:
   先找可能有正收益的动作。

2. longrisk veto:
   去掉长期风险动作。

3. bad/null veto:
   去掉明确坏动作和空动作。

4. memory/offdiag veto:
   去掉容易忘旧知识或破坏 offdiag geometry 的动作。
```

### Ranker families

```text
R7A-ValueRankOnly:
  只用 value score，作为 baseline。

R7B-ValueRankLongRiskVeto:
  value score + longrisk veto。

R7C-ValueRankBadNullVeto:
  value score + bad/null veto。

R7D-ValueRiskBadNullMemoryOffdiag:
  value + longrisk + bad/null + memory/offdiag。

R7E-DatasetInvariantNormalizedRank:
  只用 legal normalized features，不读 dataset_name。

R7F-GroupDROPairwiseRank:
  训练时 group-DRO，commit-time 不读 group id。

R7G-CoreExpansionRank:
  先接收 clean core，再用 ranker 补 expansion。
```

### 必须记录

```text
ranker_id
feature_list
red_field_count
dataset_name_commit_feature_count
accepted_count
coverage
GradeAB_precision
V_integrated_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
per_dataset_precision
per_dataset_coverage
LDO_drop
LSO_drop
LTO_drop
max_dataset_share
max_template_share
feature_cost_ms_q90
```

### Strong pass

```text
accepted_count >= 87
GradeAB_precision >= 0.75
V_integrated_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
memory_UCB <= 0.05
offdiag_UCB <= 0.05
LDO_drop <= 0.10
LSO_drop <= 0.10
LTO_drop <= 0.10
```

### Weak pass

```text
accepted_count >= 87
GradeAB_precision >= 0.70
V_integrated_LCB > 0
longrisk_UCB <= 0.10
bad_UCB <= 0.08
null_UCB <= 0.18
LDO_drop <= 0.20
```

### 可视化

```text
fig_p4_ranker_precision_coverage.svg
fig_p4_value_vs_longrisk_scatter.svg
fig_p4_veto_ablation_waterfall.svg
fig_p4_per_dataset_precision_heatmap.svg
fig_p4_ranker_drop_comparison.svg
```

---

## P5. Rank-safe certificate v15

### 目标

把 P4 ranker 冻结成 accepted region。

### 证书形式

这里的 certificate 不是概率预测器，而是一张训练当下可算的“通过单”：

```text
value_score
longrisk_score
bad_score
null_score
memory_score
offdiag_score
support_score
cost_score
```

接受规则：

$$
Accept(a)=1
$$

当且仅当：

$$
S_v(a)\ge \tau_v
$$

$$
S_l(a)\le \tau_l
$$

$$
S_b(a)\le \tau_b
$$

$$
S_n(a)\le \tau_n
$$

$$
S_m(a)\le \tau_m
$$

$$
S_o(a)\le \tau_o
$$

$$
Support(a)\ge \tau_s
$$

$$
Cost(a)\le \tau_c.
$$

### Calibration protocol

```text
calibration split:
  freeze thresholds

heldout split:
  evaluate

leaveout:
  LDO, LSO, LTO, leave-family-out, leave-template-out

no refit after heldout
```

### 必须记录

```text
certificate_id
thresholds
calibration_hash
heldout_hash
accepted_cal
precision_cal
V_LCB_cal
bad_UCB_cal
null_UCB_cal
longrisk_UCB_cal
accepted_heldout
coverage_heldout
precision_heldout
V_LCB_heldout
bad_UCB_heldout
null_UCB_heldout
longrisk_UCB_heldout
memory_UCB_heldout
offdiag_UCB_heldout
LDO_drop
LSO_drop
LTO_drop
leave_family_drop
leave_template_drop
ECE_optional
feature_cost_q90
certificate_cost_q90
```

### Pass

P5 strong pass requires P4 strong pass and:

```text
accepted_heldout >= 87
coverage_heldout >= 0.03
precision_heldout >= 0.75
V_LCB_heldout > 0
longrisk_UCB_heldout <= 0.05
bad_UCB_heldout <= 0.05
null_UCB_heldout <= 0.15
LDO_drop <= 0.10
LSO_drop <= 0.10
LTO_drop <= 0.10
feature/certificate cost within runtime budget
```

P5 weak pass:

```text
precision_heldout >= 0.70
V_LCB_heldout > 0
longrisk_UCB_heldout <= 0.10
LDO_drop <= 0.20
```

### 可视化

```text
fig_p5_calibration_vs_heldout.svg
fig_p5_accepted_region_value_risk.svg
fig_p5_leaveout_drop_bar.svg
fig_p5_certificate_threshold_sensitivity.svg
```

---

## P6. Existing-action minimal controller

### 目标

只有 P5 pass 后才生成 controller artifact。

### Controller 限制

```text
feature_groups <= 8
no dataset_name branch
no outcome-derived fields
no payload_norm_bucket selector
no candidate_id selector
no source_payload_hash selector
thresholds frozen
```

### 必须记录

```text
controller_id
controller_version
feature_names
feature_legality_hash
thresholds
accepted_action_count
accepted_event_count
accepted_per_active_step
precision
coverage
V_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
LDO/LSO/LTO
feature_cost_ms_q90
certificate_cost_ms_q90
```

### Pass

```text
P6 pass = 1
if P5 strong pass and legality audit pass.
```

如果 P5 weak pass only：

```text
do not open official paired replay；
allow P7 runtime preflight only as diagnostic.
```

---

## P7. Selected controller runtime

### 目标

如果 P6 strong pass，真实测 selected controller runtime。

### 必须记录

```text
selected_controller_in_timed_path
audit_outside_timed_path
feature_compute_ms_q90
rank_compute_ms_q90
certificate_compute_ms_q90
payload_apply_ms_q90
step_ratio_q90
memory_ratio
kernel_count
sync_count
zero_candidate_launch_count
active_step_count
selected_actions_per_active_step
empty_step_kernel_count
```

### Pass

```text
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
zero_candidate_launch_count = 0
selected_controller_in_timed_path = 1
audit_outside_timed_path = 1
no CPU offload
no proxy rows
```

### 可视化

```text
fig_p7_step_time_distribution.svg
fig_p7_runtime_component_stack.svg
fig_p7_selected_actions_per_step.svg
```

---

## P8. Generated route stop-rule enforcement

### 目标

确认 APGU 不被 blind run。

### 必须记录

```text
APGH_best_precision
APGL_best_precision
APGT_best_precision
APGH_best_V_LCB
APGL_best_V_LCB
APGT_best_V_LCB
APGH_best_longrisk_UCB
APGL_best_longrisk_UCB
APGT_best_longrisk_UCB
new_positive_created_rate_mean
longrisk_created_rate_mean
generated_route_stop_triggered
APGU_run
APGU_not_run_reason
```

### Pass

```text
generated_route_stop_triggered = 1
APGU_run = 0
no fake APGU rows
```

### 判断

如果 existing-action route fails and generated stop remains true：

```text
route = R_generated_route_requires_mechanism_reset
```

不是继续 APGU blind variant。

---

## P9. Generated damage mechanism notebook

### 目标

不生成新 APGU，但把 APGH/APGL/APGT 的失败做成可复用 notebook。

### 分解维度

```text
value direction lost
longrisk created
memory fail created
offdiag fail created
cover entropy collapse
basis rank drop
AdamW conflict
payload OOD
source action quality
```

### 必须记录

```text
primitive_family
primitive_id
source_action_id
generated_action_id
Damage_V_LCB
value_direction_cosine
longrisk_created
memory_delta
offdiag_delta
cover_entropy_delta
basis_rank_delta
AdamW_conflict_delta
payload_norm_shift
source_quality
```

### 可视化

```text
fig_p9_generated_damage_matrix.svg
fig_p9_value_direction_lost_by_primitive.svg
fig_p9_longrisk_created_by_primitive.svg
fig_p9_memory_offdiag_damage_heatmap.svg
```

### Pass

```text
dominant damage mode assigned fraction >= 0.80
stop/go recommendation produced
```

---

## P10. Base-Acc Sentinel continuation

### 目标

继续健康检查，但不用于 controller。

### 记录

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0..9
models:
  LQ-t2-h256
  MatchedMLP
  AdamWStrongLRGridMLP
  QuadraticFeatureMLP
metrics:
  train_acc
  val_acc
  test_acc
  train_loss
  val_loss
  test_loss
  ECE
  NLL
  CEp99
```

### Pass

```text
LQ_catastrophic_fail = 0
base_acc_used_for_controller = 0
```

---

## P11. System / paired replay boundary

### 目标

只有 P6 + P7 strong pass 后打开。

### 如果打开，必须记录

```text
RealFunctional
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledPayload
CertificatePassNoPayload
```

Horizons：

```text
h20
h80
h240
short-run
full-run
```

指标：

```text
accuracy
CE
NLL
ECE
CEp99
margin_p10
old_family_retention
old_stratum_retention
sample_efficiency
runtime
memory
```

### Gate

```text
if P6 strong pass and P7 strong pass:
  open official paired replay
else:
  not_run
```

---

# 7. Route Decision

v9.6.7 必须输出明确 route：

```text
R1-ExistingActionControllerReady:
  P5/P6 strong pass

R2-DatasetShiftSolvedButTargetAbsent:
  P1 score shift solved but P2 target absent

R3-TargetExistsButLegalRankFails:
  P2 pass, P4/P5 fail

R4-LegalRankStrongButLDOStillFails:
  P4 TopK strong but LDO/LSO/LTO fail

R5-CertificateAcceptedRegionFails:
  P4 pass but P5 heldout accepted region fails

R6-RuntimeFails:
  P6 pass but P7 step_ratio/memory fails

R7-GeneratedRouteStoppedExistingActionFails:
  generated stop true and existing-action route fails

R8-NoUsableFunctionalUpdateRoute:
  no target, no legal rank, generated route stopped
```

---

# 8. 最重要的判断标准

本轮不能只看：

```text
AUC
TopK precision
single dataset precision
pooled precision
Base-Acc Sentinel
implementation pass
branch-horizon row count
```

必须同时看：

```text
accepted_count >= 87
coverage >= 0.03
precision / GradeAB >= 0.75
V_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
memory/offdiag fail low
LDO/LSO/LTO drop <= 0.10
no dataset_name at commit
runtime step_ratio_q90 <= 1.50
```

---

# 9. 本轮预期结论

我预期 v9.6.7 最可能出现三种结果。

## Case A：existing-action route 过 weak，但不过 strong

表现：

```text
accepted_count >= 87
precision >= 0.70
V_LCB > 0
longrisk <= 0.10
LDO_drop around 0.15-0.20
```

含义：

```text
路线还活着，但不能 paired replay official；
下一轮应专攻 LDO / bad-null。
```

## Case B：target 仍然不存在

表现：

```text
core+expansion 只要补足 coverage 就 bad/null 或 LDO 崩。
```

含义：

```text
existing-action universe 中可部署 action 密度不够；
需要重建 action generation，但不是 APGU blind variant。
```

## Case C：ranker 仍然只在 dataset-specific distribution 里有效

表现：

```text
TopK pooled high；
leave-dataset-out 仍大幅掉；
score normalization / group-DRO 不能修。
```

含义：

```text
legal rank 不是通用训练几何规则；
必须回到更底层的 geometry generator。
```

---

# 10. 本轮一句话

v9.6.7 的一句话目标是：

$$
\boxed{
\text{不用数据集名字，不用事后结果字段，把已有强局部好动作转成跨数据集稳定的 accepted region；如果做不到，就停止 existing-action controller promotion。}
}
$$
