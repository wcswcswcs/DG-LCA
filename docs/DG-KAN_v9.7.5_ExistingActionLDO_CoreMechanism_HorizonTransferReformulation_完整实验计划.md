# DG-KAN v9.7.5 Existing-Action LDO / Core Mechanism / Horizon-Transfer Reformulation 完整实验计划

> 本计划基于 v9.7.4 `ExistingAction LDO / Transfer Mismatch / Horizon Transfer` 的真实结果制定。  
> v9.7.5 不继续小修 `ExactT4`、`WT80-LCB`、`E1 top10`、`R8A threshold` 或 APG/APGU/APGV 小变体。  
> 本轮目标是把当前最强的 existing-action 线索拆成可解释、可复现、可跨数据集稳定的规则；同时重新判断 transfer 思路到底是错在“原理”，还是错在“只看一步 CE 下降”的目标太窄。

---

# 0. 一句话判断

v9.7.4 的结果说明：

$$
\boxed{
\text{已有动作里确实有一批高质量动作，但当前选择规则换数据集后不稳；one-step / windowed transfer 还不能解释这些高质量动作。}
}
$$

这意味着下一轮不能再问：

```text
Exact transfer threshold 再调一下能不能过？
WT80 再调一下能不能过？
Core77 再补 10 个动作能不能过？
APGU/APGV/APGW 能不能重新打开？
```

而要问四个更根本的问题：

```text
1. Core77 为什么自己就 LDO fail？
2. OldRank 为什么能找到 OldOnly 这批高价值动作？
3. Transfer 为什么会排掉这些高价值动作？
4. 有没有一个不用 dataset_name、不用 outcome 字段、能跨数据集稳定选择动作的最小规则？
```

---

# 1. 对 v9.7.4 的独立判断

## 1.1 有进展，但不是能力成功

v9.7.4 没有 strict PureKAN functional success，也没有 full functional success，更没有 external-ready success。

本轮 route 是：

```text
route = R2-ExistingActionLDOBlocked
system_legal_controller_pass = 0
primary_blocker = existing_action_ldo_blocked
secondary_blocker = transfer_objective_mismatch
```

这说明：

```text
controller 没有选中；
selected runtime 没有测；
paired replay 没有打开；
short/full training 没有打开；
generated route 继续 stopped_no_new_objective。
```

但是 v9.7.4 不是空跑。它完成了三件关键排查：

```text
1. OldRank / ExactTransfer mismatch 被拆成 Intersection / OldOnly / ExactOnly；
2. Core77 / Expansion10 的 LDO 责任被拆清楚；
3. horizon transfer 真实 materialize 到 window 80，但仍没有成为可用 selector。
```

## 1.2 OldRank 与 ExactTransfer 的冲突不是小差异

v9.7.4 的核心表明：

```text
Intersection count = 18
Intersection precision = 0.9444444444444444

OldOnly count = 69
OldOnly precision = 0.855072463768116
OldOnly V LCB = 0.13450092269587752
OldOnly longrisk UCB = 0.0
OldOnly dataset count = 3
OldOnly template count = 69

ExactOnly count = 69
ExactOnly precision = 0.014492753623188406
ExactOnly V LCB = -0.3037141101011471
ExactOnly longrisk UCB = 0.0
```

这不是 “exact transfer 稍微不如 old rank”。这是两个选择规则在选完全不同的东西。

更重要的是，OldOnly 不是坏动作。OldOnly 里有 69 个动作，precision 约 0.855，V LCB 为正，longrisk 为 0，并且覆盖 3 个 dataset 与 69 个 template。也就是说：

$$
\boxed{
\text{OldRank 找到了一批 exact transfer 看不见、但实际很好的动作。}
}
$$

所以 exact transfer 当前不能作为辅助 gate。它会杀掉高价值动作。

## 1.3 Core77 本身就是 LDO blocker，Expansion10 不是主因

v9.7.4 继续验证 Core + Expansion：

```text
Core77 precision = 1.0
Core77 V LCB = 0.16402807605399972
Core77 LDO drop = 0.4415584415584416

C0+E1-87 accepted = 87
C0+E1-87 precision = 0.8850574712643678
C0+E1-87 V LCB = 0.14120469391749882
C0+E1-87 risk/bad/null/memory/offdiag UCB = 0
C0+E1-87 LDO drop = 0.39080459770114945

E1-old-rank-top10 单独 LDO drop = 0.0
```

这个结果很关键。之前我们以为可能是“补足的 10 个 expansion 动作”把 LDO 搞坏了。v9.7.4 说明不是这样。

真正的问题是：

$$
\boxed{
\text{Core77 自己已经 LDO fail。}
}
$$

所以 v9.7.5 不应该继续换 expansion10，而要先解释 Core77 为什么换 dataset 后掉。

## 1.4 Horizon transfer materializer 过了，但 score 没过

v9.7.4 的 horizon transfer 不是没跑。它真实执行了：

```text
subset actions = 198
windows = 1,5,20,80
check samples/action = 32
windowed action rows = 198
nan/inf = 0 / 0
materializer pass = 1
```

最好的是：

```text
best = WT80-LCB
TopK87 precision = 0.47126436781609193
V LCB = -0.10386482729963209
longrisk UCB = 0.0
LDO drop = 0.13793103448275862
weak/strong pass = 0 / 0
```

它比 WT20 更接近一些，但仍然不够。它能降低 longrisk 和 LDO，但 precision 与 V 不够。

结论不是：

```text
horizon transfer 没用。
```

更准确是：

```text
当前 WT80 仍像“安全但没价值”的 selector。
```

它没有解释 OldOnly 这批高价值动作。

## 1.5 OldRank 机制仍没有解释出来

v9.7.4 的 P5 结果：

```text
old rank precision = 0.8735632183908046
OldOnly good count = 59
P5_old_rank_mechanism_explained = 0
P5_generative_mechanism_ready = 0
```

这说明我们还不知道 OldRank 为什么能找到好动作。它可能用了某种合法统计组合，也可能仍是隐含 group / score shift / support pattern 的局部规律。

所以不能把 OldRank 直接写成 controller，也不能用它生成新动作。

---

# 2. 当前慢的根本原因

现在慢，不是因为没有动作，也不是因为 outcome table 不可信。当前慢的根本原因是：

$$
\boxed{
\text{我们能找到局部好动作，但还没有找到跨数据集稳定、训练当下可解释、可部署的选择规则。}
}
$$

从 v9.5.3 到 v9.7.4，项目一直在排除假成功：

```text
几何 target 很干净 -> 数量不够；
GCERT18 TopK 很强 -> 用了 outcome-derived fields；
payload0 pocket 很强 -> intervention 后证明不是因果机制；
accepted region 很强 -> 多轮发现 group / template / dataset stability 不足；
exact transfer 原理更简单 -> 实测会杀掉高价值 OldOnly 动作；
windowed transfer 更接近 horizon -> 实测仍 value-negative；
generated action 工程闭合 -> 多轮 value-negative / high-longrisk。
```

这些实验确实慢，但它们把很多错误路线排除了。现在的危险不是“没有实验”，而是继续在错误层面做小修。

---

# 3. 当前真正卡在哪里

## 3.1 卡点 A：Core77 的跨数据集稳定性

Core77 是目前最干净的一批动作：precision 1.0，V LCB 正，risk/bad/null/memory/offdiag 全干净。

但是 Core77 LDO drop = 0.4416。

这说明一个很重要的事实：

```text
Core77 在 pooled 数据里非常好；
但 leave-one-dataset 后，它的支持结构或 score 结构发生了明显变化。
```

这不能用 dataset-specific threshold 修，因为我们不是在数据集上打榜。

v9.7.5 要做的是诊断：

```text
Fashion-MNIST / KMNIST / MNIST 中，是哪个数据集导致 Core77 掉？
掉的是 precision？V？support？bad/null？还是 accepted count？
Core77 在不同 dataset 上的 score 分布是否尺度不同？
Core77 的动作是不是集中在某些 family / step / template / memory 状态？
```

诊断允许看 dataset；controller 不允许按 dataset 分支。

## 3.2 卡点 B：OldRank 的机制未知

OldRankTop87 与 OldOnly 都很强，但我们不知道为什么强。

这很危险。因为一个不知道原因的强 rank，可能是：

```text
1. 真正的训练几何规律；
2. 某种合法但脆弱的 group pocket；
3. 某些 split 上的 scale artifact；
4. 某些 template / family / step 的局部规律；
5. 与 outcome labels 高相关但不稳定的 proxy。
```

v9.7.4 已经证明 exact transfer 解释不了 OldRank。下一轮必须做 OldRank 机制解剖，而不是继续直接调 threshold。

## 3.3 卡点 C：transfer 与真实好动作目标失配

Exact transfer 测的是：

```text
这个动作是否让当前 check samples 的 CE 立刻下降。
```

但 functional update 真正要的是：

```text
h20 有收益；
h80 不坏；
h240 不出 longrisk；
bad/null 低；
memory/offdiag 不伤；
跨数据集稳定；
能打过 AdamWParallel / bestLR / NoOp / Random。
```

v9.7.2 已经证明 exact transfer 测量本身很准；v9.7.4 说明即使测得准，它也不是好动作的充分条件。

所以 transfer 不能再作为单一 selector。它最多作为一个特征，或者需要被重新定义为更接近 horizon / geometry 的响应向量。

## 3.4 卡点 D：generated route 没有新目标

generated route 当前状态：

```text
generated_route_status = stopped_no_new_objective
APGU/APGV/APGW/APGX run = 0
direct solved sandbox allowed = 0
```

这是对的。因为 APGH/APGL/APGT 多轮失败，v9.7.4 又没有给出新的可生成目标。

如果下一轮没有找到新的可生成目标，generated route 继续停。

---

# 4. v9.7.5 总体目标

v9.7.5 不是为了直接 system pass，而是为了回答两个第一性问题。

## 4.1 主线 A：existing-action 路线是否还有官方希望？

问题：

$$
\boxed{
\text{Core77 / OldOnly 这些已有好动作，能否被一个跨 dataset 稳定、训练当下合法的规则选出来？}
}
$$

如果可以，下一步进入 minimal controller 和 runtime。

如果不可以，existing-action route 进入 stop / hold 状态，不再调 threshold。

## 4.2 主线 B：transfer 思路是否应该重写？

问题：

$$
\boxed{
\text{one-step / windowed transfer 为什么无法解释 OldRank？有没有 response-vector 形式能解释 OldOnly？}
}
$$

这里不再把 transfer 当单一分数，而是把每个动作的 response 写成向量：

$$
R(a)=
(r_{1}, r_{5}, r_{20}, r_{80}, r_{memory}, r_{hardtail}, r_{oldfamily})
$$

然后问：

```text
OldOnly 的 response vector 到底长什么样？
ExactOnly 的 response vector 到底长什么样？
WT80 选中的动作为什么 V LCB 为负？
transfer 是否只看到了 safe / low-risk，却看不到 value？
```

## 4.3 主线 C：generated route 是否继续停止？

默认继续停止。只有当主线 B 找到一个新的可生成目标，才允许 direct solved sandbox。

---

# 5. 本轮硬约束

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
no fake rows
no proxy rows in official gates
manual forward / backward / update contract 继续有效
official controller 不使用 dataset_name 分支
official controller 不使用 validation/test/future outcome
official controller 不使用 outcome-derived fields
old outcome table 不进入 official path
APGU/APGV/APGW/APGX 不允许 blind run
```

允许：

```text
dataset 作为 leaveout / autopsy 轴；
dataset-blind score transport；
training split 内的 cross-fitting；
response-vector diagnostic；
Core77 / OldOnly / ExactOnly 分组诊断；
branch-horizon diagnostic；
Base-Acc Sentinel 健康检查，但不用于 controller。
```

---

# 6. 关键定义

## 6.1 Core77

Core77 指当前最干净的 existing-action core 集合。

它的性质：

```text
accepted = 77
precision = 1.0
V LCB = 0.16402807605399972
longrisk/bad/null/memory/offdiag = 0
coverage = 0.02677
LDO drop = 0.4415584415584416
```

它的问题是：数量不足且 LDO fail。

## 6.2 Expansion10

Expansion10 指把 Core77 补到 87 个动作所需的额外 10 个动作。

v9.7.4 证明 E1-old-rank-top10 单独 LDO drop = 0.0，因此 Expansion10 不是主因。

## 6.3 OldRank

OldRank 是目前最强 existing-action diagnostic rank。

它能选到：

```text
TopK87 precision ≈ 0.8736
V LCB ≈ 0.1411
longrisk UCB = 0
```

但 LDO drop 高，所以不能 official。

## 6.4 OldOnly

OldOnly 指 OldRank 选中但 ExactTransfer 没选中的动作。

v9.7.4 显示：

```text
count = 69
precision = 0.8551
V LCB = 0.1345
longrisk UCB = 0
dataset count = 3
template count = 69
```

这是当前最值得研究的集合。

## 6.5 ExactOnly

ExactOnly 指 ExactTransfer 选中但 OldRank 没选中的动作。

v9.7.4 显示：

```text
count = 69
precision = 0.0145
V LCB = -0.3037
longrisk UCB = 0
```

它说明 exact transfer 会选到很多“安全但没用”甚至 value-negative 的动作。

## 6.6 LDO

LDO 是 leave-dataset-out，意思是：

```text
用部分 dataset 校准规则；
留下一个 dataset 不参与校准；
看规则在留下的 dataset 上是否仍然好。
```

LDO 不是 dataset tuning。它是防止规则只在某个数据集好。

---

# 7. 实验阶段设计

---

## P0. Boundary reproduction 与 fail-fast guard

### 目标

确认 v9.7.4 使用的输入 artifacts 可复现，并且本轮不从错误状态开始。

### 假设

$$
H0:
\text{v9.7.4 boundary 可复现，且没有 forbidden field 进入 official path。}
$$

### 执行

读取 v9.7.4 artifacts：

```text
p0_boundary_reproduction_v9740.csv
p0_field_legality_audit_v9740.csv
p1_old_exact_transfer_mismatch_v9740.csv
p2_core_expansion_ldo_responsibility_v9740.csv
p4_horizon_transfer_materializer_v9740.csv
```

### 必须记录

```text
source_route_v9740
OldRankTop87_precision
ExactT4Top87_precision
Core77_count
Core77_LDO_drop
C0E1_precision
C0E1_LDO_drop
best_horizon_score_id
best_horizon_precision
best_horizon_V_LCB
generated_route_status
system_legal_controller_pass
field_green_count
field_yellow_count
field_red_count
dataset_name_used_in_controller
outcome_derived_field_used_in_controller
```

### 判断标准

P0 pass 必须满足：

```text
source_route_v9740 = R2-ExistingActionLDOBlocked
system_legal_controller_pass = 0
field_red_count = 0
outcome_derived_field_used_in_controller = 0
dataset_name_used_in_controller = 0
generated_route_status = stopped_no_new_objective
```

如果 P0 不过，停止全轮。

### 可视化

```text
boundary_status_table.md
field_legality_bar_chart.png
v9700_to_v9740_route_timeline.png
```

---

## P1. Core77 LDO failure autopsy

### 目标

查清 Core77 为什么自身 LDO fail。

不能继续假设：

```text
Core77 是干净的，所以只要补 10 个动作就能成。
```

v9.7.4 已经显示 Core77 自己 LDO drop = 0.4416。

### 假设

$$
H1a:
\text{Core77 的 LDO fail 来自某个 dataset 的 support / score / value 分布偏移。}
$$

$$
H1b:
\text{Core77 的 LDO fail 来自 action family / step bucket / template 的隐藏集中。}
$$

$$
H1c:
\text{Core77 的 LDO fail 不是随机样本波动，而是结构性问题。}
$$

### 执行

对 Core77 逐 dataset、seed、family、template、step bucket、memory/offdiag bucket 拆分。

不能用 dataset 生成新规则，只做诊断。

### 必须记录

每个 dataset：

```text
dataset_id
core77_action_count
core77_precision
core77_V_LCB
core77_longrisk_UCB
core77_bad_UCB
core77_null_UCB
core77_memory_UCB
core77_offdiag_UCB
core77_score_mean
core77_score_std
core77_support_count
core77_template_count
core77_family_count
```

每个 group：

```text
group_axis
group_id
base_share
core77_share
precision
V_LCB
longrisk_UCB
LDO_drop_if_removed
support_count
```

### 判断标准

H1a 成立，如果：

```text
某个 heldout dataset 的 precision drop >= 0.25
或 V_LCB 从正变负
或 support_count 明显低于其他 dataset
```

H1b 成立，如果：

```text
max family share > 0.35
或 max template share > 0.25
或 max step_bucket share > 0.35
```

H1c 成立，如果 bootstrap 后：

$$
P(LDO\_drop > 0.20) \ge 0.95.
$$

### 可视化

```text
core77_per_dataset_precision_bar.png
core77_per_dataset_V_LCB_bar.png
core77_score_distribution_by_dataset.png
core77_group_share_heatmap.png
core77_ldo_drop_waterfall.png
```

---

## P2. OldOnly mechanism anatomy

### 目标

OldOnly 是 v9.7.4 最重要的集合。它是 exact transfer 没选中、但实际非常好的动作。

本阶段要解释：

```text
OldOnly 为什么好？
它和 Core77 是同一种机制吗？
它跨 dataset / template 是否真稳定？
它的合法训练当下特征是什么？
```

### 假设

$$
H2a:
\text{OldOnly 的高质量来自 delayed value，不来自 one-step CE transfer。}
$$

$$
H2b:
\text{OldOnly 的好处与 memory/offdiag safe、hard-tail stability 或 AdamW compatibility 有关。}
$$

$$
H2c:
\text{OldOnly 不是 dataset-specific pocket。}
$$

### 执行

构造四个集合：

```text
Intersection = OldRank ∩ ExactTransfer
OldOnly = OldRank \ ExactTransfer
ExactOnly = ExactTransfer \ OldRank
Neither = neither selected
```

对四个集合做同样的 action feature / outcome / group 对比。

### 必须记录

每个集合：

```text
count
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
dataset_count
template_count
family_count
mean_payload_norm
mean_action_norm
mean_adamw_cosine
mean_exact_transfer_lcb
mean_WT1_lcb
mean_WT5_lcb
mean_WT20_lcb
mean_WT80_lcb
mean_memory_proxy
mean_offdiag_proxy
mean_hardtail_proxy
```

### 判断标准

OldOnly 可继续作为主研究对象，当且仅当：

```text
OldOnly precision >= 0.75
OldOnly V_LCB > 0
OldOnly longrisk_UCB <= 0.05
OldOnly dataset_count >= 3
OldOnly template_count >= 32
```

如果 OldOnly 也被拆成单 group / single family pocket，则 existing-action route 降级。

### 可视化

```text
old_exact_venn.png
set_quality_table.md
oldonly_vs_exactonly_feature_boxplots.png
oldonly_response_vector_heatmap.png
oldonly_dataset_template_scatter.png
```

---

## P3. Transfer mismatch deep autopsy

### 目标

解释为什么 exact transfer 会选出 ExactOnly 这种低质量动作，并排掉 OldOnly 高质量动作。

### 假设

$$
H3a:
\text{Exact transfer 主要捕捉 immediate CE decrease，无法捕捉 delayed horizon value。}
$$

$$
H3b:
\text{Exact transfer 选出的 ExactOnly 是 low-risk / low-movement / low-value 动作。}
$$

$$
H3c:
\text{OldOnly 的 value 来自几何改善，而不是 immediate CE transfer。}
$$

### 执行

对每个 action 记录 response vector：

$$
R(a)=
(
T_1(a),T_5(a),T_{20}(a),T_{80}(a),M(a),O(a),H(a)
)
$$

其中：

```text
T1/T5/T20/T80 = windowed transfer score
M = memory response
O = offdiag response
H = hard-tail response
```

不要直接用这些构造 controller，先做机制分析。

### 必须记录

```text
action_id
set_label = Intersection / OldOnly / ExactOnly / Neither
T1_lcb
T5_lcb
T20_lcb
T80_lcb
T80_minus_T1
transfer_persistence = min(T1,T5,T20,T80)
transfer_slope
memory_response_lcb
offdiag_response_lcb
hardtail_response_lcb
actual_V_integrated
actual_GradeAB
actual_longrisk
actual_bad
actual_null
```

### 判断标准

H3a 成立，如果：

```text
OldOnly 的 T1/T5 低，但 actual V 高；
OldOnly 的 T80 或 non-CE response 更接近 actual V；
ExactOnly 的 T1 高但 actual V 低。
```

H3b 成立，如果：

```text
ExactOnly 的 movement / V / GradeAB 显著低于 OldOnly；
ExactOnly longrisk 虽低但 V_LCB <= 0。
```

H3c 成立，如果：

```text
OldOnly 在 cover / memory / offdiag / hardtail 指标上明显优于 ExactOnly，
但 exact transfer 不高。
```

### 可视化

```text
response_vector_pca.png
T1_vs_actualV_scatter.png
T80_vs_actualV_scatter.png
OldOnly_ExactOnly_response_heatmap.png
transfer_persistence_by_set.png
```

---

## P4. Dataset-blind stability repair for existing-action route

### 目标

在不使用 dataset_name 分支、不按 dataset 调参的前提下，尝试把 Core77 / OldRank 的 LDO drop 降下来，同时保住质量。

### 假设

$$
H4:
\text{LDO fail 可以通过 dataset-blind score normalization / support balancing / response stability 处理，而不需要 dataset-specific threshold。}
$$

### 候选方法

方法必须 dataset-blind。允许训练阶段用 LDO split 选择通用规则，但在线时不能知道 dataset。

候选规则：

```text
S1. raw OldRank baseline
S2. score z-normalized by non-dataset groups，例如 step_bucket / family / template bucket
S3. score percentile normalized by action family / step bucket
S4. OldRank with memory/offdiag veto before ranking
S5. OldRank with response-stability veto
S6. OldRank + support-balanced topK，限制 family/template max share
S7. Core77 + expansion chosen by worst-split validation, frozen globally
```

### 必须记录

每个 rule：

```text
rule_id
accepted_count
coverage
GradeAB_precision
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
max_family_share
max_template_share
feature_cost_q90_ms
```

### 判断标准

Weak pass：

```text
accepted_count >= 87
precision >= 0.75
V_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
LDO_drop <= 0.20
LSO_drop <= 0.20
LTO_drop <= 0.10
```

Strong pass：

```text
accepted_count >= 87
precision >= 0.80
V_LCB > 0.05
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.10
LDO_drop <= 0.10
LSO_drop <= 0.10
LTO_drop <= 0.10
```

### 可视化

```text
rule_frontier_precision_vs_LDO.png
rule_frontier_V_vs_LDO.png
risk_veto_order_waterfall.png
accepted_region_group_share_heatmap.png
score_distribution_before_after_transport.png
```

---

## P5. Core77 expansion responsibility v2

### 目标

虽然 v9.7.4 已经证明 Expansion10 不是主因，但仍要判断是否存在一种 expansion 能“补数量但不扩大 LDO”。

### 假设

$$
H5:
\text{存在一种 expansion 只补 10 个动作，可以保持 Core77 的 clean risk，并改善或不恶化 LDO。}
$$

### 候选 expansion

```text
E0. no expansion, Core77 only
E1. old-rank top10 baseline
E2. OldOnly top10 by value-risk score
E3. response-vector stable top10
E4. memory/offdiag-safe top10
E5. worst-split-balanced top10
E6. template-balanced top10
E7. random clean-risk diagnostic top10
```

### 必须记录

```text
expansion_id
expansion_count
expansion_precision
expansion_V_LCB
expansion_LDO_drop
full87_precision
full87_V_LCB
full87_longrisk_UCB
full87_bad_UCB
full87_null_UCB
full87_memory_UCB
full87_offdiag_UCB
full87_LDO_drop
full87_LSO_drop
full87_LTO_drop
```

### 判断标准

Expansion pass：

```text
full87_precision >= 0.75
full87_V_LCB > 0
full87_longrisk_UCB <= 0.05
full87_bad_UCB <= 0.05
full87_null_UCB <= 0.15
full87_LDO_drop <= 0.10
```

如果所有 expansion 都失败，而 Core77 本身 LDO fail，则 existing-action route 需要进入 theory-reset，不再继续补 10 个动作。

### 可视化

```text
core77_vs_expansion10_lift_table.md
expansion10_LDO_waterfall.png
expansion10_action_scatter.png
```

---

## P6. OldRank mechanism ablation v2

### 目标

OldRank 目前好但不可解释。P6 要用消融方式找出 OldRank 的最小必要组成。

### 假设

$$
H6:
\text{OldRank 的有效性来自少数组合信号，而不是单个不可解释黑箱。}
$$

### 执行

把 OldRank 分解成可审计部分：

```text
value-related part
risk-veto part
memory/offdiag part
support/balance part
payload/action norm part
step/family/template support part
```

逐个删除或替换，观察 TopK87 变化。

### 必须记录

```text
ablation_id
removed_component
accepted_count
precision
V_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
LDO_drop
OldOnly_overlap
ExactOnly_overlap
Core77_overlap
```

### 判断标准

机制解释成立，如果存在一个 green-field-only minimal score 满足：

```text
precision >= 0.75
V_LCB > 0
longrisk_UCB <= 0.05
LDO_drop <= 0.15
OldRankTop87 overlap >= 0.50
OldOnly overlap >= 0.50
```

否则：

```text
old-rank mechanism not explained；
不能转 generated objective；
只能继续作为 diagnostic。
```

### 可视化

```text
oldrank_component_ablation_bar.png
overlap_matrix_oldrank_components.png
OldOnly_recovered_by_component_heatmap.png
```

---

## P7. Existing-action minimal controller boundary

### 目标

只有 P4/P5/P6 出现 weak pass，才允许构造 minimal controller。

### Controller 形式

不允许复杂模型。优先采用：

```text
1. 固定 topK rule；
2. 固定 threshold + hard veto；
3. two-stage: clean core first, expansion second；
4. group/template share caps, 但不能使用 dataset_name。
```

### 必须记录

```text
controller_id
calibration_rule
accepted_count_cal
accepted_count_heldout
coverage_heldout
precision_heldout
V_LCB_heldout
longrisk_UCB_heldout
bad_UCB_heldout
null_UCB_heldout
memory_UCB_heldout
offdiag_UCB_heldout
LDO_drop
LSO_drop
LTO_drop
feature_cost_q90_ms
```

### Pass 标准

```text
accepted_count_heldout >= 87
precision_heldout >= 0.75
V_LCB_heldout > 0
longrisk_UCB_heldout <= 0.05
bad_UCB_heldout <= 0.05
null_UCB_heldout <= 0.15
memory_UCB_heldout <= 0.10
offdiag_UCB_heldout <= 0.10
LDO_drop <= 0.10
LSO_drop <= 0.10
LTO_drop <= 0.10
feature_cost_q90_ms <= budget
```

如果不过，不打开 runtime。

---

## P8. Selected runtime boundary

### 目标

如果 P7 controller 过线，测 selected runtime。

### 必须记录

```text
selected_controller_id
online_step_count
active_step_count
accepted_action_count
feature_compute_q90_ms
score_compute_q90_ms
payload_lookup_q90_ms
payload_apply_q90_ms
kernel_launch_count
sync_count
step_ratio_q50
step_ratio_q90
step_ratio_q99
peak_memory_ratio
```

### Pass 标准

```text
step_ratio_q90 <= 1.50
peak_memory_ratio <= 1.05
no CPU offload
no fake/proxy rows
```

如果不过，不能 paired replay。

---

## P9. Generated route stop / reopen decision

### 目标

明确 generated route 是否继续停止。

### 默认状态

```text
generated_route_status = stopped_no_new_objective
APGU/APGV/APGW/APGX run = 0
```

### 允许重新打开的条件

只有当 P3/P6 找到新目标时，才允许 direct solved sandbox：

```text
新目标必须解释 OldOnly；
必须不是 one-step CE transfer；
必须能给出可优化的 action objective；
必须有 green-field commit-time terms；
必须先做 <= 32 actions smoke。
```

### Pass 标准

direct sandbox smoke 必须满足：

```text
generated_actions >= 32
branch_horizon_rows complete
precision >= 0.30
V_LCB > 0
longrisk_UCB <= 0.15
new_positive_rate >= 0.10
longrisk_created_rate <= 0.20
```

否则 generated route 继续停止。

---

## P10. Paired replay boundary

只有 P7 和 P8 都过，才打开。

### 必须记录

```text
RealFunctional
AdamWParallel
bestLR
NoOp
Random
ShuffledPayload
```

每个分支记录：

```text
accuracy
CE
NLL
ECE
CEp99
margin_p10
hard_tail_acc
V_integrated
runtime
memory
```

### Pass 标准

```text
RealFunctional beats AdamWParallel / bestLR / NoOp / Random
ShuffledPayload fails
LDO / LSO / LTO stable
runtime still pass
```

---

## P11. Short/full training boundary

只有 paired replay 过才打开。

继续作为 boundary，不在 v9.7.5 主动运行。

记录：

```text
final acc
best val CE
time to target
steps to target
ECE
NLL
CEp99
robustness
continual retention
forgetting
runtime
memory
```

---

# 8. 并行执行计划

为避免一轮只清一个 blocker，v9.7.5 分四个 batch 并行。

## Batch A：existing-action LDO root cause

```text
P0 boundary reproduction
P1 Core77 LDO autopsy
P2 OldOnly mechanism anatomy
P4 dataset-blind stability repair
P5 Core77 expansion responsibility
```

这批只读已有 AP0 / v9.7.4 artifacts，可并行。

## Batch B：transfer mismatch

```text
P3 transfer mismatch deep autopsy
P6 OldRank mechanism ablation
```

这批需要 horizon transfer / response vector artifacts，和 Batch A 可并行。

## Batch C：controller/runtime boundary

```text
P7 minimal controller
P8 selected runtime
```

只在 Batch A/B 出现 weak pass 后转 official；但 runtime component profiler 可以提前 dry-run。

## Batch D：generated route decision

```text
P9 generated route stop/reopen decision
```

默认停止，只有 P3/P6 给出新目标才运行小 smoke。

---

# 9. 本轮主要可视化清单

必须输出：

```text
1. old_exact_venn.png
2. set_quality_table.md
3. core77_per_dataset_precision_bar.png
4. core77_LDO_waterfall.png
5. oldonly_vs_exactonly_feature_boxplots.png
6. response_vector_pca.png
7. T1_T80_actualV_scatter.png
8. score_transport_frontier_precision_vs_LDO.png
9. expansion10_LDO_waterfall.png
10. oldrank_component_ablation_bar.png
11. accepted_region_group_share_heatmap.png
12. route_decision_flowchart.png
```

---

# 10. 最终 route decision

v9.7.5 的 route 只允许以下几类。

## R1：Existing-action controller pass

条件：

```text
P7 pass = 1
P8 pass = 1
```

下一步：paired replay。

## R2：Existing-action high-quality but LDO blocked

条件：

```text
precision / V / risk 过；
LDO 仍不过；
```

下一步：停止 threshold 小修，进入 dataset-invariant theory reset。

## R3：OldRank mechanism unexplained

条件：

```text
OldOnly 高质量；
但 P6 找不到 green-field minimal mechanism；
```

下一步：OldRank 保持 diagnostic，不转 controller。

## R4：Transfer objective mismatch confirmed

条件：

```text
ExactOnly / WTOnly 仍 low precision 或 value-negative；
无法解释 OldOnly；
```

下一步：transfer 不再作为 selector，仅作为 diagnostic。

## R5：Generated route remains stopped

条件：

```text
没有新可生成目标；
APG* blind route 继续停止。
```

## R6：Transfer-derived direct update smoke allowed

条件：

```text
P3/P6 找到新目标；
P9 smoke 允许。
```

但这不是 system pass。

---

# 11. 本轮停止规则

立即停止对应分支：

```text
1. 如果 Core77 LDO fail 被证明无法由 score transport / support balancing 降低到 <=0.20，停止 core-expansion 小修。
2. 如果 OldOnly 机制无法由 green fields 解释，停止把 OldRank 转 controller。
3. 如果 WT80/response vector 仍 precision <0.60 或 V_LCB <=0，停止 transfer selector 路线。
4. 如果没有新 objective，不允许 APGU/APGV/APGW/APGX 运行。
5. 如果 minimal controller 没过，不允许 selected runtime official measurement。
6. 如果 selected runtime 没过，不允许 paired replay。
```

---

# 12. 预期解释模板

v9.7.5 结束后必须能回答：

```text
1. Core77 的 LDO fail 是 dataset-level score shift、support shift，还是 hidden group concentration？
2. OldOnly 为什么好？它是否跨 dataset/template 稳定？
3. ExactTransfer / WT80 为什么选不到 OldOnly？
4. 是否存在 dataset-blind 修复 LDO 的简单规则？
5. 是否还应该继续 existing-action route？
6. 是否有新的 generated objective？
7. 是否可以打开 selected runtime / paired replay？
```

---

# 13. 结论

v9.7.4 之后，项目不能继续围绕 exact transfer threshold、WT80 threshold 或 expansion10 小修。当前最真实的问题是：

$$
\boxed{
\text{高质量动作已经存在，但我们还没有理解它们为什么好，也没有让选择规则跨数据集稳定。}
}
$$

v9.7.5 的任务是把这个问题拆清楚。它不是为了立刻跑 short/full training，而是为了决定：

```text
existing-action route 是否还有 official controller 希望；
transfer route 是否只是诊断工具；
generated route 是否继续停止；
下一步是否能进入 selected runtime / paired replay。
```

只有当 accepted region 同时满足数量、质量、风险、LDO/LSO/LTO 与 runtime，才允许继续向 functional success 推进。
