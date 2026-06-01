# DG-KAN v9.6.5 Template-Lineage Balanced Controller / Offdiag-Memory Causal Primitive 完整实验计划

> 本计划基于 v9.6.4 `Degenerate Pocket Audit / Lineage-Balanced Geometry Controller` 的真实结果制定。  
> v9.6.4 的 terminal route 是：
>
> ```text
> route = R2-LineageCollapsedAcceptedRegion
> base_candidate = LQ-t2-h256
> success_v9640_strict_purekan_functional = False
> success_v9640_full_functional = False
> success_v9640_external_ready = False
> ```
>
> v9.6.5 不再小修 `R4C`、`CERT12`、`APGL1` 或 payload bucket 阈值。  
> 本轮的核心目标是查清：**现在那批看起来好的 accepted actions，到底只是一个 candidate template 的复制扩张，还是能被扩展成跨模板、跨 group、跨 horizon 都稳定的 functional update rule。**

---

# 0. 本轮一句话目标

v9.6.5 的核心问题是：

$$
\boxed{
\text{能否把 v9.6.4 的局部好动作，变成 template-lineage balanced、memory/offdiag safe、runtime legal 的 accepted region？}
}
$$

现在不能再只问：

```text
TopK87 precision 高不高？
某个 rank score AUC 高不高？
某个 APGL primitive 是否 materialized？
```

必须问：

```text
1. accepted actions 是否来自多个真实 candidate templates？
2. 离开某个 template 后，precision / value / longrisk 是否仍然稳定？
3. memory/offdiag safe 是否真的是因果几何信号，而不是另一个隐藏 pocket？
4. generator 是否能生成新的、跨 template 的 value-positive / horizon-safe actions？
5. selected controller 是否能在 runtime 中以 step_ratio_q90 <= 1.50 执行？
```

---

# 1. 对 v9.6.4 的独立判断

## 1.1 v9.6.4 有进展，但不是能力成功

v9.6.4 的真实进展是把 v9.6.3 的“强 accepted region”进一步拆开了。

v9.6.3 已经确认 accepted-region scope 没有 duplicate / universe mismatch：accepted action、unique action、unique event 都是 `87`。v9.6.4 继续往下查，发现更细的 template 层完全塌缩：

```text
accepted action count = 87
accepted strict lineage count = 87
strict lineage entropy = 4.4659081186545775
max strict lineage share = 0.011494252873563218
candidate_template_unique_count = 1
candidate_template_max_share = 1.0
candidate_to_action_expansion_ratio = 87.0
candidate_id_collision_count = 86
```

这说明：

$$
\boxed{
\text{accepted region 表面上有 87 条不同 lineage，实际上只来自 1 个 candidate template。}
}
$$

这是关键发现。它解释了为什么 TopK / accepted region 看起来好，但不能 official：它不是一个跨 template 的规律，而是一个模板扩张出来的局部口袋。

## 1.2 Degenerate pocket audit 是有价值的

v9.6.4 把一些 base share = `1.0` 的轴识别为 degenerate / forbidden selector，例如：

```text
payload_norm_bucket
 action_norm_bucket
 candidate_origin
```

这些轴本来整个 universe 都一样，不能解释 accepted region 为什么好，也不能作为 selector。

但是剔除 degenerate 轴后，仍然看到一个 non-degenerate adjusted pocket：

```text
best adjusted axis = offdiag_bucket
best adjusted group = offdiag_fail_0
base share = 0.13803894297635605
adjusted drop = 0.8735632183908046
```

我的判断是：`offdiag_fail_0` 不能直接作为 selector，但它也不能被简单丢掉。它可能是我们一直在找的“好几何”线索的一部分：动作没有进入 offdiag risk，可能更不容易造成 h240 longrisk、old-family forgetting 或局部 cover 破坏。

## 1.3 memory/offdiag 不是空信号，但还不是 controller

P3 给了很重要的信号：

```text
matched_pair_count_memory = 511
matched_pair_count_offdiag = 397
matched_pair_count_joint = 397
GradeAB_lift_offdiag_safe = 0.19143576826196473
V_lift_offdiag_safe = 0.29043934585539005
longrisk_drop_offdiag_safe = 0.906693153540672
joint_safe_count = 397
joint_safe_GradeAB_precision = 0.19395465994962216
base_GradeAB_precision = 0.027468706536856746
leave_dataset_out_lift_drop = 0.06191769597280811
leave_stratum_out_lift_drop = 0.0044361708668599065
memory_offdiag_causality_pass = 0
```

这不是没信号。`longrisk_drop` 很强，GradeAB precision 从 base `0.0275` 到 joint safe `0.1940` 也有明显提升。

但它还不是 official，因为：

```text
1. joint GradeAB precision 仍太低；
2. memory/offdiag causality gate 没过；
3. accepted region 仍被 candidate-template collapse 污染；
4. 还没有证明 memory/offdiag safe 能跨 template 稳定选出好动作。
```

所以本轮不能说 “memory/offdiag 失败”。更准确是：

$$
\boxed{
\text{memory/offdiag 是强诊断线索，但还没有变成可部署规则。}
}
$$

## 1.4 P4-P6 说明 existing-action route 还没死，但需要 lineage balance

P4 的 target 有 weak signal：

```text
best target = T4.1-ValuePositiveNoLongRiskLineageBalanced
count = 97
coverage = 0.03372739916550765
V LCB = 0.16622185363738462
longrisk UCB = 0.0
best_max_lineage_share = 1.0
out_of_pocket_target_pass = 0
out_of_pocket_target_weak_pass = 1
```

这说明目标区域不是完全没有。`count = 97` 已经超过常见 coverage 下限对应的约 `87` 个 action。但是 `max_lineage_share = 1.0` 说明它仍然不够多样。

P5/P6 的 ranker 与 certificate 也类似：

```text
best ranker = R4C-value-longrisk-memory-veto
TopK87 precision = 0.5862068965517241
V LCB = 0.011510927737868978
longrisk UCB = 0.0
LDO drop = 0.2298850574712643
max group share = 1.0
max lineage share = 1.0

best certificate = CERT12-FrozenLineageBalancedTopK87
accepted = 87
coverage = 0.030250347705146036
precision = 0.5862068965517241
V LCB = 0.011510927737868978
longrisk UCB = 0.0
LFO drop = 0.06896551724137923
```

这些结果比随机强，但离 official 还远。关键不是 threshold，而是 accepted region 没有跨 template / group 多样性。

## 1.5 APGL 工程成功，科学失败

APGL 链路本身是闭合的：

```text
APGL generated actions = 512
branch-horizon rows = 12288 / 12288
unresolved exception = 0
quality audit pass = 1
```

但 outcome 继续失败：

```text
best APGL = APGL1-SourceReplayPreserver
GradeAB precision = 0.03125
V LCB = -0.5827183891309656
h240 longrisk UCB = 0.7869099970100811
new positive created rate = 0.015625
longrisk created rate = 0.671875
```

Per primitive 都是 value-negative / high-longrisk。APGL7 `LineageDiverseEnsembleIntersection` 也没有救回来，GradeAB precision 为 `0.0`，V LCB 为 `-0.5898`，h240 longrisk UCB 为 `0.8825`。

所以 APGL 不是没实现；它是**生成目标错误**。继续加 APGL9/APGL10 小变体不是正路。

---

# 2. 当前进度如何

## 2.1 已经完成的东西

```text
1. canonical outcome universe 已可信；
2. old table / no-transform replay / fake-proxy 问题已经被清理；
3. good action / good geometry 的评价表已经建立；
4. existing-action rank 有过强信号，但被证明集中在 group/template pocket；
5. payload0 pocket 已经被 intervention 证伪，不能再作为机制；
6. v9.6.4 进一步证明 accepted region 是 candidate-template collapsed；
7. memory/offdiag 方向有强 longrisk-drop 诊断信号；
8. APG/APGS/APGR/APGC/APGM/APGA/APGD/APGE/APGF/APGH/APGL 等生成器工程链路多次闭合；
9. 但所有 generated primitive 到目前都没有产生 value-positive / horizon-safe frontier。
```

## 2.2 还没完成的东西

```text
1. template-lineage balanced accepted region；
2. group-stable / template-stable ranker；
3. rank-safe certificate；
4. selected controller；
5. selected runtime；
6. official leaveout；
7. official paired replay；
8. short/full functional training；
9. sample efficiency / robustness / continual / anti-forgetting validation。
```

现在最准确的状态是：

$$
\boxed{
\text{我们不是找不到好动作；我们找不到足够多、足够多样、训练当下可识别的好动作。}
}
$$

---

# 3. 为什么会感觉很慢

这个感觉是合理的。

最近几轮一直在做：

```text
发现一个强信号
-> 查它是否合法
-> 查它是不是 outcome-derived
-> 查它是不是 payload pocket
-> 查它是不是 scope bug
-> 查它是不是 group pocket
-> 查它是不是 candidate-template collapse
-> 最后大多被降级成 diagnostic
```

这会让人感觉一直没有进入主线。但这是必要的，因为 DG-KAN 的终极目标不是做一个在某个数据集上看起来高分的 hack，而是做一个能替代 MLP 的 functional training system。只要某个信号来自：

```text
dataset branch；
outcome-derived field；
payload bucket shortcut；
single candidate template；
lineage collapse；
old table；
proxy row；
fake runtime；
```

它就不能 official。

真正需要提速的地方是实验执行方式。v9.6.5 不能再 “一个大 runner 跑完才发现另一个 collapse”。它要把以下内容并行跑：

```text
lineage hierarchy audit；
template leaveout；
memory/offdiag causal intervention；
group/template balanced ranker；
rank-safe certificate；
APGL/APGH damage autopsy；
new value-preserving generator smoke；
runtime preflight。
```

---

# 4. 当前真正卡在哪里

## 4.1 卡在 candidate-template lineage collapse

v9.6.4 最大 blocker 是：

```text
candidate_template_unique_count = 1
candidate_to_action_expansion_ratio = 87.0
candidate_id_collision_count = 86
```

这说明当前 accepted region 不是从 87 个独立来源中选出来的，而是一个 candidate template 扩张出 87 个动作。

这会导致三个问题：

```text
1. 泛化风险：换 template 后可能完全失效；
2. 统计风险：precision 看起来高，但 support 实际只有一个模板；
3. 机制风险：我们不知道是 template 本身好，还是 ranker 真懂好几何。
```

所以 v9.6.5 的第一原则是：

$$
\boxed{
\text{任何 accepted region 都必须报告 candidate-template diversity。}
}
$$

## 4.2 卡在 degenerate axis 与真实几何信号的区分

`payload_norm_bucket` 这种 base share = 1.0 的轴不能解释问题。v9.6.4 已经把它识别出来，这是好事。

但是 `offdiag_bucket=offdiag_fail_0` 的 base share 只有 `0.1380`，并且 adjusted drop 很强。这可能是真实几何信号，也可能是另一个未解释的 pocket。

下一步必须做 causal intervention：

```text
同一 candidate template；
相近 payload norm；
相近 step / family / source；
只改变 memory/offdiag safety；
看 V、GradeAB、longrisk 是否改变。
```

## 4.3 卡在 TopK 到 controller 的转换

TopK 看起来好，但 frozen accepted region 一直难过。原因是 TopK 只是在一个排序表里挑前 N，而 controller 要在真实训练流里根据 frozen rule 接受动作。

v9.6.4 的 `CERT12` 已经说明：

```text
accepted = 87
coverage = 0.03025
precision = 0.5862
V LCB = 0.0115
longrisk UCB = 0.0
max group share = 1.0
max lineage share = 1.0
```

precision 不够，lineage 也塌缩。不能再只调 accepted count。

## 4.4 卡在 generator 仍然 value-negative / high-longrisk

APGL 是最新一轮 generator，但结果仍是：

```text
best GradeAB precision = 0.03125
best V LCB = -0.5827
best h240 longrisk UCB = 0.7869
```

生成器反复失败，说明当前 generator 做的是：

```text
生成合法 payload
```

而不是：

```text
生成跨 template、保 value direction、压 longrisk、保 memory/offdiag 的好几何动作。
```

---

# 5. 是否还在正确道路上

## 5.1 高层路线仍然正确

仍然正确，因为项目仍然守住了几条关键线：

```text
1. 不按数据集调参；
2. 不用 teacher / distillation / loss modification；
3. 不用 fake / proxy / CPU offload；
4. 不把 diagnostic promoted 成 official；
5. 不在 controller 未过时打开 paired replay / short-full；
6. functional update 仍作为 update rule，而不是 loss trick。
```

这很重要。现在慢，是因为我们在排除假成功，而不是因为路线本身没有意义。

## 5.2 具体路线必须再 pivot

不要继续做：

```text
调 R4C threshold；
调 CERT12 accepted count；
调 APGL1 / APGL7 小参数；
继续新增 APGL 小变体；
继续用 strict_lineage entropy 安慰自己；
继续把 candidate-template collapse 下的 87 actions 当作独立 support。
```

应该转成：

```text
candidate-template lineage hierarchy；
template leaveout；
memory/offdiag causal intervention；
template-balanced ranker；
rank-safe certificate；
value-preserving template-diverse generator。
```

---

# 6. v9.6.5 总体目标

v9.6.5 的总体目标是：

$$
\boxed{
\text{将 accepted region 从单模板扩张，推进到多模板、多 group、memory/offdiag-safe 的 functional update rule。}
}
$$

本轮不追求直接 full functional success。最低有效推进是回答下面四个问题：

```text
Q1. v9.6.4 的 87 accepted actions 为什么只有 1 个 candidate template？
Q2. memory/offdiag safe 是否是跨 template 成立的因果几何信号？
Q3. 是否存在 template-balanced ranker / certificate 能同时满足 value、longrisk、precision、coverage？
Q4. 如果 existing-action route 仍不行，能否生成 template-diverse、value-preserving、memory/offdiag-safe 的新动作？
```

---

# 7. 成功标准总览

## 7.1 Existing-action controller official gate

一个 existing-action controller 必须满足：

$$
Coverage \ge 0.03
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

并且必须满足 diversity gate：

$$
CandidateTemplateCount \ge 16
$$

$$
MaxCandidateTemplateShare \le 0.25
$$

$$
MaxGroupShare \le 0.35
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

其中 `LTO` 表示 leave-template-out。

## 7.2 Generated-action weak gate

generated primitive 的 weak gate：

$$
Precision_{GradeAB} \ge 0.30
$$

$$
LCB(V_{integrated}) > 0
$$

$$
UCB(LongRisk_{240}) \le 0.15
$$

$$
NewPositiveCreatedRate \ge 0.10
$$

$$
MaxTemplateShare \le 0.35
$$

## 7.3 Generated-action official candidate gate

如果要把 generated primitive 推为 official candidate，必须满足：

$$
Precision_{GradeAB} \ge 0.75
$$

$$
Coverage \ge 0.03
$$

$$
LCB(V_{integrated}) > 0
$$

$$
UCB(LongRisk_{240}) \le 0.05
$$

$$
CandidateTemplateCount \ge 16
$$

$$
MaxTemplateShare \le 0.25
$$

---

# 8. P0：v9.6.4 boundary reproduction

## 目标

确认 v9.6.5 没有跳过 v9.6.4 的失败边界。

## 假设

H0：v9.6.4 的 primary blocker 是真实的 candidate-template collapse，而不是 run artifact。

## 必须记录

```text
source_route_v9640
system_legal_controller_pass_v9640
accepted_action_count_v9640
accepted_unique_action_count_v9640
candidate_template_unique_count_v9640
candidate_to_action_expansion_ratio_v9640
best_adjusted_axis_v9640
memory_offdiag_causality_pass_v9640
best_ranker_v9640
best_ranker_TopK87_precision_v9640
best_apgl_primitive_v9640
best_apgl_V_LCB_v9640
best_apgl_h240_longrisk_UCB_v9640
no_fake_v9640
no_proxy_v9640
```

## 通过标准

```text
P0 pass iff:
  source_route_v9640 = R2-LineageCollapsedAcceptedRegion
  candidate_template_unique_count_v9640 = 1
  system_legal_controller_pass_v9640 = 0
  no_fake_v9640 = 1
  no_proxy_v9640 = 1
```

## 可视化

```text
fig_p0_route_timeline_v9300_to_v9640.svg
fig_p0_blocker_transition_sankey.svg
```

---

# 9. P1：Candidate-template lineage hierarchy rebuild

## 目标

重新定义 lineage 层级，避免 strict lineage 看起来分散但 candidate-template 实际塌缩。

## 背景

v9.6.4 显示：

```text
strict lineage count = 87
candidate template count = 1
```

说明当前 strict lineage 太细，不能代表独立 support。

## 假设

H1：accepted region 的塌缩来自 candidate-template 层，而不是 action_id / event_id / strict_lineage 层。

## 新 lineage 层级

每个 action 必须计算：

```text
action_id
candidate_id
event_id
strict_lineage_id
source_payload_hash
candidate_template_id
generator_template_id
source_event_family
source_step_bucket
source_dataset
source_stratum
memory_bucket
offdiag_bucket
cover_bucket
value_direction_bucket
payload_norm_bucket
action_norm_bucket
```

其中：

```text
candidate_template_id = hash(
  candidate_family,
  candidate_generation_rule,
  source_payload_role,
  action_shape_signature,
  source_event_family,
  functional_update_recipe,
  excluding action_id / event_id / random suffix
)
```

## 必须记录

```text
unique_action_count
unique_event_count
unique_candidate_id_count
unique_strict_lineage_count
unique_source_payload_count
unique_candidate_template_count
unique_generator_template_count
entropy_action
entropy_candidate_template
max_action_share
max_candidate_template_share
candidate_to_action_expansion_ratio
template_to_action_expansion_ratio
candidate_id_collision_count
template_collision_count
```

## 判断标准

```text
lineage_hierarchy_pass = 1 iff:
  candidate_template_id missing count = 0
  candidate_template collision trace complete = 1
  every accepted action has candidate_template_id
  template_to_action_expansion_ratio is measured
```

如果：

$$
CandidateTemplateCount < 16
$$

或者：

$$
MaxCandidateTemplateShare > 0.25
$$

则任何 existing-action accepted region 均不能 official。

## 可视化

```text
fig_p1_lineage_hierarchy_sunburst.svg
fig_p1_candidate_template_expansion_hist.svg
fig_p1_action_to_template_sankey.svg
fig_p1_entropy_by_lineage_level.svg
fig_p1_template_collision_table.svg
```

---

# 10. P2：Accepted-region leave-template-out audit

## 目标

在 v9.6.4 的 accepted region 和所有候选 ranker 上执行 leave-template-out，不再只做 leave-dataset-out / leave-stratum-out。

## 假设

H2：如果 accepted region 只是一个 template pocket，则 leave-template-out 后 precision / value 会大幅下降。

## 实验对象

```text
v9.6.3 CERT11 accepted region
v9.6.4 CERT12 accepted region
R4C-value-longrisk-memory-veto
T4.1-ValuePositiveNoLongRiskLineageBalanced
memory/offdiag-safe rule variants
```

## 分割方式

```text
Split A: leave-candidate-template-out
Split B: leave-source-payload-out
Split C: leave-generator-template-out
Split D: leave-event-family-out
Split E: leave-memory-bucket-out
Split F: leave-offdiag-bucket-out
```

## 必须记录

```text
accepted_count
coverage
GradeAB_precision
V_integrated_LCB
h240_longrisk_UCB
bad_UCB
null_UCB
candidate_template_count
max_candidate_template_share
leave_template_out_precision_drop
leave_template_out_V_drop
leave_template_out_longrisk_increase
leave_payload_out_drop
leave_generator_template_out_drop
leave_family_out_drop
```

## 通过标准

```text
accepted_region_template_stability_pass = 1 iff:
  accepted_count >= 87
  coverage >= 0.03
  GradeAB_precision >= 0.75
  V_integrated_LCB > 0
  h240_longrisk_UCB <= 0.05
  candidate_template_count >= 16
  max_candidate_template_share <= 0.25
  leave_template_out_precision_drop <= 0.10
  leave_template_out_V_drop <= 0.05
  leave_template_out_longrisk_increase <= 0.05
```

## 可视化

```text
fig_p2_lto_precision_drop_by_ranker.svg
fig_p2_template_fold_heatmap.svg
fig_p2_accepted_template_share_bar.svg
fig_p2_value_longrisk_by_template_fold.svg
```

---

# 11. P3：Memory/offdiag causal intervention panel

## 目标

判断 memory/offdiag safe 是否真的能降低 longrisk 并提高 GradeAB，而不是另一个 hidden pocket。

## 假设

H3：在匹配 candidate_template、payload norm、source family、step bucket 后，memory/offdiag safe 仍能提高 V、提高 GradeAB、降低 longrisk。

## 实验设计

构造 matched pairs：

```text
same candidate_template_id
same source_event_family
same source_step_bucket
same payload_norm_bucket if non-degenerate
same action_norm_bucket if non-degenerate
same source dataset/seed only for diagnostic, not selector
matched on candidate source and action shape
```

对每个 pair 比较：

```text
memory_safe vs memory_fail
offdiag_safe vs offdiag_fail
joint_safe vs joint_fail
```

如果天然 pairs 不够，则生成 intervention clones：

```text
clone source payload
apply memory projection only
apply offdiag projection only
apply joint memory/offdiag projection
keep value direction cosine within tolerance
replay h20/h80/h240 branches
```

## 必须记录

```text
matched_pair_count_memory
matched_pair_count_offdiag
matched_pair_count_joint
intervention_clone_count
value_direction_cosine_preserved_rate
GradeAB_lift_memory_safe
GradeAB_lift_offdiag_safe
GradeAB_lift_joint_safe
V_lift_memory_safe
V_lift_offdiag_safe
V_lift_joint_safe
longrisk_drop_memory_safe
longrisk_drop_offdiag_safe
longrisk_drop_joint_safe
LDO_lift_drop
LSO_lift_drop
LTO_lift_drop
```

## 通过标准

```text
memory_offdiag_causal_pass = 1 iff:
  matched_pair_count_joint >= 512 or intervention_clone_count >= 512
  GradeAB_lift_joint_safe_LCB >= 0.15
  V_lift_joint_safe_LCB >= 0.05
  longrisk_drop_joint_safe_UCB >= 0.50
  value_direction_cosine_preserved_rate >= 0.95
  LDO_lift_drop <= 0.10
  LSO_lift_drop <= 0.10
  LTO_lift_drop <= 0.10
```

## 可视化

```text
fig_p3_memory_offdiag_matched_lift.svg
fig_p3_intervention_lift_waterfall.svg
fig_p3_value_direction_cosine_vs_lift.svg
fig_p3_longrisk_drop_by_template.svg
fig_p3_joint_safe_scatter_V_vs_longrisk.svg
```

---

# 12. P4：Template-balanced target map

## 目标

重新扫描 target，不只看 action count / coverage，而是加入 candidate-template diversity。

## 假设

H4：存在一个 target，既满足 value / longrisk，又有足够 template support。

## 候选 target

```text
T4.1 existing: ValuePositiveNoLongRiskLineageBalanced
T4.2 TemplateBalancedValuePositiveNoLongRisk
T4.3 MemoryOffdiagSafeValuePositive
T4.4 StrictTemplateBalancedGradeAB
T4.5 SoftTemplateBalancedGradeABC
T4.6 ValuePositiveNoLongRiskNoBadNoNull
T4.7 SignalChannelMemorySafeNoOffdiagRisk
T4.8 CoverStableMemorySafeValuePositive
```

## 必须记录

```text
target_id
action_count
coverage
GradeAB_precision
V_integrated_LCB
h20_V_LCB
h80_V_LCB
h240_V_LCB
h240_longrisk_UCB
bad_UCB
null_UCB
candidate_template_count
max_candidate_template_share
generator_template_count
max_generator_template_share
LDO_drop
LSO_drop
LTO_drop
support_by_template
support_by_dataset
support_by_stratum
support_by_family
```

## 通过标准

```text
template_balanced_target_pass = 1 iff:
  action_count >= 87
  coverage >= 0.03
  V_integrated_LCB > 0
  h240_longrisk_UCB <= 0.05
  bad_UCB <= 0.05
  null_UCB <= 0.15
  candidate_template_count >= 16
  max_candidate_template_share <= 0.25
  LDO_drop <= 0.10
  LSO_drop <= 0.10
  LTO_drop <= 0.10
```

## 可视化

```text
fig_p4_target_density_vs_template_count.svg
fig_p4_target_value_risk_frontier.svg
fig_p4_support_by_template_heatmap.svg
fig_p4_target_lattice_parallel_coordinates.svg
```

---

# 13. P5：Template-deconfounded legal ranker

## 目标

构造一个不依赖 degenerate axis、不依赖 outcome-derived fields、不塌缩到单 template 的 legal ranker。

## 特征约束

禁止：

```text
outcome labels
V_integrated
risk_score derived from future outcome
payload_norm_bucket as selector
candidate_template_id as direct selector
action_id / event_id / row order
validation/test fields
dataset_name branch
```

允许：

```text
commit-time value direction features
AdamW compatibility
memory/offdiag safety features
cover stability features
population-risk / SNR features
cost features
template quota as regularization, not direct positive selector
```

## Ranker 结构

使用两阶段 score，而不是单一混合分数：

$$
Score(a)=ValueScore(a)-\lambda_R RiskVeto(a)-\lambda_M MemoryVeto(a)-\lambda_O OffdiagVeto(a)-\lambda_C Cost(a)-\lambda_T TemplatePenalty(a)
$$

其中：

```text
ValueScore: 预测 V_integrated / GradeAB 的正向信号；
RiskVeto: 预测 h240 longrisk；
MemoryVeto: 旧 family / old stratum forgetting 风险；
OffdiagVeto: population-risk offdiag failure；
Cost: feature compute + payload apply cost；
TemplatePenalty: 防止单 template 垄断。
```

## 候选 ranker

```text
R5A-template-residualized-linear
R5B-pairwise-within-template
R5C-group-DRO-ranker
R5D-value-rank-risk-memory-offdiag-veto
R5E-template-quota-constrained-ranker
R5F-leave-template-adversarial-ranker
R5G-monotone-minimal-5-feature-ranker
R5H-no-template-regularization-negative-control
```

## 必须记录

```text
ranker_id
feature_count
red_field_count
yellow_field_count
TopK64_precision
TopK87_precision
accepted_count_at_coverage_003
coverage
V_integrated_LCB
h240_longrisk_UCB
bad_UCB
null_UCB
candidate_template_count
max_candidate_template_share
max_group_share
LDO_drop
LSO_drop
LTO_drop
feature_compute_ms_q90
```

## 通过标准

```text
template_deconfounded_ranker_pass = 1 iff:
  red_field_count = 0
  feature_count <= 12
  accepted_count >= 87
  coverage >= 0.03
  GradeAB_precision >= 0.75
  V_integrated_LCB > 0
  h240_longrisk_UCB <= 0.05
  candidate_template_count >= 16
  max_candidate_template_share <= 0.25
  LDO_drop <= 0.10
  LSO_drop <= 0.10
  LTO_drop <= 0.10
  feature_compute_ms_q90 <= budget
```

## 可视化

```text
fig_p5_rank_score_hist_by_template.svg
fig_p5_ranker_ablation_waterfall.svg
fig_p5_value_vs_risk_score_scatter.svg
fig_p5_template_quota_map.svg
fig_p5_lto_drop_by_ranker.svg
```

---

# 14. P6：Rank-safe certificate v13

## 目标

把 P5 的 ranker 冻结成可部署 accepted rule，不再只看 TopK diagnostic。

## Controller 形式

$$
Accept(a)=1
$$

当且仅当：

$$
ValueScore(a) \ge \tau_V
$$

$$
RiskVeto(a) \le \tau_R
$$

$$
MemoryVeto(a) \le \tau_M
$$

$$
OffdiagVeto(a) \le \tau_O
$$

$$
Cost(a) \le \tau_C
$$

并满足 template quota：

$$
AcceptedPerTemplate \le Q_T
$$

## 必须记录

```text
certificate_id
ranker_id
thresholds
quota_QT
accepted_count
coverage
GradeAB_precision
V_integrated_LCB
h240_longrisk_UCB
bad_UCB
null_UCB
candidate_template_count
max_candidate_template_share
LDO_drop
LSO_drop
LTO_drop
calibration_split_metrics
heldout_split_metrics
cal_to_heldout_drift
```

## 通过标准

```text
rank_safe_certificate_v13_pass = 1 iff:
  accepted_count >= 87
  coverage >= 0.03
  GradeAB_precision >= 0.75
  V_integrated_LCB > 0
  h240_longrisk_UCB <= 0.05
  bad_UCB <= 0.05
  null_UCB <= 0.15
  candidate_template_count >= 16
  max_candidate_template_share <= 0.25
  LDO_drop <= 0.10
  LSO_drop <= 0.10
  LTO_drop <= 0.10
```

## 可视化

```text
fig_p6_calibration_vs_heldout_drift.svg
fig_p6_accepted_region_by_template.svg
fig_p6_threshold_sensitivity_grid.svg
fig_p6_precision_value_longrisk_frontier.svg
```

---

# 15. P7：Existing-action minimal controller

## 目标

只有 P6 pass 后，才打开 existing-action controller。

## 必须记录

```text
controller_id
certificate_id
accepted_count
coverage
GradeAB_precision
V_integrated_LCB
h240_longrisk_UCB
bad_UCB
null_UCB
candidate_template_count
max_candidate_template_share
no_dataset_branch
no_future_outcome_feature
no_outcome_at_commit
no_old_table
no_proxy
no_fake
```

## 通过标准

```text
existing_action_controller_pass = 1 iff:
  P6 pass
  all legality audits pass
  no diagnostic-derived field used
  no dataset-specific branch used
```

---

# 16. P8：Selected-controller runtime

## 目标

测量 selected controller 的真实 online runtime，不拿 offline materializer 或 diagnostic estimate 代替。

## 必须记录

```text
controller_id
step_count
active_step_count
accepted_step_count
zero_candidate_step_count
feature_compute_ms_q50/q90/q99
certificate_compute_ms_q50/q90/q99
payload_lookup_ms_q90
payload_apply_ms_q90
extra_kernel_count
extra_sync_count
step_ratio_q50/q90/q99
memory_ratio
runtime_path_materialized
materializer_in_timed_path
```

## 通过标准

```text
selected_runtime_pass = 1 iff:
  step_ratio_q90 <= 1.50
  memory_ratio <= 1.05
  materializer_in_timed_path = 0
  diagnostic_derived_from_measured_components = 0
  selected_controller_id matches P7 controller_id
```

## 可视化

```text
fig_p8_step_ratio_hist.svg
fig_p8_runtime_component_stack.svg
fig_p8_active_vs_zero_step_trace.svg
fig_p8_payload_apply_time_by_template.svg
```

---

# 17. P9：APGL/APGH damage decomposition v2

## 目标

解释为什么 APGH/APGL 生成动作仍 value-negative / high-longrisk，特别是生成过程中 value direction 是否丢失、longrisk 是否被制造、memory/offdiag 是否被破坏。

## 必须记录

```text
source_action_id
generated_action_id
primitive_id
source_candidate_template_id
generated_candidate_template_id
value_direction_cosine
payload_norm_delta
action_norm_delta
memory_fail_delta
offdiag_fail_delta
cover_entropy_delta
V_integrated_delta
longrisk_delta
new_positive_created
source_positive_preserved
longrisk_created
dominant_damage_mode
```

## 通过标准

P9 是 diagnostic，不直接 official。它必须产出可行动结论：

```text
if value_direction_lost fraction >= 0.50:
  generator reset must preserve value direction.
if longrisk_created fraction >= 0.50:
  generator reset must include hard longrisk veto.
if offdiag_fail_delta high:
  generator reset must include offdiag nullspace constraint.
if memory_fail_delta high:
  generator reset must include old-family / old-stratum memory constraint.
```

## 可视化

```text
fig_p9_source_to_generated_damage_matrix.svg
fig_p9_value_direction_cosine_hist.svg
fig_p9_longrisk_created_by_primitive.svg
fig_p9_memory_offdiag_delta_by_primitive.svg
fig_p9_template_diversity_source_vs_generated.svg
```

---

# 18. P10：APGT template-diverse value-preserving primitive

## 目标

如果 existing-action route 仍未过，则并行实现新的 generated primitive。APGT 不再只做合法 payload，而要同时满足：

```text
template diversity；
value direction preservation；
memory/offdiag safety；
cover stability；
longrisk veto；
low runtime cost。
```

## APGT primitive family

```text
APGT1-NoTransformTemplateDiverseReplay
APGT2-ValueDirectionPreservingTemplatePerturb
APGT3-MemoryOffdiagNullspaceValuePreserver
APGT4-TemplateMixtureAnchor
APGT5-SignalChannelSNRPopulationRiskDelta
APGT6-CoverMemoryBoundarySymmetricDelta
APGT7-ConservativeSourceReplayPlusSmallDelta
APGT8-ShuffledPayloadNegativeControl
```

## 每个 primitive 必须写入

```text
primitive_id
generated_action_id
source_action_id
source_candidate_template_id
generated_candidate_template_id
payload_hash
certificate_hash
action_apply_error_linf
action_apply_error_relative
value_direction_cosine
memory_veto_score
offdiag_veto_score
cover_stability_score
longrisk_veto_score
cost_estimate
```

## 通过标准：implementation

```text
apgt_implementation_pass = 1 iff:
  generated_action_count = expected
  payload_hash_missing_count = 0
  certificate_hash_missing_count = 0
  action_apply_linf_max <= 1e-6
  negative_control_generated = 1
  candidate_template_count >= planned_min
```

---

# 19. P11：APGT branch-horizon smoke outcome

## 目标

真实 materialize APGT outcomes，不使用 proxy / fake rows。

## 必须记录

```text
primitive_id
generated_action_count
branch_horizon_rows_expected
branch_horizon_rows_actual
branch_completion_rate
horizon_completion_rate
secondary_delta_completion_rate
unresolved_exception_count
duplicate_row_count
label_exclusivity_violation_count
rows_per_sec
wallclock_sec
```

## 通过标准

```text
apgt_branch_horizon_pass = 1 iff:
  actual_rows = expected_rows
  unresolved_exception_count = 0
  duplicate_row_count = 0
  label_exclusivity_violation_count = 0
  branch_completion_rate = 1.0
  horizon_completion_rate = 1.0
```

---

# 20. P12：APGT outcome / geometry pass

## 目标

判断 APGT 是否真正生成 value-positive / horizon-safe / template-diverse frontier。

## 必须记录

```text
primitive_id
GradeAB_precision
GoodGeometry_A_precision
GoodGeometry_B_precision
V_integrated_LCB
h20_V_LCB
h80_V_LCB
h240_V_LCB
h240_longrisk_UCB
bad_UCB
null_UCB
memory_fail_UCB
offdiag_fail_UCB
cover_collapse_rate
candidate_template_count
max_candidate_template_share
new_positive_created_rate
source_positive_preserved_rate
longrisk_created_rate
source_to_generated_damage_LCB
```

## Weak pass

```text
apgt_weak_pass = 1 iff:
  GradeAB_precision >= 0.30
  V_integrated_LCB > 0
  h240_longrisk_UCB <= 0.15
  new_positive_created_rate >= 0.10
  candidate_template_count >= 16
  max_candidate_template_share <= 0.35
```

## Official candidate pass

```text
apgt_official_candidate_pass = 1 iff:
  GradeAB_precision >= 0.75
  coverage >= 0.03
  V_integrated_LCB > 0
  h240_longrisk_UCB <= 0.05
  bad_UCB <= 0.05
  null_UCB <= 0.15
  candidate_template_count >= 16
  max_candidate_template_share <= 0.25
```

## 可视化

```text
fig_p12_apgt_gradeab_by_primitive.svg
fig_p12_apgt_value_longrisk_frontier.svg
fig_p12_apgt_template_diversity.svg
fig_p12_apgt_damage_waterfall.svg
```

---

# 21. P13：APGT certificate / controller

## 目标

如果 APGT weak 或 official candidate pass，则尝试生成 APGT controller。

## 必须记录

```text
certificate_id
primitive_id
accepted_count
coverage
GradeAB_precision
V_integrated_LCB
h240_longrisk_UCB
bad_UCB
null_UCB
candidate_template_count
max_candidate_template_share
LDO_drop
LSO_drop
LTO_drop
feature_compute_ms_q90
payload_apply_ms_q90
```

## 通过标准

```text
apgt_controller_pass = 1 iff:
  apgt_official_candidate_pass = 1
  accepted_count >= 87
  coverage >= 0.03
  GradeAB_precision >= 0.75
  V_integrated_LCB > 0
  h240_longrisk_UCB <= 0.05
  candidate_template_count >= 16
  max_candidate_template_share <= 0.25
  LDO/LSO/LTO drops <= 0.10
```

---

# 22. P14：System controller route decision

## 目标

统一判断 existing-action route 和 generated-action route。

## Route rules

```text
R0-BoundaryReproductionFailed:
  P0 fail

R1-LineageHierarchyIncomplete:
  P1 fail

R2-TemplateCollapsedAcceptedRegion:
  P2 fail due to candidate_template_count < 16 or max share > 0.25

R3-MemoryOffdiagCausalityUnresolved:
  P3 fail and no alternative stable signal

R4-NoTemplateBalancedTarget:
  P4 fail

R5-LegalRankTemplateUnstable:
  P5/P6 fail

R6-ExistingActionControllerPassRuntimePending:
  P7 pass but P8 not measured

R7-ExistingActionSystemPass:
  P7 and P8 pass

R8-APGTGeneratedFrontierFail:
  P10-P12 implementation pass but outcome fail

R9-APGTSystemPass:
  APGT controller and runtime pass

R10-ReadyForPairedReplay:
  system controller pass and legality audit pass
```

---

# 23. P15：Official leaveout / paired replay boundary

只有以下条件同时满足，才打开：

```text
P7/P8 pass for existing-action route
or P13 APGT controller/runtime pass for generated-action route
no fake/proxy/cpu offload
no outcome-derived feature
no dataset branch
candidate_template diversity pass
```

## 必须记录

```text
leave_dataset_out_pass
leave_stratum_out_pass
leave_template_out_pass
paired_replay_pass
beats_AdamWParallel
beats_bestLR
beats_NoOp
beats_Random
shuffled_payload_control_fail
short_run_boundary_open
full_run_boundary_open
```

---

# 24. P16：Base-Acc Sentinel continuation

## 目标

继续只作为 base health monitor，不参与 controller。

## 记录

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
mean_test_acc_LQ
mean_test_acc_MLP
mean_test_acc_AdamWStrongLRGridMLP
LQ_minus_MLP
LQ_minus_StrongLRGridMLP
catastrophic_fail
base_acc_used_for_controller = 0
```

## 判断

```text
base_acc_sentinel_pass = 1 iff:
  LQ_catastrophic_fail = 0
  base_acc_used_for_controller = 0
```

Base-Acc 不能用于：

```text
selector
controller
certificate
threshold
route promotion
```

---

# 25. 必须可视化清单

```text
fig_p1_lineage_hierarchy_sunburst.svg
fig_p1_candidate_template_expansion_hist.svg
fig_p1_action_to_template_sankey.svg
fig_p2_lto_precision_drop_by_ranker.svg
fig_p2_template_fold_heatmap.svg
fig_p3_memory_offdiag_matched_lift.svg
fig_p3_intervention_lift_waterfall.svg
fig_p4_target_density_vs_template_count.svg
fig_p4_value_risk_frontier.svg
fig_p5_ranker_ablation_waterfall.svg
fig_p5_value_vs_risk_score_scatter.svg
fig_p5_template_quota_map.svg
fig_p6_calibration_vs_heldout_drift.svg
fig_p6_accepted_region_by_template.svg
fig_p8_runtime_component_stack.svg
fig_p9_source_to_generated_damage_matrix.svg
fig_p9_value_direction_cosine_hist.svg
fig_p12_apgt_value_longrisk_frontier.svg
fig_p12_apgt_template_diversity.svg
```

---

# 26. 并行执行安排

## Batch A：不需要生成新动作，立即跑

```text
P0 boundary reproduction
P1 lineage hierarchy rebuild
P2 leave-template-out audit
P4 template-balanced target map
P5 template-deconfounded ranker
P6 rank-safe certificate
```

这些都基于现有 canonical AP0 / v9.6.4 artifacts，可以并行。

## Batch B：memory/offdiag causality

```text
P3 matched-pair audit
P3 intervention clone panel
P3 memory/offdiag lift visualization
```

可以和 Batch A 并行，但需要 branch-horizon replay。

## Batch C：generated primitive

```text
P9 APGL/APGH damage decomposition v2
P10 APGT primitive implementation
P11 APGT branch-horizon smoke
P12 APGT outcome / geometry pass
P13 APGT certificate/controller
```

APGT 不等 P5/P6 结束。这样避免一轮只清一个 blocker。

## Batch D：runtime boundary

```text
P8 selected-controller runtime preflight
```

只有 P7 或 P13 出现 candidate controller 后转 official measurement；但 runtime component profiler 可以提前 dry-run。

---

# 27. Stop rules

## 停止 existing-action route

如果满足：

```text
P2 leave-template-out fail
P4 no template-balanced target
P5/P6 no ranker/certificate with template_count >= 16
```

则停止 existing-action controller promotion，不再调 R4C / CERT12。

## 停止 APGL small variants

如果 P9 显示：

```text
value_direction_lost fraction >= 0.50
or longrisk_created_rate >= 0.50
```

则停止 APGL small variants，转 APGT。

## 停止 generated primitive promotion

如果 P12 中 APGT 全部：

```text
GradeAB precision < 0.30
or V_integrated_LCB <= 0
or h240_longrisk_UCB > 0.15
```

则 generated route 停在 `R8-APGTGeneratedFrontierFail`，不能打开 controller。

---

# 28. 本轮预期结论类型

v9.6.5 最好情况：

```text
existing-action route：
  template-balanced rank-safe certificate pass；
  selected runtime pass；
  ready for leaveout / paired replay。
```

中等情况：

```text
memory/offdiag causal pass；
existing-action ranker still fails template diversity；
APGT weak pass；
下一轮围绕 APGT controller。
```

差情况：

```text
template-balanced target absent；
memory/offdiag causal fail；
APGT generated frontier fail；
需要重新定义 action source / candidate generation universe。
```

最重要的是：v9.6.5 不能再只输出 “ranker failed / generator failed”。它必须明确回答：

```text
1. 是否有足够多 candidate templates 支持好动作？
2. memory/offdiag safe 是真实机制还是 pocket？
3. existing-action route 是否应停止？
4. generated-action route 是否能重建 value-preserving frontier？
```

---

# 29. 结论

v9.6.4 之后，项目的核心 blocker 已经不是：

```text
stable accept；
canonical truth；
payload replay；
runtime preflight；
单个 ranker AUC；
单个 APGL implementation。
```

而是：

$$
\boxed{
\text{我们需要一个跨 candidate template 成立的好训练几何规则。}
}
$$

v9.6.5 的实验必须围绕这个命题设计。不能再把一个 template 扩张出的 87 个动作当成独立 support，也不能继续让 generator 只保证 payload 合法。下一步要么找到 template-balanced existing-action controller，要么生成真正 template-diverse、memory/offdiag-safe、value-preserving 的新动作。
