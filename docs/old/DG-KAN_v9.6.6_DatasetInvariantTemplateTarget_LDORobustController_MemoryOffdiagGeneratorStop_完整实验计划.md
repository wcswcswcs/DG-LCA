# DG-KAN v9.6.6 Dataset-Invariant Template Target / LDO-Robust Controller / Memory-Offdiag Generator Stop 完整实验计划

> 本计划基于 v9.6.5 真实实验结果制定。v9.6.6 不再围绕 `R5B`、`CERT13`、`APGT4` 做阈值小修，而是正面回答三个问题：
>
> 1. v9.6.5 找到的 87-action 高质量区域，是否能成为跨 dataset / stratum / template 稳定的规则？
> 2. memory/offdiag safe 到底是可用于训练当下选择的真实几何信号，还是只在事后 matched-pair 表里看起来有效？
> 3. generated-action primitive 连续多轮 value-negative / high-longrisk 后，是否应该暂时停止盲目新变体，转向机制约束和 stop-rule？

---

# 0. v9.6.6 的总体判断

v9.6.5 不是失败回退。它完成了三件非常重要的事：

```text
1. 新 candidate_template_id hierarchy 修正了 v9.6.4 的“单模板塌缩”判断；
2. accepted-region leave-template-out 已过，template pocket 不再是 terminal blocker；
3. memory/offdiag matched-pair causal gate 首次过。
```

但 v9.6.5 仍然不能打开 controller / runtime / paired replay，因为：

```text
1. P4 没有 official template-balanced target；
2. P6 frozen certificate 因 LDO drop = 0.3793103448 失败；
3. APGT 完整生成 512 个动作、12288 行 branch-horizon replay，但 best APGT4 仍 V_integrated_LCB < 0 且 h240 longrisk 高。
```

因此 v9.6.6 的核心不是继续“造一个新 APG* primitive”，而是：

$$
\boxed{
\text{先把 existing-action route 的 LDO failure 拆清楚；如果它仍不可修，再停止 selection route，转向更强的 memory/offdiag-safe generator。}
}
$$

---

# 1. v9.6.5 数据复盘与独立判断

## 1.1 v9.6.5 的真实推进

v9.6.5 复现 v9.6.4 boundary 后，重新定义了 `candidate_template_id`。在新口径下，accepted region：

```text
unique candidate template count = 87
max candidate template share = 0.011494252873563218
template-to-action expansion ratio = 1.0
```

这说明 v9.6.4 的 `candidate_template_unique_count = 1` 不是最终真相，而是旧 template 口径太粗。v9.6.5 纠正了这一层判断。

P2 leave-template-out 也通过：

```text
best object = RANK5-invariant-risk-minimization
GradeAB precision = 0.8735632183908046
V_integrated_LCB = 0.14111334880346277
h240_longrisk_UCB = 0.0
LTO precision drop = 0.011494252873563315
```

这说明 accepted region 不再是“单模板 expansion”。这是一个重要进展。

P3 memory/offdiag causal gate 首次过：

```text
matched joint pairs = 397
intervention clones = 512
clone replay rows = 12288
matched joint GradeAB lift = 0.19143576826196473
V lift = 0.29043934585539005
longrisk drop = 0.906693153540672
memory_offdiag_causal_pass = 1
```

这说明 memory/offdiag safe 不只是普通相关性。它已经在 matched-pair 证据上显示：当动作处在 memory/offdiag safe 状态时，GradeAB 与 V 有明显提升，longrisk 明显下降。

## 1.2 v9.6.5 的真实失败

P4 没有找到 official template-balanced target。最佳目标 `T4.1-existing-ValuePositiveNoLongRisk` 是：

```text
count = 97
coverage = 0.03372739916550765
GradeAB = 0.8144329896907216
V_integrated_LCB = 0.16622185363738462
h240_longrisk_UCB = 0.0
bad UCB = 0.1505214153590923
null UCB = 0.18923506172101434
LDO drop = 0.3333333333333333
pass = 0
```

它的 count / coverage / value / longrisk 都够，但 bad/null 和 LDO 过不了。

两个更干净的目标也没过：

```text
T4.2-TemplateBalancedValuePositiveNoLongRisk:
  count = 79
  coverage = 0.027468706536856746
  GradeAB = 1.0
  V LCB = 0.16201929527024647
  longrisk UCB = 0.0
  bad/null = 0.0 / 0.0
  LDO drop = 0.41379310344827586

T4.3-MemoryOffdiagSafeValuePositive:
  count = 77
  coverage = 0.026773296244784424
  GradeAB = 1.0
  V LCB = 0.16402807605399972
  longrisk UCB = 0.0
  bad/null = 0.0 / 0.0
  LDO drop = 0.39080459770114945
```

这说明我们现在有两个相互冲突的区域：

```text
1. coverage 够的目标：T4.1，bad/null 与 LDO 不过；
2. 很干净的目标：T4.2/T4.3，coverage 不够且 LDO 仍不过。
```

P5 ranker 层很强：

```text
best ranker = R5B-pairwise-within-template
TopK87 precision = 0.8735632183908046
V_integrated_LCB = 0.14111334880346277
h240_longrisk_UCB = 0.0
candidate_template_count = 87
max_candidate_template_share = 0.011494252873563218
LTO drop = 0.011494252873563315
```

但 P6 certificate 仍失败：

```text
best certificate = CERT13-FixedTopK87Quota1
accepted = 87
coverage = 0.030250347705146036
precision = 0.8735632183908046
V_integrated_LCB = 0.14111334880346277
h240_longrisk_UCB = 0.0
LTO drop = 0.011494252873563315
LDO drop = 0.37931034482758624
certificate pass = 0
```

所以真正卡住的不是 template，也不是 TopK 本身，而是：

$$
\boxed{
\text{这批动作跨 template 稳，但跨 dataset 不稳。}
}
$$

P12 APGT 继续失败：

```text
best APGT = APGT4-TemplateMixtureAnchor
GradeAB precision = 0.03125
V_integrated_LCB = -0.6095415439891327
h240_longrisk_UCB = 0.7869099970100811
```

这说明 APGT 工程闭合，但没有生成可用动作。当前 generated-action route 不是小调参数问题，而是生成目标本身仍没抓住 memory/offdiag-safe、value-preserving、longrisk-veto 的组合。

---

# 2. v9.6.6 的第一性目标

v9.6.6 的目标不是 “把 v9.6.5 的 best threshold 再扫一遍”。

v9.6.6 要回答：

$$
\boxed{
\text{现有动作里是否存在一个不依赖 dataset 特例、template 稳定、bad/null/longrisk 都可控的 accepted region？}
}
$$

如果存在，则推进到 selected controller 和 runtime。

如果不存在，则应停止 existing-action selection 主线，并明确转向：

$$
\boxed{
\text{生成 memory/offdiag-safe、value-preserving、dataset-invariant 的新动作。}
}
$$

---

# 3. 本轮硬约束

继续遵守：

```text
no teacher
no distillation
no auxiliary loss
no CE loss modification
no dataset-name branch in controller
no validation/test feature
no future outcome feature
no outcome-at-commit feature
no old table as official
no fake rows
no proxy rows
no CPU offload
manual forward / manual backward / manual AdamW update only
functional update remains update rule, not loss trick
```

Functional update 仍然是：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{AdamW}
+
\Delta\theta_{functional}.
$$

不能改成：

$$
L = CE + \lambda L_{functional}.
$$

本轮新增硬约束：

```text
1. 不能用 dataset_name 作为选择条件；
2. 可以用 dataset 做 leave-out 诊断，但不能按 dataset 调阈值；
3. 不能把 R5B ranker pass 写成 controller pass；
4. 不能把 CERT13 TopK 数字写成 official，除非 LDO/LSO/LTO 全部过；
5. 不能把 memory/offdiag matched-pair causal pass 直接当 generator pass；
6. 不能继续新增 APGT 小变体，除非先过 source-preservation and no-longrisk-created preflight；
7. selected runtime 只能在 selected controller pass 后打开。
```

---

# 4. 总体成功标准

## 4.1 Existing-action controller 最低 pass

令 action universe size 为：

$$
N=2876.
$$

coverage 下限为：

$$
N_{min}=\lceil 0.03N \rceil = 87.
$$

existing-action controller pass 需要同时满足：

```text
accepted_count >= 87
coverage >= 0.03
GradeAB precision >= 0.75
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.05
bad_event_UCB <= 0.05
null_event_UCB <= 0.15
candidate_template_count >= 32
max_candidate_template_share <= 0.05
LTO_drop <= 0.10
LDO_drop <= 0.10
LSO_drop <= 0.10
no forbidden field used
no dataset_name dispatch
no diagnostic/outcome field used
```

## 4.2 Runtime pass

只有 controller pass 后才测 selected runtime。pass 标准：

```text
selected_runtime_measured = 1
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
payload_apply_error_linf_max <= 1e-6
controller_feature_compute_q90_ms <= budget
no CPU offload
no proxy runtime
```

## 4.3 Generated primitive pass

AP generated route 只有在以下条件下才允许进入 official candidate：

```text
generated_actions >= 512
branch_horizon_rows complete
unresolved_exception_count = 0
quality_audit_pass = 1
GradeAB precision >= 0.75
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.05
new_positive_created_rate >= 0.05
longrisk_created_rate <= 0.05
source_positive_preserved_rate >= 0.80
candidate_template_count >= 32
```

如果 generated route 连续三轮满足：

```text
best_generated_V_integrated_LCB < 0
best_generated_h240_longrisk_UCB > 0.50
best_generated_GradeAB_precision < 0.10
```

则触发 generator stop-rule，不再新增 APG* 小变体，改为重新定义 generator objective。

---

# 5. v9.6.6 实验结构

## P0. Boundary reproduction and artifact legality

### 目标

确认 v9.6.5 边界没有回退，并确认本轮不会读入 forbidden score、旧表或 outcome-derived field。

### 假设

$$
H0:
\text{v9.6.5 的 R4-NoTemplateBalancedTarget boundary 可稳定复现。}
$$

### 必须记录

```text
source_route_v9650
lineage_hierarchy_pass_v9650
memory_offdiag_causal_pass_v9650
template_balanced_target_pass_v9650
template_deconfounded_ranker_pass_v9650
rank_safe_certificate_v13_pass_v9650
apgt_weak_pass_v9650
system_legal_controller_pass_v9650
no_fake
no_proxy
cpu_offload_used
forbidden_field_count
outcome_field_used_count
```

### Pass 标准

```text
p0_pass = 1
source_route_v9650 = R4-NoTemplateBalancedTarget
no_fake = 1
no_proxy = 1
forbidden_field_count = 0 for official path
```

### 可视化

```text
fig_p0_route_timeline_v9280_to_v9650.svg
fig_p0_gate_status_waterfall.svg
fig_p0_field_legality_matrix.svg
```

---

## P1. Dataset-LDO failure autopsy

### 目标

v9.6.5 的最大硬失败是 `CERT13` 的 LDO drop = 0.3793。P1 要回答：到底是哪一个 dataset / source group / event family / geometry bucket 导致 LDO 崩？

这不是为了按 dataset 调参，而是为了知道当前规则是否真的泛化。

### 假设

$$
H1a:
\text{LDO drop 来自一个或少数 dataset 的 target density 不足。}
$$

$$
H1b:
\text{LDO drop 来自 ranker score 在不同 dataset 上尺度不一致。}
$$

$$
H1c:
\text{LDO drop 来自 memory/offdiag-safe 区域在 dataset 间分布不同。}
$$

### 必须记录

对每个 dataset $d$ 记录：

```text
dataset_id
action_count
GradeAB_count
T4.1/T4.2/T4.3/T4.5_count
R5B_TopK87_overlap
CERT13_accepted_count
precision
V_integrated_LCB
h240_longrisk_UCB
bad_UCB
null_UCB
memory_safe_rate
offdiag_safe_rate
rank_score_mean/std/quantiles
risk_score_mean/std/quantiles
value_score_mean/std/quantiles
```

同时记录 leave-one-dataset-out：

```text
train_datasets
heldout_dataset
threshold_selected
accepted_count_heldout
precision_heldout
V_LCB_heldout
longrisk_UCB_heldout
bad_UCB_heldout
null_UCB_heldout
rank_score_shift
support_shortfall_reason
```

### Pass 标准

P1 本身是诊断阶段，pass 表示问题可归因：

```text
LDO_primary_failure_axis identified
explained_LDO_drop_fraction >= 0.80
no forbidden dataset-specific selector produced
```

### 可视化

```text
fig_p1_ldo_drop_by_dataset.svg
fig_p1_score_distribution_by_dataset.svg
fig_p1_target_density_by_dataset.svg
fig_p1_memory_offdiag_rate_by_dataset.svg
fig_p1_rank_score_shift_vs_precision_drop.svg
fig_p1_dataset_family_template_heatmap.svg
```

---

## P2. Template-balanced target density repair without dataset-specific tuning

### 目标

判断是否存在一个 target region，既满足 count / value / longrisk，又满足 bad/null 和 LDO。不能按 dataset 调阈值。

### 背景

v9.6.5 的目标冲突如下：

```text
T4.1: count=97, coverage pass, value pass, longrisk pass, but bad/null and LDO fail.
T4.2: count=79, clean bad/null, but coverage fail and LDO fail.
T4.3: count=77, clean bad/null, but coverage fail and LDO fail.
T4.5: count=92, coverage pass, but bad/null and LDO fail.
```

所以 P2 不再只找一个二元标签，而是构造 target lattice 的三层结构：

```text
Core target:
  clean but sparse, e.g. T4.2/T4.3.

Expansion target:
  near-core actions with controlled bad/null risk.

Reject target:
  longrisk/bad/null/memory/offdiag fail actions.
```

### 假设

$$
H2:
\text{存在 Core + Expansion 组合，使 count 达到 87，同时 bad/null/LDO 不崩。}
$$

### 目标函数

定义 action score：

$$
S_{target}(a)
=
V_{LCB}(a)
-
\lambda_L LongRisk_{UCB}(a)
-
\lambda_B Bad_{UCB}(a)
-
\lambda_N Null_{UCB}(a)
-
\lambda_M MemoryFail(a)
-
\lambda_O OffdiagFail(a).
$$

其中所有阈值必须在 calibration split 冻结，不能按 dataset 改。

### 必须记录

```text
target_id
core_count
expansion_count
accepted_count
coverage
GradeAB_precision
V_integrated_LCB
h240_longrisk_UCB
bad_UCB
null_UCB
candidate_template_count
max_candidate_template_share
min_dataset_count
max_dataset_share
LDO_drop
LSO_drop
LTO_drop
leave_family_out_drop
```

### Pass 标准

```text
accepted_count >= 87
coverage >= 0.03
GradeAB_precision >= 0.75
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
candidate_template_count >= 32
max_candidate_template_share <= 0.05
LDO_drop <= 0.10
LSO_drop <= 0.10
LTO_drop <= 0.10
```

### 可视化

```text
fig_p2_core_expansion_frontier.svg
fig_p2_count_vs_bad_null_longrisk.svg
fig_p2_target_lattice_parallel_coordinates.svg
fig_p2_ldo_drop_vs_coverage.svg
fig_p2_template_dataset_support_map.svg
fig_p2_reject_reason_stacked_bar.svg
```

---

## P3. Memory/offdiag signal: from matched-pair evidence to usable veto

### 目标

v9.6.5 的 P3 说明 memory/offdiag safe matched-pair evidence 很强，但 intervention clones 本身没有 lift。P3 要回答：memory/offdiag safe 到底能否作为 **veto**，而不是 generator recipe？

### 假设

$$
H3:
\text{memory/offdiag safe 更适合作为风险否决条件，而不是单独生成好动作的操作。}
$$

### 必须记录

```text
memory_safe_indicator
offdiag_safe_indicator
joint_memory_offdiag_safe
GradeAB_rate
V_integrated_mean/LCB
longrisk_rate/UCB
bad_rate/UCB
null_rate/UCB
old_family_fail_rate
old_stratum_fail_rate
cover_entropy_delta
basis_effective_rank_delta
population_risk_offdiag_score
```

并区分三类：

```text
matched-pair causal evidence
intervention clone effect
online legal feature proxy
```

### Pass 标准

memory/offdiag veto 可进入 ranker 的条件：

```text
matched_pair_joint_lift_GradeAB >= 0.15
matched_pair_joint_lift_V >= 0.15
matched_pair_longrisk_drop >= 0.50
legal_proxy_AUC_longrisk >= 0.70 or TopK risk veto removes >= 70% longrisk
intervention_clone_positive_lift not required
```

### 可视化

```text
fig_p3_memory_offdiag_lift_matrix.svg
fig_p3_veto_precision_recall_longrisk.svg
fig_p3_memory_offdiag_proxy_vs_true.svg
fig_p3_old_family_fail_by_veto.svg
fig_p3_cover_entropy_by_veto.svg
```

---

## P4. Dataset-invariant ranker: value rank + risk veto + memory/offdiag veto

### 目标

构造一个不使用 dataset_name 的 ranker，但训练时用 dataset leave-out 做稳定性约束。

### 假设

$$
H4:
\text{把 value rank、longrisk veto、memory/offdiag veto 分开，可以降低 LDO drop。}
$$

### Ranker 形式

不要把所有信号混成一个黑箱分数。采用两阶段：

第一阶段 value rank：

$$
S_V(a)=f_V(x_a),
$$

第二阶段 risk/memory veto：

$$
Accept(a)=1
\iff
S_V(a)\in TopK
\land
R_L(a)\le \tau_L
\land
R_B(a)\le \tau_B
\land
R_N(a)\le \tau_N
\land
M(a)\le \tau_M
\land
O(a)\le \tau_O.
$$

其中 $M(a)$ 是 memory fail risk，$O(a)$ 是 offdiag fail risk。

### 候选 ranker

```text
R6A-ValueOnlyBaseline
R6B-ValueRankLongRiskVeto
R6C-ValueRankLongRiskBadNullVeto
R6D-ValueRankMemoryOffdiagVeto
R6E-GroupAdversarialScoreNormalization
R6F-LeaveDatasetOutMinimaxRanker
R6G-TemplateQuotaRanker
R6H-DatasetShiftCalibratedButNoDatasetDispatch
```

注意：`R6H` 可以用 dataset 做 calibration diagnostics，但线上不能根据 dataset dispatch。

### 必须记录

```text
ranker_id
feature_groups_used
forbidden_field_count
TopK87_precision
TopK87_V_LCB
TopK87_longrisk_UCB
TopK87_bad_UCB
TopK87_null_UCB
candidate_template_count
max_template_share
LDO_drop
LSO_drop
LTO_drop
leave_family_out_drop
min_dataset_precision
macro_dataset_precision
veto_reject_count_by_reason
```

### Pass 标准

```text
TopK87_precision >= 0.75
V_integrated_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
LDO_drop <= 0.10
LSO_drop <= 0.10
LTO_drop <= 0.10
forbidden_field_count = 0
```

### 可视化

```text
fig_p4_value_score_vs_risk_veto.svg
fig_p4_veto_waterfall.svg
fig_p4_ranker_leaveout_drop_bars.svg
fig_p4_dataset_macro_micro_precision.svg
fig_p4_template_quota_map.svg
fig_p4_rejected_longrisk_examples.svg
```

---

## P5. Rank-safe certificate v14

### 目标

把 P4 ranker 转成 frozen accepted region。以前 TopK 好看，但 certificate/accepted region 会崩。P5 要严格检测这个断层。

### 假设

$$
H5:
\text{如果 P4 ranker 真稳定，则 frozen certificate 的 heldout accepted region 也应稳定。}
$$

### Certificate 形式

```text
CERT14A-fixed-topK87
CERT14B-score-threshold-frozen
CERT14C-topK-plus-veto
CERT14D-conformal-risk-bound
CERT14E-minimax-leaveout-bound
CERT14F-template-quota-no-dataset-dispatch
```

### 必须记录

```text
certificate_id
calibration_split
heldout_split
accepted_count_cal
accepted_count_heldout
coverage_heldout
GradeAB_precision_heldout
V_integrated_LCB_heldout
h240_longrisk_UCB_heldout
bad_UCB_heldout
null_UCB_heldout
LDO_drop
LSO_drop
LTO_drop
candidate_template_count
max_candidate_template_share
ECE
threshold_sensitivity
```

### Pass 标准

```text
accepted_count_heldout >= 87
coverage_heldout >= 0.03
precision_heldout >= 0.75
V_integrated_LCB_heldout > 0
longrisk_UCB_heldout <= 0.05
bad_UCB_heldout <= 0.05
null_UCB_heldout <= 0.15
LDO_drop <= 0.10
LSO_drop <= 0.10
LTO_drop <= 0.10
```

ECE 不是唯一否决项。若 certificate 是 rank-based controller，则 TopK / accepted region 指标优先于概率 ECE。但必须标注：

```text
probability_calibrated = 0/1
rank_controller = 0/1
```

### 可视化

```text
fig_p5_calibration_vs_heldout_drift.svg
fig_p5_threshold_sensitivity_grid.svg
fig_p5_accepted_region_by_dataset_template.svg
fig_p5_precision_value_longrisk_frontier.svg
fig_p5_ece_vs_topk_quality.svg
```

---

## P6. Existing-action minimal controller

### 目标

只有 P5 通过后才打开。P6 把 certificate 变成 actual controller candidate。

### 假设

$$
H6:
\text{通过 P5 的 accepted region 可以作为 system-legal existing-action controller。}
$$

### 必须记录

```text
controller_id
certificate_id
accepted_count
coverage
precision
V_LCB
longrisk_UCB
bad_UCB
null_UCB
candidate_template_count
max_template_share
LDO/LSO/LTO drops
feature_compute_ms_q50/q90/q99
payload_lookup_ms_q90
payload_apply_ms_q90
```

### Pass 标准

```text
controller_selected = 1
all P5 quality gates pass
feature_compute_ms_q90 within budget
no forbidden feature
```

### 可视化

```text
fig_p6_controller_accept_trace.svg
fig_p6_feature_cost_breakdown.svg
fig_p6_accepted_action_geometry_cards.svg
```

---

## P7. Selected controller runtime

### 目标

测 selected controller 的真实 online runtime。不能再用 diagnostic runtime。

### 必须记录

```text
step_count
active_step_count
accepted_step_count
controller_feature_time_q90
score_time_q90
veto_time_q90
payload_lookup_time_q90
payload_apply_time_q90
base_step_time_q90
full_step_time_q90
step_ratio_q90
memory_ratio
kernel_count
sync_count
empty_step_kernel_count
```

### Pass 标准

```text
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
empty_step_kernel_count = 0
payload_apply_error_linf_max <= 1e-6
```

### 可视化

```text
fig_p7_runtime_breakdown_stacked.svg
fig_p7_step_ratio_distribution.svg
fig_p7_payload_apply_vs_feature_compute.svg
fig_p7_active_step_timeline.svg
```

---

## P8. Generated-action stop-rule and APGT/APGL/APGH damage consolidation

### 目标

判断 generated-action route 是否应该暂停盲目变体。

### 背景

APGL/APGH/APGT 多轮共同现象：

```text
generated actions all materialized;
branch-horizon rows complete;
best primitive GradeAB precision near 0.03;
V_integrated_LCB negative;
h240 longrisk UCB high;
dominant damage often longrisk-created or value-direction-lost.
```

### 必须记录

```text
primitive_family
best_primitive
best_GradeAB_precision
best_V_LCB
best_longrisk_UCB
new_positive_created_rate
source_positive_preserved_rate
longrisk_created_rate
value_direction_cosine_before/after
memory_fail_delta
offdiag_fail_delta
cover_entropy_delta
basis_rank_delta
```

### Stop-rule

如果以下条件连续满足：

```text
best_GradeAB_precision < 0.10
best_V_LCB < 0
best_longrisk_UCB > 0.50
new_positive_created_rate < 0.05
longrisk_created_rate > 0.50
```

则：

```text
generated_route_blind_variant_stop = 1
```

并禁止继续新增 APG* 小变体，除非新 primitive 明确改变生成目标。

### 可视化

```text
fig_p8_generated_family_failure_timeline.svg
fig_p8_damage_mode_by_primitive.svg
fig_p8_value_direction_cosine_hist.svg
fig_p8_longrisk_created_waterfall.svg
```

---

## P9. APGU memory/offdiag-safe generator only if stop-rule allows

### 目标

如果 P8 没有触发 stop-rule，才运行 APGU1-APGU4。否则本阶段记录 not_run。

### 新 primitive 原则

APGU 不再做普通 residual blend。它必须显式约束：

```text
1. preserve value direction;
2. no longrisk created;
3. memory safe;
4. offdiag safe;
5. template diverse;
6. cover entropy non-collapse.
```

### 候选

```text
APGU1-ValueDirectionPreservingMemoryVeto
APGU2-OffdiagSafeTrustRegion
APGU3-CoverEntropyConstrainedResidual
APGU4-PopRiskSignalChannelProjectedUpdate
```

### Pass 标准

```text
generated_actions >= 256
quality_audit_pass = 1
GradeAB precision >= 0.75
V_integrated_LCB > 0
h240_longrisk_UCB <= 0.05
longrisk_created_rate <= 0.05
source_positive_preserved_rate >= 0.80
```

### 可视化

```text
fig_p9_apgu_value_vs_longrisk.svg
fig_p9_source_to_generated_preservation.svg
fig_p9_template_diversity_generated.svg
```

---

## P10. System gate

### 目标

只有 P6/P7 或 P9 通过后，才进入 system gate。

### Pass 标准

```text
system_legal_controller_pass = 1
official_eligible = 1
controller_selected = 1
selected_runtime_pass = 1
no_fake = 1
no_proxy = 1
no_dataset_dispatch = 1
```

---

## P11. Leaveout and paired replay boundary

### 目标

只有 system pass 后才打开。

### 必须记录

```text
leave_dataset_out_precision
leave_dataset_out_V_LCB
leave_dataset_out_longrisk_UCB
leave_stratum_out_precision
leave_template_out_precision
paired_replay_RealFunctional
paired_replay_AdamWParallel
paired_replay_bestLR
paired_replay_NoOp
paired_replay_Random
paired_replay_ShuffledPayload
```

### Pass 标准

```text
RealFunctional beats AdamWParallel
RealFunctional beats bestLR
RealFunctional beats NoOp
RealFunctional beats Random
ShuffledPayload fails
LDO/LSO/LTO stable
```

---

## P12. Short/full training boundary

### 目标

仍不打榜。只在 system + paired replay 过后，做短训/长训验证。

### 必须记录

```text
test_acc
val_loss_AUC
time_to_target
steps_to_target
ECE
NLL
CEp99
margin_p10
hard_stratum_acc
robustness_noise
forgetting_old_family
forgetting_old_stratum
runtime_step_ratio
memory_ratio
```

### Baselines

```text
MatchedMLP
AdamWStrongLRGridMLP
QuadraticFeatureMLP
KAN base without functional update
NoOp functional controller
Random payload controller
Shuffled payload controller
```

### Pass 标准

```text
Functional KAN improves at least one official axis:
  sample efficiency / calibration / robustness / hard-tail / anti-forgetting / final acc
while runtime and memory remain within envelope.
```

---

# 6. 并行执行计划

v9.6.6 必须避免“一轮只发现一个 blocker”。建议并行执行：

```text
Lane A: P1/P2 dataset-LDO and target lattice
Lane B: P3/P4 memory/offdiag veto + group-invariant ranker
Lane C: P5 certificate frozen accepted region
Lane D: P8 generated-route stop-rule
Lane E: Base-Acc Sentinel continuation
```

只有当 P5 pass 后，才启动：

```text
Lane F: P6/P7 controller/runtime
```

只有当 system pass 后，才启动：

```text
Lane G: P11/P12 paired replay and short/full
```

---

# 7. 必须落盘的 artifact 清单

```text
p0_boundary_reproduction_v9660.csv
p0_field_legality_audit_v9660.csv
p1_ldo_failure_autopsy_v9660.csv
p1_dataset_score_shift_trace_v9660.csv
p2_core_expansion_target_lattice_v9660.csv
p2_target_leaveout_grid_v9660.csv
p3_memory_offdiag_veto_audit_v9660.csv
p4_dataset_invariant_ranker_v9660.csv
p4_ranker_ablation_v9660.csv
p5_rank_safe_certificate_v14.csv
p5_certificate_threshold_sensitivity_v9660.csv
p6_existing_action_controller_v9660.csv
p7_selected_runtime_trace_v9660.csv
p8_generated_route_stop_rule_v9660.csv
p8_generated_damage_consolidation_v9660.csv
p9_apgu_primitive_implementation_v9660.csv
p9_apgu_branch_horizon_outcome_v9660.csv
p10_system_gate_v9660.csv
p11_leaveout_paired_replay_boundary_v9660.csv
p12_short_full_boundary_v9660.csv
base_acc_sentinel_v9660.csv
no_fake_audit_v9660.csv
contract_audit_v9660.csv
route_decision_v9660.json
failure_taxonomy_v9660.csv
run_manifest_v9660.json
```

---

# 8. 必须可视化的图

```text
fig_route_progress_v9300_to_v9660.svg
fig_p1_ldo_drop_by_dataset.svg
fig_p1_score_shift_by_dataset.svg
fig_p2_target_frontier_count_vs_risk.svg
fig_p2_core_expansion_tradeoff.svg
fig_p3_memory_offdiag_veto_pr_curve.svg
fig_p4_value_rank_risk_veto_scatter.svg
fig_p4_ranker_leaveout_drop.svg
fig_p5_calibration_heldout_drift.svg
fig_p5_accepted_region_support_map.svg
fig_p7_runtime_breakdown.svg
fig_p8_generated_damage_timeline.svg
fig_p12_short_full_comparison_if_open.svg
```

---

# 9. Route decision table

```text
R0-BoundaryReproductionFail:
  P0 fails.

R1-DatasetLDOFailureUnexplained:
  P1 cannot explain LDO drop.

R2-NoTemplateBalancedTargetStill:
  P2 cannot find target satisfying coverage, bad/null, value, longrisk, and leaveout.

R3-MemoryOffdiagVetoNotUsable:
  P3 matched-pair signal does not translate into legal veto.

R4-LegalRankStillLDOUnstable:
  P4 ranker strong in TopK but LDO/LSO/LTO fail.

R5-CertificateAcceptedRegionFail:
  P5 frozen certificate fails accepted-region gates.

R6-ExistingActionControllerPassRuntimeFail:
  P6 passes but P7 runtime fails.

R7-ExistingActionSystemPass:
  P6 and P7 pass; proceed to paired replay.

R8-GeneratedRouteStop:
  P8 stop-rule triggered; no more blind APG variants.

R9-GeneratedPrimitivePass:
  APGU passes; proceed to controller/runtime.

R10-PairedReplayFail:
  system pass but causal advantage fails.

R11-StrictPureKANFunctionalLocalPass:
  system + leaveout + paired replay pass.
```

---

# 10. 最终判断标准

v9.6.6 的最低有效成功不是 short/full training，而是以下任一结果：

```text
A. Existing-action route pass:
   template/dataset/stratum stable accepted region + selected runtime pass.

B. Existing-action route decisively fails:
   no target/ranker/certificate can pass LDO/LSO/LTO despite legal signal;
   stop selection route and document why.

C. Generated route decisively stops:
   APG families consistently value-negative/high-longrisk;
   stop blind primitive variants and require new mathematical generator objective.
```

真正的 v9.6.6 强成功是：

```text
system_legal_controller_pass = 1
selected_runtime_pass = 1
leaveout_ready = 1
paired_replay_ready = 1
```

但在当前 v9.6.5 边界下，更现实的目标是：

$$
\boxed{
\text{把 LDO failure 和 generated longrisk failure 拆成可决定的路线，而不是继续做无穷小补丁。}
}
$$
