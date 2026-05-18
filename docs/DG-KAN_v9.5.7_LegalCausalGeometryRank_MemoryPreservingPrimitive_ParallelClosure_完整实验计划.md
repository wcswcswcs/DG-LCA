# DG-KAN v9.5.7 Legal Causal Geometry Rank / Memory-Preserving Primitive / Parallel Closure 完整实验计划

> 本计划基于 v9.5.6 `Calibrated Geometry Rank / Cover-Memory Primitive / Parallel Closure` 的真实结果制定。  
> v9.5.7 不再继续修 GCERT18 阈值、APGC 小变体或单一 AUC 指标。  
> 本轮目标是把项目从“事后 outcome 排名很强，但训练当下不可用”推进到“训练当下能看见、能生成、能选择、能部署的好训练形状”。

---

# 0. 本轮总判断

v9.5.6 的核心结果不是功能成功，而是一次关键排雷：

```text
route = R1-GradeScopeInconsistent
success_v9560_strict_purekan_functional = False
success_v9560_full_functional = False
success_v9560_external_ready = False
primary_blocker = legacy_gcert18_uses_outcome_derived_score
```

v9.5.6 最重要的发现是：

```text
legacy GCERT18 TopK64 GradeAB precision = 0.890625
legacy TopK64 V_integrated LCB = 0.1650359732307112
legacy TopK64 h240 longrisk UCB = 0.0
```

但这个强信号不能 official，因为 legacy GCERT18 score 使用了：

```text
V_integrated
risk_score
```

这两个字段是 outcome-derived，不是训练当下可用的 commit-time 字段。

对应地，legal commit-time surrogate 完全没有复现这个强 TopK：

```text
legal TopK64 GradeAB precision = 0.046875
legal TopK64 V_integrated LCB = -0.5602833203663737
legal TopK64 h240 longrisk UCB = 0.6991203985679182
```

所以 v9.5.6 的本质结论是：

$$
\boxed{
\text{我们事后知道哪些动作好，但训练当下还不知道如何合法地认出它们。}
}
$$

这不是小问题。它说明 GCERT18 的强 TopK 不是一个可部署证书，而是一个泄漏了结果信息的诊断排名。

---

# 1. v9.5.7 的整体目标

v9.5.7 的目标不是继续提高一个 AUC，也不是让 APGC8 之外某个 APGC 变体看起来好一点。

本轮要回答三个第一性问题：

```text
Q1: 如果不使用 V_integrated / risk_score 这类结果字段，训练当下是否存在足够强的合法信号？
Q2: 如果合法信号不足，能否直接生成一种更不遗忘、更少 long-risk 的新动作？
Q3: 如果排名信号有但概率校准差，能否用 rank / conformal / LCB-UCB 方式形成 accepted region，而不是强行要求 ECE 过线？
```

本轮的核心假设是：

$$
\boxed{
\text{好 functional update 应该是一个 memory-safe、signal-supported、low-longrisk 的训练形状修正。}
}
$$

不是：

```text
短期 acc 上升；
单 horizon V_ctrl 上升；
全局 AUC 好看；
某个 diagnostic rank 很强；
某个生成器能产出合法 payload。
```

v9.5.7 的最低有效推进是：

```text
1. 把所有 field 分成 legal / diagnostic / forbidden 三类；
2. 证明 legacy rank 的强信号到底来自哪些 forbidden 字段；
3. 建立 legal-only causal geometry feature ledger；
4. 实测 legal-only upper-bound 是否能接近 legacy TopK；
5. 若 legal selection 失败，实现 memory-preserving primitive；
6. 如果出现可用 rank，使用 accepted-region gate，而不是只看 ECE；
7. 若 controller 过线，再测 selected runtime；
8. 若 runtime 过线，再打开 leaveout / paired replay / short-full。
```

---

# 2. 本轮不做什么

v9.5.7 明确不做：

```text
1. 不再调 legacy GCERT18 threshold；
2. 不再把 GCERT18 outcome-derived rank 当 official；
3. 不再用 V_integrated / risk_score / GradeAB / future branch-horizon outcome 作为 commit-time feature；
4. 不继续微调 APGC1-APGC7；
5. 不因为 APGC8 negative control 是 best 就把 APGC 写成 pass；
6. 不只看 AUC；
7. 不把 ECE 当唯一拒绝理由，也不把高 AUC 当成功理由；
8. 不按 dataset 调 threshold；
9. 不在 controller 未过前打开 selected runtime；
10. 不在 selected runtime 未过前打开 official paired replay；
11. 不把 Base-Acc Sentinel 当 functional success；
12. 不用 validation/test/outcome-at-commit 做 controller 特征。
```

允许做：

```text
1. outcome-derived rank 作为 diagnostic oracle；
2. 用 calibration split 的 outcome label 训练 legal model，但 official feature 必须 commit-time legal；
3. 用 heldout / leaveout 判断 legal model 是否真的能选动作；
4. 用 current train batch / train-memory buffer 做 legal microprobe；
5. 用 train-time per-example gradient、action response、cover/memory/cost 特征；
6. 用 rank-based accepted region 和 conformal bound 替代概率校准；
7. 并行跑 generator、certificate、runtime preflight，但 downstream 必须 gate-block。
```

---

# 3. 关键定义

## 3.1 Action 的完整评价向量

每个动作 $a$ 不再用单个标签判断，而是记录完整向量：

$$
Q(a)=
(
V_{20},V_{80},V_{240},
Bad_{20},Bad_{80},Bad_{240},
Null_{20},Null_{80},Null_{240},
LongRisk_{240},
MemoryFail,
CoverCollapse,
CurvatureRise,
Cost
).
$$

其中：

```text
V20/V80/V240:
  RealFunctional 相比 AdamWParallel / bestLR / NoOp / Random 的收益。

Bad:
  明确伤害模型的事件。

Null:
  不明显有用的事件。

LongRisk240:
  h240 长程风险。

MemoryFail:
  old family / old stratum / memory buffer 上性能变坏。

CoverCollapse:
  basis effective rank、cover entropy、hard-tail cover 出现塌缩。

CurvatureRise:
  局部曲率、Jacobian spectral、CEp99 tail 变坏。

Cost:
  feature compute、certificate compute、payload apply、kernel launch、memory。
```

## 3.2 三类字段

### Green fields：可 official 使用

```text
current batch gradient
per-example gradient statistics
current train batch logits
current train batch CE / margin / NLL
current train-memory buffer CE / margin / NLL
payload norm / linf / sparsity
action-AdamW cosine
linearized delta on current batch
linearized delta on train-memory buffer
cover/rank/entropy measured before commit or from current batch probe
feature cost / payload apply cost
step / family / bucket id, if not dataset-specific and not label-outcome-derived
```

### Amber fields：只能 diagnostic 或 calibration label 使用

```text
calibration split outcome label
calibration branch-horizon replay outcome
calibration GradeA/B/C/D/E label
calibration V20/V80/V240 label
legacy GCERT18 rank as diagnostic teacher
```

Amber fields 可以用于离线训练和审计，但不能作为 online commit-time feature。

### Red fields：禁止作为 official score feature

```text
V_integrated
risk_score derived from branch-horizon outcomes
future h20/h80/h240 label
GradeAB label itself
RealFunctional branch outcome
AdamWParallel / bestLR branch outcome
paired replay outcome
validation/test metric
old table official outcome
```

若任何 official score 使用 Red field：

```text
grade_scope_audit_pass = 0
controller_pass = 0
system_legal_controller_pass = 0
```

## 3.3 好动作的最低条件

一个动作 $a$ 只有满足以下条件，才可以进入 official 候选：

$$
ApplyOK(a)=1,
$$

$$
LCB(V_{20}(a))>0,
$$

$$
LCB(V_{80}(a))\ge -\epsilon_{80},
$$

$$
UCB(LongRisk_{240}(a))\le \tau_L,
$$

$$
UCB(Bad(a))\le \tau_B,
$$

$$
UCB(Null(a))\le \tau_N,
$$

$$
UCB(MemoryFail(a))\le \tau_M,
$$

$$
UCB(CoverCollapse(a))\le \tau_C,
$$

$$
Cost(a)\le C_{max}.
$$

v9.5.7 不再用单个 $YRobust$ 或单个 $GradeB$ 作为终点。Grade 只作为分层标签：

```text
GradeA: 最干净，但可能太稀疏；
GradeB: 可用候选，需要 memory / longrisk gate；
GradeC: 诊断候选，不直接 official；
GradeD/E: 风险或无效动作。
```

---

# 4. 核心假设

## H1：legacy GCERT18 的强 TopK 来自 outcome-derived rank，不是 legal certificate

H1 成立标准：

```text
legacy TopK64 GradeAB precision >= 0.80
legacy uses_outcome_derived_score = 1
legal surrogate TopK64 GradeAB precision < 0.25
legal TopK64 V_integrated LCB <= 0
legal TopK64 longrisk UCB > 0.20
```

H1 失败标准：

```text
去除 V_integrated / risk_score 后，legal score 仍能达到：
TopK64 GradeAB precision >= 0.75
V_integrated LCB > 0
h240 longrisk UCB <= 0.05
```

若 H1 失败，说明我们已经有 legal rank，可以直接进入 P10/P11。

## H2：legal signal 不是完全没有，但目前缺少 action-effect / memory-aware 特征

H2 成立标准：

```text
legal static scalar features fail；
加入 per-example action-effect、memory compatibility、population-risk SNR 后，TopK64 GradeAB precision 显著上升；
并且 V_integrated LCB > 0、h240 longrisk UCB <= 0.10。
```

H2 失败标准：

```text
legal rich feature upper-bound 仍 TopK64 GradeAB precision < 0.25；
TopK64 V_integrated LCB <= 0；
TopK64 longrisk UCB >= 0.30。
```

若 H2 失败，AP0 existing-action selection 主线停止，转向新 primitive generation。

## H3：memory 是当前最重要的风险来源

v9.5.6 已显示：

```text
APGR longrisk memory fail rate = 0.27735368956743
GradeB memory fail rate = 0.02531645569620253
ratio ≈ 10.95
```

H3 成立标准：

```text
memory fail 与 h240 longrisk / V_integrated negative 有稳定相关；
memory-safe subset 的 longrisk 显著下降；
memory-safe subset 的 V_integrated LCB 不显著下降。
```

H3 失败标准：

```text
memory fail 与 longrisk 无关；
或者 memory-safe gate 只会清掉所有 value-positive 动作。
```

## H4：现有 APGC generator 不是只是参数不好，而是目标错了

H4 成立标准：

```text
APGC1-APGC7 都未产生 usable frontier；
APGC8 negative control 是 best diagnostic；
APGC best V_integrated LCB <= 0；
APGC best h240 longrisk >= 0.50。
```

若 H4 成立，不再微调 APGC，改为 APGM memory-preserving primitive。

## H5：rank-based controller 可能比 calibrated probability 更适合当前阶段

v9.5.5/v9.5.6 反复出现：

```text
AUC / TopK 有时强；
ECE 很差；
probability calibration 不可靠。
```

H5 成立标准：

```text
rank-only accepted region 在 calibration-frozen threshold 下 heldout 过线；
即使 ECE > 0.10，也能满足 LCB/UCB decision gates。
```

H5 失败标准：

```text
rank TopK 在 heldout 不稳定；
coverage / risk / value 无法同时过线。
```

---

# 5. 并行执行架构

v9.5.7 不再一轮只清一个 blocker。采用并行批次：

```text
Batch A: P0-P2
  boundary reproduction
  field legality ledger
  legacy rank forensic

Batch B: P3-P6
  legal causal geometry features
  immediate legal microprobe
  legal rank upper-bound
  legacy-to-legal distillation sandbox

Batch C: P7-P9
  memory blocker anatomy
  APGM1-APGM8 memory-preserving primitive
  APGM branch-horizon smoke

Batch D: P10-P12
  rank/controller/certificate
  selected runtime
  system gate

Batch E: P13-P16
  leaveout
  paired replay
  short/full boundary
  Base-Acc Sentinel
```

每个 Batch 都可以独立落盘，但 downstream official 必须遵守 gate。

---

# 6. P0：v9.5.6 boundary reproduction

## 目标

确认 v9.5.6 的核心边界没有回退。

## 必须记录

```text
source_route_v9560
canonical_full_control_outcome_ready
ledger_rows
GradeAB_count_canonical_AP0
GradeAB_count_GCERT18_eval_universe
legacy_TopK64_GradeAB_precision
legacy_TopK64_V_integrated_LCB
legacy_TopK64_h240_longrisk_UCB
outcome_field_used_by_certificate_count
legacy_gcert18_legal_commit_time_pass
legal_TopK64_GradeAB_precision
legal_TopK64_V_integrated_LCB
legal_TopK64_h240_longrisk_UCB
apgc_generated_actions
apgc_branch_horizon_rows
best_apgc_primitive
best_apgc_GradeB_precision
best_apgc_V_integrated_LCB
best_apgc_h240_longrisk
system_legal_controller_pass
```

## 判断标准

P0 pass：

```text
source_route_v9560 = R1-GradeScopeInconsistent
outcome_field_used_by_certificate_count >= 1
legal surrogate does not reproduce legacy TopK
system_legal_controller_pass = 0
```

若 P0 不复现，停止后续，先查 artifact mismatch。

## 可视化

```text
v9560 route Sankey
legacy vs legal TopK64 bar chart
V_integrated LCB comparison
h240 longrisk comparison
```

## Artifact

```text
p0_v9560_boundary_reproduction.csv
p0_v9560_boundary_dashboard.json
```

---

# 7. P1：Field Legality Ledger 与 legacy rank forensic

## 目标

查清 legacy GCERT18 的强 ranking 到底用了哪些字段，并把所有字段分成 Green / Amber / Red。

## 实验设计

对 legacy GCERT18 score 做逐项拆解：

```text
legacy_full_score
legacy_without_V_integrated
legacy_without_risk_score
legacy_without_both
legal_only_terms
legal_only_terms_plus_memory
legal_only_terms_plus_cover
legal_only_terms_plus_cost
```

每个 score 都在相同 split 上计算：

```text
TopK32 / TopK64 / TopK87 / TopK128
GradeA/B precision
T5 precision
V_integrated LCB
h240 longrisk UCB
bad/null UCB
memory fail UCB
coverage
```

## 必须记录

```text
field_name
field_source
field_available_at_commit
field_uses_future_horizon
field_uses_branch_outcome
field_uses_validation_or_test
field_uses_dataset_name
field_cost_ms_q50/q90
field_legality_class: green/amber/red
score_variant_id
score_variant_red_field_count
score_variant_topk64_gradeab_precision
score_variant_topk64_V_lcb
score_variant_topk64_longrisk_ucb
```

## 判断标准

P1 pass：

```text
每个 field 都有 legality class；
所有 official candidate score red_field_count = 0；
legacy rank 的 leakage contribution 可解释；
field ledger hash 写入 provenance。
```

若 `legacy_without_both` 仍保持强 TopK，则说明 GCERT18 不是主要靠 outcome fields，需要重新检查 legal surrogate实现。若 `legacy_without_both` 崩掉，则进入 P3-P6。

## 可视化

```text
field legality heatmap
score ablation TopK curve
red-field contribution waterfall
TopK risk/value scatter
```

## Artifact

```text
p1_field_legality_ledger.csv
p1_legacy_rank_forensic.csv
p1_score_ablation_trace.csv
```

---

# 8. P2：Multi-grade target density 与 accepted-region 目标确认

## 目标

不再让一个二元标签支配实验。把 GradeA/B/C/T5/T_rank 拆开，明确哪个目标可以 official，哪个只能 diagnostic。

## 实验设计

在 canonical AP0 action universe 上重新计算：

```text
GradeA
GradeB
GradeC
T5-like clean target
T_rank_legacy_GCERT18_TopK87_diagnostic
MemorySafeGradeB
NoLongRiskGradeB
ValuePositiveGradeB
```

每个 target 记录 density 和质量。

## 必须记录

```text
target_id
target_source
target_uses_outcome_derived_rank
target_count
target_coverage
target_coverage_lcb
V_integrated_mean
V_integrated_LCB
h20/h80/h240 V_LCB
h240_longrisk_rate/UCB
bad_rate/UCB
null_rate/UCB
memory_fail_rate/UCB
cover_collapse_rate/UCB
GradeA/B/C composition
```

## 判断标准

Official target candidate：

$$
Coverage_{LCB}\ge 0.03,
$$

$$
V_{integrated}^{LCB}>0,
$$

$$
LongRisk_{240}^{UCB}\le 0.05,
$$

$$
Bad^{UCB}\le 0.05,
$$

$$
Null^{UCB}\le 0.15,
$$

$$
MemoryFail^{UCB}\le 0.10.
$$

Diagnostic target：

```text
coverage good but uses outcome-derived rank；
or quality good but coverage below 0.03；
or coverage/quality good only on calibration split。
```

## 可视化

```text
coverage vs V_integrated frontier
coverage vs longrisk frontier
Grade density stacked bar
target overlap Jaccard heatmap
```

## Artifact

```text
p2_multigrade_target_density.csv
p2_target_overlap_matrix.csv
p2_target_frontier_dashboard.json
```

---

# 9. P3：Legal Causal Geometry Feature Factory v1

## 目标

构建真正训练当下可用的 action-effect feature，而不是继续使用 payload norm / static state scalar。

## 特征组

### F1：per-example linearized action effect

对当前 batch 每个样本 $i$：

$$
d_i = g_i^T \Delta\theta_a.
$$

记录：

```text
mean(d_i)
std(d_i)
median(d_i)
p10/p90(d_i)
hard-tail subset mean(d_i)
old-family subset mean(d_i)
sign agreement fraction
negative-tail fraction
```

### F2：population-risk SNR gate

对 edge/basis group $g$：

$$
SNR_g=rac{\mu_g^2}{\sigma_g^2/(b-1)+\epsilon}.
$$

其中：

```text
mu_g = 当前 batch per-example gradient/action response mean
sigma_g^2 = 当前 batch per-example response variance
b = batch size
```

记录：

```text
fraction groups passing SNR gate
weighted SNR mean
hard-tail SNR
old-family SNR
low-SNR payload mass
```

### F3：off-diagonal agreement

用 mini-batch 内 pair agreement 近似 population-risk transfer：

$$
\Omega_B(a)=
\frac{1}{b(b-1)}
\sum_{i\ne j}
 r_i^T J_i M_a J_j^T r_j.
$$

记录：

```text
Omega_B_mean
Omega_B_hard_tail
Omega_B_old_family
Omega_B_positive_fraction
```

### F4：AdamW compatibility

```text
action_adamw_cosine
action_adamw_conflict_fraction
payload_mass_conflicting_with_adamw
SNR_pass_mass_conflicting_with_adamw
```

### F5：memory compatibility

用 train-memory buffer，不用 validation/test：

```text
old_family_linearized_CE_delta
old_stratum_linearized_CE_delta
memory_margin_delta
memory_forget_risk_proxy
memory_gradient_conflict
```

### F6：cover and rank stability proxy

```text
basis_effective_rank_delta_proxy
hard_tail_cover_entropy_delta_proxy
active_basis_count_delta_proxy
edge_group_payload_concentration
cover_collapse_proxy
```

### F7：cost features

```text
feature_compute_ms_q50/q90
payload_apply_ms_q50/q90
extra_kernel_count
memory_delta_mb
```

## 必须记录

```text
action_id
feature_group_id
feature_name
feature_value
feature_cost_ms
legality_class
available_at_commit
uses_outcome_field
uses_dataset_name
uses_validation_or_test
```

## 判断标准

P3 pass：至少一个 legal feature group 或组合满足：

```text
TopK64 GradeAB precision >= 0.50
TopK64 V_integrated LCB > 0
TopK64 h240 longrisk UCB <= 0.10
TopK64 bad/null UCB <= official gate
feature_compute_q90 <= 0.25 ms
leave-seed TopK degradation <= 0.10
```

P3 weak pass：

```text
TopK64 GradeAB precision >= 0.30
V_integrated LCB near 0 or positive
longrisk UCB <= 0.20
```

P3 fail：

```text
all legal feature groups TopK64 GradeAB precision < 0.25
or TopK64 V_integrated LCB <= 0
or longrisk UCB >= 0.30
```

## 可视化

```text
feature TopK precision curves
feature cost vs precision scatter
SNR vs longrisk scatter
memory compatibility vs V_integrated scatter
cover proxy vs longrisk scatter
```

## Artifact

```text
p3_legal_causal_geometry_feature_ledger.csv
p3_feature_topk_frontier.csv
p3_feature_cost_trace.csv
```

---

# 10. P4：Legal immediate-effect microprobe

## 目标

测试训练当下是否可以通过一个便宜、可回滚、只看 train batch / train-memory 的 microprobe 识别好动作。

## 实验设计

对每个 candidate action $a$：

1. 复制当前参数，不改变 official model。
2. 临时施加小比例动作：

$$
\theta' = \theta + \epsilon \Delta\theta_a,
$$

其中 $\epsilon \in \{0.05,0.10,0.25\}$。

3. 只在 current train batch 和 train-memory buffer 上测：

```text
CE delta
margin delta
NLL delta
hard-tail CEp99 delta
old-family CE delta
old-stratum CE delta
cover/rank proxy delta
```

4. 立刻 restore 参数。

这不是 future horizon，不是 validation/test，不是 branch-horizon outcome；它是 commit-time legal probe。

## 必须记录

```text
action_id
epsilon
probe_batch_size
probe_memory_size
CE_delta_current
margin_delta_current
hard_tail_CEp99_delta
memory_CE_delta
memory_margin_delta
cover_proxy_delta
probe_cost_ms_q50/q90
restore_error_linf
```

## 判断标准

P4 pass：

```text
legal microprobe TopK64 GradeAB precision >= 0.50
TopK64 V_integrated LCB > 0
TopK64 h240 longrisk UCB <= 0.10
probe_cost_q90 <= 0.50 ms
restore_error_linf <= 1e-8
```

若 P4 cost 过高但 signal 强，标记为 diagnostic，不 official；后续尝试蒸馏成 cheap certificate。

## 可视化

```text
probe epsilon sensitivity curve
probe delta vs true V_integrated scatter
probe cost histogram
memory probe delta vs longrisk scatter
```

## Artifact

```text
p4_legal_immediate_microprobe.csv
p4_probe_topk_frontier.csv
p4_probe_cost_trace.csv
```

---

# 11. P5：Legal rank upper-bound v2

## 目标

用 P3/P4 的 legal features 判断：如果给足 legal 信息，是否能接近 legacy outcome-derived rank。

## 设计

训练三个 diagnostic rankers，只在 calibration split 学习，heldout 评估：

```text
UB-L1: monotone linear ranker
UB-T1: shallow tree ranker
UB-M1: small MLP diagnostic ranker
```

注意：这些 ranker 的输入必须全部来自 Green fields。

评估不是看 train AUC，而是看 heldout accepted region。

## 必须记录

```text
ranker_id
input_feature_groups
red_field_count
train/cal/heldout split id
TopK32/64/87/128 GradeAB precision
TopK V_integrated LCB
TopK h240 longrisk UCB
TopK bad/null UCB
TopK memory fail UCB
coverage
leave-dataset-out pass
leave-stratum-out pass
feature cost
```

## 判断标准

P5 pass：

```text
TopK87 coverage >= 0.03
TopK87 GradeAB precision >= 0.60
TopK87 V_integrated LCB > 0
TopK87 h240 longrisk UCB <= 0.10
bad/null UCB pass
LDO/LSO diagnostic drop <= 0.10
red_field_count = 0
```

P5 strong pass：

```text
TopK64 GradeAB precision >= 0.75
TopK64 V_integrated LCB > 0
h240 longrisk UCB <= 0.05
```

P5 fail：继续 AP0 existing-action selection 主线停止，进入 P7-P9 新 primitive。

## 可视化

```text
ranker TopK frontier
calibration-to-heldout drop plot
feature importance grouped by legal class
rank score vs true V/risk scatter
```

## Artifact

```text
p5_legal_rank_upper_bound.csv
p5_ranker_heldout_trace.csv
p5_leaveout_rank_trace.csv
```

---

# 12. P6：Legacy-to-legal distillation sandbox

## 目标

把 legacy outcome-derived rank 当作 diagnostic teacher，测试 legal student 是否能学到同样排序。这个阶段不能 official。

## 设计

Teacher：

```text
legacy GCERT18 rank
legacy TopK87 diagnostic accepted region
```

Student：

```text
Green-field-only ranker
```

训练目标只在 calibration split 上使用 teacher rank。Official 判断仍以真实 heldout outcome/geometry 判断，而不是 teacher agreement。

## 必须记录

```text
teacher_rank_id
student_rank_id
teacher_red_field_count
student_red_field_count
student_teacher_kendall_tau
student_teacher_topk_overlap
student_true_gradeab_topk_precision
student_true_V_lcb
student_true_longrisk_ucb
```

## 判断标准

P6 pass：

```text
student red_field_count = 0
student TopK87 true GradeAB precision >= 0.60
student V_integrated LCB > 0
student longrisk UCB <= 0.10
```

若 student 只学到 teacher overlap 但真实 outcome 不好，说明 legacy rank 中不可合法复现的部分太多。

## 可视化

```text
teacher vs student rank scatter
TopK overlap curve
student TopK true quality curve
```

## Artifact

```text
p6_legacy_to_legal_distillation_sandbox.csv
p6_student_rank_quality_trace.csv
```

---

# 13. P7：Memory blocker anatomy

## 目标

v9.5.6 显示 memory 是强 blocker。本阶段要查清哪些 action 破坏 old family / old stratum，破坏机制是什么。

## 必须记录

```text
action_id
family_id
old_family_id
old_stratum_id
memory_fail_label
old_family_CE_delta
old_family_margin_delta
old_stratum_CE_delta
old_stratum_margin_delta
memory_gradient_conflict
payload_mass_on_old_family_edges
basis_rank_delta_old_family
cover_entropy_delta_old_family
h240_longrisk
V_integrated
Grade
```

## 分析

```text
1. memory fail 与 h240 longrisk 的条件概率；
2. memory fail 与 V_integrated negative 的关系；
3. memory-safe 但 value-positive 的 action density；
4. memory fail 是否集中在少数 family / edge group；
5. memory fail 是否由 action-AdamW conflict 引起；
6. memory fail 是否由 cover entropy collapse 引起。
```

## 判断标准

P7 pass：

```text
memory failure modes can explain >= 70% high-longrisk actions
and memory-safe filter does not remove all positive actions
```

P7 fail：memory 不是主因，回到 signal/reservoir或cover机制。

## 可视化

```text
memory fail heatmap by family/stratum
old CE delta vs longrisk scatter
memory conflict vs V_integrated scatter
family-level failure Pareto chart
```

## Artifact

```text
p7_memory_blocker_anatomy.csv
p7_memory_family_failure_heatmap.csv
```

---

# 14. P8：APGM1-APGM8 memory-preserving primitives

## 目标

不再微调 APGC。实现一组带明确机制的新动作生成器：先保证 signal 支持，再做 memory 保留，再压 longrisk。

## Primitive family

### APGM1：SNRMaskedResidualUpdate

只保留 SNR 过线的 edge/basis group：

$$
M_g = \mathbf{1}\left[\mu_g^2 > \lambda \frac{\sigma_g^2}{b-1}\right].
$$

$$
\Delta\theta'=M\odot \Delta\theta.
$$

### APGM2：PopulationRiskOffDiagonalGate

只保留 $\Omega_B$ 为正的 group：

$$
M_g=\mathbf{1}[\Omega_{B,g}>0].
$$

### APGM3：MemoryOrthogonalProjection

将动作投影掉与 memory 梯度冲突的方向：

$$
\Delta\theta'
=
\Delta\theta
-
\frac{\langle\Delta\theta,g_{mem}\rangle}{\|g_{mem}\|^2+\epsilon}g_{mem}.
$$

### APGM4：OldFamilyMarginGuard

如果动作会降低 old-family margin，则缩放相关 group：

$$
\Delta\theta'_g=s_g\Delta\theta_g,
$$

其中 $s_g\in[0,1]$ 由 train-memory margin probe 决定。

### APGM5：CoverRankPreservingUpdate

约束 basis effective rank 不下降：

```text
若 predicted rank delta < -tau_rank，则缩小该 edge group payload。
```

### APGM6：SymmetricBoundaryDampedUpdate

使用对称边界，避免单向猛拉：

$$
\Delta\theta'=
\alpha\Delta\theta_{signal}
+
(1-\alpha)\Delta\theta_{AdamWCompatible}
-
\beta\Delta\theta_{memoryConflict}.
$$

### APGM7：SignalMemoryIntersectionPrimitive

只保留同时满足：

```text
SNR pass
Omega pass
memory conflict low
cover collapse low
AdamW compatible
```

的 group。

### APGM8：NegativeControlShuffledPayload

shape-preserving shuffle，用于确认生成器不是凭偶然 pass。

## 必须记录

```text
primitive_id
generated_action_count
payload_hash_missing_count
certificate_hash_missing_count
action_apply_linf_max
SNR_pass_group_fraction
memory_projection_norm_ratio
cover_rank_guard_applied_fraction
payload_norm_ratio_vs_source
AdamW_cosine
feature_compute_ms
payload_apply_ms
```

## 判断标准

P8 implementation pass：

```text
generated_action_count >= 512
payload/certificate hash missing = 0
action_apply_linf_max <= 1e-8
negative control generated = 1
```

P8 science pass 不在本阶段判断，交给 P9。

## Artifact

```text
p8_apgm_primitive_implementation.csv
apgm_payload_trace.csv
apgm_certificate_trace.csv
```

---

# 15. P9：APGM branch-horizon outcome 与 geometry pass

## 目标

判断 APGM 是否真的创造了新的好训练形状。

## 实验设计

对 APGM1-APGM8 每个 primitive 至少 64 actions，推荐 128 actions。

Branches：

```text
RealAPGM
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
```

## 必须记录

```text
primitive_id
action_count
branch_horizon_rows_expected/actual
GradeA/B/C precision
T5 precision
V20/V80/V240 LCB
V_integrated LCB
h240 longrisk UCB
bad/null UCB
memory fail UCB
cover collapse UCB
new positive created rate
longrisk created rate
source-to-generated damage
beats AdamWParallel/bestLR/NoOp/Random
negative_control_pass
```

## 判断标准

APGM weak pass：

```text
GradeB precision >= 0.30
V_integrated LCB > 0
h240 longrisk UCB <= 0.20
bad/null UCB <= relaxed gate
negative control pass = 0
```

APGM official candidate pass：

```text
accepted_count >= 87
coverage >= 0.03
GradeAB precision >= 0.60
V_integrated LCB > 0
h240 longrisk UCB <= 0.10
bad UCB <= 0.05
null UCB <= 0.15
memory fail UCB <= 0.10
negative control pass = 0
```

APGM fail：

```text
best primitive V_integrated LCB <= 0
or h240 longrisk >= 0.50
or negative control is best
```

## 可视化

```text
primitive quality radar
source-to-generated damage heatmap
V vs longrisk frontier
memory fail vs longrisk scatter
negative control comparison
```

## Artifact

```text
p9_apgm_branch_horizon_outcome.csv
p9_apgm_geometry_pass.csv
p9_apgm_damage_matrix.csv
```

---

# 16. P10：Rank-based certificate v5

## 目标

解决 v9.5.5/v9.5.6 中“rank 可能强，但 ECE 差”的问题。不要强行把 rank 解释成概率；直接做 accepted-region 证书。

## 设计

三类 certificate：

```text
CERT-R1: legal rank-only score
CERT-R2: legal rank + conformal risk bound
CERT-R3: legal rank + LCB/UCB accepted-region bound
CERT-R4: APGM primitive-specific certificate
CERT-R5: memory-safe rank certificate
CERT-R6: cost-constrained rank certificate
```

所有 certificate 必须：

```text
red_field_count = 0
threshold frozen on calibration split
heldout evaluated once
leaveout evaluated separately
```

## 必须记录

```text
certificate_id
input_feature_groups
red_field_count
rank_auc
TopK32/64/87/128 GradeAB precision
accepted_count_cal/heldout
coverage_cal/heldout
V_integrated_LCB_cal/heldout
h240_longrisk_UCB_cal/heldout
bad/null UCB
memory fail UCB
ECE
Brier
rank stability
cost q90
```

## 判断标准

Certificate official pass 不要求 ECE 过线，但要求 accepted region 过线：

```text
red_field_count = 0
heldout accepted_count >= 87
heldout coverage >= 0.03
heldout V_integrated LCB > 0
heldout longrisk UCB <= 0.10
heldout bad UCB <= 0.05
heldout null UCB <= 0.15
heldout memory fail UCB <= 0.10
cost_q90 <= 0.50 ms
```

ECE 只作为 calibration quality，不作为唯一否决条件。但如果 ECE 高且 accepted region 不稳，则 fail。

## 可视化

```text
rank reliability plot
accepted-region LCB/UCB plot
coverage-quality curve
ECE vs accepted-quality scatter
```

## Artifact

```text
p10_rank_based_certificate_v5.csv
p10_accepted_region_trace.csv
p10_rank_calibration_dashboard.json
```

---

# 17. P11：Minimal geometry controller

## 目标

从 P5/P9/P10 中选一个最小、可解释、dataset-agnostic 的 controller。

## Controller 形式

最多 5 个 feature groups：

$$
Score(a)=
+w_1 Signal(a)
-w_2 LongRiskProxy(a)
-w_3 MemoryRisk(a)
-w_4 CoverCollapseRisk(a)
-w_5 Cost(a).
$$

权重约束：

$$
w_i\ge 0.
$$

Accept：

$$
Accept(a)=1
\iff
Score(a)\ge \tau
\land
Cost(a)\le C_{max}
\land
ApplyOK(a)=1.
$$

## 必须记录

```text
controller_id
feature_groups
feature_count
red_field_count
threshold_calibration_split
accepted_count_cal/heldout
coverage_cal/heldout
precision_gradeab_cal/heldout
V_integrated_LCB
h240_longrisk_UCB
bad/null UCB
memory_fail_UCB
support_balance
family_balance
cost_q90
```

## 判断标准

P11 pass：

```text
red_field_count = 0
feature_count <= 5
heldout coverage in [0.03, 0.15]
heldout V_integrated LCB > 0
heldout GradeAB precision >= 0.60
heldout h240 longrisk UCB <= 0.10
heldout bad UCB <= 0.05
heldout null UCB <= 0.15
heldout memory fail UCB <= 0.10
support_balance_pass = 1
```

## 可视化

```text
controller accepted-region scatter
family balance heatmap
heldout vs calibration comparison
threshold sensitivity but not tuned on heldout
```

## Artifact

```text
p11_minimal_geometry_controller.csv
p11_controller_accepted_region.csv
```

---

# 18. P12：Selected runtime

## 目标

只对 P11 选中的 controller 测 online runtime。不能再测没有 controller 的 runtime smoke。

## 必须记录

```text
controller_id
runtime_mode
step_count
active_step_count
accepted_action_count
feature_compute_ms_q50/q90
certificate_compute_ms_q50/q90
payload_apply_ms_q50/q90
kernel_launch_count
sync_count
step_time_q50/q90
base_step_time_q50/q90
step_ratio_q90
memory_ratio
selected_runtime_pass
```

## 判断标准

P12 pass：

$$
StepRatio_{q90}\le 1.50,
$$

$$
MemoryRatio\le 1.05.
$$

Strong pass：

$$
StepRatio_{q90}\le 1.20.
$$

## 可视化

```text
runtime stacked bar
step ratio histogram
feature/certificate/payload apply contribution chart
memory trace
```

## Artifact

```text
p12_selected_runtime.csv
p12_runtime_attribution_trace.csv
```

---

# 19. P13：Leaveout validation

## 目标

确认 controller 不是某个 dataset / family / stratum 的偶然规则。

## 设计

```text
leave-dataset-out
leave-seed-out
leave-family-out
leave-stratum-out
leave-time-bucket-out
```

## 必须记录

```text
split_id
heldout_group
accepted_count
coverage
V_integrated_LCB
GradeAB precision
h240 longrisk UCB
bad/null UCB
memory fail UCB
runtime cost
```

## 判断标准

P13 pass：

```text
所有 major splits coverage >= 0.02
macro coverage >= 0.03
macro V_integrated LCB > 0
macro longrisk UCB <= 0.10
no split catastrophic fail
```

不能按 dataset 调 threshold；所有 threshold 从 calibration 冻结。

## 可视化

```text
leaveout forest plot
per-dataset quality table
per-family fail Pareto
```

## Artifact

```text
p13_leaveout_validation.csv
p13_leaveout_forest_plot_data.csv
```

---

# 20. P14：Official paired replay

## 目标

验证 selected controller 的 RealFunctional 是否真的打过强对照。

## Branches

```text
RealFunctionalSelected
AdamWOnly
AdamWParallel
bestLR
NoOp
RandomPayload
ShuffledFunctionalPayload
CertificatePassNoPayload
```

## 必须记录

```text
branch
horizon
CE_delta
NLL_delta
ECE_delta
margin_delta
CEp99_delta
test/train proxy delta
old-family delta
hard-tail delta
runtime delta
beats_adamwparallel
beats_bestlr
beats_noop
beats_random
beats_shuffled
```

## 判断标准

P14 pass：

```text
RealFunctional beats AdamWParallel rate >= 0.60
RealFunctional beats bestLR rate >= 0.55
RealFunctional beats NoOp rate >= 0.70
ShuffledPayload does not pass
CertificatePassNoPayload does not pass
V_integrated LCB > 0
longrisk UCB <= 0.10
```

## 可视化

```text
paired replay branch comparison
win-rate matrix
horizon value curve
shuffled control comparison
```

## Artifact

```text
p14_official_paired_replay.csv
p14_branch_winrate_matrix.csv
```

---

# 21. P15：Short / full training boundary

## 目标

只有 P11-P14 过线后，才跑 short/full training。训练结果不是调参用，而是最终验证。

## 必须记录

```text
dataset
seed
model
controller_id
final_train_acc
final_val_acc
final_test_acc
val_loss_auc
test_nll
test_ece
CEp99
hard_stratum_acc
steps_to_target
time_to_target
sample_efficiency
forgetting_score
runtime_step_ratio
memory_ratio
```

## Baselines

```text
MatchedMLP
AdamWStrongLRGridMLP
QuadraticFeatureMLP
LQ-t2-h256 AdamW only
LQ-t2-h256 + selected functional controller
NoOp functional controller
Shuffled functional controller
```

## 判断标准

Short-run pass：

```text
functional controller improves at least one of:
  steps_to_target
  val_loss_auc
  calibration
  CEp99
  hard-stratum performance
without hurting final test acc / runtime envelope.
```

Full-run pass：

```text
LQ + functional controller beats LQ AdamW-only and matched strong MLP on pre-registered metrics;
no dataset-specific threshold;
no catastrophic seed;
runtime still within envelope.
```

## 可视化

```text
accuracy curves
loss curves
ECE/NLL curves
sample efficiency curves
hard-tail CEp99 curve
forgetting curve
runtime curve
```

## Artifact

```text
p15_short_full_training.csv
p15_training_curves.json
```

---

# 22. P16：Base-Acc Sentinel continuation

## 目标

继续监控 base 是否崩，但不用于 controller。

## 必须记录

```text
MNIST/Fashion-MNIST/KMNIST
seeds 0..9
LQ-t2-h256
MatchedMLP
AdamWStrongLRGridMLP
QuadraticFeatureMLP
mean/std test acc
catastrophic fail
base_acc_used_for_controller = 0
```

## 判断标准

Sentinel pass：

```text
LQ_catastrophic_fail = 0
base_acc_used_for_controller = 0
```

Sentinel 不决定 v9.5.7 success。

## Artifact

```text
p16_base_acc_sentinel.csv
```

---

# 23. 统一记录表字段

每个 action row 必须记录：

```text
action_id
candidate_id
event_id
source
primitive_id
payload_hash
certificate_hash
split_id
dataset
seed
step
family_id
bucket_id
horizon
apply_error_linf
payload_norm
payload_linf
payload_sparsity
action_adamw_cosine
SNR_group_mean
SNR_group_pass_fraction
Omega_B
memory_conflict_score
cover_collapse_proxy
feature_compute_ms
certificate_compute_ms
payload_apply_ms
V20/V80/V240
V_integrated
bad/null
h240_longrisk
memory_fail
cover_collapse
curvature_rise
GradeA/B/C/D/E
T5_label
accepted_by_controller
selected_runtime_cost
```

所有字段必须附带：

```text
field_legality_class
field_source
available_at_commit
uses_outcome_field
uses_dataset_name
uses_validation_or_test
```

---

# 24. Dashboard 与可视化总清单

v9.5.7 必须生成以下 dashboard 数据：

```text
1. Field legality heatmap；
2. Legacy rank ablation waterfall；
3. Legal vs legacy TopK quality curves；
4. Coverage vs value vs longrisk frontier；
5. Grade density and overlap matrix；
6. SNR / Omega / memory / cover feature scatter；
7. Microprobe epsilon sensitivity；
8. Memory blocker family heatmap；
9. APGM source-to-generated damage matrix；
10. Certificate accepted-region LCB/UCB chart；
11. Controller calibration-to-heldout comparison；
12. Selected runtime attribution；
13. Leaveout forest plot；
14. Paired replay branch win-rate matrix；
15. Base-Acc Sentinel curves。
```

---

# 25. Route decision table

最终 route 必须落入以下之一：

```text
R0-BoundaryRegression:
  v9.5.6 boundary 没复现。

R1-OutcomeLeakageConfirmedLegalSignalAbsent:
  legacy rank 确认靠 outcome-derived fields，legal feature/rank完全复现不了。

R2-LegalSignalExistsControllerFail:
  legal TopK 有信号，但 accepted-region / heldout / support 不过。

R3-MemoryPrimitiveRequired:
  memory blocker 成立，AP0 selection 不足，需 APGM。

R4-APGMGeneratedFrontierFail:
  APGM 工程闭合，但生成动作 value-negative / high-longrisk。

R5-RankControllerPassRuntimeFail:
  controller 过线，但 selected runtime 不过。

R6-SystemPassPairedReplayBlocked:
  controller/runtime 过线，但 paired replay 没打过 controls。

R7-StrictPureKANFunctionalLocalSuccess:
  controller/runtime/paired replay/leaveout 过线，但 short/full 还未过。

R8-FullFunctionalCandidateReady:
  short/full boundary 也过线，可进入 external-ready hard baseline stage。
```

---

# 26. 停止条件

## 立即停止 AP0 existing-action selection 的条件

```text
P5 legal rich upper-bound fail；
P6 legal distillation fail；
TopK64/87 V_integrated LCB <= 0；
longrisk UCB >= 0.30；
```

此时不再加 AP0 scalar features。

## 立即停止 APGM generator 的条件

```text
APGM8 negative control 是 best；
best APGM V_integrated LCB <= 0；
best APGM h240 longrisk >= 0.50；
source-to-generated damage LCB << 0；
```

此时不要调 APGM 阈值，进入 primitive family redesign。

## 打开 selected runtime 的条件

```text
P11 controller pass = 1
red_field_count = 0
coverage >= 0.03
V_integrated LCB > 0
longrisk/bad/null/memory pass
```

## 打开 paired replay 的条件

```text
P12 selected runtime pass = 1
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
```

---

# 27. v9.5.7 成功标准

## Minimum useful success

```text
P1 field legality ledger pass；
legacy rank leakage fully explained；
P3/P4/P5 至少给出一个 legal TopK signal 或明确证明 AP0 selection dead-end；
P7 memory blocker anatomy pass；
APGM implementation pass；
no fake/proxy/outcome-at-commit/dataset tuning。
```

## Scientific success

```text
要么 legal existing-action rank TopK87 过 quality gate；
要么 APGM generated frontier 过 weak pass；
并且明确 memory / signal / reservoir 机制。
```

## System success

```text
minimal geometry controller pass；
selected runtime pass；
step_ratio_q90 <= 1.50；
```

## Strict functional local success

```text
system pass；
leaveout pass；
official paired replay pass；
RealFunctional beats AdamWParallel / bestLR / NoOp / Random；
shuffled controls fail。
```

---

# 28. 最终执行建议

v9.5.7 应该优先并行跑以下四组，而不是顺序等待：

```text
A. P1-P2:
   field legality + target density，几小时内完成。

B. P3-P6:
   legal signal / microprobe / rank upper-bound，和 A 并行。

C. P7-P9:
   memory anatomy + APGM primitive，和 B 并行预生成，但 official 等 P1 legality。

D. P10-P12:
   certificate/controller/runtime 代码先 dry-run，只有上游 pass 才 official。
```

这样即使最终仍失败，也会同时回答：

```text
1. 是不是只有 outcome leakage 才能找到好动作？
2. legal action-effect signal 是否真的没有？
3. memory-preserving primitive 是否能减少 longrisk？
4. rank-based controller 是否比 ECE-based certificate 更适合？
5. 如果失败，下一轮应该重设目标、重设 primitive，还是重设 runtime？
```

---

# 29. 最后一条判断

v9.5.6 之后，不能再把“GCERT18 TopK 很强”当好消息直接推进。它更像一个答案泄漏的提示器：它告诉我们“好动作长什么样”，但没有告诉我们“训练当下怎么合法找到它”。

v9.5.7 的关键就是把这个提示器拆掉：

$$
\boxed{
\text{用合法的 action-effect、memory、signal、cover、cost 信息，重建一个不依赖 outcome 的好动作选择规则。}
}
$$
