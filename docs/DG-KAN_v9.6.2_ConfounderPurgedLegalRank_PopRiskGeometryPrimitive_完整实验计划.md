# DG-KAN v9.6.2 Confounder-Purged Legal Rank / Population-Risk Geometry Primitive / Parallel Closure 完整实验计划

> 本计划基于 v9.6.1 `Group Pocket Causal Intervention / Group-Invariant Controller / Memory-Safe Primitive` 的真实执行结果制定。  
> v9.6.1 的 terminal route 是：
>
> ```text
> route = R2-GroupPocketConfounded_StopPayloadPocketRoute
> base_candidate = LQ-t2-h256
> success_v9610_strict_purekan_functional = False
> success_v9610_full_functional = False
> success_v9610_external_ready = False
> ```
>
> v9.6.2 不再沿着 `payload0 pocket` 做阈值修补，也不再继续盲目新增 APGE 小变体。  
> 本轮核心问题是：
>
> $$
> \boxed{
> \text{如果 payload0 不是因果机制，那么训练当下到底有没有一个跨 group 稳定的好动作识别规则，或者我们必须重建动作生成方式？}
> }
> $$

---

# 0. v9.6.1 独立判断

v9.6.1 的主要进展不是 system pass，而是把 v9.6.0 的 payload0 group pocket 进一步拆开。

v9.6.0 看到：

```text
payload0 removal precision drop = 0.8505747126436781
best pairwise ranker TopK87 precision = 0.8505747126436781
best pairwise LDO drop = 0.3563218390804597
best APGD V LCB = -0.875224386072408
best APGD longrisk UCB = 0.8560881119635937
```

这说明 rank signal 很强，但它集中在 payload0 pocket，且 generated primitive 仍失败。

v9.6.1 做了更重要的一步：它不再只看 “payload0 去掉后 precision 掉多少”，而是做 intervention clone replay 和 matched-pair attribution。

关键实测：

```text
source_action_count = 104
intervention_clone_count = 520
branch_horizon_rows = 12480 / 12480
matched_pair_count_total = 520

payload0_normalized_GradeAB_precision = 0.009615384615384616
memory_projection_GradeAB_precision = 0.009615384615384616
memory_projection_longrisk_rate = 0.8557692307692307

mean_V_lift = -0.5198887260597593
mean_GradeAB_lift = -0.6134615384615385
payload0_causal_lift_supported = 0
payload0_confounded_supported = 1
```

因此，v9.6.1 的正确信息不是 “payload0 很重要，所以继续围绕 payload0 调参”，而是：

$$
\boxed{
\text{payload0 看起来像高分区域，但 intervention 后不能产生好动作；它更像混杂口袋，不是可利用机制。}
}
$$

这意味着 v9.6.2 必须停止：

```text
调 payload_norm_bucket；
调 GIR9 / VR4 threshold；
调 RC7 accepted count；
把 payload0 当作 action 生成规则；
继续把 TopK pocket 当成 controller。
```

---

# 1. 当前项目状态

## 1.1 已经可信的东西

目前可信的基础设施包括：

```text
1. canonical AP0 outcome universe 已经闭合；
2. old outcome table 已经 quarantine；
3. no-transform replay semantics 已经闭合；
4. action payload / certificate / action apply replay 已经多轮闭合；
5. branch-horizon materializer 能稳定产出 12288+ rows；
6. Base-Acc Sentinel 继续健康，但不用于 controller；
7. no fake / no proxy / no CPU offload audit 持续通过；
8. dataset_name 没有用于 selector / controller。
```

## 1.2 仍未成功的东西

```text
1. 没有 selected minimal controller；
2. 没有 selected runtime；
3. 没有 system_legal_controller_pass；
4. 没有 official leaveout；
5. 没有 official paired replay；
6. 没有 short/full functional training；
7. 没有 external-ready Beyond-MLP evidence。
```

## 1.3 当前真正的 blocker

v9.6.1 后，primary blocker 不能再写成 “payload0 pocket unresolved”。更准确是：

$$
\boxed{
\text{已有 legal rank 能找到局部好区域，但该区域是混杂口袋；我们还没有跨 group 稳定的 rank，也没有能生成好几何动作的 primitive。}
}
$$

具体拆成四个问题：

```text
B1. Existing-action rank:
    TopK 很强，但 LDO / LSO / group leaveout 仍不稳。

B2. Accepted-region conversion:
    TopK 强不等于 frozen heldout accepted region 强。
    RC7 heldout GradeAB precision 只有 0.287356，longrisk UCB 0.369781。

B3. Generated-action quality:
    APGE1-APGE8 工程闭合，但 best APGE6 GradeAB precision 只有 0.03125，
    V LCB = -0.869364，h240 longrisk UCB = 0.882533。

B4. Geometry mechanism:
    memory / offdiag / longrisk created 是主损伤模式；
    但我们还没有把 population-risk signal、cover stability、memory safety 转成可用 action。
```

---

# 2. v9.6.2 总体目标

v9.6.2 的总体目标不是再修一个 pocket，也不是继续堆一个新 primitive family。  
本轮要并行回答两个大问题：

## 2.1 Existing-action route

$$
\boxed{
\text{能否在不使用 outcome-derived 字段、不按 dataset 调参的前提下，构造一个跨 group 稳定的 accepted region？}
}
$$

也就是从当前：

```text
strong pooled TopK diagnostic
```

推进到：

```text
group-stable heldout accepted region
```

## 2.2 Generated-action route

$$
\boxed{
\text{能否直接生成 memory-safe、offdiag-safe、signal-channel-supported 的 functional update action？}
}
$$

也就是从当前：

```text
合法 payload generator
```

推进到：

```text
好训练形状 generator
```

---

# 3. 本轮硬约束

继续遵守：

```text
no teacher
no self-teacher
no distillation as training trick
no loss modification
no label smoothing / focal / margin auxiliary loss
no sampler / class weight
no CPU offload
no fake / proxy rows
KAN path 不使用 PyTorch loss.backward graph
functional update 是 update rule，不是 loss trick
official controller 不使用 dataset_name 分支
official controller 不使用 validation/test/future outcome at commit
official controller 不使用 outcome-derived fields
old v9.3.5 table 不进入 official
```

允许：

```text
dataset / stratum / family / payload bucket 作为 diagnostics 和 leaveout axes；
offline calibration split 训练 legal rank / certificate；
heldout split 冻结评估；
legal green fields 作为 commit-time features；
outcome labels 只用于训练和评估，不作为 commit-time input；
APGF/APGI/APGR/APGE 等 generator 离线设计，但 official commit-time spec 必须不读未来 outcome。
```

---

# 4. 成功标准

## 4.1 Existing-action minimal controller pass

至少满足：

$$
N_{accept} \ge 87
$$

因为 canonical action count 是 2876，coverage 下限 $0.03$ 对应：

$$
\lceil 0.03 \times 2876 \rceil = 87.
$$

正式 gate：

$$
Coverage \in [0.03, 0.15]
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
MaxGroupShare \le 0.35
$$

$$
LDO\_drop \le 0.10
$$

$$
LSO\_drop \le 0.10
$$

$$
LeavePayloadBucketDrop \le 0.10
$$

## 4.2 Generated primitive pass

每个 generated primitive 至少有：

```text
generated actions >= 64
branch-horizon completion = 1.0
quality audit pass = 1
action apply L∞ max <= 1e-6
```

weak scientific pass：

$$
Precision_{GradeAB} \ge 0.30
$$

$$
LCB(V_{integrated}) > 0
$$

$$
UCB(LongRisk_{240}) \le 0.10
$$

$$
NewPositiveCreatedRate \ge 0.10
$$

strong pass：

$$
Precision_{GradeAB} \ge 0.50
$$

$$
LCB(V_{integrated}) > 0.05
$$

$$
UCB(LongRisk_{240}) \le 0.05
$$

$$
MemoryFailUCB \le 0.05
$$

$$
OffdiagFailUCB \le 0.05
$$

## 4.3 System runtime pass

只有在 selected controller 存在时才测 official runtime：

$$
StepRatio_{q90} \le 1.50
$$

$$
MemoryRatio \le 1.05
$$

$$
PayloadApplyError_{L\infty} \le 10^{-6}
$$

---

# 5. 核心假设

## H1：payload0 不是可利用机制，后续应停止 payload-pocket route

v9.6.1 已经强烈支持 H1，但 v9.6.2 仍需要把它写成 formal stop condition。

成立标准：

```text
payload0 intervention clones GradeAB precision <= 0.05
payload0 intervention clones longrisk UCB >= 0.25
matched-pair GradeAB lift <= 0
matched-pair V lift <= 0
```

若成立：

```text
禁止把 payload0 / payload_norm_bucket 作为 official selector；
payload_norm_bucket 只能作为 stratification / leaveout axis；
不能再做 payload0 threshold patch。
```

## H2：当前 legal rank 的问题是 group confounding，不是 value signal 缺失

成立标准：

```text
pooled TopK87 GradeAB precision >= 0.75
pooled V LCB > 0
pooled longrisk UCB <= 0.05
but
LDO/LSO/drop > 0.10
or MaxGroupShare > 0.35
```

这表示 rank 有用，但不是通用规则。

## H3：只要把 value rank、risk veto、memory veto 拆开，并做 group-balanced selection，existing-action controller 仍可能过线

成立标准：

```text
GroupBalancedRank TopK87 GradeAB precision >= 0.75
V LCB > 0
longrisk UCB <= 0.05
max group share <= 0.35
LDO/LSO drop <= 0.10
```

若 H3 成立，优先走 existing-action controller，不急着依赖 generated primitive。

## H4：如果 legal rank 经过 deconfounding 仍失败，则 AP0 existing-action selection 不是主路线

成立标准：

```text
best deconfounded ranker TopK87 GradeAB precision < 0.60
or V LCB <= 0
or longrisk UCB > 0.10
or LDO/LSO drop > 0.25
```

若 H4 成立：

```text
停止 AP0 existing-action rank 主线；
generated primitive / geometry-preserving action 变成主线。
```

## H5：APGE/APGD/APGA 失败来自 memory/offdiag/longrisk，不是 replay/materializer

成立标准：

```text
APGE/APGD/APGA implementation pass = 1
branch-horizon pass = 1
dominant damage mode contains memory/offdiag/longrisk
longrisk created rate >= 0.50
V LCB < 0
```

v9.6.1 基本已支持，但本轮要拆清楚到底是：

```text
source action OOD；
delta direction OOD；
scale too大；
memory projection 错；
offdiag population-risk fail；
cover collapse；
old-family interference；
```

## H6：population-risk / SNR gate 必须从 scalar SNR 变成 group-structured gate

Generalization 相关工作提示，真正有用的是 signal channel，不是训练集残差 reservoir。因此不能只用一个 payload norm 或一个 SNR 标量。

成立标准：

```text
edge/basis/family structured SNR rank
优于 scalar SNR rank
并且 longrisk UCB 更低。
```

## H7：Deep Manifold 风格的 cover / boundary / fixed-point 约束能解释 longrisk

成立标准：

```text
cover collapse / basis rank drop / old-family memory fail / offdiag fail
对 longrisk 的解释率 >= 0.70
并且 memory-safe / cover-safe subset 的 longrisk 显著低于全集。
```

---

# 6. 实验并行执行布局

本轮不再串行跑一个大 runner 后才知道下一个 blocker。v9.6.2 分成四条并行线：

```text
Track A: Existing-action legal rank deconfounding
Track B: Accepted-region / certificate conversion
Track C: Generated primitive failure autopsy and reset
Track D: Runtime / leaveout / short-full boundary
```

Track A 和 Track C 可同时跑；Track B 依赖 Track A 的 candidate rank；Track D 只在 Track B 或 Track C 有 selected controller 后打开 official runtime。

---

# 7. P0 Boundary Reproduction and Ledger Freeze

## 目标

确认 v9.6.1 boundary 稳定复现，并冻结本轮所有表的 universe。

## 输入

```text
v9.6.1 route_decision
v9.6.1 intervention_clone_trace
v9.6.1 matched_pair_attribution
v9.6.1 rank/certificate artifacts
v9.6.1 APGE outcomes
canonical AP0 outcome universe
Geometry Outcome Ledger
Base-Acc Sentinel
```

## 必须记录

```text
source_route_v9610
system_legal_controller_pass_v9610
payload0_confounded_supported_v9610
H1_falsified_v9610
matched_pair_count_total_v9610
best_ranker_v9610
best_ranker_TopK87_precision_v9610
best_ranker_LDO_drop_v9610
best_certificate_v9610
best_apge_primitive_v9610
best_apge_V_lcb_v9610
best_apge_longrisk_ucb_v9610
```

## Pass 标准

```text
p0_pass = 1
no_fake = 1
no_proxy = 1
old_table_official_use = 0
field_legality_ledger_pass = 1
```

## 可视化

```text
v9.5.8 -> v9.6.1 route waterfall
TopK precision vs LDO/LSO drop trend
Generated primitive V/longrisk trend
```

---

# 8. P1 Confounder Ledger and Group Pocket Stop Audit

## 目标

把 payload0 彻底降级为 diagnostic axis，禁止继续作为可利用机制。

## 实验内容

构建 `confounder_ledger_v9620.csv`，每个 action 记录：

```text
action_id
candidate_id
event_id
dataset_id
seed_id
family_id
stratum_id
step_bucket
horizon_bucket
payload_norm_bucket
payload_linf_bucket
payload_direction_bucket
state_nll_bucket
hard_tail_bucket
memory_fail_bucket
offdiag_fail_bucket
cover_bucket
ranker_score
GradeAB_label
V_integrated
longrisk_240
bad
null
```

同时复算：

```text
payload0 intervention precision
payload0 matched-pair V lift
payload0 matched-pair GradeAB lift
payload0 normalized longrisk
memory projection longrisk
```

## 判断标准

如果：

$$
Precision_{payload0\_intervention} < 0.05
$$

且：

$$
MeanLift_{GradeAB} < 0
$$

且：

$$
MeanLift_V < 0
$$

则：

```text
payload_pocket_route_stop = 1
payload_norm_bucket_allowed_as_selector = 0
payload_norm_bucket_allowed_as_stratification = 1
```

## 可视化

```text
clone type vs GradeAB precision bar chart
clone type vs longrisk bar chart
matched-pair V lift distribution
payload bucket x family x step heatmap
```

---

# 9. P2 Hidden Confounder Search Beyond Payload0

## 目标

payload0 不是因果机制后，要找 rank pocket 的真正混杂来源。

## 方法

对每个 suspect axis 做 group drop / counterfactual matching：

```text
dataset_id
seed_id
family_id
stratum_id
step_bucket
event_phase
payload_direction_bucket
state_nll_bucket
hard_tail_bucket
memory_fail_bucket
offdiag_fail_bucket
cover_bucket
```

每个 axis 记录：

```text
group_count
max_topK_group_share
drop_if_removed
identified_drop_fraction
matched_pair_count
within_group_AUC
between_group_AUC
SMD_before_matching
SMD_after_matching
```

## Pass / Fail

若某 axis 满足：

```text
matched_pair_count >= 300
SMD_after_matching <= 0.10
drop_if_removed >= 0.25
within_group_AUC >= 0.75
between_group_AUC <= 0.60
```

则该 axis 是 strong confounder。

若没有 axis strong：

```text
confounder_unresolved = 1
```

## 可视化

```text
axis drop bar chart
SMD before/after matching plot
within-group vs between-group AUC scatter
TopK group share Pareto chart
```

---

# 10. P3 Group-Balanced Target Map v2

## 目标

确认是否存在足够密度、足够干净、跨 group 支持的目标区域。

## 目标标签

不再只用 GradeAB。并行评估：

```text
T1 = GradeAB
T2 = ValuePositiveNoLongRisk
T3 = MemorySafeValuePositive
T4 = CoverStableValuePositive
T5 = OffdiagSafeValuePositive
T6 = GradeAB ∧ MemorySafe
T7 = GradeAB ∧ OffdiagSafe
T8 = ValuePositiveNoLongRisk ∧ group_support
T9 = V20/V80/V240 vector-positive relaxed target
T10 = PopulationRiskSignalPositive target
```

## 每个 target 记录

```text
global_count
global_coverage
coverage_lcb
min_group_density_dataset
min_group_density_stratum
min_group_density_family
min_group_density_payload_bucket
positive_group_coverage
V_integrated_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_fail_UCB
offdiag_fail_UCB
cover_collapse_UCB
```

## Official density pass

```text
global_count >= 87
global_coverage >= 0.03
min_group_density_dataset >= 0.02
min_group_density_stratum >= 0.02
positive_group_coverage >= 0.80
V_integrated_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
```

## 可视化

```text
target count vs V_LCB scatter
target count vs longrisk_UCB scatter
group density heatmap
V20/V80/V240 ternary plot
memory/offdiag fail stacked bars
```

---

# 11. P4 Legal Feature Deconfounding v2

## 目标

把当前 “value 强但 longrisk 高” 的 feature 拆成 value feature、risk feature、memory feature、cover feature。

## Feature groups

```text
F1 linearized action effect:
  ControlTransferImprovement
  estimated_delta_CE
  estimated_delta_margin
  action_gradient_alignment

F2 population-risk / SNR:
  per-edge mean gradient
  per-edge gradient variance
  group SNR
  offdiag agreement
  leave-one-out transfer score

F3 memory:
  old_family_gradient_conflict
  old_stratum_response
  memory_probe_delta
  old-family margin reserve

F4 cover:
  basis effective rank delta estimate
  cover entropy delta estimate
  hard-tail cover concentration

F5 curvature / fixed point:
  local Lipschitz proxy
  jacobian spectral proxy
  CEp99 tail delta proxy

F6 cost:
  feature compute ms
  payload apply ms
  kernel count estimate
```

## 必须记录

每个 feature 在 full / within-group / leaveout 上：

```text
AUC_GradeAB
AUC_ValuePositiveNoLongRisk
AUC_longrisk
TopK64 precision
TopK87 precision
TopK64 longrisk_UCB
TopK87 longrisk_UCB
within_group_AUC_mean
within_group_AUC_min
LDO_drop
LSO_drop
leave_payload_bucket_drop
feature_cost_ms_q90
```

## Pass 标准

单 feature 不要求 official，但至少一个组合应满足：

```text
TopK87 GradeAB precision >= 0.70
V_LCB > 0
longrisk_UCB <= 0.10
LDO/LSO drop <= 0.25
feature_cost_ms_q90 <= 0.50
```

若无组合达到该 weak gate，则 existing-action rank 主线降级。

## 可视化

```text
feature AUC vs TopK precision scatter
feature TopK longrisk vs V_LCB plot
feature cost vs utility Pareto
within-group AUC heatmap
```

---

# 12. P5 Two-Head Ranker: Value Rank + Risk/Memory Veto

## 目标

不再用一个分数同时承担 value 和 risk。构造两头评分：

$$
S_{value}(a)
$$

$$
S_{risk}(a)
$$

接受条件：

$$
Accept(a)=1
\iff
S_{value}(a) \ge q_v
\land
S_{risk}(a) \le q_r
\land
S_{memory}(a) \le q_m
\land
S_{cover}(a) \le q_c
\land
Cost(a) \le C_{max}
$$

## Ranker candidates

```text
RANK1 group-balanced linear ranker
RANK2 monotone two-head score
RANK3 pairwise within-group ranker v4
RANK4 group-adversarial ranker
RANK5 invariant risk minimization ranker
RANK6 conformal group-balanced ranker
RANK7 value-rank + memory veto
RANK8 value-rank + offdiag veto
RANK9 value-rank + memory + offdiag + cover veto
```

## 训练与评估

```text
calibration split: train ranker / freeze thresholds
heldout split: evaluate
leave-dataset-out: no dataset branch
leave-stratum-out
leave-family-out
leave-payload-bucket-out
```

## Pass 标准

```text
TopK87 GradeAB precision >= 0.75
V_LCB > 0.05
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
max_group_share <= 0.35
LDO_drop <= 0.10
LSO_drop <= 0.10
leave_payload_bucket_drop <= 0.10
```

## 可视化

```text
value score vs risk score quadrant plot
accepted region overlay plot
leaveout drop bar chart
per-group precision/longrisk heatmap
calibration vs heldout reliability plot
```

---

# 13. P6 Rank-Safe Certificate v10

## 目标

把 ranker 变成可 frozen、可 audit、可运行时执行的 certificate。  
这个 certificate 不要求输出概率；它只要可靠地定义 accepted region。

## Certificate form

$$
Cert(a)=
\left(
S_{value},
S_{risk},
S_{memory},
S_{cover},
S_{cost},
GroupSupport
\right)
$$

接受规则：

$$
Accept(a)=1
\iff
S_{value}\ge \tau_v
\land
S_{risk}\le \tau_r
\land
S_{memory}\le \tau_m
\land
S_{cover}\le \tau_c
\land
S_{cost}\le \tau_c'
\land
GroupSupport\ge \tau_s
$$

## 必须记录

```text
certificate_id
feature_count
red_field_count
green_field_count
outcome_field_used = 0
dataset_name_used = 0
future_outcome_used = 0
thresholds_frozen = 1
accepted_count_cal
accepted_count_heldout
coverage_heldout
GradeAB_precision_heldout
V_LCB_heldout
longrisk_UCB_heldout
bad_UCB_heldout
null_UCB_heldout
LDO_drop
LSO_drop
max_group_share
feature_cost_ms_q90
```

## Pass 标准

同 P5 official gate。

## 可视化

```text
certificate pass/fail Sankey
threshold sensitivity curve
accepted count vs V/risk Pareto
per-group accepted density
```

---

# 14. P7 Existing-Action Minimal Controller

## 目标

只有 P6 pass 后才能打开。把 certificate 变成真实 online controller candidate。

## Controller 输入

```text
current state
candidate/action payload metadata
legal green features
rank-safe certificate
cost estimate
```

禁止：

```text
outcome-derived V_integrated
risk_score derived from future branch-horizon
dataset_name branch
validation/test feature
old table
```

## 必须记录

```text
controller_id
accepted_count
coverage
precision_GradeAB
V_LCB
longrisk_UCB
bad_UCB
null_UCB
support_family_count
support_stratum_count
LDO_drop
LSO_drop
feature_cost_ms_q90
payload_apply_ms_q90
```

## Pass 标准

```text
system_controller_candidate_pass = 1
```

进入 P14 runtime。

---

# 15. P8 APGE/APGD/APGA Damage Autopsy v2

## 目标

解释为什么 generated primitives 一直生成 value-negative / high-longrisk actions。

## 对每个 generated family 记录

```text
primitive_id
source_distribution
payload_norm_MMD
payload_direction_MMD
feature_MMD_to_GradeAB
nearest_neighbor_GradeAB_distance
new_positive_created_rate
longrisk_created_rate
memory_fail_rate
offdiag_fail_rate
cover_collapse_rate
V_LCB
Damage_V_LCB
dominant_damage_mode
```

## Damage modes

```text
D1 scale-too-large
D2 direction-OOD
D3 memory-offdiag-longrisk-created
D4 cover-collapse
D5 old-family-conflict
D6 signal-channel-mismatch
D7 risk-veto-ineffective
D8 no-positive-created
D9 source-action-bad
```

## Pass 标准

不是 promotion gate。它必须输出一个明确 route：

```text
AP generated route =
  source_OOD
  delta_direction_OOD
  memory_offdiag_longrisk
  cover_collapse
  signal_channel_mismatch
  ranker_mimic_failure
```

## 可视化

```text
generated vs existing feature PCA
MMD/KS table
damage mode stacked bars
nearest-neighbor distance to GradeAB histogram
```

---

# 16. P9 APGF1-APGF8 Population-Risk Geometry Primitive

## 目标

不再做 “合法 payload builder”。新 primitive 必须显式约束：

```text
signal channel
memory safety
offdiag population risk
cover stability
longrisk veto
cost
```

## Primitive family

```text
APGF1-PopRiskDiagonalSNRGate
  使用 per-parameter 或 per-edge group SNR：
  update only if μ² > σ² / (b - 1)

APGF2-EdgeGroupSNRMask
  对 KAN edge / basis group 计算 coherent gradient support；
  只更新多样本一致支持的 edge group。

APGF3-MemoryAnchoredResidual
  在 old-family gradient-sensitive directions 上做投影，
  避免遗忘旧 family / old stratum。

APGF4-OffdiagPopulationRiskGuard
  显式约束 offdiag population-risk fail；
  对 self-block-only value 做惩罚。

APGF5-CoverEntropyPreservingDelta
  限制 basis effective rank drop 和 hard-tail cover entropy collapse。

APGF6-SymmetricBoundaryCorrection
  生成正向修正和边界反向修正，减少单向 drift。

APGF7-RankImitationWithRiskVeto
  模仿 P5/P6 selected high-rank existing actions，
  但只用 legal green features，并加 memory/offdiag veto。

APGF8-NegativeControlShuffled
  shape-preserving shuffled payload negative control。
```

## 生成动作数量

```text
actions_per_primitive = 64
primitive_count = 8
generated_actions_expected = 512
```

## Materializer

```text
branch_count >= 6
horizon_count = 3
branch_horizon_rows_expected = 512 * branch_count * horizon_count
quality_audit_pass = 1
```

## Pass 标准

Weak pass：

```text
best_APGF_GradeAB_precision >= 0.30
best_APGF_V_LCB > 0
best_APGF_longrisk_UCB <= 0.10
new_positive_created_rate >= 0.10
```

Strong pass：

```text
best_APGF_GradeAB_precision >= 0.50
best_APGF_V_LCB > 0.05
best_APGF_longrisk_UCB <= 0.05
memory_fail_UCB <= 0.05
offdiag_fail_UCB <= 0.05
```

## 可视化

```text
APGF primitive outcome bar chart
new positive vs longrisk scatter
memory/offdiag/cover fail heatmap
source-to-generated damage arrows
```

---

# 17. P10 APGF Certificate and Generated-Action Controller

## 目标

如果 APGF 产生了 positive frontier，建立 generated-action certificate。

## 记录

```text
primitive_id
certificate_id
certificate_green_fields
red_field_count
TopK64 GradeAB precision
TopK87 GradeAB precision
accepted_count
coverage
V_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
cover_UCB
LDO/LSO/leave-family/leave-payload drop
```

## Pass 标准

同 P6/P7，但针对 generated actions。

---

# 18. P11 Decision Route

## 目标

根据 P5-P10 结果明确下一步，而不是继续无边界 patching。

## Route table

```text
R1-existing-controller-pass:
  P6/P7 pass
  -> open selected runtime

R2-generated-controller-pass:
  P9/P10 pass
  -> open selected runtime

R3-existing-rank-signal-but-group-unstable:
  P5 strong pooled but LDO/LSO fail
  -> continue deconfounding only if new confounder identified

R4-generated-primitive-longrisk-fail:
  APGF still longrisk high
  -> stop primitive variants, revisit action objective

R5-no-legal-signal-no-generator:
  existing and generated both fail
  -> revisit functional update primitive at base training rule level

R6-runtime-blocked:
  controller pass but runtime fail
  -> optimize selected path only
```

---

# 19. P12 Selected Runtime

## 前提

P7 或 P10 pass。

## 测量

```text
selected_controller_id
selected_action_count
feature_compute_ms_q50/q90/q99
certificate_compute_ms_q50/q90/q99
payload_apply_ms_q50/q90/q99
kernel_count
sync_count
step_ratio_q50/q90/q99
memory_ratio
zero_candidate_launch_count
active_step_launches_q90
```

## Pass 标准

$$
StepRatio_{q90}\le1.50
$$

$$
MemoryRatio\le1.05
$$

$$
PayloadApplyError_{L\infty}\le10^{-6}
$$

---

# 20. P13 Official Leaveout and Paired Replay Boundary

## 前提

P12 pass。

## Leaveout axes

```text
dataset
seed
stratum
family
payload_bucket
step_bucket
```

## Paired replay branches

```text
RealFunctional
AdamWOnly
AdamWParallel
BestLR
NoOp
RandomPayload
ShuffledFunctionalPayload
CertificatePassNoPayload
```

## 记录

```text
beats_AdamWParallel_rate
beats_BestLR_rate
beats_NoOp_rate
beats_Random_rate
shuffle_control_fail_rate
V20/V80/V240 LCB
CEp99_delta
margin_p10_delta
NLL_delta
ECE_delta
memory_forgetting_delta
hard_tail_delta
runtime_ratio
```

## Pass 标准

```text
beats_AdamWParallel_rate >= 0.60
beats_BestLR_rate >= 0.55
shuffle_control_fail_rate >= 0.95
LDO/LSO pass = 1
```

---

# 21. P14 Short/Full Training Boundary

## 前提

P13 pass。

## 注意

本阶段仍不能按 dataset 调参。dataset 只用于诊断和 leaveout。

## 记录

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
robust_noise_acc
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

## Pass 标准

不能只靠 acc。需要：

```text
functional KAN beats KAN base on at least one non-acc metric
functional KAN does not worsen ECE/NLL/CEp99
functional KAN beats AdamWParallel/bestLR in paired replay
functional KAN step_ratio_q90 <= 1.50
```

---

# 22. 本轮必须落盘的 artifacts

```text
p0_boundary_reproduction_v9620.csv
p1_confounder_ledger_v9620.csv
p1_payload_pocket_stop_audit.csv
p2_hidden_confounder_search.csv
p3_group_balanced_target_map_v2.csv
p4_legal_feature_deconfounding_v2.csv
p5_two_head_ranker_value_risk_memory.csv
p6_rank_safe_certificate_v10.csv
p7_existing_action_minimal_controller.csv
p8_generated_damage_autopsy_v2.csv
p9_apgf_population_risk_geometry_primitive.csv
p9_apgf_branch_horizon_outcome.csv
p10_apgf_certificate_controller.csv
p11_decision_route_v9620.json
p12_selected_runtime.csv
p13_leaveout_paired_replay_boundary.csv
p14_short_full_boundary.csv
no_fake_proxy_audit_v9620.csv
field_legality_ledger_v9620.csv
run_manifest_v9620.json
failure_taxonomy_v9620.csv
```

---

# 23. 可视化总清单

必须生成：

```text
1. route waterfall: v9.5.5 -> v9.6.2
2. payload0 clone GradeAB / longrisk bar chart
3. matched-pair lift distribution
4. confounder axis drop Pareto chart
5. SMD before/after matching plot
6. group density heatmap
7. value score vs risk score quadrant plot
8. TopK precision vs LDO/LSO drop plot
9. accepted region per-group precision heatmap
10. certificate threshold sensitivity curve
11. generated vs existing action feature PCA
12. APGF primitive outcome comparison chart
13. memory/offdiag/cover fail heatmap
14. selected runtime step timeline
15. paired replay branch comparison plot
```

---

# 24. 加速执行策略

本轮必须避免继续“一轮只发现一个 blocker”。

## 并行批次 A：当天可完成

```text
P0 boundary reproduction
P1 confounder ledger
P2 hidden confounder search
P3 target map
Base-Acc Sentinel reuse audit
```

## 并行批次 B：同一时间跑

```text
P4 legal feature deconfounding
P5 two-head ranker
P8 generated damage autopsy
```

## 并行批次 C：需要 GPU materializer

```text
P9 APGF generation
P9 branch-horizon replay
P10 generated certificate
```

## 并行批次 D：条件触发

```text
P7 controller
P12 selected runtime
P13 paired replay
P14 short/full training
```

---

# 25. Stop Conditions

立即停止当前子路线的条件：

## Stop payload-pocket route

```text
payload0 causal lift supported = 0
matched-pair V lift <= 0
```

## Stop existing-action rank route

```text
best deconfounded ranker LDO/LSO drop > 0.25
and no new confounder identified
and accepted region precision < 0.50
```

## Stop APGF generated primitive route

```text
best APGF GradeAB precision < 0.10
or V_LCB < 0
or longrisk_UCB > 0.50
```

## Stop certificate route

```text
TopK strong but accepted-region weak for two consecutive versions
and no calibration/threshold reason identified
```

---

# 26. 预期结论格式

本轮最后必须用下面格式判断：

```text
If existing-action rank pass:
  route = R1-ExistingActionControllerCandidateReady

If generated primitive pass:
  route = R2-GeneratedPrimitiveControllerCandidateReady

If both fail but legal rank has strong pooled signal:
  route = R3-LegalRankSignalButNotGroupStable

If payload pocket remains only confounded:
  route = R4-PayloadPocketStopped

If generated primitive longrisk remains high:
  route = R5-GeneratedPrimitiveLongRiskFail

If no path has signal:
  route = R6-FunctionalUpdatePrimitiveResetRequired
```

---

# 27. 本轮最重要的一句话

v9.6.2 不是再证明 payload0 pocket 强不强。v9.6.1 已经说明 payload0 不是可利用机制。  
v9.6.2 要回答的是：

$$
\boxed{
\text{离开 payload0 这个混杂口袋后，是否还存在跨 group 稳定的合法 rank；如果不存在，能否用 population-risk / memory / cover 约束直接生成好动作？}
}
$$
