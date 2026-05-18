# DG-KAN v9.6.3 Group-Stable Legal Rank / Memory-OffDiag Safe Primitive / Accepted-Region Closure 完整实验计划

> 本计划基于 v9.6.2 `Confounder-Purged Legal Rank / PopRisk Geometry Primitive` 的真实执行结果制定。  
> 本轮不在 MNIST / Fashion-MNIST / KMNIST 上调榜，不按 dataset 写规则，不调 dataset-specific threshold。  
> 本轮要解决的是一个更根本的问题：
>
> $$
> \boxed{
> \text{训练当下能否用合法信号选出一批跨 group 稳定、长期安全、真正有用的 functional update 动作？}
> }
> $$
>
> v9.6.3 不再小修 payload0、GIR9、RC7、RANK5、APGF5。  
> v9.6.3 要么把 existing-action legal rank 推成一个真正可冻结的 accepted region，要么明确终止 existing-action selector 主线，转向 memory/offdiag-safe action generator。

---

# 0. v9.6.2 独立判断

## 0.1 v9.6.2 的真实进展

v9.6.2 的 route 是：

```text
route = R4-PayloadPocketStopped
base_candidate = LQ-t2-h256
success_v9620_strict_purekan_functional = False
success_v9620_full_functional = False
success_v9620_external_ready = False
```

它没有达到 strict functional success，也没有打开 selected runtime、leaveout、paired replay、short/full training。

但它有真实进展：

```text
1. payload_norm_bucket 不再允许作为 selector，只能作为 stratification / diagnostic。
2. payload0 pocket 被正式停止为 confounded route。
3. hidden confounder search 没找到可 official 的 strong confounder。
4. legal rank 仍有很强 pooled TopK 信号。
5. rank-safe certificate 已经接近 action-count / value / longrisk gate，但仍 group unstable。
6. APGF1-APGF8 工程链路完整闭合，但 generated outcome 继续失败。
```

v9.6.2 的关键数据：

```text
P1 payload pocket stop:
  matched-pair V lift = -0.5198887260597593
  GradeAB lift = -0.6134615384615385
  payload0 intervention precision = 0.009615384615384616
  memory projection longrisk = 0.8557692307692307
  payload_norm_bucket_allowed_as_selector = 0

P2 hidden confounder:
  strong_confounder_count = 0
  best_axis = payload_norm_bucket
  drop_if_removed = 0.11494252873563218

P3 target map:
  best = T2-ValuePositiveNoLongRisk
  count = 97
  coverage = 0.03372739916550765
  V LCB = 0.16622185363738462
  longrisk UCB = 0.0
  target_map_pass = 0

P4 legal feature deconfounding:
  best = COMBO-ValueRiskMemoryCover
  TopK87 precision = 0.6551724137931034
  V LCB = 0.049639752010919136
  longrisk UCB = 0.0
  LDO drop = 0.22988505747126436

P5 two-head ranker:
  best = RANK5-invariant-risk-minimization
  TopK87 GradeAB precision = 0.8735632183908046
  V LCB = 0.14111334880346277
  longrisk UCB = 0.0
  LDO drop = 0.37931034482758624
  max_group_share = 1.0

P6 certificate:
  best = CERT10-FixedTopKGroupBalanced
  accepted = 87
  coverage = 0.030250347705146036
  GradeAB precision = 0.8735632183908046
  V LCB = 0.14111334880346277
  longrisk UCB = 0.0
  pass = 0
  reason = group_drop / max_group_share

P8 generated damage:
  dominant damage mode = D7-risk-veto-ineffective
  fraction = 0.7369791666666666
  longrisk_created_rate = 0.7369791666666666
  Damage V LCB = -0.7420588191683418

P9 APGF:
  generated actions = 512
  branch-horizon rows = 12288 / 12288
  quality audit pass = 1
  best APGF5 GradeAB precision = 0.03125
  best APGF5 V LCB = -0.33527855012819596
  best APGF5 h240 longrisk UCB = 0.828904255301089
```

## 0.2 我对 route 的独立看法

`R4-PayloadPocketStopped` 是对的，但它有点停在了“已经确认的清理动作”上。

v9.6.2 真正的新边界不是 payload pocket stop。payload pocket 在 v9.6.1 已经被 intervention 证伪。v9.6.2 的新边界是：

$$
\boxed{
\text{existing-action legal rank 已经几乎满足 value / coverage / longrisk，但仍不能跨 group 稳定。}
}
$$

尤其 P6 很重要：

```text
accepted = 87
coverage = 0.03025
precision = 0.87356
V LCB = 0.1411
longrisk UCB = 0.0
```

这些数已经不像早期那种完全不可用的 controller。它失败在：

```text
LDO drop = 0.3793
max_group_share = 1.0
```

所以 v9.6.3 的核心不是继续证明 payload pocket 已经停用，而是回答：

```text
为什么 P6 这种看起来几乎过线的 accepted region 仍然 max_group_share = 1.0？
这个 group concentration 是统计口径问题、hidden group 问题、support 不均问题，还是 ranker 真在利用局部口袋？
```

## 0.3 当前真正 blocker

当前主 blocker 应写成：

$$
\boxed{
\text{legal rank 的 accepted region 还没有 group-stable；generated primitive 仍然制造 longrisk。}
}
$$

不是：

```text
truth base 不可信；
old outcome table 问题；
payload0 selector 还能用；
action apply 不可信；
branch-horizon materializer 缺失；
APGF 没实现；
Base-Acc Sentinel 崩了。
```

这些都不是当前主因。

---

# 1. v9.6.3 总体目标

v9.6.3 有两个并行总目标。

## 1.1 目标 A：existing-action accepted region closure

判断现有 AP0 action universe 中，是否可以用合法训练当下信号选出一个可部署的 accepted region：

$$
Accept(a)=1
$$

且满足：

$$
Coverage \ge 0.03,
$$

$$
Precision_{GradeAB} \ge 0.75,
$$

$$
LCB(V_{integrated}) > 0,
$$

$$
UCB(LongRisk_{240}) \le 0.05,
$$

$$
UCB(Bad) \le 0.05,
$$

$$
UCB(Null) \le 0.15,
$$

$$
LDO\_drop \le 0.10,
$$

$$
LSO\_drop \le 0.10,
$$

$$
MaxGroupShare \le 0.35
\quad
\text{or}
\quad
MaxGroupShare \le BaseGroupShare + 0.15.
$$

如果 strong gate 太严，允许记录 weak gate：

$$
LDO\_drop \le 0.20,
$$

$$
LSO\_drop \le 0.20,
$$

但 weak gate 不能进入 official paired replay，只能决定是否继续 existing-action route。

## 1.2 目标 B：generated-action longrisk / memory / offdiag reset

判断 APGF / APGE / APGD 这类 generated action 失败是否来自统一的生成机制错误：

```text
risk-veto ineffective；
memory/offdiag unsafe；
source distribution OOD；
cover collapse；
long-horizon damage。
```

在明确失败机制后，只测试少量带硬约束的新 primitive，不再盲目新增 APG* 小变体。

新的 generated primitive 必须生成：

```text
h20 / h80 / h240 不明显坏；
V_integrated LCB > 0；
h240 longrisk UCB <= 0.10 weak，<= 0.05 strong；
memory/offdiag fail rate 明显低于 APGF；
source-to-generated Damage LCB >= 0 weak，> 0 strong；
TopK64 GradeAB precision >= 0.30 weak，>= 0.60 strong。
```

## 1.3 本轮停止条件

v9.6.3 必须有明确停止条件，避免继续慢速绕圈。

### Existing-action route stop

如果经过 group-deconfounding、multi-axis balancing、risk/memory/cover veto 后，最好的 legal accepted region仍满足：

```text
LDO_drop > 0.20
or
LSO_drop > 0.20
or
max_group_share = 1.0
or
heldout accepted region precision < 0.75
or
V_LCB <= 0
or
longrisk_UCB > 0.05
```

则 existing-action selector 主线进入：

```text
R-existing-action-rank-not-deployable
```

并停止继续调 rank threshold / TopK / certificate threshold。

### Generated-action route stop

如果 APGH / APGI 新 primitive 在 512 actions、12288 branch-horizon rows上仍满足：

```text
best GradeAB precision < 0.10
or
V_LCB < 0
or
longrisk_UCB > 0.50
or
longrisk_created_rate > 0.50
```

则 generated-action primitive 主线进入：

```text
R-generated-primitive-family-reset-required
```

不再新增同类小变体。

---

# 2. 全局硬约束

继续遵守：

```text
no teacher
no self-teacher
no distillation
no task loss modification
no label smoothing
no focal / margin / calibration loss
no sampler / class weight
no fake / proxy rows
no CPU offload as official
KAN path 不使用 PyTorch loss.backward graph
official controller 不使用 dataset_name 分支
official controller 不使用 future outcome
official controller 不使用 V_integrated / risk_score / GradeAB / LongRisk label
payload_norm_bucket 不允许作为 selector，只允许 stratification / diagnostic
```

Functional update 仍然是 update rule：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{\text{AdamW-equivalent}}
+
\Delta\theta_{\text{functional}}.
$$

任务 loss 保持：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

允许：

```text
group labels for audit / split / reweighting；
group labels for training-time balancing；
group labels for leaveout evaluation；
不允许 group label 作为 official online selector input，除非该 group 是合法泛化状态变量且不是 dataset_name。
```

---

# 3. 实验并行结构

v9.6.3 分四条并行 lane，不再一轮只查一个 blocker。

```text
Lane A: Existing-action rank / accepted region
Lane B: Group pocket / hidden confounder / target support
Lane C: Generated-action damage / memory-offdiag-safe primitive
Lane D: Runtime / base sentinel / system boundary
```

四条 lane 可以并行落盘，但 final route 必须按 gate 汇总：

```text
P0-P2 先确认数据边界；
P3-P7 决定 existing-action route；
P8-P12 决定 generated-action route；
P13-P16 只有 upstream pass 才打开 system / paired replay / short-full。
```

---

# 4. 核心假设

## H1：P6 accepted region 失败不是因为 value/risk 不够，而是 group concentration

v9.6.2 P6 的 value/risk 已经很强：

```text
precision = 0.8736
V LCB = 0.1411
longrisk UCB = 0.0
coverage = 0.03025
```

失败原因是：

```text
group drop；
max group share = 1.0。
```

H1 要验证：

```text
只要打散 group concentration，accepted region 是否仍保持 precision / V / longrisk？
```

H1 成立标准：

```text
P5/P6 新 ranker 在 TopK87 或 accepted=87 时：
  precision >= 0.75
  V_LCB > 0
  longrisk_UCB <= 0.05
  LDO/LSO drop <= 0.10 strong 或 <= 0.20 weak
  max_group_share <= 0.35 或 <= base_share + 0.15
```

H1 失败：

```text
去 group concentration 后 precision / V 立刻崩，说明之前只是局部口袋。
```

## H2：没有单一 hidden confounder，但有多轴组合 confounder

v9.6.2 P2 strong confounder count = 0。  
这不等于没有混杂，可能是：

```text
payload_norm_bucket × step_bucket；
family × horizon；
dataset × action_norm；
candidate_origin × memory_score；
rank_score × payload bucket。
```

H2 成立标准：

```text
单轴 confounder fail；
二轴 / 三轴 interaction 能解释 >= 70% group drop；
interaction-conditioned leaveout drop 明显下降。
```

H2 失败：

```text
多轴也解释不了 drop，说明 ranker 本身不稳定或 target 支持不足。
```

## H3：Value rank 与 longrisk/memory veto 不能压成一个分数

前几轮反复看到：

```text
value signal strong，但 longrisk 高；
TopK strong，但 accepted region 不稳；
certificate AUC 高，但 frozen accepted region 崩。
```

H3 认为需要三段式：

$$
Score_{value}(a)
$$

$$
Veto_{risk}(a)
$$

$$
Veto_{memory/offdiag/cover}(a)
$$

最终：

$$
Accept(a)=1
\iff
Score_{value}(a)\ge \tau_v
\land
RiskVeto(a)=0
\land
MemoryVeto(a)=0
\land
SupportOK(a)=1
\land
CostOK(a)=1.
$$

H3 成立标准：

```text
三段式 ranker 比单分数 RANK5 在 LDO/LSO drop、max_group_share、heldout precision 上同时改善。
```

## H4：APGF/APGE/APGD generated action 失败主要来自 source OOD + longrisk veto ineffective

v9.6.2 APGF dominant damage：

```text
D7-risk-veto-ineffective
longrisk_created_rate = 0.73698
Damage V LCB = -0.742
```

H4 成立标准：

```text
generated action 分布相对 canonical GradeAB action：
  payload/action norm OOD
  memory/offdiag fail 高
  cover entropy 下降
  old-family score 下降
  longrisk created rate 高
```

如果 H4 成立，生成器不能继续做“小 residual blend”，而要直接变成 memory/offdiag-safe constrained update。

## H5：现阶段 runtime 不应阻塞科学判断，但必须保留 selected-runtime preflight

没有 selected controller 前，official runtime 不能打开。  
但可以继续记录：

```text
feature compute ms；
rank compute ms；
certificate compute ms；
payload apply ms；
estimated accepted action count；
step_ratio_q90 preflight。
```

H5 成立标准：

```text
selected controller 一旦存在，可以在同轮测 official runtime；
无 selected controller 时，runtime 只做 preflight，不写 pass。
```

---

# 5. 指标定义

## 5.1 Action outcome metrics

每个 action 需要记录：

```text
action_id
candidate_id
event_id
dataset
seed
family
stratum
step_bucket
payload_norm_bucket
action_norm_bucket
candidate_origin
branch
horizon
V20
V80
V240
V_integrated
V_integrated_LCB
weak_CP20
weak_CP80
weak_CP240
GradeA
GradeB
GradeAB
Bad
Null
LongRisk240
h240_longrisk_UCB
beats_AdamWParallel
beats_bestLR
beats_NoOp
beats_Random
shuffled_payload_fail
```

## 5.2 Geometry metrics

```text
signal_score
control_transfer_improvement
population_risk_offdiag_score
memory_old_family_fail
memory_old_stratum_fail
memory_forget_risk
cover_entropy_delta
basis_effective_rank_delta
hard_tail_cover_entropy_delta
curvature_delta
jacobian_spectral_delta
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
```

## 5.3 Rank / controller metrics

```text
TopK64 precision
TopK87 precision
Accepted count
Coverage
GradeAB precision
V_integrated_LCB
h240_longrisk_UCB
bad_UCB
null_UCB
LDO drop
LSO drop
leave-payload-bucket-out drop
leave-family-out drop
leave-step-bucket-out drop
max_group_share
macro_group_precision
macro_group_V_LCB
macro_group_longrisk_UCB
calibration-to-heldout drift
```

## 5.4 Runtime metrics

```text
feature_compute_ms_q50/q90
rank_compute_ms_q50/q90
certificate_compute_ms_q50/q90
payload_apply_ms_q50/q90
controller_launches_per_active_step_q90
step_ratio_q50/q90
memory_ratio
kernel_count
sync_count
empty_step_kernel_count
selected_action_count_per_step
```

---

# 6. 可视化要求

v9.6.3 必须生成以下图，不要只给 CSV。

## 6.1 Group pocket 图

```text
1. TopK action group share bar chart
2. group × rank score heatmap
3. group × GradeAB density heatmap
4. payload_norm_bucket × step_bucket interaction heatmap
5. family × memory_fail heatmap
6. leave-one-group-out drop waterfall
```

## 6.2 Rank 图

```text
1. precision-coverage curve
2. V_LCB vs coverage curve
3. longrisk_UCB vs coverage curve
4. TopK precision vs K
5. score distribution by GradeAB / non-GradeAB
6. score distribution by longrisk / non-longrisk
7. calibration split vs heldout split score drift
```

## 6.3 Controller 图

```text
1. accepted region 2D map: value score vs risk score
2. accepted region 2D map: value score vs memory score
3. accepted region group support heatmap
4. accepted region leaveout robustness plot
```

## 6.4 Generator 图

```text
1. source action vs generated action V_integrated scatter
2. source action vs generated action longrisk scatter
3. Damage distribution histogram
4. generated action OOD distance histogram
5. memory/offdiag fail rate by primitive
6. cover entropy delta by primitive
```

## 6.5 Runtime 图

```text
1. per-step selected action count histogram
2. controller compute time histogram
3. payload apply time histogram
4. step_ratio_q90 by controller candidate
```

---

# 7. 实验阶段

---

## P0：v9.6.2 boundary reproduction

### 目标

确认 v9.6.2 的边界可复现，不在错误 artifact 上继续。

### 输入

```text
v9.6.2 route_decision
v9.6.2 ranker/certificate table
v9.6.2 APGF outcome table
v9.6.1 payload pocket intervention table
v9.4.8 canonical outcome universe
```

### 必须记录

```text
route_v9620
payload_pocket_route_stop
strong_confounder_count
best_target_id
best_target_count
best_target_coverage
best_ranker_id
best_ranker_TopK87_precision
best_ranker_LDO_drop
best_ranker_max_group_share
best_certificate_id
best_certificate_accepted_count
best_certificate_precision
best_certificate_V_LCB
best_certificate_longrisk_UCB
APGF best primitive
APGF best V_LCB
APGF best longrisk_UCB
```

### Pass

```text
p0_pass = 1
if all key values match v9.6.2 within tolerance.
```

### Fail

```text
p0_pass = 0
stop, rerun v9.6.2 boundary.
```

---

## P1：P6 accepted-region scope audit

### 目标

审计 v9.6.2 P6 的 `CERT10-FixedTopKGroupBalanced` 为什么叫 group-balanced，但仍出现 `max_group_share = 1.0`。

### 假设

H1a：group-balanced 只在某一显式 group 轴上做了平衡，但另一个隐藏轴仍集中。  
H1b：accepted=87 里有重复或 action-level / row-level scope 混用。  
H1c：TopK / accepted region 的统计 universe 与 GradeAB count universe 不一致。  
H1d：某些 group 实际没有正例，导致 balancing 只是在有 support 的 group 内集中。

### 必须记录

```text
accepted_action_count
accepted_unique_action_count
accepted_unique_candidate_count
accepted_unique_event_count
accepted_row_count
GradeAB_count_in_eval_universe
GradeAB_count_in_accepted
group axes:
  dataset
  seed
  family
  stratum
  step_bucket
  event_bucket
  payload_norm_bucket
  action_norm_bucket
  candidate_origin
  memory_bucket
  offdiag_bucket
  cover_bucket
for each axis:
  group_count
  max_group_share
  top_group_id
  top_group_base_share
  top_group_lift
  GradeAB_precision_by_group
  V_LCB_by_group
  longrisk_UCB_by_group
```

### 判断标准

Pass:

```text
scope_consistency_pass = 1
duplicate_action_id_count = 0
row_action_scope_mismatch = 0
universe_mismatch = 0
all group share metrics reproducible.
```

Fail:

```text
scope_consistency_pass = 0
route = R-scope-mismatch-repair-required
```

### 可视化

```text
accepted group share bar chart
accepted vs base group share lift chart
accepted GradeAB precision by group
accepted V_LCB by group
accepted longrisk by group
```

---

## P2：multi-axis hidden pocket search

### 目标

寻找导致 group instability 的多轴组合，不再只找单轴 confounder。

### 方法

对以下轴做 1D、2D、3D interaction search：

```text
dataset
seed
family
stratum
step_bucket
payload_norm_bucket
action_norm_bucket
candidate_origin
memory_bucket
offdiag_bucket
cover_bucket
value_score_bucket
risk_score_bucket
support_bucket
```

对每个 group interaction $g$ 计算：

$$
Lift(g)
=
Precision_{accepted}(g)-Precision_{base}(g).
$$

$$
DropIfRemoved(g)
=
Precision_{TopK}
-
Precision_{TopK\setminus g}.
$$

$$
Support(g)=|g|.
$$

### 必须记录

```text
axis_set
group_id
support
base_share
accepted_share
GradeAB_base_precision
GradeAB_accepted_precision
V_LCB
longrisk_UCB
drop_if_removed
lift
causal_intervention_available
matched_pair_available
```

### Pass

```text
multi_axis_pocket_explained = 1
if interaction groups explain >= 70% of rank drop
and no forbidden field is needed.
```

### Fail

```text
multi_axis_pocket_explained = 0
if no combination explains > 30% drop.
```

### 可视化

```text
interaction heatmaps
drop_if_removed Pareto chart
support vs lift scatter
```

---

## P3：group-balanced target support map

### 目标

判断 `T2-ValuePositiveNoLongRisk` 这类 target 是否真的有足够跨 group 支持。

v9.6.2 中：

```text
T2 count = 97
coverage = 0.0337
V LCB = 0.1662
longrisk = 0
target_map_pass = 0
```

现在要明确它为什么不 pass。

### 必须记录

```text
target_id
target_count
target_coverage
V_LCB
longrisk_UCB
bad_UCB
null_UCB
macro_group_precision
min_group_positive_count
positive_group_coverage
max_group_share
LDO target count
LSO target count
leave-payload-bucket-out target count
group_missing_positive_count
```

### Pass

```text
target_support_pass = 1
if:
  count >= 87
  coverage >= 0.03
  V_LCB > 0
  longrisk_UCB <= 0.05
  bad_UCB <= 0.05
  positive_group_coverage >= 0.80
  max_group_share <= 0.35 or <= base_share + 0.15
```

### Weak pass

```text
target_support_weak_pass = 1
if:
  count >= 87
  V_LCB > 0
  longrisk_UCB <= 0.10
  positive_group_coverage >= 0.60
```

### Fail interpretation

如果 target density pass 但 group support fail：

```text
说明不是好动作数量不够，而是好动作分布太偏。
```

### 可视化

```text
target density by group
positive support coverage plot
target count under leave-one-group-out
```

---

## P4：legal feature deconfounding v2

### 目标

重建 legal features，不让 ranker直接利用 group pocket。

### 特征集合

只允许 commit-time legal features：

```text
value-like:
  control_transfer_improvement_proxy
  gradient_action_alignment
  AdamW_conflict_relief
  linearized_CE_delta_proxy
  linearized_margin_delta_proxy

risk-like:
  hard_tail_fraction
  hard_tail_margin_proxy
  risk_veto_score
  longrisk_proxy_no_outcome

memory-like:
  old_family_signal_score
  old_stratum_similarity
  memory_projection_score
  population_risk_offdiag_score

cover-like:
  cover_entropy_delta_proxy
  basis_effective_rank_proxy
  hard_tail_cover_entropy_proxy

support/cost:
  family_support_count
  stratum_support_count
  feature_compute_ms
  payload_apply_estimate_ms
```

禁止：

```text
V_integrated
GradeAB
LongRisk label
Bad label
Null label
risk_score derived from outcomes
future horizon metrics
dataset_name as input
payload_norm_bucket as selector
```

### 方法

每个 feature 记录：

```text
raw AUC
within-group AUC
leave-dataset-out AUC
leave-stratum-out AUC
leave-payload-bucket-out AUC
TopK64 precision
TopK87 precision
TopK87 V_LCB
TopK87 longrisk_UCB
feature cost q90
```

### Pass

```text
feature_deconfounding_pass = 1
if at least one feature group has:
  within_group_AUC >= 0.70
  TopK87 precision >= 0.60
  V_LCB > 0
  longrisk_UCB <= 0.10
  LDO/LSO drop <= 0.20
```

### 可视化

```text
feature AUC heatmap by group
TopK precision by feature
feature-cost vs precision scatter
```

---

## P5：value-rank + risk-veto + memory-veto ranker

### 目标

不再把 value 和 longrisk 混进一个分数。构造三段式 ranker：

$$
Accept(a)=1
\iff
S_v(a)\ge \tau_v
\land
S_r(a)\le \tau_r
\land
S_m(a)\le \tau_m
\land
Support(a)\ge \tau_s.
$$

其中：

```text
S_v: value rank score
S_r: longrisk / bad risk veto score
S_m: memory / offdiag / cover veto score
```

### 候选

```text
RANK-A: value only
RANK-B: value + risk veto
RANK-C: value + memory veto
RANK-D: value + risk + memory veto
RANK-E: pairwise within-group value rank + risk/memory veto
RANK-F: group DRO ranker
RANK-G: leave-one-axis stable ranker
RANK-H: conformal group-balanced ranker
```

### 必须记录

```text
ranker_id
feature_groups_used
forbidden_field_count
TopK64 precision
TopK87 precision
accepted_count
coverage
V_LCB
longrisk_UCB
bad_UCB
null_UCB
LDO_drop
LSO_drop
leave_payload_bucket_drop
leave_family_drop
max_group_share
macro_group_precision
macro_group_V_LCB
macro_group_longrisk_UCB
calibration_to_heldout_drift
```

### Strong pass

```text
ranker_strong_pass = 1
if:
  accepted_count >= 87
  coverage >= 0.03
  precision >= 0.75
  V_LCB > 0
  longrisk_UCB <= 0.05
  bad_UCB <= 0.05
  null_UCB <= 0.15
  LDO_drop <= 0.10
  LSO_drop <= 0.10
  max_group_share <= 0.35 or <= base_share + 0.15
```

### Weak pass

```text
ranker_weak_pass = 1
if:
  accepted_count >= 87
  precision >= 0.70
  V_LCB > 0
  longrisk_UCB <= 0.10
  LDO_drop <= 0.20
  LSO_drop <= 0.20
```

### 可视化

```text
precision-coverage curves
risk-veto thresholds vs precision
memory-veto thresholds vs precision
group drop waterfall
accepted region value-risk plot
accepted region value-memory plot
```

---

## P6：rank-safe certificate v11

### 目标

把 ranker 变成一个可冻结的 accepted-region certificate，而不是只看 TopK。

### 证书形式

$$
Cert(a)=
[
S_v(a),
S_r(a),
S_m(a),
Support(a),
Cost(a)
].
$$

接受规则：

$$
Accept(a)=1
\iff
S_v(a)\ge \tau_v
\land
S_r(a)\le \tau_r
\land
S_m(a)\le \tau_m
\land
Support(a)\ge \tau_s
\land
Cost(a)\le \tau_c.
$$

### Calibration protocol

```text
split by seed/family/stratum/payload bucket
thresholds frozen on calibration split
evaluate on heldout split
then LDO/LSO/leave-payload-bucket-out
```

### 必须记录

```text
certificate_id
thresholds
accepted_cal
precision_cal
V_LCB_cal
longrisk_UCB_cal
accepted_heldout
coverage_heldout
precision_heldout
V_LCB_heldout
longrisk_UCB_heldout
bad_UCB_heldout
null_UCB_heldout
LDO_drop
LSO_drop
max_group_share
feature_cost_q90
certificate_cost_q90
```

### Pass

同 P5 strong pass。  
只有 P6 pass 后，P7 controller 才能打开。

### Fail interpretation

如果 P5 TopK pass 但 P6 fail：

```text
说明当前分数只能做排序诊断，不能形成 frozen accepted region。
```

---

## P7：existing-action minimal controller

### 目标

在 P6 pass 的情况下，生成 official controller artifact。

### Controller 约束

```text
no dataset_name branch
no outcome-derived fields
no payload_norm_bucket selector
no test/validation outcome
no future outcome
no old table
feature count <= 8 groups
thresholds frozen
```

### 必须记录

```text
controller_id
controller_version
feature_names
feature_legality_hash
thresholds
calibration split hash
heldout split hash
accepted action count
accepted event count
precision
coverage
V_LCB
longrisk_UCB
bad_UCB
null_UCB
support metrics
LDO/LSO metrics
```

### Pass

```text
existing_action_controller_pass = 1
if P6 strong pass and legality audit pass.
```

---

## P8：selected controller runtime preflight

### 目标

如果 P7 pass，立刻测 selected controller runtime，不等下一轮。

### 必须记录

```text
feature_compute_ms_q90
rank_compute_ms_q90
certificate_compute_ms_q90
payload_apply_ms_q90
step_ratio_q90
memory_ratio
kernel_count
sync_count
empty_step_kernel_count
active_step_count
selected_actions_per_active_step
```

### Pass

```text
selected_runtime_pass = 1
if:
  step_ratio_q90 <= 1.50
  memory_ratio <= 1.05
  empty_step_kernel_count = 0
```

### Weak pass

```text
step_ratio_q90 <= 2.00
```

Only strong pass can open official paired replay.

---

## P9：generated damage autopsy v2

### 目标

系统解释 APGF/APGE/APGD 为什么持续生成 high-longrisk / value-negative 动作。

### 必须记录

```text
primitive_family
source_action_id
generated_action_id
source_distribution_distance
payload_norm_delta
action_norm_delta
memory_score_delta
offdiag_score_delta
cover_entropy_delta
basis_rank_delta
V_integrated_source
V_integrated_generated
LongRisk_source
LongRisk_generated
Damage_V
Damage_longrisk
damage_mode
```

### Damage modes

```text
D1-source-OOD
D2-value-direction-lost
D3-risk-veto-ineffective
D4-memory-offdiag-fail
D5-cover-collapse
D6-curvature-spike
D7-longrisk-created
D8-negative-control-like
```

### Pass

```text
damage_autopsy_pass = 1
if assigned_damage_fraction >= 0.90
and dominant modes identified.
```

---

## P10：APGH memory/offdiag-safe primitive spec

### 目标

只在 P9 解释清楚后生成新 primitive。  
APGH 不再是泛泛 residual blend，而是围绕 memory/offdiag safety 设计。

### APGH candidates

```text
APGH1-PopRiskOffdiagProjectedDelta
APGH2-MemoryPreservingEdgeMaskDelta
APGH3-CoverEntropyNonCollapseDelta
APGH4-OldFamilyOrthogonalizedValueDelta
APGH5-ValueRankAnchoredRiskVetoDelta
APGH6-MultiHorizonBoundarySymmetricDelta
APGH7-SignalChannelSNRMemoryDelta
APGH8-NegativeControlShuffledPayload
```

### 生成约束

每个 generated action 必须落盘：

```text
payload tensor
certificate tensor
payload hash
certificate hash
action apply replay
source action link
feature legality record
```

### Preflight pass

```text
payload_hash_missing = 0
certificate_hash_missing = 0
action_apply_linf_max = 0
no_transform_equivalence = 1
negative_control_divergence = 1
```

---

## P11：APGH branch-horizon smoke

### 目标

真实 materialize APGH outcomes，不写 proxy。

### 设置

```text
8 primitives
64 actions per primitive
512 generated actions
branches:
  RealAPGH
  AdamWParallel
  AdamWOnly
  bestLR
  NoOp
  Random
  ShuffledAPGH
  CertificatePassNoPayload
horizons:
  h20
  h80
  h240
```

预期：

$$
512 \times 8 \times 3 = 12288
$$

或根据实际 branch count 记录 expected rows。

### 必须记录

```text
expected_rows
actual_rows
branch_missing
horizon_missing
duplicate_rows
label_exclusivity_violation
NaN/Inf count
rows_per_sec
unresolved_exception_count
```

### Pass

```text
apgh_branch_horizon_pass = 1
if actual_rows = expected_rows
and quality_audit_pass = 1.
```

---

## P12：APGH outcome / geometry pass

### 目标

判断 APGH 是否真正生成可用 frontier。

### 必须记录

```text
primitive_id
generated_action_count
GradeAB_precision
V_integrated_LCB
h240_longrisk_UCB
bad_UCB
null_UCB
memory_fail_rate
offdiag_fail_rate
cover_collapse_rate
new_positive_created_rate
longrisk_created_rate
Damage_V_LCB
```

### Strong pass

```text
apgh_strong_pass = 1
if some primitive has:
  GradeAB_precision >= 0.60
  V_integrated_LCB > 0
  h240_longrisk_UCB <= 0.05
  bad_UCB <= 0.05
  memory_fail_rate <= canonical GradeAB memory fail rate + 0.05
  longrisk_created_rate <= 0.10
```

### Weak pass

```text
apgh_weak_pass = 1
if:
  GradeAB_precision >= 0.30
  V_integrated_LCB > 0
  h240_longrisk_UCB <= 0.10
  longrisk_created_rate <= 0.20
```

If fail:

```text
generated route stops unless P9/P12 reveal a clearly new mechanism.
```

---

## P13：generated-action certificate/controller

### 目标

如果 APGH pass，构造 generated-action controller。  
如果 APGH fail，不运行 threshold search。

### Pass

```text
apgh_controller_pass = 1
if APGH strong pass
and certificate accepted region meets same gates as P6.
```

---

## P14：system route decision

### Route table

```text
R0-boundary_not_reproduced
R1-scope_mismatch_in_certificate
R2-hidden_multiaxis_pocket_explains_rank
R3-existing_action_rank_group_unstable
R4-existing_action_controller_pass_runtime_fail
R5-existing_action_system_pass
R6-generated_damage_explained_APGH_not_run
R7-APGH_generated_frontier_fail
R8-APGH_generated_frontier_pass_controller_fail
R9-APGH_system_pass
R10-both_existing_and_generated_routes_fail
```

### 必须输出

```text
route
primary_blocker
secondary_blocker
existing_action_route_status
generated_action_route_status
runtime_status
paired_replay_status
next_required_implementation
```

---

## P15：official leaveout + paired replay boundary

只有以下都过才打开：

```text
P6 certificate pass
P7 controller pass
P8 selected runtime pass
```

如果打开，必须记录：

```text
RealFunctional vs AdamWParallel
RealFunctional vs bestLR
RealFunctional vs NoOp
RealFunctional vs Random
RealFunctional vs ShuffledPayload
leave-dataset-out
leave-stratum-out
leave-family-out
```

Pass:

```text
paired_replay_pass = 1
if RealFunctional beats all controls
and shuffled payload fails
and LDO/LSO pass.
```

---

## P16：short/full training boundary

只有 P15 pass 才打开。  
不能因为 Base-Acc Sentinel 好就打开。

记录：

```text
train acc
val acc
test acc
CE
NLL
ECE
CEp99
margin_p10
hard-stratum acc
time_to_target
steps_to_target
runtime q90
memory
forgetting
continual retained acc
```

对照：

```text
LQ base no functional
MatchedMLP
AdamWStrongLRGridMLP
QuadraticFeatureMLP
NoOp functional
Shuffled functional
```

---

# 8. Base-Acc Sentinel

继续跑，但不用于 controller。

记录：

```text
MNIST
Fashion-MNIST
KMNIST
seeds 0..9
LQ-t2-h256
MatchedMLP
AdamWStrongLRGridMLP
QuadraticFeatureMLP
```

Pass 只代表 base health：

```text
LQ_catastrophic_fail = 0
```

不得写成 functional success。

---

# 9. 加速策略

## 9.1 并行 lane

```text
Lane A:
  P1-P6 existing-action rank/certificate

Lane B:
  P2-P3 group support / target map

Lane C:
  P9-P12 generated damage / APGH primitive

Lane D:
  P8/P16 runtime and base sentinel
```

## 9.2 分层 preflight

任何 generated primitive 必须先过：

```text
1 action preflight
3 action preflight
16 action preflight
64 action smoke
512 action official smoke
```

如果 16 action 时：

```text
V_LCB << 0
or
longrisk > 0.50
or
apply error > tolerance
```

则停止该 primitive，不跑 full 512。

## 9.3 快速失败规则

```text
如果 P1 发现 scope mismatch，停止 rank。
如果 P5 best weak ranker LDO/LSO drop > 0.50，停止 existing rank threshold search。
如果 P9 发现 generated actions OOD distance 远高于 canonical safe region，先修 generator，不跑 P11。
如果 APGH negative control best，立即判 generated family invalid。
```

---

# 10. 最终成功定义

v9.6.3 的最低有效成功不是 full functional success，而是下面二选一：

## Success A：existing-action system candidate

```text
existing_action_controller_pass = 1
selected_runtime_pass = 1
system_legal_controller_pass = 1
```

## Success B：generated-action frontier candidate

```text
apgh_strong_pass = 1
apgh_controller_pass = 1
selected_runtime_pass = 1
```

如果都没有：

```text
success_v9630_strict_purekan_functional = False
success_v9630_full_functional = False
external_ready = False
```

但必须输出一个明确 route：

```text
existing-action route 是否停止；
generated-action route 是否停止；
下一步到底是 rank、primitive、runtime，还是目标定义问题。
```

---

# 11. 最终报告必须回答的问题

最终复盘不能只写 route。必须回答：

```text
1. P6 v9.6.2 accepted region 为什么 max_group_share = 1.0？
2. 是否存在多轴 hidden pocket？
3. T2 target 为什么 coverage 够却 target_map_pass = 0？
4. value rank 和 risk/memory veto 分开后，LDO/LSO 是否下降？
5. existing-action rank 是否能形成 frozen accepted region？
6. APGF/APGE/APGD/APGH 的 generated damage 主因是否一致？
7. APGH 是否产生了新 positive？
8. runtime 是否可以在 selected controller 下进入 envelope？
9. Base-Acc Sentinel 是否仍健康？
10. 是否可以打开 paired replay？
```

---

# 12. 预期解释

如果 v9.6.3 成功：

```text
说明我们终于从 pooled TopK diagnostic 走到了 group-stable system controller。
```

如果 v9.6.3 失败但 existing-action route停止：

```text
说明当前 AP0 action universe 中的好动作可以事后找到，但不能用合法训练当下信号跨 group 稳定选择。
```

如果 generated route也失败：

```text
说明当前 generator families 仍不能生成 memory/offdiag-safe、longrisk-safe action。
```

如果两条都失败，下一步不应继续调 threshold，而应重定义 functional update primitive：

```text
从“挑选已有动作”转向“训练过程中构造 population-risk-safe direction”。
```

这个方向应直接吸收两个思想：

```text
1. Deep Manifold:
   好动作应该稳定局部 cover、避免曲率/边界漂移、保护 fixed-region。

2. Generalization signal/reservoir:
   好动作应该走多个样本一致支持的 signal channel，而不是进入训练噪声 reservoir。
```

---

# 13. v9.6.3 输出文件清单

必须落盘：

```text
p0_boundary_reproduction_v9630.csv
p1_certificate_scope_audit_v9630.csv
p2_multiaxis_hidden_pocket_search_v9630.csv
p3_group_balanced_target_support_map_v9630.csv
p4_legal_feature_deconfounding_v2_v9630.csv
p5_value_rank_risk_memory_veto_ranker_v9630.csv
p6_rank_safe_certificate_v11_v9630.csv
p7_existing_action_controller_v9630.csv
p8_selected_controller_runtime_preflight_v9630.csv
p9_generated_damage_autopsy_v2_v9630.csv
p10_apgh_primitive_spec_preflight_v9630.csv
p11_apgh_branch_horizon_smoke_v9630.csv
p12_apgh_outcome_geometry_pass_v9630.csv
p13_apgh_controller_v9630.csv
p14_route_decision_v9630.json
p15_leaveout_paired_replay_boundary_v9630.csv
p16_short_full_boundary_v9630.csv
base_acc_sentinel_v9630.csv
no_fake_audit_v9630.csv
contract_audit_v9630.csv
failure_taxonomy_v9630.csv
```

每个 CSV 必须有 hash，manifest 必须记录输入 artifact、seed、device、data-root、runner commit、参数、row count、quality audit。
