# DG-KAN v9.2.41 Control-Gap Value Alignment 与 RoleWise Late-Attach Parallel Validation 完整实验计划

> 本计划基于 v9.2.40 `Parallel Base-Repair Confirmation 与 Snapshot Functional Re-entry` 的真实复盘制定。  
> v9.2.40 的 terminal route 是：
>
> ```text
> route = R11-BasePreservedButValueUnobservable
> base_candidate = LQ-t2-h256
> success_v9240_strict_purekan_functional = False
> success_v9240_full_functional = False
> success_v9240_external_ready = False
> ```
>
> v9.2.40 的关键事实是：**base、snapshot attach、inactive equivalence、no-event preservation、functional carrier actuatability 都已经过关；现在真正失败的是 value observability。**
>
> 具体地说：
>
> ```text
> Repaired base:
>   best = R2-LQ-fanin-output-scale-confirmed
>   robust pass = 1
>   near rate = 0.800000
>   macro delta = -0.004817
>   CI95 low = -0.007560
>   step q90 = 1.014925
>   memory = 0.969501
>
> Snapshot attach:
>   snapshot_attach_pass = 1
>   checkpoint_inactive_equivalence_pass = 1
>   max_logit_diff_inactive = 0.0
>   no_event_replay_preservation_pass = 1
>   max_no_event_logit_diff = 0.0
>
> Functional carrier:
>   carrier pass = 1
>   max r_z_tail = 0.30905587311056065
>   max r_perp_tail = 0.2706856067015393
>   carrier bad-event rate = 0.044444444444444446
>
> Value observability:
>   pass = 0
>   corr = -0.25921111692352256
>   AUC = 0.44358974358974357
>   precision = 0.08695652173913043
>   coverage = 0.10222222222222223
>   bad-event rate = 0.043478260869565216
> ```
>
> 因此 v9.2.41 的核心任务不是继续修 base，也不是继续修 attach，也不是针对 MNIST / Fashion-MNIST / KMNIST 分别调参。当前应该把全部实验资源集中到一个问题：
>
> $$
> \boxed{
> \text{active functional carrier 已经能动；为什么它的 value score 不能预测 Real-vs-control value？}
> }
> $$
>
> v9.2.41 必须采用并行验证。过去几轮串行 gate 虽然干净，但推进太慢。本轮要把 score failure autopsy、score redesign、role-wise late attach、value oracle upper bound、LDO/LSO、paired replay scout 放在同一个 gated runner 里并行跑。diagnostic rows 可以提前测，但 official success 必须严格受 gate 控制。

---

## 0. 硬约束

本计划继续遵守：

```text
no teacher
no self-teacher
no distillation
no loss modification
no label smoothing
no focal / margin / calibration loss
no sampler / class weight
no CPU offload
no fake / proxy rows
KAN path 不使用 PyTorch loss.backward graph
official controller 不使用 dataset_name 分支
PureKANConv / PureKANFormer 继续 deferred
```

Functional update 仍然是 update rule，不是 loss：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{\text{AdamW}}
+
\Delta\theta_{\text{functional}}.
$$

任务目标保持标准 CE：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

数据集只能作为诊断切片，不能作为 official controller 条件：

```text
allowed:
  report MNIST / Fashion-MNIST / KMNIST slice metrics
  report miss rows and failure attribution by dataset
  report leave-dataset-out generalization
  report signal-stratum distribution by dataset

forbidden:
  if dataset == Fashion: use controller A
  if dataset == KMNIST: use primitive B
  if dataset == MNIST: abstain
  tune threshold separately per dataset
  promote a value score because it rescues only one dataset
```

---

# Part I. 当前结果的独立判断

## 1. v9.2.40 没有达到 strict functional success

v9.2.40 的 route 是 `R11-BasePreservedButValueUnobservable`，说明它没有达到 strict PureKAN functional success。下游仍未打开：

```text
leave_dataset_out_pass = 0
leave_stratum_out_pass = 0
paired_replay_pass = 0
short_run_pass = 0
full_run_pass = 0
external_ready = 0
```

所以不能声明：

```text
strict PureKAN functional causal evidence
full functional success
external-ready success
Beyond-MLP success
```

但是这轮非常重要，因为它第一次把主 blocker 从 base / attach / actuation 全部推进到了 value mechanism。

## 2. v9.2.40 的真实进展

v9.2.40 至少完成了四个以前没有同时完成的 gate。

第一，repaired base robust pass。`R2-LQ-fanin-output-scale-confirmed` 在 30 rows 上 near rate 达到 `0.8`，macro delta `-0.004817`，CI95 low `-0.007560`，step q90 `1.014925`，memory `0.969501`。这说明 v9.2.39 的 base repair 不是单纯 3-seed 偶然。

第二，snapshot late attach 已经实现，且 attach-off 与 checkpoint 完全等价。`max_logit_diff_inactive=0.0`，说明 functional channel inactive 时没有污染 repaired base。

第三，no-event replay preservation pass。`max_no_event_logit_diff=0.0`，说明没有 event 时 attach branch 与 base continuation 完全一致。

第四，functional carrier 终于 active。`max r_z_tail=0.3090558731`，`max r_perp_tail=0.2706856067`，都明显超过 $0.10$ gate。这意味着当前已经不是 v9.2.23 的 functional event silent 状态。

因此当前最重要的推进可以写成：

$$
\boxed{
\text{base preserved + attach preserved + carrier active。}
}
$$

## 3. 当前真正失败点

失败点非常集中：

$$
\boxed{
\text{value score 与 grounded Real-vs-control value 方向错配。}
}
$$

证据是：

$$
Corr(S_{\text{value}},V_{\text{grounded}})=-0.259211,
$$

$$
AUC(Y_{\text{beat}})=0.443590<0.5,
$$

$$
Precision_{\text{accepted}}=0.086957.
$$

这不是“弱正相关不够”，而是更严重的 **score sign / value definition / control-gap mismatch**。  
同时 bad-event rate 只有 `0.043478`，说明 accepted events 大多不是任务灾难；问题是它们不能击败 AdamWParallel / bestLR controls。也就是说：

```text
当前 controller 可以做到相对 task-safe，
但无法识别 control-resistant good events。
```

这和过去多轮一致。v9.2.34 说明 movement-observability 不等于 value-observability；v9.2.35 说明 control-gap 是更有信息的统计；v9.2.40 现在说明即使 carrier active，如果 score 没有和 Real-vs-control value 对齐，paired replay 仍不能打开。

## 4. 进展是否太慢

我同意进展慢，但现在已经到了可以加速的节点。以前慢，是因为 base / attach / carrier 都没同时过关；functional 失败无法解释。v9.2.40 后不同了：base、attach、carrier 都已经过关，下一步可以直接对 value mechanism 做并行大诊断。

v9.2.41 不应再按：

```text
先做一个 score；
失败后下轮再做另一个 score；
再失败再下轮做 role-wise；
```

这样会继续拖慢。  
本轮应该一次性并行比较：

```text
score sign flip
control-gap decomposition
role-wise FT7 score
LCB/UCB conservative score
family-value score
oracle upper bound
shuffled controls
leave-dataset-out
paired replay scout
```

---

# Part II. v9.2.41 总体目标

v9.2.41 的总体目标是：

$$
\boxed{
\text{在 repaired base + snapshot late attach + active carrier 已闭合的基础上，找到或证伪 dataset-agnostic control-resistant value score。}
}
$$

目标分为六层。

## 1. Value failure attribution success

必须解释 v9.2.40 的：

```text
corr = -0.259211
AUC = 0.443590
precision = 0.086957
coverage = 0.102222
bad-event = 0.043478
```

归因必须落到以下机制之一：

```text
F1-score_sign_mismatch:
  当前 score 与 grounded value 方向相反。

F2-control_gap_missing:
  score 预测 Real 的 task-safe movement，但没有预测 Real 是否超过 AdamWParallel / bestLR。

F3-tail_metric_mismatch:
  score 优化 tail movement，但 grounded value 由 CEp99 / margin / control gap 主导。

F4-role_mismatch:
  stack/head/edge role 的贡献没有分离，导致聚合 score 反向。

F5-event_family_noise:
  single-event value 噪声大，但 family-level value 可预测。

F6-carrier_branch_mismatch:
  carrier 能动，但实际 functional branch 与 score branch 不一致。

F7-label_or_value_grounding_issue:
  grounded value 定义在新 repaired base / late-attach setting 下需要重新归一化。

F8-true_mechanism_absent:
  carrier movement 存在，但没有 control-resistant value。
```

成功标准：

```text
>= 90% failed accepted events assigned to one primary failure mode
score sign / control gap / role / family attribution all measured
no dataset_name used in attribution rule
```

## 2. Value score redesign success

新的 score 必须满足：

$$
AUC(Y_{\text{beat}})\geq0.70
$$

or:

$$
Corr(S_{\text{value}},V_{\text{grounded}})\geq0.35.
$$

Accept / abstain gate：

$$
Precision_{\text{accepted beats controls}}\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05.
$$

注意：bad-event 低不是 success，precision 必须同时过。v9.2.40 的 bad-event 已经过，但 precision 失败。

## 3. Role-wise late-attach success

FT7 role-wise functional core 在历史路线中有信号，但当前 strict FC-PureKAN route 还没有用 snapshot late attach 方式重新验证。v9.2.41 要实现 role-wise late attach，并确保它仍然 edge-owned、manual、graph-free。

成功标准：

```text
strict PureKAN contract pass
inactive equivalence pass
no-event preservation pass
carrier pass
value observability pass
```

## 4. Oracle upper-bound clarification

必须回答：

```text
当前 carrier 是否存在可被任何 legal score 选中的 good events？
```

如果 oracle upper bound 都很低，说明 current carrier 本身没有 control-resistant value。  
如果 oracle upper bound 高，而 legal score 低，说明问题是 scoring / routing。

Oracle diagnostic 不允许进入 official route；只能用于判断 research direction。

Oracle upper-bound pass：

$$
OracleBeatRate_{\text{Real vs controls}}\geq0.60,
$$

并且：

$$
OracleCoverage\in[0.03,0.15].
$$

## 5. Dataset-agnostic generalization success

任何 score 都必须通过 leave-dataset-out 和 leave-stratum-out。

LDO pass：

$$
Acc_{\text{heldout,Real}}\geq Acc_{\text{heldout,AdamW}}-0.005.
$$

At least two LDO splits satisfy：

$$
BeatRate_{\text{heldout,Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{heldout,Real vs bestLR}}\geq0.50.
$$

## 6. Official paired replay success

只有 value observability + LDO/LSO 通过后，official paired replay 才打开。

Paired replay pass：

$$
BeatRate_{\text{macro,Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro,Real vs bestLR}}\geq0.60.
$$

Task safety：

$$
Acc_{\text{Real,slice}}\geq Acc_{\text{AdamW,slice}}-0.005.
$$

---

# Part III. 核心假设

## H1：v9.2.40 的主要失败是 score sign / control-gap mismatch，而不是 carrier 不可用

v9.2.40 已经满足 carrier gate：

$$
r_{z,\text{tail}}=0.309056,
$$

$$
r_{\perp,\text{tail}}=0.270686.
$$

但 value score：

$$
Corr=-0.259211,
$$

$$
AUC=0.443590.
$$

H1 认为 carrier 有可用 movement，但 score 没有正确预测 control gap。

H1 成立标准：

```text
sign-flipped score or control-gap decomposed score improves AUC by >= 0.10
or reaches AUC >= 0.60 diagnostic
```

如果 sign-flip / decomposition / role-wise score 都无改善，则 H1 失败。

## H2：accepted bad-event 低说明 task safety 不是主 blocker

v9.2.40 accepted bad-event rate 是 `0.043478`，已经低于 `0.05`。  
H2 认为当前主要问题不是 RealFunctional 伤 task，而是 RealFunctional 不赢 controls。

H2 成立标准：

```text
task-safe rate remains high across RealFunctional candidates
but Real-vs-control precision remains low before score redesign
```

## H3：control-gap score 比 movement score 更重要

过去 OP/VAOP 的结果显示 movement / orthogonal-tail score 与 value 弱相关或反相关，而 control-gap 更有信息。v9.2.41 应优先构造：

$$
S_{\text{gap}}
=
Gain_{\text{Real}}
-
\max(Gain_{\text{AdamWParallel}},Gain_{\text{bestLR}}).
$$

H3 成立标准：

$$
AUC(S_{\text{gap}})-AUC(S_{\text{movement}})\geq0.10
$$

or:

$$
Corr(S_{\text{gap}},V)-Corr(S_{\text{movement}},V)\geq0.10.
$$

## H4：role-wise late attach 可能比 scalar value score 更接近 v8 functional core

v8/v9.2.20 的 functional signal 主要来自 role-wise functional mechanism。当前 A1/A2 类 late attach 可能 carrier active 但缺少 role separation。  
H4 认为 role-wise late attach 可以提升 value observability。

H4 成立标准：

Role-wise late attach score satisfies:

$$
AUC(Y_{\text{beat}})\geq0.70
$$

or:

$$
Corr(S_{\text{role}},V_{\text{grounded}})\geq0.35.
$$

## H5：如果 oracle upper bound 高但 legal score 低，下一步是 score learning / calibration；如果 oracle upper bound 低，下一步是 carrier redesign

H5 是关键分叉。

Oracle upper bound high：

```text
good events exist
score/controller is blocker
```

Oracle upper bound low：

```text
current carrier has no control-resistant value
carrier/mechanism is blocker
```

## H6：不能用 dataset branch 修 v9.2.40 precision

即使某个 dataset slice 上 precision 较高，也不能成为 official controller。  
Official controller 只能使用 signal features / value features / role features，不得使用 dataset name。

---

# Part IV. 并行执行设计

v9.2.41 必须并行，不再串行小步。  
同一个 runner 同时运行以下 lanes：

```text
Lane A:
  v9.2.40 boundary reproduction + score failure autopsy.

Lane B:
  score redesign matrix:
    sign flip
    control-gap
    LCB/UCB
    family-value
    role-wise
    hybrid score

Lane C:
  role-wise snapshot late attach implementation.

Lane D:
  oracle upper-bound and shuffled-control matrix.

Lane E:
  LDO / LSO validation.

Lane F:
  official paired replay on eligible survivor.

Lane G:
  short-run scout only if paired replay passes.
```

Gate discipline：

```text
Lane B/C/D diagnostic rows may run before all lanes finish.
official_eligible = 1 only if:
  repaired base pass
  snapshot attach pass
  inactive equivalence pass
  no-event preservation pass
  carrier pass
  value observability pass
  LDO/LSO pass
```

---

# Part V. Candidate 与 score 设计

## 1. Base and attach candidates

### B0：Repaired base reference

```text
R2-LQ-fanin-output-scale-confirmed
```

### A0：Current snapshot attach survivor

v9.2.40 best attach survivor。作为 reference。

### A1：LateAttachZeroLinearTail

保留 v9.2.40 carrier，用于 score diagnosis。

### A2：LateAttachControlGapChannel

Functional event score uses control-gap lower bound.

### A3：LateAttachRoleWiseFT7EdgeCarrier

Edge-owned role-wise functional channel：

$$
\phi_{ij}(h)
=
\phi^{base}_{ij}(h;\theta_T^\star)
+
\lambda(e)
[
\alpha_s(e)\phi^{F,s}_{ij}(h;\theta_{F,s})
+
\alpha_h(e)\phi^{F,h}_{ij}(h;\theta_{F,h})
].
$$

Role weights use only signal features：

$$
\alpha_r(e)
=
f(
CEp99,
MarginP10,
Curvature,
BranchRatio,
EffectiveDerivative,
ControlGap,
Uncertainty
).
$$

No dataset name.

### A4：FamilyValueLateAttach

Event family：

$$
family(e)=(stratum(e), horizon(e), risk\_bucket(e), role\_bucket(e)).
$$

Accept if：

$$
\mathbb{E}[V_{\text{grounded}}\mid family]>0
$$

and：

$$
Reliability(family)\geq0.30.
$$

### A5：Hybrid RoleWise + ControlGap

Uses role-wise carrier but accepts only if control-gap lower bound is positive.

---

## 2. Score candidates

### S0：Current v9.2.40 score

Reference only.

### S1：Sign-flipped current score

Diagnostic only.  
If S1 improves strongly, failure is sign mismatch.

### S2：Direct control-gap score

$$
S_2
=
\widehat{Gain}_{\text{Real}}
-
\max(
\widehat{Gain}_{\text{AdamWParallel}},
\widehat{Gain}_{\text{bestLR}}
).
$$

### S3：Conservative LCB/UCB score

$$
S_3
=
LCB(Gain_{\text{Real}})
-
UCB(
\max(Gain_{\text{AdamWParallel}},Gain_{\text{bestLR}})
).
$$

Accept if：

$$
S_3>0.
$$

### S4：Tail CE + margin value score

$$
S_4
=
w_{ce}(-\Delta CEp99)
+
w_m(\Delta MarginP10)
-
w_r Risk
-
S_{\text{control}}.
$$

Default:

```text
w_ce = 1.0
w_m = 1.0
w_r = 2.0
```

### S5：Role-wise FT7 score

$$
S_5
=
\alpha_s S_{\text{stack}}
+
\alpha_h S_{\text{head}}
-
S_{\text{control}}.
$$

### S6：Family-value reliability score

$$
S_6
=
\mathbb{E}[V\mid family]
-
\kappa\operatorname{Std}[V\mid family].
$$

### S7：Hybrid monotone score

A monotone rule using only:

```text
control_gap_score
tail_value_score
role_score
uncertainty
branch_ratio
effective_derivative
```

No learned black-box that hides dataset features.

### S8：Oracle upper-bound score

Uses posthoc outcome; diagnostic only.  
Never official.

---

# Part VI. 实验阶段

## P0：v9.2.40 boundary reproduction

### 目标

确认当前 boundary 稳定。

### 必须记录

```text
route
source_route_v9240
repaired_base_robust_pass
snapshot_attach_pass
checkpoint_inactive_equivalence_pass
no_event_replay_preservation_pass
functional_carrier_pass
max_r_z_tail
max_r_perp_tail
carrier_bad_event_rate
value_observability_pass
value_auc
value_corr
accepted_precision
accepted_coverage
accepted_bad_event_rate
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R11-BasePreservedButValueUnobservable
base / attach / carrier pass = 1
value observability pass = 0
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_gate_ladder_base_attach_carrier_value.svg
```

---

## P1：Value failure autopsy

### 目标

解释 v9.2.40 value failure，不允许直接进入新 score promotion。

### 必须记录

```text
event_id
dataset_slice
seed
horizon
signal_stratum
attach_candidate
controller
current_score
grounded_value
Y_beat
score_sign
score_rank
value_rank
real_gain
adamwparallel_gain
bestlr_gain
control_gap
tail_CEp99_delta
margin_delta
task_risk
uncertainty
role_stack_score
role_head_score
failure_mode
```

### 判断标准

P1 pass：

```text
>= 90% accepted events assigned to primary failure mode
sign mismatch measured
control-gap mismatch measured
role mismatch measured
family-value reliability measured
dataset_name not used for route
```

Key diagnostics：

$$
Corr(S_{\text{current}},V),
$$

$$
Corr(-S_{\text{current}},V),
$$

$$
AUC(S_{\text{current}}),
$$

$$
AUC(-S_{\text{current}}),
$$

$$
Corr(S_{\text{control-gap}},V).
$$

### 可视化

```text
p1_score_vs_value_scatter.svg
p1_sign_flip_auc.svg
p1_control_gap_decomposition.svg
p1_failure_mode_sankey.svg
p1_score_component_heatmap.svg
```

---

## P2：Parallel score redesign matrix

### 目标

并行测试 S0-S8，不再一轮只测一个 score。

### 设置

```text
base = R2 repaired checkpoint
attach = A0/A1/A2/A3/A4/A5
scores = S0-S8
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
events = signal strata S1-S8
branches = RealFunctional, AdamWParallel, bestLR, NoOp, Random
```

### 必须记录

```text
score_id
attach_candidate
event_id
signal_stratum
horizon
score_value
grounded_value
Y_beat
corr
auc
precision
coverage
bad_event_rate
accepted_event_count
dataset_name_used
posthoc_used_at_commit
validation_used
test_used
official_eligible
gate_missing_reason
```

### 判断标准

Diagnostic score pass：

$$
AUC(Y_{\text{beat}})\geq0.60
$$

or:

$$
Corr(S,V)\geq0.20.
$$

Official score pass：

$$
AUC(Y_{\text{beat}})\geq0.70
$$

or:

$$
Corr(S,V)\geq0.35.
$$

Accept / abstain pass：

$$
Precision\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05.
$$

Legality pass：

```text
dataset_name_used = 0
posthoc_used_at_commit = 0 for official scores
validation_used = 0
test_used = 0
```

### 可视化

```text
p2_score_matrix_auc_corr.svg
p2_precision_coverage_bad_event.svg
p2_score_legality_matrix.svg
p2_score_by_signal_stratum.svg
```

---

## P3：Role-wise late attach implementation and audit

### 目标

实现 A3/A5 role-wise late attach，重新把 FT7-style mechanism 放到 strict FC-PureKAN snapshot route 中。

### 必须记录

```text
attach_candidate
base_checkpoint_hash
role_channels
edge_owned_param_fraction
external_residual_used
ordinary_mlp_path_used
manual_forward
manual_backward
manual_update
uses_loss_backward
task_param_hash_before_attach
task_param_hash_after_attach
optimizer_state_hash_before_attach
optimizer_state_hash_after_attach
functional_param_zero_init
inactive_equivalence_max_logit_diff
no_event_max_logit_diff
implemented
implementation_status
```

### 判断标准

Implementation pass：

```text
A3 implemented = 1
strict PureKAN contract pass = 1
inactive equivalence pass = 1
no-event preservation pass = 1
```

### 可视化

```text
p3_rolewise_attach_contract.svg
p3_rolewise_inactive_noevent_equivalence.svg
```

---

## P4：Oracle upper-bound and shuffled-control matrix

### 目标

判断 current carrier 是否存在 posthoc good events，以及 score 是否只是没找到它们。

### 必须记录

```text
attach_candidate
event_id
signal_stratum
horizon
oracle_value
oracle_accept
oracle_precision
oracle_coverage
oracle_bad_event_rate
real_beats_adamwparallel
real_beats_bestlr
ValueScoreShuffled_pass
FunctionalChannelShuffled_pass
TailMaskShuffled_pass
RoleScoreShuffled_pass
```

### 判断标准

Oracle-good-events exist if：

$$
OraclePrecision\geq0.75,
$$

$$
OracleCoverage\in[0.03,0.15],
$$

$$
OracleBadEventRate\leq0.05.
$$

If oracle pass but legal score fail：

```text
score/controller blocker
```

If oracle fail：

```text
carrier/mechanism blocker
```

Shuffle controls must fail for official route：

```text
ValueScoreShuffled = fail
FunctionalChannelShuffled = fail
TailMaskShuffled = fail
RoleScoreShuffled = fail
```

### 可视化

```text
p4_oracle_upper_bound.svg
p4_oracle_vs_legal_score_gap.svg
p4_shuffle_control_matrix.svg
```

---

## P5：Leave-dataset-out / leave-stratum-out validation

### 目标

防止 score/controller 隐式 dataset tuning。

### 设置

Leave-dataset-out：

```text
train/calibrate on MNIST + Fashion, evaluate KMNIST
train/calibrate on MNIST + KMNIST, evaluate Fashion
train/calibrate on Fashion + KMNIST, evaluate MNIST
```

Leave-stratum-out：

```text
train/calibrate on all but one signal stratum
evaluate held-out stratum
```

### 必须记录

```text
split_type
heldout
attach_candidate
score_id
threshold
precision
coverage
bad_event_rate
task_safe
CEp99_delta
margin_delta
ECE_delta
NLL_delta
curvature_delta
beats_adamwparallel
beats_bestlr
shuffle_control_pass
dataset_name_used
```

### 判断标准

LDO pass：

$$
Acc_{\text{heldout,Real}}\geq Acc_{\text{heldout,AdamW}}-0.005.
$$

At least two LDO splits satisfy：

$$
BeatRate_{\text{heldout,Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{heldout,Real vs bestLR}}\geq0.50.
$$

LSO pass：

At least $70\%$ held-out strata task-safe and CE-tail non-worse：

$$
CEp99_{\text{heldout,Real}}\leq CEp99_{\text{AdamW}}+\epsilon.
$$

### 可视化

```text
p5_leave_dataset_out_matrix.svg
p5_leave_stratum_out_matrix.svg
p5_hidden_dataset_tuning_audit.svg
```

---

## P6：Official paired replay

### 目标

只有 P2/P5 survivor 可以进入 official paired replay。

### 设置

```text
base = R2 repaired base checkpoint
attach_candidate = best official score survivor
controller = best legal signal-based controller
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, InvertedRole, ValueScoreShuffled, FunctionalChannelShuffled, TailMaskShuffled, RoleScoreShuffled
```

### 必须记录

```text
attach_candidate
controller
score_id
dataset
seed
horizon
signal_stratum
branch
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
real_beats_adamwparallel
real_beats_bestlr
real_beats_random
real_beats_noop
task_safe
event_count
coverage
bad_event_rate
step_q90
memory_ratio
base_checkpoint_hash
```

### 判断标准

Official paired replay pass：

$$
BeatRate_{\text{macro,Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro,Real vs bestLR}}\geq0.60.
$$

Task safety：

$$
Acc_{\text{each slice,Real}}\geq Acc_{\text{AdamW}}-0.005.
$$

Anti-overfit controls fail：

```text
ValueScoreShuffled = fail
FunctionalChannelShuffled = fail
TailMaskShuffled = fail
RoleScoreShuffled = fail
DatasetRouteShuffled = fail
EventRouteShuffled = fail
InvertedRoleMask = fail
```

System：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p6_official_paired_replay_pareto.svg
p6_macro_beat_rate.svg
p6_signal_stratum_win_matrix.svg
p6_shuffle_control_matrix.svg
p6_system_gate_distribution.svg
```

---

## P7：Short-run validation

### 目标

验证 paired replay survivor 是否能在连续训练中保持机制优势。只有 P6 survivor 进入。

### 设置

```text
steps = 50,240,640
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, ValueScoreShuffled
```

### 必须记录

```text
candidate
dataset
seed
steps
train_loss
holdout_loss
val_acc_proxy
CEp99
margin_p10
ECE_proxy
NLL_proxy
curvature
local_lipschitz
basis_usage_entropy
functional_channel_usage_entropy
event_count
coverage
bad_event_rate
step_ratio_q90
memory_ratio
base_checkpoint_hash
```

### 判断标准

Task safety：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005.
$$

Control superiority：

$$
MetricGain_{\text{functional}}>MetricGain_{\text{AdamWParallel}},
$$

$$
MetricGain_{\text{functional}}>MetricGain_{\text{bestLR}}.
$$

Mechanism pass, at least one：

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamW}},
$$

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamW}},
$$

$$
ECE_{\text{functional}}\leq ECE_{\text{AdamW}},
$$

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

### 可视化

```text
p7_short_run_task_mechanism_pareto.svg
p7_short_run_controls.svg
p7_event_timeline.svg
p7_ce_tail_margin_panel.svg
```

---

## P8：Full 10-seed validation

### 目标

验证 strict PureKAN functional route 是否在 full training 上成立。

### 设置

```text
epochs = 20
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
baseline = MLP-match
diagnostic baseline = QuadraticFeatureMLP
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole
```

### 必须记录

```text
candidate
dataset
seed
val_acc
test_acc
delta_vs_mlp
delta_vs_adamw
delta_vs_adamwparallel
delta_vs_bestlr
delta_vs_quadratic_feature_mlp
near_pass
full_pass
CEp99
margin_p10
wrong_confidence_p95
ECE
NLL
curvature_ratio
local_lipschitz_ratio
basis_usage_entropy
functional_channel_usage_entropy
event_count
coverage
bad_event_rate
step_ratio_q90
memory_ratio
base_checkpoint_hash
```

### 判断标准

Task safety：

$$
\Delta Acc_{\text{functional-vs-AdamW}}\geq-0.005.
$$

Macro improvement：

$$
\Delta Acc_{\text{macro,functional}}
-
\Delta Acc_{\text{macro,AdamW}}
\geq0.003.
$$

Hard-stratum repair：

$$
\Delta Acc_{\text{hard-stratum,functional}}
-
\Delta Acc_{\text{hard-stratum,AdamW}}
\geq0.005.
$$

Control superiority：

$$
MetricGain_{\text{functional}}>MetricGain_{\text{AdamWParallel}},
$$

$$
MetricGain_{\text{functional}}>MetricGain_{\text{bestLR}}.
$$

Mechanism pass, at least one：

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamW}},
$$

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamW}},
$$

$$
ECE_{\text{functional}}\leq ECE_{\text{AdamW}},
$$

$$
NLL_{\text{functional}}\leq NLL_{\text{AdamW}},
$$

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

---

## P9：Robustness / strong baseline / external-ready gate

### 目标

确认 strict PureKAN functional advantage 不是 clean MNIST-family artifact。

### 设置

```text
label_noise = 0.05,0.10,0.20
input_noise = 0.05,0.10
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
strong baselines = MLP-match, hidden-bracket MLP, QuadraticFeatureMLP, repaired LQ AdamW-only
```

### 必须记录

```text
candidate
baseline_id
params
flops
forward_ratio
backward_ratio
step_ratio
memory_ratio
dataset
seed
clean_acc
noisy_acc
acc_drop
CEp99
margin_p10
ECE
NLL
curvature
local_lipschitz
```

### 判断标准

Robustness pass：

$$
AccDrop_{\text{functional}}\leq AccDrop_{\text{AdamW}}.
$$

Noise-tail pass：

$$
CEp99_{\text{functional}}\leq CEp99_{\text{AdamW}}.
$$

Strong baseline pass：

$$
Acc_{\text{functional}}\geq Acc_{\text{QuadraticFeatureMLP}}-0.005
$$

or:

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{QuadraticFeatureMLP}}.
$$

---

# Part VII. Required artifacts

```text
run_manifest.json
contract_audit_v9241.csv
p0_v9240_boundary_reproduction.csv
p1_value_failure_autopsy.csv
p2_parallel_score_redesign_matrix.csv
p3_rolewise_late_attach_implementation.csv
p4_oracle_upper_bound_and_shuffle_controls.csv
p5_leave_dataset_and_stratum_out_validation.csv
p6_official_paired_replay.csv
p7_short_run_functional_validation.csv
p8_full_10seed_functional_validation.csv
p9_robustness_external_ready.csv
value_failure_trace_v9241.csv
score_component_trace_v9241.csv
rolewise_attach_trace_v9241.csv
oracle_upper_bound_trace_v9241.csv
leave_dataset_out_trace_v9241.csv
paired_replay_branch_trace_v9241.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9240_boundary_unstable
F3_dataset_tuning_detected
F4_value_failure_unattributed
F5_score_sign_mismatch
F6_control_gap_missing
F7_tail_metric_mismatch
F8_role_mismatch
F9_event_family_noise
F10_carrier_branch_mismatch
F11_value_grounding_issue
F12_true_mechanism_absent
F13_rolewise_attach_not_implemented
F14_rolewise_attach_contract_fail
F15_value_observability_fail
F16_value_score_system_too_expensive
F17_leave_dataset_out_fail
F18_leave_stratum_out_fail
F19_paired_replay_control_equivalent
F20_shuffle_control_pass
F21_functional_lr_equivalent
F22_short_run_task_drop
F23_full_run_no_macro_hard_stratum_gain
F24_strong_baseline_explains_gain
F25_robustness_fail
F26_external_not_ready
F27_fake_or_proxy_violation
F28_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-ValueFailureAttributed:
  v9.2.40 value failure is explained by sign / control-gap / role / family / grounding.

R2-ScoreSignMismatchConfirmed:
  sign-flipped or monotone-transformed score recovers diagnostic predictivity.

R3-ControlGapScorePass:
  direct control-gap score passes value observability.

R4-RoleWiseLateAttachPass:
  role-wise snapshot attach passes contract / inactive / no-event / carrier / value gates.

R5-FamilyValueScorePass:
  event-family score predicts value without dataset-specific route.

R6-OracleHighLegalScoreLow:
  good events exist, but legal controller cannot yet find them.

R7-OracleLowCarrierMechanismReset:
  even oracle cannot find enough good events; carrier mechanism must reset.

R8-LeaveDatasetOutPass:
  controller generalizes without dataset-specific tuning.

R9-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R10-BasePreservedCarrierActiveButValueUnobservable:
  base and carrier stay valid, but no legal score predicts value.

R11-ControlDominatedFunctional:
  RealFunctional has task-safe effect but controls dominate.

R12-StrictPureKANFunctionalShortRunPass:
  short-run task-safe mechanism gain.

R13-StrictPureKANFunctionalFullPass:
  full 10-seed macro / hard-stratum / geometry gain.

R14-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` 必须记录：

```text
route
v9240_boundary_pass
dataset_tuning_detected
value_failure_mode
score_sign_mismatch_pass
best_score_id
best_attach_candidate
value_observability_pass
value_auc
value_corr
accepted_precision
accepted_coverage
accepted_bad_event_rate
oracle_upper_bound_pass
rolewise_attach_pass
leave_dataset_out_pass
leave_stratum_out_pass
paired_replay_pass
short_run_pass
full_run_pass
external_ready
primary_blocker
next_required_implementation
success_v9241_strict_purekan_functional
success_v9241_full_functional
success_v9241_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 value failure autopsy
  P2 score redesign matrix
  P3 role-wise late attach implementation

Batch 2:
  P4 oracle upper-bound and shuffle controls
  P5 leave-dataset-out / leave-stratum-out
  diagnostic P6 paired replay scout for score survivors

Batch 3:
  official P6 paired replay
  P7 short-run if P6 pass

Batch 4:
  P8 full 10-seed
  P9 robustness / strong baseline
```

Gate rule：

```text
P2/P4/P6 diagnostic rows may be measured before all gates finish.
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier pass
  value observability pass
  LDO/LSO pass
```

This preserves causality while avoiding slow one-gate-per-version iteration.

---

# Part X. 停止条件

## Minimum diagnostic success

```text
v9.2.40 boundary reproduced
value failure attributed
score redesign matrix completed
role-wise late attach audited
oracle upper-bound measured
no fake/proxy/offload/loss/teacher violation
```

## Value mechanism success

```text
Minimum diagnostic success
+
at least one legal score passes observability
+
precision / coverage / bad-event gate pass
+
leave-dataset-out pass
```

## Local functional success

```text
Value mechanism success
+
official paired replay beats AdamWParallel / bestLR
```

## Full functional success

```text
Local functional success
+
short/full run task-safe mechanism gain
+
macro or hard-stratum improvement
```

## Failure stop

```text
1. v9.2.40 boundary cannot be reproduced；
2. value failure cannot be attributed；
3. all legal scores fail observability；
4. oracle upper bound fails, meaning carrier has no good events；
5. role-wise late attach cannot be implemented as strict PureKAN；
6. leave-dataset-out fails；
7. paired replay remains control-equivalent；
8. shuffle controls pass, indicating overfit；
9. short-run task drops；
10. full run gives no macro / hard-stratum / geometry gain；
11. functional breaks system gate；
12. gains are explained by QuadraticFeatureMLP；
13. any teacher/loss/fake/proxy/offload violation occurs。
```

---

# Part XI. 最终解释规则

## Case A：sign flip or score decomposition passes

可以声明：

```text
v9.2.40 failed because the value score was directionally misaligned, not because carrier lacked actuatability.
```

但不能声明 functional success unless LDO and paired replay pass.

## Case B：role-wise late attach passes

可以声明：

```text
Strict snapshot PureKAN route recovered a role-wise functional value mechanism.
```

但 full success 仍需 short/full validation.

## Case C：oracle high but legal scores fail

必须声明：

```text
Good functional events exist, but current legal controller cannot identify them.
```

下一步修 controller / calibration，不修 base.

## Case D：oracle low

必须声明：

```text
Current carrier movement is not control-resistant value.
```

下一步重做 carrier / role-wise mechanism.

## Case E：LDO fails

必须声明：

```text
Controller is implicitly dataset-specific; official success is not allowed.
```

不能用 dataset-specific tuning 写成功。

## Case F：paired replay passes

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full validation。

---

# Part XII. 最终建议

v9.2.41 的一句话策略是：

$$
\boxed{
\text{base、attach、carrier 已经打开；现在必须并行解决 value score 与 control-gap causality，而不是继续修 base。}
}
$$

当前最关键的问题不是：

```text
R2 base 还要不要修；
snapshot attach 是否可行；
functional carrier 是否能动；
Fashion 怎么调；
KMNIST 怎么调；
MNIST 是否 abstain；
再换哪个 horizon。
```

而是：

```text
1. v9.2.40 的 score 为什么与 grounded value 负相关？
2. sign flip / control-gap / role-wise / family-value 哪个能解释 value？
3. 是否存在 oracle-good functional events？
4. 如果存在，legal score 为什么找不到？
5. 如果不存在，当前 carrier 是否只是 task-safe movement 而非 control-resistant value？
6. role-wise FT7 mechanism 能否在 strict snapshot late-attach route 中重新产生 value？
7. official paired replay 能否打过 AdamWParallel / bestLR？
```

v9.2.41 的成败会决定下一步主线：

```text
if legal value score or role-wise score passes:
  enter paired replay / short-run / full-run.

if oracle high but legal score fail:
  focus on controller calibration.

if oracle low:
  reset functional carrier mechanism.

if paired replay passes:
  strict PureKAN functional finally gets local causal evidence.
```
