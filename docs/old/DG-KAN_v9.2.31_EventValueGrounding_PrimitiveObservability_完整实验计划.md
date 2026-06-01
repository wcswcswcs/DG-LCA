# DG-KAN v9.2.31 Event-Value Grounding 与 Primitive Observability：从不可预测 pre-event feature 到可验证 functional causality 的完整实验计划

> 本计划基于 v9.2.30 `Pre-Event Signal-Value Functional Controller` 的真实复盘制定。  
> v9.2.30 的 terminal route 是：
>
> ```text
> route = R7-PrimitiveEffectUnpredictable
> base_candidate = LQ-t2-h256
> success_v9230_strict_purekan_functional = False
> success_v9230_full_functional = False
> success_v9230_external_ready = False
> ```
>
> v9.2.30 的核心事实不是“functional update 失败”，也不是“应该回到 Fashion / KMNIST 数据集调参”。更准确地说：
>
> $$
> \boxed{
> \text{当前 pre-event features 尚不能可靠预测 RealFunctional 是否超过 AdamWParallel / best LR。}
> }
> $$
>
> 但这个结论也不能被过度解释成“primitive 已经彻底不可预测”。因为 v9.2.30 的 best movement abs corr 为 `0.299708`，几乎贴到 `0.30` gate；同时 v9.2.29 已经证明 signal strata 可以解释 failure，且 high-precision abstention 局部可行。真正的问题是：我们还没有把 **event value 的定义、归一化、测量可靠性、pre-event observability** 建成一个可靠闭环。
>
> 本轮继续遵守：
>
> ```text
> no teacher
> no self-teacher
> no distillation
> no loss modification
> no label smoothing
> no focal / margin / calibration loss
> no sampler / class weight
> no CPU offload
> no fake / proxy rows
> KAN path 不使用 PyTorch loss.backward graph
> official controller 不使用 dataset_name 分支
> PureKANConv / PureKANFormer 继续 deferred
> ```
>
> Functional update 仍然是 update rule，不是 loss：
>
> $$
> \theta_{t+1}
> =
> \theta_t
> +
> \Delta\theta_{\text{AdamW}}
> +
> \Delta\theta_{\text{functional}}.
> $$
>
> 任务损失保持标准 CE：
>
> $$
> L_{\text{task}}=CE(y,p_\theta(x)).
> $$

---

## 0. 当前实验结果的独立判断

### 0.1 v9.2.30 没有达到目标

v9.2.30 已经按计划复现 v9.2.29 boundary，并真实执行 first-wave pre-event movement replay。核心结果是：

```text
P0:
  source route = R1-SignalStrataExplainFailures
  event-value corr = 0.197001
  fake/proxy = 0

P1:
  pre-event movement replay rows = 486
  pre-event feature pass = 0
  best movement abs corr = 0.299708

P2:
  failure reclassification pass = 0
  attributed fraction = 0.000000
  abstain precision = 0.000000

P3:
  best predictor = None
  corr = 0.000000
  precision = 0.000000
  coverage = 0.000000

route:
  R7-PrimitiveEffectUnpredictable
  blocker = pre_event_features_not_predictive
```

因此不能声明：

```text
strict PureKAN functional update success
event-value predictor success
official signal-routed paired replay success
short-run / full-run / external-ready
```

### 0.2 这轮不是无意义失败

v9.2.30 至少推进了三件事。

第一，它没有倒退回 dataset tuning。P0 复现的是 v9.2.29 的 dataset-agnostic boundary：signal strata 能解释 failure，但 event-value predictor 还不过。这个路线是正确的，因为我们不是在 MNIST/Fashion/KMNIST 上打榜。

第二，P1 的 best movement abs corr 是 `0.299708`，这不是完全无信号。它离 gate `0.30` 只有极小距离，说明 pre-event movement features 可能有弱 signal，只是当前 measurement / label / aggregation / coverage 还没有足够稳定。

第三，P2/P3 崩为 0 反而暴露了一个更本质的问题：当前 failure-mode reclassification 和 predictor pipeline 可能过于依赖单事件标签，而单事件 Real-vs-control gap 方差太大；也可能是 value label 的归一化方式不对，使得原本接近 gate 的 movement feature 无法进入 predictor。

### 0.3 当前最重要的新边界

v9.2.29 的边界是：

$$
\text{signal strata explain failures, but event-value corr}<0.30.
$$

v9.2.30 的边界是：

$$
\text{pre-event movement has near-threshold signal, but event-value grounding fails.}
$$

所以当前不能继续问：

```text
Fashion 怎么调过？
KMNIST 怎么调过？
哪个 dataset 应该 abstain？
```

而应该问：

```text
1. actual event value 的 label 是否定义正确？
2. Real-vs-control gap 是否需要归一化到 control variance / stratum baseline？
3. pre-event movement feature 是否需要按 tail / horizon / primitive 归一化？
4. 单事件标签是否太噪，是否应改为 event-family value 或 bootstrap expected value？
5. 当前 primitive 是否真的不可预测，还是测量协议无法读出它的 signal？
```

---

## 1. 当前问题的本质

### 1.1 不是 functional core 消失

v8-FT7 functional core 已经通过 strong controls full replay 被保留；strict PureKAN interface 后续也做到 P4/P5/actuatability pass。当前不是回到“functional 是否存在”的阶段。

### 1.2 不是数据集调参问题

Fashion / KMNIST 的差异可以诊断，但不能用于 official policy。  
数据集只能作为评估切片：

```text
MNIST:
  easy-mode / abstention safety diagnostic

Fashion-MNIST:
  delayed / control-dominated tail signal diagnostic

KMNIST:
  hard-mode / actual-effect magnitude diagnostic
```

Official controller 只能看 dataset-agnostic observables：

```text
tail severity
margin risk
wrong confidence
curvature spike
actual logit movement
tail logit movement
non-AdamW component
branch ratio
effective derivative
cosine with AdamW / bestLR
control-gap uncertainty
horizon agreement
event-value confidence
```

### 1.3 不是简单 predictor 调参

v9.2.30 的 P2/P3 失败说明：如果不先修 event-value grounding，继续换 predictor 只是调参。  
现在必须先把 label、feature、aggregation、reliability 做对，再训练或校准 predictor。

### 1.4 真正 blocker

当前 blocker 是：

$$
\boxed{
\text{event value 的可观测性和可预测性没有闭合。}
}
$$

更具体地说：

```text
P1:
  pre-event movement features 有近阈值相关性。

P2:
  failure-mode attribution 失败，说明事件标签或分类规则没有稳定落到机制类别。

P3:
  predictor 没有 candidate，说明 pipeline 不能把 P1 的弱 signal 转成可用 controller。
```

因此 v9.2.31 的重点不是继续做 route patch，而是建立：

$$
\boxed{
\text{grounded event-value label}
+
\text{pre-event observability}
+
\text{high-precision adequate-coverage abstention}
+
\text{leave-dataset-out validation}.
}
$$

---

## 2. v9.2.31 总体目标

v9.2.31 的总体目标是：

$$
\boxed{
\text{将 pre-event functional movement 从 near-threshold diagnostic signal 转化为可泛化 event-value controller。}
}
$$

这分为四个层级。

### 2.1 Event-value grounding success

先证明 actual event value 的定义可靠。  
不是直接训练 predictor，而是先建立一个稳定 label：

$$
V(e)
=
Gain_{\text{Real}}(e)
-
\max\left(
Gain_{\text{AdamWParallel}}(e),
Gain_{\text{bestLR}}(e)
\right).
$$

这里 $Gain$ 不能只用单一 metric。它应是预注册 multi-objective value：

$$
Gain
=
w_{ce}G_{ce}
+
w_{m}G_{margin}
+
w_{curv}G_{curv}
+
w_{ece}G_{ece}
-
w_{risk}Risk_{\text{task drop}}.
$$

默认：

```text
w_ce = 1.0
w_margin = 1.0
w_curv = 0.5
w_ece = 0.25
w_risk = 2.0
```

每一项都必须按 control distribution 归一化：

$$
\tilde{G}_m(e)
=
\frac{
G_m(e)-\mu_{m,\text{controls}}(s)
}{
\sigma_{m,\text{controls}}(s)+\epsilon
},
$$

其中 $s$ 是 signal stratum，不是 dataset name。

Event-value grounding 成功标准：

$$
Reliability(V)\geq0.30,
$$

其中 reliability 可以用 bootstrap split-half correlation、repeat replay correlation 或 horizon-consistency correlation 测量。

### 2.2 Pre-event observability success

验证 update commit 前的 features 能解释 grounded value：

$$
Corr(S_{\text{pred}},V_{\text{grounded}})\geq0.30.
$$

至少一个 movement feature 或 uncertainty-adjusted movement feature满足：

$$
|Corr(f,V_{\text{grounded}})|\geq0.30.
$$

### 2.3 Event-value controller success

controller 必须满足：

$$
Precision_{\text{accepted beats controls}}\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05.
$$

其中 accepted event 不能只覆盖单个 dataset 或单个 seed；至少要覆盖两个 signal strata：

$$
|\{s: Coverage_s>0\}|\geq2.
$$

### 2.4 Local functional causality success

official paired replay 通过标准：

$$
BeatRate_{\text{macro,Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro,Real vs bestLR}}\geq0.60,
$$

并且：

$$
Acc_{\text{Real,dataset}}\geq Acc_{\text{AdamW,dataset}}-0.005.
$$

Official controller 不允许使用 dataset_name。

---

## 3. 核心假设

### H1：v9.2.30 失败主要来自 event-value label 未归一化，而不是 primitive 完全不可预测

P1 best movement corr 已到 `0.299708`，接近 gate。H1 认为如果将 actual value 按 control distribution、signal stratum、horizon 和 metric scale 归一化，则 correlation 会超过 0.30。

H1 成立标准：

$$
Corr(f_{\text{best movement}},V_{\text{grounded}})\geq0.30.
$$

如果 grounded 后仍低于 0.20，H1 失败，转向 primitive unpredictability。

### H2：单事件 outcome 太噪，需要 event-family value

Functional event 的 effect 可能在单 row 上高方差。H2 认为应该把 event 按同一 signal stratum、primitive、horizon bucket 聚合成 event-family value：

$$
V_{\text{family}}
=
\mathbb{E}_{e\sim family}[V(e)].
$$

H2 成立标准：

$$
Corr(S_{\text{pred}},V_{\text{family}})>
Corr(S_{\text{pred}},V_{\text{single}})+0.10.
$$

### H3：v9.2.29 的 high-precision abstention 可以扩展，但必须引入 uncertainty calibration

v9.2.29 已有：

```text
precision = 0.830189
coverage = 0.020167
bad-event = 0
```

H3 认为加入 uncertainty features 后，可以把 coverage 提升到 $[0.03,0.15]$，同时 precision 仍不低于 $0.75$。

H3 成立标准：

$$
Precision\geq0.75,
$$

$$
Coverage\geq0.03,
$$

$$
BadEventRate\leq0.05.
$$

### H4：如果 signal-value predictor 需要 dataset_name 才能过，则不能进入 official route

允许做 dataset-stratified analysis，但 official controller 必须在 leave-dataset-out 下通过。

H4 成立标准：

对每个 held-out dataset：

$$
Acc_{\text{heldout,Real}}\geq Acc_{\text{heldout,AdamW}}-0.005.
$$

至少两个 split 满足：

$$
BeatRate_{\text{heldout,Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{heldout,Real vs bestLR}}\geq0.50.
$$

### H5：如果 grounded value 仍不可预测，则当前 primitive/interface 缺少 causal observability

H5 是停止条件。  
如果在 grounded label、family aggregation、uncertainty calibration、leave-dataset-out 后仍然不能预测 event value，则不允许继续做 Fashion/KMNIST-specific patch。应回到：

```text
functional primitive / interface redesign；
basis channel redesign；
event-time actuation mechanism；
AdamW-only full-pass repair。
```

---

## 4. Candidate 与变量设计

### 4.1 Baselines

```text
B0-MLP-match:
  same-parameter MLP reference。

LQ0-LQ-t2-h256:
  strict FC-PureKAN AdamW-only base。

N2a-Rational:
  current base-qualified rational functional channel。

N2c-SharedRBF:
  local smooth functional channel。

N3c-SharedRBFDerivativeBand:
  derivative-controlled local channel。

V8-FT7:
  mechanism source only, not official strict candidate。
```

### 4.2 Event-value labels

原始 metric deltas：

```text
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
local_lipschitz_delta
acc_delta
```

转成 signed gains：

$$
G_{ce}=-\Delta CEp99,
$$

$$
G_{margin}=+\Delta MarginP10,
$$

$$
G_{ece}=-\Delta ECE,
$$

$$
G_{nll}=-\Delta NLL,
$$

$$
G_{curv}=-\Delta Curvature.
$$

Task risk：

$$
Risk_{\text{task drop}}
=
\max(0, Acc_{\text{AdamW}}-Acc_{\text{Real}}-0.005).
$$

Control-relative value：

$$
V_{\text{ctrl}}
=
Gain_{\text{Real}}
-
\max(Gain_{\text{AdamWParallel}},Gain_{\text{bestLR}}).
$$

Normalized value：

$$
V_{\text{norm}}
=
\frac{
V_{\text{ctrl}}-\mu_{\text{ctrlgap,stratum}}
}{
\sigma_{\text{ctrlgap,stratum}}+\epsilon
}.
$$

Binary accepted label：

$$
Y_{\text{beat}}=\mathbb{1}[V_{\text{norm}}>0].
$$

High-confidence label：

$$
Y_{\text{strong}}=\mathbb{1}[V_{\text{norm}}>q_{0.75}].
$$

### 4.3 Pre-event features

Tail / risk features：

```text
ce_tail_rank
margin_tail_rank
wrong_confidence_rank
curvature_rank
top2_margin
logit_norm_p95
nll_rank
ece_proxy_bucket
```

Movement features：

```text
pre_real_logit_delta_norm
pre_real_tail_logit_delta_norm
pre_real_nonadamw_delta_norm
pre_adamwparallel_logit_delta_norm
pre_bestlr_logit_delta_norm
pre_real_vs_adamw_delta_ratio
pre_real_vs_bestlr_delta_ratio
pre_tail_real_vs_control_ratio
pre_tail_nonadamw_ratio
```

Role / primitive features：

```text
branch_ratio
effective_derivative
functional_channel_entropy
dominant_basis_fraction
primitive_family
basis_usage_entropy
```

Alignment features：

```text
cos_real_adamw
cos_real_bestlr
cos_real_random
cos_tail_real_adamw
projected_tail_component_ratio
orthogonal_component_ratio
```

Uncertainty features：

```text
microbatch_score_variance
bootstrap_control_gap_std
horizon_agreement_score
event_score_entropy
stratum_sample_count
primitive_agreement_score
```

Forbidden features：

```text
dataset_name
seed_id
class_name as route key
posthoc outcome label at deployment
test metric
```

### 4.4 Predictors

所有 predictor 必须是 auditable controller，不是 task model，不参与 CE loss。

```text
EV0-OldStrataOnly:
  v9.2.29 reference.

EV1-GroundedLinear:
  ridge/logistic score on grounded value labels.

EV2-MonotoneSignalRule:
  monotone rule using tail severity, movement, control gap, uncertainty.

EV3-FamilyValuePredictor:
  predicts event-family value rather than single-event value.

EV4-ConformalAbstainController:
  accepts only if lower confidence bound of V_norm > 0.

EV5-HorizonAgreementController:
  accepts if multiple horizons agree on positive value.

EV6-OrthogonalTailController:
  accepts non-AdamW tail movement with low control dominance.

EV7-PrimitiveAgreementController:
  accepts if N2a/N2c/N3c agree on positive movement.

EV8-LDOCalibratedSignalController:
  thresholds calibrated by leave-dataset-out; no dataset_name.
```

### 4.5 Controls

Every paired replay / short-run / full-run must include：

```text
AdamWOnly
AdamWParallelTrustRatio-0.003
AdamWParallelTrustRatio-0.01
AdamWParallelTrustRatio-0.03
BestLRScale
NoOpMatchedOverhead
RandomMatchedNorm
ShuffledRoleMask
InvertedRoleMask
FrozenFuncChannel
ShuffledFuncChannel
DatasetRouteShuffled
EventRouteShuffled
SeverityBucketShuffled
SignalScoreShuffled
PreMovementShuffled
ControlGapShuffled
FamilyValueShuffled
UncertaintyShuffled
```

---

## 5. 实验阶段

## P0：v9.2.30 boundary reproduction

### 目标

复现当前 terminal boundary，确认不是 measurement noise。

### 必须记录

```text
route
source_route_v9229
p1_rows
pre_event_feature_pass
best_movement_abs_corr
failure_mode_reclassification_pass
attributed_fraction
abstain_precision
best_predictor
predictor_corr
predictor_precision
predictor_coverage
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R7-PrimitiveEffectUnpredictable
source route = R1-SignalStrataExplainFailures
pre-event feature pass = 0
best movement abs corr approximately 0.2997
predictor pass = 0
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_gate_ladder.svg
p0_corr_and_coverage_recap.svg
```

---

## P1：Event-value label grounding

### 目标

重定义 actual event value，不再用未归一化的单一 control gap 直接训练 predictor。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
primitives = N2a,N2c,N3c
events = S1-S8 signal strata
branches = RealFunctional, AdamWParallel, bestLR, NoOp, Random
```

### 必须记录

```text
event_id
signal_stratum
horizon
primitive
branch
raw_CEp99_delta
raw_margin_delta
raw_ECE_delta
raw_NLL_delta
raw_curvature_delta
raw_acc_delta
signed_gain_ce
signed_gain_margin
signed_gain_ece
signed_gain_nll
signed_gain_curvature
task_risk
control_relative_value
normalized_value
bootstrap_value_mean
bootstrap_value_std
split_half_value_1
split_half_value_2
value_reliability
```

### 判断标准

Event-value grounding pass：

$$
Corr(V_{\text{split1}},V_{\text{split2}})\geq0.30.
$$

Control-normalization pass：

$$
Var(V_{\text{norm}} | stratum)
\leq
Var(V_{\text{raw}} | stratum).
$$

如果 value reliability 低于 0.20，后续 predictor 不允许打开；必须先修 value label。

### 可视化

```text
p1_raw_vs_normalized_value.svg
p1_split_half_value_reliability.svg
p1_value_distribution_by_stratum.svg
p1_control_gap_variance.svg
```

---

## P2：Pre-event feature relevance and observability audit

### 目标

判断 pre-event features 是否真正可观察 event value。

### 必须记录

```text
event_id
signal_stratum
primitive
horizon
all_pre_event_features
V_raw
V_norm
V_family
feature_corr_raw
feature_corr_norm
feature_corr_family
feature_missing_rate
feature_stability_across_microbatch
```

### 判断标准

Feature relevance pass：

至少三个 feature 与 grounded value 满足：

$$
|Corr(f,V_{\text{norm}})|\geq0.20.
$$

至少一个 movement feature 满足：

$$
|Corr(f_{\text{movement}},V_{\text{norm}})|\geq0.30.
$$

如果 single-event 不过但 family-value 过，则记录：

```text
R1-FamilyValuePredictable
```

而不是直接判 primitive unpredictable。

### 可视化

```text
p2_feature_value_correlation_heatmap.svg
p2_movement_features_vs_value.svg
p2_uncertainty_features_vs_bad_event.svg
p2_single_vs_family_value_predictability.svg
```

---

## P3：Failure-mode reclassification with grounded value

### 目标

重新分类 failure，不允许出现 v9.2.30 的 attributed fraction = 0。

### Failure modes

```text
M1-Silent:
  movement too small.

M2-LowMagnitude:
  movement nonzero but metric effect tiny.

M3-ControlDominated:
  Real improves but AdamWParallel / bestLR improves more.

M4-Misaligned:
  Real moves tail but CE/margin worsen.

M5-HighUncertainty:
  event value variance too high.

M6-GoodEvent:
  Real beats controls with task safety.

M7-Abstain:
  no confident positive value.
```

### 必须记录

```text
event_id
primary_failure_mode
secondary_failure_mode
movement_ratio
tail_movement_ratio
control_relative_value
normalized_value
uncertainty
abstain_recommended
best_action
```

### 判断标准

P3 pass：

```text
>= 90% events assigned to exactly one primary failure mode
GoodEvent precision >= 0.70
Abstain precision >= 0.75
No mode uses dataset_name
```

### 可视化

```text
p3_failure_mode_sankey.svg
p3_failure_mode_by_stratum.svg
p3_good_event_vs_abstain_boundary.svg
```

---

## P4：Event-value predictor redesign

### 目标

训练 / 校准 official predictor，不使用 dataset name，不参与 loss。

### 设置

```text
predictors = EV0-EV8
labels = V_norm, Y_beat, Y_strong
validation = leave-dataset-out + leave-stratum-out + horizon split
```

### 必须记录

```text
predictor
feature_set
label_type
train_split
eval_split
corr_value
auc_beat
precision
recall
coverage
bad_event_rate
feature_importance
dataset_name_used
seed_id_used
posthoc_metric_used
```

### 判断标准

Predictor pass：

$$
Corr(S_{\text{pred}},V_{\text{norm}})\geq0.30
$$

or:

$$
AUC(Y_{\text{beat}})\geq0.65.
$$

Accepted event pass：

$$
Precision_{\text{accepted beats controls}}\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05.
$$

Anti-leak pass：

```text
dataset_name_used = 0
seed_id_used = 0
posthoc_metric_used = 0
```

### 可视化

```text
p4_predicted_vs_grounded_value.svg
p4_precision_recall_coverage.svg
p4_predictor_feature_importance.svg
p4_accepted_event_distribution.svg
```

---

## P5：Leave-dataset-out and leave-stratum-out validation

### 目标

防止 implicit dataset tuning。

### 设置

Leave-dataset-out：

```text
train on MNIST + Fashion, evaluate KMNIST
train on MNIST + KMNIST, evaluate Fashion
train on Fashion + KMNIST, evaluate MNIST
```

Leave-stratum-out：

```text
train on all but one signal stratum
evaluate held-out stratum
```

### 必须记录

```text
split_type
heldout
predictor
primitive_selector
thresholds
corr
precision
coverage
bad_event_rate
task_safe
CEp99_delta
margin_delta
curvature_delta
beats_adamwparallel
beats_bestlr
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

At least 70% held-out strata satisfy task safety and no worse CE tail：

$$
CEp99_{\text{heldout,Real}}\leq CEp99_{\text{AdamW}}+\epsilon.
$$

### 可视化

```text
p5_leave_dataset_out_matrix.svg
p5_leave_stratum_out_matrix.svg
p5_generalization_pareto.svg
p5_hidden_dataset_tuning_audit.svg
```

---

## P6：Official signal-value paired replay

### 目标

只允许 P4/P5 pass 的 controller 进入 official paired replay。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
controller = best P4/P5 survivor
primitive = selected by signal-value features, not dataset
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, InvertedRole, SeverityBucketShuffled, SignalScoreShuffled, PreMovementShuffled, ControlGapShuffled, FamilyValueShuffled, UncertaintyShuffled
```

### 必须记录

```text
controller
primitive
dataset
seed
horizon
signal_stratum
failure_mode
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
step_ratio_q90
memory_ratio
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
Acc_{\text{each dataset,Real}}\geq Acc_{\text{AdamW}}-0.005.
$$

Anti-overfit controls fail：

```text
SeverityBucketShuffled = fail
SignalScoreShuffled = fail
PreMovementShuffled = fail
ControlGapShuffled = fail
FamilyValueShuffled = fail
UncertaintyShuffled = fail
DatasetRouteShuffled = fail
EventRouteShuffled = fail
InvertedRoleMask = fail
```

System gate：

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
p6_anti_overfit_control_matrix.svg
p6_system_gate_distribution.svg
```

---

## P7：Short-run validation

### 目标

验证 local causality 是否能在连续训练中保持。

### 设置

```text
steps = 50,240,640
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, SignalScoreShuffled
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

Mechanism pass，至少一个：

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

## P8：Full 10-seed functional validation

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

Mechanism pass，至少一个：

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

### 可视化

```text
p8_macro_delta_vs_controls.svg
p8_hard_stratum_repair_matrix.svg
p8_seedwise_win_matrix.svg
p8_task_geometry_pareto.svg
p8_ce_tail_margin_panel.svg
p8_functional_channel_usage_trace.svg
```

---

## P9：AdamW-only full-pass repair

### 目标

Functional 是核心，但 base 仍未 full-pass。P9 并行执行，不使用 functional update，不使用 teacher/loss/sampler/class weight。

### Allowed repairs

```text
orthogonal lift init
fan-in output scale
centered / normalized T2
Legendre2-only
bounded rational base
piecewise local base
shared RBF base
dual-role basis with functional channel frozen during AdamW-only phase
hidden bracket h224/h256/h288
basis-balanced init
```

### 必须记录

```text
candidate
dataset
seed
test_acc
delta_vs_mlp
near_pass
full_pass
CEp99
margin_p10
ECE
NLL
basis_entropy
lift_condition_number
effective_rank
P4_step_q90
memory_ratio
```

### 判断标准

Full-pass：

$$
\Delta Acc_{\text{macro}}\geq0.
$$

Robust near-pass：

$$
near\_pass\_rate\geq0.80,
$$

$$
\Delta Acc_{\text{macro}}\geq-0.01.
$$

### 可视化

```text
p9_adamw_fullpass_gap.svg
p9_hard_stratum_miss_rows.svg
p9_ce_tail_margin.svg
p9_basis_entropy_vs_gap.svg
```

---

## P10：Robustness and external-ready gate

### 目标

确认 strict PureKAN functional advantage 不是 clean MNIST-family artifact。

### 设置

```text
label_noise = 0.05,0.10,0.20
input_noise = 0.05,0.10
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
```

Strong baselines：

```text
MLP-match
same-shape MLP
hidden-bracket MLP
QuadraticFeatureMLP
LQ/A7c AdamW-only
strict PureKAN functional candidate
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

### 可视化

```text
p10_noise_robustness_curve.svg
p10_strong_baseline_pareto.svg
p10_external_ready_scorecard.svg
```

---

## 6. Required artifacts

```text
run_manifest.json
contract_audit_v9231.csv
p0_v9230_boundary_reproduction.csv
p1_event_value_label_grounding.csv
p2_pre_event_feature_relevance.csv
p3_failure_mode_reclassification.csv
p4_event_value_predictor_redesign.csv
p5_leave_dataset_and_stratum_out_validation.csv
p6_official_signal_value_paired_replay.csv
p7_short_run_functional_validation.csv
p8_full_10seed_functional_validation.csv
p9_adamw_only_fullpass_repair.csv
p10_robustness_external_ready.csv
event_value_grounding_trace_v9231.csv
pre_event_feature_trace_v9231.csv
failure_mode_trace_v9231.csv
event_value_prediction_trace_v9231.csv
leave_dataset_out_trace_v9231.csv
paired_replay_branch_trace_v9231.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9230_boundary_unstable
F3_dataset_tuning_detected
F4_event_value_label_unreliable
F5_value_normalization_fail
F6_pre_event_features_not_predictive
F7_failure_mode_unattributed
F8_event_value_predictor_fail
F9_abstention_precision_or_coverage_fail
F10_leave_dataset_out_fail
F11_leave_stratum_out_fail
F12_signal_value_router_control_equivalent
F13_signal_value_router_overfit_shuffled_pass
F14_functional_lr_equivalent
F15_short_run_task_drop
F16_full_run_no_macro_hard_stratum_gain
F17_adamw_fullpass_fail
F18_strong_baseline_explains_gain
F19_robustness_fail
F20_external_not_ready
F21_fake_or_proxy_violation
F22_artifact_missing
```

---

## 7. Route decision

### Route cases

```text
R1-EventValueGrounded:
  value label reliability and normalization pass.

R2-PreEventFeaturesPredictValue:
  pre-event movement / uncertainty features predict grounded value.

R3-FamilyValuePredictable:
  single-event value noisy, but event-family value predictable.

R4-EventValuePredictorPass:
  predictor reaches corr / AUC / precision / coverage gates.

R5-LeaveDatasetOutPass:
  controller generalizes without dataset-specific tuning.

R6-SignalValuePairedReplayPass:
  official signal-value routed paired replay beats AdamWParallel / bestLR.

R7-AbstentionOnlyDiagnostic:
  precision high but coverage too low for training advantage.

R8-ControlDominatedSignal:
  Real has effect but strong optimizer controls dominate.

R9-PrimitiveEffectUnpredictable:
  grounded features cannot predict value; return to primitive/interface.

R10-StrictPureKANFunctionalShortRunPass:
  short-run task-safe mechanism gain.

R11-StrictPureKANFunctionalFullPass:
  full 10-seed macro / hard-stratum / geometry gain.

R12-AdamWFullPassNoFunctional:
  base reaches full-pass but functional remains unproven.

R13-ReturnToInterfacePrimitiveDesign:
  signal-value routing fails; no dataset-specific patch allowed.

R14-ExternalReady:
  strict PureKAN functional route passes task/geometry/system/control/robustness/strong-baseline gates.
```

### route_decision.json 必须记录

```text
route
v9230_boundary_pass
dataset_tuning_detected
event_value_grounding_pass
value_reliability
value_normalization_pass
pre_event_feature_pass
best_predictive_feature
best_predictor
event_value_predictor_pass
abstention_precision_pass
abstention_coverage_pass
leave_dataset_out_pass
leave_stratum_out_pass
paired_replay_pass
short_run_pass
full_run_pass
functional_task_safe
functional_control_pass
functional_system_pass
hard_stratum_repair_pass
adamw_fullpass
strong_baseline_pass
robustness_pass
external_ready
primary_blocker
next_required_implementation
success_v9231_strict_purekan_functional
success_v9231_full_functional
success_v9231_external_ready
```

---

## 8. 第一轮执行顺序

```text
Step 1:
  P0 复现 v9.2.30 boundary。

Step 2:
  P1 先修 event-value label。
  如果 value reliability 不过，不允许继续 predictor sweep。

Step 3:
  P2 做 pre-event feature relevance。
  判断 v9.2.30 near-threshold movement corr 是否在 grounded value 下变成 pass。

Step 4:
  P3 重新分类 failure modes。
  必须解决 v9.2.30 attributed fraction = 0 的问题。

Step 5:
  P4 训练/校准 event-value predictor。
  不使用 dataset_name，重点预测 Real-vs-control gap。

Step 6:
  P5 做 leave-dataset-out 和 leave-stratum-out。
  防止隐式 dataset tuning。

Step 7:
  P6 official signal-value paired replay。
  加入 FamilyValueShuffled / UncertaintyShuffled 等强 controls。

Step 8:
  P7 short-run validation。

Step 9:
  P8 full 10-seed validation。

Step 10:
  P9 并行 AdamW-only full-pass repair。

Step 11:
  P10 robustness / strong baseline / external-ready。
```

---

## 9. 停止条件

### Minimum diagnostic success

```text
v9.2.30 boundary reproduced
no dataset-specific route used
event-value label grounded
pre-event features predict grounded value or event-family value
failure modes attributed without dataset name
no fake/proxy/offload/loss/teacher violation
```

### Local functional success

```text
Minimum diagnostic success
+
event-value predictor passes corr / AUC / precision / coverage gates
+
leave-dataset-out pass
+
official paired replay beats AdamWParallel / bestLR
```

### Full functional success

```text
Local functional success
+
short/full run task-safe mechanism gain
+
macro or hard-stratum improvement
```

### External-ready success

```text
Full functional success
+
robustness pass
+
strong baseline challenge pass
```

### Failure stop

```text
1. v9.2.30 boundary cannot be reproduced；
2. event-value label reliability < 0.20；
3. value normalization does not reduce noise；
4. pre-event features remain non-predictive；
5. failures cannot be classified without dataset name；
6. event-value predictor cannot reach corr >= 0.30 or AUC >= 0.65；
7. high-precision abstention remains too low coverage；
8. leave-dataset-out fails；
9. signal-value router remains control-equivalent；
10. shuffled signal controls pass, indicating routing overfit；
11. full run gives no macro / hard-stratum / geometry gain；
12. functional breaks system gate；
13. gains are explained by QuadraticFeatureMLP；
14. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 10. 最终解释规则

### Case A：value grounding fixes predictor

可以声明：

```text
v9.2.30 failed because event-value label and normalization were insufficient, not because functional primitive was necessarily unpredictable.
```

但必须继续 P6/P7/P8 才能声明 functional success。

### Case B：single-event value fails but family value passes

可以声明：

```text
Functional event value is not reliably predictable at single-event granularity, but event-family value is predictable.
```

下一步应使用 family-level controller / abstention，而不是 dataset-specific route。

### Case C：high precision but low coverage

必须声明：

```text
Functional update can identify rare good events, but current coverage is insufficient for training advantage.
```

这不是 success，只是 diagnostic。

### Case D：leave-dataset-out fails

必须声明：

```text
Controller is implicitly dataset-specific; official route not allowed.
```

不能用 dataset tuning 写成功。

### Case E：all grounded predictors fail

必须声明：

```text
Current primitive/interface does not provide predictable causal actuatability; return to primitive/interface redesign.
```

---

## 11. 最终建议

v9.2.31 的一句话策略是：

$$
\boxed{
\text{先把 event value 定义和测量做可靠，再判断 pre-event features 是否真的无法预测 functional causal gain。}
}
$$

当前最关键的问题不是：

```text
Fashion 怎么调过；
KMNIST 怎么调过；
MNIST 是否要 abstain；
```

而是：

```text
1. actual Real-vs-control value 的 label 是否可靠？
2. pre-event movement features 在 grounded value 下是否仍接近/超过 predictive gate？
3. 单事件不可预测时，event-family value 是否可预测？
4. high-precision abstention 能否提升 coverage？
5. 这种 signal-value controller 能否 leave-dataset-out 泛化？
6. 如果不能，是否说明 current strict PureKAN functional primitive 缺少可预测的 causal actuatability？
```
