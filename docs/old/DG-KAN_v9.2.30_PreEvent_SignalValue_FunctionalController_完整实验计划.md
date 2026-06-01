# DG-KAN v9.2.30 Pre-Event Signal-Value Functional Controller：从 Signal-Strata 诊断到可泛化事件价值预测的完整实验计划

> 本计划基于 v9.2.29 `Dataset-Agnostic Functional Causality` 的真实执行结果制定。  
> v9.2.29 的 terminal route 是：
>
> ```text
> route = R1-SignalStrataExplainFailures
> base_candidate = LQ-t2-h256
> success_v9229_strict_purekan_functional = False
> success_v9229_full_functional = False
> success_v9229_external_ready = False
> ```
>
> 本轮最重要的事实是：**failure 已经可以被 dataset-agnostic signal strata 解释，但 event-value predictor 没有达到 correlation gate。**  
> 换句话说，我们已经从“Fashion/KMNIST/MNIST 各自怎么调”推进到“什么样的 signal event 值得 functional update”。这是正确方向，但还没有形成可用 controller。
>
> 用户特别强调：**可以诊断不同数据集的问题，但不能针对数据集调参，因为目标不是在这些数据集上打榜。**  
> 因此 v9.2.30 的 official route 禁止 dataset-name branch。数据集只能用于诊断分层、leave-dataset-out 验证和反过拟合审计，不能作为 functional controller 的条件。
>
> 本计划继续遵守：
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
> 训练目标保持标准 CE：
>
> $$
> L_{\text{task}}=CE(y,p_\theta(x)).
> $$

---

## 0. 当前实验结果的独立判断

### 0.1 v9.2.29 没有达到 strict PureKAN functional success

v9.2.29 的核心结果是：

```text
P0:
  v9.2.28 boundary reproduced
  source route = R5-KMNISTSurvivorNotPreserved
  fake/proxy = 0

P1:
  signal strata explain failure pass
  association margin = 0.256620
  p1 rows = 2628

P2:
  event-value predictor corr = 0.197001
  gate requires corr >= 0.30
  event-value predictor pass = 0

P2 abstention:
  precision = 0.830189
  coverage = 0.020167
  accepted event count = 53
  recall = 0.067485
  bad-event = 0.000000

Downstream:
  leave-dataset-out = not_opened
  paired replay = not_opened
  short-run = not_opened
  full-run = not_opened
  external-ready = not_opened
```

因此，v9.2.29 只能说明：

$$
\boxed{
\text{failure modes 可以被 signal strata 解释，但 controller 还不能可靠预测 event value。}
}
$$

不能声明：

```text
strict PureKAN functional update success
paired replay causal success
short-run / full-run functional success
external-ready
```

### 0.2 v9.2.29 的真实进展

v9.2.29 有一个非常重要的正结果：**没有走向 dataset tuning。**

上一轮 v9.2.28 的状态是：

```text
Fashion:
  effect exists but control-dominated

KMNIST:
  diagnosis survivor not preserved under stronger replay

Routed:
  best routed = 0.5 / 0.5
  pass = 0
```

v9.2.29 没有继续写：

```text
if dataset == Fashion: use Fashion controller
if dataset == KMNIST: use KMNIST primitive
if dataset == MNIST: abstain
```

而是把不同数据集暴露的问题转成了 signal strata。P1 的 association margin 为正，说明 signal stratum 对 failure mode 的解释力强于 dataset label。这是路线上的关键进展。

### 0.3 v9.2.29 的失败点

P2 的连续 event-value prediction 没过：

$$
Corr(S_{\text{pred}}, S_{\text{actual}})=0.197001<0.30.
$$

但 abstention 局部可行：

$$
Precision_{\text{accepted beats controls}}=0.830189,
$$

$$
Coverage=0.020167,
$$

$$
BadEventRate=0.
$$

这说明现在不是完全没有可用事件。更精确地说：

$$
\boxed{
\text{我们能识别少量高置信好事件，但不能稳定排序大多数事件的 Real-vs-control value。}
}
$$

这就是当前 blocker：

```text
event_value_predictor_correlation_below_gate
```

### 0.4 当前最重要的新判断

v9.2.29 已经把路线从 dataset-specific patch 收紧为 dataset-agnostic signal routing，这是正确方向。  
但是当前 predictor 输入仍然不够，它能做高精度低覆盖 abstention，却不能连续预测 event value。下一步的本质不是再做 Fashion/KMNIST 调参，而是引入 **pre-event realized movement features**：

```text
before committing update:
  estimate RealFunctional 的 actual logit movement；
  estimate tail logit movement；
  estimate non-AdamW movement；
  estimate Real vs AdamWParallel / best LR gap；
  estimate branch ratio / effective derivative / uncertainty。
```

如果这些 feature 仍然不能把 correlation 提升到 gate，说明当前 functional primitive/interface 还不能给出可预测的 useful update，需要回到 primitive/interface 设计，而不是数据集特化。

---

## 1. v9.2.30 总体目标

v9.2.30 的总体目标是：

$$
\boxed{
\text{建立不使用 dataset name 的 pre-event signal-value predictor，并让 official signal-routed paired replay 通过。}
}
$$

这个目标有三层。

### 1.1 Diagnostic success

诊断成功要求解释 v9.2.29 的 P2 failure：

```text
为什么 signal strata 能解释 failure，
但 event-value predictor correlation 只有 0.197？
```

本轮必须把失败归因到以下机制之一：

```text
D1-missing_pre_event_movement_features:
  predictor 缺少 event-time 实际 movement 估计。

D2-noisy_control_gap_labels:
  Real-vs-control gap 本身方差过大，单事件难预测。

D3-insufficient_coverage:
  high-precision accepted events 太少，coverage 仅约 0.02。

D4-signal_stratum_too_coarse:
  strata 能解释 failure family，但不足以排序 event value。

D5-primitive_effect_unpredictable:
  当前 primitive/interface 的 functional movement 不稳定，不适合 controller。

D6-control_dominance:
  Real 有 effect，但 AdamWParallel / best LR 在多数 signal event 上仍更强。
```

### 1.2 Event-value predictor success

新的 predictor 必须满足：

$$
Corr(S_{\text{pred}},S_{\text{actual}})\geq0.30.
$$

并且 accepted event policy 必须满足：

$$
Precision_{\text{accepted beats controls}}\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05.
$$

注意这里 coverage 下限从 v9.2.29 的 $0.02$ 提高到 $0.03$，因为 v9.2.29 虽然 precision 高，但 recall 太低：

$$
Recall=0.067485.
$$

v9.2.30 的目标不是只找到极少数好事件，而是找到足够多、能进入 paired replay 和 short-run 的事件。

### 1.3 Local functional causality success

只有 P2/P3 predictor 通过后，official paired replay 才能打开。成功要求：

$$
BeatRate_{\text{macro,Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro,Real vs best LR}}\geq0.60.
$$

并且每个数据集都 task-safe：

$$
Acc_{\text{Real,dataset}}\geq Acc_{\text{AdamW,dataset}}-0.005.
$$

但 official policy 不能使用 dataset name。MNIST/Fashion/KMNIST 只作为 evaluation partitions，不作为 route branch。

### 1.4 Full functional success

只有 official paired replay 通过后，才打开 short/full validation。Full success 要求：

$$
\Delta Acc_{\text{functional-vs-AdamW}}\geq-0.005.
$$

并至少一个：

$$
\Delta Acc_{\text{macro,functional}}
-
\Delta Acc_{\text{macro,AdamW}}
\geq0.003,
$$

$$
\Delta Acc_{\text{hard-stratum,functional}}
-
\Delta Acc_{\text{hard-stratum,AdamW}}
\geq0.005,
$$

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamW}},
$$

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamW}},
$$

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

其中 hard-stratum 必须由 pre-registered signal features 定义，而不是 dataset name。

---

## 2. 核心假设

### H1：v9.2.29 predictor 失败是因为缺少 pre-event realized movement features

v9.2.29 的 P1 signal strata 通过，说明 failure family 可解释；P2 predictor correlation 低，说明只靠 coarse strata / static tail severity 不足以排序 event value。

H1 认为必须加入 event-time movement estimates：

$$
r_z
=
\frac{\|\Delta z_{\text{Real}}\|}
{\|\Delta z_{\text{AdamWParallel}}\|+\epsilon},
$$

$$
r_{z,\text{tail}}
=
\frac{\|\Delta z_{\text{Real,tail}}\|}
{\|\Delta z_{\text{AdamWParallel,tail}}\|+\epsilon},
$$

$$
r_{z,\perp}
=
\frac{
\|\Delta z_{\text{Real}}-\operatorname{proj}_{\Delta z_{\text{AdamW}}}\Delta z_{\text{Real}}\|
}{
\|\Delta z_{\text{AdamW}}\|+\epsilon
}.
$$

H1 成立标准：

加入这些 features 后：

$$
Corr(S_{\text{pred}},S_{\text{actual}})\geq0.30.
$$

### H2：当前 high-precision abstention 可行，但 coverage 太低

v9.2.29 的 accepted precision 是 $0.830189$，bad-event 为 $0$，这说明 high-confidence accepted events 真实存在。  
但 coverage 只有 $0.020167$，recall 只有 $0.067485$，不足以支撑 short-run / full-run。

H2 成立标准：

通过 pre-event features 和 uncertainty calibration，把 coverage 提高到：

$$
Coverage\in[0.03,0.15],
$$

同时保持：

$$
Precision\geq0.75.
$$

### H3：control dominance 需要被显式建模，而不是事后比较

Fashion 与 KMNIST 都出现过 “Real 有 effect，但 AdamWParallel / best LR 更强”。这说明 predictor 不能只预测 Real 是否改善，还要预测：

$$
S_{\text{gap}}
=
Gain_{\text{Real}}
-
\max(Gain_{\text{AdamWParallel}},Gain_{\text{bestLR}}).
$$

H3 成立标准：

gap predictor 的 correlation 达到：

$$
Corr(S_{\text{gap,pred}},S_{\text{gap,actual}})\geq0.30.
$$

accepted events 中：

$$
Precision_{\text{Real beats controls}}\geq0.75.
$$

### H4：官方 controller 必须 leave-dataset-out 泛化

不能针对数据集调参，因此所有 thresholds / weights / calibration 必须在 leave-dataset-out 下验证。

H4 成立标准：

对三个 split：

```text
train-controller on MNIST + Fashion, evaluate KMNIST
train-controller on MNIST + KMNIST, evaluate Fashion
train-controller on Fashion + KMNIST, evaluate MNIST
```

均满足：

$$
Acc_{\text{eval,Real}}\geq Acc_{\text{eval,AdamW}}-0.005.
$$

至少两个 split 满足：

$$
BeatRate_{\text{Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{Real vs best LR}}\geq0.50.
$$

### H5：如果 pre-event predictor 仍失败，问题在 primitive/interface，而不是数据集 controller

如果加入 pre-event movement、tail movement、control gap、uncertainty 后，correlation 仍低于 $0.30$，则说明当前 functional primitive 的 movement 不可预测或不稳定。下一步必须回到：

```text
functional primitive/interface redesign
basis channel redesign
AdamW-only full-pass repair
```

而不是继续 Fashion/KMNIST 特化调参。

---

## 3. Candidate 设计

### 3.1 Baselines

```text
B0-MLP-match:
  same-parameter MLP reference.

LQ0-LQ-t2-h256:
  strict FC-PureKAN AdamW-only base.

N2a-Rational:
  current base-qualified rational functional channel.

N2c-SharedRBF:
  local smooth functional channel.

N3c-SharedRBFDerivativeBand:
  derivative-controlled local channel.

V8-FT7:
  mechanism source only, not official strict candidate.
```

### 3.2 Pre-event features

每个 event 必须记录以下 feature。它们都必须在 update commit 之前可计算。

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
stratum_sample_count
event_score_entropy
```

Forbidden features：

```text
dataset_name
seed_id
class_name as route key
posthoc outcome label at deployment
test metric
```

### 3.3 Signal-value predictor candidates

```text
EV0-OldStrataOnly:
  v9.2.29 reference.

EV1-PreMovementLinear:
  linear/ridge score using pre-event movement and tail features.

EV2-MonotoneRuleScore:
  hand-auditable monotone score:
  tail severity + actual movement - control dominance - uncertainty.

EV3-ControlGapLogistic:
  logistic predictor of Real beats AdamWParallel / bestLR.

EV4-QuantileAbstainScore:
  accepts only top quantile of predicted control gap.

EV5-HorizonAgreementScore:
  requires h20/h80/h240 prediction agreement.

EV6-OrthogonalTailScore:
  rewards non-AdamW tail movement, penalizes AdamW-parallel movement.

EV7-PrimitiveEnsembleScore:
  compares N2a/N2c/N3c pre-event movement, chooses primitive only by signal features.

EV8-LDOCalibratedScore:
  thresholds calibrated by leave-dataset-out, no dataset labels in score.
```

### 3.4 Primitive candidates

Primitive selection must be signal-based, not dataset-based.

```text
P0-N2a-Rational
P1-N2c-SharedRBF
P2-N3c-SharedRBFDerivativeBand
P3-N2a-OrthogonalTail
P4-N2c-OrthogonalTail
P5-LightHybrid-RationalSharedRBF
P6-LightHybrid-RationalPiecewise
```

Official primitive selector:

```text
choose primitive by event signal score,
not by dataset identity.
```

### 3.5 Controls

Every replay/short/full run must include:

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
```

New controls:

```text
PreMovementShuffled:
  keep event strata but shuffle movement features.

ControlGapShuffled:
  keep movement features but shuffle predicted Real-vs-control gap.

Purpose:
  ensure predictor is using causal signal, not row distribution artifacts.
```

---

## 4. 实验阶段

## P0：v9.2.29 boundary reproduction

### 目标

复现 v9.2.29 terminal boundary，确认当前不是测量噪声。

### 必须记录

```text
route
source_route_v9228
association_margin
dataset_failure_association
signal_strata_failure_association
event_value_corr
accepted_precision
accepted_coverage
accepted_recall
accepted_bad_event_rate
accepted_event_count
event_value_predictor_pass
abstention_precision_pass
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R1-SignalStrataExplainFailures
signal strata explain failure = 1
event_value_predictor_pass = 0
abstention local pass = 1
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_gate_ladder.svg
p0_predictor_failure_recap.svg
```

---

## P1：Pre-event realized movement feature extraction

### 目标

补齐 v9.2.29 predictor 缺失的 pre-event movement features。  
P1 不训练新 controller，只测 features 是否能解释 actual Real-vs-control gap。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
primitives = P0,P1,P2
events = signal strata S1-S8
branches = RealFunctional, AdamWParallel, bestLR, NoOp, Random
```

### 必须记录

```text
event_id
dataset
seed
horizon
signal_stratum
primitive
pre_real_logit_delta_norm
pre_real_tail_logit_delta_norm
pre_real_nonadamw_delta_norm
pre_adamwparallel_logit_delta_norm
pre_bestlr_logit_delta_norm
pre_real_vs_adamw_delta_ratio
pre_real_vs_bestlr_delta_ratio
pre_tail_real_vs_control_ratio
branch_ratio
effective_derivative
functional_channel_entropy
cos_real_adamw
cos_real_bestlr
microbatch_score_variance
horizon_agreement_score
actual_real_minus_adamwparallel
actual_real_minus_bestlr
actual_control_gap
```

### 判断标准

Feature relevance pass：

至少三个 feature 与 actual control gap 有 $|corr|\geq0.20$，并且至少一个 movement feature 满足：

$$
|Corr(f_{\text{movement}},S_{\text{actual}})|\geq0.30.
$$

如果所有 movement features 都低于 $0.20$，则说明 current primitive 的 event effect 不可预测，不能继续做 controller sweep。

### 可视化

```text
p1_feature_actual_gap_correlation.svg
p1_tail_movement_vs_control_gap.svg
p1_primitive_movement_pareto.svg
p1_uncertainty_vs_bad_event.svg
```

---

## P2：Failure-mode reclassification with pre-event features

### 目标

把 v9.2.29 的 coarse signal strata 细化为可用于 controller 的 failure modes。

### 必须记录

```text
event_id
signal_stratum_old
signal_stratum_new
failure_mode
silent
misaligned
control_dominated
low_magnitude
high_uncertainty
route_overfit_risk
abstain_recommended
best_candidate_action
```

### Failure mode definitions

Silent：

$$
r_{z,\text{tail}}<0.10.
$$

Low magnitude：

$$
r_{z,\text{tail}}\geq0.10
$$

but:

$$
|\Delta CEp99_{\text{Real}}-\Delta CEp99_{\text{NoOp}}|<0.0005
$$

and:

$$
|\Delta Margin_{\text{Real}}-\Delta Margin_{\text{NoOp}}|<0.001.
$$

Control dominated：

$$
Gain_{\text{Real}}>0
$$

but:

$$
Gain_{\text{Real}}<Gain_{\text{AdamWParallel}}
$$

or:

$$
Gain_{\text{Real}}<Gain_{\text{bestLR}}.
$$

Misaligned：

$$
r_{z,\text{tail}}\geq0.10
$$

but:

$$
\Delta CEp99_{\text{Real}}\geq0
$$

and:

$$
\Delta Margin_{\text{Real}}\leq0.
$$

High uncertainty：

$$
\sigma_{\text{microbatch score}}
>
\sigma_{\max}.
$$

### 判断标准

P2 pass：

```text
>= 90% replay events assigned to exactly one primary failure mode
abstain recommendation precision >= 0.75
no failure mode uses dataset_name
```

### 可视化

```text
p2_failure_mode_sankey.svg
p2_old_strata_to_new_modes.svg
p2_abstain_recommendation_precision.svg
```

---

## P3：Event-value predictor redesign

### 目标

训练或校准不使用 dataset_name 的 event-value predictor。  
P3 的 predictor 是 controller，不是 teacher，不参与 task loss，不改变 CE。

### 设置

```text
predictors = EV0-EV8
features = P1 pre-event features
labels = actual Real-vs-control gap from replay only
forbidden = dataset_name, seed_id, test metric
validation = leave-dataset-out + time/horizon split
```

### 必须记录

```text
predictor
feature_set
train_split
eval_split
corr_real_vs_adamwparallel
corr_real_vs_bestlr
corr_gap
precision
recall
coverage
bad_event_rate
accepted_event_count
abstained_event_count
feature_importance
dataset_name_used
seed_id_used
```

### 判断标准

Predictor pass：

$$
Corr(S_{\text{pred}},S_{\text{actual}})\geq0.30.
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
posthoc metric used at deployment = 0
```

### 可视化

```text
p3_predicted_vs_actual_gap.svg
p3_precision_recall_coverage.svg
p3_predictor_feature_importance.svg
p3_accepted_event_distribution.svg
```

---

## P4：Leave-dataset-out and leave-stratum-out validation

### 目标

防止 event-value predictor 隐式针对数据集调参。  
即使没有 dataset_name，feature distribution 也可能隐式过拟合，所以必须做 LDO 和 LSO。

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
primitive
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

At least two LDO splits satisfy:

$$
BeatRate_{\text{heldout,Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{heldout,Real vs bestLR}}\geq0.50.
$$

LSO pass：

At least 70% held-out strata satisfy task safety and no worse CE tail:

$$
CEp99_{\text{heldout,Real}}\leq CEp99_{\text{AdamW}}+\epsilon.
$$

### 可视化

```text
p4_leave_dataset_out_matrix.svg
p4_leave_stratum_out_matrix.svg
p4_generalization_pareto.svg
```

---

## P5：Official signal-routed paired replay

### 目标

只有 P3/P4 pass 的 predictor 才能进入 official paired replay。  
这是 local functional causality gate。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
controller = best P3/P4 survivor
primitives = selected by signal features, not dataset
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, InvertedRole, SeverityBucketShuffled, SignalScoreShuffled, PreMovementShuffled, ControlGapShuffled
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
real_beats_best_lr
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
p5_official_paired_replay_pareto.svg
p5_macro_beat_rate.svg
p5_signal_stratum_win_matrix.svg
p5_anti_overfit_control_matrix.svg
p5_system_gate_distribution.svg
```

---

## P6：Short-run validation

### 目标

验证 local causality 是否能在连续训练中保持。只有 P5 survivor 进入。

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

Mechanism pass：

At least one:

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
p6_short_run_task_mechanism_pareto.svg
p6_short_run_controls.svg
p6_event_timeline.svg
p6_ce_tail_margin_panel.svg
```

---

## P7：Full 10-seed functional validation

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

Mechanism pass：

At least one:

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
p7_macro_delta_vs_controls.svg
p7_hard_stratum_repair_matrix.svg
p7_seedwise_win_matrix.svg
p7_task_geometry_pareto.svg
p7_ce_tail_margin_panel.svg
p7_functional_channel_usage_trace.svg
```

---

## P8：AdamW-only full-pass repair

### 目标

Functional 是核心，但 base 仍未 full-pass。P8 并行执行，不使用 functional update，不使用 teacher/loss/sampler/class weight。

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
p8_adamw_fullpass_gap.svg
p8_hard_stratum_miss_rows.svg
p8_ce_tail_margin.svg
p8_basis_entropy_vs_gap.svg
```

---

## P9：Robustness and external-ready gate

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
p9_noise_robustness_curve.svg
p9_strong_baseline_pareto.svg
p9_external_ready_scorecard.svg
```

---

## 5. Required artifacts

```text
run_manifest.json
contract_audit_v9230.csv
p0_v9229_boundary_reproduction.csv
p1_pre_event_movement_features.csv
p2_failure_mode_reclassification.csv
p3_event_value_predictor_redesign.csv
p4_leave_dataset_and_stratum_out_validation.csv
p5_official_signal_routed_paired_replay.csv
p6_short_run_functional_validation.csv
p7_full_10seed_functional_validation.csv
p8_adamw_only_fullpass_repair.csv
p9_robustness_external_ready.csv
pre_event_feature_trace_v9230.csv
event_value_prediction_trace_v9230.csv
leave_dataset_out_trace_v9230.csv
paired_replay_branch_trace_v9230.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9229_boundary_unstable
F3_dataset_tuning_detected
F4_pre_event_feature_extraction_fail
F5_pre_event_features_not_predictive
F6_failure_mode_unattributed
F7_event_value_predictor_fail
F8_abstention_coverage_or_precision_fail
F9_leave_dataset_out_fail
F10_leave_stratum_out_fail
F11_signal_router_control_equivalent
F12_signal_router_overfit_shuffled_pass
F13_functional_lr_equivalent
F14_short_run_task_drop
F15_full_run_no_macro_hard_stratum_gain
F16_adamw_fullpass_fail
F17_strong_baseline_explains_gain
F18_robustness_fail
F19_external_not_ready
F20_fake_or_proxy_violation
F21_artifact_missing
```

---

## 6. Route decision

### Route cases

```text
R1-PreEventFeaturesExplainValue:
  pre-event movement features explain Real-vs-control gap.

R2-EventValuePredictorPass:
  predictor reaches corr / precision / coverage gates.

R3-LeaveDatasetOutPass:
  controller generalizes without dataset-specific tuning.

R4-SignalRoutedPairedPass:
  official signal-routed paired replay beats AdamWParallel / best LR.

R5-AbstentionOnlyDiagnostic:
  high precision but coverage too low; not enough for functional success.

R6-ControlDominatedSignal:
  Real has effect but strong optimizer controls dominate.

R7-PrimitiveEffectUnpredictable:
  movement features cannot predict event value; return to interface/primitive.

R8-StrictPureKANFunctionalShortRunPass:
  short-run task-safe mechanism gain.

R9-StrictPureKANFunctionalFullPass:
  full 10-seed macro / hard-stratum / geometry gain.

R10-AdamWFullPassNoFunctional:
  base reaches full-pass but functional remains unproven.

R11-ReturnToInterfacePrimitiveDesign:
  signal routing fails; no dataset-specific patch allowed.

R12-ExternalReady:
  strict PureKAN functional route passes task/geometry/system/control/robustness/strong-baseline gates.
```

### route_decision.json 必须记录

```text
route
v9229_boundary_pass
dataset_tuning_detected
pre_event_feature_pass
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
success_v9230_strict_purekan_functional
success_v9230_full_functional
success_v9230_external_ready
```

---

## 7. 第一轮执行顺序

```text
Step 1:
  P0 复现 v9.2.29 boundary。

Step 2:
  P1 提取 pre-event movement features。
  如果 movement features 与 actual control gap 完全无关，停止 controller sweep。

Step 3:
  P2 用 pre-event features 重新归类 failure mode。
  目标是分清 silent / low-magnitude / control-dominated / misaligned / high-uncertainty。

Step 4:
  P3 重设计 event-value predictor。
  不使用 dataset_name；重点预测 Real-vs-control gap。

Step 5:
  P4 做 leave-dataset-out 和 leave-stratum-out 验证。
  防止隐式 dataset tuning。

Step 6:
  P5 official signal-routed paired replay。
  加入 PreMovementShuffled / ControlGapShuffled 等强 controls。

Step 7:
  P6 short-run validation。

Step 8:
  P7 full 10-seed validation。

Step 9:
  P8 并行 AdamW-only full-pass repair。

Step 10:
  P9 robustness / strong baseline / external-ready。
```

---

## 8. 停止条件

### Minimum success

```text
v9.2.29 boundary reproduced
no dataset-specific route used
pre-event features improve event-value prediction
event-value predictor passes corr / precision / coverage gates
official signal-routed paired replay beats AdamWParallel / best LR
no fake/proxy/offload/loss/teacher violation
```

### Full functional success

```text
Minimum success
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
1. v9.2.29 boundary cannot be reproduced；
2. pre-event features are not predictive；
3. failures cannot be classified without dataset name；
4. event-value predictor cannot reach corr >= 0.30；
5. high-precision abstention remains too low coverage；
6. leave-dataset-out fails；
7. signal router remains control-equivalent；
8. shuffled signal controls pass, indicating routing overfit；
9. full run gives no macro / hard-stratum / geometry gain；
10. functional breaks system gate；
11. gains are explained by QuadraticFeatureMLP；
12. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 9. 最终解释规则

### Case A：pre-event features improve predictor

可以声明：

```text
v9.2.29 failed because controller lacked event-time movement features; functional update remains viable through signal-value prediction.
```

但还不能声明 strict functional success unless paired replay / short/full pass.

### Case B：abstention high precision but low coverage

必须声明：

```text
Functional update can identify rare good events, but current coverage is insufficient for training advantage.
```

这只能是 diagnostic success，不是 full functional success。

### Case C：leave-dataset-out fails

必须声明：

```text
Controller is implicitly dataset-specific; official route not allowed.
```

不能用 dataset tuning 写成功。

### Case D：paired replay passes

可以声明：

```text
Strict PureKAN functional has dataset-agnostic local causality evidence.
```

但 full success 仍需 short/full validation。

### Case E：all predictors fail

必须声明：

```text
Current primitive/interface does not provide predictable functional event value; return to primitive/interface redesign.
```

---

## 10. 最终建议

v9.2.30 的一句话策略是：

$$
\boxed{
\text{从“signal strata 能解释失败”推进到“pre-event signal-value 能预测何时 functional update 应该触发”。}
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
1. 哪些 pre-event movement features 能预测 RealFunctional 是否超过 AdamWParallel / best LR？
2. 如何在不使用 dataset name 的情况下提高 accepted event coverage？
3. 哪些事件应该 abstain？
4. 这种 signal-value rule 能否 leave-dataset-out 泛化？
5. 如果不能，是否说明 current strict PureKAN functional primitive 仍缺少可预测的 causal actuatability？
```
