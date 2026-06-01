# DG-KAN v9.2.32 Probe-to-Commit Functional Controller 与 Observable Primitive Repair 完整实验计划

> 本计划基于 v9.2.31 `Event-Value Grounding 与 Primitive Observability` 的真实复盘制定。  
> v9.2.31 的 terminal route 是：
>
> ```text
> route = R9-PrimitiveEffectUnpredictable
> base_candidate = LQ-t2-h256
> success_v9231_strict_purekan_functional = False
> success_v9231_full_functional = False
> success_v9231_external_ready = False
> ```
>
> v9.2.31 最重要的事实是：**event-value label 已经 grounded，可靠性过关；但当前 pre-event features 仍不能预测 grounded value。**  
> 因此，下一步不能继续针对 Fashion / KMNIST / MNIST 做调参，也不能继续简单换 predictor。当前要解决的是更本质的问题：
>
> $$
> \boxed{
> \text{functional update 在 commit 前是否有可观测、可验证、可泛化的 causal value？}
> }
> $$
>
> 本轮的核心方向是：从 “静态 pre-event features 预测 value” 升级为 **probe-to-commit controller**。  
> 也就是说，在真正提交 functional update 前，用训练流内部的极小成本 virtual probe / leave-one-out signal estimate / control-contrastive probe 判断：
>
> $$
> \boxed{
> \text{RealFunctional 是否会比 AdamWParallel / best LR 更有价值。}
> }
> $$
>
> 如果 probe-to-commit 仍失败，则说明 current primitive/interface 没有可观测 causal actuatability，必须进入 observable primitive redesign，而不是继续 dataset-specific patch。
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

## 0. 当前结果的独立判断

### 0.1 v9.2.31 达成了什么

v9.2.31 不是完全失败。它完成了 v9.2.30 没完成的一件关键事：**event-value label grounding 通过**。核心结果是：

```text
P1 grounded value reliability = 0.511631
event-value grounding pass = 1
```

这说明 v9.2.30 的失败不能继续简单归因于 “value label 不可靠”。  
经过 control-relative / normalized value grounding 后，event value 本身已经有可测可靠性。

换句话说：

$$
\boxed{
\text{我们现在有了相对可靠的 event-value label。}
}
$$

这一步非常重要，因为如果 label 都不可靠，任何 predictor / controller 都没有意义。v9.2.31 把这个问题推进了。

### 0.2 v9.2.31 没有达成什么

v9.2.31 没有证明 current primitive 的 functional value 可预测。核心失败是：

```text
P2 pre-event observability pass = 0
best feature = effective_derivative
best movement feature = pre_adamwparallel_logit_delta_norm
best movement abs corr = 0.184709

P3 failure-mode reclassification pass = 0
GoodEvent precision = 0.000000
abstain precision = 0.000000

P4 event-value predictor pass = 0
best predictor = not_opened
corr = 0.000000
precision = 0.000000
coverage = 0.000000
```

当前 blocker 是：

```text
pre_event_features_do_not_predict_grounded_value
```

也就是说：**event value 本身可靠了，但现有 pre-event features 仍然读不出这个 value。**

### 0.3 当前最重要的判断

v9.2.31 之后，问题已经不能再写成：

```text
event-value label 不可靠；
signal strata 太粗；
Fashion / KMNIST 数据集需要分别调；
functional channel 完全不能动；
strict PureKAN interface 不能 P4/P5；
v8 functional core 不存在。
```

这些都不是当前主 blocker。

当前真正 blocker 是：

$$
\boxed{
\text{current features observe不到 functional update 的 causal value。}
}
$$

这有两种可能：

1. **观测协议不够强**：静态 pre-event features 不够，需要 virtual probe / control-contrastive probe 才能看出 event value。
2. **primitive 本身不可观测**：当前 N2a/N2c/N3c 这类 functional channel 的因果效果对 commit 前可计算信息不稳定，说明需要 observable primitive redesign。

v9.2.32 的任务就是区分这两种情况。

---

## 1. v9.2.32 总体目标

v9.2.32 的总体目标是：

$$
\boxed{
\text{建立不使用 dataset name 的 probe-to-commit functional controller；若失败，证明需要 observable primitive redesign。}
}
$$

本轮不追求直接 full-run。必须先完成 local causality gate。

### 1.1 Diagnostic success

诊断成功要求明确 v9.2.31 的失败来源：

```text
D1-static_feature_insufficient:
  静态 pre-event features 不够，但 virtual probe 能预测 event value。

D2-probe_value_predictive:
  probe score 能预测 Real-vs-control gap。

D3-probe_too_expensive:
  probe 有预测力，但系统 overhead 不能进入 official route。

D4-primitive_unobservable:
  probe / features 均无法预测 grounded value，说明 current primitive effect 不可观测。

D5-value_is_family_predictable_only:
  single-event value 不可预测，但 event-family value 可预测。

D6-control_dominance:
  Real 有正效应，但 AdamWParallel / best LR 在大多数 event 上仍更强。
```

### 1.2 Probe-to-commit success

Probe-to-commit controller 必须在 update commit 前预测：

$$
V_{\text{ctrl}}(e)
=
Gain_{\text{Real}}(e)
-
\max(Gain_{\text{AdamWParallel}}(e),Gain_{\text{bestLR}}(e)).
$$

成功标准：

$$
Corr(S_{\text{probe}},V_{\text{grounded}})\geq0.35,
$$

或：

$$
AUC(Y_{\text{beat}})\geq0.70.
$$

Accepted events 必须满足：

$$
Precision_{\text{accepted beats controls}}\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05.
$$

### 1.3 Official paired replay success

只有 probe-to-commit 通过后才能进入 official paired replay。标准是：

$$
BeatRate_{\text{macro,Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro,Real vs bestLR}}\geq0.60.
$$

并且每个 dataset slice 都 task-safe：

$$
Acc_{\text{Real,dataset}}\geq Acc_{\text{AdamW,dataset}}-0.005.
$$

注意：dataset 只作为 evaluation slice，不允许作为 controller 条件。

### 1.4 Observable primitive success

如果 current primitives 下 probe-to-commit 失败，则进入 observable primitive repair。  
新的 primitive 必须同时满足：

```text
contract pass
grad pass
P4 pass
P5 near-pass
functional actuatability pass
pre-commit value observability pass
paired replay pass
```

其中 observability pass 是：

$$
Corr(S_{\text{probe}},V_{\text{grounded}})\geq0.35.
$$

---

## 2. 核心假设

### H1：静态 pre-event features 不够，但 virtual probe 可以预测 event value

v9.2.31 的 P2 失败说明 `effective_derivative`、`pre_adamwparallel_logit_delta_norm` 等静态 features 不够。  
但这不代表 event value 不可预测。可能必须在 commit 前做极小 virtual probe：

```text
1. copy current logits / small state；
2. apply candidate update virtually；
3. evaluate CE-tail / margin-tail / curvature proxy on micro-holdout；
4. compare Real vs AdamWParallel / bestLR；
5. only then commit or abstain。
```

H1 成立标准：

virtual probe score 满足：

$$
Corr(S_{\text{probe}},V_{\text{grounded}})\geq0.35,
$$

并且 accepted precision / coverage 过 gate。

### H2：probe 必须是训练流内部的 population-risk proxy，不能用 validation/test

Probe 不能变成 hidden validation tuning。  
允许使用当前 training batch 内的 split / leave-one-out / micro-holdout：

```text
batch A:
  compute update direction

batch B:
  estimate population-risk proxy / control gap

or leave-one-out within batch:
  each point is treated as one-point test against remaining batch
```

不允许：

```text
test set metric
validation-set hyperparameter tuning
dataset-name route
class-specific hand rule
```

H2 成立标准：

所有 probe rows 记录：

```text
probe_uses_train_stream_only = 1
probe_uses_validation_metric = 0
probe_uses_test_metric = 0
dataset_name_used = 0
```

### H3：如果 virtual probe 有预测力但太慢，需要 distill 成 observable primitive / cheap sufficient statistics

Probe 可能能预测，但 overhead 太高。  
如果：

$$
Corr(S_{\text{probe}},V_{\text{grounded}})\geq0.35
$$

但：

$$
StepRatio_{\text{probe route}}>1.50,
$$

则不能进入 official route。  
此时要提取 probe 中真正有用的 sufficient statistics，例如：

```text
tail movement ratio
control gap lower bound
branch ratio band
orthogonal tail component
microbatch uncertainty
```

并设计 cheaper controller。

### H4：如果 probe 也不能预测，则 current primitive 缺少 causal observability

如果 virtual probe、control-contrastive probe、leave-one-out probe 都无法预测 grounded value，则说明 current primitive 的 functional effect 太不稳定或太噪。  
这时不能继续 controller patch，应进入 observable primitive redesign。

H4 成立标准：

所有 probe：

$$
Corr(S_{\text{probe}},V_{\text{grounded}})<0.25,
$$

且：

$$
AUC(Y_{\text{beat}})<0.60.
$$

Route 应写：

```text
R9-PrimitiveLacksCausalObservability
```

### H5：observable primitive 应该让 functional value 有可计算 sufficient statistic

当前 N2a/N2c/N3c 可能能产生 movement，但 value 不可预测。  
Observable primitive 的设计目标不是更大 movement，而是让 movement 的影响可被 pre-commit statistics 估计。

候选形式：

```text
ObservableTailLinearChannel
ObservablePiecewiseTailChannel
ObservableSharedRBFLocalChannel
ObservableOrthogonalTailChannel
ObservableControlGapChannel
```

H5 成立标准：

新 primitive 的 pre-commit value predictor 过 gate，同时 P4/P5 不破坏。

### H6：不能因为不同数据集表现不同而写 dataset-specific controller

可以诊断：

```text
MNIST 暴露 easy-mode abstention；
Fashion 暴露 delayed / control-dominated tail signal；
KMNIST 暴露 hard-mode / effect magnitude fragility。
```

但 official controller 只能使用 signal features：

```text
tail severity
movement ratio
control gap
uncertainty
branch ratio
effective derivative
horizon agreement
```

H6 成立标准：

Leave-dataset-out pass：

$$
Acc_{\text{heldout,Real}}\geq Acc_{\text{heldout,AdamW}}-0.005,
$$

且至少两个 split 满足：

$$
BeatRate_{\text{Real vs AdamWParallel}}\geq0.50.
$$

---

## 3. Candidate 设计

### 3.1 Baselines

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

### 3.2 Probe candidates

#### CP0：NoProbe reference

```text
使用 v9.2.31 的 static pre-event features。
```

#### CP1：Linearized logit probe

用一阶近似预测 update 后 logits：

$$
z'_{\text{probe}}
=
z
+
J_\theta z\cdot\Delta\theta_{\text{functional}}.
$$

记录 tail CE / margin / control gap。  
该 probe 不真正 commit 参数。

#### CP2：Virtual micro-holdout probe

从当前 train batch 切分：

```text
B_update:
  compute candidate update direction

B_probe:
  evaluate virtual updated logits
```

使用：

$$
S_{\text{probe}}
=
Gain_{\text{Real}}^{B_{\text{probe}}}
-
\max(Gain_{\text{AdamWParallel}}^{B_{\text{probe}}},Gain_{\text{bestLR}}^{B_{\text{probe}}}).
$$

#### CP3：Leave-one-out population-risk probe

对 batch 内每个样本 $i$，用其余样本构造 update，用 $i$ 做 one-point test proxy：

$$
S_{\text{LOO}}
=
\frac{1}{b}
\sum_i
\left[
Gain_{\text{Real},i}
-
\max(Gain_{\text{AdamWParallel},i},Gain_{\text{bestLR},i})
\right].
$$

这对应 “single-run population-risk proxy” 的思想，但不引入 validation/test。

#### CP4：Control-contrastive virtual replay

在同一 microbatch 上同时虚拟评估：

```text
RealFunctional
AdamWParallel
bestLR
NoOp
RandomMatchedNorm
```

只接受：

$$
S_{\text{Real}}>S_{\text{AdamWParallel}}
$$

且：

$$
S_{\text{Real}}>S_{\text{bestLR}}.
$$

#### CP5：Horizon-consistency probe

同一 event 在多个 small horizons 上预测 control gap：

```text
h = 1,5,20
```

接受条件：

$$
\operatorname{sign}(S_h)
\text{ consistent for at least two horizons.}
$$

#### CP6：Uncertainty-lower-bound probe

用 bootstrap / microbatch splits 估计 lower confidence bound：

$$
LCB(S)=\mu_S-\kappa\sigma_S.
$$

只接受：

$$
LCB(S)>0.
$$

### 3.3 Observable primitive candidates

只有当 CP route 失败或太慢时进入。

```text
OP0-N2a-current:
  reference.

OP1-ObservableTailLinearChannel:
  output-edge tail channel with analytic logit movement.

OP2-ObservablePiecewiseTailChannel:
  piecewise local edge channel with bounded derivative.

OP3-ObservableSharedRBFLocalChannel:
  shared-center RBF local channel with normalized basis entropy.

OP4-ObservableOrthogonalTailChannel:
  projected non-AdamW tail movement channel.

OP5-ObservableControlGapChannel:
  channel designed so control gap has closed-form lower bound.

OP6-LightHybridObservable:
  Rational + local channel, but event-time activation is probe-visible.
```

All official primitive candidates must satisfy:

```text
edge_owned_param_fraction = 1
external_residual_used = 0
ordinary_mlp_path_used = 0
manual_forward = 1
manual_backward = 1
manual_update = 1
uses_loss_backward = 0
```

### 3.4 Controls

Every replay / short / full run must include:

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
ProbeScoreShuffled
ProbeTargetShuffled
VirtualOutcomeShuffled
```

New controls:

```text
ProbeScoreShuffled:
  keep events and updates, shuffle probe scores.

ProbeTargetShuffled:
  keep probe scores, shuffle target strata.

VirtualOutcomeShuffled:
  keep event features, shuffle virtual outcome estimates.

Purpose:
  ensure probe-to-commit is using causal information rather than row distribution artifacts.
```

---

## 4. 实验阶段

## P0：v9.2.31 boundary reproduction

### 目标

复现 latest boundary，确认当前不是 measurement noise。

### 必须记录

```text
route
source_route_v9230
value_reliability
event_value_grounding_pass
pre_event_observability_pass
best_feature
best_movement_feature
best_movement_abs_corr
failure_reclassification_pass
good_event_precision
abstain_precision
event_value_predictor_pass
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R9-PrimitiveEffectUnpredictable
event-value grounding pass = 1
pre-event observability pass = 0
predictor pass = 0
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_gate_ladder.svg
p0_value_grounding_vs_observability.svg
```

---

## P1：Static feature insufficiency autopsy

### 目标

解释为什么 grounded value 可靠，但 static pre-event features 预测失败。

### 必须记录

```text
event_id
signal_stratum
horizon
primitive
V_grounded
feature_vector
feature_missing_rate
feature_variance
feature_rank
microbatch_value_variance
within_family_value_variance
between_family_value_variance
best_static_feature_corr
best_movement_feature_corr
```

### 判断标准

Static feature insufficiency confirmed if：

```text
value reliability >= 0.30
but all static feature |corr| < 0.25
and feature rank / variance is sufficient
```

If feature variance/rank is degenerate, mark：

```text
F_static_feature_degenerate
```

If value variance dominates:

```text
F_single_event_value_too_noisy
```

### 可视化

```text
p1_static_feature_corr_heatmap.svg
p1_value_variance_decomposition.svg
p1_feature_rank_distribution.svg
p1_single_vs_family_value_variance.svg
```

---

## P2：Probe-to-value audit

### 目标

测试 CP1-CP6 probe 是否能预测 grounded event value。  
P2 不是 full training，只是 pre-commit probe audit。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 1,5,20,80,240
primitives = N2a,N2c,N3c
events = pre-registered signal strata
probes = CP0-CP6
branches = RealFunctional, AdamWParallel, bestLR, NoOp, Random
```

### 必须记录

```text
probe
event_id
dataset
seed
horizon
primitive
signal_stratum
probe_score_real
probe_score_adamwparallel
probe_score_bestlr
probe_control_gap
probe_uncertainty
probe_lcb
grounded_value
actual_real_minus_adamwparallel
actual_real_minus_bestlr
accepted
bad_event
probe_overhead_ratio
step_ratio_with_probe
memory_ratio_with_probe
dataset_name_used
validation_used
test_metric_used
```

### 判断标准

Probe predictability pass：

$$
Corr(S_{\text{probe}},V_{\text{grounded}})\geq0.35
$$

or:

$$
AUC(Y_{\text{beat}})\geq0.70.
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

Probe legality pass：

```text
dataset_name_used = 0
validation_used = 0
test_metric_used = 0
```

System diagnostic pass：

$$
StepRatio_{\text{with probe}}\leq1.50.
$$

If predictability passes but system fails, route is:

```text
R3-ProbePredictiveButTooExpensive
```

### 可视化

```text
p2_probe_score_vs_grounded_value.svg
p2_probe_precision_coverage.svg
p2_probe_overhead_pareto.svg
p2_probe_type_comparison.svg
```

---

## P3：Probe-to-commit controller calibration

### 目标

从 P2 probes 中选择 official controller，并校准 accept / abstain threshold。

### Controller candidates

```text
PC0-NoProbeStatic:
  v9.2.31 reference.

PC1-LinearizedLogitProbeController:
  CP1 based.

PC2-VirtualMicroHoldoutController:
  CP2 based.

PC3-LOOPopRiskController:
  CP3 based.

PC4-ControlContrastiveProbeController:
  CP4 based.

PC5-HorizonConsistencyController:
  CP5 based.

PC6-UncertaintyLCBController:
  CP6 based.

PC7-HybridProbeController:
  CP4 + CP6 + event-family value.
```

### 必须记录

```text
controller
threshold
coverage
precision
recall
bad_event_rate
accepted_signal_strata_count
accepted_dataset_slice_count
probe_overhead_ratio
step_ratio_with_probe
memory_ratio_with_probe
feature_or_probe_importance
```

### 判断标准

Controller pass：

$$
Precision\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05,
$$

$$
AcceptedSignalStrataCount\geq2.
$$

System route pass：

$$
StepRatio_{\text{with probe}}\leq1.50,
$$

$$
MemoryRatio_{\text{with probe}}\leq1.05.
$$

### 可视化

```text
p3_controller_threshold_curve.svg
p3_precision_coverage_bad_event.svg
p3_accepted_strata_distribution.svg
p3_probe_system_cost.svg
```

---

## P4：Leave-dataset-out / leave-stratum-out validation

### 目标

防止 probe-to-commit controller 隐式 dataset tuning。

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
controller
probe_type
threshold
precision
coverage
bad_event_rate
task_safe
CEp99_delta
margin_delta
ECE_delta
curvature_delta
beats_adamwparallel
beats_bestlr
shuffle_control_pass
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
p4_leave_dataset_out_matrix.svg
p4_leave_stratum_out_matrix.svg
p4_generalization_pareto.svg
p4_hidden_dataset_tuning_audit.svg
```

---

## P5：Observable primitive repair gate

### 目标

如果 P2/P3/P4 失败或 probe too expensive，则进入 observable primitive repair。  
若 P2/P3/P4 pass，可以跳过 P5。

### 设置

```text
primitives = OP0-OP6
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
horizons = 20,80,240
events = signal strata
```

### 必须记录

```text
primitive
contract_pass
grad_pass
P4_forward_q90
P4_backward_q90
P4_step_q90
P4_memory
P5_nearpass
functional_actuatability_pass
probe_observability_corr
accepted_precision
accepted_coverage
paired_replay_ready
```

### 判断标准

Observable primitive pass：

```text
contract_pass = 1
grad_pass = 1
P4 pass = 1
P5 near-pass = 1
functional_actuatability_pass = 1
probe_observability_corr >= 0.35
accepted_precision >= 0.75
coverage in [0.03,0.15]
```

P4 system gate：

$$
forward_{q90}\leq1.25,
$$

$$
backward_{q90}\leq1.50,
$$

$$
step_{q90}\leq1.50,
$$

$$
memory\leq1.05.
$$

### 可视化

```text
p5_observable_primitive_pareto.svg
p5_probe_observability_by_primitive.svg
p5_system_trainability_functional_matrix.svg
```

---

## P6：Official probe-gated paired replay

### 目标

只允许 P3/P4 或 P5 survivor 进入 official paired replay。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
controller = best probe-to-commit survivor
primitive = selected by signal/probe features, not dataset
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, InvertedRole, ProbeScoreShuffled, ProbeTargetShuffled, VirtualOutcomeShuffled
```

### 必须记录

```text
controller
primitive
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
ProbeScoreShuffled = fail
ProbeTargetShuffled = fail
VirtualOutcomeShuffled = fail
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
p6_probe_shuffle_control_matrix.svg
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
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, ProbeScoreShuffled
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
contract_audit_v9232.csv
p0_v9231_boundary_reproduction.csv
p1_static_feature_insufficiency_autopsy.csv
p2_probe_to_value_audit.csv
p3_probe_to_commit_controller_calibration.csv
p4_leave_dataset_and_stratum_out_validation.csv
p5_observable_primitive_repair_gate.csv
p6_official_probe_gated_paired_replay.csv
p7_short_run_functional_validation.csv
p8_full_10seed_functional_validation.csv
p9_adamw_only_fullpass_repair.csv
p10_robustness_external_ready.csv
probe_value_trace_v9232.csv
virtual_microholdout_trace_v9232.csv
loo_population_risk_trace_v9232.csv
observable_primitive_trace_v9232.csv
paired_replay_branch_trace_v9232.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9231_boundary_unstable
F3_dataset_tuning_detected
F4_static_feature_degenerate
F5_single_event_value_too_noisy
F6_probe_not_predictive
F7_probe_too_expensive
F8_probe_precision_or_coverage_fail
F9_leave_dataset_out_fail
F10_leave_stratum_out_fail
F11_observable_primitive_contract_fail
F12_observable_primitive_p4_fail
F13_observable_primitive_p5_fail
F14_primitive_lacks_causal_observability
F15_probe_gated_replay_control_equivalent
F16_probe_shuffle_control_pass
F17_functional_lr_equivalent
F18_short_run_task_drop
F19_full_run_no_macro_hard_stratum_gain
F20_adamw_fullpass_fail
F21_strong_baseline_explains_gain
F22_robustness_fail
F23_external_not_ready
F24_fake_or_proxy_violation
F25_artifact_missing
```

---

## 7. Route decision

### Route cases

```text
R1-StaticFeatureInsufficientButProbePredictive:
  static features fail, virtual probe predicts grounded value.

R2-ProbeToCommitControllerPass:
  probe controller passes precision / coverage / bad-event gates.

R3-ProbePredictiveButTooExpensive:
  probe predicts value but exceeds system budget.

R4-LeaveDatasetOutProbePass:
  probe controller generalizes without dataset-specific tuning.

R5-ObservablePrimitivePass:
  new primitive makes functional value observable and P4/P5-qualified.

R6-ProbeGatedPairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R7-AbstentionOnlyDiagnostic:
  precision high but coverage too low for training advantage.

R8-ControlDominatedSignal:
  Real has effect but optimizer controls dominate.

R9-PrimitiveLacksCausalObservability:
  probe and features cannot predict grounded value.

R10-StrictPureKANFunctionalShortRunPass:
  short-run task-safe mechanism gain.

R11-StrictPureKANFunctionalFullPass:
  full 10-seed macro / hard-stratum / geometry gain.

R12-AdamWFullPassNoFunctional:
  base reaches full-pass but functional remains unproven.

R13-ReturnToInterfacePrimitiveDesign:
  probe-to-commit fails; no dataset-specific patch allowed.

R14-ExternalReady:
  strict PureKAN functional route passes task/geometry/system/control/robustness/strong-baseline gates.
```

### route_decision.json 必须记录

```text
route
v9231_boundary_pass
dataset_tuning_detected
static_feature_insufficiency_confirmed
best_probe
probe_predictive_pass
probe_system_pass
probe_controller_pass
leave_dataset_out_pass
leave_stratum_out_pass
observable_primitive_pass
best_observable_primitive
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
success_v9232_strict_purekan_functional
success_v9232_full_functional
success_v9232_external_ready
```

---

## 8. 第一轮执行顺序

```text
Step 1:
  P0 复现 v9.2.31 boundary。

Step 2:
  P1 做 static feature insufficiency autopsy。
  确认失败是 feature 不足、single-event 噪声，还是 feature degeneracy。

Step 3:
  P2 做 probe-to-value audit。
  先测试 CP1-CP6 是否能预测 grounded value，不进入 full training。

Step 4:
  P3 校准 probe-to-commit controller。
  只允许 train-stream internal probe，不允许 validation/test 或 dataset branch。

Step 5:
  P4 做 leave-dataset-out / leave-stratum-out。
  防止 probe controller 隐式 dataset tuning。

Step 6:
  如果 P2/P3/P4 失败或 probe 太贵，进入 P5 observable primitive repair。
  不允许继续 Fashion/KMNIST-specific patch。

Step 7:
  P6 official probe-gated paired replay。
  加入 ProbeScoreShuffled / VirtualOutcomeShuffled 等强 controls。

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
v9.2.31 boundary reproduced
no dataset-specific route used
static feature insufficiency attributed
probe-to-value audit completed
either probe predictive or primitive observability failure established
no fake/proxy/offload/loss/teacher violation
```

### Local functional success

```text
Minimum diagnostic success
+
probe-to-commit controller passes precision / coverage / bad-event gates
+
leave-dataset-out pass
+
official paired replay beats AdamWParallel / bestLR
```

### Observable primitive success

```text
Current probes fail or too expensive
+
new observable primitive passes contract/P4/P5/actuatability/observability
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
1. v9.2.31 boundary cannot be reproduced；
2. all probes fail to predict grounded value；
3. probe is predictive but too expensive and no cheap sufficient statistic can be extracted；
4. failures cannot be classified without dataset name；
5. leave-dataset-out fails；
6. all observable primitives fail P4/P5/observability；
7. probe-gated paired replay remains control-equivalent；
8. shuffled probe controls pass, indicating overfit；
9. full run gives no macro / hard-stratum / geometry gain；
10. functional breaks system gate；
11. gains are explained by QuadraticFeatureMLP；
12. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 10. 最终解释规则

### Case A：probe predicts grounded value

可以声明：

```text
v9.2.31 failed because static pre-event features were insufficient; probe-to-commit reveals observable functional value.
```

但还不能声明 strict functional success unless paired replay / short/full pass。

### Case B：probe predicts but too expensive

必须声明：

```text
Functional causal value is observable but current probe is too costly for official route.
```

下一步提取 cheap sufficient statistics 或 redesign primitive。

### Case C：probe fails

必须声明：

```text
Current primitive/interface does not provide predictable causal actuatability under stronger observation.
```

下一步回到 observable primitive design。

### Case D：observable primitive passes

可以声明：

```text
The functional interface now exposes measurable causal value before commit.
```

但 full success 仍需 P6-P8。

### Case E：all fail

必须声明：

```text
Current strict PureKAN functional primitive cannot support dataset-agnostic causal functional update; return to primitive/interface redesign, not dataset tuning.
```

---

## 11. 最终建议

v9.2.32 的一句话策略是：

$$
\boxed{
\text{不要再用静态 features 硬预测 event value；先用 probe-to-commit 判断 functional update 是否有可观测 causal value。}
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
1. 在 commit 前，能否通过 virtual probe 判断 RealFunctional 是否会超过 AdamWParallel / best LR？
2. 如果能，probe 是否足够便宜，能否进入 official route？
3. 如果 probe 太贵，能否提取 cheap sufficient statistics？
4. 如果 probe 也不能预测，是否说明 current functional primitive 缺少 causal observability？
5. 新 primitive 能否让 functional value 在不使用 dataset name 的情况下可观测、可预测、可验证？
```
