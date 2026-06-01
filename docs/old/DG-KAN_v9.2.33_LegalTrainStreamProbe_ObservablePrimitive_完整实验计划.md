# DG-KAN v9.2.33 Legal Train-Stream Probe-to-Commit 与 Observable Primitive 实现闭环完整实验计划

> 本计划基于 v9.2.32 `Probe-to-Commit 与 Observable Primitive` 的真实复盘结果制定。  
> v9.2.32 的 terminal route 是：
>
> ```text
> route = R13-ReturnToInterfacePrimitiveDesign
> base_candidate = LQ-t2-h256
> success_v9232_strict_purekan_functional = False
> success_v9232_full_functional = False
> success_v9232_external_ready = False
> ```
>
> v9.2.32 的核心结果不是 “functional update 已失败”，也不是 “继续针对 Fashion / KMNIST / MNIST 做调参”。更准确地说：
>
> $$
> \boxed{
> \text{source-logged probe 显示有可分类信号，但它不是 legal train-stream probe；observable primitives 也尚未实现。}
> }
> $$
>
> 因此 v9.2.33 的核心任务不是继续换 target、调 threshold、按数据集写 route，而是把 v9.2.32 暴露出的两个硬缺口闭合：
>
> $$
> \boxed{
> \text{把 source-logged probe 变成真正 legal 的 train-stream probe-to-commit controller。}
> }
> $$
>
> 以及：
>
> $$
> \boxed{
> \text{把 OP1-OP6 observable primitives 从 not\_implemented 变成可审计候选。}
> }
> $$
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

## 0. 对 v9.2.32 的独立判断

### 0.1 没有达到目标，但不是无信号

v9.2.32 没有达到 strict PureKAN functional success。下游 P6-P10 都没有打开，原因是没有 legal probe controller，也没有 observable primitive survivor。当前 route 停在：

```text
R13-ReturnToInterfacePrimitiveDesign
```

这意味着当前不能声明：

```text
strict PureKAN functional update 成功；
probe-to-commit controller 成功；
observable primitive 成功；
short-run / full-run 可以打开；
external-ready 可以打开。
```

但是，v9.2.32 并不是“完全没有 functional signal”。关键证据是 P2 source-logged probe audit：

```text
best probe = CP5-HorizonConsistencyProbe
best probe corr = 0.095796
best probe AUC = 0.837635
precision = 0.760000
coverage = 0.051440
predictive pass = 1
legality pass = 0
```

这说明 **source-logged CP5 可以把 good/bad event 分开**，AUC 和 precision 都达到了有意义的水平。但它不是本轮新测的 legal train-stream virtual probe，因此不能进入 official route。

### 0.2 当前不能把 CP5 写成成功

CP5 的核心问题不是 AUC 不够，而是 legality 不过。v9.2.32 使用的是 v9.2.30 / v9.2.31 真实落盘 source rows 做 probe audit；这可以作为诊断，但不能直接作为在线 controller。官方 controller 必须满足：

```text
probe_uses_train_stream_only = 1
probe_uses_validation_metric = 0
probe_uses_test_metric = 0
dataset_name_used = 0
posthoc_outcome_used_at_commit = 0
```

v9.2.32 没有实现新的 train-stream virtual microholdout probe，所以：

$$
\boxed{
\text{CP5 是 strong diagnostic clue，不是 official functional controller。}
}
$$

### 0.3 OP gate 也没有真正失败，只是没有实现

P5 observable primitive gate 的结果是：

```text
observable_primitive_pass = 0
best_observable_primitive = OP0-current-reference
best_observable_primitive_corr = 0.133410
not_implemented OP rows = 6
```

这不能解释为 “OP1-OP6 都失败”。准确说法是：

```text
OP0/current reference observability 不够；
OP1-OP6 未实现；
observable primitive factory 尚未真正开始。
```

因此 v9.2.33 不应该继续只做 controller patch，也不应该立刻放弃 primitive。下一步必须真正实现 legal probe 与 observable primitive。

### 0.4 当前问题的本质

从 v9.2.29 到 v9.2.32，问题已经逐步收紧：

```text
v9.2.29:
  signal strata 能解释 failure，但 event-value predictor corr 不够。

v9.2.30:
  pre-event movement first-wave 接近 gate，但 predictor 未闭合。

v9.2.31:
  event-value label grounded，value reliability 过关；
  但 static pre-event features 不预测 grounded value。

v9.2.32:
  source-logged CP5 probe 有分类信号；
  但不是 legal train-stream probe；
  OP1-OP6 没实现。
```

所以当前真正问题不是 dataset，也不是 target，也不是 loss，而是：

$$
\boxed{
\text{functional causal value 是否能在 commit 前、用训练流内部信息、以可接受系统成本被观测。}
}
$$

这就是 v9.2.33 的核心。

---

## 1. v9.2.33 总体目标

v9.2.33 的总体目标是：

$$
\boxed{
\text{把 v9.2.32 的 source-logged diagnostic signal 转化为 legal train-stream probe-to-commit route；若不能，则实现 observable primitive repair。}
}
$$

这个目标分为五层。

### 1.1 Legal probe success

Legal probe 必须在 commit update 前计算，不能使用 validation / test，也不能使用 dataset name。  
它要预测：

$$
V_{\text{ctrl}}(e)
=
Gain_{\text{Real}}(e)
-
\max(
Gain_{\text{AdamWParallel}}(e),
Gain_{\text{bestLR}}(e)
).
$$

成功标准：

$$
AUC(Y_{\text{beat}})\geq0.70
$$

或：

$$
Corr(S_{\text{probe}},V_{\text{grounded}})\geq0.35.
$$

其中：

$$
Y_{\text{beat}}
=
\mathbb{1}
[
V_{\text{ctrl}}>0
].
$$

### 1.2 Probe-to-commit controller success

Probe 不只是要预测，还要能形成 accept / abstain policy：

$$
Precision_{\text{accepted beats controls}}\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05.
$$

其中 coverage 太低只能算 diagnostic，不算 route success。

### 1.3 System legality success

Probe 不能通过超大开销换来预测力。Official route 要求：

$$
StepRatio_{\text{with probe}}\leq1.50,
$$

$$
MemoryRatio_{\text{with probe}}\leq1.05.
$$

如果 probe predictive pass 但 system fail，则 route 是：

```text
R3-ProbePredictiveButTooExpensive
```

此时下一步应提取 cheap sufficient statistics，而不是进入 short-run。

### 1.4 Observable primitive success

如果 legal probe 不能预测或太贵，则进入 observable primitive repair。新 primitive 必须满足：

```text
contract pass
grad pass
P4 pass
P5 near-pass
functional actuatability pass
probe observability pass
paired replay pass
```

其中 probe observability pass 是：

$$
Corr(S_{\text{probe}},V_{\text{grounded}})\geq0.35
$$

或：

$$
AUC(Y_{\text{beat}})\geq0.70.
$$

### 1.5 Local functional causality success

只有 legal probe 或 observable primitive 通过后，才打开 official paired replay。成功标准：

$$
BeatRate_{\text{macro, Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro, Real vs bestLR}}\geq0.60,
$$

并且：

$$
Acc_{\text{Real,dataset}}\geq Acc_{\text{AdamW,dataset}}-0.005.
$$

这里 dataset 只作为评估切片，不作为 controller 条件。

---

## 2. 核心假设

### H1：v9.2.32 的 source-logged CP5 信号可以被 legal train-stream probe 复现

v9.2.32 的 CP5-HorizonConsistencyProbe 虽然 legality 不过，但它给出：

$$
AUC=0.837635,
$$

$$
Precision=0.760000,
$$

$$
Coverage=0.051440.
$$

H1 认为：如果把 CP5 的 horizon consistency 逻辑真正实现为 train-stream virtual probe，而不是 source-logged posthoc score，那么仍能保留大部分分类能力。

H1 成立标准：

legal CP5 或其变体满足：

$$
AUC_{\text{legal}}\geq0.70,
$$

$$
Precision_{\text{legal}}\geq0.75,
$$

$$
Coverage_{\text{legal}}\in[0.03,0.15].
$$

如果 source-logged CP5 过而 legal CP5 不过，则说明 v9.2.32 的信号主要来自 source-log / measurement protocol，而不是可部署 controller。

### H2：静态 features 不够，但 virtual microholdout probe 可能够

v9.2.31 已经证明 grounded value 可靠，但 static features 预测不过；v9.2.32 source-logged probe 有分类信号。H2 认为，需要 commit 前 virtual evaluation，而不是仅用静态 summary features。

H2 成立标准：

virtual microholdout probe 满足：

$$
Corr(S_{\text{probe}},V_{\text{grounded}})\geq0.35
$$

或：

$$
AUC(Y_{\text{beat}})\geq0.70.
$$

### H3：如果 legal probe 有预测力但太贵，需要抽取 cheap sufficient statistics

Probe 可能能预测，但开销过大。如果：

$$
AUC\geq0.70
$$

且：

$$
StepRatio_{\text{with probe}}>1.50,
$$

则不能 official pass。此时要提取 cheaper statistic，例如：

```text
horizon agreement score
tail virtual gain lower bound
control gap lower bound
orthogonal tail movement ratio
microbatch uncertainty
primitive agreement
```

H3 成立标准：

cheap statistic route 满足 legal probe 的 precision / coverage / bad-event gate，同时：

$$
StepRatio\leq1.50.
$$

### H4：如果 legal probes 全部失败，则当前 primitive 缺少 causal observability

如果 CP1-CP6 legal probes 都不满足：

$$
AUC\geq0.60
$$

且：

$$
Corr<0.25,
$$

则 current N2a/N2c/N3c family 无法在 commit 前提供可观测 functional value。此时必须进入 OP primitive implementation，而不是继续 dataset patch。

### H5：Observable primitive 的目标不是更大 movement，而是更可观测的 movement

过去多轮已经说明：movement 大不等于 causal gain。Observable primitive 需要让 functional movement 的 metric effect 可由 pre-commit statistics 估计。  
因此 OP family 的核心不是最大化 $r_z$，而是最大化：

$$
Corr(S_{\text{probe}},V_{\text{grounded}}).
$$

H5 成立标准：

至少一个 OP primitive 同时满足：

```text
P4 pass
P5 near-pass
actuatability pass
observability pass
paired replay pass
```

### H6：数据集只能诊断，不能调参

允许记录：

```text
MNIST = easy-mode / abstention safety slice
Fashion-MNIST = delayed/control-dominated tail slice
KMNIST = hard-mode/effect-magnitude slice
```

但 official controller 不允许：

```text
if dataset == Fashion: ...
if dataset == KMNIST: ...
if dataset == MNIST: ...
```

H6 成立标准：

leave-dataset-out 必须通过：

$$
Acc_{\text{heldout,Real}}\geq Acc_{\text{heldout,AdamW}}-0.005.
$$

至少两个 held-out split 满足：

$$
BeatRate_{\text{heldout,Real vs AdamWParallel}}\geq0.50.
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
  mechanism source only，不作为 official strict candidate。
```

### 3.2 Legal train-stream probe candidates

#### LP0：Static no-probe reference

复现 v9.2.31 / v9.2.32 static feature path。用于证明 legal probe 是否真正优于 static features。

#### LP1：Legal HorizonConsistencyProbe

合法复现 CP5。  
对同一 training-stream event，在 commit 前计算多个短 horizon 的 virtual score：

```text
h = 1,5,20
```

定义：

$$
S_h
=
Gain_{\text{Real}}^{probe,h}
-
\max(
Gain_{\text{AdamWParallel}}^{probe,h},
Gain_{\text{bestLR}}^{probe,h}
).
$$

接受条件：

$$
\sum_h \mathbb{1}[S_h>0]\geq2
$$

且：

$$
\min_h S_h > -\tau.
$$

#### LP2：Legal Virtual Microholdout Probe

从当前训练 batch 切分：

```text
B_update:
  compute candidate update direction

B_probe:
  estimate Real / controls after virtual update
```

Probe score：

$$
S_{\text{micro}}
=
Gain_{\text{Real}}^{B_{\text{probe}}}
-
\max(
Gain_{\text{AdamWParallel}}^{B_{\text{probe}}},
Gain_{\text{bestLR}}^{B_{\text{probe}}}
).
$$

#### LP3：Legal Leave-One-Out Population Risk Probe

batch 内 leave-one-out：

$$
S_{\text{LOO}}
=
\frac{1}{b}
\sum_i
\left[
Gain_{\text{Real},i}
-
\max(
Gain_{\text{AdamWParallel},i},
Gain_{\text{bestLR},i}
)
\right].
$$

该 probe 只用当前 train batch，不使用 validation/test。

#### LP4：Control-Contrastive Virtual Probe

同一 event 上虚拟比较：

```text
RealFunctional
AdamWParallelTrustRatio-0.003
AdamWParallelTrustRatio-0.01
AdamWParallelTrustRatio-0.03
BestLRScale
NoOp
RandomMatchedNorm
```

接受：

$$
S_{\text{Real}}>
S_{\text{AdamWParallel}}
$$

且：

$$
S_{\text{Real}}>
S_{\text{bestLR}}.
$$

#### LP5：Uncertainty-LCB Probe

用 microbatch splits 估计：

$$
LCB(S)=\mu_S-\kappa\sigma_S.
$$

接受：

$$
LCB(S)>0.
$$

#### LP6：Cheap Sufficient-Statistic Probe

如果 LP1-LP5 predictive but expensive，则从它们提取：

```text
horizon_agreement_score
tail_gain_lcb
control_gap_lcb
orthogonal_tail_component_ratio
microbatch_uncertainty
primitive_agreement_score
```

形成低成本 controller。

### 3.3 Observable primitive candidates

只有 legal probe 失败或太贵时进入。

```text
OP0-current-reference:
  N2a/N2c/N3c current reference。

OP1-ObservableTailLinearChannel:
  output-edge tail channel with analytic logit movement。

OP2-ObservablePiecewiseTailChannel:
  bounded piecewise local edge channel。

OP3-ObservableSharedRBFLocalChannel:
  normalized shared-center RBF local channel。

OP4-ObservableOrthogonalTailChannel:
  non-AdamW orthogonal tail component channel。

OP5-ObservableControlGapChannel:
  channel designed to expose closed-form lower bound of Real-control gap。

OP6-LightHybridObservable:
  rational + local channel, with event-time activation visible to probe。
```

所有 OP 必须仍然是 edge-owned PureKAN，不允许 external residual 或 ordinary MLP path。

### 3.4 Controls

每个 P2-P8 official run 必须包含：

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
ProbeSplitShuffled
```

新增 `ProbeSplitShuffled` 用于检查 microholdout split 是否被过拟合。

---

## 4. 实验阶段

## P0：v9.2.32 boundary reproduction

### 目标

复现 latest boundary，确认不是 measurement artifact。

### 必须记录

```text
route
source_route_v9231
static_feature_insufficiency_confirmed
best_static_corr
best_movement_corr
best_probe
best_probe_corr
best_probe_auc
best_probe_precision
best_probe_coverage
probe_predictive_pass
probe_legality_pass
best_observable_primitive
best_observable_primitive_corr
observable_primitive_pass
not_implemented_op_count
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R13-ReturnToInterfacePrimitiveDesign
best probe = CP5-HorizonConsistencyProbe
probe predictive pass = 1
probe legality pass = 0
observable primitive pass = 0
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_source_logged_probe_vs_legal_gap.svg
p0_observable_primitive_status.svg
```

---

## P1：Source-logged CP5 legality autopsy

### 目标

解释 CP5 为什么有 AUC/precision，但 legality fail。  
P1 不实现新 probe，只做信息来源审计。

### 必须记录

```text
probe_row_id
probe_type
score_source
uses_source_logged_grounded_value
uses_posthoc_replay_outcome
uses_validation_metric
uses_test_metric
uses_dataset_name
available_before_commit
available_after_commit_only
can_be_recomputed_in_train_stream
required_state
required_extra_forward
required_extra_backward
estimated_cost
```

### 判断标准

P1 必须把 CP5 的每个 feature 分成：

```text
LegalBeforeCommit
LegalButNeedsProbe
PosthocOnly
IllegalValidationOrTest
DatasetSpecific
```

如果 CP5 的主要 AUC 来自 `PosthocOnly` features，则下一步必须重新设计 probe；如果主要来自 `LegalButNeedsProbe` features，则进入 P2 legal implementation。

### 可视化

```text
p1_cp5_information_source_map.svg
p1_legal_vs_posthoc_feature_importance.svg
p1_recomputability_cost_estimate.svg
```

---

## P2：Legal train-stream probe implementation

### 目标

真正实现 LP1-LP5，而不是复用 source-logged rows。  
P2 是 v9.2.33 的核心阶段。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 1,5,20,80,240
primitives = N2a,N2c,N3c
events = pre-registered signal strata
probes = LP0-LP5
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
probe_score_noop
probe_score_random
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
probe_uses_train_stream_only
probe_uses_validation_metric
probe_uses_test_metric
dataset_name_used
posthoc_outcome_used_at_commit
```

### 判断标准

Legal pass：

```text
probe_uses_train_stream_only = 1
probe_uses_validation_metric = 0
probe_uses_test_metric = 0
dataset_name_used = 0
posthoc_outcome_used_at_commit = 0
```

Predictive pass：

$$
AUC(Y_{\text{beat}})\geq0.70
$$

or:

$$
Corr(S_{\text{probe}},V_{\text{grounded}})\geq0.35.
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

System diagnostic：

$$
StepRatio_{\text{with probe}}\leq1.50,
$$

$$
MemoryRatio_{\text{with probe}}\leq1.05.
$$

### 可视化

```text
p2_legal_probe_auc_corr.svg
p2_probe_precision_coverage.svg
p2_probe_cost_pareto.svg
p2_probe_legality_matrix.svg
```

---

## P3：Probe-to-commit controller calibration

### 目标

把 P2 中通过的 legal probe 转成实际 accept / abstain controller。

### Candidate controllers

```text
PC0-StaticReference:
  v9.2.31 static baseline。

PC1-LegalHorizonConsistencyController:
  LP1 based。

PC2-VirtualMicroHoldoutController:
  LP2 based。

PC3-LOOPopRiskController:
  LP3 based。

PC4-ControlContrastiveController:
  LP4 based。

PC5-UncertaintyLCBController:
  LP5 based。

PC6-HybridProbeController:
  combines LP1 + LP4 + LP5。

PC7-CheapStatisticController:
  distilled from predictive legal probe but using cheap sufficient stats。
```

### 必须记录

```text
controller
threshold
probe_type
coverage
precision
recall
bad_event_rate
accepted_signal_strata_count
accepted_dataset_slice_count
probe_overhead_ratio
step_ratio_with_probe
memory_ratio_with_probe
selected_primitive_policy
dataset_name_used
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

If predictive but expensive：

```text
route = R3-ProbePredictiveButTooExpensive
next = cheap sufficient statistic extraction or observable primitive
```

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

证明 controller 不是隐式 dataset tuning。

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

至少两个 LDO splits 满足：

$$
BeatRate_{\text{heldout,Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{heldout,Real vs bestLR}}\geq0.50.
$$

LSO pass：

至少 $70\%$ held-out strata task safe 且 CE tail 不劣化：

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

## P5：Observable primitive implementation gate

### 目标

如果 legal probe 不过或太贵，真正实现 OP1-OP6，而不是继续把它们记为 not_implemented。

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
basis_formula
edge_owned_param_fraction
external_residual_used
ordinary_mlp_path_used
manual_forward
manual_backward
manual_update
uses_loss_backward
GradRelErrMax
GradCosMin
P4_forward_q90
P4_backward_q90
P4_step_q90
P4_memory
P5_nearpass
functional_actuatability_pass
probe_observability_corr
probe_auc
accepted_precision
accepted_coverage
bad_event_rate
paired_replay_ready
```

### 判断标准

Contract pass：

```text
edge_owned_param_fraction = 1
external_residual_used = 0
ordinary_mlp_path_used = 0
manual_forward = 1
manual_backward = 1
manual_update = 1
uses_loss_backward = 0
```

Grad pass：

$$
GradRelErrMax\leq10^{-4},
$$

$$
GradCosMin\geq0.999.
$$

P4 pass：

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

P5 near-pass：

$$
near\_pass\_rate\geq0.80,
$$

$$
\Delta Acc_{\text{macro}}\geq-0.01.
$$

Observability pass：

$$
Corr(S_{\text{probe}},V_{\text{grounded}})\geq0.35
$$

or:

$$
AUC(Y_{\text{beat}})\geq0.70.
$$

### 可视化

```text
p5_observable_primitive_pareto.svg
p5_probe_observability_by_primitive.svg
p5_system_trainability_functional_matrix.svg
p5_contract_grad_p4_p5_matrix.svg
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
controller = best legal probe / observable primitive survivor
primitive = selected by signal/probe features, not dataset
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, InvertedRole, ProbeScoreShuffled, ProbeTargetShuffled, VirtualOutcomeShuffled, ProbeSplitShuffled
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
ProbeSplitShuffled = fail
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

## P7：Short-run functional validation

### 目标

验证 local causality 是否能在连续训练中保持。只有 P6 survivor 才能进入。

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

## 5. Required artifacts

```text
run_manifest.json
contract_audit_v9233.csv
p0_v9232_boundary_reproduction.csv
p1_source_logged_cp5_legality_autopsy.csv
p2_legal_train_stream_probe_implementation.csv
p3_probe_to_commit_controller_calibration.csv
p4_leave_dataset_and_stratum_out_validation.csv
p5_observable_primitive_implementation_gate.csv
p6_official_probe_gated_paired_replay.csv
p7_short_run_functional_validation.csv
p8_full_10seed_functional_validation.csv
p9_adamw_only_fullpass_repair.csv
p10_robustness_external_ready.csv
legal_probe_trace_v9233.csv
virtual_microholdout_trace_v9233.csv
horizon_consistency_legal_trace_v9233.csv
observable_primitive_trace_v9233.csv
paired_replay_branch_trace_v9233.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9232_boundary_unstable
F3_dataset_tuning_detected
F4_cp5_signal_posthoc_only
F5_legal_probe_not_implemented
F6_legal_probe_not_predictive
F7_legal_probe_too_expensive
F8_probe_precision_or_coverage_fail
F9_leave_dataset_out_fail
F10_leave_stratum_out_fail
F11_observable_primitive_contract_fail
F12_observable_primitive_grad_fail
F13_observable_primitive_p4_fail
F14_observable_primitive_p5_fail
F15_observable_primitive_observability_fail
F16_probe_gated_replay_control_equivalent
F17_probe_shuffle_control_pass
F18_functional_lr_equivalent
F19_short_run_task_drop
F20_full_run_no_macro_hard_stratum_gain
F21_adamw_fullpass_fail
F22_strong_baseline_explains_gain
F23_robustness_fail
F24_external_not_ready
F25_fake_or_proxy_violation
F26_artifact_missing
```

---

## 6. Route decision

### Route cases

```text
R1-CP5SignalLegalizable:
  source-logged CP5 signal can be recomputed as legal train-stream probe.

R2-LegalProbePredictive:
  legal probe predicts grounded value or beat label.

R3-ProbePredictiveButTooExpensive:
  legal probe has predictive power but exceeds system budget.

R4-ProbeToCommitControllerPass:
  controller passes precision / coverage / bad-event gates.

R5-LeaveDatasetOutProbePass:
  controller generalizes without dataset-specific tuning.

R6-ObservablePrimitivePass:
  OP primitive passes contract / grad / P4 / P5 / observability.

R7-ProbeGatedPairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R8-AbstentionOnlyDiagnostic:
  precision high but coverage too low for training advantage.

R9-CP5PosthocOnly:
  CP5 source signal cannot be computed legally before commit.

R10-PrimitiveLacksCausalObservability:
  legal probes and OP reference cannot predict grounded value.

R11-StrictPureKANFunctionalShortRunPass:
  short-run task-safe mechanism gain.

R12-StrictPureKANFunctionalFullPass:
  full 10-seed macro / hard-stratum / geometry gain.

R13-AdamWFullPassNoFunctional:
  base reaches full-pass but functional remains unproven.

R14-ReturnToInterfacePrimitiveDesign:
  legal probe and observable primitive fail; no dataset-specific patch allowed.

R15-ExternalReady:
  strict PureKAN functional route passes task/geometry/system/control/robustness/strong-baseline gates.
```

### route_decision.json 必须记录

```text
route
v9232_boundary_pass
dataset_tuning_detected
cp5_legality_classification
best_legal_probe
legal_probe_predictive_pass
legal_probe_system_pass
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
success_v9233_strict_purekan_functional
success_v9233_full_functional
success_v9233_external_ready
```

---

## 7. 第一轮执行顺序

```text
Step 1:
  P0 复现 v9.2.32 boundary。

Step 2:
  P1 做 CP5 legality autopsy。
  先分清 CP5 信号来自 legal-before-commit、legal-but-needs-probe，还是 posthoc-only。

Step 3:
  P2 实现 legal train-stream probes。
  这是本轮核心，不允许再只复用 source-logged rows。

Step 4:
  P3 校准 probe-to-commit controller。
  若 predictive but too expensive，转 cheap statistic，不进入 paired replay。

Step 5:
  P4 做 leave-dataset-out / leave-stratum-out。
  防止隐式 dataset tuning。

Step 6:
  若 legal probe 不过或太贵，P5 实现 OP1-OP6 observable primitives。
  不允许再把 OP1-OP6 只写 not_implemented。

Step 7:
  P6 official probe-gated paired replay。
  只有 legal controller 或 OP survivor 才能进入。

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

## 8. 停止条件

### Minimum diagnostic success

```text
v9.2.32 boundary reproduced
no dataset-specific route used
CP5 legality source classified
legal train-stream probe implemented
either legal probe predictive or primitive observability failure established
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
legal probes fail or are too expensive
+
at least one OP primitive passes contract / grad / P4 / P5 / actuatability / observability
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
1. v9.2.32 boundary cannot be reproduced；
2. CP5 signal is posthoc-only and cannot be legalized；
3. legal probes are not predictive；
4. legal probes are predictive but too expensive and no cheap sufficient statistic can be extracted；
5. failures cannot be classified without dataset name；
6. leave-dataset-out fails；
7. all observable primitives fail contract / grad / P4 / P5 / observability；
8. probe-gated paired replay remains control-equivalent；
9. shuffled probe controls pass, indicating overfit；
10. full run gives no macro / hard-stratum / geometry gain；
11. functional breaks system gate；
12. gains are explained by QuadraticFeatureMLP；
13. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 9. 最终解释规则

### Case A：legal CP5 / legal probe passes

可以声明：

```text
v9.2.32 failed because the probe was not yet implemented legally; source-logged signal was real and legalizable.
```

但还不能声明 strict functional success unless P6 / P7 / P8 pass。

### Case B：probe predictive but too expensive

必须声明：

```text
Functional causal value is observable, but current probe-to-commit route is too expensive.
```

下一步必须提取 cheap sufficient statistics 或设计 observable primitive。

### Case C：probe signal is posthoc-only

必须声明：

```text
CP5 source-logged signal cannot be used as online controller.
```

这时不能继续 patch CP5，需要进入 observable primitive。

### Case D：OP primitive passes

可以声明：

```text
Functional interface now exposes measurable causal value before commit.
```

但 full success 仍需 official paired replay / short-run / full-run。

### Case E：all legal probes and OP primitives fail

必须声明：

```text
Current strict PureKAN functional primitive cannot support dataset-agnostic causal functional update.
```

下一步回到 deeper primitive / interface design，不允许 dataset-specific patch。

---

## 10. 最终建议

v9.2.33 的一句话策略是：

$$
\boxed{
\text{先把 source-logged CP5 变成合法 train-stream probe；若做不到，就真正实现 observable primitives，而不是继续数据集特化调参。}
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
1. CP5 的信号是否可以在 commit 前合法计算？
2. legal train-stream probe 能否预测 RealFunctional 是否超过 AdamWParallel / best LR？
3. probe 是否足够便宜，能否进入 official route？
4. 如果 probe 太贵，是否能提取 cheap sufficient statistics？
5. 如果 probe 失败，OP1-OP6 是否能让 functional value 可观测？
6. 所有 controller 能否在不使用 dataset name 的情况下 leave-dataset-out 泛化？
```
