# DG-KAN v9.2.29 Dataset-Agnostic Functional Causality：从数据集诊断到通用信号路由的完整实验计划

> 本计划基于 v9.2.28 `Fashion-First Routed Functional 与 KMNIST Survivor Integration` 的完整复盘制定。  
> v9.2.28 的最新真实 route 是：
>
> ```text
> route = R5-KMNISTSurvivorNotPreserved
> base_candidate = LQ-t2-h256
> success_v9228_strict_purekan_functional = False
> success_v9228_full_functional = False
> success_v9228_external_ready = False
> ```
>
> v9.2.28 的核心结果不是 “functional update 失败”，也不是 “继续针对 Fashion / KMNIST 做调参”。更准确地说：
>
> $$
> \boxed{
> \text{当前 functional signal 具有局部诊断价值，但尚未形成 dataset-agnostic control-resistant causality。}
> }
> $$
>
> 用户特别强调：可以诊断不同数据集的问题，但不能针对数据集调参，因为目标不是在这些数据集上打榜。  
> 因此 v9.2.29 的路线必须收紧：**Fashion / KMNIST / MNIST 只能作为诊断分层，不允许成为 official controller 的硬编码 dataset branch。**
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
> 任务损失保持标准 CE：
>
> $$
> L_{\text{task}}=CE(y,p_\theta(x)).
> $$

---

## 0. 当前结果的独立判断

### 0.1 v9.2.28 没有达到 strict PureKAN functional success

v9.2.28 的 route 是：

```text
R5-KMNISTSurvivorNotPreserved
```

关键结果如下：

```text
best_kmnist_target = K6-KMNIST-MetricCausalTarget
best_kmnist_primitive = N2a-TinyInit-RationalFunc-BranchRatioCap
kmnist_best_actual_effect_rate = 0.766667
kmnist_best_beats_adamwparallel = 0.266667
kmnist_best_beats_best_lr = 0.483333
kmnist_survivor_preserved = 0

fashion_best_candidate = F14-Fashion-F7-ControlBeatAbstain
fashion_best_beats_adamwparallel = 0.283333
fashion_best_beats_best_lr = 0.566667
fashion_signal_confirmed = 0
fashion_failure_mode = FashionEffectControlDominated

best_routed_controller = U0-GlobalBestSingle
best_routed_beats_adamwparallel = 0.500000
best_routed_beats_best_lr = 0.500000
unified_routed_controller_pass = 0
```

因此，v9.2.28 没有证明：

```text
strict PureKAN functional update 成功；
routed functional controller 成功；
short-run functional success；
full-run functional success；
external-ready。
```

### 0.2 v9.2.28 的真实进展

v9.2.28 有三个重要进展。

第一，Fashion missing controllers 不再是 blocker。v9.2.27 中未实现的 Fashion controllers 在本轮进入 measured replay。因此当前不能再说 Fashion 失败只是因为 F9/F10/F11 没实现。

第二，Fashion 的状态从 “partial delayed signal” 变成了更明确的 **effect exists but control dominated**。F14 / F7-family controller 可以产生 effect，但不能稳定击败 AdamWParallel / best LR。这个结果不是“没有信号”，而是“functional event ranking / selection 没有强过 optimizer control”。

第三，KMNIST 的 diagnosis survivor 没能升级为 integration survivor。K6 + N2a 在 stronger replay 下仍有 actual-effect rate，但 control superiority 不够：

$$
BeatRate_{\text{KMNIST,Real vs AdamWParallel}}=0.266667,
$$

$$
BeatRate_{\text{KMNIST,Real vs best LR}}=0.483333.
$$

这说明 KMNIST 的问题已经从 v9.2.26 的 “actual effect = 0” 推进到了更深的 “actual effect 有，但 effect magnitude / control superiority 不够”。

### 0.3 这轮最重要的新边界

v9.2.28 之前，我们以为可以沿着：

```text
Fashion delayed controller
+
KMNIST survivor integration
+
dataset-routed controller
```

继续推进。但 v9.2.28 表明，单纯 routed controller 没有闭合：

```text
U0 GlobalBestSingle = 0.5 / 0.5
U1 DatasetRouted = 0.472 / 0.472
U2 MetricRouted = 0.472 / 0.472
U3 EventRouted = 0.5 / 0.5
U4 AbstainHeavy = 0.389 / 0.389
U6 KMNISTOnly = 0.389 / 0.389
```

没有任何 routed controller 达到 macro gate：

$$
BeatRate_{\text{macro}}\geq0.60.
$$

因此，当前 blocker 不是某一个 dataset 的 target 没写好，而是：

$$
\boxed{
\text{我们还没有找到跨数据集稳定的 functional signal-selection principle。}
}
$$

### 0.4 当前不能做什么

v9.2.29 不能继续做以下事情：

```text
1. 按 dataset name 写 controller；
2. 针对 Fashion 单独调 horizon/cap 直到过；
3. 针对 KMNIST 单独调 primitive/target 直到过；
4. 用 routed policy 把局部结果包装成 global success；
5. 用 short/full run 代替 paired replay gate；
6. 提前打开 PureKANConv / PureKANFormer；
7. 改 loss、teacher、sampler、class weight 或 label smoothing。
```

Fashion / KMNIST / MNIST 可以作为诊断分层，但 official policy 必须由 **dataset-agnostic observables** 决定，例如：

```text
tail severity
margin risk
wrong confidence
actual logit movement
signal-to-control gap
branch ratio
effective derivative
cosine with AdamW
predicted control beat
event uncertainty
```

---

## 1. v9.2.29 总体目标

v9.2.29 的总体目标是：

$$
\boxed{
\text{建立 dataset-agnostic functional signal-selection rule，并验证它能在强 controls 下产生稳定 local causality。}
}
$$

这个目标有三层。

### 1.1 Diagnostic success

诊断成功要求解释 v9.2.28 的两个现象：

```text
Fashion:
  effect exists but control dominated.

KMNIST:
  actual-effect survivor not preserved under stronger replay.
```

诊断不能停留在 dataset name，而要转成机制分类：

```text
M1-event_over_acceptance:
  controller 接受了太多低价值 event。

M2-effect_magnitude_too_small:
  actual movement 有方向，但 magnitude 不足以击败 controls。

M3-control_dominated_signal:
  Real 有 effect，但 AdamWParallel / LR 更强。

M4-misaligned_tail_movement:
  actual movement 没有落到 CE-tail / margin-tail / wrong-confidence samples。

M5-routing_overfit:
  route policy 依赖 dataset label 或 event label，shuffle control 可以复现。

M6-insufficient_abstention:
  应该 abstain 的 event 被更新，导致 macro beat rate 降低。
```

### 1.2 Local causality success

Local causality success 要求一个 **不使用 dataset name** 的 official controller 在 paired replay 中满足：

$$
BeatRate_{\text{macro,Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro,Real vs best LR}}\geq0.60,
$$

并且：

$$
Acc_{\text{Real}}\geq Acc_{\text{AdamW}}-0.005.
$$

同时，route-shuffle / event-shuffle / label-shuffle controls 不能通过：

```text
DatasetRouteShuffled = fail
EventRouteShuffled = fail
SeverityBucketShuffled = fail
InvertedRoleMask = fail
```

### 1.3 Full functional success

Full functional success 只有在 local causality success 后打开。要求：

$$
\Delta Acc_{\text{functional-vs-AdamW}}\geq-0.005,
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

这里的 hard-stratum 不是 dataset name，而是由 pre-registered diagnostics 定义的高风险样本簇，例如：

```text
low margin
high CE
wrong-confidence
high local curvature
low signal-to-control ratio
```

---

## 2. 核心假设

### H1：Fashion 与 KMNIST 不是需要 dataset-specific tuning，而是暴露了不同 signal strata

Fashion 体现的是 delayed / control-dominated tail signal；KMNIST 体现的是 hard-mode / weak-magnitude / fragile control-superiority signal。  
H1 认为这不是 dataset name 的问题，而是样本事件类型不同。

H1 成立标准：

如果把 events 按以下 dataset-agnostic strata 分组：

```text
S1-CEp99Tail
S2-MarginP10Tail
S3-WrongConfidenceTail
S4-CurvatureSpike
S5-HighActualMovementLowMetricGain
S6-ControlDominatedEffect
S7-LowUncertaintyHighSignal
```

则 Fashion / KMNIST 的 failure 应分别落入不同 strata，而不是只能用 dataset name 解释。

量化要求：

$$
I(\text{Dataset};\text{FailureMode})
<
I(\text{SignalStratum};\text{FailureMode}),
$$

其中 $I$ 是 mutual information 或 normalized association score。

### H2：当前失败的主因是 event selection，而不是 primitive 完全错误

Fashion 的 CEp99 / margin 有明显 effect，KMNIST actual-effect rate 仍高，但 control beat 不够。H2 认为主要问题是：

```text
更新了低价值 event；
没有 abstain；
没有估计 Real vs control gap；
没有把 effect magnitude 与 control superiority 分开。
```

H2 成立标准：

如果加入 event-value predictor / abstention 后：

$$
Precision_{\text{accepted event beats controls}}\geq0.70,
$$

且：

$$
Coverage\in[0.02,0.15],
$$

则 H2 成立。

### H3：official controller 必须是 signal-routed，不是 dataset-routed

H3 要求 official route 不允许出现：

```text
if dataset == Fashion: use F14
if dataset == KMNIST: use K6+N2a
if dataset == MNIST: abstain
```

允许出现：

```text
if CE-tail and delayed-margin signal:
  use delayed margin controller

if hard-tail and actual-effect predicted:
  use metric-causal controller

if predicted Real-control gap <= 0:
  abstain
```

H3 成立标准：

Leave-dataset-out replay 中，controller 在 unseen dataset 上仍满足 task-safe，并在至少一个 mechanism metric 上不劣于 AdamW：

$$
Acc_{\text{unseen}}\geq Acc_{\text{AdamW}}-0.005.
$$

且：

$$
CEp99_{\text{unseen,functional}}\leq CEp99_{\text{AdamW}}+\epsilon.
$$

### H4：KMNIST survivor 不保留是 effect magnitude 问题，而不是 target survivor 完全无效

K6 + N2a 在 stronger replay 中 actual-effect rate 为 $0.766667$，但 beats AdamWParallel 只有 $0.266667$。H4 认为它不是无效，而是 effect magnitude/control gap 不够。

H4 成立标准：

如果按 effect magnitude 分层，高-magnitude bucket 满足：

$$
BeatRate_{\text{high-mag,Real vs AdamWParallel}}\geq0.50,
$$

但 low-magnitude bucket 不满足，则 KMNIST blocker 是 effect magnitude selection。

### H5：Fashion signal 不能通过调 horizon/cap 解决，必须通过 event ranking / abstention 解决

F14 ControlBeatAbstain 是当前 best Fashion，但 beats AdamWParallel 仍只有 $0.283333$。H5 认为 Fashion 的 controller 不应再继续 horizon/cap tuning，而要做 event ranking。

H5 成立标准：

若新增 event-value predictor 可以提升 Fashion beat rate：

$$
BeatRate_{\text{Fashion,Real vs AdamWParallel}}\geq0.50
$$

且保持：

$$
\Delta CEp99<0
$$

或：

$$
\Delta Margin>0,
$$

则 Fashion 是 ranking problem；若不能，则 Fashion 只保留 diagnostic，不作为 route driver。

### H6：如果 signal-routed controller 仍失败，下一步应回到 primitive/interface，而不是 dataset-specific tuning

如果 dataset-agnostic signal routing 失败，则不能写 v9.2.30 Fashion/KMNIST specific patch。必须回到：

```text
functional primitive interface
event-value predictor
basis/channel design
AdamW-only base full-pass
```

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

### 3.2 Dataset-agnostic signal strata

每个 event 必须记录以下 stratum features：

```text
ce_tail_rank
margin_tail_rank
wrong_confidence_rank
curvature_rank
actual_logit_movement
tail_logit_movement
nonadamw_logit_movement
cos_with_adamw
cos_with_best_lr
branch_ratio
effective_derivative
predicted_real_minus_adamwparallel
predicted_real_minus_best_lr
control_gap_uncertainty
event_age_or_horizon
```

预注册 stratum：

```text
S1-HighCEHighMarginRisk
S2-LowMarginHighWrongConfidence
S3-HighCurvatureLowConfidence
S4-HighActualMovementButControlDominated
S5-LowActualMovementSilent
S6-DelayedTailSignal
S7-HighSignalLowControlUncertainty
S8-AbstainCandidate
```

### 3.3 Signal-routed controller candidates

```text
SR0-AdamWOnly:
  no functional update reference.

SR1-GlobalBestReplay:
  non-routed global best reference.

SR2-SeverityRouted:
  route by CE/margin/wrong-confidence severity, no dataset label.

SR3-ControlGapRouted:
  update only if predicted Real-control gap is positive.

SR4-DelayedTailRouted:
  delayed signal only if event_age/horizon and tail severity agree.

SR5-ActualEffectRouted:
  require actual movement magnitude and tail movement above threshold.

SR6-AbstainHeavySignalRouted:
  abstain unless SR2/SR3/SR5 all agree.

SR7-FT7MechanismRouted:
  route by branch ratio / derivative-scale band.

SR8-OrthogonalTailRouted:
  only use non-AdamW orthogonal component on tail events.

SR9-CrossDatasetCalibratedRouter:
  thresholds selected by leave-one-dataset-out calibration, not per dataset.
```

### 3.4 Primitive candidates

Primitives are allowed, but selection must not be dataset-specific.

```text
P0-N2a-Rational
P1-N2c-SharedRBF
P2-N3c-SharedRBFDerivativeBand
P3-N2a-OrthogonalTail
P4-N2c-OrthogonalTail
P5-LightHybrid-RationalSharedRBF
P6-LightHybrid-RationalPiecewise
```

Official primitive selection criterion:

```text
choose primitive by cross-stratum performance,
not by dataset-specific best row.
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
```

`SeverityBucketShuffled` 和 `SignalScoreShuffled` 是 v9.2.29 新增必要 controls，用于区分真正 signal routing 与 bucket overfit。

---

## 4. 实验阶段

## P0：v9.2.28 boundary reproduction

### 目标

复现 latest boundary，确认不是测量噪声。

### 必须记录

```text
route
fashion_best_candidate
fashion_best_effect_size
fashion_best_beats_adamwparallel
fashion_best_beats_best_lr
fashion_signal_confirmed
kmnist_best_target
kmnist_best_primitive
kmnist_best_actual_effect_rate
kmnist_best_beats_adamwparallel
kmnist_best_beats_best_lr
kmnist_survivor_preserved
best_routed_controller
best_routed_beats_adamwparallel
best_routed_beats_best_lr
routing_overfit_controls_pass
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R5-KMNISTSurvivorNotPreserved
Fashion effect control-dominated reproduced
KMNIST survivor not preserved reproduced
Routed pass = 0
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_gate_ladder.svg
p0_fashion_kmnist_routed_status.svg
```

---

## P1：Dataset-stratified diagnosis without dataset tuning

### 目标

诊断不同数据集的问题，但不针对数据集调参。  
P1 只用于把 failure 转换成 signal strata。

### 必须记录

```text
dataset
seed
event_id
candidate
branch
ce_tail_rank
margin_tail_rank
wrong_confidence_rank
curvature_rank
actual_logit_movement
tail_logit_movement
cos_with_adamw
branch_ratio
effective_derivative
predicted_control_gap
actual_control_gap
failure_mode
signal_stratum
```

### 判断标准

P1 pass requires:

```text
Every failure row assigned to a signal stratum.
No route decision uses dataset name.
Association(signal_stratum, failure_mode) > Association(dataset, failure_mode).
```

Quantitative gate:

$$
A_{\text{stratum-failure}}
\geq
A_{\text{dataset-failure}}+0.10.
$$

where $A$ can be normalized mutual information or Cramér-style association score.

### 可视化

```text
p1_signal_stratum_failure_map.svg
p1_dataset_vs_stratum_association.svg
p1_tail_movement_vs_control_gap.svg
p1_event_value_distribution.svg
```

---

## P2：Event-value predictor and abstention calibration

### 目标

建立 dataset-agnostic event-value predictor，判断 event 是否值得 functional update。

### Predictor inputs

```text
ce_tail_rank
margin_tail_rank
wrong_confidence_rank
curvature_rank
actual_logit_movement
tail_logit_movement
branch_ratio
effective_derivative
cos_with_adamw
event_age_or_horizon
primitive_family
```

Forbidden inputs:

```text
dataset_name
seed_id
class_name as route key
posthoc outcome label
```

### 必须记录

```text
event_id
features
predicted_real_minus_adamwparallel
predicted_real_minus_best_lr
actual_real_minus_adamwparallel
actual_real_minus_best_lr
accepted
abstained
precision
recall
coverage
bad_event_rate
```

### 判断标准

Predictor pass:

$$
Corr(S_{\text{pred}},S_{\text{actual}})\geq0.30.
$$

Abstention pass:

$$
Precision_{\text{accepted beats controls}}\geq0.70,
$$

$$
Coverage\in[0.02,0.15],
$$

$$
BadEventRate\leq0.05.
$$

### 可视化

```text
p2_predicted_vs_actual_control_gap.svg
p2_abstention_precision_recall.svg
p2_coverage_vs_precision.svg
p2_event_value_calibration_curve.svg
```

---

## P3：Leave-dataset-out controller validation

### 目标

防止 dataset tuning。  
任何 official controller 必须在 leave-one-dataset-out 下仍然 task-safe，并至少在一个机制指标不劣。

### 设置

```text
splits:
  train-controller on MNIST + Fashion, evaluate KMNIST
  train-controller on MNIST + KMNIST, evaluate Fashion
  train-controller on Fashion + KMNIST, evaluate MNIST

controllers:
  SR1-SR9
primitives:
  P0-P6
horizons:
  20,80,240,640
controls:
  all strong controls
```

### 必须记录

```text
train_datasets
eval_dataset
controller
primitive
thresholds
CEp99_delta
margin_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
real_beats_adamwparallel
real_beats_best_lr
task_safe
coverage
bad_event_rate
shuffle_control_pass
```

### 判断标准

Leave-dataset-out pass:

$$
Acc_{\text{eval,Real}}\geq Acc_{\text{eval,AdamW}}-0.005.
$$

and at least one:

$$
CEp99_{\text{eval,Real}}\leq CEp99_{\text{eval,AdamW}}+\epsilon,
$$

$$
MarginP10_{\text{eval,Real}}\geq MarginP10_{\text{eval,AdamW}}-\epsilon,
$$

$$
Curvature_{\text{eval,Real}}\leq1.05Curvature_{\text{eval,AdamW}}.
$$

Control superiority precondition:

At least two leave-one-dataset-out splits must satisfy:

$$
BeatRate_{\text{Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{Real vs best LR}}\geq0.50.
$$

### 可视化

```text
p3_leave_dataset_out_matrix.svg
p3_controller_generalization_pareto.svg
p3_shuffle_control_comparison.svg
```

---

## P4：Official signal-routed paired replay

### 目标

只允许 P2/P3 pass 的 dataset-agnostic controller 进入 official paired replay。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
controllers = P2/P3 survivors
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, InvertedRole, SeverityBucketShuffled, SignalScoreShuffled
```

### 必须记录

```text
controller
primitive
dataset
seed
horizon
branch
signal_stratum
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

Official paired replay pass:

$$
BeatRate_{\text{macro,Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro,Real vs best LR}}\geq0.60.
$$

Task safety:

$$
Acc_{\text{each dataset,Real}}\geq Acc_{\text{AdamW}}-0.005.
$$

Anti-overfit controls:

```text
SeverityBucketShuffled = fail
SignalScoreShuffled = fail
DatasetRouteShuffled = fail
EventRouteShuffled = fail
InvertedRoleMask = fail
```

System:

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p4_official_paired_replay_pareto.svg
p4_macro_beat_rate.svg
p4_signal_stratum_win_matrix.svg
p4_real_vs_controls_by_stratum.svg
p4_system_gate_distribution.svg
```

---

## P5：Short-run validation

### 目标

验证 local causality 是否能在连续训练中保持。只有 P4 survivor 进入。

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

Task safety:

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005.
$$

Control superiority:

$$
MetricGain_{\text{functional}}>MetricGain_{\text{AdamWParallel}},
$$

$$
MetricGain_{\text{functional}}>MetricGain_{\text{best LR control}}.
$$

Mechanism pass:

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
p5_short_run_task_mechanism_pareto.svg
p5_short_run_controls.svg
p5_event_timeline.svg
p5_ce_tail_margin_panel.svg
```

---

## P6：Full 10-seed functional validation

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
delta_vs_best_lr
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

Task safety:

$$
\Delta Acc_{\text{functional-vs-AdamW}}\geq-0.005.
$$

Macro improvement:

$$
\Delta Acc_{\text{macro,functional}}
-
\Delta Acc_{\text{macro,AdamW}}
\geq0.003.
$$

Hard-stratum repair:

$$
\Delta Acc_{\text{hard-stratum,functional}}
-
\Delta Acc_{\text{hard-stratum,AdamW}}
\geq0.005.
$$

Control superiority:

$$
MetricGain_{\text{functional}}>MetricGain_{\text{AdamWParallel}},
$$

$$
MetricGain_{\text{functional}}>MetricGain_{\text{best LR control}}.
$$

Mechanism pass:

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
p6_macro_delta_vs_controls.svg
p6_hard_stratum_repair_matrix.svg
p6_seedwise_win_matrix.svg
p6_task_geometry_pareto.svg
p6_ce_tail_margin_panel.svg
p6_functional_channel_usage_trace.svg
```

---

## P7：AdamW-only full-pass repair

### 目标

Functional 是核心，但 base 仍未 full-pass。P7 并行执行，不使用 functional update。

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

Full-pass:

$$
\Delta Acc_{\text{macro}}\geq0.
$$

Robust near-pass:

$$
near\_pass\_rate\geq0.80,
$$

$$
\Delta Acc_{\text{macro}}\geq-0.01.
$$

### 可视化

```text
p7_adamw_fullpass_gap.svg
p7_hard_stratum_miss_rows.svg
p7_ce_tail_margin.svg
p7_basis_entropy_vs_gap.svg
```

---

## P8：Robustness and external-ready gate

### 目标

确认 strict PureKAN functional advantage 不是 clean MNIST-family artifact。

### 设置

```text
label_noise = 0.05,0.10,0.20
input_noise = 0.05,0.10
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
```

Strong baselines:

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

Robustness pass:

$$
AccDrop_{\text{functional}}\leq AccDrop_{\text{AdamW}}.
$$

Noise-tail pass:

$$
CEp99_{\text{functional}}\leq CEp99_{\text{AdamW}}.
$$

Strong baseline pass:

$$
Acc_{\text{functional}}\geq Acc_{\text{QuadraticFeatureMLP}}-0.005
$$

or:

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{QuadraticFeatureMLP}}.
$$

### 可视化

```text
p8_noise_robustness_curve.svg
p8_strong_baseline_pareto.svg
p8_external_ready_scorecard.svg
```

---

## 6. Required artifacts

```text
run_manifest.json
contract_audit_v9229.csv
p0_v9228_boundary_reproduction.csv
p1_dataset_stratified_diagnosis.csv
p2_event_value_predictor_abstention.csv
p3_leave_dataset_out_controller_validation.csv
p4_official_signal_routed_paired_replay.csv
p5_short_run_functional_validation.csv
p6_full_10seed_functional_validation.csv
p7_adamw_only_fullpass_repair.csv
p8_robustness_external_ready.csv
signal_stratum_trace_v9229.csv
event_value_prediction_trace_v9229.csv
leave_dataset_out_trace_v9229.csv
paired_replay_branch_trace_v9229.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy:

```text
F1_contract_violation
F2_v9228_boundary_unstable
F3_dataset_tuning_detected
F4_signal_stratum_unattributed
F5_event_value_predictor_fail
F6_abstention_precision_fail
F7_leave_dataset_out_fail
F8_signal_router_control_equivalent
F9_signal_router_overfit_shuffled_pass
F10_functional_lr_equivalent
F11_short_run_task_drop
F12_full_run_no_macro_hard_stratum_gain
F13_adamw_fullpass_fail
F14_strong_baseline_explains_gain
F15_robustness_fail
F16_external_not_ready
F17_fake_or_proxy_violation
F18_artifact_missing
```

---

## 7. Route decision

### Route cases

```text
R1-SignalStrataExplainFailures:
  Dataset-specific failures are explained by dataset-agnostic signal strata.

R2-EventValuePredictorPass:
  event-value predictor achieves control-beat precision and coverage.

R3-LeaveDatasetOutPass:
  signal-routed controller generalizes under leave-dataset-out.

R4-SignalRoutedPairedPass:
  official paired replay beats AdamWParallel / best LR under anti-overfit controls.

R5-FashionControlDominatedDiagnostic:
  Fashion effect exists but remains control-dominated; no dataset-specific patch allowed.

R6-KMNISTEffectMagnitudeDiagnostic:
  KMNIST actual effect exists but magnitude/control gap insufficient.

R7-StrictPureKANFunctionalShortRunPass:
  short-run task-safe mechanism gain.

R8-StrictPureKANFunctionalFullPass:
  full 10-seed macro / hard-stratum / geometry gain.

R9-AdamWFullPassNoFunctional:
  base reaches full-pass but functional remains unproven.

R10-ReturnToInterfacePrimitiveDesign:
  signal routing fails; redesign primitive/interface rather than dataset tuning.

R11-ExternalReady:
  strict PureKAN functional route passes task/geometry/system/control/robustness/strong-baseline gates.
```

### route_decision.json 必须记录

```text
route
v9228_boundary_pass
dataset_tuning_detected
signal_strata_explain_failures
event_value_predictor_pass
leave_dataset_out_pass
best_signal_router
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
success_v9229_strict_purekan_functional
success_v9229_full_functional
success_v9229_external_ready
```

---

## 8. 第一轮执行顺序

```text
Step 1:
  P0 复现 v9.2.28 boundary。

Step 2:
  P1 做 dataset-stratified diagnosis。
  只诊断，不允许 route 使用 dataset name。

Step 3:
  P2 训练/校准 event-value predictor。
  目标是预测 Real vs controls 的 gap，而不是预测 dataset。

Step 4:
  P3 做 leave-dataset-out controller validation。
  防止 Fashion/KMNIST-specific tuning。

Step 5:
  P4 official signal-routed paired replay。
  加入 SeverityBucketShuffled / SignalScoreShuffled controls。

Step 6:
  P5 short-run validation。
  只有 P4 survivor 才进入。

Step 7:
  P6 full 10-seed validation。

Step 8:
  P7 并行 AdamW-only full-pass repair。

Step 9:
  P8 robustness / external-ready。
```

---

## 9. 停止条件

### Minimum success

```text
v9.2.28 boundary reproduced
no dataset-specific route used
signal strata explain failures
event-value predictor passes
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
1. v9.2.28 boundary cannot be reproduced；
2. failures cannot be explained without dataset name；
3. event-value predictor cannot predict control gap；
4. leave-dataset-out fails；
5. signal router remains control-equivalent；
6. shuffled signal controls pass, indicating routing overfit；
7. full run gives no macro / hard-stratum / geometry gain；
8. functional breaks system gate；
9. gains are explained by QuadraticFeatureMLP；
10. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 10. 最终解释规则

### Case A：signal strata explain failures, but no controller passes

必须声明：

```text
We understand the failure modes, but current functional interface cannot yet exploit them.
```

下一步回到 primitive/interface/event-value predictor design。

### Case B：event-value predictor passes, but leave-dataset-out fails

必须声明：

```text
Current predictor is dataset-specific; it cannot support official functional success.
```

不能通过 dataset tuning 写成功。

### Case C：paired replay passes under anti-overfit controls

可以声明：

```text
Strict PureKAN functional has dataset-agnostic local causality evidence.
```

但仍需要 short/full validation。

### Case D：short/full run passes

可以声明：

```text
Strict FC-PureKAN functional update has been recovered under strong controls.
```

### Case E：all fail

必须声明：

```text
Current routing/controller family cannot generalize; return to deeper primitive/interface design.
```

---

## 11. 最终建议

v9.2.29 的一句话策略是：

$$
\boxed{
\text{不要针对数据集调参；把 Fashion/KMNIST 的差异转化为 dataset-agnostic signal strata，再学习何时 functional update 应该触发或 abstain。}
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
1. 哪类 signal event 真的值得 functional update？
2. 哪类 event 应该 abstain？
3. RealFunctional 何时能预测性地超过 AdamWParallel / best LR？
4. 这种选择规则能否 leave-dataset-out 泛化？
5. 如果不能，是否说明 current primitive/interface 仍不具备通用 functional actuatability？
```
