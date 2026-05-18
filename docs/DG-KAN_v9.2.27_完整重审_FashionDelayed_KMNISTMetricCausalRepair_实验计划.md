# DG-KAN v9.2.27 完整重审版：Fashion Delayed Signal Verification 与 KMNIST Metric-Causal Primitive Repair 实验计划

> 本计划基于重新补全后的 v9.2.23-v9.2.26 复盘结果制定。  
> 这版计划修正一个重要点：当前最新边界不是 v9.2.23 的 `FunctionalEventSilent`，而是 v9.2.26 的：
>
> ```text
> route = R2-FashionPartialDelayedSignalKMNISTPrimitiveBlocker
> base_candidate = LQ-t2-h256
> success_v9226_fashion_delayed_controller = False
> success_v9226_strict_purekan_functional = False
> ```
>
> 因此下一步不能再只围绕 “N2a event silent” 做放大；v9.2.24 已经证明 activation scale 可以修复 silence，但仍 control-equivalent。v9.2.25 又证明 Fashion-MNIST 存在 delayed h80 局部信号，而 KMNIST 放大/orthogonal 后仍没有可归因收益。v9.2.26 则进一步确认：Fashion delayed controller 仍只是 partial signal，KMNIST 的核心 blocker 是 `realized_effect_not_metric_causal`。
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

## 0. 当前状态的独立判断

### 0.1 最新实验没有达到目标

v9.2.26 没有达到 strict PureKAN functional success。当前 route 是：

```text
route = R2-FashionPartialDelayedSignalKMNISTPrimitiveBlocker
base_candidate = LQ-t2-h256
success_v9226_fashion_delayed_controller = False
success_v9226_strict_purekan_functional = False
success_v9226_external_ready = 0
```

关键结果是：

```text
Fashion best candidate = F3-Fashion-O2-DelayedController
Fashion best scope = h50
Fashion best CEp99 delta = -0.0002285639444986979
Fashion best margin delta = +0.0011636714140574138
Fashion beats AdamWParallel = 0.6666666666666666
Fashion beats best LR = 0.6666666666666666
Fashion task safe = 1.0

KMNIST target oracle useful rate = 1.0
KMNIST raw fit pass rate = 1.0
KMNIST safe fit pass rate = 1.0
KMNIST max raw R2 = 1.0
KMNIST max safe R2 = 1.0
KMNIST max cap rz = 0.640339195728302
KMNIST actual effect pass rate = 0.0
KMNIST mean actual CEp99 delta = 2.1192762586805554e-07
KMNIST mean actual margin delta = 1.0596381293402777e-07
```

这些数值说明：Fashion 有 delayed 局部信号，但幅度和覆盖率都不够；KMNIST 更深，它不是 “不能 fit target”，而是 **fit 出来的方向没有转化成 metric-causal gain**。

### 0.2 当前已经完成的进展

过去几轮已经完成了非常多前置门：

```text
1. v8-FT7 functional core 已经在 strong controls 下被保留；
2. strict PureKAN interface 的 contract / gradcheck / P4 已经可实现；
3. base-neutral interface N2a 已经能做到 P4/P5/actuatability pass；
4. v9.2.23 证明 actuatability proxy 有排序信息，但 realized movement 太小；
5. v9.2.24 证明 event-time activation scale 可以修复 silence，但仍输给 controls；
6. v9.2.25 证明 Fashion-MNIST 存在 delayed h80 局部信号；
7. v9.2.26 证明 KMNIST 的 blocker 是 target/safe-fit 到 actual metric effect 的断裂。
```

因此当前 blocker 已经不是：

```text
P4 system；
P5 base qualification；
functional channel dead；
target oracle completely wrong；
solver cannot fit；
event silent in all datasets；
FT7 core 不存在；
no-fake/no-proxy 可信度。
```

当前 blocker 是：

$$
\boxed{
\text{functional movement 没有稳定进入 metric-causal signal channel。}
}
$$

### 0.3 这轮结果的本质

Fashion 的结果说明 delayed functional update 可能在某些 tail/margin 结构上有用，但当前 signal 只是 partial：

$$
\text{BeatRate}=2/3,
$$

并且 effect size 很小：

$$
\Delta CEp99=-2.2856\times10^{-4},
$$

$$
\Delta Margin=1.1637\times10^{-3}.
$$

KMNIST 的结果更关键。它满足：

$$
OracleUsefulRate=1,
$$

$$
SafeFitPassRate=1,
$$

但：

$$
ActualEffectPassRate=0.
$$

这意味着 KMNIST 的问题不是 “模型不会动”，也不是 “target 方向完全错”，而是 **oracle logit movement / safe parameter fit 与实际训练轨迹的 metric 改善之间没有因果传递**。这与 signal-channel 视角一致：不是所有可实现的输出方向都会转化为可泛化、可积累的训练信号；只有进入 coherent signal channel 的方向才有价值。

---

## 1. v9.2.27 总体目标

v9.2.27 的总体目标是：

$$
\boxed{
\text{把 Fashion 的 partial delayed signal 变成可复现 survivor，并把 KMNIST 的 oracle-safe-fit 断点修成 actual metric-causal effect。}
}
$$

这不是再做一个 output target patch，而是把 functional update 的因果链拆成四层：

$$
\text{oracle target useful}
\rightarrow
\text{safe parameter fit}
\rightarrow
\text{realized logit movement}
\rightarrow
\text{metric-causal gain}.
$$

v9.2.26 已经说明 Fashion 和 KMNIST 卡在不同位置：

```text
Fashion:
  delayed signal partial exists；
  需要验证稳定性、放大机制、horizon controller。

KMNIST:
  oracle useful + safe fit pass；
  actual metric effect = 0；
  需要重新设计 primitive / target-to-effect transfer。
```

因此本轮必须同时做两条线：

```text
Line A:
  Fashion delayed signal verification and amplification.

Line B:
  KMNIST metric-causal primitive repair.
```

最后再尝试构造统一或 routed strict PureKAN functional route。

---

## 2. 成功标准

### 2.1 Fashion local survivor

Fashion delayed signal 不能只在单个 h50/h80 row 上成立。它必须在多 seed、多 horizon、多 control 下稳定：

$$
\text{BeatRate}_{\text{Real vs AdamWParallel}}\geq0.75,
$$

$$
\text{BeatRate}_{\text{Real vs best LR}}\geq0.75.
$$

并且 task safe：

$$
Acc_{\text{Real}}\geq Acc_{\text{AdamW}}-0.005.
$$

机制幅度必须非微弱，至少满足一个：

$$
\Delta CEp99_{\text{Real}}
\leq
\Delta CEp99_{\text{best control}}
-
0.0005,
$$

$$
\Delta MarginP10_{\text{Real}}
\geq
\Delta MarginP10_{\text{best control}}
+
0.001,
$$

$$
Curvature_{\text{Real}}
\leq
0.90Curvature_{\text{best control}}.
$$

如果仍只是 $2/3$ beat rate 且 effect size 接近 0，则不能进入 short/full run，只能记录：

```text
FashionPartialOnly
```

### 2.2 KMNIST actual effect survivor

KMNIST 必须从：

```text
oracle useful = 1
safe fit = 1
actual effect = 0
```

推进到：

```text
actual effect pass rate >= 0.50
```

定义 actual effect pass：

$$
\Delta CEp99_{\text{Real}}
<
\Delta CEp99_{\text{AdamWParallel}},
$$

或：

$$
\Delta MarginP10_{\text{Real}}
>
\Delta MarginP10_{\text{AdamWParallel}},
$$

或：

$$
Curvature_{\text{Real}}
\leq0.90Curvature_{\text{AdamWParallel}}.
$$

同时：

$$
Acc_{\text{Real}}\geq Acc_{\text{AdamW}}-0.005.
$$

### 2.3 Unified routed survivor

最终 official candidate 至少需要：

```text
MNIST task-safe；
Fashion pass；
KMNIST actual-effect repaired or explicitly diagnosed with a primitive blocker；
Real beats AdamWParallel / best LR on >= 2 datasets；
P4/P5 strict interface pass；
paired replay pass。
```

更严格的 macro route：

$$
\text{BeatRate}_{\text{macro, Real vs AdamWParallel}}\geq0.60,
$$

$$
\text{BeatRate}_{\text{macro, Real vs best LR}}\geq0.60.
$$

### 2.4 Full functional success

进入 full 10-seed 后，成功标准为：

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
\Delta Acc_{\text{KMNIST,functional}}
-
\Delta Acc_{\text{KMNIST,AdamW}}
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

---

## 3. 核心假设

### H1：Fashion delayed signal 是真实但未稳定的 functional effect

v9.2.25 显示 Fashion h80 local signal，v9.2.26 把它提升到 50/240-step ablation 但未 pass。H1 认为 Fashion 不是噪声，而是 delayed event alignment 不稳定。

H1 成立标准：

Fashion delayed controller 在 10 seeds 下满足：

$$
\text{BeatRate}_{h50,h80,h160,h240}\geq0.75,
$$

并且 effect size 达到 Fashion local survivor 标准。

若只在一个 horizon 有弱信号，H1 不成立；Fashion route 只能作为 diagnostic。

### H2：KMNIST blocker 是 realized-effect-not-metric-causal，而不是 target oracle failure

v9.2.26 已经显示：

$$
OracleUsefulRate_{\text{KMNIST}}=1,
$$

$$
SafeFitPassRate_{\text{KMNIST}}=1,
$$

$$
ActualEffectPassRate_{\text{KMNIST}}=0.
$$

H2 要进一步判断：KMNIST 是 silent movement 还是 misaligned movement。

Silent movement：

$$
r_{z,\text{actual}}<0.10
$$

或：

$$
r_{z,\text{tail}}<0.10.
$$

Misaligned movement：

$$
r_{z,\text{actual}}\geq0.10
$$

但：

$$
\Delta CEp99_{\text{Real}}\geq\Delta CEp99_{\text{AdamWParallel}},
$$

且：

$$
\Delta MarginP10_{\text{Real}}\leq\Delta MarginP10_{\text{AdamWParallel}}.
$$

### H3：KMNIST 需要 local / hybrid functional primitive

KMNIST 的 hard-mode 可能需要局部 basis，而 rational channel 太 global。H3 候选包括：

```text
piecewise local functional channel；
shared-RBF functional channel；
rational + piecewise hybrid；
rational + shared-RBF hybrid；
warmstart attach；
orthogonal-tail local channel。
```

H3 成立标准：

至少一个 local / hybrid primitive 满足：

```text
contract pass = 1；
grad pass = 1；
P4 pass = 1；
P5 near-pass = 1；
ActualEffectPassRate >= 0.50；
task_safe = 1。
```

### H4：统一 controller 需要 routing，而不是 global one-size-fits-all

Fashion 是 delayed controller 问题，KMNIST 是 target/primitive-to-metric 问题，MNIST 可能应当 abstain。H4 认为统一 functional route 需要路由：

```text
Fashion:
  delayed O2 / margin-tail controller；

KMNIST:
  local / hybrid primitive + hard-tail realized-effect controller；

MNIST:
  abstain or only fire on high-confidence role-signal events。
```

这不是 class weight、sampler 或 loss change；它只是 functional update event policy。

H4 成立标准：

routed controller 比 global controller 更好：

$$
MetricGain_{\text{routed}}
>
MetricGain_{\text{global}},
$$

且：

$$
Acc_{\text{MNIST,routed}}\geq Acc_{\text{MNIST,AdamW}}-0.005.
$$

### H5：如果 Fashion 不稳定且 KMNIST actual effect 仍为 0，则当前 interface family 要回到 primitive design

如果 P1/P2/P3 都失败，不能继续调 cap 或 threshold。必须进入：

```text
R8-DelayedSignalNotGeneralized_PrimitiveReset
```

下一步回到 deeper primitive/interface design。

---

## 4. Candidate 设计

### 4.1 Baselines

```text
B0-MLP-match:
  same-parameter MLP reference。

LQ0-LQ-t2-h256:
  current strict FC-PureKAN base。

N2a-TinyInit-RationalFunc-BranchRatioCap:
  current base-qualified strict functional interface。

F3-Fashion-O2-DelayedController:
  latest Fashion partial signal source.

AdamWOnly:
  strict PureKAN no functional reference。
```

### 4.2 Fashion controllers

```text
F0-Fashion-NoFunctional:
  AdamW-only reference.

F1-Fashion-O2-h20:
  short delayed probe.

F2-Fashion-O2-h50:
  v9.2.26 best scope.

F3-Fashion-O2-h80:
  v9.2.25 best h80 reference.

F4-Fashion-O2-h160:
  delayed longer horizon.

F5-Fashion-O2-h240:
  full delayed horizon.

F6-Fashion-O1O2-BalancedDelayed:
  balance CEp99 and margin.

F7-Fashion-O2-BranchBand:
  FT7-like branch ratio calibrated.

F8-Fashion-O2-DerivativeBand:
  effective derivative scale calibrated.

F9-Fashion-O2-AbstainUnlessControlBeat:
  fire only if predicted Real > AdamWParallel / best LR.

F10-Fashion-O2-MultiHorizonVote:
  event accepted only if h50/h80/h160 predicted gains agree.

F11-Fashion-O2-TailOnlyEvent:
  fire only on CEp99-tail samples.

F12-Fashion-O2-MarginTailHybrid:
  require margin-tail not harmed while CEp99 improves.
```

### 4.3 KMNIST targets

```text
K0-KMNIST-OracleReference:
  diagnostic only.

K1-KMNIST-O2-MarginTail:
  margin-tail target.

K2-KMNIST-O1-CEp99Tail:
  CE-tail target.

K3-KMNIST-O6-HardMode:
  hard-mode output target.

K4-KMNIST-ConfusionTail:
  top2 confusion / wrong-confidence target.

K5-KMNIST-CurvatureTail:
  curvature / local-Lipschitz tail target.

K6-KMNIST-MetricCausalTarget:
  target selected by realized metric response, not direct oracle fit.

K7-KMNIST-AbstainUnlessActualEffect:
  abstain if event-time actual logit movement does not predict metric gain.

K8-KMNIST-ClassModeBucketTarget:
  target per confusion-pair / class-mode bucket.
```

### 4.4 KMNIST primitives

```text
P0-N2a-Rational:
  current reference.

P1-Piecewise2-BranchCap:
  local piecewise functional channel.

P2-Piecewise4-BranchCap:
  finer local piecewise channel.

P3-SharedRBF4-BranchCap:
  local shared RBF functional channel.

P4-RationalPlusPiecewise:
  global rational + local correction.

P5-RationalPlusRBF:
  global rational + smooth local bump.

P6-WarmstartAttach-Piecewise:
  task channel warmstart, attach piecewise functional channel.

P7-OrthogonalTail-Piecewise:
  orthogonal to AdamW on non-tail, active on tail.

P8-OrthogonalTail-RBF:
  shared RBF version of P7.

P9-ConfusionPairPiecewise:
  piecewise local basis specialized to top confusion buckets.

P10-HybridRationalPiecewiseRBF:
  global rational + piecewise + shared-RBF light hybrid.
```

### 4.5 Unified / routed controllers

```text
U0-GlobalBestSingle:
  one controller for all datasets.

U1-FashionDelayed-KMNISTLocal-MNISTAbstain:
  dataset-routed controller.

U2-TailMetricRouter:
  route by observed CEp99 / margin / curvature event.

U3-ControlBeatRouter:
  fire only when predicted Real > AdamWParallel / best LR.

U4-HybridFT7Router:
  route by branch ratio / derivative scale band.

U5-OracleSafeFitActualEffectRouter:
  route by oracle useful + safe fit + actual effect predictor.
```

### 4.6 Controls

Every paired replay / short / full run must include:

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
```

---

## 5. 实验阶段

## P0：v9.2.26 boundary reproduction

### 目标

复现最新 boundary，确认不是 measurement noise。

### 必须记录

```text
route
source_route_v9225
best_fashion_candidate
best_fashion_scope
fashion_best_CEp99_delta
fashion_best_margin_delta
fashion_best_beats_adamwparallel
fashion_best_beats_best_lr
fashion_ablation_pass
fashion_all_scope_pass_count
kmnist_target_oracle_useful_rate
kmnist_raw_fit_pass_rate
kmnist_safe_fit_pass_rate
kmnist_actual_effect_pass_rate
kmnist_max_cap_rz
kmnist_max_raw_R2
kmnist_max_safe_R2
kmnist_mean_actual_CEp99_delta
kmnist_mean_actual_margin_delta
kmnist_diagnosis_blocker
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R2-FashionPartialDelayedSignalKMNISTPrimitiveBlocker
Fashion partial delayed signal reproduced
KMNIST realized_effect_not_metric_causal reproduced
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_fashion_partial_signal_recap.svg
p0_kmnist_oracle_fit_actual_gap.svg
p0_route_gate_ladder.svg
```

---

## P1：Fashion delayed signal verification and amplification

### 目标

验证 Fashion delayed signal 是否能从 partial signal 变成 stable local survivor。

### 设置

```text
dataset = Fashion-MNIST
seeds = 0,1,2,3,4,5,6,7,8,9
horizons = 20,50,80,160,240,640
controllers = F0-F12
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, InvertedRole
```

### 必须记录

```text
candidate
seed
horizon
branch
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
task_safe
control_rank
real_beats_adamwparallel
real_beats_best_lr
event_count
event_coverage
bad_event_rate
branch_ratio
effective_derivative
actual_logit_delta
actual_tail_logit_delta
step_ratio_q90
memory_ratio
```

### 判断标准

Fashion local survivor：

$$
\text{BeatRate}_{\text{Real vs AdamWParallel}}\geq0.75,
$$

$$
\text{BeatRate}_{\text{Real vs best LR}}\geq0.75,
$$

$$
Acc_{\text{Real}}\geq Acc_{\text{AdamW}}-0.005.
$$

Mechanism magnitude gate：

At least one:

$$
\Delta CEp99_{\text{Real}}
\leq
\Delta CEp99_{\text{best control}}
-
0.0005,
$$

$$
\Delta MarginP10_{\text{Real}}
\geq
\Delta MarginP10_{\text{best control}}
+
0.001,
$$

$$
Curvature_{\text{Real}}
\leq
0.90Curvature_{\text{best control}}.
$$

If Fashion only beats at $2/3$ again with tiny effect:

```text
route = FashionPartialOnly
```

### 可视化

```text
p1_fashion_horizon_effect.svg
p1_fashion_controller_pareto.svg
p1_fashion_real_vs_controls.svg
p1_fashion_seedwise_win_matrix.svg
p1_fashion_branch_derivative_band.svg
p1_fashion_effect_size_distribution.svg
```

---

## P2：Fashion mechanism diagnosis

### 目标

如果 P1 signal 仍 weak，解释它是 delayed-horizon 问题、event selection 问题、branch activation 问题，还是 true partial dataset-specific effect。

### 必须记录

```text
candidate
horizon
seed
tail_sample_count
accepted_event_count
CEp99_tail_overlap
margin_tail_overlap
wrong_confidence_p95
top2_margin
logit_norm_p95
actual_tail_logit_delta
branch_ratio
effective_derivative
cos_functional_adamw
cos_functional_best_lr
time_to_effect
```

### 判断标准

Delayed alignment pass：

$$
Corr(horizon, -\Delta CEp99)\geq0.30
$$

or:

$$
Corr(horizon, \Delta MarginP10)\geq0.30.
$$

Branch mechanism pass：

$$
Corr(r_{\text{branch}}, -\Delta CEp99)\geq0.30
$$

or:

$$
Corr(s_{\text{eff}}, \Delta MarginP10)\geq0.30.
$$

If no mechanism explains the Fashion signal:

```text
Fashion signal = weak diagnostic only
```

### 可视化

```text
p2_fashion_time_to_effect.svg
p2_fashion_tail_overlap.svg
p2_fashion_branch_ratio_vs_gain.svg
p2_fashion_derivative_vs_gain.svg
```

---

## P3：KMNIST target-to-realized-effect diagnosis

### 目标

解释 KMNIST 为什么 oracle useful + safe fit pass，但 actual effect pass = 0。

### 设置

```text
dataset = KMNIST
seeds = 0,1,2,3,4,5,6,7,8,9
targets = K0-K8
primitive = P0-N2a-Rational
horizons = 1,5,20,80,240
```

### 必须记录

```text
target
seed
event_id
horizon
oracle_CEp99_delta
oracle_margin_delta
oracle_ECE_delta
safe_fit_R2
safe_fit_rz
actual_logit_delta_norm
actual_tail_logit_delta_norm
actual_nonadamw_delta_norm
actual_CEp99_delta
actual_margin_delta
actual_ECE_delta
actual_NLL_delta
actual_curvature_delta
wrong_confidence_p95_delta
top2_margin_delta
confusion_pair_id
class_or_mode_bucket
task_safe
bad_event
```

### 判断标准

Silent failure：

$$
r_{z,\text{actual}}<0.10
$$

or:

$$
r_{z,\text{tail}}<0.10.
$$

Misaligned failure：

$$
r_{z,\text{actual}}\geq0.10
$$

but:

$$
\Delta CEp99_{\text{Real}}\geq\Delta CEp99_{\text{AdamWParallel}},
$$

and:

$$
\Delta MarginP10_{\text{Real}}\leq\Delta MarginP10_{\text{AdamWParallel}}.
$$

KMNIST metric-causal target survivor：

$$
ActualEffectPassRate\geq0.50,
$$

with task safety.

### 可视化

```text
p3_kmnist_oracle_vs_actual.svg
p3_kmnist_safe_fit_vs_metric_gain.svg
p3_kmnist_tail_logit_delta.svg
p3_kmnist_confusion_pair_heatmap.svg
p3_kmnist_target_failure_taxonomy.svg
p3_kmnist_class_bucket_effect.svg
```

---

## P4：KMNIST primitive/interface repair

### 目标

如果 P3 shows rational N2a cannot convert oracle target to metric gain, test local / hybrid primitive interfaces.

### 设置

```text
dataset = KMNIST
seeds = 0,1,2
primitives = P0-P10
targets = best P3 targets
horizons = 20,80,240
```

### 必须记录

```text
primitive
target
seed
contract_pass
grad_pass
P4_forward_q90
P4_backward_q90
P4_step_q90
P4_memory
P5_nearpass
safe_fit_R2
actual_effect_pass
actual_CEp99_delta
actual_margin_delta
actual_curvature_delta
task_safe
branch_ratio
effective_derivative
functional_channel_entropy
dominant_basis_fraction
```

### 判断标准

Primitive survivor：

```text
contract_pass = 1
grad_pass = 1
P4 pass = 1
P5 near-pass = 1
ActualEffectPassRate >= 0.50
task_safe = 1
```

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

### 可视化

```text
p4_kmnist_primitive_pareto.svg
p4_kmnist_basis_family_effect.svg
p4_kmnist_actual_effect_by_primitive.svg
p4_kmnist_branch_derivative_by_primitive.svg
p4_kmnist_local_vs_global_primitive.svg
```

---

## P5：Unified / routed controller construction

### 目标

Combine Fashion survivor and KMNIST repaired primitive into one official strict PureKAN functional route.

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
controllers = U0-U5
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, InvertedRole
```

### 必须记录

```text
controller
dataset
seed
horizon
branch
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
real_beats_adamwparallel
real_beats_best_lr
task_safe
event_count
event_coverage
bad_event_rate
selected_event_type
selected_primitive
selected_target
step_ratio_q90
memory_ratio
```

### 判断标准

Unified paired replay pass：

```text
MNIST task-safe
Fashion pass
KMNIST actual-effect pass or repaired primitive pass
Real beats AdamWParallel / best LR on >= 2 datasets
no dataset task drop beyond -0.005
```

Macro route:

$$
\text{BeatRate}_{\text{macro, Real vs AdamWParallel}}\geq0.60,
$$

$$
\text{BeatRate}_{\text{macro, Real vs best LR}}\geq0.60.
$$

### 可视化

```text
p5_unified_controller_pareto.svg
p5_dataset_routing_matrix.svg
p5_real_vs_controls_all_datasets.svg
p5_event_type_usage_heatmap.svg
p5_routed_vs_global_comparison.svg
```

---

## P6：Short-run functional validation

### 目标

Only P5 survivor enters short-run.

### 设置

```text
steps = 50,240,640
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole
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
event_coverage
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
p6_short_run_task_mechanism_pareto.svg
p6_short_run_controls.svg
p6_event_timeline.svg
p6_ce_tail_margin_panel.svg
p6_step_ratio_distribution.svg
```

---

## P7：Full 10-seed validation

### 目标

Validate strict PureKAN functional advantage over full training.

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
event_coverage
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

KMNIST repair:

$$
\Delta Acc_{\text{KMNIST,functional}}
-
\Delta Acc_{\text{KMNIST,AdamW}}
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
p7_macro_delta_vs_controls.svg
p7_kmnist_repair_matrix.svg
p7_seedwise_win_matrix.svg
p7_task_geometry_pareto.svg
p7_ce_tail_margin_panel.svg
p7_ece_nll_panel.svg
p7_functional_channel_usage_trace.svg
```

---

## P8：AdamW-only full-pass repair

### 目标

Functional 是核心，但 base 仍未 full-pass。Parallel repair avoids attributing weak base to functional.

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
p8_adamw_fullpass_gap.svg
p8_kmnist_miss_rows.svg
p8_ce_tail_margin.svg
p8_basis_entropy_vs_gap.svg
```

---

## P9：Robustness and external-ready gate

### 目标

Confirm strict PureKAN functional advantage is not a clean-setting accident.

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
p9_noise_robustness_curve.svg
p9_strong_baseline_pareto.svg
p9_external_ready_scorecard.svg
```

---

## 6. Required artifacts

```text
run_manifest.json
contract_audit_v9227.csv
p0_v9226_boundary_reproduction.csv
p1_fashion_delayed_signal_verification.csv
p2_fashion_mechanism_diagnosis.csv
p3_kmnist_target_to_realized_effect_diagnosis.csv
p4_kmnist_primitive_interface_repair.csv
p5_unified_routed_controller.csv
p6_short_run_functional_validation.csv
p7_full_10seed_functional_validation.csv
p8_adamw_only_fullpass_repair.csv
p9_robustness_external_ready.csv
fashion_delayed_event_trace_v9227.csv
kmnist_realized_effect_trace_v9227.csv
primitive_interface_trace_v9227.csv
routed_controller_trace_v9227.csv
paired_replay_branch_trace_v9227.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy:

```text
F1_contract_violation
F2_v9226_boundary_unstable
F3_fashion_signal_not_reproduced
F4_fashion_signal_too_weak
F5_fashion_mechanism_unattributed
F6_kmnist_silent_movement
F7_kmnist_misaligned_movement
F8_kmnist_primitive_repair_fail
F9_unified_controller_fail
F10_paired_replay_control_equivalent
F11_functional_lr_equivalent
F12_short_run_task_drop
F13_full_run_no_macro_kmnist_gain
F14_adamw_fullpass_fail
F15_strong_baseline_explains_gain
F16_robustness_fail
F17_external_not_ready
F18_fake_or_proxy_violation
F19_artifact_missing
```

---

## 7. Route decision

### Route cases

```text
R1-FashionDelayedSignalConfirmed:
  Fashion delayed signal passes multi-seed / multi-horizon strong-control replay.

R2-FashionPartialOnly:
  Fashion signal exists but remains weak / horizon-local.

R3-KMNISTSilentMovement:
  KMNIST actual logit movement is too small despite oracle/safe-fit.

R4-KMNISTMisalignedMovement:
  KMNIST moves logits but not along metric-causal direction.

R5-KMNISTPrimitiveRepairPass:
  local / hybrid primitive achieves actual effect pass.

R6-UnifiedRoutedControllerPass:
  Fashion + KMNIST + MNIST route passes paired replay.

R7-StrictPureKANFunctionalShortRunPass:
  short-run shows task-safe mechanism gain.

R8-StrictPureKANFunctionalFullPass:
  full 10-seed shows macro/KMNIST/geometry/tail gain.

R9-DelayedSignalNotGeneralized_PrimitiveReset:
  Fashion signal fails replication and KMNIST repair fails.

R10-AdamWFullPassNoFunctional:
  base reaches full-pass but functional remains unproven.

R11-ExternalReady:
  strict PureKAN functional route passes task/geometry/system/control/robustness/strong-baseline gates.
```

### route_decision.json 必须记录

```text
route
v9226_boundary_pass
fashion_signal_confirmed
fashion_best_candidate
fashion_best_horizon
fashion_best_effect_size
fashion_mechanism_type
kmnist_failure_mode
kmnist_actual_effect_pass
best_kmnist_primitive
unified_controller_pass
paired_replay_pass
short_run_pass
full_run_pass
functional_task_safe
functional_control_pass
functional_system_pass
functional_kmnist_repair_pass
adamw_fullpass
strong_baseline_pass
robustness_pass
external_ready
primary_blocker
next_required_implementation
success_v9227_strict_purekan_functional
success_v9227_full_functional
success_v9227_external_ready
```

---

## 8. 第一轮执行顺序

```text
Step 1:
  P0 复现 v9.2.26 boundary。

Step 2:
  P1 先验证 Fashion delayed signal。
  如果 Fashion 仍只有 2/3 weak signal，则不能把它作为 full survivor。

Step 3:
  P2 对 Fashion 做 mechanism diagnosis。
  只有知道 delayed effect 的机制，才能设计 controller。

Step 4:
  P3 做 KMNIST oracle/safe-fit/actual-effect 断点定位。
  先判断 silent 还是 misaligned。

Step 5:
  P4 做 KMNIST primitive/interface repair。
  比较 rational、piecewise、RBF、hybrid、warmstart、orthogonal tail。

Step 6:
  P5 构造 unified/routed controller。
  只有 Fashion 和 KMNIST 都有可靠信号时才进入。

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

## 9. 停止条件

### Minimum success

```text
v9.2.26 boundary reproduced
Fashion signal confirmed or KMNIST repaired
at least one unified/routed strict functional candidate beats AdamWParallel / best LR in paired replay
no fake/proxy/offload/loss/teacher violation
```

### Full functional success

```text
Minimum success
+
short/full run task-safe mechanism gain
+
KMNIST repair or macro improvement
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
1. v9.2.26 boundary cannot be reproduced；
2. Fashion delayed signal is not reproduced；
3. Fashion signal is weak and mechanism unattributed；
4. KMNIST remains actual-effect pass rate = 0 under all repaired primitives；
5. unified controller remains control-equivalent；
6. full run gives no macro/KMNIST/geometry/tail gain；
7. functional breaks system gate；
8. gains are explained by QuadraticFeatureMLP；
9. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 10. 最终解释规则

### Case A：Fashion signal confirmed, KMNIST still fails

必须声明：

```text
functional signal is dataset-specific and delayed; strict PureKAN functional is not yet general.
```

可以继续研究 Fashion route，但不能声明 global strict PureKAN functional success。

### Case B：KMNIST silent movement

必须声明：

```text
current functional event is too weak under repaired primitive; repair event-time activation band or primitive sensitivity.
```

### Case C：KMNIST misaligned movement

必须声明：

```text
functional channel moves output but not along metric-causal hard-tail direction; repair target/primitive alignment.
```

### Case D：KMNIST primitive repair passes

可以声明：

```text
KMNIST blocker was primitive/interface mismatch, not functional concept failure.
```

### Case E：unified routed controller passes

可以声明：

```text
strict PureKAN functional has local paired-replay evidence under strong controls.
```

但 full success 仍需要 short/full validation。

### Case F：all fail

必须声明：

```text
current delayed-controller + rational/local interface family cannot generalize FT7 core; return to deeper primitive/interface design.
```

---

## 11. 最终建议

v9.2.27 的一句话策略是：

$$
\boxed{
\text{确认 Fashion delayed signal 是否真能站住，同时把 KMNIST 的 oracle-safe-fit 断点转化成 actual metric-causal effect。}
}
$$

当前最关键的问题不是 functional 是否存在，也不是 base 是否可训练，而是：

```text
1. Fashion 的 delayed signal 是否能在 10 seeds / 多 horizon 中稳定超过 AdamWParallel / best LR？
2. Fashion 的 delayed effect 是 CE-tail compression、margin repair、还是 curvature smoothing？
3. KMNIST 为什么 target oracle useful 与 safe fit 都通过，但 actual effect pass = 0？
4. KMNIST 是 silent movement 还是 misaligned movement？
5. local / hybrid primitive 是否能把 KMNIST oracle target 转成 hard-tail / margin-tail metric gain？
6. routed controller 是否能把 Fashion 与 KMNIST 的不同机制统一到一个 strict PureKAN functional route？
7. RealFunctional 最终能否超过 AdamWParallel / best LR，并进入 short/full validation？
```
