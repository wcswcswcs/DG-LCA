# DG-KAN v9.2.15 Control-Resistant Functional Causality：P4-Qualified Actuator 后的 Functional Advantage 完整实验计划

> 本计划基于 v9.2.14 `P4-Qualified Functional Actuator Closure` 的真实结果制定。  
> v9.2.14 的结论是一个重要分水岭：**我们终于获得了 P4-qualified 且 controllability-retaining 的 strict FC-PureKAN actuator，并且 A7c 在 AdamW-only base qualification 上保持 near-pass；但 functional paired replay 仍然没有击败 controls。**  
> 因此 v9.2.15 不再继续做 P4 closure，也不继续调 SNR/event 阈值。  
> 本轮的核心任务是：
>
> $$
> \boxed{
> \text{在 P4-qualified actuator 已经存在的前提下，证明或证伪 functional update 的独特因果收益。}
> }
> $$
>
> 也就是说，v9.2.15 的问题不再是 “有没有可控 actuator”，而是：
>
> $$
> \boxed{
> \text{为什么 RealFunctional 没有比 NoOp / Random / Shuffled / AdamWParallel controls 更好？}
> }
> $$
>
> 本计划继续遵守：no teacher、no self-teacher、no distillation、no loss modification、no label smoothing、no focal/margin/calibration loss、no sampler/class weight、no CPU offload、no fake/proxy rows、KAN path 不使用 PyTorch loss.backward graph、PureKANConv/PureKANFormer 继续 deferred。

---

## 0. 当前状态与独立判断

### 0.1 v9.2.14 的真实 route

v9.2.14 的最终 route 是：

```text
route = R4-ActuatorP4ClosedBaseQualified
base_candidate = LQ-t2-h256
best_actuator_candidate = A7c-BasisEntropy-ValueOnly
success_v9214_p4_qualified_actuator = true
success_v9214_absorbable_actuator = false
success_v9214_functional_advantage = false
success_v9214_external_ready = false
```

这说明 v9.2.14 没有达到 functional advantage，也没有进入 external-ready。  
但它确实推进了一个关键 gate：**P4-qualified actuator 已经找到。**

### 0.2 v9.2.14 的核心正向结果

v9.2.14 复现了 v9.2.13 的边界：

```text
source route = R8-NoPureKANActuatorFound
post-actuator controllability pass count = 273
source actuator P4 pass count = 0
```

P2 之后找到多个 P4-qualified + controllability-retaining actuator：

```text
A4b
A4d
A4e
A7c
```

最佳 route candidate 是：

```text
A7c-BasisEntropy-ValueOnly
```

其关键指标为：

```text
forward q90 = 1.104638
backward q90 = 1.387102
step q90 = 1.213123
compact memory = 0.969731
target fit R2 = 1.0
rz = 1.106568
```

这意味着：

$$
T_{\text{forward}}\leq1.25T_{\text{MLP}},
$$

$$
T_{\text{backward}}\leq1.50T_{\text{MLP}},
$$

$$
T_{\text{step}}\leq1.50T_{\text{MLP}},
$$

$$
M_{\text{peak}}\leq1.05M_{\text{MLP}},
$$

并且 actuator 对 output target 的 controllability 很强：

$$
R^2_{\text{fit}}=1.0,
$$

$$
r_z=1.106568.
$$

P4 AdamW-only base qualification 也真实打开：

```text
A7c:
  datasets = MNIST,Fashion-MNIST,KMNIST
  seeds = 0,1,2
  rows = 9
  near-pass = 8/9
  macro delta = -0.004167
```

所以 v9.2.14 已经把上一轮的硬 blocker 从：

```text
No P4-qualified actuator
```

推进为：

```text
P4-qualified actuator exists, but functional causality is not established.
```

### 0.3 v9.2.14 的核心失败点

P5 paired replay 真实打开，但 RealFunctional 没有击败 controls：

```text
Real CEp99 mean delta = +0.001747
best control CEp99 mean delta = -0.008564

Real margin mean delta = -0.000325
best control margin mean delta = +0.000943
```

这两个数说明：在当前 paired replay 下，RealFunctional 不仅没有带来 CE-tail / margin-tail 改善，反而在平均机制指标上弱于 best control。

因此，本轮真正失败的是：

$$
\boxed{
\text{functional causality against matched controls.}
}
$$

而不是：

```text
P4 system
actuator controllability
AdamW base qualification
memory
SNR overhead
event coverage
output target existence
```

### 0.4 当前是否达到终极目标

没有。

当前已经达成：

```text
1. Clean FC-PureKAN LQ base 成立；
2. LQ P4 system gate 成立；
3. LQ P5 robust near-pass 成立；
4. Direct logit target 有用；
5. Actuator controllability 有强信号；
6. A7c 成为 P4-qualified actuator；
7. A7c AdamW-only base near-pass；
8. Paired replay infrastructure 真实打开。
```

当前仍未达成：

```text
1. RealFunctional beats controls；
2. short-run functional advantage；
3. full 10-seed functional re-entry；
4. robustness / noisy-signal advantage；
5. strong baseline / external-ready；
6. significant Beyond-MLP。
```

因此，v9.2.15 必须把问题聚焦到：

$$
\boxed{
\text{control-resistant functional causality.}
}
$$

---

## 1. 当前问题的本质

### 1.1 不是系统实现问题

A7c 已经通过 P4：

```text
forward q90 = 1.104638
backward q90 = 1.387102
step q90 = 1.213123
memory = 0.969731
```

这些指标均在 P4 envelope 内。因此当前不是 v9.2.3-v9.2.4 那种 forward/backward kernel blocker，也不是 v9.2.13 那种 actuator P4 blocker。

### 1.2 不是普通 optimizer 问题

A7c AdamW-only base qualification 为：

```text
near-pass = 8/9
macro delta = -0.004167
```

这和 LQ-t2-h256 的 near-pass 状态一致。说明 AdamW base 已经能训练到接近 MLP-match。当前不应回到 lr / wd / scheduler sweep。

### 1.3 不是 output target 完全无效

v9.2.13 direct logit oracle 已经证明 O1/O2/O3/O4/O6 output targets 有 useful signal。  
v9.2.14 中 A7c target fit R2 为 1.0，rz 为 1.106568。  
因此不是 target 不能被 actuator 实现，也不是 actuator 无法移动 logits。

### 1.4 当前真正 blocker

当前 blocker 是：

$$
\boxed{
\text{我们能移动输出，但移动方向在真实训练分支中没有比 controls 更好。}
}
$$

这意味着问题进入了 functional update 的最本质层面：

```text
1. target 是否只在 direct oracle 中有用，但参数实现后不稳？
2. solver 是否只拟合 target，却没有约束 multi-step downstream effect？
3. RealFunctional 是否过大/过小/方向错误？
4. best control 是否本身就是更好的 regularizer？
5. paired replay 的 scoring 是否被 controls 的随机 tail improvement 主导？
6. A7c 的 basis-entropy actuator 是否可控但机制不对？
7. A4 bounded rational family 是否比 A7c 更适合 causal replay？
```

---

## 2. v9.2.15 整体目标

v9.2.15 的整体目标是：

$$
\boxed{
\text{在 P4-qualified actuator 条件下，建立 RealFunctional 相对 controls 的因果优势，或者证明当前 actuator-target-solver 不具备 functional advantage。}
}
$$

本轮的成功不是单点 accuracy，而是完整因果链：

```text
P4-qualified actuator
+
AdamW base near-pass
+
paired replay beats controls
+
short-run mechanism gain
+
full-run task-safe improvement
+
robustness / strong baseline challenge
```

v9.2.15 分成三层成功。

### 2.1 Minimum success

最低成功：

```text
P0 boundary reproduction pass
P1 paired replay failure attribution pass
P2 all P4-qualified actuator causality matrix measured
At least one RealFunctional branch beats best matched control in paired replay
No fake/proxy/offload/loss/teacher violation
```

Paired replay gate 为：

$$
CEp99_{\text{real}}
\leq
CEp99_{\text{best-control}}
-
0.05|CEp99_{\text{AdamW}}|,
$$

或：

$$
MarginP10_{\text{real}}
\geq
MarginP10_{\text{best-control}}
+
0.02,
$$

或：

$$
Curvature_{\text{real}}
\leq
0.90Curvature_{\text{best-control}},
$$

并且：

$$
Acc_{\text{real}}\geq Acc_{\text{AdamW}}-0.005.
$$

### 2.2 Functional advantage success

Functional advantage success 要求：

```text
Minimum success
+
short-run pass
+
full 10-seed task-safe mechanism gain
```

其中 full 10-seed 至少满足：

$$
\Delta Acc_{\text{functional-vs-AdamW}}\geq-0.005,
$$

且至少一个成立：

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
ECE_{\text{functional}}\leq ECE_{\text{AdamW}},
$$

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

### 2.3 External-ready success

External-ready success 要求：

```text
Functional advantage success
+
controls pass
+
strong baseline challenge pass
+
noise robustness pass
+
system broad gate pass
```

其中：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

强系统目标是：

$$
StepRatio_{q90}\leq1.20.
$$

---

## 3. 核心假设

### H1：A7c 可控但机制不一定最适合 functional causality

A7c 是 best route candidate，因为它 P4 和 base qualification 最好。但它的 actuator 是 `BasisEntropy-ValueOnly`，它可能更适合 basis usage / representation balance，而不一定最适合 CE tail、margin tail 或 KMNIST hard-mode correction。

H1 成立标准：

若 A4b/A4d/A4e 在 paired replay 中优于 A7c，且保持 P4/P5 qualification，则说明 bounded rational actuator 更适合 causal functional direction。

### H2：当前 RealFunctional 失败可能来自 target-solver mismatch，而不是 actuator 本身

A7c target fit R2 = 1.0，说明 LS target fit 很强。但 paired replay 失败说明 “fit target” 不等于 “改善 future branch”。  
H2 假设：当前 solver 过度拟合 immediate target，没有直接优化 paired replay horizon 5/20 的机制收益。

H2 成立标准：

若引入 horizon-aware solver 或 trust-region solver 后，paired replay 能击败 controls，则 H2 成立。

### H3：RealFunctional 可能被 control 中的 stochastic regularization 击败

Best control CEp99 和 margin 都优于 RealFunctional。说明 random/shuffled/control 可能在当前 metric 上产生类似 regularization。  
H3 需要拆分 controls：

```text
NoOpMatchedOverhead
RandomMatchedNorm
ShuffledTarget
ShuffledSNR
AdamWParallel
ActuatorNoAbsorbControl
AbsorbRandomTarget
```

若 RandomMatchedNorm 始终 best，则说明 RealFunctional 的 target/direction 没有比随机方向更有信息。  
若 NoOp 始终 best，则说明 functional update 本身可能扰动了 AdamW trajectory。  
若 AdamWParallel best，则说明 extra step size / parallel AdamW 更有效，不需要 functional geometry。

### H4：Functional 应优先在 KMNIST hard rows / CE-tail / margin-tail 上证明价值

当前 LQ/A7c base 是 near-pass，不是 full-pass；主要缺口来自 hard-mode stability。Functional 价值应优先表现为：

```text
KMNIST miss recovery
CEp99 reduction
margin_p10 improvement
wrong confidence reduction
ECE/NLL improvement
curvature/local-Lipschitz improvement
```

而不是只在 MNIST easy regime 上小幅波动。

H4 成立标准：

$$
\Delta CEp99_{\text{KMNIST,real}}
<
\Delta CEp99_{\text{best-control}},
$$

或：

$$
\Delta MarginP10_{\text{KMNIST,real}}
>
\Delta MarginP10_{\text{best-control}}.
$$

### H5：如果所有 P4-qualified actuators 都 control-equivalent，则当前 functional target family 不足

若 A4/A7c 全部 paired replay fail，且 controls consistently better，则不能再继续调 gate；应返回 target mechanism 或 basis factory，而不是继续 functional full run。

H5 成立标准：

如果：

```text
A4b/A4d/A4e/A7c all paired replay fail
and best controls beat RealFunctional on CEp99/margin/curvature
```

则 route 写为：

```text
R5-ControlResistantCausalityFail
```

并停止 full re-entry。

---

## 4. Candidate 设计

### 4.1 Base references

```text
B0-MLP-match:
  official same-parameter MLP baseline。

QF-QuadraticFeatureMLP:
  strong diagnostic baseline。

LQ0-LQ-t2-h256:
  current FC-PureKAN near-pass base。

LQ1-LQ-t2-h256-fanin-output-scale:
  previous best repair reference。
```

### 4.2 P4-qualified actuator candidates

```text
A4b-BoundedRational-BranchlessDerivative:
  bounded rational family，branchless derivative。

A4d-BoundedRational-ValueOnlyActuator:
  value-only bounded rational actuator。

A4e-BoundedRational-FusedCoeffGrad:
  fused coeffgrad bounded rational actuator。

A7c-BasisEntropy-ValueOnly:
  current best route candidate。
```

Optional secondary candidates only if P4/P5 pass：

```text
A5c-PiecewiseLinear2-FixedKnots
A7b-BasisEntropy-LowRank
```

### 4.3 Output targets

```text
O1-HardTailLogitCorrection:
  CEp99 / wrong-confidence tail。

O2-MarginTailExpansion:
  margin_p10 low samples。

O3-CalibrationTailCompression:
  wrong confidence compression。

O4-CurvatureOutputFlattening:
  local curvature / local Lipschitz reduction。

O5-BasisEntropyOutputCorrection:
  basis usage entropy correction。

O6-KMNISTHardModeOutputTarget:
  KMNIST hard-mode correction。
```

### 4.4 Solvers

```text
SOL0-ProjectedGradient:
  simple J^T target + task projection。

SOL1-LeastSquaresSketch:
  LS target fit。

SOL2-ConstrainedLeastSquares:
  LS + holdout non-increase.

SOL3-TrustRegionQP:
  ||delta|| <= rho ||AdamW step||。

SOL4-HorizonAwareReplaySolver:
  optimize predicted horizon 5/20 mechanism gain using replay-calibrated scoring。

SOL5-ControlContrastiveSolver:
  choose update that maximizes predicted metric gap vs Random/NoOp/AdamWParallel controls.

SOL6-AbstainingSolver:
  skip event if predicted advantage over controls is low.
```

### 4.5 Event controllers

```text
E1-CalibratedCEp99Tail
E2-CalibratedMarginTail
E3-CalibratedWrongConfidence
E4-CalibratedCurvatureSpike
E5-CalibratedBasisEntropyCollapse
E6-CompositeSparseEvent
E7-KMNISTHardModeEvent
```

Target coverage:

$$
0.03\leq Coverage\leq0.15.
$$

### 4.6 Controls

```text
C0-AdamWOnly
C1-NoOpMatchedOverhead
C2-RandomMatchedNorm
C3-ShuffledTarget
C4-ShuffledSNRMask
C5-AdamWParallelDirection
C6-ActuatorNoAbsorbControl
C7-AbsorbRandomTarget
C8-InvertedEventController
C9-InvertedTargetSign
```

---

## 5. 实验阶段

## P0：v9.2.14 boundary reproduction

### 目标

复现 v9.2.14 的新边界：P4-qualified actuator 成立，但 paired replay causality 失败。

### 必须记录

```text
source_route
best_actuator_candidate
A7c_forward_q90
A7c_backward_q90
A7c_step_q90
A7c_memory
A7c_target_fit_R2
A7c_rz
A7c_p5_near_pass_count
A7c_macro_delta
paired_replay_real_CEp99_mean_delta
paired_replay_best_control_CEp99_mean_delta
paired_replay_real_margin_mean_delta
paired_replay_best_control_margin_mean_delta
fake_proxy_count
```

### 判断标准

P0 pass：

```text
source_route = R4-ActuatorP4ClosedBaseQualified
A7c P4 pass = 1
A7c base near-pass = 1
paired replay real does not beat best control
fake/proxy = 0
```

### 可视化

```text
p0_v9214_boundary_dashboard.svg
p0_a7c_p4_and_base_summary.svg
p0_paired_replay_real_vs_control.svg
```

---

## P1：P5 paired replay failure autopsy

### 目标

拆清 P5 failure 是 actuator-specific、target-specific、solver-specific、event-specific、scale-specific，还是 control-specific。

### 必须记录

```text
actuator_candidate
output_target
solver
event_type
branch
dataset
seed
horizon
CEp99_delta
margin_p10_delta
curvature_delta
ECE_delta
NLL_delta
accuracy_delta
logit_displacement_norm
target_fit_R2
rz
delta_norm
delta_norm_vs_adamw
cos_delta_adamw
cos_delta_random
cos_delta_control
holdout_delta
bad_event
event_coverage
```

### 判断标准

P1 必须归入至少一种机制：

```text
M1-actuator_specific_failure:
  A7c fails, A4 family better.

M2-target_specific_failure:
  CE-tail target fails, margin/calibration target better.

M3-solver_specific_failure:
  LS fits target but horizon-aware/control-contrastive solver improves.

M4-control_dominance:
  Random/NoOp/AdamWParallel consistently better.

M5-scale_failure:
  rz too large or too small; trust-region bracket changes outcome.

M6-event_failure:
  event accepts low-value updates despite sparse coverage.

M7-mechanism_metric_mismatch:
  target improves direct oracle metric but paired replay metric does not.
```

### 可视化

```text
p1_failure_factor_matrix.svg
p1_actuator_target_solver_heatmap.svg
p1_controls_rank_by_metric.svg
p1_logit_displacement_vs_mechanism_gain.svg
p1_cosine_with_controls.svg
```

---

## P2：All P4-qualified actuator causality matrix

### 目标

不要只看 A7c route best。系统性比较 A4b/A4d/A4e/A7c 的 paired replay 因果表现。

### 设置

```text
actuators = A4b,A4d,A4e,A7c
targets = O1,O2,O3,O4,O6
solvers = SOL1,SOL2,SOL3
events = E1,E2,E6,E7
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
horizons = 1,5,20,80
branches = AdamW,RealFunctional,NoOp,Random,ShuffledTarget,AdamWParallel
```

### 必须记录

```text
actuator
target
solver
event
dataset
seed
horizon
branch
CEp99_delta
margin_p10_delta
curvature_delta
ECE_delta
NLL_delta
acc_delta
control_rank
real_beats_best_control
P4_forward_q90
P4_backward_q90
P4_step_q90
P4_memory
target_fit_R2
rz
```

### 判断标准

Causality survivor：

RealFunctional beats best control in at least one mechanism metric and remains task-safe:

$$
Acc_{\text{real}}\geq Acc_{\text{AdamW}}-0.005.
$$

and at least one:

$$
CEp99_{\text{real}}\leq CEp99_{\text{best-control}}-0.05|CEp99_{\text{AdamW}}|,
$$

$$
MarginP10_{\text{real}}\geq MarginP10_{\text{best-control}}+0.02,
$$

$$
Curvature_{\text{real}}\leq0.90Curvature_{\text{best-control}},
$$

$$
ECE_{\text{real}}\leq ECE_{\text{best-control}}-0.02|ECE_{\text{AdamW}}|.
$$

Promotion requires:

```text
real_beats_best_control = 1
P4 pass = 1
base near-pass = 1
```

### 可视化

```text
p2_actuator_causality_pareto.svg
p2_real_vs_best_control_by_actuator.svg
p2_horizon_effect_by_actuator.svg
p2_kmnist_causality_matrix.svg
```

---

## P3：Control-contrastive target and solver redesign

### 目标

如果 P2 仍然 control-equivalent，则 target/solver 不能只拟合 output target，要直接优化相对 controls 的机制差异。

### 方法

构造 control-contrastive score：

$$
S_{\text{cc}}
=
w_1 \Delta CEp99_{\text{vs best control}}
+
w_2 \Delta MarginP10_{\text{vs best control}}
+
w_3 \Delta Curvature_{\text{vs best control}}
+
w_4 \Delta ECE_{\text{vs best control}}
-
w_5 BadEventPenalty.
$$

候选 solver：

```text
SOL4-HorizonAwareReplaySolver
SOL5-ControlContrastiveSolver
SOL6-AbstainingSolver
```

候选 target：

```text
O7-ControlResistantTailTarget:
  selected to beat Random/NoOp on CE-tail.

O8-ControlResistantMarginTarget:
  selected to beat controls on margin tail.

O9-ControlResistantCalibrationTarget:
  selected to beat controls on wrong confidence / ECE.

O10-KMNISTControlResistantTarget:
  selected to beat controls on KMNIST hard rows.
```

### 必须记录

```text
solver
target
control_contrastive_score
predicted_score
actual_paired_score
bad_event_rate
coverage
CEp99_vs_best_control
margin_vs_best_control
curvature_vs_best_control
ECE_vs_best_control
task_safety
```

### 判断标准

Control-contrastive pass：

$$
S_{\text{cc}}>0,
$$

$$
BadEventRate\leq0.05,
$$

$$
Coverage\in[0.03,0.15].
$$

Prediction pass：

$$
Corr(S_{\text{pred}},S_{\text{actual}})\geq0.30.
$$

### 可视化

```text
p3_control_contrastive_score.svg
p3_predicted_vs_actual_cc_score.svg
p3_target_solver_cc_heatmap.svg
p3_abstention_precision_curve.svg
```

---

## P4：Short-run causal validation

### 目标

只对 P2/P3 survivors 做 50/240-step short-run，验证 paired replay 的 causality 是否能转成连续训练收益。

### 设置

```text
steps = 50,240
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
functional_candidates = top P2/P3 survivors
controls = AdamW,NoOp,Random,ShuffledTarget,AdamWParallel
```

### 必须记录

```text
candidate
actuator
target
solver
event_controller
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
actuator_usage_entropy
event_count
event_coverage
bad_event_rate
step_ratio_q90
memory_ratio
```

### 判断标准

Short-run task safety：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005.
$$

Short-run mechanism pass：

至少一个：

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

Control pass：

Functional must beat best control on at least one mechanism metric.

System pass：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p4_short_run_task_mechanism_pareto.svg
p4_short_run_controls.svg
p4_event_timeline.svg
p4_ce_margin_curvature_panel.svg
p4_step_ratio_distribution.svg
```

---

## P5：Full 10-seed functional re-entry

### 目标

只有 P4 pass 后打开。验证 functional 是否能在完整训练中带来 task-safe advantage。

### 设置

```text
epochs = 20
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
baseline = MLP-match
diagnostic baseline = QuadraticFeatureMLP
base = P4 survivor actuator
controls = AdamW,NoOp,Random,ShuffledTarget,AdamWParallel
```

### 必须记录

```text
candidate
actuator
target
solver
dataset
seed
val_acc
test_acc
delta_vs_MLP
delta_vs_AdamW_base
delta_vs_QuadraticFeatureMLP
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
actuator_usage_entropy
event_count
event_coverage
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

KMNIST repair：

$$
\Delta Acc_{\text{KMNIST,functional}}
-
\Delta Acc_{\text{KMNIST,AdamW}}
\geq0.005.
$$

Mechanism pass：

至少一个：

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

System pass：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p5_macro_delta_vs_adamw.svg
p5_kmnist_repair_matrix.svg
p5_seedwise_win_matrix.svg
p5_task_geometry_pareto.svg
p5_ce_tail_margin_panel.svg
p5_ece_nll_panel.svg
p5_controls_comparison.svg
```

---

## P6：Noise / robustness / signal separation

### 目标

验证 functional advantage 不是 clean setting 偶然有效，而是符合 signal/noise 分离预期。

### 设置

```text
label_noise = 0.05,0.10,0.20
input_noise = 0.05,0.10
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
```

### 必须记录

```text
noise_type
noise_level
dataset
seed
candidate
clean_acc
noisy_acc
acc_drop
CEp99
margin_p10
ECE
NLL
curvature
event_coverage
bad_event_rate
SNR_active_fraction
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

Signal separation pass：

当 label noise 增大时：

$$
BadEventRate_{\text{functional}}
\leq
BadEventRate_{\text{controls}},
$$

且低 SNR / high noise events 被更多 abstain。

### 可视化

```text
p6_noise_robustness_curve.svg
p6_noise_ce_tail.svg
p6_event_coverage_vs_noise.svg
p6_signal_noise_dashboard.svg
```

---

## P7：Strong baseline / external-ready gate

### 目标

判断是否可以进入 external fair validation。

### Baselines

```text
MLP-match
same-shape MLP
hidden-bracket MLP
QuadraticFeatureMLP
LQ/A7c AdamW-only
LQ/A7c functional
```

### 必须记录

```text
baseline_id
params
flops
forward_ratio
backward_ratio
step_ratio
memory_ratio
dataset
seed
acc
ECE
NLL
CEp99
margin_p10
curvature
local_lipschitz
```

### 判断标准

External-ready if：

```text
functional task-safe = 1
functional mechanism pass = 1
functional control pass = 1
functional system pass = 1
strong baseline challenge pass = 1
```

且至少一个成立：

$$
\Delta Acc_{\text{macro,functional}}\geq0,
$$

或：

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{MLP}},
$$

with no task drop.

### 可视化

```text
p7_strong_baseline_pareto.svg
p7_external_ready_scorecard.svg
p7_lq_functional_vs_quadratic_mlp.svg
```

---

## 6. Required artifacts

```text
run_manifest.json
contract_audit_v9215.csv
p0_v9214_boundary_reproduction.csv
p1_paired_replay_failure_autopsy.csv
p2_p4qualified_actuator_causality_matrix.csv
p3_control_contrastive_target_solver.csv
p4_short_run_causal_validation.csv
p5_full_functional_reentry_10seed.csv
p6_noise_robustness_signal_separation.csv
p7_strong_baseline_external_ready.csv
functional_event_trace_v9215.csv
paired_replay_branch_trace_v9215.csv
control_rank_trace_v9215.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9214_boundary_unstable
F3_paired_replay_autopsy_incomplete
F4_actuator_specific_failure
F5_target_specific_failure
F6_solver_specific_failure
F7_control_dominance
F8_scale_failure
F9_event_failure
F10_no_causality_survivor
F11_short_run_task_drop
F12_short_run_control_equivalent
F13_full_run_no_macro_or_kmnist_gain
F14_functional_system_fail
F15_quadratic_baseline_explains_gain
F16_noise_robustness_fail
F17_fake_or_proxy_violation
F18_artifact_missing
```

---

## 7. Route decision

### Route cases

```text
R1-A4FunctionalCausalityPass:
  A4 family beats controls in paired replay.

R2-A7cFunctionalCausalityPass:
  A7c beats controls in paired replay.

R3-ControlContrastiveSolverPass:
  new target/solver beats controls after P3.

R4-ShortRunFunctionalPass:
  P4 short-run shows task-safe mechanism gain.

R5-FullFunctionalAdvantage:
  full 10-seed shows macro/KMNIST/geometry/tail gain.

R6-ControlDominanceConfirmed:
  best controls consistently beat real functional.

R7-ActuatorSpecificMismatch:
  P4-qualified actuator exists but wrong actuator family chosen.

R8-TargetSolverMismatch:
  target useful but solver/horizon objective wrong.

R9-FunctionalSystemFail:
  causality exists but step/memory gate fails.

R10-ReturnToTargetOrPrimitiveDesign:
  no P4-qualified actuator/target/solver combination beats controls.

R11-ExternalReady:
  task-safe mechanism-positive functional passes strong baseline and robustness gates.
```

### route_decision.json 必须记录

```text
route
base_candidate
best_actuator_candidate
best_target
best_solver
best_event_controller
paired_replay_pass
short_run_pass
full_reentry_pass
functional_task_safe
functional_mechanism_pass
functional_control_pass
functional_system_pass
functional_kmnist_repair_pass
noise_robustness_pass
strong_baseline_pass
external_ready
primary_blocker
next_required_implementation
success_v9215_paired_replay_causality
success_v9215_short_run
success_v9215_full_functional
success_v9215_external_ready
```

---

## 8. 第一轮执行顺序

```text
Step 1:
  P0 复现 v9.2.14 boundary。

Step 2:
  P1 拆 P5 paired replay failure。
  不允许直接进入 full run。

Step 3:
  P2 对所有 P4-qualified actuators 做 causality matrix。
  重点比较 A4b/A4d/A4e 与 A7c。

Step 4:
  如果 P2 没 survivor，进入 P3 control-contrastive target/solver redesign。

Step 5:
  只有 paired replay survivor 出现，才进入 P4 short-run。

Step 6:
  P4 通过后，进入 P5 full 10-seed functional re-entry。

Step 7:
  P6 noise / robustness / signal separation。

Step 8:
  P7 strong baseline / external-ready gate。
```

---

## 9. 停止条件

### Minimum success

```text
P0 boundary reproduced
P1 failure attribution complete
at least one P4-qualified actuator beats controls in paired replay
no fake/proxy/offload/loss/teacher violation
```

### Functional advantage success

```text
Minimum success
+
short-run task-safe mechanism gain
+
full 10-seed task-safe macro/KMNIST/geometry/tail gain
```

### External-ready success

```text
Functional advantage success
+
noise robustness pass
+
strong baseline challenge pass
```

### Failure stop

```text
1. v9.2.14 boundary cannot be reproduced；
2. all P4-qualified actuators remain control-equivalent；
3. best controls dominate real functional across metrics；
4. control-contrastive solver cannot produce positive score；
5. short-run harms task；
6. full run gives no macro/KMNIST/geometry/tail gain；
7. functional breaks system gate；
8. gains are explained by QuadraticFeatureMLP；
9. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 10. 最终解释规则

### Case A：A4 beats controls

可以声明：

```text
A7c was system-best, but A4 bounded rational family is causality-best for functional update.
```

### Case B：A7c beats controls after solver redesign

可以声明：

```text
A7c remains best route; failure was solver/target mismatch, not actuator family.
```

### Case C：controls dominate all RealFunctional branches

必须声明：

```text
P4-qualified actuator exists, but current functional targets do not produce unique causal benefit.
```

### Case D：paired replay passes but full run fails

必须声明：

```text
functional causality is local but not robust across training horizon/seeds.
```

### Case E：full run improves KMNIST or macro safely

可以声明：

```text
functional update becomes a valid task-safe advantage route and can enter external fair validation.
```

---

## 11. 最终建议

v9.2.15 的一句话策略是：

$$
\boxed{
\text{现在不要再修 P4，也不要再调 SNR；要证明 RealFunctional 在 P4-qualified actuator 上能击败 controls。}
}
$$

当前最关键的问题是：

```text
1. A7c paired replay 为什么输给 best controls？
2. A4b/A4d/A4e 是否比 A7c 更适合 functional causality？
3. 当前 output target 是否只在 direct oracle 中有效，参数实现后不稳定？
4. solver 是否应该从 target-fit 改成 control-contrastive / horizon-aware？
5. functional update 的收益能否优先出现在 KMNIST、CE-tail、margin-tail？
6. 如果所有 RealFunctional 都输给 controls，是否应停止 functional gate 修补，回到 target/primitive 设计？
```
