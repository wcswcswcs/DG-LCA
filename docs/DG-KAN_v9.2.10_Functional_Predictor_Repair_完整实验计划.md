# DG-KAN v9.2.10 Functional Predictor Repair：从 SNR Gate 到 Population-Risk-Safe Functional Update 的完整实验计划

> 本计划基于 v9.2.9 `SNR-Gated Functional Update Mainline` 的真实 terminal route 制定。  
> v9.2.9 的核心结论不是 “functional update 已失败”，而是：**当前 low-cost role-level SNR gate 已经具备系统可用性，但当前 functional direction / predictor / task-safe projection 组合没有通过 one-step population-risk safety audit。**  
> 因此 v9.2.10 不应该直接打开 P3 short-run 或 P4 full functional training，也不应该继续在 optimizer / loss / teacher / sampler 上做小修。  
> v9.2.10 的目标是重构 functional update 的因果链：先让 functional predictor 能预测 holdout non-harm，再让 functional direction 与 PureKAN 几何目标对齐，最后才重新打开 short-run 与 full functional re-entry。

---

## 0. 当前状态与结论

### 0.1 v9.2.9 达到什么

v9.2.9 执行到：

```text
route = R4-SNRGateNotUseful
base_candidate = LQ-t2-h256
success_v929_p1_snr_gate = true
success_v929_functional_opened = false
success_v929_functional_advantage = false
success_v929_external_ready = false
```

v9.2.9 中，P0 functional diagnostic open gate 已经通过。也就是说，当前 base `LQ-t2-h256` 仍满足：

```text
PureKAN equivalence = pass
P4 system gate = pass
P5 robust near-pass = pass
```

这说明 base candidate 本身已经满足 functional diagnostic 的前置条件。

P1 第一波 ghost-batch SNR 失败：

```text
mean overhead ratio = 4.061492
max overhead ratio = 5.726611
```

这说明完整 microbatch / per-gradient SNR 太贵，不能作为在线 update gate。

随后实现低开销 `ema_role_scalar` estimator 后，P1 calibrated gate 通过：

```text
eligible rows = 270
mean amortized overhead = 0.125983
max amortized overhead = 0.130215
active fraction range = 0.051099 - 0.298428
max role CV = 0.206763
```

这说明：**SNR estimator overhead 已经不是主 blocker。**

P2 one-step population-risk audit 已真实打开，但失败：

```text
p2_prediction_corr = 0.231090 < 0.30
p2_bad_step_rate = 0.819444 > 0.05
p2_holdout_nonharm_rate = 0.180556 < 0.70
p2_snr_gate_usefulness_pass = 1
p2_one_step_population_risk_pass = 0
```

并且 P3 short-run、P4 full functional re-entry、P5 strong baseline、P6 robustness 全部保持 `not_run`。

### 0.2 独立判断

v9.2.9 没有达到 functional update advantage。更准确的说法是：

$$
\boxed{
\text{SNR instrumentation 已经可用，但当前 functional update 的 predictor 和 direction 不安全。}
}
$$

这不是系统实现 blocker。P1 role-level scalar EMA 的 amortized overhead 约 `0.126`，低于计划上限 `0.20`。这也不是 LQ base blocker，LQ base 已经 P4 pass + P5 robust near-pass。它也不是传统 optimizer blocker，因为 AdamW-only base 已经 near-pass。

当前真正 blocker 是：

$$
\boxed{
\text{当前 functional direction 与 SNR gate 不能可靠预测 holdout non-harm。}
}
$$

P2 中 SNR-gated functional 比 no-SNR geometry control 略好，说明 SNR 不是完全无信息；但 bad-step rate 高达 `0.819444`，holdout non-harm 只有 `0.180556`，说明这个信号远不足以打开真实 functional training。更重要的是，NoOp control 为 `0` delta，而 Random matched-norm control 的 bad-step rate低于当前 geometry direction，这提示当前 quadratic coeff damping direction 本身可能不是合适的 population-risk direction。

### 0.3 当前不能做什么

本轮不能做：

```text
直接打开 P3 short-run functional training；
直接打开 P4 full functional re-entry；
把 functional 当作 AdamW full-pass 的替代；
把 SNR-gated direction 的微弱 improvement 写成 advantage；
继续增大 functional strength；
继续调整 loss、label smoothing、teacher、sampler、class weight；
把 role-level scalar SNR 说成 per-parameter SNR；
提前研究 PureKANConv / PureKANFormer。
```

---

## 1. v9.2.10 总体目标

v9.2.10 的总体目标是：

$$
\boxed{
\text{修复 functional predictor / direction / gate，使 functional update 具备 one-step population-risk safety。}
}
$$

这不是追单点 accuracy 的实验，而是建立 functional update 的因果资格。

v9.2.10 要回答五个问题。

### Q1：当前 P2 失败到底来自哪里

P2 失败可能来自四个组件：

```text
1. SNR gate 太粗，只是 role-level scalar，不能定位参数方向；
2. functional direction 本身错误，即 quadratic coeff damping 不对应 population-risk-safe direction；
3. task-safe projection 只保证 train microbatch 一阶 CE，不保证 holdout；
4. functional step fraction = 0.10 * AdamW step norm 过大或事件触发不合适。
```

v9.2.10 必须把这四个因素拆开，而不是直接换一个阈值。

### Q2：functional predictor 能否预测 one-step holdout delta

最低要求：

$$
Corr(\widehat{\Delta R}, \Delta L_{\text{holdout}})\geq0.30.
$$

更强要求：

$$
Corr(\widehat{\Delta R}, \Delta L_{\text{holdout}})\geq0.50.
$$

其中 $\widehat{\Delta R}$ 是 functional predictor 预测的 population-risk / holdout change，$\Delta L_{\text{holdout}}$ 是真实 one-step holdout CE change。

### Q3：functional update 能否做到 high-precision abstention

Functional update 不需要频繁触发；它应该只在高置信事件触发。我们追求的是 precision first：

$$
BadStepRate\leq0.05,
$$

$$
HoldoutNonharmFraction\geq0.70.
$$

如果 coverage 很低但 precision 高，仍然可以进入 short-run。因为 functional update 的目标是少量高质量几何校正，不是每步替代 AdamW。

### Q4：functional direction 是否真的改善几何 / margin / calibration

通过 P2 safety 后，还必须改善至少一个机制指标：

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}},
$$

或：

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamW}},
$$

或：

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamW}},
$$

或：

$$
ECE_{\text{functional}}\leq ECE_{\text{AdamW}}.
$$

### Q5：functional update 是否能经受 controls

Functional 必须优于：

```text
NoOpMatchedOverhead
RandomDirectionMatchedNorm
ShuffledSNRMask
InvertedSNRMask
GeometryOnlyNoSNR
```

如果不能赢 controls，就不能声明 causality。

---

## 2. 和论文的关系：参考但不照搬

论文 **A Theory of Generalization in Deep Learning** 给出的重要启发是：训练中的 update direction 应该区分 coherent signal 和 minibatch noise。其核心 practical gate 可以写成：

$$
\mu_k^2>\frac{\sigma_k^2}{b-1}.
$$

这对我们有三条启发。

第一，functional update 应是 update rule，而不是 loss。我们继续保持：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

Functional 更新写成：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{\text{AdamW-equivalent}}
+
\lambda_f\Delta\theta_{\text{functional-safe}}.
$$

第二，SNR 只能决定哪些方向有资格更新，不应单独定义 functional direction。Functional direction 仍要来自 DG-KAN 的函数空间 / 几何机制：

$$
\Delta\theta_{\text{functional-safe}}
=
\Pi_{\text{task-safe}}
\left(
q_{\text{SNR}}
\odot
d_{\text{geometry}}
\right).
$$

第三，我们要比论文做得更具体，因为我们有 PureKAN role structure。论文中的 gate 是 architecture-agnostic；我们必须做：

```text
role-aware SNR
basis-channel-aware SNR
direction-specific SNR
event-level abstention
NoOp / Random / ShuffledSNR controls
KMNIST miss-row mechanism validation
```

---

## 3. 硬约束

### 3.1 Clean training contract

所有实验必须满足：

```text
loss_type = CE
label_smoothing = 0
external_teacher_used = 0
self_teacher_used = 0
teacher_logits_used = 0
distillation_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
uses_loss_backward = 0
fake_data_used = 0
proxy_row_used = 0
```

训练目标保持：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

不允许：

$$
L=CE+\lambda L_{\text{geo}}.
$$

### 3.2 Functional update contract

Functional update 是 update rule：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{\text{AdamW}}
+
\lambda_f\Delta\theta_{\text{functional-safe}}.
$$

其中：

$$
\Delta\theta_{\text{functional-safe}}
=
q\odot\Pi_{\text{safe}}(d).
$$

这里 $d$ 是 functional direction，$q$ 是 gate，$\Pi_{\text{safe}}$ 是 task-safety projection 或 holdout guard。  
Functional update 不能新增非 edge-owned parameters。

### 3.3 Base candidate contract

Base 固定为：

```text
LQ-t2-h256
LQ-t2-h256-fanin-output-scale
```

不打开新的大架构，不打开 PureKANConv，不打开 PureKANFormer。

### 3.4 P2 gating contract

P2 不过，不得打开：

```text
P3 short-run functional safety
P4 full functional re-entry
P5 strong baseline challenge
P6 robustness
external fair validation
```

---

## 4. 核心假设

### H1：当前 P2 失败主要来自 functional direction，而不是 SNR estimator

v9.2.9 已证明 low-cost `ema_role_scalar` P1 通过，且 SNR-gated functional 比 no-SNR geometry control略好。因此，SNR estimator 有弱信号，但当前 direction 的 holdout safety 很差。

H1 成立标准：

如果换 direction 后，使用同一个 role-level SNR gate 能满足：

$$
BadStepRate\leq0.05,
$$

$$
HoldoutNonharmFraction\geq0.70,
$$

则 H1 成立。

若所有 direction 都失败，则说明 role-level SNR gate 或 predictor 本身不足。

### H2：role-level scalar SNR 太粗，只适合事件筛选，不适合参数方向筛选

当前 `ema_role_scalar` 只维护 role-level scalar，不测 per-parameter histogram。它可能只能判断“是否触发事件”，无法判断“更新哪些参数”。因此需要两级 gate：

```text
event-level SNR gate:
  用 role scalar 判断是否允许功能性事件发生；

within-event direction gate:
  用 direction-specific score 或 low-cost subspace score 判断更新哪些子方向。
```

H2 成立标准：

两级 gate 比单级 role scalar gate 满足：

$$
BadStepRate_{\text{two-level}}\leq BadStepRate_{\text{role-only}},
$$

并且：

$$
HoldoutNonharm_{\text{two-level}}\geq HoldoutNonharm_{\text{role-only}}.
$$

### H3：train microbatch projection 不足，需要 exchangeability holdout guard

当前 task-safe projection 只保证 train microbatch 上的一阶 CE 方向，不保证 holdout。P2 正是因此失败。

H3 成立标准：

引入 holdout precheck / abstention 后：

$$
BadStepRate\leq0.05,
$$

$$
HoldoutNonharmFraction\geq0.70.
$$

同时 overhead 不超过：

$$
T_{\text{functional-event}}\leq0.30T_{\text{step}}.
$$

amortized overhead 不超过：

$$
T_{\text{functional-amortized}}\leq0.20T_{\text{step}}.
$$

### H4：functional step fraction 需要高精度小步，而不是默认 $0.10$ AdamW step norm

v9.2.9 使用：

$$
\|\delta_f\|=0.10\|\delta_{\text{AdamW}}\|.
$$

如果 direction 本身 noisy，这可能过大。H4 要求用预注册 scale bracket 判定是否是 magnitude issue，而不是盲调。

候选：

$$
\rho \in \{0.01,0.03,0.10\}.
$$

H4 成立标准：

如果 $\rho=0.01$ 或 $0.03$ 显著降低 bad-step rate，但保持机制改善，则当前 blocker 部分来自 step magnitude。

### H5：Functional benefit 应集中在 margin / CE-tail / curvature events，而不是全局均匀事件

如果 functional direction 只在 CE tail / margin tail / curvature spike 时有用，全局触发会导致大部分 bad step。

H5 成立标准：

event-gated functional 满足：

$$
BadStepRate_{\text{event}}\leq BadStepRate_{\text{uniform}},
$$

$$
CEp99_{\text{event}}<CEp99_{\text{AdamW}},
$$

或：

$$
MarginP10_{\text{event}}>MarginP10_{\text{AdamW}}.
$$

### H6：真正的 functional advantage 应优先在 KMNIST miss rows 上出现

v9.2.7 的 miss rows 全部来自 KMNIST seeds `0/3/4/5/7/9`。Functional update 的价值不一定是提高 MNIST，而是改善 harder modes。

H6 成立标准：

$$
\Delta Acc_{\text{KMNIST,functional}}
-
\Delta Acc_{\text{KMNIST,AdamW}}
\geq0.005.
$$

---

## 5. Candidate 设计

### 5.1 Base candidates

```text
A0-LQ0-AdamW:
  LQ-t2-h256 AdamW-only reference。

A1-LQ1-FaninScale-AdamW:
  v9.2.7 best repair reference。
```

### 5.2 Current reference candidates

```text
F0-AdamWOnly:
  no functional update。

F1-SNRGatedTaskProjected-current:
  v9.2.9 current failed P2 reference。

F2-NoSNRGeometry-current:
  geometry direction without SNR gate。

F3-NoOpMatchedOverhead:
  no parameter change, same audit overhead。

F4-RandomDirectionMatchedNorm:
  random direction with matched norm and role allocation。
```

### 5.3 Direction repair candidates

```text
D1-QuadraticCoeffDamping:
  current direction, kept as reference。

D2-QuadraticCoeffCurvatureSignCorrected:
  direction sign chosen by one-step curvature/CE agreement audit。

D3-LiftHighCurvatureCorrection:
  acts on lift parameters, not only quadratic coefficients。

D4-BasisUsageEntropyCorrection:
  adjusts quadratic channel to increase basis usage entropy, only if task-safe.

D5-CEtailGeometryCorrection:
  triggered by CEp99 / margin_p10 events; uses geometry direction but only on tail event.

D6-LocalLipschitzCorrection:
  finite-difference / Hutchinson proxy direction that reduces local Lipschitz.

D7-AdamWResidualFunctional:
  functional direction defined as residual between role-preconditioned AdamW and standard AdamW; diagnostic only.

D8-FisherDiagGeometryDirection:
  geometry direction scaled by diagonal Fisher / gradient variance proxy.

D9-SignalChannelProjection:
  projects geometry direction onto recent coherent gradient subspace.
```

### 5.4 Gate repair candidates

```text
G1-RoleSNROnly:
  current role scalar SNR.

G2-HighConfidenceRoleSNR:
  higher threshold, lower coverage, precision-first.

G3-TwoLevelRolePlusSubspaceGate:
  role-level SNR event gate + subspace projection gate.

G4-HoldoutPrecheckGate:
  accepts functional event only if exchangeability holdout CE does not increase.

G5-PredictedDeltaGate:
  uses calibrated predictor of holdout delta; abstains if confidence low.

G6-ShuffledSNRControl:
  same active fraction, shuffled masks.

G7-InvertedSNRControl:
  updates low-SNR roles; should fail if SNR meaningful.
```

### 5.5 Magnitude / event candidates

```text
M1-stepfrac-0.01
M2-stepfrac-0.03
M3-stepfrac-0.10-current

E1-uniform-stride12
E2-curvature-event
E3-margin-tail-event
E4-CEp99-event
E5-SNR-high-confidence-event
```

---

## 6. 实验阶段

## P0：v9.2.9 reproduction and gate audit

### 目标

复现 v9.2.9 P2 terminal failure，确保不是测量噪声或 runner issue。

### 设置

```text
base = LQ0,LQ1
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
P2 measured steps = 8
functional step fraction = 0.10
SNR estimator = ema_role_scalar
```

### 必须记录

```text
candidate_id
dataset
seed
p2_prediction_corr
bad_step_rate
holdout_nonharm_fraction
mean_holdout_delta
median_holdout_delta
CEp99_before
CEp99_after
margin_p10_before
margin_p10_after
SNR_active_fraction
functional_step_norm
projection_removed_norm
```

### 判断标准

Repeat pass：

$$
|bad\_step\_rate-0.819444|\leq0.10.
$$

$$
|p2\_prediction\_corr-0.231090|\leq0.10.
$$

If repeat unstable，先修 measurement，不进入 P1。

### 可视化

```text
p0_p2_repeat_dashboard.svg
p0_holdout_delta_distribution.svg
p0_bad_step_by_dataset.svg
```

---

## P1：P2 failure attribution

### 目标

把 P2 failure 拆成 direction、gate、projection、magnitude、dataset 五个维度。

### 必须记录

```text
functional_direction
gate_type
step_fraction
event_type
dataset
seed
control_type
bad_step_rate
holdout_nonharm_fraction
prediction_corr
mean_holdout_delta
median_holdout_delta
positive_delta_fraction
task_grad_dot_functional
projection_removed_norm
functional_norm
SNR_active_fraction
SNR_mean_by_role
CEp99_delta
margin_p10_delta
curvature_delta
```

### 判断标准

Attribution pass 要求明确归因：

```text
direction_dominant_failure
gate_dominant_failure
projection_failure
magnitude_failure
dataset_specific_failure
```

如果无法归因，不进入 direction factory，只先修 P1 instrumentation。

### 可视化

```text
p1_failure_factor_heatmap.svg
p1_direction_vs_gate_matrix.svg
p1_step_fraction_bad_step_curve.svg
p1_dataset_failure_breakdown.svg
p1_projection_removed_vs_holdout_delta.svg
```

---

## P2：direction factory one-step audit

### 目标

找出至少一个 functional direction，在不改变 loss/teacher/sampler 的前提下通过 one-step safety。

### 候选

```text
D1-D9 x G1/G2/G4 x M1/M2/M3
```

为了控制组合爆炸，先做 two-stage selection：

```text
Stage A:
  D1-D9 with fixed conservative G2 + M1.

Stage B:
  top 3 directions with G1/G2/G3/G4/G5 and M1/M2/M3.
```

### 必须记录

```text
direction_id
gate_id
step_fraction
dataset
seed
bad_step_rate
holdout_nonharm_fraction
prediction_corr
mean_holdout_delta
CEp99_delta
margin_p10_delta
curvature_delta
local_lipschitz_delta
basis_usage_entropy_delta
functional_event_coverage
amortized_overhead_ratio
```

### 判断标准

P2 direction survivor：

$$
BadStepRate\leq0.05,
$$

$$
HoldoutNonharmFraction\geq0.70,
$$

$$
FunctionalEventCoverage\geq0.05.
$$

Mechanism benefit:

At least one:

$$
CEp99_{\text{after}}<CEp99_{\text{before}},
$$

$$
MarginP10_{\text{after}}>MarginP10_{\text{before}},
$$

$$
Curvature_{\text{after}}<Curvature_{\text{before}}.
$$

System gate：

$$
AmortizedOverhead\leq0.20.
$$

### 可视化

```text
p2_direction_factory_pareto.svg
p2_safety_vs_coverage.svg
p2_mechanism_delta_by_direction.svg
p2_direction_control_comparison.svg
```

---

## P3：calibrated functional predictor

### 目标

训练或校准一个不使用 validation/test 的 online predictor，用于决定是否执行 functional update。它只能使用 training batch / exchangeability holdout audit 中产生的 features。

### Features

```text
SNR_mean_by_role
SNR_active_fraction_by_role
task_grad_dot_functional
projection_removed_norm
functional_norm_ratio
CEp99
margin_p10
curvature_proxy
event_type
dataset_id
step_index
```

### Predictor 类型

```text
PRED0-threshold_rule:
  hand-coded threshold rule。

PRED1-logistic_calibrator:
  logistic calibration on previous audit events only。

PRED2-isotonic_calibrator:
  monotone calibrator for predicted non-harm probability。

PRED3-abstaining_rule:
  threshold + uncertainty abstention。
```

### 必须记录

```text
predictor_id
training_events
evaluation_events
prediction_corr
AUROC_nonharm
precision_at_coverage_05
precision_at_coverage_10
coverage
bad_step_rate
holdout_nonharm_fraction
calibration_ECE_event
```

### 判断标准

Predictor pass：

$$
AUROC_{\text{nonharm}}\geq0.70.
$$

Precision-first gate：

$$
Precision_{\text{nonharm}}@Coverage\geq0.10 \geq0.90.
$$

or：

$$
BadStepRate\leq0.05
$$

at:

$$
Coverage\geq0.05.
$$

### 可视化

```text
p3_predictor_roc.svg
p3_precision_coverage_curve.svg
p3_calibration_curve.svg
p3_abstention_dashboard.svg
```

---

## P4：short-run functional safety

### 目标

只对 P2/P3 survivor 做 50/240-step short-run，验证 one-step safety 是否能转化为多步安全。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
steps = 50,240
base = LQ0,LQ1
functional candidates = top 2 from P2/P3
controls = NoOp, Random, ShuffledSNR, InvertedSNR
```

### 必须记录

```text
candidate_id
dataset
seed
step
train_loss
holdout_loss
val_acc_proxy
CE_p99
margin_p10
ECE_proxy
NLL_proxy
curvature
local_lipschitz
basis_usage_entropy
functional_event_count
functional_event_coverage
bad_event_rate
step_ratio
memory_ratio
```

### 判断标准

Short-run task safety：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005.
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
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

Control pass：

Functional must beat NoOp and Random on at least one mechanism metric without task harm.

### 可视化

```text
p4_short_run_loss_curve.svg
p4_short_run_mechanism_bar.svg
p4_short_run_event_timeline.svg
p4_control_matrix.svg
```

---

## P5：full 10-seed functional re-entry

### 目标

完整验证 functional update 是否能在 real task setting 中带来 task-safe geometry/generalization advantage。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
epochs = 20
base = LQ0/LQ1
functional candidates = P4 survivor
controls = NoOpMatchedOverhead, RandomDirectionMatchedNorm, ShuffledSNRMask
baseline = MLP-match
diagnostic baseline = QuadraticFeatureMLP
```

### 必须记录

```text
candidate_id
functional_mode
dataset
seed
val_acc
test_acc
train_acc
val_loss
test_loss
delta_vs_MLP_match
delta_vs_AdamW_LQ
delta_vs_QuadraticFeatureMLP
near_pass
full_pass
CE_p50
CE_p90
CE_p99
margin_p10
wrong_confidence_p95
ECE
NLL
curvature_ratio
jacobian_norm_ratio
local_lipschitz_ratio
basis_usage_entropy
lift_condition_number
functional_event_count
event_coverage
bad_event_rate
step_ratio
memory_ratio
functional_update_time_ratio
```

### 判断标准

Task safety：

$$
\Delta Acc_{\text{functional-vs-AdamW}}\geq-0.005.
$$

Functional macro improvement：

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

Geometry pass：

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

Calibration / tail pass：

At least one:

$$
ECE_{\text{functional}}\leq ECE_{\text{AdamW}},
$$

$$
NLL_{\text{functional}}\leq NLL_{\text{AdamW}},
$$

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamW}},
$$

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamW}}.
$$

System pass：

$$
StepRatio_{\text{functional}}\leq1.50,
$$

$$
MemoryRatio_{\text{functional}}\leq1.05.
$$

### 可视化

```text
p5_macro_delta_vs_adamw.svg
p5_kmnist_repair_matrix.svg
p5_seedwise_win_matrix.svg
p5_task_geometry_pareto.svg
p5_ce_tail_margin_panel.svg
p5_ece_nll_panel.svg
p5_event_coverage_by_dataset.svg
```

---

## P6：noise and robustness diagnostic

### 目标

验证 SNR-gated functional 是否真正 suppress noise，而不只是 one-step audit 过拟合。

### 设置

```text
label_noise = 0.05,0.10,0.20
input_noise = 0.05,0.10
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
candidates = AdamW-only, best functional, NoOp, Random
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
SNR_active_fraction
event_coverage
bad_event_rate
```

### 判断标准

Robustness useful：

$$
AccDrop_{\text{functional}}\leq AccDrop_{\text{AdamW}}.
$$

Noise-tail pass：

$$
CEp99_{\text{functional}}\leq CEp99_{\text{AdamW}}.
$$

SNR relevance：

$$
ActiveFraction_{\text{noise}=0.20}
<
ActiveFraction_{\text{noise}=0.05}.
$$

### 可视化

```text
p6_noise_robustness_curve.svg
p6_snr_active_fraction_vs_noise.svg
p6_ce_tail_noise.svg
p6_noise_task_geometry_pareto.svg
```

---

## P7：strong baseline and external-ready gate

### 目标

判断 functional advantage 是否足以进入 external fair validation。

### Baselines

```text
MLP-match
same-shape MLP
hidden-bracket MLP
QuadraticFeatureMLP
LQ AdamW-only
LQ functional
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
functional geometry pass = 1
functional control pass = 1
functional system pass = 1
strong baseline challenge pass = 1
```

and at least one:

$$
\Delta Acc_{\text{macro,functional}}\geq0,
$$

or:

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

## 7. Required artifacts

```text
run_manifest.json
contract_audit_v9210.csv
p0_v929_p2_reproduction.csv
p1_p2_failure_attribution.csv
p2_direction_factory_one_step.csv
p3_calibrated_functional_predictor.csv
p4_short_run_functional_safety.csv
p5_full_functional_reentry_10seed.csv
p6_noise_robustness_diagnostic.csv
p7_strong_baseline_external_ready.csv
functional_event_trace_v9210.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy:

```text
F1_contract_violation
F2_base_gate_regression
F3_p2_reproduction_unstable
F4_direction_failure
F5_gate_failure
F6_projection_failure
F7_magnitude_failure
F8_predictor_calibration_fail
F9_one_step_safety_fail
F10_short_run_task_drop
F11_control_causality_fail
F12_full_functional_no_gain
F13_kmnist_repair_fail
F14_system_overhead_fail
F15_noise_robustness_fail
F16_quadratic_baseline_explains_gain
F17_fake_or_proxy_violation
F18_artifact_missing
```

---

## 8. Route decision

### Route cases

```text
R1-FunctionalPredictorRepaired:
  P2 one-step safety passes with calibrated predictor and direction.

R2-FunctionalShortRunSafe:
  P4 short-run passes task-safety and mechanism gates.

R3-FunctionalFullAdvantage:
  P5 full 10-seed passes task-safe geometry/generalization advantage.

R4-FunctionalGeometryOnly:
  geometry/calibration improves but task does not improve.

R5-FunctionalUnsafe:
  bad-step or task drop remains unacceptable.

R6-SNRGateTooCoarse:
  role-level scalar SNR cannot safely gate functional update.

R7-DirectionWrong:
  all gates fail because functional direction is not population-risk aligned.

R8-HoldoutGuardRequired:
  functional only works with exchangeability holdout precheck.

R9-SystemOverheadFail:
  functional predictor/update breaks step or memory gate.

R10-ExternalFairReady:
  functional passes task/geometry/system/control and strong baseline challenge.
```

### route_decision.json 必须记录

```text
route
base_candidate
best_functional_candidate
best_direction
best_gate
best_predictor
p2_safety_pass
p4_short_run_pass
p5_full_reentry_pass
functional_task_safe
functional_geometry_pass
functional_calibration_pass
functional_kmnist_repair_pass
functional_control_pass
functional_system_pass
noise_robustness_pass
quadratic_baseline_challenge_pass
external_fair_ready
primary_blocker
next_required_implementation
success_v9210_functional_predictor
success_v9210_functional_advantage
success_v9210_external_ready
```

---

## 9. 第一轮执行顺序

```text
Step 1:
  P0 复现 v9.2.9 P2 terminal failure。
  如果复现不稳，先修 measurement。

Step 2:
  P1 做 failure attribution。
  必须分清 direction / gate / projection / magnitude / dataset 哪个是主因。

Step 3:
  P2 direction factory。
  不允许直接 full training；先找到 one-step safe direction。

Step 4:
  P3 calibrated predictor。
  建立 abstaining high-precision event gate。

Step 5:
  P4 short-run functional safety。
  只跑 P2/P3 survivors。

Step 6:
  P5 full 10-seed functional re-entry。
  只有 P4 过才跑。

Step 7:
  P6 noisy-signal robustness。
  验证 SNR gate 的真实 signal/noise 分离作用。

Step 8:
  P7 strong baseline and external-ready gate。
```

---

## 10. 停止条件

### 成功停止

Minimum success:

```text
P2 one-step safety pass
P4 short-run task-safety pass
controls pass
system overhead pass
```

Functional advantage success:

```text
Minimum success
+
P5 full 10-seed task-safe geometry/calibration/tail advantage
+
KMNIST repair or macro improvement
```

External-ready success:

```text
Functional advantage success
+
strong baseline challenge pass
+
noise robustness pass
```

### 失败停止

```text
1. v9.2.9 P2 failure cannot be reproduced；
2. all directions fail one-step safety；
3. role-level SNR remains too coarse after two-level gate；
4. holdout guard required but overhead > 0.20 amortized；
5. short-run functional harms task by > 0.005；
6. full run does not beat NoOp / Random / ShuffledSNR controls；
7. functional breaks step/memory gate；
8. gains are explained by QuadraticFeatureMLP；
9. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 11. 最终解释规则

### Case A：functional predictor repaired, but no task gain

可以声明：

```text
Functional update became one-step safe, but has not yet produced task/generalization advantage.
```

### Case B：functional improves geometry but not accuracy

必须声明：

```text
Functional update gives geometry/calibration evidence, but not task advantage.
```

### Case C：functional improves KMNIST / macro without task harm

可以声明：

```text
SNR-gated functional update gives task-safe improvement on LQ near-pass base.
```

### Case D：SNR gate not useful

必须声明：

```text
Role-level scalar SNR is insufficient for population-risk-safe functional update.
```

### Case E：direction wrong

必须声明：

```text
The current geometry direction is not population-risk aligned; need a new functional direction, not a new SNR estimator.
```

---

## 12. 最终建议

v9.2.10 的一句话策略是：

$$
\boxed{
\text{不要直接训练 functional；先把 functional predictor 和 direction 修到 one-step holdout-safe。}
}
$$

当前最关键的不是继续优化系统，也不是再调 AdamW，而是：

```text
1. 为什么当前 SNR-gated task-projected functional 有 81.9% bad-step？
2. 是 direction 错，gate 太粗，projection 不足，还是 step magnitude 过大？
3. 能否用 high-precision abstention / holdout guard 把 bad-step 降到 <= 5%？
4. 哪个 functional direction 真正改善 CE tail、margin、curvature？
5. 安全的 one-step 机制能否转化为 short-run 和 full-run advantage？
```

只有这些问题回答完，functional update 才能成为 DG-KAN 的真正核心贡献。
