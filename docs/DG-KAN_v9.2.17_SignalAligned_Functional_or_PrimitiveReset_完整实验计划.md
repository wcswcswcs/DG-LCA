# DG-KAN v9.2.17 Signal-Aligned Functional Update 与 Primitive Reset 决策完整实验计划

> 本计划基于 v9.2.16 `Causal Target Discovery 与 Primitive Redesign` 的真实 terminal route 制定。  
> v9.2.16 的结论非常严厉：**当前 P4-qualified actuator、direct oracle target、control-derived target、control-contrastive solver 全部没有产生 control-resistant functional causality。**  
> 因此 v9.2.17 不再继续手写新的 output target，也不继续调 SNR threshold、event coverage、step fraction、actuator target fit。  
> 本轮要做一个更本质的分叉判断：
>
> $$
> \boxed{
> \text{functional update 是否应从“额外 output correction”重定义为“signal-aligned optimizer metric”，否则回到 primitive / basis design？}
> }
> $$
>
> v9.2.17 的目标不是继续把旧路线补丁化，而是正式回答：
>
> ```text
> 1. AdamWParallelDirection 为什么成为最强 control？
> 2. 如果最强控制是 AdamWParallel，functional update 是否应该沿 signal channel 调整 optimizer metric？
> 3. 这种 signal-aligned functional update 是否能击败 scalar-LR / extra-AdamW / random / shuffled controls？
> 4. 如果仍然失败，是否应该停止当前 LQ/A4/A7c functional 路线，回到 primitive / basis factory？
> ```

---

## 0. 最新结果的独立判断

### 0.1 v9.2.16 是否达到目标

没有。

v9.2.16 的 terminal route 是：

```text
route = R8-NoExtractableFunctionalAdvantage
base_candidate = LQ-t2-h256
success_v9216_paired_replay_causality = false
success_v9216_short_run = false
success_v9216_full_functional = false
success_v9216_external_ready = false
```

最终 artifact：

```text
results/real_rerun_20260506/v9216_causal_target_discovery_first_20260510T040000Z/
```

v9.2.16 已经真实执行了三件关键事。

第一，v9.2.15 terminal boundary 被复现：

```text
source route = R10-ReturnToTargetOrPrimitiveDesign
p2_survivor_count = 0
p3_pass_count = 0
source row-level beat still concentrated in A4e + KMNIST + horizon80
```

第二，P2 dedicated replication 复验了 `A4e + KMNIST + horizon20/80/160`：

```text
rows = 8640
RealFunctional row-level beat = 0
a4e_kmnist_h80_signal_pass = 0
```

这说明 v9.2.15 的 `45` 个 row-level beat 不可复现，不能继续当作 functional clue。

第三，P3 / P4 也没有救回来：

```text
P3 control-derived target discovery:
  rows = 3024
  CD1-CD7 no survivor

P4 control-contrastive solver validation:
  row-level pass-like rows = 123
  prediction correlation = -0.080205
  control_contrastive_pass = 0
```

因此 P5 short-run、P6 full re-entry、P7 robustness、P8 external-ready 都正确保持：

```text
not_run
```

### 0.2 这轮真正证明了什么

v9.2.16 不是普通失败。它排除了三种此前仍可能成立的解释。

第一，**A4e/KMNIST/horizon80 稀疏信号没有复现**。  
v9.2.15 中唯一 row-level clue 已经被 dedicated replication 推翻。因此继续追 `A4e delayed hard-mode target` 没有根据。

第二，**control-derived targets 没有产生 survivor**。  
从 winning controls 中提取 target prototypes 是合理的，但本轮 CD1-CD7 没有过 paired replay。这说明不是“手写 target 不好，control target 就会好”。

第三，**control-contrastive posthoc 也失败**。  
P4 出现 123 个 row-level pass-like rows，但整体 prediction correlation 是负的：

$$
Corr(S_{\text{pred}},S_{\text{actual}})=-0.080205.
$$

这说明当前 predictor / solver 无法稳定预测 Real-vs-Control 的 causal score。

### 0.3 当前最重要的新事实

P1 control-dominance autopsy 显示：

```text
best control = AdamWParallelDirection
AdamWParallelDirection wins 18007 / 34560 metric groups
CEp99 / ECE / Margin / NLL 四个指标中它都排第一
```

这是 v9.2.16 最重要的信息。

它说明当前 RealFunctional 不是输给纯随机噪声，也不是输给 NoOp，而是输给 **沿 AdamW 方向的额外 signal-aligned movement**。这改变了问题性质。

当前不应再问：

```text
怎样写一个更复杂的 output correction target？
```

而应问：

```text
如果 AdamWParallel 是最强 control，functional update 是否应重定义为 signal-aligned update metric / preconditioner？
```

### 0.4 当前 blocker 的本质

到 v9.2.16 为止，以下前置项已经基本不是主 blocker：

```text
Clean FC-PureKAN base
P4 system gate
AdamW-only near-pass base
low-cost SNR instrumentation
P4-qualified actuator
output target oracle usefulness
actuator controllability
event coverage
paired replay infrastructure
```

真正 blocker 是：

$$
\boxed{
\text{functional update 的独立因果方向没有成立；最可靠的方向仍是 AdamW signal direction。}
}
$$

换句话说：

```text
我们能移动 output；
也能低成本移动；
也能构造 actuator；
但额外手写 functional target 没有比 AdamW signal-aligned control 更好。
```

这说明此前的 functional 设计可能把问题想反了：  
不是在 AdamW 之外找一个几何修正方向，而是应在 AdamW 的 signal channel 上学习更好的 update metric。

---

## 1. 当前进度在总计划中的位置

DG-KAN 总计划的终极目标是：

$$
\text{Next-Gen Beyond-MLP}
=
\text{Clean FullEdge PureKAN}
+
\text{Graph-Free Manual Training}
+
\text{Kernel-Native Efficiency}
+
\text{Task-Safe Functional Update}
+
\text{External Fair Advantage}
+
\text{Scalable Extension}.
$$

截至当前，状态是：

| 模块 | 状态 |
|---|---|
| Clean FC-PureKAN | LQ-t2-h256 strict equivalence 已通过 |
| Graph-free manual training | LQ path manual forward/backward/update 成立 |
| Kernel-native efficiency | LQ base 与 A7c actuator P4 已通过 |
| AdamW-only task | LQ/A7c near-pass，未 full-pass |
| Functional update | control-resistant causality 未成立 |
| External fair | 未打开 |
| PureKANConv / PureKANFormer | 继续 deferred |

因此当前已经不是系统实现阶段。  
也不是 actuator 设计阶段。  
更不是普通 optimizer sweep 阶段。

当前阶段是：

$$
\boxed{
\text{Functional causality definition / signal alignment 阶段。}
}
$$

---

## 2. 当前问题的本质

### 2.1 不是系统实现问题

v9.2.14 已找到 P4-qualified actuator。以 A7c 为例：

```text
forward q90 = 1.104638
backward q90 = 1.387102
step q90 = 1.213123
compact memory = 0.969731
target fit R2 = 1.0
rz = 1.106568
```

这些说明当前有可控、系统可比的 FC-PureKAN actuator。继续优化 kernel 不是主线。

### 2.2 不是 AdamW base 训练不了

A7c AdamW-only base qualification 是：

```text
near-pass = 8/9
macro delta = -0.004167
```

LQ-t2-h256 之前也达到 robust near-pass。当前不是 “模型训不动”，而是 “还没有显著 Beyond-MLP / functional advantage”。

### 2.3 不是 output target 不能被拟合

A7c 的 target fit R2 达到：

$$
R^2_{\text{fit}}=1.0,
$$

并且：

$$
r_z=1.106568.
$$

v9.2.13 也显示 direct logit oracle target 有 useful signal。  
所以当前不是 `target fit` 问题。

### 2.4 也不是 control-derived target 没做

v9.2.16 已经做了 CD1-CD7 target discovery，仍没有 survivor。  
所以当前不能继续靠“再写几个 target”推进。

### 2.5 真正问题

真正问题是：

$$
\boxed{
\text{我们的 functional 目标没有捕捉到训练轨迹上的 coherent signal；AdamWParallel 才捕捉到了。}
}
$$

这和论文 `A Theory of Generalization in Deep Learning` 的启发一致：更新方向应区分 coherent population signal 和 minibatch noise。我们此前把 SNR 当成 gate，但 functional direction 仍是手写几何 target。现在结果说明：

```text
SNR gate / actuator / target fit 都不够；
functional direction 必须直接建立在 signal channel / AdamW coherent direction 上；
否则 controls 会继续赢。
```

---

## 3. v9.2.17 总体目标

v9.2.17 的总体目标是：

$$
\boxed{
\text{判断 functional update 是否应重定义为 signal-aligned optimizer metric；若失败，则正式回到 primitive / basis design。}
}
$$

这个目标分成两条互斥路线。

### 路线 A：Signal-aligned functional update

Functional update 不再是：

$$
\Delta\theta_{\text{functional}}
=
J^\dagger\Delta z_{\text{hand-designed target}}.
$$

而是：

$$
\Delta\theta_{\text{functional}}
=
\left(
P_{\text{signal}}(\theta,t)-I
\right)
\Delta\theta_{\text{AdamW}},
$$

或：

$$
\theta_{t+1}
=
\theta_t
+
P_{\text{signal}}(\theta,t)
\Delta\theta_{\text{AdamW}}.
$$

其中 $P_{\text{signal}}$ 是 role-aware / basis-aware / SNR-aware / curvature-aware 的 update metric，不是 loss，不是 teacher，不是 label smoothing。

这条路线要证明：

```text
signal-aligned functional metric
  beats scalar LR / AdamWParallel / Random / Shuffled metric controls
  without task drop
  while improving CE-tail / margin / ECE / curvature / KMNIST
```

### 路线 B：Return to primitive / basis design

如果 signal-aligned metric 也输给 scalar LR / AdamWParallel controls，则说明：

```text
当前 FC-LQ/A4/A7c family 没有可提取 functional advantage。
```

此时应该停止 functional gate 修补，回到：

```text
1. AdamW-only full-pass repair；
2. basis factory；
3. primitive redesign；
4. strong MLP / QuadraticFeatureMLP baseline challenge；
5. external fair only after base success。
```

---

## 4. v9.2.17 核心假设

### H1：AdamWParallel dominance 表明当前 functional 应沿 signal channel 重新定义

如果 AdamWParallelDirection 在 CEp99 / ECE / Margin / NLL 四个指标上都最强，那么它很可能不是普通噪声，而是一个 signal-aligned update。

H1 成立标准：

在 P1/P2 中，AdamWParallelDirection 的收益能被以下变量解释：

```text
extra step norm
cosine with task gradient
role-specific coherent gradient
SNR by role
dataset / hard-mode condition
```

如果 AdamWParallel 的收益只等价于 scalar LR 增大，则 H1 弱成立：功能性更新目前没有独特方向，只是 base optimizer under-step。

### H2：如果 signal-aligned functional 只是 scalar LR disguise，则不能算 functional advantage

必须加入 scalar LR / trust-region controls：

```text
LRx1.003
LRx1.01
LRx1.03
LRx1.10
AdamWParallel same norm
AdamWParallel trust-ratio
```

若 signal-aligned functional 不超过这些 controls，则不能声称 functional advantage。

H2 成立标准：

Functional metric 必须满足：

$$
MetricGain_{\text{functional}}
>
MetricGain_{\text{best scalar-LR control}}
+
\delta.
$$

### H3：Role / basis-specific signal metric 可能比全局 AdamWParallel 更有价值

AdamWParallel 是全局方向，可能粗糙。PureKAN 有清楚的 roles：

```text
lift_identity
T2 quadratic coefficient
A4 bounded rational actuator
A7 basis entropy actuator
output linear coefficient
```

Functional metric 应该选择性增强或抑制这些 roles。

H3 成立标准：

role-aware metric 比 global AdamWParallel 在 paired replay 中更好：

$$
CEp99_{\text{role-func}} < CEp99_{\text{AdamWParallel}},
$$

或：

$$
MarginP10_{\text{role-func}} > MarginP10_{\text{AdamWParallel}},
$$

或：

$$
Curvature_{\text{role-func}}\leq0.90Curvature_{\text{AdamWParallel}}.
$$

### H4：功能优势如果存在，应优先出现在 KMNIST / hard-mode / tail metrics

当前 LQ/A7c 不是在 MNIST easy mode 上失败，而是 full-pass 和 KMNIST hard modes 没闭合。Functional update 的价值应体现在：

```text
KMNIST repair
CEp99 reduction
margin_p10 improvement
wrong_confidence_p95 reduction
ECE/NLL improvement
curvature/local-Lipschitz improvement
```

H4 成立标准：

$$
\Delta Acc_{\text{KMNIST,functional}}
-
\Delta Acc_{\text{KMNIST,AdamW}}
\geq0.005,
$$

或机制上：

$$
CEp99_{\text{KMNIST,functional}}<CEp99_{\text{KMNIST,AdamW}},
$$

$$
MarginP10_{\text{KMNIST,functional}}>MarginP10_{\text{KMNIST,AdamW}}.
$$

### H5：如果 v9.2.17 仍无 survivor，应正式停止当前 functional line

这是一个硬停止假设。  
如果：

```text
AdamWParallel dominance explained；
A4e signal not revived；
control-derived target failed；
signal-aligned functional metric also fails；
```

则当前 FC-LQ/A4/A7c family 下没有可提取 functional advantage。

H5 触发后 route 必须写：

```text
R9-ReturnToPrimitiveBasisFactory
```

不能继续下一轮调 gate。

---

## 5. Candidate 设计

### 5.1 Base candidates

```text
B0-MLP-match:
  official same-parameter MLP baseline。

QF-QuadraticFeatureMLP:
  strong diagnostic baseline。

LQ0-LQ-t2-h256:
  current FC-PureKAN near-pass base。

A7c-BasisEntropy-ValueOnly:
  current P4-qualified actuator route best。

A4e-BoundedRational-FusedCoeffGrad:
  previous row-level clue candidate, but v9.2.16 replication failed。
```

### 5.2 Existing controls

```text
C0-AdamWOnly
C1-NoOpMatchedOverhead
C2-RandomMatchedNorm
C3-ShuffledTarget
C4-ShuffledSNRMask
C5-AdamWParallelDirection
C6-InvertedTargetSign
C7-InvertedEventController
```

### 5.3 New scalar optimizer controls

```text
OC1-LRScale-1.003
OC2-LRScale-1.01
OC3-LRScale-1.03
OC4-LRScale-1.10
OC5-AdamWParallel-SameNorm
OC6-AdamWParallel-TrustRatio-0.003
OC7-AdamWParallel-TrustRatio-0.01
OC8-AdamWParallel-TrustRatio-0.03
```

### 5.4 Signal-aligned functional candidates

```text
SF0-AdamWOnly:
  reference。

SF1-GlobalSNRMetricAdamW:
  role-level SNR scales AdamW update.

SF2-RoleSNRMetricAdamW:
  separate scale per PureKAN role.

SF3-BasisChannelSNRMetricAdamW:
  separate scale per basis / actuator channel.

SF4-CurvatureDampedSignalMetric:
  damp high-curvature components of AdamW update.

SF5-TailAwareSignalMetric:
  event-triggered metric only on CEp99 / margin-tail events.

SF6-KMNISTHardModeSignalMetric:
  hard-mode event metric, no class weight / sampler.

SF7-ControlContrastiveSignalMetric:
  metric chosen to maximize predicted Real-vs-AdamWParallel gap.

SF8-OrthogonalResidualFunctional:
  residual correction orthogonal to AdamW, only if it beats AdamWParallel in paired replay.

SF9-AbstainUnlessBeyondAdamWParallel:
  skip functional update unless predicted gain exceeds AdamWParallel control.
```

### 5.5 Metric parameterization

Functional metric can be written as:

$$
P_{\text{signal}} =
I + \sum_r \alpha_r P_r,
$$

where $P_r$ selects role $r$:

```text
lift
T2 coefficient
A4 actuator
A7 actuator
output linear
```

Role scales must be bounded:

$$
|\alpha_r|\leq \alpha_{\max}.
$$

Default brackets:

```text
alpha_max = 0.003, 0.01, 0.03
```

This prevents functional update from becoming unbounded LR tuning.

### 5.6 Events

```text
E0-uniform-low-frequency
E1-CalibratedCEp99Tail
E2-CalibratedMarginTail
E3-CalibratedWrongConfidence
E4-CalibratedCurvatureSpike
E5-KMNISTHardModeEvent
E6-ControlDominanceEvent
E7-AbstainUnlessBeyondAdamWParallel
```

Coverage target:

$$
0.03\leq Coverage\leq0.15.
$$

---

## 6. 实验阶段

## P0：v9.2.16 boundary reproduction

### 目标

复现 v9.2.16 terminal boundary，确认当前不是测量噪声。

### 必须记录

```text
source_route
p2_a4e_kmnist_h80_signal_pass
a4e_row_level_beat_count
p3_control_derived_survivor_count
p4_control_contrastive_pass
best_control
best_control_win_count
best_control_by_metric
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R8-NoExtractableFunctionalAdvantage
best_control = AdamWParallelDirection
a4e_row_level_beat_count = 0
p3_control_derived_survivor_count = 0
p4_control_contrastive_pass = 0
fake_proxy_count = 0
```

### 可视化

```text
p0_v9216_boundary_dashboard.svg
p0_best_control_by_metric.svg
p0_no_survivor_summary.svg
```

---

## P1：AdamWParallel dominance autopsy

### 目标

解释 AdamWParallelDirection 为什么赢。P1 不做新 functional，只做 dominance attribution。

### 必须记录

```text
dataset
seed
horizon
metric
best_control
AdamWParallel_win
AdamWParallel_delta
RealFunctional_delta
Random_delta
NoOp_delta
ShuffledTarget_delta
delta_norm
delta_norm_vs_adamw
cos_with_adamw
cos_with_task_gradient
role_grad_norm
role_snr
role_update_fraction
CEp99_before
CEp99_after
margin_p10_before
margin_p10_after
ECE_before
ECE_after
```

### 判断标准

P1 必须给出以下归因之一：

```text
A1-understep:
  AdamWParallel 等价于更大 LR / extra AdamW step。

A2-role_signal:
  AdamWParallel wins because specific PureKAN roles have coherent signal.

A3-tail_signal:
  AdamWParallel wins primarily on tail/hard-mode events.

A4-control_metric_artifact:
  AdamWParallel wins due to aggregation metric artifact, not true task/mechanism improvement.

A5-no_functional_signal:
  AdamWParallel is simply task gradient; no independent functional signal visible.
```

### 可视化

```text
p1_adamwparallel_win_map.svg
p1_control_rank_by_metric_dataset_horizon.svg
p1_cosine_and_norm_attribution.svg
p1_role_snr_vs_control_gain.svg
```

---

## P2：Scalar LR / extra-AdamW control matrix

### 目标

判断 AdamWParallel 是否只是 scalar LR / trust-ratio disguise。若是，functional update 不能拿它当独特优势。

### 设置

```text
candidates = OC1-OC8
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
horizons = 1,5,20,80
events = E1,E2,E5,E6
```

### 必须记录

```text
control_id
lr_scale
trust_ratio
dataset
seed
event
horizon
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
step_norm
cos_with_base_adamw
system_step_ratio
memory_ratio
```

### 判断标准

LR equivalence if：

$$
MetricGain_{\text{AdamWParallel}}
\leq
MetricGain_{\text{best LR control}}+\epsilon.
$$

Use:

```text
epsilon_CEp99 = 0.02 relative
epsilon_margin = 0.005 absolute
epsilon_ECE = 0.005 absolute
```

If LR equivalence holds, then AdamWParallel is not functional evidence.

### 可视化

```text
p2_lr_scale_control_pareto.svg
p2_adamwparallel_vs_lr_controls.svg
p2_metric_gain_vs_step_norm.svg
```

---

## P3：Signal-aligned functional metric factory

### 目标

构造不等价于 scalar LR 的 signal-aligned functional update，并在 paired replay 中测试是否击败 AdamWParallel / LR controls。

### 必跑 candidates

```text
SF1-SF9
```

### 必须记录

```text
functional_id
metric_type
role_scale_vector
basis_scale_vector
event_type
coverage
dataset
seed
horizon
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
delta_vs_AdamWParallel
delta_vs_best_LR_control
delta_vs_Random
delta_vs_NoOp
step_ratio
memory_ratio
```

### 判断标准

Signal-functional paired replay pass：

$$
Acc_{\text{SF}}\geq Acc_{\text{AdamW}}-0.005.
$$

and at least one:

$$
CEp99_{\text{SF}}\leq CEp99_{\text{AdamWParallel}}-0.03|CEp99_{\text{AdamW}}|,
$$

$$
MarginP10_{\text{SF}}\geq MarginP10_{\text{AdamWParallel}}+0.01,
$$

$$
ECE_{\text{SF}}\leq ECE_{\text{AdamWParallel}}-0.005,
$$

$$
Curvature_{\text{SF}}\leq0.90Curvature_{\text{AdamWParallel}}.
$$

and:

$$
MetricGain_{\text{SF}}>MetricGain_{\text{best LR control}}+\epsilon.
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
p3_signal_functional_vs_controls.svg
p3_role_metric_heatmap.svg
p3_functional_gain_vs_lr_control.svg
p3_kmnist_tail_gain.svg
```

---

## P4：Short-run validation for survivors

### 目标

只有 P3 survivor 才进入 short-run。验证 paired replay survivor 是否能转化为 50/240-step 连续训练收益。

### 设置

```text
steps = 50,240
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
functional_candidates = top P3 survivors
controls = AdamWOnly, AdamWParallel, best LR control, Random, NoOp
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
event_count
event_coverage
bad_event_rate
step_ratio_q90
memory_ratio
```

### 判断标准

Task safety：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005.
$$

Mechanism pass：

至少一个：

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamWParallel}},
$$

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamWParallel}},
$$

$$
ECE_{\text{functional}}\leq ECE_{\text{AdamWParallel}},
$$

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamWParallel}}.
$$

Control pass：

Functional must beat:

```text
AdamWParallel
best LR control
Random
NoOp
```

on at least one mechanism metric.

### 可视化

```text
p4_short_run_functional_vs_adamwparallel.svg
p4_short_run_task_mechanism_pareto.svg
p4_event_timeline.svg
p4_control_comparison.svg
```

---

## P5：Full 10-seed functional re-entry

### 目标

只有 P4 pass 后打开。验证 signal-aligned functional update 是否能在完整训练中带来 task-safe advantage。

### 设置

```text
epochs = 20
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
baseline = MLP-match
diagnostic baseline = QuadraticFeatureMLP
controls = AdamWOnly, AdamWParallel, best LR control, Random, NoOp
```

### 必须记录

```text
candidate
dataset
seed
val_acc
test_acc
delta_vs_MLP
delta_vs_AdamW_base
delta_vs_AdamWParallel
delta_vs_best_LR_control
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

Control superiority：

$$
MetricGain_{\text{functional}}
>
MetricGain_{\text{AdamWParallel}},
$$

and:

$$
MetricGain_{\text{functional}}
>
MetricGain_{\text{best LR control}}.
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

### 可视化

```text
p5_macro_delta_vs_controls.svg
p5_kmnist_repair_matrix.svg
p5_seedwise_win_matrix.svg
p5_task_geometry_pareto.svg
p5_ce_tail_margin_panel.svg
p5_ece_nll_panel.svg
```

---

## P6：Noise / robustness / signal-channel validation

### 目标

验证 signal-aligned functional 是否真有 signal/noise 分离作用，而不是 clean setting 偶然有效。

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
role_snr
active_fraction
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

Signal-channel relevance：

当 label noise 增大时，functional metric 应减少 low-SNR roles 的 active fraction：

$$
ActiveFraction_{\text{low-SNR,noise}=0.20}
<
ActiveFraction_{\text{low-SNR,noise}=0.05}.
$$

### 可视化

```text
p6_noise_robustness_curve.svg
p6_noise_ce_tail.svg
p6_snr_active_fraction_vs_noise.svg
p6_signal_channel_dashboard.svg
```

---

## P7：Primitive / basis reset decision

### 目标

如果 P3-P6 无 survivor，则正式停止当前 functional line，回到 primitive / basis design。P7 不是实验补丁，而是路线决策。

### 触发条件

以下任一成立：

```text
P3 no paired replay survivor；
P4 no short-run survivor；
P5 full run no macro/KMNIST/geometry/tail gain；
functional fails to beat AdamWParallel and LR controls；
```

### 必须记录

```text
current_family
functional_attempts_count
paired_replay_survivor_count
short_run_survivor_count
full_run_survivor_count
best_control
control_dominance_type
return_to_primitive_required
next_primitive_goal
```

### Primitive reset 方向

若触发 reset，下一阶段不应继续 functional gate，而应做：

```text
1. AdamW-only full-pass repair for LQ/A7c；
2. basis factory:
   centered/normalized T2；
   Legendre2-only；
   bounded rational as base basis, not actuator-only；
   piecewise-linear local basis；
   shared RBF local basis；
3. stronger baseline challenge:
   QuadraticFeatureMLP；
   same-shape MLP；
   hidden-bracket MLP；
4. external fair only after AdamW base full-pass or robust near-pass with clear mechanism。
```

### 判断标准

Return route：

```text
R9-ReturnToPrimitiveBasisFactory
```

must be selected if no control-resistant functional survivor appears.

### 可视化

```text
p7_route_decision_tree.svg
p7_functional_attempt_exhaustion_table.md
p7_next_basis_factory_map.svg
```

---

## 7. Required artifacts

```text
run_manifest.json
contract_audit_v9217.csv
p0_v9216_boundary_reproduction.csv
p1_adamwparallel_dominance_autopsy.csv
p2_scalar_lr_extra_adamw_control_matrix.csv
p3_signal_aligned_functional_metric_factory.csv
p4_short_run_signal_functional_validation.csv
p5_full_functional_reentry_10seed.csv
p6_noise_robustness_signal_channel_validation.csv
p7_primitive_basis_reset_decision.csv
paired_replay_branch_trace_v9217.csv
role_snr_metric_trace_v9217.csv
functional_metric_event_trace_v9217.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9216_boundary_unstable
F3_adamwparallel_dominance_unattributed
F4_adamwparallel_equivalent_to_lr_control
F5_signal_metric_no_paired_survivor
F6_signal_metric_control_equivalent
F7_short_run_task_drop
F8_short_run_control_equivalent
F9_full_run_no_macro_or_kmnist_gain
F10_functional_system_fail
F11_noise_robustness_fail
F12_quadratic_baseline_explains_gain
F13_return_to_primitive_required
F14_fake_or_proxy_violation
F15_artifact_missing
```

---

## 8. Route decision

### Route cases

```text
R1-AdamWParallelIsLRControl:
  AdamWParallel dominance is explained by scalar LR / extra AdamW step.

R2-SignalAlignedFunctionalPairedPass:
  signal-aligned functional metric beats AdamWParallel and LR controls in paired replay.

R3-SignalAlignedFunctionalShortRunPass:
  short-run shows task-safe mechanism gain.

R4-SignalAlignedFunctionalFullAdvantage:
  full 10-seed shows macro/KMNIST/geometry/tail gain.

R5-GeometryOnlyFunctional:
  functional improves curvature/ECE/NLL but not task.

R6-AdamWParallelDominanceUnresolved:
  controls still dominate and attribution unclear.

R7-SignalMetricControlEquivalent:
  signal metric is no better than controls.

R8-NoFunctionalAdvantageCurrentFamily:
  all functional designs fail on current LQ/A4/A7c family.

R9-ReturnToPrimitiveBasisFactory:
  stop functional gate tuning; go back to primitive/basis design.

R10-ExternalReady:
  functional passes task/geometry/system/control/robustness/strong-baseline gates.
```

### route_decision.json 必须记录

```text
route
base_candidate
best_functional_candidate
best_control
best_lr_control
adamwparallel_dominance_type
lr_equivalence_pass
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
return_to_primitive_required
external_ready
primary_blocker
next_required_implementation
success_v9217_signal_functional
success_v9217_full_functional
success_v9217_external_ready
```

---

## 9. 第一轮执行顺序

```text
Step 1:
  P0 复现 v9.2.16 boundary。

Step 2:
  P1 解释 AdamWParallel 为什么赢。
  如果归因不清，不进入 P3。

Step 3:
  P2 做 scalar LR / extra AdamW control matrix。
  判断 AdamWParallel 是否只是 LR disguise。

Step 4:
  P3 只在 P1/P2 之后做 signal-aligned functional metric factory。
  必须击败 AdamWParallel 和 LR controls 才能进入 P4。

Step 5:
  P4 short-run validation。

Step 6:
  P5 full 10-seed functional re-entry。

Step 7:
  P6 noise / robustness / signal-channel validation。

Step 8:
  P7 route decision。
  如果仍无 survivor，正式回到 primitive / basis factory。
```

---

## 10. 停止条件

### Minimum success

```text
P0 boundary reproduced
P1 AdamWParallel dominance attributed
P3 at least one signal-aligned functional beats AdamWParallel / LR controls in paired replay
no fake/proxy/offload/loss/teacher violation
```

### Functional advantage success

```text
Minimum success
+
P4 short-run task-safe mechanism gain
+
P5 full 10-seed task-safe macro/KMNIST/geometry/tail gain
```

### External-ready success

```text
Functional advantage success
+
P6 noise robustness pass
+
strong baseline challenge pass
```

### Failure stop

```text
1. v9.2.16 boundary cannot be reproduced；
2. AdamWParallel dominance is only LR/extra-step effect；
3. no signal-aligned metric beats AdamWParallel and scalar LR controls；
4. short-run harms task；
5. full run gives no macro/KMNIST/geometry/tail gain；
6. functional breaks system gate；
7. gains are explained by QuadraticFeatureMLP；
8. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 11. 最终解释规则

### Case A：AdamWParallel 等价于 LR control

必须声明：

```text
Current functional failures indicate optimizer under-step / scalar update issue, not independent functional advantage.
```

这时不能说 functional update 成功。

### Case B：signal-aligned functional beats AdamWParallel and LR controls

可以声明：

```text
Functional update becomes signal-aligned optimizer metric with causal advantage beyond scalar optimizer controls.
```

但只有 P5/P6 通过后，才能进入 external fair。

### Case C：paired replay pass but full run fail

必须声明：

```text
Functional effect is local but not robust over training horizon / seeds.
```

### Case D：all signal-aligned metrics fail

必须声明：

```text
No extractable functional advantage in current LQ/A4/A7c family; return to primitive / basis design.
```

### Case E：functional improves only curvature

必须声明：

```text
Functional update has geometry-only evidence, not task advantage.
```

---

## 12. 最终建议

v9.2.17 的一句话策略是：

$$
\boxed{
\text{把 functional update 从“手写 output target”改成“signal-aligned optimizer metric”；若仍输给 AdamWParallel/LR controls，就正式回到 primitive/basis design。}
}
$$

当前最关键的问题不是系统、AdamW、SNR、event、actuator，而是：

```text
1. AdamWParallel 为什么赢？
2. 它是不是只是 LR / extra-step control？
3. 是否存在 role-aware / basis-aware signal metric 能超过它？
4. 如果不存在，是否应停止当前 functional line？
5. 下一步是否应回到 AdamW-only full-pass 与 basis factory？
```
