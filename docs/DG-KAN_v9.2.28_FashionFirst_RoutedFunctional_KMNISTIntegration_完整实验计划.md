# DG-KAN v9.2.28 Fashion-First Routed Functional Controller 与 KMNIST Survivor Integration 完整实验计划

> 本计划基于 v9.2.27 `Fashion Delayed 与 KMNIST Metric-Causal Repair` 的完整复盘结果制定。  
> v9.2.27 的最新 route 是：
>
> ```text
> route = R4-KMNISTRepairOnlyFashionStillPartial
> base_candidate = LQ-t2-h256
> success_v9227_fashion_survivor = False
> success_v9227_kmnist_metric_causal = True
> success_v9227_strict_purekan_functional = False
> ```
>
> 本轮结论的核心不是 “functional update 又失败了”，而是：
>
> $$
> \boxed{
> \text{KMNIST metric-causal repair 已有 survivor；Fashion delayed signal 仍不稳定，成为当前 macro blocker。}
> }
> $$
>
> 因此 v9.2.28 不能继续把 KMNIST 当作主要 failure，也不能把 Fashion 的 partial signal 直接包装成 success。下一步必须集中解决两个问题：
>
> $$
> \boxed{
> \text{Fashion signal 如何从 partial diagnostic 变成 stable survivor？}
> }
> $$
>
> 和：
>
> $$
> \boxed{
> \text{KMNIST target/primitive survivors 能否进入 routed paired replay、short-run、full-run，而不是只停留在 diagnostic actual-effect gate？}
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

### 0.1 v9.2.27 达成了什么

v9.2.27 完成了两条重要推进。

第一，Fashion delayed signal 被更完整地复验。P1 覆盖 10 seeds、6 个 horizons、6 个实际实现的 controllers：

```text
measured controllers = F0,F3,F6,F7,F8,F12
not_implemented controllers = F1,F2,F4,F5,F9,F10,F11
```

Fashion 的 best candidate 是：

```text
F7-Fashion-O2-BranchBand
rows = 60
beats AdamWParallel = 0.266667
beats best LR = 0.600000
task safe = 0.933333
CEp99 delta = -0.056023
margin delta = +0.008311
survivor = 0
```

它说明 Fashion 上确实有 **CE-tail 与 margin 的同向改善**，但 strong-control beat rate 明显不过 gate。尤其 Real vs AdamWParallel 只有 `0.266667`，远低于预设 `0.75`，因此不能把它升级为 Fashion survivor。

第二，KMNIST 从 v9.2.26 的 `actual effect pass rate = 0` 推进到了 target/primitive survivor。P3 target diagnosis 中，measured targets `K1,K2,K3,K5,K6,K8` 全部出现 survivor，其中 best target 是：

```text
K6-KMNIST-MetricCausalTarget
actual effect rate = 0.760000
```

P4 primitive repair 中，measured primitives `N2a,N2b,N2c,N3c` 里有 3 个 route survivor：

```text
N2a actual effect = 0.703704, P4 = 1, P5 near = 1
N2c actual effect = 0.666667, P4 = 1, P5 near = 1
N3c actual effect = 0.518519, P4 = 1, P5 near = 1
```

这说明 KMNIST 的主要状态已经从：

```text
oracle useful + safe fit pass, actual effect = 0
```

推进到：

```text
metric-causal target survivor + primitive survivor measured subset pass
```

### 0.2 v9.2.27 没有达成什么

v9.2.27 没有达成 strict PureKAN functional success。原因非常明确：

```text
Fashion local survivor count = 0
Fashion delayed signal not stable
P5 unified routed controller = not opened
P6 short-run = not opened
P7 full 10-seed = not opened
P8 AdamW full-pass repair = not opened / not central to this run
P9 robustness / external-ready = not opened
```

因此不能声明：

```text
strict FC-PureKAN functional update 成功；
functional 已经可 full-run；
external-ready 可以打开；
PureKANConv / PureKANFormer 可以打开。
```

### 0.3 当前最重要的新边界

v9.2.27 之后，当前 blocker 发生了变化。

之前的 blocker 是：

```text
KMNIST target/safe-fit 到 actual metric effect 断裂。
```

现在新的 blocker 是：

```text
Fashion delayed signal not stable。
```

更准确地说：

$$
\boxed{
\text{KMNIST 已经有 measured target/primitive survivors；Fashion 仍只有 partial signal，无法构成 macro functional route。}
}
$$

这意味着下一步不应该继续围绕 KMNIST 做大范围 target patch。KMNIST 需要进入 integration / preservation / short-run validation，而 Fashion 需要重新做 controller 设计和机制解释。

---

## 1. 当前问题的本质

### 1.1 Fashion 不是完全没信号，而是信号不稳定且 control beat rate 不够

`F7-Fashion-O2-BranchBand` 有很强的 CEp99 改善：

$$
\Delta CEp99=-0.056023,
$$

并且 margin 也改善：

$$
\Delta Margin=+0.008311.
$$

这比 v9.2.26 的 best Fashion signal 强很多。但它仍不通过 survivor gate，因为：

$$
BeatRate_{\text{vs AdamWParallel}}=0.266667,
$$

$$
BeatRate_{\text{vs best LR}}=0.600000.
$$

也就是说，Fashion 的问题不是 “没有 metric effect”，而是 “metric effect 不稳定、不能稳定击败 strong controls”。这提示 Fashion controller 的问题可能是 event selection / horizon selection / abstention，而不是 target 本身完全错误。

### 1.2 Fashion 的机制还没闭合

P2 Fashion mechanism diagnosis 显示，`F7` 的 horizon vs `-CEp99` 相关性达到 `0.472709`，有 delayed alignment 信号；但 branch vs `-CEp99`、derivative vs margin 等机制没有形成同样清晰的支撑。因此，Fashion 当前是：

```text
有 delayed horizon signal；
有 CEp99 / margin improvement；
但缺稳定 control superiority；
也缺完整 branch / derivative 机制解释。
```

这说明下一步要优先做 **controller policy**，不是继续只换 basis。

### 1.3 KMNIST 已不是主要 failure，但仍不是 full success

KMNIST 的 best target `K6` actual effect rate 是 `0.76`，best primitive `N2a` actual effect rate 是 `0.703704`。这是真实推进。  
但它还不是 full success，因为 P3/P4 仍是 diagnosis / primitive subset level，不是 routed paired replay / short-run / full-run。并且它的 mean CEp99 / margin delta 仍然非常小，例如：

$$
\Delta CEp99_{\text{K6}}\approx 10^{-6},
$$

$$
\Delta Margin_{\text{K6}}\approx -10^{-6}.
$$

所以 KMNIST 的下一步不是继续证明 “能不能 actual effect”，而是回答：

```text
1. K6/N2a 的 actual-effect pass 是否能在 routed paired replay 中保留？
2. 它是否能在 short-run / full-run 中产生非微小机制收益？
3. N2c / N3c 是否比 N2a 在 full trajectory 更稳？
```

### 1.4 统一路线不能再用 global one-size-fits-all controller

Fashion 和 KMNIST 的机制已经分化：

```text
Fashion:
  delayed / horizon-sensitive / CE-tail + margin signal；
  blocker 是 stability and control superiority。

KMNIST:
  metric-causal target/primitive survivor 已出现；
  blocker 是 integration and effect magnitude。

MNIST:
  目前没有强 functional need，应倾向 abstain，避免 functional update 伤害 easy mode。
```

因此，下一步应该从 global controller 转向 **routed functional controller**。但 routed 不能变成 dataset overfitting。必须同时测试：

```text
dataset-routed controller；
metric-routed controller；
event-routed controller；
control-beat abstaining controller。
```

最终 official route 应优先选择可解释的 metric/event router，而不是只靠 dataset name。

---

## 2. v9.2.28 总体目标

v9.2.28 的总体目标是：

$$
\boxed{
\text{把 KMNIST measured survivor 接入 routed functional route，同时把 Fashion partial signal 修成 stable survivor，或明确 Fashion abstain 条件。}
}
$$

本轮不以 full 10-seed 为第一目标。它必须先完成 **local paired replay route** 的严格闭合。

### 2.1 Minimum success

Minimum success 要求：

```text
v9.2.27 boundary reproduced
Fashion failure mode attributed
KMNIST survivor preserved under routed paired replay
at least one routed controller beats AdamWParallel / best LR on >= 2 datasets
no fake/proxy/offload/loss/teacher violation
```

形式化 gate：

$$
BeatRate_{\text{macro, Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro, Real vs best LR}}\geq0.60,
$$

且：

$$
Acc_{\text{Real}}\geq Acc_{\text{AdamW}}-0.005.
$$

### 2.2 Fashion survivor success

Fashion survivor 要求：

$$
BeatRate_{\text{Fashion, Real vs AdamWParallel}}\geq0.75,
$$

$$
BeatRate_{\text{Fashion, Real vs best LR}}\geq0.75.
$$

同时至少一个机制指标达到非微弱效果：

$$
\Delta CEp99_{\text{Real}}
\leq
\Delta CEp99_{\text{best control}}
-
0.0005,
$$

或：

$$
\Delta MarginP10_{\text{Real}}
\geq
\Delta MarginP10_{\text{best control}}
+
0.001,
$$

或：

$$
Curvature_{\text{Real}}
\leq
0.90Curvature_{\text{best control}}.
$$

### 2.3 Fashion abstain route success

如果 Fashion 不能形成 stable survivor，但可以明确区分 “有效事件” 与 “无效事件”，则允许 Fashion abstain route。  
Fashion abstain 不是失败包装，必须满足：

$$
BadEventRate_{\text{Fashion, abstain}}\leq0.05,
$$

$$
Acc_{\text{Fashion, routed}}\geq Acc_{\text{Fashion, AdamW}}-0.005,
$$

并且 routed macro 仍满足：

$$
BeatRate_{\text{macro, Real vs AdamWParallel}}\geq0.60.
$$

换句话说，Fashion abstain 只能作为保护机制，不能作为 functional success 本身。

### 2.4 KMNIST integration success

KMNIST 的 P3/P4 survivor 必须进入 routed paired replay：

$$
ActualEffectPassRate_{\text{KMNIST}}\geq0.50,
$$

$$
BeatRate_{\text{KMNIST, Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{KMNIST, Real vs best LR}}\geq0.50.
$$

并且至少一个机制指标非微弱：

$$
|\Delta CEp99_{\text{Real}}-\Delta CEp99_{\text{best control}}|\geq0.0005
$$

或：

$$
|\Delta Margin_{\text{Real}}-\Delta Margin_{\text{best control}}|\geq0.001
$$

或：

$$
Curvature_{\text{Real}}\leq0.90Curvature_{\text{best control}}.
$$

### 2.5 Full functional success

只有 routed paired replay pass 后，才进入 short/full run。Full functional success 要求：

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

### H1：Fashion 的 F7 有真实 tail/margin effect，但缺少 abstention 与 horizon voting

F7 的 CEp99 与 margin 都改善，但 beat rate 不够。H1 认为它不是 target 错，而是 controller 接受了太多低价值事件。  
因此要测试：

```text
multi-horizon agreement；
tail-only event；
control-beat abstention；
branch-band + delayed-horizon joint gate；
CE-tail 与 margin-tail 双条件。
```

H1 成立标准：

新增 Fashion controller 满足 Fashion survivor success，或至少满足 Fashion abstain route success。

### H2：Fashion 当前 failure 是 control-dominance，不是 effect absence

F7 有：

$$
\Delta CEp99=-0.056023,
$$

$$
\Delta Margin=+0.008311.
$$

如果新增 controller 仍不能提升 beat rate，但能保持 effect size，说明 Fashion 的主要 blocker 是 control dominance / event ranking，而不是 effect absence。

H2 成立标准：

若：

$$
|\Delta CEp99_{\text{Real}}|>0.01
$$

且：

$$
BeatRate_{\text{Real vs AdamWParallel}}<0.50,
$$

则标记为：

```text
FashionEffectExistsButControlDominated
```

### H3：KMNIST survivor 需要进入 integration，而不是继续 target patch

KMNIST 已经有 K6 target survivor 和 N2a/N2c/N3c primitive survivors。H3 认为继续大规模 target search 的边际价值低，下一步应测试这些 survivors 在 routed paired replay 和 short-run 中是否保留。

H3 成立标准：

K6 + N2a/N2c/N3c 至少一个 route 满足 KMNIST integration success。

### H4：N2a 是当前 measured subset 的 best primitive，但 full trajectory 可能 N2c/N3c 更稳

N2a actual effect rate 最高，但 CE/margin delta 很小。N2c/N3c 可能在 full trajectory 上有更好的 smoothness / locality / robustness。  
H4 要求 N2a、N2c、N3c 都进入 routed replay，不只选 N2a。

H4 成立标准：

如果 N2c/N3c 在 routed replay 中 beat rate 或 curvature 优于 N2a，则把 best primitive 从 N2a 改为 corresponding local/RBF primitive。

### H5：统一 functional success 需要 routed policy，而不是 global policy

Fashion、KMNIST、MNIST 的 functional needs 不同。H5 认为全局 controller 会继续 control-equivalent，而 routed controller 才能形成 macro survivor。

H5 成立标准：

$$
MetricGain_{\text{routed}}>
MetricGain_{\text{global}},
$$

并且：

$$
Acc_{\text{MNIST,routed}}\geq Acc_{\text{MNIST,AdamW}}-0.005.
$$

### H6：如果 Fashion 无法 stable，允许 Fashion abstain，但不能声明 broad functional success

如果 Fashion 不过，但 KMNIST + MNIST route 有 local gain，结果只能写为：

```text
KMNIST-routed functional evidence
```

不能写成 global strict PureKAN functional success。  
Global success 至少要求：

```text
>= 2 datasets have positive strong-control evidence
or
macro full-run gain with no dataset harm
```

---

## 4. Candidate 设计

### 4.1 Baselines

```text
B0-MLP-match:
  same-parameter MLP reference。

LQ0-LQ-t2-h256:
  strict FC-PureKAN AdamW-only base。

N2a-TinyInit-RationalFunc-BranchRatioCap:
  v9.2.27 measured best KMNIST primitive reference。

N2c-TinyInit-SharedRBFFunc-BranchRatioCap:
  P4/P5/actual-effect survivor, locality alternative。

N3c-SharedRBFFunc-DerivativeBand:
  P4/P5/actual-effect survivor, derivative-band alternative。

V8-FT7:
  mechanism source only, not official strict candidate。
```

### 4.2 Fashion controllers

First implement the previously missing planned controllers:

```text
F1-Fashion-O2-h20
F2-Fashion-O2-h50
F4-Fashion-O2-h160
F5-Fashion-O2-h240
F9-Fashion-O2-AbstainUnlessControlBeat
F10-Fashion-O2-MultiHorizonVote
F11-Fashion-O2-TailOnlyEvent
```

Then add mechanism-driven controllers:

```text
F13-Fashion-F7-BranchBandPlusHorizonVote:
  F7 branch-band with h50/h80/h160 agreement.

F14-Fashion-F7-ControlBeatAbstain:
  F7 event accepted only if predicted Real > AdamWParallel and Real > best LR.

F15-Fashion-F7-CEAndMarginJoint:
  CEp99-tail accepted only if margin-tail not harmed.

F16-Fashion-F7-TailOverlapGate:
  accepted event must overlap CEp99-tail and margin-p10-tail.

F17-Fashion-F7-CurvatureProtected:
  require CE/margin improvement and no curvature deterioration.

F18-Fashion-F7-DelayedEnsemble:
  ensemble vote over h50/h80/h160/h240 without averaging parameters.
```

### 4.3 KMNIST target/primitive integration candidates

Targets:

```text
K6-KMNIST-MetricCausalTarget
K2-KMNIST-O1-CEp99Tail
K1-KMNIST-O2-MarginTail
K8-KMNIST-ClassModeBucketTarget
```

Primitives:

```text
P-KM0-N2a-Rational
P-KM1-N2c-SharedRBF-BranchCap
P-KM2-N3c-SharedRBF-DerivativeBand
P-KM3-N2a+N2c-MixedLight
P-KM4-N2a-OrthogonalTail
P-KM5-N2c-OrthogonalTail
```

### 4.4 Routed controllers

```text
U0-GlobalBestSingle:
  one global controller for all datasets.

U1-DatasetRouted:
  Fashion uses best Fashion controller;
  KMNIST uses K6 + best primitive;
  MNIST abstains unless high-confidence strong event.

U2-MetricRouted:
  route by CEp99-tail / margin-tail / curvature event, not dataset name.

U3-ControlBeatRouter:
  update only if predicted Real beats AdamWParallel and best LR.

U4-HybridFT7Router:
  route by FT7-like branch ratio / derivative-scale band.

U5-AbstainHeavyRouter:
  only fire when event confidence is high; otherwise AdamW only.

U6-KMNISTOnlyRouter:
  official diagnostic route to test KMNIST integration without Fashion claim.

U7-FashionOnlyRouter:
  official diagnostic route to test Fashion delayed controller without KMNIST claim.
```

### 4.5 Controls

Every paired replay / short-run / full-run must include:

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
```

The two new controls are required because routed policies can otherwise overfit event routing:

```text
DatasetRouteShuffled:
  keep same update magnitudes, shuffle dataset-route assignment.

EventRouteShuffled:
  keep same dataset, shuffle event type assignment.
```

---

## 5. 实验阶段

## P0：v9.2.27 boundary reproduction

### 目标

复现 latest boundary，确认不是 measurement artifact。

### 必须记录

```text
route
source_route_v9226
fashion_local_survivor_count
fashion_best_candidate
fashion_best_CEp99_delta
fashion_best_margin_delta
fashion_best_beats_adamwparallel
fashion_best_beats_best_lr
fashion_best_task_safe
kmnist_target_survivor_count
kmnist_best_target
kmnist_best_actual_effect_rate
kmnist_primitive_survivor_count
kmnist_best_primitive
kmnist_best_primitive_actual_effect_rate
downstream_status
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R4-KMNISTRepairOnlyFashionStillPartial
Fashion survivor count = 0
KMNIST target survivor count >= 1
KMNIST primitive survivor count >= 1
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_gate_ladder.svg
p0_fashion_vs_kmnist_status.svg
```

---

## P1：Fashion controller completion and stable-survivor test

### 目标

补齐未实现 Fashion controllers，并测试 Fashion signal 是否能成为 stable survivor。

### 设置

```text
dataset = Fashion-MNIST
seeds = 0,1,2,3,4,5,6,7,8,9
horizons = 20,50,80,160,240,640
controllers = F0,F3,F6,F7,F8,F12,F1,F2,F4,F5,F9,F10,F11,F13,F14,F15,F16,F17,F18
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
tail_overlap
step_ratio_q90
memory_ratio
```

### 判断标准

Fashion survivor：

$$
BeatRate_{\text{Real vs AdamWParallel}}\geq0.75,
$$

$$
BeatRate_{\text{Real vs best LR}}\geq0.75,
$$

$$
Acc_{\text{Real}}\geq Acc_{\text{AdamW}}-0.005.
$$

Mechanism magnitude gate：

$$
\Delta CEp99_{\text{Real}}
\leq
\Delta CEp99_{\text{best control}}-0.0005
$$

or:

$$
\Delta MarginP10_{\text{Real}}
\geq
\Delta MarginP10_{\text{best control}}+0.001
$$

or:

$$
Curvature_{\text{Real}}\leq0.90Curvature_{\text{best control}}.
$$

Fashion abstain candidate：

$$
BadEventRate\leq0.05,
$$

$$
Acc_{\text{Fashion,routed}}\geq Acc_{\text{Fashion,AdamW}}-0.005,
$$

and event precision:

$$
Precision_{\text{accepted events beating controls}}\geq0.70.
$$

### 可视化

```text
p1_fashion_controller_pareto.svg
p1_fashion_seed_horizon_heatmap.svg
p1_fashion_real_vs_controls.svg
p1_fashion_event_precision_recall.svg
p1_fashion_effect_size_distribution.svg
```

---

## P2：Fashion mechanism attribution

### 目标

解释 Fashion 是 stable delayed signal、weak partial signal、control-dominated signal，还是 no reliable signal。

### 必须记录

```text
candidate
horizon
seed
event_id
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

Delayed alignment:

$$
Corr(horizon,-\Delta CEp99)\geq0.30
$$

or:

$$
Corr(horizon,\Delta MarginP10)\geq0.30.
$$

Branch mechanism:

$$
Corr(r_{\text{branch}},-\Delta CEp99)\geq0.30
$$

or:

$$
Corr(s_{\text{eff}},\Delta MarginP10)\geq0.30.
$$

Control dominance:

If:

$$
|\Delta CEp99_{\text{Real}}|>0.01
$$

and:

$$
BeatRate_{\text{Real vs AdamWParallel}}<0.50,
$$

then record:

```text
FashionEffectExistsButControlDominated
```

### 可视化

```text
p2_fashion_time_to_effect.svg
p2_fashion_tail_overlap.svg
p2_fashion_branch_ratio_vs_gain.svg
p2_fashion_derivative_vs_gain.svg
p2_fashion_control_dominance_map.svg
```

---

## P3：KMNIST survivor integration replay

### 目标

把 v9.2.27 的 KMNIST target/primitive survivors 放入 stronger paired replay，而不是继续停在 diagnosis table。

### 设置

```text
dataset = KMNIST
seeds = 0,1,2,3,4,5,6,7,8,9
targets = K6,K2,K1,K8
primitives = N2a,N2c,N3c,N2a+N2c,N2a-OrthogonalTail,N2c-OrthogonalTail
horizons = 1,5,20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, InvertedRole
```

### 必须记录

```text
target
primitive
seed
horizon
branch
actual_effect_pass
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
functional_channel_entropy
branch_ratio
effective_derivative
actual_tail_logit_delta
```

### 判断标准

KMNIST integration survivor：

$$
ActualEffectPassRate\geq0.50,
$$

$$
BeatRate_{\text{Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{Real vs best LR}}\geq0.50,
$$

$$
Acc_{\text{Real}}\geq Acc_{\text{AdamW}}-0.005.
$$

Magnitude gate:

At least one:

$$
|\Delta CEp99_{\text{Real}}-\Delta CEp99_{\text{best control}}|\geq0.0005,
$$

$$
|\Delta Margin_{\text{Real}}-\Delta Margin_{\text{best control}}|\geq0.001,
$$

$$
Curvature_{\text{Real}}\leq0.90Curvature_{\text{best control}}.
$$

### 可视化

```text
p3_kmnist_target_primitive_pareto.svg
p3_kmnist_seed_horizon_heatmap.svg
p3_kmnist_n2a_n2c_n3c_comparison.svg
p3_kmnist_actual_effect_vs_metric_gain.svg
```

---

## P4：Routed controller construction

### 目标

构造 official routed strict PureKAN functional route，比较 dataset routing、metric routing、event routing、abstain-heavy routing。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
controllers = U0,U1,U2,U3,U4,U5,U6,U7
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, InvertedRole, DatasetRouteShuffled, EventRouteShuffled
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
real_beats_random
real_beats_noop
task_safe
event_count
event_coverage
bad_event_rate
selected_event_type
selected_primitive
selected_target
route_confidence
step_ratio_q90
memory_ratio
```

### 判断标准

Routed paired replay pass：

$$
BeatRate_{\text{macro, Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro, Real vs best LR}}\geq0.60,
$$

$$
Acc_{\text{each dataset, Real}}\geq Acc_{\text{AdamW}}-0.005.
$$

Dataset condition:

```text
At least 2 datasets must be positive
or one dataset positive plus one dataset abstain-safe with no task/metric harm.
```

Anti-routing-overfit controls:

```text
DatasetRouteShuffled must not pass.
EventRouteShuffled must not pass.
InvertedRoleMask must not pass.
```

### 可视化

```text
p4_routed_controller_pareto.svg
p4_dataset_routing_matrix.svg
p4_real_vs_controls_all_datasets.svg
p4_event_type_usage_heatmap.svg
p4_routed_vs_global_comparison.svg
p4_route_shuffle_controls.svg
```

---

## P5：Short-run functional validation

### 目标

Only P4 survivors enter short-run. This prevents local diagnosis rows from being promoted into full training without causality.

### 设置

```text
steps = 50,240,640
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, DatasetRouteShuffled, EventRouteShuffled
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
p5_short_run_task_mechanism_pareto.svg
p5_short_run_controls.svg
p5_event_timeline.svg
p5_ce_tail_margin_panel.svg
p5_step_ratio_distribution.svg
```

---

## P6：Full 10-seed functional validation

### 目标

Validate strict PureKAN routed functional advantage over full training.

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
p6_macro_delta_vs_controls.svg
p6_kmnist_repair_matrix.svg
p6_seedwise_win_matrix.svg
p6_task_geometry_pareto.svg
p6_ce_tail_margin_panel.svg
p6_ece_nll_panel.svg
p6_functional_channel_usage_trace.svg
```

---

## P7：AdamW-only full-pass repair

### 目标

Functional 是核心，但 base 仍未 full-pass。Base repair 不能使用 functional update，也不能使用 teacher/loss/sampler/class weight。它是并行线，用于避免 weak base 污染 functional 判断。

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
p7_kmnist_miss_rows.svg
p7_ce_tail_margin.svg
p7_basis_entropy_vs_gap.svg
```

---

## P8：Robustness and external-ready gate

### 目标

Confirm strict PureKAN routed functional advantage is not a clean-setting artifact.

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
strict PureKAN routed functional candidate
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
contract_audit_v9228.csv
p0_v9227_boundary_reproduction.csv
p1_fashion_controller_completion.csv
p2_fashion_mechanism_attribution.csv
p3_kmnist_survivor_integration_replay.csv
p4_routed_controller_construction.csv
p5_short_run_functional_validation.csv
p6_full_10seed_functional_validation.csv
p7_adamw_only_fullpass_repair.csv
p8_robustness_external_ready.csv
fashion_controller_trace_v9228.csv
fashion_mechanism_trace_v9228.csv
kmnist_survivor_replay_trace_v9228.csv
routed_controller_trace_v9228.csv
paired_replay_branch_trace_v9228.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy:

```text
F1_contract_violation
F2_v9227_boundary_unstable
F3_fashion_controller_missing_implementation
F4_fashion_signal_not_reproduced
F5_fashion_signal_control_dominated
F6_fashion_mechanism_unattributed
F7_kmnist_survivor_not_preserved
F8_kmnist_effect_too_small
F9_routed_controller_control_equivalent
F10_routing_overfit_shuffled_route_pass
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
R1-FashionStableSurvivor:
  Fashion delayed/branch controller passes strong-control gate.

R2-FashionEffectControlDominated:
  Fashion has CEp99/margin effect but cannot beat AdamWParallel / LR.

R3-FashionAbstainRoute:
  Fashion cannot be positive survivor, but abstention protects task and macro route.

R4-KMNISTSurvivorPreserved:
  K6 + N2a/N2c/N3c survivor remains valid in stronger paired replay.

R5-KMNISTSurvivorNotPreserved:
  KMNIST diagnosis survivor disappears under stronger replay.

R6-RoutedControllerPass:
  routed route beats AdamWParallel / best LR on macro gate.

R7-StrictPureKANFunctionalShortRunPass:
  short-run shows task-safe mechanism gain.

R8-StrictPureKANFunctionalFullPass:
  full 10-seed shows macro/KMNIST/geometry/tail gain.

R9-KMNISTOnlyFunctionalEvidence:
  KMNIST route works but Fashion remains partial; cannot claim global functional success.

R10-DelayedSignalNotGeneralized_PrimitiveReset:
  Fashion fails and KMNIST survivor not preserved; return to deeper primitive/interface design.

R11-AdamWFullPassNoFunctional:
  base reaches full-pass but functional remains unproven.

R12-ExternalReady:
  strict PureKAN routed functional route passes task/geometry/system/control/robustness/strong-baseline gates.
```

### route_decision.json 必须记录

```text
route
v9227_boundary_pass
fashion_signal_confirmed
fashion_abstain_pass
fashion_failure_mode
fashion_best_candidate
fashion_best_effect_size
kmnist_survivor_preserved
best_kmnist_target
best_kmnist_primitive
unified_routed_controller_pass
routing_overfit_controls_pass
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
success_v9228_strict_purekan_functional
success_v9228_full_functional
success_v9228_external_ready
```

---

## 8. 第一轮执行顺序

```text
Step 1:
  P0 复现 v9.2.27 boundary。

Step 2:
  P1 先补齐 Fashion missing controllers。
  不补 F9/F10/F11，就无法判断 Fashion 是 controller 不足还是 signal 不稳定。

Step 3:
  P2 做 Fashion mechanism attribution。
  如果 Fashion effect exists but control dominated，需要做 abstain / control-beat router，而不是继续换 target。

Step 4:
  P3 做 KMNIST survivor integration replay。
  重点验证 K6 + N2a/N2c/N3c 是否能从 diagnosis survivor 转成 paired replay survivor。

Step 5:
  P4 构造 routed controller。
  必须加入 DatasetRouteShuffled 和 EventRouteShuffled controls 防止路由过拟合。

Step 6:
  P5 short-run validation。
  只有 routed paired replay survivor 才进入。

Step 7:
  P6 full 10-seed validation。
  不允许只凭 P1/P3 diagnosis 直接 full-run。

Step 8:
  P7 并行 AdamW-only full-pass repair。

Step 9:
  P8 robustness / strong baseline / external-ready。
```

---

## 9. 停止条件

### Minimum success

```text
v9.2.27 boundary reproduced
Fashion stable survivor OR Fashion abstain route established
KMNIST survivor preserved under routed replay
routed controller beats AdamWParallel / best LR on macro gate
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
1. v9.2.27 boundary cannot be reproduced；
2. Fashion missing controllers cannot be implemented；
3. Fashion delayed signal is not reproduced；
4. Fashion remains control-dominated and no abstain route works；
5. KMNIST survivor disappears under stronger paired replay；
6. routed controller remains control-equivalent；
7. routing shuffle controls pass, implying routing overfit；
8. full run gives no macro/KMNIST/geometry/tail gain；
9. functional breaks system gate；
10. gains are explained by QuadraticFeatureMLP；
11. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 10. 最终解释规则

### Case A：Fashion stable survivor + KMNIST preserved

可以声明：

```text
Strict PureKAN functional has local paired-replay evidence across multiple datasets.
```

但还不能声明 full functional success unless P5/P6 pass.

### Case B：Fashion abstain + KMNIST preserved

必须声明：

```text
Functional route is not globally active; evidence is KMNIST-routed with Fashion protected by abstention.
```

这不能写成 global strict PureKAN functional success，除非 full macro run 证明无害且有宏观收益。

### Case C：Fashion effect exists but remains control-dominated

必须声明：

```text
Fashion has delayed tail/margin effect but controller cannot select events better than optimizer controls.
```

下一步应做 event ranking / control-beat predictor，而不是继续 target patch。

### Case D：KMNIST survivor disappears in routed replay

必须声明：

```text
KMNIST P3/P4 diagnosis survivor was not stable under stronger replay.
```

下一步回到 primitive/target effect magnitude repair。

### Case E：routed controller passes but full run fails

必须声明：

```text
Functional causality is local but not robust over training horizon.
```

### Case F：full run passes

可以声明：

```text
Strict FC-PureKAN functional update has been recovered under strong controls.
```

External-ready 仍需 robustness / strong baseline.

---

## 11. 最终建议

v9.2.28 的一句话策略是：

$$
\boxed{
\text{把 KMNIST survivor 接入 routed replay，同时补齐 Fashion controller，判断 Fashion 是可修 delayed signal 还是必须 abstain。}
}
$$

当前最关键的问题不是 functional 是否存在，也不是 base 是否可训练，而是：

```text
1. F7 的 CEp99 / margin effect 为什么 beat rate 不够？
2. 补齐 F9/F10/F11 后，Fashion 是否能从 partial signal 变成 stable survivor？
3. KMNIST K6 + N2a/N2c/N3c 是否能在 routed replay 中保留优势？
4. routed policy 是否真的有 causal value，还是 dataset/event routing overfit？
5. 如果 Fashion 不稳定，是否允许 abstain 并由 KMNIST route 支撑局部 functional claim？
6. routed route 能否进入 short-run / full-run，并最终超过 AdamWParallel / best LR controls？
```
