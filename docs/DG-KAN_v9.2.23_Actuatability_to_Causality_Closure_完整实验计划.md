# DG-KAN v9.2.23 Actuatability-to-Causality Closure：从 Base-Qualified Interface 到真正 Functional Advantage 的完整实验计划

> 本计划基于 v9.2.22 `Base-Qualified Strict PureKAN Functional Interface` 的真实 terminal route 制定。  
> v9.2.22 的核心进展是：**strict PureKAN functional interface 已经不再破坏 base，并且 functional actuatability audit 通过。**  
> 但 v9.2.22 的核心失败也非常清楚：**RealFunctional 在 paired replay 中仍然没有击败 AdamWParallel / best LR control。**
>
> 因此 v9.2.23 不再继续做泛泛的 output target patch，也不再继续调 SNR threshold / event coverage / loss / sampler / teacher。  
> 本轮的核心任务是：
>
> $$
> \boxed{
> \text{把 functional actuatability 从“可移动”推进到“移动后有因果收益”。}
> }
> $$
>
> 更具体地说，v9.2.23 要回答：
>
> $$
> \boxed{
> \text{为什么 N2a 已经 P4/P5/actuatability pass，但 paired replay 仍输给 AdamWParallel / LR？}
> }
> $$
>
> 本轮继续遵守：**no teacher、no self-teacher、no distillation、no loss modification、no label smoothing、no focal / margin / calibration loss、no sampler / class weight、no CPU offload、no fake / proxy rows、KAN path 不使用 PyTorch loss.backward graph、PureKANConv / PureKANFormer 继续 deferred。**

---

## 0. 当前状态与独立判断

### 0.1 v9.2.22 的 terminal route

v9.2.22 的最终 route 是：

```text
route = R3-FunctionalActuatabilityRetained
base_candidate = LQ-t2-h256
best_base_neutral_candidate = N2a-TinyInit-RationalFunc-BranchRatioCap
success_v9222_strict_purekan_functional = false
success_v9222_full_functional = false
success_v9222_external_ready = false
```

关键 gate：

```text
interface_contract_pass = 1
interface_p4_pass = 1
interface_p5_nearpass = 1
functional_actuatability_pass = 1
paired_replay_pass = 0
functional_task_safe = 1.0
functional_control_pass = 0
```

关键数值：

```text
p2_best_near_pass_count = 9/9
p2_best_macro_delta = -0.003166655699412028
p3_max_r_perp = 139.16336059570312
p3_bad_event_rate = 0.0
p4_real_beats_adamwparallel_rate = 0.17592592592592593
p4_real_beats_best_lr_rate = 0.17592592592592593
```

P4 branch aggregate：

| branch | rows | CEp99 delta | margin delta | acc delta | task-safe |
|---|---:|---:|---:|---:|---:|
| RealFunctional | 108 | `+0.000010` | `+0.000003` | `0.000000` | `1.000000` |
| AdamWParallelTrustRatio-0.03 | 108 | `-0.003665` | `+0.001175` | `0.000000` | `1.000000` |
| LRScale-1.03 | 108 | `-0.003665` | `+0.001175` | `0.000000` | `1.000000` |
| RandomMatchedNorm | 108 | `+0.001802` | `+0.002940` | `+0.000109` | `1.000000` |
| NoOpMatchedOverhead | 108 | `0.000000` | `0.000000` | `0.000000` | `1.000000` |

### 0.2 本轮达到什么，没有达到什么

v9.2.22 达到了三个重要目标。

第一，v9.2.21 的 `I1c` base-breaking 问题被定位，failure mechanism 是：

```text
B5-hard_tail_amplification
```

第二，base-neutral interface factory 找到了 `N2a-TinyInit-RationalFunc-BranchRatioCap`，它通过：

```text
contract
P4
P5 near-pass
```

这意味着 strict PureKAN functional interface 不再必然破坏 base。

第三，P3 functional actuatability pass，说明 functional channel 没有被完全冻死，至少在 audit metric 上仍能产生 non-AdamW displacement。

但 v9.2.22 没有达成 strict PureKAN functional success。原因是：

```text
paired_replay_pass = 0
RealFunctional 没有 beat AdamWParallel / best LR
short/full validation 没打开
external-ready 没打开
```

### 0.3 当前最重要的独立判断

v9.2.22 的结果不是“functional interface 失败回原点”，而是进入了一个更细的新边界：

$$
\boxed{
\text{base qualification 和 actuatability 已闭合；causal usefulness 未闭合。}
}
$$

也就是说，当前 blocker 已经不是：

```text
P4 system；
P5 base near-pass；
functional channel completely dead；
task safety；
no-fake/no-proxy；
FT7 core 是否存在。
```

当前 blocker 是：

$$
\boxed{
\text{actuatability metric 与 paired replay causal gain 脱节。}
}
$$

P3 中 $r_{\perp}$ 很大，bad event rate 为 0；但 P4 中 RealFunctional 的 CEp99 delta 几乎为 0，margin delta 也几乎为 0，而 AdamWParallel / LR control 明显改善 CEp99 和 margin。这说明当前 functional channel **可以动**，但它的实际移动没有进入有价值的 task-tail / margin-tail / curvature trajectory。

---

## 1. 当前问题的本质

### 1.1 不是 base 破坏问题

v9.2.21 的 blocker 是 `I1c` 破坏 P5 near-pass。v9.2.22 中 `N2a` 已达到：

$$
near\_pass\_count=9/9,
$$

$$
\Delta Acc_{\text{macro}}=-0.0031666557.
$$

这说明 base-neutral interface 方向是正确的，P5 gate 已被修复。当前不能再把主要问题归因于 base trainability。

### 1.2 不是 functional channel 完全静音

P3 actuatability pass：

$$
r_{\perp}=139.16336059570312,
$$

$$
bad\_event\_rate=0.
$$

但这个数本身也暴露出风险：$r_{\perp}$ 可能因为 AdamW reference displacement 很小或 normalization denominator 很小而被放大。也就是说，P3 的 actuatability metric 需要校准：它证明“存在非 AdamW displacement”，但不证明“这个 displacement 对 logits / CE tail / margin tail 有因果价值”。

### 1.3 真正问题是 functional event 的 causal alignment

Paired replay 中 RealFunctional 的平均效果几乎为零：

$$
\Delta CEp99_{\text{Real}} \approx +0.000010,
$$

$$
\Delta Margin_{\text{Real}} \approx +0.000003.
$$

而 AdamWParallel / LR 的效果是：

$$
\Delta CEp99_{\text{AdamWParallel}}=-0.003665,
$$

$$
\Delta Margin_{\text{AdamWParallel}}=+0.001175.
$$

因此当前的问题不是“不能动”，而是：

$$
\boxed{
\text{functional event 没有把可动性转化成对 hard-tail / margin / curvature 的有用更新。}
}
$$

### 1.4 为什么不能继续小修

如果继续只调：

```text
functional step fraction
event threshold
SNR tau
target type
single candidate basis
```

很可能只是在同一个失败机制上局部扰动。v9.2.23 应该先解释：

```text
P3 的 actuatability 为什么没有带来 P4 gain？
RealFunctional 的实际 logit movement 到底落在哪些样本 / class / tail 上？
N2a 的 branch ratio cap 是否过度保守？
functional channel 是否在 paired replay 中被 safety projection / event policy 中和？
AdamWParallel 的优势是不是仍来自 task gradient signal，而 RealFunctional 没有接上 signal channel？
```

这才是本轮的核心。

---

## 2. v9.2.23 总体目标

v9.2.23 的总体目标是：

$$
\boxed{
\text{建立 actuatability-to-causality calibration，并获得一个 paired-replay pass 的 strict PureKAN functional interface。}
}
$$

这个目标分成四个层级。

### 2.1 Minimum diagnostic success

最低诊断成功不是 full training，而是解释 v9.2.22 为什么 P3 pass、P4 fail：

```text
P0 reproduces v9.2.22 route
P1 calibrates proxy actuatability vs realized logit movement
P2 identifies whether failure is silence, misalignment, scale, event, or control dominance
```

### 2.2 Local functional causality success

如果某 candidate 在 paired replay 中满足：

$$
Acc_{\text{real}}\geq Acc_{\text{AdamW}}-0.005,
$$

并且：

$$
RealFunctional > AdamWParallel,
$$

$$
RealFunctional > best\ LR,
$$

则 strict PureKAN functional local causality 成立。

### 2.3 Full functional success

如果 paired replay survivor 在 short/full validation 中满足：

$$
\Delta Acc_{\text{functional-vs-AdamW}}\geq-0.005,
$$

并且至少一个：

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
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}},
$$

则 full functional success 成立。

### 2.4 External-ready success

只有在：

```text
P4 pass
P5 near-pass or full-pass
paired replay pass
short/full validation pass
robustness pass
strong baseline pass
```

全部成立后，才允许 external-ready。PureKANConv / PureKANFormer 继续 deferred。

---

## 3. 核心假设

### H1：v9.2.22 paired replay 失败是因为 functional movement 在 realized logits 上太小

虽然 P3 的 $r_{\perp}$ 很大，但 P4 的 CEp99 / margin delta 约为 0。H1 假设：P3 的 $r_{\perp}$ 是 proxy metric，不能反映实际 event 后 logits 的有效变化。

H1 成立标准：

若测得：

$$
\frac{
\|\Delta z_{\text{real functional}}\|
}{
\|\Delta z_{\text{AdamWParallel}}\|+\epsilon
}
<0.10,
$$

或：

$$
\frac{
\|\Delta z_{\text{tail, real functional}}\|
}{
\|\Delta z_{\text{tail, AdamWParallel}}\|+\epsilon
}
<0.10,
$$

则说明 RealFunctional 在实际 paired replay 中近似 silent。

### H2：base-neutral repair 让 functional channel 过度保守

`N2a-TinyInit-RationalFunc-BranchRatioCap` 修复了 base P5，但可能通过 tiny init / branch cap 把 functional channel 压得太弱。H2 假设：需要在 base-neutral 不变的前提下，把 event-time branch ratio 调到 FT7-like safe band，而不是训练期常驻放大。

H2 成立标准：

存在 event-time branch ratio band：

$$
r_{\min}\leq r_{\text{branch,event}}\leq r_{\max},
$$

使得：

```text
bad_event_rate <= 0.05
task_safe_rate >= 0.95
RealFunctional beats AdamWParallel / best LR
```

### H3：current functional event 没有对准 hard-tail / margin-tail signal channel

P4 中 AdamWParallel / LR 改善 CEp99 和 margin，说明 task gradient signal 仍然最有效。H3 假设：RealFunctional 的 event 不是无用，而是没有投到 hard-tail / margin-tail 样本和角色上。

H3 成立标准：

如果 tail-conditioned event 或 role-signal event 满足：

$$
\Delta CEp99_{\text{real}}<\Delta CEp99_{\text{AdamWParallel}},
$$

或：

$$
\Delta MarginP10_{\text{real}}>\Delta MarginP10_{\text{AdamWParallel}},
$$

则 H3 成立。

### H4：functional channel 需要 FT7-like branch / derivative target，而不是 output-target fit

v9.2.20/21 显示 FT7 的 branch ratio 与 derivative scale 对 curvature gain 有强相关。v9.2.23 不应回到 old output target，而应直接用 FT7 mechanism target：

$$
r_{\text{branch}},
\quad
s_{\text{eff}},
\quad
\cos(\Delta\theta_{\text{func}},\Delta\theta_{\text{AdamW}}),
\quad
r_{\perp}.
$$

H4 成立标准：

FT7-mechanism-matched event 比 generic rational functional event 更好：

$$
Curvature_{\text{FT7-match}}\leq0.90Curvature_{\text{generic}},
$$

或：

$$
CEp99_{\text{FT7-match}}<CEp99_{\text{generic}},
$$

or:

$$
MarginP10_{\text{FT7-match}}>MarginP10_{\text{generic}}.
$$

### H5：如果 calibrated actuatability 仍不能 beat controls，则 current N2a rational interface family 不足

如果 P1/P2 校准后，所有 branch-ratio / derivative / event-time activation variants 仍然 control-equivalent，则当前 rational base-neutral interface family不足。下一步应返回 dual-role basis factory，而不是继续 patch N2a。

H5 触发标准：

```text
P4 paired replay pass count = 0
and RealFunctional beats AdamWParallel rate < 0.30
and RealFunctional beats best LR rate < 0.30
after calibrated event-time activation
```

---

## 4. Candidate 设计

### 4.1 Baselines

```text
B0-MLP-match:
  same-parameter MLP reference。

LQ0-LQ-t2-h256:
  current strict FC-PureKAN base。

I1c-current:
  v9.2.21 best interface, P4 pass but P5 fail。

N2a-current:
  v9.2.22 best base-neutral interface, P4/P5/actuatability pass but paired replay fail。

V8-FT7:
  mechanism source only, not final strict candidate。
```

### 4.2 Calibration candidates

```text
CAL0-N2a-current:
  unchanged reference。

CAL1-N2a-realized-logit-calibrated:
  event accepted only if realized logit displacement exceeds minimum threshold.

CAL2-N2a-tail-logit-calibrated:
  event accepted only if tail-sample displacement is non-trivial.

CAL3-N2a-branch-ratio-calibrated:
  event-time alpha chosen to reach FT7-like branch ratio band.

CAL4-N2a-derivative-scale-calibrated:
  event-time alpha chosen to reach FT7-like effective derivative scale.

CAL5-N2a-nonadamw-orthogonal-calibrated:
  preserves only non-AdamW functional component with controlled norm.

CAL6-N2a-control-contrastive-calibrated:
  event accepted only if predicted Real > AdamWParallel / LR.
```

### 4.3 Interface variants

```text
V1-TinyInit-Rational-BranchCap:
  current N2a reference。

V2-ZeroInit-Rational-EventOnly:
  no base-phase functional channel update; event-only activation.

V3-Warmstart-Rational-Attach:
  train task channel first, attach functional channel at event phase.

V4-TinyInit-Piecewise-BranchCap:
  local piecewise functional basis.

V5-TinyInit-SharedRBF-BranchCap:
  local shared-RBF functional basis.

V6-DualRole-Rational-TaskOrthogonal:
  functional update projected away from task-channel AdamW direction.

V7-DualRole-Rational-TailCoupled:
  event only fires on CEp99 / margin-tail signal.
```

### 4.4 Controls

Every paired replay must include:

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

### 4.5 Forbidden shortcuts

The following are disallowed:

```text
teacher / distillation
loss modification
label smoothing
focal / margin / calibration loss
sampler / class weight
CPU offload
external residual
ordinary MLP path
PyTorch loss.backward for KAN path
fake/proxy rows
opening P5 short/full before paired replay pass
```

---

## 5. 实验阶段

## P0：v9.2.22 boundary reproduction

### 目标

复现 v9.2.22 boundary，确认 P4 failure 不是测量噪声。

### 必须记录

```text
route
best_base_neutral_candidate
interface_contract_pass
interface_p4_pass
interface_p5_nearpass
functional_actuatability_pass
paired_replay_pass
p2_best_near_pass_count
p2_best_macro_delta
p3_max_r_perp
p3_bad_event_rate
p4_real_beats_adamwparallel_rate
p4_real_beats_best_lr_rate
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R3-FunctionalActuatabilityRetained
best_candidate = N2a
P4/P5/actuatability pass = 1
paired_replay_pass = 0
fake_proxy_count = 0
```

### 可视化

```text
p0_v9222_boundary_dashboard.svg
p0_gate_ladder.svg
p0_real_vs_controls_recap.svg
```

---

## P1：Actuatability-to-realized-effect calibration

### 目标

解释为什么 P3 actuatability pass 没有带来 P4 gain。  
这一阶段不做新 full training，只做 event-level measurement。

### 必须记录

```text
candidate
dataset
seed
event_id
branch
target_fit_R2
p3_proxy_r_perp
actual_logit_delta_norm
actual_tail_logit_delta_norm
actual_nonadamw_logit_delta_norm
actual_r_z
actual_r_z_tail
actual_r_z_perp
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
branch_ratio_event
effective_derivative_event
projection_removed_norm
functional_step_norm
adamwparallel_step_norm
cos_functional_adamw
```

### 判断标准

Proxy calibration pass：

$$
Corr(r_{\perp,\text{proxy}}, r_{\perp,\text{actual}})\geq0.30.
$$

Silence diagnosis：

If:

$$
r_{z,\text{actual}}<0.10
$$

or:

$$
r_{z,\text{tail}}<0.10,
$$

then mark:

```text
failure = functional_event_silent
```

Misalignment diagnosis：

If:

$$
r_{z,\text{actual}}\geq0.10
$$

but:

$$
\Delta CEp99_{\text{real}}\geq\Delta CEp99_{\text{AdamWParallel}},
$$

and:

$$
\Delta Margin_{\text{real}}\leq\Delta Margin_{\text{AdamWParallel}},
$$

then mark:

```text
failure = functional_event_misaligned
```

### 可视化

```text
p1_proxy_vs_actual_rperp.svg
p1_actual_logit_delta_distribution.svg
p1_tail_logit_delta_vs_CEp99.svg
p1_branch_ratio_derivative_vs_gain.svg
p1_projection_removed_norm.svg
```

---

## P2：FT7-mechanism calibrated event-time activation

### 目标

在不破坏 base 的前提下，调节 event-time functional branch ratio / derivative scale，使 strict interface 更接近 FT7 的有效机制。

### 设置

```text
candidates = CAL1-CAL6
interfaces = V1,V2,V3
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
horizons = 1,5,20,80
```

### 必须记录

```text
candidate
interface
dataset
seed
event_type
horizon
branch_ratio_target
branch_ratio_actual
effective_derivative_target
effective_derivative_actual
functional_step_norm
bad_event_rate
holdout_nonharm
CEp99_delta
margin_p10_delta
curvature_delta
ECE_delta
NLL_delta
real_beats_adamwparallel
real_beats_best_lr
step_ratio_q90
memory_ratio
```

### 判断标准

Event-time activation survivor：

$$
bad\_event\_rate\leq0.05,
$$

$$
holdout\_nonharm\geq0.70,
$$

and:

$$
real\_beats\_adamwparallel\_rate\geq0.50,
$$

$$
real\_beats\_best\_lr\_rate\geq0.50.
$$

Mechanism survivor：

At least one:

$$
CEp99_{\text{real}}<CEp99_{\text{AdamWParallel}},
$$

$$
MarginP10_{\text{real}}>MarginP10_{\text{AdamWParallel}},
$$

$$
Curvature_{\text{real}}\leq0.90Curvature_{\text{AdamWParallel}}.
$$

### 可视化

```text
p2_branch_ratio_band_pareto.svg
p2_derivative_band_pareto.svg
p2_real_vs_adamwparallel_by_event.svg
p2_event_time_activation_dashboard.svg
```

---

## P3：Interface family comparison under calibrated activation

### 目标

如果 N2a rational interface仍然不够，比较 rational / piecewise / shared-RBF / warmstart attach / orthogonal variants，判断问题是否来自 basis family。

### 设置

```text
interfaces = V1,V2,V3,V4,V5,V6,V7
events = CEp99-tail, margin-tail, curvature-spike, role-signal, FT7-style-event
horizons = 1,5,20,80,240
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, InvertedRole
```

### 必须记录

```text
interface
basis_family
base_phase_policy
event_phase_policy
contract_pass
P4_pass
P5_nearpass
actuatability_pass
actual_r_z_tail
actual_r_z_perp
bad_event_rate
paired_replay_real_beats_adamwparallel_rate
paired_replay_real_beats_best_lr_rate
CEp99_delta_real
CEp99_delta_adamwparallel
margin_delta_real
margin_delta_adamwparallel
curvature_delta_real
curvature_delta_adamwparallel
```

### 判断标准

Interface survivor：

```text
contract_pass = 1
P4_pass = 1
P5_nearpass = 1
actuatability_pass = 1
paired_replay_real_beats_adamwparallel_rate >= 0.50
paired_replay_real_beats_best_lr_rate >= 0.50
```

If no interface survives, route must be:

```text
R8-InterfaceFamilyControlEquivalent
```

not full-run.

### 可视化

```text
p3_interface_family_pareto.svg
p3_basis_family_control_gap.svg
p3_paired_replay_heatmap.svg
p3_kmnist_tail_repair_by_interface.svg
```

---

## P4：Strict paired replay control gate

### 目标

Only P2/P3 survivors enter P4.  
P4 is the official local causality gate before any short/full run.

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
horizons = 1,5,20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, InvertedRole, FrozenFuncChannel
```

### 必须记录

```text
candidate
dataset
seed
event_id
horizon
branch
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
local_lipschitz_delta
acc_delta
control_rank
real_beats_adamwparallel
real_beats_best_lr
real_beats_random
real_beats_noop
task_safe
event_coverage
bad_event_rate
step_ratio_q90
memory_ratio
```

### 判断标准

P4 paired replay pass：

$$
Acc_{\text{real}}\geq Acc_{\text{AdamW}}-0.005.
$$

and:

$$
real\_beats\_adamwparallel\_rate\geq0.50,
$$

$$
real\_beats\_best\_lr\_rate\geq0.50.
$$

Mechanism gate requires at least one:

$$
CEp99_{\text{real}}<CEp99_{\text{AdamWParallel}},
$$

$$
MarginP10_{\text{real}}>MarginP10_{\text{AdamWParallel}},
$$

$$
Curvature_{\text{real}}\leq0.90Curvature_{\text{AdamWParallel}}.
$$

System gate:

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p4_paired_replay_branch_curves.svg
p4_real_vs_strong_controls.svg
p4_horizon_effect_heatmap.svg
p4_control_rank_matrix.svg
p4_seedwise_causal_win_matrix.svg
```

---

## P5：Short-run functional validation

### 目标

验证 paired replay survivor 是否能在连续训练中保持 task-safe mechanism gain。

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
p5_short_run_task_mechanism_pareto.svg
p5_short_run_controls.svg
p5_event_timeline.svg
p5_ce_tail_margin_panel.svg
p5_step_ratio_distribution.svg
```

---

## P6：Full 10-seed functional validation

### 目标

Only P5 survivor enters P6.  
This validates whether strict PureKAN functional update has real training advantage.

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

## P7：AdamW-only full-pass repair 并行线

### 目标

Functional 是核心，但 base 仍未 full-pass。并行推进 AdamW-only full-pass，避免 functional 判断被 weak base 污染。  
P7 不允许使用 functional update。

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
p8_noise_robustness_curve.svg
p8_strong_baseline_pareto.svg
p8_external_ready_scorecard.svg
```

---

## 6. Required artifacts

```text
run_manifest.json
contract_audit_v9223.csv
p0_v9222_boundary_reproduction.csv
p1_actuatability_realized_effect_calibration.csv
p2_ft7_mechanism_calibrated_activation.csv
p3_interface_family_comparison.csv
p4_strict_paired_replay_control_gate.csv
p5_short_run_functional_validation.csv
p6_full_10seed_functional_validation.csv
p7_adamw_only_fullpass_repair.csv
p8_robustness_external_ready.csv
actual_logit_displacement_trace_v9223.csv
branch_derivative_calibration_trace_v9223.csv
paired_replay_branch_trace_v9223.csv
functional_channel_usage_trace_v9223.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy:

```text
F1_contract_violation
F2_v9222_boundary_unstable
F3_proxy_actuatability_miscalibrated
F4_functional_event_silent
F5_functional_event_misaligned
F6_branch_ratio_band_no_survivor
F7_derivative_band_no_survivor
F8_interface_family_control_equivalent
F9_paired_replay_control_equivalent
F10_functional_lr_equivalent
F11_short_run_task_drop
F12_full_run_no_macro_kmnist_gain
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
R1-ActuatabilityProxyMiscalibrated:
  P3 proxy r_perp does not predict actual logit movement or causal gain.

R2-FunctionalEventSilent:
  RealFunctional actual logit displacement is too small.

R3-FunctionalEventMisaligned:
  RealFunctional moves logits, but not in useful tail/margin/curvature directions.

R4-FT7MechanismActivationPass:
  branch-ratio / derivative-scale calibrated event beats strong controls.

R5-InterfaceFamilySurvivor:
  at least one interface family beats AdamWParallel / best LR in paired replay.

R6-StrictPureKANFunctionalPairedPass:
  P4 strict paired replay passes.

R7-StrictPureKANFunctionalFullPass:
  P5/P6 short/full validation shows task-safe gain.

R8-InterfaceFamilyControlEquivalent:
  all calibrated interface families remain control-equivalent.

R9-AdamWFullPassNoFunctional:
  base reaches full-pass but functional remains unproven.

R10-ExternalReady:
  strict PureKAN functional route passes task/geometry/system/control/robustness/strong-baseline gates.
```

### route_decision.json 必须记录

```text
route
v9222_boundary_pass
actuatability_proxy_calibration_pass
failure_mode
best_calibrated_candidate
best_interface_family
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
success_v9223_strict_purekan_functional
success_v9223_full_functional
success_v9223_external_ready
```

---

## 8. 第一轮执行顺序

```text
Step 1:
  P0 复现 v9.2.22 boundary。

Step 2:
  P1 做 actuatability-to-realized-effect calibration。
  如果 proxy actuatability 与 actual logit movement 不相关，先修 metric，不进入 activation sweep。

Step 3:
  P2 做 FT7-mechanism calibrated event-time activation。
  重点是 branch ratio band、derivative scale band、tail-conditioned events。

Step 4:
  P3 比较 interface families。
  如果 rational N2a 仍 control-equivalent，必须比较 piecewise / shared-RBF / warmstart / orthogonal variants。

Step 5:
  P4 paired replay control gate。
  只有 P4 pass，才打开 short-run。

Step 6:
  P5 short-run validation。

Step 7:
  P6 full 10-seed validation。

Step 8:
  P7 并行 AdamW-only full-pass repair。

Step 9:
  P8 robustness / external-ready gate。
```

---

## 9. 停止条件

### Minimum success

```text
v9.2.22 boundary reproduced
actuatability-to-realized-effect calibrated
at least one strict interface candidate beats AdamWParallel / best LR in paired replay
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
1. v9.2.22 boundary cannot be reproduced；
2. proxy actuatability cannot be calibrated；
3. RealFunctional actual logit movement is silent under all safe activation bands；
4. calibrated branch/derivative events all remain control-equivalent；
5. all interface families remain control-equivalent；
6. full run gives no macro/KMNIST/geometry/tail gain；
7. functional breaks system gate；
8. gains are explained by QuadraticFeatureMLP；
9. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 10. 最终解释规则

### Case A：proxy actuatability miscalibrated

必须声明：

```text
P3 actuatability pass was insufficient; it measured non-AdamW movement but not realized causal output effect.
```

下一步先修 actuatability metric。

### Case B：event silent

必须声明：

```text
Base-neutral repair restored P5 but made functional event too weak.
```

下一步调 event-time activation band，而不是改 base training。

### Case C：event misaligned

必须声明：

```text
Functional channel moves output, but not along useful hard-tail / margin-tail / curvature directions.
```

下一步做 FT7-mechanism alignment。

### Case D：paired replay passes

可以声明：

```text
Strict PureKAN functional interface has local causal evidence under strong controls.
```

但不能声明 full functional success unless P5/P6 pass.

### Case E：full run passes

可以声明：

```text
Strict FC-PureKAN functional update has been recovered under strong controls.
```

### Case F：all calibrated interfaces fail

必须声明：

```text
Current base-neutral rational/piecewise/RBF interface family cannot carry FT7 functional core; return to deeper primitive/interface redesign.
```

---

## 11. 最终建议

v9.2.23 的一句话策略是：

$$
\boxed{
\text{先校准“可动性”是否真的产生有用 logit movement，再用 FT7 的 branch/derivative 机制激活 strict functional channel。}
}
$$

当前最关键的问题不是 functional 是否存在，也不是 base 是否可训练，而是：

```text
1. P3 r_perp 为什么这么大但 P4 几乎没有 CE/margin gain？
2. RealFunctional 是 silent，还是 misaligned？
3. base-neutral interface 是否过度压制了 event-time branch ratio？
4. FT7 的 branch ratio / derivative scale 安全区间能否迁移到 N2a？
5. rational interface 不行时，piecewise / shared-RBF / warmstart / orthogonal variants 是否能承载 FT7 core？
6. RealFunctional 能否最终超过 AdamWParallel / best LR controls？
```
