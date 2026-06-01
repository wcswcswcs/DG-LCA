# DG-KAN v9.2.11 Functional Causality Controller：从 One-Step Safe 到 Multi-Step Advantage 的完整实验计划

> 本计划基于 v9.2.10 `Functional Predictor Repair` 的真实 terminal route 制定。  
> v9.2.10 已经把 functional update 的问题定位得更清楚：**one-step safety 可以被修复，但 short-run functional 仍没有通过系统与因果机制 gate。**  
> 因此 v9.2.11 不再继续简单调 SNR 阈值、functional step fraction 或单个 direction。  
> 本轮目标是建立一个完整的 functional causality controller：先证明 functional event 在短窗口内有可重复的因果收益，再控制系统 overhead，最后才重新打开 10-seed full functional re-entry。
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
> Functional update 必须是 update rule，不是 loss。训练目标保持：
>
> $$
> L_{\text{task}}=CE(y,p_\theta(x)).
> $$

---

## 0. 当前状态与目标是否达成

### 0.1 v9.2.10 的真实 terminal route

v9.2.10 的最终 route 是：

```text
route = R5-FunctionalUnsafe
base_candidate = LQ-t2-h256
best_functional_candidate = D9-SignalChannelProjection
best_gate = G1-RoleSNROnly
best_step_fraction = 0.01
success_v9210_functional_predictor = true
success_v9210_functional_advantage = false
success_v9210_external_ready = false
```

这说明：**functional predictor 的 one-step 安全性已经有进展，但 functional advantage 没有达成。**

v9.2.10 有三个关键结果。

第一，v9.2.9 的失败被真实复现。当前旧方向 `D1-QuadraticCoeffDamping + G1 + rho0.10` 仍然有：

```text
bad-step rate = 0.819444
prediction corr = 0.231090
```

第二，P2 direction factory 找到了真正 one-step safe 的 survivor。最佳是：

```text
D9-SignalChannelProjection + G1 + rho0.01
bad-step rate = 0.027778
holdout non-harm = 0.972222
prediction corr = 0.649843
overhead = 0.126000
```

第三，P4 short-run 没有通过。BestFunctional 在 50/240 steps 下 task-safe，但 step time ratio 分别约：

```text
50 steps:  step ratio ≈ 1.429962
240 steps: step ratio ≈ 1.446869
```

超过本轮 short-run strong system gate `<=1.20`。更重要的是，BestFunctional 的机制收益很小，而且 `GeometryD1Control` 与 `RandomMatchedControl` 也出现相近甚至更好的 CEp99 / margin 变化，因此不能证明 functional causality。

因此 v9.2.10 的准确结论是：

$$
\boxed{
\text{one-step predictor 已修复，但 short-run causal advantage 与 strong system overhead 没有通过。}
}
$$

### 0.2 达到目标了吗

没有达到终极目标，也没有达到 functional advantage 目标。

当前达成的是：

```text
1. LQ base 仍然是可用 FC-PureKAN near-pass base；
2. low-cost SNR / role-level gate 可用；
3. D9 direction 能通过 one-step safety；
4. P3 calibrated predictor 通过 precision-first gate。
```

当前未达成的是：

```text
1. P4 short-run system strong gate；
2. P4 short-run functional causality；
3. P5 full 10-seed functional re-entry；
4. P6 robustness；
5. P7 strong baseline / external ready；
6. Beyond-MLP significant advantage。
```

因此不能声明：

```text
functional update 成功；
functional update 有独特优势；
functional update 可以进入 external fair；
PureKAN 已经显著超过 MLP。
```

### 0.3 独立判断：现在到底卡在哪里

v9.2.10 之前，卡点是 functional predictor 不安全。现在这个问题部分解决了：D9 + G1 + rho0.01 的 one-step bad-step rate 已经降到 `0.027778`，holdout non-harm 达到 `0.972222`，prediction corr 达到 `0.649843`。

新的 blocker 是两个：

```text
1. multi-step causal effect weak:
   one-step safe 不等于多步有益。P4 中 functional 的机制收益很小，controls 也有类似收益。

2. short-run functional overhead too high under strong gate:
   step ratio 约 1.43，低于 broad 1.50，但高于本轮 short-run 1.20 strong gate。
```

更深层地说：

$$
\boxed{
\text{functional update 现在不是“安全性完全不行”，而是“因果收益不够独特，且执行策略不够经济”。}
}
$$

这不是 optimizer 问题，也不是 LQ base 问题。AdamW-only base 已经 near-pass，LQ 的 P4 已经通过。现在的问题是：

```text
functional event 何时发生？
发生时更新哪个子空间？
更新幅度多大？
收益如何和 NoOp / Random / Geometry-only control 区分？
系统 overhead 如何 amortize？
```

---

## 1. v9.2.11 的整体目标

v9.2.11 的整体目标是：

$$
\boxed{
\text{把 one-step safe functional update 推进为 multi-step causal functional advantage。}
}
$$

这不是简单追 final accuracy，也不是继续调 AdamW。它要回答四个问题。

### Q1：functional event 是否有可重复的短窗口因果收益

在相同 checkpoint、相同 batch sequence、相同 AdamW update 下，只替换 functional event 为：

```text
real functional
NoOp matched overhead
Random matched norm
Shuffled gate
Geometry-only no-SNR
```

如果 real functional 不能在 paired replay 中显著优于 controls，则不能进入 full run。

### Q2：functional 机制收益是否不是随机噪声

必须证明 functional 至少改善一个机制指标，并且优于 controls：

```text
CEp99
margin_p10
ECE
NLL
curvature
local Lipschitz
basis usage entropy
KMNIST miss-row margin
```

### Q3：functional overhead 是否可以被 controller amortize

当前 P4 step ratio 约 `1.43`，在 broad gate `<=1.50` 内，但没有达到 strong gate `<=1.20`。v9.2.11 要明确分两层：

```text
Broad system pass:
  StepRatio <= 1.50
  MemoryRatio <= 1.05

Strong system pass:
  StepRatio <= 1.20
  MemoryRatio <= 1.05
```

如果 functional 有明确机制收益但只过 broad gate，仍可作为 diagnostic survivor；如果要进入 external-ready，必须进一步压 overhead。

### Q4：functional 是否能改善 KMNIST 与 macro gap

v9.2.7 的 robust miss rows 全部来自 KMNIST；v9.2.10 还没有证明 functional 能修复这些 modes。v9.2.11 的 functional advantage 必须优先在 KMNIST 或 CE-tail / margin-tail 上体现，而不是只在 MNIST already-easy mode 上微调。

---

## 2. 核心假设

### H1：v9.2.10 的 P4 失败不是 one-step safety 失败，而是 event-level causal attribution 不足

v9.2.10 的 D9 已经通过 P2 one-step safety。P4 不过的主要原因不是 bad-step，而是 short-run 中 real functional 与 controls 的机制收益差距不够。

H1 成立标准：

在 paired event replay 中，real functional 相比 NoOp / Random / ShuffledSNR 满足至少一个：

$$
\Delta CEp99_{\text{real}} < \Delta CEp99_{\text{control}},
$$

$$
\Delta MarginP10_{\text{real}} > \Delta MarginP10_{\text{control}},
$$

$$
\Delta Curvature_{\text{real}} < \Delta Curvature_{\text{control}}.
$$

并且 task 不伤：

$$
Acc_{\text{real}}\geq Acc_{\text{AdamW}}-0.005.
$$

### H2：当前 D9 SignalChannelProjection 过于保守，导致 one-step safe 但 multi-step benefit 弱

D9 的 safety 很好，但可能把 functional direction 投影得太接近 AdamW coherent gradient，导致它不是一个有独特几何作用的 correction。

H2 成立标准：

如果 D9 的 functional direction 与 AdamW step cosine 过高：

$$
\cos(d_{\text{D9}}, \Delta\theta_{\text{AdamW}})\geq0.90,
$$

且机制收益接近 NoOp / AdamW，则 D9 只是安全但弱。

此时需要引入 orthogonal-signal geometry direction：

$$
d_{\perp}
=
d_{\text{geometry}}
-
\frac{\langle d_{\text{geometry}}, g\rangle}{\|g\|^2+\epsilon}g,
$$

再通过 SNR / holdout gate 过滤。

### H3：functional 需要“高价值事件”，不是固定 stride 触发

v9.2.10 的 short-run使用固定窗口 / 固定候选，可能触发了很多低价值事件。Functional update 应该只在以下事件发生：

```text
margin tail event
CEp99 spike event
curvature spike event
basis collapse / entropy low event
KMNIST hard-mode event
high-confidence SNR event
```

H3 成立标准：

event-gated functional 比 uniform-stride functional 满足：

$$
BadEventRate_{\text{event}}\leq BadEventRate_{\text{uniform}},
$$

并且：

$$
MechanismGain_{\text{event}}\geq MechanismGain_{\text{uniform}}.
$$

### H4：functional overhead 可以通过 amortized event controller 进入 strong gate

当前 step ratio 约 `1.43`。如果 functional event coverage 低，但每次 event 计算重，平均 step 仍可控制。H4 假设：通过事件稀疏化、缓存 SNR、减少 per-event holdout audit，可以把 amortized step ratio 降到：

$$
StepRatio\leq1.20.
$$

H4 成立标准：

$$
EventCoverage\leq0.10,
$$

且：

$$
StepRatio_{q90}\leq1.20.
$$

若 functional收益明显但 step ratio 只能到 `<=1.50`，则 route 写为：

```text
FunctionalUsefulButStrongSystemNotClosed
```

而不是写 success。

### H5：functional 的主要价值应体现在 KMNIST / hard modes / robustness，而不是 MNIST 平均精度

v9.2.7 的难点集中在 KMNIST。H5 假设 functional 的核心价值是修复 hard modes 与 signal/noise 分离，而不一定提高 easy dataset。

H5 成立标准：

$$
\Delta Acc_{\text{KMNIST,functional}}
-
\Delta Acc_{\text{KMNIST,AdamW}}
\geq0.005.
$$

或者至少：

$$
CEp99_{\text{KMNIST,functional}}
<
CEp99_{\text{KMNIST,AdamW}},
$$

$$
MarginP10_{\text{KMNIST,functional}}
>
MarginP10_{\text{KMNIST,AdamW}}.
$$

### H6：如果 functional 的收益被 random/control 解释，则当前 functional direction 不成立

如果 RandomMatchedControl、GeometryD1Control 或 ShuffledSNR 能达到同样 CEp99 / margin / curvature 改善，则不能声明 functional causality。

H6 成立标准：

Real functional 必须在 paired replay 和 full run 中至少有一个机制指标显著优于 controls：

$$
MetricGain_{\text{real}}-MetricGain_{\text{best control}}\geq\delta_{\text{mechanism}}.
$$

其中：

```text
CEp99 delta threshold = 0.05 relative
MarginP10 delta threshold = 0.02 absolute
Curvature ratio threshold = 0.90
```

---

## 3. Functional controller 的设计

### 3.1 更新形式

保持 AdamW 主更新不变：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{\text{AdamW}}
+
\lambda_f\Delta\theta_{\text{functional-safe}}.
$$

Functional 部分：

$$
\Delta\theta_{\text{functional-safe}}
=
a(e)
\cdot
q_{\text{gate}}
\odot
\Pi_{\text{safe}}
\left(
d_{\text{functional}}
\right).
$$

其中：

```text
a(e):
  event-level accept probability / event controller。

q_gate:
  SNR / role / subspace gate。

Pi_safe:
  task-safe projection or holdout-safe projection。

d_functional:
  geometry-aware direction。
```

### 3.2 Event controller

定义事件分数：

$$
s_{\text{event}}
=
w_1 z(CEp99)
+
w_2 z(-MarginP10)
+
w_3 z(Curvature)
+
w_4 z(1-BasisEntropy)
+
w_5 z(SNR_{\text{role}})
$$

只在：

$$
s_{\text{event}}\geq\tau_{\text{event}}
$$

时触发 functional candidate。

事件类型：

```text
E0-uniform-stride:
  固定频率 baseline。

E1-CEp99-tail:
  CE tail 高时触发。

E2-margin-tail:
  margin_p10 低时触发。

E3-curvature-spike:
  curvature / local Lipschitz 高时触发。

E4-basis-entropy-collapse:
  basis usage entropy 低时触发。

E5-KMNIST-hardmode:
  不用 class labels 调 loss，只在 KMNIST dataset mode 下加强 mechanism audit。
```

### 3.3 Direction family

不继续只用 D9。v9.2.11 要比较三类方向。

#### A. Safe signal projection directions

```text
D9-SignalChannelProjection:
  v9.2.10 best one-step survivor。

D10-OrthogonalSignalGeometry:
  geometry direction 去除 task-gradient 平行成分，只保留 task-safe orthogonal correction。

D11-SignalSubspaceCurvature:
  将 geometry direction 投影到近期 coherent gradient subspace，再做 curvature correction。

D12-SignalSubspaceMarginTail:
  使用近期 coherent gradient subspace 中与 margin tail 相关的方向。
```

#### B. PureKAN role-specific directions

```text
D13-LiftConditionCorrection:
  针对 lift condition number / effective rank。

D14-QuadraticBasisEntropyCorrection:
  针对 T2 basis usage entropy / dominant basis fraction。

D15-OutputScaleTailCorrection:
  针对 CEp99 / logit norm tail，但不改 loss。

D16-KMNISTHardModeGeometry:
  针对 KMNIST hard rows 的 mechanism direction，不使用 class weight / sampler。
```

#### C. Negative / diagnostic directions

```text
D17-RandomMatchedNorm:
  random control。

D18-ShuffledSignalMask:
  shuffled SNR / signal mask。

D19-InvertedSNRDirection:
  低 SNR 方向 control。

D20-GeometryD1Control:
  v9.2.10 中表现接近 functional 的 geometry control。
```

### 3.4 Safety controller

安全控制分三层。

第一层，一阶 task projection：

$$
\delta_{\text{proj}}
=
\delta
-
\frac{\max(0,g^T\delta)}{\|g\|^2+\epsilon}g.
$$

第二层，exchangeability holdout precheck：

$$
L_{\text{holdout}}(\theta+\rho\delta_{\text{proj}})
-
L_{\text{holdout}}(\theta)
\leq \epsilon_h.
$$

第三层，abstention：

如果 predictor confidence 不足，跳过 event：

$$
p_{\text{nonharm}}<\tau_{\text{nonharm}}
\Rightarrow
\Delta\theta_{\text{functional}}=0.
$$

### 3.5 Overhead controller

记录每个 event 的成本：

$$
Overhead_{\text{event}}
=
\frac{T_{\text{functional event}}}{T_{\text{AdamW step}}}.
$$

记录 amortized cost：

$$
Overhead_{\text{amortized}}
=
EventCoverage
\cdot
Overhead_{\text{event}}.
$$

要求：

$$
Overhead_{\text{amortized}}\leq0.20.
$$

强目标：

$$
StepRatio_{q90}\leq1.20.
$$

最低可用目标：

$$
StepRatio_{q90}\leq1.50.
$$

---

## 4. Candidate 设计

### 4.1 Base candidates

```text
A0-LQ0-AdamW:
  LQ-t2-h256 AdamW-only base。

A1-LQ1-FaninScale-AdamW:
  v9.2.7 best repair reference。
```

### 4.2 Functional candidates

```text
F0-AdamWOnly:
  no functional update。

F1-D9-SignalChannelProjection:
  v9.2.10 best one-step survivor。

F2-D10-OrthogonalSignalGeometry:
  task-gradient-orthogonal geometry direction。

F3-D11-SignalSubspaceCurvature:
  coherent signal subspace + curvature direction。

F4-D12-SignalSubspaceMarginTail:
  coherent signal subspace + margin-tail correction。

F5-D13-LiftConditionCorrection:
  lift condition / effective rank correction。

F6-D14-QuadraticBasisEntropyCorrection:
  T2 basis usage entropy correction。

F7-D15-OutputScaleTailCorrection:
  CEp99/logit tail update direction，不改 loss。

F8-D16-KMNISTHardModeGeometry:
  hard-mode geometry direction，不能用 sampler/class weight。

F9-EventControllerBest:
  使用 P2/P3 选出的 best direction + best event controller。

F10-LowOverheadControllerBest:
  F9 的 low-overhead implementation。
```

### 4.3 Controls

```text
C0-NoOpMatchedOverhead:
  触发同样 event，做同样测量，不改参数。

C1-RandomMatchedNorm:
  同 norm / 同 role / 同 event coverage 随机方向。

C2-ShuffledSNRMask:
  同 active ratio，mask 打乱。

C3-InvertedSNRMask:
  更新低 SNR 方向，应失败。

C4-GeometryD1Control:
  v9.2.10 中接近 real functional 的 control，必须比较。

C5-HoldoutOnlyControl:
  只做 holdout precheck，不改变方向。
```

### 4.4 System variants

```text
S0-full-audit:
  每个 event 做完整 holdout / curvature / SNR。

S1-cached-snr:
  SNR EMA 缓存，减少每次 event 计算。

S2-sparse-event:
  event coverage <= 0.10。

S3-no-holdout-runtime:
  用 P3 predictor 替代 runtime holdout，只用于 short-run 后期，需单独审计。

S4-amortized-controller:
  低频 audit + 高频 cheap predictor。
```

---

## 5. 实验阶段

## P0：v9.2.10 复现与 audit

### 目标

复现 v9.2.10 的关键结果，确认最新文件不是测量噪声，并建立本轮 baseline。

### 设置

```text
base = LQ0,LQ1
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
steps = 8
reference functional = D9 + G1 + rho0.01
```

### 必须记录

```text
candidate_id
dataset
seed
bad_step_rate
holdout_nonharm_fraction
prediction_corr
mean_holdout_delta
CEp99_delta
margin_p10_delta
curvature_delta
overhead_ratio
step_ratio
memory_ratio
```

### 判断标准

复现通过：

$$
|bad\_step\_rate-0.027778|\leq0.05,
$$

$$
|prediction\_corr-0.649843|\leq0.15.
$$

若 P0 不稳定，先修 runner / measurement，不进入后续实验。

### 可视化

```text
p0_v9210_reproduction_dashboard.svg
p0_holdout_delta_distribution.svg
p0_event_safety_by_dataset.svg
```

---

## P1：P4 short-run 失败归因

### 目标

分析 v9.2.10 P4 为什么 task-safe 但没有通过：到底是 overhead、机制弱、control 太强、事件覆盖不合理，还是 direction 与 AdamW 太相似。

### 必须记录

```text
functional_candidate
control_candidate
event_type
event_coverage
step_ratio_q50
step_ratio_q90
memory_ratio
CEp99_delta
margin_p10_delta
curvature_delta
ECE_delta
NLL_delta
basis_entropy_delta
cos_with_adamw_step
cos_with_task_gradient
functional_norm_ratio
control_norm_ratio
```

### 判断标准

P1 必须明确主因：

```text
F1-overhead_dominant:
  mechanism good but step q90 > gate。

F2-mechanism_weak:
  mechanism metrics close to NoOp / AdamW。

F3-control_equivalent:
  random / geometry control as good as real functional。

F4-direction_too_adamw_like:
  cos(functional, AdamW) >= 0.90 and no unique mechanism gain。

F5-event_bad:
  low-value events dominate。
```

### 可视化

```text
p1_failure_factor_matrix.svg
p1_functional_vs_controls_mechanism.svg
p1_event_coverage_vs_gain.svg
p1_cosine_with_adamw.svg
p1_overhead_breakdown.svg
```

---

## P2：paired event replay causality

### 目标

在同一 checkpoint、同一 batch sequence 上做 paired replay，直接比较 real functional 与 controls 的因果效果。这一步是 v9.2.11 的核心。

### 方法

对每个 candidate 选定 event checkpoint $\theta_t$，复制多个 branch：

```text
Branch A:
  AdamW only

Branch B:
  AdamW + real functional

Branch C:
  AdamW + NoOp matched overhead

Branch D:
  AdamW + Random matched norm

Branch E:
  AdamW + ShuffledSNR

Branch F:
  AdamW + GeometryD1Control
```

然后运行相同后续 batch sequence：

```text
horizon = 1, 5, 20 steps
```

### 必须记录

```text
event_id
dataset
seed
checkpoint_step
event_type
branch
horizon
holdout_loss_delta
val_proxy_acc_delta
CEp99_delta
margin_p10_delta
curvature_delta
local_lipschitz_delta
basis_entropy_delta
step_time
memory_ratio
```

### 判断标准

Event-level causality pass：

real functional 必须在 `horizon=5` 或 `20` 上优于 best control：

$$
CEp99_{\text{real}}\leq CEp99_{\text{best control}}-0.05|CEp99_{\text{AdamW}}|,
$$

或：

$$
MarginP10_{\text{real}}\geq MarginP10_{\text{best control}}+0.02,
$$

或：

$$
Curvature_{\text{real}}\leq0.90Curvature_{\text{best control}}.
$$

并且：

$$
Acc_{\text{real}}\geq Acc_{\text{AdamW}}-0.005.
$$

### 可视化

```text
p2_paired_replay_branch_curves.svg
p2_event_level_causality_forest.svg
p2_real_vs_best_control_delta.svg
p2_horizon_effects.svg
```

---

## P3：event controller 与 direction selection

### 目标

基于 P2 结果选择真正有因果收益的 event type + direction + gate 组合。

### 候选组合

```text
Event types:
  E1-CEp99-tail
  E2-margin-tail
  E3-curvature-spike
  E4-basis-entropy-collapse
  E5-high-confidence-SNR

Directions:
  D9-D16

Gates:
  RoleSNR
  HighConfidenceRoleSNR
  HoldoutPrecheck
  PredictedNonharm
  TwoLevelRoleSubspace
```

### 必须记录

```text
event_type
direction_id
gate_id
coverage
bad_event_rate
holdout_nonharm_rate
paired_replay_gain
mechanism_gain
overhead_event
overhead_amortized
step_ratio_q90_estimated
```

### 判断标准

Candidate promotion：

$$
bad\_event\_rate\leq0.05,
$$

$$
holdout\_nonharm\geq0.70,
$$

$$
paired\_replay\_gain>0,
$$

$$
coverage\geq0.03.
$$

Strong candidate：

$$
step\_ratio\_q90\leq1.20.
$$

Broad candidate：

$$
step\_ratio\_q90\leq1.50.
$$

### 可视化

```text
p3_event_direction_gate_pareto.svg
p3_coverage_vs_gain.svg
p3_overhead_vs_gain.svg
p3_candidate_promotion_table.md
```

---

## P4：short-run controller validation

### 目标

对 P3 survivor 做 50/240-step short-run，验证 paired replay 的收益能否转化为真实连续训练收益。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
steps = 50,240
base = LQ0,LQ1
candidates = top P3 survivors
controls = AdamW, NoOp, Random, ShuffledSNR, GeometryD1Control
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
step_ratio_q50
step_ratio_q90
memory_ratio
```

### 判断标准

Short-run broad pass：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
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
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

Control pass：

functional 必须优于 NoOp / Random / GeometryD1Control 至少一个机制指标。

Strong system pass：

$$
StepRatio_{q90}\leq1.20.
$$

### 可视化

```text
p4_short_run_task_mechanism_pareto.svg
p4_short_run_controls.svg
p4_event_timeline.svg
p4_step_ratio_distribution.svg
p4_ce_margin_curvature_panel.svg
```

---

## P5：full 10-seed functional re-entry

### 目标

只有 P4 broad pass 后才打开。验证 functional update 在完整 training setting 下是否有 task-safe advantage。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
epochs = 20
base = LQ0/LQ1
candidate = best P4 survivor
controls = AdamW, NoOp, Random, ShuffledSNR, GeometryD1Control
baseline = MLP-match
diagnostic baseline = QuadraticFeatureMLP
```

### 必须记录

```text
candidate
dataset
seed
val_acc
test_acc
train_acc
val_loss
test_loss
delta_vs_MLP
delta_vs_AdamW_LQ
delta_vs_QuadraticFeatureMLP
near_pass
full_pass
CEp50
CEp90
CEp99
margin_p10
wrong_confidence_p95
ECE
NLL
curvature_ratio
jacobian_norm_ratio
local_lipschitz_ratio
basis_usage_entropy
lift_condition_number
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

Tail / calibration pass：

至少一个：

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
p5_event_coverage_heatmap.svg
p5_controls_comparison.svg
```

---

## P6：robustness and noisy-signal validation

### 目标

验证 functional 是否符合论文启发的 signal/noise 分离，而不只是 full run 偶然有效。

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
event_coverage
SNR_active_fraction
bad_event_rate
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

SNR relevance：

$$
ActiveFraction_{\text{noise}=0.20}
<
ActiveFraction_{\text{noise}=0.05}.
$$

### 可视化

```text
p6_noise_robustness_curve.svg
p6_noise_ce_tail.svg
p6_snr_active_fraction_vs_noise.svg
p6_noise_task_geometry_pareto.svg
```

---

## P7：strong baseline and external-ready gate

### 目标

判断 functional 是否准备进入 external fair validation。

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
contract_audit_v9211.csv
p0_v9210_reproduction.csv
p1_p4_failure_attribution.csv
p2_paired_event_replay_causality.csv
p3_event_controller_direction_selection.csv
p4_short_run_controller_validation.csv
p5_full_functional_reentry_10seed.csv
p6_robustness_noisy_signal_validation.csv
p7_strong_baseline_external_ready.csv
functional_event_trace_v9211.csv
paired_replay_branch_trace_v9211.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_base_gate_regression
F3_v9210_reproduction_unstable
F4_p4_failure_unattributed
F5_event_causality_fail
F6_control_equivalence_fail
F7_event_controller_no_survivor
F8_short_run_task_drop
F9_short_run_system_overhead_fail
F10_full_run_no_macro_gain
F11_kmnist_repair_fail
F12_geometry_no_gain
F13_strong_baseline_explains_gain
F14_noise_robustness_fail
F15_external_ready_fail
F16_fake_or_proxy_violation
F17_artifact_missing
```

---

## 7. Route decision

### Route cases

```text
R1-EventCausalityEstablished:
  paired event replay proves real functional beats controls.

R2-ShortRunFunctionalSafe:
  short-run controller is task-safe and mechanism-positive.

R3-FullFunctionalAdvantage:
  full 10-seed functional run improves macro/KMNIST or geometry/tail without task/system harm.

R4-FunctionalUsefulButStrongSystemNotClosed:
  functional mechanism works but step q90 remains >1.20 and <=1.50.

R5-FunctionalControlEquivalent:
  controls explain functional gains.

R6-FunctionalDirectionTooAdamWLike:
  functional direction is safe but indistinguishable from AdamW.

R7-EventControllerNoSurvivor:
  no event/direction/gate combination survives paired replay.

R8-FunctionalOverheadDominant:
  mechanism gain exists but overhead breaks broad system gate.

R9-ExternalReady:
  task-safe mechanism-positive functional passes strong baseline and robustness gates.
```

### route_decision.json 必须记录

```text
route
base_candidate
best_functional_candidate
best_direction
best_event_type
best_gate
best_system_variant
p2_event_causality_pass
p4_short_run_pass
p5_full_reentry_pass
functional_task_safe
functional_mechanism_pass
functional_control_pass
functional_system_broad_pass
functional_system_strong_pass
functional_kmnist_repair_pass
noise_robustness_pass
strong_baseline_challenge_pass
external_ready
primary_blocker
next_required_implementation
success_v9211_event_causality
success_v9211_short_run
success_v9211_full_functional
success_v9211_external_ready
```

---

## 8. 第一轮执行顺序

```text
Step 1:
  P0 复现 v9.2.10 D9 one-step survivor 与 P4 failure。

Step 2:
  P1 对 P4 failure 做因子归因。
  如果归因不清，不进入 P2。

Step 3:
  P2 做 paired event replay causality。
  这是 v9.2.11 的核心，不允许跳过。

Step 4:
  P3 根据 P2 选择 event / direction / gate / system controller。

Step 5:
  P4 做 short-run controller validation。
  只跑 P3 survivors。

Step 6:
  P5 做 full 10-seed functional re-entry。
  只有 P4 broad pass 后才打开。

Step 7:
  P6 做 robustness/noisy-signal validation。

Step 8:
  P7 判断 external-ready。
```

---

## 9. 停止条件

### Minimum success

```text
P2 event causality pass
P4 short-run broad pass
control pass
no fake/proxy
```

### Functional advantage success

```text
Minimum success
+
P5 full 10-seed task-safe mechanism gain
+
KMNIST repair or macro improvement
```

### External-ready success

```text
Functional advantage success
+
strong baseline challenge pass
+
noise robustness pass
+
system broad pass
```

### 失败停止

```text
1. v9.2.10 one-step survivor cannot be reproduced；
2. P4 failure cannot be attributed；
3. real functional does not beat controls in paired replay；
4. no event controller survivor；
5. short-run functional harms task；
6. short-run overhead exceeds broad gate；
7. full run does not improve macro/KMNIST/geometry/tail；
8. gains are explained by QuadraticFeatureMLP or random/geometry controls；
9. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 10. 最终解释规则

### Case A：paired replay passes but full run does not

可以声明：

```text
Functional has local causal mechanism but does not yet produce global training advantage.
```

### Case B：short-run passes but full run fails

必须声明：

```text
Functional controller is locally useful but not robust across seeds/tasks.
```

### Case C：full run improves geometry but not task

必须声明：

```text
Functional gives geometry/calibration evidence but not task advantage.
```

### Case D：controls explain gains

必须声明：

```text
Observed gains are not uniquely functional; current direction/controller is insufficient.
```

### Case E：functional improves KMNIST or macro safely

可以声明：

```text
Functional update gives task-safe advantage on LQ near-pass base and becomes the main route for external fair validation.
```

---

## 11. 最终建议

v9.2.11 的一句话策略是：

$$
\boxed{
\text{不要把 one-step safe 当成 functional 成功；必须用 paired replay 和 matched controls 证明 event-level causality，再做 full re-entry。}
}
$$

当前最关键的问题不是系统实现，不是 AdamW，也不是 LQ 表达力，而是：

```text
1. functional event 是否真的比 NoOp / Random / Geometry control 有因果收益；
2. D9 是否过于接近 AdamW，导致 safe but weak；
3. 哪类 event 才值得触发 functional；
4. 如何把 step ratio 从约 1.43 压到 strong gate，或诚实记录 broad-only；
5. KMNIST / CE-tail / margin-tail 是否能从 functional 中得到真实改善。
```
