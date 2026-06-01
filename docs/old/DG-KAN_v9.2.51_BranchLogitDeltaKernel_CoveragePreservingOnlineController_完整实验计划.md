# DG-KAN v9.2.51 Branch-Logit Delta Kernel 与 Coverage-Preserving Online Controller Closure 完整实验计划

> 本计划基于 v9.2.50 `Metric Kernel Compression 与 Coverage-Preserving Cascade Controller` 的真实复盘制定。  
> v9.2.50 的 terminal route 是：
>
> ```text
> route = R10-PredictiveButMetricKernelTooExpensive
> base_candidate = LQ-t2-h256
> success_v9250_strict_purekan_functional = False
> success_v9250_full_functional = False
> success_v9250_external_ready = False
> ```
>
> v9.2.50 的关键事实是：
>
> ```text
> P1 F7 metric subphase attribution:
>   pass = 1
>   dominant subphase = M0-logit preparation
>   ratio = 0.251454
>
> P2 metric-kernel compression:
>   pass = 0
>   best = MK4-PreallocatedWorkspaceExact
>   metric error = 0.0
>   controller agreement = 0.801759
>   step ratio = 5.072608
>
> P3 recall-first event-sparse cascade:
>   pass = 0
>   recall = 0.557284
>   coverage = 0.057407
>   precision = 0.588710
>   bad-event = 0.356452
>
> P4 cheap surrogate:
>   pass = 0
>   best = CS3-RoleBranchSignal
>   AUC = 0.543916
>   recall = 0.192362
>   max AUC = 0.594640
>
> P5 signal auxiliary:
>   pass = 0
>   best = SA3-TrustRatio
>   AUC = 0.573312
>   bad-event = 0.432099
>
> P6 oracle support:
>   oracle precision = 1.0
>   oracle coverage = 0.130926
>   oracle bad-event = 0.0
>   measured signal strata = 3
>   support measurement pass = 0
>
> P7 support-region controller:
>   pass = 0
>   best = C3-CheapSurrogatePrefilter
>   precision = 0.172566
>   coverage = 0.020926
>   bad-event = 0.066372
>
> current blocker:
>   metric_kernel_compression_failed_while_oracle_support_remains
> ```
>
> 这轮最关键的信息不是 “metric compression 又失败了”，而是 **成本主因已经从抽象的 gap-probe expensive 具体化到 logit preparation / branch logit path**。也就是说，继续只压缩 CEp99、MarginP10、RiskScore、GapScore 的标量 metric kernel，不会触及根因。真正需要压缩的是：
>
> $$
> \boxed{
> \text{多分支候选更新后的 logits 生成路径。}
> }
> $$
>
> v9.2.51 的核心任务是：
>
> $$
> \boxed{
> \text{把 expensive online gap-probe 改写为 Branch-Logit Delta Kernel，再用 coverage-preserving cascade controller 打开 LDO/LSO 与 paired replay。}
> }
> $$

---

## 0. 硬约束

本计划继续遵守：

```text
no teacher
no self-teacher
no distillation
no loss modification
no label smoothing
no focal / margin / calibration loss
no sampler / class weight
no CPU offload
no fake / proxy rows
KAN path 不使用 PyTorch loss.backward graph
official controller 不使用 dataset_name 分支
PureKANConv / PureKANFormer 继续 deferred
```

Functional update 仍然是 update rule，不是 loss：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{\text{AdamW-equivalent}}
+
\Delta\theta_{\text{functional}}.
$$

任务目标保持标准 CE：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

数据集只能作为诊断切片，不能作为 official controller 条件：

```text
allowed:
  report MNIST / Fashion-MNIST / KMNIST slice metrics
  report overhead / safety / support distribution by dataset
  report leave-dataset-out generalization
  report signal-stratum / family composition by dataset

forbidden:
  if dataset == Fashion: use controller A
  if dataset == KMNIST: use threshold B
  if dataset == MNIST: abstain
  tune threshold separately per dataset
  choose logit-delta approximation separately per dataset
  use validation/test metric at commit time
  use posthoc replay outcome at commit time for official controller
```

Controller calibration 可以使用 train-stream calibration split，但不能使用 validation / test / dataset name。任何 calibration rule 必须固定后在 heldout train-stream rows、leave-dataset-out、leave-stratum-out 上评价。

---

# Part I. 对 v9.2.50 的独立判断

## 1. v9.2.50 没有达到目标

v9.2.50 没有 strict PureKAN functional success。P8-P12 downstream 均未打开，原因是 P7 没有 system-legal metric kernel，也没有 legal controller survivor。它不能声明：

```text
strict PureKAN local causal evidence
full functional success
external-ready success
Beyond-MLP success
```

v9.2.50 的 route `R10-PredictiveButMetricKernelTooExpensive` 是准确的。它不是说 functional update 没有价值，也不是说 safe-good event 不存在；它说当前能够找到 good events 的路径仍然不够便宜、不够 legal、不够覆盖。

## 2. v9.2.50 的真实进展

v9.2.50 至少推进了三件事。

第一，成本归因从 v9.2.49 的粗粒度 `F7-metric computation CE/margin/risk/gap` 继续拆到了 F7 内部 subphase。v9.2.50 的 dominant subphase 是：

```text
M0-logit preparation
ratio = 0.251454
```

这非常重要。它说明下一步如果还只压缩 CEp99、MarginP10、RiskScore 这种标量 reduction，就会偏离瓶颈。真正要压缩的是 branch logits 的生成与复用。

第二，exact metric route 没有数值误差，但 step ratio 仍然高。`MK4-PreallocatedWorkspaceExact` 的 metric error 是 `0.0`，说明数值保持可以做到；但 step ratio 是 `5.072608`，说明单纯 prealloc workspace 不足以接近 MLP envelope。

第三，oracle support 仍然非常强：

$$
Precision_{\text{oracle}}=1.0,
$$

$$
Coverage_{\text{oracle}}=0.130926,
$$

$$
BadEvent_{\text{oracle}}=0.0.
$$

这说明 safe-good events 还在。当前失败不是 “functional event disappeared”，而是 “legal/system path cannot identify them”。

## 3. 当前真正 blocker

当前 blocker 是三层耦合：

### 3.1 Branch-logit path blocker

F7 dominant subphase 已经变成 `M0-logit preparation`。这意味着多 branch probe 的主要成本很可能来自：

```text
RealFunctional branch logits
AdamWParallel branch logits
bestLR branch logits
shadow parameter / delta view
logit buffer preparation
branch dimension materialization
temporary allocation
branch-wise synchronization
```

这不是靠 CEp99 top-k 或 risk-score fusion 能解决的。必须转向 branch-logit delta kernel：

$$
z_{\theta+\Delta\theta}
\approx
z_\theta
+
\Delta z(\Delta\theta).
$$

如果 $\Delta z$ 可以直接从 manual forward cache、edge-basis local Jacobian、low-rank sketch、或者 delta-view fused branch kernel 得到，就不需要为每个 branch 重跑完整 forward。

### 3.2 Cheap surrogate insufficient

P4 的 cheap surrogate best AUC 只有 `0.543916`，最高 AUC 也只有 `0.594640`。这说明当前 cheap features 不能单独判断 safe-good，也不能作为 high-quality prefilter。它们最多可以作为 weak recall signal 或 support density feature，不能作为 official accept rule。

### 3.3 Cascade objective 错配

P3 recall-first cascade 的 coverage 是 `0.057407`，表面上进入 `[0.03,0.15]`，但 precision 只有 `0.588710`，bad-event `0.356452`。这说明 cascade 不是 “coverage 不够”，而是 **recall 与 safety 没有同时控制**。P7 best controller coverage 反而低到 `0.020926`，precision `0.172566`，bad-event `0.066372`。所以当前 controller 在两个极端之间摆动：

```text
放宽：
  coverage 可以上来，但 bad-event / precision 崩。

收紧：
  bad-event 可能下降，但 coverage 不够，precision 也不稳。
```

下一步必须把 cascade 改成：

```text
high-recall cheap candidate discovery
+ branch-logit delta confirmation
+ risk-safe filter
+ family-balanced coverage control
```

---

# Part II. v9.2.51 总体目标

v9.2.51 的总体目标是：

$$
\boxed{
\text{用 branch-logit delta kernel 替代 expensive branch forward probe，并构建 coverage-preserving online controller。}
}
$$

更具体地说，本轮要同时闭合四件事：

```text
1. 找出 M0-logit preparation 的真实子瓶颈；
2. 实现至少一种 system-legal branch-logit delta approximation；
3. 用 branch-logit delta confirmation 替代 expensive gap-probe；
4. 在不使用 dataset-specific tuning 的条件下，通过 controller + LDO/LSO + paired replay。
```

v9.2.51 不再以 “继续压缩 F7 scalar metrics” 为主线，而以 **branch-output displacement** 为主线。

---

## 1. Branch-logit delta kernel success

对于每个 candidate update $\Delta\theta_b$，不直接 full forward 得到 $z_{\theta+\Delta\theta_b}$，而是构造：

$$
\widehat z_b
=
z_0
+
\widehat{\Delta z}_b.
$$

其中 $b$ 表示 branch：

```text
RealFunctional
AdamWParallel
bestLR
NoOp
Random
Shuffled variants
```

Branch-logit delta kernel pass 要求：

$$
\operatorname{AUC}(\widehat S_{\text{gap}},Y_{\text{safe-good}})\geq0.70
$$

or:

$$
Corr(\widehat S_{\text{gap}},V_{\text{safe-grounded}})\geq0.35.
$$

同时要求 branch-gap agreement：

$$
Agreement_{\text{safe-good decision}}\geq0.90.
$$

System gate：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

如果 branch-logit delta AUC 好但 system fail，route 记为 `R6-BranchDeltaPredictiveButTooExpensive`。如果 system pass 但 AUC fail，route 记为 `R7-BranchDeltaCheapButUninformative`。

---

## 2. Coverage-preserving cascade success

Official controller 不再是 single score，也不是 signal-only。形式为：

$$
Accept(e)
=
Candidate(e)
\land
BranchDeltaConfirm(e)
\land
RiskSafe(e)
\land
SupportBalanced(e).
$$

其中：

$$
Candidate(e)
=
\mathbb{1}
[
S_{\text{cheap-recall}}(e)\geq\tau_c
],
$$

$$
BranchDeltaConfirm(e)
=
\mathbb{1}
[
\widehat S_{\text{gap}}(e)\geq\tau_g
],
$$

$$
RiskSafe(e)
=
\mathbb{1}
[
S_{\text{risk}}(e)\leq\rho
],
$$

$$
SupportBalanced(e)
=
\mathbb{1}
[
Density(e)\geq d_0
\land
Rel(family(e))\geq r_0
].
$$

Controller pass：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{heldout}}\leq0.05.
$$

Support pass：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

System pass：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

---

## 3. Support expansion success

v9.2.50 的 measured signal strata 只有 `3`，support measurement pass = `0`。v9.2.51 必须扩大 measured support。

Measurement pass：

```text
natural_real_event_count >= 4000
balanced_diagnostic_real_event_count >= 4000
measured_signal_strata_count >= 6
measured_family_count >= 12
```

Official accepted support：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

注意：balanced diagnostic rows 不能直接混入 official heldout controller pass，除非明确分开 official natural split 与 diagnostic split。

---

## 4. Leave-out and paired replay success

只有在 branch-logit delta + controller + support + system gate 都过后，才打开 LDO/LSO 和 paired replay。

Leave-dataset-out：

$$
Acc_{\text{heldout,Real}}\geq Acc_{\text{heldout,AdamW}}-0.005.
$$

至少两个 held-out datasets 满足：

$$
BeatRate_{\text{Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{Real vs bestLR}}\geq0.50.
$$

Leave-stratum-out：

至少 $70\%$ held-out strata task-safe and：

$$
CEp99_{\text{heldout,Real}}\leq CEp99_{\text{AdamW}}+\epsilon,
$$

$$
BeatRate_{\text{heldout-stratum,Real vs AdamWParallel}}\geq0.50.
$$

Official paired replay：

$$
BeatRate_{\text{macro,Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro,Real vs bestLR}}\geq0.60.
$$

Task safety：

$$
Acc_{\text{Real,slice}}\geq Acc_{\text{AdamW,slice}}-0.005.
$$

---

# Part III. 核心假设

## H1：v9.2.50 失败的主因是 branch-logit preparation，而不是 scalar metric reduction

v9.2.50 的 dominant subphase 是 `M0-logit preparation`。H1 认为继续压缩 CEp99 / MarginP10 / RiskScore 不能解决根因，必须压缩 branch logits。

H1 成立标准：

```text
M0 子阶段中，branch forward / delta apply / logit buffer preparation 贡献 >= 60%。
```

且 branch-logit delta kernel 能使：

$$
StepRatio_{q90}
$$

比 v9.2.50 best exact metric path 下降至少 `50%`。

H1 失败标准：

```text
M0 不是 branch logit path，而是其他不可压缩 logging / sync / data movement；
branch-logit delta implementation 不降低 cost。
```

---

## H2：safe-good 支持仍然存在，问题是 legal identification

v9.2.50 oracle precision `1.0`，coverage `0.130926`，bad-event `0.0`。H2 认为 good events 没消失。

H2 成立标准：

fresh v9.2.51 natural rows 上：

$$
OraclePrecision_{\text{safe-good}}\geq0.75,
$$

$$
OracleCoverage_{\text{safe-good}}\in[0.03,0.15],
$$

$$
OracleBadEventRate\leq0.05.
$$

H2 失败标准：

```text
fresh natural replay oracle precision < 0.75
or oracle coverage < 0.03
or oracle bad-event > 0.05
```

如果 H2 失败，下一步回到 carrier/support reset，不继续 controller。

---

## H3：cheap surrogate 应作为 recall generator，不应作为 final accept rule

v9.2.50 cheap surrogate max AUC 只有 `0.594640`。H3 认为 cheap surrogate 可以作为 broad candidate generator，但不能单独 official accept。

H3 成立标准：

cheap candidate generator 达到：

$$
Recall_{\text{safe-good}}\geq0.80,
$$

$$
CandidateRate\leq0.25,
$$

$$
BadEventRate_{\text{candidate}}\leq0.20.
$$

H3 失败标准：

```text
cheap recall < 0.60
or candidate rate must be too high
or bad-event remains uncontrolled
```

如果 H3 失败，则直接用 branch-logit delta scan 做 candidate discovery，但 system burden 会更高。

---

## H4：branch-logit linearization / delta-view 可以保留 gap-probe 的 predictive value

H4 认为 expensive gap-probe 的核心信息是 output displacement gap，而不是 full shadow training. 因此 first-order 或 low-rank output delta 可以近似：

$$
\Delta z
\approx
J_\theta \Delta\theta.
$$

H4 成立标准：

至少一种 branch-logit delta candidate 达到：

$$
AUC_{\text{safe-good}}\geq0.70
$$

or:

$$
Corr_{\text{safe-grounded}}\geq0.35.
$$

并且：

$$
Agreement_{\text{exact-gap}}\geq0.90.
$$

H4 失败标准：

```text
all cheap / linear / cached / low-rank branch-logit deltas AUC < 0.60
or exact-gap agreement < 0.80
```

---

## H5：coverage failure 来自 support imbalance，不是单纯 threshold

P3/P7 显示 coverage 与 safety/precision互相冲突。H5 认为 controller failure 的根因是 support/family imbalance。

H5 成立标准：

Family-balanced controller 比 non-balanced controller：

$$
Coverage_{\text{balanced}}-Coverage_{\text{plain}}\geq0.02,
$$

且：

$$
BadEventRate_{\text{balanced}}\leq0.05.
$$

H5 失败标准：

```text
family balance 只降低 coverage，不提升 safety / precision
```

---

## H6：如果 branch-logit delta controller 本地 pass 但 LDO/LSO fail，不能 dataset tuning

如果 local heldout pass 但 LDO/LSO fail，下一步只允许扩展 signal strata / family support，不能按 dataset name 写 route。

---

# Part IV. 并行执行设计

v9.2.51 runner 必须并行执行以下 lanes：

```text
Lane A:
  v9.2.50 boundary reproduction

Lane B:
  M0-logit preparation subphase attribution

Lane C:
  branch-logit delta kernel candidates:
    delta-view exact
    linearized JVP
    low-rank output sketch
    cached branch baseline
    top-class subset delta
    fused multi-branch delta

Lane D:
  cheap recall generator:
    branch ratio
    risk LCB
    family reliability
    signal aux
    role gap
    support density

Lane E:
  coverage-preserving cascade:
    cheap candidate discovery
    branch-delta confirm
    risk gate
    support/family balance

Lane F:
  support expansion:
    natural rows
    balanced diagnostic rows
    strata/family coverage

Lane G:
  LDO/LSO

Lane H:
  paired replay scout and official replay

Lane I:
  short-run scout if paired replay passes

Lane J:
  system audit:
    step q90
    memory ratio
    kernel count
    sync count
    read/write MB
    temp allocation
```

Gate discipline：

```text
diagnostic rows may be measured early
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  branch-logit delta predictivity pass
  system gate pass
  controller pass
  LDO/LSO pass
```

---

# Part V. Candidate designs

## 1. Base and carrier

Official base remains：

```text
R2-LQ-fanin-output-scale-confirmed
```

Carrier candidates：

```text
A0-current-v9250
A3-RoleWiseFT7ResetCarrier
A6-HybridRoleControlRiskCarrier
```

A1 risk-bounded carrier remains diagnostic unless it passes carrier-active gate：

$$
r_{z,\text{tail}}\geq0.10,
$$

$$
r_{\perp,\text{tail}}\geq0.10.
$$

---

## 2. Branch-logit delta candidates

### BLD0：Exact gap-probe reference

Reference only. Expected predictive but too expensive.

### BLD1：DeltaViewExactBranchLogits

Avoid full parameter copy by using reversible parameter delta view：

$$
\theta_b = \theta + \Delta\theta_b.
$$

Forward computes branch logits using shared base cache and branch delta view. It is exact or near-exact but should reduce materialization.

Pass condition：

```text
metric_error <= 1e-5
step ratio improves >= 30% vs exact reference
```

### BLD2：Linearized Logit Delta JVP

Approximate：

$$
z_{\theta+\Delta\theta_b}
\approx
z_\theta
+
J_\theta \Delta\theta_b.
$$

Because LQ / FullEdge manual path already owns edge-wise local derivatives, JVP should be constructed without full autograd graph.

Gap：

$$
S_{\text{gap-linear}}
=
\widehat{Gain}_{F,\text{linear}}
-
\max(
\widehat{Gain}_{AdamW,\text{linear}},
\widehat{Gain}_{bestLR,\text{linear}}
).
$$

### BLD3：Low-Rank Output Delta Sketch

Maintain train-stream output displacement sketch $U_k$：

$$
\widehat{\Delta z}
=
U_k U_k^T \Delta z.
$$

Use only rank $k \in \{4,8,16\}$.

### BLD4：Top-Class / Tail-Class Delta

Compute delta only for:

```text
true class
current top-1
current top-2
hard negative class
```

This approximates CE/margin/gap without all 10 logits.

### BLD5：Cached Control Baseline Delta

Cache AdamWParallel and bestLR branch gain as EMA：

$$
S_{\text{gap-cache}}
=
\widehat{Gain}_{F}
-
EMA(\max(Gain_{\text{AdamWParallel}},Gain_{\text{bestLR}})).
$$

### BLD6：Fused Multi-Branch Delta Kernel

Evaluate $\Delta z$ for Real / AdamWParallel / bestLR in one fused branch dimension:

```text
input:
  base activations
  branch deltas
  edge basis local derivatives

output:
  branch logits / selected logits / selected metrics
```

### BLD7：Hybrid Delta Cascade

Pipeline：

```text
cheap recall candidate
→ top-class linear delta
→ low-rank delta
→ exact delta-view only for borderline events
```

---

## 3. Cheap recall features

Cheap features are not official accept unless heldout gates pass. They are primarily candidate generators.

### CR0：RoleBranchSignal

Same family as v9.2.50 best cheap surrogate, but used as recall only.

### CR1：RiskTailLCB

$$
S_{\text{risk}}
=
CEp99
+
WrongConfidenceP95
-
MarginP10
+
Uncertainty
+
Curvature.
$$

### CR2：BranchDerivativeMass

```text
branch_ratio_tail
effective_derivative
tail_activation_mass
functional_step_norm / task_step_norm
```

### CR3：FamilyReliability

Family must not include dataset name：

$$
family(e)
=
(stratum,horizon,risk\_bucket,role\_bucket,attach\_type,branch\_bucket,probe\_bucket).
$$

Reliability：

$$
Rel(f)
=
\mathbb{E}[Y_{\text{safe-good}}\mid f]
-
\kappa\sqrt{\operatorname{Var}(Y_{\text{safe-good}}\mid f)}.
$$

### CR4：SignalAux

Signal-channel remains auxiliary. It is forbidden to become official signal-only accept unless bad-event gate passes.

### CR5：SupportDensity

$$
Density(e)=\frac{k}{N\cdot Volume(\mathcal{N}_k(e))}.
$$

---

## 4. Controller candidates

### C0：v9.2.50 reference

Reference only.

### C1：CheapRecallThenBranchDelta

$$
Accept(e)
=
CheapRecall(e)
\land
BranchDeltaConfirm(e)
\land
RiskSafe(e).
$$

### C2：FamilyBalancedBranchDelta

$$
Accept(e)
=
CheapRecall(e)
\land
BranchDeltaConfirm(e)
\land
RiskSafe(e)
\land
Rel(family(e))\geq r_0
\land
FamilyBalance(e).
$$

### C3：TopClassDeltaController

Uses BLD4 selected logits only:

$$
Accept(e)
=
RiskSafe(e)
\land
TopClassGap(e)\geq\tau_g
\land
SupportStable(e).
$$

### C4：LowRankDeltaController

Uses BLD3:

$$
Accept(e)
=
RiskSafe(e)
\land
LowRankGap(e)\geq\tau_g
\land
FamilyReliable(e).
$$

### C5：HybridBorderlineExactController

$$
Accept(e)
=
CheapHighConfidenceAccept(e)
\lor
[
CheapBorderline(e)
\land
ExactDeltaViewConfirm(e)
].
$$

### C6：ParetoCostAwareController

Accept if event is on Pareto frontier of:

```text
low risk
positive branch-delta gap
positive value
high support density
low probe cost
family balance
```

### C7：Oracle

Posthoc diagnostic only. Never official.

---

# Part VI. 实验阶段

## P0：v9.2.50 boundary reproduction

### 目标

确认 v9.2.50 boundary 稳定。

### 必须记录

```text
route
source_route_v9249
dominant_metric_subphase
metric_kernel_compression_pass
best_metric_kernel
step_ratio
controller_agreement
event_sparse_cascade_pass
cascade_recall
cascade_coverage
cascade_precision
cascade_bad_event
cheap_surrogate_pass
cheap_surrogate_auc
signal_aux_pass
signal_aux_bad_event
oracle_support_pass
oracle_precision
oracle_coverage
oracle_bad_event
measured_signal_strata
support_region_controller_pass
best_controller
controller_precision
controller_coverage
controller_bad_event
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R10-PredictiveButMetricKernelTooExpensive
oracle support pass = 1
support-region controller pass = 0
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_oracle_high_legal_system_low_ladder.svg
p0_metric_kernel_vs_controller_gap.svg
```

---

## P1：M0 logit-preparation subphase attribution

### 目标

拆开 M0-logit preparation，确认真实成本来自哪里。

### 必须记录

```text
row_id
subphase_id
subphase_name
time_ms
time_ratio
read_MB
write_MB
temp_alloc_MB
kernel_count
sync_count
branch_count
delta_view_used
param_copy_MB
logit_buffer_MB
unknown_fraction
```

Subphases：

```text
L0-base logits reuse
L1-RealFunctional delta preparation
L2-AdamWParallel delta preparation
L3-bestLR delta preparation
L4-shadow parameter / delta view
L5-branch forward preparation
L6-logit buffer allocation
L7-selected-logit extraction
L8-host sync / logging
```

### 判断标准

P1 pass：

```text
unknown_fraction <= 0.10
dominant_logit_subphase_identified = 1
M0 attribution sums to F7 within ±0.05
```

### 可视化

```text
p1_m0_logit_prep_waterfall.svg
p1_logit_prep_memory_traffic.svg
p1_branch_count_vs_time.svg
```

---

## P2：Branch-logit delta kernel matrix

### 目标

并行测试 BLD1-BLD7，替代 full branch forward gap-probe。

### 必须记录

```text
branch_delta_id
uses_full_branch_forward
uses_autograd_graph
uses_dataset_name
uses_validation
uses_test
uses_posthoc_commit
metric_error_max
gap_error_mean
gap_error_p95
accept_agreement
AUC_safe_good
corr_safe_grounded
precision_at_gate
coverage_at_gate
bad_event_at_gate
per_probe_overhead_q90
amortized_overhead
step_ratio_q90
memory_ratio
read_MB
write_MB
kernel_count
sync_count
```

### 判断标准

Branch-delta predictivity pass：

$$
AUC_{\text{safe-good}}\geq0.70
$$

or:

$$
Corr_{\text{safe-grounded}}\geq0.35.
$$

Decision agreement pass：

$$
Agreement_{\text{accept}}\geq0.90.
$$

System pass：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

Diagnostic pass：

$$
AUC\geq0.60
$$

and:

$$
StepRatio_{q90}\leq2.00.
$$

### 可视化

```text
p2_branch_delta_auc_cost_pareto.svg
p2_branch_delta_gap_error.svg
p2_branch_delta_accept_agreement.svg
p2_branch_delta_memory_traffic.svg
```

---

## P3：Cheap recall generator matrix

### 目标

让 cheap features 做 candidate recall，而不是 official accept。

### 必须记录

```text
recall_id
features_used
AUC_safe_good
recall_safe_good
candidate_rate
candidate_bad_event
candidate_precision
candidate_coverage
feature_overhead
memory_overhead
dataset_name_used
posthoc_used_at_commit
```

### 判断标准

Recall generator pass：

$$
Recall_{\text{safe-good}}\geq0.80,
$$

$$
CandidateRate\leq0.25,
$$

$$
BadEventRate_{\text{candidate}}\leq0.20.
$$

### 可视化

```text
p3_recall_candidate_rate_pareto.svg
p3_recall_bad_event_curve.svg
p3_feature_ablation_recall.svg
```

---

## P4：Coverage-preserving cascade controller

### 目标

用 cheap recall + branch-delta confirm + risk/support balance 形成 official local controller。

### Split design

```text
calibration split:
  choose thresholds / coefficients

heldout natural split:
  official local controller gate

balanced diagnostic split:
  support/failure analysis only

leave-dataset-out:
  P6

leave-stratum-out:
  P6
```

### 必须记录

```text
controller_id
branch_delta_id
cheap_recall_id
features_used
thresholds
coefficients
calibration_split_id
heldout_split_id
precision_cal
coverage_cal
bad_event_cal
precision_heldout
coverage_heldout
bad_event_heldout
AUC_heldout
corr_heldout
safe_good_recall
accepted_strata_count
accepted_family_count
max_family_share
amortized_overhead
step_q90
memory_ratio
dataset_name_used
posthoc_used_at_commit
validation_used
test_used
official_eligible
```

### 判断标准

Controller pass：

$$
AUC_{\text{heldout}}\geq0.70
$$

or:

$$
Corr_{\text{heldout}}\geq0.35.
$$

and:

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{heldout}}\leq0.05,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

Support:

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

### 可视化

```text
p4_controller_precision_coverage_bad.svg
p4_controller_cost_vs_value.svg
p4_controller_family_coverage.svg
p4_cascade_stage_sankey.svg
p4_oracle_legal_gap.svg
```

---

## P5：Online support and stratum expansion

### 目标

扩大 natural / balanced support，避免 measured strata 太窄。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7
horizons = 20,80,240,640
signal_strata = S1-S8
carriers = A0,A3,A6
row_sources = natural, balanced_diagnostic
```

### 必须记录

```text
row_source
row_id
dataset
seed
horizon
signal_stratum
event_family
carrier_id
branch_delta_id
safe_good
bad_event
oracle_accept
controller_accept
risk_safe
value_positive
control_resistant
feature_values
```

### 判断标准

Measurement pass：

```text
natural_real_event_count >= 4000
balanced_diagnostic_real_event_count >= 4000
measured_signal_strata_count >= 6
measured_family_count >= 12
```

Official accepted support：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

### 可视化

```text
p5_signal_strata_coverage.svg
p5_family_support_heatmap.svg
p5_oracle_support_by_stratum.svg
p5_natural_vs_balanced_distribution.svg
```

---

## P6：Leave-dataset-out / leave-stratum-out

### 目标

证明 controller 不是 dataset-specific 或 single-stratum overfit。

### 设置

Leave-dataset-out：

```text
calibrate on MNIST + Fashion, evaluate KMNIST
calibrate on MNIST + KMNIST, evaluate Fashion
calibrate on Fashion + KMNIST, evaluate MNIST
```

Leave-stratum-out：

```text
calibrate on all but one signal stratum
evaluate held-out stratum
```

### 必须记录

```text
split_type
heldout
controller_id
branch_delta_id
cheap_recall_id
threshold
precision
coverage
bad_event_rate
task_safe
CEp99_delta
margin_delta
ECE_delta
NLL_delta
curvature_delta
beats_adamwparallel
beats_bestlr
dataset_name_used
shuffle_control_pass
```

### 判断标准

LDO pass：

$$
Acc_{\text{heldout,Real}}\geq Acc_{\text{heldout,AdamW}}-0.005.
$$

At least two held-out datasets：

$$
BeatRate_{\text{Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{Real vs bestLR}}\geq0.50.
$$

LSO pass：

At least $70\%$ held-out strata task-safe and：

$$
CEp99_{\text{heldout,Real}}\leq CEp99_{\text{AdamW}}+\epsilon,
$$

$$
BeatRate_{\text{heldout-stratum,Real vs AdamWParallel}}\geq0.50.
$$

### 可视化

```text
p6_leave_dataset_out_matrix.svg
p6_leave_stratum_out_matrix.svg
p6_hidden_dataset_tuning_audit.svg
p6_leaveout_failure_modes.svg
```

---

## P7：Official paired replay

### 目标

验证 RealFunctional 是否在 strong controls 下有局部因果优势。

### 设置

```text
base = R2 repaired base checkpoint
branch_delta_id = best P6 survivor
controller_id = best P6 survivor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random,
           ShuffledBranchDelta, ShuffledCheapRecall, ShuffledRiskScore,
           ShuffledBranchRatio, ShuffledControlGap, ShuffledValueScore,
           ShuffledSignalChannel, FunctionalChannelShuffled,
           TailMaskShuffled, RoleScoreShuffled, DatasetRouteShuffled,
           EventRouteShuffled, InvertedRoleMask
```

### 必须记录

```text
controller_id
branch_delta_id
dataset
seed
horizon
signal_stratum
event_family
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
step_q90
memory_ratio
base_checkpoint_hash
```

### 判断标准

Paired replay pass：

$$
BeatRate_{\text{macro,Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro,Real vs bestLR}}\geq0.60.
$$

Task safety：

$$
Acc_{\text{each slice,Real}}\geq Acc_{\text{AdamW}}-0.005.
$$

Shuffle controls fail：

```text
ShuffledBranchDelta = fail
ShuffledCheapRecall = fail
ShuffledRiskScore = fail
ShuffledBranchRatio = fail
ShuffledControlGap = fail
ShuffledValueScore = fail
ShuffledSignalChannel = fail
FunctionalChannelShuffled = fail
TailMaskShuffled = fail
RoleScoreShuffled = fail
DatasetRouteShuffled = fail
EventRouteShuffled = fail
InvertedRoleMask = fail
```

System：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p7_official_paired_replay_pareto.svg
p7_macro_beat_rate.svg
p7_signal_stratum_win_matrix.svg
p7_shuffle_control_matrix.svg
p7_system_gate_distribution.svg
```

---

## P8：Short-run scout

### 目标

如果 P7 pass，验证局部 paired replay 优势能否在连续训练中保持。

### 设置

```text
steps = 50,240,640
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledBranchDelta
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
base_checkpoint_hash
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

Mechanism pass, at least one：

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamW}},
$$

or:

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamW}},
$$

or:

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

### 可视化

```text
p8_short_run_task_mechanism_pareto.svg
p8_short_run_controls.svg
p8_event_timeline.svg
p8_ce_tail_margin_panel.svg
```

---

## P9：Full 10-seed validation

### 目标

如果 short-run pass，验证 full functional route。

### 设置

```text
epochs = 20
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
baseline = MLP-match
diagnostic baseline = QuadraticFeatureMLP
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledBranchDelta
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
base_checkpoint_hash
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

---

## P10：Robustness / strong baseline / external-ready

### 目标

确认 functional advantage 不是 clean MNIST-family artifact。

### 设置

```text
label_noise = 0.05,0.10,0.20
input_noise = 0.05,0.10
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
strong baselines = MLP-match, hidden-bracket MLP, QuadraticFeatureMLP, repaired LQ AdamW-only
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

---

# Part VII. Required artifacts

```text
run_manifest.json
contract_audit_v9251.csv
p0_v9250_boundary_reproduction.csv
p1_m0_logit_preparation_subphase_attribution.csv
p2_branch_logit_delta_kernel_matrix.csv
p3_cheap_recall_generator_matrix.csv
p4_coverage_preserving_cascade_controller.csv
p5_online_support_stratum_expansion.csv
p6_leave_dataset_and_stratum_out.csv
p7_official_paired_replay.csv
p8_short_run_functional_validation.csv
p9_full_10seed_functional_validation.csv
p10_robustness_external_ready.csv
m0_logit_prep_trace_v9251.csv
branch_logit_delta_trace_v9251.csv
cheap_recall_trace_v9251.csv
cascade_controller_trace_v9251.csv
support_density_trace_v9251.csv
leaveout_trace_v9251.csv
paired_replay_branch_trace_v9251.csv
system_branch_delta_overhead_trace_v9251.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9250_boundary_unstable
F3_dataset_tuning_detected
F4_m0_subphase_unattributed
F5_branch_logit_delta_not_predictive
F6_branch_logit_delta_not_system_legal
F7_branch_logit_delta_agreement_fail
F8_cheap_recall_generator_fail
F9_cascade_precision_fail
F10_cascade_coverage_fail
F11_cascade_bad_event_fail
F12_support_measurement_too_narrow
F13_oracle_support_collapse
F14_leave_dataset_out_fail
F15_leave_stratum_out_fail
F16_paired_replay_control_equivalent
F17_shuffle_control_pass
F18_functional_lr_equivalent
F19_short_run_task_drop
F20_full_run_no_macro_hard_stratum_gain
F21_strong_baseline_explains_gain
F22_robustness_fail
F23_external_not_ready
F24_fake_or_proxy_violation
F25_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-M0LogitPrepAttributed:
  M0 logit preparation cost is fully attributed.

R2-BranchLogitDeltaPredictive:
  at least one branch-logit delta kernel predicts safe-good.

R3-BranchLogitDeltaSystemPass:
  branch-logit delta kernel passes system envelope.

R4-CheapRecallGeneratorPass:
  cheap features serve as high-recall candidate generator.

R5-CascadeControllerPass:
  coverage-preserving cascade passes heldout precision / coverage / bad-event / system gate.

R6-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R7-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R8-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R9-BranchDeltaPredictiveButTooExpensive:
  delta kernel predicts safe-good but remains too expensive.

R10-BranchDeltaCheapButUninformative:
  delta kernel is cheap but loses safe-good signal.

R11-CheapRecallFailOracleHigh:
  oracle support exists but cheap recall cannot find enough candidates.

R12-OracleSupportCollapse:
  fresh natural replay no longer has enough safe-good support.

R13-ControllerStillCoverageLimited:
  precision and safety are acceptable but coverage remains below 0.03.

R14-ControllerUnsafe:
  coverage exists but bad-event remains above 0.05.

R15-StrictPureKANFunctionalShortRunPass:
  short-run task-safe mechanism gain.

R16-StrictPureKANFunctionalFullPass:
  full 10-seed macro / hard-stratum / geometry gain.

R17-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` 必须记录：

```text
route
v9250_boundary_pass
dataset_tuning_detected
m0_subphase_attribution_pass
dominant_m0_subphase
best_branch_delta_id
branch_delta_predictivity_pass
branch_delta_system_pass
branch_delta_auc
branch_delta_corr
branch_delta_accept_agreement
branch_delta_step_ratio_q90
branch_delta_memory_ratio
best_cheap_recall_id
cheap_recall_pass
safe_good_recall
candidate_rate
candidate_bad_event
best_controller_id
cascade_controller_pass
controller_auc
controller_corr
accepted_precision
accepted_coverage
accepted_bad_event_rate
accepted_signal_strata_count
accepted_family_count
max_family_share
oracle_support_pass
oracle_precision
oracle_coverage
oracle_bad_event
leave_dataset_out_pass
leave_stratum_out_pass
paired_replay_pass
short_run_pass
full_run_pass
external_ready
primary_blocker
next_required_implementation
success_v9251_strict_purekan_functional
success_v9251_full_functional
success_v9251_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 M0 logit preparation subphase attribution
  P2 branch-logit delta kernel matrix
  P3 cheap recall generator matrix

Batch 2:
  P4 coverage-preserving cascade controller
  P5 online support / stratum expansion
  P6 leave-dataset-out / leave-stratum-out
  P7 paired replay scout

Batch 3:
  official P7 paired replay
  P8 short-run if paired replay passes

Batch 4:
  P9 full 10-seed
  P10 robustness / strong baseline
```

Gate rule：

```text
P4/P7 diagnostic rows may be measured before all gates finish.
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  branch-logit delta predictivity pass
  branch-logit delta system gate pass
  cascade controller pass
  LDO/LSO pass
```

---

# Part X. 停止条件

## Minimum diagnostic success

```text
v9.2.50 boundary reproduced
M0 logit preparation cost attributed
branch-logit delta matrix measured
cheap recall generator measured
support expansion measured
no fake/proxy/offload/loss/teacher violation
```

## Branch-logit system success

```text
Minimum diagnostic success
+
at least one branch-logit delta kernel predicts safe-good
+
system overhead gate pass
+
accept agreement pass
```

## Legal controller success

```text
Branch-logit system success
+
cheap recall or direct branch-delta scan finds sufficient candidates
+
cascade controller heldout pass
+
multi-stratum / multi-family pass
```

## Local functional success

```text
Legal controller success
+
LDO / LSO pass
+
official paired replay beats AdamWParallel / bestLR
```

## Full functional success

```text
Local functional success
+
short/full run task-safe mechanism gain
+
macro or hard-stratum improvement
```

## Failure stop

```text
1. v9.2.50 boundary cannot be reproduced；
2. M0 cost cannot be attributed；
3. all branch-logit delta kernels fail safe-good predictivity；
4. all branch-logit delta kernels remain too expensive；
5. cheap recall cannot find enough safe-good candidates；
6. cascade controller cannot simultaneously meet precision / coverage / bad-event；
7. online support remains too narrow；
8. fresh natural oracle support collapses；
9. leave-dataset-out fails；
10. leave-stratum-out fails；
11. paired replay remains control-equivalent；
12. shuffle controls pass, indicating overfit；
13. short-run task drops；
14. full run gives no macro / hard-stratum / geometry gain；
15. functional breaks system gate；
16. gains are explained by QuadraticFeatureMLP；
17. any teacher/loss/fake/proxy/offload violation occurs。
```

---

# Part XI. 最终解释规则

## Case A：branch-logit delta + controller + LDO/LSO + paired replay pass

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full validation and external robustness.

## Case B：branch-logit delta predictive but too expensive

必须声明：

```text
safe-good is observable through branch-output displacement, but branch-delta kernel is not system-legal.
```

下一步做 deeper kernelization / fused branch delta / CUDA-level implementation。

## Case C：branch-logit delta cheap but uninformative

必须声明：

```text
linearized / low-rank branch delta loses the safe-good signal.
```

下一步回到 exact compressed probe or richer train-stream sufficient statistics.

## Case D：cheap recall fails but oracle high

必须声明：

```text
good events exist, but cheap candidate generator cannot find them.
```

下一步设计 richer recall features; do not tune dataset.

## Case E：controller local pass but LDO/LSO fail

必须声明：

```text
controller is not dataset-agnostic or stratum-agnostic enough.
```

不能用 dataset-specific tuning 写成功。

## Case F：oracle support collapses

必须声明：

```text
fresh natural replay no longer has enough safe-good support.
```

下一步回到 carrier/support mechanism.

---

# Part XII. 最终建议

v9.2.51 的一句话策略是：

$$
\boxed{
\text{不要继续只压缩 scalar metrics；真正压缩 branch logits，用 branch-logit delta kernel 恢复 system-legal safe-good controller。}
}
$$

当前最关键的问题不是：

```text
base 是否稳定；
attach 是否污染；
carrier 是否 silent；
gap_probe 有没有 AUC；
signal-channel 是否有一点 AUC；
C3/C5 threshold 是否差一点；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. M0 logit preparation 的真实子瓶颈是什么？
2. 多分支 logits 能否通过 delta-view / linearized JVP / low-rank sketch / selected-class delta 低成本估计？
3. branch-logit delta 是否保留 v9.2.48 gap_probe 的 safe-good predictivity？
4. cheap surrogate 能否作为 high-recall candidate generator？
5. cascade controller 能否同时满足 precision / coverage / bad-event / system？
6. support 是否能扩展到 >=6 measured strata？
7. controller 能否 LDO/LSO？
8. official paired replay 能否打过 AdamWParallel / bestLR？
```

v9.2.51 的结果将给出清晰分叉：

```text
if branch-logit delta + cascade + LDO/LSO + paired replay pass:
  strict PureKAN functional obtains local causal evidence.

if branch-logit delta predicts but too expensive:
  kernelization remains blocker.

if branch-logit delta cheap but loses signal:
  exact probe information cannot be compressed by current approximation.

if cheap recall fails:
  legal candidate discovery is blocker.

if controller local pass but leave-out fails:
  no dataset tuning; broaden signal-stratum/family support.

if oracle support collapses:
  carrier/support stability is blocker.
```
