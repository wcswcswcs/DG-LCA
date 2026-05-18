# DG-KAN v9.2.50 Metric-Kernel Compression 与 Coverage-Preserving Cascade Controller 完整实验计划

> 本计划基于 v9.2.49 `Cost-Amortized Online MicroProbe Kernelization 与 Cheap-Gap Controller Closure` 的真实复盘制定。  
> v9.2.49 的 terminal route 是：
>
> ```text
> route = R5-SignalChannelSketchPass
> base_candidate = LQ-t2-h256
> success_v9249_strict_purekan_functional = False
> success_v9249_full_functional = False
> success_v9249_external_ready = False
> ```
>
> v9.2.49 的关键事实是：
>
> ```text
> P1 cost attribution:
>   pass = 1
>   dominant phase = F7-metric computation CE/margin/risk/gap
>   dominant time ratio = 0.554049
>
> P2 event-sparse amortized probe:
>   pass = 0
>   best amortized overhead = 0.204674 > 0.20
>   accepted coverage = 0.000980
>
> P3 cheap gap surrogate:
>   pass = 0
>   best = CG5-FamilyReliabilityLCB
>   AUC = 0.657509
>   corr = 0.273813
>   precision = 0.326087
>
> P4 kernelized / fused probe:
>   pass = 0
>   K5 cheap smoke overhead = 0.05 but not equivalent
>   K6 amortized equivalent but overhead = 0.204674 > 0.20
>
> P5 signal-channel sketch:
>   diagnostic pass = 1
>   best = S4-NoiseReservoirEnergy
>   AUC = 0.609584
>   bad-event = 0.510870
>   official success = 0
>
> P6 oracle support:
>   oracle precision = 1.0
>   oracle coverage = 0.137255
>   oracle bad-event = 0.0
>   measured signal strata = 2
>   support measurement pass = 0
>
> P7 controller:
>   pass = 0
>   best = C3-SignalSketchBest
>   precision = 0.242424
>   coverage = 0.021569
>   bad-event = 0.424242
>
> current blocker:
>   signal_channel_predictive_but_controller_failed
> ```
>
> 因此 v9.2.50 的核心不是继续证明 `gap_probe` 有预测性，也不是继续把 signal-channel sketch 当成 official controller。  
> 现在的本质问题是：
>
> $$
> \boxed{
> \text{safe-good support 存在，online probe 可预测，但当前 metric path 太贵，cheap/signal surrogate 不足以形成安全且有覆盖的 controller。}
> }
> $$
>
> v9.2.50 必须同时解决两个耦合问题：
>
> $$
> \boxed{
> \text{F7 metric computation 的系统压缩}
> }
> \quad+\quad
> \boxed{
> \text{coverage-preserving cascade controller}
> }.
> $$
>
> 不能只把 overhead 从 `0.204674` 调到 `0.199`，因为 P2 的 coverage 只有 `0.000980`。  
> 也不能只提高 coverage，因为 P7 的 best controller bad-event 达到 `0.424242`。  
> 本轮要做的是：**保留 v9.2.48 gap-probe 的 safe-good 可观测性，同时把成本降到 system gate 内，并把 accepted coverage 稳定推回 `[0.03,0.15]`。**

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
  if dataset == Fashion: use cheaper metric kernel A
  if dataset == KMNIST: use threshold B
  if dataset == MNIST: abstain
  tune threshold separately per dataset
  use validation/test metric at commit time
  use posthoc replay outcome at commit time for official controller
```

---

# Part I. 对 v9.2.49 的独立判断

## 1. v9.2.49 没有达到目标

v9.2.49 没有达到 strict PureKAN functional success。原因不是 full-run 失败，而是更前面的 online controller gate 没有闭合：

```text
system-legal online feature pass = 0
support-region controller pass = 0
support measurement pass = 0
LDO / LSO / paired replay = not_run
```

`route = R5-SignalChannelSketchPass` 容易让人误解为接近成功。我的判断更严格：这只是 **diagnostic signal-channel pass**，不是 official controller pass。因为 best signal sketch 的 bad-event 是 `0.510870`，远高于 `0.05`，不能作为安全 controller。P7 的 best controller 也只有 precision `0.242424`，coverage `0.021569`，bad-event `0.424242`。这些都不能进入 leave-out 或 paired replay。

## 2. v9.2.49 的真实进展

v9.2.49 有进展，而且进展很明确。

第一，它把 v9.2.48 的 `probe too expensive` 问题定位到具体 phase。F7 metric computation CE/margin/risk/gap 占比 `0.554049`。这说明主要成本不只是 “shadow branch 太多” 的笼统说法，而是 metric computation path 本身太重。

第二，event-sparse amortized probe 距离 system gate 非常近：

$$
0.204674 - 0.20 = 0.004674.
$$

这说明 system 侧不是完全不可修。但由于 accepted coverage 只有 `0.000980`，不能把它当成 near-success。它是 “系统接近、coverage 崩”。

第三，cheap gap surrogate 有弱信号：AUC `0.657509`，corr `0.273813`，但 precision `0.326087` 不够。这说明 cheap surrogate 适合做 high-recall prefilter 或 feature component，不适合直接作为 official accept rule。

第四，signal-channel sketch 从 v9.2.48 的弱信号推进到了 diagnostic pass，best AUC `0.609584`。但 bad-event `0.510870` 说明它捕捉到的是某种 signal/noise 结构，不是 safe-good decision boundary。它不能替代 gap-probe，只能作为辅助特征。

第五，oracle support 仍然强：

$$
Precision_{\text{oracle}}=1.0,
$$

$$
Coverage_{\text{oracle}}=0.137255,
$$

$$
BadEvent_{\text{oracle}}=0.
$$

这说明 safe-good events 仍存在，而且理论上 coverage 充足。当前失败是 legal/system/controller 没找到它们，不是 functional event 不存在。

## 3. 当前真正 blocker

当前 blocker 不是：

```text
base 不稳；
attach 污染；
carrier silent；
online feature 完全无信号；
oracle support 消失；
signal-channel 思路完全无效。
```

当前 blocker 是：

$$
\boxed{
\text{safe-good support 存在，但 system-legal controller 无法同时保住 precision、coverage、bad-event。}
}
$$

更具体地说，当前有三个耦合 blocker：

### 3.1 Metric-kernel blocker

F7 metric computation 是主耗时：

$$
TimeRatio_{F7}=0.554049.
$$

如果 F7 不拆解、不融合、不近似，`gap_probe` 仍然难以 system-legal。

### 3.2 Coverage blocker

P2 event-sparse best coverage 只有：

$$
Coverage=0.000980.
$$

P7 best controller coverage 也只有：

$$
Coverage=0.021569<0.03.
$$

所以即使 overhead 接近 gate，也不够。下一步必须让 probe 调用或 cheap controller 覆盖到足够多 safe-good family。

### 3.3 Safety blocker

Signal-channel / signal-sketch controller 的 bad-event 很高：

$$
BadEvent_{\text{S4}}=0.510870,
$$

$$
BadEvent_{\text{C3}}=0.424242.
$$

这说明 signal-channel 不能单独作为 accept 规则。它可以作为 support/risk auxiliary，但必须被 RiskSafe 和 control-gap confirmation 约束。

## 4. 为什么进展仍显缓慢

进展慢是真实的，但 v9.2.49 已经把问题从 “online probe 是否有信号” 推进到 “如何让有信号的 probe system-legal and coverage-legal”。这比 v9.2.47 之前清楚很多。

现在不能继续一轮只试一个 controller。v9.2.50 必须并行：

```text
1. F7 metric subphase decomposition；
2. streaming / fused metric kernel；
3. quantile / top-k approximation；
4. cheap surrogate as recall prefilter；
5. event-sparse cascade；
6. support/family coverage repair；
7. controller heldout gate；
8. LDO / LSO；
9. paired replay scout。
```

如果还继续 “再调 C3/C5 threshold”，进展会继续慢，而且会偏离本质。

## 5. 是否在正确道路上

是，但路线必须进一步收紧：

```text
online gap-probe predictivity 已证明；
oracle safe-good support 仍存在；
现在必须做 metric-kernel compression + cascade controller。
```

不能回到：

```text
basis sweep；
dataset-specific controller；
offline autopsy；
signal-channel 单独 official promotion；
只调 threshold；
只看 AUC。
```

尤其是 signal-channel。v9.2.49 的 R5 不能被解读为 “signal-channel 已经可用”。正确解读是：

$$
\boxed{
\text{signal-channel 有 diagnostic signal，但安全边界不成立；它是辅助，不是主 controller。}
}
$$

---

# Part II. v9.2.50 总体目标

v9.2.50 的总体目标是：

$$
\boxed{
\text{把 predictive but expensive gap-probe 转化为 metric-compressed、coverage-preserving、system-legal 的 online functional controller。}
}
$$

目标分为八层。

## 1. F7 metric subphase attribution success

必须进一步拆解 F7 metric computation：

```text
CE mean / CE tail
CEp99 quantile
margin p10
wrong confidence p95
risk score
gap score
branch reduction
top-k / sorting
temporary allocation
synchronization
logging
```

成功标准：

```text
F7 unknown subphase fraction <= 0.10
dominant F7 subphase identified = 1
each subphase time / memory / kernel count recorded
```

如果 F7 的主要成本来自 quantile/sort，则优先做 streaming quantile / top-k approximation。  
如果来自 branch reductions，则优先做 fused metric kernel。  
如果来自 temporary allocation，则优先做 preallocated workspace / in-place reductions。

## 2. Metric-kernel compression success

v9.2.50 必须把 F7 metric computation 压缩。候选包括：

```text
streaming CEp99 approximation
top-k CE tail
histogram margin quantile
fused CE/margin/risk kernel
preallocated metric workspace
no-sync batched metric logging
branch-shared metric reduction
```

Metric-kernel success：

$$
F7Time_{compressed}\leq0.50F7Time_{reference}.
$$

Probe overhead success：

$$
per\_probe\_overhead_{q90}\leq2.0
$$

or amortized：

$$
Overhead_{\text{amortized}}\leq0.20.
$$

Numerical sanity：

$$
|\Delta CEp99|\leq\epsilon_{ce},
$$

$$
|\Delta MarginP10|\leq\epsilon_m,
$$

$$
|\Delta RiskScore|\leq\epsilon_r.
$$

Practical suggested tolerances：

```text
epsilon_ce = 0.02
epsilon_m = 0.02
epsilon_r = 0.05
```

The approximate metric does not need bitwise equivalence, but it must preserve controller decisions:

$$
Agreement_{\text{accept}} \geq 0.90
$$

on calibration rows.

## 3. Event-sparse cascade success

v9.2.49 event-sparse was almost system-legal but coverage collapsed. v9.2.50 must change the cascade objective from “minimize probe calls” to “maximize safe-good recall under system budget”.

Define:

$$
p_{\text{probe}}=\frac{\#\text{probe-called rows}}{\#\text{all rows}}.
$$

Amortized overhead:

$$
Overhead_{\text{amortized}}
=
p_{\text{probe}}
\cdot
Overhead_{\text{per-probe}}
+
Overhead_{\text{prefilter}}.
$$

Cascade success requires:

$$
Overhead_{\text{amortized}}\leq0.20,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

But also:

$$
Recall_{\text{safe-good,prefilter}}\geq0.80,
$$

$$
Coverage_{\text{accepted}}\in[0.03,0.15],
$$

$$
Precision_{\text{accepted}}\geq0.75,
$$

$$
BadEventRate_{\text{accepted}}\leq0.05.
$$

This prevents the old failure mode:

```text
system near-pass but coverage 0.000980
```

## 4. Cheap surrogate as prefilter, not official accept rule

Cheap gap surrogate failed official pass, but its AUC `0.657509` may be enough for prefiltering.

Prefilter pass:

$$
AUC_{\text{cheap}}\geq0.60,
$$

$$
Recall_{\text{safe-good}}\geq0.80,
$$

$$
BadEventRate_{\text{prefilter}}\leq0.15,
$$

$$
ProbeCallRate\leq0.10.
$$

Official accept still requires compressed or amortized probe confirmation:

$$
Accept(e)
=
CheapPrefilter(e)
\land
CompressedGapConfirm(e)
\land
RiskSafe(e)
\land
SupportStable(e).
$$

## 5. Signal-channel auxiliary success

Signal-channel sketch should be demoted to auxiliary. It can help if it improves support stability or reduces false positives, but it must not be the sole accept rule.

Auxiliary pass:

$$
AUC_{\text{signal}}\geq0.60
$$

or:

$$
\Delta Precision_{\text{controller}}\geq0.05
$$

or:

$$
\Delta BadEventRate_{\text{controller}}\leq-0.05.
$$

But official signal-only fail if:

$$
BadEventRate_{\text{signal-only}}>0.05.
$$

Given v9.2.49 bad-event `0.510870`, signal-only controller should not be promoted unless dramatically changed.

## 6. Stratum / family support success

v9.2.49 measured signal strata count was only `2`. v9.2.50 must expand measurement and accepted support.

Measurement pass:

```text
natural_real_event_count >= 3000
balanced_diagnostic_real_event_count >= 3000
measured_signal_strata_count >= 6
measured_family_count >= 8
```

Official accepted support:

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

This avoids another narrow support success.

## 7. Support-region controller success

Controller form:

$$
Accept(e)
=
CheapPrefilter(e)
\land
MetricCompressedGap(e)
\land
RiskSafe(e)
\land
SupportStable(e)
\land
OptionalSignalAux(e).
$$

Heldout gate:

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{heldout}}\leq0.05.
$$

System gate:

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

## 8. Local functional causality success

Only after P1-P7 pass，LDO/LSO and official paired replay open.

Leave-dataset-out:

$$
Acc_{\text{heldout,Real}}\geq Acc_{\text{heldout,AdamW}}-0.005.
$$

At least two held-out datasets:

$$
BeatRate_{\text{Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{Real vs bestLR}}\geq0.50.
$$

Official paired replay:

$$
BeatRate_{\text{macro,Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro,Real vs bestLR}}\geq0.60.
$$

---

# Part III. 核心假设

## H1：v9.2.49 的失败主要是 metric path implementation，不是 online observability failure

证据：

```text
v9.2.48 gap_probe AUC = 0.884406
v9.2.49 oracle precision = 1.0
v9.2.49 oracle coverage = 0.137255
v9.2.49 oracle bad-event = 0.0
```

H1 成立标准：

```text
F7 metric compression reduces overhead substantially
and controller precision / coverage / bad-event improves without losing safe-good predictivity
```

H1 失败标准：

```text
all compressed / approximated / amortized metric paths lose safe-good predictivity
```

## H2：F7 CE/margin/risk/gap metrics are over-computed

F7 dominates total overhead at `0.554049`. H2 assumes we can approximate or fuse CEp99/margin/risk without losing controller decisions.

H2 成立标准：

$$
F7Time_{compressed}\leq0.50F7Time_{reference}
$$

and:

$$
Agreement_{\text{controller}}\geq0.90.
$$

H2 失败标准：

```text
streaming/top-k/fused metric kernels either remain too expensive or destroy controller agreement
```

## H3：event-sparse failed because prefilter optimized cost too aggressively

v9.2.49 event-sparse got overhead `0.204674` but coverage `0.000980`. H3 says the prefilter was too narrow.

H3 成立标准：

A recall-first prefilter gets:

$$
Recall_{\text{safe-good}}\geq0.80
$$

and accepted coverage:

$$
Coverage\geq0.03
$$

while keeping amortized overhead below `0.20`.

## H4：cheap surrogate is useful as recall prefilter, not final controller

Cheap surrogate precision was too low (`0.326087`), but AUC `0.657509` can still support high-recall prefiltering.

H4 成立标准：

cheap surrogate prefilter keeps:

$$
Recall_{\text{safe-good}}\geq0.80
$$

with:

$$
ProbeCallRate\leq0.10.
$$

H4 失败标准：

```text
cheap surrogate either misses most safe-good events or admits too many bad events
```

## H5：signal-channel sketch should remain auxiliary until it controls bad-event

S4 had AUC `0.609584` but bad-event `0.510870`. H5 says signal-channel is not official unless paired with risk/gap.

H5 成立标准：

signal auxiliary improves final controller precision or bad-event by at least `0.05`.

H5 失败标准：

signal-only or signal-dominant controller keeps bad-event above `0.05`.

## H6：support concentration is still a hidden blocker

Measured strata count is only `2`. H6 says coverage failure is partly due to support concentration.

H6 成立标准：

balanced support expansion raises:

```text
measured_signal_strata_count >= 6
accepted_signal_strata_count >= 2
accepted_family_count >= 4
```

without dataset-specific rules.

## H7：if compressed/amortized probes pass system but paired replay still fails, then microprobe value is not causally independent

If support-region controller passes local heldout but paired replay fails against AdamWParallel / bestLR, then the issue is functional causal equivalence, not probe implementation.

---

# Part IV. 并行执行设计

v9.2.50 runner 必须并行执行以下 lanes：

```text
Lane A:
  v9.2.49 boundary reproduction

Lane B:
  F7 metric subphase attribution:
    CE mean / CEp99 / margin / risk / gap / branch reduction / sort / temp allocation

Lane C:
  metric-kernel compression:
    streaming CEp99
    top-k tail
    histogram quantile
    fused CE-margin-risk-gap kernel
    preallocated metric workspace

Lane D:
  recall-first event-sparse cascade:
    cheap prefilter
    compressed gap confirm
    amortized overhead

Lane E:
  cheap surrogate as prefilter:
    branch ratio
    role-gap
    family reliability
    support density
    cached/linearized gap

Lane F:
  signal-channel auxiliary:
    noise reservoir energy
    SNR
    low-rank signal-channel sketch
    only used as auxiliary unless safety pass

Lane G:
  support expansion:
    natural rows
    balanced diagnostic rows
    family coverage

Lane H:
  support-region controller calibration

Lane I:
  LDO / LSO

Lane J:
  paired replay scout and official replay

Lane K:
  short-run scout if paired replay passes
```

Gate discipline:

```text
diagnostic rows may be measured early
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  online feature predictivity pass
  system gate pass
  controller pass
  LDO/LSO pass
```

---

# Part V. Candidate feature and metric designs

## 1. Base and carrier

Official base remains:

```text
R2-LQ-fanin-output-scale-confirmed
```

Carrier candidates:

```text
A0-current-v9249
A3-RoleWiseFT7ResetCarrier
A6-HybridRoleControlRiskCarrier
```

## 2. Metric kernel candidates

### MK0：Reference F7 gap-probe metrics

Reference only. Expected predictive but too expensive.

### MK1：Streaming CEp99 / MarginP10

Approximate tail metrics using streaming top-k or histogram.

$$
CEp99 \approx \operatorname{TopKMean}_{k=\lceil0.01B\rceil}(CE_i).
$$

$$
MarginP10 \approx \operatorname{BottomKMean}_{k=\lceil0.10B\rceil}(Margin_i).
$$

### MK2：Fused CE-Margin-Risk Kernel

Compute CE, margin, wrong confidence, risk in one pass over logits.

```text
input: logits, labels
output: CEmean, CEtail, MarginP10, WrongConfP95, RiskScore
```

### MK3：Fused Multi-Branch Metric Kernel

Compute Real / AdamWParallel / bestLR probe metrics in one vectorized branch dimension.

### MK4：Preallocated Metric Workspace

Avoid repeated temporary allocation for:

```text
per-sample CE
per-sample margin
top-k buffer
branch metric buffer
```

### MK5：No-Sync Metric Logging

Metrics remain on device until batch aggregation; only aggregate scalars sync.

### MK6：Approximate Gap Metric

Use linearized or cached control branch:

$$
S_{\text{gap-approx}}
=
Gain_{\text{Real,probe}}
-
EMA(
\max(Gain_{\text{AdamWParallel}},Gain_{\text{bestLR}})
).
$$

### MK7：Hybrid MetricCompressedGap

Pipeline:

```text
cheap prefilter
→ fused CE/margin/risk
→ approximate gap
→ exact compressed gap only for borderline rows
```

---

# Part VI. Controller candidates

## C0：v9.2.49 reference

Reference only.

## C1：RecallFirstEventSparseCascade

$$
Accept(e)
=
CheapRecallPrefilter(e)
\land
CompressedGapConfirm(e)
\land
RiskSafe(e).
$$

## C2：MetricCompressedSupportController

$$
Accept(e)
=
MetricCompressedGap(e)
\land
RiskSafe(e)
\land
SupportStable(e).
$$

## C3：CheapPrefilterExactBorderline

$$
Accept(e)
=
CheapHighConfidenceAccept(e)
\lor
[
CheapBorderline(e)
\land
ExactCompressedProbe(e)
].
$$

## C4：FamilyBalancedMetricController

$$
Accept(e)
=
MetricCompressedGap(e)
\land
RiskSafe(e)
\land
Rel(family(e))\geq r_0
\land
FamilyBalance(e).
$$

Family must not include dataset name:

$$
family(e)
=
(stratum,horizon,risk\_bucket,role\_bucket,attach\_type,branch\_bucket,probe\_bucket).
$$

## C5：SignalAuxRiskGapController

Signal-channel is auxiliary only:

$$
Accept(e)
=
RiskSafe(e)
\land
MetricCompressedGap(e)
\land
SupportStable(e)
\land
[
SignalAux(e) \lor CheapValueHigh(e)
].
$$

## C6：ParetoFrontCostAwareController

Accepts event on Pareto front of:

```text
low risk
positive compressed gap
positive value
high support density
low probe cost
family balance
```

## C7：Oracle

Posthoc diagnostic only. Never official.

---

# Part VII. 实验阶段

## P0：v9.2.49 boundary reproduction

### 目标

确认 v9.2.49 boundary 稳定。

### 必须记录

```text
route
source_route_v9248
probe_cost_attribution_pass
dominant_probe_cost_phase
dominant_phase_time_ratio
event_sparse_probe_pass
best_amortized_overhead
best_event_sparse_coverage
cheap_gap_surrogate_pass
cheap_gap_auc
cheap_gap_precision
kernelized_probe_pass
signal_channel_pass
signal_channel_auc
signal_channel_bad_event
oracle_support_pass
oracle_precision
oracle_coverage
oracle_bad_event
support_region_controller_pass
best_controller_precision
best_controller_coverage
best_controller_bad_event
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R5-SignalChannelSketchPass
signal_channel diagnostic pass = 1
support-region controller pass = 0
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_probe_system_controller_ladder.svg
p0_signal_diagnostic_not_controller.svg
```

---

## P1：F7 metric subphase attribution

### 目标

把 F7 metric computation 的 `0.554049` time ratio 分解到 subphase。

### 必须记录

```text
row_id
metric_subphase
time_ms
time_ratio
read_MB
write_MB
temp_alloc_MB
kernel_count
sync_count
sort_or_topk_used
branch_count
unknown_fraction
```

Subphases:

```text
M0-logit preparation
M1-CE per sample
M2-CEp99 / top-tail
M3-margin per sample
M4-margin p10 / quantile
M5-wrong confidence p95
M6-risk score
M7-gap score
M8-branch reduction
M9-temp allocation
M10-host sync / logging
```

### 判断标准

P1 pass：

```text
unknown_fraction <= 0.10
dominant_metric_subphase_identified = 1
phase_time_sum_close_to_F7 within ±0.05
```

### 可视化

```text
p1_f7_metric_cost_waterfall.svg
p1_metric_kernel_count.svg
p1_metric_memory_traffic.svg
```

---

## P2：Metric-kernel compression

### 目标

压缩 F7 metric computation。

### 必须记录

```text
metric_kernel_id
reference_metric_id
CEp99_error
MarginP10_error
RiskScore_error
GapScore_error
controller_decision_agreement
F7_time_ratio
per_probe_overhead
amortized_overhead
step_ratio_q90
memory_ratio
numeric_sanity_pass
system_pass
```

### 判断标准

Metric compression pass：

$$
F7Time_{compressed}\leq0.50F7Time_{reference}.
$$

Controller agreement:

$$
Agreement_{\text{accept}}\geq0.90.
$$

System pass:

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p2_metric_compression_pareto.svg
p2_metric_error_vs_controller_agreement.svg
p2_f7_time_reduction.svg
```

---

## P3：Recall-first event-sparse cascade

### 目标

修复 v9.2.49 event-sparse coverage collapse。

### 必须记录

```text
prefilter_id
probe_candidate
probe_call_rate
safe_good_recall_prefilter
bad_event_rate_prefilter
per_probe_overhead
amortized_overhead
step_ratio_all_q90
memory_ratio
accepted_precision
accepted_coverage
accepted_bad_event_rate
accepted_strata_count
accepted_family_count
max_family_share
```

### 判断标准

Event-sparse cascade pass：

$$
Overhead_{\text{amortized}}\leq0.20,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05,
$$

$$
Recall_{\text{safe-good,prefilter}}\geq0.80,
$$

$$
Precision_{\text{accepted}}\geq0.75,
$$

$$
Coverage_{\text{accepted}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{accepted}}\leq0.05.
$$

### 可视化

```text
p3_probe_call_rate_vs_safe_good_recall.svg
p3_amortized_overhead_vs_coverage.svg
p3_prefilter_controller_pareto.svg
```

---

## P4：Cheap gap surrogate as prefilter

### 目标

验证 cheap surrogate 是否可以作为 recall prefilter。

### 必须记录

```text
surrogate_id
features_used
AUC_safe_good
corr_safe_grounded
precision_at_gate
coverage_at_gate
bad_event_at_gate
recall_safe_good
probe_call_rate_if_prefilter
feature_overhead
memory_overhead
dataset_name_used
posthoc_used_at_commit
```

### 判断标准

Surrogate prefilter pass：

$$
AUC\geq0.60,
$$

$$
Recall_{\text{safe-good}}\geq0.80,
$$

$$
ProbeCallRate\leq0.10.
$$

Official surrogate pass only if:

$$
AUC\geq0.70
$$

or:

$$
Corr\geq0.35
$$

and controller heldout gate pass.

### 可视化

```text
p4_surrogate_recall_precision.svg
p4_surrogate_cost_predictivity.svg
p4_surrogate_feature_ablation.svg
```

---

## P5：Signal-channel auxiliary audit

### 目标

决定 signal-channel 是辅助、废弃、还是重新设计。

### 必须记录

```text
signal_feature_id
AUC_safe_good
corr_safe_grounded
bad_event_at_gate
controller_precision_delta
controller_coverage_delta
controller_bad_event_delta
signal_only_official_eligible
```

### 判断标准

Signal auxiliary pass：

$$
AUC_{\text{signal}}\geq0.60
$$

and one of:

$$
\Delta Precision\geq0.05,
$$

$$
\Delta Coverage\geq0.05,
$$

$$
\Delta BadEventRate\leq-0.05.
$$

Signal-only promotion forbidden if:

$$
BadEventRate>0.05.
$$

### 可视化

```text
p5_signal_aux_ablation.svg
p5_signal_bad_event_curve.svg
p5_signal_vs_gap_probe.svg
```

---

## P6：Online support and stratum expansion

### 目标

扩大 measured support，避免 narrow two-strata conclusion。

### 设置

```text
natural rows:
  official distribution

balanced diagnostic rows:
  stratified by signal_stratum / event_family
  not official unless separated
```

### 必须记录

```text
row_source
row_id
dataset
seed
signal_stratum
event_family
carrier_id
probe_candidate
safe_good
bad_event
oracle_accept
controller_accept
feature_values
```

### 判断标准

Measurement pass：

```text
natural_real_event_count >= 3000
balanced_diagnostic_real_event_count >= 3000
measured_signal_strata_count >= 6
measured_family_count >= 8
```

Accepted support:

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

### 可视化

```text
p6_signal_strata_coverage.svg
p6_family_support_heatmap.svg
p6_oracle_support_by_stratum.svg
```

---

## P7：Support-region controller calibration

### 目标

建立 official system-legal controller。

### Split design

```text
calibration split:
  choose thresholds / monotone coefficients

heldout split:
  official controller gate

balanced diagnostic split:
  feature diagnosis only

leave-dataset-out:
  P8

leave-stratum-out:
  P8
```

### 必须记录

```text
controller_id
metric_kernel_id
surrogate_id
features_used
calibration_split_id
heldout_split_id
thresholds
coefficients
precision_cal
coverage_cal
bad_event_cal
precision_heldout
coverage_heldout
bad_event_heldout
AUC_heldout
corr_heldout
accepted_strata_count
accepted_family_count
max_family_share
feature_overhead
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

### 可视化

```text
p7_controller_precision_coverage_bad.svg
p7_controller_cost_vs_value.svg
p7_controller_family_coverage.svg
p7_oracle_legal_gap.svg
```

---

## P8：Leave-dataset-out / leave-stratum-out

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
metric_kernel_id
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

At least two held-out datasets:

$$
BeatRate_{\text{Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{Real vs bestLR}}\geq0.50.
$$

LSO pass:

At least $70\%$ held-out strata task-safe and:

$$
CEp99_{\text{heldout,Real}}\leq CEp99_{\text{AdamW}}+\epsilon,
$$

$$
BeatRate_{\text{heldout-stratum,Real vs AdamWParallel}}\geq0.50.
$$

### 可视化

```text
p8_leave_dataset_out_matrix.svg
p8_leave_stratum_out_matrix.svg
p8_hidden_dataset_tuning_audit.svg
```

---

## P9：Official paired replay

### 目标

验证 RealFunctional 是否在 strong controls 下有局部因果优势。

### 设置

```text
base = R2 repaired base checkpoint
metric_kernel_id = best P8 survivor
controller_id = best P8 survivor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random,
           ShuffledGapProbe, ShuffledCheapGap, ShuffledRiskScore,
           ShuffledBranchRatio, ShuffledControlGap, ShuffledValueScore,
           ShuffledSignalChannel, ShuffledMetricKernel,
           FunctionalChannelShuffled, TailMaskShuffled, RoleScoreShuffled,
           DatasetRouteShuffled, EventRouteShuffled, InvertedRoleMask
```

### 必须记录

```text
controller_id
metric_kernel_id
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
ShuffledGapProbe = fail
ShuffledCheapGap = fail
ShuffledRiskScore = fail
ShuffledBranchRatio = fail
ShuffledControlGap = fail
ShuffledValueScore = fail
ShuffledSignalChannel = fail
ShuffledMetricKernel = fail
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
p9_official_paired_replay_pareto.svg
p9_macro_beat_rate.svg
p9_signal_stratum_win_matrix.svg
p9_shuffle_control_matrix.svg
p9_system_gate_distribution.svg
```

---

## P10：Short-run scout

### 目标

如果 P9 pass，验证局部 paired replay 优势能否在连续训练中保持。

### 设置

```text
steps = 50,240,640
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledGapProbe
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
p10_short_run_task_mechanism_pareto.svg
p10_short_run_controls.svg
p10_event_timeline.svg
p10_ce_tail_margin_panel.svg
```

---

## P11：Full 10-seed validation

### 目标

如果 short-run pass，验证 full functional route。

### 设置

```text
epochs = 20
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
baseline = MLP-match
diagnostic baseline = QuadraticFeatureMLP
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledGapProbe
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

## P12：Robustness / strong baseline / external-ready

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

# Part VIII. Required artifacts

```text
run_manifest.json
contract_audit_v9250.csv
p0_v9249_boundary_reproduction.csv
p1_f7_metric_subphase_attribution.csv
p2_metric_kernel_compression.csv
p3_recall_first_event_sparse_cascade.csv
p4_cheap_gap_surrogate_prefilter.csv
p5_signal_channel_auxiliary_audit.csv
p6_online_support_stratum_expansion.csv
p7_support_region_controller_calibration.csv
p8_leave_dataset_and_stratum_out.csv
p9_official_paired_replay.csv
p10_short_run_functional_validation.csv
p11_full_10seed_functional_validation.csv
p12_robustness_external_ready.csv
metric_subphase_trace_v9250.csv
metric_kernel_trace_v9250.csv
event_sparse_cascade_trace_v9250.csv
cheap_surrogate_trace_v9250.csv
signal_aux_trace_v9250.csv
support_density_trace_v9250.csv
controller_calibration_trace_v9250.csv
leaveout_trace_v9250.csv
paired_replay_branch_trace_v9250.csv
system_metric_kernel_overhead_trace_v9250.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9249_boundary_unstable
F3_dataset_tuning_detected
F4_f7_subphase_unattributed
F5_metric_kernel_compression_fail
F6_metric_approximation_breaks_controller
F7_event_sparse_prefilter_low_recall
F8_event_sparse_coverage_fail
F9_event_sparse_amortization_fail
F10_cheap_surrogate_prefilter_fail
F11_signal_aux_bad_event_fail
F12_signal_support_too_narrow
F13_controller_fail
F14_oracle_support_collapse
F15_leave_dataset_out_fail
F16_leave_stratum_out_fail
F17_paired_replay_control_equivalent
F18_shuffle_control_pass
F19_functional_lr_equivalent
F20_short_run_task_drop
F21_full_run_no_macro_hard_stratum_gain
F22_strong_baseline_explains_gain
F23_robustness_fail
F24_external_not_ready
F25_fake_or_proxy_violation
F26_artifact_missing
```

---

# Part IX. Route decision

```text
R1-F7MetricCostAttributed:
  F7 subphase cost is fully attributed.

R2-MetricKernelCompressed:
  metric compression reduces F7 cost while preserving controller-relevant metrics.

R3-EventSparseCascadePass:
  recall-first sparse probe passes system and coverage gates.

R4-CheapSurrogatePrefilterPass:
  cheap surrogate becomes legal high-recall prefilter.

R5-SignalChannelAuxiliaryPass:
  signal sketch improves controller as auxiliary without safety failure.

R6-ControllerPass:
  support-region controller passes heldout precision / coverage / bad-event / system gate.

R7-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R8-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R9-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R10-PredictiveButMetricKernelTooExpensive:
  probe remains predictive but metric path cannot meet system envelope.

R11-CheapFeatureFailOracleHigh:
  oracle support exists but legal cheap features still cannot find it.

R12-OracleSupportCollapse:
  fresh natural replay no longer has enough safe-good support.

R13-ControllerCoverageLimited:
  precision and safety are good but coverage remains below 0.03.

R14-SignalChannelUnsafe:
  signal-channel remains predictive but unsafe as controller.

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
v9249_boundary_pass
dataset_tuning_detected
f7_subphase_attribution_pass
dominant_metric_subphase
metric_kernel_compression_pass
metric_kernel_id
f7_time_reduction
metric_error_max
controller_agreement
event_sparse_cascade_pass
probe_call_rate
safe_good_recall_prefilter
amortized_overhead
cheap_surrogate_prefilter_pass
cheap_surrogate_auc
cheap_surrogate_recall
signal_aux_pass
signal_aux_bad_event
measured_signal_strata_count
accepted_signal_strata_count
accepted_family_count
best_controller_id
support_region_controller_pass
controller_auc
controller_corr
accepted_precision
accepted_coverage
accepted_bad_event_rate
step_ratio_q90
memory_ratio
leave_dataset_out_pass
leave_stratum_out_pass
paired_replay_pass
short_run_pass
full_run_pass
external_ready
primary_blocker
next_required_implementation
success_v9250_strict_purekan_functional
success_v9250_full_functional
success_v9250_external_ready
```

---

# Part X. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 F7 metric subphase attribution
  P2 metric-kernel compression
  P3 recall-first event-sparse cascade
  P4 cheap surrogate prefilter
  P5 signal-channel auxiliary audit

Batch 2:
  P6 online support / stratum expansion
  P7 support-region controller calibration
  P8 leave-dataset-out / leave-stratum-out
  P9 paired replay scout

Batch 3:
  official P9 paired replay
  P10 short-run if paired replay passes

Batch 4:
  P11 full 10-seed
  P12 robustness / strong baseline
```

Gate rule：

```text
P7/P9 diagnostic rows may be measured before all gates finish.
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  online feature predictivity pass
  metric/system gate pass
  controller pass
  LDO/LSO pass
```

---

# Part XI. 停止条件

## Minimum diagnostic success

```text
v9.2.49 boundary reproduced
F7 metric subphase attribution completed
metric compression branch measured
event-sparse cascade measured
cheap surrogate prefilter measured
support expansion measured
no fake/proxy/offload/loss/teacher violation
```

## System-legal online feature success

```text
Minimum diagnostic success
+
at least one of:
  metric-kernel compression pass
  event-sparse cascade pass
  cheap surrogate prefilter + compressed confirm pass
+
system overhead gate pass
```

## Legal controller success

```text
System-legal online feature success
+
support-region controller heldout pass
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
1. v9.2.49 boundary cannot be reproduced；
2. F7 cost cannot be attributed；
3. metric compression breaks controller-relevant metrics；
4. event-sparse prefilter recall remains too low；
5. event-sparse coverage remains below 0.03；
6. cheap surrogate cannot serve as prefilter；
7. signal-channel remains unsafe and adds no auxiliary value；
8. online support remains too narrow；
9. all support-region controllers fail heldout gate；
10. leave-dataset-out fails；
11. leave-stratum-out fails；
12. paired replay remains control-equivalent；
13. shuffle controls pass, indicating overfit；
14. short-run task drops；
15. full run gives no macro / hard-stratum / geometry gain；
16. functional breaks system gate；
17. gains are explained by QuadraticFeatureMLP；
18. any teacher/loss/fake/proxy/offload violation occurs。
```

---

# Part XII. 最终解释规则

## Case A：Metric compression + controller pass

可以声明：

```text
v9.2.49 failed because F7 metric path was over-computed; compressed metrics preserved value while making controller system-legal.
```

但不能声明 strict success unless LDO/LSO and paired replay pass.

## Case B：Event-sparse cascade pass

可以声明：

```text
expensive probe was usable after recall-first sparse invocation and compressed confirmation.
```

但仍需 LDO/LSO and paired replay.

## Case C：Cheap surrogate prefilter pass but exact confirm needed

可以声明：

```text
cheap features are sufficient to find candidates but not to commit final accept.
```

下一步应 keep cascade, not promote cheap-only controller.

## Case D：Signal auxiliary helps but signal-only unsafe

必须声明：

```text
signal-channel is useful as auxiliary but not sufficient as controller.
```

不要把 R5 diagnostic pass 写成 official functional success.

## Case E：Compressed/amortized probes still too expensive

必须声明：

```text
safe-good is observable but current metric/probe implementation is not system-legal.
```

下一步做 deeper metric kernelization or lower-cost sufficient statistics.

## Case F：Oracle high but legal controller fail

必须声明：

```text
safe-good support exists but legal feature/controller cannot identify it.
```

下一步设计 richer cheap sufficient statistics.

## Case G：LDO/LSO fail

必须声明：

```text
controller is not dataset-agnostic or stratum-agnostic enough.
```

不能用 dataset-specific tuning 写成功。

## Case H：Paired replay passes

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full validation。

---

# Part XIII. 最终建议

v9.2.50 的一句话策略是：

$$
\boxed{
\text{不要把 signal-channel diagnostic pass 当成功；要压缩 F7 metric path，并用 recall-first cascade 恢复 coverage。}
}
$$

当前最关键的问题不是：

```text
base 是否稳定；
attach 是否污染；
carrier 是否 silent；
gap_probe 有没有 AUC；
signal-channel 有没有一点 AUC；
C3 threshold 是否差一点；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. F7 metric computation 中到底哪一个 subphase 最贵？
2. CEp99 / margin / risk / gap 是否可以 streaming / fused / approximate？
3. metric compression 是否能保持 accept decision agreement？
4. cheap surrogate 是否能作为 high-recall prefilter？
5. event-sparse probe 是否能在 recall-first 设置下同时保住 coverage 和 overhead？
6. signal-channel 是否只能作为辅助，而不能 official？
7. online support 是否能从 2 个 signal strata 扩到 >=6 个 measured strata？
8. controller 能否 heldout pass，并通过 LDO/LSO？
9. official paired replay 能否打过 AdamWParallel / bestLR？
```

v9.2.50 的结果将给出清晰分叉：

```text
if metric-compressed cascade + controller + LDO/LSO + paired replay pass:
  strict PureKAN functional obtains local causal evidence.

if metric compression works but controller fails:
  support-region geometry / controller calibration remains blocker.

if controller works but LDO/LSO fails:
  no dataset tuning; broaden signal/family support.

if all compressed/amortized probes fail system:
  online safe-good is observable but not system-legal.

if cheap features fail while oracle remains high:
  legal sufficient statistics are still missing.

if oracle support collapses:
  carrier/support stability is the blocker.
```
