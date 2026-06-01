# DG-KAN v9.2.49 Cost-Amortized Online MicroProbe Kernelization 与 Cheap-Gap Controller Closure 完整实验计划

> 本计划基于 v9.2.48 `Real Train-Stream MicroProbe 与 Online Support-Region Controller` 的真实复盘制定。  
> v9.2.48 的 terminal route 是：
>
> ```text
> route = R8-MicroProbePredictiveButTooExpensive
> base_candidate = LQ-t2-h256
> success_v9248_strict_purekan_functional = False
> success_v9248_full_functional = False
> success_v9248_external_ready = False
> ```
>
> v9.2.48 的关键结果是：
>
> ```text
> P1 real online train-stream microprobe:
>   implemented = 1
>   rows = 2304
>   update/probe/candidate hashes exist
>   commit-time order pass = 1
>
> P2 online feature predictivity:
>   best feature = gap_probe
>   safe-good AUC = 0.884406
>   component all pass = 1
>
> P2 system:
>   probe overhead q90 = 5.121495
>   step ratio q90 = 6.121495
>   microprobe system pass = 0
>   online microprobe pass = 0
>
> P2 signal-channel:
>   signal-channel AUC = 0.536473
>   corr = 0.060158
>   signal_channel_pass = 0
>
> P3 oracle support:
>   oracle precision = 1.0
>   oracle coverage = 0.141059
>   oracle bad-event = 0.0
>   oracle support pass = 1
>   measured signal strata = 2
>   fresh natural support pass = 0
>
> P4 controller:
>   best = C5-ParetoFrontOnlineController
>   precision = 1.0
>   coverage = 0.004340
>   bad-event = 0.0
>   support-region controller pass = 0
>
> Downstream:
>   LDO / LSO / paired replay / short-run / full / robustness = not_run
>
> Current blocker:
>   microprobe_predictive_but_system_overhead_failed
> ```
>
> 因此 v9.2.49 的核心不是继续证明 online microprobe 有 signal。这个已经在 v9.2.48 被证明。  
> 核心也不是再调一个 controller threshold。P4 controller 没打开的直接原因之一是 P2 system gate 已经失败，另一个是 coverage 太低。  
> v9.2.49 的核心任务必须是：
>
> $$
> \boxed{
> \text{把 expensive but predictive gap\_probe 变成 cost-amortized / kernelized / cheap-surrogate online controller。}
> }
> $$
>
> 本轮继续加快实验，采用并行 runner。所有 diagnostic rows 可以并行测，但 official success 必须严格受 gate 控制。

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
  if dataset == Fashion: use cheaper probe A
  if dataset == KMNIST: use threshold B
  if dataset == MNIST: abstain
  tune threshold separately per dataset
  use validation/test metric at commit time
  use posthoc replay outcome at commit time for official controller
```

---

# Part I. 对 v9.2.48 的独立判断

## 1. v9.2.48 没有达到目标

v9.2.48 没有达到 strict PureKAN functional success。原因很清楚：

```text
online microprobe implemented = 1
online feature predictive = 1
but system gate = 0
controller gate = 0
LDO / LSO / paired replay = not_run
```

这意味着不能声明：

```text
strict PureKAN local causal evidence
full functional success
external-ready success
Beyond-MLP success
```

P5-P9 保持关闭是正确的，因为一个 `6.121495x` step ratio 的 microprobe 即使 AUC 很高，也不能作为 official functional update 路线。

## 2. v9.2.48 的真实进展

v9.2.48 是最近几轮中最重要的正进展之一。它首次把 `fresh online train-stream microprobe` 从计划变成真实 measured rows：

```text
rows = 2304
commit-time order pass = 1
update/probe/candidate hashes exist
```

这直接推翻了 v9.2.47 的 `OfflineFeatureOnly` 局限。更重要的是，online `gap_probe` 对 safe-good 的 AUC 达到：

$$
AUC_{\text{gap-probe}}=0.884406.
$$

这说明过去 v9.2.46/v9.2.47 的 LF8 / boundary-probe signal 不是纯 offline artifact。至少有一种真实 online train-stream probe 能识别 safe-good event。

因此这轮的正面结论是：

$$
\boxed{
\text{safe-good event 是 online 可观测的。}
}
$$

这比 “source-measured rows 有 AUC” 强得多。

## 3. 当前真正 blocker

当前 blocker 从 feature representation 转成了 system implementation：

$$
\boxed{
\text{online feature is predictive but too expensive。}
}
$$

具体是：

$$
Overhead_{q90}=5.121495,
$$

$$
StepRatio_{q90}=6.121495.
$$

这远超：

$$
Overhead_{\text{probe}}\leq0.20,
$$

$$
StepRatio_{q90}\leq1.50.
$$

所以不能继续把注意力放在“再找一个更高 AUC 的 feature”。当前必须回答：

```text
1. gap_probe 的成本到底来自哪里？
2. 是否能只在稀疏候选事件上调用 expensive probe？
3. 是否能用 cheap surrogate 近似 gap_probe？
4. 是否能通过 kernelization / fusion / cached control branch 降低 per-probe cost？
5. 是否能在保持 precision 的同时把 coverage 从 0.004340 提到 >=0.03？
```

## 4. 为什么仍然感觉进展慢

感觉慢是合理的，但 v9.2.48 不是原地踏步。它把问题从不可验证的 offline diagnostic 推进成非常具体的系统瓶颈：

```text
旧 blocker:
  online microprobe not implemented

新 blocker:
  online microprobe predictive but system-expensive
```

这比之前清晰很多。  
现在的慢主要来自两个原因：

```text
1. 我们之前花了很多轮才把 offline feature 转成 online probe；
2. 当前 probe 是 brute-force shadow control-gap probe，几乎必然昂贵。
```

所以 v9.2.49 不能再是“再跑一个 controller”。必须做并行系统化：

```text
profile where cost comes from
try event-sparse amortization
try cheap gap surrogate
try kernelized shadow probe
try family-balanced coverage repair
run controller + LDO/LSO + paired replay scout in same runner
```

## 5. 是否在正确道路上

是，但必须收紧。

正确路线：

```text
online microprobe implemented
→ predictive gap_probe confirmed
→ amortize/kernelize/surrogate the probe
→ support-region controller
→ LDO/LSO
→ paired replay
```

错误路线：

```text
继续在 offline LF8 上做后验解释；
继续调 C5/C1/C6 threshold；
因为 signal-channel failed 就放弃 online probe；
回到 basis sweep；
按 dataset 调 controller；
把 oracle support 写成 success。
```

v9.2.48 的结果说明，functional update 主线仍值得继续，因为 online safe-good predictivity 已经成立。但它也说明，现在最重要的研究对象不是 feature existence，而是 **system-legal online decision mechanism**。

## 6. 离目标还差多远

离 strict PureKAN local functional causality 还差四道门：

```text
1. cost-amortized / kernelized online probe system pass；
2. support-region controller heldout pass；
3. leave-dataset-out / leave-stratum-out pass；
4. official paired replay beat AdamWParallel / bestLR。
```

离 full functional success 还要继续过：

```text
short-run task-safe mechanism gain；
full 10-seed macro / hard-stratum / geometry gain；
robustness；
strong baseline；
external-ready。
```

当前最近的一道门是：

$$
\boxed{
\text{把 } StepRatio_{q90}=6.121495 \text{ 压回 } \leq1.50。
}
$$

这不是小修小补，而是 v9.2.49 的主问题。

---

# Part II. v9.2.49 总体目标

v9.2.49 的总体目标是：

$$
\boxed{
\text{把 v9.2.48 的 predictive gap\_probe 变成 system-legal online controller，并打开 LDO/LSO 与 paired replay。}
}
$$

目标分为八层。

## 1. Microprobe cost attribution success

必须把 `probe overhead q90 = 5.121495` 分解到具体组件：

```text
shadow update materialization
probe forward
probe backward / JVP / VJP
AdamWParallel shadow branch
bestLR shadow branch
gap computation
risk/value feature computation
temporary tensor allocation
synchronization / kernel launch
controller decision
logging overhead
```

成功标准：

```text
>= 90% overhead attributed to named components
kernel_count / sync_count / read-write MB measured
no unknown overhead fraction > 0.10
```

## 2. Event-sparse amortization success

如果 expensive probe 只在少量 candidate events 上调用，则 per-event overhead 可以高，但 amortized step overhead 必须过 gate。

定义：

$$
p_{\text{probe}}=\frac{\#\text{probe-called steps}}{\#\text{all train steps}}.
$$

Amortized overhead：

$$
Overhead_{\text{amortized}}
=
p_{\text{probe}}
\cdot
Overhead_{\text{per-probe}}
+
Overhead_{\text{cheap-prefilter}}.
$$

成功标准：

$$
Overhead_{\text{amortized}}\leq0.20,
$$

$$
StepRatio_{q90,\text{all steps}}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

同时 probe-called events 必须保留 value：

$$
Precision\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05.
$$

## 3. Cheap gap surrogate success

构造 cheap surrogate 去近似 expensive `gap_probe`，避免每次都跑 shadow controls。

Target:

$$
Y_{\text{safe-good}}
=
Y_{\text{risk-safe}}
\cdot
Y_{\text{value-positive}}
\cdot
Y_{\text{control-resistant}}.
$$

Cheap surrogate 候选必须只使用 commit-time train-stream features：

```text
probe CE / margin first-order delta
branch ratio
effective derivative
functional step norm
task step norm
role-gap
risk LCB
family reliability
support density
SNR
low-rank signal-channel sketch
cached AdamWParallel baseline
```

成功标准：

$$
AUC(S_{\text{cheap}},Y_{\text{safe-good}})\geq0.70
$$

or:

$$
Corr(S_{\text{cheap}},V_{\text{safe-grounded}})\geq0.35.
$$

如果 cheap surrogate 只能做 prefilter，则 minimum diagnostic gate 是：

$$
AUC\geq0.60
$$

and:

$$
Recall_{\text{safe-good}}\geq0.70
$$

at:

$$
BadEventRate_{\text{prefilter}}\leq0.10.
$$

## 4. Kernelized / fused probe success

如果 expensive gap_probe 仍必要，则必须 kernelize：

```text
no full parameter copy
delta-view / reversible shadow update
single fused probe forward for Real / AdamWParallel / bestLR where possible
fused CEp99 / margin / risk computation
cached control branch direction
vectorized candidate branches
no repeated synchronization
```

System success：

$$
Overhead_{\text{per-probe}}\leq1.00
$$

or event-sparse：

$$
Overhead_{\text{amortized}}\leq0.20.
$$

Strict global gate：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

## 5. Stratum support expansion success

v9.2.48 online rows only covered two signal strata:

```text
S4-RoleWiseCurvature: 2132
S1-CEHardTail: 172
```

v9.2.49 必须扩大 online support，不能只在 narrow stratum 上成功。

Measurement success：

```text
measured_signal_strata_count >= 6
natural_real_event_count >= 3000
diagnostic_balanced_real_event_count >= 3000
```

Official accepted support：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

## 6. Support-region controller success

Official controller：

$$
Accept(e)
=
CheapPrefilter(e)
\land
[
CheapController(e)
\lor
AmortizedProbeController(e)
].
$$

Heldout gate：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{heldout}}\leq0.05.
$$

System gate：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

## 7. Leave-out success

Leave-dataset-out：

$$
Acc_{\text{heldout,Real}}\geq Acc_{\text{heldout,AdamW}}-0.005.
$$

At least two held-out datasets satisfy：

$$
BeatRate_{\text{Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{Real vs bestLR}}\geq0.50.
$$

Leave-stratum-out：

At least $70\%$ held-out strata task-safe and：

$$
CEp99_{\text{heldout,Real}}\leq CEp99_{\text{AdamW}}+\epsilon,
$$

$$
BeatRate_{\text{heldout-stratum,Real vs AdamWParallel}}\geq0.50.
$$

## 8. Official paired replay success

Only after controller + LDO/LSO pass，official paired replay opens。

Paired replay pass：

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

Shuffle controls must fail：

```text
ShuffledGapProbe = fail
ShuffledCheapGap = fail
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

---

# Part III. 核心假设

## H1：v9.2.48 的失败是 system implementation failure，不是 online observability failure

证据：

$$
AUC_{\text{gap-probe}}=0.884406.
$$

但：

$$
StepRatio_{q90}=6.121495.
$$

H1 成立标准：

```text
cost attribution identifies concrete overhead source
and at least one amortized/kernelized design keeps AUC or precision while passing system gate
```

H1 失败标准：

```text
all low-cost / amortized / kernelized probes lose safe-good predictivity
```

## H2：gap_probe 主要贵在 shadow controls，而不是 probe CE/margin本身

`gap_probe` 是最强 feature，但它可能需要 Real / AdamWParallel / bestLR shadow branch。H2 认为成本主因是 shadow controls。

H2 成立标准：

```text
shadow branch components account for >= 60% overhead
cached / linearized control-gap surrogate reduces overhead by >= 50%
```

## H3：event-sparse amortization 可以让 expensive probe 变 system-legal

如果 prefilter 能把 probe call rate 限制在：

$$
p_{\text{probe}}\leq0.03,
$$

则即使：

$$
Overhead_{\text{per-probe}}\approx5,
$$

amortized overhead 也可能接近：

$$
0.03\times5=0.15.
$$

H3 成立标准：

$$
Overhead_{\text{amortized}}\leq0.20
$$

and controller heldout pass。

H3 失败标准：

```text
prefilter recall too low
or probe call rate must be high
or rare probes lose coverage / LDO
```

## H4：signal-channel sketch v9.2.48 failed because sketch was too weak, not because signal-channel idea is useless

v9.2.48 signal-channel AUC only：

$$
AUC=0.536473.
$$

H4 认为 cheap signal-channel still worth one redesign, but not as main blocker.

H4 成立标准：

Improved sketch gets：

$$
AUC_{\text{channel}}\geq0.60
$$

or improves controller precision/coverage by at least $0.05$.

H4 失败标准：

```text
all cheap signal-channel / SNR features remain AUC < 0.60
```

If H4 fails, signal-channel is demoted to diagnostic, not main route.

## H5：coverage failure is caused by support concentration, not only by threshold

C5 had precision `1.0` but coverage `0.004340`; P3 measured only 2 strata.  
H5 成立标准：

```text
family-balanced support expansion raises coverage >= 0.03
without reducing precision below 0.75
and without bad-event > 0.05
```

## H6：if microprobe is predictive but cannot be made system-legal, next step is kernelization / amortization, not controller tuning

If:

```text
AUC pass
controller pass in diagnostic
system fail
```

then route must be:

```text
R6-PredictiveButTooExpensive
```

and next step is kernel / amortized feature implementation, not more threshold tuning.

## H7：if cheap surrogate and amortized probe both fail while oracle support remains high, legal feature representation remains blocker

If:

```text
oracle support pass = 1
all legal online features fail
```

then route goes to legal sufficient statistics reset.

## H8：dataset tuning remains forbidden

If performance fails mainly on one dataset, only diagnostic slicing is allowed. The controller must use event features, not dataset name.

---

# Part IV. 并行执行设计

v9.2.49 runner 必须并行执行以下 lanes：

```text
Lane A:
  v9.2.48 boundary reproduction

Lane B:
  microprobe phase-cost attribution

Lane C:
  event-sparse amortized probe:
    cheap prefilter
    probe-call rate control
    amortized overhead measurement

Lane D:
  cheap gap surrogate:
    first-order / cached-control / branch-ratio / role-gap / family support

Lane E:
  kernelized probe:
    delta-view shadow update
    fused branch evaluation
    fused metrics
    reduced sync

Lane F:
  signal-channel sketch v2:
    low-rank output displacement covariance
    drift-diffusion SNR
    channel ratio

Lane G:
  support / stratum expansion:
    natural rows
    balanced diagnostic rows

Lane H:
  support-region controller calibration

Lane I:
  LDO / LSO

Lane J:
  paired replay scout and official replay

Lane K:
  short-run scout if paired replay passes
```

Gate discipline：

```text
diagnostic rows may be measured early
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  online feature pass
  system gate pass
  controller pass
  LDO/LSO pass
```

---

# Part V. Candidate feature and probe designs

## 1. Baseline

Official base remains：

```text
R2-LQ-fanin-output-scale-confirmed
```

Attach/carrier candidates：

```text
A0-current-v9248
A3-RoleWiseFT7ResetCarrier
A6-HybridRoleControlRiskCarrier
```

## 2. Probe candidates

### MP0：v9.2.48 gap_probe reference

Reference only. Expected predictive but too expensive.

### MP1：EventSparseGapProbe

Uses cheap prefilter：

$$
Prefilter(e)=
RiskSafe_{\text{cheap}}(e)
\land
BranchActive(e)
\land
FamilyCandidate(e).
$$

Run expensive gap_probe only if $Prefilter(e)=1$.

### MP2：LinearizedGapProbe

Approximate branch outcomes using first-order logit displacement:

$$
z_{\theta+\Delta\theta}
\approx
z_\theta
+
J_\theta \Delta\theta.
$$

Gap surrogate：

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

### MP3：CachedControlGapProbe

Cache AdamWParallel / bestLR branch statistics over recent train-stream windows:

$$
S_{\text{gap-cache}}
=
\widehat{Gain}_{F,\text{probe}}
-
EMA(
\max(
Gain_{AdamWParallel},
Gain_{bestLR}
)
).
$$

### MP4：OneForwardMultiBranchProbe

Evaluate Real / AdamWParallel / bestLR deltas in one fused/vectorized probe pass over $B_{\text{probe}}$.

### MP5：DeltaViewShadowProbe

Avoid full parameter copy. Use reversible delta view:

$$
\theta'=\theta+\Delta\theta
$$

without materializing a full copied parameter set.

### MP6：CheapNoShadowProbe

No shadow branch. Uses only current-step statistics:

```text
branch_ratio
effective_derivative
functional_step_norm / task_step_norm
probe CE/margin before commit
risk LCB
support density
family reliability
```

### MP7：SignalChannelSketchProbe

Low-rank sketch $U_k$ from train-stream output covariance:

$$
S_{\text{channel}}
=
\frac{\|U_k^T\Delta z_F\|^2}{\|\Delta z_F\|^2+\epsilon}.
$$

### MP8：HybridAmortizedProbe

Pipeline：

```text
cheap no-shadow prefilter
→ linearized gap
→ expensive gap_probe only for borderline / high-value candidates
```

---

# Part VI. Controller candidates

## C0：v9.2.48 best reference

Reference only.

## C1：CheapOnlySupportController

$$
Accept(e)=
RiskSafe_{\text{cheap}}(e)
\land
ValuePositive_{\text{cheap}}(e)
\land
ControlResistant_{\text{cheap}}(e)
\land
SupportStable(e).
$$

## C2：EventSparseGapController

$$
Accept(e)=
Prefilter(e)
\land
GapProbe(e)
\land
RiskSafe(e).
$$

## C3：LinearizedGapController

$$
Accept(e)=
RiskSafe(e)
\land
S_{\text{gap-linear}}(e)\geq\tau_g
\land
SupportStable(e).
$$

## C4：CachedControlGapController

$$
Accept(e)=
RiskSafe(e)
\land
S_{\text{gap-cache}}(e)\geq\tau_g
\land
FamilyReliable(e).
$$

## C5：HybridAmortizedController

$$
Accept(e)=
CheapPrefilter(e)
\land
[
CheapHighConfidence(e)
\lor
ExpensiveProbeConfirmed(e)
].
$$

## C6：ParetoFrontControllerV2

Accept if event lies on Pareto frontier of:

```text
low risk
positive value
positive control gap
high support density
low probe overhead
family balance
```

## C7：Oracle

Posthoc diagnostic only. Never official.

---

# Part VII. 实验阶段

## P0：v9.2.48 boundary reproduction

### 目标

复现 v9.2.48 boundary，确认本轮不是解析错误。

### 必须记录

```text
route
source_route_v9247
online_microprobe_implemented
online_feature_predictivity_pass
best_feature_id
best_feature_auc
best_feature_corr
microprobe_system_pass
probe_overhead_q90
step_ratio_q90
signal_channel_pass
oracle_support_pass
oracle_precision
oracle_coverage
oracle_bad_event
fresh_natural_support_pass
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
route = R8-MicroProbePredictiveButTooExpensive
best_feature_id = gap_probe
online_feature_predictivity_pass = 1
microprobe_system_pass = 0
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_predictive_but_expensive_ladder.svg
p0_feature_vs_system_gap.svg
```

---

## P1：microprobe phase-cost attribution

### 目标

定位 $5.121495$ q90 overhead 的来源。

### 必须记录

```text
row_id
probe_candidate
phase
phase_time_ms
phase_time_ratio
read_MB
write_MB
temp_alloc_MB
kernel_count
sync_count
shadow_param_copy_MB
delta_view_used
control_branch_count
probe_forward_count
metric_kernel_count
controller_time_ms
logging_time_ms
```

Phases：

```text
F0-base task step reference
F1-cheap prefilter
F2-candidate functional update compute
F3-shadow apply / delta view
F4-probe forward Real
F5-probe forward AdamWParallel
F6-probe forward bestLR
F7-metric computation CE/margin/risk/gap
F8-controller decision
F9-rollback / commit
F10-logging
```

### 判断标准

Cost attribution pass：

```text
unknown_overhead_fraction <= 0.10
dominant_phase_identified = 1
phase_time_sum_close_to_total within ±0.05
```

### 可视化

```text
p1_probe_cost_waterfall.svg
p1_memory_traffic_by_phase.svg
p1_kernel_sync_count_by_phase.svg
p1_dominant_overhead_pareto.svg
```

---

## P2：event-sparse amortized microprobe

### 目标

验证 expensive gap_probe 是否可通过稀疏调用变 system-legal。

### 设置

```text
probe candidates:
  MP1-EventSparseGapProbe
  MP8-HybridAmortizedProbe

prefilter candidates:
  PF0-BranchRisk
  PF1-RiskTailLCB
  PF2-FamilyCandidate
  PF3-RoleGapCandidate
  PF4-HighDerivativeBranch
```

### 必须记录

```text
row_id
prefilter_id
probe_candidate
prefilter_accept
probe_called
probe_call_rate
per_probe_overhead
amortized_overhead
step_ratio_all_q50
step_ratio_all_q90
memory_ratio
safe_good
bad_event
precision
coverage
bad_event_rate
accepted_strata_count
accepted_family_count
max_family_share
```

### 判断标准

Amortized system pass：

$$
Overhead_{\text{amortized}}\leq0.20,
$$

$$
StepRatio_{q90,\text{all}}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

Controller diagnostic pass：

$$
Precision\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05.
$$

### 可视化

```text
p2_probe_call_rate_vs_precision.svg
p2_amortized_overhead_vs_coverage.svg
p2_prefilter_recall_safe_good.svg
p2_event_sparse_system_pareto.svg
```

---

## P3：cheap gap surrogate matrix

### 目标

寻找无需 expensive shadow controls 的 gap surrogate。

### 必须记录

```text
surrogate_id
features_used
uses_shadow_control
uses_dataset_name
uses_validation
uses_test
uses_posthoc
AUC_safe_good
corr_safe_grounded
precision_at_gate
coverage_at_gate
bad_event_at_gate
recall_safe_good
feature_overhead
memory_overhead
```

### 判断标准

Official cheap surrogate pass：

$$
AUC\geq0.70
$$

or:

$$
Corr\geq0.35.
$$

and heldout accept gate pass。

Prefilter pass：

$$
AUC\geq0.60,
$$

$$
Recall_{\text{safe-good}}\geq0.70,
$$

$$
BadEventRate_{\text{prefilter}}\leq0.10.
$$

### 可视化

```text
p3_surrogate_auc_corr_matrix.svg
p3_surrogate_precision_coverage.svg
p3_surrogate_cost_predictivity_pareto.svg
```

---

## P4：kernelized / fused microprobe implementation

### 目标

降低 per-probe cost，而不是只靠稀疏调用。

### Candidates

```text
K0-reference-gap-probe
K1-delta-view-shadow-probe
K2-one-forward-multibranch-probe
K3-fused-metric-kernel
K4-cached-control-gap
K5-linearized-gap-jvp
K6-hybrid-fused-amortized
```

### 必须记录

```text
kernel_candidate
grad_correctness_if_applicable
numeric_equivalence_to_reference
max_logit_diff
max_metric_diff
per_probe_overhead_q50
per_probe_overhead_q90
step_ratio_q90
memory_ratio
read_MB
write_MB
kernel_count
sync_count
temporary_alloc_MB
```

### 判断标准

Kernelization pass：

$$
per\_probe\_overhead_{q90}\leq1.00
$$

or:

$$
amortized\_overhead\leq0.20.
$$

Numerical sanity：

$$
\max|\Delta metric|\leq10^{-5}
$$

for deterministic reference where applicable.

### 可视化

```text
p4_kernelized_probe_overhead.svg
p4_probe_numeric_equivalence.svg
p4_memory_traffic_reduction.svg
```

---

## P5：signal-channel sketch v2

### 目标

验证 cheap signal-channel / SNR 是否能替代或辅助 expensive gap_probe。

### Features

```text
SNR_mean
SNR_p90
signal_channel_ratio
signal_channel_ratio_tail
low_rank_output_displacement_energy
noise_reservoir_energy
drift_diffusion_ratio
```

### 必须记录

```text
feature_id
rank_k
sketch_update_cost
AUC_safe_good
corr_safe_grounded
AUC_risk_safe
AUC_value_positive
AUC_control_resistant
controller_ablation_gain
feature_overhead
```

### 判断标准

Signal diagnostic pass：

$$
AUC_{\text{signal}}\geq0.60
$$

or controller gain：

$$
\Delta Precision\geq0.05
$$

or:

$$
\Delta Coverage\geq0.05
$$

without bad-event increase。

Official pass only if full controller gates pass.

### 可视化

```text
p5_signal_channel_auc.svg
p5_snr_vs_gap_probe.svg
p5_signal_feature_ablation.svg
```

---

## P6：online support / stratum expansion

### 目标

避免 v9.2.48 的 narrow online distribution。

### 设置

Two row sources：

```text
natural rows:
  official evaluation distribution

balanced diagnostic rows:
  stratified by signal_stratum / event_family
  only for feature and failure diagnosis unless separately marked official
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
accepted_by_controller
oracle_accept
feature_values
```

### 判断标准

Measurement pass：

```text
natural_real_event_count >= 3000
diagnostic_balanced_real_event_count >= 3000
measured_signal_strata_count >= 6
```

Official accepted support：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

### 可视化

```text
p6_signal_strata_coverage.svg
p6_family_support_heatmap.svg
p6_natural_vs_balanced_distribution.svg
```

---

## P7：support-region controller calibration

### 目标

用 cheap / amortized / kernelized features 建立 official controller。

### Split design

```text
calibration split:
  choose thresholds / monotone coefficients

heldout split:
  official controller gate

balanced diagnostic split:
  not official unless declared and separated

leave-dataset-out:
  P8

leave-stratum-out:
  P8
```

### 必须记录

```text
controller_id
probe_candidate
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

and：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{heldout}}\leq0.05.
$$

System：

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
probe_candidate
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
p8_leave_dataset_out_matrix.svg
p8_leave_stratum_out_matrix.svg
p8_hidden_dataset_tuning_audit.svg
p8_leaveout_failure_modes.svg
```

---

## P9：Official paired replay

### 目标

验证 RealFunctional 是否在 strong controls 下有局部因果优势。

### 设置

```text
base = R2 repaired base checkpoint
probe_candidate = best P8 survivor
controller_id = best P8 survivor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random,
           ShuffledGapProbe, ShuffledCheapGap, ShuffledRiskScore,
           ShuffledBranchRatio, ShuffledControlGap, ShuffledValueScore,
           ShuffledSignalChannel, FunctionalChannelShuffled,
           TailMaskShuffled, RoleScoreShuffled, DatasetRouteShuffled,
           EventRouteShuffled, InvertedRoleMask
```

### 必须记录

```text
controller_id
probe_candidate
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
contract_audit_v9249.csv
p0_v9248_boundary_reproduction.csv
p1_microprobe_phase_cost_attribution.csv
p2_event_sparse_amortized_microprobe.csv
p3_cheap_gap_surrogate_matrix.csv
p4_kernelized_fused_microprobe.csv
p5_signal_channel_sketch_v2.csv
p6_online_support_stratum_expansion.csv
p7_support_region_controller_calibration.csv
p8_leave_dataset_and_stratum_out.csv
p9_official_paired_replay.csv
p10_short_run_functional_validation.csv
p11_full_10seed_functional_validation.csv
p12_robustness_external_ready.csv
probe_phase_trace_v9249.csv
memory_traffic_trace_v9249.csv
event_sparse_probe_trace_v9249.csv
cheap_gap_surrogate_trace_v9249.csv
kernelized_probe_trace_v9249.csv
signal_channel_trace_v9249.csv
support_density_trace_v9249.csv
controller_calibration_trace_v9249.csv
leaveout_trace_v9249.csv
paired_replay_branch_trace_v9249.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9248_boundary_unstable
F3_dataset_tuning_detected
F4_probe_cost_unattributed
F5_event_sparse_prefilter_low_recall
F6_event_sparse_amortization_fail
F7_cheap_gap_surrogate_not_predictive
F8_kernelized_probe_not_equivalent
F9_kernelized_probe_still_too_expensive
F10_signal_channel_sketch_fail
F11_stratum_support_too_narrow
F12_controller_fail
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

# Part IX. Route decision

```text
R1-ProbeCostAttributed:
  v9.2.48 overhead source is identified.

R2-EventSparseAmortizedProbePass:
  expensive gap_probe becomes system-legal through sparse invocation.

R3-CheapGapSurrogatePass:
  cheap online surrogate predicts safe-good and passes heldout controller gate.

R4-KernelizedProbePass:
  fused / delta-view / cached-control probe passes system and predictivity gates.

R5-SignalChannelSketchPass:
  improved signal-channel / SNR feature contributes predictive value.

R6-ControllerPass:
  support-region controller passes heldout precision / coverage / bad-event / system gate.

R7-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R8-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R9-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R10-PredictiveButTooExpensiveAgain:
  predictive probe remains outside system envelope.

R11-CheapFeatureFailOracleHigh:
  oracle support exists but legal cheap features fail.

R12-OracleSupportCollapse:
  fresh natural replay shows no sufficient safe-good oracle support.

R13-ControllerStillCoverageLimited:
  precision and safety are good but coverage remains below 0.03.

R14-StrictPureKANFunctionalShortRunPass:
  short-run task-safe mechanism gain.

R15-StrictPureKANFunctionalFullPass:
  full 10-seed macro / hard-stratum / geometry gain.

R16-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` 必须记录：

```text
route
v9248_boundary_pass
dataset_tuning_detected
probe_cost_attribution_pass
dominant_probe_cost_phase
event_sparse_probe_pass
probe_call_rate
per_probe_overhead_q90
amortized_overhead
cheap_gap_surrogate_pass
cheap_gap_auc
cheap_gap_corr
kernelized_probe_pass
kernelized_probe_overhead_q90
signal_channel_pass
signal_channel_auc
signal_channel_corr
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
success_v9249_strict_purekan_functional
success_v9249_full_functional
success_v9249_external_ready
```

---

# Part X. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 microprobe phase-cost attribution
  P2 event-sparse amortized probe
  P3 cheap gap surrogate
  P4 kernelized/fused probe smoke
  P5 signal-channel sketch v2

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
  system gate pass
  controller pass
  LDO/LSO pass
```

---

# Part XI. 停止条件

## Minimum diagnostic success

```text
v9.2.48 boundary reproduced
microprobe cost attribution completed
event-sparse / cheap-surrogate / kernelized branches all measured
online support coverage measured
no fake/proxy/offload/loss/teacher violation
```

## System-legal online feature success

```text
Minimum diagnostic success
+
at least one of:
  event-sparse amortized probe pass
  cheap gap surrogate pass
  kernelized probe pass
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
1. v9.2.48 boundary cannot be reproduced；
2. probe overhead cannot be attributed；
3. event-sparse amortization fails；
4. cheap gap surrogate fails；
5. kernelized probe remains too expensive；
6. signal-channel sketch remains nonpredictive；
7. online support remains too narrow；
8. all support-region controllers fail heldout gate；
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

# Part XII. 最终解释规则

## Case A：event-sparse amortized probe passes

可以声明：

```text
v9.2.48 failed because predictive gap_probe was called too often; sparse invocation made it system-legal.
```

但不能声明 strict success unless LDO/LSO and paired replay pass.

## Case B：cheap surrogate passes

可以声明：

```text
expensive shadow control-gap can be approximated by legal cheap train-stream statistics.
```

但仍需 controller and paired replay.

## Case C：kernelized probe passes

可以声明：

```text
the system blocker was implementation overhead, not conceptual feature failure.
```

但 full success still requires downstream.

## Case D：all system-legal variants lose predictivity

必须声明：

```text
safe-good is only observable through expensive shadow controls; current legal feature set is insufficient.
```

下一步设计 richer low-cost sufficient statistics，不再调 threshold。

## Case E：oracle support collapses

必须声明：

```text
fresh natural replay no longer contains enough safe-good events.
```

下一步回到 carrier/support mechanism.

## Case F：LDO/LSO fail

必须声明：

```text
controller is not dataset-agnostic or stratum-agnostic enough.
```

不能用 dataset-specific tuning 写成功。

## Case G：paired replay passes

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full validation。

---

# Part XIII. 最终建议

v9.2.49 的一句话策略是：

$$
\boxed{
\text{不要再证明 gap\_probe 有 AUC；现在要把它变便宜、变稀疏、变可部署。}
}
$$

当前最关键的问题不是：

```text
base 是否稳定；
attach 是否污染；
carrier 是否 silent；
online feature 是否完全没信号；
C5 threshold 是否差一点；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. gap_probe 为什么需要 5.121495x q90 overhead？
2. 能否通过 event-sparse prefilter 把 amortized overhead 压到 <=0.20？
3. 能否用 cheap surrogate 逼近 expensive gap_probe？
4. 能否通过 kernelized / fused / delta-view shadow probe 把 per-probe cost 降到可接受范围？
5. 能否扩大 online signal-stratum support，避免只在 S4/S1 上成立？
6. support-region controller 能否在 heldout 上同时满足 precision / coverage / bad-event / system？
7. LDO/LSO 与 paired replay 能否打过 AdamWParallel / bestLR？
```

v9.2.49 的结果将给出清晰分叉：

```text
if amortized/kernelized/cheap probe + controller + LDO/LSO + paired replay pass:
  strict PureKAN functional obtains local causal evidence.

if probe remains predictive but too expensive:
  system kernelization / amortization is the blocker.

if cheap features fail but oracle remains high:
  legal sufficient statistics are still missing.

if oracle support collapses:
  carrier/support stability is the blocker.

if controller passes local but LDO/LSO fails:
  no dataset tuning; broaden signal-stratum/family support.
```
