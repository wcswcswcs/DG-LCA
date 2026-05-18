# DG-KAN v9.2.55 Lower-Level CUDA Branch-Delta Extension 与 Exact-Signal Functional Controller Closure 完整实验计划

> 本计划基于 v9.2.54 `Custom Fused Branch-Delta Kernel 与 Exact-Signal Controller Closure` 的真实复盘制定。  
> v9.2.54 的 terminal route 是：
>
> ```text
> route = R14-NeedsLowerLevelCUDAExtension
> base_candidate = LQ-t2-h256
> success_v9254_strict_purekan_functional = False
> success_v9254_full_functional = False
> success_v9254_external_ready = False
> ```
>
> v9.2.54 的关键事实是：
>
> ```text
> P0:
>   v9.2.53 boundary reproduced
>   source route = R12-FusedBranchDeltaPredictiveButNeedsCustomKernel
>   D0 vectorization pass = 1
>   FBD6 system pass = 0
>   oracle support pass = 1
>   fake/proxy/offload = 0
>
> P1:
>   FBD6 residual cost attribution pass = 1
>   dominant residual subphase = K7-gap_risk_support_reduction
>   ratio = 0.662327
>
> P2:
>   CBD0-FBD6-D0VectorizedFusedDeltaReference:
>     AUC = 0.888401
>     agreement = 1.0
>     step ratio = 2.863280 > 1.50
>     system pass = 0
>
>   CBD1-Triton selected-logit formula:
>     step ratio = 1.35
>     agreement = 0.386822
>     AUC = 0.534988
>     official = 0
>     not true exact branch-delta kernel
>
>   true custom branch-delta implemented = 0
>   CBD5-CUDAExtensionBranchDelta = not_implemented_cuda_extension_required
>
> P3:
>   all controllers official_eligible = 0
>
> P4:
>   oracle precision = 1.0
>   oracle coverage = 0.121693
>   oracle bad-event = 0.0
>   measured signal strata = 3
>   support measurement pass = 0
>
> Downstream:
>   LDO / LSO / paired replay / short-run / full / robustness = not_run
>
> current blocker:
>   triton_formula_kernel_not_true_branch_delta_kernel
> ```
>
> v9.2.55 的核心判断是：
>
> $$
> \boxed{
> \text{exact branch-delta signal 已经存在，但当前低成本公式 kernel 不是 true branch-delta；下一步必须做 lower-level exact kernel。}
> }
> $$
>
> 因此 v9.2.55 不是继续试一个 cheap score，也不是继续改 controller threshold。  
> 它是一个 **stop-go 实验**：
>
> $$
> \boxed{
> \text{能否实现 true lower-level CUDA/C++ branch-delta extension，并同时保留 AUC / agreement / system gate？}
> }
> $$
>
> 如果能，才进入 exact-signal controller、LDO/LSO、paired replay。  
> 如果不能，必须诚实 route 到 lower-level kernelization blocker，而不是继续在 Triton formula / Python diagnostic 上绕圈。

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
  report per-dataset failure mode

forbidden:
  if dataset == Fashion: use controller A
  if dataset == KMNIST: use threshold B
  if dataset == MNIST: skip branch-delta
  tune threshold separately per dataset
  choose selected/full-logit mode separately per dataset
  use validation/test metric at commit time
  use posthoc replay outcome at commit time for official controller
```

Official success 只允许来自：

```text
train-stream online features
commit-time decision
true custom branch-delta implementation
system gate pass
heldout natural split
LDO / LSO
official paired replay
```

---

# Part I. 对 v9.2.54 的独立判断

## 1. v9.2.54 没有达到目标

v9.2.54 没有达到 strict PureKAN functional success。P5-P9 全部 gate-blocked，原因是：

```text
true custom branch-delta implemented = 0
CBD5 CUDA extension not implemented
controllers official_eligible = 0
support measurement pass = 0
```

因此不能声明：

```text
strict PureKAN local causal evidence
full functional success
external-ready success
Beyond-MLP success
```

v9.2.54 的 `R14-NeedsLowerLevelCUDAExtension` 是合理 route。它不是 functional update 失败，而是承认当前实现仍停在 formula / diagnostic kernel，尚未进入 true branch-delta extension。

## 2. v9.2.54 的真实进展

v9.2.54 有三点清晰进展。

第一，v9.2.53 boundary 被稳定复现。D0 vectorization 保留，FBD6 system 仍 fail，oracle support 仍 pass，fake/proxy/offload 全为 0。这说明本轮不是数据污染或 route 误读。

第二，FBD6 residual cost 被进一步归因。dominant residual subphase 是：

```text
K7-gap_risk_support_reduction
ratio = 0.662327
```

这很关键。v9.2.53 之前我们重点盯 D0 controller scalar prep；v9.2.54 说明 D0 之后真正的 residual bottleneck 是 branch-delta 之后的 gap / risk / support reduction path。也就是说，现在要 fuse 的不只是 logits，而是：

```text
branch-delta logits
gap score
risk score
support/risk predicates
accept components
```

第三，exact reference signal 仍然强：

$$
AUC_{\text{CBD0}}=0.888401,
$$

$$
Agreement_{\text{CBD0}}=1.0.
$$

这说明 safe-good event 的 online observable signal 仍在。我们要压缩的是 implementation，不是换掉核心 signal。

## 3. v9.2.54 的真实失败

v9.2.54 的失败点也很干净：

```text
CBD1 step ratio = 1.35
but AUC = 0.534988
and agreement = 0.386822
and not true exact branch-delta kernel
```

这个结果本质上说明：

$$
\boxed{
\text{便宜的 formula selected-logit kernel 不是 exact branch-delta 的可用替代。}
}
$$

它过了系统速度，但丢了 safe-good signal 与 accept agreement。  
因此 v9.2.55 不应继续推广 CBD1，也不能把它作为 official candidate。它最多是 negative control：证明 “只做 formula shortcut 会丢失 exact branch-delta 信息”。

## 4. 当前真正 blocker

当前 blocker 是：

$$
\boxed{
\text{true branch-delta extension absence}
}
$$

更具体地说：

```text
1. exact branch-delta reference 有 value，但 step ratio 2.863280；
2. cheap Triton formula 有 speed，但 signal/agreement 丢失；
3. true custom CUDA/C++ branch-delta 没实现；
4. controller 因 P2 不 eligible 全部不能 official；
5. support measurement 仍然过窄。
```

因此下一步必须是 lower-level extension，而不是：

```text
继续调 C0/C1/C4 threshold；
继续试更多 selected-logit formula；
继续堆 cheap surrogate；
继续做 offline/source rows autopsy；
回到 basis sweep；
按 dataset 分支。
```

## 5. 是否还在正确道路上

是，但路线已经进入一个明确的系统工程 stop-go 阶段。

正确路线：

```text
online gap-probe signal exists
→ D0 vectorization solved
→ exact branch-delta signal retained
→ true custom branch-delta extension
→ exact-signal controller
→ LDO / LSO
→ paired replay
```

错误路线：

```text
把 CBD1 speed 当作 success；
把 CBD0 AUC 当作 functional success；
把 oracle support 当作 controller pass；
继续 threshold tuning；
继续 formula proxy；
dataset-specific tuning。
```

当前 base / basis 不是主 blocker。`LQ-t2-h256` 仍然是 base candidate。v9.2.54 的信息指向 system primitive：**能否把 exact branch-delta signal 做成 system-legal kernel**。

---

# Part II. v9.2.55 总体目标

v9.2.55 的总体目标是：

$$
\boxed{
\text{实现 true lower-level branch-delta extension，保留 CBD0 的 exact signal，并关闭 exact-signal controller / LDO / paired replay。}
}
$$

具体目标分六层：

```text
1. K7 residual path 深度归因；
2. true CUDA/C++ branch-delta extension 实现；
3. fused branch-delta + gap/risk/support reduction；
4. exact-signal controller；
5. support expansion + LDO/LSO；
6. official paired replay。
```

---

## 1. K7 residual path attribution success

v9.2.54 已经定位到 K7 是 dominant residual subphase，但 K7 内部还不够细。v9.2.55 必须拆开 K7：

```text
K7a branch logits gather
K7b CE / margin selected metric
K7c RealFunctional gain computation
K7d AdamWParallel / bestLR gain computation
K7e gap score reduction
K7f risk score reduction
K7g support / family lookup
K7h accept component construction
K7i temporary allocation
K7j kernel launch / sync
K7k logging / hash / timestamp
```

K7 attribution pass：

```text
unknown_fraction <= 0.10
dominant_k7_subphase_identified = 1
K7_subphase_sum_close_to_K7 within ±0.05
```

这一步要回答：

```text
K7 贵在 score math？
贵在 branch-wise reduction？
贵在 support/family lookup？
贵在 temp allocation？
贵在 torch op fragmentation？
贵在 sync / launch count？
```

---

## 2. True lower-level branch-delta extension success

v9.2.55 必须实现至少一个真正 lower-level branch-delta extension candidate。所谓 true branch-delta extension 必须满足：

```text
1. 不只是公式 proxy；
2. 不只是 posthoc diagnostic；
3. 不只是 selected-logit heuristic；
4. 必须从 functional delta / branch delta 真实计算 branch-output displacement；
5. 必须在 kernel 内或 extension 内融合至少 branch-delta + metric reduction 的关键部分；
6. 必须落盘 correctness / agreement / system measurements。
```

候选层级：

```text
C++/CUDA extension:
  preferred official implementation target

Triton v2:
  only allowed if it computes true branch-delta, not formula proxy

torch.compile:
  diagnostic only unless it passes all gates and avoids formula/proxy semantics
```

True extension pass：

$$
AUC(S_{\text{custom-delta}},Y_{\text{safe-good}})\geq0.70
$$

or:

$$
Corr(S_{\text{custom-delta}},V_{\text{safe-grounded}})\geq0.35.
$$

Agreement pass：

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

Strict target：

$$
StepRatio_{q90}\leq1.20.
$$

如果 true extension 不能实现，route 必须是：

```text
R4-TrueCustomExtensionNotImplemented
```

而不是再写 formula kernel pass。

---

## 3. Fused branch-delta metric-reduction success

因为 K7 是 dominant，custom kernel 不应只输出 logits 再交回 Python/Torch 做 reduction。它应该尽量输出 compact event-level scores：

```text
gap_score
risk_score
support_score
candidate_score
accept_components
```

Fused metric-reduction pass：

$$
T_{\text{K7,fused}}\leq0.35T_{\text{K7,ref}}.
$$

End-to-end pass：

$$
StepRatio_{q90}\leq1.50.
$$

Controller agreement：

$$
Agreement_{\text{accept}}\geq0.90.
$$

If K7 fused pass but full step still fails, record residual phases:

```text
branch_delta_math
delta_state_read
metric_reduction
support lookup
logging/sync
```

---

## 4. Exact-signal controller success

Because weak candidate generator repeatedly failed, official v9.2.55 controller should use branch-delta as primary value signal.

Controller form:

$$
Accept(e)
=
BranchDeltaConfirm(e)
\land
RiskSafe(e)
\land
SupportBalanced(e).
$$

where:

$$
BranchDeltaConfirm(e)
=
\mathbb{1}[S_{\text{custom-delta}}(e)\geq\tau_g],
$$

$$
RiskSafe(e)
=
\mathbb{1}[S_{\text{risk}}(e)\leq\rho],
$$

$$
SupportBalanced(e)
=
\mathbb{1}[Rel(family(e))\geq r_0]
\cdot
\mathbb{1}[Density(e)\geq d_0].
$$

Optional candidate gate is allowed only as amortization:

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

If custom branch-delta is cheap enough, use:

$$
Candidate(e)=1.
$$

Controller heldout pass：

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

---

## 5. Support expansion success

v9.2.54 still measured only 3 signal strata. v9.2.55 must expand support in the same runner, not after another cycle.

Measurement pass：

```text
natural_real_event_count >= 12000
balanced_diagnostic_real_event_count >= 6000
measured_signal_strata_count >= 6
measured_family_count >= 12
```

Official split：

```text
natural rows:
  official controller / leave-out / paired replay

balanced diagnostic rows:
  failure analysis / support analysis only
```

Balanced rows cannot be mixed into official heldout gate unless calibration / heldout / diagnostic roles are explicitly separated.

---

## 6. Local functional causality success

Only after custom branch-delta + controller + support gate pass.

Leave-dataset-out：

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

Leave-stratum-out：

At least $70\%$ held-out strata task-safe and：

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

## H1：v9.2.54 的 failure 是 true extension 缺失，不是 value signal 消失

Evidence:

```text
CBD0 AUC = 0.888401
CBD0 agreement = 1.0
oracle precision = 1.0
oracle coverage = 0.121693
oracle bad-event = 0.0
true custom branch-delta implemented = 0
```

H1 成立标准：

A true custom extension reaches:

$$
AUC_{\text{safe-good}}\geq0.70
$$

or:

$$
Corr_{\text{safe-grounded}}\geq0.35,
$$

with:

$$
Agreement_{\text{accept}}\geq0.90.
$$

H1 失败标准：

All true extension candidates:

```text
AUC < 0.60
or agreement < 0.80
```

If H1 fails, exact branch-delta information is not preserved by current lower-level implementation.

---

## H2：K7 reduction can be fused without losing exact signal

K7 ratio is `0.662327`. H2 says K7 is mostly implementation fragmentation / reduction overhead, not irreducible math.

H2 成立标准：

$$
T_{\text{K7,fused}}\leq0.35T_{\text{K7,ref}},
$$

and:

$$
Agreement_{\text{accept}}\geq0.90.
$$

H2 失败标准：

```text
K7 remains > 50% of step after fused implementation
or fused reduction destroys accept agreement
```

---

## H3：CBD1 failure means formula shortcuts are not enough

CBD1 got step ratio `1.35` but agreement `0.386822` and AUC `0.534988`. H3 says cheap formula kernel should be treated as negative control.

H3 成立标准：

Formula kernels remain diagnostic only, and route selection refuses official promotion unless true branch-delta correctness is present.

H3 失败标准：

Any result promotes formula proxy as official success without true branch-delta implementation.

---

## H4：candidate generator should not block if custom branch-delta is cheap enough

Candidate generators repeatedly failed recall/safety. H4 says once custom branch-delta is cheap enough, all-pass or broad-pass candidate mode should be tested.

H4 成立标准：

All-pass branch-delta controller achieves:

$$
Coverage\in[0.03,0.15],
$$

$$
Precision\geq0.75,
$$

$$
BadEventRate\leq0.05.
$$

H4 失败标准：

All-pass branch-delta has high bad-event or low precision. Then candidate/risk/support filters remain necessary.

---

## H5：oracle support still exists; no carrier reset unless oracle collapses

H5 成立标准：

fresh v9.2.55 natural rows:

$$
OraclePrecision_{\text{safe-good}}\geq0.75,
$$

$$
OracleCoverage_{\text{safe-good}}\in[0.03,0.15],
$$

$$
OracleBadEventRate\leq0.05.
$$

H5 失败标准：

```text
fresh oracle precision < 0.75
or coverage < 0.03
or bad-event > 0.05
```

If H5 fails, return to carrier/support design. If H5 holds but controller fails, continue legal/system/controller.

---

# Part IV. 并行执行设计

v9.2.55 runner must execute these lanes in parallel:

```text
Lane A:
  v9.2.54 boundary reproduction

Lane B:
  K7 residual subphase attribution:
    gap reduction
    risk reduction
    support lookup
    metric reduction
    sync / launch / allocation

Lane C:
  true custom branch-delta extension:
    C++/CUDA extension
    Triton v2 true branch-delta
    selected/full logits
    fused metric reduction

Lane D:
  formula kernel negative-control audit:
    CBD1-like formula shortcuts
    explicitly non-official unless true branch-delta correctness holds

Lane E:
  exact-signal controller:
    branch-delta primary
    risk-safe
    support-balanced
    optional broad candidate mode

Lane F:
  support expansion:
    natural rows
    balanced diagnostic rows
    strata / family coverage

Lane G:
  LDO / LSO

Lane H:
  paired replay scout and official replay

Lane I:
  short-run scout if paired replay passes

Lane J:
  system audit:
    kernel count
    sync count
    read/write MB
    temp allocation
    occupancy if available
```

Gate discipline:

```text
diagnostic rows may be measured early
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  true custom branch-delta implemented
  custom branch-delta predictivity pass
  custom branch-delta agreement pass
  custom branch-delta system pass
  exact-signal controller pass
  LDO/LSO pass
```

---

# Part V. Candidate designs

## 1. Base and carrier

Official base remains:

```text
R2-LQ-fanin-output-scale-confirmed
LQ-t2-h256
```

Carrier candidates:

```text
A0-current-v9254
A3-RoleWiseFT7ResetCarrier
A6-HybridRoleControlRiskCarrier
```

A1 risk-bounded carrier remains diagnostic unless carrier-active gate passes:

$$
r_{z,\text{tail}}\geq0.10,
$$

$$
r_{\perp,\text{tail}}\geq0.10.
$$

---

## 2. K7 residual attribution candidates

### RCA0：v9.2.54 CBD0 reference

Reference only.

### RCA1：K7 gap-only reduction

Measure only gap score reduction.

### RCA2：K7 risk-only reduction

Measure only risk score reduction.

### RCA3：K7 support-only reduction

Measure only support / family / density reduction.

### RCA4：K7 fused metric reduction smoke

Fuse gap + risk + support but without custom branch-delta math, to isolate reduction cost.

### RCA5：K7 no-sync delayed logging

Keep event scores on device until bulk flush.

---

## 3. True custom branch-delta candidates

### CBD0：v9.2.54 FBD6 reference

Reference only. Predictive but too expensive.

### CBD1：v9.2.54 Triton formula selected-logit negative control

Diagnostic only. It is explicitly not official unless true branch-delta correctness is added.

### CBD2：CUDASelectedLogitBranchDeltaExtension

C++/CUDA extension computing selected branch-delta logits:

```text
true class
top-1
top-2
hard negative
tail class
```

for branches:

```text
RealFunctional
AdamWParallel
bestLR
NoOp
```

### CBD3：CUDAFullLogitSmallCBranchDeltaExtension

For 10-class datasets, compute all logits in one kernel. This is allowed as current FC validation but must be labeled small-C specialization.

### CBD4：CUDABranchDeltaMetricFusedExtension

Compute branch deltas and score components in one extension:

```text
gap_score
risk_score
support_score
accept_components
```

### CBD5：TritonV2TrueBranchDelta

Allowed only if it computes true branch-delta, not formula proxy. Must pass agreement and correctness.

### CBD6：HybridCustomBranchDelta

Pipeline:

```text
CUDA selected/full branch-delta
+ fused K7 reduction
+ delayed bulk logging
+ optional exact fallback
```

---

## 4. Controller candidates

### C0：v9.2.54 reference

Reference only.

### C1：AllPassExactSignalController

No cheap candidate gate:

$$
Accept(e)
=
BranchDeltaConfirm(e)
\land
RiskSafe(e)
\land
SupportBalanced(e).
$$

### C2：BroadCandidateExactSignalController

Candidate only removes obviously unsafe/inactive rows:

$$
Accept(e)
=
BroadCandidate(e)
\land
BranchDeltaConfirm(e)
\land
RiskSafe(e)
\land
SupportBalanced(e).
$$

### C3：FamilyBalancedExactSignalController

$$
Accept(e)
=
BranchDeltaConfirm(e)
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

### C4：BorderlineExactFallbackController

$$
Accept(e)
=
HighConfidenceCustomDelta(e)
\lor
[
Borderline(e)
\land
ExactFallback(e)
].
$$

### C5：Oracle

Posthoc diagnostic only. Never official.

---

# Part VI. 实验阶段

## P0：v9.2.54 boundary reproduction

### 目标

确认 v9.2.54 boundary 稳定。

### 必须记录

```text
route
source_route_v9253
fbd6_residual_attribution_pass
dominant_fbd6_residual_subphase
CBD0_AUC
CBD0_agreement
CBD0_step_ratio
CBD1_AUC
CBD1_agreement
CBD1_step_ratio
true_custom_branch_delta_implemented
cuda_extension_status
controller_official_eligible_count
oracle_support_pass
oracle_precision
oracle_coverage
oracle_bad_event
measured_signal_strata_count
support_measurement_pass
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R14-NeedsLowerLevelCUDAExtension
true custom branch-delta implemented = 0
CBD0 signal retained
CBD1 formula not official
oracle support pass = 1
fake/proxy/offload = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_exact_signal_vs_formula_shortcut.svg
p0_route_ladder.svg
```

---

## P1：K7 residual subphase attribution

### 目标

拆开 K7-gap_risk_support_reduction。

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
class_count
selected_class_count
host_item_count
python_dispatch_count
unknown_fraction
```

Subphases：

```text
K7a-branch_logits_gather
K7b-ce_margin_selected_metric
K7c-real_gain_compute
K7d-control_gain_compute
K7e-gap_score_reduction
K7f-risk_score_reduction
K7g-support_family_lookup
K7h-accept_component_construct
K7i-temp_allocation
K7j-kernel_launch_sync
K7k-logging_hash_timestamp
```

### 判断标准

P1 pass：

```text
unknown_fraction <= 0.10
dominant_k7_subphase_identified = 1
subphase sum within ±0.05 of K7 total
```

### 可视化

```text
p1_k7_residual_waterfall.svg
p1_k7_kernel_count.svg
p1_k7_memory_traffic.svg
p1_k7_sync_and_allocation.svg
```

---

## P2：True custom branch-delta implementation matrix

### 目标

实现并比较 CBD2-CBD6。CBD5/CUDA-extension not implemented 不能再被默默跳过；必须落盘为 implemented 或 explicit not_implemented route。

### 必须记录

```text
custom_delta_id
implementation_status
kernel_level
uses_cuda_extension
uses_triton
uses_true_branch_delta
uses_formula_proxy
uses_full_logits
uses_selected_logits
uses_full_branch_forward
uses_autograd_graph
AUC_safe_good
corr_safe_grounded
agreement_exact_accept
gap_error_mean
gap_error_p95
CE_error
margin_error
risk_error
precision_at_gate
coverage_at_gate
bad_event_at_gate
per_probe_overhead_q90
amortized_overhead
step_ratio_q50
step_ratio_q90
memory_ratio
read_MB
write_MB
kernel_count
sync_count
dataset_name_used
posthoc_used_at_commit
```

### 判断标准

Implementation pass：

```text
implementation_status = implemented
uses_true_branch_delta = 1
uses_formula_proxy = 0
```

Predictivity pass：

$$
AUC_{\text{safe-good}}\geq0.70
$$

or:

$$
Corr_{\text{safe-grounded}}\geq0.35.
$$

Agreement pass：

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
p2_custom_delta_auc_cost_pareto.svg
p2_custom_delta_agreement.svg
p2_custom_delta_error_distribution.svg
p2_formula_vs_true_delta_comparison.svg
p2_custom_delta_memory_traffic.svg
```

---

## P3：Exact-signal controller calibration

### 目标

用 branch-delta as primary value signal，避免 weak candidate generator 主导。

### Split design

```text
calibration split:
  choose thresholds / coefficients

heldout natural split:
  official local controller gate

balanced diagnostic split:
  support/failure analysis only

leave-dataset-out:
  P5

leave-stratum-out:
  P5
```

### 必须记录

```text
controller_id
custom_delta_id
candidate_mode
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

Support：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

### 可视化

```text
p3_controller_precision_coverage_bad.svg
p3_controller_cost_vs_value.svg
p3_controller_family_coverage.svg
p3_branch_delta_threshold_curve.svg
p3_oracle_legal_gap.svg
```

---

## P4：Support expansion

### 目标

扩大 support measurement，避免 3-strata narrow conclusion。

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
custom_delta_id
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
natural_real_event_count >= 12000
balanced_diagnostic_real_event_count >= 6000
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
p4_signal_strata_coverage.svg
p4_family_support_heatmap.svg
p4_oracle_support_by_stratum.svg
p4_natural_vs_balanced_distribution.svg
```

---

## P5：Leave-dataset-out / leave-stratum-out

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
custom_delta_id
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
p5_leave_dataset_out_matrix.svg
p5_leave_stratum_out_matrix.svg
p5_hidden_dataset_tuning_audit.svg
p5_leaveout_failure_modes.svg
```

---

## P6：Official paired replay

### 目标

验证 RealFunctional 是否在 strong controls 下有局部因果优势。

### 设置

```text
base = R2 repaired base checkpoint
controller_id = best P5 survivor
custom_delta_id = best P5 survivor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random,
           ShuffledCustomBranchDelta, ShuffledRiskScore, ShuffledSupportBalance,
           ShuffledCandidateGate, ShuffledBranchRatio, ShuffledControlGap,
           ShuffledValueScore, ShuffledSignalChannel, FunctionalChannelShuffled,
           TailMaskShuffled, RoleScoreShuffled, DatasetRouteShuffled,
           EventRouteShuffled, InvertedRoleMask
```

### 必须记录

```text
controller_id
custom_delta_id
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
ShuffledCustomBranchDelta = fail
ShuffledRiskScore = fail
ShuffledSupportBalance = fail
ShuffledCandidateGate = fail
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
p6_official_paired_replay_pareto.svg
p6_macro_beat_rate.svg
p6_signal_stratum_win_matrix.svg
p6_shuffle_control_matrix.svg
p6_system_gate_distribution.svg
```

---

## P7：Short-run scout

### 目标

如果 P6 pass，验证局部 paired replay 优势能否在连续训练中保持。

### 设置

```text
steps = 50,240,640
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledCustomBranchDelta
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
p7_short_run_task_mechanism_pareto.svg
p7_short_run_controls.svg
p7_event_timeline.svg
p7_ce_tail_margin_panel.svg
```

---

## P8：Full 10-seed validation

### 目标

如果 short-run pass，验证 full functional route。

### 设置

```text
epochs = 20
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
baseline = MLP-match
diagnostic baseline = QuadraticFeatureMLP
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledCustomBranchDelta
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

## P9：Robustness / strong baseline / external-ready

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
contract_audit_v9255.csv
p0_v9254_boundary_reproduction.csv
p1_k7_residual_subphase_attribution.csv
p2_true_custom_branch_delta_implementation_matrix.csv
p3_exact_signal_controller_calibration.csv
p4_online_support_stratum_expansion.csv
p5_leave_dataset_and_stratum_out.csv
p6_official_paired_replay.csv
p7_short_run_functional_validation.csv
p8_full_10seed_functional_validation.csv
p9_robustness_external_ready.csv
k7_residual_trace_v9255.csv
true_custom_branch_delta_trace_v9255.csv
custom_kernel_memory_traffic_trace_v9255.csv
formula_negative_control_trace_v9255.csv
exact_signal_controller_trace_v9255.csv
support_density_trace_v9255.csv
leaveout_trace_v9255.csv
paired_replay_branch_trace_v9255.csv
system_custom_kernel_overhead_trace_v9255.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9254_boundary_unstable
F3_dataset_tuning_detected
F4_k7_residual_unattributed
F5_true_custom_extension_not_implemented
F6_custom_branch_delta_not_predictive
F7_custom_branch_delta_agreement_fail
F8_custom_branch_delta_system_fail
F9_formula_proxy_promoted_illegally
F10_custom_kernel_signal_lost
F11_controller_precision_fail
F12_controller_coverage_fail
F13_controller_bad_event_fail
F14_support_measurement_too_narrow
F15_oracle_support_collapse
F16_leave_dataset_out_fail
F17_leave_stratum_out_fail
F18_paired_replay_control_equivalent
F19_shuffle_control_pass
F20_functional_lr_equivalent
F21_short_run_task_drop
F22_full_run_no_macro_hard_stratum_gain
F23_strong_baseline_explains_gain
F24_robustness_fail
F25_external_not_ready
F26_fake_or_proxy_violation
F27_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-K7ResidualCostAttributed:
  K7 residual gap/risk/support path is fully attributed.

R2-TrueCustomBranchDeltaImplemented:
  at least one true branch-delta extension is implemented and measured.

R3-CustomBranchDeltaPredictive:
  true custom branch-delta preserves safe-good signal.

R4-CustomBranchDeltaSystemPass:
  true custom branch-delta passes step/memory system envelope.

R5-ExactSignalControllerPass:
  branch-delta-primary controller passes heldout precision / coverage / bad-event / system gate.

R6-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R7-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R8-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R9-TrueCustomExtensionNotImplemented:
  lower-level true branch-delta extension was not implemented.

R10-CustomBranchDeltaPredictiveButStillTooExpensive:
  true custom branch-delta preserves signal but remains outside system envelope.

R11-CustomBranchDeltaSystemPassButSignalLost:
  custom kernel is cheap but loses exact branch-delta signal.

R12-ControllerStillCoverageLimited:
  precision and safety are acceptable but coverage remains below 0.03.

R13-ControllerUnsafe:
  coverage exists but bad-event remains above 0.05.

R14-OracleSupportCollapse:
  fresh natural replay no longer has enough safe-good support.

R15-NeedsLowerLevelCUDAExtensionAgain:
  Triton / torch-level implementation cannot close system gate and C++/CUDA path is still required.

R16-StrictPureKANFunctionalShortRunPass:
  short-run task-safe mechanism gain.

R17-StrictPureKANFunctionalFullPass:
  full 10-seed macro / hard-stratum / geometry gain.

R18-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` 必须记录：

```text
route
v9254_boundary_pass
dataset_tuning_detected
k7_residual_attribution_pass
dominant_k7_subphase
true_custom_branch_delta_implemented
best_custom_delta_id
uses_true_branch_delta
uses_formula_proxy
custom_delta_predictivity_pass
custom_delta_system_pass
custom_delta_auc
custom_delta_corr
custom_delta_accept_agreement
custom_delta_step_ratio_q90
custom_delta_memory_ratio
formula_proxy_negative_control_pass
best_controller_id
exact_signal_controller_pass
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
support_measurement_pass
leave_dataset_out_pass
leave_stratum_out_pass
paired_replay_pass
short_run_pass
full_run_pass
external_ready
primary_blocker
next_required_implementation
success_v9255_strict_purekan_functional
success_v9255_full_functional
success_v9255_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 K7 residual subphase attribution
  P2 true custom branch-delta implementation matrix
  P4 support expansion

Batch 2:
  P3 exact-signal controller calibration
  P5 leave-dataset-out / leave-stratum-out
  P6 paired replay scout

Batch 3:
  official P6 paired replay
  P7 short-run if paired replay passes

Batch 4:
  P8 full 10-seed
  P9 robustness / strong baseline
```

Gate rule：

```text
P3/P6 diagnostic rows may be measured before all gates finish.
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  true custom branch-delta implemented
  custom branch-delta predictivity pass
  custom branch-delta agreement pass
  custom branch-delta system pass
  exact-signal controller pass
  LDO/LSO pass
```

---

# Part X. 停止条件

## Minimum diagnostic success

```text
v9.2.54 boundary reproduced
K7 residual cost attributed
at least one true custom branch-delta candidate implemented or explicitly route as not implemented
formula proxy audited as non-official
exact-signal controller measured
support expansion measured
no fake/proxy/offload/loss/teacher violation
```

## Branch-delta system success

```text
Minimum diagnostic success
+
at least one true custom branch-delta path predicts safe-good
+
system overhead gate pass
+
accept agreement pass
```

## Legal controller success

```text
Branch-delta system success
+
branch-delta-primary controller heldout pass
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
1. v9.2.54 boundary cannot be reproduced；
2. K7 residual cost cannot be attributed；
3. no true custom branch-delta extension is implemented；
4. true custom branch-delta loses safe-good predictivity；
5. true custom branch-delta remains too expensive；
6. true custom branch-delta passes system but agreement fails；
7. formula proxy is the only cheap path and cannot preserve signal；
8. exact-signal controller cannot simultaneously meet precision / coverage / bad-event；
9. online support remains too narrow；
10. fresh natural oracle support collapses；
11. leave-dataset-out fails；
12. leave-stratum-out fails；
13. paired replay remains control-equivalent；
14. shuffle controls pass, indicating overfit；
15. short-run task drops；
16. full run gives no macro / hard-stratum / geometry gain；
17. functional breaks system gate；
18. gains are explained by QuadraticFeatureMLP；
19. any teacher/loss/fake/proxy/offload violation occurs。
```

---

# Part XI. 最终解释规则

## Case A：true custom branch-delta + controller + LDO/LSO + paired replay pass

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full validation and external robustness.

## Case B：true custom extension still not implemented

必须声明：

```text
v9.2.55 did not test the real blocker; lower-level branch-delta extension remains unimplemented.
```

下一步必须工程实现 extension，不能再调 controller。

## Case C：custom branch-delta predicts but remains too expensive

必须声明：

```text
safe-good is observable through true branch-output displacement, but current custom implementation is not system-legal.
```

下一步进入 lower-level CUDA/C++ optimization，继续系统实现，不调 dataset。

## Case D：custom branch-delta is cheap but loses signal

必须声明：

```text
current compression loses exact branch-delta information.
```

下一步回到 exact branch-delta compression or richer output-delta statistics.

## Case E：controller fails despite system-legal branch-delta

必须声明：

```text
value is observable and cheap, but accept/abstain support geometry is not deployable.
```

下一步修 support/risk/family controller，不修 base/attach。

## Case F：LDO/LSO fail

必须声明：

```text
controller is not dataset-agnostic or stratum-agnostic enough.
```

不能用 dataset-specific tuning 写成功。

## Case G：oracle support collapses

必须声明：

```text
fresh natural replay no longer has enough safe-good support.
```

下一步回到 carrier/support mechanism。

---

# Part XII. 最终建议

v9.2.55 的一句话策略是：

$$
\boxed{
\text{不要再用 formula proxy 代替 true branch-delta；实现 lower-level exact branch-delta extension，并直接冲 exact-signal controller / LDO / paired replay。}
}
$$

当前最关键的问题不是：

```text
base 是否稳定；
attach 是否污染；
carrier 是否 silent；
CBD0 有没有 AUC；
CBD1 是否够快；
D0 是否还能再省一点；
C0/C1 threshold 是否差一点；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. K7 gap/risk/support reduction 具体贵在哪里？
2. 是否能实现 true custom branch-delta extension，而不是 formula proxy？
3. custom extension 是否保留 CBD0 的 AUC=0.888401 与 agreement=1.0？
4. custom extension 是否能把 step ratio 从 2.863280 压到 <=1.50？
5. branch-delta-primary controller 能否在 heldout 上同时满足 precision / coverage / bad-event / system？
6. support 是否能扩展到 >=6 measured strata？
7. controller 能否 LDO/LSO？
8. official paired replay 能否打过 AdamWParallel / bestLR？
```

v9.2.55 的结果将给出清晰分叉：

```text
if true custom branch-delta + exact-signal controller + LDO/LSO + paired replay pass:
  strict PureKAN functional obtains local causal evidence.

if true custom extension not implemented:
  implementation remains blocker; stop controller tuning.

if custom branch-delta predicts but too expensive:
  lower-level CUDA/C++ optimization remains blocker.

if custom branch-delta cheap but loses signal:
  exact branch-delta information cannot be compressed by current approximation.

if controller local pass but leave-out fails:
  no dataset tuning; broaden signal-stratum/family support.

if oracle support collapses:
  carrier/support stability is blocker.
```
