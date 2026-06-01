# DG-KAN v9.2.54 Custom Fused Branch-Delta Kernel 与 Exact-Signal Controller Closure 完整实验计划

> 本计划基于 v9.2.53 `D0 Vectorized Controller-Prep 与 Real Fused Branch-Delta Kernel` 的真实复盘制定。  
> v9.2.53 的 terminal route 是：
>
> ```text
> route = R12-FusedBranchDeltaPredictiveButNeedsCustomKernel
> base_candidate = LQ-t2-h256
> success_v9253_strict_purekan_functional = False
> success_v9253_full_functional = False
> success_v9253_external_ready = False
> ```
>
> v9.2.53 的关键事实是：
>
> ```text
> P1 D0 micro-attribution:
>   pass = 1
>   dominant subphase = D0a-role_scalar_compute
>   ratio = 0.323653
>
> P2 D0 vectorization:
>   pass = 1
>   best = D0V6-D0VectorizedAll
>   decision agreement = 0.999917
>   D0 time ratio = 0.022203
>   but system step ratio = 3.375423
>
> P3 real fused branch-delta audit:
>   implemented candidate = 1
>   best = FBD6-D0VectorizedFusedDelta
>   AUC = 0.888401
>   agreement = 1.0
>   correctness pass = 1
>   step ratio = 2.863280 > 1.50
>   system pass = 0
>
> P4 selected-logit exactness:
>   not pass
>   best reference = SLR0-V9252-SLD0-reference
>   AUC/agreement retained
>   step ratio = 6.139253
>   low-cost selected repair still lacks official agreement/safety
>
> P5 candidate generator:
>   not pass
>   best = CG0-V9252-CG1-Reference
>   recall = 0.576766
>   candidate bad-event = 0.251535
>
> P6 cascade controller:
>   not pass
>   best = C4-D0VectorizedParetoController
>   precision = 0.152439
>   coverage = 0.040675
>   bad-event = 0.256098
>
> P7 oracle support:
>   oracle precision = 1.0
>   oracle coverage = 0.121693
>   oracle bad-event = 0.0
>   measured signal strata = 3
>   balanced diagnostic rows = 36
>   support measurement pass = 0
>
> current blocker:
>   fused_branch_delta_predictive_but_not_system_legal
> ```
>
> v9.2.54 的核心判断是：
>
> $$
> \boxed{
> \text{D0 已经基本解决，exact-like branch-delta signal 仍然强；下一步不是再调 cheap score，而是做真正 custom fused branch-delta kernel。}
> }
> $$
>
> 换句话说，v9.2.53 把问题从 “controller-prep 是否太慢” 推进成了更硬的系统问题：
>
> $$
> \boxed{
> \text{FBD6 保留 AUC 和 agreement，但 step ratio 仍是 }2.863280>1.50.
> }
> $$
>
> v9.2.54 必须回答：
>
> $$
> \boxed{
> \text{是否存在一个 system-legal 的 exact-signal branch-delta path？}
> }
> $$
>
> 如果存在，立刻进入 controller + LDO/LSO + paired replay；如果不存在，必须明确进入 lower-level CUDA/Triton kernelization stop-go，而不是继续在 Python/torch-level diagnostic 上绕圈。

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
  choose branch-delta approximation separately per dataset
  choose candidate generator separately per dataset
  use validation/test metric at commit time
  use posthoc replay outcome at commit time for official controller
```

Official success 必须满足：

```text
base robust pass
attach equivalence pass
no-event preservation pass
carrier active
branch-delta predictivity pass
branch-delta system gate pass
cascade / exact-signal controller pass
LDO / LSO pass
official paired replay pass
```

---

# Part I. 对 v9.2.53 的独立判断

## 1. v9.2.53 没有达到目标

v9.2.53 不能声明 strict PureKAN functional success。原因不是 paired replay 输了，而是 paired replay 根本没有资格打开：

```text
fused branch-delta system pass = 0
selected-logit exactness pass = 0
candidate generator pass = 0
cascade controller pass = 0
support measurement pass = 0
LDO / LSO / paired replay = not_run
```

因此不能声明：

```text
strict PureKAN local causal evidence
full functional success
external-ready success
Beyond-MLP success
```

## 2. v9.2.53 的真实进展

v9.2.53 有实质进展，而且进展很干净。

第一，D0 vectorization 成功。D0 time ratio 从 v9.2.52 的 dominant D0 blocker 被压到 `0.022203`，decision agreement `0.999917`。这说明 controller scalar preparation / role score path 可以批量化，D0 不再是主要 blocker。

第二，real fused branch-delta 不再只是 not-implemented。FBD6 已经是 implemented candidate，并且保留 exact-like value signal：

$$
AUC=0.888401,
$$

$$
Agreement=1.0.
$$

这非常关键。它证明 exact branch-output displacement 的核心信息没有因为 D0 vectorization 或 fused path 被破坏。

第三，oracle support 继续强：

$$
OraclePrecision=1.0,
$$

$$
OracleCoverage=0.121693,
$$

$$
OracleBadEvent=0.0.
$$

这说明 good events 没有消失。当前失败不是 carrier/support collapse。

## 3. v9.2.53 的真实失败

v9.2.53 的失败也很清楚。FBD6 的 step ratio 仍是：

$$
StepRatio=2.863280.
$$

而 gate 是：

$$
StepRatio\leq1.50.
$$

也就是说，即使 D0 批量化成功、branch-delta signal 保留，system path 仍慢了约：

$$
2.863280 - 1.50 = 1.363280.
$$

这不是 threshold 问题，不是 candidate generator 的局部问题，而是 branch-delta compute path 仍未进入 MLP-comparable system envelope。

同时，P5/P6 说明 cheap candidate / cascade 不能救它：

```text
candidate recall = 0.576766 < 0.80
candidate bad-event = 0.251535 > 0.20
controller precision = 0.152439
controller bad-event = 0.256098
```

这意味着继续调 CG0 / C4 / threshold 不会从根本上解决问题。只要 branch-delta confirm 仍贵，controller 即使局部变好也不能 official；只要 cheap candidate signal 仍弱，controller 会在 coverage 与 bad-event 之间来回振荡。

## 4. 当前 blocker 的本质

当前 blocker 是：

$$
\boxed{
\text{exact value signal 已经存在，但 exact-signal path 仍不是 custom-kernel-native。}
}
$$

更具体地说，v9.2.53 之后剩下的不是 D0，而是 FBD6 的 residual path：

```text
branch-delta tensor generation
selected/full logit delta compute
multi-branch Real / AdamWParallel / bestLR evaluation
metric/gap computation
temporary tensor allocation
kernel launch / sync overhead
possibly Python/torch.compile fragmentation
```

v9.2.54 必须把它拆开，然后实现一个真正 custom fused path，而不是继续用 PyTorch tensor ops 组合出 “fused diagnostic”。

---

# Part II. v9.2.54 总体目标

v9.2.54 的总体目标是：

$$
\boxed{
\text{用 custom fused branch-delta kernel 将 FBD6 的 AUC/agreement 保留下来，并把 step ratio 压到 } \leq 1.50.
}
$$

具体目标分为六层：

```text
1. FBD6 residual system-cost attribution；
2. true custom fused branch-delta kernel implementation；
3. exact-signal controller，不再依赖 weak cheap recall；
4. support expansion；
5. LDO / LSO；
6. official paired replay。
```

---

## 1. Residual branch-delta cost attribution success

v9.2.53 已经说明 D0 可降到很小，但 FBD6 仍 `2.863280`。v9.2.54 必须拆分 FBD6 residual cost。

要拆的 subphases：

```text
K0-D0 vectorized prep reference
K1-functional delta state read / pack
K2-branch delta materialization
K3-selected/full logit delta compute
K4-RealFunctional branch metric
K5-AdamWParallel branch metric
K6-bestLR branch metric
K7-gap / risk / support metric reduction
K8-temporary allocation
K9-kernel launch / sync
K10-logging / hash / timestamp
```

Attribution pass：

```text
unknown_fraction <= 0.10
dominant_residual_subphase_identified = 1
subphase_time_sum_close_to_FBD6 within ±0.05
```

这一步必须回答：

```text
FBD6 仍慢，是因为 logit delta math 重？
还是因为多 branch 重复？
还是因为 torch op fragmentation / kernel launch 多？
还是因为 temporary allocation / sync？
```

## 2. True custom fused branch-delta kernel success

v9.2.54 必须实现至少一个真正的 custom kernel candidate，而不是继续只写 diagnostic matrix。

候选分两档：

```text
Triton-level:
  custom Triton fused selected/full logit branch-delta kernel

CUDA-extension-level:
  if Triton cannot satisfy launch/memory constraints, implement or explicitly route to CUDA-extension requirement
```

Custom fused branch-delta 的目标是：

$$
\widehat z_{b,\mathcal{C}}
=
z_{0,\mathcal{C}}
+
\widehat{\Delta z}_{b,\mathcal{C}},
$$

其中 $b$ 是 branch：

```text
RealFunctional
AdamWParallel
bestLR
NoOp
```

$\mathcal{C}$ 可以是：

```text
selected logits:
  true class
  top-1
  top-2
  hard negative
  optional tail class

or full logits:
  all C classes for small-C datasets
```

Official custom kernel pass：

$$
AUC(\widehat S_{\text{gap}},Y_{\text{safe-good}})\geq0.70
$$

or:

$$
Corr(\widehat S_{\text{gap}},V_{\text{safe-grounded}})\geq0.35.
$$

Agreement:

$$
Agreement_{\text{accept}}\geq0.90.
$$

System:

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

If AUC/agreement pass but step ratio remains above `1.50`, route must be:

```text
R8-CustomBranchDeltaPredictiveButStillTooExpensive
```

If step ratio passes but AUC/agreement fail, route must be:

```text
R9-CustomBranchDeltaSystemPassButSignalLost
```

## 3. Exact-signal controller success

v9.2.53 shows cheap candidate generator is not strong enough. v9.2.54 should not force weak cheap recall as the front gate. Instead, use branch-delta exact signal as primary decision once system gate is solved.

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

Optional candidate generator is allowed only as an amortization layer:

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

But if custom branch-delta is cheap enough, Candidate can be all-pass:

$$
Candidate(e)=1.
$$

Heldout controller pass：

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

Support：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

## 4. Support expansion success

v9.2.53 measured signal strata = `3` and balanced diagnostic rows = `36`，support measurement 仍未过。v9.2.54 必须扩大 support，否则 controller local pass 也不可信。

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
  official controller / leave-out / paired replay basis

balanced diagnostic rows:
  attribution / support analysis only
```

## 5. Leave-out success

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

## 6. Official paired replay success

Only after custom branch-delta system pass + controller pass + LDO/LSO pass.

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
ShuffledCustomBranchDelta = fail
ShuffledRiskScore = fail
ShuffledSupportBalance = fail
ShuffledCandidateGate = fail
FunctionalChannelShuffled = fail
TailMaskShuffled = fail
RoleScoreShuffled = fail
DatasetRouteShuffled = fail
EventRouteShuffled = fail
InvertedRoleMask = fail
```

---

# Part III. 核心假设

## H1：v9.2.53 之后 D0 不再是主 blocker

Evidence：

$$
D0TimeRatio=0.022203,
$$

$$
DecisionAgreement=0.999917.
$$

H1 成立标准：

FBD6 residual attribution confirms D0 accounts for less than `10%` of remaining step overhead.

H1 失败标准：

If D0 still accounts for more than `25%` in a more accurate profiler, then D0 vectorization was not actually deployed on the critical path.

## H2：FBD6 的 residual cost is torch-op fragmentation / branch-delta kernelization issue

H2 成立标准：

Attribution shows at least one of:

```text
kernel launch count high
temporary allocation high
branch-wise repeated compute high
multi-branch logit materialization high
sync / scalar extraction high
```

accounts for at least `60%` of residual overhead.

H2 失败标准：

If residual cost is dominated by unavoidable dense matrix compute that cannot be fused/reused, then branch-delta system route may be fundamentally too expensive at this scale.

## H3：custom fused branch-delta can preserve exact signal

FBD6 already has：

$$
AUC=0.888401,
$$

$$
Agreement=1.0.
$$

H3 成立标准：

At least one custom fused candidate reaches：

$$
AUC_{\text{safe-good}}\geq0.70,
$$

$$
Agreement_{\text{accept}}\geq0.90.
$$

H3 失败标准：

All custom fused candidates either:

```text
AUC < 0.60
or agreement < 0.80
```

If H3 fails, exact gap-probe information is not preserved by the current kernel approximation.

## H4：system target requires true kernel work, not more Python-level controller tuning

H4 成立标准：

Custom fused kernel reduces step ratio from FBD6 `2.863280` by at least `40%` and preferably below `1.50`.

H4 失败标准：

If all Python / torch.compile / Triton smoke candidates stay above `2.00`, next route must explicitly move to lower-level CUDA extension, not another controller matrix.

## H5：cheap candidate generator is optional once custom branch-delta is cheap enough

The candidate generator failed in v9.2.53. H5 says cheap recall should not block route if custom branch-delta is affordable.

H5 成立标准：

All-pass or broad candidate mode plus branch-delta confirm achieves:

$$
Coverage\in[0.03,0.15],
$$

$$
Precision\geq0.75,
$$

$$
BadEventRate\leq0.05.
$$

H5 失败标准：

If all-pass branch-delta controller has high bad-event or low precision, then candidate/risk/support filters remain necessary.

## H6：oracle support still exists; no carrier reset unless oracle collapses

H6 成立标准：

fresh v9.2.54 natural rows：

$$
OraclePrecision_{\text{safe-good}}\geq0.75,
$$

$$
OracleCoverage_{\text{safe-good}}\in[0.03,0.15],
$$

$$
OracleBadEventRate\leq0.05.
$$

H6 失败标准：

```text
fresh oracle precision < 0.75
or coverage < 0.03
or bad-event > 0.05
```

If H6 fails, return to carrier/support design. If H6 holds but controller fails, continue legal/system/controller.

---

# Part IV. 并行执行设计

v9.2.54 runner must execute these lanes in parallel:

```text
Lane A:
  v9.2.53 boundary reproduction

Lane B:
  FBD6 residual cost attribution:
    branch-delta compute
    branch metric
    allocation
    sync
    launch count
    memory traffic

Lane C:
  true custom fused branch-delta kernels:
    Triton selected-logit kernel
    Triton full-logit small-C kernel
    multi-branch fused kernel
    borderline exact confirm
    CUDA-extension placeholder only if actually implemented

Lane D:
  exact-signal controller:
    branch-delta primary score
    risk-safe filter
    support-balanced filter
    optional broad candidate mode

Lane E:
  candidate generator ablation:
    optional only
    not allowed to block if all-pass branch-delta passes

Lane F:
  support expansion:
    natural rows
    balanced diagnostic rows
    strata/family coverage

Lane G:
  LDO / LSO

Lane H:
  paired replay scout and official replay

Lane I:
  short-run scout if paired replay passes

Lane J:
  custom-kernel system audit:
    kernel count
    sync count
    read/write MB
    temp allocation
    occupancy if available
```

Gate discipline：

```text
diagnostic rows may be measured early
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  custom branch-delta predictivity pass
  custom branch-delta system gate pass
  controller pass
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
A0-current-v9253
A3-RoleWiseFT7ResetCarrier
A6-HybridRoleControlRiskCarrier
```

A1 risk-bounded carrier remains diagnostic unless it passes carrier-active gate:

$$
r_{z,\text{tail}}\geq0.10,
$$

$$
r_{\perp,\text{tail}}\geq0.10.
$$

---

## 2. FBD residual cost candidates

### RCA0：FBD6 reference

Reference only.

### RCA1：No-sync FBD6

Remove all host scalar extraction from branch-delta path. Store decisions and metrics in device buffers until block flush.

### RCA2：Preallocated branch buffers

Preallocate:

```text
branch_delta_buffer
selected_logit_buffer
full_logit_buffer
branch_metric_buffer
decision_buffer
```

### RCA3：Multi-branch tensor layout audit

Audit layout:

```text
branch-major
event-major
class-major
```

and record memory coalescing proxy.

### RCA4：Branch metric fusion audit

Check whether gap / risk / CE / margin are computed in separate kernels or fused.

---

## 3. Custom fused branch-delta candidates

### CBD0：FBD6-D0VectorizedFusedDelta reference

Reference only. Predictive but too expensive.

### CBD1：TritonSelectedLogitBranchDelta

Computes selected logits:

```text
true class
top-1
top-2
hard negative
tail class
```

for RealFunctional / AdamWParallel / bestLR in one branch dimension.

### CBD2：TritonFullLogitSmallCBranchDelta

For 10-class MNIST-family only, compute all 10 logits in one fused block. This is allowed as FC current validation, but must be labeled small-C specialization and not claimed as general C-scale solution.

### CBD3：TritonBranchDeltaMetricFused

Computes branch logit deltas and gap/risk metrics in one kernel:

```text
input:
  base logits
  functional delta state
  branch deltas
  labels
  selected class ids

output:
  safe-good score components
  gap score
  risk score
  accept candidates
```

### CBD4：BorderlineExactFallback

Cheap selected-logit branch-delta handles high-confidence rows; exact full-logit confirm only for borderline:

$$
|S_{\text{selected-gap}}-\tau_g|<\epsilon_b.
$$

### CBD5：CUDAExtensionBranchDelta

Only valid if actually implemented. Otherwise record:

```text
not_implemented_cuda_extension_absent
```

and do not count as pass.

### CBD6：HybridCustomBranchDelta

Pipeline:

```text
D0Vectorized prep
+ preallocated buffers
+ Triton multi-branch selected/full logits
+ fused gap/risk metric
+ optional exact fallback
```

---

## 4. Controller candidates

### C0：v9.2.53 reference

Reference only.

### C1：AllPassBranchDeltaController

No cheap candidate generator. Use branch-delta as primary score:

$$
Accept(e)
=
BranchDeltaConfirm(e)
\land
RiskSafe(e)
\land
SupportBalanced(e).
$$

### C2：BroadCandidateBranchDeltaController

Candidate generator only removes obviously unsafe or inactive rows:

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

### C3：BorderlineExactController

$$
Accept(e)
=
HighConfidenceBranchDelta(e)
\lor
[
Borderline(e)
\land
ExactFallback(e)
].
$$

### C4：FamilyBalancedBranchDeltaController

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

### C5：ParetoCostAwareController

Accept if event lies on Pareto frontier of:

```text
positive branch-delta gap
low risk
high support density
family balance
low custom-kernel cost
```

### C6：Oracle

Posthoc diagnostic only. Never official.

---

# Part VI. 实验阶段

## P0：v9.2.53 boundary reproduction

### 目标

确认 v9.2.53 boundary 稳定。

### 必须记录

```text
route
source_route_v9252
d0_vectorization_pass
d0_time_ratio_vs_ref
d0_decision_agreement
real_fused_branch_delta_implemented
best_fused_branch_delta_id
fused_branch_delta_auc
fused_branch_delta_agreement
fused_branch_delta_step_ratio
candidate_generator_recall
candidate_bad_event
cascade_precision
cascade_coverage
cascade_bad_event
oracle_support_pass
oracle_precision
oracle_coverage
oracle_bad_event
measured_signal_strata_count
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R12-FusedBranchDeltaPredictiveButNeedsCustomKernel
D0 vectorization pass = 1
FBD6 AUC/agreement retained
FBD6 system pass = 0
oracle support pass = 1
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_signal_vs_system_ladder.svg
p0_d0_solved_but_fbd_residual.svg
```

---

## P1：FBD6 residual cost attribution

### 目标

拆开 D0 之后的 FBD6 residual cost。

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
K0-D0_vectorized_reference
K1-functional_delta_state_read_pack
K2-branch_delta_materialization
K3-logit_delta_compute
K4-real_branch_metric
K5-adamwparallel_branch_metric
K6-bestlr_branch_metric
K7-gap_risk_support_reduction
K8-temp_allocation
K9-kernel_launch_sync
K10-logging_hash_timestamp
```

### 判断标准

P1 pass：

```text
unknown_fraction <= 0.10
dominant_residual_subphase_identified = 1
subphase sum within ±0.05 of total FBD6 residual
```

### 可视化

```text
p1_fbd6_residual_waterfall.svg
p1_kernel_count_by_subphase.svg
p1_memory_traffic_by_subphase.svg
p1_sync_and_item_count.svg
```

---

## P2：Custom fused branch-delta kernel matrix

### 目标

实现并比较 CBD1-CBD6，确认是否存在 system-legal branch-delta path。

### 必须记录

```text
custom_delta_id
implementation_status
kernel_level
uses_triton
uses_cuda_extension
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
p2_custom_delta_bad_event_curve.svg
p2_custom_delta_memory_traffic.svg
```

---

## P3：Exact-signal controller calibration

### 目标

不再让 weak cheap candidate generator 主导；用 branch-delta as primary signal。

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

Support:

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

扩大 support measurement，避免 narrow stratum conclusion。

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
contract_audit_v9254.csv
p0_v9253_boundary_reproduction.csv
p1_fbd6_residual_cost_attribution.csv
p2_custom_fused_branch_delta_kernel_matrix.csv
p3_exact_signal_controller_calibration.csv
p4_online_support_stratum_expansion.csv
p5_leave_dataset_and_stratum_out.csv
p6_official_paired_replay.csv
p7_short_run_functional_validation.csv
p8_full_10seed_functional_validation.csv
p9_robustness_external_ready.csv
fbd6_residual_trace_v9254.csv
custom_branch_delta_kernel_trace_v9254.csv
custom_kernel_memory_traffic_trace_v9254.csv
exact_signal_controller_trace_v9254.csv
support_density_trace_v9254.csv
leaveout_trace_v9254.csv
paired_replay_branch_trace_v9254.csv
system_custom_kernel_overhead_trace_v9254.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9253_boundary_unstable
F3_dataset_tuning_detected
F4_fbd6_residual_unattributed
F5_custom_kernel_not_implemented
F6_custom_branch_delta_not_predictive
F7_custom_branch_delta_agreement_fail
F8_custom_branch_delta_system_fail
F9_custom_kernel_signal_lost
F10_controller_precision_fail
F11_controller_coverage_fail
F12_controller_bad_event_fail
F13_support_measurement_too_narrow
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

# Part VIII. Route decision

```text
R1-FBD6ResidualCostAttributed:
  FBD6 residual cost after D0 vectorization is fully attributed.

R2-CustomBranchDeltaImplemented:
  at least one true custom fused branch-delta kernel is implemented and measured.

R3-CustomBranchDeltaPredictive:
  custom branch-delta preserves safe-good signal.

R4-CustomBranchDeltaSystemPass:
  custom branch-delta passes step/memory system envelope.

R5-ExactSignalControllerPass:
  branch-delta-primary controller passes heldout precision / coverage / bad-event / system gate.

R6-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R7-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R8-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R9-CustomBranchDeltaPredictiveButStillTooExpensive:
  custom branch-delta preserves signal but remains outside system envelope.

R10-CustomBranchDeltaSystemPassButSignalLost:
  custom kernel is cheap but loses exact branch-delta signal.

R11-ControllerStillCoverageLimited:
  precision and safety are acceptable but coverage remains below 0.03.

R12-ControllerUnsafe:
  coverage exists but bad-event remains above 0.05.

R13-OracleSupportCollapse:
  fresh natural replay no longer has enough safe-good support.

R14-NeedsLowerLevelCUDAExtension:
  Triton / torch-level implementation cannot close system gate, and route requires C++/CUDA kernelization.

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
v9253_boundary_pass
dataset_tuning_detected
fbd6_residual_attribution_pass
dominant_fbd6_residual_subphase
custom_branch_delta_implemented
best_custom_delta_id
custom_delta_predictivity_pass
custom_delta_system_pass
custom_delta_auc
custom_delta_corr
custom_delta_accept_agreement
custom_delta_step_ratio_q90
custom_delta_memory_ratio
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
success_v9254_strict_purekan_functional
success_v9254_full_functional
success_v9254_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 FBD6 residual cost attribution
  P2 custom fused branch-delta kernel matrix

Batch 2:
  P3 exact-signal controller calibration
  P4 online support / stratum expansion
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
  custom branch-delta predictivity pass
  custom branch-delta system gate pass
  exact-signal controller pass
  LDO/LSO pass
```

---

# Part X. 停止条件

## Minimum diagnostic success

```text
v9.2.53 boundary reproduced
FBD6 residual cost attributed
at least one custom branch-delta candidate implemented or explicitly routed to not-implemented
exact-signal controller measured
support expansion measured
no fake/proxy/offload/loss/teacher violation
```

## Branch-delta system success

```text
Minimum diagnostic success
+
at least one custom branch-delta path predicts safe-good
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
1. v9.2.53 boundary cannot be reproduced；
2. FBD6 residual cost cannot be attributed；
3. no true custom branch-delta candidate is implemented；
4. custom branch-delta loses safe-good predictivity；
5. custom branch-delta remains too expensive；
6. custom branch-delta passes system but agreement fails；
7. exact-signal controller cannot simultaneously meet precision / coverage / bad-event；
8. online support remains too narrow；
9. fresh natural oracle support collapses；
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

# Part XI. 最终解释规则

## Case A：custom branch-delta + controller + LDO/LSO + paired replay pass

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full validation and external robustness。

## Case B：custom branch-delta predicts but remains too expensive

必须声明：

```text
safe-good is observable through exact branch-output displacement, but current custom kernel is not system-legal.
```

下一步进入 lower-level CUDA/C++ kernelization，不再调 controller threshold。

## Case C：custom branch-delta is cheap but loses signal

必须声明：

```text
current compression loses exact branch-delta information.
```

下一步回到 exact branch-delta compression or richer output-delta statistics。

## Case D：controller fails despite system-legal branch-delta

必须声明：

```text
value is observable and cheap, but accept/abstain support geometry is not deployable.
```

下一步修 support/risk/family controller，不修 base/attach。

## Case E：LDO/LSO fail

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

下一步回到 carrier/support mechanism。

---

# Part XII. 最终建议

v9.2.54 的一句话策略是：

$$
\boxed{
\text{D0 已经基本解决；不要继续调 cheap score。现在要做 true custom fused branch-delta kernel，并用 branch-delta exact signal 直接关闭 controller / LDO / paired replay。}
}
$$

当前最关键的问题不是：

```text
base 是否稳定；
attach 是否污染；
carrier 是否 silent；
FBD6 有没有 AUC；
D0 是否还能再省一点；
C4 threshold 是否差一点；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. FBD6 residual cost 具体来自哪个 branch-delta subphase？
2. 是否能实现真正 custom fused selected/full logit branch-delta kernel？
3. custom kernel 是否保留 FBD6 的 AUC=0.888401 与 agreement=1.0？
4. custom kernel 是否能把 step ratio 从 2.863280 压到 <=1.50？
5. 如果 custom kernel cheap enough，是否可以绕过 weak cheap candidate generator，直接用 branch-delta exact signal 做 controller？
6. controller 能否 heldout pass，并通过 LDO/LSO？
7. official paired replay 能否打过 AdamWParallel / bestLR？
```

v9.2.54 的结果将给出清晰分叉：

```text
if custom branch-delta + exact-signal controller + LDO/LSO + paired replay pass:
  strict PureKAN functional obtains local causal evidence.

if custom branch-delta predicts but too expensive:
  lower-level CUDA/Triton/C++ kernelization is the blocker.

if custom branch-delta system passes but signal lost:
  exact branch-delta information cannot be compressed by current approximation.

if controller local pass but leave-out fails:
  no dataset tuning; broaden signal-stratum/family support.

if oracle support collapses:
  carrier/support stability is blocker.
```
