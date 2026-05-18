# DG-KAN v9.2.57 True Branch-Delta Compute Closure 与 Exact-Signal Controller Feasibility 完整实验计划

> 本计划基于 v9.2.56 `True Branch-Delta Tensor Interface 与 K7d Control-Gain CUDA Fusion` 的真实复盘制定。  
> v9.2.56 的 terminal route 是：
>
> ```text
> route = R12-TrueDeltaPredictiveButTooExpensive
> base_candidate = LQ-t2-h256
> success_v9256_strict_purekan_functional = False
> success_v9256_full_functional = False
> success_v9256_external_ready = False
> ```
>
> v9.2.56 的关键事实是：
>
> ```text
> P1:
>   true branch-delta tensor interface pass = 1
>   uses_true_branch_delta = 1
>   uses_source_measured_gap = 0
>   uses_formula_proxy = 0
>
> P2:
>   K7d CUDA fusion local pass = 1
>   reference K7d time = 2.081355 ms
>   fused K7d time = 0.329207 ms
>   time reduction = 0.841830
>   max abs error = 9.536743e-07
>   reference dominant subphase = K7d7-support_score_compute
>   ratio = 0.629845
>
> P3:
>   best route-level signal = TBD0-CBD0Reference
>   AUC = 0.888401
>   agreement = 1.0
>   step ratio = 2.863280 > 1.50
>   system pass = 0
>
>   true branch-logit candidates TBD1/TBD2/TBD3 are implemented,
>   but none passes system gate.
>
>   TBD3-CUDAK7dTrueBranchDeltaFusion:
>     step ratio = 4.924871
>
> P4:
>   formula proxy / source-measured gap negative controls rejected
>   official_eligible = 0
>
> P5:
>   exact-signal controller not pass
>   best diagnostic precision = 0.147593
>   coverage = 0.182044
>   bad-event = 0.315622
>
> P6:
>   oracle support pass = 1
>   oracle precision = 1.0
>   oracle coverage = 0.121693
>   oracle bad-event = 0.0
>   measured signal strata = 3
>   balanced diagnostic rows = 36
>   support measurement pass = 0
>
> current blocker:
>   true_branch_delta_predictive_but_too_expensive
> ```
>
> v9.2.56 的核心进展是：  
> **真正的 branch-delta tensor interface 已经从 not implemented 变成 implemented，并且 formula/source-measured proxy 被正确排除。K7d control-gain fusion 也在局部通过。**
>
> 但 v9.2.56 也暴露出一个更深的问题：  
> **local K7d fusion 过了，并不等于 true branch-delta end-to-end path 过了。真正的 true branch logits path 仍然太贵；同时 exact-signal controller 的 heldout precision / bad-event 也没有过。**
>
> 因此 v9.2.57 不能继续只做 kernel speed，也不能只做 controller threshold。它必须并行回答两个 stop-go 问题：
>
> $$
> \boxed{
> \text{Q1: true branch-delta compute path 能不能进入 system envelope？}
> }
> $$
>
> $$
> \boxed{
> \text{Q2: exact reference signal 是否存在可部署的 safe / coverage controller region？}
> }
> $$
>
> 如果 Q1 不成立，继续 controller tuning 没意义。  
> 如果 Q2 不成立，继续 kernelization 也不能产生 functional success。  
> v9.2.57 的任务是把这两个问题在同一轮并行验证清楚。

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
  report per-dataset failure mode
  report per-dataset support distribution
  report leave-dataset-out

forbidden:
  if dataset == Fashion: threshold = A
  if dataset == KMNIST: use candidate generator B
  if dataset == MNIST: skip exact branch-delta
  tune threshold separately per dataset
  choose selected/full-logit mode separately per dataset
  use validation/test metric at commit time
  use posthoc replay outcome at commit time for official controller
```

Official success 必须满足：

```text
base robust pass
attach equivalence pass
no-event preservation pass
carrier active
uses_true_branch_delta = 1
uses_source_measured_gap = 0
uses_formula_proxy = 0
true branch-delta predictivity pass
true branch-delta agreement pass
true branch-delta system pass
exact-signal controller heldout pass
LDO / LSO pass
official paired replay pass
```

---

# Part I. 对 v9.2.56 的独立判断

## 1. v9.2.56 没有达到目标

v9.2.56 没有 strict PureKAN functional success。原因不是 paired replay 失败，而是 paired replay 仍没有资格打开：

```text
true branch-delta system pass = 0
exact-signal controller pass = 0
support measurement pass = 0
LDO / LSO / paired replay = not_run
```

`TBD0-CBD0Reference` 的 AUC 和 agreement 很强，但 step ratio `2.863280`。  
`TBD1/TBD2/TBD3` 作为 true branch-logit candidates 已经实现，但没有过 system gate。  
`TBD3` 甚至 step ratio `4.924871`，说明当前 true logits path 并没有因为 K7d fusion 而整体变快。

因此不能声明：

```text
strict PureKAN local causal evidence
full functional success
external-ready success
Beyond-MLP success
```

## 2. v9.2.56 的真实进展

v9.2.56 是一个重要进展，不是原地失败。

第一，P1 通过了 true branch-delta tensor interface gate。此前 v9.2.55 的核心 blocker 是 `true_custom_branch_delta_implemented = 0`，而 v9.2.56 已经明确：

```text
uses_true_branch_delta = 1
uses_source_measured_gap = 0
uses_formula_proxy = 0
```

这意味着我们终于不再用 source-measured scalar gap 或 formula shortcut 代替真实 branch-delta。

第二，K7d control-gain CUDA fusion 在局部通过：

$$
T_{\text{K7d,ref}}=2.081355ms,
$$

$$
T_{\text{K7d,fused}}=0.329207ms,
$$

$$
\text{time reduction}=0.841830.
$$

最大绝对误差只有：

$$
9.536743\times10^{-7}.
$$

这说明 control-gain reduction 本身可以被正确且高效地融合。

第三，negative controls 被正确拒绝。formula proxy 与 source-measured gap 都没有被倒灌成 official success。这非常重要，因为 v9.2.54-v9.2.55 最大风险就是把“快但错”的 proxy 写成成功。

第四，oracle support 继续强：

$$
OraclePrecision=1.0,
$$

$$
OracleCoverage=0.121693,
$$

$$
OracleBadEvent=0.0.
$$

这说明 safe-good events 仍然存在，不是 carrier/support collapse。

## 3. v9.2.56 的真实失败

v9.2.56 的失败不是单一系统失败，而是两个 blocker 同时存在。

### 3.1 Compute blocker：true branch-delta end-to-end path 仍然太贵

局部 K7d fusion 过了，但 end-to-end true branch-delta path 没过：

```text
TBD0 reference step ratio = 2.863280
TBD3 true CUDA K7d fusion step ratio = 4.924871
system gate = 0
```

这说明现在不能再说 “K7d fusion 没做”。K7d 局部确实做了，而且做对了。  
真正问题是：**true branch logits / branch delta tensor path 的其他部分正在吞掉收益，甚至让 TBD3 比 reference 更慢。**

可能来源包括：

```text
branch-delta tensor layout 不连续；
selected/full logits gather/scatter 过重；
extension input/output marshalling 太贵；
branch dimension 与 event dimension 排列不适合 coalescing；
delta state read / pack 重复；
support/risk tensors 在 kernel 外又被重新构造；
launch/sync 或 Python bridge fragmentation；
TBD3 把 K7d 做快了，但引入了更贵的 branch-logit compute / copy。
```

### 3.2 Decision blocker：exact-signal controller 目前不安全

P5 exact-signal controller 的 best diagnostic 结果是：

```text
precision = 0.147593
coverage = 0.182044
bad-event = 0.315622
```

这说明即使 system path 被压下去，当前 controller 也不能直接 official。  
AUC `0.888401` 很强，但 AUC 不等于可部署 policy。我们必须验证 exact reference signal 是否存在一个 threshold / risk / support region，能同时满足：

$$
Precision\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05.
$$

如果 exact reference 自身都找不到这样的 region，那么继续 kernelization 只能得到一个“便宜但不能用”的 signal。

### 3.3 Support blocker：measurement 仍太窄

P6 仍只有：

```text
measured signal strata = 3
balanced diagnostic rows = 36
support measurement pass = 0
```

这已经连续多轮存在。它会让 controller calibration 很容易变成 narrow-family artifact。  
所以 v9.2.57 必须并行扩大 support measurement，不能等 kernel 做完以后再补。

## 4. 当前真正 blocker

当前 blocker 应改写为：

$$
\boxed{
\text{true branch-delta signal exists, but system-legal compute path and deployable accept region are both unclosed.}
}
$$

也就是：

```text
good events exist;
exact branch-delta can identify them;
true branch-delta interface exists;
K7d fusion works locally;
but true branch-delta end-to-end compute is too expensive;
and exact-signal controller currently unsafe / over-coverage / low precision.
```

这不是 basis blocker。  
这不是 dataset blocker。  
这不是 functional update 应该放弃。  
这是 **system primitive + decision geometry** 的双 blocker。

---

# Part II. v9.2.57 总体目标

v9.2.57 的总体目标是：

$$
\boxed{
\text{同时关闭 true branch-delta end-to-end compute gate 与 exact reference controller feasibility gate。}
}
$$

本轮不再一轮只做 kernel，也不再一轮只做 controller。  
必须并行推进：

```text
Lane A:
  true branch-delta compute path attribution

Lane B:
  true branch-delta kernel v2 / layout repair

Lane C:
  exact reference controller feasibility

Lane D:
  exact-signal controller calibration

Lane E:
  support expansion

Lane F:
  LDO / LSO scout

Lane G:
  paired replay scout and official paired replay if gates pass
```

v9.2.57 的核心 stop-go 结论必须是以下之一：

```text
Case 1:
  true branch-delta compute + exact controller both pass
  → enter LDO/LSO + paired replay.

Case 2:
  compute pass but controller fail
  → blocker is support/risk/accept geometry, not system.

Case 3:
  controller feasible under exact reference but compute fail
  → blocker is kernelization.

Case 4:
  exact reference controller infeasible
  → branch-delta score alone insufficient; need richer risk/support sufficient statistics.

Case 5:
  oracle support collapses
  → return to carrier/support reset.
```

---

# Part III. 核心假设

## H1：v9.2.56 的 system failure 来自 true branch-logit path，而不是 K7d reduction 本身

证据：

```text
K7d local fusion pass
T_K7d reduced from 2.081355ms to 0.329207ms
but TBD3 step ratio = 4.924871
```

H1 成立标准：

P1 residual attribution 显示 K7d fused path 之外的 subphase 贡献至少 `60%` of residual overhead，例如：

```text
branch logits compute
delta state read/pack
selected/full logits gather
extension marshaling
layout copy
launch/sync
temporary allocation
```

H1 失败标准：

如果 K7d fusion 在 end-to-end path 中并未真正生效，或者 K7d 仍占 >50% residual，说明 P2 local fusion 是 isolated smoke，不是 deployed critical path。

## H2：true branch-delta compute can be made system-legal only by layout/interface repair, not more formula shortcuts

H2 成立标准：

至少一个 v2 true branch-delta kernel 达到：

$$
AUC_{\text{safe-good}}\geq0.70
$$

or:

$$
Corr_{\text{safe-grounded}}\geq0.35,
$$

and:

$$
Agreement_{\text{accept}}\geq0.90,
$$

and:

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

H2 失败标准：

所有 true branch-delta kernels either:

```text
signal lost:
  AUC < 0.60 or agreement < 0.80

or system fail:
  StepRatio_q90 > 1.50
```

Formula proxy 不计入 H2 成立。

## H3：exact reference signal may have a feasible controller region, but current P5 calibration failed to find it

证据冲突：

```text
AUC = 0.888401 and agreement = 1.0
but controller precision = 0.147593
bad-event = 0.315622
```

H3 成立标准：

CBD0 exact reference threshold/risk/support sweep 中存在 heldout region satisfying：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{heldout}}\leq0.05.
$$

H3 失败标准：

即使使用 exact reference signal，所有 controller regions 都满足不了 precision / coverage / bad-event gate。  
此时 blocker 不是 kernel，而是 branch-delta score lacks sufficient risk/support separation.

## H4：support measurement 太窄会放大 controller failure

H4 成立标准：

support expansion 后：

```text
measured_signal_strata_count >= 6
measured_family_count >= 12
```

并且 controller 的 precision/bad-event variance 明显下降。

H4 失败标准：

support 扩大后 exact controller 仍然 bad-event 高，说明 failure 是 signal semantics，而不是 support coverage artifact。

## H5：oracle support 仍存在；不应回到 carrier reset

H5 成立标准：

fresh v9.2.57 natural rows：

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
oracle precision < 0.75
or oracle coverage < 0.03
or oracle bad-event > 0.05
```

如果 H5 失败，下一步回到 carrier/support reset。  
如果 H5 成立但 controller fail，继续 legal/system/controller route。

---

# Part IV. 并行执行设计

v9.2.57 runner 必须并行执行以下 lanes：

```text
Lane A:
  v9.2.56 boundary reproduction

Lane B:
  true branch-delta residual attribution:
    delta state read/pack
    branch logits compute
    selected/full logits gather
    extension marshaling
    K7d fused deployment
    launch/sync/allocation

Lane C:
  true branch-delta kernel v2:
    layout repair
    branch-major/event-major/class-major variants
    full-logit small-C kernel
    selected-logit exact kernel
    fused branch logits + K7d output scores

Lane D:
  exact reference controller feasibility:
    CBD0 exact reference threshold sweep
    risk/support/family ablation
    precision/coverage/bad-event frontier

Lane E:
  system-legal exact-signal controller:
    only if compute gate passes
    otherwise diagnostic only

Lane F:
  support expansion:
    natural rows
    balanced diagnostic rows
    signal strata / family coverage

Lane G:
  LDO / LSO scout

Lane H:
  paired replay scout and official replay

Lane I:
  short-run scout if paired replay passes
```

Gate discipline：

```text
P5/P8 diagnostic rows may be measured before all gates finish.
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  uses_true_branch_delta = 1
  uses_source_measured_gap = 0
  uses_formula_proxy = 0
  true branch-delta predictivity pass
  true branch-delta agreement pass
  true branch-delta system pass
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
A0-current-v9256
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

## 2. True branch-delta kernel v2 candidates

### TBD0：v9.2.56 CBD0 reference

Reference only. Predictive but too expensive.

```text
uses_true_branch_delta = reference_exact
uses_source_measured_gap = 0
uses_formula_proxy = 0
official_eligible = 0 unless system pass
```

### TBD1：LayoutRepairedFullLogitSmallC

For 10-class datasets, compute all logits in a contiguous full-logit layout.  
This is current FC validation only, not a claim of large-C scalability.

Layout variants:

```text
branch-major:
  [branch, event, class]

event-major:
  [event, branch, class]

class-major:
  [class, event, branch]
```

Measure which layout minimizes memory traffic and kernel launches.

### TBD2：SelectedLogitExactV2

Compute selected logits, but preserve exact accept agreement by including:

```text
true class
current top-1
current top-2
hard negative
previous control-dominant class
tail-risk class
```

Unlike formula proxy, this must compute from true branch delta.

### TBD3：BranchDeltaLogitsPlusK7dFused

Single extension path:

```text
true branch logits
→ real/control gains
→ max control gain
→ gap
→ risk
→ support components
```

Output compact event-level tensors:

```text
gap_score
risk_score
support_score
accept_components
```

### TBD4：TwoStageExactBorderline

Use selected-logit exact v2 for all rows, and full-logit exact only for borderline rows:

$$
|Gap_{\text{selected}}-\tau_g|<\epsilon_b.
$$

Record exact fallback call rate.

### TBD5：DeltaStatePrepacked

Prepack functional/control branch delta state once per step:

```text
delta_real
delta_adamwparallel
delta_bestlr
delta_noop
```

Then pass a contiguous delta state tensor into kernel.

### TBD6：HybridBestLayoutTrueDelta

Combines:

```text
best delta-state packing
best branch/event/class layout
K7d fused scoring
delayed bulk logging
optional borderline exact fallback
```

---

## 3. Exact reference controller candidates

### C0：AllPassExactReferenceController

Uses CBD0 exact reference signal:

$$
Accept(e)
=
Gap(e)\geq\tau_g
\land
RiskSafe(e)
\land
SupportBalanced(e).
$$

No candidate generator.

### C1：RiskFirstExactReferenceController

$$
Accept(e)
=
RiskSafe(e)
\land
Gap(e)\geq\tau_g
\land
SupportBalanced(e).
$$

### C2：SupportBalancedExactReferenceController

$$
Accept(e)
=
Gap(e)\geq\tau_g
\land
Rel(family(e))\geq r_0
\land
Density(e)\geq d_0.
$$

### C3：ParetoExactReferenceController

Accepts events on Pareto frontier of:

```text
positive control gap
low risk
high support density
family balance
low system cost
```

### C4：Oracle

Posthoc diagnostic only. Never official.

---

## 4. System-legal controller candidates

Only evaluated as official if P3 true branch-delta system gate passes.

### C5：AllPassTrueDeltaController

$$
Accept(e)
=
TrueDeltaGap(e)\geq\tau_g
\land
RiskSafe(e)
\land
SupportBalanced(e).
$$

### C6：BroadCandidateTrueDeltaController

Broad candidate only removes obviously inactive rows:

$$
Accept(e)
=
BroadCandidate(e)
\land
TrueDeltaGap(e)\geq\tau_g
\land
RiskSafe(e)
\land
SupportBalanced(e).
$$

### C7：BorderlineFallbackTrueDeltaController

$$
Accept(e)
=
HighConfidenceDelta(e)
\lor
[
Borderline(e)
\land
ExactFallback(e)
].
$$

---

# Part VI. 实验阶段

## P0：v9.2.56 boundary reproduction

### 目标

确认 v9.2.56 boundary 稳定，避免把一次 implementation artifact 误读成科学边界。

### 必须记录

```text
route
source_route_v9255
true_branch_delta_interface_pass
uses_true_branch_delta
uses_source_measured_gap
uses_formula_proxy
k7d_fusion_pass
k7d_time_reduction
CBD0_AUC
CBD0_agreement
CBD0_step_ratio
TBD3_step_ratio
exact_signal_controller_precision
exact_signal_controller_coverage
exact_signal_controller_bad_event
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
route = R12-TrueDeltaPredictiveButTooExpensive
true branch-delta interface pass = 1
K7d fusion pass = 1
CBD0 exact signal retained
true branch-delta system pass = 0
controller pass = 0
oracle support pass = 1
fake/proxy/offload = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_signal_system_controller_ladder.svg
p0_k7d_local_vs_end_to_end_gap.svg
```

---

## P1：true branch-delta residual attribution

### 目标

解释为什么 K7d 局部 fusion 过了，但 true branch-delta end-to-end 仍然太慢。

### 必须记录

```text
row_id
candidate_id
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
event_count
class_count
selected_class_count
layout
host_item_count
python_dispatch_count
unknown_fraction
```

Subphases：

```text
R0-base_logit_or_activation_read
R1-delta_state_read
R2-delta_state_pack
R3-branch_logits_compute_real
R4-branch_logits_compute_adamwparallel
R5-branch_logits_compute_bestlr
R6-selected_or_full_logit_gather
R7-layout_transform_or_copy
R8-K7d_fused_gain_compute
R9-risk_support_component_compute
R10-temp_allocation
R11-kernel_launch_sync
R12-logging_hash_timestamp
```

### 判断标准

P1 pass：

```text
unknown_fraction <= 0.10
dominant_residual_subphase_identified = 1
subphase sum within ±0.05 of total residual
```

### 可视化

```text
p1_true_delta_residual_waterfall.svg
p1_layout_memory_traffic.svg
p1_kernel_count_by_subphase.svg
p1_k7d_deployed_vs_local_time.svg
```

---

## P2：true branch-delta correctness and legality gate

### 目标

防止 source gap / formula proxy 再次混入 official path，同时确认 true branch-delta output 与 exact reference 的一致性。

### 必须记录

```text
candidate_id
implementation_status
uses_true_branch_delta
uses_source_measured_gap
uses_formula_proxy
uses_selected_logits
uses_full_logits
uses_autograd_graph
uses_dataset_name
uses_validation
uses_test
uses_posthoc_replay
commit_time_order_valid
input_tensor_shapes
output_tensor_shapes
branch_count
class_count
selected_class_count
logit_error_mean
logit_error_p95
gain_error_mean
gain_error_p95
gap_error_mean
gap_error_p95
accept_agreement
```

### 判断标准

P2 legality pass：

```text
implementation_status = implemented
uses_true_branch_delta = 1
uses_source_measured_gap = 0
uses_formula_proxy = 0
uses_posthoc_replay = 0
commit_time_order_valid = 1
```

Agreement pass：

$$
Agreement_{\text{accept}}\geq0.90.
$$

### 可视化

```text
p2_legality_matrix.svg
p2_true_delta_error_distribution.svg
p2_accept_agreement_by_candidate.svg
p2_proxy_rejection_table.svg
```

---

## P3：true branch-delta kernel v2 matrix

### 目标

比较 TBD1-TBD6，寻找 system-legal true branch-delta path。

### 必须记录

```text
custom_delta_id
kernel_level
layout
uses_cuda_extension
uses_triton
uses_true_branch_delta
uses_formula_proxy
uses_source_measured_gap
uses_selected_logits
uses_full_logits
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

Strict target：

$$
StepRatio_{q90}\leq1.20.
$$

### 可视化

```text
p3_true_delta_auc_cost_pareto.svg
p3_true_delta_agreement.svg
p3_true_delta_layout_ablation.svg
p3_true_delta_memory_traffic.svg
p3_true_delta_step_ratio_distribution.svg
```

---

## P4：formula / source proxy negative-control audit

### 目标

确保 fast proxy 不能被 route logic 误升为 official。

### 必须记录

```text
proxy_candidate_id
proxy_type
step_ratio_q90
AUC_safe_good
agreement_exact_accept
uses_true_branch_delta
uses_formula_proxy
uses_source_measured_gap
official_eligible
reason_not_official
```

### 判断标准

P4 pass：

```text
all formula/source-measured proxy rows official_eligible = 0
no proxy row can set success_v9257_strict_purekan_functional = 1
```

### 可视化

```text
p4_proxy_speed_signal_matrix.svg
p4_negative_control_summary.svg
```

---

## P5：exact reference controller feasibility audit

### 目标

在 system gate 之前，先回答 exact reference signal 是否存在可部署 controller region。  
这是 v9.2.57 的关键新增 stop-go。

### 必须记录

```text
controller_id
reference_signal = CBD0
thresholds
risk_gate
support_gate
family_gate
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
accepted_strata_count
accepted_family_count
max_family_share
oracle_gap_to_legal
dataset_name_used
posthoc_used_at_commit
official_eligible
```

### 判断标准

Exact reference controller feasible：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{heldout}}\leq0.05.
$$

Support：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

If no reference controller passes, route must be:

```text
R11-ExactSignalControllerInfeasible
```

even if kernel later becomes fast.

### 可视化

```text
p5_reference_precision_coverage_bad_frontier.svg
p5_reference_threshold_surface.svg
p5_reference_risk_support_ablation.svg
p5_oracle_legal_gap.svg
```

---

## P6：system-legal exact-signal controller calibration

### 目标

如果 P3 system pass，则使用 system-legal true branch-delta candidate 建立 official controller。

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
p6_controller_precision_coverage_bad.svg
p6_controller_cost_vs_value.svg
p6_controller_family_coverage.svg
p6_branch_delta_threshold_curve.svg
```

---

## P7：support expansion

### 目标

修复连续多轮 measured strata 太窄的问题。

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
p7_signal_strata_coverage.svg
p7_family_support_heatmap.svg
p7_oracle_support_by_stratum.svg
p7_natural_vs_balanced_distribution.svg
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
controller_id = best P8 survivor
custom_delta_id = best P8 survivor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random,
           ShuffledTrueBranchDelta, ShuffledControlGain, ShuffledRiskScore,
           ShuffledSupportBalance, ShuffledCandidateGate, ShuffledBranchRatio,
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
ShuffledTrueBranchDelta = fail
ShuffledControlGain = fail
ShuffledRiskScore = fail
ShuffledSupportBalance = fail
ShuffledCandidateGate = fail
ShuffledBranchRatio = fail
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
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledTrueBranchDelta
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

---

# Part VII. Required artifacts

```text
run_manifest.json
contract_audit_v9257.csv
p0_v9256_boundary_reproduction.csv
p1_true_branch_delta_residual_attribution.csv
p2_true_branch_delta_correctness_legality_gate.csv
p3_true_branch_delta_kernel_v2_matrix.csv
p4_formula_proxy_negative_control_audit.csv
p5_exact_reference_controller_feasibility.csv
p6_system_legal_exact_signal_controller.csv
p7_online_support_stratum_expansion.csv
p8_leave_dataset_and_stratum_out.csv
p9_official_paired_replay.csv
p10_short_run_functional_validation.csv
true_delta_residual_trace_v9257.csv
true_delta_correctness_trace_v9257.csv
true_delta_kernel_v2_trace_v9257.csv
proxy_negative_control_trace_v9257.csv
reference_controller_feasibility_trace_v9257.csv
system_legal_controller_trace_v9257.csv
support_density_trace_v9257.csv
leaveout_trace_v9257.csv
paired_replay_branch_trace_v9257.csv
system_true_delta_kernel_overhead_trace_v9257.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9256_boundary_unstable
F3_dataset_tuning_detected
F4_true_delta_residual_unattributed
F5_k7d_local_fusion_not_deployed_end_to_end
F6_true_delta_legality_fail
F7_source_measured_gap_used
F8_formula_proxy_promoted_illegally
F9_true_delta_not_predictive
F10_true_delta_agreement_fail
F11_true_delta_system_fail
F12_reference_controller_infeasible
F13_controller_precision_fail
F14_controller_coverage_fail
F15_controller_bad_event_fail
F16_support_measurement_too_narrow
F17_oracle_support_collapse
F18_leave_dataset_out_fail
F19_leave_stratum_out_fail
F20_paired_replay_control_equivalent
F21_shuffle_control_pass
F22_functional_lr_equivalent
F23_short_run_task_drop
F24_full_run_no_macro_hard_stratum_gain
F25_strong_baseline_explains_gain
F26_robustness_fail
F27_external_not_ready
F28_fake_or_proxy_violation
F29_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-TrueDeltaResidualAttributed:
  true branch-delta residual compute path is attributed.

R2-TrueDeltaLegalityPass:
  true branch-delta candidate passes no-proxy / no-source-gap legality gate.

R3-TrueDeltaKernelV2Predictive:
  true branch-delta v2 preserves safe-good signal.

R4-TrueDeltaKernelV2SystemPass:
  true branch-delta v2 passes step/memory envelope.

R5-ExactReferenceControllerFeasible:
  exact reference signal admits safe coverage-preserving controller region.

R6-SystemLegalExactSignalControllerPass:
  system-legal true branch-delta controller passes heldout gate.

R7-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R8-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R9-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R10-K7dFusionLocalButNotEndToEnd:
  K7d local fusion works, but end-to-end path does not deploy the speedup.

R11-ExactSignalControllerInfeasible:
  exact reference signal cannot produce precision / coverage / bad-event feasible region.

R12-TrueDeltaPredictiveButTooExpensiveAgain:
  true branch-delta preserves signal but remains outside system envelope.

R13-TrueDeltaSystemPassButSignalLost:
  true branch-delta path is cheap but loses exact signal.

R14-ControllerStillCoverageLimited:
  precision/safety acceptable but coverage below 0.03.

R15-ControllerUnsafe:
  coverage exists but bad-event above 0.05.

R16-OracleSupportCollapse:
  fresh natural replay no longer has enough safe-good support.

R17-StrictPureKANFunctionalShortRunPass:
  short-run task-safe mechanism gain.

R18-StrictPureKANFunctionalFullPass:
  full 10-seed macro / hard-stratum / geometry gain.

R19-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` 必须记录：

```text
route
v9256_boundary_pass
dataset_tuning_detected
true_delta_residual_attribution_pass
dominant_true_delta_residual_subphase
k7d_local_fusion_deployed_end_to_end
true_delta_legality_pass
uses_true_branch_delta
uses_source_measured_gap
uses_formula_proxy
best_true_delta_id
true_delta_predictivity_pass
true_delta_agreement_pass
true_delta_system_pass
true_delta_auc
true_delta_corr
true_delta_accept_agreement
true_delta_step_ratio_q90
true_delta_memory_ratio
reference_controller_feasible
reference_controller_precision
reference_controller_coverage
reference_controller_bad_event
best_controller_id
system_legal_controller_pass
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
success_v9257_strict_purekan_functional
success_v9257_full_functional
success_v9257_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 true branch-delta residual attribution
  P2 true branch-delta correctness / legality gate
  P3 true branch-delta kernel v2 matrix
  P4 formula proxy audit
  P5 exact reference controller feasibility
  P7 support expansion

Batch 2:
  P6 system-legal exact-signal controller calibration
  P8 leave-dataset-out / leave-stratum-out
  P9 paired replay scout

Batch 3:
  official P9 paired replay
  P10 short-run if paired replay passes

Batch 4:
  full 10-seed / robustness / strong baseline only if P10 passes
```

Gate rule：

```text
P5 reference feasibility can run before system pass.
P6 official controller cannot pass unless P3 system pass.
P8/P9 diagnostic rows may be measured before all gates finish.
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  true branch-delta legality pass
  true branch-delta predictivity pass
  true branch-delta agreement pass
  true branch-delta system pass
  system-legal controller pass
  LDO/LSO pass
```

---

# Part X. 停止条件

## Minimum diagnostic success

```text
v9.2.56 boundary reproduced
true branch-delta residual cost attributed
true branch-delta legality gate measured
true branch-delta kernel v2 matrix measured
formula/source proxy rejected
exact reference controller feasibility measured
support expansion measured
no fake/proxy/offload/loss/teacher violation
```

## Compute success

```text
Minimum diagnostic success
+
at least one true branch-delta path predicts safe-good
+
accept agreement pass
+
system overhead gate pass
```

## Decision success

```text
exact reference controller feasible
+
system-legal true branch-delta controller heldout pass
+
multi-stratum / multi-family pass
```

## Local functional success

```text
Compute success
+
Decision success
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
1. v9.2.56 boundary cannot be reproduced；
2. true branch-delta residual cannot be attributed；
3. source-measured gap input appears in official path；
4. formula proxy is promoted illegally；
5. K7d local fusion is not deployed end-to-end；
6. true branch-delta loses safe-good predictivity；
7. true branch-delta remains too expensive；
8. true branch-delta passes system but agreement fails；
9. exact reference controller infeasible；
10. exact-signal controller cannot simultaneously meet precision / coverage / bad-event；
11. online support remains too narrow；
12. fresh natural oracle support collapses；
13. leave-dataset-out fails；
14. leave-stratum-out fails；
15. paired replay remains control-equivalent；
16. shuffle controls pass, indicating overfit；
17. short-run task drops；
18. full run gives no macro / hard-stratum / geometry gain；
19. functional breaks system gate；
20. gains are explained by QuadraticFeatureMLP；
21. any teacher/loss/fake/proxy/offload violation occurs。
```

---

# Part XI. 最终解释规则

## Case A：true delta kernel + exact controller + LDO/LSO + paired replay pass

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full validation and external robustness。

## Case B：exact reference controller infeasible

必须声明：

```text
exact branch-delta score has high AUC but cannot form a deployable accept/abstain policy under current risk/support gates.
```

下一步应修 risk/support sufficient statistics，而不是继续 kernelization。

## Case C：controller feasible but true delta compute too expensive

必须声明：

```text
decision geometry is viable, but system kernelization remains blocker.
```

下一步继续 true branch-delta kernel implementation，不调 dataset。

## Case D：true delta system pass but signal lost

必须声明：

```text
current compression loses exact branch-delta information.
```

下一步回到 exact branch-delta representation or richer output-delta statistics。

## Case E：system and reference controller pass but LDO/LSO fail

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

v9.2.57 的一句话策略是：

$$
\boxed{
\text{同时验证 compute feasibility 与 controller feasibility；不要只修 kernel，也不要只调 threshold。}
}
$$

当前最关键的问题不是：

```text
base 是否稳定；
attach 是否污染；
carrier 是否 silent；
CBD0 有没有 AUC；
K7d local fusion 是否能跑；
formula proxy 是否够快；
C0/C6 threshold 是否差一点；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. true branch-delta end-to-end 为什么仍然慢？
2. K7d local speedup 是否真正部署到 critical path？
3. true branch-delta v2 是否能同时保留 AUC / agreement / system gate？
4. exact CBD0 reference 是否存在 precision / coverage / bad-event feasible controller region？
5. support 是否能扩展到 >=6 measured signal strata？
6. system-legal true delta controller 能否 heldout pass？
7. controller 能否 LDO/LSO？
8. official paired replay 能否打过 AdamWParallel / bestLR？
```

v9.2.57 的结果将给出清晰分叉：

```text
if true delta compute + reference feasible controller + LDO/LSO + paired replay pass:
  strict PureKAN functional obtains local causal evidence.

if reference controller feasible but compute fail:
  kernelization is the blocker.

if compute pass but reference controller infeasible:
  risk/support/accept geometry is the blocker.

if both fail:
  exact signal remains diagnostic only; reset sufficient statistics and kernel representation.

if oracle support collapses:
  carrier/support stability is the blocker.
```
