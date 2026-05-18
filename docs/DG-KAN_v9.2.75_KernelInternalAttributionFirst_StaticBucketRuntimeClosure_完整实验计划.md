# DG-KAN v9.2.75 Kernel-Internal Attribution First 与 Materialized Static-Bucket Runtime Closure 完整实验计划

> 本计划基于 v9.2.74 `Materialized Persistent Basis/Delta Kernel 与 Static-Bucket System Closure` 的真实执行结果制定。  
> v9.2.74 的 terminal route 是：
>
> ```text
> route = R13-KernelInternalAttributionIncomplete
> base_candidate = LQ-t2-h256
> success_v9274_strict_purekan_functional = False
> success_v9274_full_functional = False
> success_v9274_external_ready = False
> ```
>
> v9.2.74 的关键事实是：
>
> ```text
> P0:
>   v9.2.73 boundary reproduced = 1
>   payload_binding_contract_pass = 1
>   candidate_tensor_payload_missing_count = 0
>   candidate_branch_logits_missing_count = 0
>   candidate_true_delta_logits_missing_count = 0
>   functional_update_payload_missing_count = 0
>   controller_precision = 0.8380281690140845
>   controller_coverage = 0.03130511463844797
>   controller_bad_event = 0.02464788732394366
>   controller_null_rate = 0.13028169014084506
>   controller_precision_lcb = 0.7907157243773478
>   controller_bad_event_ucb = 0.04999458813129621
>
> P1:
>   kernel_internal_attribution_pass = 0
>   unknown_fraction = 0.0
>   dominant_internal_component = fused_common_basis_delta_pipeline_ms
>   basis_lift_time_ms = ""
>   basis_quadratic_time_ms = ""
>   basis_norm_time_ms = ""
>   kernel_launch_time_ms = ""
>   sync_time_ms = ""
>   failure_reason = kernel_internal_timing_fields_missing_for_basis_lift_quadratic_norm_launch_sync
>
> P2:
>   best_selected_feature_runtime_id = SF2-AcceptBitInsideKernelRuntime
>   selected_feature_runtime_pass = 0
>   materialized_runtime_path = 0
>   selected_feature_materialized_count_after = 2493
>   failure_reason = accept_bit_inside_kernel_runtime_not_materialized
>
> P3:
>   static_bucket_workspace_pass = 0
>   kernel_count_before = 3750
>   kernel_count_after = 3750
>   sync_count_before = 1250
>   sync_count_after = 1250
>   allocation_count_before = 2493
>   allocation_count_after = 2493
>   kernel_count_reduction = 0.0
>   sync_count_reduction = 0.0
>   allocation_count_reduction = 0.0
>   persistent_workspace_used = 0
>   cuda_graph_attempted = 0
>
> P4:
>   basis_delta_runtime_pass = 0
>   basis_delta_bridge_single_pass = 0
>   bridge_score_inside_kernel = 0
>   accept_bit_inside_kernel = 0
>   uses_true_branch_delta = 1
>   uses_source_measured_gap = 0
>   uses_formula_proxy = 0
>   cuda_vs_torch_logits_error_max = 2.6702880859375e-05
>   cuda_vs_torch_delta_error_max = 8.149072527885437e-10
>   basis_delta_agreement = 1.0
>   basis_delta_step_ratio_q90 = 2.713295831053225
>
> P5:
>   audit_separation_pass = 1
>   hash_in_timed_path = 0
>   csv_write_in_timed_path = 0
>   cpu_metadata_in_timed_path = 0
>   norm_item_in_timed_path = 0
>   selected_feature_audit_in_timed_path = 0
>   audit_disagreement_count = 0
>
> P6:
>   system_legal_controller_pass = 0
>   official_eligible = 0
>   reason = kernel_internal_or_selected_feature_or_static_bucket_runtime_failed
>   step_ratio_q90 = 2.713295831053225
>   memory_ratio = 1.0
>   accepted_signal_strata_count = 15
>   accepted_family_count = 65
> ```
>
> v9.2.75 的核心判断是：
>
> $$
> \boxed{
> \text{v9.2.74 没有给出新的 runtime survivor；它证明的是“计划中要实现的 runtime pieces 仍未 materialize”。}
> }
> $$
>
> 因此 v9.2.75 不能继续写 “diagnostic / not implemented / audit-only” rows。  
> 本轮的任务不是继续证明 `C3-T2PlusBackfill` 好，也不是继续证明 payload bound。那些已经稳定。  
> 本轮必须把失败点变成可执行工程目标：
>
> $$
> \boxed{
> \text{先做真实 kernel-internal attribution，再并行 materialize selected-feature runtime、static bucket/workspace、single-pass basis-delta-bridge。}
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

v9.2.75 的新增硬约束是：

```text
audit-only cost subtraction 不允许进入 route survivor；
diagnostic_derived_from_measured_components = 1 的 row 不能 official；
materialized_runtime_path 必须为 1；
kernel_internal_attribution 不能只填 aggregate component；
basis_lift_time_ms / basis_quadratic_time_ms / basis_norm_time_ms / W2_delta_time_ms / probe_logits_time_ms / selected_feature_writeback_time_ms / kernel_launch_time_ms / sync_time_ms / allocation_time_ms 必须有真实值；
static_bucket_workspace_used 必须真实改变 kernel/sync/allocation；
persistent_workspace_used 必须真实减少 allocation；
selected-feature runtime 必须真实改变 selected_feature_materialized_count_after；
single-pass basis/delta/bridge candidate 必须真实改变 basis_delta_bridge_single_pass / bridge_score_inside_kernel / accept_bit_inside_kernel；
official system pass 必须来自真实 timed runtime path。
```

允许按 dataset / family / horizon 诊断 cost 分布，但不允许按 dataset 改 controller 或系统 route：

```text
allowed diagnostics:
  per-dataset candidate_rate
  per-dataset step_ratio
  per-family kernel_count
  per-horizon bucket occupancy
  per-stratum agreement drift

forbidden:
  if dataset == KMNIST: threshold = ...
  if dataset == Fashion: bucket route = ...
  if dataset == MNIST: use different selected-feature mode
  dataset-specific controller threshold
  dataset-specific prefilter rule
  dataset-specific kernel path selected for official route
```

---

# Part I. 对 v9.2.74 的独立判断

## 1. v9.2.74 没有达到目标

v9.2.74 没有 strict PureKAN functional success，也没有 system-legal controller success。P6 仍然：

```text
official_eligible = 0
system_legal_controller_pass = 0
step_ratio_q90 = 2.713295831053225
```

P7-P10 全部 gate-blocked。因此不能声明：

```text
system-legal functional controller
leave-dataset-out / leave-stratum-out pass
official paired replay pass
short-run / full-run pass
external-ready
```

这个失败不是因为 controller 数值坏了。P6 的 reference decision metrics 仍然过线：

$$
Precision=0.838028,
$$

$$
Coverage=0.031305,
$$

$$
BadEvent=0.024648,
$$

$$
NullRate=0.130282,
$$

$$
PrecisionLCB=0.790716,
$$

$$
BadEventUCB=0.049995.
$$

所以 v9.2.74 的失败不是 decision failure，而是 runtime implementation failure。

## 2. v9.2.74 的真实进展

v9.2.74 有进展，但它是 **审计性质的进展**，不是 cost closure。

它严格否定了三个可能被误写成成功的方向：

```text
1. kernel-internal attribution 没有落地；
2. selected-feature elimination 没有 materialized runtime；
3. static bucket / persistent workspace / CUDA graph 没有实现。
```

这很重要。因为如果我们不做 v9.2.74，可能会把 v9.2.73 的 audit-only cost removal 当成实际 runtime improvement。v9.2.74 明确阻止了这种错误。

## 3. v9.2.74 的真实失败

v9.2.74 的真实失败有四个层次。

第一，P1 没有真正做到 kernel-internal attribution。虽然 `unknown_fraction = 0.0`，但这是 aggregate 层的 unknown fraction，不是 kernel 内部闭合。真正需要的字段：

```text
basis_lift_time_ms
basis_quadratic_time_ms
basis_norm_time_ms
W2_delta_time_ms
probe_logits_time_ms
selected_feature_writeback_time_ms
kernel_launch_time_ms
sync_time_ms
allocation_time_ms
```

仍为空。因此 P1 停在 `KernelInternalAttributionIncomplete` 是正确的。

第二，P2 没有把 selected-feature elimination 变成 runtime。`SF2-AcceptBitInsideKernelRuntime` 是 best selected-feature runtime candidate，但：

```text
materialized_runtime_path = 0
selected_feature_materialized_count_after = 2493
```

这说明 selected-feature 仍然全量 materialize，没有变成 bridge-score / accept-bit inside kernel。

第三，P3 static bucket / persistent workspace 完全没有推进。v9.2.74 中：

```text
kernel_count_after = 3750
sync_count_after = 1250
allocation_count_after = 2493
persistent_workspace_used = 0
cuda_graph_attempted = 0
```

和 v9.2.73 一样。也就是说，最可能降低碎片化 overhead 的路径没有实现。

第四，P4 single-pass basis/delta/bridge runtime 仍是 BF5 reference。它保持数值一致：

$$
LogitsError=2.67\times10^{-5},
$$

$$
DeltaError=8.15\times10^{-10},
$$

$$
Agreement=1.0,
$$

但：

```text
basis_delta_bridge_single_pass = 0
bridge_score_inside_kernel = 0
accept_bit_inside_kernel = 0
step_ratio_q90 = 2.713296
```

这不能进入 official system controller。

## 4. 问题的本质

当前 blocker 应该写成：

$$
\boxed{
\text{runtime execution graph is still fragmented and not instrumented deeply enough.}
}
$$

更具体地说，当前不是：

```text
reference frontier 不可部署；
safe-useful target 错；
PF5 prefilter 错；
payload missing；
true-delta signal 消失；
hash/csv/cpu metadata 污染 timed path。
```

当前是：

```text
1. fused_common_basis_delta_pipeline 仍是 aggregate black box；
2. selected-feature 仍然全量 materialize；
3. kernel/sync/allocation 数量没有下降；
4. avg candidates per kernel 很低的问题没有被实装修复；
5. static bucket / persistent workspace / CUDA graph 没有 materialize；
6. single-pass basis-delta-bridge 还没有成为真实 runtime path。
```

因此，v9.2.75 必须把 “未实现” 变成 “实现并测量”。如果 v9.2.75 仍然只生成 not_implemented 或 audit-only rows，那就是路线执行失败，而不是科学假设失败。

---

# Part II. v9.2.75 总体目标

v9.2.75 的总体目标是：

$$
\boxed{
\text{在保持 payload/decision 全部过关的前提下，完成真实 kernel-internal attribution，并至少 materialize 一条 runtime cost-reduction path。}
}
$$

本轮不要求一次性必过 official `1.50`，但必须完成两个硬推进：

```text
1. P1 kernel-internal attribution 必须从 aggregate block 进入真实 subphase；
2. P2/P3/P4 至少一个 materialized runtime candidate 必须产生真实 step-ratio reduction。
```

v9.2.75 的强目标是：

$$
StepRatio_{q90}\leq1.50.
$$

v9.2.75 的最低有效推进目标是：

$$
StepRatio_{q90}\leq2.00,
$$

且必须满足：

```text
materialized_runtime_path = 1
audit_only_cost_removal = 0
diagnostic_derived_from_measured_components = 0
agreement_reference_accept >= 0.90
payload binding remains pass
```

如果连 `<=2.00` 都没有，但 P1 真正完成 kernel-internal attribution，也仍然算有价值诊断推进。  
如果 P1 仍然只有 aggregate attribution，本轮必须判定为执行失败，不能继续写 “system still too expensive” 泛泛结论。

---

# Part III. 核心假设

## H1：aggregate black box 可以被真实拆解

H1 认为 `fused_common_basis_delta_pipeline_ms` 内部至少有一个可操作的主成本，例如：

```text
basis lift
quadratic / T2 recurrence
basis normalization
candidate gather/scatter
W2 delta
probe logits
selected-feature writeback
kernel launch
sync
allocation
```

H1 成立标准：

```text
kernel_internal_attribution_pass = 1
unknown_fraction <= 0.05
basis_lift_time_ms non-empty
basis_quadratic_time_ms non-empty
basis_norm_time_ms non-empty
W2_delta_time_ms non-empty
probe_logits_time_ms non-empty
selected_feature_writeback_time_ms non-empty
kernel_launch_time_ms non-empty
sync_time_ms non-empty
allocation_time_ms non-empty
dominant_subcomponent identified
```

H1 失败标准：

仍然只能写：

```text
dominant_internal_component = fused_common_basis_delta_pipeline_ms
```

且细分字段为空。若 H1 失败，不允许推进 P6 official controller。

## H2：selected-feature materialization 是可被 runtime 消除的真实成本，不只是 accounting 项

H2 成立标准：

```text
materialized_runtime_path = 1
audit_only = 0
bridge_score_inside_kernel = 1 or accept_bit_inside_kernel = 1
selected_feature_materialized_count_after <= 0.25 * 2493
agreement_reference_accept >= 0.95
audit_agreement >= 0.99
```

并且 step 有真实下降：

$$
StepRatio_{q90}^{after}
\leq
StepRatio_{q90}^{BF5}-0.25.
$$

H2 失败标准：

selected-feature removal 只能通过 audit-only subtraction 实现，或者 runtime removal 使 agreement 低于 `0.90`。

## H3：static bucket / persistent workspace 可以减少碎片化 kernel overhead

H3 成立标准：

```text
persistent_workspace_used = 1
allocation_count_after <= 0.10 * 2493
kernel_count_after <= 0.50 * 3750
sync_count_after <= 0.50 * 1250
avg_candidates_per_kernel_after >= 8
agreement_reference_accept >= 0.90
```

如果 CUDA graph capture 成功，则记录：

```text
cuda_graph_attempted = 1
cuda_graph_capture_pass = 1
graph_replay_used = 1
```

如果 CUDA graph capture 失败，也必须记录明确原因：

```text
dynamic_shape
pointer_mutation
workspace_alias
graph_unsafe_op
stream_semantics
unsupported_triton_kernel
```

H3 失败标准：

kernel/sync/allocation 没有变化，或 persistent workspace 只是 nominal flag。

## H4：single-pass basis-delta-bridge 可以保持 exactness，同时减少 writeback / launch / sync

H4 成立标准：

```text
basis_delta_bridge_single_pass = 1
uses_true_branch_delta = 1
uses_source_measured_gap = 0
uses_formula_proxy = 0
bridge_score_inside_kernel = 1
or accept_bit_inside_kernel = 1
cuda_vs_torch_check_count >= 24
cuda_vs_torch_logits_error_max <= 5e-5
cuda_vs_torch_delta_error_max <= 1e-8
agreement_reference_accept >= 0.90
```

Intermediate success：

$$
StepRatio_{q90}\leq2.00.
$$

Full success：

$$
StepRatio_{q90}\leq1.50.
$$

H4 失败标准：

低成本只能通过 proxy/source gap/selected-logit formula 实现，或 full exactness/audit agreement 失败。

## H5：如果真实 runtime fusion 仍不进 envelope，则当前 FC-PureKAN functional path 的 blocker 是 kernel-native lower bound，而不是 controller

H5 成立标准：

P1 attribution 完整，P2/P3/P4 均 materialized，agreement 保持，但：

$$
StepRatio_{q90}>1.50.
$$

此时下一步不应再调 decision；应该进入：

```text
arithmetic simplification
basis recurrence redesign
candidate count reduction without dataset tuning
patch-local KANConv-like primitive feasibility
or alternative observable primitive with cheaper sufficient statistics
```

H5 失败标准：

还没有 materialized path，就声称 lower bound 已到。这是不允许的。

---

# Part IV. Runtime design

## 1. 当前 baseline

当前 baseline 是 BF5-style payload-bound runtime：

```text
candidate_count = 2493
event_count = 24192
candidate_rate = 0.10305059523809523
kernel_count = 3750
sync_count = 1250
allocation_count = 2493
step_ratio_q90 = 2.713295831053225
```

v9.2.73 audit-only selected-feature subtraction 给出 diagnostic lower value：

```text
step_ratio_q90 = 2.3464118796089455
```

但这不是 official measured runtime。

## 2. v9.2.75 目标 runtime

v9.2.75 的目标 runtime 形式为：

```text
PF5 selector
→ static bucket candidate table
→ persistent candidate/basis/delta workspace
→ materialized basis-delta-bridge kernel
→ bridge score / accept bit inside kernel
→ accepted update payload
→ no-hash timed path + post-step audit
```

用公式写：

$$
Accept_{\text{sys}}(e)
=
PF5(e)
\land
K_{\text{basis-delta-bridge}}(e)
\land
FrozenC3(e).
$$

其中：

$$
K_{\text{basis-delta-bridge}}(e)
$$

必须使用 true branch-delta，不得使用 source-measured gap 或 formula proxy。

## 3. Cost model

v9.2.75 必须记录：

$$
T_{\text{sys}}
=
T_{\text{PF5}}
+
T_{\text{bucket}}
+
T_{\text{workspace}}
+
T_{\text{basis}}
+
T_{\text{delta}}
+
T_{\text{bridge}}
+
T_{\text{accept}}
+
T_{\text{update}}
+
T_{\text{launch}}
+
T_{\text{sync}}
+
T_{\text{allocation}}
+
T_{\text{post-audit-outside-timed}}.
$$

Official timed path 只能包含真实 controller 执行，不包含 CSV/hash/posthoc audit。

---

# Part V. Candidate designs

## 1. Kernel attribution candidates

### IA0：v9.2.74 aggregate reference

复现现有失败：

```text
kernel_internal_attribution_pass = 0
dominant_internal_component = fused_common_basis_delta_pipeline_ms
basis_lift_time_ms = ""
...
```

### IA1：CUDA-event nested instrumentation

用 CUDA event 分段记录：

```text
candidate_gather
basis_lift
basis_quadratic
basis_norm
W2_delta
probe_logits
selected_feature_writeback
bridge_score
accept_bit
workspace_writeback
```

每段必须落盘。

### IA2：NVTX + torch profiler targeted attribution

只在小样本 diagnostic run 中启用 NVTX/profiler，不进入 official step pass。目标是定位 kernel names 与 subphase。

### IA3：Triton/CUDA manual timer wrapper

若 fused kernel 内部无法直接分段，则把 fused kernel 拆成 minimal sub-kernels 做 timing：

```text
kernel_basis_only
kernel_delta_only
kernel_bridge_only
kernel_accept_only
```

这不是最终 route，只用于 attribution。

### IA4：Launch/sync/allocation hard audit

记录：

```text
kernel_count
launch_count
sync_count
cudaEventSynchronize_count
torch_empty_count
cudaMalloc_count
contiguous_count
clone_count
cpu_transfer_count
```

## 2. Materialized selected-feature candidates

### SF0：BF5 selected-feature reference

全量 materialize selected features。Baseline。

### SF1：BridgeScoreInsideKernelRuntimeV2

kernel 输出：

```text
bridge_score
bad_ucb
null_ucb
support_lcb
```

不输出 full selected-feature tensor。

### SF2：AcceptBitInsideKernelRuntimeV2

kernel 输出：

```text
accept_bit
reject_reason_code
borderline_flag
```

只有 borderline rows 进入 full selected-feature pass。

### SF3：TwoPassBorderlineRuntimeV2

Pass 1：

```text
bridge score / accept bit
```

Pass 2：

```text
full selected-feature only for borderline candidates
```

### SF4：AuditSubsetSelectedFeatureRuntimeV2

official timed path 不输出 selected-feature，post-step audit subset 输出并比较。

## 3. Static bucket / workspace candidates

### BW0：NoWorkspaceReference

现状 reference。

### BW1：FixedBucketCandidateTensorV2

bucket sizes：

```text
1,2,4,8,16,32
```

padding with mask，不改变 accept decision。

### BW2：FamilyBranchHorizonBucketV2

按：

```text
family_id
branch_id
horizon
bucket_size
```

分组，减少 dynamic shape。

### BW3：PersistentWorkspaceV2

预分配：

```text
candidate_workspace
basis_workspace
delta_workspace
bridge_score_workspace
accept_bit_workspace
update_payload_workspace
```

timed path allocation 应接近 0。

### BW4：BatchMajorCandidateFusionV2

把 candidates 排成 batch-major layout：

```text
candidate_dim major
basis_dim minor
branch/family metadata table
```

目标：

```text
avg_candidates_per_kernel_after >= 8
```

### BW5：CudaGraphStaticBucketV2

尝试 CUDA graph per bucket。若失败，必须记录具体 failure reason。

## 4. Basis / delta / bridge runtime candidates

### BD0：BF5 reference runtime

现状 reference。

### BD1：SinglePassBasisDeltaBridgeV2

一个 runtime path 输出：

```text
basis
delta logits
bridge score
accept bit
```

### BD2：T2RecurrenceBasisDeltaV2

利用 T2/quadratic 结构：

$$
\phi(x)=a_0+a_1x+a_2x^2.
$$

用 fused FMA/recurrence，而不是 full basis tensor materialization。

### BD3：W2FocusedTrueDeltaV2

因为 update payload 是 compressed W2-only，尝试只计算 official controller required W2-relevant delta，但必须保持 reference agreement。

### BD4：CommonBasisCacheV2

在 GPU workspace 中缓存 repeated family/bucket basis，不做 CPU summary。

### BD5：BasisDeltaBridgeWithInsideAcceptV2

single-pass + accept-bit inside kernel。

## 5. System candidates

### SYS0：v9.2.74 reference

Expected fail.

### SYS1：IA1 + SF1

先实现 materialized bridge-score inside kernel，不改 bucket。

### SYS2：IA1 + SF2

先实现 accept-bit inside kernel，不改 bucket。

### SYS3：BW1 + BW3 + BF5 reference

先实现 static bucket + persistent workspace，不改 selected feature。

### SYS4：BD1 + SF1 + BW3

single-pass basis/delta/bridge + persistent workspace。

### SYS5：BD2 + SF2 + BW4

T2 recurrence + accept-bit + batch-major candidate fusion。

### SYS6：BD5 + BW4 + SF3

single-pass inside accept + batch-major + two-pass borderline.

### SYS7：HybridBestRuntimeV9275

Best materialized survivor from SYS1-SYS6. Only SYS7 can be promoted to P6 if all audit gates pass.

---

# Part VI. 实验阶段

## P0：v9.2.74 boundary reproduction

### 目标

确认最新失败边界稳定，避免在不稳定 boundary 上继续实现。

### 必须记录

```text
route
source_route_v9274
payload_binding_contract_pass
candidate_tensor_payload_missing_count
candidate_branch_logits_missing_count
candidate_true_delta_logits_missing_count
functional_update_payload_missing_count
controller_precision
controller_coverage
controller_bad_event
controller_null_rate
controller_precision_lcb
controller_bad_event_ucb
agreement_reference_accept
controller_step_ratio_q90
kernel_internal_attribution_pass
selected_feature_runtime_pass
static_bucket_workspace_pass
basis_delta_runtime_pass
system_legal_controller_pass
official_eligible
fake_proxy_count
cpu_offload_used
```

### 判断标准

P0 pass：

```text
route = R13-KernelInternalAttributionIncomplete
payload binding pass = 1
all missing payload counts = 0
decision metrics pass
system controller pass = 0
official_eligible = 0
fake/proxy/offload = 0
```

### 可视化

```text
p0_v9274_boundary_ladder.svg
p0_decision_payload_runtime_split.svg
p0_v9273_to_v9274_regression_or_audit_map.svg
```

---

## P1：real kernel-internal attribution

### 目标

把 `fused_common_basis_delta_pipeline_ms` 从 aggregate block 拆成真实 subphase。P1 是本轮最重要的硬门；如果 P1 不过，后续所有优化都容易继续修错地方。

### 必须记录

```text
internal_cost_candidate_id
instrumentation_type
materialized_runtime_path
basis_lift_time_ms
basis_quadratic_time_ms
basis_norm_time_ms
candidate_gather_time_ms
W2_delta_time_ms
probe_logits_time_ms
selected_feature_writeback_time_ms
bridge_score_time_ms
accept_bit_time_ms
workspace_writeback_time_ms
kernel_launch_time_ms
sync_time_ms
allocation_time_ms
cuda_event_total_ms
wallclock_total_ms
unknown_fraction
kernel_count
sync_count
allocation_count
avg_candidates_per_kernel
bytes_read
bytes_written
effective_bandwidth_proxy
occupancy_proxy
```

### 判断标准

P1 pass：

```text
kernel_internal_attribution_pass = 1
unknown_fraction <= 0.05
basis_lift_time_ms is not empty
basis_quadratic_time_ms is not empty
basis_norm_time_ms is not empty
W2_delta_time_ms is not empty
probe_logits_time_ms is not empty
selected_feature_writeback_time_ms is not empty
kernel_launch_time_ms is not empty
sync_time_ms is not empty
allocation_time_ms is not empty
dominant_subcomponent identified
at least one removable_or_fusible_component_ratio >= 0.20
```

### 可视化

```text
p1_kernel_internal_waterfall.svg
p1_launch_sync_allocation_breakdown.svg
p1_candidates_per_kernel_histogram.svg
p1_memory_bandwidth_proxy.svg
```

---

## P2：materialized selected-feature runtime

### 目标

把 v9.2.73/v9.2.74 的 audit-only selected-feature removal 变成真实 runtime。

### 必须记录

```text
selected_feature_runtime_id
mode
materialized_runtime_path
audit_only
diagnostic_derived_from_measured_components
selected_feature_materialized_count_before
selected_feature_materialized_count_after
borderline_count
audit_subset_count
bridge_score_inside_kernel
accept_bit_inside_kernel
selected_feature_time_ms_before
selected_feature_time_ms_after
agreement_reference_accept
audit_agreement
precision
coverage
bad_event
null_rate
precision_lcb
bad_event_ucb
step_ratio_q90
memory_ratio
projection_used
source_gap_used
formula_proxy_used
```

### 判断标准

P2 pass：

```text
materialized_runtime_path = 1
audit_only = 0
diagnostic_derived_from_measured_components = 0
selected_feature_materialized_count_after <= 0.25 * selected_feature_materialized_count_before
agreement_reference_accept >= 0.90
audit_agreement >= 0.99
projection_used = 0
source_gap_used = 0
formula_proxy_used = 0
```

Diagnostic system improvement：

$$
StepRatio_{q90}\leq2.00.
$$

### 可视化

```text
p2_selected_feature_runtime_cost.svg
p2_accept_bit_agreement_confusion.svg
p2_borderline_candidate_distribution.svg
p2_selected_feature_materialization_count.svg
```

---

## P3：static bucket and persistent workspace materialization

### 目标

真正降低 kernel/sync/allocation，而不是保持 `3750/1250/2493`。

### 必须记录

```text
bucket_workspace_id
bucket_strategy
bucket_sizes
candidate_count
fused_step_count_before
fused_step_count_after
kernel_count_before
kernel_count_after
sync_count_before
sync_count_after
allocation_count_before
allocation_count_after
persistent_workspace_used
workspace_memory_MB
cuda_graph_attempted
cuda_graph_capture_pass
cuda_graph_failure_reason
avg_candidates_per_kernel_before
avg_candidates_per_kernel_after
agreement_reference_accept
cuda_vs_torch_logits_error_max
cuda_vs_torch_delta_error_max
step_ratio_q90
memory_ratio
```

### 判断标准

P3 pass：

```text
persistent_workspace_used = 1
allocation_count_after <= 0.10 * allocation_count_before
kernel_count_after <= 0.50 * kernel_count_before
sync_count_after <= 0.50 * sync_count_before
avg_candidates_per_kernel_after >= 8
agreement_reference_accept >= 0.90
```

Intermediate system improvement：

$$
StepRatio_{q90}\leq2.00.
$$

### 可视化

```text
p3_kernel_sync_allocation_reduction.svg
p3_bucket_occupancy_histogram.svg
p3_cuda_graph_capture_status.svg
p3_workspace_memory_tradeoff.svg
```

---

## P4：single-pass basis/delta/bridge runtime

### 目标

将 basis prep、W2 delta、probe logits、bridge score、accept bit 合并为真实 measured runtime。

### 必须记录

```text
basis_delta_runtime_id
selected_feature_runtime_id
bucket_workspace_id
uses_true_branch_delta
uses_source_measured_gap
uses_formula_proxy
basis_delta_bridge_single_pass
bridge_score_inside_kernel
accept_bit_inside_kernel
candidate_count
kernel_count
sync_count
allocation_count
cuda_vs_torch_check_count
cuda_vs_torch_logits_error_max
cuda_vs_torch_delta_error_max
agreement_reference_accept
AUC_bridge_accept
precision
coverage
bad_event
null_rate
precision_lcb
bad_event_ucb
basis_delta_total_ms
selected_feature_total_ms
kernel_launch_time_ms
sync_time_ms
step_ratio_q90
memory_ratio
```

### 判断标准

P4 pass：

```text
uses_true_branch_delta = 1
uses_source_measured_gap = 0
uses_formula_proxy = 0
basis_delta_bridge_single_pass = 1
cuda_vs_torch_check_count >= 24
cuda_vs_torch_logits_error_max <= 5e-5
cuda_vs_torch_delta_error_max <= 1e-8
agreement_reference_accept >= 0.90
```

Intermediate system improvement：

$$
StepRatio_{q90}\leq2.00.
$$

Full system candidate:

$$
StepRatio_{q90}\leq1.50.
$$

### 可视化

```text
p4_single_pass_cost_quality_pareto.svg
p4_error_vs_step_ratio.svg
p4_reference_agreement_confusion.svg
p4_kernel_count_before_after.svg
```

---

## P5：integrated materialized runtime candidates

### 目标

组合 P2/P3/P4 survivor，形成真实 measured SYS candidates，不再输出 not_implemented / audit-only rows 作为 route best。

### 必须记录

```text
system_candidate_id
selected_feature_runtime_id
bucket_workspace_id
basis_delta_runtime_id
materialized_runtime_path
audit_only_cost_removal
diagnostic_derived_from_measured_components
candidate_count
candidate_rate
kernel_count
sync_count
allocation_count
avg_candidates_per_kernel
agreement_reference_accept
precision
coverage
bad_event
null_rate
precision_lcb
bad_event_ucb
step_ratio_q90
memory_ratio
projection_used
source_measured_gap_used
formula_proxy_used
dataset_name_used
```

### 判断标准

P5 diagnostic pass：

```text
materialized_runtime_path = 1
audit_only_cost_removal = 0
diagnostic_derived_from_measured_components = 0
agreement_reference_accept >= 0.90
projection_used = 0
source_measured_gap_used = 0
formula_proxy_used = 0
```

P5 improvement pass：

$$
StepRatio_{q90}\leq2.00.
$$

P5 system survivor:

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p5_materialized_runtime_pareto.svg
p5_step_ratio_reduction_ladder.svg
p5_kernel_sync_allocation_vs_step.svg
```

---

## P6：system-legal exact-signal controller v7

### 目标

只有 P5 出现真实 system survivor，P6 才允许 official controller pass。

### 必须记录

```text
controller_id
system_candidate_id
selected_feature_runtime_id
bucket_workspace_id
basis_delta_runtime_id
prefilter_id
thresholds
calibration_split_id
heldout_split_id
event_count
candidate_count
accepted_count
candidate_rate
precision_cal
coverage_cal
bad_event_cal
null_rate_cal
precision_heldout
coverage_heldout
bad_event_heldout
null_rate_heldout
precision_lcb
bad_event_ucb
accepted_signal_strata_count
accepted_family_count
max_family_share
max_stratum_share
AUC_safe_good
AUC_bridge_accept
agreement_reference_accept
step_ratio_q90
memory_ratio
kernel_count
sync_count
allocation_count
avg_candidates_per_kernel
candidate_tensor_payload_missing_count
candidate_branch_logits_missing_count
candidate_true_delta_logits_missing_count
functional_update_payload_missing_count
materialized_system_path
audit_only_cost_removal
diagnostic_derived_from_measured_components
projection_used
full_trace_projection_used
full_online_row_binding
full_online_payload_binding
full_online_update_payload_binding
source_measured_gap_used
formula_proxy_used
cpu_offload_used
dataset_name_used
posthoc_used_at_commit
validation_used
test_used
official_eligible
system_legal_controller_pass
```

### 判断标准

P6 pass：

```text
official_eligible = 1
system_legal_controller_pass = 1
materialized_system_path = 1
audit_only_cost_removal = 0
diagnostic_derived_from_measured_components = 0
full_online_row_binding = 1
full_online_payload_binding = 1
full_online_update_payload_binding = 1
projection_used = 0
full_trace_projection_used = 0
source_measured_gap_used = 0
formula_proxy_used = 0
cpu_offload_used = 0
dataset_name_used = 0
```

Decision gate：

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
NullRate_{\text{heldout}}\leq0.15,
$$

$$
Precision_{\text{LCB}}\geq0.75,
$$

$$
BadEventRate_{\text{UCB}}\leq0.05.
$$

System gate：

$$
Agreement_{\text{reference}}\geq0.90,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

Support balance：

```text
accepted_signal_strata_count >= 5
accepted_family_count >= 32
max_family_share <= 0.50
max_stratum_share <= 0.60
```

### 可视化

```text
p6_system_controller_cost_quality_frontier.svg
p6_reference_vs_system_accept_overlap.svg
p6_step_ratio_progress_v9273_to_v9275.svg
p6_kernel_sync_allocation_progress.svg
p6_family_strata_balance.svg
```

---

## P7：leave-dataset-out / leave-stratum-out

### 目标

只有 P6 pass 后打开。证明 system controller 不是 pooled calibration artifact，也不是 dataset-specific route。

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
system_candidate_id
candidate_rate
reference_accept_recall
precision
coverage
bad_event_rate
null_rate
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

At least $70\%$ held-out strata task-safe and:

$$
CEp99_{\text{heldout,Real}}\leq CEp99_{\text{AdamW}}+\epsilon,
$$

$$
BeatRate_{\text{heldout-stratum,Real vs AdamWParallel}}\geq0.50.
$$

### 可视化

```text
p7_leave_dataset_out_matrix.svg
p7_leave_stratum_out_matrix.svg
p7_dataset_tuning_audit.svg
p7_leaveout_failure_modes.svg
```

---

## P8：official paired replay

### 目标

验证 RealFunctional 是否在 strong controls 下有局部因果优势。

### 设置

```text
base = R2 repaired base checkpoint
controller_id = best P7 survivor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches =
  RealFunctional
  AdamWOnly
  AdamWParallel
  bestLR
  NoOp
  Random
  ShuffledTrueBranchDelta
  ShuffledPF5Prefilter
  ShuffledCandidatePayload
  ShuffledBasisDeltaKernel
  ShuffledSelectedFeature
  ShuffledStaticBucket
  ShuffledPersistentWorkspace
  ShuffledFunctionalUpdatePayload
  ShuffledFrozenBridge
  ShuffledSupportStat
  ShuffledControlGain
  ShuffledCandidateGate
  ShuffledBranchRatio
  ShuffledSignalChannel
  FunctionalChannelShuffled
  TailMaskShuffled
  RoleScoreShuffled
  DatasetRouteShuffled
  EventRouteShuffled
  InvertedRoleMask
```

### 必须记录

```text
controller_id
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
candidate_rate
coverage
bad_event_rate
null_rate
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

Shuffle controls must fail：

```text
ShuffledTrueBranchDelta = fail
ShuffledPF5Prefilter = fail
ShuffledCandidatePayload = fail
ShuffledBasisDeltaKernel = fail
ShuffledSelectedFeature = fail
ShuffledStaticBucket = fail
ShuffledPersistentWorkspace = fail
ShuffledFunctionalUpdatePayload = fail
ShuffledFrozenBridge = fail
ShuffledSupportStat = fail
ShuffledControlGain = fail
ShuffledCandidateGate = fail
ShuffledBranchRatio = fail
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
p8_official_paired_replay_pareto.svg
p8_macro_beat_rate.svg
p8_signal_stratum_win_matrix.svg
p8_shuffle_control_matrix.svg
p8_system_gate_distribution.svg
```

---

## P9：short-run scout

### 目标

如果 P8 paired replay pass，验证 local causal advantage 能否进入连续训练。

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
candidate_rate
coverage
bad_event_rate
null_rate
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

## P10：full run / robustness / strong baseline

### 目标

只有 P9 pass 后打开。验证 functional advantage 不是 local replay artifact。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
controls =
  AdamWOnly
  AdamWParallel
  bestLR
  StrongLRGrid
  QuadraticFeatureMLP
  NoOp
  Random
  ShuffledTrueBranchDelta
  ShuffledPF5
  ShuffledBasisDeltaKernel
  ShuffledSelectedFeature
  ShuffledStaticBucket
  ShuffledPersistentWorkspace
  ShuffledFunctionalUpdatePayload
  ShuffledFrozenBridge
```

### 必须记录

```text
dataset
seed
candidate
final_acc
best_val_acc
test_acc
train_loss
val_loss
test_loss
ECE
NLL
CEp99
margin_p10
curvature
local_lipschitz
functional_event_count
candidate_rate
coverage
bad_event_rate
null_rate
step_ratio_q90
memory_ratio
strong_baseline_beaten
robustness_pass
base_checkpoint_hash
```

### 判断标准

Full functional pass：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005
$$

and at least one:

$$
Acc_{\text{functional}}>Acc_{\text{AdamWParallel}},
$$

$$
ECE_{\text{functional}}<ECE_{\text{AdamW}},
$$

$$
NLL_{\text{functional}}<NLL_{\text{AdamW}},
$$

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

Strong baseline pass：

```text
Functional not explained by QuadraticFeatureMLP
Functional not explained by LR grid
Functional not explained by shuffled controller/kernel/payload
```

---

# Part VII. Required artifacts

```text
run_manifest.json
contract_audit_v9275.csv
p0_v9274_boundary_reproduction.csv
p1_real_kernel_internal_attribution.csv
p2_materialized_selected_feature_runtime.csv
p3_static_bucket_persistent_workspace_materialization.csv
p4_single_pass_basis_delta_bridge_runtime.csv
p5_integrated_materialized_runtime_candidates.csv
p6_system_legal_exact_signal_controller_v7.csv
p7_leave_dataset_and_stratum_out.csv
p8_official_paired_replay.csv
p9_short_run_functional_validation.csv
p10_full_run_robustness_strong_baseline.csv

kernel_internal_cost_trace_v9275.csv
selected_feature_runtime_trace_v9275.csv
static_bucket_workspace_trace_v9275.csv
basis_delta_bridge_runtime_trace_v9275.csv
integrated_runtime_trace_v9275.csv
system_controller_trace_v9275.csv
leaveout_trace_v9275.csv
paired_replay_branch_trace_v9275.csv
short_run_trace_v9275.csv

route_decision.json
aggregate_decision.json
failure_table.csv
artifact_hashes.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9274_boundary_unstable
F3_dataset_tuning_detected
F4_payload_binding_regression
F5_decision_metric_regression
F6_kernel_internal_attribution_still_aggregate
F7_kernel_internal_subphase_missing
F8_no_dominant_fusible_component
F9_selected_feature_runtime_not_materialized
F10_selected_feature_elimination_agreement_fail
F11_selected_feature_runtime_no_cost_reduction
F12_static_bucket_not_materialized
F13_persistent_workspace_not_materialized
F14_cuda_graph_capture_fail_unexplained
F15_kernel_sync_allocation_unfixed
F16_avg_candidates_per_kernel_too_low
F17_basis_delta_bridge_single_pass_not_materialized
F18_basis_delta_runtime_error_fail
F19_basis_delta_runtime_agreement_fail
F20_basis_delta_runtime_system_fail
F21_audit_only_cost_removal_used
F22_source_gap_or_formula_proxy_used
F23_system_controller_precision_fail
F24_system_controller_coverage_fail
F25_system_controller_bad_event_fail
F26_system_controller_null_rate_fail
F27_system_controller_lcb_ucb_fail
F28_system_controller_step_ratio_fail
F29_system_controller_memory_fail
F30_system_controller_projection_used
F31_leave_dataset_out_fail
F32_leave_stratum_out_fail
F33_paired_replay_control_equivalent
F34_shuffle_control_pass
F35_functional_lr_equivalent
F36_short_run_task_drop
F37_full_run_no_macro_hard_stratum_gain
F38_strong_baseline_explains_gain
F39_robustness_fail
F40_external_not_ready
F41_fake_or_proxy_violation
F42_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-BoundaryReproduced:
  v9.2.74 boundary reproduced.

R2-KernelInternalAttributionPass:
  fused_common_basis_delta pipeline is decomposed into real subphases.

R3-SelectedFeatureRuntimePass:
  selected-feature elimination is materialized runtime, not audit-only.

R4-StaticBucketWorkspacePass:
  kernel/sync/allocation count is reduced through real static bucket / persistent workspace.

R5-SinglePassBasisDeltaBridgePass:
  basis/delta/bridge runtime is true-delta exact and materialized.

R6-MaterializedRuntimeDiagnosticPass:
  at least one runtime candidate reaches step_ratio_q90 <= 2.00 with agreement.

R7-SystemLegalExactSignalControllerPass:
  system controller passes decision + compute gates.

R8-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R9-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R10-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R11-ShortRunFunctionalPass:
  short-run task-safe mechanism gain.

R12-FullFunctionalPass:
  full run task / geometry / system / control gates pass.

R13-PayloadBindingRegression:
  payload binding no longer reproduces.

R14-KernelInternalAttributionStillIncomplete:
  internal cost still only aggregate; no actionable fusion target.

R15-SelectedFeatureRuntimeStillDiagnostic:
  selected-feature removal remains audit-only.

R16-StaticBucketWorkspaceStillUnimplemented:
  kernel/sync/allocation remain unchanged.

R17-BasisDeltaBridgeRuntimeStillReference:
  true low-level fused path is still BF5 reference, not single-pass.

R18-SystemStillTooExpensive:
  materialized runtime improves but step_ratio_q90 remains >1.50.

R19-ComputePassButLeaveoutFail:
  system controller overfits pooled calibration.

R20-ComputePassButPairedReplayFail:
  controller is system-legal but not causally superior to controls.

R21-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` 必须记录：

```text
route
v9274_boundary_pass
dataset_tuning_detected
reference_controller_id
controller_precision
controller_coverage
controller_bad_event
controller_null_rate
controller_precision_lcb
controller_bad_event_ucb
payload_binding_contract_pass
candidate_tensor_payload_missing_count
candidate_branch_logits_missing_count
candidate_true_delta_logits_missing_count
functional_update_payload_missing_count

kernel_internal_attribution_pass
unknown_fraction
dominant_internal_component
basis_lift_time_ms
basis_quadratic_time_ms
basis_norm_time_ms
candidate_gather_time_ms
W2_delta_time_ms
probe_logits_time_ms
selected_feature_writeback_time_ms
bridge_score_time_ms
accept_bit_time_ms
kernel_launch_time_ms
sync_time_ms
allocation_time_ms

best_selected_feature_runtime_id
selected_feature_runtime_pass
materialized_runtime_path
selected_feature_materialized_count_after
audit_only_cost_removal

best_bucket_workspace_id
static_bucket_workspace_pass
persistent_workspace_used
kernel_count_reduction
sync_count_reduction
allocation_count_reduction
avg_candidates_per_kernel_after
cuda_graph_capture_pass
cuda_graph_failure_reason

best_basis_delta_runtime_id
basis_delta_runtime_pass
basis_delta_bridge_single_pass
bridge_score_inside_kernel
accept_bit_inside_kernel
cuda_vs_torch_logits_error_max
cuda_vs_torch_delta_error_max
basis_delta_agreement
basis_delta_step_ratio_q90

best_integrated_runtime_id
integrated_runtime_pass
integrated_step_ratio_q90
integrated_memory_ratio

best_system_controller_id
system_legal_controller_pass
official_eligible
controller_reference_agreement
controller_step_ratio_q90
controller_memory_ratio
accepted_signal_strata_count
accepted_family_count
max_family_share
max_stratum_share

leave_dataset_out_pass
leave_stratum_out_pass
paired_replay_pass
short_run_pass
full_run_pass
robustness_pass
strong_baseline_pass
external_ready
primary_blocker
next_required_implementation
success_v9275_strict_purekan_functional
success_v9275_full_functional
success_v9275_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 real kernel-internal attribution
  P2 materialized selected-feature runtime
  P3 static bucket / persistent workspace materialization
  P4 single-pass basis/delta/bridge runtime
  P5 integrated materialized runtime candidates

Batch 2:
  P6 system-legal controller
  P7 leave-dataset-out / leave-stratum-out
  P8 paired replay scout

Batch 3:
  official P8 paired replay
  P9 short-run if P8 passes

Batch 4:
  P10 full run / robustness / strong baseline only if P9 passes
```

Gate rule：

```text
P2/P3/P4 can begin in parallel after P0, but P5/P6 cannot pass unless P1 attribution pass.
P6 cannot pass unless:
  P0 pass
  P1 real kernel-internal attribution pass
  payload binding remains pass
  at least one P5 materialized runtime candidate reaches step_ratio_q90 <= 1.50
  audit_only_cost_removal = 0
  diagnostic_derived_from_measured_components = 0
  projection_used = 0
  source_measured_gap_used = 0
  formula_proxy_used = 0
  full_online_payload_binding = 1
  full_online_update_payload_binding = 1

P7/P8 diagnostic rows may be measured before all gates finish,
but official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  reference controller reproduced
  frozen controller pass
  PF5 runtime selector pass
  payload binding pass
  true branch-delta legality pass
  true branch-delta predictivity pass
  true branch-delta agreement pass
  true branch-delta system pass
  functional update payload pass
  system-legal controller pass
```

---

# Part X. 停止条件

## Minimum diagnostic success

```text
v9.2.74 boundary reproduced
kernel-internal attribution measured
selected-feature runtime measured
static bucket/workspace measured
single-pass basis/delta/bridge measured
integrated runtime measured
system controller measured
no fake/proxy/offload/loss/teacher violation
```

## Attribution success

```text
Minimum diagnostic success
+
kernel_internal_attribution_pass = 1
+
unknown_fraction <= 0.05
+
all required subphase timing fields non-empty
+
dominant actionable component identified
```

## Runtime materialization success

```text
Attribution success
+
at least one of:
  selected-feature runtime pass
  static bucket workspace pass
  single-pass basis/delta/bridge pass
+
materialized_runtime_path = 1
+
audit_only_cost_removal = 0
+
diagnostic_derived_from_measured_components = 0
```

## System success

```text
Runtime materialization success
+
payload binding remains pass
+
agreement_reference_accept >= 0.90
+
step_ratio_q90 <= 1.50
+
memory_ratio <= 1.05
+
decision gates pass
+
official_eligible = 1
```

## Local functional success

```text
System success
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
+
robustness / strong baseline pass
```

## Failure stop

```text
1. v9.2.74 boundary cannot be reproduced；
2. payload binding regresses；
3. decision metrics regress；
4. kernel-internal attribution still only aggregate；
5. selected-feature runtime remains audit-only；
6. selected-feature runtime loses agreement；
7. static bucket / persistent workspace not implemented；
8. kernel/sync/allocation count does not decrease；
9. avg candidates per kernel remains <2；
10. CUDA graph capture fails with no recorded reason；
11. basis/delta/bridge single-pass not materialized；
12. basis/delta runtime loses numerical correctness；
13. low-cost path uses source gap / formula proxy；
14. system step_ratio_q90 remains >1.50；
15. memory ratio >1.05；
16. system controller fails precision / coverage / bad-event / null-rate；
17. precision LCB below 0.75；
18. bad-event UCB above 0.05；
19. leave-dataset-out fails；
20. leave-stratum-out fails；
21. paired replay remains control-equivalent；
22. shuffle controls pass；
23. short-run task drops；
24. full run gives no macro / hard-stratum / geometry gain；
25. functional breaks system gate；
26. gains are explained by QuadraticFeatureMLP；
27. any teacher/loss/fake/proxy/offload/projection-as-pass violation occurs。
```

---

# Part XI. 最终解释规则

## Case A：P6 system pass + P7/P8 pass

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full run and robustness。

## Case B：P1 attribution pass but all runtime candidates fail

必须声明：

```text
we finally know the real internal cost, but no materialized runtime path closed it.
```

下一步根据 P1 dominant component 进入 targeted kernel work，不调 controller。

## Case C：selected-feature runtime pass but system still slow

必须声明：

```text
selected-feature was a real cost, but not the remaining dominant blocker.
```

下一步继续 static bucket / basis-delta runtime fusion。

## Case D：static bucket/workspace pass but system still slow

必须声明：

```text
fragmentation improved, but arithmetic or memory bandwidth of basis/delta path remains dominant.
```

下一步做 T2 recurrence / memory coalescing / arithmetic simplification。

## Case E：basis/delta/bridge runtime correct but still expensive

必须声明：

```text
exact true-delta path is correct, but lower-level fused implementation remains above system envelope.
```

下一步继续 kernel-native layout or reconsider primitive/system architecture, not decision tuning。

## Case F：system pass but paired replay fail

必须声明：

```text
controller is system-legal but not causally superior to matched controls.
```

下一步回到 functional event/value target，而不是继续 kernelization。

## Case G：LDO/LSO fail

必须声明：

```text
controller is not dataset-agnostic or stratum-agnostic enough.
```

不能通过 dataset-specific tuning 写成功；下一步修 support/family reliability。

---

# Part XII. 最终建议

v9.2.75 的一句话策略是：

$$
\boxed{
\text{不要再写 audit-only 或 not-implemented rows；先拿到真实 kernel-internal attribution，再至少落地一条 materialized runtime cost-reduction path。}
}
$$

当前最关键的问题不是：

```text
oracle 是否存在；
reference frontier 是否 deployable；
PF5 candidate rate 是否可行；
payload 是否 missing；
C3/T2/C4/E2 controller 是否要重调；
safe-useful target 是否要重设；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. fused_common_basis_delta_pipeline 内部到底哪个 subphase 慢？
2. basis_lift / quadratic / norm / W2_delta / probe_logits / launch / sync 字段为什么仍为空？
3. accept-bit / bridge-score inside kernel 能否真实 materialize？
4. selected_feature_materialized_count_after 能否从 2493 降下来？
5. static bucket / persistent workspace 能否把 kernel_count 3750、sync_count 1250、allocation_count 2493 降下来？
6. avg_candidates_per_kernel 能否从极低碎片化提升到 >=8？
7. single-pass basis-delta-bridge 是否能保持 true-delta exactness？
8. step q90 能否从 2.713296 或 2.346412 降到 <=1.50？
9. system controller 是否 official eligible？
10. LDO/LSO 是否通过？
11. official paired replay 是否打过 AdamWParallel / bestLR？
```

v9.2.75 的结果将给出清晰分叉：

```text
if kernel-internal attribution still incomplete:
  stop at R14; do not run more controller experiments.

if selected-feature runtime materializes but still slow:
  continue basis/delta/static-bucket work.

if static bucket/workspace materializes but still slow:
  inspect arithmetic/memory lower bound and T2 recurrence.

if single-pass basis-delta-bridge passes system gate:
  open system-legal controller, LDO/LSO, paired replay.

if system passes but paired replay fails:
  system is legal, but functional causal advantage is insufficient.

if LDO/LSO fails:
  repair dataset-agnostic support/family reliability, not dataset-specific tuning.
```
