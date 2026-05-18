# DG-KAN v9.2.74 Materialized Persistent Basis-Delta Kernel 与 Static-Bucket System Closure 完整实验计划

> 本计划基于 v9.2.73 `Step-Level Basis/Delta Fusion 与 System-Legal Controller Closure` 的真实执行结果制定。  
> v9.2.73 的 terminal route 是：
>
> ```text
> route = R18-SystemStillTooExpensive
> base_candidate = LQ-t2-h256
> success_v9273_strict_purekan_functional = False
> success_v9273_full_functional = False
> success_v9273_external_ready = False
> ```
>
> v9.2.73 的关键事实是：
>
> ```text
> Reference / decision:
>   controller_precision = 0.8380281690140845
>   controller_coverage = 0.03130511463844797
>   controller_bad_event = 0.02464788732394366
>   controller_null_rate = 0.13028169014084506
>   precision_lcb = 0.7907157243773478
>   bad_event_ucb = 0.04999458813129621
>   agreement_reference_accept = 1.0
>
> Payload / legality:
>   payload_binding_contract_pass = 1
>   candidate_tensor_payload_missing_count = 0
>   candidate_branch_logits_missing_count = 0
>   candidate_true_delta_logits_missing_count = 0
>   functional_update_payload_missing_count = 0
>   full_online_row_binding = 1
>   full_online_payload_binding = 1
>   full_online_update_payload_binding = 1
>   projection_used = 0
>   source_measured_gap_used = 0
>   formula_proxy_used = 0
>   cpu_offload_used = 0
>   dataset_name_used = 0
>
> Cost / system:
>   bf5_internal_cost_attribution_pass = 1
>   unknown_fraction = 0.0
>   dominant_internal_component = fused_common_basis_delta_pipeline_ms
>   fused_common_basis_delta_pipeline_ms = 1086.943672
>   true_delta_selected_feature_ms = 323.118709
>   total_wallclock_time_ms = 1410.062380
>   kernel_count = 3750
>   sync_count = 1250
>   allocation_count = 2493
>   avg_candidates_per_kernel = 0.6648
>
> Diagnostic candidates:
>   BD0-BF5Reference:
>     step_ratio_q90 = 2.713296
>   BD1-AuditOnlySelectedFeatureCostRemovalDiagnostic:
>     step_ratio_q90 = 2.3464118796089455
>     agreement = 1.0
>     logits_error = 2.670288e-05
>     delta_error = 8.149073e-10
>     official system candidate = 0
>   SF3-AuditOnlySelectedFeature:
>     selected_feature_total_ms_after = 0.0
>     step_ratio_q90 = 2.3464118796089455
>     audit-only diagnostic = 1
>   static bucket / persistent workspace:
>     not implemented
>     kernel_count_after = 3750
>     sync_count_after = 1250
>     allocation_count_after = 2493
>   no-hash timed path:
>     hash/csv/cpu metadata/norm item in timed path = 0
>     audit disagreement = 0
>
> Downstream:
>   P7-P10 not_run
>   reason = P6_system_controller_step_ratio_failed
> ```
>
> v9.2.74 的核心判断是：
>
> $$
> \boxed{
> \text{v9.2.73 没有实现真正 runtime closure；它证明的是 selected-feature removal 不够，且 static-bucket / persistent workspace 仍是未实现主 blocker。}
> }
> $$
>
> 因此 v9.2.74 不应该继续写 audit-only diagnostic，也不应该继续调 controller。  
> 本轮的核心任务是把 v9.2.73 的三个诊断结论转成真实系统实现：
>
> ```text
> 1. selected-feature 不要只是 accounting 扣除，要在 runtime 内改为 bridge-score / accept-bit 输出；
> 2. fused_common_basis_delta_pipeline 不要只 aggregate attribution，要做 kernel-internal timing 和 true fused implementation；
> 3. kernel/sync/allocation 不能保持 3750/1250/2493，要通过 static bucket + persistent workspace 降下来。
> ```
>
> v9.2.74 的目标不是再证明 signal，而是：
>
> $$
> \boxed{
> \text{在 payload/decision 全部保持过关的前提下，把真实 measured step ratio 从 }2.346412\text{ 或 }2.713296\text{ 压到 }\leq1.50。
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

v9.2.74 的新增硬约束是：

```text
audit-only cost removal 不能作为 official pass；
diagnostic_derived_from_measured_components = 1 的 row 不能作为 route-level system survivor；
selected-feature elimination 必须 materialized_runtime = 1；
static_bucket_workspace_used 必须真实改变 kernel/sync/allocation；
persistent_workspace_used 必须使 allocation_count_after = 0 或显著低于 baseline；
kernel-internal attribution 必须填 basis_lift / quadratic / norm / W2_delta / probe_logits / selected_feature / launch / sync 字段；
official system pass 必须来自真实 timed runtime path。
```

Official system pass 不允许依赖：

```text
projected step ratio
microprobe-only timing
audit-only component subtraction
formula-derived branch logits
source-measured scalar gap
CPU-heavy norm/hash summary
posthoc oracle label
validation/test metric at commit time
dataset_name branch
```

允许使用：

```text
PF5 runtime selector
full-online event table
candidate id / family id / branch id / bucket id
payload-bound candidate tensors
payload-bound branch/control logits
payload-bound true-delta logits
FrozenC3-T2PlusBackfill lookup
manual update-ready functional payload
static bucketed candidate tensors
persistent GPU workspace
CUDA / Triton fused kernels
CUDA graph capture if exactness and binding pass
calibration-split frozen thresholds
```

---

# Part I. 对 v9.2.73 的独立判断

## 1. v9.2.73 没有达到目标

v9.2.73 没有 strict PureKAN functional success。失败不是因为 reference controller、payload、true-delta signal 或 no-fake audit，而是因为系统 envelope 仍没过：

```text
system_legal_controller_pass = 0
official_eligible = 0
controller_step_ratio_q90 = 2.3464118796089455
required_step_ratio_q90 <= 1.50
```

因此 P7 leave-out、P8 official paired replay、P9 short-run、P10 full-run 都不能打开。  
这不是保守过头，而是正确 gate：v9.2.73 的 best row `BD1/SF3` 是 audit-only diagnostic，不是真实 runtime fused kernel。

## 2. v9.2.73 的真实进展

v9.2.73 的真实进展主要是三点。

第一，确认 v9.2.72 BF5 boundary 稳定：payload binding 继续 pass，decision frontier 继续 pass，no fake/proxy/offload 继续 pass。C3-T2PlusBackfill 仍然是合法 reference controller，不需要回到 controller search。

第二，selected-feature materialization 被定量确认为真实成本项。将 selected feature component 从 `323.118709 ms` audit-only 移除后，q90 step ratio 从 `2.713296` 降到 `2.346412`。这说明 selected-feature 输出确实值得改成 runtime 内 bridge-score / accept-bit / two-pass borderline，而不是继续完整 materialize。

第三，P5 no-hash timed path 审计通过：hash/csv/cpu metadata/norm item 没有污染 timed path。这排除了一个常见假 blocker，说明当前 step cost 主要来自 GPU/control path，而不是 artifact 写入或 CPU audit。

## 3. v9.2.73 的真实失败

v9.2.73 的失败也很明确。

第一，P1 只做到 aggregate attribution，没有做到 kernel-internal attribution。  
虽然 unknown fraction = `0.0`，但：

```text
basis_lift_time_ms = empty
basis_quadratic_time_ms = empty
basis_norm_time_ms = empty
W2_delta_time_ms = empty
probe_logits_time_ms = empty
sync_time_ms = empty
launch_overhead = empty
```

所以 `fused_common_basis_delta_pipeline_ms` 仍是一个黑箱大块。它告诉我们“慢在这里”，但还没告诉我们这个 block 里面到底哪个 kernel / memory movement / dynamic shape / launch / writeback 慢。

第二，P2/P3 的 best improvement 是 audit-only，不是 materialized runtime。  
`BD1-AuditOnlySelectedFeatureCostRemovalDiagnostic` 与 `SF3-AuditOnlySelectedFeature` 证明 selected feature 值得处理，但它们没有实现真实 kernel，不能作为 system survivor。

第三，P4 static bucket / persistent workspace / CUDA graph 完全没闭合：

```text
kernel_count_before = 3750
kernel_count_after = 3750
sync_count_before = 1250
sync_count_after = 1250
allocation_count_before = 2493
allocation_count_after = 2493
cuda_graph_attempted = 0
persistent_workspace_used = 0
```

这说明 v9.2.73 没有攻击到最像 system overhead 的部分。`avg_candidates_per_kernel = 0.6648` 也很异常，提示当前 path 极可能被小粒度 launch / sync / sparse dynamic candidate shape 拖垮。

## 4. 当前 blocker 的本质

当前 blocker 应该写成：

$$
\boxed{
\text{runtime materialization of basis/delta/accept pipeline is still fragmented; selected-feature removal is diagnostic, static bucket and persistent workspace are not implemented.}
}
$$

更具体地说：

```text
1. decision region 已经可部署；
2. payload / update payload 已经闭合；
3. true-delta signal 和 agreement 已经稳定；
4. branch/delta 局部 CUDA path 已经能保持数值；
5. 但 full runtime 仍由 fragmented basis/delta pipeline 和 per-candidate small-kernel overhead 主导；
6. v9.2.73 的 best cost improvement 是 audit-only subtraction，不是真实 fused runtime；
7. static bucket / persistent workspace / CUDA graph 没实现，所以 kernel/sync/allocation 没有下降。
```

因此，下一步如果继续只做 “再扣一个 component” 或 “再调 selected feature accounting”，就是小修小补。v9.2.74 必须实现真实 runtime survivor。

## 5. 是否还在正确道路上

是。路线仍是正确的，因为过去几轮把 blocker 逐层推到了更底层：

```text
v9.2.67:
  exact-reference controller 首次 deployable。

v9.2.68:
  frozen controller + cheap prefilter 成立，但 system path 是 projection。

v9.2.69:
  materialized microprobe 成立，但 full-online binding 缺失。

v9.2.70:
  full-online identity / selector / bridge lookup 绑定，但 payload 缺失。

v9.2.71:
  payload 和 update payload 全绑定，但 full step cost 太高。

v9.2.72:
  candidate pack、state replay、update payload 不再是 blocker；
  BF4/BF5 低层 CUDA 路线显著降 cost。

v9.2.73:
  selected-feature cost 可疑项被量化；
  no-hash timed path 排除；
  但真实 static-bucket/persistent/fused runtime 未实现。
```

这不是科学路线失败。它是系统实现还没闭合。  
当前不应该回到 basis sweep、dataset tuning 或 controller threshold；应该继续系统 kernel 化。

---

# Part II. v9.2.74 总体目标

v9.2.74 的总体目标是：

$$
\boxed{
\text{把 v9.2.73 audit-only selected-feature removal 和未实现 static bucket/workspace 转化为真实 materialized runtime path，使 step q90 }\leq1.50。
}
$$

本轮必须并行回答八个问题：

```text
Q1:
  v9.2.73 boundary 是否稳定复现？

Q2:
  fused_common_basis_delta_pipeline_ms 内部到底由哪些 kernel / memory / launch / sync 组成？

Q3:
  selected-feature materialization 能否真正改为 bridge-score-inside-kernel 或 accept-bit-inside-kernel，而不是 audit-only subtraction？

Q4:
  1250 fused steps / 3750 kernels / 1250 sync / 2493 allocations 能否通过 static bucket 和 persistent workspace 显著降低？

Q5:
  avg_candidates_per_kernel = 0.6648 是否说明当前 grouping / bucketing 设计错误？
  如果是，如何把 candidate work 合并成 batch-major kernels？

Q6:
  CUDA graph capture 是否可用？
  如果不可用，具体失败原因是什么：dynamic shape、pointer mutation、workspace alias、stream semantics、or graph-unsafe op？

Q7:
  runtime cost closure 后，decision metrics / payload binding / true-delta agreement 是否保持？

Q8:
  system controller 过线后，LDO/LSO 和 official paired replay 是否能打开？
```

v9.2.74 的核心 stop-go 是：

$$
\boxed{
materialized\_runtime\_path=1
\land
audit\_only\_cost\_removal=0
\land
StepRatio_{q90}\leq1.50
\land
Agreement_{\text{reference}}\geq0.90
\land
OfficialEligible=1.
}
$$

---

# Part III. 核心假设

## H1：v9.2.73 的 remaining blocker 是 runtime fragmentation，而不是 decision/payload failure

H1 成立标准：

P0 复现时满足：

```text
payload_binding_contract_pass = 1
all missing payload counts = 0
controller decision metrics pass
agreement_reference_accept >= 0.90
step_ratio_q90 > 1.50
dominant component = fused_common_basis_delta_pipeline_ms or selected-feature/launch path
```

H1 失败标准：

payload binding regression、decision regression、agreement regression。若 H1 失败，v9.2.74 必须先修稳定性，而不是继续 kernel fusion。

## H2：kernel-internal attribution 会显示 launch/sync/allocation 或 selected-feature writeback 是主要可移除成本

H2 成立标准：

P1 内部归因中至少一个可修 component 满足：

$$
ComponentRatio \geq 0.20.
$$

候选 components：

```text
kernel_launch_overhead
sync_time
allocation_time
selected_feature_writeback
basis_norm_recompute
candidate_gather_scatter
bridge_feature_writeback
```

H2 失败标准：

全部成本都集中在 arithmetic lower bound / memory bandwidth，且没有可移除 component。这时下一步要换成 arithmetic simplification / candidate-count reduction，而不是 static bucket。

## H3：selected-feature runtime materialization 可以被 accept-bit / bridge-score path 替代

v9.2.73 audit-only 消除 selected feature 后 step q90 仍是 `2.346412`，但它仍贡献了 `323.118709 ms`。H3 认为这部分可以真实减少，而不破坏 accept agreement。

H3 成立标准：

materialized candidate 满足：

```text
selected_feature_materialized_count_after <= 0.25 * before
bridge_score_inside_kernel = 1 or accept_bit_inside_kernel = 1
audit_only = 0
diagnostic_derived_from_measured_components = 0
```

同时：

$$
Agreement_{\text{reference}}\geq0.95,
$$

$$
StepRatio_{q90}\leq2.00
$$

作为 intermediate pass。

H3 失败标准：

selected-feature 不输出就无法保持 reference agreement，或只能用 proxy/source gap 保持。

## H4：static bucket / persistent workspace 能显著降低 3750 kernels、1250 sync、2493 allocations

H4 成立标准：

```text
kernel_count_after <= 0.50 * kernel_count_before
sync_count_after <= 0.50 * sync_count_before
allocation_count_after <= 0.10 * allocation_count_before
persistent_workspace_used = 1
```

若使用 CUDA graph：

```text
cuda_graph_attempted = 1
cuda_graph_capture_pass = 1
graph_replay_used = 1
```

H4 失败标准：

kernel/sync/allocation 不下降，或 graph capture 因动态 shape / memory alias 失败且没有替代 bucket strategy。

## H5：batch-major grouping 比 current carrier-major tiny kernels 更接近 MLP-like path

`avg_candidates_per_kernel = 0.6648` 强烈暗示当前 kernel 粒度过碎。H5 认为应把 candidates 按 `(family_id, branch_id, horizon, bucket_size)` 分组，再形成 batch-major contiguous kernels。

H5 成立标准：

```text
avg_candidates_per_kernel_after >= 8
or
effective_candidates_per_launch_after >= 8
```

并且 step q90 有显著下降：

$$
\Delta StepRatio_{q90}\leq-0.40.
$$

H5 失败标准：

candidate grouping 后 memory/coalescing 变差或 decision agreement 掉线。

## H6：system pass 后，才进入科学因果 gate

H6 成立标准：

P6 system pass 后 P7/P8 打开。  
H6 失败标准：

system pass 但 LDO/LSO or paired replay fail。此时 blocker 才转为 functional causal advantage / dataset-agnostic reliability，不再是 kernel。

---

# Part IV. Runtime cost model

v9.2.74 采用如下 full runtime model：

$$
T_{\text{sys}}
=
T_{\text{PF5}}
+
T_{\text{candidate\_bucket}}
+
T_{\text{basis\_lift}}
+
T_{\text{basis\_quadratic}}
+
T_{\text{basis\_norm}}
+
T_{\text{W2\_delta}}
+
T_{\text{probe\_logits}}
+
T_{\text{bridge\_score}}
+
T_{\text{accept\_bit}}
+
T_{\text{update\_payload}}
+
T_{\text{launch}}
+
T_{\text{sync}}
+
T_{\text{allocation}}
+
T_{\text{audit\_outside\_timed}}.
$$

Official timed path 不包含 audit-only CSV/hash，但必须保留 post-step audit。

官方 system gate：

$$
\frac{T_{\text{sys,q90}}}{T_{\text{MLP,q90}}}\leq1.50.
$$

当前 best diagnostic：

$$
StepRatio_{q90}=2.346412.
$$

需要继续减少：

$$
2.346412-1.50=0.846412.
$$

相对还需压缩：

$$
\frac{1.50}{2.346412}\approx0.639.
$$

也就是说，v9.2.74 需要把当前 best diagnostic runtime 再压掉约 `36.1%`，而且必须从 audit-only 变成真实 runtime。

---

# Part V. Candidate designs

## 1. Kernel-internal attribution candidates

### IA0：v9.2.73 aggregate reference

复现：

```text
fused_common_basis_delta_pipeline_ms = 1086.943672
true_delta_selected_feature_ms = 323.118709
step_ratio_q90 = 2.713296 or 2.346412 diagnostic
```

### IA1：CUDA-event internal instrumentation

在 fused path 内部分段埋点：

```text
basis_lift
basis_quadratic
basis_norm
candidate_gather
W2_delta
probe_logits
selected_feature_writeback
bridge_score
accept_bit
workspace_writeback
```

### IA2：launch/sync/allocation attribution

记录：

```text
kernel_count
launch_count
sync_count
allocation_count
cudaMalloc_count
torch_empty_count
contiguous_call_count
clone_count
cpu_transfer_count
```

### IA3：memory bandwidth and occupancy proxy

记录：

```text
bytes_read
bytes_written
effective_bandwidth
occupancy_proxy
register_pressure_proxy
shared_memory_bytes
avg_candidates_per_kernel
```

## 2. Selected-feature runtime candidates

### SF0：v9.2.73 audit-only selected feature

Reference only. Not official.

### SF1：BridgeScoreInsideKernelRuntime

不再写出 full selected-feature tensor，而是在 kernel 内计算 bridge score：

```text
bridge_score
bad_ucb
null_ucb
support_lcb
backfill_flag
trim_flag
```

### SF2：AcceptBitInsideKernelRuntime

kernel 直接输出：

```text
accept_bit
reject_reason_code
borderline_flag
```

只有 borderline rows materialize selected features。

### SF3：TwoPassBorderlineRuntime

Pass 1:

```text
bridge score / accept bit
```

Pass 2:

```text
full selected features only for borderline candidates
```

### SF4：AuditSubsetSelectedFeatureRuntime

official timed path 不 materialize selected features；post-step audit subset materializes full selected features and compares.

## 3. Static bucket / workspace candidates

### BW0：v9.2.73 no workspace reference

Reference:

```text
kernel_count = 3750
sync_count = 1250
allocation_count = 2493
```

### BW1：FixedBucketCandidateTensor

Bucket sizes：

```text
1,2,4,8,16,32
```

Pad candidates inside buckets but keep mask exact.

### BW2：FamilyBranchHorizonBucket

Bucket by:

```text
family_id
branch_id
horizon
bucket_size
```

### BW3：PersistentBasisDeltaWorkspace

Preallocate:

```text
candidate_workspace
basis_workspace
delta_workspace
bridge_score_workspace
accept_bit_workspace
update_payload_workspace
```

Timed path allocation must be near zero.

### BW4：CudaGraphStaticBucket

Attempt CUDA graph capture per bucket shape. Record failure reason if capture fails.

### BW5：BatchMajorCarrierFusion

Instead of carrier-major tiny kernels, fuse all candidates in a batch-major layout:

```text
candidate_dim major
basis_dim minor
branch/family bucket encoded as metadata
```

## 4. Basis/delta fused runtime candidates

### BD0：BF5 reference

Current best real path.

### BD1：SinglePassBasisDeltaBridgeRuntime

Single pass computes:

```text
basis
delta logits
bridge score
accept bit
```

### BD2：CommonBasisCacheRuntime

For repeated family/bucket/horizon, compute basis once and reuse in delta/bridge.

### BD3：RecurrenceT2BasisRuntime

Exploit T2 / quadratic structure:

$$
\phi(x)=a_0+a_1x+a_2x^2
$$

Compute all required T2 components via fused recurrence / FMA, not generic basis tensor materialization.

### BD4：W2DeltaOnlyRuntime

Since current update payload is W2-only compressed, restrict runtime to W2-relevant delta while preserving accept agreement.

### BD5：MaskedFullDeltaAuditRuntime

Use W2-only / selected delta timed path, but periodically run full delta audit subset.

## 5. System candidates

### SYS0：v9.2.73 best diagnostic reference

Expected fail.

### SYS1：IA1 + SF1 + BW3

Internal attribution + bridge-score inside kernel + persistent workspace.

### SYS2：IA1 + SF2 + BW1

Accept-bit inside kernel + fixed buckets.

### SYS3：BD1 + SF1 + BW3

Single-pass basis/delta/bridge runtime with persistent workspace.

### SYS4：BD3 + SF2 + BW5

T2 recurrence + accept-bit + batch-major carrier fusion.

### SYS5：BD4 + SF3 + BW3

W2-focused runtime + two-pass borderline + persistent workspace.

### SYS6：BD2 + BW4

Common basis cache + CUDA graph bucket replay.

### SYS7：HybridBestRuntimeV9274

Best materialized candidate from above. Only SYS7 can be promoted if all audits pass.

---

# Part VI. 实验阶段

## P0：v9.2.73 boundary reproduction

### 目标

确认 v9.2.73 boundary 稳定。

### 必须记录

```text
route
source_route_v9273
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
bd1_step_ratio_q90
kernel_count
sync_count
allocation_count
selected_feature_time_ms
system_legal_controller_pass
official_eligible
fake_proxy_count
cpu_offload_used
```

### 判断标准

P0 pass：

```text
route = R18-SystemStillTooExpensive
payload binding pass = 1
all missing payload counts = 0
decision metrics pass
system controller pass = 0
step_ratio_q90 > 1.50
fake/proxy/offload = 0
```

### 可视化

```text
p0_v9273_boundary_ladder.svg
p0_step_ratio_progress_v9271_v9273.svg
p0_payload_decision_system_gate.svg
```

---

## P1：kernel-internal cost attribution

### 目标

将 `fused_common_basis_delta_pipeline_ms` 从 aggregate block 拆成 kernel-internal subcomponents。

### 必须记录

```text
internal_cost_candidate_id
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
```

### 判断标准

P1 pass：

```text
unknown_fraction <= 0.05
at least one internal removable component ratio >= 0.20
```

如果 P1 仍只有 aggregate block，route 必须停在：

```text
R14-KernelInternalAttributionIncomplete
```

### 可视化

```text
p1_internal_cost_waterfall.svg
p1_launch_sync_allocation_breakdown.svg
p1_candidates_per_kernel_histogram.svg
p1_bandwidth_occupancy_proxy.svg
```

---

## P2：materialized selected-feature elimination

### 目标

把 v9.2.73 的 audit-only selected feature elimination 改成真实 runtime path。

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
selected_feature_materialized_count_after <= 0.25 * before
agreement_reference_accept >= 0.90
audit_agreement >= 0.99
projection_used = 0
source_gap_used = 0
formula_proxy_used = 0
```

Intermediate system diagnostic：

$$
StepRatio_{q90}\leq2.00.
$$

### 可视化

```text
p2_selected_feature_runtime_cost.svg
p2_accept_bit_confusion.svg
p2_borderline_candidate_distribution.svg
p2_selected_feature_materialization_count.svg
```

---

## P3：static bucket and persistent workspace

### 目标

真正降低 kernel/sync/allocation count，而不是保持 v9.2.73 的 3750/1250/2493。

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

P3 system candidate pass：

$$
StepRatio_{q90}\leq1.80.
$$

### 可视化

```text
p3_bucket_kernel_sync_allocation.svg
p3_avg_candidates_per_kernel.svg
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

System diagnostic pass：

$$
StepRatio_{q90}\leq1.80.
$$

Full system candidate pass：

$$
StepRatio_{q90}\leq1.50.
$$

### 可视化

```text
p4_single_pass_cost_quality_pareto.svg
p4_error_vs_step_ratio.svg
p4_kernel_count_before_after.svg
p4_reference_agreement_confusion.svg
```

---

## P5：audit separation and official timed path

### 目标

保证 no-fake/no-proxy 审计保留，但 hash、CSV、CPU metadata、selected-feature audit 不进入 official timed path。

### 必须记录

```text
audit_runtime_id
hash_in_timed_path
csv_write_in_timed_path
cpu_metadata_in_timed_path
norm_item_in_timed_path
selected_feature_audit_in_timed_path
audit_stream_used
post_step_audit_used
audit_subset_size
audit_disagreement_count
timed_path_cuda_ms
timed_path_wallclock_ms
artifact_io_time_ms
hash_norm_time_ms
step_ratio_q90
agreement_reference_accept
```

### 判断标准

P5 pass：

```text
hash_in_timed_path = 0
csv_write_in_timed_path = 0
cpu_metadata_in_timed_path = 0
norm_item_in_timed_path = 0
selected_feature_audit_in_timed_path = 0
audit_disagreement_count = 0
agreement_reference_accept >= 0.90
```

### 可视化

```text
p5_timed_vs_audit_path.svg
p5_cpu_metadata_elimination.svg
p5_audit_subset_agreement.svg
```

---

## P6：system-legal exact-signal controller v6

### 目标

组合 P2-P5 survivor，建立 official system-legal controller。

### 必须记录

```text
controller_id
system_candidate_id
selected_feature_runtime_id
bucket_workspace_id
basis_delta_runtime_id
audit_runtime_id
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
p6_step_ratio_progress_v9271_to_v9274.svg
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
contract_audit_v9274.csv
p0_v9273_boundary_reproduction.csv
p1_kernel_internal_cost_attribution.csv
p2_materialized_selected_feature_elimination.csv
p3_static_bucket_persistent_workspace.csv
p4_single_pass_basis_delta_bridge_runtime.csv
p5_audit_separation_official_timed_path.csv
p6_system_legal_exact_signal_controller_v6.csv
p7_leave_dataset_and_stratum_out.csv
p8_official_paired_replay.csv
p9_short_run_functional_validation.csv
p10_full_run_robustness_strong_baseline.csv

kernel_internal_cost_trace_v9274.csv
selected_feature_runtime_trace_v9274.csv
static_bucket_workspace_trace_v9274.csv
basis_delta_bridge_runtime_trace_v9274.csv
audit_separation_trace_v9274.csv
system_controller_trace_v9274.csv
leaveout_trace_v9274.csv
paired_replay_branch_trace_v9274.csv
short_run_trace_v9274.csv

route_decision.json
aggregate_decision.json
failure_table.csv
artifact_hashes.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9273_boundary_unstable
F3_dataset_tuning_detected
F4_payload_binding_regression
F5_decision_metric_regression
F6_kernel_internal_attribution_incomplete
F7_no_dominant_fusible_component
F8_selected_feature_runtime_not_materialized
F9_selected_feature_elimination_agreement_fail
F10_static_bucket_not_implemented
F11_persistent_workspace_not_implemented
F12_cuda_graph_capture_fail_unexplained
F13_kernel_sync_allocation_unfixed
F14_avg_candidates_per_kernel_too_low
F15_basis_delta_runtime_error_fail
F16_basis_delta_runtime_agreement_fail
F17_basis_delta_runtime_system_fail
F18_audit_only_cost_removal_used
F19_source_gap_or_formula_proxy_used
F20_system_controller_precision_fail
F21_system_controller_coverage_fail
F22_system_controller_bad_event_fail
F23_system_controller_null_rate_fail
F24_system_controller_lcb_ucb_fail
F25_system_controller_step_ratio_fail
F26_system_controller_memory_fail
F27_system_controller_projection_used
F28_leave_dataset_out_fail
F29_leave_stratum_out_fail
F30_paired_replay_control_equivalent
F31_shuffle_control_pass
F32_functional_lr_equivalent
F33_short_run_task_drop
F34_full_run_no_macro_hard_stratum_gain
F35_strong_baseline_explains_gain
F36_robustness_fail
F37_external_not_ready
F38_fake_or_proxy_violation
F39_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-BoundaryReproduced:
  v9.2.73 boundary reproduced.

R2-KernelInternalCostAttributed:
  fused_common_basis_delta pipeline has internal attribution with unknown_fraction <= 0.05.

R3-MaterializedSelectedFeatureRuntimePass:
  selected-feature runtime path is materialized and not audit-only.

R4-StaticBucketWorkspacePass:
  kernel/sync/allocation count is reduced through static bucket and persistent workspace.

R5-SinglePassBasisDeltaBridgePass:
  basis/delta/bridge score runtime is true-delta exact and materialized.

R6-SystemLegalExactSignalControllerPass:
  system controller passes decision + compute gates.

R7-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R8-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R9-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R10-ShortRunFunctionalPass:
  short-run task-safe mechanism gain.

R11-FullFunctionalPass:
  full run task / geometry / system / control gates pass.

R12-PayloadBindingRegression:
  v9.2.73 payload binding no longer reproduces.

R13-KernelInternalAttributionIncomplete:
  internal cost still only aggregate; no actionable fusion target.

R14-SelectedFeatureRuntimeStillDiagnostic:
  selected-feature removal remains audit-only, not materialized.

R15-StaticBucketWorkspaceUnimplemented:
  kernel/sync/allocation remain unchanged.

R16-BasisDeltaRuntimeStillTooSlow:
  true low-level fused path is correct but above system envelope.

R17-SystemStillTooExpensive:
  decision and exactness pass but step_ratio_q90 remains >1.50.

R18-ComputePassButLeaveoutFail:
  system controller overfits pooled calibration.

R19-ComputePassButPairedReplayFail:
  controller is system-legal but not causally superior to controls.

R20-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` 必须记录：

```text
route
v9273_boundary_pass
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
selected_feature_materialized_count_after
audit_only_cost_removal
best_bucket_workspace_id
static_bucket_workspace_pass
kernel_count_reduction
sync_count_reduction
allocation_count_reduction
avg_candidates_per_kernel_after
cuda_graph_capture_pass
cuda_graph_failure_reason
best_basis_delta_runtime_id
basis_delta_runtime_pass
cuda_vs_torch_logits_error_max
cuda_vs_torch_delta_error_max
basis_delta_agreement
basis_delta_step_ratio_q90
best_audit_runtime_id
audit_separation_pass
hash_in_timed_path
cpu_metadata_in_timed_path
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
success_v9274_strict_purekan_functional
success_v9274_full_functional
success_v9274_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 kernel-internal cost attribution
  P2 materialized selected-feature runtime
  P3 static bucket / persistent workspace
  P4 single-pass basis/delta/bridge runtime
  P5 audit separation

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
P2/P3/P4/P5 can run in parallel after P1 attribution.
P6 cannot pass unless:
  P0 pass
  P1 internal attribution pass
  payload binding remains pass
  at least one P2/P3/P4/P5 materialized runtime candidate reaches step_ratio_q90 <= 1.50
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
v9.2.73 boundary reproduced
kernel-internal cost attributed
materialized selected-feature runtime measured
static bucket / persistent workspace measured
single-pass basis/delta/bridge runtime measured
audit separation measured
system controller measured
no fake/proxy/offload/loss/teacher violation
```

## Runtime materialization success

```text
Minimum diagnostic success
+
selected-feature cost reduction is materialized_runtime, not audit-only
+
kernel/sync/allocation count reduced
+
persistent workspace or static bucket actually used
+
basis/delta/bridge runtime uses true branch-delta
+
no source gap / formula proxy / projection
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
1. v9.2.73 boundary cannot be reproduced；
2. payload binding regresses；
3. kernel-internal attribution still only aggregate；
4. selected-feature removal remains audit-only；
5. selected-feature runtime elimination loses agreement；
6. static bucket / persistent workspace not implemented；
7. kernel/sync/allocation count does not decrease；
8. avg candidates per kernel remains <2；
9. CUDA graph capture fails with no fallback；
10. basis/delta runtime loses numerical correctness；
11. basis/delta runtime uses source gap / formula proxy；
12. system step_ratio_q90 remains >1.50；
13. memory ratio >1.05；
14. system controller fails precision / coverage / bad-event / null-rate；
15. precision LCB below 0.75；
16. bad-event UCB above 0.05；
17. leave-dataset-out fails；
18. leave-stratum-out fails；
19. paired replay remains control-equivalent；
20. shuffle controls pass；
21. short-run task drops；
22. full run gives no macro / hard-stratum / geometry gain；
23. functional breaks system gate；
24. gains are explained by QuadraticFeatureMLP；
25. any teacher/loss/fake/proxy/offload/projection-as-pass violation occurs。
```

---

# Part XI. 最终解释规则

## Case A：P6 system pass + P7/P8 pass

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full run and robustness。

## Case B：selected-feature runtime materialization passes but system still slow

必须声明：

```text
selected-feature materialization was a real cost, but not the dominant remaining blocker.
```

下一步继续 static bucket / basis-delta fusion，不回到 controller tuning。

## Case C：static bucket / persistent workspace passes but system still slow

必须声明：

```text
launch/allocation fragmentation improved, but arithmetic or memory bandwidth of basis/delta path remains dominant.
```

下一步做 arithmetic simplification / T2 recurrence / memory coalescing。

## Case D：basis/delta runtime correct but still expensive

必须声明：

```text
exact true-delta path is correct, but lower-level fused basis/delta implementation remains above system envelope.
```

下一步继续 kernel-native layout, not decision tuning.

## Case E：system pass but paired replay fail

必须声明：

```text
controller is system-legal but not causally superior to matched controls.
```

下一步回到 functional event/value target，而不是继续 kernelization。

## Case F：LDO/LSO fail

必须声明：

```text
controller is not dataset-agnostic or stratum-agnostic enough.
```

不能通过 dataset-specific tuning 写成功；下一步修 support/family reliability。

---

# Part XII. 最终建议

v9.2.74 的一句话策略是：

$$
\boxed{
\text{不要再做 audit-only cost subtraction；把 selected-feature、basis/delta、bucket/workspace 做成真实 runtime，真正把 }2.346412\text{ 压到 }\leq1.50。
}
$$

当前最关键的问题不是：

```text
oracle 是否存在；
reference frontier 是否 deployable；
payload 是否 missing；
PF5 candidate rate 是否可行；
C3/T2/C4/E2 controller 是否要重调；
safe-useful target 是否要重设；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. fused_common_basis_delta_pipeline 内部到底哪个 kernel 慢？
2. selected-feature removal 能否从 audit-only 变成 materialized runtime？
3. bridge score / accept bit 能否直接在 kernel 内输出？
4. 3750 kernel / 1250 sync / 2493 allocation 能否被 static bucket + persistent workspace 显著降低？
5. avg_candidates_per_kernel = 0.6648 是否说明当前 kernel 粒度根本错误？
6. candidate work 能否改成 batch-major fused runtime？
7. CUDA graph 是否可用；不可用原因是什么？
8. step q90 能否从 2.346412 压到 <=1.50？
9. system controller 是否 official eligible？
10. LDO/LSO 是否通过？
11. official paired replay 是否打过 AdamWParallel / bestLR？
```

v9.2.74 的结果将给出清晰分叉：

```text
if system controller passes:
  open LDO/LSO and official paired replay.

if selected-feature runtime passes but system still slow:
  continue basis/delta runtime fusion and static bucket.

if static bucket/workspace fails:
  repair candidate layout / bucket shape / persistent workspace before more math kernels.

if basis/delta fusion correct but still slow:
  continue arithmetic simplification / memory layout / recurrence.

if compute passes but paired replay fails:
  system is legal, but functional causal advantage is insufficient.

if LDO/LSO fails:
  repair dataset-agnostic support/family reliability, not dataset-specific tuning.
```
