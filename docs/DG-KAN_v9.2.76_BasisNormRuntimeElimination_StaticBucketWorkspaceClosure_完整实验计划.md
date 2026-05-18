# DG-KAN v9.2.76 BasisNorm Runtime Elimination 与 Static-Bucket Persistent Workspace Closure 完整实验计划

> 本计划基于 v9.2.75 `Kernel-Internal Attribution First 与 Static-Bucket Runtime Closure` 的真实执行结果制定。  
> v9.2.75 的 terminal route 是：
>
> ```text
> route = R16-StaticBucketWorkspaceStillUnimplemented
> base_candidate = LQ-t2-h256
> success_v9275_strict_purekan_functional = False
> success_v9275_full_functional = False
> success_v9275_external_ready = False
> ```
>
> v9.2.75 的核心事实是：
>
> ```text
> P0:
>   v9.2.74 boundary reproduced = 1
>   payload_binding_contract_pass = 1
>   candidate_tensor_payload_missing_count = 0
>   candidate_branch_logits_missing_count = 0
>   candidate_true_delta_logits_missing_count = 0
>   functional_update_payload_missing_count = 0
>   controller_precision = 0.8380281690140845
>   controller_coverage = 0.03130511463844797
>   controller_bad_event = 0.02464788732394366
>   controller_null_rate = 0.13028169014084506
>
> P1 real kernel-internal attribution:
>   kernel_internal_attribution_pass = 1
>   materialized_runtime_path = 1
>   sample_step_count = 24
>   unknown_fraction = 0.012945534729646649
>   dominant_subcomponent = basis_norm_time_ms
>   dominant_removable_or_fusible_component_ratio = 0.6766760956112658
>
> P1 subphase mean timing:
>   candidate_gather_time_ms = 0.005768
>   basis_lift_time_ms = 0.177963
>   basis_quadratic_time_ms = 0.155845
>   basis_norm_time_ms = 5.664183
>   W2_delta_time_ms = 1.788365
>   probe_logits_time_ms = 0.095465
>   selected_feature_writeback_time_ms = 0.134961
>   bridge_score_time_ms = 0.183738
>   accept_bit_time_ms = 0.093108
>   workspace_writeback_time_ms = 0.023667
>   kernel_launch_time_ms = 0.025506
>   sync_time_ms = 0.004730
>   allocation_time_ms = 0.017298
>
> P2 materialized selected-feature runtime:
>   selected_feature_runtime_id = SF2-AcceptBitTensorGatherRuntimeV2
>   materialized_runtime_path = 1
>   audit_only = 0
>   diagnostic_derived_from_measured_components = 0
>   selected_feature_materialized_count_before = 2493
>   selected_feature_materialized_count_after = 24
>   borderline_count = 24
>   accept_bit_inside_kernel = 1
>   bridge_score_inside_kernel = 0
>   agreement_reference_accept = 1.0
>   audit_agreement = 1.0
>   selected_feature_runtime_pass = 1
>
> P3 static bucket / persistent workspace:
>   bucket_workspace_id = BW1-FixedBucketCandidateTensorDiagnosticOnly
>   bucket_strategy = fixed_bucket_table_materialized_no_integrated_runtime
>   candidate_count = 2493
>   bucket_sizes = {"1": 7260, "2": 365, "4": 439}
>   kernel_count_before = 3750
>   kernel_count_after = 3750
>   sync_count_before = 1250
>   sync_count_after = 1250
>   allocation_count_before = 2493
>   allocation_count_after = 2493
>   persistent_workspace_used = 0
>   cuda_graph_attempted = 0
>   avg_candidates_per_kernel_after = 0.6648
>   static_bucket_workspace_pass = 0
>
> P4 basis/delta/bridge runtime:
>   basis_delta_runtime_id = BD0-BF5ReferenceRuntimePlusP1Attribution
>   uses_true_branch_delta = 1
>   uses_source_measured_gap = 0
>   uses_formula_proxy = 0
>   basis_delta_bridge_single_pass = 0
>   bridge_score_inside_kernel = 0
>   accept_bit_inside_kernel = 1
>   cuda_vs_torch_check_count = 24
>   cuda_vs_torch_logits_error_max = 2.6702880859375e-05
>   cuda_vs_torch_delta_error_max = 8.149072527885437e-10
>   agreement_reference_accept = 1.0
>   step_ratio_q90 = 2.713295831053225
>   basis_delta_runtime_pass = 0
>
> P5 integrated runtime:
>   system_candidate_id = SYS2-IA1PlusSF2NoStaticBucket
>   materialized_runtime_path = 0
>   diagnostic_derived_from_measured_components = 1
>   kernel_count = 3750
>   sync_count = 1250
>   allocation_count = 2493
>   avg_candidates_per_kernel = 0.6648
>   step_ratio_q90 = 2.713295831053225
>   integrated_runtime_pass = 0
>
> P6:
>   official_eligible = 0
>   system_legal_controller_pass = 0
>   reason = integrated_materialized_runtime_failed
>   precision_heldout = 0.8380281690140845
>   coverage_heldout = 0.03130511463844797
>   bad_event_heldout = 0.02464788732394366
>   null_rate_heldout = 0.13028169014084506
>   precision_lcb = 0.7907157243773478
>   bad_event_ucb = 0.04999458813129621
>   agreement_reference_accept = 1.0
>   step_ratio_q90 = 2.713295831053225
>   memory_ratio = 1.0
> ```
>
> v9.2.76 的核心判断是：
>
> $$
> \boxed{
> \text{v9.2.75 已经完成“知道慢在哪里”；v9.2.76 必须完成“让慢的地方真的变快”。}
> }
> $$
>
> 更具体地说，v9.2.75 第一次把 aggregate black box 拆出真实 subphase，并证明 dominant 是 `basis_norm_time_ms`，不是 selected-feature writeback、kernel launch、sync、allocation 或 candidate gather。  
> 因此 v9.2.76 不应继续写更多 attribution-only rows，而要围绕 `basis_norm` 做 runtime elimination / fusion / cache / recurrence，并同时把 static bucket / persistent workspace 接入 basis-delta runtime。

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

v9.2.76 的新增硬约束是：

```text
不能再把 attribution pass 当 runtime pass；
不能再把 selected-feature runtime pass 当 integrated system pass；
不能再把 static bucket occupancy 统计当 static bucket runtime；
不能再把 accept_bit_inside_kernel = 1 当 bridge_score_inside_kernel = 1；
不能使用 source-measured gap 或 formula proxy；
不能按 dataset 选择不同 controller / threshold / kernel route；
不能把 not_implemented 写成 failure-training 或 success；
official pass 必须来自真实 measured runtime path。
```

允许按 dataset / family / horizon 诊断问题：

```text
per-dataset candidate_rate
per-dataset basis_norm_time
per-family basis_norm_time
per-horizon bucket occupancy
per-stratum agreement drift
per-bucket kernel occupancy
```

但 official route 必须 dataset-agnostic：

```text
same PF5 selector
same C3-T2PlusBackfill frozen rule
same basis-norm runtime
same static bucket strategy
same pass thresholds
same system gate
```

---

# Part I. 对 v9.2.75 的独立判断

## 1. v9.2.75 没有达到目标

v9.2.75 没有 strict PureKAN functional success。原因很直接：

```text
integrated_runtime_pass = 0
system_legal_controller_pass = 0
official_eligible = 0
step_ratio_q90 = 2.713295831053225
required step_ratio_q90 <= 1.50
```

P7 leave-dataset-out / leave-stratum-out、P8 official paired replay、P9 short-run、P10 full-run 全部因 P3/P5/P6 gate 未过而不得打开。这是正确 gate，不是保守过头。因为当前 best candidate 仍然没有 integrated materialized runtime，不能把 P1 attribution pass 或 P2 accept-bit runtime pass 倒灌成 official system success。

## 2. v9.2.75 的真实进展

v9.2.75 有两项非常关键的真实推进。

第一，kernel-internal attribution 终于不是 aggregate 黑箱。v9.2.74 停在 `R13-KernelInternalAttributionIncomplete`，因为 `basis_lift_time_ms`、`basis_quadratic_time_ms`、`basis_norm_time_ms`、`kernel_launch_time_ms`、`sync_time_ms` 等字段为空。v9.2.75 中这些字段有了真实测量，并且 unknown fraction 只有 `0.012946`，满足 `<=0.05`。

第二，selected-feature runtime 不再只是 audit-only subtraction。`SF2-AcceptBitTensorGatherRuntimeV2` 真实 materialize，`audit_only = 0`，`diagnostic_derived_from_measured_components = 0`，并把 selected-feature materialized count 从 `2493` 降到 `24`，同时 agreement 和 audit agreement 都是 `1.0`。这说明 selected-feature 路线在局部 runtime 层已经修好。

## 3. v9.2.75 的真实失败

v9.2.75 的失败也很明确。

第一，P3 没有把 static bucket 接入 basis/delta runtime。虽然 bucket table 被 materialize，但它是 diagnostic-only。kernel/sync/allocation 完全没下降：

```text
kernel_count: 3750 -> 3750
sync_count: 1250 -> 1250
allocation_count: 2493 -> 2493
persistent_workspace_used = 0
cuda_graph_attempted = 0
avg_candidates_per_kernel_after = 0.6648
```

第二，P4 没有形成 single-pass basis/delta/bridge runtime：

```text
basis_delta_bridge_single_pass = 0
bridge_score_inside_kernel = 0
accept_bit_inside_kernel = 1
```

这说明 accept-bit 已经进 kernel，但 bridge score 没进 kernel；basis/delta 仍不是 single pass；整体仍是 BF5 reference runtime plus P1 attribution。

第三，P5 integrated runtime 没有 survivor。`SYS2-IA1PlusSF2NoStaticBucket` 的 `materialized_runtime_path = 0`，`diagnostic_derived_from_measured_components = 1`，所以 P6 不能 official。

## 4. 当前 blocker 的本质

当前 blocker 不是：

```text
controller quality；
payload missing；
selected-feature materialization；
kernel launch / sync / allocation 主导；
candidate gather；
source/proxy contamination；
dataset-specific failure。
```

P1 已经说明 basis norm / quantile / tail/context construction 才是 dominant：

$$
T_{\text{basis-norm}}=5.664183\text{ ms},
$$

而 W2 delta 是：

$$
T_{\text{W2-delta}}=1.788365\text{ ms}.
$$

selected-feature writeback 只有：

$$
T_{\text{selected-feature-writeback}}=0.134961\text{ ms}.
$$

kernel launch / sync / allocation 更小：

$$
T_{\text{launch}}=0.025506\text{ ms},
$$

$$
T_{\text{sync}}=0.004730\text{ ms},
$$

$$
T_{\text{allocation}}=0.017298\text{ ms}.
$$

因此当前最准确的本质是：

$$
\boxed{
\text{basis-normalization / quantile-tail-context construction 是主成本，且 static bucket/workspace 尚未接入 integrated runtime。}
}
$$

换句话说，如果下一步继续主要修 selected-feature 或 kernel launch，会偏离数据。selected-feature 已经修到局部 runtime pass；launch/sync/allocation 在 P1 subphase 中不是 dominant。真正应该打的是 basis_norm。

## 5. 是否还在正确道路上

是，而且 v9.2.75 比 v9.2.74 真实前进了一步。现在路线已经非常具体：

```text
decision frontier solved
payload binding solved
selected-feature runtime solved locally
kernel-internal attribution solved
basis_norm dominant identified
static bucket/workspace not integrated
single-pass bridge score not integrated
system controller not official
```

所以不能回去调 C3/T2PlusBackfill threshold，也不能回去重设 safe-useful target。现在该修的是 basis norm runtime 与 static bucket integration。

---

# Part II. v9.2.76 总体目标

v9.2.76 的总体目标是：

$$
\boxed{
\text{在保持 decision / payload / true-delta agreement 全部不退化的前提下，把 basis_norm 主成本压下来，并让 static bucket + persistent workspace 接入 integrated runtime。}
}
$$

强目标是：

$$
StepRatio_{q90}\leq1.50.
$$

最低有效推进目标是：

$$
StepRatio_{q90}\leq2.00,
$$

并且：

```text
materialized_runtime_path = 1
integrated_runtime_pass = 1
audit_only_cost_removal = 0
diagnostic_derived_from_measured_components = 0
agreement_reference_accept >= 0.90
payload binding remains pass
```

v9.2.76 必须回答九个问题：

```text
Q1:
  v9.2.75 boundary 是否稳定复现？

Q2:
  basis_norm_time_ms = 5.664183 的内部组成是什么？
  它到底是 quantile、tail、context、normalization、sorting/topk、还是 metadata gather？

Q3:
  basis_norm 是否可以从 per-candidate recompute 改成 per-bucket / per-family / per-step cache？

Q4:
  basis_norm 是否真的必须进入 official timed path？
  哪些部分是 controller decision 必需，哪些只是 audit / confidence summary？

Q5:
  selected-feature accept-bit runtime 能否与 bridge_score_inside_kernel 合并？

Q6:
  static bucket / persistent workspace 能否真实接入 basis_norm + W2_delta runtime，
  而不只是生成 bucket table？

Q7:
  avg_candidates_per_kernel = 0.6648 是否可以提升到 >=8？
  如果不能，是什么阻止 batch-major layout？

Q8:
  integrated materialized runtime 能否达到 step q90 <=2.00 或 <=1.50？

Q9:
  如果 system pass，LDO/LSO 和 official paired replay 是否能打开？
```

---

# Part III. 核心假设

## H1：basis_norm 主成本可拆为可融合或可缓存的 subcomponents

H1 认为 `basis_norm_time_ms = 5.664183` 不是不可约下界，而是由多个可拆组件组成：

```text
basis value normalization
quantile / tail statistic
context construction
family/bucket reliability lookup
candidate-level normalization
per-row temporary writeback
CPU/GPU metadata interaction
```

H1 成立标准：

```text
basis_norm_internal_attribution_pass = 1
basis_norm_unknown_fraction <= 0.05
at least one removable/cacheable/fusible component ratio >= 0.20
```

H1 失败标准：

basis_norm 内部仍无法拆分，或拆分后没有任何可操作 component。若 H1 失败，v9.2.76 不能继续随意写 basis_norm optimization success。

## H2：basis_norm 可以从 per-candidate recompute 改为 bucket/family cache

H2 认为大量 basis_norm 计算在 family / horizon / bucket 层重复，可以缓存：

$$
N_{\text{candidate}}=2493
$$

但真正 distinct normalization contexts 应远少于 candidate 数量。

H2 成立标准：

```text
basis_norm_cache_used = 1
cache_key = family_id / horizon / bucket_id / role_id / norm_context
cache_hit_rate >= 0.50
basis_norm_time_after <= 0.50 * basis_norm_time_before
agreement_reference_accept >= 0.95
```

H2 失败标准：

cache key 太细，hit rate <0.20，或 cache 改变 agreement。

## H3：basis_norm 的一部分是 audit/confidence summary，不应进入 timed controller path

H3 认为 basis_norm 中可能混有用于 audit 的 quantile / tail / context summary，而 official controller 只需要 frozen C3 accept decision 所需的 minimal basis-norm features。

H3 成立标准：

```text
audit_norm_removed_from_timed_path = 1
decision_required_norm_fields preserved = 1
post_step_audit_subset_agreement >= 0.99
agreement_reference_accept >= 0.95
basis_norm_time_after <= 0.60 * before
```

H3 失败标准：

去掉 audit norm 后 reference agreement <0.90，说明这些 norm fields 是 controller 必需。

## H4：static bucket / persistent workspace 必须接入 basis_norm and delta, 不能只接入 candidate table

H4 成立标准：

```text
static_bucket_runtime_used = 1
persistent_workspace_used = 1
basis_norm_bucketed = 1
W2_delta_bucketed = 1
allocation_count_after <= 0.10 * allocation_count_before
avg_candidates_per_kernel_after >= 8
kernel_count_after <= 0.50 * kernel_count_before
sync_count_after <= 0.50 * sync_count_before
```

H4 失败标准：

bucket table 有了，但 basis_norm / delta runtime 仍按 old path 执行。

## H5：bridge score 必须进入 kernel，accept-bit alone 不足以形成 integrated runtime

v9.2.75 已有：

```text
accept_bit_inside_kernel = 1
bridge_score_inside_kernel = 0
```

H5 认为这不足以形成 official integrated runtime，因为 bridge score / bad UCB / null UCB / support LCB 仍可能在 kernel 外 materialize 或 lookup。

H5 成立标准：

```text
bridge_score_inside_kernel = 1
bad_ucb_inside_kernel = 1 or lookup_without_materialization = 1
null_ucb_inside_kernel = 1 or lookup_without_materialization = 1
support_lcb_inside_kernel = 1 or lookup_without_materialization = 1
agreement_reference_accept >= 0.95
```

H5 失败标准：

bridge score 不进 kernel，P5 integrated runtime 仍 diagnostic-derived。

## H6：如果 basis_norm 优化和 static bucket 接入后仍 >1.50，才可以讨论 lower bound 或 architecture pivot

H6 成立标准：

```text
basis_norm optimization materialized = 1
static bucket workspace materialized = 1
bridge-score single-pass materialized = 1
agreement >= 0.90
step_ratio_q90 > 1.50
```

只有这时才能说：当前 FC-PureKAN true-delta controller 的 lower-level runtime 可能接近下界，需要考虑 primitive/system architecture pivot。  
H6 失败标准：P3/P4/P5 还没 materialize 就声称 lower bound 已到。

---

# Part IV. Runtime architecture design

v9.2.76 的目标 runtime 不再是：

```text
PF5 selector
→ candidate rows
→ BF5 reference basis/delta
→ selected-feature accept-bit
→ diagnostic bucket table
→ P6 not_run
```

而应是：

```text
PF5 selector
→ static bucket runtime
→ persistent candidate/basis/delta workspace
→ basis_norm cache / basis_norm fused kernel
→ W2_delta / probe_logits / bridge_score / accept_bit single-pass kernel
→ accepted update payload
→ post-step audit subset
```

形式化写为：

$$
Accept_{\text{sys}}(e)
=
PF5(e)
\land
K_{\text{basis-norm-delta-bridge}}(e)
\land
FrozenC3(e).
$$

其中：

$$
K_{\text{basis-norm-delta-bridge}}(e)
$$

必须使用 true branch-delta，不得使用 source gap 或 formula proxy。

---

# Part V. Candidate designs

## 1. Basis norm internal attribution candidates

### BN0：v9.2.75 basis_norm reference

Reference:

```text
basis_norm_time_ms = 5.664183
dominant_removable_or_fusible_component_ratio = 0.676676
```

### BN1：BasisNormSubphaseTimer

将 basis_norm 拆为：

```text
norm_input_gather
norm_scale_shift
quantile_tail_compute
family_context_lookup
bucket_context_lookup
candidate_norm_writeback
audit_norm_writeback
```

### BN2：BasisNormCUDAEventAndProfiler

在 diagnostic run 中用 CUDA event / profiler 标记 basis_norm 内部 kernel names，确认是否是 topk/sort/reduction/gather 主导。

### BN3：BasisNormCounterfactualAblation

不改变 official path，只做 counterfactual timing：

```text
without_audit_norm
without_tail_context
without_quantile_context
without_candidate_writeback
```

这些不能 official，只用于定位。

## 2. Basis norm runtime optimization candidates

### BO0：OriginalBasisNormRuntime

v9.2.75 reference。

### BO1：BucketCachedBasisNorm

按 bucket/family/horizon 缓存 basis_norm。

### BO2：FamilyCachedBasisNorm

按 family_id 缓存 reliability / context norm，候选行只 lookup。

### BO3：FusedBasisNormKernel

一个 kernel 内完成 basis lift/quadratic/norm，避免 basis intermediate writeback。

### BO4：AuditNormOutOfTimedPath

controller timed path 只保留 decision-required norm；audit norm post-step subset 计算。

### BO5：T2RecurrenceBasisNorm

利用 T2/quadratic 结构，以 fused FMA 计算 normalized T2 components：

$$
\phi_0=1,
$$

$$
\phi_1=\tilde{x},
$$

$$
\phi_2=\tilde{x}^2 - c.
$$

### BO6：BasisNormBridgeScoreFusion

basis_norm 直接输出 bridge_score 所需 partials，不输出 full normalized feature tensor。

## 3. Static bucket / persistent workspace candidates

### BW0：DiagnosticBucketOnlyReference

v9.2.75 reference，expected fail。

### BW1：RuntimeFixedBucketTensor

真实 runtime bucket candidate rows，not diagnostic-only。

### BW2：PersistentBasisNormWorkspace

预分配：

```text
basis_norm_workspace
bucket_context_workspace
bridge_score_workspace
accept_bit_workspace
delta_workspace
```

要求 timed allocation 大幅下降。

### BW3：BatchMajorBucketRuntime

按 bucket size 和 family/horizon 合并候选：

```text
candidate_dim major
basis_dim minor
family/horizon metadata table
```

目标：

```text
avg_candidates_per_kernel_after >= 8
```

### BW4：CudaGraphBucketRuntime

对固定 bucket shape 尝试 CUDA graph capture。失败必须记录：

```text
dynamic_shape
pointer_mutation
workspace_alias
graph_unsafe_op
stream_semantics
unsupported_triton_kernel
```

## 4. Bridge score / accept runtime candidates

### BR0：AcceptBitOnlyReference

v9.2.75 selected-feature runtime reference。

### BR1：BridgeScoreInsideKernel

kernel 内输出：

```text
bridge_score
bad_ucb
null_ucb
support_lcb
accept_bit
```

### BR2：AcceptBitPlusCompactReasonCode

输出：

```text
accept_bit
reject_reason_code
borderline_flag
```

仅 borderline rows materialize full bridge features。

### BR3：TwoPassBorderlineBridge

Pass 1 输出 accept-bit / bridge-score；Pass 2 只对 borderline count 计算 selected features。

### BR4：AuditSubsetBridgeFeature

official timed path 不 materialize full bridge features；post-step audit subset 验证。

## 5. Integrated system candidates

### SYS0：v9.2.75 reference

Expected fail:

```text
step_ratio_q90 = 2.713295831053225
integrated_runtime_pass = 0
```

### SYS1：BN1 + BO1

basis_norm internal attribution + bucket-cached norm。

### SYS2：BO3 + BR1

fused basis_norm kernel + bridge score inside kernel。

### SYS3：BO4 + BR2

audit norm out of timed path + accept-bit compact reason。

### SYS4：BO5 + BW2 + BR1

T2 recurrence basis norm + persistent workspace + bridge score inside kernel。

### SYS5：BO6 + BW3 + BR1

basis_norm/bridge-score fusion + batch-major bucket runtime。

### SYS6：BO1 + BW3 + BR3

cached basis norm + batch-major bucket + two-pass borderline。

### SYS7：HybridBestRuntimeV9276

Best materialized runtime survivor from SYS1-SYS6. Only SYS7 can enter official P6 if all gates pass.

---

# Part VI. 实验阶段

## P0：v9.2.75 boundary reproduction

### 目标

确认 v9.2.75 boundary 稳定，避免在偶然结果上做 runtime work。

### 必须记录

```text
route
source_route_v9275
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
kernel_internal_attribution_pass
dominant_subcomponent
basis_norm_time_ms
selected_feature_runtime_pass
static_bucket_workspace_pass
basis_delta_runtime_pass
integrated_runtime_pass
system_legal_controller_pass
official_eligible
step_ratio_q90
fake_proxy_count
cpu_offload_used
```

### 判断标准

P0 pass：

```text
route = R16-StaticBucketWorkspaceStillUnimplemented
payload binding pass = 1
kernel_internal_attribution_pass = 1
selected_feature_runtime_pass = 1
static_bucket_workspace_pass = 0
integrated_runtime_pass = 0
system controller pass = 0
fake/proxy/offload = 0
```

### 可视化

```text
p0_v9275_boundary_ladder.svg
p0_decision_payload_runtime_split.svg
p0_v9274_to_v9275_progress.svg
```

---

## P1：basis_norm internal attribution

### 目标

把 `basis_norm_time_ms = 5.664183` 继续拆开。v9.2.75 已经知道 basis_norm 是 dominant，但还不知道 basis_norm 内部哪一项可修。

### 必须记录

```text
basis_norm_attribution_id
norm_input_gather_time_ms
norm_scale_shift_time_ms
quantile_tail_compute_time_ms
family_context_lookup_time_ms
bucket_context_lookup_time_ms
candidate_norm_writeback_time_ms
audit_norm_writeback_time_ms
norm_temp_allocation_time_ms
norm_kernel_launch_time_ms
norm_sync_time_ms
basis_norm_total_time_ms
basis_norm_unknown_fraction
dominant_basis_norm_subcomponent
dominant_basis_norm_component_ratio
kernel_count
sync_count
allocation_count
bytes_read
bytes_written
```

### 判断标准

P1 pass：

```text
basis_norm_internal_attribution_pass = 1
basis_norm_unknown_fraction <= 0.05
dominant_basis_norm_subcomponent identified
dominant_basis_norm_component_ratio >= 0.20
```

### 可视化

```text
p1_basis_norm_internal_waterfall.svg
p1_basis_norm_kernel_count.svg
p1_basis_norm_memory_io.svg
p1_basis_norm_component_ratio.svg
```

---

## P2：basis_norm cache / fusion runtime

### 目标

真实降低 basis_norm，不再停留在 attribution。

### 必须记录

```text
basis_norm_runtime_id
basis_norm_strategy
cache_used
cache_key
cache_hit_rate
fused_basis_norm_kernel_used
audit_norm_out_of_timed_path
t2_recurrence_used
basis_norm_time_before
basis_norm_time_after
basis_norm_time_reduction
candidate_count
distinct_cache_key_count
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
uses_source_measured_gap
uses_formula_proxy
projection_used
```

### 判断标准

P2 pass：

```text
basis_norm_runtime_pass = 1
basis_norm_time_after <= 0.50 * basis_norm_time_before
agreement_reference_accept >= 0.90
audit_agreement >= 0.99
uses_source_measured_gap = 0
uses_formula_proxy = 0
projection_used = 0
```

Intermediate improvement：

$$
StepRatio_{q90}\leq2.00.
$$

Full system candidate：

$$
StepRatio_{q90}\leq1.50.
$$

### 可视化

```text
p2_basis_norm_before_after.svg
p2_cache_hit_rate_by_family_horizon.svg
p2_basis_norm_strategy_pareto.svg
p2_agreement_vs_norm_reduction.svg
```

---

## P3：static bucket / persistent workspace integrated runtime

### 目标

把 static bucket 从 diagnostic table 接入 basis_norm + delta runtime，并启用 persistent workspace。

### 必须记录

```text
bucket_workspace_id
bucket_strategy
bucket_sizes
candidate_count
runtime_bucket_used
basis_norm_bucketed
W2_delta_bucketed
bridge_score_bucketed
persistent_workspace_used
workspace_memory_MB
kernel_count_before
kernel_count_after
sync_count_before
sync_count_after
allocation_count_before
allocation_count_after
kernel_count_reduction
sync_count_reduction
allocation_count_reduction
avg_candidates_per_kernel_before
avg_candidates_per_kernel_after
cuda_graph_attempted
cuda_graph_capture_pass
cuda_graph_failure_reason
agreement_reference_accept
cuda_vs_torch_logits_error_max
cuda_vs_torch_delta_error_max
step_ratio_q90
memory_ratio
```

### 判断标准

P3 pass：

```text
static_bucket_workspace_pass = 1
runtime_bucket_used = 1
persistent_workspace_used = 1
basis_norm_bucketed = 1
W2_delta_bucketed = 1
allocation_count_after <= 0.10 * allocation_count_before
kernel_count_after <= 0.50 * kernel_count_before
sync_count_after <= 0.50 * sync_count_before
avg_candidates_per_kernel_after >= 8
agreement_reference_accept >= 0.90
```

Intermediate improvement：

$$
StepRatio_{q90}\leq2.00.
$$

### 可视化

```text
p3_kernel_sync_allocation_reduction.svg
p3_avg_candidates_per_kernel_before_after.svg
p3_bucket_occupancy_histogram.svg
p3_workspace_memory_tradeoff.svg
p3_cuda_graph_capture_status.svg
```

---

## P4：single-pass basis-norm-delta-bridge runtime

### 目标

把 basis_norm、W2_delta、probe logits、bridge score、accept bit 合成真实 measured runtime。

### 必须记录

```text
basis_delta_bridge_runtime_id
basis_norm_runtime_id
bucket_workspace_id
uses_true_branch_delta
uses_source_measured_gap
uses_formula_proxy
basis_norm_delta_bridge_single_pass
bridge_score_inside_kernel
accept_bit_inside_kernel
bad_ucb_inside_kernel
null_ucb_inside_kernel
support_lcb_inside_kernel
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
basis_norm_time_ms
W2_delta_time_ms
bridge_score_time_ms
selected_feature_time_ms
step_ratio_q90
memory_ratio
```

### 判断标准

P4 pass：

```text
basis_norm_delta_bridge_single_pass = 1
uses_true_branch_delta = 1
uses_source_measured_gap = 0
uses_formula_proxy = 0
bridge_score_inside_kernel = 1
accept_bit_inside_kernel = 1
cuda_vs_torch_check_count >= 24
cuda_vs_torch_logits_error_max <= 5e-5
cuda_vs_torch_delta_error_max <= 1e-8
agreement_reference_accept >= 0.90
```

Intermediate improvement：

$$
StepRatio_{q90}\leq2.00.
$$

Full system candidate：

$$
StepRatio_{q90}\leq1.50.
$$

### 可视化

```text
p4_single_pass_runtime_pareto.svg
p4_error_vs_step_ratio.svg
p4_reference_agreement_confusion.svg
p4_basis_norm_delta_bridge_cost.svg
```

---

## P5：integrated materialized runtime candidates

### 目标

组合 P2/P3/P4 survivors，形成真实 integrated runtime，不允许 diagnostic-derived row 作为 best。

### 必须记录

```text
system_candidate_id
basis_norm_runtime_id
bucket_workspace_id
basis_delta_bridge_runtime_id
selected_feature_runtime_id
materialized_runtime_path
audit_only_cost_removal
diagnostic_derived_from_measured_components
candidate_count
candidate_rate
kernel_count
sync_count
allocation_count
avg_candidates_per_kernel
basis_norm_time_ms
W2_delta_time_ms
bridge_score_time_ms
selected_feature_time_ms
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

P5 pass：

```text
integrated_runtime_pass = 1
materialized_runtime_path = 1
audit_only_cost_removal = 0
diagnostic_derived_from_measured_components = 0
agreement_reference_accept >= 0.90
projection_used = 0
source_measured_gap_used = 0
formula_proxy_used = 0
dataset_name_used = 0
```

Intermediate improvement：

$$
StepRatio_{q90}\leq2.00.
$$

System survivor：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p5_integrated_runtime_pareto.svg
p5_step_ratio_reduction_ladder.svg
p5_runtime_component_before_after.svg
p5_kernel_sync_allocation_vs_step.svg
```

---

## P6：system-legal exact-signal controller v8

### 目标

只有 P5 有真实 system survivor 时，P6 才能 official pass。

### 必须记录

```text
controller_id
system_candidate_id
basis_norm_runtime_id
bucket_workspace_id
basis_delta_bridge_runtime_id
selected_feature_runtime_id
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
p6_step_ratio_progress_v9272_to_v9276.svg
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
  ShuffledBasisNormRuntime
  ShuffledStaticBucket
  ShuffledBasisDeltaBridgeKernel
  ShuffledAcceptBit
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
ShuffledBasisNormRuntime = fail
ShuffledStaticBucket = fail
ShuffledBasisDeltaBridgeKernel = fail
ShuffledAcceptBit = fail
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
  ShuffledBasisNormRuntime
  ShuffledStaticBucket
  ShuffledBasisDeltaBridgeKernel
  ShuffledAcceptBit
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
contract_audit_v9276.csv
p0_v9275_boundary_reproduction.csv
p1_basis_norm_internal_attribution.csv
p2_basis_norm_cache_fusion_runtime.csv
p3_static_bucket_persistent_workspace_integrated_runtime.csv
p4_single_pass_basis_norm_delta_bridge_runtime.csv
p5_integrated_materialized_runtime_candidates.csv
p6_system_legal_exact_signal_controller_v8.csv
p7_leave_dataset_and_stratum_out.csv
p8_official_paired_replay.csv
p9_short_run_functional_validation.csv
p10_full_run_robustness_strong_baseline.csv

basis_norm_internal_trace_v9276.csv
basis_norm_runtime_trace_v9276.csv
static_bucket_workspace_runtime_trace_v9276.csv
basis_norm_delta_bridge_runtime_trace_v9276.csv
integrated_runtime_trace_v9276.csv
system_controller_trace_v9276.csv
leaveout_trace_v9276.csv
paired_replay_branch_trace_v9276.csv
short_run_trace_v9276.csv

route_decision.json
aggregate_decision.json
failure_table.csv
artifact_hashes.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9275_boundary_unstable
F3_dataset_tuning_detected
F4_payload_binding_regression
F5_decision_metric_regression
F6_basis_norm_internal_attribution_fail
F7_basis_norm_no_actionable_component
F8_basis_norm_cache_agreement_fail
F9_basis_norm_cache_no_reduction
F10_basis_norm_fused_kernel_error_fail
F11_basis_norm_audit_removal_agreement_fail
F12_static_bucket_not_integrated
F13_persistent_workspace_not_integrated
F14_kernel_sync_allocation_unfixed
F15_avg_candidates_per_kernel_too_low
F16_bridge_score_not_inside_kernel
F17_single_pass_basis_norm_delta_bridge_not_materialized
F18_single_pass_runtime_error_fail
F19_single_pass_runtime_agreement_fail
F20_integrated_runtime_still_diagnostic
F21_integrated_runtime_step_ratio_fail
F22_integrated_runtime_memory_fail
F23_source_gap_or_formula_proxy_used
F24_system_controller_precision_fail
F25_system_controller_coverage_fail
F26_system_controller_bad_event_fail
F27_system_controller_null_rate_fail
F28_system_controller_lcb_ucb_fail
F29_system_controller_projection_used
F30_leave_dataset_out_fail
F31_leave_stratum_out_fail
F32_paired_replay_control_equivalent
F33_shuffle_control_pass
F34_functional_lr_equivalent
F35_short_run_task_drop
F36_full_run_no_macro_hard_stratum_gain
F37_strong_baseline_explains_gain
F38_robustness_fail
F39_external_not_ready
F40_fake_or_proxy_violation
F41_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-BoundaryReproduced:
  v9.2.75 boundary reproduced.

R2-BasisNormInternalAttributionPass:
  basis_norm internal attribution has unknown_fraction <= 0.05.

R3-BasisNormRuntimePass:
  basis_norm cache/fusion/runtime elimination reduces basis_norm time and preserves agreement.

R4-StaticBucketWorkspaceRuntimePass:
  static bucket and persistent workspace are integrated into basis_norm/delta runtime.

R5-SinglePassBasisNormDeltaBridgePass:
  basis_norm/delta/bridge score/accept bit are materialized in a true runtime path.

R6-IntegratedRuntimeDiagnosticPass:
  integrated materialized runtime reaches step_ratio_q90 <= 2.00.

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

R14-BasisNormAttributionIncomplete:
  basis_norm remains an internal black box.

R15-BasisNormDominantUnfixed:
  basis_norm is attributed but runtime cache/fusion fails.

R16-StaticBucketWorkspaceStillUnimplemented:
  static bucket / persistent workspace still not integrated.

R17-SinglePassBridgeRuntimeFail:
  bridge score / basis_norm / delta single-pass runtime cannot be materialized.

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
v9275_boundary_pass
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

basis_norm_internal_attribution_pass
basis_norm_unknown_fraction
dominant_basis_norm_subcomponent
norm_input_gather_time_ms
norm_scale_shift_time_ms
quantile_tail_compute_time_ms
family_context_lookup_time_ms
bucket_context_lookup_time_ms
candidate_norm_writeback_time_ms
audit_norm_writeback_time_ms

best_basis_norm_runtime_id
basis_norm_runtime_pass
basis_norm_strategy
basis_norm_cache_hit_rate
basis_norm_time_before
basis_norm_time_after
basis_norm_time_reduction

best_bucket_workspace_id
static_bucket_workspace_pass
runtime_bucket_used
persistent_workspace_used
basis_norm_bucketed
W2_delta_bucketed
kernel_count_reduction
sync_count_reduction
allocation_count_reduction
avg_candidates_per_kernel_after
cuda_graph_capture_pass
cuda_graph_failure_reason

best_basis_delta_bridge_runtime_id
basis_norm_delta_bridge_single_pass
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
audit_only_cost_removal
diagnostic_derived_from_measured_components

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
success_v9276_strict_purekan_functional
success_v9276_full_functional
success_v9276_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 basis_norm internal attribution
  P2 basis_norm cache / fusion runtime
  P3 static bucket / persistent workspace integrated runtime
  P4 single-pass basis_norm-delta-bridge runtime
  P5 integrated runtime candidates

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
P2/P3/P4 can run in parallel after P0, but P5/P6 cannot pass unless P1 attribution pass.
P6 cannot pass unless:
  P0 pass
  P1 basis_norm internal attribution pass
  P2 or P4 basis_norm runtime pass
  P3 static bucket workspace pass
  P5 integrated runtime pass
  step_ratio_q90 <= 1.50
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
v9.2.75 boundary reproduced
basis_norm internal attribution measured
basis_norm cache/fusion runtime measured
static bucket/workspace runtime measured
single-pass basis_norm-delta-bridge measured
integrated runtime measured
system controller measured
no fake/proxy/offload/loss/teacher violation
```

## Attribution success

```text
Minimum diagnostic success
+
basis_norm_internal_attribution_pass = 1
+
basis_norm_unknown_fraction <= 0.05
+
dominant_basis_norm_subcomponent identified
+
dominant component ratio >= 0.20
```

## Runtime materialization success

```text
Attribution success
+
at least one of:
  basis_norm runtime pass
  static bucket workspace pass
  single-pass basis_norm-delta-bridge pass
+
materialized_runtime_path = 1
+
audit_only_cost_removal = 0
+
diagnostic_derived_from_measured_components = 0
```

## Integrated runtime success

```text
Runtime materialization success
+
integrated_runtime_pass = 1
+
step_ratio_q90 <= 2.00
+
agreement_reference_accept >= 0.90
+
payload binding remains pass
```

## System success

```text
Integrated runtime success
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
1. v9.2.75 boundary cannot be reproduced；
2. payload binding regresses；
3. decision metrics regress；
4. basis_norm internal attribution fails；
5. basis_norm has no actionable subcomponent；
6. basis_norm cache/fusion reduces cost but breaks agreement；
7. audit norm removal breaks reference agreement；
8. static bucket remains diagnostic-only；
9. persistent workspace not integrated；
10. kernel/sync/allocation count does not decrease；
11. avg candidates per kernel remains <2；
12. bridge score cannot be moved inside kernel；
13. single-pass basis_norm-delta-bridge not materialized；
14. low-cost path uses source gap / formula proxy；
15. integrated runtime remains diagnostic-derived；
16. system step_ratio_q90 remains >1.50；
17. memory ratio >1.05；
18. system controller fails precision / coverage / bad-event / null-rate；
19. precision LCB below 0.75；
20. bad-event UCB above 0.05；
21. leave-dataset-out fails；
22. leave-stratum-out fails；
23. paired replay remains control-equivalent；
24. shuffle controls pass；
25. short-run task drops；
26. full run gives no macro / hard-stratum / geometry gain；
27. functional breaks system gate；
28. gains are explained by QuadraticFeatureMLP；
29. any teacher/loss/fake/proxy/offload/projection-as-pass violation occurs。
```

---

# Part XI. 最终解释规则

## Case A：P6 system pass + P7/P8 pass

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full run and robustness。

## Case B：basis_norm runtime pass but system still slow

必须声明：

```text
basis_norm was a real dominant cost and has been reduced, but remaining blocker moved to static bucket / W2_delta / bridge score / launch.
```

下一步基于 P5 component 做 targeted kernel work，不调 controller。

## Case C：basis_norm cache/fusion breaks agreement

必须声明：

```text
basis_norm fields are decision-critical under current C3 controller.
```

下一步不能删 norm，只能做 exact fused norm or two-pass borderline norm。

## Case D：static bucket/workspace pass but system still slow

必须声明：

```text
fragmentation improved, but arithmetic or basis_norm/delta memory bandwidth remains dominant.
```

下一步做 T2 recurrence / memory coalescing / arithmetic simplification。

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

v9.2.76 的一句话策略是：

$$
\boxed{
\text{别再泛泛修 runtime；现在主攻 basis_norm，把 basis_norm cache/fusion 和 static bucket/workspace 接成 integrated measured path。}
}
$$

当前最关键的问题不是：

```text
oracle 是否存在；
reference frontier 是否 deployable；
PF5 candidate rate 是否可行；
payload 是否 missing；
selected-feature 是否 missing；
C3/T2/C4/E2 controller 是否要重调；
safe-useful target 是否要重设；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. basis_norm_time_ms = 5.664183 内部到底哪个 subcomponent 慢？
2. quantile / tail / context 是否是 controller 必需，还是 audit 必需？
3. basis_norm 能否按 family/horizon/bucket 缓存？
4. T2 recurrence 能否减少 basis_norm writeback？
5. basis_norm 能否直接输出 bridge-score partials？
6. bridge score 能否进入 kernel，而不是只有 accept-bit 进入 kernel？
7. static bucket 是否能接入 basis_norm + W2_delta runtime？
8. persistent workspace 是否能真实降低 allocation_count？
9. avg_candidates_per_kernel 能否从 0.6648 拉到 >=8？
10. integrated runtime 是否能从 diagnostic-derived 变成 materialized？
11. step q90 能否从 2.713296 降到 <=2.00，再降到 <=1.50？
12. system controller 是否 official eligible？
13. LDO/LSO 是否通过？
14. official paired replay 是否打过 AdamWParallel / bestLR？
```

v9.2.76 的结果将给出清晰分叉：

```text
if basis_norm runtime passes and integrated system passes:
  open system-legal controller, LDO/LSO, paired replay.

if basis_norm runtime passes but system remains >1.50:
  continue static bucket / bridge-score single-pass / W2_delta fusion.

if basis_norm cannot be reduced without losing agreement:
  treat basis_norm as decision-critical and implement exact fused norm, not cache approximation.

if static bucket/workspace remains diagnostic-only:
  stop running P6/P7; implement runtime bucket first.

if compute passes but paired replay fails:
  system is legal, but functional causal advantage is insufficient.

if LDO/LSO fails:
  repair dataset-agnostic support/family reliability, not dataset-specific tuning.
```
