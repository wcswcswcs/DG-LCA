# DG-KAN v9.2.77 Quantile-Tail BasisNorm Runtime Closure 与 Bucketed Single-Pass Bridge Controller 完整实验计划

> 本计划基于 v9.2.76 `BasisNorm Runtime Elimination 与 Static-Bucket Workspace Closure` 的真实执行结果制定。  
> v9.2.76 的 terminal route 是：
>
> ```text
> route = R15-BasisNormDominantUnfixed
> base_candidate = LQ-t2-h256
> success_v9276_strict_purekan_functional = False
> success_v9276_full_functional = False
> success_v9276_external_ready = False
> ```
>
> v9.2.76 的关键事实是：
>
> ```text
> P0 boundary:
>   source route = R16-StaticBucketWorkspaceStillUnimplemented
>   payload_binding_contract_pass = 1
>   candidate_tensor_payload_missing_count = 0
>   candidate_branch_logits_missing_count = 0
>   candidate_true_delta_logits_missing_count = 0
>   functional_update_payload_missing_count = 0
>   system_legal_controller_pass = 0
>
> P1 basis_norm internal attribution:
>   basis_norm_internal_attribution_pass = 1
>   basis_norm_total_time_ms = 0.7972274324856699
>   basis_norm_unknown_fraction = 0.0
>   dominant_basis_norm_subcomponent = quantile_tail_compute_time_ms
>   dominant_basis_norm_component_ratio = 0.466308
>   quantile_tail_compute_time_ms = 0.371753276946644
>   norm_scale_shift_time_ms = 0.1534663800460597
>   candidate_norm_writeback_time_ms = 0.11656367375204961
>   norm_dispatch_overhead_time_ms = 0.050312
>   audit_norm_writeback_time_ms = 0.034132
>   norm_kernel_launch_time_ms = 0.020302
>   family_context_lookup_time_ms = 0.015874
>   norm_temp_allocation_time_ms = 0.014208
>   bucket_context_lookup_time_ms = 0.011043
>   norm_input_gather_time_ms = 0.005260
>   norm_sync_time_ms = 0.004311
>
> P2 basis_norm runtime:
>   best = BO3-NoCloneLogsumexpTop2BasisNormRuntime
>   basis_norm_strategy = no_clone_logsumexp_top2_quantile_exact_tail
>   cache_used = 0
>   fused_basis_norm_kernel_used = 0
>   basis_norm_time_before = 0.7972274324856699
>   basis_norm_time_after = 0.5810023285448551
>   basis_norm_time_reduction = 0.27122135432124805
>   agreement_reference_accept = 1.0
>   audit_agreement = 1.0
>   cuda_vs_torch_check_count = 24
>   cuda_vs_torch_logits_error_max = 4.76837158203125e-07
>   cuda_vs_torch_delta_error_max = 3.4924596548080444e-10
>   basis_norm_runtime_pass = 0
>
> P3 static bucket / persistent workspace:
>   best = BW2-PersistentBasisNormWorkspaceMaterializedNotIntegrated
>   runtime_bucket_used = 1
>   persistent_workspace_used = 1
>   workspace_memory_MB = 0.2988319396972656
>   allocation_count_before = 2493
>   allocation_count_after = 1
>   allocation_count_reduction = 0.9995988768551946
>   kernel_count_before = 3750
>   kernel_count_after = 3750
>   sync_count_before = 1250
>   sync_count_after = 1250
>   kernel_count_reduction = 0.0
>   sync_count_reduction = 0.0
>   avg_candidates_per_kernel_after = 0.6648
>   basis_norm_bucketed = 0
>   W2_delta_bucketed = 0
>   static_bucket_workspace_pass = 0
>
> P4/P5/P6 boundary:
>   basis_norm_delta_bridge_single_pass = 0
>   bridge_score_inside_kernel = 0
>   accept_bit_inside_kernel = 1
>   materialized_runtime_path = 0
>   diagnostic_derived_from_measured_components = 1
>   integrated_runtime_pass = 0
>   official_eligible = 0
>   controller_step_ratio_q90 = 2.713295831053225
> ```
>
> v9.2.77 的核心判断是：
>
> $$
> \boxed{
> \text{v9.2.76 已经定位 basis_norm 主成本到 quantile/tail compute，但没有形成可 official 的 integrated runtime。}
> }
> $$
>
> 因此 v9.2.77 不应继续泛泛地“优化 basis_norm”，也不应继续写 diagnostic bucket / diagnostic workspace。  
> 本轮必须围绕三个真实实现目标并行推进：
>
> ```text
> 1. quantile/tail compute exact runtime reduction；
> 2. bucket/family/horizon cache 接入 basis_norm 和 W2_delta；
> 3. bridge_score_inside_kernel + single-pass basis_norm-delta-bridge integrated runtime。
> ```
>
> 最终目标是：
>
> $$
> \boxed{
> \text{把 controller step ratio q90 从 }2.713296\text{ 压到 }\leq1.50，
> \text{且保持 reference agreement 与 decision gates。}
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

v9.2.77 新增硬约束：

```text
quantile_tail runtime 不能只是 attribution；
basis_norm runtime pass 不能只看 local same-run timing，必须进入 P5 integrated runtime；
static bucket / persistent workspace 必须实际降低 kernel/sync/allocation 或 avg_candidates_per_kernel；
basis_norm_bucketed 必须为 1 才能声明 bucket runtime pass；
W2_delta_bucketed 必须为 1 才能声明 static-bucket integrated pass；
bridge_score_inside_kernel 必须为 1 才能声明 single-pass bridge runtime pass；
materialized_runtime_path 必须为 1；
diagnostic_derived_from_measured_components 必须为 0；
audit_only_cost_removal 必须为 0；
projection_used 必须为 0；
source_measured_gap_used 必须为 0；
formula_proxy_used 必须为 0；
dataset_name_used 必须为 0；
official pass 必须来自真实 measured runtime path。
```

允许按 dataset / family / horizon 诊断问题：

```text
per-dataset basis_norm_time
per-family quantile_tail_time
per-horizon quantile_tail_time
per-bucket cache_hit_rate
per-bucket occupancy
per-stratum agreement drift
per-dataset step_ratio
```

但 official route 必须 dataset-agnostic：

```text
same PF5 selector
same frozen C3-T2PlusBackfill controller
same quantile-tail runtime
same static bucket strategy
same basis_norm cache key schema
same bridge-score kernel
same pass thresholds
```

---

# Part I. 对 v9.2.76 的独立判断

## 1. v9.2.76 没有达到目标

v9.2.76 没有 strict PureKAN functional success，也没有 system-legal controller success。失败不是因为 controller decision metrics 坏了，也不是 payload binding 回退，而是：

```text
basis_norm_runtime_pass = 0
static_bucket_workspace_pass = 0
integrated_runtime_pass = 0
system_legal_controller_pass = 0
official_eligible = 0
controller_step_ratio_q90 = 2.713295831053225
```

P7 leave-dataset-out / leave-stratum-out、P8 official paired replay、P9 short-run、P10 full-run 均因 P2/P5/P6 gate 未过而不得打开。这个 gate 是正确的。不能把 P1 attribution pass、P2 local reduction 或 P3 workspace materialization 写成 system success。

## 2. v9.2.76 的真实进展

v9.2.76 的真实进展有三点。

第一，basis_norm 内部 attribution 闭合。v9.2.75 只知道 basis_norm 是 dominant；v9.2.76 进一步证明 dominant subcomponent 是：

```text
quantile_tail_compute_time_ms = 0.371753
component ratio = 0.466308
```

这把主 blocker 从 “basis_norm 很慢” 缩小到 “quantile/tail/context construction 很慢”。

第二，no-clone/logsumexp/top2 path 是数值正确的。它没有使用 source gap 或 formula proxy，agreement = `1.0`，audit agreement = `1.0`，CUDA-vs-torch logits/delta error 很小。说明这个方向不是错误实现。

第三，persistent workspace 不再只是空 flag。P3 把 allocation count 从 `2493` 降到 `1`，workspace memory 只有约 `0.298832 MB`。这说明 workspace materialization 是真实的。

## 3. v9.2.76 的真实失败

v9.2.76 的失败也非常明确。

第一，P2 的 no-clone/logsumexp/top2 只把 basis_norm local time 从 `0.797227 ms` 降到 `0.581002 ms`，降幅为 `27.12%`，没有达到计划中至少 `50%` 的 basis_norm runtime reduction。这个失败不是数值问题，而是优化力度不够。

第二，P3 persistent workspace 没有接入 basis_norm / W2_delta runtime。虽然 allocation count 几乎清零，但：

```text
basis_norm_bucketed = 0
W2_delta_bucketed = 0
kernel_count_reduction = 0.0
sync_count_reduction = 0.0
avg_candidates_per_kernel_after = 0.6648
```

这意味着 workspace 只是被 materialize，没有改变主要执行图。

第三，P4/P5 没有 integrated survivor：

```text
basis_norm_delta_bridge_single_pass = 0
bridge_score_inside_kernel = 0
materialized_runtime_path = 0
diagnostic_derived_from_measured_components = 1
```

所以 P6 不能 official。

## 4. 当前 blocker 的本质

当前 blocker 应写成：

$$
\boxed{
\text{quantile/tail basis_norm 是主成本，但 runtime reduction 和 bucketed integrated execution 都没有闭合。}
}
$$

更具体地说：

```text
1. decision frontier 已稳定；
2. payload binding 已稳定；
3. selected-feature runtime 已局部解决；
4. basis_norm 内部 dominant 已定位；
5. no-clone exact-tail 只降 27.12%，不足；
6. persistent workspace 只解决 allocation，不解决 basis_norm / W2_delta 执行；
7. bridge_score 仍不在 kernel；
8. integrated runtime 仍不是 materialized path。
```

所以 v9.2.77 不能继续泛泛尝试 “basis_norm optimization”。必须直接攻击 quantile/tail compute、cache key 设计、bucketed runtime、bridge-score kernel 化。

## 5. 是否还在正确道路上

是。当前路线已经从 controller science 进入 system runtime closure 的最后阶段：

```text
reference controller solved
payload binding solved
basis_norm attribution solved
selected-feature runtime solved
current blocker = quantile/tail basis_norm + static bucket integration + bridge score kernelization
```

不应回去做：

```text
C3/T2/C4/E2 threshold tuning
safe-useful target reset
support row expansion
dataset-specific route
controller re-search
selected-feature-only optimization
allocation-only workspace optimization
```

---

# Part II. v9.2.77 总体目标

v9.2.77 的总体目标是：

$$
\boxed{
\text{把 quantile/tail basis_norm 从 dominant runtime 变成 bucketed / cached / fused runtime，并形成 official-eligible integrated controller。}
}
$$

强目标：

$$
StepRatio_{q90}\leq1.50.
$$

最低有效推进目标：

$$
StepRatio_{q90}\leq2.00
$$

并且必须满足：

```text
materialized_runtime_path = 1
integrated_runtime_pass = 1
diagnostic_derived_from_measured_components = 0
audit_only_cost_removal = 0
agreement_reference_accept >= 0.90
payload binding remains pass
```

本轮必须回答十个问题：

```text
Q1:
  v9.2.76 boundary 是否稳定复现？

Q2:
  quantile_tail_compute_time_ms 内部到底是 topk/sort、tail mask、logsumexp、threshold extraction，
  还是 context lookup / writeback 主导？

Q3:
  quantile/tail 是否需要每 candidate 重新算？
  distinct cache key 数量是否远小于 candidate_count = 2493？

Q4:
  quantile/tail 是否可以按 family/horizon/bucket/role 缓存？
  cache hit rate 能否 >= 0.50？

Q5:
  top2/logsumexp path 为什么只能降 27.12%？
  剩余 72.88% 到底来自 exact tail 还是 writeback / dispatch？

Q6:
  persistent workspace 能否接入 basis_norm_bucketed = 1 和 W2_delta_bucketed = 1？

Q7:
  avg_candidates_per_kernel 能否从 0.6648 提升到 >=8？

Q8:
  bridge_score 能否进入 kernel？
  不能只让 accept_bit_inside_kernel = 1。

Q9:
  integrated materialized runtime 能否达到 step q90 <=2.00 或 <=1.50？

Q10:
  system pass 后，LDO/LSO 与 official paired replay 是否能打开？
```

---

# Part III. 核心假设

## H1：quantile/tail compute 是可拆解、可融合、可缓存的主成本

H1 成立标准：

```text
quantile_tail_internal_attribution_pass = 1
quantile_tail_unknown_fraction <= 0.05
dominant_quantile_tail_subcomponent identified
at least one removable/cacheable/fusible subcomponent ratio >= 0.20
```

必须记录：

```text
topk_time_ms
sort_time_ms
tail_mask_time_ms
logsumexp_time_ms
threshold_compute_time_ms
tail_context_lookup_time_ms
tail_writeback_time_ms
tail_temp_allocation_time_ms
tail_kernel_launch_time_ms
tail_sync_time_ms
```

H1 失败标准：

quantile_tail 仍只是 aggregate block，不能定位可修点。

## H2：quantile/tail cache 可以显著减少 runtime，不破坏 agreement

H2 认为 quantile/tail 的 context 在 family / horizon / bucket / role 层重复，不必逐 candidate 重算。

H2 成立标准：

```text
quantile_tail_cache_used = 1
cache_key_schema recorded
cache_hit_rate >= 0.50
quantile_tail_time_after <= 0.50 * quantile_tail_time_before
agreement_reference_accept >= 0.95
audit_agreement >= 0.99
```

H2 失败标准：

cache hit rate <0.20，或 cache 后 agreement <0.90。

## H3：exact top-k / streaming-tail kernel 比 no-clone/logsumexp/top2 更有希望

v9.2.76 的 BO3 是：

```text
no_clone_logsumexp_top2_quantile_exact_tail
```

但只降了 `27.12%`。H3 认为需要更底层的 top-k / tail-mask fused kernel，而不是 Python/Torch no-clone 重排。

H3 成立标准：

```text
fused_topk_tail_kernel_used = 1
quantile_tail_time_after <= 0.40 * before
cuda_vs_torch_check_count >= 24
logits_error <= 5e-5
delta_error <= 1e-8
agreement_reference_accept >= 0.90
```

H3 失败标准：

fused top-k/tail kernel 正确但仍不能降到 0.50 以下。此时需要考虑 cache 或 approximation-with-exact-audit two-pass，不允许直接用 proxy official。

## H4：basis_norm_bucketed 和 W2_delta_bucketed 必须一起接入 runtime

v9.2.76 allocation reduction 已经显示 workspace 可以 materialize，但它没有接入 basis_norm/W2_delta。H4 认为只有 bucketed execution 接入主 compute，kernel/sync 才会下降。

H4 成立标准：

```text
runtime_bucket_used = 1
persistent_workspace_used = 1
basis_norm_bucketed = 1
W2_delta_bucketed = 1
bridge_score_bucketed = 1
kernel_count_after <= 0.50 * before
sync_count_after <= 0.50 * before
allocation_count_after <= 0.10 * before
avg_candidates_per_kernel_after >= 8
```

H4 失败标准：

allocation 降了，但 kernel/sync/avg_candidates_per_kernel 不变。

## H5：bridge_score_inside_kernel 是 integrated runtime 的必要条件

v9.2.76 仍是：

```text
bridge_score_inside_kernel = 0
accept_bit_inside_kernel = 1
```

H5 认为这不足以 official，因为 bridge score / bad UCB / null UCB / support LCB 仍在 kernel 外 materialize 或 lookup。

H5 成立标准：

```text
bridge_score_inside_kernel = 1
bad_ucb_inside_kernel or compact_lookup_inside_kernel = 1
null_ucb_inside_kernel or compact_lookup_inside_kernel = 1
support_lcb_inside_kernel or compact_lookup_inside_kernel = 1
agreement_reference_accept >= 0.95
```

H5 失败标准：

accept bit 可以进 kernel，但 bridge score 不能进 kernel，P5 仍 diagnostic-derived。

## H6：如果 exact quantile/tail 不能降成本，必须转向 cheap sufficient statistics，但不能按 dataset 调参

H6 是失败分叉假设。若 exact quantile/tail cache/fusion 全部不能把 step q90 压入 envelope，则下一步不应再继续小修 exact path，而应设计：

```text
dataset-agnostic cheap sufficient statistics
two-pass borderline exact confirmation
agreement-certified quantile-tail surrogate
observable primitive extraction
```

H6 成立标准：

```text
P1 attribution complete
P2/P3/P4 materialized
agreement maintained
step_ratio_q90 still > 1.50
```

此时才能讨论 architecture / primitive pivot。若 P2/P3/P4 没 materialize，不允许声称 lower bound 已到。

---

# Part IV. Runtime architecture

v9.2.77 目标 runtime：

```text
PF5 selector
→ static bucket runtime table
→ persistent candidate/basis/tail/delta/bridge workspace
→ quantile-tail cache/fused kernel
→ basis_norm fused kernel
→ W2_delta/probe_logits fused kernel
→ bridge_score + accept_bit inside kernel
→ accepted update payload
→ post-step audit subset
```

形式化：

$$
Accept_{\text{sys}}(e)
=
PF5(e)
\land
K_{\text{qt-basisnorm-delta-bridge}}(e)
\land
FrozenC3(e).
$$

其中 $K_{\text{qt-basisnorm-delta-bridge}}$ 必须使用 true branch-delta，不得使用 source-measured gap 或 formula proxy。

Cost model：

$$
T_{\text{sys}}
=
T_{\text{PF5}}
+
T_{\text{bucket}}
+
T_{\text{qt-cache}}
+
T_{\text{qt-tail}}
+
T_{\text{basis-norm}}
+
T_{\text{W2-delta}}
+
T_{\text{probe-logits}}
+
T_{\text{bridge-score}}
+
T_{\text{accept-bit}}
+
T_{\text{update-payload}}
+
T_{\text{launch/sync/allocation}}.
$$

官方 gate：

$$
\frac{T_{\text{sys,q90}}}{T_{\text{MLP,q90}}}\leq1.50.
$$

---

# Part V. Candidate designs

## 1. Quantile-tail attribution candidates

### QTA0：v9.2.76 quantile-tail reference

Reference:

```text
quantile_tail_compute_time_ms = 0.371753
component_ratio = 0.466308
```

### QTA1：QuantileTailSubphaseTimer

记录：

```text
topk_time_ms
sort_time_ms
tail_mask_time_ms
logsumexp_time_ms
threshold_time_ms
tail_context_lookup_time_ms
tail_writeback_time_ms
tail_launch_time_ms
tail_sync_time_ms
```

### QTA2：ProfilerMarkedQuantileTail

只用于 diagnostic run；用 profiler/NVTX 标记 kernel names，不进入 official pass。

### QTA3：QuantileTailCounterfactualAblation

不作为 official，只拆：

```text
without_tail_mask
without_logsumexp
without_top2
without_tail_context
without_tail_writeback
```

用于判断哪个部分值得 fusion。

## 2. Quantile-tail runtime candidates

### QTR0：BO3 reference

v9.2.76 no-clone/logsumexp/top2 path。

### QTR1：BucketCachedQuantileTail

按：

```text
family_id
horizon
bucket_id
role_id
norm_context_id
```

缓存 quantile/tail context。

### QTR2：FamilyHorizonCachedTail

按 family/horizon 缓存 tail context，candidate 只做 lookup。

### QTR3：FusedTopKExactTailKernel

一个 kernel 中完成：

```text
top-k / top2
tail mask
threshold extraction
tail statistic
```

### QTR4：StreamingTailMaskKernel

streaming 扫描，不 materialize full tail mask。

### QTR5：TwoPassBorderlineExactTail

Pass 1 cheap tail score / bridge score；Pass 2 只对 borderline rows 做 full exact tail。

### QTR6：AuditTailOutOfTimedPath

只允许 audit-required tail summary post-step subset；decision-required tail 必须保留。

## 3. Basis_norm runtime candidates

### BNR0：BO3 no-clone reference

Reference.

### BNR1：QuantileTailCachedBasisNorm

接入 QTR1/QTR2 的 basis_norm。

### BNR2：FusedBasisNormTailKernel

basis lift / scale-shift / quantile-tail / candidate norm writeback 一次完成。

### BNR3：BasisNormBridgePartialFusion

basis_norm 不输出 full normalized tensor，只输出 bridge-score partials。

### BNR4：T2RecurrenceNormTail

利用 T2 recurrence 计算 normalized basis：

$$
\phi_0=1,
$$

$$
\phi_1=\tilde{x},
$$

$$
\phi_2=\tilde{x}^2-c.
$$

### BNR5：NormContextCompactLookup

把 family/bucket/tail context 放入 compact lookup table，避免 repeated context construction。

## 4. Static bucket / workspace candidates

### SB0：v9.2.76 workspace reference

Allocation reduced but not integrated.

### SB1：BucketedBasisNormRuntime

让 basis_norm_bucketed = 1。

### SB2：BucketedW2DeltaRuntime

让 W2_delta_bucketed = 1。

### SB3：BucketedBridgeScoreRuntime

让 bridge_score_bucketed = 1。

### SB4：BatchMajorBucketRuntimeV2

目标：

```text
avg_candidates_per_kernel_after >= 8
```

### SB5：PersistentWorkspaceIntegratedV2

预分配：

```text
candidate_workspace
quantile_tail_workspace
basis_norm_workspace
delta_workspace
bridge_score_workspace
accept_bit_workspace
update_payload_workspace
```

timed path allocation 接近 0，并且 kernel/sync 也下降。

### SB6：CudaGraphBucketRuntimeV2

对固定 bucket shape graph capture。失败必须记录明确原因：

```text
dynamic_shape
pointer_mutation
workspace_alias
graph_unsafe_op
stream_semantics
unsupported_triton_kernel
```

## 5. Bridge / accept candidates

### BR0：accept-bit-only reference

v9.2.76 current.

### BR1：BridgeScoreInsideKernelV2

输出：

```text
bridge_score
bad_ucb
null_ucb
support_lcb
accept_bit
```

### BR2：CompactBridgeLookupInsideKernel

从 compact lookup table 读取 bad/null/support，kernel 内生成 bridge score。

### BR3：TwoPassBorderlineBridgeScore

只对 borderline rows materialize full bridge features。

### BR4：AuditSubsetBridgeFeature

post-step audit subset 计算 full bridge features，official timed path 不 materialize full features。

## 6. Integrated system candidates

### SYS0：v9.2.76 reference

Expected fail.

### SYS1：QTR1 + BNR1

Quantile-tail cached basis_norm, no new bucket runtime.

### SYS2：QTR3 + BNR2

Fused exact top-k/tail + fused basis_norm.

### SYS3：QTR1 + SB1 + SB2

Cached tail + bucketed basis_norm + bucketed W2_delta.

### SYS4：QTR3 + SB4 + BR1

Fused tail + batch-major bucket + bridge score inside kernel.

### SYS5：BNR3 + SB5 + BR2

basis_norm bridge partial + persistent integrated workspace + compact bridge lookup.

### SYS6：QTR5 + BR3 + SB4

two-pass borderline exact tail + two-pass bridge + batch-major bucket.

### SYS7：HybridBestRuntimeV9277

Best materialized survivor from SYS1-SYS6. Only SYS7 can be promoted to P6 if all audits pass.

---

# Part VI. 实验阶段

## P0：v9.2.76 boundary reproduction

### 目标

确认 v9.2.76 boundary 稳定。

### 必须记录

```text
route
source_route_v9276
payload_binding_contract_pass
candidate_tensor_payload_missing_count
candidate_branch_logits_missing_count
candidate_true_delta_logits_missing_count
functional_update_payload_missing_count
basis_norm_internal_attribution_pass
dominant_basis_norm_subcomponent
quantile_tail_compute_time_ms
basis_norm_runtime_pass
basis_norm_time_before
basis_norm_time_after
basis_norm_time_reduction
runtime_bucket_used
persistent_workspace_used
basis_norm_bucketed
W2_delta_bucketed
static_bucket_workspace_pass
integrated_runtime_pass
system_legal_controller_pass
official_eligible
controller_step_ratio_q90
fake_proxy_count
cpu_offload_used
```

### 判断标准

P0 pass：

```text
route = R15-BasisNormDominantUnfixed
payload binding pass = 1
basis_norm internal attribution pass = 1
basis_norm_runtime_pass = 0
static_bucket_workspace_pass = 0
integrated_runtime_pass = 0
system_controller_pass = 0
fake/proxy/offload = 0
```

### 可视化

```text
p0_v9276_boundary_ladder.svg
p0_basis_norm_runtime_status.svg
p0_step_ratio_history_v9272_v9276.svg
```

---

## P1：quantile-tail internal attribution

### 目标

把 quantile_tail_compute_time_ms 继续拆开，不再停留在 subcomponent aggregate。

### 必须记录

```text
quantile_tail_attribution_id
topk_time_ms
sort_time_ms
tail_mask_time_ms
logsumexp_time_ms
threshold_compute_time_ms
tail_context_lookup_time_ms
tail_writeback_time_ms
tail_temp_allocation_time_ms
tail_kernel_launch_time_ms
tail_sync_time_ms
quantile_tail_total_time_ms
quantile_tail_unknown_fraction
dominant_quantile_tail_subcomponent
dominant_quantile_tail_component_ratio
kernel_count
sync_count
allocation_count
bytes_read
bytes_written
```

### 判断标准

P1 pass：

```text
quantile_tail_internal_attribution_pass = 1
quantile_tail_unknown_fraction <= 0.05
dominant_quantile_tail_subcomponent identified
dominant_quantile_tail_component_ratio >= 0.20
```

### 可视化

```text
p1_quantile_tail_internal_waterfall.svg
p1_tail_kernel_count.svg
p1_tail_memory_io.svg
p1_tail_component_ratio.svg
```

---

## P2：quantile-tail cache / exact-tail fusion runtime

### 目标

真实降低 quantile_tail compute，而不是只做 no-clone local improvement。

### 必须记录

```text
quantile_tail_runtime_id
quantile_tail_strategy
cache_used
cache_key_schema
cache_hit_rate
fused_topk_tail_kernel_used
streaming_tail_kernel_used
two_pass_borderline_used
borderline_count
quantile_tail_time_before
quantile_tail_time_after
quantile_tail_time_reduction
candidate_count
distinct_cache_key_count
agreement_reference_accept
audit_agreement
cuda_vs_torch_check_count
cuda_vs_torch_logits_error_max
cuda_vs_torch_delta_error_max
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
quantile_tail_runtime_pass = 1
quantile_tail_time_after <= 0.50 * quantile_tail_time_before
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

Full candidate：

$$
StepRatio_{q90}\leq1.50.
$$

### 可视化

```text
p2_quantile_tail_before_after.svg
p2_cache_hit_rate_by_family_horizon_bucket.svg
p2_tail_runtime_strategy_pareto.svg
p2_agreement_vs_tail_reduction.svg
```

---

## P3：basis_norm runtime integration

### 目标

把 P2 的 quantile-tail reduction 接入 basis_norm，而不是只测 tail kernel。

### 必须记录

```text
basis_norm_runtime_id
quantile_tail_runtime_id
basis_norm_strategy
basis_norm_time_before
basis_norm_time_after
basis_norm_time_reduction
norm_scale_shift_time_ms
candidate_norm_writeback_time_ms
audit_norm_writeback_time_ms
basis_norm_bucketed
basis_norm_cache_used
basis_norm_fused_kernel_used
basis_norm_bridge_partial_used
agreement_reference_accept
audit_agreement
precision
coverage
bad_event
null_rate
step_ratio_q90
memory_ratio
```

### 判断标准

P3 pass：

```text
basis_norm_runtime_pass = 1
basis_norm_time_after <= 0.50 * basis_norm_time_before
basis_norm_bucketed or basis_norm_fused_kernel_used = 1
agreement_reference_accept >= 0.90
audit_agreement >= 0.99
```

Intermediate improvement：

$$
StepRatio_{q90}\leq2.00.
$$

### 可视化

```text
p3_basis_norm_runtime_before_after.svg
p3_basis_norm_component_reduction.svg
p3_basis_norm_strategy_pareto.svg
```

---

## P4：static bucket / persistent workspace integrated runtime

### 目标

让 bucket/workspace 真正接入 basis_norm / W2_delta / bridge score runtime，降低 kernel/sync/allocation 和提升 candidates-per-kernel。

### 必须记录

```text
bucket_workspace_id
bucket_strategy
bucket_sizes
candidate_count
runtime_bucket_used
persistent_workspace_used
workspace_memory_MB
basis_norm_bucketed
W2_delta_bucketed
bridge_score_bucketed
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

P4 pass：

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
p4_kernel_sync_allocation_reduction.svg
p4_avg_candidates_per_kernel_before_after.svg
p4_bucket_occupancy_histogram.svg
p4_workspace_memory_tradeoff.svg
p4_cuda_graph_capture_status.svg
```

---

## P5：single-pass quantile-tail basis_norm delta bridge runtime

### 目标

把 quantile-tail、basis_norm、W2_delta、probe logits、bridge score、accept bit 合成真实 measured integrated runtime。

### 必须记录

```text
basis_delta_bridge_runtime_id
quantile_tail_runtime_id
basis_norm_runtime_id
bucket_workspace_id
uses_true_branch_delta
uses_source_measured_gap
uses_formula_proxy
quantile_tail_inside_kernel
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
quantile_tail_time_ms
basis_norm_time_ms
W2_delta_time_ms
bridge_score_time_ms
selected_feature_time_ms
step_ratio_q90
memory_ratio
```

### 判断标准

P5 pass：

```text
basis_norm_delta_bridge_single_pass = 1
quantile_tail_inside_kernel = 1
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
p5_single_pass_runtime_pareto.svg
p5_error_vs_step_ratio.svg
p5_reference_agreement_confusion.svg
p5_runtime_component_breakdown.svg
```

---

## P6：integrated materialized runtime candidates

### 目标

组合 P2-P5 survivors，形成真实 integrated runtime。禁止 diagnostic-derived row 进入 best route。

### 必须记录

```text
system_candidate_id
quantile_tail_runtime_id
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
quantile_tail_time_ms
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

P6 pass：

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
p6_integrated_runtime_pareto.svg
p6_step_ratio_reduction_ladder.svg
p6_runtime_component_before_after.svg
p6_kernel_sync_allocation_vs_step.svg
```

---

## P7：system-legal exact-signal controller v9

### 目标

只有 P6 有真实 system survivor 时，P7 才能 official pass。

### 必须记录

```text
controller_id
system_candidate_id
quantile_tail_runtime_id
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

P7 pass：

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
p7_system_controller_cost_quality_frontier.svg
p7_reference_vs_system_accept_overlap.svg
p7_step_ratio_progress_v9272_to_v9277.svg
p7_family_strata_balance.svg
```

---

## P8：leave-dataset-out / leave-stratum-out

### 目标

只有 P7 pass 后打开。证明 system controller 不是 pooled calibration artifact，也不是 dataset-specific route。

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
p8_leave_dataset_out_matrix.svg
p8_leave_stratum_out_matrix.svg
p8_dataset_tuning_audit.svg
p8_leaveout_failure_modes.svg
```

---

## P9：official paired replay

### 目标

验证 RealFunctional 是否在 strong controls 下有局部因果优势。

### 设置

```text
base = R2 repaired base checkpoint
controller_id = best P8 survivor
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
  ShuffledQuantileTailRuntime
  ShuffledBasisNormRuntime
  ShuffledStaticBucket
  ShuffledBasisDeltaBridgeKernel
  ShuffledBridgeScore
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
ShuffledQuantileTailRuntime = fail
ShuffledBasisNormRuntime = fail
ShuffledStaticBucket = fail
ShuffledBasisDeltaBridgeKernel = fail
ShuffledBridgeScore = fail
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
p9_official_paired_replay_pareto.svg
p9_macro_beat_rate.svg
p9_signal_stratum_win_matrix.svg
p9_shuffle_control_matrix.svg
p9_system_gate_distribution.svg
```

---

## P10：short-run scout

### 目标

如果 P9 paired replay pass，验证 local causal advantage 能否进入连续训练。

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

## P11：full run / robustness / strong baseline

### 目标

只有 P10 pass 后打开。验证 functional advantage 不是 local replay artifact。

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
  ShuffledQuantileTailRuntime
  ShuffledBasisNormRuntime
  ShuffledStaticBucket
  ShuffledBasisDeltaBridgeKernel
  ShuffledBridgeScore
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
contract_audit_v9277.csv
p0_v9276_boundary_reproduction.csv
p1_quantile_tail_internal_attribution.csv
p2_quantile_tail_cache_exact_tail_fusion_runtime.csv
p3_basis_norm_runtime_integration.csv
p4_static_bucket_persistent_workspace_integrated_runtime.csv
p5_single_pass_quantile_tail_basisnorm_delta_bridge_runtime.csv
p6_integrated_materialized_runtime_candidates.csv
p7_system_legal_exact_signal_controller_v9.csv
p8_leave_dataset_and_stratum_out.csv
p9_official_paired_replay.csv
p10_short_run_functional_validation.csv
p11_full_run_robustness_strong_baseline.csv

quantile_tail_internal_trace_v9277.csv
quantile_tail_runtime_trace_v9277.csv
basis_norm_runtime_trace_v9277.csv
static_bucket_workspace_runtime_trace_v9277.csv
basisnorm_delta_bridge_runtime_trace_v9277.csv
integrated_runtime_trace_v9277.csv
system_controller_trace_v9277.csv
leaveout_trace_v9277.csv
paired_replay_branch_trace_v9277.csv
short_run_trace_v9277.csv

route_decision.json
aggregate_decision.json
failure_table.csv
artifact_hashes.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9276_boundary_unstable
F3_dataset_tuning_detected
F4_payload_binding_regression
F5_decision_metric_regression
F6_quantile_tail_internal_attribution_fail
F7_quantile_tail_no_actionable_component
F8_quantile_tail_cache_agreement_fail
F9_quantile_tail_cache_no_reduction
F10_fused_topk_tail_kernel_error_fail
F11_streaming_tail_kernel_no_reduction
F12_two_pass_borderline_tail_recall_fail
F13_basis_norm_runtime_integration_fail
F14_basis_norm_reduction_insufficient
F15_static_bucket_not_integrated
F16_persistent_workspace_not_integrated
F17_basis_norm_not_bucketed
F18_W2_delta_not_bucketed
F19_bridge_score_not_inside_kernel
F20_single_pass_quantile_tail_basisnorm_delta_bridge_not_materialized
F21_single_pass_runtime_error_fail
F22_single_pass_runtime_agreement_fail
F23_integrated_runtime_still_diagnostic
F24_integrated_runtime_step_ratio_fail
F25_integrated_runtime_memory_fail
F26_source_gap_or_formula_proxy_used
F27_system_controller_precision_fail
F28_system_controller_coverage_fail
F29_system_controller_bad_event_fail
F30_system_controller_null_rate_fail
F31_system_controller_lcb_ucb_fail
F32_system_controller_projection_used
F33_leave_dataset_out_fail
F34_leave_stratum_out_fail
F35_paired_replay_control_equivalent
F36_shuffle_control_pass
F37_functional_lr_equivalent
F38_short_run_task_drop
F39_full_run_no_macro_hard_stratum_gain
F40_strong_baseline_explains_gain
F41_robustness_fail
F42_external_not_ready
F43_fake_or_proxy_violation
F44_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-BoundaryReproduced:
  v9.2.76 boundary reproduced.

R2-QuantileTailInternalAttributionPass:
  quantile_tail internal attribution has unknown_fraction <= 0.05.

R3-QuantileTailRuntimePass:
  quantile_tail cache/fusion/runtime elimination reduces time and preserves agreement.

R4-BasisNormRuntimeIntegrationPass:
  quantile_tail reduction is integrated into basis_norm runtime.

R5-StaticBucketWorkspaceRuntimePass:
  static bucket and persistent workspace are integrated into basis_norm / W2_delta / bridge runtime.

R6-SinglePassQuantileTailBasisNormDeltaBridgePass:
  quantile-tail / basis_norm / delta / bridge score / accept bit are materialized in true runtime.

R7-IntegratedRuntimeDiagnosticPass:
  integrated materialized runtime reaches step_ratio_q90 <= 2.00.

R8-SystemLegalExactSignalControllerPass:
  system controller passes decision + compute gates.

R9-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R10-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R11-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R12-ShortRunFunctionalPass:
  short-run task-safe mechanism gain.

R13-FullFunctionalPass:
  full run task / geometry / system / control gates pass.

R14-PayloadBindingRegression:
  payload binding no longer reproduces.

R15-QuantileTailAttributionIncomplete:
  quantile_tail remains internal black box.

R16-QuantileTailDominantUnfixed:
  quantile_tail is attributed but runtime cache/fusion fails.

R17-BasisNormRuntimeStillInsufficient:
  quantile_tail improved but basis_norm integrated time remains above gate.

R18-StaticBucketWorkspaceStillUnintegrated:
  static bucket / persistent workspace still not integrated into compute.

R19-SinglePassBridgeRuntimeFail:
  bridge score / basis_norm / delta single-pass runtime cannot be materialized.

R20-SystemStillTooExpensive:
  materialized runtime improves but step_ratio_q90 remains >1.50.

R21-ComputePassButLeaveoutFail:
  system controller overfits pooled calibration.

R22-ComputePassButPairedReplayFail:
  controller is system-legal but not causally superior to controls.

R23-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` 必须记录：

```text
route
v9276_boundary_pass
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

quantile_tail_internal_attribution_pass
quantile_tail_unknown_fraction
dominant_quantile_tail_subcomponent
topk_time_ms
sort_time_ms
tail_mask_time_ms
logsumexp_time_ms
threshold_compute_time_ms
tail_context_lookup_time_ms
tail_writeback_time_ms

best_quantile_tail_runtime_id
quantile_tail_runtime_pass
quantile_tail_strategy
quantile_tail_cache_hit_rate
quantile_tail_time_before
quantile_tail_time_after
quantile_tail_time_reduction

best_basis_norm_runtime_id
basis_norm_runtime_pass
basis_norm_time_before
basis_norm_time_after
basis_norm_time_reduction
basis_norm_bucketed

best_bucket_workspace_id
static_bucket_workspace_pass
runtime_bucket_used
persistent_workspace_used
W2_delta_bucketed
bridge_score_bucketed
kernel_count_reduction
sync_count_reduction
allocation_count_reduction
avg_candidates_per_kernel_after
cuda_graph_capture_pass
cuda_graph_failure_reason

best_basis_delta_bridge_runtime_id
quantile_tail_inside_kernel
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
success_v9277_strict_purekan_functional
success_v9277_full_functional
success_v9277_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 quantile-tail internal attribution
  P2 quantile-tail cache / exact-tail fusion runtime
  P3 basis_norm runtime integration
  P4 static bucket / persistent workspace integrated runtime
  P5 single-pass quantile-tail basis_norm delta bridge runtime
  P6 integrated materialized runtime candidates

Batch 2:
  P7 system-legal controller
  P8 leave-dataset-out / leave-stratum-out
  P9 paired replay scout

Batch 3:
  official P9 paired replay
  P10 short-run if P9 passes

Batch 4:
  P11 full run / robustness / strong baseline only if P10 passes
```

Gate rule：

```text
P2/P3/P4/P5 can run in parallel after P0,
but P6/P7 cannot pass unless:
  P1 quantile-tail internal attribution pass
  P2 quantile-tail runtime pass or P3 basis_norm runtime pass
  P4 static bucket workspace pass
  P5 single-pass runtime pass
  P6 integrated runtime pass
  step_ratio_q90 <= 1.50 for official pass
  audit_only_cost_removal = 0
  diagnostic_derived_from_measured_components = 0
  projection_used = 0
  source_measured_gap_used = 0
  formula_proxy_used = 0
  full_online_payload_binding = 1
  full_online_update_payload_binding = 1
```

P8/P9 diagnostic rows may be measured before all gates finish, but official status requires:

```text
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
v9.2.76 boundary reproduced
quantile-tail internal attribution measured
quantile-tail runtime measured
basis_norm runtime integration measured
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
quantile_tail_internal_attribution_pass = 1
+
quantile_tail_unknown_fraction <= 0.05
+
dominant_quantile_tail_subcomponent identified
+
dominant component ratio >= 0.20
```

## Runtime materialization success

```text
Attribution success
+
at least one of:
  quantile_tail runtime pass
  basis_norm runtime pass
  static bucket workspace pass
  single-pass quantile-tail basis_norm-delta-bridge pass
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
1. v9.2.76 boundary cannot be reproduced；
2. payload binding regresses；
3. decision metrics regress；
4. quantile-tail internal attribution fails；
5. quantile-tail has no actionable subcomponent；
6. quantile-tail cache/fusion reduces cost but breaks agreement；
7. quantile-tail runtime reduction <50%；
8. basis_norm runtime integration fails；
9. basis_norm remains >50% of previous time；
10. static bucket remains diagnostic-only；
11. persistent workspace not integrated；
12. basis_norm_bucketed remains 0；
13. W2_delta_bucketed remains 0；
14. kernel/sync/allocation count does not decrease；
15. avg candidates per kernel remains <2；
16. bridge score cannot be moved inside kernel；
17. single-pass quantile-tail basis_norm-delta-bridge not materialized；
18. low-cost path uses source gap / formula proxy；
19. integrated runtime remains diagnostic-derived；
20. system step_ratio_q90 remains >1.50；
21. memory ratio >1.05；
22. system controller fails precision / coverage / bad-event / null-rate；
23. precision LCB below 0.75；
24. bad-event UCB above 0.05；
25. leave-dataset-out fails；
26. leave-stratum-out fails；
27. paired replay remains control-equivalent；
28. shuffle controls pass；
29. short-run task drops；
30. full run gives no macro / hard-stratum / geometry gain；
31. functional breaks system gate；
32. gains are explained by QuadraticFeatureMLP；
33. any teacher/loss/fake/proxy/offload/projection-as-pass violation occurs。
```

---

# Part XI. 最终解释规则

## Case A：P7 system pass + P8/P9 pass

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full run and robustness。

## Case B：quantile-tail runtime pass but system still slow

必须声明：

```text
quantile-tail was a real dominant cost and has been reduced, but remaining blocker moved to basis_norm writeback / W2_delta / bridge score / bucket fragmentation.
```

下一步基于 P6 component 做 targeted kernel work，不调 controller。

## Case C：quantile-tail cache/fusion breaks agreement

必须声明：

```text
quantile-tail fields are decision-critical under current C3 controller.
```

下一步不能删除 tail，只能做 exact fused tail 或 two-pass borderline exact tail。

## Case D：static bucket/workspace pass but system still slow

必须声明：

```text
fragmentation improved, but arithmetic or basis_norm/delta memory bandwidth remains dominant.
```

下一步做 T2 recurrence / memory coalescing / arithmetic simplification。

## Case E：bridge score cannot enter kernel

必须声明：

```text
accept-bit alone is insufficient; bridge-score materialization remains the integration blocker.
```

下一步修 compact bridge lookup / bridge-score partials，不继续只优化 accept-bit。

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

v9.2.77 的一句话策略是：

$$
\boxed{
\text{不要再泛泛修 basis_norm；直接打 quantile/tail compute，并把 cache、bucket、bridge-score 都接成 measured integrated runtime。}
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
1. quantile_tail_compute_time_ms = 0.371753 内部到底哪个 subcomponent 慢？
2. topk/sort/tail-mask/logsumexp/threshold 谁是主要成本？
3. quantile-tail 能否按 family/horizon/bucket/role cache？
4. cache hit rate 能否 >=0.50？
5. no-clone/top2 为什么只能降 27.12%？
6. fused exact-tail kernel 能否降到 50% 以下？
7. basis_norm_bucketed 能否从 0 变成 1？
8. W2_delta_bucketed 能否从 0 变成 1？
9. bridge_score_inside_kernel 能否从 0 变成 1？
10. avg_candidates_per_kernel 能否从 0.6648 拉到 >=8？
11. integrated runtime 是否能从 diagnostic-derived 变成 materialized？
12. step q90 能否从 2.713296 降到 <=2.00，再降到 <=1.50？
13. system controller 是否 official eligible？
14. LDO/LSO 是否通过？
15. official paired replay 是否打过 AdamWParallel / bestLR？
```

v9.2.77 的结果将给出清晰分叉：

```text
if quantile-tail runtime passes and integrated system passes:
  open system-legal controller, LDO/LSO, paired replay.

if quantile-tail runtime passes but system remains >1.50:
  continue bucketed W2_delta / bridge-score single-pass / memory coalescing.

if quantile-tail cannot be reduced without losing agreement:
  treat quantile-tail as decision-critical and implement exact fused tail or two-pass borderline.

if static bucket/workspace remains diagnostic-only:
  stop running P7/P8; integrate bucketed basis_norm/W2_delta first.

if bridge_score_inside_kernel remains 0:
  stop claiming integrated runtime; accept-bit alone is insufficient.

if compute passes but paired replay fails:
  system is legal, but functional causal advantage is insufficient.

if LDO/LSO fails:
  repair dataset-agnostic support/family reliability, not dataset-specific tuning.
```
