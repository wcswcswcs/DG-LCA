# DG-KAN v9.2.70 Full-Online Row Binding 与 Official System-Legal Controller Closure 完整实验计划

> 本计划基于 v9.2.69 `Materialized Event-Sparse True-Delta 与 System-Legal Controller` 的真实执行结果制定。  
> v9.2.69 的 terminal route 是：
>
> ```text
> route = R18-ReferenceFeasibleButComputeFail
> base_candidate = LQ-t2-h256
> success_v9269_strict_purekan_functional = False
> success_v9269_full_functional = False
> success_v9269_external_ready = False
> ```
>
> v9.2.69 的关键事实是：
>
> ```text
> reference_controller_id = C3-T2PlusBackfill
> reference_controller_reproduced = 1
> reference_precision = 0.8380281690140845
> reference_coverage = 0.03130511463844797
> reference_bad_event = 0.02464788732394366
> reference_null_rate = 0.13028169014084506
> reference_precision_lcb = 0.7907157243773478
> reference_bad_event_ucb = 0.04999458813129621
>
> materialization_gap_mapped = 1
> official_materialization_gap_mapped = 0
> unmapped_component = full online row binding
>
> PF5 runtime selector:
>   candidate_count = 2493
>   event_count = 24192
>   candidate_rate = 0.10305059523809523
>   reference_accept_recall = 1.0
>   candidate_pack_time_ms = 6.339111
>   projection_used = 0
>
> candidate-only branch forward:
>   candidate_event_rate = 0.08333333333333333
>   branch_forward_ratio_vs_full = 0.018018326773696327
>   logit_max_abs_diff_vs_full_reference = 0.0
>
> materialized true-delta microprobe:
>   materialized_system_path = 1
>   projection_used = 0
>   full_trace_projection_used = 0
>   agreement = 1.0
>   step_ratio_q90 = 1.075342155736442
>   memory_ratio = 1.0
>   official pass = 0 because full_online_row_binding = 0
>
> frozen bridge lookup:
>   uses_cpu_summary = 0
>   online_frontier_search_used = 0
>   bridge_lookup_time_ms = 0.141594
>   diagnostic pass = 1
>   official pass = 0 because full_online_row_binding = 0
>
> P6 boundary:
>   official_eligible = 0
>   system_legal_controller_pass = 0
>   reason = P4_or_P5_materialized_compute_not_official
> ```
>
> v9.2.70 的核心判断是：
>
> $$
> \boxed{
> \text{v9.2.69 已经把 projection 推进到真实 materialized microprobe；现在唯一主 blocker 是 full-online row binding。}
> }
> $$
>
> 因此 v9.2.70 不应该继续做：
>
> ```text
> 继续调 C3/T2/C4/E2 decision frontier；
> 继续重设 safe-useful / null / bad-event target；
> 继续扩 support rows；
> 继续把 microprobe 当 official；
> 继续用 projection 或 full-trace-derived sparse estimate 当 pass；
> 继续按 dataset 单独调 prefilter / threshold / binding rule。
> ```
>
> 本轮的核心任务是：
>
> $$
> \boxed{
> \text{把 PF5 runtime selector、candidate pack、candidate-only branch forward、true-delta confirm、FrozenC3 lookup 串成全量 online controller row path。}
> }
> $$
>
> 换句话说，v9.2.70 要从：
>
> ```text
> materialized microprobe path
> ```
>
> 推进到：
>
> ```text
> materialized full-online controller path
> ```
>
> 只有全量 online row binding 过了，system-legal controller 才能 official，LDO/LSO 与 paired replay 才能打开。

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

v9.2.70 的新增硬约束：

```text
full_online_row_binding 必须为 1；
materialized_system_path 必须为 1；
projection_used 必须为 0；
full_trace_projection_used 必须为 0；
source_measured_gap_used 必须为 0；
formula_proxy_used 必须为 0；
online_frontier_search_used 必须为 0；
uses_cpu_summary 必须为 0；
official_eligible 只有在全量 online path 过 gate 后才能为 1。
```

允许使用：

```text
PF5 runtime selector；
contiguous candidate tensors；
candidate-only true branch logits；
candidate-only true branch-delta；
FrozenC3-T2PlusBackfill lookup；
compact support/family table；
calibration-split frozen thresholds；
legal train-stream event table；
materialized CUDA/Triton kernels；
manual graph-free KAN path。
```

禁止使用：

```text
microprobe-only timing 替代 full controller timing；
offline projection row 替代 online event row；
full trace sparse projection 替代 actual candidate-only execution；
posthoc oracle label at commit time；
validation/test metric at commit time；
dataset_name branch；
CPU-heavy frontier summary；
重新在线搜索 C3/T2 threshold；
duplicated balanced rows；
fake/proxy rows。
```

---

# Part I. 对 v9.2.69 的独立判断

## 1. v9.2.69 没有达到最终目标

v9.2.69 没有 strict PureKAN functional success。失败点不是 reference decision，也不是 PF5 selector，也不是 materialized microprobe 的数值质量，而是 official 全量绑定没闭合：

```text
materialized_event_sparse_exact_diagnostic_pass = 1
fused_bridge_materialized_diagnostic_pass = 1
system_legal_controller_pass = 0
official_eligible = 0
full_online_row_binding = 0
```

因此不能声明：

```text
strict PureKAN local causal evidence；
official system-legal controller；
LDO / LSO success；
official paired replay success；
short-run / full-run success；
external-ready。
```

尤其不能把：

```text
step_ratio_q90 = 1.075342
agreement = 1.0
candidate_rate = 0.103051
candidate branch forward ratio = 0.018018
frozen bridge lookup time = 0.141594 ms
```

写成 official system success，因为这些是在 microprobe / diagnostic materialized path 上成立，尚未绑定到 full online controller rows。

## 2. v9.2.69 的真实进展

v9.2.69 是一次真实进展。它把 v9.2.68 的 projection-only path 推进为 materialized microprobe path。

第一，PF5 已经不只是统计 prefilter。它能 runtime 输出 candidate set：

```text
candidate_count = 2493
event_count = 24192
candidate_rate = 0.10305059523809523
reference_accept_recall = 1.0
candidate_pack_time_ms = 6.339111
candidate_indices_materialized = 1
candidate_tensor_contiguous = 1
projection_used = 0
```

第二，candidate-only branch forward 有真实 materialized 证据：

```text
candidate_event_rate = 0.08333333333333333
branch_forward_ratio_vs_full = 0.018018326773696327
logit_max_abs_diff_vs_full_reference = 0.0
```

这说明 candidate packing / candidate-only forward 的方向不是空想，至少在 microprobe 中，它确实能显著降低 branch-forward 成本，并且保持 exact logits。

第三，materialized event-sparse true-delta microprobe 已经过 diagnostic：

```text
materialized_system_path = 1
projection_used = 0
full_trace_projection_used = 0
agreement = 1.0
step_ratio_q90 = 1.075342155736442
memory_ratio = 1.0
```

第四，FrozenC3 lookup 已经摆脱 CPU-heavy frontier summary：

```text
uses_cpu_summary = 0
online_frontier_search_used = 0
bridge_lookup_time_ms = 0.141594
```

这说明 v9.2.68 的系统闭合路线没有跑偏：PF5 + materialized candidate path + FrozenC3 lookup 是一个可行方向。

## 3. v9.2.69 的真实失败

v9.2.69 的失败可以压缩成一句话：

$$
\boxed{
\text{microprobe path 已 materialize，但 full online controller rows 没有被绑定到同一条 path。}
}
$$

P6 的数值指标已经很接近 official：

```text
precision_heldout = 0.8380281690140845
coverage_heldout = 0.03130511463844797
bad_event_heldout = 0.02464788732394366
null_rate_heldout = 0.13028169014084506
precision_lcb = 0.7907157243773478
bad_event_ucb = 0.04999458813129621
agreement_reference_accept = 1.0
step_ratio_q90 = 1.075342155736442
memory_ratio = 1.0
accepted_signal_strata_count = 15
accepted_family_count = 65
max_family_share = 0.09507042253521127
max_stratum_share = 0.15140845070422534
```

这些点估计已经满足 system-legal controller 的数值门槛。但 official 仍为 0，因为：

```text
full_online_row_binding = 0
official_eligible = 0
reason = P4_or_P5_materialized_compute_not_official
```

也就是说，当前缺的是完整在线执行合同，不是新的 score。

## 4. 当前 blocker 的本质

当前 blocker 应写成：

$$
\boxed{
\text{full-online row binding and controller integration failure。}
}
$$

更具体地说，要把以下对象绑定成同一张全量 online event table：

```text
event_id
row_id
dataset
seed
step
batch_id
horizon
signal_stratum
event_family
family_id
bucket_id
PF5 candidate flag
candidate_index
candidate_pack_index
branch_forward_index
true_delta_index
bridge_lookup_index
accept_decision
functional_update_payload
timing record
memory record
audit flags
```

v9.2.69 证明这些组件分别存在，但还没有证明它们在全量 online controller rows 上逐行一致。v9.2.70 的核心是做这件事。

## 5. 当前是否在正确道路上

是。现在路线已经非常清晰：

```text
reference controller deployable
→ frozen controller stable
→ PF5 runtime selector materialized
→ candidate-only branch forward materialized
→ event-sparse true-delta microprobe materialized
→ frozen bridge lookup materialized
→ full online row binding
→ system-legal controller
→ LDO / LSO
→ official paired replay
```

错误路线是：

```text
继续做 reference frontier；
继续重新调 decision threshold；
继续扩 support；
继续把 microprobe 当 official；
继续做 projection；
继续按 dataset 调参；
继续用 proxy/source-gap 替代 true-delta。
```

---

# Part II. v9.2.70 总体目标

v9.2.70 的总体目标是：

$$
\boxed{
\text{将 v9.2.69 的 materialized microprobe path 绑定为 full-online system-legal controller。}
}
$$

本轮必须回答七个问题：

```text
Q1:
  v9.2.69 boundary 是否稳定复现？

Q2:
  full online event table 是否能逐行绑定 PF5 / candidate pack / branch forward / true-delta / bridge lookup / accept decision？

Q3:
  全量绑定后，candidate_rate 是否仍约 0.103，reference_accept_recall 是否仍 >=0.95？

Q4:
  全量 candidate-only branch forward 是否仍保持 logit exactness 与低 step cost？

Q5:
  full-online materialized true-delta controller 是否仍满足 precision / coverage / bad-event / null / LCB-UCB？

Q6:
  full-online materialized path 的真实 step_ratio_q90 是否仍 <=1.50？

Q7:
  system controller 过线后，LDO/LSO 与 official paired replay 是否可以打开？
```

v9.2.70 的核心 stop-go：

$$
\boxed{
full\_online\_row\_binding=1
\land
official\_eligible=1
\land
system\_legal\_controller\_pass=1.
}
$$

---

# Part III. 核心假设

## H1：full online binding 不会破坏 PF5 candidate sparsity

v9.2.69 PF5:

$$
CandidateRate=0.10305059523809523,
$$

$$
ReferenceAcceptRecall=1.0.
$$

H1 认为这不是 microprobe artifact，全量 online rows 上也能保持。

H1 成立标准：

$$
CandidateRate_{\text{full-online}}\leq0.25,
$$

$$
Recall_{\text{reference-accepted}}\geq0.95,
$$

并且：

```text
candidate_id_missing_count = 0
candidate_duplicate_count = 0
candidate_pack_unbound_count = 0
candidate_family_unbound_count = 0
```

H1 失败标准：

```text
candidate_rate > 0.25
reference_accept_recall < 0.95
unbound candidates > 0
candidate/event/family ids mismatch
```

## H2：full online candidate-only branch forward 仍保持 exactness

v9.2.69 microprobe:

$$
branch\_forward\_ratio\_vs\_full=0.018018326773696327,
$$

$$
logit\_max\_abs\_diff=0.0.
$$

H2 成立标准：

$$
LogitMaxAbsDiff_{\text{full-online}}\leq10^{-6},
$$

or accept-equivalent numerical agreement if floating epsilon produces harmless differences:

$$
Agreement_{\text{accept}}\geq0.99.
$$

Cost:

$$
CandidateForwardRatio_{\text{full-online}}\leq0.10.
$$

H2 失败标准：

```text
candidate-only forward silently runs full-row forward；
logit diff breaks bridge decisions；
candidate gather/scatter overhead dominates；
step_ratio_q90 > 1.50。
```

## H3：FrozenC3 bridge lookup 不需要 CPU summary or online search

v9.2.69:

```text
uses_cpu_summary = 0
online_frontier_search_used = 0
bridge_lookup_time_ms = 0.141594
```

H3 成立标准：

```text
uses_cpu_summary = 0
online_frontier_search_used = 0
frozen_bridge_lookup_used = 1
bridge_lookup_time_ratio <= 0.10
```

H3 失败标准：

```text
full-online path reintroduces CPU-heavy summary；
online search used；
bridge lookup becomes dominant subphase。
```

## H4：full online binding converts diagnostic system metrics into official controller metrics

P6 already has numeric decision metrics passing, but official eligible is 0. H4 says once binding closes, the same controller can pass official.

H4 成立标准：

```text
full_online_row_binding = 1
official_eligible = 1
system_legal_controller_pass = 1
```

with:

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
BadEventRate_{\text{UCB}}\leq0.05,
$$

$$
Agreement_{\text{reference-accept}}\geq0.90,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

H4 失败标准：

```text
binding closes but decision metrics drift；
binding closes but cost exceeds 1.50；
binding closes but official audit detects proxy/projection/CPU/offload；
binding closes but support balance fails。
```

## H5：如果 system controller pass，LDO/LSO 与 paired replay become the real scientific gates

H5 成立标准：

System controller pass 后，P7/P8 打开并 measured；LDO/LSO pass；paired replay RealFunctional beats AdamWParallel / bestLR.

H5 失败标准：

System path legal，但 leave-out 或 paired replay fail。此时不能 dataset-specific tuning，要修 dataset-agnostic support/family reliability 或 functional target。

---

# Part IV. Full-online binding contract

## 1. Online event table

全量 online event table 必须包含：

```text
global_row_id
event_id
dataset
seed
step
batch_id
microbatch_id
horizon
signal_stratum
event_family
family_id
bucket_id
role_id
branch_id
candidate_source
source_hash
```

Contract:

```text
global_row_id unique
event_id unique within run
family_id non-null
bucket_id non-null
source_hash non-null
dataset_name not used in controller rule
```

## 2. PF5 selector binding

PF5 selector 输出：

```text
candidate_flag
candidate_index
candidate_rank
candidate_score
candidate_family_id
candidate_bucket_id
candidate_branch_id
candidate_event_id
```

Contract:

```text
candidate_event_id must match online event table event_id
candidate_family_id must match online event table family_id
candidate_index must map to contiguous candidate tensor row
no duplicate candidate_index
no missing selected candidate
```

## 3. Candidate pack binding

Candidate pack tensors:

```text
candidate_x
candidate_y
candidate_logits_base
candidate_branch_metadata
candidate_family_features
candidate_support_features
candidate_event_ids
candidate_global_row_ids
```

Contract:

```text
candidate_global_row_ids preserve source order or record stable permutation
candidate tensor shape = candidate_count
candidate row -> event table row bijection
candidate pack hash exists
```

## 4. Candidate-only branch forward binding

Branch forward outputs:

```text
candidate_branch_logits
candidate_control_logits
candidate_true_delta_logits
candidate_delta_features
candidate_forward_time
candidate_forward_memory
```

Contract:

```text
no full row branch replay
uses_candidate_only_forward = 1
uses_full_row_forward = 0
logit diff vs full reference <= 1e-6 on audit subset
```

## 5. Frozen bridge lookup binding

Bridge lookup outputs:

```text
bridge_score
bad_ucb
null_ucb
support_lcb
backfill_flag
trim_flag
accept_decision
```

Contract:

```text
bridge lookup uses candidate family/bucket ids
no online frontier search
no CPU summary
lookup table hash exists
accept decision row-bound to global_row_id
```

## 6. Functional update payload binding

Accepted rows must produce:

```text
accepted_event_id
accepted_global_row_id
functional_update_payload_id
delta_theta_summary
branch_delta_summary
accept_reason
abstain_reason
reject_reason
```

Contract:

```text
accepted_global_row_id must be subset of online event table
accepted_event_id must be subset of candidate_event_id
update payload exists for accepted rows
no update payload for rejected rows
```

---

# Part V. Candidate designs

## 1. Binding candidates

### BIND0：v9.2.69 microprobe reference

Diagnostic only.

### BIND1：FullOnlineEventTableV1

Build full online event table with row/event/family/bucket ids.

### BIND2：PF5FullOnlineSelectorBinding

PF5 selector bound to full event table.

### BIND3：CandidatePackFullOnlineBinding

Candidate pack preserves full row/event ids.

### BIND4：TrueDeltaFullOnlineBinding

Branch forward and delta outputs bound to candidate ids.

### BIND5：FrozenBridgeFullOnlineBinding

Bridge lookup and accept decisions bound to row ids.

### BIND6：EndToEndFullOnlineBinding

Complete full online binding from event row to update payload.

## 2. Candidate branch forward candidates

### BF0：v9.2.69 BF1 microprobe reference

Diagnostic only.

### BF1：FullOnlineCandidatePackedForward

Materialize candidate-only branch forward for all PF5 candidates.

### BF2：FamilyGroupedCandidateForward

Group candidate rows by family/branch/horizon.

### BF3：PersistentCandidateWorkspaceForward

Preallocate candidate tensors to reduce allocation/sync.

### BF4：TritonCandidatePackForward

Fused gather/scatter candidate pack.

### BF5：HybridGroupedPersistentForward

Family-grouped + persistent workspace.

## 3. True-delta candidates

### TD0：v9.2.69 EC2 microprobe reference

Diagnostic only.

### TD1：FullOnlineCandidateOnlyTBD0

Exact true branch-delta on all PF5 candidates.

### TD2：FullOnlineSelectedBridgeLogits

Only bridge-required logits, must keep accept agreement.

### TD3：FullOnlineTwoPassBorderline

Selected logits first, full logits for borderline candidates.

### TD4：FusedCandidateDeltaFullOnline

Fused candidate delta kernel.

### TD5：PersistentWorkspaceTrueDelta

Candidate-only true delta with preallocated workspace.

## 4. Bridge candidates

### BR0：v9.2.69 BR1 reference

Diagnostic only.

### BR1：FullOnlineFrozenC3Lookup

Frozen C3 lookup bound to all candidate rows.

### BR2：QuantizedBucketFrozenC3Lookup

Quantized compact table, requires agreement.

### BR3：FusedBridgeScoreFullOnline

Fused bridge lookup + accept score.

### BR4：NoCPUBridgeAudit

Audit-only candidate ensuring no CPU summary / online search.

## 5. System candidates

### SYS0：v9.2.69 microprobe reference

Diagnostic only.

### SYS1：PF5 + FullOnlineCandidateOnlyTBD0 + FrozenC3Lookup

Primary candidate.

### SYS2：PF5 + FamilyGroupedCandidateForward + FrozenC3Lookup

Reduce branch-forward overhead.

### SYS3：PF5 + PersistentWorkspaceTrueDelta + FrozenC3Lookup

Reduce allocation/sync overhead.

### SYS4：PF5 + SelectedBridgeLogits + FrozenC3Lookup

Lower logit compute, must preserve agreement.

### SYS5：PF5 + TwoPassBorderline + FrozenC3Lookup

Balance exactness and cost.

### SYS6：PF5 + FusedCandidateDelta + FusedBridgeScore

Target final fused path.

### SYS7：HybridBestFullOnlineSystem

Best combination from binding / forward / true-delta / bridge candidates.

---

# Part VI. 实验阶段

## P0：v9.2.69 boundary reproduction

### 目标

确认 v9.2.69 boundary 稳定，避免基于偶然 microprobe 制定 route。

### 必须记录

```text
route
source_route_v9268
reference_controller_id
reference_controller_reproduced
reference_precision
reference_coverage
reference_bad_event
reference_null_rate
reference_precision_lcb
reference_bad_event_ucb
materialization_gap_mapped
official_materialization_gap_mapped
pf5_runtime_selector_pass
candidate_rate
reference_accept_recall
candidate_branch_forward_pass
candidate_forward_time_ratio
materialized_event_sparse_exact_diagnostic_pass
materialized_event_sparse_exact_pass
materialized_system_path
projection_used
full_trace_projection_used
full_online_row_binding
exact_agreement
exact_step_ratio_q90
exact_memory_ratio
fused_bridge_materialized_diagnostic_pass
system_legal_controller_pass
primary_blocker
fake_proxy_count
cpu_offload_used
```

### 判断标准

P0 pass：

```text
route = R18-ReferenceFeasibleButComputeFail
reference_controller_id = C3-T2PlusBackfill
reference_controller_reproduced = 1
PF5 runtime selector pass = 1
materialized microprobe diagnostic pass = 1
full_online_row_binding = 0
system_legal_controller_pass = 0
fake/proxy/offload = 0
```

### 可视化

```text
p0_v9269_boundary_ladder.svg
p0_projection_to_microprobe_to_full_online.svg
p0_current_blocker_dashboard.svg
```

---

## P1：full-online row binding contract audit

### 目标

建立完整 row binding contract，明确从 online event row 到 accepted update payload 的每个 id 是否可追踪。

### 必须记录

```text
component
required_for_official
materialized
bound_to_full_online_row
id_fields
hash_fields
missing_count
duplicate_count
mismatch_count
uses_projection
uses_posthoc
uses_cpu_summary
uses_online_search
implementation_file
blocking_reason
```

Components：

```text
online event table
PF5 selector output
candidate indices
candidate family ids
candidate branch ids
candidate tensors
candidate branch logits
true branch-delta tensors
FrozenC3 lookup rows
bridge score
accept decision
functional update payload
timing record
memory record
```

### 判断标准

P1 pass：

```text
all required components materialized = 1
all required components bound_to_full_online_row = 1
missing_count = 0
duplicate_count = 0
mismatch_count = 0
uses_projection = 0
uses_posthoc = 0
uses_cpu_summary = 0
```

### 可视化

```text
p1_binding_contract_matrix.svg
p1_component_row_binding_sankey.svg
p1_missing_duplicate_mismatch_dashboard.svg
```

---

## P2：full-online event table and PF5 selector binding

### 目标

把 PF5 runtime selector 从 microprobe/candidate trace 推进到 full online event table。

### 必须记录

```text
event_table_id
event_count
global_row_count
dataset
seed
step
horizon
family_count
bucket_count
candidate_count
candidate_rate
reference_accept_recall
reference_accept_precision
candidate_indices_materialized
candidate_family_ids_materialized
candidate_branch_ids_materialized
candidate_event_ids_materialized
candidate_duplicate_count
candidate_missing_count
candidate_pack_ready
selector_time_ms
candidate_pack_time_ms
candidate_pack_memory_MB
projection_used
posthoc_used_at_commit
dataset_name_used
```

### 判断标准

P2 pass：

$$
CandidateRate\leq0.25,
$$

$$
Recall_{\text{reference-accepted}}\geq0.95,
$$

```text
candidate_indices_materialized = 1
candidate_family_ids_materialized = 1
candidate_branch_ids_materialized = 1
candidate_event_ids_materialized = 1
candidate_duplicate_count = 0
candidate_missing_count = 0
projection_used = 0
posthoc_used_at_commit = 0
dataset_name_used = 0
```

### 可视化

```text
p2_candidate_rate_recall_curve.svg
p2_event_candidate_binding_heatmap.svg
p2_candidate_distribution_by_family_stratum.svg
p2_selector_pack_time_breakdown.svg
```

---

## P3：full-online candidate-only branch forward

### 目标

在全量 PF5 candidates 上执行 candidate-only true branch forward，证明微探针结果不是接口样本 artifact。

### 必须记录

```text
branch_forward_id
event_count
candidate_count
candidate_rate
uses_candidate_only_forward
uses_full_row_forward
candidate_pack_contiguous
candidate_pack_hash
branch_forward_time_ms_mean
branch_forward_time_ms_q90
branch_forward_ratio_vs_full
kernel_count
sync_count
allocation_count
read_MB
write_MB
largest_temp_tensor_MB
logit_max_abs_diff_vs_full_reference
logit_rel_err
accept_agreement_after_forward
full_online_row_binding
```

### 判断标准

P3 pass：

```text
uses_candidate_only_forward = 1
uses_full_row_forward = 0
full_online_row_binding = 1
```

and:

$$
LogitMaxAbsDiff\leq10^{-6},
$$

or:

$$
AcceptAgreement_{\text{after-forward}}\geq0.99.
$$

Cost diagnostic pass：

$$
BranchForwardRatio_{\text{candidate/full}}\leq0.10.
$$

### 可视化

```text
p3_candidate_forward_cost_full_online.svg
p3_logit_error_histogram.svg
p3_branch_forward_ratio_by_family.svg
p3_kernel_sync_allocation_breakdown.svg
```

---

## P4：full-online materialized true-delta exact confirmation

### 目标

真正实现全量 online event table 上的 event-sparse true-delta exact confirmation。

### 必须记录

```text
exact_candidate_id
prefilter_id
branch_forward_id
true_delta_id
event_count
candidate_count
candidate_rate
uses_true_branch_delta
uses_source_measured_gap
uses_formula_proxy
materialized_system_path
projection_used
full_trace_projection_used
full_online_row_binding
candidate_only_branch_forward_used
agreement_reference_accept
AUC_safe_good
AUC_bridge_accept
precision
coverage
bad_event
null_rate
precision_lcb
bad_event_ucb
step_ratio_q90
memory_ratio
kernel_count
sync_count
dominant_subphase
```

### 判断标准

Full-online true-delta pass：

```text
materialized_system_path = 1
projection_used = 0
full_trace_projection_used = 0
full_online_row_binding = 1
uses_true_branch_delta = 1
uses_source_measured_gap = 0
uses_formula_proxy = 0
```

and:

$$
Agreement_{\text{reference-accept}}\geq0.90,
$$

$$
Precision\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05,
$$

$$
NullRate\leq0.15,
$$

$$
Precision_{\text{LCB}}\geq0.75,
$$

$$
BadEventRate_{\text{UCB}}\leq0.05,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

Diagnostic pass：

$$
Agreement_{\text{reference-accept}}\geq0.90
$$

and:

$$
StepRatio_{q90}\leq2.00.
$$

### 可视化

```text
p4_full_online_true_delta_frontier.svg
p4_microprobe_vs_full_online_step_ratio.svg
p4_reference_agreement_confusion.svg
p4_subphase_cost_waterfall_full_online.svg
```

---

## P5：full-online fused / compact bridge materialized compute

### 目标

把 FrozenC3 bridge lookup 绑定到 full-online candidate rows，避免 CPU summary 或 online frontier search 回流。

### 必须记录

```text
bridge_system_id
exact_candidate_id
bridge_compute_id
uses_fused_kernel
uses_compact_lookup
uses_quantized_bucket
uses_cpu_summary
online_frontier_search_used
materialized_system_path
full_online_row_binding
agreement_reference_accept
AUC_bridge_accept
precision
coverage
bad_event
null_rate
precision_lcb
bad_event_ucb
bridge_lookup_time_ms
bridge_score_time_ms
step_ratio_q90
memory_ratio
kernel_count
sync_count
read_MB
write_MB
```

### 判断标准

Full-online fused bridge pass：

```text
materialized_system_path = 1
full_online_row_binding = 1
uses_cpu_summary = 0
online_frontier_search_used = 0
projection_used = 0
```

and:

$$
Agreement_{\text{reference-accept}}\geq0.90,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

Decision gate remains:

$$
Precision\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05,
$$

$$
NullRate\leq0.15.
$$

### 可视化

```text
p5_full_online_bridge_cost_signal_pareto.svg
p5_bridge_lookup_time_by_family.svg
p5_cpu_summary_elimination_audit.svg
p5_online_search_elimination_audit.svg
```

---

## P6：system-legal exact-signal controller v3

### 目标

把 P4/P5 的 full-online materialized path 升级为 official system controller。

### 必须记录

```text
controller_id
prefilter_id
branch_forward_id
exact_candidate_id
bridge_system_id
thresholds
calibration_split_id
heldout_split_id
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
accepted_strata_count
accepted_family_count
max_family_share
max_stratum_share
AUC_safe_good
AUC_bridge_accept
agreement_reference_accept
candidate_rate
step_ratio_q90
memory_ratio
materialized_system_path
projection_used
full_trace_projection_used
full_online_row_binding
source_measured_gap_used
formula_proxy_used
cpu_offload_used
dataset_name_used
posthoc_used_at_commit
validation_used
test_used
official_eligible
```

### 判断标准

System-legal controller pass：

```text
official_eligible = 1
materialized_system_path = 1
full_online_row_binding = 1
projection_used = 0
full_trace_projection_used = 0
source_measured_gap_used = 0
formula_proxy_used = 0
cpu_offload_used = 0
dataset_name_used = 0
```

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
NullRate_{\text{heldout}}\leq0.15,
$$

$$
Precision_{\text{LCB}}\geq0.75,
$$

$$
BadEventRate_{\text{UCB}}\leq0.05,
$$

$$
Agreement_{\text{reference-accept}}\geq0.90,
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
p6_system_controller_decision_frontier.svg
p6_reference_vs_system_accept_overlap.svg
p6_full_online_binding_audit.svg
p6_cost_vs_quality_frontier.svg
p6_family_strata_balance.svg
```

---

## P7：leave-dataset-out / leave-stratum-out

### 目标

证明 full-online system controller 不是 pooled calibration artifact，也不是 dataset-specific route。

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
prefilter_id
exact_candidate_id
bridge_system_id
threshold
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
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random,
           ShuffledTrueBranchDelta, ShuffledPF5Prefilter,
           ShuffledCandidatePack, ShuffledFullOnlineBinding,
           ShuffledExactConfirm, ShuffledFrozenBridge,
           ShuffledSupportStat, ShuffledControlGain,
           ShuffledCandidateGate, ShuffledBranchRatio,
           ShuffledSignalChannel, FunctionalChannelShuffled,
           TailMaskShuffled, RoleScoreShuffled,
           DatasetRouteShuffled, EventRouteShuffled,
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
ShuffledCandidatePack = fail
ShuffledFullOnlineBinding = fail
ShuffledExactConfirm = fail
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
controls = AdamWOnly, AdamWParallel, bestLR, StrongLRGrid, QuadraticFeatureMLP,
           NoOp, Random, ShuffledTrueBranchDelta, ShuffledPF5,
           ShuffledFullOnlineBinding, ShuffledFrozenBridge
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
Functional not explained by shuffled controller
```

---

# Part VII. Required artifacts

```text
run_manifest.json
contract_audit_v9270.csv
p0_v9269_boundary_reproduction.csv
p1_full_online_row_binding_contract_audit.csv
p2_full_online_event_table_pf5_selector_binding.csv
p3_full_online_candidate_only_branch_forward.csv
p4_full_online_materialized_true_delta_exact_confirmation.csv
p5_full_online_fused_compact_bridge_compute.csv
p6_system_legal_exact_signal_controller_v3.csv
p7_leave_dataset_and_stratum_out.csv
p8_official_paired_replay.csv
p9_short_run_functional_validation.csv
p10_full_run_robustness_strong_baseline.csv
full_online_event_table_v9270.csv
full_online_binding_trace_v9270.csv
pf5_full_online_selector_trace_v9270.csv
candidate_pack_full_online_trace_v9270.csv
candidate_branch_forward_full_online_trace_v9270.csv
full_online_true_delta_trace_v9270.csv
full_online_bridge_lookup_trace_v9270.csv
system_controller_trace_v9270.csv
leaveout_trace_v9270.csv
paired_replay_branch_trace_v9270.csv
short_run_trace_v9270.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9269_boundary_unstable
F3_dataset_tuning_detected
F4_reference_controller_not_reproducible
F5_full_online_binding_contract_incomplete
F6_event_table_missing_ids
F7_pf5_full_online_selector_fail
F8_pf5_recall_fail
F9_candidate_rate_too_high
F10_candidate_pack_binding_fail
F11_candidate_duplicate_or_missing
F12_candidate_branch_forward_not_candidate_only
F13_candidate_branch_forward_numerical_mismatch
F14_candidate_branch_forward_too_expensive
F15_full_online_true_delta_agreement_fail
F16_full_online_true_delta_system_fail
F17_full_online_true_delta_memory_fail
F18_frozen_bridge_cpu_summary_used
F19_frozen_bridge_online_search_used
F20_frozen_bridge_binding_fail
F21_system_controller_projection_used
F22_system_controller_precision_fail
F23_system_controller_coverage_fail
F24_system_controller_bad_event_fail
F25_system_controller_null_rate_fail
F26_system_controller_lcb_ucb_fail
F27_full_online_row_binding_fail
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
  v9.2.69 boundary reproduced.

R2-FullOnlineBindingContractPass:
  all required components have full online row binding.

R3-PF5FullOnlineSelectorPass:
  PF5 runtime selector reaches candidate_rate / recall gate on full online rows.

R4-CandidateBranchForwardFullOnlinePass:
  candidate-only branch forward is row-bound, exact, and cheap.

R5-FullOnlineTrueDeltaPass:
  materialized true-delta exact confirmation passes decision and system gates.

R6-FrozenBridgeFullOnlinePass:
  FrozenC3 bridge lookup passes without CPU summary / online search.

R7-SystemLegalExactSignalControllerPass:
  system controller passes decision + compute + full-online materialization gates.

R8-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R9-LeaveStratumOutPass:
  controller generalizes across held-out strata.

R10-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R11-ShortRunFunctionalPass:
  short-run task-safe mechanism gain.

R12-FullFunctionalPass:
  full run task / geometry / system / control gates pass.

R13-ReferenceControllerUnstable:
  C3-T2PlusBackfill cannot be reproduced or frozen.

R14-FullOnlineBindingFail:
  microprobe path cannot bind to full online rows.

R15-PF5FullOnlineSelectorFail:
  PF5 works in microprobe but not full online event table.

R16-CandidatePackBindingFail:
  candidate tensors cannot preserve event/family/branch identity.

R17-CandidateForwardFullOnlineTooExpensive:
  candidate-only branch forward materializes but is too slow.

R18-MaterializedTrueDeltaFullOnlineTooExpensive:
  true-delta path remains >1.50 despite binding.

R19-FrozenBridgeBindingFail:
  bridge lookup cannot be made row-bound without CPU summary/search.

R20-ComputePassButLeaveoutFail:
  system controller overfits pooled calibration.

R21-ComputePassButPairedReplayFail:
  controller is system-legal but not causally superior to controls.

R22-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` 必须记录：

```text
route
v9269_boundary_pass
dataset_tuning_detected
reference_controller_id
reference_controller_reproduced
reference_precision
reference_coverage
reference_bad_event
reference_null_rate
reference_precision_lcb
reference_bad_event_ucb
full_online_binding_contract_pass
event_table_pass
event_count
global_row_count
best_prefilter_id
pf5_full_online_selector_pass
candidate_count
candidate_rate
reference_accept_recall
candidate_pack_binding_pass
candidate_duplicate_count
candidate_missing_count
best_branch_forward_id
candidate_branch_forward_full_online_pass
candidate_forward_ratio
branch_logit_error_max
accept_agreement_after_forward
best_exact_candidate_id
full_online_true_delta_pass
materialized_system_path
projection_used
full_trace_projection_used
full_online_row_binding
exact_agreement
exact_step_ratio_q90
exact_memory_ratio
best_bridge_id
frozen_bridge_full_online_pass
uses_cpu_summary
online_frontier_search_used
bridge_lookup_time_ms
best_system_controller_id
system_legal_controller_pass
official_eligible
controller_precision
controller_coverage
controller_bad_event
controller_null_rate
controller_precision_lcb
controller_bad_event_ucb
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
success_v9270_strict_purekan_functional
success_v9270_full_functional
success_v9270_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 full-online row binding contract audit
  P2 full-online event table / PF5 selector binding
  P3 full-online candidate-only branch forward
  P4 full-online materialized true-delta confirmation
  P5 full-online fused / compact bridge compute

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
P2/P3/P4/P5 can run in parallel after P1 contract audit.
P6 cannot pass unless:
  P0 pass
  P1 full-online binding contract pass
  P2 PF5 full-online selector pass
  P4 or P5 full-online materialized compute pass
  projection_used = 0
  full_online_row_binding = 1
P7/P8 diagnostic rows may be measured before all gates finish,
but official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  reference controller reproduced
  frozen controller pass
  PF5 runtime selector pass
  full-online row binding pass
  true branch-delta legality pass
  true branch-delta predictivity pass
  true branch-delta agreement pass
  true branch-delta system pass
  system-legal controller pass
```

---

# Part X. 停止条件

## Minimum diagnostic success

```text
v9.2.69 boundary reproduced
full-online row binding contract audited
PF5 full-online selector measured
candidate branch forward full-online measured
full-online materialized true-delta measured
full-online frozen bridge measured
system controller measured
no fake/proxy/offload/loss/teacher violation
```

## Full-online materialization success

```text
Minimum diagnostic success
+
full_online_row_binding = 1
+
materialized_system_path = 1
+
projection_used = 0
+
candidate-only true branch forward materialized on full online rows
+
frozen bridge lookup materialized on full online rows
```

## System success

```text
Full-online materialization success
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
1. v9.2.69 boundary cannot be reproduced；
2. reference controller becomes unstable；
3. full-online binding contract cannot be completed；
4. event table lacks stable row/event/family/bucket ids；
5. PF5 full-online candidate rate >0.25；
6. PF5 full-online reference recall <0.95；
7. candidate pack has duplicate/missing rows；
8. candidate-only branch forward silently reverts to full-row forward；
9. candidate-only branch forward mismatches full logits；
10. full-online true-delta loses reference agreement；
11. full-online true-delta remains step_ratio_q90 >1.50；
12. frozen bridge lookup uses CPU summary or online search；
13. system controller uses projection；
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

## Case A：full-online system controller + LDO/LSO + paired replay pass

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full run and robustness。

## Case B：full-online binding pass but compute fail

必须声明：

```text
materialized full-online execution is correct, but true-delta system implementation remains too expensive.
```

下一步继续 kernel / candidate packing / fused branch-delta，不调 decision frontier。

## Case C：full-online binding fails

必须声明：

```text
microprobe path cannot yet be promoted to official online controller because row identities are not bound end-to-end.
```

下一步继续 event table / candidate pack / bridge lookup binding，不调 reference controller。

## Case D：PF5 fails on full online rows

必须声明：

```text
PF5 was a microprobe-successful selector but not a robust runtime selector.
```

下一步重新设计 executable cheap prefilter，而不是回到 oracle/reference frontier。

## Case E：compute pass but paired replay fail

必须声明：

```text
controller is system-legal but not causally superior to matched controls.
```

下一步回到 functional event/value target，而不是 kernelization。

## Case F：LDO/LSO fail

必须声明：

```text
controller is not dataset-agnostic or stratum-agnostic enough.
```

不能通过 dataset-specific tuning 写成功；下一步修 support/family reliability。

---

# Part XII. 最终建议

v9.2.70 的一句话策略是：

$$
\boxed{
\text{不要再调 decision；把 v9.2.69 的 materialized microprobe path 绑定到 full online rows，形成 official system controller。}
}
$$

当前最关键的问题不是：

```text
oracle 是否存在；
reference frontier 是否 deployable；
C3/T2/C4/E2 是否能 bridge；
safe-useful target 是否可行；
PF5 candidate rate 是否看起来好；
microprobe step ratio 是否 <=1.50；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. full online event table 是否有稳定 row/event/family/bucket ids？
2. PF5 selector 是否能在 full online rows 上保持 candidate_rate <=0.25 和 recall >=0.95？
3. candidate pack 是否能逐行绑定 event_id / family_id / branch_id？
4. candidate-only branch forward 是否在 full online rows 上保持 exact logits？
5. true branch-delta exact confirmation 是否保持 agreement >=0.90？
6. FrozenC3 lookup 是否完全摆脱 CPU summary / online frontier search？
7. full_online_row_binding 是否能从 0 变成 1？
8. official_eligible 是否能从 0 变成 1？
9. LDO/LSO 是否通过？
10. official paired replay 是否打过 AdamWParallel / bestLR？
```

v9.2.70 的结果将给出清晰分叉：

```text
if full-online system controller passes:
  open LDO/LSO and official paired replay.

if full-online binding passes but compute fails:
  continue candidate-packing / branch-forward / fused-delta kernelization.

if full-online binding fails:
  repair row/event/family/candidate/bridge binding contract.

if PF5 fails at full-online scale:
  redesign executable prefilter.

if compute passes but paired replay fails:
  system is legal, but functional causal advantage is insufficient.

if LDO/LSO fails:
  repair dataset-agnostic support/family reliability, not dataset-specific tuning.
```
