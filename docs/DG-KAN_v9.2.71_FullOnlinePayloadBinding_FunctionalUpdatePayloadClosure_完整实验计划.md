# DG-KAN v9.2.71 Full-Online Payload Binding 与 Functional Update Payload Closure 完整实验计划

> 本计划基于 v9.2.70 `Full-Online Row Binding 与 System-Legal Controller Closure` 的真实执行结果制定。  
> v9.2.70 的 terminal route 是：
>
> ```text
> route = R14-FullOnlineBindingFail
> base_candidate = LQ-t2-h256
> success_v9270_strict_purekan_functional = False
> success_v9270_full_functional = False
> success_v9270_external_ready = False
> ```
>
> v9.2.70 的关键事实是：
>
> ```text
> Reference:
>   reference_controller_id = C3-T2PlusBackfill
>   reference_controller_reproduced = 1
>   reference_precision = 0.8380281690140845
>   reference_coverage = 0.03130511463844797
>   reference_bad_event = 0.02464788732394366
>   reference_null_rate = 0.13028169014084506
>   reference_precision_lcb = 0.7907157243773478
>   reference_bad_event_ucb = 0.04999458813129621
>
> P1 full-online binding:
>   event_count = 24192
>   candidate_count = 2493
>   accepted_count = 951
>   event_identity_binding_pass = 1
>   full_online_row_binding_contract_pass = 0
>   full_online_row_binding = 0
>
> P1 blocker:
>   candidate_tensor_payload_missing_count = 2493
>   candidate_branch_logits_missing_count = 2493
>   candidate_true_delta_logits_missing_count = 2493
>   functional_update_payload_missing_count = 951
>
> P2 PF5 selector:
>   pf5_full_online_selector_pass = 1
>   candidate_rate = 0.10305059523809523
>   reference_accept_recall = 1.0
>   candidate metadata bound = 1
>   candidate payload bound = 0
>
> P3 candidate branch forward:
>   candidate_branch_forward_diagnostic_pass = 1
>   candidate_branch_forward_full_online_pass = 0
>   candidate_forward_time_ratio = 0.018018326773696327
>   branch_logit_error_max = 0.0
>   primary_binding_blocker = candidate_x_y_logits_payload_missing_for_full_online_candidate_rows
>
> P4 true-delta:
>   uses_true_branch_delta = 1
>   uses_source_measured_gap = 0
>   uses_formula_proxy = 0
>   agreement_reference_accept = 1.0
>   step_ratio_q90 = 1.075342155736442
>   memory_ratio = 1.0
>   precision = 0.8380281690140845
>   coverage = 0.03130511463844797
>   bad_event = 0.02464788732394366
>   null_rate = 0.13028169014084506
>   full_online_materialized_true_delta_pass = 0
>   materialization_blocker = candidate_true_delta_logits_not_bound_to_full_online_rows
>
> P5 frozen bridge lookup:
>   bridge_system_id = BR1-FullOnlineFrozenC3LookupTensorGather
>   uses_compact_lookup = 1
>   uses_cpu_summary = 0
>   online_frontier_search_used = 0
>   materialized_system_path = 1
>   full_online_row_binding = 1
>   bridge_lookup_time_ms = 0.152631
>   agreement_reference_accept = 1.0
>   fused_bridge_full_online_pass = 1
>
> P6 system controller:
>   controller_id = C3-T2PlusBackfill
>   official_eligible = 0
>   system_legal_controller_pass = 0
>   reason = P1_or_P4_full_online_binding_failed
> ```
>
> v9.2.71 的核心判断是：
>
> $$
> \boxed{
> \text{v9.2.70 不是 decision failure，也不是 compute failure；它是 payload binding failure。}
> }
> $$
>
> v9.2.70 已经把 `event_id / candidate_id / bridge_lookup / accept_decision` 绑定清楚，但没有把真正执行 functional update 所需的 payload 绑定清楚。  
> 因此 v9.2.71 不应继续调 `C3-T2PlusBackfill`、PF5 threshold、C0/T2/C4/E2 decision frontier，也不应继续重设 safe-useful/null/bad target。  
> 本轮要做的是：
>
> $$
> \boxed{
> \text{把 candidate payload、branch logits、true-delta logits、accepted update payload 从 missing 变成 full-online materialized。}
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

v9.2.71 新增硬约束：

```text
candidate_tensor_payload_missing_count = 0
candidate_branch_logits_missing_count = 0
candidate_true_delta_logits_missing_count = 0
functional_update_payload_missing_count = 0
full_online_row_binding = 1
full_online_payload_binding = 1
full_online_update_payload_binding = 1
materialized_system_path = 1
projection_used = 0
full_trace_projection_used = 0
source_measured_gap_used = 0
formula_proxy_used = 0
cpu_offload_used = 0
online_frontier_search_used = 0
uses_cpu_summary = 0
official_eligible = 1 only if all above are true
```

允许使用：

```text
PF5 runtime selector
full-online event table
candidate indices / family ids / branch ids / bucket ids
candidate x/y/base-logits payload
candidate-only true branch forward
candidate-only true branch-delta logits
FrozenC3-T2PlusBackfill lookup
compact support/family table
accepted functional update payload
materialized graph-free manual update path
CUDA/Triton kernels where applicable
```

禁止使用：

```text
microprobe-only payload 替代 full-online payload
diagnostic branch logits 替代 all candidate branch logits
true-delta summary score 替代 true-delta logits
bridge accept decision 替代 functional update payload
offline projection row 替代 online event row
source-measured scalar gap
formula proxy
posthoc oracle label at commit time
dataset_name branch
validation/test metric at commit time
CPU-heavy frontier summary
online threshold search
duplicated balanced rows
fake/proxy rows
```

---

# Part I. 对 v9.2.70 的独立判断

## 1. v9.2.70 没有达到目标

v9.2.70 不能算 strict PureKAN functional success，因为：

```text
full_online_row_binding_contract_pass = 0
candidate_pack_binding_pass = 0
full_online_materialized_true_delta_pass = 0
system_legal_controller_pass = 0
official_eligible = 0
LDO / LSO / paired replay / short-run / full-run = not_run
```

这不是保守过头，而是正确 gate。  
原因很简单：如果 candidate payload 没有全量绑定，系统无法证明真正被接受的 row 在同一条 online path 中拥有：

```text
input x
label y
base logits
branch logits
true-delta logits
bridge score
accept decision
functional update payload
```

缺其中任何一环，controller 就不能 official。

## 2. v9.2.70 的真实进展

v9.2.70 不是“没进展”。它完成了三件非常关键的事情。

第一，event identity binding 已经闭合：

```text
event_count = 24192
candidate_count = 2493
event/global/candidate duplicate = 0
event_identity_binding_pass = 1
```

这说明 row identity 本身不是问题。过去我们担心 full-online row table 可能混乱，现在看不是。row/event/source hash 可以绑定。

第二，PF5 selector 在 full-online event table 上仍然成立：

```text
candidate_rate = 0.10305059523809523
reference_accept_recall = 1.0
candidate metadata bound = 1
```

这说明 PF5 不是 microprobe-only 的统计结论；它能在全量 event table 上输出稳定 candidate ids / metadata。当前缺的是 payload，不是 selector 逻辑。

第三，FrozenC3 bridge lookup 已经 full-online materialized：

```text
uses_compact_lookup = 1
uses_cpu_summary = 0
online_frontier_search_used = 0
full_online_row_binding = 1
fused_bridge_full_online_pass = 1
bridge_lookup_time_ms = 0.152631
```

这非常重要：bridge lookup 已经不是 CPU-heavy frontier summary，也不是 online search。它是可绑定的 compact lookup。

## 3. v9.2.70 的真实失败

v9.2.70 的失败是 payload 层：

```text
candidate tensor payload missing = 2493
candidate branch logits missing = 2493
candidate true-delta logits missing = 2493
functional update payload missing = 951
```

这说明所有 PF5 candidates 都缺完整 payload；所有 accepted rows 都缺 update payload。

更精确地说：

```text
PF5 says which rows are candidates；
bridge lookup says which candidate should accept；
但 candidate branch forward / true-delta / functional update 没有在 full-online row table 中留下逐行 payload。
```

因此 P4 的 `agreement = 1.0`、`step_ratio_q90 = 1.075342`、`memory_ratio = 1.0` 仍只能算 diagnostic cost/decision evidence。它不能 official，因为这些值没有绑定到全量 candidate payload。

## 4. 当前 blocker 的本质

当前 blocker 应写成：

$$
\boxed{
\text{candidate payload and update payload are not persisted as row-bound first-class tensors.}
}
$$

不是：

```text
decision geometry fail
PF5 fail
bridge lookup fail
C3-T2PlusBackfill fail
true-delta signal fail
kernel cost fail
optimizer fail
basis fail
```

更具体地说，本轮 blocker 是四个 payload 表缺失：

```text
1. candidate_x_y_logits_payload:
   candidate_x
   candidate_y
   base_logits
   base_loss / base_margin if needed
   candidate_global_row_id
   candidate_event_id

2. candidate_branch_logits_payload:
   branch_logits
   control_logits
   branch_id
   horizon
   event_id

3. candidate_true_delta_logits_payload:
   true_delta_logits
   selected_delta_logits
   delta_norm
   bridge_required_delta_features
   event_id

4. accepted_functional_update_payload:
   accepted_event_id
   delta_theta block references
   functional_update_scale
   update reason
   abstain/reject reason
   payload hash
```

v9.2.71 不能再绕过这些表。

## 5. 是否还在正确道路上

是。v9.2.70 证明方向仍然正确，因为三个最关键的高层组件都已成立：

```text
reference controller stable
PF5 candidate selector stable
FrozenC3 bridge lookup stable
```

现在要完成的是系统工程中最后一段“payload 物化与端到端绑定”。  
如果 v9.2.71 仍不能绑定 payload，那说明系统实现层需要重构 event table / candidate tensor lifecycle，而不是去调 decision frontier。  
如果 payload 绑定成功但 step ratio 变慢，才回到 kernelization / candidate packing / memory workspace。  
如果 payload 绑定成功且 system pass，才打开 LDO/LSO 和 paired replay。

---

# Part II. v9.2.71 总体目标

v9.2.71 的总体目标是：

$$
\boxed{
\text{将 PF5 candidates 的 full-online tensor/logit/update payload 全量持久化并绑定，形成 official system-legal controller。}
}
$$

本轮要回答八个问题：

```text
Q1:
  v9.2.70 boundary 是否稳定复现？

Q2:
  candidate_x / candidate_y / base_logits 是否能对 2493 个 PF5 candidates 全量绑定？

Q3:
  candidate branch logits 是否能对 2493 个 PF5 candidates 全量 materialize？

Q4:
  candidate true-delta logits 是否能对 2493 个 PF5 candidates 全量 materialize？

Q5:
  accepted functional update payload 是否能对 951 个 accepted rows 全量 materialize？

Q6:
  payload 全量绑定后，decision metrics 是否保持 v9.2.70 的 reference 水平？

Q7:
  payload 全量绑定后，真实 step_ratio_q90 是否仍 <=1.50？

Q8:
  system controller 过线后，LDO/LSO 与 official paired replay 是否可以打开？
```

v9.2.71 的核心 stop-go 是：

$$
\boxed{
candidate\_tensor\_payload\_missing=0
\land
candidate\_true\_delta\_logits\_missing=0
\land
functional\_update\_payload\_missing=0
\land
official\_eligible=1.
}
$$

---

# Part III. 核心假设

## H1：candidate payload 缺失是生命周期问题，不是 PF5 selector 问题

v9.2.70 PF5 已过：

$$
CandidateRate=0.10305059523809523,
$$

$$
Recall_{\text{reference}}=1.0.
$$

H1 认为 candidate payload missing 来自 runner 没有把 `x/y/base_logits` 写入 candidate payload table，而不是 PF5 selector 本身失效。

H1 成立标准：

```text
candidate_tensor_payload_missing_count = 0
candidate_x_bound = 1
candidate_y_bound = 1
candidate_base_logits_bound = 1
candidate_payload_hash_present = 1
```

同时保持：

$$
CandidateRate\leq0.25,
$$

$$
Recall_{\text{reference}}\geq0.95.
$$

H1 失败标准：

```text
candidate payload 一旦写入，candidate ids/indices 出现 mismatch；
candidate_rate > 0.25；
reference_accept_recall < 0.95；
candidate payload duplicate/missing > 0。
```

## H2：branch logits missing 是 materialized trace scope 问题，不是 true branch forward 不可行

v9.2.69 microprobe 中：

$$
candidate\_forward\_ratio\_vs\_full=0.018018326773696327,
$$

$$
logit\_max\_abs\_diff=0.0.
$$

H2 认为 full-online branch logits missing 是因为 P3 仍引用 microprobe diagnostic，而不是 full-online candidate loop 没法跑。

H2 成立标准：

```text
candidate_branch_logits_missing_count = 0
uses_candidate_only_forward = 1
uses_full_row_forward = 0
branch_logits_hash_present = 1
```

并且：

$$
LogitMaxAbsDiff\leq10^{-6}
$$

or:

$$
AcceptAgreement_{\text{after-branch-forward}}\geq0.99.
$$

Cost：

$$
BranchForwardRatio_{\text{candidate/full}}\leq0.10.
$$

H2 失败标准：

```text
branch logits 只能在 microprobe subset 上存在；
full candidate loop fallback 到 full-row forward；
candidate gather/scatter overhead 使 step ratio >1.50；
logit diff 破坏 accept agreement。
```

## H3：true-delta logits missing 是 delta payload 未持久化问题，不是 true-delta signal/cost 问题

v9.2.70 P4 diagnostic:

$$
Agreement=1.0,
$$

$$
StepRatio_{q90}=1.075342155736442,
$$

$$
MemoryRatio=1.0.
$$

H3 认为 full-online true-delta pass 不过，是因为 `true_delta_logits` 没有逐 candidate 写出，而不是 true-delta 不能执行。

H3 成立标准：

```text
candidate_true_delta_logits_missing_count = 0
true_delta_logits_bound = 1
uses_true_branch_delta = 1
uses_source_measured_gap = 0
uses_formula_proxy = 0
```

and:

$$
Agreement_{\text{reference}}\geq0.90,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

H3 失败标准：

```text
true_delta logits 只能生成 aggregate score，不能 row-bound；
source gap / formula proxy 回流；
full-online materialization 使 cost 超线。
```

## H4：functional update payload missing 是最后的 official blocker

v9.2.70 accepted_count = `951`，functional update payload missing = `951`。H4 认为 accepted rows 已经确定，但没有生成 update payload。

H4 成立标准：

```text
functional_update_payload_missing_count = 0
accepted_update_payload_count = accepted_count
accepted_event_id_subset_of_candidate_event_id = 1
payload_hash_present = 1
no payload for rejected rows = 1
```

and update payload contains:

```text
delta_theta block refs
functional_update_scale
branch_delta_summary
accept_reason
abstain_reason
reject_reason
payload hash
```

H4 失败标准：

```text
accepted rows cannot map to update payload；
payload generated for rejected rows；
payload uses proxy/source gap；
payload cannot be applied by manual update path。
```

## H5：payload binding 后 system controller 可 official

H5 成立标准：

```text
official_eligible = 1
system_legal_controller_pass = 1
full_online_row_binding = 1
full_online_payload_binding = 1
full_online_update_payload_binding = 1
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
Agreement_{\text{reference}}\geq0.90,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

H5 失败标准：

```text
payload binding closes but official_eligible remains 0；
decision metrics drift；
step ratio >1.50；
memory ratio >1.05；
support balance fails；
audit detects projection/proxy/CPU/offload。
```

---

# Part IV. Full-online payload architecture

## 1. Event table

Every event row must have:

```text
global_row_id
event_id
dataset
seed
step
batch_id
microbatch_id
sample_indices_hash
horizon
signal_stratum
event_family
family_id
bucket_id
role_id
branch_id
source_hash
```

Hard constraints:

```text
global_row_id unique
event_id unique
family_id non-null
bucket_id non-null
source_hash non-null
dataset_name not used in controller rule
```

## 2. Candidate payload table

For every PF5 candidate:

```text
candidate_event_id
candidate_global_row_id
candidate_index
candidate_rank
candidate_score
candidate_x_hash
candidate_y_hash
candidate_base_logits_hash
candidate_x_shape
candidate_y_shape
candidate_base_logits_shape
candidate_payload_materialized
candidate_payload_device
candidate_payload_dtype
candidate_payload_stride
```

Hard constraints:

```text
candidate_tensor_payload_missing_count = 0
candidate_event_id must map to event table event_id
candidate_global_row_id must map to event table global_row_id
candidate_x/y/logits payload device must be cuda unless explicitly diagnostic
candidate payload hash must be reproducible
```

## 3. Candidate branch logits table

For every PF5 candidate:

```text
candidate_event_id
candidate_global_row_id
branch_id
horizon
branch_logits_hash
control_logits_hash
branch_logits_shape
control_logits_shape
uses_candidate_only_forward
uses_full_row_forward
logit_max_abs_diff_vs_reference
branch_forward_time_ms
branch_forward_memory_MB
```

Hard constraints:

```text
candidate_branch_logits_missing_count = 0
uses_candidate_only_forward = 1
uses_full_row_forward = 0
logit_max_abs_diff <= 1e-6 on audit subset
```

## 4. Candidate true-delta logits table

For every PF5 candidate:

```text
candidate_event_id
candidate_global_row_id
branch_id
true_delta_logits_hash
selected_delta_logits_hash
delta_norm
bridge_required_delta_features
uses_true_branch_delta
uses_source_measured_gap
uses_formula_proxy
true_delta_time_ms
true_delta_memory_MB
```

Hard constraints:

```text
candidate_true_delta_logits_missing_count = 0
uses_true_branch_delta = 1
uses_source_measured_gap = 0
uses_formula_proxy = 0
```

## 5. Bridge decision table

For every PF5 candidate:

```text
candidate_event_id
candidate_global_row_id
bridge_lookup_id
bridge_score
bad_ucb
null_ucb
support_lcb
backfill_flag
trim_flag
accept_decision
accept_reason
reject_reason
abstain_reason
```

Hard constraints:

```text
frozen_bridge_full_online_pass = 1
uses_cpu_summary = 0
online_frontier_search_used = 0
accept_decision row-bound to candidate_event_id
```

## 6. Functional update payload table

For every accepted row:

```text
accepted_event_id
accepted_global_row_id
functional_update_payload_id
delta_theta_block_ref
delta_theta_norm
delta_theta_shape
functional_update_scale
branch_delta_summary_hash
bridge_score
accept_reason
payload_device
payload_dtype
payload_hash
manual_update_ready
```

Hard constraints:

```text
functional_update_payload_missing_count = 0
accepted_event_id subset of candidate_event_id
no update payload for rejected rows
manual_update_ready = 1
payload uses true branch-delta, not source gap / formula proxy
```

## 7. End-to-end row binding equation

For each accepted event $e$:

$$
event\_id(e)
=
candidate\_event\_id(e)
=
branch\_logits\_event\_id(e)
=
true\_delta\_event\_id(e)
=
bridge\_event\_id(e)
=
update\_payload\_event\_id(e).
$$

If any equality fails, the row is not official.

---

# Part V. Candidate designs

## 1. Payload binding candidates

### PB0：v9.2.70 current diagnostic

Reference only. Expected failure:

```text
candidate payload missing = 2493
true-delta logits missing = 2493
update payload missing = 951
```

### PB1：CandidateXYBaseLogitsPayload

Materialize:

```text
candidate_x
candidate_y
candidate_base_logits
```

for all PF5 candidates.

### PB2：CandidateBranchLogitsPayload

Materialize:

```text
branch_logits
control_logits
branch metadata
```

for all PF5 candidates.

### PB3：CandidateTrueDeltaLogitsPayload

Materialize:

```text
true_delta_logits
selected_delta_logits
bridge_required_delta_features
```

for all PF5 candidates.

### PB4：AcceptedFunctionalUpdatePayload

Materialize:

```text
delta_theta_block_ref
functional_update_scale
payload hash
manual update readiness
```

for all accepted rows.

### PB5：EndToEndPayloadBinding

Full chain from event row to update payload.

## 2. Candidate tensor lifecycle candidates

### TL1：StoreFullCandidatePayload

Store all candidate tensors explicitly.

### TL2：StorePayloadRefsWithHash

Store references to batch tensors plus hash / offset / slice metadata.

### TL3：RecomputeCandidatePayloadFromStableIndices

Recompute payload using stable indices, but record deterministic reconstruction hash.

### TL4：HybridPayloadRefAndMaterializedLogits

Store x/y refs, materialize logits and true-delta.

Decision rule:

```text
TL1 is easiest to audit.
TL2/TL3 may be faster but must prove reconstruction identity.
Official candidate can use TL2/TL3 only if hash/equality audit passes.
```

## 3. Branch forward candidates

### BF1：FullOnlineCandidatePackedForwardV2

Run candidate-only branch forward on PB1 payload.

### BF2：FamilyGroupedCandidateForwardV2

Group candidates by family / branch / horizon.

### BF3：PersistentWorkspaceCandidateForwardV2

Preallocate candidate tensors to reduce allocation/sync.

### BF4：TritonCandidatePackForwardV2

Fused gather/scatter candidate pack.

### BF5：HybridGroupedPersistentForwardV2

Combine grouping and persistent workspace.

## 4. True-delta candidates

### TD1：FullOnlineCandidateOnlyTBD0V2

Exact true branch-delta for all PF5 candidates.

### TD2：SelectedBridgeLogitsWithFullAudit

Use selected bridge logits, with full-logit audit subset.

### TD3：TwoPassBorderlineTrueDelta

Selected logits first, full logits for borderline candidates.

### TD4：FusedCandidateDeltaV2

Fused candidate delta kernel.

### TD5：PersistentWorkspaceTrueDeltaV2

Candidate-only true delta with preallocated workspace.

## 5. Update payload candidates

### UP0：NoUpdatePayloadDiagnostic

Reference failure candidate.

### UP1：DeltaThetaBlockRefPayload

Store block refs and scales, not full dense delta.

### UP2：SparseAcceptedDeltaPayload

Store sparse accepted event update payload.

### UP3：FusedApplyReadyPayload

Payload layout directly consumable by manual update kernel.

### UP4：TwoStagePayloadCommit

Stage 1 stores accepted row payload; Stage 2 commits only if all audits pass.

## 6. System candidates

### SYS0：v9.2.70 diagnostic system

Expected failure due payload missing.

### SYS1：PB1 + PB2 + PB3 + UP1 + FrozenC3Lookup

Primary conservative full-online payload path.

### SYS2：PayloadRefs + BranchLogits + TrueDelta + UP1

More memory-efficient if hash identity passes.

### SYS3：PB1 + FamilyGroupedForward + TD1 + UP1

Branch-forward optimized.

### SYS4：PB1 + PersistentWorkspaceForward + TD5 + UP1

Allocation/sync optimized.

### SYS5：PB1 + TD2 SelectedBridgeLogits + UP1

Cheaper, must prove agreement.

### SYS6：PB1 + TD3 TwoPassBorderline + UP1

Balance exactness and cost.

### SYS7：HybridBestPayloadBoundSystem

Best combination after P2-P5.

---

# Part VI. 实验阶段

## P0：v9.2.70 boundary reproduction

### 目标

确认 v9.2.70 boundary 稳定，避免基于偶然 payload missing 状态制定 route。

### 必须记录

```text
route
source_route_v9270
reference_controller_id
reference_controller_reproduced
reference_precision
reference_coverage
reference_bad_event
reference_null_rate
reference_precision_lcb
reference_bad_event_ucb
event_identity_binding_pass
pf5_full_online_selector_pass
candidate_pack_binding_pass
candidate_tensor_payload_missing_count
candidate_branch_logits_missing_count
candidate_true_delta_logits_missing_count
functional_update_payload_missing_count
fused_bridge_full_online_pass
system_legal_controller_pass
official_eligible
primary_blocker
fake_proxy_count
cpu_offload_used
```

### 判断标准

P0 pass：

```text
route = R14-FullOnlineBindingFail
reference_controller_id = C3-T2PlusBackfill
event_identity_binding_pass = 1
pf5_full_online_selector_pass = 1
fused_bridge_full_online_pass = 1
candidate payload missing > 0
system_legal_controller_pass = 0
fake/proxy/offload = 0
```

### 可视化

```text
p0_v9270_boundary_ladder.svg
p0_identity_vs_payload_binding_gap.svg
p0_missing_payload_dashboard.svg
```

---

## P1：payload binding contract audit

### 目标

把 v9.2.70 的 missing payload 分解到具体数据结构与生命周期，避免盲目写入一些 tensor 但仍无法 official。

### 必须记录

```text
component
required_for_official
current_materialized
current_full_binding
missing_count
duplicate_count
mismatch_count
source_tensor
target_table
hash_present
device
dtype
shape
lifetime_scope
implementation_file
blocking_reason
```

Components：

```text
online event table
PF5 selector output
candidate id tensors
candidate x payload
candidate y payload
candidate base logits payload
candidate branch logits payload
candidate control logits payload
candidate true-delta logits payload
candidate selected-delta logits payload
FrozenC3 lookup rows
bridge score
accept decision
accepted functional update payload
timing record
memory record
```

### 判断标准

P1 pass：

```text
all required components mapped = 1
all missing payload categories assigned implementation candidate
unknown blocker count = 0
```

P1 official-ready diagnostic：

```text
candidate_tensor_payload_missing_count = 0
candidate_branch_logits_missing_count = 0
candidate_true_delta_logits_missing_count = 0
functional_update_payload_missing_count = 0
```

### 可视化

```text
p1_payload_binding_matrix.svg
p1_missing_payload_sankey.svg
p1_tensor_lifecycle_timeline.svg
```

---

## P2：candidate tensor payload materialization

### 目标

对 `2493` 个 PF5 candidates 全量绑定 `x/y/base_logits`，关闭 candidate payload missing。

### 必须记录

```text
payload_candidate_id
event_count
candidate_count
candidate_rate
candidate_x_bound
candidate_y_bound
candidate_base_logits_bound
candidate_tensor_payload_missing_count
candidate_payload_hash_present
candidate_payload_device
candidate_payload_dtype
candidate_payload_shape
candidate_duplicate_count
candidate_missing_count
candidate_event_id_mismatch_count
candidate_pack_time_ms
candidate_pack_memory_MB
candidate_reconstruction_error
projection_used
posthoc_used_at_commit
dataset_name_used
```

### 判断标准

P2 pass：

```text
candidate_tensor_payload_missing_count = 0
candidate_x_bound = 1
candidate_y_bound = 1
candidate_base_logits_bound = 1
candidate_payload_hash_present = 1
candidate_duplicate_count = 0
candidate_missing_count = 0
candidate_event_id_mismatch_count = 0
projection_used = 0
posthoc_used_at_commit = 0
dataset_name_used = 0
```

Candidate gate preserved：

$$
CandidateRate\leq0.25,
$$

$$
Recall_{\text{reference-accepted}}\geq0.95.
$$

### 可视化

```text
p2_candidate_payload_binding_heatmap.svg
p2_candidate_payload_missing_before_after.svg
p2_candidate_pack_cost_memory.svg
p2_candidate_rate_recall_after_payload.svg
```

---

## P3：candidate branch logits payload materialization

### 目标

对所有 PF5 candidates 真实执行 candidate-only branch forward，并将 branch/control logits 全量绑定到 candidate rows。

### 必须记录

```text
branch_forward_id
payload_candidate_id
candidate_count
candidate_rate
uses_candidate_only_forward
uses_full_row_forward
candidate_branch_logits_missing_count
candidate_control_logits_missing_count
branch_logits_hash_present
control_logits_hash_present
branch_forward_time_ms_mean
branch_forward_time_ms_q90
branch_forward_ratio_vs_full
kernel_count
sync_count
allocation_count
read_MB
write_MB
logit_max_abs_diff_vs_full_reference
logit_rel_err
accept_agreement_after_forward
projection_used
```

### 判断标准

P3 pass：

```text
uses_candidate_only_forward = 1
uses_full_row_forward = 0
candidate_branch_logits_missing_count = 0
candidate_control_logits_missing_count = 0
branch_logits_hash_present = 1
control_logits_hash_present = 1
projection_used = 0
```

Numerical exactness：

$$
LogitMaxAbsDiff\leq10^{-6}
$$

or:

$$
AcceptAgreement_{\text{after-forward}}\geq0.99.
$$

Cost diagnostic：

$$
BranchForwardRatio_{\text{candidate/full}}\leq0.10.
$$

### 可视化

```text
p3_branch_logits_binding_before_after.svg
p3_candidate_forward_cost_full_online.svg
p3_logit_error_histogram.svg
p3_branch_forward_ratio_by_family_horizon.svg
```

---

## P4：candidate true-delta logits materialization

### 目标

对所有 PF5 candidates 生成 row-bound true-delta logits，关闭 true-delta logits missing。

### 必须记录

```text
true_delta_candidate_id
payload_candidate_id
branch_forward_id
candidate_count
candidate_true_delta_logits_missing_count
selected_delta_logits_missing_count
uses_true_branch_delta
uses_source_measured_gap
uses_formula_proxy
true_delta_logits_hash_present
selected_delta_logits_hash_present
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
projection_used
full_trace_projection_used
```

### 判断标准

P4 pass：

```text
candidate_true_delta_logits_missing_count = 0
selected_delta_logits_missing_count = 0
uses_true_branch_delta = 1
uses_source_measured_gap = 0
uses_formula_proxy = 0
true_delta_logits_hash_present = 1
projection_used = 0
full_trace_projection_used = 0
```

Decision/system gate：

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

### 可视化

```text
p4_true_delta_logits_binding.svg
p4_true_delta_agreement_confusion.svg
p4_step_ratio_after_payload_binding.svg
p4_subphase_cost_waterfall_payload_bound.svg
```

---

## P5：accepted functional update payload materialization

### 目标

对所有 accepted rows 生成可被 manual update path 消费的 functional update payload。

### 必须记录

```text
update_payload_candidate_id
accepted_count
functional_update_payload_count
functional_update_payload_missing_count
accepted_event_id_missing_count
accepted_candidate_id_missing_count
payload_for_rejected_rows_count
delta_theta_block_ref_present
delta_theta_shape
delta_theta_norm
functional_update_scale
branch_delta_summary_hash
manual_update_ready
payload_device
payload_dtype
payload_hash_present
payload_apply_time_ms
payload_memory_MB
uses_true_branch_delta
uses_source_measured_gap
uses_formula_proxy
```

### 判断标准

P5 pass：

```text
functional_update_payload_missing_count = 0
functional_update_payload_count = accepted_count
accepted_event_id_missing_count = 0
accepted_candidate_id_missing_count = 0
payload_for_rejected_rows_count = 0
delta_theta_block_ref_present = 1
manual_update_ready = 1
payload_hash_present = 1
uses_true_branch_delta = 1
uses_source_measured_gap = 0
uses_formula_proxy = 0
```

Payload apply diagnostic：

$$
PayloadApplyTimeRatio\leq0.10.
$$

### 可视化

```text
p5_update_payload_binding.svg
p5_accepted_rows_payload_coverage.svg
p5_update_payload_norm_distribution.svg
p5_payload_apply_cost.svg
```

---

## P6：end-to-end full-online payload-bound controller

### 目标

组合 P2-P5，建立真正 payload-bound full-online system controller。

### 必须记录

```text
controller_id
payload_candidate_id
branch_forward_id
true_delta_candidate_id
update_payload_candidate_id
prefilter_id
bridge_system_id
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
accepted_strata_count
accepted_family_count
max_family_share
max_stratum_share
AUC_safe_good
AUC_bridge_accept
agreement_reference_accept
step_ratio_q90
memory_ratio
candidate_tensor_payload_missing_count
candidate_branch_logits_missing_count
candidate_true_delta_logits_missing_count
functional_update_payload_missing_count
materialized_system_path
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
p6_payload_bound_system_controller_frontier.svg
p6_reference_vs_system_accept_overlap.svg
p6_payload_binding_audit_dashboard.svg
p6_cost_vs_quality_payload_bound.svg
p6_family_strata_balance.svg
```

---

## P7：leave-dataset-out / leave-stratum-out

### 目标

如果 P6 pass，证明 payload-bound system controller 不是 pooled calibration artifact，也不是 dataset-specific route。

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
payload_candidate_id
prefilter_id
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
  ShuffledBranchLogits
  ShuffledTrueDeltaLogits
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
ShuffledBranchLogits = fail
ShuffledTrueDeltaLogits = fail
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
  ShuffledCandidatePayload
  ShuffledTrueDeltaLogits
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
Functional not explained by shuffled controller/payload
```

---

# Part VII. Required artifacts

```text
run_manifest.json
contract_audit_v9271.csv
p0_v9270_boundary_reproduction.csv
p1_payload_binding_contract_audit.csv
p2_candidate_tensor_payload_materialization.csv
p3_candidate_branch_logits_payload_materialization.csv
p4_candidate_true_delta_logits_materialization.csv
p5_accepted_functional_update_payload_materialization.csv
p6_payload_bound_system_legal_controller.csv
p7_leave_dataset_and_stratum_out.csv
p8_official_paired_replay.csv
p9_short_run_functional_validation.csv
p10_full_run_robustness_strong_baseline.csv

full_online_event_table_v9271.csv
candidate_payload_table_v9271.csv
candidate_branch_logits_table_v9271.csv
candidate_true_delta_logits_table_v9271.csv
bridge_decision_table_v9271.csv
functional_update_payload_table_v9271.csv
end_to_end_payload_binding_trace_v9271.csv
payload_lifecycle_trace_v9271.csv
system_controller_trace_v9271.csv
leaveout_trace_v9271.csv
paired_replay_branch_trace_v9271.csv
short_run_trace_v9271.csv

route_decision.json
aggregate_decision.json
failure_table.csv
artifact_hashes.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9270_boundary_unstable
F3_dataset_tuning_detected
F4_reference_controller_not_reproducible
F5_payload_binding_contract_incomplete
F6_candidate_tensor_payload_missing
F7_candidate_x_y_payload_mismatch
F8_candidate_base_logits_missing
F9_candidate_branch_logits_missing
F10_candidate_branch_forward_not_candidate_only
F11_candidate_branch_forward_numerical_mismatch
F12_candidate_true_delta_logits_missing
F13_true_delta_uses_source_gap_or_formula_proxy
F14_true_delta_agreement_fail
F15_true_delta_system_fail
F16_functional_update_payload_missing
F17_payload_for_rejected_rows
F18_update_payload_not_manual_ready
F19_payload_hash_missing
F20_system_controller_projection_used
F21_system_controller_precision_fail
F22_system_controller_coverage_fail
F23_system_controller_bad_event_fail
F24_system_controller_null_rate_fail
F25_system_controller_lcb_ucb_fail
F26_system_controller_payload_binding_fail
F27_system_controller_step_ratio_fail
F28_system_controller_memory_fail
F29_leave_dataset_out_fail
F30_leave_stratum_out_fail
F31_paired_replay_control_equivalent
F32_shuffle_control_pass
F33_functional_lr_equivalent
F34_short_run_task_drop
F35_full_run_no_macro_hard_stratum_gain
F36_strong_baseline_explains_gain
F37_robustness_fail
F38_external_not_ready
F39_fake_or_proxy_violation
F40_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-BoundaryReproduced:
  v9.2.70 boundary reproduced.

R2-PayloadBindingContractPass:
  all required candidate/logit/update payload fields are mapped and implementable.

R3-CandidateTensorPayloadPass:
  candidate x/y/base-logits payload is full-online materialized for all PF5 candidates.

R4-CandidateBranchLogitsPayloadPass:
  branch/control logits are full-online materialized for all PF5 candidates.

R5-CandidateTrueDeltaLogitsPayloadPass:
  true-delta logits are full-online materialized for all PF5 candidates.

R6-FunctionalUpdatePayloadPass:
  accepted update payload is materialized for all accepted rows.

R7-PayloadBoundSystemLegalControllerPass:
  payload-bound system controller passes decision + compute + official eligibility gates.

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

R13-CandidatePayloadBindingFail:
  candidate metadata binds but candidate x/y/base-logits payload cannot bind.

R14-BranchLogitsBindingFail:
  candidate branch logits cannot bind to full-online candidates.

R15-TrueDeltaLogitsBindingFail:
  true-delta logits cannot bind or require proxy/source gap.

R16-UpdatePayloadBindingFail:
  accepted rows cannot generate manual-update-ready payload.

R17-PayloadBoundSystemStillExpensive:
  payload binding succeeds but step ratio exceeds gate.

R18-PayloadBoundSystemDecisionDrift:
  payload binding succeeds but decision metrics drift below gate.

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
v9270_boundary_pass
dataset_tuning_detected
reference_controller_id
reference_controller_reproduced
reference_precision
reference_coverage
reference_bad_event
reference_null_rate
reference_precision_lcb
reference_bad_event_ucb
payload_binding_contract_pass
candidate_tensor_payload_pass
candidate_tensor_payload_missing_count
candidate_x_bound
candidate_y_bound
candidate_base_logits_bound
candidate_branch_logits_payload_pass
candidate_branch_logits_missing_count
candidate_control_logits_missing_count
candidate_branch_forward_full_online_pass
candidate_forward_ratio
branch_logit_error_max
candidate_true_delta_logits_payload_pass
candidate_true_delta_logits_missing_count
selected_delta_logits_missing_count
uses_true_branch_delta
uses_source_measured_gap
uses_formula_proxy
true_delta_agreement
true_delta_step_ratio_q90
true_delta_memory_ratio
functional_update_payload_pass
functional_update_payload_missing_count
accepted_count
functional_update_payload_count
payload_for_rejected_rows_count
manual_update_ready
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
success_v9271_strict_purekan_functional
success_v9271_full_functional
success_v9271_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 payload binding contract audit
  P2 candidate tensor payload materialization
  P3 candidate branch logits materialization
  P4 candidate true-delta logits materialization
  P5 accepted functional update payload materialization

Batch 2:
  P6 payload-bound system-legal controller
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
  P1 payload binding contract pass
  P2 candidate tensor payload pass
  P3 branch logits payload pass
  P4 true-delta logits payload pass
  P5 functional update payload pass
  projection_used = 0
  full_online_row_binding = 1
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
  full-online row binding pass
  candidate payload binding pass
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
v9.2.70 boundary reproduced
payload binding contract audited
candidate tensor payload materialized or failure localized
candidate branch logits materialized or failure localized
candidate true-delta logits materialized or failure localized
functional update payload materialized or failure localized
system controller measured
no fake/proxy/offload/loss/teacher violation
```

## Payload binding success

```text
Minimum diagnostic success
+
candidate_tensor_payload_missing_count = 0
+
candidate_branch_logits_missing_count = 0
+
candidate_true_delta_logits_missing_count = 0
+
functional_update_payload_missing_count = 0
+
payload hashes present
+
row/event/candidate/update ids consistent
```

## System success

```text
Payload binding success
+
official_eligible = 1
+
system_legal_controller_pass = 1
+
agreement_reference_accept >= 0.90
+
step_ratio_q90 <= 1.50
+
memory_ratio <= 1.05
+
decision gates pass
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
1. v9.2.70 boundary cannot be reproduced；
2. reference controller becomes unstable；
3. payload binding contract cannot be completed；
4. candidate tensor payload remains missing；
5. candidate branch logits remain missing；
6. true-delta logits remain missing；
7. accepted functional update payload remains missing；
8. payload hashes missing or inconsistent；
9. payload id binding produces duplicate/missing/mismatch rows；
10. branch forward silently reverts to full-row replay；
11. true-delta uses source gap / formula proxy；
12. payload-bound true-delta loses agreement；
13. payload-bound system exceeds step ratio 1.50；
14. payload-bound system exceeds memory ratio 1.05；
15. system controller fails precision / coverage / bad-event / null-rate；
16. precision LCB below 0.75；
17. bad-event UCB above 0.05；
18. leave-dataset-out fails；
19. leave-stratum-out fails；
20. paired replay remains control-equivalent；
21. shuffle controls pass；
22. short-run task drops；
23. full run gives no macro / hard-stratum / geometry gain；
24. functional breaks system gate；
25. gains are explained by QuadraticFeatureMLP；
26. any teacher/loss/fake/proxy/offload/projection-as-pass violation occurs。
```

---

# Part XI. 最终解释规则

## Case A：payload-bound system controller + LDO/LSO + paired replay pass

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full run and robustness。

## Case B：payload binding pass but compute fail

必须声明：

```text
full-online payload binding is correct, but payload-bound true-delta implementation remains too expensive.
```

下一步继续 candidate packing / branch-forward / fused-delta kernelization，不调 decision frontier。

## Case C：candidate payload binding fails

必须声明：

```text
PF5 candidate metadata is executable, but candidate tensor payload lifecycle is not yet official.
```

下一步修 event table / candidate pack / payload storage，不调 controller。

## Case D：true-delta logits binding fails

必须声明：

```text
true-delta microprobe cannot yet be promoted because full candidate true-delta logits are not row-bound.
```

下一步修 branch logits / delta logits tensor lifecycle，不用 proxy。

## Case E：functional update payload fails

必须声明：

```text
controller can accept rows, but accepted rows cannot yet produce manual-update-ready functional payload.
```

下一步修 update payload schema and apply path。

## Case F：compute pass but paired replay fail

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

v9.2.71 的一句话策略是：

$$
\boxed{
\text{不要再调 controller；把 2493 个 candidates 和 951 个 accepted rows 的 payload 全量绑定，让 system controller 转正。}
}
$$

当前最关键的问题不是：

```text
oracle 是否存在；
reference frontier 是否 deployable；
PF5 candidate rate 是否可行；
bridge lookup 是否可行；
microprobe step ratio 是否可行；
safe-useful target 是否可行；
C3/T2/C4/E2 是否还要重调；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. 2493 个 PF5 candidates 是否都有 x/y/base-logits payload？
2. 2493 个 PF5 candidates 是否都有 branch/control logits？
3. 2493 个 PF5 candidates 是否都有 true-delta logits？
4. 951 个 accepted rows 是否都有 manual-update-ready payload？
5. payload id 是否能从 event table 一路追踪到 update payload？
6. payload binding 后 step_ratio_q90 是否仍 <=1.50？
7. payload binding 后 decision metrics 是否仍保持 reference 水平？
8. official_eligible 是否能从 0 变成 1？
9. LDO/LSO 是否通过？
10. official paired replay 是否打过 AdamWParallel / bestLR？
```

v9.2.71 的结果将给出清晰分叉：

```text
if payload-bound system controller passes:
  open LDO/LSO and official paired replay.

if payload binding passes but compute fails:
  continue candidate-packing / branch-forward / fused-delta kernelization.

if candidate payload binding fails:
  repair tensor lifecycle / candidate pack storage.

if true-delta logits binding fails:
  repair branch logits / delta logits materialization.

if update payload binding fails:
  repair accepted-row functional update payload and manual apply path.

if compute passes but paired replay fails:
  system is legal, but functional causal advantage is insufficient.

if LDO/LSO fails:
  repair dataset-agnostic support/family reliability, not dataset-specific tuning.
```
