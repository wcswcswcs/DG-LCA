# DG-KAN v9.2.72 Payload-Bound System Cost Closure 与 Fused Candidate-Forward / True-Delta Pipeline 完整实验计划

> 本计划基于 v9.2.71 `Full-Online Payload Binding 与 Functional-Update Payload Closure` 的真实执行结果制定。  
> v9.2.71 的 terminal route 是：
>
> ```text
> route = R17-PayloadBoundSystemStillExpensive
> base_candidate = LQ-t2-h256
> success_v9271_strict_purekan_functional = False
> success_v9271_full_functional = False
> success_v9271_external_ready = False
> ```
>
> v9.2.71 的关键事实是：
>
> ```text
> Reference / controller:
>   reference_controller_id = C3-T2PlusBackfill
>   reference_controller_reproduced = 1
>   controller_precision = 0.8380281690140845
>   controller_coverage = 0.03130511463844797
>   controller_bad_event = 0.02464788732394366
>   controller_null_rate = 0.13028169014084506
>   controller_precision_lcb = 0.7907157243773478
>   controller_bad_event_ucb = 0.04999458813129621
>
> Payload binding:
>   payload_binding_contract_pass = 1
>   candidate_tensor_payload_missing_count = 0
>   candidate_branch_logits_missing_count = 0
>   candidate_control_logits_missing_count = 0
>   candidate_true_delta_logits_missing_count = 0
>   selected_delta_logits_missing_count = 0
>   functional_update_payload_missing_count = 0
>   payload_for_rejected_rows_count = 0
>   full_online_payload_binding = 1
>   full_online_update_payload_binding = 1
>
> Candidate / PF5:
>   event_count = 24192
>   candidate_count = 2493
>   candidate_rate = 0.10305059523809523
>   reference_accept_recall = 1.0
>   candidate_pack_time_ms = 608.7826690636575
>   candidate_pack_memory_MB = 484.4794921875
>
> Branch logits:
>   branch_forward_id = BF1-FullOnlineCandidatePackedForwardV2
>   branch_forward_time_ms_mean = 1.4103165509405622
>   branch_forward_time_ms_q90 = 1.4828811399638653
>   branch_forward_ratio_vs_full = 0.10305059523809523
>   logit_max_abs_diff_vs_full_reference = 0.0
>   state_reproduction_extra_non_candidate_forward_count = 902
>
> True-delta:
>   true_delta_candidate_id = TD1-FullOnlineCandidateOnlyTBD0V2
>   uses_true_branch_delta = 1
>   uses_source_measured_gap = 0
>   uses_formula_proxy = 0
>   agreement_reference_accept = 1.0
>   AUC_safe_good = 0.8790720756550714
>   AUC_bridge_accept = 0.8113610999018152
>   step_ratio_q90 = 4.248161404287591
>   memory_ratio = 1.0
>   true_delta_system_pass = 0
>
> Functional update payload:
>   update_payload_candidate_id = UP1-DeltaThetaBlockRefPayload
>   accepted_count = 951
>   functional_update_payload_count = 951
>   functional_update_payload_missing_count = 0
>   delta_theta_block_ref_present = 1
>   manual_update_ready = 1
>   payload_apply_time_ms = 0.7214542036654759
>   payload_memory_MB = 746.68359375
>
> Optimized exact-W2 / hybrid attempt:
>   exact-W2 identity max error = 1.9073486328125e-06
>   hybrid branch_forward_q90 = 1.3954988680779934
>   hybrid true_delta_step_ratio_q90 = 4.53819861019958
>   optimized system pass = 0
>
> Current blocker:
>   payload_bound_true_delta_step_ratio_fail
>   next_required_implementation = optimize_payload_bound_candidate_forward
> ```
>
> v9.2.72 的核心判断是：
>
> $$
> \boxed{
> \text{v9.2.71 已经关闭 payload binding；现在失败不是 payload 缺失，而是 payload-bound full-online path 的 step-level cost。}
> }
> $$
>
> 因此 v9.2.72 不应该继续做：
>
> ```text
> 继续调 C3/T2/C4/E2 decision frontier；
> 继续重设 safe-useful / null / bad-event target；
> 继续扩 support rows；
> 继续把 exact-W2 / hybrid 的局部 branch-forward 改善当成 system closure；
> 继续在 Python-level algebra patch 上小修小补；
> 按 dataset 单独调 prefilter / threshold / binding rule。
> ```
>
> 本轮要回答的是：
>
> $$
> \boxed{
> \text{为什么 branch-forward 局部 q90 只有约 1.4ms，但 full payload-bound step ratio 却是 4.25 到 4.54？}
> }
> $$
>
> 也就是说，v9.2.72 的目标不是再证明 signal，不是再证明 payload，而是做 **step-level cost closure**：找出 full-online payload-bound path 中 candidate pack、state reproduction extra forwards、true-delta construction、hash/norm bookkeeping、sync/allocation、payload apply、CPU/GPU transfer 的真实主成本，并把它们 fused / async / cached / preallocated 到 system envelope 以内。

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

v9.2.72 的新增硬约束：

```text
payload_binding_contract_pass must remain 1
candidate_tensor_payload_missing_count must remain 0
candidate_branch_logits_missing_count must remain 0
candidate_true_delta_logits_missing_count must remain 0
functional_update_payload_missing_count must remain 0
full_online_payload_binding must remain 1
full_online_update_payload_binding must remain 1
uses_true_branch_delta must remain 1
uses_source_measured_gap must remain 0
uses_formula_proxy must remain 0
projection_used must remain 0
full_trace_projection_used must remain 0
cpu_offload_used must remain 0
online_frontier_search_used must remain 0
uses_cpu_summary must remain 0
```

Official system pass 不允许依赖：

```text
projected step ratio
microprobe-only timing
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
preallocated GPU workspaces
fused CUDA/Triton kernels
async GPU timing and stream-aware scheduling
calibration-split frozen thresholds
```

---

# Part I. 对 v9.2.71 的独立判断

## 1. v9.2.71 没有达到最终目标

v9.2.71 没有 strict PureKAN functional success，原因是：

```text
system_legal_controller_pass = 0
official_eligible = 0
true_delta_system_pass = 0
LDO / LSO = not_run
official paired replay = not_run
short-run / full-run = not_run
```

这不是因为 reference controller 不过，也不是因为 payload binding 不过。v9.2.71 的失败点比 v9.2.70 更后移：

```text
v9.2.70:
  payload missing:
    candidate tensor payload missing = 2493
    branch logits missing = 2493
    true-delta logits missing = 2493
    update payload missing = 951

v9.2.71:
  payload missing 已全部归零；
  decision metrics 仍保持 reference frontier；
  但 payload-bound true-delta step ratio = 4.248161 > 1.50。
```

因此，不能声明：

```text
strict PureKAN local causal evidence
system-legal functional controller
LDO / LSO success
paired replay success
short-run success
full-run success
external-ready success
```

但也不能把 v9.2.71 说成没有进展。它完成了一个此前一直卡住的关键系统合同：full-online payload binding。

## 2. v9.2.71 的真实进展

v9.2.71 的真实进展很大，主要有四点。

第一，candidate payload 完成闭合。P2 中 `candidate_tensor_payload_missing_count = 0`，并且 `candidate_x_bound = 1`、`candidate_y_bound = 1`、`candidate_base_logits_bound = 1`。这说明 PF5 candidates 不再只是 metadata，而是真正有 x/y/base-logits payload。

第二，branch/control logits 完成闭合。P3 中 `candidate_branch_logits_missing_count = 0`、`candidate_control_logits_missing_count = 0`，并且 `logit_max_abs_diff_vs_full_reference = 0.0`。这说明 candidate-only branch forward 的数值没有漂移。

第三，true-delta logits 完成闭合。P4 中 `candidate_true_delta_logits_missing_count = 0`、`selected_delta_logits_missing_count = 0`、`uses_true_branch_delta = 1`，且 source gap / formula proxy 都是 0。它不是 proxy route。

第四，accepted functional update payload 完成闭合。P5 中 `functional_update_payload_count = 951`，与 `accepted_count = 951` 一致，`payload_for_rejected_rows_count = 0`，`manual_update_ready = 1`。这说明 accepted rows 已经能生成 update payload，不再只是 accept decision。

从路线角度看，这是从：

```text
full-online row identity bound
```

推进到了：

```text
full-online payload and update payload bound
```

这是 functional update 进入 official controller 前必须完成的一步。

## 3. v9.2.71 的真实失败

v9.2.71 的真实失败是：

$$
\boxed{
\text{payload-bound full-online path 太慢。}
}
$$

最关键的冲突是：

```text
局部 branch forward q90:
  1.482881 ms

true-delta payload-bound step ratio:
  4.248161

hybrid branch q90:
  1.395499 ms

hybrid step ratio:
  4.538199
```

这说明 branch-forward 局部 q90 并不是唯一主因。exact-W2/hybrid 把 branch forward q90 从 `1.482881` 降到 `1.395499`，但 step ratio 没有下降，反而维持在 `4.5` 附近。

所以当前不能继续只修：

```text
single-candidate branch forward
selected logits formula
exact-W2 algebra path
minor direct-forward identity correction
```

它们不解决 full step cost。

## 4. 当前 blocker 的本质

当前 blocker 应写成：

$$
\boxed{
\text{payload-bound full step contains expensive common-prep / materialization / sync / bookkeeping beyond branch forward.}
}
$$

更具体地说，v9.2.72 必须把 step ratio `4.248161` 分解为：

```text
1. PF5 selector cost；
2. candidate x/y/base-logits payload pack cost；
3. state reproduction extra non-candidate forward cost；
4. candidate branch/control forward cost；
5. true-delta logits construction cost；
6. selected-delta / bridge feature extraction cost；
7. candidate hash / norm / audit bookkeeping cost；
8. functional update payload construction cost；
9. payload apply cost；
10. sync/allocation/CPU-GPU transfer cost；
11. timing protocol / measurement overhead；
12. route artifact writing overhead accidentally included in step timing。
```

目前已有强烈迹象：

```text
candidate_pack_time_ms = 608.782669
payload_apply_time_ms = 0.721454
payload_memory_MB = 746.683594
state_reproduction_extra_non_candidate_forward_count = 902
```

这提示真正成本可能在 **candidate payload materialization / state reproduction / sync bookkeeping**，而不是纯 branch-forward math。

## 5. 是否还在正确道路上

是。路线是对的，但下一步必须从 “payload binding closure” 转为 “payload-bound cost closure”。

正确路线是：

```text
reference frontier pass
→ frozen controller pass
→ PF5 selector pass
→ payload binding pass
→ payload-bound cost attribution
→ fused / async / preallocated payload-bound candidate pipeline
→ system-legal controller
→ LDO / LSO
→ official paired replay
```

错误路线是：

```text
重新调 C3/T2PlusBackfill；
重设 safe-useful/null/bad target；
扩 support rows；
继续 exact-W2 局部 patch；
把 branch-forward q90 当 full system cost；
把 payload-bound decision metrics 写成 official pass；
按 dataset 单独调 candidate pack 或 threshold。
```

---

# Part II. v9.2.72 总体目标

v9.2.72 的总体目标是：

$$
\boxed{
\text{在保持 v9.2.71 payload binding 全部过关的前提下，把 full-online payload-bound step ratio 从 }4.248161\text{ 压到 }\leq1.50。
}
$$

本轮必须并行回答九个问题：

```text
Q1:
  v9.2.71 boundary 是否稳定复现？

Q2:
  step_ratio_q90 = 4.248161 的主要组成是什么？

Q3:
  candidate_pack_time_ms = 608.78 是否被误计入 per-step q90，还是确实每 step 发生？

Q4:
  902 个 state_reproduction_extra_non_candidate_forward 是否必要？
  是否可以通过 cached state / replay state table / detach checkpoint 消除？

Q5:
  candidate payload 是否可以从 materialized full tensor 改成 stable ref + hash，
  只在 true-delta kernel 中 gather needed slices？

Q6:
  true-delta logits / selected-delta / branch logits 是否可以 fused 成一个 candidate kernel，
  避免多次 materialization 和 hash/norm pass？

Q7:
  functional update payload 的 746MB payload memory 是否是 block-ref 设计错误，
  是否可以改成 compressed block-ref / lazy apply / accepted-only sparse payload？

Q8:
  sync/allocation/CPU bookkeeping 是否在 step q90 中占主导？

Q9:
  如果 cost closure 成功，system-legal controller 是否能打开 LDO/LSO 和 paired replay？
```

v9.2.72 的核心 stop-go 是：

$$
\boxed{
payload\_binding\_contract\_pass=1
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

## H1：v9.2.71 的 failure 是 step-cost attribution 缺失，不是 decision/payload failure

H1 成立标准：

```text
P0 reproduces:
  payload_binding_contract_pass = 1
  candidate_tensor_payload_missing_count = 0
  candidate_branch_logits_missing_count = 0
  candidate_true_delta_logits_missing_count = 0
  functional_update_payload_missing_count = 0
  controller metrics still pass decision gate
  step_ratio_q90 > 1.50
```

H1 失败标准：

payload binding 或 decision metrics 回退。若回退，v9.2.72 必须先修 payload stability，而不能进入 cost closure。

## H2：full step cost 主要来自 payload materialization / common-prep / sync，不是 branch-forward math

证据：

```text
branch q90 ≈ 1.48 ms
hybrid branch q90 ≈ 1.40 ms
first step ratio = 4.248
hybrid step ratio = 4.538
```

H2 成立标准：

在 P1 attribution 中：

```text
payload_pack + state_reproduction + sync/allocation + bookkeeping component ratio >= 0.50
```

或 branch-forward 组件占比：

```text
branch_forward_component_ratio <= 0.30
```

H2 失败标准：

branch forward 仍占 `>=0.50` 且 payload/materialization/sync 占比很低。若 H2 失败，则 v9.2.72 应重点做 branch-forward kernel，而不是 payload lifecycle。

## H3：candidate_pack_time_ms = 608.78 是 contract/audit materialization cost，不应每 step 全量发生

H3 成立标准：

通过 prepack / lazy-ref / persistent candidate payload table，把 online per-step candidate pack cost 降到：

$$
T_{\text{candidate-pack,q90}}\leq0.10T_{\text{step}}.
$$

或：

$$
CandidatePackTime_{\text{q90}}\leq5\text{ ms}
$$

以具体 MLP denominator 为准。

H3 失败标准：

candidate pack 必须每 step 复制 large payload，且不能降到 `<=0.10` step ratio。此时需要重构 event table / payload ref model。

## H4：902 个 extra non-candidate forward 是 state-reproduction artifact，可以被 eliminated 或 moved outside measured step

H4 成立标准：

```text
state_reproduction_extra_non_candidate_forward_count = 0
```

or if retained：

```text
extra_non_candidate_forward_time_ratio <= 0.05
```

并且 reference agreement 不下降。

H4 失败标准：

这些 extra forwards 是 functional update 合同必需，移除后 reference state 或 logits mismatch。若如此，系统成本必须重新定义为包含它们，并继续 fused state replay。

## H5：functional update payload memory 746MB 可压缩为 accepted sparse block refs

H5 成立标准：

```text
payload_memory_MB_after <= 128
payload_apply_time_ms_after <= 0.10 step time
functional_update_payload_missing_count = 0
manual_update_ready = 1
payload_hash_present = 1
```

H5 失败标准：

payload memory 必须保持数百 MB，或压缩后 update apply 不等价。

## H6：fused payload-bound candidate true-delta can preserve exact agreement

H6 成立标准：

至少一个 fused candidate satisfies:

$$
Agreement_{\text{reference}}\geq0.90,
$$

$$
AUC_{\text{bridge-accept}}\geq0.70,
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
NullRate\leq0.15.
$$

H6 失败标准：

fused path 降成本但改变 accept decisions，agreement `<0.90`。

## H7：system controller pass 后，LDO/LSO 与 paired replay 是真正 scientific gate

H7 成立标准：

P7/P8 opened and measured after P6 system pass.  
H7 失败标准：

system pass but LDO/LSO or paired replay fail. Then blocker moves from system implementation to functional causal advantage / dataset-agnostic reliability.

---

# Part IV. Step-level cost decomposition

## 1. 定义 total step cost

v9.2.72 必须定义：

$$
T_{\text{payload-bound-step}}
=
T_{\text{selector}}
+
T_{\text{candidate-pack}}
+
T_{\text{state-reproduction}}
+
T_{\text{branch-forward}}
+
T_{\text{true-delta}}
+
T_{\text{bridge-lookup}}
+
T_{\text{update-payload}}
+
T_{\text{payload-apply}}
+
T_{\text{audit-hash-norm}}
+
T_{\text{sync-allocation}}
+
T_{\text{other}}.
$$

官方 system gate：

$$
\frac{T_{\text{payload-bound-step,q90}}}{T_{\text{MLP-step,q90}}}\leq1.50.
$$

## 2. Attribution rows

Every measured candidate must write:

```text
selector_time_ms
candidate_pack_time_ms
state_reproduction_time_ms
extra_non_candidate_forward_time_ms
branch_forward_time_ms
true_delta_logits_time_ms
selected_delta_feature_time_ms
bridge_lookup_time_ms
update_payload_construct_time_ms
payload_apply_time_ms
hash_norm_audit_time_ms
sync_allocation_time_ms
artifact_write_time_ms
other_time_ms
unknown_fraction
```

Pass condition for attribution:

```text
unknown_fraction <= 0.10
dominant_component identified
at least one removable/fusible component ratio >= 0.25
```

## 3. Cost closure target

The full path must move from:

$$
StepRatio_{q90}=4.248161
$$

to:

$$
StepRatio_{q90}\leq1.50.
$$

Required reduction:

$$
4.248161-1.50=2.748161.
$$

Relative compression:

$$
\frac{1.50}{4.248161}\approx0.353.
$$

So v9.2.72 needs about `64.7%` reduction in q90 step ratio.

This cannot be achieved by shaving branch-forward q90 from `1.48` to `1.40` ms; it requires removing or fusing whole step-level components.

---

# Part V. Candidate designs

## 1. Cost attribution candidates

### CA0：v9.2.71 direct payload-bound reference

Reproduce first direct full-online payload-bound path:

```text
step_ratio_q90 = 4.248161
branch_forward_q90 = 1.482881
payload binding pass = 1
```

### CA1：Fine-grained CUDA event timers

Add CUDA event timing for every subphase:

```text
selector
candidate_pack
branch_forward
true_delta
bridge
update_payload
payload_apply
sync
hash/norm
```

### CA2：Wallclock-vs-CUDA split audit

Separate:

```text
cuda_kernel_time
cuda_event_time
python_wallclock_time
artifact_io_time
hash_time
cpu_summary_time
```

### CA3：Allocation / sync audit

Record:

```text
cudaMalloc count
torch.empty count
sync count
cudaEvent synchronize count
tensor clone count
contiguous call count
detach/copy count
cpu transfer count
```

### CA4：State reproduction audit

Measure the 902 extra forwards:

```text
extra_non_candidate_forward_count
extra_non_candidate_forward_time_ms
required_for_state_reproduction
can_cache_state
can_detach_checkpoint
```

### CA5：Payload memory pressure audit

Measure:

```text
candidate_payload_MB
branch_logits_payload_MB
true_delta_logits_payload_MB
update_payload_MB
peak_workspace_MB
persistent_workspace_MB
temporary_workspace_MB
```

---

## 2. Candidate pack optimization candidates

### CP0：v9.2.71 explicit materialized candidate payload

Reference. Expected high cost:

```text
candidate_pack_time_ms ≈ 608.78
candidate_pack_memory_MB ≈ 484.48
```

### CP1：StableRefCandidatePayload

Instead of storing full x/y/base logits tensors, store:

```text
batch_ref
sample_indices
offsets
strides
base_logits_ref
hash
```

Then gather inside branch-forward / true-delta kernel.

Pass condition:

```text
candidate_tensor_payload_missing_count = 0
payload_hash_present = 1
reconstruction_error = 0
candidate_pack_time_ratio <= 0.10
```

### CP2：PersistentCandidateWorkspace

Preallocate candidate payload workspace once:

```text
candidate_x_workspace
candidate_y_workspace
base_logits_workspace
branch_logits_workspace
true_delta_workspace
update_payload_workspace
```

No per-step allocation.

### CP3：FusedCandidateGather

Use Triton/CUDA fused gather for:

```text
candidate_x
candidate_y
base_logits
family/bucket ids
branch ids
```

Avoid Python indexing / repeated `.contiguous()` calls.

### CP4：LazyPayloadMaterialization

Only materialize candidate payload for rows passing PF5 and only fields required by downstream:

```text
x/y refs materialized as refs
branch logits materialized
true-delta logits materialized
update payload materialized only for accepted rows
```

### CP5：CandidatePackNoHashFastPath

Move expensive hash/norm audit out of timed system step; keep audit on sampled subset.

Official fast path must still record hashes but not inside step timer.

---

## 3. State reproduction candidates

### SR0：v9.2.71 source-state reproduction reference

Reference with:

```text
state_reproduction_extra_non_candidate_forward_count = 902
```

### SR1：CachedStateTable

Cache required non-candidate state once per batch:

```text
base logits
probe logits
family state
support state
```

Then candidate path does not forward non-candidates.

### SR2：DetachCheckpointState

Store state checkpoint before candidate confirmation and restore without non-candidate replay.

### SR3：TwoStreamStateReplay

Separate:

```text
stream A: AdamW-equivalent state update
stream B: candidate true-delta confirmation
```

Avoid replaying non-candidates for confirmation.

### SR4：NoExtraForwardAudit

Attempt to remove extra non-candidate forward entirely and compare:

$$
Agreement_{\text{reference}}\geq0.99.
$$

---

## 4. Branch / true-delta fusion candidates

### FD0：v9.2.71 direct candidate forward + true delta

Reference.

### FD1：FusedBranchForwardTrueDelta

Single kernel / compact module computes:

```text
candidate branch logits
control logits
true_delta logits
selected_delta logits
delta_norm
bridge_required_delta_features
```

### FD2：SelectedDeltaFullAudit

Compute selected delta logits for all candidates; full delta only on audit subset / borderline rows.

Pass only if:

$$
Agreement_{\text{reference}}\geq0.90.
$$

### FD3：TwoPassBorderlinePayloadBound

Pass 1:

```text
selected logits + cheap bridge
```

Pass 2:

```text
full logits only for borderline candidates
```

### FD4：FamilyGroupedFusedDelta

Group candidates by:

```text
family_id
branch_id
horizon
role
```

to reduce kernel launches and repeated state loads.

### FD5：PersistentWorkspaceFusedDelta

Preallocate all delta buffers; no allocation in timed section.

### FD6：AsyncFusedDeltaBridge

Run delta and bridge score in same stream with no sync until final accept decision.

---

## 5. Bridge / update payload candidates

### BU0：v9.2.71 bridge/update reference

Reference:

```text
bridge_lookup_time_ms = small
payload_apply_time_ms = 0.721454
payload_memory_MB = 746.683594
```

### BU1：CompressedUpdateBlockRef

Store update payload as:

```text
block_id
offset
scale
delta_norm
branch_delta_hash
```

not full dense tensors.

### BU2：AcceptedOnlySparseApply

Apply update only for accepted rows:

```text
accepted_count = 951
no payload for rejected rows
```

but avoid storing per-candidate full update buffers.

### BU3：FusedAcceptAndPayloadBuild

Bridge accept decision and update payload construction in one pass.

### BU4：LazyUpdateCommit

Build payload metadata first; materialize actual delta_theta only at commit.

### BU5：NoHashInTimedPath

Move payload hash to post-step audit subset, not q90 step timer.

---

## 6. System candidates

### SYS0：v9.2.71 direct full-online payload-bound reference

Expected fail:

```text
step_ratio_q90 = 4.248161
```

### SYS1：CostAttributedReference

No optimization, only fine attribution. Must explain unknown fraction.

### SYS2：StableRefCandidatePayload + direct true-delta

Targets candidate pack reduction.

### SYS3：PersistentCandidateWorkspace + direct true-delta

Targets allocation/sync reduction.

### SYS4：FusedCandidateGather + direct true-delta

Targets candidate pack kernelization.

### SYS5：NoExtraForwardStateReplay + direct true-delta

Targets 902 extra forward removal.

### SYS6：FusedBranchForwardTrueDelta + compressed payload

Targets branch + delta + payload fusion.

### SYS7：TwoPassBorderlinePayloadBound

Targets exactness/cost balance.

### SYS8：AsyncFusedDeltaBridge + LazyUpdateCommit

Targets sync and payload-apply overhead.

### SYS9：HybridBestCostClosedSystem

Best combination from CP/SR/FD/BU.

---

# Part VI. 实验阶段

## P0：v9.2.71 boundary reproduction

### 目标

确认 v9.2.71 boundary 稳定。特别是 payload binding 是否继续过，step ratio 是否仍在 `4.x` 区间。

### 必须记录

```text
route
source_route_v9271
reference_controller_id
reference_controller_reproduced
payload_binding_contract_pass
candidate_tensor_payload_missing_count
candidate_branch_logits_missing_count
candidate_true_delta_logits_missing_count
functional_update_payload_missing_count
candidate_rate
reference_accept_recall
controller_precision
controller_coverage
controller_bad_event
controller_null_rate
controller_precision_lcb
controller_bad_event_ucb
true_delta_agreement
true_delta_step_ratio_q90
true_delta_memory_ratio
system_legal_controller_pass
official_eligible
fake_proxy_count
cpu_offload_used
```

### 判断标准

P0 pass：

```text
route = R17-PayloadBoundSystemStillExpensive
payload_binding_contract_pass = 1
all missing payload counts = 0
controller metrics pass decision gate
true_delta_step_ratio_q90 > 1.50
fake/proxy/offload = 0
```

### 可视化

```text
p0_v9271_boundary_ladder.svg
p0_payload_pass_but_system_fail_dashboard.svg
p0_v9270_to_v9271_progress.svg
```

---

## P1：payload-bound step cost attribution

### 目标

定位 `step_ratio_q90 = 4.248161` 的主成本。P1 是本轮最关键诊断门；如果不先知道主成本，后续 fusion 可能继续修错地方。

### 必须记录

```text
cost_candidate_id
selector_time_ms
candidate_pack_time_ms
state_reproduction_time_ms
extra_non_candidate_forward_time_ms
extra_non_candidate_forward_count
branch_forward_time_ms
true_delta_logits_time_ms
selected_delta_feature_time_ms
bridge_lookup_time_ms
update_payload_construct_time_ms
payload_apply_time_ms
hash_norm_audit_time_ms
sync_allocation_time_ms
artifact_write_time_ms
other_time_ms
total_step_time_ms
mlp_step_time_ms
step_ratio_q90
component_ratio_selector
component_ratio_candidate_pack
component_ratio_state_reproduction
component_ratio_branch_forward
component_ratio_true_delta
component_ratio_bridge
component_ratio_update_payload
component_ratio_payload_apply
component_ratio_hash_norm
component_ratio_sync_allocation
unknown_fraction
```

### 判断标准

P1 pass：

```text
unknown_fraction <= 0.10
dominant_component identified
at least one removable_or_fusible_component_ratio >= 0.25
```

Dominant cases：

```text
if candidate_pack dominates:
  route = R2-CandidatePackDominant

if state_reproduction dominates:
  route = R3-StateReproductionDominant

if sync/allocation dominates:
  route = R4-SyncAllocationDominant

if true_delta dominates:
  route = R5-TrueDeltaComputeDominant

if update_payload dominates:
  route = R6-UpdatePayloadDominant
```

### 可视化

```text
p1_step_cost_waterfall.svg
p1_component_ratio_stacked_bar.svg
p1_cuda_vs_wallclock_split.svg
p1_allocation_sync_count.svg
p1_candidate_pack_vs_branch_forward.svg
```

---

## P2：candidate payload lifecycle optimization

### 目标

解决 candidate pack / payload materialization 的潜在主成本，特别是 `candidate_pack_time_ms = 608.782669` 是否每 step 发生，以及如何把它移出 timed path 或压成 fused gather。

### 必须记录

```text
candidate_payload_candidate_id
implementation
payload_storage_mode
candidate_count
candidate_rate
candidate_x_bound
candidate_y_bound
candidate_base_logits_bound
candidate_payload_hash_present
candidate_tensor_payload_missing_count
candidate_pack_time_ms_mean
candidate_pack_time_ms_q90
candidate_pack_time_ratio
candidate_pack_memory_MB
candidate_reconstruction_error
candidate_event_id_mismatch_count
candidate_duplicate_count
candidate_missing_count
allocation_count
contiguous_call_count
cpu_transfer_count
projection_used
posthoc_used_at_commit
dataset_name_used
```

### 判断标准

P2 pass：

```text
candidate_tensor_payload_missing_count = 0
candidate_x/y/base_logits bound = 1
candidate_event_id_mismatch_count = 0
candidate_duplicate_count = 0
candidate_missing_count = 0
projection_used = 0
```

Cost pass：

$$
CandidatePackTimeRatio\leq0.10
$$

or:

$$
CandidatePackTime_{q90}\leq5\text{ ms}
$$

whichever is more conservative under measured MLP denominator.

### 可视化

```text
p2_candidate_payload_cost_before_after.svg
p2_candidate_pack_memory_before_after.svg
p2_payload_storage_mode_pareto.svg
p2_candidate_reconstruction_error.svg
```

---

## P3：state reproduction / extra non-candidate forward elimination

### 目标

处理 `state_reproduction_extra_non_candidate_forward_count = 902`。如果这些 forward 是为了复现 source train-stream state，需要判断是否能通过 cache/checkpoint/stream separation 消除。

### 必须记录

```text
state_reproduction_candidate_id
extra_non_candidate_forward_count_before
extra_non_candidate_forward_count_after
extra_non_candidate_forward_time_ms
state_cache_used
detach_checkpoint_used
two_stream_replay_used
state_reconstruction_error
reference_state_agreement
reference_accept_agreement
controller_precision
controller_coverage
controller_bad_event
controller_null_rate
step_ratio_q90
memory_ratio
```

### 判断标准

P3 pass：

```text
extra_non_candidate_forward_count_after = 0
```

or:

$$
ExtraForwardTimeRatio\leq0.05.
$$

Agreement must satisfy:

$$
Agreement_{\text{reference-accept}}\geq0.95
$$

for state-reproduction-only change.

### 可视化

```text
p3_extra_forward_before_after.svg
p3_state_reconstruction_error.svg
p3_state_reproduction_cost_waterfall.svg
p3_agreement_vs_extra_forward_removed.svg
```

---

## P4：fused branch-forward / true-delta / bridge candidate pipeline

### 目标

把 branch/control logits、true-delta logits、selected-delta features、bridge-required features 合并，避免多次 materialization、hash/norm、sync。

### 必须记录

```text
fused_delta_candidate_id
candidate_payload_candidate_id
state_reproduction_candidate_id
uses_true_branch_delta
uses_source_measured_gap
uses_formula_proxy
candidate_branch_logits_missing_count
candidate_true_delta_logits_missing_count
selected_delta_logits_missing_count
branch_forward_time_ms_q90
true_delta_time_ms_q90
bridge_feature_time_ms_q90
fused_kernel_time_ms_q90
kernel_count
sync_count
allocation_count
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
```

### 判断标准

P4 pass：

```text
uses_true_branch_delta = 1
uses_source_measured_gap = 0
uses_formula_proxy = 0
all missing logits counts = 0
```

Decision gate：

$$
Agreement_{\text{reference}}\geq0.90,
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
BadEventRate_{\text{UCB}}\leq0.05.
$$

System diagnostic pass：

$$
StepRatio_{q90}\leq2.00.
$$

Full system pass requires `<=1.50` and is checked in P6.

### 可视化

```text
p4_fused_delta_cost_signal_pareto.svg
p4_reference_agreement_confusion.svg
p4_kernel_count_before_after.svg
p4_sync_allocation_before_after.svg
```

---

## P5：compressed functional update payload and lazy apply

### 目标

解决 `payload_memory_MB = 746.683594` 和 payload apply cost。虽然 payload apply time 当前约 `0.72ms`，但 payload memory 和 update construction 可能引发 sync/allocation/step q90 失控。

### 必须记录

```text
update_payload_candidate_id
accepted_count
functional_update_payload_count
functional_update_payload_missing_count
payload_for_rejected_rows_count
payload_storage_mode
delta_theta_block_ref_present
delta_theta_dense_materialized
functional_update_scale
manual_update_ready
payload_hash_present
payload_memory_MB
payload_construct_time_ms
payload_apply_time_ms
payload_apply_time_ratio
payload_device
payload_dtype
agreement_reference_accept
controller_precision
controller_coverage
controller_bad_event
controller_null_rate
step_ratio_q90
memory_ratio
```

### 判断标准

P5 pass：

```text
functional_update_payload_missing_count = 0
functional_update_payload_count = accepted_count
payload_for_rejected_rows_count = 0
manual_update_ready = 1
payload_hash_present = 1
uses_true_branch_delta = 1
uses_source_measured_gap = 0
uses_formula_proxy = 0
```

Cost pass：

$$
PayloadMemoryMB\leq128
$$

or at least:

$$
PayloadMemoryReduction\geq0.50.
$$

Apply pass：

$$
PayloadApplyTimeRatio\leq0.10.
$$

### 可视化

```text
p5_update_payload_memory_before_after.svg
p5_payload_apply_time_before_after.svg
p5_payload_norm_distribution.svg
p5_accepted_payload_coverage.svg
```

---

## P6：payload-bound system-legal controller v4

### 目标

组合 P2-P5 的最佳 survivor，建立真正 official system controller。

### 必须记录

```text
controller_id
system_candidate_id
candidate_payload_candidate_id
state_reproduction_candidate_id
fused_delta_candidate_id
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
p6_payload_bound_system_ladder.svg
p6_reference_vs_system_accept_overlap.svg
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

验证 RealFunctional 是否在 strong controls 下有局部因果优势。P8 只有在 P6 system pass 与 P7 leave-out pass 后 official 打开。

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
  ShuffledStateReproduction
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
ShuffledStateReproduction = fail
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
contract_audit_v9272.csv
p0_v9271_boundary_reproduction.csv
p1_payload_bound_step_cost_attribution.csv
p2_candidate_payload_lifecycle_optimization.csv
p3_state_reproduction_extra_forward_elimination.csv
p4_fused_branch_forward_true_delta_bridge_pipeline.csv
p5_compressed_functional_update_payload_lazy_apply.csv
p6_payload_bound_system_legal_controller_v4.csv
p7_leave_dataset_and_stratum_out.csv
p8_official_paired_replay.csv
p9_short_run_functional_validation.csv
p10_full_run_robustness_strong_baseline.csv

payload_bound_step_cost_trace_v9272.csv
candidate_payload_optimization_trace_v9272.csv
state_reproduction_trace_v9272.csv
fused_delta_bridge_trace_v9272.csv
compressed_update_payload_trace_v9272.csv
system_controller_trace_v9272.csv
leaveout_trace_v9272.csv
paired_replay_branch_trace_v9272.csv
short_run_trace_v9272.csv

route_decision.json
aggregate_decision.json
failure_table.csv
artifact_hashes.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9271_boundary_unstable
F3_dataset_tuning_detected
F4_payload_binding_regression
F5_cost_attribution_unknown_fraction_high
F6_no_dominant_fusible_component
F7_candidate_pack_dominant_unfixed
F8_candidate_pack_reconstruction_mismatch
F9_state_reproduction_extra_forward_unfixed
F10_state_reproduction_agreement_fail
F11_fused_delta_agreement_fail
F12_fused_delta_uses_proxy_or_source_gap
F13_fused_delta_system_fail
F14_update_payload_compression_fail
F15_update_payload_apply_too_expensive
F16_hash_norm_audit_in_timed_path
F17_sync_allocation_dominant_unfixed
F18_system_controller_precision_fail
F19_system_controller_coverage_fail
F20_system_controller_bad_event_fail
F21_system_controller_null_rate_fail
F22_system_controller_lcb_ucb_fail
F23_system_controller_step_ratio_fail
F24_system_controller_memory_fail
F25_system_controller_projection_used
F26_leave_dataset_out_fail
F27_leave_stratum_out_fail
F28_paired_replay_control_equivalent
F29_shuffle_control_pass
F30_functional_lr_equivalent
F31_short_run_task_drop
F32_full_run_no_macro_hard_stratum_gain
F33_strong_baseline_explains_gain
F34_robustness_fail
F35_external_not_ready
F36_fake_or_proxy_violation
F37_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-BoundaryReproduced:
  v9.2.71 boundary reproduced.

R2-StepCostAttributed:
  payload-bound step cost attributed with unknown_fraction <= 0.10.

R3-CandidatePayloadLifecyclePass:
  candidate pack / payload materialization no longer dominates.

R4-StateReproductionPass:
  extra non-candidate forwards eliminated or made negligible.

R5-FusedDeltaBridgePass:
  fused branch-forward / true-delta / bridge pipeline preserves agreement and improves system cost.

R6-CompressedUpdatePayloadPass:
  accepted update payload memory/apply cost compressed without losing manual update readiness.

R7-SystemLegalExactSignalControllerPass:
  payload-bound system controller passes decision + compute gates.

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
  v9.2.71 payload binding no longer reproduces.

R14-CostAttributionIncomplete:
  step-level unknown_fraction remains >0.10.

R15-CandidatePackDominantStill:
  candidate pack / payload materialization remains dominant and unfixed.

R16-StateReproductionDominantStill:
  extra non-candidate forward remains dominant and necessary.

R17-FusedDeltaSignalLost:
  fused low-cost delta changes accept decisions.

R18-PayloadBoundSystemStillExpensive:
  decision and payload are correct but step_ratio_q90 remains >1.50.

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
v9271_boundary_pass
dataset_tuning_detected
reference_controller_id
reference_controller_reproduced
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
step_cost_attribution_pass
dominant_cost_component
unknown_fraction
selector_time_ms
candidate_pack_time_ms
state_reproduction_time_ms
extra_non_candidate_forward_count
extra_non_candidate_forward_time_ms
branch_forward_time_ms
true_delta_logits_time_ms
bridge_lookup_time_ms
update_payload_construct_time_ms
payload_apply_time_ms
hash_norm_audit_time_ms
sync_allocation_time_ms
best_candidate_payload_id
candidate_payload_lifecycle_pass
candidate_pack_time_ratio
best_state_reproduction_id
state_reproduction_pass
extra_non_candidate_forward_count_after
best_fused_delta_id
fused_delta_bridge_pass
fused_delta_agreement
fused_delta_step_ratio_q90
best_update_payload_id
compressed_update_payload_pass
payload_memory_MB
payload_apply_time_ratio
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
success_v9272_strict_purekan_functional
success_v9272_full_functional
success_v9272_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 payload-bound step cost attribution
  P2 candidate payload lifecycle optimization
  P3 state reproduction / extra forward elimination
  P4 fused branch-forward / true-delta / bridge pipeline
  P5 compressed update payload / lazy apply

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
P2/P3/P4/P5 can run in parallel after P1 attribution.
P6 cannot pass unless:
  P0 pass
  P1 attribution pass
  payload binding remains pass
  P2/P3/P4/P5 produce at least one cost-closed system candidate
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
v9.2.71 boundary reproduced
payload-bound step cost attributed
candidate payload optimization measured
state reproduction extra forward measured
fused delta/bridge measured
compressed update payload measured
system controller measured
no fake/proxy/offload/loss/teacher violation
```

## Cost attribution success

```text
Minimum diagnostic success
+
unknown_fraction <= 0.10
+
dominant cost component identified
+
removable/fusible component ratio >= 0.25
```

## System success

```text
Cost attribution success
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
1. v9.2.71 boundary cannot be reproduced；
2. payload binding regresses；
3. step cost attribution unknown_fraction >0.10；
4. no dominant/removable cost component found；
5. candidate pack remains dominant and cannot be reduced；
6. state reproduction extra forwards are necessary and too expensive；
7. fused delta loses reference agreement；
8. fused path uses source gap / formula proxy；
9. update payload compression fails；
10. payload apply or hash/norm audit remains in timed path；
11. sync/allocation remains dominant；
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

## Case B：payload binding remains pass but cost attribution points to candidate pack

必须声明：

```text
payload binding is correct, but candidate payload lifecycle dominates full step cost.
```

下一步修 candidate pack / stable ref / fused gather，不调 controller。

## Case C：state reproduction dominates

必须声明：

```text
PF5 candidate sparsity is not enough because non-candidate state reproduction dominates.
```

下一步修 cached state / detach checkpoint / two-stream replay。

## Case D：fused delta loses signal

必须声明：

```text
low-cost fused delta changes accept decisions and cannot replace true branch-delta.
```

下一步做 two-pass borderline or exact audit, not formula proxy.

## Case E：update payload dominates

必须声明：

```text
accepted update payload is correct but too heavy.
```

下一步做 sparse block refs / lazy commit / fused apply.

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

v9.2.72 的一句话策略是：

$$
\boxed{
\text{不要再调 decision 或 payload；先解释并压缩 full payload-bound step cost，把 }4.248161\text{ 拉到 }\leq1.50。
}
$$

当前最关键的问题不是：

```text
oracle 是否存在；
reference frontier 是否 deployable；
PF5 candidate rate 是否可行；
payload 是否 missing；
bridge lookup 是否可行；
true-delta signal 是否存在；
safe-useful target 是否可行；
C3/T2/C4/E2 是否还要重调；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. candidate_pack_time_ms = 608.78 到底是不是进入了 timed step？
2. 902 个 extra non-candidate forward 是否必要？
3. branch-forward q90 已经只有 1.48ms，为什么 step ratio 仍 4.25？
4. exact-W2/hybrid 降低 branch q90 后为什么 step ratio 反而仍 4.54？
5. true-delta logits / selected-delta / bridge features 是否多次 materialize？
6. hash/norm/audit 是否进入 timed step？
7. payload apply / payload memory 是否引发 sync/allocation？
8. 能否用 stable-ref payload + fused gather + persistent workspace 降低 candidate pack？
9. 能否用 cached state 消除 extra non-candidate forwards？
10. 能否用 fused true-delta/bridge/update payload pipeline 保持 agreement？
11. system controller 是否能 official eligible？
12. LDO/LSO 是否通过？
13. official paired replay 是否打过 AdamWParallel / bestLR？
```

v9.2.72 的结果将给出清晰分叉：

```text
if cost attribution identifies candidate pack:
  fix payload lifecycle / fused gather / stable refs.

if state reproduction dominates:
  fix cached state / checkpoint / two-stream replay.

if sync/allocation dominates:
  fix persistent workspace / async stream / no hash in timed path.

if fused pipeline passes system gate:
  open system-legal controller, LDO/LSO, paired replay.

if system pass but paired replay fails:
  system is legal, but functional causal advantage is insufficient.

if cost remains >1.50 after attribution and fusion:
  true-delta system path remains primary blocker; do not return to decision tuning.
```
