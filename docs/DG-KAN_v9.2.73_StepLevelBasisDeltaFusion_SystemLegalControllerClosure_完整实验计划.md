# DG-KAN v9.2.73 Step-Level Basis/Delta Fusion 与 System-Legal Controller Closure 完整实验计划

> 本计划基于 v9.2.72 `Payload-Bound System Cost Closure 与 Fused Candidate-Forward / True-Delta Pipeline` 的真实执行结果制定。v9.2.72 的 terminal route 是：
>
> ```text
> route = R18-PayloadBoundSystemStillExpensive
> base_candidate = LQ-t2-h256
> success_v9272_strict_purekan_functional = False
> success_v9272_full_functional = False
> success_v9272_external_ready = False
> ```
>
> v9.2.72 的关键事实是：payload binding、candidate payload、branch/control logits、true-delta logits、selected-delta logits、functional update payload 全部没有 regression；controller decision metrics 仍满足 reference frontier；candidate pack q90 只有 `0.279152 ms`；state reproduction extra non-candidate forward 已从 `902` 降到 `0`；compressed W2-only update payload memory 已降到 `9.287109 MB`。但 system envelope 仍未闭合：initial payload-bound path step q90 为 `4.457842809866833`，BF3 降到 `4.043819522772543`，BF4 降到 `2.969948325415854`，BF5 async combined basis+delta path 降到 `2.713295831053225`，仍高于 gate `1.50`。
>
> v9.2.73 的核心判断是：
>
> $$
> \boxed{\text{当前失败不是 decision / payload / state replay / update payload，而是 full-step basis-delta-selected-feature pipeline 太慢。}}
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
\theta_{t+1}=\theta_t+\Delta\theta_{\text{AdamW-equivalent}}+\Delta\theta_{\text{functional}}.
$$

任务目标保持标准 CE：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

v9.2.73 的新增硬约束：

```text
payload_binding_contract_pass = 1
candidate_tensor_payload_missing_count = 0
candidate_branch_logits_missing_count = 0
candidate_true_delta_logits_missing_count = 0
functional_update_payload_missing_count = 0
uses_true_branch_delta = 1
uses_source_measured_gap = 0
uses_formula_proxy = 0
projection_used = 0
full_trace_projection_used = 0
cpu_offload_used = 0
online_frontier_search_used = 0
uses_cpu_summary = 0
dataset_name_used = 0
```

允许按 dataset / family / horizon 诊断 cost distribution，但 official controller 和 official kernel dispatch 不能根据 dataset name 选择不同 threshold、不同 accept rule 或不同 kernel route。不同数据集只能用于定位问题，不能用于打榜式调参。

---

# Part I. 对 v9.2.72 的独立判断

## 1. v9.2.72 没有达到目标

v9.2.72 没有 strict PureKAN functional success。失败原因不是 controller decision，而是 system envelope：

```text
system_legal_controller_pass = 0
official_eligible = 0
true_delta_step_ratio_q90 = 2.713295831053225 after BF5
required step_ratio_q90 <= 1.50
```

P7/P8 downstream 仍不应打开。即使 controller precision / coverage / bad-event / null-rate 已经满足 reference frontier，system envelope 没过就不能写成 official functional controller。

## 2. v9.2.72 的真实进展

v9.2.72 有三层重要推进。

第一，payload 与 decision 没有 regression。候选 payload、branch logits、true-delta logits、functional update payload missing count 全为 0。Reference controller 的 decision metrics 也保持：

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
NullRate=0.130282.
$$

第二，step-cost attribution 真正闭合。unknown fraction 为 0，candidate pack q90 只有 `0.279152 ms`，state extra non-candidate forward 已从 `902` 降到 `0`，update payload memory 降到约 `9.287 MB`。这些都排除了之前的主要嫌疑。

第三，低层 kernel 有实质推进。BF4 把 candidate delta/logits total 从 BF3 的约 `1134 ms` 降到约 `111 ms`，并把 branch q90 降到约 `0.866 ms`。BF5 再把 step q90 从 `2.969948` 降到 `2.713296`，且 CUDA-vs-torch 数值对照仍正确。这说明 lower-level CUDA path 是有效方向，而不是虚假优化。

## 3. v9.2.72 的真实失败

v9.2.72 的失败不是“没有优化”，而是优化还没到系统阈值。最关键的是：

```text
BF2:
  step q90 = 4.457843
  dominant = fused_candidate_delta_logits_ms

BF3:
  step q90 = 4.043820
  dominant 仍偏 candidate delta/logits

BF4:
  step q90 = 2.969948
  dominant 转为 common_basis_prep_ms

BF5:
  step q90 = 2.713296
  dominant = fused_common_basis_delta_pipeline_ms
  true_delta_selected_feature_ms_total = 323.118709
```

这说明 blocker 已经移动：不是单独 delta/logits kernel 慢，而是 **common basis prep + delta + selected-feature + sync/materialization 的组合 pipeline 慢**。

## 4. 问题的本质

当前本质是：

$$
\boxed{\text{kernel 层已经局部有效，但 full step 仍被 per-step basis/feature materialization 与 selected-feature path 控制。}}
$$

BF4 之后不能再只说“candidate delta/logits 太慢”。BF4 已经证明 delta/logits 可以快；BF5 也把 common basis 与 delta 合并了一步。但 full step 仍是 `2.713`，说明还有至少三个未闭合层：

```text
1. fused_common_basis_delta_pipeline 本身仍太慢；
2. true_delta_selected_feature / hash / audit / row-feature materialization 仍有成本；
3. per-step kernel launch / sync / small-batch dynamic-shape overhead 仍可能存在。
```

如果 v9.2.73 继续在 Python 里重排 rows 或仅微调 BF5 threshold，就会变成小修小补。下一步必须把 BF5 拆成 GPU-kernel 内部阶段，并做 persistent / fused / static-bucket / selected-feature-elimination path。

---

# Part II. v9.2.73 总体目标

v9.2.73 的总体目标是：

$$
\boxed{\text{在保持 payload-bound true-delta exactness 和 C3-T2PlusBackfill decision metrics 的前提下，把 step_ratio_q90 从 }2.713296\text{ 压到 }\leq1.50。}
$$

本轮必须回答八个问题：

```text
Q1: v9.2.72 BF5 boundary 是否稳定复现？
Q2: BF5 fused_common_basis_delta_pipeline_ms 内部到底由哪些 kernel / memory movement / sync 组成？
Q3: true_delta_selected_feature_ms_total = 323.118709 是否可以被融合进 basis/delta kernel，或改成 accept-bit / bridge-score-only 输出？
Q4: 1250 个 fused_common_basis_delta_step 是否可以进一步 bucket/static-shape 合并，减少 launch overhead？
Q5: full-row payload hashing / selected-feature materialization / CPU metadata 是否仍进入 timed path？
Q6: persistent workspace / CUDA graph / static bucket capture 是否能在不改变 exactness 的前提下降低 q90？
Q7: system controller 是否能在 step_ratio_q90 <= 1.50 下保持 precision / coverage / bad-event / null-rate / LCB-UCB？
Q8: system pass 后，LDO/LSO 与 official paired replay 是否能打开？
```

v9.2.73 的核心 stop-go 是：

$$
\boxed{StepRatio_{q90}\leq1.50 \land Agreement_{\text{reference}}\geq0.90 \land OfficialEligible=1.}
$$

---

# Part III. 核心假设

## H1：remaining cost 主要来自 fused_common_basis_delta pipeline，而不是 payload/schema/decision

H1 成立标准：

```text
payload_binding_contract_pass = 1
candidate missing counts = 0
controller metrics pass decision gate
candidate_pack_q90 <= 0.50 ms
state_extra_non_candidate_forward_count = 0
dominant component remains fused_common_basis_delta_pipeline or selected-feature path
```

H1 失败标准：payload binding regression 或 decision metrics drift。若 H1 失败，不能继续 kernel fusion，必须先修 payload / decision stability。

## H2：selected-feature materialization 是 BF5 后的核心残余之一

BF5 中：

```text
true_delta_selected_feature_ms_total = 323.1187085621059
```

H2 认为这部分可以通过 bridge-score-in-kernel / accept-bit-only / audit-out-of-timed-path 降低。

H2 成立标准：

$$
T_{\text{selected-feature,total}} \leq 0.25 \cdot T_{\text{selected-feature,BF5}}
$$

并且：

$$
Agreement_{\text{reference}}\geq0.95.
$$

H2 失败标准：selected feature 是 true-delta exactness 必需且不能减少；此时需要把它和 basis/delta 融合，而不是省略。

## H3：1250 fused steps 存在 launch / small-batch overhead

H3 成立标准：

```text
kernel_launch_count decreases by >= 50%
sync_count decreases by >= 50%
step_ratio_q90 improves by >= 0.40
```

H3 失败标准：kernel launch / sync 很低，cost 来自 arithmetic / memory bandwidth；此时应做 arithmetic simplification / basis recurrence / memory coalescing。

## H4：common basis prep 可从 per-candidate/per-step recompute 变成 cached recurrence or fused row-feature transform

BF4 dominant 转为 common_basis_prep，BF5 fused total 仍 `1086.943672 ms`。H4 认为 common basis prep 中有重复的 t2 / lift / normalization / W2-specific prep。

H4 成立标准：

$$
T_{\text{common-basis-delta,total}} \leq 0.50 \cdot 1086.943672\text{ ms}
$$

with:

$$
\text{CUDA-vs-torch logits error}\leq5\times10^{-5},
$$

$$
\text{CUDA-vs-torch delta error}\leq10^{-8}.
$$

## H5：official system success 不能牺牲 true-delta exactness

任何 low-cost candidate 必须满足：

```text
uses_true_branch_delta = 1
uses_source_measured_gap = 0
uses_formula_proxy = 0
projection_used = 0
CUDA-vs-torch check count >= 24
agreement_reference_accept >= 0.90
```

---

# Part IV. Candidate designs

## 1. Cost attribution candidates

### CA0：BF5 boundary reproduction

复现 BF5：

```text
step_ratio_q90 = 2.713296
branch_q90 = 0.766299
fused_common_basis_delta_pipeline_ms_total = 1086.943672
true_delta_selected_feature_ms_total = 323.118709
```

### CA1：BF5 internal kernel timing

把 BF5 内部拆成：

```text
basis_lift_time
basis_quadratic_time
normalization_time
candidate_gather_time
W2_delta_time
probe_logits_time
selected_feature_time
bridge_feature_time
accept_bit_time
hash_norm_time
sync_time
launch_overhead
```

### CA2：GPU-vs-wallclock audit

分别记录：

```text
cuda_event_time_ms
python_wallclock_time_ms
artifact_write_time_ms
cpu_hash_norm_time_ms
cuda_synchronize_time_ms
```

### CA3：kernel launch / occupancy / memory bandwidth audit

记录：

```text
kernel_count
launch_count
avg_candidates_per_kernel
occupancy_proxy
bytes_read
bytes_written
effective_bandwidth
shared_memory_bytes
register_pressure_proxy
```

## 2. Basis / delta fusion candidates

### BD0：BF5 reference

Current best diagnostic path。

### BD1：SinglePassBasisDeltaFeatureKernel

一个 kernel 同时计算：

```text
common basis prep
candidate W2 delta
probe logits
selected-delta features
bridge partial score
```

输出：

```text
bridge_score
true_delta_selected_features if needed
accept-bit compatible features
```

### BD2：BasisRecurrenceCachedKernel

对 T2/quadratic/lift normalization 做 per-step shared cache：

```text
basis_cache[event_id, basis_id]
basis_norm_cache
basis_w2_cache
```

但 cache 必须 GPU-resident，且不能引入 CPU summary。

### BD3：FamilyGroupedBasisDeltaKernel

按 family / branch / horizon / bucket 分组，减少 dynamic shape 与 repeated basis prep。

### BD4：StaticBucketCudaGraphKernel

把 candidates bucket 到固定 shape：

```text
bucket_size in {1,2,4,8,16}
```

对每个 bucket 尝试 CUDA graph capture / replay。若 graph capture 失败，记录原因，不写 pass。

### BD5：PersistentWorkspaceBasisDeltaKernel

预分配：

```text
basis_workspace
delta_workspace
selected_feature_workspace
bridge_score_workspace
```

要求 timed path 内 allocation_count = 0。

## 3. Selected-feature / bridge-score candidates

### SF0：BF5 selected-feature reference

Reference path with selected feature materialization。

### SF1：BridgeScoreInsideKernel

在 kernel 内直接产生 bridge_score / accept pre-score，减少 selected feature writeback。

### SF2：AcceptBitInsideKernel

对非 borderline candidates 直接输出 accept/reject bit，只有 borderline rows 输出 full selected features。

### SF3：AuditOnlySelectedFeature

official timed path 不输出 full selected features；post-step audit subset 输出 selected features for verification。

Pass requires:

```text
audit subset agreement >= 0.99
full accept agreement >= 0.90
```

### SF4：TwoPassBorderlineFeature

Pass 1 produces cheap exact-compatible bridge score. Pass 2 materializes selected features only for borderline candidates。

## 4. Step-level async / sync candidates

### AS0：BF5 async reference

Reference。

### AS1：NoMidStepCpuNorm

确保 timed path 内没有 `.cpu()`、`.item()`、Python norm、hash、CSV append。

### AS2：StreamSeparatedAudit

Use separate stream for audit/hashing after accept decision; timed path only includes controller。

### AS3：PersistentEventTableNoCopy

Keep event/candidate/family/bucket tables GPU resident。

### AS4：AsyncBridgeAndPayloadCommit

Overlap bridge score with update payload commit where safe。

### AS5：NoHashInTimedPathWithPostAudit

Hashes recorded on sampled rows after timing; official pass requires no disagreement。

## 5. System candidates

```text
SYS0 = BF5 reference, expected step_ratio_q90 around 2.713296.
SYS1 = BD1 + SF1.
SYS2 = BD3 + SF1.
SYS3 = BD4 + SF2.
SYS4 = BD5 + SF3 + AS1.
SYS5 = BD2 + BD5 + SF4.
SYS6 = BD1 + AS2 + AS5.
SYS7 = HybridBestV9273.
```

---

# Part V. 实验阶段

## P0：v9.2.72 boundary reproduction

### 目标

确认 BF5 boundary 稳定。不能在不稳定或 regression 的基础上继续系统优化。

### 必须记录

```text
route
source_route_v9272
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
bf5_step_ratio_q90
bf5_branch_q90
bf5_cuda_vs_torch_logits_error_max
bf5_cuda_vs_torch_delta_error_max
system_legal_controller_pass
official_eligible
fake_proxy_count
cpu_offload_used
```

### 判断标准

```text
route = R18-PayloadBoundSystemStillExpensive
payload binding pass = 1
all missing payload counts = 0
true_delta_agreement >= 0.90
bf5_step_ratio_q90 > 1.50
fake/proxy/offload = 0
```

### 可视化

```text
p0_boundary_ladder.svg
p0_bf2_bf3_bf4_bf5_progress.svg
p0_decision_vs_system_status.svg
```

## P1：BF5 internal step cost attribution

### 目标

把 BF5 的 fused_common_basis_delta_pipeline 拆开，判断真正该融合/缓存/异步的是哪一段。

### 必须记录

```text
cost_candidate_id
basis_lift_time_ms
basis_quadratic_time_ms
basis_norm_time_ms
candidate_gather_time_ms
W2_delta_time_ms
probe_logits_time_ms
selected_feature_time_ms
bridge_score_time_ms
accept_bit_time_ms
hash_norm_time_ms
cpu_metadata_time_ms
cuda_sync_time_ms
kernel_launch_time_ms
allocation_time_ms
total_cuda_event_time_ms
total_wallclock_time_ms
unknown_fraction
kernel_count
sync_count
allocation_count
avg_candidates_per_kernel
bytes_read
bytes_written
```

### 判断标准

```text
unknown_fraction <= 0.05
dominant internal component identified
at least one removable/fusible component ratio >= 0.20
```

### 可视化

```text
p1_bf5_internal_cost_waterfall.svg
p1_cuda_vs_wallclock_split.svg
p1_kernel_launch_distribution.svg
p1_selected_feature_cost.svg
```

## P2：basis/delta/feature single-pass fusion

### 目标

把 common basis prep、candidate delta/logits、selected features、bridge score 继续下沉为更少 kernel 的 path。

### 必须记录

```text
basis_delta_candidate_id
uses_true_branch_delta
uses_source_measured_gap
uses_formula_proxy
kernel_count
sync_count
allocation_count
basis_delta_feature_fused
bridge_score_inside_kernel
accept_bit_inside_kernel
candidate_count
fused_step_count
cuda_vs_torch_check_count
cuda_vs_torch_logits_error_max
cuda_vs_torch_delta_error_max
selected_feature_error_max
agreement_reference_accept
AUC_bridge_accept
precision
coverage
bad_event
null_rate
precision_lcb
bad_event_ucb
branch_q90_ms
basis_delta_total_ms
selected_feature_total_ms
step_ratio_q90
memory_ratio
```

### 判断标准

Diagnostic pass:

```text
uses_true_branch_delta = 1
uses_source_measured_gap = 0
uses_formula_proxy = 0
cuda_vs_torch_check_count >= 24
cuda_vs_torch_logits_error_max <= 5e-5
cuda_vs_torch_delta_error_max <= 1e-8
agreement_reference_accept >= 0.90
step_ratio_q90 <= 2.00
```

System pass candidate:

$$
StepRatio_{q90}\leq1.50.
$$

### 可视化

```text
p2_basis_delta_fusion_pareto.svg
p2_error_vs_cost.svg
p2_kernel_count_before_after.svg
p2_step_ratio_before_after.svg
```

## P3：selected-feature materialization elimination

### 目标

验证 selected-feature 是否可以从 full materialized tensor 改成 bridge-score / accept-bit / audit-only path。

### 必须记录

```text
selected_feature_candidate_id
mode
selected_feature_materialized_count
borderline_count
audit_subset_count
bridge_score_inside_kernel
accept_bit_inside_kernel
selected_feature_total_ms_before
selected_feature_total_ms_after
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

```text
agreement_reference_accept >= 0.90
audit_agreement >= 0.99
selected_feature_total_ms_after <= 0.25 * selected_feature_total_ms_before
projection_used = 0
source_gap_used = 0
formula_proxy_used = 0
```

System candidate pass:

$$
StepRatio_{q90}\leq1.50
$$

with all decision gates satisfied。

### 可视化

```text
p3_selected_feature_cost_reduction.svg
p3_accept_agreement_confusion.svg
p3_borderline_distribution.svg
p3_selected_feature_materialization_count.svg
```

## P4：static bucket / persistent workspace / CUDA graph path

### 目标

减少 1250 fused steps 的 launch / dynamic-shape overhead。重点不是改变 controller，而是稳定候选 batch shape。

### 必须记录

```text
bucket_candidate_id
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
cuda_graph_attempted
cuda_graph_capture_pass
cuda_graph_failure_reason
persistent_workspace_used
workspace_memory_MB
agreement_reference_accept
cuda_vs_torch_logits_error_max
cuda_vs_torch_delta_error_max
step_ratio_q90
memory_ratio
```

### 判断标准

```text
kernel_count_after <= 0.50 * kernel_count_before
sync_count_after <= 0.50 * sync_count_before
allocation_count_after = 0
agreement_reference_accept >= 0.90
```

P4 system candidate pass:

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p4_bucket_kernel_count.svg
p4_bucket_step_ratio.svg
p4_cuda_graph_capture_status.svg
p4_workspace_memory.svg
```

## P5：async audit / no-hash timed path / logging separation

### 目标

确认 no-fake/no-proxy 的审计仍保留，但不把 full-row hash、CSV、CPU metadata、norm summary 放进 official step timer。

### 必须记录

```text
audit_candidate_id
hash_in_timed_path
csv_write_in_timed_path
cpu_metadata_in_timed_path
norm_item_in_timed_path
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

```text
hash_in_timed_path = 0
csv_write_in_timed_path = 0
cpu_metadata_in_timed_path = 0
audit_disagreement_count = 0
agreement_reference_accept >= 0.90
```

System candidate pass:

$$
StepRatio_{q90}\leq1.50.
$$

### 可视化

```text
p5_timed_vs_audit_path.svg
p5_hash_norm_before_after.svg
p5_wallclock_cuda_gap.svg
```

## P6：system-legal exact-signal controller v5

### 目标

组合 P2-P5 的最佳 survivor，建立 official system-legal controller。

### 必须记录

```text
controller_id
system_candidate_id
basis_delta_candidate_id
selected_feature_candidate_id
bucket_candidate_id
audit_candidate_id
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

Decision gate:

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

System gate:

$$
Agreement_{\text{reference}}\geq0.90,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

Support balance:

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
p6_step_ratio_progress_bf2_to_v9273.svg
p6_family_strata_balance.svg
```

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

LSO pass:

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

### 判断标准

$$
BeatRate_{\text{macro,Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro,Real vs bestLR}}\geq0.60.
$$

Task safety:

$$
Acc_{\text{each slice,Real}}\geq Acc_{\text{AdamW}}-0.005.
$$

System:

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

---

# Part VI. Required artifacts

```text
run_manifest.json
contract_audit_v9273.csv
p0_v9272_boundary_reproduction.csv
p1_bf5_internal_step_cost_attribution.csv
p2_basis_delta_feature_single_pass_fusion.csv
p3_selected_feature_materialization_elimination.csv
p4_static_bucket_persistent_workspace_cuda_graph.csv
p5_async_audit_nohash_timed_path.csv
p6_system_legal_exact_signal_controller_v5.csv
p7_leave_dataset_and_stratum_out.csv
p8_official_paired_replay.csv
p9_short_run_functional_validation.csv
p10_full_run_robustness_strong_baseline.csv
bf5_internal_cost_trace_v9273.csv
basis_delta_feature_kernel_trace_v9273.csv
selected_feature_elimination_trace_v9273.csv
static_bucket_workspace_trace_v9273.csv
async_audit_trace_v9273.csv
system_controller_trace_v9273.csv
leaveout_trace_v9273.csv
paired_replay_branch_trace_v9273.csv
short_run_trace_v9273.csv
route_decision.json
aggregate_decision.json
failure_table.csv
artifact_hashes.csv
figures/
```

Failure taxonomy:

```text
F1_contract_violation
F2_v9272_boundary_unstable
F3_dataset_tuning_detected
F4_payload_binding_regression
F5_decision_metric_regression
F6_bf5_internal_cost_unknown_fraction_high
F7_no_dominant_fusible_component
F8_basis_delta_kernel_error_fail
F9_basis_delta_kernel_agreement_fail
F10_basis_delta_kernel_system_fail
F11_selected_feature_elimination_agreement_fail
F12_selected_feature_cost_unfixed
F13_static_bucket_capture_fail
F14_kernel_launch_overhead_unfixed
F15_persistent_workspace_memory_fail
F16_hash_norm_or_cpu_metadata_in_timed_path
F17_sync_allocation_dominant_unfixed
F18_system_controller_precision_fail
F19_system_controller_coverage_fail
F20_system_controller_bad_event_fail
F21_system_controller_null_rate_fail
F22_system_controller_lcb_ucb_fail
F23_system_controller_step_ratio_fail
F24_system_controller_memory_fail
F25_system_controller_projection_used
F26_source_gap_or_formula_proxy_used
F27_leave_dataset_out_fail
F28_leave_stratum_out_fail
F29_paired_replay_control_equivalent
F30_shuffle_control_pass
F31_functional_lr_equivalent
F32_short_run_task_drop
F33_full_run_no_macro_hard_stratum_gain
F34_strong_baseline_explains_gain
F35_robustness_fail
F36_external_not_ready
F37_fake_or_proxy_violation
F38_artifact_missing
```

---

# Part VII. Route decision

```text
R1-BoundaryReproduced
R2-BF5InternalCostAttributed
R3-BasisDeltaFeatureFusionPass
R4-SelectedFeatureEliminationPass
R5-StaticBucketPersistentWorkspacePass
R6-AsyncAuditNoHashTimedPathPass
R7-SystemLegalExactSignalControllerPass
R8-LeaveDatasetOutPass
R9-LeaveStratumOutPass
R10-PairedReplayPass
R11-ShortRunFunctionalPass
R12-FullFunctionalPass
R13-PayloadBindingRegression
R14-BF5CostAttributionIncomplete
R15-BasisDeltaFeatureKernelStillTooSlow
R16-SelectedFeatureCostStillDominant
R17-LaunchSyncDominantStill
R18-SystemStillTooExpensive
R19-ComputePassButLeaveoutFail
R20-ComputePassButPairedReplayFail
R21-ExternalReady
```

`route_decision.json` 必须记录：

```text
route
v9272_boundary_pass
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
bf5_internal_cost_attribution_pass
unknown_fraction
dominant_internal_component
basis_lift_time_ms
basis_quadratic_time_ms
basis_norm_time_ms
candidate_gather_time_ms
W2_delta_time_ms
probe_logits_time_ms
selected_feature_time_ms
bridge_score_time_ms
hash_norm_time_ms
sync_time_ms
kernel_launch_time_ms
best_basis_delta_candidate_id
basis_delta_feature_fusion_pass
cuda_vs_torch_logits_error_max
cuda_vs_torch_delta_error_max
basis_delta_agreement
basis_delta_step_ratio_q90
best_selected_feature_candidate_id
selected_feature_elimination_pass
selected_feature_time_reduction
best_bucket_candidate_id
static_bucket_workspace_pass
kernel_count_reduction
sync_count_reduction
allocation_count_after
best_audit_candidate_id
async_audit_pass
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
success_v9273_strict_purekan_functional
success_v9273_full_functional
success_v9273_external_ready
```

---

# Part VIII. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 BF5 internal cost attribution
  P2 basis/delta/feature single-pass fusion
  P3 selected-feature materialization elimination
  P4 static bucket / persistent workspace / CUDA graph path
  P5 async audit / no-hash timed path

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

Gate rule:

```text
P2/P3/P4/P5 can run in parallel after P1 attribution.
P6 cannot pass unless:
  P0 pass
  P1 attribution pass
  payload binding remains pass
  at least one P2/P3/P4/P5 system candidate reaches step_ratio_q90 <= 1.50
  projection_used = 0
  source_measured_gap_used = 0
  formula_proxy_used = 0
  full_online_payload_binding = 1
  full_online_update_payload_binding = 1
```

---

# Part IX. 停止条件

## Minimum diagnostic success

```text
v9.2.72 boundary reproduced
BF5 internal cost attributed
basis/delta/feature fusion measured
selected-feature elimination measured
static bucket / persistent workspace measured
async audit / no-hash timed path measured
system controller measured
no fake/proxy/offload/loss/teacher violation
```

## Cost closure success

```text
Minimum diagnostic success
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
Cost closure success
+
LDO / LSO pass
+
official paired replay beats AdamWParallel / bestLR
```

## Failure stop

```text
1. v9.2.72 boundary cannot be reproduced；
2. payload binding regresses；
3. BF5 internal attribution unknown_fraction >0.05；
4. no dominant/removable/fusible component found；
5. basis/delta/feature fusion loses numerical correctness；
6. fused path uses source gap / formula proxy；
7. selected-feature elimination loses agreement；
8. static bucket / CUDA graph changes accept decisions；
9. persistent workspace exceeds memory ratio；
10. hash/norm/audit remains in timed path；
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

# Part X. 最终解释规则

## Case A：P6 system pass + P7/P8 pass

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full run and robustness。

## Case B：basis/delta fusion correct but still expensive

必须声明：

```text
payload-bound exact path is correct, but basis/delta feature construction remains too expensive.
```

下一步继续 lower-level kernelization / memory layout / arithmetic simplification，不调 decision frontier。

## Case C：selected-feature elimination loses agreement

必须声明：

```text
selected-feature materialization is functionally necessary under current controller.
```

下一步做 two-pass borderline，而不是用 proxy。

## Case D：static bucket / CUDA graph fails

必须声明：

```text
dynamic candidate shape or stream semantics prevent capture; launch overhead cannot be solved by graph alone.
```

下一步做 fused persistent kernels and bucketed workspace。

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

# Part XI. 最终建议

v9.2.73 的一句话策略是：

$$
\boxed{\text{不要再调 controller；把 BF5 剩余的 basis/delta/selected-feature/full-step sync 成本打穿，把 }2.713296\text{ 压到 }\leq1.50。}
$$

当前最关键的问题不是：

```text
oracle 是否存在；
reference frontier 是否 deployable；
PF5 candidate rate 是否可行；
payload 是否 missing；
state replay 是否 extra forward；
update payload 是否过大；
C3/T2/C4/E2 是否要重调；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. BF5 fused_common_basis_delta_pipeline 内部到底哪个 kernel 慢？
2. true_delta_selected_feature 是否必须 materialize？
3. bridge_score / accept-bit 能否在 kernel 内直接输出？
4. 1250 个 fused steps 是否可以 static bucket / persistent workspace / CUDA graph？
5. hash/norm/CPU metadata 是否仍在 timed path？
6. common basis prep 是否可以 recurrence/cache/fused row transform？
7. selected feature 和 bridge lookup 能否与 delta kernel 合并？
8. step q90 是否能从 2.713296 压到 <=1.50？
9. system controller 是否 official eligible？
10. LDO/LSO 是否通过？
11. official paired replay 是否打过 AdamWParallel / bestLR？
```

v9.2.73 的结果将给出清晰分叉：

```text
if system controller passes:
  open LDO/LSO and official paired replay.

if basis/delta fusion correct but still expensive:
  continue lower-level kernelization / memory layout, not decision tuning.

if selected-feature elimination fails:
  use two-pass borderline exact confirmation.

if launch/sync dominates:
  use static buckets / persistent workspace / CUDA graph where valid.

if compute passes but paired replay fails:
  system is legal, but functional causal advantage is insufficient.

if LDO/LSO fails:
  repair dataset-agnostic support/family reliability, not dataset-specific tuning.
```
