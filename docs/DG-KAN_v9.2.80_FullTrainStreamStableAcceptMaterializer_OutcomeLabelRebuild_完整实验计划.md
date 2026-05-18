# DG-KAN v9.2.80 Full-Train-Stream StableAccept Materializer 与 Outcome-Label Rebuild 完整实验计划

> 本计划基于 v9.2.79 `Stable-Accept Official Promotion 与 Full-System Batch-Major Runtime Closure` 的真实执行结果制定。  
> v9.2.79 的 terminal route 是：
>
> ```text
> route = R15-StableAcceptOfficialCalibrationBlocked
> base_candidate = LQ-t2-h256
> success_v9279_strict_purekan_functional = False
> success_v9279_full_functional = False
> success_v9279_external_ready = False
> ```
>
> v9.2.79 的关键事实是：
>
> ```text
> v9.2.78 boundary:
>   source_route_v9278 = R18-StableAcceptLocalClosedButSystemNotOfficial
>   stable_accept_contract_id = AC2Q2-stable-quantized-1e5-event-tie
>   stable_accept_contract_pass_v9278 = 1
>   native_bucket_kernel_v2_pass_v9278 = 1
>
> v9.2.79 route:
>   stable_accept_official_calibration_pass = 0
>   stable_accept_heldout_support_pass = 0
>   full_system_integration_pass = 0
>   full_system_step_attribution_pass = 0
>   batch_major_runtime_pass = 0
>   system_legal_controller_pass = 0
>   official_eligible = 0
>
> Full-row fields:
>   candidate_count = 2493
>   accepted_count_old_reference = 951
>   missing_stable_accept_fields =
>     stable_accept_contract_id;
>     score_ref;
>     score_native;
>     score_quantized_ref;
>     score_quantized_native;
>     stable_rank_ref;
>     stable_rank_native;
>     accept_ref;
>     accept_native
>
> Outcome fields:
>   missing_outcome_fields =
>     safe_good_label;
>     bad_event_label;
>     null_event_label
>
> Runtime:
>   native_kernel_used_in_p6 = 0
>   new_full_system_step_ratio_measured = 0
>   old_step_ratio_reused_as_measurement = 0
>   source_boundary_step_ratio_q90 = 2.713295831053225
>   controller_step_ratio_q90 = ""
>   native_time_ms_q90_local_v9278 = 0.6102416664361954
>   q90_reduction_local_v9278 = 0.4443256711105608
>   kernel_count_after_local_v9278 = 216
>   sync_count_after_local_v9278 = 24
>
> Outcome recovery:
>   outcome_label_recovery_pass = 0
>   outcome_label_best_join_mode = event_family_dataset_seed
>   outcome_source_hash_covered_candidate_rows = 0
>   outcome_event_family_dataset_seed_ambiguous_rows = 1648
>
> Stable accept trace coverage:
>   stable_accept_trace_coverage_pass = 0
>   stable_accept_local_trace_rows = 72
>   stable_accept_trace_matched_full_candidate_rows = 3
>   stable_accept_trace_full_row_coverage = 0.0012033694344163659
> ```
>
> v9.2.80 的核心判断是：
>
> $$
> \boxed{
> \text{v9.2.79 不是 stable accept 失败，也不是 native kernel 失败；它失败在 official promotion 所需 full-row materialization 缺失。}
> }
> $$
>
> v9.2.80 的目标不是继续 join 旧 artifact，也不是继续局部修 native kernel。旧 artifact 已经证明：
>
> ```text
> source_hash join coverage = 0；
> event_family join label ambiguous 很多；
> event_family+dataset+seed 仍有 1648 个 ambiguous candidate rows；
> v9.2.78 local stable trace 只覆盖 3/2493 full candidates。
> ```
>
> 因此 v9.2.80 必须重跑 full train-stream materializer，直接在同一个 event lifecycle 中同时记录：
>
> ```text
> full candidate row identity；
> AC2Q2 stable score/rank/accept；
> native stable kernel outputs；
> safe_good / bad_event / null_event outcome labels；
> calibration / heldout split id；
> P6 full-system native runtime trace。
> ```

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

v9.2.80 的新增硬约束是：

```text
不能用旧 outcome artifact join 结果作为 official label；
不能用 event_family 聚合 label 替代 row-level outcome；
不能用 v9.2.78 local stable trace 覆盖 3/2493 的结果补 full rows；
不能复用旧 C3 heldout/support metrics 作为 AC2Q2 official metrics；
不能把 v9.2.78 local native q90 替代 P6 full-system step ratio；
不能把 source boundary step_ratio_q90 = 2.713296 复用成新测量；
不能用 diagnostic_derived_from_measured_components = 1 的 row official；
不能把 native kernel local pass 写成 P6 system pass；
不能按 dataset 调阈值、tie policy、bucket route 或 fallback rule。
```

允许使用：

```text
AC2Q2-stable-quantized-1e5-event-tie；
native CUDA / Triton stable bucket kernel；
full train-stream materializer；
calibration-split frozen top-K / threshold policy；
row-level outcome labels；
post-step exact audit subset；
dataset/family/horizon diagnostics；
batch-major persistent workspace；
compact bridge lookup inside kernel；
borderline exact fallback if pre-registered and measured。
```

---

# Part I. 对 v9.2.79 的独立判断

## 1. v9.2.79 没有达到目标

v9.2.79 没有 strict PureKAN functional success。它的失败不是因为 stable accept local correctness 回退，也不是因为 native bucket kernel v2 回退，而是因为 official promotion 需要的 full-row materialization 不存在：

```text
stable_accept_official_calibration_pass = 0
stable_accept_heldout_support_pass = 0
system_legal_controller_pass = 0
official_eligible = 0
```

因此，不能声明：

```text
system-legal controller pass
leave-dataset-out pass
leave-stratum-out pass
official paired replay pass
short-run pass
full-run pass
external-ready
```

这一点是正确 gate。stable accept 是 rule change，必须重新跑 calibration / heldout / support；不能用旧 C3 metrics，也不能用局部 kernel pass 代替 full-system controller。

## 2. v9.2.79 的真实进展

v9.2.79 的真实进展是 **防止错误 officialization**。它严格证明：

```text
v9.2.78 的 stable accept / native bucket kernel local pass 不能直接 promotion；
full-online rows 缺 AC2Q2 stable score/rank/accept 字段；
full-online rows 缺 outcome labels；
P6 没有 native stable kernel full-system trace；
旧 outcome artifact join 不可靠；
local stable trace 覆盖率不足。
```

这不是令人兴奋的 performance 进展，但它是必要的科学进展。没有 v9.2.79，下一步很容易犯两个错误：

```text
错误 1:
  把 v9.2.78 local q90 = 0.610ms 当成 P6 system step ratio。

错误 2:
  把旧 C3 accept/outcome labels 拼接到 AC2Q2 stable accept rule 上，声称 heldout/support pass。
```

v9.2.79 阻止了这两种错误。

## 3. v9.2.79 的真实失败

v9.2.79 的失败集中在三处。

第一，P1/P2 无法 rerun，因为 full-online row 没有：

```text
score_ref
score_native
score_quantized_ref
score_quantized_native
stable_rank_ref
stable_rank_native
accept_ref
accept_native
safe_good_label
bad_event_label
null_event_label
```

这意味着 AC2Q2 stable accept rule 没有 full candidate table，calibration / heldout / support 都不能真实计算。

第二，P3/P4/P5 没有 full-system native runtime trace：

```text
native_bucket_kernel_used_in_p6 = 0
native_runtime_full_system_materialization_present = 0
new_full_system_step_ratio_measured = 0
batch_major_native_full_system_runtime_measured = 0
```

因此无法判断旧 `2.713296` 是 old path 残留，还是 native stable kernel 的真实 full-system 下界。

第三，P12/P13 说明旧 artifact 不能补：

```text
source_hash_covered_candidate_rows = 0
event_family_dataset_seed_ambiguous_rows = 1648
stable_accept_trace_matched_full_candidate_rows = 3
stable_accept_trace_full_row_coverage = 0.0012033694344163659
```

这说明继续做 join 修补不是正路。

## 4. 当前 blocker 的本质

当前 blocker 是：

$$
\boxed{
\text{full train-stream candidate lifecycle 缺少 stable-accept/outcome co-materialization。}
}
$$

更直白地说，当前不是：

```text
reference frontier 不存在；
stable accept 不一致；
native bucket kernel 不可用；
quantile-tail 无法优化；
payload binding 回退；
decision/support metrics 已知失败；
dataset-specific failure。
```

当前是：

```text
stable accept 的 full-row input/output 没有落盘；
outcome labels 没有和 candidate rows 同步落盘；
native kernel 没有进入 P6 full-system trace；
calibration / heldout / support 无法重算；
P7/P8/P9/P10 不能打开。
```

这是一个 materializer / lifecycle 问题，不是 controller science 问题。

## 5. 是否还在正确道路上

是。路线仍然正确，但下一步必须更靠近数据生命周期，而不是继续在局部 kernel 上打转：

```text
stable accept local correctness 已闭合
native bucket kernel local runtime 已闭合
official promotion 所需 full-row table 缺失
→ 重跑 full train-stream materializer
→ 重跑 calibration / heldout / support
→ 接入 native kernel full-system timing
→ system-legal controller
→ LDO / LSO
→ official paired replay
```

错误路线是：

```text
继续 join 旧 outcome artifact；
继续补 event_family 级别 label；
继续从 72-row local trace 外推 full rows；
继续调 C3/T2 threshold；
继续按 dataset 调 accept rule；
继续只优化 native q90；
继续跳过 P6 直接 paired replay。
```

---

# Part II. v9.2.80 总体目标

v9.2.80 的总体目标是：

$$
\boxed{
\text{重跑 full train-stream materializer，为每个 candidate row 同时落盘 stable score/rank/accept、outcome labels 与 native runtime trace。}
}
$$

强目标：

$$
OfficialEligible=1,
$$

$$
SystemLegalControllerPass=1,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
AcceptDisagreement=0.
$$

最低有效推进目标：

```text
stable_accept_full_row_materialization_present = 1
outcome_labels_present = 1
full_row_join_required_for_label = 0
native_bucket_kernel_used_in_p6 = 1
new_full_system_step_ratio_measured = 1
old_step_ratio_reused_as_measurement = 0
```

如果 v9.2.80 只完成 full materialization，但 step ratio 仍 >1.50，它仍然是有效进展，因为它会把 blocker 从 “cannot rerun official gates” 推到 “official gates measured and failed on X”。

---

# Part III. 核心假设

## H1：缺失的是 materialization，而不是 stable accept rule 本身

H1 成立标准：

```text
full-row stable score/rank/accept materialized = 1
accept_disagreement_count = 0
score_quantized_disagreement_count = 0
rank_disagreement_count = 0
stable_accept_official_calibration_pass measured
stable_accept_heldout_support_pass measured
```

H1 失败标准：

materializer 生成 full rows 后，AC2Q2 stable accept 与 native accept 再次发生 disagreement。此时 blocker 回到 accept contract，而不是 materialization。

## H2：outcome labels 必须在同一次 train-stream 中产生，旧 artifact join 不可靠

H2 成立标准：

```text
safe_good_label present for every candidate row
bad_event_label present for every candidate row
null_event_label present for every candidate row
label_source = same_run_train_stream_replay
label_join_mode = direct_event_id_or_candidate_id
ambiguous_label_count = 0
missing_label_count = 0
```

H2 失败标准：

仍需要 event_family / dataset / seed join 才能补 label，或 ambiguity 不为 0。

## H3：native stable kernel 接入 P6 后，full-system step ratio 会显著低于旧 2.713296

H3 成立标准：

```text
native_bucket_kernel_used_in_p6 = 1
bridge_score_inside_kernel = 1
accept_bit_inside_kernel = 1
kernel_count_after <= 0.50 * kernel_count_before
sync_count_after <= 0.50 * sync_count_before
avg_candidates_per_kernel_after >= 8
new_full_system_step_ratio_measured = 1
step_ratio_q90 < 2.713296
```

Full pass:

$$
StepRatio_{q90}\leq1.50.
$$

H3 失败标准：

native kernel 接入 P6 且 attribution 完整，但 step ratio 仍 >1.50。此时才说明 full-system runtime 仍昂贵。

## H4：official decision gates 会保持 reference frontier

H4 成立标准：

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

H4 失败标准：

AC2Q2 stable accept materially changes accepted region, causing precision / bad-event / null-rate / coverage gate failure.

## H5：如果 P6 pass 后 paired replay fail，才回到 functional event/value target

H5 成立标准：

```text
P6 system legal controller pass = 1
P7 leave-out measured
P8 official paired replay measured
P8 fails vs AdamWParallel / bestLR
```

H5 失败标准：

P6 未过时提前讨论 functional value target。这会混淆 system closure 与 functional causal advantage。

---

# Part IV. Full train-stream materializer design

## 1. Row identity contract

Every candidate row must satisfy:

$$
event\_id
=
candidate\_event\_id
=
stable\_score\_event\_id
=
outcome\_event\_id
=
native\_kernel\_event\_id
=
update\_payload\_event\_id.
$$

Required row fields:

```text
event_id
global_row_id
candidate_id
dataset
seed
step
batch_id
sample_id
family_id
bucket_id
horizon
candidate_rank_old
candidate_rank_stable
candidate_source_hash
payload_hash
```

Pass condition:

```text
duplicate_event_id_count = 0
duplicate_candidate_id_count = 0
event_id_mismatch_count = 0
candidate_missing_count = 0
payload_hash_missing_count = 0
```

## 2. Stable accept full-row fields

Every candidate row must record:

```text
stable_accept_contract_id
score_ref
score_native
score_quantized_ref
score_quantized_native
stable_rank_ref
stable_rank_native
tie_key_ref
tie_key_native
accept_ref
accept_native
accept_disagreement
score_quantized_disagreement
rank_disagreement
```

Stable accept rule:

$$
score_q(e)=round(10^5\cdot score(e)).
$$

$$
rank(e)=stable\_sort(-score_q(e), event\_id(e)).
$$

$$
Accept(e)=1 \iff rank(e)\leq K.
$$

Pass condition:

```text
stable_accept_full_row_materialization_present = 1
accept_disagreement_count = 0
score_quantized_disagreement_count = 0
rank_disagreement_count = 0
missing_stable_accept_fields = ""
```

## 3. Outcome label fields

Every candidate row must record:

```text
safe_good_label
bad_event_label
null_event_label
task_safe_label
CEp99_delta
margin_delta
ECE_delta
NLL_delta
curvature_delta
real_beats_adamwparallel
real_beats_bestlr
outcome_horizon
outcome_branch
outcome_source = same_run
```

The three primary labels should be derived in the same run:

$$
SafeGood(e)=1
\iff
TaskSafe(e)=1
\land
BadEvent(e)=0
\land
NullEvent(e)=0
\land
Useful(e)=1.
$$

$$
BadEvent(e)=1
\iff
CEp99(e)>\tau_{CE}
\lor
MarginP10(e)<-\tau_M
\lor
TaskSafe(e)=0.
$$

$$
NullEvent(e)=1
\iff
|CEp99_{\Delta}(e)|<\epsilon_{CE}
\land
|Margin_{\Delta}(e)|<\epsilon_M
\land
|Curvature_{\Delta}(e)|<\epsilon_C.
$$

The exact thresholds must be the same frozen thresholds used in the prior C3/C3Q2 controller protocol, not tuned by dataset.

Pass condition:

```text
outcome_labels_present = 1
missing_outcome_fields = ""
missing_label_count = 0
ambiguous_label_count = 0
label_join_mode = direct_same_run_event_id
```

## 4. Native runtime fields

Every measured step must record:

```text
native_bucket_kernel_used_in_p6
stable_accept_cuda_kernel_used
basis_norm_bucketed
W2_delta_bucketed
bridge_score_inside_kernel
accept_bit_inside_kernel
kernel_count_before
kernel_count_after
sync_count_before
sync_count_after
allocation_count_before
allocation_count_after
avg_candidates_per_kernel_before
avg_candidates_per_kernel_after
native_time_ms_q90
eager_time_ms_q90
q90_reduction
new_full_system_step_ratio_measured
old_step_ratio_reused_as_measurement
```

Pass condition:

```text
native_bucket_kernel_used_in_p6 = 1
bridge_score_inside_kernel = 1
accept_bit_inside_kernel = 1
new_full_system_step_ratio_measured = 1
old_step_ratio_reused_as_measurement = 0
```

System gate:

$$
StepRatio_{q90}\leq1.50.
$$

---

# Part V. Candidate designs

## 1. Materializer candidates

### MAT0：v9.2.79 audit reference

Expected fail. Keeps old artifact state:

```text
stable_accept_full_row_materialization_present = 0
outcome_labels_present = 0
native_kernel_used_in_p6 = 0
```

### MAT1：FullRowStableAcceptMaterializer

Materializes stable accept fields for all 2493 candidates.

### MAT2：FullRowOutcomeLabelMaterializer

Materializes safe_good / bad_event / null_event labels in the same train-stream.

### MAT3：JointStableAcceptOutcomeMaterializer

Materializes stable accept fields and outcome labels in one pass.

### MAT4：FullTraceNativeRuntimeMaterializer

Adds native kernel P6 trace for the same rows.

### MAT5：AllInOneTrainStreamMaterializer

One runner pass that writes identity, payload, stable accept, outcome labels, native runtime, and update payload.

## 2. Outcome measurement candidates

### OUT0：old artifact join reference

Expected fail; diagnostic only.

### OUT1：same-run paired replay labels

Measures outcome labels from same-run paired replay windows.

### OUT2：same-run short-horizon labels

Uses horizons matching C3/C3Q2 protocol.

### OUT3：multi-horizon same-run labels

Records labels for multiple horizons but freezes official horizon choice before heldout.

### OUT4：branch-control matched labels

Records RealFunctional / AdamWParallel / bestLR / NoOp / Random controls for each event.

## 3. Stable accept candidates

### SA0：AC2Q2 reference

Stable quantized score + event-id tie.

### SA1：AC2Q2 native full-row

Native kernel emits the same stable accept fields.

### SA2：AC2Q2 reference-native audit

Reference and native both emit rows; compare disagreement.

### SA3：AC2Q2 borderline audit

Tracks rows near quantized score tie boundaries; official rule remains stable event-id tie.

## 4. Native runtime candidates

### RT0：local v9.2.78 native kernel reference

Diagnostic only.

### RT1：P6 integrated native stable kernel

Native stable kernel wired into full P6 train-stream.

### RT2：batch-major persistent P6 runtime

Keeps P3 local kernel/sync gains in P6.

### RT3：compact bridge lookup P6 runtime

Ensures bridge score stays inside kernel / compact lookup.

### RT4：post-step audit P6 runtime

Moves audit outside timed path while preserving no-fake verification.

---

# Part VI. 实验阶段

## P0：v9.2.79 boundary reproduction

### 目标

确认 v9.2.79 boundary 稳定，避免在不稳定 artifact 上继续。

### 必须记录

```text
route
source_route_v9279
stable_accept_contract_id
stable_accept_contract_pass_v9278
native_bucket_kernel_v2_pass_v9278
stable_accept_official_calibration_pass
stable_accept_heldout_support_pass
system_legal_controller_pass
official_eligible
missing_stable_accept_fields
missing_outcome_fields
native_bucket_kernel_used_in_p6
new_full_system_step_ratio_measured
old_step_ratio_reused_as_measurement
outcome_label_recovery_pass
stable_accept_trace_coverage_pass
fake_proxy_count
cpu_offload_used
```

### 判断标准

P0 pass:

```text
route = R15-StableAcceptOfficialCalibrationBlocked
stable_accept_contract_pass_v9278 = 1
native_bucket_kernel_v2_pass_v9278 = 1
official_eligible = 0
stable_accept_official_calibration_pass = 0
fake/proxy/offload = 0
```

### 可视化

```text
p0_v9279_boundary_ladder.svg
p0_local_pass_vs_official_gap.svg
p0_missing_fields_heatmap.svg
```

---

## P1：full-row stable accept materialization

### 目标

为全量 candidate rows materialize AC2Q2 stable score/rank/accept fields。

### 必须记录

```text
event_id
candidate_id
dataset
seed
step
family_id
bucket_id
horizon
stable_accept_contract_id
score_ref
score_native
score_quantized_ref
score_quantized_native
stable_rank_ref
stable_rank_native
tie_key_ref
tie_key_native
accept_ref
accept_native
accept_disagreement
score_quantized_disagreement
rank_disagreement
missing_stable_accept_field_count
```

### 判断标准

P1 pass:

```text
candidate_count = 2493 or current full candidate count
stable_accept_full_row_materialization_present = 1
missing_stable_accept_field_count = 0
accept_disagreement_count = 0
score_quantized_disagreement_count = 0
rank_disagreement_count = 0
```

### 可视化

```text
p1_stable_score_distribution.svg
p1_rank_agreement_confusion.svg
p1_accept_overlap.svg
p1_missing_fields_zero_audit.svg
```

---

## P2：same-run outcome label materialization

### 目标

在同一 train-stream lifecycle 中为每个 candidate row 生成 outcome labels，避免旧 artifact join。

### 必须记录

```text
event_id
candidate_id
dataset
seed
step
horizon
branch
safe_good_label
bad_event_label
null_event_label
task_safe_label
useful_label
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
real_beats_adamwparallel
real_beats_bestlr
outcome_source
label_join_mode
missing_label_count
ambiguous_label_count
```

### 判断标准

P2 pass:

```text
outcome_labels_present = 1
label_source = same_run_train_stream
label_join_mode = direct_event_id_or_candidate_id
missing_label_count = 0
ambiguous_label_count = 0
```

### 可视化

```text
p2_outcome_label_balance.svg
p2_bad_null_safegood_distribution.svg
p2_outcome_by_family_horizon.svg
p2_label_missing_ambiguity_audit.svg
```

---

## P3：joint stable-accept/outcome table

### 目标

合并 P1/P2 成为 official controller table。不是 join 旧 artifact，而是 same-run table。

### 必须记录

```text
event_id
candidate_id
stable_accept_contract_id
accept_ref
accept_native
safe_good_label
bad_event_label
null_event_label
calibration_split_id
heldout_split_id
support_stratum_id
family_id
bucket_id
horizon
row_complete
```

### 判断标准

P3 pass:

```text
row_complete = 1 for all candidate rows
stable_accept_full_row_materialization_present = 1
outcome_labels_present = 1
complete_candidate_rows = candidate_count
incomplete_candidate_rows = 0
```

### 可视化

```text
p3_complete_row_lifecycle.svg
p3_accept_vs_outcome_matrix.svg
p3_calibration_heldout_split_balance.svg
```

---

## P4：stable accept official calibration rerun

### 目标

用 P3 complete table 重跑 calibration，不复用旧 C3 metrics。

### 必须记录

```text
controller_id
accept_contract_id
calibration_rerun
calibration_split_id
event_count
candidate_count
accepted_count
candidate_rate
precision_cal
coverage_cal
bad_event_cal
null_rate_cal
precision_lcb_cal
bad_event_ucb_cal
accepted_signal_strata_count_cal
accepted_family_count_cal
max_family_share_cal
max_stratum_share_cal
dataset_name_used
validation_used
test_used
```

### 判断标准

P4 pass:

```text
calibration_rerun = 1
dataset_name_used = 0
validation_used = 0
test_used = 0
accepted_signal_strata_count_cal >= 5
accepted_family_count_cal >= 32
max_family_share_cal <= 0.50
max_stratum_share_cal <= 0.60
```

Decision pre-gate:

$$
Precision_{\text{cal}}\geq0.75,
$$

$$
Coverage_{\text{cal}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{cal}}\leq0.05,
$$

$$
NullRate_{\text{cal}}\leq0.15.
$$

### 可视化

```text
p4_calibration_frontier.svg
p4_calibration_support_balance.svg
p4_stable_accept_old_accept_overlap.svg
```

---

## P5：stable accept heldout/support rerun

### 目标

用 complete table 重跑 heldout 和 support balance，证明 AC2Q2 不只是 calibration artifact。

### 必须记录

```text
controller_id
accept_contract_id
heldout_split_id
event_count
candidate_count
accepted_count
candidate_rate
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
accept_disagreement_count
score_quantized_disagreement_count
rank_disagreement_count
dataset_name_used
posthoc_used_at_commit
```

### 判断标准

P5 pass:

```text
accept_disagreement_count = 0
score_quantized_disagreement_count = 0
rank_disagreement_count = 0
dataset_name_used = 0
posthoc_used_at_commit = 0
accepted_signal_strata_count >= 5
accepted_family_count >= 32
max_family_share <= 0.50
max_stratum_share <= 0.60
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

### 可视化

```text
p5_heldout_decision_metrics.svg
p5_lcb_ucb_gate.svg
p5_support_balance_heldout.svg
p5_bad_null_coverage_tradeoff.svg
```

---

## P6：native stable kernel full-system integration

### 目标

证明 native stable kernel 真正进入 full P6 train-stream controller path。

### 必须记录

```text
runtime_candidate_id
native_kernel_id
controller_id
native_kernel_used_in_p6
basis_norm_bucketed
W2_delta_bucketed
bridge_score_inside_kernel
accept_bit_inside_kernel
compact_bridge_lookup_inside_kernel
stable_accept_inside_kernel
candidate_count
accepted_count
kernel_count_before
kernel_count_after
sync_count_before
sync_count_after
allocation_count_before
allocation_count_after
avg_candidates_per_kernel_before
avg_candidates_per_kernel_after
native_time_ms_mean
native_time_ms_q90
eager_time_ms_q90
q90_reduction
accept_disagreement_count
bridge_score_error_max
logits_error_max
delta_error_max
tail_disagreement_count
```

### 判断标准

P6 pass:

```text
native_kernel_used_in_p6 = 1
basis_norm_bucketed = 1
W2_delta_bucketed = 1
bridge_score_inside_kernel = 1
accept_bit_inside_kernel = 1
stable_accept_inside_kernel = 1
accept_disagreement_count = 0
q90_reduction >= 0.40
avg_candidates_per_kernel_after >= 8
kernel_count_after <= 0.50 * kernel_count_before
sync_count_after <= 0.50 * sync_count_before
allocation_count_after <= 0.10 * allocation_count_before
```

### 可视化

```text
p6_kernel_integration_ladder.svg
p6_kernel_sync_allocation_before_after.svg
p6_candidates_per_kernel_hist.svg
p6_native_vs_eager_q90.svg
```

---

## P7：full-system step attribution

### 目标

得到新的 P6 full-system native step ratio，不能复用 old boundary。

### 必须记录

```text
system_runtime_id
controller_id
runtime_candidate_id
selector_time_ms
candidate_pack_time_ms
quantile_tail_time_ms
basis_norm_time_ms
W2_delta_time_ms
probe_logits_time_ms
bridge_score_time_ms
stable_accept_time_ms
update_payload_time_ms
payload_apply_time_ms
kernel_launch_time_ms
sync_time_ms
allocation_time_ms
audit_outside_timed_time_ms
other_time_ms
unknown_fraction
total_step_time_ms_q50
total_step_time_ms_q90
mlp_step_time_ms_q90
step_ratio_q90
memory_ratio
new_full_system_step_ratio_measured
old_step_ratio_reused_as_measurement
```

### 判断标准

P7 pass:

```text
unknown_fraction <= 0.05
new_full_system_step_ratio_measured = 1
old_step_ratio_reused_as_measurement = 0
native stable kernel component present = 1
audit outside timed path = 1
```

System diagnostic pass:

$$
StepRatio_{q90}\leq2.00.
$$

Full system pass:

$$
StepRatio_{q90}\leq1.50.
$$

### 可视化

```text
p7_full_system_cost_waterfall.svg
p7_old_vs_native_step_ratio.svg
p7_component_ratio_stacked_bar.svg
p7_runtime_q50_q90_distribution.svg
```

---

## P8：system-legal exact-signal controller v12

### 目标

组合 P4-P7，建立 official system controller。

### 必须记录

```text
controller_id
system_candidate_id
accept_contract_id
native_kernel_id
runtime_candidate_id
bucket_strategy
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
accept_disagreement_count
score_quantized_disagreement_count
rank_disagreement_count
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
stable_accept_full_row_materialization_present
outcome_labels_present
materialized_system_path
native_bucket_kernel_used
stable_accept_contract_used
bridge_score_inside_kernel
accept_bit_inside_kernel
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

P8 pass:

```text
official_eligible = 1
system_legal_controller_pass = 1
materialized_system_path = 1
stable_accept_full_row_materialization_present = 1
outcome_labels_present = 1
native_bucket_kernel_used = 1
stable_accept_contract_used = 1
bridge_score_inside_kernel = 1
accept_bit_inside_kernel = 1
accept_disagreement_count = 0
score_quantized_disagreement_count = 0
rank_disagreement_count = 0
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
p8_system_controller_cost_quality_frontier.svg
p8_reference_vs_stable_accept_overlap.svg
p8_step_ratio_progress_v9278_v9280.svg
p8_family_strata_balance.svg
p8_accept_contract_audit_dashboard.svg
```

---

## P9：leave-dataset-out / leave-stratum-out

### 目标

只有 P8 pass 后 official 打开。证明 controller 不是 pooled calibration artifact，也不是 dataset-specific route。

### 设置

Leave-dataset-out:

```text
calibrate on MNIST + Fashion, evaluate KMNIST
calibrate on MNIST + KMNIST, evaluate Fashion
calibrate on Fashion + KMNIST, evaluate MNIST
```

Leave-stratum-out:

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
accept_contract_id
native_kernel_id
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

LDO pass:

$$
Acc_{\text{heldout,Real}}\geq Acc_{\text{heldout,AdamW}}-0.005.
$$

At least two held-out datasets:

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
p9_leave_dataset_out_matrix.svg
p9_leave_stratum_out_matrix.svg
p9_dataset_tuning_audit.svg
p9_leaveout_failure_modes.svg
```

---

## P10：official paired replay

### 目标

验证 RealFunctional 是否在 strong controls 下有局部因果优势。

### 设置

```text
base = R2 repaired base checkpoint
controller_id = best P9 survivor
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
  ShuffledStableAcceptScore
  ShuffledStableAcceptRank
  ShuffledStableAcceptTiePolicy
  ShuffledOutcomeLabel
  ShuffledNativeBucketKernel
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

Paired replay pass:

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

Shuffle controls must fail:

```text
ShuffledTrueBranchDelta = fail
ShuffledPF5Prefilter = fail
ShuffledCandidatePayload = fail
ShuffledStableAcceptScore = fail
ShuffledStableAcceptRank = fail
ShuffledStableAcceptTiePolicy = fail
ShuffledOutcomeLabel = fail
ShuffledNativeBucketKernel = fail
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

System:

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p10_official_paired_replay_pareto.svg
p10_macro_beat_rate.svg
p10_signal_stratum_win_matrix.svg
p10_shuffle_control_matrix.svg
p10_system_gate_distribution.svg
```

---

## P11：short-run scout

### 目标

如果 P10 paired replay pass，验证 local causal advantage 能否进入连续训练。

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

Task safety:

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005.
$$

Control superiority:

$$
MetricGain_{\text{functional}}>MetricGain_{\text{AdamWParallel}},
$$

$$
MetricGain_{\text{functional}}>MetricGain_{\text{bestLR}}.
$$

Mechanism pass, at least one:

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

## P12：full run / robustness / strong baseline

### 目标

只有 P11 pass 后打开。验证 functional advantage 不是 local replay artifact。

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
  ShuffledStableAcceptScore
  ShuffledStableAcceptTiePolicy
  ShuffledOutcomeLabel
  ShuffledNativeBucketKernel
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

Full functional pass:

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

Strong baseline pass:

```text
Functional not explained by QuadraticFeatureMLP
Functional not explained by LR grid
Functional not explained by shuffled controller/kernel/payload
```

---

# Part VII. Required artifacts

```text
run_manifest.json
contract_audit_v9280.csv
p0_v9279_boundary_reproduction.csv
p1_full_row_stable_accept_materialization.csv
p2_same_run_outcome_label_materialization.csv
p3_joint_stable_accept_outcome_table.csv
p4_stable_accept_official_calibration_rerun.csv
p5_stable_accept_heldout_support_rerun.csv
p6_native_stable_kernel_full_system_integration.csv
p7_full_system_step_attribution_native_stable.csv
p8_system_legal_exact_signal_controller_v12.csv
p9_leave_dataset_and_stratum_out.csv
p10_official_paired_replay.csv
p11_short_run_functional_validation.csv
p12_full_run_robustness_strong_baseline.csv

full_row_identity_trace_v9280.csv
stable_accept_full_row_trace_v9280.csv
outcome_label_full_row_trace_v9280.csv
joint_controller_table_v9280.csv
native_kernel_full_system_trace_v9280.csv
full_system_step_cost_trace_v9280.csv
system_controller_trace_v9280.csv
leaveout_trace_v9280.csv
paired_replay_branch_trace_v9280.csv
short_run_trace_v9280.csv

route_decision.json
aggregate_decision.json
failure_table.csv
artifact_hashes.csv
figures/
```

Failure taxonomy:

```text
F1_contract_violation
F2_v9279_boundary_unstable
F3_dataset_tuning_detected
F4_payload_binding_regression
F5_stable_accept_full_row_fields_missing
F6_score_quantized_disagreement
F7_rank_disagreement
F8_accept_disagreement_reappears
F9_outcome_labels_missing
F10_outcome_labels_ambiguous
F11_outcome_label_join_used_for_official
F12_joint_table_incomplete
F13_stable_accept_calibration_fail
F14_stable_accept_heldout_fail
F15_support_balance_fail
F16_native_kernel_not_used_in_p6
F17_bridge_score_not_inside_kernel
F18_accept_bit_not_inside_kernel
F19_native_runtime_q90_regression
F20_kernel_sync_reduction_lost
F21_avg_candidates_per_kernel_too_low
F22_full_system_step_attribution_incomplete
F23_full_system_step_ratio_fail
F24_memory_ratio_fail
F25_source_gap_or_formula_proxy_used
F26_projection_used
F27_system_controller_precision_fail
F28_system_controller_coverage_fail
F29_system_controller_bad_event_fail
F30_system_controller_null_rate_fail
F31_system_controller_lcb_ucb_fail
F32_leave_dataset_out_fail
F33_leave_stratum_out_fail
F34_paired_replay_control_equivalent
F35_shuffle_control_pass
F36_functional_lr_equivalent
F37_short_run_task_drop
F38_full_run_no_macro_hard_stratum_gain
F39_strong_baseline_explains_gain
F40_robustness_fail
F41_external_not_ready
F42_fake_or_proxy_violation
F43_artifact_missing
```

---

# Part VIII. Route decision

```text
R1-BoundaryReproduced:
  v9.2.79 boundary reproduced.

R2-StableAcceptFullRowMaterialized:
  full candidate rows contain stable score/rank/accept fields.

R3-OutcomeLabelsMaterialized:
  same-run safe_good / bad_event / null_event labels materialized.

R4-JointControllerTablePass:
  stable accept and outcome labels complete in one candidate table.

R5-StableAcceptCalibrationPass:
  AC2Q2 stable accept official calibration pass.

R6-StableAcceptHeldoutSupportPass:
  heldout decision gates and support balance pass.

R7-NativeStableKernelIntegrated:
  native stable kernel used in full P6 train-stream path.

R8-FullSystemNativeRuntimePass:
  full-system native runtime reaches step_ratio_q90 <= 2.00 with attribution.

R9-SystemLegalExactSignalControllerPass:
  system controller passes decision + compute gates.

R10-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R11-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R12-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R13-ShortRunFunctionalPass:
  short-run task-safe mechanism gain.

R14-FullFunctionalPass:
  full run task / geometry / system / control gates pass.

R15-PayloadBindingRegression:
  payload binding no longer reproduces.

R16-StableAcceptMaterializationFail:
  stable accept full-row fields still missing.

R17-OutcomeLabelMaterializationFail:
  outcome labels missing or ambiguous.

R18-NativeKernelIntegrationFail:
  native stable kernel cannot be integrated into P6 full-system path.

R19-FullSystemRuntimeStillOldPath:
  step ratio remains old source boundary because runtime integration did not occur.

R20-SystemStillTooExpensive:
  native stable path is correct and integrated but step_ratio_q90 remains >1.50.

R21-ComputePassButLeaveoutFail:
  system controller overfits pooled calibration.

R22-ComputePassButPairedReplayFail:
  controller is system-legal but not causally superior to controls.

R23-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` must record:

```text
route
v9279_boundary_pass
dataset_tuning_detected
reference_controller_id
stable_controller_id
accept_contract_id
accept_rule_changed
calibration_rerun
heldout_rerun
support_rerun

payload_binding_contract_pass
candidate_tensor_payload_missing_count
candidate_branch_logits_missing_count
candidate_true_delta_logits_missing_count
functional_update_payload_missing_count

stable_accept_full_row_materialization_present
missing_stable_accept_fields
score_quantized_disagreement_count
rank_disagreement_count
accept_disagreement_count

outcome_labels_present
missing_outcome_fields
missing_label_count
ambiguous_label_count
label_source
label_join_mode

joint_controller_table_pass
complete_candidate_rows
incomplete_candidate_rows

stable_accept_calibration_pass
stable_accept_heldout_support_pass
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

native_kernel_integrated_pass
native_kernel_used_in_p6
native_bucket_kernel_used
stable_accept_cuda_kernel_used
basis_norm_bucketed
W2_delta_bucketed
bridge_score_inside_kernel
accept_bit_inside_kernel
kernel_count_reduction
sync_count_reduction
allocation_count_reduction
avg_candidates_per_kernel_after
native_time_ms_q90
eager_time_ms_q90
q90_reduction

full_system_step_attribution_pass
selector_time_ms
quantile_tail_time_ms
basis_norm_time_ms
W2_delta_time_ms
bridge_score_time_ms
stable_accept_time_ms
update_payload_time_ms
kernel_launch_time_ms
sync_time_ms
allocation_time_ms
unknown_fraction
controller_step_ratio_q90
controller_memory_ratio
new_full_system_step_ratio_measured
old_step_ratio_reused_as_measurement

best_system_controller_id
system_legal_controller_pass
official_eligible

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
success_v9280_strict_purekan_functional
success_v9280_full_functional
success_v9280_external_ready
```

---

# Part IX. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 full-row stable accept materialization
  P2 same-run outcome label materialization
  P6 native stable kernel integration smoke

Batch 2:
  P3 joint stable/outcome table
  P4 calibration rerun
  P5 heldout/support rerun
  P7 full-system step attribution

Batch 3:
  P8 system-legal controller
  P9 leave-dataset-out / leave-stratum-out scout

Batch 4:
  official P9 LDO/LSO
  official P10 paired replay

Batch 5:
  P11 short-run if P10 passes
  P12 full run / robustness / strong baseline only if P11 passes
```

Gate rule:

```text
P4 cannot pass unless:
  P1 pass
  P2 pass
  P3 pass

P8 cannot pass unless:
  P4 stable accept calibration pass
  P5 stable accept heldout/support pass
  P6 native stable kernel integration pass
  P7 full-system attribution pass
  step_ratio_q90 <= 1.50
  stable_accept_full_row_materialization_present = 1
  outcome_labels_present = 1
  accept_disagreement_count = 0
  materialized_system_path = 1
  native_bucket_kernel_used = 1
  bridge_score_inside_kernel = 1
  accept_bit_inside_kernel = 1
  audit_only_cost_removal = 0
  diagnostic_derived_from_measured_components = 0
  projection_used = 0
  source_measured_gap_used = 0
  formula_proxy_used = 0
  full_online_payload_binding = 1
  full_online_update_payload_binding = 1

P9/P10 diagnostic rows may be measured before all gates finish,
but official status requires:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  stable accept controller calibrated
  PF5 runtime selector pass
  payload binding pass
  stable accept materialization pass
  outcome label materialization pass
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
v9.2.79 boundary reproduced
full-row stable accept materialization measured
same-run outcome label materialization measured
joint stable/outcome table measured
calibration rerun measured
heldout/support rerun measured
native kernel full-system integration measured
full-system native step attribution measured
system controller measured
no fake/proxy/offload/loss/teacher violation
```

## Materialization success

```text
Minimum diagnostic success
+
stable_accept_full_row_materialization_present = 1
+
outcome_labels_present = 1
+
missing stable accept fields = none
+
missing outcome fields = none
+
ambiguous_label_count = 0
+
accept_disagreement_count = 0
```

## Official controller decision success

```text
Materialization success
+
calibration decision gates pass
+
heldout decision gates pass
+
support balance pass
```

## Runtime materialization success

```text
Official controller decision success
+
native_kernel_used_in_p6 = 1
+
bridge_score_inside_kernel = 1
+
accept_bit_inside_kernel = 1
+
new_full_system_step_ratio_measured = 1
+
old_step_ratio_reused_as_measurement = 0
+
q90_reduction >= 0.40 or full step_ratio_q90 <= 1.50
```

## System success

```text
Runtime materialization success
+
step_ratio_q90 <= 1.50
+
memory_ratio <= 1.05
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
1. v9.2.79 boundary cannot be reproduced；
2. payload binding regresses；
3. stable accept full-row fields missing；
4. outcome labels missing；
5. outcome labels ambiguous；
6. labels still require old artifact join；
7. accept disagreement reappears；
8. score quantized or rank disagreement appears；
9. calibration decision gates fail；
10. heldout precision / coverage / bad-event / null-rate fails；
11. support balance fails；
12. native kernel does not enter P6 full-system path；
13. P6 step ratio remains old 2.713296 because runtime integration did not occur；
14. native q90 reduction disappears in full-system path；
15. bridge score cannot stay inside kernel；
16. accept bit cannot stay inside kernel；
17. kernel/sync/allocation remains old path；
18. avg candidates per kernel remains <2；
19. low-cost path uses source gap / formula proxy；
20. system step_ratio_q90 remains >1.50；
21. memory ratio >1.05；
22. leave-dataset-out fails；
23. leave-stratum-out fails；
24. paired replay remains control-equivalent；
25. shuffle controls pass；
26. short-run task drops；
27. full run gives no macro / hard-stratum / geometry gain；
28. functional breaks system gate；
29. gains are explained by QuadraticFeatureMLP；
30. any teacher/loss/fake/proxy/offload/projection-as-pass violation occurs。
```

---

# Part XI. 最终解释规则

## Case A：P8 system pass + P9/P10 pass

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full run and robustness。

## Case B：full-row materialization still missing

必须声明：

```text
v9.2.80 failed to execute the real required next step; old artifacts cannot support official promotion.
```

下一步继续 materializer，不做 controller or paired replay。

## Case C：outcome labels materialize but decision gates fail

必须声明：

```text
AC2Q2 stable accept changed the official accept region enough to fail decision gates.
```

下一步修 accept contract / support reliability，不按 dataset 调 threshold。

## Case D：decision gates pass but runtime remains old path

必须声明：

```text
controller is decision-legal, but native stable kernel is not connected to full-system runtime.
```

下一步修 P6 runtime wiring，不做 paired replay。

## Case E：native kernel integrated but system still slow

必须声明：

```text
decision correctness is closed, but full-system runtime remains above envelope.
```

下一步基于 P7 waterfall 修 batch-major bucket / launch / remaining components。

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

v9.2.80 的一句话策略是：

$$
\boxed{
\text{不要再 join 旧 artifact，也不要再局部修 kernel；重跑 full train-stream materializer，把 stable accept、outcome labels 和 native runtime 同步落到每个 candidate row。}
}
$$

当前最关键的问题不是：

```text
stable accept 是否能清零 disagreement；
native CUDA kernel 是否有局部 q90 reduction；
quantile-tail 是否可优化；
basis_norm 是否可局部降低；
PF5 candidate rate 是否可行；
payload binding 是否过；
C3/T2/C4/E2 controller 是否要重调；
Fashion/KMNIST/MNIST 谁更好。
```

而是：

```text
1. 2493 个 candidate rows 是否都有 AC2Q2 score/rank/accept？
2. 2493 个 candidate rows 是否都有 same-run safe_good/bad/null labels？
3. stable accept 与 outcome labels 是否在同一 event lifecycle 中直接绑定？
4. calibration / heldout / support 是否能真实 rerun？
5. native stable kernel 是否进入 P6 full-system train-stream path？
6. P6 是否产生新的 step_ratio_q90，而不是旧 2.713296？
7. P3 local native q90 reduction 是否能在 full-system 保持？
8. system controller 是否 official eligible？
9. LDO/LSO 是否通过？
10. official paired replay 是否打过 AdamWParallel / bestLR？
```

v9.2.80 的结果将给出清晰分叉：

```text
if materialization fails:
  implement materializer; do not run more controller experiments.

if materialization succeeds but decision gates fail:
  repair stable accept contract/support reliability.

if decision gates pass but native runtime not integrated:
  repair P6 wiring.

if native runtime integrated but step ratio >1.50:
  target components from full-system waterfall.

if P6 passes:
  open LDO/LSO and official paired replay.

if P6 passes but paired replay fails:
  system is legal, but functional causal advantage is insufficient.
```
