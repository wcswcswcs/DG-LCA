# DG-KAN v9.2.75 Kernel-Internal Attribution First 与 Static-Bucket Runtime Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.2.75_KernelInternalAttributionFirst_StaticBucketRuntimeClosure_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把局部 materialized runtime 或 gate-blocked downstream 写成 official system pass。

## 0. 最新结论

```text
route = R16-StaticBucketWorkspaceStillUnimplemented
base_candidate = LQ-t2-h256
success_v9275_strict_purekan_functional = False
success_v9275_full_functional = False
success_v9275_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9275_kernel_internal_attribution_first_static_bucket_runtime_closure_first_20260513T133000Z/
```

说明：本轮没有保存单独 stdout `run.log`。可审计日志是上面 artifact 目录中的 `run_manifest.json`、`route_decision.json`、各阶段 `p*.csv`、`*_trace_v9275.csv`、`failure_table.csv`、`artifact_hashes.csv` 和 `v9275_provenance_audit.csv`。

核心结论：

1. P0 复现 v9.2.74 boundary：source route = `R13-KernelInternalAttributionIncomplete`，payload binding pass = `1`，candidate/update payload missing count 全部为 `0`，system legal controller pass = `0`。
2. 本轮 manifest 记录 `device = cuda`，`triton_available = true`；P1 使用真实 MNIST/Fashion-MNIST/KMNIST train-stream targeted CUDA subphase timing，`attribution_steps_per_dataset = 8`。
3. P1 kernel-internal attribution 首次过 gate：`kernel_internal_attribution_pass = 1`，unknown fraction = `0.012946 <= 0.05`，所有必需 subphase timing 字段均有真实值。
4. P1 dominant subcomponent = `basis_norm_time_ms`，平均 `5.664183 ms`，dominant removable/fusible ratio = `0.676676`。
5. P2 materialized selected-feature runtime 过 gate：best = `SF2-AcceptBitTensorGatherRuntimeV2`，`materialized_runtime_path = 1`，`audit_only = 0`，selected-feature materialized count 从 `2493` 降到 `24`，agreement/audit agreement 均为 `1.0`。
6. P3 static bucket / persistent workspace 未闭合：bucket table 只到 diagnostic，`persistent_workspace_used = 0`，kernel/sync/allocation 仍为 `3750/1250/2493`，reduction 全部为 `0.0`。
7. P4 single-pass basis/delta/bridge runtime 未闭合：仍是 `BD0-BF5ReferenceRuntimePlusP1Attribution`，`basis_delta_bridge_single_pass = 0`，`bridge_score_inside_kernel = 0`，step ratio q90 = `2.713296 > 1.50`。
8. P5 integrated materialized runtime candidate 未过：`materialized_runtime_path = 0`，`diagnostic_derived_from_measured_components = 1`，没有形成可提交 SYS survivor。
9. P6 system-legal controller 未打开：decision metrics 仍满足 reference frontier，但 official eligible = `0`，reason = `integrated_materialized_runtime_failed`。
10. P7-P10 因 `P3_static_bucket_workspace_failed` gate-blocked，全部以 `not_run` 落盘。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9275_kernel_internal_attribution_first_static_bucket_runtime_closure.py` | v9.2.75 runner；复现 v9.2.74 boundary，执行真实 kernel-internal attribution、materialized accept-bit selected-feature runtime、static bucket/workspace audit、single-pass basis/delta/bridge gate、integrated runtime candidate 与 system controller gate |

代码检查：

```text
python -m py_compile experiments/run_v9275_kernel_internal_attribution_first_static_bucket_runtime_closure.py
```

正式运行：

```bash
python experiments/run_v9275_kernel_internal_attribution_first_static_bucket_runtime_closure.py \
  --out-dir results/real_rerun_20260506/v9275_kernel_internal_attribution_first_static_bucket_runtime_closure_first_20260513T133000Z \
  --fresh --device auto --data-root data --seed 1314
```

实际参数来自 `run_manifest.json`：

```text
device = cuda
triton_available = true
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7
microprobe_steps = 336
attribution_steps_per_dataset = 8
train_size = 2048
batch_size = 64
hidden_dim = 256
source_v9274_artifact = results/real_rerun_20260506/v9274_materialized_persistent_basis_delta_kernel_static_bucket_system_closure_first_20260513T123000Z
source_v9272_bf5_artifact = results/real_rerun_20260506/v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline_async_basis_cuda_ext_20260513T110000Z
completed_at = 2026-05-13T12:49:26Z
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R16-StaticBucketWorkspaceStillUnimplemented",
  "base_candidate": "LQ-t2-h256",
  "v9274_boundary_pass": 1,
  "payload_binding_contract_pass": 1,
  "candidate_tensor_payload_missing_count": 0,
  "candidate_branch_logits_missing_count": 0,
  "candidate_true_delta_logits_missing_count": 0,
  "functional_update_payload_missing_count": 0,
  "kernel_internal_attribution_pass": 1,
  "unknown_fraction": 0.012945534729646649,
  "dominant_internal_component": "basis_norm_time_ms",
  "basis_lift_time_ms": 0.17796338458235064,
  "basis_quadratic_time_ms": 0.15584472566843033,
  "basis_norm_time_ms": 5.6641833507455885,
  "candidate_gather_time_ms": 0.005767738912254572,
  "W2_delta_time_ms": 1.7883653442064922,
  "probe_logits_time_ms": 0.09546518170585234,
  "selected_feature_writeback_time_ms": 0.13496051542460918,
  "bridge_score_time_ms": 0.18373810841391483,
  "accept_bit_time_ms": 0.09310848933334152,
  "kernel_launch_time_ms": 0.025505626884599526,
  "sync_time_ms": 0.004729799305399259,
  "allocation_time_ms": 0.017298307890693348,
  "best_selected_feature_runtime_id": "SF2-AcceptBitTensorGatherRuntimeV2",
  "selected_feature_runtime_pass": 1,
  "materialized_runtime_path": 1,
  "selected_feature_materialized_count_after": 24,
  "best_bucket_workspace_id": "BW1-FixedBucketCandidateTensorDiagnosticOnly",
  "static_bucket_workspace_pass": 0,
  "persistent_workspace_used": 0,
  "kernel_count_reduction": 0.0,
  "sync_count_reduction": 0.0,
  "allocation_count_reduction": 0.0,
  "avg_candidates_per_kernel_after": 0.6648,
  "best_basis_delta_runtime_id": "BD0-BF5ReferenceRuntimePlusP1Attribution",
  "basis_delta_runtime_pass": 0,
  "basis_delta_bridge_single_pass": 0,
  "bridge_score_inside_kernel": 0,
  "accept_bit_inside_kernel": 1,
  "integrated_runtime_pass": 0,
  "system_legal_controller_pass": 0,
  "official_eligible": 0,
  "primary_blocker": "static_bucket_workspace_not_integrated",
  "next_required_implementation": "implement_static_bucket_persistent_workspace_runtime"
}
```

判断：v9.2.75 真实推进了 P1/P2，但没有完成 integrated runtime。当前 terminal blocker 从 v9.2.74 的 kernel-internal attribution 缺失，推进到 static bucket / persistent workspace 未接入。

## 3. P0 v9.2.74 boundary reproduction

Artifact：

```text
p0_v9274_boundary_reproduction.csv
```

Summary：

```text
source_route_v9274 = R13-KernelInternalAttributionIncomplete
payload_binding_contract_pass = 1
candidate_tensor_payload_missing_count = 0
candidate_branch_logits_missing_count = 0
candidate_true_delta_logits_missing_count = 0
functional_update_payload_missing_count = 0
controller_precision = 0.8380281690140845
controller_coverage = 0.03130511463844797
controller_bad_event = 0.02464788732394366
controller_null_rate = 0.13028169014084506
system_legal_controller_pass = 0
official_eligible = 0
v9274_boundary_pass = 1
```

判断：v9.2.74 的 failed boundary 被稳定复现；没有 payload/decision regression。

## 4. P1 real kernel-internal attribution

Artifacts：

```text
p1_real_kernel_internal_attribution.csv
kernel_internal_cost_trace_v9275.csv
```

Summary：

```text
internal_cost_candidate_id = IA1-TargetedCudaEventSubphaseAttribution
instrumentation_type = real_train_stream_cuda_event_subphase_timing
materialized_runtime_path = 1
sample_step_count = 24
kernel_internal_attribution_pass = 1
unknown_fraction = 0.012945534729646649
dominant_subcomponent = basis_norm_time_ms
dominant_removable_or_fusible_component_ratio = 0.6766760956112658
```

Subphase timing：

| subphase | mean ms |
|---|---:|
| candidate_gather_time_ms | `0.005768` |
| basis_lift_time_ms | `0.177963` |
| basis_quadratic_time_ms | `0.155845` |
| basis_norm_time_ms | `5.664183` |
| W2_delta_time_ms | `1.788365` |
| probe_logits_time_ms | `0.095465` |
| selected_feature_writeback_time_ms | `0.134961` |
| bridge_score_time_ms | `0.183738` |
| accept_bit_time_ms | `0.093108` |
| workspace_writeback_time_ms | `0.023667` |
| kernel_launch_time_ms | `0.025506` |
| sync_time_ms | `0.004730` |
| allocation_time_ms | `0.017298` |

Additional：

```text
cuda_event_total_ms = 8.370597672183067
wallclock_total_ms = 8.480380735515306
kernel_count = 240
sync_count = 240
allocation_count = 72
avg_candidates_per_kernel = 0.3
bytes_read = 15925248
bytes_written = 958464
```

判断：H1 成立。v9.2.75 首次把 aggregate black box 拆成真实 subphase；dominant 是 basis norm / quantile / tail/context construction，而不是 selected-feature writeback。

## 5. P2 materialized selected-feature runtime

Artifacts：

```text
p2_materialized_selected_feature_runtime.csv
selected_feature_runtime_trace_v9275.csv
```

Summary：

```text
selected_feature_runtime_id = SF2-AcceptBitTensorGatherRuntimeV2
mode = accept_bit_tensor_gather_runtime
materialized_runtime_path = 1
audit_only = 0
diagnostic_derived_from_measured_components = 0
selected_feature_materialized_count_before = 2493
selected_feature_materialized_count_after = 24
borderline_count = 24
accept_bit_inside_kernel = 1
bridge_score_inside_kernel = 0
accept_bit_runtime_time_ms = 0.05821790546178818
audit_time_ms = 0.10046130046248436
agreement_reference_accept = 1.0
audit_agreement = 1.0
selected_feature_runtime_pass = 1
```

判断：H2 在 selected-feature runtime 层成立：accept-bit tensor gather 是真实 materialized runtime，不是 audit-only subtraction。但它尚未与 static bucket / single-pass basis-delta-bridge 组成 integrated system path。

## 6. P3 static bucket / persistent workspace materialization

Artifacts：

```text
p3_static_bucket_persistent_workspace_materialization.csv
static_bucket_workspace_trace_v9275.csv
```

Summary：

```text
bucket_workspace_id = BW1-FixedBucketCandidateTensorDiagnosticOnly
bucket_strategy = fixed_bucket_table_materialized_no_integrated_runtime
bucket_sizes = {"1": 7260, "2": 365, "4": 439}
candidate_count = 2493
kernel_count_before = 3750
kernel_count_after = 3750
sync_count_before = 1250
sync_count_after = 1250
allocation_count_before = 2493
allocation_count_after = 2493
persistent_workspace_used = 0
cuda_graph_attempted = 0
cuda_graph_capture_pass = 0
avg_candidates_per_kernel_after = 0.6648
static_bucket_workspace_pass = 0
```

判断：H3 未成立。虽然 bucket occupancy 已被统计，但 static bucket 没有接入 basis/delta runtime，persistent workspace 没有启用，kernel/sync/allocation 没有下降。

## 7. P4 single-pass basis/delta/bridge runtime

Artifacts：

```text
p4_single_pass_basis_delta_bridge_runtime.csv
basis_delta_bridge_runtime_trace_v9275.csv
```

Summary：

```text
basis_delta_runtime_id = BD0-BF5ReferenceRuntimePlusP1Attribution
selected_feature_runtime_id = SF2-AcceptBitTensorGatherRuntimeV2
bucket_workspace_id = BW1-FixedBucketCandidateTensorDiagnosticOnly
uses_true_branch_delta = 1
uses_source_measured_gap = 0
uses_formula_proxy = 0
basis_delta_bridge_single_pass = 0
bridge_score_inside_kernel = 0
accept_bit_inside_kernel = 1
cuda_vs_torch_check_count = 24
cuda_vs_torch_logits_error_max = 2.6702880859375e-05
cuda_vs_torch_delta_error_max = 8.149072527885437e-10
agreement_reference_accept = 1.0
step_ratio_q90 = 2.713295831053225
basis_delta_runtime_pass = 0
```

判断：H4 未成立。accept-bit runtime 已有，但 basis/delta/bridge 仍不是 single-pass；bridge score 也不在 kernel 内，因此不能写成 full runtime candidate。

## 8. P5 integrated materialized runtime candidates

Artifacts：

```text
p5_integrated_materialized_runtime_candidates.csv
integrated_runtime_trace_v9275.csv
```

Summary：

```text
system_candidate_id = SYS2-IA1PlusSF2NoStaticBucket
selected_feature_runtime_id = SF2-AcceptBitTensorGatherRuntimeV2
bucket_workspace_id = BW1-FixedBucketCandidateTensorDiagnosticOnly
basis_delta_runtime_id = BD0-BF5ReferenceRuntimePlusP1Attribution
materialized_runtime_path = 0
audit_only_cost_removal = 0
diagnostic_derived_from_measured_components = 1
kernel_count = 3750
sync_count = 1250
allocation_count = 2493
avg_candidates_per_kernel = 0.6648
step_ratio_q90 = 2.713295831053225
integrated_runtime_pass = 0
```

判断：P5 没有 integrated survivor。这里没有把 P2 的局部 accept-bit runtime 倒灌成 whole-system runtime。

## 9. P6 system-legal controller boundary

Artifacts：

```text
p6_system_legal_exact_signal_controller_v7.csv
system_controller_trace_v9275.csv
```

Boundary：

```text
controller_id = C3-T2PlusBackfill
system_candidate_id = SYS2-IA1PlusSF2NoStaticBucket
status = not_run
official_eligible = 0
system_legal_controller_pass = 0
reason = integrated_materialized_runtime_failed
precision_heldout = 0.8380281690140845
coverage_heldout = 0.03130511463844797
bad_event_heldout = 0.02464788732394366
null_rate_heldout = 0.13028169014084506
precision_lcb = 0.7907157243773478
bad_event_ucb = 0.04999458813129621
agreement_reference_accept = 1.0
step_ratio_q90 = 2.713295831053225
memory_ratio = 1.0
accepted_signal_strata_count = 15
accepted_family_count = 65
```

判断：decision/support/payload binding 仍满足 reference frontier；official blocker 是 integrated materialized runtime 未闭合，不是 controller 质量问题。

## 10. Downstream boundary

这些 artifact 均已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p7_leave_dataset_and_stratum_out.csv` | `P3_static_bucket_workspace_failed` |
| `p8_official_paired_replay.csv` | same |
| `p9_short_run_functional_validation.csv` | same |
| `p10_full_run_robustness_strong_baseline.csv` | same |

没有把 P1 attribution pass、P2 accept-bit runtime pass、decision metrics 或 payload binding pass 写成 LDO/LSO、paired replay、short-run 或 full-run success。

## 11. No-fake audit

```text
rows_checked = 69
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

Contract audit：

```text
manual_forward/manual_backward/manual_adamw_update = 1/1/1
train_stream_probe = 1
payload_binding_contract_pass = 1
candidate_tensor_payload_missing_count = 0
candidate_branch_logits_missing_count = 0
candidate_true_delta_logits_missing_count = 0
functional_update_payload_missing_count = 0
kernel_internal_attribution_real_timing = 1
materialized_selected_feature_runtime = 1
static_bucket_workspace_used = 0
basis_delta_bridge_single_pass = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
projection_used_for_official = 0
```

## 12. Hash

| artifact | SHA256 |
|---|---|
| plan | `f1fd2a8734bdb3afcd2906a3d3bf64bd0fc9157304da2f5dd64ae4ce0d410a61` |
| runner | `0f3ae895a8aa407912148ea0559d111a2408eeee754f7d3552c5bbe728bab616` |
| run manifest | `4cd299330216094de46c38174ce17e0b544c9763825aab24bdcd0fdcbef73e84` |
| route | `de46b47b36b76cc315e15625de583b4a12ba3e8b9b375be6a85ef6baf1e5addd` |
| P0 boundary | `094d42a920647f151494493d834dc15aedd6ef08b32493ffe7baabb7e75c325c` |
| P1 kernel attribution | `1987e0d55ac99e07cce1ef1da6f62295ed5019002b75f7dc85e7769bba29a093` |
| P2 selected feature | `1ee0e8bfe4bab97d916c66d32584e25d49b93fa385381754df70724250e08b40` |
| P3 static bucket | `5a8e6da35bd5ed286fea8d2ddcb3db60d00581d77a590a35104e5b672e986c7d` |
| P4 basis/delta bridge | `61c7c53487c5e39028fe2a48b9691f5cd91c6d0c6d72a8d3414bf87af0f5d65b` |
| P5 integrated runtime | `6a4a635a90264268313389fd382ac3ba5203f238737c68d42ab618bd9ffcc6da` |
| P6 system controller | `055f21a727effd416142a48e30bd061c1056b720230c047d490e3f0005b4781f` |
| P7 boundary | `529b740675518163167ffbc4414f36510a12fafb8f5423eeefc355de3a492f04` |
| P8 boundary | `fec17a312ab3eb3a75578110e9d8ff6d377fd4e3ddc941de96b915d9e8091ee8` |
| P9 boundary | `1ff6f10b212d3bd8aac3282554776364154b479f925f77f817ad6856bb42cc9d` |
| P10 boundary | `4aaa74b0e837b38f4b1d377b93f6b1db844e496318b4d76291cbb8e76d7101a8` |
| failure table | `9fb032f80bdb476ffb8747ad68f7eeec8e58a8cc52df1026cf09d44f9157bc66` |
| provenance audit | `8e60dbbce60ec8b22cc9e0adc4f83bc5874823b3086da23dbcc2a15cdf4a4d5f` |

## 13. 最终分析结论

v9.2.75 的真实推进是：

```text
v9.2.74: lower-level runtime pieces 未 materialize；
          terminal blocker 是 kernel-internal attribution incomplete。
v9.2.75: P1 kernel-internal attribution 真实闭合；
          P2 selected-feature accept-bit runtime 真实 materialize；
          但 static bucket / persistent workspace 和 single-pass basis-delta-bridge 未闭合。
```

机制判断：

1. H1 成立：aggregate black box 被拆出真实 subphase，dominant 是 `basis_norm_time_ms`。
2. H2 成立于局部 runtime：accept-bit tensor gather 把 selected-feature materialized count 从 `2493` 降到 `24`，agreement = `1.0`。
3. H3 未成立：static bucket / persistent workspace 未接入 runtime，kernel/sync/allocation 没有下降。
4. H4 未成立：basis/delta/bridge 仍不是 single-pass，bridge score 不在 kernel 内。
5. H5 尚不能声明 lower bound：因为 P3/P4 还没 materialize，不能说当前 cost 是不可降低下界。
6. P6 未打开：`official_eligible = 0`，因此 LDO/LSO、paired replay、short/full validation 全部 gate-blocked。

最终一句话：

> v9.2.75 真实执行后停在 `R16-StaticBucketWorkspaceStillUnimplemented`：kernel-internal attribution 和 selected-feature accept-bit runtime 已经真实推进，但 static bucket / persistent workspace 仍未接入，single-pass basis-delta-bridge 未闭合，system-legal controller 仍不能转正。
