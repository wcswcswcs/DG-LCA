# DG-KAN v9.2.74 Materialized Persistent Basis/Delta Kernel 与 Static-Bucket System Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.2.74_MaterializedPersistentBasisDeltaKernel_StaticBucketSystemClosure_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把诊断性 audit-only cost removal 写成 official system pass。

## 0. 最新结论

```text
route = R13-KernelInternalAttributionIncomplete
base_candidate = LQ-t2-h256
success_v9274_strict_purekan_functional = False
success_v9274_full_functional = False
success_v9274_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9274_materialized_persistent_basis_delta_kernel_static_bucket_system_closure_first_20260513T123000Z/
```

说明：本轮没有保存单独 stdout `run.log`。可审计日志是上面 artifact 目录中的 `run_manifest.json`、`route_decision.json`、各阶段 `p*.csv`、`*_trace_v9274.csv`、`failure_table.csv`、`artifact_hashes.csv` 和 `v9274_provenance_audit.csv`。

核心结论：

1. P0 复现 v9.2.73 boundary：source route = `R18-SystemStillTooExpensive`，payload binding pass = `1`，candidate/update payload missing count 全部为 `0`，system legal controller pass = `0`。
2. 本轮 manifest 记录 `device = cuda`，`triton_available = true`；source artifact 为 v9.2.73 first run 和 v9.2.72 BF4/BF5 runs。
3. P1 kernel-internal attribution 未闭合：只能复用 BF5 aggregate component，`basis_lift_time_ms`、`basis_quadratic_time_ms`、`basis_norm_time_ms`、`kernel_launch_time_ms`、`sync_time_ms` 等字段仍为空。
4. P1 route 按计划停在 `R13-KernelInternalAttributionIncomplete`，没有把 aggregate attribution 写成 kernel-internal attribution success。
5. P2 selected-feature runtime 未 materialize：`SF2-AcceptBitInsideKernelRuntime` 为 best candidate，但 `materialized_runtime_path = 0`，`selected_feature_materialized_count_after = 2493`，pass = `0`。
6. P3 static bucket / persistent workspace / CUDA graph 未实现：kernel count = `3750`、sync count = `1250`、allocation count = `2493`，reduction 全部为 `0.0`。
7. P4 single-pass basis/delta/bridge runtime 未过：BF5 reference runtime agreement = `1.0`，CUDA-vs-torch logits error = `2.670288e-05`，delta error = `8.149073e-10`，但 bridge score / accept bit 不在 kernel 内，step ratio q90 = `2.713296 > 1.50`。
8. P5 audit separation 通过：timed path 内 hash/csv/cpu metadata/norm item 均为 `0`，audit disagreement = `0`。
9. P6 system controller 未打开：decision/support/payload metrics 仍满足 reference frontier，但 official eligible = `0`，reason = `kernel_internal_or_selected_feature_or_static_bucket_runtime_failed`。
10. P7-P10 因 `P1_kernel_internal_attribution_incomplete` gate-blocked，均以 `not_run` 落盘。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9274_materialized_persistent_basis_delta_kernel_static_bucket_system_closure.py` | v9.2.74 runner；复现 v9.2.73 boundary，审计 kernel-internal attribution、selected-feature materialized runtime、static bucket/workspace/CUDA graph、single-pass basis/delta/bridge runtime、audit separation 与 system controller gate |

代码检查：

```text
python -m py_compile experiments/run_v9274_materialized_persistent_basis_delta_kernel_static_bucket_system_closure.py
```

正式运行：

```bash
python experiments/run_v9274_materialized_persistent_basis_delta_kernel_static_bucket_system_closure.py \
  --out-dir results/real_rerun_20260506/v9274_materialized_persistent_basis_delta_kernel_static_bucket_system_closure_first_20260513T123000Z \
  --fresh --device auto --data-root data --seed 1314
```

实际参数来自 `run_manifest.json`：

```text
device = cuda
triton_available = true
source_v9273_artifact = results/real_rerun_20260506/v9273_step_level_basis_delta_fusion_system_legal_controller_closure_first_20260513T113000Z
source_v9272_bf5_artifact = results/real_rerun_20260506/v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline_async_basis_cuda_ext_20260513T110000Z
source_v9272_bf4_artifact = results/real_rerun_20260506/v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline_cuda_ext_20260513T103000Z
completed_at = 2026-05-13T12:06:42Z
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R13-KernelInternalAttributionIncomplete",
  "base_candidate": "LQ-t2-h256",
  "v9273_boundary_pass": 1,
  "payload_binding_contract_pass": 1,
  "candidate_tensor_payload_missing_count": 0,
  "candidate_branch_logits_missing_count": 0,
  "candidate_true_delta_logits_missing_count": 0,
  "functional_update_payload_missing_count": 0,
  "kernel_internal_attribution_pass": 0,
  "unknown_fraction": 0.0,
  "dominant_internal_component": "fused_common_basis_delta_pipeline_ms",
  "W2_delta_time_ms": 111.27027263864875,
  "selected_feature_writeback_time_ms": 323.1187085621059,
  "best_selected_feature_runtime_id": "SF2-AcceptBitInsideKernelRuntime",
  "selected_feature_runtime_pass": 0,
  "selected_feature_materialized_count_after": 2493,
  "best_bucket_workspace_id": "BW0-V9273NoWorkspaceReference",
  "static_bucket_workspace_pass": 0,
  "kernel_count_reduction": 0.0,
  "sync_count_reduction": 0.0,
  "allocation_count_reduction": 0.0,
  "best_basis_delta_runtime_id": "BD0-BF5ReferenceRuntime",
  "basis_delta_runtime_pass": 0,
  "cuda_vs_torch_logits_error_max": 2.6702880859375e-05,
  "cuda_vs_torch_delta_error_max": 8.149072527885437e-10,
  "basis_delta_agreement": 1.0,
  "basis_delta_step_ratio_q90": 2.713295831053225,
  "audit_separation_pass": 1,
  "system_legal_controller_pass": 0,
  "official_eligible": 0,
  "controller_step_ratio_q90": 2.713295831053225,
  "primary_blocker": "kernel_internal_attribution_incomplete",
  "next_required_implementation": "instrument_real_kernel_internal_timing_and_workspace"
}
```

判断：v9.2.74 没有 payload/decision regression；失败点是更低层 persistent basis/delta kernel 没有真实 materialize，也没有得到 kernel-internal timing。按计划，第一 terminal blocker 是 P1。

## 3. P0 v9.2.73 boundary reproduction

Artifact：

```text
p0_v9273_boundary_reproduction.csv
```

Summary：

```text
source_route_v9273 = R18-SystemStillTooExpensive
payload_binding_contract_pass = 1
candidate_tensor_payload_missing_count = 0
candidate_branch_logits_missing_count = 0
candidate_true_delta_logits_missing_count = 0
functional_update_payload_missing_count = 0
controller_precision = 0.8380281690140845
controller_coverage = 0.03130511463844797
controller_bad_event = 0.02464788732394366
controller_null_rate = 0.13028169014084506
bd1_step_ratio_q90 = 2.3464118796089455
kernel_count = 3750
sync_count = 1250
allocation_count = 2493
selected_feature_time_ms = 323.1187085621059
v9273_boundary_pass = 1
```

判断：v9.2.73 的 failed boundary 被正确复现，没有把 v9.2.73 的 audit-only `2.346412` 写成 system success。

## 4. P1 kernel-internal cost attribution

Artifacts：

```text
p1_kernel_internal_cost_attribution.csv
kernel_internal_cost_trace_v9274.csv
```

| row | source | total wallclock ms | kernel | sync | allocation | pass |
|---|---|---:|---:|---:|---:|---:|
| IA0-V9273AggregateReference | BF5 aggregate | `1410.062380` | `3750` | `1250` | `2493` | `0` |
| IA1-BF4SeparateCommonAndDeltaReference | BF4 landed timing | `1440.329792` | `3750` | `1250` | `2493` | `0` |

Summary：

```text
kernel_internal_attribution_pass = 0
unknown_fraction = 0.0
dominant_internal_component = fused_common_basis_delta_pipeline_ms
basis_lift_time_ms = ""
basis_quadratic_time_ms = ""
basis_norm_time_ms = ""
kernel_launch_time_ms = ""
sync_time_ms = ""
failure_reason = kernel_internal_timing_fields_missing_for_basis_lift_quadratic_norm_launch_sync
```

判断：P1 只能在 landed aggregate component 层闭合，不能声称完成 kernel-internal attribution。v9.2.74 按计划停在 `R13-KernelInternalAttributionIncomplete`。

## 5. P2 materialized selected-feature elimination

Artifacts：

```text
p2_materialized_selected_feature_elimination.csv
selected_feature_runtime_trace_v9274.csv
```

| candidate | mode | selected materialized after | step q90 | pass |
|---|---|---:|---:|---:|
| SF0-V9273AuditOnlyReference | audit only | `0` | `2.346412` | `0` |
| SF2-AcceptBitInsideKernelRuntime | not implemented | `2493` | `2.713296` | `0` |

Summary：

```text
best_selected_feature_runtime_id = SF2-AcceptBitInsideKernelRuntime
selected_feature_runtime_pass = 0
materialized_runtime_path = 0
selected_feature_materialized_count_after = 2493
failure_reason = accept_bit_inside_kernel_runtime_not_materialized
```

判断：selected-feature elimination 仍停留在 v9.2.73 的 audit-only 诊断层。本轮没有把 bridge score / accept bit inside kernel 落成真实 runtime。

## 6. P3 static bucket / persistent workspace / CUDA graph

Artifacts：

```text
p3_static_bucket_persistent_workspace.csv
static_bucket_workspace_trace_v9274.csv
```

Summary：

```text
best_bucket_workspace_id = BW0-V9273NoWorkspaceReference
static_bucket_workspace_pass = 0
kernel_count_before = 3750
kernel_count_after = 3750
sync_count_before = 1250
sync_count_after = 1250
allocation_count_before = 2493
allocation_count_after = 2493
kernel_count_reduction = 0.0
sync_count_reduction = 0.0
allocation_count_reduction = 0.0
persistent_workspace_used = 0
cuda_graph_attempted = 0
cuda_graph_capture_pass = 0
```

判断：P3 没有完成 static bucket、persistent workspace 或 CUDA graph capture；kernel/sync/allocation 都没有下降。

## 7. P4 single-pass basis/delta/bridge runtime

Artifacts：

```text
p4_single_pass_basis_delta_bridge_runtime.csv
basis_delta_bridge_runtime_trace_v9274.csv
```

Summary：

```text
best_basis_delta_runtime_id = BD0-BF5ReferenceRuntime
basis_delta_runtime_pass = 0
uses_true_branch_delta = 1
uses_source_measured_gap = 0
uses_formula_proxy = 0
basis_delta_bridge_single_pass = 0
bridge_score_inside_kernel = 0
accept_bit_inside_kernel = 0
cuda_vs_torch_logits_error_max = 2.6702880859375e-05
cuda_vs_torch_delta_error_max = 8.149072527885437e-10
basis_delta_agreement = 1.0
basis_delta_step_ratio_q90 = 2.713295831053225
```

判断：数值一致性仍保持，但 runtime 没有达到 v9.2.74 所需的 single-pass basis/delta/bridge kernel；step ratio 也仍高于 `1.50`。

## 8. P5 audit separation / no-hash timed path

Artifacts：

```text
p5_audit_separation_official_timed_path.csv
audit_separation_trace_v9274.csv
```

Summary：

```text
audit_separation_pass = 1
hash_in_timed_path = 0
csv_write_in_timed_path = 0
cpu_metadata_in_timed_path = 0
norm_item_in_timed_path = 0
selected_feature_audit_in_timed_path = 0
audit_disagreement_count = 0
```

判断：timed path 没有 hash/csv/cpu metadata/norm item 污染；当前 failure 不是 artifact I/O 或审计写盘导致。

## 9. P6 system controller boundary

Artifacts：

```text
p6_system_legal_exact_signal_controller_v6.csv
system_controller_trace_v9274.csv
```

Boundary：

```text
controller_id = C3-T2PlusBackfill
status = not_run
official_eligible = 0
system_legal_controller_pass = 0
reason = kernel_internal_or_selected_feature_or_static_bucket_runtime_failed
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

判断：decision/support/payload binding 仍满足 reference frontier；唯一 blocker 是 lower-level materialized runtime 没闭合。因此不能打开 system-legal controller。

## 10. Downstream boundary

这些 artifact 均已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p7_leave_dataset_and_stratum_out.csv` | `P1_kernel_internal_attribution_incomplete` |
| `p8_official_paired_replay.csv` | same |
| `p9_short_run_functional_validation.csv` | same |
| `p10_full_run_robustness_strong_baseline.csv` | same |

没有把 audit-only selected-feature removal、BF5 reference runtime、decision metrics 或 payload binding pass 写成 LDO/LSO、paired replay、short-run 或 full-run success。

## 11. No-fake audit

```text
rows_checked = 29
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
materialized_selected_feature_runtime = 0
static_bucket_workspace_used = 0
basis_delta_bridge_single_pass = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
projection_used_for_official = 0
```

## 12. Hash

| artifact | SHA256 |
|---|---|
| plan | `68b5858556b6d30cea2a623ec8397133330461bd2e51197560328273d668820b` |
| runner | `c0f9588994de5fc092a233f35454b13e034eb7571d6c3175872e3999de3af991` |
| run manifest | `75c5d6f0b55d24c47ff196fd0148d97efe0f059e99bead0c34a2117805b14337` |
| route | `332aa15d2758b2b2d7f580e365a5e3175164e2636aa532e14859b741727e4d5e` |
| P0 boundary | `413ca255f889e0fc7018b3314a0c077c5a7cdaae72b399d886ebaeaf3de30432` |
| P1 kernel attribution | `636de66ed0fabcfc00b86ac51b0307979c99a96b0e0e794827fe6167e73de0a7` |
| P2 selected feature | `6bb1820d932137fa7ef3f1c34cde5a12b32308b60db13ffd921d85470776c6fd` |
| P3 static bucket | `90e7523264218ceb6a8c65b7c6dcc72640365a5f474482ffe74033d4eb7498e3` |
| P4 basis/delta bridge | `05481e38bbe7e429e4b6df8480fbd0aaed70a26a2db8e5f8fad27bf9b48a998a` |
| P5 audit separation | `b1c6fd1357a5ee5d111c38b3cfe68494d2f04014f147cb25698f564f9cfbe31b` |
| P6 system controller | `071b61ecdb599988317aadfa5b6256c0113bf45e7c91a3b51cc51f2ee702f69f` |
| P7 boundary | `24dff0b5390f15ac2129363704910071b92b1963eb96f6ae4ddbea8e83979960` |
| P8 boundary | `ab8bc17add2d0d8804a01c69b2a33a8b57c94d1535b67ed893c0a4fbc1862eca` |
| P9 boundary | `79ef1818e12920b7c5cf50d3abdca5f94b1481ca22349dc9cc9653295f7774be` |
| P10 boundary | `33dfdd3dfa9ba18aa32807058569066bd0dc9b805398f62982479fc6e6a451bf` |
| failure table | `c72239605b4527271a5b2097831adbb64225a98616cd49d91b21d031a39f7b68` |
| provenance audit | `d6cd59e54f873275c7fa634e8203dff12f35bd33fb863ca00ab26225b08c330d` |

## 13. 最终分析结论

v9.2.74 的真实推进是：

```text
v9.2.73: BF5 剩余 cost 只能拆到 aggregate component；
          selected-feature removal 只是 audit-only diagnostic。
v9.2.74: 严格审计 persistent basis/delta kernel、selected-feature runtime、
          static bucket/workspace/CUDA graph 与 single-pass bridge closure；
          结果这些 materialized runtime gate 均未闭合。
```

机制判断：

1. H1 未成立：kernel-internal basis lift / quadratic / norm / launch / sync timing 没有落盘，因此 P1 未过。
2. H2 未成立：accept-bit / bridge-score inside kernel runtime 没有 materialize，selected-feature rows 仍为 `2493`。
3. H3 未成立：static bucket、persistent workspace、CUDA graph capture 都没有实现，kernel/sync/allocation 没有下降。
4. H4 未成立：single-pass basis/delta/bridge runtime 仍是 BF5 reference，bridge score / accept bit 不在 kernel 内，step q90 仍 `2.713296`。
5. H5 成立：audit separation 通过，timed path 没有 hash/csv/cpu metadata/norm item 污染。
6. H6 未打开：P6 `official_eligible = 0`，因此 LDO/LSO、paired replay、short/full validation 全部 gate-blocked。

最终一句话：

> v9.2.74 真实执行后停在 `R13-KernelInternalAttributionIncomplete`：payload/decision 没有回退，但 materialized persistent basis/delta kernel、selected-feature runtime、static bucket/workspace/CUDA graph 都未闭合，system-legal controller 仍不能转正。
