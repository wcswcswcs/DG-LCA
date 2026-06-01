# DG-KAN v9.2.73 Step-Level Basis/Delta Fusion 与 System-Legal Controller Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.2.73_StepLevelBasisDeltaFusion_SystemLegalControllerClosure_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把诊断性 cost removal 或 gate-blocked downstream 写成 official system pass。

## 0. 最新结论

```text
route = R18-SystemStillTooExpensive
base_candidate = LQ-t2-h256
success_v9273_strict_purekan_functional = False
success_v9273_full_functional = False
success_v9273_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9273_step_level_basis_delta_fusion_system_legal_controller_closure_first_20260513T113000Z/
```

核心结论：

1. P0 复现 v9.2.72 boundary：source route = `R18-PayloadBoundSystemStillExpensive`，payload binding pass = `1`，candidate/update payload missing count 全部为 `0`，system legal controller pass = `0`。
2. 本轮 manifest 记录 `device = cuda`，`triton_available = true`；source artifact 为 v9.2.72 BF5 async combined basis+delta 路径。
3. P1 BF5 step cost attribution 闭合：unknown fraction = `0.0`，dominant internal component = `fused_common_basis_delta_pipeline_ms`，total = `1086.943672 ms`，selected feature component = `323.118709 ms`。
4. P1 只在已落盘 BF5 aggregate component 级别闭合，没有得到 basis lift / quadratic / norm / kernel launch 等更细粒度拆分；这些字段仍为空。
5. P2 best diagnostic candidate = `BD1-AuditOnlySelectedFeatureCostRemovalDiagnostic`，agreement = `1.0`，CUDA-vs-torch logits error = `2.670288e-05`，delta error = `8.149073e-10`，但 step ratio q90 = `2.346412 > 1.50`。
6. P3 selected-feature materialization elimination 是 audit-only diagnostic：selected feature total 从 `323.118709 ms` 记为 `0.0` 后，step q90 仍是 `2.346412`，system candidate pass = `0`。
7. P4 static bucket / persistent workspace / CUDA graph 未实现：kernel count 仍 `3750`，sync count 仍 `1250`，allocation count 仍 `2493`，pass = `0`。
8. P5 no-hash timed path audit 通过：hash/csv/cpu metadata/norm item in timed path 均为 `0`，audit disagreement = `0`，但 system candidate pass = `0`。
9. P6 system controller 未打开：decision metrics 仍满足 reference frontier，但 `step_ratio_q90 = 2.346412 > 1.50`，official eligible = `0`。
10. P7-P10 因 `P6_system_controller_step_ratio_failed` gate-blocked，均以 `not_run` 落盘。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9273_step_level_basis_delta_fusion_system_legal_controller_closure.py` | v9.2.73 runner；复现 v9.2.72 BF5 boundary，执行 BF5 internal cost attribution、basis/delta/feature fusion diagnostic、selected-feature materialization elimination audit、static bucket/workspace audit、async no-hash audit、system controller gate |

代码检查：

```text
python -m py_compile experiments/run_v9273_step_level_basis_delta_fusion_system_legal_controller_closure.py
```

正式运行：

```bash
python experiments/run_v9273_step_level_basis_delta_fusion_system_legal_controller_closure.py \
  --out-dir results/real_rerun_20260506/v9273_step_level_basis_delta_fusion_system_legal_controller_closure_first_20260513T113000Z \
  --fresh --device auto --data-root data --seed 1314
```

实际参数来自 `run_manifest.json`：

```text
device = cuda
triton_available = true
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7
microprobe_steps = 336
train_size = 2048
batch_size = 64
hidden_dim = 256
source_artifact = results/real_rerun_20260506/v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline_async_basis_cuda_ext_20260513T110000Z
completed_at = 2026-05-13T11:29:00Z
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R18-SystemStillTooExpensive",
  "base_candidate": "LQ-t2-h256",
  "v9272_boundary_pass": 1,
  "payload_binding_contract_pass": 1,
  "bf5_internal_cost_attribution_pass": 1,
  "unknown_fraction": 0.0,
  "dominant_internal_component": "fused_common_basis_delta_pipeline_ms",
  "selected_feature_time_ms": 323.1187085621059,
  "best_basis_delta_candidate_id": "BD1-AuditOnlySelectedFeatureCostRemovalDiagnostic",
  "basis_delta_feature_fusion_pass": 0,
  "basis_delta_agreement": 1.0,
  "cuda_vs_torch_logits_error_max": 2.6702880859375e-05,
  "cuda_vs_torch_delta_error_max": 8.149072527885437e-10,
  "basis_delta_step_ratio_q90": 2.3464118796089455,
  "selected_feature_elimination_pass": 1,
  "selected_feature_time_reduction": 1.0,
  "static_bucket_workspace_pass": 0,
  "kernel_count_reduction": 0.0,
  "sync_count_reduction": 0.0,
  "allocation_count_after": 2493,
  "async_audit_pass": 1,
  "hash_in_timed_path": 0,
  "cpu_metadata_in_timed_path": 0,
  "system_legal_controller_pass": 0,
  "official_eligible": 0,
  "controller_step_ratio_q90": 2.3464118796089455,
  "primary_blocker": "basis_delta_selected_feature_pipeline_still_expensive",
  "next_required_implementation": "deeper_basis_feature_kernel_fusion_or_reduce_selected_feature_payload"
}
```

判断：v9.2.73 没有 payload/decision regression；但即使把 selected-feature materialization 作为 audit-only cost removal 处理，best system q90 仍远高于 `1.50`。因此 route 只能停在 `R18-SystemStillTooExpensive`。

## 3. P0 v9.2.72 boundary reproduction

Artifact：

```text
p0_v9272_boundary_reproduction.csv
```

Summary：

```text
source_route_v9272 = R18-PayloadBoundSystemStillExpensive
payload_binding_contract_pass = 1
candidate_tensor_payload_missing_count = 0
candidate_branch_logits_missing_count = 0
candidate_true_delta_logits_missing_count = 0
functional_update_payload_missing_count = 0
candidate_rate = 0.10305059523809523
reference_accept_recall = 1.0
controller_precision = 0.8380281690140845
controller_coverage = 0.03130511463844797
controller_bad_event = 0.02464788732394366
controller_null_rate = 0.13028169014084506
bf5_step_ratio_q90 = 2.713295831053225
system_legal_controller_pass = 0
```

判断：v9.2.72 的 BF5 payload-bound path 被正确作为边界复现；没有把 BF5 的 `2.713296` 写成 system success。

## 4. P1 BF5 internal step cost attribution

Artifacts：

```text
p1_bf5_internal_step_cost_attribution.csv
bf5_internal_cost_trace_v9273.csv
```

| component | total ms | ratio of step extra | scope |
|---|---:|---:|---|
| fused_common_basis_delta_pipeline_ms | `1086.943672` | `0.770848` | measured BF5 aggregate component |
| true_delta_selected_feature_ms | `323.118709` | `0.229152` | measured BF5 component |
| unknown_ms | `0.000000` | `0.000000` | landed step trace |

Summary：

```text
bf5_internal_cost_attribution_pass = 1
unknown_fraction = 0.0
dominant_internal_component = fused_common_basis_delta_pipeline_ms
total_wallclock_time_ms = 1410.0623801350594
kernel_count = 3750
sync_count = 1250
allocation_count = 2493
avg_candidates_per_kernel = 0.6648
```

判断：本轮能把 v9.2.72 BF5 的剩余 cost 分到 landed aggregate component，但不能声称完成了 kernel-internal attribution。`basis_lift_time_ms`、`basis_quadratic_time_ms`、`basis_norm_time_ms`、`W2_delta_time_ms`、`probe_logits_time_ms`、`sync_time_ms` 等细粒度字段仍为空，说明下一步需要在 fused kernel 内继续埋点或更深重写。

## 5. P2 basis/delta/feature single-pass fusion diagnostic

Artifacts：

```text
p2_basis_delta_feature_single_pass_fusion.csv
basis_delta_feature_kernel_trace_v9273.csv
```

| candidate | step q90 | logits err | delta err | system candidate |
|---|---:|---:|---:|---:|
| BD0-BF5Reference | `2.713296` | `2.670288e-05` | `8.149073e-10` | `0` |
| BD1-AuditOnlySelectedFeatureCostRemovalDiagnostic | `2.346412` | `2.670288e-05` | `8.149073e-10` | `0` |
| BD2-FusedCommonOnlyLowerBoundDiagnostic | `2.346412` | `2.670288e-05` | `8.149073e-10` | `0` |

Summary：

```text
best_basis_delta_candidate_id = BD1-AuditOnlySelectedFeatureCostRemovalDiagnostic
basis_delta_feature_fusion_pass = 0
basis_delta_system_candidate_pass = 0
basis_delta_agreement = 1.0
basis_delta_step_ratio_q90 = 2.3464118796089455
```

判断：数值一致性继续保持，decision metrics 没有 drift；但 `BD1/BD2` 是从已测 BF5 component 做的 diagnostic cost accounting，不是新的 official measured fused kernel。它们仍未达到 `step_ratio_q90 <= 1.50`。

## 6. P3 selected-feature materialization elimination

Artifacts：

```text
p3_selected_feature_materialization_elimination.csv
selected_feature_elimination_trace_v9273.csv
```

Summary：

```text
selected_feature_candidate_id = SF3-AuditOnlySelectedFeature
mode = audit_only_selected_feature_diagnostic
selected_feature_total_ms_before = 323.1187085621059
selected_feature_total_ms_after = 0.0
selected_feature_time_reduction = 1.0
audit_subset_count = 24
agreement_reference_accept = 1.0
audit_agreement = 1.0
step_ratio_q90 = 2.3464118796089455
selected_feature_elimination_pass = 1
system_candidate_pass = 0
diagnostic_derived_from_measured_components = 1
```

判断：selected-feature payload 是真实成本项，理论上去除后有改善空间；但这是 audit-only diagnostic，不能写成 official runtime pass。即使完全扣掉该 component，step q90 仍是 `2.346412`，系统仍失败。

## 7. P4 static bucket / persistent workspace / CUDA graph

Artifacts：

```text
p4_static_bucket_persistent_workspace_cuda_graph.csv
static_bucket_workspace_trace_v9273.csv
```

Summary：

```text
bucket_candidate_id = BD5-PersistentWorkspaceDiagnosticNotImplemented
bucket_sizes = 1,2,3 observed carriers per step
candidate_count = 2493
fused_step_count_before = 1250
fused_step_count_after = 1250
kernel_count_before = 3750
kernel_count_after = 3750
sync_count_before = 1250
sync_count_after = 1250
allocation_count_before = 2493
allocation_count_after = 2493
cuda_graph_attempted = 0
cuda_graph_capture_pass = 0
persistent_workspace_used = 0
static_bucket_workspace_pass = 0
```

判断：P4 没有完成 static bucket / persistent workspace / CUDA graph capture。kernel/sync/allocation 都没有下降，因此不能把这一阶段写成 cost closure。

## 8. P5 async audit / no-hash timed path

Artifacts：

```text
p5_async_audit_nohash_timed_path.csv
async_audit_trace_v9273.csv
```

Summary：

```text
audit_candidate_id = AS5-NoHashTimedPathWithPostAudit
hash_in_timed_path = 0
csv_write_in_timed_path = 0
cpu_metadata_in_timed_path = 0
norm_item_in_timed_path = 0
post_step_audit_used = 1
audit_subset_size = 24
audit_disagreement_count = 0
step_ratio_q90 = 2.3464118796089455
async_audit_pass = 1
system_candidate_pass = 0
```

判断：timed path 内没有 hash/csv/cpu metadata/norm item 倒灌；cost failure 不是 artifact I/O 或 hash schema 问题。

## 9. P6 system-legal controller boundary

Artifacts：

```text
p6_system_legal_exact_signal_controller_v5.csv
system_controller_trace_v9273.csv
```

Boundary：

```text
controller_id = C3-T2PlusBackfill
system_candidate_id = SYS7-HybridBestV9273
status = not_run
official_eligible = 0
system_legal_controller_pass = 0
reason = step_ratio_q90_still_above_1p50
precision_heldout = 0.8380281690140845
coverage_heldout = 0.03130511463844797
bad_event_heldout = 0.02464788732394366
null_rate_heldout = 0.13028169014084506
precision_lcb = 0.7907157243773478
bad_event_ucb = 0.04999458813129621
agreement_reference_accept = 1.0
step_ratio_q90 = 2.3464118796089455
memory_ratio = 1.0
accepted_signal_strata_count = 15
accepted_family_count = 65
max_family_share = 0.09507042253521127
max_stratum_share = 0.15140845070422534
full_online_row_binding = 1
full_online_payload_binding = 1
full_online_update_payload_binding = 1
projection_used = 0
source_measured_gap_used = 0
formula_proxy_used = 0
cpu_offload_used = 0
```

判断：controller 的 decision/support/payload binding 仍满足 reference frontier；唯一 official blocker 是系统成本 `2.346412 > 1.50`。因此 P6 不能打开。

## 10. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p7_leave_dataset_and_stratum_out.csv` | `P6_system_controller_step_ratio_failed` |
| `p8_official_paired_replay.csv` | same |
| `p9_short_run_functional_validation.csv` | same |
| `p10_full_run_robustness_strong_baseline.csv` | same |

没有把 selected-feature diagnostic、async audit pass、decision metrics 或 payload binding pass 写成 LDO/LSO、paired replay、short-run 或 full-run success。

## 11. No-fake audit

```text
rows_checked = 33
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
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
projection_used_for_official = 0
```

## 12. Hash

| artifact | SHA256 |
|---|---|
| plan | `a55c55b892a3c2dcf156482b1624c61a8eeacb125abdd431b89e448232bc8151` |
| runner | `32c2493a34d43868cdbabc17f0eef1c8cd97b2f451598c2d559eb72de131050c` |
| run manifest | `16c4cb770a4bf88ec7ab4ad94b8618cf144f17828e6a9952fa58c35e9756f7df` |
| route | `7fbe4047cb78dbaf07614a8936558d2b66cafe1244d0837e740b77a38505f062` |
| P0 boundary | `d9e1958c3ead40e33809a215ca2c38ba7d96c35a5ae0c1cc7fc8b1cc42afddaa` |
| P1 cost attribution | `e88f402e42fe5f9f0e6766b7482866fe57a8f678c6d2f8b6cf7ce36ab6bc072b` |
| P2 fusion diagnostic | `916c1775d06ea9610aeedf0069f89f83706e30a34e93d314f8f20a6acc6b2116` |
| P3 selected feature | `e8bc1555555e87010f3c30a091f386ba5bc3737e922b31f475f14513e3c85c3f` |
| P4 bucket workspace | `ec34dea8fa544e16dffff98cd531fffdea57570ba30be48f3ab720b74517a4eb` |
| P5 async audit | `715ba1e6645cdae94cd1ed1942555a72a8cfee303fa4fdd4265f15e43bc338cd` |
| P6 system controller | `c9b8d7c45b1a1acdfaf1149370d80d4136377556a65b7fe358f6ffaf3733dd0b` |
| P7 boundary | `2d55fd649e746faafb16be78f4cc8d42c35cbbb3da994d1b9f0b3745c1882416` |
| P8 boundary | `32ab4567be536887276297775ab6f096a7f8218cbceb918d8e70aad5116b99ad` |
| P9 boundary | `78c8ab3578a760f74b574739d55dc56b6f5efb9a4b61a96abcc2cef32c368215` |
| P10 boundary | `70ee8dfa110df30ee5c33e01b1760d80201620889a9afb73e387bd66f9a1b1a6` |
| failure table | `b2e66f8e2402f03bf273469ea276158265d4076eb330492b0306d19e12f2af47` |
| provenance audit | `cf3334b450dd4a06cb61faf73adc6975c49bda4ad113e38a6efd7b82cb37c032` |

## 13. 最终分析结论

v9.2.73 的真实推进是：

```text
v9.2.72: BF5 async combined basis+delta path step q90 = 2.713296，
          payload/decision 都过，但 system cost 失败。
v9.2.73: 将 BF5 剩余 cost 拆到 aggregate component；
          证明 selected-feature materialization 是可疑成本项；
          但即使 audit-only 消除 selected-feature component，step q90 仍是 2.346412。
```

机制判断：

1. H1 只在 aggregate component 层成立：unknown fraction = `0.0`，dominant 是 `fused_common_basis_delta_pipeline_ms`；但 kernel-internal basis lift/quadratic/norm 等细分仍未落地。
2. H2 未闭合：best diagnostic q90 从 BF5 的 `2.713296` 降到 `2.346412`，仍高于 official `1.50`。
3. H3 只作为 diagnostic 成立：selected-feature materialization 可以从 accounting 中移除，但这是基于已测组件的 audit-only row，不能写成 official measured runtime success。
4. H4 未成立：static bucket / persistent workspace / CUDA graph 没实现，kernel/sync/allocation 没下降。
5. H5 成立：timed path 没有 hash/csv/cpu metadata 污染，no fake/proxy/offload 均为 `0`。
6. H6 未打开：P6 `official_eligible = 0`，因此 LDO/LSO、paired replay、short/full validation 全部 gate-blocked。

最终一句话：

> v9.2.73 真实执行后停在 `R18-SystemStillTooExpensive`：selected-feature audit 显示仍有可优化空间，但 best diagnostic step q90 仍为 `2.346412 > 1.50`，static bucket/workspace/CUDA graph 未闭合，system-legal controller 仍不能转正。
