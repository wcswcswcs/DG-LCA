# DG-KAN v9.2.76 BasisNorm Runtime Elimination 与 Static-Bucket Workspace Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.2.76_BasisNormRuntimeElimination_StaticBucketWorkspaceClosure_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 basis-norm 局部优化、persistent workspace materialization 或 gate-blocked downstream 写成 official system pass。

## 0. 最新结论

```text
route = R15-BasisNormDominantUnfixed
base_candidate = LQ-t2-h256
success_v9276_strict_purekan_functional = False
success_v9276_full_functional = False
success_v9276_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9276_basis_norm_runtime_elimination_static_bucket_workspace_closure_first_20260513T150000Z/
```

说明：本轮没有保存单独 stdout `run.log`。可审计日志是上面 artifact 目录中的 `run_manifest.json`、`route_decision.json`、各阶段 `p*.csv`、`*_trace_v9276.csv`、`failure_table.csv`、`artifact_hashes.csv` 和 `v9276_provenance_audit.csv`。

核心结论：

1. P0 复现 v9.2.75 boundary：source route = `R16-StaticBucketWorkspaceStillUnimplemented`，payload binding pass = `1`，candidate/update payload missing count 全部为 `0`，system legal controller pass = `0`。
2. 本轮 manifest 记录 `device = cuda`，`triton_available = true`；实际参数保持 MNIST/Fashion-MNIST/KMNIST、seeds `0..7`、targeted attribution steps per dataset = `8`，basis-norm warmup steps = `1`。
3. P1 basis_norm internal attribution 过 gate：unknown fraction = `0.0`，dominant basis-norm subcomponent = `quantile_tail_compute_time_ms`，均值 `0.371753 ms`，component ratio = `0.466308`。
4. P1 其他主要 basis-norm subcomponents：`norm_scale_shift_time_ms = 0.153466`，`candidate_norm_writeback_time_ms = 0.116564`，`norm_dispatch_overhead_time_ms = 0.050312`。
5. P2 no-clone/logsumexp/top2 basis_norm runtime 数值正确：audit agreement = `1.0`，logits max error = `4.768372e-07`，delta max error = `3.492460e-10`。
6. P2 但 runtime reduction 未过 gate：same-run `basis_norm_time_before = 0.797227 ms`，`basis_norm_time_after = 0.581002 ms`，reduction = `0.271221 < 0.50`，因此 `basis_norm_runtime_pass = 0`。
7. 注意：v9.2.75 source boundary 的 `basis_norm_time_ms = 5.664183` 仍是落盘事实；v9.2.76 的 `0.797227 -> 0.581002 ms` 是本轮 warmup-excluded targeted timing 内部对照，不是 full-system step ratio closure。
8. P3 static bucket / persistent workspace 只完成 materialization：`runtime_bucket_used = 1`，`persistent_workspace_used = 1`，workspace memory = `0.298832 MB`，allocation count 从 `2493` 到 `1`；但 `basis_norm_bucketed = 0`、`W2_delta_bucketed = 0`、kernel/sync reduction = `0.0`，`avg_candidates_per_kernel_after = 0.6648`，所以 `static_bucket_workspace_pass = 0`。
9. P4 single-pass basis-norm/delta/bridge 未闭合：`basis_norm_delta_bridge_single_pass = 0`，`bridge_score_inside_kernel = 0`，`accept_bit_inside_kernel = 1`。
10. P5/P6 不能 official：`materialized_runtime_path = 0`，`diagnostic_derived_from_measured_components = 1`，system controller `official_eligible = 0`，step ratio q90 仍保持 source boundary 的 `2.713296 > 1.50`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9276_basis_norm_runtime_elimination_static_bucket_workspace_closure.py` | v9.2.76 runner；复现 v9.2.75 boundary，执行 basis_norm 内部分解、no-clone/logsumexp/top2 runtime 对照、static bucket/persistent workspace materialization、single-pass gate、integrated runtime 与 system controller gate |

代码检查：

```text
python -m py_compile experiments/run_v9276_basis_norm_runtime_elimination_static_bucket_workspace_closure.py
```

正式运行：

```bash
python experiments/run_v9276_basis_norm_runtime_elimination_static_bucket_workspace_closure.py \
  --out-dir results/real_rerun_20260506/v9276_basis_norm_runtime_elimination_static_bucket_workspace_closure_first_20260513T150000Z \
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
basis_norm_warmup_steps = 1
train_size = 2048
batch_size = 64
hidden_dim = 256
completed_at = 2026-05-13T13:29:16Z
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R15-BasisNormDominantUnfixed",
  "base_candidate": "LQ-t2-h256",
  "v9275_boundary_pass": 1,
  "payload_binding_contract_pass": 1,
  "candidate_tensor_payload_missing_count": 0,
  "candidate_branch_logits_missing_count": 0,
  "candidate_true_delta_logits_missing_count": 0,
  "functional_update_payload_missing_count": 0,
  "basis_norm_internal_attribution_pass": 1,
  "basis_norm_unknown_fraction": 0.0,
  "dominant_basis_norm_subcomponent": "quantile_tail_compute_time_ms",
  "quantile_tail_compute_time_ms": 0.371753276946644,
  "norm_scale_shift_time_ms": 0.1534663800460597,
  "candidate_norm_writeback_time_ms": 0.11656367375204961,
  "best_basis_norm_runtime_id": "BO3-NoCloneLogsumexpTop2BasisNormRuntime",
  "basis_norm_runtime_pass": 0,
  "basis_norm_time_before": 0.7972274324856699,
  "basis_norm_time_after": 0.5810023285448551,
  "basis_norm_time_reduction": 0.27122135432124805,
  "runtime_bucket_used": 1,
  "persistent_workspace_used": 1,
  "basis_norm_bucketed": 0,
  "W2_delta_bucketed": 0,
  "static_bucket_workspace_pass": 0,
  "basis_norm_delta_bridge_single_pass": 0,
  "bridge_score_inside_kernel": 0,
  "integrated_runtime_pass": 0,
  "system_legal_controller_pass": 0,
  "official_eligible": 0,
  "controller_step_ratio_q90": 2.713295831053225,
  "primary_blocker": "basis_norm_runtime_not_reduced",
  "next_required_implementation": "fuse_or_cache_basis_norm_runtime"
}
```

判断：v9.2.76 真实定位了 basis_norm 内部 dominant，但 no-clone/logsumexp/top2 只带来 `27.12%` 局部降低，未达到计划 `>=50%` runtime reduction；static bucket/workspace 也没有接入 basis_norm/W2_delta runtime，因此 route 停在 `R15`。

## 3. P1 basis_norm internal attribution

Artifacts：

```text
p1_basis_norm_internal_attribution.csv
basis_norm_internal_trace_v9276.csv
```

Summary：

```text
basis_norm_attribution_id = BN1-BasisNormSubphaseTimer
sample_step_count = 24
basis_norm_internal_attribution_pass = 1
basis_norm_total_time_ms = 0.797227
basis_norm_unknown_fraction = 0.0
dominant_basis_norm_subcomponent = quantile_tail_compute_time_ms
dominant_basis_norm_component_ratio = 0.466308
```

Subphase：

| subcomponent | mean ms |
|---|---:|
| `quantile_tail_compute_time_ms` | `0.371753` |
| `norm_scale_shift_time_ms` | `0.153466` |
| `candidate_norm_writeback_time_ms` | `0.116564` |
| `norm_dispatch_overhead_time_ms` | `0.050312` |
| `audit_norm_writeback_time_ms` | `0.034132` |
| `norm_kernel_launch_time_ms` | `0.020302` |
| `family_context_lookup_time_ms` | `0.015874` |
| `norm_temp_allocation_time_ms` | `0.014208` |
| `bucket_context_lookup_time_ms` | `0.011043` |
| `norm_input_gather_time_ms` | `0.005260` |
| `norm_sync_time_ms` | `0.004311` |

判断：H1 成立。basis_norm 内部不是黑箱，主要成本是 quantile/tail/context construction，而不是 gather、sync 或 allocation。

## 4. P2 basis_norm cache / fusion runtime

Artifacts：

```text
p2_basis_norm_cache_fusion_runtime.csv
basis_norm_runtime_trace_v9276.csv
```

Best summary：

```text
basis_norm_runtime_id = BO3-NoCloneLogsumexpTop2BasisNormRuntime
basis_norm_strategy = no_clone_logsumexp_top2_quantile_exact_tail
cache_used = 0
fused_basis_norm_kernel_used = 0
basis_norm_time_before = 0.7972274324856699
basis_norm_time_after = 0.5810023285448551
basis_norm_time_reduction = 0.27122135432124805
agreement_reference_accept = 1.0
audit_agreement = 1.0
cuda_vs_torch_check_count = 24
cuda_vs_torch_logits_error_max = 4.76837158203125e-07
cuda_vs_torch_delta_error_max = 3.4924596548080444e-10
basis_norm_runtime_pass = 0
```

判断：no-clone/logsumexp/top2 path 数值上正确，没有 source gap / formula proxy；但 `27.12%` 的降低不足以过 P2 gate。当前 blocker 不是数值一致性，而是 basis_norm runtime 没被真正压到一半以下。

## 5. P3 static bucket / persistent workspace

Artifacts：

```text
p3_static_bucket_persistent_workspace_integrated_runtime.csv
static_bucket_workspace_runtime_trace_v9276.csv
```

Summary：

```text
bucket_workspace_id = BW2-PersistentBasisNormWorkspaceMaterializedNotIntegrated
runtime_bucket_used = 1
persistent_workspace_used = 1
workspace_memory_MB = 0.2988319396972656
allocation_count_before = 2493
allocation_count_after = 1
allocation_count_reduction = 0.9995988768551946
kernel_count_before/after = 3750 / 3750
sync_count_before/after = 1250 / 1250
kernel_count_reduction = 0.0
sync_count_reduction = 0.0
avg_candidates_per_kernel_after = 0.6648
basis_norm_bucketed = 0
W2_delta_bucketed = 0
static_bucket_workspace_pass = 0
```

Bucket sizes：

```text
{"1": 7260, "2": 365, "4": 439}
```

判断：persistent workspace 是真实 CUDA tensor materialization，但没有把 bucketed runtime 接入 basis_norm / W2_delta / bridge score。不能把 allocation reduction 单独写成 static bucket pass。

## 6. P4/P5/P6 runtime 与 system boundary

Artifacts：

```text
p4_single_pass_basis_norm_delta_bridge_runtime.csv
p5_integrated_materialized_runtime_candidates.csv
p6_system_legal_exact_signal_controller_v8.csv
```

P4 summary：

```text
basis_delta_bridge_runtime_id = BD1-NoCloneBasisNormPlusReferenceDeltaBridge
uses_true_branch_delta = 1
uses_source_measured_gap = 0
uses_formula_proxy = 0
basis_norm_delta_bridge_single_pass = 0
bridge_score_inside_kernel = 0
accept_bit_inside_kernel = 1
agreement_reference_accept = 1.0
step_ratio_q90 = 2.713295831053225
```

P5/P6 boundary：

```text
system_candidate_id = SYS7-BasisNormRuntimeNoCloneNoStaticBucket
materialized_runtime_path = 0
diagnostic_derived_from_measured_components = 1
integrated_runtime_pass = 0
official_eligible = 0
system_legal_controller_pass = 0
reason = P5_integrated_runtime_or_step_ratio_failed
```

Decision metrics still held：

```text
precision_heldout = 0.8380281690140845
coverage_heldout = 0.03130511463844797
bad_event_heldout = 0.02464788732394366
null_rate_heldout = 0.13028169014084506
precision_lcb = 0.7907157243773478
bad_event_ucb = 0.04999458813129621
accepted_signal_strata_count = 15
accepted_family_count = 65
```

判断：decision/support/payload 都没有 regression，但 P4/P5/P6 不允许打开。失败点仍是真实 runtime path：basis_norm runtime pass 不过，static bucket 未接入，bridge score 不在 kernel 内，integrated runtime 仍非 official。

## 7. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p7_leave_dataset_and_stratum_out.csv` | `P2_basis_norm_runtime_failed` |
| `p8_official_paired_replay.csv` | same |
| `p9_short_run_functional_validation.csv` | same |
| `p10_full_run_robustness_strong_baseline.csv` | same |

没有把 basis_norm attribution pass、no-clone 数值一致性、persistent workspace materialization 或 decision metrics 写成 LDO/LSO、paired replay、short-run 或 full-run success。

## 8. No-fake audit

```text
rows_checked = 117
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
basis_norm_internal_attribution_pass = 1
basis_norm_runtime_pass = 0
static_bucket_workspace_used = 1
basis_norm_delta_bridge_single_pass = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
projection_used_for_official = 0
```

## 9. Hash

| artifact | SHA256 |
|---|---|
| plan | `1fd19db67807bfdea7c0fac25420250c580050ccac1d7735496f9fd2e1b7fe21` |
| runner | `f919b8057b382dd195e355a1a55972cc39e830fb058e3a399cd1fcfa03de4d80` |
| run manifest | `6c35dd4ed9de3ca0de523eab5da0559db5d755991ed4de5e6d761874cd28e0aa` |
| route | `a46bb385c5cbaaf60e9e4467c951ed83d03970024f1a4ab2292ef3cf21157121` |
| P0 boundary | `edf76d3fc343ea90a9378192157563dea6b45417486c3e97a79715312f1e063d` |
| P1 basis norm | `29056c34285919e2e3f58520ded7d9d68e625dce4f92cd810318228691effeca` |
| P2 basis norm runtime | `58da913973b936c702512b445fb5fbfb6a0ec524ee1ef778c122db9e928bf341` |
| P3 static bucket | `9efa282a612f6c806c6d11045910786ec7d409007f5e8cdd7870ec96e4440d46` |
| P4 single pass | `cec85dae06f60c8b9c2fcd3ff03d3ad293a59408d376dc07801aff1e4aeb8843` |
| P5 integrated runtime | `8bebbbff14134bf7e8501781bb3f862261e3df9451576a3860783c72449ebefe` |
| P6 system controller | `963b6538d72ad4b85add3e449f34a7b7d9f297d76d43aa17c274079358b38b0d` |
| failure table | `b32e9f7d8ddd95f3c750551008abacd6999ff1b21a0694973da2796a95186c30` |
| provenance audit | `2899eddedaa04eb1e0d873155edee8997e3352c33795a98cae732b3017fb9ccb` |

## 10. 最终分析结论

v9.2.76 的真实推进是：

```text
v9.2.75: kernel-internal attribution 首次闭合，
          dominant 是 basis_norm，但 static bucket/workspace 未接入。
v9.2.76: basis_norm 内部 dominant 进一步定位为 quantile/tail compute；
          no-clone/logsumexp/top2 path 数值正确但 runtime 降幅不足；
          persistent workspace materialized，但未接入 bucketed basis_norm/W2_delta runtime。
```

机制判断：

1. H1 成立：basis_norm 内部 attribution 闭合，dominant 是 `quantile_tail_compute_time_ms`，unknown fraction = `0.0`。
2. H2 未成立：本轮没有形成 bucket/family cache，cache hit rate = `0.0`；no-clone runtime 只降低 `27.12%`。
3. H3 未成立：没有把 audit norm 正式移出 timed path。
4. H4 未成立：persistent workspace 虽然 materialized，但 basis_norm/W2_delta 没有 bucketed，kernel/sync 没降。
5. H5 未成立：bridge score 仍不在 kernel 内，single-pass basis-norm/delta/bridge 未闭合。
6. H6 尚不能声明 lower bound：P2/P3/P4 没 materialize 到 passing runtime，不能说 system cost 已经不可降。

最终一句话：

> v9.2.76 真实执行后停在 `R15-BasisNormDominantUnfixed`：basis_norm 主成本已进一步定位到 quantile/tail compute，no-clone runtime 数值正确但降幅不足，static bucket/persistent workspace 仍未真正接入 basis_norm/W2_delta runtime，system-legal controller 仍不能转正。
