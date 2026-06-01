# DG-KAN v9.2.79 Stable-Accept Official Promotion 与 Full-System Batch-Major Runtime Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.2.79_StableAcceptOfficialPromotion_FullSystemBatchMajorClosure_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 v9.2.78 的 local native kernel pass 或旧 controller metrics 写成 official system pass。

## 0. 最新结论

```text
route = R15-StableAcceptOfficialCalibrationBlocked
base_candidate = LQ-t2-h256
success_v9279_strict_purekan_functional = False
success_v9279_full_functional = False
success_v9279_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9279_stable_accept_official_promotion_full_system_batch_major_closure_continued_probe_20260513T173000Z
```

核心结论：

1. P0 复现 v9.2.78 boundary：source route = `R18-StableAcceptLocalClosedButSystemNotOfficial`，stable accept local pass = `1`，native bucket kernel v2 pass = `1`，source system pass = `0`。
2. P1 按 plan 尝试 official calibration rerun，但 full-online row 没有 AC2Q2 official 所需的 stable score/rank/accept 字段；calibration rerun 不能真实执行，pass = `0`。
3. P2 heldout/support rerun 同样 blocked：缺少 full-row stable accept materialization 和 outcome labels，不能复用旧 C3 metrics。
4. P3 证实 v9.2.78 native stable kernel 是 local pass，但没有出现在 P6 full-system trace 中：`native_bucket_kernel_used_in_p6 = 0`。
5. P4 没有新的 full-system native step timing；source boundary step ratio `2.713295831053225` 没有被当作 v9.2.79 measured native ratio 复用。
6. P5 batch-major native runtime 仍只是 v9.2.78 local evidence：local native q90 = `0.6102416664361954`，q90 reduction = `0.4443256711105608`，不能替代 end-to-end step timing。
7. P6 official controller 未打开：`official_eligible = 0`，reason = `P1_P2_full_row_stable_accept_materialization_missing`。
8. 继续追溯 outcome labels 后仍不能补：`source_hash` join coverage = `0`，`event_family+dataset+seed` 仍有 `1648` 个 candidate label ambiguous。
9. v9.2.78 stable accept trace 只覆盖局部子集：local stable rows = `72`，matched full candidates = `3` / `2493`。
10. 当前 blocker：`stable_accept_full_row_materialization_missing`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9279_stable_accept_official_promotion_full_system_batch_major_closure.py` | v9.2.79 runner；复现 v9.2.78 boundary，审计 AC2Q2 stable accept 是否具备 full-row official calibration/heldout/support rerun 与 full-system native runtime materialization |

代码检查：

```text
python -m py_compile experiments/run_v9279_stable_accept_official_promotion_full_system_batch_major_closure.py
```

正式运行：

```bash
python experiments/run_v9279_stable_accept_official_promotion_full_system_batch_major_closure.py \
  --out-dir results/real_rerun_20260506/v9279_stable_accept_official_promotion_full_system_batch_major_closure_continued_probe_20260513T173000Z \
  --fresh --device auto --data-root data --seed 1314
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R15-StableAcceptOfficialCalibrationBlocked",
  "base_candidate": "LQ-t2-h256",
  "v9278_boundary_pass": 1,
  "source_route_v9278": "R18-StableAcceptLocalClosedButSystemNotOfficial",
  "stable_accept_contract_id": "AC2Q2-stable-quantized-1e5-event-tie",
  "stable_accept_contract_pass_v9278": 1,
  "native_bucket_kernel_v2_pass_v9278": 1,
  "stable_accept_official_calibration_pass": 0,
  "stable_accept_heldout_support_pass": 0,
  "full_system_integration_pass": 0,
  "full_system_step_attribution_pass": 0,
  "batch_major_runtime_pass": 0,
  "system_legal_controller_pass": 0,
  "official_eligible": 0,
  "candidate_count": 2493,
  "accepted_count_old_reference": 951,
  "missing_stable_accept_fields": "stable_accept_contract_id;score_ref;score_native;score_quantized_ref;score_quantized_native;stable_rank_ref;stable_rank_native;accept_ref;accept_native",
  "missing_outcome_fields": "safe_good_label;bad_event_label;null_event_label",
  "native_kernel_used_in_p6": 0,
  "new_full_system_step_ratio_measured": 0,
  "old_step_ratio_reused_as_measurement": 0,
  "source_boundary_step_ratio_q90": 2.713295831053225,
  "controller_step_ratio_q90": "",
  "native_time_ms_q90_local_v9278": 0.6102416664361954,
  "q90_reduction_local_v9278": 0.4443256711105608,
  "kernel_count_after_local_v9278": 216,
  "sync_count_after_local_v9278": 24,
  "primary_blocker": "stable_accept_full_row_materialization_missing",
  "next_required_implementation": "materialize_full_row_stable_score_rank_accept_and_outcome_labels",
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0,
  "outcome_label_recovery_pass": 0,
  "outcome_label_best_join_mode": "event_family_dataset_seed",
  "outcome_source_hash_covered_candidate_rows": 0,
  "outcome_event_family_dataset_seed_ambiguous_rows": 1648,
  "stable_accept_trace_coverage_pass": 0,
  "stable_accept_local_trace_rows": 72,
  "stable_accept_trace_matched_full_candidate_rows": 3,
  "stable_accept_trace_full_row_coverage": 0.0012033694344163659
}
```

判断：v9.2.79 没有把 v9.2.78 的局部 stable-accept closure 提升成 official。真正缺的是 full-row stable score/rank/accept materialization、outcome labels 和 P6 native full-system step trace。

## 3. P1 stable accept official calibration rerun

Artifacts：

```text
p1_stable_accept_official_calibration_rerun.csv
stable_accept_full_row_field_trace_v9279.csv
```

Summary：

```text
calibration_rerun_attempted = 1
calibration_rerun = 0
stable_accept_full_row_materialization_present = 0
outcome_labels_present = 0
event_count = 24192
candidate_count = 2493
accepted_count_old_reference = 951
missing_stable_accept_fields = stable_accept_contract_id;score_ref;score_native;score_quantized_ref;score_quantized_native;stable_rank_ref;stable_rank_native;accept_ref;accept_native
missing_outcome_fields = safe_good_label;bad_event_label;null_event_label
stable_accept_official_calibration_pass = 0
```

判断：这是本轮 terminal blocker。现有 full-online 表有 old `accept_decision/bridge_score/candidate_rank`，但没有 AC2Q2 full-row `score_ref/score_native/score_quantized/rank/accept`，也没有 rerun 所需 outcome label。不能编造 rerun 指标。

## 4. P2 heldout/support rerun

Artifact：

```text
p2_stable_accept_heldout_support_rerun.csv
```

Summary：

```text
heldout_rerun_attempted = 1
heldout_rerun = 0
support_balance_rerun = 0
stable_accept_heldout_support_pass = 0
reason = stable_accept_full_row_or_outcome_materialization_missing
```

判断：stable accept 是 rule change，不能沿用旧 heldout/support summary；由于 P1 full-row materialization 缺失，P2 必须 blocked。

## 5. P3-P5 full-system runtime audit

Artifacts：

```text
p3_native_stable_kernel_full_system_integration_audit.csv
p4_full_system_step_attribution_native_stable_path.csv
p5_batch_major_native_runtime_closure.csv
```

Summary：

```text
native_kernel_local_pass_v9278 = 1
native_bucket_kernel_used_in_p6 = 0
native_runtime_full_system_materialization_present = 0
new_full_system_step_ratio_measured = 0
old_step_ratio_reused_as_measurement = 0
batch_major_native_full_system_runtime_measured = 0
batch_major_runtime_pass = 0
```

判断：v9.2.78 local native kernel 确实是好信号，但 v9.2.79 official promotion 需要 end-to-end P6 trace。现有 artifact 不能证明 native stable kernel 已进入 full train-stream controller runtime。

## 6. P6 system controller boundary

Artifact：

```text
p6_system_legal_exact_signal_controller_v11.csv
```

Boundary：

```text
controller_id = C3Q2-StableAcceptNativeBucket
official_eligible = 0
system_legal_controller_pass = 0
reason = P1_P2_full_row_stable_accept_materialization_missing
source_boundary_step_ratio_q90 = 2.713295831053225
step_ratio_q90 = 
old_metrics_reused_for_official = 0
native_bucket_kernel_used_in_p6 = 0
```

判断：P6 没有 official。这里没有用旧 decision metrics 或旧 step ratio 盖章，也没有把 local P3 q90 写成 system step ratio。

## 7. 继续追溯：outcome label 与 stable accept trace

Artifacts：

```text
p12_outcome_label_recovery_probe.csv
outcome_label_recovery_trace_v9279.csv
p13_stable_accept_trace_coverage_probe.csv
```

Outcome join summary：

```text
label_source = results/real_rerun_20260506/v9267_borderline_localized_bridge_repair_C0C4_pareto_calibration_first_20260512T223000Z/C0_T2_C4_E2_membership_trace_v9267.csv
candidate_count = 2493
best_join_mode = event_family_dataset_seed
source_hash_covered_candidate_rows = 0
event_family_unique_label_candidate_rows = 166
event_family_ambiguous_label_candidate_rows = 2327
event_family_dataset_seed_unique_label_candidate_rows = 845
event_family_dataset_seed_ambiguous_label_candidate_rows = 1648
outcome_label_recovery_pass = 0
```

Stable trace coverage：

```text
local_stable_accept_trace_rows = 72
candidate_count_full_online = 2493
matched_full_candidate_rows_by_dataset_seed_step_rank = 3
full_row_coverage = 0.0012033694344163659
stable_accept_trace_coverage_pass = 0
```

判断：不能用 `event_family` 级别统计补 outcome label，因为同一 family 内 label 冲突很多；也不能用 v9.2.78 local trace 补 full rows，因为覆盖率太低。这进一步确认下一步必须重跑 full train-stream materializer，而不是 join 旧 artifact。

## 8. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p7_leave_dataset_and_stratum_out.csv` | `P6_system_controller_not_official` |
| `p8_official_paired_replay.csv` | same |
| `p9_short_run_functional_validation.csv` | same |
| `p10_full_run_robustness_strong_baseline.csv` | same |

没有把 stable accept local pass、native local q90 reduction 或旧 reference frontier 写成 LDO/paired replay/short-run/full-run success。

## 9. No-fake audit

```text
rows_checked = 24223
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
stable_accept_local_contract_pass = 1
native_bucket_kernel_v2_local_pass = 1
stable_accept_official_calibration_pass = 0
stable_accept_heldout_support_pass = 0
full_system_integration_pass = 0
batch_major_runtime_pass = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
projection_used_for_official = 0
```

## 10. Hash

| artifact | SHA256 |
|---|---|
| plan | `9e9fc6d8e2eac53b1dac72524630d4f88acfff72bb5979ba593368a90ea96448` |
| runner | `973550500654ad1ea947d25323685173d10b654d2937bbe67e8b3038e4ac6b96` |
| run manifest | `e5541723d368b831c48e63f4b4a54d1e42b046a8a1c8b844333322d01e6830f7` |
| route | `79d6ae003c5fcf79b7080b7a8e9f2ff886867c9c88ffc40a22ebce3142df8ba1` |
| P0 boundary | `30fd36d9147f1aa02f738402cc3ab563ce37f5d37f7689416159cb3fc7c359bd` |
| P1 calibration | `196ea227a0b90129a6ac4ddfc86ef43359a087ccf0ba337ce5c38be8518dcda2` |
| P2 heldout support | `95e23feae4c34f36faa66076a0322517985d8e6241a5f50ce549d5aff2351a75` |
| P3 integration | `eee5d0e26e0c29ef1120e48dfade13f5228de42cd25cfc67ae748972fc726a27` |
| P4 attribution | `44f03b01eeef48c47560efdac71c586eacbc608cea99f3ca50b7fa1c232b0442` |
| P5 batch major | `05b0c2115a1c1ed30aab6ec728b7382d52df3e7efcaf1241c0507453d71f792f` |
| P6 system controller | `4cfe7014e65b154461f85f98bb462d1d25b9bcaf159aa7e29180f002ebd92717` |
| P12 outcome recovery | `c6003653ac57894416519be4bce355de17033565646f37ad56b8c7e5e8f5a1fa` |
| P13 stable trace coverage | `80a7510260cf55733179b8e47c42eead3a5e671ac9b67661319e382ceb6723ef` |
| provenance audit | `1a147be094b8a047c80c3fdf3087cf104d3335e2108f2f25e63dc8d4cd2791ca` |

## 11. 最终分析结论

v9.2.79 的真实推进是：

```text
v9.2.78: stable accept local correctness 与 native bucket kernel local runtime 已闭合，
          但 official promotion 仍 blocked。
v9.2.79: 对 official promotion 所需 full-row materialization 做了严格审计；
          发现现有 full-online artifact 缺 AC2Q2 stable score/rank/accept 与 outcome labels；
          继续尝试 outcome join 和 local stable trace coverage，仍不能可靠补齐全量 rows。
```

机制判断：

1. H1 未能执行：不是 stable accept 指标失败，而是 full-row materialization 缺失，calibration/heldout 无法真实 rerun。
2. H2 未能执行：P6 没有 native stable kernel full-system trace，不能判断 `2.713296` 是旧路径残留还是 native full-system 下界。
3. H3 未能 official：v9.2.78 local q90 reduction 仍是局部证据，不能替代 batch-major full-system timing。
4. H4 成立：rule change 后没有绕过 rerun gate；P7-P10 继续 gate-blocked。
5. P12/P13 进一步确认：旧 outcome label artifact 不能按 `source_hash` join，按 family join 又 label ambiguous；v9.2.78 local stable trace 覆盖率也不足。
6. 当前下一步必须重跑 full train-stream materializer：为全量 candidate rows 记录 `score_ref/score_native/score_quantized/rank/accept` 和真实 outcome labels，再重跑 calibration/heldout/support 与 P6 native runtime。

最终一句话：

> v9.2.79 继续执行后仍停在 `R15-StableAcceptOfficialCalibrationBlocked`：v9.2.78 的 stable accept/native kernel 局部闭合没有回退，但旧 artifact 无法可靠 join 出全量 outcome labels 或 stable accept rows；official promotion 需要重跑 full train-stream materialization，strict PureKAN functional 仍未成功。
