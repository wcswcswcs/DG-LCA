# DG-KAN v9.2.80 Full-Train-Stream StableAccept Materializer 与 Outcome-Label Rebuild 实验复盘

> 本复盘记录 `DG-KAN_v9.2.80_FullTrainStreamStableAcceptMaterializer_OutcomeLabelRebuild_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把旧 outcome join、局部 native q90 或旧 step ratio 写成 official。

## 0. 最新结论

```text
route = R15-StableAcceptNativeQuantizationMismatch
base_candidate = LQ-t2-h256
success_v9280_strict_purekan_functional = False
success_v9280_full_functional = False
success_v9280_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9280_full_train_stream_stable_accept_materializer_outcome_label_rebuild_rerun_20260513T190000Z
```

核心结论：

1. P0 复现 v9.2.79 blocker：source route = `R15-StableAcceptOfficialCalibrationBlocked`，primary blocker = `stable_accept_full_row_materialization_missing`。
2. P1 重跑 full train-stream materializer：event count = `24192`，candidate count = `2876`，stable candidate rows = `2876`。
3. AC2Q2 full-row stable fields 已落盘：materialization present = `1`，accept disagreement = `0`，quantized disagreement = `18`，rank disagreement = `2`。
4. same-run primary outcome labels 已落盘：outcome_labels_present = `1`，missing primary label count = `0`，label_join_mode = `direct_same_run_event_id`。
5. 但 CEp99/margin/ECE/NLL/curvature/real-beats 等 secondary outcome delta fields 缺失计数为 `14380`；这些没有被编造。
6. P2 official calibration/heldout rerun 已真实执行：precision = `0.5128205128205128`，coverage = `0.051587301587301584`，bad-event = `0.3034188034188034`，null-rate = `0.11752136752136752`，precision LCB = `0.47816314913056096`，bad UCB = `0.33529565729782573`。
7. P4 native stable bucket kernel used in P6 = `1`；new full-system step ratio measured = `1`；old step ratio reused = `0`。
8. Runtime result：native q90 = `0.6184950470924377`，eager q90 = `1.0737529955804348`，q90 reduction = `0.4239875933867825`，step ratio q90 = `2.213009156635521`。
9. P6 official result：official eligible = `0`，system pass = `0`，reason = `stable_accept_heldout_support_failed`。
10. 当前 blocker：`stable_accept_native_quantized_rank_mismatch`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9280_full_train_stream_stable_accept_materializer_outcome_label_rebuild.py` | v9.2.80 runner；重跑 full train-stream materializer，落 AC2Q2 stable accept fields、same-run outcome labels、native stable bucket full-system runtime trace，并重新计算 P6 gate |

代码检查：

```text
python -m py_compile experiments/run_v9280_full_train_stream_stable_accept_materializer_outcome_label_rebuild.py
```

正式运行：

```bash
python experiments/run_v9280_full_train_stream_stable_accept_materializer_outcome_label_rebuild.py \
  --out-dir results/real_rerun_20260506/v9280_full_train_stream_stable_accept_materializer_outcome_label_rebuild_rerun_20260513T190000Z \
  --fresh --device auto --data-root data --seed 1314
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R15-StableAcceptNativeQuantizationMismatch",
  "base_candidate": "LQ-t2-h256",
  "v9279_boundary_pass": 1,
  "stable_accept_contract_id": "AC2Q2-stable-quantized-1e5-event-tie",
  "full_train_stream_materializer_used": 1,
  "stable_accept_full_row_materialization_present": 1,
  "stable_accept_full_row_materialization_pass": 0,
  "outcome_labels_present": 1,
  "missing_label_count": 0,
  "secondary_outcome_delta_field_missing_count": 14380,
  "accept_disagreement_count": 0,
  "score_quantized_disagreement_count": 18,
  "rank_disagreement_count": 2,
  "stable_accept_official_calibration_pass": 0,
  "stable_accept_heldout_support_pass": 0,
  "controller_precision": 0.5128205128205128,
  "controller_coverage": 0.051587301587301584,
  "controller_bad_event": 0.3034188034188034,
  "controller_null_rate": 0.11752136752136752,
  "controller_precision_lcb": 0.47816314913056096,
  "controller_bad_event_ucb": 0.33529565729782573,
  "accepted_signal_strata_count": 21,
  "accepted_family_count": 66,
  "native_bucket_kernel_used_in_p6": 1,
  "stable_accept_cuda_kernel_used": 1,
  "basis_norm_bucketed": 1,
  "W2_delta_bucketed": 1,
  "bridge_score_inside_kernel": 1,
  "accept_bit_inside_kernel": 1,
  "kernel_count_before": 3750,
  "kernel_count_after": 72576,
  "sync_count_before": 1250,
  "sync_count_after": 8064,
  "avg_candidates_per_kernel_after": 0.03962742504409171,
  "native_time_ms_q90": 0.6184950470924377,
  "eager_time_ms_q90": 1.0737529955804348,
  "q90_reduction": 0.4239875933867825,
  "new_full_system_step_ratio_measured": 1,
  "old_step_ratio_reused_as_measurement": 0,
  "controller_step_ratio_q90": 2.213009156635521,
  "batch_major_runtime_pass": 0,
  "official_eligible": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "stable_accept_native_quantized_rank_mismatch",
  "next_required_implementation": "make_native_score_quantization_rank_bit_exact_or_redefine_audited_contract",
  "success_v9280_strict_purekan_functional": 0,
  "success_v9280_full_functional": 0,
  "success_v9280_external_ready": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. Materializer

Artifacts：

```text
p1_full_train_stream_stable_accept_outcome_materializer.csv
full_row_stable_accept_outcome_table_v9280.csv
```

Summary：

```text
stable_accept_full_row_materialization_pass = 0
outcome_labels_present = 1
candidate_missing_count = 0
duplicate_event_id_count = 0
accept_disagreement_count = 0
score_quantized_disagreement_count = 18
rank_disagreement_count = 2
secondary_outcome_delta_field_missing_count = 14380
```

判断：v9.2.80 真实解决了 v9.2.79 的 primary full-row materialization blocker；没有使用旧 artifact join，也没有用 family label 补 row label。secondary delta outcome fields 未在当前 measured rows 中提供，因此保持缺失并记录。

## 4. Calibration / Heldout

Artifact：

```text
p2_stable_accept_official_calibration_heldout_rerun.csv
```

Summary：

```text
stable_accept_official_calibration_pass = 0
stable_accept_heldout_support_pass = 0
accepted_count_held_native = 468
precision_heldout = 0.5128205128205128
coverage_heldout = 0.051587301587301584
bad_event_heldout = 0.3034188034188034
null_rate_heldout = 0.11752136752136752
precision_lcb = 0.47816314913056096
bad_event_ucb = 0.33529565729782573
```

判断：official gate 不再因为缺 row 而 blocked；现在是实测 heldout/support 结果。

## 5. Full-System Runtime

Artifacts：

```text
p4_full_system_step_attribution_native_stable_path.csv
p5_batch_major_native_runtime_closure.csv
```

Summary：

```text
native_bucket_kernel_used_in_p6 = 1
kernel_count_before/after = 3750 / 72576
sync_count_before/after = 1250 / 8064
avg_candidates_per_kernel_after = 0.03962742504409171
new_full_system_step_ratio_measured = 1
old_step_ratio_reused_as_measurement = 0
controller_step_ratio_q90 = 2.213009156635521
batch_major_runtime_pass = 0
```

判断：P6 runtime trace 是本轮新测量，不是复用 v9.2.79 的 `2.713296`，也不是 v9.2.78 local q90 代替。

## 6. P6 Boundary

Artifact：

```text
p6_system_legal_exact_signal_controller_v12.csv
```

Boundary：

```text
official_eligible = 0
system_legal_controller_pass = 0
reason = stable_accept_heldout_support_failed
step_ratio_q90 = 2.213009156635521
```

## 7. Downstream Boundary

P7-P10 仍按 P6 gate 处理；如果 P6 未 pass，均以 `not_run` 落盘，未把 materializer pass 写成 paired replay / short-run / full-run success。

## 8. No-fake Audit

```text
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

## 9. Hash

| artifact | SHA256 |
|---|---|
| plan | `6ee8743882efc33a39c2e75f66a3101cc5e4368e6873ca823d6fc5c792342155` |
| runner | `7d2c3157ccb1d2c9041a2352c918d92278505e66b50e4963c6f3a5aa7773a204` |
| run manifest | `60b9dc256a72cdc77fed8fdc4262899d1da6fc1c07bcf24b8ce2f1d6ea70dc90` |
| route | `6e7619a9095416d093d51bfd37bb8ac6f6532d2cb3ba28a15fcac956f25e268d` |
| P1 materializer | `65e79fe6f3fc3d87592b776f8cf695c2d94562a1ac284d805784250dcb052f05` |
| P2 calibration | `4c44aaf0d4f62bbdfcaddab07ca016fce2c28a9bb838b961aea590bccacadf98` |
| P4 step attribution | `59bc4e391203e4caa4a933735808184e86d29d87e9c52f2436b2579e7600ec75` |
| P5 runtime | `ec2f92a242ddb04e0668b30692630b61ab4d8d5c13ac1417251a73c303a0c323` |
| P6 system controller | `a7ed22fa1a4c47ff081dbccf9c469e76a5cf8ef152cad6bc3c02ebdc80668d3c` |
| provenance audit | `cf25dda00f8b16e9d3ef829cfc2318790f6fc7b2a5ea9d8835c368e81350fd05` |


## 10. 最终分析结论

v9.2.80 的真实推进是：

```text
v9.2.79: official promotion blocked by missing full-row stable accept/outcome materialization.
v9.2.80: full train-stream materializer rerun; AC2Q2 stable rows and primary outcome labels materialized for all candidates; calibration/heldout and native runtime gates are now measured rather than blocked.
```

最终一句话：

> v9.2.80 真实执行后 route = `R15-StableAcceptNativeQuantizationMismatch`：full-row materialization blocker 已推进为实测 P6 gate；strict PureKAN functional 状态由 `system_legal_controller_pass = 0` 决定，没有伪造任何缺失字段或 downstream 成功。
