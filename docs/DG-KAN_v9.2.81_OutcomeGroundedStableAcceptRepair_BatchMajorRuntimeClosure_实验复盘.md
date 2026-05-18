# DG-KAN v9.2.81 Outcome-Grounded StableAccept Repair 与 Full-System BatchMajor Runtime Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.2.81_OutcomeGroundedStableAcceptRepair_BatchMajorRuntimeClosure_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 borderline fallback、dataset-agnostic repair diagnostic 或 batch-major estimate 写成 official system pass。

## 0. 最新结论

```text
route = R16-OutcomeGroundedStableAcceptRegionUnsafe
base_candidate = LQ-t2-h256
success_v9281_strict_purekan_functional = False
success_v9281_full_functional = False
success_v9281_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9281_outcome_grounded_stable_accept_repair_batch_major_runtime_closure_first_20260513T210000Z/
```

核心结论：

1. P0 复现 v9.2.80 boundary：source route = `R15-StableAcceptNativeQuantizationMismatch`，full train-stream materializer = `1`，outcome labels present = `1`，official eligible = `0`。
2. P1 用 `QR5-BorderlineExactFallback` 将 score quantized disagreement 从 `18` 降为 `0`，rank disagreement 从 `2` 降为 `0`，accept disagreement 保持 `0`。
3. P1 不是 native integer emit 重写：`native_scoreq_emit_bitexact = 0`，fallback row count = `20`；不能把它写成 native kernel 已彻底修复。
4. P2 candidate lifecycle 审计显示 v9.2.72 old candidates = `2493`，v9.2.80/v9.2.81 new candidates = `2876`，shared = `2123`，old-only = `370`，new-only = `753`，Jaccard = `0.654036`。
5. P2 没有 pass：shared row 中 `candidate_id_mismatch_count = 2120`、`payload_hash_mismatch_count = 468`，说明 candidate lifecycle/schema drift 没有完全闭合。
6. P3 primary outcome labels 可靠：missing primary label = `0`，ambiguous label = `0`，safe/bad exclusivity violation = `0`。
7. P3 downstream 仍不 ready：secondary delta missing count = `14380`；没有补造 ECE/NLL/curvature/real-beats 字段。
8. P4 decision autopsy 过 gate：heldout accepted = `468`，safe = `240`，bad = `142`，null = `57`；bad accepted attribution fraction = `1.0`。
9. P4 dominant failure mode = `FMA3-risk-ucb-underestimation`；failure counts = `FMA3:70, FMA2:41, FMA5:20, FMA7:11`。
10. P5 测了 dataset-agnostic repair candidates，但没有一个过 official decision gate。
11. 最佳非 posthoc repair = `DR2-BadLevelTailRiskFilter`：precision = `0.630037`，coverage = `0.030093`，bad-event = `0.183150`，null-rate = `0.194139`，仍远低于 official gate。
12. P6 runtime fragmentation autopsy 过 gate：step count = `8064`，active step = `1412`，zero-candidate step = `6652`，empty-step kernel fraction = `0.824901`。
13. P6 dominant runtime fragmentation mode = `FR2-per-step-sync-empty-step-fixed-launch`；kernel_count_after = `72576`，sync_count_after = `8064`，avg candidates/kernel = `0.039627`。
14. P7 没有 materialize 新 batch-major runtime；`RT1-ActiveStepBatchMajorGroupingDiagnostic` 只是 diagnostic estimate，不能写成 measured runtime pass。
15. P8 system controller 未打开：`official_eligible = 0`，`system_legal_controller_pass = 0`，reason = `decision_repair_failed_and_batch_major_runtime_not_materialized`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9281_outcome_grounded_stable_accept_repair_batch_major_runtime_closure.py` | v9.2.81 runner；读取 v9.2.80 full-row materializer，执行 QR fallback、candidate lifecycle audit、outcome label audit、decision failure autopsy、dataset-agnostic repair candidate rerun、runtime fragmentation autopsy 和 P8 gate |

代码检查：

```text
python -m py_compile experiments/run_v9281_outcome_grounded_stable_accept_repair_batch_major_runtime_closure.py
```

正式运行：

```bash
python experiments/run_v9281_outcome_grounded_stable_accept_repair_batch_major_runtime_closure.py \
  --out-dir results/real_rerun_20260506/v9281_outcome_grounded_stable_accept_repair_batch_major_runtime_closure_first_20260513T210000Z \
  --fresh --device auto --data-root data --seed 1314
```

实际参数来自 `run_manifest.json`：

```text
source_v9280 = results/real_rerun_20260506/v9280_full_train_stream_stable_accept_materializer_outcome_label_rebuild_rerun_20260513T190000Z
source_v9272 = results/real_rerun_20260506/v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline_async_basis_cuda_ext_20260513T110000Z
event_count = 24192
candidate_count = 2876
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R16-OutcomeGroundedStableAcceptRegionUnsafe",
  "v9280_boundary_pass": 1,
  "full_train_stream_materializer_used": 1,
  "stable_accept_full_row_materialization_present": 1,
  "outcome_labels_present": 1,
  "candidate_count": 2876,
  "event_count": 24192,
  "quantized_rank_repair_pass": 1,
  "score_quantized_disagreement_count_before": 18,
  "rank_disagreement_count_before": 2,
  "accept_disagreement_count_before": 0,
  "score_quantized_disagreement_count_after": 0,
  "rank_disagreement_count_after": 0,
  "accept_disagreement_count_after": 0,
  "borderline_exact_fallback_row_count": 20,
  "candidate_lifecycle_audit_pass": 0,
  "candidate_jaccard": 0.6540357362908195,
  "outcome_label_primary_pass": 1,
  "outcome_downstream_ready_pass": 0,
  "missing_secondary_delta_count": 14380,
  "decision_autopsy_pass": 1,
  "dominant_decision_failure_mode": "FMA3-risk-ucb-underestimation",
  "best_repair_candidate_id": "DR2-BadLevelTailRiskFilter",
  "best_repair_precision_heldout": 0.63003663003663,
  "best_repair_coverage_heldout": 0.03009259259259259,
  "best_repair_bad_event_heldout": 0.18315018315018314,
  "best_repair_null_rate_heldout": 0.19413919413919414,
  "decision_repair_pass": 0,
  "runtime_fragmentation_autopsy_pass": 1,
  "dominant_runtime_fragmentation_mode": "FR2-per-step-sync-empty-step-fixed-launch",
  "empty_step_kernel_fraction": 0.8249007936507936,
  "controller_step_ratio_q90": 2.213009156635521,
  "batch_major_runtime_pass": 0,
  "official_eligible": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "outcome_grounded_decision_repair_failed",
  "next_required_implementation": "redesign_dataset_agnostic_accept_score_or_risk_support_features"
}
```

判断：v9.2.81 真实推进了 quantized/rank contract 和 failure autopsy，但 official blocker 现在是 outcome-grounded decision region 不安全；batch-major runtime 也仍未 materialize。

## 3. P1 stable quantized/rank bit-exact repair

Artifacts：

```text
p1_stable_quantized_rank_bitexact_repair.csv
stable_quantized_rank_trace_v9281.csv
```

Summary：

```text
quantized_repair_id = QR5-BorderlineExactFallback
score_quantized_disagreement_count_before = 18
rank_disagreement_count_before = 2
accept_disagreement_count_before = 0
score_quantized_disagreement_count_after = 0
rank_disagreement_count_after = 0
accept_disagreement_count_after = 0
fallback_row_count = 20
fallback_event_count = 20
native_scoreq_emit_bitexact = 0
official_native_integer_emit_repaired = 0
```

判断：QR5 关闭了 full-row contract mismatch，但它依赖 same-run reference score 的 borderline fallback；不是 native kernel 直接 emit bit-exact integer score_q。因此它可以解除“accept/rank audit mismatch”诊断，但不能单独证明 native quantization kernel 已完全重写。

## 4. P2 candidate-set / row-lifecycle audit

Artifacts：

```text
p2_candidate_set_row_lifecycle_continuity_audit.csv
candidate_set_diff_trace_v9281.csv
```

Summary：

```text
old_candidate_count = 2493
new_candidate_count = 2876
shared_candidate_count = 2123
old_only_candidate_count = 370
new_only_candidate_count = 753
candidate_jaccard = 0.6540357362908195
old_only_accept_rate = 0.467568
new_only_accept_rate = 0.423639
shared_accept_rate = 0.488931
new_only_bad_event_rate = 0.278884
shared_bad_event_rate = 0.277909
new_only_precision = 0.398406
shared_precision = 0.544984
candidate_id_mismatch_count = 2120
payload_hash_mismatch_count = 468
candidate_lifecycle_audit_pass = 0
```

判断：candidate count 从 `2493` 到 `2876` 不是一个小 drift；new-only rows 的 precision 明显更低，但 shared rows 本身 bad-event 也很高，因此 decision failure 不能只归咎于 new-only candidates。P2 未过，原因是 shared rows 的 candidate id/hash schema 没有完全连续。

## 5. P3 outcome label / secondary delta materialization

Artifacts：

```text
p3_outcome_label_secondary_delta_materialization.csv
outcome_secondary_delta_trace_v9281.csv
```

Summary：

```text
missing_primary_label_count = 0
ambiguous_label_count = 0
label_exclusivity_violation_count = 0
label_source = same_run_train_stream
outcome_label_primary_pass = 1
missing_secondary_delta_count = 14380
outcome_downstream_ready_pass = 0
```

判断：primary official decision gate 可以真实评估；paired replay / downstream readiness 仍不能打开，因为 ECE/NLL/curvature/real-beats 等 secondary fields 没有全量 materialize。

## 6. P4 decision failure autopsy

Artifacts：

```text
p4_decision_failure_autopsy.csv
bad_accepted_failure_trace_v9281.csv
```

Heldout current AC2Q2：

```text
accepted_count = 468
safe_good_count = 240
bad_event_count = 142
null_event_count = 57
precision = 0.5128205128205128
coverage = 0.051587301587301584
bad_event_rate = 0.3034188034188034
null_rate = 0.12179487179487179
```

Failure mode：

```text
bad_accepted_attribution_fraction = 1.0
dominant_failure_mode = FMA3-risk-ucb-underestimation
failure_mode_counts = {
  "FMA2-candidate-set-drift": 41,
  "FMA3-risk-ucb-underestimation": 70,
  "FMA5-null-bad-conflict": 20,
  "FMA7-horizon-tail-risk": 11
}
```

判断：P4 回答了“为什么 precision 掉到 0.513”：bad accepted 主要来自风险分层不足和 candidate drift，而不是 quantized/rank mismatch。P4 attribution fraction 达到 `1.0`。

## 7. P5 dataset-agnostic decision repair

Artifacts：

```text
p5_dataset_agnostic_stable_accept_decision_repair.csv
decision_repair_trace_v9281.csv
```

Best non-posthoc candidate：

```text
repair_candidate_id = DR2-BadLevelTailRiskFilter
dataset_name_used = 0
accepted_count_heldout = 273
precision_heldout = 0.63003663003663
coverage_heldout = 0.03009259259259259
bad_event_heldout = 0.18315018315018314
null_rate_heldout = 0.19413919413919414
precision_lcb = 0.5713310098976009
bad_event_ucb = 0.23332190828443805
accepted_signal_strata_count = 14
accepted_family_count = 57
max_family_share = 0.0989010989010989
max_stratum_share = 0.32234432234432236
decision_gate_pass = 0
```

判断：dataset-agnostic filter 有改善，但远没到 official frontier：bad-event 仍 `0.183150 > 0.05`，null-rate `0.194139 > 0.15`，precision LCB 也只有 `0.571331`。这说明当前 AC2Q2 accepted region 在 same-run labels 下需要重新设计 accept score 或更强的合法 risk/support features，而不是再调一个轻量 threshold。

## 8. P6/P7 runtime fragmentation 与 batch-major boundary

Artifacts：

```text
p6_full_system_runtime_fragmentation_autopsy.csv
runtime_fragmentation_trace_v9281.csv
p7_batch_major_native_runtime_rewiring.csv
batch_major_runtime_trace_v9281.csv
```

P6 summary：

```text
step_count = 8064
total_step_candidates = 2876
active_step_count = 1412
zero_candidate_step_count = 6652
kernel_count_after = 72576
sync_count_after = 8064
kernels_per_step = 9.0
syncs_per_step = 1.0
avg_candidates_per_kernel_after = 0.03962742504409171
empty_step_kernel_fraction = 0.8249007936507936
dominant_runtime_fragmentation_mode = FR2-per-step-sync-empty-step-fixed-launch
runtime_fragmentation_autopsy_pass = 1
```

P7 boundary：

```text
RT0-v9280-full-system-native-reference:
  materialized = 1
  step_ratio_q90 = 2.213009156635521
  batch_major_runtime_pass = 0

RT1-ActiveStepBatchMajorGroupingDiagnostic:
  diagnostic_derived_from_measured_components = 1
  batch_major_runtime_materialized = 0
  batch_major_runtime_pass = 0
```

判断：runtime failure 的来源很明确：P6 path 在大量 zero-candidate steps 上仍固定 launch/sync，导致 avg candidates per kernel 只有 `0.0396`。但本轮没有实现新的 measured batch-major grouped kernel，所以不能把 RT1 diagnostic 写成 runtime pass。

## 9. P8 system controller boundary

Artifact：

```text
p8_system_legal_exact_signal_controller_v13.csv
system_controller_trace_v9281.csv
```

Boundary：

```text
controller_id = C3Q2-v13-OutcomeGroundedStableAcceptRepair
decision_repair_candidate_id = DR2-BadLevelTailRiskFilter
runtime_candidate_id = RT0-v9280-full-system-native-reference
accept_contract_id = AC2Q2+QR5-borderline-exact-fallback
precision_heldout = 0.63003663003663
coverage_heldout = 0.03009259259259259
bad_event_heldout = 0.18315018315018314
null_rate_heldout = 0.19413919413919414
step_ratio_q90 = 2.213009156635521
secondary_outcome_delta_fields_present = 0
decision_gate_pass = 0
runtime_gate_pass = 0
official_eligible = 0
system_legal_controller_pass = 0
reason = decision_repair_failed_and_batch_major_runtime_not_materialized
```

判断：P8 不能打开。没有使用 dataset_name，没有 source-measured gap / formula proxy / CPU offload，也没有把 diagnostic-derived runtime 写成 official。

## 10. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p9_leave_dataset_and_stratum_out.csv` | `P8_system_controller_not_official` |
| `p10_official_paired_replay.csv` | same |
| `p11_short_run_functional_validation.csv` | same |
| `p12_full_run_robustness_strong_baseline.csv` | same |

没有把 QR5 fallback、primary label pass、decision autopsy pass 或 runtime fragmentation autopsy pass 写成 LDO/paired replay/short/full success。

## 11. No-fake audit

```text
rows_checked = 2876
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
stable_accept_full_row_materialization_present = 1
outcome_labels_present = 1
quantized_rank_repair_pass = 1
outcome_label_primary_pass = 1
outcome_downstream_ready_pass = 0
decision_repair_pass = 0
runtime_fragmentation_autopsy_pass = 1
batch_major_runtime_pass = 0
system_legal_controller_pass = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
projection_used_for_official = 0
```

## 12. Hash

| artifact | SHA256 |
|---|---|
| plan | `dc767381306273f379adb8eb1d72187f0e60b055212bc228af25686ab6e22561` |
| runner | `089a6dc1d23918965dbd2d017388d76a3ced09908c05de937e7a0848ef4233c2` |
| run manifest | `e480634fe28f7d9a2a82c5b56dc46503fa2aa084cd740800d973bcc8c5ba6d34` |
| route | `29b65c39c3105f19ab73f35de8f25807c91ed9a0c436f989c895f51e3307be1d` |
| P0 boundary | `938cd671c480ff3c2f1b7e4fc58c6a0d9fc2f085b4716519a8285fce2a499241` |
| P1 quantized rank | `e1af685c9919331b5345c03dceb7aa79063c38effe8cc6971b2a8620c7883e3f` |
| P2 candidate set | `6f05e31d4f9ce6f3590fd60b70e7cbc8b80bdb63cc4dc7667d40349682f5b7cb` |
| P3 outcome labels | `99626650344985cc7308ddcd50cf31969e39007069ed40e787d5addc8c81dd28` |
| P4 decision autopsy | `223455d553836024503fdf014473f4d1708eb24fea484de6760e9e024ff143f4` |
| P5 decision repair | `1ce885aae75681cb7b2824931eb0b0def494e61d1892ed3559b2273ce8b3374e` |
| P6 runtime autopsy | `36189c9be0ae132d400adc21ccc15b58691ae2e95c89be5e1bd9278d085bd3df` |
| P7 batch major | `0152e9ede6a7bf796362ec4d3278e7cf2bff9f33bb77a4311e6681f2f55fab5b` |
| P8 system controller | `d631a2ff0eea6df9c099ff49311e670b44f923246eb28c6f5be26cd0098664bd` |
| contract audit | `26aacf0f204e750bc198c873c93920426cd791cf7467d9e28577110de8624272` |
| provenance audit | `99f7e2b0578bf645346463fb5710b62b295fcfb4a62d856a3511e12ad073d8e0` |

## 13. 最终分析结论

v9.2.81 的真实推进是：

```text
v9.2.80:
  full-row materializer 和 primary same-run labels 已落盘；
  native stable kernel 已进入 P6；
  但 quantized/rank mismatch、decision gate、runtime gate 均失败。

v9.2.81:
  quantized/rank mismatch 通过 QR5 borderline exact fallback 清零；
  primary labels 通过一致性审计；
  decision failure 被定位到 risk/candidate/null/tail failure modes；
  runtime failure 被定位到 empty-step fixed launch/sync fragmentation；
  但 dataset-agnostic repair 未恢复 safe-good frontier，batch-major runtime 也未 materialize。
```

机制判断：

1. H1 局部成立：quantized/rank mismatch 可以清零，但只是 fallback，不是 native emit 重写；修完后 decision metrics 没恢复，说明它不是主因。
2. H2 成立：same-run outcome labels 是可用的，primary missing/ambiguous/exclusivity violation 都为 `0`；decision failure 是实测问题。
3. H3 未成立：本轮测得的 dataset-agnostic repair candidates 全部不过 official gate；最佳 DR2 仍 `bad_event = 0.183150`。
4. H4 部分成立：candidate set drift 明显存在，Jaccard 只有 `0.654036`，new-only precision 更差；但 shared rows bad-event 也高，所以 drift 不是唯一原因。
5. H5 成立：runtime failure 来自 full-system wiring fragmentation，empty-step kernel fraction = `0.824901`；但本轮未实现 measured batch-major rewiring。
6. H6 未打开：P8 未 pass，因此 P9-P12 全部 gate-blocked。

最终一句话：

> v9.2.81 真实执行后停在 `R16-OutcomeGroundedStableAcceptRegionUnsafe`：QR5 已清掉 quantized/rank mismatch，primary labels 和 failure autopsy 都闭合，但 same-run outcome 下的 StableAccept region 仍不安全，batch-major runtime 也仍未 materialize，strict PureKAN functional 仍未成功。
