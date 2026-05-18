# DG-KAN v9.6.3 Group-Stable Accepted Region / Memory-Offdiag Primitive 实验复盘

> 本复盘记录 `DG-KAN_v9.6.3_GroupStableAcceptedRegion_MemoryOffdiagPrimitive_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 accepted-region diagnostic、multi-axis pocket diagnostic、rank diagnostic、APGH implementation/smoke、runtime boundary 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R2-hidden_multiaxis_pocket_explains_rank
base_candidate = LQ-t2-h256
success_v9630_strict_purekan_functional = False
success_v9630_full_functional = False
success_v9630_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9630_group_stable_accepted_region_memory_offdiag_primitive_first_20260515T130000Z/
```

核心结论：

1. P0 复现 v9.6.2 boundary：source route = `R4-PayloadPocketStopped`，payload pocket route stop = `1`，v9.6.2 system pass = `0`。
2. P1 accepted-region scope audit 过：accepted action / unique action / unique event = `87 / 87 / 87`，duplicate action id = `0`，scope mismatch = `0`，universe mismatch = `0`。
3. P1 accepted region 本身确实有强 diagnostic：GradeAB precision = `0.8735632183908046`，V LCB = `0.14111334880346277`，h240 longrisk UCB = `0.0`。
4. 但 P1/P2 显示该 accepted region 被 group pocket 解释：`payload_norm_bucket=payload0` 的 accepted share = `1.0`，drop if removed = `0.8735632183908046`，且该字段禁止作为 selector。
5. P2 multi-axis hidden pocket search 过解释 gate：multi-axis pocket explained = `1`，official usable = `0`，primary blocker 因此为 `multi_axis_group_pocket_explains_rank`。
6. P3 group-balanced target support 未过：best target = `T2-ValuePositiveNoLongRisk`，count = `97`，coverage = `0.03372739916550765`，V LCB = `0.16622185363738462`，longrisk UCB = `0.0`，但 positive group coverage = `0.11988847583643122`，max group share = `1.0`。
7. P4 legal feature deconfounding 未过：best = `COMBO-ValueRiskMemoryCover`，TopK87 precision = `0.6551724137931034`，V LCB = `0.049639752010919136`，longrisk UCB = `0.0`，但 LDO drop = `0.22988505747126436`。
8. P5 value/risk/memory ranker 未过：best = `RANK-D-value-risk-memory-veto`，TopK87 precision = `0.5862068965517241`，V LCB = `-0.0057476968109164694`，LDO drop = `0.2298850574712643`，max group share = `1.0`。
9. P6 rank-safe certificate v11 未过：best = `CERT11-FrozenTopK87`，accepted = `87`，coverage = `0.030250347705146036`，precision = `0.5862068965517241`，V LCB = `-0.0057476968109164694`。
10. P7/P8 gate-blocked：existing-action controller 与 selected runtime 均 not_run。
11. P9 generated damage autopsy 过 diagnostic：dominant damage mode = `D2-value-direction-lost`，fraction = `0.5911458333333334`，longrisk created rate = `0.744140625`，Damage V LCB = `-0.7263338315208386`。
12. P10/P11 APGH implementation 与 branch-horizon materializer 过：generated actions = `512`，expected/actual rows = `12288 / 12288`，unresolved exception = `0`，quality audit pass = `1`。
13. P12 APGH outcome 未过：best = `APGH7-SignalChannelSNRMemoryDelta`，GradeAB precision = `0.046875`，V LCB = `-0.48807717058315`，h240 longrisk UCB = `0.7869099970100811`。
14. P13-P16 gate-blocked：没有 APGH controller、official paired replay、short/full training 或 continual validation。
15. Base-Acc Sentinel 继续健康：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
16. 当前 primary blocker：`multi_axis_group_pocket_explains_rank`；secondary blocker：`apgh_generated_frontier_fail`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9630_group_stable_accepted_region_memory_offdiag_primitive.py` | v9.6.3 runner；读取 v9.6.2/v9.6.1/v9.5.9/v9.5.8/v9.5.6 artifacts，执行 accepted-region scope audit、multi-axis hidden pocket search、group-balanced target support、legal feature/rank/certificate gate、generated damage autopsy、APGH primitive 与 system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9630_group_stable_accepted_region_memory_offdiag_primitive.py
```

正式运行：

```bash
python experiments/run_v9630_group_stable_accepted_region_memory_offdiag_primitive.py \
  --out-dir results/real_rerun_20260506/v9630_group_stable_accepted_region_memory_offdiag_primitive_first_20260515T130000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --apgh-actions-per-primitive 64
```

运行结果：

```json
{
  "apgh_generated_actions": 512,
  "apgh_rows": 12288,
  "best_apgh_V_integrated_LCB": -0.48807717058315,
  "best_apgh_primitive": "APGH7-SignalChannelSNRMemoryDelta",
  "best_ranker": "RANK-D-value-risk-memory-veto",
  "best_ranker_LDO_drop": 0.2298850574712643,
  "best_ranker_TopK87_precision": 0.5862068965517241,
  "out_dir": "results/real_rerun_20260506/v9630_group_stable_accepted_region_memory_offdiag_primitive_first_20260515T130000Z",
  "primary_blocker": "multi_axis_group_pocket_explains_rank",
  "route": "R2-hidden_multiaxis_pocket_explains_rank",
  "system_legal_controller_pass": 0
}
```

说明：本轮没有 CPU offload，没有 proxy rows；APGH branch-horizon replay 完整落盘，但没有转成 generated frontier pass。

## 2. Route

`p14_route_decision_v9630.json` 摘要：

```json
{
  "route": "R2-hidden_multiaxis_pocket_explains_rank",
  "source_route_v9620": "R4-PayloadPocketStopped",
  "p0_pass": 1,
  "scope_consistency_pass": 1,
  "multi_axis_pocket_explained": 1,
  "target_support_pass": 0,
  "feature_deconfounding_pass": 0,
  "ranker_weak_pass": 0,
  "rank_safe_certificate_v11_pass": 0,
  "existing_action_controller_pass": 0,
  "selected_runtime_pass": 0,
  "damage_autopsy_pass": 1,
  "apgh_implementation_pass": 1,
  "apgh_branch_horizon_pass": 1,
  "apgh_weak_pass": 0,
  "best_apgh_primitive": "APGH7-SignalChannelSNRMemoryDelta",
  "best_apgh_GradeAB_precision": 0.046875,
  "best_apgh_V_integrated_LCB": -0.48807717058315,
  "best_apgh_h240_longrisk_UCB": 0.7869099970100811,
  "system_legal_controller_pass": 0,
  "primary_blocker": "multi_axis_group_pocket_explains_rank",
  "secondary_blocker": "apgh_generated_frontier_fail"
}
```

判断：v9.6.3 不是 certificate scope mismatch，也不是 APGH engineering failure。P1 证明 accepted-region scope 一致；P2 证明该 accepted region 主要由 group pocket 解释，且关键 pocket route 已不能 official。因此 route 优先停在 `R2-hidden_multiaxis_pocket_explains_rank`。

## 3. P0 v9.6.2 boundary

Artifact：

```text
p0_boundary_reproduction_v9630.csv
```

Summary：

```text
source_route_v9620 = R4-PayloadPocketStopped
payload_pocket_route_stop_v9620 = 1
strong_confounder_count_v9620 = 0
best_target_id_v9620 = T2-ValuePositiveNoLongRisk
best_target_count_v9620 = 97
best_target_coverage_v9620 = 0.03372739916550765
best_ranker_id_v9620 = RANK5-invariant-risk-minimization
best_ranker_TopK87_precision_v9620 = 0.8735632183908046
best_ranker_LDO_drop_v9620 = 0.37931034482758624
best_ranker_max_group_share_v9620 = 1.0
best_certificate_id_v9620 = CERT10-FixedTopKGroupBalanced
best_certificate_accepted_count_v9620 = 87
best_certificate_precision_v9620 = 0.8735632183908046
best_certificate_V_LCB_v9620 = 0.14111334880346277
best_certificate_longrisk_UCB_v9620 = 0.0
best_apgf_primitive_v9620 = APGF5-CoverEntropyPreservingDelta
best_apgf_V_LCB_v9620 = -0.33527855012819596
best_apgf_longrisk_UCB_v9620 = 0.828904255301089
p0_pass = 1
```

判断：P0 pass。v9.6.3 没有跳过 v9.6.2 的 payload-pocket-stopped / APGF generated frontier failure boundary。

## 4. P1 accepted-region scope audit

Artifacts：

```text
p1_certificate_scope_audit_v9630.csv
fig_p1_accepted_group_share.svg
```

Summary：

```text
accepted_action_count = 87
accepted_unique_action_count = 87
accepted_unique_candidate_count = 1
accepted_unique_event_count = 87
duplicate_action_id_count = 0
row_action_scope_mismatch = 0
universe_mismatch = 0
GradeAB_count_in_eval_universe = 79
GradeAB_count_in_accepted = 76
GradeAB_precision = 0.8735632183908046
V_LCB = 0.14111334880346277
h240_longrisk_UCB = 0.0
scope_consistency_pass = 1
worst_axis = payload_norm_bucket
worst_axis_top_group = payload0
worst_axis_max_group_share = 1.0
```

Representative groups：

| axis | top group | max group share | base share | lift |
|---|---:|---:|---:|---:|
| `payload_norm_bucket` | `payload0` | `1.0` | `1.0` | `0.0` |
| `action_norm_bucket` | `anorm0` | `1.0` | `1.0` | `0.0` |
| `candidate_origin` | `canonical_AP0` | `1.0` | `1.0` | `0.0` |
| `memory_bucket` | `memory_fail_0` | `1.0` | `0.17767732962447844` | `0.8223226703755215` |
| `offdiag_bucket` | `offdiag_fail_0` | `1.0` | `0.13803894297635605` | `0.861961057023644` |
| `dataset_id` | `KMNIST` | `0.41379310344827586` | `0.28789986091794156` | `0.1258932425303343` |

判断：P1 回答了 v9.6.2 P6 的口径疑问：accepted action count 与 eval universe 没有 duplicate/scope mismatch。问题不是统计口径，而是 accepted region 被多个 group pocket 高度集中。

## 5. P2 multi-axis hidden pocket search

Artifacts：

```text
p2_multiaxis_hidden_pocket_search_v9630.csv
fig_p2_drop_if_removed_pareto.svg
```

Summary：

```text
interaction_candidate_count_scanned = 469
interaction_row_count_written = 256
best_axis_set = payload_norm_bucket
best_group_id = payload0
best_support = 2876
best_accepted_share = 1.0
best_drop_if_removed = 0.8735632183908046
best_drop_explained_fraction = 1.0
best_forbidden_selector_field_used = 1
multi_axis_pocket_explained = 1
multi_axis_pocket_official_usable = 0
```

判断：P2 是本轮 terminal blocker。`payload_norm_bucket` 能解释 rank pocket，但已在 v9.6.2 被停止为 confounded route，只能作为 stratification/diagnostic。`offdiag_bucket` / `risk_score_bucket` 也显示 accepted concentration，但这仍是 pocket 解释，不是 official accepted region。

## 6. P3 target support map

Artifacts：

```text
p3_group_balanced_target_support_map_v9630.csv
fig_p3_target_count_by_id.svg
```

Summary：

```text
best_target_id = T2-ValuePositiveNoLongRisk
best_target_count = 97
best_target_coverage = 0.03372739916550765
best_V_LCB = 0.16622185363738462
best_longrisk_UCB = 0.0
best_positive_group_coverage = 0.11988847583643122
best_max_group_share = 1.0
target_support_pass = 0
target_support_weak_pass = 0
```

Representative targets：

| target | count | coverage | V LCB | longrisk UCB |
|---|---:|---:|---:|---:|
| `T2-ValuePositiveNoLongRisk` | `97` | `0.03372739916550765` | `0.16622185363738462` | `0.0` |
| `T3-MemorySafeValuePositive` | `90` | `0.03129346314325452` | `0.1601306386208506` | `0.0` |
| `T1-GradeAB` | `79` | `0.027468706536856746` | `0.16201929527024647` | `0.0` |

判断：target 的 value/risk 是干净的，但 group support 不稳，max group share 仍为 `1.0`。这解释了为什么 coverage 达到 0.03 也不能直接 official。

## 7. P4-P6 legal feature / rank / certificate

P4 artifacts：

```text
p4_legal_feature_deconfounding_v2_v9630.csv
fig_p4_feature_topk_precision.svg
```

P4 summary：

```text
feature_count = 23
best_feature_id = COMBO-ValueRiskMemoryCover
best_TopK87_precision = 0.6551724137931034
best_TopK87_V_LCB = 0.049639752010919136
best_TopK87_longrisk_UCB = 0.0
best_LDO_drop = 0.22988505747126436
best_LSO_drop = 0.011494252873563204
feature_deconfounding_pass = 0
```

P5 artifacts：

```text
p5_value_rank_risk_memory_veto_ranker_v9630.csv
fig_p5_ranker_precision.svg
```

P5 summary：

```text
ranker_count = 8
best_ranker_id = RANK-D-value-risk-memory-veto
best_TopK87_precision = 0.5862068965517241
best_V_LCB = -0.0057476968109164694
best_longrisk_UCB = 0.0
best_LDO_drop = 0.2298850574712643
best_LSO_drop = 0.03448275862068961
best_max_group_share = 1.0
ranker_weak_pass = 0
ranker_strong_pass = 0
existing_action_rank_not_deployable = 1
```

P6 artifact：

```text
p6_rank_safe_certificate_v11_v9630.csv
```

P6 summary：

```text
certificate_count = 5
best_certificate_id = CERT11-FrozenTopK87
best_ranker_id = RANK-D-value-risk-memory-veto
best_accepted_heldout = 87
best_coverage_heldout = 0.030250347705146036
best_precision_heldout = 0.5862068965517241
best_V_LCB_heldout = -0.0057476968109164694
best_longrisk_UCB_heldout = 0.0
best_LDO_drop = 0.2298850574712643
best_LSO_drop = 0.03448275862068961
best_max_group_share = 1.0
rank_safe_certificate_v11_pass = 0
```

判断：分离 value/risk/memory 之后，TopK precision 与 V LCB 反而低于 v9.6.2 的 pooled diagnostic。P4-P6 都不能打开 existing-action controller。

## 8. P7-P8 controller/runtime boundary

P7：

```text
p7_existing_action_controller_v9630.csv = not_run
reason = P6_certificate_not_passed
source_controller_pass = 0
system_controller_candidate_pass = 0
```

P8：

```text
p8_selected_controller_runtime_preflight_v9630.csv = not_run
reason = P7_controller_not_selected
selected_runtime_pass = 0
```

判断：没有把 P4-P6 rank/certificate diagnostic 写成 official controller 或 runtime pass。

## 9. P9 generated damage autopsy

Artifacts：

```text
p9_generated_damage_autopsy_v2_v9630.csv
fig_p9_damage_mode_histogram.svg
```

Summary：

```text
damage_row_count = 1536
assigned_damage_fraction = 1.0
dominant_damage_mode = D2-value-direction-lost
dominant_damage_mode_fraction = 0.5911458333333334
longrisk_created_rate = 0.744140625
memory_offdiag_fail_rate = 0.3860677083333333
Damage_V_LCB = -0.7263338315208386
damage_autopsy_pass = 1
```

判断：generated families 的主要问题不是 replay/materializer，而是 value direction 丢失与 longrisk created 仍高。该诊断支持后续 APGH 设计，但不能 official。

## 10. P10-P12 APGH primitive

P10 artifact：

```text
p10_apgh_primitive_spec_preflight_v9630.csv
```

P10 summary：

```text
primitive_count = 8
generated_action_count_expected/actual = 512 / 512
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_linf_max = 0.0
no_transform_equivalence = 1
negative_control_divergence = 1
apgh_implementation_pass = 1
apgh_preflight_pass = 1
```

P11 artifacts：

```text
p11_apgh_branch_horizon_smoke_v9630.csv
```

P11 summary：

```text
branch_horizon_rows_expected/actual = 12288 / 12288
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
secondary_delta_completion_rate = 1
rows_per_sec = 14.969132850524907
wallclock_sec = 820.8892340459861
unresolved_exception_count = 0
duplicate_rows = 0
label_exclusivity_violation = 0
quality_audit_pass = 1
apgh_branch_horizon_pass = 1
```

P12 artifacts：

```text
p12_apgh_outcome_geometry_pass_v9630.csv
fig_p12_apgh_gradeab_by_primitive.svg
```

P12 summary：

```text
best_primitive_id = APGH7-SignalChannelSNRMemoryDelta
best_GradeAB_precision = 0.046875
best_V_integrated_LCB = -0.48807717058315
best_h240_longrisk_UCB = 0.7869099970100811
best_new_positive_created_rate = 0.015625
best_longrisk_created_rate = 0.671875
best_Damage_V_LCB = -0.5605118701699974
canonical_GradeAB_memory_fail_rate = 0.02531645569620253
apgh_weak_pass = 0
apgh_strong_pass = 0
generated_family_reset_required = 1
```

Per primitive：

| primitive | GradeAB precision | V LCB | h240 longrisk UCB | longrisk created |
|---|---:|---:|---:|---:|
| `APGH1-PopRiskOffdiagProjectedDelta` | `0.015625` | `-0.606722092637292` | `0.8010605393336523` | `0.6875` |
| `APGH2-MemoryPreservingEdgeMaskDelta` | `0.03125` | `-0.5220645581548111` | `0.8825326673766913` | `0.78125` |
| `APGH3-CoverEntropyNonCollapseDelta` | `0.0` | `-0.5469341942831044` | `0.8150608413036654` | `0.703125` |
| `APGH4-OldFamilyOrthogonalizedValueDelta` | `0.0` | `-0.49432771281063476` | `0.8150608413036654` | `0.703125` |
| `APGH5-ValueRankAnchoredRiskVetoDelta` | `0.015625` | `-0.4775887618516529` | `0.8010605393336523` | `0.6875` |
| `APGH6-MultiHorizonBoundarySymmetricDelta` | `0.0` | `-0.5378927687094387` | `0.8560881119635937` | `0.75` |
| `APGH7-SignalChannelSNRMemoryDelta` | `0.046875` | `-0.48807717058315` | `0.7869099970100811` | `0.671875` |
| `APGH8-NegativeControlShuffledPayload` | `0.0` | `-0.5804815654495317` | `0.8010605393336523` | `0.6875` |

判断：APGH implementation/preflight/replay 全部闭合，但 outcome 仍 value-negative / high-longrisk。APGH 没有生成 value-positive / horizon-safe frontier。

## 11. P13-P16 boundary

P13：

```text
p13_apgh_controller_v9630.csv = not_run
reason = P12_APGH_strong_pass_failed
apgh_controller_pass = 0
```

P15-P16：

| artifact | status / reason |
|---|---|
| `p15_leaveout_paired_replay_boundary_v9630.csv` | `not_run`, `P8_or_system_not_official` |
| `p16_short_full_boundary_v9630.csv` | `not_run`, `P15_paired_replay_not_open` |

判断：没有 controller/runtime/APGH official pass，因此 leaveout、paired replay、short/full 都未打开。

## 12. Base-Acc Sentinel

Artifact：

```text
base_acc_sentinel_v9630.csv
```

Summary：

```text
sentinel_row_count = 120
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7,8,9
model_count = 4
sentinel_complete = 1
mean_test_acc_LQ = 0.6537760416666667
mean_test_acc_MLP = 0.562890625
mean_test_acc_QuadraticFeatureMLP = 0.40234375
mean_test_acc_AdamWStrongLRGridMLP = 0.628515625
LQ_catastrophic_fail = 0
base_acc_used_for_controller = 0
base_acc_sentinel_pass = 1
```

判断：Base-Acc Sentinel 继续健康，但仍是 isolated diagnostic，没有用于 selector、generator、certificate 或 controller。

## 13. Figures

本轮额外落盘的诊断图：

```text
fig_p1_accepted_group_share.svg
fig_p2_drop_if_removed_pareto.svg
fig_p3_target_count_by_id.svg
fig_p4_feature_topk_precision.svg
fig_p5_ranker_precision.svg
fig_p9_damage_mode_histogram.svg
fig_p12_apgh_gradeab_by_primitive.svg
```

判断：这些图只用于复核诊断，不构成 official pass。

## 14. No-fake audit

```text
rows_checked = 14803
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = 1
no_proxy = 1
```

Contract audit：

```text
manual_forward/manual_backward/manual_adamw_update = 1/1/1
train_stream_probe = 1
v9620_boundary_pass = 1
scope_consistency_pass = 1
multi_axis_pocket_explained = 1
target_support_pass = 0
feature_deconfounding_pass = 0
ranker_weak_pass = 0
rank_safe_certificate_pass = 0
existing_controller_pass = 0
selected_runtime_pass = 0
damage_autopsy_pass = 1
apgh_implementation_pass = 1
apgh_branch_horizon_pass = 1
apgh_weak_pass = 0
apgh_controller_pass = 0
system_legal_controller_pass = 0
base_acc_sentinel_pass = 1
base_acc_used_for_controller = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_selector/controller = 0/0
uses_validation_or_test_for_controller = 0
uses_future_outcome_for_features = 0
uses_outcome_at_commit = 0
uses_old_table_for_official = 0
payload_norm_bucket_used_as_selector = 0
diagnostic_promoted_to_official = 0
fake/proxy/cpu_offload = 0/0/0
```

Failure table：

```text
route = R2-hidden_multiaxis_pocket_explains_rank
F0_boundary_reproduction_failed = 0
F1_scope_mismatch = 0
F2_hidden_multiaxis_pocket = 1
F3_existing_action_rank_group_unstable = 0
F6_apgh_generated_frontier_fail = 0
F8_system_not_official = 1
primary_blocker = multi_axis_group_pocket_explains_rank
```

说明：failure taxonomy 的主路由优先级停在 R2，所以 `F6_apgh_generated_frontier_fail = 0`；但 route JSON 仍记录 secondary blocker = `apgh_generated_frontier_fail`，P12 也明确 `apgh_weak_pass = 0`。

## 15. Hash

| artifact | SHA256 |
|---|---|
| plan | `5fa7393241cbd7f9e41928c3191a0e588d7677bd485c12471bc344c0acb73e21` |
| runner | `da560163845ad7f438f5177ec4ce7b4ca3b824eec6b66055d17b913bdcd201c6` |
| run manifest | `aa3ddaeca9a8421ef23d6f11f030120660ec6ce7497e5c693e41d54a5238b367` |
| route | `16e65944691ecd534d2d7b4dd6defb7959fa9154757c1d44ed1cfe948e1d769e` |
| P0 boundary | `e479a3960d57246f0ece3ddf8ded2a2ae91a8d9ef32d0083cbc4ca30899a2eee` |
| P1 accepted scope | `d5163de5d97e4bf1b015499c1541de8c0534677320caaeeff723c5a837a87c99` |
| P2 hidden pocket | `9789b9ec4a7f198c946ce082d175a321866f857b15beff52f88971b9ced4e242` |
| P3 target support | `11e03aff32040d2890250346c4ccf4779d810e3297d806b8b7a46551e292c1ae` |
| P4 feature | `d5cf8ee82fb5c96a76d3258f570675f0ed56edf3b7077fe25ae475d3216beb4f` |
| P5 ranker | `6fd17d10e706c7014f29c6dc0c6cc17e26888560e1fa9b2d2f9fbe9f2dad7dde` |
| P6 certificate | `192d8ac1113709c2aed4c33ee95ba5c5ce500004f24e810a7edd32699f905db9` |
| P7 controller | `fe110575f204485ac43ef03badc93857ca8ff2f632d3cd2043eb2a7af7bd7928` |
| P8 runtime | `6f0b2910363c8d274987285998433cf624a6ec8db708e84e1307bdd4e93e10c1` |
| P9 damage | `69ec949cfa4bb4a29026ccb2c868fd7f72859bd6017342f786272c52bf91828e` |
| P10 APGH implementation | `8b76a60c6e4c010f9e08960201fb352c6e7525af2e3e4b65b69ac68bd0837f58` |
| P11 APGH smoke | `d75769469c98e3ed4aa6322e6db8600e0ca89be6598f96906312cfa5e502cfac` |
| P12 APGH outcome | `f0f1318ca0ccccecb9af769f9e69cdb14976e226675a5fea1f5fe36405f5c2ed` |
| P13 APGH controller | `e3098e1091188915dfe6093cd66d82811edd2dcd20b3d6aef4f87d2483aa4509` |
| P15 paired boundary | `20fbfc8f0ecba5d2d7b73001fff8886f624637a61ba25c1aa3ea85a842b7af07` |
| P16 short/full | `bca971338eb43082a12acb3ee3580f7b30d8bd398c3dbb0fd4e20e3adfed2a7c` |
| Base-Acc Sentinel | `a36a9cf09e90a129e22ae6969eb63982cd756b53e57d075eede033dce25044cd` |
| no-fake audit | `ecf0a9e875bf3252dd2d21d3bdf37956751b0faed71e273c28dbb16d481210bc` |
| contract audit | `b5a461021fa9aa2d0544f0e7ec6c26bedc9c66abb8ed8f654a4303eeb0754264` |
| failure taxonomy | `2bc528a46bd443c175e0b317326650d0c6c3580d899c7359ba5089c92233adce` |

## 16. 最终分析结论

v9.6.3 的真实推进是：

```text
v9.6.2:
  payload pocket route 被正式停止为 confounded diagnostic；
  legal rank 仍有强 TopK signal，但 group instability 未解；
  APGF generated frontier 失败。

v9.6.3:
  先审计 accepted-region scope，确认没有 duplicate / universe / row-action 口径错误；
  进一步用 multi-axis pocket search 解释 accepted region；
  证明 v9.6.2 的强 accepted region 主要是 group pocket，而不是可 official 的 stable controller；
  group-balanced target、feature、ranker、certificate 都无法同时满足 support、value、risk 与 group stability；
  APGH1-APGH8 工程链路和 12288 行 branch-horizon replay 完整闭合；
  但 APGH generated actions 仍 value-negative / high-longrisk，没有形成 generated frontier。
```

机制判断：

1. H0 成立：v9.6.2 boundary 被复现，没有跳过 payload-pocket-stopped / APGF failure。
2. H1 成立：accepted-region scope audit 证明 P6 强 precision 不是 duplicate 或 universe mismatch。
3. H2 成立于诊断层：multi-axis pocket 能解释 rank accepted region，best pocket drop explained fraction = `1.0`。
4. H3 未成立：pocket 不能 official，因为 `payload_norm_bucket` 禁止作为 selector，且其它 pocket 仍表现为 group concentration。
5. H4 未成立：T2/T3 target value/risk 干净，但 positive group coverage 与 max group share 不过。
6. H5 未成立：legal feature / ranker / certificate 没有形成 deployable accepted region；best certificate V LCB 为负且 max group share = `1.0`。
7. H6 未打开：existing-action controller 与 runtime 均 not_run。
8. H7 成立于诊断层：generated damage 主要是 value direction lost 与 longrisk created。
9. H8/H9 成立于 implementation 层：APGH payload/certificate/action apply 与 branch-horizon materializer 完整闭合。
10. H10 未成立：APGH 没有生成 value-positive / horizon-safe frontier；best APGH7 的 V LCB 为负且 longrisk UCB 高。
11. H11-H13 未打开：没有 controller/runtime/APGH official pass，就不能打开 leaveout、paired replay 或 short/full training。
12. Base-Acc Sentinel 继续健康，但没有用于 controller，也不是 functional success。

最终一句话：

> v9.6.3 真实执行后停在 `R2-hidden_multiaxis_pocket_explains_rank`：v9.6.2 的强 accepted region 不是 scope 错误，而是被 multi-axis group pocket 解释；existing-action rank/certificate 仍不能 official，APGH 虽完整落盘却继续生成 value-negative / high-longrisk 动作，因此 strict PureKAN functional 仍未成功。
