# DG-KAN v9.6.0 Group-Pocket Deconfounded Rank / Memory-Safe LongRisk Primitive 实验复盘

> 本复盘记录 `DG-KAN_v9.6.0_GroupPocketDeconfoundedRank_MemorySafeLongRiskPrimitive_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 group-pocket diagnostic、two-score rank diagnostic、group-stable pairwise rank diagnostic、rank-safe certificate diagnostic、APGD implementation/smoke、runtime boundary 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R1-GroupPocketMechanismUnresolved
base_candidate = LQ-t2-h256
success_v9600_strict_purekan_functional = False
success_v9600_full_functional = False
success_v9600_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9600_group_pocket_deconfounded_rank_memory_safe_longrisk_primitive_first_20260515T100000Z/
```

核心结论：

1. P0 复现 v9.5.9 boundary：source route = `R1-GroupSpecificRankPocket`，group-specific rank pocket = `1`，existing controller/system pass = `0 / 0`，no-fake/no-proxy = `1 / 1`。
2. P1 group pocket 仍未完成因果归因：dominant axis = `payload_norm_bucket`，dominant group = `payload0`，max drop = `0.8505747126436781`，但 payload_norm bucket 解释 fraction = `0.18181818181818182`，failure attribution fraction = `0.03689567430025445`，`p1_pass = 0`。
3. P1 matched-pair lift 没有形成可用配对证据：matched pair count = `0`，matched_pair_lift_pass = `0`。
4. P2 group-balanced target density 有 weak target：best = `ValuePositiveNoLongRisk`，count = `97`，coverage = `0.03372739916550765`，positive group coverage = `1.0`；但 official density pass = `0`。
5. P3 feature invariance 未过：best = `ControlTransferImprovement`，within-group AUC mean = `0.9842536058137868`，TopK64 group-balanced precision = `0.4603174603174603`，longrisk UCB = `0.5176365555683604`。
6. P4 two-score ranker 未过：best = `VR4-PairwiseValue-RiskVeto`，TopK87 GradeAB precision = `0.8390804597701149`，V LCB = `0.13479400136362346`，longrisk UCB = `0.0`，但 leaveout drop = `0.3448275862068965`，max group share = `1.0`。
7. P5 group-stable pairwise ranker 仍未过：best = `GIR9-PairwiseWithinGroupRanker`，TopK87 precision = `0.8505747126436781`，V LCB = `0.15630165181090255`，longrisk UCB = `0.03389313880385762`，LDO/LSO drop = `0.3563218390804597 / 0.3563218390804597`。
8. P6 rank-safe certificate v8 未过：best = `RC7-TopKFixedCountGroupBalanced`，heldout accepted = `87`，coverage = `0.10046189376443418`，GradeAB precision = `0.28735632183908044`，V LCB = `0.03544770490685384`，longrisk UCB = `0.369780988566077`。
9. P7/P8 existing-action controller/runtime 均 gate-blocked：reason = `P5_or_P6_group_stable_rank_certificate_failed` / `P7_controller_not_selected`。
10. P9 APGA OOD/damage decomposition 过 diagnostic：dominant mode = `D7-longrisk-veto-ineffective`，longrisk created rate = `0.66796875`，Damage V LCB = `-0.24198602000541142`。
11. P10 memory/cover/offdiag blocker 仍被确认：best subcomponent = `population_risk_offdiag_fail`，P(longrisk | subfail) = `0.873054114158636`，P(subfail | longrisk) = `1.0`，memory-safe value-positive density = `0.021362229102167184`。
12. P11 APGD1-APGD8 implementation 过：generated actions = `512`，payload/certificate hash missing = `0 / 0`，action apply L∞ max = `0.0`。
13. P12 APGD branch-horizon smoke 完整落盘：expected/actual rows = `12288 / 12288`，unresolved exception = `0`，quality audit pass = `1`，rows/sec = `15.058470711652918`。
14. P13 APGD outcome 未过：best = `APGD2-MemorySafeResidualBlend`，GradeAB precision = `0.0625`，V LCB = `-0.875224386072408`，h240 longrisk UCB = `0.8560881119635937`，longrisk created rate = `0.75`。
15. P15 leaveout/paired replay 未打开：reason = `controller_runtime_or_system_not_official`。
16. P16 Base-Acc Sentinel 复用前序真实 sentinel：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
17. 当前 primary blocker：`group_pocket_causal_attribution_incomplete`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9600_group_pocket_deconfounded_rank_memory_safe_longrisk_primitive.py` | v9.6.0 runner；读取 v9.5.9/v9.5.8/v9.5.7/v9.5.6/v9.5.5/v9.3.3 artifacts，执行 group-pocket causal autopsy、group-balanced target density、two-score ranker、pairwise ranker、certificate、APGA OOD decomposition、APGD primitive、APGD branch-horizon smoke 与 system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9600_group_pocket_deconfounded_rank_memory_safe_longrisk_primitive.py
```

正式运行：

```bash
python experiments/run_v9600_group_pocket_deconfounded_rank_memory_safe_longrisk_primitive.py \
  --out-dir results/real_rerun_20260506/v9600_group_pocket_deconfounded_rank_memory_safe_longrisk_primitive_first_20260515T100000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --apgd-actions-per-primitive 64
```

运行结果：

```json
{
  "apgd_generated_actions": 512,
  "apgd_rows": 12288,
  "best_apgd_V_integrated_LCB": -0.875224386072408,
  "best_apgd_primitive": "APGD2-MemorySafeResidualBlend",
  "best_pairwise_LDO_drop_max": 0.3563218390804597,
  "best_pairwise_ranker": "GIR9-PairwiseWithinGroupRanker",
  "out_dir": "results/real_rerun_20260506/v9600_group_pocket_deconfounded_rank_memory_safe_longrisk_primitive_first_20260515T100000Z",
  "primary_blocker": "group_pocket_causal_attribution_incomplete",
  "route": "R1-GroupPocketMechanismUnresolved",
  "system_legal_controller_pass": 0
}
```

## 2. Route

`route_decision_v9600.json` 摘要：

```json
{
  "route": "R1-GroupPocketMechanismUnresolved",
  "source_route_v9590": "R1-GroupSpecificRankPocket",
  "p0_pass": 1,
  "p1_pass": 0,
  "group_pocket_mechanism_resolved": 0,
  "payload_norm_bucket_drop_explained_fraction": 0.18181818181818182,
  "payload0_removal_precision_drop": 0.8505747126436781,
  "group_balanced_target_density_weak_pass": 1,
  "group_balanced_target_density_pass": 0,
  "feature_invariant_pass": 0,
  "two_score_ranker_pass": 0,
  "group_stable_pairwise_ranker_weak_pass": 0,
  "best_pairwise_ranker_id": "GIR9-PairwiseWithinGroupRanker",
  "best_pairwise_TopK87_precision": 0.8505747126436781,
  "best_pairwise_LDO_drop_max": 0.3563218390804597,
  "rank_safe_certificate_pass": 0,
  "source_controller_pass": 0,
  "selected_runtime_pass": 0,
  "apga_ood_damage_autopsy_pass": 1,
  "dominant_apga_damage_mode": "D7-longrisk-veto-ineffective",
  "memory_cover_blocker_confirmed": 1,
  "apgd_implementation_pass": 1,
  "apgd_branch_horizon_pass": 1,
  "apgd_branch_horizon_rows_actual": 12288,
  "apgd_unresolved_exception_count": 0,
  "apgd_official_candidate_pass": 0,
  "best_apgd_primitive_id": "APGD2-MemorySafeResidualBlend",
  "best_apgd_GradeAB_precision": 0.0625,
  "best_apgd_V_integrated_LCB": -0.875224386072408,
  "best_apgd_h240_longrisk_UCB": 0.8560881119635937,
  "system_legal_controller_pass": 0,
  "primary_blocker": "group_pocket_causal_attribution_incomplete"
}
```

判断：v9.6.0 的 route 优先停在 P1。虽然 P4/P5 继续显示强 TopK diagnostic，且 APGD 工程链路完整闭合，但 group pocket 的因果归因仍不充分，existing-action controller 和 generated frontier 都不能 official。

## 3. P0 v9.5.9 boundary

Artifact：

```text
p0_v9590_boundary_reproduction.csv
```

Summary：

```text
source_route_v9590 = R1-GroupSpecificRankPocket
group_specific_rank_pocket = 1
dominant_drop_axis = payload_norm_bucket
dominant_drop_group = payload0
max_topk_group_share = 1.0
max_drop = 0.8505747126436781
group_balanced_target_density_weak/pass = 1 / 0
rank_safe_certificate_pass = 0
existing_action_controller_pass = 0
apga_ood_damage_autopsy_pass = 1
memory_cover_blocker_confirmed = 1
system_legal_controller_pass = 0
p0_pass = 1
```

判断：P0 pass。v9.6.0 没有跳过 v9.5.9 的 group-specific rank pocket boundary。

## 4. P1 group pocket causal autopsy

Artifacts：

```text
p1_group_pocket_causal_autopsy.csv
p1_matched_pair_lift_trace.csv
```

Summary：

```text
ranker_count = 4
group_row_count = 5522
dominant_drop_axis = payload_norm_bucket
dominant_drop_group = payload0
dominant_drop_ranker = R2-ValueRankLongRiskVeto
max_drop = 0.8505747126436781
identified_drop_group_fraction = 0.03689567430025445
failure_mode_assigned_fraction = 0.03689567430025445
payload_norm_bucket_drop_explained_fraction = 0.18181818181818182
payload0_removal_precision_drop = 0.8505747126436781
within_payload0_precision = 0.8505747126436781
cross_payload_precision = 0.0
H1_payload_pocket_confirmed = 0
p1_pass = 0
```

Matched pair trace：

```text
matched_pair_count = 0
mean_score_lift = 0.0
mean_gradeab_lift = 0.0
mean_V_lift = 0.0
sign_test_p_value_proxy = 1.0
matched_pair_lift_pass = 0
```

判断：P1 是本轮 terminal blocker。`payload0` pocket 的 drop 很强，但归因覆盖和配对证据不足，不能声明 group pocket mechanism 已闭合。

## 5. P2/P3 target 与 feature invariance

P2 artifact：

```text
p2_group_balanced_target_density_map.csv
```

P2 summary：

```text
target_candidate_count = 10
weak_density_target_count = 2
official_density_target_count = 0
best_target_id = ValuePositiveNoLongRisk
best_global_count = 97
best_global_coverage = 0.03372739916550765
best_group_axis = dataset
best_group_min_density = 0.029197080291970802
best_positive_group_coverage = 1.0
group_balanced_target_density_weak_pass = 1
group_balanced_target_density_pass = 0
```

P3 artifact：

```text
p3_feature_invariance_deconfounding_v2.csv
```

P3 summary：

```text
feature_count = 12
best_feature_id = ControlTransferImprovement
best_AUC_within_group_mean = 0.9842536058137868
best_AUC_within_group_min = 0.9130434782608695
best_TopK64_group_balanced_precision = 0.4603174603174603
best_longrisk_UCB_group_balanced = 0.5176365555683604
feature_invariant_pass = 0
```

判断：group-balanced target 有弱信号，但仍没有 official density。`ControlTransferImprovement` 的 within-group AUC 很强，但 group-balanced TopK longrisk 太高。

## 6. P4/P5 rankers

P4 artifact：

```text
p4_two_score_ranker_value_risk_veto.csv
```

P4 summary：

```text
ranker_count = 6
best_ranker_id = VR4-PairwiseValue-RiskVeto
best_TopK87_GradeAB_precision = 0.8390804597701149
best_TopK87_V_integrated_LCB = 0.13479400136362346
best_TopK87_h240_longrisk_UCB = 0.0
best_leaveout_drop_max = 0.3448275862068965
best_max_group_share = 1.0
two_score_ranker_pass = 0
```

P5 artifact：

```text
p5_group_stable_pairwise_ranker_v2.csv
```

P5 summary：

```text
ranker_count = 10
best_ranker_id = GIR9-PairwiseWithinGroupRanker
best_TopK87_precision = 0.8505747126436781
best_TopK87_V_integrated_LCB = 0.15630165181090255
best_TopK87_h240_longrisk_UCB = 0.03389313880385762
best_LDO_drop_max = 0.3563218390804597
best_LSO_drop_max = 0.3563218390804597
best_max_group_share = 0.4367816091954023
group_stable_pairwise_ranker_weak_pass = 0
legal_rank_group_stable_pass = 0
```

判断：P4/P5 继续证明 rank signal 不是空的，但无法通过 group-stability gate。P4 的 max group share 仍为 `1.0`，P5 的 LDO/LSO drop 仍为 `0.3563`。

## 7. P6-P8 certificate / controller / runtime

P6 artifact：

```text
p6_rank_safe_certificate_v8.csv
```

P6 summary：

```text
certificate_count = 7
best_certificate_id = RC7-TopKFixedCountGroupBalanced
best_ranker_id = GIR9-PairwiseWithinGroupRanker
best_heldout_accepted_count = 87
best_coverage_heldout = 0.10046189376443418
best_GradeAB_precision_heldout = 0.28735632183908044
best_V_integrated_LCB_heldout = 0.03544770490685384
best_h240_longrisk_UCB_heldout = 0.369780988566077
rank_safe_certificate_pass = 0
```

P7/P8：

```text
p7_existing_action_minimal_controller.csv = not_run
reason = P5_or_P6_group_stable_rank_certificate_failed
source_controller_pass = 0

p8_selected_existing_action_runtime_preflight.csv = not_run
reason = P7_controller_not_selected
selected_runtime_pass = 0
official_runtime_pass = 0
```

判断：rank-safe certificate 的 heldout accepted region 质量不足，existing-action controller 和 runtime 必须关闭。

## 8. P9/P10 APGA OOD 与 blocker

P9 artifact：

```text
p9_apga_ood_damage_decomposition_v2.csv
```

P9 summary：

```text
apga_action_count = 512
dominant_damage_mode = D7-longrisk-veto-ineffective
longrisk_created_rate = 0.66796875
Damage_V_integrated_LCB = -0.24198602000541142
payload_norm_MMD_proxy = 98.1330574962105
payload_norm_KS_proxy = 1.0
longrisk_veto_false_negative_rate = 0.0
apga_ood_damage_autopsy_pass = 1
```

P10 artifact：

```text
p10_memory_cover_offdiag_blocker_v4.csv
```

P10 summary：

```text
subcomponent_count = 7
best_subcomponent_id = population_risk_offdiag_fail
best_P_longrisk_given_subfail = 0.873054114158636
best_P_subfail_given_longrisk = 1.0
best_memory_safe_value_positive_density = 0.021362229102167184
memory_cover_blocker_v3_pass = 0
memory_cover_blocker_confirmed = 1
```

判断：APGA OOD/destructive 机制仍指向 longrisk veto ineffective 与 payload OOD；memory/cover/offdiag blocker 有诊断证据，但不是 official pass。

## 9. P11-P13 APGD primitive

P11 artifact：

```text
p11_apgd_memory_safe_longrisk_primitive.csv
```

P11 summary：

```text
primitive_count = 8
generated_action_count = 512
generated_action_count_expected = 512
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_linf_max = 0.0
negative_control_generated = 1
apgd_implementation_pass = 1
```

P12 artifacts：

```text
p12_apgd_branch_horizon_smoke_outcome.csv
```

P12 summary：

```text
branch_horizon_rows_expected/actual = 12288 / 12288
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
secondary_delta_completion_rate = 1
rows_per_sec = 15.058470711652918
wallclock_sec = 816.0191187602468
unresolved_exception_count = 0
duplicate_row_count = 0
quality_audit_pass = 1
apgd_branch_horizon_pass = 1
```

P13 artifact：

```text
p13_apgd_outcome_geometry_ood_pass.csv
```

P13 summary：

```text
best_primitive_id = APGD2-MemorySafeResidualBlend
best_GradeB_precision = 0.0625
best_GradeAB_precision = 0.0625
best_V_integrated_LCB = -0.875224386072408
best_h240_longrisk_UCB = 0.8560881119635937
best_bad_UCB = 0.2831265318504755
best_null_UCB = 0.09866091513007542
best_memory_fail_UCB = 0.0
best_new_positive_created_rate = 0.0
best_longrisk_created_rate = 0.75
best_source_to_generated_damage_LCB = -1.11350191294148
apgd_weak_pass = 0
apgd_official_candidate_pass = 0
```

Per primitive：

| primitive | GradeAB precision | V_integrated LCB | h240 longrisk UCB | pass |
|---|---:|---:|---:|---:|
| `APGD1-AP0GoodAnchorProjection` | `0.015625` | `-0.8832596050687327` | `0.9205565823304426` | `0` |
| `APGD2-MemorySafeResidualBlend` | `0.0625` | `-0.875224386072408` | `0.8560881119635937` | `0` |
| `APGD3-PopRiskOffdiagPositiveGate` | `0.015625` | `-0.8465523579160792` | `0.8825326673766913` | `0` |
| `APGD4-CoverEntropyPreservingUpdate` | `0.0` | `-0.9652602742070555` | `0.8825326673766913` | `0` |
| `APGD5-AdamWCompatibleLowNormTrustRegion` | `0.0` | `-0.7609413813366084` | `0.8694088506054019` | `0` |
| `APGD6-ValueRiskTwoScoreProjectedUpdate` | `0.015625` | `-0.8441825544806216` | `0.9081265318504754` | `0` |
| `APGD7-OldFamilyConstrainedEdgeUpdate` | `0.0` | `-0.8988629891113775` | `0.9327075862332016` | `0` |
| `APGD8-NegativeControlShuffledAnchor` | `0.015625` | `-0.8997045952083764` | `0.8694088506054019` | `0` |

判断：APGD 工程链路完整闭合，但 science gate 失败。APGD 没有把 memory-safe / longrisk-veto mechanism 转成 generated-action frontier。

## 10. P15/P16 boundary

P15：

```text
p15_leaveout_paired_replay_boundary.csv = not_run
reason = controller_runtime_or_system_not_official
official_leaveout_pass = 0
official_paired_replay_pass = 0
```

P16 Base-Acc Sentinel：

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

判断：没有 controller/runtime，就不能打开 leaveout、paired replay 或 short/full training。Base-Acc Sentinel 继续健康，但没有用于 controller。

## 11. No-fake audit

```text
rows_checked = 19072
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
v9590_boundary_pass = 1
group_pocket_autopsy_pass = 0
group_balanced_target_density_pass = 0
feature_invariant_pass = 0
two_score_ranker_pass = 0
group_stable_pairwise_ranker_pass = 0
rank_safe_certificate_pass = 0
source_controller/selected_runtime/system = 0/0/0
apga_ood_damage_autopsy_pass = 1
memory_cover_blocker_pass = 0
apgd_implementation_pass = 1
apgd_branch_horizon_pass = 1
apgd_official_candidate_pass = 0
base_acc_sentinel_pass = 1
base_acc_used_for_controller = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_selector/controller = 0/0
uses_validation_or_test_for_controller = 0
uses_future_outcome_for_features = 0
uses_outcome_at_commit = 0
uses_old_table_for_official = 0
diagnostic_promoted_to_official = 0
fake/proxy/cpu_offload = 0/0/0
```

Failure table：

```text
route = R1-GroupPocketMechanismUnresolved
F0_boundary_reproduction_failed = 0
F1_group_pocket_mechanism_unresolved = 1
F2_group_balanced_target_absent = 0
F3_legal_rank_group_stable_fail = 0
F4_rank_certificate_accepted_region_fail = 0
F5_existing_action_runtime_fail = 0
F6_apga_ood_mechanism_unresolved = 0
F7_apgd_generated_frontier_fail = 0
F8_system_not_official = 1
F9_base_acc_catastrophic = 0
primary_blocker = group_pocket_causal_attribution_incomplete
```

## 12. Hash

| artifact | SHA256 |
|---|---|
| plan | `2cd9da9d9b207dc6287fd1e1f6c79c3ea2028a0bb594c07dfc88e074352ffab5` |
| runner | `e01fe90ee0de26df8f07f2442965986d9dea3c8da23ff768348a187c93718ff8` |
| run manifest | `1623a832b6015dea9da8b68187b95ca29cf591aae1490f8665e01b9c3ce73d69` |
| P0 boundary | `dbabb375fda02b162dcfe8ecf17672dbdcb8817f9a613089afd05899bf423cc9` |
| P1 autopsy | `d6d33b6ef995fdba633a3ff710dbf7dee1f3ff6f9aabe056d5b8a933bd5c44f0` |
| P1 matched pair | `8217314910cbb7f71e1da9c05e8dd8945a4761330aeeb08d91d238088680f635` |
| P2 target density | `2f60d7b996333bb055c10cd0589ab724068326551d6379ef1396e129867c5034` |
| P3 feature invariance | `f8e966ff8994003538bc65cf899498f812d1091cd1a3aa0d094327c2bce85a16` |
| P4 two-score rank | `341d9c4f8f28c3e329b926831f67be18d3abc9f6f9d4b1d01e51013fcfcc3107` |
| P5 pairwise rank | `fbeae00e3c5c2fe7161ac62bb687674950f7fb6d1f19c4b74aa4d3c66aa97358` |
| P6 certificate | `5efc2ac25ffa7e319a6cf4a164cfc366bed581287c100d2049fdbeeb32793f6e` |
| P7 controller | `244a304d1c2dfc4fb3241002e50db2aaecfb8d00ba213c764eff7abe872d3754` |
| P8 runtime | `23812bea9ec6e3f39571088cc6a94b40b85f03753cde2f6b7df9bbde331d986f` |
| P9 APGA OOD | `755a837c60601e1286101e63a8defe09ed21bdb4c1fafa5f608262e0b38054d0` |
| P10 blocker | `921429c71b1d1e2f2d6e470541118d0acf1d518ea03b399b6e89dae922f414ee` |
| P11 APGD implementation | `b9f50fd46407c20170524d73def1ab88b27d348a8de14899cc591aa64394d45c` |
| P12 APGD smoke | `7dfb42671f3468f5e1600f498a4ea5468edd55124ef5201d43379036cf803687` |
| P13 APGD outcome | `90a84b10dd646c1888c92f47a2a9f28b823d1932f67a625e14bfec55fa74fb35` |
| route | `9d7d4f0396adfe0554d20f5301e472cbd1726703e0444ad46430dc1184956141` |
| P15 boundary | `91d74fd21b39a8214d61cd7d089705434c0f93012977fb7198f51fcd5fdca083` |
| P16 Base-Acc Sentinel | `56b1591cb9f2f496ba7cc7663c2b839b0864f4944b5c13c392c2e705133a7250` |
| no-fake audit | `cccb8baa23d1feea2a4e8d8d3aed418f2c970d9b4aeab31a758fe4be81638ea8` |
| contract audit | `402b069b234b131f015041d7743b93bad723e36727d4fd897f10d737c3146035` |
| provenance audit | `40f57bc6116e234218ffcb1009c7d38cbe4ec667446d771522c70028db3f346b` |
| failure taxonomy | `14813eb185336bcb6efc2a41eb8ff5c519e40009312c3791c62405a1027d36fb` |

## 13. 最终分析结论

v9.6.0 的真实推进是：

```text
v9.5.9:
  legal rank pocket 被定位在 payload_norm_bucket / payload0；
  group-invariant ranker 降低 drop 但仍未 official；
  APGA OOD destructive 指向 longrisk veto ineffective。

v9.6.0:
  对 group pocket 做 causal autopsy 和 matched-pair trace；
  发现 payload0 drop 强，但归因覆盖与配对证据不足；
  group-balanced target 有 weak density，但没有 official density；
  two-score ranker 和 pairwise ranker 都有强 TopK diagnostic，但仍 group unstable；
  APGA OOD / memory-cover-offdiag blocker 继续成立；
  APGD1-APGD8 工程链路与 branch-horizon materializer 完整闭合；
  但 APGD generated actions 仍 value-negative / high-longrisk，不能成为 generated frontier。
```

机制判断：

1. H1 未闭合：payload0 pocket 很强，但 payload_norm bucket 解释 fraction 只有 `0.1818`，matched pair count 为 `0`，不能宣布 group pocket mechanism resolved。
2. H2 部分成立于 diagnostic 层：P4/P5 的 TopK87 precision 可达 `0.8391-0.8506`，且 V LCB > 0、longrisk UCB 低；但 drop 和 group share 仍不满足 official gate。
3. H3 成立于诊断层：APGA OOD damage 继续由 longrisk veto ineffective 主导，memory/cover/offdiag blocker 仍被确认。
4. H4/H5 成立于 implementation 层：APGD payload/certificate/action apply 与真实 branch-horizon replay 均闭合。
5. H6 未成立：APGD outcome 没有生成 value-positive / horizon-safe / GradeAB-positive frontier，best APGD2 的 V LCB 为负且 h240 longrisk UCB 很高。
6. P7/P8/P15 未打开：没有 source controller，就不能打开 runtime、official leaveout、paired replay 或 short/full training。
7. Base-Acc Sentinel 继续健康，但没有用于 controller，也不是 functional success。

最终一句话：

> v9.6.0 真实执行后停在 `R1-GroupPocketMechanismUnresolved`：group pocket 的 payload0 drop 仍很强，但因果归因和 matched-pair 证据不足；ranker/certificate 仍不能 official，APGD 虽完整落盘却没有生成可用 frontier，因此 strict PureKAN functional 仍未成功。
