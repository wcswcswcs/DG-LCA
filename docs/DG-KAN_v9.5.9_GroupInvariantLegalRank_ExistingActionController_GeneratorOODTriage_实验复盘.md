# DG-KAN v9.5.9 Group-Invariant Legal Rank / Existing-Action Controller / Generator OOD Triage 实验复盘

> 本复盘记录 `DG-KAN_v9.5.9_GroupInvariantLegalRank_ExistingActionController_GeneratorOODTriage_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 group-specific rank pocket、group-invariant rank diagnostic、rank-safe certificate diagnostic、APGA OOD autopsy、APGI/APGC boundary、runtime boundary 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R1-GroupSpecificRankPocket
base_candidate = LQ-t2-h256
success_v9590_strict_purekan_functional = False
success_v9590_full_functional = False
success_v9590_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9590_group_invariant_legal_rank_existing_action_controller_first_20260515T090000Z/
```

核心结论：

1. P0 复现 v9.5.8 boundary：source route = `R1-LegalRankStillGroupUnstable`，field legality ledger pass = `1`，legacy rank leakage explained = `1`，APGA implementation / branch-horizon pass = `1 / 1`，APGA weak pass = `0`。
2. P1 group-drop autopsy 确认 rank signal 仍是 group-specific pocket：dominant axis = `payload_norm_bucket`，dominant group = `payload0`，max topK group share = `1.0`，max drop = `0.8505747126436781`。
3. P1 failure mode 主因是 `GDF4-topk-group-concentration`，count = `22`，fraction = `0.7586206896551724`；但 P1 全量归因覆盖率不足，`p1_pass = 0`。
4. P2 group-balanced target density 有 weak signal：best target = `ValuePositiveNoLongRisk`，global count = `97`，coverage = `0.03372739916550765`；但 official density pass = `0`。
5. P3 feature invariance 未过：best feature = `ControlTransferImprovement`，within-group AUC mean = `0.9842536058137868`，TopK64 group-balanced precision = `0.4603174603174603`，但 longrisk UCB = `0.5176365555683604`。
6. P4 group-invariant ranker 仍未过 official gate：best = `GIR9-PairwiseWithinGroupRanker`，TopK87 GradeAB precision = `0.8505747126436781`，V_integrated LCB = `0.15630165181090255`，h240 longrisk UCB = `0.03389313880385762`，但 LDO/LSO drop = `0.3563218390804597 / 0.3563218390804597`。
7. P5 rank-safe certificate v7 未过：best = `RC7-TopKFixedCountGroupBalanced`，heldout accepted = `87`，coverage = `0.10046189376443418`，GradeAB precision = `0.28735632183908044`，V_integrated LCB = `0.03544770490685384`，h240 longrisk UCB = `0.369780988566077`。
8. P6 existing-action controller 未打开：reason = `P4_or_P5_group_invariant_rank_certificate_failed`，source controller pass = `0`。
9. P7 APGA OOD/damage autopsy 完成：dominant damage mode = `D7-longrisk-veto-ineffective`，longrisk created rate = `0.66796875`，Damage V_integrated LCB = `-0.24198602000541142`。
10. P8 memory/cover blocker v3 未过 official gate，但 blocker 仍被确认：best subcomponent = `population_risk_offdiag_fail`，P(longrisk | subfail) = `0.873054114158636`，P(subfail | longrisk) = `1.0`，memory-safe value-positive density = `0.021362229102167184`。
11. P9/P10 APGI/APGC 均 gate-blocked：没有 existing-action controller，且 APGA OOD destructive triage open。
12. P11-P15 均 gate-blocked：没有 selected runtime、system controller、leaveout、official paired replay 或 short/full training。
13. P16 Base-Acc Sentinel 复用前序真实 sentinel：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
14. 当前 primary blocker：`group_specific_rank_pocket_group_invariant_rank_failed`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9590_group_invariant_legal_rank_existing_action_controller.py` | v9.5.9 runner；读取 v9.5.8/v9.5.7/v9.5.6/v9.5.5 artifacts，执行 group-drop autopsy、group-balanced target density、feature invariance、group-invariant ranker、rank-safe certificate、existing-action controller boundary、APGA OOD triage、memory/cover blocker 与 system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9590_group_invariant_legal_rank_existing_action_controller.py
```

正式运行：

```bash
python experiments/run_v9590_group_invariant_legal_rank_existing_action_controller.py \
  --out-dir results/real_rerun_20260506/v9590_group_invariant_legal_rank_existing_action_controller_first_20260515T090000Z \
  --fresh --seed 1314
```

运行结果：

```json
{
  "apga_damage_mode": "D7-longrisk-veto-ineffective",
  "best_ranker": "GIR9-PairwiseWithinGroupRanker",
  "best_ranker_LDO_drop_max": 0.3563218390804597,
  "best_ranker_TopK87_precision": 0.8505747126436781,
  "dominant_drop_axis": "payload_norm_bucket",
  "out_dir": "results/real_rerun_20260506/v9590_group_invariant_legal_rank_existing_action_controller_first_20260515T090000Z",
  "route": "R1-GroupSpecificRankPocket",
  "system_legal_controller_pass": 0
}
```

说明：本轮没有 CPU offload，没有 proxy rows；APGA branch-horizon replay 使用 v9.5.8 已真实落盘的完整 APGA outcome，不重新生成 fake/proxy APGA rows。

## 2. Route

`route_decision_v9590.json` 摘要：

```json
{
  "route": "R1-GroupSpecificRankPocket",
  "source_route_v9580": "R1-LegalRankStillGroupUnstable",
  "p0_pass": 1,
  "p1_pass": 0,
  "group_specific_rank_pocket": 1,
  "dominant_drop_axis": "payload_norm_bucket",
  "dominant_drop_group": "payload0",
  "dominant_miss_reason": "GDF4-topk-group-concentration",
  "max_topk_group_share": 1.0,
  "max_drop": 0.8505747126436781,
  "group_balanced_target_density_weak_pass": 1,
  "group_balanced_target_density_pass": 0,
  "best_density_target_id": "ValuePositiveNoLongRisk",
  "best_density_target_count": 97,
  "best_density_target_coverage": 0.03372739916550765,
  "feature_invariant_pass": 0,
  "best_invariant_feature_id": "ControlTransferImprovement",
  "best_invariant_feature_AUC_mean": 0.9842536058137868,
  "group_invariant_ranker_weak_pass": 0,
  "best_group_invariant_ranker_id": "GIR9-PairwiseWithinGroupRanker",
  "best_group_invariant_TopK87_precision": 0.8505747126436781,
  "best_group_invariant_V_integrated_LCB": 0.15630165181090255,
  "best_group_invariant_h240_longrisk_UCB": 0.03389313880385762,
  "best_group_invariant_LDO_drop_max": 0.3563218390804597,
  "rank_safe_certificate_pass": 0,
  "existing_action_controller_pass": 0,
  "apga_ood_damage_autopsy_pass": 1,
  "apga_generator_ood_destructive": 1,
  "dominant_apga_damage_mode": "D7-longrisk-veto-ineffective",
  "memory_cover_blocker_confirmed": 1,
  "apgi_apgc_implementation_pass": 0,
  "selected_runtime_pass": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "group_specific_rank_pocket_group_invariant_rank_failed"
}
```

判断：v9.5.9 的关键不是 APGI/APGC implementation failure，而是 legal rank 的高 TopK 区域仍是 group-specific pocket。P4 把 LDO/LSO drop 从 v9.5.8 的 `0.8854` 降到 `0.3563`，但仍远未达到 official group-invariant gate；P5 certificate 的 heldout precision/value/risk 也不能支持 controller。

## 3. P0 v9.5.8 boundary

Artifact：

```text
p0_v9580_boundary_reproduction.csv
```

Summary：

```text
source_route_v9580 = R1-LegalRankStillGroupUnstable
field_legality_ledger_pass = 1
legacy_rank_leakage_explained = 1
target_density_pass = 0
legal_feature_weak_pass = 0
group_stable_rank_weak_pass = 0
best_ranker_id = R2-ValueRankLongRiskVeto
best_ranker_TopK87_GradeAB_precision = 0.8505747126436781
best_ranker_LDO/LSO drop = 0.8854166666666666 / 0.8854166666666666
distillation_pass = 1
apga_implementation/branch_horizon/weak pass = 1 / 1 / 0
certificate_official_pass = 0
system_legal_controller_pass = 0
no_fake/no_proxy = 1 / 1
p0_pass = 1
```

判断：P0 pass。v9.5.9 没有跳过 v9.5.8 的 group-unstable rank 与 APGA failure boundary。

## 4. P1 group-drop autopsy

Artifacts：

```text
p1_group_drop_autopsy.csv
p1_group_drop_failure_modes.csv
```

Summary：

```text
ranker_count = 4
group_row_count = 5522
max_topk_group_share = 1.0
dominant_drop_axis = payload_norm_bucket
dominant_drop_group = payload0
dominant_drop_ranker = R2-ValueRankLongRiskVeto
max_drop = 0.8505747126436781
dominant_miss_reason = GDF4-topk-group-concentration
identified_drop_group_fraction = 0.03689567430025445
failure_mode_assigned_fraction = 0.03689567430025445
group_specific_rank_pocket = 1
p1_pass = 0
```

Failure modes：

| mode | count | fraction |
|---|---:|---:|
| `GDF4-topk-group-concentration` | `22` | `0.7586206896551724` |
| `GDF9-dataset-specific-pocket` | `5` | `0.1724137931034483` |
| `GDF2-score-calibration-shift` | `2` | `0.06896551724137931` |

判断：P1 没有达到全量 failure attribution gate，但足以说明核心 rank pocket 是 group-specific；route 因此优先停在 R1，而不是继续把 TopK diagnostic 当成 controller。

## 5. P2 group-balanced target density

Artifact：

```text
p2_group_balanced_target_density.csv
```

Summary：

```text
target_candidate_count = 10
target_group_row_count = 40
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

判断：group-balanced target 不再完全为空，但仍没有 official target density closure。该结果只能说明后续 rank/certificate 有诊断基础，不能打开 existing-action controller。

## 6. P3 feature invariance / deconfounding

Artifact：

```text
p3_feature_invariance_deconfounding.csv
```

Summary：

```text
feature_count = 12
best_feature_id = ControlTransferImprovement
best_AUC_within_group_mean = 0.9842536058137868
best_AUC_within_group_min = 0.9130434782608695
best_TopK64_group_balanced_precision = 0.4603174603174603
best_longrisk_UCB_group_balanced = 0.5176365555683604
feature_invariant_pass = 0
```

判断：ControlTransferImprovement 仍有很强 rank signal，但 group-balanced TopK 的 longrisk 太高，不能作为 legal invariant feature gate。

## 7. P4 group-invariant rankers

Artifact：

```text
p4_group_invariant_rankers.csv
```

Summary：

```text
ranker_count = 10
best_ranker_id = GIR9-PairwiseWithinGroupRanker
best_TopK87_precision = 0.8505747126436781
best_TopK87_V_integrated_LCB = 0.15630165181090255
best_TopK87_h240_longrisk_UCB = 0.03389313880385762
best_LDO_drop_max = 0.3563218390804597
best_LSO_drop_max = 0.3563218390804597
best_max_group_share = 0.4367816091954023
group_invariant_ranker_weak_pass = 0
group_invariant_ranker_strong_pass = 0
ranker_topk_signal_present = 1
```

Representative rankers：

| ranker | TopK87 GradeAB | V LCB | h240 longrisk UCB | LDO/LSO drop | pass |
|---|---:|---:|---:|---:|---:|
| `GIR1-GroupQuantileNormalizedValueRiskRank` | `0.8505747126436781` | `0.13618366993768866` | `0.0` | `0.3563218390804597 / 0.3563218390804597` | `0` |
| `GIR3-ValueRankLongRiskVetoGroupBalanced` | `0.8505747126436781` | `0.13419830825382068` | `0.0` | `0.3563218390804597 / 0.3563218390804597` | `0` |
| `GIR4-MemoryVetoGroupBalanced` | `0.8275862068965517` | `0.17452101582656632` | `0.03389313880385762` | `0.3448275862068965 / 0.3448275862068965` | `0` |
| `GIR8-LegalDistilledStudentNoRedNoGroupName` | `0.7241379310344828` | `0.12807606232790217` | `0.0` | `0.28735632183908044 / 0.28735632183908044` | `0` |
| `GIR9-PairwiseWithinGroupRanker` | `0.8505747126436781` | `0.15630165181090255` | `0.03389313880385762` | `0.3563218390804597 / 0.3563218390804597` | `0` |

判断：P4 是本轮最重要的技术推进。Group normalization / pairwise within-group rank 显著降低 group drop，但 weak pass 仍未打开；不能把 strong TopK diagnostic 写成 group-invariant selector。

## 8. P5 rank-safe certificate v7

Artifact：

```text
p5_rank_safe_certificate_v7.csv
```

Summary：

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

判断：P5 未过。Certificate accepted region 的 heldout coverage 足够，但 precision/value/risk 不够，尤其 longrisk UCB 高；不能打开 existing-action controller。

## 9. P6-P8 controller / APGA OOD / blocker

P6：

```text
p6_existing_action_controller.csv = not_run
reason = P4_or_P5_group_invariant_rank_certificate_failed
existing_action_controller_pass = 0
```

P7 artifact：

```text
p7_apga_ood_damage_autopsy.csv
```

P7 summary：

```text
apga_action_count = 512
APGA_failure_attribution_fraction = 1.0
dominant_damage_mode = D7-longrisk-veto-ineffective
dominant_damage_mode_fraction = 0.66796875
new_positive_created_rate = 0.0078125
positive_lost_rate = 0.154296875
longrisk_created_rate = 0.66796875
Damage_V_integrated_LCB = -0.24198602000541142
apga_ood_damage_autopsy_pass = 1
apga_generator_ood_destructive = 1
```

P7 damage modes：

| mode | count | fraction |
|---|---:|---:|
| `D7-longrisk-veto-ineffective` | `342` | `0.66796875` |
| `D2-generated-OOD-payload` | `143` | `0.279296875` |
| `D1-source-good-destroyed` | `27` | `0.052734375` |

P8 artifact：

```text
p8_memory_cover_blocker_v3.csv
```

P8 summary：

```text
subcomponent_count = 7
best_subcomponent_id = population_risk_offdiag_fail
best_P_longrisk_given_subfail = 0.873054114158636
best_P_subfail_given_longrisk = 1.0
best_memory_safe_value_positive_density = 0.021362229102167184
memory_cover_blocker_v3_pass = 0
memory_cover_blocker_confirmed = 1
```

判断：P7/P8 是后续 generator 方向的诊断边界。APGA 失败主要由 longrisk veto 无效和 OOD payload 造成；memory/cover blocker 仍存在，但 memory-safe value-positive density 低于 official 要求。

## 10. P9-P15 boundary

| artifact | status / reason |
|---|---|
| `p9_apgi_apgc_implementation.csv` | `not_run`, `P6_existing_controller_failed_and_P7_APGA_OOD_destructive_triage_open` |
| `p10_apgi_apgc_branch_horizon_outcome.csv` | `not_run`, `P9_generated_primitive_not_open` |
| `p11_selected_runtime_preflight.csv` | `not_run`, `P6_controller_not_selected` |
| `p12_system_legal_controller_boundary.csv` | `not_run`, `P11_runtime_not_selected` |
| `p13_leaveout_official.csv` | `not_run`, `P12_system_controller_not_official` |
| `p14_official_paired_replay.csv` | `not_run`, `P13_leaveout_not_open` |
| `p15_short_full_training_boundary.csv` | `not_run`, `P14_paired_replay_not_open` |

判断：没有把 group-invariant rank diagnostic、rank-safe certificate diagnostic、APGA OOD autopsy 或 Base-Acc Sentinel 写成 official controller/runtime/system/downstream pass。

## 11. Base-Acc Sentinel

Artifact：

```text
base_acc_sentinel_v9590.csv
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

## 12. No-fake audit

```text
rows_checked = 6253
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
v9580_boundary_pass = 1
group_drop_autopsy_pass = 0
group_balanced_target_density_pass = 0
feature_invariant_pass = 0
group_invariant_ranker_pass = 0
rank_safe_certificate_pass = 0
existing_action_controller_pass = 0
apga_ood_damage_autopsy_pass = 1
memory_cover_blocker_v3_pass = 0
apgi_apgc_implementation/outcome = 0/0
selected_runtime/system = 0/0
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
route = R1-GroupSpecificRankPocket
F0_boundary_regression = 0
F1_group_specific_rank_pocket = 1
F2_group_balanced_target_density_insufficient = 0
F3_group_invariant_rank_controller_pending = 0
F4_existing_controller_runtime_pending = 0
F5_system_paired_replay_pending = 0
F6_apga_generator_ood_destructive = 0
F7_memory_cover_blocker = 0
F8_generated_primitive_still_fails = 0
F9_system_not_official = 1
F10_base_acc_catastrophic = 0
primary_blocker = group_specific_rank_pocket_group_invariant_rank_failed
```

## 13. Hash

| artifact | SHA256 |
|---|---|
| plan | `f6e55b0a4df83428c80150ae54e777be362d9418129a286ce3bc0aa08cc191e5` |
| runner | `75b886bbef74d4f19f68982f118402a8ec4b4977da3446ce8c98201ad580e924` |
| run manifest | `3383c57c1e1fd213bf5b799c0198017bbe7932a0f93dc43d6d1da453c6b9aa9d` |
| route | `910f6b4e34594a50b40ad7b675afd31293e606f5b27a067145ec85fc4f99827f` |
| P0 boundary | `6ba382753b0e9678e236e58712c519d3763df47ca3f5e327eaab62e01e19d505` |
| P1 group drop | `ed024e0085ee4e7272c187be2dd4b71a5634d46e59278e10b108d048ddc76566` |
| P1 failure modes | `c7be24103a1611ce74bee7fd28401c9d16349f2ba7e4682d0a5d691c550b9df4` |
| P2 target density | `3e52ed51a5fdbabfcc73903374e2a322aac2da53916c70d9af724e147295062d` |
| P3 feature invariance | `d214974030502404d0f04bcf486fcd19cf0ff0aac3443b515ced545c902ef9f5` |
| P4 group invariant rank | `d6b5be507cf2c4ec7050133691c1b922e3b746c5343382e631377ae3cca143fa` |
| P5 certificate | `194a8fa0f4f77e9206ff2cdb998441daebf656d4eb51c78bf98f91c789e1f66a` |
| P6 controller | `0b3b39a321624dd2421b9d68e0d5cc3f8032c0c870af53df46208f9ea0c32d16` |
| P7 APGA OOD | `cdda96948848c76c65ab70bce7cbcc5239c918bf39d81042db1a7391a234f10a` |
| P8 memory cover | `d972880b8ca83a6fb301b9570c512cb3bf24e327ecfbeaff4ff0709ac56b736b` |
| P9 APGI/APGC implementation | `0024d574a47a055a8a6e87b79166dc9eae16428ce2c1bd18ed9c88f536cc024b` |
| P10 APGI/APGC outcome | `2a8b4cd7b2710535624474ba3870fb9a7302d9ee72db18fccd6bde5b9acfbe58` |
| Base-Acc Sentinel | `795cec963b3fca422b48a3a191a2e386d311fb48f8242e585b3cc0c1c0758625` |
| contract audit | `f2d5ceb5de0658c90d8f174a819417a906180c4296015aaf3a9f57e2779c511a` |
| no-fake audit | `7fe7d41d6acd5bdae3fd2d31a7b039240b21b1282f411670f8bed64c9669845a` |
| failure taxonomy | `1ab15540fe7588cb2e6011c211ce2b693d3a57ba6f691343d38104afb839d3d4` |

## 14. 最终分析结论

v9.5.9 的真实推进是：

```text
v9.5.8:
  legal rank 已能在 TopK87 找到高 GradeAB、正 value、低 longrisk 的 diagnostic 区域；
  但该区域对 group leaveout 极不稳定；
  APGA generated frontier 与 rank certificate heldout accepted region 均未闭合。

v9.5.9:
  对 group drop 做 autopsy，确认 rank pocket 主要集中在 payload_norm_bucket / payload0；
  扫描 group-balanced target density，找到 weak 但非 official 的 ValuePositiveNoLongRisk target；
  构造 group-invariant rankers，把 LDO/LSO drop 从 0.8854 降到 0.3563；
  但 group-invariant ranker 和 rank-safe certificate 仍未过 official gate；
  APGA OOD damage autopsy 指向 longrisk veto ineffective；
  APGI/APGC 因 controller 未选中与 APGA OOD destructive triage 未打开。
```

机制判断：

1. H0 成立：v9.5.8 boundary 被复现，没有跳过 group-unstable rank / APGA failure。
2. H1 成立于诊断层：legal rank 的失败不是随机，而是 group-specific pocket；max topK group share = `1.0`。
3. H2 部分成立：group-balanced target density 有 weak target，best coverage = `0.0337`，但 official density仍未闭合。
4. H3 未成立：feature invariance 的 AUC 很高，但 group-balanced TopK longrisk UCB = `0.5176`，不能 official。
5. H4 部分成立：group-invariant ranker 显著降低 drop，但 `0.3563` 仍高于 gate。
6. H5 未成立：rank-safe certificate heldout accepted region 的 GradeAB precision 只有 `0.2874`，longrisk UCB = `0.3698`。
7. H6 未打开：existing-action controller 不选中，runtime/system/leaveout/paired replay/short-full 全部关闭。
8. H7 成立于诊断层：APGA OOD destructive 由 longrisk-veto-ineffective 主导，longrisk created rate = `0.66796875`。
9. H8 未打开：APGI/APGC primitive 不能在 controller 缺失和 APGA OOD destructive triage 未完成时启动。
10. Base-Acc Sentinel 继续健康，但没有用于 controller，也不是 functional success。

最终一句话：

> v9.5.9 真实执行后停在 `R1-GroupSpecificRankPocket`：group-invariant ranker 已经把 rank pocket 的 group drop 明显压低，但仍未达到 official 稳定性；rank-safe certificate 的 heldout accepted region 质量不足，APGA OOD autopsy 又显示 longrisk veto 失效，因此 strict PureKAN functional 仍未成功。
