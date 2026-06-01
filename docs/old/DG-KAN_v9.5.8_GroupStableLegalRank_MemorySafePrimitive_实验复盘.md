# DG-KAN v9.5.8 Group-Stable Legal Rank / Memory-Safe Primitive 实验复盘

> 本复盘记录 `DG-KAN_v9.5.8_GroupStableLegalRank_MemorySafePrimitive_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 legacy outcome-derived rank、legal rank diagnostic、distillation sandbox、APGA implementation/smoke、certificate diagnostic、runtime boundary 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R1-LegalRankStillGroupUnstable
base_candidate = LQ-t2-h256
success_v9580_strict_purekan_functional = False
success_v9580_full_functional = False
success_v9580_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9580_group_stable_legal_rank_memory_safe_primitive_first_20260515T080000Z/
```

核心结论：

1. P0 复现 v9.5.7 boundary：source route = `R4-APGMGeneratedFrontierFail`，field legality ledger pass = `1`，legacy rank leakage explained = `1`，APGM implementation / branch-horizon pass = `1 / 1`，APGM weak pass = `0`。
2. P1 Field Legality / legacy rank forensic 继续确认 legacy rank 强但非法：legacy best TopK64 GradeAB precision = `0.890625`，legacy best red field count = `2`；best legal variant TopK64 GradeAB precision 只有 `0.09375`，V_integrated LCB = `-0.3660500491083909`。
3. P2 target density 仍未过：best target = `MemorySafeGradeB`，accepted = `37`，coverage = `0.01286509040333797`，V_integrated LCB = `0.21696278396362909`，h240 longrisk UCB = `0.0`，但 density 不足。
4. P3 legal causal feature 有 diagnostic signal 但风险不闭合：best = `ControlTransferImprovement`，AUC_GradeAB = `0.9850698985803052`，TopK64 GradeAB precision = `0.484375`，V_integrated LCB = `0.34107731007668374`，h240 longrisk UCB = `0.5428703985679182`。
5. P4 immediate microprobe 未过：best epsilon = `0.05`，TopK64 GradeAB precision = `0.234375`，V_integrated LCB = `-0.2566991749421587`。
6. P5 group-stable legal ranker 出现强 TopK diagnostic：best = `R2-ValueRankLongRiskVeto`，TopK87 GradeAB precision = `0.8505747126436781`，V_integrated LCB = `0.13419830825382068`，h240 longrisk UCB = `0.0`。
7. 但 P5 group-stable gate 未过：best ranker LDO/LSO drop max = `0.8854166666666666 / 0.8854166666666666`，`group_stable_rank_weak_pass = 0`，`legal_rank_still_group_unstable = 1`。
8. P6 distillation sandbox 过 diagnostic：student red field count = `0`，teacher overlap = `0.7816091954022989`，true TopK87 GradeAB precision = `0.7241379310344828`，V LCB = `0.12807606232790217`，longrisk UCB = `0.0`；但它仍不是 official controller。
9. P7 memory anatomy 本轮未过 v9580 gate：memory explained high-longrisk fraction = `0.6981002517738613`，低于 `0.70`；memory-safe value-positive count = `132`。
10. P8 APGA1-APGA8 implementation 过：generated actions = `512`，payload/certificate hash missing = `0 / 0`，action apply L∞ max = `0.0`。
11. P9 APGA branch-horizon replay 完整落盘：expected/actual rows = `12288 / 12288`，unresolved exception = `0`，quality audit pass = `1`，rows/sec = `14.788987206992267`。
12. P9 APGA outcome 未过：best = `APGA7-SignalMemoryIntersectionPrimitive`，GradeB precision = `0.0`，GradeAB precision = `0.03125`，V_integrated LCB = `-0.3580921649906729`，h240 longrisk UCB = `0.8150608413036654`。
13. P10 rank-based certificate v6 未过：best = `CERT-R2-LegalRankConformalRiskBound`，TopK64 GradeAB precision = `0.859375`，heldout accepted = `87`，coverage heldout = `0.08430232558139535`，但 heldout V_integrated LCB = `-0.14706519079354996`。
14. P11-P15 均 gate-blocked：没有 minimal geometry controller、selected runtime、leaveout、official paired replay 或 short/full training。
15. P16 Base-Acc Sentinel 复用前序真实 sentinel：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
16. 当前 primary blocker：`legal_rank_group_stability_drop_failed`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9580_group_stable_legal_rank_memory_safe_primitive.py` | v9.5.8 runner；读取 v9.5.7/v9.5.6/v9.5.5/v9.3.3 artifacts，执行 field legality、target density、group-stable legal rank、distillation sandbox、memory anatomy、APGA primitive、rank certificate 与 system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9580_group_stable_legal_rank_memory_safe_primitive.py
```

正式运行：

```bash
python experiments/run_v9580_group_stable_legal_rank_memory_safe_primitive.py \
  --out-dir results/real_rerun_20260506/v9580_group_stable_legal_rank_memory_safe_primitive_first_20260515T080000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --apga-actions-per-primitive 64
```

运行结果：

```json
{
  "apga_generated_actions": 512,
  "best_apga_V_integrated_LCB": -0.3580921649906729,
  "best_apga_primitive": "APGA7-SignalMemoryIntersectionPrimitive",
  "certificate_official_pass": 0,
  "legacy_best_TopK64_GradeAB_precision": 0.890625,
  "legal_best_TopK64_GradeAB_precision": 0.09375,
  "out_dir": "results/real_rerun_20260506/v9580_group_stable_legal_rank_memory_safe_primitive_first_20260515T080000Z",
  "route": "R1-LegalRankStillGroupUnstable"
}
```

## 2. Route

`route_decision_v9580.json` 摘要：

```json
{
  "route": "R1-LegalRankStillGroupUnstable",
  "source_route_v9570": "R4-APGMGeneratedFrontierFail",
  "field_legality_ledger_pass": 1,
  "legacy_rank_leakage_explained": 1,
  "legacy_best_TopK64_GradeAB_precision": 0.890625,
  "legal_best_TopK64_GradeAB_precision": 0.09375,
  "target_density_pass": 0,
  "legal_feature_weak_pass": 0,
  "group_stable_rank_weak_pass": 0,
  "legal_rank_topk_signal_present": 1,
  "legal_rank_still_group_unstable": 1,
  "best_ranker_id": "R2-ValueRankLongRiskVeto",
  "best_ranker_TopK87_GradeAB_precision": 0.8505747126436781,
  "best_ranker_LDO_drop_max": 0.8854166666666666,
  "distillation_pass": 1,
  "apga_implementation_pass": 1,
  "apga_branch_horizon_pass": 1,
  "apga_weak_pass": 0,
  "certificate_official_pass": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "legal_rank_group_stability_drop_failed"
}
```

判断：v9.5.8 的核心推进不是 APGA，而是 legal rank 的强 TopK signal 被明确定位为 group-unstable；因此 route 优先停在 `R1-LegalRankStillGroupUnstable`，没有把 APGA/certificate 的后续 diagnostic 写成 official。

## 3. P0/P1 boundary 与 field legality

Artifacts：

```text
p0_v9570_boundary_reproduction.csv
p1_field_legality_legacy_rank_forensic.csv
```

P0 summary：

```text
source_route_v9570 = R4-APGMGeneratedFrontierFail
field_legality_ledger_pass_v9570 = 1
legacy_rank_leakage_explained_v9570 = 1
apgm_implementation/branch_horizon/weak pass = 1 / 1 / 0
best_apgm = APGM2-PopulationRiskOffDiagonalGate
best_apgm_V_integrated_LCB = -0.3176437233277188
best_apgm_h240_longrisk_UCB = 0.7726149255507754
certificate_official_pass = 0
system_legal_controller_pass = 0
p0_pass = 1
```

P1 summary：

```text
field_count = 15
green/red = 11 / 4
score_variant_count = 8
legacy_best_score_variant = legacy_full_score
legacy_best_red_field_count = 2
legacy_best_TopK64_GradeAB_precision = 0.890625
legal_best_score_variant = legal_only_terms_plus_cost
legal_best_TopK64_GradeAB_precision = 0.09375
legal_best_V_integrated_LCB = -0.3660500491083909
field_legality_ledger_pass = 1
legacy_rank_leakage_explained = 1
legal_official_score_variant_pass = 0
```

判断：legacy rank 的强信号继续被解释为 red-field diagnostic；green-only legal variant 仍不能复现。

## 4. P2-P4 target / legal feature / microprobe

P2 artifact：

```text
p2_multigrade_target_density.csv
```

Summary：

```text
target_candidate_count = 8
official_target_candidate_count = 0
best_target_id = MemorySafeGradeB
best_accepted_count = 37
best_coverage = 0.01286509040333797
best_V_integrated_LCB = 0.21696278396362909
best_h240_longrisk_UCB = 0.0
target_density_pass = 0
```

P3 artifact：

```text
p3_legal_causal_geometry_feature_factory_v1.csv
```

Summary：

```text
feature_count = 12
feature_value_row_count = 34512
best_feature = ControlTransferImprovement
best_AUC_GradeAB = 0.9850698985803052
best_TopK64_GradeAB_precision = 0.484375
best_TopK64_V_integrated_LCB = 0.34107731007668374
best_TopK64_h240_longrisk_UCB = 0.5428703985679182
legal_feature_weak_pass = 0
```

P4 artifact：

```text
p4_legal_immediate_microprobe.csv
```

Summary：

```text
best_epsilon = 0.05
best_TopK64_GradeAB_precision = 0.234375
best_TopK64_V_integrated_LCB = -0.2566991749421587
best_TopK64_h240_longrisk_UCB = 0.09866091513007542
restore_error_linf_max = 0.0
microprobe_pass = 0
```

判断：target density 仍不足；legal feature 层有明显排序信号，但 TopK risk 太高；microprobe 没有补上 value。

## 5. P5 group-stable legal ranker

Artifact：

```text
p5_legal_rank_upper_bound.csv
```

Summary：

```text
ranker_count = 9
best_ranker_id = R2-ValueRankLongRiskVeto
best_TopK87_GradeAB_precision = 0.8505747126436781
best_TopK87_V_integrated_LCB = 0.13419830825382068
best_TopK87_h240_longrisk_UCB = 0.0
best_LDO_drop_max = 0.8854166666666666
best_LSO_drop_max = 0.8854166666666666
group_stable_rank_weak_pass = 0
legal_rank_topk_signal_present = 1
legal_rank_still_group_unstable = 1
```

Representative rankers：

| ranker | TopK87 GradeAB | V LCB | h240 longrisk UCB | LDO/LSO drop | strong diagnostic |
|---|---:|---:|---:|---:|---:|
| `R1-ValueRankMemoryVeto` | `0.8275862068965517` | `0.17452101582656632` | `0.03389313880385762` | `0.8229166666666666 / 0.8229166666666666` | `1` |
| `R2-ValueRankLongRiskVeto` | `0.8505747126436781` | `0.13419830825382068` | `0.0` | `0.8854166666666666 / 0.8854166666666666` | `1` |
| `R5-PairwiseDiagnosticMLPRank` | `0.8275862068965517` | `0.16125862511726052` | `0.054480608015849245` | `0.8385416666666666 / 0.8385416666666666` | `1` |
| `R7-LegacyGreenFieldDistilledRank` | `0.7241379310344828` | `0.12807606232790217` | `0.0` | `0.7604166666666666 / 0.7604166666666666` | `0` |

判断：P5 是本轮 primary blocker。legal rank 已经能抓到 value/risk 干净的 TopK 区域，但 group stability drop 远超 gate，不能作为 official selector/controller。

## 6. P6/P7 distillation 与 memory anatomy

P6 artifact：

```text
p6_legacy_to_legal_distillation_sandbox.csv
```

Summary：

```text
teacher_rank_id = legacy_GCERT18_outcome_diagnostic
student_rank_id = green_field_student_mlp
student red field count = 0
student_teacher_kendall_tau_proxy = 0.95305283321131
student_teacher_topk87_overlap = 0.7816091954022989
student_true_gradeab_topk87_precision = 0.7241379310344828
student_true_V_lcb = 0.12807606232790217
student_true_longrisk_ucb = 0.0
student_true_bad_ucb = 0.03389313880385762
student_true_null_ucb = 0.10637805008576995
distillation_pass = 1
```

P7 artifact：

```text
p7_memory_blocker_anatomy.csv
```

Summary：

```text
action_count = 5948
high_longrisk_count = 4369
memory_explained_high_longrisk_count = 3050
memory_failure_explained_fraction = 0.6981002517738613
P_longrisk_given_memory_fail = 0.8885551216581556
P_memory_fail_given_longrisk = 0.6770428015564203
memory_safe_value_positive_count = 132
memory_safe_value_positive_density = 0.02219233355749832
memory_blocker_anatomy_pass = 0
```

判断：distillation sandbox 是重要 diagnostic，但它仍依赖 legacy teacher 体系，不能 official。memory anatomy 在 v9580 纳入更多 generated rows 后略低于 `0.70` gate，不能继续写作 pass。

## 7. P8/P9 APGA primitive

P8 artifact：

```text
p8_apga_primitive_implementation.csv
```

Summary：

```text
primitive_count = 8
generated_action_count = 512
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_linf_max = 0.0
negative_control_generated = 1
apga_implementation_pass = 1
```

P9 artifacts：

```text
p9_apga_branch_horizon_outcome.csv
p9_apga_geometry_pass.csv
```

Materializer summary：

```text
expected_rows = 12288
actual_rows = 12288
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
secondary_delta_completion_rate = 1
rows_per_sec = 14.788987206992267
wallclock_sec = 830.8885407778434
unresolved_exception_count = 0
duplicate_row_count = 0
label_exclusivity_violation = 0
quality_audit_pass = 1
```

Outcome summary：

```text
best_primitive_id = APGA7-SignalMemoryIntersectionPrimitive
best_GradeB_precision = 0.0
best_GradeAB_precision = 0.03125
best_V_integrated_LCB = -0.3580921649906729
best_h240_longrisk_UCB = 0.8150608413036654
best_bad_UCB = 0.07387819590291736
best_null_UCB = 0.16516274587521124
best_memory_fail_UCB = 0.0
best_new_positive_created_rate = 0.03125
best_longrisk_created_rate = 0.703125
apga_weak_pass = 0
```

Per primitive：

| primitive | GradeAB precision | V_integrated LCB | h240 longrisk UCB | pass |
|---|---:|---:|---:|---:|
| `APGA1-SNRMaskedResidualUpdate` | `0.015625` | `-0.3625250058159727` | `0.8425830323796916` | `0` |
| `APGA2-PopulationRiskOffDiagonalGate` | `0.0` | `-0.36870374809998424` | `0.7726149255507754` | `0` |
| `APGA3-MemoryOrthogonalProjection` | `0.0` | `-0.3832315371323727` | `0.6991203985679182` | `0` |
| `APGA4-OldFamilyMarginGuard` | `0.0` | `-0.4002741437811319` | `0.828904255301089` | `0` |
| `APGA5-CoverRankPreservingUpdate` | `0.015625` | `-0.38055601580046217` | `0.7726149255507754` | `0` |
| `APGA6-SymmetricBoundaryDampedUpdate` | `0.0` | `-0.42570693741227184` | `0.7726149255507754` | `0` |
| `APGA7-SignalMemoryIntersectionPrimitive` | `0.03125` | `-0.3580921649906729` | `0.8150608413036654` | `0` |
| `APGA8-NegativeControlShuffledPayload` | `0.015625` | `-0.38498459745289065` | `0.7581802303291282` | `0` |

判断：APGA implementation/replay 完整闭合，但 generated frontier 仍失败。它没有把 group-stable legal rank 的 source-side diagnostic 转成 generated-action frontier。

## 8. P10-P16 boundary

P10：

```text
best_certificate_id = CERT-R2-LegalRankConformalRiskBound
best_rank_auc = 0.9834507955724663
best_TopK64_GradeAB_precision = 0.859375
best_heldout_accepted_count = 87
best_coverage_heldout = 0.08430232558139535
best_V_integrated_LCB_heldout = -0.14706519079354996
best_h240_longrisk_UCB_heldout = 0.07282499698983116
best_ECE = 0.3367782233169089
certificate_official_pass = 0
```

P11-P15：

| artifact | status / reason |
|---|---|
| `p11_minimal_geometry_controller.csv` | `not_run`, `P5_P9_P10_no_accepted_region_gate` |
| `p12_selected_runtime.csv` | `not_run`, `P11_controller_not_selected` |
| `p13_leaveout_validation.csv` | `not_run`, `P12_runtime_not_selected` |
| `p14_official_paired_replay.csv` | `not_run`, `P13_leaveout_not_open` |
| `p15_short_full_training.csv` | `not_run`, `P14_paired_replay_not_open` |

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

判断：certificate TopK 仍强，但 heldout accepted-region 的 V LCB 为负，不能 controller。没有 controller，就不能打开 runtime/paired replay/short-full。

## 9. No-fake audit

```text
rows_checked = 54239
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
v9570_boundary_pass = 1
field_legality_ledger_pass = 1
legacy_rank_leakage_explained = 1
legal_feature_weak_pass = 0
microprobe_pass = 0
group_stable_rank_weak_pass = 0
legal_rank_still_group_unstable = 1
distillation_pass = 1
memory_blocker_anatomy_pass = 0
apga_implementation_pass = 1
apga_branch_horizon_pass = 1
apga_weak_pass = 0
certificate_official_pass = 0
source_controller/selected_runtime/system = 0/0/0
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
route = R1-LegalRankStillGroupUnstable
F0_boundary_regression = 0
F1_legal_rank_still_group_unstable = 1
F2_legal_rank_accepted_controller_pending = 0
F3_distilled_rank_promising_not_official = 0
F4_apga_generated_frontier_fail = 0
F5_apga_generated_frontier_pass = 0
F6_certificate_accepted_region_fail = 0
F7_selected_runtime_fail = 0
F8_system_not_official = 1
F9_base_acc_catastrophic = 0
primary_blocker = legal_rank_group_stability_drop_failed
```

## 10. Hash

| artifact | SHA256 |
|---|---|
| plan | `9ca32c6168d8d82cc2c66627c0f2e35620f652217a63add125bfcb0b4de4bc48` |
| runner | `a34d76ceda1f07701c2e331c12d22e240f3ee2c3c9237a58be81ec8aa1e0e2b3` |
| run manifest | `43cfa763fd991acc4815755f43bbaa27beccd6fbe5b1c67434b692e5fb90ff5e` |
| route | `1c46a0fbf9a4afb2624b88fb570020db625cd3b7fff66b0aa969dd19d85fdf24` |
| P0 boundary | `d62289c0f781ebdbb5e23294777c1bfc35c47e4cc74e25fb2aab15769b8c14f5` |
| P1 field legality | `9008a2e4ac13e27227a87d7e28c0c055438003c27c81327f88df5d8b6eaf7f52` |
| P2 target density | `d61e7e890ffa7c0eb670f01243cdeaa0075e8f5352c9d41035aac01eb7aab60c` |
| P3 legal feature | `4c912a80ec0a644db52dd73a678ee7571a23a0757be158ab605ad996b2af1a64` |
| P4 microprobe | `fa938ee3b07d10daf294c3415be290de6c13446093536725232c88b54deb6212` |
| P5 legal rank | `b7b03ddebc372aab3ba571aeb184e16b4420a1a5216df1d0564d4502ff24ec05` |
| P6 distillation | `e2801725d473ee40bb05e24ee1d3e9b22e445e00b30c10be0bbc345fb0a67667` |
| P7 memory | `84b13d50362bf4f34c1cb4c9e8f1e1c8280409115988aa97faf0d6a552fc4c3b` |
| P8 APGA implementation | `a736e80b6269a555d791f88300c678edebd387e558a7893a41bbef1054a4d694` |
| P9 APGA smoke | `0fe40473b4131efc3f77389ddd48c9f025790694c37439e24d37893239635bfa` |
| P9 APGA geometry | `bbcbbd811819b97792cdd3cf2c7d131681379ba44f4a764576942f6f714e7e51` |
| P10 certificate | `f35630eb8a4a9e137517725ff1b1be050ce8b98a57c7699977ef404b4ad1f6e0` |
| P16 Base-Acc Sentinel | `0c9c93887d6dc8e45d2e29754474161a3646e71c1ceaf8760861eeea630f78ca` |
| no-fake audit | `e479cbf0e490cee43ce27dab5a307e0120931fbddcd09323e7095f1e65e06e73` |
| contract audit | `ebed2ffc26d57e9b022c79053dc97fbc86dbd90802c0345c2aa029fcd0c7e864` |
| failure table | `495a816bb1933f239a002f46243b1cfe0fa2dd46422cccacb21c98213f12cfca` |

## 11. 最终分析结论

v9.5.8 的真实推进是：

```text
v9.5.7:
  legal rank/distillation 已显示非 outcome-derived signal；
  memory blocker 被确认；
  但 APGM generated frontier 与 rank certificate accepted region 未闭合。

v9.5.8:
  将 legal rank 从单纯 TopK diagnostic 推进到 group-stable gate；
  证明 R1/R2/R5 等 legal ranker 能抓到高 GradeAB、正 V、低 longrisk 的 TopK87；
  但这些 ranker 对 LDO/LSO 极不稳定，drop max 约 0.82-0.89；
  APGA1-APGA8 工程链路和 full branch-horizon replay 完整闭合；
  但 APGA generated actions 仍 value-negative / high-longrisk；
  rank certificate TopK 仍强，但 heldout accepted-region V LCB 为负。
```

机制判断：

1. H0 成立：v9.5.7 boundary 被复现，没有跳过 legacy leakage / APGM failure。
2. H1 部分成立：legal rank TopK signal 很强；best R2 TopK87 GradeAB precision = `0.8506`，V LCB > 0，longrisk UCB = `0.0`。
3. H2 未成立：group stability 没闭合；best R2 的 LDO/LSO drop 都是 `0.8854`，不能 official。
4. H3 成立于 diagnostic 层：distillation student 是 green-field，TopK87 precision = `0.7241`，risk/bad/null 都较干净，但它仍是 sandbox，不是 controller。
5. H4 未成立：memory anatomy 在 v9580 full prior ledger 下低于 pass threshold，说明 memory explanation 仍不够完整。
6. H5/H6 成立于 implementation 层：APGA payload/certificate/action apply 与 branch-horizon replay 全部完整落盘。
7. H7 未成立：APGA outcome 没有形成 generated frontier；best APGA7 的 V_integrated LCB 为负，h240 longrisk UCB 很高。
8. H8 未成立：certificate TopK strong，但 heldout accepted V LCB 为负，不能打开 controller。
9. H9-H12 未打开：没有 controller，就不能打开 selected runtime、leaveout、official paired replay 或 short/full training。
10. Base-Acc Sentinel 继续健康，但没有用于 controller，也不是 functional success。

最终一句话：

> v9.5.8 真实执行后停在 `R1-LegalRankStillGroupUnstable`：legal rank 已经能在 TopK87 找到高 GradeAB、正 value、低 longrisk 的 diagnostic 区域，但该区域对 group leaveout 极不稳定；APGA 虽完整落盘仍未生成 value-positive / horizon-safe frontier，certificate heldout accepted region 也不过，因此 strict PureKAN functional 仍未成功。
