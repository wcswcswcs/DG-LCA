# DG-KAN v9.5.7 Legal Causal Geometry Rank / Memory-Preserving Primitive / Parallel Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.5.7_LegalCausalGeometryRank_MemoryPreservingPrimitive_ParallelClosure_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 legacy outcome-derived rank、legal rank diagnostic、distillation sandbox、APGM implementation/smoke、certificate diagnostic、runtime boundary 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R4-APGMGeneratedFrontierFail
base_candidate = LQ-t2-h256
success_v9570_strict_purekan_functional = False
success_v9570_full_functional = False
success_v9570_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9570_legal_causal_geometry_rank_memory_preserving_primitive_first_20260515T070000Z/
```

核心结论：

1. P0 复现 v9.5.6 boundary：source route = `R1-GradeScopeInconsistent`，legacy GCERT18 outcome-derived leakage 仍成立，system pass = `0`。
2. P1 Field Legality Ledger 过：field count = `15`，green = `11`，red = `4`；legacy full score red field count = `2`。
3. P1 legacy rank 强但非法：legacy best TopK64 GradeAB precision = `0.890625`；best legal score variant TopK64 GradeAB precision = `0.09375`，V_integrated LCB = `-0.3660500491083909`。
4. P2 target density 未过：best non-legacy target = `MemorySafeGradeB`，accepted = `37`，coverage = `0.01286509040333797`，target_density_pass = `0`。
5. P3 legal causal geometry feature 有 diagnostic signal 但风险不闭合：best = `ControlTransferImprovement`，AUC_GradeAB = `0.9850698985803052`，TopK64 GradeAB precision = `0.484375`，V_integrated LCB = `0.34107731007668374`，但 h240 longrisk UCB = `0.5428703985679182`，legal_feature_weak_pass = `0`。
6. P4 immediate microprobe 未过：best epsilon = `0.05`，TopK64 GradeAB precision = `0.234375`，V_integrated LCB = `-0.2566991749421587`。
7. P5 legal rank upper-bound 出现强 TopK diagnostic，但 official pass 未过：best ranker = `UB-M1-small-mlp-diagnostic-ranker`，TopK87 GradeAB precision = `0.8275862068965517`，V_integrated LCB = `0.16125862511726052`，h240 longrisk UCB = `0.054480608015849245`；但 LDO/LSO drop = `0.8385416666666666`，legal_rank_upper_bound_pass = `0`。
8. P6 legacy-to-legal distillation sandbox 过 diagnostic：student red field count = `0`，teacher overlap = `0.7816091954022989`，true TopK87 GradeAB precision = `0.7241379310344828`，V LCB = `0.12807606232790217`，longrisk UCB = `0.0`；但该阶段不能 official。
9. P7 memory blocker anatomy 过：high-longrisk count = `3993`，memory explained fraction = `0.7638367142499374`，memory-safe value-positive count = `121`。
10. P8 APGM1-APGM8 implementation 过：generated actions = `512`，payload/certificate hash missing = `0 / 0`，action apply L∞ max = `0.0`。
11. P9 APGM branch-horizon 完整落盘：expected/actual rows = `12288 / 12288`，unresolved exception = `0`，quality audit pass = `1`。
12. P9 APGM outcome 未过：best = `APGM2-PopulationRiskOffDiagonalGate`，GradeB precision = `0.0`，GradeAB precision = `0.015625`，V_integrated LCB = `-0.3176437233277188`，h240 longrisk UCB = `0.7726149255507754`。
13. P10 rank-based certificate v5 未过：best = `CERT-R2-LegalRankConformalRiskBound`，TopK64 GradeAB precision = `0.859375`，heldout accepted = `87`，coverage = `0.08630952380952381`，但 heldout V_integrated LCB = `-0.1561252169840407`，h240 longrisk UCB = `0.2517897676033668`。
14. P11-P15 均 gate-blocked：没有 minimal geometry controller、selected runtime、leaveout、official paired replay 或 short/full training。
15. P16 Base-Acc Sentinel 复用前序真实 sentinel：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
16. 当前 primary blocker：`apgm_generated_frontier_failed`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9570_legal_causal_geometry_rank_memory_preserving_primitive.py` | v9.5.7 runner；读取 v9.5.6/v9.5.5 artifacts，执行 field legality、legacy rank forensic、legal causal geometry features、rank upper-bound、distillation sandbox、memory anatomy、APGM primitive、rank certificate 与 system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9570_legal_causal_geometry_rank_memory_preserving_primitive.py
```

正式运行：

```bash
python experiments/run_v9570_legal_causal_geometry_rank_memory_preserving_primitive.py \
  --out-dir results/real_rerun_20260506/v9570_legal_causal_geometry_rank_memory_preserving_primitive_first_20260515T070000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --apgm-actions-per-primitive 64
```

运行结果：

```json
{
  "apgm_generated_actions": 512,
  "best_apgm_V_integrated_LCB": -0.3176437233277188,
  "best_apgm_primitive": "APGM2-PopulationRiskOffDiagonalGate",
  "certificate_official_pass": 0,
  "legacy_best_TopK64_GradeAB_precision": 0.890625,
  "legal_best_TopK64_GradeAB_precision": 0.09375,
  "out_dir": "results/real_rerun_20260506/v9570_legal_causal_geometry_rank_memory_preserving_primitive_first_20260515T070000Z",
  "route": "R4-APGMGeneratedFrontierFail"
}
```

说明：第一次启动时曾发现 APGM payload index 取错导致 generated actions = `0`；该空跑目录已删除，没有作为正式 artifact。正式 artifact 使用和 v9.5.5/v9.5.6 一致的 v9.3.3 payload registry，APGM generated actions = `512`，branch-horizon rows = `12288 / 12288`。

## 2. Route

`route_decision_v9570.json` 摘要：

```json
{
  "route": "R4-APGMGeneratedFrontierFail",
  "source_route_v9560": "R1-GradeScopeInconsistent",
  "field_legality_ledger_pass": 1,
  "legacy_rank_leakage_explained": 1,
  "legacy_best_TopK64_GradeAB_precision": 0.890625,
  "legal_best_TopK64_GradeAB_precision": 0.09375,
  "legal_best_V_integrated_LCB": -0.3660500491083909,
  "target_density_pass": 0,
  "legal_feature_weak_pass": 0,
  "microprobe_pass": 0,
  "legal_rank_upper_bound_pass": 0,
  "distillation_pass": 1,
  "memory_blocker_anatomy_pass": 1,
  "memory_failure_explained_fraction": 0.7638367142499374,
  "apgm_implementation_pass": 1,
  "apgm_branch_horizon_pass": 1,
  "apgm_branch_horizon_rows_actual": 12288,
  "apgm_unresolved_exception_count": 0,
  "apgm_weak_pass": 0,
  "best_apgm_primitive": "APGM2-PopulationRiskOffDiagonalGate",
  "best_apgm_GradeB_precision": 0.0,
  "best_apgm_V_integrated_LCB": -0.3176437233277188,
  "best_apgm_h240_longrisk_UCB": 0.7726149255507754,
  "certificate_official_pass": 0,
  "best_certificate_id": "CERT-R2-LegalRankConformalRiskBound",
  "source_controller_pass": 0,
  "selected_runtime_pass": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "apgm_generated_frontier_failed"
}
```

判断：v9.5.7 的正向推进是 legal rank/distillation 已经不再完全死，但 AP0 official accepted-region 和 APGM generated frontier 都没有闭合；因此 route 正确停在 `R4-APGMGeneratedFrontierFail`。

## 3. P0/P1 boundary 与 field legality

Artifacts：

```text
p0_v9560_boundary_reproduction.csv
p1_field_legality_legacy_rank_forensic.csv
```

P0 summary：

```text
source_route_v9560 = R1-GradeScopeInconsistent
outcome_field_used_by_certificate_count = 2
legacy_gcert18_legal_commit_time_pass = 0
apgc_implementation/branch_horizon/weak pass = 1 / 1 / 0
official_certificate_pass = 0
system_legal_controller_pass = 0
p0_pass = 1
```

P1 summary：

```text
field_count = 15
green_field_count = 11
red_field_count = 4
score_variant_count = 8
legacy_best_score_variant = legacy_full_score
legacy_best_red_field_count = 2
legacy_best_TopK64_GradeAB_precision = 0.890625
legal_best_score_variant = legal_only_terms_plus_cost
legal_best_TopK64_GradeAB_precision = 0.09375
legal_best_V_integrated_LCB = -0.3660500491083909
legal_best_h240_longrisk_UCB = 0.4601149255507754
field_legality_ledger_pass = 1
legacy_rank_leakage_explained = 1
legal_official_score_variant_pass = 0
```

判断：P1 过 ledger/forensic，但不过 official signal。legacy strong TopK 仍来自 red fields，不可 official。

## 4. P2 target density

Artifact：

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

判断：Memory-safe GradeB outcome 很干净，但 density 明显不够；不能靠 AP0 target density 打开 controller。

## 5. P3-P6 legal signal / rank / distillation

P3 artifact：

```text
p3_legal_causal_geometry_feature_factory_v1.csv
```

P3 summary：

```text
feature_count = 12
feature_value_row_count = 34512
best_feature_group_id = F1-linearized-action-effect
best_feature_name = ControlTransferImprovement
best_AUC_GradeAB = 0.9850698985803052
best_TopK64_GradeAB_precision = 0.484375
best_TopK64_V_integrated_LCB = 0.34107731007668374
best_TopK64_h240_longrisk_UCB = 0.5428703985679182
legal_feature_pass = 0
legal_feature_weak_pass = 0
```

P4 artifact：

```text
p4_legal_immediate_microprobe.csv
```

P4 summary：

```text
best_epsilon = 0.05
best_TopK64_GradeAB_precision = 0.234375
best_TopK64_V_integrated_LCB = -0.2566991749421587
best_TopK64_h240_longrisk_UCB = 0.09866091513007542
best_feature_cost_ms_q90 = 0.33
restore_error_linf_max = 0.0
microprobe_pass = 0
```

P5 artifact：

```text
p5_legal_rank_upper_bound.csv
```

P5 best ranker：

```text
ranker_id = UB-M1-small-mlp-diagnostic-ranker
TopK64 GradeAB precision = 0.921875
TopK64 V_integrated LCB = 0.19684482149968938
TopK64 h240 longrisk UCB = 0.0
TopK87 GradeAB precision = 0.8275862068965517
TopK87 V_integrated LCB = 0.16125862511726052
TopK87 h240 longrisk UCB = 0.054480608015849245
TopK87 bad UCB = 0.054480608015849245
TopK87 memory fail UCB = 0.07282499698983116
LDO/LSO drop max = 0.8385416666666666 / 0.8385416666666666
rank_upper_bound_pass = 0
rank_upper_bound_strong_pass = 1
```

P6 artifact：

```text
p6_legacy_to_legal_distillation_sandbox.csv
```

P6 summary：

```text
teacher_rank_id = legacy_GCERT18_outcome_diagnostic
student_rank_id = green_field_student_mlp
teacher/student red field count = 2 / 0
student_teacher_kendall_tau_proxy = 0.95305283321131
student_teacher_topk87_overlap = 0.7816091954022989
student_true_gradeab_topk87_precision = 0.7241379310344828
student_true_V_lcb = 0.12807606232790217
student_true_longrisk_ucb = 0.0
student_true_bad_ucb = 0.03389313880385762
student_true_null_ucb = 0.10637805008576995
distillation_pass = 1
```

判断：v9.5.7 首次把 legal rank 从“完全看不见”推进到“强 diagnostic 但 official 不稳”。P5/P6 是重要线索，但 P5 LDO/LSO drop 太大，P6 又是 legacy teacher sandbox，不能直接打开 official controller。

## 6. P7 memory blocker anatomy

Artifact：

```text
p7_memory_blocker_anatomy.csv
```

Summary：

```text
action_count = 5436
high_longrisk_count = 3993
memory_explained_high_longrisk_count = 3050
memory_failure_explained_fraction = 0.7638367142499374
P_longrisk_given_memory_fail = 0.8885551216581556
P_memory_fail_given_longrisk = 0.7407963936889557
memory_safe_value_positive_count = 121
memory_safe_value_positive_density = 0.022259013980868287
memory_blocker_anatomy_pass = 1
```

判断：memory blocker 成立。它解释了超过 `70%` 的 high-longrisk actions，同时 memory-safe positive 没被完全清空。

## 7. P8/P9 APGM primitive

P8 artifact：

```text
p8_apgm_primitive_implementation.csv
```

P8 summary：

```text
primitive_count = 8
generated_action_count = 512
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_linf_max = 0.0
negative_control_generated = 1
apgm_implementation_pass = 1
```

P9 artifacts：

```text
p9_apgm_branch_horizon_outcome.csv
p9_apgm_geometry_pass.csv
```

P9 materializer summary：

```text
expected_rows = 12288
actual_rows = 12288
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
secondary_delta_completion_rate = 1
rows_per_sec = 14.66621490637321
wallclock_sec = 837.8439889531583
unresolved_exception_count = 0
duplicate_row_count = 0
label_exclusivity_violation = 0
quality_audit_pass = 1
```

P9 best primitive：

```text
best_primitive_id = APGM2-PopulationRiskOffDiagonalGate
best_GradeB_precision = 0.0
best_GradeAB_precision = 0.015625
best_V_integrated_LCB = -0.3176437233277188
best_h240_longrisk_UCB = 0.7726149255507754
best_bad_UCB = 0.04600980021300741
best_null_UCB = 0.18584173698566175
best_memory_fail_UCB = 0.0
best_new_positive_created_rate = 0.015625
best_longrisk_created_rate = 0.65625
best_source_to_generated_damage_LCB = -0.22506442482174488
negative_control_is_best = 0
apgm_weak_pass = 0
apgm_official_candidate_pass = 0
```

判断：APGM 工程链路闭合，但 science gate 失败。best APGM2 仍 value-negative 且 high-longrisk，不能继续调阈值写成 generator pass。

## 8. P10 rank-based certificate v5

Artifact：

```text
p10_rank_based_certificate_v5.csv
```

Summary：

```text
certificate_count = 6
best_certificate_id = CERT-R2-LegalRankConformalRiskBound
best_rank_auc = 0.9855670103092784
best_TopK64_GradeAB_precision = 0.859375
best_heldout_accepted_count = 87
best_coverage_heldout = 0.08630952380952381
best_V_integrated_LCB_heldout = -0.1561252169840407
best_h240_longrisk_UCB_heldout = 0.2517897676033668
best_ECE = 0.3344656054779706
certificate_official_pass = 0
```

Reference：

```text
CERT-R6-CostConstrainedRankCertificate:
  TopK64 GradeAB precision = 0.921875
  TopK87 GradeAB precision = 0.7011494252873564
  heldout accepted = 87
  heldout V_integrated LCB = -0.18259147198909964
  heldout h240 longrisk UCB = 0.3570360950674304
```

判断：rank-based certificate 不再被 ECE 单独否决，但 accepted-region 质量仍不过。高 TopK precision 没有带来 heldout V/risk closure。

## 9. P11-P16 boundary

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

判断：没有把 legal rank diagnostic、distillation sandbox、APGM smoke、certificate diagnostic 或 Base-Acc Sentinel 写成 official controller/runtime/system/downstream pass。

## 10. No-fake audit

```text
rows_checked = 53721
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
v9560_boundary_pass = 1
field_legality_ledger_pass = 1
legacy_rank_leakage_explained = 1
legal_feature_weak_pass = 0
microprobe_pass = 0
legal_rank_upper_bound_pass = 0
distillation_pass = 1
memory_blocker_anatomy_pass = 1
apgm_implementation_pass = 1
apgm_branch_horizon_pass = 1
apgm_weak_pass = 0
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
route = R4-APGMGeneratedFrontierFail
F0_boundary_regression = 0
F1_outcome_leakage_confirmed_legal_signal_absent = 0
F2_legal_signal_exists_controller_fail = 0
F3_memory_primitive_required = 0
F4_apgm_generated_frontier_fail = 1
F5_runtime_fail = 0
F6_paired_replay_blocked = 0
F7_system_not_official = 1
F8_base_acc_catastrophic = 0
primary_blocker = apgm_generated_frontier_failed
```

## 11. Hash

| artifact | SHA256 |
|---|---|
| plan | `aa74948cb658bedadc5051e6f5f100c83dae3276fd06473d30a75b0e94e40366` |
| runner | `ec650d871e54f4e16269df518a92b2755174798583685c11369e187722ab7bc2` |
| run manifest | `ccb4c5d73b9404bf71499333f4fe039ea7855a8cb9af3ee64259e0b034e15d2a` |
| route | `571c2bc040f891451d90aaa291453e65e3b228a45c13688ebcca3729b2f6d41b` |
| P0 boundary | `725662905194dd1ff042951d540f34d8805559fa5151a82a986f67d29450a0c3` |
| P1 field legality | `9008a2e4ac13e27227a87d7e28c0c055438003c27c81327f88df5d8b6eaf7f52` |
| P2 target density | `d61e7e890ffa7c0eb670f01243cdeaa0075e8f5352c9d41035aac01eb7aab60c` |
| P3 legal feature | `4c912a80ec0a644db52dd73a678ee7571a23a0757be158ab605ad996b2af1a64` |
| P4 microprobe | `fa938ee3b07d10daf294c3415be290de6c13446093536725232c88b54deb6212` |
| P5 legal rank | `f13099aedcd58e406b99b8133b49293b3f270f298562a9b84c1da823eca9a8c9` |
| P6 distillation | `e2801725d473ee40bb05e24ee1d3e9b22e445e00b30c10be0bbc345fb0a67667` |
| P7 memory | `8a5660822d15f07ee1e3f2f32f6ac7e2e540beffd25b2e5f683087fcc87625b2` |
| P8 APGM implementation | `8e4fc37208f7f88f48abba456f402025b9f93ea1988e91ddbea8973b663f38f3` |
| P9 APGM smoke | `ef62677ce1d4f69d72f74c8836d35f9123c7432222acacbb02977346e9418b08` |
| P9 APGM geometry | `7daddd5c04290486114f96aaad625d06872a28131beae4f3cb734742501020b1` |
| P10 certificate | `24ff92a15ffd16f6eb0ccad1f627c2a3866055dd18d574e4788e3d282d4e1209` |
| P16 Base-Acc Sentinel | `23fb7bdd73c49f6856a356858aac50e652ba90ee38d1b44a353be1e61c603827` |
| no-fake audit | `bc6c8bc3bdda8f1cdc15f86f138f58cc7aef90977ffbb7728609e9c8468f9bd1` |
| contract audit | `7016662789d7adc03ef3fd1b010055840838b01e5ffac564744c9439f8f0792d` |
| failure table | `2c5e40d315c526de23dd7013afdb7e2d94308b33e803b45850b29f83163535c5` |

## 12. 最终分析结论

v9.5.7 的真实推进是：

```text
v9.5.6:
  legacy GCERT18 TopK 强信号被证明使用 outcome-derived V_integrated/risk_score；
  legal surrogate 无法复现；
  APGC 工程闭合但 generated frontier 失败。

v9.5.7:
  建立 field legality ledger，并完整解释 legacy rank leakage；
  扩展 legal causal geometry/action-effect features；
  训练 legal rank upper-bound 与 legacy-to-legal distillation sandbox；
  证明 legal rank diagnostic 已经能抓到一部分好区域，但 official leaveout/accepted-region 仍不稳；
  memory blocker anatomy 成立；
  APGM1-APGM8 memory-preserving primitive 工程链路和 branch-horizon materializer 完整闭合；
  但 APGM generated actions 仍 value-negative / high-longrisk；
  rank-based certificate v5 TopK 高，但 heldout V/risk 不过，不能 controller。
```

机制判断：

1. H1 成立：legacy rank leakage 被拆清楚；legacy full score red field count = `2`，best legal score variant不能复现 TopK。
2. H2 部分成立：legal causal/rank signal 并非完全不存在，P5/P6 给出强 diagnostic；但 official pass 仍失败于 leaveout、bad/risk 或 accepted-region heldout V。
3. H3 成立：memory 是当前主要 blocker，memory failure explained high-longrisk fraction = `0.7638`。
4. H4 未成立：APGM1-APGM8 implementation 与 materializer 过，但 best APGM 仍 V_integrated LCB < 0 且 h240 longrisk UCB 高。
5. H5 部分成立：rank-based certificate 比概率校准更贴近当前问题，TopK precision 很高；但 heldout accepted region 的 V/risk 没闭合，controller 仍不能打开。
6. P11-P15 未打开：没有 controller，就不能打开 selected runtime、leaveout、official paired replay 或 short/full training。
7. Base-Acc Sentinel 继续健康，但没有用于 controller，也不是 functional success。

最终一句话：

> v9.5.7 真实执行后停在 `R4-APGMGeneratedFrontierFail`：legal rank/distillation 已经显示出可研究的非 outcome-derived signal，memory blocker 也被确认；但 APGM1-APGM8 虽完整落盘，却没有生成 value-positive / horizon-safe frontier，rank-based certificate 的 heldout accepted region 也不过，因此 strict PureKAN functional 仍未成功。
