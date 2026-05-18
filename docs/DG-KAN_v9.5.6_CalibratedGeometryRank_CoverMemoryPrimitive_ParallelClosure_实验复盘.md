# DG-KAN v9.5.6 Calibrated Geometry Rank / Cover-Memory Primitive / Parallel Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.5.6_CalibratedGeometryRank_CoverMemoryPrimitive_ParallelClosure_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 legacy GCERT18 diagnostic、outcome-derived TopK、APGC implementation/smoke、certificate diagnostic、runtime boundary 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R1-GradeScopeInconsistent
base_candidate = LQ-t2-h256
success_v9560_strict_purekan_functional = False
success_v9560_full_functional = False
success_v9560_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9560_calibrated_geometry_rank_cover_memory_primitive_first_20260515T060000Z/
```

核心结论：

1. P0 复现 v9.5.5 boundary：source route = `R2a-CleanGeometryTargetTooSparse`，system pass = `0`，no-fake/no-proxy = `1 / 1`。
2. P1 Grade universe / TopK scope audit 完成：canonical AP0 GradeAB count = `79`，GCERT18 eval universe GradeAB count = `86`，legacy GCERT18 TopK64 GradeAB = `57 / 64 = 0.890625`。
3. P1 universe count 本身没有越界：`TopK64_GradeAB_count <= GradeAB_count_GCERT18_eval_universe`，universe consistency pass = `1`。
4. 但 P1 发现 legacy GCERT18 score 使用了 outcome-derived fields：`V_integrated,risk_score`，`outcome_field_used_by_certificate_count = 2`，因此 `grade_scope_audit_pass = 0`。
5. P2 rank/calibration dissection 证明 legacy diagnostic 很强但不能 official：legacy TopK64 GradeAB precision = `0.890625`，V_integrated LCB = `0.1650359732307112`，h240 longrisk UCB = `0.0`，但 ECE = `0.6217805447724831` 且 uses outcome-derived score = `1`。
6. P2 legal commit-time surrogate 不复现该强 TopK：legal TopK64 GradeAB precision = `0.046875`，V_integrated LCB = `-0.5602833203663737`，h240 longrisk UCB = `0.6991203985679182`。
7. P3 best target 是 `T_rank_legacy_GCERT18_TopK87_diagnostic`，accepted = `87`，coverage = `0.030250347705146036`，V_integrated LCB = `0.10620831245148316`，h240 longrisk UCB = `0.0`；但它来自 legacy outcome-derived rank，不能 official，`target_density_pass = 0`。
8. P4 cover/memory anatomy 显示 memory blocker 成立：APGR longrisk memory fail rate = `0.27735368956743`，GradeB memory fail rate = `0.02531645569620253`，`memory_blocker_confirmed = 1`；cover blocker 未成立。
9. P5 signal/reservoir v5 未过：best = `old_family_signal_score`，TopK64 GradeAB precision = `0.296875`，TopK64 V_integrated LCB = `-0.48271869313319804`，TopK64 longrisk = `0.390625`。
10. P6 existing-action controller 未过：best = `C1-GCERT18-rank-only-legacy-diagnostic`，accepted = `64`，coverage = `0.022253129346314324`，V_integrated LCB = `0.16207759868285035`，h240 longrisk UCB = `0.0`，但它是 legacy diagnostic 且 coverage 不足，controller pass = `0`。
11. P7 APGC1-APGC8 implementation 过：generated actions = `512`，payload/certificate hash missing = `0 / 0`，action apply L∞ max = `0.0`。
12. P8 APGC branch-horizon smoke 完整落盘：expected/actual rows = `12288 / 12288`，unresolved exception = `0`，duplicate row = `0`，label exclusivity violation = `0`，quality audit pass = `1`。
13. P9 APGC outcome 未过：best = `APGC8-NegativeControlShuffledPayload`，GradeB precision = `0.046875`，OfficialGeo precision = `0.125`，V_integrated LCB = `-0.3361628959034951`，h240 longrisk = `0.78125`。
14. APGC8 negative control 成为 best diagnostic 本身说明 APGC1-APGC7 没有产生真实可用 frontier；不能写成 generator pass。
15. P10 certificate v4 未过：best = `GCERT25-GCERT18RankOnly`，AUC_GradeAB = `0.9914807162534435`，TopK64 GradeAB precision = `0.90625`，V_integrated LCB = `0.1678302289887135`，h240 longrisk UCB = `0.0`，但 uses outcome-derived score = `1` 且 ECE = `0.6259564563021709`。
16. P10 最好的 legal candidate 仍未过：`GCERT27-GCERT18ConformalRiskBound` AUC_GradeAB = `0.9102100550964187`，TopK64 GradeAB precision = `0.125`，TopK64 V_integrated LCB = `-0.2732468908819606`。
17. P11-P15 gate-blocked：没有 minimal geometry controller、selected runtime、leaveout、official paired replay 或 short/full training。
18. Base-Acc Sentinel 复用前序真实 sentinel：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
19. 当前 primary blocker：`legacy_gcert18_uses_outcome_derived_score`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9560_calibrated_geometry_rank_cover_memory_primitive.py` | v9.5.6 runner；读取 v9.5.5 artifacts，执行 Grade/GCERT18 universe audit、rank/calibration dissection、multi-grade target density、cover/memory anatomy、signal/reservoir v5、existing-action controller candidate、APGC primitive、certificate v4 与 system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9560_calibrated_geometry_rank_cover_memory_primitive.py
```

正式运行：

```bash
python experiments/run_v9560_calibrated_geometry_rank_cover_memory_primitive.py \
  --out-dir results/real_rerun_20260506/v9560_calibrated_geometry_rank_cover_memory_primitive_first_20260515T060000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --apgc-actions-per-primitive 64
```

运行结果：

```json
{
  "apgc_generated_actions": 512,
  "apgc_rows": 12288,
  "best_apgc_GradeB_precision": 0.046875,
  "best_apgc_primitive": "APGC8-NegativeControlShuffledPayload",
  "grade_scope_audit_pass": 0,
  "legacy_TopK64_GradeAB_precision": 0.890625,
  "legal_TopK64_GradeAB_precision": 0.046875,
  "official_certificate_pass": 0,
  "out_dir": "results/real_rerun_20260506/v9560_calibrated_geometry_rank_cover_memory_primitive_first_20260515T060000Z",
  "route": "R1-GradeScopeInconsistent",
  "system_legal_controller_pass": 0
}
```

说明：本轮没有 CPU offload，没有 proxy rows；legacy GCERT18 的强 TopK 保留为 diagnostic，但由于使用 outcome-derived score，不能 official。

## 2. Route

`route_decision_v9560.json` 摘要：

```json
{
  "route": "R1-GradeScopeInconsistent",
  "source_route_v9550": "R2a-CleanGeometryTargetTooSparse",
  "grade_scope_audit_pass": 0,
  "GradeAB_count_canonical_AP0": 79,
  "GradeAB_count_GCERT18_eval_universe": 86,
  "TopK64_GradeAB_precision_legacy": 0.890625,
  "LegalTopK64_GradeAB_precision": 0.046875,
  "outcome_field_used_by_certificate_count": 2,
  "legacy_gcert18_legal_commit_time_pass": 0,
  "rank_signal_legacy_diagnostic_pass": 1,
  "legacy_TopK64_V_integrated_LCB": 0.1650359732307112,
  "legacy_ECE": 0.6217805447724831,
  "legal_TopK64_V_integrated_LCB": -0.5602833203663737,
  "legal_ECE": 0.6315531094435708,
  "target_density_pass": 0,
  "memory_blocker_confirmed": 1,
  "signal_reservoir_audit_pass": 0,
  "existing_action_controller_pass": 0,
  "apgc_implementation_pass": 1,
  "apgc_branch_horizon_pass": 1,
  "apgc_weak_pass": 0,
  "official_certificate_pass": 0,
  "source_controller_pass": 0,
  "selected_runtime_pass": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "legacy_gcert18_uses_outcome_derived_score"
}
```

判断：v9.5.6 的关键推进是把 v9.5.5 的强 GCERT18 TopK diagnostic 解剖清楚。它不是 Grade universe 越界，而是 legacy score 本身混入 outcome-derived fields；legal commit-time surrogate 不能复现该信号。

## 3. P0 v9.5.5 boundary

Artifact：

```text
p0_v9550_boundary_reproduction.csv
```

Summary：

```text
source_route_v9550 = R2a-CleanGeometryTargetTooSparse
ledger_rows = 4412
T5_action_count = 57
GradeA/B/C/D/E = 40 / 39 / 18 / 89 / 2690
best_signal = SNR-old-family
best_signal_AUC_GradeB = 0.9302100351642583
best_signal_TopK64_precision_GradeB = 0.296875
cover/curvature/memory pass = 0 / 1 / 0
best_APGR = APGR5-MemoryGuardedResidualUpdate
best_APGR_GradeB_precision = 0.046875
best_APGR_V_integrated_LCB = -0.8658484647235336
best_APGR_h240_longrisk = 0.78125
GCERT18_AUC = 0.992657726818137
GCERT18_TopK64_precision = 0.890625
GCERT18_ECE = 0.6217805447724831
system_legal_controller_pass = 0
p0_pass = 1
```

判断：P0 pass。v9.5.6 没有跳过 v9.5.5 的 target density / APGR / certificate boundary。

## 4. P1 Grade scope audit

Artifact：

```text
grade_scope_audit_v9560.csv
```

Summary：

```text
GradeA_count_canonical_AP0 = 40
GradeB_count_canonical_AP0 = 39
GradeAB_count_canonical_AP0 = 79
GradeC_count_canonical_AP0 = 18
GradeB_count_full_ledger = 48
GradeAB_count_full_ledger = 108
GradeB_count_GCERT18_eval_universe = 40
GradeAB_count_GCERT18_eval_universe = 86
TopK64_GradeB_count = 22
TopK64_GradeAB_count = 57
TopK64_GradeAB_precision = 0.890625
LegalTopK64_GradeAB_precision = 0.046875
universe_consistency_pass = 1
grade_definition_hash_match = 1
split_definition_hash_match = 1
outcome_field_used_by_certificate_count = 2
outcome_fields_used_by_legacy_gcert18 = V_integrated,risk_score
future_feature_used_count = 0
legacy_gcert18_legal_commit_time_pass = 0
grade_scope_audit_pass = 0
```

判断：P1 是本轮 terminal blocker。TopK64 count 没有超过 eval universe positive count，但 legacy GCERT18 的分数不是 legal commit-time certificate。

## 5. P2 rank / calibration dissection

Artifact：

```text
p2_gcert18_rank_calibration_dissection.csv
```

Summary：

```text
legacy_TopK64_GradeAB_precision = 0.890625
legacy_TopK64_V_integrated_LCB = 0.1650359732307112
legacy_TopK64_h240_longrisk_UCB = 0.0
legacy_ECE = 0.6217805447724831
legacy_uses_outcome_derived_score = 1

legal_TopK64_GradeAB_precision = 0.046875
legal_TopK64_V_integrated_LCB = -0.5602833203663737
legal_TopK64_h240_longrisk_UCB = 0.6991203985679182
legal_ECE = 0.6315531094435708

negative_control_TopK64_GradeAB_precision = 0.0
rank_signal_legacy_diagnostic_pass = 1
ranking_pass = 0
calibration_pass = 0
rank_calibration_pass = 0
```

判断：H1/H2 不成立于 official 层。legacy ranking 确实强，但它是 outcome diagnostic；legal surrogate ranking 弱且风险高。

## 6. P3 multi-grade target density

Artifact：

```text
p3_multigrade_target_density_audit.csv
```

Summary：

```text
target_candidate_count = 7
weak_official_target_count = 0
strong_target_count = 0
best_target_id = T_rank_legacy_GCERT18_TopK87_diagnostic
best_accepted_count = 87
best_coverage = 0.030250347705146036
best_V_integrated_LCB = 0.10620831245148316
best_h240_longrisk_UCB = 0.0
target_density_pass = 0
```

判断：legacy TopK87 能构成看起来可用的 diagnostic target，但它来自 outcome-derived rank。去掉这个后，A/B/C/T5 组合没有 official target density closure。

## 7. P4/P5 cover-memory 与 signal-reservoir

P4 artifact：

```text
p4_cover_memory_failure_anatomy.csv
```

P4 summary：

```text
corr_cover_collapse_longrisk = 0.09170759886071153
effect_size_cover_positive_vs_longrisk_abs = 0.3074931657698947
APGR_longrisk_memory_fail_rate = 0.27735368956743
GradeB_memory_fail_rate = 0.02531645569620253
forget_risk_UCB_generated = 0.12212837860977424
cover_blocker_confirmed = 0
memory_blocker_confirmed = 1
```

P5 artifact：

```text
p5_signal_reservoir_audit_v5.csv
```

P5 summary：

```text
best_feature_id = old_family_signal_score
best_TopK64_GradeAB_precision = 0.296875
best_TopK64_longrisk = 0.390625
best_TopK64_V_integrated_LCB = -0.48271869313319804
best_reservoir_leak = 0.09992041255839938
best_control_transfer_improvement = -0.3312589558576292
signal_reservoir_audit_pass = 0
```

判断：memory failure 与 APGR long-risk 强相关；signal/reservoir 单独不能形成 deployable TopK。

## 8. P6 existing-action controller

Artifact：

```text
p6_existing_action_rank_controller_candidate.csv
```

Summary：

```text
controller_candidate_count = 7
best_controller_id = C1-GCERT18-rank-only-legacy-diagnostic
best_heldout_accepted_count = 64
best_coverage = 0.022253129346314324
best_V_integrated_LCB = 0.16207759868285035
best_h240_longrisk_UCB = 0.0
negative_control_TopK_GradeAB_precision = 0.0
existing_action_controller_pass = 0
```

判断：best controller 仍是 legacy diagnostic，且 coverage 低于 official `[0.03, 0.15]` 下界；不能 selected。

## 9. P7-P9 APGC primitive

P7 artifact：

```text
p7_apgc_cover_memory_primitive_implementation.csv
```

P7 summary：

```text
primitive_count = 8
generated_action_count = 512
source_action_count = 64
payload_hash_missing = 0
certificate_hash_missing = 0
action_apply_linf_max = 0.0
negative_control_pass = 0
apgc_implementation_pass = 1
```

P8 artifacts：

```text
p8_apgc_branch_horizon_smoke.csv
apgc_branch_horizon_outcome_trace_v9560.csv
```

P8 summary：

```text
expected_rows = 12288
actual_rows = 12288
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
secondary_delta_completion_rate = 1
rows_per_sec = 20.078808122802613
wallclock_sec = 611.9885166911408
unresolved_exception_count = 0
duplicate_row_count = 0
label_exclusivity_violation = 0
quality_audit_pass = 1
```

P9 artifact：

```text
p9_apgc_outcome_geometry_pass.csv
```

P9 summary：

```text
best_primitive_id = APGC8-NegativeControlShuffledPayload
best_GradeB_precision = 0.046875
best_OfficialGeo_precision = 0.125
best_V_integrated_LCB = -0.3361628959034951
best_h240_longrisk = 0.78125
best_new_positive_created_rate = 0.046875
best_longrisk_created_rate = 0.78125
best_Damage_V_integrated_LCB = -0.22485958350151117
apgc_weak_pass = 0
apgc_strong_pass = 0
```

判断：APGC engineering/materializer 过，但 outcome 失败。best primitive 是 negative control，说明 APGC1-APGC7 没有产生真实更好的 cover-memory frontier。

## 10. P10 certificate v4

Artifact：

```text
p10_geometry_certificate_v4.csv
```

Summary：

```text
certificate_count = 8
best_certificate_id = GCERT25-GCERT18RankOnly
best_uses_outcome_derived_score = 1
best_AUC_GradeAB = 0.9914807162534435
best_TopK64_precision_GradeAB = 0.90625
best_TopK64_V_integrated_LCB = 0.1678302289887135
best_TopK64_h240_longrisk_UCB = 0.0
best_ECE = 0.6259564563021709
ranking_certificate_pass = 0
calibration_certificate_pass = 0
official_certificate_pass = 0
```

Legal candidate reference：

```text
GCERT27-GCERT18ConformalRiskBound:
  uses_outcome_derived_score = 0
  AUC_GradeAB = 0.9102100550964187
  TopK64_GradeAB = 0.125
  TopK64_V_integrated_LCB = -0.2732468908819606
  TopK64_h240_longrisk_UCB = 0.0
  ECE = 0.5618282212651694
```

判断：certificate v4 复现了强 diagnostic rank，但 official certificate 仍失败。高 AUC 不能替代 legal score、TopK value/risk 与 calibration gate。

## 11. P11-P15 boundary

| artifact | status / reason |
|---|---|
| `p11_minimal_geometry_controller.csv` | `not_run`, `P1_grade_scope_or_legacy_gcert18_outcome_field_failed` |
| `p12_selected_runtime.csv` | `not_run`, `P11_controller_not_selected` |
| `p13_leaveout_boundary.csv` | `not_run`, `P12_runtime_not_selected` |
| `p14_official_paired_replay_boundary.csv` | `not_run`, `P13_leaveout_not_open` |
| `p15_short_full_training_boundary.csv` | `not_run`, `P14_paired_replay_not_open` |

判断：没有把 legacy rank diagnostic、APGC smoke、certificate v4 diagnostic 或 Base-Acc Sentinel 写成 official system pass。

## 12. Base-Acc Sentinel

Artifact：

```text
base_acc_sentinel_v9560.csv
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

## 13. No-fake audit

```text
rows_checked = 16397
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
v9550_boundary_pass = 1
grade_scope_audit_pass = 0
rank_calibration_pass = 0
target_density_pass = 0
cover_memory_anatomy_pass = 1
signal_reservoir_audit_pass = 0
existing_action_controller_pass = 0
apgc_implementation_pass = 1
apgc_branch_horizon_pass = 1
apgc_weak_pass = 0
official_certificate_pass = 0
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
route = R1-GradeScopeInconsistent
F0_boundary_regression = 0
F1_grade_scope_inconsistent = 1
F2_ranking_strong_calibration_weak = 0
F3_clean_geometry_target_too_sparse = 0
F4_cover_memory_primary_blocker = 0
F5_apgc_generated_frontier_fail = 0
F6_certificate_controller_fail = 0
F7_runtime_fail = 0
F8_system_not_official = 1
F9_base_acc_catastrophic = 0
primary_blocker = legacy_gcert18_uses_outcome_derived_score
```

## 14. Hash

| artifact | SHA256 |
|---|---|
| plan | `9d4cb8e9393ae344df5b9b2d404adec699c9c7191fe54cee12a7d1ec18a13e6d` |
| runner | `2cde23d7f51bffad5f106ec99b4e9c2a6cadfb1c4c361aef93dae457fb9bf32d` |
| run manifest | `a9fa4efe9f7c9d7157adaf773db322089ee1e984f0568cdcb7e86e76e4549a86` |
| route | `c6995a9b73a43806c18e07085b0f05d21ffc8100172d0e1dd116cb9a0e074814` |
| P0 boundary | `9dfb007f7a8af845744cb9f4e77b3b4beefa0a70733ae23b7f12513cc2da45e3` |
| P1 grade scope | `dc0e86409924475c9e6d1da0bb3e23a56713d5f470cb829ced6d456333353085` |
| P2 rank calibration | `afb5d5e7a7cba05ce19369020835d349b2ae7a02d119817dfadb976af76d2687` |
| P3 target density | `5532bb954e366ff310757f3f8f748cfea5842d5786ccb554b4e394376a27592e` |
| P4 cover memory | `d8912a942ffd6395aa1d4edbf19a069a0a7801147327f97079b99d20b35bfa96` |
| P5 signal reservoir | `4b6627b9383d38cb84d28090ed85409ce7c30365feae77001568aa9d2e1acc6c` |
| P6 existing controller | `046ed520520ee6db2888c2e8ecb03c769dbde5148d6be042210969c4d9597318` |
| P7 APGC implementation | `f08fc2d30ec90440204d5799942c51a3a8b2149ac9c024656dff91a48757dd5e` |
| P8 APGC smoke | `2338174eae1e4bb8c20faeca14ab11e8a67fdafb4d331f027627411058b8ac87` |
| APGC trace | `d66ba1c4cfc3a8cfde3f1eae9b63e4834755b9d351d52c0ff8c15596f8b05eed` |
| P9 APGC outcome | `8c7230b89e96946467fef603d84a5b6be5f5e5d963f317640f95a5d174cc3cb7` |
| P10 certificate | `567b850eda7f6afca3c46af52598a228ebddd45fc458fb1bbef0d97008fa67f5` |
| P11 controller | `c7f4ba41af0c322198d7d35b1b1bdc3caddcc71d31ec8f17a06988c014fe1c46` |
| P12 runtime | `fb0b0a17a4285fb5031eff3c054862f64f5e75a0d83079a3142f4fc3a1779c17` |
| P13 leaveout | `0f8727aa5ce4d80d4eeebe77c91f088e5760048b225a52ac77e5d48b314d7453` |
| P14 paired replay | `74c5c01d526c4b91c2bb50f2c726addf334c8eaffe9ccbaf18dbedd3c0381af0` |
| P15 short/full | `f977715517bbd32e6b58e0f36507e2173f363b00aa528ac079061a97b9b53d5c` |
| Base-Acc Sentinel | `185cb86c672c7452f984585dd3fa61b3f6656ebc5f523c43d10c3e28a75191ff` |
| dashboard | `eef752600a771a6cbbd81412680f584567f623f89df957b73675f2d598f15cfe` |
| no-fake audit | `dbc9ee60061b04e8bf011d9405b043216093a763ea63f5a63e8d56185300f274` |
| contract audit | `6b297dfe1690d285fcc8cb10cbeff5a33222394520948d252cbf5ecd07668e02` |
| failure table | `1c84462e199847b5f99ddfd59b61d1168f8fc37d1d770cbf04a9c8c133709c21` |

## 15. 最终分析结论

v9.5.6 的真实推进是：

```text
v9.5.5:
  T5-like target 真实但太稀疏；
  GCERT18 TopK diagnostic 很强；
  APGR 不能生成新的正例；
  GCERT18 calibration/ECE 不过。

v9.5.6:
  先查 Grade / GCERT universe；
  证明 legacy GCERT18 TopK 强信号不是 universe count 越界；
  但进一步发现 legacy GCERT18 score 使用 V_integrated / risk_score outcome-derived fields；
  legal commit-time surrogate 不能复现 TopK；
  APGC1-APGC8 工程链路和 branch-horizon materializer 完整闭合；
  但 APGC outcome 继续 value-negative / high-longrisk，且 best 是 negative control；
  certificate v4 仍只有 outcome-derived diagnostic 强，legal certificate/controller 未打开。
```

机制判断：

1. H1 失败于 official 层：legacy GCERT18 ranking signal 强，但不是 legal commit-time score；`outcome_field_used_by_certificate_count = 2`。
2. H2 未成立：legacy ECE = `0.6218`，legal surrogate ECE = `0.6316`，且 legal TopK64 GradeAB precision 只有 `0.046875`。
3. H3 未成立：A/B/C/T5 组合没有 official density pass；唯一达 coverage 的 TopK87 是 legacy outcome-derived diagnostic。
4. H4 部分成立：memory blocker 成立，APGR longrisk memory fail rate 约为 GradeB 的 `10.95x`；cover blocker 未成立。
5. H5 未成立：APGC1-APGC8 implementation 与 materializer 过，但没有产生 GradeB / safe-C / official generated frontier。
6. H6 未打开：P1/P10/P11 不过，selected runtime 不能打开。
7. P13-P15 未打开：没有 controller/runtime，就不能打开 leaveout、official paired replay 或 short/full training。
8. Base-Acc Sentinel 继续健康，但没有用于 controller，也不是 functional success。

最终一句话：

> v9.5.6 真实执行后停在 `R1-GradeScopeInconsistent`：v9.5.5 的 GCERT18 TopK 强信号不是简单口径越界，而是依赖 `V_integrated/risk_score` 这类 outcome-derived 字段；legal commit-time surrogate 无法复现该 ranking，APGC 虽完整落盘但未生成可用 frontier，因此 strict PureKAN functional 仍未成功。
