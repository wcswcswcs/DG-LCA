# DG-KAN v9.5.5 Trainable Geometry / Signal-Reservoir Primitive / Parallel Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.5.5_TrainableGeometry_SignalReservoirPrimitive_ParallelClosure_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 T5-like target、multi-grade diagnostic、high-AUC certificate diagnostic、APGR implementation/smoke、runtime boundary 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R2a-CleanGeometryTargetTooSparse
base_candidate = LQ-t2-h256
success_v9550_strict_purekan_functional = False
success_v9550_full_functional = False
success_v9550_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9550_trainable_geometry_signal_reservoir_primitive_first_20260515T050000Z/
```

核心结论：

1. P0 复现 v9.5.4 boundary：source route = `R3-LegalGeometrySignalAbsent`，system pass = `0`，no-fake/no-proxy = `1 / 1`。
2. P1 Trainable Geometry Ledger v2 完整落盘：rows = `4412`，canonical AP0 = `2876`，generated APY/APG/APGS = `512 / 512 / 512`。
3. P1 ledger audit 过：missing / NaN / Inf = `0 / 0 / 0`，future outcome leakage count = `0`，feature cost recorded = `1`。
4. P2 T5 anatomy 确认 clean geometry target 真实但太薄：T5 action count = `57`，coverage = `0.019819193324061197`，V_integrated LCB = `0.1960342568901401`，h240 longrisk/bad/null = `0.0 / 0.0 / 0.0`。
5. P2 official target candidate count = `0`，weak target variant count = `7`；best target = `T5-h80-stricter`，count = `55`，coverage = `0.01912378303198887`。
6. P3 multi-grade label 完成：Grade A/B/C/D/E count = `40 / 39 / 18 / 89 / 2690`；Grade A official ready = `0`，Grade B diagnostic ready = `0`。
7. P4 signal upper-bound v4 未过：best = `SNR-old-family`，AUC_GradeB = `0.9302100351642583`，TopK64 GradeB precision = `0.296875`，TopK64 V_integrated LCB = `-0.48271869313319804`，TopK64 h240 longrisk = `0.390625`。
8. P4 failure class = high AUC / low TopK，dominant reason = `high_score_high_risk`。
9. P5 reservoir/noise rejection 未过：TopK64 signal channel score = `1.0160237990338536`，reservoir leak = `0.0`，但 TopK64 GradeB precision = `0.078125`，control transfer improvement = `-0.35061125748652555`。
10. P6 cover stability 未过：cover collapse count = `92`，basis effective rank delta mean = `-0.5057796221440767`，hard-tail cover entropy delta mean = `-9.90125173852573`。
11. P7 curvature/fixed-point audit 过：curvature delta UCB = `0.0655680861250561`，jacobian spectral delta UCB = `0.0655680861250561`，CEp99 delta UCB = `-0.1201715130761731`。
12. P8 memory anti-forgetting 未过：old family fail count = `4249`，old stratum fail count = `1216`，forget risk UCB = `0.25856111245831553`。
13. P9 APGS autopsy 完成：dominant damage mode = `horizon_longrisk`，best APGS new positive created rate = `0.015625`，longrisk created rate = `0.765625`，Damage V_integrated LCB = `-1.2417936990052212`。
14. P10 APGR1-APGR8 implementation 过：generated actions = `512`，payload/certificate hash missing = `0 / 0`，action apply L∞ max = `0.0`，APGR8 negative control pass = `0`。
15. P11 APGR branch-horizon smoke 完整落盘：expected/actual rows = `12288 / 12288`，unresolved exception = `0`，quality audit pass = `1`，rows/sec = `20.86568389283173`。
16. P12 APGR outcome 未过：best = `APGR5-MemoryGuardedResidualUpdate`，GradeB precision = `0.046875`，OfficialGeo precision = `0.0`，V_integrated LCB = `-0.8658484647235336`，h240 longrisk = `0.78125`。
17. P13 geometry certificate v3 未过：best = `GCERT18-TransferRiskBalancedCertificate`，AUC_GradeB = `0.992657726818137`，TopK64 GradeB precision = `0.890625`，TopK64 V_integrated LCB = `0.1650359732307112`，TopK64 longrisk/bad/null = `0.0 / 0.0 / 0.0`，但 ECE = `0.6217805447724831`，certificate pass = `0`。
18. P14-P17 gate-blocked：没有 source controller、selected runtime、conditional paired replay 或 short/full training。
19. Base-Acc Sentinel 复用前序真实 sentinel：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
20. 当前 primary blocker：`clean_geometry_target_too_sparse`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9550_trainable_geometry_signal_reservoir_primitive.py` | v9.5.5 runner；读取 v9.5.4 artifacts，构建 Trainable Geometry Ledger v2，执行 T5 anatomy、multi-grade label、signal/reservoir/cover/curvature/memory audit、APGS autopsy、APGR primitive、geometry certificate 与 system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9550_trainable_geometry_signal_reservoir_primitive.py
```

正式运行：

```bash
python experiments/run_v9550_trainable_geometry_signal_reservoir_primitive.py \
  --out-dir results/real_rerun_20260506/v9550_trainable_geometry_signal_reservoir_primitive_first_20260515T050000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --apgr-actions-per-primitive 64
```

运行结果：

```json
{
  "GradeA_count": 40,
  "T5_action_count": 57,
  "apgr_generated_actions": 512,
  "best_apgr_GradeB_precision": 0.046875,
  "best_apgr_primitive": "APGR5-MemoryGuardedResidualUpdate",
  "geometry_certificate_v3_pass": 0,
  "ledger_rows": 4412,
  "out_dir": "results/real_rerun_20260506/v9550_trainable_geometry_signal_reservoir_primitive_first_20260515T050000Z",
  "route": "R2a-CleanGeometryTargetTooSparse",
  "system_legal_controller_pass": 0
}
```

说明：第一次正式尝试的 certificate diagnostic 中发现 `GCERT21` 直接使用了 outcome-derived `GradeAB` 标签作为分数。该结果没有作为正式结论使用；随后修正为只使用 commit-time signal/cover/curvature/memory/cost 分数，并对同一正式 out-dir 执行 `--fresh` 重跑。本文和最终 artifact 只对应修正后的 fresh run。

## 2. Route

`route_decision_v9550.json` 摘要：

```json
{
  "route": "R2a-CleanGeometryTargetTooSparse",
  "source_route_v9540": "R3-LegalGeometrySignalAbsent",
  "trainable_geometry_ledger_pass": 1,
  "ledger_rows": 4412,
  "T5_action_count": 57,
  "T5_coverage": 0.019819193324061197,
  "T5_V_integrated_lcb": 0.1960342568901401,
  "T5_h240_longrisk": 0.0,
  "T5_bad_rate": 0.0,
  "T5_null_rate": 0.0,
  "official_target_candidate_count": 0,
  "weak_target_candidate_count": 7,
  "GradeA_count": 40,
  "GradeB_count": 39,
  "raw_signal_upper_bound_pass": 0,
  "best_signal_feature_id": "SNR-old-family",
  "best_signal_AUC_GradeB": 0.9302100351642583,
  "best_signal_TopK64_precision_GradeB": 0.296875,
  "high_AUC_low_TopK_failure": 1,
  "signal_failure_reason": "high_score_high_risk",
  "reservoir_noise_rejection_pass": 0,
  "cover_stability_pass": 0,
  "curvature_fixed_point_pass": 1,
  "memory_antiforgetting_pass": 0,
  "apgs_dominant_damage_mode": "horizon_longrisk",
  "apgr_implementation_pass": 1,
  "apgr_branch_horizon_pass": 1,
  "apgr_weak_pass": 0,
  "best_apgr_primitive": "APGR5-MemoryGuardedResidualUpdate",
  "best_apgr_GradeB_precision": 0.046875,
  "best_apgr_V_integrated_lcb": -0.8658484647235336,
  "best_apgr_h240_longrisk": 0.78125,
  "geometry_certificate_v3_pass": 0,
  "best_certificate_id": "GCERT18-TransferRiskBalancedCertificate",
  "best_certificate_AUC_GradeB": 0.992657726818137,
  "best_certificate_TopK64_precision_GradeB": 0.890625,
  "best_certificate_ECE": 0.6217805447724831,
  "system_legal_controller_pass": 0,
  "primary_blocker": "clean_geometry_target_too_sparse"
}
```

判断：v9.5.5 的 primary blocker 回到 target density：T5-like clean region 真实且干净，但仍不到 official 支持度。后续 signal/certificate 有强 diagnostic signal，APGR 工程链路也闭合，但不能越过 P2/P12/P13 gates。

## 3. P0 v9.5.4 boundary

Artifact：

```text
p0_v9540_boundary_reproduction.csv
```

Summary：

```text
source_route_v9540 = R3-LegalGeometrySignalAbsent
ledger_rows = 3900
canonical_ap0_rows = 2876
generated_apy/apg_rows = 512 / 512
weak_geometry_target_count = 32
official_geometry_target_count = 0
selected_target_id = T5-RelaxedV80NonNegative
selected_action_count = 57
selected_coverage = 0.019819193324061197
best_apgs_primitive = APGS7-SymmetricBoundaryResidualUpdate
best_apgs_V_integrated_lcb = -0.9218810936698074
best_certificate_id = GCERT16-OfficialMinimalGeometryCertificate
best_certificate_AUC = 0.9977771191464138
best_certificate_TopK64_precision = 0.21875
system_legal_controller_pass = 0
p0_pass = 1
```

判断：P0 pass。v9.5.5 没有跳过 v9.5.4 的 `LegalGeometrySignalAbsent` boundary。

## 4. P1 Trainable Geometry Ledger v2

Artifacts：

```text
trainable_geometry_ledger_v9550.csv
p1_trainable_geometry_ledger_v2_audit.csv
```

Summary：

```text
row_count_total = 4412
canonical_ap0_rows = 2876
generated_apy_rows = 512
generated_apg_rows = 512
generated_apgs_rows = 512
missing_required_field_count = 0
nan_count = 0
inf_count = 0
identity_join_pass = 1
commit_time_field_count = 12
diagnostic_field_count = 7
future_outcome_leakage_count = 0
feature_cost_recorded = 1
trainable_geometry_ledger_pass = 1
```

判断：P1 pass。v9.5.5 将 APGS 也并入 Trainable Geometry Ledger v2，形成 AP0/APY/APG/APGS 统一 action table。

## 5. P2/P3 target 与 multi-grade label

P2 artifact：

```text
p2_t5_anatomy_target_density_audit.csv
```

P2 summary：

```text
target_variant_count = 10
T5_action_count = 57
T5_coverage = 0.019819193324061197
T5_V_integrated_lcb = 0.1960342568901401
T5_h240_longrisk = 0.0
T5_bad_rate = 0.0
T5_null_rate = 0.0
best_target_id = T5-h80-stricter
best_action_count = 55
best_coverage = 0.01912378303198887
best_V_integrated_lcb = 0.1986248258448371
official_target_candidate_count = 0
weak_target_candidate_count = 7
clean_target_too_sparse = 1
```

P3 artifact：

```text
p3_multigrade_geometry_label.csv
```

P3 summary：

```text
GradeA_count = 40
GradeB_count = 39
GradeC_count = 18
GradeD_count = 89
GradeE_count = 2690
GradeA_official_ready = 0
GradeB_diagnostic_ready = 0
```

判断：T5-like target 不是假的，且 outcome 很干净；但支持度仍不够。multi-grade label 也没有达到 Grade A/B count >= 87 的下一层目标。

## 6. P4/P5 signal 与 reservoir

P4 artifact：

```text
p4_signal_channel_raw_upper_bound_v4.csv
```

P4 summary：

```text
feature_count = 8
best_feature_id = SNR-old-family
best_AUC_GradeB = 0.9302100351642583
best_TopK64_precision_GradeB = 0.296875
best_TopK64_V_integrated_lcb = -0.48271869313319804
best_TopK64_h240_longrisk = 0.390625
best_calibration_ece = 0.6203814147108345
raw_signal_upper_bound_pass = 0
high_AUC_low_TopK_failure = 1
dominant_failure_reason = high_score_high_risk
```

Selected feature rows：

```text
SNR-old-family:
  AUC_GradeB = 0.9302100351642583
  TopK64_precision_GradeB = 0.296875
  TopK64_V_integrated_lcb = -0.48271869313319804
  TopK64_h240_longrisk = 0.390625

leave-one-out-transfer:
  AUC_GradeB = 0.75022967646167
  TopK64_precision_GradeB = 0.125
  TopK64_V_integrated_lcb = -0.260665499652037
  TopK64_h240_longrisk = 0.09375
```

P5 artifact：

```text
p5_reservoir_noise_rejection_audit.csv
```

P5 summary：

```text
train_only_improvement_topk64 = -0.613242644987622
control_transfer_improvement_topk64 = -0.35061125748652555
train_to_control_transfer_gap_topk64 = 0.26263138750109644
reservoir_leak_score_topk64 = 0.0
signal_channel_score_topk64 = 1.0160237990338536
shuffle_payload_pass_rate = 1.0
TopK64_precision_GradeB = 0.078125
reservoir_noise_rejection_pass = 0
```

判断：v9.5.5 明确了 v9.5.4 的“高 AUC 但 TopK 不行”：signal ranking 能排序一部分正例，但高分区域仍带 high risk / negative value，不能作为 deployable signal。

## 7. P6-P8 cover / curvature / memory

P6 artifact：

```text
p6_cover_stability_audit.csv
```

P6 summary：

```text
cover_collapse_count = 92
basis_effective_rank_delta_mean = -0.5057796221440767
hard_tail_cover_entropy_delta_mean = -9.90125173852573
per_class_max_cover_gap = 0.4122030419613547
cover_stability_pass = 0
```

P7 artifact：

```text
p7_curvature_fixed_point_stability_audit.csv
```

P7 summary：

```text
curvature_delta_ucb = 0.0655680861250561
HVP_norm_proxy_ucb = 0.14239633322071588
jacobian_spectral_proxy_delta_ucb = 0.0655680861250561
local_Lipschitz_proxy_delta_ucb = 0.0655680861250561
fixed_point_residual_delta_mean = 0.0
CEp99_delta_ucb = -0.1201715130761731
margin_p10_delta_lcb = 0.37332414823228677
curvature_fixed_point_pass = 1
```

P8 artifact：

```text
p8_memory_antiforgetting_audit_v3.csv
```

P8 summary：

```text
old_family_margin_delta_lcb = -0.09049638936041043
old_family_CE_delta_ucb = 0.10342444498332624
old_family_fail_count = 4249
old_stratum_fail_count = 1216
forget_risk_ucb = 0.25856111245831553
memory_antiforgetting_pass = 0
```

判断：curvature/fixed-point 这一侧相对健康，但 cover 与 memory 两个 guard 都未过，解释了为什么单靠 signal/certificate 很难稳定选中可部署动作。

## 8. P9 APGS failure autopsy

Artifact：

```text
p9_apgs_failure_autopsy_trainable_geometry.csv
```

Summary：

```text
primitive_count = 8
best_primitive_id = APGS7-SymmetricBoundaryResidualUpdate
best_new_positive_created_rate = 0.015625
best_longrisk_created_rate = 0.765625
best_Damage_V_integrated_lcb = -1.2417936990052212
dominant_damage_mode = horizon_longrisk
apgs_failure_autopsy_pass = 1
```

判断：APGS 的主失败不是 payload/apply/materializer，而是生成动作引入高 long-risk；这支持 v9.5.5 不再继续微调 APGS，而转向 APGR 的计划。

## 9. P10-P12 APGR primitive

P10 artifact：

```text
p10_apgr_signal_reservoir_primitive_implementation.csv
```

P10 summary：

```text
primitive_count = 8
generated_action_count = 512
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_error_linf_max = 0.0
APGR8_negative_control_pass = 0
apgr_implementation_pass = 1
```

P11 artifacts：

```text
p11_apgr_branch_horizon_smoke.csv
apgr_branch_horizon_outcome_trace_v9550.csv
```

P11 summary：

```text
expected_rows = 12288
actual_rows = 12288
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
secondary_delta_completion_rate = 1
rows_per_sec = 20.86568389283173
wallclock_sec = 588.9095254731365
unresolved_exception_count = 0
quality_audit_pass = 1
```

P12 artifact：

```text
p12_apgr_outcome_geometry_pass.csv
```

P12 summary：

```text
best_primitive_id = APGR5-MemoryGuardedResidualUpdate
best_GradeB_precision = 0.046875
best_OfficialGeo_precision = 0.0
best_OutcomeGood_precision = 0.0
best_V_integrated_lcb = -0.8658484647235336
best_h240_longrisk = 0.78125
best_bad_rate = 0.078125
best_null_rate = 0.09375
apgr_weak_pass = 0
apgr_strong_pass = 0
```

Per primitive：

| primitive | GradeB precision | V_integrated LCB | h240 longrisk | bad | null |
|---|---:|---:|---:|---:|---:|
| APGR1-GroupSNRProjectedEdgeUpdate | 0.015625 | -0.6913772426342403 | 0.765625 | 0.078125 | 0.125 |
| APGR2-SignalReservoirMaskedUpdate | 0.0 | -0.7634637728980411 | 0.75 | 0.09375 | 0.0625 |
| APGR3-CoverBalancedEdgeUpdate | 0.0 | -0.6313754768429957 | 0.734375 | 0.109375 | 0.109375 |
| APGR4-CurvatureTrustRegionUpdate | 0.015625 | -0.6933167751453171 | 0.71875 | 0.046875 | 0.078125 |
| APGR5-MemoryGuardedResidualUpdate | 0.046875 | -0.8658484647235336 | 0.78125 | 0.078125 | 0.09375 |
| APGR6-SymmetricBoundaryKANUpdate | 0.015625 | -0.6996537660873948 | 0.78125 | 0.109375 | 0.078125 |
| APGR7-GeoScoreConstrainedBlend | 0.015625 | -0.6958038817788107 | 0.84375 | 0.078125 | 0.0625 |
| APGR8-NegativeControlShuffledPayload | 0.0 | -0.7744890808931708 | 0.765625 | 0.0625 | 0.109375 |

判断：APGR implementation/smoke 完整闭合，但 generated outcomes 仍 value-negative、高 long-risk；APGR 未创造新的 positive geometry frontier。

## 10. P13 geometry certificate v3

Artifact：

```text
p13_geometry_certificate_v3.csv
```

Summary：

```text
certificate_count = 8
best_certificate_id = GCERT18-TransferRiskBalancedCertificate
best_AUC_GradeB = 0.992657726818137
best_TopK64_precision_GradeB = 0.890625
best_TopK64_V_integrated_lcb = 0.1650359732307112
best_TopK64_h240_longrisk = 0.0
best_TopK64_bad = 0.0
best_TopK64_null = 0.0
best_ECE = 0.6217805447724831
geometry_certificate_v3_pass = 0
```

Selected certificate rows：

| certificate | AUC GradeB | TopK64 GradeB | TopK64 V LCB | TopK64 longrisk | ECE / pass |
|---|---:|---:|---:|---:|---:|
| GCERT18-TransferRiskBalancedCertificate | 0.992657726818137 | 0.890625 | 0.1650359732307112 | 0.0 | 0.6217805447724831 / 0 |
| GCERT20-TrainableGeometryMinimalCertificate | 0.9809100897271562 | 0.734375 | 0.11009567199488746 | 0.0 | failed |
| GCERT24-ControllerCostBoundCertificate | 0.9548793543025369 | 0.609375 | 0.18655581376058877 | 0.140625 | failed |

判断：P13 是强 diagnostic 进展：不使用 outcome-derived grade 作为分数后，GCERT18 仍能在 TopK64 上抓到很多 GradeB，且 V/risk/bad/null 很干净；但 ECE 远高于 `0.10`，因此不能 official certificate，也不能打开 controller。

## 11. P14-P17 boundary

P14：

```text
p14_minimal_geometry_controller.csv = not_run
reason = P12_or_P13_gate_failed
source_controller_pass = 0
```

P15：

```text
p15_selected_runtime.csv = not_run
reason = P14_controller_not_selected
selected_runtime_pass = 0
```

P16-P17：

| artifact | status / reason |
|---|---|
| `p16_conditional_paired_replay_boundary.csv` | `not_run`, `P14_or_P15_not_passed` |
| `p17_short_full_training_boundary.csv` | `not_run`, `P16_paired_replay_not_open` |

判断：没有把 T5-like target、high-AUC certificate diagnostic、APGR smoke 或 Base-Acc Sentinel 写成 official controller/runtime/downstream pass。

## 12. Base-Acc Sentinel

Artifact：

```text
base_acc_sentinel_v9550.csv
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
rows_checked = 17414
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
v9540_boundary_pass = 1
trainable_geometry_ledger_pass = 1
t5_anatomy_pass = 1
multigrade_label_pass = 1
raw_signal_upper_bound_pass = 0
reservoir_noise_rejection_pass = 0
cover_stability_pass = 0
curvature_fixed_point_pass = 1
memory_antiforgetting_pass = 0
apgs_failure_autopsy_pass = 1
apgr_implementation_pass = 1
apgr_branch_horizon_pass = 1
apgr_weak_pass = 0
geometry_certificate_v3_pass = 0
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
route = R2a-CleanGeometryTargetTooSparse
F0_boundary_regression = 0
F1_geometry_ledger_fail = 0
F2_clean_geometry_target_too_sparse = 1
F3_no_deployable_signal = 0
F4_high_auc_low_topk = 0
F5_cover_curvature_memory_damage = 0
F6_apgr_implementation_fail = 0
F7_apgr_value_fail = 0
F8_certificate_fail = 0
F9_controller_runtime_blocked = 1
F10_base_acc_catastrophic = 0
primary_blocker = clean_geometry_target_too_sparse
```

## 14. Hash

| artifact | SHA256 |
|---|---|
| plan | `35cd3455f79339127730023c0016641465286960b70ef69f1cd9623bbe77853a` |
| runner | `3d2b14ebc416a2a05d3dbdb97d6d1bb8854a13af5a2dd5deb219bc1a6cf6a532` |
| run manifest | `defbcb2e3c18aadef396daacb2736b23548a864f94bc75969203df4b3119d262` |
| route | `054f92fbdfed86cd9de6cab3bdb4d14d8be8774ef16218ddfe6b1d62495618c4` |
| Trainable Geometry Ledger | `18c2ae059af0c46263c0eef342caecce6b14c3eeb5b0ed5325d55797f3cbc4ef` |
| P0 boundary | `8bf8d45649920050e408970f2b9a713a6b579d30a0161016bf7ca9bd1d774f8f` |
| P1 ledger audit | `fba5097c08e47a7a9bb0437a4d5a082af59ab2f380736018eb45f9b6a19aea4c` |
| P2 T5 anatomy | `93c2d5d6df7450b85f71558374bd4663f313f0b14c51710a82be7ddc17556a62` |
| P3 multigrade | `69c616fa07e0acb20ced197959d26565d1c28008dc0d2729e7e45062bb4ac356` |
| P4 signal | `22fabf00cc9f5e713352c363b288c4f64d89c623eeb632032205610a078783d4` |
| P5 reservoir | `36fdb3b9b860d1c29f6ff55c7b6c3d4a0f4e9d63b776d83e3e11ddc9fd154726` |
| P6 cover | `cca185eecd52286a03fdfdc619db4bb105d103fbe2f71174969942a4ac48f3ba` |
| P7 curvature | `a02a75530c046e66d418b3e84ca7951d3678e66af930121fb7731c26596ed65f` |
| P8 memory | `f9c55ebb2ee619a7fbfeabcc6df4971825d3c602e416c8b913ebfe574326637a` |
| P9 APGS autopsy | `da78136970edb3fe8a2255be7f1ffaa16db9493857bf557d09763227081a81a9` |
| P10 APGR implementation | `1b899f9eafad77d76d66f07478179f634402181fb315322442cdf34b4993c87e` |
| P11 APGR smoke | `48d5c730dde25f4837c73ac3b2d539368aa02d83bc93342efd6edd445c1cc390` |
| APGR trace | `33603d0869c23a089e21524cd881fd3501a60ce803ab4fd4354c669b027f89b0` |
| P12 APGR outcome | `e7c555468633c12e7a27c897c72cd73a78b29011f811e3d420d8bd360813f97c` |
| P13 certificate | `d11449dd2e2c5c9348f00ea8e0d0c195c487660918e19895dfca12b2b9006f13` |
| P14 controller | `c28028da24ac061aa24c2755b616bc6a747751d7c42549f30b7fd2aabd62f0f1` |
| P15 runtime | `d48debae2c824da687bc32082e212a871b27c067ad82f08ceebc0c063db54716` |
| P16 paired replay | `d4afd0d46eba9bfb233899fc75fb7b31cecd28a2284ed9f9a0ae9392979050df` |
| P17 short/full | `8503954cfa5150cac6ce14c9b1e6b615c20cd2b469ae458d53a9e1f8f5edd9a9` |
| Base-Acc Sentinel | `85decd8d26942406a42dd206ac890e817f86a32a2e6da5b578752fbe6e898f56` |
| dashboard | `a85be50c3c758811ac5073ba911f2a96a5c37116e7dec73682ddec8c4831bb8e` |
| no-fake audit | `c165816bf67bbd318ae798baea362db9135255dec3d11a7164d65d94b9844e9b` |
| contract audit | `fb02396d9d24722ce60a9b5b7ac3156d740c41a2a3bc7dbff1ae6a8e9ffaa0c3` |
| failure table | `d2596e5bef1c3d813539ac39ab8de316d9541b726947a1548bf1f2b8d3eb6291` |

## 15. 最终分析结论

v9.5.5 的真实推进是：

```text
v9.5.4:
  relaxed geometry target 和 GeoScore diagnostic 存在；
  但 commit-time signal / APGS / certificate 未转成 controller。

v9.5.5:
  将 APGS 并入 Trainable Geometry Ledger v2；
  将 target 改为 multi-grade geometry label；
  明确 T5-like target 真实且干净，但 official density 不够；
  解释 high-AUC-low-TopK 的主因是 high-score high-risk；
  定位 APGS 主失败为 horizon long-risk；
  APGR1-APGR8 工程链路和 branch-horizon materializer 完整闭合；
  但 APGR generated actions 仍 value-negative / high-longrisk；
  GCERT18 有很强 TopK diagnostic，但 calibration/ECE 不过，不能 controller。
```

机制判断：

1. H1 成立：T5-like target 不是偶然，`57` 个动作 outcome 干净，但 coverage 只有 `0.0198`，不足 official。
2. H2 成立：SC7/GCERT 类信号不是完全假的；但 high AUC 无法转成可部署 TopK，主因是 high-score high-risk / calibration bad。
3. H3 部分成立：signal、cover、curvature、memory 必须组合；本轮 curvature 过，但 cover/memory 失败。
4. H4 成立：APGS 失败主因是 horizon long-risk，new positive created rate 极低。
5. H5 未成立：APGR1-APGR8 implementation 与 smoke 过，但没有任何 primitive 产生 value-positive / horizon-safe / GradeB-positive frontier。
6. H6 部分成立：raw signal 和 certificate 都显示高 AUC/TopK diagnostic，但 APGR 仍不能生成正例；下一步不能只继续 feature search。
7. P13 提供一个重要线索：GCERT18 的 TopK64 GradeB precision = `0.890625` 且 TopK64 risk/bad/null 很干净，但 ECE = `0.6218`，因此当前 certificate 是 ranking diagnostic，不是 calibrated controller。
8. P14-P17 未打开：没有 controller，就不能打开 selected runtime、official paired replay 或 short/full training。
9. Base-Acc Sentinel 继续健康，但没有用于 controller，也不是 functional success。

最终一句话：

> v9.5.5 真实执行后停在 `R2a-CleanGeometryTargetTooSparse`：Trainable Geometry Ledger v2、T5 anatomy、APGS failure autopsy 和 APGR branch-horizon smoke 都已闭合；clean geometry target 真实但太稀疏，APGR 仍未创造新的正例，GCERT18 虽有强 ranking diagnostic 但 calibration 失败，因此 strict PureKAN functional 仍未成功。
