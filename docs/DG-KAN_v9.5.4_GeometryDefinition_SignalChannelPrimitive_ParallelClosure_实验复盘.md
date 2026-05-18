# DG-KAN v9.5.4 Geometry Definition / Signal-Channel Primitive / Parallel Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.5.4_GeometryDefinition_SignalChannelPrimitive_ParallelClosure_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 relaxed geometry target、GeoScore diagnostic、APGS implementation/preflight、geometry certificate diagnostic、runtime boundary 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R3-LegalGeometrySignalAbsent
base_candidate = LQ-t2-h256
success_v9540_strict_purekan_functional = False
success_v9540_full_functional = False
success_v9540_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9540_geometry_definition_signal_channel_primitive_first_20260515T040000Z/
```

核心结论：

1. P0 复现 v9.5.3 boundary：route = `R2-NoGoodGeometryActionInCanonicalAP0`，GeometryCard pass = `1`，APG implementation/preflight pass = `1 / 1`，system pass = `0`。
2. P1 Geometry Outcome Ledger 完整落盘：rows = `3900`，canonical AP0 = `2876`，generated APY = `512`，generated APG = `512`。
3. P1 ledger audit 过：required field missing / NaN / Inf = `0 / 0 / 0`，identity join pass = `1`。
4. P2 target-density geometry lattice 扫描 `329` 个 target，weak geometry target = `32`，official geometry target = `0`。
5. P2 选出的 best relaxed target = `T5-RelaxedV80NonNegative`，action count = `57`，coverage = `0.019819193324061197`。
6. P2 best target 的 outcome 很干净：V_integrated LCB = `0.1960342568901401`，h240 longrisk = `0.0`，bad/null = `0.0 / 0.0`；但 coverage < `0.03`，不能 official。
7. P3 signal-channel raw upper-bound 未过：best = `SC7-leave-one-out-transfer`，AUC_OfficialGeoCandidate = `0.9706870146967946`，TopK64 precision = `0.125`，TopK64 V_integrated LCB = `-0.260665499652037`。
8. P4 cover/curvature 未过：cover metric 对 long-risk 有信号，但 curvature metric pass = `0`；combined TopK64 OfficialGeo precision = `0.046875`，longrisk = `0.59375`。
9. P5 memory audit v2 未过：TopK64 forget risk = `0.06995676321837122`，但 old_family_fail_count_topk64 = `1`，memory_pass = `0`。
10. P6 GeoScore diagnostic weak pass：best = `GS1-VSignalRisk`，heldout accepted = `91`，coverage = `0.06594202898550725`，V_integrated LCB = `0.04369212799578348`，h240 longrisk = `0.0`。
11. P6 GeoScore official pass 未过：best bad event = `0.06593406593406594` > `0.05`，且 OfficialGeo precision 仍低；不能 controller。
12. P7 APGS1-APGS8 implementation 过：generated actions = `512`，payload/certificate hash missing = `0 / 0`，action apply L∞ max = `0.0`。
13. P8 APGS branch-horizon smoke 完整落盘：expected/actual rows = `12288 / 12288`，unresolved exception = `0`，rows/sec = `21.05364247842129`。
14. P8 APGS outcome 未过：best = `APGS7-SymmetricBoundaryResidualUpdate`，OfficialGeo precision = `0.015625`，OutcomeGood precision = `0.03125`，V_integrated LCB = `-0.9218810936698074`，h240 longrisk = `0.765625`。
15. P9 source-to-generated damage 未过：best = `APGS1-SignalChannelEdgeMask`，new positive created rate = `0.0`，longrisk created rate = `0.75`，Damage integrated LCB = `-2.3023029125421157`。
16. P10 geometry certificate v2 未过：best = `GCERT16-OfficialMinimalGeometryCertificate`，AUC_OfficialGeoCandidate = `0.9977771191464138`，TopK64 precision = `0.21875`，TopK64 longrisk = `0.1875`，ECE = `0.6297824644891717`。
17. P11-P16 gate-blocked：没有 source controller、selected runtime、leaveout、official paired replay、short/full training 或 continual validation。
18. P15 Base-Acc Sentinel 复用前序真实 sentinel：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
19. 当前 primary blocker：`commit_time_geometry_signal_absent`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9540_geometry_definition_signal_channel_primitive.py` | v9.5.4 runner；读取 v9.5.3/v9.5.2/v9.4.9/v9.3.3 artifacts，构建 Geometry Outcome Ledger，扫描 geometry target lattice，审计 signal/cover/curvature/memory，生成 APGS1-APGS8，执行 branch-horizon smoke、damage matrix、geometry certificate 与 system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9540_geometry_definition_signal_channel_primitive.py
```

正式运行：

```bash
python experiments/run_v9540_geometry_definition_signal_channel_primitive.py \
  --out-dir results/real_rerun_20260506/v9540_geometry_definition_signal_channel_primitive_first_20260515T040000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --apgs-actions-per-primitive 64
```

运行结果：

```json
{
  "apgs_generated_actions": 512,
  "best_apgs_OfficialGeoCandidate_precision": 0.015625,
  "best_apgs_primitive": "APGS7-SymmetricBoundaryResidualUpdate",
  "geometry_certificate_weak_pass": 0,
  "ledger_rows": 3900,
  "out_dir": "results/real_rerun_20260506/v9540_geometry_definition_signal_channel_primitive_first_20260515T040000Z",
  "route": "R3-LegalGeometrySignalAbsent",
  "selected_geometry_target_id": "T5-RelaxedV80NonNegative",
  "system_legal_controller_pass": 0,
  "weak_geometry_target_count": 32
}
```

说明：正式 artifact 对应修复后的 fresh run。APGS 的 `ShuffledAPGS` 负控分支采用 shape-preserving tensor roll；最终 P8 rows = `12288 / 12288`，unresolved exception = `0`。

## 2. Route

`route_decision_v9540.json` 摘要：

```json
{
  "route": "R3-LegalGeometrySignalAbsent",
  "source_route_v9530": "R2-NoGoodGeometryActionInCanonicalAP0",
  "geometry_ledger_pass": 1,
  "ledger_rows": 3900,
  "weak_geometry_target_count": 32,
  "official_geometry_target_count": 0,
  "selected_geometry_target_id": "T5-RelaxedV80NonNegative",
  "selected_action_count": 57,
  "selected_coverage": 0.019819193324061197,
  "selected_V_integrated_lcb": 0.1960342568901401,
  "selected_h240_longrisk": 0.0,
  "selected_bad_event_rate": 0.0,
  "selected_null_event_rate": 0.0,
  "signal_channel_weak_pass": 0,
  "best_signal_feature_id": "SC7-leave-one-out-transfer",
  "best_signal_TopK64_precision": 0.125,
  "best_signal_TopK64_longrisk": 0.09375,
  "cover_curvature_pass": 0,
  "memory_pass": 0,
  "geoscore_weak_pass": 1,
  "best_geoscore_id": "GS1-VSignalRisk",
  "best_geoscore_accepted_count_heldout": 91,
  "best_geoscore_V_integrated_lcb": 0.04369212799578348,
  "apgs_implementation_pass": 1,
  "apgs_generated_action_count": 512,
  "apgs_branch_horizon_pass": 1,
  "apgs_branch_horizon_rows_actual": 12288,
  "apgs_unresolved_exception_count": 0,
  "apgs_weak_pass": 0,
  "best_apgs_primitive": "APGS7-SymmetricBoundaryResidualUpdate",
  "best_apgs_OfficialGeoCandidate_precision": 0.015625,
  "best_apgs_OutcomeGood_precision": 0.03125,
  "best_apgs_V_integrated_lcb": -0.9218810936698074,
  "best_apgs_h240_longrisk": 0.765625,
  "source_to_generated_damage_pass": 0,
  "geometry_certificate_weak_pass": 0,
  "best_certificate_id": "GCERT16-OfficialMinimalGeometryCertificate",
  "best_certificate_AUC_OfficialGeoCandidate": 0.9977771191464138,
  "best_certificate_TopK64_precision": 0.21875,
  "best_certificate_TopK64_longrisk": 0.1875,
  "source_controller_pass": 0,
  "selected_runtime_pass": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "commit_time_geometry_signal_absent"
}
```

判断：v9.5.4 证明 relaxed geometry target 在 AP0 中存在，但 commit-time signal/cover/curvature/memory 单项和组合证书都没有达到 controller 所需的可用筛选能力；APGS 也没有把 relaxed geometry target 转成 generated-action frontier。

## 3. P0 v9.5.3 boundary

Artifact：

```text
p0_v9530_boundary_reproduction.csv
```

Summary：

```text
route_v9530 = R2-NoGoodGeometryActionInCanonicalAP0
source_route_v9520 = R1-CanonicalActionDensityInsufficientForMultiHorizonTarget
canonical_full_control_outcome_ready = 1
geometry_card_rows = 3388
GGA_count = 5
GGA_coverage = 0.0017385257301808068
best_snr_AUC_GGA = 0.5108043788672061
best_snr_TopK64_GGA_precision = 0.0
best_apg_primitive = APG1-SNRProjectedEdgeUpdate
best_apg_GGA_precision = 0.03125
best_apg_V_integrated_lcb = -0.4061862277341243
best_apg_h240_longrisk = 0.6875
best_certificate_id = GCERT8-MinimalControllerCertificate
best_certificate_AUC_GGA = 0.9899049685968893
best_certificate_TopK64_GGA_precision = 0.125
best_certificate_TopK64_longrisk = 0.0
geometry_card_pass = 1
apg_implementation_pass = 1
apg_preflight_branch_horizon_pass = 1
system_legal_controller_pass = 0
p0_pass = 1
```

判断：P0 pass。v9.5.4 没有跳过 v9.5.3 的 GGA 稀疏与 APG/GCERT failure boundary。

## 4. P1 Geometry Outcome Ledger

Artifacts：

```text
geometry_outcome_ledger_v9540.csv
p1_geometry_outcome_ledger_audit.csv
```

Summary：

```text
ledger_rows = 3900
canonical_ap0_rows = 2876
generated_apy_rows = 512
generated_apg_rows = 512
required_field_missing_count = 0
nan_count = 0
inf_count = 0
identity_join_pass = 1
commit_time_field_separation_pass = 1
feature_cost_recorded = 1
outcome_field_complete = 1
geometry_ledger_pass = 1
```

判断：P1 pass。统一 ledger 将 AP0/APY/APG 放到同一套 outcome + signal + cover + curvature + memory + runtime 字段下，后续 P2-P10 都来自该 ledger 或 APGS 新落盘结果。

## 5. P2 target-density geometry lattice

Artifact：

```text
p2_target_density_geometry_lattice_audit.csv
```

Summary：

```text
target_candidate_count = 329
weak_geometry_target_count = 32
official_geometry_target_count = 0
selected_geometry_target_id = T5-RelaxedV80NonNegative
selected_action_count = 57
selected_coverage = 0.019819193324061197
selected_V_integrated_lcb = 0.1960342568901401
selected_h240_longrisk = 0.0
selected_bad_event_rate = 0.0
selected_null_event_rate = 0.0
selected_support_family_count = 23
weak_geometry_target_pass = 1
official_geometry_target_pass = 0
```

Representative target:

```text
T0-OutcomeGood:
  action_count = 79
  coverage = 0.027468706536856746
  V_integrated_lcb = 0.16201929527024647
  h240_longrisk = 0.0
  bad/null = 0.0 / 0.0
  support_family_count = 34
  support_balance_pass = 1
```

判断：P2 回答了 v9.5.3 的 GGA=5 问题：原 GGA 定义确实过严，relaxed geometry/outcome targets 能找到 `57-79` 个干净动作；但 official coverage 仍低于 `0.03`，不能直接打开 controller。

## 6. P3 signal-channel raw upper-bound

Artifact：

```text
p3_signal_channel_raw_upper_bound_audit.csv
```

Summary：

```text
feature_count = 10
best_feature_id = SC7-leave-one-out-transfer
best_AUC_OfficialGeoCandidate = 0.9706870146967946
best_TopK64_precision = 0.125
best_TopK64_longrisk = 0.09375
best_TopK64_V_integrated_lcb = -0.260665499652037
signal_channel_weak_pass = 0
signal_channel_strong_pass = 0
```

Example weak feature:

```text
SC1-parameter-group-SNR:
  AUC_OfficialGeoCandidate = 0.5417125661624439
  TopK64_precision = 0.0
  TopK64_longrisk = 0.765625
  TopK64_V_integrated_lcb = -1.2522607078264898
```

判断：P3 未过。部分 signal score 的 AUC 看起来很高，但 TopK64 precision 和 V_integrated LCB 不够，说明它不能作为 accepted region。

## 7. P4/P5 cover-curvature 与 memory

P4 artifact：

```text
p4_cover_curvature_hardtail_stability_audit.csv
```

P4 summary：

```text
metric_count = 10
cover_metric_longrisk_pass = 1
curvature_metric_longrisk_pass = 0
combined_TopK64_OfficialGeoCandidate_precision = 0.046875
combined_TopK64_longrisk = 0.59375
cover_curvature_pass = 0
```

P4 representative metric:

```text
M1-cover-entropy-delta:
  AUC_OfficialGeoCandidate = 0.7968779386872297
  AUC_LongRisk = 0.5895183642536495
  TopK64_precision = 0.03125
  TopK64_longrisk = 0.625
```

P5 artifact：

```text
p5_memory_antiforgetting_audit_v2.csv
```

P5 summary：

```text
forget_risk_mean_topk64 = 0.06995676321837122
old_family_fail_count_topk64 = 1
old_family_probe_loss_delta_topk64 = 0.02798270528734849
old_family_margin_delta_topk64 = -0.024484867126429926
old_new_tradeoff_corr = 0.8347833451467507
AUC_OfficialGeoCandidate = 0.9844165614336764
AUC_LongRisk = 0.8800326366967263
TopK64_precision = 0.15625
memory_pass = 0
```

判断：cover 和 memory 均有 diagnostic signal，但都无法独立满足 weak pass。尤其 TopK/long-risk/old-family fail 仍无法闭合。

## 8. P6 Geometry score assembly

Artifact：

```text
p6_geometry_score_assembly.csv
```

Summary：

```text
score_count = 5
best_score_id = GS1-VSignalRisk
best_accepted_count_heldout = 91
best_coverage_heldout = 0.06594202898550725
best_V_integrated_lcb = 0.04369212799578348
best_h240_longrisk = 0.0
best_bad_event_rate = 0.06593406593406594
best_null_event_rate = 0.01098901098901099
geoscore_weak_pass = 1
geoscore_official_pass = 0
```

Best score row：

```text
GS1-VSignalRisk:
  accepted_count_cal = 87
  accepted_count_heldout = 91
  precision_OutcomeGood = 0.5054945054945055
  precision_OfficialGeoCandidate = 0.0989010989010989
  support_family_count = 41
  support_balance_pass = 1
```

判断：P6 是本轮正向推进之一。多项 geometry score 能形成一个 weak diagnostic accepted region，但 bad_event > official threshold，OfficialGeo precision 低，不能 controller。

## 9. P7/P8 APGS primitive

P7 artifacts：

```text
p7_apgs_geometry_signal_primitive_implementation.csv
```

P7 summary：

```text
primitive_count = 8
generated_action_count = 512
payload_hash_missing = 0
certificate_hash_missing = 0
action_apply_error_linf_max = 0.0
negative_control_materialized = 1
apgs_implementation_pass = 1
```

P8 artifacts：

```text
p8_apgs_branch_horizon_smoke_outcome.csv
apgs_branch_horizon_outcome_trace_v9540.csv
```

P8 materializer summary：

```text
primitive_count = 8
action_count = 512
branch_count = 8
horizon_count = 3
branch_horizon_rows_expected = 12288
branch_horizon_rows_actual = 12288
completion_rate = 1.0
unresolved_exception_count = 0
rows_per_sec = 21.05364247842129
wallclock_sec = 583.6519743599929
apgs_branch_horizon_smoke_pass = 1
```

P8 outcome summary：

```text
best_primitive_id = APGS7-SymmetricBoundaryResidualUpdate
best_OfficialGeoCandidate_precision = 0.015625
best_OutcomeGood_precision = 0.03125
best_V_integrated_lcb = -0.9218810936698074
best_h240_longrisk = 0.765625
apgs_weak_pass = 0
apgs_strong_pass = 0
```

判断：APGS 工程链路完整闭合，但 outcome 比目标差很多。APGS 没有把 P2/P6 的 relaxed geometry target 转成 generated-action frontier。

## 10. P9 damage matrix

Artifact：

```text
p9_source_to_generated_geometry_damage_matrix.csv
```

Summary：

```text
primitive_count = 8
best_primitive_id = APGS1-SignalChannelEdgeMask
best_positive_preserved_rate = 0.0
best_longrisk_created_rate = 0.75
best_new_positive_created_rate = 0.0
best_Damage_integrated_LCB = -2.3023029125421157
source_to_generated_damage_pass = 0
```

判断：P9 未过。APGS 没有保留/制造 OfficialGeo positives，且生成动作中 long-risk created rate 很高。

## 11. P10 geometry certificate v2

Artifact：

```text
p10_geometry_certificate_v2.csv
```

Summary：

```text
certificate_count = 8
base_rate_OfficialGeoCandidate = 0.004132231404958678
best_certificate_id = GCERT16-OfficialMinimalGeometryCertificate
best_AUC_OfficialGeoCandidate = 0.9977771191464138
best_AUC_OutcomeGood = 0.9921091317883085
best_TopK64_precision = 0.21875
best_TopK64_longrisk = 0.1875
best_TopK64_V_integrated_lcb = 0.27462000176921225
best_ECE = 0.6297824644891717
geometry_certificate_weak_pass = 0
geometry_certificate_strong_pass = 0
```

Representative row：

```text
GCERT9-GeometryOutcomeLinearScore:
  AUC_OfficialGeoCandidate = 0.9872343128122618
  TopK64_precision = 0.15625
  TopK64_longrisk = 0.015625
  TopK64_V_integrated_lcb = 0.2738222617427317
  ECE = 0.6207785649927459
  certificate_weak_pass = 0
```

判断：P10 重复暴露 v9.5.3 的证书问题：AUC 很高，但 calibration/ECE 和 TopK/risk 同时过线仍失败；不能把高 AUC 写成 certificate pass。

## 12. P11-P16 boundary

P11：

```text
p11_minimal_geometry_controller.csv = not_run
reason = P6_or_P8_or_P10_gate_failed
selected_primitive_id = APGS7-SymmetricBoundaryResidualUpdate
selected_certificate_id = GCERT16-OfficialMinimalGeometryCertificate
source_controller_pass = 0
dataset_name_used/future_outcome_used/validation_test_used = 0/0/0
```

P12：

```text
p12_selected_runtime_preflight.csv = not_run
reason = P11_controller_not_selected
selected_runtime_pass = 0
```

P13-P14-P16：

| artifact | status / reason |
|---|---|
| `p13_leaveout_boundary.csv` | `not_run`, `P12_runtime_not_selected` |
| `p14_official_paired_replay_boundary.csv` | `not_run`, `P13_leaveout_not_open` |
| `p16_short_full_training_boundary.csv` | `not_run`, `P14_paired_replay_not_open` |

P15 Base-Acc Sentinel：

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

判断：没有把 relaxed geometry target、GeoScore diagnostic、APGS preflight、certificate diagnostic 或 Base-Acc Sentinel 写成 official system pass。

## 13. No-fake audit

```text
rows_checked = 17739
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
v9530_boundary_pass = 1
geometry_ledger_pass = 1
target_density_geometry_lattice_pass = 1
signal_channel_weak_pass = 0
cover_curvature_pass = 0
memory_pass = 0
geoscore_weak_pass = 1
apgs_implementation_pass = 1
apgs_branch_horizon_pass = 1
apgs_weak_pass = 0
source_to_generated_damage_pass = 0
geometry_certificate_weak_pass = 0
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
route = R3-LegalGeometrySignalAbsent
F0_boundary_reproduction_failed = 0
F1_geometry_ledger_failed = 0
F2_geometry_target_too_sparse = 0
F3_legal_geometry_signal_absent = 1
F4_apgs_primitive_value_fail = 0
F5_certificate_topk_fail = 0
F6_controller_quality_fail = 0
F7_selected_runtime_fail = 0
F8_leaveout_fail = 0
F9_paired_replay_fail = 0
F10_base_acc_catastrophic = 0
primary_blocker = commit_time_geometry_signal_absent
```

## 14. Hash

| artifact | SHA256 |
|---|---|
| plan | `72a52f3656cbb01fa26834346552b4eddde92e6022799f89486b2c8252f267fc` |
| runner | `d83477c915ab770bef860b40155ec95edf28589a87c79c07d4302d40cd2d580d` |
| run manifest | `b65a71dc30ba0f18b421d48369cc5727892232282e26de92fb90d92b2805ce78` |
| route | `b9a40a97661d84fe3c15eae3e939d7fd5d80348bafc249842f72507665684fe8` |
| Geometry Outcome Ledger | `719cf78adbf65981ba3dc08e3caa589d314ba1a2642e6daa656f0427cacf7384` |
| P0 boundary | `fffa6f2205d84aa1b7dacd804a4260a7271a42790b6704533941029a76a6ad39` |
| P1 ledger audit | `0198659615f6376d5747ddaa3e01099ef3c10726c01ac19012076268e33c12f5` |
| P2 target lattice | `d3e0d75bad8da773b07307524e46912eba9e63d9fd260b88278a9aca70d825f5` |
| P3 signal audit | `7fcb7f6d84b04c86b4c59866439029d7846961a5d5683831cc168245a0ec725b` |
| P4 cover curvature | `5b6f0ce4528c9d5f05b04c6e57797f45fdc3e6271e00d4b7620ffc8a27605824` |
| P5 memory | `87acdb9f860b662df659997c1d19182fd7c59663ff222263b76b821969dd97be` |
| P6 GeoScore | `320870164aeeb7e0c3d0b3cf5c38ac43f7d78bc70a706976db9df1c1e8b478d2` |
| P7 APGS implementation | `199e838aac3994389d8b2cee1f481fb37d1bcdef044ef7e338c62cfbcb3b6b58` |
| P8 APGS smoke | `718689577c611663e021bce4b136283702a7094da7fa3e68fc0775da25c645f0` |
| APGS trace | `df207fbb2b4f9fe47267c5ff15a035a2ae6c0d7cc63352a6269bdf943eec0ef4` |
| P9 damage | `bc83c0a0ff47d7d8311ebd8cab90f330aa258afdbfdd10b0ffdb7e11b1f549d4` |
| P10 certificate | `d1fde49ed082981d1dd9890bb94d1d15ba724b9119b2ca79a40ff5269a89ba00` |
| P11 controller | `107446f77343e504a4974d217f490289b09cc45ad09d615d0ef20eb094fdf2b4` |
| P12 runtime | `e842a2252f223ad4b8c85b94969237ae6282ec22f10d0e4769a13a2d446c1ae4` |
| P13 leaveout | `e49c7683a0c9ddcb7220eab879c2ae4ed4e6d684c89bd8d1c5cb5669830366e6` |
| P14 paired replay | `09f12101f6fef759d146e112596f6c2c7f57f9b9b9c1c7f950294c4db9a4e90c` |
| P15 Base-Acc Sentinel | `7e3b1d415319271826f41ab500dcd7bdbb4d1c8e85045b5f7d64b4ec8bf5dd27` |
| P16 short/full | `40edd3f185e640f16ff96fe64472f3a5b0e4e7051667ee63fd1f22c74c4abe7d` |
| no-fake audit | `c28bc8c3e5cb88d62107376c1b29bf75ffe7b73b1815529f5897b7e4132626a0` |
| contract audit | `abc6526fe7fce6b615bf255a568a2f12ee6a72352a8a57df2077ca2de137dc06` |
| failure table | `33afd08a5603d620969cf16371107625cd0b08cf1bdcef47181ef4bd18999b25` |

## 15. 最终分析结论

v9.5.4 的真实推进是：

```text
v9.5.3:
  GeometryCard 和 APG 工程链路闭合；
  但 GoodGeomAction 只有 5 个，APG/GCERT 未能转 controller。

v9.5.4:
  建立 Geometry Outcome Ledger；
  把单一 GGA 改成 OutcomeGood / SignalGood / GeometryStable / OfficialGeoCandidate 多级标签；
  找到 32 个 weak relaxed geometry target，证明 GGA=5 主要是定义过严；
  GeoScore 能形成 heldout weak diagnostic accepted region；
  APGS1-APGS8 工程链路和 8 分支 branch-horizon smoke 完整闭合；
  但 commit-time signal 与 APGS generator 仍未能转成 official controller。
```

机制判断：

1. H0 成立：v9.5.3 boundary 被复现。
2. H1 部分成立：GGA=5 不是“好几何完全不存在”，relaxed geometry target 存在；但 official coverage 仍不足。
3. H2 未成立：signal-channel features 没有达到 weak gate，best TopK64 precision 只有 `0.125`，且 TopK64 V_integrated LCB 为负。
4. H3 未成立：cover/curvature combined score TopK64 precision 只有 `0.046875`，longrisk = `0.59375`。
5. H4 成立于 diagnostic 层：GeoScore 能找到 91 个 heldout accepted actions，V_integrated LCB > 0，longrisk = 0；但 bad event 超过 official gate。
6. H5/H6 成立于 implementation 层：APGS payload/certificate/action apply 与 full branch-horizon materializer 完整闭合。
7. H7 未成立：APGS generated actions 仍 value-negative、longrisk 高，best OfficialGeo precision 只有 `0.015625`。
8. H8 未成立：source-to-generated damage 显示 APGS 没有 preservation/improvement，longrisk_created_rate 高。
9. H9 未成立：GCERT v2 仍是高 AUC、弱 TopK/calibration，不能 controller。
10. H10-H13 未打开：没有 controller，就不能打开 runtime、leaveout、official paired replay、short/full 或 continual validation。
11. Base-Acc Sentinel 继续健康，但没有用于 controller，也不是 functional success。

最终一句话：

> v9.5.4 真实执行后停在 `R3-LegalGeometrySignalAbsent`：Geometry Outcome Ledger 与 relaxed geometry target 证明“好几何”并非完全不存在，GeoScore 也有弱 diagnostic accepted region；但 commit-time signal/cover/curvature/memory 仍不能可靠筛选，APGS1-APGS8 虽完整落盘却继续生成 value-negative/high-longrisk 动作，certificate 仍无法 TopK/calibration 同时过线，strict PureKAN functional 仍未成功。
