# DG-KAN v9.5.1 Primitive Family Reset / Multi-Horizon Objective / Constructive Certificate 实验复盘

> 本复盘记录 `DG-KAN_v9.5.1_PrimitiveFamilyReset_MultiHorizonObjectiveConstructiveCertificate_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 multi-horizon target diagnostic、raw legal upper-bound probe、mechanism anatomy、APX preflight、constructive generator smoke、certificate diagnostic、runtime boundary 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R1-TargetConflictUnresolved
base_candidate = LQ-t2-h256
success_v9510_strict_purekan_functional = False
success_v9510_full_functional = False
success_v9510_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9510_primitive_family_reset_multihorizon_certificate_first_20260515T010000Z/
```

核心结论：

1. P0 复现 v9.5.0 boundary：source route = `R9-PrimitiveFamilyResetRequired`，canonical truth base ready = `1`，v9.5.0 system pass = `0`。
2. P1 multi-horizon target audit 过数据重算，但 official target 未打开：`T_A=120`，`T_B=25`，`T_C=16`，`T_D=64`。
3. P1 选择的 strict candidate 是 `T_C`，但 coverage 只有 `0.005563282336578581`；`multi_horizon_objective_pass = 0`，route 因此优先停在 `R1-TargetConflictUnresolved`。
4. P1 显示 YRobust legacy target 仍不等价于 strict multi-horizon：`P_h80_weak(T_A)=0.175`，`Jaccard_TA_TC=0.13333333333333333`。
5. P2 raw legal upper-bound probes 未过：best probe = `UB4-gradient-action-bilinear-probe`，best AUC_TB = `0.44422307962118557`，TopK64 T_B precision = `0.0`，TopK64 long-risk = `0.609375`。
6. P3 mechanism anatomy 未过：best mechanism = `support_memory_similarity`，best cluster purity T_B = `0.0625`，dominant miss reason = `M8-outcome-only-pattern`。
7. P4 APX1-APX8 primitive family spec 完整 materialize：primitive count = `8`，generated action count = `512`，payload/certificate hash missing = `0 / 0`，action apply L∞ max = `0.0`。
8. P5 APX deterministic preflight ladder 过：single / three / sixteen action pass 均为 `1`，no-transform equivalence pass = `1`，negative controls 有 divergence。
9. P6 APX branch-horizon smoke 完整落盘：generated actions = `512`，expected/actual branch-horizon rows = `12288 / 12288`，unresolved exception = `0`。
10. P6 APX smoke 未过：best primitive = `APX7-EnsembleIntersectionPrimitive`，h20 weak CP = `0.125`，h20 V_ctrl LCB = `-1.6380800012017955`，h240 long-risk = `0.84375`。
11. P6 APX8 negative control 没有误通过：`APX8_negative_control_weak_pass = 0`。
12. P7 source-to-generated damage 未过：best damage row 是 APX8 negative control，new-positive rate = `0.0625`，longrisk-created rate = `0.171875`，Damage h20 LCB = `-0.655000294467341`。
13. P8 certificate v4 未过：best certificate = `CERT24-AdamWConflictReliefBound`，AUC_TB = `0.6041257367387033`，TopK64 T_B precision = `0.0`，TopK64 long-risk = `0.734375`。
14. P9-P14 gate-blocked：没有 source controller、selected runtime、official paired replay、short/full training 或 continual validation。
15. P14 Base-Acc Sentinel 复用前序真实 sentinel：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
16. 当前 primary blocker：`multi_horizon_target_conflict_unresolved`；后续不能把 APX generator/certificate 阈值调参当成 official path，必须先解决 multi-horizon target 定义与可用支持度冲突。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9510_primitive_family_reset_multihorizon_certificate.py` | v9.5.1 runner；读取 v9.5.0/v9.4.9/v9.4.8 canonical artifacts，执行 multi-horizon target audit、raw legal upper-bound probe、mechanism anatomy、APX1-APX8 primitive reset、APX branch-horizon smoke、CERT20-CERT26 certificate、controller/runtime/system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9510_primitive_family_reset_multihorizon_certificate.py
```

正式运行：

```bash
python experiments/run_v9510_primitive_family_reset_multihorizon_certificate.py \
  --out-dir results/real_rerun_20260506/v9510_primitive_family_reset_multihorizon_certificate_first_20260515T010000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --probe-action-limit 2876 --apx-actions-per-primitive 64
```

运行结果：

```json
{
  "best_apx_h20_weak_CP": 0.125,
  "best_apx_h240_longrisk": 0.84375,
  "best_apx_primitive": "APX7-EnsembleIntersectionPrimitive",
  "certificate_effect_valid_pass": 0,
  "out_dir": "results/real_rerun_20260506/v9510_primitive_family_reset_multihorizon_certificate_first_20260515T010000Z",
  "route": "R1-TargetConflictUnresolved",
  "system_legal_controller_pass": 0
}
```

## 2. Route

`route_decision_v9510.json` 摘要：

```json
{
  "route": "R1-TargetConflictUnresolved",
  "source_route_v9500": "R9-PrimitiveFamilyResetRequired",
  "canonical_full_control_outcome_ready": 1,
  "multi_horizon_objective_pass": 0,
  "selected_official_target_candidate": "T_C",
  "T_A_count": 120,
  "T_B_count": 25,
  "T_C_count": 16,
  "T_D_count": 64,
  "legal_information_upper_bound_pass": 0,
  "best_raw_probe_id": "UB4-gradient-action-bilinear-probe",
  "best_raw_AUC_TB": 0.44422307962118557,
  "best_raw_TopK64_precision_TB": 0.0,
  "mechanism_pass": 0,
  "best_mechanism_id": "support_memory_similarity",
  "best_cluster_purity_TB": 0.0625,
  "apx_primitive_family_spec_pass": 1,
  "apx_preflight_pass": 1,
  "apx_smoke_weak_pass": 0,
  "best_apx_primitive_id": "APX7-EnsembleIntersectionPrimitive",
  "best_apx_h20_weak_CP": 0.125,
  "best_apx_h20_V_ctrl_lcb": -1.6380800012017955,
  "best_apx_h240_longrisk": 0.84375,
  "certificate_effect_valid_pass": 0,
  "best_certificate_id": "CERT24-AdamWConflictReliefBound",
  "best_certificate_AUC_TB": 0.6041257367387033,
  "source_controller_pass": 0,
  "selected_runtime_pass": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "multi_horizon_target_conflict_unresolved"
}
```

判断：v9.5.1 的 APX implementation/preflight/materializer 都真实闭合，但 official route 在 P1 已经因 target conflict 未打开。P2/P3/P6/P8 的后续实测也全部未形成可 official 的 selector/generator/certificate。

## 3. P0 v9.5.0 boundary

Artifact：

```text
p0_v9500_boundary_reproduction.csv
```

Summary：

```text
route_v9500 = R9-PrimitiveFamilyResetRequired
canonical_full_control_outcome_ready = 1
YRobust_count = 120
YStableHorizon_count = 113
YStrictAllH_count = 16
P_h80_weak_given_YRobust = 0.175
UB2_AUC_YRobust = 0.51014755684567
UB2_TopK64_YRobust_precision = 0.015625
best_cluster_purity_YRobust = 0.046875
constructive_best_generator_id = SG2-HorizonGuardedTrustRegion
constructive_best_h20_weak_CP = 0.1875
constructive_best_h20_V_ctrl_lcb = -1.5456723592527004
constructive_best_h240_longrisk = 0.71875
best_certificate_id = CERT18-MinimalDistilledLegalCertificate
best_certificate_AUC_YRobust = 0.5284178387150466
runtime_preflight_pass = 1
official_runtime_pass = 0
system_legal_controller_pass = 0
p0_pass = 1
```

判断：P0 pass。v9.5.1 没有跳过 v9.5.0 的 `PrimitiveFamilyResetRequired` boundary。

## 4. P1 multi-horizon target audit v2

Artifact：

```text
p1_multi_objective_label_audit_v2.csv
```

Summary：

```text
action_count = 2876
T_A_count = 120
T_B_count = 25
T_C_count = 16
T_D_count = 64
T_A_coverage = 0.04172461752433936
T_B_coverage = 0.008692628650904033
T_C_coverage = 0.005563282336578581
T_D_coverage = 0.022253129346314324
Jaccard_TA_TB = 0.20833333333333334
Jaccard_TA_TC = 0.13333333333333333
Jaccard_TB_TC = 0.64
selected_official_target_candidate = T_C
multi_horizon_objective_pass = 0
target_conflict_unresolved = 1
```

Target rows：

| target | count | coverage | h20 weak | h80 weak | h240 weak | h240 long-risk | integrated V LCB | official pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `T_A-LegacyYRobust` | `120` | `0.041725` | `1.0` | `0.175` | `0.566667` | `0.0` | `-0.530451` | `0` |
| `T_B-StableHorizon` | `25` | `0.008693` | `1.0` | `0.84` | `0.72` | `0.0` | `0.595976` | `0` |
| `T_C-StrictAllH` | `16` | `0.005563` | `1.0` | `1.0` | `1.0` | `0.0` | `0.784806` | `0` |
| `T_D-IntegratedRobustScoreTop` | `64` | `0.022253` | `0.625` | `0.6875` | `0.71875` | `0.015625` | `0.487061` | `0` |

判断：P1 是本轮 route 的 terminal blocker。更严格的 T_B/T_C 目标虽更 horizon-safe，但 support 太稀疏；legacy T_A 支持度较高但 h80 blind spot 明显。因此不能在本轮宣布 official multi-horizon objective 已闭合。

## 5. P2 raw legal upper-bound probe v2

Artifact：

```text
p2_raw_legal_upper_bound_probe_v2.csv
```

Summary：

```text
probe_count = 7
best_probe_id = UB4-gradient-action-bilinear-probe
best_AUC_TB = 0.44422307962118557
best_TopK64_precision_TB = 0.0
best_TopK64_longrisk = 0.609375
best_LDO_AUC_drop_max = 0.065671502050481
best_LSO_AUC_drop_max = 0.027662413884261994
legal_information_upper_bound_pass = 0
legal_information_upper_bound_weak_pass = 0
existing_ap0_selection_route_stopped = 1
```

Probe rows：

| probe | AUC_TB | TopK64 T_B | TopK64 long-risk | pass |
|---|---:|---:|---:|---:|
| `UB0-scalar-feature-probe` | `0.375630` | `0.015625` | `0.75` | `0` |
| `UB3-raw-logit-tensor-probe` | `0.372445` | `0.0` | `0.703125` | `0` |
| `UB4-gradient-action-bilinear-probe` | `0.444223` | `0.0` | `0.609375` | `0` |
| `UB5-hard-tail-probe` | `0.397194` | `0.015625` | `0.578125` | `0` |
| `UB6-basis-edge-probe` | `0.405591` | `0.0` | `0.578125` | `0` |
| `UB7-full-legal-tensor-probe` | `0.361978` | `0.0` | `0.734375` | `0` |
| `UB8-small-mlp-diagnostic-probe` | `0.414325` | `0.0` | `0.6875` | `0` |

判断：P2 未过。raw legal signal 对 T_B 没有可用 upper-bound observability；不能从这里蒸馏 official selector。

## 6. P3 mechanism anatomy v2

Artifact：

```text
p3_mechanism_anatomy_v2.csv
```

Summary：

```text
positive_count_TB = 25
positive_count_TD = 64
near_miss_negative_count = 2851
mechanism_count = 9
best_mechanism_id = support_memory_similarity
best_AUC_TB = 0.5348368993335672
best_cluster_purity_TB = 0.0625
best_cluster_longrisk_rate = 0.671875
dominant_miss_reason = M8-outcome-only-pattern
mechanism_pass = 0
```

判断：P3 未过。机制 anatomy 没有找到足够稳定、可 legal 化的 multi-horizon source mechanism。

## 7. P4/P5 APX primitive spec 与 deterministic preflight

P4 artifacts：

```text
p4_apx_primitive_family_spec.csv
```

P4 summary：

```text
primitive_count = 8
generated_action_count_total = 512
generated_action_count_expected = 512
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_linf_max = 0.0
certificate_fields_complete = 1
negative_control_APX8_generated = 1
apx_primitive_family_spec_pass = 1
```

P5 artifacts：

```text
p5_apx_preflight_ladder.csv
```

P5 summary：

```text
single_action_pass = 1
three_action_pass = 1
sixteen_action_pass = 1
no_transform_equivalence_pass = 1
negative_control_divergence_present = 1
apx_preflight_pass = 1
```

判断：P4/P5 pass。APX payload/certificate/action apply 和 deterministic replay 链路不是本轮 blocker。

## 8. P6 APX branch-horizon smoke

Artifacts：

```text
p6_apx_branch_horizon_smoke_outcome.csv
apx_branch_horizon_outcome_trace_v9510.csv
```

Summary：

```text
primitive_count = 8
generated_action_count_total = 512
branch_horizon_rows_expected/actual = 12288 / 12288
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
secondary_delta_completion_rate = 1
unresolved_exception_count = 0
wallclock_sec = 585.8676345599815
best_primitive_id = APX7-EnsembleIntersectionPrimitive
best_h20_weak_CP = 0.125
best_h20_V_ctrl_lcb = -1.6380800012017955
best_h80_V_ctrl_lcb = -1.4437290970183942
best_h240_longrisk = 0.84375
best_YStableHorizon_precision = 0.015625
best_YStrictAllH_precision = 0.0
APX8_negative_control_weak_pass = 0
apx_smoke_weak_pass = 0
apx_smoke_strong_pass = 0
```

Per primitive：

| primitive | h20 weak CP | h20 V_ctrl LCB | h80 V_ctrl LCB | h240 long-risk | YStable precision | pass |
|---|---:|---:|---:|---:|---:|---:|
| `APX1-ConstrainedTailDescentQP` | `0.140625` | `-1.917409` | `-1.582761` | `0.890625` | `0.015625` | `0` |
| `APX2-AdamWConflictOrthogonalResidual` | `0.09375` | `-1.773838` | `-1.637255` | `0.859375` | `0.0` | `0` |
| `APX3-HorizonGuardedTwoScaleUpdate` | `0.125` | `-1.771869` | `-1.446857` | `0.890625` | `0.0` | `0` |
| `APX4-LowRankEdgeLocalRepair` | `0.109375` | `-1.841788` | `-1.696534` | `0.828125` | `0.0` | `0` |
| `APX5-BasisResponseMatchedRepair` | `0.203125` | `-1.686215` | `-1.950520` | `0.75` | `0.0` | `0` |
| `APX6-NoHarmConservativeShrink` | `0.125` | `-1.657653` | `-1.726515` | `0.890625` | `0.0` | `0` |
| `APX7-EnsembleIntersectionPrimitive` | `0.125` | `-1.638080` | `-1.443729` | `0.84375` | `0.015625` | `0` |
| `APX8-RandomizedOrthogonalNegativeControl` | `0.15625` | `-1.801684` | `-1.493965` | `0.734375` | `0.015625` | `0` |

判断：P6 未过。APX1-APX8 都完成真实 branch-horizon replay，但没有任何 primitive 形成可用 multi-horizon value-positive frontier。

## 9. P7 damage audit

Artifact：

```text
p7_apx_damage_audit.csv
```

Summary：

```text
best_primitive_id = APX8-RandomizedOrthogonalNegativeControl
best_new_positive_created_rate = 0.0625
best_longrisk_created_rate = 0.171875
best_Damage_h20_lcb = -0.655000294467341
source_to_generated_damage_pass = 0
```

判断：P7 未过。即便按 damage 视角看，APX family 也没有形成稳定 source-to-generated preservation 或 improvement。

## 10. P8 certificate v4

Artifact：

```text
p8_effect_certificate_v4.csv
```

Summary：

```text
certificate_count = 7
best_certificate_id = CERT24-AdamWConflictReliefBound
best_AUC_TB = 0.6041257367387033
best_AUC_LongRisk = 0.5424872051624388
best_TopK64_TB_precision = 0.0
best_TopK64_LongRisk = 0.734375
best_ECE_TB = 0.17106634366960455
certificate_effect_valid_pass = 0
certificate_effect_strong_pass = 0
```

Best certificate row：

```text
certificate_id = CERT24-AdamWConflictReliefBound
AUC_TA = 0.5174197060424606
AUC_TB = 0.6041257367387033
AUC_TC = 0.6294117647058823
AUC_LongRisk = 0.5424872051624388
TopK16_TB_precision = 0.0
TopK64_TB_precision = 0.0
TopK64_LongRisk = 0.734375
ECE_TB = 0.17106634366960455
monotone_sign_pass = 1
calibration_to_heldout_drift = 0.34115717153002934
```

判断：P8 未过。CERT20-CERT26 没有把 T_B / long-risk 分离成可用 controller gate。

## 11. P9-P14 boundary

P9：

```text
p9_minimal_certificate_controller.csv = not_run
reason = P6_or_P8_gate_failed
source_controller_pass = 0
```

P10：

```text
p10_selected_controller_runtime.csv = not_run
reason = P9_controller_not_selected
selected_runtime_pass = 0
```

P11：

```text
primitive_generation_pass = 1
branch_horizon_outcome_pass = 1
certificate_effect_valid_pass = 0
controller_pass = 0
runtime_pass = 0
official_eligible = 0
system_legal_controller_pass = 0
```

P12-P14：

| artifact | status / reason |
|---|---|
| `p12_official_paired_replay_boundary.csv` | `not_run`, `P11_system_controller_not_official` |
| `p13_short_full_training_boundary.csv` | `not_run`, `P12_official_paired_replay_not_open` |
| `p14_continual_antiforgetting_boundary.csv` | `not_run`, `P13_short_full_not_open` |

判断：没有把 APX preflight、完整 APX branch-horizon rows、certificate diagnostic 或 Base-Acc Sentinel 写成 official controller/runtime/system/downstream pass。

## 12. Base-Acc Sentinel

Artifact：

```text
base_acc_sentinel_v9510.csv
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
base_acc_reused_from_v9500 = 1
```

判断：Base-Acc Sentinel 继续健康，但仍是 isolated diagnostic，没有用于 selector、generator、certificate 或 controller。

## 13. No-fake audit

```text
rows_checked = 12481
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
v9500_boundary_pass = 1
multi_horizon_objective_pass = 0
raw_legal_upper_bound_pass = 0
mechanism_pass = 0
apx_primitive_family_spec_pass = 1
apx_preflight_pass = 1
apx_smoke_weak_pass = 0
source_to_generated_damage_pass = 0
certificate_effect_valid_pass = 0
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
route = R1-TargetConflictUnresolved
F0_reproduction_fail = 0
F1_target_conflict_unresolved = 1
F2_legal_selection_dead = 1
F3_apx_implementation_fail = 0
F4_apx_generated_value_fail = 0
F5_certificate_effect_fail = 0
F6_controller_runtime_blocked = 1
F7_base_acc_catastrophic = 0
primary_blocker = multi_horizon_target_conflict_unresolved
```

## 14. Hash

| artifact | SHA256 |
|---|---|
| plan | `4991a5865be2c13c0a02c14e407b064a396fa722ff532e44531c8e6508f3b8ce` |
| runner | `f6df3363258e6ae1e45e2a9ae1f41ceadcb7763531405b00c7439db0edecc554` |
| run manifest | `1781f6cbb56ff9d01c0060b6ca5f3962e46a4c3d8933a83ac00532a173c6caf1` |
| route | `05584367114c9d243c82c9c46a559f1c04d27c1e0e17bbd8d7a9a20d894ec125` |
| P0 boundary | `0f39f009e6101d6ae364626ffa67f90dd71a15c78e91251e11130b8da730b122` |
| P1 target audit | `830748366ee32027f9ded184b68dffcb760558428308dd3168e75d3977c8ec81` |
| P2 legal probe | `aa1186fe1ad8185db20d04d513b132e5e9050cd2e54c83424d3b066f128ba0e1` |
| P3 mechanism | `41dd5dc2fb2ea79ea0d7a3fdf7f0c1ddcb501f7c2095189fc176a00b266de4e0` |
| P4 APX spec | `db1050c1271210d19e6167d9b186c7f61e8b7cc877154d6ef5690b897ebe7336` |
| P5 preflight | `e0842e6986b6d0225868597edb211f945f2dd2e0d3475d8b66644b63a44d403f` |
| P6 APX smoke | `5cac651d26f6427bbe67bb9f21f416ad254ff6f0574d7626c9e7948f9894278d` |
| P7 damage | `0760be16d8417bd87e920ed3030088bb87cbd66acd3e485ff6b60680f60aa20e` |
| P8 certificate | `54c268b3b7b761c88cc887e4c9214aa2198b2245e67cbb8ec7cb1943b3833f28` |
| P9 controller | `604f7fe73953dfb9df8c1850705fc3570be6b229b78c71bb88f17462d1f109b3` |
| P10 runtime | `7f1328217634e7833cdef271f568a7a23aa56e2cb0e33d30064b23c7f631d15d` |
| P11 system | `3faffeb3361c10898fcc5a2ad818e7fc2eab61b052d9228fc3924e83159f285d` |
| Base-Acc Sentinel | `1fec0a736aff11ad5bbbcdd8226c4de343f6f898c8b17092a8d05ba619fb26d1` |
| no-fake audit | `fef003a238bdbbc0a94b83b78f5f529506d7ed26df8d77e5b585922cc12a4a23` |
| contract audit | `3ff4a6bcaade88dd289aabb1ec04aae0179bdb9d53b1fe44c6537c674c6cac40` |
| failure table | `40b6a4b3cb0ac051d60fc79f91dca04b9e5f28b645f79e82f61885750b8b618b` |

## 15. 最终分析结论

v9.5.1 的真实推进是：

```text
v9.5.0:
  canonical robust frontier 仍存在；
  但 high-capacity legal probe、mechanism anatomy、SG1-SG4、CERT14-CERT18 都未转成 controller；
  route 停在 PrimitiveFamilyResetRequired。

v9.5.1:
  将目标从 legacy YRobust 推进到 multi-horizon target audit；
  实现并实测 APX1-APX8 primitive family reset；
  APX payload/certificate/action apply/preflight/branch-horizon materializer 都闭合；
  但 multi-horizon target 本身出现支持度与严格性冲突，legal probe、mechanism、APX generator、certificate 也未给出 official path。
```

机制判断：

1. H0 成立：v9.5.0 boundary 被复现，canonical truth base 仍可用。
2. H1 未成立：multi-horizon objective 没有 official closure；T_A 有 h80 blind spot，T_B/T_C 更严格但支持度太稀疏。
3. H2 未成立：raw legal upper-bound probes 对 T_B 不具备可用 observability，best AUC_TB 只有 `0.4442`。
4. H3 未成立：mechanism anatomy 没有找到 legal-stable cluster，best cluster purity T_B 只有 `0.0625`。
5. H4/H5 成立于 implementation 层：APX1-APX8 spec、payload/certificate/action apply 与 deterministic preflight 均闭合。
6. H6 未成立：APX branch-horizon smoke 没有产生 value-positive / horizon-safe primitive；best APX7 的 V_ctrl LCB 为负且 h240 long-risk 高。
7. H7 未成立：CERT20-CERT26 没有形成 effect-valid separation；best TopK64 T_B precision 为 `0.0`。
8. H8-H10 未打开：没有 controller，就不能打开 selected runtime、official paired replay、short/full 或 continual validation。
9. Base-Acc Sentinel 继续健康，但没有用于 controller，也不是 functional success。

最终一句话：

> v9.5.1 真实执行后停在 `R1-TargetConflictUnresolved`：APX1-APX8 primitive family reset 的工程链路已经闭合并完整落盘，但 multi-horizon objective 在严格性与支持度之间仍未闭合，legal signal、mechanism、APX generator 和 certificate 都没有形成可 official 的 selector/controller，strict PureKAN functional 仍未成功。
