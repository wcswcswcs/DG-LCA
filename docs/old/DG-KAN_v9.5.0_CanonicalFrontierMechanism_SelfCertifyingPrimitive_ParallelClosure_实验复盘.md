# DG-KAN v9.5.0 Canonical Frontier Mechanism / Self-Certifying Primitive / Parallel Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.5.0_CanonicalFrontierMechanism_SelfCertifyingPrimitive_ParallelClosure_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 legal upper-bound probe、mechanism anatomy、constructive generator smoke、certificate diagnostic、runtime preflight 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R9-PrimitiveFamilyResetRequired
base_candidate = LQ-t2-h256
success_v9500_strict_purekan_functional = False
success_v9500_full_functional = False
success_v9500_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9500_canonical_frontier_mechanism_self_certifying_primitive_first_20260515T000000Z/
```

核心结论：

1. P0 复现 v9.4.9 boundary：canonical truth base ready，v9.4.9 route = `R1-FrontierExistsButLegalOpaque`，system pass = `0`。
2. P1 多目标 label audit 过：YRobust count = `120`，YStableHorizon count = `113`，YStrictAllH count = `16`。
3. P1 确认 v9.4.9 的 YRobust 仍存在 h80 blind spot：`P_h80_weak_given_YRobust = 0.175`，objective conflict class = `OC1-YRobustHasH80BlindSpot`。
4. P2 high-capacity legal upper-bound probe 未过：UB2 AUC_YRobust = `0.51014755684567`，TopK64 YRobust precision = `0.015625`，TopK64 long-risk = `0.65625`。
5. P2 最好的单特征 probe 仍只是 `UB0-ScalarFeatureProbe`，AUC_YRobust = `0.5375105829704886`，不具备 official selector 能力。
6. P3 robust source mechanism anatomy 未找到可复用机制：best cluster purity = `0.046875`，dominant miss reason = `M8-outcome-only-pattern`，mechanism_pass = `0`。
7. P4 feature distillation 未运行，原因 = `P2_legal_upper_bound_probe_failed`。
8. P5 constructive source generator v3 已真实 materialize：generated actions = `256`，branch-horizon rows = `4608`，payload/certificate hash missing = `0`，action apply L∞ max = `0.0`。
9. P5 best generator = `SG2-HorizonGuardedTrustRegion`，但 h20 weak CP = `0.1875`，h20 V_ctrl LCB = `-1.5456723592527004`，h240 long-risk = `0.71875`，YRobust precision = `0.0625`，constructive_generator_pass = `0`。
10. P6 certificate v3 未过：best = `CERT18-MinimalDistilledLegalCertificate`，AUC_YRobust = `0.5284178387150466`，TopK64 YRobust precision = `0.015625`，TopK64 long-risk = `0.796875`。
11. P8 runtime preflight diagnostic 过，但 controller 未选中：runtime_preflight_pass = `1`，official_runtime_pass = `0`。
12. P14 Base-Acc Sentinel 复用 v9.4.9：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
13. P7/P9-P13 均 gate-blocked：没有 source controller、selected runtime、official paired replay 或 short/full validation。
14. 当前 primary blocker：`legal_upper_bound_mechanism_generator_certificate_all_fail`；下一步需要重置 primitive family，而不是继续调现有 legal feature / SG1-SG4 / CERT14-CERT18 阈值。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9500_canonical_frontier_mechanism_self_certifying_primitive.py` | v9.5.0 runner；读取 v9.4.9/v9.4.8 canonical artifacts，执行 multi-objective label audit、high-capacity legal upper-bound probe、mechanism anatomy、constructive source generator v3、certificate v3、runtime/system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9500_canonical_frontier_mechanism_self_certifying_primitive.py
```

正式运行：

```bash
python experiments/run_v9500_canonical_frontier_mechanism_self_certifying_primitive.py \
  --out-dir results/real_rerun_20260506/v9500_canonical_frontier_mechanism_self_certifying_primitive_first_20260515T000000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --probe-action-limit 2876 --generator-actions 64 --diagnostic-oracle-actions 16
```

运行结果：

```json
{
  "UB2_AUC_YRobust": 0.51014755684567,
  "UB2_TopK64_YRobust_precision": 0.015625,
  "certificate_effect_valid_pass": 0,
  "constructive_generator_pass": 0,
  "out_dir": "results/real_rerun_20260506/v9500_canonical_frontier_mechanism_self_certifying_primitive_first_20260515T000000Z",
  "route": "R9-PrimitiveFamilyResetRequired"
}
```

## 2. Route

`route_decision_v9500.json` 摘要：

```json
{
  "route": "R9-PrimitiveFamilyResetRequired",
  "source_route_v9490": "R1-FrontierExistsButLegalOpaque",
  "canonical_full_control_outcome_ready": 1,
  "YRobust_definition_consistency_pass": 1,
  "YRobust_count": 120,
  "YStableHorizon_count": 113,
  "YStrictAllH_count": 16,
  "objective_conflict_class": "OC1-YRobustHasH80BlindSpot",
  "legal_upper_bound_probe_pass": 0,
  "legal_upper_bound_probe_weak_pass": 0,
  "UB2_AUC_YRobust": 0.51014755684567,
  "UB2_TopK64_YRobust_precision": 0.015625,
  "UB2_TopK64_LongRisk_rate": 0.65625,
  "mechanism_pass": 0,
  "best_mechanism_candidate": "M4-support-memory-pattern",
  "best_cluster_purity_YRobust": 0.046875,
  "distillation_pass": 0,
  "constructive_generator_pass": 0,
  "constructive_best_generator_id": "SG2-HorizonGuardedTrustRegion",
  "constructive_best_h20_weak_CP": 0.1875,
  "constructive_best_h20_V_ctrl_lcb": -1.5456723592527004,
  "constructive_best_h240_longrisk": 0.71875,
  "constructive_best_YRobust_precision": 0.0625,
  "certificate_effect_valid_pass": 0,
  "best_certificate_id": "CERT18-MinimalDistilledLegalCertificate",
  "best_certificate_AUC_YRobust": 0.5284178387150466,
  "best_certificate_TopK64_YRobust_precision": 0.015625,
  "best_certificate_TopK64_LongRisk_rate": 0.796875,
  "source_controller_pass": 0,
  "selected_runtime_pass": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "legal_upper_bound_mechanism_generator_certificate_all_fail"
}
```

判断：v9.5.0 不是 canonical truth base 问题，也不是 runner/preflight 问题；高容量 legal probe、机制解释、构造 generator、certificate 均未给出可 official 的路径，因此 route 正确停在 `R9-PrimitiveFamilyResetRequired`。

## 3. P0/P1 boundary 与 label audit

Artifacts：

```text
p0_v9490_boundary_reproduction.csv
p1_multi_objective_robust_label_audit.csv
```

P0 summary：

```text
route_v9490 = R1-FrontierExistsButLegalOpaque
canonical_full_control_outcome_ready = 1
canonical_row_count_actual = 51768
quality_audit_pass = 1
YRobust_definition_consistency_pass = 1
YRobust_action_count = 120
best_legal_feature_id = NegHardTailFraction
best_legal_AUC_YRobust = 0.5375105829704886
best_legal_TopK64_YRobust_precision = 0.03125
existing_no_transform_sanity_pass = 1
existing_generator_preserve_pass = 0
direct_generator_pass = 0
certificate_effect_valid_pass = 0
runtime_preflight_pass = 1
system_legal_controller_pass = 0
p0_pass = 1
```

P1 summary：

```text
action_count = 2876
YRobust_count = 120
YImmediateOnly_count = 422
YNoLongRiskOnly_count = 781
YStableHorizon_count = 113
YStrictAllH_count = 16
Jaccard_YRobust_YStableHorizon = 0.9416666666666667
Jaccard_YRobust_YStrictAllH = 0.13333333333333333
P_YStrict_given_YRobust = 0.13333333333333333
P_YStable_given_YRobust = 0.9416666666666667
P_h80_weak_given_YRobust = 0.175
P_h80_bad_given_YRobust = 0.058333333333333334
P_h240_longrisk_given_h20weak = 0.7156398104265402
V_ctrl_h20_lcb_YRobust = 0.3013685750004344
V_ctrl_h80_lcb_YRobust = -0.8039138915476802
V_ctrl_h240_lcb_YRobust = 0.08296381891423209
objective_conflict_class = OC1-YRobustHasH80BlindSpot
p1_pass = 1
```

判断：canonical label audit 过，但它也说明 YRobust 不是 all-horizon strict label。YRobust 与 YStableHorizon 高重叠，但与 YStrictAllH 只有 `0.1333` Jaccard。

## 4. P2 high-capacity legal upper-bound probe

Artifact：

```text
p2_high_capacity_legal_upper_bound_probe.csv
```

Summary：

```text
probe_count = 3
best_probe_id = UB0-ScalarFeatureProbe
UB2_AUC_YRobust = 0.51014755684567
UB2_AUC_LongRisk = 0.5819000791470454
UB2_TopK64_YRobust_precision = 0.015625
UB2_TopK64_LongRisk_rate = 0.65625
UB2_LDO_AUC_drop_max = 0.052348278597785514
UB2_LSO_AUC_drop_max = 0.011086563857551524
legal_upper_bound_probe_pass = 0
legal_upper_bound_probe_weak_pass = 0
```

Probe rows：

| probe | AUC_YRobust | TopK64 YRobust | TopK64 LongRisk | pass |
|---|---:|---:|---:|---:|
| `UB0-ScalarFeatureProbe` | `0.5375105829704886` | `0.03125` | `0.546875` | `0` |
| `UB1-TensorSketchCentroidProbe` | `0.4966134494436381` | `0.03125` | `0.6875` | `0` |
| `UB2-HighCapacityLegalKNNProbe` | `0.51014755684567` | `0.015625` | `0.65625` | `0` |

判断：即使用高容量 legal KNN/centroid probe，YRobust 仍无法被 commit-time legal signal 稳定识别；不能继续走 distillation 或 controller。

## 5. P3/P4 mechanism 与 distillation

P3 artifact：

```text
p3_robust_source_mechanism_anatomy.csv
```

P3 summary：

```text
positive_count = 120
negative_count_total = 489
cluster_count = 4
best_cluster_id = C4-SupportTailPrototype
best_cluster_purity_YRobust = 0.046875
best_cluster_longrisk_rate = 0.75
best_mechanism_candidate = M4-support-memory-pattern
prototype_count = 16
prototype_reconstruction_error = 0.953125
dominant_miss_reason = M8-outcome-only-pattern
mechanism_pass = 0
```

P4 artifact：

```text
p4_feature_distillation_if_legal_upper_bound_exists.csv
```

P4 summary：

```text
stage = P4_FEATURE_DISTILLATION_IF_LEGAL_UPPER_BOUND_EXISTS
status = not_run
reason = P2_legal_upper_bound_probe_failed
distillation_pass = 0
```

判断：P3 没有找到可以被 legal feature factory 复用的稳定 mechanism；P4 因 P2 fail 正确阻断，没有把失败的 upper-bound probe 强行蒸馏成 official selector。

## 6. P5 constructive source generator v3

Artifacts：

```text
p5_constructive_source_generator_v3.csv
constructive_generator_trace_v9500.csv
```

Summary：

```text
generator_panel_count = 4
generated_action_count_total = 256
branch_horizon_rows_expected/actual = 4608 / 4608
branch_horizon_completion_rate = 1.0
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_error_linf_max = 0.0
certificate_fields_complete = 1
commit_time_available = 1
uses_dataset_name = 0
uses_outcome_at_commit = 0
uses_future_outcome = 0
best_source_panel_id = S6-ConstructiveLegalSeedPanel
best_generator_id = SG2-HorizonGuardedTrustRegion
best_h20_weak_CP = 0.1875
best_h20_V_ctrl_lcb = -1.5456723592527004
best_h80_V_ctrl_lcb = -1.6351651850951239
best_h240_V_ctrl_lcb = -1.1452243531703268
best_h240_longrisk = 0.71875
best_YRobust_precision = 0.0625
best_YStableHorizon_precision = 0.0625
best_YStrictAllH_precision = 0.015625
best_source_positive_lost_rate = 0.8285714285714286
best_Damage_median = -0.13573043805081397
constructive_generator_pass = 0
```

Per generator：

| generator | h20 weak CP | h20 V_ctrl LCB | h240 long-risk | YRobust precision |
|---|---:|---:|---:|---:|
| `SG1-LinearizedTailDescentConstrained` | `0.15625` | `-1.334099663834028` | `0.84375` | `0.015625` |
| `SG2-HorizonGuardedTrustRegion` | `0.1875` | `-1.5456723592527004` | `0.71875` | `0.0625` |
| `SG3-PrototypeProjectedRobustSource` | `0.125` | `-1.6716476608300255` | `0.796875` | `0.015625` |
| `SG4-AdamWOrthogonalTailRepair` | `0.09375` | `-2.038026067171468` | `0.75` | `0.03125` |

判断：SG1-SG4 的 materialization/apply/outcome 链路是真实闭合的，但 value/horizon 指标均未过。不能把 self-certifying fields 或 durable payload 写成 source generator pass。

## 7. P6 certificate v3

Artifact：

```text
p6_effect_valid_certificate_v3.csv
```

Summary：

```text
certificate_count = 5
base_rate_YRobust = 0.04172461752433936
best_certificate_id = CERT18-MinimalDistilledLegalCertificate
best_AUC_YRobust = 0.5284178387150466
best_AUC_LongRisk = 0.5673022703202085
best_TopK64_YRobust_precision = 0.015625
best_TopK64_LongRisk_rate = 0.796875
best_ECE_YRobust = 0.6570605011883082
certificate_effect_valid_pass = 0
```

判断：CERT14-CERT18 没有形成对 YRobust / long-risk 的有效分离；certificate TopK64 的 YRobust precision 低于 base rate，且 long-risk 高，不能进入 source controller。

## 8. P7-P13 boundary

P7：

```text
p7_minimal_source_controller_candidate.csv = not_run
reason = P4_or_P6_certificate_not_effect_valid
source_controller_pass = 0
```

P8：

```text
runtime_candidate_id = RT-v9500-preflight
controller_id = not_selected
certificate_id = CERT18-MinimalDistilledLegalCertificate
generator_id = SG2-HorizonGuardedTrustRegion
runtime_preflight_pass = 1
official_runtime_pass = 0
```

P9：

```text
system_candidate_id = SYS-v9500-self-certifying-primitive
decision/runtime/certificate/generator gate = 0/0/0/0
system_legal_controller_pass = 0
official_eligible = 0
```

P10-P13：

| artifact | status / reason |
|---|---|
| `p10_leave_dataset_stratum_out_boundary.csv` | `not_run`, `P9_system_controller_not_official` |
| `p11_diagnostic_paired_replay_scout.csv` | `not_run`, `P9_system_controller_not_official` |
| `p12_official_paired_replay_boundary.csv` | `not_run`, `P11_scout_not_open` |
| `p13_short_full_functional_training_boundary.csv` | `not_run`, `P12_official_paired_replay_not_open` |

判断：runtime preflight 只是 timed path diagnostic；controller 未选中，所以 paired replay、leaveout、short/full validation 都必须关闭。

## 9. P14 Base-Acc Sentinel

Artifact：

```text
p14_base_acc_sentinel_continuation.csv
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
LQ_minus_MLP_mean_test_acc = 0.09088541666666672
LQ_minus_AdamWStrongLRGridMLP_mean_test_acc = 0.025260416666666674
LQ_catastrophic_fail = 0
base_acc_used_for_controller = 0
base_acc_reused_from_v9470/v9480/v9490 = 1/1/1
```

判断：Base-Acc Sentinel 继续健康，但仍是 isolated diagnostic，没有用于 selector、generator、certificate 或 controller。

## 10. No-fake audit

```text
rows_checked = 7683
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
canonical_truth_pass = 1
multi_objective_label_audit_pass = 1
legal_upper_bound_probe_pass = 0
mechanism_pass = 0
distillation_pass = 0
constructive_generator_measured = 1
constructive_generator_pass = 0
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
route = R9-PrimitiveFamilyResetRequired
F0_boundary_regression = 0
F1_legal_upper_bound_fail = 1
F2_distillation_fail = 0
F3_mechanism_generator_fail = 0
F4_constructive_generator_certificate_fail = 0
F5_runtime_fail = 0
F6_primitive_family_reset_required = 1
F7_system_not_official = 1
F8_base_acc_catastrophic = 0
primary_blocker = legal_upper_bound_mechanism_generator_certificate_all_fail
```

## 11. Hash

| artifact | SHA256 |
|---|---|
| plan | `05d270e36132f03635ae5d9b2aca8c514a5ebae1325f410d361acc3c38034c59` |
| runner | `7d5a41603c5d4fcac4be79ebe4b48607c696dd5e066f14595f419b738a73650f` |
| run manifest | `63fea4de92624882e91484136af505aa6985dee0a32929e436bcdee61c9ed52b` |
| route | `51891dffcfaefd4319f35f19078f07080749a70cdd6d776b2f0f48cb8efae2b6` |
| P0 boundary | `5f5279c49e60c83b6758371c7851940593f0ad1be504502688b82fae4bb1e3b4` |
| P1 label audit | `bf940fd6d84b17be89dc237e468791841a25d1c8a6ff424f2084c623e2ac7664` |
| P2 upper-bound probe | `97640923bcb0e20d26f24b0fa2a7496449b5bc00962905aaaf6147d5e43c4a89` |
| P3 mechanism anatomy | `e851485e55372a0b16f23ab86de1b3c668065331a1fe95b730e3118711a2405b` |
| P4 distillation | `15691b89958f5de2ff93f418feb7baaaa7ad8e2e3e6a2ea121d9a1843986f641` |
| P5 constructive generator | `5b888fcf128e0dc415d02d329e343c12f6b33604ee2fcf55ce956d2492d1b647` |
| P6 certificate | `c890d8d840ca4108443a3426baa993da140192e8251876657ee0dfc993e8ed51` |
| P7 controller | `2e6de85a2b9a62b01b6cd7546dab08fe1834d1e795fd3fb09e09353f1105f918` |
| P8 runtime | `e7040b4d936a24d54fb2f9fe81aa92e7e7faa9ce3eab35ae0bb04c8b27d7ec8f` |
| P9 system | `9e7d1bc3b6e2d0b4233e178a9d87ebb02729af256425f533d340211c66ef44b6` |
| P14 Base-Acc Sentinel | `d0df97c5e3b3b142efa2dc3fed408a922dbc6122ba24e03c352fbbe46a6d581f` |
| no-fake audit | `a729b7d4f53c1c2df4e1c60888926a6b0814aa830f0916e21f2d75bdbaa9bbe8` |
| contract audit | `bb009d0e805e196618efc9765fec50a56067d427d6dc00459c81ccf79a8bd2ed` |
| failure table | `3f5d59d6c7750bf8d645605ef9cef542a3a645d2b167b6ada00603128b02afad` |

## 12. 最终分析结论

v9.5.0 的真实推进是：

```text
v9.4.9:
  canonical AP0 robust source frontier 确认存在；
  但 legal features、真实 generator、certificate 均未转成 controller。

v9.5.0:
  明确 YRobust / StableHorizon / StrictAllH 的关系；
  使用高容量 legal upper-bound probe 检验 commit-time observability；
  审计 robust source mechanism anatomy；
  实测 SG1-SG4 self-certifying constructive generators 与 CERT14-CERT18；
  但 legal upper-bound、mechanism、generator、certificate 全部没有达到 official gate。
```

机制判断：

1. H0 成立：v9.4.9 boundary 被复现，canonical truth base 可用。
2. H1 成立：multi-objective label audit 闭合，并暴露 YRobust 的 h80 blind spot。
3. H2 未成立：high-capacity legal upper-bound probe 也看不见 YRobust，UB2 AUC 只有 `0.5101`，TopK64 precision 只有 `0.015625`。
4. H3 未成立：mechanism anatomy 没有找到 legal-stable cluster，best purity 只有 `0.046875`。
5. H4 未打开：P2 fail 后 feature distillation 正确 not_run。
6. H5 未成立：constructive generator v3 落盘完整，但 best SG2 的 V_ctrl LCB 为负、long-risk 高。
7. H6 未成立：certificate v3 没有 effect-valid separation。
8. H7-H9 未打开：没有 controller，就不能打开 official runtime、paired replay 或 short/full validation。
9. Base-Acc Sentinel 继续健康，但没有用于 controller，也不是 functional success。

最终一句话：

> v9.5.0 真实执行后停在 `R9-PrimitiveFamilyResetRequired`：canonical robust frontier 仍存在，但即使引入 high-capacity legal probe、mechanism anatomy、SG1-SG4 self-certifying primitive 和 CERT14-CERT18，也没有得到可 official 的 selector/generator/certificate/controller；strict PureKAN functional 仍未成功，下一步需要重置 primitive family。
