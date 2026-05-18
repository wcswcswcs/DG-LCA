# DG-KAN v9.4.4 Source Value Objective Audit / Oracle-Seeded Generator Reset / Parallel Baseline 实验复盘

> 本复盘记录 `DG-KAN_v9.4.4_SourceValueObjectiveAudit_OracleSeededGeneratorReset_ParallelBaseline_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 exhaustive oracle source survivor、oracle-seeded generator smoke、direct generator smoke、effect certificate diagnostic 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R3-GeneratorDestructive
base_candidate = LQ-t2-h256
success_v9440_strict_purekan_functional = False
success_v9440_full_functional = False
success_v9440_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9440_source_value_objective_audit_oracle_seeded_generator_reset_first_20260514T110000Z/
```

核心结论：

1. P0 复现 v9.4.3 boundary：source route = `R2-FullAP0OracleSourceFrontierAbsent`，source outcome materializer closed = `1`，v9.4.3 system pass = `0`。
2. P1 source value objective audit 过诊断 gate：full AP0 universe h20 weak CP rate = `0.10674547983310154`，V_ctrl h20 LCB = `-0.7504334275695862`，`P(LongRisk_h240 | WeakCP_h20) = 0.8013029315960912`。
3. P1 说明 weak CP 与 value/horizon objective 不等价：`P(Vctrl_h20 > 0 | WeakCP_h20) = 1.0`，但 weak CP rows 大量落入 h240 long-risk。
4. P2 exhaustive oracle source frontier 找到 survivor：best = `ORC-D-HorizonRobust-K16`，h20 weak CP = `0.8125`，h20 V_ctrl LCB = `0.13772944106165844`，h240 long-risk = `0.0`。
5. P2 纠正 v9.4.3 的 source-frontier absent 结论：full AP0 universe 中存在 oracle survivor；但它是 outcome/oracle diagnostic，不能 official。
6. P4 legal selector 仍失败：best legal feature = `LS-E-state_NLL_proxy`，topK h20 weak CP = `0.125`，V_ctrl LCB = `-0.2595459073949538`，h240 long-risk = `0.828125`。
7. P5 oracle-seeded AP0b-AP0f generator 真实运行：generated action = `80`，branch-horizon rows = `2160 / 2160`，但 source-positive lost rate = `0.8888888888888888`，Damage median = `-0.8996349092340097`。
8. P5 route blocker：oracle survivor 经过现有 AP0b-AP0f transform 后被破坏，best generated primitive 仍只有 h20 weak CP = `0.125`，V_ctrl LCB = `-0.81038825849006`。
9. P6 direct AP0l-AP0q generator 也未过：generated action = `96`，rows = `2592 / 2592`，best = `AP0l-LinearizedTrustRegionSource`，h20 weak CP = `0.125`，V_ctrl LCB = `-0.8829934048526893`，h240 long-risk = `0.75`。
10. P7 effect certificate 不充分：AUC weak CP = `0.5015054877456535`，AUC long-risk = `0.4984089703765437`，monotone sign pass = `0`。
11. P10 Base-Acc Sentinel strong baseline 完成：rows = `120`，LQ mean test acc = `0.6537760416666667`，MatchedMLP = `0.562890625`，AdamWStrongLRGridMLP = `0.628515625`；sentinel 没有用于 controller。
12. P11 system controller 未打开：`official_eligible = 0`，reason = `oracle_source_survivor_transform_generator_destructive`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9440_source_value_objective_audit_oracle_seeded_generator.py` | v9.4.4 runner；读取 v9.4.3/v9.4.2/v9.4.1/v9.3.5/v9.3.3 artifacts，执行 source objective audit、exhaustive AP0 oracle frontier、legal selector capacity、oracle-seeded generator preservation、direct generator reset、effect certificate、Base-Acc strong baseline 与 system gate |

代码检查：

```text
python -m py_compile experiments/run_v9440_source_value_objective_audit_oracle_seeded_generator.py
```

正式运行：

```bash
python experiments/run_v9440_source_value_objective_audit_oracle_seeded_generator.py \
  --out-dir results/real_rerun_20260506/v9440_source_value_objective_audit_oracle_seeded_generator_reset_first_20260514T110000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --oracle-generator-actions 16 --direct-actions-per-generator 16 \
  --sentinel-seeds 0,1,2,3,4,5,6,7,8,9
```

运行结果：

```json
{
  "best_oracle": "ORC-D-HorizonRobust-K16",
  "direct_rows": 2592,
  "out_dir": "results/real_rerun_20260506/v9440_source_value_objective_audit_oracle_seeded_generator_reset_first_20260514T110000Z",
  "route": "R3-GeneratorDestructive",
  "strong_baseline_rows": 120
}
```

## 2. Route

`route_decision.json` 摘要：

```json
{
  "route": "R3-GeneratorDestructive",
  "source_route_v9430": "R2-FullAP0OracleSourceFrontierAbsent",
  "objective_mismatch_pass": 1,
  "Corr_WeakCP_h20_V_ctrl_h20": 0.3665950128325245,
  "P_Vctrl_positive_given_WeakCP_h20": 1.0,
  "P_LongRisk_h240_given_WeakCP_h20": 0.8013029315960912,
  "exhaustive_oracle_pass": 1,
  "exhaustive_oracle_strong_pass": 1,
  "best_oracle_selector": "ORC-D-HorizonRobust",
  "best_oracle_K": 16,
  "best_oracle_h20_weak_CP": 0.8125,
  "best_oracle_h20_V_ctrl_lcb": 0.13772944106165844,
  "best_oracle_h240_long_risk": 0.0,
  "legal_selector_capacity_pass": 0,
  "best_legal_feature_id": "LS-E-state_NLL_proxy",
  "best_legal_topK_h20_weak_CP": 0.125,
  "best_legal_topK_h20_V_ctrl_lcb": -0.2595459073949538,
  "best_legal_topK_h240_long_risk": 0.828125,
  "oracle_seeded_generated_action_count": 80,
  "oracle_seeded_generator_preserve_pass": 0,
  "oracle_seeded_generator_destructive_fail": 1,
  "oracle_seeded_source_positive_lost_rate": 0.8888888888888888,
  "direct_generated_action_count": 96,
  "direct_generator_pass": 0,
  "direct_best_primitive": "AP0l-LinearizedTrustRegionSource",
  "direct_best_h20_weak_CP": 0.125,
  "direct_best_h20_V_ctrl_lcb": -0.8829934048526893,
  "direct_best_h240_longrisk": 0.75,
  "certificate_effect_valid_pass": 0,
  "certificate_AUC_weak_CP": 0.5015054877456535,
  "certificate_AUC_longrisk": 0.4984089703765437,
  "base_acc_sentinel_pass": 1,
  "base_acc_used_for_controller": 0,
  "mean_test_acc_LQ": 0.6537760416666667,
  "mean_test_acc_MLP": 0.562890625,
  "mean_test_acc_AdamWStrongLRGridMLP": 0.628515625,
  "system_legal_controller_pass": 0,
  "primary_blocker": "oracle_source_survivor_transform_generator_destructive"
}
```

判断：v9.4.4 的关键变化是：AP0 full universe 不是 frontier absent，而是 oracle survivor 找得到、legal selector 看不见、现有 source generator 会破坏 survivor。route 因此停在 `R3-GeneratorDestructive`。

## 3. P0 v9.4.3 boundary reproduction

Artifact：

```text
p0_v9430_boundary_reproduction.csv
```

Summary：

```text
route_v9430 = R2-FullAP0OracleSourceFrontierAbsent
source_outcome_materializer_closed_from_v9420 = 1
PANEL_ORC64_h20_weak_CP = 0.578125
PANEL_ORC64_h20_V_ctrl_lcb = -0.020407727203802614
PANEL_ORC64_h240_long_risk = 0.0
best_legal_panel_id = PANEL-LGL64-AdamWConflictLow
best_legal_h20_weak_CP = 0.171875
best_legal_h20_V_ctrl_lcb = -0.3014420568912851
direct_generated_action_count = 60
direct_branch_horizon_rows = 1620
best_direct_primitive_id = AP0j-LowRankEdgeSafeSource
best_direct_h20_weak_CP = 0.25
best_direct_h20_V_ctrl_lcb = -0.20831185402129682
best_direct_h240_long_risk = 0.8333333333333334
certificate_AUC_weak_CP = 0.5069044879171462
base_acc_sentinel_complete = 1
system_legal_controller_pass = 0
p0_pass = 1
```

判断：P0 复现 v9.4.3 边界，没有跳过 source outcome/materializer failure history。

## 4. P1 source objective decomposition

Artifacts：

```text
p1_source_objective_decomposition.csv
```

Summary：

```text
action_count = 2876
weak_CP_h20_rate = 0.10674547983310154
V_ctrl_h20_mean = -0.7136044722406444
V_ctrl_h20_median = -0.4998842151835561
V_ctrl_h20_lcb = -0.7504334275695862
V_ctrl_h20_p10 = -2.0255753127858043
V_ctrl_h20_p90 = 0.2629032526165247
V_ctrl_h20_outlier_count = 287
V_ctrl_h20_negative_tail_mass = 0.797287899860918
Corr_WeakCP_h20_V_ctrl_h20 = 0.3665950128325245
P_Vctrl_positive_given_WeakCP_h20 = 1.0
P_LongRisk_h240_given_WeakCP_h20 = 0.8013029315960912
objective_mismatch_pass = 1
```

Horizon summary：

```text
h20:  weak_CP = 0.10674547983310154, V_ctrl_lcb = -0.7504334275695862, long_risk = 0.0
h80:  weak_CP = 0.1060500695410292,  V_ctrl_lcb = -0.7403427720034111, long_risk = 0.0
h240: weak_CP = 0.09770514603616133, V_ctrl_lcb = -0.548859027728239,  long_risk = 0.9022948539638387
```

判断：P1 说明 earlier weak CP target 不是足够的 source objective。h20 weak CP 与 h20 V_ctrl 有正相关，但 long-horizon risk 没被 weak CP 排除。

## 5. P2 exhaustive AP0 source frontier

Artifacts：

```text
p2_exhaustive_ap0_source_frontier.csv
```

Summary：

```text
selector_count = 7
K_grid = 16,32,64,96,128,192,256,384,512
weak_source_survivor_pass = 1
strong_source_survivor_pass = 1
best_selector_id = ORC-D-HorizonRobust
best_panel_id = ORC-D-HorizonRobust-K16
best_K = 16
best_h20_weak_CP = 0.8125
best_h20_V_ctrl_lcb = 0.13772944106165844
best_h240_long_risk = 0.0
best_horizon_robust_action_coverage = 0.8125
support_balance_pass = 1
frontier_absent = 0
```

判断：P2 推翻了“full AP0 source frontier absent”作为主 blocker 的说法。AP0 full universe 里存在小而强的 oracle survivor，但该 survivor 使用 outcome/oracle 信息，只能作为 upper-bound diagnosis。

## 6. P3 source objective reset decision

Artifact：

```text
p3_source_objective_reset_decision.csv
```

Summary：

```text
source_objective_route = P4_legal_selector_and_P5_generator
weakCP_Vctrl_mismatch = 1
exhaustive_oracle_pass = 1
exhaustive_oracle_strong_pass = 1
exhaustive_oracle_best_selector = ORC-D-HorizonRobust
exhaustive_oracle_best_K = 16
frontier_absence_confidence = 0
objective_redesign_required = 0
```

判断：P3 没有继续沿 `FullAP0OracleSourceFrontierAbsent`；下一步必须测 legal selector 能否识别 survivor，以及 generator 能否保留 survivor value。

## 7. P4 legal source selector capacity

Artifacts：

```text
p4_legal_source_selector_capacity.csv
```

Summary：

```text
feature_count = 9
best_feature_id = LS-E-state_NLL_proxy
best_AUC_weak_CP = 0.5750092242383822
best_AUC_longrisk = 0.3980992738567873
best_topK_h20_weak_CP = 0.125
best_topK_h20_V_ctrl_lcb = -0.2595459073949538
best_topK_h240_long_risk = 0.828125
legal_selector_capacity_pass = 0
```

判断：P4 fail。当前 legal commit-time features 没有识别 P2 oracle survivor；不能从 oracle diagnosis 直接走 official controller。

## 8. P5 oracle-seeded generator preservation

Artifacts：

```text
p5_oracle_seeded_generator_preservation.csv
```

Summary：

```text
source_panel_id = P2_best_oracle_survivor
generated_action_count = 80
branch_horizon_row_count_expected/actual = 2160 / 2160
branch_horizon_completion = 1
best_primitive_id = AP0d-TailMarginRepairSource
best_h20_weak_CP = 0.125
best_h20_V_ctrl_lcb = -0.81038825849006
best_h240_longrisk = 0.75
Damage_mean = -1.1690044933745716
Damage_median = -0.8996349092340097
source_positive_horizon_count = 225
source_positive_lost_rate = 0.8888888888888888
generator_preserve_pass = 0
generator_destructive_fail = 1
```

判断：P5 是本轮 route 的主 blocker。即使从 P2 oracle survivor 出发，现有 AP0b-AP0f transform 也无法保留 source-positive value，且长 horizon risk 重新升高。

## 9. P6 direct value-producing generator reset

Artifacts：

```text
p6_direct_value_producing_generator_reset.csv
```

Summary：

```text
generated_action_count = 96
branch_horizon_row_count_expected/actual = 2592 / 2592
payload_tensor_written = 1
certificate_tensor_written = 1
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_error_linf_max = 0.0
commit_time_available = 1
uses_dataset_name = 0
uses_outcome_at_commit = 0
best_primitive_id = AP0l-LinearizedTrustRegionSource
best_h20_weak_CP = 0.125
best_h20_V_ctrl_lcb = -0.8829934048526893
best_h240_longrisk = 0.75
source_survivor_pass = 0
```

判断：P6 implementation path 落盘完整，但 direct AP0l-AP0q 没产生 value-positive/horizon-safe frontier。不能把 durable payload/certificate 写成 functional pass。

## 10. P7 effect-valid certificate

Artifacts：

```text
p7_effect_valid_certificate_redesign.csv
```

Summary：

```text
joined_outcome_count = 528
certificate_pass_action_count = 342
AUC_weak_CP = 0.5015054877456535
AUC_strong_CP = 0.5056820906257492
AUC_longrisk = 0.4984089703765437
P_weak_CP_given_cert_pass = 0.13157894736842105
P_weak_CP_given_cert_fail = 0.11827956989247312
P_longrisk_given_cert_pass = 0.2573099415204678
P_longrisk_given_cert_fail = 0.24731182795698925
monotone_sign_pass = 0
certificate_effect_valid_pass = 0
```

判断：certificate 对 weak CP 和 long-risk 几乎没有有效分离，也不满足 monotone sign。P7 不能进入 controller。

## 11. P10 Base-Acc Sentinel strong baseline

Artifacts：

```text
p10_base_acc_sentinel_strong_baseline.csv
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
LQ_minus_QuadraticFeatureMLP_mean_test_acc = 0.2514322916666667
LQ_minus_AdamWStrongLRGridMLP_mean_test_acc = 0.025260416666666674
LQ_catastrophic_fail = 0
AdamWStrongLRGridMLP_diagnostic_materialized = 1
QuadraticFeatureMLP_materialized = 1
base_acc_used_for_controller = 0
base_acc_sentinel_pass = 1
```

判断：Base-Acc Sentinel 说明 LQ-t2-h256 没有 catastrophic base failure；但它只是 isolated sentinel / parallel baseline diagnostic，没有用于 selector、generator 或 controller。

## 12. P11 / downstream boundary

P11：

```text
system_candidate_id = SYS-v9440-source-value-objective-audit
primitive_id = AP0l-LinearizedTrustRegionSource
controller_id = not_selected
certificate_id = effect-valid-v9440
runtime_candidate_id = not_selected
controller_pass = 0
runtime_pass = 0
base_acc_sentinel_pass = 1
strong_baseline_diagnostic_complete = 1
official_eligible = 0
system_legal_controller_pass = 0
reason_if_fail = oracle_source_survivor_transform_generator_destructive
```

P12 downstream 均 gate-block。没有把 P2 oracle、P5/P6 materialized generator rows、P7 certificate diagnostic 或 P10 Base-Acc Sentinel 写成 LDO、paired replay、short/full functional success。

## 13. No-fake audit

```text
rows_checked = 5777
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

Contract audit：

```text
manual_forward/manual_backward/manual_adamw_update = 1/1/1
train_stream_probe = 1
source_outcome_materializer_closed = 1
objective_decomposition_pass = 1
exhaustive_oracle_pass = 1
diagnostic_oracle_used_for_official = 0
legal_selector_capacity_pass = 0
oracle_seeded_generator_measured = 1
oracle_seeded_generator_preserve_pass = 0
direct_generator_measured = 1
direct_generator_pass = 0
certificate_effect_valid_pass = 0
base_acc_sentinel_pass = 1
base_acc_used_for_controller = 0
controller/runtime/system = 0/0/0
uses_dataset_name_for_selector/controller = 0/0
uses_validation_or_test_for_controller = 0
uses_future_outcome_for_features = 0
uses_outcome_at_commit = 0
diagnostic_promoted_to_official = 0
fake/proxy/cpu_offload = 0/0/0
```

Failure table：

```text
route = R3-GeneratorDestructive
F0_objective_mismatch = 1
F1_full_ap0_source_frontier_absent = 0
F2_legal_selector_opaque = 1
F3_generator_destructive = 1
F4_direct_generator_objective_fail = 1
F5_certificate_effect_fail = 1
F6_controller_not_selected = 1
F7_runtime_not_selected = 1
F8_base_acc_catastrophic = 0
primary_blocker = oracle_source_survivor_transform_generator_destructive
```

## 14. Hash

| artifact | SHA256 |
|---|---|
| plan | `148823cd5a2fcbcdbd2264f6bfb78dc7dafe42cc8a4d119d37c06d357993b1ce` |
| runner | `ba3ae9c27a320db3b9e756e565cd645c9c5287dcbf54d538a416fab366143a07` |
| run manifest | `60aa83a76ab255e1cd2b4278b4079da014fb5ae0d987170173036737a66f5c23` |
| route | `9ddcb686e3ec118bb72b1d4edc34a0d99499271384c2c9f38516fec6b9765d4c` |
| P0 boundary | `9bd824d5546f626815bf4653def98be8d698faf90f771e7ea1d2007d95232498` |
| P1 objective | `5025b61a759e3a9eeabdccc6f67d7b52eb43628cea530b908ccbcfada69cf276` |
| P2 exhaustive oracle | `3612bbaa3092097039f04ef07c7a12f9f39171586310915e4b3fdf20646c866d` |
| P3 decision | `02aac5f9ee7f22dc913d726c9fff7c0504511d0a9580e3d4d8965ef649427e7d` |
| P4 legal selector | `b704949a828240756ad4b25478fbc34ed9b56155c0b05c473c08f8af701c5d24` |
| P5 oracle seeded generator | `8e85e483ed527e41446e7f2e2db924d43238c34e9d6440d14de29d7a11fb82b0` |
| P6 direct generator | `0231d3715fae40f20f2ce49cc15773a049531430380aa66b0ce7ca5dd8463fe9` |
| P7 certificate | `5eae3941d64a6dfde04db3ddfd04288cb19877910f29e93a9de8ce15c1165c96` |
| P10 Base-Acc Sentinel | `664defffec59f5ef0743faffa776b11af8597ed0428295589fb73308c1d6e097` |
| P11 system | `8169b44bf4cb76424eaeea65e14853c7227824e2190ebeee6920a0062c464a57` |
| contract audit | `97f813593500f9b22458e3dcbdf449167c384d20be68d5cc477bbc9c61505378` |
| provenance audit | `b0b2560b28b8bbfa0863e8390e35e8eb6e12b777dd71dc2c98dea21cf6fa1e66` |
| failure table | `553739b7ac3e73fada0ad74361acb0012505fd28b0c593af236363b22cf6abba` |

## 15. 最终分析结论

v9.4.4 的真实推进是：

```text
v9.4.3:
  source outcome materializer 已闭合；
  ORC64 panel 接近但 V_ctrl LCB 为负；
  direct generator 和 legal selector 均失败；
  route 被标成 FullAP0OracleSourceFrontierAbsent。

v9.4.4:
  对 full AP0 universe 做 objective audit；
  exhaustive oracle 找到小而强的 K16 horizon-robust source survivor；
  但 legal commit-time selector 仍无法识别 survivor；
  现有 AP0b-AP0f generator 从 oracle survivor 出发仍显著 destructive；
  新 direct AP0l-AP0q generator 和 effect certificate 也未形成 frontier。
```

机制判断：

1. H1 成立：weak CP 与 source value objective 存在 mismatch，尤其不能排除 h240 long-risk。
2. H2 成立但只能诊断：exhaustive oracle survivor 存在，best K16 h20 weak CP = `0.8125`，V_ctrl LCB > `0`，h240 long-risk = `0`。
3. H3 未成立：legal selector capacity fail，best topK h20 weak CP 只有 `0.125`。
4. H4 未成立：oracle-seeded generator destructive，source-positive lost rate = `0.8889`。
5. H5 未成立：direct AP0l-AP0q generator fail，best h20 weak CP = `0.125` 且 V_ctrl LCB 为负。
6. H6 未成立：effect certificate 不具备有效 weak CP/long-risk separation。
7. H7 部分成立：Base-Acc strong baseline complete，LQ 没有 catastrophic fail，但它没有用于 controller。
8. H8 未打开：P11 system controller pass = `0`，不能进入 official paired replay 或 short/full functional validation。

最终一句话：

> v9.4.4 真实执行后停在 `R3-GeneratorDestructive`：full AP0 universe 里确实存在 oracle horizon-robust source survivor，因此 v9.4.3 的 frontier-absent 结论被纠正；但当前 legal selector 看不见它，现有 source generator 又会破坏它，direct generator/certificate 也未闭合，strict PureKAN functional 仍未成功。
