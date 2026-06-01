# DG-KAN v9.4.5 Robust Source Generator / Objective-Aligned Certificate / Parallel Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.4.5_RobustSourceGenerator_ObjectiveAlignedCertificate_ParallelClosure_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 oracle survivor、no-transform diagnostic、generator smoke、certificate diagnostic 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R1-OracleSourceIdentityBug
base_candidate = LQ-t2-h256
success_v9450_strict_purekan_functional = False
success_v9450_full_functional = False
success_v9450_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9450_robust_source_generator_objective_aligned_certificate_parallel_closure_first_20260514T120000Z/
```

核心结论：

1. P0 复现 v9.4.4 boundary：source route = `R3-GeneratorDestructive`，v9.4.4 best oracle = `ORC-D-HorizonRobust-K16`。
2. P1 继续确认 objective mismatch：`Y_robust_base_rate = 0.02121001390820584`，`P(LongRisk_h240 | WeakCP_h20) = 0.8013029315960912`，weak CP 不能等同于 horizon-safe source objective。
3. P2 oracle survivor anatomy 过诊断 gate：oracle action count = `16`，dominant legal miss reason = `M8-outcome-only-pattern`。
4. P3 generator preflight 全过：preflight row = `30`，pass = `30`，unresolved exception = `0`；payload/certificate/action-apply/branch-horizon 基础链路不是本轮 blocker。
5. P4 oracle-seeded AP0r-AP0w 真实运行：generated action = `96`，branch-horizon rows = `2592`。
6. P4 best transform 仍失败：best = `AP0r-ObjectiveSolvedLastEdgeTrustRegionSource`，h20 weak CP = `0.125`，h20 V_ctrl LCB = `-0.8833607173563967`，h240 long-risk = `0.75`。
7. P4 最关键异常：`AP0w-NoTransformReplaySource` no-transform diagnostic 也未复现 oracle survivor，h20 weak CP = `0.0`，h20 V_ctrl LCB = `-1.714831942008832`，h240 long-risk = `0.625`，source-positive lost rate = `0.8444444444444444`。
8. 因 no-transform oracle-seeded failed，本轮 route 正确停在 `R1-OracleSourceIdentityBug`；不能继续把失败主要归因于 AP0r-AP0v transform。
9. P5 legal/representative seeded generator 也未过：generated action = `320`，rows = `8640`，best = `S1_legal_topK_state_NLL / AP0u-AdamWCompatibleResidualSource`，h20 weak CP = `0.0`。
10. P6 direct AP0r-AP0v 也未过：generated action = `64`，rows = `1728`，best = `AP0t-TailProjectedHorizonGuardSource`，h20 weak CP = `0.125`，h20 V_ctrl LCB = `-0.7052174296088407`，h240 long-risk = `0.875`。
11. P7 effect-valid certificate 未过：best certificate = `CERT8-AdamWConflictHorizonGuard`，AUC robust source = `0.549163179916318`，AUC long-risk = `0.4924701402111823`。
12. P10 Base-Acc Sentinel continuation 完成：rows = `120`，LQ mean test acc = `0.6537760416666667`，MatchedMLP = `0.562890625`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
13. P11 system controller 未打开：official eligible = `0`，reason = `no_transform_oracle_seeded_failed`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9450_robust_source_generator_objective_aligned_certificate.py` | v9.4.5 runner；读取 v9.4.4/v9.3.5/v9.3.3 artifacts，执行 robust source objective、oracle survivor anatomy、generator preflight、oracle/legal/direct seeded generators、objective-aligned certificate、Base-Acc Sentinel continuation 与 system gate |

代码检查：

```text
python -m py_compile experiments/run_v9450_robust_source_generator_objective_aligned_certificate.py
```

正式运行：

```bash
python experiments/run_v9450_robust_source_generator_objective_aligned_certificate.py \
  --out-dir results/real_rerun_20260506/v9450_robust_source_generator_objective_aligned_certificate_parallel_closure_first_20260514T120000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --oracle-actions 16 --seeded-actions 16 --direct-actions-per-generator 16 \
  --sentinel-seeds 0,1,2,3,4,5,6,7,8,9
```

运行结果：

```json
{
  "direct_pass": 0,
  "oracle_no_transform_pass": 0,
  "out_dir": "results/real_rerun_20260506/v9450_robust_source_generator_objective_aligned_certificate_parallel_closure_first_20260514T120000Z",
  "route": "R1-OracleSourceIdentityBug",
  "sentinel_rows": 120,
  "transform_pass": 0
}
```

输入 artifact：

```text
source_v9440 = results/real_rerun_20260506/v9440_source_value_objective_audit_oracle_seeded_generator_reset_first_20260514T110000Z
source_v9350 = results/real_rerun_20260506/v9350_control_positive_frontier_completion_legal_probe_runtime_first_20260514T023000Z
source_v9330 = results/real_rerun_20260506/v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z
```

## 2. Route

`route_decision.json` 摘要：

```json
{
  "route": "R1-OracleSourceIdentityBug",
  "source_route_v9440": "R3-GeneratorDestructive",
  "objective_mismatch_pass": 1,
  "Y_robust_base_rate": 0.02121001390820584,
  "P_LongRisk_h240_given_WeakCP_h20": 0.8013029315960912,
  "oracle_survivor_anatomy_pass": 1,
  "dominant_oracle_legal_miss_reason": "M8-outcome-only-pattern",
  "preflight_all_pass": 1,
  "oracle_seeded_no_transform_pass": 0,
  "oracle_seeded_transform_preservation_pass": 0,
  "oracle_seeded_best_generator": "AP0r-ObjectiveSolvedLastEdgeTrustRegionSource",
  "oracle_seeded_best_h20_weak_CP": 0.125,
  "oracle_seeded_best_h20_V_ctrl_LCB": -0.8833607173563967,
  "oracle_seeded_best_h240_longrisk": 0.75,
  "legal_representative_official_candidate_pass": 0,
  "direct_generator_pass": 0,
  "certificate_effect_valid_pass": 0,
  "base_acc_sentinel_pass": 1,
  "source_controller_pass": 0,
  "selected_runtime_pass": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "no_transform_oracle_seeded_failed"
}
```

判断：v9.4.5 没有回退 v9.4.4 的 oracle survivor 结论，但 no-transform replay path 对 oracle survivor 没有复现 value/horizon-safe outcome。因此本轮不能继续推广 AP0r-AP0v transform，也不能把 certificate 或 runtime 打开。

## 3. P0 v9.4.4 boundary reproduction

Artifact：

```text
p0_v9440_boundary_reproduction.csv
```

Summary：

```text
route_v9440 = R3-GeneratorDestructive
best_oracle_selector = ORC-D-HorizonRobust
best_oracle_K = 16
best_oracle_h20_weak_CP = 0.8125
best_oracle_h20_V_ctrl_lcb = 0.13772944106165844
best_oracle_h240_long_risk = 0.0
legal_selector_capacity_pass = 0
best_legal_feature_id = LS-E-state_NLL_proxy
best_legal_topK_h20_weak_CP = 0.125
oracle_seeded_source_positive_lost_rate = 0.8888888888888888
direct_best_primitive = AP0l-LinearizedTrustRegionSource
direct_best_h20_weak_CP = 0.125
certificate_AUC_weak_CP = 0.5015054877456535
base_acc_sentinel_complete = 1
system_legal_controller_pass = 0
p0_pass = 1
```

判断：P0 pass。上轮边界被复现，v9.4.5 是在 v9.4.4 的 destructive generator blocker 上继续推进。

## 4. P1 robust source objective decomposition

Artifact：

```text
p1_source_objective_decomposition_v2.csv
```

Summary：

```text
action_count = 2876
WeakCP_h20_rate = 0.10674547983310154
V_ctrl_h20_LCB = -0.7504334275695862
LongRisk_h240_rate = 0.9022948539638387
P_Vctrl_positive_given_WeakCP_h20 = 1.0
P_LongRisk_h240_given_WeakCP_h20 = 0.8013029315960912
P_Yrobust_given_WeakCP_h20 = 0.1986970684039088
Y_robust_base_rate = 0.02121001390820584
objective_mismatch_pass = 1
```

判断：P1 继续支持 v9.4.4 的 source objective reset。weak CP 在 h20 上与 V_ctrl 有信号，但它不能排除 h240 long-risk，因此 robust source 目标必须显式包含 horizon safety。

## 5. P2 oracle survivor anatomy

Artifact：

```text
p2_oracle_survivor_anatomy.csv
```

Summary：

```text
oracle_action_count = 16
joined_with_legal_feature_count = 16
miss_reason_assigned_fraction = 1.0
dominant_miss_reason = M8-outcome-only-pattern
dominant_miss_reason_count = 6
p2_pass = 1
```

判断：P2 pass。oracle survivor 可以被 join 到 legal feature table，但现有 legal feature/rank 仍无法识别它；主 miss reason 是 outcome-only pattern，不能作为 official selector。

## 6. P3 generator preflight matrix

Artifact：

```text
p3_generator_preflight_matrix.csv
```

Summary：

```text
preflight_row_count = 30
preflight_pass_count = 30
preflight_all_pass = 1
unresolved_exception_count = 0
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_error_linf_max = 0.0
branch_horizon_completion_rate = 1.0
```

判断：P3 pass。AP0r-AP0w / legal representative / direct generator 的 smoke 链路均可写 payload、certificate、action apply replay 和 branch-horizon rows；本轮失败不是 row sink 或 apply missing。

## 7. P4 oracle-seeded generator preservation

Artifacts：

```text
p4_oracle_seeded_generator_preservation.csv
oracle_seeded_generator_trace_v9450.csv
ap0r_ap0w_oracle_payload_shards_v9450/
```

Summary：

```text
source_panel_type = S0_oracle_K16_diagnostic_only
generator_panel_count = 6
generated_action_count_total = 96
branch_horizon_row_count_actual = 2592

best_generator_id = AP0r-ObjectiveSolvedLastEdgeTrustRegionSource
best_h20_weak_CP = 0.125
best_h20_V_ctrl_LCB = -0.8833607173563967
best_h240_longrisk = 0.75
best_source_positive_lost_rate = 0.8444444444444444
best_Damage_median = -0.5820385176921263
any_preservation_pass = 0
any_transform_preservation_pass = 0
```

No-transform diagnostic：

```text
generator_id = AP0w-NoTransformReplaySource
h20_weak_CP = 0.0
h20_V_ctrl_LCB = -1.714831942008832
h80_weak_CP = 0.1875
h240_weak_CP = 0.25
h240_longrisk = 0.625
source_positive_lost_rate = 0.8444444444444444
Damage_median = -1.1904930071905255
generator_preservation_pass = 0
```

判断：P4 是本轮 terminal blocker。按照计划，AP0w no-transform 应作为 identity/replay sanity diagnostic；它也失败，说明需要先审计 oracle source identity、payload replay 与 outcome join/replay consistency，不能把 AP0r-AP0v transform 调参推进成 official。

## 8. P5 legal/representative seeded generator

Artifacts：

```text
p5_legal_representative_seeded_generator.csv
legal_representative_generator_trace_v9450.csv
```

Summary：

```text
generator_panel_count = 20
generated_action_count_total = 320
branch_horizon_row_count_actual = 8640
best_panel_id = S1_legal_topK_state_NLL
best_generator_id = AP0u-AdamWCompatibleResidualSource
best_h20_weak_CP = 0.0
best_h20_V_ctrl_LCB = -0.2707542846308277
best_h240_longrisk = 0.6875
best_source_positive_lost_rate = 0.75
best_Damage_median = -0.11289792868774384
legal_representative_official_candidate_pass = 0
legal_representative_weak_pass = 0
```

判断：P5 未过。即便不使用 oracle seed，当前 legal/representative seeded route 也没有形成 h20 value-positive 或 h240 safe frontier。

## 9. P6 direct objective-solved generator

Artifacts：

```text
p6_direct_objective_solved_generator.csv
direct_objective_generator_trace_v9450.csv
ap0r_ap0v_direct_payload_shards_v9450/
```

Summary：

```text
generator_panel_count = 4
generated_action_count_total = 64
branch_horizon_row_count_actual = 1728
best_generator_id = AP0t-TailProjectedHorizonGuardSource
best_h20_weak_CP = 0.125
best_h20_V_ctrl_LCB = -0.7052174296088407
best_h240_longrisk = 0.875
direct_generator_pass = 0
```

判断：P6 未过。direct AP0r-AP0v implementation 落盘，但 objective-solved generator 没有产生可用 robust source frontier。

## 10. P7 objective-aligned certificate

Artifacts：

```text
p7_effect_valid_certificate_redesign.csv
```

Summary：

```text
certificate_count = 5
action_count = 480
best_certificate_id = CERT8-AdamWConflictHorizonGuard
best_AUC_robust_source = 0.549163179916318
best_AUC_longrisk_h240 = 0.4924701402111823
best_P_Yrobust_given_cert_pass = 0.0
best_P_longrisk_given_cert_pass = 0.0
monotone_sign_pass = 0
certificate_effect_valid_pass = 0
certificate_effect_weak_pass = 0
```

判断：P7 未过。CERT6-CERT10 没有形成 robust source / long-risk 的有效分离，且 certificate pass rows 没有打开可用 controller。

## 11. P8-P14 boundary

P8：

```text
p8_minimal_source_certificate_controller.csv = not_run
reason = upstream_generator_or_certificate_failed
source_controller_pass = 0
```

P9：

```text
p9_selected_source_online_runtime.csv = not_run
reason = P8_controller_not_selected
selected_runtime_pass = 0
```

P11：

```text
system_candidate_id = SYS-v9450-robust-source-generator
controller_id = not_selected
generator_id = AP0u-AdamWCompatibleResidualSource
certificate_id = CERT8-AdamWConflictHorizonGuard
runtime_candidate_id = not_selected
official_eligible = 0
system_legal_controller_pass = 0
reason = no_transform_oracle_seeded_failed
```

P12-P14：

| artifact | status / reason |
|---|---|
| `p12_leaveout_boundary_v9450.csv` | `not_run`, `P11_system_controller_not_official` |
| `p13_official_paired_replay_v9450.csv` | `not_run`, `P11_system_controller_not_official` |
| `p14_short_full_training_boundary_v9450.csv` | `not_run`, `P11_system_controller_not_official` |

判断：没有把 no-transform diagnostic、generator smoke、certificate diagnostic 或 Base-Acc Sentinel 写成 official controller/runtime/downstream pass。

## 12. P10 Base-Acc Sentinel continuation

Artifacts：

```text
p10_base_acc_sentinel_strong_baseline_continuation.csv
base_acc_training_trace_v9450.csv
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
base_acc_sentinel_pass = 1
```

判断：Base-Acc Sentinel 没有发现 LQ catastrophic fail；但它仍是 isolated health/baseline diagnostic，没有用于 selector、generator、certificate 或 controller。

## 13. No-fake audit

```text
rows_checked = 18053
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
objective_decomposition_v2_pass = 1
oracle_survivor_anatomy_pass = 1
generator_preflight_all_pass = 1
oracle_seeded_generator_measured = 1
oracle_seeded_transform_preservation_pass = 0
no_transform_diagnostic_promoted_to_official = 0
legal_representative_generator_measured = 1
legal_representative_official_candidate_pass = 0
direct_generator_measured = 1
direct_generator_pass = 0
certificate_effect_valid_pass = 0
base_acc_sentinel_pass = 1
base_acc_used_for_controller = 0
source_controller/selected_runtime/system = 0/0/0
uses_dataset_name_for_selector/controller = 0/0
uses_validation_or_test_for_controller = 0
uses_future_outcome_for_features = 0
uses_outcome_at_commit = 0
diagnostic_promoted_to_official = 0
fake/proxy/cpu_offload = 0/0/0
```

Failure table：

```text
route = R1-OracleSourceIdentityBug
F1_oracle_source_identity_bug = 1
F2_transform_still_destructive = 0
F3_selector_opaque_but_generator_can_preserve = 0
F4_direct_generator_objective_fail = 1
F5_certificate_effect_fail = 1
F6_runtime_fail = 0
F7_base_catastrophic = 0
F8_system_not_official = 1
primary_blocker = no_transform_oracle_seeded_failed
```

## 14. Hash

| artifact | SHA256 |
|---|---|
| plan | `89608f55c463143996010cb5e953a6023ed4e15d358bf4daee6b6311ede1444f` |
| runner | `090f07debc3cd9241b7a710020dab0a970b45fb2717d0b67d01d18298e68005d` |
| run manifest | `89b5a7560cb8e4fa5027194511fbb3ac71fb949328c8879fa06613afd72e4eb3` |
| route | `296d98dc2eb8c3979ee068933e643d80159335752231501c5780c16fa00c6689` |
| P0 boundary | `a41b23fd621c4736a6a0ec689ca8eab66721a38acef14884289313f941ea4caa` |
| P1 objective | `7f4f8c4489566deb049db00a25be87fb33f7bb7875394c9297e8c393234d232f` |
| P2 oracle anatomy | `1cc01e2ccce2761b02f4f0a89b30ccdc7f52a1b393785d6e033a63cd3843100a` |
| P3 preflight | `4f14ebec3e3bd93df3413923bc227f961943be37f723537e43359c46cb37ecb9` |
| P4 oracle-seeded generator | `3dafa961bf04f6ae13971d926b4fbea6cad9be400d657f938771329446d632ee` |
| P5 legal representative generator | `121752a0eb3e3ce7c8aa39e21ce36ec6e56cfe7fc244aaa4b8efb605210ff495` |
| P6 direct generator | `0431aafa03f06441faec8220961c73dea5b8ab1290d905c356149f136a41df38` |
| P7 certificate | `ba823f1140495e50858f0ddcdfce9f11b7e3aa3bda9eaed805e1dd32dea5e72a` |
| P8 controller | `4bc8ccb477b3852f0059dafab87e1ee5f22a69fb13e98bdd44d479ee0f809a2e` |
| P9 runtime | `1c5b0665a011ac7b7beb50616f3c0856a824023ebcfdcdea6af1755892e57248` |
| P10 Base-Acc Sentinel | `417b4a761a73a48b133520a1795538ce24dc817a9a0d0be29b9d7a24fbad50fb` |
| P11 system | `127c8863d5808defcda55e7baf7a39910046a45302181f4a631237953bb78f52` |
| contract audit | `fee0b27e719e5ba575f60e959cc76cdd3ea211e83085bcf750c858a70b8309d2` |
| provenance audit | `334e4bb246c1fb5e4c6bb7e0954db3f12bda3d0188891570700704132671c590` |
| failure table | `6b6a2fada1bd1cde64ec4e8ff4bb7edee9980a79b22cdefbd355a035ea052683` |

## 15. 最终分析结论

v9.4.5 的真实推进是：

```text
v9.4.4:
  full AP0 universe 存在 K16 horizon-robust oracle survivor；
  但 legal selector 看不见它，现有 source generator 会破坏它。

v9.4.5:
  进一步重定义 robust source objective；
  审计 oracle survivor 的 legal miss reason；
  证明 generator/apply/outcome preflight 链路可运行；
  但 AP0w no-transform replay 也没复现 oracle survivor outcome。
```

机制判断：

1. H1 成立：weak CP 与 robust source objective 不等价，weak CP rows 中 h240 long-risk 条件概率仍为 `0.8013`。
2. H2 成立但只能诊断：oracle survivor legal miss reason 可归因，dominant = `M8-outcome-only-pattern`。
3. H3 未成立且暴露更早 blocker：AP0w no-transform oracle-seeded diagnostic 失败，说明 source identity / replay / outcome join consistency 需要先闭合。
4. H4 未成立：AP0r-AP0v transform 没有 preservation pass；但由于 AP0w 也失败，不能把 transform destruction 写成本轮 primary blocker。
5. H5 未成立：legal/representative seeded generator 没有 official survivor。
6. H6 未成立：direct objective-solved generator 没有形成 robust source frontier。
7. H7 未成立：CERT6-CERT10 没有 effect-valid separation。
8. H8 继续成立：Base-Acc Sentinel 没有 catastrophic fail，但没有用于 controller。
9. P11 未打开：不能进入 selected runtime、official paired replay 或 short/full functional validation。

最终一句话：

> v9.4.5 真实执行后停在 `R1-OracleSourceIdentityBug`：robust source objective 与 oracle survivor 诊断仍成立，但 AP0w no-transform replay 都无法复现 oracle survivor 的 horizon-safe value；下一步必须先审计 source identity / payload replay / outcome join consistency，再继续 generator 或 certificate 设计。
