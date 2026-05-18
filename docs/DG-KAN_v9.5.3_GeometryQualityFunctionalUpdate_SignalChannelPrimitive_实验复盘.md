# DG-KAN v9.5.3 Geometry Quality Functional Update / Signal Channel Primitive 实验复盘

> 本复盘记录 `DG-KAN_v9.5.3_GeometryQualityFunctionalUpdate_SignalChannelPrimitive_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 GeometryCard diagnostic、GoodGeomAction diagnostic、SNR gate、APG preflight/smoke、geometry certificate diagnostic、runtime boundary 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R2-NoGoodGeometryActionInCanonicalAP0
base_candidate = LQ-t2-h256
success_v9530_strict_purekan_functional = False
success_v9530_full_functional = False
success_v9530_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9530_geometry_quality_functional_update_signal_channel_first_20260515T030000Z/
```

核心结论：

1. P0 复现 v9.5.2 boundary：source route = `R1-CanonicalActionDensityInsufficientForMultiHorizonTarget`，canonical truth base ready = `1`，v9.5.2 system pass = `0`。
2. P1 GeometryCard table 真实落盘：rows = `3388`，canonical AP0 rows = `2876`，generated APY rows = `512`，required field missing / NaN / Inf 均为 `0`。
3. P2 target-vs-geometry audit 显示 strict target 组有 geometry signal：best group = `T_C_StrictAllH`，snr mean = `0.6534713038783501`，forget risk mean = `0.05570613797463307`，target_vs_geometry_pass = `1`。
4. P3 SNR gate 未过：best gate = `Omega-batch-action`，AUC_GGA = `0.5108043788672061`，TopK64 GGA precision = `0.0`，snr_gate_pass = `0`。
5. P4 cover/curvature audit 未过：GoodGeomAction preliminary group 的 cover/curv/hard-tail stability 不足，cover_curvature_pass = `0`。
6. P5 memory anti-forgetting audit 过 diagnostic：best group = `GoodGeomAction_preliminary`，forget risk mean = `0.052221369818122795`，old family fail count = `0`。
7. P6 GoodGeomAction label 太稀疏：GGA count = `5`，coverage = `0.0017385257301808068`，虽然 h240 long-risk = `0.0`、V_integrated LCB = `0.1307816569689571`，但 GGA weak/official pass 均为 `0`。
8. P7 APY failure geometric autopsy 完成：assigned failure fraction = `1.0`，dominant failure reason = `longrisk_dominant`。
9. P8 APG1-APG8 geometry-aware primitive implementation 过：generated actions = `512`，payload/certificate hash missing = `0 / 0`，action apply L∞ max = `0.0`。
10. P9 APG preflight 与 branch-horizon materializer 过：expected/actual rows = `9216 / 9216`，unresolved exception = `0`，no-transform metric diff = `0.0`。
11. P10 APG outcome 未过：best = `APG1-SNRProjectedEdgeUpdate`，GGA precision = `0.03125`，V_integrated LCB = `-0.4061862277341243`，h240 long-risk = `0.6875`。
12. P11 geometry certificate 未过：best = `GCERT8-MinimalControllerCertificate`，AUC_GGA = `0.9899049685968893`，TopK64 GGA precision = `0.125`，但 geometry_certificate_weak_pass = `0`。
13. P12-P16 gate-blocked：没有 source controller、selected runtime、official paired replay、short/full 或 continual validation。
14. P14 Base-Acc Sentinel 复用前序真实 sentinel：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
15. 当前 primary blocker：`canonical_ap0_good_geom_action_weak_pass_absent`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9530_geometry_quality_functional_update_signal_channel.py` | v9.5.3 runner；读取 v9.5.2/v9.5.1/v9.5.0/v9.4.9/v9.4.8 canonical artifacts，执行 GeometryCard、target-vs-geometry、SNR gate、cover/curvature、memory anti-forgetting、GoodGeomAction、APY autopsy、APG primitive、geometry certificate 与 system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9530_geometry_quality_functional_update_signal_channel.py
```

正式运行：

```bash
python experiments/run_v9530_geometry_quality_functional_update_signal_channel.py \
  --out-dir results/real_rerun_20260506/v9530_geometry_quality_functional_update_signal_channel_first_20260515T030000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --geometry-action-limit 2876 --apg-actions-per-primitive 64
```

运行结果：

```json
{
  "GGA_count": 5,
  "apg_generated_actions": 512,
  "best_apg_GGA_precision": 0.03125,
  "best_apg_primitive": "APG1-SNRProjectedEdgeUpdate",
  "geometry_card_rows": 3388,
  "geometry_certificate_weak_pass": 0,
  "out_dir": "results/real_rerun_20260506/v9530_geometry_quality_functional_update_signal_channel_first_20260515T030000Z",
  "route": "R2-NoGoodGeometryActionInCanonicalAP0",
  "system_legal_controller_pass": 0
}
```

说明：本轮没有 CPU offload，没有 proxy rows；APG branch-horizon smoke 完整落盘且 unresolved exception = `0`。

## 2. Route

`route_decision_v9530.json` 摘要：

```json
{
  "route": "R2-NoGoodGeometryActionInCanonicalAP0",
  "source_route_v9520": "R1-CanonicalActionDensityInsufficientForMultiHorizonTarget",
  "canonical_full_control_outcome_ready": 1,
  "geometry_card_pass": 1,
  "geometry_card_rows": 3388,
  "target_vs_geometry_pass": 1,
  "best_geometry_group": "T_C_StrictAllH",
  "snr_gate_pass": 0,
  "best_snr_AUC_GGA": 0.5108043788672061,
  "best_snr_TopK64_GGA_precision": 0.0,
  "cover_curvature_pass": 0,
  "memory_antiforgetting_pass": 1,
  "GGA_count": 5,
  "GGA_coverage": 0.0017385257301808068,
  "GGA_bad_event_rate": 0.0,
  "GGA_null_event_rate": 0.0,
  "GGA_h240_longrisk": 0.0,
  "GGA_V_integrated_lcb": 0.1307816569689571,
  "GGA_weak_pass": 0,
  "apy_failure_autopsy_pass": 1,
  "dominant_apy_failure_reason": "longrisk_dominant",
  "apg_implementation_pass": 1,
  "apg_preflight_branch_horizon_pass": 1,
  "apg_branch_horizon_rows_actual": 9216,
  "apg_unresolved_exception_count": 0,
  "apg_weak_pass": 0,
  "best_apg_primitive": "APG1-SNRProjectedEdgeUpdate",
  "best_apg_GGA_precision": 0.03125,
  "best_apg_V_integrated_lcb": -0.4061862277341243,
  "best_apg_h240_longrisk": 0.6875,
  "geometry_certificate_weak_pass": 0,
  "best_certificate_id": "GCERT8-MinimalControllerCertificate",
  "best_certificate_AUC_GGA": 0.9899049685968893,
  "best_certificate_TopK64_GGA_precision": 0.125,
  "best_certificate_TopK64_longrisk": 0.0,
  "source_controller_pass": 0,
  "selected_runtime_pass": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "canonical_ap0_good_geom_action_weak_pass_absent"
}
```

判断：v9.5.3 的工程链路没有卡在 APG implementation 或 replay materializer；route 停在 canonical AP0 中 GoodGeomAction 支持度太低，且 APG/certificate 未能把 geometry diagnostic 转成 official controller。

## 3. P0 v9.5.2 boundary

Artifact：

```text
p0_v9520_boundary_reproduction.csv
```

Summary：

```text
route_v9520 = R1-CanonicalActionDensityInsufficientForMultiHorizonTarget
canonical_full_control_outcome_ready = 1
target_lattice_candidate_count = 8000
official_target_candidate_count = 0
weak_target_candidate_count = 0
selected_target_action_count = 27
selected_target_coverage = 0.009388038942976356
selected_target_V_integrated_lcb = 0.24423130340766275
selected_target_h240_longrisk = 0.0
selected_target_bad_event = 0.1111111111111111
selected_target_null_event = 0.25925925925925924
best_apy_primitive = APY3-LongRiskBarrierPrimitive
best_apy_target_precision = 0.046875
best_apy_V_integrated_lcb = -0.9770587887784598
best_apy_h240_longrisk = 0.71875
best_certificate_id = CERT30-BasisEdgeLocalityCertificate
best_certificate_AUC_target = 0.691913214990138
best_certificate_TopK64_precision = 0.015625
best_certificate_TopK64_longrisk = 0.671875
system_legal_controller_pass = 0
p0_pass = 1
```

判断：P0 pass。v9.5.3 没有跳过 v9.5.2 的 target density blocker。

## 4. P1 GeometryCard table

Artifacts：

```text
p1_geometry_card_table_v9530.csv
```

Summary：

```text
geometry_card_rows = 3388
canonical_ap0_card_rows = 2876
generated_apy_card_rows = 512
missing_required_field_count = 0
nan_count = 0
inf_count = 0
feature_cost_recorded = 1
commit_time_fields_separated = 1
geometry_card_pass = 1
```

判断：P1 pass。GeometryCard 的字段、hash 与 commit-time/diagnostic separation 均闭合；但该表本身只是 diagnostic substrate，不能直接成为 selector/controller。

## 5. P2 target-vs-geometry audit

Artifact：

```text
p2_target_vs_geometry_audit.csv
```

Summary：

```text
group_count = 9
best_geometry_group = T_C_StrictAllH
best_group_action_count = 16
best_group_snr_mean = 0.6534713038783501
best_group_cover_delta_mean = -0.07027188124691248
best_group_curv_delta_mean = 0.0704833897282479
best_group_forget_risk_mean = 0.05570613797463307
target_vs_geometry_pass = 1
```

Selected v9.5.2 target：

```text
action_count = 27
coverage = 0.009388038942976356
V_integrated_lcb = 0.24423130340766275
h20/h80/h240 V_lcb = 0.20138636510067395 / 0.2892031636318345 / 0.10183648386076839
h240_longrisk = 0.0
bad_event = 0.1111111111111111
null_event = 0.25925925925925924
snr_mean = 0.6435277183768728
cover_delta_mean = -0.05982371220437362
curv_delta_mean = 0.06000774807963926
forget_risk_mean = 0.07735228828777979
geometry_positive_pattern = 1
```

判断：P2 显示 strict/selected target 与 geometry quality 有一定关联，说明 v9.5.3 的方向有 diagnostic 进展；但该信号仍需通过 P3/P6/P10/P11 gates。

## 6. P3-P5 signal channel audits

P3 artifacts：

```text
p3_signal_channel_snr_gate_audit.csv
```

P3 summary：

```text
best_gate_id = Omega-batch-action
best_AUC_GoodGeomAction = 0.5108043788672061
best_TopK64_GGA_precision = 0.0
best_TopK64_h240_longrisk = 0.71875
snr_filtered_V_integrated_lcb = -1.3081510430858228
snr_gate_pass = 0
```

P4 artifacts：

```text
p4_cover_curvature_plasticity_audit.csv
```

P4 summary：

```text
best_group_id = GoodGeomAction_preliminary
best_cover_entropy_delta_mean = -0.016444980654001445
best_curv_proxy_delta_mean = 0.016546221809825328
best_hard_tail_fraction_delta_mean = 0.022539572842297975
cover_curvature_pass = 0
```

P5 artifacts：

```text
p5_memory_antiforgetting_audit.csv
```

P5 summary：

```text
best_group_id = GoodGeomAction_preliminary
best_forget_risk_score_mean = 0.052221369818122795
best_memory_margin_p10_delta_mean = -0.018277479436342975
best_plasticity_stability_ratio_mean = 3.861067261327215
old_family_fail_count = 0
memory_antiforgetting_pass = 1
```

判断：SNR 与 cover/curvature 不能独立筛出 usable source；anti-forgetting 侧相对健康，但只是一条必要条件，不足以打开 controller。

## 7. P6 GoodGeomAction label

Artifact：

```text
p6_good_geom_action_label.csv
```

Summary：

```text
GGA_count = 5
GGA_coverage = 0.0017385257301808068
GGA_coverage_lcb = 0.0007428048176512459
GGA_bad_event_rate = 0.0
GGA_null_event_rate = 0.0
GGA_h240_longrisk = 0.0
GGA_V_integrated_lcb = 0.1307816569689571
family_count = 4
stratum_count = 5
dataset_count = 2
max_family_share = 0.4
max_stratum_share = 0.2
support_balance_pass = 1
Jaccard_GGA_T9520 = 0.14285714285714285
Jaccard_GGA_T_A = 0.041666666666666664
Jaccard_GGA_T_B = 0.2
Jaccard_GGA_T_C = 0.23529411764705882
Jaccard_GGA_T_D = 0.078125
GGA_weak_pass = 0
GGA_official_pass = 0
```

判断：P6 是本轮 primary blocker。GGA 的 value/risk profile 有信号，但 canonical AP0 里只有 `5` 个 action，支持度远不足，不能 official。

## 8. P7 APY failure geometric autopsy

Artifact：

```text
p7_apy_failure_geometric_autopsy.csv
```

Summary：

```text
primitive_count = 8
apy_card_count = 512
assigned_failure_action_count = 512
assigned_failure_fraction = 1.0
dominant_failure_reason = longrisk_dominant
apy_failure_autopsy_pass = 1
```

Example primitive：

```text
APY1-ConstrainedMultiHorizonQP:
  target_precision = 0.015625
  V_integrated_lcb = -1.021495051023128
  h240_longrisk = 0.71875
  snr_mean = 0.7788687939469198
  cover_delta_mean = -0.0819886944422856
  curv_delta_mean = 0.08492478880916045
  forget_risk_mean = 0.2548688542577933
  assigned_failure_reason = longrisk_dominant
```

判断：P7 pass。APY 失败主要不是 replay/materializer，而是生成 action 仍带高 long-risk。

## 9. P8-P10 APG primitive

P8 artifacts：

```text
p8_apg_geometry_aware_generator.csv
```

P8 summary：

```text
primitive_count = 8
generated_action_count_expected = 512
generated_action_count_actual = 512
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_error_linf_max = 0.0
commit_time_geometry_fields_present = 1
uses_dataset_name = 0
uses_outcome_at_commit = 0
uses_future_step = 0
apg_implementation_pass = 1
```

P9 artifacts：

```text
p9_apg_preflight_branch_horizon_smoke.csv
apg_branch_horizon_outcome_trace_v9530.csv
```

P9 summary：

```text
stage_A/B/C/D pass = 1 / 1 / 1 / 1
action_count = 512
branch_count = 6
horizon_count = 3
branch_horizon_rows_expected/actual = 9216 / 9216
metric_abs_diff_no_transform = 0.0
label_match_no_transform = 1.0
negative_control_divergence_present = 1
unresolved_exception_count = 0
rows_per_sec = 20.609490015147806
wallclock_sec = 447.17263713106513
apg_preflight_branch_horizon_pass = 1
```

P10 artifacts：

```text
p10_apg_outcome_geometry_evaluation.csv
```

P10 summary：

```text
primitive_count = 8
generated_action_count = 512
best_primitive_id = APG1-SNRProjectedEdgeUpdate
best_GGA_precision = 0.03125
best_V_integrated_lcb = -0.4061862277341243
best_h240_longrisk = 0.6875
best_forget_risk = 0.1724040751051507
apg_weak_pass = 0
```

Best primitive：

```text
APG1-SNRProjectedEdgeUpdate:
  GGA_precision = 0.03125
  V_integrated_lcb = -0.4061862277341243
  h240_longrisk = 0.6875
  forget_risk = 0.1724040751051507
  snr_mean = 0.7476443293020221
  cover_delta_mean = 0.03434642703039938
  curv_delta_mean = -0.020820636322084772
```

判断：APG 工程链路和 branch-horizon replay 完整闭合，但 outcome 不过。不能把 APG implementation pass 写成 generator pass。

## 10. P11 geometry certificate

Artifact：

```text
p11_geometry_certificate_v1.csv
```

Summary：

```text
certificate_count = 8
base_rate_GGA = 0.002656434474616293
best_certificate_id = GCERT8-MinimalControllerCertificate
best_AUC_GGA = 0.9899049685968893
best_AUC_LongRisk = 0.9515504703814787
best_TopK64_GGA_precision = 0.125
best_TopK64_longrisk = 0.0
best_TopK64_forgetrisk = 0.046875
best_ECE_GGA = 0.6247721285956511
geometry_certificate_weak_pass = 0
```

判断：GCERT8 有很强 diagnostic ranking signal，但 GGA base count 太低且 calibration/gate 未达 weak pass；不能转 controller。

## 11. P12-P16 boundary

P12：

```text
p12_minimal_geometry_controller.csv = not_run
reason = canonical_ap0_good_geom_action_weak_pass_absent
source_controller_pass = 0
```

P13：

```text
p13_selected_runtime.csv = not_run
reason = P12_controller_not_selected
selected_runtime_pass = 0
```

P14：

```text
p14_base_acc_sentinel_continuation.csv
sentinel_row_count = 120
mean_test_acc_LQ = 0.6537760416666667
mean_test_acc_MLP = 0.562890625
mean_test_acc_QuadraticFeatureMLP = 0.40234375
mean_test_acc_AdamWStrongLRGridMLP = 0.628515625
LQ_catastrophic_fail = 0
base_acc_used_for_controller = 0
base_acc_sentinel_pass = 1
```

P15-P16：

| artifact | status / reason |
|---|---|
| `p15_leaveout_paired_replay_boundary.csv` | `not_run`, `P12_system_controller_not_official` |
| `p16_short_full_continual_boundary.csv` | `not_run`, `P15_paired_replay_not_open` |

判断：没有把 GeometryCard、GGA diagnostic、APG preflight、GCERT diagnostic 或 Base-Acc Sentinel 写成 official controller/runtime/downstream pass。

## 12. No-fake audit

```text
rows_checked = 22530
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
v9520_boundary_pass = 1
geometry_card_pass = 1
target_vs_geometry_pass = 1
snr_gate_pass = 0
cover_curvature_pass = 0
memory_antiforgetting_pass = 1
GGA_weak_pass = 0
apy_failure_autopsy_pass = 1
apg_implementation_pass = 1
apg_branch_horizon_pass = 1
apg_weak_pass = 0
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
route = R2-NoGoodGeometryActionInCanonicalAP0
F0_reproduction_fail = 0
F1_geometry_card_fail = 0
F2_no_good_geometry_action = 1
F3_signal_gate_fail = 0
F4_memory_antiforgetting_fail = 0
F5_apg_implementation_fail = 0
F6_apg_value_fail = 0
F7_certificate_fail = 0
F8_controller_runtime_blocked = 1
F9_base_acc_catastrophic = 0
primary_blocker = canonical_ap0_good_geom_action_weak_pass_absent
```

## 13. Hash

| artifact | SHA256 |
|---|---|
| plan | `4fb9080f9dc78441494f958b0a3bdf5ae9c05f2a1366eb2efc0a8ab044a2f17b` |
| runner | `27b01fae9fad52e365713406cd27a1b804de6445e3eccf1d83189d808297a6c7` |
| run manifest | `429df9db99fdea51b0dfa89f70b4b2820d9b5d132ec8e9bcf5ef473016c25c55` |
| route | `de9539d51f90b976a01ed5f84bb2df6950701c0bcf573bb655d4055e1437d3a1` |
| P0 boundary | `d11db222097598907a966b79c4cf01dfe88bbd0a0238da387d116fe1f4505149` |
| P1 GeometryCard | `b2ad36949041d37590d74ad51761b329781cc8c87d6a38e27f42c54cf9f0d8ac` |
| P2 target geometry | `68fb5b320c4ddd21f090899adaa791abd869fd817cd0905780a990f6177b5e51` |
| P3 SNR gate | `c1c4f5d44d323b352db4fa7cedfbd7ade54d3e23da9b2f2758646325ee95fd21` |
| P4 cover curvature | `35b543763131b017c378e15bda7a47d78c0506b087805567bc36cfd17ebef69d` |
| P5 memory | `69394253a99db4eb36b4ac7aba60e7a49699a35fd46e28711a34ece373d6e9fb` |
| P6 GGA | `bb87b0b924aadb921f46cad64ffd78a00fbbb1bd689f03761822309cdb5c7553` |
| P7 APY autopsy | `7bf48e3c9aaa4cd9271784825c4a343a74d7ee549031c3340570d3e42c982e1a` |
| P8 APG generator | `3d748437322884a2a595254f4a292693bb4ebb9f51f8f6605e641e6eb08d273b` |
| P9 APG smoke | `74114ffabda5fddf50a85e546776828d6755fabfb6e18788c58538cfb5b32b94` |
| APG trace | `393ac74d18ebf7e7731467831ac0232211a2b976f4c0c41c702e4daeca6ad18b` |
| P10 APG outcome | `68525471da1ff71859a10205b13ff4fbd12c444f85f044efb784353c3fc5df52` |
| P11 certificate | `913729b29e3db2e0d08c0fa1228c3aa169eab04c9297a59bf9ed02370f1a3851` |
| P12 controller | `e93ba0eef83722d2ed18bc823f7847f28070e35b09640e00d2f4bf091e7b158e` |
| P13 runtime | `33bf0d49a380c3ddb2b6fd887f82bb2f1ee94f64e660bb31a3c8429a591ca3c9` |
| P14 Base-Acc Sentinel | `f7e47de95bfa5ee18fb1427bb71428adf313b6b0e68870ecb11c6dac446719a1` |
| P15 paired boundary | `19297b902d6df9fee8c34d6e862968c1f426d82f5e2bae47a1fab6fb35d2863c` |
| P16 short/full boundary | `9bf59ac3d1a4e212cf45388efdfc753167f862151409f397daa0f586950dbbbc` |
| no-fake audit | `4a90122660b57ff76df7d785c8d82f3d1bf95c69c072011868e8936e2a878ae5` |
| contract audit | `05325b73ae483fb2ac72b9203929dadcc9b7b6eab40194813ca4dc9bf346c2e9` |
| failure table | `c823c1050ac5507f997599ffd89822a5716b892f4aff4bab969c72864faa9866` |

## 14. 最终分析结论

v9.5.3 的真实推进是：

```text
v9.5.2:
  target lattice 找到 value/risk 更平衡但过稀疏的 diagnostic target；
  APY1-APY8 工程链路完整闭合；
  但没有任何 multi-horizon target 满足 weak pass。

v9.5.3:
  将目标从 outcome-only target lattice 推进到 GeometryCard / signal channel / anti-forgetting audit；
  发现 strict/selected target 与 geometry quality 有一定 diagnostic 关联；
  定位 APY 失败主要由 high long-risk 主导；
  APG1-APG8 geometry-aware primitive 的 implementation 与 branch-horizon materializer 均闭合；
  但 canonical AP0 中 GoodGeomAction 只有 5 个，APG 和 GCERT 也未转成 official controller。
```

机制判断：

1. H0 成立：v9.5.2 boundary 被复现，canonical truth base 仍可用。
2. H1 成立于 materialization 层：GeometryCard table 完整落盘，字段与 no-fake contract 通过。
3. H2 部分成立：strict/selected targets 与 geometry quality 有 diagnostic pattern，但不能直接 official。
4. H3 未成立：SNR gate 无法筛出 GoodGeomAction，TopK64 GGA precision = `0.0`。
5. H4 未成立：cover/curvature/plasticity 没有形成独立稳定 gate。
6. H5 成立于 diagnostic 层：GoodGeomAction preliminary anti-forgetting 侧健康，但 GGA official 支持度不足。
7. H6 成立：APY failure autopsy 指向 `longrisk_dominant`。
8. H7/H8 成立于 implementation 层：APG1-APG8 payload/certificate/action apply/preflight/branch-horizon replay 均闭合。
9. H9 未成立：APG outcome 没有形成 value-positive / horizon-safe / GGA-positive primitive。
10. H10 未成立：geometry certificate ranking 很强，但 weak pass 未过，不能 controller。
11. H11-H13 未打开：没有 source controller，就不能打开 selected runtime、official paired replay、short/full 或 continual validation。
12. Base-Acc Sentinel 继续健康，但没有用于 controller，也不是 functional success。

最终一句话：

> v9.5.3 真实执行后停在 `R2-NoGoodGeometryActionInCanonicalAP0`：GeometryCard 和 APG 工程链路已经闭合，并且找到了 geometry/anti-forgetting 的 diagnostic 信号；但 canonical AP0 里的 GoodGeomAction 只有 `5` 个，SNR/cover/curvature/APG/GCERT 都没有形成可 official 的 selector/controller，strict PureKAN functional 仍未成功。
