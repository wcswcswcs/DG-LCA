# DG-KAN v9.4.2 Source Outcome Materializer Closure / Value-Producing Source Triage 实验复盘

> 本复盘记录 `DG-KAN_v9.4.2_SourceOutcomeMaterializerClosure_ValueProducingSourceTriage_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 materializer closure、Base-Acc Sentinel、source outcome diagnostic 或未选中 controller/runtime 写成 official system pass。

## 0. 最新结论

```text
route = R2-SourcePanelValuePoorDespiteStratification
base_candidate = LQ-t2-h256
success_v9420_strict_purekan_functional = False
success_v9420_full_functional = False
success_v9420_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9420_source_outcome_materializer_closure_value_triage_first_20260514T090000Z/
```

核心结论：

1. P0 复现 v9.4.1 boundary：source route = `R4-GeneratedSourceImmediateDirectionFail`，v9.4.1 expected source outcome rows = `7020`，actual = `0`。
2. P1 source outcome materializer preflight 过：P1a/P1b/P1c 均 pass，unresolved exception = `0`。
3. P2 source outcome materializer closure 成功：generated action = `260`，branch = `9`，horizon = `3`，expected/actual rows = `7020 / 7020`。
4. P2 quality audit 过：branch/horizon/secondary missing = `0 / 0 / 0`，NaN/Inf/label exclusivity/duplicate row id violation 均为 `0`。
5. P2 throughput = `23.6331087573844 rows/sec`，wallclock = `297.04090443905443 sec`。
6. P3 same-panel source AP0 baseline 不好：h20 weak CP = `0.11153846153846154`，h20 V_ctrl LCB = `-1.29393883306781`，horizon-robust coverage = `0.15384615384615385`，h240 long-risk = `0.8038461538461539`。
7. P4 generated source h20 immediate direction 未过：best primitive = `AP0d-TailMarginRepairSource`，h20 weak CP = `0.17307692307692307`，bad-event = `0.36538461538461536`，V_ctrl LCB = `-1.8014724919297649`。
8. P5 source-to-generated damage 明显：paired horizon = `780`，Damage median = `-0.1004650485701859`，source-positive lost rate = `0.9215686274509803`。
9. P6 horizon extension 未过：selected primitive = `AP0d-TailMarginRepairSource`，h80 weak CP = `0.19230769230769232`，h240 weak CP = `0.11538461538461539`，h240 long-risk = `0.8653846153846154`。
10. P7 certificate sufficiency 未过：AUC weak CP = `0.47741176470588237`，Lift weak = `0.9928430846305243`，Lift long-risk = `0.9099491648511256`。
11. P8 triage route = `Case A: source AP0 bad, generated bad`；stop threshold tuning flag = `1`。
12. P9 Base-Acc Sentinel 仍 complete 且未用于 controller：mean test acc LQ = `0.6553819444444444`，MatchedMLP = `0.5559895833333334`，catastrophic fail = `0`。
13. P10-P14 全部 gate-blocked：没有 source controller、selected runtime、official paired replay、short/full/continual success。
14. 当前 primary blocker：`source_panel_value_poor_despite_stratification`；下一步不能继续只修 row sink，必须重建 source selector 或 source panel input。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9420_source_outcome_materializer_closure.py` | v9.4.2 runner；读取 v9.4.1/v9.3.3 artifacts，执行 source outcome materializer preflight、260-action full branch/horizon replay、AP0 baseline、generated source triage、certificate lift、Base-Acc Sentinel 与 system gate |

代码检查：

```text
python -m py_compile experiments/run_v9420_source_outcome_materializer_closure.py
```

正式运行：

```bash
python experiments/run_v9420_source_outcome_materializer_closure.py \
  --out-dir results/real_rerun_20260506/v9420_source_outcome_materializer_closure_value_triage_first_20260514T090000Z \
  --fresh --device auto --data-root data --seed 1314 --seeds 0 --sentinel-seeds 0,1,2
```

运行结果：

```json
{
  "h20_pass": 0,
  "out_dir": "results/real_rerun_20260506/v9420_source_outcome_materializer_closure_value_triage_first_20260514T090000Z",
  "route": "R2-SourcePanelValuePoorDespiteStratification",
  "source_rows": 7020
}
```

## 2. Route

`route_decision.json` 摘要：

```json
{
  "route": "R2-SourcePanelValuePoorDespiteStratification",
  "source_route_v9410": "R4-GeneratedSourceImmediateDirectionFail",
  "p1_preflight_pass": 1,
  "source_outcome_materialized": 1,
  "source_outcome_materializer_pass": 1,
  "branch_horizon_row_count_expected": 7020,
  "branch_horizon_row_count_actual": 7020,
  "rows_per_sec_total": 23.6331087573844,
  "same_panel_source_ap0_baseline_bad": 1,
  "AP0_weak_CP_precision_h20": 0.11153846153846154,
  "AP0_V_ctrl_lcb_h20": -1.29393883306781,
  "best_h20_primitive_id": "AP0d-TailMarginRepairSource",
  "best_weak_CP_precision_h20": 0.17307692307692307,
  "best_V_ctrl_lcb_h20": -1.8014724919297649,
  "generator_damage_fail": 1,
  "Damage_median": -0.1004650485701859,
  "source_positive_lost_after_generation_rate": 0.9215686274509803,
  "h20_immediate_direction_pass": 0,
  "horizon_extension_pass": 0,
  "long_risk_rate_h240": 0.8653846153846154,
  "certificate_sufficiency_pass": 0,
  "AUC_certificate_weak_CP": 0.47741176470588237,
  "base_acc_sentinel_complete": 1,
  "base_acc_used_for_controller": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "source_panel_value_poor_despite_stratification"
}
```

判断：v9.4.2 真实解决了 v9.4.1 的 `0-row source outcome materializer` blocker；新的结论是 source panel / generated source action 在真实 branch-horizon outcomes 下不具备 value-positive frontier。

## 3. P1/P2 source outcome materializer

Artifacts：

```text
p1_source_outcome_materializer_preflight.csv
source_outcome_materializer_failure_trace_v9420.csv
source_outcome_join_key_trace_v9420.csv
source_outcome_row_sink_trace_v9420.csv
p2_source_outcome_materializer_scaleup.csv
source_outcome_trace_v9420.csv
branch_horizon_completion_trace_v9420.csv
```

P1 summary：

```text
p1a_pass = 1
p1b_pass = 1
p1c_pass = 1
unresolved_exception_count = 0
preflight_pass = 1
```

P2 summary：

```text
materializer_id = SOM2-DirectGeneratedActionReplayClosure
generated_action_count_input = 260
branch_count = 9
horizon_count = 3
branch_horizon_row_count_expected/actual = 7020 / 7020
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
secondary_delta_completion_rate = 1
rows_per_sec_total = 23.6331087573844
wallclock_sec = 297.04090443905443
row_count_failed/retried/unresolved = 0 / 0 / 0
quality_audit_pass = 1
missing_branch/horizon/secondary = 0 / 0 / 0
metric_nan/inf = 0 / 0
label_exclusivity_violation = 0
duplicate_outcome_row_id_count = 0
```

判断：P2 pass。v9.4.1 的 source outcome rows absent 问题被真实闭合，且不是 fake/proxy rows。

## 4. P3 same-panel source AP0 baseline

Artifacts：

```text
p3_same_panel_source_ap0_baseline.csv
source_ap0_outcome_trace_v9420.csv
```

Summary：

```text
source_action_count = 52
branch_horizon_row_count_expected/actual = 1404 / 780
AP0_weak_CP_precision_h20 = 0.11153846153846154
AP0_strong_CP_precision_h20 = 0.08076923076923077
AP0_weak_CP_precision_h80 = 0.13076923076923078
AP0_weak_CP_precision_h240 = 0.15
AP0_long_risk_rate_h240 = 0.8038461538461539
AP0_V_ctrl_lcb_h20 = -1.29393883306781
AP0_V_ctrl_lcb_all = -1.2099435679819464
AP0_horizon_robust_action_count = 8
AP0_horizon_robust_coverage = 0.15384615384615385
source_ap0_baseline_useful = 0
source_ap0_baseline_bad = 1
```

判断：P3 是本轮 route 的关键。即使 source panel 已 stratified，same-panel AP0 baseline 仍 value-poor / long-risk high，因此不能把 failure 只归因于 generated-source transform。

## 5. P4 generated source h20 immediate direction

Artifact：

```text
p4_generated_source_immediate_direction.csv
```

Per primitive h20：

| primitive | weak CP | strong CP | bad-event | V_ctrl LCB | pass |
|---|---:|---:|---:|---:|---:|
| `AP0b-LastEdgeLinearizedDescentSource` | `0.096154` | `0.057692` | `0.288462` | `-1.505584` | `0` |
| `AP0c-AdamWResidualOrthogonalSource` | `0.096154` | `0.096154` | `0.326923` | `-1.499603` | `0` |
| `AP0d-TailMarginRepairSource` | `0.173077` | `0.134615` | `0.365385` | `-1.801472` | `0` |
| `AP0e-CurvatureGuardedLowRankEdgeSource` | `0.019231` | `0.019231` | `0.384615` | `-2.044336` | `0` |
| `AP0f-SupportMemorySource` | `0.115385` | `0.076923` | `0.365385` | `-1.603251` | `0` |

Best primitive：

```text
best_primitive_id = AP0d-TailMarginRepairSource
best_weak_CP_precision_h20 = 0.17307692307692307
best_V_ctrl_lcb_h20 = -1.8014724919297649
best_bad_event_rate_h20 = 0.36538461538461536
h20_immediate_direction_pass = 0
```

判断：P4 未过。generated source actions 已有真实 h20 rows，但没有形成 immediate value-positive frontier。

## 6. P5/P6 damage 与 horizon extension

P5 artifacts：

```text
p5_source_to_generated_damage_matrix.csv
```

P5 summary：

```text
paired_horizon_count = 780
Damage_mean = -0.09249623563492862
Damage_median = -0.1004650485701859
source_positive_horizon_count = 102
source_positive_lost_after_generation_rate = 0.9215686274509803
source_negative_fixed_after_generation_rate = 0.13569321533923304
generator_value_preserving_pass = 0
generator_damage_fail = 1
```

P6 artifacts：

```text
p6_horizon_extension_longrisk_audit.csv
```

P6 summary：

```text
selected_primitive_id = AP0d-TailMarginRepairSource
weak_CP_precision_h80 = 0.19230769230769232
weak_CP_precision_h240 = 0.11538461538461539
long_risk_rate_h240 = 0.8653846153846154
horizon_robust_action_coverage = 0.038461538461538464
horizon_extension_pass = 0
```

判断：generated transform 有明显 damage，long horizon 风险也高；但更早的 P3 已经显示 source AP0 baseline 本身不够好。

## 7. P7 certificate sufficiency

Artifacts：

```text
p7_certificate_sufficiency_lift_audit.csv
certificate_component_ablation_v9420.csv
```

Summary：

```text
certificate_id = v9410-source-cert-v1
joined_outcome_count = 780
certificate_pass_action_count = 243
AUC_certificate_weak_CP = 0.47741176470588237
AUC_certificate_strong_CP = 0.45378723404255317
AUC_certificate_longrisk = 0.4895340819542947
P_weak_CP_given_cert_pass = 0.12757201646090535
P_weak_CP_given_cert_fail = 0.12849162011173185
P_longrisk_given_cert_pass = 0.25925925925925924
P_longrisk_given_cert_fail = 0.2849162011173184
Lift_weak = 0.9928430846305243
Lift_longrisk = 0.9099491648511256
monotone_sign_pass = 0
certificate_sufficiency_pass = 0
```

判断：certificate 对 weak CP 没有正 lift，AUC 低于随机附近；不能用 certificate threshold patch 进入 controller。

## 8. P8-P14 boundary

P8：

```text
triage_case = Case A: source AP0 bad, generated bad
route = R2-SourcePanelValuePoorDespiteStratification
source_AP0_status = bad
generated_source_status = h20_fail
generator_damage_status = damage_fail
certificate_status = fail_or_not_open
next_required_implementation = rebuild_source_selector_or_source_panel_input
stop_threshold_tuning_flag = 1
```

P10-P14 均被 gate-block：

| artifact | status / reason |
|---|---|
| `p10_minimal_source_certificate_controller.csv` | `not_run`, upstream source/certificate gate failed |
| `p11_selected_source_online_runtime.csv` | not selected |
| `p12_system_integration_gate_v9420.csv` | official eligible = `0` |
| `p13_leaveout_paired_replay_boundary_v9420.csv` | P12 system controller not official |
| `p14_short_full_training_mlp_comparison_boundary_v9420.csv` | official controller not open |

没有把 source outcome materializer closure、Base-Acc Sentinel 或 triage diagnostic 写成 official controller/runtime/downstream pass。

## 9. P9 Base-Acc Sentinel

Artifacts：

```text
p9_base_acc_sentinel_extended.csv
base_acc_training_trace_v9420.csv
```

Summary：

```text
sentinel_row_count = 27
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
model_count = 3
sentinel_complete = 1
mean_test_acc_LQ = 0.6553819444444444
mean_test_acc_MLP = 0.5559895833333334
LQ_minus_MLP_mean_test_acc = 0.09939236111111105
LQ_catastrophic_fail = 0
base_acc_used_for_controller = 0
```

判断：Base-Acc Sentinel 仍健康，但它只是固定配置 sentinel，没有用于 selector/controller。

## 10. No-fake audit

```text
rows_checked = 7033
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
p1_preflight_pass = 1
source_outcome_materialized = 1
h20_immediate_direction_pass = 0
horizon_extension_pass = 0
certificate_sufficiency_pass = 0
base_acc_sentinel_complete = 1
base_acc_used_for_controller = 0
source_controller_pass = 0
selected_runtime_pass = 0
system_legal_controller_pass = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_selector/controller = 0/0
uses_validation_or_test_for_controller = 0
uses_future_outcome_for_features = 0
uses_outcome_at_commit = 0
source_measured_gap/formula_proxy = 0/0
diagnostic_promoted_to_official = 0
fake/proxy/cpu_offload = 0/0/0
```

Failure table：

```text
F1_materializer_path_fail = 0
F2_source_panel_value_poor = 1
F3_generated_source_immediate_fail = 1
F4_generator_transform_damage = 1
F5_horizon_longrisk_fail = 0
F6_certificate_not_effect_valid = 0
F7_controller_not_selected = 1
F8_runtime_not_selected = 1
F9_base_acc_sentinel_catastrophic = 0
F10_downstream_not_open = 1
primary_blocker = source_panel_value_poor_despite_stratification
```

## 11. Hash

| artifact | SHA256 |
|---|---|
| plan | `92c55e4cae110bbe71a1d008113c43eedfd91ca9c900606c894635645e740a87` |
| runner | `e73af98e16de623c62a978124048bab833d9eeb7deea49d7b70300fb2aee75a6` |
| run manifest | `c2971b9d9c8af8fd819bb07cfcad38053a0d52352daec1ad1117f24cf953c5d5` |
| route | `e899a57213edba45efde1ea22400d3732e21801a3d1de7c2782a4527f513ae6a` |
| P0 boundary | `41d32760d5be24fc6679a0281792dd4c17d2b09add29df719948c402467f7fbe` |
| P1 preflight | `5c45864a32b56fc59d7f2082821fa8fb4f48d7ecd88be0bfab015ebe0bd2e23c` |
| P2 materializer | `44203767fc9514000b407ec6550b1213f6d4ceca1346e427216f73370f9c6d69` |
| source outcome trace | `132f3d22326c8a7bc3f4ac87ffd9b09245a36dce431fa98dd886dfe8ec0133cc` |
| P3 source AP0 | `9eac12abb089624109b5bbbe0365634469dd36da0ecca5269c7759215db5f85c` |
| P4 generated source | `d14b64accf277715d0997f7c96ffc5e0508ce8a5b8334970a701bd24b89f44b5` |
| P5 damage | `3966b384b36730065cef4995f76ba36541c3c4733b119f93067769ae7b5dcd31` |
| P6 horizon | `ca76727c4566d059280045b7e63a740470a792d4737fe289498bf548516bbbcf` |
| P7 certificate | `e6ada7eb25f35e26c5b3e7cc0502a44f6651fd3e2c3c53e4470047b9e5393693` |
| P8 triage | `eedeb712753fd1725ce6a494c67c32b2f86d3fc6bb6031b0a3795635f73d4a7a` |
| P9 Base-Acc Sentinel | `987b451d7547148c3391092905ff40993767f1629852c1e3788aff70113c9a29` |
| P12 system | `5c8a33f7e1514ebe44040bb716640ae3e2fee6aef764bff56ae14197b3e68405` |
| contract audit | `516510cf0b5c2c3d7bda1f1a05847059f6cde5fb4053e478a18f7e029b79690f` |
| provenance audit | `fc8749f679f15387382a231a2513840c0856c7973cea4457c4caadcb42b21cda` |
| failure table | `6c143af916eb26039dbe181c90beec61ee8ceccb25bc9e026300f437640326f8` |

## 12. 最终分析结论

v9.4.2 的真实推进是：

```text
v9.4.1:
  stratified source panel、Base-Acc Sentinel、AP0b-AP0f source payload/certificate 已落盘；
  但 source outcome materializer 产出 0 行，因此无法判断 generated source value。

v9.4.2:
  修通 source outcome materializer preflight 和 full 260-action branch/horizon replay；
  7020 行 source outcome 完整落盘并通过 quality audit；
  但 same-panel AP0 baseline 和 generated AP0b-AP0f source actions 都没有形成 value-positive frontier。
```

机制判断：

1. H1 成立：v9.4.1 的 `0-row` blocker 被真实解决，P2 expected/actual = `7020 / 7020`。
2. H2 未成立：stratified source panel 仍 value-poor，AP0 h20 weak CP 只有 `0.1115`，V_ctrl LCB 为负。
3. H3 未成立：generated source h20 immediate direction 没过，best AP0d weak CP 只有 `0.1731`，bad-event `0.3654`。
4. H4 部分成立：generator 有 damage，source-positive lost rate = `0.9216`；但 P3 显示 source panel 本身已经不好。
5. H5 未成立：horizon extension 不过，h240 long-risk = `0.8654`。
6. H6 未成立：certificate 没有 positive lift，AUC weak CP = `0.4774`。
7. H7 未打开：controller/runtime/system/downstream 全部 gate-blocked。

最终一句话：

> v9.4.2 真实执行后停在 `R2-SourcePanelValuePoorDespiteStratification`：source outcome materializer 已经从 0 行修到完整 7020 行，这是本轮真实推进；但实测后发现 stratified source panel 本身 value-poor，generated source primitives 与 certificate 也不能恢复 frontier，strict PureKAN functional 仍不能转正。
