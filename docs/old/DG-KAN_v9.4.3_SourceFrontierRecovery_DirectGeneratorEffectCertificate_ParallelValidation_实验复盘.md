# DG-KAN v9.4.3 Source Frontier Recovery / Direct Generator Effect Certificate / Parallel Validation 实验复盘

> 本复盘记录 `DG-KAN_v9.4.3_SourceFrontierRecovery_DirectGeneratorEffectCertificate_ParallelValidation_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 oracle panel、direct generator smoke、effect-certificate diagnostic、Base-Acc Sentinel 或未选中 runtime 写成 official system pass。

## 0. 最新结论

```text
route = R2-FullAP0OracleSourceFrontierAbsent
base_candidate = LQ-t2-h256
success_v9430_strict_purekan_functional = False
success_v9430_full_functional = False
success_v9430_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9430_source_frontier_recovery_direct_generator_first_20260514T100000Z/
```

核心结论：

1. P0 复现 v9.4.2 boundary：source route = `R2-SourcePanelValuePoorDespiteStratification`，source outcome materializer 仍 closed，`7020 / 7020` 行没有回退。
2. P1 multi-panel source frontier recovery 没找到 official-positive AP0 source frontier：best oracle panel = `PANEL-ORC64`，h20 weak CP = `0.578125 < 0.60`，h20 V_ctrl LCB = `-0.020407727203802614 <= 0`，h240 long-risk = `0.0`。
3. P1 不是完全无信号：`PANEL-ORC128` h20 weak CP = `0.6875`，但 h20 V_ctrl LCB = `-0.041945819477686164` 且 h240 long-risk = `0.2109375`，仍不能 official。
4. 当前 legal source selector 仍失败：best legal panel = `PANEL-LGL64-AdamWConflictLow`，h20 weak CP = `0.171875`，h20 V_ctrl LCB = `-0.3014420568912851`，h240 long-risk = `0.890625`。
5. P2 same-panel AP0 baseline 完整：`PANEL-ORC64` expected/actual rows = `1152 / 1152`，branch/horizon completion = `1.0 / 1.0`，但 source survivor pass = `0`。
6. P3 复用 v9.4.2 measured AP0b-AP0f damage：paired horizon = `780`，Damage median = `-0.1004650485701859`，source-positive lost rate = `0.9215686274509803`。
7. P4 AP0g-AP0k direct generator 已真实 materialize：generated action = `60`，branch-horizon expected/actual rows = `1620 / 1620`，source outcome materialized = `1`。
8. P4 direct generator 仍不过：best primitive = `AP0j-LowRankEdgeSafeSource`，h20 weak CP = `0.25`，h20 V_ctrl LCB = `-0.20831185402129682`，h240 long-risk = `0.8333333333333334`。
9. P5 effect certificate 未过：AUC weak CP = `0.5069044879171462`，AUC strong CP = `0.6217142857142857`，AUC long-risk = `0.5581680996666082`，monotone sign pass = `0`。
10. P8 Base-Acc Sentinel 本轮实测完成：datasets = `MNIST,Fashion-MNIST,KMNIST`，seeds = `0,1,2,3,4`，model count = `3`，row count = `45`，sentinel complete = `1`。
11. Base-Acc Sentinel 只作为隔离健康监控，没有用于 selector/controller：mean test acc LQ = `0.64921875`，MatchedMLP = `0.5609375`，LQ minus MLP = `0.08828125`，catastrophic fail = `0`。
12. 计划中的 `AdamWStrongLRGridMLP` diagnostic 本 runner 未 materialize，artifact 明确记录 `AdamWStrongLRGridMLP_diagnostic_materialized = 0`，没有把它写成 pass。
13. P9 system controller 未打开：official eligible = `0`，system legal controller pass = `0`，reason = `full_ap0_oracle_source_frontier_absent`。
14. 当前 primary blocker：`full_ap0_oracle_source_frontier_absent`。更精确地说，AP0 oracle panel 有接近 gate 的信号，但在同一 protocol 下 h20 V_ctrl LCB 仍为负；legal selector 和 direct generator 也都没有给出可 official 的 h20-positive / h240-safe frontier。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9430_source_frontier_recovery_direct_generator.py` | v9.4.3 runner；读取 v9.4.2/v9.4.1/v9.4.0/v9.3.5/v9.3.3 artifacts，执行 multi-panel AP0 source frontier、same-panel AP0 baseline、AP0b-AP0f damage carry-forward、AP0g-AP0k direct generator smoke、effect certificate audit、Base-Acc Sentinel 与 system gate |

代码检查：

```text
python -m py_compile experiments/run_v9430_source_frontier_recovery_direct_generator.py
```

正式运行：

```bash
python experiments/run_v9430_source_frontier_recovery_direct_generator.py \
  --out-dir results/real_rerun_20260506/v9430_source_frontier_recovery_direct_generator_first_20260514T100000Z \
  --fresh --device auto --data-root data --seed 1314 --direct-actions-per-generator 12
```

运行结果：

```json
{
  "direct_rows": 1620,
  "oracle_h20": 0.578125,
  "out_dir": "results/real_rerun_20260506/v9430_source_frontier_recovery_direct_generator_first_20260514T100000Z",
  "route": "R2-FullAP0OracleSourceFrontierAbsent"
}
```

实际输入来自 `run_manifest.json`：

```text
source_v9420 = results/real_rerun_20260506/v9420_source_outcome_materializer_closure_value_triage_first_20260514T090000Z
source_v9410 = results/real_rerun_20260506/v9410_value_producing_source_generator_legal_source_identifiability_base_acc_first_20260514T080000Z
source_v9400 = results/real_rerun_20260506/v9400_source_action_selection_candidate_source_rebuild_horizon_controller_first_20260514T070000Z
source_v9350 = results/real_rerun_20260506/v9350_control_positive_frontier_completion_legal_probe_runtime_first_20260514T023000Z
source_v9330 = results/real_rerun_20260506/v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z
direct_actions_per_generator = 12
direct_generated_action_count = 60
device = cuda
```

## 2. Route

`route_decision.json` 摘要：

```json
{
  "route": "R2-FullAP0OracleSourceFrontierAbsent",
  "source_route_v9420": "R2-SourcePanelValuePoorDespiteStratification",
  "p0_pass": 1,
  "p1_diagnostic_oracle_pass": 0,
  "oracle_weak_CP_precision_h20": 0.578125,
  "oracle_V_ctrl_lcb_h20": -0.020407727203802614,
  "oracle_long_risk_rate_h240": 0.0,
  "p1_legal_selector_weak_pass": 0,
  "best_legal_panel_id": "PANEL-LGL64-AdamWConflictLow",
  "best_legal_weak_CP_precision_h20": 0.171875,
  "best_legal_V_ctrl_lcb_h20": -0.3014420568912851,
  "p3_damage_median": "-0.1004650485701859",
  "p3_source_positive_lost_after_generation_rate": "0.9215686274509803",
  "direct_generated_action_count": 60,
  "direct_branch_horizon_row_count_actual": 1620,
  "direct_generator_weak_pass": 0,
  "best_direct_primitive_id": "AP0j-LowRankEdgeSafeSource",
  "best_direct_h20_weak_CP_precision": 0.25,
  "best_direct_h20_V_ctrl_lcb": -0.20831185402129682,
  "best_direct_h240_long_risk_rate": 0.8333333333333334,
  "certificate_effect_valid_pass": 0,
  "AUC_certificate_weak_CP": 0.5069044879171462,
  "base_acc_sentinel_complete": 1,
  "base_acc_used_for_controller": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "full_ap0_oracle_source_frontier_absent"
}
```

判断：v9.4.3 没有回退 v9.4.2 materializer closure，并且 direct generator 确实跑出了 full smoke rows；但 AP0 oracle panel、legal selector 和 direct generator 都没有达到 official source frontier gate。

## 3. P0 v9.4.2 boundary reanalysis

Artifact：

```text
p0_v9420_boundary_reanalysis.csv
```

Summary：

```text
route_v9420 = R2-SourcePanelValuePoorDespiteStratification
source_outcome_materialized = 1
branch_horizon_row_count_expected/actual = 7020 / 7020
quality_audit_pass = 1
same_panel_comparison_valid = 1
p0_pass = 1
```

判断：P0 pass。v9.4.2 的 source outcome materializer closure 没有回退。

## 4. P1 multi-panel source frontier

Artifacts：

```text
p1_multi_panel_source_frontier.csv
source_panel_trace_v9430.csv
source_selector_feature_trace_v9430.csv
```

Summary：

```text
oracle_panel_id = PANEL-ORC64
oracle_weak_CP_precision_h20 = 0.578125
oracle_V_ctrl_lcb_h20 = -0.020407727203802614
oracle_long_risk_rate_h240 = 0.0
p1_diagnostic_oracle_pass = 0

best_legal_panel_id = PANEL-LGL64-AdamWConflictLow
best_legal_weak_CP_precision_h20 = 0.171875
best_legal_V_ctrl_lcb_h20 = -0.3014420568912851
best_legal_long_risk_rate_h240 = 0.890625
p1_legal_selector_weak_pass = 0
p1_legal_selector_strong_pass = 0
```

Representative / legal / oracle panel examples：

| panel | h20 weak CP | h20 V_ctrl LCB | h240 weak CP | h240 long-risk |
|---|---:|---:|---:|---:|
| `PANEL-RND64` | `0.062500` | `-1.058158` | `0.109375` | `0.890625` |
| `PANEL-REP256` | `0.085938` | `-0.752893` | `0.078125` | `0.921875` |
| `PANEL-ORC64` | `0.578125` | `-0.020408` | `1.000000` | `0.000000` |
| `PANEL-ORC128` | `0.687500` | `-0.041946` | `0.789063` | `0.210938` |
| `PANEL-LGL64-GradientAlignment` | `0.140625` | `-0.378674` | `0.171875` | `0.828125` |
| `PANEL-LGL64-AdamWConflictLow` | `0.171875` | `-0.301442` | `0.109375` | `0.890625` |

判断：P1 没有 official survivor。Oracle panel 接近但没过计划 gate，legal panel 明显更弱。不能把 oracle diagnostic 写成 controller，也不能把 legal feature topK 写成 source selector pass。

## 5. P2 same-panel AP0 baseline

Artifacts：

```text
p2_same_panel_ap0_baseline_full.csv
ap0_panel_baseline_trace_v9430.csv
```

Best panel summary：

```text
best_panel_id = PANEL-ORC64
source_action_count = 64
expected_rows = 1152
actual_rows = 1152
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
weak_CP_precision_h20 = 0.578125
weak_CP_precision_h80 = 0.625
weak_CP_precision_h240 = 1.0
V_ctrl_lcb_h20 = -0.020407727203802614
V_ctrl_lcb_all = -0.2094056033585386
bad_event_h20 = 0.0
null_rate_h20 = 0.0
long_risk_rate_h240 = 0.0
horizon_robust_coverage = 0.203125
source_survivor = 0
```

判断：P2 confirms P1。即使用 outcome oracle 挑出的 `PANEL-ORC64`，h20 V_ctrl LCB 仍略为负，因此没有 source survivor pass。

## 6. P3 AP0b-AP0f generator damage

Artifacts：

```text
p3_ap0b_ap0f_generator_damage_across_panels.csv
generator_damage_trace_v9430.csv
```

Summary：

```text
measured_panel_id = PANEL-S256-v9420
orc64_generated_outcomes_materialized = 0
legal_generated_outcomes_materialized = 0
paired_horizon_count = 780
Damage_mean = -0.09249623563492862
Damage_median = -0.1004650485701859
source_positive_lost_after_generation_rate = 0.9215686274509803
generator_preserve_pass = 0
generator_improve_pass = 0
```

判断：P3 延续 v9.4.2 的 measured damage 结论。AP0b-AP0f 对已测 PANEL-S256 source positives 的保真很差；本轮没有伪造 ORC64/legal panel 的 generated outcomes。

## 7. P4 direct source generator reset

Artifacts：

```text
p4_direct_source_generator_reset.csv
direct_generator_payload_trace_v9430.csv
direct_generator_outcome_trace_v9430.csv
direct_generator_completion_trace_v9430.csv
direct_generator_retry_manifest_v9430.csv
direct_source_payload_shards_v9430/
```

Summary：

```text
direct_action_count = 60
generated_action_count_total = 60
branch_horizon_row_count_expected/actual = 1620 / 1620
source_outcome_materialized = 1
action_apply_error_linf_max = 0.0
action_apply_cosine_min = 1.0
best_direct_primitive_id = AP0j-LowRankEdgeSafeSource
best_h20_weak_CP_precision = 0.25
best_h20_V_ctrl_lcb = -0.20831185402129682
best_h20_bad_event_rate = 0.0
best_h240_long_risk_rate = 0.8333333333333334
direct_generator_weak_pass = 0
direct_generator_strong_pass = 0
```

Per-primitive examples：

| primitive | h20 weak CP | h20 V_ctrl LCB | h240 weak CP | h240 long-risk |
|---|---:|---:|---:|---:|
| `AP0g-GradientAlignedLastEdgeSource` | `0.166667` | `-0.404503` | `0.000000` | `0.333333` |
| `AP0h-TailMarginConservativeSource` | `0.083333` | `-0.500647` | `0.250000` | `0.750000` |
| `AP0i-AdamWResidualBlendSource` | `0.166667` | `-0.372920` | `0.000000` | high / not survivor |
| `AP0j-LowRankEdgeSafeSource` | `0.250000` | `-0.208312` | not sufficient | `0.833333` |

判断：P4 是真实 direct generator smoke，不是 estimate。它完整跑出了 1620 行，但 best h20 V_ctrl LCB 仍为负，h240 long-risk 过高，因此不能 official。

## 8. P5 effect-valid certificate

Artifacts：

```text
p5_effect_valid_certificate_redesign.csv
certificate_effect_trace_v9430.csv
```

Summary：

```text
certificate_id = CERT5-MinimalHybridEffectCert-direct-smoke
joined_outcome_count = 180
certificate_pass_action_count = 180
AUC_weak_CP = 0.5069044879171462
AUC_strong_CP = 0.6217142857142857
AUC_longrisk = 0.5581680996666082
P_weak_CP_given_cert_pass = 0.12222222222222222
P_longrisk_given_cert_pass = 0.22777777777777777
monotone_sign_pass = 0
certificate_effect_valid_pass = 0
certificate_effect_valid_strong_pass = 0
```

判断：certificate 在 direct smoke 上不是 effect-valid sufficient statistic。AUC weak CP 接近随机，且不能可靠压低 long-risk。

## 9. P6/P7 controller runtime boundary

Artifacts：

```text
p6_minimal_source_certificate_controller.csv
controller_frontier_trace_v9430.csv
p7_selected_source_online_runtime.csv
runtime_component_trace_v9430.csv
```

Boundary：

```text
P6 status = not_run, reason = upstream_source_or_certificate_gate_failed
source_controller_pass = 0
P7 status = not_run, reason = P6_controller_not_selected
selected_runtime_pass = 0
```

判断：没有 source survivor / certificate pass，就不能选择 controller 或跑 selected runtime。没有把 direct generator smoke runtime 估计写成 runtime pass。

## 10. P8 Base-Acc Sentinel extended

Artifacts：

```text
p8_base_acc_sentinel_extended.csv
base_acc_training_trace_v9430.csv
```

Summary：

```text
sentinel_row_count = 45
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
model_count = 3
sentinel_complete = 1
mean_test_acc_LQ = 0.64921875
mean_test_acc_MLP = 0.5609375
LQ_minus_MLP_mean_test_acc = 0.08828124999999998
LQ_minus_QuadraticFeatureMLP_mean_test_acc = 0.23776041666666664
LQ_catastrophic_fail = 0
base_acc_used_for_controller = 0
AdamWStrongLRGridMLP_diagnostic_materialized = 0
```

判断：Base-Acc Sentinel 没有发现 LQ-t2-h256 catastrophic fail，且严格未用于 selector/controller。计划中的 `AdamWStrongLRGridMLP` diagnostic 尚未 materialize，因此不能把 P8 写成完整强 LR grid comparison。

## 11. P9 / downstream boundary

Artifact：

```text
p9_system_integration_gate_v9430.csv
```

Boundary：

```text
system_candidate_id = SYS-v9430-source-frontier-direct-generator
controller_id = not_selected
primitive_id = AP0j-LowRankEdgeSafeSource
certificate_id = CERT5-MinimalHybridEffectCert-direct-smoke
runtime_candidate_id = not_selected
decision_gate_pass = 0
runtime_gate_pass = 0
source_lifecycle_pass = 1
outcome_materializer_pass = 1
certificate_effect_valid_pass = 0
base_acc_sentinel_complete = 1
base_acc_used_for_controller = 0
official_eligible = 0
system_legal_controller_pass = 0
reason_if_fail = full_ap0_oracle_source_frontier_absent
```

Downstream artifact 均为 `not_run`：

| artifact | reason |
|---|---|
| `p10_leave_dataset_stratum_out_boundary.csv` | `P9_system_controller_not_official` |
| `p11_official_paired_replay_boundary.csv` | same |
| `p12_short_full_training_mlp_comparison_boundary.csv` | `P11_official_paired_replay_not_open` |

没有把 oracle panel、direct generator smoke、Base-Acc Sentinel 或 certificate diagnostic 写成 LDO/paired replay/short-full success。

## 12. No-fake audit

```text
rows_checked = 2670
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
source_outcome_materializer_closed_from_v9420 = 1
multi_panel_source_frontier_measured = 1
diagnostic_oracle_panel_used_for_official = 0
legal_source_selector_weak_pass = 0
direct_generator_materialized = 1
direct_generator_weak_pass = 0
certificate_effect_valid_pass = 0
base_acc_sentinel_complete = 1
base_acc_used_for_controller = 0
source_controller_pass = 0
selected_runtime_pass = 0
system_legal_controller_pass = 0
uses_dataset_name_for_selector/controller = 0/0
uses_validation_or_test_for_controller = 0
uses_future_outcome_for_features = 0
uses_outcome_at_commit = 0
diagnostic_promoted_to_official = 0
fake/proxy/cpu_offload = 0/0/0
```

Failure table：

```text
F1_source_outcome_materializer_regression = 0
F2_oracle_source_absent = 1
F3_legal_source_selector_opaque = 1
F4_ap0b_ap0f_transform_damage = 1
F5_direct_generator_fail = 1
F6_certificate_fail = 1
F7_controller_not_selected = 1
F8_runtime_not_selected = 1
F9_base_acc_regression = 0
primary_blocker = full_ap0_oracle_source_frontier_absent
```

## 13. Hash

| artifact | SHA256 |
|---|---|
| plan | `1e21fc7c01ed6a3cc0c57e5d4775e76320aa97af2079a9a788847bcc08b32edf` |
| runner | `c5e5d8f3eb925758679c55177631e816d746a77f0c880148c1caa2fbfb0042f3` |
| run manifest | `c2cb031deae61738a045f97aac02f009f445e5d73f02dd338a92f5751612365c` |
| route | `62832fc13c6ad526fdc613f874b2405041e09148c545465df20a18f5c99000a7` |
| P0 boundary | `d23282e6298ba5e97c1a8cd208e70ca4c4294e8196c06e12d9d4707ed68ebf3c` |
| P1 multi-panel | `9164214a5321d9229e0a49f037c4fa622e0ebe93975f500754b58af2c321b776` |
| P2 AP0 baseline | `dd3a552254326fc4ee98a6d2781677e9a80d542840bc1a3bcb424d2c4be2638c` |
| P3 damage | `cb0fdb512dbc89f746639c3be4b85c8512b5a04a02261266facee96c588bb427` |
| P4 direct generator | `114453dc122ec78c41708df1387083987b845abe86d4c409170b9a758e5b377b` |
| direct outcome trace | `5645edf7695ac5f24111bb64e6e07f43c3f4ec18ab5ad279148022db28acf3be` |
| P5 certificate | `d45692d8dab1ca692c1279e466eac298cc4fc95aa12dff9cf0fc9083e877672e` |
| P8 Base-Acc Sentinel | `d282d2b587244b9c9032d29f0ab8a2a4cad371b81bb248ea30689625f83adfb6` |
| P9 system | `af6bc4f09ec4e09201930f2c031978165f9cf4bff9186cf2d30c0ace8072dd4e` |
| contract audit | `71642add24657bb51e3f8c8eade082afb05af3f39acc9327999d9cc5e95017a7` |
| provenance audit | `a6173ea50a65a5f69561914e2d8c56f9f159e56d62868b94aebd7e72d4b526ef` |
| failure table | `696c29249e6caf66c241119fd23f26ee7804f516d9812526aa096af16050363a` |

## 14. 最终分析结论

v9.4.3 的真实推进是：

```text
v9.4.2:
  source outcome materializer 已从 0 行修到完整 7020 行；
  但 stratified source panel 和 AP0b-AP0f generated source 都 value-poor。

v9.4.3:
  不再只看单一 stratified panel，而是跑 multi-panel AP0 source frontier；
  oracle panel 有接近 gate 的 source signal，但 h20 V_ctrl LCB 仍为负；
  legal selector 仍明显找不到 frontier；
  direct generator AP0g-AP0k 真实 materialize 60 actions / 1620 rows，但 h20 V_ctrl LCB 为负且 h240 long-risk 高；
  Base-Acc Sentinel 5-seed 完成且未用于 controller。
```

机制判断：

1. H1 未成立：full AP0 source oracle frontier 在本轮计划 gate 下没有通过；best ORC64 h20 weak CP = `0.578125`，但 V_ctrl LCB = `-0.020408`。
2. H2 未成立：legal source selector 仍不够，best legal h20 weak CP = `0.171875`，V_ctrl LCB 为负，h240 long-risk = `0.890625`。
3. H3 成立为 diagnostic：AP0b-AP0f transform damage 仍明显，source-positive lost rate = `0.921569`。
4. H4 未成立：direct generator 确实跑通了 payload/outcome，但 best h20 weak CP = `0.25`、V_ctrl LCB = `-0.208312`，不满足 h20-positive gate。
5. H5 未成立：certificate AUC weak CP = `0.506904`，monotone sign pass = `0`，不是 effect-valid certificate。
6. H6 部分成立：Base-Acc Sentinel 没有 catastrophic fail；但 StrongLRGrid diagnostic 未 materialize，且 sentinel 没有用于 controller。
7. H7 未打开：P9 system controller pass = `0`，P10-P12 全部 gate-blocked。

最终一句话：

> v9.4.3 真实执行后停在 `R2-FullAP0OracleSourceFrontierAbsent`：multi-panel AP0 oracle source 只有接近但未过 gate 的信号，legal selector 和 AP0g-AP0k direct generator 都没形成 h20-positive / h240-safe frontier；strict PureKAN functional 仍不能转正。
