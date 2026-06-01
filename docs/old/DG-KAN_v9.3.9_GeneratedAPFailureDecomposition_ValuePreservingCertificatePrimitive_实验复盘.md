# DG-KAN v9.3.9 Generated-AP Failure Decomposition / Value-Preserving Certificate Primitive 实验复盘

> 本复盘记录 `DG-KAN_v9.3.9_GeneratedAPFailureDecomposition_ValuePreservingCertificatePrimitive_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 AP0 旧 outcome、certificate diagnostic、low-distortion sweep 或未选中 runtime 写成 official system pass。

## 0. 最新结论

```text
route = R1-SourceAP0PanelBad
base_candidate = LQ-t2-h256
success_v9390_strict_purekan_functional = False
success_v9390_full_functional = False
success_v9390_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9390_generated_ap_failure_decomposition_value_preserving_first_20260514T060000Z/
```

核心结论：

1. P0 复现 v9.3.8 boundary：source route = `R5-APGeneratedActionsValueFail`，generated AP action = `256`，certificate-pass outcome rows = `249`，weak CP precision = `0.10040160642570281`。
2. P1 同源 AP0 same-panel baseline 已真实 materialize：source AP0 action = `64`，branch-horizon rows = `1728`，completion = `1.0`。
3. P1 证明 source AP0 panel 本身不够好：weak CP precision = `0.07291666666666667`，strong CP precision = `0.03125`，horizon-robust precision = `0.0`，long-risk rate = `0.3072916666666667`，V_ctrl LCB = `-1.4980123384538895`。
4. 因 P1 failed，本轮 route 正确停在 `R1-SourceAP0PanelBad`；不能继续把 AP generated failure 只归因于 AP1-AP4 transform。
5. P2 transformation damage 已量化：paired horizon = `768`，Damage median = `0.04009834537282586`，source-positive lost rate = `0.9107142857142857`；但 generator damage gate = `0`。
6. P3 certificate lift 很弱：cert-pass weak CP = `0.10040160642570281`，cert-fail weak CP = `0.04431599229287091`，Lift_weak = `2.2655840754321632`；但 long-risk lift = `0.9786218042903322`，AUC_certificate_weak_CP = `0.5964409722222223`，certificate sufficiency pass = `0`。
7. P4 horizon dissection 指向 immediate direction failure：weak CP h20/h80/h240 = `0.02734375 / 0.1015625 / 0.05859375`，long-risk h240 = `0.94140625`，immediate_direction_fail = `1`。
8. P5/P6 AP5-AP8 低扰动 value-preserving primitive 已真实生成：generated action = `256`，certificate pass action = `64`，payload/certificate hash missing = `0`，action apply L∞ max = `0.0`。
9. P7 AP5-AP8 smoke outcomes 已真实 materialize：branch-horizon rows = `6912`，completion = `1.0`，但 cert-pass weak CP precision 仍只有 `0.07291666666666667`，strong CP = `0.026041666666666668`，horizon robust = `0.0`，long-risk = `0.296875`，V_ctrl LCB = `-1.4386910355299591`。
10. P8 frontier/calibration 未过：best candidate = `C0-cert-pass-only-AP7-HorizonGuardedConservativeCertificate`，accepted = `64`，coverage = `0.007054673721340388`，weak CP precision = `0.07291666666666667`。
11. P9/P10 及 downstream 均 gate-blocked：没有 selected controller，因此 payload runtime diagnostic 与 system controller 均为 `not_run`。
12. 当前 primary blocker：`source_AP0_same_panel_bad`；下一步需要重建 source action selection / candidate source，而不是继续只做 AP transform 或 certificate threshold。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9390_generated_ap_failure_decomposition_value_preserving.py` | v9.3.9 runner；复用 v9.3.8 generated AP outcomes，materialize AP0 same-panel baseline，分解 source-to-generated damage / certificate lift / horizon failure，并真实生成 AP5-AP8 low-distortion payload 与 smoke outcomes |

代码检查：

```text
python -m py_compile experiments/run_v9390_generated_ap_failure_decomposition_value_preserving.py
```

正式运行：

```bash
python experiments/run_v9390_generated_ap_failure_decomposition_value_preserving.py \
  --out-dir results/real_rerun_20260506/v9390_generated_ap_failure_decomposition_value_preserving_first_20260514T060000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --source-actions 64 --smoke-actions-per-primitive 64
```

运行结果：

```json
{
  "ap_v2_weak": 0.07291666666666667,
  "out_dir": "results/real_rerun_20260506/v9390_generated_ap_failure_decomposition_value_preserving_first_20260514T060000Z",
  "route": "R1-SourceAP0PanelBad",
  "source_AP0_weak": 0.07291666666666667
}
```

## 2. Route

`route_decision.json` 摘要：

```json
{
  "route": "R1-SourceAP0PanelBad",
  "source_route_v9380": "R5-APGeneratedActionsValueFail",
  "source_AP0_same_panel_pass": 0,
  "source_AP0_panel_bad": 1,
  "source_AP0_weak_CP_precision": 0.07291666666666667,
  "source_AP0_long_risk_rate": 0.3072916666666667,
  "source_AP0_V_ctrl_lcb": -1.4980123384538895,
  "Damage_median": 0.04009834537282586,
  "source_positive_lost_after_generation_rate": 0.9107142857142857,
  "certificate_sufficiency_pass": 0,
  "AUC_certificate_weak_CP": 0.5964409722222223,
  "Lift_weak": 2.2655840754321632,
  "Lift_longrisk": 0.9786218042903322,
  "horizon_failure_mode": "immediate_direction_fail",
  "ap5_ap8_generation_pass": 1,
  "ap_v2_smoke_outcome_pass": 1,
  "ap_v2_cert_pass_weak_CP_precision": 0.07291666666666667,
  "ap_v2_cert_pass_long_risk_rate": 0.296875,
  "ap_v2_cert_pass_V_ctrl_lcb": -1.4386910355299591,
  "ap_v2_frontier_pass": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "source_AP0_same_panel_bad",
  "next_required_implementation": "rebuild_source_action_selection_or_candidate_source"
}
```

判断：v9.3.9 把 v9.3.8 的 generated AP failure 往前追溯到了 source AP0 panel。AP transform 有损伤，certificate 也不充分，但更早的 blocker 是同一 source panel 上 AP0 自身 already weak / long-risk high。

## 3. P0 v9.3.8 boundary

Artifact：

```text
p0_v9380_boundary_reanalysis.csv
```

Summary：

```text
source_route = R5-APGeneratedActionsValueFail
generated_action_count_total = 256
primitive_materialized_count = 4
actions_per_primitive = 64
certificate_pass_outcome_count = 249
weak_CP_precision_certificate_pass = 0.10040160642570281
strong_CP_precision_certificate_pass = 0.028112449799196786
long_risk_rate_certificate_pass = 0.3092369477911647
branch_horizon_row_count_actual = 6912
branch_horizon_completion_confidence = 1
```

判断：v9.3.8 的 materializer 与 generated AP outcome boundary 被复现；不是从旧 AP0 outcomes 直接做结论。

## 4. P1 source AP0 same-panel baseline

Artifacts：

```text
p1_source_ap0_same_panel_baseline.csv
source_ap0_baseline_summary_v9390.csv
source_ap0_branch_horizon_trace_v9390.csv
```

Summary：

```text
source_action_count = 64
branch_horizon_row_count_actual = 1728
branch_horizon_completion_rate = 1.0
source_AP0_weak_CP_count = 14
source_AP0_weak_CP_precision = 0.07291666666666667
source_AP0_strong_CP_count = 6
source_AP0_strong_CP_precision = 0.03125
source_AP0_horizon_robust_CP_count = 0
source_AP0_horizon_robust_CP_precision = 0.0
source_AP0_long_risk_count = 59
source_AP0_long_risk_rate = 0.3072916666666667
source_AP0_bad_event_rate = 0.17708333333333334
source_AP0_null_rate = 0.036458333333333336
source_AP0_V_ctrl_mean = -1.329557194900796
source_AP0_V_ctrl_lcb = -1.4980123384538895
source_AP0_same_panel_pass = 0
```

判断：这是本轮决定 route 的关键结果。AP0 source panel 本身没有形成 control-positive / horizon-safe value frontier，因此不能把 v9.3.8 failure 主要归因于 AP1-AP4 generation。

## 5. P2 transformation damage

Artifacts：

```text
p2_transformation_damage_matrix.csv
source_generated_payload_geometry_trace_v9390.csv
damage_attribution_summary_v9390.csv
```

Summary：

```text
paired_horizon_count = 768
Damage_mean = 0.06315663311083124
Damage_median = 0.04009834537282586
Damage_lcb = -0.05223670554137398
P_Damage_lt_minus_0p05 = 0.4635416666666667
source_positive_horizon_count = 56
source_positive_lost_after_generation_count = 51
source_positive_lost_after_generation_rate = 0.9107142857142857
generator_damage_pass = 0
operation_attribution_pass = 0
```

判断：generated transforms 确实会丢掉大量 source-positive horizon rows，但 damage 指标本身没有达到足以把 route 改成 generator-damage-primary。主 blocker 仍是 source AP0 panel bad。

## 6. P3 certificate sufficiency

Artifacts：

```text
p3_certificate_sufficiency_lift_audit.csv
certificate_component_trace_v9390.csv
certificate_ablation_summary_v9390.csv
```

Summary：

```text
cert_pass_rows = 249
cert_fail_rows = 519
P_weak_CP_given_cert_pass = 0.10040160642570281
P_weak_CP_given_cert_fail = 0.04431599229287091
P_longrisk_given_cert_pass = 0.3092369477911647
P_longrisk_given_cert_fail = 0.3159922928709056
Lift_weak = 2.2655840754321632
Lift_longrisk = 0.9786218042903322
AUC_certificate_weak_CP = 0.5964409722222223
AUC_certificate_strong_CP = 0.5625831117021277
AUC_certificate_longrisk = 0.5071688962025716
monotone_sign_pass = 0
certificate_sufficiency_pass = 0
certificate_insufficient = 1
```

判断：certificate 对 weak CP 有轻微信号，但强度远不够，而且不能降低 long-risk；它不是 sufficient statistic。

## 7. P4 horizon failure dissection

Artifact：

```text
p4_horizon_failure_dissection.csv
horizon_value_curve_trace_v9390.csv
```

Summary：

```text
weak_CP_precision_h20 = 0.02734375
weak_CP_precision_h80 = 0.1015625
weak_CP_precision_h240 = 0.05859375
long_risk_rate_h20 = 0.0
long_risk_rate_h80 = 0.0
long_risk_rate_h240 = 0.94140625
V_ctrl_lcb_h20 = -1.3643616847235815
short_only_CP_count = 7
weak_CP_h20_count = 7
long_horizon_risk_primary = 0
immediate_direction_fail = 1
```

判断：不是单纯长 horizon 把一个好 immediate action 拖坏；h20 本身 weak CP precision 也只有 `0.027344`，所以 failure mode 被判为 `immediate_direction_fail`。

## 8. P5-P7 AP5-AP8 value-preserving attempt

Artifacts：

```text
p5_generic_low_distortion_sweep.csv
p6_real_ap5_ap8_generator.csv
ap5_ap8_payload_trace_v9390.csv
ap5_ap8_certificate_trace_v9390.csv
action_apply_replay_trace_v9390.csv
p7_generated_ap_v2_smoke_outcome.csv
ap_v2_smoke_outcome_trace_v9390.csv
branch_horizon_completion_trace_v9390.csv
```

P6 generation：

```text
primitive_count = 4
generated_action_count = 256
source_action_count = 64
max_generated_action_count_per_primitive = 64
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_error_linf_max = 0.0
certificate_schema_contract_pass = 1
certificate_pass_count = 64
ap5_ap8_generation_pass = 1
```

P7 outcomes：

```text
generated_action_count_input = 256
branch_horizon_row_count_actual = 6912
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
ap_v2_smoke_outcome_pass = 1
weak_CP_precision = 0.06901041666666667
strong_CP_precision = 0.026041666666666668
horizon_robust_CP_precision = 0.0
long_risk_rate = 0.3059895833333333
V_ctrl_lcb = -1.326907307781892
cert_pass_weak_CP_precision = 0.07291666666666667
cert_pass_strong_CP_precision = 0.026041666666666668
cert_pass_horizon_robust_CP_precision = 0.0
cert_pass_long_risk_rate = 0.296875
cert_pass_V_ctrl_lcb = -1.4386910355299591
ap_v2_value_smoke_pass = 0
```

判断：AP5-AP8 的 implementation 过了，value 没过。低扰动并没有救回 source panel；这支持下一步重建 source action selection / candidate source。

## 9. P8-P13 boundary

P8：

```text
best_controller_id = C0-cert-pass-only-AP7-HorizonGuardedConservativeCertificate
best_primitive_id = AP7-HorizonGuardedConservativeCertificate
generated_ap_frontier_pass = 0
weak_survivor_for_scaleup = 0
best_weak_CP_precision_heldout = 0.07291666666666667
best_long_risk_rate_heldout = 0.296875
best_V_ctrl_lcb_heldout = -1.4386910355299591
```

P9/P10：

```text
p9_payload_runtime_diagnostic_v9390.csv = not_run, reason = P8_frontier_not_official
p10_system_gate_generated_ap_v2.csv = not_run, reason = P8_frontier_not_official
```

Downstream：

| artifact | reason |
|---|---|
| `p11_leave_dataset_stratum_out_v9390.csv` | `P10_system_controller_not_official` |
| `p12_diagnostic_paired_replay_scout_v9390.csv` | same |
| `p13_short_full_sampleeff_continual_robustness_v9390.csv` | same |

没有把 AP5-AP8 generation pass、smoke outcome pass 或 low-distortion diagnostic 写成 official system/runtime/downstream success。

## 10. No-fake audit

```text
rows_checked = 10435
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
source_AP0_same_panel_complete = 1
generator_damage_quantified = 1
certificate_lift_quantified = 1
horizon_failure_quantified = 1
ap5_ap8_generation_pass = 1
ap_v2_smoke_outcome_pass = 1
ap_v2_frontier_pass = 0
system_legal_controller_pass = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
uses_validation_or_test = 0
uses_future_outcome_for_features = 0
uses_outcome_at_commit = 0
diagnostic_promoted_to_official = 0
```

Failure table：

```text
F2_source_AP0_panel_bad = 1
F10_certificate_no_CP_lift = 1
F13_immediate_direction_fail = 1
F15_low_distortion_sweep_no_survivor = 1
F19_generated_AP_frontier_absent = 1
primary_blocker = source_AP0_same_panel_bad
```

## 11. Hash

| artifact | SHA256 |
|---|---|
| plan | `a3ca8e4641335397d41c9c0bb8d72dfcb880d92ec217a73aee449d5f63a074be` |
| runner | `3a27ee3ea7e3bc374e0c7660ba3e55135d36aa56635a9e6ba5f5116423cbc9df` |
| run manifest | `016907382b855171446da24182a9cb1dbac050dfaa0714dafc8cbaff0298d4bd` |
| route | `8935cebcc94dd516d371969ac31905803b9810fa376a55a3761b47f18a4f3566` |
| P0 boundary | `683dda206c5324c66c14936fe6a99821e4f96b1aaf77bac2ec5143b4d1706f88` |
| P1 source AP0 | `a2bb94657f03cd5ef1473bae019e684ac2ab3ab04537abb0318a7596b271d1cb` |
| P2 damage | `41f0a89e2ac60190593a8c83685ee5eff08f05521962eedb9642276e35d98eae` |
| P3 certificate | `b0b0db6c68d80ef6aec1f139e5278631a81f18f84127a33a92b3e5be1cee9399` |
| P4 horizon | `f3ddae3eff3bbe95423a0a1fc48ba36a5ce4e396dbe3c2e5eb002cca46a32a43` |
| P5 sweep | `febeae0d8eafb0bed915cf309cfd7ec1c410c9ae63c0826f1cf49d6be5f64f6e` |
| P6 AP5-AP8 | `88d38efd55387088cfd4aabd9ae431b2be67ac9d35866dfc1e817a8af56138b8` |
| P7 APv2 outcome | `ea4649da990d5a77ecc4c6dc2550ff901c3507edfa03c29d9d16b06638595b26` |
| P8 frontier | `00c0e9b58000059333042115d433ae7512967b8a6b39429ddb5638f3241e4261` |
| contract audit | `8b4ce21b56660261d3e3364ae4b3bf79b68b26ae49fe71c48fc3f9af071c551c` |
| provenance audit | `2f25c54df2c4d1e40f2889545a764234901c0f94b514a14c69ad881fbe04f8c1` |
| failure table | `0b1780a3be75d0b59e9af2e16d43bf4d8b3a4ce64bef9848c3ee8a6450a7ad0d` |

## 12. 最终分析结论

v9.3.9 的真实推进是：

```text
v9.3.8:
  AP1-AP4 真实 payload/certificate/outcome 链路打通；
  但 certificate-pass generated rows value / horizon safety 不足。

v9.3.9:
  用同源 AP0 same-panel baseline 追溯 failure；
  发现 source AP0 panel 本身 weak CP precision 只有 0.0729、long-risk 约 0.3073；
  AP5-AP8 low-distortion value-preserving generation 也不能恢复 frontier。
```

机制判断：

1. H1 反向成立：v9.3.8 的 failure 不能主要归因于 AP1-AP4 transform，因为 source AP0 same-panel baseline 本身不通过。
2. H2 部分成立：generation 后 source-positive horizon rows 大量丢失，source-positive lost rate = `0.9107`，但 route 仍优先停在 source panel bad。
3. H3 未成立：certificate lift 不足，AUC weak CP 只有 `0.5964`，且不能降低 long-risk。
4. H4 成立：horizon dissection 显示 immediate direction already weak，h20 weak CP precision 只有 `0.0273`。
5. H5 未成立：AP5-AP8 低扰动变体虽然生成与 replay 闭合，但 cert-pass weak CP precision 仍只有 `0.0729`。
6. H6 未打开：没有 generated AP frontier / controller / selected runtime，因此 downstream 全部 gate-blocked。

最终一句话：

> v9.3.9 真实执行后停在 `R1-SourceAP0PanelBad`：v9.3.8 的 AP generated failure 被追溯到同源 AP0 panel 本身 already value-poor / long-risk high；AP5-AP8 低扰动 certificate primitive 也未恢复 frontier，strict PureKAN functional 仍未成功。
