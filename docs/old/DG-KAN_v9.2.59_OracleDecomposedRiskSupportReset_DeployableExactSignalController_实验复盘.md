# DG-KAN v9.2.59 Oracle-Decomposed Risk/Support Reset 与 Deployable Exact-Signal Controller 实验复盘

> 本复盘记录 `DG-KAN_v9.2.59_OracleDecomposedRiskSupportReset_DeployableExactSignalController_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R11-RiskSupportStatsStillTinyCleanCore
base_candidate = LQ-t2-h256
success_v9259_strict_purekan_functional = False
success_v9259_full_functional = False
success_v9259_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9259_oracle_decomposed_risk_support_reset_deployable_exact_signal_controller_first_20260512T123000Z/
```

核心结论：

1. P0 复现 v9.2.58 boundary：source route = `R10-ExactReferenceStillInfeasible`，risk_stat_pass = `1`，exact_reference_feasible = `0`，oracle_support_pass = `1`，fake/proxy/offload = `0`。
2. 本轮 manifest 记录 `device = cuda`，`triton_available = true`；运行中 GPU 采样有活动：`NVIDIA L4, 32-33 %, 326 MiB`。
3. P1 oracle-decomposed factor label/autopsy pass = `1`：FP `1108`，FN `1388`，accepted bad-event `1035`，三类 attribution fraction 均为 `1.0`。
4. P2 四个 bad-event submode 都有预测性，但组合 risk gate 仍是 clean-too-tiny：risk union bad-event = `0.0`，coverage = `0.001860`，因此 `mode_specific_risk_pass = 0`。
5. P3 support reset 未过：best = `SUPR2-LeaveFamilyOutReliabilityV2`，precision = `0.382818`，coverage = `0.082279`，bad-event = `0.542577`，Jaccard = `0.328803`。
6. P4 exact reference feasibility v3 仍未恢复：best = `RISKM3-ControlDominanceRisk + SUPR4-ModeAwareSupportPocket`，precision = `0.886792`，coverage = `0.003286`，bad-event = `0.037736`。
7. P5 true-delta compute v4 仍未过：best = `TBD0-V9256CBD0Reference`，AUC = `0.883468`，agreement = `1.0`，step ratio = `2.863280 > 1.50`。
8. P6 oracle support 仍通过：precision = `1.0`，coverage = `0.119978`，bad-event = `0.0`；但 support measurement 仍失败：signal strata = `3`，balanced diagnostic rows = `162`。
9. P7-P10 因 `P2_risk_union_gate_tiny_coverage` / `P4_exact_reference_infeasible` gate-blocked，均以 `not_run` 落盘。
10. 当前 blocker：`risk_union_gate_tiny_coverage`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9259_oracle_decomposed_risk_support_reset_deployable_exact_signal_controller.py` | v9.2.59 runner；执行 v9.2.58 boundary 复现、oracle-decomposed factor labels、mode-specific risk、support reset、exact reference feasibility v3、true-delta compute v4、support/downstream/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9259_oracle_decomposed_risk_support_reset_deployable_exact_signal_controller.py
```

正式运行：

```bash
python experiments/run_v9259_oracle_decomposed_risk_support_reset_deployable_exact_signal_controller.py \
  --out-dir results/real_rerun_20260506/v9259_oracle_decomposed_risk_support_reset_deployable_exact_signal_controller_first_20260512T123000Z \
  --fresh --device auto --data-root data --seed 1314
```

实际参数来自 `run_manifest.json`：

```text
device = cuda
triton_available = true
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7
microprobe_steps = 224
interface_events = 24
interface_steps = 2
train_size = 2048
batch_size = 64
hidden_dim = 256
completed_at = 2026-05-12T12:12:14Z
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R11-RiskSupportStatsStillTinyCleanCore",
  "base_candidate": "LQ-t2-h256",
  "v9258_boundary_pass": 1,
  "factor_label_autopsy_pass": 1,
  "fp_mode_attribution_fraction": 1.0,
  "fn_mode_attribution_fraction": 1.0,
  "bad_event_mode_attribution_fraction": 1.0,
  "best_risk_stat_id": "RISKM6-ModeSpecificRiskUnion",
  "mode_specific_risk_pass": 0,
  "mode_specific_risk_diagnostic_pass": 0,
  "risk_union_gate_bad_event": 0.0,
  "risk_union_gate_coverage": 0.0018601190476190475,
  "best_support_stat_id": "SUPR2-LeaveFamilyOutReliabilityV2",
  "support_stat_pass": 0,
  "support_diagnostic_pass": 0,
  "legal_oracle_jaccard": 0.06420765027322405,
  "accepted_family_count": 14,
  "accepted_signal_strata_count": 1,
  "max_family_share": 0.22641509433962265,
  "exact_reference_feasible": 0,
  "reference_controller_precision": 0.8867924528301887,
  "reference_controller_coverage": 0.0032862103174603175,
  "reference_controller_bad_event": 0.03773584905660377,
  "best_true_delta_id": "TBD0-V9256CBD0Reference",
  "true_delta_compute_pass": 0,
  "true_delta_auc": 0.8834681020409492,
  "true_delta_agreement": 1.0,
  "true_delta_step_ratio_q90": 2.8632798851361203,
  "true_delta_memory_ratio": 0.9695007261731864,
  "oracle_support_pass": 1,
  "oracle_precision": 1.0,
  "oracle_coverage": 0.11997767857142858,
  "oracle_bad_event": 0.0,
  "support_measurement_pass": 0,
  "natural_real_event_count": 16128,
  "balanced_diagnostic_real_event_count": 162,
  "measured_signal_strata_count": 3,
  "measured_family_count": 482,
  "primary_blocker": "risk_union_gate_tiny_coverage",
  "next_required_implementation": "reset_risk_support_target_decomposition"
}
```

判断：bad-event submodes 不是完全不可预测；问题是组合 risk/support gate 一旦把 bad-event 压住，coverage 仍远低于 `0.03`。

## 3. P1 oracle-decomposed labels

Artifacts：

```text
p1_oracle_decomposed_label_construction.csv
factor_label_trace_v9259.csv
```

| metric | value |
|---|---:|
| row count | `16128` |
| factor label validity pass | `1` |
| factor label autopsy pass | `1` |
| Y_value rate | `0.636657` |
| Y_control rate | `0.491009` |
| Y_tail_safe rate | `0.452629` |
| Y_consistent rate | `0.441034` |
| Y_support rate | `0.455977` |
| Y_factor_safe rate | `0.004526` |
| factor_safe oracle precision | `0.383562` |
| false positive count | `1108` |
| false negative count | `1388` |
| bad-event accepted count | `1035` |

主要归因：

| type | primary mode | attribution fraction |
|---|---|---:|
| FP | `FPA6-tail_instability` | `1.0` |
| FN | `FNA4-family_scarcity` | `1.0` |
| bad-event | `BE1-tail_instability` | `1.0` |

判断：factor labels 可构造，failure autopsy 能归因；但 `Y_factor_safe` 本身只覆盖 `0.004526`，说明 factorized safe core 仍非常窄。

## 4. P2 mode-specific risk statistics

Artifacts：

```text
p2_mode_specific_risk_statistics.csv
risk_submode_trace_v9259.csv
```

Best submode statistics：

| submode | best risk stat | AUC | corr |
|---|---|---:|---:|
| tail instability | `RISKM7-ModeSpecificRiskMean` | `0.867727` | `0.587667` |
| delta-gain mismatch | `RISKM2-DeltaGainMismatchRisk` | `0.755497` | `0.009467` |
| control dominance | `RISKM5-FamilyRiskLCBv2` | `0.745808` | `0.385827` |
| family scarcity | `RISKM5-FamilyRiskLCBv2` | `0.896216` | `0.664642` |

Summary：

```text
best_risk_stat_id = RISKM6-ModeSpecificRiskUnion
predictive_submode_count = 4
risk_union_gate_precision = 0.9
risk_union_gate_bad_event = 0.0
risk_union_gate_coverage = 0.0018601190476190475
mode_specific_risk_pass = 0
mode_specific_risk_diagnostic_pass = 0
```

判断：H2 的 “submodes 有 legal signal” 成立，但组合 gate 没有达到 `coverage >= 0.03`，甚至低于 diagnostic 的 `0.01`。因此 route 写为 clean core too tiny，而不是写成 controller success。

## 5. P3 support sufficient statistics reset

Artifacts：

```text
p3_support_sufficient_statistics_reset.csv
support_family_trace_v9259.csv
```

| support stat | precision | coverage | bad-event | Jaccard | families | strata | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| SUPR1-DeltaTailFamilyV2 | `0.385647` | `0.078621` | `0.537855` | `0.324917` | `114` | `2` | `0` |
| SUPR2-LeaveFamilyOutReliabilityV2 | `0.382818` | `0.082279` | `0.542577` | `0.328803` | `119` | `2` | `0` |
| SUPR3-KNNDensityInRiskSupportSpace | `0.284260` | `0.079179` | `0.533281` | `0.221341` | `140` | `2` | `0` |
| SUPR4-ModeAwareSupportPocket | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0` | `0` | `0` |
| SUPR5-SupportRiskJointExpansion | `1.000000` | `0.000620` | `0.000000` | `0.013774` | `6` | `2` | `0` |

判断：support reset 增加了 family granularity（measured reset families = `482`），但没有形成 deployable support gate。宽 support 的 bad-event 太高；干净 support 又只剩 `0.000620` coverage。

## 6. P4 exact reference feasibility v3

Artifacts：

```text
p4_exact_reference_feasibility_v3.csv
reference_feasibility_v3_trace.csv
accepted_region_geometry_trace_v9259.csv
```

Best summary：

```text
best_reference_controller_id = C-RISKM3-ControlDominanceRisk+SUPR4-ModeAwareSupportPocket
best_risk_stat_id = RISKM3-ControlDominanceRisk
best_support_stat_id = SUPR4-ModeAwareSupportPocket
exact_reference_feasible = 0
precision = 0.8867924528301887
coverage = 0.0032862103174603175
bad_event = 0.03773584905660377
accepted_signal_strata_count = 1
accepted_family_count = 14
max_family_share = 0.22641509433962265
legal_oracle_jaccard = 0.06420765027322405
```

判断：P4 仍是不可部署。它能把 bad-event 压到 `<=0.05`，precision 也超过 `0.75`，但 coverage 只有 `0.003286`，且 accepted strata 只有 `1`，没有达到 `coverage >= 0.03` / `accepted_signal_strata_count >= 2`。

## 7. P5 true-delta compute v4

Artifacts：

```text
p5_true_delta_compute_v4_parallel_lane.csv
true_delta_compute_v4_trace.csv
```

Summary：

```text
best_true_delta_id = TBD0-V9256CBD0Reference
true_delta_auc = 0.8834681020409492
true_delta_agreement = 1.0
true_delta_step_ratio_q90 = 2.8632798851361203
true_delta_memory_ratio = 0.9695007261731864
true_delta_compute_pass = 0
true_delta_compute_diagnostic_pass = 0
```

判断：true-delta exact signal 仍强，但 compute lane 没进 system envelope；本轮主 blocker 已经在 P2/P4，P5 也没有提供可转正系统路径。

## 8. P6 support expansion

Artifacts：

```text
p6_online_support_stratum_expansion.csv
support_density_trace_v9259.csv
```

Summary：

```text
natural_real_event_count = 16128
balanced_diagnostic_real_event_count = 162
measured_signal_strata_count = 3
measured_family_count = 482
support_measurement_pass = 0
oracle_support_pass = 1
oracle_precision = 1.0
oracle_coverage = 0.11997767857142858
oracle_bad_event = 0.0
```

判断：natural rows 达到计划的 `>=16000`，但 balanced diagnostic rows 和 measured signal strata 没达标。这里没有复制/补造 balanced rows；`162` 是真实 selector 能选到的行数。

## 9. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p7_system_legal_exact_signal_controller.csv` | `P4_exact_reference_infeasible` |
| `p8_leave_dataset_and_stratum_out.csv` | `P2_risk_union_gate_tiny_coverage` |
| `p9_official_paired_replay.csv` | same |
| `p10_short_run_functional_validation.csv` | same |

没有把 factor labels、submode AUC、oracle support、clean-tiny slice 或 exact reference signal 写成 LDO/LSO、paired replay、short-run 或 functional success。

## 10. No-fake audit

```text
rows_checked = 65057
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

## 11. Hash

| artifact | SHA256 |
|---|---|
| plan | `e1a0036da6aa5e338bfa123838df5fe139efbba00a849ca5b81b5b7b6f0b539b` |
| runner | `51d8fb9253fa29f6fb4e31ba83a808968cd1a2b102db1fc6f36849e1ec500feb` |
| run manifest | `6afa1fddf051883f948b41b4ff0fc2594782ae29364fd0e8a1364f7c1063ac16` |
| route | `1623c8b2a5ae937bda82c7e046319ed4a9ff7e5e64b3d77f8c886859a850c016` |
| P0 boundary | `a3d4ae2422f82a4d629f03ca9ec2185afc2b7e4a9b24f91904d11e4563520915` |
| P1 factor labels | `1a0be31d74288b28e19ea24023a3aa0a93e2b4f5335602a1eb9502eee2ac0448` |
| P2 risk statistics | `def00c86e59e9b46d7adccc974e15f48ec6f667c375d97aa5521bd3b3f29c6c1` |
| P3 support reset | `4aa7a37d5955878890df9e5f2299ebfbd9ec3d72f6f989ae3034b1fafc9af31a` |
| P4 reference feasibility | `dd8e2fc420f3b6d8d95e1e5ed58bc25818e99d48507e40222c35e3c0db208255` |
| P5 compute lane | `320786c0597cbfd0d5c21f0941440695093100a4cf8f6a6ada748e9521d48a1b` |
| P6 support expansion | `963c18f7345e09c7c1e84b43946e65055f032e41a5a00a466da1dce36791439e` |
| P7 boundary | `6a881b23d4fa01cf4e94c0d4ea4d2fa470ea73799257ca0b98a2e693351099b7` |
| P8 boundary | `1419af92ad25703893304c36be056e00d12d2b702272f9ded4f4a0661005c6ff` |
| P9 boundary | `867fbcc8d1a44bd54074d4c4e4bfdbc98a94117ca323a01b270a5a8d53cc7a2c` |
| P10 boundary | `d4492e935c561f5d850fb559a56f2a264182cedeacc0e62e1c91a88511e86b83` |
| failure table | `38b260718db74b8157c0139c9e476b18f3aeef20efcbf841f528d0fede4544a5` |
| provenance audit | `e3b3411afd339f53c699d58c345d62963b421591df69829709b167d4804b07a5` |

## 12. 最终分析结论

v9.2.59 的真实推进是：

```text
v9.2.58: risk/support 有诊断信号，但 exact reference accept region 仍不可部署。
v9.2.59: oracle factor decomposition 与 bad-event submode risk 均已测；
          submode risk 有预测性，但组合后仍只能得到 clean-too-tiny slice。
```

机制判断：

1. H1 没有闭合：factorized target 后，exact reference best coverage 仍只有 `0.003286`。
2. H2 只部分成立：四个 bad-event submodes 都有 predictive risk statistic，但 combined risk utility 不过，coverage 只有 `0.001860`。
3. H3 未成立：support reset 没有把 Jaccard 提到 `>=0.60`，best 只有 `0.328803`。
4. H4 未成立：natural rows 足够，但 measured signal strata 仍是 `3`，balanced diagnostic rows 只有 `162`。
5. H5 未成立：true-delta compute 仍 `step_ratio_q90 = 2.863280`，超过 `1.50`。

最终一句话：

> v9.2.59 真实执行后停在 `R11-RiskSupportStatsStillTinyCleanCore`：oracle-decomposed submode risk 有信号，但组合 risk/support 仍只能得到 coverage 过小的 clean core，strict PureKAN functional 仍未成功。
