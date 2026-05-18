# DG-KAN v9.2.63 Conditional Bad-Event Separation 与 Risk-Adjusted Safe-Useful Frontier 实验复盘

> 本复盘记录 `DG-KAN_v9.2.63_ConditionalBadEventSeparation_RiskAdjustedSafeUsefulFrontier_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R16-SupportRegression
base_candidate = LQ-t2-h256
success_v9263_strict_purekan_functional = False
success_v9263_full_functional = False
success_v9263_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9263_conditional_bad_event_separation_risk_adjusted_safe_useful_frontier_first_20260512T180000Z/
```

核心结论：

1. P0 复现 v9.2.62 boundary：source route = `R12-UsefulTargetStillTiny`，exact reference deployable = `0`，true-delta compute pass = `0`，fake/proxy/offload = `0`。
2. 本轮 manifest 记录 `device = cuda`，`triton_available = true`；运行中 GPU 采样有活动。
3. P1 conditional bad-event autopsy pass = `1`：conditional bad count = `2309`，risky useful count = `5358`，harmless-null leak count = `9226`，三类 attribution fraction 均为 `1.0`。
4. P2 conditional bad-event statistic 有预测性：best = `CB5-FamilyInstabilityUCB`，AUC = `0.810598`，corr = `0.483450`，conditional bad after gate = `0.0`；但 coverage 只有 `0.000496`，utility pass = `0`。
5. P3 safe-useful score 有诊断推进：best = `SU2-SafeUsefulMargin`，AUC = `0.864844`，coverage = `0.032201`，bad-event = `0.029525`；但 precision = `0.110398`、null rate = `0.847240`，confidence/utility pass = `0`。
6. P4 support confidence 回退：natural rows = `24192`、balanced rows = `6000`、signal strata = `25`、duplicate = `0`，但 measured family count = `538 < 700`，因此 support confidence pass = `0`。
7. P4 accepted support 子 gate 仍通过：accepted strata = `9`，accepted families = `79`，max family share = `0.135254`，max stratum share = `0.453613`。
8. P5 exact reference deployable frontier 未过：best reference precision = `1.0`、bad-event = `0.0`，但 coverage = `0.000124`，precision LCB = `0.438494`，bad-event UCB = `0.561506`，accepted family/strata 仅 `2/2`。
9. P6 true-delta compute v8 仍未过：best = `TBD0-V9256CBD0Reference`，AUC = `0.879072`，agreement = `1.0`，step ratio q90 = `2.863280 > 1.50`。
10. P7-P10 因 `P5_reference_deployable_frontier_failed` / `P4_support_confidence_failed` gate-blocked，全部以 `not_run` 落盘；当前 blocker = `support_stability_regression`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9263_conditional_bad_event_separation_risk_adjusted_safe_useful_frontier.py` | v9.2.63 runner；执行 v9.2.62 boundary 复现、conditional bad-event autopsy、conditional bad statistic factory、risk-adjusted safe-useful score、support confidence audit、exact-reference frontier v7、true-delta compute lane、downstream/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9263_conditional_bad_event_separation_risk_adjusted_safe_useful_frontier.py
```

正式运行：

```bash
python experiments/run_v9263_conditional_bad_event_separation_risk_adjusted_safe_useful_frontier.py \
  --out-dir results/real_rerun_20260506/v9263_conditional_bad_event_separation_risk_adjusted_safe_useful_frontier_first_20260512T180000Z \
  --fresh --device auto --data-root data --seed 1314
```

实际参数来自 `run_manifest.json`：

```text
device = cuda
triton_available = true
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7
microprobe_steps = 336
interface_events = 24
interface_steps = 2
train_size = 2048
batch_size = 64
hidden_dim = 256
completed_at = 2026-05-12T17:46:15Z
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R16-SupportRegression",
  "base_candidate": "LQ-t2-h256",
  "v9262_boundary_pass": 1,
  "conditional_bad_autopsy_pass": 1,
  "conditional_bad_attribution_fraction": 1.0,
  "risky_useful_attribution_fraction": 1.0,
  "harmless_null_attribution_fraction": 1.0,
  "best_conditional_bad_stat_id": "CB5-FamilyInstabilityUCB",
  "conditional_bad_stat_pass": 1,
  "conditional_bad_submode_diagnostic_pass": 1,
  "conditional_bad_utility_pass": 0,
  "conditional_bad_auc": 0.8105975789737436,
  "conditional_bad_corr": 0.4834497142706288,
  "conditional_bad_after_gate": 0.0,
  "coverage_after_conditional_bad_gate": 0.000496031746031746,
  "best_safe_useful_score_id": "SU2-SafeUsefulMargin",
  "safe_useful_score_pass": 1,
  "safe_useful_utility_pass": 0,
  "safe_useful_auc": 0.8648443443723428,
  "safe_useful_precision": 0.110397946084724,
  "safe_useful_coverage": 0.032200727513227514,
  "safe_useful_bad_event": 0.029525032092426188,
  "safe_useful_null_rate": 0.8472400513478819,
  "safe_useful_confidence_pass": 0,
  "precision_lcb": 0.09027354335902985,
  "bad_event_ucb": 0.04391430146938983,
  "support_confidence_pass": 0,
  "accepted_support_pass": 1,
  "natural_real_event_count": 24192,
  "balanced_diagnostic_real_event_count": 6000,
  "measured_signal_strata_count": 25,
  "measured_family_count": 538,
  "duplicate_row_count": 0,
  "accepted_signal_strata_count": 2,
  "accepted_family_count": 2,
  "exact_reference_deployable": 0,
  "reference_precision": 1.0,
  "reference_coverage": 0.0001240079365079365,
  "reference_bad_event": 0.0,
  "reference_null_rate": 0.0,
  "reference_precision_lcb": 0.43849391955098227,
  "reference_bad_event_ucb": 0.5615060804490177,
  "best_true_delta_id": "TBD0-V9256CBD0Reference",
  "true_delta_compute_pass": 0,
  "true_delta_auc": 0.8790720756550714,
  "true_delta_agreement": 1.0,
  "true_delta_step_ratio_q90": 2.8632798851361203,
  "oracle_support_pass": 1,
  "oracle_precision": 1.0,
  "oracle_coverage": 0.11997767857142858,
  "oracle_bad_event": 0.0,
  "primary_blocker": "support_stability_regression",
  "next_required_implementation": "repair_support_generator",
  "success_v9263_strict_purekan_functional": 0,
  "success_v9263_full_functional": 0,
  "success_v9263_external_ready": 0
}
```

判断：conditional bad-event 确实有 legal signal，但可用 clean gate 仍是 tiny slice；同时 P4 support confidence family count 回退，P5/P6 也没有打开 official downstream。

## 3. P1 conditional bad-event autopsy

Artifacts：

```text
p1_conditional_bad_event_autopsy.csv
conditional_bad_trace_v9263.csv
```

Summary：

```text
conditional_bad_count = 2309
risky_useful_count = 5358
harmless_null_leak_count = 9226
conditional_bad_event_attribution_fraction = 1.0
risky_useful_attribution_fraction = 1.0
harmless_null_attribution_fraction = 1.0
conditional_bad_autopsy_pass = 1
P_bad_given_useful_positive = 0.3125838632518523
P_bad_given_useful_null_rejected = 0.27102997614355356
P_bad_given_useful_null_support = 0.26464183381088824
P_safe_useful_given_useful_null_support = 0.29260744985673354
```

判断：v9.2.62 的 main failure 可以归因到 useful-positive / null-rejected / support-stable 条件下的 bad-event 混入；不是单纯 null contamination。

## 4. P2 conditional bad-event statistic factory

Artifacts：

```text
p2_conditional_bad_event_stat_factory.csv
conditional_bad_trace_v9263.csv
```

| bad statistic | submode | AUC conditional bad | corr | bad-event after gate | coverage | precision | pass |
|---|---|---:|---:|---:|---:|---:|---:|
| CB1-UsefulTailHarmUCB | `Y_B1_useful_tail_harm` | `0.737100` | `0.372801` | `0.000000` | `0.000000` | `0.000000` | `1` |
| CB2-UsefulControlFragility | `Y_B2_useful_control_fragile` | `0.268000` | `-0.307209` | `0.486471` | `0.070271` | `0.414118` | `0` |
| CB3-DeltaOverreachRisk | `Y_B3_useful_delta_overreach` | `0.303170` | `-0.215927` | `0.508516` | `0.067956` | `0.392336` | `0` |
| CB4-HorizonInconsistencyRisk | `Y_B5_useful_horizon_inconsistent` | `0.682504` | `0.245472` | `0.472621` | `0.063409` | `0.430900` | `0` |
| CB5-FamilyInstabilityUCB | `Y_B4_useful_family_unstable` | `0.810598` | `0.483450` | `0.000000` | `0.000496` | `0.750000` | `1` |
| CB6-ConditionalBadMixture | `bad_event` | `0.719186` | `0.324690` | `0.040816` | `0.002025` | `0.816327` | `1` |

Summary：

```text
best_conditional_bad_stat_id = CB5-FamilyInstabilityUCB
predictive_bad_submode_count = 5
conditional_bad_stat_pass = 1
conditional_bad_submode_diagnostic_pass = 1
conditional_bad_utility_pass = 0
```

判断：conditional bad-event 可预测，但一旦压住 bad-event，coverage 退回 `0.000496`，不能写成 deployable bad-event separation。

## 5. P3 risk-adjusted safe-useful score factory

Artifacts：

```text
p3_risk_adjusted_safe_useful_score_factory.csv
safe_useful_score_trace_v9263.csv
```

| safe-useful score | AUC | precision | coverage | bad-event | null rate | precision LCB | bad UCB | score pass | utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SU1-RiskAdjustedUsefulLCB | `0.646416` | `0.103679` | `0.037078` | `0.043478` | `0.841695` | `0.085390` | `0.058884` | `0` | `0` |
| SU2-SafeUsefulMargin | `0.864844` | `0.110398` | `0.032201` | `0.029525` | `0.847240` | `0.090274` | `0.043914` | `1` | `0` |
| SU3-PersistentRiskAdjusted | `0.624914` | `0.252583` | `0.080026` | `0.361054` | `0.384814` | `0.233731` | `0.382705` | `0` | `0` |
| SU4-EfficientSafeGain | `0.646571` | `0.072176` | `0.039517` | `0.032427` | `0.892259` | `0.057427` | `0.045659` | `0` | `0` |

判断：`SU2` 同时达到 coverage 和 bad-event point estimate，但 precision 极低、null rate 极高，说明 safe-useful score 仍没有从 harmless-null 中分离出 useful rows。

## 6. P4 support confidence and family stability

Artifacts：

```text
p4_support_confidence_family_stability_audit.csv
support_confidence_trace_v9263.csv
```

Summary：

```text
natural_real_event_count = 24192
balanced_diagnostic_real_event_count = 6000
measured_signal_strata_count = 25
measured_family_count = 538
duplicate_row_count = 0
support_confidence_pass = 0
accepted_signal_strata_count = 9
accepted_family_count = 79
max_family_share = 0.13525390625
max_stratum_share = 0.45361328125
accepted_support_pass = 1
```

判断：balanced rows、signal strata、duplicate gate 都过，但 measured family count 低于计划的 `>=700`，所以 v9.2.62 的 support stability 没有在 v9.2.63 中保持。

## 7. P5 exact reference deployable frontier v7

Artifacts：

```text
p5_exact_reference_deployable_frontier_v7.csv
reference_frontier_v7_trace.csv
```

Best summary：

```text
best_reference_controller_id = C-SU1-RiskAdjustedUsefulLCB+CB6-ConditionalBadMixture+N4-HybridNullProbabilityV2+S2-LeaveConditionFamilyOutReliability
best_safe_useful_score_id = SU1-RiskAdjustedUsefulLCB
best_conditional_bad_stat_id = CB6-ConditionalBadMixture
best_null_stat_id = N4-HybridNullProbabilityV2
best_support_stat_id = S2-LeaveConditionFamilyOutReliability
exact_reference_deployable = 0
precision = 1.0
coverage = 0.0001240079365079365
bad_event = 0.0
null_rate = 0.0
precision_lcb = 0.43849391955098227
bad_event_ucb = 0.5615060804490177
accepted_signal_strata_count = 2
accepted_family_count = 2
max_family_share = 0.6666666666666666
max_stratum_share = 0.6666666666666666
```

判断：best reference frontier 是 clean-too-tiny slice；coverage、LCB/UCB、family/strata 都不过，不能写成 exact-reference deployable controller。

## 8. P6 true-delta compute v8

Artifacts：

```text
p6_true_delta_compute_v8_parallel_lane.csv
true_delta_compute_v8_trace.csv
```

Summary：

```text
best_true_delta_id = TBD0-V9256CBD0Reference
true_delta_auc = 0.8790720756550714
true_delta_safe_useful_auc = 0.8477842093837368
true_delta_conditional_bad_auc = 0.8351009988610835
true_delta_agreement = 1.0
true_delta_step_ratio_q90 = 2.8632798851361203
true_delta_memory_ratio = 0.9695007261731864
true_delta_compute_pass = 0
```

判断：true-delta exact signal 仍存在，但 compute path 继续超过 system envelope；低成本 proxy/source-gap rows 没有被转正。

## 9. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p7_system_legal_exact_signal_controller.csv` | `P5_reference_deployable_frontier_failed` |
| `p8_leave_dataset_and_stratum_out.csv` | `P4_support_confidence_failed` |
| `p9_official_paired_replay.csv` | `P4_support_confidence_failed` |
| `p10_short_run_functional_validation.csv` | `P4_support_confidence_failed` |

没有把 conditional bad-event diagnostic、safe-useful point estimate、clean-too-tiny reference frontier、oracle support 或 true-delta reference signal 写成 system-legal controller / LDO / paired replay / short-run success。

## 10. No-fake audit

```text
rows_checked = 36324
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
conditional_bad_event_separation = 1
risk_adjusted_safe_useful_frontier = 1
support_confidence_family_stability = 1
exact_reference_deployable_frontier_v7 = 1
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
```

## 11. Hash

| artifact | SHA256 |
|---|---|
| plan | `bc480309ff54cc93a0a6185efa644f6a1b309e9901dc15efa8e5dd2e9bc34927` |
| runner | `1da240e5c4befa2a4bb382b084b8134c0df7aa155e6105a31e4cbc1800738668` |
| run manifest | `bf5a4b0b47e257b24079632b9a2fe98967746cc2033159f07221ae81333d31ae` |
| route | `4ee0c669d1cad174744057807fa49b796d3f1680aa0eee115b374b99694cc2ee` |
| P0 boundary | `9bbef40b2876656cd76e613d111627de89d1b562211e3a1611163f6f4c2cde85` |
| P1 autopsy | `b2e7ed915e97869dfcb24403fae3df2c446de8d5af838004f069e7e9b2e573f4` |
| P2 conditional bad | `c7fe3ed44de14776480a33cd776d965aac4051a6b7d171d8725c669fe370b61d` |
| P3 safe-useful | `89f9afb79430a79933661e9b84f2d8c0fe06c73e86828dd1f5a84d77bcbf93cb` |
| P4 support confidence | `4e8f29445b63638e6930f02e9c5947c60f4f777aed540cedc445bc935f8dfd34` |
| P5 deployable frontier | `1fcc74ef3d9e1ff359d6bb0ab217eaa8d4fa5f5f286f6c74b60cee74ce72c1fe` |
| P6 compute lane | `febecd2bd1066db984ecfffb8cac4d5d3cf32e4845a3aee037861272855b5b97` |
| P7 system controller | `b02e624c8b00d27ac4816397f267440b9a714156325cc52718e34d8bd7b49d33` |
| P8 boundary | `bc3e45b55d33c7121628830a74f1c60b637d629be916d379f4ff382ff8d6c7cf` |
| P9 boundary | `5c895a356237a56d29b03f207408c1745487d649d45cada6518d46090d82fadf` |
| P10 boundary | `84a0aefec5b8ed4d721b1c7623b60527dfd636018253d69f811475b6a298d81a` |
| failure table | `107000803ffdfa8a0008596cbf5f75803927eb78ab6429aa8ebb12b0b1f00e4d` |
| provenance audit | `abb1089e1b204565dd82e54f03da4b5e4bc536e915905f4e986519665486b781` |

## 12. 最终分析结论

v9.2.63 的真实推进是：

```text
v9.2.62: support stability pass，但 useful gate 带入大量 bad-event。
v9.2.63: conditional bad-event 有可测 legal signal，
          但 clean conditional bad gate 是 tiny slice；
          safe-useful score 的 point-estimate safety 由高 null / 低 precision 换来；
          support confidence family count 回退。
```

机制判断：

1. H1 成立：conditional bad-event autopsy 闭合，useful-positive / null-rejected / support-stable 区域里 bad-event 混入可归因。
2. H2 部分成立：conditional bad-event statistic 有强信号，`CB5` AUC `0.810598`，但 utility 失败，coverage 只有 `0.000496`。
3. H3 未闭合：risk-adjusted safe-useful score 能达到 coverage 和 bad-event point estimate，但 precision 只有 `0.110398`，null rate `0.847240`。
4. H4 出现回退：support accepted subset 过关，但 support confidence 的 measured family count 从 v9.2.62 的 pass 状态回落到 `538 < 700`。
5. H5 未闭合：exact-reference frontier 仍是 clean-too-tiny，coverage `0.000124`，LCB/UCB 不可部署。
6. H6 继续未闭合：true-delta compute 仍 `step_ratio_q90 = 2.863280`，超过 `1.50`。

最终一句话：

> v9.2.63 真实执行后停在 `R16-SupportRegression`：conditional bad-event 可以预测，但可用 clean slice 极小，safe-useful 分数仍被 null/low-precision 污染，同时 support confidence family coverage 回退，strict PureKAN functional 仍未成功。
