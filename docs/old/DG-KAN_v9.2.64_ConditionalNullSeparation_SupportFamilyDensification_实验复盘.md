# DG-KAN v9.2.64 Conditional Null Separation 与 Support-Family Densification 实验复盘

> 本复盘记录 `DG-KAN_v9.2.64_ConditionalNullSeparation_SupportFamilyDensification_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R13-SafeUsefulStillTiny
base_candidate = LQ-t2-h256
success_v9264_strict_purekan_functional = False
success_v9264_full_functional = False
success_v9264_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9264_conditional_null_separation_support_family_densification_first_20260512T190000Z/
```

核心结论：

1. P0 复现 v9.2.63 boundary：source route = `R16-SupportRegression`，conditional bad stat pass = `1`，support confidence pass = `0`，exact reference deployable = `0`，true-delta compute pass = `0`，fake/proxy/offload = `0`。
2. 本轮 manifest 记录 `device = cuda`，`triton_available = true`。
3. P1 conditional-null autopsy pass = `1`：SU2 count = `6527`，harmless-null count = `7955`，bad-event count = `2666`，safe-useful miss count = `982`，三类 attribution fraction 均为 `1.0`。
4. P2 conditional-null statistics 有强信号：best = `CN6-FamilyNullUCB`，conditional-null AUC = `0.893849`，corr = `0.714209`，null rate after gate = `0.020472`；但 precision = `0.472441`，bad-event = `0.374803`，utility/diagnostic pass = `0`。
5. P3 support-family densification 修复成功：natural rows = `24192`，balanced diagnostic rows = `6000`，measured signal strata = `318`，measured family count = `2730`，duplicate rows = `0`，support confidence pass = `1`。
6. P4 null-aware safe-useful score 仍未过：best = `NASU4-UsefulPocketScore`，AUC = `0.960014`，coverage = `0.069775`，null rate = `0.064771`；但 precision = `0.353870`，bad-event = `0.451817`，precision LCB = `0.317603`。
7. P5 exact-reference deployable frontier 未过：best = `C-NASU2-SupportBackedValue+CN7-ConditionalNullMixture+CB6-ConditionalBadMixture+SF4-NullAwareSupportPocket`，precision = `0.813953`，bad-event = `0.046512`，null rate = `0.139535`，但 coverage = `0.004740`，precision LCB = `0.673827`，bad-event UCB = `0.154558`。
8. P6 true-delta compute v9 仍未过：best = `TBD0-V9256CBD0Reference`，AUC = `0.879072`，agreement = `1.0`，step ratio q90 = `2.863280 > 1.50`。
9. P7-P10 因 `P5_reference_deployable_frontier_failed` gate-blocked，全部以 `not_run` 落盘。
10. 当前 blocker：`reference_still_tiny_after_null_repair`。

说明：正式结果来自上述 `T190000Z` artifact。运行中发现 runner 的 threshold frontier search 与 per-row quantile 计算存在 CPU-bound 性能问题，已修为等价预计算/向量化后重新正式运行；未落盘的中断尝试不计入本文结论。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9264_conditional_null_separation_support_family_densification.py` | v9.2.64 runner；执行 v9.2.63 boundary 复现、conditional-null autopsy、conditional-null statistic factory、support-family densification、null-aware safe-useful frontier、exact-reference frontier v8、true-delta compute lane、downstream/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9264_conditional_null_separation_support_family_densification.py
```

正式运行：

```bash
python experiments/run_v9264_conditional_null_separation_support_family_densification.py \
  --out-dir results/real_rerun_20260506/v9264_conditional_null_separation_support_family_densification_first_20260512T190000Z \
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
completed_at = 2026-05-12T19:27:16Z
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R13-SafeUsefulStillTiny",
  "base_candidate": "LQ-t2-h256",
  "v9263_boundary_pass": 1,
  "conditional_null_autopsy_pass": 1,
  "conditional_null_attribution_fraction": 1.0,
  "safe_useful_miss_attribution_fraction": 1.0,
  "bad_event_attribution_fraction": 1.0,
  "best_conditional_null_stat_id": "CN6-FamilyNullUCB",
  "conditional_null_stat_pass": 1,
  "conditional_null_submode_diagnostic_pass": 1,
  "conditional_null_utility_pass": 0,
  "conditional_null_auc": 0.8938489161325155,
  "conditional_null_corr": 0.7142090002609384,
  "null_rate_after_gate": 0.02047244094488189,
  "coverage_after_null_gate": 0.06999559082892416,
  "support_family_densification_pass": 1,
  "support_confidence_pass": 1,
  "natural_real_event_count": 24192,
  "balanced_diagnostic_real_event_count": 6000,
  "measured_signal_strata_count": 318,
  "measured_family_count": 2730,
  "duplicate_row_count": 0,
  "best_safe_useful_score_id": "NASU4-UsefulPocketScore",
  "null_aware_safe_useful_score_pass": 1,
  "null_aware_safe_useful_utility_pass": 0,
  "null_aware_safe_useful_confidence_pass": 0,
  "safe_useful_auc": 0.9600141277639995,
  "safe_useful_precision": 0.353870458135861,
  "safe_useful_coverage": 0.06977513227513228,
  "safe_useful_bad_event": 0.4518167456556082,
  "safe_useful_null_rate": 0.06477093206951026,
  "precision_lcb": 0.31760316609079076,
  "bad_event_ucb": 0.49076159926086477,
  "exact_reference_deployable": 0,
  "reference_precision": 0.813953488372093,
  "reference_coverage": 0.004739858906525573,
  "reference_bad_event": 0.046511627906976744,
  "reference_null_rate": 0.13953488372093023,
  "reference_precision_lcb": 0.6738270676469403,
  "reference_bad_event_ucb": 0.15455775923863058,
  "true_delta_compute_pass": 0,
  "true_delta_auc": 0.8790720756550714,
  "true_delta_agreement": 1.0,
  "true_delta_step_ratio_q90": 2.8632798851361203,
  "oracle_support_pass": 1,
  "oracle_precision": 1.0,
  "oracle_coverage": 0.11997767857142858,
  "oracle_bad_event": 0.0,
  "primary_blocker": "reference_still_tiny_after_null_repair"
}
```

判断：v9.2.64 修复了 v9.2.63 的 support regression，但 conditional-null repair 后 exact-reference frontier 仍是 tiny slice；safe-useful score 的 high AUC 没有转化成可部署 precision/bad-event/coverage gate。

## 3. P1 conditional-null autopsy

Artifacts：

```text
p1_conditional_null_autopsy.csv
conditional_null_trace_v9264.csv
```

Summary：

```text
su2_count = 6527
low_bad_support_count = 12586
harmless_null_count = 7955
bad_event_count = 2666
safe_useful_miss_count = 982
nonnull_null_submode_count = 6
conditional_null_attribution_fraction = 1.0
safe_useful_miss_attribution_fraction = 1.0
bad_event_attribution_fraction = 1.0
conditional_null_autopsy_pass = 1
P_null_given_SU2 = 0.6266278535314846
P_safe_useful_given_SU2 = 0.24957867320361574
P_bad_given_SU2 = 0.12379347326489964
P_null_given_low_bad_support_stable = 0.6320514857778484
P_safe_useful_given_low_bad_support_stable = 0.12942952486890197
```

判断：v9.2.63 的 low-precision / high-null 问题可以归因，且六个 null submode 都有非零覆盖；但 SU2-like region 中 null 比例仍高。

## 4. P2 conditional-null statistic factory

Artifacts：

```text
p2_conditional_null_stat_factory.csv
null_submode_trace_v9264.csv
```

| null stat | AUC conditional null | corr | null rate | precision | coverage | bad-event | stat pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| CN1-ControlEquivalentNull | `0.702892` | `0.410402` | `0.038270` | `0.464226` | `0.066248` | `0.410982` | `1` |
| CN2-LowRealGainNull | `0.617625` | `0.270941` | `0.100946` | `0.548896` | `0.069885` | `0.287066` | `0` |
| CN3-NonPersistentNull | `0.689839` | `0.368803` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `1` |
| CN4-DeltaSilentNull | `0.817849` | `0.531645` | `0.111635` | `0.511006` | `0.070106` | `0.314465` | `1` |
| CN5-SupportOnlyNull | `0.500000` | `0.000000` | `0.148936` | `0.462486` | `0.098435` | `0.293393` | `0` |
| CN6-FamilyNullUCB | `0.893849` | `0.714209` | `0.020472` | `0.472441` | `0.069996` | `0.374803` | `1` |
| CN7-ConditionalNullMixture | `0.889282` | `0.638306` | `0.053883` | `0.575277` | `0.069555` | `0.307448` | `1` |

Summary：

```text
best_conditional_null_stat_id = CN6-FamilyNullUCB
predictive_null_submode_count = 5
conditional_null_stat_pass = 1
conditional_null_submode_diagnostic_pass = 1
conditional_null_utility_pass = 0
conditional_null_diagnostic_pass = 0
```

判断：conditional-null 是可预测的，`CN6` 能把 null rate 压到 `0.020472`；但它仍把大量 bad-event 带入 accepted region，不能 official。

## 5. P3 support-family densification

Artifacts：

```text
p3_support_family_densification_confidence_recovery.csv
support_family_densification_trace_v9264.csv
```

Summary：

```text
natural_real_event_count = 24192
balanced_diagnostic_real_event_count = 6000
measured_signal_strata_count = 318
measured_family_count = 2730
duplicate_row_count = 0
best_support_family_stat_id = SF4-NullAwareSupportPocket
support_family_densification_pass = 1
support_confidence_pass = 1
accepted_support_pass = 1
accepted_signal_strata_count = 23
accepted_family_count = 127
max_family_share = 0.25063163213744316
max_stratum_share = 0.3117736230419404
```

判断：v9.2.63 的 support family count regression 已修复。rows、balanced rows、signal strata、family count、duplicate 和 accepted-support balance 都过 gate。

## 6. P4 null-aware safe-useful score

Artifacts：

```text
p4_null_aware_safe_useful_score_factory.csv
null_aware_safe_useful_score_trace_v9264.csv
```

| score | AUC safe-useful | precision | coverage | bad-event | null rate | precision LCB | bad UCB | utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NASU1-NullAwareSafeUsefulLCB | `0.955165` | `0.517665` | `0.071759` | `0.374808` | `0.029186` | `0.479289` | `0.412626` | `0` |
| NASU2-SupportBackedValue | `0.946505` | `0.750000` | `0.002205` | `0.100000` | `0.150000` | `0.531295` | `0.301038` | `0` |
| NASU3-FourClassMargin | `0.922191` | `0.755245` | `0.015763` | `0.153846` | `0.090909` | `0.678706` | `0.221958` | `0` |
| NASU4-UsefulPocketScore | `0.960014` | `0.353870` | `0.069775` | `0.451817` | `0.064771` | `0.317603` | `0.490762` | `0` |

Summary：

```text
best_safe_useful_score_id = NASU4-UsefulPocketScore
null_aware_safe_useful_score_pass = 1
null_aware_safe_useful_utility_pass = 0
null_aware_safe_useful_confidence_pass = 0
```

判断：null-aware scores 继续有 ranking signal，但不是 deployable safe-useful frontier。能扩大 coverage 的 score precision/bad-event 很差；更精确的 score coverage 又过小且 UCB 不安全。

## 7. P5 exact-reference deployable frontier v8

Artifacts：

```text
p5_exact_reference_deployable_frontier_v8.csv
reference_frontier_v8_trace.csv
```

Best summary：

```text
best_reference_controller_id = C-NASU2-SupportBackedValue+CN7-ConditionalNullMixture+CB6-ConditionalBadMixture+SF4-NullAwareSupportPocket
best_safe_useful_score_id = NASU2-SupportBackedValue
best_conditional_null_stat_id = CN7-ConditionalNullMixture
best_conditional_bad_stat_id = CB6-ConditionalBadMixture
best_support_stat_id = SF4-NullAwareSupportPocket
exact_reference_deployable = 0
precision = 0.813953488372093
coverage = 0.004739858906525573
bad_event = 0.046511627906976744
null_rate = 0.13953488372093023
precision_lcb = 0.6738270676469403
bad_event_ucb = 0.15455775923863058
accepted_signal_strata_count = 6
accepted_family_count = 18
max_family_share = 0.16279069767441862
max_stratum_share = 0.37209302325581395
```

判断：best reference frontier point estimate 看起来接近安全，但 coverage 只有 `0.004740`，precision LCB 低于 `0.75`，bad-event UCB 高于 `0.05`，accepted family 也只有 `18 < 32`。不能写成 deployable controller。

## 8. P6 true-delta compute v9

Artifacts：

```text
p6_true_delta_compute_v9_parallel_lane.csv
true_delta_compute_v9_trace.csv
```

Summary：

```text
best_true_delta_id = TBD0-V9256CBD0Reference
true_delta_auc = 0.8790720756550714
true_delta_safe_useful_auc = 0.7984599264206053
true_delta_conditional_bad_auc = 0.526522839906146
true_delta_conditional_null_auc = 0.5002098952086693
true_delta_agreement = 1.0
true_delta_step_ratio_q90 = 2.8632798851361203
true_delta_memory_ratio = 0.9695007261731864
true_delta_compute_pass = 0
true_delta_compute_diagnostic_pass = 0
```

判断：exact/true delta signal 仍存在，但 compute path 继续超过 system envelope；低成本 formula/source-gap rows 没有被转正。

## 9. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p7_system_legal_exact_signal_controller.csv` | `P5_reference_deployable_frontier_failed` |
| `p8_leave_dataset_and_stratum_out.csv` | `P5_reference_deployable_frontier_failed` |
| `p9_official_paired_replay.csv` | `P5_reference_deployable_frontier_failed` |
| `p10_short_run_functional_validation.csv` | `P5_reference_deployable_frontier_failed` |

没有把 conditional-null predictivity、support densification pass、tiny exact-reference frontier、oracle support 或 true-delta reference signal 写成 system-legal controller / LDO / paired replay / short-run success。

## 10. No-fake audit

```text
rows_checked = 60567
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
conditional_null_separation = 1
support_family_densification = 1
null_aware_safe_useful_frontier = 1
exact_reference_deployable_frontier_v8 = 1
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
```

## 11. Hash

| artifact | SHA256 |
|---|---|
| plan | `bf27a87d3be4921df37be6fb12a0e13d3c7124acb826112a23117bbdc721a8ce` |
| runner | `4f55341c321543617743b12a0a588f55864ff576e75075cb2ae62a19d0fef6fe` |
| run manifest | `cc516e3e7b1caadfa9fa8ac8b9d4a30a7517dd4a72f3d290e0ac4e2dfd07da17` |
| route | `2a27512a0dd0086e861f425a0b95197822e2c1c2ab2e42cf35b7f360910054b3` |
| P0 boundary | `bd80fa6220ec87510acf9547818bc447301584fd420542969437e5e7358127dd` |
| P1 conditional null | `94c6fc8fce77f126db89a641ed1d51f57e0701de0737f8501172a4ee05df6961` |
| P2 null statistics | `cea5a88ca17f82e82f93fbbe39e576eb1d15b8feb2aab706fa67f091190d04f4` |
| P3 support densification | `70b9429b14d8c4798df08cfdb1322d360926b8f8536e4d5e8ec478f3ae9b6468` |
| P4 safe-useful score | `4b3cac1ac2b91f00b2d8adef99771cdc91851cb6108ea928e0cb8b0fc2fb5c42` |
| P5 deployable frontier | `c58fe4f76e3be471b6268792b5452217300f55b67592f0f6fe3426f83ffceedb` |
| P6 compute lane | `ff3f06ebadd90c7dc4e1d45622b8ccda8523077acd5f371e1b43c381dc4ad5f6` |
| P7 system controller | `b02e624c8b00d27ac4816397f267440b9a714156325cc52718e34d8bd7b49d33` |
| P8 boundary | `7355b8a5d018f3e72aa74bee422ef5ab0bebff3bedc74bb85d60fed775985806` |
| P9 boundary | `ff68aa604b9f21f3a0867d744a0d0f15ec619098c33b0a4b9d8c94fd46737682` |
| P10 boundary | `fa51df76d30203df3a88bcb76431ad22929ca9c726f804d1cd5de1ca077eeac8` |
| failure table | `3a7e74d38801d7e0450ce9ea9e8b2cbbd941ce84cdfdc71f264716b9a24f5ec0` |
| provenance audit | `d18e85a08d242200789784ae22c3fb289e6915d06d630e1d6642fc13c9400e9a` |
| contract audit | `86e4bd8fe312faccb73a240c4709a2a54425a7f3afff6dad8a990668d9aa4e03` |

## 12. 最终分析结论

v9.2.64 的真实推进是：

```text
v9.2.63: conditional bad-event 可预测，但 support confidence family coverage 回退。
v9.2.64: support-family densification 成功修复 support regression；
          conditional-null statistics 也有强信号；
          但 null-aware safe-useful frontier 仍不能转成 deployable exact-reference controller。
```

机制判断：

1. H1 成立：conditional-null autopsy 闭合，六个 null submode 都有覆盖。
2. H2 部分成立：`CN6-FamilyNullUCB` 有强 conditional-null signal，AUC `0.893849`，但 gate 后 precision/bad-event 不合格。
3. H3 成立：support-family densification 明确恢复并大幅超过 family/strata gate，duplicate = `0`。
4. H4 未闭合：null-aware safe-useful score AUC 很高，但 best score precision 只有 `0.353870`，bad-event `0.451817`。
5. H5 未闭合：exact-reference deployable frontier 仍是 tiny slice，coverage `0.004740`，LCB/UCB 不可部署。
6. H6 继续未闭合：true-delta compute 仍 `step_ratio_q90 = 2.863280`，超过 `1.50`。

最终一句话：

> v9.2.64 真实执行后停在 `R13-SafeUsefulStillTiny`：support-family densification 修复成功，conditional-null 有强信号，但 safe-useful / exact-reference frontier 仍只能得到 tiny 或 unsafe region，strict PureKAN functional 仍未成功。
