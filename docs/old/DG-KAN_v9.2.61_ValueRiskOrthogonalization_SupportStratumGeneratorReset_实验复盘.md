# DG-KAN v9.2.61 Value-Risk Orthogonalization 与 Support-Stratum Generator Reset 实验复盘

> 本复盘记录 `DG-KAN_v9.2.61_ValueRiskOrthogonalization_SupportStratumGeneratorReset_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R13-ValueRiskOrthogonalizationFail
base_candidate = LQ-t2-h256
success_v9261_strict_purekan_functional = False
success_v9261_full_functional = False
success_v9261_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9261_value_risk_orthogonalization_support_stratum_generator_reset_first_20260512T153000Z/
```

核心结论：

1. P0 复现 v9.2.60 boundary：source route = `R12-CleanCoreNotExpandable`，clean-core/near-core 均未过，oracle support pass = `1`，fake/proxy/offload = `0`。
2. 本轮 manifest 记录 `device = cuda`，`triton_available = true`；正式运行中 GPU 采样有活动：`NVIDIA L4, 31 %, 326 MiB`。
3. P1 three-class decomposition pass = `1`：accepted count = `494`，safe-good accepted = `368`，harmless-null accepted = `122`，bad-event accepted = `4`，useful missed = `5389`。
4. P2 support-stratum generator reset pass = `1`：natural rows = `24192`，balanced diagnostic rows = `6000`，signal strata = `16`，families = `829`，duplicate rows = `0`。
5. P3 value/control statistic 有强预测性：best non-reference = `VAL3-NoRegretControlMargin`，useful AUC = `0.998329`，value_stat_pass = `1`；但 value gate coverage = `0.0078125`，utility pass = `0`。
6. P4 risk-null aware statistic 未过：best = `RNULL5-ThreeClassSoftmaxMargin`，bad-event AUC = `0.749987`，但 null AUC = `0.471273`，coverage = `0.009673`，`risk_null_stat_pass = 0`。
7. P5 support frontier 有 diagnostic 推进：best = `SUP3-ValueRiskKNNPocket`，precision = `0.841398`，bad-event = `0.024194`，strata = `3`，但 coverage = `0.015377 < 0.03`，official pass = `0`。
8. P6 exact reference deployable frontier 未过：best = `C-VAL2-ControlGapLCB+RNULL3-RiskValueJointBudget+SUP3-ValueRiskKNNPocket`，precision = `0.750600`，bad-event = `0.045564`，但 coverage = `0.017237`，precision LCB = `0.706911`，bad-event UCB = `0.070063`。
9. P7 true-delta compute v6 仍未过：best = `TBD0-V9256CBD0Reference`，AUC = `0.879072`，agreement = `1.0`，step ratio = `2.863280 > 1.50`。
10. P8-P11 因 P6/P3 gate-blocked，全部以 `not_run` 落盘。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9261_value_risk_orthogonalization_support_stratum_generator_reset.py` | v9.2.61 runner；执行 v9.2.60 boundary 复现、three-class decomposition、support-stratum generator reset、value/control statistic factory、risk-null aware statistic factory、support frontier v2、exact-reference deployable frontier v5、true-delta compute lane、downstream/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9261_value_risk_orthogonalization_support_stratum_generator_reset.py
```

正式运行：

```bash
python experiments/run_v9261_value_risk_orthogonalization_support_stratum_generator_reset.py \
  --out-dir results/real_rerun_20260506/v9261_value_risk_orthogonalization_support_stratum_generator_reset_first_20260512T153000Z \
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
completed_at = 2026-05-12T15:13:21Z
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R13-ValueRiskOrthogonalizationFail",
  "base_candidate": "LQ-t2-h256",
  "v9260_boundary_pass": 1,
  "three_class_decomposition_pass": 1,
  "support_stratum_generator_pass": 1,
  "natural_real_event_count": 24192,
  "balanced_diagnostic_real_event_count": 6000,
  "measured_signal_strata_count": 16,
  "measured_family_count": 829,
  "duplicate_row_count": 0,
  "best_value_stat_id": "VAL3-NoRegretControlMargin",
  "value_stat_pass": 1,
  "value_auc_useful": 0.9983290748814346,
  "value_precision_after_gate": 0.7566137566137566,
  "value_coverage_after_gate": 0.0078125,
  "best_risk_null_stat_id": "RNULL5-ThreeClassSoftmaxMargin",
  "risk_null_stat_pass": 0,
  "bad_auc": 0.7499873610012162,
  "null_auc": 0.4712729547640464,
  "best_support_frontier_id": "SUP3-ValueRiskKNNPocket",
  "support_frontier_pass": 0,
  "accepted_signal_strata_count": 4,
  "accepted_family_count": 78,
  "exact_reference_deployable": 0,
  "reference_precision": 0.750599520383693,
  "reference_coverage": 0.017237103174603176,
  "reference_bad_event": 0.045563549160671464,
  "reference_precision_lcb": 0.7069107016177334,
  "reference_bad_event_ucb": 0.0700631899382484,
  "true_delta_compute_pass": 0,
  "true_delta_auc": 0.8790720756550714,
  "true_delta_agreement": 1.0,
  "true_delta_step_ratio_q90": 2.8632798851361203,
  "oracle_support_pass": 1,
  "oracle_precision": 1.0,
  "oracle_coverage": 0.11997767857142858,
  "oracle_bad_event": 0.0,
  "primary_blocker": "value_control_statistic_failed",
  "next_required_implementation": "reset_value_control_target_decomposition"
}
```

判断：v9.2.61 确实推进了 support measurement 和 value/risk/support 的分解，但 value/control gate 没有形成可部署 coverage，risk-null 分离也没有闭合，因此 route 不能打开 system-legal controller 或 downstream。

## 3. P1 three-class accepted row decomposition

Artifacts：

```text
p1_three_class_accepted_row_decomposition.csv
three_class_trace_v9261.csv
```

Summary：

```text
accepted_count = 494
safe_good_accepted = 368
harmless_null_accepted = 122
bad_event_accepted = 4
useful_missed = 5389
accepted_row_decomposition_fraction = 1.0
bad_event_mode_attribution_fraction = 1.0
harmless_null_attribution_fraction = 1.0
value_miss_attribution_fraction = 1.0
three_class_decomposition_pass = 1
```

RB3 reference slice：

```text
rb3_precision = 0.7566137566137566
rb3_coverage = 0.0078125
rb3_bad_event = 0.0
rb3_null_rate = 0.24338624338624337
```

判断：v9.2.60 的 accept-region 失败不只是 bad-event 问题，也有 harmless-null 混入和 useful rows missed。三类归因闭合，但 reference slice coverage 仍太小。

## 4. P2 support-stratum generator reset

Artifacts：

```text
p2_support_stratum_generator_reset.csv
support_stratum_generator_trace_v9261.csv
```

Summary：

```text
natural_real_event_count = 24192
balanced_diagnostic_real_event_count = 6000
measured_signal_strata_count = 16
measured_family_count = 829
duplicate_row_count = 0
support_stratum_generator_pass = 1
```

判断：这是本轮明确推进点。v9.2.60 卡住的 balanced diagnostic rows / signal strata 在 v9.2.61 中达到计划 gate，且没有复制补造 rows。

## 5. P3 value/control statistics factory

Artifacts：

```text
p3_value_control_statistics_factory.csv
value_control_trace_v9261.csv
```

| value stat | reference only | useful AUC | corr useful | precision | coverage | bad-event | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| VAL0-CurrentExactGapReference | `1` | `0.881163` | `0.261080` | `0.756614` | `0.007812` | `0.000000` | `1` |
| VAL2-ControlGapLCB | `0` | `0.872492` | `0.443056` | `0.756614` | `0.007812` | `0.000000` | `1` |
| VAL3-NoRegretControlMargin | `0` | `0.998329` | `0.256669` | `0.756614` | `0.007812` | `0.000000` | `1` |
| VAL4-ValueConsistencyAcrossHorizon | `0` | `0.876550` | `0.351920` | `0.756614` | `0.007812` | `0.000000` | `1` |
| VAL6-HybridValueControlLCB | `0` | `0.906444` | `0.519399` | `0.756614` | `0.007812` | `0.000000` | `1` |

Summary：

```text
best_value_stat_id = VAL3-NoRegretControlMargin
value_stat_pass = 1
value_gate_utility_pass = 0
value_diagnostic_pass = 0
value_auc_useful = 0.9983290748814346
value_precision_after_gate = 0.7566137566137566
value_coverage_after_gate = 0.0078125
value_bad_event_after_gate = 0.0
value_null_rate_after_gate = 0.24338624338624337
```

判断：`VAL3` 对 useful rows 有强信号；但 gate 后 coverage 只有 `0.0078125`，且 null 混入仍高，不能写成 deployable value-control success。`VAL0` 是 reference-only，没有被写成 route-level best。

## 6. P4 risk-null aware statistics factory

Artifacts：

```text
p4_risk_null_aware_statistics_factory.csv
risk_null_trace_v9261.csv
```

| risk-null stat | bad AUC | null AUC | precision | coverage | bad-event | null rate | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| RNULL3-RiskValueJointBudget | `0.751769` | `0.461055` | `0.792271` | `0.008557` | `0.000000` | `0.207729` | `0` |
| RNULL5-ThreeClassSoftmaxMargin | `0.749987` | `0.471273` | `0.794872` | `0.009673` | `0.008547` | `0.196581` | `0` |

Summary：

```text
best_risk_null_stat_id = RNULL5-ThreeClassSoftmaxMargin
risk_null_stat_pass = 0
bad_auc = 0.7499873610012162
null_auc = 0.4712729547640464
precision_after_gate = 0.7948717948717948
coverage_after_gate = 0.009672619047619048
bad_event_after_gate = 0.008547008547008548
null_rate_after_gate = 0.19658119658119658
```

判断：bad-event 可预测，但 harmless-null 没有被有效分离；coverage 也低于 `0.03`，因此 P4 不允许转正。

## 7. P5 support frontier v2

Artifacts：

```text
p5_support_stratum_frontier_v2.csv
support_frontier_v2_trace_v9261.csv
```

Best summary：

```text
best_support_frontier_id = SUP3-ValueRiskKNNPocket
support_frontier_pass = 0
support_frontier_diagnostic_pass = 1
precision = 0.8413978494623656
coverage = 0.015376984126984126
bad_event = 0.024193548387096774
null_rate = 0.13440860215053763
legal_oracle_jaccard = 0.2501998401278977
accepted_signal_strata_count = 3
accepted_family_count = 76
max_family_share = 0.13978494623655913
max_stratum_share = 0.6155913978494624
```

判断：P5 比 v9.2.60 的 support frontier 有推进，至少 diagnostic 过了；但 coverage 仍低于 official `0.03`，不能打开 exact-reference deployable controller。

## 8. P6 exact reference deployable frontier v5

Artifacts：

```text
p6_exact_reference_deployable_frontier_v5.csv
reference_deployable_frontier_v5_trace.csv
```

Best summary：

```text
best_reference_controller_id = C-VAL2-ControlGapLCB+RNULL3-RiskValueJointBudget+SUP3-ValueRiskKNNPocket
best_value_stat_id = VAL2-ControlGapLCB
best_risk_stat_id = RNULL3-RiskValueJointBudget
best_support_stat_id = SUP3-ValueRiskKNNPocket
exact_reference_deployable = 0
precision = 0.750599520383693
coverage = 0.017237103174603176
bad_event = 0.045563549160671464
null_rate = 0.2038369304556355
precision_lcb = 0.7069107016177334
bad_event_ucb = 0.0700631899382484
accepted_signal_strata_count = 4
accepted_family_count = 78
max_family_share = 0.13908872901678657
max_stratum_share = 0.60431654676259
legal_oracle_jaccard = 0.24151234567901234
```

判断：point estimate 接近 official safety，但 coverage、precision LCB、bad-event UCB 都不过。这里不能把 borderline exact-reference frontier 写成 deployable controller。

## 9. P7 true-delta compute v6

Artifacts：

```text
p7_true_delta_compute_v6_parallel_lane.csv
true_delta_compute_v6_trace.csv
```

Summary：

```text
best_true_delta_id = TBD0-V9256CBD0Reference
true_delta_auc = 0.8790720756550714
true_delta_agreement = 1.0
true_delta_step_ratio_q90 = 2.8632798851361203
true_delta_memory_ratio = 0.9695007261731864
true_delta_compute_pass = 0
```

判断：true-delta exact signal 仍存在，但 compute path 继续超过 system gate。即使 P6 过了，P7 也还会阻止 official controller。

## 10. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p8_system_legal_exact_signal_controller.csv` | `P6_reference_deployable_frontier_failed` |
| `p9_leave_dataset_and_stratum_out.csv` | `P3_value_gate_failed` |
| `p10_official_paired_replay.csv` | `P3_value_gate_failed` |
| `p11_short_run_functional_validation.csv` | `P3_value_gate_failed` |

没有把 value/control predictivity、support-frontier diagnostic、exact-reference borderline frontier、oracle support 或 true-delta reference signal 写成 system-legal controller / LDO / paired replay / short-run success。

## 11. No-fake audit

```text
rows_checked = 108969
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
value_risk_orthogonalization = 1
support_stratum_generator_reset = 1
support_frontier_v2 = 1
exact_reference_deployable_frontier_v5 = 1
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
```

## 12. Hash

| artifact | SHA256 |
|---|---|
| plan | `1abddc50fe8a657cb4a53f8e2907920b019b68518f3387d5e393083e7b864769` |
| runner | `e602b9a2a7999bd37aab6125b5431b29ebf23cf0d4a7142f79fa1d9648b5f92f` |
| run manifest | `ec94784090c8f0fe4f170bfe6e3ae2e279e31f08bdd7c510fa74e2d332aedae5` |
| route | `36d05e03c84941607d8117b26883948f8ff08eff0f1b92e7b5cea2961118f0ce` |
| P0 boundary | `e31d852f202ceb2cc86e4bf35df93f88067f585ad530cacacca9ca4371d280a5` |
| P1 decomposition | `b0d8ecd750073481868a31fcc7c92e85575adbe9cde359de23914db009f10132` |
| P2 support generator | `452fedbb13c2b4c863cdc55d865f283e1e0048845199e208809959fd106963ce` |
| P3 value/control | `e86f5070ee167effc7c4f957fbdd9795ea106b2341bd600b22aebac3fc65793b` |
| P4 risk-null | `a3da075f4c03b491fb4e8e5b344f8994b1301dc9e094e181cdcc4e7a7b63bf49` |
| P5 support frontier | `5c889961d6dee96c167519bd363d3b58e04ff18c513cbcbba27734c6fcbab442` |
| P6 deployable frontier | `a2d1aa9b1eea0e0216ea7b0e6ace18dc5f84095aaf2c686791128165fa00c37f` |
| P7 compute lane | `f80de2669c8e3eb1e2872f4aa493f97f544eeaa2f244b0b695719071b00c7bee` |
| P8 system controller | `14f379f1bd0974086e343a1ec159e94b2421a74222a938803966c228379cfdd2` |
| P9 boundary | `3477bb9be60931991d6e6b1e9d028d47d28480c785a01ccb49783d487387f95a` |
| P10 boundary | `898a07819d622a38bd0d73d53e3a665c557a1d5bfb16aa2bf109497bd366417b` |
| P11 boundary | `20e06302c64fa7ceb088cc89cb4ed8cdc25465ad455b91e69e3dd0c27fdf3db7` |
| failure table | `0a7e8ca9ae9a19ec164d3a0dbb2ed0fefdc70dbe0cbf324f7011914cf1e08d46` |
| provenance audit | `94c23d2ecd3c9f292dd892e576157e112e143637669302a6b2ae189e612d8785` |

## 13. 最终分析结论

v9.2.61 的真实推进是：

```text
v9.2.60: clean core 不稳定，support measurement 仍不足。
v9.2.61: support-stratum generator reset 过 gate；
          value/control 与 support frontier 均出现诊断推进；
          但 value gate 和 exact-reference deployable frontier 仍未闭合。
```

机制判断：

1. H1 成立：three-class decomposition 闭合，accepted rows 中 safe-good、harmless-null、bad-event、missed useful 的结构可解释。
2. H2 成立：support-stratum generator reset 终于达到 rows/strata/family/balanced gate，且 duplicate = `0`。
3. H3 只部分成立：value/control statistics 有强 useful signal，但 coverage 仍只有 `0.0078125`，不能形成 utility gate。
4. H4 未成立：risk-null aware statistic 能压 bad-event，但 null AUC 不足，harmless-null 分离失败。
5. H5 只达到 diagnostic：support frontier v2 的 precision/bad-event/strata 有推进，但 coverage 不够。
6. H6 未闭合：exact-reference deployable frontier 的 point estimate 接近安全，但 LCB/UCB 和 coverage gate 失败。
7. H7 继续未闭合：true-delta compute 仍 `step_ratio_q90 = 2.863280`，超过 `1.50`。

最终一句话：

> v9.2.61 真实执行后停在 `R13-ValueRiskOrthogonalizationFail`：support-stratum generator reset 成功，value/control 与 support frontier 有诊断信号，但 value gate coverage、risk-null 分离和 exact-reference deployable frontier 均未闭合，strict PureKAN functional 仍未成功。
