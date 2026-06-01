# DG-KAN v9.2.48 Real Train-Stream MicroProbe 与 Online Support-Region Controller 实验复盘

> 本复盘记录 `DG-KAN_v9.2.48_RealTrainStreamMicroProbe_OnlineSupportRegionController_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R8-MicroProbePredictiveButTooExpensive
base_candidate = LQ-t2-h256
success_v9248_strict_purekan_functional = False
success_v9248_full_functional = False
success_v9248_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9248_real_train_stream_microprobe_online_support_region_controller_first_20260512T000000Z/
```

核心结论：

1. P0 复现 v9.2.47 boundary：source route = `R7-OfflineFeatureOnly`，source blocker = `fresh_online_microprobe_not_implemented`。
2. P1 真实 online train-stream microprobe 已实现：rows = `2304`，update/probe/candidate hash 均存在，commit-time order pass = `1`。
3. P2 online feature 有预测性：best feature = `gap_probe`，safe-good AUC = `0.884406`，component all pass = `1`。
4. P2 system gate 失败：probe overhead q90 = `5.121495`，step ratio q90 = `6.121495`，因此 `online_microprobe_pass = 0`。
5. P2 signal-channel 未过：signal-channel AUC = `0.536473`，corr = `0.060158`，`signal_channel_pass = 0`。
6. P3 oracle support pass = `1`：oracle precision = `1.0`，coverage = `0.141059`，bad-event = `0.0`；但 natural support 覆盖只有 `2` 个 signal strata，`fresh_natural_support_pass = 0`。
7. P4 support-region controller pass = `0`；best = `C5-ParetoFrontOnlineController`，precision = `1.0`，coverage = `0.004340`，system gate = `0`。
8. 当前 blocker：`microprobe_predictive_but_system_overhead_failed`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9248_real_train_stream_microprobe_online_support_region_controller.py` | v9.2.48 runner；生成真实 train-stream update/probe rows、microprobe predictivity/system audit、natural oracle support、controller boundary、route/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9248_real_train_stream_microprobe_online_support_region_controller.py
```

正式运行：

```bash
python experiments/run_v9248_real_train_stream_microprobe_online_support_region_controller.py \
  --out-dir results/real_rerun_20260506/v9248_real_train_stream_microprobe_online_support_region_controller_first_20260512T000000Z \
  --fresh --device auto --data-root data --seed 1314
```

实际 microprobe 范围：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3
steps per seed/dataset = 64
carriers = A1,A2,A3
fresh_train_stream_update_probe_rows = 2304
```

## 2. Route

```json
{
  "route": "R8-MicroProbePredictiveButTooExpensive",
  "base_candidate": "LQ-t2-h256",
  "v9247_boundary_pass": 1,
  "online_microprobe_implemented": 1,
  "online_microprobe_pass": 0,
  "microprobe_auc": 0.8844062657908034,
  "microprobe_corr": -0.009828343547021239,
  "microprobe_overhead": 5.121495081599736,
  "signal_channel_pass": 0,
  "signal_channel_auc": 0.5364729661445174,
  "signal_channel_corr": 0.06015797452862053,
  "fresh_train_stream_update_probe_rows": 2304,
  "fresh_natural_row_count": 2304,
  "oracle_support_pass": 1,
  "oracle_precision": 1.0,
  "oracle_coverage": 0.1410590277777778,
  "oracle_bad_event": 0.0,
  "best_controller_id": "C5-ParetoFrontOnlineController",
  "support_region_controller_pass": 0,
  "controller_auc": 0.49945062023081527,
  "controller_corr": -0.02732600258866758,
  "accepted_precision": 1.0,
  "accepted_coverage": 0.004340277777777778,
  "accepted_bad_event_rate": 0.0,
  "primary_blocker": "microprobe_predictive_but_system_overhead_failed",
  "next_required_implementation": "amortize_or_kernelize_online_microprobe"
}
```

判断：v9.2.47 的 “online microprobe 未实现” 已被推进；本轮真实 online feature 能预测 safe-good，但系统开销远超 gate，不能进入 official support-region controller / LDO / paired replay。

## 3. P1 real online microprobe

Artifacts：

```text
p1_real_online_microprobe_implementation.csv
online_microprobe_trace_v9248.csv
commit_time_order_trace_v9248.csv
system_probe_overhead_trace_v9248.csv
```

| metric | value |
|---|---:|
| fresh train-stream update/probe rows | `2304` |
| online microprobe implemented | `1` |
| microprobe legality pass | `1` |
| commit-time order valid | `1` |
| update batch hash exists | `1` |
| probe batch hash exists | `1` |
| candidate update hash exists | `1` |
| uses dataset name | `0` |
| uses validation/test/posthoc | `0/0/0` |
| overhead q90 | `5.121495` |
| step ratio q90 | `6.121495` |
| memory ratio | `0.969501` |

Row distribution：

| slice | count |
|---|---:|
| MNIST | `768` |
| Fashion-MNIST | `768` |
| KMNIST | `768` |
| A1-RiskBoundedTailCarrier | `768` |
| A2-LateAttachControlGapChannel | `768` |
| A3-LateAttachRoleWiseFT7EdgeCarrier | `768` |

判断：P1 是本轮最大推进。它不是 source-measured event-time diagnostic，而是真实 train-stream batch split：`B_update` 产生 AdamW-equivalent task update 与 candidate functional update，`B_probe` 计算 pre-commit feature，随后才写 posthoc audit label。

## 4. P2 microprobe feature predictivity / signal-channel

Artifacts：

```text
p2_microprobe_feature_predictivity_signal_channel.csv
signal_channel_trace_v9248.csv
```

| feature | group | AUC safe-good | corr grounded | precision | coverage | bad-event |
|---|---|---:|---:|---:|---:|---:|
| risk_safe_score | risk | `0.463550` | `0.127002` | `0.173913` | `0.029948` | `0.130435` |
| value_probe | value | `0.693231` | `0.269192` | `0.376812` | `0.029948` | `0.289855` |
| gap_probe | gap | `0.884406` | `-0.009828` | `0.500000` | `0.039931` | `0.489130` |
| snr_probe | signal | `0.481245` | `-0.038137` | `0.144928` | `0.029948` | `0.594203` |
| signal_channel_ratio | signal | `0.536473` | `0.060158` | `0.347826` | `0.029948` | `0.507246` |
| support_density | support | `0.582714` | `-0.020085` | `0.173913` | `0.029948` | `0.565217` |
| family_reliability_pre | support | `0.650971` | `0.245588` | `0.228261` | `0.039931` | `0.423913` |

Summary：

```text
best_feature_id = gap_probe
online_feature_predictivity_pass = 1
risk_component_pass = 1
value_component_pass = 1
gap_component_pass = 1
component_all_pass = 1
signal_channel_pass = 0
microprobe_system_pass = 0
online_microprobe_pass = 0
```

判断：`gap_probe` 在 online train-stream rows 上确实有 safe-good predictivity；但 best accepted slice 的 bad-event 仍高，且 system gate 失败。SNR / signal-channel sketch 没有达到计划要求的 signal gate。

## 5. P3 fresh natural replay / oracle support

Artifact：

```text
p3_fresh_natural_replay_oracle_support.csv
```

| metric | value |
|---|---:|
| natural real events | `2304` |
| measured signal strata | `2` |
| carrier active | `1` |
| fresh natural support pass | `0` |
| oracle accepted count | `325` |
| oracle precision | `1.0` |
| oracle coverage | `0.141059` |
| oracle bad-event | `0.0` |
| oracle support pass | `1` |

Signal strata actually observed：

| stratum | rows |
|---|---:|
| S4-RoleWiseCurvature | `2132` |
| S1-CEHardTail | `172` |

判断：oracle safe-good support 仍存在，但 online microprobe distribution 明显集中在少数 signal strata。因为 terminal blocker 已经是 P2 system overhead，P5/P6 不打开；后续若做 kernelization，也需要同时修 stratum coverage。

## 6. P4 support-region controller

Artifact：

```text
p4_support_region_controller_calibration.csv
controller_calibration_trace_v9248.csv
```

| controller | official | AUC | corr | precision | coverage | bad-event | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1-ProbeRiskValueGapTriStage | `1` | `0.567528` | `-0.040187` | `0.400000` | `0.008681` | `0.600000` | `0` |
| C3-FamilyBalancedProbeController | `1` | `0.486531` | `0.094054` | `0.750000` | `0.013889` | `0.250000` | `0` |
| C5-ParetoFrontOnlineController | `1` | `0.499451` | `-0.027326` | `1.000000` | `0.004340` | `0.000000` | `0` |
| C7-Oracle | `0` | `1.000000` | `0.386562` | `1.000000` | `0.111979` | `0.000000` | `0` |

判断：C5 的 precision 是 `1.0`，但 coverage 只有 `0.004340`，低于 `[0.03,0.15]` gate；同时 P2 system gate 未过，所以 P4 不能成为 official controller success。`C7` 仍是 posthoc oracle，`official_eligible = 0`。

## 7. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p5_leave_dataset_and_stratum_out.csv` | `P2_microprobe_system_failed` |
| `p6_official_paired_replay.csv` | same |
| `p7_short_run_functional_validation.csv` | same |
| `p8_full_10seed_functional_validation.csv` | same |
| `p9_robustness_external_ready.csv` | same |

没有把 predictive microprobe、oracle support 或 diagnostic controller rows 倒灌成 LDO/LSO、paired replay、short-run、full-run 或 external-ready success。

## 8. No-fake audit

```text
rows_checked = 16168
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

## 9. Hash

| artifact | SHA256 |
|---|---|
| plan | `2dbddb93ec32042e484559696e1fb804c8d17e0922138edc1f4bede8458db545` |
| runner | `79b5e14453f0afd7aba579de6314734b5e1763cb59c9547db08885ab7b8c1032` |
| run manifest | `1d610c6c98245aaffa5f2f9c9b29bec2de9371e70ca85a40e2054cfd03f54630` |
| route | `3632ea1cf459203a1ca5cbbeed0b4578398c7c30ea75866fc7319fbd2162f447` |
| P0 boundary | `b0b1e3501145784f48d5edfb948bf05c42f207376d228359f6162e1c65576918` |
| P1 online microprobe | `8a8ec7b9b76bfd7eabd9635e654b0b1317e767a03c0be76fce2144b2f9263ac9` |
| P2 predictivity | `43a80540d65d8e8c0288019f0334c486e8bdfc944a1961c45ce7a8d4059b31da` |
| P3 oracle support | `3f3cdd31ce5291f335e6208218804b93ebf5c86cc31f1e6d58072fe38fb32590` |
| P4 controller | `02ef40dba387059d4ef5ea61a28948f9b4ff9658a79388ba01e08b69c8f8819e` |
| P5 boundary | `63a1decbc45a0dcb38a65d90ae07d7bac13f046556091f2b2a7e808c4262685e` |
| P6 boundary | `8298ce13f180dad2b5ed0d95a01f63558c8cc92aed26118914efde3605b5c18e` |
| failure table | `fee3a70e4aeb9b09a02aa37fe070e2f95432b647e6f786d9b2857cb0918241ad` |
| provenance audit | `7dc75229d9d33034596d6fc6f4089bc823b0cf728ab465a15bc0ab3bf0d95b18` |

## 10. 最终分析结论

v9.2.48 的真实推进是：

```text
v9.2.47: online microprobe 未实现，route 停在 OfflineFeatureOnly。
v9.2.48: 真实 train-stream update/probe microprobe 已实现；
          online gap_probe 可预测 safe-good；
          但 microprobe 系统开销远超 gate。
```

机制判断：

1. 本轮推翻了 “online feature 完全不可用” 的担忧：`gap_probe` 在真实 online train-stream rows 上达到 AUC `0.884406`。
2. 当前 blocker 从 feature representation 转成 system implementation：probe overhead q90 `5.121495`，step ratio q90 `6.121495`，明显不在 system envelope 内。
3. Signal-channel sketch 本身没有过 gate，说明这轮 predictive value 主要来自 expensive shadow control-gap probe，而不是便宜的 channel/SNR statistic。
4. P3 oracle support 没有崩，但 online rows 的 signal-stratum coverage 过窄；下一步做 amortization/kernelization 时也要扩展 support coverage。
5. P5-P9 保持关闭是正确的：predictive but too expensive 不能写成 strict PureKAN functional success。

最终一句话：

> v9.2.48 真实执行后停在 `R8-MicroProbePredictiveButTooExpensive`：真实 online microprobe 已实现且 `gap_probe` 可预测 safe-good，但系统开销过高，strict PureKAN functional 仍未成功。
