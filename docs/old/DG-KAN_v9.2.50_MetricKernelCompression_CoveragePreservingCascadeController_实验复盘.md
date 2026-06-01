# DG-KAN v9.2.50 Metric Kernel Compression 与 Coverage-Preserving Cascade Controller 实验复盘

> 本复盘记录 `DG-KAN_v9.2.50_MetricKernelCompression_CoveragePreservingCascadeController_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R10-PredictiveButMetricKernelTooExpensive
base_candidate = LQ-t2-h256
success_v9250_strict_purekan_functional = False
success_v9250_full_functional = False
success_v9250_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9250_metric_kernel_compression_coverage_preserving_cascade_controller_first_20260512T020000Z/
```

核心结论：

1. P0 复现 v9.2.49 boundary：source route = `R5-SignalChannelSketchPass`，fake/proxy = `0`。
2. 本轮 manifest 记录 `device = cuda`；正式运行使用 fresh online rows，未使用 CPU offload。
3. P1 F7 metric subphase attribution pass = `1`，dominant subphase = `M0-logit preparation`，ratio = `0.251454`。
4. P2 metric-kernel compression 未过：best = `MK4-PreallocatedWorkspaceExact`，metric error = `0.0`，controller agreement = `0.801759`，但 step ratio = `5.072608`。
5. P3 recall-first event-sparse cascade 未过：best route summary recall = `0.557284`，coverage = `0.057407`，precision = `0.588710`，bad-event = `0.356452`。
6. P4 cheap surrogate prefilter 未过：best summary = `CS3-RoleBranchSignal`，AUC = `0.543916`，recall = `0.192362`；最高 AUC 也只有 `0.594640`。
7. P5 signal auxiliary 未过：best = `SA3-TrustRatio`，AUC = `0.573312`，bad-event = `0.432099`。
8. P6 oracle support 仍通过：oracle precision = `1.0`，coverage = `0.130926`，bad-event = `0.0`；但 measured signal strata = `3`，support measurement pass = `0`。
9. P7 support-region controller 未过：best = `C3-CheapSurrogatePrefilter`，precision = `0.172566`，coverage = `0.020926`，bad-event = `0.066372`。
10. 当前 blocker：`metric_kernel_compression_failed_while_oracle_support_remains`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9250_metric_kernel_compression_coverage_preserving_cascade_controller.py` | v9.2.50 runner；重新生成 fresh online rows，执行 metric subphase attribution、metric-kernel compression、event-sparse cascade、cheap surrogate、signal auxiliary、support/controller gate、route/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9250_metric_kernel_compression_coverage_preserving_cascade_controller.py
```

正式运行：

```bash
python experiments/run_v9250_metric_kernel_compression_coverage_preserving_cascade_controller.py \
  --out-dir results/real_rerun_20260506/v9250_metric_kernel_compression_coverage_preserving_cascade_controller_first_20260512T020000Z \
  --fresh --device auto --data-root data --seed 1314
```

实际参数来自 `run_manifest.json`：

```text
device = cuda
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
microprobe_steps = 240
train_size = 2048
batch_size = 64
hidden_dim = 256
```

## 2. Route

```json
{
  "route": "R10-PredictiveButMetricKernelTooExpensive",
  "base_candidate": "LQ-t2-h256",
  "v9249_boundary_pass": 1,
  "f7_subphase_attribution_pass": 1,
  "dominant_metric_subphase": "M0-logit preparation",
  "metric_kernel_id": "MK4-PreallocatedWorkspaceExact",
  "metric_kernel_compression_pass": 0,
  "f7_time_reduction": 0.30000000000000004,
  "metric_error_max": 0.0,
  "controller_agreement": 0.8017592592592593,
  "event_sparse_cascade_pass": 0,
  "probe_call_rate": 0.2,
  "amortized_overhead": 1.0181520959222663,
  "safe_good_recall_prefilter": 0.5572842998585573,
  "cheap_surrogate_prefilter_pass": 0,
  "cheap_surrogate_auc": 0.543916034323593,
  "cheap_surrogate_recall": 0.19236209335219237,
  "signal_aux_pass": 0,
  "signal_aux_bad_event": 0.43209876543209874,
  "measured_signal_strata_count": 3,
  "oracle_support_pass": 1,
  "oracle_precision": 1.0,
  "oracle_coverage": 0.13092592592592592,
  "oracle_bad_event": 0.0,
  "best_controller_id": "C3-CheapSurrogatePrefilter",
  "support_region_controller_pass": 0,
  "controller_auc": 0.5699914243040967,
  "controller_corr": 0.42510034311901856,
  "accepted_precision": 0.17256637168141592,
  "accepted_coverage": 0.020925925925925924,
  "accepted_bad_event_rate": 0.06637168141592921,
  "primary_blocker": "metric_kernel_compression_failed_while_oracle_support_remains",
  "next_required_implementation": "implement_real_fused_metric_kernel_or_lower_cost_sufficient_statistics",
  "success_v9250_strict_purekan_functional": 0,
  "success_v9250_full_functional": 0,
  "success_v9250_external_ready": 0
}
```

判断：oracle-good events 仍存在，但 metric kernel / cascade / controller 没有同时满足 legal、coverage、bad-event 与 system gate。

## 3. P1 F7 metric subphase attribution

Artifacts：

```text
p1_f7_metric_subphase_attribution.csv
metric_subphase_trace_v9250.csv
```

| subphase | time ms | ratio |
|---|---:|---:|
| M0-logit preparation | `2.306889` | `0.251454` |
| M5-wrong confidence p95 | `1.952160` | `0.212788` |
| M2-CEp99 / top-tail | `1.385743` | `0.151048` |
| M4-margin p10 / quantile | `1.074322` | `0.117103` |
| M3-margin per sample | `1.072474` | `0.116901` |

Summary：

```text
profile_event_count = 9
unknown_fraction = 0.0
phase_time_sum_close_to_F7 = 1
f7_subphase_attribution_pass = 1
```

判断：P1 成本归因闭合；本轮 dominant subphase 从 v9.2.49 的粗粒度 F7 进一步拆到了 logit preparation / tail metric 计算层面。

## 4. P2 metric-kernel compression

Artifacts：

```text
p2_metric_kernel_compression.csv
metric_kernel_trace_v9250.csv
system_metric_kernel_overhead_trace_v9250.csv
```

| kernel | status | numeric sanity | F7 reduction | agreement | metric error | step ratio | pass |
|---|---|---:|---:|---:|---:|---:|---:|
| MK0-ReferenceF7GapMetrics | measured_reference | `1` | `0.00` | `0.801759` | `0.000000` | `6.090760` | `0` |
| MK1-StreamingTopKTailApprox | approx_formula_measured_on_real_rows | `0` | `0.58` | `0.615741` | `0.876946` | `3.799918` | `0` |
| MK2-BranchSharedMetricReduction | approx_formula_measured_on_real_rows | `0` | `0.52` | `0.121759` | `1.074126` | `4.054456` | `0` |
| MK3-RiskValueHistogramApprox | approx_formula_measured_on_real_rows | `0` | `0.65` | `0.636296` | `3.069313` | `3.545380` | `0` |
| MK4-PreallocatedWorkspaceExact | exact_metrics_but_workspace_only | `1` | `0.30` | `0.801759` | `0.000000` | `5.072608` | `0` |

判断：MK4 保留 exact metric，但只是 workspace-style 省去部分开销，不是真正 fused metric kernel；system gate 仍失败。近似 kernel 虽更省，但 numeric/controller agreement 不够，不能写成 compression pass。

## 5. P3 recall-first event-sparse cascade

Artifact：

```text
p3_recall_first_event_sparse_cascade.csv
event_sparse_cascade_trace_v9250.csv
```

Route summary：

| metric | value |
|---|---:|
| best prefilter | `PF0-GapRecall` |
| probe call rate | `0.20` |
| amortized overhead | `1.018152` |
| safe-good recall | `0.557284` |
| accepted precision | `0.588710` |
| accepted coverage | `0.057407` |
| accepted bad-event | `0.356452` |
| cascade pass | `0` |

Notable clean-but-low-recall rows：

| prefilter | call rate | precision | coverage | bad-event | recall | overhead |
|---|---:|---:|---:|---:|---:|---:|
| PF1-ValueRiskRecall | `0.20` | `0.787634` | `0.034444` | `0.000000` | `0.257426` | `1.018152` |
| PF4-MetricCompressedRecall | `0.20` | `0.783069` | `0.035000` | `0.000000` | `0.272984` | `1.018152` |

判断：干净切片存在，但 recall 和 system overhead 都不满足 gate；recall 较高的 PF0 又带入太多 bad-event。

## 6. P4 cheap gap surrogate prefilter

Artifacts：

```text
p4_cheap_gap_surrogate_prefilter.csv
cheap_surrogate_trace_v9250.csv
```

| surrogate | AUC | corr | precision@0.03 | bad-event | recall | pass |
|---|---:|---:|---:|---:|---:|---:|
| CS0-RiskValueSignal | `0.509381` | `0.296262` | `0.194444` | `0.145062` | `0.144272` | `0` |
| CS1-FamilyLCB | `0.594640` | `0.371949` | `0.185185` | `0.243827` | `0.140028` | `0` |
| CS2-CompressedMetricApprox | `0.547509` | `0.393842` | `0.216049` | `0.083333` | `0.154173` | `0` |
| CS3-RoleBranchSignal | `0.543916` | `-0.042859` | `0.175926` | `0.194444` | `0.192362` | `0` |

判断：没有 cheap surrogate 同时满足 AUC/recall/call-rate 与 accept safety。`CS1` AUC 最高但仍低于 `0.60`，`CS3` recall 最高但 precision/bad-event 不足。

## 7. P5 signal-channel auxiliary audit

Artifacts：

```text
p5_signal_channel_auxiliary_audit.csv
signal_aux_trace_v9250.csv
```

| signal feature | AUC | corr | bad-event | precision delta | bad-event delta | pass |
|---|---:|---:|---:|---:|---:|---:|
| SA0-SignalChannel | `0.549790` | `0.080337` | `0.450617` | `0.064815` | `-0.009259` | `0` |
| SA1-NoiseReservoirEnergy | `0.543697` | `-0.091409` | `0.493827` | `0.000000` | `0.000000` | `0` |
| SA2-LowRankDisplacement | `0.489952` | `-0.182487` | `0.515432` | `-0.009259` | `0.092593` | `0` |
| SA3-TrustRatio | `0.573312` | `0.006997` | `0.432099` | `0.003086` | `-0.003086` | `0` |

判断：signal auxiliary 没有形成 v9.2.49 那种 diagnostic pass；best AUC 仍不足，且 bad-event 过高。

## 8. P6 online support / stratum expansion

Artifacts：

```text
p6_online_support_stratum_expansion.csv
support_density_trace_v9250.csv
```

Summary：

```text
natural_real_event_count = 10800
balanced_diagnostic_real_event_count = 0
measured_signal_strata_count = 3
measured_family_count = 45
support_measurement_pass = 0
oracle_support_pass = 1
oracle_precision = 1.0
oracle_coverage = 0.13092592592592592
oracle_bad_event = 0.0
```

Observed signal strata：

| stratum | rows |
|---|---:|
| S4-RoleWiseCurvature | `10462` |
| S1-CEHardTail | `279` |
| S5-UncertaintyLCB | `59` |

判断：fresh natural rows 足够，oracle-good support 也足够；但 signal strata 仍严重集中，低于计划的 multi-stratum support gate。

## 9. P7 support-region controller

Artifacts：

```text
p7_support_region_controller_calibration.csv
controller_calibration_trace_v9250.csv
```

| controller | official eligible | AUC heldout | corr heldout | precision | coverage | bad-event | step ratio | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1-MetricCompressedCascade | `0` | `0.546522` | `0.367116` | `0.161458` | `0.017778` | `0.078125` | `2.018152` | `0` |
| C2-RecallFirstGapCascade | `0` | `0.537942` | `0.342259` | `0.163158` | `0.017593` | `0.078947` | `2.018152` | `0` |
| C3-CheapSurrogatePrefilter | `0` | `0.569991` | `0.425100` | `0.172566` | `0.020926` | `0.066372` | `2.018152` | `0` |
| C4-SignalAuxCascade | `0` | `0.533683` | `0.327031` | `0.164894` | `0.017407` | `0.090426` | `2.018152` | `0` |
| C7-Oracle | `0` | `1.000000` | `0.339143` | `1.000000` | `0.047407` | `0.000000` | `2.018152` | `0` |

判断：P2-P5 没有 system-legal feature survivor，所以 P7 controllers 均 `official_eligible = 0`。即使作为 diagnostic，C3 的 precision、coverage、bad-event 与 system gate 也都没有闭合。

## 10. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p8_leave_dataset_and_stratum_out.csv` | `P7_no_system_legal_metric_kernel` |
| `p9_official_paired_replay.csv` | same |
| `p10_short_run_functional_validation.csv` | same |
| `p11_full_10seed_functional_validation.csv` | same |
| `p12_robustness_external_ready.csv` | same |

没有把 metric-kernel diagnostic、oracle support、clean-but-low-recall cascade rows 或 posthoc oracle 写成 leave-out / paired replay / short-run / full-run success。

## 11. No-fake audit

```text
rows_checked = 21746
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

## 12. Hash

| artifact | SHA256 |
|---|---|
| plan | `4f4174a4e2fb54c4299eb48fe78dfd1f0912d0af75ea99f19f5247b23b7b6b64` |
| runner | `7379c6e0c2dec51ddba3023d663a209c3130ef69a18f33bc29887dc30f7787a1` |
| run manifest | `0aff7294f4c309ddc1b1a0160da4106b7d0f9145f82c46bd704c1ea312616a74` |
| route | `640f105360a948071b245c322acb8808017beb5a64ffcf02dd96925fd70a7c32` |
| P0 boundary | `f49c3cfe79645e0b43dd9af2917c82735d0cb04c672ae31699f2ef6b59be5804` |
| P1 subphase attribution | `ea7d6fcc01f5e81c276d301623f7a0222088284f184b1b64533c6b326a79e3f2` |
| P2 metric kernel | `c905f3a1b2ddc190267d1db5fb0369ef7142a28c03171beb98722087434d15fc` |
| P3 cascade | `d69993159d1ef7ba571b1a6ec7dc5bc9fa558a4b4a9b98b51ca266b916cc41c3` |
| P4 cheap surrogate | `091a73a301988542e3231b560b41d6c4ab2fa7c1df2e4a9dec999f0dce7e1931` |
| P5 signal auxiliary | `04516567939309d1d26225e5d623c04fd451eb283c3a9365871672dff30ca8be` |
| P6 support expansion | `a52699a357c7e132e25ce930a57db7a0a627799ebf95f8c2f4fbf8899a72a642` |
| P7 controller | `1cce5a8b16f1dcc97422d01585a7f3b126e013d226302e0e45f88f7173e6af91` |
| P8 boundary | `aff4f9834e662e72e7c684708ab89f248cc6c1b5ada01ce608e39922d5825e22` |
| P9 boundary | `2874fc543fe91bd287f9c96d65442ffbb67b199a6ca4c7e580d98ef01a3bb830` |
| P10 boundary | `0ab12c50aef4767eeaeaafd512528e3802c4b59a17753653f52422cf5e3520bc` |
| P11 boundary | `34871e85000122caec143741ea2592faf431b92453e19ef165d931168544fdc1` |
| P12 boundary | `7b49b9f6b9db62c1767699d4804f7ce8dbe8cefcf22e5373a2388f4f075cda8e` |
| failure table | `4e8de239f41f98c0a8cb82fc3b4a6708281f4f9b63939d58480f72ab2135ab9a` |
| provenance audit | `1036cc2d254811c3d577128164ffa6edeb0a2bd9ecb5787ccaa4af863ffb4abc` |

## 13. 最终分析结论

v9.2.50 的真实推进是：

```text
v9.2.49: signal sketch 有诊断信号，但 controller heldout gate failed。
v9.2.50: 继续拆 metric cost，测试 metric-kernel compression、
          recall-first cascade、cheap surrogate 与 signal auxiliary。
```

机制判断：

1. F7 metric cost 已经被进一步拆开；这轮 dominant 是 logit preparation / tail metric，而不是 controller decision。
2. Exact metric route 没有丢数值，但省得不够；approx route 能降成本但 agreement / error 不合格。
3. Cascade 的核心矛盾没有解决：clean slice recall 太低，recall-first slice bad-event 太高。
4. Cheap surrogate 和 signal auxiliary 都没有达到可转正的 legal feature gate。
5. Oracle support 仍高，说明 good events 没有消失；真正 blocker 是用 system-safe metric kernel 或低成本充分统计量把这些 events 以足够 coverage 和安全性识别出来。

最终一句话：

> v9.2.50 真实执行后停在 `R10-PredictiveButMetricKernelTooExpensive`：oracle-good support 仍存在，但 metric-kernel compression / cascade / cheap surrogate / controller 均未闭合，strict PureKAN functional 仍未成功。
