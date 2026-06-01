# DG-KAN v9.2.49 Cost-Amortized Online MicroProbe Kernelization 与 Cheap-Gap Controller Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.2.49_CostAmortizedOnlineMicroProbe_KernelizedCheapGapController_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R5-SignalChannelSketchPass
base_candidate = LQ-t2-h256
success_v9249_strict_purekan_functional = False
success_v9249_full_functional = False
success_v9249_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9249_cost_amortized_online_microprobe_kernelized_cheap_gap_controller_first_20260512T010000Z/
```

核心结论：

1. P0 复现 v9.2.48 boundary：source route = `R8-MicroProbePredictiveButTooExpensive`，fake/proxy = `0`。
2. P1 microprobe phase-cost attribution pass = `1`，dominant phase = `F7-metric computation CE/margin/risk/gap`，time ratio = `0.554049`。
3. P2 event-sparse amortized probe 未过：best amortized overhead = `0.204674`，略高于 `<=0.20` gate，且 accepted coverage = `0.000980`。
4. P3 cheap gap surrogate 未过：best = `CG5-FamilyReliabilityLCB`，AUC = `0.657509`，corr = `0.273813`，precision = `0.326087`。
5. P4 kernelized/fused probe 未过：K5 cheap smoke overhead = `0.05` 但不等价；K6 amortized 等价但 overhead = `0.204674 > 0.20`。
6. P5 signal-channel sketch diagnostic pass = `1`：best = `S4-NoiseReservoirEnergy`，AUC = `0.609584`；但 bad-event = `0.510870`，不能作为 official success。
7. P6 oracle support 仍通过：oracle precision = `1.0`，coverage = `0.137255`，bad-event = `0.0`；但 measured signal strata 只有 `2`，support measurement pass = `0`。
8. P7 support-region controller pass = `0`；best = `C3-SignalSketchBest`，precision = `0.242424`，coverage = `0.021569`，bad-event = `0.424242`。
9. 当前 blocker：`signal_channel_predictive_but_controller_failed`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9249_cost_amortized_online_microprobe_kernelized_cheap_gap_controller.py` | v9.2.49 runner；重新生成 fresh online microprobe rows，执行成本分解、event-sparse、cheap surrogate、kernelized smoke、signal sketch、support/controller gate、route/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9249_cost_amortized_online_microprobe_kernelized_cheap_gap_controller.py
```

正式运行：

```bash
python experiments/run_v9249_cost_amortized_online_microprobe_kernelized_cheap_gap_controller.py \
  --out-dir results/real_rerun_20260506/v9249_cost_amortized_online_microprobe_kernelized_cheap_gap_controller_first_20260512T010000Z \
  --fresh --device auto --data-root data --seed 1314
```

实际 fresh online microprobe 范围：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
steps per seed/dataset = 68
carriers = A1,A2,A3
fresh online rows = 3060
```

## 2. Route

```json
{
  "route": "R5-SignalChannelSketchPass",
  "base_candidate": "LQ-t2-h256",
  "v9248_boundary_pass": 1,
  "probe_cost_attribution_pass": 1,
  "dominant_probe_cost_phase": "F7-metric computation CE/margin/risk/gap",
  "event_sparse_probe_pass": 0,
  "probe_call_rate": 0.03986928104575163,
  "per_probe_overhead_q90": 5.133629549832948,
  "amortized_overhead": 0.20467411930706522,
  "cheap_gap_surrogate_pass": 0,
  "cheap_gap_auc": 0.6575090187590188,
  "cheap_gap_corr": 0.27381339002793664,
  "kernelized_probe_pass": 0,
  "kernelized_probe_overhead_q90": 0.05,
  "signal_channel_pass": 1,
  "signal_channel_auc": 0.6095842352092352,
  "signal_channel_corr": 0.00958847158424319,
  "measured_signal_strata_count": 2,
  "oracle_support_pass": 1,
  "oracle_precision": 1.0,
  "oracle_coverage": 0.13725490196078433,
  "oracle_bad_event": 0.0,
  "best_controller_id": "C3-SignalSketchBest",
  "support_region_controller_pass": 0,
  "controller_auc": 0.5921277122933701,
  "controller_corr": -0.012168811209746668,
  "accepted_precision": 0.24242424242424243,
  "accepted_coverage": 0.021568627450980392,
  "accepted_bad_event_rate": 0.42424242424242425,
  "primary_blocker": "signal_channel_predictive_but_controller_failed",
  "next_required_implementation": "combine_signal_sketch_with_control_gap_prefilter"
}
```

判断：v9.2.49 确实把 v9.2.48 的 expensive probe 做了成本分解，并测试了 sparse / cheap / kernelized / signal sketch 分支。唯一过的正向 gate 是 P5 signal diagnostic，不是 official controller gate。

## 3. P1 microprobe phase-cost attribution

Artifacts：

```text
p1_microprobe_phase_cost_attribution.csv
probe_phase_trace_v9249.csv
memory_traffic_trace_v9249.csv
```

Phase cost：

| phase | time ms | ratio |
|---|---:|---:|
| F7 metric CE/margin/risk/gap | `43.382244` | `0.554049` |
| F2 candidate functional update compute | `21.386187` | `0.273130` |
| F0 base task step reference | `3.567680` | `0.045564` |
| F4 probe forward Real | `3.349693` | `0.042780` |
| F6 probe forward bestLR | `1.928096` | `0.024624` |
| F3 shadow apply / delta view | `1.223140` | `0.015621` |
| F1 cheap prefilter | `1.163799` | `0.014863` |
| F5 probe forward AdamWParallel | `1.157310` | `0.014780` |
| F9 rollback / commit | `0.520215` | `0.006644` |
| F10 logging | `0.510563` | `0.006521` |
| F8 controller decision | `0.111477` | `0.001424` |

Summary：

```text
profile_event_count = 18
unknown_overhead_fraction = 0.0
dominant_phase_identified = 1
phase_time_sum_close_to_total = 1
probe_cost_attribution_pass = 1
```

判断：成本主因不是 controller decision，而是 metric/gap label computation 与 candidate functional update compute。继续只调 threshold 不能解决 system blocker。

## 4. P2 event-sparse amortized microprobe

Artifacts：

```text
p2_event_sparse_amortized_microprobe.csv
event_sparse_probe_trace_v9249.csv
```

Best summary：

```text
best_probe_candidate = MP1-EventSparseGapProbe
best_prefilter_id = PF2-FamilyCandidate
probe_call_rate = 0.03986928104575163
per_probe_overhead_q90 = 5.133629549832948
amortized_overhead = 0.20467411930706522
step_ratio_q90 = 1.2046741193070651
accepted_precision = 1.0
accepted_coverage = 0.000980392156862745
accepted_bad_event_rate = 0.0
event_sparse_probe_pass = 0
```

Representative rows：

| probe | prefilter | call rate | amortized overhead | precision | coverage | bad-event | pass |
|---|---|---:|---:|---:|---:|---:|---:|
| MP1 | PF2-FamilyCandidate | `0.039869` | `0.204674` | `1.000000` | `0.000980` | `0.000000` | `0` |
| MP1 | PF2-FamilyCandidate | `0.030065` | `0.154344` | `1.000000` | `0.000327` | `0.000000` | `0` |
| MP8 | PF0-BranchRisk | `0.030065` | `0.154344` | `0.826087` | `0.007516` | `0.173913` | `0` |

判断：sparse invocation 接近 system gate，但 coverage 远低于 `0.03`；能挑到很干净的少数事件，不等于 controller support closure。

## 5. P3 cheap gap surrogate matrix

Artifacts：

```text
p3_cheap_gap_surrogate_matrix.csv
cheap_gap_surrogate_trace_v9249.csv
```

| surrogate | AUC | corr | precision | coverage | bad-event | pass |
|---|---:|---:|---:|---:|---:|---:|
| CG5-FamilyReliabilityLCB | `0.657509` | `0.273813` | `0.326087` | `0.030065` | `0.326087` | `0` |
| CG3-SupportDensityValue | `0.654184` | `0.038575` | `0.286885` | `0.039869` | `0.278689` | `0` |
| CG1-ValueSignalFamily | `0.610473` | `0.191512` | `0.271739` | `0.030065` | `0.358696` | `0` |
| CG4-LinearizedCheapGap | `0.518117` | `0.211589` | `0.206522` | `0.030065` | `0.163043` | `0` |
| CG0-RiskValueSignal | `0.488217` | `0.156233` | `0.195652` | `0.030065` | `0.152174` | `0` |

判断：cheap surrogate 没有达到 AUC/corr + accept gate。`CG5` 有一点 family signal，但 bad-event 与 precision 不够，不能替代 expensive `gap_probe`。

## 6. P4 kernelized / fused microprobe

Artifacts：

```text
p4_kernelized_fused_microprobe.csv
kernelized_probe_trace_v9249.csv
```

| candidate | status | numeric eq | overhead q90 | amortized overhead | step q90 | pass |
|---|---|---:|---:|---:|---:|---:|
| K0-reference-gap-probe | measured_reference | `1` | `5.133630` | `5.133630` | `6.133630` | `0` |
| K4-cached-control-gap | derived_from_real_event_sparse_trace | `1` | `5.133630` | `0.204674` | `1.204674` | `0` |
| K5-linearized-gap-jvp | cheap_surrogate_smoke | `0` | `0.050000` | `5.133630` | `1.050000` | `0` |
| K6-hybrid-fused-amortized | derived_from_real_event_sparse_trace | `1` | `5.133630` | `0.204674` | `1.204674` | `0` |

Not implemented：

```text
K1-delta-view-shadow-probe
K2-one-forward-multibranch-probe
K3-fused-metric-kernel
```

判断：真正等价的 route 仍卡在 amortized overhead `0.204674 > 0.20`；便宜 smoke 不满足 numeric equivalence。不能把 analytic/derived smoke 写成 kernelized success。

## 7. P5 signal-channel sketch v2

Artifacts：

```text
p5_signal_channel_sketch_v2.csv
signal_channel_trace_v9249.csv
```

| feature | AUC safe-good | corr | precision | coverage | bad-event | diagnostic pass |
|---|---:|---:|---:|---:|---:|---:|
| S4-NoiseReservoirEnergy | `0.609584` | `0.009588` | `0.282609` | `0.030065` | `0.510870` | `1` |
| S3-LowRankDisplacementEnergy | `0.589000` | `-0.002198` | `0.336957` | `0.030065` | `0.521739` | `0` |
| S6-TrustRatioSignal | `0.572074` | `0.014820` | `0.200436` | `0.150000` | `0.511983` | `0` |
| S2-SignalChannelTail | `0.540474` | `0.032187` | `0.336957` | `0.030065` | `0.521739` | `0` |
| S1-SignalChannelRatio | `0.540473` | `0.032186` | `0.336957` | `0.030065` | `0.521739` | `0` |

判断：P5 route 能推进到 `R5`，因为 signal sketch AUC 超过 `0.60` diagnostic gate。但它的 accepted slice bad-event 很高，不能直接进入 official controller。

## 8. P6 online support / stratum expansion

Artifacts：

```text
p6_online_support_stratum_expansion.csv
support_density_trace_v9249.csv
```

Summary：

```text
natural_real_event_count = 3060
diagnostic_balanced_real_event_count = 0
measured_signal_strata_count = 2
measured_family_count = 36
carrier_active = 1
support_measurement_pass = 0
oracle_support_pass = 1
oracle_accepted_count = 420
oracle_precision = 1.0
oracle_coverage = 0.13725490196078433
oracle_bad_event = 0.0
```

判断：fresh natural rows 数量达到 `>=3000`，但 signal-stratum 仍只有 `2` 个，低于 `>=6`。oracle support 没有消失，失败点仍是 legal/system/controller 侧。

## 9. P7 support-region controller calibration

Artifacts：

```text
p7_support_region_controller_calibration.csv
controller_calibration_trace_v9249.csv
```

| controller | official eligible | AUC heldout | corr heldout | precision | coverage | bad-event | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1-EventSparseBest | `0` | `0.509851` | `0.194360` | `0.101667` | `0.196078` | `0.473333` | `0` |
| C2-CheapSurrogateBest | `0` | `0.544370` | `0.264484` | `0.116438` | `0.190850` | `0.431507` | `0` |
| C3-SignalSketchBest | `0` | `0.592128` | `-0.012169` | `0.242424` | `0.021569` | `0.424242` | `0` |
| C5-HybridCheapAmortized | `0` | `0.546451` | `0.264110` | `0.116838` | `0.190196` | `0.431271` | `0` |
| C7-Oracle | `0` | `1.000000` | `0.370139` | `1.000000` | `0.042157` | `0.000000` | `0` |

判断：因为 P2/P3/P4 没有 system-legal official feature，P7 legal controllers `official_eligible = 0`；即使作为 diagnostic，它们也没有过 precision/coverage/bad-event gate。`C7` 是 posthoc oracle，不能转正。

## 10. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p8_leave_dataset_and_stratum_out.csv` | `P7_controller_failed` |
| `p9_official_paired_replay.csv` | same |
| `p10_short_run_functional_validation.csv` | same |
| `p11_full_10seed_functional_validation.csv` | same |
| `p12_robustness_external_ready.csv` | same |

没有把 signal sketch diagnostic、oracle support、event-sparse clean-but-tiny slice 或 posthoc oracle 写成 leave-out / paired replay / short-run / full-run success。

## 11. No-fake audit

```text
rows_checked = 6366
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
| plan | `78135827cafc4d707dfd82b8ee7dee7b7beba1ae8f10f0fd6b7f67d5a5e9c35d` |
| runner | `4a8fd05ed90b2582379bd3214f47654617a705850e4bedf699cda6cf2ab847a8` |
| run manifest | `e04e7dd26864b24681d638e1dc9a41319383df5026db3c641848c6dea7a0e79e` |
| route | `8b7295dd7e276d9d5e7c1525e5fb5c7d67a4c6ddcd41f63624542d5c31293c4c` |
| P0 boundary | `d213d09d99573592d063a81b29cf25ff8aa27971097249af2d9f4b0d09356c51` |
| P1 cost attribution | `68d583d41cdc472f2736208780347e1c99f7b982b5cda6eae57e9c227f7d3f20` |
| P2 event sparse | `beccf4d8290b0f29ad85681a04919737d200854ff1a0968b63db91025731cb00` |
| P3 cheap surrogate | `d06bbfd33f9a087e5947f70045f1e226670bb56da1ae79823d420dbff8b115c0` |
| P4 kernelized | `f55a9b810c62eeb639f5cdf1f94b8d678d0d8e5f06565a3648f5ecbfdac34ce5` |
| P5 signal sketch | `82e0530976a5c64a4327f7339c6a7371aaea863002baf406a2af416074d0bd4a` |
| P6 support expansion | `0cac1e500e6c58a1e22d77c6a573fbcf39f79700027863d2386ef8ec0627bc43` |
| P7 controller | `b9e38fe53ebe99ab67205309e2f9c497ced12307a8df73c10bdf718696b81ce4` |
| P8 boundary | `5757492a2b517df3743a80b216fbe79c490cb605a319e9f681c6e649d9765c05` |
| P9 boundary | `2255e4b1f7894d8b7bae3dc64575b62c60fc07c14bb9be5ad71929ae9b63e444` |
| failure table | `57e71d2aa2d2cf2566b7a4f2718465a367a96d0079bf0f4548193e2a06c2b74c` |
| provenance audit | `d18ab2a5f06baff0932556118068cf4b6a158d6a872127a5f48dbfcf044c876a` |

## 13. 最终分析结论

v9.2.49 的真实推进是：

```text
v9.2.48: online gap_probe 可预测 safe-good，但 expensive。
v9.2.49: 对 expensive probe 做 cost attribution，并测试 sparse / cheap / kernelized / signal sketch 分支。
```

机制判断：

1. 成本归因闭合：dominant phase 是 metric/gap computation，其次是 functional update compute。
2. Event-sparse amortization 已经非常接近 system gate，但 best overhead `0.204674` 仍高于 `0.20`，而且 coverage 只有 `0.000980`。
3. Cheap gap surrogate 没有达到 safe-good predictivity + accept gate；不能替代 expensive `gap_probe`。
4. Kernelized 分支没有真正闭合：等价分支仍略破 overhead gate，便宜 smoke 不等价，K1/K2/K3 明确 not implemented。
5. Signal sketch 有诊断信号，足以进入 `R5-SignalChannelSketchPass`，但 bad-event 过高，controller 不过。
6. Oracle support 仍高，说明 good events 没有消失；当前 blocker 是把 signal sketch / control-gap prefilter 合成一个 legal、system-safe、coverage 足够的 controller。

最终一句话：

> v9.2.49 真实执行后停在 `R5-SignalChannelSketchPass`：cheap signal sketch 出现可测诊断信号，但 support-region controller heldout gate 失败，strict PureKAN functional 仍未成功。
