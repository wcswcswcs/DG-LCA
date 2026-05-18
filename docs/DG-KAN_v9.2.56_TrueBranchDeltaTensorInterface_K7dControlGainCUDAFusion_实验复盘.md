# DG-KAN v9.2.56 True Branch-Delta Tensor Interface 与 K7d Control-Gain CUDA Fusion 实验复盘

> 本复盘记录 `DG-KAN_v9.2.56_TrueBranchDeltaTensorInterface_K7dControlGainCUDAFusion_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R12-TrueDeltaPredictiveButTooExpensive
base_candidate = LQ-t2-h256
success_v9256_strict_purekan_functional = False
success_v9256_full_functional = False
success_v9256_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9256_true_branch_delta_tensor_interface_k7d_control_gain_cuda_fusion_first_20260512T083000Z/
```

核心结论：

1. P0 复现 v9.2.55 boundary：source route = `R9-TrueCustomExtensionNotImplemented`，fake/proxy/offload = `0`。
2. P1 true branch-delta tensor interface 已实现并通过 gate：`uses_true_branch_delta = 1`，`uses_source_measured_gap = 0`，`uses_formula_proxy = 0`。
3. P2 K7d CUDA fusion 已通过局部 fusion gate：reference K7d time = `2.081355 ms`，fused K7d time = `0.329207 ms`，time reduction = `0.841830`，max abs error = `9.536743e-07`。
4. P2 的 K7d reference dominant subphase 是 `K7d7-support_score_compute`，ratio = `0.629845`；不是未解释 overhead。
5. P3 true branch-delta exact signal 仍强：best = `TBD0-CBD0Reference`，AUC = `0.888401`，agreement = `1.0`，但 step ratio = `2.863280 > 1.50`，system pass = `0`。
6. P3 中真正使用 true branch logits 的 `TBD1/TBD2/TBD3` 均未过 system gate；`TBD3-CUDAK7dTrueBranchDeltaFusion` step ratio = `4.924871`。
7. P4 negative controls 正确拒绝：formula proxy 与 source-measured gap 都 `official_eligible = 0`，没有倒灌成 official success。
8. P5 exact-signal controller 未过：best diagnostic controller precision = `0.147593`，coverage = `0.182044`，bad-event = `0.315622`。
9. P6 oracle support 仍通过：precision = `1.0`，coverage = `0.121693`，bad-event = `0.0`；但 support measurement 仍失败：measured signal strata = `3`，balanced diagnostic rows = `36`。
10. 当前 blocker：`true_branch_delta_predictive_but_too_expensive`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9256_true_branch_delta_tensor_interface_k7d_control_gain_cuda_fusion.py` | v9.2.56 runner；复现 v9.2.55 boundary，生成 true branch-delta tensor interface rows，执行 K7d CUDA fusion、true custom delta matrix、proxy audit、controller/support/downstream/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9256_true_branch_delta_tensor_interface_k7d_control_gain_cuda_fusion.py
```

正式运行：

```bash
python experiments/run_v9256_true_branch_delta_tensor_interface_k7d_control_gain_cuda_fusion.py \
  --out-dir results/real_rerun_20260506/v9256_true_branch_delta_tensor_interface_k7d_control_gain_cuda_fusion_first_20260512T083000Z \
  --fresh --device auto --data-root data --seed 1314
```

实际参数来自 `run_manifest.json`：

```text
device = cuda
triton_available = true
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7
microprobe_steps = 168
interface_events = 24
interface_steps = 2
train_size = 2048
batch_size = 64
hidden_dim = 256
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R12-TrueDeltaPredictiveButTooExpensive",
  "base_candidate": "LQ-t2-h256",
  "v9255_boundary_pass": 1,
  "true_branch_delta_interface_pass": 1,
  "uses_true_branch_delta": 1,
  "uses_source_measured_gap": 0,
  "uses_formula_proxy": 0,
  "k7d_attribution_pass": 1,
  "dominant_k7d_subphase": "K7d7-support_score_compute",
  "k7d_fusion_pass": 1,
  "k7d_time_reduction": 0.8418303698075655,
  "best_custom_delta_id": "TBD0-CBD0Reference",
  "custom_delta_predictivity_pass": 1,
  "custom_delta_agreement_pass": 1,
  "custom_delta_system_pass": 0,
  "custom_delta_auc": 0.8884008136827201,
  "custom_delta_accept_agreement": 1.0,
  "custom_delta_step_ratio_q90": 2.8632798851361203,
  "formula_proxy_negative_control_pass": 1,
  "best_controller_id": "C4-BorderlineFallbackExactSignalController",
  "exact_signal_controller_pass": 0,
  "oracle_support_pass": 1,
  "support_measurement_pass": 0,
  "primary_blocker": "true_branch_delta_predictive_but_too_expensive"
}
```

判断：v9.2.56 已经把 v9.2.55 的 “true tensor interface 未实现” 推进掉；但 true branch-delta exact signal 仍未进入 system envelope，因此不能打开 LDO/LSO 或 paired replay。

## 3. P1 true branch-delta tensor interface

Artifacts：

```text
p1_true_branch_delta_tensor_interface_gate.csv
true_branch_delta_interface_trace_v9256.csv
```

| candidate | implementation | true branch delta | source gap | formula proxy | output shape | pass |
|---|---|---:|---:|---:|---|---:|
| TBD1-SelectedLogitTrueBranchDelta | implemented true selected logits | `1` | `0` | `0` | `[4, 64, 5]` | `1` |
| TBD2-FullLogitSmallCTrueBranchDelta | implemented true full logits | `1` | `0` | `0` | `[4, 64, 10]` | `1` |
| TBD3-TrueBranchDeltaMetricFused | CUDA extension true full logits gain-only | `1` | `0` | `0` | `[1536]` | `1` |
| PX1-SourceMeasuredGapInput | rejected source-measured gap | `0` | `1` | `0` | `{}` | `0` |

Summary：

```text
interface_event_count = 24
interface_sample_count = 1536
true_branch_delta_interface_pass = 1
commit_time_order_valid = 1
uses_dataset_name/validation/test/posthoc = 0/0/0/0
```

判断：P1 是本轮关键推进。它不是 source-measured scalar gap，也不是 selected-logit formula proxy；runner 实际构造了 branch logits / selected logits tensor interface，并保留 hash。

## 4. P2 K7d control-gain CUDA fusion

Artifacts：

```text
p2_k7d_control_gain_subphase_attribution_and_fusion.csv
k7d_control_gain_trace_v9256.csv
```

Subphase summary：

| subphase | time ms | ratio |
|---|---:|---:|
| K7d7-support_score_compute | `1.310931` | `0.629845` |
| K7d0-branch_gain_prepare | `0.193703` | `0.093066` |
| K7d6-risk_score_compute | `0.187954` | `0.090304` |
| K7d11-logging_hash_timestamp | `0.085839` | `0.041242` |
| K7d1-real_gain_compute | `0.051528` | `0.024757` |

Fusion summary：

```text
implementation = implemented_cuda_extension_true_full_logits_gain_only
reference_k7d_time_ms = 2.0813550800085068
k7d_fused_time_ms = 0.3292071633040905
k7d_time_reduction = 0.8418303698075655
k7d_gap_max_abs_error = 9.5367431640625e-07
k7d_fusion_pass = 1
k7d_attribution_pass = 1
```

判断：K7d gain-only fusion 是真实 CUDA extension path，且数值误差很小；但它只压缩 K7d 局部，不等于整条 branch-forward / true delta path 过 system gate。

## 5. P3 true custom branch-delta matrix

Artifacts：

```text
p3_true_custom_branch_delta_implementation_matrix.csv
true_custom_branch_delta_trace_v9256.csv
system_true_delta_kernel_overhead_trace_v9256.csv
```

| custom delta | status | true | source gap | formula | AUC | agreement | step q90 | bad-event | system |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TBD0-CBD0Reference | reference true full branch forward | `1` | `0` | `0` | `0.888401` | `1.000000` | `2.863280` | `0.548209` | `0` |
| TBD1-SelectedLogitTrueBranchDelta | true selected logits, full forward gather | `1` | `0` | `0` | `0.888401` | `1.000000` | `5.219511` | `0.548209` | `0` |
| TBD2-FullLogitSmallCTrueBranchDelta | true full logits small-C | `1` | `0` | `0` | `0.888401` | `1.000000` | `5.219511` | `0.548209` | `0` |
| TBD3-CUDAK7dTrueBranchDeltaFusion | CUDA K7d true full logits fusion | `1` | `0` | `0` | `0.888401` | `1.000000` | `4.924871` | `0.548209` | `0` |
| TBD4-TrueBranchDeltaWithExactFallback | hybrid borderline exact diagnostic | `1` | `0` | `0` | `0.590811` | `0.449901` | `3.478231` | `0.220386` | `0` |
| PX1-FormulaProxyNegativeControl | diagnostic formula proxy rejected | `0` | `0` | `1` | `0.589056` | `0.448661` | `1.350000` | `0.225895` | `0` |
| PX2-SourceMeasuredGapInput | diagnostic source gap rejected | `0` | `1` | `0` | `0.888401` | `1.000000` | `1.020000` | `0.548209` | `0` |

Summary：

```text
true_custom_branch_delta_implemented = 1
custom_delta_predictivity_pass = 1
custom_delta_agreement_pass = 1
custom_delta_system_pass = 0
formula_proxy_negative_control_pass = 1
```

判断：P3 不能写成 custom delta success。Exact/true delta signal 强，但最好的 true route step ratio 仍高于 `1.50`；低成本 proxy 要么是 formula，要么是 source-measured gap，均不 official。

## 6. P4 negative controls

Artifacts：

```text
p4_formula_proxy_negative_control_audit.csv
formula_proxy_negative_control_trace_v9256.csv
```

| control | official eligible | AUC | agreement | step q90 | reason |
|---|---:|---:|---:|---:|---|
| PX1-FormulaProxyNegativeControl | `0` | `0.589056` | `0.448661` | `1.350000` | formula proxy |
| PX2-SourceMeasuredGapInput | `0` | `0.888401` | `1.000000` | `1.020000` | source-measured gap |

判断：`PX2` 看起来便宜且强，但它使用 source-measured gap input；P4 明确拒绝，不能进入 route。

## 7. P5 exact-signal controller

Artifacts：

```text
p5_exact_signal_controller_calibration.csv
exact_signal_controller_trace_v9256.csv
```

Summary：

```text
best_controller_id = C4-BorderlineFallbackExactSignalController
official_eligible = 0
controller_auc = 0.6098899455069986
controller_corr = 0.30531835937735224
accepted_precision = 0.14759309718437782
accepted_coverage = 0.18204365079365079
accepted_bad_event_rate = 0.315622161671208
accepted_signal_strata_count = 2
accepted_family_count = 18
exact_signal_controller_pass = 0
```

判断：由于 P3 system gate 未过，P5 controller 全部不能 official。即使按 diagnostic 观察，precision 和 bad-event 也不过。

## 8. P6 support expansion

Artifacts：

```text
p6_online_support_stratum_expansion.csv
support_density_trace_v9256.csv
```

Summary：

```text
natural_real_event_count = 12096
balanced_diagnostic_real_event_count = 36
measured_signal_strata_count = 3
measured_family_count = 45
support_measurement_pass = 0
oracle_support_pass = 1
oracle_precision = 1.0
oracle_coverage = 0.12169312169312169
oracle_bad_event = 0.0
```

判断：good events 仍存在，但 signal-stratum coverage 仍太窄，balanced diagnostic 只有 `36` rows，不能写成 support measurement pass。

## 9. Downstream boundary

这些 artifact 均已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p7_leave_dataset_and_stratum_out.csv` | `P3_true_delta_system_failed` |
| `p8_official_paired_replay.csv` | same |
| `p9_short_run_functional_validation.csv` | same |
| `p10_full_10seed_functional_validation.csv` | same |

没有把 true tensor interface、K7d local fusion、oracle support、formula proxy 或 source-gap proxy 倒灌成 LDO/LSO、paired replay、short-run 或 full-run success。

## 10. No-fake audit

```text
rows_checked = 24356
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
| plan | `a95607c068cb78e0b902c61ed3a5d469a00a50756fcf92bbd0e03c15865565f2` |
| runner | `3526e31cecc878fa6cf3069ba9a08b1c967ba3a4023b4be3366ff5b6b3d31ab7` |
| run manifest | `52a2f3d592f297623b1440bbeef99f244bfff1544a84e156e5618206021e75e2` |
| route | `ef7b9c3882c1b9c35231b89c5fd111213cd126ff78367014d9a209ad36e291fb` |
| P0 boundary | `41ab76838c53b3043b557c34f2bb95c1f9339c444850ae89d6a9099d2acf5768` |
| P1 true interface | `39580a4840f1f0378dbb3208642c68e020277093c9323c20427af9c641f8adb0` |
| P2 K7d fusion | `00e46c99910b4b77e57eaad05361b6464a40c5a1ff533264f17b1aae12138913` |
| P3 true delta matrix | `c6445eccbc576cafd70d0727e216f937fb2582f1be3b04a0954cff335d412de1` |
| P4 proxy audit | `406e54567862cd9d338e73b1f310ab6ab4e28cc95702b8b720945d6fe7473acb` |
| P5 controller | `3842ed5819faa4a53aace90697c10cf5b87b88f577d217eda58dadf0413616c2` |
| P6 support | `278a51032aab7850b9b0ef7e2cdc515a342b207e7a238767ef74da68dc95a92c` |
| P7 boundary | `31a64e049a8ed3e77170bce93933928e1aafee9bbce95db67af0332cc03b3b2d` |
| P8 boundary | `3809be3e11ec32f2a2c1228ad73dd3736fdd6d5a09771c5d67ae13de3a6ac883` |
| P9 boundary | `4b471ad09ab6012e04159f5d2fdf654b08d8595e9a702a4861e844bdb14c6c3a` |
| P10 boundary | `1347046e1d06c22809fef86d0bd9397b3b97fb2a177c4dc6a161243444050914` |
| failure table | `755108b7a04dbc1d2ea7a03f708e7a24be21aad9eedaf76beaa63ae08cbf105b` |
| provenance audit | `fede2b3243987cb394a2031ab28afc00d1a10b4c1586bb445052458b7098f72e` |

## 12. 最终分析结论

v9.2.56 的真实推进是：

```text
v9.2.55: CUDA extension path 已打通，但 true branch-delta logits kernel 未实现。
v9.2.56: true branch-delta tensor interface 已实现；
          K7d gain-only CUDA fusion 已通过局部数值与成本 gate；
          但 full branch-forward / true delta path 仍太贵。
```

机制判断：

1. H1 得到推进：true selected/full branch logits interface 已落盘，source gap 和 formula proxy 被排除。
2. H2 局部成立：K7d control-gain fusion 数值正确且大幅降局部时间。
3. H3 未闭合：true/exact branch-delta signal 仍然强，但 system step ratio 仍高于 `1.50`。
4. H4 继续成立：oracle support 仍存在，precision `1.0`，coverage `0.121693`，bad-event `0.0`。
5. 当前下一步不是再证明 signal，而是把 branch-forward/true delta path 做 lower-level fusion，压进 system envelope 后再重开 exact-signal controller。

最终一句话：

> v9.2.56 真实执行后停在 `R12-TrueDeltaPredictiveButTooExpensive`：true branch-delta tensor interface 与 K7d CUDA fusion 均已推进，但 exact/true branch-delta 路径仍太贵，strict PureKAN functional 仍未成功。
