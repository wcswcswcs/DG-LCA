# DG-KAN v9.2.55 Lower-Level CUDA Branch-Delta Extension 与 Exact-Signal Controller Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.2.55_LowerLevelCUDA_BranchDeltaExtension_ExactSignalControllerClosure_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R9-TrueCustomExtensionNotImplemented
base_candidate = LQ-t2-h256
success_v9255_strict_purekan_functional = False
success_v9255_full_functional = False
success_v9255_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9255_lower_level_cuda_branch_delta_extension_exact_signal_controller_closure_first_20260512T071500Z/
```

核心结论：

1. P0 复现 v9.2.54 boundary：source route = `R14-NeedsLowerLevelCUDAExtension`，source true custom branch-delta implemented = `0`，oracle support pass = `1`，fake/proxy/offload = `0`。
2. 本轮 manifest 记录 `device = cuda`，`triton_available = true`；正式运行中 GPU 采样有活动：`NVIDIA L4, 32 %, 326 MiB`、`32 %, 326 MiB`，编译尾段采样为 `0 %, 326 MiB`。
3. P1 K7 residual attribution pass = `1`，dominant subphase = `K7d-control_gain_compute`，ratio = `0.8661336752717377`。
4. P2 有 CUDA extension 编译/调用成功的 row：`CBD4-CUDABranchDeltaMetricFusedExtension`，step ratio = `1.0236406340263784`；但它使用 source-measured gap input，不是 true branch-delta logits kernel，因此 `uses_true_branch_delta = 0`，official pass = `0`。
5. P2 reference exact signal 仍强：`CBD0-V9254Reference` AUC = `0.8884008136827201`，agreement = `1.0`，但 step ratio = `2.8632798851361203 > 1.50`。
6. P2 formula negative control 仍不过：`CBD1` step ratio = `1.35`，但 AUC = `0.5349884272995514`，agreement = `0.38682208994708994`。
7. 真正的 selected/full-logit branch-delta tensor interface 没有实现：`true_custom_branch_delta_implemented = 0`。
8. P3 exact-signal controller 全部 `official_eligible = 0`；`C0/C6` 只是 diagnostic/oracle reference，不能进入 official route。
9. P4 oracle support 仍通过：precision = `1.0`，coverage = `0.12169312169312169`，bad-event = `0.0`；但 measured signal strata = `3`，support measurement pass = `0`。
10. P5-P9 因 `P2_true_custom_branch_delta_not_implemented` 全部 gate-blocked，均以 `not_run` 落盘。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9255_lower_level_cuda_branch_delta_extension_exact_signal_controller_closure.py` | v9.2.55 runner；复现 v9.2.54 boundary，执行 K7 residual attribution、CUDA/Triton custom branch-delta matrix、exact-signal controller boundary、support/downstream/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9255_lower_level_cuda_branch_delta_extension_exact_signal_controller_closure.py
```

编译依赖补齐：

```text
python -m pip install ninja
```

说明：第一次编译 CUDA extension 时环境缺少 `ninja`；补齐后重新正式运行，本文只记录 `20260512T071500Z` artifact 的最终结果。

正式运行：

```bash
python experiments/run_v9255_lower_level_cuda_branch_delta_extension_exact_signal_controller_closure.py \
  --out-dir results/real_rerun_20260506/v9255_lower_level_cuda_branch_delta_extension_exact_signal_controller_closure_first_20260512T071500Z \
  --fresh --device auto --data-root data --seed 1314
```

实际参数来自 `run_manifest.json`：

```text
device = cuda
triton_available = true
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7
microprobe_steps = 168
train_size = 2048
batch_size = 64
hidden_dim = 256
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R9-TrueCustomExtensionNotImplemented",
  "base_candidate": "LQ-t2-h256",
  "v9254_boundary_pass": 1,
  "k7_residual_attribution_pass": 1,
  "dominant_k7_subphase": "K7d-control_gain_compute",
  "custom_branch_delta_implemented": 1,
  "true_custom_branch_delta_implemented": 0,
  "best_custom_delta_id": "CBD0-V9254Reference",
  "uses_true_branch_delta": 0,
  "uses_formula_proxy": 0,
  "custom_delta_predictivity_pass": 1,
  "custom_delta_agreement_pass": 1,
  "custom_delta_system_pass": 0,
  "custom_delta_auc": 0.8884008136827201,
  "custom_delta_corr": -0.010621839689848079,
  "custom_delta_accept_agreement": 1.0,
  "custom_delta_step_ratio_q90": 2.8632798851361203,
  "custom_delta_memory_ratio": 0.9695007261731864,
  "formula_proxy_negative_control_pass": 1,
  "best_controller_id": "C0-V9253Reference",
  "exact_signal_controller_pass": 0,
  "controller_auc": 1.0,
  "controller_corr": 0.36769206823311557,
  "accepted_precision": 1.0,
  "accepted_coverage": 0.043402777777777776,
  "accepted_bad_event_rate": 0.0,
  "oracle_support_pass": 1,
  "oracle_precision": 1.0,
  "oracle_coverage": 0.12169312169312169,
  "oracle_bad_event": 0.0,
  "support_measurement_pass": 0,
  "natural_real_event_count": 12096,
  "balanced_diagnostic_real_event_count": 36,
  "measured_signal_strata_count": 3,
  "measured_family_count": 45,
  "primary_blocker": "true_branch_delta_extension_not_implemented",
  "next_required_implementation": "implement_true_triton_or_cuda_branch_delta_kernel_with_logits"
}
```

判断：本轮确实有 custom CUDA extension row，但它不是 true branch-delta kernel。route 正确停在 `R9-TrueCustomExtensionNotImplemented`，不能用 CUDA extension presence 代替 branch-delta correctness。

## 3. P1 K7 residual subphase attribution

Artifacts：

```text
p1_k7_residual_subphase_attribution.csv
k7_residual_trace_v9255.csv
```

| subphase | time ms | ratio | note |
|---|---:|---:|---|
| K7d-control_gain_compute | `11.403963` | `0.866134` | dominant |
| K7a-branch_logits_gather | `1.300185` | `0.098749` | branch/logit gather |
| K7b-ce_margin_selected_metric | `0.146712` | `0.011143` | selected metric |
| K7k-logging_hash_timestamp | `0.094423` | `0.007171` | logging/hash |
| K7i-temp_allocation | `0.062044` | `0.004712` | temp allocation |
| K7j-kernel_launch_sync | `0.051618` | `0.003920` | sync |

Summary：

```text
dominant_k7_subphase = K7d-control_gain_compute
dominant_k7_subphase_ratio = 0.8661336752717377
unknown_fraction = 0.0
k7_subphase_sum_close_to_k7 = 1
k7_residual_attribution_pass = 1
```

判断：v9.2.54 的 coarse `K7-gap_risk_support_reduction` 被进一步拆开，本轮 dominant 是 control-gain compute。该结论来自 real CUDA run 的落盘 timing trace，不是未解释 overhead。

## 4. P2 true custom branch-delta implementation matrix

Artifacts：

```text
p2_true_custom_branch_delta_implementation_matrix.csv
true_custom_branch_delta_trace_v9255.csv
formula_negative_control_trace_v9255.csv
system_custom_kernel_overhead_trace_v9255.csv
custom_kernel_memory_traffic_trace_v9255.csv
```

| custom delta | status | CUDA ext | true branch delta | formula proxy | AUC | agreement | step q90 | precision | bad-event | official pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CBD0-V9254Reference | measured v9254 reference | `0` | `0` | `0` | `0.888401` | `1.000000` | `2.863280` | `0.432507` | `0.548209` | `0` |
| CBD1-TritonFormulaNegativeControl | implemented Triton formula proxy | `0` | `0` | `1` | `0.534988` | `0.386822` | `1.350000` | `0.223140` | `0.079890` | `0` |
| CBD2-CUDASelectedLogitBranchDeltaExtension | not implemented: no true selected-logit delta tensors | `0` | `0` | `0` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0` |
| CBD3-CUDAFullLogitSmallCBranchDeltaExtension | not implemented: no full-logit tensors in artifact | `0` | `0` | `0` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0` |
| CBD4-CUDABranchDeltaMetricFusedExtension | implemented CUDA extension on source-measured gap input | `1` | `0` | `0` | `0.491345` | `0.422619` | `1.023641` | `0.137741` | `0.123967` | `0` |
| CBD5-TritonV2TrueBranchDelta | not implemented: no branch-delta tensor interface | `0` | `0` | `0` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0` |
| CBD6-HybridCustomBranchDelta | diagnostic Triton formula plus CUDA gap input | `1` | `0` | `1` | `0.524951` | `0.399306` | `1.920000` | `0.187328` | `0.057851` | `0` |

P2 summary：

```text
custom_branch_delta_implemented = 1
true_custom_branch_delta_implemented = 0
best_custom_delta_id = CBD0-V9254Reference
custom_delta_predictivity_pass = 1
custom_delta_agreement_pass = 1
custom_delta_system_pass = 0
formula_proxy_negative_control_pass = 1
```

判断：

1. `CBD4` 证明 lower-level CUDA extension 可以在本环境编译并调用，但它只消费 source-measured gap/risk/support scalar，不计算 true branch-output displacement。
2. `CBD1` 是 fast negative control：速度进入 envelope，但 signal/agreement 丢失。
3. `CBD0` 仍是 strongest exact-signal reference，但太贵且不是新 custom kernel。
4. P2 terminal blocker 是 `true_custom_branch_delta_implemented = 0`，不是 Triton/CUDA 完全不可用。

## 5. P3 exact-signal controller boundary

Artifacts：

```text
p3_exact_signal_controller_calibration.csv
exact_signal_controller_trace_v9255.csv
```

| controller | official | AUC heldout | corr heldout | precision | coverage | bad-event | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0-V9253Reference | `0` | `1.000000` | `0.367692` | `1.000000` | `0.043403` | `0.000000` | `0` |
| C1-AllPassBranchDeltaController | `0` | `0.496877` | `0.178845` | `0.111350` | `0.192295` | `0.438951` | `0` |
| C3-BorderlineExactController | `0` | `0.506769` | `0.203755` | `0.112827` | `0.192708` | `0.434148` | `0` |
| C5-ParetoCostAwareController | `0` | `0.673403` | `0.106672` | `0.179421` | `0.159888` | `0.304550` | `0` |
| C6-Oracle | `0` | `1.000000` | `0.367692` | `1.000000` | `0.043403` | `0.000000` | `0` |

判断：由于 P2 没有 true custom branch-delta survivor，所有 controller 都保持 `official_eligible = 0`。`C0/C6` 只证明 reference/oracle good events 仍存在，不能写成 exact-signal controller success。

## 6. P4 support 与 downstream boundary

P4 support：

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

P5-P9 均落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p5_leave_dataset_and_stratum_out.csv` | `P2_true_custom_branch_delta_not_implemented` |
| `p6_official_paired_replay.csv` | same |
| `p7_short_run_functional_validation.csv` | same |
| `p8_full_10seed_functional_validation.csv` | same |
| `p9_robustness_external_ready.csv` | same |

判断：oracle support 仍高，但 support strata 仍过窄，且 P2 terminal gate 未过；因此不能打开 LDO/LSO、paired replay、short-run、full-run 或 external-ready。

## 7. No-fake audit

```text
rows_checked = 24354
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

## 8. Hash

| artifact | SHA256 |
|---|---|
| plan | `44bccd374135bcc63e72c2fc5eb6fe1e14316e38eebae1f6b00fd80a306f9362` |
| runner | `f0edb40ddad9d2523c50015244cb06d39d7f0c11db3da46134f20f7270f2c6ba` |
| run manifest | `611e6add4315c10ad40a568ab58c16078f6cc934ff83cd30a78c12f19d1ddcf5` |
| route | `af1ba4842ecb3f9899dca8b4c8d1e75191d2867db24d775bad31445629ab785b` |
| P0 boundary | `11b64372d101f0cde990972786611e2239ad3fe18c04ef8fdeb01eaa39b7e550` |
| P1 K7 attribution | `ebd626a8be0353fd7cf1302338d3b73083eaa994c4168fd775217c0e425ae185` |
| P2 true custom branch-delta | `ac5f697c040a53696490f931dd6c78ab1cbe5848e0a6e7c95df1ac9c373ba187` |
| P3 controller | `28d750cb60d1994712a2d87fc0a31f001d065fbc696d8a868701454fe0ed9711` |
| P4 support expansion | `ba86e35c4c095026bcda3cc753118fb37bd23688b1e3e253a8837f60760dcdcc` |
| P5 boundary | `1cc148a1af83d32a5aa61785fb88a7c9b451980bab84fb7e58ddc6aab3ee8785` |
| P6 boundary | `83bfe6433100270607d6781bb02b55cd695003f8106367387a6ca758da61cdb6` |
| P7 boundary | `f804f23ff5c1bc472c0f62737d02eeced5fe14a2ef84109e4d301911bb943aca` |
| P8 boundary | `aba4d33820d67a2eb6baf0f6b46bb838c2f119bb62fec1ad211b680be64d9467` |
| P9 boundary | `93102d96f2690a59f1a5651cdcdb4540c3981938a4cd659268893eb70eba5b69` |
| failure table | `2aa86c31fa67371c8d9e45784b58735e52d0b4e1f8b131d89b539444c98abdd0` |
| provenance audit | `cdf052c905958e2a5adcf2fe9828e426156ac9fac539116adff7bace4e4effa0` |

## 9. 最终分析结论

v9.2.55 的真实推进是：

```text
v9.2.54: Triton formula kernel 速度够，但不是 true branch-delta，agreement/signal 丢失。
v9.2.55: CUDA extension 编译/调用路径已打通；
          但当前 implemented extension 仍不是 true branch-logit / full-logit delta kernel。
```

机制判断：

1. H1 得到推进：K7 residual 已进一步归因，dominant 是 `K7d-control_gain_compute`。
2. H2 没有闭合：`CBD4` 是真实 CUDA extension，但它使用 source-measured scalar gap input，不能算 true branch-delta。
3. H3 成立为 negative control：`CBD1` 低成本但 AUC/agreement 不足，不能替代 exact signal。
4. H4 成立：oracle support 仍存在，precision `1.0`，coverage `0.121693`，bad-event `0.0`。
5. 当前不能继续靠 controller threshold 或公式 proxy；下一步必须提供真实 branch logits / selected logits / full logits tensor interface，并在同一 kernel 上同时过 agreement、system、controller gate。

最终一句话：

> v9.2.55 真实执行后停在 `R9-TrueCustomExtensionNotImplemented`：CUDA extension 路径已打通，但 true branch-delta logits kernel 仍未实现，strict PureKAN functional 仍未成功。
