# DG-KAN v9.2.54 Custom Fused Branch-Delta Kernel 与 Exact-Signal Controller Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.2.54_CustomFusedBranchDeltaKernel_ExactSignalControllerClosure_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R14-NeedsLowerLevelCUDAExtension
base_candidate = LQ-t2-h256
success_v9254_strict_purekan_functional = False
success_v9254_full_functional = False
success_v9254_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9254_custom_fused_branch_delta_kernel_exact_signal_controller_closure_first_20260512T061500Z/
```

核心结论：

1. P0 复现 v9.2.53 boundary：source route = `R12-FusedBranchDeltaPredictiveButNeedsCustomKernel`，D0 vectorization pass = `1`，FBD6 system pass = `0`，oracle support pass = `1`，fake/proxy = `0`。
2. 本轮 manifest 记录 `device = cuda`；运行中 GPU 采样有活动：`NVIDIA L4, 33 %, 326 MiB`、`32 %, 326 MiB`、`30 %, 326 MiB`。
3. P1 FBD6 residual cost attribution pass = `1`，dominant residual subphase = `K7-gap_risk_support_reduction`，ratio = `0.662327`。
4. P2 reference exact branch-delta signal 仍强：`CBD0-FBD6-D0VectorizedFusedDeltaReference` AUC = `0.888401`，agreement = `1.0`，但 step ratio = `2.863280 > 1.50`，system pass = `0`。
5. P2 低成本 Triton selected-logit formula 已测，但不是 true exact branch-delta kernel：`CBD1` step ratio = `1.35`，agreement = `0.386822`，AUC = `0.534988`，official = `0`。
6. P2 true custom branch-delta implemented = `0`；`CBD5-CUDAExtensionBranchDelta` 明确 `not_implemented_cuda_extension_required`。
7. P3 controller 全部 `official_eligible = 0`；`C0/V9253Reference` 的 precision/coverage/bad-event 只是 diagnostic upper reference，不进入 official route。
8. P4 oracle support 仍通过：precision = `1.0`，coverage = `0.121693`，bad-event = `0.0`；但 measured signal strata = `3`，support measurement pass = `0`。
9. P5-P9 因 `P2_true_custom_branch_delta_not_implemented` 全部 gate-blocked，均以 `not_run` 落盘。
10. 当前 blocker：`triton_formula_kernel_not_true_branch_delta_kernel`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9254_custom_fused_branch_delta_kernel_exact_signal_controller_closure.py` | v9.2.54 runner；复现 v9.2.53 boundary，执行 FBD6 residual attribution、Triton/custom branch-delta matrix、exact-signal controller boundary、support/downstream/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9254_custom_fused_branch_delta_kernel_exact_signal_controller_closure.py
```

正式运行：

```bash
python experiments/run_v9254_custom_fused_branch_delta_kernel_exact_signal_controller_closure.py \
  --out-dir results/real_rerun_20260506/v9254_custom_fused_branch_delta_kernel_exact_signal_controller_closure_first_20260512T061500Z \
  --fresh --device auto --data-root data --seed 1314
```

运行中 GPU 采样：

```text
NVIDIA L4, 33 %, 326 MiB
NVIDIA L4, 32 %, 326 MiB
NVIDIA L4, 30 %, 326 MiB
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
  "route": "R14-NeedsLowerLevelCUDAExtension",
  "base_candidate": "LQ-t2-h256",
  "v9253_boundary_pass": 1,
  "fbd6_residual_attribution_pass": 1,
  "dominant_fbd6_residual_subphase": "K7-gap_risk_support_reduction",
  "triton_available": 1,
  "custom_branch_delta_implemented": 1,
  "true_custom_branch_delta_implemented": 0,
  "best_custom_delta_id": "CBD0-FBD6-D0VectorizedFusedDeltaReference",
  "custom_delta_predictivity_pass": 1,
  "custom_delta_agreement_pass": 1,
  "custom_delta_system_pass": 0,
  "custom_delta_auc": 0.8884008136827201,
  "custom_delta_corr": -0.010621839689848079,
  "custom_delta_accept_agreement": 1.0,
  "custom_delta_step_ratio_q90": 2.8632798851361203,
  "custom_delta_memory_ratio": 0.9695007261731864,
  "exact_signal_controller_pass": 0,
  "support_measurement_pass": 0,
  "oracle_support_pass": 1,
  "oracle_precision": 1.0,
  "oracle_coverage": 0.12169312169312169,
  "oracle_bad_event": 0.0,
  "primary_blocker": "triton_formula_kernel_not_true_branch_delta_kernel",
  "next_required_implementation": "implement_true_triton_or_cuda_branch_delta_kernel_with_logits"
}
```

判断：本轮有 Triton formula / diagnostic kernel 测量，但没有 true exact fused branch-delta kernel。route 因此停在 `R14`，不能把 selected-logit formula 写成 system-legal exact branch-delta。

## 3. P1 FBD6 residual cost attribution

Artifacts：

```text
p1_fbd6_residual_cost_attribution.csv
fbd6_residual_trace_v9254.csv
```

Summary：

```text
dominant_fbd6_residual_subphase = K7-gap_risk_support_reduction
dominant_fbd6_residual_ratio = 0.6623272214812176
unknown_fraction = 0.0
subphase_time_sum_close_to_fbd6 = 1
fbd6_residual_attribution_pass = 1
```

判断：FBD6 residual cost 已归因，主要仍落在 gap/risk/support reduction，而不是未解释开销或 CPU offload。

## 4. P2 custom fused branch-delta kernel matrix

Artifacts：

```text
p2_custom_fused_branch_delta_kernel_matrix.csv
custom_branch_delta_kernel_trace_v9254.csv
system_custom_kernel_overhead_trace_v9254.csv
custom_kernel_memory_traffic_trace_v9254.csv
```

| custom delta | status | official | AUC | agreement | step q90 | precision | coverage | bad-event | system |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CBD0-FBD6-D0VectorizedFusedDeltaReference | `measured_fbd6_reference` | `0` | `0.888401` | `1.000000` | `2.863280` | `0.432507` | `0.030010` | `0.548209` | `0` |
| CBD1-TritonSelectedLogitBranchDelta | `implemented_triton_selected_logit_formula_not_exact_branch_delta` | `0` | `0.534988` | `0.386822` | `1.350000` | `0.223140` | `0.030010` | `0.079890` | `0` |
| CBD2-TritonFullLogitSmallCBranchDelta | `not_implemented_no_full_logit_tensors_in_artifact` | `0` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0` |
| CBD3-TritonBranchDeltaMetricFused | `diagnostic_triton_metric_fusion_uses_measured_gap_input` | `0` | `0.491345` | `0.422619` | `1.431163` | `0.137741` | `0.030010` | `0.123967` | `0` |
| CBD4-BorderlineExactFallback | `hybrid_borderline_exact_diagnostic` | `0` | `0.536701` | `0.387897` | `2.100000` | `0.223140` | `0.030010` | `0.079890` | `0` |
| CBD5-CUDAExtensionBranchDelta | `not_implemented_cuda_extension_required` | `0` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0` |
| CBD6-HybridCustomBranchDelta | `diagnostic_triton_selected_plus_gap_input_not_official` | `0` | `0.538166` | `0.388889` | `1.920000` | `0.223140` | `0.030010` | `0.077135` | `0` |

判断：

1. `CBD0` 保留 exact signal，但只是 v9.2.53 reference continuation，太贵且不是 custom fused kernel。
2. `CBD1` 是真实 Triton formula run，进入 step envelope，但它不是 exact branch-delta，agreement 只有 `0.386822`，不能 official。
3. `CBD3/CBD6` 使用 measured gap input 或 diagnostic hybrid，不能作为 commit-time official kernel。
4. 真正需要的 full-logit/custom CUDA extension route 没有实现，因此 `true_custom_branch_delta_implemented = 0`。

## 5. P3 exact-signal controller boundary

Artifacts：

```text
p3_exact_signal_controller_calibration.csv
exact_signal_controller_trace_v9254.csv
```

| controller | official | AUC heldout | corr heldout | precision | coverage | bad-event | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0-V9253Reference | `0` | `1.000000` | `0.367692` | `1.000000` | `0.043403` | `0.000000` | `0` |
| C1-AllPassBranchDeltaController | `0` | `0.496877` | `0.178845` | `0.111350` | `0.192295` | `0.438951` | `0` |
| C2-BroadCandidateBranchDeltaController | `0` | `0.500450` | `0.189964` | `0.111972` | `0.191964` | `0.437123` | `0` |
| C3-BorderlineExactController | `0` | `0.506769` | `0.203755` | `0.112827` | `0.192708` | `0.434148` | `0` |
| C4-FamilyBalancedBranchDeltaController | `0` | `0.498029` | `0.179044` | `0.111831` | `0.191468` | `0.436528` | `0` |
| C5-ParetoCostAwareController | `0` | `0.673403` | `0.106672` | `0.179421` | `0.159888` | `0.304550` | `0` |
| C6-Oracle | `0` | `1.000000` | `0.367692` | `1.000000` | `0.043403` | `0.000000` | `0` |

判断：由于 P2 没有 true custom branch-delta system survivor，P3 全部 `official_eligible = 0`。`C0/C6` 只说明 oracle/reference safe-good still exists，不是 exact-signal controller success。

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

判断：oracle support 仍高，但 support strata 仍过窄，且 P2 terminal gate 未过；因此不能打开 leave-out、paired replay、short-run、full-run 或 external-ready。

## 7. No-fake audit

```text
rows_checked = 24346
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
| plan | `0d9d1749b5c14e01934156bc524104281082121c2df5d993d6900807a400973c` |
| runner | `29a6a9e43443940f5771b988065103a360e098e8375d10716e6fc78b19a94b19` |
| run manifest | `1da81ae70fd34161e6f5e6c6133d8bca9693b7f57140a0803e1afddf1c905ff3` |
| route | `7604848a59276bb856d56a6b4d57be066c57e9dd0c7c914f6684e16c06e0c3f6` |
| P0 boundary | `9e684b7eb5ed19da3a6e67a9b69af40fcd833ff944474dc13ddbc2ad93872de1` |
| P1 residual attribution | `97c8aa34082726fd7d1335dedd637a600672ae76ebe33de5ab3148da751797be` |
| P2 custom kernel matrix | `a7d54539c601cbf2509784a27b6227b3f13b7c355df3cb15000c5033e61cc5f1` |
| P3 controller | `c567f510bdfc23bed392ed8f5fd14c88622e9d9991dd791b81bb6e04a91709c4` |
| P4 support expansion | `aa00de066276b4bd9e57fe1a947f87bd779dbc505e92b289235e5ef754c33d54` |
| P5 leave-out boundary | `1cc148a1af83d32a5aa61785fb88a7c9b451980bab84fb7e58ddc6aab3ee8785` |
| P6 paired replay boundary | `83bfe6433100270607d6781bb02b55cd695003f8106367387a6ca758da61cdb6` |
| P7 short-run boundary | `f804f23ff5c1bc472c0f62737d02eeced5fe14a2ef84109e4d301911bb943aca` |
| P8 full-run boundary | `aba4d33820d67a2eb6baf0f6b46bb838c2f119bb62fec1ad211b680be64d9467` |
| P9 robustness boundary | `93102d96f2690a59f1a5651cdcdb4540c3981938a4cd659268893eb70eba5b69` |
| failure table | `e9d083e3dd107c592c499fc41cdb0d7ae7c4142f623858a78edc634daa2792cd` |
| provenance audit | `f32227b70c45f47a5c53baaa323fd42d5c99880b4f15a4360c02c8548be7c91c` |

## 9. 最终分析结论

v9.2.54 的真实推进是：

```text
v9.2.53: D0 vectorization pass，exact-like FBD6 signal strong，
          但 fused branch-delta system path 仍太贵。
v9.2.54: 继续尝试 custom/Triton branch-delta kernel；
          reference exact signal 仍强，但 true custom exact branch-delta kernel 没有实现。
```

机制判断：

1. H1 部分成立：FBD6 residual cost 已归因，dominant 是 `K7-gap_risk_support_reduction`。
2. H2 没有闭合：`CBD1` 是真实 Triton formula，但不是 exact branch-delta；它低成本，却丢失 agreement 和 safe-good signal。
3. H3 没有成立：没有 true custom branch-delta implementation，因此 exact-signal controller 全部保持 unofficial。
4. H4 成立：oracle support 仍然存在，precision `1.0`，coverage `0.121693`，bad-event `0.0`。
5. 当前不能继续靠 threshold 或 selected-logit formula 修；下一步需要真正实现 full-logit/branch-delta 的 lower-level Triton/CUDA extension，且在同一 candidate 上同时过 agreement、system、controller gate。

最终一句话：

> v9.2.54 真实执行后停在 `R14-NeedsLowerLevelCUDAExtension`：reference exact branch-delta signal 仍强，但没有 true custom fused branch-delta kernel，strict PureKAN functional 仍未成功。
