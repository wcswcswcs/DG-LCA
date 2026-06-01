# DG-KAN v12.15 B320Locked LossAgnosticSignalEstimator PrimitiveInstrumentation 执行复盘

生成时间：`2026-05-24T01:49:32Z`

## 1. 执行入口

```text
script = experiments/run_v1215_b320locked_lossagnostic_signal_estimator_primitive_instrumentation.py
plan_doc = docs/DG-KAN_v12.15_B320Locked_LossAgnosticSignalEstimator_PrimitiveInstrumentation_独立分析与下一步计划.md
out_dir = results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/official_3x3_b32_w5
conda_env = kan
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
splits = 1
windows = 5
functional_batch_size = 32
```

## 2. 复现命令

```bash
PYTHONHASHSEED=0 conda run -n kan python experiments/run_v1215_b320locked_lossagnostic_signal_estimator_primitive_instrumentation.py --probe-device auto --probe-datasets MNIST,Fashion-MNIST,KMNIST --probe-seeds 0,1,2 --probe-splits 1 --functional-batch-size 32 --windows 5 --probe-train-size 512 --probe-val-size 256 --probe-test-size 256 --out-dir results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/official_3x3_b32_w5
```

## 3. 生成 artifact

```text
v1215_anchor_monitor.csv
v1215_linec_response_identifiability.csv
v1215_linec_response_model.csv
v1215_actuator_truth.csv
v1215_b15_p3_candidates.csv
v1215_b15_p3_summary.csv
v1215_p4_short_run.csv
v1215_classic_family_status.csv
v1215_loss_agnostic_audit.csv
v1215_provenance_audit.csv
v1215_failure_table.csv
v1215_route_decision.json
v1215_hash_manifest.json
v1215_lineD_focused_repair_candidates.csv
v1215_lineD_focused_repair_family_status.csv
v1215_lineD_focused_repair_route.json
v1215_lineD_focused_repair_hash_manifest.json
```

## 3.1 Line D 执行状态

原始执行漏掉了 Batch D focused repair，只生成了 status-only monitor。发现漏项后，已补跑 Line D official focused repair，并将 v1283 raw artifacts 汇总回 v12.15 official 主目录。

候选选择遵守 v12.15 限制：不跑 B-spline；不做全 family 大扫；每个 active family 最多两个 focused repair。

```text
Rational = B7lu, B7lv
RBF = B2aa, B2ab
Chebyshev = B3an, B3ao
Fourier = B4v, B4w
Wavelet = B5p, B5q
BSpline = frozen / not run
```

smoke 命令：

```bash
LINED_IDS="B7lu-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-crossSignalR4-freezeBackbone-hiddenBias-manualAdamW-L3,B7lv-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-crossSignalR8-freezeBackbone-hiddenBias-manualAdamW-L3,B2aa-GaussianRBF-K4-orthoInputrot2L4P256-tritonL3,B2ab-GaussianRBF-K4-orthoInputsqL4P128-tritonL3,B3an-ChebyKAN-K3-h112-inputcrossL4P128-linearraw025-tritonL3-gradbuf,B3ao-ChebyKAN-K3-h112-inputcrossL4P128-linearraw010-tritonL3-gradbuf,B4v-FourierKAN-lowfreq-K4-h8-linearres050-gemmDirectL3,B4w-FourierKAN-lowfreq-K4-h8-linearres050-tritonL3-matmulTile,B5p-HatWaveletKAN-K4-inputcrossL4P128-linearraw002-tritonL3,B5q-HatWaveletKAN-K4-inputcrossL4P128-linearraw001-tritonL3"
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 conda run -n kan python experiments/run_v1283_b109_classic_family_functional_geometry.py --out-dir results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/lineD_smoke_focused_10 --device cuda:0 --datasets MNIST --seeds 0 --train-size 128 --val-size 64 --test-size 64 --batch-size 64 --epochs 1 --kernel-warmup-steps 1 --kernel-measure-steps 2 --task-compile-warmup-steps 1 --coupling-batch-size 16 --functional-batch-size 16 --sketch-batch-size 4 --expression-train-size 128 --expression-val-size 64 --expression-test-size 64 --expression-batch-size 64 --expression-steps-b0 2 --expression-steps-b1 2 --expression-steps-b2 2 --family-candidate-ids "$LINED_IDS" --family-promotion-ids "$LINED_IDS" --family-linec-ids "$LINED_IDS" --repair-candidate-ids '' --report-path results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/lineD_smoke_focused_10/report.md
```

official 命令：

```bash
LINED_IDS="B7lu-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-crossSignalR4-freezeBackbone-hiddenBias-manualAdamW-L3,B7lv-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-crossSignalR8-freezeBackbone-hiddenBias-manualAdamW-L3,B2aa-GaussianRBF-K4-orthoInputrot2L4P256-tritonL3,B2ab-GaussianRBF-K4-orthoInputsqL4P128-tritonL3,B3an-ChebyKAN-K3-h112-inputcrossL4P128-linearraw025-tritonL3-gradbuf,B3ao-ChebyKAN-K3-h112-inputcrossL4P128-linearraw010-tritonL3-gradbuf,B4v-FourierKAN-lowfreq-K4-h8-linearres050-gemmDirectL3,B4w-FourierKAN-lowfreq-K4-h8-linearres050-tritonL3-matmulTile,B5p-HatWaveletKAN-K4-inputcrossL4P128-linearraw002-tritonL3,B5q-HatWaveletKAN-K4-inputcrossL4P128-linearraw001-tritonL3"
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 conda run -n kan python experiments/run_v1283_b109_classic_family_functional_geometry.py --out-dir results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/lineD_official_focused_10_3x3 --fresh --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --test-size 256 --batch-size 128 --epochs 3 --kernel-warmup-steps 5 --kernel-measure-steps 12 --task-compile-warmup-steps 4 --coupling-batch-size 32 --functional-batch-size 32 --sketch-batch-size 8 --expression-train-size 1024 --expression-val-size 512 --expression-test-size 512 --expression-batch-size 128 --expression-steps-b0 60 --expression-steps-b1 600 --expression-steps-b2 1200 --family-candidate-ids "$LINED_IDS" --family-promotion-ids "$LINED_IDS" --family-linec-ids "$LINED_IDS" --repair-candidate-ids '' --report-path results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/lineD_official_focused_10_3x3/report.md
```

v12.15 汇总命令：

```bash
conda run -n kan python -m py_compile experiments/summarize_v1215_lined_focused_repair.py
conda run -n kan python experiments/summarize_v1215_lined_focused_repair.py --raw-dir results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/lineD_official_focused_10_3x3 --out-dir results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/official_3x3_b32_w5
```

Line D 补跑 artifact：

```text
raw_dir = results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/lineD_official_focused_10_3x3
v1215_lineD_focused_repair_candidates.csv
v1215_lineD_focused_repair_family_status.csv
v1215_lineD_focused_repair_route.json
v1215_lineD_focused_repair_hash_manifest.json
```

原因审计：

```text
计划没有漏掉 Line D。计划第 6 节写明 Line D 独立并行，第 15 节 Batch D 写明 Classic No-BSpline focused repair 并行执行。
执行时我把第 13 节标题中的 Portfolio Monitor 理解得过窄，只实现了 v1215_classic_family_status.csv 状态继承，没有实现 Batch D 的 focused repair。
该漏项已在本次补跑中修复；原 v1215_classic_family_status.csv 仍保留为 status-only monitor，补跑结果以 v1215_lineD_focused_repair_* 为准。
```

## 4. 修改审计

```text
新增 experiments/run_v1215_b320locked_lossagnostic_signal_estimator_primitive_instrumentation.py。
smoke run 暴露 weak-promotion gate 会在非 3x3 设置误开 P4；已修复为 expected_dataset_seed_count >= 9 才允许 strong/weak promotion。
official route 判定按 v12.15 停止条件收紧：B15 全 0、noise/reservoir 未过 -0.01 且 control_gap 全负时写 R4。
未修改 CE loss、sampler、class weight、teacher/distillation、dataset branch 或 B320 architecture。
B320 anchor monitor 复用 v12.14 locked anchor artifact，并记录 source sha256。
official B15 direction generator 不使用 CE vector / label-loss VJP / permuted-label CE surrogate。
新增 experiments/summarize_v1215_lined_focused_repair.py，用于把 v1283 Line D raw artifacts 汇总为 v1215_lineD_focused_repair_*。
Line D 漏跑已补：补跑 10 个 focused candidates，未改 CE loss、sampler、class weight、teacher/distillation、dataset branch 或 B-spline frozen policy。
```

## 5. Route

```text
route = R4-LossAgnosticFunctionalMechanismNotFound
p4_open = 0
p3_survivor_count = 0
next_recommended_action = stop current cotangent/role/moment mechanism family; move to higher-level estimator or Line C sketch redesign
```

## 6. Artifact Hash

| artifact | sha256 |
|---|---|
| `fig_v1215_actuator_failure_taxonomy.svg` | `916ff2bce2648f5f9a3ebced22175d4b4e3cda290d4a61d61317a02099666888` |
| `fig_v1215_actuator_role_response_heatmap.svg` | `916ff2bce2648f5f9a3ebced22175d4b4e3cda290d4a61d61317a02099666888` |
| `fig_v1215_classic_family_status.svg` | `916ff2bce2648f5f9a3ebced22175d4b4e3cda290d4a61d61317a02099666888` |
| `fig_v1215_controls_gap_matrix.svg` | `916ff2bce2648f5f9a3ebced22175d4b4e3cda290d4a61d61317a02099666888` |
| `fig_v1215_coupling_vs_noise_reservoir_scatter.svg` | `916ff2bce2648f5f9a3ebced22175d4b4e3cda290d4a61d61317a02099666888` |
| `fig_v1215_fail_reason_heatmap.svg` | `916ff2bce2648f5f9a3ebced22175d4b4e3cda290d4a61d61317a02099666888` |
| `fig_v1215_loss_agnostic_audit_matrix.svg` | `916ff2bce2648f5f9a3ebced22175d4b4e3cda290d4a61d61317a02099666888` |
| `fig_v1215_noise_reservoir_effect_size_by_method.svg` | `916ff2bce2648f5f9a3ebced22175d4b4e3cda290d4a61d61317a02099666888` |
| `fig_v1215_p3_survivor_pareto.svg` | `916ff2bce2648f5f9a3ebced22175d4b4e3cda290d4a61d61317a02099666888` |
| `fig_v1215_p4_short_run_curves.svg` | `916ff2bce2648f5f9a3ebced22175d4b4e3cda290d4a61d61317a02099666888` |
| `fig_v1215_param_delta_vs_sketch_delta.svg` | `916ff2bce2648f5f9a3ebced22175d4b4e3cda290d4a61d61317a02099666888` |
| `fig_v1215_projector_angle_by_role.svg` | `916ff2bce2648f5f9a3ebced22175d4b4e3cda290d4a61d61317a02099666888` |
| `fig_v1215_projector_angle_vs_noise_release.svg` | `916ff2bce2648f5f9a3ebced22175d4b4e3cda290d4a61d61317a02099666888` |
| `fig_v1215_response_precision_recall.svg` | `916ff2bce2648f5f9a3ebced22175d4b4e3cda290d4a61d61317a02099666888` |
| `fig_v1215_response_pred_vs_actual_coupling.svg` | `916ff2bce2648f5f9a3ebced22175d4b4e3cda290d4a61d61317a02099666888` |
| `fig_v1215_response_pred_vs_actual_noise.svg` | `916ff2bce2648f5f9a3ebced22175d4b4e3cda290d4a61d61317a02099666888` |
| `fig_v1215_response_pred_vs_actual_reservoir.svg` | `916ff2bce2648f5f9a3ebced22175d4b4e3cda290d4a61d61317a02099666888` |
| `fig_v1215_role_energy_vs_response.svg` | `916ff2bce2648f5f9a3ebced22175d4b4e3cda290d4a61d61317a02099666888` |
| `fig_v1215_summary_route.svg` | `916ff2bce2648f5f9a3ebced22175d4b4e3cda290d4a61d61317a02099666888` |
| `v1215_actuator_truth.csv` | `70a7f101e686f7eaecb99cbb6f893db71a9e21c69a65899e952ef3b04f43304c` |
| `v1215_anchor_monitor.csv` | `9e7100d3539e3ce633668863aa19cda3a63a555e77cdf5b1dca4185e45706c46` |
| `v1215_b15_p3_candidates.csv` | `02b8c9c9f43e19aff191991a1a6a0b096966e02111b5d10b109df0b7f2709fb2` |
| `v1215_b15_p3_method_summary.csv` | `33a3035705166e083e8bd3fb0fb845455a9c074a0b4163f77b2a6c1a2b111aec` |
| `v1215_b15_p3_summary.csv` | `ac4e91ec239187df04aa093c6c017fb5d6c45ad5cb2b5bf06357e473b9b7f46a` |
| `v1215_classic_family_status.csv` | `8c562e517ed1cf0378947147b07f63501d4a8235dd10dd4de740fbecfcc7cd10` |
| `v1215_failure_table.csv` | `b221bd0a0f1ed2dfb70503bbca424a5155d6b7aa102f97f9f7db0b0109d6d443` |
| `v1215_linec_response_identifiability.csv` | `8e8f6aaa449d9d1c55052fac47eafc0d28120ebd41127259236e5bb97a7ee0a7` |
| `v1215_linec_response_model.csv` | `9273afd4cd8eed7e97b3250da3a26853bff69e23a00b20226f4f21f14b3f22c6` |
| `v1215_linec_response_summary.csv` | `4f4f6ba7f20920021a358305d3a952375a4c9aaed0fa60d9a50bc41bf5548595` |
| `v1215_loss_agnostic_audit.csv` | `b88f245d11385a7859b0af823cdf1fa90f371e459edacaafbd9cd20b32d98b4a` |
| `v1215_p4_short_run.csv` | `82c77419344bf8f3f1ce49c141485b4affe2fc615fc8586bb18d04e8258e76ff` |
| `v1215_provenance_audit.csv` | `e7b3fe7b132d44515c82c96177352371927f73a9b678a071f11257fbbca7aae9` |
| `v1215_route_decision.json` | `658b03ae176638109a9266109710d55e5a84daf2b3e08cbbf1ac1b588903674e` |
| `v1215_lineD_focused_repair_candidates.csv` | `a632329dada2a70539a62756659fcb8bbef375e5d0762462d9207f39c0dc2fe8` |
| `v1215_lineD_focused_repair_family_status.csv` | `92e9b5617626d882a086cde45e50ab05a2e2d622168565097ace362878458d8e` |
| `v1215_lineD_focused_repair_route.json` | `93704a87e1c200466170ca73939b772d487a3c7227bbb60f7f5309cccf14b19d` |

## 7. v12.15 continuation 执行

生成时间：`2026-05-24T02:10:28Z`

这部分仍属于 v12.15，不开 v12.16。继续执行上一节 route 的推荐方向：

```text
stop current cotangent/role/moment mechanism family;
move to higher-level estimator or Line C sketch redesign;
deepen primitive instrumentation before more B15 candidate search.
```

新增脚本：

```text
experiments/run_v1215_continuation_actuator_budget_repair.py
experiments/run_v1215_continuation_p3_from_fused_actuator.py
```

### 7.1 编译检查

```bash
conda run -n kan python -m py_compile experiments/run_v1215_continuation_actuator_budget_repair.py
conda run -n kan python -m py_compile experiments/run_v1215_continuation_p3_from_fused_actuator.py
```

### 7.2 Blocker 与修复记录

第一次 actuator continuation smoke 失败：

```text
RuntimeError: shape '[64, 6, 10]' is invalid for input of size 3600
```

修复：

```text
将 continuation runner 内 cotangent count 从 64 收回到 v12.15 已验证过的 32。
原因：v12.14 cotangent construction 要求 count 不能超过当前 logit cotangent space 可 reshape 容量。
```

第二次 smoke 失败：

```text
AttributeError: module 'v1211_runner_for_v1213' has no attribute '_apply_delta'
```

修复：

```text
safety_cap_delta / logit_drift_on_batch 改为使用 v1252._apply_delta，
与 v12.15 official evaluate_direction_full 的实际 apply path 对齐。
```

随后 smoke 通过。

### 7.3 Actuator budget-only continuation

smoke 命令：

```bash
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 conda run -n kan python experiments/run_v1215_continuation_actuator_budget_repair.py --probe-device cuda:0 --probe-datasets MNIST --probe-seeds 0 --probe-splits 1 --functional-batch-size 16 --windows 5 --probe-train-size 128 --probe-val-size 64 --probe-test-size 64 --budget-multipliers 2,4 --out-dir results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/continuation_actuator_budget_repair_smoke --fresh
```

official 命令：

```bash
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 conda run -n kan python experiments/run_v1215_continuation_actuator_budget_repair.py --probe-device cuda:0 --probe-datasets MNIST,Fashion-MNIST,KMNIST --probe-seeds 0,1,2 --probe-splits 1 --functional-batch-size 32 --windows 5 --probe-train-size 512 --probe-val-size 256 --probe-test-size 256 --budget-multipliers 2,4,8 --out-dir results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/continuation_actuator_budget_repair_3x3_b32_w5 --fresh
```

输出目录：

```text
results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/continuation_actuator_budget_repair_3x3_b32_w5
```

生成 artifact：

```text
v1215_continuation_actuator_budget_repair.csv
v1215_continuation_actuator_budget_summary.csv
v1215_continuation_estimator_feature_table.csv
v1215_continuation_estimator_cv.csv
v1215_continuation_route_decision.json
v1215_continuation_hash_manifest.json
```

### 7.4 Fused primitive actuator continuation

修复方向：

```text
不重复 cotangent/role/moment 小网格；
新增 primitive_param_space fused actuator，直接在 B320 active primitive role 参数上构造 loss-agnostic perturbation；
仍使用 train-batch unlabeled logit drift cap；
不使用 CE vector / label-loss VJP / validation / dataset-name branch。
```

smoke 命令：

```bash
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 conda run -n kan python experiments/run_v1215_continuation_actuator_budget_repair.py --probe-device cuda:0 --probe-datasets MNIST --probe-seeds 0 --probe-splits 1 --functional-batch-size 16 --windows 5 --probe-train-size 128 --probe-val-size 64 --probe-test-size 64 --budget-multipliers 2,4 --out-dir results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/continuation_fused_actuator_smoke --fresh
```

official 命令：

```bash
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 conda run -n kan python experiments/run_v1215_continuation_actuator_budget_repair.py --probe-device cuda:0 --probe-datasets MNIST,Fashion-MNIST,KMNIST --probe-seeds 0,1,2 --probe-splits 1 --functional-batch-size 32 --windows 5 --probe-train-size 512 --probe-val-size 256 --probe-test-size 256 --budget-multipliers 2,4,8 --out-dir results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/continuation_fused_primitive_actuator_3x3_b32_w5 --fresh
```

输出目录：

```text
results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/continuation_fused_primitive_actuator_3x3_b32_w5
```

### 7.5 Gate-open fused actuator P3

gate-open actuator 来源：

```text
repair_id = K6-FusedQuadSignBudgetCap
role_recipe = primitive_param_sign:quad
budget_multiplier = 8
```

smoke 命令：

```bash
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 conda run -n kan python experiments/run_v1215_continuation_p3_from_fused_actuator.py --probe-device cuda:0 --probe-datasets MNIST --probe-seeds 0 --probe-splits 1 --functional-batch-size 16 --windows 5 --probe-train-size 128 --probe-val-size 64 --probe-test-size 64 --out-dir results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/continuation_p3_fused_quad_sign_smoke --fresh
```

official 命令：

```bash
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 conda run -n kan python experiments/run_v1215_continuation_p3_from_fused_actuator.py --probe-device cuda:0 --probe-datasets MNIST,Fashion-MNIST,KMNIST --probe-seeds 0,1,2 --probe-splits 1 --functional-batch-size 32 --windows 5 --probe-train-size 512 --probe-val-size 256 --probe-test-size 256 --out-dir results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/continuation_p3_fused_quad_sign_3x3_b32_w5 --fresh
```

输出目录：

```text
results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/continuation_p3_fused_quad_sign_3x3_b32_w5
```

### 7.6 Bidirectional noise/reservoir selector P3

修复方向：

```text
在 K6 fused quad sign actuator 的 + / - 方向上做 Line C selector probe；
selector score = -NoiseSignalLeak_delta - RealSignalReservoirRatio_delta + 0.10 * CouplingR2_delta - safety_penalty；
selector 只使用 Line C response，不使用 CE vector / label-loss VJP / validation / dataset-name branch；
target_logit_drift_train 从 0.045 降到 0.04，原因是 fixed P3 中 4/9 行 holdout logit drift 超过 0.05。
```

smoke 命令：

```bash
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 conda run -n kan python experiments/run_v1215_continuation_p3_from_fused_actuator.py --candidate-mode bidirectional_noise_reservoir_selector --selector-probe-budget-multiplier 2 --target-logit-drift 0.04 --probe-device cuda:0 --probe-datasets MNIST --probe-seeds 0 --probe-splits 1 --functional-batch-size 16 --windows 5 --probe-train-size 128 --probe-val-size 64 --probe-test-size 64 --out-dir results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/continuation_p3_bidir_selector_smoke --fresh
```

official 命令：

```bash
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 conda run -n kan python experiments/run_v1215_continuation_p3_from_fused_actuator.py --candidate-mode bidirectional_noise_reservoir_selector --selector-probe-budget-multiplier 2 --target-logit-drift 0.04 --probe-device cuda:0 --probe-datasets MNIST,Fashion-MNIST,KMNIST --probe-seeds 0,1,2 --probe-splits 1 --functional-batch-size 32 --windows 5 --probe-train-size 512 --probe-val-size 256 --probe-test-size 256 --out-dir results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/continuation_p3_bidir_selector_3x3_b32_w5 --fresh
```

输出目录：

```text
results/v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation/continuation_p3_bidir_selector_3x3_b32_w5
```

## 8. v12.15 continuation artifact hash

### 8.1 Budget-only continuation

| artifact | sha256 |
|---|---|
| `v1215_continuation_actuator_budget_repair.csv` | `1a0c93d989e7d80e1f676e36722f3e89494e83549f1a7a8ed402c209512a5d13` |
| `v1215_continuation_actuator_budget_summary.csv` | `95035c2c942bf6884d96d81688e532280760fd4504d9f4810de72c2edb0d6f8b` |
| `v1215_continuation_estimator_cv.csv` | `9d7fd6d9f90c780f5e907efd157ebb7eb0980e09acc77e8acbda4a7bcb1e7cc7` |
| `v1215_continuation_estimator_feature_table.csv` | `1f5e32ec5f0d4df5529170dcdf6ad9963867c98422f6db2ec557e329d4ec37e8` |
| `v1215_continuation_route_decision.json` | `5f6863415af6b0ae67b1f8dbc21f09010ccffd48d87b4ea715bd2eca24423406` |

### 8.2 Fused primitive actuator continuation

| artifact | sha256 |
|---|---|
| `v1215_continuation_actuator_budget_repair.csv` | `f6c0e620c9c82c06ced11d8199b97ac3e0de28aef371bf4825af0fbd18c1b0ef` |
| `v1215_continuation_actuator_budget_summary.csv` | `d17f1e0360a43c56717d3dcc1e987d8f67c6e0c961aface230719550bfc9fc95` |
| `v1215_continuation_estimator_cv.csv` | `00112fb4387cbdd05790575d6e61d8efa311f258d8d9742c488a608b4b43e03a` |
| `v1215_continuation_estimator_feature_table.csv` | `4ebd7351fae44da6d0c719c5b8162b37280e85827bfaab7602c025ee7558754c` |
| `v1215_continuation_route_decision.json` | `318ea847a431a06d9d9d2972f970bfef43ad6689fe735b71eff886ce5e15eb96` |

### 8.3 Fixed K6 P3

| artifact | sha256 |
|---|---|
| `v1215_continuation_p3_candidates.csv` | `2ca715dce3c6e484997f143aad932062e1071b1778345af02d13dfee1ebaba6e` |
| `v1215_continuation_p3_controls.csv` | `d36f3f2306924170049c3a2ad4e1c35a30fbf95b1af17ed70fc87a0cd1d79c32` |
| `v1215_continuation_p3_route_decision.json` | `4a444a8f7a4d8bb8b59e518cbc2c74faf69f6bb171cbd0c35cd0cb40858cc07d` |
| `v1215_continuation_p3_summary.csv` | `9e57426784fc087c9b4c727ef9f4fc561e4c5910052e82f4c54db563db8356f7` |

### 8.4 Bidirectional selector P3

| artifact | sha256 |
|---|---|
| `v1215_continuation_p3_candidates.csv` | `f01a6e76a0dd98ac574a4e0a132190376efec20d61de0c95c3a96784ffd30e38` |
| `v1215_continuation_p3_controls.csv` | `ab2cd43a5e5a6850264be0502be9ba4b2edcde305aa76fdfe5f7efa4c4cc3b5b` |
| `v1215_continuation_p3_route_decision.json` | `7d042cc9d2cd389476227ae944944eb5054eec2544563e93d74ffaff73d86bb4` |
| `v1215_continuation_p3_selector_probes.csv` | `62428897fd9a95c212debeffe7e7f0319402b81a48361e33f9c8250a5f3aba3b` |
| `v1215_continuation_p3_summary.csv` | `80860bd385cadd45eb5fad466cd83a187878acec6996970b3dd20ac26b8714d8` |
