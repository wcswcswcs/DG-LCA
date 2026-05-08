# DG-KAN v5.7 Kernel-Verified Efficient PureKAN 结果复盘

本轮依据 `docs/DG-KAN_v5.7_KernelVerified_EfficientPureKAN_实验计划.md`。核心目标是先确认 profiler 没有把 cold compile / graph break 混入 steady-state，再判断 strict PureKAN primitive 是否存在 MLP-like kernel path 和可接受 accuracy。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 core/profiler | 12 | 0 |
| P1 efficiency v3 | 144 | 0 |
| P1 cold/warm audit | 144 | 0 |
| P1 memory decomposition | 144 | 0 |
| P2 DWM recipe | 36 | 0 |
| P3 Rational recipe | 36 | 0 |
| P4 CP rank/speed | 36 | 0 |
| P5 joint selection | 1 | 0 |
| P6-P10 gated | 5 | 0 |

## Code / Config Changes

```text
experiments/run_gafu_v57.py
  Added kernel-verified P0/P1 with cold-vs-warm compile audit,
  memory decomposition, graph-break/recompile fields, and gated DWM/RK/CP recipe probes.

experiments/analyze_gafu_v57.py
  Generates v5.7 gate summaries, failure taxonomy, route_decision.json,
  aggregate_decision.json, SVG diagnostics, and this replay.
```

## P0 Core / Profiler Consistency

| primitive | class | edge | nonKAN | mixing | backend | breaks | rollback | P0 |
|---|---|---|---|---|---|---|---|---|
| MLP-reference | MLPClassifier | 0 | 59722 | 0 | eager | 0 | 0.0000 | yes |
| RBFOnly-Dense-reference | RBFDense | 504832 | 0 | 0 | eager | 0 | 0.0000 | yes |
| ABRBF-Dense-reference | ABRBFDense | 694144 | 0 | 0 | eager | 0 | 0.0000 | yes |
| DWM-current | GEMMNativeDepthwiseMixDense | 74544 | 0 | 63104 | eager | 0 | 0.0000 | yes |
| DWM-warm-compiled | GEMMNativeDepthwiseMixDense | 74544 | 0 | 63104 | torch.compile-warm | 0 | 0.0000 | yes |
| DWM-gemm-native | GEMMNativeDepthwiseMixDense | 74544 | 0 | 63104 | eager-gemm-native | 0 | 0.0000 | yes |
| RationalKAT-current | GEMMNativeRationalKATDense | 69344 | 0 | 63104 | eager | 0 | 0.0000 | yes |
| RationalKAT-compiled | GEMMNativeRationalKATDense | 69344 | 0 | 63104 | torch.compile-warm | 0 | 0.0000 | yes |
| RationalKAT-grouped-gemm | GEMMNativeRationalKATDense | 69344 | 0 | 63104 | eager-gemm-native | 0 | 0.0000 | yes |
| CP-two-stage-r4 | GEMMNativeCPABRBFDense | 194696 | 0 | 0 | eager-gemm-native | 0 | 0.0000 | yes |
| CP-two-stage-r8 | GEMMNativeCPABRBFDense | 200080 | 0 | 0 | eager-gemm-native | 0 | 0.0000 | yes |
| CP-two-stage-r16 | GEMMNativeCPABRBFDense | 210848 | 0 | 0 | eager-gemm-native | 0 | 0.0000 | yes |

P0 verdict: pass if each non-reference primitive is core-defined, edge-covered, nonKAN-free, rollback-exact, and has no recorded graph-break issue in the manifest.

## P1 Phase-Separated Efficiency Profiler V3

| method | rows | fwd | bwd | bmem | step | cold/warm | breaks | recomp | explore | final | bottleneck |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ABRBF-Dense-reference | 12 | 6.4005 | 1.9528 | 1.9907 | 2.1049 | 2.7748 | 0 | 0 | 0.0000 | 0.0000 | memory |
| CP-two-stage-r16 | 12 | 7.8850 | 2.9018 | 3.7520 | 2.8233 | 2.0011 | 0 | 0 | 0.0000 | 0.0000 | memory |
| CP-two-stage-r4 | 12 | 7.3934 | 2.7040 | 2.0928 | 2.6510 | 1.9416 | 0 | 0 | 0.0000 | 0.0000 | memory |
| CP-two-stage-r8 | 12 | 7.5113 | 2.6058 | 2.6372 | 2.6120 | 1.9615 | 0 | 0 | 0.0000 | 0.0000 | memory |
| DWM-current | 12 | 6.2220 | 2.0347 | 1.9079 | 2.1373 | 1.9990 | 0 | 0 | 0.0000 | 0.0000 | memory |
| DWM-gemm-native | 12 | 6.2069 | 1.9457 | 1.8915 | 2.0801 | 1.8994 | 0 | 0 | 0.0000 | 0.0000 | memory |
| DWM-warm-compiled | 12 | 8.4430 | 1.5334 | 0.1306 | 2.6134 | 871.9148 | 0 | 0 | 0.0000 | 0.0000 | time |
| RBFOnly-Dense-reference | 12 | 3.5967 | 1.8509 | 1.6571 | 1.7506 | 5.8133 | 0 | 0 | 0.0000 | 0.0000 | memory |
| RationalKAT-compiled | 12 | 9.1848 | 1.6803 | 0.1323 | 2.8545 | 566.9522 | 0 | 0 | 0.0000 | 0.0000 | time |
| RationalKAT-current | 12 | 4.9286 | 3.3529 | 1.6611 | 2.7211 | 2.6088 | 0 | 0 | 0.0000 | 0.0000 | memory |
| RationalKAT-grouped-gemm | 12 | 5.0712 | 3.5775 | 1.6748 | 2.8710 | 1.4968 | 0 | 0 | 0.0000 | 0.0000 | memory |

P1 exploratory survivors:

```text
none
```

P1 final survivors:

```text
none
```

P1 verdict: v5.7 separates cold step from warm steady-state. Any large cold/warm ratio is now diagnostic rather than silently counted as steady-state.

## P2 DepthwiseMix Kernel / Recipe Repair

| dataset | recipe | runs | acc | gap | ECE | step | bmem | P2 |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | DWM-K4-linear_silu-FixedNorm-scale0.1-lr1e-3 | 3 | 0.7474 | 0.0495 | 0.1546 | 2.2769 | 1.3100 | no |
| Fashion-MNIST | DWM-K8-linear_silu-channelNorm-scale0.3-lr2e-3 | 3 | 0.7604 | 0.0365 | 0.0593 | 2.2769 | 1.3100 | no |
| Fashion-MNIST | DWM-K8-linear_silu-wideMix-scale0.3-lr2e-3 | 3 | 0.7682 | 0.0286 | 0.0373 | 2.2769 | 1.3100 | no |
| KMNIST | DWM-K4-linear_silu-FixedNorm-scale0.1-lr1e-3 | 3 | 0.5918 | 0.1048 | 0.1412 | 2.2769 | 1.3100 | no |
| KMNIST | DWM-K8-linear_silu-channelNorm-scale0.3-lr2e-3 | 3 | 0.6257 | 0.0710 | 0.0658 | 2.2769 | 1.3100 | no |
| KMNIST | DWM-K8-linear_silu-wideMix-scale0.3-lr2e-3 | 3 | 0.6465 | 0.0501 | 0.0524 | 2.2769 | 1.3100 | no |
| MNIST | DWM-K4-linear_silu-FixedNorm-scale0.1-lr1e-3 | 3 | 0.8333 | 0.0697 | 0.2327 | 2.2769 | 1.3100 | no |
| MNIST | DWM-K8-linear_silu-channelNorm-scale0.3-lr2e-3 | 3 | 0.8535 | 0.0495 | 0.1638 | 2.2769 | 1.3100 | no |
| MNIST | DWM-K8-linear_silu-wideMix-scale0.3-lr2e-3 | 3 | 0.8750 | 0.0280 | 0.0849 | 2.2769 | 1.3100 | no |

P2 all-dataset survivors:

```text
none
```

## P3 RationalKAT Kernel / Recipe Repair

| dataset | recipe | runs | acc | gap | ECE | step | bmem | P3 |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | RK-groups16-silu-smallResidual-lr3e-3 | 3 | 0.7865 | 0.0104 | 0.0355 | 2.8155 | 1.1561 | no |
| Fashion-MNIST | RK-groups32-linear_silu-smallResidual-lr3e-3 | 3 | 0.7865 | 0.0104 | 0.0355 | 2.8155 | 1.1561 | no |
| Fashion-MNIST | RK-groups8-linear_silu-smallResidual-lr2e-3 | 3 | 0.7910 | 0.0059 | 0.0447 | 2.8155 | 1.1561 | no |
| KMNIST | RK-groups16-silu-smallResidual-lr3e-3 | 3 | 0.6673 | 0.0293 | 0.0411 | 2.8155 | 1.1561 | no |
| KMNIST | RK-groups32-linear_silu-smallResidual-lr3e-3 | 3 | 0.6673 | 0.0293 | 0.0411 | 2.8155 | 1.1561 | no |
| KMNIST | RK-groups8-linear_silu-smallResidual-lr2e-3 | 3 | 0.6641 | 0.0326 | 0.0417 | 2.8155 | 1.1561 | no |
| MNIST | RK-groups16-silu-smallResidual-lr3e-3 | 3 | 0.8880 | 0.0150 | 0.0554 | 2.8155 | 1.1561 | no |
| MNIST | RK-groups32-linear_silu-smallResidual-lr3e-3 | 3 | 0.8880 | 0.0150 | 0.0554 | 2.8155 | 1.1561 | no |
| MNIST | RK-groups8-linear_silu-smallResidual-lr2e-3 | 3 | 0.8841 | 0.0189 | 0.0806 | 2.8155 | 1.1561 | no |

P3 all-dataset survivors:

```text
none
```

## P4 CP-ABRBF Rank / Speed Reference

| dataset | recipe | runs | acc | gap | ECE | step | bmem | P4 |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | CP-two-stage-r16-K12-linear_silu | 3 | 0.7695 | 0.0273 | 0.0649 | 2.8233 | 3.7520 | no |
| Fashion-MNIST | CP-two-stage-r4-K8-linear_silu | 3 | 0.7624 | 0.0345 | 0.0662 | 2.6510 | 2.0928 | no |
| Fashion-MNIST | CP-two-stage-r8-K8-linear_silu | 3 | 0.7552 | 0.0417 | 0.0591 | 2.6120 | 2.6372 | no |
| KMNIST | CP-two-stage-r16-K12-linear_silu | 3 | 0.6517 | 0.0449 | 0.0404 | 2.8233 | 3.7520 | no |
| KMNIST | CP-two-stage-r4-K8-linear_silu | 3 | 0.6504 | 0.0462 | 0.0442 | 2.6510 | 2.0928 | no |
| KMNIST | CP-two-stage-r8-K8-linear_silu | 3 | 0.6478 | 0.0488 | 0.0524 | 2.6120 | 2.6372 | no |
| MNIST | CP-two-stage-r16-K12-linear_silu | 3 | 0.8848 | 0.0182 | 0.0819 | 2.8233 | 3.7520 | no |
| MNIST | CP-two-stage-r4-K8-linear_silu | 3 | 0.8796 | 0.0234 | 0.0978 | 2.6510 | 2.0928 | no |
| MNIST | CP-two-stage-r8-K8-linear_silu | 3 | 0.8704 | 0.0326 | 0.1099 | 2.6120 | 2.6372 | no |

P4 all-dataset survivors:

```text
none
```

## P5 Joint Task-Efficiency Selection

_P5 not run; no P2/P3/P4 all-dataset survivor._

P5 all-dataset survivors:

```text
none
```

## P6-P10 Decision

```text
P6 custom backward/memory: not run
P7 functional / LightSmooth compatibility: not run
P8 3-seed selection: not run
P9/P10 confirm: not run

Reason: No strict PureKAN primitive reached the exploratory MLP-like efficiency envelope.
```

## Failure Diagnosis

| failure | count |
|---|---|
| F5_backward_memory_fail | 162 |
| F6_accuracy_recipe_fail | 27 |
| F3_forward_time_fail | 20 |
| F4_backward_time_fail | 4 |

Route decision:

```text
C_no_primitive_survivor: No strict PureKAN primitive reached the exploratory MLP-like efficiency envelope.
```

## Required Artifacts

Written under `results/v5_7/`:

```text
p0_core_profiler_consistency.csv
p1_phase_efficiency_v3.csv
p1_cold_warm_compile_audit.csv
p1_memory_decomposition.csv
p2_dwm_kernel_recipe.csv
p2_dwm_recipe_heatmap.csv
p3_rational_kernel_recipe.csv
p3_rational_safety.csv
p4_cp_rank_speed.csv
p5_joint_task_efficiency_selection.csv
p6_custom_backward_memory.csv
p7_functional_lightsmooth_compat.csv
p8_candidate_selection3.csv
p9_confirm5.csv
p10_confirm10.csv
failure_table.csv
failure_by_method.csv
failure_by_type.csv
route_decision.json
aggregate_decision.json
figures/
```

## Final Decision

```text
DG-KAN v5.7 status:
  stop_after_p1_no_efficiency_survivor

What improved:
  Profiler correctness is now explicit: cold compile, warm steady-state, memory decomposition,
  graph breaks, and recompiles are audited as first-class artifacts.
  DWM / RationalKAT / CP recipe probes remain gated behind strict task-efficiency checks.

What failed / remains open:
  See the P1-P5 gates above. Functional optimizer / LightSmooth stays gated until
  a primitive survives joint task + efficiency selection.

Conclusion:
  v5.7 keeps the kernel-first rule strict: a functional optimizer cannot rescue a primitive
  that lacks a verified MLP-like forward/backward/memory path.
```
