# DG-KAN v5.6 GEMM-Native Efficient PureKAN 结果复盘

本轮依据 `docs/DG-KAN_v5.6_GEMMNative_EfficientPureKAN_实验计划.md`。核心目标是先验证 PureKAN primitive 是否存在 MLP-like GEMM-native compute path，再决定是否继续 functional optimizer / LightSmooth。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 core/profiler | 8 | 0 |
| P1 efficiency v2 | 144 | 4 |
| P2 DWM recipe | 36 | 0 |
| P3 Rational recipe | 36 | 0 |
| P4 CP repair | 36 | 0 |
| P5 joint selection | 1 | 0 |
| P6 memory | 1 | 0 |
| P7 LightSmooth | 1 | 0 |
| P8 functional | 1 | 0 |
| P9/P10 confirm | 2 | 0 |

## Code / Config Changes

```text
experiments/dgkan_core.py
  Added core GEMMNativeDepthwiseMixDense, GEMMNativeCPABRBFDense,
  and GEMMNativeRationalKATDense.

experiments/run_gafu_v56.py
  Added core/profiler consistency checks, phase-separated efficiency profiler v2,
  DWM/Rational/CP recipe repair, gated joint selection, and gated P6-P10 artifacts.

experiments/analyze_gafu_v56.py
  Generates v5.6 gate summaries, route_decision.json, aggregate_decision.json,
  figures, failure taxonomy, and this replay.
```

## P0 Core / Profiler Consistency

| primitive | class | edge | nonKAN | mixing | core | rollback | P0 |
|---|---|---|---|---|---|---|---|
| MLP-reference | MLPClassifier | 0 | 59722 | 0 | 1 | 0.0000 | yes |
| RBFOnly-Dense | RBFDense | 504832 | 0 | 0 | 1 | 0.0000 | yes |
| ABRBF-Dense | ABRBFDense | 694144 | 0 | 0 | 1 | 0.0000 | yes |
| DWM-compiled-current | GEMMNativeDepthwiseMixDense | 74544 | 0 | 63104 | 1 | 0.0000 | yes |
| DWM-recipe-K8-channelNorm | GEMMNativeDepthwiseMixDense | 74544 | 0 | 63104 | 1 | 0.0000 | yes |
| CP-two-stage-r16 | GEMMNativeCPABRBFDense | 210848 | 0 | 0 | 1 | 0.0000 | yes |
| CP-compiled-r16 | GEMMNativeCPABRBFDense | 210848 | 0 | 0 | 1 | 0.0000 | yes |
| RationalKAT-compiled-current | GEMMNativeRationalKATDense | 69344 | 0 | 63104 | 1 | 0.0000 | yes |

P0 verdict: pass if each non-reference primitive is core-defined, edge-covered, nonKAN-free, and rollback-exact.

## P1 Phase-Separated Efficiency Profiler V2

| method | rows | fwd | bwd | bmem | step | explore | final | bottleneck |
|---|---|---|---|---|---|---|---|---|
| ABRBF-Dense | 24 | 9.7906 | 3.9639 | 1.8471 | 3.5551 | 0.0000 | 0.0000 | memory |
| CP-two-stage-r16 | 24 | 8.5099 | 2.2671 | 3.0743 | 2.5711 | 0.0000 | 0.0000 | memory |
| DWM-compiled-current | 20 | 572.5974 | 99.9813 | 0.9322 | 114.9962 | 0.0000 | 0.0000 | time |
| RBFOnly-Dense | 24 | 3.9959 | 2.1130 | 1.5557 | 1.9547 | 0.0417 | 0.0000 | memory |
| RK-compiled-current | 24 | 288.8317 | 67.6789 | 0.5204 | 67.1710 | 0.0000 | 0.0000 | time |

P1 exploratory survivors:

```text
RBFOnly-Dense
```

P1 final survivors:

```text
none
```

## P2 DepthwiseMix Recipe Repair

| dataset | recipe | runs | acc | gap | ECE | step | bmem | P2 |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | DWM-K4-linear_silu-channelNorm-lr1e-3 | 3 | 0.7409 | 0.0560 | 0.1340 | 95.9968 | 0.9435 | no |
| Fashion-MNIST | DWM-K8-linear_silu-channelNorm-lr1e-3 | 3 | 0.7487 | 0.0482 | 0.0968 | 95.9968 | 0.9435 | no |
| Fashion-MNIST | DWM-K8-linear_silu-wideMix-lr2e-3 | 3 | 0.7682 | 0.0286 | 0.0373 | 95.9968 | 0.9435 | no |
| KMNIST | DWM-K4-linear_silu-channelNorm-lr1e-3 | 3 | 0.5951 | 0.1016 | 0.1108 | 95.9968 | 0.9435 | no |
| KMNIST | DWM-K8-linear_silu-channelNorm-lr1e-3 | 3 | 0.5846 | 0.1120 | 0.0928 | 95.9968 | 0.9435 | no |
| KMNIST | DWM-K8-linear_silu-wideMix-lr2e-3 | 3 | 0.6465 | 0.0501 | 0.0524 | 95.9968 | 0.9435 | no |
| MNIST | DWM-K4-linear_silu-channelNorm-lr1e-3 | 3 | 0.8438 | 0.0592 | 0.2427 | 95.9968 | 0.9435 | no |
| MNIST | DWM-K8-linear_silu-channelNorm-lr1e-3 | 3 | 0.8346 | 0.0684 | 0.2330 | 95.9968 | 0.9435 | no |
| MNIST | DWM-K8-linear_silu-wideMix-lr2e-3 | 3 | 0.8750 | 0.0280 | 0.0849 | 95.9968 | 0.9435 | no |

P2 all-dataset survivors:

```text
none
```

## P3 RationalKAT Recipe Repair

| dataset | recipe | runs | acc | gap | ECE | step | bmem | P3 |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | RK-groups16-silu-smallResidual-lr3e-3 | 3 | 0.7865 | 0.0104 | 0.0355 | 67.1710 | 0.5204 | no |
| Fashion-MNIST | RK-groups4-linear_silu-identity-lr1e-3 | 3 | 0.7819 | 0.0150 | 0.0462 | 67.1710 | 0.5204 | no |
| Fashion-MNIST | RK-groups8-linear_silu-smallResidual-lr2e-3 | 3 | 0.7910 | 0.0059 | 0.0447 | 67.1710 | 0.5204 | no |
| KMNIST | RK-groups16-silu-smallResidual-lr3e-3 | 3 | 0.6673 | 0.0293 | 0.0411 | 67.1710 | 0.5204 | no |
| KMNIST | RK-groups4-linear_silu-identity-lr1e-3 | 3 | 0.6322 | 0.0645 | 0.0571 | 67.1710 | 0.5204 | no |
| KMNIST | RK-groups8-linear_silu-smallResidual-lr2e-3 | 3 | 0.6641 | 0.0326 | 0.0417 | 67.1710 | 0.5204 | no |
| MNIST | RK-groups16-silu-smallResidual-lr3e-3 | 3 | 0.8880 | 0.0150 | 0.0554 | 67.1710 | 0.5204 | no |
| MNIST | RK-groups4-linear_silu-identity-lr1e-3 | 3 | 0.8887 | 0.0143 | 0.0837 | 67.1710 | 0.5204 | no |
| MNIST | RK-groups8-linear_silu-smallResidual-lr2e-3 | 3 | 0.8841 | 0.0189 | 0.0806 | 67.1710 | 0.5204 | no |

P3 all-dataset survivors:

```text
none
```

## P4 CP-ABRBF Compute / Capacity Repair

| dataset | recipe | runs | acc | gap | ECE | step | bmem | P4 |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | CP-compiled-r16-K12-linear_silu | 3 | 0.7695 | 0.0273 | 0.0649 | 2.5711 | 3.0743 | no |
| Fashion-MNIST | CP-two-stage-r16-K8-linear_silu | 3 | 0.7676 | 0.0293 | 0.0461 | 2.5711 | 3.0743 | no |
| Fashion-MNIST | CP-two-stage-r8-K8-linear_silu | 3 | 0.7552 | 0.0417 | 0.0591 | 2.5711 | 3.0743 | no |
| KMNIST | CP-compiled-r16-K12-linear_silu | 3 | 0.6517 | 0.0449 | 0.0404 | 2.5711 | 3.0743 | no |
| KMNIST | CP-two-stage-r16-K8-linear_silu | 3 | 0.6556 | 0.0410 | 0.0547 | 2.5711 | 3.0743 | no |
| KMNIST | CP-two-stage-r8-K8-linear_silu | 3 | 0.6478 | 0.0488 | 0.0524 | 2.5711 | 3.0743 | no |
| MNIST | CP-compiled-r16-K12-linear_silu | 3 | 0.8848 | 0.0182 | 0.0819 | 2.5711 | 3.0743 | no |
| MNIST | CP-two-stage-r16-K8-linear_silu | 3 | 0.8717 | 0.0312 | 0.0727 | 2.5711 | 3.0743 | no |
| MNIST | CP-two-stage-r8-K8-linear_silu | 3 | 0.8704 | 0.0326 | 0.1099 | 2.5711 | 3.0743 | no |

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
P7 LightSmooth compatibility: not run
P8 functional training smoke: not run
P9/P10 confirm: not run

Reason: GEMM-native kernel path exists in exploratory profiler, but recipe/task quality does not produce all-dataset survivors.
```

## Failure Diagnosis

| failure | count |
|---|---|
| F2_memory_fail | 82 |
| F3_accuracy_recipe_unproven | 61 |
| F1_efficiency_fail | 33 |
| F4_joint_gate_failed | 20 |

Route decision:

```text
B_recipe_blocker: GEMM-native kernel path exists in exploratory profiler, but recipe/task quality does not produce all-dataset survivors.
```

## Required Artifacts

Written under `results/v5_6/`:

```text
p0_core_profiler_consistency.csv
p1_phase_efficiency_v2.csv
p1_efficiency_gate_summary.csv
p2_dwm_recipe_repair.csv
p2_dwm_gate_summary.csv
p3_rationalkat_recipe_repair.csv
p3_rational_gate_summary.csv
p4_cp_capacity_repair.csv
p4_cp_gate_summary.csv
p5_joint_task_efficiency_selection.csv
p5_joint_gate_summary.csv
p6_custom_backward_memory.csv
p7_lightsmooth_compatibility.csv
p8_functional_training_smoke.csv
p9_confirm5.csv
p10_confirm10.csv
p10_failure_diagnosis.csv
failure_table.csv
failure_by_type.csv
failure_by_method.csv
aggregate_decision.json
route_decision.json
figures/
```

## Final Decision

```text
DG-KAN v5.6 status:
  stop_after_p4_no_recipe_survivor

What improved:
  GEMM-native efficient primitives now live in core dgkan_core.py.
  P1 separates forward/backward/memory/optimizer behavior under the v5.6 gates.
  DWM/Rational/CP recipe repair is evaluated before any functional optimizer work.

What failed / remains open:
  See P1-P5 gates above. Functional optimizer / LightSmooth remains gated behind
  task + efficiency survival, consistent with the v5.6 kernel-first rule.

Conclusion:
  v5.6 keeps the mainline honest: FGO-v3 should wait until a GEMM-native PureKAN
  primitive can satisfy the task-efficiency envelope.
```
