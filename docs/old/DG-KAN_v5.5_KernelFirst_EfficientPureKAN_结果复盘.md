# DG-KAN v5.5 Kernel-First Efficient PureKAN 结果复盘

本轮依据 `docs/DG-KAN_v5.5_KernelFirst_EfficientPureKAN_实验计划.md`。核心目标是先回答：当前 PureKAN edge primitive 是否存在真正接近 MLP 的 kernel / compute path，再决定是否继续 functional optimizer / LightSmooth。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 core consistency | 8 | 0 |
| P1 phase efficiency | 144 | 0 |
| P2 Depthwise repair | 12 | 0 |
| P3 CP repair | 15 | 0 |
| P4 RationalKAT | 15 | 0 |
| P5 accuracy frontier | 54 | 0 |
| P6 true memory | 1 | 0 |
| P7 LightSmooth | 1 | 0 |
| P8 selection | 1 | 0 |
| P9 confirm/failure | 1 | 0 |

## Code / Config Changes

```text
experiments/run_gafu_v55.py
  Added phase-separated efficiency profiler and kernel repair probes for DWM / CP / RationalKAT.
  Added compact P5 efficient accuracy frontier and gated P6-P9 outputs.

experiments/analyze_gafu_v55.py
  Generates v5.5 gate summaries, route_decision.json, aggregate_decision.json, figures, and this replay.
```

## P0 Core / Runner Consistency

| primitive | class | edge | nonKAN | mixing | rollback | P0 |
|---|---|---|---|---|---|---|
| MLP-reference | MLPClassifier | 0 | 59722 | 0 | 0.0000 | yes |
| RBFOnly-Dense | RBFDense | 1009664 | 0 | 0 | 0.0000 | yes |
| ABRBF-Dense | ABRBFDense | 1198976 | 0 | 0 | 0.0000 | yes |
| ABRBF-DepthwiseMix | ABRBFDepthwiseMixDense | 82864 | 0 | 63104 | 0.0000 | yes |
| ABRBF-CPRank4 | CPABRBFDense | 194856 | 0 | 0 | 0.0000 | yes |
| ABRBF-CPRank8 | CPABRBFDense | 200400 | 0 | 0 | 0.0000 | yes |
| ABRBF-CPRank16 | CPABRBFDense | 211488 | 0 | 0 | 0.0000 | yes |
| RationalKAT-AB | RationalKATDense | 69344 | 0 | 63104 | 0.0000 | yes |

P0 verdict: pass. All PureKAN candidates keep trainable parameters inside the audited edge system; MLP is retained only as reference.

## P1 Phase-Separated Efficiency Profiler

| method | rows | fwd | bwd | bmem | step | early | final | bottleneck |
|---|---|---|---|---|---|---|---|---|
| ABRBF-Dense | 16 | 10.9916 | 7.6673 | 2.4620 | 5.3707 | 0.0000 | 0.0000 | memory |
| CP-0-current-r8 | 16 | 9.3261 | 3.2329 | 1.6660 | 3.1941 | 0.0000 | 0.0000 | memory |
| CP-1-two-stage-r8 | 16 | 8.8008 | 3.4513 | 2.5689 | 2.9890 | 0.0000 | 0.0000 | memory |
| DWM-0-current | 16 | 11.7158 | 2.9198 | 1.6633 | 2.9844 | 0.0000 | 0.0000 | memory |
| DWM-1-vectorized | 16 | 7.8718 | 2.5552 | 2.1921 | 2.4354 | 0.0000 | 0.0000 | memory |
| RBFOnly-Dense | 16 | 3.6404 | 1.2863 | 2.1748 | 1.4318 | 0.0000 | 0.0000 | memory |
| RK-0-current | 16 | 4.2360 | 2.7816 | 1.5500 | 2.2300 | 0.0000 | 0.0000 | memory |
| RK-1-clean | 16 | 3.7053 | 3.1200 | 1.5496 | 2.3229 | 0.0000 | 0.0000 | memory |

P1 survivors:

```text
none
```

Observation:

```text
Several cleaned paths show exploratory speed pockets, but no primitive satisfies the final MLP-like envelope.
Backward peak memory remains the broadest blocker.
```

## P2 DepthwiseMix Kernel Repair

| variant | rows | step | bmem | pass rate | P2 |
|---|---|---|---|---|---|
| DWM-0-current | 2 | 2.3629 | 2.0502 | 0.0000 | no |
| DWM-1-vectorized | 2 | 2.4538 | 2.9200 | 0.0000 | no |
| DWM-2-compiled | 2 | 1.5322 | 0.1539 | 1.0000 | yes |
| DWM-3-fused-triton-forward | 2 |  |  | 0.0000 | no |
| DWM-4-custom-backward-recompute | 2 | 4.4570 | 4.3368 | 0.0000 | no |
| DWM-5-streaming-backward | 2 | 4.5447 | 4.3368 | 0.0000 | no |

P2 survivors:

```text
DWM-2-compiled
```

Interpretation: compiled/vectorized DepthwiseMix exposes a real kernel-path gain, but its trainable accuracy path still needs P5 confirmation.

## P3 CP-ABRBF Compute Graph Repair

| variant | rows | best step | best bmem | monotonic | rank16 step | P3 |
|---|---|---|---|---|---|---|
| CP-0-current | 3 | 1.7138 | 1.5258 | 0 | 2.1694 | no |
| CP-1-two-stage-gemm | 3 | 1.6965 | 1.7682 | 1 | 1.6965 | yes |
| CP-2-batched-bmm | 3 | 1.7856 | 1.7682 | 0 | 1.7856 | no |
| CP-3-compiled | 3 | 1.7024 | 0.2852 | 1 | 1.7024 | yes |
| CP-4-custom-backward-recompute | 3 | 3.0104 | 2.5619 | 1 | 3.5740 | no |

P3 survivors:

```text
CP-1-two-stage-gemm, CP-3-compiled
```

Interpretation: two-stage/compiled CP repairs the rank-speed behavior enough to reach exploratory gate, but memory and task quality remain joint blockers.

## P4 RationalKAT Kernel / Recipe Repair

Kernel audit:

| variant | step | bmem | kernel |
|---|---|---|---|
| RK-0-current | 1.6874 | 1.2941 | no |
| RK-1-torch-eager-clean | 1.6144 | 1.2941 | no |
| RK-2-triton-fused |  |  | no |
| RK-3-compiled | 1.5576 | 0.1027 | yes |
| RK-4-grouped-rational-plus-gemm | 1.7763 | 1.2941 | no |
| RK-5-custom-backward-recompute | 3.5685 | 1.2941 | no |

Recipe audit:

| dataset | recipe | acc | ECE |
|---|---|---|---|
| Fashion-MNIST | groups4-linear_silu-lr1e-3 | 0.7051 | 0.3881 |
| Fashion-MNIST | groups8-linear_silu-lr1e-3 | 0.7148 | 0.3976 |
| Fashion-MNIST | groups8-linear_silu-lr2e-3 | 0.7324 | 0.2911 |
| KMNIST | groups4-linear_silu-lr1e-3 | 0.4805 | 0.2612 |
| KMNIST | groups8-linear_silu-lr1e-3 | 0.4805 | 0.2612 |
| KMNIST | groups8-linear_silu-lr2e-3 | 0.5703 | 0.2791 |
| MNIST | groups4-linear_silu-lr1e-3 | 0.7637 | 0.4933 |
| MNIST | groups8-linear_silu-lr1e-3 | 0.7637 | 0.4933 |
| MNIST | groups8-linear_silu-lr2e-3 | 0.7793 | 0.4291 |

P4 kernel survivors:

```text
RK-3-compiled
```

P4 verdict: compiled RationalKAT is the best kernel signal, but compact recipes remain far below MLP on task quality, especially KMNIST.

## P5 Efficient Primitive Accuracy Frontier

| dataset | method | runs | acc | gap | ECE | step | bmem | P5 |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | ABRBF-Dense-AdamW | 3 | 0.7891 | 0.0150 | 0.0389 | 5.3707 | 2.4620 | no |
| Fashion-MNIST | CP-1-two-stage-r16-AdamW | 3 | 0.7969 | 0.0072 | 0.0441 | 2.9890 | 2.5689 | no |
| Fashion-MNIST | DWM-1-vectorized-AdamW | 3 | 0.7871 | 0.0169 | 0.0822 | 2.4354 | 2.1921 | no |
| Fashion-MNIST | MLP-AdamW | 3 | 0.8040 | 0.0000 | 0.0422 | 1.0000 | 1.0000 | no |
| Fashion-MNIST | RBFOnly-Dense-AdamW | 3 | 0.7891 | 0.0150 | 0.0320 | 1.4318 | 2.1748 | no |
| Fashion-MNIST | RK-0-current-AdamW | 3 | 0.7637 | 0.0404 | 0.2608 | 2.2300 | 1.5500 | no |
| KMNIST | ABRBF-Dense-AdamW | 3 | 0.6862 | 0.0423 | 0.0453 | 5.3707 | 2.4620 | no |
| KMNIST | CP-1-two-stage-r16-AdamW | 3 | 0.6712 | 0.0573 | 0.0450 | 2.9890 | 2.5689 | no |
| KMNIST | DWM-1-vectorized-AdamW | 3 | 0.6120 | 0.1165 | 0.0804 | 2.4354 | 2.1921 | no |
| KMNIST | MLP-AdamW | 3 | 0.7285 | 0.0000 | 0.0438 | 1.0000 | 1.0000 | no |
| KMNIST | RBFOnly-Dense-AdamW | 3 | 0.6647 | 0.0638 | 0.0563 | 1.4318 | 2.1748 | no |
| KMNIST | RK-0-current-AdamW | 3 | 0.5918 | 0.1367 | 0.2425 | 2.2300 | 1.5500 | no |
| MNIST | ABRBF-Dense-AdamW | 3 | 0.8763 | 0.0260 | 0.0449 | 5.3707 | 2.4620 | no |
| MNIST | CP-1-two-stage-r16-AdamW | 3 | 0.8880 | 0.0143 | 0.0496 | 2.9890 | 2.5689 | no |
| MNIST | DWM-1-vectorized-AdamW | 3 | 0.8665 | 0.0358 | 0.1480 | 2.4354 | 2.1921 | no |
| MNIST | MLP-AdamW | 3 | 0.9023 | 0.0000 | 0.0430 | 1.0000 | 1.0000 | no |
| MNIST | RBFOnly-Dense-AdamW | 3 | 0.8516 | 0.0508 | 0.0648 | 1.4318 | 2.1748 | no |
| MNIST | RK-0-current-AdamW | 3 | 0.8457 | 0.0566 | 0.3786 | 2.2300 | 1.5500 | no |

P5 all-dataset survivors:

```text
none
```

P5 verdict:

```text
No method passes task + efficiency across all datasets.
CP-1-two-stage-r16 is the best task signal among efficient candidates on MNIST/Fashion,
but KMNIST remains below the required accuracy frontier.
DWM and RK kernel paths are promising for speed, but not yet for accuracy.
```

## P6-P9 Decision

```text
P6 true memory audit: not run; P5 produced no all-dataset efficient accuracy survivor.
P7 functional / LightSmooth smoke: not run.
P8 3-seed joint candidate selection: not run.
P9 confirm: not run.
```

## Failure Diagnosis

| failure | count |
|---|---|
| F2_memory_fail | 115 |
| F1_efficiency_fail | 31 |
| F3_accuracy_recipe_unproven | 9 |
| F6_implementation_missing | 3 |

Route decision:

```text
mixed_BC: Kernel repair exposes exploratory efficiency pockets, but task+efficiency does not survive P5.
```

## Required Artifacts

Written under `results/v5_5/`:

```text
p0_core_runner_consistency.csv
p1_phase_efficiency_profiler.csv
p1_efficiency_gate_summary.csv
p2_depthwise_kernel_repair.csv
p2_depthwise_gate_summary.csv
p3_cp_compute_graph_repair.csv
p3_cp_gate_summary.csv
p4_rationalkat_kernel_recipe.csv
p4_rational_kernel_gate_summary.csv
p4_rational_recipe_summary.csv
p5_efficient_accuracy_frontier.csv
p5_accuracy_gate_summary.csv
p6_custom_backward_true_memory.csv
p7_functional_lightsmooth_smoke.csv
p8_candidate_selection3.csv
p9_confirm5.csv
p9_confirm10.csv
failure_table.csv
failure_by_type.csv
failure_by_method.csv
aggregate_decision.json
route_decision.json
figures/
```

## Final Decision

```text
DG-KAN v5.5 status:
  stop_after_p5_no_joint_accuracy_efficiency_survivor

What improved:
  Kernel-first profiling now separates forward, backward, optimizer and memory phases.
  DWM compiled and CP two-stage/compiled expose exploratory efficiency improvements.
  RationalKAT compiled is the clearest memory/kernel signal.

What failed:
  No primitive passes the final efficiency envelope.
  P5 produced no task + efficiency all-dataset survivor.
  CP keeps the best accuracy signal but is not final efficient system.
  DWM/RK are better efficiency primitives but need architecture/recipe repair.

Conclusion:
  v5.5 supports the kernel-first diagnosis.
  The next step should repair the task recipe for efficient DWM/RK-style primitives
  or move to a new GEMM-native edge design before returning to functional optimizer work.
```
