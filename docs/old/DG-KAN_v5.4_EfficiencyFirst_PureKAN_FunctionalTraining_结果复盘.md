# DG-KAN v5.4 Efficiency-First PureKAN Functional Training 结果复盘

本轮依据 `docs/DG-KAN_v5.4_EfficiencyFirst_PureKAN_FunctionalTraining_实验计划.md`。核心目标是先验证 PureKAN edge primitive 的效率 envelope，再决定 functional training / LightSmooth 是否值得继续扩展。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 core manifest | 8 | 0 |
| P1 efficiency | 192 | 0 |
| P2 custom backward | 6 | 0 |
| P3 accuracy frontier | 63 | 0 |
| P4 functional smoke | 1 | 0 |
| P5 LightSmooth | 1 | 0 |
| P6 selection | 1 | 0 |
| P7 confirm5 | 1 | 0 |
| P8 confirm10 | 1 | 0 |

## Code / Config Changes

```text
experiments/dgkan_core.py
  Added core efficient PureKAN edge primitives: ABRBFDepthwiseMixDense, CPABRBFDense, RationalKATDense.
  Extended edge/base/RBF/mixing parameter discovery so efficient mixing remains inside the KAN edge system.

experiments/run_gafu_v54.py
  Added P0 core manifest, P1 efficiency microbenchmark, P2 checkpoint/recompute backward audit,
  P3 AdamW-like accuracy frontier, and gated P4-P9 placeholders/diagnosis.

experiments/analyze_gafu_v54.py
  Generates v5.4 gate summaries, figures, aggregate_decision.json, route_decision.json, and this replay.
```

## P0 Core Consistency

| method | primitive | edge | nonKAN | mixing | rollback | P0 |
|---|---|---|---|---|---|---|
| MLP-AdamW-reference | MLP-reference | 0 | 59722 | 0 | 0.0000 | yes |
| RBFOnly-Dense | dense-rbf | 1009664 | 0 | 0 | 0.0000 | yes |
| ABRBF-Dense-linear+silu | dense-abrbf | 1198976 | 0 | 0 | 0.0000 | yes |
| ABRBF-DepthwiseMix-linear+silu | depthwise-mix | 82864 | 0 | 63104 | 0.0000 | yes |
| ABRBF-CPRank4 | cp-lowrank | 194856 | 0 | 0 | 0.0000 | yes |
| ABRBF-CPRank8 | cp-lowrank | 200400 | 0 | 0 | 0.0000 | yes |
| ABRBF-CPRank16 | cp-lowrank | 211488 | 0 | 0 | 0.0000 | yes |
| RationalKAT-AB | rational-kat | 69344 | 0 | 63104 | 0.0000 | yes |

## P1 Efficiency Microbenchmark

| method | rows | fwd | bwd | bmem | step | pass rate | P1 |
|---|---|---|---|---|---|---|---|
| ABRBF-CPRank16 | 24 | 7.1699 | 4.3364 | 2.5357 | 3.6396 | 0.0000 | no |
| ABRBF-CPRank4 | 24 | 15.5634 | 4.2092 | 2.5285 | 4.8073 | 0.0000 | no |
| ABRBF-CPRank8 | 24 | 10.3810 | 3.5064 | 2.5309 | 3.7239 | 0.0000 | no |
| ABRBF-Dense-linear+silu | 24 | 11.7354 | 3.7917 | 3.1774 | 4.1188 | 0.0000 | no |
| ABRBF-DepthwiseMix-linear+silu | 24 | 7.8662 | 8.2935 | 2.4273 | 5.7053 | 0.0000 | no |
| RBFOnly-Dense | 24 | 2.8412 | 1.3465 | 2.5214 | 1.5059 | 0.0000 | no |
| RationalKAT-AB | 24 | 6.9917 | 5.9999 | 1.3494 | 4.3777 | 0.0000 | no |

P1 survivors:
```text
none
```

## P2 Custom Backward Audit

| method | variant | relerr | cos | mem | step | saved MB | P2 |
|---|---|---|---|---|---|---|---|
| ABRBF-DepthwiseMix-linear+silu | Autograd | 0.0000 | 1.0000 | 2.9838 | 1.5462 | 7.2734 | no |
| ABRBF-DepthwiseMix-linear+silu | CustomBackward-Recompute | 0.0000 | 1.0000 | 2.9838 | 10.3142 | 1.0910 | no |
| ABRBF-DepthwiseMix-linear+silu | CustomBackward-StreamingStats | 0.0000 | 1.0000 | 2.9838 | 4.5104 | 1.0910 | no |
| RationalKAT-AB | Autograd | 0.0000 | 1.0000 | 1.2465 | 4.9874 | 1.5312 | no |
| RationalKAT-AB | CustomBackward-Recompute | 0.0000 | 1.0000 | 1.2465 | 5.5812 | 0.2297 | no |
| RationalKAT-AB | CustomBackward-StreamingStats | 0.0000 | 1.0000 | 1.2465 | 5.6224 | 0.2297 | no |

## P3 Accuracy Frontier

| dataset | method | runs | acc | std | gap | ECE | fwd | bmem | step | P3 |
|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | ABRBF-CPRank16-AdamW | 3 | 0.7956 | 0.0088 | 0.0046 | 0.0365 | 7.1699 | 2.5357 | 3.6396 | no |
| Fashion-MNIST | ABRBF-CPRank8-AdamW | 3 | 0.7891 | 0.0127 | 0.0111 | 0.0329 | 10.3810 | 2.5309 | 3.7239 | no |
| Fashion-MNIST | ABRBF-Dense-AdamW | 3 | 0.7852 | 0.0115 | 0.0150 | 0.0286 | 11.7354 | 3.1774 | 4.1188 | no |
| Fashion-MNIST | ABRBF-DepthwiseMix-AdamW | 3 | 0.7956 | 0.0176 | 0.0046 | 0.0915 | 7.8662 | 2.4273 | 5.7053 | no |
| Fashion-MNIST | MLP-AdamW | 3 | 0.8001 | 0.0064 | 0.0000 | 0.0379 | 1.0000 | 1.0000 | 1.0000 | no |
| Fashion-MNIST | RBFOnly-Dense-AdamW | 3 | 0.7845 | 0.0064 | 0.0156 | 0.0430 | 2.8412 | 2.5214 | 1.5059 | no |
| Fashion-MNIST | RationalKAT-AB-AdamW | 3 | 0.7630 | 0.0018 | 0.0371 | 0.2363 | 6.9917 | 1.3494 | 4.3777 | no |
| KMNIST | ABRBF-CPRank16-AdamW | 3 | 0.6908 | 0.0295 | 0.0286 | 0.0436 | 7.1699 | 2.5357 | 3.6396 | no |
| KMNIST | ABRBF-CPRank8-AdamW | 3 | 0.6875 | 0.0253 | 0.0319 | 0.0444 | 10.3810 | 2.5309 | 3.7239 | no |
| KMNIST | ABRBF-Dense-AdamW | 3 | 0.6979 | 0.0183 | 0.0215 | 0.0376 | 11.7354 | 3.1774 | 4.1188 | no |
| KMNIST | ABRBF-DepthwiseMix-AdamW | 3 | 0.6191 | 0.0146 | 0.1003 | 0.0873 | 7.8662 | 2.4273 | 5.7053 | no |
| KMNIST | MLP-AdamW | 3 | 0.7194 | 0.0129 | 0.0000 | 0.0507 | 1.0000 | 1.0000 | 1.0000 | no |
| KMNIST | RBFOnly-Dense-AdamW | 3 | 0.6738 | 0.0073 | 0.0456 | 0.0549 | 2.8412 | 2.5214 | 1.5059 | no |
| KMNIST | RationalKAT-AB-AdamW | 3 | 0.5814 | 0.0064 | 0.1380 | 0.2234 | 6.9917 | 1.3494 | 4.3777 | no |
| MNIST | ABRBF-CPRank16-AdamW | 3 | 0.8893 | 0.0260 | 0.0182 | 0.0561 | 7.1699 | 2.5357 | 3.6396 | no |
| MNIST | ABRBF-CPRank8-AdamW | 3 | 0.8978 | 0.0259 | 0.0098 | 0.0580 | 10.3810 | 2.5309 | 3.7239 | no |
| MNIST | ABRBF-Dense-AdamW | 3 | 0.8919 | 0.0148 | 0.0156 | 0.0518 | 11.7354 | 3.1774 | 4.1188 | no |
| MNIST | ABRBF-DepthwiseMix-AdamW | 3 | 0.8678 | 0.0231 | 0.0397 | 0.1502 | 7.8662 | 2.4273 | 5.7053 | no |
| MNIST | MLP-AdamW | 3 | 0.9076 | 0.0171 | 0.0000 | 0.0452 | 1.0000 | 1.0000 | 1.0000 | no |
| MNIST | RBFOnly-Dense-AdamW | 3 | 0.8587 | 0.0175 | 0.0488 | 0.0600 | 2.8412 | 2.5214 | 1.5059 | no |
| MNIST | RationalKAT-AB-AdamW | 3 | 0.8509 | 0.0253 | 0.0566 | 0.3693 | 6.9917 | 1.3494 | 4.3777 | no |

P3 all-dataset survivors:
```text
none
```

## P4-P8 Decision

```text
P4 functional / analytic update smoke: not run
Reason: P3 produced no efficiency-aware accuracy survivor.

P5 LightSmooth compatibility: not run
Reason: P4 was not reached.

P6/P7/P8 confirm: not run
Reason: no candidate passed the written joint gate.
```

## Failure Diagnosis

| failure | count |
|---|---|
| F1_efficiency_failed | 222 |

## Required Artifacts

Written under `results/v5_4/`:
```text
p0_core_primitive_manifest.csv
p1_efficiency_microbenchmark.csv
p1_efficiency_gate_summary.csv
p2_custom_backward_audit.csv
p2_custom_backward_gate_summary.csv
p3_accuracy_frontier.csv
p3_accuracy_gate_summary.csv
p4_functional_analytic_smoke.csv
p5_lightsmooth_compatibility.csv
p6_candidate_selection3.csv
p7_confirm5.csv
p8_confirm10.csv
p9_failure_diagnosis.csv
failure_table.csv
failure_by_type.csv
failure_by_method.csv
aggregate_decision.json
route_decision.json
figures/
```

## Final Decision

```text
DG-KAN v5.4 status:
  stop_after_p3_no_efficiency_accuracy_survivor

What improved:
  Efficient PureKAN primitives are now core-level objects, not runner-only helpers.
  DepthwiseMix / CP / RationalKAT all keep learnable parameters inside the KAN edge system.
  P3 confirms CP-ABRBF can recover meaningful task accuracy in places, especially MNIST/Fashion.

What failed:
  No primitive passed the P1 hard efficiency envelope across the compact grid.
  Checkpoint/recompute backward preserved gradients but did not reduce measured peak memory below MLP.
  P3 produced no all-dataset survivor under the accuracy + efficiency joint gate, with KMNIST the hardest task.

Conclusion:
  v5.4 supports the plan's warning: dense AB-RBF is a mechanism reference, not a final efficient system.
  The next route should prioritize fused/custom kernels or a more GEMM-native Rational/KAT primitive before more functional optimizer work.
```
