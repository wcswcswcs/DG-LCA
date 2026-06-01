# DG-KAN v6.0 Efficient Functional PureKAN Acceleration 结果复盘

本轮依据 `docs/DG-KAN_v6.0_EfficientFunctionalPureKAN_AccelerationPlan.md`。核心目标是修正 v5.9 的 gating：即使 P1 没有 survivor，也必须对 top memory-failure primitive 运行 P2 custom-backward / recompute 诊断，然后再决定是否进入 sparse primitive recipe、RationalKAT-v3、accelerated convergence、LightSmooth 或 functional optimizer。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 core manifest | 10 | 0 |
| P1 phase efficiency | 120 | 0 |
| P1 saved tensor audit | 120 | 0 |
| P2 custom backward | 25 | 0 |
| P3 sparse primitive | 1 | 0 |
| P4 RationalKAT-v3 | 1 | 0 |
| P5 accelerated convergence | 1 | 0 |
| P6 joint gate | 1 | 0 |
| P7-P10 gated | 5 | 0 |

## Code / Config Changes

```text
experiments/dgkan_core.py
  Added SparseInterpKANDense, SparseSplineKANDense, and RationalKATV3Dense.
  Extended edge/base/residual parameter discovery for sparse interpolation tables.

experiments/run_gafu_v60.py
  Added P0 manifest, P1 profiler truth / roofline / saved tensor audit,
  mandatory P2 custom-backward diagnostics, sparse P3, RationalKAT-v3 P4,
  accelerated convergence P5, and P6-P10 gated artifacts.

experiments/analyze_gafu_v60.py
  Generates v6.0 summaries, route_decision.json, aggregate_decision.json,
  required SVG diagnostics, and this replay.
```

## P0 Core Manifest

| primitive | class | edge | nonKAN | mixing | rollback | P0 |
|---|---|---|---|---|---|---|
| MLP-reference | MLPClassifier | 0 | 59722 | 0 | 0.0000 | yes |
| Dense-ABRBF-reference | ABRBFDense | 694144 | 0 | 0 | 0.0000 | yes |
| DWM2-lite-RBFK2 | DWM2LiteDense | 65184 | 0 | 63104 | 0.0000 | yes |
| DWM2-lite-LUTK8 | DWM2LiteDense | 71424 | 0 | 63104 | 0.0000 | yes |
| SparseInterpKAN-K8 | SparseInterpKANDense | 73504 | 0 | 63104 | 0.0000 | yes |
| SparseInterpKAN-K16 | SparseInterpKANDense | 81824 | 0 | 63104 | 0.0000 | yes |
| SparseSplineKAN-K8 | SparseSplineKANDense | 73504 | 0 | 63104 | 0.0000 | yes |
| RationalKAT-AB-v3-g8 | RationalKATV3Dense | 71424 | 0 | 63104 | 0.0000 | yes |
| RationalKAT-AB-v3-g16 | RationalKATV3Dense | 71424 | 0 | 63104 | 0.0000 | yes |
| CP-ABRBF-light-r4 | GEMMNativeCPABRBFDense | 194696 | 0 | 0 | 0.0000 | yes |

P0 verdict: pass if each non-reference primitive is edge-covered, nonKAN-free, and rollback-exact.

## P1 Profiler Truth / Memory Decomposition

| primitive | rows | fwd | bwd | bmem | step | saved MB | explore | bottleneck |
|---|---|---|---|---|---|---|---|---|
| CP-ABRBF-light-r4 | 12 | 8.8762 | 5.9992 | 1.4814 | 5.3799 | 34.35 | 0.0000 | forward |
| DWM2-lite-LUTK8 | 12 | 10.8805 | 3.9118 | 1.3061 | 4.4718 | 27.64 | 0.0000 | forward |
| DWM2-lite-RBFK2 | 12 | 4.1223 | 2.6352 | 1.2382 | 2.5355 | 6.62 | 0.0000 | forward |
| Dense-ABRBF-reference | 12 | 17.3758 | 3.1992 | 1.5699 | 5.4097 | 32.42 | 0.0000 | forward |
| RationalKAT-AB-v3-g16 | 12 | 9.3639 | 7.2084 | 1.3101 | 6.1471 | 18.02 | 0.0000 | forward |
| RationalKAT-AB-v3-g8 | 12 | 9.2891 | 7.6168 | 1.3101 | 6.3242 | 18.02 | 0.0000 | forward |
| SparseInterpKAN-K16 | 12 | 16.2635 | 5.5026 | 1.3994 | 6.3549 | 46.82 | 0.0000 | forward |
| SparseInterpKAN-K8 | 12 | 16.3343 | 5.3861 | 1.3123 | 6.3035 | 28.77 | 0.0000 | forward |
| SparseSplineKAN-K8 | 12 | 17.1656 | 5.9634 | 1.3127 | 6.7813 | 30.00 | 0.0000 | forward |

P1 exploratory survivors:

```text
none
```

P1 verdict: exploratory survivor list shown above. Regardless of P1 status, P2 was run for the mandatory memory-failure candidates.

## P2 Mandatory Custom Backward Audit

| primitive | variant | grad rel | grad cos | bmem | bwd | saved MB | P2 |
|---|---|---|---|---|---|---|---|
| DWM2-lite-RBFK2 | autograd | 0.00 | 1.0000 | 1.3962 | 5.6908 | 11.9170 | no |
| DWM2-lite-RBFK2 | recompute-backward | 0.00 | 1.0000 | 1.4018 | 13.9063 | 2.0312 | no |
| DWM2-lite-RBFK2 | streaming-coeff-gradient | 0.00 | 1.0000 | 1.4018 | 11.5513 | 2.0312 | no |
| DWM2-lite-RBFK2 | index-only-backward | 0.00 | 1.0000 | 1.4018 | 11.4316 | 2.0312 | no |
| DWM2-lite-RBFK2 | fused-analytic-diagnostic | 0.00 | 1.0000 | 1.4018 | 13.4985 | 2.0312 | no |
| DWM2-lite-LUTK8 | autograd | 0.00 | 1.0000 | 1.4783 | 7.2843 | 54.5078 | no |
| DWM2-lite-LUTK8 | recompute-backward | 0.00 | 1.0000 | 1.4783 | 48.6064 | 2.0312 | no |
| DWM2-lite-LUTK8 | streaming-coeff-gradient | 0.00 | 1.0000 | 1.4783 | 107.4681 | 2.0312 | no |
| DWM2-lite-LUTK8 | index-only-backward | 0.00 | 1.0000 | 1.4783 | 20.8642 | 2.0312 | no |
| DWM2-lite-LUTK8 | fused-analytic-diagnostic | 0.00 | 1.0000 | 1.4783 | 20.6223 | 2.0312 | no |
| SparseInterpKAN-K8 | autograd | 0.00 | 1.0000 | 1.4881 | 9.2182 | 56.7905 | no |
| SparseInterpKAN-K8 | recompute-backward | 0.00 | 1.0000 | 1.4881 | 27.5243 | 2.0312 | no |
| SparseInterpKAN-K8 | streaming-coeff-gradient | 0.00 | 1.0000 | 1.4881 | 100.5915 | 2.0312 | no |
| SparseInterpKAN-K8 | index-only-backward | 0.00 | 1.0000 | 1.4881 | 108.9867 | 2.0312 | no |
| SparseInterpKAN-K8 | fused-analytic-diagnostic | 0.00 | 1.0000 | 1.4881 | 31.1985 | 2.0312 | no |
| SparseSplineKAN-K8 | autograd | 0.00 | 1.0000 | 1.4881 | 11.2681 | 59.0405 | no |
| SparseSplineKAN-K8 | recompute-backward | 0.00 | 1.0000 | 1.4881 | 49.0028 | 2.0312 | no |
| SparseSplineKAN-K8 | streaming-coeff-gradient | 0.00 | 1.0000 | 1.4881 | 133.5919 | 2.0312 | no |
| SparseSplineKAN-K8 | index-only-backward | 0.00 | 1.0000 | 1.4881 | 17.9017 | 2.0312 | no |
| SparseSplineKAN-K8 | fused-analytic-diagnostic | 0.00 | 1.0000 | 1.4881 | 118.8902 | 2.0312 | no |
| RationalKAT-AB-v3-g8 | autograd | 0.00 | 1.0000 | 1.4760 | 10.1216 | 34.7159 | no |
| RationalKAT-AB-v3-g8 | recompute-backward | 0.00 | 1.0000 | 1.4860 | 117.4544 | 2.0312 | no |
| RationalKAT-AB-v3-g8 | streaming-coeff-gradient | 0.00 | 1.0000 | 1.4860 | 23.1802 | 2.0312 | no |
| RationalKAT-AB-v3-g8 | index-only-backward | 0.00 | 1.0000 | 1.4860 | 122.7994 | 2.0312 | no |
| RationalKAT-AB-v3-g8 | fused-analytic-diagnostic | 0.00 | 1.0000 | 1.4860 | 92.1493 | 2.0312 | no |

P2 survivors:

```text
none
```

## P2-P10 Decision

```text
P2 custom backward / recompute: run for mandatory candidates
P3 SparseInterp/SparseSpline: not run; P2 produced no efficiency candidate
P4 RationalKAT-v3: not run; P2 produced no Rational candidate
P5 accelerated convergence: not run; no P3/P4 task-efficiency survivor
P6 joint gate: not run; no convergence survivor
P7/P8/P9/P10: gated behind P6
Final decision: stop_after_p2_no_efficiency_candidate
Route case: A_no_p1_p2_efficiency_candidate
```

## Failure Diagnosis

| failure | count |
|---|---|
| F1_backward_memory_fail | 2 |
| F2_forward_time_fail | 106 |
| F9_gated_not_run | 9 |

Interpretation:

```text
v6.0 explicitly tests whether custom backward/recompute can rescue P1 memory failures.
If P2 still has no efficiency candidate, the blocker remains primitive/kernel/backward design,
not optimizer tuning or functional smoothing.
```

## Required Artifacts

Written under `results/v6_0/`:

```text
p0_core_manifest.csv
p1_profiler_truth.csv
p1_phase_efficiency.csv
p1_roofline_scaling.csv
p1_memory_decomposition.csv
p1_saved_tensor_audit.csv
p1_kernel_breakdown.csv
p2_custom_backward_audit.csv
p2_custom_backward_correctness.csv
p2_custom_backward_memory.csv
p3_sparse_primitive_validation.csv
p3_recipe_grid.csv
p3_task_efficiency_selection.csv
p4_rationalkat_v3_recipe.csv
p5_accelerated_convergence.csv
p6_joint_task_efficiency_convergence.csv
p7_lightsmooth_geometry.csv
p8_functional_training_smoke.csv
p9_candidate_selection3.csv
p10_confirm5.csv
p10_confirm10.csv
failure_table.csv
route_decision.json
aggregate_decision.json
figures/
```

## Final Decision

```text
DG-KAN v6.0 status:
  stop_after_p2_no_efficiency_candidate

Route:
  A_no_p1_p2_efficiency_candidate

What improved:
  v6.0 adds SparseInterp/SparseSpline/RationalKAT-v3 core primitives.
  P2 custom backward diagnostics now run even when P1 has no survivor.
  Profiler truth, roofline proxy, saved tensors, and kernel breakdown are first-class artifacts.

What failed / remains open:
  See the P1-P6 gates above.
  LightSmooth and functional training remain gated until a primitive passes task + efficiency + convergence.

Conclusion:
  If no P2/P6 survivor exists, the next move remains primitive/kernel/backward design,
  not functional optimizer tuning.
```
