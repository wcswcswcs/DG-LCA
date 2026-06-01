# DG-KAN v6.2 Graph-Free Kernel / Cache Redesign 结果复盘

本轮依据 `docs/DG-KAN_v6.2_GraphFreeKernelCache_实验计划.md`。目标是在 v6.1 已验证 analytic adjoint 正确之后，继续拆 kernel/cache 效率，并允许 near-miss 候选进入小规模 wall-clock 收敛补偿验证。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 cache manifest | 10 | 0 |
| P1 gradient correctness | 48 | 0 |
| P2 efficiency v2 | 108 | 0 |
| P3 kernel/cache ablation | 18 | 0 |
| P4 near-miss convergence | 27 | 0 |
| P5-P9 gated | 5 | 0 |

## Code / Config Changes

```text
experiments/run_gafu_v62.py
  Added graph-free variants:
    MLP-manual-linear-reference
    DWM2-lite-RBFK2-cacheMin
    DWM2-lite-poly2 / piecewiseLinear / fastRational
    SparseInterpKAN vectorized/fusedIndex diagnostics
    RationalKAT-lite-fastpoly
  Added cache decomposition, op/kernel counters, P3 ablation, and P4 near-miss convergence smoke.

experiments/analyze_gafu_v62.py
  Generates v6.2 gate summaries, route_decision.json, aggregate_decision.json,
  required SVG diagnostics, failure taxonomy, and this replay.
```

## P0 Graph-Free Cache Manifest

| primitive | edge | nonKAN | loss.backward | cache | x | hidden | P0 |
|---|---|---|---|---|---|---|---|
| MLP-autograd-reference | 0 | 59722 | 1 | 0.0000 | 0.0000 | 0.0000 | yes |
| MLP-manual-linear-reference | 768 | 0 | 0 | 0.0015 | 0.0010 | 0.0005 | yes |
| Dense-ABRBF-autograd-reference | 694144 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | yes |
| DWM2-lite-RBFK2-manual-v61 | 864 | 0 | 0 | 0.0015 | 0.0010 | 0.0005 | yes |
| DWM2-lite-RBFK2-manual-cacheMin | 864 | 0 | 0 | 0.0015 | 0.0010 | 0.0005 | yes |
| DWM2-lite-poly2-manual | 864 | 0 | 0 | 0.0015 | 0.0010 | 0.0005 | yes |
| SparseInterpKAN-K8-manual-v61 | 1296 | 0 | 0 | 0.0015 | 0.0010 | 0.0005 | yes |
| SparseInterpKAN-K8-manual-vectorized | 1296 | 0 | 0 | 0.0015 | 0.0010 | 0.0005 | yes |
| RationalKAT-lite-manual-v61 | 1056 | 0 | 0 | 0.0015 | 0.0010 | 0.0005 | yes |
| RationalKAT-lite-manual-fastpoly | 912 | 0 | 0 | 0.0015 | 0.0010 | 0.0005 | yes |

## P1 Gradient / Streaming Correctness

| primitive | rows | max coeff rel | min coeff cos | max input rel | pass |
|---|---|---|---|---|---|
| DWM2-lite-RBFK2-manual-cacheMin | 6 | 0.00 | 1.0000 | 0.00 | 1.0000 |
| DWM2-lite-fastRational-manual | 6 | 0.00 | 1.0000 | 0.00 | 1.0000 |
| DWM2-lite-piecewiseLinear-manual | 6 | 0.00 | 1.0000 | 0.00 | 1.0000 |
| DWM2-lite-poly2-manual | 6 | 0.00 | 1.0000 | 0.00 | 1.0000 |
| MLP-manual-linear-reference | 6 | 0.00 | 1.0000 | 0.00 | 1.0000 |
| RationalKAT-lite-manual-fastpoly | 6 | 0.00 | 1.0000 | 0.00 | 1.0000 |
| SparseInterpKAN-K8-manual-fusedIndex | 6 | 0.00 | 1.0000 | 0.00 | 1.0000 |
| SparseInterpKAN-K8-manual-vectorized | 6 | 0.00 | 1.0000 | 0.00 | 1.0000 |

P1 survivors:

```text
DWM2-lite-RBFK2-manual-cacheMin, DWM2-lite-fastRational-manual, DWM2-lite-piecewiseLinear-manual, DWM2-lite-poly2-manual, MLP-manual-linear-reference, RationalKAT-lite-manual-fastpoly, SparseInterpKAN-K8-manual-fusedIndex, SparseInterpKAN-K8-manual-vectorized
```

## P2 Graph-Free Efficiency V2

| primitive | rows | fwd | bwd | step | bmem | cache | near |
|---|---|---|---|---|---|---|---|
| DWM2-lite-RBFK2-cacheMin | 12 | 2.0948 | 1.8540 | 1.4367 | 1.4954 | 1.0755 | 0.3333 |
| DWM2-lite-fastRational | 12 | 2.3495 | 2.4963 | 1.8142 | 1.5795 | 1.0755 | 0.1667 |
| DWM2-lite-piecewiseLinear | 12 | 5.4187 | 4.9179 | 3.5616 | 1.5541 | 1.0755 | 0.0000 |
| DWM2-lite-poly2 | 12 | 1.6616 | 1.6062 | 1.2415 | 1.1731 | 1.0755 | 0.6667 |
| MLP-manual-linear-reference | 12 | 0.5698 | 0.4572 | 0.4455 | 1.0071 | 1.0755 | 1.0000 |
| RationalKAT-lite-fastpoly | 12 | 2.3480 | 2.4991 | 1.8146 | 1.5795 | 1.0755 | 0.1667 |
| SparseInterpKAN-K8-fusedIndex | 12 | 6.2795 | 6.2028 | 4.3895 | 1.6454 | 1.0755 | 0.0000 |
| SparseInterpKAN-K8-vectorized | 12 | 6.3842 | 6.3049 | 4.4620 | 1.6454 | 1.0755 | 0.0000 |

P2 final survivors:

```text
none
```

P2 near-miss candidates:

```text
DWM2-lite-RBFK2-cacheMin, DWM2-lite-fastRational, DWM2-lite-poly2, MLP-manual-linear-reference, RationalKAT-lite-fastpoly
```

## P4 Near-Miss Convergence Smoke

| dataset | method | runs | acc | gap | auc time Δ | step | P4 |
|---|---|---|---|---|---|---|---|
| Fashion-MNIST | DWM2-lite-RBFK2-cacheMin | 3 | 0.7734 | 0.0143 | 0.0076 | 1.4367 | no |
| Fashion-MNIST | DWM2-lite-poly2 | 3 | 0.7682 | 0.0195 | 0.0100 | 1.2415 | yes |
| Fashion-MNIST | MLP-AdamW-autograd-reference | 3 | 0.7878 | 0.0000 | 0.0000 | 1.0000 | yes |
| KMNIST | DWM2-lite-RBFK2-cacheMin | 3 | 0.6615 | 0.0234 | -0.0508 | 1.4367 | no |
| KMNIST | DWM2-lite-poly2 | 3 | 0.6445 | 0.0404 | -0.0226 | 1.2415 | no |
| KMNIST | MLP-AdamW-autograd-reference | 3 | 0.6849 | 0.0000 | 0.0000 | 1.0000 | yes |
| MNIST | DWM2-lite-RBFK2-cacheMin | 3 | 0.8919 | 0.0039 | -0.0236 | 1.4367 | no |
| MNIST | DWM2-lite-poly2 | 3 | 0.8854 | 0.0104 | -0.0160 | 1.2415 | no |
| MNIST | MLP-AdamW-autograd-reference | 3 | 0.8958 | 0.0000 | 0.0000 | 1.0000 | yes |

P4 survivors:

```text
none
```

## P5-P9 Decision

```text
P5 acceleration package: not run; P4 produced no near-miss convergence survivor
P6 geometry maintenance: gated behind P5
P7 3-seed joint selection: gated behind P6
P8/P9 confirm: gated behind P7
Final decision: stop_after_p4_no_nearmiss_convergence_survivor
Route case: B_nearmiss_no_convergence_compensation
```

## Failure Diagnosis

| failure | count |
|---|---|
| F2_forward_time_fail | 77 |
| F3_backward_time_fail | 1 |
| F5_backward_memory_fail | 8 |
| F6_nearmiss_convergence_fail | 15 |
| F9_gated_not_run | 5 |

Interpretation:

```text
v6.2 checks whether the v6.1 near miss can be explained by cache/kernel overhead and
whether slower step time can be compensated by faster convergence.
If manual MLP is also slow, the runtime framework is the blocker.
If near-miss candidates fail P4, the remaining blocker is still kernel/cache plus task recipe,
not LightSmooth or functional geometry.
```

## Required Artifacts

Written under `results/v6_2/`:

```text
p0_graphfree_cache_manifest.csv
p0_cache_manifest_summary.csv
p1_gradient_streaming_correctness.csv
p1_gradient_gate_summary.csv
p2_graphfree_efficiency_v2.csv
p2_efficiency_selection.csv
p2_efficiency_gate_summary.csv
p3_kernel_cache_ablation.csv
p4_nearmiss_convergence_smoke.csv
p4_training_trace.csv
p4_nearmiss_gate_summary.csv
p5_acceleration_package.csv
p6_manual_geometry_maintenance.csv
p7_joint_selection3.csv
p8_confirm5.csv
p9_confirm10.csv
failure_table.csv
route_decision.json
aggregate_decision.json
figures/
```

## Final Decision

```text
DG-KAN v6.2 status:
  stop_after_p4_no_nearmiss_convergence_survivor

Route:
  B_nearmiss_no_convergence_compensation

What improved:
  v6.2 decomposes manual cache and kernel/op counts instead of reporting only one cache MB.
  DWM2-lite alternatives test whether exp/RBF is the main time bottleneck.
  Near-miss candidates are allowed into a compact convergence-compensation smoke.

What failed / remains open:
  See P2/P4 gates above.
  Functional optimizer and LightSmooth remain gated until graph-free primitive efficiency
  and wall-clock convergence are jointly credible.

Conclusion:
  v6.2 keeps the graph-free route alive only if manual runtime and near-miss convergence justify it;
  otherwise the next move remains kernel/cache design.
```
