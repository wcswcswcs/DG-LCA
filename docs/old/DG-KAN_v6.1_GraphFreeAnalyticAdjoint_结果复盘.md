# DG-KAN v6.1 Graph-Free Analytic Adjoint 结果复盘

本轮依据 `docs/DG-KAN_v6.1_GraphFreeAnalyticAdjoint_实验计划.md`。核心目标是把“functional update rule”和“analytic adjoint / graph-free training”分开：先确认不依赖 `loss.backward()` 的 manual forward/backward/update 是否正确且更省显存，再决定是否进入 task recipe、LightSmooth 或 functional optimizer。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 graph-free invariants | 6 | 0 |
| P1 analytic gradient | 15 | 0 |
| P2 graph-free microbenchmark | 108 | 0 |
| P3 primitive selection | 1 | 0 |
| P4 manual smoke | 1 | 0 |
| P5 manual Adam | 1 | 0 |
| P6 functional no autograd | 1 | 0 |
| P7-P10 gated | 5 | 0 |

## Code / Config Changes

```text
experiments/run_gafu_v61.py
  Added graph-free ManualLayer / ManualStack primitives for SparseInterp, LUT, RBF-lite, and Rational-lite.
  Added P0 graph-free invariants, P1 analytic gradient correctness, P2 graph-free efficiency benchmark,
  and gated P3-P10 artifacts.

experiments/analyze_gafu_v61.py
  Generates graph-free gate summaries, route_decision.json, aggregate_decision.json,
  required SVG diagnostics, failure taxonomy, and this replay.
```

## P0 Graph-Free Invariants

| primitive | edge | nonKAN | loss.backward | saved | rollback | P0 |
|---|---|---|---|---|---|---|
| MLP-AdamW-autograd-reference | 0 | 59722 | 1 | 0 | 0.0000 | no |
| Dense-ABRBF-autograd-reference | 694144 | 0 | 1 | 0 | 0.0000 | no |
| SparseInterpKAN-manual | 1296 | 0 | 0 | 0 | 0.0000 | yes |
| DWM2-lite-RBFK2-manual | 864 | 0 | 0 | 0 | 0.0000 | yes |
| DWM2-lite-LUTK8-manual | 1152 | 0 | 0 | 0 | 0.0000 | yes |
| RationalKAT-lite-manual | 1056 | 0 | 0 | 0 | 0.0000 | yes |

P0 verdict: pass for manual primitives when edge coverage is nonzero, rollback is exact, and no autograd saved-tensor path is used.

## P1 Analytic Gradient Correctness

| primitive | rows | max coeff rel | min coeff cos | max input rel | min input cos | P1 rate |
|---|---|---|---|---|---|---|
| DWM2-lite-LUTK8-manual | 3 | 0.00 | 1.0000 | 0.00 | 1.0000 | 1.0000 |
| DWM2-lite-RBFK2-manual | 3 | 0.00 | 1.0000 | 0.00 | 1.0000 | 1.0000 |
| RationalKAT-lite-manual | 3 | 0.00 | 1.0000 | 0.00 | 1.0000 | 1.0000 |
| SparseInterpKAN-K16-manual | 3 | 0.00 | 1.0000 | 0.00 | 1.0000 | 1.0000 |
| SparseInterpKAN-K8-manual | 3 | 0.00 | 1.0000 | 0.00 | 1.0000 | 1.0000 |

P1 survivors:

```text
DWM2-lite-LUTK8-manual, DWM2-lite-RBFK2-manual, RationalKAT-lite-manual, SparseInterpKAN-K16-manual, SparseInterpKAN-K8-manual
```

## P2 Graph-Free Efficiency Benchmark

| primitive | rows | fwd | bwd | bmem | step | cache MB | P2 rate |
|---|---|---|---|---|---|---|---|
| DWM2-lite-LUTK8-manual | 12 | 5.4254 | 4.6017 | 1.1749 | 3.5006 | 1.08 | 0.0000 |
| DWM2-lite-RBFK2-manual | 12 | 2.1639 | 2.0543 | 1.1747 | 1.5584 | 1.08 | 0.0000 |
| DWM2-lite-RBFK4-manual | 12 | 2.2257 | 2.1993 | 1.3012 | 1.6476 | 1.08 | 0.0000 |
| Dense-ABRBF-autograd-reference | 12 | 12.4203 | 6.5919 | 1.6739 | 6.2057 | 0.00 | 0.0000 |
| RationalKAT-lite-manual | 12 | 4.0184 | 4.7778 | 1.2661 | 3.3437 | 1.08 | 0.0000 |
| SparseInterpKAN-K16-manual | 12 | 6.4512 | 6.2625 | 1.2069 | 4.5773 | 1.08 | 0.0000 |
| SparseInterpKAN-K32-manual | 12 | 6.3820 | 6.0818 | 1.2105 | 4.4738 | 1.08 | 0.0000 |
| SparseInterpKAN-K8-manual | 12 | 6.4231 | 6.2056 | 1.2051 | 4.5483 | 1.08 | 0.0000 |

P2 survivors:

```text
none
```

## P3-P10 Decision

```text
P3 primitive efficiency selection: not run; P2 produced no graph-free efficiency survivor
P4 manual training smoke: gated behind P3
P5 manual Adam convergence: gated behind P4
P6 functional without autograd: gated behind P5
P7/P8/P9/P10 confirm: gated behind P6
Final decision: stop_after_p2_no_graphfree_efficiency_survivor
Route case: D_no_primitive_passed_p2_efficiency
```

## Failure Diagnosis

| failure | count |
|---|---|
| F3_forward_time_fail | 84 |
| F9_gated_not_run | 9 |

Interpretation:

```text
v6.1 verifies the graph-free analytic-adjoint premise directly.
If P1 fails, the blocker is manual adjoint correctness.
If P1 passes but P2 fails, the blocker is still primitive/cache/kernel efficiency,
not LightSmooth or functional optimizer design.
```

## Required Artifacts

Written under `results/v6_1/`:

```text
p0_graphfree_invariants.csv
p0_graphfree_gate_summary.csv
p1_analytic_gradient_correctness.csv
p1_gradient_gate_summary.csv
p2_graphfree_microbenchmark.csv
p2_efficiency_selection.csv
p2_efficiency_gate_summary.csv
p3_primitive_efficiency_selection.csv
p4_manual_training_smoke.csv
p5_manual_adam_convergence.csv
p6_functional_without_autograd.csv
p7_acceleration_package.csv
p8_manual_lightsmooth_geometry.csv
p9_joint_selection3.csv
p10_confirm5.csv
p10_confirm10.csv
failure_table.csv
route_decision.json
aggregate_decision.json
figures/
```

## Final Decision

```text
DG-KAN v6.1 status:
  stop_after_p2_no_graphfree_efficiency_survivor

Route:
  D_no_primitive_passed_p2_efficiency

What improved:
  v6.1 adds a true graph-free manual-adjoint benchmark path.
  P1 compares manual gradients against autograd without making autograd the training path.
  P2 measures graph-free cache/memory/time ratios directly against an MLP autograd reference.

What failed / remains open:
  See P1-P3 gates above.
  Task recipe, LightSmooth, and functional optimizer remain gated until graph-free adjoint + efficiency survive.

Conclusion:
  v6.1 makes the next blocker explicit: either fix the manual adjoint formulas, or redesign the primitive/cache path
  until a graph-free candidate passes the P2 efficiency envelope.
```
