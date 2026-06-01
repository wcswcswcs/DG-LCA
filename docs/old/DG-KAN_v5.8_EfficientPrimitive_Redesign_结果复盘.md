# DG-KAN v5.8 Efficient Primitive Redesign 结果复盘

本轮依据 `docs/DG-KAN_v5.8_EfficientPrimitive_Redesign_实验计划.md`。核心目标是在继续 functional optimizer / LightSmooth 之前，重新设计并验证更接近 MLP-like kernel path 的 strict PureKAN primitive。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 core manifest | 9 | 0 |
| P1 efficiency | 27 | 0 |
| P2 DWM-v2 recipe | 1 | 0 |
| P3 RationalKAT-v2 recipe | 1 | 0 |
| P4 LUTKAN feasibility | 1 | 0 |
| P5 joint selection | 1 | 0 |
| P6-P10 gated | 5 | 0 |

## Code / Config Changes

```text
experiments/dgkan_core.py
  Added core efficient primitive candidates:
    DWM2Dense
    RationalKATV2Dense
    LUTKANDense
  Extended edge/base/RBF/mixing parameter discovery for the new primitives.

experiments/run_gafu_v58.py
  Added P0 core manifest, P1 small/medium/large efficiency decomposition,
  gated DWM-v2 / RationalKAT-v2 / LUTKAN recipe probes,
  joint selection, and required gated artifacts.

experiments/analyze_gafu_v58.py
  Generates v5.8 gate summaries, route_decision.json, aggregate_decision.json,
  failure taxonomy, figures, and this replay.
```

## P0 Core Manifest

| primitive | class | edge | nonKAN | mixing | rollback | P0 |
|---|---|---|---|---|---|---|
| MLP-reference | MLPClassifier | 0 | 59722 | 0 | 0.0000 | yes |
| Dense-ABRBF-reference | ABRBFDense | 694144 | 0 | 0 | 0.0000 | yes |
| DWM-v1-current | GEMMNativeDepthwiseMixDense | 74544 | 0 | 63104 | 0.0000 | yes |
| DWM2-prePostMix | DWM2Dense | 82514 | 0 | 79588 | 0.0000 | yes |
| DWM2-gated | DWM2Dense | 82780 | 0 | 79588 | 0.0000 | yes |
| DWM2-residual | DWM2Dense | 82514 | 0 | 79588 | 0.0000 | yes |
| RationalKAT-AB-v2 | RationalKATV2Dense | 69344 | 0 | 63104 | 0.0000 | yes |
| LUTKAN-AB | LUTKANDense | 71424 | 0 | 63104 | 0.0000 | yes |
| CP-ABRBF-reference | GEMMNativeCPABRBFDense | 200080 | 0 | 0 | 0.0000 | yes |

P0 verdict: core manifest passes when each strict PureKAN primitive is edge-covered, nonKAN-free, and rollback-exact.

## P1 Efficiency Decomposition

| method | rows | fwd | bwd | bmem | step | explore | final | bottleneck |
|---|---|---|---|---|---|---|---|---|
| CP-ABRBF-reference | 3 | 13.7354 | 3.7246 | 3.9940 | 4.2614 | 0.0000 | 0.0000 | memory |
| DWM-v1-current | 3 | 12.1642 | 2.7904 | 3.5011 | 3.4751 | 0.0000 | 0.0000 | memory |
| DWM2-gated | 3 | 7.1559 | 3.9212 | 3.3212 | 3.5408 | 0.0000 | 0.0000 | memory |
| DWM2-prePostMix | 3 | 6.1837 | 3.1438 | 3.1843 | 2.9472 | 0.0000 | 0.0000 | memory |
| DWM2-residual | 3 | 6.3861 | 3.4361 | 3.1979 | 3.1471 | 0.0000 | 0.0000 | memory |
| Dense-ABRBF-reference | 3 | 12.7981 | 3.2618 | 4.0062 | 4.0050 | 0.0000 | 0.0000 | memory |
| LUTKAN-AB | 3 | 11.8364 | 3.1283 | 2.3854 | 3.5872 | 0.0000 | 0.0000 | memory |
| RationalKAT-AB-v2 | 3 | 7.7779 | 4.9495 | 2.2472 | 4.2200 | 0.0000 | 0.0000 | memory |

P1 exploratory survivors:

```text
none
```

P1 final survivors:

```text
none
```

## P2 DWM-v2 Recipe

_P2 not run or no recipe rows._

P2 all-dataset survivors:

```text
none
```

## P3 RationalKAT-v2 Recipe

_P3 not run or no recipe rows._

P3 all-dataset survivors:

```text
none
```

## P4 LUTKAN Feasibility

_P4 not run or no recipe rows._

P4 all-dataset survivors:

```text
none
```

## P5-P10 Decision

```text
P5 joint task-efficiency selection: not run / no survivor
P6 custom backward audit: not run
P7 LightSmooth compatibility: not run unless P6 survives
P8 functional training smoke: not run unless P7 survives
P9/P10 confirm: not run unless P8 survives
Final decision: stop_after_p1_no_efficiency_survivor
Route case: E_no_efficient_primitive
```

## Failure Diagnosis

| failure | count |
|---|---|
| F2_memory_fail | 24 |
| F4_gated_not_run | 4 |

## Required Artifacts

Written under `results/v5_8/`:

```text
p0_core_manifest.csv
p1_efficiency_decomposition.csv
p1_component_timing.csv
p1_memory_decomposition.csv
p2_dwm_v2_recipe.csv
p3_rationalkat_v2_recipe.csv
p4_lutkan_feasibility.csv
p5_joint_task_efficiency.csv
p6_custom_backward_audit.csv
p7_lightsmooth_compatibility.csv
p8_functional_training_smoke.csv
p9_confirm5.csv
p10_confirm10.csv
failure_table.csv
failure_by_primitive.csv
failure_by_dataset.csv
failure_by_gate.csv
route_decision.json
aggregate_decision.json
figures/
```

## Final Decision

```text
DG-KAN v5.8 status:
  stop_after_p1_no_efficiency_survivor

Route:
  E_no_efficient_primitive

What improved:
  v5.8 adds three redesigned strict PureKAN primitive families in core.
  P1 now measures small/medium/large efficiency decomposition before recipe spending.
  Functional optimizer and LightSmooth remain correctly gated behind primitive survival.

What failed / remains open:
  See the P1-P5 gates above.
  If no P1/P5 survivor exists, the bottleneck remains primitive efficiency or task recipe,
  not functional smoothing.

Conclusion:
  v5.8 follows the kernel-first rule: do not resume FGO-v3 until an efficient primitive
  satisfies the joint task + efficiency envelope.
```
