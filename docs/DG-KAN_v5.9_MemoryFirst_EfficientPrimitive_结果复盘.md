# DG-KAN v5.9 Memory-First Efficient PureKAN Primitive 结果复盘

本轮依据 `docs/DG-KAN_v5.9_MemoryFirst_EfficientPrimitive_实验计划.md`。核心目标是先确认 primitive 的 profiler truth 和 backward saved tensor 来源，再决定是否值得进入 custom backward、task recipe、LightSmooth 或 functional optimizer。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 core manifest | 12 | 0 |
| P1 phase efficiency | 36 | 0 |
| P1 saved tensor audit | 36 | 0 |
| P2 custom backward | 1 | 0 |
| P3 recipe grid | 1 | 0 |
| P4-P7 gated | 4 | 0 |

## Code / Config Changes

```text
experiments/dgkan_core.py
  Added DWM2LiteDense: single-GEMM KAN residual primitive with RBF/LUT residual modes.

experiments/run_gafu_v59.py
  Added P0 memory-first manifest, P1 saved_tensors_hooks profiler,
  memory decomposition, kernel breakdown, gated P2 custom-backward diagnostics,
  and gated P3-P7 artifacts.

experiments/analyze_gafu_v59.py
  Generates v5.9 summaries, route_decision.json, aggregate_decision.json,
  required SVG diagnostics, and this replay.
```

## P0 Core Manifest

| primitive | class | edge | nonKAN | mixing | rollback | P0 |
|---|---|---|---|---|---|---|
| MLP-reference | MLPClassifier | 0 | 59722 | 0 | 0.0000 | yes |
| Dense-ABRBF-reference | ABRBFDense | 694144 | 0 | 0 | 0.0000 | yes |
| DWM2-prePostMix-current | DWM2Dense | 82514 | 0 | 79588 | 0.0000 | yes |
| DWM2-lite-RBFK2 | DWM2LiteDense | 65184 | 0 | 63104 | 0.0000 | yes |
| DWM2-lite-RBFK4 | DWM2LiteDense | 67264 | 0 | 63104 | 0.0000 | yes |
| DWM2-lite-LUTK8 | DWM2LiteDense | 71424 | 0 | 63104 | 0.0000 | yes |
| RationalKAT-v2-current | RationalKATV2Dense | 69344 | 0 | 63104 | 0.0000 | yes |
| RationalKAT-lite-groups8 | RationalKATV2Dense | 69344 | 0 | 63104 | 0.0000 | yes |
| RationalKAT-lite-groups16 | RationalKATV2Dense | 69344 | 0 | 63104 | 0.0000 | yes |
| LUTKAN-current | LUTKANDense | 71424 | 0 | 63104 | 0.0000 | yes |
| LUTKAN-v2-linearInterp | LUTKANDense | 71424 | 0 | 63104 | 0.0000 | yes |
| CP-ABRBF-r4-reference | GEMMNativeCPABRBFDense | 194696 | 0 | 0 | 0.0000 | yes |

P0 verdict: pass. The new DWM2-lite variants remain strict PureKAN: edge-covered, nonKAN-free, and rollback-exact.

## P1 Profiler Truth / Memory Decomposition

| primitive | rows | fwd | bwd | bmem | step | saved MB | explore | bottleneck |
|---|---|---|---|---|---|---|---|---|
| CP-ABRBF-r4-reference | 3 | 12.7260 | 4.3978 | 3.3879 | 5.1465 | 100.03 | 0.0000 | memory |
| DWM2-lite-LUTK8 | 3 | 11.5534 | 4.5022 | 2.4121 | 4.9355 | 75.51 | 0.0000 | memory |
| DWM2-lite-RBFK2 | 3 | 4.0353 | 2.8083 | 1.9146 | 2.6146 | 17.95 | 0.0000 | forward |
| DWM2-lite-RBFK4 | 3 | 4.0243 | 2.9188 | 2.1792 | 2.6746 | 29.40 | 0.0000 | memory |
| DWM2-prePostMix-current | 3 | 5.8611 | 4.0823 | 2.7599 | 3.7020 | 57.26 | 0.0000 | memory |
| Dense-ABRBF-reference | 3 | 11.9902 | 4.4451 | 3.2072 | 5.2374 | 90.20 | 0.0000 | memory |
| LUTKAN-current | 3 | 11.0385 | 4.1155 | 2.4121 | 4.6201 | 75.51 | 0.0000 | memory |
| LUTKAN-v2-linearInterp | 3 | 11.5914 | 4.2533 | 2.4121 | 4.7980 | 75.51 | 0.0000 | memory |
| RationalKAT-lite-groups16 | 3 | 7.7778 | 6.4465 | 2.3941 | 5.3943 | 42.28 | 0.0000 | memory |
| RationalKAT-lite-groups8 | 3 | 7.7583 | 6.1292 | 2.3941 | 5.2113 | 42.28 | 0.0000 | memory |
| RationalKAT-v2-current | 3 | 7.7474 | 6.4180 | 2.3941 | 5.3725 | 42.28 | 0.0000 | memory |

P1 exploratory survivors:

```text
none
```

P1 verdict: no exploratory survivor. DWM2-lite-RBFK2 is the closest route, but still misses the forward/memory envelope under the written gate.

## P2-P7 Decision

```text
P2 custom backward / recompute: not run; P1 produced no exploratory survivor
P3 minimal task recipe: not run; P2 produced no survivor
P4 LightSmooth compatibility: not run; P3 produced no task-efficiency survivor
P5 functional smoke: not run unless P4 survives
P6/P7 confirm: not run unless P3/P5 survives
Final decision: stop_after_p1_no_efficiency_survivor
Route case: C_no_p1_efficiency_survivor
```

## Failure Diagnosis

| failure | count |
|---|---|
| F1_backward_memory_fail | 29 |
| F2_forward_time_fail | 4 |
| F9_gated_not_run | 6 |

Interpretation:

```text
DWM2-lite reduces saved tensors substantially versus dense AB-RBF and current DWM2,
but it still does not reach the relaxed P1 envelope.
The dominant blocker remains profiler-level primitive efficiency, especially saved tensors /
workspace memory and forward kernel path, so task recipe and functional optimizer remain gated.
```

## Required Artifacts

Written under `results/v5_9/`:

```text
p0_core_manifest.csv
p1_phase_efficiency.csv
p1_memory_decomposition.csv
p1_saved_tensor_audit.csv
p1_kernel_breakdown.csv
p2_custom_backward_correctness.csv
p2_custom_backward_memory.csv
p3_recipe_grid.csv
p3_task_efficiency_selection.csv
p4_lightsmooth_compatibility.csv
p5_functional_smoke.csv
p6_confirm5.csv
p7_confirm10.csv
failure_table.csv
route_decision.json
aggregate_decision.json
figures/
```

## Final Decision

```text
DG-KAN v5.9 status:
  stop_after_p1_no_efficiency_survivor

Route:
  C_no_p1_efficiency_survivor

What improved:
  v5.9 adds saved-tensor-aware profiling and a stricter memory-cause diagnosis.
  DWM2-lite gives a cleaner single-GEMM primitive and materially lowers saved tensor volume.

What failed / remains open:
  No primitive passed P1 exploratory efficiency.
  P2 custom backward, P3 task recipe, LightSmooth, and functional training are correctly gated.

Conclusion:
  The next move is still primitive/kernel/backward design, not optimizer tuning.
```
