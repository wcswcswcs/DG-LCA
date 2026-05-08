# DG-KAN v4.5 Accelerated Functional Optimizer 结果复盘

本轮依据 `docs/DG-KAN_v4.5_AcceleratedFunctionalOptimizer_实验计划.md`。目标是验证 PureKAN 是否需要 AFO：functional-coordinate Adam backbone + trajectory controller + phased geometry。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 coordinate smoke | 24 | 0 |
| P1 AdamW forensic | 60 | 0 |
| P2 FCAdam backbone | 90 | 0 |

## Code / Config Changes

```text
experiments/run_gafu_v45.py
  Added v4.5 coordinate correctness audit across basis_count 16/24, hidden_dim 64/96, depth 2/4.
  Added AdamW trajectory forensic logging for rank, margin, phi, role update share and Adam m/v state.
  Added 20-step FCAdam backbone audit for L2/H1-low/dataSob, AB-RBF-FCAdam and phased/no-geometry variants.

experiments/analyze_gafu_v45.py
  Generates target profiles, P2 gate summaries, failure taxonomy, figures, aggregate_decision.json,
  and this result replay.
```

## P0 Coordinate Correctness

| rows | errors | max nonKAN | max roundtrip | max u->a | max rollback | pass |
|---|---|---|---|---|---|---|
| 24 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | true |

P0 verdict: pass. Functional-coordinate whitening and roundtrip are correct to numerical precision, with strict PureKAN non-KAN parameter count still zero.

## P1 AdamW Trajectory Forensic

| dataset | holdout Δ | rank ratio | phi ratio | input share | block share | output share |
|---|---|---|---|---|---|---|
| Fashion-MNIST | 1.0410 | 0.3267 | 1.0363 | 0.6785 | 0.2101 | 0.1114 |
| KMNIST | 0.4858 | 0.2081 | 1.0345 | 0.6938 | 0.2080 | 0.0982 |
| MNIST | 0.2009 | 0.4730 | 1.0124 | 0.7120 | 0.1983 | 0.0897 |

Observation:

```text
AdamW itself is not a pure geometry-improving trajectory in the first 20 steps.
It lowers holdout loss while reducing effective rank and slightly increasing phi.
This supports v4.5's phased policy: early optimization should prioritize task trajectory,
rank/margin formation, and only later consolidate geometry.
```

## P2 FCAdam Backbone Audit

| dataset | method | runs | hold20 | cos | R2 | rank/A | margin/A | phi/A | bad | P2 |
|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | AB-RBF-FCAdam-H1-low | 3 | 1.2393 | 0.1622 | 0.0906 | 0.7815 | 1.2813 | 1.0261 | 0.0000 | no |
| Fashion-MNIST | D0-allFullSobolev | 3 | 0.7351 | 0.5151 | 0.2674 | 0.4753 | 0.8915 | 0.9497 | 0.0667 | no |
| Fashion-MNIST | D6-allTaskAware | 3 | 0.8802 | 0.6034 | 0.3682 | 0.6601 | 0.9294 | 0.9544 | 0.0833 | no |
| Fashion-MNIST | F4-FNG-leftFullRight | 3 | 1.2013 | 0.2244 | 0.0011 | 0.4744 | 1.0690 | 0.9515 | 0.0167 | no |
| Fashion-MNIST | FCAdam-H1-low | 3 | 1.2070 | -0.0361 | 0.0411 | 0.8224 | 1.3345 | 0.9959 | 0.0000 | no |
| Fashion-MNIST | FCAdam-L2 | 3 | 1.3216 | -0.0630 | 0.0369 | 0.8577 | 1.4672 | 1.0837 | 0.0000 | yes |
| Fashion-MNIST | FCAdam-dataSob | 3 | 1.5207 | -0.0539 | 0.0158 | 0.8212 | 1.8907 | 1.2919 | 0.0000 | no |
| Fashion-MNIST | FCAdam-dataSob-noGeometryGate | 3 | 1.5207 | -0.0539 | 0.0158 | 0.8212 | 1.8907 | 1.2919 | 0.0000 | no |
| Fashion-MNIST | FCAdam-dataSob-phasedGeometry | 3 | 1.5207 | -0.0539 | 0.0158 | 0.8212 | 1.8907 | 1.2919 | 0.0000 | no |
| Fashion-MNIST | PureKAN-AdamW-one-step | 3 | 1.0896 | 0.4556 | 0.2016 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | yes |
| KMNIST | AB-RBF-FCAdam-H1-low | 3 | 0.8285 | 0.3797 | 0.1526 | 0.7674 | 1.2838 | 1.0342 | 0.0000 | no |
| KMNIST | D0-allFullSobolev | 3 | 0.3708 | 0.6193 | 0.3675 | 0.3383 | 0.7678 | 0.9577 | 0.1000 | no |
| KMNIST | D6-allTaskAware | 3 | 0.4430 | 0.8071 | 0.6264 | 0.5162 | 0.8728 | 0.9620 | 0.1167 | no |
| KMNIST | F4-FNG-leftFullRight | 3 | 0.6652 | 0.5413 | 0.2730 | 0.3028 | 0.9775 | 0.9619 | 0.0167 | no |
| KMNIST | FCAdam-H1-low | 3 | 0.8371 | 0.2556 | 0.1084 | 0.8158 | 1.3646 | 1.0114 | 0.0000 | no |
| KMNIST | FCAdam-L2 | 3 | 0.8976 | 0.2770 | 0.1360 | 0.9302 | 1.3593 | 1.1112 | 0.0000 | yes |
| KMNIST | FCAdam-dataSob | 3 | 1.3247 | 0.2068 | 0.1046 | 0.7836 | 2.1900 | 1.3305 | 0.0000 | no |
| KMNIST | FCAdam-dataSob-noGeometryGate | 3 | 1.3247 | 0.2068 | 0.1046 | 0.7836 | 2.1900 | 1.3305 | 0.0000 | no |
| KMNIST | FCAdam-dataSob-phasedGeometry | 3 | 1.3247 | 0.2068 | 0.1046 | 0.7836 | 2.1900 | 1.3305 | 0.0000 | no |
| KMNIST | PureKAN-AdamW-one-step | 3 | 0.5210 | 0.5340 | 0.2127 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | yes |
| MNIST | AB-RBF-FCAdam-H1-low | 3 | 0.7428 | 0.4326 | 0.1582 | 0.9675 | 1.5019 | 1.0532 | 0.0167 | yes |
| MNIST | D0-allFullSobolev | 3 | 0.3983 | 0.6473 | 0.4098 | 0.4566 | 0.9856 | 0.9637 | 0.1333 | no |
| MNIST | D6-allTaskAware | 3 | 0.2977 | 0.8368 | 0.6677 | 0.6147 | 0.7252 | 0.9650 | 0.1333 | no |
| MNIST | F4-FNG-leftFullRight | 3 | 0.6127 | 0.6163 | 0.3532 | 0.3308 | 1.2163 | 0.9662 | 0.1000 | no |
| MNIST | FCAdam-H1-low | 3 | 0.7625 | 0.4143 | 0.1466 | 0.9848 | 1.5791 | 1.0334 | 0.0000 | yes |
| MNIST | FCAdam-L2 | 3 | 0.8712 | 0.3733 | 0.1426 | 1.0802 | 1.6470 | 1.1399 | 0.0167 | yes |
| MNIST | FCAdam-dataSob | 3 | 1.4264 | 0.2950 | 0.1342 | 0.9437 | 2.6652 | 1.3805 | 0.0000 | no |
| MNIST | FCAdam-dataSob-noGeometryGate | 3 | 1.4264 | 0.2950 | 0.1342 | 0.9437 | 2.6652 | 1.3805 | 0.0000 | no |
| MNIST | FCAdam-dataSob-phasedGeometry | 3 | 1.4264 | 0.2950 | 0.1342 | 0.9437 | 2.6652 | 1.3805 | 0.0000 | no |
| MNIST | PureKAN-AdamW-one-step | 3 | 0.2652 | 0.5462 | 0.2059 | 1.0000 | 1.0000 | 1.0000 | 0.0167 | yes |

Best non-Adam points by 20-step holdout descent:

| dataset | best hold20 | hold20 | cos | R2 | rank/A | phi/A |
|---|---|---|---|---|---|---|
| MNIST | FCAdam-dataSob | 1.4264 | 0.2950 | 0.1342 | 0.9437 | 1.3805 |
| Fashion-MNIST | FCAdam-dataSob | 1.5207 | -0.0539 | 0.0158 | 0.8212 | 1.2919 |
| KMNIST | FCAdam-dataSob | 1.3247 | 0.2068 | 0.1046 | 0.7836 | 1.3305 |

P2 survivors:

```text
none
```

P2 verdict:

```text
No candidate enters P3.

FCAdam-dataSob has very strong 20-step holdout descent on all datasets, but its
function-space alignment to the AdamW target trajectory collapses over 20 steps.
AB-RBF-FCAdam improves some rank/margin behavior but still fails the global
cos/R2 requirements.

The v4.5 positive signal is real task descent, not a solved optimizer.
The blocker moved from immediate descent to temporal trajectory mismatch:
the methods learn locally, but their multi-step function displacement no longer
resembles AdamW enough to satisfy the written AFO backbone gate.
```

## P3-P8 Decision

```text
P3 Functional Nesterov / Adan-like dynamics: not run.
P4 Lyapunov restart controller: not run.
P5 rank/margin preservation ablation: not run.
P6 3-seed full-budget selection: not run.
P7/P8 confirm: not run.

Reason: P2 produced no survivor under the written gate.
```

## P9 Failure Diagnosis

Generated:

```text
p9_failure_diagnosis.csv
failure_table.csv
failure_type_heatmap_by_dataset.csv
failure_type_heatmap_by_method.csv
rank_margin_phi_scatter.csv
```

Diagnosis:

```text
F1 coordinate mismatch:
  Dominant after 20 steps. FCAdam-dataSob descends strongly, but cos/R2 to AdamW's
  function trajectory is far below the P2 requirement.

F3/F4 rank and margin:
  Some variants preserve rank/margin, but not together with Adam-like function alignment.

F5 geometry conflict:
  dataSob variants often have phi/A > 1.0. This is acceptable in Phase I up to 1.15,
  but it still does not compensate for low function alignment.

F8 temporal dynamics:
  The core unsolved issue is now temporal: one-step/short-step descent exists,
  but the 20-step trajectory drifts away from AdamW.
```

## Required Artifacts

Written under `results/v4_5/`:

```text
p0_coordinate_invariants.csv
p1_adamw_trajectory_forensic.csv
p1_adamw_target_profile.csv
p2_fcadam_backbone_audit.csv
p2_fcadam_gate_summary.csv
optimizer_dynamics_trace.csv
p3_temporal_dynamics.csv
p4_lyapunov_restart.csv
p5_rank_margin_ablation.csv
p6_candidate_selection.csv
p7_confirm5.csv
p8_confirm10.csv
p9_failure_diagnosis.csv
failure_table.csv
failure_type_heatmap_by_dataset.csv
failure_type_heatmap_by_method.csv
rank_margin_phi_scatter.csv
aggregate_decision.json
figures/
```

## Final Decision

```text
PureKAN Accelerated Functional Optimizer is not solved in v4.5.

What improved:
  FCAdam-dataSob confirms strong task-learning descent over 20 steps.
  P0 fully validates the function-coordinate implementation.
  P1 gives a concrete AdamW target profile and confirms phased geometry is the right framing.

What failed:
  FCAdam variants do not preserve enough AdamW function-space trajectory over 20 steps.
  No candidate satisfies the P2 global cos/R2 + rank/margin + Phase-I phi budget gate.
  P3/P4/P5 expansion is therefore not justified.

Conclusion:
  v4.5 supports the idea that functional-coordinate adaptive dynamics are the right
  direction, but the current AFO backbone still lacks temporal trajectory control.
  The next redesign should target multi-step function trajectory matching or restart
  dynamics only after improving P2 cos/R2 retention.
```
