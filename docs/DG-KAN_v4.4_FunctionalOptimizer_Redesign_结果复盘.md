# DG-KAN v4.4 Functional Optimizer Redesign 结果复盘

本轮依据 `docs/DG-KAN_v4.4_FunctionalOptimizer_Redesign_实验计划.md`。目标是验证 PureKAN functional optimizer 是否应该改写为 functional-coordinate Adam、Shadow-Adam function distillation 或 global output-space kernel step。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 smoke | 33 | 0 |
| P1 trajectory | 48 | 0 |
| P2 horizon | 144 | 0 |

## Code / Config Changes

```text
experiments/run_gafu_v44.py
  Added FCAdam-v2 functional-coordinate Adam with L2/H1/H1-low/dataSob metrics.
  Added SFD direct/prox/residual using shadow AdamW layer-output displacement as teacher.
  Added GFK diagnostic output/block/all-lowrank target-fit steps.
  Added AB-RBF PureKAN with constant/linear edge basis and no non-KAN trainable params.
  P1 records coefficient/function alignment to AdamW, R2, rank/margin/geometry changes.
  P2 records one-step and five-step horizon metrics for the written gate.

experiments/analyze_gafu_v44.py
  Generates P1/P2 gate summaries, failure reports, required placeholder artifacts,
  aggregate_decision.json, SVG diagnostics, and this replay.
```

## P0 Implementation Invariants

| rows | errors | strict | max nonKAN | min cov | max rollback | pass |
|---|---|---|---|---|---|---|
| 33 | 0 | 33 | 0.0000 | 1.0000 | 0.0000 | true |

P0 verdict: pass. All successful rows keep strict PureKAN trainable parameters at zero non-KAN params, coefficient coverage at 1.0, fixed alpha, and exact rollback.

## P1 AdamW Functional Trajectory Audit

| method | train all | holdout+ | cos f | R2 | cos coeff | rank Δ | phi ratio | P1 |
|---|---|---|---|---|---|---|---|---|
| AB-RBF-FCAdam-H1-low | 1 | 1 | 0.9746 | 0.9345 | 0.0578 | -13.3237 | 1.0084 | no |
| D0-allFullSobolev | 1 | 3 | 0.6776 | 0.3497 | 0.1361 | -5.9794 | 0.9963 | no |
| D6-allTaskAware | 0 | 2 | 0.8701 | 0.7332 | 0.6945 | -12.4961 | 1.0006 | no |
| EKFNG-leftRightFull-lowSob | 0 | 0 | 0.4763 | -59.8609 | 0.2657 | -23.6604 | 1.5052 | no |
| F4-FNG-leftFullRight | 0 | 1 | 0.6199 | 0.0844 | 0.1093 | -11.6313 | 0.9941 | no |
| FCAdam-H1-low | 1 | 1 | 0.8620 | 0.7100 | 0.0327 | -10.3199 | 1.0019 | no |
| FCAdam-L2 | 0 | 1 | 0.8949 | 0.7862 | 0.0298 | -12.2748 | 1.0031 | no |
| FCAdam-dataSob | 1 | 2 | 0.8752 | 0.7278 | 0.2083 | -7.6473 | 1.0176 | no |
| FPA-edge-prox | 1 | 2 | 0.8761 | 0.7368 | 0.9508 | -11.1536 | 1.0003 | no |
| GFK-all-lowrank | 1 | 3 | 0.1076 | 0.0009 | 0.0277 | 0.0003 | 1.0000 | no |
| GFK-block-output | 1 | 3 | 0.1096 | 0.0009 | 0.0081 | 0.0001 | 1.0000 | no |
| GFK-output-only | 1 | 3 | 0.1105 | 0.0008 | -0.0041 | 0.0000 | 1.0000 | no |
| PureKAN-AdamW-one-step | 1 | 2 | 1.0000 | 1.0000 | 1.0000 | -17.7872 | 1.0036 | yes |
| SFD-direct | 1 | 3 | 0.6416 | 0.3033 | 0.6115 | -3.6581 | 0.9995 | no |
| SFD-prox | 1 | 2 | 0.5014 | 0.1369 | 0.6115 | -1.3635 | 0.9995 | no |
| SFD-residual | 1 | 2 | 0.4239 | 0.0845 | 0.6115 | -0.8752 | 1.0006 | no |

P1 survivors:

```text
none
```

Observation:

```text
FCAdam and AB-RBF-FCAdam produce high function-space R2 on several datasets.
SFD often gives positive local descent but weaker R2 to AdamW function displacement.
GFK is stable but the output displacement is extremely small, with R2 near zero.
EKFNG-leftRightFull-lowSob remains numerically aggressive and fails local direction on multiple datasets.
```

## P2 One-Step / Five-Step Horizon Gate

| dataset | method | runs | hold5 | bad | R2 | rank/A | margin/A | phi/A | P2 |
|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | AB-RBF-FCAdam-H1-low | 3 | 0.4513 | 0.0000 | 0.4130 | 0.8440 | 1.0224 | 1.0266 | no |
| Fashion-MNIST | D0-allFullSobolev | 3 | 0.2931 | 0.0000 | 0.3129 | 0.2719 | 0.7868 | 0.9885 | no |
| Fashion-MNIST | D6-allTaskAware | 3 | 0.4225 | 0.0000 | 0.5640 | 0.5987 | 1.0053 | 0.9869 | no |
| Fashion-MNIST | EKFNG-leftRightFull-lowSob | 3 | 0.3971 | 0.2000 | -3.8822 | 1.0154 | 0.2633 | 1.3667 | no |
| Fashion-MNIST | F4-FNG-leftFullRight | 3 | 0.5557 | 0.0000 | 0.1676 | 0.5282 | 1.0238 | 0.9879 | no |
| Fashion-MNIST | FCAdam-H1-low | 3 | 0.4858 | 0.0667 | 0.2540 | 0.8169 | 1.1080 | 0.9961 | no |
| Fashion-MNIST | FCAdam-L2 | 3 | 0.5208 | 0.0667 | 0.2860 | 0.8054 | 1.1289 | 1.0122 | no |
| Fashion-MNIST | FCAdam-dataSob | 3 | 0.6420 | 0.0000 | 0.2358 | 0.6949 | 1.1497 | 1.0642 | no |
| Fashion-MNIST | FPA-edge-prox | 3 | 0.4947 | 0.0000 | 0.5831 | 0.6860 | 0.9521 | 0.9924 | no |
| Fashion-MNIST | GFK-all-lowrank | 3 | 0.0075 | 0.0000 | 0.0009 | 0.0000 | 0.0173 | 0.9874 | no |
| Fashion-MNIST | GFK-block-output | 3 | 0.0075 | 0.0000 | 0.0009 | -0.0000 | 0.0171 | 0.9875 | no |
| Fashion-MNIST | GFK-output-only | 3 | 0.0074 | 0.0000 | 0.0008 | -0.0000 | 0.0170 | 0.9875 | no |
| Fashion-MNIST | PureKAN-AdamW-one-step | 3 | 0.4185 | 0.0000 | 0.4785 | 1.0000 | 1.0000 | 1.0000 | yes |
| Fashion-MNIST | SFD-direct | 3 | 0.3836 | 0.0000 | 0.1034 | 0.4886 | 0.9900 | 0.9872 | no |
| Fashion-MNIST | SFD-prox | 3 | 0.2452 | 0.0000 | 0.0700 | 0.1699 | 0.6408 | 0.9880 | no |
| Fashion-MNIST | SFD-residual | 3 | 0.1831 | 0.0000 | 0.0609 | 0.0986 | 0.5217 | 0.9885 | no |
| KMNIST | AB-RBF-FCAdam-H1-low | 3 | 0.2502 | 0.0000 | 0.4450 | 0.8716 | 1.5318 | 1.0234 | no |
| KMNIST | D0-allFullSobolev | 3 | 0.1049 | 0.0667 | 0.4372 | 0.2019 | 0.6939 | 0.9860 | no |
| KMNIST | D6-allTaskAware | 3 | 0.1421 | 0.0000 | 0.6536 | 0.4635 | 0.7980 | 0.9859 | no |
| KMNIST | EKFNG-leftRightFull-lowSob | 3 | -1.0316 | 0.3333 | -6.5442 | 1.0872 | -4.3102 | 1.5513 | no |
| KMNIST | F4-FNG-leftFullRight | 3 | 0.2170 | 0.0000 | 0.3762 | 0.4474 | 1.0150 | 0.9908 | no |
| KMNIST | FCAdam-H1-low | 3 | 0.1798 | 0.0000 | 0.3892 | 0.8070 | 1.2717 | 1.0000 | no |
| KMNIST | FCAdam-L2 | 3 | 0.1913 | 0.0000 | 0.4165 | 0.9627 | 1.3089 | 1.0108 | no |
| KMNIST | FCAdam-dataSob | 3 | 0.2997 | 0.0000 | 0.4761 | 0.9011 | 1.2821 | 1.0820 | no |
| KMNIST | FPA-edge-prox | 3 | 0.1482 | 0.0000 | 0.7407 | 0.5411 | 0.5556 | 0.9881 | no |
| KMNIST | GFK-all-lowrank | 3 | 0.0033 | 0.0000 | 0.0008 | 0.0000 | 0.0135 | 0.9886 | no |
| KMNIST | GFK-block-output | 3 | 0.0033 | 0.0000 | 0.0007 | 0.0000 | 0.0135 | 0.9886 | no |
| KMNIST | GFK-output-only | 3 | 0.0032 | 0.0000 | 0.0007 | -0.0000 | 0.0134 | 0.9886 | no |
| KMNIST | PureKAN-AdamW-one-step | 3 | 0.1268 | 0.0667 | 0.4834 | 1.0000 | 1.0000 | 1.0000 | yes |
| KMNIST | SFD-direct | 3 | 0.1190 | 0.0000 | 0.2227 | 0.3574 | 0.9912 | 0.9889 | no |
| KMNIST | SFD-prox | 3 | 0.1083 | 0.0000 | 0.1238 | 0.1682 | 0.7823 | 0.9885 | no |
| KMNIST | SFD-residual | 3 | 0.0690 | 0.0000 | 0.1028 | 0.0859 | 0.5845 | 0.9869 | no |
| MNIST | AB-RBF-FCAdam-H1-low | 3 | 0.1856 | 0.0000 | 0.4222 | 0.9527 | 1.1001 | 1.0320 | no |
| MNIST | D0-allFullSobolev | 3 | 0.1519 | 0.0000 | 0.5261 | 0.5124 | 1.0375 | 0.9945 | no |
| MNIST | D6-allTaskAware | 3 | 0.0667 | 0.0000 | 0.6171 | 0.6112 | 0.3641 | 0.9979 | no |
| MNIST | EKFNG-leftRightFull-lowSob | 3 | -1.1332 | 0.2000 | -23.4381 | 1.1353 | -3.9763 | 1.7347 | no |
| MNIST | F4-FNG-leftFullRight | 3 | 0.1965 | 0.0000 | 0.5463 | 0.5883 | 1.0297 | 0.9878 | no |
| MNIST | FCAdam-H1-low | 3 | 0.1559 | 0.0000 | 0.4072 | 0.9988 | 1.4100 | 0.9992 | no |
| MNIST | FCAdam-L2 | 3 | 0.1697 | 0.0667 | 0.4348 | 1.1138 | 1.3523 | 1.0150 | no |
| MNIST | FCAdam-dataSob | 3 | 0.2588 | 0.0000 | 0.4826 | 1.0451 | 1.2523 | 1.1053 | no |
| MNIST | FPA-edge-prox | 3 | 0.1279 | 0.0000 | 0.7271 | 0.6865 | 0.3158 | 0.9926 | no |
| MNIST | GFK-all-lowrank | 3 | 0.0058 | 0.0000 | 0.0007 | -0.0000 | 0.0218 | 0.9927 | no |
| MNIST | GFK-block-output | 3 | 0.0058 | 0.0000 | 0.0006 | -0.0000 | 0.0216 | 0.9927 | no |
| MNIST | GFK-output-only | 3 | 0.0057 | 0.0000 | 0.0006 | -0.0000 | 0.0213 | 0.9927 | no |
| MNIST | PureKAN-AdamW-one-step | 3 | 0.1112 | 0.0667 | 0.4807 | 1.0000 | 1.0000 | 1.0000 | yes |
| MNIST | SFD-direct | 3 | 0.1142 | 0.0000 | 0.2672 | 0.5474 | 1.0338 | 0.9985 | no |
| MNIST | SFD-prox | 3 | 0.1283 | 0.0000 | 0.2100 | 0.3163 | 1.0444 | 0.9952 | no |
| MNIST | SFD-residual | 3 | 0.0795 | 0.0000 | 0.1663 | 0.2275 | 0.6430 | 0.9956 | no |

Best non-Adam points by holdout 5-step descent:

| dataset | best by hold5 | hold5 | R2 | rank/A | phi/A |
|---|---|---|---|---|---|
| MNIST | FCAdam-dataSob | 0.2588 | 0.4826 | 1.0451 | 1.1053 |
| Fashion-MNIST | FCAdam-dataSob | 0.6420 | 0.2358 | 0.6949 | 1.0642 |
| KMNIST | FCAdam-dataSob | 0.2997 | 0.4761 | 0.9011 | 1.0820 |

P2 survivors:

```text
none
```

P2 verdict:

```text
No candidate enters P3.

Many methods achieve positive 5-step holdout descent, especially FCAdam-dataSob,
FPA-edge-prox, F4-FNG and AB-RBF-FCAdam.
However the full written gate is stricter: candidates must keep Adam-like function
alignment or Adam-level train descent, retain rank/margin formation, and reduce
phi_prime_p95 below 90% of PureKAN-AdamW.

The persistent blocker is the geometry gate plus representation retention:
the candidates can learn locally, but they do not yet combine Adam-like function
trajectory, feature formation, and strict geometry improvement.
```

## P3-P9 Decision

```text
P3 short-horizon training: not run.
P4 FCAdam deep validation: not run.
P5 SFD deep validation: not run.
P6 GFK deep validation: not run.
P7/P8/P9 confirm: not run.

Reason: P2 produced no survivor under the written gate.
```

## Failure Diagnosis

Generated:

```text
failure_table.csv
failure_type_by_dataset.csv
failure_type_by_role.csv
proposal_vs_adam_alignment.csv
feature_rank_failure.csv
margin_failure.csv
geometry_regularization_failure.csv
```

Diagnosis:

```text
F3/F8 low Adam alignment or insufficient train dynamics:
  GFK is too small; SFD has low R2; several FCAdam variants fall just below the R2 gate.

F5/F6 representation failure:
  Some methods with positive holdout descent do not retain enough rank/margin movement vs AdamW.

F7 geometry gate failure:
  This is the broadest blocker. Functional candidates usually keep phi close to AdamW,
  but the v4.4 P2 gate requires phi <= 90% of AdamW while preserving learning.

AB-RBF diagnostic:
  AB-RBF-FCAdam improves local function alignment and holdout descent in places,
  but does not clear the joint rank/margin/geometry gate.
```

## Required Artifacts

Written under `results/v4_4/`:

```text
p0_invariants.csv
p1_adam_functional_trajectory.csv
p1_gate_summary.csv
p2_horizon_audit.csv
p2_gate_summary.csv
p3_short_horizon_train_scorecard.csv
p4_fcadam_scorecard.csv
p5_sfd_scorecard.csv
p6_gfk_scorecard.csv
p7_candidate_selection.csv
p8_confirm5.csv
p9_confirm10.csv
optimizer_dynamics_trace.csv
failure_table.csv
failure_type_by_dataset.csv
failure_type_by_role.csv
proposal_vs_adam_alignment.csv
feature_rank_failure.csv
margin_failure.csv
geometry_regularization_failure.csv
aggregate_decision.json
figures/p1_function_r2_vs_adam.svg
figures/p2_holdout_5step.svg
figures/p2_phi_ratio_vs_adam.svg
```

## Final Decision

```text
PureKAN functional optimization remains unsolved in v4.4.

What improved:
  FCAdam-v2 and AB-RBF-FCAdam show that functional-coordinate Adam can track
  AdamW function displacement much better than old Sobolev-only updates.
  SFD confirms that AdamW function displacement can be fit safely in one-step diagnostics.
  GFK gives a stable but too-small global output-space diagnostic.

What failed:
  None of FCAdam, SFD, GFK, FPA, EK-FNG, or AB-RBF-FCAdam passes the P2 joint gate.
  Positive 5-step loss descent did not become a validated representation-learning optimizer.

Conclusion:
  v4.4 partially supports the functional-coordinate Adam hypothesis, but not enough
  to justify P3/P4 expansion. The next useful redesign should keep more Adam-like
  functional trajectory while explicitly meeting the phi/rank/margin gate, rather
  than running more seeds on the current candidates.
```
