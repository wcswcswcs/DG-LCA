# DG-KAN v4.3 Functional Update Deep Redesign 结果复盘

本轮依据 `docs/DG-KAN_v4.3_FunctionalUpdate_DeepRedesign_实验计划.md`。目标是验证 PureKAN functional update 是否应从 Sobolev-only/BFT 改写为：

```text
task-learning dynamics + function-space constraint + temporal optimizer state
```

因此本轮实现并测试了：

```text
FPA: Functional-Proximal Adam
EK-FNG: Edge-feature KFAC Functional Natural Gradient
CFT: Coordinated Functional Targeting
```

## Run Inventory

| stage | rows | errors |
|---|---:|---:|
| P0 smoke | 36 | 0 |
| P1 proposal audit | 39 | 0 |
| P2 short-horizon | 108 | 0 |

## Code / Config Changes

```text
experiments/run_gafu_v43.py
  Added strict-PureKAN FPA, EK-FNG and CFT experimental update paths.
  FPA keeps Adam-like task proposal and projects it through Sobolev/edge-feature proximal metrics.
  EK-FNG uses edge-feature covariance plus optional output-side left covariance.
  CFT uses small-tau coordinated FTF updates with output-only or block/output cyclic roles.
  P1/P2 record Adam alignment, feature rank, margin, geometry and optimizer traces.

experiments/analyze_gafu_v43.py
  Generates gate summaries, P9 failure reports, figures, aggregate_decision.json,
  and this replay.
```

## P0 Implementation Smoke

```text
P0 rows = 36
errors = 0
strict PureKAN nonKAN params = 0 for all rows
functional coverage = 1.0 for all rows
```

P0 verdict: pass. FPA / EK-FNG / CFT did not introduce hidden non-KAN trainable parameters and all coefficient groups are covered.

## P1 One-Step Proposal Quality

| method | train all | holdout+ | cos Adam | proj Adam | rank ratio | phi ratio | P1 |
|---|---|---|---|---|---|---|---|
| AdamW-one-step | 0 | 3 | 1.0000 | 1.0000 | 0.6126 | 1.0044 | no |
| BFT-mixed-reverse | 1 | 3 | 0.0764 | 0.0164 | 0.9985 | 1.0002 | no |
| CFT-block-output-sequential | 1 | 3 | 0.0319 | 0.0732 | 0.9456 | 1.0001 | no |
| CFT-output-only | 1 | 3 | 0.0667 | 0.2014 | 0.8912 | 1.0003 | no |
| D0-allFullSobolev | 1 | 3 | 0.1374 | 0.0546 | 0.9196 | 0.9977 | no |
| D6-allTaskAware | 0 | 2 | 0.6894 | 0.1571 | 0.7928 | 0.9998 | no |
| EKFNG-leftRightFull | 1 | 3 | 0.5353 | 1.1019 | 0.5356 | 0.9972 | no |
| EKFNG-leftRightFull-lowSob | 1 | 3 | 0.5089 | 1.8333 | 0.5318 | 0.9981 | no |
| EKFNG-rightFull | 1 | 3 | 0.6586 | 1.0598 | 0.5254 | 1.0007 | no |
| F4-FNG-leftFullRight | 1 | 3 | 0.1402 | 0.3225 | 0.7795 | 1.0021 | no |
| FPA-edge-prox | 0 | 3 | 0.9297 | 0.5930 | 0.7962 | 1.0001 | no |
| FPA-sob-prox | 1 | 3 | 0.5302 | 0.0972 | 0.9808 | 1.0001 | no |
| raw-gradient | 1 | 3 | 0.5799 | 0.0012 | 0.9995 | 0.9998 | yes |

P1 survivors:

```text
raw-gradient
```

Observation:

```text
FPA-sob and EK-FNG variants keep much more Adam alignment than old FNG/CFT.
However many proposals reduce effective rank on the first step.
This already hints that local descent is not enough; the proposal must preserve feature formation.
```

## P2 Multi-Step Short-Horizon Learning

| dataset | method | runs | acc | gap | AUC vs D6 | ECE red | phi red | J red | rank | P2 |
|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | BFT-mixed-reverse | 3 | 0.7183 | 0.0760 | 0.1703 | -0.1350 | 0.1210 | 0.6563 | 21.0647 | no |
| Fashion-MNIST | CFT-block-output-sequential | 3 | 0.5720 | 0.2223 | -0.4357 | -4.5639 | 0.0752 | 0.5428 | 16.6249 | no |
| Fashion-MNIST | CFT-output-only | 3 | 0.1010 | 0.6933 | -3.0695 | -15.8518 | -0.4940 | -572.8201 | 8.2294 | no |
| Fashion-MNIST | D0-allFullSobolev | 3 | 0.7300 | 0.0643 | -0.1220 | -4.7549 | 0.1206 | 0.6304 | 21.0329 | no |
| Fashion-MNIST | D6-allTaskAware | 3 | 0.7233 | 0.0710 | 0.0000 | -2.9803 | 0.1185 | 0.6761 | 19.1138 | no |
| Fashion-MNIST | EKFNG-leftRightFull | 3 | 0.6967 | 0.0977 | 0.1326 | -1.0504 | 0.1131 | 0.6471 | 20.7584 | no |
| Fashion-MNIST | EKFNG-leftRightFull-lowSob | 3 | 0.7273 | 0.0670 | 0.2042 | -0.2838 | 0.0946 | 0.5925 | 19.6971 | no |
| Fashion-MNIST | EKFNG-rightFull | 3 | 0.6977 | 0.0967 | 0.1880 | -0.8088 | 0.1088 | 0.6482 | 15.8537 | no |
| Fashion-MNIST | F4-FNG-leftFullRight | 3 | 0.6990 | 0.0953 | 0.0480 | -2.4927 | 0.1215 | 0.6688 | 25.3197 | no |
| Fashion-MNIST | FPA-edge-prox | 3 | 0.6927 | 0.1017 | 0.0831 | -0.2852 | 0.0966 | 0.6385 | 16.1422 | no |
| Fashion-MNIST | FPA-sob-prox | 3 | 0.7440 | 0.0503 | -0.2962 | -7.6149 | 0.1193 | 0.6577 | 26.8818 | no |
| Fashion-MNIST | PureKAN-AdamW | 3 | 0.7943 | 0.0000 | 0.3148 | 0.0000 | 0.0000 | 0.0000 | 13.1295 | yes |
| KMNIST | BFT-mixed-reverse | 3 | 0.4337 | 0.2403 | 0.0188 | -0.4330 | 0.1336 | 0.8092 | 29.4992 | no |
| KMNIST | CFT-block-output-sequential | 3 | 0.2230 | 0.4510 | -0.2035 | -0.1342 | 0.0912 | 0.7582 | 26.0259 | no |
| KMNIST | CFT-output-only | 3 | 0.1247 | 0.5493 | -3.7122 | -12.6564 | -0.8477 | -1255.1258 | 11.7944 | no |
| KMNIST | D0-allFullSobolev | 3 | 0.4693 | 0.2047 | -0.0108 | -2.6123 | 0.1359 | 0.8255 | 27.8941 | no |
| KMNIST | D6-allTaskAware | 3 | 0.4573 | 0.2167 | 0.0000 | -1.6150 | 0.1327 | 0.8137 | 24.9843 | no |
| KMNIST | EKFNG-leftRightFull | 3 | 0.4930 | 0.1810 | 0.1410 | -0.8807 | 0.1235 | 0.7970 | 29.2781 | no |
| KMNIST | EKFNG-leftRightFull-lowSob | 3 | 0.4583 | 0.2157 | 0.1875 | -1.5589 | 0.0881 | 0.7758 | 27.5354 | no |
| KMNIST | EKFNG-rightFull | 3 | 0.4353 | 0.2387 | 0.1176 | -2.4126 | 0.0974 | 0.8209 | 13.4684 | no |
| KMNIST | F4-FNG-leftFullRight | 3 | 0.5650 | 0.1090 | 0.1761 | -1.8652 | 0.1334 | 0.8059 | 30.1441 | no |
| KMNIST | FPA-edge-prox | 3 | 0.4367 | 0.2373 | 0.0948 | -0.1476 | 0.0888 | 0.7743 | 14.0573 | no |
| KMNIST | FPA-sob-prox | 3 | 0.4830 | 0.1910 | -0.0886 | -4.0194 | 0.1360 | 0.8313 | 32.5603 | no |
| KMNIST | PureKAN-AdamW | 3 | 0.6740 | 0.0000 | 0.4014 | 0.0000 | 0.0000 | 0.0000 | 19.3626 | yes |
| MNIST | BFT-mixed-reverse | 3 | 0.6863 | 0.1850 | 0.2614 | -0.4123 | 0.1703 | 0.9966 | 27.4750 | no |
| MNIST | CFT-block-output-sequential | 3 | 0.2907 | 0.5807 | -0.0990 | 0.0869 | 0.1038 | 0.9946 | 23.3564 | no |
| MNIST | CFT-output-only | 3 | 0.1210 | 0.7503 | -15.8004 | -8.6038 | -2.3398 | -87.1991 | 8.3833 | no |
| MNIST | D0-allFullSobolev | 3 | 0.6570 | 0.2143 | 0.1000 | -2.6056 | 0.1714 | 0.9967 | 23.4018 | no |
| MNIST | D6-allTaskAware | 3 | 0.5360 | 0.3353 | 0.0000 | -1.6489 | 0.1674 | 0.9966 | 21.0226 | no |
| MNIST | EKFNG-leftRightFull | 3 | 0.6530 | 0.2183 | 0.1662 | -0.5767 | 0.1574 | 0.9964 | 26.7310 | no |
| MNIST | EKFNG-leftRightFull-lowSob | 3 | 0.6690 | 0.2023 | 0.1856 | -0.2774 | 0.0986 | 0.9948 | 27.3767 | no |
| MNIST | EKFNG-rightFull | 3 | 0.5273 | 0.3440 | -0.0900 | -0.2444 | 0.1044 | 0.9961 | 7.0158 | no |
| MNIST | F4-FNG-leftFullRight | 3 | 0.7007 | 0.1707 | 0.2728 | -1.4276 | 0.1701 | 0.9970 | 30.4589 | no |
| MNIST | FPA-edge-prox | 3 | 0.5140 | 0.3573 | 0.0775 | -0.7611 | 0.1006 | 0.9960 | 9.9929 | no |
| MNIST | FPA-sob-prox | 3 | 0.6563 | 0.2150 | 0.0125 | -3.5883 | 0.1719 | 0.9965 | 30.4065 | no |
| MNIST | PureKAN-AdamW | 3 | 0.8713 | 0.0000 | 0.3809 | 0.0000 | 0.0000 | 0.0000 | 13.7687 | yes |

Best non-Adam points:

| dataset | best non-Adam | acc | gap | AUC vs D6 |
|---|---|---|---|---|
| MNIST | F4-FNG-leftFullRight | 0.7007 | 0.1707 | 0.2728 |
| Fashion-MNIST | FPA-sob-prox | 0.7440 | 0.0503 | -0.2962 |
| KMNIST | F4-FNG-leftFullRight | 0.5650 | 0.1090 | 0.1761 |

P2 survivors:

```text
none
```

P2 verdict:

```text
No candidate enters P3.

FPA did preserve some Adam-like direction locally, but short-horizon accuracy was far below PureKAN-AdamW.
EK-FNG improved some old functional baselines but did not close the gap.
CFT remained unstable/underfit; output-only CFT collapsed on all datasets.
The hardest blocker is KMNIST, where even the best non-Adam candidate remains far below PureKAN-AdamW.
```

## P3-P8 Decision

```text
P3 FPA deep validation: not run.
P4 EK-FNG deep validation: not run.
P5 CFT validation: not run.
P6 candidate selection: not run.
P7/P8 confirm: not run.

Reason: P2 produced no survivor under the written gate.
```

## P9 Failure Diagnosis

Generated:

```text
failure_type_by_dataset.csv
failure_type_by_role.csv
proposal_vs_adam_alignment.csv
feature_rank_failure.csv
margin_failure.csv
geometry_regularization_failure.csv
```

Diagnosis:

```text
F1 Adam component not retained:
  Partially true for old FNG/CFT; less true for FPA/EK-FNG, but retaining alignment alone was not enough.

F2 feature rank collapse:
  Present in P1 for several high-descent methods.

F3 margin / accuracy not improving:
  Dominant P2 failure. All functional candidates have large accuracy gaps vs PureKAN-AdamW.

F6 geometry over-regularization:
  FPA-sob and CFT show the classic geometry-safe but learning-weak profile.

F8 temporal dynamics missing:
  Still likely. FPA as implemented is too damped; EK-FNG lacks enough long-term adaptive dynamics.
```

## Required Artifacts

Written under `results/v4_3/`:

```text
p0_invariants.csv
p1_proposal_direction_audit.csv
p1_gate_summary.csv
p2_short_horizon_scorecard.csv
p2_gate_summary.csv
p3_fpa_scorecard.csv
p4_ekfng_scorecard.csv
p5_cft_scorecard.csv
p6_candidate_selection.csv
p7_confirm5.csv
p8_confirm10.csv
optimizer_dynamics_trace.csv
role_update_trace.csv
failure_table.csv
failure_type_by_dataset.csv
failure_type_by_role.csv
proposal_vs_adam_alignment.csv
feature_rank_failure.csv
margin_failure.csv
geometry_regularization_failure.csv
aggregate_decision.json
figures/
```

## Final Decision

```text
PureKAN functional optimization remains unsolved in v4.3.

What improved:
  FPA and EK-FNG make the direction more Adam-aligned than Sobolev-only / CFT.
  EK-FNG validates that edge-feature/task geometry is a meaningful direction to test.

What failed:
  The improved one-step direction did not produce short-horizon representation learning.
  FPA was too damped and underfit.
  EK-FNG remained below old F4-FNG or AdamW on the hard datasets.
  CFT still has the FTF failure mode in softer form.

Conclusion:
  The v4.3 hypothesis is partially supported but not solved:
  functional geometry should constrain Adam-like task dynamics,
  but the current FPA/EK-FNG implementations do not retain enough useful feature-forming motion.

Next recommended step:
  Do not expand seeds.
  Redesign FPA to retain a larger Adam component and add adaptive damping by role,
  especially for KMNIST input/block feature formation.
```
