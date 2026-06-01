# DG-KAN v4.1 Functional Update Redesign 结果复盘

本轮依据 `docs/DG-KAN_v4.1_FunctionalUpdate_Redesign_DeepPlan.md`。目标是验证重新设计的 PureKAN functional update：FTF、FC-Adam 与 FTR/FGN trust-region 是否能把 v3.7-v3.9 的局部方向修复转成长程训练收益。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 smoke | 30 | 0 |
| P1 shadow | 30 | 0 |
| P2 micro | 72 | 5 |

## Code / Config Changes

```text
experiments/dgkan_core.py
  Added FTF fields and per-role target-fitting updates for PureKAN coeffs.
  Added FC-Adam functional-coordinate updates through Sobolev Cholesky coordinates.
  Added an FTR-CG-small diagnostic path with layer-output trust scaling.
  Result rows now include FTF fit/residual/trust stats, FC-Adam reconstruction diagnostics,
  and FTR residual / predicted change statistics.

experiments/run_gafu_v41.py
  Added P0/P1/P2 packages and one-batch direction audit for v4.1 candidates.

experiments/analyze_gafu_v41.py
  Writes the required v4.1 CSV/JSON artifacts and this replay.
```

## P0 Implementation Smoke

P0 rows: `30`; errors: `0`.

Key audit:

```text
PureKAN alphaFixed1 / fixed norm paths ran without implementation errors.
Strict PureKAN rows kept learnable_nonKAN_params = 0.
Functional rows covered input/block/output coeff groups.
FTF, FC-Adam, and FTR all produced finite one-epoch smoke rows.
```

## P1 One-Batch Direction Gate

| method | bad | min train | min val | min FTF R2 | max delta | P1 pass |
|---|---|---|---|---|---|---|
| D0-allFullSobolev | 0 | 0.0016 | -0.0039 |  |  | yes |
| D6-allTaskAware | 1 | -0.0080 | -0.0091 |  |  | no |
| F4-FNG-leftFull-right | 0 | 0.0488 | 0.0046 |  |  | yes |
| FC-Adam-one-step | 0 | 0.0212 | -0.0377 |  |  | no |
| FTF-all-sequential | 0 | 0.0784 | 0.0636 | 1.0000 | 0.1000 | yes |
| FTF-all-simultaneous | 0 | 0.1247 | 0.0231 | 0.9999 | 0.1000 | yes |
| FTF-blocks-output | 0 | 0.0664 | 0.0086 | 0.9997 | 0.0997 | yes |
| FTF-output-only | 0 | 0.0093 | -0.0666 | 1.0000 | 0.0998 | no |
| FTR-CG-small | 0 | 0.0388 | 0.0023 |  |  | yes |

P1 survivors used for P2:

```text
D0-allFullSobolev, F4-FNG-leftFull-right, FTF-all-sequential, FTF-all-simultaneous, FTF-blocks-output, FTR-CG-small
```

Observation:

```text
FTF repaired the one-step direction signal strongly: blocks/output and all-layer modes had positive train and validation descent on all three datasets, with fit R2 near 1.0.
D6 still had a KMNIST bad train step and was not treated as a P2 candidate, but was later added as a P2 baseline because the plan requires AUC comparison vs D6.
FC-Adam improved train loss but failed the validation-descent gate on Fashion/KMNIST.
```

## P2 Micro-Run Scorecard

| dataset | method | runs | errors | acc | std | gap vs AdamW | AUC imp vs AdamW | AUC imp vs D6 | ECE red | phi ratio | FTF R2 | P2 pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MNIST | D0-allFullSobolev | 3 | 0 | 0.8417 | 0.0304 | 0.0723 | -0.8900 | 0.0003 | -8.4610 | 0.7658 |  | 0 |
| MNIST | D6-allTaskAware | 3 | 0 | 0.8523 | 0.0068 | 0.0617 | -0.8906 | 0.0000 | -5.7526 | 0.7685 |  | 0 |
| MNIST | F4-FNG-leftFull-right | 3 | 0 | 0.9233 | 0.0110 | -0.0093 | -0.0388 | 0.4505 | -0.4553 | 0.7661 |  | 0 |
| MNIST | FTF-all-sequential | 3 | 0 | 0.1003 | 0.0005 | 0.8137 | -518293472445.2520 | -274141826146.2508 | -42.8198 | 509227.2591 | 0.9453 | 0 |
| MNIST | FTF-all-simultaneous | 3 | 0 | 0.1003 | 0.0005 | 0.8137 | -518293472445.2520 | -274141826146.2508 | -42.8198 | 509227.2591 | 0.9453 | 0 |
| MNIST | FTF-blocks-output | 3 | 0 | 0.1023 | 0.0033 | 0.8117 | -1754569223347315.0000 | -928047209855757.8750 | -42.7223 | 22285636.6922 | 0.7068 | 0 |
| MNIST | FTR-CG-small | 3 | 0 | 0.6687 | 0.0107 | 0.2453 | -1.9351 | -0.5525 | -18.2742 | 0.7644 |  | 0 |
| MNIST | PureKAN-AdamW | 3 | 0 | 0.9140 | 0.0043 | 0.0000 | 0.0000 | 0.4711 | 0.0000 | 1.0000 |  | 1 |
| Fashion-MNIST | D0-allFullSobolev | 3 | 0 | 0.8170 | 0.0174 | 0.0190 | -0.5294 | -0.1251 | -0.1849 | 0.7614 |  | 0 |
| Fashion-MNIST | D6-allTaskAware | 3 | 0 | 0.8060 | 0.0134 | 0.0300 | -0.3594 | 0.0000 | 0.4009 | 0.7636 |  | 0 |
| Fashion-MNIST | F4-FNG-leftFull-right | 3 | 0 | 0.8227 | 0.0164 | 0.0133 | -0.1005 | 0.1904 | 0.3178 | 0.7607 |  | 1 |
| Fashion-MNIST | FTF-all-sequential | 2 | 1 | 0.1000 | 0.0000 | 0.7360 | -105351734110180880.0000 | -77501361878272944.0000 | -10.9915 | 203698804.2161 | 0.9480 | 0 |
| Fashion-MNIST | FTF-all-simultaneous | 2 | 1 | 0.1000 | 0.0000 | 0.7360 | -105351734110180880.0000 | -77501361878272944.0000 | -10.9915 | 203698804.2161 | 0.9480 | 0 |
| Fashion-MNIST | FTR-CG-small | 3 | 0 | 0.7167 | 0.0194 | 0.1193 | -1.7757 | -1.0420 | -3.1114 | 0.7598 |  | 0 |
| Fashion-MNIST | PureKAN-AdamW | 3 | 0 | 0.8360 | 0.0156 | 0.0000 | 0.0000 | 0.2644 | 0.0000 | 1.0000 |  | 1 |
| KMNIST | D0-allFullSobolev | 3 | 0 | 0.5750 | 0.0185 | 0.1930 | -1.0548 | -0.1156 | -0.5424 | 0.7607 |  | 0 |
| KMNIST | D6-allTaskAware | 3 | 0 | 0.5610 | 0.0550 | 0.2070 | -0.8419 | 0.0000 | 0.2455 | 0.7640 |  | 0 |
| KMNIST | F4-FNG-leftFull-right | 3 | 0 | 0.7123 | 0.0132 | 0.0557 | -0.1847 | 0.3568 | 0.3224 | 0.7648 |  | 0 |
| KMNIST | FTF-all-sequential | 3 | 0 | 0.1070 | 0.0099 | 0.6610 | -788570634358.2345 | -428138390693.1385 | -9.8694 | 626491.2995 | 0.9186 | 0 |
| KMNIST | FTF-all-simultaneous | 3 | 0 | 0.1070 | 0.0099 | 0.6610 | -788570634358.2345 | -428138390693.1385 | -9.8694 | 626491.2995 | 0.9186 | 0 |
| KMNIST | FTF-blocks-output | 3 | 0 | 0.1000 | 0.0000 | 0.6680 | -117251294060245.6875 | -63659205857378.9766 | -9.9546 | 6377580.6882 | 0.6710 | 0 |
| KMNIST | FTR-CG-small | 3 | 0 | 0.3730 | 0.0140 | 0.3950 | -2.1432 | -0.7065 | -1.2296 | 0.7613 |  | 0 |
| KMNIST | PureKAN-AdamW | 3 | 0 | 0.7680 | 0.0067 | 0.0000 | 0.0000 | 0.4571 | 0.0000 | 1.0000 |  | 1 |

## P2 Failure Diagnosis

```text
No candidate passed the P2 joint gate.

FTF:
  P1 target fit was excellent, but P2 training was catastrophic.
  MNIST/KMNIST dropped to near chance accuracy, Fashion produced numerical eigensolve failures for several FTF rows, and val-loss AUC exploded.
  This matches the plan's failure mode: fit_R2 high + one-step descent positive + short-run acc bad => layer-local target fitting causes cross-layer drift.

FTR-CG-small:
  Local direction was acceptable, but short training underfit badly on all datasets.

F4-FNG-leftFull-right:
  Best practical candidate in P2.
  It matched/beat PureKAN-AdamW on MNIST and stayed within about 1.3 points on Fashion.
  It failed KMNIST by about 5.6 points, so it cannot enter P3.

D0/D6:
  Geometry is stable but accuracy remains below AdamW, especially on KMNIST.
```

## P3-P5 Decision

```text
P3 refinement: not run.
Reason: P2 produced no survivor.

P4 5-seed confirm: not run.
Reason: P3 was not reached.

P5 10-seed final: not run.
Reason: P4 was not reached.
```

## Artifacts

Required files were written under `results/v4_1/`:

```text
p0_invariants.csv
p1_direction_audit.csv
p2_micro_run_scorecard.csv
p2_failure_diagnosis.csv
p3_refinement_scorecard.csv
p4_confirm5_scorecard.csv
p5_confirm10_scorecard.csv
functional_target_fit.csv
representation_audit.csv
geometry_audit.csv
compute_audit.csv
failure_table.csv
aggregate_decision.json
figures/p2_kmnist_acc_gap.svg
figures/p2_kmnist_auc_vs_d6.svg
```

## Final Decision

```text
PureKAN functional optimization is still not solved in v4.1.

What improved:
  FTF gives a real one-step target-fitting direction.
  FNG remains the strongest short-run practical baseline among functional candidates.

What failed:
  FTF target fitting is not yet a stable optimizer; high local fit creates long-horizon drift.
  FTR-CG-small is too weak in this approximation.
  FNG does not close the KMNIST accuracy gap.

Next recommended direction:
  Add accepted-step / backtracking and cross-layer drift control to FTF before more seeds.
  In particular, target fitting should be sequential with validation of actual loss decrease
  and an activation/logit trust region that rejects or shrinks unsafe layer updates.
```
