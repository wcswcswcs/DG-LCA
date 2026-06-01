# DG-KAN Optimizer v3.2 结果复盘

## Final Scorecard

| dataset | method | runs | acc | acc std | gap | AUC imp | phi red | J red | branch/A | ECE red | noKAN | margin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | AdamW-final-alphaFixed1 | 10.0000 | 0.8347 | 0.0096 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.0000 | 0.1909 | 4.2942 |
| Fashion-MNIST | F-V3-HARD-base30-final-alphaFixed1 | 10.0000 | 0.8469 | 0.0111 | -0.0122 | 0.1034 | 0.3562 | 0.9999 | 0.6562 | 0.2215 | 0.1588 | 3.8263 |
| Fashion-MNIST | GA-FU-v2-final-alphaFixed1 | 10.0000 | 0.8448 | 0.0140 | -0.0101 | 0.0418 | 0.3491 | 0.9999 | 0.6349 | 0.1334 | 0.1437 | 4.1009 |
| KMNIST | AdamW-final-alphaFixed1 | 10.0000 | 0.7776 | 0.0155 | -0.0000 | -0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.2654 | 4.1473 |
| KMNIST | GA-FU-v2-final-alphaFixed1 | 10.0000 | 0.7786 | 0.0136 | -0.0010 | 0.0565 | 0.2553 | 0.9999 | 0.6225 | 0.1244 | 0.1825 | 3.6939 |
| KMNIST | K-V3-FULL-base-final-alphaFixed1 | 10.0000 | 0.7774 | 0.0150 | 0.0002 | 0.0812 | 0.2803 | 1.0000 | 0.5801 | 0.1708 | 0.1759 | 3.5037 |

## Paired Delta vs AdamW

| dataset | method | acc delta | acc CI lo | acc CI hi | AUC delta | AUC CI lo | AUC CI hi |
|---|---|---|---|---|---|---|---|
| Fashion-MNIST | GA-FU-v2-final-alphaFixed1 | 0.0101 | 0.0027 | 0.0181 | 0.0231 | 0.0107 | 0.0335 |
| Fashion-MNIST | F-V3-HARD-base30-final-alphaFixed1 | 0.0122 | 0.0058 | 0.0187 | 0.0573 | 0.0492 | 0.0657 |
| KMNIST | GA-FU-v2-final-alphaFixed1 | 0.0010 | -0.0070 | 0.0081 | 0.0301 | 0.0169 | 0.0434 |
| KMNIST | K-V3-FULL-base-final-alphaFixed1 | -0.0002 | -0.0078 | 0.0073 | 0.0433 | 0.0343 | 0.0536 |

## Target-Matched / Wall-Clock

| dataset | method | runs | relaxed loss reached | epochs relaxed loss | time relaxed loss | step ms |
|---|---|---|---|---|---|---|
| Fashion-MNIST | AdamW-final-alphaFixed1 | 10.0000 | 1.0000 | 1.3000 | 2.7127 | 86.9609 |
| Fashion-MNIST | F-V3-HARD-base30-final-alphaFixed1 | 10.0000 | 1.0000 | 2.0000 | 4.6458 | 96.7807 |
| Fashion-MNIST | GA-FU-v2-final-alphaFixed1 | 10.0000 | 1.0000 | 2.5000 | 5.4558 | 90.7315 |
| KMNIST | AdamW-final-alphaFixed1 | 10.0000 | 1.0000 | 2.1000 | 4.3883 | 87.0856 |
| KMNIST | GA-FU-v2-final-alphaFixed1 | 10.0000 | 1.0000 | 2.8000 | 6.1409 | 91.4172 |
| KMNIST | K-V3-FULL-base-final-alphaFixed1 | 10.0000 | 1.0000 | 2.3000 | 5.3378 | 96.6950 |

## Decision

```text
Fashion:
  F-V3-HARD-base30-final-alphaFixed1 is confirmed over 10 seeds.
  It improves accuracy, validation-loss AUC, geometry, calibration, and keeps healthy branch utilization.

KMNIST:
  K-V3-FULL-base-final-alphaFixed1 is a geometry/convergence Pareto candidate.
  It approximately matches AdamW accuracy, improves AUC/J/ECE, but does not reach the stricter 9% AUC target and no-KAN ratio is below the 0.7 expression threshold.

Alpha mode:
  fixed1 is selected for P5 because it passed Fashion and reached the 5-seed KMNIST medium line.
  Final interpretation should still be dataset-specific: Fashion clean default, KMNIST Pareto/medium candidate.

Wall-clock:
  True-Gram methods improve matched-step loss trajectories, but step time is about 1.1x AdamW.
  Do not claim unconditional wall-clock faster convergence unless target_summary shows lower time-to-target.
```
