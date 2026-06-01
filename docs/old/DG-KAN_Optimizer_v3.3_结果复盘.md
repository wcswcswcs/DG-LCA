# DG-KAN Optimizer v3.3 结果复盘

## Final Scorecard

| dataset | method | runs | acc | acc std | gap | AUC imp | phi red | J red | branch/A | ECE red | noKAN | time/A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | AdamW-alphaFixed1-alphaFixed1 | 10.0000 | 0.8347 | 0.0096 | 0.0000 | 0.0000 | 0.0000 | -0.0000 | 1.0000 | -0.0000 | 1.0000 | 1.0000 |
| Fashion-MNIST | F-V3-HARD-base30-v32best-alphaFixed1 | 10.0000 | 0.8469 | 0.0111 | -0.0122 | 0.1034 | 0.3562 | 0.9999 | 0.6562 | 0.2215 | 0.8318 | 1.3671 |
| Fashion-MNIST | U-FULL-f085-alphaFixed1 | 10.0000 | 0.8449 | 0.0103 | -0.0102 | 0.0816 | 0.3601 | 0.9998 | 0.6141 | 0.2073 | 0.7926 | 1.3760 |
| KMNIST | AdamW-alphaFixed1-alphaFixed1 | 10.0000 | 0.7776 | 0.0155 | -0.0000 | -0.0000 | 0.0000 | -0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| KMNIST | K-V3-FULL-base-v32best-alphaFixed1 | 10.0000 | 0.7774 | 0.0150 | 0.0002 | 0.0812 | 0.2804 | 0.9999 | 0.5801 | 0.1708 | 0.6628 | 1.3923 |
| KMNIST | U-FULL-f085-alphaFixed1 | 10.0000 | 0.7804 | 0.0138 | -0.0028 | 0.0914 | 0.2806 | 0.9999 | 0.6002 | 0.1744 | 0.6959 | 1.3968 |

## Paired Delta vs AdamW

| dataset | method | acc delta | acc CI lo | acc CI hi | AUC delta | AUC CI lo | AUC CI hi | noKAN |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | F-V3-HARD-base30-v32best-alphaFixed1 | 0.0122 | 0.0058 | 0.0187 | 0.0573 | 0.0492 | 0.0657 | 0.8318 |
| Fashion-MNIST | U-FULL-f085-alphaFixed1 | 0.0102 | 0.0057 | 0.0147 | 0.0452 | 0.0345 | 0.0576 | 0.7926 |
| KMNIST | K-V3-FULL-base-v32best-alphaFixed1 | -0.0002 | -0.0078 | 0.0073 | 0.0433 | 0.0343 | 0.0536 | 0.6628 |
| KMNIST | U-FULL-f085-alphaFixed1 | 0.0028 | -0.0041 | 0.0099 | 0.0487 | 0.0402 | 0.0575 | 0.6959 |

## Unified Optimizer Smoke

| dataset | method | acc | AUC imp | branch/A | phi red | J red | clip | moment amp | moment cos | bad rate |
|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | AdamW-alphaFixed1-alphaFixed1 | 0.8120 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 |
| Fashion-MNIST | AllFunctionalRestSGD-negative-alphaFixed1 | 0.7740 | -0.8157 | 0.6564 | 0.2004 | 0.9966 | 0.0000 | 0.0000 |  | 0.0000 |
| Fashion-MNIST | Hybrid-U-FULL-base-alphaFixed1 | 0.8310 | -0.0151 | 0.6160 | 0.2028 | 0.9965 | 0.0000 | 0.0000 |  | 0.0000 |
| Fashion-MNIST | UO-NormPGAdam-alphaFixed1 | 0.8230 | -0.3177 | 13.8577 | -13.0241 | -64.7230 | 0.0000 | 0.5840 | 0.4208 | 0.0417 |
| Fashion-MNIST | UO-PGAdam-alphaFixed1 | 0.8050 | -0.3072 | 12.6268 | -12.7632 | -22.8812 | 0.0000 | 0.6172 | 0.4273 | 0.0417 |
| Fashion-MNIST | UO-PostAdam-alphaFixed1 | 0.8170 | -0.1695 | 7.4791 | -12.3808 | -68.4518 | 0.0000 | 0.0164 | 0.0997 | 0.0458 |
| Fashion-MNIST | UO-PostAdam-trust030-alphaFixed1 | 0.8270 | -0.0185 | 2.4402 | -1.9422 | -67.4379 | 0.0708 | 0.0610 | 0.0634 | 0.1083 |
| KMNIST | AdamW-alphaFixed1-alphaFixed1 | 0.7590 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 |
| KMNIST | AllFunctionalRestSGD-negative-alphaFixed1 | 0.6290 | -0.9530 | 0.6863 | 0.1997 | 0.9996 | 0.0000 | 0.0000 |  | 0.0000 |
| KMNIST | Hybrid-U-FULL-base-alphaFixed1 | 0.8010 | 0.0098 | 0.6001 | 0.2082 | 0.9996 | 0.0000 | 0.0000 |  | 0.0000 |
| KMNIST | UO-NormPGAdam-alphaFixed1 | 0.7430 | -0.8466 | 36.4838 | -38.6146 | -18.9618 | 0.0000 | 1.2329 | 0.4220 | 0.0375 |
| KMNIST | UO-PGAdam-alphaFixed1 | 0.7470 | -0.8477 | 35.1335 | -39.7938 | -43.1957 | 0.0000 | 1.4040 | 0.4188 | 0.0375 |
| KMNIST | UO-PostAdam-alphaFixed1 | 0.7710 | -0.5285 | 20.3786 | -49.8648 | -43.1957 | 0.0000 | 0.0099 | 0.1522 | 0.0417 |
| KMNIST | UO-PostAdam-trust030-alphaFixed1 | 0.7830 | -0.0246 | 4.4553 | -5.6763 | -6.1620 | 0.0833 | 0.0767 | 0.0759 | 0.1208 |

## Target-Matched / Wall-Clock

| dataset | method | runs | relaxed loss reached | epochs relaxed loss | time relaxed loss | step ms |
|---|---|---|---|---|---|---|
| Fashion-MNIST | AdamW-alphaFixed1-alphaFixed1 | 10.0000 | 1.0000 | 1.3000 | 1.0383 | 33.1100 |
| Fashion-MNIST | F-V3-HARD-base30-v32best-alphaFixed1 | 10.0000 | 1.0000 | 2.0000 | 2.1736 | 45.2634 |
| Fashion-MNIST | U-FULL-f085-alphaFixed1 | 10.0000 | 1.0000 | 2.3000 | 2.5160 | 45.5594 |
| KMNIST | AdamW-alphaFixed1-alphaFixed1 | 10.0000 | 1.0000 | 2.1000 | 1.6371 | 32.5068 |
| KMNIST | K-V3-FULL-base-v32best-alphaFixed1 | 10.0000 | 1.0000 | 2.3000 | 2.5037 | 45.2608 |
| KMNIST | U-FULL-f085-alphaFixed1 | 10.0000 | 1.0000 | 2.7000 | 2.9444 | 45.4044 |

## Decision

```text
P0:
  implementation smoke passed with no run failures and finite full-Sobolev metric statistics.

P1/P2:
  U-FULL-f085-alphaFixed1 was the best cross-dataset full-start candidate.
  In 3-seed P2 it passed Fashion and KMNIST gates, including no-KAN expression.

P3/P4:
  Unified AdamW-style optimizer variants are quarantined.
  PGAdam, NormPGAdam, and PostAdam produced branch over-activation and geometry spikes.
  PostAdam-trust030 was safer but still exceeded planned branch/trust/bad-step limits.
  P4 was therefore intentionally not expanded.

P5:
  U-FULL-f085-alphaFixed1 passed the 5-seed unified full-start gate.

P6:
  Fashion passes strict unified clean conditions.
  KMNIST matches/improves AdamW accuracy, improves AUC/geometry/ECE, and improves expression over v3.2 full-base.
  However KMNIST no-KAN ratio is 0.696, just below the strict 0.700 threshold.

Final:
  full_sobolev_gram from start + alphaFixed1 + branch_final_scale=0.85 is confirmed as the v3.3 unified Pareto profile.
  It is not claimed as strict unified clean default because KMNIST expression misses the no-KAN threshold by 0.004.
  Hybrid optimizer remains the mainline; unified AdamW-style moment dynamics are not accepted.

Wall-clock:
  True-Gram methods improve matched-step loss trajectories, but step time is about 1.37x-1.40x AdamW in this audit=64 run.
  Do not claim unconditional wall-clock faster convergence unless target_summary shows lower time-to-target.
```
