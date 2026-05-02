# DG-KAN Optimizer v3.4 AFU 结果复盘

## P2 Single-Component Scorecard

| dataset | method | runs | acc | acc std | gap | AUC imp | phi red | J red | branch/A | ECE red | noKAN | time/A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | AFU-0-Hybrid-alphaFixed1 | 3.0000 | 0.8487 | 0.0084 | -0.0157 | 0.1195 | 0.3596 | 0.9995 | 0.6265 | 0.2037 | 0.7707 | 1.3008 |
| Fashion-MNIST | AFU-1-BranchLocal-alphaFixed1 | 3.0000 | 0.8477 | 0.0123 | -0.0147 | 0.0877 | 0.3596 | 0.9995 | 0.6144 | 0.2060 | 0.7882 | 1.4429 |
| Fashion-MNIST | AFU-2-HeadCov-alphaFixed1 | 3.0000 | 0.8507 | 0.0090 | -0.0177 | 0.0992 | 0.3598 | 0.9995 | 0.6537 | 0.5520 | 1.0303 | 1.5316 |
| Fashion-MNIST | AFU-HeadOnly-alphaFixed1 | 3.0000 | 0.8497 | 0.0095 | -0.0167 | 0.0872 | 0.3600 | 0.9995 | 0.6602 | 0.4970 | 0.9841 | 1.4740 |
| Fashion-MNIST | AFU-LNOnly-alphaFixed1 | 3.0000 | 0.8483 | 0.0160 | -0.0153 | 0.1145 | 0.3595 | 0.9995 | 0.6246 | 0.2222 | 0.8280 | 1.5639 |
| Fashion-MNIST | AFU-StemOnly-alphaFixed1 | 3.0000 | 0.7827 | 0.0110 | 0.0503 | -0.2826 | 0.3567 | 0.9994 | 0.6323 | 0.1457 | 1.8917 | 1.3977 |
| Fashion-MNIST | AdamW-alphaFixed1-alphaFixed1 | 3.0000 | 0.8330 | 0.0022 | -0.0000 | -0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.0000 | 1.0000 | 1.0000 |
| Fashion-MNIST | F-V3-HARD-base30-v32best-alphaFixed1 | 3.0000 | 0.8473 | 0.0126 | -0.0143 | 0.1077 | 0.3558 | 0.9996 | 0.6649 | 0.1964 | 0.7309 | 1.2787 |
| KMNIST | AFU-0-Hybrid-alphaFixed1 | 3.0000 | 0.7860 | 0.0128 | 0.0017 | 0.1111 | 0.2813 | 0.9997 | 0.5977 | 0.1621 | 0.7650 | 1.4226 |
| KMNIST | AFU-1-BranchLocal-alphaFixed1 | 3.0000 | 0.7793 | 0.0151 | 0.0083 | 0.0914 | 0.2818 | 0.9997 | 0.5880 | 0.1313 | 0.7482 | 1.5902 |
| KMNIST | AFU-2-HeadCov-alphaFixed1 | 3.0000 | 0.7650 | 0.0163 | 0.0227 | 0.1115 | 0.2795 | 0.9997 | 0.6605 | 0.3504 | 0.9173 | 1.7224 |
| KMNIST | AFU-HeadOnly-alphaFixed1 | 3.0000 | 0.7713 | 0.0084 | 0.0163 | 0.1242 | 0.2803 | 0.9997 | 0.6600 | 0.3837 | 0.9125 | 1.6494 |
| KMNIST | AFU-LNOnly-alphaFixed1 | 3.0000 | 0.7820 | 0.0171 | 0.0057 | 0.1107 | 0.2814 | 0.9997 | 0.6109 | 0.2163 | 0.7542 | 1.7617 |
| KMNIST | AFU-StemOnly-alphaFixed1 | 3.0000 | 0.6363 | 0.0156 | 0.1513 | -0.7905 | 0.2644 | 0.9996 | 0.6157 | -0.3976 | 0.5588 | 1.5794 |
| KMNIST | AdamW-alphaFixed1-alphaFixed1 | 3.0000 | 0.7877 | 0.0048 | -0.0000 | 0.0000 | 0.0000 | -0.0000 | 1.0000 | -0.0000 | 1.0000 | 1.0000 |
| KMNIST | K-V3-FULL-base-v32best-alphaFixed1 | 3.0000 | 0.7853 | 0.0170 | 0.0023 | 0.0997 | 0.2815 | 0.9997 | 0.5775 | 0.1776 | 0.7386 | 1.4624 |

## P3 Cumulative AFU Scorecard

| dataset | method | runs | acc | acc std | gap | AUC imp | phi red | J red | branch/A | ECE red | noKAN | time/A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | AFU-0-Hybrid-alphaFixed1 | 3.0000 | 0.8487 | 0.0084 | -0.0157 | 0.1195 | 0.3596 | 0.9995 | 0.6265 | 0.2037 | 0.7707 | 1.4475 |
| Fashion-MNIST | AFU-2-HeadCov-alphaFixed1 | 3.0000 | 0.8507 | 0.0090 | -0.0177 | 0.0992 | 0.3598 | 0.9995 | 0.6537 | 0.5520 | 1.0303 | 1.7321 |
| Fashion-MNIST | AFU-3-HeadCov-LNHead-alphaFixed1 | 3.0000 | 0.8480 | 0.0086 | -0.0150 | 0.0828 | 0.3602 | 0.9995 | 0.6608 | 0.5973 | 1.0573 | 1.7632 |
| Fashion-MNIST | AFU-5-FullDiagLite-alphaFixed1 | 3.0000 | 0.7997 | 0.0090 | 0.0333 | -0.4006 | 0.3531 | 0.9995 | 0.6737 | 0.3128 | 3.3487 | 1.9666 |
| Fashion-MNIST | AFU-6-FullKFACLite-alphaFixed1 | 3.0000 | 0.8003 | 0.0099 | 0.0327 | -0.3981 | 0.3532 | 0.9995 | 0.6735 | 0.3193 | 3.3535 | 1.9684 |
| Fashion-MNIST | AdamW-alphaFixed1-alphaFixed1 | 3.0000 | 0.8330 | 0.0022 | -0.0000 | -0.0000 | 0.0000 | 0.0000 | 1.0000 | -0.0000 | 1.0000 | 1.0000 |
| Fashion-MNIST | F-V3-HARD-base30-v32best-alphaFixed1 | 3.0000 | 0.8473 | 0.0126 | -0.0143 | 0.1077 | 0.3558 | 0.9996 | 0.6649 | 0.1964 | 0.7309 | 1.4189 |
| KMNIST | AFU-0-Hybrid-alphaFixed1 | 3.0000 | 0.7860 | 0.0128 | 0.0017 | 0.1111 | 0.2813 | 0.9997 | 0.5977 | 0.1621 | 0.7650 | 1.3912 |
| KMNIST | AFU-2-HeadCov-alphaFixed1 | 3.0000 | 0.7650 | 0.0163 | 0.0227 | 0.1115 | 0.2795 | 0.9997 | 0.6605 | 0.3504 | 0.9173 | 1.6460 |
| KMNIST | AFU-3-HeadCov-LNHead-alphaFixed1 | 3.0000 | 0.7603 | 0.0107 | 0.0273 | 0.0647 | 0.2792 | 0.9997 | 0.6784 | 0.6546 | 0.9089 | 1.6728 |
| KMNIST | AFU-5-FullDiagLite-alphaFixed1 | 3.0000 | 0.6243 | 0.0094 | 0.1633 | -0.8353 | 0.2524 | 0.9995 | 0.6983 | 0.6788 | 1.6619 | 1.8791 |
| KMNIST | AFU-6-FullKFACLite-alphaFixed1 | 3.0000 | 0.6237 | 0.0107 | 0.1640 | -0.8308 | 0.2523 | 0.9995 | 0.6980 | 0.6838 | 1.6583 | 1.8499 |
| KMNIST | AdamW-alphaFixed1-alphaFixed1 | 3.0000 | 0.7877 | 0.0048 | -0.0000 | 0.0000 | 0.0000 | -0.0000 | 1.0000 | -0.0000 | 1.0000 | 1.0000 |
| KMNIST | K-V3-FULL-base-v32best-alphaFixed1 | 3.0000 | 0.7853 | 0.0170 | 0.0023 | 0.0997 | 0.2815 | 0.9997 | 0.5775 | 0.1776 | 0.7386 | 1.3513 |

## P4 Expression Micro-Check

| dataset | method | runs | acc | acc std | gap | AUC imp | phi red | J red | branch/A | ECE red | noKAN | time/A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KMNIST | AFU-0-Hybrid-f0875-alphaFixed1 | 3.0000 | 0.7780 | 0.0204 | 0.0097 | 0.0797 | 0.2815 | 0.9996 | 0.6034 | 0.1402 | 0.7758 | 1.3917 |
| KMNIST | AFU-0-Hybrid-f090-alphaFixed1 | 3.0000 | 0.7773 | 0.0207 | 0.0103 | 0.0740 | 0.2817 | 0.9996 | 0.6131 | 0.1565 | 0.7518 | 1.4175 |
| KMNIST | AFU-2-HeadCov-f0875-alphaFixed1 | 3.0000 | 0.7673 | 0.0148 | 0.0203 | 0.1088 | 0.2795 | 0.9996 | 0.6679 | 0.3738 | 0.9281 | 1.6832 |
| KMNIST | AFU-2-HeadCov-f090-alphaFixed1 | 3.0000 | 0.7720 | 0.0114 | 0.0157 | 0.1185 | 0.2800 | 0.9996 | 0.6757 | 0.4036 | 0.9508 | 1.6719 |
| KMNIST | AdamW-alphaFixed1-alphaFixed1 | 3.0000 | 0.7877 | 0.0048 | -0.0000 | 0.0000 | 0.0000 | -0.0000 | 1.0000 | -0.0000 | 1.0000 | 1.0000 |

## P5 Partial AFU Confirm

| dataset | method | runs | acc | acc std | gap | AUC imp | phi red | J red | branch/A | ECE red | noKAN | time/A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | AFU-0-Hybrid-alphaFixed1 | 5.0000 | 0.8482 | 0.0089 | -0.0090 | 0.0962 | 0.3601 | 0.9997 | 0.6199 | 0.1982 | 0.8291 | 1.3859 |
| Fashion-MNIST | AFU-2-HeadCov-alphaFixed1 | 5.0000 | 0.8482 | 0.0081 | -0.0090 | 0.0999 | 0.3598 | 0.9997 | 0.6549 | 0.5199 | 1.0763 | 1.6569 |
| Fashion-MNIST | AFU-LNOnly-alphaFixed1 | 5.0000 | 0.8446 | 0.0133 | -0.0054 | 0.1057 | 0.3595 | 0.9998 | 0.6233 | 0.1963 | 0.7691 | 1.6664 |
| Fashion-MNIST | AdamW-alphaFixed1-alphaFixed1 | 5.0000 | 0.8392 | 0.0095 | -0.0000 | 0.0000 | -0.0000 | -0.0000 | 1.0000 | -0.0000 | 1.0000 | 1.0000 |
| Fashion-MNIST | F-V3-HARD-base30-v32best-alphaFixed1 | 5.0000 | 0.8474 | 0.0101 | -0.0082 | 0.1060 | 0.3563 | 0.9998 | 0.6592 | 0.2065 | 0.8169 | 1.3730 |
| KMNIST | AFU-0-Hybrid-alphaFixed1 | 5.0000 | 0.7848 | 0.0122 | -0.0056 | 0.0980 | 0.2772 | 0.9999 | 0.5999 | 0.1836 | 0.7189 | 1.4104 |
| KMNIST | AFU-2-HeadCov-alphaFixed1 | 5.0000 | 0.7636 | 0.0147 | 0.0156 | 0.1196 | 0.2761 | 0.9999 | 0.6584 | 0.3408 | 0.9351 | 1.6656 |
| KMNIST | AFU-LNOnly-alphaFixed1 | 5.0000 | 0.7766 | 0.0201 | 0.0026 | 0.1052 | 0.2771 | 0.9999 | 0.6174 | 0.2011 | 0.7257 | 1.6821 |
| KMNIST | AdamW-alphaFixed1-alphaFixed1 | 5.0000 | 0.7792 | 0.0180 | -0.0000 | -0.0000 | 0.0000 | -0.0000 | 1.0000 | -0.0000 | 1.0000 | 1.0000 |
| KMNIST | K-V3-FULL-base-v32best-alphaFixed1 | 5.0000 | 0.7826 | 0.0151 | -0.0034 | 0.0906 | 0.2765 | 0.9999 | 0.5784 | 0.1859 | 0.6555 | 1.3980 |

## Paired Delta vs AdamW

| dataset | baseline | method | acc delta | acc CI lo | acc CI hi | AUC delta | AUC CI lo | AUC CI hi |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | AdamW-alphaFixed1-alphaFixed1 | AFU-0-Hybrid-alphaFixed1 | 0.0090 | 0.0010 | 0.0170 | 0.0539 | 0.0355 | 0.0735 |
| Fashion-MNIST | AdamW-alphaFixed1-alphaFixed1 | AFU-LNOnly-alphaFixed1 | 0.0054 | -0.0080 | 0.0220 | 0.0592 | 0.0481 | 0.0707 |
| Fashion-MNIST | AdamW-alphaFixed1-alphaFixed1 | AFU-2-HeadCov-alphaFixed1 | 0.0090 | -0.0010 | 0.0192 | 0.0560 | 0.0432 | 0.0688 |
| KMNIST | AdamW-alphaFixed1-alphaFixed1 | AFU-0-Hybrid-alphaFixed1 | 0.0056 | -0.0058 | 0.0184 | 0.0511 | 0.0374 | 0.0643 |
| KMNIST | AdamW-alphaFixed1-alphaFixed1 | AFU-LNOnly-alphaFixed1 | -0.0026 | -0.0124 | 0.0056 | 0.0548 | 0.0330 | 0.0771 |
| KMNIST | AdamW-alphaFixed1-alphaFixed1 | AFU-2-HeadCov-alphaFixed1 | -0.0156 | -0.0290 | -0.0030 | 0.0623 | 0.0502 | 0.0770 |

## Paired Delta vs Hybrid

| dataset | baseline | method | acc delta | acc CI lo | acc CI hi | AUC delta | AUC CI lo | AUC CI hi |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | AFU-0-Hybrid-alphaFixed1 | AFU-LNOnly-alphaFixed1 | -0.0036 | -0.0122 | 0.0046 | 0.0053 | -0.0051 | 0.0151 |
| Fashion-MNIST | AFU-0-Hybrid-alphaFixed1 | AFU-2-HeadCov-alphaFixed1 | 0.0000 | -0.0040 | 0.0024 | 0.0021 | -0.0146 | 0.0195 |
| KMNIST | AFU-0-Hybrid-alphaFixed1 | AFU-LNOnly-alphaFixed1 | -0.0082 | -0.0178 | 0.0006 | 0.0037 | -0.0192 | 0.0294 |
| KMNIST | AFU-0-Hybrid-alphaFixed1 | AFU-2-HeadCov-alphaFixed1 | -0.0212 | -0.0240 | -0.0178 | 0.0112 | -0.0066 | 0.0380 |

## Target-Matched / Wall-Clock

| dataset | method | runs | relaxed loss reached | epochs relaxed loss | time relaxed loss | step ms |
|---|---|---|---|---|---|---|
| Fashion-MNIST | AFU-0-Hybrid-alphaFixed1 | 5.0000 | 1.0000 | 2.2000 | 2.4022 | 45.5125 |
| Fashion-MNIST | AFU-2-HeadCov-alphaFixed1 | 5.0000 | 1.0000 | 2.6000 | 3.3946 | 54.4131 |
| Fashion-MNIST | AFU-LNOnly-alphaFixed1 | 5.0000 | 1.0000 | 2.0000 | 2.6268 | 54.7260 |
| Fashion-MNIST | AdamW-alphaFixed1-alphaFixed1 | 5.0000 | 1.0000 | 1.4000 | 1.1023 | 32.8403 |
| KMNIST | AFU-0-Hybrid-alphaFixed1 | 5.0000 | 1.0000 | 2.8000 | 3.1010 | 46.0484 |
| KMNIST | AFU-2-HeadCov-alphaFixed1 | 5.0000 | 1.0000 | 4.0000 | 5.2234 | 54.3798 |
| KMNIST | AFU-LNOnly-alphaFixed1 | 5.0000 | 1.0000 | 2.8000 | 3.6908 | 54.9193 |
| KMNIST | AdamW-alphaFixed1-alphaFixed1 | 5.0000 | 1.0000 | 2.2000 | 1.7203 | 32.6491 |

## Decision

```text
P0:
  AFU implementation smoke passed without run failures.
  Group metrics were finite. Stem/full AFU already showed early underfit.

P2:
  AFU-2 HeadCov is expression-positive:
    Fashion noKAN and ECE improve strongly.
    KMNIST noKAN/ECE improve, but accuracy drops by about 2.1 points vs Hybrid.
  AFU-LNOnly is the safest component:
    small accuracy cost, positive AUC, geometry preserved.
  AFU-StemOnly fails hard on both datasets.

P3:
  AFU-5 FullDiagLite and AFU-6 FullKFACLite fail as true all-functional optimizers.
  Removing non-KAN AdamW causes strong non-KAN underfit:
    Fashion drops about 4.8-4.9 points vs Hybrid.
    KMNIST drops about 16.2 points vs Hybrid.

P4:
  KMNIST branch_final_scale 0.875 / 0.90 does not solve the strict expression gate.
  Higher final scale trades away accuracy/AUC.

P5:
  AFU-LNOnly does not pass medium gate:
    Fashion is close, but KMNIST is 0.82 points below Hybrid.
  AFU-2 HeadCov does not pass because KMNIST underfits despite better AUC/noKAN/ECE.

P6:
  Not run. No AFU candidate passed P5 strongly enough.

Final:
  True all-functional update is not viable with the current non-KAN metrics.
  Partial AFU is informative, especially HeadCov for expression/calibration analysis and LNOnly as a safe component, but neither replaces Hybrid.
  Hybrid remains the main optimizer:
    KAN coeff uses Sobolev functional update.
    non-KAN parameters still need AdamW.
```
