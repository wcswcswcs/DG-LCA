# DG-KAN v3.7 PureKAN U-FULL 结果复盘

本轮依据 `docs/DG-KAN_v3.7_PureKAN_UFULL_代码检查与实验计划.md`。正式 P2 使用修复后的 `results/gafu_v3_7_p2_schedule_fixed`；旧 P2 只作为发现 warmup 被 epoch-end geometry trigger 截断的中间记录。

## Code / Config Fixes

```text
experiments/dgkan_core.py
  PureKAN alpha_mode=fixed1 now registers alpha as a fixed buffer with value 1.0.
  PureKANClassifier forwards alpha_mode into all residual blocks.
  PureKAN functional update now supports role-specific metrics/LR/trust for input/block/output.
  Result rows include phase/metric traces, active/transition/geometry step counts, per-role update stats, coeff coverage, and PureKAN contribution audit.

experiments/run_gafu_v37.py
  Added v3.7 P0/P1/P2/P3/P5/P6 packages and one-batch shadow step.
  PureKAN warmup now sets geometry_min_epochs=999 so max_active_frac controls 0.10/0.20 schedule length.

experiments/analyze_gafu_v37.py
  Generates this result replay from P0/P1/P2-fixed/P3 CSVs.
```

## P0 Code / Config Audit

| dataset | method | alpha | alpha train | nonKAN params | linear | coeff seen | active | trans | geom | active metric | geom metric | clip |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | U-diagwarmup-smoke | 1.0 | 0 | 0 | 0 | 1.000 | 1 | 0 | 1 | basis_diag_gram\|full_sobolev_gram | full_sobolev_gram | 0.000 |
| Fashion-MNIST | U-full-smoke | 1.0 | 0 | 0 | 0 | 1.000 | 0 | 0 | 2 | full_sobolev_gram | full_sobolev_gram | 0.000 |
| Fashion-MNIST | U-identitywarmup-smoke | 1.0 | 0 | 0 | 0 | 1.000 | 1 | 0 | 1 | identity\|full_sobolev_gram | full_sobolev_gram | 0.000 |
| KMNIST | U-diagwarmup-smoke | 1.0 | 0 | 0 | 0 | 1.000 | 1 | 0 | 1 | basis_diag_gram\|full_sobolev_gram | full_sobolev_gram | 0.000 |
| KMNIST | U-full-smoke | 1.0 | 0 | 0 | 0 | 1.000 | 0 | 0 | 2 | full_sobolev_gram | full_sobolev_gram | 0.000 |
| KMNIST | U-identitywarmup-smoke | 1.0 | 0 | 0 | 0 | 1.000 | 1 | 0 | 1 | identity\|full_sobolev_gram | full_sobolev_gram | 0.000 |
| MNIST | U-diagwarmup-smoke | 1.0 | 0 | 0 | 0 | 1.000 | 1 | 0 | 1 | basis_diag_gram\|full_sobolev_gram | full_sobolev_gram | 0.000 |
| MNIST | U-full-smoke | 1.0 | 0 | 0 | 0 | 1.000 | 0 | 0 | 2 | full_sobolev_gram | full_sobolev_gram | 0.000 |
| MNIST | U-identitywarmup-smoke | 1.0 | 0 | 0 | 0 | 1.000 | 1 | 0 | 1 | identity\|full_sobolev_gram | full_sobolev_gram | 0.000 |

P0 verdict: pass. PureKAN has no non-KAN trainable parameters, alphaFixed1 is truly fixed at 1.0, and functional-update rows cover all coeff parameters (`coeff seen = 1.0`). Warmup smoke rows have real active steps.

## P1 One-Batch Shadow Step

| dataset | method | train descent | val descent | bad | update norm | input cos | block cos | output cos |
|---|---|---|---|---|---|---|---|---|
| MNIST | AdamW-one-step | 0.0229 | 0.0481 | 0 | 1.5117 |  |  |  |
| MNIST | basis-diag-gram | -0.0044 | 0.0491 | 1 | 0.5389 | 0.9915 | 0.9964 | 0.9958 |
| MNIST | full-sobolev-gram | -0.0408 | 0.0414 | 1 | 1.5164 | 0.1303 | 0.2967 | 0.2904 |
| MNIST | identity-functional | -0.0217 | 0.0333 | 1 | 1.0021 | 1.0000 | 1.0000 | 1.0000 |
| MNIST | role-inputDiag-blockFull-outputDiag | -0.0045 | 0.0490 | 1 | 0.5587 | 0.9915 | 0.2967 | 0.9958 |
| MNIST | role-inputIdentity-blockFull-outputIdentity | -0.0223 | 0.0315 | 1 | 1.0028 | 1.0000 | 0.2967 | 1.0000 |
| Fashion-MNIST | AdamW-one-step | 0.0517 | 0.0078 | 0 | 1.6145 |  |  |  |
| Fashion-MNIST | basis-diag-gram | 0.0017 | -0.0540 | 0 | 0.5522 | 0.9994 | 0.9966 | 0.9959 |
| Fashion-MNIST | full-sobolev-gram | 0.0125 | -0.0397 | 0 | 0.6113 | 0.5445 | 0.3165 | 0.2936 |
| Fashion-MNIST | identity-functional | 0.0266 | -0.0343 | 0 | 1.0540 | 1.0000 | 1.0000 | 1.0000 |
| Fashion-MNIST | role-inputDiag-blockFull-outputDiag | 0.0014 | -0.0532 | 0 | 0.5735 | 0.9994 | 0.3165 | 0.9959 |
| Fashion-MNIST | role-inputIdentity-blockFull-outputIdentity | 0.0266 | -0.0355 | 0 | 1.0541 | 1.0000 | 0.3165 | 1.0000 |
| KMNIST | AdamW-one-step | 0.0646 | 0.0501 | 0 | 1.5942 |  |  |  |
| KMNIST | basis-diag-gram | 0.0148 | 0.0040 | 0 | 0.6109 | 0.9908 | 0.9965 | 0.9962 |
| KMNIST | full-sobolev-gram | -0.0021 | 0.0031 | 1 | 1.1753 | 0.3084 | 0.2945 | 0.2896 |
| KMNIST | identity-functional | 0.0151 | -0.0129 | 0 | 1.1218 | 1.0000 | 1.0000 | 1.0000 |
| KMNIST | role-inputDiag-blockFull-outputDiag | 0.0143 | 0.0048 | 0 | 0.6295 | 0.9908 | 0.2945 | 0.9962 |
| KMNIST | role-inputIdentity-blockFull-outputIdentity | 0.0120 | -0.0128 | 0 | 1.1221 | 1.0000 | 0.2945 | 1.0000 |

P1 verdict: the full Sobolev direction is not reliably descent for PureKAN. It is a bad train step on MNIST and KMNIST, and on Fashion it improves the train batch but worsens validation loss. This points to a direction/metric mismatch rather than a missing implementation hook.

## P2 Metric Schedule Repair Test

| dataset | method | runs | acc | std | gap vs AdamW | AUC imp vs full | AUC imp vs AdamW | ECE red | phi p95 | J | active | trans | geom |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MNIST | AdamW | 3 | 0.9303 | 0.0076 | 0.0000 | 0.3510 | 0.0000 | 0.0000 | 0.2274 | 226275.4421 | 0.0 | 0.0 | 0.0 |
| MNIST | U-diag-to-full-smooth-0p20 | 3 | 0.8950 | 0.0107 | 0.0353 | -0.0707 | -0.4217 | -0.8321 | 0.1501 | 24.5974 | 96.0 | 49.0 | 335.0 |
| MNIST | U-diagwarmup-0p10 | 3 | 0.8930 | 0.0142 | 0.0373 | -0.1108 | -0.4618 | -1.2273 | 0.1497 | 31.0922 | 48.0 | 0.0 | 432.0 |
| MNIST | U-diagwarmup-0p20 | 3 | 0.8917 | 0.0135 | 0.0387 | -0.1176 | -0.4686 | -1.1325 | 0.1500 | 24.1682 | 96.0 | 0.0 | 384.0 |
| MNIST | U-full | 3 | 0.8870 | 0.0455 | 0.0433 | 0.0000 | -0.3510 | -0.3594 | 0.1489 | 34.4779 | 0.0 | 0.0 | 480.0 |
| MNIST | U-identitywarmup-0p10 | 3 | 0.8937 | 0.0153 | 0.0367 | -0.0242 | -0.3753 | -0.6659 | 0.1490 | 42.6543 | 48.0 | 0.0 | 432.0 |
| MNIST | U-identitywarmup-0p20 | 3 | 0.8933 | 0.0210 | 0.0370 | -0.0387 | -0.3897 | -0.6675 | 0.1492 | 38.2002 | 96.0 | 0.0 | 384.0 |
| Fashion-MNIST | AdamW | 3 | 0.8563 | 0.0090 | 0.0000 | 0.0388 | 0.0000 | 0.0000 | 0.2396 | 369574.0127 | 0.0 | 0.0 | 0.0 |
| Fashion-MNIST | U-diag-to-full-smooth-0p20 | 3 | 0.8460 | 0.0127 | 0.0103 | 0.1211 | 0.0822 | 0.7611 | 0.1498 | 54.0007 | 144.0 | 71.0 | 505.0 |
| Fashion-MNIST | U-diagwarmup-0p10 | 3 | 0.8350 | 0.0107 | 0.0213 | 0.0724 | 0.0336 | 0.7173 | 0.1495 | 97.4180 | 72.0 | 0.0 | 648.0 |
| Fashion-MNIST | U-diagwarmup-0p20 | 3 | 0.8450 | 0.0127 | 0.0113 | 0.1097 | 0.0708 | 0.7543 | 0.1497 | 75.3736 | 144.0 | 0.0 | 576.0 |
| Fashion-MNIST | U-full | 3 | 0.8320 | 0.0120 | 0.0243 | 0.0000 | -0.0388 | 0.6596 | 0.1490 | 31.4545 | 0.0 | 0.0 | 720.0 |
| Fashion-MNIST | U-identitywarmup-0p10 | 3 | 0.8370 | 0.0122 | 0.0193 | 0.0761 | 0.0373 | 0.7093 | 0.1490 | 46.0660 | 72.0 | 0.0 | 648.0 |
| Fashion-MNIST | U-identitywarmup-0p20 | 3 | 0.8303 | 0.0093 | 0.0260 | 0.1025 | 0.0637 | 0.6540 | 0.1492 | 43.0049 | 144.0 | 0.0 | 576.0 |
| KMNIST | AdamW | 3 | 0.7720 | 0.0102 | 0.0000 | 0.4078 | 0.0000 | 0.0000 | 0.2172 | 20374.7067 | 0.0 | 0.0 | 0.0 |
| KMNIST | U-diag-to-full-smooth-0p20 | 3 | 0.7267 | 0.0156 | 0.0453 | 0.1475 | -0.2603 | 0.6515 | 0.1503 | 33.0119 | 96.0 | 49.0 | 335.0 |
| KMNIST | U-diagwarmup-0p10 | 3 | 0.6883 | 0.0264 | 0.0837 | 0.0058 | -0.4021 | 0.5123 | 0.1499 | 31.9268 | 48.0 | 0.0 | 432.0 |
| KMNIST | U-diagwarmup-0p20 | 3 | 0.7100 | 0.0209 | 0.0620 | 0.1045 | -0.3033 | 0.6437 | 0.1501 | 31.3366 | 96.0 | 0.0 | 384.0 |
| KMNIST | U-full | 3 | 0.6783 | 0.0147 | 0.0937 | 0.0000 | -0.4078 | 0.7098 | 0.1493 | 35.5405 | 0.0 | 0.0 | 480.0 |
| KMNIST | U-identitywarmup-0p10 | 3 | 0.6870 | 0.0248 | 0.0850 | 0.0724 | -0.3355 | 0.6234 | 0.1495 | 27.4743 | 48.0 | 0.0 | 432.0 |
| KMNIST | U-identitywarmup-0p20 | 3 | 0.7160 | 0.0148 | 0.0560 | 0.1436 | -0.2642 | 0.6583 | 0.1497 | 27.1292 | 96.0 | 0.0 | 384.0 |

P2 verdict: schedule repair is real after the fix. 0.10 and 0.20 now have distinct active-step counts, and smooth has transition steps. Fashion benefits most (`diag-to-full-smooth-0.20` reaches acc 0.8460 and val-loss AUC 0.5242), but MNIST and KMNIST remain far below PureKAN-AdamW in accuracy.

## P3 Role-Wise PureKAN U-FULL Sweep

| dataset | method | runs | acc | std | gap vs AdamW | AUC imp vs full | AUC imp vs AdamW | ECE red | phi p95 | J | active | trans | geom |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MNIST | U-allDiag-o2 | 3 | 0.8977 | 0.0129 | 0.0327 | -0.0449 | -0.3959 | 0.1843 | 0.1494 | 31.4169 | 0.0 | 0.0 | 480.0 |
| MNIST | U-allfull | 3 | 0.8870 | 0.0455 | 0.0433 | 0.0000 | -0.3510 | -0.3594 | 0.1489 | 34.4779 | 0.0 | 0.0 | 480.0 |
| MNIST | U-iDiag-blockFull-oId | 3 | 0.8703 | 0.0464 | 0.0600 | -0.0343 | -0.3853 | 0.2063 | 0.1487 | 27.6383 | 0.0 | 0.0 | 480.0 |
| MNIST | U-iId-blockFull-oDiag-o2 | 3 | 0.8817 | 0.0058 | 0.0487 | -0.0333 | -0.3843 | 0.1423 | 0.1487 | 27.9629 | 0.0 | 0.0 | 480.0 |
| MNIST | U-iId-blockFull-oDiag-o4 | 3 | 0.9057 | 0.0144 | 0.0247 | -0.1433 | -0.4943 | 0.2380 | 0.1486 | 28.1544 | 0.0 | 0.0 | 480.0 |
| MNIST | U-ioDiag-blockFull-i2o2 | 3 | 0.8980 | 0.0227 | 0.0323 | -0.0198 | -0.3708 | 0.2073 | 0.1489 | 29.5763 | 0.0 | 0.0 | 480.0 |
| MNIST | U-ioDiag-blockFull-o2 | 3 | 0.8963 | 0.0088 | 0.0340 | -0.0291 | -0.3801 | 0.2247 | 0.1488 | 29.7143 | 0.0 | 0.0 | 480.0 |
| MNIST | U-ioDiag-blockFull-o4 | 3 | 0.9020 | 0.0195 | 0.0283 | -0.1123 | -0.4633 | 0.3372 | 0.1489 | 28.7376 | 0.0 | 0.0 | 480.0 |
| Fashion-MNIST | U-allDiag-o2 | 3 | 0.8457 | 0.0091 | 0.0107 | 0.0550 | 0.0161 | 0.5238 | 0.1497 | 38.1382 | 0.0 | 0.0 | 720.0 |
| Fashion-MNIST | U-allfull | 3 | 0.8320 | 0.0120 | 0.0243 | 0.0000 | -0.0388 | 0.6596 | 0.1490 | 31.4545 | 0.0 | 0.0 | 720.0 |
| Fashion-MNIST | U-iDiag-blockFull-oId | 3 | 0.8433 | 0.0083 | 0.0130 | 0.0673 | 0.0285 | 0.5679 | 0.1487 | 25.5139 | 0.0 | 0.0 | 720.0 |
| Fashion-MNIST | U-iId-blockFull-oDiag-o2 | 3 | 0.8453 | 0.0082 | 0.0110 | 0.0650 | 0.0262 | 0.5902 | 0.1487 | 26.6611 | 0.0 | 0.0 | 720.0 |
| Fashion-MNIST | U-iId-blockFull-oDiag-o4 | 3 | 0.8247 | 0.0211 | 0.0317 | 0.0680 | 0.0292 | 0.5529 | 0.1488 | 30.7958 | 0.0 | 0.0 | 720.0 |
| Fashion-MNIST | U-ioDiag-blockFull-i2o2 | 3 | 0.8470 | 0.0050 | 0.0093 | 0.0734 | 0.0346 | 0.5923 | 0.1488 | 28.3585 | 0.0 | 0.0 | 720.0 |
| Fashion-MNIST | U-ioDiag-blockFull-o2 | 3 | 0.8383 | 0.0170 | 0.0180 | 0.0622 | 0.0233 | 0.5533 | 0.1489 | 27.4369 | 0.0 | 0.0 | 720.0 |
| Fashion-MNIST | U-ioDiag-blockFull-o4 | 3 | 0.8473 | 0.0005 | 0.0090 | 0.0465 | 0.0077 | 0.4576 | 0.1489 | 29.9797 | 0.0 | 0.0 | 720.0 |
| KMNIST | U-allDiag-o2 | 3 | 0.7430 | 0.0220 | 0.0290 | 0.1475 | -0.2604 | 0.6150 | 0.1499 | 32.0730 | 0.0 | 0.0 | 480.0 |
| KMNIST | U-allfull | 3 | 0.6783 | 0.0147 | 0.0937 | 0.0000 | -0.4078 | 0.7098 | 0.1493 | 35.5405 | 0.0 | 0.0 | 480.0 |
| KMNIST | U-iDiag-blockFull-oId | 3 | 0.7277 | 0.0242 | 0.0443 | 0.1365 | -0.2713 | 0.7125 | 0.1492 | 26.0145 | 0.0 | 0.0 | 480.0 |
| KMNIST | U-iId-blockFull-oDiag-o2 | 3 | 0.7063 | 0.0223 | 0.0657 | 0.1343 | -0.2736 | 0.6394 | 0.1491 | 25.6610 | 0.0 | 0.0 | 480.0 |
| KMNIST | U-iId-blockFull-oDiag-o4 | 3 | 0.7480 | 0.0234 | 0.0240 | 0.1387 | -0.2691 | 0.6493 | 0.1489 | 54.4384 | 0.0 | 0.0 | 480.0 |
| KMNIST | U-ioDiag-blockFull-i2o2 | 3 | 0.7253 | 0.0212 | 0.0467 | 0.1568 | -0.2510 | 0.6767 | 0.1491 | 28.1022 | 0.0 | 0.0 | 480.0 |
| KMNIST | U-ioDiag-blockFull-o2 | 3 | 0.7440 | 0.0086 | 0.0280 | 0.1569 | -0.2509 | 0.7600 | 0.1493 | 29.7018 | 0.0 | 0.0 | 480.0 |
| KMNIST | U-ioDiag-blockFull-o4 | 3 | 0.7407 | 0.0161 | 0.0313 | 0.1467 | -0.2612 | 0.5869 | 0.1490 | 28.2855 | 0.0 | 0.0 | 480.0 |

### P3 Gate Check

| method | min acc | max acc gap | min AUC imp vs full | ECE ok | enter P5 |
|---|---|---|---|---|---|
| U-allDiag-o2 | 0.7430 | 0.0327 | -0.0449 | yes | no |
| U-iDiag-blockFull-oId | 0.7277 | 0.0600 | -0.0343 | yes | no |
| U-iId-blockFull-oDiag-o2 | 0.7063 | 0.0657 | -0.0333 | yes | no |
| U-iId-blockFull-oDiag-o4 | 0.7480 | 0.0317 | -0.1433 | yes | no |
| U-ioDiag-blockFull-i2o2 | 0.7253 | 0.0467 | -0.0198 | yes | no |
| U-ioDiag-blockFull-o2 | 0.7440 | 0.0340 | -0.0291 | yes | no |
| U-ioDiag-blockFull-o4 | 0.7407 | 0.0313 | -0.1123 | yes | no |

P3 verdict: no candidate enters P5. Fashion is mostly repaired by role-aware metrics, but MNIST and KMNIST still miss the `acc gap < 1%` requirement. The best KMNIST role candidate is still about 2.4 points behind PureKAN-AdamW; the best MNIST role candidate is also about 2.5 points behind.

## P4-P7 Decision

```text
P4 capacity/epoch budget check: not expanded.
Reason: P3 did not produce a candidate close enough to PureKAN-AdamW on all three datasets.

P5 3-seed candidate selection: not run.
Reason: P3 entry gate failed for every role-wise candidate.

P6 5-seed confirm and P7 speed proxy: not allowed by plan because P5 was not reached.
```

## Failure Diagnosis

```text
Implementation failure: no.
  alphaFixed1, coeff coverage, pure-only parameterization, metric traces, and active-step counts pass audit.

Schedule failure: partially fixed but not sufficient.
  The original warmup length was indeed collapsed by epoch-end geometry triggers.
  After fixing it, Fashion improves, but MNIST/KMNIST still do not approach AdamW.

Direction failure: yes.
  P1 shadow steps show full Sobolev is not reliable descent on PureKAN.
  P2/P3 improve some trajectories by avoiding full metric on input/output, but full-network PureKAN remains unstable on hard datasets.

Budget failure: unlikely as the primary cause.
  The gap is already visible in one-step shadow descent and 3-seed role sweeps.
  Extending 5/10-seed confirm is not justified until the metric direction is repaired.
```

## Final Decision

```text
PureKAN U-FULL is not accepted in v3.7.

What is confirmed:
  1. PureKAN alpha/fixed1 and functional coverage bugs are fixed.
  2. Warmup schedule now actually executes 10%/20% active windows.
  3. Role-aware metrics substantially improve Fashion and partially rescue KMNIST.

What is not confirmed:
  1. Full-network PureKAN functional update does not match PureKAN-AdamW on MNIST/KMNIST.
  2. No role/schedule candidate qualifies for P5/P6.
  3. The bottleneck is metric direction quality, not missing CUDA/KAT backward or missing parameter coverage.

Next recommended research direction:
  Redesign the PureKAN functional metric itself, likely with layer-local or block-local objectives and a safer output/input metric, before spending more seeds on confirmation.
```
