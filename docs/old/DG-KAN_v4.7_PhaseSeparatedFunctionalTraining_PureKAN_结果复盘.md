# DG-KAN v4.7 Phase-Separated Functional Training 结果复盘

本轮依据 `docs/DG-KAN_v4.7_PhaseSeparatedFunctionalTraining_PureKAN_实验计划.md`。目标是验证 PSFT：先用 functional-coordinate dynamics 学表示，再用 teacher-preserving Sobolev consolidation 降几何。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 PSFT smoke | 48 | 0 |
| P1 Phase-I dynamics | 120 | 0 |
| P2 consolidation sweep | 864 | 0 |

## Code / Config Changes

```text
experiments/run_gafu_v47.py
  Added PSFT phase-I methods, role-staged Phase-I control, teacher snapshots,
  KL/feature distillation, and Sobolev-style consolidation sweep.

experiments/analyze_gafu_v47.py
  Generates P1/P2 gate summaries, failure taxonomy, figures, aggregate_decision.json,
  and this result replay.
```

## P0 Implementation Smoke

| rows | errors | max nonKAN | min cov | max rollback | pass |
|---|---|---|---|---|---|
| 48 | 0 | 0.0000 | 1.0000 | 0.0000 | true |

P0 verdict: pass. PSFT snapshot/distillation paths ran with strict PureKAN non-KAN count at zero and finite loss/rollback checks.

## P1 Phase-I Learning Dynamics

| dataset | method | runs | hold100 | hold/A | rank | margin Δ | phi | role | P1 |
|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | FCAdam-dataSob | 1 | 1.7537 | 0.9580 | 0.3362 | 4.5014 | 1.8633 | 0.0121 | no |
| Fashion-MNIST | FLD-AdanLite | 1 | 1.6657 | 0.9100 | 0.3401 | 4.0494 | 1.8288 | 0.0108 | no |
| Fashion-MNIST | FLD-LyapunovRestart | 1 | 1.6657 | 0.9100 | 0.3401 | 4.0494 | 1.8288 | 0.0108 | no |
| Fashion-MNIST | FLD-WinLite | 1 | 1.7440 | 0.9528 | 0.3344 | 4.4471 | 1.8621 | 0.0114 | no |
| Fashion-MNIST | PSFT-A-Adan-PhaseI-only | 1 | 1.7450 | 0.9533 | 0.3344 | 4.4453 | 1.8639 | 0.0114 | no |
| Fashion-MNIST | PSFT-A-FCAdam-PhaseI-only | 1 | 1.7486 | 0.9553 | 0.3340 | 4.4398 | 1.8638 | 0.0113 | no |
| Fashion-MNIST | PSFT-D-role-staged-PhaseI-only | 1 | 1.6939 | 0.9254 | 0.3371 | 4.4635 | 1.9636 | 0.0342 | no |
| Fashion-MNIST | PureKAN-AdamW | 1 | 1.8305 | 1.0000 | 0.3423 | 2.7921 | 1.1569 | 0.0790 | yes |
| KMNIST | FCAdam-dataSob | 1 | 1.6198 | 0.8547 | 0.3514 | 3.8614 | 1.9127 | 0.0359 | no |
| KMNIST | FLD-AdanLite | 1 | 1.6059 | 0.8474 | 0.3411 | 3.7082 | 1.8794 | 0.0426 | no |
| KMNIST | FLD-LyapunovRestart | 1 | 1.6059 | 0.8474 | 0.3411 | 3.7082 | 1.8794 | 0.0426 | no |
| KMNIST | FLD-WinLite | 1 | 1.6222 | 0.8560 | 0.3506 | 3.8315 | 1.9022 | 0.0315 | no |
| KMNIST | PSFT-A-Adan-PhaseI-only | 1 | 1.6222 | 0.8560 | 0.3506 | 3.8331 | 1.9039 | 0.0316 | no |
| KMNIST | PSFT-A-FCAdam-PhaseI-only | 1 | 1.6227 | 0.8562 | 0.3505 | 3.8268 | 1.9025 | 0.0307 | no |
| KMNIST | PSFT-D-role-staged-PhaseI-only | 1 | 1.6148 | 0.8521 | 0.3504 | 3.9508 | 1.9824 | 0.0567 | no |
| KMNIST | PureKAN-AdamW | 1 | 1.8952 | 1.0000 | 0.4716 | 2.9795 | 1.1879 | 0.0711 | yes |
| MNIST | FCAdam-dataSob | 1 | 1.9144 | 1.0162 | 0.3209 | 4.7558 | 1.9615 | 0.0247 | no |
| MNIST | FLD-AdanLite | 1 | 1.9343 | 1.0268 | 0.3194 | 4.6853 | 1.8986 | 0.0267 | no |
| MNIST | FLD-LyapunovRestart | 1 | 1.9343 | 1.0268 | 0.3194 | 4.6853 | 1.8986 | 0.0267 | no |
| MNIST | FLD-WinLite | 1 | 1.9159 | 1.0170 | 0.3209 | 4.7193 | 1.9503 | 0.0213 | no |
| MNIST | PSFT-A-Adan-PhaseI-only | 1 | 1.9158 | 1.0169 | 0.3209 | 4.7211 | 1.9523 | 0.0213 | no |
| MNIST | PSFT-A-FCAdam-PhaseI-only | 1 | 1.9161 | 1.0171 | 0.3209 | 4.7141 | 1.9499 | 0.0206 | no |
| MNIST | PSFT-D-role-staged-PhaseI-only | 1 | 1.9098 | 1.0137 | 0.3197 | 4.8414 | 2.0182 | 0.0449 | no |
| MNIST | PureKAN-AdamW | 1 | 1.8839 | 1.0000 | 0.3448 | 3.1572 | 1.2214 | 0.0630 | yes |

P1 survivors:

```text
none
```

## P2 Geometry Consolidation Test

P2 grid mode is recorded in `p2_consolidation_sweep.csv`. The table below shows the top rows by dataset/geometry signal, truncated for readability.

| dataset | teacher | steps | tau_z | tau_h | lambda | acc drop | phi red | J red | KL | P2 |
|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 2.0 | 0.1 | 0.0001 | 0.6875 | 0.0541 | -1.1591 | 5.1665 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 2.0 | 0.1 | 0.0003 | 0.6895 | 0.0532 | 0.8618 | 5.1870 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 2.0 | 0.1 | 0.001 | 0.6992 | 0.0520 | -2.4677 | 5.2387 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 2.0 | 0.1 | 0.003 | 0.7012 | 0.0506 | -17.0242 | 5.2958 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 1.0 | 0.1 | 0.0003 | 0.5742 | 0.0489 | -0.0843 | 4.5792 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 1.0 | 0.1 | 0.0001 | 0.5703 | 0.0487 | -0.9209 | 4.5444 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 1.0 | 0.1 | 0.001 | 0.6113 | 0.0482 | 0.7913 | 4.6679 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 1.0 | 0.1 | 0.003 | 0.6367 | 0.0460 | 0.8977 | 4.7802 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 2.0 | 0.5 | 0.0003 | 0.5703 | 0.0396 | 0.7950 | 4.6827 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 2.0 | 0.5 | 0.001 | 0.5781 | 0.0395 | 0.7906 | 4.7524 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 2.0 | 0.5 | 0.0001 | 0.5664 | 0.0395 | 0.8660 | 4.6537 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 2.0 | 0.5 | 0.003 | 0.5977 | 0.0393 | 0.3589 | 4.8221 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 20 | 2.0 | 0.1 | 0.0001 | 0.6348 | 0.0352 | 0.9229 | 5.0232 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 20 | 2.0 | 0.1 | 0.0003 | 0.6367 | 0.0350 | 0.8019 | 5.0379 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 20 | 2.0 | 0.1 | 0.001 | 0.6348 | 0.0347 | 0.8873 | 5.0812 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 20 | 2.0 | 0.1 | 0.0003 | 0.6484 | 0.0346 | 0.9484 | 5.0901 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 20 | 2.0 | 0.1 | 0.003 | 0.6348 | 0.0345 | 0.7589 | 5.1622 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 20 | 2.0 | 0.1 | 0.0001 | 0.6484 | 0.0343 | 0.2624 | 5.0769 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 20 | 2.0 | 0.1 | 0.003 | 0.6504 | 0.0343 | 0.4407 | 5.2003 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 20 | 2.0 | 0.1 | 0.001 | 0.6504 | 0.0339 | 0.6114 | 5.1273 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 100 | 2.0 | 0.1 | 0.003 | 0.3867 | 0.0322 | 0.7597 | 4.1487 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 100 | 2.0 | 0.1 | 0.001 | 0.3750 | 0.0303 | -1.9439 | 4.1645 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 20 | 1.0 | 0.1 | 0.003 | 0.5312 | 0.0299 | 0.8980 | 4.6829 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 20 | 1.0 | 0.1 | 0.003 | 0.5801 | 0.0292 | -9.5062 | 4.7137 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 20 | 1.0 | 0.1 | 0.001 | 0.5508 | 0.0288 | 0.4053 | 4.6022 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 20 | 1.0 | 0.1 | 0.001 | 0.5195 | 0.0285 | 0.7399 | 4.5714 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 20 | 1.0 | 0.1 | 0.0003 | 0.4980 | 0.0283 | -1.4379 | 4.5032 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 100 | 2.0 | 0.1 | 0.0003 | 0.3867 | 0.0282 | 0.9205 | 4.1677 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 20 | 1.0 | 0.1 | 0.0001 | 0.4883 | 0.0280 | 0.8840 | 4.4775 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 20 | 1.0 | 0.1 | 0.0003 | 0.5391 | 0.0279 | 0.8383 | 4.5331 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 100 | 2.0 | 0.1 | 0.0001 | 0.3906 | 0.0277 | 0.6626 | 4.1719 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 20 | 1.0 | 0.1 | 0.0001 | 0.5332 | 0.0274 | 0.0861 | 4.5069 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 100 | 2.0 | 0.1 | 0.003 | 0.3613 | 0.0268 | 0.5713 | 4.2272 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 0.5 | 0.1 | 0.0001 | 0.1797 | 0.0265 | -2.5562 | 3.0535 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 0.5 | 0.1 | 0.0003 | 0.1855 | 0.0264 | 0.9284 | 3.0739 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 0.5 | 0.1 | 0.001 | 0.1992 | 0.0259 | 0.6564 | 3.1266 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 100 | 1.0 | 0.1 | 0.001 | 0.2422 | 0.0244 | 0.1819 | 3.7577 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 50 | 2.0 | 0.1 | 0.003 | 0.3965 | 0.0238 | 0.7481 | 4.0190 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 100 | 1.0 | 0.1 | 0.003 | 0.2383 | 0.0238 | 0.4616 | 3.7294 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 100 | 2.0 | 0.1 | 0.001 | 0.3477 | 0.0238 | 0.8880 | 4.1983 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 0.5 | 0.1 | 0.003 | 0.2090 | 0.0236 | -0.5448 | 3.1957 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 20 | 2.0 | 0.5 | 0.003 | 0.5117 | 0.0217 | 0.0300 | 4.5404 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 20 | 2.0 | 0.5 | 0.003 | 0.4961 | 0.0217 | 0.9142 | 4.4696 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 100 | 1.0 | 0.1 | 0.0003 | 0.2402 | 0.0216 | 0.5873 | 3.7580 | no |
| Fashion-MNIST | TeacherA-AdamW100 | 100 | 1.0 | 0.5 | 0.003 | 0.0820 | 0.0214 | 0.5083 | 2.4464 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 50 | 2.0 | 0.1 | 0.001 | 0.3750 | 0.0213 | 0.5725 | 3.9737 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 100 | 2.0 | 0.1 | 0.0003 | 0.3320 | 0.0211 | 0.1838 | 4.1720 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 20 | 2.0 | 0.5 | 0.001 | 0.5020 | 0.0207 | 0.5300 | 4.4432 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 100 | 1.0 | 0.1 | 0.0001 | 0.2344 | 0.0207 | 0.9019 | 3.7594 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 100 | 2.0 | 0.1 | 0.0001 | 0.3359 | 0.0205 | 0.7776 | 4.1618 | no |
| Fashion-MNIST | TeacherA-AdamW100 | 100 | 0.5 | 0.5 | 0.003 | 0.0332 | 0.0204 | 0.5357 | 2.1262 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 50 | 2.0 | 0.1 | 0.0003 | 0.3750 | 0.0203 | 0.8365 | 3.9581 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 50 | 2.0 | 0.1 | 0.0001 | 0.3770 | 0.0201 | -1.4205 | 3.9561 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 20 | 2.0 | 0.5 | 0.001 | 0.4844 | 0.0197 | 0.7630 | 4.3626 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 20 | 2.0 | 0.5 | 0.0003 | 0.4883 | 0.0195 | -3.0646 | 4.3758 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 50 | 2.0 | 0.1 | 0.001 | 0.3672 | 0.0191 | 0.6996 | 4.0393 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 20 | 2.0 | 0.5 | 0.0001 | 0.4824 | 0.0190 | 0.6028 | 4.3484 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 50 | 2.0 | 0.1 | 0.0003 | 0.3633 | 0.0188 | 0.7809 | 4.0312 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 20 | 2.0 | 0.5 | 0.0003 | 0.4727 | 0.0188 | -5.7174 | 4.2867 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 50 | 2.0 | 0.1 | 0.003 | 0.3711 | 0.0187 | 0.4103 | 4.0616 | no |
| Fashion-MNIST | TeacherC-FCAdam100 | 50 | 2.0 | 0.1 | 0.0001 | 0.3594 | 0.0186 | 0.5930 | 4.0294 | no |
| Fashion-MNIST | TeacherB-PSFT-FCAdam100 | 20 | 2.0 | 0.5 | 0.0001 | 0.4688 | 0.0185 | 0.2571 | 4.2565 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 100 | 0.5 | 0.5 | 0.0001 | 0.2051 | 0.0183 | 0.7893 | 3.5029 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 1.0 | 0.5 | 0.0003 | 0.2676 | 0.0182 | 0.9123 | 3.1963 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 100 | 0.5 | 0.5 | 0.0003 | 0.2070 | 0.0179 | 0.6572 | 3.5072 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 1.0 | 0.5 | 0.0001 | 0.2598 | 0.0177 | 0.8627 | 3.1693 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 1.0 | 0.5 | 0.001 | 0.2793 | 0.0175 | -0.7660 | 3.2536 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 100 | 0.5 | 0.5 | 0.001 | 0.2148 | 0.0173 | 0.5510 | 3.5180 | no |
| Fashion-MNIST | TeacherA-AdamW100 | 100 | 2.0 | 0.5 | 0.003 | 0.2246 | 0.0172 | 0.1945 | 2.7433 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 50 | 1.0 | 0.1 | 0.003 | 0.1914 | 0.0171 | 0.9172 | 3.3689 | no |
| Fashion-MNIST | TeacherA-AdamW100 | 100 | 1.0 | 0.5 | 0.001 | 0.0938 | 0.0169 | 0.4886 | 2.4740 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 100 | 2.0 | 0.5 | 0.003 | 0.4492 | 0.0167 | 0.0973 | 4.2842 | no |
| Fashion-MNIST | TeacherA-AdamW100 | 20 | 2.0 | 0.1 | 0.001 | 0.5684 | 0.0162 | 0.4325 | 3.0831 | no |
| Fashion-MNIST | TeacherA-AdamW100 | 20 | 2.0 | 0.1 | 0.003 | 0.5762 | 0.0160 | 0.4995 | 3.0894 | no |
| Fashion-MNIST | TeacherA-AdamW100 | 20 | 2.0 | 0.1 | 0.0003 | 0.5645 | 0.0159 | 0.3875 | 3.0760 | no |
| Fashion-MNIST | TeacherA-AdamW100 | 100 | 2.0 | 0.5 | 0.001 | 0.2617 | 0.0157 | 0.2106 | 2.7841 | no |
| Fashion-MNIST | TeacherA-AdamW100 | 20 | 2.0 | 0.1 | 0.0001 | 0.5586 | 0.0156 | 0.3732 | 3.0733 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 20 | 1.0 | 0.5 | 0.003 | 0.3027 | 0.0156 | 0.6107 | 3.2721 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 50 | 1.0 | 0.1 | 0.001 | 0.2109 | 0.0156 | 0.3470 | 3.3890 | no |
| Fashion-MNIST | TeacherD-FLD-Adan100 | 50 | 1.0 | 0.1 | 0.0003 | 0.2012 | 0.0155 | 0.8380 | 3.4139 | no |

Best diagnostic points:

| dataset | teacher | best by | phi red | acc drop | KL |
|---|---|---|---|---|---|
| MNIST | TeacherA-AdamW100 | phi | 0.0449 | 0.1328 | 1.9357 |
| MNIST | TeacherA-AdamW100 | safe | 0.0161 | 0.0703 | 1.1131 |
| Fashion-MNIST | TeacherD-FLD-Adan100 | phi | 0.0541 | 0.6875 | 5.1665 |
| Fashion-MNIST | TeacherA-AdamW100 | safe | 0.0081 | 0.0059 | 1.0592 |
| KMNIST | TeacherB-PSFT-FCAdam100 | phi | 0.0599 | 0.3027 | 3.8296 |
| KMNIST | TeacherA-AdamW100 | safe | 0.0082 | 0.0957 | 0.8802 |

P2 all-dataset survivors:

```text
none
```

## P3-P8 Decision

```text
P3 one-cycle PSFT micro-run: not run; P2 produced no survivor
P4-P8: not run by gate.
```

## Failure Diagnosis

Generated:

```text
p9_failure_diagnosis.csv
failure_table.csv
failure_by_dataset.csv
failure_by_method.csv
```

Diagnosis:

```text
Phase separation is implemented, but P2 is the bottleneck.
If consolidation preserves teacher predictions, geometry reduction is weak;
when geometry reduction appears, accuracy/teacher preservation usually fails.
This supports the v4.7 diagnostic: representation learning and geometry projection
cannot yet be connected by the current Sobolev distillation step.
```

## Required Artifacts

Written under `results/v4_7/`:

```text
p0_psft_invariants.csv
p1_phase_i_dynamics.csv
p1_phase_i_gate_summary.csv
p2_consolidation_sweep.csv
p2_consolidation_gate_summary.csv
p3_one_cycle_micro_run.csv
p4_cycle_ablation.csv
p5_role_stage_ablation.csv
p6_cifar_precheck.csv
p7_confirm5.csv
p8_final10.csv
p9_failure_diagnosis.csv
failure_table.csv
aggregate_decision.json
figures/
```

## Final Decision

```text
PureKAN PSFT status:
  stop_after_p2_consolidation_failed

What improved:
  Phase-I and teacher snapshot/distillation machinery are now auditable.
  P2 directly tests whether a learned PureKAN function can be geometrically consolidated.

What failed:
  No P2 configuration passed the joint teacher-preservation + phi/J reduction gate across all datasets.

Conclusion:
  PSFT is the right diagnostic framing, but the current consolidation operator is not sufficient.
  Next work should redesign the Phase-II projection itself before spending P3/P7 seed budget.
```
