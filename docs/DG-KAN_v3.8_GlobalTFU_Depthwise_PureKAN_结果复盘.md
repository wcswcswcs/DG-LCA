# DG-KAN v3.8 GlobalTFU / Depthwise PureKAN 结果复盘

本轮依据 `docs/DG-KAN_v3.8_修改版_GlobalTFU_Depthwise_PureKAN_实验计划.md`。目标是验证 PureKAN 的 task-aware functional update 是否能修复 v3.7 中 full Sobolev 方向不可靠的问题。

## Code / Config Changes

```text
experiments/dgkan_core.py
  TrainConfig / RuntimeState added TFU fields, role-specific input/shallow/deep/output metrics, LR multipliers and cosine safeguard counters.
  PureKAN functional update now supports tfu_task_diag and tfu_data_task_diag.
  Shallow/deep block roles are split from blocks.{idx} and also aggregated into block stats.
  Result rows include TFU metric min/max, per-role conditions, fallback counters, and safeguard fail rates.

experiments/run_gafu_v38.py
  Added P0 smoke, P1 shadow audit, P2 global TFU, P3 depth-wise sweep, and conditional P4 entry wiring.

experiments/analyze_gafu_v38.py
  Generates this replay and appends a concise log entry.
```

## P0 Implementation Smoke

| dataset | method | alpha | alpha train | nonKAN | linear | coeff seen | input | shallow | deep | output | cond in | cond out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | D0-allFullSobolev | 1.0 | 0 | 0 | 0 | 1.000 | full_sobolev_gram |  |  | full_sobolev_gram | 6551.62 | 6551.62 |
| Fashion-MNIST | D1-frontTask-backSob | 1.0 | 0 | 0 | 0 | 1.000 | tfu_data_task_diag | tfu_data_task_diag | full_sobolev_gram | full_sobolev_gram | 6.65 | 6551.62 |
| Fashion-MNIST | D2-frontSob-backTask | 1.0 | 0 | 0 | 0 | 1.000 | basis_diag_gram | full_sobolev_gram | tfu_task_diag | tfu_task_diag | 6918.15 | 21.23 |
| Fashion-MNIST | D3-taskSobTask | 1.0 | 0 | 0 | 0 | 1.000 | tfu_data_task_diag | full_sobolev_gram | full_sobolev_gram | tfu_task_diag | 6.65 | 21.38 |
| Fashion-MNIST | D6-allTaskAware | 1.0 | 0 | 0 | 0 | 1.000 | tfu_data_task_diag | tfu_data_task_diag | tfu_task_diag | tfu_task_diag | 6.66 | 21.40 |
| Fashion-MNIST | D7-inputOutputTask-middleSob | 1.0 | 0 | 0 | 0 | 1.000 | tfu_data_task_diag | full_sobolev_gram | full_sobolev_gram | tfu_task_diag | 6.65 | 21.38 |
| KMNIST | D0-allFullSobolev | 1.0 | 0 | 0 | 0 | 1.000 | full_sobolev_gram |  |  | full_sobolev_gram | 6551.62 | 6551.62 |
| KMNIST | D1-frontTask-backSob | 1.0 | 0 | 0 | 0 | 1.000 | tfu_data_task_diag | tfu_data_task_diag | full_sobolev_gram | full_sobolev_gram | 5.68 | 6551.62 |
| KMNIST | D2-frontSob-backTask | 1.0 | 0 | 0 | 0 | 1.000 | basis_diag_gram | full_sobolev_gram | tfu_task_diag | tfu_task_diag | 6918.15 | 23.02 |
| KMNIST | D3-taskSobTask | 1.0 | 0 | 0 | 0 | 1.000 | tfu_data_task_diag | full_sobolev_gram | full_sobolev_gram | tfu_task_diag | 5.69 | 23.53 |
| KMNIST | D6-allTaskAware | 1.0 | 0 | 0 | 0 | 1.000 | tfu_data_task_diag | tfu_data_task_diag | tfu_task_diag | tfu_task_diag | 5.69 | 23.59 |
| KMNIST | D7-inputOutputTask-middleSob | 1.0 | 0 | 0 | 0 | 1.000 | tfu_data_task_diag | full_sobolev_gram | full_sobolev_gram | tfu_task_diag | 5.69 | 23.53 |
| MNIST | D0-allFullSobolev | 1.0 | 0 | 0 | 0 | 1.000 | full_sobolev_gram |  |  | full_sobolev_gram | 6551.62 | 6551.62 |
| MNIST | D1-frontTask-backSob | 1.0 | 0 | 0 | 0 | 1.000 | tfu_data_task_diag | tfu_data_task_diag | full_sobolev_gram | full_sobolev_gram | 8.89 | 6551.62 |
| MNIST | D2-frontSob-backTask | 1.0 | 0 | 0 | 0 | 1.000 | basis_diag_gram | full_sobolev_gram | tfu_task_diag | tfu_task_diag | 6918.15 | 16.55 |
| MNIST | D3-taskSobTask | 1.0 | 0 | 0 | 0 | 1.000 | tfu_data_task_diag | full_sobolev_gram | full_sobolev_gram | tfu_task_diag | 8.89 | 16.10 |
| MNIST | D6-allTaskAware | 1.0 | 0 | 0 | 0 | 1.000 | tfu_data_task_diag | tfu_data_task_diag | tfu_task_diag | tfu_task_diag | 8.88 | 16.05 |
| MNIST | D7-inputOutputTask-middleSob | 1.0 | 0 | 0 | 0 | 1.000 | tfu_data_task_diag | full_sobolev_gram | full_sobolev_gram | tfu_task_diag | 8.89 | 16.10 |

P0 verdict: pass. TFU rows have pure alpha fixed at 1.0, no trainable non-KAN parameters, no linear layer, full coefficient coverage, and finite role metric conditions.

## P1 One-Batch Direction Audit

| dataset | method | train descent | val descent | bad | input cos | shallow cos | deep cos | output cos |
|---|---|---|---|---|---|---|---|---|
| MNIST | AdamW-one-step | 0.0229 | 0.0481 | 0 |  |  |  |  |
| MNIST | D0-allFullSobolev | -0.0408 | 0.0414 | 1 | 0.1303 |  |  | 0.2904 |
| MNIST | D1-frontTask-backSob | 0.0215 | 0.0901 | 0 | 0.8348 | 0.9327 | 0.2977 | 0.2904 |
| MNIST | D2-frontSob-backTask | -0.0052 | 0.0487 | 1 | 0.9915 | 0.2958 | 0.9629 | 0.9191 |
| MNIST | D3-taskSobTask | 0.0232 | 0.0908 | 0 | 0.8348 | 0.2958 | 0.2977 | 0.9191 |
| MNIST | D6-allTaskAware | 0.0222 | 0.0901 | 0 | 0.8348 | 0.9327 | 0.9629 | 0.9191 |
| MNIST | D7-inputOutputTask-middleSob | 0.0232 | 0.0908 | 0 | 0.8348 | 0.2958 | 0.2977 | 0.9191 |
| MNIST | D8-inputDiag-blockTask-outputFisher | -0.0053 | 0.0493 | 1 | 0.9915 | 0.9034 | 0.9629 | 0.9191 |
| Fashion-MNIST | AdamW-one-step | 0.0517 | 0.0078 | 0 |  |  |  |  |
| Fashion-MNIST | D0-allFullSobolev | 0.0125 | -0.0397 | 0 | 0.5445 |  |  | 0.2936 |
| Fashion-MNIST | D1-frontTask-backSob | 0.0627 | 0.0243 | 0 | 0.8944 | 0.9435 | 0.3206 | 0.2936 |
| Fashion-MNIST | D2-frontSob-backTask | 0.0024 | -0.0522 | 0 | 0.9994 | 0.3124 | 0.9652 | 0.9166 |
| Fashion-MNIST | D3-taskSobTask | 0.0657 | 0.0275 | 0 | 0.8944 | 0.3124 | 0.3206 | 0.9166 |
| Fashion-MNIST | D6-allTaskAware | 0.0638 | 0.0250 | 0 | 0.8944 | 0.9435 | 0.9652 | 0.9166 |
| Fashion-MNIST | D7-inputOutputTask-middleSob | 0.0657 | 0.0275 | 0 | 0.8944 | 0.3124 | 0.3206 | 0.9166 |
| Fashion-MNIST | D8-inputDiag-blockTask-outputFisher | 0.0025 | -0.0526 | 0 | 0.9994 | 0.9118 | 0.9652 | 0.9166 |
| KMNIST | AdamW-one-step | 0.0646 | 0.0501 | 0 |  |  |  |  |
| KMNIST | D0-allFullSobolev | -0.0021 | 0.0031 | 1 | 0.3084 |  |  | 0.2896 |
| KMNIST | D1-frontTask-backSob | 0.0438 | 0.0545 | 0 | 0.8771 | 0.9413 | 0.2847 | 0.2896 |
| KMNIST | D2-frontSob-backTask | 0.0147 | 0.0054 | 0 | 0.9908 | 0.3044 | 0.9672 | 0.9271 |
| KMNIST | D3-taskSobTask | 0.0434 | 0.0548 | 0 | 0.8771 | 0.3044 | 0.2847 | 0.9271 |
| KMNIST | D6-allTaskAware | 0.0458 | 0.0551 | 0 | 0.8771 | 0.9413 | 0.9672 | 0.9271 |
| KMNIST | D7-inputOutputTask-middleSob | 0.0434 | 0.0548 | 0 | 0.8771 | 0.3044 | 0.2847 | 0.9271 |
| KMNIST | D8-inputDiag-blockTask-outputFisher | 0.0149 | 0.0048 | 0 | 0.9908 | 0.9098 | 0.9672 | 0.9271 |

P1 verdict: strong positive direction signal. D0 full Sobolev remains a bad train step on MNIST/KMNIST, while D6 GlobalTFU has positive train and validation descent on all three datasets.

## P2 Global TFU Micro-Run

| dataset | method | runs | acc | std | gap vs AdamW | acc minus D0 | AUC imp vs D0 | AUC imp vs AdamW | ECE red | phi p95 | J | safeguard fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MNIST | AdamW | 3 | 0.9303 | 0.0076 | 0.0000 | 0.0433 | 0.4541 | 0.0000 | 0.0000 | 0.2274 | 226275.4421 | 0.0000 |
| MNIST | D0-allFullSobolev | 3 | 0.8870 | 0.0455 | 0.0433 | 0.0000 | 0.0000 | -0.8318 | -0.3594 | 0.1489 | 34.4779 | 0.0000 |
| MNIST | D6-allTaskAware | 3 | 0.8620 | 0.0708 | 0.0683 | -0.0250 | 0.0510 | -0.7384 | 0.0949 | 0.1508 | 32.8464 | 0.0000 |
| MNIST | D6-noSob | 3 | 0.8837 | 0.0395 | 0.0467 | -0.0033 | 0.0730 | -0.6980 | 0.3943 | 0.1507 | 36.8407 | 0.0000 |
| MNIST | D6-lowSob | 3 | 0.8910 | 0.0352 | 0.0393 | 0.0040 | 0.0692 | -0.7051 | 0.1009 | 0.1507 | 45.4414 | 0.0000 |
| MNIST | D6-withSafeguard | 3 | 0.8620 | 0.0708 | 0.0683 | -0.0250 | 0.0510 | -0.7384 | 0.0949 | 0.1508 | 32.8464 | 0.0000 |
| MNIST | Hybrid-DGKAN-UFULL-f085 | 3 | 0.9373 | 0.0068 | -0.0070 | 0.0503 | 0.6148 | 0.2944 | 0.1910 | 0.1488 | 10.9172 | 0.0000 |
| MNIST | MLP-AdamW | 3 | 0.9533 | 0.0046 | -0.0230 | 0.0663 | 0.6697 | 0.3949 | 0.3670 |  |  | 0.0000 |
| Fashion-MNIST | AdamW | 3 | 0.8563 | 0.0090 | 0.0000 | 0.0243 | 0.0602 | 0.0000 | 0.0000 | 0.2396 | 369574.0127 | 0.0000 |
| Fashion-MNIST | D0-allFullSobolev | 3 | 0.8320 | 0.0120 | 0.0243 | 0.0000 | 0.0000 | -0.0640 | 0.6596 | 0.1490 | 31.4545 | 0.0000 |
| Fashion-MNIST | D6-allTaskAware | 3 | 0.8303 | 0.0358 | 0.0260 | -0.0017 | 0.0971 | 0.0393 | 0.3910 | 0.1506 | 31.2786 | 0.0000 |
| Fashion-MNIST | D6-noSob | 3 | 0.8483 | 0.0048 | 0.0080 | 0.0163 | 0.1166 | 0.0601 | 0.4310 | 0.1507 | 31.1504 | 0.0000 |
| Fashion-MNIST | D6-lowSob | 3 | 0.8473 | 0.0101 | 0.0090 | 0.0153 | 0.1161 | 0.0595 | 0.4507 | 0.1508 | 38.3286 | 0.0000 |
| Fashion-MNIST | D6-withSafeguard | 3 | 0.8303 | 0.0358 | 0.0260 | -0.0017 | 0.0971 | 0.0393 | 0.3910 | 0.1506 | 31.2786 | 0.0000 |
| Fashion-MNIST | Hybrid-DGKAN-UFULL-f085 | 3 | 0.8490 | 0.0116 | 0.0073 | 0.0170 | 0.2180 | 0.1679 | 0.2075 | 0.1488 | 10.2621 | 0.0000 |
| Fashion-MNIST | MLP-AdamW | 3 | 0.8500 | 0.0137 | 0.0063 | 0.0180 | 0.2179 | 0.1678 | 0.1011 |  |  | 0.0000 |
| KMNIST | AdamW | 3 | 0.7720 | 0.0102 | 0.0000 | 0.0937 | 0.4381 | 0.0000 | 0.0000 | 0.2172 | 20374.7067 | 0.0000 |
| KMNIST | D0-allFullSobolev | 3 | 0.6783 | 0.0147 | 0.0937 | 0.0000 | 0.0000 | -0.7797 | 0.7098 | 0.1493 | 35.5405 | 0.0000 |
| KMNIST | D6-allTaskAware | 3 | 0.7500 | 0.0134 | 0.0220 | 0.0717 | 0.1873 | -0.4464 | 0.7258 | 0.1512 | 32.7251 | 0.0000 |
| KMNIST | D6-noSob | 3 | 0.7333 | 0.0175 | 0.0387 | 0.0550 | 0.1604 | -0.4943 | 0.6471 | 0.1511 | 51.3783 | 0.0000 |
| KMNIST | D6-lowSob | 3 | 0.7200 | 0.0051 | 0.0520 | 0.0417 | 0.1601 | -0.4948 | 0.6547 | 0.1511 | 42.9699 | 0.0000 |
| KMNIST | D6-withSafeguard | 3 | 0.7500 | 0.0134 | 0.0220 | 0.0717 | 0.1873 | -0.4464 | 0.7258 | 0.1512 | 32.7251 | 0.0000 |
| KMNIST | Hybrid-DGKAN-UFULL-f085 | 3 | 0.7667 | 0.0139 | 0.0053 | 0.0883 | 0.4753 | 0.0662 | 0.0897 | 0.1485 | 10.2174 | 0.0000 |
| KMNIST | MLP-AdamW | 3 | 0.7920 | 0.0120 | -0.0200 | 0.1137 | 0.5531 | 0.2046 | 0.0269 |  |  | 0.0000 |

P2 verdict: D6 does not pass the global gate. It repairs one-batch direction and improves KMNIST substantially over D0, but the 3-seed accuracy gap vs PureKAN-AdamW is too large on MNIST and KMNIST. Fashion only passes the accuracy side for `D6-noSob` / `D6-lowSob`, not the unified D6 profile.

## P3 Depth-Wise Sweep

| dataset | method | runs | acc | std | gap vs AdamW | acc minus D0 | AUC imp vs D0 | AUC imp vs AdamW | ECE red | phi p95 | J | safeguard fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MNIST | AdamW | 3 | 0.9303 | 0.0076 | 0.0000 | 0.0433 | 0.4541 | 0.0000 | 0.0000 | 0.2274 | 226275.4421 | 0.0000 |
| MNIST | D0-allFullSobolev | 3 | 0.8870 | 0.0455 | 0.0433 | 0.0000 | 0.0000 | -0.8318 | -0.3594 | 0.1489 | 34.4779 | 0.0000 |
| MNIST | D6-allTaskAware | 3 | 0.8620 | 0.0708 | 0.0683 | -0.0250 | 0.0510 | -0.7384 | 0.0949 | 0.1508 | 32.8464 | 0.0000 |
| MNIST | D1-frontTask-backSob | 3 | 0.8483 | 0.0878 | 0.0820 | -0.0387 | -0.0356 | -0.8970 | -0.0305 | 0.1508 | 39.1325 | 0.0000 |
| MNIST | D2-frontSob-backTask | 3 | 0.8877 | 0.0237 | 0.0427 | 0.0007 | -0.0355 | -0.8969 | 0.0268 | 0.1492 | 26.0784 | 0.0000 |
| MNIST | D3-taskSobTask | 3 | 0.9080 | 0.0037 | 0.0223 | 0.0210 | 0.0672 | -0.7086 | 0.0759 | 0.1493 | 31.8315 | 0.0000 |
| MNIST | D7-inputOutputTask-middleSob | 3 | 0.9080 | 0.0037 | 0.0223 | 0.0210 | 0.0672 | -0.7086 | 0.0759 | 0.1493 | 31.8315 | 0.0000 |
| MNIST | D8-inputDiag-blockTask-outputFisher | 3 | 0.8883 | 0.0109 | 0.0420 | 0.0013 | -0.0230 | -0.8739 | 0.1354 | 0.1495 | 28.8581 | 0.0000 |
| Fashion-MNIST | AdamW | 3 | 0.8563 | 0.0090 | 0.0000 | 0.0243 | 0.0602 | 0.0000 | 0.0000 | 0.2396 | 369574.0127 | 0.0000 |
| Fashion-MNIST | D0-allFullSobolev | 3 | 0.8320 | 0.0120 | 0.0243 | 0.0000 | 0.0000 | -0.0640 | 0.6596 | 0.1490 | 31.4545 | 0.0000 |
| Fashion-MNIST | D6-allTaskAware | 3 | 0.8303 | 0.0358 | 0.0260 | -0.0017 | 0.0971 | 0.0393 | 0.3910 | 0.1506 | 31.2786 | 0.0000 |
| Fashion-MNIST | D1-frontTask-backSob | 3 | 0.8447 | 0.0140 | 0.0117 | 0.0127 | 0.0642 | 0.0043 | 0.5762 | 0.1505 | 47.8118 | 0.0000 |
| Fashion-MNIST | D2-frontSob-backTask | 3 | 0.8393 | 0.0107 | 0.0170 | 0.0073 | 0.0822 | 0.0235 | 0.5151 | 0.1496 | 29.9848 | 0.0000 |
| Fashion-MNIST | D3-taskSobTask | 3 | 0.8490 | 0.0079 | 0.0073 | 0.0170 | 0.1170 | 0.0604 | 0.5534 | 0.1492 | 40.1297 | 0.0000 |
| Fashion-MNIST | D7-inputOutputTask-middleSob | 3 | 0.8490 | 0.0079 | 0.0073 | 0.0170 | 0.1170 | 0.0604 | 0.5534 | 0.1492 | 40.1297 | 0.0000 |
| Fashion-MNIST | D8-inputDiag-blockTask-outputFisher | 3 | 0.8347 | 0.0107 | 0.0217 | 0.0027 | 0.0758 | 0.0167 | 0.4926 | 0.1500 | 32.1401 | 0.0000 |
| KMNIST | AdamW | 3 | 0.7720 | 0.0102 | 0.0000 | 0.0937 | 0.4381 | 0.0000 | 0.0000 | 0.2172 | 20374.7067 | 0.0000 |
| KMNIST | D0-allFullSobolev | 3 | 0.6783 | 0.0147 | 0.0937 | 0.0000 | 0.0000 | -0.7797 | 0.7098 | 0.1493 | 35.5405 | 0.0000 |
| KMNIST | D6-allTaskAware | 3 | 0.7500 | 0.0134 | 0.0220 | 0.0717 | 0.1873 | -0.4464 | 0.7258 | 0.1512 | 32.7251 | 0.0000 |
| KMNIST | D1-frontTask-backSob | 3 | 0.7003 | 0.0554 | 0.0717 | 0.0220 | 0.0919 | -0.6162 | 0.6356 | 0.1511 | 32.1348 | 0.0000 |
| KMNIST | D2-frontSob-backTask | 3 | 0.7250 | 0.0208 | 0.0470 | 0.0467 | 0.1020 | -0.5982 | 0.7363 | 0.1497 | 29.6158 | 0.0000 |
| KMNIST | D3-taskSobTask | 3 | 0.7107 | 0.0253 | 0.0613 | 0.0323 | 0.1457 | -0.5204 | 0.6758 | 0.1495 | 31.4745 | 0.0000 |
| KMNIST | D7-inputOutputTask-middleSob | 3 | 0.7107 | 0.0253 | 0.0613 | 0.0323 | 0.1457 | -0.5204 | 0.6758 | 0.1495 | 31.4745 | 0.0000 |
| KMNIST | D8-inputDiag-blockTask-outputFisher | 3 | 0.7363 | 0.0212 | 0.0357 | 0.0580 | 0.1321 | -0.5446 | 0.7372 | 0.1501 | 33.4249 | 0.0000 |

### P3 Gate Check

| method | max gap vs AdamW | min AUC imp vs D0 | improves D0 acc all | enter P4 |
|---|---|---|---|---|
| D6-allTaskAware | 0.0683 | 0.0510 | no | no |
| D1-frontTask-backSob | 0.0820 | -0.0356 | no | no |
| D2-frontSob-backTask | 0.0470 | -0.0355 | yes | no |
| D3-taskSobTask | 0.0613 | 0.0672 | yes | no |
| D7-inputOutputTask-middleSob | 0.0613 | 0.0672 | yes | no |
| D8-inputDiag-blockTask-outputFisher | 0.0420 | -0.0230 | yes | no |

P3 verdict: no method enters P4. `D3`/`D7` are best for Fashion and improve MNIST over D0, while `D8` is comparatively best on KMNIST among depth-wise variants. None satisfy the joint gate because MNIST/KMNIST still remain more than 1-2 points behind PureKAN-AdamW and AUC improvement vs D0 does not reach the required 20% on all datasets.

## Final Decision

```text
P0:
  Implementation smoke passed.

P1:
  GlobalTFU fixes the immediate direction failure seen in v3.7 one-batch audits.

P2:
  D6 GlobalTFU is not accepted as a full PureKAN optimizer.
  It improves direction and calibration/geometry but does not match PureKAN-AdamW accuracy.

P3:
  Depth-wise task placement helps identify useful structure:
    D3/D7: best Fashion and better MNIST than D0.
    D8: best KMNIST depth-wise candidate.
  No candidate reaches the P4 entry gate.

P4-P7:
  Not run by plan because P3 produced no qualifying candidate.

Final:
  v3.8 confirms that the v3.7 bottleneck is partly metric direction quality, not implementation coverage.
  Task-aware diagonal metrics are a real repair for local descent, but global training still underfits or becomes seed-sensitive.
  Next work should focus on layer-local/block-local objective design or adaptive per-role LR schedules before spending 5/10-seed confirmation budget.
```
