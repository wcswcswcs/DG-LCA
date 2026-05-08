# DG-KAN v3.9 Normalization / FNG PureKAN 结果复盘

本轮依据 `docs/DG-KAN_v3.9_Normalization_FNG_PureKAN_实验计划.md`。目标是判断 PureKAN functional update 追不上 PureKAN-AdamW 的主因，是缺少 learnable normalization，还是缺少真正 task/Fisher-aware 的 function-space metric。

## Code / Config Changes

```text
experiments/dgkan_core.py
  Added PureKAN norm modes: fixed / affine_channel / scalar_gain / none.
  Added functional diagonal norm update for gamma/beta or scalar gain/bias.
  Added FNG/KFAC-style coefficient update metrics:
    fng_right
    fng_leftdiag_right
    fng_leftfull_right
    fng_leftlowrank_right
  Result rows now include norm audit, functional coverage, FNG metric conditions,
  FNG fallback/bad-step rates, timing, and per-role metric traces.

experiments/run_gafu_v39.py
  Added P0 smoke, P1 shadow audit, P2 normalization ablation,
  P3 FNG package, and optional P4 dynamics wiring.

experiments/analyze_gafu_v39.py
  Generates this replay, appends the log entry, and writes basic dashboards.
```

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 smoke | 27 | 0 |
| P1 shadow | 30 | 0 |
| P2 norm | 99 | 0 |
| P3 relaxed | 54 | 0 |

## P0 Implementation Smoke

| dataset | method | norm | norm update | nonKAN | raw nonKAN | func cov | input coeff | block coeff | output coeff | norm params | norm cov | FNG | FNG cond | fallback |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | AffineNorm-AdamW-LN | affine_channel | adamw | 192 | 192 | 0.000 | 0.000 | 0.000 | 0.000 | 192 | 0.000 | none |  | 0.000 |
| Fashion-MNIST | AffineNorm-FDiag-LN | affine_channel | fdiag | 0 | 192 | 1.000 | 1.000 | 1.000 | 1.000 | 192 | 1.000 | none |  | 0.000 |
| Fashion-MNIST | FixedNorm-AdamW | fixed | none | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0 |  | none |  | 0.000 |
| Fashion-MNIST | FixedNorm-D0-allFullSobolev | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | none |  | 0.000 |
| Fashion-MNIST | FixedNorm-D6-allTaskAware | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | none |  | 0.000 |
| Fashion-MNIST | FixedNorm-FNG-leftDiagRight | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | leftdiag | 1959.61 | 0.000 |
| Fashion-MNIST | FixedNorm-FNG-leftFullRight | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | leftfull | 1959.72 | 0.000 |
| Fashion-MNIST | FixedNorm-FNG-right | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | right | 1959.76 | 0.000 |
| Fashion-MNIST | ScalarGainNorm-FDiag | scalar_gain | fdiag | 0 | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 6 | 1.000 | none |  | 0.000 |
| KMNIST | AffineNorm-AdamW-LN | affine_channel | adamw | 192 | 192 | 0.000 | 0.000 | 0.000 | 0.000 | 192 | 0.000 | none |  | 0.000 |
| KMNIST | AffineNorm-FDiag-LN | affine_channel | fdiag | 0 | 192 | 1.000 | 1.000 | 1.000 | 1.000 | 192 | 1.000 | none |  | 0.000 |
| KMNIST | FixedNorm-AdamW | fixed | none | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0 |  | none |  | 0.000 |
| KMNIST | FixedNorm-D0-allFullSobolev | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | none |  | 0.000 |
| KMNIST | FixedNorm-D6-allTaskAware | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | none |  | 0.000 |
| KMNIST | FixedNorm-FNG-leftDiagRight | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | leftdiag | 2119.50 | 0.000 |
| KMNIST | FixedNorm-FNG-leftFullRight | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | leftfull | 2119.50 | 0.000 |
| KMNIST | FixedNorm-FNG-right | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | right | 2119.50 | 0.000 |
| KMNIST | ScalarGainNorm-FDiag | scalar_gain | fdiag | 0 | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 6 | 1.000 | none |  | 0.000 |
| MNIST | AffineNorm-AdamW-LN | affine_channel | adamw | 192 | 192 | 0.000 | 0.000 | 0.000 | 0.000 | 192 | 0.000 | none |  | 0.000 |
| MNIST | AffineNorm-FDiag-LN | affine_channel | fdiag | 0 | 192 | 1.000 | 1.000 | 1.000 | 1.000 | 192 | 1.000 | none |  | 0.000 |
| MNIST | FixedNorm-AdamW | fixed | none | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0 |  | none |  | 0.000 |
| MNIST | FixedNorm-D0-allFullSobolev | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | none |  | 0.000 |
| MNIST | FixedNorm-D6-allTaskAware | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | none |  | 0.000 |
| MNIST | FixedNorm-FNG-leftDiagRight | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | leftdiag | 2263.79 | 0.000 |
| MNIST | FixedNorm-FNG-leftFullRight | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | leftfull | 2263.79 | 0.000 |
| MNIST | FixedNorm-FNG-right | fixed | none | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |  | right | 2263.79 | 0.000 |
| MNIST | ScalarGainNorm-FDiag | scalar_gain | fdiag | 0 | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 6 | 1.000 | none |  | 0.000 |

P0 verdict: pass. Strict PureKAN functional variants have zero learnable non-KAN parameters and full coefficient coverage. Diagnostic `AffineNorm-AdamW-LN` correctly exposes learnable non-KAN norm parameters and is not eligible for final PureKAN claims.

## P1 One-Batch Direction Audit

| dataset | method | train descent | val descent | bad | mean cos | max cond | norm | norm update |
|---|---|---|---|---|---|---|---|---|
| MNIST | AdamW-one-step | 0.0229 | 0.0481 | 0 |  |  | fixed | none |
| MNIST | D0-allFullSobolev | -0.0284 | 0.0138 | 1 | 0.2488 |  | fixed | none |
| MNIST | D6-allTaskAware | -0.0137 | 0.0278 | 1 | 0.9032 |  | fixed | none |
| MNIST | F2-FNG-right-only | 0.0021 | -0.0228 | 0 | 0.3000 | 2115.5 | fixed | none |
| MNIST | F3-FNG-leftDiag-right | -0.0070 | -0.0113 | 1 | 0.3056 | 2115.5 | fixed | none |
| MNIST | F4-FNG-leftFull-right | -0.0060 | 0.0051 | 1 | 0.2784 | 2115.5 | fixed | none |
| MNIST | F4-FNG-leftFull-right+AffineNorm-AdamW-LN | -0.0002 | -0.0362 | 1 | 0.2738 | 2115.5 | affine_channel | adamw |
| MNIST | F4-FNG-leftFull-right+AffineNorm-FDiag | 0.0569 | -0.0004 | 0 | 0.2742 | 2115.5 | affine_channel | fdiag |
| MNIST | F4-FNG-leftFull-right-lowSob | 0.0729 | 0.0298 | 0 | 0.2786 | 2066.3 | fixed | none |
| MNIST | F4-FNG-leftFull-right-noSob | 0.0007 | -0.0128 | 0 | 0.2189 | 2051.0 | fixed | none |
| Fashion-MNIST | AdamW-one-step | 0.0517 | 0.0078 | 0 |  |  | fixed | none |
| Fashion-MNIST | D0-allFullSobolev | 0.0404 | 0.0315 | 0 | 0.3946 |  | fixed | none |
| Fashion-MNIST | D6-allTaskAware | 0.0075 | 0.0169 | 0 | 0.9180 |  | fixed | none |
| Fashion-MNIST | F2-FNG-right-only | 0.0126 | 0.0110 | 0 | 0.3665 | 1445.5 | fixed | none |
| Fashion-MNIST | F3-FNG-leftDiag-right | 0.0979 | 0.1009 | 0 | 0.3653 | 1445.5 | fixed | none |
| Fashion-MNIST | F4-FNG-leftFull-right | 0.0906 | 0.0373 | 0 | 0.3313 | 1445.5 | fixed | none |
| Fashion-MNIST | F4-FNG-leftFull-right+AffineNorm-AdamW-LN | 0.0095 | 0.0163 | 0 | 0.3197 | 1445.5 | affine_channel | adamw |
| Fashion-MNIST | F4-FNG-leftFull-right+AffineNorm-FDiag | 0.0486 | -0.0122 | 0 | 0.3316 | 1445.5 | affine_channel | fdiag |
| Fashion-MNIST | F4-FNG-leftFull-right-lowSob | 0.0540 | -0.0628 | 0 | 0.2775 | 1395.8 | fixed | none |
| Fashion-MNIST | F4-FNG-leftFull-right-noSob | 0.0321 | 0.0366 | 0 | 0.2255 | 1379.3 | fixed | none |
| KMNIST | AdamW-one-step | 0.0646 | 0.0501 | 0 |  |  | fixed | none |
| KMNIST | D0-allFullSobolev | 0.0305 | -0.0270 | 0 | 0.3086 |  | fixed | none |
| KMNIST | D6-allTaskAware | 0.0312 | -0.0280 | 0 | 0.9172 |  | fixed | none |
| KMNIST | F2-FNG-right-only | 0.0697 | 0.0385 | 0 | 0.3642 | 1885.7 | fixed | none |
| KMNIST | F3-FNG-leftDiag-right | 0.0110 | 0.0232 | 0 | 0.3608 | 1885.7 | fixed | none |
| KMNIST | F4-FNG-leftFull-right | 0.0724 | 0.0953 | 0 | 0.3321 | 1885.7 | fixed | none |
| KMNIST | F4-FNG-leftFull-right+AffineNorm-AdamW-LN | 0.0160 | 0.0204 | 0 | 0.3165 | 1885.7 | affine_channel | adamw |
| KMNIST | F4-FNG-leftFull-right+AffineNorm-FDiag | -0.0114 | 0.0088 | 1 | 0.3259 | 1885.7 | affine_channel | fdiag |
| KMNIST | F4-FNG-leftFull-right-lowSob | 0.0472 | -0.0386 | 0 | 0.2780 | 1836.3 | fixed | none |
| KMNIST | F4-FNG-leftFull-right-noSob | 0.0338 | 0.0536 | 0 | 0.2210 | 1820.1 | fixed | none |

### P1 Gate Check

| method | bad | min train | val >=0 | mean cos | max cond | P1 pass |
|---|---|---|---|---|---|---|
| D0-allFullSobolev | 1 | -0.0284 | 2/3 | 0.3174 |  | no |
| D6-allTaskAware | 1 | -0.0137 | 2/3 | 0.9128 |  | no |
| F2-FNG-right-only | 0 | 0.0021 | 2/3 | 0.3436 | 2115.5 | no |
| F3-FNG-leftDiag-right | 1 | -0.0070 | 2/3 | 0.3439 | 2115.5 | no |
| F4-FNG-leftFull-right | 1 | -0.0060 | 3/3 | 0.3139 | 2115.5 | no |
| F4-FNG-leftFull-right+AffineNorm-AdamW-LN | 1 | -0.0002 | 2/3 | 0.3034 | 2115.5 | no |
| F4-FNG-leftFull-right+AffineNorm-FDiag | 1 | -0.0114 | 1/3 | 0.3105 | 2115.5 | no |
| F4-FNG-leftFull-right-lowSob | 0 | 0.0472 | 1/3 | 0.2781 | 2066.3 | no |
| F4-FNG-leftFull-right-noSob | 0 | 0.0007 | 2/3 | 0.2218 | 2051.0 | no |

P1 verdict: strict fail. FNG variants repair several bad-step cases compared with D0/D6, but no candidate satisfies the full direction gate because the mean raw/preconditioned cosine remains below `0.5`. Therefore no method formally enters P3/P4 under the written plan.

## P2 Normalization Ablation

| dataset | method | runs | acc | std | gap vs AdamW | acc-D0 | AUC imp vs D6 | AUC imp vs AdamW | ECE red | phi p95 | J | FNG cond | bad | fallback | nonKAN | norm cov | time/A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MNIST | FixedNorm-AdamW | 3 | 0.9303 | 0.0076 | 0.0000 |  | 0.4248 | 0.0000 | 0.0000 | 0.2274 | 226275.4421 |  | 0.0000 | 0.0000 | 0 |  | 1.0000 |
| MNIST | AffineNorm-AdamW-LN | 3 | 0.9297 | 0.0090 | 0.0007 |  | 0.4247 | -0.0002 | -0.0171 | 0.2291 | 27892.0690 |  | 0.0000 | 0.0000 | 960 | 0.000 | 1.0952 |
| MNIST | AffineNorm-FDiag-LN | 3 | 0.9320 | 0.0100 | -0.0017 |  | 0.4427 | 0.0312 | -0.0051 | 0.2101 | 48074.1139 |  | 0.0000 | 0.0000 | 0 | 1.000 | 1.1995 |
| MNIST | ScalarGainNorm-FDiag | 3 | 0.9320 | 0.0075 | -0.0017 |  | 0.4226 | -0.0038 | -0.0102 | 0.2189 | 59139.8268 |  | 0.0000 | 0.0000 | 0 | 1.000 | 1.1791 |
| MNIST | NoNorm-AdamW | 3 | 0.8397 | 0.0152 | 0.0907 |  | -0.5936 | -1.7703 | -2.4315 | 0.0719 | 9.0234 |  | 0.0000 | 0.0000 | 0 |  | 1.1864 |
| MNIST | FixedNorm-D6-allTaskAware | 3 | 0.8620 | 0.0708 | 0.0683 |  | 0.0000 | -0.7384 | 0.0949 | 0.1508 | 32.8464 |  | 0.0000 | 0.0000 | 0 |  | 1.2855 |
| MNIST | AffineNorm-AdamW-LN-D6 | 3 | 0.8647 | 0.0811 | 0.0657 |  | 0.0041 | -0.7313 | 0.0852 | 0.1506 | 37.5073 |  | 0.0000 | 0.0000 | 960 | 0.000 | 1.3283 |
| MNIST | AffineNorm-FDiag-LN-D6 | 3 | 0.8493 | 0.0839 | 0.0810 |  | 0.0053 | -0.7292 | 0.2002 | 0.1496 | 32.1558 |  | 0.0000 | 0.0000 | 0 | 1.000 | 1.4102 |
| MNIST | FixedNorm-FNG-leftFullRight | 3 | 0.9233 | 0.0087 | 0.0070 |  | 0.4337 | 0.0155 | 0.5400 | 0.1486 | 30.6556 | 2119.3 | 0.0000 | 0.0000 | 0 |  | 1.5111 |
| MNIST | AffineNorm-AdamW-LN-FNG-leftFullRight | 3 | 0.9257 | 0.0090 | 0.0047 |  | 0.4250 | 0.0005 | 0.5529 | 0.1488 | 31.6827 | 2119.3 | 0.0000 | 0.0000 | 960 | 0.000 | 1.6108 |
| MNIST | AffineNorm-FDiag-LN-FNG-leftFullRight | 3 | 0.9273 | 0.0094 | 0.0030 |  | 0.3990 | -0.0447 | 0.5127 | 0.1482 | 27.3618 | 2119.3 | 0.0000 | 0.0000 | 0 | 1.000 | 1.6310 |
| Fashion-MNIST | FixedNorm-AdamW | 3 | 0.8563 | 0.0090 | 0.0000 |  | -0.0409 | 0.0000 | 0.0000 | 0.2396 | 369574.0127 |  | 0.0000 | 0.0000 | 0 |  | 1.0000 |
| Fashion-MNIST | AffineNorm-AdamW-LN | 3 | 0.8467 | 0.0041 | 0.0097 |  | -0.0571 | -0.0156 | 0.0010 | 0.2454 | 63761.4883 |  | 0.0000 | 0.0000 | 960 | 0.000 | 1.0704 |
| Fashion-MNIST | AffineNorm-FDiag-LN | 3 | 0.8480 | 0.0182 | 0.0083 |  | -0.1331 | -0.0885 | -0.0651 | 0.2349 | 55643.2305 |  | 0.0000 | 0.0000 | 0 | 1.000 | 1.2243 |
| Fashion-MNIST | ScalarGainNorm-FDiag | 3 | 0.8387 | 0.0236 | 0.0177 |  | -0.0815 | -0.0389 | -0.1034 | 0.2392 | 358548.6582 |  | 0.0000 | 0.0000 | 0 | 1.000 | 1.1656 |
| Fashion-MNIST | NoNorm-AdamW | 3 | 0.8007 | 0.0161 | 0.0557 |  | -0.4457 | -0.3888 | 0.3523 | 0.1031 | 10.8600 |  | 0.0000 | 0.0000 | 0 |  | 1.1346 |
| Fashion-MNIST | FixedNorm-D6-allTaskAware | 3 | 0.8303 | 0.0358 | 0.0260 |  | 0.0000 | 0.0393 | 0.3910 | 0.1506 | 31.2786 |  | 0.0000 | 0.0000 | 0 |  | 1.2882 |
| Fashion-MNIST | AffineNorm-AdamW-LN-D6 | 3 | 0.8467 | 0.0103 | 0.0097 |  | 0.0033 | 0.0424 | 0.4203 | 0.1505 | 32.0193 |  | 0.0000 | 0.0000 | 960 | 0.000 | 1.3238 |
| Fashion-MNIST | AffineNorm-FDiag-LN-D6 | 3 | 0.8373 | 0.0148 | 0.0190 |  | 0.0170 | 0.0557 | 0.3893 | 0.1497 | 28.6478 |  | 0.0000 | 0.0000 | 0 | 1.000 | 1.4019 |
| Fashion-MNIST | FixedNorm-FNG-leftFullRight | 3 | 0.8327 | 0.0068 | 0.0237 |  | 0.0467 | 0.0842 | 0.2462 | 0.1485 | 30.3611 | 1460.5 | 0.0000 | 0.0000 | 0 |  | 1.4663 |
| Fashion-MNIST | AffineNorm-AdamW-LN-FNG-leftFullRight | 3 | 0.8337 | 0.0097 | 0.0227 |  | 0.0474 | 0.0848 | 0.2462 | 0.1485 | 26.9015 | 1461.1 | 0.0000 | 0.0000 | 960 | 0.000 | 1.5405 |
| Fashion-MNIST | AffineNorm-FDiag-LN-FNG-leftFullRight | 3 | 0.8360 | 0.0127 | 0.0203 |  | 0.0372 | 0.0751 | 0.2063 | 0.1480 | 28.4187 | 1460.5 | 0.0000 | 0.0000 | 0 | 1.000 | 1.5994 |
| KMNIST | FixedNorm-AdamW | 3 | 0.7720 | 0.0102 | 0.0000 |  | 0.3087 | 0.0000 | 0.0000 | 0.2172 | 20374.7067 |  | 0.0000 | 0.0000 | 0 |  | 1.0000 |
| KMNIST | AffineNorm-AdamW-LN | 3 | 0.7733 | 0.0082 | -0.0013 |  | 0.3115 | 0.0042 | 0.0195 | 0.2160 | 80444.0033 |  | 0.0000 | 0.0000 | 960 | 0.000 | 1.0700 |
| KMNIST | AffineNorm-FDiag-LN | 3 | 0.7703 | 0.0200 | 0.0017 |  | 0.3196 | 0.0158 | -0.0019 | 0.2070 | 19478.2301 |  | 0.0000 | 0.0000 | 0 | 1.000 | 1.1891 |
| KMNIST | ScalarGainNorm-FDiag | 3 | 0.7730 | 0.0151 | -0.0010 |  | 0.3066 | -0.0030 | 0.0147 | 0.2166 | 46882.1016 |  | 0.0000 | 0.0000 | 0 | 1.000 | 1.1472 |
| KMNIST | NoNorm-AdamW | 3 | 0.6027 | 0.0095 | 0.1693 |  | -0.4709 | -1.1276 | 0.6468 | 0.0968 | 11.2521 |  | 0.0000 | 0.0000 | 0 |  | 1.1478 |
| KMNIST | FixedNorm-D6-allTaskAware | 3 | 0.7500 | 0.0134 | 0.0220 |  | 0.0000 | -0.4464 | 0.7258 | 0.1512 | 32.7251 |  | 0.0000 | 0.0000 | 0 |  | 1.2443 |
| KMNIST | AffineNorm-AdamW-LN-D6 | 3 | 0.7190 | 0.0363 | 0.0530 |  | -0.0273 | -0.4860 | 0.6002 | 0.1510 | 34.2373 |  | 0.0000 | 0.0000 | 960 | 0.000 | 1.3166 |
| KMNIST | AffineNorm-FDiag-LN-D6 | 3 | 0.7097 | 0.0042 | 0.0623 |  | -0.0579 | -0.5302 | 0.6152 | 0.1500 | 37.9564 |  | 0.0000 | 0.0000 | 0 | 1.000 | 1.3997 |
| KMNIST | FixedNorm-FNG-leftFullRight | 3 | 0.7280 | 0.0112 | 0.0440 |  | 0.2678 | -0.0591 | 0.3609 | 0.1488 | 30.5086 | 1895.8 | 0.0000 | 0.0000 | 0 |  | 1.4663 |
| KMNIST | AffineNorm-AdamW-LN-FNG-leftFullRight | 3 | 0.7287 | 0.0033 | 0.0433 |  | 0.2658 | -0.0619 | 0.3678 | 0.1487 | 37.2219 | 1895.8 | 0.0000 | 0.0000 | 960 | 0.000 | 1.5362 |
| KMNIST | AffineNorm-FDiag-LN-FNG-leftFullRight | 3 | 0.7240 | 0.0118 | 0.0480 |  | 0.2659 | -0.0619 | 0.3023 | 0.1483 | 28.3178 | 1895.8 | 0.0000 | 0.0000 | 0 | 1.000 | 1.6243 |

P2 verdict:

```text
NoNorm is a hard failure boundary, so normalization is necessary.
Learnable AffineNorm / ScalarGain can slightly help AdamW on MNIST/KMNIST,
but it does not close the PureKAN functional optimizer gap on Fashion/KMNIST.
AffineNorm-FDiag also does not rescue FNG-leftFullRight.

Conclusion: missing learnable normalization is not the main bottleneck.
The core issue remains KAN coefficient optimizer dynamics / metric direction.
```

## P3 Relaxed FNG Diagnostic

Because P1 had no formal survivor, this run is intentionally labeled relaxed diagnostic. It includes only the no-bad-step FNG variants plus baselines, to check whether the P1 local signal becomes a short-run training signal.

| dataset | method | runs | acc | std | gap vs AdamW | acc-D0 | AUC imp vs D6 | AUC imp vs AdamW | ECE red | phi p95 | J | FNG cond | bad | fallback | nonKAN | norm cov | time/A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MNIST | AdamW | 3 | 0.9303 | 0.0076 | 0.0000 | 0.0433 | 0.4248 | 0.0000 | 0.0000 | 0.2274 | 226275.4421 |  | 0.0000 | 0.0000 | 0 |  | 1.0000 |
| MNIST | D0-allFullSobolev | 3 | 0.8870 | 0.0455 | 0.0433 | 0.0000 | -0.0537 | -0.8318 | -0.3594 | 0.1489 | 34.4779 |  | 0.0000 | 0.0000 | 0 |  | 1.3322 |
| MNIST | D6-allTaskAware | 3 | 0.8620 | 0.0708 | 0.0683 | -0.0250 | 0.0000 | -0.7384 | 0.0949 | 0.1508 | 32.8464 |  | 0.0000 | 0.0000 | 0 |  | 1.3209 |
| MNIST | F2-FNG-right-only | 3 | 0.9297 | 0.0079 | 0.0007 | 0.0427 | 0.4046 | -0.0350 | 0.4066 | 0.1485 | 22.9295 | 2119.3 | 0.0000 | 0.0000 | 0 |  | 1.3819 |
| MNIST | F4-FNG-leftFull-right-noSob | 3 | 0.9030 | 0.0062 | 0.0273 | 0.0160 | 0.3915 | -0.0578 | 0.3140 | 0.1484 | 32.0859 | 2101.4 | 0.0000 | 0.0000 | 0 |  | 1.5255 |
| MNIST | F4-FNG-leftFull-right-lowSob | 3 | 0.9153 | 0.0074 | 0.0150 | 0.0283 | 0.4503 | 0.0444 | 0.3972 | 0.1488 | 28.4579 | 2079.7 | 0.0000 | 0.0000 | 0 |  | 1.4826 |
| Fashion-MNIST | AdamW | 3 | 0.8563 | 0.0090 | 0.0000 | 0.0243 | -0.0409 | 0.0000 | 0.0000 | 0.2396 | 369574.0127 |  | 0.0000 | 0.0000 | 0 |  | 1.0000 |
| Fashion-MNIST | D0-allFullSobolev | 3 | 0.8320 | 0.0120 | 0.0243 | 0.0000 | -0.1076 | -0.0640 | 0.6596 | 0.1490 | 31.4545 |  | 0.0000 | 0.0000 | 0 |  | 1.3977 |
| Fashion-MNIST | D6-allTaskAware | 3 | 0.8303 | 0.0358 | 0.0260 | -0.0017 | 0.0000 | 0.0393 | 0.3910 | 0.1506 | 31.2786 |  | 0.0000 | 0.0000 | 0 |  | 1.3386 |
| Fashion-MNIST | F2-FNG-right-only | 3 | 0.8467 | 0.0045 | 0.0097 | 0.0147 | 0.0900 | 0.1258 | 0.2483 | 0.1484 | 26.0255 | 1458.8 | 0.0000 | 0.0000 | 0 |  | 1.4323 |
| Fashion-MNIST | F4-FNG-leftFull-right-noSob | 3 | 0.8140 | 0.0159 | 0.0423 | -0.0180 | -0.0987 | -0.0555 | 0.0314 | 0.1484 | 32.0918 | 1394.1 | 0.0000 | 0.0000 | 0 |  | 1.5666 |
| Fashion-MNIST | F4-FNG-leftFull-right-lowSob | 3 | 0.8317 | 0.0040 | 0.0247 | -0.0003 | -0.0152 | 0.0247 | 0.1581 | 0.1485 | 27.0725 | 1477.1 | 0.0000 | 0.0000 | 0 |  | 1.5458 |
| KMNIST | AdamW | 3 | 0.7720 | 0.0102 | 0.0000 | 0.0937 | 0.3087 | 0.0000 | 0.0000 | 0.2172 | 20374.7067 |  | 0.0000 | 0.0000 | 0 |  | 1.0000 |
| KMNIST | D0-allFullSobolev | 3 | 0.6783 | 0.0147 | 0.0937 | 0.0000 | -0.2304 | -0.7797 | 0.7098 | 0.1493 | 35.5405 |  | 0.0000 | 0.0000 | 0 |  | 1.3990 |
| KMNIST | D6-allTaskAware | 3 | 0.7500 | 0.0134 | 0.0220 | 0.0717 | 0.0000 | -0.4464 | 0.7258 | 0.1512 | 32.7251 |  | 0.0000 | 0.0000 | 0 |  | 1.3298 |
| KMNIST | F2-FNG-right-only | 3 | 0.7397 | 0.0148 | 0.0323 | 0.0613 | 0.2836 | -0.0362 | 0.3761 | 0.1489 | 26.7338 | 1895.8 | 0.0000 | 0.0000 | 0 |  | 1.3588 |
| KMNIST | F4-FNG-leftFull-right-noSob | 3 | 0.6970 | 0.0079 | 0.0750 | 0.0187 | 0.2066 | -0.1477 | 0.1838 | 0.1488 | 28.4713 | 1829.8 | 0.0000 | 0.0000 | 0 |  | 1.4830 |
| KMNIST | F4-FNG-leftFull-right-lowSob | 3 | 0.7070 | 0.0128 | 0.0650 | 0.0287 | 0.2584 | -0.0727 | 0.1985 | 0.1487 | 26.5250 | 1845.7 | 0.0000 | 0.0000 | 0 |  | 1.5253 |

### P3 Gate Check

| method | formal P1 pass | max acc gap | AUC < D6 all | max bad | max fallback | geom <= AdamW | enter P4 |
|---|---|---|---|---|---|---|---|
| D6-allTaskAware | no | 0.0683 | no | 0.0000 | 0.0000 | yes | no |
| F2-FNG-right-only | no | 0.0323 | yes | 0.0000 | 0.0000 | yes | no |
| F4-FNG-leftFull-right-lowSob | no | 0.0650 | no | 0.0000 | 0.0000 | yes | no |
| F4-FNG-leftFull-right-noSob | no | 0.0750 | no | 0.0000 | 0.0000 | yes | no |

P3 verdict: no formal survivor enters P4. The relaxed run is useful diagnostically, but it cannot override the P1 gate. FNG improves some Sobolev-only trajectories, yet the joint accuracy/AUC/geometry requirements are not met across MNIST, Fashion-MNIST, and KMNIST.

## Visualizations

Generated dashboards:

```text
results/gafu_v3_9_figures/p1_direction_quality.svg
results/gafu_v3_9_figures/p2_norm_scorecard.svg
results/gafu_v3_9_figures/p3_fng_relaxed_scorecard.svg
matplotlib unavailable; SVG fallback used: No module named 'matplotlib'
```

## P4-P7 Decision

```text
P4 temporal dynamics:
  not run
  reason: no formal P1/P3 survivor.

P5 combined norm + FNG selection:
  not run
  reason: P3 entry gate was not reached.

P6 5-seed confirm / P7 10-seed final:
  not run by plan.
```

## Failure Diagnosis

```text
Normalization bottleneck:
  not primary.
  NoNorm fails, but learnable/functional norm does not recover PureKAN functional training.

FNG direction bottleneck:
  yes.
  FNG fixes many one-step bad directions, but the cosine gate and cross-dataset training gates fail.

Temporal dynamics bottleneck:
  possible but not yet justified.
  Since no candidate formally passed P1/P3, larger temporal-dynamics sweeps would be premature.

Architecture bottleneck:
  unlikely as the main cause.
  PureKAN-AdamW remains trainable; optimizer metric is the weak link.
```

## Final Decision

```text
PureKAN functional optimization remains unsolved in v3.9.

What is confirmed:
  1. Normalization is necessary, but learnable norm is not sufficient.
  2. FNG/KFAC-style task metrics repair part of the local descent problem.
  3. Current FNG still does not replace AdamW for full PureKAN training.

Accepted contribution:
  Hybrid DG-KAN branch functional update remains the valid mainline.

Next recommended direction:
  Redesign the PureKAN functional metric around safer layer-local/block-local objectives
  and only revisit temporal momentum/safeguards once a candidate passes the direction gate.
```
