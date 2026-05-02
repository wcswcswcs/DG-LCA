# DG-KAN v3.6 RationalUFull / PureKAN 结果复盘

## Decision

```json
{
  "p0_pass": true,
  "p3_rational_confirm_triggered": false,
  "p4_run": false,
  "p8_purekan_confirm_triggered": false,
  "p9_run": false,
  "total_rows": 266
}
```

## P0 Implementation Smoke

| dataset | method | runs | acc | condG | clip | den p01 | pure nonKAN |
|---|---|---|---|---|---|---|---|
| Fashion-MNIST | Hybrid-UFULL-smoke-alphaFixed1 | 1 | 0.4609 | 6551.6172 | 0.0000 | nan | nan |
| Fashion-MNIST | PureKAN-AdamW-smoke-alphaFixed1 | 1 | 0.1094 | nan | 0.0000 | nan | 0.0000 |
| Fashion-MNIST | PureKAN-UFULL-smoke-alphaFixed1 | 1 | 0.1328 | 6551.6172 | 0.0000 | nan | 0.0000 |
| Fashion-MNIST | Rational-DGKAN-AdamW-triton-alphaFixed1 | 1 | 0.3516 | nan | 0.0000 | 1.0001 | nan |
| Fashion-MNIST | Rational-UFULL-branch-smoke-alphaFixed1 | 1 | 0.3438 | 8508.2334 | 0.0000 | 1.0001 | nan |
| Fashion-MNIST | Rational-UFULL-poly-separate-smoke-alphaFixed1 | 1 | 0.3438 | 251534.2344 | 0.5000 | 1.0001 | nan |
| Fashion-MNIST | Rational-UFULL-tangent-joint-smoke-alphaFixed1 | 1 | 0.3516 | 8507.8916 | 0.0000 | 1.0001 | nan |
| KMNIST | Hybrid-UFULL-smoke-alphaFixed1 | 1 | 0.2969 | 6551.6172 | 0.0000 | nan | nan |
| KMNIST | PureKAN-AdamW-smoke-alphaFixed1 | 1 | 0.1016 | nan | 0.0000 | nan | 0.0000 |
| KMNIST | PureKAN-UFULL-smoke-alphaFixed1 | 1 | 0.1328 | 6551.6172 | 0.0000 | nan | 0.0000 |
| KMNIST | Rational-DGKAN-AdamW-triton-alphaFixed1 | 1 | 0.2578 | nan | 0.0000 | 1.0001 | nan |
| KMNIST | Rational-UFULL-branch-smoke-alphaFixed1 | 1 | 0.2422 | 8467.6592 | 0.0000 | 1.0001 | nan |
| KMNIST | Rational-UFULL-poly-separate-smoke-alphaFixed1 | 1 | 0.2422 | 251534.2344 | 0.5000 | 1.0001 | nan |
| KMNIST | Rational-UFULL-tangent-joint-smoke-alphaFixed1 | 1 | 0.2422 | 8467.5293 | 0.0000 | 1.0001 | nan |

## P1 Rational Metric Audit

| dataset | method | runs | acc | val AUC | condG | clip | den p01 |
|---|---|---|---|---|---|---|---|
| Fashion-MNIST | Rational-DGKAN-AdamW-triton-alphaFixed1 | 3 | 0.8487 | 0.5293 | nan | 0.0000 | 1.0001 |
| Fashion-MNIST | Rational-UFULL-identity-alphaFixed1 | 3 | 0.8433 | 0.5328 | nan | 0.0008 | 1.0001 |
| Fashion-MNIST | Rational-UFULL-legendre-separate-alphaFixed1 | 3 | 0.8433 | 0.5096 | 250153.9531 | 0.0000 | 1.0001 |
| Fashion-MNIST | Rational-UFULL-poly-diag-alphaFixed1 | 3 | 0.8447 | 0.5095 | 250153.9531 | 0.0000 | 1.0001 |
| Fashion-MNIST | Rational-UFULL-poly-rho1e-1-alphaFixed1 | 3 | 0.8453 | 0.5253 | 10636.6377 | 0.0204 | 1.0002 |
| Fashion-MNIST | Rational-UFULL-poly-rho1e-2-alphaFixed1 | 3 | 0.8397 | 0.5221 | 82248.4141 | 0.2635 | 1.0004 |
| Fashion-MNIST | Rational-UFULL-poly-separate-current-alphaFixed1 | 3 | 0.8440 | 0.5266 | 250153.9531 | 0.4543 | 1.0004 |
| KMNIST | Rational-DGKAN-AdamW-triton-alphaFixed1 | 3 | 0.7930 | 0.4747 | nan | 0.0000 | 1.0002 |
| KMNIST | Rational-UFULL-identity-alphaFixed1 | 3 | 0.7993 | 0.4714 | nan | 0.0008 | 1.0001 |
| KMNIST | Rational-UFULL-legendre-separate-alphaFixed1 | 3 | 0.7840 | 0.4385 | 250153.9531 | 0.0000 | 1.0001 |
| KMNIST | Rational-UFULL-poly-diag-alphaFixed1 | 3 | 0.7817 | 0.4390 | 250153.9531 | 0.0000 | 1.0001 |
| KMNIST | Rational-UFULL-poly-rho1e-1-alphaFixed1 | 3 | 0.7890 | 0.4314 | 10636.6377 | 0.0429 | 1.0001 |
| KMNIST | Rational-UFULL-poly-rho1e-2-alphaFixed1 | 3 | 0.7917 | 0.4321 | 82248.4141 | 0.1892 | 1.0003 |
| KMNIST | Rational-UFULL-poly-separate-current-alphaFixed1 | 3 | 0.7950 | 0.4249 | 250153.9531 | 0.2607 | 1.0002 |

## P2 Rational Toy

| target | method | runs | mse | loss AUC | bad | cond | den p01 |
|---|---|---|---|---|---|---|---|
| silu | adam | 3 | 0.00022 | 0.00347 | 0.0100 | nan | 1.0078 |
| silu | poly-diag | 3 | 0.0645 | 0.152 | 0.0000 | 71667.7734 | 1.0019 |
| silu | poly-separate-current | 3 | 0.121 | 0.203 | 0.0000 | 71667.7734 | 1.0047 |
| silu | tangent-diag | 3 | 0.00317 | 0.0335 | 0.0000 | 113207.6875 | 1.0042 |
| silu | tangent-joint-H1 | 3 | 0.0706 | 0.0697 | 0.0000 | 489526.3125 | 1.0016 |
| silu | tangent-joint-L2 | 3 | 0.00267 | 0.0463 | 0.0000 | 148502.0938 | 1.0027 |
| smoothstep | adam | 3 | 4.98e-05 | 0.00226 | 0.0333 | nan | 1.0054 |
| smoothstep | poly-diag | 3 | 0.0148 | 0.0194 | 0.0000 | 71667.7734 | 1.0027 |
| smoothstep | poly-separate-current | 3 | 0.00373 | 0.00885 | 0.0133 | 71667.7734 | 1.0045 |
| smoothstep | tangent-diag | 3 | 0.000839 | 0.00644 | 0.0000 | 63036.1992 | 1.0057 |
| smoothstep | tangent-joint-H1 | 3 | 0.0248 | 0.0124 | 0.0000 | 91104.8125 | 1.0021 |
| smoothstep | tangent-joint-L2 | 3 | 2.06e-05 | 0.00351 | 0.0000 | 146524.1094 | 1.0031 |
| tanh_sin | adam | 3 | 0.000492 | 0.0066 | 0.0167 | nan | 1.0014 |
| tanh_sin | poly-diag | 3 | 0.0332 | 0.0351 | 0.0000 | 71667.7734 | 1.0022 |
| tanh_sin | poly-separate-current | 3 | 0.0197 | 0.0273 | 0.0000 | 71667.7734 | 1.0003 |
| tanh_sin | tangent-diag | 3 | 0.0225 | 0.027 | 0.0000 | 47487.4609 | 1.0047 |
| tanh_sin | tangent-joint-H1 | 3 | 0.00136 | 0.0216 | 0.0000 | 3537.6443 | 1.0003 |
| tanh_sin | tangent-joint-L2 | 3 | 0.000517 | 0.009 | 0.0000 | 4469.1133 | 1.0003 |

## P3 Rational Tangent DGKAN

| dataset | method | runs | acc | val AUC | condG | clip | den p01 |
|---|---|---|---|---|---|---|---|
| Fashion-MNIST | RBF-UFULL-f085-reference-alphaFixed1 | 3 | 0.8403 | 0.5205 | 3836.6843 | 0.0000 | nan |
| Fashion-MNIST | Rational-DGKAN-AdamW-triton-alphaFixed1 | 3 | 0.8440 | 0.5299 | nan | 0.0000 | 1.0002 |
| Fashion-MNIST | Rational-UFULL-poly-current-alphaFixed1 | 3 | 0.8437 | 0.5245 | 250153.9531 | 0.4547 | 1.0004 |
| Fashion-MNIST | Rational-UFULL-tangent-act-H1-alphaFixed1 | 3 | 0.8383 | 0.5166 | 18193.7865 | 0.0000 | 1.0001 |
| Fashion-MNIST | Rational-UFULL-tangent-act-L2-alphaFixed1 | 3 | 0.8407 | 0.5199 | 15341.3952 | 0.0000 | 1.0001 |
| Fashion-MNIST | Rational-UFULL-tangent-branch-H1-alphaFixed1 | 3 | 0.8357 | 0.4875 | 17753.4349 | 0.0000 | 1.0001 |
| Fashion-MNIST | Rational-UFULL-tangent-branch-L2-alphaFixed1 | 3 | 0.8393 | 0.4882 | 14914.3740 | 0.0000 | 1.0001 |
| Fashion-MNIST | Rational-UFULL-tangent-diag-alphaFixed1 | 3 | 0.8403 | 0.5097 | 15918.6956 | 0.0000 | 1.0001 |
| KMNIST | RBF-UFULL-f085-reference-alphaFixed1 | 3 | 0.7610 | 0.4841 | 3836.6843 | 0.0000 | nan |
| KMNIST | Rational-DGKAN-AdamW-triton-alphaFixed1 | 3 | 0.8010 | 0.4711 | nan | 0.0000 | 1.0001 |
| KMNIST | Rational-UFULL-poly-current-alphaFixed1 | 3 | 0.7907 | 0.4306 | 250153.9531 | 0.2912 | 1.0002 |
| KMNIST | Rational-UFULL-tangent-act-H1-alphaFixed1 | 3 | 0.7823 | 0.4338 | 18051.0501 | 0.0000 | 1.0001 |
| KMNIST | Rational-UFULL-tangent-act-L2-alphaFixed1 | 3 | 0.7873 | 0.4283 | 15299.2191 | 0.0000 | 1.0001 |
| KMNIST | Rational-UFULL-tangent-branch-H1-alphaFixed1 | 3 | 0.7720 | 0.4226 | 17639.1400 | 0.0000 | 1.0001 |
| KMNIST | Rational-UFULL-tangent-branch-L2-alphaFixed1 | 3 | 0.7743 | 0.4226 | 14814.0218 | 0.0000 | 1.0001 |
| KMNIST | Rational-UFULL-tangent-diag-alphaFixed1 | 3 | 0.7823 | 0.4312 | 15954.1966 | 0.0000 | 1.0001 |

### P3 Paired Delta vs Rational AdamW

| dataset | method | runs | acc delta | AUC imp |
|---|---|---|---|---|
| Fashion-MNIST | RBF-UFULL-f085-reference-alphaFixed1 | 3 | -0.0037 | 0.0094 |
| Fashion-MNIST | Rational-UFULL-poly-current-alphaFixed1 | 3 | -0.0003 | 0.0054 |
| Fashion-MNIST | Rational-UFULL-tangent-act-H1-alphaFixed1 | 3 | -0.0057 | 0.0134 |
| Fashion-MNIST | Rational-UFULL-tangent-act-L2-alphaFixed1 | 3 | -0.0033 | 0.0101 |
| Fashion-MNIST | Rational-UFULL-tangent-branch-H1-alphaFixed1 | 3 | -0.0083 | 0.0425 |
| Fashion-MNIST | Rational-UFULL-tangent-branch-L2-alphaFixed1 | 3 | -0.0047 | 0.0418 |
| Fashion-MNIST | Rational-UFULL-tangent-diag-alphaFixed1 | 3 | -0.0037 | 0.0203 |
| KMNIST | RBF-UFULL-f085-reference-alphaFixed1 | 3 | -0.0400 | -0.0130 |
| KMNIST | Rational-UFULL-poly-current-alphaFixed1 | 3 | -0.0103 | 0.0405 |
| KMNIST | Rational-UFULL-tangent-act-H1-alphaFixed1 | 3 | -0.0187 | 0.0373 |
| KMNIST | Rational-UFULL-tangent-act-L2-alphaFixed1 | 3 | -0.0137 | 0.0428 |
| KMNIST | Rational-UFULL-tangent-branch-H1-alphaFixed1 | 3 | -0.0290 | 0.0485 |
| KMNIST | Rational-UFULL-tangent-branch-L2-alphaFixed1 | 3 | -0.0267 | 0.0485 |
| KMNIST | Rational-UFULL-tangent-diag-alphaFixed1 | 3 | -0.0187 | 0.0399 |

## P5 U-FULL f100 Failure Audit

| dataset | method | runs | acc | val AUC | branch | noKAN | margin |
|---|---|---|---|---|---|---|---|
| Fashion-MNIST | AdamW-alphaFixed1-alphaFixed1 | 5 | 0.8322 | 0.5527 | 0.7659 | 0.2050 | 4.1511 |
| Fashion-MNIST | F-V3-HARD-base30-v32best-alphaFixed1 | 5 | 0.8376 | 0.5194 | 0.6210 | 0.1912 | 3.9238 |
| Fashion-MNIST | U-FULL-f085-reference-alphaFixed1 | 5 | 0.8420 | 0.5257 | 0.6110 | 0.1892 | 3.8446 |
| Fashion-MNIST | U-FULL-f100-alphaFixed1 | 5 | 0.8498 | 0.5253 | 0.6095 | 0.1780 | 3.8809 |
| KMNIST | AdamW-alphaFixed1-alphaFixed1 | 5 | 0.7740 | 0.5123 | 0.7718 | 0.2486 | 3.8950 |
| KMNIST | K-V3-FULL-base-v32best-alphaFixed1 | 5 | 0.7644 | 0.4864 | 0.5346 | 0.2000 | 3.7396 |
| KMNIST | U-FULL-f085-reference-alphaFixed1 | 5 | 0.7642 | 0.4854 | 0.5469 | 0.1994 | 3.7369 |
| KMNIST | U-FULL-f100-alphaFixed1 | 5 | 0.7724 | 0.4806 | 0.5683 | 0.1858 | 3.5859 |

## P6 CIFAR Branch Audit

| dataset | method | runs | acc | val AUC | branch | noKAN | dead | outgrid | conv p95 |
|---|---|---|---|---|---|---|---|---|---|
| CIFAR10 | ConvStem-DGKAN-AdamW-alphaFixed1 | 3 | 0.5063 | 1.5982 | 1.7772 | 0.2512 | 0.0000 | 0.0135 | 0.6095 |
| CIFAR10 | ConvStem-DGKAN-U-FULL-f085-alphaFixed1 | 3 | 0.3610 | 1.8121 | 0.7542 | -0.0278 | 0.0000 | 0.0144 | 0.7065 |
| CIFAR10 | ConvStem-DGKAN-U-FULL-f100-alphaFixed1 | 3 | 0.3772 | 1.7837 | 0.8412 | -0.0077 | 0.0000 | 0.0133 | 0.6824 |
| CIFAR10 | ConvStem-DGKAN-UFULL-branchboost-probe-alphaFixed1 | 3 | 0.3503 | 1.7794 | 0.7770 | -0.0475 | 0.0000 | 0.0128 | 0.6837 |
| CIFAR10 | ConvStem-DGKAN-UFULL-diag-warmup-probe-alphaFixed1 | 3 | 0.4040 | 1.7821 | 0.7630 | 0.0158 | 0.0000 | 0.0146 | 0.6981 |

## P8 PureKAN 3-Seed

| dataset | method | runs | acc | val AUC | condG | pure nonKAN | dead | outgrid |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | Hybrid-DGKAN-UFULL-f085-alphaFixed1 | 3 | 0.8463 | 0.5018 | 3836.6843 | nan | 0.0000 | 0.0201 |
| Fashion-MNIST | MLP-AdamW-alphaFixed1 | 3 | 0.8390 | 0.4820 | nan | nan | nan | nan |
| Fashion-MNIST | PureKAN-AdamW-alphaFixed1 | 3 | 0.8577 | 0.5426 | nan | 0.0000 | 0.0156 | 0.0025 |
| Fashion-MNIST | PureKAN-UFULL-alphaFixed1 | 3 | 0.8333 | 0.6363 | 3835.3533 | 0.0000 | 0.0156 | 0.0083 |
| Fashion-MNIST | PureKAN-UFULL-diagwarmup-alphaFixed1 | 3 | 0.8333 | 0.6363 | 3835.3533 | 0.0000 | 0.0156 | 0.0083 |
| KMNIST | Hybrid-DGKAN-UFULL-f085-alphaFixed1 | 3 | 0.7620 | 0.4566 | 3836.6843 | nan | 0.0000 | 0.0266 |
| KMNIST | MLP-AdamW-alphaFixed1 | 3 | 0.7697 | 0.4528 | nan | nan | nan | nan |
| KMNIST | PureKAN-AdamW-alphaFixed1 | 3 | 0.7740 | 0.4917 | nan | 0.0000 | 0.0156 | 0.0023 |
| KMNIST | PureKAN-UFULL-alphaFixed1 | 3 | 0.6693 | 0.9495 | 3835.3533 | 0.0000 | 0.0156 | 0.0104 |
| KMNIST | PureKAN-UFULL-diagwarmup-alphaFixed1 | 3 | 0.6693 | 0.9495 | 3835.3533 | 0.0000 | 0.0156 | 0.0104 |
| MNIST | Hybrid-DGKAN-UFULL-f085-alphaFixed1 | 3 | 0.9423 | 0.2804 | 3836.6843 | nan | 0.0000 | 0.0192 |
| MNIST | MLP-AdamW-alphaFixed1 | 3 | 0.9427 | 0.2848 | nan | nan | nan | nan |
| MNIST | PureKAN-AdamW-alphaFixed1 | 3 | 0.9343 | 0.3931 | nan | 0.0000 | 0.0312 | 0.0324 |
| MNIST | PureKAN-UFULL-alphaFixed1 | 3 | 0.8980 | 0.7370 | 3835.3533 | 0.0000 | 0.0312 | 0.0379 |
| MNIST | PureKAN-UFULL-diagwarmup-alphaFixed1 | 3 | 0.8980 | 0.7370 | 3835.3533 | 0.0000 | 0.0312 | 0.0379 |

### P8 Paired Delta vs PureKAN-AdamW

| dataset | method | runs | acc delta | AUC imp |
|---|---|---|---|---|
| Fashion-MNIST | Hybrid-DGKAN-UFULL-f085-alphaFixed1 | 3 | -0.0113 | 0.0408 |
| Fashion-MNIST | MLP-AdamW-alphaFixed1 | 3 | -0.0187 | 0.0606 |
| Fashion-MNIST | PureKAN-UFULL-alphaFixed1 | 3 | -0.0243 | -0.0937 |
| Fashion-MNIST | PureKAN-UFULL-diagwarmup-alphaFixed1 | 3 | -0.0243 | -0.0937 |
| KMNIST | Hybrid-DGKAN-UFULL-f085-alphaFixed1 | 3 | -0.0120 | 0.0351 |
| KMNIST | MLP-AdamW-alphaFixed1 | 3 | -0.0043 | 0.0389 |
| KMNIST | PureKAN-UFULL-alphaFixed1 | 3 | -0.1047 | -0.4578 |
| KMNIST | PureKAN-UFULL-diagwarmup-alphaFixed1 | 3 | -0.1047 | -0.4578 |
| MNIST | Hybrid-DGKAN-UFULL-f085-alphaFixed1 | 3 | 0.0080 | 0.1127 |
| MNIST | MLP-AdamW-alphaFixed1 | 3 | 0.0083 | 0.1082 |
| MNIST | PureKAN-UFULL-alphaFixed1 | 3 | -0.0363 | -0.3439 |
| MNIST | PureKAN-UFULL-diagwarmup-alphaFixed1 | 3 | -0.0363 | -0.3439 |

## Final Interpretation

P0 通过：Rational tangent metric 有限且 condition 已压到 smoke gate 内，PureKAN 的 learnable non-KAN 参数为 0。

Rational/KAT：tangent metric 修复了旧 poly metric 的高 clip 问题，但 P3 中 condition 仍约 1.5e4-1.8e4，且相对 Rational-AdamW 没有稳定 AUC/accuracy 优势，因此 P4 不触发。

U-FULL f100：Fashion 上 f100 有更强 accuracy signal，但 KMNIST 与 f085/v3.2 best 仍不稳；branch 更大不等于任务对齐更好。

CIFAR-small：ConvStem U-FULL 仍明显低于 ConvStem AdamW，diag warmup/branch boost probe 只能局部缓解，不能恢复到 AdamW。

PureKAN：结构实现成功，AdamW 可训练；但 PureKAN-UFULL 在 Fashion/KMNIST/MNIST 都明显慢或掉点，说明当前 U-FULL 仍更像 hybrid branch optimizer，而不是完整 PureKAN optimizer。P9 不触发。

