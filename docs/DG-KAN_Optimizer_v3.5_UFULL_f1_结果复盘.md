# DG-KAN Optimizer v3.5 U-FULL f1 结果复盘

## P0 Config Freeze

```json
{
  "pass": true,
  "checks": [
    {
      "dataset": "Fashion-MNIST",
      "method": "U-FULL-f100-alphaFixed1",
      "branch_final_scale": 1.0,
      "alpha_trainable": 0.0,
      "alpha_final_mean": 1.0,
      "v3_metric_active": "full_sobolev_gram",
      "v3_metric_geometry": "full_sobolev_gram",
      "nonkan_update": "adamw",
      "trust_clip_rate": 0.0,
      "v3_phase_final": "GEOMETRY",
      "v3_metric_mix_auc": 1.0,
      "pass": true
    },
    {
      "dataset": "KMNIST",
      "method": "U-FULL-f100-alphaFixed1",
      "branch_final_scale": 1.0,
      "alpha_trainable": 0.0,
      "alpha_final_mean": 1.0,
      "v3_metric_active": "full_sobolev_gram",
      "v3_metric_geometry": "full_sobolev_gram",
      "nonkan_update": "adamw",
      "trust_clip_rate": 0.0,
      "v3_phase_final": "GEOMETRY",
      "v3_metric_mix_auc": 1.0,
      "pass": true
    }
  ]
}
```

## P1 Seed0 Sanity

| dataset | method | runs | acc | acc std | gap | AUC imp | phi red | J red | branch/A | ECE red | noKAN | time/A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | AdamW-alphaFixed1-alphaFixed1 | 1.0000 | 0.8310 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| Fashion-MNIST | F-V3-HARD-base30-v32best-alphaFixed1 | 1.0000 | 0.8370 | 0.0000 | -0.0060 | 0.0997 | 0.3660 | 0.9992 | 0.6401 | 0.1303 | 0.7269 | 1.3025 |
| Fashion-MNIST | U-FULL-f085-reference-alphaFixed1 | 1.0000 | 0.8400 | 0.0000 | -0.0090 | 0.1219 | 0.3690 | 0.9989 | 0.6192 | 0.1423 | 0.6426 | 1.3208 |
| Fashion-MNIST | U-FULL-f100-alphaFixed1 | 1.0000 | 0.8430 | 0.0000 | -0.0120 | 0.1064 | 0.3687 | 0.9976 | 0.6278 | 0.2175 | 0.6908 | 1.3248 |
| KMNIST | AdamW-alphaFixed1-alphaFixed1 | 1.0000 | 0.7920 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| KMNIST | K-V3-FULL-base-v32best-alphaFixed1 | 1.0000 | 0.8020 | 0.0000 | -0.0100 | 0.1098 | 0.2877 | 0.9994 | 0.5755 | 0.2608 | 0.7724 | 1.5288 |
| KMNIST | U-FULL-f085-reference-alphaFixed1 | 1.0000 | 0.7970 | 0.0000 | -0.0050 | 0.1124 | 0.2882 | 0.9993 | 0.5989 | 0.2009 | 0.8097 | 1.4676 |
| KMNIST | U-FULL-f100-alphaFixed1 | 1.0000 | 0.8100 | 0.0000 | -0.0180 | 0.0762 | 0.2871 | 0.9983 | 0.6220 | 0.2216 | 0.7948 | 1.5203 |

P1 f100 pass: `True`

```json
{
  "Fashion-MNIST": "pass",
  "KMNIST": "pass"
}
```

## P2 5-Seed Clean Confirm

| dataset | method | runs | acc | acc std | gap | AUC imp | phi red | J red | branch/A | ECE red | noKAN | time/A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | AdamW-alphaFixed1-alphaFixed1 | 5.0000 | 0.8392 | 0.0095 | -0.0000 | 0.0000 | -0.0000 | -0.0000 | 1.0000 | -0.0000 | 1.0000 | 1.0000 |
| Fashion-MNIST | F-V3-HARD-base30-v32best-alphaFixed1 | 5.0000 | 0.8474 | 0.0101 | -0.0082 | 0.1060 | 0.3563 | 0.9998 | 0.6592 | 0.2065 | 0.8169 | 1.4056 |
| Fashion-MNIST | U-FULL-f085-reference-alphaFixed1 | 5.0000 | 0.8482 | 0.0089 | -0.0090 | 0.0962 | 0.3601 | 0.9997 | 0.6199 | 0.1982 | 0.8291 | 1.4099 |
| Fashion-MNIST | U-FULL-f100-alphaFixed1 | 5.0000 | 0.8438 | 0.0080 | -0.0046 | 0.0982 | 0.3603 | 0.9994 | 0.6332 | 0.1746 | 0.7365 | 1.4252 |
| KMNIST | AdamW-alphaFixed1-alphaFixed1 | 5.0000 | 0.7792 | 0.0180 | -0.0000 | -0.0000 | 0.0000 | -0.0000 | 1.0000 | -0.0000 | 1.0000 | 1.0000 |
| KMNIST | K-V3-FULL-base-v32best-alphaFixed1 | 5.0000 | 0.7826 | 0.0151 | -0.0034 | 0.0906 | 0.2765 | 0.9999 | 0.5784 | 0.1859 | 0.6555 | 1.3324 |
| KMNIST | U-FULL-f085-reference-alphaFixed1 | 5.0000 | 0.7848 | 0.0122 | -0.0056 | 0.0980 | 0.2772 | 0.9999 | 0.5999 | 0.1836 | 0.7189 | 1.3561 |
| KMNIST | U-FULL-f100-alphaFixed1 | 5.0000 | 0.7736 | 0.0214 | 0.0056 | 0.0895 | 0.2778 | 0.9998 | 0.6246 | 0.0995 | 0.6394 | 1.3611 |

P2 f100 pass: `False`

```json
{
  "Fashion-MNIST": "pass",
  "KMNIST": "fail gap=0.0056 auc=0.0895 phi=0.278 J=1.000 branch=0.625 ece=0.099 noKAN=0.639"
}
```

Selected unified profile: `U-FULL-f085-reference-alphaFixed1`

## P2 Paired Delta vs AdamW

| dataset | method | acc delta | acc CI lo | acc CI hi | AUC delta | AUC CI lo | AUC CI hi | noKAN |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | U-FULL-f085-reference-alphaFixed1 | 0.0090 | 0.0010 | 0.0170 | 0.0539 | 0.0355 | 0.0735 | 0.8291 |
| Fashion-MNIST | U-FULL-f100-alphaFixed1 | 0.0046 | -0.0034 | 0.0128 | 0.0550 | 0.0393 | 0.0718 | 0.7365 |
| Fashion-MNIST | F-V3-HARD-base30-v32best-alphaFixed1 | 0.0082 | -0.0002 | 0.0196 | 0.0594 | 0.0481 | 0.0723 | 0.8169 |
| KMNIST | U-FULL-f085-reference-alphaFixed1 | 0.0056 | -0.0058 | 0.0184 | 0.0511 | 0.0374 | 0.0643 | 0.7189 |
| KMNIST | U-FULL-f100-alphaFixed1 | -0.0056 | -0.0158 | 0.0078 | 0.0466 | 0.0277 | 0.0670 | 0.6394 |
| KMNIST | K-V3-FULL-base-v32best-alphaFixed1 | 0.0034 | -0.0090 | 0.0156 | 0.0472 | 0.0315 | 0.0647 | 0.6555 |

## P3 10-Seed Final

Trigger decision: `not triggered because P2 did not pass`

_P3 not run._

## P4 Small-Data Generalization

| dataset | method | train | runs | acc | gap | AUC imp | phi red | J red | branch/A | ECE red | noKAN | time/A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | AdamW-alphaFixed1-alphaFixed1 | 1000.0000 | 5.0000 | 0.7956 | -0.0000 | -0.0000 | -0.0000 | 0.0000 | 1.0000 | -0.0000 | 1.0000 | 1.0000 |
| Fashion-MNIST | AdamW-alphaFixed1-alphaFixed1 | 2000.0000 | 5.0000 | 0.8122 | 0.0000 | 0.0000 | 0.0000 | -0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| Fashion-MNIST | AdamW-alphaFixed1-alphaFixed1 | 500.0000 | 5.0000 | 0.7626 | 0.0000 | 0.0000 | -0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| Fashion-MNIST | AdamW-alphaFixed1-alphaFixed1 | 6000.0000 | 5.0000 | 0.8392 | -0.0000 | 0.0000 | -0.0000 | -0.0000 | 1.0000 | -0.0000 | 1.0000 | 1.0000 |
| Fashion-MNIST | U-FULL-f085-reference-alphaFixed1 | 1000.0000 | 5.0000 | 0.7936 | 0.0020 | 0.1336 | 0.1515 | 0.9955 | 0.6532 | 0.6159 | 0.6616 | 1.0069 |
| Fashion-MNIST | U-FULL-f085-reference-alphaFixed1 | 2000.0000 | 5.0000 | 0.8054 | 0.0068 | 0.1423 | 0.1998 | 0.9992 | 0.6620 | 0.3990 | 0.6514 | 1.1197 |
| Fashion-MNIST | U-FULL-f085-reference-alphaFixed1 | 500.0000 | 5.0000 | 0.7594 | 0.0032 | 0.1064 | 0.1093 | 0.9976 | 0.6635 | 0.7616 | 0.6993 | 1.0504 |
| Fashion-MNIST | U-FULL-f085-reference-alphaFixed1 | 6000.0000 | 5.0000 | 0.8482 | -0.0090 | 0.0962 | 0.3601 | 0.9997 | 0.6199 | 0.1982 | 0.8291 | 1.4305 |
| KMNIST | AdamW-alphaFixed1-alphaFixed1 | 1000.0000 | 5.0000 | 0.6404 | -0.0000 | -0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| KMNIST | AdamW-alphaFixed1-alphaFixed1 | 2000.0000 | 5.0000 | 0.6826 | 0.0000 | 0.0000 | -0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| KMNIST | AdamW-alphaFixed1-alphaFixed1 | 500.0000 | 5.0000 | 0.5770 | -0.0000 | 0.0000 | 0.0000 | -0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| KMNIST | AdamW-alphaFixed1-alphaFixed1 | 6000.0000 | 5.0000 | 0.7792 | -0.0000 | -0.0000 | 0.0000 | -0.0000 | 1.0000 | -0.0000 | 1.0000 | 1.0000 |
| KMNIST | U-FULL-f085-reference-alphaFixed1 | 1000.0000 | 5.0000 | 0.6474 | -0.0070 | 0.0836 | 0.1088 | 0.9854 | 0.6411 | 0.2409 | 0.5611 | 1.0032 |
| KMNIST | U-FULL-f085-reference-alphaFixed1 | 2000.0000 | 5.0000 | 0.6978 | -0.0152 | 0.0806 | 0.1211 | 0.9957 | 0.6385 | 0.2634 | 0.6798 | 1.0957 |
| KMNIST | U-FULL-f085-reference-alphaFixed1 | 500.0000 | 5.0000 | 0.5968 | -0.0198 | 0.0972 | 0.1034 | 0.9104 | 0.6420 | 0.2859 | 0.5037 | 1.0928 |
| KMNIST | U-FULL-f085-reference-alphaFixed1 | 6000.0000 | 5.0000 | 0.7848 | -0.0056 | 0.0980 | 0.2772 | 0.9999 | 0.5999 | 0.1836 | 0.7189 | 1.4415 |

## P5 Label-Noise Generalization

| dataset | method | noise | runs | acc | AUC imp | ECE | ECE red | noKAN |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | AdamW-alphaFixed1-noise0.2-alphaFixed1 | 0.2000 | 5.0000 | 0.7450 | 0.0000 | 0.1105 | -0.0000 | 1.0000 |
| Fashion-MNIST | AdamW-alphaFixed1-noise0.4-alphaFixed1 | 0.4000 | 5.0000 | 0.5894 | -0.0000 | 0.1753 | -0.0000 | 1.0000 |
| Fashion-MNIST | U-FULL-f085-reference-alphaFixed1-noise0.2 | 0.2000 | 5.0000 | 0.7776 | 0.0701 | 0.0516 | 0.5336 | 1.2033 |
| Fashion-MNIST | U-FULL-f085-reference-alphaFixed1-noise0.4 | 0.4000 | 5.0000 | 0.6550 | 0.0772 | 0.0346 | 0.8025 | 1.5455 |
| KMNIST | AdamW-alphaFixed1-noise0.2-alphaFixed1 | 0.2000 | 5.0000 | 0.6264 | -0.0000 | 0.2322 | -0.0000 | 1.0000 |
| KMNIST | AdamW-alphaFixed1-noise0.4-alphaFixed1 | 0.4000 | 5.0000 | 0.4794 | 0.0000 | 0.3310 | 0.0000 | 1.0000 |
| KMNIST | U-FULL-f085-reference-alphaFixed1-noise0.2 | 0.2000 | 5.0000 | 0.6644 | 0.0682 | 0.1357 | 0.4155 | 1.2578 |
| KMNIST | U-FULL-f085-reference-alphaFixed1-noise0.4 | 0.4000 | 5.0000 | 0.4660 | 0.1611 | 0.2012 | 0.3920 | 1.1452 |

## P6 CIFAR-Small ConvStem Precheck

| dataset | method | train | runs | acc | gap | AUC imp | phi red | J red | branch/A | ECE red | noKAN | time/A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CIFAR10 | ConvStem-DGKAN-AdamW-alphaFixed1 | 10000.0000 | 3.0000 | 0.4756 | -0.0000 | 0.0000 | -0.0000 | -0.0000 | 1.0000 | -0.0000 | 1.0000 | 1.0000 |
| CIFAR10 | ConvStem-DGKAN-U-FULL-f085-alphaFixed1 | 10000.0000 | 3.0000 | 0.4025 | 0.0731 | -0.0874 | 0.4908 | 1.0000 | 0.4598 | 0.1905 | 0.1224 | 1.0794 |
| CIFAR10 | ConvStem-DGKAN-U-FULL-f100-alphaFixed1 | 10000.0000 | 3.0000 | 0.4177 | 0.0579 | -0.0961 | 0.4900 | 1.0000 | 0.5098 | 0.2592 | 0.1926 | 1.0871 |

```text
P6 CIFAR-small: 9 successful rows; f085 acc=0.4025, gap_vs_conv_adamw=0.0731, AUC_imp=-0.0874; f100 acc=0.4177, gap_vs_conv_adamw=0.0579, AUC_imp=-0.0961
```

## P7 Target-Matched / Speed Proxy

| dataset | method | runs | relaxed loss reached | epochs relaxed loss | time relaxed loss | step ms |
|---|---|---|---|---|---|---|
| Fashion-MNIST | AdamW-alphaFixed1-alphaFixed1 | 5.0000 | 1.0000 | 1.4000 | 1.0960 | 32.6156 |
| Fashion-MNIST | F-V3-HARD-base30-v32best-alphaFixed1 | 5.0000 | 1.0000 | 2.0000 | 2.2005 | 45.8445 |
| Fashion-MNIST | U-FULL-f085-reference-alphaFixed1 | 5.0000 | 1.0000 | 2.2000 | 2.4294 | 45.9854 |
| Fashion-MNIST | U-FULL-f100-alphaFixed1 | 5.0000 | 1.0000 | 2.0000 | 2.2313 | 46.4852 |
| KMNIST | AdamW-alphaFixed1-alphaFixed1 | 5.0000 | 1.0000 | 2.2000 | 1.8101 | 34.2870 |
| KMNIST | K-V3-FULL-base-v32best-alphaFixed1 | 5.0000 | 1.0000 | 2.4000 | 2.6380 | 45.6830 |
| KMNIST | U-FULL-f085-reference-alphaFixed1 | 5.0000 | 1.0000 | 2.8000 | 3.1268 | 46.4981 |
| KMNIST | U-FULL-f100-alphaFixed1 | 5.0000 | 1.0000 | 2.6000 | 2.9124 | 46.6692 |

## P8 Rational/KAT Precheck

| dataset | method | runs | acc | acc std | gap vs MLP | AUC imp | ECE red | time/MLP | clip | condG | rawG | dirN | den min | den p01 | mem MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | MLP-AdamW-alphaFixed1-alphaFixed1 | 3.0000 | 0.8500 | 0.0137 | 0.0000 | -0.0000 | 0.0000 | 1.0000 | 0.0000 |  | 0.0000 | 0.0000 |  |  |  |
| Fashion-MNIST | Rational-DGKAN-AdamW-torch-alphaFixed1 | 3.0000 | 0.8457 | 0.0061 | 0.0043 | -0.0361 | -0.0311 | 4.0186 | 0.0000 |  | 0.0000 | 0.0000 | 1.0004 | 1.0006 |  |
| Fashion-MNIST | Rational-DGKAN-AdamW-triton-alphaFixed1 | 3.0000 | 0.8503 | 0.0060 | -0.0003 | -0.0473 | -0.0395 | 2.8456 | 0.0000 |  | 0.0000 | 0.0000 | 1.0005 | 1.0008 |  |
| Fashion-MNIST | Rational-DGKAN-U-FULL-torch-c0p02-alphaFixed1 | 3.0000 | 0.8383 | 0.0090 | 0.0117 | -0.0265 | -0.0832 | 5.5948 | 0.0851 | 36422.6719 | 0.4159 | 0.5515 | 1.0001 | 1.0001 |  |
| Fashion-MNIST | Rational-DGKAN-U-FULL-triton-c0p02-alphaFixed1 | 3.0000 | 0.8450 | 0.0106 | 0.0050 | -0.0243 | 0.0303 | 4.9895 | 0.0464 | 36422.6719 | 0.4248 | 0.3901 | 1.0001 | 1.0001 |  |
| Fashion-MNIST | U-FULL-f085-reference-alphaFixed1 | 3.0000 | 0.8487 | 0.0084 | 0.0013 | 0.0061 | 0.0642 | 4.6658 | 0.0000 | 4725.9624 | 0.7991 | 0.8281 |  |  |  |
| Fashion-MNIST | U-FULL-f100-alphaFixed1 | 3.0000 | 0.8423 | 0.0098 | 0.0077 | -0.0019 | 0.0423 | 4.6842 | 0.0000 | 4725.9624 | 0.8241 | 0.9588 |  |  |  |
| KMNIST | MLP-AdamW-alphaFixed1-alphaFixed1 | 3.0000 | 0.7920 | 0.0120 | 0.0000 | 0.0000 | -0.0000 | 1.0000 | 0.0000 |  | 0.0000 | 0.0000 |  |  |  |
| KMNIST | Rational-DGKAN-AdamW-torch-alphaFixed1 | 3.0000 | 0.7903 | 0.0172 | 0.0017 | -0.1282 | -0.1289 | 4.9322 | 0.0000 |  | 0.0000 | 0.0000 | 1.0004 | 1.0007 |  |
| KMNIST | Rational-DGKAN-AdamW-triton-alphaFixed1 | 3.0000 | 0.7890 | 0.0059 | 0.0030 | -0.1314 | -0.1331 | 3.2510 | 0.0000 |  | 0.0000 | 0.0000 | 1.0006 | 1.0010 |  |
| KMNIST | Rational-DGKAN-U-FULL-torch-c0p02-alphaFixed1 | 3.0000 | 0.7863 | 0.0025 | 0.0057 | -0.0464 | 0.0001 | 6.3566 | 0.0650 | 36422.6719 | 0.2799 | 0.3374 | 1.0000 | 1.0000 |  |
| KMNIST | Rational-DGKAN-U-FULL-triton-c0p02-alphaFixed1 | 3.0000 | 0.7947 | 0.0085 | -0.0027 | -0.0183 | 0.0110 | 5.6982 | 0.0424 | 36422.6719 | 0.2191 | 0.2101 | 1.0000 | 1.0000 |  |
| KMNIST | U-FULL-f085-reference-alphaFixed1 | 3.0000 | 0.7860 | 0.0128 | 0.0060 | -0.1173 | 0.1165 | 5.3174 | 0.0000 | 4725.9624 | 0.3950 | 0.4217 |  |  |  |
| KMNIST | U-FULL-f100-alphaFixed1 | 3.0000 | 0.7833 | 0.0200 | 0.0087 | -0.1477 | 0.0809 | 5.3836 | 0.0000 | 4725.9624 | 0.4460 | 0.4609 |  |  |  |

```text
P8 Rational/KAT: 42 successful rows; Fashion-MNIST triton acc=0.8503, gap_vs_mlp=-0.0003, time/MLP=2.846, denom_min=1.001; Fashion-MNIST torch acc=0.8457, time/MLP=4.019; Fashion-MNIST U-FULL triton acc=0.8450, AUC_imp=-0.0243, condG=36422.7, clip=0.046, denom_min=1.000; Fashion-MNIST U-FULL torch acc=0.8383, AUC_imp=-0.0265; KMNIST triton acc=0.7890, gap_vs_mlp=0.0030, time/MLP=3.251, denom_min=1.001; KMNIST torch acc=0.7903, time/MLP=4.932; KMNIST U-FULL triton acc=0.7947, AUC_imp=-0.0183, condG=36422.7, clip=0.042, denom_min=1.000; KMNIST U-FULL torch acc=0.7863, AUC_imp=-0.0464
```

## Decision

```text
P0:
  config check pass = True

P1:
  U-FULL-f100 seed0 sanity pass = True

P2:
  U-FULL-f100 5-seed clean pass = False
  selected unified profile = U-FULL-f085-reference-alphaFixed1

P3:
  not triggered because P2 did not pass

P4/P5:
  See scorecards above for small-data and label-noise behavior.

P7:
  Target-matched wall-clock proxy was generated from clean confirm rows.
  Low-level kernel profiling and solve microbenchmarks were not expanded in this run.

P6/P8:
  CIFAR-small and Rational/KAT prechecks were included when their rows are present.
  Rational uses third_party/rational_kat_cu directly, with Triton autograd or the PyTorch fallback.
  No custom backward rewrite was accepted in this round.
```
