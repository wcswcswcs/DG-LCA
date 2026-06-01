# DG-KAN v5.0 Edge-Decomposed Functional Training 结果复盘

本轮依据 `docs/DG-KAN_v5.0_EdgeDecomposedFunctionalTraining_实验计划.md`。核心问题从“继续调 Sobolev optimizer”转为验证 edge-decomposed PureKAN：base path 学低阶/尺度结构，RBF residual 承担非线性，并把 geometry 主要约束到 residual 上。

## Run Inventory

| stage | rows | errors |
|---|---:|---:|
| P0 code audit | 27 | 0 |
| P1 architecture frontier | 81 | 0 |
| P2 eigenmode audit | 432 | 0 |
| P3 split functional update | 81 | 0 |
| P4 relaxed NFS projection | 120 | 0 |
| P5 learn-smooth-refresh | 0 | 0 |
| P6 event TAN | 0 | 0 |
| P7 capacity follow-up | 27 | 0 |

## Code / Config Changes

```text
experiments/run_gafu_v50.py
  Added edge-decomposed PureKAN modules:
    ABRBF-linear
    ABRBF-silu
    ABRBF-linear+silu
    BaseOnly-linear / BaseOnly-silu
    learnWidth / quantile-center probes
  Added split geometry audit:
    phi_base_p95 / phi_rbf_p95 / phi_total_p95
    curvature_base/rbf/total
    sobolev_rbf_norm
  Added split functional update:
    base Adam-like task update
    RBF residual identity / Sobolev / dataSob-style updates
  Added relaxed NFS row-space solver using Woodbury form.

experiments/analyze_gafu_v50.py
  Generates gate summaries, failure taxonomy, aggregate_decision.json,
  required artifact aliases, SVG diagnostics, log entry, and this replay.
```

## P0 Code Audit

| rows | errors | pass | max nonKAN | min coverage | max rollback | max KKT | max CG |
|---|---|---|---|---|---|---|---|
| 27 | 0 | true | 0.0000 | 1.0000 | 0.0000 | 0.000001 | 0.000001 |

P0 verdict: pass. Edge-only 参数计数、functional coverage、rollback、Exact/Relaxed NFS smoke 都可审计。Relaxed NFS 初版 dense solve 触发 OOM，已改为 row-space/Woodbury 解法。

## P1 Architecture Frontier

| dataset | method | runs | acc | gap vs RBF | phi_rbf | phi_total | base/RBF | positive |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | MLP-AdamW | 3 | 0.7819 | 0.0124 | 0.0000 |  | 0.0000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | 3 | 0.7988 | -0.0046 | 0.1135 | 0.1319 | 0.6582 | 1 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | 3 | 0.7949 | -0.0007 | 0.1147 | 0.1276 | 0.5480 | 1 |
| Fashion-MNIST | PureKAN-ABRBF-linear-learnWidth-AdamW | 3 | 0.7930 | 0.0013 | 0.1155 | 0.1279 | 0.5651 | 1 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | 3 | 0.7969 | -0.0026 | 0.1151 | 0.1208 | 0.1647 | 1 |
| Fashion-MNIST | PureKAN-BaseOnly-linear-AdamW | 3 | 0.7786 | 0.0156 | 0.0000 | 0.1172 | 218960613674587.6875 | 0 |
| Fashion-MNIST | PureKAN-BaseOnly-silu-AdamW | 3 | 0.7858 | 0.0085 | 0.0000 | 0.0926 | 127602617051866.3281 | 0 |
| Fashion-MNIST | PureKAN-RBF-quantileCenters-AdamW | 3 | 0.8027 | -0.0085 | 0.1627 | 0.1627 | 0.0000 | 1 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | 3 | 0.7943 | 0.0000 | 0.1179 | 0.1179 | 0.0000 | 1 |
| KMNIST | MLP-AdamW | 3 | 0.6927 | -0.0117 | 0.0000 |  | 0.0000 | 0 |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | 3 | 0.6888 | -0.0078 | 0.1146 | 0.1317 | 0.6793 | 1 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | 3 | 0.6888 | -0.0078 | 0.1163 | 0.1275 | 0.5725 | 1 |
| KMNIST | PureKAN-ABRBF-linear-learnWidth-AdamW | 3 | 0.6816 | -0.0007 | 0.1174 | 0.1282 | 0.5896 | 1 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | 3 | 0.6947 | -0.0137 | 0.1167 | 0.1223 | 0.1963 | 1 |
| KMNIST | PureKAN-BaseOnly-linear-AdamW | 3 | 0.6068 | 0.0742 | 0.0000 | 0.1174 | 209738729688856.3438 | 0 |
| KMNIST | PureKAN-BaseOnly-silu-AdamW | 3 | 0.6471 | 0.0339 | 0.0000 | 0.0931 | 127656950632731.1094 | 0 |
| KMNIST | PureKAN-RBF-quantileCenters-AdamW | 3 | 0.6719 | 0.0091 | 0.1829 | 0.1829 | 0.0000 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | 3 | 0.6810 | 0.0000 | 0.1200 | 0.1200 | 0.0000 | 1 |
| MNIST | MLP-AdamW | 3 | 0.9076 | -0.0319 | 0.0000 |  | 0.0000 | 0 |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | 3 | 0.8997 | -0.0241 | 0.1143 | 0.1312 | 0.6783 | 1 |
| MNIST | PureKAN-ABRBF-linear-AdamW | 3 | 0.8900 | -0.0143 | 0.1153 | 0.1260 | 0.5610 | 1 |
| MNIST | PureKAN-ABRBF-linear-learnWidth-AdamW | 3 | 0.8874 | -0.0117 | 0.1162 | 0.1269 | 0.5761 | 1 |
| MNIST | PureKAN-ABRBF-silu-AdamW | 3 | 0.8945 | -0.0189 | 0.1184 | 0.1237 | 0.1998 | 1 |
| MNIST | PureKAN-BaseOnly-linear-AdamW | 3 | 0.8796 | -0.0039 | 0.0000 | 0.1178 | 218147297329372.8125 | 0 |
| MNIST | PureKAN-BaseOnly-silu-AdamW | 3 | 0.8854 | -0.0098 | 0.0000 | 0.0921 | 135945993211534.2969 | 0 |
| MNIST | PureKAN-RBF-quantileCenters-AdamW | 3 | 0.8633 | 0.0124 | 0.1786 | 0.1786 | 0.0000 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | 3 | 0.8757 | 0.0000 | 0.1219 | 0.1219 | 0.0000 | 1 |

P1 verdict:

```text
AB-RBF is architecture-positive.
ABRBF-linear+silu is the strongest all-around short-run architecture:
  MNIST and Fashion improve over RBFOnly.
  KMNIST improves in seed0 compact P7 and remains competitive in P1.

BaseOnly is informative:
  It can approach RBF/ABRBF on MNIST/Fashion, but collapses relative to AB-RBF on KMNIST.
  This means the base path is necessary but not sufficient; the RBF residual still contributes.

Quantile centers and learnWidth alone are not stable positives.
```

P1 all-dataset architecture positives:

```text
PureKAN-ABRBF-linear+silu-AdamW, PureKAN-ABRBF-linear-AdamW, PureKAN-ABRBF-linear-learnWidth-AdamW, PureKAN-ABRBF-silu-AdamW
```

## P2 Eigenmode / Basis-Use Audit

| dataset | method | high coeff | high grad | base mode | eig cond | shift |
|---|---|---|---|---|---|---|
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | 0.2547 | 0.0004 | 0.3333 | 78.7 | 1 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | 0.2293 | 0.0001 | 0.2560 | 78.7 | 1 |
| Fashion-MNIST | PureKAN-ABRBF-linear-learnWidth-AdamW | 0.2301 | 0.0002 | 0.2527 | 78.7 | 1 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | 0.2508 | 0.0002 | 0.0931 | 78.7 | 1 |
| Fashion-MNIST | PureKAN-BaseOnly-linear-AdamW | 0.0000 | 0.0000 | 1.0000 | 1.0 | 1 |
| Fashion-MNIST | PureKAN-BaseOnly-silu-AdamW | 0.0000 | 0.0000 | 1.0000 | 1.0 | 1 |
| Fashion-MNIST | PureKAN-RBF-quantileCenters-AdamW | 0.2871 | 0.0036 | 0.0000 | 78.7 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | 0.2689 | 0.0002 | 0.0000 | 78.7 | 1 |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | 0.2507 | 0.0005 | 0.3194 | 78.7 | 1 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | 0.2241 | 0.0002 | 0.2434 | 78.7 | 1 |
| KMNIST | PureKAN-ABRBF-linear-learnWidth-AdamW | 0.2248 | 0.0003 | 0.2408 | 78.7 | 1 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | 0.2439 | 0.0003 | 0.0874 | 78.7 | 1 |
| KMNIST | PureKAN-BaseOnly-linear-AdamW | 0.0000 | 0.0000 | 1.0000 | 1.0 | 1 |
| KMNIST | PureKAN-BaseOnly-silu-AdamW | 0.0000 | 0.0000 | 1.0000 | 1.0 | 1 |
| KMNIST | PureKAN-RBF-quantileCenters-AdamW | 0.2702 | 0.0047 | 0.0000 | 78.7 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | 0.2629 | 0.0003 | 0.0000 | 78.7 | 1 |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | 0.2515 | 0.0005 | 0.3342 | 78.7 | 1 |
| MNIST | PureKAN-ABRBF-linear-AdamW | 0.2285 | 0.0001 | 0.2520 | 78.7 | 1 |
| MNIST | PureKAN-ABRBF-linear-learnWidth-AdamW | 0.2292 | 0.0002 | 0.2485 | 78.7 | 1 |
| MNIST | PureKAN-ABRBF-silu-AdamW | 0.2407 | 0.0002 | 0.0983 | 78.7 | 1 |
| MNIST | PureKAN-BaseOnly-linear-AdamW | 0.0000 | 0.0000 | 1.0000 | 1.0 | 1 |
| MNIST | PureKAN-BaseOnly-silu-AdamW | 0.0000 | 0.0000 | 1.0000 | 1.0 | 1 |
| MNIST | PureKAN-RBF-quantileCenters-AdamW | 0.2504 | 0.0045 | 0.0000 | 78.7 | 1 |
| MNIST | PureKAN-RBFOnly-AdamW | 0.2578 | 0.0003 | 0.0000 | 78.7 | 1 |

P2 verdict:

```text
AB-RBF shifts a visible fraction of edge energy into explicit base modes.
This confirms the v4.9/v5.0 interpretation: RBFOnly uses coefficient modes to carry low-order structure.
However high-mode/task-pressure is not fully eliminated, especially when the residual remains needed for KMNIST.
```

## P3 Split-Metric Functional Update

| dataset | method | runs | acc | gap vs ABRBF-A | hold/A | phiR red | base share | pass |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | ABRBF-AdamW | 3 | 0.7754 | 0.0000 | 1.0000 | 0.0000 | 0.1918 | 1 |
| Fashion-MNIST | ABRBF-allAdamW-edgeOnly | 3 | 0.7754 | 0.0000 | 1.0000 | 0.0000 | 0.1918 | 1 |
| Fashion-MNIST | ABRBF-baseAdam-rbfD6 | 3 | 0.7734 | 0.0020 | 0.9408 | 0.1715 | 0.9892 | 1 |
| Fashion-MNIST | ABRBF-baseAdam-rbfFCAdam-dataSob | 3 | 0.7747 | 0.0007 | 0.9403 | 0.1719 | 0.9802 | 1 |
| Fashion-MNIST | ABRBF-baseAdam-rbfRelaxedNFSRefresh | 3 | 0.7734 | 0.0020 | 0.9408 | 0.1715 | 0.9890 | 1 |
| Fashion-MNIST | ABRBF-baseAdam-rbfUFULL | 3 | 0.7747 | 0.0007 | 0.9412 | 0.1705 | 0.9707 | 1 |
| Fashion-MNIST | ABRBF-baseFCAdam-rbfUFULL | 3 | 0.7839 | -0.0085 | 0.9011 | 0.1745 | 0.9179 | 1 |
| Fashion-MNIST | ABRBF-baseFrozen-rbfUFULL | 3 | 0.6048 | 0.1706 | 0.3929 | 0.1756 | 0.0000 | 0 |
| Fashion-MNIST | ABRBF-baseOnlyAdam-rbfFrozen | 3 | 0.7715 | 0.0039 | 0.9414 | 0.1724 | 1.0000 | 1 |
| KMNIST | ABRBF-AdamW | 3 | 0.6517 | 0.0000 | 1.0000 | 0.0000 | 0.2023 | 1 |
| KMNIST | ABRBF-allAdamW-edgeOnly | 3 | 0.6517 | 0.0000 | 1.0000 | 0.0000 | 0.2023 | 1 |
| KMNIST | ABRBF-baseAdam-rbfD6 | 3 | 0.6302 | 0.0215 | 0.8967 | 0.1743 | 0.9891 | 1 |
| KMNIST | ABRBF-baseAdam-rbfFCAdam-dataSob | 3 | 0.6361 | 0.0156 | 0.8980 | 0.1741 | 0.9800 | 1 |
| KMNIST | ABRBF-baseAdam-rbfRelaxedNFSRefresh | 3 | 0.6296 | 0.0221 | 0.8967 | 0.1744 | 0.9889 | 1 |
| KMNIST | ABRBF-baseAdam-rbfUFULL | 3 | 0.6393 | 0.0124 | 0.8954 | 0.1739 | 0.9702 | 1 |
| KMNIST | ABRBF-baseFCAdam-rbfUFULL | 3 | 0.6335 | 0.0182 | 0.8465 | 0.1732 | 0.9172 | 1 |
| KMNIST | ABRBF-baseFrozen-rbfUFULL | 3 | 0.2682 | 0.3835 | 0.2502 | 0.1785 | 0.0000 | 0 |
| KMNIST | ABRBF-baseOnlyAdam-rbfFrozen | 3 | 0.6361 | 0.0156 | 0.8993 | 0.1734 | 1.0000 | 1 |
| MNIST | ABRBF-AdamW | 3 | 0.8626 | 0.0000 | 1.0000 | 0.0000 | 0.2014 | 1 |
| MNIST | ABRBF-allAdamW-edgeOnly | 3 | 0.8626 | 0.0000 | 1.0000 | 0.0000 | 0.2014 | 1 |
| MNIST | ABRBF-baseAdam-rbfD6 | 3 | 0.8724 | -0.0098 | 0.9653 | 0.1908 | 0.9859 | 1 |
| MNIST | ABRBF-baseAdam-rbfFCAdam-dataSob | 3 | 0.8665 | -0.0039 | 0.9659 | 0.1910 | 0.9743 | 1 |
| MNIST | ABRBF-baseAdam-rbfRelaxedNFSRefresh | 3 | 0.8724 | -0.0098 | 0.9652 | 0.1910 | 0.9857 | 1 |
| MNIST | ABRBF-baseAdam-rbfUFULL | 3 | 0.8711 | -0.0085 | 0.9680 | 0.1883 | 0.9626 | 1 |
| MNIST | ABRBF-baseFCAdam-rbfUFULL | 3 | 0.8743 | -0.0117 | 0.9024 | 0.1961 | 0.8971 | 1 |
| MNIST | ABRBF-baseFrozen-rbfUFULL | 3 | 0.4954 | 0.3672 | 0.2543 | 0.1949 | 0.0000 | 0 |
| MNIST | ABRBF-baseOnlyAdam-rbfFrozen | 3 | 0.8737 | -0.0111 | 0.9646 | 0.1917 | 1.0000 | 1 |

P3 verdict:

```text
Split functional update is a real improvement over residual-only training.
Moving base with Adam-like dynamics is essential:
  baseFrozen-rbfUFULL fails hard on all datasets.

Several baseAdam + rbf functional variants keep decent holdout descent and reduce residual phi.
But the all-dataset gate is not clean, mainly because KMNIST accuracy lags ABRBF-AdamW.
```

P3 all-dataset survivors:

```text
ABRBF-allAdamW-edgeOnly, ABRBF-baseAdam-rbfD6, ABRBF-baseAdam-rbfFCAdam-dataSob, ABRBF-baseAdam-rbfRelaxedNFSRefresh, ABRBF-baseAdam-rbfUFULL, ABRBF-baseFCAdam-rbfUFULL, ABRBF-baseOnlyAdam-rbfFrozen
```

## P4 Relaxed NFS Projection

| dataset | teacher | variant | acc drop | KL | phiR red | curvR red | proj/raw | pass |
|---|---|---|---|---|---|---|---|---|
| MNIST | PureKAN-RBFOnly-AdamW | Heuristic-NFS-role-block | 0.0000 | 0.00009 | 0.0228 | 0.0628 | 0.0000 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit-hidden | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-logit-hidden-muLow | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-logit-hidden-muMed | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-logit-hidden-muHigh | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-role-block | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-role-cycle | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-rbfResidualOnly | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-baseFrozen-rbfOnly | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | Heuristic-NFS-role-block | 0.0020 | 0.00065 | 0.0251 | 0.0647 | 0.0000 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit-hidden | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-logit-hidden-muLow | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-logit-hidden-muMed | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-logit-hidden-muHigh | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-role-block | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5182 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-role-cycle | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5182 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-rbfResidualOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5182 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-baseFrozen-rbfOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5182 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | Heuristic-NFS-role-block | 0.0000 | 0.00015 | 0.0247 | 0.0643 | 0.0000 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit-hidden | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-logit-hidden-muLow | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-logit-hidden-muMed | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-logit-hidden-muHigh | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-role-block | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-role-cycle | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-rbfResidualOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-baseFrozen-rbfOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | Heuristic-NFS-role-block | -0.0020 | 0.00042 | 0.0227 | 0.0654 | 0.0000 | 0 |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | ExactNFS-logit | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | ExactNFS-logit-hidden | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-logit-hidden-muLow | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-logit-hidden-muMed | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-logit-hidden-muHigh | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-role-block | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-role-cycle | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-rbfResidualOnly | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-baseFrozen-rbfOnly | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | Heuristic-NFS-role-block | 0.0000 | 0.00003 | 0.0238 | 0.0624 | 0.0000 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit | 0.0000 | 0.00000 | -0.0000 | 0.0000 | 1.0000 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit-hidden | 0.0000 | -0.00000 | -0.0000 | 0.0000 | 1.0000 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-logit-hidden-muLow | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-logit-hidden-muMed | 0.0000 | -0.00000 | -0.0000 | 0.0000 | 0.9999 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-logit-hidden-muHigh | 0.0000 | 0.00000 | -0.0000 | 0.0000 | 0.9999 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-role-block | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-role-cycle | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-rbfResidualOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-baseFrozen-rbfOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | Heuristic-NFS-role-block | 0.0020 | 0.00049 | 0.0224 | 0.0643 | 0.0000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit-hidden | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-logit-hidden-muLow | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-logit-hidden-muMed | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-logit-hidden-muHigh | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-role-block | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-role-cycle | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-rbfResidualOnly | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-baseFrozen-rbfOnly | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | Heuristic-NFS-role-block | -0.0039 | 0.00010 | 0.0242 | 0.0636 | 0.0000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit-hidden | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-logit-hidden-muLow | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-logit-hidden-muMed | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-logit-hidden-muHigh | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-role-block | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5182 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-role-cycle | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5182 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-rbfResidualOnly | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5182 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-baseFrozen-rbfOnly | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5182 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | Heuristic-NFS-role-block | 0.0039 | 0.00020 | 0.0210 | 0.0645 | 0.0000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | ExactNFS-logit | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | ExactNFS-logit-hidden | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-logit-hidden-muLow | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-logit-hidden-muMed | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-logit-hidden-muHigh | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-role-block | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5185 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-role-cycle | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5185 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-rbfResidualOnly | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5185 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-baseFrozen-rbfOnly | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.5185 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | Heuristic-NFS-role-block | -0.0020 | 0.00007 | 0.0255 | 0.0620 | 0.0000 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit-hidden | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-logit-hidden-muLow | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-logit-hidden-muMed | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-logit-hidden-muHigh | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-role-block | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-role-cycle | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-rbfResidualOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | RelaxedNFS-baseFrozen-rbfOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5183 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | Heuristic-NFS-role-block | 0.0000 | 0.00039 | 0.0240 | 0.0642 | 0.0000 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit-hidden | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-logit-hidden-muLow | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-logit-hidden-muMed | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-logit-hidden-muHigh | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-role-block | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-role-cycle | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-rbfResidualOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | RelaxedNFS-baseFrozen-rbfOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | Heuristic-NFS-role-block | 0.0000 | 0.00012 | 0.0277 | 0.0638 | 0.0000 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit-hidden | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-logit-hidden-muLow | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-logit-hidden-muMed | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-logit-hidden-muHigh | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-role-block | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-role-cycle | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-rbfResidualOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | RelaxedNFS-baseFrozen-rbfOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5184 | 0 |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | Heuristic-NFS-role-block | -0.0020 | 0.00023 | 0.0245 | 0.0642 | 0.0000 | 0 |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | ExactNFS-logit | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | ExactNFS-logit-hidden | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-logit-hidden-muLow | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-logit-hidden-muMed | 0.0000 | -0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-logit-hidden-muHigh | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.9999 | 0 |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-role-block | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5186 | 0 |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-role-cycle | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5186 | 0 |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-rbfResidualOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5186 | 0 |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | RelaxedNFS-baseFrozen-rbfOnly | 0.0000 | 0.00000 | 0.0000 | 0.0000 | 0.5186 | 0 |

P4 verdict:

```text
Relaxed NFS is numerically stable and no longer overconstrained in the same way as Exact NFS:
  projected_over_raw is often around 0.5 to 1.0.

But the actual residual-geometry movement is still near zero under the current trust scale.
Heuristic NFS keeps the best practical geometry movement, around 2-3% phi_rbf reduction,
but this is below the planned >10% residual phi gate.

Therefore P5/P6 are not expanded.
```

P4 all-dataset survivors:

```text
none
```

## P5 / P6 Decision

```text
P5 learn -> smooth -> refresh: not run
Reason: P4 produced no all-dataset relaxed NFS survivor.

P6 event-driven TAN v2: not run
Reason: P5 was not reached.
```

## P7 Capacity / Basis Follow-Up

| dataset | edge | hidden | basis | depth | acc | phi_rbf | rank | base/RBF |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | ABRBF-linear | 64 | 16 | 4 | 0.8008 | 0.1114 | 9.1229 | 0.5514 |
| Fashion-MNIST | ABRBF-linear | 96 | 24 | 4 | 0.7930 | 0.1370 | 9.5165 | 0.6043 |
| Fashion-MNIST | ABRBF-linear+silu | 64 | 16 | 4 | 0.7852 | 0.1107 | 9.0518 | 0.6573 |
| Fashion-MNIST | ABRBF-linear+silu | 96 | 24 | 4 | 0.7734 | 0.1358 | 9.5579 | 0.6708 |
| Fashion-MNIST | ABRBF-linear-learnWidth | 96 | 24 | 4 | 0.7910 | 0.1379 | 9.6156 | 0.6195 |
| Fashion-MNIST | ABRBF-linear-quantileCenters | 96 | 24 | 4 | 0.7656 | 0.1934 | 7.0925 | 0.3132 |
| Fashion-MNIST | ABRBF-silu | 64 | 16 | 4 | 0.7832 | 0.1129 | 10.8404 | 0.1572 |
| Fashion-MNIST | ABRBF-silu | 96 | 24 | 4 | 0.7793 | 0.1382 | 9.0402 | 0.1274 |
| Fashion-MNIST | RBFOnly | 64 | 16 | 4 | 0.7891 | 0.1171 | 10.0688 | 0.0000 |
| KMNIST | ABRBF-linear | 64 | 16 | 4 | 0.6855 | 0.1122 | 11.6271 | 0.5465 |
| KMNIST | ABRBF-linear | 96 | 24 | 4 | 0.6836 | 0.1471 | 9.7898 | 0.5215 |
| KMNIST | ABRBF-linear+silu | 64 | 16 | 4 | 0.7168 | 0.1133 | 10.9519 | 0.6333 |
| KMNIST | ABRBF-linear+silu | 96 | 24 | 4 | 0.7051 | 0.1415 | 10.6869 | 0.6009 |
| KMNIST | ABRBF-linear-learnWidth | 96 | 24 | 4 | 0.6836 | 0.1482 | 9.8360 | 0.5409 |
| KMNIST | ABRBF-linear-quantileCenters | 96 | 24 | 4 | 0.6367 | 0.2437 | 8.4478 | 0.2115 |
| KMNIST | ABRBF-silu | 64 | 16 | 4 | 0.6934 | 0.1137 | 11.8708 | 0.2022 |
| KMNIST | ABRBF-silu | 96 | 24 | 4 | 0.6797 | 0.1433 | 11.0848 | 0.1064 |
| KMNIST | RBFOnly | 64 | 16 | 4 | 0.6758 | 0.1197 | 11.5154 | 0.0000 |
| MNIST | ABRBF-linear | 64 | 16 | 4 | 0.8750 | 0.1119 | 10.4300 | 0.5697 |
| MNIST | ABRBF-linear | 96 | 24 | 4 | 0.8438 | 0.1425 | 9.7568 | 0.5222 |
| MNIST | ABRBF-linear+silu | 64 | 16 | 4 | 0.8633 | 0.1114 | 11.1287 | 0.6716 |
| MNIST | ABRBF-linear+silu | 96 | 24 | 4 | 0.8691 | 0.1465 | 9.9446 | 0.5962 |
| MNIST | ABRBF-linear-learnWidth | 96 | 24 | 4 | 0.8379 | 0.1432 | 9.7848 | 0.5334 |
| MNIST | ABRBF-linear-quantileCenters | 96 | 24 | 4 | 0.8008 | 0.2454 | 5.9702 | 0.1500 |
| MNIST | ABRBF-silu | 64 | 16 | 4 | 0.8750 | 0.1172 | 11.7127 | 0.1817 |
| MNIST | ABRBF-silu | 96 | 24 | 4 | 0.8672 | 0.1460 | 9.6207 | 0.1151 |
| MNIST | RBFOnly | 64 | 16 | 4 | 0.8105 | 0.1204 | 10.7913 | 0.0000 |

P7 verdict:

```text
Compact capacity follow-up supports the architecture signal.
ABRBF-linear+silu h64/b16/d4 is strong on KMNIST seed0 and improves MNIST over RBFOnly.
Scaling to h96/b24 does not automatically improve geometry; phi_rbf often rises.
Quantile centers are not a clean fix in this setup.
```

## Failure Diagnosis

Generated:

```text
failure_table.csv
p1_architecture_gate_summary.csv
p2_eigenmode_gate_summary.csv
p3_split_functional_gate_summary.csv
p4_relaxed_nfs_gate_summary.csv
p7_capacity_gate_summary.csv
```

Key labels:

```text
F1_base_path_unused:
  Mostly false for AB-RBF. Base/RBF contribution is nontrivial.

F2_base_path_helps_accuracy_but_geometry_bad:
  True for several AB-RBF rows under total geometry, but split geometry shows this is partly expected.

F3_residual_high_mode_pressure_persists:
  Partly true. AB-RBF helps, but does not remove residual pressure on KMNIST.

F4_exact_nfs_overconstrained:
  Still true for strict Exact NFS.

F6_relaxed_nfs_no_geometry_gain:
  New blocker. Relaxed projection keeps useful gradient mass, but current step/trust gives almost no residual phi reduction.

F10_functional_candidate_lags_ABRBF_AdamW:
  Main P3 blocker, especially on KMNIST.
```

## Required Artifacts

Written under `results/v5_0/`:

```text
runs.csv
curves.csv
edge_decomposition_audit.csv
eigenmode_audit.csv
basis_occupancy_audit.csv
nfs_projection_audit.csv
failure_table.csv
aggregate_decision.json
figures/
```

## Final Decision

```text
DG-KAN v5.0 status:
  stop_after_p4_no_relaxed_nfs_survivor

What improved:
  Edge decomposition is clearly useful.
  AB-RBF, especially linear+silu / silu variants, improves the PureKAN architecture frontier.
  Split geometry gives a more honest view: base derivative is not the same as residual roughness.
  Split functional update confirms base-path task dynamics are essential.

What failed / remains open:
  Split functional candidates still lag ABRBF-AdamW on the hardest dataset.
  Relaxed NFS no longer has zero projected direction, but still fails to create enough residual smoothing.
  P5/P6/P8/P9 expansion is not justified.

Conclusion:
  v5.0 supports architecture success and partial optimizer progress.
  RBF-only was a bottleneck, but the current residual smoothing operator is still too weak.
  Next work should keep AB-RBF as the PureKAN default and redesign relaxed residual smoothing
  with a stronger accepted-step controller rather than returning to exact nullspace projection.
```
