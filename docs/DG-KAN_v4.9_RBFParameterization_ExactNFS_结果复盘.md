# DG-KAN v4.9 RBF Parameterization / Exact NFS 结果复盘

本轮依据 `docs/DG-KAN_v4.9_RBFParameterization_ExactNFS_下一步实验计划.md`。目标是把 PureKAN functional failure 拆成三个问题：RBF-only 参数化、RBF eigenmode 使用方式、以及 v4.8 NFS 是否需要真正的 Jacobian-nullspace projection。

## Run Inventory

| stage | rows | errors |
|---|---:|---:|
| P0 code audit | 21 | 0 |
| P1 architecture frontier | 45 | 0 |
| P2 eigenmode audit | 270 | 0 |
| P3 exact NFS projection | 45 | 0 |
| P4 TAN one-cycle v2 | 0 | 0 |
| P5 event TAN cycles | 0 | 0 |
| P6 capacity/basis expansion | 24 | 0 |

## Code / Config Changes

```text
experiments/run_gafu_v49.py
  Added strict edge-only PureKAN parameterizations:
    RBFOnly
    ABRBF-linear
    ABRBF-silu
    RBF-learnWidth
    ABRBF-linear-learnWidth
    RBF-quantileCenters smoke
  Added local base/RBF contribution audit, width/basis coverage audit, and eigenmode energy audit.
  Added Exact NFS sketch projection:
    explicit Jacobian rows for logits / hidden sketch / margin constraints
    KKT direct solve in coefficient space
    identity and Sobolev-diagonal metrics

experiments/analyze_gafu_v49.py
  Generates gate summaries, failure table, aggregate_decision.json,
  SVG diagnostics, log entry, and this replay.
```

## P0 Code Audit

| rows | errors | pass | max nonKAN | min coverage | max rollback | max KKT |
|---|---|---|---|---|---|---|
| 21 | 0 | true | 0.0000 | 1.0000 | 0.0000 | 0.000001 |

P0 verdict: pass. 新 edge 参数化仍保持 strict edge-only：learnable non-KAN params 为 0，functional coverage 为 1.0，rollback 为 0。

## P1 Architecture Frontier

| dataset | method | runs | acc | std | gap vs RBF | phi red | J red | base/RBF | positive |
|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | 3 | 0.7949 | 0.0042 | -0.0007 | -0.0597 | -30.6799 | 0.5480 | 1 |
| Fashion-MNIST | PureKAN-ABRBF-linear-learnWidth-AdamW | 3 | 0.7930 | 0.0057 | 0.0013 | -0.0682 | -630.9040 | 0.5651 | 1 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | 3 | 0.7969 | 0.0193 | -0.0026 | -0.0168 | -1.5237 | 0.1647 | 1 |
| Fashion-MNIST | PureKAN-RBF-learnWidth-AdamW | 3 | 0.7943 | 0.0066 | 0.0000 | -0.0145 | -0.2339 | 0.0000 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | 3 | 0.7943 | 0.0097 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | 3 | 0.6888 | 0.0198 | -0.0078 | -0.0558 | -29.6148 | 0.5725 | 1 |
| KMNIST | PureKAN-ABRBF-linear-learnWidth-AdamW | 3 | 0.6816 | 0.0181 | -0.0007 | -0.0661 | -3.1688 | 0.5896 | 1 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | 3 | 0.6947 | 0.0082 | -0.0137 | -0.0198 | -0.5194 | 0.1963 | 1 |
| KMNIST | PureKAN-RBF-learnWidth-AdamW | 3 | 0.6751 | 0.0239 | 0.0059 | -0.0110 | 0.5166 | 0.0000 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | 3 | 0.6810 | 0.0166 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |
| MNIST | PureKAN-ABRBF-linear-AdamW | 3 | 0.8900 | 0.0072 | -0.0143 | -0.0313 | -2.8761 | 0.5610 | 1 |
| MNIST | PureKAN-ABRBF-linear-learnWidth-AdamW | 3 | 0.8874 | 0.0088 | -0.0117 | -0.0401 | -2.2144 | 0.5761 | 1 |
| MNIST | PureKAN-ABRBF-silu-AdamW | 3 | 0.8945 | 0.0105 | -0.0189 | -0.0169 | -56.2586 | 0.1998 | 1 |
| MNIST | PureKAN-RBF-learnWidth-AdamW | 3 | 0.8757 | 0.0157 | 0.0000 | -0.0070 | 0.3571 | 0.0000 | 1 |
| MNIST | PureKAN-RBFOnly-AdamW | 3 | 0.8757 | 0.0133 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |

P1 verdict:

```text
AB-RBF is architecture-positive.
ABRBF-silu improves mean accuracy on all three datasets vs RBFOnly:
  MNIST +1.88 points
  Fashion +0.26 points
  KMNIST +1.37 points

ABRBF-linear also improves MNIST/KMNIST and keeps Fashion roughly tied.
learnWidth alone is not a clear positive signal.
The base path is actively used: ABRBF-linear base/RBF norm is about 0.55-0.59.
```

P1 all-dataset architecture positives:

```text
PureKAN-ABRBF-linear-AdamW, PureKAN-ABRBF-linear-learnWidth-AdamW, PureKAN-ABRBF-silu-AdamW
```

## P2 Eigenmode Audit

| dataset | method | high coeff | base mode | grad high | eig cond | shift |
|---|---|---|---|---|---|---|
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | 0.2293 | 0.2560 | 0.0001 | 78.7 | 1 |
| Fashion-MNIST | PureKAN-ABRBF-linear-learnWidth-AdamW | 0.2301 | 0.2527 | 0.0002 | 78.7 | 1 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | 0.2508 | 0.0931 | 0.0002 | 78.7 | 1 |
| Fashion-MNIST | PureKAN-RBF-learnWidth-AdamW | 0.2697 | 0.0000 | 0.0003 | 78.7 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | 0.2689 | 0.0000 | 0.0002 | 78.7 | 1 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | 0.2241 | 0.2434 | 0.0002 | 78.7 | 1 |
| KMNIST | PureKAN-ABRBF-linear-learnWidth-AdamW | 0.2248 | 0.2408 | 0.0003 | 78.7 | 1 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | 0.2439 | 0.0874 | 0.0003 | 78.7 | 1 |
| KMNIST | PureKAN-RBF-learnWidth-AdamW | 0.2640 | 0.0000 | 0.0004 | 78.7 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | 0.2629 | 0.0000 | 0.0003 | 78.7 | 1 |
| MNIST | PureKAN-ABRBF-linear-AdamW | 0.2285 | 0.2520 | 0.0001 | 78.7 | 1 |
| MNIST | PureKAN-ABRBF-linear-learnWidth-AdamW | 0.2292 | 0.2485 | 0.0002 | 78.7 | 1 |
| MNIST | PureKAN-ABRBF-silu-AdamW | 0.2407 | 0.0983 | 0.0002 | 78.7 | 1 |
| MNIST | PureKAN-RBF-learnWidth-AdamW | 0.2586 | 0.0000 | 0.0003 | 78.7 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | 0.2578 | 0.0000 | 0.0003 | 78.7 | 1 |

P2 verdict:

```text
RBFOnly keeps about 26-27% of coefficient energy in the high Sobolev-eigen band.
ABRBF-linear shifts about 24-26% of coefficient energy into explicit base modes
and reduces high-mode coefficient energy to about 22-23%.
ABRBF-silu uses a smaller base channel but still reduces high-mode pressure.

This supports the v4.9 hypothesis:
  fixed-grid RBF-only was forcing low-order structure through RBF coefficient modes.
```

## P3 Exact NFS Projection

| dataset | teacher | variant | metric | acc drop | phi red | J red | KL | KKT | pass |
|---|---|---|---|---|---|---|---|---|---|
| MNIST | PureKAN-RBFOnly-AdamW | Heuristic-NFS-role-block | identity | 0.0000 | 0.0290 | 0.6509 | 0.00009 |  | 1 |
| MNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit | identity | 0.0000 | 0.0000 | 0.0001 | -0.00000 | 0.000000 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit-hidden | identity | 0.0000 | 0.0000 | 0.0002 | 0.00000 | 0.000001 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit-hidden | sobolev_diag | 0.0000 | 0.0000 | 0.0001 | 0.00000 | 0.000000 | 0 |
| MNIST | PureKAN-RBFOnly-AdamW | ExactNFS-margin | identity | 0.0000 | 0.0000 | 0.0002 | -0.00000 | 0.000000 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | Heuristic-NFS-role-block | identity | 0.0020 | 0.0399 | 0.1721 | 0.00065 |  | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit | identity | 0.0000 | 0.0000 | 0.0004 | 0.00000 | 0.000000 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit-hidden | identity | 0.0000 | 0.0000 | 0.0005 | 0.00000 | 0.000001 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit-hidden | sobolev_diag | 0.0000 | 0.0000 | 0.0003 | 0.00000 | 0.000001 | 0 |
| MNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-margin | identity | 0.0000 | 0.0000 | 0.0005 | 0.00000 | 0.000000 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | Heuristic-NFS-role-block | identity | 0.0000 | 0.0300 | 0.7110 | 0.00015 |  | 1 |
| MNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit | identity | 0.0000 | 0.0000 | 0.0001 | -0.00000 | 0.000000 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit-hidden | identity | 0.0000 | 0.0000 | 0.0001 | 0.00000 | 0.000001 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit-hidden | sobolev_diag | 0.0000 | 0.0000 | 0.0001 | 0.00000 | 0.000000 | 0 |
| MNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-margin | identity | 0.0000 | 0.0000 | 0.0001 | 0.00000 | 0.000000 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | Heuristic-NFS-role-block | identity | 0.0000 | 0.0306 | 0.3500 | 0.00003 |  | 1 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit | identity | 0.0000 | 0.0000 | 0.0000 | 0.00000 | 0.000000 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit-hidden | identity | 0.0000 | 0.0000 | 0.0000 | -0.00000 | 0.000001 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit-hidden | sobolev_diag | 0.0000 | 0.0000 | 0.0000 | 0.00000 | 0.000000 | 0 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | ExactNFS-margin | identity | 0.0000 | 0.0000 | 0.0000 | 0.00000 | 0.000000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | Heuristic-NFS-role-block | identity | 0.0020 | 0.0414 | -0.2019 | 0.00049 |  | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit | identity | 0.0000 | 0.0000 | 0.0000 | -0.00000 | 0.000000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit-hidden | identity | 0.0000 | 0.0000 | 0.0001 | -0.00000 | 0.000001 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit-hidden | sobolev_diag | 0.0000 | 0.0000 | 0.0000 | -0.00000 | 0.000000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-margin | identity | 0.0000 | 0.0000 | 0.0001 | 0.00000 | 0.000000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | Heuristic-NFS-role-block | identity | -0.0039 | 0.0313 | 0.5270 | 0.00010 |  | 1 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit | identity | 0.0000 | 0.0000 | 0.0001 | -0.00000 | 0.000000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit-hidden | identity | 0.0000 | 0.0000 | 0.0001 | -0.00000 | 0.000001 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit-hidden | sobolev_diag | 0.0000 | 0.0000 | 0.0000 | -0.00000 | 0.000000 | 0 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-margin | identity | 0.0000 | 0.0000 | 0.0001 | -0.00000 | 0.000000 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | Heuristic-NFS-role-block | identity | -0.0020 | 0.0305 | 0.4150 | 0.00007 |  | 1 |
| KMNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit | identity | 0.0000 | 0.0000 | -0.0001 | -0.00000 | 0.000000 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit-hidden | identity | 0.0000 | 0.0000 | -0.0001 | 0.00000 | 0.000001 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | ExactNFS-logit-hidden | sobolev_diag | 0.0000 | 0.0000 | 0.0000 | -0.00000 | 0.000000 | 0 |
| KMNIST | PureKAN-RBFOnly-AdamW | ExactNFS-margin | identity | 0.0000 | 0.0000 | -0.0001 | -0.00000 | 0.000000 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | Heuristic-NFS-role-block | identity | 0.0000 | 0.0389 | 0.3271 | 0.00039 |  | 1 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit | identity | 0.0000 | 0.0000 | 0.0013 | 0.00000 | 0.000000 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit-hidden | identity | 0.0000 | 0.0000 | 0.0013 | 0.00000 | 0.000001 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-logit-hidden | sobolev_diag | 0.0000 | 0.0000 | 0.0014 | 0.00000 | 0.000000 | 0 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | ExactNFS-margin | identity | 0.0000 | 0.0000 | 0.0017 | 0.00000 | 0.000000 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | Heuristic-NFS-role-block | identity | 0.0000 | 0.0309 | 0.4123 | 0.00012 |  | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit | identity | 0.0000 | 0.0000 | 0.0000 | -0.00000 | 0.000000 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit-hidden | identity | 0.0000 | 0.0000 | 0.0000 | -0.00000 | 0.000001 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-logit-hidden | sobolev_diag | 0.0000 | 0.0000 | 0.0000 | 0.00000 | 0.000000 | 0 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | ExactNFS-margin | identity | 0.0000 | 0.0000 | 0.0000 | 0.00000 | 0.000000 | 0 |

P3 verdict:

```text
Exact NFS is mathematically stable but too conservative.
KKT residual is typically around 1e-7 to 1e-6, and KL/logit drift is essentially zero,
but geometry reduction is also essentially zero.

Heuristic NFS-role-block remains useful and function-preserving, but it is not a true nullspace projection.
It gives small phi reductions and larger J reductions, reproducing the v4.8 local signal.
No ExactNFS variant has an all-dataset survivor.
```

P3 all-dataset survivors including heuristic:

```text
PureKAN-RBFOnly-AdamW|Heuristic-NFS-role-block|identity|logit_hidden
```

P3 exact-only survivors:

```text
none
```

## P4 / P5 Decision

```text
P4 TAN one-cycle v2: not run.
Reason: P3 produced no exact-NFS all-dataset survivor.

P5 event-driven TAN cycles: not run.
Reason: P4 was not reached.
```

## P6 Capacity / Basis Expansion

| dataset | edge | hidden | basis | depth | acc | phi | rank | base/RBF |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | ABRBF-linear | 96 | 24 | 4 | 0.7930 | 0.1881 | 9.5165 | 0.6043 |
| Fashion-MNIST | ABRBF-linear | 96 | 32 | 4 | 0.7559 | 0.2378 | 7.9458 | 0.5690 |
| Fashion-MNIST | ABRBF-linear-learnWidth | 96 | 24 | 4 | 0.7910 | 0.1895 | 9.6156 | 0.6195 |
| Fashion-MNIST | ABRBF-silu | 96 | 24 | 4 | 0.7793 | 0.1842 | 9.0402 | 0.1274 |
| Fashion-MNIST | RBF-learnWidth | 96 | 24 | 4 | 0.7617 | 0.1853 | 10.1644 | 0.0000 |
| Fashion-MNIST | RBFOnly | 64 | 16 | 4 | 0.7891 | 0.1700 | 10.0688 | 0.0000 |
| Fashion-MNIST | RBFOnly | 96 | 24 | 4 | 0.7578 | 0.1836 | 10.0119 | 0.0000 |
| Fashion-MNIST | RBFOnly | 96 | 32 | 2 | 0.7715 | 0.2316 | 9.6263 | 0.0000 |
| KMNIST | ABRBF-linear | 96 | 24 | 4 | 0.6836 | 0.2044 | 9.7898 | 0.5215 |
| KMNIST | ABRBF-linear | 96 | 32 | 4 | 0.6465 | 0.2492 | 8.8968 | 0.5334 |
| KMNIST | ABRBF-linear-learnWidth | 96 | 24 | 4 | 0.6836 | 0.2059 | 9.8360 | 0.5409 |
| KMNIST | ABRBF-silu | 96 | 24 | 4 | 0.6797 | 0.1912 | 11.0848 | 0.1064 |
| KMNIST | RBF-learnWidth | 96 | 24 | 4 | 0.6660 | 0.1934 | 11.0521 | 0.0000 |
| KMNIST | RBFOnly | 64 | 16 | 4 | 0.6758 | 0.1719 | 11.5154 | 0.0000 |
| KMNIST | RBFOnly | 96 | 24 | 4 | 0.6680 | 0.1916 | 10.8185 | 0.0000 |
| KMNIST | RBFOnly | 96 | 32 | 2 | 0.6777 | 0.2640 | 9.4357 | 0.0000 |
| MNIST | ABRBF-linear | 96 | 24 | 4 | 0.8438 | 0.2054 | 9.7568 | 0.5222 |
| MNIST | ABRBF-linear | 96 | 32 | 4 | 0.8398 | 0.2756 | 7.9179 | 0.4903 |
| MNIST | ABRBF-linear-learnWidth | 96 | 24 | 4 | 0.8379 | 0.2060 | 9.7848 | 0.5334 |
| MNIST | ABRBF-silu | 96 | 24 | 4 | 0.8672 | 0.1990 | 9.6207 | 0.1151 |
| MNIST | RBF-learnWidth | 96 | 24 | 4 | 0.8125 | 0.2022 | 9.1507 | 0.0000 |
| MNIST | RBFOnly | 64 | 16 | 4 | 0.8105 | 0.1767 | 10.7913 | 0.0000 |
| MNIST | RBFOnly | 96 | 24 | 4 | 0.8320 | 0.2019 | 8.9646 | 0.0000 |
| MNIST | RBFOnly | 96 | 32 | 2 | 0.8457 | 0.2899 | 7.6966 | 0.0000 |

P6 verdict:

```text
Capacity expansion confirms the parameterization signal but does not solve geometry.
ABRBF-silu h96/b24/d4 is the best MNIST point in this compact seed0 expansion.
ABRBF-linear improves/ties Fashion and KMNIST relative to larger RBF-only configs.
learnWidth alone again does not help.

However phi generally rises with larger capacity, so architecture alone does not expose
a smooth accurate frontier.
```

## Failure Diagnosis

Generated:

```text
failure_table.csv
p1_architecture_gate_summary.csv
p2_eigenmode_gate_summary.csv
p3_exact_nfs_gate_summary.csv
p6_capacity_gate_summary.csv
```

Diagnosis:

```text
F1_architecture_frontier_absent:
  false as a blanket diagnosis. AB-RBF is accuracy-positive and actively used.

F2_nfs_not_exact:
  true for the current exact projection as an optimizer primitive.
  The exact nullspace direction under logits/hidden/margin constraints is nearly zero.

F3_nfs_worse_than_heuristic:
  true. Heuristic NFS gives useful geometry movement; exact NFS preserves function too strictly.

F7_basis_coverage_bad:
  partly true. RBFOnly relies on high Sobolev modes more than AB-RBF.

F8_base_path_not_used:
  false. ABRBF-linear base/RBF norm is consistently nontrivial.
```

## Required Artifacts

Written under `results/v4_9/`:

```text
p0_code_audit.csv
p1_architecture_frontier.csv
p1_training_trace.csv
p2_eigenmode_audit.csv
p3_exact_nfs_projection.csv
p4_tan_one_cycle_v2.csv
p5_event_tan_cycles.csv
p6_capacity_basis_expansion_full.csv
p7_candidate_selection.csv
p8_confirm5.csv
p9_confirm10.csv
failure_table.csv
aggregate_decision.json
figures/
```

## Final Decision

```text
DG-KAN v4.9 status:
  stop_after_p3_no_exact_nfs_survivor

What improved:
  AB-RBF, especially ABRBF-silu, improves the PureKAN architecture frontier.
  Eigenmode audit confirms base paths reduce pressure on high RBF Sobolev modes.
  Exact NFS implementation is auditable and produces tiny KKT residuals.

What failed:
  Exact NFS found almost no useful geometry-moving direction under strict function constraints.
  Heuristic NFS remains better for practical smoothing, but it is not mathematically exact.
  Capacity/basis expansion improves accuracy in places but does not produce a smooth frontier.

Conclusion:
  PureKAN failure is partly parameterization-driven: RBF-only is too restrictive.
  But exact function-preserving smoothing is also too conservative in the current form.
  The next default should move from RBFOnly to AB-RBF for PureKAN probes, while redesigning NFS
  as a relaxed/trust-region projection instead of a hard Jacobian nullspace projection.
```
