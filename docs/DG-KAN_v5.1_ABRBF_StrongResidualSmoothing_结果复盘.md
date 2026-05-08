# DG-KAN v5.1 AB-RBF Strong Residual Smoothing 结果复盘

本轮依据 `docs/DG-KAN_v5.1_ABRBF_StrongResidualSmoothing_实验计划.md`。核心目标是把 v5.0 的 AB-RBF 架构正信号固化为默认 PureKAN edge primitive，并测试更强的 residual smoothing proposal / accepted-step controller 是否能把单步 NFS 信号推进到多步稳定训练。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 core smoke | 18 | 0 |
| P1 architecture confirm5 | 105 | 0 |
| P2 split geometry | 540 | 0 |
| P3 residual smoothing audit | 432 | 0 |
| P4 accepted smoothing | 18 | 0 |
| P5 learn-smooth-refresh | 1 | 0 |
| P6 event TAN | 1 | 0 |
| P7 split residual training | 135 | 0 |
| P8 capacity follow-up | 27 | 0 |

## Code / Config Changes

```text
experiments/dgkan_core.py
  Added core ABRBFDense with linear / silu / linear+silu / base-only edge paths.
  Added edge_named_params, base_named_params, and rbf_residual_named_params helpers.

experiments/run_gafu_v51.py
  Added P0 edge manifest, P1 5-seed AB-RBF confirm, P2 split geometry calibration,
  P3 S0-S7 residual smoothing proposals with C0-C5 controllers,
  P4 multi-step accepted smoothing, plus P7/P8 follow-up diagnostics.

experiments/analyze_gafu_v51.py
  Generates v5.1 gate summaries, failure table, aggregate decision, figures, and this replay.
```

## P0 Core Smoke

| rows | errors | manifest | max nonKAN | min edge cov | min base cov | min rbf cov | max rollback | pass |
|---|---|---|---|---|---|---|---|---|
| 18 | 0 | 162 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | true |

P0 verdict: pass. Core AB-RBF edge paths keep strict PureKAN non-KAN trainable parameters at zero, edge/base/RBF parameter coverage is auditable, and rollback is exact.

## P1 Architecture Confirm5

| dataset | method | runs | acc | std | gap vs RBF | phiR | base/RBF | P1 |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | 5 | 0.8063 | 0.0122 | 0.0000 | 0.1129 | 0.6592 | yes |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | 5 | 0.8039 | 0.0188 | 0.0023 | 0.1152 | 0.1643 | yes |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | 5 | 0.8008 | 0.0090 | 0.0055 | 0.1144 | 0.5517 | no |
| Fashion-MNIST | PureKAN-BaseOnly-linear-AdamW | 5 | 0.7914 | 0.0167 | 0.0148 | 0.0000 |  | no |
| Fashion-MNIST | PureKAN-BaseOnly-silu-AdamW | 5 | 0.7883 | 0.0081 | 0.0180 | 0.0000 |  | no |
| KMNIST | PureKAN-ABRBF-linear-AdamW | 5 | 0.6977 | 0.0190 | -0.0223 | 0.1168 | 0.5660 | yes |
| KMNIST | PureKAN-ABRBF-silu-AdamW | 5 | 0.6898 | 0.0118 | -0.0145 | 0.1163 | 0.1920 | yes |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | 5 | 0.6891 | 0.0103 | -0.0137 | 0.1142 | 0.6742 | yes |
| KMNIST | PureKAN-BaseOnly-silu-AdamW | 5 | 0.6406 | 0.0108 | 0.0348 | 0.0000 |  | no |
| KMNIST | PureKAN-BaseOnly-linear-AdamW | 5 | 0.5965 | 0.0249 | 0.0789 | 0.0000 |  | no |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | 5 | 0.8965 | 0.0105 | -0.0203 | 0.1144 | 0.6684 | yes |
| MNIST | PureKAN-ABRBF-silu-AdamW | 5 | 0.8938 | 0.0095 | -0.0176 | 0.1177 | 0.1962 | yes |
| MNIST | PureKAN-ABRBF-linear-AdamW | 5 | 0.8910 | 0.0062 | -0.0148 | 0.1160 | 0.5555 | yes |
| MNIST | PureKAN-BaseOnly-silu-AdamW | 5 | 0.8906 | 0.0146 | -0.0145 | 0.0000 |  | no |
| MNIST | PureKAN-BaseOnly-linear-AdamW | 5 | 0.8809 | 0.0169 | -0.0047 | 0.0000 |  | no |

P1 survivors:

```text
PureKAN-ABRBF-linear+silu-AdamW
PureKAN-ABRBF-silu-AdamW
```

P1 verdict: AB-RBF remains architecture-positive. The base path is consistently used; BaseOnly remains diagnostic rather than sufficient, especially on KMNIST.

## P2 Split Geometry / Eigenmode Audit

| dataset | method | high coeff | high grad | base mode | phiR | eig cond | shift |
|---|---|---|---|---|---|---|---|
| Fashion-MNIST | PureKAN-ABRBF-linear+silu-AdamW | 0.2587 | 0.0004 | 0.3253 | 0.1122 | 78.7 | 1 |
| Fashion-MNIST | PureKAN-ABRBF-linear-AdamW | 0.2321 | 0.0001 | 0.2504 | 0.1148 | 78.7 | 1 |
| Fashion-MNIST | PureKAN-ABRBF-silu-AdamW | 0.2515 | 0.0002 | 0.0939 | 0.1149 | 78.7 | 1 |
| Fashion-MNIST | PureKAN-BaseOnly-linear-AdamW | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0 | 1 |
| Fashion-MNIST | PureKAN-BaseOnly-silu-AdamW | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0 | 1 |
| Fashion-MNIST | PureKAN-RBFOnly-AdamW | 0.2688 | 0.0002 | 0.0000 | 0.1177 | 78.7 | 0 |
| KMNIST | PureKAN-ABRBF-linear+silu-AdamW | 0.2528 | 0.0005 | 0.3154 | 0.1138 | 78.7 | 1 |
| KMNIST | PureKAN-ABRBF-linear-AdamW | 0.2254 | 0.0002 | 0.2402 | 0.1171 | 78.7 | 1 |
| KMNIST | PureKAN-ABRBF-silu-AdamW | 0.2457 | 0.0003 | 0.0875 | 0.1167 | 78.7 | 1 |
| KMNIST | PureKAN-BaseOnly-linear-AdamW | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0 | 1 |
| KMNIST | PureKAN-BaseOnly-silu-AdamW | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0 | 1 |
| KMNIST | PureKAN-RBFOnly-AdamW | 0.2615 | 0.0003 | 0.0000 | 0.1197 | 78.7 | 0 |
| MNIST | PureKAN-ABRBF-linear+silu-AdamW | 0.2519 | 0.0005 | 0.3343 | 0.1143 | 78.7 | 1 |
| MNIST | PureKAN-ABRBF-linear-AdamW | 0.2277 | 0.0001 | 0.2535 | 0.1168 | 78.7 | 1 |
| MNIST | PureKAN-ABRBF-silu-AdamW | 0.2436 | 0.0002 | 0.0964 | 0.1178 | 78.7 | 1 |
| MNIST | PureKAN-BaseOnly-linear-AdamW | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0 | 1 |
| MNIST | PureKAN-BaseOnly-silu-AdamW | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0 | 1 |
| MNIST | PureKAN-RBFOnly-AdamW | 0.2570 | 0.0003 | 0.0000 | 0.1220 | 78.7 | 0 |

P2 verdict: AB-RBF shifts visible energy into explicit base modes, confirming that RBF-only had been carrying low-order structure through residual coefficient modes. The residual high-mode pressure is reduced but not removed.

## P3 Residual Smoothing Proposal Audit

| teacher | proposal | controller | dataset | pass | acc drop | phiR red | curvR red | KL | score |
|---|---|---|---|---|---|---|---|---|---|
| PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | C1-logit-kl-trust | MNIST | 1 | 0.0020 | 0.0897 | 0.7011 | 0.0044 | 0.4778 |
| PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | C2-logit-hidden-trust | MNIST | 1 | 0.0020 | 0.0897 | 0.7011 | 0.0044 | 0.4778 |
| PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | C3-logit-hidden-margin-trust | MNIST | 1 | 0.0020 | 0.0897 | 0.7011 | 0.0044 | 0.4778 |
| PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | C4-score-accepted | MNIST | 1 | 0.0020 | 0.0897 | 0.7011 | 0.0044 | 0.4778 |
| PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | C5-score-adaptive-eta | MNIST | 1 | 0.0020 | 0.0897 | 0.7011 | 0.0044 | 0.4778 |
| PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | C1-logit-kl-trust | MNIST | 1 | 0.0020 | 0.0897 | 0.7011 | 0.0044 | 0.4778 |
| PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | C2-logit-hidden-trust | MNIST | 1 | 0.0020 | 0.0897 | 0.7011 | 0.0044 | 0.4778 |
| PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | C3-logit-hidden-margin-trust | MNIST | 1 | 0.0020 | 0.0897 | 0.7011 | 0.0044 | 0.4778 |
| PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | C4-score-accepted | MNIST | 1 | 0.0020 | 0.0897 | 0.7011 | 0.0044 | 0.4778 |
| PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | C5-score-adaptive-eta | MNIST | 1 | 0.0020 | 0.0897 | 0.7011 | 0.0044 | 0.4778 |
| PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | C1-logit-kl-trust | Fashion-MNIST | 1 | 0.0020 | 0.0632 | 0.5484 | 0.0028 | 0.3539 |
| PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | C2-logit-hidden-trust | Fashion-MNIST | 1 | 0.0020 | 0.0632 | 0.5484 | 0.0028 | 0.3539 |
| PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | C3-logit-hidden-margin-trust | Fashion-MNIST | 1 | 0.0020 | 0.0632 | 0.5484 | 0.0028 | 0.3539 |
| PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | C4-score-accepted | Fashion-MNIST | 1 | 0.0020 | 0.0632 | 0.5484 | 0.0028 | 0.3539 |
| PureKAN-ABRBF-linear+silu-AdamW | S0-laplacian | C5-score-adaptive-eta | Fashion-MNIST | 1 | 0.0020 | 0.0632 | 0.5484 | 0.0028 | 0.3539 |
| PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | C1-logit-kl-trust | Fashion-MNIST | 1 | 0.0020 | 0.0632 | 0.5484 | 0.0028 | 0.3539 |
| PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | C2-logit-hidden-trust | Fashion-MNIST | 1 | 0.0020 | 0.0632 | 0.5484 | 0.0028 | 0.3539 |
| PureKAN-ABRBF-linear+silu-AdamW | S1-sobolev-grad | C3-logit-hidden-margin-trust | Fashion-MNIST | 1 | 0.0020 | 0.0632 | 0.5484 | 0.0028 | 0.3539 |

P3 all-dataset survivors:

```text
PureKAN-ABRBF-linear+silu-AdamW|S0-laplacian|C1-logit-kl-trust
PureKAN-ABRBF-linear+silu-AdamW|S0-laplacian|C2-logit-hidden-trust
PureKAN-ABRBF-linear+silu-AdamW|S0-laplacian|C3-logit-hidden-margin-trust
PureKAN-ABRBF-linear+silu-AdamW|S0-laplacian|C4-score-accepted
PureKAN-ABRBF-linear+silu-AdamW|S0-laplacian|C5-score-adaptive-eta
PureKAN-ABRBF-linear+silu-AdamW|S1-sobolev-grad|C1-logit-kl-trust
PureKAN-ABRBF-linear+silu-AdamW|S1-sobolev-grad|C2-logit-hidden-trust
PureKAN-ABRBF-linear+silu-AdamW|S1-sobolev-grad|C3-logit-hidden-margin-trust
PureKAN-ABRBF-linear+silu-AdamW|S1-sobolev-grad|C4-score-accepted
PureKAN-ABRBF-linear+silu-AdamW|S1-sobolev-grad|C5-score-adaptive-eta
PureKAN-ABRBF-linear+silu-AdamW|S6-heuristic-role-block|C0-line-search
PureKAN-ABRBF-linear+silu-AdamW|S6-heuristic-role-block|C1-logit-kl-trust
PureKAN-ABRBF-linear+silu-AdamW|S6-heuristic-role-block|C2-logit-hidden-trust
PureKAN-ABRBF-linear+silu-AdamW|S6-heuristic-role-block|C3-logit-hidden-margin-trust
PureKAN-ABRBF-linear+silu-AdamW|S6-heuristic-role-block|C4-score-accepted
PureKAN-ABRBF-linear+silu-AdamW|S6-heuristic-role-block|C5-score-adaptive-eta
PureKAN-ABRBF-linear-AdamW|S6-heuristic-role-block|C0-line-search
PureKAN-ABRBF-linear-AdamW|S6-heuristic-role-block|C1-logit-kl-trust
PureKAN-ABRBF-linear-AdamW|S6-heuristic-role-block|C2-logit-hidden-trust
PureKAN-ABRBF-linear-AdamW|S6-heuristic-role-block|C3-logit-hidden-margin-trust
PureKAN-ABRBF-linear-AdamW|S6-heuristic-role-block|C4-score-accepted
PureKAN-ABRBF-linear-AdamW|S6-heuristic-role-block|C5-score-adaptive-eta
PureKAN-ABRBF-silu-AdamW|S6-heuristic-role-block|C0-line-search
PureKAN-ABRBF-silu-AdamW|S6-heuristic-role-block|C1-logit-kl-trust
PureKAN-ABRBF-silu-AdamW|S6-heuristic-role-block|C2-logit-hidden-trust
PureKAN-ABRBF-silu-AdamW|S6-heuristic-role-block|C3-logit-hidden-margin-trust
PureKAN-ABRBF-silu-AdamW|S6-heuristic-role-block|C4-score-accepted
```

P3 verdict: strong single-step residual smoothing is real. `S0-laplacian`, `S1-sobolev-grad`, and `S6-heuristic-role-block` can produce accepted residual phi reductions under trust controllers across all datasets.

## P4 Accepted Residual Smoothing

| dataset | proposal | controller | steps | acc drop | phiR red | curvR red | holdout Δ | accept | P4 |
|---|---|---|---|---|---|---|---|---|---|
| MNIST | S0-laplacian | C1-logit-kl-trust | 20 | 0.0547 | 0.4719 | 0.9808 | 0.3213 | 1.0000 | no |
| MNIST | S0-laplacian | C2-logit-hidden-trust | 20 | 0.0547 | 0.4719 | 0.9808 | 0.3213 | 1.0000 | no |
| MNIST | S0-laplacian | C3-logit-hidden-margin-trust | 20 | 0.0547 | 0.4719 | 0.9808 | 0.3213 | 1.0000 | no |
| Fashion-MNIST | S0-laplacian | C1-logit-kl-trust | 20 | 0.0586 | 0.4567 | 0.9762 | 0.3988 | 1.0000 | no |
| Fashion-MNIST | S0-laplacian | C2-logit-hidden-trust | 20 | 0.0586 | 0.4567 | 0.9762 | 0.3988 | 1.0000 | no |
| Fashion-MNIST | S0-laplacian | C3-logit-hidden-margin-trust | 20 | 0.0586 | 0.4567 | 0.9762 | 0.3988 | 1.0000 | no |
| KMNIST | S0-laplacian | C1-logit-kl-trust | 20 | -0.0117 | 0.4477 | 0.9746 | 0.2806 | 1.0000 | no |
| KMNIST | S0-laplacian | C2-logit-hidden-trust | 20 | -0.0117 | 0.4477 | 0.9746 | 0.2806 | 1.0000 | no |
| KMNIST | S0-laplacian | C3-logit-hidden-margin-trust | 20 | -0.0117 | 0.4477 | 0.9746 | 0.2806 | 1.0000 | no |
| Fashion-MNIST | S0-laplacian | C1-logit-kl-trust | 5 | 0.0195 | 0.2004 | 0.8661 | 0.1291 | 1.0000 | no |
| Fashion-MNIST | S0-laplacian | C2-logit-hidden-trust | 5 | 0.0195 | 0.2004 | 0.8661 | 0.1291 | 1.0000 | no |
| Fashion-MNIST | S0-laplacian | C3-logit-hidden-margin-trust | 5 | 0.0195 | 0.2004 | 0.8661 | 0.1291 | 1.0000 | no |
| MNIST | S0-laplacian | C1-logit-kl-trust | 5 | 0.0039 | 0.1980 | 0.8792 | 0.0386 | 1.0000 | no |
| MNIST | S0-laplacian | C2-logit-hidden-trust | 5 | 0.0039 | 0.1980 | 0.8792 | 0.0386 | 1.0000 | no |
| MNIST | S0-laplacian | C3-logit-hidden-margin-trust | 5 | 0.0039 | 0.1980 | 0.8792 | 0.0386 | 1.0000 | no |
| KMNIST | S0-laplacian | C1-logit-kl-trust | 5 | -0.0117 | 0.1902 | 0.8604 | 0.0630 | 1.0000 | no |
| KMNIST | S0-laplacian | C2-logit-hidden-trust | 5 | -0.0117 | 0.1902 | 0.8604 | 0.0630 | 1.0000 | no |
| KMNIST | S0-laplacian | C3-logit-hidden-margin-trust | 5 | -0.0117 | 0.1902 | 0.8604 | 0.0630 | 1.0000 | no |

P4 all-dataset survivors:

```text
none
```

P4 verdict: no survivor. Multi-step accepted smoothing reduces residual phi strongly, but it violates the joint task/holdout gate. The bottleneck moved from “no geometry movement” to “geometry movement is too task-expensive when accumulated.”

## P5 / P6 Decision

```text
P5 learn-smooth-refresh: not run
Reason: P4 produced no all-dataset accepted smoothing survivor.

P6 event TAN: not run
Reason: P5 was not reached.
```

## P7 Split Residual Training

| dataset | method | runs | acc | gap vs ABRBF | hold/A | phiR red | base share | P7 |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | ABRBF-baseFCAdam-rbfUFULL | 5 | 0.7824 | -0.0082 |  | 0.1742 | 0.9212 | yes |
| Fashion-MNIST | ABRBF-baseAdam-rbfFCAdam-dataSob | 5 | 0.7754 | -0.0012 |  | 0.1730 | 0.9816 | yes |
| Fashion-MNIST | ABRBF-baseAdam-rbfD6 | 5 | 0.7750 | -0.0008 |  | 0.1731 | 0.9900 | yes |
| Fashion-MNIST | ABRBF-baseAdam-rbfRelaxedNFSRefresh | 5 | 0.7750 | -0.0008 |  | 0.1730 | 0.9898 | yes |
| Fashion-MNIST | ABRBF-baseAdam-rbfUFULL | 5 | 0.7746 | -0.0004 |  | 0.1722 | 0.9726 | yes |
| Fashion-MNIST | ABRBF-AdamW | 5 | 0.7742 | 0.0000 |  | 0.0000 | 0.1967 | yes |
| Fashion-MNIST | ABRBF-allAdamW-edgeOnly | 5 | 0.7742 | 0.0000 |  | 0.0000 | 0.1967 | no |
| Fashion-MNIST | ABRBF-baseOnlyAdam-rbfFrozen | 5 | 0.7742 | 0.0000 |  | 0.1741 | 1.0000 | yes |
| Fashion-MNIST | ABRBF-baseFrozen-rbfUFULL | 5 | 0.6129 | 0.1613 |  | 0.1758 | 0.0000 | no |
| KMNIST | ABRBF-AdamW | 5 | 0.6465 | 0.0000 |  | 0.0000 | 0.2076 | yes |
| KMNIST | ABRBF-allAdamW-edgeOnly | 5 | 0.6465 | 0.0000 |  | 0.0000 | 0.2076 | no |
| KMNIST | ABRBF-baseAdam-rbfUFULL | 5 | 0.6391 | 0.0074 |  | 0.1745 | 0.9690 | yes |
| KMNIST | ABRBF-baseAdam-rbfFCAdam-dataSob | 5 | 0.6363 | 0.0102 |  | 0.1747 | 0.9792 | yes |
| KMNIST | ABRBF-baseOnlyAdam-rbfFrozen | 5 | 0.6344 | 0.0121 |  | 0.1741 | 1.0000 | yes |
| KMNIST | ABRBF-baseAdam-rbfD6 | 5 | 0.6320 | 0.0145 |  | 0.1742 | 0.9886 | yes |
| KMNIST | ABRBF-baseAdam-rbfRelaxedNFSRefresh | 5 | 0.6316 | 0.0148 |  | 0.1743 | 0.9884 | yes |
| KMNIST | ABRBF-baseFCAdam-rbfUFULL | 5 | 0.6258 | 0.0207 |  | 0.1747 | 0.9169 | no |
| KMNIST | ABRBF-baseFrozen-rbfUFULL | 5 | 0.2637 | 0.3828 |  | 0.1787 | 0.0000 | no |
| MNIST | ABRBF-baseFCAdam-rbfUFULL | 5 | 0.8766 | -0.0191 |  | 0.1970 | 0.9011 | yes |
| MNIST | ABRBF-baseAdam-rbfD6 | 5 | 0.8758 | -0.0184 |  | 0.1935 | 0.9855 | yes |
| MNIST | ABRBF-baseAdam-rbfRelaxedNFSRefresh | 5 | 0.8758 | -0.0184 |  | 0.1935 | 0.9853 | yes |
| MNIST | ABRBF-baseOnlyAdam-rbfFrozen | 5 | 0.8754 | -0.0180 |  | 0.1943 | 1.0000 | yes |
| MNIST | ABRBF-baseAdam-rbfUFULL | 5 | 0.8727 | -0.0152 |  | 0.1921 | 0.9619 | yes |
| MNIST | ABRBF-baseAdam-rbfFCAdam-dataSob | 5 | 0.8691 | -0.0117 |  | 0.1936 | 0.9737 | yes |
| MNIST | ABRBF-AdamW | 5 | 0.8574 | 0.0000 |  | 0.0000 | 0.2037 | yes |
| MNIST | ABRBF-allAdamW-edgeOnly | 5 | 0.8574 | 0.0000 |  | 0.0000 | 0.2037 | no |
| MNIST | ABRBF-baseFrozen-rbfUFULL | 5 | 0.4844 | 0.3730 |  | 0.1961 | 0.0000 | no |

P7 verdict: moving the base path is essential. Functional residual variants reduce residual phi, but KMNIST accuracy still lags the ABRBF-AdamW teacher enough to block a clean all-dataset survivor.

## P8 Capacity / Basis Follow-Up

| dataset | edge | hidden | basis | depth | acc | phiR | rank | base/RBF |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | ABRBF-linear | 64 | 16 | 4 | 0.8008 | 0.1114 | 9.1229 | 0.5514 |
| Fashion-MNIST | ABRBF-linear | 96 | 24 | 4 | 0.7930 | 0.1370 | 9.5165 | 0.6043 |
| Fashion-MNIST | ABRBF-linear-learnWidth | 96 | 24 | 4 | 0.7910 | 0.1379 | 9.6156 | 0.6195 |
| Fashion-MNIST | RBFOnly | 64 | 16 | 4 | 0.7891 | 0.1171 | 10.0688 | 0.0000 |
| Fashion-MNIST | ABRBF-linear+silu | 64 | 16 | 4 | 0.7852 | 0.1107 | 9.0518 | 0.6573 |
| Fashion-MNIST | ABRBF-silu | 64 | 16 | 4 | 0.7832 | 0.1129 | 10.8404 | 0.1572 |
| Fashion-MNIST | ABRBF-silu | 96 | 24 | 4 | 0.7793 | 0.1382 | 9.0402 | 0.1274 |
| Fashion-MNIST | ABRBF-linear+silu | 96 | 24 | 4 | 0.7734 | 0.1358 | 9.5579 | 0.6708 |
| Fashion-MNIST | ABRBF-linear-quantileCenters | 96 | 24 | 4 | 0.7656 | 0.1934 | 7.0925 | 0.3132 |
| KMNIST | ABRBF-linear+silu | 64 | 16 | 4 | 0.7168 | 0.1133 | 10.9519 | 0.6333 |
| KMNIST | ABRBF-linear+silu | 96 | 24 | 4 | 0.7051 | 0.1415 | 10.6869 | 0.6009 |
| KMNIST | ABRBF-silu | 64 | 16 | 4 | 0.6934 | 0.1137 | 11.8708 | 0.2022 |
| KMNIST | ABRBF-linear | 64 | 16 | 4 | 0.6855 | 0.1122 | 11.6271 | 0.5465 |
| KMNIST | ABRBF-linear | 96 | 24 | 4 | 0.6836 | 0.1471 | 9.7898 | 0.5215 |
| KMNIST | ABRBF-linear-learnWidth | 96 | 24 | 4 | 0.6836 | 0.1482 | 9.8360 | 0.5409 |
| KMNIST | ABRBF-silu | 96 | 24 | 4 | 0.6797 | 0.1433 | 11.0848 | 0.1064 |
| KMNIST | RBFOnly | 64 | 16 | 4 | 0.6758 | 0.1197 | 11.5154 | 0.0000 |
| KMNIST | ABRBF-linear-quantileCenters | 96 | 24 | 4 | 0.6367 | 0.2437 | 8.4478 | 0.2115 |
| MNIST | ABRBF-linear | 64 | 16 | 4 | 0.8750 | 0.1119 | 10.4300 | 0.5697 |
| MNIST | ABRBF-silu | 64 | 16 | 4 | 0.8750 | 0.1172 | 11.7127 | 0.1817 |
| MNIST | ABRBF-linear+silu | 96 | 24 | 4 | 0.8691 | 0.1465 | 9.9446 | 0.5962 |
| MNIST | ABRBF-silu | 96 | 24 | 4 | 0.8672 | 0.1460 | 9.6207 | 0.1151 |
| MNIST | ABRBF-linear+silu | 64 | 16 | 4 | 0.8633 | 0.1114 | 11.1287 | 0.6716 |
| MNIST | ABRBF-linear | 96 | 24 | 4 | 0.8438 | 0.1425 | 9.7568 | 0.5222 |
| MNIST | ABRBF-linear-learnWidth | 96 | 24 | 4 | 0.8379 | 0.1432 | 9.7848 | 0.5334 |
| MNIST | RBFOnly | 64 | 16 | 4 | 0.8105 | 0.1204 | 10.7913 | 0.0000 |
| MNIST | ABRBF-linear-quantileCenters | 96 | 24 | 4 | 0.8008 | 0.2454 | 5.9702 | 0.1500 |

P8 verdict: compact ABRBF h64/b16 remains a strong default, especially on KMNIST for linear+silu. Scaling h96/b24 does not automatically improve residual geometry, and quantile centers again worsen the tradeoff.

## Failure Diagnosis

```text
F6_accumulated_smoothing_task_cost:
  P3 single-step residual smoothing works, but P4 multi-step smoothing increases holdout/task cost.

F10_split_functional_lags_ABRBF_AdamW:
  Split functional training lowers residual phi but does not match ABRBF-AdamW on the hardest dataset.

Architecture diagnosis:
  AB-RBF is now the right PureKAN default. The remaining bottleneck is the residual smoothing controller, not edge parameterization coverage.
```

## Required Artifacts

Written under `results/v5_1/`:

```text
p0_core_smoke.csv
edge_param_manifest.csv
p1_architecture_confirm5.csv
p1_architecture_gate_summary.csv
p2_split_geometry_calibration.csv
p2_eigenmode_gate_summary.csv
p3_residual_smoothing_proposal_audit.csv
p3_residual_smoothing_gate_summary.csv
p4_accepted_residual_smoothing.csv
p4_accepted_residual_smoothing_trace.csv
p4_accepted_smoothing_gate_summary.csv
p5_learn_smooth_refresh_v3.csv
p6_event_driven_tan_v2.csv
p7_split_residual_training.csv
p7_split_residual_gate_summary.csv
p8_capacity_basis_followup.csv
p8_capacity_gate_summary.csv
failure_table.csv
aggregate_decision.json
figures/
```

## Final Decision

```text
DG-KAN v5.1 status:
  stop_after_p4_no_accepted_smoothing_survivor

What improved:
  AB-RBF is promoted into core dgkan_core.py and remains architecture-positive.
  P3 confirms strong residual smoothing can produce accepted single-step residual phi reductions.
  P7 confirms split base/residual training is meaningful and base dynamics are essential.

What failed / remains open:
  P4 multi-step accepted smoothing fails the joint task/holdout gate.
  P5/P6 expansion is not justified.
  Split functional candidates still do not cleanly beat ABRBF-AdamW on KMNIST.

Conclusion:
  v5.1 solves the edge primitive and single-step residual smoothing direction problem,
  but not the accumulated smoothing controller problem.
  Next work should keep AB-RBF as default and redesign multi-step smoothing with task-aware recovery,
  smaller adaptive eta, or interleaved refresh before spending confirm-seed budget.
```
