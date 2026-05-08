# DG-KAN v4.8 Functional Geometry Feasibility 结果复盘

本轮依据 `docs/DG-KAN_v4.8_FunctionalGeometry_Feasibility_Redesign_实验计划.md`。目标从“继续调 functional optimizer”改为 feasibility-first：先判断当前 PureKAN/RBF 是否存在准确且几何好的可达解，再判断 NFS 是否能在保持 teacher function 的同时降低 geometry。

## Run Inventory

| stage | rows | errors |
|---|---|---|
| P0 FGF/NFS smoke | 21 | 0 |
| P1 feasibility frontier | 27 | 0 |
| P2 NFS AdamW projection | 1440 | 0 |
| P3 NFS functional teachers | 108 | 0 |
| P4 TAN one-cycle | 120 | 0 |
| P5 TAN alternating cycles | 18 | 0 |
| P6 capacity/basis expansion | 1 | 0 |

## Code / Config Changes

```text
experiments/run_gafu_v48.py
  Added FGF AdamW geometry regularization frontier.
  Added offline NFS projection variants with teacher snapshot, backtracking,
  logit/hidden/margin constraints, and role-specific smoothing proposals.
  Added functional-teacher NFS, TAN one-cycle, and gated TAN multi-cycle probes.
  Added compact P6 capacity/basis feasibility expansion.

experiments/analyze_gafu_v48.py
  Generates frontier/NFS/capacity gate summaries, failure taxonomy,
  SVG diagnostics, aggregate_decision.json, and this replay.
```

## P0 Implementation Smoke

| rows | errors | max nonKAN | min cov | max rollback | max CG residual | pass |
|---|---|---|---|---|---|---|
| 21 | 0 | 0.0000 | 1.0000 | 0.0000 | 0.0533 | true |

P0 verdict: pass. NFS temporary apply / rollback / teacher snapshot / finite projection diagnostics passed for the smoke grid.

## P1 Accuracy-Geometry Feasibility Frontier

| dataset | method | acc | gap | phi red | J red | ECE | rank | P1 |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | A0-PureKAN-AdamW | 0.7754 | 0.0000 | 0.0000 | 0.0000 | 0.0655 | 16.8570 | yes |
| Fashion-MNIST | A1-AdamW-Sobolev1e-6 | 0.7754 | 0.0000 | -0.0000 | -0.0000 | 0.0655 | 16.8570 | no |
| Fashion-MNIST | A2-AdamW-Sobolev3e-6 | 0.7754 | 0.0000 | -0.0000 | 0.0000 | 0.0655 | 16.8571 | no |
| Fashion-MNIST | A3-AdamW-Sobolev1e-5 | 0.7754 | 0.0000 | -0.0000 | 0.0000 | 0.0655 | 16.8570 | no |
| Fashion-MNIST | A4-AdamW-PhiProxy1e-5 | 0.7754 | 0.0000 | -0.0000 | -0.0002 | 0.0655 | 16.8570 | no |
| Fashion-MNIST | A5-AdamW-JacProxy1e-5 | 0.7754 | 0.0000 | -0.0000 | -0.0002 | 0.0655 | 16.8570 | no |
| Fashion-MNIST | A6-AdamW-MixedGeom | 0.7754 | 0.0000 | -0.0000 | -0.0005 | 0.0655 | 16.8570 | no |
| Fashion-MNIST | A7-AdamW-LateGeom | 0.7754 | 0.0000 | 0.0000 | -0.0000 | 0.0655 | 16.8570 | no |
| Fashion-MNIST | A8-AdamW-PosthocGeom | 0.5664 | 0.2090 | -0.0052 | 0.0731 | 0.3956 | 11.9923 | no |
| KMNIST | A0-PureKAN-AdamW | 0.6797 | 0.0000 | 0.0000 | 0.0000 | 0.1014 | 19.8530 | yes |
| KMNIST | A1-AdamW-Sobolev1e-6 | 0.6797 | 0.0000 | 0.0000 | -0.0002 | 0.1014 | 19.8530 | no |
| KMNIST | A2-AdamW-Sobolev3e-6 | 0.6797 | 0.0000 | 0.0000 | -0.0006 | 0.1014 | 19.8530 | no |
| KMNIST | A3-AdamW-Sobolev1e-5 | 0.6797 | 0.0000 | 0.0000 | -0.0024 | 0.1014 | 19.8531 | no |
| KMNIST | A4-AdamW-PhiProxy1e-5 | 0.6797 | 0.0000 | -0.0000 | 0.0013 | 0.1014 | 19.8529 | no |
| KMNIST | A5-AdamW-JacProxy1e-5 | 0.6797 | 0.0000 | -0.0000 | 0.0011 | 0.1014 | 19.8530 | no |
| KMNIST | A6-AdamW-MixedGeom | 0.6797 | 0.0000 | -0.0000 | 0.0031 | 0.1014 | 19.8529 | no |
| KMNIST | A7-AdamW-LateGeom | 0.6797 | 0.0000 | 0.0000 | 0.0000 | 0.1014 | 19.8530 | no |
| KMNIST | A8-AdamW-PosthocGeom | 0.1973 | 0.4824 | 0.0280 | -2.7726 | 0.0751 | 15.8675 | no |
| MNIST | A0-PureKAN-AdamW | 0.8711 | 0.0000 | 0.0000 | 0.0000 | 0.0427 | 16.2443 | yes |
| MNIST | A1-AdamW-Sobolev1e-6 | 0.8711 | 0.0000 | 0.0000 | 0.0040 | 0.0427 | 16.2443 | no |
| MNIST | A2-AdamW-Sobolev3e-6 | 0.8711 | 0.0000 | -0.0000 | 0.0107 | 0.0427 | 16.2444 | no |
| MNIST | A3-AdamW-Sobolev1e-5 | 0.8711 | 0.0000 | -0.0000 | 0.0358 | 0.0427 | 16.2443 | no |
| MNIST | A4-AdamW-PhiProxy1e-5 | 0.8711 | 0.0000 | -0.0000 | 0.0084 | 0.0427 | 16.2443 | no |
| MNIST | A5-AdamW-JacProxy1e-5 | 0.8711 | 0.0000 | -0.0000 | 0.0116 | 0.0427 | 16.2442 | no |
| MNIST | A6-AdamW-MixedGeom | 0.8711 | 0.0000 | -0.0000 | 0.0324 | 0.0427 | 16.2442 | no |
| MNIST | A7-AdamW-LateGeom | 0.8711 | 0.0000 | 0.0000 | 0.0004 | 0.0427 | 16.2443 | no |
| MNIST | A8-AdamW-PosthocGeom | 0.2148 | 0.6562 | 0.0435 | 0.8932 | 0.0521 | 11.6523 | no |

P1 all-dataset survivors:

```text
none
```

## P2 Offline NFS Projection From AdamW Teachers

Top diagnostic rows are sorted by dataset and smoothability/geometry signal.

| dataset | teacher | variant | proj | eta | acc drop | KL | logit | phi red | J red | smooth | P2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | TeacherA-AdamW20 | NFS-role-block | diag | 0.05 | 0.0039 | 0.0001 | 0.0147 | 0.1482 | 0.4284 | 14.6723 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-role-block | cg5 | 0.05 | 0.0020 | 0.0001 | 0.0126 | 0.1289 | 0.3912 | 12.7956 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-role-block | diag | 0.05 | 0.0000 | 0.0007 | 0.0180 | 0.1349 | 0.4547 | 12.6464 | yes |
| Fashion-MNIST | TeacherD-bestFGF | NFS-role-block | diag | 0.05 | -0.0039 | 0.0017 | 0.0217 | 0.1325 | 0.8597 | 11.3154 | yes |
| Fashion-MNIST | TeacherC-AdamW-final | NFS-role-block | diag | 0.05 | -0.0039 | 0.0017 | 0.0217 | 0.1325 | 0.8597 | 11.3153 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-role-block | cg5 | 0.05 | 0.0020 | 0.0005 | 0.0154 | 0.1172 | 0.4227 | 11.1732 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-role-block | diag | 0.035 | 0.0020 | 0.0001 | 0.0104 | 0.1086 | 0.3466 | 10.8035 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-role-block | cg10 | 0.05 | 0.0020 | 0.0001 | 0.0104 | 0.1086 | 0.3466 | 10.8035 | yes |
| Fashion-MNIST | TeacherD-bestFGF | NFS-role-block | cg5 | 0.05 | -0.0059 | 0.0013 | 0.0186 | 0.1150 | 0.8247 | 10.2175 | yes |
| Fashion-MNIST | TeacherC-AdamW-final | NFS-role-block | cg5 | 0.05 | -0.0059 | 0.0013 | 0.0186 | 0.1150 | 0.8246 | 10.2172 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-role-block | diag | 0.035 | 0.0020 | 0.0003 | 0.0128 | 0.0987 | 0.3865 | 9.5517 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-role-block | cg10 | 0.05 | 0.0020 | 0.0003 | 0.0128 | 0.0987 | 0.3865 | 9.5517 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-role-block | cg5 | 0.035 | 0.0000 | 0.0000 | 0.0089 | 0.0937 | 0.3111 | 9.3364 | yes |
| Fashion-MNIST | TeacherD-bestFGF | NFS-role-block | diag | 0.035 | -0.0059 | 0.0009 | 0.0154 | 0.0968 | 0.7698 | 8.9087 | yes |
| Fashion-MNIST | TeacherD-bestFGF | NFS-role-block | cg10 | 0.05 | -0.0059 | 0.0009 | 0.0154 | 0.0968 | 0.7698 | 8.9087 | yes |
| Fashion-MNIST | TeacherC-AdamW-final | NFS-role-block | cg10 | 0.05 | -0.0059 | 0.0009 | 0.0154 | 0.0968 | 0.7696 | 8.9081 | yes |
| Fashion-MNIST | TeacherC-AdamW-final | NFS-role-block | diag | 0.035 | -0.0059 | 0.0009 | 0.0154 | 0.0968 | 0.7696 | 8.9081 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-role-block | lowrank32 | 0.05 | 0.0000 | 0.0000 | 0.0082 | 0.0873 | 0.2946 | 8.6984 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-H | diag | 0.05 | 0.0020 | 0.0073 | 0.1117 | 0.1482 | 0.4224 | 8.5895 | no |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-H | cg5 | 0.05 | 0.0020 | 0.0053 | 0.0958 | 0.1288 | 0.3849 | 8.3970 | no |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-role-block | cg5 | 0.035 | 0.0020 | 0.0002 | 0.0109 | 0.0853 | 0.3580 | 8.3272 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-H | diag | 0.035 | 0.0000 | 0.0037 | 0.0796 | 0.1086 | 0.3401 | 7.9314 | no |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-H | cg10 | 0.05 | 0.0000 | 0.0037 | 0.0796 | 0.1086 | 0.3401 | 7.9314 | no |
| Fashion-MNIST | TeacherD-bestFGF | NFS-role-block | cg5 | 0.035 | -0.0059 | 0.0006 | 0.0132 | 0.0835 | 0.7086 | 7.8586 | yes |
| Fashion-MNIST | TeacherC-AdamW-final | NFS-role-block | cg5 | 0.035 | -0.0059 | 0.0006 | 0.0132 | 0.0835 | 0.7084 | 7.8578 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-role-block | cg10 | 0.035 | 0.0000 | 0.0000 | 0.0074 | 0.0783 | 0.2713 | 7.8128 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-role-block | lowrank32 | 0.05 | 0.0020 | 0.0002 | 0.0101 | 0.0793 | 0.3454 | 7.7677 | yes |
| Fashion-MNIST | TeacherD-bestFGF | NFS-role-block | lowrank32 | 0.05 | -0.0039 | 0.0005 | 0.0122 | 0.0778 | 0.6722 | 7.3849 | yes |
| Fashion-MNIST | TeacherC-AdamW-final | NFS-role-block | lowrank32 | 0.05 | -0.0039 | 0.0005 | 0.0122 | 0.0778 | 0.6720 | 7.3839 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-H | cg5 | 0.035 | -0.0020 | 0.0027 | 0.0681 | 0.0938 | 0.3045 | 7.3826 | no |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-role-block | lowrank64 | 0.05 | 0.0000 | 0.0000 | 0.0068 | 0.0725 | 0.2548 | 7.2312 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-H | lowrank32 | 0.05 | -0.0039 | 0.0023 | 0.0631 | 0.0873 | 0.2880 | 7.0816 | no |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-role-block | cg10 | 0.035 | 0.0000 | 0.0002 | 0.0090 | 0.0713 | 0.3263 | 7.0142 | yes |
| Fashion-MNIST | TeacherC-AdamW-final | NFS-role-block | cg10 | 0.035 | -0.0039 | 0.0004 | 0.0109 | 0.0698 | 0.6082 | 6.6922 | yes |
| Fashion-MNIST | TeacherD-bestFGF | NFS-role-block | cg10 | 0.035 | -0.0039 | 0.0004 | 0.0109 | 0.0698 | 0.6085 | 6.6920 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-H | cg10 | 0.035 | -0.0039 | 0.0019 | 0.0564 | 0.0786 | 0.2648 | 6.6262 | no |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-role-block | diag | 0.02 | 0.0000 | 0.0000 | 0.0060 | 0.0651 | 0.2331 | 6.4966 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-role-block | lowrank64 | 0.05 | 0.0000 | 0.0001 | 0.0083 | 0.0659 | 0.3108 | 6.4958 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-role-cycle | diag | 0.05 | 0.0020 | 0.0004 | 0.0270 | 0.0660 | 0.2300 | 6.3358 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-role-block | lowrank32 | 0.035 | 0.0000 | 0.0000 | 0.0058 | 0.0629 | 0.2264 | 6.2785 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-H | lowrank64 | 0.05 | -0.0020 | 0.0016 | 0.0520 | 0.0725 | 0.2485 | 6.2634 | no |
| Fashion-MNIST | TeacherD-bestFGF | NFS-role-block | lowrank64 | 0.05 | -0.0039 | 0.0004 | 0.0101 | 0.0645 | 0.5516 | 6.2185 | yes |
| Fashion-MNIST | TeacherC-AdamW-final | NFS-role-block | lowrank64 | 0.05 | -0.0039 | 0.0004 | 0.0101 | 0.0644 | 0.5513 | 6.2171 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-role-block | diag | 0.02 | 0.0000 | 0.0001 | 0.0074 | 0.0589 | 0.2906 | 5.8272 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-H | diag | 0.02 | -0.0020 | 0.0013 | 0.0463 | 0.0651 | 0.2270 | 5.7857 | no |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-H | cg10 | 0.05 | -0.0020 | 0.0073 | 0.0416 | 0.0985 | 0.3731 | 5.7098 | no |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-H | diag | 0.035 | -0.0020 | 0.0073 | 0.0416 | 0.0985 | 0.3731 | 5.7098 | no |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-H | cg5 | 0.05 | -0.0020 | 0.0106 | 0.0502 | 0.1174 | 0.4077 | 5.6847 | no |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-H | lowrank32 | 0.035 | 0.0000 | 0.0012 | 0.0446 | 0.0629 | 0.2203 | 5.6317 | no |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-role-block | lowrank32 | 0.035 | 0.0000 | 0.0001 | 0.0071 | 0.0568 | 0.2844 | 5.6238 | yes |
| Fashion-MNIST | TeacherD-bestFGF | NFS-role-block | diag | 0.02 | 0.0000 | 0.0003 | 0.0090 | 0.0576 | 0.4544 | 5.5986 | yes |
| Fashion-MNIST | TeacherC-AdamW-final | NFS-role-block | diag | 0.02 | 0.0000 | 0.0003 | 0.0090 | 0.0576 | 0.4540 | 5.5981 | yes |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-role-block | cg5 | 0.02 | 0.0000 | 0.0000 | 0.0051 | 0.0560 | 0.2054 | 5.5891 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-H | cg5 | 0.035 | -0.0020 | 0.0053 | 0.0355 | 0.0850 | 0.3472 | 5.5682 | no |
| Fashion-MNIST | TeacherA-AdamW20 | NFS-role-cycle | cg5 | 0.05 | 0.0000 | 0.0003 | 0.0229 | 0.0567 | 0.2019 | 5.4978 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-H | diag | 0.05 | 0.0000 | 0.0147 | 0.0588 | 0.1348 | 0.4397 | 5.4677 | no |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-H | lowrank32 | 0.05 | -0.0020 | 0.0045 | 0.0329 | 0.0790 | 0.3340 | 5.4485 | no |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-role-cycle | diag | 0.05 | -0.0039 | 0.0009 | 0.0156 | 0.0592 | 0.2836 | 5.4346 | yes |
| Fashion-MNIST | TeacherD-bestFGF | NFS-role-block | lowrank32 | 0.035 | 0.0000 | 0.0003 | 0.0086 | 0.0556 | 0.4171 | 5.4100 | yes |
| Fashion-MNIST | TeacherC-AdamW-final | NFS-role-block | lowrank32 | 0.035 | 0.0000 | 0.0003 | 0.0086 | 0.0556 | 0.4166 | 5.4094 | yes |
| Fashion-MNIST | TeacherD-bestFGF | NFS-H | cg5 | 0.035 | 0.0098 | 0.0054 | 0.0336 | 0.0834 | 0.6860 | 5.4055 | no |
| Fashion-MNIST | TeacherC-AdamW-final | NFS-H | cg5 | 0.035 | 0.0098 | 0.0054 | 0.0336 | 0.0834 | 0.6858 | 5.4049 | no |
| Fashion-MNIST | TeacherD-bestFGF | NFS-H | lowrank32 | 0.05 | 0.0098 | 0.0046 | 0.0311 | 0.0775 | 0.6451 | 5.2940 | no |
| Fashion-MNIST | TeacherC-AdamW-final | NFS-H | lowrank32 | 0.05 | 0.0098 | 0.0046 | 0.0311 | 0.0775 | 0.6448 | 5.2934 | no |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-Z | cg10 | 0.05 | -0.0020 | 0.0037 | 0.0299 | 0.0723 | 0.3170 | 5.2709 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-ZH | cg10 | 0.05 | -0.0020 | 0.0037 | 0.0299 | 0.0723 | 0.3170 | 5.2709 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-ZH-margin | cg10 | 0.05 | -0.0020 | 0.0037 | 0.0299 | 0.0723 | 0.3170 | 5.2709 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-Z | diag | 0.035 | -0.0020 | 0.0037 | 0.0299 | 0.0723 | 0.3170 | 5.2709 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-ZH | diag | 0.035 | -0.0020 | 0.0037 | 0.0299 | 0.0723 | 0.3170 | 5.2709 | yes |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-ZH-margin | diag | 0.035 | -0.0020 | 0.0037 | 0.0299 | 0.0723 | 0.3170 | 5.2709 | yes |
| Fashion-MNIST | TeacherD-bestFGF | NFS-H | diag | 0.05 | 0.0098 | 0.0151 | 0.0557 | 0.1321 | 0.8525 | 5.2623 | no |
| Fashion-MNIST | TeacherB-AdamW100 | NFS-Z | lowrank32 | 0.05 | -0.0020 | 0.0037 | 0.0296 | 0.0716 | 0.3156 | 5.2458 | yes |

Best diagnostic points:

| dataset | best by | teacher | variant | smooth | acc drop | KL | phi red |
|---|---|---|---|---|---|---|---|
| MNIST | smoothability | TeacherA-AdamW20 | NFS-role-block | 9.8283 | -0.0078 | 0.0000 | 0.0984 |
| MNIST | phi | TeacherA-AdamW20 | NFS-H | 0.0000 | 0.0938 | 0.0011 | 0.1507 |
| Fashion-MNIST | smoothability | TeacherA-AdamW20 | NFS-role-block | 14.6723 | 0.0039 | 0.0001 | 0.1482 |
| Fashion-MNIST | phi | TeacherA-AdamW20 | NFS-role-block | 14.6723 | 0.0039 | 0.0001 | 0.1482 |
| KMNIST | smoothability | TeacherA-AdamW20 | NFS-role-block | 14.8527 | -0.0039 | 0.0001 | 0.1494 |
| KMNIST | phi | TeacherA-AdamW20 | NFS-role-block | 14.8527 | -0.0039 | 0.0001 | 0.1494 |

P2 all-dataset survivors:

```text
TeacherA-AdamW20|NFS-role-block|diag|0.035, TeacherA-AdamW20|NFS-role-block|diag|0.05, TeacherA-AdamW20|NFS-role-block|cg5|0.035, TeacherA-AdamW20|NFS-role-block|cg5|0.05, TeacherA-AdamW20|NFS-role-block|cg10|0.05, TeacherA-AdamW20|NFS-role-block|lowrank32|0.05, TeacherB-AdamW100|NFS-Z|diag|0.02, TeacherB-AdamW100|NFS-Z|diag|0.035, TeacherB-AdamW100|NFS-Z|diag|0.05, TeacherB-AdamW100|NFS-Z|cg5|0.02, TeacherB-AdamW100|NFS-Z|cg5|0.035, TeacherB-AdamW100|NFS-Z|cg5|0.05, TeacherB-AdamW100|NFS-Z|cg10|0.02, TeacherB-AdamW100|NFS-Z|cg10|0.035, TeacherB-AdamW100|NFS-Z|cg10|0.05, TeacherB-AdamW100|NFS-Z|lowrank32|0.02, TeacherB-AdamW100|NFS-Z|lowrank32|0.035, TeacherB-AdamW100|NFS-Z|lowrank32|0.05, TeacherB-AdamW100|NFS-Z|lowrank64|0.035, TeacherB-AdamW100|NFS-Z|lowrank64|0.05, TeacherB-AdamW100|NFS-H|diag|0.02, TeacherB-AdamW100|NFS-H|cg5|0.02, TeacherB-AdamW100|NFS-H|cg10|0.02, TeacherB-AdamW100|NFS-H|lowrank32|0.02, TeacherB-AdamW100|NFS-H|lowrank32|0.035, TeacherB-AdamW100|NFS-H|lowrank64|0.035, TeacherB-AdamW100|NFS-H|lowrank64|0.05, TeacherB-AdamW100|NFS-ZH|diag|0.02, TeacherB-AdamW100|NFS-ZH|diag|0.035, TeacherB-AdamW100|NFS-ZH|diag|0.05, TeacherB-AdamW100|NFS-ZH|cg5|0.02, TeacherB-AdamW100|NFS-ZH|cg5|0.035, TeacherB-AdamW100|NFS-ZH|cg5|0.05, TeacherB-AdamW100|NFS-ZH|cg10|0.02, TeacherB-AdamW100|NFS-ZH|cg10|0.035, TeacherB-AdamW100|NFS-ZH|cg10|0.05, TeacherB-AdamW100|NFS-ZH|lowrank32|0.02, TeacherB-AdamW100|NFS-ZH|lowrank32|0.035, TeacherB-AdamW100|NFS-ZH|lowrank32|0.05, TeacherB-AdamW100|NFS-ZH|lowrank64|0.035, TeacherB-AdamW100|NFS-ZH|lowrank64|0.05, TeacherB-AdamW100|NFS-ZH-margin|diag|0.02, TeacherB-AdamW100|NFS-ZH-margin|diag|0.035, TeacherB-AdamW100|NFS-ZH-margin|diag|0.05, TeacherB-AdamW100|NFS-ZH-margin|cg5|0.02, TeacherB-AdamW100|NFS-ZH-margin|cg5|0.035, TeacherB-AdamW100|NFS-ZH-margin|cg5|0.05, TeacherB-AdamW100|NFS-ZH-margin|cg10|0.02, TeacherB-AdamW100|NFS-ZH-margin|cg10|0.035, TeacherB-AdamW100|NFS-ZH-margin|cg10|0.05, TeacherB-AdamW100|NFS-ZH-margin|lowrank32|0.02, TeacherB-AdamW100|NFS-ZH-margin|lowrank32|0.035, TeacherB-AdamW100|NFS-ZH-margin|lowrank32|0.05, TeacherB-AdamW100|NFS-ZH-margin|lowrank64|0.035, TeacherB-AdamW100|NFS-ZH-margin|lowrank64|0.05, TeacherB-AdamW100|NFS-role-block|cg10|0.02, TeacherB-AdamW100|NFS-role-block|lowrank32|0.02, TeacherB-AdamW100|NFS-role-block|lowrank64|0.035, TeacherB-AdamW100|NFS-role-cycle|diag|0.035, TeacherB-AdamW100|NFS-role-cycle|diag|0.05, TeacherB-AdamW100|NFS-role-cycle|cg5|0.035, TeacherB-AdamW100|NFS-role-cycle|cg5|0.05, TeacherB-AdamW100|NFS-role-cycle|cg10|0.05, TeacherB-AdamW100|NFS-role-cycle|lowrank32|0.05, TeacherC-AdamW-final|NFS-role-block|diag|0.035, TeacherC-AdamW-final|NFS-role-block|diag|0.05, TeacherC-AdamW-final|NFS-role-block|cg5|0.02, TeacherC-AdamW-final|NFS-role-block|cg5|0.035, TeacherC-AdamW-final|NFS-role-block|cg5|0.05, TeacherC-AdamW-final|NFS-role-block|cg10|0.05, TeacherC-AdamW-final|NFS-role-block|lowrank32|0.05, TeacherD-bestFGF|NFS-role-block|diag|0.035, TeacherD-bestFGF|NFS-role-block|diag|0.05, TeacherD-bestFGF|NFS-role-block|cg5|0.02, TeacherD-bestFGF|NFS-role-block|cg5|0.035, TeacherD-bestFGF|NFS-role-block|cg5|0.05, TeacherD-bestFGF|NFS-role-block|cg10|0.05, TeacherD-bestFGF|NFS-role-block|lowrank32|0.05
```

## P3 Offline NFS Projection From Functional Teachers

| dataset | teacher | method | variant | proj | acc drop | KL | phi red | smooth | P3 |
|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | TeacherE-FCAdam20 | FCAdam-dataSob | NFS-role-block | diag | 0.0000 | 0.0011 | 0.1356 | 12.2511 | yes |
| Fashion-MNIST | TeacherG-FLD-Adan20 | FLD-AdanLite | NFS-role-block | diag | 0.0078 | 0.0010 | 0.1258 | 11.3989 | no |
| Fashion-MNIST | TeacherI-D6-100 | D6-allTaskAware | NFS-role-block | diag | 0.0020 | 0.0004 | 0.1120 | 10.7596 | yes |
| Fashion-MNIST | TeacherJ-FNG-100 | F4-FNG-leftFullRight | NFS-role-block | diag | 0.0000 | 0.0021 | 0.1021 | 8.4408 | yes |
| Fashion-MNIST | TeacherE-FCAdam20 | FCAdam-dataSob | NFS-role-cycle | diag | 0.0000 | 0.0012 | 0.0731 | 6.5380 | yes |
| Fashion-MNIST | TeacherG-FLD-Adan20 | FLD-AdanLite | NFS-role-cycle | diag | 0.0078 | 0.0012 | 0.0729 | 6.4812 | no |
| Fashion-MNIST | TeacherH-FLD-Adan100 | FLD-AdanLite | NFS-role-block | diag | -0.0078 | 0.0026 | 0.0797 | 6.3248 | yes |
| Fashion-MNIST | TeacherF-FCAdam100 | FCAdam-dataSob | NFS-role-block | diag | -0.0059 | 0.0034 | 0.0788 | 5.8921 | yes |
| Fashion-MNIST | TeacherI-D6-100 | D6-allTaskAware | NFS-Z | cg5 | 0.0000 | 0.0017 | 0.0561 | 4.7881 | no |
| Fashion-MNIST | TeacherI-D6-100 | D6-allTaskAware | NFS-Z | diag | 0.0000 | 0.0017 | 0.0558 | 4.7725 | no |
| Fashion-MNIST | TeacherI-D6-100 | D6-allTaskAware | NFS-ZH | diag | 0.0000 | 0.0017 | 0.0558 | 4.7725 | no |
| Fashion-MNIST | TeacherI-D6-100 | D6-allTaskAware | NFS-role-cycle | diag | -0.0020 | 0.0005 | 0.0481 | 4.6050 | no |
| Fashion-MNIST | TeacherE-FCAdam20 | FCAdam-dataSob | NFS-role-cycle | cg5 | 0.0020 | 0.0013 | 0.0480 | 4.2320 | no |
| Fashion-MNIST | TeacherJ-FNG-100 | F4-FNG-leftFullRight | NFS-role-cycle | diag | 0.0000 | 0.0017 | 0.0481 | 4.1158 | no |
| Fashion-MNIST | TeacherI-D6-100 | D6-allTaskAware | NFS-role-cycle | cg5 | -0.0020 | 0.0003 | 0.0412 | 3.9924 | no |
| Fashion-MNIST | TeacherG-FLD-Adan20 | FLD-AdanLite | NFS-role-cycle | cg5 | 0.0059 | 0.0014 | 0.0400 | 3.5108 | no |
| Fashion-MNIST | TeacherJ-FNG-100 | F4-FNG-leftFullRight | NFS-role-cycle | cg5 | 0.0000 | 0.0019 | 0.0415 | 3.4928 | no |
| Fashion-MNIST | TeacherJ-FNG-100 | F4-FNG-leftFullRight | NFS-Z | cg5 | 0.0000 | 0.0029 | 0.0413 | 3.1931 | no |
| Fashion-MNIST | TeacherJ-FNG-100 | F4-FNG-leftFullRight | NFS-Z | diag | 0.0000 | 0.0029 | 0.0411 | 3.1827 | no |
| Fashion-MNIST | TeacherJ-FNG-100 | F4-FNG-leftFullRight | NFS-ZH | diag | 0.0000 | 0.0029 | 0.0411 | 3.1827 | no |
| Fashion-MNIST | TeacherE-FCAdam20 | FCAdam-dataSob | NFS-Z | diag | 0.0000 | 0.0013 | 0.0340 | 3.0039 | no |
| Fashion-MNIST | TeacherE-FCAdam20 | FCAdam-dataSob | NFS-ZH | diag | 0.0000 | 0.0013 | 0.0340 | 3.0039 | no |
| Fashion-MNIST | TeacherE-FCAdam20 | FCAdam-dataSob | NFS-Z | cg5 | 0.0000 | 0.0013 | 0.0339 | 2.9976 | no |
| Fashion-MNIST | TeacherH-FLD-Adan100 | FLD-AdanLite | NFS-role-cycle | cg5 | -0.0078 | 0.0050 | 0.0443 | 2.9457 | yes |
| Fashion-MNIST | TeacherG-FLD-Adan20 | FLD-AdanLite | NFS-Z | diag | 0.0059 | 0.0014 | 0.0334 | 2.9314 | no |
| Fashion-MNIST | TeacherG-FLD-Adan20 | FLD-AdanLite | NFS-ZH | diag | 0.0059 | 0.0014 | 0.0334 | 2.9314 | no |
| Fashion-MNIST | TeacherG-FLD-Adan20 | FLD-AdanLite | NFS-Z | cg5 | 0.0059 | 0.0014 | 0.0333 | 2.9260 | no |
| Fashion-MNIST | TeacherF-FCAdam100 | FCAdam-dataSob | NFS-role-cycle | cg5 | -0.0078 | 0.0050 | 0.0433 | 2.8938 | yes |
| Fashion-MNIST | TeacherF-FCAdam100 | FCAdam-dataSob | NFS-role-cycle | diag | -0.0059 | 0.0048 | 0.0313 | 2.1147 | yes |
| Fashion-MNIST | TeacherH-FLD-Adan100 | FLD-AdanLite | NFS-role-cycle | diag | -0.0098 | 0.0051 | 0.0285 | 1.8831 | yes |
| Fashion-MNIST | TeacherH-FLD-Adan100 | FLD-AdanLite | NFS-Z | cg5 | -0.0098 | 0.0051 | 0.0246 | 1.6327 | yes |
| Fashion-MNIST | TeacherH-FLD-Adan100 | FLD-AdanLite | NFS-Z | diag | -0.0098 | 0.0048 | 0.0239 | 1.6166 | yes |
| Fashion-MNIST | TeacherH-FLD-Adan100 | FLD-AdanLite | NFS-ZH | diag | -0.0098 | 0.0048 | 0.0239 | 1.6166 | yes |
| Fashion-MNIST | TeacherF-FCAdam100 | FCAdam-dataSob | NFS-Z | diag | -0.0059 | 0.0047 | 0.0235 | 1.5991 | no |
| Fashion-MNIST | TeacherF-FCAdam100 | FCAdam-dataSob | NFS-ZH | diag | -0.0059 | 0.0047 | 0.0235 | 1.5991 | no |
| Fashion-MNIST | TeacherF-FCAdam100 | FCAdam-dataSob | NFS-Z | cg5 | -0.0059 | 0.0047 | 0.0234 | 1.5971 | no |
| KMNIST | TeacherI-D6-100 | D6-allTaskAware | NFS-role-block | diag | 0.0020 | 0.0004 | 0.1114 | 10.7330 | yes |
| KMNIST | TeacherJ-FNG-100 | F4-FNG-leftFullRight | NFS-role-block | diag | -0.0039 | 0.0015 | 0.0786 | 6.8109 | yes |
| KMNIST | TeacherE-FCAdam20 | FCAdam-dataSob | NFS-role-block | diag | 0.0039 | 0.0006 | 0.0690 | 6.4991 | yes |
| KMNIST | TeacherG-FLD-Adan20 | FLD-AdanLite | NFS-role-block | diag | 0.0020 | 0.0006 | 0.0652 | 6.1242 | yes |
| KMNIST | TeacherI-D6-100 | D6-allTaskAware | NFS-Z | diag | 0.0098 | 0.0007 | 0.0515 | 4.8160 | no |
| KMNIST | TeacherI-D6-100 | D6-allTaskAware | NFS-ZH | diag | 0.0098 | 0.0007 | 0.0515 | 4.8160 | no |
| KMNIST | TeacherI-D6-100 | D6-allTaskAware | NFS-Z | cg5 | 0.0098 | 0.0007 | 0.0515 | 4.8132 | no |
| KMNIST | TeacherE-FCAdam20 | FCAdam-dataSob | NFS-role-cycle | cg5 | 0.0020 | 0.0007 | 0.0515 | 4.8057 | yes |
| KMNIST | TeacherG-FLD-Adan20 | FLD-AdanLite | NFS-role-cycle | cg5 | 0.0020 | 0.0008 | 0.0510 | 4.7363 | yes |
| KMNIST | TeacherI-D6-100 | D6-allTaskAware | NFS-role-cycle | diag | 0.0098 | 0.0002 | 0.0480 | 4.6887 | no |
| KMNIST | TeacherG-FLD-Adan20 | FLD-AdanLite | NFS-role-cycle | diag | 0.0000 | 0.0008 | 0.0502 | 4.6649 | yes |
| KMNIST | TeacherF-FCAdam100 | FCAdam-dataSob | NFS-role-block | diag | 0.0000 | 0.0025 | 0.0569 | 4.5520 | no |
| KMNIST | TeacherH-FLD-Adan100 | FLD-AdanLite | NFS-role-block | diag | 0.0020 | 0.0023 | 0.0530 | 4.3066 | no |
| KMNIST | TeacherJ-FNG-100 | F4-FNG-leftFullRight | NFS-role-cycle | diag | -0.0137 | 0.0014 | 0.0487 | 4.2567 | no |
| KMNIST | TeacherF-FCAdam100 | FCAdam-dataSob | NFS-role-cycle | cg5 | -0.0039 | 0.0033 | 0.0554 | 4.1739 | yes |
| KMNIST | TeacherE-FCAdam20 | FCAdam-dataSob | NFS-role-cycle | diag | 0.0000 | 0.0008 | 0.0436 | 4.0503 | yes |
| KMNIST | TeacherI-D6-100 | D6-allTaskAware | NFS-role-cycle | cg5 | 0.0059 | 0.0002 | 0.0409 | 4.0216 | no |
| KMNIST | TeacherF-FCAdam100 | FCAdam-dataSob | NFS-role-cycle | diag | -0.0039 | 0.0034 | 0.0527 | 3.9467 | no |

P3 all-dataset survivors:

```text
TeacherE-FCAdam20|NFS-role-block|diag|0.035, TeacherI-D6-100|NFS-role-block|diag|0.035, TeacherJ-FNG-100|NFS-role-block|diag|0.035
```

## P4 TAN One-Cycle Micro-Run

| dataset | task | NFS | refresh | task acc | final acc | acc drop | phi red | recovery | P4 |
|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | D6-allTaskAware | P2best-NFS-role-block-diag-eta0.05 | D6-5 | 0.7383 | 0.7598 | -0.0215 | 0.1526 | 37.9383 | yes |
| Fashion-MNIST | D6-allTaskAware | P2best-NFS-role-block-diag-eta0.05 | FCAdam5 | 0.7383 | 0.7598 | -0.0215 | 0.1425 | 25.4144 | yes |
| Fashion-MNIST | PureKAN-AdamW | P2best-NFS-role-block-diag-eta0.05 | D6-5 | 0.7734 | 0.7852 | -0.0117 | 0.1370 | 32.3783 | yes |
| Fashion-MNIST | D6-allTaskAware | P2best-NFS-role-block-diag-eta0.05 | FCAdam10 | 0.7383 | 0.7500 | -0.0117 | 0.1330 | 45.9922 | yes |
| Fashion-MNIST | D6-allTaskAware | P2best-NFS-role-block-cg5-eta0.05 | D6-5 | 0.7383 | 0.7617 | -0.0234 | 0.1326 | 49.5606 | yes |
| Fashion-MNIST | D6-allTaskAware | P2best-NFS-role-block-cg5-eta0.05 | FCAdam5 | 0.7383 | 0.7578 | -0.0195 | 0.1229 | 33.1104 | yes |
| Fashion-MNIST | PureKAN-AdamW | P2best-NFS-role-block-cg5-eta0.05 | D6-5 | 0.7734 | 0.7871 | -0.0137 | 0.1189 | 47.2846 | yes |
| Fashion-MNIST | D6-allTaskAware | P2best-NFS-role-block-cg5-eta0.05 | FCAdam10 | 0.7383 | 0.7539 | -0.0156 | 0.1133 | 60.0243 | yes |
| Fashion-MNIST | D6-allTaskAware | P2best-NFS-role-block-diag-eta0.05 | none | 0.7383 | 0.7363 | 0.0020 | 0.1530 | 0.0000 | no |
| Fashion-MNIST | PureKAN-AdamW | P2best-NFS-role-block-diag-eta0.05 | none | 0.7734 | 0.7754 | -0.0020 | 0.1369 | 0.0000 | no |
| Fashion-MNIST | D6-allTaskAware | P2best-NFS-role-block-cg5-eta0.05 | none | 0.7383 | 0.7363 | 0.0020 | 0.1330 | 0.0000 | no |
| Fashion-MNIST | PureKAN-AdamW | P2best-NFS-role-block-diag-eta0.05 | FCAdam5 | 0.7734 | 0.7695 | 0.0039 | 0.1251 | -106.0962 | no |
| Fashion-MNIST | PureKAN-AdamW | P2best-NFS-role-block-cg5-eta0.05 | none | 0.7734 | 0.7754 | -0.0020 | 0.1186 | 0.0000 | no |
| Fashion-MNIST | PureKAN-AdamW | P2best-NFS-role-block-diag-eta0.05 | FCAdam10 | 0.7734 | 0.7852 | -0.0117 | 0.1135 | -116.1521 | no |
| Fashion-MNIST | PureKAN-AdamW | P2best-NFS-role-block-cg5-eta0.05 | FCAdam5 | 0.7734 | 0.7695 | 0.0039 | 0.1070 | -157.4310 | no |
| Fashion-MNIST | PureKAN-AdamW | P2best-NFS-role-block-cg5-eta0.05 | FCAdam10 | 0.7734 | 0.7871 | -0.0137 | 0.0958 | -172.0602 | no |
| Fashion-MNIST | FCAdam-dataSob | P2best-NFS-role-block-cg5-eta0.05 | D6-5 | 0.7578 | 0.7559 | 0.0020 | 0.0796 | 971078872.6807 | no |
| Fashion-MNIST | FCAdam-dataSob | P2best-NFS-role-block-diag-eta0.05 | D6-5 | 0.7578 | 0.7559 | 0.0020 | 0.0796 | 969409942.6270 | no |
| Fashion-MNIST | FCAdam-dataSob | P2best-NFS-role-block-cg5-eta0.05 | none | 0.7578 | 0.7559 | 0.0020 | 0.0795 | 0.0000 | no |
| Fashion-MNIST | FCAdam-dataSob | P2best-NFS-role-block-diag-eta0.05 | none | 0.7578 | 0.7539 | 0.0039 | 0.0794 | 0.0000 | no |
| Fashion-MNIST | FLD-AdanLite | P2best-NFS-role-block-cg5-eta0.05 | none | 0.7578 | 0.7559 | 0.0020 | 0.0726 | 0.0000 | no |
| Fashion-MNIST | FLD-AdanLite | P2best-NFS-role-block-cg5-eta0.05 | D6-5 | 0.7578 | 0.7520 | 0.0059 | 0.0725 | 4901826381.6833 | no |
| Fashion-MNIST | FLD-AdanLite | P2best-NFS-role-block-diag-eta0.05 | none | 0.7578 | 0.7559 | 0.0020 | 0.0724 | 0.0000 | no |
| Fashion-MNIST | FLD-AdanLite | P2best-NFS-role-block-diag-eta0.05 | D6-5 | 0.7578 | 0.7520 | 0.0059 | 0.0723 | 4887402057.6477 | no |
| Fashion-MNIST | FCAdam-dataSob | P2best-NFS-role-block-cg5-eta0.05 | FCAdam5 | 0.7578 | 0.7676 | -0.0098 | 0.0682 | -116045057773.5901 | no |
| Fashion-MNIST | FCAdam-dataSob | P2best-NFS-role-block-diag-eta0.05 | FCAdam5 | 0.7578 | 0.7676 | -0.0098 | 0.0682 | -116060197353.3630 | no |
| Fashion-MNIST | FLD-AdanLite | P2best-NFS-role-block-cg5-eta0.05 | FCAdam5 | 0.7578 | 0.7305 | 0.0273 | 0.0667 | -137469112873.0774 | no |
| Fashion-MNIST | FLD-AdanLite | P2best-NFS-role-block-diag-eta0.05 | FCAdam5 | 0.7578 | 0.7305 | 0.0273 | 0.0665 | -137489855289.4592 | no |
| Fashion-MNIST | FCAdam-L2 | P2best-NFS-role-block-diag-eta0.05 | D6-5 | 0.7480 | 0.7656 | -0.0176 | 0.0654 | 54638564586.6394 | no |
| Fashion-MNIST | FCAdam-L2 | P2best-NFS-role-block-cg5-eta0.05 | D6-5 | 0.7480 | 0.7656 | -0.0176 | 0.0652 | 54693877696.9910 | no |
| Fashion-MNIST | FCAdam-L2 | P2best-NFS-role-block-diag-eta0.05 | none | 0.7480 | 0.7559 | -0.0078 | 0.0647 | 0.0000 | no |
| Fashion-MNIST | FCAdam-L2 | P2best-NFS-role-block-cg5-eta0.05 | none | 0.7480 | 0.7559 | -0.0078 | 0.0646 | 0.0000 | no |
| Fashion-MNIST | FCAdam-L2 | P2best-NFS-role-block-diag-eta0.05 | FCAdam5 | 0.7480 | 0.7461 | 0.0020 | 0.0569 | -17277181148.5291 | no |
| Fashion-MNIST | FCAdam-L2 | P2best-NFS-role-block-cg5-eta0.05 | FCAdam5 | 0.7480 | 0.7480 | 0.0000 | 0.0569 | -17116785049.4385 | no |
| Fashion-MNIST | FLD-AdanLite | P2best-NFS-role-block-cg5-eta0.05 | FCAdam10 | 0.7578 | 0.7500 | 0.0078 | 0.0540 | -65561056137.0850 | no |
| Fashion-MNIST | FLD-AdanLite | P2best-NFS-role-block-diag-eta0.05 | FCAdam10 | 0.7578 | 0.7500 | 0.0078 | 0.0539 | -65584719181.0608 | no |
| Fashion-MNIST | FCAdam-dataSob | P2best-NFS-role-block-cg5-eta0.05 | FCAdam10 | 0.7578 | 0.7676 | -0.0098 | 0.0501 | -88990688323.9746 | no |
| Fashion-MNIST | FCAdam-dataSob | P2best-NFS-role-block-diag-eta0.05 | FCAdam10 | 0.7578 | 0.7676 | -0.0098 | 0.0501 | -89143216609.9548 | no |
| Fashion-MNIST | FCAdam-L2 | P2best-NFS-role-block-diag-eta0.05 | FCAdam10 | 0.7480 | 0.7617 | -0.0137 | 0.0440 | 4307031631.4697 | no |
| Fashion-MNIST | FCAdam-L2 | P2best-NFS-role-block-cg5-eta0.05 | FCAdam10 | 0.7480 | 0.7617 | -0.0137 | 0.0439 | 4423618316.6504 | no |
| KMNIST | D6-allTaskAware | P2best-NFS-role-block-diag-eta0.05 | D6-5 | 0.4570 | 0.5469 | -0.0898 | 0.1344 | 55.5976 | yes |
| KMNIST | D6-allTaskAware | P2best-NFS-role-block-cg5-eta0.05 | D6-5 | 0.4570 | 0.5469 | -0.0898 | 0.1341 | 55.8216 | yes |
| KMNIST | PureKAN-AdamW | P2best-NFS-role-block-diag-eta0.05 | D6-5 | 0.6953 | 0.7031 | -0.0078 | 0.1319 | 4.3098 | yes |
| KMNIST | D6-allTaskAware | P2best-NFS-role-block-diag-eta0.05 | FCAdam5 | 0.4570 | 0.5547 | -0.0977 | 0.1247 | 52.7462 | yes |
| KMNIST | D6-allTaskAware | P2best-NFS-role-block-cg5-eta0.05 | FCAdam5 | 0.4570 | 0.5547 | -0.0977 | 0.1244 | 52.9578 | yes |
| KMNIST | PureKAN-AdamW | P2best-NFS-role-block-cg5-eta0.05 | D6-5 | 0.6953 | 0.7031 | -0.0078 | 0.1144 | 5.0371 | yes |
| KMNIST | D6-allTaskAware | P2best-NFS-role-block-diag-eta0.05 | FCAdam10 | 0.4570 | 0.6211 | -0.1641 | 0.1108 | 88.0937 | yes |
| KMNIST | D6-allTaskAware | P2best-NFS-role-block-cg5-eta0.05 | FCAdam10 | 0.4570 | 0.6211 | -0.1641 | 0.1105 | 88.4503 | yes |
| KMNIST | D6-allTaskAware | P2best-NFS-role-block-diag-eta0.05 | none | 0.4570 | 0.4453 | 0.0117 | 0.1331 | 0.0000 | no |
| KMNIST | D6-allTaskAware | P2best-NFS-role-block-cg5-eta0.05 | none | 0.4570 | 0.4453 | 0.0117 | 0.1327 | 0.0000 | no |
| KMNIST | PureKAN-AdamW | P2best-NFS-role-block-diag-eta0.05 | none | 0.6953 | 0.6914 | 0.0039 | 0.1319 | 0.0000 | no |
| KMNIST | PureKAN-AdamW | P2best-NFS-role-block-diag-eta0.05 | FCAdam5 | 0.6953 | 0.6973 | -0.0020 | 0.1184 | 5.5609 | no |
| KMNIST | PureKAN-AdamW | P2best-NFS-role-block-cg5-eta0.05 | none | 0.6953 | 0.6934 | 0.0020 | 0.1145 | 0.0000 | no |
| KMNIST | PureKAN-AdamW | P2best-NFS-role-block-diag-eta0.05 | FCAdam10 | 0.6953 | 0.6953 | 0.0000 | 0.1053 | 3.7286 | no |
| KMNIST | PureKAN-AdamW | P2best-NFS-role-block-cg5-eta0.05 | FCAdam5 | 0.6953 | 0.6973 | -0.0020 | 0.1009 | 6.4375 | no |
| KMNIST | PureKAN-AdamW | P2best-NFS-role-block-cg5-eta0.05 | FCAdam10 | 0.6953 | 0.6934 | 0.0020 | 0.0885 | 4.2107 | no |
| KMNIST | FLD-AdanLite | P2best-NFS-role-block-cg5-eta0.05 | D6-5 | 0.5957 | 0.5996 | -0.0039 | 0.0580 | 1392304897.3083 | no |
| KMNIST | FLD-AdanLite | P2best-NFS-role-block-cg5-eta0.05 | none | 0.5957 | 0.5996 | -0.0039 | 0.0579 | 0.0000 | no |
| KMNIST | FCAdam-L2 | P2best-NFS-role-block-diag-eta0.05 | none | 0.6504 | 0.6504 | 0.0000 | 0.0578 | 0.0000 | no |
| KMNIST | FCAdam-L2 | P2best-NFS-role-block-cg5-eta0.05 | none | 0.6504 | 0.6504 | 0.0000 | 0.0575 | 0.0000 | no |
| KMNIST | FLD-AdanLite | P2best-NFS-role-block-diag-eta0.05 | D6-5 | 0.5957 | 0.5996 | -0.0039 | 0.0574 | 1381993293.7622 | no |
| KMNIST | FLD-AdanLite | P2best-NFS-role-block-diag-eta0.05 | none | 0.5957 | 0.5996 | -0.0039 | 0.0574 | 0.0000 | no |
| KMNIST | FCAdam-dataSob | P2best-NFS-role-block-diag-eta0.05 | D6-5 | 0.5977 | 0.5898 | 0.0078 | 0.0568 | 382900238.0371 | no |
| KMNIST | FCAdam-dataSob | P2best-NFS-role-block-diag-eta0.05 | none | 0.5977 | 0.5898 | 0.0078 | 0.0567 | 0.0000 | no |
| KMNIST | FCAdam-L2 | P2best-NFS-role-block-diag-eta0.05 | D6-5 | 0.6504 | 0.6543 | -0.0039 | 0.0566 | 14720141887.6648 | no |
| KMNIST | FCAdam-dataSob | P2best-NFS-role-block-cg5-eta0.05 | D6-5 | 0.5977 | 0.5898 | 0.0078 | 0.0564 | 379681587.2192 | no |
| KMNIST | FCAdam-dataSob | P2best-NFS-role-block-cg5-eta0.05 | none | 0.5977 | 0.5898 | 0.0078 | 0.0563 | 0.0000 | no |
| KMNIST | FCAdam-L2 | P2best-NFS-role-block-cg5-eta0.05 | D6-5 | 0.6504 | 0.6543 | -0.0039 | 0.0563 | 14726161956.7871 | no |
| KMNIST | FCAdam-L2 | P2best-NFS-role-block-diag-eta0.05 | FCAdam5 | 0.6504 | 0.5781 | 0.0723 | 0.0481 | -171368062496.1853 | no |
| KMNIST | FCAdam-L2 | P2best-NFS-role-block-cg5-eta0.05 | FCAdam5 | 0.6504 | 0.5781 | 0.0723 | 0.0477 | -171245813369.7510 | no |
| KMNIST | FLD-AdanLite | P2best-NFS-role-block-cg5-eta0.05 | FCAdam5 | 0.5957 | 0.6016 | -0.0059 | 0.0446 | -63986122608.1848 | no |
| KMNIST | FLD-AdanLite | P2best-NFS-role-block-diag-eta0.05 | FCAdam5 | 0.5957 | 0.6016 | -0.0059 | 0.0440 | -65106213092.8040 | no |

P4 all-dataset survivors:

```text
PureKAN-AdamW|P2best-NFS-role-block-diag-eta0.05|NFS-role-block|D6-5, PureKAN-AdamW|P2best-NFS-role-block-cg5-eta0.05|NFS-role-block|D6-5
```

## P5 Alternating TAN Cycles

| dataset | task | NFS | refresh | cycles | final acc | final phi | rank | margin | P5 |
|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | PureKAN-AdamW | P2best-NFS-role-block-cg5-eta0.05 | D6-5 | 3 | 0.7773 | 0.1199 | 12.5031 | 1.9166 | yes |
| Fashion-MNIST | PureKAN-AdamW | P2best-NFS-role-block-diag-eta0.05 | D6-5 | 3 | 0.7773 | 0.1143 | 12.2508 | 1.9063 | yes |
| KMNIST | PureKAN-AdamW | P2best-NFS-role-block-cg5-eta0.05 | D6-5 | 3 | 0.6621 | 0.1319 | 15.7638 | 1.8226 | yes |
| KMNIST | PureKAN-AdamW | P2best-NFS-role-block-diag-eta0.05 | D6-5 | 3 | 0.6621 | 0.1295 | 15.6598 | 1.8213 | yes |
| MNIST | PureKAN-AdamW | P2best-NFS-role-block-cg5-eta0.05 | D6-5 | 3 | 0.8242 | 0.1489 | 12.6641 | 1.8399 | no |
| MNIST | PureKAN-AdamW | P2best-NFS-role-block-diag-eta0.05 | D6-5 | 3 | 0.8242 | 0.1483 | 12.6372 | 1.8385 | no |

P5 all-dataset survivors:

```text
none
```

## P6 Capacity / Basis Expansion

_P6 not run yet._

P6 all-dataset survivors:

```text
none
```

## P3-P9 Decision

```text
P3 functional-teacher NFS: run
P4 TAN one-cycle: run
P5 alternating TAN cycles: run
P7/P8/P9 confirm: not run; P5 no all-dataset survivor
Final decision: stop_after_p5_no_multicycle_survivor
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
F1 checks whether an accurate and smooth PureKAN solution is visible under AdamW-style geometry regularization.
F2 checks whether NFS can lower geometry in a teacher-preserving tangent/nullspace.
F4 checks whether smoothing destroys logits/features or accuracy.
F5 checks whether refresh and alternating cycles can retain the one-cycle geometry gain.
F6 remains the capacity/basis fallback, but it was not triggered because P2/P4 already gave mechanistic survivors.

Interpretation:
  P1 did not expose an AdamW+regularization smooth frontier.
  P2/P3 showed NFS role-block can preserve function while reducing geometry.
  P4 showed one-cycle TAN can pass across datasets.
  P5 showed the same recipe does not yet survive alternating cycles on MNIST.
```

## Required Artifacts

Written under `results/v4_8/`:

```text
p0_fgf_invariants.csv
p1_feasibility_frontier.csv
p1_training_trace.csv
p1_frontier_gate_summary.csv
p2_nfs_adamw_projection.csv
p2_nfs_gate_summary.csv
p3_nfs_functional_teachers.csv
p3_nfs_functional_gate_summary.csv
p4_tan_one_cycle_micro_run.csv
p4_tan_one_cycle_gate_summary.csv
p5_tan_alternating_cycles.csv
p5_tan_cycle_gate_summary.csv
p6_capacity_basis_expansion.csv
p6_capacity_gate_summary.csv
p7_candidate_selection.csv
p8_confirm5.csv
p9_confirm10.csv
p9_failure_diagnosis.csv
failure_table.csv
aggregate_decision.json
figures/
```

## Final Decision

```text
DG-KAN v4.8 status:
  stop_after_p5_no_multicycle_survivor

What improved:
  v4.8 now asks the right feasibility question before spending seed budget.
  P1 exposes the accuracy-geometry Pareto frontier directly.
  P2 measures function-preserving smoothability instead of relying on unconstrained Sobolev consolidation.
  P4 confirms a real one-cycle TAN signal with PureKAN-AdamW task teacher + role-block NFS + D6 refresh.

What failed / remains open:
  P5 fails the all-dataset multi-cycle gate because MNIST does not preserve the task/geometry tradeoff over 3 cycles.

Conclusion:
  NFS is feasible as a local projection, and TAN is viable for one cycle, but the current alternating controller is not stable enough for P7 seed confirmation.
  Next work should redesign the multi-cycle controller / refresh policy before spending 5/10-seed budget.
```
