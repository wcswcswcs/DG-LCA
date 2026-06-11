# DG-KAN v16.3 MultiSchemeFunctionalDynamics EfficiencyCensus 4GPU 实验结果复盘

生成时间：2026-06-02 01:41:17（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；Line P timing 只作为 efficiency/engineering 证据，不作为 functional promotion。

## 1. 计划理解

v16.3 的新增核心是 Line P：对 D-CHE、D-FOU、D-RBF/FastKAN、D-WAV、Rational、LQ 和 MLP 做 same-param efficiency census，拆 forward/backward/update/functional/audit/memory 组件；同时保留 v16.2 的 A/B/C/D/F functional matrix 和 no-action/no-controller 边界。

## 2. 本轮代码修改

新增：
```text
experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py
```

复用：
```text
experiments/run_v162_multischeme_functional_dynamics_4gpu.py  # M1a..M8f train-stream surface
experiments/run_v149_line_d_all_basis_substrate_repair.py      # Line F substrate rows
```

过程说明：
```text
1. Line P 执行 P0..P8 x batch 8/32/128 phase-level timing，并写入 fallback P-FB1..P-FB6 columns。
2. D-CHE 与 MLP 仍执行 M1a..M8f mechanism family，并保留 matched controls。
3. LineC/tail/calibration/AUC 只作为 readback/audit/gate，不生成方向。
4. Rational 只做 monitor/sanity replay；不启动 reset/controller/action route。
5. D-FOU/RBF/WAV 只做 substrate-only repair gate；不进入 official FU proof。
6. 四卡分片执行由 v163_gpu_assignment_manifest.csv 记录；cpu_offload_used=0。
7. runtime blocker 修复：Line P optimizer-update phase 改为持久 AdamW optimizer + live gradients 计时，避免 warmup 后无梯度 empty step；修复后重跑 v163_efficiency_census.csv。
```

## 2.1 人工复核分析 / Insight

Line P rows=27，exploration pass rows=6，official pass rows=5。Line P 没有改写 functional route，它只说明哪些 carrier/phase 的 runtime/memory 成本可承受。
全局 best h800 source 来自 `MLP` / `B-M4a-MLP-AdamSubspaceProximal-alpha000`，source=0.14581246508492363，retention=0.8296399083402421，tail=0.01966376412104744，LineC=0.0，AUC=1.0373429473020996。
S2 weak rows=0，S3 productive rows=0，promotion_allowed=0。
Line M attribution rows=55，KAN_specific_advantage_rows=0，不能把局部 D-CHE source 写成 KAN-specific promotion。
Line C LQ gate=0，best=C4-LQ-MacroDeltaPriorityReanchor，near=9/9。
Line F all-basis gate=0，best_family=D-FOU，pass=0/9。
h1600 extension readback：A-line A-M1k-D-CHE-MidOnlyPulse-AdamWRecovery: source=0.0, bad=0.8888888888888888, tail=0.35597136078213754, LineC=0.4444444444444444; A-M7b-D-CHE-LateAttachMidCheckpoint: source=0.0, bad=0.8888888888888888, tail=0.35597136078213754, LineC=0.4444444444444444；B-line B-M4a-MLP-AdamSubspaceProximal-alpha000: source=0.0, bad=1.0, tail=0.14282379020659178, LineC=0.0; B-M4b-MLP-PerExampleLowRankProximal-alpha025: source=-0.04176027907265557, bad=1.0, tail=0.12097317708890282, LineC=0.0。
最强 h800 row 没有打开 S2/S3 的直接证据是 tail=0.01966376412104744、LineC=0.0，且 best_carrier=MLP 不是 KAN-specific promotion 证据；retention=0.8296399083402421 只说明 source 没有单独坍塌，不能替代 debt recovery。
最终 route=R6-FunctionalDynamicsAllMechanismsNoGo，这是 Line P coverage、A/B/C/D/F gates、forbidden/no-action audits 与 required manifest 同时闭合后的结果。

## 2.2 完整计划执行对照（artifact 自动写入）

required / forbidden / no-action / provenance：
```text
required_artifact_manifest_rows = 40
required_artifact_missing_rows = 0
forbidden_information_violation_sum = 0
no_action_search_violation_sum = 0
direction_provenance_rows = 5994
direction_source = train_stream_only / optimizer_state / split_gradient; audit metrics not used for direction
```

## 3. Line P efficiency census

| subject | batch | fwd ratio | bwd ratio | upd ratio | mem ratio | step ratio | explore | official |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P8-MLP-functional-path-top2 | 128 | 0.9351369769660625 | 0.6233412123546986 | 0.8293382202361886 | 0.9971094989798232 | 0.9298089989287891 | 1 | 1 |
| P8-MLP-functional-path-top2 | 32 | 1.0579755041594572 | 0.6034694701730535 | 1.5153227440399044 | 0.9971079418185942 | 0.9399598602323048 | 1 | 0 |
| P0-MLP-same-param-reference | 8 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1 | 1 |
| P0-MLP-same-param-reference | 32 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1 | 1 |
| P0-MLP-same-param-reference | 128 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1 | 1 |
| P8-MLP-functional-path-top2 | 8 | 1.1707169004385487 | 0.6855964379494667 | 1.2494678126063976 | 0.9971075317604355 | 1.042052792028299 | 1 | 1 |
| P2-D-FOU-current | 32 | 10.206566900761114 | 1.8895866383008697 | 1.2760172047263558 | 1.0006651932279038 | 1.2909807912828777 | 0 | 0 |
| P6-LQ-current-or-reanchored | 8 | 6.642720714882587 | 1.5380342281276562 | 2.3977604809825275 | 0.9879597197898424 | 1.3040030578034578 | 0 | 0 |

Line P blocked / fallback rows（只列前 40）：

| subject | batch | reason | dominant | P-FB1 | P-FB6 |
| --- | --- | --- | --- | --- | --- |
| P1-D-CHE-current | 8 | forward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 | audit_readback | 1 | 1 |
| P1-D-CHE-current | 32 | forward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 | audit_readback | 1 | 1 |
| P1-D-CHE-current | 128 | forward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 | audit_readback | 1 | 1 |
| P2-D-FOU-current | 8 | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75 | backward | 1 | 1 |
| P2-D-FOU-current | 32 | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75 | audit_readback | 1 | 1 |
| P2-D-FOU-current | 128 | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75 | backward | 1 | 1 |
| P3-D-RBF-FastKAN-current | 8 | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75 | backward | 1 | 1 |
| P3-D-RBF-FastKAN-current | 32 | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 | backward | 1 | 1 |
| P3-D-RBF-FastKAN-current | 128 | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 | audit_readback | 1 | 1 |
| P4-D-WAV-current | 8 | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 | backward | 1 | 1 |
| P4-D-WAV-current | 32 | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 | backward | 1 | 1 |
| P4-D-WAV-current | 128 | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 | backward | 1 | 1 |
| P5-Rational-current | 8 | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75 | backward | 1 | 1 |
| P5-Rational-current | 32 | forward_ratio_vs_same_param_mlp>1.75 | backward | 1 | 1 |
| P5-Rational-current | 128 | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75 | audit_readback | 1 | 1 |
| P6-LQ-current-or-reanchored | 8 | forward_ratio_vs_same_param_mlp>1.75;optimizer_update_ratio_vs_same_param_mlp>1.75 | backward | 1 | 1 |
| P6-LQ-current-or-reanchored | 32 | forward_ratio_vs_same_param_mlp>1.75;optimizer_update_ratio_vs_same_param_mlp>1.75 | audit_readback | 1 | 1 |
| P6-LQ-current-or-reanchored | 128 | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;optimizer_update_ratio_vs_same_param_mlp>1.75 | backward | 1 | 1 |
| P7-D-CHE-functional-path-top2 | 8 | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 | audit_readback | 1 | 1 |
| P7-D-CHE-functional-path-top2 | 32 | forward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 | audit_readback | 1 | 1 |
| P7-D-CHE-functional-path-top2 | 128 | forward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 | audit_readback | 1 | 1 |

## 4. Carrier × mechanism h800 summary

```text
best_carrier = MLP
best_mechanism_family = M4-FunctionSpaceProximal
best_method = B-M4a-MLP-AdamSubspaceProximal-alpha000
S2_weak_productive_dynamics_reached = 0
S3_productive_debt_recovery_reached = 0
```

| carrier | mechanism | best method | source | retention | tail | LineC | AUC | S2 | S3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | M1-ProductivePulseRecovery | A-M1k-D-CHE-MidOnlyPulse-AdamWRecovery | 0.011553770966000028 | 0.3313499175839954 | 0.0 | 0.5555555555555556 | 0.8240228348029618 | 0 | 0 |
| D-CHE | M2-SplitConsensusSignalSubspace | A-M2a-D-CHE-SplitConsensusDiagMetric | -0.003950059413909912 | 0.6622720393869612 | 0.0 | 0.5555555555555556 | 1.0022949381791681 | 0 | 0 |
| D-CHE | M3-PopRiskDriftSNR | A-M3e-D-CHE-SNRRecoveryScheduler | 0.00030302339129977755 | 0.6587435636255476 | 0.0 | 0.6666666666666666 | 0.997345948008792 | 0 | 0 |
| D-CHE | M4-FunctionSpaceProximal | A-M4e-D-CHE-RecoveryAwareProximal-alpha200 | -0.00742564598719279 | 0.6590441034899818 | 0.0 | 0.6666666666666666 | 1.0162779129524935 | 0 | 0 |
| D-CHE | M5-OptimizerAwareAlignedFU | A-M5c-D-CHE-MGUPStyleReweightFU | 0.0017752382490370008 | 0.36246414482593536 | 0.028983882948097554 | 0.5555555555555556 | 1.0172114389061366 | 0 | 0 |
| D-CHE | M6-CarrierSpecificReparamFU | A-M6d-D-CHE-OutputJacobianCarrier | -0.003950059413909912 | 0.6622720393869612 | 0.0 | 0.5555555555555556 | 1.0022949381791681 | 0 | 0 |
| D-CHE | M7-LateAttachSnapshotFU | A-M7b-D-CHE-LateAttachMidCheckpoint | 0.011553770966000028 | 0.3313499175839954 | 0.0 | 0.5555555555555556 | 0.8240228348029618 | 0 | 0 |
| D-CHE | M8-RecoveryOnlyDeconfound | A-M8d-D-CHE-EMASWARecoveryOnly | 0.0024184385935465493 | 0.3568440311484867 | 0.0 | 0.6666666666666666 | 0.9952999692018842 | 0 | 0 |
| MLP | M1-ProductivePulseRecovery | B-M1b-MLP-PulseOnce-LRCooldownRecovery | 0.042249851756625705 | 0.6911552084816827 | 0.0 | 0.0 | 1.0171674683285459 | 0 | 0 |
| MLP | M2-SplitConsensusSignalSubspace | B-M2b-MLP-HiddenSplitConsensusRoleBlockMetric | -0.08094873693254259 | 0.6327619221475389 | 4.0141583012519976e-05 | 0.0 | 1.0165849527747042 | 0 | 0 |
| MLP | M3-PopRiskDriftSNR | B-M3e-MLP-SNRRecoveryScheduler | -0.045729511313968234 | 0.8547681536939409 | 0.008586558760465293 | 0.0 | 0.9936044585754902 | 0 | 0 |
| MLP | M4-FunctionSpaceProximal | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.14581246508492363 | 0.8296399083402421 | 0.01966376412104744 | 0.0 | 1.0373429473020996 | 0 | 0 |
| MLP | M5-OptimizerAwareAlignedFU | B-M5g-MLP-DecoupledDecayPlusFU | -0.07765578561358982 | 0.630284716685613 | 0.0005459059760351227 | 0.0 | 1.0160851508412163 | 0 | 0 |
| MLP | M6-CarrierSpecificReparamFU | B-M6e-MLP-DualBankSignalReservoirCarrier | -0.07765578561358982 | 0.630284716685613 | 0.0005459059760351227 | 0.0 | 1.0160851508412163 | 0 | 0 |
| MLP | M7-LateAttachSnapshotFU | B-M7e-MLP-LateAttachHighSourceLowDebtWindow | 0.027479873763190374 | 0.6914904216925303 | 0.0 | 0.0 | 0.7005985681032758 | 0 | 0 |
| MLP | M8-RecoveryOnlyDeconfound | B-M8b-MLP-LRCooldownOnly | 0.06201504998736911 | 0.6907538937197791 | 0.0 | 0.0 | 1.018283152855018 | 0 | 0 |

Line A/B h1600 top rows：

| carrier | method | dataset | seed | source | bad | tail | LineC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | A-M1k-D-CHE-MidOnlyPulse-AdamWRecovery | MNIST | 0 | 0.0 | 1 | 1.0 | 0.0 |
| D-CHE | A-M7b-D-CHE-LateAttachMidCheckpoint | MNIST | 0 | 0.0 | 1 | 1.0 | 0.0 |
| D-CHE | A-M1k-D-CHE-MidOnlyPulse-AdamWRecovery | MNIST | 1 | 0.0 | 1 | 0.0 | 1.0 |
| D-CHE | A-M7b-D-CHE-LateAttachMidCheckpoint | MNIST | 1 | 0.0 | 1 | 0.0 | 1.0 |
| D-CHE | A-M1k-D-CHE-MidOnlyPulse-AdamWRecovery | MNIST | 2 | 0.0 | 0 | 1.0 | 1.0 |
| D-CHE | A-M7b-D-CHE-LateAttachMidCheckpoint | MNIST | 2 | 0.0 | 0 | 1.0 | 1.0 |
| D-CHE | A-M1k-D-CHE-MidOnlyPulse-AdamWRecovery | Fashion-MNIST | 0 | 0.0 | 1 | 0.17081939582895914 | 1.0 |
| D-CHE | A-M7b-D-CHE-LateAttachMidCheckpoint | Fashion-MNIST | 0 | 0.0 | 1 | 0.17081939582895914 | 1.0 |
| D-CHE | A-M1k-D-CHE-MidOnlyPulse-AdamWRecovery | Fashion-MNIST | 1 | 0.0 | 1 | 1.0 | 0.0 |
| D-CHE | A-M7b-D-CHE-LateAttachMidCheckpoint | Fashion-MNIST | 1 | 0.0 | 1 | 1.0 | 0.0 |
| D-CHE | A-M1k-D-CHE-MidOnlyPulse-AdamWRecovery | Fashion-MNIST | 2 | 0.0 | 1 | 0.032922851210278645 | 0.0 |
| D-CHE | A-M7b-D-CHE-LateAttachMidCheckpoint | Fashion-MNIST | 2 | 0.0 | 1 | 0.032922851210278645 | 0.0 |
| D-CHE | A-M1k-D-CHE-MidOnlyPulse-AdamWRecovery | KMNIST | 0 | 0.0 | 1 | 0.0 | 1.0 |
| D-CHE | A-M7b-D-CHE-LateAttachMidCheckpoint | KMNIST | 0 | 0.0 | 1 | 0.0 | 1.0 |
| D-CHE | A-M1k-D-CHE-MidOnlyPulse-AdamWRecovery | KMNIST | 1 | 0.0 | 1 | 0.0 | 0.0 |
| D-CHE | A-M7b-D-CHE-LateAttachMidCheckpoint | KMNIST | 1 | 0.0 | 1 | 0.0 | 0.0 |
| D-CHE | A-M1k-D-CHE-MidOnlyPulse-AdamWRecovery | KMNIST | 2 | 0.0 | 1 | 0.0 | 0.0 |
| D-CHE | A-M7b-D-CHE-LateAttachMidCheckpoint | KMNIST | 2 | 0.0 | 1 | 0.0 | 0.0 |
| MLP | B-M4a-MLP-AdamSubspaceProximal-alpha000 | MNIST | 0 | 0.0 | 1 | 0.0 | 0.0 |
| MLP | B-M4b-MLP-PerExampleLowRankProximal-alpha025 | MNIST | 0 | -0.017630577087402344 | 1 | 0.0 | 0.0 |
| MLP | B-M4a-MLP-AdamSubspaceProximal-alpha000 | MNIST | 1 | 0.0 | 1 | 0.0 | 0.0 |
| MLP | B-M4b-MLP-PerExampleLowRankProximal-alpha025 | MNIST | 1 | -0.031209945678710938 | 1 | 0.0 | 0.0 |
| MLP | B-M4a-MLP-AdamSubspaceProximal-alpha000 | MNIST | 2 | 0.0 | 1 | 0.0 | 0.0 |
| MLP | B-M4b-MLP-PerExampleLowRankProximal-alpha025 | MNIST | 2 | -0.014403581619262695 | 1 | 0.0 | 0.0 |
| MLP | B-M4a-MLP-AdamSubspaceProximal-alpha000 | Fashion-MNIST | 0 | 0.0 | 1 | 1.0 | 0.0 |
| MLP | B-M4b-MLP-PerExampleLowRankProximal-alpha025 | Fashion-MNIST | 0 | -0.0764971375465393 | 1 | 1.0 | 0.0 |
| MLP | B-M4a-MLP-AdamSubspaceProximal-alpha000 | Fashion-MNIST | 1 | 0.0 | 1 | 0.285414111859326 | 0.0 |
| MLP | B-M4b-MLP-PerExampleLowRankProximal-alpha025 | Fashion-MNIST | 1 | -0.09208858013153076 | 1 | 0.08875859380012537 | 0.0 |
| MLP | B-M4a-MLP-AdamSubspaceProximal-alpha000 | Fashion-MNIST | 2 | 0.0 | 1 | 0.0 | 0.0 |
| MLP | B-M4b-MLP-PerExampleLowRankProximal-alpha025 | Fashion-MNIST | 2 | -0.03213024139404297 | 1 | 0.0 | 0.0 |
| MLP | B-M4a-MLP-AdamSubspaceProximal-alpha000 | KMNIST | 0 | 0.0 | 1 | 0.0 | 0.0 |
| MLP | B-M4b-MLP-PerExampleLowRankProximal-alpha025 | KMNIST | 0 | -0.04903674125671387 | 1 | 0.0 | 0.0 |
| MLP | B-M4a-MLP-AdamSubspaceProximal-alpha000 | KMNIST | 1 | 0.0 | 1 | 0.0 | 0.0 |
| MLP | B-M4b-MLP-PerExampleLowRankProximal-alpha025 | KMNIST | 1 | -0.048119544982910156 | 1 | 0.0 | 0.0 |
| MLP | B-M4a-MLP-AdamSubspaceProximal-alpha000 | KMNIST | 2 | 0.0 | 1 | 0.0 | 0.0 |
| MLP | B-M4b-MLP-PerExampleLowRankProximal-alpha025 | KMNIST | 2 | -0.01472616195678711 | 1 | 0.0 | 0.0 |

## 5. Line C / D / F / M results

```text
line_c_rows = 63
line_c_reanchor_gate_pass = 0
line_c_best_method = C4-LQ-MacroDeltaPriorityReanchor
line_d_rat_rows = 45
line_d_rat_best_method = D1-RAT-G7AnalogPulseOnce-then-AdamWRecovery
line_f_rows = 153
line_f_best_family = D-FOU
line_m_rows = 55
kan_specific_advantage_rows = 0
```

Line M attribution summary（前 80）：

| mechanism | KAN method | best MLP | KAN source | MLP source | delta | KAN adv |
| --- | --- | --- | --- | --- | --- | --- |
| M1-ProductivePulseRecovery | A-M1a-D-CHE-PulseOnce-AdamWRecovery | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.006111714575025771 | 0.14581246508492363 | -0.1519241796599494 | 0 |
| M1-ProductivePulseRecovery | A-M1b-D-CHE-PulseOnce-LRCooldownRecovery | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.0790273944536845 | 0.14581246508492363 | -0.22483985953860813 | 0 |
| M1-ProductivePulseRecovery | A-M1c-D-CHE-PulseOnce-MomentumDampingRecovery | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.0002638962533738878 | 0.14581246508492363 | -0.14607636133829752 | 0 |
| M1-ProductivePulseRecovery | A-M1d-D-CHE-PulseOnce-DecoupledGlobalDecayRecovery | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.006374849213494195 | 0.14581246508492363 | -0.1521873142984178 | 0 |
| M1-ProductivePulseRecovery | A-M1e-D-CHE-PulseOnce-DegreeWiseDecayRecovery | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.00624284479353163 | 0.14581246508492363 | -0.15205530987845525 | 0 |
| M1-ProductivePulseRecovery | A-M1f-D-CHE-PulseOnce-EMAConsolidation | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.006536993715498183 | 0.14581246508492363 | -0.13927547136942545 | 0 |
| M1-ProductivePulseRecovery | A-M1g-D-CHE-PulseOnce-LookaheadConsolidation | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.007317953639560276 | 0.14581246508492363 | -0.13849451144536334 | 0 |
| M1-ProductivePulseRecovery | A-M1h-D-CHE-PulseOnce-SWAConsolidation | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.008660276730855307 | 0.14581246508492363 | -0.1371521883540683 | 0 |
| M1-ProductivePulseRecovery | A-M1i-D-CHE-PulseEvery50-AdamWRecovery | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.0038753681712680394 | 0.14581246508492363 | -0.14968783325619167 | 0 |
| M1-ProductivePulseRecovery | A-M1j-D-CHE-EarlyOnlyPulse-AdamWRecovery | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.0008363723754882812 | 0.14581246508492363 | -0.1466488374604119 | 0 |
| M1-ProductivePulseRecovery | A-M1k-D-CHE-MidOnlyPulse-AdamWRecovery | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.011553770966000028 | 0.14581246508492363 | -0.1342586941189236 | 0 |
| M1-ProductivePulseRecovery | A-M1l-D-CHE-LateOnlyPulse-AdamWRecovery | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.0016614993413289387 | 0.14581246508492363 | -0.14415096574359468 | 0 |
| M2-SplitConsensusSignalSubspace | A-M2a-D-CHE-SplitConsensusDiagMetric | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.003950059413909912 | 0.14581246508492363 | -0.14976252449883354 | 0 |
| M2-SplitConsensusSignalSubspace | A-M2b-D-CHE-SplitConsensusRoleBlockMetric | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.006719297832912869 | 0.14581246508492363 | -0.1525317629178365 | 0 |
| M2-SplitConsensusSignalSubspace | A-M2c-D-CHE-SplitConsensusLowRank-r4 | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.003950059413909912 | 0.14581246508492363 | -0.14976252449883354 | 0 |
| M2-SplitConsensusSignalSubspace | A-M2d-D-CHE-SplitConsensusLowRank-r8 | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.003950059413909912 | 0.14581246508492363 | -0.14976252449883354 | 0 |
| M2-SplitConsensusSignalSubspace | A-M2e-D-CHE-SplitConsensusMetricOnlyNoProjection | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.003950059413909912 | 0.14581246508492363 | -0.14976252449883354 | 0 |
| M2-SplitConsensusSignalSubspace | A-M2f-D-CHE-SplitConsensusProjectionPlusAdamV | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.003950059413909912 | 0.14581246508492363 | -0.14976252449883354 | 0 |
| M2-SplitConsensusSignalSubspace | A-M2g-D-CHE-NegativeEigenQuarantineDiagnostic | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.003950059413909912 | 0.14581246508492363 | -0.14976252449883354 | 0 |
| M3-PopRiskDriftSNR | A-M3a-D-CHE-ParameterSNRPreconditioner | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.003950059413909912 | 0.14581246508492363 | -0.14976252449883354 | 0 |
| M3-PopRiskDriftSNR | A-M3b-D-CHE-DegreeRoleSNRPreconditioner | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.006719297832912869 | 0.14581246508492363 | -0.1525317629178365 | 0 |
| M3-PopRiskDriftSNR | A-M3c-D-CHE-BasisChannelSNRCoverBoundary | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.004168497191535102 | 0.14581246508492363 | -0.14998096227645874 | 0 |
| M3-PopRiskDriftSNR | A-M3d-D-CHE-SNRPulseScheduler | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.0005242184356406883 | 0.14581246508492363 | -0.1463366835205643 | 0 |
| M3-PopRiskDriftSNR | A-M3e-D-CHE-SNRRecoveryScheduler | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.00030302339129977755 | 0.14581246508492363 | -0.14550944169362384 | 0 |
| M3-PopRiskDriftSNR | A-M3f-D-CHE-SNRReservoirGuard | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.0041160980860392255 | 0.14581246508492363 | -0.14992856317096284 | 0 |
| M4-FunctionSpaceProximal | A-M4a-D-CHE-AdamSubspaceProximal-alpha000 | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.21249383687973022 | 0.14581246508492363 | -0.35830630196465385 | 0 |
| M4-FunctionSpaceProximal | A-M4b-D-CHE-PerExampleLowRankProximal-alpha025 | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.14567075173060098 | 0.14581246508492363 | -0.2914832168155246 | 0 |
| M4-FunctionSpaceProximal | A-M4c-D-CHE-OutputJacobianSketchProximal-alpha050 | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.08113973670535618 | 0.14581246508492363 | -0.2269522017902798 | 0 |
| M4-FunctionSpaceProximal | A-M4d-D-CHE-SplitB1B2TransferProximal-alpha100 | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.02800975243250529 | 0.14581246508492363 | -0.1738222175174289 | 0 |
| M4-FunctionSpaceProximal | A-M4e-D-CHE-RecoveryAwareProximal-alpha200 | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.00742564598719279 | 0.14581246508492363 | -0.1532381110721164 | 0 |
| M4-FunctionSpaceProximal | A-M4f-D-CHE-ControlResidualizedProximal-alpha100 | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.08113973670535618 | 0.14581246508492363 | -0.2269522017902798 | 0 |
| M5-OptimizerAwareAlignedFU | A-M5a-D-CHE-CautiousFU | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.005507760577731662 | 0.14581246508492363 | -0.1513202256626553 | 0 |
| M5-OptimizerAwareAlignedFU | A-M5b-D-CHE-SoftCautiousFU | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.0028851760758293998 | 0.14581246508492363 | -0.148697641160753 | 0 |
| M5-OptimizerAwareAlignedFU | A-M5c-D-CHE-MGUPStyleReweightFU | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.0017752382490370008 | 0.14581246508492363 | -0.14403722683588663 | 0 |
| M5-OptimizerAwareAlignedFU | A-M5d-D-CHE-AdamSecondMomentScaledFU | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.044587539301978216 | 0.14581246508492363 | -0.19040000438690186 | 0 |
| M5-OptimizerAwareAlignedFU | A-M5e-D-CHE-SophiaDiagLiteClippedFU | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.053700016604529485 | 0.14581246508492363 | -0.19951248168945312 | 0 |
| M5-OptimizerAwareAlignedFU | A-M5f-D-CHE-BlockSecondMomentFU | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.030444860458374023 | 0.14581246508492363 | -0.17625732554329765 | 0 |
| M5-OptimizerAwareAlignedFU | A-M5g-D-CHE-DecoupledDecayPlusFU | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.006374849213494195 | 0.14581246508492363 | -0.1521873142984178 | 0 |
| M6-CarrierSpecificReparamFU | A-M6a-D-CHE-LowDegreeSignalHighDegreeReservoir | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.006111714575025771 | 0.14581246508492363 | -0.1519241796599494 | 0 |
| M6-CarrierSpecificReparamFU | A-M6b-D-CHE-ReadoutBasisDecoupledCarrier | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.006435599591996934 | 0.14581246508492363 | -0.15224806467692056 | 0 |
| M6-CarrierSpecificReparamFU | A-M6c-D-CHE-OrthogonalDegreeBankCarrier | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.00624284479353163 | 0.14581246508492363 | -0.15205530987845525 | 0 |
| M6-CarrierSpecificReparamFU | A-M6d-D-CHE-OutputJacobianCarrier | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.003950059413909912 | 0.14581246508492363 | -0.14976252449883354 | 0 |
| M6-CarrierSpecificReparamFU | A-M6e-D-CHE-DualBankSignalReservoirCarrier | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.006374504831102159 | 0.14581246508492363 | -0.1521869699160258 | 0 |
| M6-CarrierSpecificReparamFU | A-M6f-D-CHE-HighDegreeQuarantineCarrier | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.00630873441696167 | 0.14581246508492363 | -0.1521211995018853 | 0 |
| M7-LateAttachSnapshotFU | A-M7a-D-CHE-LateAttachEarlyCheckpoint | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.0008363723754882812 | 0.14581246508492363 | -0.1466488374604119 | 0 |
| M7-LateAttachSnapshotFU | A-M7b-D-CHE-LateAttachMidCheckpoint | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.011553770966000028 | 0.14581246508492363 | -0.1342586941189236 | 0 |
| M7-LateAttachSnapshotFU | A-M7c-D-CHE-LateAttachLateCheckpoint | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.0016614993413289387 | 0.14581246508492363 | -0.14415096574359468 | 0 |
| M7-LateAttachSnapshotFU | A-M7d-D-CHE-LateAttachAfterLossPlateau | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.0013408859570821126 | 0.14581246508492363 | -0.1444715791278415 | 0 |
| M7-LateAttachSnapshotFU | A-M7e-D-CHE-LateAttachHighSourceLowDebtWindow | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.05003203286064996 | 0.14581246508492363 | -0.19584449794557357 | 0 |
| M8-RecoveryOnlyDeconfound | A-M8a-D-CHE-AdamWRecoveryOnly | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.005154377884334988 | 0.14581246508492363 | -0.1509668429692586 | 0 |
| M8-RecoveryOnlyDeconfound | A-M8b-D-CHE-LRCooldownOnly | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.08024213049146864 | 0.14581246508492363 | -0.22605459557639226 | 0 |
| M8-RecoveryOnlyDeconfound | A-M8c-D-CHE-DecayRecoveryOnly | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -1.5172741545571222 | 0.14581246508492363 | -1.663086619642046 | 0 |
| M8-RecoveryOnlyDeconfound | A-M8d-D-CHE-EMASWARecoveryOnly | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.0024184385935465493 | 0.14581246508492363 | -0.14339402649137709 | 0 |
| M8-RecoveryOnlyDeconfound | A-M8e-D-CHE-LookaheadRecoveryOnly | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.0006769895553588867 | 0.14581246508492363 | -0.14513547552956474 | 0 |
| M8-RecoveryOnlyDeconfound | A-M8f-D-CHE-MomentumDampingOnly | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.003891203138563368 | 0.14581246508492363 | -0.149703668223487 | 0 |

## 6. Final route / no-go

```text
route = R6-FunctionalDynamicsAllMechanismsNoGo
minimum_success = S1-CarrierMechanismMatrixCoverageCompleted
line_p_efficiency_census_complete = 1
S2_weak_productive_dynamics_reached = 0
S3_productive_debt_recovery_reached = 0
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

No-go boundary：

| boundary | status | evidence |
| --- | --- | --- |
| LineP-EfficiencyCensus | 1 | rows=27;explore_pass=6;official_pass=5 |
| LineA-DCHENoGo | 1 | best=A-M1k-D-CHE-MidOnlyPulse-AdamWRecovery;source_h800=0.011553770966000028;retention=0.3313499175839954;tail=0.0;LineC=0.5555555555555556 |
| LineB-MLPGenericNoPromotion | 1 | best=B-M4a-MLP-AdamSubspaceProximal-alpha000;source_h800=0.14581246508492363;retention=0.8296399083402421;tail=0.01966376412104744;LineC=0.0 |
| LineC-LQReanchor | 1 | best=C4-LQ-MacroDeltaPriorityReanchor;delta=0.3125;near=9/9;route=R-C-LQReanchorStillBlocked |
| LineD-RationalMonitorOnly | 1 | best=D1-RAT-G7AnalogPulseOnce-then-AdamWRecovery;pass=0;no reset/controller |
| LineF-AllBasisSubstrate | 1 | best_family=D-FOU;pass=0/9 |
| R6-FunctionalDynamicsAllMechanismsNoGo | 1 | S2=0;S3=0;S5=0 |

## 7. 科学结论

```text
1. v16.3 已执行 Line P efficiency census，并生成 phase/memory/fallback artifacts。
2. A/B/C/D/F functional matrix 使用 train-stream kernels 和 horizon-target readback；H=800/H=1600 指标均来自 artifact。
3. Line P 解释 runtime/engineering 可承载性，不决定 functional promotion。
4. MLP generic positive、LQ/Rational monitor、recovery-only 与 substrate-only rows 不写成 KAN-specific promotion。
5. 当前 route = R6-FunctionalDynamicsAllMechanismsNoGo，promotion_allowed = 0。
```
