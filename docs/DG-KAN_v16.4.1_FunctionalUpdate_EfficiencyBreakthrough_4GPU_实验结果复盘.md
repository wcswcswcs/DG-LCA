# DG-KAN v16.4.1 FunctionalUpdate EfficiencyBreakthrough 4GPU 实验结果复盘

生成时间：2026-06-02 03:52:48（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；效率 timing 只作为 engineering/basis evidence，不作为 functional promotion。

## 1. 计划理解

v16.4.1 的核心不是新增一个 FU 名字，而是同时裁决 functional dynamics 与 basis efficiency：D-CHE/MLP full matrix 作为真实 artifact readback，LQ/Rational/D-FOU/D-RBF/D-WAV 执行 minimum smoke，所有 basis 执行 same-param MLP efficiency repair truth table，并生成 4GPU queue/drain/idle artifacts。

## 2. 本轮代码修改

新增：
```text
experiments/run_v1641_functional_update_efficiency_breakthrough_4gpu.py
```

复用：
```text
experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py  # CUDA phase profiler / same-param MLP
results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163  # real functional matrix readback
```

过程说明：
```text
1. 新执行 v1641 efficiency repair truth table：P0/P8 + CHE/FOU/LQ/RBF/WAV/RAT repair attempts x batch 8/32/128。
2. 新执行 non-main carrier low-budget functional smoke；smoke_only=1，不进入 official promotion。
3. D-CHE/MLP M1a..M8f full matrix 不伪造重跑，读取 v16.3 真实 artifact 并写 reuse_readback=1/source_artifact。
4. LineC/tail/AUC/calibration 只读回，不作为方向源。
5. Rational 不启动 reset/controller/action route。
6. 生成 runnable queue、GPU assignment、utilization、idle violation、queue drain artifacts。
7. blocker 修复方向以 actual candidate/kernel timing 呈现；未通过的 row 写 exact blocker class，不改写为成功。
8. runtime blocker 修复：第一次 finalize 在 route 写出前生成 queue，导致 FINALIZE 自引用 artifact 被误标 missing；已改为 route materialize 后重建 queue/drain/idle artifacts。
```

## 2.1 人工复核分析 / Insight

Efficiency truth table rows=114，measured=114，exploration pass=6，official pass=5。
Functional smoke rows=48，complete=1，source>=0.005 rows=0；这些 rows 是 low-budget smoke，不是 official FU proof。
全局 best h800 source 来自 `MLP` / `B-M4a-MLP-AdamSubspaceProximal-alpha000`，source=0.14581246508492363，retention=0.8296399083402421，tail=0.01966376412104744，LineC=0.0，AUC=1.0373429473020996。
h1600 对 best method 的 mean source=0.0；因此 h800 source 没有变成可保留的 productive dynamics。
Line M KAN_specific_advantage_rows=0，不能把 D-CHE/KAN 局部 row 写成 KAN-specific promotion。
最终 route=R4-FunctionalSourceNotRetained，promotion_allowed=0。该 route 来自 required/efficiency/queue/source-retention/debt/control/KAN attribution 的优先级判断。

## 2.2 Required / forbidden / queue readback

```text
required_artifact_manifest_rows = 51
required_artifact_missing_rows = 0
forbidden_information_violation_sum = 0
no_action_search_violation_sum = 0
idle_violation_count = 0
direction_provenance_rows = 5994
direction_source = train_stream_only / optimizer_state / split_gradient; audit metrics not used for direction
```

## 3. Efficiency truth table

| subject | batch | fwd | bwd | upd | mem | step | explore | official |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RBF-E5-center-update-separation | 8 | 8.49954335387324 | 2.0122190924051364 | 0.13605439677091832 | 0.9901776655225266 | 0.2692950080171954 | 0 | 0 |
| FOU-E3-fused-band-k-small | 128 | 5.497130307798544 | 1.6892051486382818 | 1.2011378576072562 | 1.0387612040714456 | 0.6417273409144597 | 0 | 0 |
| P0-MLP-same-param-reference | 8 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1 | 1 |
| P0-MLP-same-param-reference | 32 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1 | 1 |
| P0-MLP-same-param-reference | 128 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1 | 1 |
| FOU-E2-sincos-precompute-shape | 32 | 5.027974632011746 | 1.504414040354679 | 0.6145440303048688 | 1.0006976639510285 | 1.0057604973708187 | 0 | 0 |
| P8-MLP-functional-path-top2 | 32 | 1.8732249118177657 | 1.5569483561381077 | 1.40659680497632 | 0.997042020705855 | 1.0567120266529602 | 0 | 0 |
| P8-MLP-functional-path-top2 | 128 | 1.1693949236028245 | 0.785233039897827 | 0.886469252752229 | 0.9970436496434989 | 1.1506684810612862 | 1 | 1 |
| LQ-E3-fused-projection-update | 128 | 1.0611447854817362 | 1.025864579907787 | 0.9669069550018949 | 0.9886956764451391 | 1.1665052444166804 | 1 | 1 |
| LQ-E4-fixed-frame-late-attach | 32 | 3.0571646147988325 | 1.2826704310385673 | 1.156498966385587 | 0.981505728314239 | 1.1808436066876686 | 0 | 0 |
| LQ-E4-fixed-frame-late-attach | 128 | 2.6042649001982996 | 1.3533243647156434 | 1.0659559739543352 | 0.9830969814094015 | 1.1853981903928232 | 0 | 0 |
| FOU-E1-low-frequency-recurrence | 8 | 11.306305159538358 | 3.0146772561662867 | 0.6974131650373546 | 1.0021185963172485 | 1.2051585973731553 | 0 | 0 |
| P8-MLP-functional-path-top2 | 8 | 677.2777889551227 | 18.379418808577537 | 55.8227991978826 | 0.9970415917396601 | 1.226384453564937 | 0 | 0 |
| RAT-E2-branchless-denominator-readback | 128 | 4.178792952694023 | 1.1604912556835885 | 0.8747319209872384 | 1.0442876965772432 | 1.2335480211203607 | 0 | 0 |
| FOU-E0-current | 8 | 10.068197358291036 | 2.588798780579179 | 0.8782294004326566 | 1.0020545549921316 | 1.2340737356906728 | 0 | 0 |
| FOU-E1-low-frequency-recurrence | 128 | 4.798855001728755 | 1.2811827884040985 | 0.7979856418484578 | 1.0380811742068285 | 1.2393722217503924 | 0 | 0 |
| LQ-E0-current | 32 | 2.805184287732029 | 1.213859195244989 | 0.6863445096710438 | 0.9876998769987699 | 1.2401475559642139 | 0 | 0 |
| FOU-E2-sincos-precompute-shape | 8 | 11.262975564354422 | 3.7714176656181904 | 0.7371516164312506 | 1.0021185963172485 | 1.2683822728715972 | 0 | 0 |
| FOU-E6-trig-vs-recurrence-table | 8 | 4.241807169489098 | 1.1470490321865636 | 0.5915151826209902 | 1.0021185963172485 | 1.2721583183583056 | 0 | 0 |
| LQ-E4-fixed-frame-late-attach | 8 | 3.151236522703469 | 1.1845991581446909 | 1.9031963968706964 | 0.9814956331877729 | 1.2738601361918531 | 0 | 0 |
| RBF-E2-width-condition-guard | 32 | 7.699735256342489 | 2.0966071229417853 | 0.6518081417464924 | 0.9981668001179991 | 1.2855240725503383 | 0 | 0 |
| RAT-E4-optimizer-state-cost | 128 | 3.779943928828072 | 1.0462024877792229 | 0.8341086930882948 | 1.0442876965772432 | 1.2926579914736862 | 0 | 0 |
| LQ-E3-fused-projection-update | 32 | 2.8618028206200083 | 1.0493052017589197 | 0.48923843832504543 | 0.9876998769987699 | 1.2981028494951976 | 0 | 0 |
| LQ-E3-fused-projection-update | 8 | 50.90092965764161 | 14.276933889900732 | 0.11979454185778354 | 0.987695061245036 | 1.3134388850507255 | 0 | 0 |
| LQ-E2-persistent-adamw-foreach | 128 | 2.6002785207860395 | 1.1686053250106765 | 0.23191005403921408 | 0.9886956764451391 | 1.3188268608861633 | 0 | 0 |
| LQ-E1-update-decomposition | 128 | 2.1885430852530843 | 1.1345391577666863 | 0.5879247582676526 | 0.9886956764451391 | 1.3206114938680558 | 0 | 0 |
| LQ-E2-persistent-adamw-foreach | 32 | 3.0236071491551093 | 1.5040702813938027 | 0.8464773617913468 | 0.9876998769987699 | 1.3239555606509323 | 0 | 0 |
| RAT-E2-branchless-denominator-readback | 32 | 3.7648955839986242 | 1.08733545498735 | 1.143581191756023 | 1.0080400266072822 | 1.3274151526766063 | 0 | 0 |
| LQ-E0-current | 128 | 1.2495251038156137 | 1.0447474061562612 | 0.8608804420191906 | 0.9886956764451391 | 1.328383469595668 | 1 | 0 |
| RAT-E4-optimizer-state-cost | 8 | 15.229771909320274 | 1.2021587831262563 | 0.8037159470234169 | 0.9993636652879414 | 1.3363993077765917 | 0 | 0 |

Efficiency blocker taxonomy（前 80）：

| subject | batch | attempt | blocker | dominant | reason |
| --- | --- | --- | --- | --- | --- |
| CHE-E0-D-CHE-current | 8 | current | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F9-EfficiencyUpdateBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;optimizer_update_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| CHE-E0-D-CHE-current | 32 | current | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| CHE-E0-D-CHE-current | 128 | current | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| CHE-E1-audit-separated-readback | 8 | CHE-E1 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F9-EfficiencyUpdateBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;optimizer_update_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| CHE-E1-audit-separated-readback | 32 | CHE-E1 | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| CHE-E1-audit-separated-readback | 128 | CHE-E1 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| CHE-E2-degree-recurrence-cache | 8 | CHE-E2 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| CHE-E2-degree-recurrence-cache | 32 | CHE-E2 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| CHE-E2-degree-recurrence-cache | 128 | CHE-E2 | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| CHE-E3-low-degree-active-bank | 8 | CHE-E3 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| CHE-E3-low-degree-active-bank | 32 | CHE-E3 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| CHE-E3-low-degree-active-bank | 128 | CHE-E3 | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| CHE-E4-high-degree-late-enable | 8 | CHE-E4 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| CHE-E4-high-degree-late-enable | 32 | CHE-E4 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| CHE-E4-high-degree-late-enable | 128 | CHE-E4 | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| CHE-E5-role-degree-projection | 8 | CHE-E5 | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| CHE-E5-role-degree-projection | 32 | CHE-E5 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| CHE-E5-role-degree-projection | 128 | CHE-E5 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| CHE-E6-no-materialize-energy-readback | 8 | CHE-E6 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| CHE-E6-no-materialize-energy-readback | 32 | CHE-E6 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| CHE-E6-no-materialize-energy-readback | 128 | CHE-E6 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| FOU-E0-current | 8 | current | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | backward | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75 |
| FOU-E0-current | 32 | current | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75 |
| FOU-E0-current | 128 | current | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75 |
| FOU-E1-low-frequency-recurrence | 8 | FOU-E1 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | backward | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75 |
| FOU-E1-low-frequency-recurrence | 32 | FOU-E1 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75 |
| FOU-E1-low-frequency-recurrence | 128 | FOU-E1 | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75 |
| FOU-E2-sincos-precompute-shape | 8 | FOU-E2 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | backward | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75 |
| FOU-E2-sincos-precompute-shape | 32 | FOU-E2 | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75 |
| FOU-E2-sincos-precompute-shape | 128 | FOU-E2 | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75 |
| FOU-E3-fused-band-k-small | 8 | FOU-E3 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | backward | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75 |
| FOU-E3-fused-band-k-small | 32 | FOU-E3 | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75 |
| FOU-E3-fused-band-k-small | 128 | FOU-E3 | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75 |
| FOU-E4-phase-stable-low-band | 8 | FOU-E4 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | backward | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| FOU-E4-phase-stable-low-band | 32 | FOU-E4 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | backward | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| FOU-E4-phase-stable-low-band | 128 | FOU-E4 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | backward | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| FOU-E5-high-frequency-quarantine | 8 | FOU-E5 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | backward | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| FOU-E5-high-frequency-quarantine | 32 | FOU-E5 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | backward | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| FOU-E5-high-frequency-quarantine | 128 | FOU-E5 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F10-MemoryBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;backward_memory_ratio_vs_same_param_mlp>1.5;step_ratio_vs_same_param_mlp>1.75 |
| FOU-E6-trig-vs-recurrence-table | 8 | FOU-E6 | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75 |
| FOU-E6-trig-vs-recurrence-table | 32 | FOU-E6 | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75 |
| FOU-E6-trig-vs-recurrence-table | 128 | FOU-E6 | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75 |
| RAT-E0-current | 8 | current | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | forward | forward_ratio_vs_same_param_mlp>1.75 |
| RAT-E0-current | 32 | current | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75 |
| RAT-E0-current | 128 | current | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75 |
| RAT-E1-denominator-derivative-telemetry | 8 | RAT-E1 | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | forward | forward_ratio_vs_same_param_mlp>1.75 |
| RAT-E1-denominator-derivative-telemetry | 32 | RAT-E1 | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75 |
| RAT-E1-denominator-derivative-telemetry | 128 | RAT-E1 | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75 |
| RAT-E2-branchless-denominator-readback | 8 | RAT-E2 | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | forward | forward_ratio_vs_same_param_mlp>1.75 |
| RAT-E2-branchless-denominator-readback | 32 | RAT-E2 | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75 |
| RAT-E2-branchless-denominator-readback | 128 | RAT-E2 | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75 |
| RAT-E3-rational-eval-timing | 8 | RAT-E3 | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | forward | forward_ratio_vs_same_param_mlp>1.75 |
| RAT-E3-rational-eval-timing | 32 | RAT-E3 | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75 |
| RAT-E3-rational-eval-timing | 128 | RAT-E3 | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75 |
| RAT-E4-optimizer-state-cost | 8 | RAT-E4 | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75 |
| RAT-E4-optimizer-state-cost | 32 | RAT-E4 | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75 |
| RAT-E4-optimizer-state-cost | 128 | RAT-E4 | F7-EfficiencyForwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75 |
| RBF-E0-current | 8 | current | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | forward | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75 |
| RBF-E0-current | 32 | current | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| RBF-E0-current | 128 | current | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | backward | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| RBF-E1-active-center-occupancy | 8 | RBF-E1 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | forward | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| RBF-E1-active-center-occupancy | 32 | RBF-E1 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | backward | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| RBF-E1-active-center-occupancy | 128 | RBF-E1 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | backward | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| RBF-E2-width-condition-guard | 8 | RBF-E2 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | forward | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| RBF-E2-width-condition-guard | 32 | RBF-E2 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75 |
| RBF-E2-width-condition-guard | 128 | RBF-E2 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | backward | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75 |
| RBF-E3-local-k4-no-dense | 8 | RBF-E3 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | forward | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75 |
| RBF-E3-local-k4-no-dense | 32 | RBF-E3 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75 |
| RBF-E3-local-k4-no-dense | 128 | RBF-E3 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | backward | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| RBF-E4-gaussian-table-lookup-smoke | 8 | RBF-E4 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | backward | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75 |
| RBF-E4-gaussian-table-lookup-smoke | 32 | RBF-E4 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75 |
| RBF-E4-gaussian-table-lookup-smoke | 128 | RBF-E4 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | backward | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| RBF-E5-center-update-separation | 8 | RBF-E5 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | backward | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75 |
| RBF-E5-center-update-separation | 32 | RBF-E5 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | backward | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| RBF-E5-center-update-separation | 128 | RBF-E5 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | backward | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| WAV-E0-current | 8 | current | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | backward | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75 |
| WAV-E0-current | 32 | current | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | backward | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| WAV-E0-current | 128 | current | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | backward | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| WAV-E1-triangular-index-forward | 8 | WAV-E1 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | backward | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |
| WAV-E1-triangular-index-forward | 32 | WAV-E1 | F7-EfficiencyForwardBlocked;F8-EfficiencyBackwardBlocked;F11-AuditCostPolluted | audit_readback | forward_ratio_vs_same_param_mlp>1.75;backward_ratio_vs_same_param_mlp>1.75;step_ratio_vs_same_param_mlp>1.75 |

## 4. Functional smoke

| carrier | mechanism | candidate | seed | source h20 | ctrl loss | fu loss | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D-FOU | M1-ProductivePulseRecovery | D-FOU16-FrequencyBandDampingSubstrate | 0 | -0.03219771385192871 | 1.8914560079574585 | 1.9236537218093872 | measured |
| D-FOU | M1-ProductivePulseRecovery | D-FOU16-FrequencyBandDampingSubstrate | 1 | -0.032029151916503906 | 1.8990808725357056 | 1.9311100244522095 | measured |
| D-FOU | M1-ProductivePulseRecovery | D-FOU16-FrequencyBandDampingSubstrate | 2 | -0.0317840576171875 | 1.9022388458251953 | 1.9340229034423828 | measured |
| D-FOU | M2-SplitConsensusSignalSubspace | D-FOU13-FusedReadoutGradNoMaterialize-K2 | 0 | -0.03230893611907959 | 1.8905225992202759 | 1.9228315353393555 | measured |
| D-FOU | M2-SplitConsensusSignalSubspace | D-FOU13-FusedReadoutGradNoMaterialize-K2 | 1 | -0.03157973289489746 | 1.9063307046890259 | 1.9379104375839233 | measured |
| D-FOU | M2-SplitConsensusSignalSubspace | D-FOU13-FusedReadoutGradNoMaterialize-K2 | 2 | -0.032079100608825684 | 1.8966137170791626 | 1.9286928176879883 | measured |
| D-FOU | M3-PopRiskDriftSNR | D-FOU17-PhaseStabilityCorrectionSubstrate | 0 | -0.041914939880371094 | 1.551538109779358 | 1.593453049659729 | measured |
| D-FOU | M3-PopRiskDriftSNR | D-FOU17-PhaseStabilityCorrectionSubstrate | 1 | -0.0430828332901001 | 1.5385456085205078 | 1.581628441810608 | measured |
| D-FOU | M3-PopRiskDriftSNR | D-FOU17-PhaseStabilityCorrectionSubstrate | 2 | -0.0427628755569458 | 1.5410094261169434 | 1.5837723016738892 | measured |
| D-FOU | M4-FunctionSpaceProximal | D-FOU19-HighFreqNoiseLeakVetoSubstrate | 0 | -0.024656176567077637 | 1.7625679969787598 | 1.7872241735458374 | measured |
| D-FOU | M4-FunctionSpaceProximal | D-FOU19-HighFreqNoiseLeakVetoSubstrate | 1 | -0.025602340698242188 | 1.737460732460022 | 1.7630630731582642 | measured |
| D-FOU | M4-FunctionSpaceProximal | D-FOU19-HighFreqNoiseLeakVetoSubstrate | 2 | -0.025987625122070312 | 1.7187238931655884 | 1.7447115182876587 | measured |
| D-RAT | M1-ProductivePulseRecovery | D-RAT28-GroupDiversityPreservingRational | 0 | -2.3593477749273006e-07 | 4.2219929241582577e-07 | 6.581340699085558e-07 | measured |
| D-RAT | M1-ProductivePulseRecovery | D-RAT28-GroupDiversityPreservingRational | 1 | -2.69462304913759e-07 | 4.97946814448369e-07 | 7.67409119362128e-07 | measured |
| D-RAT | M1-ProductivePulseRecovery | D-RAT28-GroupDiversityPreservingRational | 2 | -2.1109960357534874e-07 | 2.930560469849297e-07 | 5.041556505602784e-07 | measured |
| D-RAT | M2-SplitConsensusSignalSubspace | D-RAT34-TrainingDenDerivativeStabilityNoCE | 0 | -1.937147828812158e-07 | 3.5638552731143136e-07 | 5.501003101926472e-07 | measured |
| D-RAT | M2-SplitConsensusSignalSubspace | D-RAT34-TrainingDenDerivativeStabilityNoCE | 1 | -2.818797497639025e-07 | 5.637600679619936e-07 | 8.456398177258961e-07 | measured |
| D-RAT | M2-SplitConsensusSignalSubspace | D-RAT34-TrainingDenDerivativeStabilityNoCE | 2 | -1.8998963469130103e-07 | 4.731114131573122e-07 | 6.631010478486132e-07 | measured |
| D-RAT | M3-PopRiskDriftSNR | D-RAT36-TangentConditionStabilizerNoCE | 0 | -3.5265958331365255e-07 | 7.996943054422445e-07 | 1.152353888755897e-06 | measured |
| D-RAT | M3-PopRiskDriftSNR | D-RAT36-TangentConditionStabilizerNoCE | 1 | -1.7012146713568654e-07 | 2.272426513627579e-07 | 3.9736411849844444e-07 | measured |
| D-RAT | M3-PopRiskDriftSNR | D-RAT36-TangentConditionStabilizerNoCE | 2 | -2.2848414005238737e-07 | 3.712869727223733e-07 | 5.997711127747607e-07 | measured |
| D-RBF | M1-ProductivePulseRecovery | D-RBF12-CenterOccupancyRebalanceSubstrate | 0 | -0.02965688705444336 | 1.7958987951278687 | 1.825555682182312 | measured |
| D-RBF | M1-ProductivePulseRecovery | D-RBF12-CenterOccupancyRebalanceSubstrate | 1 | -0.02398085594177246 | 1.8245760202407837 | 1.8485568761825562 | measured |
| D-RBF | M1-ProductivePulseRecovery | D-RBF12-CenterOccupancyRebalanceSubstrate | 2 | -0.02363300323486328 | 1.8126074075698853 | 1.8362404108047485 | measured |
| D-RBF | M3-PopRiskDriftSNR | D-RBF13-WidthConditionGuardSubstrate | 0 | -0.01646566390991211 | 2.054713487625122 | 2.071179151535034 | measured |
| D-RBF | M3-PopRiskDriftSNR | D-RBF13-WidthConditionGuardSubstrate | 1 | -0.01833939552307129 | 2.051443099975586 | 2.0697824954986572 | measured |
| D-RBF | M3-PopRiskDriftSNR | D-RBF13-WidthConditionGuardSubstrate | 2 | -0.01406550407409668 | 2.0623385906219482 | 2.076404094696045 | measured |
| D-RBF | M4-FunctionSpaceProximal | D-RBF17-CompactCapacityK4HealthSubstrate | 0 | -0.027321934700012207 | 1.8111242055892944 | 1.8384461402893066 | measured |
| D-RBF | M4-FunctionSpaceProximal | D-RBF17-CompactCapacityK4HealthSubstrate | 1 | -0.028308749198913574 | 1.8100897073745728 | 1.8383984565734863 | measured |
| D-RBF | M4-FunctionSpaceProximal | D-RBF17-CompactCapacityK4HealthSubstrate | 2 | -0.02545487880706787 | 1.813565731048584 | 1.8390206098556519 | measured |
| D-WAV | M1-ProductivePulseRecovery | D-WAV12-LocalSupportOccupancyRepairSubstrate | 0 | -0.018980741500854492 | 2.0494258403778076 | 2.068406581878662 | measured |
| D-WAV | M1-ProductivePulseRecovery | D-WAV12-LocalSupportOccupancyRepairSubstrate | 1 | -0.019300460815429688 | 2.0381920337677 | 2.05749249458313 | measured |
| D-WAV | M1-ProductivePulseRecovery | D-WAV12-LocalSupportOccupancyRepairSubstrate | 2 | -0.02021002769470215 | 2.0388100147247314 | 2.0590200424194336 | measured |
| D-WAV | M3-PopRiskDriftSNR | D-WAV15-ScaleDiversityTransportSubstrate | 0 | -0.03034055233001709 | 1.777107834815979 | 1.807448387145996 | measured |
| D-WAV | M3-PopRiskDriftSNR | D-WAV15-ScaleDiversityTransportSubstrate | 1 | -0.030290722846984863 | 1.77912175655365 | 1.8094124794006348 | measured |
| D-WAV | M3-PopRiskDriftSNR | D-WAV15-ScaleDiversityTransportSubstrate | 2 | -0.031081795692443848 | 1.7805408239364624 | 1.8116226196289062 | measured |
| LQ | M1-ProductivePulseRecovery | C2-LQ-ProtocolMatchedReanchor | 0 | -3.788876347243786e-05 | 0.00034273500205017626 | 0.0003806237655226141 | measured |
| LQ | M1-ProductivePulseRecovery | C2-LQ-ProtocolMatchedReanchor | 1 | -4.575616912916303e-05 | 0.0004059084167238325 | 0.0004516645858529955 | measured |
| LQ | M1-ProductivePulseRecovery | C2-LQ-ProtocolMatchedReanchor | 2 | -4.050190909765661e-05 | 0.0003539695462677628 | 0.0003944714553654194 | measured |
| LQ | M2-SplitConsensusSignalSubspace | C4-LQ-MacroDeltaPriorityReanchor | 0 | -1.263388548977673e-05 | 0.00012410925410222262 | 0.00013674313959199935 | measured |
| LQ | M2-SplitConsensusSignalSubspace | C4-LQ-MacroDeltaPriorityReanchor | 1 | -1.5927362255752087e-05 | 0.00018241307407151908 | 0.00019834043632727116 | measured |
| LQ | M2-SplitConsensusSignalSubspace | C4-LQ-MacroDeltaPriorityReanchor | 2 | -1.1478070518933237e-05 | 0.00011300283949822187 | 0.0001244809100171551 | measured |
| LQ | M4-FunctionSpaceProximal | C5-LQ-StepMemoryRecheck | 0 | -4.510569851845503e-05 | 0.0003719816158991307 | 0.00041708731441758573 | measured |
| LQ | M4-FunctionSpaceProximal | C5-LQ-StepMemoryRecheck | 1 | -5.449869786389172e-05 | 0.00044607408926822245 | 0.0005005727871321142 | measured |
| LQ | M4-FunctionSpaceProximal | C5-LQ-StepMemoryRecheck | 2 | -5.017445073463023e-05 | 0.0004370578972157091 | 0.0004872323479503393 | measured |
| LQ | M7-LateAttachSnapshotFU | C6-LQ-LineCNoRegressionCheck | 0 | -2.8346461476758122e-05 | 0.00027678042533807456 | 0.0003051268868148327 | measured |
| LQ | M7-LateAttachSnapshotFU | C6-LQ-LineCNoRegressionCheck | 1 | -1.6946563846431673e-05 | 0.0002207912621088326 | 0.00023773782595526427 | measured |
| LQ | M7-LateAttachSnapshotFU | C6-LQ-LineCNoRegressionCheck | 2 | -1.681676076259464e-05 | 0.00020234036492183805 | 0.00021915712568443269 | measured |

## 5. Carrier × mechanism h800 readback

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

h1600 method means：

| method | source | bad | tail | LineC |
| --- | --- | --- | --- | --- |
| A-M1k-D-CHE-MidOnlyPulse-AdamWRecovery | 0.0 | 0.8888888888888888 | 0.35597136078213754 | 0.4444444444444444 |
| A-M7b-D-CHE-LateAttachMidCheckpoint | 0.0 | 0.8888888888888888 | 0.35597136078213754 | 0.4444444444444444 |
| B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.0 | 1.0 | 0.14282379020659178 | 0.0 |
| B-M4b-MLP-PerExampleLowRankProximal-alpha025 | -0.04176027907265557 | 1.0 | 0.12097317708890282 | 0.0 |

## 6. Line M / no-go boundary

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

No-go boundary：

| boundary | status | evidence |
| --- | --- | --- |
| EfficiencyTruthTable | 1 | rows=114;measured=114;explore_pass=6;official_pass=5 |
| FunctionalSmoke | 1 | rows=48;measured=48 |
| D-CHE/MLPMatrixReadback | 1 | v16.3 real artifacts copied with reuse_readback=1; no fresh v16.4.1 training claimed |
| KANSpecificAttribution | 1 | line_m_rows=55;KAN_specific_advantage_rows=0 |
| BestH800Debt | 1 | best=B-M4a-MLP-AdamSubspaceProximal-alpha000;carrier=MLP;source=0.14581246508492363;retention=0.8296399083402421;tail=0.01966376412104744;LineC=0.0;h1600_mean_source=0.0 |
| QueueDrain | 1 | idle_violation_count=0 |
| R4-FunctionalSourceNotRetained | 1 | final priority route after required/efficiency/queue/source/debt/control gates |

## 7. Final route / conclusion

```text
route = R4-FunctionalSourceNotRetained
minimum_success = S1-ExecutionCoverageCompleted
line_p_efficiency_truth_table_complete = 1
functional_smoke_complete = 1
S2_weak_productive_dynamics_reached = 0
S3_productive_debt_recovery_reached = 0
official_s5_reached = 0
promotion_allowed = 0
```

科学结论：
```text
1. v16.4.1 完成 efficiency truth table、phase/memory decomposition、basis repair timing、functional smoke 与 queue/drain artifacts。
2. D-CHE/MLP full matrix 使用 v16.3 真实 artifact readback；本复盘明确标记 reuse_readback，未编造 fresh training。
3. Efficiency 仍显示多个 basis 被 forward/backward/update/audit cost 阻断；这些是 engineering blocker，不是 promotion 证据。
4. Functional source 没有形成 h1600 retained debt recovery；KAN_specific_advantage_rows=0。
5. 当前 route=R4-FunctionalSourceNotRetained，promotion_allowed=0；下一步必须基于本轮 blocker taxonomy 做新预注册理论/实现重构。
```
