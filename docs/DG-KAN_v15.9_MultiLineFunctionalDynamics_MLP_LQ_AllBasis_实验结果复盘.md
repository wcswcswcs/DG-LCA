# DG-KAN v15.9 MultiLineFunctionalDynamics MLP LQ AllBasis 实验结果复盘

生成时间：2026-05-31（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不把 MLP generic positive、LQ base/reanchor、Rational monitor 或 substrate-only rows 写成 KAN-specific promotion。

## 1. 计划理解

v15.9 的目标是停止 D-CHE 单线等待，改为同时推进 D-CHE dynamics、MLP active dynamics、LQ reanchor、Rational monitor 与 all-basis substrate，并通过 Line F 做 carrier/generic/recovery-only 路由。

## 2. 本轮代码修改

新增：

```text
experiments/run_v159_multiline_functional_dynamics_mlp_lq_allbasis.py
```

修改：

```text
experiments/run_v149_line_d_all_basis_substrate_repair.py
  新增 v15.9 substrate-only candidates:
  D-FOU82..86, D-RBF80..84, D-WAV69..72。
```

过程说明：

```text
1. Line A 使用 D-CHE G7R pulse/recovery dynamics，执行 A0..A13 与 ACTRL controls。
2. Line B 将 MLP functional dynamics 作为 active line 执行 B0..B10 与 controls，不写成 KAN-specific promotion。
3. Line C 执行 C0..C4 LQ reanchor；若 reanchor gate 未开，则 C5..C9 functional fail-closed deferred。
4. Line D 执行 Rational monitor/sanity replay；不重启 reset/controller/action route。
5. Line E 调用 v149 substrate-only runner 读取 v15.9 D-FOU/RBF/WAV rows；不过 gate 不进入 official FU proof。
6. 自查修复 LQ primitive forward 调用签名：显式传入 A 与 basis weights。
7. 自查修复 Rational monitor 参数桥接：loss_interface=CE、linec_mode=exact、output_geometry_repair=none。
8. 初始 official 命令使用 --device cuda:0；用户要求四卡并行后，runner 新增 --run-lines / --line-a-methods / --artifact-suffix / --skip-h400 分片执行。
9. 正式分片执行：cuda:0..3 并行跑 A base shards；B/C/D 分别用 cuda:1/2/3；A base merge 后再跑全局 top2 A400。
10. cpu_offload_used=0；LineC/CEp99/NLL/ECE/AUCtime/Brier 不进入方向。
11. 用户再次追问后反方复核发现 final route taxonomy 与计划第 8/13 节不完全一致；已修正为 A/B/C/D/E 全部 gate 未开时写 R-F5-AllFunctionalDynamicsNoGo，LineE all-basis blocked 保留为 no-go evidence；该修复只重算 route/manifest/docs，不改训练指标。
12. 继续按计划最低合同复核发现 D-CHE no-regression monitor 未单独落 artifact；已从 A0-D-CHE-AdamW 实际 rows 派生 v159_dche_no_regression_monitor.csv 并纳入 manifest/contract/deep coverage。该修复只补覆盖审计，不新增训练、不改指标。
```

<!-- V15.9_MANUAL_ANALYSIS_START -->
## 2.1 人工复核分析 / Insight

这段是人工复核分析，不是 Python 自动生成的指标结论；下方 Line A/B/C/D/E/F/Z 数据仍由 runner 从 artifact 写入。

v15.9 的关键修正是把 MLP 从 passive control 恢复成 active mechanism line，同时把 LQ reanchor、Rational monitor 与 all-basis substrate 纳入同一轮覆盖。当前结果不能只按 D-CHE 单线判断。

反方复核后，我认为 v15.9 没有达成 promotion 目标。Line A 的 D-CHE dynamics 最好是 A12 lookahead，只有很小的 source 正值，同时 bad event 仍然接近全覆盖；进入 h400 后 A12 和 A5 的 source 都转负，说明短程表面 gain 没能变成稳定 functional dynamics。

Line B 的 MLP active dynamics 是本轮最强的表面 source，B9 base source 达到 0.19443613953060573，但 bad_event_fraction=1.0，h400 后 B9 也转负。因此这只能说明 generic MLP dynamics 有可观测局部机制，不能写成 KAN-specific promotion，也不能覆盖 Line B gate 失败。

Line C 的 LQ 历史/当前/repaired reanchor 都没有打开 reanchor gate；C5..C9 functional 按计划 fail-closed deferred。Line D Rational monitor 也没有正向 gate，Line E all-basis substrate 仍然 0/9。也就是说，A/B/C/D/E 不是某个单线漏项，而是全 surface 按计划跑完后都没有过 promotion gate。

用户再次追问后，我复核计划第 8/13 节发现 final route taxonomy 初版把全局失败写成 R-F6-AllBasisCarrierBlocked，容易误读成只是 Line E 阻塞。已修正为 R-F5-AllFunctionalDynamicsNoGo；LineE all-basis blocked 仍保留为 no-go evidence。该修复只重算 route/manifest/docs，不改变任何训练指标。

因此当前最诚实的结论是：v15.9 完成了 multi-line coverage，但未找到 productive functional dynamics。继续在本版里新增 G9/G10、action/controller/reset、或用 audit metric 反推方向都会偏离计划；下一步只能进入新的预注册 theory/substrate-level 计划。
<!-- V15.9_MANUAL_ANALYSIS_END -->

## 2.2 完整计划执行对照（artifact 自动写入）

| contract item | status | details |
|---|---:|---|
| Line R provenance/no-action audit | 1 | audit files present |
| Line A D-CHE dynamics | 1 | A0..A13 + ACTRL controls |
| D-CHE no-regression monitor | 1 | A0-D-CHE-AdamW rows materialized as monitor |
| Line B MLP active dynamics | 1 | B0..B10 + controls |
| Line C LQ reanchor | 1 | C0..C4 reanchor plus functional defer table |
| Line D Rational monitor | 1 | D0..D3 + random control, no reset |
| Line E all-basis | 1 | D-FOU/RBF/WAV v15.9 substrate-only rows |
| Line F cross-line routing | 1 | D-CHE/MLP/LQ/RAT/all-basis comparison |
| Line Z closure | 1 | no-go + next queue + exhaustion |
| Required figures | 1 | figures=12 |
| No forbidden continuation | 1 | no G9/G10/action/controller/reset/audit-directed branch |

深度覆盖审计：

| audit item | status | details |
|---|---:|---|
| Line A exact surface | 1 | A methods x dataset/seed |
| D-CHE no-regression monitor coverage | 1 | A0-D-CHE-AdamW monitor rows present |
| Line B exact surface | 1 | B methods x dataset/seed |
| Line C reanchor coverage | 1 | C0..C4 x dataset/seed |
| Line D rational coverage | 1 | D0..D3 + control |
| Direction provenance train-stream-only | 1 | direction_rows=1224 |
| Budget exhaustion certificate | 1 | planned/executed/deferred rows present |

required / forbidden / no-action / provenance：

```text
required_artifact_manifest_rows = 30
required_artifact_missing_rows = 0
forbidden_information_violation_sum = 0
no_action_search_violation_sum = 0
direction_provenance_rows = 1224
direction_source = train_stream_only / optimizer_state / split_gradient; audit metrics not used for direction
```

## 3. Line A / B dynamics summary

```text
line_a_best_method = A12-D-CHE-G7R-Pulse-LookaheadConsolidation
line_a_gate_pass = 0
line_a_source_vs_best_control_mean = 0.00728044244978163
line_a_bad_event_fraction = 0.8888888888888888
line_b_best_method = B9-MLP-FunctionalPulse-LookaheadConsolidation
line_b_gate_pass = 0
line_b_source_vs_best_control_mean = 0.19443613953060573
line_b_bad_event_fraction = 1.0
```

Line A method summary：

| method | rows | pass | source | bad event | tail recovery | LineC recovery |
|---|---:|---:|---:|---:|---:|---:|
| A1-D-CHE-G7R-PulseOnce-AdamWRecovery | 9 | 0/9 | -0.009109576543172201 | 1.0 | 0.411967175694845 | 0.3333333333333333 |
| A2-D-CHE-G7R-PulseEvery50-AdamWRecovery | 9 | 0/9 | -0.009109576543172201 | 1.0 | 0.411967175694845 | 0.3333333333333333 |
| A3-D-CHE-G7R-EarlyPulseOnly-AdamWRecovery | 9 | 0/9 | -0.002955668502383762 | 0.8888888888888888 | 0.3019419036887385 | 0.2222222222222222 |
| A4-D-CHE-G7R-MidPulseOnly-AdamWRecovery | 9 | 0/9 | 0.0029962129063076442 | 1.0 | 0.19439953957008232 | 0.2222222222222222 |
| A5-D-CHE-G7R-LatePulseOnly-AdamWRecovery | 9 | 0/9 | 0.003614438904656304 | 1.0 | 0.12783177237430454 | 0.2222222222222222 |
| A6-D-CHE-G7R-Pulse-GlobalDecoupledDecayRecovery | 9 | 0/9 | -0.009276005956861708 | 1.0 | 0.41235602177646047 | 0.3333333333333333 |
| A7-D-CHE-G7R-Pulse-DegreeWiseDecayRecovery | 9 | 0/9 | -0.009192817740970187 | 1.0 | 0.41216133576937236 | 0.3333333333333333 |
| A8-D-CHE-G7R-Pulse-HighDegreeExtraDecayRecovery | 9 | 0/9 | -0.009234388669331869 | 1.0 | 0.4122587021118807 | 0.3333333333333333 |
| A9-D-CHE-G7R-Pulse-ReadoutBasisDecoupledDecayRecovery | 9 | 0/9 | -0.009435898727840848 | 1.0 | 0.4126790299213987 | 0.3333333333333333 |
| A10-D-CHE-G7R-Pulse-MomentumEMARecovery | 9 | 0/9 | -0.037468532721201576 | 1.0 | 0.31025570456543095 | 0.1111111111111111 |
| A11-D-CHE-G7R-Pulse-LRCooldownRecovery | 9 | 0/9 | -0.3679084512922499 | 1.0 | 0.4404134368917332 | 0.4444444444444444 |
| A12-D-CHE-G7R-Pulse-LookaheadConsolidation | 9 | 0/9 | 0.00728044244978163 | 0.8888888888888888 | 0.35144129432147225 | 0.4444444444444444 |
| A13-D-CHE-G7R-Pulse-ScheduleFreeLongEMARecovery | 9 | 0/9 | -0.07067770428127712 | 1.0 | 0.27359340830899154 | 0.0 |

Line B method summary：

| method | rows | pass | source | bad event | tail recovery | LineC recovery |
|---|---:|---:|---:|---:|---:|---:|
| B1-MLP-CautiousAdamW | 9 | 0/9 | 0.1727886994679769 | 1.0 | 0.011307779315939534 | 0.0 |
| B2-MLP-MGUP | 9 | 0/9 | -1.2865702973471747 | 1.0 | 0.005013902353734657 | 0.0 |
| B3-MLP-SplitConsensusMetric | 9 | 0/9 | -0.06233644485473633 | 1.0 | 0.019558172063732553 | 0.0 |
| B4-MLP-FunctionalPulseOnce-AdamWRecovery | 9 | 0/9 | -0.06233644485473633 | 1.0 | 0.019558172063732553 | 0.0 |
| B5-MLP-FunctionalPulseEvery50-AdamWRecovery | 9 | 0/9 | -0.06233644485473633 | 1.0 | 0.019558172063732553 | 0.0 |
| B6-MLP-FunctionalPulse-GlobalDecayRecovery | 9 | 0/9 | -0.06073580847846137 | 1.0 | 0.01988482771248279 | 0.0 |
| B7-MLP-FunctionalPulse-MomentumEMARecovery | 9 | 0/9 | -1.2865702973471747 | 1.0 | 0.005013902353734657 | 0.0 |
| B8-MLP-FunctionalPulse-LRCooldownRecovery | 9 | 0/9 | 0.17085079352060953 | 1.0 | 0.010052368890886077 | 0.0 |
| B9-MLP-FunctionalPulse-LookaheadConsolidation | 9 | 0/9 | 0.19443613953060573 | 1.0 | 0.0432604967811506 | 0.0 |
| B10-MLP-AmortizedPersistentFMS-Recheck | 9 | 0/9 | -0.593610061539544 | 1.0 | 0.003250913778028912 | 0.0 |

Line A/B h400 global top2：

| line | method | rows | source | bad event | tail recovery | LineC recovery |
|---|---|---:|---:|---:|---:|---:|
| A | A12-D-CHE-G7R-Pulse-LookaheadConsolidation | 9 | -0.010166439745161269 | 1.0 | 0.35144129432147225 | 0.4444444444444444 |
| A | A5-D-CHE-G7R-LatePulseOnly-AdamWRecovery | 9 | -0.0020043916172451442 | 1.0 | 0.0929027353528567 | 0.4444444444444444 |
| B | B1-MLP-CautiousAdamW | 9 | -0.023681739966074627 | 1.0 | 0.011307779315939534 | 0.0 |
| B | B9-MLP-FunctionalPulse-LookaheadConsolidation | 9 | -0.013595481713612875 | 1.0 | 0.0432604967811506 | 0.0 |

## 4. Line C / D / E results

```text
line_c_rows = 45
line_c_reanchor_gate_pass = 0
line_c_best_method = C0-HistoricalLQReference
line_c_best_macro_delta_vs_MLP = 0.03125
line_d_rat_rows = 45
line_d_rat_best_method = D0-RAT-AdamW
line_d_rat_gate_pass = 0
line_e_rows = 126
line_e_best_family = D-FOU
line_e_best_dataset_seed_pass_count = 0 / 9
```

Line C reanchor rows：

| method | rows? | best delta vs MLP | gate pass rows |
|---|---:|---:|---:|
| C0-HistoricalLQReference | 9 | 0.03125 | 0 |
| C1-CurrentLQReproduction | 9 | 0.03125 | 0 |
| C2-ExactProtocolLQReplay | 9 | 0.03125 | 0 |
| C3-RepairedLQ-FaninOutputScaleConfirmed | 9 | 0.015625 | 0 |
| C4-RepeatedCurrentLQ | 9 | 0.03125 | 0 |

Line D Rational summary：

| method | rows | pass | source | auc median |
|---|---:|---:|---:|---:|
| D0-RAT-AdamW | 9 | 0/9 | -0.0024232864379882812 | 1.0096176219441537 |
| D1-RAT-SplitConsensusMetricPulse-AdamWRecovery | 9 | 0/9 | -0.04729383521609836 | 0.9990510031545555 |
| D2-RAT-SplitConsensusMetricPulse-DecayRecovery | 9 | 0/9 | -0.07412189245223999 | 1.0109978901037862 |
| D3-RAT-SplitConsensusMetricPulse-MomentumRecovery | 9 | 0/9 | -0.037095083130730525 | 1.0255072922104558 |
| DCTRL-RAT-RandomMatchedPulse-SameRecovery | 9 | 0/9 | -0.03902600871192084 | 1.0 |

Line E all-basis summary：

| family | rows | pass | best candidate | max mean delta vs MLP |
|---|---:|---:|---|---:|
| D-FOU | 45 | 0/9 | D-FOU83-BandwiseSNRWarmupV5 | -0.109375 |
| D-RBF | 45 | 0/9 | D-RBF83-GaussianLocalK4TaskHealthV5 | -0.328125 |
| D-WAV | 36 | 0/9 | D-WAV71-SupportOverlapDampingV5 | -0.0234375 |

## 5. Line F / Z route

```text
route = R-F5-AllFunctionalDynamicsNoGo
minimum_success = S1-MultiLineCoverageCompleted
promotion_allowed = 0
line_a_gate_pass = 0
line_b_gate_pass = 0
line_c_reanchor_gate_pass = 0
line_d_rational_gate_pass = 0
line_e_allbasis_gate_pass = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

No-go boundary：

| boundary | status | evidence |
|---|---:|---|
| LineA-DCHENoGo | 1 | best=A12-D-CHE-G7R-Pulse-LookaheadConsolidation;source=0.00728044244978163;bad=0.8888888888888888 |
| LineB-MLPNoGo | 1 | best=B9-MLP-FunctionalPulse-LookaheadConsolidation;source=0.19443613953060573;bad=1.0 |
| LineC-LQReanchor | 1 | best=C0-HistoricalLQReference;delta=0.03125;route=LQBaseNotReanchored |
| LineD-RationalMonitorOnly | 1 | best=D0-RAT-AdamW;pass=0;no reset/controller |
| LineE-AllBasisCarrierBlocked | 1 | best_family=D-FOU;pass=0/9 |
| LineF-AllFunctionalDynamicsNoGo | 1 | A=0;B=0;C=0;D=0;E=0;route=R-F5-AllFunctionalDynamicsNoGo |
| NoForbiddenContinuation | 1 | no G9/G10/action bank/controller/reset/audit-directed branch |

Next hypothesis queue：

| priority | hypothesis | allowed next step |
|---:|---|---|
| 1 | MLPFunctionalDynamicsActiveFollowup | Only if pre-registered from v15.9 artifacts; do not relabel generic MLP positive as KAN-specific. |
| 2 | LQReanchorOrCarrierRepair | Repair LQ base/attach only in a new plan if reanchor failed or functional was deferred. |
| 3 | AllBasisSubstrateBeforeFU | Continue substrate-only acceleration before official FU proof. |

## 6. 科学结论

```text
1. v15.9 已执行 Line R/A/B/C/D/E/F/Z，并生成 required artifacts。
2. MLP functional dynamics 已作为 active line 执行；任何 MLP positive 不写成 KAN-specific promotion。
3. LQ 仅在 reanchor gate 通过后才允许 functional；未通过时只写 LQBaseNotReanchored/deferred。
4. Rational 只是 monitor/sanity replay；没有 reset/controller/action bank。
5. 当前 route = R-F5-AllFunctionalDynamicsNoGo，promotion_allowed = 0。
```

## 7. 详细审计补充（用户质疑后追加）

本节不是新增实验；它从已经落盘的 artifact 重新汇总，目的是把原来过薄的复盘补成可审计版本。所有数值均来自 `results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/official_v159/`。

### 7.1 Artifact inventory

| artifact | exists | rows |
|---|---|---|
| v159_line_a_dche_dynamics.csv | 1 | 162 |
| v159_line_a_h400_long.csv | 1 | 18 |
| v159_line_b_mlp_dynamics.csv | 1 | 126 |
| v159_line_b_h400_long.csv | 1 | 18 |
| v159_line_c_lq_reanchor.csv | 1 | 45 |
| v159_line_c_lq_functional.csv | 1 | 6 |
| v159_line_d_rational_monitor.csv | 1 | 45 |
| v159_line_e_allbasis_results.csv | 1 | 126 |
| v159_failure_taxonomy.csv | 1 | 327 |
| v159_budget_exhaustion_certificate.csv | 1 | 5 |
| v159_required_artifact_manifest.csv | 1 | 30 |
| v159_execution_contract_coverage_audit.csv | 1 | 11 |
| v159_deep_coverage_audit.csv | 1 | 7 |
| v159_dche_no_regression_monitor.csv | 1 | 9 |
| v159_no_go_boundary.csv | 1 | 7 |
| v159_line_f_crossline_route.csv | 1 | 4 |

### 7.2 Route / gate final state

```text
route = R-F5-AllFunctionalDynamicsNoGo
minimum_success = S1-MultiLineCoverageCompleted
promotion_allowed = 0
official_s5_reached = 0
line_a_gate_pass = 0
line_b_gate_pass = 0
line_c_reanchor_gate_pass = 0
line_d_rational_gate_pass = 0
line_e_allbasis_gate_pass = 0
all_functional_dynamics_no_go = 1
allbasis_carrier_blocked = 1
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
source_vs_best_control_mean = 0.19443613953060573
```

### 7.3 Line A D-CHE full surface including controls

| method | rows | strict | real-lite | source | bad | control-equiv | tail-rec | LineC-rec | AUCtime |
|---|---|---|---|---|---|---|---|---|---|
| A0-D-CHE-AdamW | 9 | 0 | 0 | 0.0 | 1.0 | 1.0 | 0.3111327676657694 | 0.2222222222222222 | 1.0 |
| A1-D-CHE-G7R-PulseOnce-AdamWRecovery | 9 | 0 | 0 | -0.009109576543172201 | 1.0 | 1.0 | 0.411967175694845 | 0.3333333333333333 | 0.998865251351694 |
| A10-D-CHE-G7R-Pulse-MomentumEMARecovery | 9 | 0 | 0 | -0.037468532721201576 | 1.0 | 0.6666666666666666 | 0.31025570456543095 | 0.1111111111111111 | 1.020688144505905 |
| A11-D-CHE-G7R-Pulse-LRCooldownRecovery | 9 | 0 | 0 | -0.3679084512922499 | 1.0 | 1.0 | 0.4404134368917332 | 0.4444444444444444 | 1.2191178299033498 |
| A12-D-CHE-G7R-Pulse-LookaheadConsolidation | 9 | 2 | 0 | 0.00728044244978163 | 0.8888888888888888 | 0.4444444444444444 | 0.35144129432147225 | 0.4444444444444444 | 0.9736167742245979 |
| A13-D-CHE-G7R-Pulse-ScheduleFreeLongEMARecovery | 9 | 0 | 0 | -0.07067770428127712 | 1.0 | 0.7777777777777778 | 0.27359340830899154 | 0.0 | 1.0344938780906456 |
| A2-D-CHE-G7R-PulseEvery50-AdamWRecovery | 9 | 0 | 0 | -0.009109576543172201 | 1.0 | 1.0 | 0.411967175694845 | 0.3333333333333333 | 0.998865251351694 |
| A3-D-CHE-G7R-EarlyPulseOnly-AdamWRecovery | 9 | 0 | 0 | -0.002955668502383762 | 0.8888888888888888 | 0.8888888888888888 | 0.3019419036887385 | 0.2222222222222222 | 0.9945282222990319 |
| A4-D-CHE-G7R-MidPulseOnly-AdamWRecovery | 9 | 0 | 0 | 0.0029962129063076442 | 1.0 | 1.0 | 0.19439953957008232 | 0.2222222222222222 | 0.988677533377537 |
| A5-D-CHE-G7R-LatePulseOnly-AdamWRecovery | 9 | 0 | 0 | 0.003614438904656304 | 1.0 | 1.0 | 0.12783177237430454 | 0.2222222222222222 | 0.9880031883905329 |
| A6-D-CHE-G7R-Pulse-GlobalDecoupledDecayRecovery | 9 | 0 | 0 | -0.009276005956861708 | 1.0 | 1.0 | 0.41235602177646047 | 0.3333333333333333 | 0.998938436950493 |
| A7-D-CHE-G7R-Pulse-DegreeWiseDecayRecovery | 9 | 0 | 0 | -0.009192817740970187 | 1.0 | 1.0 | 0.41216133576937236 | 0.3333333333333333 | 0.9989018390240562 |
| A8-D-CHE-G7R-Pulse-HighDegreeExtraDecayRecovery | 9 | 0 | 0 | -0.009234388669331869 | 1.0 | 1.0 | 0.4122587021118807 | 0.3333333333333333 | 0.9989201377454795 |
| A9-D-CHE-G7R-Pulse-ReadoutBasisDecoupledDecayRecovery | 9 | 0 | 0 | -0.009435898727840848 | 1.0 | 1.0 | 0.4126790299213987 | 0.3333333333333333 | 0.9990138625976274 |
| ACTRL1-D-CHE-RandomMatchedPulse-SameAdamWRecovery | 9 | 0 | 0 | -0.0053556495242648655 | 1.0 | 1.0 | 0.32955518978704457 | 0.3333333333333333 | 1.0089474156434524 |
| ACTRL2-D-CHE-RandomMatchedPulse-SameDecayRecovery | 9 | 0 | 0 | -0.005034552680121528 | 1.0 | 1.0 | 0.3423242005584385 | 0.3333333333333333 | 1.008954500176263 |
| ACTRL3-D-CHE-NoOpMatchedOverhead | 9 | 0 | 0 | -1.3650072481897142 | 1.0 | 1.0 | 1.0 | 0.5555555555555556 | 1.6319296207692424 |
| ACTRL4-D-CHE-AdamWExtraStepsMatchedTime | 9 | 0 | 0 | 0.0 | 1.0 | 1.0 | 0.3111327676657694 | 0.2222222222222222 | 1.0 |

Line A h400 top-2：

| method | rows | source | bad | control-equiv | tail-rec | LineC-rec |
|---|---|---|---|---|---|---|
| A12-D-CHE-G7R-Pulse-LookaheadConsolidation | 9 | -0.010166439745161269 | 1.0 | 1.0 | 0.35144129432147225 | 0.4444444444444444 |
| A5-D-CHE-G7R-LatePulseOnly-AdamWRecovery | 9 | -0.0020043916172451442 | 1.0 | 1.0 | 0.0929027353528567 | 0.4444444444444444 |

### 7.4 Line B MLP active full surface including controls

| method | rows | strict | real-lite | source | bad | control-equiv | tail-rec | LineC-rec | AUCtime |
|---|---|---|---|---|---|---|---|---|---|
| B0-MLP-AdamW | 9 | 0 | 0 | -0.04674856530295478 | 1.0 | 1.0 | 0.015588674131671646 | 0.0 | 1.0080759947794788 |
| B1-MLP-CautiousAdamW | 9 | 0 | 0 | 0.1727886994679769 | 1.0 | 0.0 | 0.011307779315939534 | 0.0 | 0.8575057981608143 |
| B10-MLP-AmortizedPersistentFMS-Recheck | 9 | 0 | 0 | -0.593610061539544 | 1.0 | 1.0 | 0.003250913778028912 | 0.0 | 1.2655665941340086 |
| B2-MLP-MGUP | 9 | 0 | 0 | -1.2865702973471747 | 1.0 | 1.0 | 0.005013902353734657 | 0.0 | 1.498636967192503 |
| B3-MLP-SplitConsensusMetric | 9 | 0 | 0 | -0.06233644485473633 | 1.0 | 0.7777777777777778 | 0.019558172063732553 | 0.0 | 1.0199626296840725 |
| B4-MLP-FunctionalPulseOnce-AdamWRecovery | 9 | 0 | 0 | -0.06233644485473633 | 1.0 | 0.7777777777777778 | 0.019558172063732553 | 0.0 | 1.0199626296840725 |
| B5-MLP-FunctionalPulseEvery50-AdamWRecovery | 9 | 0 | 0 | -0.06233644485473633 | 1.0 | 0.7777777777777778 | 0.019558172063732553 | 0.0 | 1.0199626296840725 |
| B6-MLP-FunctionalPulse-GlobalDecayRecovery | 9 | 0 | 0 | -0.06073580847846137 | 1.0 | 0.7777777777777778 | 0.01988482771248279 | 0.0 | 1.0193851547065436 |
| B7-MLP-FunctionalPulse-MomentumEMARecovery | 9 | 0 | 0 | -1.2865702973471747 | 1.0 | 1.0 | 0.005013902353734657 | 0.0 | 1.498636967192503 |
| B8-MLP-FunctionalPulse-LRCooldownRecovery | 9 | 0 | 0 | 0.17085079352060953 | 1.0 | 0.0 | 0.010052368890886077 | 0.0 | 0.8920850510742798 |
| B9-MLP-FunctionalPulse-LookaheadConsolidation | 9 | 0 | 0 | 0.19443613953060573 | 1.0 | 0.0 | 0.0432604967811506 | 0.0 | 0.8454083781183424 |
| BCTRL1-MLP-RandomMatchedPulse-SameRecovery | 9 | 0 | 0 | -0.07159572839736938 | 1.0 | 1.0 | 0.0214247481000739 | 0.0 | 1.032432852109011 |
| BCTRL2-MLP-NoOpMatchedOverhead | 9 | 0 | 0 | -1.0657751030392117 | 1.0 | 1.0 | 1.0 | 0.0 | 2.252171625637907 |
| BCTRL3-MLP-AdamWExtraStepsMatchedTime | 9 | 0 | 0 | -0.04674856530295478 | 1.0 | 1.0 | 0.015588674131671646 | 0.0 | 1.0080759947794788 |

Line B h400 top-2：

| method | rows | source | bad | control-equiv | tail-rec | LineC-rec |
|---|---|---|---|---|---|---|
| B1-MLP-CautiousAdamW | 9 | -0.023681739966074627 | 1.0 | 1.0 | 0.011307779315939534 | 0.0 |
| B9-MLP-FunctionalPulse-LookaheadConsolidation | 9 | -0.013595481713612875 | 1.0 | 1.0 | 0.0432604967811506 | 0.0 |

### 7.5 Line C LQ reanchor / functional defer

| method | rows | best macro delta vs MLP | gate pass rows | route |
|---|---|---|---|---|
| C0-HistoricalLQReference | 9 | 0.03125 | 0 |  |
| C1-CurrentLQReproduction | 9 | 0.03125 | 0 |  |
| C2-ExactProtocolLQReplay | 9 | 0.03125 | 0 |  |
| C3-RepairedLQ-FaninOutputScaleConfirmed | 9 | 0.015625 | 0 |  |
| C4-RepeatedCurrentLQ | 9 | 0.03125 | 0 |  |

Functional defer rows：

| method | status | reason | promotion_allowed |
|---|---|---|---|
| C5-LQ-G7Analog-PulseOnce-AdamWRecovery | None | None | 0 |
| C6-LQ-G7Analog-Pulse-DecayRecovery | None | None | 0 |
| C7-LQ-G7Analog-Pulse-MomentumRecovery | None | None | 0 |
| C8-LQ-SplitConsensusMetric | None | None | 0 |
| C9-LQ-SnapshotLateAttachFunctionalPulse | None | None | 0 |
| CCTRL-RandomMatchedPulse-SameRecovery | None | None | 0 |

### 7.6 Line D Rational monitor

| method | rows | strict | real-lite | source | bad | control-equiv | tail-rec | LineC-rec | AUCtime |
|---|---|---|---|---|---|---|---|---|---|
| D0-RAT-AdamW | 9 | 0 | 0 | -0.0024232864379882812 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| D1-RAT-SplitConsensusMetricPulse-AdamWRecovery | 9 | 0 | 0 | -0.04729383521609836 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| D2-RAT-SplitConsensusMetricPulse-DecayRecovery | 9 | 0 | 0 | -0.07412189245223999 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| D3-RAT-SplitConsensusMetricPulse-MomentumRecovery | 9 | 0 | 0 | -0.037095083130730525 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| DCTRL-RAT-RandomMatchedPulse-SameRecovery | 9 | 0 | 0 | -0.03902600871192084 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |

### 7.7 Line E all-basis substrate full detail

Family summary：

| family | rows | pass rows | best mean delta vs MLP | best candidate |
|---|---|---|---|---|
| D-FOU | 45 | 0 | -0.109375 | None |
| D-RBF | 45 | 0 | -0.328125 | None |
| D-WAV | 36 | 0 | -0.0234375 | None |

Candidate summary：

| candidate | family | rows | pass rows | mean delta vs MLP | step ratio | memory ratio |
|---|---|---|---|---|---|---|
|  | D-FOU | 126 | 0 | -0.3919890873015873 | 0.0 | 0.0 |

### 7.8 Failure taxonomy

| failure_class | rows |
|---|---|
| source,tail_debt,linec_debt,control_equivalent | 171 |
| source,linec_debt,control_equivalent | 53 |
| tail_debt,linec_debt | 40 |
| source,tail_debt,control_equivalent | 28 |
| source,control_equivalent | 26 |
| linec_debt | 4 |
| tail_debt | 2 |
| LQBaseNotReanchored | 1 |
| RationalMonitorNoPromotion | 1 |
| AllBasisCarrierBlocked | 1 |

| line | rows |
|---|---|
| A | 162 |
| B | 126 |
| A400 | 18 |
| B400 | 18 |
| C | 1 |
| D | 1 |
| E | 1 |

### 7.9 Budget exhaustion certificate

| line_name | planned_rows | executed_rows | planned_horizons | executed_horizons | why_deferred | does_defer_affect_route |
|---|---|---|---|---|---|---|
| Line A D-CHE | 162 | 162 | 1/5/20/50/100 plus top2 h400 and positive h800 | 1/5/20/50/100 plus top2 h400; no h800 because h400 gate failed | h800 deferred by pre-registered h400 gate fail | 0 |
| Line B MLP | 126 | 126 | h100 all plus top2 h400 | h100 all plus top2 h400 |  | 0 |
| Line C LQ | 45 | 45 | reanchor then functional if pass | reanchor; functional deferred | LQBaseNotReanchored | 0 |
| Line D Rational | 45 | 45 | monitor/sanity replay | monitor/sanity replay | no reset/controller route by plan | 0 |
| Line E All-basis | 126 | 126 | substrate-only + D-CHE no-regression monitor | substrate-only + A0 monitor materialized | official FU proof gated by substrate only | 0 |

### 7.10 No-go boundary

| boundary | status | evidence |
|---|---|---|
| LineA-DCHENoGo | 1 | best=A12-D-CHE-G7R-Pulse-LookaheadConsolidation;source=0.00728044244978163;bad=0.8888888888888888 |
| LineB-MLPNoGo | 1 | best=B9-MLP-FunctionalPulse-LookaheadConsolidation;source=0.19443613953060573;bad=1.0 |
| LineC-LQReanchor | 1 | best=C0-HistoricalLQReference;delta=0.03125;route=LQBaseNotReanchored |
| LineD-RationalMonitorOnly | 1 | best=D0-RAT-AdamW;pass=0;no reset/controller |
| LineE-AllBasisCarrierBlocked | 1 | best_family=D-FOU;pass=0/9 |
| LineF-AllFunctionalDynamicsNoGo | 1 | A=0;B=0;C=0;D=0;E=0;route=R-F5-AllFunctionalDynamicsNoGo |
| NoForbiddenContinuation | 1 | no G9/G10/action bank/controller/reset/audit-directed branch |

### 7.11 Contract coverage

| contract_item | status | details |
|---|---|---|
| Line R provenance/no-action audit | 1 | audit files present |
| Line A D-CHE dynamics | 1 | A0..A13 + ACTRL controls |
| D-CHE no-regression monitor | 1 | A0-D-CHE-AdamW rows materialized as monitor |
| Line B MLP active dynamics | 1 | B0..B10 + controls |
| Line C LQ reanchor | 1 | C0..C4 reanchor plus functional defer table |
| Line D Rational monitor | 1 | D0..D3 + random control, no reset |
| Line E all-basis | 1 | D-FOU/RBF/WAV v15.9 substrate-only rows |
| Line F cross-line routing | 1 | D-CHE/MLP/LQ/RAT/all-basis comparison |
| Line Z closure | 1 | no-go + next queue + exhaustion |
| Required figures | 1 | figures=12 |
| No forbidden continuation | 1 | no G9/G10/action/controller/reset/audit-directed branch |

### 7.12 Deep coverage

| audit_item | status | details |
|---|---|---|
| Line A exact surface | 1 | A methods x dataset/seed |
| D-CHE no-regression monitor coverage | 1 | A0-D-CHE-AdamW monitor rows present |
| Line B exact surface | 1 | B methods x dataset/seed |
| Line C reanchor coverage | 1 | C0..C4 x dataset/seed |
| Line D rational coverage | 1 | D0..D3 + control |
| Direction provenance train-stream-only | 1 | direction_rows=1224 |
| Budget exhaustion certificate | 1 | planned/executed/deferred rows present |

### 7.13 D-CHE no-regression monitor

该 artifact 从 Line A 的 `A0-D-CHE-AdamW` 实际 rows 派生，不新增训练。

| rows | datasets | seeds | bad fraction | mean AUCtime | source artifact |
|---|---|---|---|---|---|
| 9 | Fashion-MNIST,KMNIST,MNIST | 0,1,2 | 1.0 | 1.0 | v159_line_a_dche_dynamics.csv |

### 7.14 审计判断

```text
1. 本轮不是没跑，而是原复盘展示太瘦；实际 artifacts 覆盖 A/B/C/D/E/F/Z。
2. A/B 的 h400 top-2 均未保留 source，不能升级 h800 或 promotion。
3. Line C reanchor gate 未开，C5..C9 functional 按计划 fail-closed deferred，不影响其它线完成。
4. MLP active line 有最强 base source，但 bad_event=1.0 且 h400 转负，只能作为 generic dynamics 线索。
5. 所有 gate 关闭且 contract/deep coverage 完整，因此 final route 为 R-F5-AllFunctionalDynamicsNoGo。
```

