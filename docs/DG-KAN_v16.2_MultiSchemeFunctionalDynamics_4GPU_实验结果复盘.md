# DG-KAN v16.2 MultiSchemeFunctionalDynamics 4GPU 实验结果复盘

生成时间：2026-06-01（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不把短程 source、MLP generic positive、LQ/Rational monitor、recovery-only 或 substrate-only rows 写成 KAN-specific promotion。

## 1. 计划理解

v16.2 的目标是把 functional dynamics 从单线候选扩展为 carrier × mechanism × horizon × controls matrix，并强制四卡分片执行。核心判断不是局部 gain，而是 h800/h1600 source retention、tail/LineC/calibration/AUC debt recovery、matched controls 与 KAN-vs-MLP attribution 是否同时成立。

## 2. 本轮代码修改

新增：

```text
experiments/run_v162_multischeme_functional_dynamics_4gpu.py
```

修改：

```text
experiments/run_v149_line_d_all_basis_substrate_repair.py
  新增 v16.2 substrate-only candidates:
  D-FOU97..102, D-RBF95..100, D-WAV81..85。
```

过程说明：

```text
1. D-CHE 与 MLP 都执行 M1a..M8f multi-scheme mechanism family，并保留 matched controls。
2. A/B full surface 执行到 H=800；top-2 source-retaining candidates 执行 H=1600 extension。
3. LQ 先执行 C0..C6 reanchor；reanchor 未开时 C7..C10 functional fail-closed deferred。
4. Rational 只做 monitor/sanity replay；不启动 reset/controller/action route。
5. D-FOU/RBF/WAV 只做 substrate-only repair gate；不进入 official FU proof。
6. Line G/DGS、LineC/tail/AUC/calibration 只作为 readback/audit/gate，不生成方向。
7. 四卡分片执行并写入 v162_gpu_assignment_manifest.csv；cpu_offload_used=0。
8. method_surface_manifest 记录每个 v16.2 方法映射到的内部实现与参数 overrides，防止黑箱解读。
9. runtime blocker 修复：exact per-step eval full surface 过慢，改为 horizon-target readback；仍训练到 H=800/H=1600，指标只来自实际 horizon artifact。
10. runtime blocker 修复：exact per-step full trace 未在本轮冒充覆盖；正式执行为 horizon-target readback + M1a..M8f surface + matched controls，budget certificate 明确写入 deferred_items。
11. runtime blocker 修复：初始 batch=32/hidden=32 official shard 超过 75 分钟未落任何 artifact；已停止该未落盘尝试，不写入指标，正式重跑改用 batch=8/hidden=16 以完成同一 H=800/H=1600 全 surface。
12. runtime blocker 修复：batch=8/hidden=16 重跑 A shard 约 45 分钟仍无中途 artifact；runner 新增 row-level checkpoint/resume，之后每个 dataset/seed/method 完成即落盘，partial artifact 不作为 final coverage，最终仍由 merge/finalize manifest 判定。
13. runtime blocker 修复：B1600 首次与 MERGE_B 并行启动导致 pre-merge race，产出 header-only h1600 文件；已移为 .premerge_race_empty.csv 备份并串行重跑 B1600，最终 h1600 artifact 有 18 条真实数据 rows。
14. runtime insight：MLP shard 比 D-CHE shard 快，主要来自 D-CHE basis/functional carrier 前后向与 D-CHE-only LineC horizon readback 更重；该观察只作为执行分析，不进入 scientific gate。
```

<!-- V16.2_MANUAL_ANALYSIS_START -->
## 2.1 人工复核分析 / Insight

这段是人工复核分析，不是 Python 自动生成的指标结论；下方 Line G/A/B/C/D/F/M/Z 数据仍由 runner 从 artifact 写入。

v16.2 的关键问题不是“某条线有没有局部 gain”，而是 carrier × mechanism matrix 里是否存在能跨过 h800/h1600 debt gates 的机制。本轮把 D-CHE 与 MLP 都按 M1a..M8f 展开，并保留 LQ/Rational/all-basis 的 fail-closed 约束；所以 route 不是单线结论，而是 matrix 结论。

全局 best h800 source 来自 `D-CHE` / `A-M4a-D-CHE-AdamSubspaceProximal-alpha000`，source=0.12557220458984375，retention=0.7634300026628706，tail=0.0，LineC=0.2222222222222222，AUC=1.0299587646966346。这个 row 的意义必须同时看 debt：source 不能单独作为 promotion。

D-CHE best 是 `A-M4a-D-CHE-AdamSubspaceProximal-alpha000`，source=0.12557220458984375，tail=0.0，LineC=0.2222222222222222；MLP best 是 `B-M4a-MLP-AdamSubspaceProximal-alpha000`，source=-0.19552964634365505，tail=0.0，LineC=0.0。二者对比决定“KAN-specific 还是 generic dynamics”。

S2 weak rows=0，S3 productive rows=0。如果 S2 有但 S3 没有，本轮只能说明存在弱动态线索，不能 promotion；如果 S2/S3 都没有，则说明当前机制 family 没有形成可偿还 debt。

Line M attribution 中 KAN-specific advantage rows=0 / 55。这条证据链用于防止把 MLP generic positive 错写成 KAN-specific promotion。

Line C LQ reanchor gate=0，best=C2-LQ-ProtocolMatchedReanchor，near=8/9。未开 gate 时 C functional fail-closed 是计划约束，不是漏跑。

Line F all-basis gate=0，best_family=D-FOU，pass=4/9。substrate-only rows 只能作为下一版 carrier repair 证据，不能直接进入 official FU proof。

最终 route=R6-FunctionalDynamicsAllMechanismsNoGo，promotion_allowed=0。这个 route 是 A/B/C/D/F gates、forbidden/no-action audits 与 required manifest 同时闭合后的结果。

运行观察：MLP shard 明显快于 D-CHE shard，主要不是单纯前馈差异，而是 D-CHE 的 basis/functional carrier 前向图更重、split-gradient 反向更贵，并且 D-CHE horizon readback 额外执行 LineC pack；MLP dense carrier 没有同等 LineC readback。因此该差异是 runtime/engineering insight，不进入 scientific gate。

最强 h800 row `A-M4a-D-CHE-AdamSubspaceProximal-alpha000` 没有打开 S2 的直接原因是 retention=0.7634300026628706 低于 0.40 且 tail recovery=0.0；虽然 source=0.12557220458984375、LineC=0.2222222222222222、AUC=1.0299587646966346 看起来有局部信号，但它没有形成 source-retaining debt recovery。

h1600 extension 关闭了“再等更久会巩固”的解释。A-line top2 聚合为 A-M4a-D-CHE-AdamSubspaceProximal-alpha000: source=0.0, bad=1.0, tail=0.017574313511256316, LineC=0.4444444444444444; A-M4b-D-CHE-PerExampleLowRankProximal-alpha025: source=-0.02831966347164578, bad=1.0, tail=0.03057429645997782, LineC=0.4444444444444444；B-line top2 聚合为 B-M4a-MLP-AdamSubspaceProximal-alpha000: source=-0.002453327178955078, bad=1.0, tail=0.0, LineC=0.0; B-M4b-MLP-PerExampleLowRankProximal-alpha025: source=-0.12256171968248156, bad=1.0, tail=0.0, LineC=0.0。也就是说，h800 的 best source 没有在 h1600 变成稳定 promotion 证据。

运行层面有两个明确修复：第一，exact per-step full trace 运行时间不可接受，改为 horizon-target readback；第二，A/B 分片训练时间长且原实现只在 shard 结束落盘，已改为每完成一个 dataset/seed/method row 就 checkpoint 并支持 resume。A/B 仍执行计划内 M1a..M8f surface 与 matched controls，budget/deferred/code-review logs 明确记录 runtime 修复，不改变任何已训练指标。

<!-- V16.2_MANUAL_ANALYSIS_END -->

## 2.2 完整计划执行对照（artifact 自动写入）

| contract item | status | details |
| --- | --- | --- |
| Line R provenance/no-action audit | 1 | audit files present |
| 4GPU execution manifest | 1 | cuda:0..3 shard plan and row accounting present |
| Method surface | 1 | carrier x mechanism method mapping present |
| Line G dynamic geometry | 1 | horizon debt/DGS readback present |
| Line A D-CHE mechanism matrix | 1 | D-CHE M1a..M8f plus controls |
| Line B MLP active mechanism matrix | 1 | MLP M1a..M8f plus controls |
| Line C LQ reanchor | 1 | LQ C0..C6 and fail-closed functional table |
| Line D Rational monitor | 1 | Rational monitor rows present |
| Line F all-basis substrate | 1 | D-FOU/RBF/WAV substrate rows present |
| Line M crossline attribution | 1 | KAN-vs-MLP attribution present |
| Line Z closure | 1 | no-go and next queue present |
| Required figures | 1 | figures=14 |
| No forbidden continuation | 1 | no G9/G10/action/controller/reset |

深度覆盖审计：

| audit item | status | details |
| --- | --- | --- |
| Line A exact surface | 1 | A methods x dataset/seed |
| Line A h800 coverage | 1 | H=800 rows present for A methods |
| Line B exact surface | 1 | B methods x dataset/seed |
| Line B h800 coverage | 1 | H=800 rows present for B methods |
| Line C reanchor coverage | 1 | C0..C6 x dataset/seed |
| Line D rational coverage | 1 | D0..D3 + control |
| Direction provenance train-stream-only | 1 | direction_rows=5994 |
| Budget exhaustion certificate | 1 | mandatory/fallback/deferred rows present |
| Code review packet | 1 | implementation decisions and risks recorded |

required / forbidden / no-action / provenance：

```text
required_artifact_manifest_rows = 54
required_artifact_missing_rows = 0
forbidden_information_violation_sum = 0
no_action_search_violation_sum = 0
direction_provenance_rows = 5994
direction_source = train_stream_only / optimizer_state / split_gradient; audit metrics not used for direction
```

## 3. GPU / method surface

| round | gpu | run lines | suffix | rows | artifact exists |
| --- | --- | --- | --- | --- | --- |
| 1 | cuda:0 | A | a0 | 144 | 1 |
| 1 | cuda:1 | A | a1 | 144 | 1 |
| 1 | cuda:2 | A | a2 | 144 | 1 |
| 1 | cuda:3 | A | a3 | 135 | 1 |
| 2 | cuda:0 | B | b0 | 144 | 1 |
| 2 | cuda:1 | B | b1 | 144 | 1 |
| 2 | cuda:2 | B | b2 | 144 | 1 |
| 2 | cuda:3 | B,C,D | b3 | 135 | 1 |
| 3 | cuda:3 | F |  | 153 | 1 |
| 4 | cuda:0 | MERGE_A,MERGE_B,A1600 |  | 18 | 1 |
| 4 | cuda:1 | B1600 |  | 18 | 1 |
| 5 | cuda:0 | FINALIZE |  |  | 1 |

Method surface manifest summary：

| carrier | method | mechanism | internal | overrides |
| --- | --- | --- | --- | --- |
| D-CHE | A-M1a-D-CHE-PulseOnce-AdamWRecovery | M1-ProductivePulseRecovery | T1-G7R-PulseOnce-then-AdamWRecovery | {} |
| D-CHE | A-M1b-D-CHE-PulseOnce-LRCooldownRecovery | M1-ProductivePulseRecovery | T1-G7R-PulseOnce-then-AdamWRecovery|LR-Cooldown | {"lr": 0.0015} |
| D-CHE | A-M1c-D-CHE-PulseOnce-MomentumDampingRecovery | M1-ProductivePulseRecovery | T1-G7R-PulseOnce-then-AdamWRecovery|MomentumDamp | {"beta1": 0.7} |
| D-CHE | A-M1d-D-CHE-PulseOnce-DecoupledGlobalDecayRecovery | M1-ProductivePulseRecovery | W1-G7R-GlobalDecoupledWeightDecayRecovery | {} |
| D-CHE | A-M1e-D-CHE-PulseOnce-DegreeWiseDecayRecovery | M1-ProductivePulseRecovery | W2-G7R-DegreeWiseDecoupledDecay | {} |
| D-CHE | A-M1f-D-CHE-PulseOnce-EMAConsolidation | M1-ProductivePulseRecovery | T1-G7R-PulseOnce-then-AdamWRecovery|EMA | {"beta1": 0.5} |
| D-CHE | A-M1g-D-CHE-PulseOnce-LookaheadConsolidation | M1-ProductivePulseRecovery | T1-G7R-PulseOnce-then-AdamWRecovery|Lookahead | {"beta1": 0.45} |
| D-CHE | A-M1h-D-CHE-PulseOnce-SWAConsolidation | M1-ProductivePulseRecovery | T1-G7R-PulseOnce-then-AdamWRecovery|SWA | {"beta1": 0.3} |
| D-CHE | A-M1i-D-CHE-PulseEvery50-AdamWRecovery | M1-ProductivePulseRecovery | T2-G7R-PulseEvery50-then-AdamWRecovery | {} |
| D-CHE | A-M1j-D-CHE-EarlyOnlyPulse-AdamWRecovery | M1-ProductivePulseRecovery | T3-G7R-PulseEarlyOnly-then-AdamWRecovery | {} |
| D-CHE | A-M1k-D-CHE-MidOnlyPulse-AdamWRecovery | M1-ProductivePulseRecovery | T4-G7R-PulseMidOnly-then-AdamWRecovery | {} |
| D-CHE | A-M1l-D-CHE-LateOnlyPulse-AdamWRecovery | M1-ProductivePulseRecovery | T5-G7R-PulseLateOnly-then-AdamWRecovery | {} |
| D-CHE | A-M2a-D-CHE-SplitConsensusDiagMetric | M2-SplitConsensusSignalSubspace | T1-G7R-PulseOnce-then-AdamWRecovery|SC-Diag | {} |
| D-CHE | A-M2b-D-CHE-SplitConsensusRoleBlockMetric | M2-SplitConsensusSignalSubspace | T1-G7R-PulseOnce-then-AdamWRecovery|SC-RoleBlock | {} |
| D-CHE | A-M2c-D-CHE-SplitConsensusLowRank-r4 | M2-SplitConsensusSignalSubspace | T1-G7R-PulseOnce-then-AdamWRecovery|SC-LowRank-r4 | {} |
| D-CHE | A-M2d-D-CHE-SplitConsensusLowRank-r8 | M2-SplitConsensusSignalSubspace | T1-G7R-PulseOnce-then-AdamWRecovery|SC-LowRank-r8 | {} |
| D-CHE | A-M2e-D-CHE-SplitConsensusMetricOnlyNoProjection | M2-SplitConsensusSignalSubspace | T1-G7R-PulseOnce-then-AdamWRecovery|SC-MetricOnly | {} |
| D-CHE | A-M2f-D-CHE-SplitConsensusProjectionPlusAdamV | M2-SplitConsensusSignalSubspace | T1-G7R-PulseOnce-then-AdamWRecovery|SC-ProjectionPlusAdamV | {} |
| D-CHE | A-M2g-D-CHE-NegativeEigenQuarantineDiagnostic | M2-SplitConsensusSignalSubspace | T1-G7R-PulseOnce-then-AdamWRecovery|SC-NegativeEigenQuarantine | {"lambda_noise": 0.1} |
| D-CHE | A-M3a-D-CHE-ParameterSNRPreconditioner | M3-PopRiskDriftSNR | T1-G7R-PulseOnce-then-AdamWRecovery|SNR-Parameter | {"lambda_noise": 0.1} |
| D-CHE | A-M3b-D-CHE-DegreeRoleSNRPreconditioner | M3-PopRiskDriftSNR | T1-G7R-PulseOnce-then-AdamWRecovery|SNR-DegreeRole | {"lambda_noise": 0.15} |
| D-CHE | A-M3c-D-CHE-BasisChannelSNRCoverBoundary | M3-PopRiskDriftSNR | W5-G7R-SignalReservoirDualDecay|SNR-BasisChannel | {"lambda_noise": 0.2} |
| D-CHE | A-M3d-D-CHE-SNRPulseScheduler | M3-PopRiskDriftSNR | T2-G7R-PulseEvery50-then-AdamWRecovery|SNR-PulseScheduler | {"lambda_noise": 0.1} |
| D-CHE | A-M3e-D-CHE-SNRRecoveryScheduler | M3-PopRiskDriftSNR | T1-G7R-PulseOnce-then-AdamWRecovery|SNR-RecoveryScheduler | {"beta1": 0.6, "lambda_noise": 0.1} |
| D-CHE | A-M3f-D-CHE-SNRReservoirGuard | M3-PopRiskDriftSNR | W3-G7R-HighDegreeExtraDecay|SNR-ReservoirGuard | {"lambda_noise": 0.1} |
| D-CHE | A-M4a-D-CHE-AdamSubspaceProximal-alpha000 | M4-FunctionSpaceProximal | T1-G7R-PulseOnce-then-AdamWRecovery|PX-AdamSubspace-alpha000 | {"lr": 0.001, "proximal_alpha": 0.0} |
| D-CHE | A-M4b-D-CHE-PerExampleLowRankProximal-alpha025 | M4-FunctionSpaceProximal | T1-G7R-PulseOnce-then-AdamWRecovery|PX-LowRank-r4-alpha025 | {"lr": 0.0012, "proximal_alpha": 0.025} |
| D-CHE | A-M4c-D-CHE-OutputJacobianSketchProximal-alpha050 | M4-FunctionSpaceProximal | T1-G7R-PulseOnce-then-AdamWRecovery|PX-OutputJacobian-LowRank-r8-alpha050 | {"lr": 0.0015, "proximal_alpha": 0.05} |
| D-CHE | A-M4d-D-CHE-SplitB1B2TransferProximal-alpha100 | M4-FunctionSpaceProximal | T1-G7R-PulseOnce-then-AdamWRecovery|PX-SplitB1B2-alpha100 | {"lr": 0.002, "proximal_alpha": 0.1} |
| D-CHE | A-M4e-D-CHE-RecoveryAwareProximal-alpha200 | M4-FunctionSpaceProximal | T1-G7R-PulseOnce-then-AdamWRecovery|PX-RecoveryAware-alpha200 | {"beta1": 0.6, "lr": 0.0025, "proximal_alpha": 0.2} |
| D-CHE | A-M4f-D-CHE-ControlResidualizedProximal-alpha100 | M4-FunctionSpaceProximal | T1-G7R-PulseOnce-then-AdamWRecovery|PX-ControlResidualized-alpha100 | {"lr": 0.0015, "proximal_alpha": 0.1} |
| D-CHE | A-M5a-D-CHE-CautiousFU | M5-OptimizerAwareAlignedFU | T1-G7R-PulseOnce-then-AdamWRecovery|OA-Cautious | {"beta1": 0.8} |
| D-CHE | A-M5b-D-CHE-SoftCautiousFU | M5-OptimizerAwareAlignedFU | T1-G7R-PulseOnce-then-AdamWRecovery|OA-SoftCautious | {"beta1": 0.75} |
| D-CHE | A-M5c-D-CHE-MGUPStyleReweightFU | M5-OptimizerAwareAlignedFU | T1-G7R-PulseOnce-then-AdamWRecovery|OA-MGUP | {"beta1": 0.65} |
| D-CHE | A-M5d-D-CHE-AdamSecondMomentScaledFU | M5-OptimizerAwareAlignedFU | T1-G7R-PulseOnce-then-AdamWRecovery|OA-SecondMomentScaled | {"beta2": 0.95} |
| D-CHE | A-M5e-D-CHE-SophiaDiagLiteClippedFU | M5-OptimizerAwareAlignedFU | T1-G7R-PulseOnce-then-AdamWRecovery|OA-SophiaDiagLite | {"beta2": 0.9} |
| D-CHE | A-M5f-D-CHE-BlockSecondMomentFU | M5-OptimizerAwareAlignedFU | T1-G7R-PulseOnce-then-AdamWRecovery|OA-BlockSecondMoment | {"beta1": 0.7, "beta2": 0.95} |
| D-CHE | A-M5g-D-CHE-DecoupledDecayPlusFU | M5-OptimizerAwareAlignedFU | W1-G7R-GlobalDecoupledWeightDecayRecovery|OA-DecoupledDecay | {} |
| D-CHE | A-M6a-D-CHE-LowDegreeSignalHighDegreeReservoir | M6-CarrierSpecificReparamFU | T1-G7R-PulseOnce-then-AdamWRecovery|CR-LowDegreeSignal | {} |
| D-CHE | A-M6b-D-CHE-ReadoutBasisDecoupledCarrier | M6-CarrierSpecificReparamFU | W4-G7R-ReadoutBasisDecoupledDecay|CR-ReadoutBasis | {} |
| D-CHE | A-M6c-D-CHE-OrthogonalDegreeBankCarrier | M6-CarrierSpecificReparamFU | W2-G7R-DegreeWiseDecoupledDecay|CR-OrthogonalDegreeBank | {} |
| D-CHE | A-M6d-D-CHE-OutputJacobianCarrier | M6-CarrierSpecificReparamFU | T1-G7R-PulseOnce-then-AdamWRecovery|CR-OutputJacobian-LowRank-r8 | {} |
| D-CHE | A-M6e-D-CHE-DualBankSignalReservoirCarrier | M6-CarrierSpecificReparamFU | W5-G7R-SignalReservoirDualDecay|CR-DualBank | {} |
| D-CHE | A-M6f-D-CHE-HighDegreeQuarantineCarrier | M6-CarrierSpecificReparamFU | W3-G7R-HighDegreeExtraDecay|CR-HighDegreeQuarantine | {} |
| D-CHE | A-M7a-D-CHE-LateAttachEarlyCheckpoint | M7-LateAttachSnapshotFU | T3-G7R-PulseEarlyOnly-then-AdamWRecovery|LateAttachEarly | {} |
| D-CHE | A-M7b-D-CHE-LateAttachMidCheckpoint | M7-LateAttachSnapshotFU | T4-G7R-PulseMidOnly-then-AdamWRecovery|LateAttachMid | {} |
| D-CHE | A-M7c-D-CHE-LateAttachLateCheckpoint | M7-LateAttachSnapshotFU | T5-G7R-PulseLateOnly-then-AdamWRecovery|LateAttachLate | {} |
| D-CHE | A-M7d-D-CHE-LateAttachAfterLossPlateau | M7-LateAttachSnapshotFU | T5-G7R-PulseLateOnly-then-AdamWRecovery|LateAttachPlateau | {"beta1": 0.6} |
| D-CHE | A-M7e-D-CHE-LateAttachHighSourceLowDebtWindow | M7-LateAttachSnapshotFU | T5-G7R-PulseLateOnly-then-AdamWRecovery|LateAttachSourceDebt | {"lr": 0.0015} |
| D-CHE | A-M8a-D-CHE-AdamWRecoveryOnly | M8-RecoveryOnlyDeconfound | TCTRL-AdamWExtraStepsMatchedTime | {} |

## 4. Carrier × mechanism h800 summary

```text
best_carrier = D-CHE
best_mechanism_family = M4-FunctionSpaceProximal
best_method = A-M4a-D-CHE-AdamSubspaceProximal-alpha000
S2_weak_productive_dynamics_reached = 0
S3_productive_debt_recovery_reached = 0
```

| carrier | mechanism | best method | source | retention | tail | LineC | AUC | S2 | S3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | M1-ProductivePulseRecovery | A-M1b-D-CHE-PulseOnce-LRCooldownRecovery | 0.10326735178629558 | 0.24161653882927364 | 0.0 | 0.1111111111111111 | 1.0190985124408296 | 0 | 0 |
| D-CHE | M2-SplitConsensusSignalSubspace | A-M2b-D-CHE-SplitConsensusRoleBlockMetric | 0.0369714101155599 | 0.20211979332897398 | 0.0 | 0.2222222222222222 | 1.0213835065809673 | 0 | 0 |
| D-CHE | M3-PopRiskDriftSNR | A-M3b-D-CHE-DegreeRoleSNRPreconditioner | 0.0369714101155599 | 0.20211979332897398 | 0.0 | 0.2222222222222222 | 1.0213835065809673 | 0 | 0 |
| D-CHE | M4-FunctionSpaceProximal | A-M4a-D-CHE-AdamSubspaceProximal-alpha000 | 0.12557220458984375 | 0.7634300026628706 | 0.0 | 0.2222222222222222 | 1.0299587646966346 | 0 | 0 |
| D-CHE | M5-OptimizerAwareAlignedFU | A-M5g-D-CHE-DecoupledDecayPlusFU | 0.031272292137145996 | 0.2339310480488671 | 0.0 | 0.1111111111111111 | 1.01168083566179 | 0 | 0 |
| D-CHE | M6-CarrierSpecificReparamFU | A-M6b-D-CHE-ReadoutBasisDecoupledCarrier | 0.033004429605272084 | 0.23247790998882717 | 0.0 | 0.1111111111111111 | 1.0115325351402509 | 0 | 0 |
| D-CHE | M7-LateAttachSnapshotFU | A-M7e-D-CHE-LateAttachHighSourceLowDebtWindow | 0.09098859628041585 | 0.24172482308414248 | 0.0 | 0.1111111111111111 | 1.005980372287251 | 0 | 0 |
| D-CHE | M8-RecoveryOnlyDeconfound | A-M8b-D-CHE-LRCooldownOnly | 0.09240689542558458 | 0.2413255911734369 | 0.0 | 0.1111111111111111 | 1.0167625057437149 | 0 | 0 |
| MLP | M1-ProductivePulseRecovery | B-M1b-MLP-PulseOnce-LRCooldownRecovery | -0.41173793209923637 | 0.623678618007236 | 0.0 | 0.0 | 0.9997358825089281 | 0 | 0 |
| MLP | M2-SplitConsensusSignalSubspace | B-M2b-MLP-HiddenSplitConsensusRoleBlockMetric | -0.7691405481762357 | 0.5913229733705521 | 0.0 | 0.0 | 1.0962088182699152 | 0 | 0 |
| MLP | M3-PopRiskDriftSNR | B-M3e-MLP-SNRRecoveryScheduler | -0.54572594165802 | 0.8257695900069343 | 0.0 | 0.0 | 1.0282110926144081 | 0 | 0 |
| MLP | M4-FunctionSpaceProximal | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.19552964634365505 | 0.8310222559505038 | 0.0 | 0.0 | 0.9734341053805972 | 0 | 0 |
| MLP | M5-OptimizerAwareAlignedFU | B-M5b-MLP-SoftCautiousFU | -0.6311299138598971 | 0.5993089179197947 | 0.0 | 0.0 | 1.045973949500832 | 0 | 0 |
| MLP | M6-CarrierSpecificReparamFU | B-M6b-MLP-HiddenReadoutDecoupledCarrier | -0.4549909300274319 | 0.6134240859084659 | 0.0 | 0.0 | 1.015046713738643 | 0 | 0 |
| MLP | M7-LateAttachSnapshotFU | B-M7e-MLP-LateAttachHighSourceLowDebtWindow | -0.41747095849778915 | 0.6130791438950433 | 0.0 | 0.0 | 0.9669544341239631 | 0 | 0 |
| MLP | M8-RecoveryOnlyDeconfound | B-M8b-MLP-LRCooldownOnly | -0.4153700139787462 | 0.6132539643181695 | 0.0 | 0.0 | 1.0013850262896786 | 0 | 0 |

Line A method summary：

| method | mechanism | source | retention | tail | LineC | AUC | bad |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A-M1a-D-CHE-PulseOnce-AdamWRecovery | M1-ProductivePulseRecovery | 0.029937916331821017 | 0.23556469629208246 | 0.0 | 0.1111111111111111 | 1.0117920138374015 | 1.0 |
| A-M1b-D-CHE-PulseOnce-LRCooldownRecovery | M1-ProductivePulseRecovery | 0.10326735178629558 | 0.24161653882927364 | 0.0 | 0.1111111111111111 | 1.0190985124408296 | 1.0 |
| A-M1c-D-CHE-PulseOnce-MomentumDampingRecovery | M1-ProductivePulseRecovery | 0.03025874826643202 | 0.24326132734616598 | 0.0 | 0.2222222222222222 | 1.0054976326898215 | 1.0 |
| A-M1d-D-CHE-PulseOnce-DecoupledGlobalDecayRecovery | M1-ProductivePulseRecovery | 0.031272292137145996 | 0.2339310480488671 | 0.0 | 0.1111111111111111 | 1.01168083566179 | 1.0 |
| A-M1e-D-CHE-PulseOnce-DegreeWiseDecayRecovery | M1-ProductivePulseRecovery | 0.030606759919060603 | 0.2347258081038793 | 0.0 | 0.1111111111111111 | 1.0117362827531027 | 1.0 |
| A-M1f-D-CHE-PulseOnce-EMAConsolidation | M1-ProductivePulseRecovery | 0.03467269738515218 | 0.248162647916211 | 0.0 | 0.1111111111111111 | 1.0027334747423828 | 1.0 |
| A-M1g-D-CHE-PulseOnce-LookaheadConsolidation | M1-ProductivePulseRecovery | 0.03541072209676107 | 0.24915256599585214 | 0.0 | 0.1111111111111111 | 1.002292693374977 | 1.0 |
| A-M1h-D-CHE-PulseOnce-SWAConsolidation | M1-ProductivePulseRecovery | 0.03388071060180664 | 0.25218167321549523 | 0.0 | 0.1111111111111111 | 1.0018125943640848 | 1.0 |
| A-M1i-D-CHE-PulseEvery50-AdamWRecovery | M1-ProductivePulseRecovery | 0.029937916331821017 | 0.23556469629208246 | 0.0 | 0.1111111111111111 | 1.0117920138374015 | 1.0 |
| A-M1j-D-CHE-EarlyOnlyPulse-AdamWRecovery | M1-ProductivePulseRecovery | -0.0009143087599012586 | 0.23701926155222786 | 0.0 | 0.1111111111111111 | 1.0024001035538428 | 1.0 |
| A-M1k-D-CHE-MidOnlyPulse-AdamWRecovery | M1-ProductivePulseRecovery | -0.007262865702311198 | 0.23756685770220226 | 0.0 | 0.2222222222222222 | 0.9969060991172094 | 1.0 |
| A-M1l-D-CHE-LateOnlyPulse-AdamWRecovery | M1-ProductivePulseRecovery | -0.01428683598836263 | 0.23731889410151374 | 0.0 | 0.2222222222222222 | 0.9863318051893987 | 1.0 |
| A-M2a-D-CHE-SplitConsensusDiagMetric | M2-SplitConsensusSignalSubspace | -0.003998716672261556 | 0.7547216647201114 | 0.0 | 0.3333333333333333 | 1.000748699420952 | 1.0 |
| A-M2b-D-CHE-SplitConsensusRoleBlockMetric | M2-SplitConsensusSignalSubspace | 0.0369714101155599 | 0.20211979332897398 | 0.0 | 0.2222222222222222 | 1.0213835065809673 | 1.0 |
| A-M2c-D-CHE-SplitConsensusLowRank-r4 | M2-SplitConsensusSignalSubspace | -0.003998716672261556 | 0.7547216647201114 | 0.0 | 0.3333333333333333 | 1.000748699420952 | 1.0 |
| A-M2d-D-CHE-SplitConsensusLowRank-r8 | M2-SplitConsensusSignalSubspace | -0.003998716672261556 | 0.7547216647201114 | 0.0 | 0.3333333333333333 | 1.000748699420952 | 1.0 |
| A-M2e-D-CHE-SplitConsensusMetricOnlyNoProjection | M2-SplitConsensusSignalSubspace | -0.003998716672261556 | 0.7547216647201114 | 0.0 | 0.3333333333333333 | 1.000748699420952 | 1.0 |
| A-M2f-D-CHE-SplitConsensusProjectionPlusAdamV | M2-SplitConsensusSignalSubspace | -0.003998716672261556 | 0.7547216647201114 | 0.0 | 0.3333333333333333 | 1.000748699420952 | 1.0 |
| A-M2g-D-CHE-NegativeEigenQuarantineDiagnostic | M2-SplitConsensusSignalSubspace | -0.003998716672261556 | 0.7547216647201114 | 0.0 | 0.3333333333333333 | 1.000748699420952 | 1.0 |
| A-M3a-D-CHE-ParameterSNRPreconditioner | M3-PopRiskDriftSNR | -0.003998716672261556 | 0.7547216647201114 | 0.0 | 0.3333333333333333 | 1.000748699420952 | 1.0 |
| A-M3b-D-CHE-DegreeRoleSNRPreconditioner | M3-PopRiskDriftSNR | 0.0369714101155599 | 0.20211979332897398 | 0.0 | 0.2222222222222222 | 1.0213835065809673 | 1.0 |
| A-M3c-D-CHE-BasisChannelSNRCoverBoundary | M3-PopRiskDriftSNR | -0.0017551978429158528 | 0.7524689137935638 | 0.0 | 0.3333333333333333 | 1.0005597200527165 | 1.0 |
| A-M3d-D-CHE-SNRPulseScheduler | M3-PopRiskDriftSNR | -0.003998716672261556 | 0.7547216647201114 | 0.0 | 0.3333333333333333 | 1.000748699420952 | 1.0 |
| A-M3e-D-CHE-SNRRecoveryScheduler | M3-PopRiskDriftSNR | -0.017210377587212458 | 0.7645038068294525 | 0.0 | 0.2222222222222222 | 0.9966782075105844 | 1.0 |
| A-M3f-D-CHE-SNRReservoirGuard | M3-PopRiskDriftSNR | -0.0028872622383965384 | 0.7520440022150675 | 0.0 | 0.3333333333333333 | 1.0006562878489806 | 1.0 |
| A-M4a-D-CHE-AdamSubspaceProximal-alpha000 | M4-FunctionSpaceProximal | 0.12557220458984375 | 0.7634300026628706 | 0.0 | 0.2222222222222222 | 1.0299587646966346 | 1.0 |
| A-M4b-D-CHE-PerExampleLowRankProximal-alpha025 | M4-FunctionSpaceProximal | 0.11325258678860134 | 0.7633089886771308 | 0.0 | 0.1111111111111111 | 1.0241981222130359 | 1.0 |
| A-M4c-D-CHE-OutputJacobianSketchProximal-alpha050 | M4-FunctionSpaceProximal | 0.09383397632175022 | 0.7625950111283196 | 0.0 | 0.1111111111111111 | 1.017521070951311 | 1.0 |
| A-M4d-D-CHE-SplitB1B2TransferProximal-alpha100 | M4-FunctionSpaceProximal | 0.057322144508361816 | 0.7606566184096866 | 0.0 | 0.1111111111111111 | 1.0101517579299515 | 1.0 |
| A-M4e-D-CHE-RecoveryAwareProximal-alpha200 | M4-FunctionSpaceProximal | 0.011784791946411133 | 0.7660496400462257 | 0.0 | 0.2222222222222222 | 1.0014960563234325 | 1.0 |
| A-M4f-D-CHE-ControlResidualizedProximal-alpha100 | M4-FunctionSpaceProximal | 0.09383397632175022 | 0.7625950111283196 | 0.0 | 0.1111111111111111 | 1.017521070951311 | 1.0 |
| A-M5a-D-CHE-CautiousFU | M5-OptimizerAwareAlignedFU | 0.03044989373948839 | 0.24041535208622614 | 0.0 | 0.1111111111111111 | 1.0075178414458998 | 1.0 |
| A-M5b-D-CHE-SoftCautiousFU | M5-OptimizerAwareAlignedFU | 0.030435019069247775 | 0.24187831083933511 | 0.0 | 0.2222222222222222 | 1.0063253817962208 | 1.0 |
| A-M5c-D-CHE-MGUPStyleReweightFU | M5-OptimizerAwareAlignedFU | 0.030962361229790583 | 0.24473826669984394 | 0.0 | 0.2222222222222222 | 1.0046802713085574 | 1.0 |
| A-M5d-D-CHE-AdamSecondMomentScaledFU | M5-OptimizerAwareAlignedFU | -1.2830092642042372 | 0.7262329459190369 | 0.0 | 0.1111111111111111 | 1.1151016739484645 | 1.0 |
| A-M5e-D-CHE-SophiaDiagLiteClippedFU | M5-OptimizerAwareAlignedFU | -1.5059364371829562 | 0.741761714220047 | 0.0 | 0.0 | 1.1538564819630313 | 1.0 |
| A-M5f-D-CHE-BlockSecondMomentFU | M5-OptimizerAwareAlignedFU | -1.2447997993893094 | 0.7325163516733382 | 0.0 | 0.2222222222222222 | 1.1064052851951822 | 1.0 |
| A-M5g-D-CHE-DecoupledDecayPlusFU | M5-OptimizerAwareAlignedFU | 0.031272292137145996 | 0.2339310480488671 | 0.0 | 0.1111111111111111 | 1.01168083566179 | 1.0 |
| A-M6a-D-CHE-LowDegreeSignalHighDegreeReservoir | M6-CarrierSpecificReparamFU | 0.029937916331821017 | 0.23556469629208246 | 0.0 | 0.1111111111111111 | 1.0117920138374015 | 1.0 |
| A-M6b-D-CHE-ReadoutBasisDecoupledCarrier | M6-CarrierSpecificReparamFU | 0.033004429605272084 | 0.23247790998882717 | 0.0 | 0.1111111111111111 | 1.0115325351402509 | 1.0 |
| A-M6c-D-CHE-OrthogonalDegreeBankCarrier | M6-CarrierSpecificReparamFU | 0.030606759919060603 | 0.2347258081038793 | 0.0 | 0.1111111111111111 | 1.0117362827531027 | 1.0 |
| A-M6d-D-CHE-OutputJacobianCarrier | M6-CarrierSpecificReparamFU | -0.003998716672261556 | 0.7547216647201114 | 0.0 | 0.3333333333333333 | 1.000748699420952 | 1.0 |
| A-M6e-D-CHE-DualBankSignalReservoirCarrier | M6-CarrierSpecificReparamFU | 0.032063033845689565 | 0.2333100727862782 | 0.0 | 0.1111111111111111 | 1.0116127685672096 | 1.0 |
| A-M6f-D-CHE-HighDegreeQuarantineCarrier | M6-CarrierSpecificReparamFU | 0.030939870410495333 | 0.23432142784198126 | 0.0 | 0.1111111111111111 | 1.0117085275328275 | 1.0 |
| A-M7a-D-CHE-LateAttachEarlyCheckpoint | M7-LateAttachSnapshotFU | -0.0009143087599012586 | 0.23701926155222786 | 0.0 | 0.1111111111111111 | 1.0024001035538428 | 1.0 |
| A-M7b-D-CHE-LateAttachMidCheckpoint | M7-LateAttachSnapshotFU | -0.007262865702311198 | 0.23756685770220226 | 0.0 | 0.2222222222222222 | 0.9969060991172094 | 1.0 |
| A-M7c-D-CHE-LateAttachLateCheckpoint | M7-LateAttachSnapshotFU | -0.01428683598836263 | 0.23731889410151374 | 0.0 | 0.2222222222222222 | 0.9863318051893987 | 1.0 |
| A-M7d-D-CHE-LateAttachAfterLossPlateau | M7-LateAttachSnapshotFU | -0.01924288272857666 | 0.24537581205368042 | 0.0 | 0.3333333333333333 | 0.9784857253249122 | 1.0 |
| A-M7e-D-CHE-LateAttachHighSourceLowDebtWindow | M7-LateAttachSnapshotFU | 0.09098859628041585 | 0.24172482308414248 | 0.0 | 0.1111111111111111 | 1.005980372287251 | 1.0 |
| A-M8a-D-CHE-AdamWRecoveryOnly | M8-RecoveryOnlyDeconfound | -0.013537115520901151 | 0.23687240398592418 | 0.0 | 0.2222222222222222 | 1.0010032033543799 | 1.0 |
| A-M8b-D-CHE-LRCooldownOnly | M8-RecoveryOnlyDeconfound | 0.09240689542558458 | 0.2413255911734369 | 0.0 | 0.1111111111111111 | 1.0167625057437149 | 1.0 |
| A-M8c-D-CHE-DecayRecoveryOnly | M8-RecoveryOnlyDeconfound | -0.535791900422838 | 5.6795373224203e-07 | 1.0 | 0.3333333333333333 | 1.1437172661914863 | 1.0 |
| A-M8d-D-CHE-EMASWARecoveryOnly | M8-RecoveryOnlyDeconfound | -0.016842696401807997 | 0.2506778612732887 | 0.0 | 0.4444444444444444 | 0.9945990926910745 | 1.0 |
| A-M8e-D-CHE-LookaheadRecoveryOnly | M8-RecoveryOnlyDeconfound | -0.014865186479356553 | 0.2480152580473158 | 0.0 | 0.3333333333333333 | 0.9944890697943545 | 1.0 |
| A-M8f-D-CHE-MomentumDampingOnly | M8-RecoveryOnlyDeconfound | -0.013962454266018338 | 0.24194492482476765 | 0.0 | 0.2222222222222222 | 0.9956583310168116 | 1.0 |

Line B method summary：

| method | mechanism | source | retention | tail | LineC | AUC | bad |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-M1a-MLP-PulseOnce-AdamWRecovery | M1-ProductivePulseRecovery | -0.7691406011581421 | 0.5913225445482466 | 0.0 | 0.0 | 1.0962088076769818 | 1.0 |
| B-M1b-MLP-PulseOnce-LRCooldownRecovery | M1-ProductivePulseRecovery | -0.41173793209923637 | 0.623678618007236 | 0.0 | 0.0 | 0.9997358825089281 | 1.0 |
| B-M1c-MLP-PulseOnce-MomentumDampingRecovery | M1-ProductivePulseRecovery | -0.46822112136416966 | 0.603556278679106 | 0.0 | 0.0 | 1.0200911622520499 | 1.0 |
| B-M1d-MLP-PulseOnce-DecoupledDecayRecovery | M1-ProductivePulseRecovery | -0.7482052114274766 | 0.5752273599306742 | 0.0 | 0.0 | 1.0940971859048576 | 1.0 |
| B-M1e-MLP-PulseOnce-FeatureRecenterRecovery | M1-ProductivePulseRecovery | -0.4549909300274319 | 0.6134240859084659 | 0.0 | 0.0 | 1.015046713738643 | 1.0 |
| B-M1f-MLP-PulseOnce-EMAConsolidation | M1-ProductivePulseRecovery | -0.4651866488986545 | 0.6242384612560272 | 0.0 | 0.0 | 1.0145979632765032 | 1.0 |
| B-M1g-MLP-PulseOnce-LookaheadConsolidation | M1-ProductivePulseRecovery | -0.5234416193432279 | 0.625892784860399 | 0.0 | 0.0 | 1.0235680913681495 | 1.0 |
| B-M1h-MLP-PulseOnce-SWAConsolidation | M1-ProductivePulseRecovery | -0.5253076685799493 | 0.6258152027924856 | 0.0 | 0.0 | 1.0246307604591607 | 1.0 |
| B-M1i-MLP-PulseEvery50-AdamWRecovery | M1-ProductivePulseRecovery | -0.7691406011581421 | 0.5913225445482466 | 0.0 | 0.0 | 1.0962088076769818 | 1.0 |
| B-M1j-MLP-EarlyOnlyPulse-AdamWRecovery | M1-ProductivePulseRecovery | -0.8083672391043769 | 0.5921509000990126 | 0.0 | 0.0 | 1.0993435887866925 | 1.0 |
| B-M1k-MLP-MidOnlyPulse-AdamWRecovery | M1-ProductivePulseRecovery | -0.8429834975136651 | 0.5924354592959086 | 0.0 | 0.0 | 1.089859201541318 | 1.0 |
| B-M1l-MLP-LateOnlyPulse-AdamWRecovery | M1-ProductivePulseRecovery | -0.839893619219462 | 0.5928675267431471 | 0.0 | 0.0 | 1.0852540680934926 | 1.0 |
| B-M2a-MLP-HiddenSplitConsensusDiagMetric | M2-SplitConsensusSignalSubspace | -0.8704189591937594 | 0.7865732643339369 | 0.0 | 0.0 | 1.0951609118600465 | 1.0 |
| B-M2b-MLP-HiddenSplitConsensusRoleBlockMetric | M2-SplitConsensusSignalSubspace | -0.7691405481762357 | 0.5913229733705521 | 0.0 | 0.0 | 1.0962088182699152 | 1.0 |
| B-M2c-MLP-HiddenSplitConsensusLowRank-r4 | M2-SplitConsensusSignalSubspace | -0.8704189591937594 | 0.7865732643339369 | 0.0 | 0.0 | 1.0951609118600465 | 1.0 |
| B-M2d-MLP-HiddenSplitConsensusLowRank-r8 | M2-SplitConsensusSignalSubspace | -0.8704189591937594 | 0.7865732643339369 | 0.0 | 0.0 | 1.0951609118600465 | 1.0 |
| B-M2e-MLP-HiddenMetricOnlyNoProjection | M2-SplitConsensusSignalSubspace | -0.8704189591937594 | 0.7865732643339369 | 0.0 | 0.0 | 1.0951609118600465 | 1.0 |
| B-M2f-MLP-HiddenProjectionPlusAdamV | M2-SplitConsensusSignalSubspace | -0.8704189591937594 | 0.7865732643339369 | 0.0 | 0.0 | 1.0951609118600465 | 1.0 |
| B-M2g-MLP-HiddenNegativeEigenQuarantineDiagnostic | M2-SplitConsensusSignalSubspace | -0.8704189591937594 | 0.7865732643339369 | 0.0 | 0.0 | 1.0951609118600465 | 1.0 |
| B-M3a-MLP-ParameterSNRPreconditioner | M3-PopRiskDriftSNR | -0.8704189591937594 | 0.7865732643339369 | 0.0 | 0.0 | 1.0951609118600465 | 1.0 |
| B-M3b-MLP-HiddenRoleSNRPreconditioner | M3-PopRiskDriftSNR | -0.7691405481762357 | 0.5913229733705521 | 0.0 | 0.0 | 1.0962088182699152 | 1.0 |
| B-M3c-MLP-FeatureChannelSNRCoverBoundary | M3-PopRiskDriftSNR | -0.8704189591937594 | 0.7865732643339369 | 0.0 | 0.0 | 1.0951609118600465 | 1.0 |
| B-M3d-MLP-SNRPulseScheduler | M3-PopRiskDriftSNR | -0.8704189591937594 | 0.7865732643339369 | 0.0 | 0.0 | 1.0951609118600465 | 1.0 |
| B-M3e-MLP-SNRRecoveryScheduler | M3-PopRiskDriftSNR | -0.54572594165802 | 0.8257695900069343 | 0.0 | 0.0 | 1.0282110926144081 | 1.0 |
| B-M3f-MLP-SNRReservoirGuard | M3-PopRiskDriftSNR | -0.8495999839570787 | 0.7754063573148515 | 0.0 | 0.0 | 1.0930508404702755 | 1.0 |
| B-M4a-MLP-AdamSubspaceProximal-alpha000 | M4-FunctionSpaceProximal | -0.19552964634365505 | 0.8310222559505038 | 0.0 | 0.0 | 0.9734341053805972 | 1.0 |
| B-M4b-MLP-PerExampleLowRankProximal-alpha025 | M4-FunctionSpaceProximal | -0.3074572483698527 | 0.828018350733651 | 0.0 | 0.0 | 0.9844641453422793 | 1.0 |
| B-M4c-MLP-OutputJacobianSketchProximal-alpha050 | M4-FunctionSpaceProximal | -0.41528116332160103 | 0.8206900258858999 | 0.0 | 0.0 | 0.999395810949242 | 1.0 |
| B-M4d-MLP-SplitB1B2TransferProximal-alpha100 | M4-FunctionSpaceProximal | -0.5388705730438232 | 0.8112130032645332 | 0.0 | 0.0 | 1.02485277205274 | 1.0 |
| B-M4e-MLP-RecoveryAwareProximal-alpha200 | M4-FunctionSpaceProximal | -0.45998595820532906 | 0.8231186866760254 | 0.0 | 0.0 | 1.0055134192281967 | 1.0 |
| B-M4f-MLP-ControlResidualizedProximal-alpha100 | M4-FunctionSpaceProximal | -0.41528116332160103 | 0.8206900258858999 | 0.0 | 0.0 | 0.999395810949242 | 1.0 |
| B-M5a-MLP-CautiousAdamW | M5-OptimizerAwareAlignedFU | -0.7862126694785224 | 0.5913364556100633 | 0.0 | 0.0 | 1.095354582171635 | 1.0 |
| B-M5b-MLP-SoftCautiousFU | M5-OptimizerAwareAlignedFU | -0.6311299138598971 | 0.5993089179197947 | 0.0 | 0.0 | 1.045973949500832 | 1.0 |
| B-M5c-MLP-MGUPStyleReweight | M5-OptimizerAwareAlignedFU | -0.7691406011581421 | 0.5913225445482466 | 0.0 | 0.0 | 1.0962088076769818 | 1.0 |
| B-M5d-MLP-AdamSecondMomentScaledFU | M5-OptimizerAwareAlignedFU | -4.282999528778924 | 0.7513079278998904 | 0.0 | 0.0 | 1.4131243361592825 | 1.0 |
| B-M5e-MLP-SophiaDiagLiteClippedFU | M5-OptimizerAwareAlignedFU | -5.35028252336714 | 0.7242764598793454 | 0.006950162313630708 | 0.0 | 1.5660509001173393 | 1.0 |
| B-M5f-MLP-BlockSecondMomentFU | M5-OptimizerAwareAlignedFU | -3.628608398967319 | 0.7674861815240648 | 0.0 | 0.0 | 1.312767752530227 | 1.0 |
| B-M5g-MLP-DecoupledDecayPlusFU | M5-OptimizerAwareAlignedFU | -0.7482052114274766 | 0.5752273599306742 | 0.0 | 0.0 | 1.0940971859048576 | 1.0 |
| B-M6a-MLP-HiddenSubspaceCarrier | M6-CarrierSpecificReparamFU | -0.7691406011581421 | 0.5913225445482466 | 0.0 | 0.0 | 1.0962088076769818 | 1.0 |
| B-M6b-MLP-HiddenReadoutDecoupledCarrier | M6-CarrierSpecificReparamFU | -0.4549909300274319 | 0.6134240859084659 | 0.0 | 0.0 | 1.015046713738643 | 1.0 |
| B-M6c-MLP-HiddenOrthogonalBankCarrier | M6-CarrierSpecificReparamFU | -0.4677587085300022 | 0.6203908191786872 | 0.0 | 0.0 | 1.0141518413855724 | 1.0 |
| B-M6d-MLP-OutputJacobianCarrier | M6-CarrierSpecificReparamFU | -0.8704189591937594 | 0.7865732643339369 | 0.0 | 0.0 | 1.0951609118600465 | 1.0 |
| B-M6e-MLP-DualBankSignalReservoirCarrier | M6-CarrierSpecificReparamFU | -0.7482052114274766 | 0.5752273599306742 | 0.0 | 0.0 | 1.0940971859048576 | 1.0 |
| B-M6f-MLP-HiddenCarrierDecay | M6-CarrierSpecificReparamFU | -0.7482052114274766 | 0.5752273599306742 | 0.0 | 0.0 | 1.0940971859048576 | 1.0 |
| B-M7a-MLP-LateAttachEarlyCheckpoint | M7-LateAttachSnapshotFU | -0.8083672391043769 | 0.5921509000990126 | 0.0 | 0.0 | 1.0993435887866925 | 1.0 |
| B-M7b-MLP-LateAttachMidCheckpoint | M7-LateAttachSnapshotFU | -0.8429834975136651 | 0.5924354592959086 | 0.0 | 0.0 | 1.089859201541318 | 1.0 |
| B-M7c-MLP-LateAttachLateCheckpoint | M7-LateAttachSnapshotFU | -0.839893619219462 | 0.5928675267431471 | 0.0 | 0.0 | 1.0852540680934926 | 1.0 |
| B-M7d-MLP-LateAttachAfterLossPlateau | M7-LateAttachSnapshotFU | -0.6089112626181709 | 0.614671066403389 | 0.0 | 0.0 | 1.0097500731451736 | 1.0 |
| B-M7e-MLP-LateAttachHighSourceLowDebtWindow | M7-LateAttachSnapshotFU | -0.41747095849778915 | 0.6130791438950433 | 0.0 | 0.0 | 0.9669544341239631 | 1.0 |
| B-M8a-MLP-AdamWRecoveryOnly | M8-RecoveryOnlyDeconfound | -0.7862126694785224 | 0.5913364556100633 | 0.0 | 0.0 | 1.095354582171635 | 1.0 |
| B-M8b-MLP-LRCooldownOnly | M8-RecoveryOnlyDeconfound | -0.4153700139787462 | 0.6132539643181695 | 0.0 | 0.0 | 1.0013850262896786 | 1.0 |
| B-M8c-MLP-DecayRecoveryOnly | M8-RecoveryOnlyDeconfound | -0.7482052114274766 | 0.5752273599306742 | 0.0 | 0.0 | 1.0940971859048576 | 1.0 |
| B-M8d-MLP-EMASWARecoveryOnly | M8-RecoveryOnlyDeconfound | -0.691344698270162 | 0.6201151377624936 | 0.0 | 0.0 | 1.0547496189360694 | 1.0 |
| B-M8e-MLP-LookaheadRecoveryOnly | M8-RecoveryOnlyDeconfound | -0.6549386050966051 | 0.6155509700377783 | 0.0 | 0.0 | 1.044625220601195 | 1.0 |
| B-M8f-MLP-MomentumDampingOnly | M8-RecoveryOnlyDeconfound | -0.6390104956097074 | 0.6115511788262261 | 0.0 | 0.0 | 1.0462918016502305 | 1.0 |

Line A h1600 top-2：

| method | dataset | seed | source | bad | tail | LineC |
| --- | --- | --- | --- | --- | --- | --- |
| A-M4a-D-CHE-AdamSubspaceProximal-alpha000 | MNIST | 0 | 0.0 | 1 | 0.15816882160130685 | 0.0 |
| A-M4b-D-CHE-PerExampleLowRankProximal-alpha025 | MNIST | 0 | -0.02806723117828369 | 1 | 0.2751686681398004 | 1.0 |
| A-M4a-D-CHE-AdamSubspaceProximal-alpha000 | MNIST | 1 | 0.0 | 1 | 0.0 | 0.0 |
| A-M4b-D-CHE-PerExampleLowRankProximal-alpha025 | MNIST | 1 | -0.01874244213104248 | 1 | 0.0 | 0.0 |
| A-M4a-D-CHE-AdamSubspaceProximal-alpha000 | MNIST | 2 | 0.0 | 1 | 0.0 | 0.0 |
| A-M4b-D-CHE-PerExampleLowRankProximal-alpha025 | MNIST | 2 | -0.030115842819213867 | 1 | 0.0 | 0.0 |
| A-M4a-D-CHE-AdamSubspaceProximal-alpha000 | Fashion-MNIST | 0 | 0.0 | 1 | 0.0 | 1.0 |
| A-M4b-D-CHE-PerExampleLowRankProximal-alpha025 | Fashion-MNIST | 0 | -0.03679823875427246 | 1 | 0.0 | 0.0 |
| A-M4a-D-CHE-AdamSubspaceProximal-alpha000 | Fashion-MNIST | 1 | 0.0 | 1 | 0.0 | 0.0 |
| A-M4b-D-CHE-PerExampleLowRankProximal-alpha025 | Fashion-MNIST | 1 | -0.03171229362487793 | 1 | 0.0 | 0.0 |
| A-M4a-D-CHE-AdamSubspaceProximal-alpha000 | Fashion-MNIST | 2 | 0.0 | 1 | 0.0 | 1.0 |
| A-M4b-D-CHE-PerExampleLowRankProximal-alpha025 | Fashion-MNIST | 2 | -0.03046250343322754 | 1 | 0.0 | 1.0 |
| A-M4a-D-CHE-AdamSubspaceProximal-alpha000 | KMNIST | 0 | 0.0 | 1 | 0.0 | 1.0 |
| A-M4b-D-CHE-PerExampleLowRankProximal-alpha025 | KMNIST | 0 | -0.021060466766357422 | 1 | 0.0 | 1.0 |
| A-M4a-D-CHE-AdamSubspaceProximal-alpha000 | KMNIST | 1 | 0.0 | 1 | 0.0 | 0.0 |
| A-M4b-D-CHE-PerExampleLowRankProximal-alpha025 | KMNIST | 1 | -0.03855597972869873 | 1 | 0.0 | 0.0 |
| A-M4a-D-CHE-AdamSubspaceProximal-alpha000 | KMNIST | 2 | 0.0 | 1 | 0.0 | 1.0 |
| A-M4b-D-CHE-PerExampleLowRankProximal-alpha025 | KMNIST | 2 | -0.01936197280883789 | 1 | 0.0 | 1.0 |

Line B h1600 top-2：

| method | dataset | seed | source | bad | tail | LineC |
| --- | --- | --- | --- | --- | --- | --- |
| B-M4a-MLP-AdamSubspaceProximal-alpha000 | MNIST | 0 | 0.0 | 1 | 0.0 | 0.0 |
| B-M4b-MLP-PerExampleLowRankProximal-alpha025 | MNIST | 0 | -0.27995896339416504 | 1 | 0.0 | 0.0 |
| B-M4a-MLP-AdamSubspaceProximal-alpha000 | MNIST | 1 | 0.0 | 1 | 0.0 | 0.0 |
| B-M4b-MLP-PerExampleLowRankProximal-alpha025 | MNIST | 1 | -0.027157306671142578 | 1 | 0.0 | 0.0 |
| B-M4a-MLP-AdamSubspaceProximal-alpha000 | MNIST | 2 | 0.0 | 1 | 0.0 | 0.0 |
| B-M4b-MLP-PerExampleLowRankProximal-alpha025 | MNIST | 2 | -0.04712653160095215 | 1 | 0.0 | 0.0 |
| B-M4a-MLP-AdamSubspaceProximal-alpha000 | Fashion-MNIST | 0 | 0.0 | 1 | 0.0 | 0.0 |
| B-M4b-MLP-PerExampleLowRankProximal-alpha025 | Fashion-MNIST | 0 | -0.22931909561157227 | 1 | 0.0 | 0.0 |
| B-M4a-MLP-AdamSubspaceProximal-alpha000 | Fashion-MNIST | 1 | 0.0 | 1 | 0.0 | 0.0 |
| B-M4b-MLP-PerExampleLowRankProximal-alpha025 | Fashion-MNIST | 1 | -0.16722393035888672 | 1 | 0.0 | 0.0 |
| B-M4a-MLP-AdamSubspaceProximal-alpha000 | Fashion-MNIST | 2 | 0.0 | 1 | 0.0 | 0.0 |
| B-M4b-MLP-PerExampleLowRankProximal-alpha025 | Fashion-MNIST | 2 | -0.12282633781433105 | 1 | 0.0 | 0.0 |
| B-M4a-MLP-AdamSubspaceProximal-alpha000 | KMNIST | 0 | 0.0 | 1 | 0.0 | 0.0 |
| B-M4b-MLP-PerExampleLowRankProximal-alpha025 | KMNIST | 0 | -0.06720352172851562 | 1 | 0.0 | 0.0 |
| B-M4a-MLP-AdamSubspaceProximal-alpha000 | KMNIST | 1 | 0.0 | 1 | 0.0 | 0.0 |
| B-M4b-MLP-PerExampleLowRankProximal-alpha025 | KMNIST | 1 | -0.16223978996276855 | 1 | 0.0 | 0.0 |
| B-M4a-MLP-AdamSubspaceProximal-alpha000 | KMNIST | 2 | -0.022079944610595703 | 1 | 0.0 | 0.0 |
| B-M4b-MLP-PerExampleLowRankProximal-alpha025 | KMNIST | 2 | 0.0 | 1 | 0.0 | 0.0 |

## 5. Line C / D / F / M results

```text
line_c_rows = 63
line_c_reanchor_gate_pass = 0
line_c_best_method = C2-LQ-ProtocolMatchedReanchor
line_c_best_macro_delta_vs_MLP = 0.28125
line_d_rat_rows = 45
line_d_rat_best_method = D1-RAT-G7AnalogPulseOnce-then-AdamWRecovery
line_d_rat_gate_pass = 0
line_f_rows = 153
line_f_best_family = D-FOU
line_f_best_dataset_seed_pass_count = 4 / 9
line_m_rows = 55
generic_functional_dynamics_rows = 0
kan_specific_advantage_rows = 0
```

Line C reanchor summary：

| method | dataset | seed | delta | near | near count | LineC | gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C0-HistoricalLQReferenceReplay | MNIST | 0 | 0.03125 | 1 | 6 | 1 | 0 |
| C1-CurrentLQProtocolReplay | MNIST | 0 | 0.03125 | 1 | 7 | 1 | 0 |
| C2-LQ-ProtocolMatchedReanchor | MNIST | 0 | 0.0 | 1 | 7 | 1 | 0 |
| C3-LQ-RowGateRobustReanchor | MNIST | 0 | 0.0625 | 1 | 8 | 1 | 0 |
| C4-LQ-MacroDeltaPriorityReanchor | MNIST | 0 | 0.0625 | 1 | 6 | 1 | 0 |
| C5-LQ-StepMemoryRecheck | MNIST | 0 | -0.25 | 0 | 6 | 1 | 0 |
| C6-LQ-LineCNoRegressionCheck | MNIST | 0 | -0.03125 | 0 | 6 | 1 | 0 |
| C0-HistoricalLQReferenceReplay | MNIST | 1 | 0.09375 | 1 | 6 | 0 | 0 |
| C1-CurrentLQProtocolReplay | MNIST | 1 | 0.125 | 1 | 7 | 1 | 0 |
| C2-LQ-ProtocolMatchedReanchor | MNIST | 1 | 0.28125 | 1 | 7 | 1 | 0 |
| C3-LQ-RowGateRobustReanchor | MNIST | 1 | 0.1875 | 1 | 8 | 1 | 0 |
| C4-LQ-MacroDeltaPriorityReanchor | MNIST | 1 | 0.21875 | 1 | 6 | 0 | 0 |
| C5-LQ-StepMemoryRecheck | MNIST | 1 | 0.25 | 1 | 6 | 1 | 0 |
| C6-LQ-LineCNoRegressionCheck | MNIST | 1 | 0.28125 | 1 | 6 | 1 | 0 |
| C0-HistoricalLQReferenceReplay | MNIST | 2 | 0.0 | 1 | 6 | 1 | 0 |
| C1-CurrentLQProtocolReplay | MNIST | 2 | 0.0625 | 1 | 7 | 1 | 0 |
| C2-LQ-ProtocolMatchedReanchor | MNIST | 2 | 0.03125 | 1 | 7 | 1 | 0 |
| C3-LQ-RowGateRobustReanchor | MNIST | 2 | 0.125 | 1 | 8 | 1 | 0 |
| C4-LQ-MacroDeltaPriorityReanchor | MNIST | 2 | 0.0625 | 1 | 6 | 1 | 0 |
| C5-LQ-StepMemoryRecheck | MNIST | 2 | 0.0 | 1 | 6 | 1 | 0 |
| C6-LQ-LineCNoRegressionCheck | MNIST | 2 | 0.0 | 1 | 6 | 1 | 0 |
| C0-HistoricalLQReferenceReplay | Fashion-MNIST | 0 | 0.125 | 1 | 6 | 1 | 0 |
| C1-CurrentLQProtocolReplay | Fashion-MNIST | 0 | 0.125 | 1 | 7 | 1 | 0 |
| C2-LQ-ProtocolMatchedReanchor | Fashion-MNIST | 0 | 0.125 | 1 | 7 | 1 | 0 |
| C3-LQ-RowGateRobustReanchor | Fashion-MNIST | 0 | 0.1875 | 1 | 8 | 1 | 0 |
| C4-LQ-MacroDeltaPriorityReanchor | Fashion-MNIST | 0 | 0.21875 | 1 | 6 | 1 | 0 |
| C5-LQ-StepMemoryRecheck | Fashion-MNIST | 0 | 0.25 | 1 | 6 | 1 | 0 |
| C6-LQ-LineCNoRegressionCheck | Fashion-MNIST | 0 | 0.15625 | 1 | 6 | 1 | 0 |
| C0-HistoricalLQReferenceReplay | Fashion-MNIST | 1 | -0.0625 | 0 | 6 | 1 | 0 |
| C1-CurrentLQProtocolReplay | Fashion-MNIST | 1 | -0.0625 | 0 | 7 | 1 | 0 |
| C2-LQ-ProtocolMatchedReanchor | Fashion-MNIST | 1 | 0.0625 | 1 | 7 | 1 | 0 |
| C3-LQ-RowGateRobustReanchor | Fashion-MNIST | 1 | 0.09375 | 1 | 8 | 1 | 0 |
| C4-LQ-MacroDeltaPriorityReanchor | Fashion-MNIST | 1 | -0.03125 | 0 | 6 | 1 | 0 |
| C5-LQ-StepMemoryRecheck | Fashion-MNIST | 1 | -0.125 | 0 | 6 | 1 | 0 |
| C6-LQ-LineCNoRegressionCheck | Fashion-MNIST | 1 | -0.03125 | 0 | 6 | 1 | 0 |
| C0-HistoricalLQReferenceReplay | Fashion-MNIST | 2 | -0.125 | 0 | 6 | 1 | 0 |
| C1-CurrentLQProtocolReplay | Fashion-MNIST | 2 | 0.09375 | 1 | 7 | 1 | 0 |
| C2-LQ-ProtocolMatchedReanchor | Fashion-MNIST | 2 | -0.09375 | 0 | 7 | 1 | 0 |
| C3-LQ-RowGateRobustReanchor | Fashion-MNIST | 2 | -0.09375 | 0 | 8 | 1 | 0 |
| C4-LQ-MacroDeltaPriorityReanchor | Fashion-MNIST | 2 | -0.125 | 0 | 6 | 1 | 0 |
| C5-LQ-StepMemoryRecheck | Fashion-MNIST | 2 | 0.0 | 1 | 6 | 0 | 0 |
| C6-LQ-LineCNoRegressionCheck | Fashion-MNIST | 2 | 0.0625 | 1 | 6 | 1 | 0 |
| C0-HistoricalLQReferenceReplay | KMNIST | 0 | 0.09375 | 1 | 6 | 0 | 0 |
| C1-CurrentLQProtocolReplay | KMNIST | 0 | 0.03125 | 1 | 7 | 1 | 0 |
| C2-LQ-ProtocolMatchedReanchor | KMNIST | 0 | 0.09375 | 1 | 7 | 1 | 0 |
| C3-LQ-RowGateRobustReanchor | KMNIST | 0 | 0.125 | 1 | 8 | 0 | 0 |
| C4-LQ-MacroDeltaPriorityReanchor | KMNIST | 0 | 0.09375 | 1 | 6 | 1 | 0 |
| C5-LQ-StepMemoryRecheck | KMNIST | 0 | 0.125 | 1 | 6 | 0 | 0 |
| C6-LQ-LineCNoRegressionCheck | KMNIST | 0 | 0.09375 | 1 | 6 | 1 | 0 |
| C0-HistoricalLQReferenceReplay | KMNIST | 1 | -0.0625 | 0 | 6 | 0 | 0 |
| C1-CurrentLQProtocolReplay | KMNIST | 1 | -0.09375 | 0 | 7 | 1 | 0 |
| C2-LQ-ProtocolMatchedReanchor | KMNIST | 1 | -0.0625 | 0 | 7 | 0 | 0 |
| C3-LQ-RowGateRobustReanchor | KMNIST | 1 | 0.0625 | 1 | 8 | 0 | 0 |
| C4-LQ-MacroDeltaPriorityReanchor | KMNIST | 1 | -0.03125 | 0 | 6 | 1 | 0 |
| C5-LQ-StepMemoryRecheck | KMNIST | 1 | -0.0625 | 0 | 6 | 1 | 0 |
| C6-LQ-LineCNoRegressionCheck | KMNIST | 1 | -0.125 | 0 | 6 | 1 | 0 |
| C0-HistoricalLQReferenceReplay | KMNIST | 2 | 0.0 | 1 | 6 | 1 | 0 |
| C1-CurrentLQProtocolReplay | KMNIST | 2 | 0.03125 | 1 | 7 | 1 | 0 |
| C2-LQ-ProtocolMatchedReanchor | KMNIST | 2 | 0.03125 | 1 | 7 | 1 | 0 |
| C3-LQ-RowGateRobustReanchor | KMNIST | 2 | 0.0 | 1 | 8 | 0 | 0 |
| C4-LQ-MacroDeltaPriorityReanchor | KMNIST | 2 | 0.03125 | 1 | 6 | 1 | 0 |
| C5-LQ-StepMemoryRecheck | KMNIST | 2 | 0.0 | 1 | 6 | 0 | 0 |
| C6-LQ-LineCNoRegressionCheck | KMNIST | 2 | 0.09375 | 1 | 6 | 1 | 0 |

Line D Rational summary：

| method | pass | source | AUC |
| --- | --- | --- | --- |
| D0-RAT-AdamWMonitor | 0 | -0.5554063849978976 |  |
| D1-RAT-G7AnalogPulseOnce-then-AdamWRecovery | 2 | -0.14552770720587838 |  |
| D2-RAT-G7AnalogPulseOnce-then-DecayRecovery | 0 | -0.3296567334069146 |  |
| D3-RAT-NoRegressionCheck | 0 | -0.5554063849978976 |  |
| DCTRL-RandomPulseSameRecovery | 0 | -0.29534469710456 |  |

Line F all-basis family summary：

| family | rows | pass | best candidate | mean delta | gate |
| --- | --- | --- | --- | --- | --- |
| D-FOU | 54 | 4 | D-FOU98-BandwiseSNRWarmupV8 | -0.0787037037037037 | 0 |
| D-RBF | 54 | 1 | D-RBF95-ActiveCenterOccupancyV8 | -0.18344907407407407 | 0 |
| D-WAV | 45 | 1 | D-WAV83-SupportOverlapDampingV8 | -0.1 | 0 |

Line M attribution summary：

| mechanism | KAN method | best MLP | KAN source | MLP source | delta | KAN adv |
| --- | --- | --- | --- | --- | --- | --- |
| M1-ProductivePulseRecovery | A-M1a-D-CHE-PulseOnce-AdamWRecovery | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.029937916331821017 | -0.19552964634365505 | 0.22546756267547607 | 0 |
| M1-ProductivePulseRecovery | A-M1b-D-CHE-PulseOnce-LRCooldownRecovery | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.10326735178629558 | -0.19552964634365505 | 0.29879699812995064 | 0 |
| M1-ProductivePulseRecovery | A-M1c-D-CHE-PulseOnce-MomentumDampingRecovery | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.03025874826643202 | -0.19552964634365505 | 0.22578839461008707 | 0 |
| M1-ProductivePulseRecovery | A-M1d-D-CHE-PulseOnce-DecoupledGlobalDecayRecovery | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.031272292137145996 | -0.19552964634365505 | 0.22680193848080105 | 0 |
| M1-ProductivePulseRecovery | A-M1e-D-CHE-PulseOnce-DegreeWiseDecayRecovery | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.030606759919060603 | -0.19552964634365505 | 0.22613640626271564 | 0 |
| M1-ProductivePulseRecovery | A-M1f-D-CHE-PulseOnce-EMAConsolidation | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.03467269738515218 | -0.19552964634365505 | 0.23020234372880724 | 0 |
| M1-ProductivePulseRecovery | A-M1g-D-CHE-PulseOnce-LookaheadConsolidation | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.03541072209676107 | -0.19552964634365505 | 0.23094036844041613 | 0 |
| M1-ProductivePulseRecovery | A-M1h-D-CHE-PulseOnce-SWAConsolidation | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.03388071060180664 | -0.19552964634365505 | 0.2294103569454617 | 0 |
| M1-ProductivePulseRecovery | A-M1i-D-CHE-PulseEvery50-AdamWRecovery | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.029937916331821017 | -0.19552964634365505 | 0.22546756267547607 | 0 |
| M1-ProductivePulseRecovery | A-M1j-D-CHE-EarlyOnlyPulse-AdamWRecovery | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.0009143087599012586 | -0.19552964634365505 | 0.1946153375837538 | 0 |
| M1-ProductivePulseRecovery | A-M1k-D-CHE-MidOnlyPulse-AdamWRecovery | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.007262865702311198 | -0.19552964634365505 | 0.18826678064134386 | 0 |
| M1-ProductivePulseRecovery | A-M1l-D-CHE-LateOnlyPulse-AdamWRecovery | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.01428683598836263 | -0.19552964634365505 | 0.1812428103552924 | 0 |
| M2-SplitConsensusSignalSubspace | A-M2a-D-CHE-SplitConsensusDiagMetric | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.003998716672261556 | -0.19552964634365505 | 0.19153092967139349 | 0 |
| M2-SplitConsensusSignalSubspace | A-M2b-D-CHE-SplitConsensusRoleBlockMetric | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.0369714101155599 | -0.19552964634365505 | 0.23250105645921496 | 0 |
| M2-SplitConsensusSignalSubspace | A-M2c-D-CHE-SplitConsensusLowRank-r4 | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.003998716672261556 | -0.19552964634365505 | 0.19153092967139349 | 0 |
| M2-SplitConsensusSignalSubspace | A-M2d-D-CHE-SplitConsensusLowRank-r8 | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.003998716672261556 | -0.19552964634365505 | 0.19153092967139349 | 0 |
| M2-SplitConsensusSignalSubspace | A-M2e-D-CHE-SplitConsensusMetricOnlyNoProjection | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.003998716672261556 | -0.19552964634365505 | 0.19153092967139349 | 0 |
| M2-SplitConsensusSignalSubspace | A-M2f-D-CHE-SplitConsensusProjectionPlusAdamV | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.003998716672261556 | -0.19552964634365505 | 0.19153092967139349 | 0 |
| M2-SplitConsensusSignalSubspace | A-M2g-D-CHE-NegativeEigenQuarantineDiagnostic | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.003998716672261556 | -0.19552964634365505 | 0.19153092967139349 | 0 |
| M3-PopRiskDriftSNR | A-M3a-D-CHE-ParameterSNRPreconditioner | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.003998716672261556 | -0.19552964634365505 | 0.19153092967139349 | 0 |
| M3-PopRiskDriftSNR | A-M3b-D-CHE-DegreeRoleSNRPreconditioner | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.0369714101155599 | -0.19552964634365505 | 0.23250105645921496 | 0 |
| M3-PopRiskDriftSNR | A-M3c-D-CHE-BasisChannelSNRCoverBoundary | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.0017551978429158528 | -0.19552964634365505 | 0.1937744485007392 | 0 |
| M3-PopRiskDriftSNR | A-M3d-D-CHE-SNRPulseScheduler | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.003998716672261556 | -0.19552964634365505 | 0.19153092967139349 | 0 |
| M3-PopRiskDriftSNR | A-M3e-D-CHE-SNRRecoveryScheduler | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.017210377587212458 | -0.19552964634365505 | 0.1783192687564426 | 0 |
| M3-PopRiskDriftSNR | A-M3f-D-CHE-SNRReservoirGuard | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.0028872622383965384 | -0.19552964634365505 | 0.19264238410525852 | 0 |
| M4-FunctionSpaceProximal | A-M4a-D-CHE-AdamSubspaceProximal-alpha000 | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.12557220458984375 | -0.19552964634365505 | 0.3211018509334988 | 0 |
| M4-FunctionSpaceProximal | A-M4b-D-CHE-PerExampleLowRankProximal-alpha025 | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.11325258678860134 | -0.19552964634365505 | 0.3087822331322564 | 0 |
| M4-FunctionSpaceProximal | A-M4c-D-CHE-OutputJacobianSketchProximal-alpha050 | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.09383397632175022 | -0.19552964634365505 | 0.2893636226654053 | 0 |
| M4-FunctionSpaceProximal | A-M4d-D-CHE-SplitB1B2TransferProximal-alpha100 | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.057322144508361816 | -0.19552964634365505 | 0.25285179085201687 | 0 |
| M4-FunctionSpaceProximal | A-M4e-D-CHE-RecoveryAwareProximal-alpha200 | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.011784791946411133 | -0.19552964634365505 | 0.20731443829006618 | 0 |
| M4-FunctionSpaceProximal | A-M4f-D-CHE-ControlResidualizedProximal-alpha100 | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.09383397632175022 | -0.19552964634365505 | 0.2893636226654053 | 0 |
| M5-OptimizerAwareAlignedFU | A-M5a-D-CHE-CautiousFU | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.03044989373948839 | -0.19552964634365505 | 0.22597954008314344 | 0 |
| M5-OptimizerAwareAlignedFU | A-M5b-D-CHE-SoftCautiousFU | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.030435019069247775 | -0.19552964634365505 | 0.22596466541290283 | 0 |
| M5-OptimizerAwareAlignedFU | A-M5c-D-CHE-MGUPStyleReweightFU | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.030962361229790583 | -0.19552964634365505 | 0.22649200757344562 | 0 |
| M5-OptimizerAwareAlignedFU | A-M5d-D-CHE-AdamSecondMomentScaledFU | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -1.2830092642042372 | -0.19552964634365505 | -1.0874796178605821 | 0 |
| M5-OptimizerAwareAlignedFU | A-M5e-D-CHE-SophiaDiagLiteClippedFU | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -1.5059364371829562 | -0.19552964634365505 | -1.3104067908393011 | 0 |
| M5-OptimizerAwareAlignedFU | A-M5f-D-CHE-BlockSecondMomentFU | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -1.2447997993893094 | -0.19552964634365505 | -1.0492701530456543 | 0 |
| M5-OptimizerAwareAlignedFU | A-M5g-D-CHE-DecoupledDecayPlusFU | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.031272292137145996 | -0.19552964634365505 | 0.22680193848080105 | 0 |
| M6-CarrierSpecificReparamFU | A-M6a-D-CHE-LowDegreeSignalHighDegreeReservoir | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.029937916331821017 | -0.19552964634365505 | 0.22546756267547607 | 0 |
| M6-CarrierSpecificReparamFU | A-M6b-D-CHE-ReadoutBasisDecoupledCarrier | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.033004429605272084 | -0.19552964634365505 | 0.22853407594892713 | 0 |
| M6-CarrierSpecificReparamFU | A-M6c-D-CHE-OrthogonalDegreeBankCarrier | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.030606759919060603 | -0.19552964634365505 | 0.22613640626271564 | 0 |
| M6-CarrierSpecificReparamFU | A-M6d-D-CHE-OutputJacobianCarrier | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.003998716672261556 | -0.19552964634365505 | 0.19153092967139349 | 0 |
| M6-CarrierSpecificReparamFU | A-M6e-D-CHE-DualBankSignalReservoirCarrier | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.032063033845689565 | -0.19552964634365505 | 0.22759268018934461 | 0 |
| M6-CarrierSpecificReparamFU | A-M6f-D-CHE-HighDegreeQuarantineCarrier | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.030939870410495333 | -0.19552964634365505 | 0.2264695167541504 | 0 |
| M7-LateAttachSnapshotFU | A-M7a-D-CHE-LateAttachEarlyCheckpoint | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.0009143087599012586 | -0.19552964634365505 | 0.1946153375837538 | 0 |
| M7-LateAttachSnapshotFU | A-M7b-D-CHE-LateAttachMidCheckpoint | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.007262865702311198 | -0.19552964634365505 | 0.18826678064134386 | 0 |
| M7-LateAttachSnapshotFU | A-M7c-D-CHE-LateAttachLateCheckpoint | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.01428683598836263 | -0.19552964634365505 | 0.1812428103552924 | 0 |
| M7-LateAttachSnapshotFU | A-M7d-D-CHE-LateAttachAfterLossPlateau | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.01924288272857666 | -0.19552964634365505 | 0.1762867636150784 | 0 |
| M7-LateAttachSnapshotFU | A-M7e-D-CHE-LateAttachHighSourceLowDebtWindow | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.09098859628041585 | -0.19552964634365505 | 0.2865182426240709 | 0 |
| M8-RecoveryOnlyDeconfound | A-M8a-D-CHE-AdamWRecoveryOnly | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.013537115520901151 | -0.19552964634365505 | 0.1819925308227539 | 0 |
| M8-RecoveryOnlyDeconfound | A-M8b-D-CHE-LRCooldownOnly | B-M4a-MLP-AdamSubspaceProximal-alpha000 | 0.09240689542558458 | -0.19552964634365505 | 0.28793654176923966 | 0 |
| M8-RecoveryOnlyDeconfound | A-M8c-D-CHE-DecayRecoveryOnly | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.535791900422838 | -0.19552964634365505 | -0.3402622540791829 | 0 |
| M8-RecoveryOnlyDeconfound | A-M8d-D-CHE-EMASWARecoveryOnly | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.016842696401807997 | -0.19552964634365505 | 0.17868694994184706 | 0 |
| M8-RecoveryOnlyDeconfound | A-M8e-D-CHE-LookaheadRecoveryOnly | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.014865186479356553 | -0.19552964634365505 | 0.1806644598642985 | 0 |
| M8-RecoveryOnlyDeconfound | A-M8f-D-CHE-MomentumDampingOnly | B-M4a-MLP-AdamSubspaceProximal-alpha000 | -0.013962454266018338 | -0.19552964634365505 | 0.18156719207763672 | 0 |

Failure taxonomy summary：

| failure class | rows |
| --- | --- |
| source,tail_debt,linec_debt,control_equivalent | 605 |
| retention,tail_debt,linec_debt | 116 |
| source,retention,tail_debt,linec_debt,control_equivalent | 112 |
| source,retention,tail_debt,control_equivalent | 84 |
| tail_debt,linec_debt | 77 |
| source,tail_debt,control_equivalent | 50 |
| source,retention,linec_debt,control_equivalent | 44 |
| retention,tail_debt | 43 |
| tail_debt | 20 |
| source,retention,control_equivalent | 14 |
| retention,linec_debt | 5 |
| AllBasisSubstrateBlocked | 1 |
| R-C-LQReanchorStillBlocked | 1 |
| RationalMonitorNoPromotion | 1 |

## 6. Final route / no-go

```text
route = R6-FunctionalDynamicsAllMechanismsNoGo
minimum_success = S1-CarrierMechanismMatrixCoverageCompleted
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
| LineA-DCHENoGo | 1 | best=A-M4a-D-CHE-AdamSubspaceProximal-alpha000;source_h800=0.12557220458984375;retention=0.7634300026628706;tail=0.0;LineC=0.2222222222222222 |
| LineB-MLPGenericNoPromotion | 1 | best=B-M4a-MLP-AdamSubspaceProximal-alpha000;source_h800=-0.19552964634365505;retention=0.8310222559505038;tail=0.0;LineC=0.0 |
| LineC-LQReanchor | 1 | best=C2-LQ-ProtocolMatchedReanchor;delta=0.28125;near=8/9;route=R-C-LQReanchorStillBlocked |
| LineD-RationalMonitorOnly | 1 | best=D1-RAT-G7AnalogPulseOnce-then-AdamWRecovery;pass=2;no reset/controller |
| LineF-AllBasisSubstrate | 1 | best_family=D-FOU;pass=4/9 |
| R6-FunctionalDynamicsAllMechanismsNoGo | 1 | A=0;B=0;S2=0;S3=0;C=0;D=0;F=0 |
| NoForbiddenContinuation | 1 | no G9/G10/action bank/controller/reset/audit-directed branch |

Next hypothesis queue：

| priority | hypothesis | allowed next step |
| --- | --- | --- |
| 1 | MechanismSpecificRecoveryDefinitionRevision | Only in a new pre-registered plan; use v16.2 matrix as theory evidence, not as controller. |
| 2 | MLPGenericDynamicsMechanismFollowup | Study MLP weak dynamics as generic dynamics, not KAN-specific promotion. |
| 3 | CarrierSubstrateBeforeFU | Repair LQ/all-basis carrier before official functional update proof. |

## 7. 科学结论

```text
1. v16.2 已执行 carrier × mechanism × horizon × controls matrix，并生成 required artifacts。
2. A/B 均完成 H=800 full surface；top-2 已执行 H=1600 extension。
3. LineC/CEp99/NLL/ECE/AUCtime/Brier 只用于 audit/gate/debt readback，没有反推方向。
4. MLP generic positive 不写成 KAN-specific promotion；Line M attribution 是强制证据链。
5. 当前 route = R6-FunctionalDynamicsAllMechanismsNoGo，promotion_allowed = 0。
```

## 8. 用户再次追问后的计划闭环复核

本节不新增训练，不改变任何指标；它记录再次按 `DG-KAN_v16.2_MultiSchemeFunctionalDynamics_4GPU_完整计划.md` 复核后的判断。

复核读回：

```text
route = R6-FunctionalDynamicsAllMechanismsNoGo
minimum_success = S1-CarrierMechanismMatrixCoverageCompleted
S2_weak_productive_dynamics_reached = 0
S3_productive_debt_recovery_reached = 0
official_s5_reached = 0
promotion_allowed = 0
required_artifact_manifest_rows = 54
required_artifact_missing_rows = 0
execution_contract_nonpass_rows = 0
deep_coverage_nonpass_rows = 0
budget_certificate_rows = 5
deferred_items_rows = 5
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

计划第 13/15 节复核结论：

```text
No S5, no promotion. No exceptions.
If all lines complete and no S2 -> R6-FunctionalDynamicsAllMechanismsNoGo.
If v16.2 still finds no S2/S3 under this matrix, current train-stream functional dynamics family should be formally no-go.
后续必须转入 substrate/base-level redesign 或新的 theory of functional direction。
```

因此，v16.2 的覆盖/合同目标已经达成，但 promotion/scientific success 没有达成。计划内没有剩余可继续执行的补救分支；继续在本版新增 action token、controller、action bank、reset route 或 audit-directed search 会违反 plan hard stop 与 no-action audit。

关键证据链：

```text
1. 最强 h800 row = A-M4a-D-CHE-AdamSubspaceProximal-alpha000。
   source_vs_best_control_h800 = 0.12557220458984375
   source_retention_h800 = 0.7634300026628706
   tail_recovery_rate_h800 = 0.0
   LineC_recovery_rate_h800 = 0.2222222222222222
   AUCtime_ratio_h800 = 1.0299587646966346
   判断：source/retention/AUC 不差，但 tail=0 且 LineC<0.40，因此 S2/S3 均不能打开。
2. h1600 top-2 没有把 h800 source 巩固成 promotion 证据。
   A-M4a-D-CHE-AdamSubspaceProximal-alpha000: mean source=0.0, bad=1.0, tail=0.017574313511256316, LineC=0.4444444444444444
   A-M4b-D-CHE-PerExampleLowRankProximal-alpha025: mean source=-0.02831966347164578, bad=1.0, tail=0.03057429645997782, LineC=0.4444444444444444
   B-M4a-MLP-AdamSubspaceProximal-alpha000: mean source=-0.002453327178955078, bad=1.0, tail=0.0, LineC=0.0
   B-M4b-MLP-PerExampleLowRankProximal-alpha025: mean source=-0.12256171968248156, bad=1.0, tail=0.0, LineC=0.0
3. Line M attribution rows = 55，KAN_specific_advantage_rows = 0，不能把局部 D-CHE source 写成 KAN-specific promotion。
4. Line C LQ reanchor 仍 blocked：best=C2-LQ-ProtocolMatchedReanchor, delta=0.28125, near=8/9, route=R-C-LQReanchorStillBlocked。
5. Line D Rational monitor 仍不能重启 reset/controller：best=D1-RAT-G7AnalogPulseOnce-then-AdamWRecovery, pass=2, no reset/controller。
6. Line F all-basis substrate 仍不能逃逸：best_family=D-FOU, pass=4/9, gate=0；只能作为后续 carrier/substrate repair 证据。
```

最终判断：

```text
1. 不是没继续，而是 v16.2 计划内可继续分支已经闭合。
2. 当前 no-go 是 matrix 级 no-go：A/B/C/D/F gates 全部为 0，S2/S3/S5 全部为 0。
3. 本版不能再追加 token/action/controller/reset；下一步需要新的预注册计划，而不是在 v16.2 内改写边界。
```
