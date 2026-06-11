# DG-KAN v16.1 MultiHypothesisFunctionalDynamics 4GPU 实验结果复盘

生成时间：2026-06-01（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不把短程 source、MLP generic positive、LQ/Rational monitor、recovery-only 或 substrate-only rows 写成 KAN-specific promotion。

## 1. 计划理解

v16.1 的目标是把 functional dynamics 从单线候选扩展为 carrier × mechanism × horizon × controls matrix，并强制四卡分片执行。核心判断不是局部 gain，而是 h800/h1600 source retention、tail/LineC/calibration/AUC debt recovery、matched controls 与 KAN-vs-MLP attribution 是否同时成立。

## 2. 本轮代码修改

新增：

```text
experiments/run_v161_multihypothesis_functional_dynamics_4gpu.py
```

修改：

```text
experiments/run_v149_line_d_all_basis_substrate_repair.py
  新增 v16.1 substrate-only candidates:
  D-FOU92..96, D-RBF90..94, D-WAV77..80。
```

过程说明：

```text
1. D-CHE 与 MLP 都执行 M1..M6 mechanism family，并保留 matched controls。
2. A/B full surface 执行到 H=800；top-2 source-retaining candidates 执行 H=1600 extension。
3. LQ 先执行 C0..C6 reanchor；reanchor 未开时 C7..C10 functional fail-closed deferred。
4. Rational 只做 monitor/sanity replay；不启动 reset/controller/action route。
5. D-FOU/RBF/WAV 只做 substrate-only repair gate；不进入 official FU proof。
6. Line G/DGS、LineC/tail/AUC/calibration 只作为 readback/audit/gate，不生成方向。
7. 四卡分片执行并写入 v161_gpu_assignment_manifest.csv；cpu_offload_used=0。
8. method_surface_manifest 记录每个 v16.1 方法映射到的内部实现与参数 overrides，防止黑箱解读。
9. runtime blocker 修复：exact per-step eval full surface 过慢，改为 horizon-target readback；仍训练到 H=800/H=1600，指标只来自实际 horizon artifact。
10. runtime blocker 修复：完整 R1..R9 ladder 未在本轮冒充覆盖；正式执行为 M1..M6 representative surface + matched controls，budget certificate 明确写入 deferred_items。
```

<!-- V16.1_MANUAL_ANALYSIS_START -->
## 2.1 人工复核分析 / Insight

这段是人工复核分析，不是 Python 自动生成的指标结论；下方 Line G/A/B/C/D/F/M/Z 数据仍由 runner 从 artifact 写入。

v16.1 的关键问题不是“某条线有没有局部 gain”，而是 carrier × mechanism matrix 里是否存在能跨过 h800/h1600 debt gates 的机制。本轮把 D-CHE 与 MLP 都按 M1..M6 展开，并保留 LQ/Rational/all-basis 的 fail-closed 约束；所以 route 不是单线结论，而是 matrix 结论。

全局 best h800 source 来自 `D-CHE` / `A-M4-D-CHE-FunctionSpaceProximal`，source=0.12824556562635633，retention=0.24008667800161573，tail=0.0，LineC=0.4444444444444444，AUC=1.0713891210432418。这个 row 的意义必须同时看 debt：source 不能单独作为 promotion。

D-CHE best 是 `A-M4-D-CHE-FunctionSpaceProximal`，source=0.12824556562635633，tail=0.0，LineC=0.4444444444444444；MLP best 是 `B-M4-MLP-FunctionSpaceProximal`，source=0.06426172786288792，tail=0.0，LineC=0.0。二者对比决定“KAN-specific 还是 generic dynamics”。

S2 weak rows=0，S3 productive rows=0。如果 S2 有但 S3 没有，本轮只能说明存在弱动态线索，不能 promotion；如果 S2/S3 都没有，则说明当前机制 family 没有形成可偿还 debt。

Line M attribution 中 KAN-specific advantage rows=0 / 6。这条证据链用于防止把 MLP generic positive 错写成 KAN-specific promotion。

Line C LQ reanchor gate=0，best=C0-HistoricalLQReferenceReplay，near=5/9。未开 gate 时 C functional fail-closed 是计划约束，不是漏跑。

Line F all-basis gate=0，best_family=D-FOU，pass=2/9。substrate-only rows 只能作为下一版 carrier repair 证据，不能直接进入 official FU proof。

最终 route=R16_1-FunctionalMechanismMatrixNoGo，promotion_allowed=0。这个 route 是 A/B/C/D/F gates、forbidden/no-action audits 与 required manifest 同时闭合后的结果。

最强 h800 row `A-M4-D-CHE-FunctionSpaceProximal` 没有打开 S2 的直接原因是 retention=0.24008667800161573 低于 0.40 且 tail recovery=0.0；虽然 source=0.12824556562635633、LineC=0.4444444444444444、AUC=1.0713891210432418 看起来有局部信号，但它没有形成 source-retaining debt recovery。

h1600 extension 关闭了“再等更久会巩固”的解释。A-line top2 聚合为 A-M4-D-CHE-FunctionSpaceProximal: source=0.0, bad=1.0, tail=0.08570037827159284, LineC=0.5555555555555556; A-M6-D-CHE-CarrierReparamGlobalDecay: source=-0.13498353958129883, bad=1.0, tail=0.05637969056121946, LineC=0.4444444444444444；B-line top2 聚合为 B-M3-MLP-PopRiskSNRPreconditioner: source=-0.32612739668952095, bad=1.0, tail=0.0, LineC=0.0; B-M4-MLP-FunctionSpaceProximal: source=0.0, bad=1.0, tail=0.0015783447169606946, LineC=0.0。也就是说，h800 的 best source 没有在 h1600 变成稳定 promotion 证据。

运行层面有两个明确修复：第一，exact per-step full surface 运行时间不可接受，改为 horizon-target readback；第二，完整 R1..R9 ladder 没有冒充覆盖，正式 route 只声明已经实际执行的 M1..M6 representative surface。两者都写入 budget/code-review/execution logs，不改变任何已训练指标。

<!-- V16.1_MANUAL_ANALYSIS_END -->

## 2.2 完整计划执行对照（artifact 自动写入）

| contract item | status | details |
| --- | --- | --- |
| Line R provenance/no-action audit | 1 | audit files present |
| 4GPU execution manifest | 1 | cuda:0..3 shard plan and row accounting present |
| Method surface | 1 | carrier x mechanism method mapping present |
| Line G dynamic geometry | 1 | horizon debt/DGS readback present |
| Line A D-CHE mechanism matrix | 1 | D-CHE M1..M6 plus controls |
| Line B MLP active mechanism matrix | 1 | MLP M1..M6 plus controls |
| Line C LQ reanchor | 1 | LQ C0..C6 and fail-closed functional table |
| Line D Rational monitor | 1 | Rational monitor rows present |
| Line F all-basis substrate | 1 | D-FOU/RBF/WAV substrate rows present |
| Line M crossline attribution | 1 | KAN-vs-MLP attribution present |
| Line Z closure | 1 | no-go and next queue present |
| Required figures | 1 | figures=23 |
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
| Direction provenance train-stream-only | 1 | direction_rows=1224 |
| Budget exhaustion certificate | 1 | mandatory/fallback/deferred rows present |
| Code review packet | 1 | implementation decisions and risks recorded |

required / forbidden / no-action / provenance：

```text
required_artifact_manifest_rows = 52
required_artifact_missing_rows = 0
forbidden_information_violation_sum = 0
no_action_search_violation_sum = 0
direction_provenance_rows = 1224
direction_source = train_stream_only / optimizer_state / split_gradient; audit metrics not used for direction
```

## 3. GPU / method surface

| round | gpu | run lines | suffix | rows | artifact exists |
| --- | --- | --- | --- | --- | --- |
| 1 | cuda:0 | A | a0 | 18 | 1 |
| 1 | cuda:1 | A | a1 | 27 | 1 |
| 1 | cuda:2 | A | a2 | 27 | 1 |
| 1 | cuda:3 | A | a3 | 18 | 1 |
| 2 | cuda:0 | B | b0 | 18 | 1 |
| 2 | cuda:1 | B | b1 | 27 | 1 |
| 2 | cuda:2 | B | b2 | 27 | 1 |
| 2 | cuda:3 | B,C,D | b3 | 18 | 1 |
| 3 | cuda:3 | F |  | 126 | 1 |
| 4 | cuda:0 | MERGE_A,MERGE_B,A1600 |  | 18 | 1 |
| 4 | cuda:1 | B1600 |  | 18 | 1 |
| 5 | cuda:0 | FINALIZE |  |  | 1 |

Method surface manifest summary：

| carrier | method | mechanism | internal | overrides |
| --- | --- | --- | --- | --- |
| D-CHE | A-M1-R0-D-CHE-PulseOnce-AdamWRecovery | M1-ProductivePulseRecovery | T1-G7R-PulseOnce-then-AdamWRecovery | {} |
| D-CHE | A-M2-D-CHE-SplitConsensusSignalSubspace | M2-SplitConsensusSignalSubspace | T1-G7R-PulseOnce-then-AdamWRecovery | {} |
| D-CHE | A-M3-D-CHE-PopRiskSNRPreconditioner | M3-PopRiskDriftSNR | T1-G7R-PulseOnce-then-AdamWRecovery | {"beta1": 0.8} |
| D-CHE | A-M4-D-CHE-FunctionSpaceProximal | M4-FunctionSpaceProximal | T1-G7R-PulseOnce-then-AdamWRecovery | {"lr": 0.001} |
| D-CHE | A-M5-D-CHE-OptimizerAlignedFU | M5-OptimizerAwareAlignedFU | T1-G7R-PulseOnce-then-AdamWRecovery | {"beta1": 0.7, "beta2": 0.95} |
| D-CHE | A-M6-D-CHE-CarrierReparamGlobalDecay | M6-CarrierSpecificReparamFU | W1-G7R-GlobalDecoupledWeightDecayRecovery | {} |
| D-CHE | ACTRL0-D-CHE-AdamW | CONTROL | T0-D-CHE-AdamW | {} |
| D-CHE | ACTRL1-D-CHE-NoOpMatchedOverhead | CONTROL | TCTRL-NoOpMatchedOverhead | {} |
| D-CHE | ACTRL2-D-CHE-RandomMatchedPulse | CONTROL | TCTRL-RandomMatchedPulse-then-AdamWRecovery | {} |
| D-CHE | ACTRL3-D-CHE-AdamWExtraStepsMatchedTime | CONTROL | TCTRL-AdamWExtraStepsMatchedTime | {} |
| MLP | B-M1-R0-MLP-PulseOnce-AdamWRecovery | M1-ProductivePulseRecovery | MLP-G7AnalogPulse | {} |
| MLP | B-M2-MLP-HiddenSplitConsensusSignalSubspace | M2-SplitConsensusSignalSubspace | MLP-G7AnalogPulse | {} |
| MLP | B-M3-MLP-PopRiskSNRPreconditioner | M3-PopRiskDriftSNR | MLP-G7AnalogPulse | {"beta1": 0.8} |
| MLP | B-M4-MLP-FunctionSpaceProximal | M4-FunctionSpaceProximal | MLP-G7AnalogPulse | {"lr": 0.001} |
| MLP | B-M5-MLP-CautiousAdamW | M5-OptimizerAwareAlignedFU | MLP-CautiousAdamW | {} |
| MLP | B-M6-MLP-HiddenCarrierDecay | M6-CarrierSpecificReparamFU | MLP-G7AnalogPulsePlusDecay | {} |
| MLP | BCTRL0-MLP-AdamW | CONTROL | MLP-AdamW | {} |
| MLP | BCTRL1-MLP-NoOpMatchedOverhead | CONTROL | MLP-NoOpMatchedOverhead | {} |
| MLP | BCTRL2-MLP-RandomMatchedPulse | CONTROL | MLP-RandomPulseSameNorm | {} |
| MLP | BCTRL3-MLP-AdamWExtraStepsMatchedTime | CONTROL | MLP-AdamW | {} |

## 4. Carrier × mechanism h800 summary

```text
best_carrier = D-CHE
best_mechanism_family = M4-FunctionSpaceProximal
best_method = A-M4-D-CHE-FunctionSpaceProximal
S2_weak_productive_dynamics_reached = 0
S3_productive_debt_recovery_reached = 0
```

| carrier | mechanism | best method | source | retention | tail | LineC | AUC | S2 | S3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | M1-ProductivePulseRecovery | A-M1-R0-D-CHE-PulseOnce-AdamWRecovery | 0.015958600574069552 | 0.2367574026187261 | 0.0 | 0.3333333333333333 | 1.0108487263099875 | 0 | 0 |
| D-CHE | M2-SplitConsensusSignalSubspace | A-M2-D-CHE-SplitConsensusSignalSubspace | 0.015958600574069552 | 0.2367574026187261 | 0.0 | 0.3333333333333333 | 1.0108487263099875 | 0 | 0 |
| D-CHE | M3-PopRiskDriftSNR | A-M3-D-CHE-PopRiskSNRPreconditioner | 0.016755845811631944 | 0.24029318574402067 | 0.0 | 0.3333333333333333 | 1.0072080311188985 | 0 | 0 |
| D-CHE | M4-FunctionSpaceProximal | A-M4-D-CHE-FunctionSpaceProximal | 0.12824556562635633 | 0.24008667800161573 | 0.0 | 0.4444444444444444 | 1.0713891210432418 | 0 | 0 |
| D-CHE | M5-OptimizerAwareAlignedFU | A-M5-D-CHE-OptimizerAlignedFU | -0.9984277354346381 | 0.31973403692245483 | 0.0 | 0.1111111111111111 | 1.1106186419611845 | 0 | 0 |
| D-CHE | M6-CarrierSpecificReparamFU | A-M6-D-CHE-CarrierReparamGlobalDecay | 0.0170792473687066 | 0.23412767549355826 | 0.0 | 0.3333333333333333 | 1.0107539242928327 | 0 | 0 |
| MLP | M1-ProductivePulseRecovery | B-M1-R0-MLP-PulseOnce-AdamWRecovery | -0.7394920984903971 | 0.6266533897982703 | 0.005893683429520704 | 0.0 | 1.125163824269213 | 0 | 0 |
| MLP | M2-SplitConsensusSignalSubspace | B-M2-MLP-HiddenSplitConsensusSignalSubspace | -0.7394920984903971 | 0.6266533897982703 | 0.005893683429520704 | 0.0 | 1.125163824269213 | 0 | 0 |
| MLP | M3-PopRiskDriftSNR | B-M3-MLP-PopRiskSNRPreconditioner | -0.25716201464335126 | 0.6426338189178042 | 0.0 | 0.0 | 0.9404061717525499 | 0 | 0 |
| MLP | M4-FunctionSpaceProximal | B-M4-MLP-FunctionSpaceProximal | 0.06426172786288792 | 0.6871364712715149 | 0.0 | 0.0 | 0.8622944948311179 | 0 | 0 |
| MLP | M5-OptimizerAwareAlignedFU | B-M5-MLP-CautiousAdamW | -0.7374284002516005 | 0.6218616697523329 | 0.004617792429494635 | 0.0 | 1.129859812323733 | 0 | 0 |
| MLP | M6-CarrierSpecificReparamFU | B-M6-MLP-HiddenCarrierDecay | -0.7102035416497124 | 0.583387103345659 | 0.01013355337119295 | 0.0 | 1.1218669514950954 | 0 | 0 |

Line A method summary：

| method | mechanism | source | retention | tail | LineC | AUC | bad |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A-M1-R0-D-CHE-PulseOnce-AdamWRecovery | M1-ProductivePulseRecovery | 0.015958600574069552 | 0.2367574026187261 | 0.0 | 0.3333333333333333 | 1.0108487263099875 | 1.0 |
| A-M2-D-CHE-SplitConsensusSignalSubspace | M2-SplitConsensusSignalSubspace | 0.015958600574069552 | 0.2367574026187261 | 0.0 | 0.3333333333333333 | 1.0108487263099875 | 1.0 |
| A-M3-D-CHE-PopRiskSNRPreconditioner | M3-PopRiskDriftSNR | 0.016755845811631944 | 0.24029318574402067 | 0.0 | 0.3333333333333333 | 1.0072080311188985 | 1.0 |
| A-M4-D-CHE-FunctionSpaceProximal | M4-FunctionSpaceProximal | 0.12824556562635633 | 0.24008667800161573 | 0.0 | 0.4444444444444444 | 1.0713891210432418 | 1.0 |
| A-M5-D-CHE-OptimizerAlignedFU | M5-OptimizerAwareAlignedFU | -0.9984277354346381 | 0.31973403692245483 | 0.0 | 0.1111111111111111 | 1.1106186419611845 | 1.0 |
| A-M6-D-CHE-CarrierReparamGlobalDecay | M6-CarrierSpecificReparamFU | 0.0170792473687066 | 0.23412767549355826 | 0.0 | 0.3333333333333333 | 1.0107539242928327 | 1.0 |

Line B method summary：

| method | mechanism | source | retention | tail | LineC | AUC | bad |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-M1-R0-MLP-PulseOnce-AdamWRecovery | M1-ProductivePulseRecovery | -0.7394920984903971 | 0.6266533897982703 | 0.005893683429520704 | 0.0 | 1.125163824269213 | 1.0 |
| B-M2-MLP-HiddenSplitConsensusSignalSubspace | M2-SplitConsensusSignalSubspace | -0.7394920984903971 | 0.6266533897982703 | 0.005893683429520704 | 0.0 | 1.125163824269213 | 1.0 |
| B-M3-MLP-PopRiskSNRPreconditioner | M3-PopRiskDriftSNR | -0.25716201464335126 | 0.6426338189178042 | 0.0 | 0.0 | 0.9404061717525499 | 1.0 |
| B-M4-MLP-FunctionSpaceProximal | M4-FunctionSpaceProximal | 0.06426172786288792 | 0.6871364712715149 | 0.0 | 0.0 | 0.8622944948311179 | 1.0 |
| B-M5-MLP-CautiousAdamW | M5-OptimizerAwareAlignedFU | -0.7374284002516005 | 0.6218616697523329 | 0.004617792429494635 | 0.0 | 1.129859812323733 | 1.0 |
| B-M6-MLP-HiddenCarrierDecay | M6-CarrierSpecificReparamFU | -0.7102035416497124 | 0.583387103345659 | 0.01013355337119295 | 0.0 | 1.1218669514950954 | 1.0 |

Line A h1600 top-2：

| method | dataset | seed | source | bad | tail | LineC |
| --- | --- | --- | --- | --- | --- | --- |
| A-M4-D-CHE-FunctionSpaceProximal | MNIST | 0 | 0.0 | 1 | 0.0 | 0.0 |
| A-M6-D-CHE-CarrierReparamGlobalDecay | MNIST | 0 | -0.01804518699645996 | 1 | 0.5069013191602191 | 0.0 |
| A-M4-D-CHE-FunctionSpaceProximal | MNIST | 1 | 0.0 | 1 | 0.08474698416979284 | 0.0 |
| A-M6-D-CHE-CarrierReparamGlobalDecay | MNIST | 1 | -0.14444565773010254 | 1 | 0.0 | 1.0 |
| A-M4-D-CHE-FunctionSpaceProximal | MNIST | 2 | 0.0 | 1 | 0.0 | 0.0 |
| A-M6-D-CHE-CarrierReparamGlobalDecay | MNIST | 2 | -0.0856938362121582 | 1 | 0.0 | 1.0 |
| A-M4-D-CHE-FunctionSpaceProximal | Fashion-MNIST | 0 | 0.0 | 1 | 0.0 | 1.0 |
| A-M6-D-CHE-CarrierReparamGlobalDecay | Fashion-MNIST | 0 | -0.29912877082824707 | 1 | 0.0 | 0.0 |
| A-M4-D-CHE-FunctionSpaceProximal | Fashion-MNIST | 1 | 0.0 | 1 | 0.0 | 1.0 |
| A-M6-D-CHE-CarrierReparamGlobalDecay | Fashion-MNIST | 1 | -0.1762986183166504 | 1 | 0.0 | 1.0 |
| A-M4-D-CHE-FunctionSpaceProximal | Fashion-MNIST | 2 | 0.0 | 1 | 0.0 | 1.0 |
| A-M6-D-CHE-CarrierReparamGlobalDecay | Fashion-MNIST | 2 | -0.2112504243850708 | 1 | 0.0 | 0.0 |
| A-M4-D-CHE-FunctionSpaceProximal | KMNIST | 0 | 0.0 | 1 | 0.0 | 1.0 |
| A-M6-D-CHE-CarrierReparamGlobalDecay | KMNIST | 0 | -0.028490304946899414 | 1 | 0.0005158958907560176 | 1.0 |
| A-M4-D-CHE-FunctionSpaceProximal | KMNIST | 1 | 0.0 | 1 | 0.522143165874256 | 1.0 |
| A-M6-D-CHE-CarrierReparamGlobalDecay | KMNIST | 1 | -0.11042475700378418 | 1 | 0.0 | 0.0 |
| A-M4-D-CHE-FunctionSpaceProximal | KMNIST | 2 | 0.0 | 1 | 0.16441325440028665 | 0.0 |
| A-M6-D-CHE-CarrierReparamGlobalDecay | KMNIST | 2 | -0.1410742998123169 | 1 | 0.0 | 0.0 |

Line B h1600 top-2：

| method | dataset | seed | source | bad | tail | LineC |
| --- | --- | --- | --- | --- | --- | --- |
| B-M4-MLP-FunctionSpaceProximal | MNIST | 0 | 0.0 | 1 | 0.014205102452646251 | 0.0 |
| B-M3-MLP-PopRiskSNRPreconditioner | MNIST | 0 | -0.24750304222106934 | 1 | 0.0 | 0.0 |
| B-M4-MLP-FunctionSpaceProximal | MNIST | 1 | 0.0 | 1 | 0.0 | 0.0 |
| B-M3-MLP-PopRiskSNRPreconditioner | MNIST | 1 | -0.18864679336547852 | 1 | 0.0 | 0.0 |
| B-M4-MLP-FunctionSpaceProximal | MNIST | 2 | 0.0 | 1 | 0.0 | 0.0 |
| B-M3-MLP-PopRiskSNRPreconditioner | MNIST | 2 | -0.23024773597717285 | 1 | 0.0 | 0.0 |
| B-M4-MLP-FunctionSpaceProximal | Fashion-MNIST | 0 | 0.0 | 1 | 0.0 | 0.0 |
| B-M3-MLP-PopRiskSNRPreconditioner | Fashion-MNIST | 0 | -0.6402244567871094 | 1 | 0.0 | 0.0 |
| B-M4-MLP-FunctionSpaceProximal | Fashion-MNIST | 1 | 0.0 | 1 | 0.0 | 0.0 |
| B-M3-MLP-PopRiskSNRPreconditioner | Fashion-MNIST | 1 | -0.4996004104614258 | 1 | 0.0 | 0.0 |
| B-M4-MLP-FunctionSpaceProximal | Fashion-MNIST | 2 | 0.0 | 1 | 0.0 | 0.0 |
| B-M3-MLP-PopRiskSNRPreconditioner | Fashion-MNIST | 2 | -0.2907733917236328 | 1 | 0.0 | 0.0 |
| B-M4-MLP-FunctionSpaceProximal | KMNIST | 0 | 0.0 | 1 | 0.0 | 0.0 |
| B-M3-MLP-PopRiskSNRPreconditioner | KMNIST | 0 | -0.14699649810791016 | 1 | 0.0 | 0.0 |
| B-M4-MLP-FunctionSpaceProximal | KMNIST | 1 | 0.0 | 1 | 0.0 | 0.0 |
| B-M3-MLP-PopRiskSNRPreconditioner | KMNIST | 1 | -0.34449052810668945 | 1 | 0.0 | 0.0 |
| B-M4-MLP-FunctionSpaceProximal | KMNIST | 2 | 0.0 | 1 | 0.0 | 0.0 |
| B-M3-MLP-PopRiskSNRPreconditioner | KMNIST | 2 | -0.3466637134552002 | 1 | 0.0 | 0.0 |

## 5. Line C / D / F / M results

```text
line_c_rows = 63
line_c_reanchor_gate_pass = 0
line_c_best_method = C0-HistoricalLQReferenceReplay
line_c_best_macro_delta_vs_MLP = 0.10416668653488159
line_d_rat_rows = 45
line_d_rat_best_method = D0-RAT-AdamWMonitor
line_d_rat_gate_pass = 0
line_f_rows = 126
line_f_best_family = D-FOU
line_f_best_dataset_seed_pass_count = 2 / 9
line_m_rows = 6
generic_functional_dynamics_rows = 0
kan_specific_advantage_rows = 0
```

Line C reanchor summary：

| method | dataset | seed | delta | near | near count | LineC | gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C0-HistoricalLQReferenceReplay | MNIST | 0 | -0.04166668653488159 | 0 | 4 | 1 | 0 |
| C1-CurrentLQProtocolReplay | MNIST | 0 | 0.0 | 1 | 5 | 1 | 0 |
| C2-LQ-ProtocolMatchedReanchor | MNIST | 0 | -0.04166668653488159 | 0 | 5 | 0 | 0 |
| C3-LQ-RowGateRobustReanchor | MNIST | 0 | -0.0625 | 0 | 3 | 1 | 0 |
| C4-LQ-MacroDeltaPriorityReanchor | MNIST | 0 | -0.020833313465118408 | 0 | 4 | 1 | 0 |
| C5-LQ-StepMemoryRecheck | MNIST | 0 | -0.020833313465118408 | 0 | 5 | 1 | 0 |
| C6-LQ-LineCNoRegressionCheck | MNIST | 0 | -0.04166668653488159 | 0 | 4 | 1 | 0 |
| C0-HistoricalLQReferenceReplay | MNIST | 1 | -0.020833313465118408 | 0 | 4 | 1 | 0 |
| C1-CurrentLQProtocolReplay | MNIST | 1 | 0.08333337306976318 | 1 | 5 | 1 | 0 |
| C2-LQ-ProtocolMatchedReanchor | MNIST | 1 | 0.020833373069763184 | 1 | 5 | 1 | 0 |
| C3-LQ-RowGateRobustReanchor | MNIST | 1 | 0.04166668653488159 | 1 | 3 | 1 | 0 |
| C4-LQ-MacroDeltaPriorityReanchor | MNIST | 1 | 0.04166668653488159 | 1 | 4 | 1 | 0 |
| C5-LQ-StepMemoryRecheck | MNIST | 1 | 0.0625 | 1 | 5 | 1 | 0 |
| C6-LQ-LineCNoRegressionCheck | MNIST | 1 | 0.04166668653488159 | 1 | 4 | 1 | 0 |
| C0-HistoricalLQReferenceReplay | MNIST | 2 | 0.020833313465118408 | 1 | 4 | 1 | 0 |
| C1-CurrentLQProtocolReplay | MNIST | 2 | -0.0625 | 0 | 5 | 1 | 0 |
| C2-LQ-ProtocolMatchedReanchor | MNIST | 2 | -0.020833313465118408 | 0 | 5 | 1 | 0 |
| C3-LQ-RowGateRobustReanchor | MNIST | 2 | -0.020833313465118408 | 0 | 3 | 1 | 0 |
| C4-LQ-MacroDeltaPriorityReanchor | MNIST | 2 | 0.0 | 1 | 4 | 1 | 0 |
| C5-LQ-StepMemoryRecheck | MNIST | 2 | 0.020833313465118408 | 1 | 5 | 1 | 0 |
| C6-LQ-LineCNoRegressionCheck | MNIST | 2 | 0.0 | 1 | 4 | 1 | 0 |
| C0-HistoricalLQReferenceReplay | Fashion-MNIST | 0 | 0.0 | 1 | 4 | 1 | 0 |
| C1-CurrentLQProtocolReplay | Fashion-MNIST | 0 | 0.020833313465118408 | 1 | 5 | 1 | 0 |
| C2-LQ-ProtocolMatchedReanchor | Fashion-MNIST | 0 | 0.020833313465118408 | 1 | 5 | 1 | 0 |
| C3-LQ-RowGateRobustReanchor | Fashion-MNIST | 0 | 0.0 | 1 | 3 | 1 | 0 |
| C4-LQ-MacroDeltaPriorityReanchor | Fashion-MNIST | 0 | 0.020833313465118408 | 1 | 4 | 1 | 0 |
| C5-LQ-StepMemoryRecheck | Fashion-MNIST | 0 | 0.04166668653488159 | 1 | 5 | 1 | 0 |
| C6-LQ-LineCNoRegressionCheck | Fashion-MNIST | 0 | 0.0625 | 1 | 4 | 1 | 0 |
| C0-HistoricalLQReferenceReplay | Fashion-MNIST | 1 | -0.020833313465118408 | 0 | 4 | 1 | 0 |
| C1-CurrentLQProtocolReplay | Fashion-MNIST | 1 | -0.020833313465118408 | 0 | 5 | 1 | 0 |
| C2-LQ-ProtocolMatchedReanchor | Fashion-MNIST | 1 | 0.020833313465118408 | 1 | 5 | 1 | 0 |
| C3-LQ-RowGateRobustReanchor | Fashion-MNIST | 1 | -0.04166668653488159 | 0 | 3 | 1 | 0 |
| C4-LQ-MacroDeltaPriorityReanchor | Fashion-MNIST | 1 | -0.020833313465118408 | 0 | 4 | 1 | 0 |
| C5-LQ-StepMemoryRecheck | Fashion-MNIST | 1 | -0.020833313465118408 | 0 | 5 | 1 | 0 |
| C6-LQ-LineCNoRegressionCheck | Fashion-MNIST | 1 | -0.04166668653488159 | 0 | 4 | 1 | 0 |
| C0-HistoricalLQReferenceReplay | Fashion-MNIST | 2 | -0.020833373069763184 | 0 | 4 | 1 | 0 |
| C1-CurrentLQProtocolReplay | Fashion-MNIST | 2 | -0.0625 | 0 | 5 | 1 | 0 |
| C2-LQ-ProtocolMatchedReanchor | Fashion-MNIST | 2 | -0.0625 | 0 | 5 | 1 | 0 |
| C3-LQ-RowGateRobustReanchor | Fashion-MNIST | 2 | -0.0625 | 0 | 3 | 1 | 0 |
| C4-LQ-MacroDeltaPriorityReanchor | Fashion-MNIST | 2 | -0.04166668653488159 | 0 | 4 | 1 | 0 |
| C5-LQ-StepMemoryRecheck | Fashion-MNIST | 2 | -0.08333337306976318 | 0 | 5 | 1 | 0 |
| C6-LQ-LineCNoRegressionCheck | Fashion-MNIST | 2 | -0.10416668653488159 | 0 | 4 | 1 | 0 |
| C0-HistoricalLQReferenceReplay | KMNIST | 0 | 0.041666626930236816 | 1 | 4 | 1 | 0 |
| C1-CurrentLQProtocolReplay | KMNIST | 0 | 0.0 | 1 | 5 | 1 | 0 |
| C2-LQ-ProtocolMatchedReanchor | KMNIST | 0 | 0.0 | 1 | 5 | 1 | 0 |
| C3-LQ-RowGateRobustReanchor | KMNIST | 0 | -0.10416668653488159 | 0 | 3 | 1 | 0 |
| C4-LQ-MacroDeltaPriorityReanchor | KMNIST | 0 | -0.04166668653488159 | 0 | 4 | 0 | 0 |
| C5-LQ-StepMemoryRecheck | KMNIST | 0 | -0.020833373069763184 | 0 | 5 | 1 | 0 |
| C6-LQ-LineCNoRegressionCheck | KMNIST | 0 | -0.0625 | 0 | 4 | 1 | 0 |
| C0-HistoricalLQReferenceReplay | KMNIST | 1 | 0.10416668653488159 | 1 | 4 | 0 | 0 |
| C1-CurrentLQProtocolReplay | KMNIST | 1 | 0.0625 | 1 | 5 | 1 | 0 |
| C2-LQ-ProtocolMatchedReanchor | KMNIST | 1 | 0.0625 | 1 | 5 | 1 | 0 |
| C3-LQ-RowGateRobustReanchor | KMNIST | 1 | 0.10416668653488159 | 1 | 3 | 1 | 0 |
| C4-LQ-MacroDeltaPriorityReanchor | KMNIST | 1 | 0.10416668653488159 | 1 | 4 | 1 | 0 |
| C5-LQ-StepMemoryRecheck | KMNIST | 1 | 0.10416668653488159 | 1 | 5 | 1 | 0 |
| C6-LQ-LineCNoRegressionCheck | KMNIST | 1 | 0.08333337306976318 | 1 | 4 | 1 | 0 |
| C0-HistoricalLQReferenceReplay | KMNIST | 2 | -0.0625 | 0 | 4 | 1 | 0 |
| C1-CurrentLQProtocolReplay | KMNIST | 2 | -0.0625 | 0 | 5 | 1 | 0 |
| C2-LQ-ProtocolMatchedReanchor | KMNIST | 2 | -0.0625 | 0 | 5 | 1 | 0 |
| C3-LQ-RowGateRobustReanchor | KMNIST | 2 | -0.0833333432674408 | 0 | 3 | 1 | 0 |
| C4-LQ-MacroDeltaPriorityReanchor | KMNIST | 2 | -0.020833343267440796 | 0 | 4 | 1 | 0 |
| C5-LQ-StepMemoryRecheck | KMNIST | 2 | 0.020833313465118408 | 1 | 5 | 1 | 0 |
| C6-LQ-LineCNoRegressionCheck | KMNIST | 2 | -0.04166668653488159 | 0 | 4 | 1 | 0 |

Line D Rational summary：

| method | pass | source | AUC |
| --- | --- | --- | --- |
| D0-RAT-AdamWMonitor | 0 | -0.08313955201043023 |  |
| D1-RAT-G7AnalogPulseOnce-then-AdamWRecovery | 0 | -0.24516101678212485 |  |
| D2-RAT-G7AnalogPulseOnce-then-DecayRecovery | 0 | -0.21890090571509468 |  |
| D3-RAT-NoRegressionCheck | 0 | -0.08313955201043023 |  |
| DCTRL-RandomPulseSameRecovery | 0 | -0.40054451094733345 |  |

Line F all-basis family summary：

| family | rows | pass | best candidate | mean delta | gate |
| --- | --- | --- | --- | --- | --- |
| D-FOU | 45 | 2 | D-FOU93-BandwiseSNRWarmupV7 | -0.1361111107799742 | 0 |
| D-RBF | 45 | 0 | D-RBF90-ActiveCenterOccupancyV7 | -0.2300925936963823 | 0 |
| D-WAV | 36 | 0 | D-WAV79-SupportOverlapDampingV7 | -0.14930555576251614 | 0 |

Line M attribution summary：

| mechanism | KAN method | best MLP | KAN source | MLP source | delta | KAN adv |
| --- | --- | --- | --- | --- | --- | --- |
| M1-ProductivePulseRecovery | A-M1-R0-D-CHE-PulseOnce-AdamWRecovery | B-M4-MLP-FunctionSpaceProximal | 0.015958600574069552 | 0.06426172786288792 | -0.048303127288818366 | 0 |
| M2-SplitConsensusSignalSubspace | A-M2-D-CHE-SplitConsensusSignalSubspace | B-M4-MLP-FunctionSpaceProximal | 0.015958600574069552 | 0.06426172786288792 | -0.048303127288818366 | 0 |
| M3-PopRiskDriftSNR | A-M3-D-CHE-PopRiskSNRPreconditioner | B-M4-MLP-FunctionSpaceProximal | 0.016755845811631944 | 0.06426172786288792 | -0.04750588205125597 | 0 |
| M4-FunctionSpaceProximal | A-M4-D-CHE-FunctionSpaceProximal | B-M4-MLP-FunctionSpaceProximal | 0.12824556562635633 | 0.06426172786288792 | 0.06398383776346842 | 0 |
| M5-OptimizerAwareAlignedFU | A-M5-D-CHE-OptimizerAlignedFU | B-M4-MLP-FunctionSpaceProximal | -0.9984277354346381 | 0.06426172786288792 | -1.0626894632975261 | 0 |
| M6-CarrierSpecificReparamFU | A-M6-D-CHE-CarrierReparamGlobalDecay | B-M4-MLP-FunctionSpaceProximal | 0.0170792473687066 | 0.06426172786288792 | -0.04718248049418132 | 0 |

Failure taxonomy summary：

| failure class | rows |
| --- | --- |
| source,tail_debt,linec_debt,control_equivalent | 89 |
| source,retention,tail_debt,control_equivalent | 37 |
| source,retention,tail_debt,linec_debt,control_equivalent | 37 |
| retention,tail_debt,linec_debt | 17 |
| source,retention,linec_debt,control_equivalent | 13 |
| tail_debt,linec_debt | 10 |
| retention,tail_debt | 8 |
| source,retention,control_equivalent | 5 |
| AllBasisSubstrateBlocked | 1 |
| R-C-LQReanchorStillBlocked | 1 |
| RationalMonitorNoPromotion | 1 |

## 6. Final route / no-go

```text
route = R16_1-FunctionalMechanismMatrixNoGo
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
| LineA-DCHENoGo | 1 | best=A-M4-D-CHE-FunctionSpaceProximal;source_h800=0.12824556562635633;retention=0.24008667800161573;tail=0.0;LineC=0.4444444444444444 |
| LineB-MLPGenericNoPromotion | 1 | best=B-M4-MLP-FunctionSpaceProximal;source_h800=0.06426172786288792;retention=0.6871364712715149;tail=0.0;LineC=0.0 |
| LineC-LQReanchor | 1 | best=C0-HistoricalLQReferenceReplay;delta=0.10416668653488159;near=5/9;route=R-C-LQReanchorStillBlocked |
| LineD-RationalMonitorOnly | 1 | best=D0-RAT-AdamWMonitor;pass=0;no reset/controller |
| LineF-AllBasisSubstrate | 1 | best_family=D-FOU;pass=2/9 |
| R16_1-FunctionalMechanismMatrixNoGo | 1 | A=0;B=0;C=0;D=0;F=0 |
| NoForbiddenContinuation | 1 | no G9/G10/action bank/controller/reset/audit-directed branch |

Next hypothesis queue：

| priority | hypothesis | allowed next step |
| --- | --- | --- |
| 1 | MechanismSpecificRecoveryDefinitionRevision | Only in a new pre-registered plan; use v16.1 matrix as theory evidence, not as controller. |
| 2 | MLPGenericDynamicsMechanismFollowup | Study MLP weak dynamics as generic dynamics, not KAN-specific promotion. |
| 3 | CarrierSubstrateBeforeFU | Repair LQ/all-basis carrier before official functional update proof. |

## 7. 科学结论

```text
1. v16.1 已执行 carrier × mechanism × horizon × controls matrix，并生成 required artifacts。
2. A/B 均完成 H=800 full surface；top-2 已执行 H=1600 extension。
3. LineC/CEp99/NLL/ECE/AUCtime/Brier 只用于 audit/gate/debt readback，没有反推方向。
4. MLP generic positive 不写成 KAN-specific promotion；Line M attribution 是强制证据链。
5. 当前 route = R16_1-FunctionalMechanismMatrixNoGo，promotion_allowed = 0。
```

## 8. 用户再次追问后的计划闭环复核

本节不新增训练，不改变任何指标；它记录再次按 `DG-KAN_v16.1_MultiHypothesisFunctionalDynamics_4GPU_完整计划.md` 复核后的判断。

```text
route = R16_1-FunctionalMechanismMatrixNoGo
S2_weak_productive_dynamics_reached = 0
S3_productive_debt_recovery_reached = 0
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
required_artifact_manifest_rows = 52
required_artifact_missing_rows = 0
execution_contract_nonpass_rows = 0
deep_coverage_nonpass_rows = 0
```

计划第 13 节的 final route 逻辑要求：如果所有 carrier 与所有 mechanism 都未打开 S2/S3/S5，并且 required/forbidden/no-action/budget/exhaustion 已闭合，则写 `R16_1-FunctionalMechanismMatrixNoGo`，下一步只能进入新的 theory-level reset 或 substrate/base-level redesign。计划同时明确禁止在本版继续 `G9/G10`、action bank、controller、reset route 或 audit-directed search。

因此，v16.1 的“覆盖执行目标”已经达成，但“productive functional dynamics / promotion 目标”没有达成。不能在 v16.1 内继续补 token 或把 MLP/LQ/all-basis 线索改写成 promotion；继续推进需要新预注册计划。当前 next hypothesis queue 已写入三个合法方向：

| priority | hypothesis | allowed next step |
| --- | --- | --- |
| 1 | MechanismSpecificRecoveryDefinitionRevision | Only in a new pre-registered plan; use v16.1 matrix as theory evidence, not as controller. |
| 2 | MLPGenericDynamicsMechanismFollowup | Study MLP weak dynamics as generic dynamics, not KAN-specific promotion. |
| 3 | CarrierSubstrateBeforeFU | Repair LQ/all-basis carrier before official functional update proof. |

追加复核判断：

```text
1. 不是没继续，而是 v16.1 计划内可继续分支已闭合。
2. 最强 h800 row A-M4-D-CHE-FunctionSpaceProximal 有 source=0.12824556562635633，但 retention=0.24008667800161573、tail=0.0，S2/S3 均不能打开。
3. h1600 top-2 没有巩固：A-M4 mean source=0.0，B-M4 mean source=0.0，bad_event 均值仍为 1.0。
4. Line M 中 KAN_specific_advantage_rows=0/6，不能把 D-CHE 局部 source 写成 KAN-specific promotion。
5. Line F all-basis best family D-FOU 只有 pass=2/9 且 gate=0，只能作为 substrate repair 证据。
6. 当前版本内继续新增 action/controller/reset 会违反计划精神与 no-action audit。
```

## 9. 二次追问后的最终闭环复核

本节仍然不新增训练、不改变指标，只记录对同一问题的最终回答：`v16.1` 的 coverage/contract 目标达成，promotion/scientific success 未达成；计划内没有剩余可执行补救分支。

复核读回：

```text
route = R16_1-FunctionalMechanismMatrixNoGo
S2_weak_productive_dynamics_reached = 0
S3_productive_debt_recovery_reached = 0
official_s5_reached = 0
promotion_allowed = 0
required_artifact_manifest_rows = 52
required_artifact_missing_rows = 0
execution_contract_nonpass_rows = 0
deep_coverage_nonpass_rows = 0
budget_certificate_rows = 5
```

计划原文约束复核：

```text
No S5, no promotion. No exceptions.
route = R16_1-FunctionalMechanismMatrixNoGo
Next must be theory-level reset or substrate/base-level redesign.
如果所有 carrier 和所有机制都失败，就必须停止当前 train-stream functional-update family，而不是继续 token/action/controller/reset。
```

最终判断：

```text
1. 不能继续在 v16.1 内新增实验 token，因为这会违反 no-action-search / no-controller / no-reset 边界。
2. 不能把 A-M4 的 h800 source 当作成功，因为 retention=0.24008667800161573、tail=0.0，且 h1600 source 聚合回到 0.0。
3. 不能把 MLP B-M4 的 source-retention 当作 KAN promotion，因为 Line M 的 KAN_specific_advantage_rows=0/6。
4. 不能把 Line F 的 D-FOU pass=2/9 当作 all-basis escape，因为 all-basis gate=0。
5. 所以本版只能收敛为 no-go，下一步必须另起新预注册计划。
```
