# DG-KAN v16.2：Multi-Scheme Functional Dynamics Portfolio + MLP Active + LQ Reanchor + All-Basis + 4GPU 并行加速完整计划

> 版本：v16.2 execution plan  
> 生成时间：2026-06-01  
> 依据：v16.1 `MultiHypothesisFunctionalDynamics 4GPU` 实验复盘  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`  
> 核心修正：v16.1 已经做到多 carrier，但机制面仍偏 representative surface；v16.2 必须做到 **每个 active carrier 同时跑多套 functional dynamics 方案**，并充分利用 4 张 GPU，避免单线等待和浅尝辄止。  
> 硬约束：strict FC-PureKAN；no active B-spline；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific scale；no fake / proxy / CPU offload；functional direction 不使用 validation / test / future / query；LineC / CEp99 / NLL / ECE / AUCtime / Brier 只能作为 audit / gate / debt readback，不能作为方向源。

---

# 0. 项目总目标与当前真实状态

DG-KAN 的总目标仍然是：

$$
\boxed{
\text{在 strict FC-PureKAN base 上，通过 functional update 改善训练动力学，}
\text{最终得到比普通 AdamW/backprop 更好的模型。}
}
$$

这里“更好”不是某个局部 source gain，不是某个 real-lite 1/9 或 2/9，不是 MLP generic positive，也不是某个 substrate-only row。真正目标必须同时满足：

```text
1. base / substrate 具备可用效率与任务健康；
2. functional update 的收益超过 AdamW / NoOp / Random / matched / generic optimizer controls；
3. source 不只是短期正，而是在 h800/h1600 长期保留；
4. tail / LineC / calibration / AUC debt 能被后续训练动力学偿还；
5. 若 MLP-FU 也成功，必须写成 generic training-dynamics insight，不能写成 KAN-specific；
6. 只有 KAN carrier 相对 MLP/generic controls 有额外优势，才允许讨论 KAN-specific functional advantage；
7. final promotion 仍要求 S5 strict 9/9。
```

v16.1 的真实结论是：

```text
route = R16_1-FunctionalMechanismMatrixNoGo
S2_weak_productive_dynamics_reached = 0
S3_productive_debt_recovery_reached = 0
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

v16.1 做对了两件事：

```text
1. 从 D-CHE 单线等待，推进到 carrier × mechanism × horizon × controls matrix。
2. 四卡分片执行已落 artifact，cpu_offload_used=0，direction provenance 仍是 train-stream-only / optimizer_state / split_gradient。
```

但 v16.1 仍然不足：

```text
1. 每个 active carrier 上的机制仍偏 representative surface，很多机制族只用一个代表点；
2. h800/h1600 有覆盖，但没有形成系统性的 recovery mechanism ladder；
3. D-CHE / MLP 都有局部 source，但 S2/S3 = 0；
4. Line M 显示 KAN-specific advantage rows = 0/6；
5. LQ reanchor 仍未开；
6. D-FOU / D-RBF / D-WAV substrate 仍未达到可进入 official FU proof 的 gate。
```

v16.2 的核心制度是：

$$
\boxed{
\text{不再“一条 carrier 一个方案”；}
\text{每个 active carrier 同时跑多套 functional dynamics 假设。}
}
$$

---

# 1. v16.1 结果的独立分析

## 1.1 D-CHE：有局部 source，但没有 productive dynamics

v16.1 的全局 best h800 source 来自：

```text
A-M4-D-CHE-FunctionSpaceProximal
source_vs_best_control_h800 = 0.12824556562635633
source_retention_h800 = 0.24008667800161573
tail_recovery_h800 = 0.0
LineC_recovery_h800 = 0.4444444444444444
AUCtime_ratio_h800 = 1.0713891210432418
```

它说明：

```text
1. D-CHE 上仍有 source signal；
2. 这个 signal 没有被长期保留；
3. tail debt 没有恢复；
4. AUCtime 仍大于 1.05；
5. 因此不能打开 S2/S3。
```

h1600 extension 进一步关闭了“再等更久会巩固”的解释：

```text
A-M4 h1600 mean source = 0.0
A-M6 h1600 mean source = -0.13498353958129883
bad_event mean = 1.0
```

所以 D-CHE 不是“完全没信号”，而是 **source / recovery / control attribution 没同时闭合**。

## 1.2 MLP：必须保留为 active functional dynamics line

v16.1 的 MLP best 是：

```text
B-M4-MLP-FunctionSpaceProximal
source_vs_best_control_h800 = 0.06426172786288792
tail_recovery_h800 = 0.0
LineC_recovery_h800 = 0.0
```

这说明 MLP 上也有局部 source，但没有形成 productive debt recovery。MLP 不能只当 control，因为它回答一个核心问题：

$$
\boxed{
\text{functional update 的长期动力学是否是 generic neural training phenomenon？}
}
$$

如果 MLP 上可行、KAN 上不可行，问题在 KAN carrier；如果 MLP 与 KAN 都不可行，当前 functional dynamics 定义本身要重写；如果 KAN 可行而 MLP 不行，才有 KAN-specific 机会。

## 1.3 LQ：不能丢，但必须先 reanchor

v16.1 的 LQ reanchor gate 仍为 0：

```text
best = C0-HistoricalLQReferenceReplay
near = 5/9
gate = 0
```

LQ 历史上有过 near-pass 和 functional-gate 价值，但当前协议下未重锚，不能直接进入 official functional proof。v16.2 必须保留 LQ，但必须按：

```text
reanchor -> late attach -> functional smoke
```

不能跳过 reanchor。

## 1.4 Rational：只做 monitor，不能重启旧 route

Rational reset / optimizer-state route 已经多轮被 generic controls 解释。v16.2 只允许 Rational 做：

```text
1. no-regression monitor；
2. dynamics sanity monitor；
3. 与 D-CHE / MLP 的 reference comparison。
```

禁止重启：

```text
reset route；
controller；
action bank；
K-RT/K-AUC/K-FL。
```

## 1.5 D-FOU / D-RBF / D-WAV：仍是 substrate repair，不能 official FU proof

v16.1 Line F：

```text
best_family = D-FOU
best_family_dataset_seed_pass_count = 2/9
D-RBF / D-WAV gate = 0
```

这些不能进入 official functional update proof。但 D-FOU/RBF/WAV 不应停止，因为它们回答 carrier escape 问题：

$$
\boxed{
\text{如果 D-CHE 不适合承载长期 debt recovery，}
\text{是否有其它 basis carrier 可以承载？}
}
$$

---

# 2. v16.2 的核心假设

v16.2 不再只问“哪个方法过 gate”，而是并行验证六个机制假设。每个机制都必须在 D-CHE 与 MLP 上主动执行，在 LQ reanchor 成功后执行，在 Rational 上 monitor，在 D-FOU/RBF/WAV 上视 substrate gate 进行 limited smoke。

---

## H1：Productive perturbation 假设

短期 bad update 可能不是失败，而是 training debt。一个 functional pulse 可以先制造 plasticity，再由后续训练动力学偿还 debt。

成立证据：

```text
source_vs_best_control_h800 >= 0.005
source_retention_h800 >= 0.50
tail_recovery_h800 >= 0.60
LineC_recovery_h800 >= 0.60
AUCtime_ratio_h800 <= 1.05
matched controls fail
```

失败证据：

```text
h800/h1600 source 转负；
tail / LineC debt 不恢复；
random pulse + same recovery 解释；
recovery-only 解释。
```

---

## H2：Recovery mechanism mismatch 假设

v16.0/v16.1 的 recovery 设置可能太单一。bad debt 未恢复不代表 pulse 无价值，可能是 recovery 机制错了。

v16.2 必须比较：

```text
ordinary AdamW recovery
LR cooldown recovery
momentum damping recovery
decoupled weight decay recovery
role-wise / degree-wise decay recovery
EMA / Lookahead / SWA consolidation
second-moment adaptation recovery
gradient-noise diffusion / batch reshuffle recovery
feature recentering recovery
late consolidation recovery
```

注意：

$$
\boxed{
\text{weight decay 是重要 recovery mechanism，但不是唯一偿还机制。}
}
$$

---

## H3：Function-space proximal source 假设

v16.1 最强 D-CHE source 是 function-space proximal。这个方向不能直接丢，但不能只跑一个代表点。需要拆成多个 proximal variants：

```text
proximal in Adam subspace
proximal in per-example gradient low-rank subspace
proximal in output-Jacobian sketch subspace
proximal with split B1/B2 consistency
proximal with recovery-aware horizon target
proximal residualized against matched controls
```

如果所有 proximal variants 都 local positive no transfer，则 route 到：

```text
R-ProximalLocalPositiveNoTransfer
```

---

## H4：PopRisk / drift-diffusion signal 假设

Generalization theory 给的启发是：泛化方向不是静态几何指标，而是 coherent population signal 能否在 signal channel 中通过 drift 积累，噪声是否被压进 reservoir。v16.2 必须恢复 PopRisk/SNR 作为机制线，而不是只作为过去失败历史。

核心量：

$$
\mu_r=\frac{1}{b}\sum_i g_{i,r}
$$

$$
\sigma_r^2=\frac{1}{b}\sum_i (g_{i,r}-\mu_r)^2
$$

$$
SNR_r=\frac{\mu_r^2}{\sigma_r^2/(b-1)+\epsilon}
$$

但这次 SNR 不直接当 one-step update，而作为：

```text
preconditioner；
pulse scheduler；
recovery scheduler；
noise-channel guard；
carrier-role trust signal。
```

---

## H5：Carrier coordinate mismatch 假设

D-CHE 上 source/hazard 曾经高度绑定。问题可能不是 update 本身，而是 carrier 坐标无法把 source 与 instability 分开。

v16.2 必须同时测试：

```text
D-CHE degree-role carrier
D-CHE readout-basis decoupled carrier
D-CHE low-degree signal / high-degree reservoir carrier
LQ carrier after reanchor
Rational monitor carrier
D-FOU low-frequency carrier
D-RBF active-center local carrier
D-WAV local support carrier
MLP hidden-subspace carrier
```

如果 MLP 可 recovery 而 D-CHE 不可 recovery，优先诊断 carrier mismatch。

---

## H6：Geometry definition mismatch 假设

“好几何”不能再定义为单步不坏。v16.2 使用动态几何：

$$
\boxed{
\text{好几何}
=
\text{source 可迁移、debt 可恢复、controls 打不过、long-horizon 更好。}
}
$$

LineC / tail / calibration / AUC 不再是 early hard-stop，而是 debt readback：

```text
LineC debt
tail debt
calibration debt
AUC debt
```

early bad 不等于 fail；long-horizon unrecovered bad 才是 fail。

---

# 3. Carrier × Mechanism 实验矩阵

v16.2 必须执行完整矩阵，不允许每个 carrier 只跑一个方案。

## 3.1 Active carriers

```text
C1-D-CHE:
  当前最强 KAN carrier，必须执行全部 M1-M8。

C2-MLP:
  active functional dynamics discovery line，必须执行全部 M1-M8。
  不是普通 control。

C3-LQ:
  先执行 R0-R6 reanchor。
  reanchor weak pass 后执行 M1-M5 short horizon；
  reanchor full pass 后执行 M1-M8。

C4-Rational:
  monitor + sanity，执行 M1/M3/M7 light rows；
  不重启 reset/controller/action route。

C5-D-FOU:
  substrate repair + limited M2/M3 smoke if >=6/9 substrate exploration。

C6-D-RBF/FastKAN:
  substrate repair + limited M3/M5 smoke if >=6/9 substrate exploration。

C7-D-WAV:
  low-budget substrate monitor + local-support recovery audit。
```

## 3.2 Functional mechanism families

```text
M1 ProductivePulseRecovery:
  pulse once / pulse periodic / early-only / mid-only / late-only
  + multiple recovery mechanisms.

M2 SplitConsensusMetric:
  split-consensus signal subspace;
  ordinary gradient projected through signal metric.

M3 PopRiskDriftDiffusion:
  train-stream per-example gradient drift-diffusion SNR;
  used as preconditioner / scheduler / recovery trust.

M4 FunctionSpaceProximal:
  function-space or output-space local solver;
  split B1/B2 consistency;
  no validation/test/future/query.

M5 OptimizerAwareAlignedFU:
  gradient-aligned / cautious / MGUP / second moment / curvature clip;
  not as extra direction unless controls fail.

M6 CarrierReparameterizedFU:
  low-degree signal / high-degree reservoir;
  readout-basis decoupling;
  output-Jacobian carrier;
  dual-bank carrier.

M7 LateAttachSnapshotFU:
  attach functional pulse to a partially trained checkpoint;
  especially for LQ / D-CHE / MLP.

M8 RecoveryOnlyDeconfound:
  recovery mechanisms without functional pulse;
  distinguishes pulse value from recovery value.
```

---

# 4. Four-GPU execution contract

服务器有 4 张 GPU，v16.2 必须显式充分利用。不能再单卡顺序跑，也不能让某张卡空闲时还有 runnable queue。

所有任务必须写入：

```text
v162_gpu_assignment_manifest.csv
v162_gpu_utilization_summary.csv
v162_queue_status.csv
v162_deferred_items.csv
```

## 4.1 GPU 使用原则

```text
1. 每张 GPU 都要有独立 queue。
2. 每个 queue 都有 primary job、fallback job、fill job。
3. 如果某 GPU 空闲超过 10 分钟且存在 runnable job，必须写 execution-contract violation。
4. 如果某 job 因预算 deferred，必须写 planned_budget / consumed_budget / defer_reason / next_priority。
5. cpu_offload_used 必须为 0。
```

## 4.2 Round 0：smoke / registry / audit

```text
GPU0:
  D-CHE M1-M8 smoke + method surface manifest。

GPU1:
  MLP M1-M8 smoke + MLP active functional manifest。

GPU2:
  LQ R0-R6 reanchor smoke + Rational monitor smoke。

GPU3:
  D-FOU / D-RBF / D-WAV substrate smoke + all-basis candidate registry。
```

## 4.3 Round 1：short-horizon matrix

```text
GPU0:
  D-CHE M1/M2/M3/M4, horizons H=1/5/20/50/100, with controls.

GPU1:
  MLP M1/M2/M3/M4, horizons H=1/5/20/50/100, with controls.

GPU2:
  LQ R0-R6 full reanchor + Rational M1/M3 light monitor.
  If LQ weak reanchor opens, immediately enqueue LQ M1/M2 short horizon.

GPU3:
  D-FOU / D-RBF / D-WAV substrate repair full rows.
```

## 4.4 Round 2：long-horizon and alternate mechanisms

```text
GPU0:
  D-CHE M5/M6/M7/M8 + top-3 D-CHE from Round 1 to H=400/800.

GPU1:
  MLP M5/M6/M7/M8 + top-3 MLP from Round 1 to H=400/800.

GPU2:
  LQ M1-M5 if reanchor opens;
  otherwise deeper LQ reanchor R7-R10 and late-attach readback.
  Rational monitor continuation.

GPU3:
  all-basis continued substrate repair.
  Any D-FOU/RBF/WAV reaching >=6/9 gets limited M1/M3 functional smoke.
```

## 4.5 Round 3：h1600 consolidation

```text
GPU0:
  top-2 D-CHE candidates to H=1600.

GPU1:
  top-2 MLP candidates to H=1600.

GPU2:
  top-1 LQ / Rational candidate if valid; otherwise extra D-CHE/MLP controls.

GPU3:
  top all-basis substrate continuation + limited functional smoke if any family opened.
```

## 4.6 Dynamic queue rule

If any GPU finishes early:

```text
Priority 1:
  unmatched controls for any positive-looking row.

Priority 2:
  h800 extension for source-retaining rows.

Priority 3:
  h1600 extension for rows with source_retention_h800 >= 0.30.

Priority 4:
  all-basis substrate rows.

Priority 5:
  LQ reanchor variants.

Priority 6:
  failure taxonomy / figures / artifact checks.
```

---

# 5. Line R：provenance / no-action-search audit

Line R 先执行，且每个 round 后再执行一次 incremental audit。

必须检查：

```text
uses_validation_test_future_query_for_direction = 0
uses_LineC_tail_AUC_calibration_for_direction = 0
uses_dataset_name_branch = 0
uses_seed_specific_scale = 0
action_token_added = 0
controller_executed = 0
action_bank_used = 0
reset_route_used = 0
fake_proxy_cpu_offload = 0
required_artifact_missing_count = 0
```

v16.2 允许的是 **预注册机制矩阵**，不是 action search。禁止：

```text
M9/M10 临时追加；
G9/G10；
F-CHE8/F-CHE9；
K-RT/K-AUC/K-FL；
dataset-specific branch；
seed-specific rescue；
audit-directed direction；
controller；
action bank。
```

---

# 6. Line A：D-CHE multi-scheme functional dynamics

D-CHE 是当前主 KAN carrier，但不再只跑一种方案。

## 6.1 A-M1 Productive pulse + recovery portfolio

Methods：

```text
A-M1a PulseOnce + AdamW recovery
A-M1b PulseOnce + LR cooldown
A-M1c PulseOnce + momentum damping
A-M1d PulseOnce + decoupled global weight decay
A-M1e PulseOnce + degree-wise decay
A-M1f PulseOnce + EMA consolidation
A-M1g PulseOnce + Lookahead consolidation
A-M1h PulseOnce + SWA-like consolidation
A-M1i PulseEvery50 + recovery
A-M1j EarlyOnlyPulse + recovery
A-M1k MidOnlyPulse + recovery
A-M1l LateOnlyPulse + recovery
```

Controls：

```text
RandomMatchedPulse + same recovery
NoOpMatchedOverhead
RecoveryOnly
AdamWExtraStepsMatchedTime
```

## 6.2 A-M2 Split-consensus metric portfolio

Methods：

```text
A-M2a Diagonal split-consensus metric
A-M2b Role-block split-consensus metric
A-M2c Lowrank-r4 split-consensus metric
A-M2d Lowrank-r8 split-consensus metric
A-M2e Metric-only no projection
A-M2f Projection plus AdamV
A-M2g Negative-eigen quarantine diagnostic
```

Controls：

```text
RandomSubspaceSameRank
RandomSubspaceSameProjectionRetention
SameActiveFractionRandomMask
SameDegreeRoleEnergyRandom
NoOpMatchedOverhead
```

## 6.3 A-M3 PopRisk / drift-diffusion SNR portfolio

Methods：

```text
A-M3a Parameter SNR preconditioner
A-M3b Degree-role SNR preconditioner
A-M3c Basis-channel SNR with cover boundary
A-M3d SNR pulse scheduler
A-M3e SNR recovery scheduler
A-M3f SNR reservoir guard
```

Controls：

```text
ShuffledPerExampleGradient
SameMaskRandomSign
SameActiveFractionRandomMask
AdamWParallelDirection
```

## 6.4 A-M4 Function-space proximal portfolio

Methods：

```text
A-M4a Adam-subspace proximal
A-M4b Per-example gradient lowrank proximal
A-M4c Output-Jacobian sketch proximal
A-M4d Split B1/B2 transfer proximal
A-M4e Recovery-aware proximal
A-M4f Control-residualized proximal
```

Fixed alpha grid：

```text
alpha = 0, 0.025, 0.05, 0.10, 0.20
```

No adaptive controller.

## 6.5 A-M5 Optimizer-aware aligned FU

Methods：

```text
A-M5a Cautious FU
A-M5b Soft cautious FU
A-M5c MGUP-style reweight
A-M5d Adam second moment scaled FU
A-M5e SophiaDiagLite clipped FU
A-M5f Block second moment FU
A-M5g Decoupled decay + FU, with decay-only control
```

This line answers whether previous FU failed because it ignored optimizer dynamics.

## 6.6 A-M6 Carrier reparameterized FU

Methods：

```text
A-M6a LowDegreeSignal / HighDegreeReservoir
A-M6b Readout-basis decoupled carrier
A-M6c Orthogonal degree bank
A-M6d Output-Jacobian carrier
A-M6e Dual-bank signal/reservoir carrier
A-M6f High-degree quarantine carrier
```

This line answers whether D-CHE carrier coordinates bind source and hazard.

## 6.7 A-M7 Late attach

Methods：

```text
A-M7a attach at early checkpoint
A-M7b attach at mid checkpoint
A-M7c attach at late checkpoint
A-M7d attach only after loss plateau
A-M7e attach after high source/low debt window
```

Checkpoint selection can use train-stream state only.

## 6.8 A-M8 Recovery-only deconfound

Methods：

```text
A-M8a AdamW recovery only
A-M8b LR cooldown only
A-M8c decay recovery only
A-M8d EMA/SWA recovery only
A-M8e Lookahead recovery only
A-M8f momentum damping only
```

If A-M8 explains gains, no functional claim.

---

# 7. Line B：MLP active functional dynamics

MLP is not only a control. It is an active mechanism discovery line.

The same M1-M8 families must be run on MLP:

```text
B-M1 Productive pulse + recovery
B-M2 Split-consensus metric
B-M3 PopRisk / drift-diffusion SNR
B-M4 Function-space proximal
B-M5 Optimizer-aware aligned FU
B-M6 Hidden-subspace / carrier reparameterized FU
B-M7 Late attach
B-M8 Recovery-only deconfound
```

Important interpretation rules:

```text
If MLP succeeds and D-CHE fails:
  functional dynamics may be generic; KAN carrier is the blocker.

If both MLP and D-CHE succeed:
  write generic training-dynamics result unless KAN advantage is positive.

If D-CHE succeeds and MLP fails:
  possible KAN-specific carrier advantage.

If both fail:
  current functional dynamics family likely wrong.
```

MLP metrics must include:

```text
hidden_subspace_rank
hidden_drift
feature_recentering_norm
logit_rms_drift
source_retention_h800
tail_recovery_h800
LineC_recovery_h800
AUCtime_h800
```

---

# 8. Line C：LQ reanchor + late attach

LQ cannot be ignored, but it cannot be promoted before reanchor.

## 8.1 Reanchor candidates

```text
C0 HistoricalLQReferenceReplay
C1 CurrentLQProtocolReplay
C2 ProtocolMatchedReanchor
C3 RowGateRobustReanchor
C4 MacroDeltaPriorityReanchor
C5 StepMemoryRecheck
C6 LineCNoRegressionCheck
C7 LateAttachReadyReanchor
C8 LQLossAgnosticMotionReanchor
C9 LQDebtRecoveryProbe
C10 LQTrainStreamSplitTransferProbe
```

Reanchor weak gate:

```text
near_count >= 6/9
LineC_no_regression = 1
step_ratio <= 1.50
memory_ratio <= 1.50
```

Reanchor official gate:

```text
near_count = 9/9
LineC_no_regression = 1
step_ratio <= 1.25
memory_ratio <= 1.25
```

If weak gate opens, run LQ M1-M5 short horizon. If official gate opens, run LQ M1-M8.

If reanchor fails, output:

```text
LQBaseNotReanchored
protocol_drift_summary
row_level_failure_table
late_attach_deferred_certificate
```

---

# 9. Line D：Rational monitor

Rational remains a monitor and sanity carrier. Run only:

```text
D0 RAT-AdamWMonitor
D1 RAT-G7AnalogPulseRecovery
D2 RAT-PopRiskSNRLight
D3 RAT-FunctionSpaceProximalLight
D4 RAT-NoRegressionCheck
DCTRL RandomPulseSameRecovery
```

Hard prohibition:

```text
no reset route
no optimizer-state transport route
no K-RT/K-AUC/K-FL
no controller
no action bank
```

---

# 10. Line F：All-basis substrate acceleration

D-FOU / D-RBF / D-WAV continue as substrate lines. They do not enter official FU proof unless they pass substrate gates.

## 10.1 D-FOU

Candidates：

```text
D-FOU97-LowFreqIdentityResidualV8
D-FOU98-BandwiseSNRWarmupV8
D-FOU99-PhaseStableBandMixV8
D-FOU100-NoMaterializeLifetimeV8
D-FOU101-HighFrequencyQuarantineV8
D-FOU102-SplitConsensusLowFreqMetricSmoke
```

## 10.2 D-RBF / FastKAN

Candidates：

```text
D-RBF95-ActiveCenterOccupancyV8
D-RBF96-WidthConditionGuardV8
D-RBF97-CompactBumpNoDenseV8
D-RBF98-GaussianLocalK4TaskHealthV8
D-RBF99-ActiveCenterSecondMomentV8
D-RBF100-SplitConsensusCenterMetricSmoke
```

## 10.3 D-WAV

Candidates：

```text
D-WAV81-TriangularSupportV8
D-WAV82-ScaleOccupancyV8
D-WAV83-SupportOverlapDampingV8
D-WAV84-LocalTailCoverageAuditV8
D-WAV85-SplitConsensusScaleMetricSmoke
```

Substrate exploration gate:

```text
family_dataset_seed_pass_count >= 6/9
mean_delta_vs_MLP >= -0.05
worst_delta_vs_MLP >= -0.10
step_ratio <= 1.75
memory_ratio <= 1.75
LineC_pass_rate >= 0.30
```

Official substrate gate:

```text
family_dataset_seed_pass_count = 9/9
step_ratio <= 1.25
memory_ratio <= 1.25
LineC_pass_rate = 1.0
```

If exploration gate opens, run limited M1/M3 smoke. If not, do not enter FU proof.

---

# 11. Metrics to record

Every row must record all fields below.

## 11.1 Identity and provenance

```text
carrier
mechanism_family
method_id
dataset
seed
gpu_id
round_id
horizon
direction_source
uses_validation_test_future_query
uses_audit_metric_for_direction
uses_dataset_name_branch
uses_seed_specific_scale
cpu_offload_used
```

## 11.2 Source / debt / recovery

```text
source_vs_best_control_h1
source_vs_best_control_h20
source_vs_best_control_h100
source_vs_best_control_h400
source_vs_best_control_h800
source_vs_best_control_h1600
source_retention_h800
source_retention_h1600

tail_debt_peak
tail_debt_h800
tail_recovery_rate_h800
tail_recovery_rate_h1600

LineC_debt_peak
LineC_debt_h800
LineC_recovery_rate_h800
LineC_recovery_rate_h1600

calibration_debt_peak
calibration_recovery_rate_h800
AUCtime_ratio_h800
AUCtime_ratio_h1600
```

## 11.3 Dynamic geometry

```text
train_split_transfer_score
B1_loss_delta
B2_loss_delta
split_consensus_score
signal_to_noise_ratio
projection_retention
source_hazard_overlap
hazard_null_source_retention
recovery_lag_h1
recovery_lag_h4
recovery_lag_h20
dynamic_geometry_score
```

## 11.4 Optimizer / recovery

```text
adam_update_norm
fu_update_norm
decay_update_norm
recovery_update_norm
decay_to_fu_norm_ratio
decay_vs_fu_cosine
momentum_norm
second_moment_mean
second_moment_p90
cautious_keep_fraction
cautious_reject_fraction
ema_distance
lookahead_sync_count
swa_distance
```

## 11.5 Efficiency

```text
step_time_ratio
memory_ratio
forward_time_ms
backward_time_ms
update_time_ms
recovery_time_ms
artifact_write_time_ms
gpu_utilization_mean
gpu_idle_seconds
```

---

# 12. Visualizations required

Must generate:

```text
fig_v162_carrier_mechanism_heatmap_source_h800.svg
fig_v162_carrier_mechanism_heatmap_s2_s3.svg
fig_v162_source_retention_vs_tail_recovery.svg
fig_v162_source_retention_vs_LineC_recovery.svg
fig_v162_horizon_curves_DCHE_top5.svg
fig_v162_horizon_curves_MLP_top5.svg
fig_v162_MLP_vs_DCHE_attribution_matrix.svg
fig_v162_LQ_reanchor_dashboard.svg
fig_v162_allbasis_substrate_heatmap.svg
fig_v162_recovery_mechanism_comparison.svg
fig_v162_controls_explainability_waterfall.svg
fig_v162_failure_taxonomy_heatmap.svg
fig_v162_gpu_utilization_dashboard.svg
fig_v162_deferred_items_dashboard.svg
```

---

# 13. Success gates

## S1：coverage success

```text
Line R complete
Line A D-CHE M1-M8 complete
Line B MLP M1-M8 complete
Line C LQ R0-R10 complete
Line D Rational monitor complete
Line F all-basis complete
Line M controls complete
Line Z route / no-go / exhaustion complete
4GPU utilization manifest complete
```

S1 is only execution success, not scientific success.

## S2：weak productive dynamics

A carrier-method pair reaches S2 if:

```text
source_vs_best_control_h800 >= 0.005
source_retention_h800 >= 0.40
tail_recovery_rate_h800 >= 0.40 or LineC_recovery_rate_h800 >= 0.40
AUCtime_ratio_h800 <= 1.10
matched controls fail
```

S2 does not allow promotion.

## S3：productive debt recovery

```text
source_vs_best_control_h800 >= 0.005
source_retention_h800 >= 0.50
tail_recovery_rate_h800 >= 0.60
LineC_recovery_rate_h800 >= 0.60
AUCtime_ratio_h800 <= 1.05
random pulse + same recovery fails
recovery-only fails
NoOp overhead fails
```

## S4：real-transfer exploration

```text
real_lite_pass_count >= 6/9
source_vs_best_control_mean >= 0.005
AUCtime_median <= 1.05
tail debt recovered
LineC debt recovered
controls fail
```

## S5：official success

S5 is strict and unchanged:

```text
real_dataset_seed_pass_count = 9/9
source_vs_best_control >= 0.005
AUCtime_ratio <= 1.0
CEp99_delta <= 0.05
NLL_delta <= 0.02
ECE_delta <= 0.02
LineC_pass = 1
step_time_ratio <= 1.25
memory_ratio <= 1.25
controls fail
forbidden audit pass
no-action-search audit pass
code review pass
promotion_allowed = 1
```

---

# 14. Failure taxonomy and fallback rules

Codex must not stop after one carrier or one method fails. Every failure must route into the next legally allowed branch.

## 14.1 If D-CHE fails but MLP succeeds

Conclusion:

```text
functional dynamics may be generic;
D-CHE carrier / KAN coordinate is blocker.
```

Required fallback:

```text
1. compare hidden-subspace MLP vs D-CHE degree-role subspace;
2. run D-CHE M6 carrier reparameterized rows;
3. run LQ reanchor and all-basis substrate acceleration.
```

## 14.2 If MLP fails and D-CHE fails

Conclusion:

```text
current functional dynamics definition is weak.
```

Required fallback:

```text
1. inspect Line G dynamic geometry failure;
2. compare M1/M2/M3/M4 mechanism classes;
3. output theory-level reset queue.
```

## 14.3 If D-CHE source positive but debt unrecovered

Required fallback:

```text
1. run recovery mechanism ladder M1b-M1h;
2. run M6 carrier reparameterization;
3. run M8 recovery-only controls;
4. extend to h1600 if source_retention_h800 >= 0.30.
```

## 14.4 If MLP source positive but debt unrecovered

Required fallback:

```text
1. run MLP M1 recovery ladder;
2. run MLP feature recentering / EMA / SWA;
3. check whether this is generic productive plasticity or generic damage.
```

## 14.5 If LQ reanchor fails

Required fallback:

```text
1. write protocol drift table;
2. run row-level failure analysis;
3. run C7/C8/C9/C10 if budget remains;
4. defer functional late attach explicitly, not as failure.
```

## 14.6 If all-basis substrate fails

Required fallback:

```text
1. run family-specific failure taxonomy;
2. distinguish task-health vs workspace vs LineC vs step/memory;
3. if D-FOU/RBF reaches >=4/9 but <6/9, schedule focused substrate repair next.
```

---

# 15. Stop rules

## 15.1 Promotion fail-closed

No S5, no promotion. No exceptions.

## 15.2 Exploration continue-open

These conditions cannot hard stop the round:

```text
D-CHE M1 fail
D-CHE M4 fail
MLP M1 fail
MLP M4 fail
LQ reanchor fail
Rational monitor fail
D-FOU/RBF/WAV <6/9
M1 pulse+recovery fail
M2 split-consensus fail
M3 PopRisk/SNR fail
M4 proximal fail
M5 optimizer-aware fail
M6 carrier reparam fail
MLP/generic controls positive
LineC/tail/AUC immediate fail
h400 fail
h800 fail
overhead high
random pulse explains result
```

These only produce failure taxonomy / fallback / next queue.

## 15.3 True hard stop

Only these can hard stop:

```text
required_artifact_missing_count > 0
forbidden_information_violation_count > 0
no_action_search_violation_count > 0
direction uses validation/test/future/query
direction uses LineC/CEp99/NLL/ECE/AUCtime/Brier
dataset-name branch
seed-specific scale
action token / controller / action bank / reset route
fake / proxy / CPU offload
GPU execution contract violation not explainable by hardware failure
```

---

# 16. Budget exhaustion certificate

Budget exhaustion is not a generic excuse. It is valid only if a certificate is written.

Every deferred item must include:

```text
item_id
carrier
mechanism_family
planned_rows
executed_rows
planned_gpu_hours
consumed_gpu_hours
budget_kind = gpu_time / wall_clock / rows / horizon / fallback_depth
deferred_reason
does_defer_affect_route
next_priority
```

If a runnable job remains and any GPU has idle time > 10 minutes without hardware failure, final stop is invalid.

---

# 17. Required artifacts

The runner/finalizer must write:

```text
v162_route_decision.json
v162_progress_table.csv
v162_gpu_assignment_manifest.csv
v162_gpu_utilization_summary.csv
v162_method_surface_manifest.csv
v162_carrier_mechanism_matrix.csv
v162_line_a_dche_results.csv
v162_line_b_mlp_results.csv
v162_line_c_lq_reanchor.csv
v162_line_d_rational_monitor.csv
v162_line_f_allbasis_substrate.csv
v162_line_m_controls_attribution.csv
v162_dynamic_geometry_debt_accounting.csv
v162_failure_taxonomy.csv
v162_budget_exhaustion_certificate.csv
v162_deferred_items.csv
v162_no_action_search_audit.csv
v162_forbidden_information_audit.csv
v162_required_artifact_manifest.csv
v162_code_review_packet.zip
v162_no_go_boundary.md
v162_next_hypothesis_queue.md
figures/*.svg
```

---

# 18. Final route logic

Routes:

```text
S5-OfficialFunctionalSuccess
S4-RealTransferExplorationPositive
S3-ProductiveDebtRecovery
S2-WeakProductiveDynamics
S1-MultiCarrierMultiMechanismCoverageComplete
R1-DCHESourceNoRecovery
R2-MLPGenericDynamicsOnly
R3-CarrierMismatch
R4-LQReanchorBlocked
R5-AllBasisSubstrateBlocked
R6-FunctionalDynamicsAllMechanismsNoGo
R0-ArtifactOrProvenanceViolation
```

Decision priority:

```text
1. If artifact/provenance violation -> R0.
2. If S5 -> S5.
3. If any carrier reaches S4 -> S4.
4. If any carrier reaches S3 -> S3.
5. If any carrier reaches S2 -> S2.
6. If coverage incomplete -> invalid stop, not no-go.
7. If all lines complete and no S2 -> R6-FunctionalDynamicsAllMechanismsNoGo.
8. Add secondary blockers:
   D-CHE no recovery,
   MLP generic only,
   LQ blocked,
   all-basis blocked.
```

---

# 19. What Codex must do next

Codex must execute in this order:

```text
1. Implement v16.2 method registry and 4GPU scheduler.
2. Run Round 0 smoke across all GPUs.
3. Run Round 1 short-horizon matrix.
4. Immediately extend source-retaining candidates to H=400/H=800.
5. Run MLP active dynamics in parallel, not after D-CHE.
6. Run LQ reanchor and all-basis substrate on dedicated GPUs.
7. For every positive-looking row, run matched controls.
8. For top rows, run H=1600 extension.
9. Write route/no-go only after all mandatory coverage or valid budget certificate.
```

Codex must not:

```text
1. Add new unregistered method tokens.
2. Start controller/action-bank/reset routes.
3. Stop because first carrier failed.
4. Stop because h400 failed.
5. Stop because LineC/tail immediate bad.
6. Use audit metrics for direction.
7. Use dataset/seed branch.
8. Call MLP positive KAN-specific.
9. Treat substrate-only as functional success.
```

---

# 20. Final interpretation

v16.1 answered a narrower question:

$$
\boxed{
\text{A representative carrier × mechanism matrix did not find productive functional dynamics.}
}
$$

v16.2 asks a broader and faster question:

$$
\boxed{
\text{Across D-CHE, MLP, LQ, Rational and all-basis carriers,}
\text{which functional dynamics mechanism family, if any, can create source, retain it,}
\text{repay training debt, and beat controls?}
}
$$

The key correction is:

```text
multi-carrier is not enough;
we need multi-carrier × multi-mechanism × multi-horizon × controls,
executed on all 4 GPUs.
```

If v16.2 still finds no S2/S3 under this matrix, then the current train-stream functional dynamics family should be formally no-go, and the project must shift to substrate/base-level redesign or a new theory of functional direction.
