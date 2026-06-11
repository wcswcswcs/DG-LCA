# DG-KAN v16.3：Multi-Scheme Functional Dynamics + Efficiency Census + 4GPU 并行加速完整计划

> 版本：v16.3 execution plan  
> 生成时间：2026-06-01  
> 公式格式：Typora 友好，只使用 `$...$` 和 `$$...$$`。  
> 核心修正：v16.2 已经完成 carrier × mechanism × horizon × controls matrix，但仍没有 S2/S3/S5；下一版不能继续“每个 carrier 跑一组代表方法”。v16.3 必须把 functional update、MLP active line、all-basis carrier repair、以及每个 basis 与同参数量 MLP 的效率真值表同时跑起来。  
> 硬约束：strict FC-PureKAN；no active B-spline；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific scale；no fake / proxy / CPU offload；functional direction 不使用 validation / test / future / query；LineC / CEp99 / NLL / ECE / AUCtime / Brier 只能作为 audit / gate / debt readback，不能作为方向源。

---

# 0. 项目总目标与当前真实状态

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是找到一个局部 source positive 的 update。项目目标是：

$$
\boxed{
\text{在 strict FC-PureKAN 或同等严格 carrier 上，通过 functional update 改善训练动力学，}
\text{得到比 ordinary AdamW/backprop 更好的模型。}
}
$$

这个“更好”必须同时满足：

```text
1. base / substrate 效率接近同参数量 MLP；
2. functional update 的长期收益超过 AdamW / random / matched / generic optimizer controls；
3. 训练轨迹长期更好，而不是只看 single-step source；
4. 短期 tail / LineC / calibration / AUC debt 可以被后续训练动力学偿还；
5. MLP-FU 成功时写成 generic training-dynamics insight，不写成 KAN-specific；
6. 只有 KAN carrier 相比 MLP analog 有额外优势时，才允许讨论 KAN-specific functional advantage；
7. 所有 positive-looking rows 必须通过 NoOp / RandomMatched / AdamWExtraSteps / RecoveryOnly / MLP analog / same-overhead controls。
```

v16.2 的真实状态是：

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

v16.2 证明了一件事：执行覆盖已经成熟，D-CHE 与 MLP 都跑了 M1a..M8f，H=800 full surface 与 H=1600 extension 已经落盘，但没有任何机制形成 productive debt recovery。最强 h800 row 是 `A-M4a-D-CHE-AdamSubspaceProximal-alpha000`，source positive，retention 不低，但 tail recovery 为 0，LineC recovery 低于 exploration gate；h1600 又没有把 h800 source 巩固成长期证据。因此本轮不是“没有继续”，而是当前 v16.2 matrix 下没有可 promotion 的机制。

本版 v16.3 的两个直接修正是：

```text
1. 多 carrier 不够；每个 active carrier 必须并行跑多套 functional dynamics 方案。
2. 我们还没有一张干净的 basis-vs-same-param-MLP efficiency truth table；必须单独建立 efficiency census。
```

---

# 1. 当前各线进展百分比

| 线 | 当前完成度 | 当前判断 |
|---|---:|---|
| 代码 / provenance / no-action audit | 99% | required / forbidden / no-action audit 成熟，不是当前 blocker |
| 4GPU 分片执行 | 85% | v16.2 已有 4GPU manifest，但需要动态队列、补位、utilization dashboard |
| Efficiency Census / basis-vs-MLP 真值表 | 10%-15% | 目前只有碎片化 step ratio / runtime insight，没有 clean forward/update/backward-memory 表 |
| D-CHE substrate | 85% | 当前最强 KAN carrier，历史上 full 3x3 substrate eligibility 站住 |
| D-CHE functional dynamics | 18%-25% | 有 h800 source，但 tail/LineC debt 和 h1600 巩固失败 |
| MLP functional dynamics | 25%-35% | 不能再只当 control；必须作为 active mechanism discovery line |
| LQ reanchor | 25%-35% | C2 近 8/9，但 reanchor gate 未开；不能 official functional |
| Rational monitor | 70%-80% substrate / 10% functional | 只做 monitor；不重启 reset/controller/action route |
| D-FOU substrate | 25%-35% | v16.2 best non-D-CHE，4/9；不能 official FU proof |
| D-RBF / FastKAN substrate | 20%-30% | workspace/telemetry 有线索，task-health 未稳 |
| D-WAV substrate | 15%-20% | 弱线索，低预算保留 |
| Dynamic geometry / debt recovery | 20%-25% | 概念已修正，但没有找到 productive debt repayment |
| Multi-scheme functional matrix | 20%-25% | v16.2 已跑 M1a..M8f，但下一版要扩大到每个 carrier 的多 scheme 直接并行 |
| Official S5 | 0% | 尚未达成 |
| 整体 next-gen MLP claim | 23%-31% | 科学 claim 仍弱；执行成熟但机制未闭合 |

---

# 2. 当前最大缺口：我们没有 clean efficiency truth table

用户明确指出：现在每个基函数和同参数量 MLP 相比，forward、参数更新速度、backward memory 到底如何，还没有底。这个批评是对的。

当前材料中有若干碎片：

```text
1. v16.2 观察到 MLP shard 明显快于 D-CHE shard，原因包括 D-CHE basis/functional carrier 前向图更重、split-gradient 反向更贵、D-CHE horizon readback 额外执行 LineC pack。
2. v14.9 曾显示 D-CHE substrate median train step ratio 很低，但那不是 current v16.2 functional path 的 phase-level forward/backward/update/memory truth table。
3. D-FOU/RBF/WAV 多轮 substrate rows 有 step ratio 和 task-health 读数，但缺少同一协议下 forward-only、backward-only、optimizer update、functional overhead 的拆分。
4. LQ/Rational 也没有和当前 D-CHE/MLP 同协议的 fresh efficiency census。
```

因此 v16.3 必须新增 Line P：Efficiency Census。它不替代 functional experiment，但它是所有后续判断的前置读表。

---

# 3. Line P：basis-vs-same-param-MLP efficiency census

## 3.1 目标

Line P 回答：

$$
\boxed{
\text{每个 active carrier 与同参数量 MLP 相比，forward、backward、update、memory 到底差多少？}
}
$$

必须区分三种成本：

```text
1. base training cost：普通 AdamW/backprop 训练成本；
2. functional update cost：functional direction / split-gradient / projection / proximal / recovery 的额外成本；
3. audit readback cost：LineC / tail / AUC / calibration / horizon readback 的成本。
```

目前 v16.2 的 D-CHE 比 MLP 慢，部分来自 D-CHE-only audit pack，因此必须把 audit cost 从 training cost 中剥离，否则无法判断 basis 本身是否慢。

## 3.2 测试对象

```text
P0-MLP-same-param-reference
P1-D-CHE-current
P2-D-FOU-current
P3-D-RBF-FastKAN-current
P4-D-WAV-current
P5-Rational-current
P6-LQ-current-or-reanchored
P7-D-CHE-functional-path-top2
P8-MLP-functional-path-top2
```

所有候选必须尽量匹配：

```text
parameter_count within ±5%;
hidden_dim / depth protocol fixed;
batch_size = 8, 32, 128 where feasible;
train_size / eval_size fixed;
no CPU offload;
no audit metric direction;
no fake/proxy timing.
```

## 3.3 必须记录的 phase-level 指标

```text
forward_only_ms
backward_grad_ms
optimizer_update_ms
functional_direction_ms
functional_projection_ms
functional_commit_ms
linec_audit_ms
tail_calibration_audit_ms
horizon_readback_ms
step_total_ms

forward_peak_allocated_mb
backward_peak_allocated_mb
backward_peak_reserved_mb
optimizer_state_mb
param_bytes
grad_bytes
activation_saved_bytes
basis_activation_bytes
workspace_temp_bytes
functional_state_bytes
peak_incremental_mb

forward_ratio_vs_same_param_mlp
backward_ratio_vs_same_param_mlp
optimizer_update_ratio_vs_same_param_mlp
functional_overhead_ratio_vs_base_training
step_ratio_vs_same_param_mlp
backward_memory_ratio_vs_same_param_mlp
peak_incremental_ratio_vs_same_param_mlp
```

## 3.4 效率 gate

Exploration efficiency gate：

```text
forward_ratio <= 1.75
backward_ratio <= 1.75
update_ratio <= 1.75
backward_memory_ratio <= 1.50
step_ratio <= 1.75
```

Official efficiency gate：

```text
forward_ratio <= 1.25
backward_ratio <= 1.40
update_ratio <= 1.40
backward_memory_ratio <= 1.25
step_ratio <= 1.25
```

注意：Line P 不决定 functional promotion。它只告诉我们每条 carrier 是否有资格继续承载 expensive functional tests。

## 3.5 如果 Line P fail，Codex 必须继续做什么

如果某 basis 慢，不能只写 `efficiency fail`。必须继续拆：

```text
P-FB1: audit cost separation
  去掉 LineC/tail/AUC readback，重测 training-only step。

P-FB2: forward/backward/update decomposition
  定位慢在 basis eval、projection、split-gradient、optimizer update 还是 horizon readback。

P-FB3: memory component decomposition
  记录 params / grads / optimizer state / basis activations / functional state / audit buffers。

P-FB4: same-param MLP sanity
  确认 MLP reference 与 KAN 参数量、batch、hidden/depth一致。

P-FB5: checkpoint/resume throughput
  若 row-level checkpoint 造成大量 overhead，记录 checkpoint interval sensitivity。

P-FB6: efficiency no-go certificate
  只有 P-FB1..P-FB5 完成后，才允许写 efficiency blocked。
```

## 3.6 可视化

必须生成：

```text
basis_vs_mlp_forward_ratio_bar.svg
basis_vs_mlp_backward_ratio_bar.svg
basis_vs_mlp_update_ratio_bar.svg
basis_vs_mlp_backward_memory_ratio_bar.svg
phase_time_stacked_bar_by_basis.svg
memory_component_stacked_bar_by_basis.svg
functional_overhead_vs_source_scatter.svg
audit_overhead_fraction_by_carrier.svg
gpu_utilization_timeline.svg
```

---

# 4. Functional update 的核心实验矩阵

v16.3 不再让每个 carrier 只跑一个代表方案。它将 active carrier 分成三层：

```text
Tier 1 active carriers:
  D-CHE, MLP。

Tier 2 conditional carriers:
  LQ after weak/full reanchor;
  D-FOU / D-RBF after >=6/9 substrate gate;
  Rational monitor only unless no-regression + efficiency sufficient。

Tier 3 substrate-only carriers:
  D-WAV and blocked Non-RAT families。
```

每个 Tier 1 carrier 必须至少并行跑以下八类 mechanisms：

```text
M1 ProductivePulseRecovery
M2 SplitConsensusMetric
M3 PopRiskDriftDiffusion
M4 FunctionSpaceProximal
M5 OptimizerAwareAlignedFU
M6 CarrierReparameterizedFU
M7 LateAttachSnapshotFU
M8 RecoveryOnlyDeconfound
```

v16.2 已经跑过这些大类，但 v16.3 的区别是：**每个大类不只跑一个代表 row，而要跑多套互相独立的子方案，并且通过效率表来决定哪些方案值得长程 H=1600。**

---

# 5. Line A：D-CHE multi-scheme functional dynamics

D-CHE 是当前最强 KAN carrier，但 v16.2 已经显示最强 h800 source 没有转成 S2/S3。因此 D-CHE 不再只复跑 M4a/M4b，而是同时跑不同机制假设。

## 5.1 D-CHE schemes

```text
A1 PulseRecovery family:
  A1a pulse-once + AdamW recovery
  A1b pulse-once + LR cooldown
  A1c pulse-once + momentum damping
  A1d pulse-once + decoupled weight decay
  A1e pulse-once + EMA/SWA consolidation
  A1f pulse-every50 + AdamW recovery
  A1g early-only pulse
  A1h late-only pulse

A2 SplitConsensus family:
  A2a diagonal split-consensus metric
  A2b degree-role split-consensus metric
  A2c low-rank r4 split-consensus
  A2d low-rank r8 split-consensus
  A2e metric-only no projection
  A2f negative-eigen quarantine diagnostic

A3 PopRisk / drift-diffusion family:
  A3a parameter SNR preconditioner
  A3b degree-role SNR preconditioner
  A3c basis-channel SNR cover boundary
  A3d recovery-aware SNR scheduler

A4 Function-space proximal family:
  A4a Adam subspace proximal alpha grid
  A4b per-example low-rank proximal
  A4c output-Jacobian sketch proximal
  A4d split B1/B2 transfer proximal
  A4e recovery-aware proximal
  A4f control-residualized proximal

A5 Optimizer-aware family:
  A5a hard cautious FU
  A5b soft cautious FU
  A5c MGUP-style reweight
  A5d Adam second-moment scaled FU
  A5e SophiaDiagLite clipped FU
  A5f block second-moment FU
  A5g decoupled decay + FU

A6 Carrier reparameterization family:
  A6a low-degree signal / high-degree reservoir
  A6b readout-basis decoupled carrier
  A6c orthogonal degree bank
  A6d output-Jacobian carrier
  A6e dual-bank signal/reservoir carrier
  A6f high-degree quarantine carrier

A7 Late attach family:
  A7a early checkpoint attach
  A7b mid checkpoint attach
  A7c late checkpoint attach
  A7d loss plateau attach
  A7e high-source low-debt window attach

A8 Recovery-only deconfound:
  A8a AdamW recovery only
  A8b LR cooldown only
  A8c decay only
  A8d EMA/SWA only
  A8e Lookahead only
  A8f momentum damping only
```

## 5.2 D-CHE success criteria

Weak D-CHE progress：

```text
source_vs_best_control_h800 >= 0.005
source_retention_h800 >= 0.40
AUCtime_ratio_h800 <= 1.10
at least one of tail_recovery_rate_h800 or LineC_recovery_rate_h800 >= 0.40
matched controls fail
```

Meaningful D-CHE progress：

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

If D-CHE source positive but debt unrecovered, do not stop. Continue Line A failure ladder:

```text
A-FB1: debt class decomposition
  source / retention / tail / LineC / AUC / calibration failure count.

A-FB2: mechanism-family contrast
  identify whether M1, M2, M3, M4, M5, M6, M7, or M8 uniquely improves any dimension.

A-FB3: efficiency interaction
  join Line P efficiency rows to see whether slow mechanism also has better source.

A-FB4: h1600 eligibility
  top-2 by source-retention and top-2 by debt-recovery must run H=1600 even if not S2.

A-FB5: no-go certificate
  Only after A1..A8 + controls + h1600 top rows are complete.
```

---

# 6. Line B：MLP active functional dynamics

MLP 不是 control-only。它回答：functional dynamics 是否是 generic training phenomenon。

## 6.1 MLP schemes

MLP 必须跑与 D-CHE 同构的 M1-M8，但用 MLP hidden/logit carrier：

```text
B1 PulseRecovery family
B2 SplitConsensusMetric family
B3 PopRiskDriftDiffusion family
B4 FunctionSpaceProximal family
B5 OptimizerAwareAlignedFU family
B6 CarrierReparameterizedFU family, mapped to hidden/readout decomposition
B7 LateAttachSnapshotFU family
B8 RecoveryOnlyDeconfound family
```

## 6.2 MLP outcomes

MLP success has three interpretations:

```text
Case M-GenericSuccess:
  MLP and D-CHE both positive; controls fail.
  This is a generic functional dynamics insight, not KAN-specific.

Case M-KANCarrierLag:
  MLP positive, D-CHE fail.
  Functional update mechanism may be right, KAN carrier may be wrong.

Case M-UniversalNoGo:
  MLP and D-CHE both fail across M1-M8.
  Current functional dynamics family likely no-go.

Case M-KANSpecific:
  D-CHE positive, MLP/generic weak, difference-in-differences > 0.005.
  Only this supports KAN-specific claim.
```

MLP line cannot be skipped if D-CHE fails. If MLP has source but bad debt, it must receive the same H=1600 debt-recovery treatment.

---

# 7. Line C：LQ reanchor and late attach

LQ 历史上有 near-pass / repaired anchor 价值，但当前 reanchor 未打开。v16.3 继续 LQ，但不把 LQ 直接 promotion。

## 7.1 LQ stages

```text
C0 HistoricalLQReferenceReplay
C1 CurrentLQProtocolReplay
C2 ProtocolMatchedReanchor
C3 RowGateRobustReanchor
C4 MacroDeltaPriorityReanchor
C5 StepMemoryRecheck
C6 LineCNoRegressionCheck
C7 LateAttachFunctionalM1 if weak reanchor opens
C8 LateAttachFunctionalM2 if weak reanchor opens
C9 LQ proximal functional if full reanchor opens
C10 LQ pulse+recovery if full reanchor opens
```

Weak reanchor gate:

```text
near_pass >= 8/9
mean_delta_vs_MLP >= 0
step_ratio <= 1.75
memory_ratio <= 1.75
LineC_no_regression = 1
```

Full reanchor gate:

```text
near_pass = 9/9
mean_delta_vs_MLP >= 0.005
step_ratio <= 1.25
memory_ratio <= 1.25
LineC_pass_rate >= 0.80
```

If weak reanchor opens, run C7/C8 short-horizon. If full reanchor opens, run C9/C10 H=800.

---

# 8. Line D：Rational monitor

Rational remains useful as a monitor, not as a reset-route revival.

Allowed:

```text
RAT-AdamW monitor
RAT-G7 analog pulse monitor
RAT-split-consensus monitor
RAT-proximal monitor
RAT-efficiency census
```

Forbidden:

```text
optimizer-state reset route
K-RT/K-AUC/K-FL action family
controller
action bank
mask grid
seed-specific rescue
```

Rational only becomes active functional candidate if:

```text
same-param efficiency census passes exploration gate;
no-regression monitor >= 6/9;
functional monitor source_vs_best_control_h800 >= 0.005;
matched controls fail.
```

---

# 9. Line F：all-basis substrate repair and limited smoke

D-FOU/RBF/WAV cannot enter official functional proof before substrate gate, but v16.2 best family D-FOU reached 4/9, so this line must not be starved.

## 9.1 Families and goals

```text
D-FOU:
  low-frequency identity residual;
  bandwise SNR warmup;
  phase-stable band mix;
  no-materialize lifetime;
  high-frequency quarantine;
  efficiency census.

D-RBF/FastKAN:
  active center occupancy;
  width condition guard;
  compact bump no-dense materialization;
  Gaussian local K4 task-health;
  efficiency census.

D-WAV:
  triangular support;
  scale occupancy;
  support overlap damping;
  local-tail coverage audit;
  low-budget monitor.
```

Exploration substrate gate:

```text
family_dataset_seed_pass_count >= 6/9
step_ratio <= 1.75
memory_ratio <= 1.75
mean_delta_vs_MLP >= -0.05
worst_delta_vs_MLP >= -0.10
LineC_pass_rate >= 0.30
```

Official FU eligibility:

```text
family_dataset_seed_pass_count = 9/9
step_ratio <= 1.25
memory_ratio <= 1.25
mean_delta_vs_MLP >= -0.02
worst_delta_vs_MLP >= -0.05
LineC_pass_rate >= 0.80
```

If any family reaches >=6/9, run limited smoke:

```text
F-smoke-M1 ProductivePulseRecovery
F-smoke-M3 PopRiskDriftDiffusion
F-smoke-M4 FunctionSpaceProximal
```

No official FU proof until 9/9 substrate eligibility.

---

# 10. Line M：cross-line controls and attribution

For every positive-looking row:

```text
NoOpMatchedOverhead
RandomMatchedPulseSameRecovery
AdamWExtraStepsMatchedTime
RecoveryOnlyNoPulse
DecayOnly if decay involved
MLPAnalog if KAN line positive
D-CHEAnalog if MLP line positive
SameActiveFractionRandomPulse
SameNormRandomPulse
GenericOptimizerControl
```

Difference-in-differences:

$$
\Delta_{KAN-specific}
=
[(KAN+FU+Recovery)-(KAN+ControlRecovery)]
-
[(MLP+FU+Recovery)-(MLP+ControlRecovery)].
$$

Interpretation:

```text
Generic functional dynamics:
  MLP and KAN both positive, controls fail.

KAN-specific advantage:
  KAN positive, MLP/generic weaker, Delta_KAN-specific > 0.005, controls fail.

Control-equivalent:
  matched controls explain source/recovery; no promotion.
```

---

# 11. 4GPU execution schedule

Server has exactly four GPUs available. v16.3 must use all four unless a hard blocker is logged.

## Round 0: smoke + efficiency census bootstrap

```text
GPU0:
  Line P efficiency census for MLP + D-CHE.
  D-CHE smoke for A1/A2/A4.

GPU1:
  Line P efficiency census for MLP + MLP functional path.
  MLP smoke for B1/B2/B4.

GPU2:
  LQ C0..C6 reanchor + Rational monitor + their efficiency census.

GPU3:
  D-FOU/D-RBF/D-WAV substrate rows + their efficiency census.
```

## Round 1: short-horizon M1-M8 matrix

```text
GPU0:
  D-CHE M1/M2/M3/M4 short-horizon H=1/5/20/50/100.

GPU1:
  MLP M1/M2/M3/M4 short-horizon H=1/5/20/50/100.

GPU2:
  D-CHE M5/M6/M7/M8 + LQ functional if weak reanchor opens.

GPU3:
  MLP M5/M6/M7/M8 + all-basis substrate continuation.
```

## Round 2: H=800 expansion

```text
GPU0:
  Top D-CHE source-retaining rows to H=800.

GPU1:
  Top MLP source-retaining rows to H=800.

GPU2:
  Top D-CHE debt-recovery rows + LQ/Rational active candidates.

GPU3:
  all-basis top family >=4/9 focused substrate; if >=6/9, limited smoke.
```

## Round 3: H=1600 consolidation

```text
GPU0:
  D-CHE top-2 by source-retention and top-2 by debt-recovery.

GPU1:
  MLP top-2 by source-retention and top-2 by debt-recovery.

GPU2:
  LQ/Rational top candidates if active; otherwise D-CHE second mechanism family.

GPU3:
  all-basis substrate continuation + efficiency profiler completion.
```

## GPU utilization contract

```text
1. If a GPU is idle for >10 minutes while runnable_queue is non-empty, record execution_contract_violation = 1.
2. Every GPU job writes start_time, end_time, rows_planned, rows_completed, failure_reason.
3. Row-level checkpoint/resume is mandatory for A/B full surfaces.
4. Header-only artifact is not final coverage.
5. Budget-deferred item must include planned_budget, consumed_budget, defer_reason, next_priority.
6. cpu_offload_used must remain 0.
```

---

# 12. Success gates

## S1: execution coverage

```text
Line P efficiency census complete;
D-CHE M1-M8 complete;
MLP M1-M8 complete;
LQ C0-C6 complete;
Rational monitor complete;
D-FOU/RBF/WAV substrate rows complete;
controls complete;
required artifacts complete.
```

## S2: weak productive dynamics

```text
source_vs_best_control_h800 >= 0.005
source_retention_h800 >= 0.40
tail_recovery_rate_h800 >= 0.40 or LineC_recovery_rate_h800 >= 0.40
AUCtime_ratio_h800 <= 1.10
matched controls fail
```

## S3: productive debt recovery

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

## S4: real-transfer exploration

```text
real_lite_pass_count >= 6/9
source_vs_best_control_mean >= 0.005
AUCtime_median <= 1.05
tail debt recovered
LineC debt recovered
controls fail
```

## S5: official success, not lowered

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

# 13. Stop / continue rules

## Promotion fail-closed

No S5 means no promotion. No exceptions.

## Exploration continue-open

The following cannot hard-stop the whole run:

```text
D-CHE M1 fail;
D-CHE M4 fail;
MLP M1 fail;
MLP M4 fail;
LQ reanchor fail;
Rational monitor fail;
D-FOU/RBF/WAV <6/9;
M1 pulse + recovery fail;
M2 split-consensus fail;
M3 PopRisk/SNR fail;
M4 proximal fail;
M5 optimizer-aware fail;
M6 carrier reparameterization fail;
M7 late attach fail;
M8 recovery-only explains result;
MLP/generic controls positive;
LineC/tail/AUC immediate fail;
h800 fail;
h1600 fail;
overhead high;
random pulse explains result.
```

They must enter:

```text
failure taxonomy;
fallback ladder;
exhaustion certificate;
next hypothesis queue.
```

## True hard stop

Only these are hard stop:

```text
required_artifact_missing_count > 0;
forbidden_information_violation_count > 0;
no_action_search_violation_count > 0;
direction uses validation/test/future/query;
direction uses LineC/CEp99/NLL/ECE/AUCtime/Brier;
dataset-name branch;
seed-specific scale;
action token / controller / action bank / reset route;
fake / proxy / CPU offload.
```

---

# 14. Required artifacts

```text
v163_route_decision.json
v163_gpu_assignment_manifest.csv
v163_gpu_utilization_dashboard.csv
v163_efficiency_census.csv
v163_efficiency_census_component_breakdown.csv
v163_method_surface_manifest.csv
v163_line_a_dche_results.csv
v163_line_b_mlp_results.csv
v163_line_c_lq_reanchor.csv
v163_line_d_rational_monitor.csv
v163_line_f_allbasis_substrate.csv
v163_line_m_attribution.csv
v163_failure_taxonomy.csv
v163_h800_summary.csv
v163_h1600_summary.csv
v163_required_artifact_manifest.csv
v163_forbidden_information_audit.csv
v163_no_action_search_audit.csv
v163_direction_provenance.csv
v163_budget_exhaustion_certificate.csv
v163_deferred_items.csv
v163_code_review_packet.zip
```

---

# 15. Required visualizations

```text
basis_vs_mlp_forward_ratio_bar.svg
basis_vs_mlp_backward_ratio_bar.svg
basis_vs_mlp_update_ratio_bar.svg
basis_vs_mlp_backward_memory_ratio_bar.svg
phase_time_stacked_bar_by_basis.svg
memory_component_stacked_bar_by_basis.svg
source_retention_curve_horizon.svg
tail_debt_curve_horizon.svg
LineC_debt_curve_horizon.svg
calibration_debt_curve_horizon.svg
AUC_debt_curve_horizon.svg
D-CHE_vs_MLP_mechanism_matrix_heatmap.svg
KAN_specific_attribution_bar.svg
carrier_mechanism_outcome_matrix.svg
allbasis_substrate_progress_heatmap.svg
gpu_utilization_timeline.svg
budget_deferred_waterfall.svg
```

---

# 16. Code review requirements for Codex

Codex must write an implementation readback file explaining:

```text
1. Which files implement each carrier.
2. Which files implement each M1-M8 mechanism.
3. Which code path computes functional direction.
4. Which code path computes LineC/tail/AUC/calibration readback.
5. Evidence that audit metrics are not used as direction.
6. Evidence that no dataset-name / seed-specific branch exists.
7. Evidence that no action bank / controller / reset route was started.
8. Efficiency profiler code path and what each timing phase means.
9. Why any GPU was idle while runnable_queue was non-empty, if applicable.
10. Exact deferred items and whether they affect final route.
```

This readback is mandatory; no final route is valid without it.

---

# 17. Final interpretation rule

v16.3 can produce four possible outcomes:

```text
Outcome A: D-CHE productive, MLP not productive.
  Possible KAN-specific functional advantage; proceed to S4/S5 confirmation.

Outcome B: D-CHE and MLP both productive.
  Generic functional dynamics insight; valuable but not KAN-specific.

Outcome C: MLP productive, D-CHE not productive.
  Functional mechanism may be right; current KAN carrier is wrong.
  Focus on carrier/substrate redesign.

Outcome D: no carrier/mechanism reaches S2.
  Current train-stream functional dynamics family likely no-go.
  Stop token/action/controller/reset expansion; shift to base/substrate-level redesign or new theory of functional direction.
```

