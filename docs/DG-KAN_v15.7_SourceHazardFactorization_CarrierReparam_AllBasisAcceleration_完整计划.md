# DG-KAN v15.7：Source-Hazard Factorization / Carrier Reparameterization + All-Basis 并行加速完整计划

> 版本：v15.7 execution plan  
> 依据：v15.6 `StabilityNullspaceSplitConsensusFU AllBasisAcceleration` 真实结果复盘  
> 目标：不继续 G9/G10、N7/Q6、action bank、controller、reset route；把问题从“如何压低 bad event”提升为“source-carrying tangent 与 hazard-carrying tangent 是否能在某个合法 carrier / coordinate 中分离”。  
> 公式格式：Typora 友好，仅使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no active B-spline；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific scaling；no fake / proxy / CPU offload；functional direction 不使用 validation/test/future/query；LineC / CEp99 / NLL / ECE / AUCtime / Brier 只用于 audit/gate/failure taxonomy，不能生成方向。

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的总目标仍然是：

$$
\boxed{
\text{strict FC-PureKAN base} + \text{functional update}
>
\text{strict FC-PureKAN base} + \text{ordinary AdamW / backprop controls}
}
$$

这里的 functional update 不是 optimizer trick、不是 dataset-specific tuning，也不是某个局部 diagnostic score 的提升。最终 claim 必须同时满足：

```text
1. 表达力与任务不差于同规模 MLP / AdamW controls；
2. step / memory / runtime 仍在可比 envelope 内；
3. source gain 能 transfer 到 real 3x3，而不是只在 local probe 上 positive；
4. tail / LineC / calibration 不坏；
5. functional gain 不能被 RandomMatched / SameActiveFraction / MLP / generic optimizer controls 解释；
6. functional direction 必须来自 train-stream legal observables。
```

## 0.2 当前最新状态

v15.6 不是能力成功。它确认了 v15.5 的 G7 split-consensus metric-only signal 仍然存在：N0-G7R-Replay 的 `source_vs_best_control_mean = 0.46147918701171875`，`control_equivalent_fraction = 0.0`，Line M 仍有 `6/9` KAN-specific delta 为正。这说明 G7 signal 不是普通 generic/MLP optimizer artifact。

但 v15.6 更重要的结论是：**source 与 hazard 基本同线**。在修正 Line U 的 combined-hazard basis readback 后：

```text
source_hazard_colinear_rows = 63 / 63
mean_hazard_overlap = 0.956108654302264
mean_source_retention_after_null = 0.04575427500383249
```

这意味着：当前 D-CHE / G7R 参数坐标里，简单 null 掉 hazard tangent 会同时 null 掉 source。Q2/Q4/Q5 通过 constrained projection 把实际 source retention 拉回约 `0.60`，但 full-null retention 只有约 `0.036-0.058`，所以它们只是“保留 source”，并没有充分移除 hazard，最终 bad event 仍然不通过。N3/N5/N6 虽然降低 tail/LineC failure fraction，但 source 变成负数。

因此，v15.6 的核心结论不是“没有信号”，而是：

$$
\boxed{
\text{当前 D-CHE split-consensus source signal 与 instability tangent 强绑定。}
}
$$

## 0.3 当前各线完成度

| 线 | 当前完成度 | 判断 |
|---|---:|---|
| 代码 / provenance / finalizer 审计 | 99% | required / forbidden / no-action-search / contract 审计闭合 |
| Historical B320 / FHQ anchor | 85% frozen | 历史强，但 label-informed init 禁用，不能作为最终 official claim |
| Line C 几何审计 | 88% | 审计成熟；只能解释失败，不能生成方向 |
| PopRisk / FU / FMS infrastructure | 95% | per-example gradient、second moment、projection telemetry、controls 都成熟 |
| Current FU / FMS / proximal / ST-FU family | 0%-3% | v15.0-v15.3 基本 no-go |
| Split-consensus signal observability | 60% | Line S gate pass；G7R source signal 稳定存在 |
| Split-consensus source strength | 45%-50% | source positive 且 control-equivalent=0；是当前唯一真实正线索 |
| Source-hazard separability | 0%-5% | v15.6 显示 63/63 colinear，当前定义失败 |
| Stability-nullspace / constrained projection | 5%-10% | N/Q 执行完整，但不能同时保 source 与去 hazard |
| Train-stream bad-event proxy | 15%-20% | B6/B10 有读数，但 leaveout min AUC 仅 0.25，不可 gating |
| D-CHE substrate | 85% | 当前最强 Non-RAT carrier，历史 9/9 eligibility |
| D-CHE stable real-transfer FU | 10%-15% | G7R/G7S/N/Q 均未开 S4/S5 |
| Rational substrate | 85% | 稳定；reset / optimizer-state route 被 generic confound 打回 |
| D-FOU substrate | 20%-30% | v15.6 仍无 >=6/9；历史 6/9 未稳定复现 |
| D-RBF / FastKAN substrate | 20%-30% | workspace/telemetry 有线索，task-health 不稳 |
| D-WAV substrate | 15%-20% | 弱线索，低预算保留 |
| Non-D-CHE official FU proof | 0%-5% | 当前不可打开 |
| Official S5 functional success | 0% | 尚未达成 |
| 整体 next-gen MLP claim | 28%-36% | 有 signal，无 separability / stability；不能 promotion |

---

# 1. v15.6 独立分析

## 1.1 v15.6 有什么进展？

v15.6 的进展不是模型能力进展，而是定位进展。

它确认了三件事：

```text
1. G7R source signal 仍然存在，不是 v15.5 的 batch mismatch 偶然。
2. G7R source gain 不是 generic/MLP controls 完全解释的普通 optimizer artifact。
3. 当前 hazard-nullspace / constrained projection 不能把 source 与 hazard 分开。
```

这比 v15.5 更深入。v15.5 的问题是“有 source，但 bad event 高”；v15.6 进一步说明，bad event 不是简单 scalar trust 能压下去，也不是普通 hazard null 能切掉，因为 source tangent 和 hazard tangent 在当前参数坐标里几乎共线。

## 1.2 为什么仍然不能算成功？

v15.6 最终状态仍然是：

```text
route = R4-LineCOrTailDominated
minimum_success = S1-StabilityNullspaceObservable
promotion_allowed = 0
real_lite_pass_count = 0 / 9
source_vs_best_control_mean = 0.46147918701171875
bad_event_fraction = 1.0
line_b_best_proxy = B6-update_cosine_to_adam
line_b_gate_pass = 0
line_d_best_non_dche_dataset_seed_pass_count = 0 / 9
```

所以它没有 S2/S3/S4/S5。它只是达到 S1：说明 split-consensus / stability readback 可观测，而不是说明 update 成功。

## 1.3 当前真正 blocker

当前 blocker 不是：

```text
缺二阶矩；
缺 cautious alignment；
缺 decoupled weight decay；
缺 proximal solver；
缺 split-transfer objective；
缺 split-consensus signal；
缺 trust scalar；
缺 hazard-null projection；
缺 constrained projection；
缺 matched controls；
缺 fallback ladder。
```

当前 blocker 是：

$$
\boxed{
\text{当前 carrier / coordinate 中，source direction 与 hazard direction 几乎不可分。}
}
$$

更直白说：

```text
G7R 有油门；
但油门和失控方向绑在同一根轴上；
踩下去就有 source gain，同时产生 tail / LineC bad event；
刹车会把 source 一起刹掉。
```

因此，下一步不能继续：

```text
G9/G10；
N7/Q6；
action bank；
controller；
reset route；
trust scalar 网格；
按 dataset/seed fail pattern 设计规则；
用 LineC/CEp99/NLL/ECE/AUCtime 反推方向。
```

下一步必须问：

$$
\boxed{
\text{是否存在一个 legal carrier / coordinate，使 source 与 hazard 可分离？}
}
$$

---

# 2. v15.7 核心假设

v15.7 不再把重点放在“再稳定 G7R update”上，而是做 source-hazard factorization。

## H1：当前 colinearity 是真实 D-CHE carrier 问题

如果在不同 metric、不同 function-space readback、不同 train split 和不同 seeds 下，source-hazard overlap 都维持高位，则说明 D-CHE 的当前参数化让 source gain 与 instability 强绑定。

判定指标：

$$
\operatorname{Overlap}(u,V_h)=\frac{\|P_{V_h}u\|_{M_t}}{\|u\|_{M_t}+\epsilon}.
$$

如果：

```text
mean_hazard_overlap >= 0.85
source_retention_after_null <= 0.20
colinear_rows_fraction >= 0.80
```

则当前 D-CHE/G7R carrier 基本不适合继续做 nullspace stabilization。

## H2：当前 colinearity 是 metric / hazard proxy artifact

如果换到 output-Jacobian metric、degree-role metric、AdamV metric、function-space displacement metric 后，overlap 明显下降，则问题不是 G7R source 本身，而是当前 parameter metric / hazard tangent 定义错误。

通过条件：

```text
某一个 metric 下：
mean_hazard_overlap <= 0.70
source_retention_after_null >= 0.50
bad_event_proxy_leaveout_min_auc >= 0.55
```

## H3：carrier reparameterization 可以分离 source / hazard

如果 D-CHE 在当前坐标中不可分，不代表所有 carrier 都不可分。D-CHE 可能需要把 source bank 与 hazard-sensitive bank 解耦，让 split-consensus source 进入一个更稳定的子空间。

通过条件：

```text
reparameterized carrier 下：
mean_hazard_overlap <= 0.70
source_retention_after_null >= 0.50
G7-like source_vs_best_control_mean > 0
bad_event_fraction < N0-G7R baseline
```

## H4：如果 reparameterization 也失败，则需要 substrate/base-level 重设计

如果所有 legal carrier 都显示 source/hazard 共线，则 functional update 本身不是当前瓶颈，瓶颈是 substrate/base architecture。

此时 route 应写：

```text
R7-SourceHazardInseparableUnderCurrentSubstrates
```

并停止 G-token / N-token / Q-token 延长。

---

# 3. 总体实验结构

v15.7 分成九条线，并行执行：

```text
Line R: provenance / no-action-search / implementation readback
Line U2: source-hazard colinearity validation across metrics and spaces
Line K: carrier reparameterization / source-hazard factorization candidates
Line G: G7-like split-consensus metric update on factored carriers
Line B: bad-event proxy robustness, readback only
Line P: micro-horizon and path audit, readback only
Line D: all-basis substrate acceleration
Line M: MLP / generic controls
Line Z: route / no-go / next hypothesis / exhaustion certificates
```

---

# 4. Line R：代码 / provenance / no-action-search 审计

## 4.1 目标

确认 v15.7 没有回到 v9/v12 的老错误：找好动作、按局部 positive 扩 token、用 audit metric 反推方向。

## 4.2 必须记录

```text
required_artifact_manifest_rows
required_artifact_missing_rows
forbidden_information_audit_rows
forbidden_information_violation_sum
no_action_search_audit_rows
no_action_search_violation_sum
direction_provenance_rows
uses_validation_for_direction
uses_test_for_direction
uses_future_for_direction
uses_query_for_direction
uses_linec_for_direction
uses_cep99_for_direction
uses_nll_for_direction
uses_ece_for_direction
uses_auctime_for_direction
uses_dataset_name_branch
uses_seed_specific_scale
cpu_offload_used
fake_proxy_used
```

## 4.3 通过条件

```text
required_artifact_missing_rows = 0
forbidden_information_violation_sum = 0
no_action_search_violation_sum = 0
cpu_offload_used = 0
fake_proxy_used = 0
```

Line R 失败才允许 hard stop。

---

# 5. Line U2：source-hazard colinearity validation

## 5.1 目标

判断 v15.6 的 colinearity 是真实机制，还是 metric / readback artifact。

## 5.2 需要比较的 metric / space

```text
M0: Identity parameter metric
M1: AdamV diagonal metric
M2: DegreeRoleSecondMoment metric
M3: Output-Jacobian diagonal metric
M4: Low-rank output-Jacobian sketch metric
M5: Function-space displacement metric
```

## 5.3 需要比较的 update sources

```text
U0: N0-G7R-Replay
U1: Q2 constrained source-retaining update
U2: Q4 constrained source-retaining update
U3: Q5 constrained source-retaining update
U4: RandomSameSourceNorm control
U5: AdamW gradient control
U6: MLP matched analog control
```

## 5.4 指标

```text
hazard_overlap
source_retention_after_null
full_null_retention
constrained_retention
colinear_flag
source_vs_best_control_predicted
bad_event_proxy_score
metric_condition
projection_rank
positive_source_energy_fraction
hazard_energy_fraction
source_hazard_angle_deg
rolewise_overlap
low_degree_overlap
high_degree_overlap
readout_overlap
```

核心公式：

$$
h(u)=\frac{\|P_{V_h}u\|_{M_t}}{\|u\|_{M_t}+\epsilon}.
$$

$$
r(u)=\frac{\langle u-P_{V_h}u,u\rangle_{M_t}}{\|u\|_{M_t}^2+\epsilon}.
$$

## 5.5 Gate

探索 gate：

```text
exists metric/space with:
mean_hazard_overlap <= 0.70
source_retention_after_null >= 0.50
colinear_rows_fraction <= 0.50
```

如果不过，不 hard stop，进入 Line K carrier reparameterization。

---

# 6. Line K：carrier reparameterization / source-hazard factorization

## 6.1 目标

不是新增 G-token，也不是 action search，而是改变 carrier / coordinate，让 source 与 hazard 在参数空间不再强绑定。

## 6.2 预注册 carrier candidates

### K1：LowDegreeSignal / HighDegreeResidual Decoupled D-CHE

把 D-CHE degree 分成：

```text
signal bank: low / mid degree
residual-hazard bank: high degree
```

G7-like metric update 只在 signal bank 上形成主通道；high-degree bank 只允许 decoupled regularization / damping，不作为 value direction。

### K2：Readout-Basis Decoupled Carrier

把 readout 和 basis-role 更新分开：

```text
basis-role: split-consensus metric update
readout: ordinary AdamW / cautious AdamW only
```

目标：避免 G7 source 通过 readout 快速放大成 tail / LineC bad event。

### K3：Orthogonal Degree Bank Carrier

用 train-stream degree activation Gram 做一次固定正交化：

$$
G_d=\mathbb E_B[\phi_d(x)\phi_d(x)^T].
$$

构造 orthogonalized degree coordinate，只改变 coordinate，不使用 labels / audit metrics。

### K4：Output-Jacobian Carrier

不直接在 parameter degree coordinate 上定义 G7，而在低秩 output-Jacobian sketch 中定义 source direction，再求最小范数参数更新。

### K5：Dual-Bank Signal/Reservoir Carrier

构造两个同构 D-CHE bank：

```text
signal bank: split-consensus update
reservoir bank: decoupled damping / no value direction
```

两者使用固定 orthogonality / group separation，不根据 dataset/seed 分支。

## 6.3 Line K 只允许做什么

允许：

```text
carrier / coordinate reparameterization
source-hazard overlap readback
same G7-like update in new carrier
matched controls
```

禁止：

```text
K6/K7 临时新增
G9/G10
N7/Q6
action bank / controller / reset
用 audit metric 生成方向
```

## 6.4 Gate

Carrier exploration gate：

```text
mean_hazard_overlap <= 0.70
source_retention_after_null >= 0.50
source_vs_best_control_mean > 0
bad_event_fraction < N0-G7R bad_event_fraction
control_equivalent_fraction <= 0.50
```

如果所有 K1..K5 都失败，route：

```text
R5-CarrierFactorizationFail
```

但不能 hard stop whole experiment；Line D / M / Z 仍必须完成。

---

# 7. Line G：factored-carrier G7-like update

## 7.1 目标

只在 Line U2 或 Line K 通过 exploration gate 的 carrier 上执行 G7-like split-consensus metric update。

## 7.2 Methods

```text
G0-D-CHE-AdamW
G1-D-CHE-CautiousAdamW
G7R-original-replay
G7K1-LowDegreeSignalCarrier
G7K2-ReadoutBasisDecoupledCarrier
G7K3-OrthogonalDegreeBankCarrier
G7K4-OutputJacobianCarrier
G7K5-DualBankCarrier
```

这些不是 G9/G10。它们都是同一个 G7 mechanism 在不同 carrier / coordinate 下的实现。

## 7.3 Controls

```text
C0-NoOpMatchedOverhead
C1-RandomSubspaceSameRank
C2-RandomSameProjectionRetention
C3-SameActiveFractionRandomMask
C4-SameDegreeEnergyRandom
C5-AdamWParallelDirection
C6-MLP-SameSplitConsensusControl
C7-SameCarrierRandomDirection
```

## 7.4 Metrics

```text
real_lite_pass_count
source_vs_best_control_mean
control_equivalent_fraction
bad_event_fraction
tail_fail_fraction
linec_fail_fraction
AUCtime_median
CEp99_delta_mean
NLL_delta_mean
ECE_delta_mean
source_retention
hazard_overlap
projection_rejection_fraction
step_time_ratio
memory_ratio
```

## 7.5 Gate

Weak exploration：

```text
real_lite_pass_count >= 3/9
source_vs_best_control_mean > 0
bad_event_fraction <= 0.80
```

Meaningful exploration：

```text
real_lite_pass_count >= 4/9
source_vs_best_control_mean >= 0.005
control_equivalent_fraction <= 0.50
bad_event_fraction <= 0.60
```

S4 exploration：

```text
real_lite_pass_count >= 6/9
source_vs_best_control_mean >= 0.005
AUCtime_median <= 1.05
tail_fail_fraction < G7R_tail_fail_fraction
linec_fail_fraction < G7R_linec_fail_fraction
controls fail
```

S5 official 不降低。

---

# 8. Line B：train-stream bad-event proxy robustness

## 8.1 目标

v15.6 已显示 B6/B10 有读数但 leaveout 不稳。v15.7 只允许 Line B 做 readback，不允许它直接生成方向。

## 8.2 Proxies

```text
B1 split_loss_disagreement
B2 recovery_lag_h1/h2/h4
B3 logit_rms_drift
B4 entropy_collapse
B5 margin_p10_drift
B6 update_cosine_to_adam
B7 degree_energy_drift
B8 degree_entropy_collapse
B9 projection_retention_drift
B10 combined_proxy
```

## 8.3 Robustness tests

```text
leave-dataset-out
leave-seed-out
leave-method-out
leave-control-out
leave-carrier-out
leave-time-budget-out
```

## 8.4 Gate

Proxy can be used for reporting only unless:

```text
AUC_mean >= 0.70
leaveout_min_AUC >= 0.60
false_positive_rate_on_controls <= 0.25
Spearman_to_bad_event >= 0.30
```

Even if this gate passes, proxy still cannot generate direction in v15.7. It can only enable next-version precommit-safe gating design.

---

# 9. Line P：micro-horizon / path audit

## 9.1 目标

解释 carrier/factorization 后的 update 是否只是短期 B1 好，还是能在 B2/B3 train split 上保持。

## 9.2 Metrics

```text
B1_loss_delta_h1/h2/h4
B2_loss_delta_h1/h2/h4
B3_loss_delta_h1/h2/h4
micro_horizon_integral
recovery_lag
source_retention_horizon
hazard_proxy_horizon
control_equivalent_path_flag
```

## 9.3 Failure taxonomy

```text
P0-OK
P1-SourceKilled
P2-HazardNotRemoved
P3-MicroHorizonGoodRealBad
P4-ControlEquivalentPath
P5-OverheadDominates
P6-ProjectionKillsValue
P7-CarrierColinear
```

Line P 只做 audit，不生成方向。

---

# 10. Line D：all-basis substrate acceleration

## 10.1 目标

D-CHE 不能成为唯一 carrier。v15.7 继续推进 D-FOU / D-RBF / D-WAV，但不允许未过 substrate gate 就进入 official FU proof。

## 10.2 Candidates

```text
D-FOU72-LowFreqIdentityResidualV5
D-FOU73-BandwiseConsensusMetricV2
D-FOU74-PhaseStableBandMixV2
D-FOU75-NoMaterializeLifetimeV4
D-FOU76-HighFrequencyQuarantineV2

D-RBF70-ActiveCenterOccupancyV4
D-RBF71-WidthConditionGuardV4
D-RBF72-CompactBumpNoDenseV4
D-RBF73-GaussianLocalK4TaskHealthV2
D-RBF74-CenterSplitConsensusMetric

D-WAV61-TriangularSupportV5
D-WAV62-ScaleOccupancyV3
D-WAV63-SupportOverlapDampingV3
D-WAV64-LocalTailCoverageAuditV2

D-CHE-no-regression
Rational-no-regression
```

## 10.3 Substrate exploration gate

```text
family_dataset_seed_pass_count >= 6/9
step_ratio <= 1.75
memory_ratio <= 1.75
mean_delta_vs_MLP >= -0.05
worst_delta_vs_MLP >= -0.10
LineC_pass_rate >= 0.30
```

Official FU eligibility：

```text
family_dataset_seed_pass_count = 9/9
```

## 10.4 Failure taxonomy

```text
D1-task_delta_fail
D2-NLL_fail
D3-LineC_fail
D4-step_or_memory_fail
D5-occupancy_fail
D6-width_or_phase_fail
D7-true_non_reproducibility
D8-finalizer_or_gate_mismatch
```

---

# 11. Line M：MLP / generic controls

## 11.1 目标

任何 positive-looking update 都必须与 MLP/generic controls 比较，避免把 generic optimizer gain 写成 KAN-specific。

## 11.2 Controls

```text
MLP-AdamW
MLP-CautiousAdamW
MLP-MGUP
MLP-SameSplitConsensusMetric
MLP-SameTrustScalar
MLP-RandomSameSourceNorm
MLP-NoOpMatchedOverhead
```

## 11.3 KAN-specific delta

$$
\Delta_{KAN-specific}
=
(G_{KAN}-C_{KAN})-(G_{MLP}-C_{MLP}).
$$

Exploration positive only if:

```text
Delta_KAN_specific > 0
MLP/generic controls do not explain source gain
generic_control_explains_fraction <= 0.50
```

---

# 12. Line Z：route / no-go / exhaustion certificate

## 12.1 Route definitions

```text
S1-SourceSignalObservable:
  G7R source signal confirmed, controls do not fully explain.

S2-SourceHazardDecouplingObserved:
  source/hazard overlap falls below exploration gate in some legal metric/carrier.

S3-StabilizedFunctionalExplorationPositive:
  real_lite >=4/9 and bad_event <=0.60.

S4-RealTransferExplorationPositive:
  real_lite >=6/9 with controls failing.

S5-OfficialFunctionalSuccess:
  official 9/9 full gate.

R1-MetricArtifactColinearity:
  colinearity only disappears under invalid/proxy metric.

R2-SourceHazardColinearCurrentCarrier:
  current D-CHE carrier shows persistent high overlap.

R3-CarrierFactorizationFail:
  K1..K5 fail to reduce overlap or keep source.

R4-StabilizationKillsSource:
  hazard removed but source becomes <=0.

R5-HazardRetainedWithSource:
  source retained but bad_event remains high.

R6-AllBasisCarrierBlocked:
  non-D-CHE families fail substrate gate.

R7-SourceHazardInseparableUnderCurrentSubstrates:
  no legal metric/carrier separates source/hazard.
```

## 12.2 Exhaustion certificate

每条线必须写：

```text
planned_surface
executed_surface
missing_surface
budget_kind
planned_budget
consumed_budget
mandatory_executed
fallback_executed
deferred_items
deferred_reason
final_stop_allowed
```

禁止用 “budget exhausted” 掩盖浅尝辄止。只有 mandatory + fallback + certificate 完成，才允许 route no-go。

---

# 13. 加速执行计划

v15.7 必须并行执行，不再串行等待单线结果。

```text
GPU group 0:
  Line U2 + Line B readback

GPU group 1:
  Line K carrier factorization candidates K1/K2/K3

GPU group 2:
  Line K K4/K5 + Line G factored-carrier update

GPU group 3:
  Line D D-FOU / D-RBF substrate acceleration

GPU group 4 if available:
  Line D D-WAV + D-CHE/Rational no-regression + Line M controls
```

最低完成要求：

```text
Line R:
  all audits pass.

Line U2:
  all six metric/space readbacks for N0/G7R and top Q rows.

Line K:
  K1..K5 carrier factorization readback + at least top-3 update rows.

Line G:
  all passed carriers from U2/K executed against matched controls.

Line B:
  B1..B10 leaveout robustness table.

Line P:
  top-2 updates + controls micro-horizon audit.

Line D:
  D-FOU and D-RBF full rows; D-WAV low-budget; D-CHE/Rational no-regression.

Line M:
  MLP/generic controls for every positive-looking result.

Line Z:
  route, no-go, failure taxonomy, exhaustion certificate, next hypothesis queue.
```

---

# 14. 成功标准

## 14.1 S2 decoupling success

```text
mean_hazard_overlap <= 0.70
source_retention_after_null >= 0.50
source_vs_best_control_mean > 0
control_equivalent_fraction <= 0.50
```

## 14.2 S3 stabilized functional positive

```text
real_lite_pass_count >= 4/9
source_vs_best_control_mean >= 0.005
bad_event_fraction <= 0.60
tail_fail_fraction <= 0.60
linec_fail_fraction <= 0.60
controls fail
```

## 14.3 S4 real-transfer exploration

```text
real_lite_pass_count >= 6/9
source_vs_best_control_mean >= 0.005
AUCtime_median <= 1.05
tail fail reduced vs G7R
LineC fail reduced vs G7R
controls fail
```

## 14.4 S5 official success，不降低

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

# 15. Stop / continue 制度

## 15.1 Promotion fail-closed

S5 不到，永远不能 promotion。

## 15.2 Exploration continue-open

这些不能 hard stop：

```text
Line U2 shows colinearity；
Line K factorization fail；
Line G real-lite <4/9；
Line B proxy leaveout fail；
Line P micro-horizon fail；
Line D all-basis fail；
MLP/generic controls positive；
D-CHE real-lite <4/9；
LineC/tail/AUC fail；
overhead high；
source killed by hazard removal。
```

它们必须进入 failure taxonomy / fallback / exhaustion certificate。

## 15.3 真正 hard stop

只有以下情况可以 hard stop：

```text
required artifact missing；
forbidden information violation；
no-action-search violation；
direction uses validation/test/future/query；
direction uses LineC/CEp99/NLL/ECE/AUCtime/Brier；
dataset-name branch；
seed-specific scale；
action token / controller / action bank / reset route；
fake / proxy / CPU offload。
```

---

# 16. 必须生成的表格与图

## 16.1 CSV artifacts

```text
v157_line_r_audit.csv
v157_line_u2_source_hazard_colinearity.csv
v157_line_k_carrier_factorization_readback.csv
v157_line_g_factored_carrier_results.csv
v157_line_b_proxy_leaveout.csv
v157_line_p_micro_horizon_audit.csv
v157_line_d_allbasis_results.csv
v157_line_m_generic_controls.csv
v157_line_c_geometry_tail_audit.csv
v157_failure_taxonomy.csv
v157_route_decision.json
v157_exhaustion_certificate.csv
```

## 16.2 Figures

```text
fig_u2_hazard_overlap_by_metric.svg
fig_u2_source_retention_vs_overlap.svg
fig_k_carrier_factorization_pareto.svg
fig_g_source_vs_bad_event_pareto.svg
fig_b_proxy_leaveout_heatmap.svg
fig_p_micro_horizon_paths.svg
fig_d_allbasis_substrate_heatmap.svg
fig_m_kan_vs_mlp_specific_delta.svg
fig_failure_taxonomy_heatmap.svg
fig_route_dashboard.svg
```

---

# 17. 最终判断

v15.6 的关键不是失败，而是把问题推到更本质的位置：

$$
\boxed{
\text{source signal 已经存在；但在当前 D-CHE/G7R carrier 中，source 与 hazard 强绑定。}
}
$$

因此 v15.7 不再继续 trust scalar、nullspace 小修或 G-token 搜索，而是直接验证：

$$
\boxed{
\text{是否存在合法 metric / function-space / carrier reparameterization，让 source 与 hazard 可分离。}
}
$$

如果能分离，functional update 继续推进；如果不能分离，说明当前 D-CHE split-consensus signal 需要 substrate/base-level 重构，而不是继续在同一个参数坐标里做稳定化。 
