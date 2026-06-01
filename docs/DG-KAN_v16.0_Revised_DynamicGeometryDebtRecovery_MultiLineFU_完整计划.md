# DG-KAN v16.0 修正版：Dynamic Geometry / Debt Recovery / Multi-Line Functional Update + MLP Active + LQ Reanchor + All-Basis 并行加速完整计划

> 版本：v16.0 revised execution plan  
> 生成时间：2026-05-31  
> 依据：v15.4-v15.9 实验复盘、v12.5.1 Manifold-Channel Geometry 诊断、Deep Manifold / Generalization 文档启发  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`  
> 核心修正：不再把“好几何”定义为单步不坏；不再把短期 bad event 直接当 failure；functional update 必须从单步安全更新转成长期训练动力学控制。  
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

这里的“更好”不是某个局部指标变好，也不是某个 one-step source 变正，而是同时满足：

```text
1. base / substrate 效率接近 MLP；
2. functional update 的收益超过 AdamW / random / matched / generic optimizer controls；
3. 训练轨迹长期更好，而不是只看 single-step source；
4. 短期 tail / LineC / calibration / AUC debt 可以被后续训练动力学偿还；
5. 若 MLP-FU 也成功，必须写成 generic training-dynamics insight，不能写成 KAN-specific；
6. 若 KAN carrier 比 MLP 更强，才可以讨论 KAN-specific functional advantage；
7. 所有 positive-looking rows 都必须通过 matched controls、MLP controls、NoOp overhead controls 和 generic optimizer controls。
```

v15.9 的真实状态是：multi-line coverage 已经完成，但 D-CHE、MLP、LQ、Rational、all-basis 都没有打开 promotion。它不是 Codex 没跑；它说明当前预注册方法集合没有找到 productive functional dynamics。同时，v15.9 也暴露了一个制度问题：`h400` fail 不应该自动阻断 `h800/h1600` 的长期动力学诊断。v16.0 的核心制度改为：

$$
\boxed{
\text{promotion fail-closed，exploration dynamics-open。}
}
$$

也就是说，S5 仍严格，但 functional dynamics 不再因为 short-horizon bad event / `h400` fail 直接停止。

---

# 1. 各条线当前进展百分比与独立判断

| 线 | 当前完成度 | 判断 |
|---|---:|---|
| 代码 / provenance / no-action audit | 99% | 工程闭包强，v15.9 artifact / forbidden / no-action audit 已完整 |
| Historical B320 / FHQ anchor | 85% frozen | 历史强，但 label-informed init 禁用，不能作为 final official claim |
| D-CHE substrate | 85% | 当前最强 KAN carrier，历史 9/9 substrate eligibility |
| D-CHE static FU / G7R source | 45%-50% | source signal 曾稳定出现，但 bad debt 未偿还，不能 promotion |
| D-CHE dynamic recovery | 5%-10% | v15.8/v15.9 尚未证明 productive plasticity，v16.0 主线继续 |
| MLP functional dynamics | 20%-30% | B9 有 source，但 bad=1.0，h400 转负；必须作为主动机制发现线继续 |
| LQ reanchor / late-attach | 20%-30% | 历史有 near-pass 与 repaired anchor；当前 reanchor 未闭合，不能丢 |
| Rational substrate / monitor | 70%-80% substrate / 10% functional | 稳定 monitor；reset / optimizer-state route 被 generic confound 打回 |
| D-FOU substrate | 20%-30% | 历史 6/9，v15.9 0/9；需要 cross-version reconciliation 与 substrate repair |
| D-RBF / FastKAN substrate | 20%-30% | workspace/telemetry 有线索，task-health 不稳 |
| D-WAV substrate | 15%-20% | 弱线索，低预算保留 |
| Dynamic geometry definition | 10%-15% | 新主线：好几何从 static no-bad 改为 dynamic recovery / source retention |
| Decoupled weight decay / recovery dynamics | 10%-15% | 过去主要当 deconfound/control；现在升级为 possible recovery mechanism 之一 |
| Official S5 | 0% | 尚未达成 |
| 整体 next-gen MLP claim | 25%-33% | 有 source 信号和诊断积累，但 productive long-horizon dynamics 未成立 |

这个百分比不是“工程成熟度”，而是“科学 claim 离最终目标的距离”。工程和审计已经很成熟；真正缺的是长期 functional dynamics。

---

# 2. 本版最重要的反思：好几何定义需要修正

过去我们逐渐把“好几何”执行成短期硬门：

```text
LineC 不能坏；
tail 不能坏；
CEp99 / NLL / ECE 不能坏；
AUCtime 不能坏；
source 要立刻正；
controls 不能解释。
```

这些作为 final promotion gate 是合理的；但作为 early exploration gate 过于保守。一个真正有价值的 functional update 可能先制造扰动，再由后续训练吸收，从而进入更好的训练轨迹。

因此，本版把“好几何”从静态定义改成动态定义：

$$
\boxed{
\text{好几何不是局部不坏，}
\text{而是训练动力学能把真实 signal 推入可泛化通道，}
\text{并能偿还短期扰动产生的 debt。}
}
$$

更直观地说：

```text
好几何不是每一步都很平滑；
好几何是模型被推一下之后，能不能走到更好的地方。
```

v12.5.1 里已经把好几何定义为 train motion 更可预测 probe motion、真实信号进入 signal channel、噪声不泄漏进 signal channel。本版继承这个定义，但加入 recovery / debt-repayment 层。也就是说：短期 LineC/tail/AUC 坏化不再自动判死，而要看它是不是可偿还的 training debt。

---

# 3. 关键概念：training debt 与 recovery

一次 functional update 可能产生短期 debt：

```text
tail debt：困难样本 / 高损失样本暂时变坏；
LineC debt：几何暂时撕裂；
calibration debt：ECE / NLL / Brier 暂时变坏；
AUC debt：短期训练路径变慢；
source debt：source gain 短期出现但难以维持。
```

定义某种审计指标 $B(t)$，例如 tail risk 或 LineC badness。若在 functional pulse 后出现峰值 $B_{peak}$，长程 $H$ 后为 $B(t+H)$，则 debt recovery 可以定义为：

$$
Recovery_B(H)
=
1-
\frac{B(t+H)-B(t)}{B_{peak}-B(t)+\epsilon}.
$$

当 $Recovery_B(H)=1$，表示 debt 完全偿还；当 $Recovery_B(H)=0$，表示没有恢复；当小于 0，表示继续恶化。

同时定义 source retention：

$$
Retention_{source}(H)
=
\frac{Source(t+H)}{Source(t+1)+\epsilon}.
$$

真正有价值的 functional update 不要求 $H=1$ 时没有 bad event，而要求：

```text
1. Source(t+1) 或 Source(t+H) 为正；
2. Source retention 在 H 后仍高；
3. tail / LineC / calibration / AUC debt 在 H 后被偿还；
4. matched controls、random pulse、NoOp overhead、MLP/generic controls 不能解释。
```

---

# 4. 当前核心假设

## H1：短期 bad update 可能是 productive plasticity，而不是 damage

D-CHE / G7R 曾经出现 source-positive / control-resistant signal，但伴随严重 bad event。v15.6/v15.7 证明静态 source-hazard 分离失败；v15.8/v15.9 尚未证明长程偿还成功。v16.0 继续问：

$$
\boxed{
\text{G7R 或其 MLP/LQ analog 的 bad event 是否能通过后续训练动力学偿还？}
}
$$

不是问：

```text
这个 update 当下坏不坏？
```

而是问：

```text
这个 update 是否启动了更好的长期训练轨迹？
```

## H2：偿还机制不一定是 weight decay

Weight decay / decoupled decay 很重要，但它只是 recovery mechanism 之一。v16.0 同时测试：

```text
ordinary AdamW recovery；
decoupled weight decay；
role-wise / degree-wise decay；
high-degree damping；
LR cooldown；
cosine restart；
momentum damping；
second-moment adaptation；
EMA / Lookahead / SWA-like consolidation；
gradient-noise diffusion；
feature re-centering；
late consolidation。
```

Weight decay 的正确定位不是“唯一救法”，而是：

$$
\boxed{
\text{functional update 负责制造 useful plasticity，}
\text{recovery dynamics 负责偿还 instability debt。}
}
$$

其中 recovery dynamics 可能包含 weight decay，但不限于 weight decay。

## H3：MLP-FU 不能丢

MLP functional update 不能只当 control。它必须作为主动机制发现线，因为它回答：

```text
1. productive perturbation 是否是 generic training phenomenon；
2. bad update 是否能被普通 MLP training dynamics 偿还；
3. 如果 MLP 上成立而 KAN 上不成立，说明问题可能在 KAN carrier；
4. 如果 MLP 上也不成立，说明当前 functional dynamics definition 本身有问题。
```

因此 v16.0 把 MLP-FU 放在 Line B 主线，而不是附属 control。

## H4：LQ 线不能丢

LQ 历史上接近过 strict PureKAN near-pass，并承载过 functional diagnostic，但后续协议下 reanchor 不稳定。v16.0 不把 LQ 当主 carrier，但恢复为 reanchor + late-attach monitor：

```text
1. 先验证 LQ base 是否能重锚；
2. 若 reanchor 稳定，做 LQ functional pulse + recovery；
3. 若 reanchor 不稳定，输出协议漂移 / row-level gate fragility 诊断；
4. 不把未 reanchored 的 LQ functional 写成 promotion。
```

## H5：好几何的探索 gate 必须 dynamics-open

短期 LineC / tail / AUC / calibration bad event 不能自动 hard stop。它们只能：

```text
1. 关闭 promotion；
2. 触发 debt accounting；
3. 进入 long-horizon recovery audit。
```

不能：

```text
1. 用来生成方向；
2. 按 dataset/seed 写 branch；
3. 直接阻断 h800/h1600 诊断。
```

---

# 5. 总体实验结构

v16.0 分为十条线，必须并行推进，不允许等一条线失败后才写下一版。

```text
Line R: provenance / forbidden / no-action / budget audit
Line G: dynamic geometry definition and debt accounting shared surface
Line A: D-CHE functional pulse + multi-recovery long-horizon dynamics
Line B: MLP functional pulse + multi-recovery active dynamics
Line C: LQ reanchor + snapshot late-attach functional dynamics
Line D: Rational monitor + no-reset no-regression dynamics
Line E: recovery mechanism matrix, including but not limited to weight decay
Line F: all-basis substrate acceleration, D-FOU / D-RBF / D-WAV
Line M: cross-line controls and generic-vs-KAN-specific attribution
Line Z: route, no-go, exhaustion certificate, next queue
```

---

# 6. Line R：Provenance / forbidden / no-action / budget audit

Line R 的目标是防止三类老错误：

```text
1. action search：看到局部 positive 后新增 G-token/FU-token/action/controller；
2. audit-directed optimization：用 LineC/tail/AUC/calibration 反推方向；
3. shallow stop：主方法失败后不跑 fallback / long horizon / cross-line controls。
```

必须记录：

```text
required_artifact_manifest_rows
required_artifact_missing_count
forbidden_information_violation_count
no_action_search_violation_count
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
uses_brier_for_direction
uses_dataset_name_branch
uses_seed_specific_scale
cpu_offload_used
fake_proxy_used
budget_kind
planned_budget
consumed_budget
mandatory_executed
fallback_executed
deferred_items
deferred_reason
final_stop_allowed
exhaustion_certificate_present
```

真正 hard stop 只允许发生在：

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

其它失败全部不能 hard stop，只能进入 route / failure taxonomy / fallback ladder。

---

# 7. Line G：Dynamic Geometry / Debt Accounting Surface

Line G 是共享审计面，不生成方向。它将“好几何”拆成四层。

## 7.1 Transfer geometry

从 train stream 中拆分：

```text
B1 = update / pulse batch
B2 = probe train split
Q  = held-out train stream probe, not validation/test
```

记录：

```text
train_motion_norm_B1
probe_motion_norm_B2
train_probe_coupling_R2
train_probe_coupling_cosine
source_vs_best_control_B2
source_vs_best_control_Q
```

## 7.2 Signal-channel geometry

记录：

```text
signal_channel_energy
reservoir_energy
noise_leakage_proxy
signal_to_reservoir_ratio
noise_to_signal_ratio
split_consensus_snr
population_risk_snr
```

这些指标只做 audit，不做方向。

## 7.3 Recovery geometry

对每个 pulse 记录多个 horizon：

```text
H = 1, 5, 20, 50, 100, 400, 800, 1600
```

每个 horizon 记录：

```text
source_vs_best_control_H
tail_debt_H
LineC_debt_H
calibration_debt_H
AUC_debt_H
tail_recovery_rate_H
LineC_recovery_rate_H
calibration_recovery_rate_H
AUC_recovery_rate_H
source_retention_H
```

## 7.4 Carrier geometry

对 D-CHE、MLP、LQ、Rational、D-FOU、D-RBF、D-WAV 记录 carrier-specific 状态：

```text
carrier_family
carrier_candidate_id
carrier_substrate_gate
basis_or_hidden_energy
source_hazard_overlap
source_retention_after_hazard_null
rolewise_debt_contribution
high_degree_energy_fraction
readout_basis_coupling
active_center_occupancy, for RBF
bandwise_energy, for Fourier
scale_occupancy, for Wavelet
```

## 7.5 Dynamic Geometry Score, audit-only

定义一个只用于排序与可视化的动态几何分数：

$$
DGS(H)
=
SourceRetention(H)
-
\lambda_t Debt_{tail}(H)
-
\lambda_c Debt_{LineC}(H)
-
\lambda_e Debt_{calibration}(H)
-
\lambda_a Debt_{AUC}(H).
$$

注意：

```text
DGS 不能生成 update 方向；
DGS 不能用于 dataset/seed branch；
DGS 只能作为 audit / route / visualization。
```

---

# 8. Line A：D-CHE functional pulse + multi-recovery dynamics

## 8.1 目标

D-CHE 是当前最强 KAN carrier。Line A 不再试图让 G7R 当下安全，而是验证：

$$
\boxed{
\text{D-CHE split-consensus pulse 是否能产生可偿还 debt，并保留长期 source？}
}
$$

## 8.2 Candidates

固定 functional pulse，不新增 G-token：

```text
A0-D-CHE-AdamW
A1-D-CHE-G7R-PulseOnce-then-AdamWRecovery
A2-D-CHE-G7R-PulseEvery50-then-AdamWRecovery
A3-D-CHE-G7R-PulseEarlyOnly-then-AdamWRecovery
A4-D-CHE-G7R-PulseMidOnly-then-AdamWRecovery
A5-D-CHE-G7R-PulseLateOnly-then-AdamWRecovery
A6-D-CHE-G7R-PulseOnce-then-MomentumDampedRecovery
A7-D-CHE-G7R-PulseOnce-then-SecondMomentAdaptRecovery
A8-D-CHE-G7R-PulseOnce-then-LRCooldownRecovery
A9-D-CHE-G7R-PulseOnce-then-EMALookaheadRecovery
A10-D-CHE-G7R-PulseOnce-then-DecoupledDecayRecovery
A11-D-CHE-G7R-PulseOnce-then-RoleWiseDecayRecovery
A12-D-CHE-G7R-PulseOnce-then-HighDegreeDecayRecovery
A13-D-CHE-G7R-PulseOnce-then-SWAConsolidation
```

Controls：

```text
ACTRL0-D-CHE-NoOpMatchedOverhead
ACTRL1-D-CHE-RandomMatchedPulse-then-AdamWRecovery
ACTRL2-D-CHE-RandomMatchedPulse-then-SameRecovery
ACTRL3-D-CHE-AdamWExtraStepsMatchedTime
ACTRL4-D-CHE-DecayOnlyRecovery
ACTRL5-D-CHE-RecoveryOnlyNoPulse
ACTRL6-D-CHE-SamePulseNormRandomDirection
```

## 8.3 Required horizons

每个 method 必须记录：

```text
H = 1, 5, 20, 50, 100, 400, 800
```

Top-2 source-retaining candidates 必须继续：

```text
H = 1600
```

`h400` fail 不允许自动 defer `h800`；`h400` 只能关闭 promotion，不能关闭 dynamics diagnosis。

## 8.4 Metrics

```text
source_vs_best_control_h1/h5/h20/h50/h100/h400/h800/h1600
tail_debt_peak
tail_debt_final
tail_recovery_rate_H
LineC_debt_peak
LineC_debt_final
LineC_recovery_rate_H
calibration_debt_peak
calibration_debt_final
AUC_debt_peak
AUC_debt_final
source_retention_H
DGS_H
pulse_norm
recovery_update_norm
recovery_to_pulse_norm_ratio
recovery_cosine_to_pulse
recovery_cosine_to_adam
step_time_ratio_H
memory_ratio_H
```

## 8.5 Exploration success

S2 productive plasticity：

```text
source_vs_best_control_h800 >= 0.005
tail_recovery_rate_h800 >= 0.60
LineC_recovery_rate_h800 >= 0.60
AUCtime_ratio_h800 <= 1.05
controls fail
```

S3 D-CHE productive dynamics：

```text
real_lite_pass_count >= 4/9
source_retention_h800 >= 0.50
tail_debt_final <= 0.40 * tail_debt_peak
LineC_debt_final <= 0.40 * LineC_debt_peak
random pulse + same recovery does not explain
recovery-only does not explain
```

S4 D-CHE real-transfer exploration：

```text
real_lite_pass_count >= 6/9
source_vs_best_control_mean >= 0.005
AUCtime_median <= 1.05
tail debt recovered
LineC debt recovered
controls fail
```

S5 official 不降低，见第 18 节。

## 8.6 Fallback ladder

如果 A-line fails，不允许直接 no-go，必须执行：

```text
A-FB1: classify damage vs debt: does any horizon recover?
A-FB2: compare recovery mechanisms: AdamW / decay / EMA / cooldown / momentum damping / SWA
A-FB3: source retention vs debt recovery Pareto
A-FB4: random pulse + same recovery deconfound
A-FB5: MLP analog comparison
A-FB6: D-CHE exhaustion certificate
```

---

# 9. Line B：MLP active functional dynamics

## 9.1 目标

MLP-FU 不能只作为 control。Line B 是主动机制发现线，回答：

$$
\boxed{
\text{functional pulse + recovery dynamics 是否在普通 MLP 上成立？}
}
$$

若 MLP 成功而 D-CHE 不成功，说明：

```text
functional dynamics 可能是 generic 机制；
KAN carrier / D-CHE coordinate 没承载好。
```

若 MLP 也失败，说明当前 functional dynamics definition 本身有问题。

## 9.2 MLP functional pulse candidates

```text
B0-MLP-AdamW
B1-MLP-SplitConsensusHiddenMetricPulseOnce-then-AdamWRecovery
B2-MLP-SplitConsensusHiddenMetricPulseEvery50-then-AdamWRecovery
B3-MLP-FMS-AmortizedPulseOnce-then-AdamWRecovery
B4-MLP-PopRiskSNRPulseOnce-then-AdamWRecovery
B5-MLP-PulseOnce-then-MomentumDampedRecovery
B6-MLP-PulseOnce-then-SecondMomentAdaptRecovery
B7-MLP-PulseOnce-then-LRCooldownRecovery
B8-MLP-PulseOnce-then-EMALookaheadRecovery
B9-MLP-PulseOnce-then-DecoupledDecayRecovery
B10-MLP-PulseOnce-then-SWAConsolidation
```

Controls：

```text
BCTRL0-MLP-NoOpMatchedOverhead
BCTRL1-MLP-RandomMatchedPulse-then-SameRecovery
BCTRL2-MLP-AdamWExtraStepsMatchedTime
BCTRL3-MLP-RecoveryOnlyNoPulse
BCTRL4-MLP-DecayOnly
BCTRL5-MLP-SameActiveFractionRandomPulse
```

## 9.3 Required horizons

同 Line A：

```text
H = 1, 5, 20, 50, 100, 400, 800
Top-2 source-retaining candidates -> H=1600
```

## 9.4 Metrics

记录与 Line A 相同的 dynamic debt / source retention metrics，并额外记录：

```text
hidden_rank_H
feature_center_drift_H
hidden_norm_H
class_centroid_separation_H
activation_sparsity_H
MLP_FU_specific_delta_vs_controls
```

## 9.5 Interpretation

```text
MLP success + D-CHE fail:
  Generic dynamics exists, KAN carrier is blocker.

MLP fail + D-CHE source-only fail:
  Current functional dynamics family likely no-go.

MLP success + D-CHE stronger:
  Potential KAN-specific carrier advantage.

MLP success only through decay-only/control:
  Not functional update success.
```

---

# 10. Line C：LQ reanchor + snapshot late-attach dynamics

## 10.1 目标

LQ 不应该被丢。它历史上有 near-pass / repaired anchor，但当前协议下 reanchor 不稳定。Line C 先 reanchor，再做 snapshot late-attach functional dynamics。

## 10.2 Reanchor candidates

```text
C0-HistoricalLQReferenceReplay
C1-CurrentLQProtocolReplay
C2-LQ-ProtocolMatchedReanchor
C3-LQ-RowGateRobustReanchor
C4-LQ-MacroDeltaPriorityReanchor
C5-LQ-StepMemoryRecheck
C6-LQ-LineCNoRegressionCheck
```

Reanchor exploration gate：

```text
near_pass >= 7/9
macro_delta >= -0.005
step_ratio <= 1.25
memory_ratio <= 1.25
LineC_no_regression = 1
```

Official LQ functional eligibility：

```text
near_pass >= 8/9
macro_delta >= -0.004
step_ratio <= 1.15
memory_ratio <= 1.15
LineC_no_regression = 1
```

## 10.3 Late-attach functional, only if reanchor opens

If C reanchor opens, run:

```text
C7-LQ-SnapshotFunctionalPulse-then-AdamWRecovery
C8-LQ-SnapshotFunctionalPulse-then-DecoupledDecayRecovery
C9-LQ-SnapshotFunctionalPulse-then-EMALookaheadRecovery
C10-LQ-SnapshotFunctionalPulse-then-SWAConsolidation
CCTRL-RandomPulseSameRecovery
```

If reanchor fails, do not run C7-C10 official functional; instead write:

```text
R-C-LQReanchorStillBlocked
```

and output protocol-drift analysis.

## 10.4 Required diagnostics

```text
historical_current_macro_delta
historical_current_nearpass_delta
row_level_flip_table
protocol_mismatch_mode
MLP_match_definition_drift
LQ_step_memory_drift
LQ_LineC_drift
snapshot_attach_eligibility
```

---

# 11. Line D：Rational monitor / no-regression dynamics

Rational remains useful as monitor but must not restart reset/controller/action route.

Run:

```text
D0-RAT-AdamWMonitor
D1-RAT-G7AnalogPulseOnce-then-AdamWRecovery, diagnostic only
D2-RAT-G7AnalogPulseOnce-then-DecayRecovery, diagnostic only
D3-RAT-NoRegressionCheck
DCTRL-RandomPulseSameRecovery
```

Hard constraints:

```text
no reset route;
no optimizer-state transport promotion;
no controller;
no action bank;
no K-RT / K-AUC / K-FL revival.
```

Metrics:

```text
rational_den_p01
r_prime_p99
r_double_prime_p99
group_diversity
source_retention_H
debt_recovery_H
RAT_vs_DCHE_dynamics_delta
```

---

# 12. Line E：Recovery mechanism matrix

Line E compares recovery mechanisms across D-CHE and MLP. It is not a direction generator.

## 12.1 Recovery families

```text
E0-AdamWRecovery
E1-GlobalDecoupledWeightDecayRecovery
E2-RoleWiseDecayRecovery
E3-DegreeWiseDecayRecovery, D-CHE only
E4-HighDegreeExtraDecayRecovery, D-CHE only
E5-ReadoutBasisDecoupledDecayRecovery, KAN only
E6-LRCooldownRecovery
E7-CosineRestartRecovery
E8-MomentumDampedRecovery
E9-SecondMomentAdaptRecovery
E10-EMALookaheadRecovery
E11-SWAConsolidationRecovery
E12-GradientNoiseDiffusionRecovery
E13-FeatureRecenteringRecovery
```

## 12.2 Decay is not automatically success

For every decay candidate, must compare:

```text
FU pulse + decay recovery
decay-only recovery
random pulse + same decay recovery
NoOp + same decay recovery
MLP analog same decay
```

If decay-only explains gain, write:

```text
R-E-DecayOnlyExplainsGain
```

If FU + decay works but random pulse + same decay also works, write:

```text
R-E-RecoveryExplainsButFUSpecificityFail
```

If FU + decay works, decay-only fails, random pulse fails, MLP analog fails or weaker, then write:

```text
S3-RecoveryHarnessedFunctionalDynamics
```

## 12.3 Metrics

```text
recovery_family
source_retention_h100/h400/h800/h1600
tail_debt_recovery_h100/h400/h800/h1600
LineC_debt_recovery_h100/h400/h800/h1600
calibration_debt_recovery_h100/h400/h800/h1600
recovery_update_norm
recovery_update_cosine_to_pulse
recovery_update_cosine_to_adam
decay_update_norm, if decay
fu_update_norm
decay_to_fu_norm_ratio
decay_vs_fu_cosine
```

---

# 13. Line F：All-basis substrate acceleration

D-CHE cannot be the only carrier. Line F continues basis substrate repair.

## 13.1 Families

```text
D-FOU: low-frequency identity residual, bandwise SNR, phase-stable band mix, high-frequency quarantine
D-RBF / FastKAN: active center occupancy, width condition, compact bump no-dense materialization, Gaussian local K4 task-health
D-WAV: triangular support, scale occupancy, support overlap damping, local-tail coverage audit
D-CHE: no-regression
Rational: no-regression monitor
LQ: reanchor monitor in Line C
```

## 13.2 Exploration substrate gate

```text
family_dataset_seed_pass_count >= 6/9
step_ratio <= 1.75
memory_ratio <= 1.75
mean_delta_vs_MLP >= -0.05
worst_delta_vs_MLP >= -0.10
LineC_pass_rate >= 0.30
```

## 13.3 Official FU eligibility

```text
family_dataset_seed_pass_count = 9/9
step_ratio <= 1.25
memory_ratio <= 1.25
mean_delta_vs_MLP >= -0.02
worst_delta_vs_MLP >= -0.05
LineC_pass_rate >= 0.80
```

If D-FOU / D-RBF / D-WAV do not pass exploration gate, they cannot enter official FU proof, but failure cannot hard-stop D-CHE / MLP / LQ dynamic lines.

---

# 14. Line M：Cross-line controls and attribution

Line M determines whether a positive result is KAN-specific, generic, or control-equivalent.

For every positive-looking result from Line A/B/C/D/E, run:

```text
NoOpMatchedOverhead
RandomMatchedPulseSameRecovery
AdamWExtraStepsMatchedTime
RecoveryOnlyNoPulse
DecayOnly, if decay involved
MLPAnalog, if KAN line positive
D-CHEAnalog, if MLP line positive
SameActiveFractionRandomPulse
SameNormRandomPulse
GenericOptimizerControl
```

Difference-in-differences estimate:

$$
\Delta_{KAN-specific}
=
[(KAN+FU+Recovery)-(KAN+ControlRecovery)]
-
[(MLP+FU+Recovery)-(MLP+ControlRecovery)].
$$

Generic functional dynamics success:

```text
MLP and KAN both positive;
controls fail;
KAN-specific delta not required.
```

KAN-specific functional advantage:

```text
KAN positive;
MLP/generic weaker;
Delta_KAN-specific > 0.005;
controls fail.
```

Control-equivalent:

```text
matched controls explain source / recovery;
No promotion.
```

---

# 15. Required visualizations

v16.0 必须生成以下图，不允许只写 CSV。

## 15.1 Dynamic debt dashboard

```text
source_retention_curve_horizon.svg
tail_debt_curve_horizon.svg
LineC_debt_curve_horizon.svg
calibration_debt_curve_horizon.svg
AUC_debt_curve_horizon.svg
recovery_rate_by_mechanism.svg
```

## 15.2 D-CHE vs MLP dynamics comparison

```text
D-CHE_vs_MLP_source_retention.svg
D-CHE_vs_MLP_tail_recovery.svg
D-CHE_vs_MLP_LineC_recovery.svg
D-CHE_vs_MLP_DGS_horizon.svg
```

## 15.3 Recovery mechanism Pareto

```text
source_retention_vs_tail_recovery.svg
source_retention_vs_LineC_recovery.svg
DGS_vs_step_time.svg
recovery_norm_vs_source_retention.svg
```

## 15.4 LQ reanchor dashboard

```text
LQ_historical_current_row_flip.svg
LQ_macro_delta_vs_nearpass.svg
LQ_protocol_drift_heatmap.svg
```

## 15.5 All-basis substrate dashboard

```text
basis_family_pass_count_heatmap.svg
basis_step_memory_pareto.svg
basis_LineC_task_health.svg
```

## 15.6 Route dashboard

```text
line_gate_status.svg
failure_taxonomy_heatmap.svg
exhaustion_certificate_dashboard.svg
```

---

# 16. Success definitions

## 16.1 S1：multi-line coverage completed

```text
Line R complete;
Line G complete;
Line A D-CHE dynamics complete;
Line B MLP active dynamics complete;
Line C LQ reanchor complete;
Line D Rational monitor complete;
Line E recovery matrix complete;
Line F all-basis substrate complete;
Line M controls complete;
Line Z route + exhaustion certificate complete.
```

## 16.2 S2：productive plasticity evidence

At least one carrier/method satisfies:

```text
source_vs_best_control_h800 >= 0.005
source_retention_h800 >= 0.50
tail_recovery_rate_h800 >= 0.60
LineC_recovery_rate_h800 >= 0.60
AUCtime_ratio_h800 <= 1.05
matched controls fail
```

## 16.3 S3：recovery mechanism established

At least one recovery mechanism satisfies:

```text
source retained;
debt recovered;
recovery-only does not explain;
random pulse + same recovery does not explain;
NoOp overhead does not explain;
MLP/KAN attribution resolved.
```

## 16.4 S4：real-transfer exploration

```text
real_lite_pass_count >= 6/9
source_vs_best_control_mean >= 0.005
AUCtime_median <= 1.05
tail debt recovered
LineC debt recovered
controls fail
```

## 16.5 S5：official success, not lowered

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

# 17. Stop / continue policy

## 17.1 Promotion fail-closed

No S5, no promotion. No exceptions.

## 17.2 Exploration dynamics-open

这些情况不能 hard stop：

```text
h1/h5/h20/h100 bad event high;
h400 fail;
h800 fail;
Line A fail;
Line B fail;
Line C reanchor fail;
Line E recovery-only explains gain;
Line F all-basis fail;
MLP/generic controls positive;
D-CHE real-lite < 4/9;
LineC/tail/AUC immediate fail;
overhead high;
random pulse explains result.
```

它们必须进入：

```text
failure taxonomy;
fallback ladder;
exhaustion certificate;
next hypothesis queue.
```

## 17.3 Budget exhaustion certificate

预算耗尽不能成为偷停理由。只有写出下列字段，才允许 stop：

```text
budget_kind: gpu_time / wall_clock / rows / seeds / horizon / fallback_depth / family_count
planned_budget
consumed_budget
mandatory_executed
fallback_executed
deferred_items
deferred_reason
whether_deferred_items_affect_route
next_priority_queue
final_stop_allowed
```

If `mandatory_executed=0` or `fallback_executed=0`, final stop is invalid unless forbidden/hard-stop violation occurred.

---

# 18. Parallel execution plan

v16.0 必须并行跑，不允许 D-CHE 失败后才跑 MLP / LQ / all-basis。

```text
GPU group 0:
  Line A D-CHE pulse + recovery, H=1..800.

GPU group 1:
  Line B MLP functional dynamics, H=1..800.

GPU group 2:
  Line C LQ reanchor + Line D Rational monitor.

GPU group 3:
  Line E recovery mechanism matrix for D-CHE + MLP.

GPU group 4:
  Line F all-basis substrate: D-FOU / D-RBF / D-WAV.

GPU group 5 if available:
  H=1600 extension for top-2 source-retaining A/B/E candidates.
```

最低完成合同：

```text
Line A:
  A0..A13 + controls, H=1/5/20/50/100/400/800.

Line B:
  B0..B10 + controls, H=1/5/20/50/100/400/800.

Line C:
  C0..C6 reanchor; if opened, C7..C10 functional dynamics.

Line D:
  D0..D3 Rational monitor.

Line E:
  E0..E13 recovery mechanisms for D-CHE and MLP top candidates.

Line F:
  D-FOU and D-RBF full substrate rows; D-WAV low-budget; D-CHE/Rational no-regression.

Line M:
  matched controls for every positive-looking result.

Line G/C/Z:
  dynamic debt metrics, geometry/tail/audit curves, route, no-go, exhaustion certificate.
```

---

# 19. Codex failure-handling instructions

Codex 不允许：

```text
1. 主方法失败后直接 final no-go；
2. h400 fail 后直接 defer h800；
3. 只跑 D-CHE，不跑 MLP active line；
4. 把 MLP 只当 control；
5. 跳过 LQ reanchor；
6. 跳过 all-basis substrate；
7. 新增 G-token / FU-token / F-CHE token；
8. 启动 action bank / controller / reset route；
9. 用 LineC / CEp99 / NLL / ECE / AUCtime / Brier 生成方向；
10. 按 dataset / seed fail pattern 写 branch；
11. 把 short-horizon positive 写成 S5；
12. 把 decay-only / recovery-only success 写成 functional update success。
```

Codex 必须：

```text
1. 同时执行 D-CHE、MLP、LQ、Rational、all-basis；
2. 对每个 pulse 记录 horizon curves；
3. 对 bad event 写 debt accounting，而不是 immediate fail-only；
4. 对 every positive-looking row 执行 controls；
5. 对每个未跑项写 budget-deferred certificate；
6. 输出 route 和 next hypothesis queue；
7. 在复盘中写明重要实现代码位置和实现语义。
```

---

# 20. 最终判断

v16.0 的核心不是再找一个 functional update 小技巧。它把问题重新定义为：

$$
\boxed{
\text{functional update 是否能制造有用扰动，}
\text{并被训练动力学恢复 / 巩固成长期收益？}
}
$$

这版同时保留五条关键线：

```text
D-CHE：当前最强 KAN carrier；
MLP：generic functional dynamics 主动机制发现线；
LQ：历史 carrier reanchor / late attach；
Rational：稳定 monitor；
D-FOU / D-RBF / D-WAV：all-basis substrate repair。
```

最重要的制度修正是：

$$
\boxed{
\text{早期允许坏；长期必须好；最终必须打过 controls。}
}
$$

如果 v16.0 仍然证明 D-CHE、MLP、LQ、Rational 和 all-basis 都无法形成 debt-repaying productive dynamics，那么应该写：

```text
R16-CurrentFunctionalDynamicsFamilyNoGo
```

并停止当前 train-stream functional-update family，而不是继续新增 token / action / controller。
