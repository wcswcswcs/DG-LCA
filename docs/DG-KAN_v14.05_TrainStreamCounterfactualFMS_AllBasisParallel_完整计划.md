# DG-KAN v14.5：Train-Stream Counterfactual FMS Controller + All-Basis Parallel Substrate 完整计划

> 版本：v14.5 execution plan  
> 日期：2026-05-29  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no active B-spline budget；no teacher；no distillation；no sampler / class weight；no dataset-name branch；no validation / test / future / query-batch direction；no label-informed initialization；LineC / CEp99 / NLL / ECE / AUCtime / Brier 只能作为 audit / gate，不能作为 direction source；不把 S4b 写成 S5；不把 MLP generic positive 写成 KAN-specific success。

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是找到一个只在 synthetic 或单个 seed 上好的更新技巧。项目目标是：

$$
\boxed{
\text{在 label-free strict FC-PureKAN / basis substrate 上，通过 functional update 得到比普通 AdamW / backprop 更好的训练过程与模型。}
}
$$

“更好”必须同时满足：

```text
1. 表达力不打折；
2. forward / backward / step / memory 与 MLP 可比；
3. 收敛轨迹更健康，AUC-step / AUC-time 不差；
4. LineC / signal-reservoir-noise / tail / calibration 不坏；
5. functional update 的收益不能被 NoOp / Random / AdamWParallel / matched controls 解释；
6. 不能靠 label-informed init、dataset branch、validation/test/future、CE-tail target 取得收益。
```

最终要证明的是：

$$
\boxed{
\text{PureKAN substrate + functional update}
>
\text{same PureKAN substrate + ordinary AdamW / controls}
}
$$

并且进一步区分：

```text
generic functional optimizer value:
  MLP-FMS 也能得到的收益。

KAN-specific functional leverage:
  Rational / Wavelet / other basis substrate 在同等条件下额外获得的收益。
```

---

## 0.2 v14.4 当前真实状态

v14.4 已经不是早期的 “functional update 完全没有信号”。它已经达到：

```text
S4b-RealTransferExplorationPositive
best real_dataset_seed_pass_count = 6 / 9
best real_short_run_pass_rows = 11
official_s5_reached = 0
promotion_allowed = 0
```

这意味着：

```text
1. synthetic S3 已经不是主 blocker；
2. real short-run 已经合法打开；
3. functional update 在真实 3x3 上有 6/9 的 exploration positive；
4. 但 S5 要求 9/9，因此不能 promotion。
```

v14.4 已经尝试过：

```text
K-RT1 split agreement；
K-RT2 train-stream tail trust；
K-RT3 projection value retention；
K-RT4 source-tail co-state；
K-RT5 delayed basis constraint；
K-RT6 projection + tail trust；
K-RT7 late projection + tail trust；
K8 baseline under lowplasticity；
low plasticity / lower lambda；
stronger train-stream tail trust；
moderate split / delayed gate；
slow FMS refresh；
train-stream entropy output geometry；
ultralow FMS plasticity；
low-lr repair；
single-refresh repair。
```

这些都没有把 6/9 推到 9/9。下一步不能继续做 K-RT8/K-RT9 小变体。

---

## 0.3 当前核心判断

v14.4 后，真正 blocker 已经变成：

$$
\boxed{
\text{real-transfer 中 source、AUCtime、tail 三者不能稳定同位。}
}
$$

当前最重要的观测是：LineC 在后续最佳 repair 中基本不再是主要 blocker；剩余失败集中在：

```text
AUCtime；
CEp99 tail；
source instability；
NLL tail；
ECE tail。
```

这说明当前 FMS 已经能比较稳定地不撕裂 LineC，但它还不能保证真实训练过程中的 source gain、AUC trajectory 和 tail/calibration 同时健康。

因此 v14.5 的核心变化是：

$$
\boxed{
\text{从固定 K-RT 方法，升级为 train-stream counterfactual controller。}
}
$$

也就是不再问：

```text
K-RT1 / K-RT2 / K-RT3 / K-RT6 哪个固定规则最好？
```

而是问：

```text
在每个 train state 上，能否用合法 train-stream counterfactual 选择 NoOp / GenericFMS / TailTrust / Projection / DelayedBasis / Hybrid 中最安全有效的 action？
```

---

# 1. 当前各线完成度

这些百分比不是官方 artifact 字段，而是根据 gate、机制清晰度、artifact 完整性和距离最终目标的距离估计。

| 线 | 完成度 | 当前判断 |
|---|---:|---|
| Line R：代码 / provenance / finalizer 审计 | 99% | 工程闭包很强，不是当前 blocker。 |
| Historical FHQ / B320-current | 85% frozen | 历史强，但 label-informed init 已禁用，不能 official。 |
| Line C：Manifold-channel 几何审计 | 88% | 审计稳定，能解释 failure，但不能作为 direction source。 |
| PopRisk / FMS 实现面 | 92% | persistent state、per-example gradients、amortized update、projection trace、K-RT surface 已成熟。 |
| Generic MLP-FMS | 45% | 有 generic signal，但不能作为 KAN promotion。 |
| Rational substrate | 85% | 当前最稳定 strict basis substrate。 |
| Rational FMS synthetic | 70% | 已达 S3，说明 KAN-specific synthetic signal 存在。 |
| Rational FMS real-transfer | 60% | 已达 S4b 6/9，但未达 S5。 |
| Official S5 functional success | 0% | promotion 仍关闭。 |
| Train-stream counterfactual controller | 0% | v14.5 新主线，尚未执行。 |
| Wavelet substrate | 50% | D-WAV17/18/19 和 train-entropy output geometry 给出真实 substrate 信号，但仍需 hardening。 |
| RBF / FastKAN substrate | 20% | 仍 task collapse，不能进入 official FMS proof。 |
| Chebyshev substrate | 15% | exact/manual rows 有信号，但 full-step incremental memory 未闭合。 |
| Fourier substrate | 20% | memory/step 局部好，但 strict substrate 不稳。 |
| Non-RAT official FMS | 10% | 有 3 个 strict substrate pass 历史线索，但尚未进入 official FMS proof。 |
| 整体 next-gen MLP claim | 45%-52% | S3/S4b 是真实进展，但 S5 与 all-basis substrate 仍未闭合。 |

---

# 2. v14.4 独立分析

## 2.1 有进展，但不能夸大

v14.4 的最大进展是：

```text
1. real short-run 不再只是 4/9，而是 best 6/9；
2. K-RT6/K-RT7 等组合修复后，LineC failure 在部分 best runs 中降到 0；
3. remaining failures 已经清楚集中在 source / AUCtime / CEp99 / NLL / ECE；
4. train-stream-only constraints 被系统尝试，且没有使用 forbidden audit target 做 direction。
```

这说明 functional update 不是虚假信号。它确实已经能在真实训练中产生部分可转移收益。

但它没有成功：

```text
1. S5 需要 9/9，当前 best 只有 6/9；
2. K-RT6/K-RT7 没补上 Fashion-MNIST seed0/2 或 KMNIST seed2；
3. lower lr、ultralow plasticity、single-refresh 都没有突破 6/9；
4. output geometry 甚至退化到 0/9；
5. promotion_allowed 始终为 0。
```

所以正确表述是：

$$
\boxed{
\text{v14.4 达到 S4b exploration positive，但未达到 S5 official success。}
}
$$

---

## 2.2 为什么固定 K-RT 方法到边界了

v14.4 的 K-RT 方法本质上是固定策略：

```text
K-RT1:
  split agreement。

K-RT2:
  train-stream tail trust。

K-RT3:
  projection value retention。

K-RT4:
  source-tail co-state。

K-RT5:
  delayed basis constraint。

K-RT6:
  projection retention + tail trust。

K-RT7:
  late projection + tail trust。
```

它们都是合理的合法 train-stream policies，但都属于 **单一固定规则**。真实 3x3 中的失败不是单一模式：

```text
Fashion-MNIST seed0:
  source / AUC 可以好，但 CEp99 tail fail。

Fashion-MNIST seed2:
  source / tail 可以好，但 AUCtime fail。

KMNIST seed2:
  source / AUCtime / CEp99 往往交错失败。
```

一个固定 K-RT 规则很难同时覆盖这些状态，因为它没有在训练过程中根据当前 train-stream state 选择 action 或 NoOp。

因此，下一步不是更复杂的固定 K-RT8，而是 **action selection / commit controller**。

---

## 2.3 当前真正问题：action bank 是否足够？policy 是否缺失？

v14.4 的结果有两种可能解释：

### 解释 A：action bank 足够，但缺少合法选择策略

如果 K-RT1..K-RT7 的候选 bank 中，在每个 dataset-seed 上都至少有一个合法 action 可以过，但固定方法选错了，那么问题是：

$$
\boxed{
\text{legal train-stream action selection problem。}
}
$$

这时应该做 counterfactual controller，而不是发明新 action。

### 解释 B：action bank 不足，即使 oracle selection 也到不了 9/9

如果把 K-RT1..K-RT7 / NoOp / K8 / lowplasticity 等合法 action 作为 bank，用 diagnostic oracle 选择每个 event / run 的 best action 后仍不能达到 9/9，那么问题是：

$$
\boxed{
\text{缺少新的 functional action family。}
}
$$

这时应转向新 action，例如 low-cost curvature-safe FMS、two-phase source-tail separation、basis-free FMS + delayed projection 等。

v14.5 必须先区分 A/B，不能盲目继续试 K-RT8。

---

# 3. v14.5 总假设

## H1：固定 K-RT 失败，不等于 action bank 失败

当前 6/9 可能来自 fixed policy 的局限。train-stream counterfactual controller 可能在同一 action bank 中选出更稳定的组合。

## H2：AUCtime blocker 是 event timing / event acceptance 问题

很多失败不是 final source 完全不行，而是 AUCtime > 1。这说明 FMS event 可能在训练还应快速下降时干扰了 trajectory。需要 train-stream slope / micro-AUC proxy，而不是只看 immediate source proxy。

## H3：tail blocker 是 train-stream risk proxy 不够因果

CEp99 / NLL / ECE 不能作为 direction，但可以用 train-stream loss q95、margin p10、logit RMS、entropy、wrong-confidence proxy 近似风险。固定 K-RT2 不够，可能需要 counterfactual tail proxy 与 source proxy 同时约束。

## H4：LineC 已不再是主 blocker

v14.4 后续最佳 runs 中 LineC failure 常为 0。下一轮仍保留 LineC audit，但不能把主要预算继续用在 LineC repair。

## H5：Wavelet 是非 Rational 中最值得继续的 substrate

Wavelet 已出现 strict substrate / task-health / train-entropy output geometry 信号。下一轮应继续作为 bounded parallel line，但不能阻塞 Rational real-transfer S5。

---

# 4. v14.5 实验总结构

v14.5 分为六条线：

```text
Line R:
  Code / provenance / implementation readback。

Line O:
  Offline action-bank oracle decomposition。
  回答：现有合法 action bank 有没有 9/9 upper bound？

Line K:
  Train-stream counterfactual FMS controller。
  用 train-stream-only micro-counterfactual 选择 action / NoOp。

Line W:
  Wavelet substrate hardening。
  继续 Non-RAT 最有希望支线。

Line D:
  All-basis bounded substrate repair。
  RBF / Chebyshev / Fourier / Wavelet / Rational status 更新。

Line C:
  Manifold-channel / tail / AUC audit。
  只做审计，不生成方向。

Line Z:
  Finalizer / route / no-go / next-hypothesis queue。
```

---

# 5. Line O：Action-bank oracle decomposition

## 5.1 目标

先回答：

$$
\boxed{
\text{当前 legal action bank 是否存在 9/9 upper bound？}
}
$$

注意：Line O 可以用 audit metric 做 **diagnostic oracle**，但 oracle 不能 promotion。它只用于决定下一步是 policy-learning 还是 action-family redesign。

## 5.2 Action bank

Action bank 包括：

```text
A0: NoOp / AdamW baseline
A1: K8 baseline lowplasticity
A2: K-RT1 TrainSplitAgreement
A3: K-RT2 TrainStreamTailTrust
A4: K-RT3 ProjectionValueRetention
A5: K-RT4 SourceTailCoState
A6: K-RT5 DelayedBasisConstraint
A7: K-RT6 ProjectionTailTrust
A8: K-RT7 LateProjectionTailTrust
A9: single-refresh variant
A10: slowrefresh160 variant
```

这些都必须来自已有合法 method surface；如果缺 artifact，Codex 需要复跑对应 action，不允许用 imagined rows。

## 5.3 Oracle levels

### O1：run-level oracle

每个 dataset-seed 选择一个 best method。这个只回答 action bank 是否覆盖 9/9。

### O2：event-level oracle

如果 artifact 有 event/state rows，则每个 event 选择 best action，再跑 replay。这个更强，但只能 diagnostic。

### O3：leave-one-dataset-out oracle

用两个 dataset 的 diagnostic best policy 规则，在第三个 dataset 上评估。防止 oracle 只是 dataset-specific。

## 5.4 必须记录字段

`v145_action_bank_oracle.csv`：

```text
dataset
seed
action_id
source
AUCtime_ratio
CEp99_delta
NLL_delta
ECE_delta
LineC_pass
pass_s5_row
oracle_level
oracle_uses_audit_metric
promotion_allowed
```

`v145_oracle_summary.csv`：

```text
oracle_level
dataset_seed_pass_count
pass_rows
failure_remaining_source
failure_remaining_auc
failure_remaining_cep99
failure_remaining_nll
failure_remaining_ece
failure_remaining_linec
action_bank_upper_bound_pass
```

## 5.5 Gate

If:

```text
O1 or O2 oracle dataset_seed_pass_count < 9
```

then:

```text
route = R2-ActionBankUpperBoundInsufficient
```

and Line K policy learning cannot be expected to reach S5 without new actions.

If:

```text
O1/O2 oracle = 9/9
```

then:

```text
Line K controller is justified.
```

---

# 6. Line K：Train-stream counterfactual FMS controller

## 6.1 目标

用 **train-stream-only counterfactual** 在每个 FMS event 上选择 action 或 NoOp。

不能使用：

```text
validation/test
future
LineC hard target
CEp99 / NLL / ECE audit
AUCtime audit
dataset-name branch
seed-specific scaling
```

允许使用：

```text
current train mini-batch
split train mini-batch
train loss
train loss q95
train margin p10
train logit RMS
train entropy
train split agreement
gradient SNR
FMS state
basis projection retention
current optimizer state
```

## 6.2 Controller principle

在 event time $t$，对每个 action $a$ 做 train-stream micro-counterfactual：

```text
1. clone current model state;
2. apply action a;
3. run 1-2 micro train-stream steps on split B1;
4. evaluate train-stream proxy on heldout train split B2;
5. estimate source proxy, trajectory proxy, tail proxy, stability proxy;
6. choose action with best conservative Pareto score;
7. if no action beats NoOp on all safety gates, commit NoOp.
```

## 6.3 Proxy definitions

### Source proxy

$$
S_a
=
L_{B2}(\theta_t)
-
L_{B2}(\theta_t + a)
$$

or after micro-step:

$$
S_a^{micro}
=
L_{B2}(\theta_t^{micro,NoOp})
-
L_{B2}(\theta_t^{micro,a})
$$

### AUC / trajectory proxy

Use train-stream local micro-AUC:

$$
A_a
=
\sum_{j=1}^{h}
L_{B_j}(\theta_{t+j}^{a})
-
\sum_{j=1}^{h}
L_{B_j}(\theta_{t+j}^{NoOp})
$$

Require:

$$
A_a \le \epsilon_A.
$$

This is not validation AUC. It is a train-stream local trajectory proxy.

### Tail proxy

Use train loss quantile / margin / logit RMS:

$$
T_a
=
q95(L_{B2}(\theta_t+a))
+
\lambda_m \max(0, m_{target}-margin\_p10)
+
\lambda_r \max(0, logitRMS-r_{max})
$$

Require non-harm relative to NoOp.

### Split agreement

$$
Agree_a
=
\cos(
\Delta\theta_a^{B1},
\Delta\theta_a^{B2}
).
$$

or sign agreement on FMS value vector.

### Conservative action score

$$
Score(a)
=
S_a
-\lambda_A \max(0,A_a)
-\lambda_T \max(0,T_a-T_{NoOp})
-\lambda_R RuntimePenalty(a)
-\lambda_D DriftPenalty(a).
$$

Commit rule:

$$
a^*
=
\arg\max_{a\in\mathcal A \cup \{NoOp\}} Score(a).
$$

Accept only if:

$$
Score(a^*) > Score(NoOp)+\epsilon,
$$

$$
S_{a^*} > 0,
$$

$$
A_{a^*} \le \epsilon_A,
$$

$$
T_{a^*} \le T_{NoOp}+\epsilon_T.
$$

## 6.4 Candidate controllers

```text
K-CF0:
  No controller; best v14.4 fixed method reference.

K-CF1:
  TrainSplitCounterfactualSourceOnly.
  Select by source proxy; no tail/AUC guard except hard rejection.

K-CF2:
  SourceTailParetoController.
  Select by source + train-tail proxy.

K-CF3:
  SourceAUCTailParetoController.
  Select by source + train micro-AUC + train-tail proxy.

K-CF4:
  ConservativeNoOpDominantController.
  NoOp unless action beats NoOp on source/tail/AUC proxies.

K-CF5:
  TwoStageController.
  Stage 1 choose source-safe candidates.
  Stage 2 choose tail/AUC-safe among them.

K-CF6:
  BankOracleDistilledController.
  Train a tiny fixed-threshold policy from train-stream proxy rows only;
  no dataset name, no validation/test/future/audit metrics.
```

## 6.5 Must-record artifacts

`v145_counterfactual_event_log.csv`：

```text
dataset
seed
step
candidate_action
selected_action
source_proxy
micro_auc_proxy
tail_proxy
split_agreement
projection_retention
fms_active_fraction
basis_rejection_fraction
score
accepted
rejection_reason
counterfactual_overhead_ms
```

`v145_controller_real_results.csv`：

```text
dataset
seed
controller_id
real_pass
source_vs_best_control
AUCtime_ratio
CEp99_delta
NLL_delta
ECE_delta
LineC_pass
step_time_ratio
memory_ratio
selected_action_counts
noop_rate
event_count
overhead_ratio
```

`v145_controller_failure_table.csv`：

```text
controller_id
dataset
seed
failure_source
failure_auc
failure_cep99
failure_nll
failure_ece
failure_linec
failure_overhead
dominant_failure
```

## 6.6 Gate

Exploration S4c:

```text
real_dataset_seed_pass_count >= 7/9
source_vs_best_control_mean > 0
LineC pass in all 9 dataset-seeds
overhead_ratio <= 1.25
```

Official S5:

```text
real_dataset_seed_pass_count = 9/9
source_vs_best_control >= 0.005 per dataset-seed
AUCtime_ratio <= 1.0
CEp99_delta <= 0.05
NLL_delta <= 0.02
ECE_delta <= 0.02
LineC_pass = 1
promotion_allowed = 1 only after provenance / manifest / code review pass
```

If controller reaches 7/9 but not 9/9:

```text
route = S4c-ControllerExplorationPositive
```

not S5.

---

# 7. Line W：Wavelet substrate hardening

## 7.1 目标

Wavelet 是当前 Non-RAT 中最有希望的 substrate，但仍不 robust。Line W 不阻塞 Line K，但必须继续推进 all-basis portfolio。

## 7.2 Candidates

```text
W1: D-WAV17 train-entropy substrate hardening
W2: D-WAV18 train-entropy substrate hardening
W3: D-WAV19 train-entropy substrate hardening
W4: Wavelet support + train entropy + delayed readout mixing
W5: Wavelet scale occupancy balanced update
W6: Wavelet local-tail coverage guard
```

## 7.3 Must-record

```text
candidate_id
dataset
seed
workspace_raw_ratio
workspace_incremental_ratio
step_ratio
mean_delta_vs_mlp
worst_delta_vs_mlp
AUCtime_ratio
CEp99_delta
NLL_delta
ECE_delta
LineC_pass_rate
RealSignalReservoirRatio
NoiseSignalLeak
scale_occupancy_entropy
support_overlap
local_tail_coverage
```

## 7.4 Gate

Wavelet substrate gate:

```text
workspace_incremental_ratio <= 1.75
step_ratio <= 1.75
mean_delta_vs_mlp >= -0.05
worst_delta_vs_mlp >= -0.10
AUCtime_ratio <= 2.0
LineC_pass_rate >= 0.30
```

If Wavelet reaches full 3x3 substrate pass, then and only then:

```text
open Wavelet-FMS synthetic proof
```

Otherwise:

```text
keep Wavelet as substrate line, not functional line.
```

---

# 8. Line D：All-basis substrate parallel repair

## 8.1 Rational

Status:

```text
main FMS carrier
```

Tasks:

```text
1. keep Rational substrate no-regression monitor;
2. expose failure states for K-CF controller;
3. avoid new Rational token search unless Line O proves action bank insufficient.
```

## 8.2 RBF / FastKAN

Current blocker:

```text
task collapse, despite some LineC rows.
```

Required repair direction:

```text
1. compact center occupancy repair;
2. identity residual stronger base;
3. width condition guard;
4. low-k active-center local support;
5. no dense RBF materialization.
```

## 8.3 Chebyshev

Current blocker:

```text
incremental memory / degree-energy / task-health.
```

Required repair:

```text
1. recurrence no-materialize lifetime repair;
2. degree energy damping;
3. low-degree identity residual;
4. high-degree late enable only after substrate gate.
```

## 8.4 Fourier

Current blocker:

```text
low-frequency efficiency okay locally, but expression/task and incremental lifetime not robust.
```

Required repair:

```text
1. band-limited identity residual;
2. phase-stable low-frequency anchor;
3. high-frequency quarantine;
4. no high-frequency noise-heavy path.
```

## 8.5 Wavelet

Handled in Line W.

## 8.6 B-spline

```text
Frozen.
No active budget.
Can reuse MatrixKAN / sparse local support / no-dense-basis implementation ideas only.
```

---

# 9. Line M：MLP generic FMS control

## 9.1 目标

Keep MLP as control:

```text
If MLP-FMS improves while Rational-FMS does not:
  FMS is generic but KAN-specific leverage is not established.

If Rational controller improves beyond MLP-FMS:
  possible KAN-specific functional advantage.
```

## 9.2 Required comparison

```text
MLP-AdamW
MLP-FMS-Amortized
MLP-FMS-SplitAgreement
MLP-FMS-SourceTailCoState
MLP-FMS-CounterfactualController, optional
```

## 9.3 Gate

Do not write MLP result as KAN success.

Record:

```text
MLP_generic_pass
MLP_source_vs_adamw
MLP_AUCtime_ratio
MLP_CEp99_delta
MLP_NLL_delta
MLP_ECE_delta
MLP_overhead_ratio
```

---

# 10. Line C：Audit only

LineC and tail metrics are gates / audit, not direction sources.

Must record:

```text
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
KernelDrift
CEp99
NLL
ECE
Brier
margin_p10
AUCtime
source_vs_best_control
LineC_pass
```

Also produce failure decomposition:

```text
source-only fail
AUC-only fail
tail-only fail
source + AUC fail
source + tail fail
AUC + tail fail
LineC fail
overhead fail
```

---

# 11. Required visualizations

Must generate:

```text
fig_v145_real_3x3_pass_matrix.svg
fig_v145_action_oracle_upper_bound.svg
fig_v145_counterfactual_selected_action_heatmap.svg
fig_v145_source_auc_tail_tradeoff.svg
fig_v145_failure_mode_by_dataset_seed.svg
fig_v145_proxy_vs_audit_scatter.svg
fig_v145_noop_rate_vs_pass.svg
fig_v145_overhead_breakdown.svg
fig_v145_wavelet_substrate_matrix.svg
fig_v145_all_basis_status_dashboard.svg
fig_v145_mlp_vs_rational_controller.svg
```

Important: these figures must not be used as direction sources.

---

# 12. Stop / go route

```text
S4c-ControllerExplorationPositive:
  controller real pass >= 7/9 but <9/9.

S5-OfficialFunctionalSuccess:
  real pass = 9/9 and all gates pass.

R2-ActionBankUpperBoundInsufficient:
  oracle action bank cannot reach 9/9.

R3-CounterfactualPolicyFail:
  oracle action bank can reach 9/9 but train-stream policy cannot.

R4-ActionFamilyMissing:
  both oracle and legal policy fail because action bank cannot cover missing states.

R5-AllBasisSubstrateStillRationalOnly:
  Rational remains sole functional carrier; Non-RAT substrate not ready.

R6-GenericOnlyNoKANSpecific:
  MLP controller passes but Rational/KAN controller does not.

R0-ForbiddenDirectionViolation:
  direction uses validation/test/future/query/LineC/CEp99/NLL/ECE/dataset-name branch.
```

---

# 13. Codex failure-response rules

## 13.1 If oracle bank < 9/9

Codex must not tune policy. It must:

```text
1. identify missing state classes;
2. list which dataset-seeds cannot be covered by any current action;
3. propose new action family, not new selection threshold;
4. keep promotion closed.
```

## 13.2 If oracle bank = 9/9 but controller < 7/9

Codex must:

```text
1. compare train proxy ranking vs oracle ranking;
2. identify proxy mismatch:
   source proxy fail / auc proxy fail / tail proxy fail / split agreement fail;
3. improve proxy, not action;
4. rerun controller.
```

## 13.3 If controller reaches 7/9 but not 9/9

Codex must:

```text
1. report remaining failed dataset-seeds;
2. group failures by source/AUC/tail/LineC/overhead;
3. try one next-level controller only if it uses train-stream features;
4. never use dataset-name branch or seed-specific scaling.
```

## 13.4 If overhead > gate

Codex must:

```text
1. reduce candidate bank size using oracle/action importance;
2. amortize counterfactual evaluation;
3. cache train-stream proxy computations;
4. report overhead vs S5 tradeoff.
```

## 13.5 If Wavelet substrate fails

Codex must:

```text
1. classify failure as workspace / task / AUC / LineC / tail;
2. if LineC reservoir fail, try train-stream reservoir proxy only;
3. if task collapses, repair substrate, not FMS;
4. do not enter Wavelet-FMS official proof.
```

---

# 14. Why this is not small repair

v14.4 exhausted fixed K-RT methods. v14.5 changes the problem from:

```text
Which fixed K-RT method is best?
```

to:

```text
Does the current legal action bank contain enough actions, and can a train-stream counterfactual controller select them without forbidden information?
```

This is a different causal question. It separates:

```text
action insufficiency
policy insufficiency
proxy insufficiency
overhead insufficiency
substrate insufficiency
```

Without this separation, continuing K-RT token search would be low-value.

---

# 15. Minimum expected outputs

At minimum v14.5 must output:

```text
v145_route_decision.json
v145_required_manifest.csv
v145_forbidden_information_audit.csv
v145_code_review_manifest.csv
v145_action_bank_oracle.csv
v145_oracle_summary.csv
v145_counterfactual_event_log.csv
v145_controller_real_results.csv
v145_controller_summary.csv
v145_controller_failure_table.csv
v145_wavelet_substrate_hardening.csv
v145_all_basis_status.csv
v145_mlp_generic_control.csv
v145_linec_tail_auc_audit.csv
v145_no_go_boundary.md
v145_next_hypothesis_queue.md
figures/*.svg
v145_code_review_packet.zip
```

缺少 oracle/action-bank 或 counterfactual-event artifact 时，不能声称完成 v14.5。

---

# 16. Final statement

v14.5 的目标不是降低 S5 gate，也不是把 6/9 包装成成功。它的目标是回答一个更深的问题：

$$
\boxed{
\text{v14.4 的 6/9 是因为 action bank 不够，还是因为缺少合法 train-stream action selection policy？}
}
$$

只有这个问题回答清楚，才能知道下一步应该：

```text
继续 functional controller；
还是设计新 functional action；
还是回到 substrate architecture；
还是承认当前 FMS 只能作为 partial/generic optimizer。
```
