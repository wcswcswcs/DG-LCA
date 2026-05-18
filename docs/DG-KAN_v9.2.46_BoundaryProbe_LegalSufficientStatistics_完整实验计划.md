# DG-KAN v9.2.46 Boundary-Probe Legal Sufficient Statistics 与 Support-Matched Paired Replay Closure 完整实验计划

> 本计划基于 v9.2.45 `Legal Feature Representation 与 Support-Matched Controller Closure` 的真实复盘制定。  
> v9.2.45 的 terminal route 是：
>
> ```text
> route = R2-LegalFeatureGapUnattributed
> base_candidate = LQ-t2-h256
> success_v9245_strict_purekan_functional = False
> success_v9245_full_functional = False
> success_v9245_external_ready = False
> ```
>
> v9.2.45 的关键事实是：
>
> ```text
> P0:
>   v9.2.44 boundary reproduced
>   source route = R8-OracleHighLegalFeatureGap
>   F1 support oracle pass retained
>
> P1:
>   F1/C1 gap attribution pass = 0
>   F1_C1_jaccard = 0.294574
>   C1 false-negative count = 85
>   C1 false-negative primary mode = G2-threshold_miscalibration
>   false-negative attribution fraction = 1.0
>   C1 false-positive count = 6
>   C1 false-positive primary mode = G5-control_gap_mismatch
>   false-positive attribution fraction = 0.666667 < 0.90
>
> P2:
>   F1 source support precision = 0.967480
>   F1 source support coverage = 0.040039
>   F1 source support bad-event = 0.032520
>   heldout F1 bad-event = 0.065574
>
> P3:
>   legal feature factory has predictivity survivor
>   best = LF8-HybridLegalMonotone
>   but P1 gate failed, so no official controller success
>
> P4:
>   no tri-stage controller survivor
>   best = C0-C1RiskFirstS7Reference
>   AUC = 0.655201
>   corr = 0.391692
>   precision = 0.789474
>   coverage = 0.012370
>   bad-event = 0.210526
>
> current blocker:
>   legal_feature_gap_unattributed
> ```
>
> 因此 v9.2.46 的核心任务不是继续修 base、attach、carrier，也不是继续在 C1/S7 上小调 threshold。当前本质问题是：
>
> $$
> \boxed{
> \text{F1 oracle support 能恢复 safe-good support，但现有 legal features 不能稳定解释并复现 F1/C1 分歧边界。}
> }
> $$
>
> 更深一层说，v9.2.45 暴露的是 **legal sufficient statistics 缺口**，不是单纯 controller threshold 缺口。C1 false-negative 主要是 threshold miscalibration，但 false-positive attribution 没闭合；同时 F1 heldout bad-event 也超过 $0.05$，说明 F1 也不能被当成最终目标，只能作为 oracle-support reference。v9.2.46 必须重建目标：
>
> $$
> \boxed{
> \text{直接学习/构造 commit-time legal statistics 去预测 } Y_{\text{safe-good}},
> \text{ 而不是只模仿 F1 或 C1。}
> }
> $$
>
> 本轮继续加快实验，采用并行 runner。所有 diagnostic rows 可以提前测，但 official success 必须严格受 gate 控制。

---

## 0. 硬约束

本计划继续遵守：

```text
no teacher
no self-teacher
no distillation
no loss modification
no label smoothing
no focal / margin / calibration loss
no sampler / class weight
no CPU offload
no fake / proxy rows
KAN path 不使用 PyTorch loss.backward graph
official controller 不使用 dataset_name 分支
PureKANConv / PureKANFormer 继续 deferred
```

Functional update 仍然是 update rule，不是 loss：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{\text{AdamW-equivalent}}
+
\Delta\theta_{\text{functional}}.
$$

任务目标保持标准 CE：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

数据集只能作为诊断切片，不能作为 official controller 条件：

```text
allowed:
  report MNIST / Fashion-MNIST / KMNIST slice metrics
  report false-positive / false-negative distribution by dataset
  report leave-dataset-out generalization
  report signal-stratum / family composition by dataset
  report per-dataset diagnostic plots

forbidden:
  if dataset == Fashion: use controller A
  if dataset == KMNIST: use primitive B
  if dataset == MNIST: abstain
  tune threshold separately per dataset
  promote a controller because it rescues only one dataset
  use validation/test metric at commit time
  use posthoc replay outcome at commit time for official controller
```

---

# Part I. 对 v9.2.45 的独立判断

## 1. v9.2.45 没有达到目标

v9.2.45 没有达到 strict PureKAN functional success。原因不是 downstream paired replay 输了，而是更前面的 P1/P4 没过：

```text
P1 F1/C1 gap autopsy failed
P4 tri-stage controller failed
LDO / LSO / paired replay / short-run / full-run / robustness all not_run
```

因此不能声明：

```text
strict PureKAN local causal evidence
full functional success
external-ready success
Beyond-MLP success
```

## 2. v9.2.45 的真实推进

v9.2.44 已经证明 F1 risk-first support 可以恢复 oracle safe-good support。v9.2.45 进一步说明，这个 support 并不是空信号：

```text
F1 source precision = 0.967480
F1 source coverage = 0.040039
F1 source bad-event = 0.032520
```

同时 P3 legal feature factory 已经出现 predictivity survivor，尤其：

```text
LF8-HybridLegalMonotone:
  legal = 1
  predictivity survivor = 1
  AUC_to_F1_support = 0.823534
  AUC_to_safe_good = 0.401034
```

这说明 legal feature 并非完全没有信息。当前不是 “所有 legal features 都没信号”。

但 v9.2.45 也证明，这些信号还没有转化成可以部署的 controller：

```text
C0-C1RiskFirstS7Reference:
  precision = 0.789474
  coverage = 0.012370
  bad-event = 0.210526

C1-BranchRiskTriStage:
  precision = 0.206349
  coverage = 0.041016
  bad-event = 0.000000

C5-SNR-GatedControlGap:
  precision = 0.128000
  coverage = 0.081380
  bad-event = 0.032000

C7-HybridLegalMonotone:
  precision = 0.141593
  coverage = 0.073568
  bad-event = 0.026549
```

这些结果揭示了一个结构性分裂：

```text
C0:
  precision 接近可用，但 coverage 太低且 bad-event 太高。

C1/C5/C7:
  bad-event 能控制，但 precision 太低。

P3 legal features:
  能预测 F1 support，但不能直接变成 safe-good controller。
```

## 3. 当前真正 blocker

当前 blocker 是：

$$
\boxed{
\text{legal feature representation 没有分离 safety、value、control-resistance 三个因子。}
}
$$

v9.2.45 的失败不是一个单一 threshold 问题，而是目标分解错误。我们同时在做三件事：

```text
1. 找 task-safe event；
2. 找 value-positive event；
3. 找 control-resistant event。
```

但是 C1 / S7 / hybrid controller 仍把它们混成单一 score。结果就是：

```text
score 相关性存在，但 accept gate 失败；
risk-safe 规则存在，但 precision 失败；
F1 oracle support 存在，但 heldout bad-event 仍超 gate；
false-positive attribution 不足，说明 risk/control/value 的边界还没解释清楚。
```

因此 v9.2.46 不应再问 “哪个单一 score 能过 gate”，而应问：

$$
\boxed{
\text{哪些 commit-time sufficient statistics 分别预测 } RiskSafe,\ ValuePositive,\ ControlResistant？
}
$$

## 4. 为什么仍然感觉进展慢

进展慢是真的。但慢的性质变了：

```text
早期慢：
  base / attach / carrier 基础设施没闭合。

中期慢：
  score 在 source rows 上过，fresh/leaveout 不过。

现在慢：
  oracle support 有，legal features 有信号，但 controller 无法同时满足 precision / coverage / bad-event。
```

这说明我们已经进入 functional update 的最核心问题：**如何用 commit-time train-stream signals 识别 safe-good functional events**。  
v9.2.46 必须加快，不能继续一轮只做一个 feature 或一个 controller。必须并行：

```text
boundary-targeted event expansion
legal feature factory v2
false-positive/false-negative attribution
tri-factor controller
train-stream micro-probe
leave-out
paired replay scout
```

## 5. 是否在正确道路上

是，但路线需要收紧。正确路线是：

```text
F1 support exists
→ explain C1/F1 gap
→ extract legal sufficient statistics
→ tri-factor controller
→ LDO/LSO
→ paired replay
```

错误路线是：

```text
继续手调 C1 threshold；
继续只拟合 F1 support；
继续按 dataset 分支；
看到 P4 没过就盲目 reset carrier；
把 P3 legal-feature predictivity 写成 controller success。
```

v9.2.45 的结论其实比 v9.2.44 更接近核心：我们已经知道 support 不是空的，也知道现有 legal feature 有一定预测力；现在要找的是 **可部署的 sufficient statistics**。

---

# Part II. v9.2.46 总体目标

v9.2.46 的总体目标是：

$$
\boxed{
\text{将 F1 oracle support 与 legal feature predictivity 转化为 heldout-valid tri-factor legal controller，并打开 LDO/LSO 与 paired replay。}
}
$$

目标分为七层。

## 1. Boundary-targeted attribution success

必须补足 v9.2.45 的 P1 failure。尤其 false-positive count 只有 6，且 attribution fraction 只有 $0.666667$，这不足以定位机制。v9.2.46 必须围绕 C1/F1 disagreement 生成 targeted rows：

```text
D_FN = F1_accept = 1, C1_accept = 0
D_FP = F1_accept = 0, C1_accept = 1
D_AGREE_GOOD = F1_accept = 1, C1_accept = 1
D_AGREE_REJECT = F1_accept = 0, C1_accept = 0
```

成功标准：

```text
false-positive count >= 50
false-negative count >= 100
false-positive attribution fraction >= 0.90
false-negative attribution fraction >= 0.90
dataset_name_used = 0
```

## 2. Target regrounding success

不能只预测 F1 support。必须同时定义：

$$
Y_{\text{risk-safe}}=\mathbb{1}[BadEvent=0],
$$

$$
Y_{\text{value-positive}}=\mathbb{1}[Gain_{\text{Real}}>0],
$$

$$
Y_{\text{control-resistant}}
=
\mathbb{1}[
Gain_{\text{Real}}>
\max(Gain_{\text{AdamWParallel}},Gain_{\text{bestLR}})
],
$$

$$
Y_{\text{safe-good}}
=
Y_{\text{risk-safe}}
\cdot
Y_{\text{value-positive}}
\cdot
Y_{\text{control-resistant}}.
$$

Target regrounding pass：

```text
all four labels measured
label reliability >= 0.50 for safe-good
no posthoc outcome used at commit
```

## 3. Legal sufficient statistics success

至少一个 commit-time legal feature group 必须能预测 safe-good，而不只是 F1 support：

$$
AUC(S_{\text{legal}},Y_{\text{safe-good}})\geq0.70
$$

or:

$$
Corr(S_{\text{legal}},V_{\text{safe-grounded}})\geq0.35.
$$

同时各子因子至少一个 feature 达到：

$$
AUC(S_{\text{risk}},Y_{\text{risk-safe}})\geq0.70,
$$

$$
AUC(S_{\text{value}},Y_{\text{value-positive}})\geq0.65,
$$

$$
AUC(S_{\text{gap}},Y_{\text{control-resistant}})\geq0.65.
$$

## 4. Tri-factor controller success

Official controller 必须采用三因子结构，而不是单一 score：

$$
Accept(e)
=
RiskSafe(e)
\land
ValuePositive(e)
\land
ControlResistant(e).
$$

其中：

$$
RiskSafe(e)=\mathbb{1}[S_{\text{risk}}(e)\leq \rho],
$$

$$
ValuePositive(e)=\mathbb{1}[S_{\text{value}}(e)\geq \tau_v],
$$

$$
ControlResistant(e)=\mathbb{1}[S_{\text{gap}}(e)\geq \tau_g].
$$

Heldout gate：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{heldout}}\leq0.05.
$$

Multi-support gate：

```text
accepted_signal_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

## 5. System gate

Controller feature overhead must not break kernel-native envelope:

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

Train-stream micro-probe is allowed only if:

$$
Overhead_{\text{probe}}\leq0.20.
$$

If overhead is higher, the feature can remain diagnostic but cannot be official.

## 6. Leave-out success

Leave-dataset-out：

$$
Acc_{\text{heldout,Real}}\geq Acc_{\text{heldout,AdamW}}-0.005.
$$

At least two held-out datasets satisfy：

$$
BeatRate_{\text{Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{Real vs bestLR}}\geq0.50.
$$

Leave-stratum-out：

At least $70\%$ held-out strata task-safe and：

$$
CEp99_{\text{heldout,Real}}\leq CEp99_{\text{AdamW}}+\epsilon,
$$

$$
BeatRate_{\text{heldout-stratum,Real vs AdamWParallel}}\geq0.50.
$$

## 7. Official paired replay success

Paired replay pass：

$$
BeatRate_{\text{macro,Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro,Real vs bestLR}}\geq0.60.
$$

Task safety：

$$
Acc_{\text{Real,slice}}\geq Acc_{\text{AdamW,slice}}-0.005.
$$

Shuffle controls must fail：

```text
ShuffledRiskScore = fail
ShuffledBranchRatio = fail
ShuffledControlGap = fail
ShuffledValueScore = fail
FunctionalChannelShuffled = fail
TailMaskShuffled = fail
RoleScoreShuffled = fail
DatasetRouteShuffled = fail
EventRouteShuffled = fail
InvertedRoleMask = fail
```

---

# Part III. 核心假设

## H1：v9.2.45 失败是 false-positive boundary 未解释，而不是所有 legal features 无效

证据：

```text
LF8-HybridLegalMonotone has AUC_to_F1_support = 0.823534
LF6-DriftDiffusionSNR has AUC_to_F1_support = 0.806572
LF1-BranchRatioRisk has AUC_to_F1_support = 0.767685
```

H1 成立标准：

如果 boundary-targeted expansion 后，false positives 被归因到 clear mode，例如：

```text
control_gap_mismatch
risk_boundary_mismatch
family_imbalance
carrier_support_mismatch
```

且 attribution fraction >= 0.90，则 H1 成立。

H1 失败标准：

如果 false positives 仍无法归因，并且 no legal feature can separate FP from safe-good rows，则 legal representation gap 加深。

## H2：F1 support 是参考 support，不是最终 target

F1 source support pass，但 heldout bad-event 已经达到 `0.065574`，超过 $0.05$。因此 F1 不是 official target。  
H2 认为 v9.2.46 应直接预测 $Y_{\text{safe-good}}$，而不是只最大化 F1 overlap。

H2 成立标准：

Tri-factor controller 对 $Y_{\text{safe-good}}$ 的 heldout precision / coverage / bad-event 过 gate，即使 F1/C1 overlap 不高也可以晋级。

## H3：train-stream micro-probe 是必要的 legal sufficient statistic candidate

v9.2.45 的 `LF7-TrainStreamMicroProbe` 未测，不能判断其价值。  
H3 认为仅靠 static branch/risk/value features 不够，需要一个 commit-time train-stream probe 估计 population-safe direction。

Micro-probe 必须只使用 train split：

```text
update batch:
  forms candidate functional update

probe batch:
  estimates train-stream risk/value/gap proxy
```

不允许 validation/test/posthoc replay。

H3 成立标准：

$$
AUC(S_{\text{probe}},Y_{\text{safe-good}})\geq0.70
$$

or:

$$
Precision_{\text{probe-controller}}\geq0.75
\land Coverage\geq0.03
\land BadEvent\leq0.05.
$$

并且：

$$
Overhead_{\text{probe}}\leq0.20.
$$

## H4：controller 必须 factorized；单一 score 不足

C0 precision 接近但 bad-event 高；C1/C5/C7 bad-event 低但 precision 低。  
H4 认为只有显式 factorization 才能闭合：

```text
risk gate:
  controls bad-event

control-gap gate:
  controls AdamWParallel / bestLR equivalence

value gate:
  controls positive mechanism gain
```

H4 成立标准：

factorized controller 比 C0/C1/C5/C7 同时改善：

$$
BadEventRate_{\text{factor}} < BadEventRate_{\text{C0}},
$$

$$
Precision_{\text{factor}} > Precision_{\text{C1/C5/C7}},
$$

$$
Coverage_{\text{factor}}\geq0.03.
$$

## H5：如果 F1 support stable but legal features fail again，下一步不是 carrier reset，而是 legal feature representation reset

如果 oracle support 仍高，carrier/support 不应成为主 blocker。  
下一步应设计 richer legal features，例如 Jacobian-sketch、micro-probe、branch-ratio kernelization、family reliability with cross-fit，而不是回到 arbitrary basis search。

## H6：如果 F1 support collapses on fresh boundary-targeted rows，才回到 carrier/support reset

如果：

```text
F1 support precision < 0.75
or coverage < 0.03
or bad-event > 0.05
```

then v9.2.44/v9.2.45 support recovery was not stable. Route must return to support/carrier design.

---

# Part IV. 并行执行设计

v9.2.46 runner 必须并行执行：

```text
Lane A:
  v9.2.45 boundary reproduction and F1/C1 gap replay

Lane B:
  boundary-targeted disagreement expansion:
    D_FP, D_FN, D_AGREE_GOOD, D_AGREE_REJECT

Lane C:
  target regrounding:
    Y_risk-safe, Y_value-positive, Y_control-resistant, Y_safe-good

Lane D:
  legal feature factory v2:
    branch ratio decomposition
    risk LCB
    control-gap LCB
    value-score monotone
    family reliability cross-fit
    drift-diffusion SNR
    train-stream micro-probe

Lane E:
  factorized tri-controller calibration

Lane F:
  leave-dataset-out / leave-stratum-out

Lane G:
  paired replay scout and official paired replay

Lane H:
  short-run scout if paired replay passes

Lane I:
  fallback:
    support collapse -> carrier/support reset
    oracle high legal low -> feature representation reset
```

Gate discipline：

```text
diagnostic rows may be measured early
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  oracle support pass
  legal controller pass
  LDO/LSO pass
```

---

# Part V. Legal feature factory v2

## 1. Base and carrier

Official base remains:

```text
R2-LQ-fanin-output-scale-confirmed
```

Carrier candidates:

```text
A0-current-v9245
A3-RoleWiseFT7ResetCarrier
A6-HybridRoleControlRiskCarrier
```

A1 remains diagnostic unless it passes carrier-active gate:

$$
r_{z,\text{tail}}\geq0.10,
$$

$$
r_{\perp,\text{tail}}\geq0.10.
$$

## 2. Feature groups

### LF0：Current C1/S7 features

Reference only.

### LF1：Branch-ratio decomposition

Split branch ratio into interpretable components：

```text
branch_ratio_mean
branch_ratio_p10
branch_ratio_p90
branch_ratio_tail
effective_derivative
tail_activation_mass
functional_step_norm
task_step_norm
```

Feature：

$$
S_{\text{branch}}
=
a_1BR_{\text{tail}}
+
a_2EffectiveDerivative
+
a_3TailActivationMass
-
a_4Uncertainty.
$$

### LF2：Risk LCB

$$
S_{\text{risk}}
=
w_1CEp99
+
w_2WrongConfidenceP95
-
w_3MarginP10
+
w_4Uncertainty
+
w_5Curvature
+
w_6BranchRisk.
$$

RiskSafe：

$$
RiskSafe(e)=\mathbb{1}[S_{\text{risk}}\leq\rho].
$$

### LF3：Control-gap LCB

$$
S_{\text{gap}}
=
LCB(Gain_{\text{Real}})
-
UCB(\max(Gain_{\text{AdamWParallel}},Gain_{\text{bestLR}})).
$$

### LF4：Value-positive monotone score

$$
S_{\text{value}}
=
-\Delta \widehat{CEp99}
+
\lambda_m \Delta \widehat{MarginP10}
-
\lambda_c \Delta \widehat{CurvatureRisk}.
$$

### LF5：Role-gap

$$
S_{\text{role-gap}}
=
\alpha_sS_{\text{stack-gap}}
+
\alpha_hS_{\text{head-gap}}
-
R_{\text{risk}}.
$$

### LF6：Family reliability with cross-fit

Family must not include dataset name：

$$
family(e)
=
(stratum,horizon,risk\_bucket,role\_bucket,attach\_type,branch\_bucket).
$$

Reliability：

$$
Rel(f)
=
\mathbb{E}[Y_{\text{safe-good}}\mid f]
-
\kappa\sqrt{\operatorname{Var}(Y_{\text{safe-good}}\mid f)}.
$$

### LF7：Drift-diffusion / SNR feature

$$
S_{\text{snr}}
=
\frac{\|\mu_{\text{train-stream}}\|^2}
{\sigma^2_{\text{train-stream}}+\epsilon}.
$$

This is a commit-time train-stream statistic, not a loss.

### LF8：Train-stream micro-probe

Use only train split. Split training minibatch into update/probe halves:

```text
update batch:
  compute candidate functional update

probe batch:
  estimate legal risk/value/gap proxy
```

Probe score：

$$
S_{\text{probe}}
=
-\Delta CE_{\text{probe}}
+
\lambda_m\Delta Margin_{\text{probe}}
-
\lambda_r Risk_{\text{probe}}
+
\lambda_g GapProxy_{\text{probe}}.
$$

Legality：

```text
uses_dataset_name = 0
uses_validation = 0
uses_test = 0
uses_posthoc_replay = 0
```

System：

$$
Overhead_{\text{probe}}\leq0.20.
$$

---

# Part VI. Controller candidates

## C0：C1-RiskFirstS7 reference

Current failed baseline.

## C1：RiskValueGapTriStage

$$
Accept(e)
=
RiskSafe_{\text{LF2}}(e)
\land
\mathbb{1}[S_{\text{value}}(e)\geq\tau_v]
\land
\mathbb{1}[S_{\text{gap}}(e)\geq\tau_g].
$$

## C2：BranchRiskControlGap

$$
Accept(e)
=
\mathbb{1}[S_{\text{branch-risk}}\leq \rho_b]
\land
\mathbb{1}[S_{\text{gap}}\geq\tau_g].
$$

## C3：ProbeGatedTriStage

$$
Accept(e)
=
RiskSafe(e)
\land
\mathbb{1}[S_{\text{probe}}\geq\tau_p]
\land
ControlResistant(e).
$$

## C4：FamilyBalancedTriStage

$$
Accept(e)
=
RiskSafe(e)
\land
ValuePositive(e)
\land
ControlResistant(e)
\land
Rel(family(e))\geq r_0
\land
FamilyBalance(e).
$$

## C5：RoleGapRiskFirst

$$
Accept(e)
=
RiskSafe(e)
\land
S_{\text{role-gap}}(e)\geq\tau_r.
$$

## C6：HybridLegalMonotoneV2

A monotone controller using only：

```text
branch_ratio_decomposition
risk_lcb
control_gap_lcb
value_positive_score
role_gap
family_reliability
snr
train_stream_probe
```

No dataset name, no hidden black-box with dataset leakage.

## C7：Oracle

Posthoc diagnostic only. Never official.

---

# Part VII. 实验阶段

## P0：v9.2.45 boundary reproduction

### 目标

复现 v9.2.45 boundary，确认本轮不是解析错误。

### 必须记录

```text
route
source_route_v9244
f1_oracle_precision
f1_oracle_coverage
f1_oracle_bad_event
heldout_f1_oracle_precision
heldout_f1_oracle_coverage
heldout_f1_oracle_bad_event
f1_c1_jaccard
p1_gap_attribution_pass
false_positive_count
false_positive_primary_mode
false_positive_attribution_fraction
false_negative_count
false_negative_primary_mode
false_negative_attribution_fraction
best_legal_feature_id
best_controller_id
controller_precision
controller_coverage
controller_bad_event
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R2-LegalFeatureGapUnattributed
F1 support exists
C1/F1 gap attribution not closed
P4 tri-stage controller pass = 0
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_f1_c1_gate_ladder.svg
p0_legal_feature_vs_controller_gap.svg
```

---

## P1：Boundary-targeted disagreement expansion

### 目标

增加 false-positive / false-negative rows，避免 v9.2.45 只有 6 个 FP 导致 attribution 不稳定。

### 设置

```text
base = R2 repaired base checkpoint
carriers = A0,A3,A6
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
horizons = 20,80,240,640
signal_strata = S1..S8
sampling = oversample C1/F1 disagreement boundary
branches = RealFunctional, AdamWParallel, bestLR, NoOp, Random
```

### 必须记录

```text
row_id
disagreement_type
dataset
seed
horizon
signal_stratum
event_family
carrier_id
F1_accept
C1_accept
safe_good
risk_safe
value_positive
control_resistant
bad_event
real_gain
adamwparallel_gain
bestlr_gain
control_gap
branch_ratio_components
effective_derivative
tail_activation_mass
risk_lcb
value_score
control_gap_lcb
role_gap
family_reliability
```

### 判断标准

P1 pass：

```text
false_positive_count >= 50
false_negative_count >= 100
FP_attribution_fraction >= 0.90
FN_attribution_fraction >= 0.90
dataset_name_used = 0
```

### 可视化

```text
p1_disagreement_support_matrix.svg
p1_fp_fn_feature_heatmap.svg
p1_gap_attribution_sankey.svg
p1_f1_c1_overlap_by_family.svg
```

---

## P2：Target regrounding

### 目标

将 safe-good 分解成 risk/value/control 三个标签，并检查标签可靠性。

### 必须记录

```text
row_id
Y_risk_safe
Y_value_positive
Y_control_resistant
Y_safe_good
label_reliability_risk
label_reliability_value
label_reliability_gap
label_reliability_safe_good
grounded_value
grounded_risk
grounded_control_gap
```

### 判断标准

P2 pass：

$$
Reliability(Y_{\text{safe-good}})\geq0.50.
$$

At least two component labels reliability：

$$
Reliability(Y_k)\geq0.60.
$$

### 可视化

```text
p2_label_reliability_panel.svg
p2_safe_good_decomposition.svg
p2_value_vs_risk_vs_gap.svg
```

---

## P3：Legal feature factory v2

### 目标

实现 LF1-LF8，并评估它们对 F1 support、safe-good、risk/value/gap 子标签的预测力。

### 必须记录

```text
feature_id
feature_group
feature_available_pre_commit
uses_dataset_name
uses_validation
uses_test
uses_posthoc
feature_compute_overhead
memory_overhead
AUC_to_F1_support
AUC_to_risk_safe
AUC_to_value_positive
AUC_to_control_resistant
AUC_to_safe_good
corr_to_safe_grounded_value
precision_at_gate
coverage_at_gate
bad_event_at_gate
```

### 判断标准

Legal feature pass：

```text
feature_available_pre_commit = 1
uses_dataset_name = 0
uses_validation = 0
uses_test = 0
uses_posthoc = 0
```

Predictivity pass：

$$
AUC(S,Y_{\text{safe-good}})\geq0.70
$$

or:

$$
Corr(S,V_{\text{safe-grounded}})\geq0.35.
$$

Component pass：

$$
AUC(S_{\text{risk}},Y_{\text{risk-safe}})\geq0.70,
$$

$$
AUC(S_{\text{value}},Y_{\text{value-positive}})\geq0.65,
$$

$$
AUC(S_{\text{gap}},Y_{\text{control-resistant}})\geq0.65.
$$

System pass：

$$
FeatureOverhead\leq0.20.
$$

### 可视化

```text
p3_feature_predictivity_matrix.svg
p3_feature_overhead_pareto.svg
p3_feature_legality_matrix.svg
p3_component_auc_panel.svg
```

---

## P4：Factorized controller calibration

### 目标

并行校准 C1-C7。所有 threshold / coefficient 必须只从 calibration split 选择，在 heldout split 上评价。

### Split design

```text
calibration rows:
  choose thresholds / monotone coefficients

heldout rows:
  official local controller gate

leave-dataset-out:
  P5

leave-stratum-out:
  P5
```

### 必须记录

```text
controller_id
features_used
thresholds
coefficients
calibration_split_id
heldout_split_id
precision_cal
coverage_cal
bad_event_cal
precision_heldout
coverage_heldout
bad_event_heldout
AUC_heldout
corr_heldout
accepted_strata_count
accepted_family_count
max_family_share
dataset_name_used
posthoc_used_at_commit
validation_used
test_used
feature_overhead
step_q90
memory_ratio
official_eligible
```

### 判断标准

Heldout controller pass：

$$
AUC_{\text{heldout}}\geq0.70
$$

or:

$$
Corr_{\text{heldout}}\geq0.35.
$$

and：

$$
Precision_{\text{heldout}}\geq0.75,
$$

$$
Coverage_{\text{heldout}}\in[0.03,0.15],
$$

$$
BadEventRate_{\text{heldout}}\leq0.05.
$$

Multi-family：

```text
accepted_strata_count >= 2
accepted_family_count >= 4
max_family_share <= 0.60
```

System：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p4_controller_precision_coverage_bad.svg
p4_controller_auc_corr.svg
p4_controller_family_coverage.svg
p4_controller_factor_ablation.svg
p4_controller_vs_f1_overlap.svg
```

---

## P5：Leave-dataset-out / leave-stratum-out

### 目标

证明 controller 不是 dataset-specific 或 single-stratum overfit。

### 设置

Leave-dataset-out：

```text
calibrate on MNIST + Fashion, evaluate KMNIST
calibrate on MNIST + KMNIST, evaluate Fashion
calibrate on Fashion + KMNIST, evaluate MNIST
```

Leave-stratum-out：

```text
calibrate on all but one signal stratum
evaluate held-out stratum
```

### 必须记录

```text
split_type
heldout
carrier_id
controller_id
threshold
precision
coverage
bad_event_rate
task_safe
CEp99_delta
margin_delta
ECE_delta
NLL_delta
curvature_delta
beats_adamwparallel
beats_bestlr
dataset_name_used
shuffle_control_pass
```

### 判断标准

LDO pass：

$$
Acc_{\text{heldout,Real}}\geq Acc_{\text{heldout,AdamW}}-0.005.
$$

At least two held-out datasets：

$$
BeatRate_{\text{Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{Real vs bestLR}}\geq0.50.
$$

LSO pass：

At least $70\%$ held-out strata task-safe and：

$$
CEp99_{\text{heldout,Real}}\leq CEp99_{\text{AdamW}}+\epsilon,
$$

$$
BeatRate_{\text{heldout-stratum,Real vs AdamWParallel}}\geq0.50.
$$

### 可视化

```text
p5_leave_dataset_out_matrix.svg
p5_leave_stratum_out_matrix.svg
p5_hidden_dataset_tuning_audit.svg
p5_leaveout_failure_modes.svg
```

---

## P6：Official paired replay

### 目标

验证 RealFunctional 是否在 strong controls 下有局部因果优势。

### 设置

```text
base = R2 repaired base checkpoint
carrier_id = best P5 survivor
controller_id = best P5 survivor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRiskScore, ShuffledBranchRatio, ShuffledControlGap, ShuffledValueScore, FunctionalChannelShuffled, TailMaskShuffled, RoleScoreShuffled, DatasetRouteShuffled, EventRouteShuffled, InvertedRoleMask
```

### 必须记录

```text
carrier_id
controller_id
dataset
seed
horizon
signal_stratum
event_family
branch
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
real_beats_adamwparallel
real_beats_bestlr
real_beats_random
real_beats_noop
task_safe
event_count
coverage
bad_event_rate
step_q90
memory_ratio
base_checkpoint_hash
```

### 判断标准

Paired replay pass：

$$
BeatRate_{\text{macro,Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro,Real vs bestLR}}\geq0.60.
$$

Task safety：

$$
Acc_{\text{each slice,Real}}\geq Acc_{\text{AdamW}}-0.005.
$$

Shuffle controls fail：

```text
ShuffledRiskScore = fail
ShuffledBranchRatio = fail
ShuffledControlGap = fail
ShuffledValueScore = fail
FunctionalChannelShuffled = fail
TailMaskShuffled = fail
RoleScoreShuffled = fail
DatasetRouteShuffled = fail
EventRouteShuffled = fail
InvertedRoleMask = fail
```

System：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p6_official_paired_replay_pareto.svg
p6_macro_beat_rate.svg
p6_signal_stratum_win_matrix.svg
p6_shuffle_control_matrix.svg
p6_system_gate_distribution.svg
```

---

## P7：Short-run scout

### 目标

如果 P6 pass，验证局部 paired replay 优势能否在连续训练中保持。

### 设置

```text
steps = 50,240,640
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRiskScore, ShuffledBranchRatio
```

### 必须记录

```text
candidate
dataset
seed
steps
train_loss
holdout_loss
val_acc_proxy
CEp99
margin_p10
ECE_proxy
NLL_proxy
curvature
local_lipschitz
basis_usage_entropy
functional_channel_usage_entropy
event_count
coverage
bad_event_rate
step_ratio_q90
memory_ratio
base_checkpoint_hash
```

### 判断标准

Task safety：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005.
$$

Control superiority：

$$
MetricGain_{\text{functional}}>MetricGain_{\text{AdamWParallel}},
$$

$$
MetricGain_{\text{functional}}>MetricGain_{\text{bestLR}}.
$$

Mechanism pass, at least one：

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamW}},
$$

or:

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamW}},
$$

or:

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

### 可视化

```text
p7_short_run_task_mechanism_pareto.svg
p7_short_run_controls.svg
p7_event_timeline.svg
p7_ce_tail_margin_panel.svg
```

---

## P8：Full 10-seed validation

### 目标

如果 short-run pass，验证 full functional route。

### 设置

```text
epochs = 20
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
baseline = MLP-match
diagnostic baseline = QuadraticFeatureMLP
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRiskScore
```

### 必须记录

```text
candidate
dataset
seed
val_acc
test_acc
delta_vs_mlp
delta_vs_adamw
delta_vs_adamwparallel
delta_vs_bestlr
delta_vs_quadratic_feature_mlp
near_pass
full_pass
CEp99
margin_p10
wrong_confidence_p95
ECE
NLL
curvature_ratio
local_lipschitz_ratio
basis_usage_entropy
functional_channel_usage_entropy
event_count
coverage
bad_event_rate
step_ratio_q90
memory_ratio
base_checkpoint_hash
```

### 判断标准

Task safety：

$$
\Delta Acc_{\text{functional-vs-AdamW}}\geq-0.005.
$$

Macro improvement：

$$
\Delta Acc_{\text{macro,functional}}
-
\Delta Acc_{\text{macro,AdamW}}
\geq0.003.
$$

Hard-stratum repair：

$$
\Delta Acc_{\text{hard-stratum,functional}}
-
\Delta Acc_{\text{hard-stratum,AdamW}}
\geq0.005.
$$

Control superiority：

$$
MetricGain_{\text{functional}}>MetricGain_{\text{AdamWParallel}},
$$

$$
MetricGain_{\text{functional}}>MetricGain_{\text{bestLR}}.
$$

---

## P9：Robustness / strong baseline / external-ready

### 目标

确认 functional advantage 不是 clean MNIST-family artifact。

### 设置

```text
label_noise = 0.05,0.10,0.20
input_noise = 0.05,0.10
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
strong baselines = MLP-match, hidden-bracket MLP, QuadraticFeatureMLP, repaired LQ AdamW-only
```

### 判断标准

Robustness pass：

$$
AccDrop_{\text{functional}}\leq AccDrop_{\text{AdamW}}.
$$

Noise-tail pass：

$$
CEp99_{\text{functional}}\leq CEp99_{\text{AdamW}}.
$$

Strong baseline pass：

$$
Acc_{\text{functional}}\geq Acc_{\text{QuadraticFeatureMLP}}-0.005
$$

or:

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{QuadraticFeatureMLP}}.
$$

---

# Part VIII. Required artifacts

```text
run_manifest.json
contract_audit_v9246.csv
p0_v9245_boundary_reproduction.csv
p1_boundary_disagreement_expansion.csv
p2_target_regrounding.csv
p3_legal_feature_factory_v2.csv
p4_factorized_controller_calibration.csv
p5_leave_dataset_and_stratum_out.csv
p6_official_paired_replay.csv
p7_short_run_functional_validation.csv
p8_full_10seed_functional_validation.csv
p9_robustness_external_ready.csv
disagreement_trace_v9246.csv
safe_good_label_trace_v9246.csv
legal_feature_trace_v9246.csv
train_stream_microprobe_trace_v9246.csv
controller_calibration_trace_v9246.csv
leaveout_trace_v9246.csv
paired_replay_branch_trace_v9246.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9245_boundary_unstable
F3_dataset_tuning_detected
F4_disagreement_rows_insufficient
F5_gap_attribution_fail
F6_target_grounding_unreliable
F7_legal_feature_system_too_expensive
F8_all_legal_features_fail_safe_good
F9_microprobe_overhead_fail
F10_factorized_controller_fail
F11_oracle_support_collapse
F12_leave_dataset_out_fail
F13_leave_stratum_out_fail
F14_paired_replay_control_equivalent
F15_shuffle_control_pass
F16_functional_lr_equivalent
F17_short_run_task_drop
F18_full_run_no_macro_hard_stratum_gain
F19_strong_baseline_explains_gain
F20_robustness_fail
F21_external_not_ready
F22_fake_or_proxy_violation
F23_artifact_missing
```

---

# Part IX. Route decision

```text
R1-BoundaryGapAttributed:
  C1/F1 false-positive and false-negative gaps are attributed.

R2-TargetRegroundingPass:
  risk/value/control/safe-good labels are reliable.

R3-LegalFeatureSafeGoodPass:
  at least one legal feature predicts safe-good.

R4-MicroProbeFeaturePass:
  train-stream micro-probe passes predictivity and overhead gates.

R5-FactorizedControllerPass:
  RiskSafe + ValuePositive + ControlResistant controller passes heldout gate.

R6-LeaveDatasetOutPass:
  controller generalizes across held-out datasets.

R7-LeaveStratumOutPass:
  controller generalizes across held-out signal strata.

R8-PairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R9-OracleSupportCollapse:
  F1/support good events collapse on boundary-targeted rows.

R10-OracleHighLegalFeatureGap:
  oracle support exists but legal features still cannot identify it.

R11-FeatureSystemTooExpensive:
  legal features work but overhead exceeds official gate.

R12-BaseAttachCarrierValidButControllerInvalid:
  base/attach/carrier remain valid, but controller cannot pass.

R13-StrictPureKANFunctionalShortRunPass:
  short-run task-safe mechanism gain.

R14-StrictPureKANFunctionalFullPass:
  full 10-seed macro / hard-stratum / geometry gain.

R15-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` 必须记录：

```text
route
v9245_boundary_pass
dataset_tuning_detected
f1_support_stability_pass
f1_oracle_precision
f1_oracle_coverage
f1_oracle_bad_event
f1_c1_jaccard
fp_count
fn_count
fp_attribution_pass
fn_attribution_pass
target_grounding_pass
best_risk_feature
best_value_feature
best_gap_feature
best_microprobe_feature
legal_feature_predictivity_pass
legal_feature_overhead
best_controller_id
factorized_controller_pass
controller_auc
controller_corr
accepted_precision
accepted_coverage
accepted_bad_event_rate
accepted_strata_count
accepted_family_count
leave_dataset_out_pass
leave_stratum_out_pass
paired_replay_pass
short_run_pass
full_run_pass
external_ready
primary_blocker
next_required_implementation
success_v9246_strict_purekan_functional
success_v9246_full_functional
success_v9246_external_ready
```

---

# Part X. 并行执行顺序

```text
Batch 1:
  P0 boundary reproduction
  P1 boundary disagreement expansion
  P2 target regrounding
  P3 legal feature factory v2

Batch 2:
  P4 factorized controller calibration
  P5 leave-dataset-out / leave-stratum-out
  P6 paired replay scout

Batch 3:
  official P6 paired replay
  P7 short-run if paired replay passes

Batch 4:
  P8 full 10-seed
  P9 robustness / strong baseline
```

Gate rule：

```text
P4/P6 diagnostic rows may be measured before all gates finish.
official_eligible = 1 only if:
  base robust pass
  attach equivalence pass
  no-event preservation pass
  carrier active
  oracle support pass
  legal controller pass
  LDO/LSO pass
```

---

# Part XI. 停止条件

## Minimum diagnostic success

```text
v9.2.45 boundary reproduced
boundary disagreement rows sufficient
F1/C1 gap attributed
safe-good target regrounded
legal feature factory v2 completed
no fake/proxy/offload/loss/teacher violation
```

## Legal controller success

```text
Minimum diagnostic success
+
at least one legal feature predicts safe-good
+
factorized controller passes heldout precision / coverage / bad-event gate
+
system overhead gate pass
```

## Local functional success

```text
Legal controller success
+
LDO / LSO pass
+
official paired replay beats AdamWParallel / bestLR
```

## Full functional success

```text
Local functional success
+
short/full run task-safe mechanism gain
+
macro or hard-stratum improvement
```

## Failure stop

```text
1. v9.2.45 boundary cannot be reproduced；
2. disagreement expansion cannot produce enough FP/FN rows；
3. F1/C1 gap cannot be attributed；
4. safe-good target grounding is unreliable；
5. F1 support collapses on boundary-targeted rows；
6. all legal features fail safe-good predictivity；
7. only predictive feature is too expensive；
8. micro-probe overhead exceeds gate；
9. factorized controller fails heldout gate；
10. leave-dataset-out fails；
11. leave-stratum-out fails；
12. paired replay remains control-equivalent；
13. shuffle controls pass, indicating overfit；
14. short-run task drops；
15. full run gives no macro / hard-stratum / geometry gain；
16. functional breaks system gate；
17. gains are explained by QuadraticFeatureMLP；
18. any teacher/loss/fake/proxy/offload violation occurs。
```

---

# Part XII. 最终解释规则

## Case A：boundary attribution closes and factorized controller passes

可以声明：

```text
v9.2.45 failed because legal controller mixed risk/value/control into one score; v9.2.46 separated them and recovered a legal controller.
```

但不能声明 strict success unless LDO/LSO and paired replay pass.

## Case B：safe-good legal feature passes but controller fails

必须声明：

```text
safe-good is observable, but accept/abstain calibration is still not deployable.
```

下一步修 controller calibration，不修 base/attach.

## Case C：micro-probe passes but is too expensive

必须声明：

```text
value is observable but not system-legal.
```

下一步做 micro-probe amortization / kernelization.

## Case D：F1 support collapses

必须声明：

```text
v9.2.44/v9.2.45 support recovery was not stable on boundary-targeted rows.
```

下一步回到 carrier/support reset。

## Case E：legal features all fail but oracle support remains high

必须声明：

```text
safe-good events exist, but current legal feature representation is insufficient.
```

下一步设计 richer train-stream sufficient statistics.

## Case F：LDO/LSO fail

必须声明：

```text
controller is not dataset-agnostic or stratum-agnostic enough.
```

不能用 dataset-specific tuning 写成功。

## Case G：paired replay passes

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full validation。

---

# Part XIII. 最终建议

v9.2.46 的一句话策略是：

$$
\boxed{
\text{不要再只模仿 F1 或调 C1；直接分解 safe-good 为 risk/value/control，并寻找 legal sufficient statistics。}
}
$$

当前最关键的问题不是：

```text
base 是否稳定；
attach 是否污染；
carrier 是否 silent；
F1 是否完全空；
C1 threshold 是否差一点；
Fashion/KMNIST/MNIST 谁更好；
是否换一个普通 basis。
```

而是：

```text
1. C1/F1 false-positive 为什么不能充分归因？
2. F1 heldout bad-event 为什么超过 gate？
3. 哪些 legal commit-time features 能分别预测 risk-safe、value-positive、control-resistant？
4. train-stream micro-probe 是否能提供缺失的 population-risk signal？
5. factorized controller 能否同时满足 precision / coverage / bad-event？
6. 这个 controller 能否 LDO/LSO？
7. official paired replay 能否打过 AdamWParallel / bestLR？
```

v9.2.46 的结果将给出清晰分叉：

```text
if factorized controller + LDO/LSO + paired replay pass:
  strict PureKAN functional obtains local causal evidence.

if legal feature works but system cost too high:
  focus on feature amortization / kernelization.

if oracle support remains high but legal features fail:
  legal sufficient statistics are still missing.

if oracle support collapses:
  carrier/support stability remains the blocker.
```
